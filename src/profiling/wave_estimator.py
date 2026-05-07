"""
Analytical wave quantization estimator.

Wave quantization: when n_thread_blocks is not a multiple of SM_count,
the last "wave" of execution has fewer than SM_count active SMs.
The idle SMs in the last wave are wasted.

  n_waves            = ceil(n_blocks / sm_count)
  last_wave_blocks   = n_blocks % sm_count  (0 means perfect fit)
  last_wave_sm_util  = last_wave_blocks / sm_count  (if last_wave_blocks > 0)
  wave_efficiency    = n_blocks / (n_waves * sm_count)
                     = 1 - idle_sm_fraction_in_last_wave / n_waves

This module computes these analytically from known kernel launch configurations
for Mamba-2 (SSM) and FlashAttention-style kernels, WITHOUT requiring ncu.

Typical launch configurations:
  Mamba-2 (mamba_chunk_scan_combined):
    grid = (batch × n_heads, ceil(seq_len / chunk_size))
    block = (chunk_size,)  [varies by kernel variant]
    → n_blocks = batch × n_heads × ceil(seq_len / chunk_size)

  FlashAttention-2 prefill:
    grid = (ceil(seq_len / BLOCK_M), n_heads, batch)
    block = (BLOCK_M × BLOCK_N / BLOCK_N, ...)  [internal]
    → n_blocks = ceil(seq_len / BLOCK_M) × n_heads × batch
    BLOCK_M ≈ 64 or 128 depending on head_dim

  MLP (gemm-based, cuBLAS/cutlass):
    grid is set by cuBLAS internally, depends on matrix size and tile config.
    We use a tile-size approximation: n_blocks ≈ ceil(M/128) × ceil(N/128)
    where M = batch×seq_len, N = intermediate_size or hidden_size.

Note: these are approximations. Actual block counts depend on kernel variant,
occupancy limits, and shared memory. Use `run_ncu_profile.py` to get exact
values from hardware counters.
"""

import math
from dataclasses import dataclass


@dataclass
class WaveStats:
    """Wave quantization statistics for a single kernel launch."""
    n_blocks: int
    sm_count: int
    n_waves: int
    last_wave_blocks: int          # blocks in the last (potentially partial) wave
    wave_efficiency: float         # n_blocks / (n_waves * sm_count), ∈ (0, 1]
    last_wave_sm_util: float       # fraction of SMs active in last wave, ∈ (0, 1]
    wasted_sm_fraction: float      # 1 - wave_efficiency  (idle SM-wave fraction)

    def is_perfect_fit(self) -> bool:
        return self.last_wave_blocks == 0

    def summary(self) -> str:
        return (
            f"n_blocks={self.n_blocks}  sm_count={self.sm_count}  "
            f"n_waves={self.n_waves}  last_wave={self.last_wave_blocks}SM  "
            f"efficiency={self.wave_efficiency:.1%}  "
            f"wasted={self.wasted_sm_fraction:.1%}"
        )


def compute_wave_stats(n_blocks: int, sm_count: int) -> WaveStats:
    """Compute wave quantization statistics from block count and SM count.

    Args:
        n_blocks: Total number of thread blocks (CTA) launched.
        sm_count: Number of SMs available (may be restricted via Green Contexts).

    Returns:
        WaveStats with efficiency and waste metrics.
    """
    sm_count = max(1, sm_count)
    n_blocks = max(1, n_blocks)
    n_waves = math.ceil(n_blocks / sm_count)
    last_wave_blocks = n_blocks % sm_count
    # last_wave_blocks == 0 means the last wave is full
    effective_last = last_wave_blocks if last_wave_blocks > 0 else sm_count
    wave_efficiency = n_blocks / (n_waves * sm_count)
    last_wave_sm_util = effective_last / sm_count

    return WaveStats(
        n_blocks=n_blocks,
        sm_count=sm_count,
        n_waves=n_waves,
        last_wave_blocks=last_wave_blocks,
        wave_efficiency=wave_efficiency,
        last_wave_sm_util=last_wave_sm_util,
        wasted_sm_fraction=1.0 - wave_efficiency,
    )


class WaveEstimator:
    """Estimates wave quantization for standard GPU kernels.

    All methods return WaveStats for the dominant kernel in each operation.
    For composite operations (e.g., SSM = in_proj + scan + out_proj),
    the scan kernel dominates latency and is used for estimation.
    """

    # ----------------------------------------------------------------
    # SSM (Mamba-2 parallel scan)
    # ----------------------------------------------------------------

    @staticmethod
    def ssm_prefill(
        batch: int,
        seq_len: int,
        n_heads: int,
        chunk_size: int = 256,
        sm_count: int = 108,
    ) -> WaveStats:
        """Wave stats for mamba_chunk_scan_combined (SSM prefill dominant kernel).

        Mamba-2 parallel scan grid (from mamba_ssm Triton kernel):
          grid = (batch × n_heads, ceil(seq_len / chunk_size))
          Each CTA processes one chunk of one (batch, head) pair.

        Args:
            batch: Batch size.
            seq_len: Sequence length.
            n_heads: Number of SSM heads.
            chunk_size: Mamba-2 chunk size (default 256).
            sm_count: Effective SM count (after Green Context SM restriction).

        Returns:
            WaveStats for the scan kernel.
        """
        n_chunks = math.ceil(seq_len / chunk_size)
        n_blocks = batch * n_heads * n_chunks
        return compute_wave_stats(n_blocks, sm_count)

    @staticmethod
    def ssm_in_proj(
        batch: int,
        seq_len: int,
        hidden_size: int,
        inner_dim: int,
        sm_count: int = 108,
        tile_m: int = 128,
        tile_n: int = 128,
    ) -> WaveStats:
        """Wave stats for the in_proj GEMM kernel before the SSM scan.

        in_proj: (batch×seq_len, hidden_size) @ (hidden_size, 2×inner_dim)
        cuBLAS uses ~128×128 output tiles.

        Args:
            tile_m, tile_n: cuBLAS output tile dimensions (approximate).
        """
        m = batch * seq_len
        n = 2 * inner_dim
        n_blocks = math.ceil(m / tile_m) * math.ceil(n / tile_n)
        return compute_wave_stats(n_blocks, sm_count)

    # ----------------------------------------------------------------
    # Attention (FlashAttention-2)
    # ----------------------------------------------------------------

    @staticmethod
    def attn_prefill(
        batch: int,
        seq_len: int,
        n_heads: int,
        head_dim: int,
        sm_count: int = 108,
    ) -> WaveStats:
        """Wave stats for FlashAttention-2 prefill.

        FA-2 forward kernel grid:
          grid = (ceil(seq_len / BLOCK_M), n_heads, batch)

        BLOCK_M (query block size) depends on head_dim:
          head_dim ≤  64 → BLOCK_M = 128
          head_dim ≤ 128 → BLOCK_M =  64
          head_dim ≤ 256 → BLOCK_M =  64
        """
        if head_dim <= 64:
            block_m = 128
        else:
            block_m = 64

        n_q_blocks = math.ceil(seq_len / block_m)
        n_blocks = n_q_blocks * n_heads * batch
        return compute_wave_stats(n_blocks, sm_count)

    @staticmethod
    def attn_with_context(
        batch: int,
        seq_len: int,
        context_len: int,
        n_heads: int,
        head_dim: int,
        sm_count: int = 108,
    ) -> WaveStats:
        """Wave stats for attention with pre-filled KV cache.

        Same grid as attn_prefill — context_len doesn't change the Q-block grid.
        (Context affects K/V access pattern but not CTA count.)
        """
        return WaveEstimator.attn_prefill(
            batch, seq_len, n_heads, head_dim, sm_count
        )

    # ----------------------------------------------------------------
    # MLP
    # ----------------------------------------------------------------

    @staticmethod
    def mlp_gemm(
        batch: int,
        seq_len: int,
        hidden_size: int,
        intermediate_size: int,
        sm_count: int = 108,
        tile_m: int = 128,
        tile_n: int = 128,
    ) -> WaveStats:
        """Wave stats for MLP gate/up projection GEMM.

        Dominant GEMM: (batch×seq_len, hidden) @ (hidden, intermediate)
        """
        m = batch * seq_len
        n = intermediate_size
        n_blocks = math.ceil(m / tile_m) * math.ceil(n / tile_n)
        return compute_wave_stats(n_blocks, sm_count)

    # ----------------------------------------------------------------
    # Sweep helper
    # ----------------------------------------------------------------

    @staticmethod
    def sweep(
        layer_type: str,
        sm_counts: list[int],
        seq_lens: list[int],
        batch_sizes: list[int],
        model_cfg: dict,
    ) -> list[dict]:
        """Compute wave stats for all (sm_count, seq_len, batch_size) combos.

        Args:
            layer_type: 'ssm' | 'attn' | 'mlp'
            sm_counts: List of SM counts to evaluate.
            seq_lens: List of sequence lengths.
            batch_sizes: List of batch sizes.
            model_cfg: Dict from models.yaml (must have n_ssm_heads, chunk_size, etc.)

        Returns:
            List of dicts with WaveStats fields + input config.
        """
        results = []
        for sm_count in sm_counts:
            for seq_len in seq_lens:
                for batch_size in batch_sizes:
                    if layer_type == "ssm":
                        stats = WaveEstimator.ssm_prefill(
                            batch=batch_size,
                            seq_len=seq_len,
                            n_heads=model_cfg.get("n_ssm_heads", 64),
                            chunk_size=model_cfg.get("chunk_size", 256),
                            sm_count=sm_count,
                        )
                    elif layer_type == "attn":
                        stats = WaveEstimator.attn_prefill(
                            batch=batch_size,
                            seq_len=seq_len,
                            n_heads=model_cfg.get("n_attn_heads", 8),
                            head_dim=model_cfg.get("attn_head_dim", 256),
                            sm_count=sm_count,
                        )
                    elif layer_type == "mlp":
                        stats = WaveEstimator.mlp_gemm(
                            batch=batch_size,
                            seq_len=seq_len,
                            hidden_size=model_cfg.get("hidden_size", 2048),
                            intermediate_size=model_cfg.get("intermediate_size", 4096),
                            sm_count=sm_count,
                        )
                    else:
                        raise ValueError(f"Unknown layer_type: {layer_type!r}")

                    results.append({
                        "layer_type": layer_type,
                        "sm_count": sm_count,
                        "seq_len": seq_len,
                        "batch_size": batch_size,
                        "n_blocks": stats.n_blocks,
                        "n_waves": stats.n_waves,
                        "last_wave_blocks": stats.last_wave_blocks,
                        "wave_efficiency": stats.wave_efficiency,
                        "last_wave_sm_util": stats.last_wave_sm_util,
                        "wasted_sm_fraction": stats.wasted_sm_fraction,
                    })
        return results
