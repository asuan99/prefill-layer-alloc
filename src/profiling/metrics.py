"""
Latency and bandwidth measurement utilities.

Provides CUDA-event based timing (eliminates CPU-GPU transfer overhead)
and bandwidth estimation from tensor sizes and measured latency.

Bandwidth estimation notes:
  achieved_bandwidth_GBs = total_bytes_transferred / latency_s
  This is a lower-bound estimate: only explicit tensor reads/writes are counted;
  intermediate scratchpad traffic (e.g. softmax tiles in flash-attn, SSM dt/B/C)
  is excluded, so actual hardware BW may be higher.

  theoretical_bw_GBs is derived from torch device properties:
    BW = 2 × memory_clock_rate_KHz × 1e3 × memory_bus_width_bits / 8 / 1e9
  This matches the spec-sheet peak BW for DDR-type memories (HBM, GDDR).
"""

import numpy as np
import torch
from typing import Callable, Optional


class LatencyMeter:
    """CUDA-event based kernel latency measurement.

    Preferred over CPU timers because CUDA events are timestamped on-device,
    avoiding CPU-side overhead and synchronization jitter.

    Usage:
        meter = LatencyMeter()
        result = meter.measure(fn, n_warmup=10, n_measure=50)
        print(result['latency_ms'], result['latency_p99_ms'])
    """

    def __init__(self, device: str = "cuda"):
        self.device = device

    def measure(
        self,
        fn: Callable,
        n_warmup: int = 10,
        n_measure: int = 50,
        args: tuple = (),
        kwargs: Optional[dict] = None,
    ) -> dict:
        """Measure function latency using CUDA events.

        Args:
            fn: Callable to measure (should run on GPU).
            n_warmup: Discarded warm-up iterations.
            n_measure: Measurement iterations.
            args: Positional args for fn.
            kwargs: Keyword args for fn.

        Returns:
            dict with latency statistics in milliseconds.
        """
        if kwargs is None:
            kwargs = {}

        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_measure)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_measure)]

        for _ in range(n_warmup):
            fn(*args, **kwargs)
        torch.cuda.synchronize()

        for i in range(n_measure):
            start_events[i].record()
            fn(*args, **kwargs)
            end_events[i].record()

        torch.cuda.synchronize()

        latencies_ms = [
            start_events[i].elapsed_time(end_events[i])
            for i in range(n_measure)
        ]
        arr = np.array(latencies_ms)

        return {
            "latency_ms": float(np.median(arr)),
            "latency_mean_ms": float(arr.mean()),
            "latency_p99_ms": float(np.percentile(arr, 99)),
            "latency_min_ms": float(arr.min()),
            "latency_max_ms": float(arr.max()),
            "latency_std_ms": float(arr.std()),
            "n_warmup": n_warmup,
            "n_measure": n_measure,
        }

    def measure_cpu(
        self,
        fn: Callable,
        n_warmup: int = 10,
        n_measure: int = 50,
        args: tuple = (),
        kwargs: Optional[dict] = None,
        sync_before: bool = True,
        sync_after: bool = True,
    ) -> dict:
        """CPU-timer measurement with configurable synchronization.

        Useful for measuring operations that include CPU-side overhead
        (e.g., SM reconfiguration calls via Green Context stream switch).
        """
        import time

        if kwargs is None:
            kwargs = {}

        for _ in range(n_warmup):
            fn(*args, **kwargs)
        torch.cuda.synchronize()

        samples_us = []
        for _ in range(n_measure):
            if sync_before:
                torch.cuda.synchronize()
            t0 = time.perf_counter_ns()
            fn(*args, **kwargs)
            if sync_after:
                torch.cuda.synchronize()
            t1 = time.perf_counter_ns()
            samples_us.append((t1 - t0) / 1_000.0)

        arr = np.array(samples_us)
        return {
            "latency_us": float(np.median(arr)),
            "latency_mean_us": float(arr.mean()),
            "latency_p99_us": float(np.percentile(arr, 99)),
            "latency_min_us": float(arr.min()),
            "latency_max_us": float(arr.max()),
            "latency_std_us": float(arr.std()),
        }


class BandwidthEstimator:
    """Estimates achieved and theoretical memory bandwidth.

    Theoretical BW formula (DDR, HBM):
      BW_GBs = 2 × mem_clock_KHz × 1e3 × bus_width_bits / 8 / 1e9

    where:
      mem_clock_KHz  = torch.cuda.get_device_properties().memory_clock_rate
      bus_width_bits = torch.cuda.get_device_properties().memory_bus_width

    This correctly handles HBM (A100: 1215 KHz × 5120 bits → 1555 GB/s),
    GDDR6 (RTX 4090: 10501 KHz × 384 bits → 1008 GB/s), etc.
    """

    def __init__(self, device_id: int = 0, theoretical_bw_GBs: Optional[float] = None):
        self.device_id = device_id
        if theoretical_bw_GBs is not None:
            self._theoretical_bw_GBs = theoretical_bw_GBs
        else:
            self._theoretical_bw_GBs = self._query_theoretical_bw()

    def _query_theoretical_bw(self) -> float:
        """Query peak memory bandwidth from torch device properties.

        Uses the correct formula:
          BW = 2 × memory_clock_rate(KHz) × 1e3 Hz/KHz × memory_bus_width(bits)
               / 8 (bits/byte) / 1e9 (bytes/GB)
        """
        try:
            props = torch.cuda.get_device_properties(self.device_id)
            # memory_clock_rate: effective clock in KHz (already accounts for DDR)
            # For HBM and GDDR, torch reports the per-pin rate; ×2 for DDR.
            mem_clock_hz = props.memory_clock_rate * 1e3      # KHz → Hz
            bus_width_bits = props.memory_bus_width            # bits

            # Factor 2 for Double Data Rate (DDR / HBM2 both transfer on both edges)
            bw_GBs = (2.0 * mem_clock_hz * bus_width_bits) / (8.0 * 1e9)
            return bw_GBs
        except Exception:
            return 1000.0  # safe fallback: 1 TB/s

    @property
    def theoretical_bw_GBs(self) -> float:
        return self._theoretical_bw_GBs

    def set_theoretical_bw(self, bw_GBs: float) -> None:
        """Override theoretical BW from hardware.yaml (recommended)."""
        self._theoretical_bw_GBs = bw_GBs

    def estimate(
        self,
        read_bytes: int,
        write_bytes: int,
        latency_ms: float,
    ) -> dict:
        """Estimate achieved bandwidth from transfer size and measured latency.

        achieved_GBs = (read_bytes + write_bytes) / latency_s

        Note: read_bytes + write_bytes should include:
          - activation tensor reads and writes
          - weight parameter reads (loaded from HBM once per forward)
        but NOT L2-cached intermediates (flash-attn tiles, etc.).

        Args:
            read_bytes: Total bytes read from HBM.
            write_bytes: Total bytes written to HBM.
            latency_ms: Kernel latency in milliseconds (use median, not mean).

        Returns:
            dict with achieved_GBs, theoretical_bw_GBs, bw_utilization_pct.
        """
        total_bytes = read_bytes + write_bytes
        achieved_GBs = (total_bytes / 1e9) / (latency_ms / 1000.0)
        utilization_pct = (
            achieved_GBs / self._theoretical_bw_GBs * 100.0
            if self._theoretical_bw_GBs > 0 else float("nan")
        )
        return {
            "achieved_bandwidth_GBs": achieved_GBs,
            "theoretical_bw_GBs": self._theoretical_bw_GBs,
            "bw_utilization_pct": utilization_pct,
            "total_bytes": total_bytes,
        }

    # ------------------------------------------------------------------
    # Layer-specific byte count helpers
    # ------------------------------------------------------------------

    @staticmethod
    def ssm_bytes(
        batch: int,
        seq_len: int,
        hidden_size: int,
        n_heads: int,
        head_dim: int,
        d_state: int,
        weight_bytes: int,
        bytes_per_elem: int = 2,
    ) -> tuple[int, int]:
        """Estimate HBM read/write bytes for an SSM (Mamba-2) prefill layer.

        Counted transfers:
          Read : in_proj(hidden→2*inner) weights + input activation
                 + SSM params (A_log, D, dt_bias)
                 + per-chunk B, C, dt projections
          Write: output activation

        Intermediate within-chunk state (SRAM resident in Triton kernel) excluded.
        """
        inner_dim = n_heads * head_dim
        bpe = bytes_per_elem

        # Activation: read input, write output
        act_read = batch * seq_len * hidden_size * bpe
        act_write = batch * seq_len * hidden_size * bpe

        # Weights read once (in_proj, out_proj, SSM params)
        # weight_bytes passed in from layer.parameters()

        # dt/B/C are computed on-the-fly from in_proj; already counted via weight_bytes
        total_read = act_read + weight_bytes
        total_write = act_write
        return total_read, total_write

    @staticmethod
    def attn_bytes(
        batch: int,
        seq_len: int,
        total_kv_len: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        hidden_size: int = 0,
        proj_weight_bytes: int = 0,
        bytes_per_elem: int = 2,
    ) -> tuple[int, int]:
        """Estimate HBM read/write bytes for a full prefill attention layer.

        Includes Q/K/V/O projection GEMMs when hidden_size and proj_weight_bytes
        are provided (recommended — omitting them undercounts by 5–10×).

        HBM transfers counted (non-fused kernel sequence):
          Read : input hidden_states + W_q/W_k/W_v/W_o weights
                 + projected Q/K/V tensors (read back by FlashAttn)
                 + attn output (read back by O proj)
          Write: projected Q/K/V tensors + attn output + final output

        FlashAttention softmax intermediates remain in SRAM — not counted.
        """
        bpe = bytes_per_elem

        # Projection weight load (once per forward)
        weight_read = proj_weight_bytes

        # Input hidden_states (read once for all three projections)
        act_in  = batch * seq_len * hidden_size * bpe if hidden_size > 0 else 0

        # Projected tensors: written after GEMM, read back by FlashAttn
        q_bytes = batch * seq_len * n_heads * head_dim * bpe
        k_new   = batch * seq_len * n_kv_heads * head_dim * bpe
        v_new   = batch * seq_len * n_kv_heads * head_dim * bpe

        # Context KV cache (pre-built, read by FlashAttn if context_len > 0)
        ctx_kv_len = total_kv_len - seq_len
        kv_ctx = batch * ctx_kv_len * n_kv_heads * head_dim * 2 * bpe if ctx_kv_len > 0 else 0

        # Attention output (written by FlashAttn, read back by O proj)
        attn_out = q_bytes  # same shape as Q

        # Final output (written after O proj)
        act_out = act_in  # same shape as input

        total_read  = weight_read + act_in + q_bytes + k_new + v_new + kv_ctx + attn_out
        total_write = q_bytes + k_new + v_new + attn_out + act_out
        return total_read, total_write

    @staticmethod
    def mlp_bytes(
        batch: int,
        seq_len: int,
        hidden_size: int,
        intermediate_size: int,
        weight_bytes: int,
        bytes_per_elem: int = 2,
    ) -> tuple[int, int]:
        """Estimate HBM read/write bytes for a gated MLP layer."""
        bpe = bytes_per_elem
        act_read = batch * seq_len * hidden_size * bpe
        act_write = batch * seq_len * hidden_size * bpe
        total_read = act_read + weight_bytes
        total_write = act_write
        return total_read, total_write
