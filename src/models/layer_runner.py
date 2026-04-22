"""
Single-layer independent execution utility.

LayerRunner runs individual SSM, Attention, or MLP layers in isolation,
applying libsmctrl SM restrictions before each benchmark run to measure
SM-scaling behavior.

Design notes:
  - SSM prefill uses mamba_chunk_scan_combined in parallel (not recurrent) mode;
    no per-token recurrent state is needed for prefill.
  - Attention prefill uses FlashInfer BatchPrefillWithPagedKVCacheWrapper with a
    pre-filled KV cache to simulate realistic context lengths.
  - SM restriction is applied via SMController.set_sm_count() before measurement
    and reset() after. If libsmctrl is unavailable, SMController falls back to
    CUDA MPS thread percentage.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch
import torch.nn as nn
import numpy as np
from typing import Optional

from src.smctrl.libsmctrl_wrapper import SMController
from src.profiling.metrics import LatencyMeter, BandwidthEstimator


class LayerRunner:
    """Runs individual hybrid model layers independently for latency benchmarking.

    Supports Zamba2 and Falcon-H1 SSM, Attention, and MLP layers.
    SM count is controlled via libsmctrl (or MPS fallback) before each run.

    Args:
        device: CUDA device string (e.g. 'cuda' or 'cuda:0').
        dtype: Computation dtype. Default: bfloat16.
        total_sm_count: Total SM count on device. Auto-detected if None.
    """

    def __init__(
        self,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        total_sm_count: Optional[int] = None,
    ):
        self.device = device
        self.dtype = dtype
        self.smctrl = SMController(total_sm_count=total_sm_count)
        self.meter = LatencyMeter(device=device)
        self.bw_estimator = BandwidthEstimator()

        # Lazy-loaded layer extractors
        self._extractors: dict = {}

    # ------------------------------------------------------------------
    # Layer extractor cache
    # ------------------------------------------------------------------

    def _get_extractor(self, model_name: str):
        if model_name not in self._extractors:
            if model_name == "zamba2":
                from src.models.zamba2 import Zamba2LayerExtractor
                self._extractors[model_name] = Zamba2LayerExtractor(
                    device=self.device, dtype=self.dtype
                )
            elif model_name == "falcon_h1":
                from src.models.falcon_h1 import FalconH1LayerExtractor
                self._extractors[model_name] = FalconH1LayerExtractor(
                    device=self.device, dtype=self.dtype
                )
            else:
                raise ValueError(f"Unknown model: {model_name!r}. Choose 'zamba2' or 'falcon_h1'.")
        return self._extractors[model_name]

    # ------------------------------------------------------------------
    # SSM layer benchmark
    # ------------------------------------------------------------------

    def run_ssm_layer(
        self,
        model_name: str,
        batch_size: int,
        seq_len: int,
        sm_count: int,
        n_warmup: int = 10,
        n_measure: int = 50,
        use_fallback_kernel: bool = False,
    ) -> dict:
        """Benchmark a single SSM (Mamba-2) prefill layer with sm_count SMs.

        SSM layer prefill uses mamba_chunk_scan_combined in parallel mode.
        No recurrent state buffer is needed; the parallel scan processes the
        full sequence in one shot.

        Args:
            model_name: 'zamba2' | 'falcon_h1'
            batch_size: Batch size.
            seq_len: Sequence length (prefill tokens).
            sm_count: Number of SMs to restrict to via libsmctrl.
            n_warmup: Warm-up iterations (discarded).
            n_measure: Measurement iterations.
            use_fallback_kernel: If True, use FallbackSSMKernel instead of
                                 loading the full model.

        Returns:
            dict with keys: latency_ms, latency_p99_ms, achieved_bandwidth_GBs,
                            sm_count, seq_len, layer_type, batch_size, model_name
        """
        extractor = self._get_extractor(model_name)

        if use_fallback_kernel:
            layer = self._build_fallback_ssm(model_name)
        else:
            try:
                layer = extractor.get_ssm_layer()
            except Exception:
                layer = self._build_fallback_ssm(model_name)

        inputs = extractor.make_ssm_inputs(batch_size, seq_len)
        hidden_states = inputs["hidden_states"]

        # Estimate memory traffic: read/write input+output + weight parameters
        cfg = extractor.get_model_config()
        hidden_size = cfg["hidden_size"]
        bytes_per_elem = 2  # bfloat16
        io_bytes = 2 * batch_size * seq_len * hidden_size * bytes_per_elem  # read + write
        weight_bytes = sum(p.numel() * bytes_per_elem for p in layer.parameters())
        total_bytes = io_bytes + weight_bytes

        def _run():
            with torch.no_grad():
                layer(hidden_states)

        # Apply SM restriction
        self.smctrl.set_sm_count(sm_count)
        try:
            result = self.meter.measure(_run, n_warmup=n_warmup, n_measure=n_measure)
        finally:
            self.smctrl.reset()

        bw = self.bw_estimator.estimate(
            read_bytes=total_bytes // 2,
            write_bytes=total_bytes // 2,
            latency_ms=result["latency_ms"],
        )

        return {
            "latency_ms": result["latency_ms"],
            "latency_p99_ms": result["latency_p99_ms"],
            "achieved_bandwidth_GBs": bw["achieved_bandwidth_GBs"],
            "theoretical_bw_GBs": bw["theoretical_bw_GBs"],
            "bw_utilization_pct": bw["bw_utilization_pct"],
            "sm_count": sm_count,
            "sm_ratio": sm_count / self.smctrl.total_sm_count,
            "seq_len": seq_len,
            "batch_size": batch_size,
            "layer_type": "ssm",
            "model_name": model_name,
        }

    # ------------------------------------------------------------------
    # Attention layer benchmark
    # ------------------------------------------------------------------

    def run_attn_layer(
        self,
        model_name: str,
        batch_size: int,
        seq_len: int,
        sm_count: int,
        context_len: int = 0,
        n_warmup: int = 10,
        n_measure: int = 50,
        use_flashinfer: bool = True,
    ) -> dict:
        """Benchmark a single Attention prefill layer with sm_count SMs.

        KV cache is pre-filled with context_len tokens to simulate the
        case where attention prefill sees a realistic prior context.

        Prefer FlashInfer for realistic performance; falls back to
        scaled_dot_product_attention if FlashInfer is unavailable.

        Args:
            model_name: 'zamba2' | 'falcon_h1'
            batch_size: Batch size.
            seq_len: Sequence length (prefill tokens).
            sm_count: SM count restriction.
            context_len: Pre-filled KV cache length (simulates decode context).
            n_warmup: Warm-up iterations.
            n_measure: Measurement iterations.
            use_flashinfer: Try FlashInfer first if True.

        Returns:
            dict matching run_ssm_layer() schema with layer_type='attn'.
        """
        extractor = self._get_extractor(model_name)
        cfg = extractor.get_model_config()

        n_heads = cfg.get("n_attn_heads", 8)
        n_kv_heads = cfg.get("n_kv_heads", 8)
        head_dim = cfg.get("attn_head_dim", 256)
        hidden_size = cfg["hidden_size"]

        # Build inputs
        query = torch.randn(
            batch_size, seq_len, n_heads, head_dim,
            device=self.device, dtype=self.dtype
        )
        # Pre-filled KV cache: (batch, context_len + seq_len, 2, n_kv_heads, head_dim)
        total_kv_len = context_len + seq_len
        kv_cache = torch.randn(
            batch_size, total_kv_len, 2, n_kv_heads, head_dim,
            device=self.device, dtype=self.dtype
        )
        key = kv_cache[:, :, 0]    # (batch, total_kv_len, n_kv_heads, head_dim)
        value = kv_cache[:, :, 1]

        # Try FlashInfer; fall back to SDPA
        attn_fn = self._build_attn_fn(
            query, key, value, n_heads, n_kv_heads, head_dim, use_flashinfer
        )

        bytes_per_elem = 2
        # Read: Q + K + V; Write: output
        qkv_bytes = (
            query.numel() + key.numel() + value.numel()
        ) * bytes_per_elem
        out_bytes = query.numel() * bytes_per_elem
        total_bytes = qkv_bytes + out_bytes

        self.smctrl.set_sm_count(sm_count)
        try:
            result = self.meter.measure(attn_fn, n_warmup=n_warmup, n_measure=n_measure)
        finally:
            self.smctrl.reset()

        bw = self.bw_estimator.estimate(
            read_bytes=qkv_bytes,
            write_bytes=out_bytes,
            latency_ms=result["latency_ms"],
        )

        return {
            "latency_ms": result["latency_ms"],
            "latency_p99_ms": result["latency_p99_ms"],
            "achieved_bandwidth_GBs": bw["achieved_bandwidth_GBs"],
            "theoretical_bw_GBs": bw["theoretical_bw_GBs"],
            "bw_utilization_pct": bw["bw_utilization_pct"],
            "sm_count": sm_count,
            "sm_ratio": sm_count / self.smctrl.total_sm_count,
            "seq_len": seq_len,
            "batch_size": batch_size,
            "context_len": context_len,
            "layer_type": "attn",
            "model_name": model_name,
        }

    # ------------------------------------------------------------------
    # MLP layer benchmark
    # ------------------------------------------------------------------

    def run_mlp_layer(
        self,
        model_name: str,
        batch_size: int,
        seq_len: int,
        sm_count: int,
        n_warmup: int = 10,
        n_measure: int = 50,
    ) -> dict:
        """Benchmark a single MLP/FFN layer with sm_count SMs.

        Args:
            model_name: 'zamba2' | 'falcon_h1'
            batch_size: Batch size.
            seq_len: Sequence length.
            sm_count: SM count restriction.
            n_warmup: Warm-up iterations.
            n_measure: Measurement iterations.

        Returns:
            dict matching run_ssm_layer() schema with layer_type='mlp'.
        """
        extractor = self._get_extractor(model_name)
        cfg = extractor.get_model_config()
        hidden_size = cfg["hidden_size"]
        intermediate_size = cfg.get("intermediate_size", 4096)

        # Use a minimal 2-layer MLP if model loading fails
        try:
            layer = extractor.get_mlp_layer()
        except Exception:
            layer = self._build_fallback_mlp(hidden_size, intermediate_size)

        inputs = torch.randn(
            batch_size, seq_len, hidden_size,
            device=self.device, dtype=self.dtype
        )

        bytes_per_elem = 2
        weight_bytes = sum(p.numel() * bytes_per_elem for p in layer.parameters())
        io_bytes = 2 * batch_size * seq_len * hidden_size * bytes_per_elem

        def _run():
            with torch.no_grad():
                layer(inputs)

        self.smctrl.set_sm_count(sm_count)
        try:
            result = self.meter.measure(_run, n_warmup=n_warmup, n_measure=n_measure)
        finally:
            self.smctrl.reset()

        bw = self.bw_estimator.estimate(
            read_bytes=io_bytes // 2 + weight_bytes,
            write_bytes=io_bytes // 2,
            latency_ms=result["latency_ms"],
        )

        return {
            "latency_ms": result["latency_ms"],
            "latency_p99_ms": result["latency_p99_ms"],
            "achieved_bandwidth_GBs": bw["achieved_bandwidth_GBs"],
            "theoretical_bw_GBs": bw["theoretical_bw_GBs"],
            "bw_utilization_pct": bw["bw_utilization_pct"],
            "sm_count": sm_count,
            "sm_ratio": sm_count / self.smctrl.total_sm_count,
            "seq_len": seq_len,
            "batch_size": batch_size,
            "layer_type": "mlp",
            "model_name": model_name,
        }

    # ------------------------------------------------------------------
    # Fallback layer builders (no HuggingFace required)
    # ------------------------------------------------------------------

    def _build_fallback_ssm(self, model_name: str) -> nn.Module:
        if model_name == "zamba2":
            from src.models.zamba2 import FallbackSSMKernel
            return FallbackSSMKernel(device=self.device, dtype=self.dtype).to(self.device)
        else:
            from src.models.falcon_h1 import FallbackSSMBranch
            return FallbackSSMBranch(device=self.device, dtype=self.dtype).to(self.device)

    def _build_fallback_mlp(self, hidden_size: int, intermediate_size: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(hidden_size, intermediate_size, bias=False, dtype=self.dtype),
            nn.SiLU(),
            nn.Linear(intermediate_size, hidden_size, bias=False, dtype=self.dtype),
        ).to(self.device)

    def _build_attn_fn(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        use_flashinfer: bool,
    ):
        """Return a callable that performs one attention forward pass."""
        if use_flashinfer:
            try:
                return self._build_flashinfer_attn(query, key, value, n_heads, n_kv_heads, head_dim)
            except (ImportError, Exception):
                pass  # fall through to SDPA

        # SDPA fallback
        # Reshape to (batch, heads, seq, head_dim) for F.scaled_dot_product_attention
        q = query.transpose(1, 2)  # (B, n_heads, seq_len, head_dim)
        k = key.transpose(1, 2)    # (B, n_kv_heads, total_len, head_dim)
        v = value.transpose(1, 2)

        # Expand KV heads if GQA
        if n_kv_heads != n_heads:
            repeat = n_heads // n_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        def _sdpa():
            import torch.nn.functional as F
            with torch.no_grad():
                F.scaled_dot_product_attention(q, k, v, is_causal=True)

        return _sdpa

    def _build_flashinfer_attn(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
    ):
        import flashinfer
        batch_size, seq_len = query.shape[:2]
        total_kv_len = key.shape[1]

        # FlashInfer ragged prefill (all sequences same length, simplified)
        qo_indptr = torch.arange(
            0, (batch_size + 1) * seq_len, seq_len,
            device=self.device, dtype=torch.int32
        )
        kv_indptr = torch.arange(
            0, (batch_size + 1) * total_kv_len, total_kv_len,
            device=self.device, dtype=torch.int32
        )
        kv_indices = torch.arange(
            0, batch_size * total_kv_len,
            device=self.device, dtype=torch.int32
        )
        kv_last_page_len = torch.full(
            (batch_size,), total_kv_len,
            device=self.device, dtype=torch.int32
        )

        # Flatten query for FlashInfer: (batch * seq_len, n_heads, head_dim)
        q_flat = query.reshape(-1, n_heads, head_dim)
        # KV cache: (batch * total_kv_len, 2, n_kv_heads, head_dim)
        kv_flat = torch.stack([
            key.reshape(-1, n_kv_heads, head_dim),
            value.reshape(-1, n_kv_heads, head_dim),
        ], dim=1)

        wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=self.device),
            "NHD"
        )
        wrapper.plan(
            qo_indptr, kv_indptr,
            batch_size, n_heads, n_kv_heads, head_dim,
            causal=True
        )

        def _flashinfer_run():
            with torch.no_grad():
                wrapper.run(q_flat, kv_flat)

        return _flashinfer_run
