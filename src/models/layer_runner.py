"""
Single-layer independent execution utility.

LayerRunner runs individual SSM, Attention, or MLP layers in isolation,
applying libsmctrl SM restrictions before each benchmark run to measure
SM-scaling behavior.

Design notes:
  - SSM prefill uses mamba_chunk_scan_combined in parallel (not recurrent) mode.
  - Attention prefill uses FlashInfer BatchPrefillWithRaggedKVCacheWrapper with a
    pre-filled KV cache to simulate realistic context lengths, falling back to
    torch.nn.functional.scaled_dot_product_attention if FlashInfer is unavailable.
  - SM restriction is applied via SMController.set_sm_count() before measurement
    and reset() after.
  - CRITICAL: call LayerRunner.verify_sm_control() after construction to confirm
    SM restriction is functional. If SM control is not working, all sweep data will
    show latency independent of sm_count.
  - This runner measures latency + bandwidth only. SM hardware counters (wave
    quantization, SM utilization, occupancy) are captured separately via ncu
    using NCURunner / run_ncu_profile.py.
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

    Args:
        device: CUDA device string.
        dtype: Computation dtype (default: bfloat16).
        total_sm_count: Total SM count. Auto-detected if None.
        theoretical_bw_GBs: Peak memory bandwidth in GB/s. If None, queried
            from torch device properties. Pass the hardware.yaml value for
            exact spec-sheet numbers.
        sm_per_tpc: SMs per TPC for TPC mask computation (default 2, Ampere+).
    """

    def __init__(
        self,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        total_sm_count: Optional[int] = None,
        theoretical_bw_GBs: Optional[float] = None,
        sm_per_tpc: int = 2,
    ):
        self.device = device
        self.dtype = dtype
        self.smctrl = SMController(
            total_sm_count=total_sm_count,
            sm_per_tpc=sm_per_tpc,
        )
        self.meter = LatencyMeter(device=device)
        self.bw_estimator = BandwidthEstimator(
            theoretical_bw_GBs=theoretical_bw_GBs
        )

        # Lazy-loaded layer extractors
        self._extractors: dict = {}

    def verify_sm_control(self, verbose: bool = True) -> bool:
        """Verify that SM restriction actually changes kernel latency.

        MUST be called before running sweeps. If this returns False, all
        sm_count sweep data will be meaningless (latency independent of SM count).
        """
        print(f"Verifying SM control (backend: {self.smctrl.get_backend_name()}) ...")
        result = self.smctrl.verify_sm_control(verbose=verbose)
        if not result:
            print(
                "\nWARNING: SM control verification FAILED.\n"
                "  SM count restriction is not affecting kernel latency.\n"
                "  Sweep results will NOT show SM-scaling behavior.\n"
                "  Fix: build libsmctrl (https://github.com/msr-fiddle/libsmctrl)\n"
                "       and set LIBSMCTRL_PATH=<path>/libsmctrl.so\n"
            )
        return result

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
                raise ValueError(
                    f"Unknown model: {model_name!r}. Choose 'zamba2' or 'falcon_h1'."
                )
        return self._extractors[model_name]

    # ------------------------------------------------------------------
    # Core measurement helper
    # ------------------------------------------------------------------

    def _measure(self, fn, n_warmup: int, n_measure: int) -> dict:
        """Run latency measurement via CUDA events. No profiler overhead."""
        return self.meter.measure(fn, n_warmup=n_warmup, n_measure=n_measure)

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

        SSM prefill runs in parallel-scan mode (mamba_chunk_scan_combined).
        No per-token recurrent state buffer is needed.

        Returns:
            dict with latency_ms, latency_p99_ms, achieved_bandwidth_GBs,
            theoretical_bw_GBs, bw_utilization_pct, sm_count, sm_ratio,
            seq_len, batch_size, layer_type='ssm', model_name.
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

        cfg = extractor.get_model_config()
        hidden_size = cfg["hidden_size"]
        n_heads = cfg.get("n_ssm_heads", 64)
        head_dim = cfg.get("head_dim", cfg.get("ssm_head_dim", 32))
        d_state = cfg.get("d_state", 128)
        bytes_per_elem = 2  # bfloat16

        weight_bytes = sum(
            p.numel() * bytes_per_elem for p in layer.parameters()
        )
        read_bytes, write_bytes = BandwidthEstimator.ssm_bytes(
            batch=batch_size,
            seq_len=seq_len,
            hidden_size=hidden_size,
            n_heads=n_heads,
            head_dim=head_dim,
            d_state=d_state,
            weight_bytes=weight_bytes,
            bytes_per_elem=bytes_per_elem,
        )

        def _run():
            with torch.no_grad():
                layer(hidden_states)

        self.smctrl.set_sm_count(sm_count)
        try:
            result = self._measure(_run, n_warmup=n_warmup, n_measure=n_measure)
        finally:
            self.smctrl.reset()

        bw = self.bw_estimator.estimate(
            read_bytes=read_bytes,
            write_bytes=write_bytes,
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

        KV cache is pre-filled with context_len tokens to simulate a realistic
        decode-time KV context.

        Returns:
            dict matching run_ssm_layer() schema with layer_type='attn'.
        """
        extractor = self._get_extractor(model_name)
        cfg = extractor.get_model_config()

        n_heads = cfg.get("n_attn_heads", 8)
        n_kv_heads = cfg.get("n_kv_heads", 8)
        head_dim = cfg.get("attn_head_dim", 256)
        bytes_per_elem = 2

        query = torch.randn(
            batch_size, seq_len, n_heads, head_dim,
            device=self.device, dtype=self.dtype
        )
        total_kv_len = context_len + seq_len
        kv_cache = torch.randn(
            batch_size, total_kv_len, 2, n_kv_heads, head_dim,
            device=self.device, dtype=self.dtype
        )
        key = kv_cache[:, :, 0]
        value = kv_cache[:, :, 1]

        attn_fn = self._build_attn_fn(
            query, key, value, n_heads, n_kv_heads, head_dim, use_flashinfer
        )

        read_bytes, write_bytes = BandwidthEstimator.attn_bytes(
            batch=batch_size,
            seq_len=seq_len,
            total_kv_len=total_kv_len,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            bytes_per_elem=bytes_per_elem,
        )

        self.smctrl.set_sm_count(sm_count)
        try:
            result = self._measure(attn_fn, n_warmup=n_warmup, n_measure=n_measure)
        finally:
            self.smctrl.reset()

        bw = self.bw_estimator.estimate(
            read_bytes=read_bytes,
            write_bytes=write_bytes,
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

        Returns:
            dict matching run_ssm_layer() schema with layer_type='mlp'.
        """
        extractor = self._get_extractor(model_name)
        cfg = extractor.get_model_config()
        hidden_size = cfg["hidden_size"]
        intermediate_size = cfg.get("intermediate_size", 4096)
        bytes_per_elem = 2

        try:
            layer = extractor.get_mlp_layer()
        except Exception:
            layer = self._build_fallback_mlp(hidden_size, intermediate_size)

        inputs = torch.randn(
            batch_size, seq_len, hidden_size,
            device=self.device, dtype=self.dtype
        )

        weight_bytes = sum(
            p.numel() * bytes_per_elem for p in layer.parameters()
        )
        read_bytes, write_bytes = BandwidthEstimator.mlp_bytes(
            batch=batch_size,
            seq_len=seq_len,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            weight_bytes=weight_bytes,
            bytes_per_elem=bytes_per_elem,
        )

        def _run():
            with torch.no_grad():
                layer(inputs)

        self.smctrl.set_sm_count(sm_count)
        try:
            result = self._measure(_run, n_warmup=n_warmup, n_measure=n_measure)
        finally:
            self.smctrl.reset()

        bw = self.bw_estimator.estimate(
            read_bytes=read_bytes,
            write_bytes=write_bytes,
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
        """Return a zero-arg callable that performs one attention forward pass."""
        if use_flashinfer:
            try:
                return self._build_flashinfer_attn(
                    query, key, value, n_heads, n_kv_heads, head_dim
                )
            except (ImportError, Exception):
                pass  # fall through to SDPA

        # SDPA fallback
        q = query.transpose(1, 2)   # (B, n_heads, seq_len, head_dim)
        k = key.transpose(1, 2)     # (B, n_kv_heads, total_len, head_dim)
        v = value.transpose(1, 2)

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

        qo_indptr = torch.arange(
            0, (batch_size + 1) * seq_len, seq_len,
            device=self.device, dtype=torch.int32
        )
        kv_indptr = torch.arange(
            0, (batch_size + 1) * total_kv_len, total_kv_len,
            device=self.device, dtype=torch.int32
        )

        q_flat = query.reshape(-1, n_heads, head_dim)
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
