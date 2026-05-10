"""
Single-layer independent execution utility.

LayerRunner runs individual SSM, Attention, or MLP layers in isolation,
applying Green Context SM restrictions before each benchmark run to measure
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

from src.smctrl import SMController
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
        smctrl: Optional[SMController] = None,
    ):
        self.device = device
        self.dtype = dtype
        self.smctrl = smctrl or SMController(
            total_sm_count=total_sm_count,
            sm_per_tpc=sm_per_tpc,
        )
        self.meter = LatencyMeter(device=device)
        self.bw_estimator = BandwidthEstimator(
            theoretical_bw_GBs=theoretical_bw_GBs
        )

        # Lazy-loaded layer extractors
        self._extractors: dict = {}
        # Per-(model, batch, seq_len, use_fallback) cached (layer, hidden_states, read_bytes, write_bytes)
        self._ssm_cache: dict = {}
        # Per-(model, batch, seq_len, ctx_len, use_flashinfer) cached (attn_fn, read_bytes, write_bytes)
        self._attn_cache: dict = {}

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
                "  Check: (1) GPU supports Green Contexts (A100/H100/H200),\n"
                "         (2) MIG mode is disabled (nvidia-smi mig -e 0),\n"
                "         (3) CUDA primary context initialized before SMController.\n"
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
        skip_sm_control: bool = False,
        force_pytorch_scan: bool = False,
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

        ssm_key = (model_name, batch_size, seq_len, use_fallback_kernel, force_pytorch_scan)
        if ssm_key not in self._ssm_cache:
            if use_fallback_kernel or force_pytorch_scan:
                layer = self._build_fallback_ssm(model_name, force_pytorch_scan=force_pytorch_scan)
            else:
                try:
                    layer = extractor.get_ssm_layer()
                except Exception:
                    layer = self._build_fallback_ssm(model_name, force_pytorch_scan=force_pytorch_scan)

            cached_inputs = extractor.make_ssm_inputs(batch_size, seq_len)
            cached_hidden = cached_inputs["hidden_states"]

            cfg = extractor.get_model_config()
            hidden_size = cfg["hidden_size"]
            n_heads = cfg.get("n_ssm_heads", 64)
            head_dim = cfg.get("head_dim", cfg.get("ssm_head_dim", 32))
            d_state = cfg.get("d_state", 128)
            bytes_per_elem = 2
            weight_bytes = sum(p.numel() * bytes_per_elem for p in layer.parameters())
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
            self._ssm_cache[ssm_key] = (layer, cached_hidden, read_bytes, write_bytes)

        layer, hidden_states, read_bytes, write_bytes = self._ssm_cache[ssm_key]

        def _run():
            with torch.no_grad():
                layer(hidden_states)

        if not skip_sm_control:
            self.smctrl.set_sm_count(sm_count)
        stream = self.smctrl.get_stream()
        try:
            with torch.cuda.stream(stream):
                result = self._measure(_run, n_warmup=n_warmup, n_measure=n_measure)
        finally:
            if not skip_sm_control:
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
        skip_sm_control: bool = False,
    ) -> dict:
        """Benchmark a full Attention prefill layer (Q/K/V/O projections + attention).

        Measures the complete attention block:
          hidden_states → W_q/W_k/W_v → Q,K,V → FlashAttn/SDPA → W_o → output

        Previously only FlashAttn kernel was measured (pre-projected Q/K/V tensors
        passed directly), which excluded the dominant projection GEMMs. Now uses
        nn.Linear layers with the correct weight shapes from the model config so
        the GEMM load matches the real model.

        Context KV cache (context_len > 0) is pre-built as random tensors; the
        benchmark only projects the current seq_len query tokens.

        Returns:
            dict matching run_ssm_layer() schema with layer_type='attn'.
        """
        attn_key = (model_name, batch_size, seq_len, context_len, use_flashinfer)
        if attn_key not in self._attn_cache:
            self._attn_cache[attn_key] = self._build_attn_cache(
                model_name, batch_size, seq_len, context_len, use_flashinfer
            )
        attn_fn, read_bytes, write_bytes = self._attn_cache[attn_key]

        if not skip_sm_control:
            self.smctrl.set_sm_count(sm_count)
        stream = self.smctrl.get_stream()
        try:
            with torch.cuda.stream(stream):
                result = self._measure(attn_fn, n_warmup=n_warmup, n_measure=n_measure)
        finally:
            if not skip_sm_control:
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
        stream = self.smctrl.get_stream()
        try:
            with torch.cuda.stream(stream):
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
    # Attn setup cache builder
    # ------------------------------------------------------------------

    def _build_attn_cache(
        self,
        model_name: str,
        batch_size: int,
        seq_len: int,
        context_len: int,
        use_flashinfer: bool,
    ) -> tuple:
        """Allocate projection layers, input tensors, and attn callable once for caching.

        Called on the first run_attn_layer() for a given (model, batch, seq, ctx) key.
        Subsequent calls reuse the returned (attn_fn, read_bytes, write_bytes).
        """
        extractor = self._get_extractor(model_name)
        cfg = extractor.get_model_config()

        hidden_size = cfg["hidden_size"]
        n_heads     = cfg.get("n_attn_heads", 8)
        n_kv_heads  = cfg.get("n_kv_heads", 8)
        head_dim    = cfg.get("attn_head_dim", 256)
        attn_hidden = n_heads * head_dim
        kv_hidden   = n_kv_heads * head_dim
        bytes_per_elem = 2

        with torch.no_grad():
            w_q = nn.Linear(hidden_size, attn_hidden, bias=False, dtype=self.dtype).to(self.device)
            w_k = nn.Linear(hidden_size, kv_hidden,   bias=False, dtype=self.dtype).to(self.device)
            w_v = nn.Linear(hidden_size, kv_hidden,   bias=False, dtype=self.dtype).to(self.device)
            w_o = nn.Linear(attn_hidden, hidden_size,  bias=False, dtype=self.dtype).to(self.device)

        hidden_states = torch.randn(
            batch_size, seq_len, hidden_size,
            device=self.device, dtype=self.dtype
        )

        ctx_k = ctx_v = None
        if context_len > 0:
            ctx_k = torch.randn(
                batch_size, context_len, n_kv_heads, head_dim,
                device=self.device, dtype=self.dtype
            )
            ctx_v = torch.randn(
                batch_size, context_len, n_kv_heads, head_dim,
                device=self.device, dtype=self.dtype
            )

        attn_fn = self._build_attn_with_proj_fn(
            hidden_states, w_q, w_k, w_v, w_o,
            n_heads, n_kv_heads, head_dim, ctx_k, ctx_v,
            use_flashinfer,
        )

        proj_weight_bytes = (
            w_q.weight.numel() + w_k.weight.numel() +
            w_v.weight.numel() + w_o.weight.numel()
        ) * bytes_per_elem

        read_bytes, write_bytes = BandwidthEstimator.attn_bytes(
            batch=batch_size,
            seq_len=seq_len,
            total_kv_len=context_len + seq_len,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            hidden_size=hidden_size,
            proj_weight_bytes=proj_weight_bytes,
            bytes_per_elem=bytes_per_elem,
        )

        return attn_fn, read_bytes, write_bytes

    # ------------------------------------------------------------------
    # Fallback layer builders (no HuggingFace required)
    # ------------------------------------------------------------------

    def _build_fallback_ssm(self, model_name: str, force_pytorch_scan: bool = False) -> nn.Module:
        if model_name == "zamba2":
            from src.models.zamba2 import FallbackSSMKernel
            return FallbackSSMKernel(
                device=self.device, dtype=self.dtype,
                force_pytorch_scan=force_pytorch_scan,
            ).to(self.device)
        else:
            from src.models.falcon_h1 import FallbackSSMBranch
            return FallbackSSMBranch(device=self.device, dtype=self.dtype).to(self.device)

    def _build_fallback_mlp(self, hidden_size: int, intermediate_size: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(hidden_size, intermediate_size, bias=False, dtype=self.dtype),
            nn.SiLU(),
            nn.Linear(intermediate_size, hidden_size, bias=False, dtype=self.dtype),
        ).to(self.device)

    def _build_attn_with_proj_fn(
        self,
        hidden_states: torch.Tensor,
        w_q: nn.Linear,
        w_k: nn.Linear,
        w_v: nn.Linear,
        w_o: nn.Linear,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        ctx_k: Optional[torch.Tensor],
        ctx_v: Optional[torch.Tensor],
        use_flashinfer: bool,
    ):
        """Return a callable: hidden → Q/K/V proj → attention → O proj → output."""
        if use_flashinfer:
            try:
                return self._build_flashinfer_with_proj(
                    hidden_states, w_q, w_k, w_v, w_o,
                    n_heads, n_kv_heads, head_dim, ctx_k, ctx_v,
                )
            except (ImportError, Exception):
                pass  # fall through to SDPA

        # SDPA fallback — includes all four GEMMs + attention kernel
        batch_size, seq_len, _ = hidden_states.shape
        attn_hidden = n_heads * head_dim

        def _full_sdpa():
            import torch.nn.functional as F
            with torch.no_grad():
                q = w_q(hidden_states).view(batch_size, seq_len, n_heads, head_dim).transpose(1, 2)
                k = w_k(hidden_states).view(batch_size, seq_len, n_kv_heads, head_dim).transpose(1, 2)
                v = w_v(hidden_states).view(batch_size, seq_len, n_kv_heads, head_dim).transpose(1, 2)

                if ctx_k is not None:
                    k = torch.cat([ctx_k.transpose(1, 2), k], dim=2)
                    v = torch.cat([ctx_v.transpose(1, 2), v], dim=2)

                if n_kv_heads != n_heads:
                    rep = n_heads // n_kv_heads
                    k = k.repeat_interleave(rep, dim=1)
                    v = v.repeat_interleave(rep, dim=1)

                out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                out = out.transpose(1, 2).reshape(batch_size, seq_len, attn_hidden)
                return w_o(out)

        return _full_sdpa

    def _build_flashinfer_with_proj(
        self,
        hidden_states: torch.Tensor,
        w_q: nn.Linear,
        w_k: nn.Linear,
        w_v: nn.Linear,
        w_o: nn.Linear,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        ctx_k: Optional[torch.Tensor],
        ctx_v: Optional[torch.Tensor],
    ):
        import flashinfer
        batch_size, seq_len, _ = hidden_states.shape
        ctx_len = ctx_k.shape[1] if ctx_k is not None else 0
        total_kv_len = ctx_len + seq_len
        attn_hidden = n_heads * head_dim

        qo_indptr = torch.arange(
            0, (batch_size + 1) * seq_len, seq_len,
            device=self.device, dtype=torch.int32
        )
        kv_indptr = torch.arange(
            0, (batch_size + 1) * total_kv_len, total_kv_len,
            device=self.device, dtype=torch.int32
        )
        workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=self.device)
        wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(workspace, "NHD")
        wrapper.plan(
            qo_indptr, kv_indptr,
            batch_size, n_heads, n_kv_heads, head_dim,
            causal=True,
        )

        def _flashinfer_run():
            with torch.no_grad():
                q = w_q(hidden_states).view(-1, n_heads, head_dim)
                k_new = w_k(hidden_states).view(-1, n_kv_heads, head_dim)
                v_new = w_v(hidden_states).view(-1, n_kv_heads, head_dim)

                if ctx_k is not None:
                    k = torch.cat([ctx_k.view(-1, n_kv_heads, head_dim), k_new], dim=0)
                    v = torch.cat([ctx_v.view(-1, n_kv_heads, head_dim), v_new], dim=0)
                else:
                    k, v = k_new, v_new

                kv_flat = torch.stack([k, v], dim=1)
                out = wrapper.run(q, kv_flat)
                out = out.view(batch_size, seq_len, attn_hidden)
                return w_o(out)

        return _flashinfer_run

    # kept for internal use by legacy callers if any
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
        """Legacy: attention kernel only (no projections). Kept for compatibility."""
        if use_flashinfer:
            try:
                return self._build_flashinfer_attn(
                    query, key, value, n_heads, n_kv_heads, head_dim
                )
            except (ImportError, Exception):
                pass

        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
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
        """Legacy: flashinfer kernel only (no projections). Kept for compatibility."""
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
            causal=True,
        )

        def _flashinfer_run():
            with torch.no_grad():
                wrapper.run(q_flat, kv_flat)

        return _flashinfer_run
