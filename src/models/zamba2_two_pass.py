"""
Two-pass SSM kernel variant for Zamba2-7B-Instruct benchmarking.

TwoPassSSMKernel is a drop-in replacement for FallbackSSMKernel (zamba2.py)
that uses the cooperative-barrier-free ssd_chunk_scan_twopass() algorithm
instead of mamba_chunk_scan_combined.  This makes the kernel safe to run under
CUDA Green Context SM restrictions (sm_count < n_thread_blocks), enabling
direct latency measurement at each SM step without the deadlock that kills the
original Triton SSD kernel.

TwoPassLayerRunner mirrors the subset of LayerRunner.run_ssm_layer() used by
_two_pass_worker.py, avoiding any modification to the existing layer_runner.py.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import math
from typing import Optional

import torch
import torch.nn as nn

from src.ops.ssd_two_pass import mamba_chunk_scan_combined_two_pass
from src.smctrl import SMController
from src.profiling.metrics import LatencyMeter, BandwidthEstimator


# ---------------------------------------------------------------------------
# TwoPassSSMKernel
# ---------------------------------------------------------------------------

class TwoPassSSMKernel(nn.Module):
    """Mamba-2 SSM layer using cooperative-barrier-free two-pass decomposition.

    Same constructor signature as FallbackSSMKernel (zamba2.py) for direct
    comparison.  Random dt, B, C tensors are generated each forward pass (same
    as FallbackSSMKernel) so the GEMM + scan bandwidth profile is comparable.

    Zamba2-7B defaults: n_heads=112, head_dim=64, d_state=64, n_groups=2.
    """

    def __init__(
        self,
        d_model: int = 3584,
        n_heads: int = 112,
        head_dim: int = 64,
        d_state: int = 64,
        chunk_size: int = 256,
        n_groups: int = 2,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.d_state = d_state
        self.chunk_size = chunk_size
        self.n_groups = n_groups
        self.device = device
        self.dtype = dtype

        inner_dim = n_heads * head_dim
        self.in_proj  = nn.Linear(d_model, 2 * inner_dim, bias=False, dtype=dtype, device=device)
        self.out_proj = nn.Linear(inner_dim, d_model,     bias=False, dtype=dtype, device=device)
        self.A_log    = nn.Parameter(torch.randn(n_heads, dtype=dtype, device=device))
        self.D        = nn.Parameter(torch.ones(n_heads,  dtype=dtype, device=device))
        self.dt_bias  = nn.Parameter(torch.randn(n_heads, dtype=dtype, device=device))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        batch, seq_len, _ = hidden_states.shape
        dev, dtype = hidden_states.device, hidden_states.dtype

        xz = self.in_proj(hidden_states)            # (batch, seq_len, 2 * inner_dim)
        x, z = xz.chunk(2, dim=-1)                 # each (batch, seq_len, inner_dim)
        x = x.view(batch, seq_len, self.n_heads, self.head_dim)

        dt  = torch.ones(batch, seq_len, self.n_heads, device=dev, dtype=dtype) * 0.1
        B   = torch.randn(batch, seq_len, self.n_groups, self.d_state, device=dev, dtype=dtype)
        C   = torch.randn(batch, seq_len, self.n_groups, self.d_state, device=dev, dtype=dtype)
        A   = -torch.exp(self.A_log.float()).to(dtype)

        y = mamba_chunk_scan_combined_two_pass(
            x, dt, A, B, C,
            chunk_size=self.chunk_size,
            D=self.D,
            dt_bias=self.dt_bias,
            dt_softplus=True,
        )                                           # (batch, seq_len, n_heads, head_dim)

        y = y.view(batch, seq_len, -1)             # (batch, seq_len, inner_dim)
        y = y * torch.sigmoid(z)
        return self.out_proj(y)


# ---------------------------------------------------------------------------
# TwoPassLayerRunner
# ---------------------------------------------------------------------------

class TwoPassLayerRunner:
    """Minimal LayerRunner for TwoPassSSMKernel.

    Mirrors the run_ssm_layer() interface used by _two_pass_worker.py without
    requiring any changes to the existing LayerRunner (layer_runner.py).
    """

    # Model config registry — same values as configs/models.yaml
    _MODEL_CONFIGS = {
        "zamba2": {
            "d_model":    3584,
            "n_heads":    112,
            "head_dim":   64,
            "d_state":    64,
            "chunk_size": 256,
            "n_groups":   2,
        },
        "falcon_h1": {
            "d_model":    3072,
            "n_heads":    24,
            "head_dim":   128,
            "d_state":    256,
            "chunk_size": 256,
            "n_groups":   1,
        },
    }

    def __init__(
        self,
        device: str = "cuda",
        total_sm_count: Optional[int] = None,
        theoretical_bw_GBs: Optional[float] = None,
    ):
        self.device = device
        self.smctrl = SMController(total_sm_count=total_sm_count)
        self.meter  = LatencyMeter(device=device)
        self.bw_estimator = BandwidthEstimator(theoretical_bw_GBs=theoretical_bw_GBs)
        self._cache: dict = {}

    def run_ssm_layer(
        self,
        model_name: str,
        batch_size: int,
        seq_len: int,
        sm_count: int,
        n_warmup: int = 10,
        n_measure: int = 50,
        skip_sm_control: bool = False,
    ) -> dict:
        """Benchmark TwoPassSSMKernel at sm_count SMs.

        Returns dict matching LayerRunner.run_ssm_layer() schema.
        """
        cfg = self._MODEL_CONFIGS[model_name]
        cache_key = (model_name, batch_size, seq_len)

        if cache_key not in self._cache:
            layer = TwoPassSSMKernel(
                d_model=cfg["d_model"],
                n_heads=cfg["n_heads"],
                head_dim=cfg["head_dim"],
                d_state=cfg["d_state"],
                chunk_size=cfg["chunk_size"],
                n_groups=cfg["n_groups"],
                device=self.device,
                dtype=torch.bfloat16,
            ).to(self.device)

            hidden = torch.randn(
                batch_size, seq_len, cfg["d_model"],
                device=self.device, dtype=torch.bfloat16
            )

            bytes_per_elem = 2
            weight_bytes = sum(p.numel() * bytes_per_elem for p in layer.parameters())
            read_bytes, write_bytes = BandwidthEstimator.ssm_bytes(
                batch=batch_size,
                seq_len=seq_len,
                hidden_size=cfg["d_model"],
                n_heads=cfg["n_heads"],
                head_dim=cfg["head_dim"],
                d_state=cfg["d_state"],
                weight_bytes=weight_bytes,
                bytes_per_elem=bytes_per_elem,
            )
            self._cache[cache_key] = (layer, hidden, read_bytes, write_bytes)

        layer, hidden_states, read_bytes, write_bytes = self._cache[cache_key]

        def _run():
            with torch.no_grad():
                layer(hidden_states)

        if not skip_sm_control:
            self.smctrl.set_sm_count(sm_count)
        stream = self.smctrl.get_stream()
        try:
            with torch.cuda.stream(stream):
                result = self.meter.measure(_run, n_warmup=n_warmup, n_measure=n_measure)
        finally:
            if not skip_sm_control:
                self.smctrl.reset()

        bw = self.bw_estimator.estimate(
            read_bytes=read_bytes,
            write_bytes=write_bytes,
            latency_ms=result["latency_ms"],
        )

        return {
            "latency_ms":              result["latency_ms"],
            "latency_p99_ms":          result["latency_p99_ms"],
            "achieved_bandwidth_GBs":  bw["achieved_bandwidth_GBs"],
            "theoretical_bw_GBs":      bw["theoretical_bw_GBs"],
            "bw_utilization_pct":      bw["bw_utilization_pct"],
            "sm_count":                sm_count,
            "sm_ratio":                sm_count / self.smctrl.total_sm_count,
            "seq_len":                 seq_len,
            "batch_size":              batch_size,
            "layer_type":              "ssm_twopass",
            "model_name":              model_name,
        }
