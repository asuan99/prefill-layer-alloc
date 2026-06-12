"""
bw_probe.py — thin wrapper that PROMOTES BandwidthEstimator to the primary
instrument for the v2 study.

In v1 bandwidth was a secondary readout. In v2 the central question is the
saturation *mechanism* (bandwidth vs grid), so achieved bandwidth is measured at
every SM-sweep point and compared to the HBM ceiling. This module is the single
call site for that, so every experiment computes BW identically.

Contract
--------
  achieved_bw_gbs : MEASURED  — (read+write bytes) / latency, via BandwidthEstimator.
  bw_util_pct     : DERIVED   — achieved / ceiling, where ceiling is the verified
                                hardware.yaml / sweep_spec value (A100-SXM4-80GB =
                                2039 GB/s), NOT torch's queried theoretical BW.

``src/profiling/metrics.py::BandwidthEstimator`` is NOT modified — only wrapped.
Byte-count helpers (ssm_scan_bytes / attn_bytes / mlp_bytes) are re-exported from
it so callers count bytes the same way the estimator expects.

This module imports torch (transitively, via metrics) and is therefore for the
GPU experiments (E1/E2/E3/E4) only. The GPU-free E0 must NOT import it.
"""

from __future__ import annotations

import os
import sys

# --- path bootstrap (same convention as the v1 runners) ---------------------
_here = os.path.dirname(os.path.abspath(__file__))
# experiments/common/ -> experiments -> characterization -> workspace
_CHAR = os.path.join(_here, "..", "..")
_WORKSPACE = os.path.join(_here, "..", "..", "..")
for _p in (_WORKSPACE, _CHAR):
    _p = os.path.abspath(_p)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shared.sweep_spec import hbm_bw_GBs  # verified ceiling (single source)
from src.profiling.metrics import BandwidthEstimator  # immutable — wrapped only

__all__ = ["BWProbe", "ssm_scan_bytes", "attn_bytes", "mlp_bytes"]

# Re-export the byte counters so experiments don't reach into src directly.
ssm_scan_bytes = BandwidthEstimator.ssm_scan_bytes
attn_bytes = BandwidthEstimator.attn_bytes
mlp_bytes = BandwidthEstimator.mlp_bytes


class BWProbe:
    """Primary bandwidth instrument for an SM sweep.

    Usage:
        probe = BWProbe()                      # ceiling from sweep_spec (2039 GB/s)
        rb, wb = ssm_scan_bytes(batch, seq, n_heads, head_dim, d_state, n_groups)
        m = probe.from_bytes(rb, wb, latency_ms)   # at each SM point
        # m["achieved_bw_gbs"]  -> MEASURED
        # m["bw_util_pct"]      -> DERIVED
    """

    def __init__(self, ceiling_gbs: float | None = None):
        # Passing the ceiling means BandwidthEstimator does NOT touch torch device
        # properties; util is computed against the hardware.yaml/sweep_spec value.
        self.ceiling_gbs = float(ceiling_gbs) if ceiling_gbs is not None else float(hbm_bw_GBs())
        self._est = BandwidthEstimator(theoretical_bw_GBs=self.ceiling_gbs)

    def from_bytes(self, read_bytes: int, write_bytes: int, latency_ms: float) -> dict:
        """Compute achieved BW (measured) + util (derived) from bytes & latency.

        Returns:
            {
              "achieved_bw_gbs": float,   # MEASURED
              "bw_util_pct":     float,   # DERIVED  (0..100)
              "ceiling_gbs":     float,   # METADATA (the verified ceiling used)
              "total_bytes":     int,     # DERIVED  (byte-count input)
            }
        """
        d = self._est.estimate(read_bytes=read_bytes, write_bytes=write_bytes,
                               latency_ms=latency_ms)
        achieved = d["achieved_bandwidth_GBs"]
        return {
            "achieved_bw_gbs": achieved,
            "bw_util_pct": self.util_pct(achieved),
            "ceiling_gbs": self.ceiling_gbs,
            "total_bytes": d["total_bytes"],
        }

    def from_achieved(self, achieved_bw_gbs: float) -> dict:
        """Wrap an already-measured achieved BW (e.g. from LayerRunner output).

        LayerRunner.run_*_layer already calls BandwidthEstimator internally and
        returns ``achieved_bandwidth_GBs``; this re-derives util against the
        canonical ceiling so the number is identical across experiments.
        """
        return {
            "achieved_bw_gbs": float(achieved_bw_gbs),
            "bw_util_pct": self.util_pct(achieved_bw_gbs),
            "ceiling_gbs": self.ceiling_gbs,
        }

    def util_pct(self, achieved_bw_gbs: float) -> float:
        """DERIVED: achieved / ceiling as a percentage."""
        if self.ceiling_gbs <= 0:
            return float("nan")
        return 100.0 * float(achieved_bw_gbs) / self.ceiling_gbs
