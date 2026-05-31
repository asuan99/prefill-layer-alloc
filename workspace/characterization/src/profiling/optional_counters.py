"""
optional_counters.py — optional hardware-counter enrichment gate (S0.4).

All ncu / CUPTI / nsys enrichment is routed through this module.  When the
required binaries are absent or permissions are insufficient
(ERR_NVGPUCTRPERM / no CAP_SYS_ADMIN), every call returns a graceful skip
and the headline measurement pipeline continues without interruption.

Headline evidence sources (counter-free, always available):
  (a) latency vs SM count — saturation point → free-SM zone
  (b) bw_utilization_pct  — memory-bound saturation direct evidence
Both are produced by BandwidthEstimator (src/profiling/metrics.py) and
WaveEstimator (src/profiling/wave_estimator.py), with no hardware counters.

Optional enrichment (this module, requires ncu/CUPTI):
  - sm_util_per_sm_pct, wave_efficiency_pct, achieved_occupancy_pct
  - Per-kernel grid size for wave-idle annotation verification
  - nsys timeline reconstruction

NVML usage contract (nvml_monitor.py):
  NVML is ONLY used for device-level aggregate metrics:
    sm_util_pct (device-wide), mem_util_pct, memory_used_mb, power_w.
  NVML must NOT be used to derive per-layer SM utilisation — NVML sampling
  frequency (~6 Hz) is far too low for sub-ms kernels and produces artefacts
  (0 % for short kernels, 100 % for long ones).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Availability probes (lazy, cached)
# ---------------------------------------------------------------------------

_ncu_available:   Optional[bool] = None
_cupti_available: Optional[bool] = None


def _probe_ncu() -> bool:
    global _ncu_available
    if _ncu_available is not None:
        return _ncu_available
    try:
        from src.profiling.ncu_runner import NCURunner
        NCURunner.check_permissions()
        _ncu_available = True
    except Exception as e:
        _log.info("ncu unavailable: %s — counter enrichment will be skipped.", e)
        _ncu_available = False
    return _ncu_available


def _probe_cupti() -> bool:
    global _cupti_available
    if _cupti_available is not None:
        return _cupti_available
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        # Probe CUPTI via torch.profiler with a trivial kernel
        import torch.profiler
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA],
            with_stack=False,
        ):
            torch.zeros(1, device="cuda")
        _cupti_available = True
    except Exception as e:
        _log.info("CUPTI unavailable: %s — occupancy enrichment will be skipped.", e)
        _cupti_available = False
    return _cupti_available


def is_ncu_available() -> bool:
    """Return True if ncu is accessible with sufficient permissions."""
    return _probe_ncu()


def is_cupti_available() -> bool:
    """Return True if CUPTI/torch.profiler hardware counters are accessible."""
    return _probe_cupti()


# ---------------------------------------------------------------------------
# Optional enrichment entry points
# ---------------------------------------------------------------------------

def enrich_with_ncu(
    row: dict,
    *,
    layer_type: str,
    model_cfg: dict,
    sm_count: int,
    seq_len: int,
    batch_size: int,
    **kwargs: Any,
) -> dict:
    """Add ncu hardware counter columns to *row* if ncu is available.

    Returns *row* unchanged (with no extra keys) when ncu is unavailable.
    Counter columns added when available:
      sm_util_per_sm_pct, wave_efficiency_pct, achieved_occupancy_pct,
      n_waves, last_wave_blocks, grid_size, analytical_wave_eff
    """
    if not is_ncu_available():
        _log.debug("ncu unavailable — skipping enrichment for %s sm=%d seq=%d",
                   layer_type, sm_count, seq_len)
        return row

    try:
        from src.profiling.ncu_runner import NCURunner
        runner = NCURunner()
        metrics = runner.profile(
            layer_type=layer_type,
            model_cfg=model_cfg,
            sm_count=sm_count,
            seq_len=seq_len,
            batch_size=batch_size,
            **kwargs,
        )
        row.update(metrics)
    except Exception as e:
        _log.warning("ncu enrichment failed for %s sm=%d seq=%d: %s",
                     layer_type, sm_count, seq_len, e)
    return row


def enrich_with_wave_annotation(
    row: dict,
    *,
    layer_type: str,
    model_cfg: dict,
    sm_count: int,
    seq_len: int,
    batch_size: int,
) -> dict:
    """Add WaveEstimator analytical wave-idle annotation (counter-free).

    Always succeeds — does not require ncu or CUPTI.
    Added keys: analytical_n_waves, analytical_wave_eff, analytical_idle_sm_last_wave
    """
    try:
        from src.profiling.wave_estimator import WaveEstimator
        ssm = model_cfg.get("ssm", {})
        if layer_type in ("ssm", "ssm_chunked"):
            stats = WaveEstimator.ssm_prefill(
                batch=batch_size,
                seq_len=seq_len,
                n_heads=ssm.get("n_heads", 64),
                chunk_size=ssm.get("chunk_size", 256),
                sm_count=sm_count,
            )
        else:
            return row
        row["analytical_n_waves"]            = stats.get("n_waves")
        row["analytical_wave_eff"]           = stats.get("wave_efficiency")
        row["analytical_idle_sm_last_wave"]  = stats.get("idle_sm_last_wave")
    except Exception as e:
        _log.debug("WaveEstimator annotation failed: %s", e)
    return row


def reconstruct_timeline(
    nsys_rep: "str | None" = None,
    **kwargs: Any,
) -> Optional[dict]:
    """Optional nsys timeline reconstruction — skipped when nsys unavailable.

    Returns None when nsys is not available or the reconstruction fails.
    This function is a stub; real implementation runs ``nsys stats``.
    """
    if nsys_rep is None:
        return None
    try:
        import subprocess, shutil
        if shutil.which("nsys") is None:
            _log.info("nsys binary not found — timeline reconstruction skipped.")
            return None
        result = subprocess.run(
            ["nsys", "stats", "--report", "cuda_gpu_trace", nsys_rep],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            _log.warning("nsys stats failed: %s", result.stderr[:200])
            return None
        return {"nsys_stdout": result.stdout}
    except Exception as e:
        _log.info("nsys timeline reconstruction skipped: %s", e)
        return None
