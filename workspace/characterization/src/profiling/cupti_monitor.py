"""
In-process CUPTI hardware counter collector via torch.profiler.

Replaces NVMLMonitor's device-level binary measurement with per-kernel
SM efficiency and occupancy collected via torch.profiler's CUPTI backend.

What changes vs NVML
---------------------
  NVML (previous):
    - Granularity: device-level, one sample per ~167ms window
    - Measures:    "was any kernel running?" (binary on/off)
    - Artifact:    0% for short kernels, 100% for long ones regardless of
                   how many SMs were actually doing work
    - Cannot see:  wave quantization idle SMs, warp stalls

  CUPTIMonitor (this module):
    - Granularity: per-kernel, every invocation during measurement phase
    - Measures:    sm_eff_pct = active SM cycles / total SM cycles × 100
                   occupancy_pct = active warps / theoretical max warps × 100
    - No artifact: short kernels report correct efficiency; wave quantization
                   (idle SMs in last wave) reduces sm_eff_pct proportionally
    - Can see:     wave quantization, warp stalls (memory latency)

Metric interpretation
----------------------
  sm_eff_pct (SM efficiency):
    Equivalent to ncu's sm__active_cycles_sum / sm__cycles_elapsed_sum × 100.
    With sm_count restriction:
      - Perfect linear scaling: sm_eff_pct ≈ 100% at all sm_counts
      - Wave quantization:      sm_eff_pct < 100% when last wave is partial
        e.g. grid_size=100, sm_count=27 → 4 waves, last wave=19/27 SMs
             → sm_eff_pct ≈ (3×27 + 19) / (4×27) = 100/108 ≈ 92.6%
      - Memory stall:           sm_eff_pct < wave_efficiency → warp stalls dominate

  occupancy_pct (achieved warp occupancy):
    Active warps as a fraction of theoretical maximum per SM.
    Low occupancy with high sm_eff → compute bound.
    Low occupancy with low sm_eff → memory latency bound (warps stalling).

Backend selection
------------------
  PRIMARY:  torch.profiler + _ExperimentalConfig(profiler_metrics=...)
            → populates _KinetoEvent.sm_efficiency() and
              _KinetoEvent.avg_estimated_achieved_occupancy()
            → requires: PyTorch ≥ 2.0, CUDA toolkit with CUPTI
            → overhead: ~2-5× vs bare execution (no kernel replay needed
              for sm_efficiency; replay needed only for stall reasons)

  FALLBACK: torch.profiler without CUPTI metrics
            → kernel timing and names only (no SM counters)
            → sm_eff_pct = NaN, occupancy_pct = NaN
            → still useful for dominant kernel identification

Note on overhead vs ncu
------------------------
  ncu (run_ncu_profile.py): 10-100× overhead, subprocess required.
  CUPTIMonitor:             2-5× overhead, in-process.

  CUPTIMonitor runs a SEPARATE short profiling pass (n_cupti_measure ≪
  n_measure) alongside the main timing pass (which has zero profiler
  overhead). Total overhead is small for the sweep.
"""

import math
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

def _detect_cupti_backend() -> str:
    """Return 'cupti', 'profiler_only', or 'none'."""
    if not torch.cuda.is_available():
        return "none"
    try:
        from torch.profiler import profile, ProfilerActivity
        from torch._C._profiler import _ExperimentalConfig
        # Try constructing config with SM metrics
        _ExperimentalConfig(
            profiler_metrics=["kineto__sm_efficiency"],
            profiler_measure_per_kernel=True,
        )
        return "cupti"
    except Exception:
        pass
    try:
        from torch.profiler import profile, ProfilerActivity  # noqa
        return "profiler_only"
    except Exception:
        return "none"


_BACKEND = _detect_cupti_backend()


# ---------------------------------------------------------------------------
# Per-kernel result dataclass
# ---------------------------------------------------------------------------

@dataclass
class KernelStats:
    """Hardware counter stats for a single CUDA kernel (averaged over calls)."""
    name: str
    call_count: int                    # times this kernel was called in measurement
    total_duration_us: float           # sum of all call durations
    avg_duration_us: float             # mean duration per call
    sm_eff_pct: float                  # SM efficiency (NaN if CUPTI unavailable)
    occupancy_pct: float               # achieved warp occupancy (NaN if CUPTI unavailable)
    sm_eff_samples: list = field(default_factory=list, repr=False)
    occupancy_samples: list = field(default_factory=list, repr=False)

    @property
    def cupti_available(self) -> bool:
        return not math.isnan(self.sm_eff_pct)


@dataclass
class CUPTIResult:
    """Full measurement result from one CUPTIMonitor.measure() call."""
    kernels: list            # list[KernelStats], sorted by total_duration_us desc
    backend: str             # 'cupti', 'profiler_only', or 'none'
    n_measure_runs: int      # number of fn() calls profiled

    @property
    def dominant(self) -> Optional[KernelStats]:
        """Kernel with the most total execution time."""
        return self.kernels[0] if self.kernels else None

    def summary(self) -> str:
        d = self.dominant
        if d is None:
            return "CUPTIResult(no kernels)"
        sm = f"{d.sm_eff_pct:.1f}%" if not math.isnan(d.sm_eff_pct) else "N/A"
        occ = f"{d.occupancy_pct:.1f}%" if not math.isnan(d.occupancy_pct) else "N/A"
        return (
            f"dominant={d.name!r}  "
            f"avg={d.avg_duration_us:.1f}µs  "
            f"sm_eff={sm}  occupancy={occ}  "
            f"n_kernels={len(self.kernels)}  backend={self.backend}"
        )


# ---------------------------------------------------------------------------
# Profiler event parser
# ---------------------------------------------------------------------------

def _parse_kineto_events(prof) -> tuple[list[KernelStats], bool]:
    """Extract per-kernel stats from torch.profiler results.

    Returns (kernel_stats_list, cupti_metrics_populated).
    """
    accum: dict[str, dict] = {}
    cupti_populated = False

    try:
        raw_events = prof.profiler.kineto_results.events()
    except AttributeError:
        # Older PyTorch: fall back to key_averages()
        return _parse_key_averages(prof), False

    for e in raw_events:
        # Filter to CUDA kernel events (device_type == CUDA)
        try:
            # DeviceType.CUDA == 1
            if e.device_type() != 1:
                continue
        except AttributeError:
            continue

        name = e.name()
        duration_us = e.duration_ns() / 1000.0
        if duration_us <= 0:
            continue

        entry = accum.setdefault(name, {
            "name": name,
            "call_count": 0,
            "total_duration_us": 0.0,
            "sm_eff_samples": [],
            "occupancy_samples": [],
        })
        entry["call_count"] += 1
        entry["total_duration_us"] += duration_us

        # SM efficiency (populated when experimental_config requested it)
        try:
            sm_eff = e.sm_efficiency()
            if sm_eff > 0:
                entry["sm_eff_samples"].append(sm_eff)
                cupti_populated = True
        except AttributeError:
            pass

        # Achieved warp occupancy
        try:
            occ = e.avg_estimated_achieved_occupancy()
            if occ > 0:
                entry["occupancy_samples"].append(occ)
        except AttributeError:
            pass

    kernels = []
    for v in accum.values():
        sm_eff = (
            float(np.mean(v["sm_eff_samples"]))
            if v["sm_eff_samples"] else float("nan")
        )
        occ = (
            float(np.mean(v["occupancy_samples"]))
            if v["occupancy_samples"] else float("nan")
        )
        kernels.append(KernelStats(
            name=v["name"],
            call_count=v["call_count"],
            total_duration_us=v["total_duration_us"],
            avg_duration_us=v["total_duration_us"] / v["call_count"],
            sm_eff_pct=sm_eff,
            occupancy_pct=occ,
            sm_eff_samples=v["sm_eff_samples"],
            occupancy_samples=v["occupancy_samples"],
        ))

    kernels.sort(key=lambda k: k.total_duration_us, reverse=True)
    return kernels, cupti_populated


def _parse_key_averages(prof) -> list[KernelStats]:
    """Fallback parser using prof.key_averages() (no CUPTI metrics)."""
    kernels = []
    for evt in prof.key_averages():
        if evt.cuda_time_total <= 0:
            continue
        kernels.append(KernelStats(
            name=evt.key,
            call_count=evt.count,
            total_duration_us=float(evt.cuda_time_total),
            avg_duration_us=float(evt.cuda_time_total) / max(evt.count, 1),
            sm_eff_pct=float("nan"),
            occupancy_pct=float("nan"),
        ))
    kernels.sort(key=lambda k: k.total_duration_us, reverse=True)
    return kernels


# ---------------------------------------------------------------------------
# CUPTIMonitor
# ---------------------------------------------------------------------------

class CUPTIMonitor:
    """Per-kernel CUPTI hardware counter collector.

    Runs a SHORT dedicated profiling pass (n_cupti_measure << n_measure) to
    collect SM efficiency and occupancy. The main latency measurement runs
    separately without profiler overhead.

    Usage:
        monitor = CUPTIMonitor()

        # Main latency measurement (no overhead)
        result = meter.measure(fn, n_warmup=100, n_measure=200)

        # CUPTI profiling pass (short, slight overhead)
        cupti = monitor.measure(fn, n_warmup=10, n_measure=20)

        print(cupti.summary())
        print(f"sm_eff={cupti.dominant.sm_eff_pct:.1f}%")

    The two passes are kept separate so the main latency stats are unaffected
    by profiler overhead.
    """

    def __init__(self):
        self.backend = _BACKEND

    def is_available(self) -> bool:
        return self.backend != "none"

    def has_sm_metrics(self) -> bool:
        return self.backend == "cupti"

    def measure(
        self,
        fn,
        n_warmup: int = 5,
        n_measure: int = 20,
    ) -> CUPTIResult:
        """Profile fn() for per-kernel SM efficiency and occupancy.

        Args:
            fn:         Callable that launches CUDA kernels (no args).
            n_warmup:   Warmup calls before profiling starts.
            n_measure:  Number of fn() calls to profile.
                        Keep small (10-30) — profiler has ~2-5× overhead.
                        For latency accuracy, use LatencyMeter.measure() separately.

        Returns:
            CUPTIResult with per-kernel KernelStats sorted by total_duration_us.
        """
        if self.backend == "none":
            return CUPTIResult(kernels=[], backend="none", n_measure_runs=0)

        # Warmup outside profiler window
        for _ in range(n_warmup):
            fn()
        torch.cuda.synchronize()

        return self._profile(fn, n_measure)

    def _profile(self, fn, n_measure: int) -> CUPTIResult:
        from torch.profiler import profile, ProfilerActivity

        # Build experimental config for CUPTI SM metrics if available
        extra_kwargs = {}
        if self.backend == "cupti":
            try:
                from torch._C._profiler import _ExperimentalConfig
                extra_kwargs["experimental_config"] = _ExperimentalConfig(
                    profiler_metrics=[
                        "kineto__sm_efficiency",
                        "kineto__occupancy",
                    ],
                    profiler_measure_per_kernel=True,
                )
            except Exception:
                # Config construction failed at runtime (GPU mismatch, etc.)
                pass

        with profile(
            activities=[ProfilerActivity.CUDA],
            record_shapes=False,
            with_stack=False,
            **extra_kwargs,
        ) as prof:
            for _ in range(n_measure):
                fn()
            torch.cuda.synchronize()

        kernels, cupti_ok = _parse_kineto_events(prof)

        actual_backend = "cupti" if cupti_ok else "profiler_only"
        return CUPTIResult(
            kernels=kernels,
            backend=actual_backend,
            n_measure_runs=n_measure,
        )
