"""
src/nvml_monitor.py — Device-level NVML aggregate monitor.

S0.4 정책 준수:
  NVML 은 device-level aggregate 측정에만 사용한다.
  per-layer SM utilization 주장에 절대 사용하지 않는다.

  sm_util_pct  = GPU-wide SM activity  (short kernels: may read 0%)
  mem_util_pct = GPU memory controller busy fraction
  power_w      = instantaneous GPU board power

Polling rate: ~10 Hz (default interval_ms=100).
Sub-ms kernels are invisible at this resolution — use CUDA events for
per-kernel latency, and this module only for serving-level under-utilization
evidence (motivation Fig A).
"""

from __future__ import annotations

import csv
import threading
import time
from pathlib import Path
from typing import Optional


class NVMLMonitor:
    """Background-thread NVML poller (device-level aggregate only).

    Usage::
        mon = NVMLMonitor(device_id=0, interval_ms=100)
        mon.start()
        # ... run workload ...
        records = mon.stop()          # list[dict]
        save_nvml_csv(records, path)  # write CSV for plot_motivation.py
    """

    def __init__(self, device_id: int = 0, interval_ms: int = 100):
        self._device_id   = device_id
        self._interval_ms = interval_ms
        self._records: list[dict] = []
        self._lock        = threading.Lock()
        self._stop_event  = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._t0_ms: float = 0.0

        try:
            import pynvml as _nvml
            _nvml.nvmlInit()
            self._handle = _nvml.nvmlDeviceGetHandleByIndex(device_id)
            self._nvml   = _nvml
            self._ok     = True
        except Exception as exc:
            print(f"  NVML unavailable: {exc} — GPU metrics will not be recorded.")
            self._ok = False

    def start(self) -> None:
        if not self._ok:
            return
        self._t0_ms = time.perf_counter_ns() / 1e6
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def _poll(self) -> None:
        nvml   = self._nvml
        handle = self._handle
        while not self._stop_event.is_set():
            ts = time.perf_counter_ns() / 1e6 - self._t0_ms
            try:
                util = nvml.nvmlDeviceGetUtilizationRates(handle)
                mem  = nvml.nvmlDeviceGetMemoryInfo(handle)
                try:
                    pwr = nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                except nvml.NVMLError:
                    pwr = float("nan")
                row = {
                    "timestamp_ms":   round(ts, 1),
                    "sm_util_pct":    util.gpu,
                    "mem_util_pct":   util.memory,
                    "memory_used_mb": mem.used / (1024 ** 2),
                    "power_w":        pwr,
                }
                with self._lock:
                    self._records.append(row)
            except Exception:
                pass
            self._stop_event.wait(timeout=self._interval_ms / 1000.0)

    def stop(self) -> list[dict]:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        with self._lock:
            return list(self._records)

    @property
    def is_available(self) -> bool:
        return self._ok


# Public alias used by run_serving.py internals (underscore-prefixed name kept
# for backward compat with old code; new code should use NVMLMonitor directly)
_NVMLMonitor = NVMLMonitor


def save_nvml_csv(records: list[dict], path: Path) -> None:
    """Write NVML time-series records to CSV for plot_motivation.py Fig A."""
    if not records:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["timestamp_ms", "sm_util_pct", "mem_util_pct", "memory_used_mb", "power_w"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    duration_s = records[-1]["timestamp_ms"] / 1000.0 if records else 0.0
    print(f"  NVML CSV: {path}  ({len(records)} samples, {duration_s:.1f}s)")
    print("  Caption: sm_util_pct = device-level aggregate (NVML ~10Hz), NOT per-layer")


def nvml_summary(records: list[dict]) -> dict:
    """Compute summary statistics from NVML records."""
    if not records:
        return {}
    import statistics
    sm = [r["sm_util_pct"] for r in records]
    return {
        "sm_util_mean":   round(statistics.mean(sm), 1),
        "sm_util_median": round(statistics.median(sm), 1),
        "sm_util_p95":    round(sorted(sm)[int(len(sm) * 0.95)], 1),
    }


# Backward-compat alias
_nvml_summary = nvml_summary
