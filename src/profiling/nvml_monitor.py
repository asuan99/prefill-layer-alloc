"""
Background NVML monitor for real-time GPU metric collection.

Spawns a daemon thread that polls pynvml at a configurable interval
and accumulates rows into a DataFrame returned on stop().
"""

import threading
import time
from typing import Optional
import pandas as pd

try:
    import pynvml
    _PYNVML_AVAILABLE = True
except ImportError:
    _PYNVML_AVAILABLE = False


class NVMLMonitor:
    """Background thread GPU metrics collector using pynvml.

    Usage:
        monitor = NVMLMonitor(device_id=0)
        monitor.start(interval_ms=100)
        # ... run experiment ...
        df = monitor.stop()
        # df columns: timestamp_ms, sm_util_pct, mem_util_pct,
        #             memory_used_mb, power_w
    """

    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self._handle = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._records: list[dict] = []
        self._lock = threading.Lock()
        self._interval_ms: int = 100
        self._start_time_ms: float = 0.0

        if not _PYNVML_AVAILABLE:
            raise ImportError("pynvml is required: pip install pynvml")

        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            ts_ms = (time.perf_counter_ns() / 1_000_000.0) - self._start_time_ms

            util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            try:
                power_mw = pynvml.nvmlDeviceGetPowerUsage(self._handle)
                power_w = power_mw / 1000.0
            except pynvml.NVMLError:
                power_w = float("nan")

            row = {
                "timestamp_ms": ts_ms,
                "sm_util_pct": util.gpu,
                "mem_util_pct": util.memory,
                "memory_used_mb": mem_info.used / (1024 ** 2),
                "power_w": power_w,
            }

            with self._lock:
                self._records.append(row)

            self._stop_event.wait(timeout=self._interval_ms / 1000.0)

    def start(self, interval_ms: int = 100) -> None:
        """Start background polling thread.

        Args:
            interval_ms: Polling interval in milliseconds.
        """
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("NVMLMonitor is already running")

        self._interval_ms = interval_ms
        self._records = []
        self._stop_event.clear()
        self._start_time_ms = time.perf_counter_ns() / 1_000_000.0

        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> pd.DataFrame:
        """Stop polling and return collected metrics as DataFrame.

        Returns:
            DataFrame with columns:
                timestamp_ms    - elapsed milliseconds since start()
                sm_util_pct     - SM utilization percentage (0-100)
                mem_util_pct    - memory controller utilization (0-100)
                memory_used_mb  - used GPU memory in MB
                power_w         - GPU power draw in Watts
        """
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

        with self._lock:
            records = list(self._records)

        if not records:
            return pd.DataFrame(columns=[
                "timestamp_ms", "sm_util_pct", "mem_util_pct",
                "memory_used_mb", "power_w"
            ])

        return pd.DataFrame(records)

    def snapshot(self) -> dict:
        """Return a single instantaneous metric snapshot."""
        util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
        try:
            power_w = pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
        except pynvml.NVMLError:
            power_w = float("nan")

        return {
            "sm_util_pct": util.gpu,
            "mem_util_pct": util.memory,
            "memory_used_mb": mem_info.used / (1024 ** 2),
            "power_w": power_w,
        }

    def __del__(self):
        try:
            self._stop_event.set()
            pynvml.nvmlShutdown()
        except Exception:
            pass
