"""
Latency and bandwidth measurement utilities.

Provides CUDA-event based timing (eliminates CPU-GPU transfer overhead)
and bandwidth estimation from tensor sizes and measured latency.
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
            dict with latency statistics in milliseconds:
                latency_ms       - median latency
                latency_mean_ms  - mean latency
                latency_p99_ms   - 99th percentile
                latency_min_ms   - minimum
                latency_max_ms   - maximum
                latency_std_ms   - standard deviation
        """
        if kwargs is None:
            kwargs = {}

        # Create CUDA events for timing
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_measure)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_measure)]

        # Warm-up
        for _ in range(n_warmup):
            fn(*args, **kwargs)
        torch.cuda.synchronize()

        # Measurement
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

        Useful for measuring operations that include CPU overhead
        (e.g., SM reconfiguration calls).

        Args:
            sync_before: If True, synchronize CUDA before starting the timer.
            sync_after: If True, synchronize CUDA after the operation.
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
    """Estimates achieved memory bandwidth from tensor sizes and latency.

    For memory-bound operations (SSM prefill, attention), bandwidth
    utilization is a better saturation indicator than raw latency.
    """

    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self._theoretical_bw_GBs = self._query_theoretical_bw()

    def _query_theoretical_bw(self) -> float:
        """Query peak memory bandwidth from pynvml or use a hardcoded table."""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_id)
            # Memory clock in MHz, bus width in bits
            mem_clock_mhz = pynvml.nvmlDeviceGetMaxClockInfo(
                handle, pynvml.NVML_CLOCK_MEM
            )
            props = torch.cuda.get_device_properties(self.device_id)
            # Approximate: BW = 2 × mem_clock × bus_width / 8 (GB/s)
            # pynvml doesn't expose bus width directly; use torch props
            # This is a rough estimate; hardware.yaml has accurate values
            return float(mem_clock_mhz * 2 / 1000)  # very rough placeholder
        except Exception:
            return 1000.0  # fallback: 1 TB/s (A100-class)

    def estimate(
        self,
        read_bytes: int,
        write_bytes: int,
        latency_ms: float,
    ) -> dict:
        """Estimate achieved bandwidth from transfer size and latency.

        Args:
            read_bytes: Total bytes read from GPU memory.
            write_bytes: Total bytes written to GPU memory.
            latency_ms: Kernel latency in milliseconds.

        Returns:
            dict with achieved_GBs and utilization_pct.
        """
        total_bytes = read_bytes + write_bytes
        achieved_GBs = (total_bytes / 1e9) / (latency_ms / 1000.0)
        utilization_pct = (achieved_GBs / self._theoretical_bw_GBs * 100.0
                           if self._theoretical_bw_GBs > 0 else float("nan"))
        return {
            "achieved_bandwidth_GBs": achieved_GBs,
            "theoretical_bw_GBs": self._theoretical_bw_GBs,
            "bw_utilization_pct": utilization_pct,
            "total_bytes": total_bytes,
        }

    def set_theoretical_bw(self, bw_GBs: float) -> None:
        """Override theoretical bandwidth from hardware.yaml."""
        self._theoretical_bw_GBs = bw_GBs
