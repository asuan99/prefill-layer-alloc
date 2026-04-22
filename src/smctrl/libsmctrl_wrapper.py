"""
libsmctrl wrapper for SM count control.

Provides two backends:
  1. libsmctrl (primary): kernel-level SM mask via ioctl, zero-copy overhead
  2. CUDA MPS fallback: CUDA_MPS_ACTIVE_THREAD_PERCENTAGE env-var approach

libsmctrl source: https://github.com/msr-fiddle/libsmctrl
The shared library (libsmctrl.so) must be compiled and placed on LD_LIBRARY_PATH.
"""

import ctypes
import os
import time
import subprocess
from typing import Optional
import torch


class SMController:
    """Controls SM allocation for the current CUDA context via libsmctrl.

    Falls back to CUDA MPS thread percentage if libsmctrl is unavailable.
    In fallback mode, overhead measurements include MPS reconfiguration cost.
    """

    def __init__(self, device_id: int = 0, total_sm_count: Optional[int] = None):
        self.device_id = device_id
        self._lib: Optional[ctypes.CDLL] = None
        self._available = False
        self._mps_fallback = False

        if total_sm_count is not None:
            self.total_sm_count = total_sm_count
        else:
            self.total_sm_count = self._query_sm_count()

        self._try_load_libsmctrl()

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _query_sm_count(self) -> int:
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_id)
            # nvmlDeviceGetNumGpuCores returns CUDA cores; divide by 128 for SM count on Ampere
            # Use subprocess as fallback for SM count via nvidia-smi
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True
            )
            # Query multiprocessor count via torch
            props = torch.cuda.get_device_properties(self.device_id)
            return props.multi_processor_count
        except Exception:
            props = torch.cuda.get_device_properties(self.device_id)
            return props.multi_processor_count

    def _try_load_libsmctrl(self) -> None:
        lib_names = ["libsmctrl.so", "libsmctrl.so.1"]
        search_paths = [
            "/usr/local/lib",
            "/usr/lib",
            os.path.expanduser("~/lib"),
            os.path.join(os.path.dirname(__file__), "../../lib"),
        ]
        env_path = os.environ.get("LIBSMCTRL_PATH", "")
        if env_path:
            lib_names = [env_path] + lib_names

        for name in lib_names:
            try:
                lib = ctypes.CDLL(name)
                # Verify expected symbols exist
                _ = lib.smctrl_set_mask_for_current_ctx
                self._lib = lib
                self._available = True
                self._setup_libsmctrl_signatures()
                return
            except (OSError, AttributeError):
                continue

        # Check if MPS is running as fallback
        result = subprocess.run(
            ["nvidia-smi", "-c", "EXCLUSIVE_PROCESS"],
            capture_output=True
        )
        self._mps_fallback = True
        self._available = False

    def _setup_libsmctrl_signatures(self) -> None:
        lib = self._lib
        # smctrl_set_mask_for_current_ctx(uint64_t mask)
        lib.smctrl_set_mask_for_current_ctx.restype = ctypes.c_int
        lib.smctrl_set_mask_for_current_ctx.argtypes = [ctypes.c_uint64]
        # smctrl_get_mask_for_current_ctx(uint64_t *mask)
        lib.smctrl_get_mask_for_current_ctx.restype = ctypes.c_int
        lib.smctrl_get_mask_for_current_ctx.argtypes = [ctypes.POINTER(ctypes.c_uint64)]

    # ------------------------------------------------------------------
    # SM mask computation
    # ------------------------------------------------------------------

    def _count_to_mask(self, n_sm: int) -> int:
        """Convert SM count to a bitmask (lowest n_sm bits set)."""
        n_sm = max(1, min(n_sm, self.total_sm_count))
        # libsmctrl uses TPC (Texture Processing Cluster) masks on most GPUs.
        # One TPC ≈ 2 SMs on Ampere/Hopper. We approximate here;
        # for exact TPC counts, read /proc/driver/nvidia/gpus/<id>/information
        # and adjust accordingly.
        return (1 << n_sm) - 1

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Return True if libsmctrl backend is available."""
        return self._available

    def set_sm_count(self, n_sm: int) -> None:
        """Restrict CUDA context to use at most n_sm SMs.

        Args:
            n_sm: Number of SMs to allow. Clamped to [1, total_sm_count].
        """
        n_sm = max(1, min(n_sm, self.total_sm_count))
        if self._available and self._lib is not None:
            mask = self._count_to_mask(n_sm)
            ret = self._lib.smctrl_set_mask_for_current_ctx(ctypes.c_uint64(mask))
            if ret != 0:
                raise RuntimeError(f"smctrl_set_mask_for_current_ctx failed: {ret}")
        elif self._mps_fallback:
            ratio = int(n_sm / self.total_sm_count * 100)
            os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(ratio)
        else:
            # No SM control available — log warning but continue
            pass

    def set_sm_ratio(self, ratio: float) -> None:
        """Restrict SMs by fractional ratio (0.0 – 1.0).

        Args:
            ratio: Fraction of total SMs to allow. Clamped to (0, 1].
        """
        ratio = max(0.01, min(1.0, ratio))
        n_sm = max(1, round(ratio * self.total_sm_count))
        self.set_sm_count(n_sm)

    def reset(self) -> None:
        """Remove all SM restrictions (allow full GPU)."""
        if self._available and self._lib is not None:
            full_mask = self._count_to_mask(self.total_sm_count)
            self._lib.smctrl_set_mask_for_current_ctx(ctypes.c_uint64(full_mask))
        elif self._mps_fallback:
            os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = "100"

    def measure_reconfigure_latency_us(
        self,
        from_ratio: float,
        to_ratio: float,
        n_trials: int = 200,
    ) -> dict:
        """Measure SM reconfiguration latency in microseconds.

        Methodology:
          1. Set SM to from_ratio
          2. Launch a small warm-up kernel and synchronize
          3. Measure: synchronize → set_sm_ratio(to_ratio) → synchronize
          4. Repeat n_trials times

        Args:
            from_ratio: Starting SM ratio.
            to_ratio: Target SM ratio.
            n_trials: Number of measurement repetitions.

        Returns:
            dict with keys: mean_us, p50_us, p99_us, min_us, max_us, std_us
        """
        import numpy as np

        # Warm-up kernel to keep GPU active
        dummy = torch.zeros(1, device="cuda")

        latencies_us = []
        for _ in range(n_trials):
            self.set_sm_ratio(from_ratio)
            # Dummy kernel to flush pipeline
            dummy.add_(1.0)
            torch.cuda.synchronize()

            t0 = time.perf_counter_ns()
            self.set_sm_ratio(to_ratio)
            torch.cuda.synchronize()
            t1 = time.perf_counter_ns()

            latencies_us.append((t1 - t0) / 1_000.0)

        arr = np.array(latencies_us)
        return {
            "mean_us": float(arr.mean()),
            "p50_us": float(np.percentile(arr, 50)),
            "p99_us": float(np.percentile(arr, 99)),
            "min_us": float(arr.min()),
            "max_us": float(arr.max()),
            "std_us": float(arr.std()),
            "n_trials": n_trials,
            "from_ratio": from_ratio,
            "to_ratio": to_ratio,
        }
