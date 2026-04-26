"""
libsmctrl wrapper for SM count control.

Provides two backends:
  1. libsmctrl (primary): kernel-level TPC mask via ioctl, μs-range overhead
  2. subprocess + CUDA MPS (fallback): CUDA_MPS_ACTIVE_THREAD_PERCENTAGE is set
     in each measurement *subprocess* before CUDA context creation. Setting this
     env-var in the already-running process is a no-op — CUDA ignores it.

libsmctrl source: https://github.com/msr-fiddle/libsmctrl
The shared library (libsmctrl.so) must be compiled and placed on LD_LIBRARY_PATH
or pointed to via the LIBSMCTRL_PATH environment variable.

TPC mask notes (Ampere/Hopper):
  libsmctrl operates at TPC (Texture Processing Cluster) granularity, NOT SM.
  Ampere (GA100/GA102): 2 SMs per TPC
  Hopper (GH100):       2 SMs per TPC
  Turing (TU102):       2 SMs per TPC
  To restrict to N SMs → set mask with ceil(N / sm_per_tpc) TPC bits.
  Override sm_per_tpc via constructor if your GPU differs.
"""

import ctypes
import os
import subprocess
import time
from typing import Optional

import torch


_LIBSMCTRL_NOT_FOUND_WARNED = False


class SMController:
    """Controls SM allocation for the current CUDA context via libsmctrl.

    SM control backend priority:
      1. libsmctrl shared library (set_sm_count/set_sm_ratio take effect immediately)
      2. CUDA MPS subprocess mode (each measurement run spawned as a child process
         with CUDA_MPS_ACTIVE_THREAD_PERCENTAGE pre-set; requires MPS daemon)
      3. No control — SM restriction silently disabled; verify_sm_control() returns False

    Call verify_sm_control() after construction to confirm SM restriction works.
    """

    def __init__(
        self,
        device_id: int = 0,
        total_sm_count: Optional[int] = None,
        sm_per_tpc: int = 2,
    ):
        """
        Args:
            device_id: CUDA device index.
            total_sm_count: Total SM count. Auto-detected from torch if None.
            sm_per_tpc: SMs per TPC (Ampere/Hopper/Turing = 2). Affects TPC mask
                        computation when using libsmctrl backend.
        """
        self.device_id = device_id
        self.sm_per_tpc = sm_per_tpc
        self._lib: Optional[ctypes.CDLL] = None
        self._available = False        # libsmctrl loaded successfully
        self._mps_available = False    # CUDA MPS daemon is running
        self._no_control = False       # neither backend works

        self.total_sm_count = total_sm_count or self._query_sm_count()
        self._tpc_count = max(1, self.total_sm_count // sm_per_tpc)

        self._try_load_libsmctrl()
        if not self._available:
            self._check_mps()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _query_sm_count(self) -> int:
        props = torch.cuda.get_device_properties(self.device_id)
        return props.multi_processor_count

    def _try_load_libsmctrl(self) -> None:
        lib_names = ["libsmctrl.so", "libsmctrl.so.1"]
        env_path = os.environ.get("LIBSMCTRL_PATH", "")
        if env_path:
            lib_names = [env_path] + lib_names

        for name in lib_names:
            try:
                lib = ctypes.CDLL(name)
                # Confirm the required symbol is present
                _ = lib.smctrl_set_mask_for_current_ctx
                lib.smctrl_set_mask_for_current_ctx.restype = ctypes.c_int
                lib.smctrl_set_mask_for_current_ctx.argtypes = [ctypes.c_uint64]
                lib.smctrl_get_mask_for_current_ctx.restype = ctypes.c_int
                lib.smctrl_get_mask_for_current_ctx.argtypes = [
                    ctypes.POINTER(ctypes.c_uint64)
                ]
                self._lib = lib
                self._available = True
                return
            except (OSError, AttributeError):
                continue

        # libsmctrl not found — do NOT set _mps_available here.
        # _check_mps() handles that separately.

    def _check_mps(self) -> None:
        """Detect whether a CUDA MPS daemon is running."""
        mps_pipe = os.environ.get("CUDA_MPS_PIPE_DIRECTORY", "/tmp/nvidia-mps")
        pipe_exists = os.path.exists(mps_pipe)

        if not pipe_exists:
            # Also try nvidia-smi check
            try:
                result = subprocess.run(
                    ["nvidia-smi", "mps"],
                    capture_output=True, timeout=3
                )
                pipe_exists = result.returncode == 0
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass

        self._mps_available = pipe_exists
        if not pipe_exists:
            self._no_control = True

    # ------------------------------------------------------------------
    # TPC mask computation (libsmctrl backend)
    # ------------------------------------------------------------------

    def _sm_count_to_tpc_mask(self, n_sm: int) -> int:
        """Convert SM count to a TPC bitmask for libsmctrl.

        libsmctrl takes a bitmask where bit i = 1 means TPC i is enabled.
        To use N SMs → enable ceil(N / sm_per_tpc) TPCs.

        Example (A100: 108 SM, 54 TPC, sm_per_tpc=2):
          n_sm=54 → n_tpc=27 → mask=0x7FFFFFF (27 bits set)
          n_sm=108 → n_tpc=54 → mask=0x3FFFFFFFFFFFFF (54 bits set)
        """
        n_sm = max(1, min(n_sm, self.total_sm_count))
        n_tpc = max(1, (n_sm + self.sm_per_tpc - 1) // self.sm_per_tpc)
        n_tpc = min(n_tpc, self._tpc_count)
        return (1 << n_tpc) - 1

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Return True if libsmctrl backend is loaded."""
        return self._available

    def get_backend_name(self) -> str:
        if self._available:
            return "libsmctrl"
        elif self._mps_available:
            return "mps_subprocess"
        else:
            return "none"

    def set_sm_count(self, n_sm: int) -> None:
        """Restrict GPU SM usage to approximately n_sm SMs.

        With libsmctrl backend: takes effect immediately for the next kernel.
        With MPS backend: has no effect on the *current* process (CUDA context
            already created). Use verify_sm_control() to check if this works.

        Args:
            n_sm: Target SM count. Clamped to [1, total_sm_count].
        """
        n_sm = max(1, min(n_sm, self.total_sm_count))
        if self._available and self._lib is not None:
            mask = self._sm_count_to_tpc_mask(n_sm)
            ret = self._lib.smctrl_set_mask_for_current_ctx(ctypes.c_uint64(mask))
            if ret != 0:
                raise RuntimeError(
                    f"smctrl_set_mask_for_current_ctx failed: {ret}. "
                    f"Ensure libsmctrl is compiled for your kernel version."
                )
        elif self._mps_available:
            # NOTE: Setting env-var in current process is a no-op if CUDA is already
            # initialized. Effective only if set BEFORE torch.cuda.* first call.
            # For subprocess mode, the caller must spawn a child with this env set.
            ratio_pct = int(n_sm / self.total_sm_count * 100)
            os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(ratio_pct)
        else:
            # No SM control — silently pass; verify_sm_control() will surface this
            pass

    def set_sm_ratio(self, ratio: float) -> None:
        """Restrict SMs by fractional ratio (0.0 – 1.0).

        Args:
            ratio: Fraction of total SMs. Clamped to (0, 1].
        """
        ratio = max(0.01, min(1.0, ratio))
        n_sm = max(1, round(ratio * self.total_sm_count))
        self.set_sm_count(n_sm)

    def reset(self) -> None:
        """Remove all SM restrictions."""
        if self._available and self._lib is not None:
            full_mask = self._sm_count_to_tpc_mask(self.total_sm_count)
            self._lib.smctrl_set_mask_for_current_ctx(ctypes.c_uint64(full_mask))
        elif self._mps_available:
            os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = "100"

    def verify_sm_control(self, verbose: bool = True) -> bool:
        """Probe whether SM restriction actually changes kernel latency.

        Runs a compute-bound kernel at full SM and at 25% SM, then checks
        whether the 25%-SM run is at least 20% slower. If not, SM control
        is not working.

        Returns:
            True if SM control appears functional, False otherwise.
        """
        import numpy as np

        if not torch.cuda.is_available():
            return False

        # Use a memory-bandwidth-bound kernel: large vector add
        # (BW-bound ops are sensitive to SM count)
        size = 64 * 1024 * 1024  # 64M elements, 256MB at fp32
        x = torch.ones(size, device="cuda", dtype=torch.float32)
        y = torch.ones(size, device="cuda", dtype=torch.float32)

        def _kernel():
            torch.add(x, y, out=x)

        n_iters = 20

        def _measure(sm_count: int) -> float:
            self.set_sm_count(sm_count)
            torch.cuda.synchronize()
            events = [(torch.cuda.Event(enable_timing=True),
                       torch.cuda.Event(enable_timing=True))
                      for _ in range(n_iters)]
            for s, e in events:
                s.record()
                _kernel()
                e.record()
            torch.cuda.synchronize()
            return float(np.median([s.elapsed_time(e) for s, e in events]))

        full_sm_lat = _measure(self.total_sm_count)
        quarter_sm = max(1, self.total_sm_count // 4)
        low_sm_lat = _measure(quarter_sm)
        self.reset()

        # Expect at least 20% slowdown with 4× fewer SMs
        ratio = low_sm_lat / full_sm_lat if full_sm_lat > 0 else 1.0
        works = ratio >= 1.20

        if verbose:
            backend = self.get_backend_name()
            print(f"  [SMController.verify] backend={backend}")
            print(f"    full SM ({self.total_sm_count} SM): {full_sm_lat:.3f} ms")
            print(f"    low SM  ({quarter_sm} SM):  {low_sm_lat:.3f} ms")
            print(f"    slowdown ratio: {ratio:.2f}x  →  SM control {'WORKS ✓' if works else 'NOT WORKING ✗'}")
            if not works:
                if backend == "none":
                    print(
                        "  ✗ Neither libsmctrl nor CUDA MPS is available.\n"
                        "    Build libsmctrl: https://github.com/msr-fiddle/libsmctrl\n"
                        "    and set LIBSMCTRL_PATH=<path>/libsmctrl.so\n"
                        "    OR start nvidia-cuda-mps-control and run each sweep as a\n"
                        "    subprocess with CUDA_MPS_ACTIVE_THREAD_PERCENTAGE set."
                    )
                elif backend == "mps_subprocess":
                    print(
                        "  ✗ MPS daemon found but setting CUDA_MPS_ACTIVE_THREAD_PERCENTAGE\n"
                        "    in the current process has no effect after CUDA is initialized.\n"
                        "    Use --subprocess-mode to spawn each measurement in a fresh process."
                    )
        return works

    def measure_reconfigure_latency_us(
        self,
        from_ratio: float,
        to_ratio: float,
        n_trials: int = 200,
    ) -> dict:
        """Measure SM reconfiguration latency in microseconds.

        Methodology (CPU-side, bracketed by CUDA sync):
          torch.cuda.synchronize()
          t0 = perf_counter_ns()
          set_sm_ratio(to_ratio)
          torch.cuda.synchronize()   # wait until first kernel under new mask is ready
          t1 = perf_counter_ns()
          overhead_us = (t1 - t0) / 1000

        Returns:
            dict: mean_us, p50_us, p99_us, min_us, max_us, std_us
        """
        import numpy as np

        dummy = torch.zeros(1, device="cuda")
        latencies_us = []

        for _ in range(n_trials):
            self.set_sm_ratio(from_ratio)
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
