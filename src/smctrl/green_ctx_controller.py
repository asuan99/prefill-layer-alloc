"""
CUDA Green Contexts SM partitioning controller.

Design:
  Preset SM counts are materialized as (GreenContext, ExternalStream) pairs at
  __init__. set_sm_count(n) does a nearest-preset lookup and swaps the active
  stream pointer — no driver call at transition time.

  Kernels must be launched inside `with torch.cuda.stream(smctrl.get_stream()):`
  to enforce SM limits. LayerRunner handles this automatically.

CUDA struct sizes (cuda.h, RESOURCE_ABI_VERSION=1):
  CUdevResource : 4 (type) + 92 (_padding) + 48 (union) = 144 bytes
  CUdevResourceDesc : typedef struct CUdevResourceDesc_st* → c_void_p (8 bytes)
"""

import bisect
import ctypes
import time
from typing import Optional, Sequence

import numpy as np
import torch


# ---------------------------------------------------------------------------
# CUDA driver constants
# ---------------------------------------------------------------------------

_CU_DEV_RESOURCE_TYPE_SM = 1
_CU_GREEN_CTX_DEFAULT_STREAM = 0x1
_CU_STREAM_NON_BLOCKING = 0x1  # required by cuGreenCtxStreamCreate

# RESOURCE_ABI_VERSION = 1 (from cuda.h)
# CUdevResource._internal_padding = 92 bytes
# CUdevResource union (_oversize) = RESOURCE_ABI_EXTERNAL_BYTES = 48 bytes
_RESOURCE_PADDING_BYTES = 92
_RESOURCE_UNION_BYTES = 48


# ---------------------------------------------------------------------------
# ctypes struct definitions — must match cuda.h exactly
# ---------------------------------------------------------------------------

class _CUdevSmUnion(ctypes.Union):
    _fields_ = [
        ("smCount", ctypes.c_uint),
        ("_oversize", ctypes.c_ubyte * _RESOURCE_UNION_BYTES),
    ]


class _CUdevResource(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("_padding", ctypes.c_ubyte * _RESOURCE_PADDING_BYTES),
        ("_impl", _CUdevSmUnion),
    ]


assert ctypes.sizeof(_CUdevResource) == 144, (
    f"CUdevResource size {ctypes.sizeof(_CUdevResource)} != 144 — "
    "cuda.h layout changed; update _RESOURCE_PADDING_BYTES or _RESOURCE_UNION_BYTES"
)


# ---------------------------------------------------------------------------
# Driver library loader
# ---------------------------------------------------------------------------

_LIB_NAMES = ("libcuda.so.1", "libcuda.so")

_REQUIRED_SYMBOLS = (
    "cuDeviceGetDevResource",
    "cuDevSmResourceSplitByCount",
    "cuDevResourceGenerateDesc",
    "cuGreenCtxCreate",
    "cuGreenCtxStreamCreate",
    "cuGreenCtxDestroy",
)


def _load_driver_lib() -> Optional[ctypes.CDLL]:
    for name in _LIB_NAMES:
        try:
            lib = ctypes.CDLL(name)
            for sym in _REQUIRED_SYMBOLS:
                getattr(lib, sym)  # raises AttributeError if symbol missing
            _bind_symbols(lib)
            return lib
        except (OSError, AttributeError):
            continue
    return None


def _bind_symbols(lib: ctypes.CDLL) -> None:
    # cuDeviceGetDevResource(CUdevice, CUdevResource*, CUdevResourceType)
    lib.cuDeviceGetDevResource.restype = ctypes.c_int
    lib.cuDeviceGetDevResource.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(_CUdevResource),
        ctypes.c_int,
    ]

    # cuDevSmResourceSplitByCount(result*, nbGroups*, input*, remaining*, flags, minCount)
    lib.cuDevSmResourceSplitByCount.restype = ctypes.c_int
    lib.cuDevSmResourceSplitByCount.argtypes = [
        ctypes.POINTER(_CUdevResource),  # result (array of nbGroups)
        ctypes.POINTER(ctypes.c_uint),   # nbGroups (in/out)
        ctypes.POINTER(_CUdevResource),  # input (const, treated as mutable ptr)
        ctypes.POINTER(_CUdevResource),  # remaining (NULL ok)
        ctypes.c_uint,                   # useFlags
        ctypes.c_uint,                   # minCount
    ]

    # cuDevResourceGenerateDesc(CUdevResourceDesc* phDesc, CUdevResource* resources, nbResources)
    lib.cuDevResourceGenerateDesc.restype = ctypes.c_int
    lib.cuDevResourceGenerateDesc.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(_CUdevResource),
        ctypes.c_uint,
    ]

    # cuGreenCtxCreate(CUgreenCtx* phCtx, CUdevResourceDesc desc, CUdevice, flags)
    # CUdevResourceDesc is a pointer type passed by value → c_void_p
    lib.cuGreenCtxCreate.restype = ctypes.c_int
    lib.cuGreenCtxCreate.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),  # phCtx
        ctypes.c_void_p,                  # desc (pointer value)
        ctypes.c_int,                     # dev
        ctypes.c_uint,                    # flags
    ]

    # cuGreenCtxStreamCreate(CUstream* phStream, CUgreenCtx, flags, priority)
    lib.cuGreenCtxStreamCreate.restype = ctypes.c_int
    lib.cuGreenCtxStreamCreate.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),  # phStream
        ctypes.c_void_p,                  # greenCtx
        ctypes.c_uint,                    # flags
        ctypes.c_int,                     # priority
    ]

    # cuGreenCtxDestroy(CUgreenCtx hCtx)
    lib.cuGreenCtxDestroy.restype = ctypes.c_int
    lib.cuGreenCtxDestroy.argtypes = [ctypes.c_void_p]


# ---------------------------------------------------------------------------
# SMController (Green Contexts backend)
# ---------------------------------------------------------------------------

class SMController:
    """SM partitioning via CUDA Green Contexts (CUDA driver 550+, toolkit 12.4+).

    New method:
        get_stream() -> torch.cuda.Stream
            Returns the ExternalStream bound to the current Green Context.
            Must be used as `with torch.cuda.stream(smctrl.get_stream()):`.

    Args:
        device_id: CUDA device index.
        total_sm_count: Total SM count. Auto-detected from torch if None.
        sm_per_tpc: SMs per TPC — kept for interface compatibility, unused.
        preset_sm_counts: Explicit SM counts to pre-create Green Contexts for.
            If None, contexts are created for 8 equally-spaced steps + full SM.
    """

    def __init__(
        self,
        device_id: int = 0,
        total_sm_count: Optional[int] = None,
        sm_per_tpc: int = 2,
        preset_sm_counts: Optional[Sequence[int]] = None,
    ):
        self.device_id = device_id
        self.sm_per_tpc = sm_per_tpc  # kept for interface compatibility

        self._lib: Optional[ctypes.CDLL] = _load_driver_lib()
        self._available = False

        self.total_sm_count = total_sm_count or self._query_sm_count()

        # Maps requested_sm_count → (green_ctx_ptr, ExternalStream)
        self._contexts: dict[int, tuple[int, torch.cuda.ExternalStream]] = {}
        # Maps requested_sm_count → actual_sm_count allocated by cuDevSmResourceSplitByCount.
        # Actual may differ from requested due to GPC-granular SM allocation on the hardware.
        self._actual_sm_counts: dict[int, int] = {}
        self._sorted_presets: list[int] = []
        self._current_sm_count: int = self.total_sm_count

        if self._lib is not None:
            self._init_contexts(preset_sm_counts)

        # Always register a fallback entry for full SM using the current stream
        if self.total_sm_count not in self._contexts:
            default_stream = torch.cuda.current_stream(self.device_id)
            self._contexts[self.total_sm_count] = (0, default_stream)

        self._sorted_presets = sorted(self._contexts.keys())

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _query_sm_count(self) -> int:
        return torch.cuda.get_device_properties(self.device_id).multi_processor_count

    def _default_preset_counts(self) -> list[int]:
        n = self.total_sm_count
        steps = []
        for i in range(1, 9):
            sm = max(1, round(n * i / 8))
            steps.append(sm)
        return sorted(set(steps))

    def _init_contexts(self, preset_sm_counts: Optional[Sequence[int]]) -> None:
        counts = list(preset_sm_counts) if preset_sm_counts else self._default_preset_counts()
        if self.total_sm_count not in counts:
            counts.append(self.total_sm_count)

        # Ensure primary context is initialized before querying device resources.
        torch.cuda.init()

        base_res = _CUdevResource()
        rc = self._lib.cuDeviceGetDevResource(
            self.device_id,
            ctypes.byref(base_res),
            _CU_DEV_RESOURCE_TYPE_SM,
        )
        if rc != 0:
            # CUDA_ERROR_NOT_SUPPORTED (801) on consumer GPUs that lack Green Context support.
            return

        for n_sm in counts:
            n_sm = max(1, min(n_sm, self.total_sm_count))
            ctx_ptr, stream, actual_sm = self._create_context(base_res, n_sm)
            if ctx_ptr is None:
                continue
            self._contexts[n_sm] = (ctx_ptr, stream)
            self._actual_sm_counts[n_sm] = actual_sm

        if self._contexts:
            self._available = True

    def _create_context(
        self,
        base_res: "_CUdevResource",
        n_sm: int,
    ) -> tuple[Optional[int], Optional[torch.cuda.ExternalStream], int]:
        """Create a Green Context targeting n_sm SMs.

        Returns (green_ctx_ptr, stream, actual_sm_count).
        actual_sm_count may exceed n_sm due to GPC-granular hardware allocation.
        Returns (None, None, 0) on failure.
        """
        lib = self._lib

        # Step 1: split base resource to get n_sm SMs.
        # cuDevSmResourceSplitByCount allocates at GPC granularity; actual smCount
        # may be larger than n_sm.
        split_res = _CUdevResource()
        nb_groups = ctypes.c_uint(1)
        rc = lib.cuDevSmResourceSplitByCount(
            ctypes.byref(split_res),
            ctypes.byref(nb_groups),
            ctypes.byref(base_res),
            None,                    # remaining — let it be absorbed into result
            0,                       # useFlags
            ctypes.c_uint(n_sm),     # minCount per group
        )
        if rc != 0 or nb_groups.value == 0:
            return None, None, 0

        actual_sm = int(split_res._impl.smCount)

        # Step 2: generate resource descriptor.
        desc = ctypes.c_void_p()
        rc = lib.cuDevResourceGenerateDesc(
            ctypes.byref(desc),
            ctypes.byref(split_res),
            1,
        )
        if rc != 0:
            return None, None, 0

        # Step 3: create Green Context.
        green_ctx = ctypes.c_void_p()
        rc = lib.cuGreenCtxCreate(
            ctypes.byref(green_ctx),
            desc,
            self.device_id,
            _CU_GREEN_CTX_DEFAULT_STREAM,
        )
        if rc != 0:
            return None, None, 0

        # Step 4: create stream bound to this Green Context.
        # CU_STREAM_NON_BLOCKING is mandatory per cuda.h documentation.
        stream_ptr = ctypes.c_void_p()
        rc = lib.cuGreenCtxStreamCreate(
            ctypes.byref(stream_ptr),
            green_ctx,
            _CU_STREAM_NON_BLOCKING,
            0,   # priority
        )
        if rc != 0:
            lib.cuGreenCtxDestroy(green_ctx)
            return None, None, 0

        ext_stream = torch.cuda.ExternalStream(stream_ptr.value, device=self.device_id)
        return green_ctx.value, ext_stream, actual_sm

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Return True if Green Contexts backend is active."""
        return self._available

    def get_backend_name(self) -> str:
        if self._available:
            return "green_ctx"
        return "none"

    def get_stream(self) -> torch.cuda.Stream:
        """Return the ExternalStream for the current SM preset.

        Must be used with `with torch.cuda.stream(smctrl.get_stream()):` to
        enforce SM limits on subsequent kernel launches.
        """
        _, stream = self._contexts.get(
            self._current_sm_count,
            (None, torch.cuda.current_stream(self.device_id)),
        )
        return stream

    def set_sm_count(self, n_sm: int) -> None:
        """Switch to the Green Context nearest to n_sm SMs.

        Finds the closest preset via bisect — O(log P) where P = #presets.
        No driver call; just updates self._current_sm_count.
        """
        n_sm = max(1, min(n_sm, self.total_sm_count))
        if not self._sorted_presets:
            return
        idx = bisect.bisect_left(self._sorted_presets, n_sm)
        if idx == 0:
            snapped = self._sorted_presets[0]
        elif idx == len(self._sorted_presets):
            snapped = self._sorted_presets[-1]
        else:
            lo = self._sorted_presets[idx - 1]
            hi = self._sorted_presets[idx]
            snapped = lo if (n_sm - lo) <= (hi - n_sm) else hi
        self._current_sm_count = snapped

    def set_sm_ratio(self, ratio: float) -> None:
        """Switch to the Green Context nearest to ratio × total_sm_count SMs."""
        ratio = max(0.01, min(1.0, ratio))
        n_sm = max(1, round(ratio * self.total_sm_count))
        self.set_sm_count(n_sm)

    def reset(self) -> None:
        """Switch back to the full-SM Green Context."""
        self.set_sm_count(self.total_sm_count)

    def verify_sm_control(self, verbose: bool = True) -> bool:
        """Check that SM restriction actually slows down a compute-bound kernel.

        Runs a large fp16 GEMM at full SM and at 25% SM, then checks
        whether the 25%-SM run is ≥2x slower.  A compute-bound kernel is used
        deliberately: bandwidth-bound workloads (e.g. elementwise ops) can
        saturate HBM with a fraction of SMs and will not show clear scaling.
        Confirmed: A100 GEMM at 25% SM gives ~4x slowdown when isolation works.

        Returns False if Green Contexts are not available.
        """
        if not self._available:
            if verbose:
                print(
                    f"  [SMController.verify] backend=none\n"
                    f"  ✗ Green Contexts unavailable.\n"
                    f"    GPU may not support Green Contexts (data-center GPUs only),\n"
                    f"    or CUDA driver < 550."
                )
            return False

        # fp16 GEMM 4096×4096×4096 — compute-bound on A100/H100
        M = N = K = 4096
        a = torch.randn(M, K, device=f"cuda:{self.device_id}", dtype=torch.float16)
        b = torch.randn(K, N, device=f"cuda:{self.device_id}", dtype=torch.float16)
        n_iters = 20

        def _measure(sm_count: int) -> float:
            self.set_sm_count(sm_count)
            stream = self.get_stream()
            events = [
                (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                for _ in range(n_iters)
            ]
            with torch.cuda.stream(stream):
                for s, e in events:
                    s.record(stream)
                    torch.mm(a, b)
                    e.record(stream)
            torch.cuda.synchronize(self.device_id)
            return float(np.median([s.elapsed_time(e) for s, e in events]))

        full_lat = _measure(self.total_sm_count)
        quarter_sm = max(1, self.total_sm_count // 4)
        low_lat = _measure(quarter_sm)
        self.reset()

        ratio = low_lat / full_lat if full_lat > 0 else 1.0
        works = ratio >= 2.0

        if verbose:
            self.set_sm_count(quarter_sm)
            preset_sm = self._current_sm_count
            actual_sm = self._actual_sm_counts.get(preset_sm, preset_sm)
            self.reset()

            print(f"  [SMController.verify] backend=green_ctx  kernel=fp16_gemm_{M}x{N}x{K}")
            print(f"    full SM (requested={self.total_sm_count}, actual={self._actual_sm_counts.get(self.total_sm_count, self.total_sm_count)}):  {full_lat:.3f} ms")
            print(f"    low  SM (requested≈{quarter_sm}, preset={preset_sm}, actual={actual_sm}):  {low_lat:.3f} ms")
            print(f"    slowdown ratio: {ratio:.2f}x  →  SM control {'WORKS ✓' if works else 'NOT WORKING ✗'}")
            if not works:
                print(
                    "  ✗ SM control not effective.\n"
                    "    Check: (1) GPU supports Green Contexts (A100/H100/H200),\n"
                    "           (2) MIG mode is disabled (nvidia-smi mig -e 0),\n"
                    "           (3) primary CUDA context initialized before SMController."
                )
        return works

    def measure_reconfigure_latency_us(
        self,
        from_ratio: float,
        to_ratio: float,
        n_trials: int = 200,
    ) -> dict:
        """Measure SM context-switch latency in microseconds.

        For Green Contexts the "reconfiguration" is a stream pointer swap
        (O(1) Python dict lookup) plus the cost of torch.cuda.synchronize()
        to flush any in-flight work before the new context takes over.

        Methodology:
          synchronize → t0 → set_sm_ratio → synchronize → t1
        """
        dummy = torch.zeros(1, device=f"cuda:{self.device_id}")
        latencies_us = []

        for _ in range(n_trials):
            self.set_sm_ratio(from_ratio)
            with torch.cuda.stream(self.get_stream()):
                dummy.add_(1.0)
            torch.cuda.synchronize(self.device_id)

            t0 = time.perf_counter_ns()
            self.set_sm_ratio(to_ratio)
            torch.cuda.synchronize(self.device_id)
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

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __del__(self) -> None:
        if self._lib is None:
            return
        for ctx_ptr, _ in self._contexts.values():
            if ctx_ptr:
                try:
                    self._lib.cuGreenCtxDestroy(ctypes.c_void_p(ctx_ptr))
                except Exception:
                    pass
