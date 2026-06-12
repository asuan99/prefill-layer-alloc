"""
_green_ctx.py — two spatial Green Context partitions + concurrent timing for E4.

Spatial SM partitioning is what makes the (b) experiment work: a Green Context
splits SMs but NOT HBM bandwidth, so a bandwidth-heavy SSM prefill in one
partition still steals memory bandwidth from a decode in the other partition. The
A/B runner uses that to measure layer-type-dependent bandwidth interference.

Built on the src.smctrl.green_ctx_controller driver primitives (reused, not
modified). The logic mirrors the v1 coexec_microbench two-partition split, which
now lives in the frozen archive — it is reimplemented here on top of src so E4
does not import archived code.
"""

from __future__ import annotations

import ctypes
import os
import sys
from typing import Optional

_here = os.path.dirname(os.path.abspath(__file__))
_CHAR = os.path.abspath(os.path.join(_here, "..", ".."))
for _p in (_CHAR,):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

from src.smctrl.green_ctx_controller import (   # reuse src primitives
    _CUdevResource,
    _load_driver_lib,
    _CU_DEV_RESOURCE_TYPE_SM,
    _CU_GREEN_CTX_DEFAULT_STREAM,
    _CU_STREAM_NON_BLOCKING,
)


def _make_stream(lib, device_id, res) -> Optional["torch.cuda.Stream"]:
    desc = ctypes.c_void_p()
    if lib.cuDevResourceGenerateDesc(ctypes.byref(desc), ctypes.byref(res), 1) != 0:
        return None
    gctx = ctypes.c_void_p()
    if lib.cuGreenCtxCreate(ctypes.byref(gctx), desc, device_id,
                            _CU_GREEN_CTX_DEFAULT_STREAM) != 0:
        return None
    sp = ctypes.c_void_p()
    if lib.cuGreenCtxStreamCreate(ctypes.byref(sp), gctx,
                                  _CU_STREAM_NON_BLOCKING, 0) != 0:
        lib.cuGreenCtxDestroy(gctx)
        return None
    return torch.cuda.ExternalStream(sp.value, device=device_id)


def create_two_partitions(n_first_sm: int, device_id: int = 0):
    """Split the device into two spatially-exclusive Green Context partitions.

    Returns (stream_first, stream_second, info). On failure returns
    (None, None, {"error": ...}) so callers can fall back to two plain streams.
    """
    lib = _load_driver_lib()
    if lib is None:
        return None, None, {"error": "libcuda.so unavailable"}
    torch.cuda.init()

    base = _CUdevResource()
    if lib.cuDeviceGetDevResource(device_id, ctypes.byref(base),
                                  _CU_DEV_RESOURCE_TYPE_SM) != 0:
        return None, None, {"error": "cuDeviceGetDevResource failed"}

    first = _CUdevResource()
    remaining = _CUdevResource()
    nb = ctypes.c_uint(1)
    rc = lib.cuDevSmResourceSplitByCount(
        ctypes.byref(first), ctypes.byref(nb), ctypes.byref(base),
        ctypes.byref(remaining), 0, ctypes.c_uint(n_first_sm),
    )
    if rc != 0 or nb.value == 0:
        return None, None, {"error": f"split rc={rc}"}

    s1 = _make_stream(lib, device_id, first)
    s2 = _make_stream(lib, device_id, remaining)
    if s1 is None or s2 is None:
        return None, None, {"error": "green ctx stream creation failed"}
    return s1, s2, {
        "actual_first_sm": int(first._impl.smCount),
        "actual_second_sm": int(remaining._impl.smCount),
    }


def measure_concurrent(fn_a, fn_b, stream_a, stream_b, n_warmup=5, n_measure=20) -> dict:
    """Enqueue fn_a and fn_b on their streams together; measure real overlap.

    Both launch blocks are issued back-to-back before any sync, so the GPU sees
    both streams queued and can co-schedule them. Returns each stream's elapsed
    time and the concurrent wall time (max of the two). This is real concurrent
    execution — no simulation.
    """
    import statistics
    for _ in range(n_warmup):
        with torch.cuda.stream(stream_a):
            fn_a()
        with torch.cuda.stream(stream_b):
            fn_b()
        torch.cuda.synchronize()

    a_ms, b_ms, conc_ms = [], [], []
    for _ in range(n_measure):
        ea0, ea1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        eb0, eb1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(stream_a):
            ea0.record(stream_a); fn_a(); ea1.record(stream_a)
        with torch.cuda.stream(stream_b):
            eb0.record(stream_b); fn_b(); eb1.record(stream_b)
        torch.cuda.synchronize()
        ta, tb = ea0.elapsed_time(ea1), eb0.elapsed_time(eb1)
        a_ms.append(ta); b_ms.append(tb); conc_ms.append(max(ta, tb))
    return {
        "a_stream_ms": statistics.median(a_ms),
        "b_stream_ms": statistics.median(b_ms),
        "concurrent_ms": statistics.median(conc_ms),
    }


def time_solo(fn, n_warmup=5, n_measure=20) -> float:
    """Median solo latency (ms) on the default stream, full SM."""
    import statistics
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    lats = []
    for _ in range(n_measure):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record(); fn(); e1.record()
        torch.cuda.synchronize()
        lats.append(e0.elapsed_time(e1))
    return statistics.median(lats)
