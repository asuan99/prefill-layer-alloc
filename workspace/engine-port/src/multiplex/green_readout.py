"""Green-context read-out for the PD-mux stream groups (A1 rev2, 절차 2).

WHY THIS EXISTS
---------------
`DESIGN_A1_REV2_STICKY_2026-08-25.md` §3.1 registers the decision channel for the
`green` axis: the question "did decode actually run on a green context of D SM?"
must be answered **inside the server process, by the driver**, and must NOT be
answered by nsys (that would make the axis circular with Q2ae) nor by
`results/smid_census/smid_l0_census.driver_readout` -- the latter returns the
CURRENT context's `primary_sm` (108 on this device), so comparing it to a green
half's target reports `mismatched` unconditionally, and its own docstring
registers it as "DESCRIPTIVE / TARGET LAYER ONLY -- never an input to a verdict".
Nor by `results/s8_frontier/...realizedprobe`, which runs in a separate process
after the server is killed (that is how the Stage 0 D108 error happened).

So the green context handle is taken FROM THE STREAM and asked for ITS OWN SM
resource:  `cuStreamGetGreenCtx` -> `cuGreenCtxGetDevResource(..., SM)`.
This mirrors `results/kernel_mech/stage0ppp/stage0ppp_a0_probe.py:132-171`
(`_green_sm_readout`), the channel A0 registered and ran.

DEFAULT OFF.  Without `PDMUX_GREEN_READOUT=1` `maybe_read_green_contexts` returns
None before touching ctypes, so a run with the flag unset behaves exactly as it
did before this module existed.  Nothing here feeds scheduling, partition
selection, batching or admission: it is a one-shot startup observation.

FAILURE CONTRACT (per field, no blanket promise -- same shape as
`smid_l0_census.driver_readout`): a driver load failure returns a dict carrying
`error` and no per-stream entries; a per-stream call that raises leaves that
stream's entry carrying `error` and no `green_sm`.  A consumer must treat every
such entry as "not answered", never as "not green".
"""
from __future__ import annotations

import ctypes
import os
from typing import Any, Dict, Optional, Sequence, Tuple

# CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM (cuda.h).  Same constant as
# results/smid_census/smid_l0_census.py:538.
CU_DEV_RESOURCE_TYPE_SM = 1

ENV_FLAG = "PDMUX_GREEN_READOUT"
ENV_PATH = "PDMUX_GREEN_READOUT_PATH"


class CUdevResource(ctypes.Structure):
    """Layout copied verbatim from `smid_l0_census._CUdevResource` (:541-549).

    The union payload is read as three `unsigned int` (smCount,
    minSmPartitionSize, smCoscheduledAlignment); the padding keeps the offset
    right without spelling out the rest of the union.
    """

    _fields_ = [
        ("type", ctypes.c_int),
        ("_pad", ctypes.c_ubyte * 92),
        ("_u", ctypes.c_ubyte * 48),
    ]

    def sm_triplet(self) -> Dict[str, int]:
        vals = (ctypes.c_uint * 3).from_buffer_copy(bytes(self._u)[:12])
        return dict(
            smCount=int(vals[0]),
            minSmPartitionSize=int(vals[1]),
            smCoscheduledAlignment=int(vals[2]),
        )


def green_sm_readout(stream_ptr: int, lib: Any = None) -> Dict[str, Any]:
    """Ask the driver what green context (if any) `stream_ptr` is attached to.

    `lib` is injectable ONLY so the producer can be exercised on a CPU node.
    ★This is the 2026-08-21 audit item R2 lesson made structural: before it,
    `driver_readout` had zero direct checks of its producer, so a producer that
    answered THE SAME WAY FOR EVERY STREAM -- either way -- was invisible to the
    whole battery.  `tests/test_green_readout.py` therefore drives this with a
    fake that answers DIFFERENTLY per stream, and asserts that a constant-answer
    fake is detected.
    """
    out: Dict[str, Any] = {
        "channel": "cuStreamGetGreenCtx + cuGreenCtxGetDevResource",
    }
    if lib is None:
        try:
            lib = ctypes.CDLL("libcuda.so.1")
        except OSError as exc:
            out["error"] = f"libcuda load failed: {exc}"
            return out
    try:
        g = ctypes.c_void_p()
        out["cuStreamGetGreenCtx_rc"] = lib.cuStreamGetGreenCtx(
            ctypes.c_void_p(stream_ptr), ctypes.byref(g)
        )
        out["green_ctx_is_null"] = g.value in (None, 0)
        if out["green_ctx_is_null"] or out["cuStreamGetGreenCtx_rc"] != 0:
            return out
        res = CUdevResource()
        rc = lib.cuGreenCtxGetDevResource(
            g, ctypes.byref(res), ctypes.c_int(CU_DEV_RESOURCE_TYPE_SM)
        )
        out["cuGreenCtxGetDevResource_rc"] = rc
        out["green_sm"] = res.sm_triplet() if rc == 0 else None
    except Exception as exc:  # noqa: BLE001 - diagnostic path, never fatal
        out["error"] = repr(exc)
    return out


def read_stream_groups(
    stream_groups: Sequence[Tuple[Any, Any]],
    sm_counts: Sequence[Tuple[int, int]],
    sticky_idx: Optional[int] = None,
    lib: Any = None,
) -> Dict[str, Any]:
    """One record per stream-group index: what the ENGINE targeted vs what the
    DRIVER says that group's streams actually are.

    `target_*_sm` comes from `sm_counts` (the engine's own table) and
    `green_sm.smCount` from the driver, so a consumer can compare them WITHOUT
    either number having been derived from the other (methodology gate: an
    identity is not evidence).  No verdict is computed here -- this module
    reports, the rule scores.
    """
    groups = []
    for idx, pair in enumerate(stream_groups):
        target = sm_counts[idx] if idx < len(sm_counts) else (None, None)
        row: Dict[str, Any] = {
            "stream_index": idx,
            "target_prefill_sm": target[0],
            "target_decode_sm": target[1],
            "is_sticky_target": (sticky_idx is not None and idx == sticky_idx),
        }
        for role, stream in (("prefill", pair[0]), ("decode", pair[1])):
            ptr = getattr(stream, "cuda_stream", None)
            if ptr is None:
                row[role] = {"error": "stream has no cuda_stream attribute"}
                continue
            row[role] = green_sm_readout(int(ptr), lib=lib)
            # The engine-side stream identity.  ★It is NOT comparable to an nsys
            # `streamId` (report-local) -- see DESIGN_A1_REV2 §3.3; it is
            # recorded so two records from the SAME boot can be told apart.
            row[role]["cuda_stream_ptr"] = int(ptr)
        groups.append(row)
    return {
        "probe": "pdmux_green_readout",
        "note": "engine-side driver read-out; nsys is deliberately NOT consulted",
        "sticky_target_index": sticky_idx,
        "groups": groups,
    }


def maybe_read_green_contexts(
    stream_groups: Sequence[Tuple[Any, Any]],
    sm_counts: Sequence[Tuple[int, int]],
    sticky_idx: Optional[int] = None,
    env: Optional[Dict[str, str]] = None,
    lib: Any = None,
) -> Optional[Dict[str, Any]]:
    """Return the read-out iff `PDMUX_GREEN_READOUT` is set, else None.

    Returning None BEFORE importing/using ctypes is what makes the default-OFF
    guarantee cheap to verify: with the flag unset this function touches nothing
    but a dict lookup.
    """
    env = os.environ if env is None else env
    if env.get(ENV_FLAG, "0") not in ("1", "true", "True"):
        return None
    return read_stream_groups(stream_groups, sm_counts, sticky_idx=sticky_idx, lib=lib)
