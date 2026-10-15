"""Multi-GPU (TP>1 / DP>1) guards for the PD-mux runtime -- stdlib only.

Two pure pieces, kept out of ``multiplexing_mixin.py`` so CPU tests can load
them without torch/sgl_kernel (same reason ``controller.py``/``profile.py`` are
standalone):

1. ``tp_pdmux_violations`` -- the M2 fail-fast predicate registered in
   ``reports/multigpu_scaling_design_2026-10-12.md`` sec 2-1 (rev2) after audit
   R-3 (``reports/audit/2026-10-12_multigpu_design/
   AUDIT_multigpu_design_rev1_2026-10-12.md``).  For ``tp_size > 1`` the ONLY
   allowed PD-mux mode is

       legacy loop  AND  PDMUX_R2_POLICY in (unset, "fixed")
                    AND  PDMUX_SLO_SCHED unset
                    AND  PDMUX_LA_COORD unset
                    AND  PDMUX_TRUE_DUAL_WORKER unset

   and everything else is refused at ``init_pdmux``.  WHY: every TP rank runs
   its own scheduler process and its own copy of the partition/admission
   decision.  A decision that reads wall clock or an ITL EWMA
   (``_r2_decide_idx`` with generic/hybrid, ``_slo_decide_idx``) or that reads
   ``arbiter.safe_to_switch()`` (true dual-worker -- timing dependent, so even
   ``fixed`` can diverge there) can pick a different stream group or a
   different admission on different ranks; the ranks then issue collectives in
   different orders or on different batches, which hangs or silently
   desynchronises.  Under the legacy loop ``FixedPolicy.decide`` only consults
   ``safe_boundary``, which that loop passes as the constant True, so legacy +
   fixed (and legacy + no R2 policy, i.e. the upstream decode-bs table) is a
   deterministic function of state that every rank shares.  PDMUX_LA_COORD
   selects a different event loop (``event_loop_pdmux_coord``, a documented
   negative result) whose per-layer stream choices were never analysed for TP.
   "Legacy loop" is therefore not a sixth independent switch: it is exactly
   LA_COORD unset (no coord loop, ``scheduler.py`` dispatch) AND
   TRUE_DUAL_WORKER unset (no worker threads), so it is not checked twice.

   "Unset" is LITERAL: absent or the empty string.  A value such as "0" is
   REFUSED even where the engine would parse it as off (PDMUX_TRUE_DUAL_WORKER
   only turns on for 1/true/True) -- PDMUX_SLO_SCHED and PDMUX_LA_COORD are
   read with plain truthiness, so "0" turns those ON, and a predicate that
   tried to mirror three different parsers would be one more thing to drift.
   The harness (``scripts/r2_eval/policy_adapter.sh``) ``unset``s these, so the
   literal reading costs nothing.  PDMUX_R2_POLICY is normalised exactly as
   ``init_pdmux`` does (``strip().lower()``).

   NOT IN THE PREDICATE (deliberately; the registered rule enumerates five
   conjuncts and this module implements exactly those):
   * PDMUX_STICKY_PARTITION -- registered as "sticky + fixed is checked
     separately in M2g".  Its own init already refuses every non-fixed policy.
   * PDMUX_DUAL_WORKER (R1 observer) -- legacy loop, observation only.
   * PDMUX_FIXED_DECODE_SM_FILE -- model-side per-layer pin (dead track); it is
     re-read from a file each step, so ranks could read different contents.
     Flagged in the M2 hand-off as an open item, not silently added here.

   DP (``dp_size > 1`` with ``tp_size == 1``) is NOT refused: under
   ``--dp-size`` without ``--enable-dp-attention`` every DP replica is its own
   scheduler process with its own TP group of size ``tp_size`` and its own
   ``gpu_id`` (``data_parallel_controller.launch_tensor_parallel_group``), and
   no PD-mux decision is ever combined across replicas (the only collective in
   the loop, ``tp_cpu_group.allreduce(flags)``, spans the replica's own TP
   group).  So a replica's decision cannot desynchronise another replica.
   ``--enable-dp-attention`` requires ``tp_size > 1`` (dp ranks share the TP
   group), so that configuration is caught by the TP predicate above.

2. ``rank_suffixed_path`` / ``rank_fields`` -- per-process output naming
   (design sec 4-1).  With ``tp_size > 1`` the path gets ``.tp{tp_rank}``, with
   ``dp_size > 1`` it gets ``.dp{dp_rank}`` (outer level first:
   ``tel.dp1.tp0.jsonl``), inserted before the last suffix exactly as the
   precedent in ``holb_probe.py``/``chunk_probe.py`` (``path.with_name(
   f"{stem}.tp{rank}{suffix}")``).  At ``tp_size == dp_size == 1`` the path is
   returned unchanged and ``rank_fields`` is EMPTY, so single-GPU telemetry is
   byte-identical to the pre-change schema (no new keys).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Optional

# The registered conjuncts (design rev2 sec 2-1 / audit R-3).  Order is the
# order the violations are reported in.
TP_FORBIDDEN_FLAGS = (
    "PDMUX_SLO_SCHED",
    "PDMUX_LA_COORD",
    "PDMUX_TRUE_DUAL_WORKER",
)
TP_ALLOWED_R2_POLICIES = ("", "fixed")


def _is_unset(env: Mapping[str, str], key: str) -> bool:
    return env.get(key, "") == ""


def tp_pdmux_violations(tp_size: int, env: Mapping[str, str]) -> List[str]:
    """Return why this (tp_size, env) PD-mux mode is refused; [] = allowed.

    Pure: reads only its arguments.  ``tp_size <= 1`` is always allowed (the
    single-GPU behaviour is unchanged by this guard).
    """
    if int(tp_size) <= 1:
        return []
    bad: List[str] = []
    policy = str(env.get("PDMUX_R2_POLICY", "")).strip().lower()
    if policy not in TP_ALLOWED_R2_POLICIES:
        bad.append(
            f"PDMUX_R2_POLICY={policy!r} (only unset or 'fixed' is allowed: "
            "generic/hybrid decide from wall clock / ITL EWMA per rank)"
        )
    for key in TP_FORBIDDEN_FLAGS:
        if not _is_unset(env, key):
            bad.append(f"{key}={env.get(key)!r} must be unset")
    # No separate "legacy loop" clause: it is exactly (LA_COORD unset AND
    # TRUE_DUAL_WORKER unset), already checked above.  A redundant clause would
    # make each of those two checks non-load-bearing, so removing one of them
    # (the mutation the M2 tests apply) would go undetected.
    return bad


def check_tp_pdmux_combination(tp_size: int, env: Mapping[str, str]) -> None:
    """Raise RuntimeError (fail-fast at init_pdmux) on a refused combination."""
    bad = tp_pdmux_violations(tp_size, env)
    if bad:
        raise RuntimeError(
            f"PD-mux with tp_size={int(tp_size)} is only defined for the legacy "
            "loop with PDMUX_R2_POLICY unset/fixed and PDMUX_SLO_SCHED, "
            "PDMUX_LA_COORD, PDMUX_TRUE_DUAL_WORKER unset (per-rank decisions "
            "would diverge and desynchronise TP collectives; "
            "reports/multigpu_scaling_design_2026-10-12.md sec 2-1). "
            "Refused because: " + "; ".join(bad)
        )


def _norm_rank(value: Optional[int]) -> int:
    return int(value or 0)


def rank_suffix(
    tp_size: int,
    tp_rank: Optional[int],
    dp_size: int = 1,
    dp_rank: Optional[int] = None,
) -> str:
    """'' at tp=dp=1; else '.dp{d}' (if dp>1) followed by '.tp{t}' (if tp>1)."""
    out = ""
    if int(dp_size or 1) > 1:
        out += f".dp{_norm_rank(dp_rank)}"
    if int(tp_size or 1) > 1:
        out += f".tp{_norm_rank(tp_rank)}"
    return out


def rank_suffixed_path(
    path: str,
    tp_size: int,
    tp_rank: Optional[int],
    dp_size: int = 1,
    dp_rank: Optional[int] = None,
) -> str:
    """Per-process output path.  Empty path stays empty (feature OFF).

    At tp=dp=1 the input string is returned UNCHANGED (not re-rendered through
    ``Path``), so even an unusual spelling such as a trailing slash or ``./``
    survives byte for byte.
    """
    if not path:
        return path
    suffix = rank_suffix(tp_size, tp_rank, dp_size, dp_rank)
    if not suffix:
        return path
    p = Path(path)
    return str(p.with_name(f"{p.stem}{suffix}{p.suffix}"))


def rank_fields(
    tp_size: int,
    tp_rank: Optional[int],
    dp_size: int = 1,
    dp_rank: Optional[int] = None,
) -> Dict[str, int]:
    """Fields added to every telemetry record; EMPTY at tp=dp=1.

    Both keys are emitted together as soon as either axis is > 1, so a
    multi-GPU file always carries the full (tp_rank, dp_rank) coordinate.
    """
    if int(tp_size or 1) <= 1 and int(dp_size or 1) <= 1:
        return {}
    return {"tp_rank": _norm_rank(tp_rank), "dp_rank": _norm_rank(dp_rank)}
