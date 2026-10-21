#!/usr/bin/env python3
"""Scorers for the split GPU gate of the engine bundle (I-4 + I-5 + I-6 + causal_conv1d int64).

Design: results/engine_gate/GATE_SPLIT_DESIGN_2026-10-20.md (G-alpha -> G-beta -> G-gamma-A / G-gamma-F).
STATUS: CPU-tested only (tests/test_engine_gate_checks.py).  NOTHING HERE HAS RUN ON GPU DATA; writing a
scorer is not a PASS of anything.

Pure functions over the JSONL the probes write (no torch, no sglang).  Every threshold is in REG below;
REG is registered BEFORE any gate Job data exists (its sha256 is printed by `--print-reg` and recorded in
the design document).  Verdicts: PASS | FAIL | UNRESOLVED (never silently PASS on missing data).

  check_i4_pdmux    (c)-I4 on a PD-mux boot: per decode_iteration, I-4 GPU time is contained in the
                    host bracket [t_host_start, t_sync]; aggregate ratio in the band; and the
                    host-time MUTANT (decode_fwd_gpu_ms := decode_fwd_host_ms, which is exactly what a
                    probe timing the call with the host clock would emit) must FAIL the band on the
                    same data -- if it does not, the cell is UNRESOLVED (non-discriminating, lesson 53).
  check_i4_fused    (c)-I4 on a vanilla event_loop_normal boot (overlap OFF: one host sync per
                    iteration): reference = interval to the next decode forward's t_host_start.
  check_i6          I-6: hit records padded to the boot capture list (independent bisect), eager
                    records unpadded, >= 1 hit with real padding (else the bs_padded := bs_real mutant
                    would pass -> UNRESOLVED), eager-reason control for a --disable-cuda-graph boot.
  check_emission    every arm's ON boot emits decode_forward with I-4 resolved (vanilla path included,
                    PREREG cost-model rev2 sec 1-7) and I-5 admission windows.
  outputs_identical strict token identity (sequential greedy passes, (a) s-pass and (b) int64).
  tost_log_ratio    (d) paired TOST on log(ON/OFF), 90 % CI inside +-ln(1.015).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

PASS, FAIL, UNRESOLVED = "PASS", "FAIL", "UNRESOLVED"

# ----------------------------------------------------------------- registered constants
REG: Dict[str, Any] = {
    "reg_version": "engine-gate-reg/v3 (2026-10-20, r3 after G-beta round 1; decision 20)",
    # (c)-I4 PD-mux: ratio = sum(gpu) / sum(t_sync - t_host_start) over the isolated-decode window.
    # SPEC sec 9 (c) states +-5 % for the prefill-span check; the decode bracket also holds sampling
    # and host bookkeeping, so the registered band is one-sided [0.90, 1.00] [제안] (deviation disclosed).
    "i4_pdmux_band": [0.90, 1.0],
    # (c)-I4 vanilla (event_loop_normal): bracket = forward start -> next forward start (adds the
    # scheduler's own host work), so a wider lower bound [제안].
    "i4_fused_band": [0.75, 1.0],
    # per-record containment slack: GPU and host clocks are different clocks
    "i4_containment_rel": 0.001,
    "i4_containment_abs_ms": 0.05,
    "i4_containment_max_violation_frac": 0.0,
    "i4_min_records": 200,
    # emission (every ON boot): resolved I-4 fraction among decode forwards
    "emission_min_resolved_frac": 0.99,
    "emission_min_decode_forwards": 50,
    # I-6
    "i6_min_padded_hits": 1,
    # (d) TOST.  Audit r1 MAJOR 3 (2026-10-20): alpha is SPLIT over at most two looks (initial n<=10, one pre-registered
    # extension to n=16): each look uses alpha = 0.025 one-sided per side (95 % two-sided CI).  This replaces the
    # earlier single 90 % CI.  Pairs are never added after seeing a look other than this one extension.
    "tost_delta_log": math.log(1.015),
    "tost_alpha_one_sided": 0.025,
    "tost_min_pairs": 8,
    "tost_initial_pairs": 10,
    "tost_extended_pairs": 16,
    "tost_verdicts": ["PASS", "OBSERVER_EFFECT", "UNRESOLVED_POWER", "UNRESOLVED"],
    # audit r1 MAJOR 4d: ON-side invalid pairs exceeding OFF-side invalid pairs by >= this => OBSERVER_EFFECT
    "invalid_asym_min_diff": 2,
    # audit r1 MAJOR 1d: the fused (event_loop_normal) I-4 window = first to last decode forward of the isolated-decode phase
    "i4_fused_window": "first_to_last_decode_forward_of_isolated_decode_phase",
    # audit r1 4g: I-5 / I-6 reconciliation (stair trajectory 1..40 all visited; 49th request recorded as a cap stop)
    "i5i6_cap": 48,
    "i5i6_stair_bs": [1, 40],
    # r2 (2026-10-20, AFTER seeing the G-alpha round-1 B2 data -- disclosed): PD-mux (agnostic) did not reach bs 40 in run 1 (original wording "split prefill admits one at a time" is WRONG, see next lines)
    # ... [re-audit r2 correction 2026-10-20: the run-1 B2 data show request 0 (out 16) decoding alone at bs=1 for 15 steps, then the
    # other 39 joining in ONE extend (bs 1 -> 39); bs 40 was missed by a RACE between request 0's finish and the batched prefill,
    # not by one-at-a-time joining -- the rerun may give M = 39 or 40; not a PD-mux invariant].  PD-mux stair criterion =
    # visited decode bs_real set is contiguous 1..M AND M == max(I-5 decode_bs_hist over the windows whose midpoint is inside the stair window)
    # AND M >= i5i6_pdmux_stair_min_m [proposal].  Non-vacuous: a serialised join (one request at a time) would give M ~ 1-4,
    # and the floor 32 = 80 % of the 40 offered; the I-5 equality is what binds M to the engine's own count.
    # fused / Llama keep the 1..40 criterion unchanged.
    "i5i6_pdmux_stair_min_m": 32,
    # r2 diagnostic only (never changes a verdict): LO-cell stall fraction (token ITL > stall_ms) within 5 % +- 1 pp => STALL_BOUNDARY
    "stall_ms": 30.0,
    "stall_boundary_frac": 0.05,
    "stall_boundary_halfwidth": 0.01,
    # r3 (2026-10-20, AFTER seeing the G-beta round-1 data -- disclosed; decision 20, user-approved after the harness-audit cap):
    # beta-0 unit criterion = sha256(output) vs the ORIGINAL kernel's contiguous run at the same (dim, T); the torch-reference atol is
    # descriptive only (round 1: max diff 0.04-0.057 at every valid T = bf16 rounding of an fp32 reference).  MATCH = equal sha + exact
    # conv state; BROKEN = crash | state mismatch | sha differs.  chunk_offset-only mutant: ref = PATCHED contig.
    "beta0_criterion": "out_sha256 == pristine contig sha (same dim,T); max_abs_diff descriptive",
    # TP2 per-rank in_proj (Nano-9B-v2: d_inner 10240 -> 5120, 8 groups -> 4, 128 heads -> 64): width 11328, xBC offset 5120, dim 6144
    "beta0_tp2_geometry": {"width": 11328, "offset": 5120, "dim": 6144},
    "beta0_int32_boundary_tokens": {"22656": 94786, "11328": 189573},
    # non-blocking crash boot: death AND 'illegal memory access' only; the traceback conv file is required in the BLOCKING boot only
    "beta2_nonblocking_signature": "died and illegal-memory-access",
    # gamma: OFF/OFF diagnostic pairs are not in the plan (extension-Job option, default 0)
    "gamma_oo_pairs_default": 0,
}

# one-sided 0.95 quantile of Student t, df 1..30 (two TOST one-sided 5 % tests = 90 % CI)
_T95 = [6.314, 2.920, 2.353, 2.132, 2.015, 1.943, 1.895, 1.860, 1.833, 1.812, 1.796, 1.782, 1.771, 1.761,
        1.753, 1.746, 1.740, 1.734, 1.729, 1.725, 1.721, 1.717, 1.714, 1.711, 1.708, 1.706, 1.703, 1.701,
        1.699, 1.697]


def reg_sha256() -> str:
    return hashlib.sha256(json.dumps(REG, sort_keys=True).encode()).hexdigest()


def load_jsonl(path: Path, event: Optional[str] = None) -> List[Dict[str, Any]]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if event is None or r.get("event") == event:
            rows.append(r)
    return rows


def host_mutant(records: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """What a probe that timed ModelRunner.forward with the HOST clock would have written."""
    out = []
    for r in records:
        m = dict(r)
        m["decode_fwd_gpu_ms"] = r.get("decode_fwd_host_ms")
        out.append(m)
    return out


def pad_real_mutant(records: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """What a probe that wrote bs_padded := bs_real would have written."""
    out = []
    for r in records:
        m = dict(r)
        m["bs_padded"] = r.get("bs_real")
        out.append(m)
    return out


def _decode_only(records):
    return [r for r in records if r.get("mode") == "decode"]


def _in_band(x, band):
    return x is not None and band[0] <= x <= band[1]


# ----------------------------------------------------------------- (c)-I4
def i4_pdmux_pairs(decode_records, phase_records, field="decode_fwd_gpu_ms"):
    """Join each PD-mux decode_iteration (t_launch, t_sync) with the LAST decode forward whose
    t_host_start lies in (previous t_sync, t_launch].  Returns [(gpu_ms, ref_ms)]."""
    fw = sorted((r for r in _decode_only(decode_records) if r.get(field) is not None),
                key=lambda r: r["t_host_start"])
    it = sorted((r for r in phase_records if r.get("event", "decode_iteration") == "decode_iteration"),
                key=lambda r: r["t_launch"])
    pairs, j, prev_sync = [], 0, float("-inf")
    for ph in it:
        cand = None
        while j < len(fw) and fw[j]["t_host_start"] <= ph["t_launch"]:
            if fw[j]["t_host_start"] > prev_sync:
                cand = fw[j]
            j += 1
        if cand is not None:
            pairs.append((float(cand[field]), (float(ph["t_sync"]) - float(cand["t_host_start"])) * 1000.0))
        prev_sync = float(ph["t_sync"])
    return pairs


def i4_fused_pairs(decode_records, field="decode_fwd_gpu_ms"):
    fw = sorted((r for r in _decode_only(decode_records) if r.get(field) is not None),
                key=lambda r: r["t_host_start"])
    return [(float(a[field]), (float(b["t_host_start"]) - float(a["t_host_start"])) * 1000.0)
            for a, b in zip(fw, fw[1:])]


def _score_pairs(pairs, band):
    if len(pairs) < REG["i4_min_records"]:
        return {"verdict": UNRESOLVED, "why": f"{len(pairs)} joined records < {REG['i4_min_records']}",
                "n": len(pairs)}
    viol = sum(1 for g, ref in pairs
               if g > ref * (1 + REG["i4_containment_rel"]) + REG["i4_containment_abs_ms"])
    ratio = sum(g for g, _ in pairs) / sum(ref for _, ref in pairs)
    ok = (viol / len(pairs) <= REG["i4_containment_max_violation_frac"]) and _in_band(ratio, band)
    return {"verdict": PASS if ok else FAIL, "n": len(pairs), "ratio": ratio,
            "containment_violations": viol}


def _with_mutant(res, mres):
    res = dict(res)
    res["host_mutant"] = mres
    if res["verdict"] == PASS and mres.get("verdict") != FAIL:
        res.update(verdict=UNRESOLVED,
                   why="host-time mutant also passes on these data: the cell does not discriminate")
    return res


def check_i4_pdmux(decode_records, phase_records):
    band = REG["i4_pdmux_band"]
    res = _score_pairs(i4_pdmux_pairs(decode_records, phase_records), band)
    if res["verdict"] == UNRESOLVED:
        return res
    mres = _score_pairs(i4_pdmux_pairs(host_mutant(decode_records), phase_records), band)
    return _with_mutant(res, mres)


def decode_window(decode_records, t_lo, t_hi):
    """Registered fused window (audit r1 MAJOR 1d): first decode forward with t_host_start >= t_lo to the last one
    <= t_hi.  Returns (a, b) or None when the phase has no decode forward."""
    ts = sorted(float(r["t_host_start"]) for r in _decode_only(decode_records)
                if t_lo <= float(r["t_host_start"]) <= t_hi)
    return (ts[0], ts[-1]) if ts else None


def check_i4_fused(decode_records, window):
    """window = (t_lo, t_hi) of the isolated-decode phase.  Consecutive forwards are paired ONLY inside
    [first decode forward >= t_lo, last decode forward <= t_hi]: phase gaps and the s-pass bs=1 stretch are excluded
    by construction.  window=None is refused (UNRESOLVED): the window is part of the registration."""
    if window is None:
        return {"verdict": UNRESOLVED, "why": "fused I-4 window not provided (registered)"}
    w = decode_window(decode_records, float(window[0]), float(window[1]))
    if w is None:
        return {"verdict": UNRESOLVED, "why": "no decode forward in the isolated-decode phase"}
    recs = [r for r in decode_records if r.get("mode") == "decode" and w[0] <= float(r["t_host_start"]) <= w[1]]
    band = REG["i4_fused_band"]
    res = _score_pairs(i4_fused_pairs(recs), band)
    res["window"] = list(w)
    if res["verdict"] == UNRESOLVED:
        return res
    return _with_mutant(res, _score_pairs(i4_fused_pairs(host_mutant(recs)), band))


# ----------------------------------------------------------------- I-6
def expected_padded_bs(capture_bs, raw_bs, disable_padding=False):
    """Independent copy of the engine rule (cuda_graph_runner.replay_prepare bisect_left)."""
    cb = sorted(int(b) for b in (capture_bs or ()))
    if not cb or raw_bs <= 0:
        return None
    if disable_padding:
        return raw_bs if raw_bs in cb else None
    for b in cb:
        if b >= raw_bs:
            return b
    return None


def check_i6(decode_records, boot, expect_all_eager_reason: Optional[str] = None):
    recs = _decode_only(decode_records)
    if not recs:
        return {"verdict": UNRESOLVED, "why": "no decode_forward records"}
    cb, dp = boot.get("capture_bs") or [], bool(boot.get("disable_cuda_graph_padding", False))
    bad, n_hit, n_padded = [], 0, 0
    for r in recs:
        if r.get("graph") == "hit":
            n_hit += 1
            exp = expected_padded_bs(cb, int(r["bs_real"]), dp)
            if r.get("bs_padded") != exp:
                bad.append((r.get("seq"), "hit_pad", r.get("bs_real"), r.get("bs_padded"), exp))
            elif exp is not None and exp > int(r["bs_real"]):
                n_padded += 1
        elif r.get("graph") == "eager":
            if r.get("bs_padded") != r.get("bs_real"):
                bad.append((r.get("seq"), "eager_pad", r.get("bs_real"), r.get("bs_padded")))
            if expect_all_eager_reason and r.get("eager_reason") != expect_all_eager_reason:
                bad.append((r.get("seq"), "eager_reason", r.get("eager_reason")))
        else:
            bad.append((r.get("seq"), "graph_field", r.get("graph")))
    if expect_all_eager_reason:
        ok = not bad and n_hit == 0
        return {"verdict": PASS if ok else FAIL, "n": len(recs), "n_hit": n_hit, "bad": bad[:20]}
    if bad:
        return {"verdict": FAIL, "n": len(recs), "n_hit": n_hit, "bad": bad[:20]}
    if n_padded < REG["i6_min_padded_hits"]:
        return {"verdict": UNRESOLVED, "n": len(recs), "n_hit": n_hit,
                "why": "no hit with bs_padded > bs_real: bs_padded := bs_real mutant would pass"}
    return {"verdict": PASS, "n": len(recs), "n_hit": n_hit, "n_padded_hits": n_padded}


# ----------------------------------------------------------------- emission (every ON boot)
def check_emission(decode_rows, admission_rows=None, require_admission=True):
    fw = [r for r in decode_rows if r.get("event") == "decode_forward" and r.get("mode") == "decode"]
    wins = [r for r in decode_rows if r.get("event") == "decode_probe_window"]
    n_dec_win = sum(int(w.get("n_decode", 0)) for w in wins)
    n_pending_last = int(wins[-1].get("n_pending", 0)) if wins else None
    resolved = sum(1 for r in fw if r.get("decode_fwd_gpu_ms") is not None)
    out = {"n_decode_forward_records": len(fw), "n_decode_counted": n_dec_win, "n_resolved": resolved,
           "n_pending_last_window": n_pending_last}
    if n_dec_win < REG["emission_min_decode_forwards"]:
        return {**out, "verdict": FAIL, "why": "decode forwards not recorded (NOT_COLLECTED)"}
    frac = resolved / n_dec_win
    out["resolved_frac"] = frac
    ok = frac >= REG["emission_min_resolved_frac"]
    if require_admission:
        aw = [r for r in (admission_rows or []) if r.get("event") == "admission_window"]
        out["n_admission_windows"] = len(aw)
        ok = ok and len(aw) > 0 and sum(int(r.get("n_decode_iter", 0)) for r in aw) > 0
    return {**out, "verdict": PASS if ok else FAIL}


# ----------------------------------------------------------------- outputs
def outputs_identical(off: Mapping[str, Sequence[int]], on: Mapping[str, Sequence[int]]):
    ids = sorted(set(off) | set(on))
    diff = [i for i in ids if not off.get(i) or not on.get(i) or list(off[i]) != list(on[i])]
    return {"verdict": PASS if ids and not diff else FAIL, "n": len(ids), "differing": diff[:20]}


# ----------------------------------------------------------------- (d) TOST
_T975 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131,
         2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042]


def tost_log_ratio(pairs: Sequence[Sequence[float]], look: str = "initial"):
    """pairs = [(on, off), ...] of a positive quantity (lambda*_thr or ITL p95).  Paired log ratio, 95 % two-sided CI
    (alpha 0.025 per side, split over the initial look and the one registered extension look).
      CI inside +-ln(1.015)                  -> PASS
      CI excludes 0 (and is not inside)      -> OBSERVER_EFFECT
      CI contains 0 and exceeds +-delta      -> UNRESOLVED_POWER  (variance too large to decide, NOT an effect)
      valid pairs < tost_min_pairs           -> UNRESOLVED
    look = "initial" (n <= 10 pairs registered) | "extended" (n <= 16, only after UNRESOLVED_POWER at the initial look)."""
    if look not in ("initial", "extended"):
        raise ValueError(look)
    d = [math.log(on / off) for on, off in pairs if on and off and on > 0 and off > 0]
    n = len(d)
    cap = REG["tost_initial_pairs"] if look == "initial" else REG["tost_extended_pairs"]
    if n > cap:
        return {"verdict": UNRESOLVED, "n": n, "look": look, "why": f"{n} pairs > registered {cap} for the {look} look"}
    if n < REG["tost_min_pairs"]:
        return {"verdict": UNRESOLVED, "n": n, "look": look, "why": f"valid pairs {n} < {REG['tost_min_pairs']}"}
    mean = statistics.mean(d)
    sd = statistics.stdev(d)
    se = sd / math.sqrt(n)
    t = _T975[n - 2] if n - 1 <= len(_T975) else 1.96
    lo, hi = mean - t * se, mean + t * se
    delta = REG["tost_delta_log"]
    if lo > -delta and hi < delta:
        v = PASS
    elif lo > 0 or hi < 0:
        v = "OBSERVER_EFFECT"
    else:
        v = "UNRESOLVED_POWER"
    return {"verdict": v, "n": n, "look": look, "mean": mean, "sd": sd, "ci95": [lo, hi], "delta": delta,
            "halfwidth": t * se}


def gamma_family_verdict(metric_results: Mapping[str, Mapping[str, Any]], n_invalid_on: int, n_invalid_off: int):
    """Family verdict from the two metric TOSTs (lambda*_thr, ITL p95) and the invalid-pair asymmetry (audit r1 4d):
    ON-side invalid pairs >= OFF-side + invalid_asym_min_diff => OBSERVER_EFFECT (an observer that breaks the server
    more often is an effect even if the surviving pairs look equivalent)."""
    if n_invalid_on - n_invalid_off >= REG["invalid_asym_min_diff"]:
        return {"verdict": "OBSERVER_EFFECT", "why": f"invalid pairs ON {n_invalid_on} vs OFF {n_invalid_off} (asymmetric)"}
    vs = [m["verdict"] for m in metric_results.values()]
    if any(v == "OBSERVER_EFFECT" for v in vs):
        return {"verdict": "OBSERVER_EFFECT", "why": "a metric CI excludes 0 beyond equivalence"}
    if vs and all(v == PASS for v in vs):
        return {"verdict": PASS}
    if any(v == "UNRESOLVED_POWER" for v in vs):
        return {"verdict": "UNRESOLVED_POWER", "why": "CI contains 0 and exceeds delta"}
    return {"verdict": UNRESOLVED, "why": "valid pairs below minimum or metric missing"}


# ----------------------------------------------------------------- instrumentation level (audit r1 MAJOR 4a)
def check_instrumentation(rec: Mapping[str, Any]):
    """rec: {on: bool, log_decode_enabled, log_admission_enabled, files: {decode_probe, admission, telemetry: bool},
    sglang_file, tree}.  OFF must be really OFF (no probe file, no ENABLED log lines), ON must have both; the server's
    interpreter must import sglang from the installed tree."""
    why = []
    on = bool(rec.get("on"))
    if on:
        if not rec.get("log_decode_enabled"):
            why.append("ON boot: no 'decode probe ENABLED' log line")
        if not rec.get("log_admission_enabled"):
            why.append("ON boot: no 'admission probe ENABLED' log line")
        f = rec.get("files") or {}
        if not f.get("decode_probe") or not f.get("admission"):
            why.append("ON boot: probe output file missing")
    else:
        if rec.get("log_decode_enabled") or rec.get("log_admission_enabled"):
            why.append("OFF boot: probe ENABLED line in the log")
        if any((rec.get("files") or {}).values()):
            why.append("OFF boot: a probe/telemetry file exists")
    sf, tree = rec.get("sglang_file"), rec.get("tree")
    if not sf or not tree or not str(sf).startswith(str(tree).rstrip("/") + "/"):
        why.append(f"sglang imported from {sf!r}, not under the installed tree {tree!r}")
    return {"verdict": FAIL if why else PASS, "why": why}


# ----------------------------------------------------------------- I-5 / I-6 reconciliation (audit r1 4g)
def check_i5_i6(decode_records, admission_rows, stair_window, cap_window, cap=None, pdmux=False):
    """(1) stair: every bs in [1, 40] appears as bs_real of a decode forward inside the stair window (I-6);
    (2) cap step (cap+1 concurrent requests): I-6 max bs_real == cap, I-5 decode_bs_hist max over the overlapped windows
    == cap, n_decode_iter_at_or_above_cap > 0 and terminal_by_category['cap'] > 0 (the 49th request is recorded as a cap stop)."""
    cap = REG["i5i6_cap"] if cap is None else cap
    lo, hi = REG["i5i6_stair_bs"]
    why = []
    dec = _decode_only(decode_records)
    st = {int(r["bs_real"]) for r in dec if stair_window[0] <= float(r["t_host_start"]) <= stair_window[1]}
    stair_info = {}
    if pdmux:
        # r2: contiguous 1..M, M == I-5 max over the stair-overlapping windows, M >= floor
        m = max(st, default=0)
        holes = [b for b in range(1, m + 1) if b not in st]
        # windows whose midpoint lies inside the stair window (an edge-overlapping window of the neighbouring phase, e.g. the
        # cap step at bs 48, must not leak into the stair maximum)
        swins = [w for w in admission_rows if w.get("event") == "admission_window"
                 and stair_window[0] <= (float(w.get("t_mono_start", 0)) + float(w.get("t_mono_end", 0))) / 2 <= stair_window[1]]
        i5_stair = max((int(k) for w in swins for k in (w.get("decode_bs_hist") or {})), default=0)
        stair_info = {"stair_M": m, "stair_i5_max": i5_stair, "stair_criterion": "pdmux_contiguous_M_eq_i5"}
        if not st:
            why.append("stair: no decode forward in the stair window")
        if holes:
            why.append(f"stair bs set not contiguous from 1: holes {holes[:10]}")
        if m != i5_stair:
            why.append(f"stair M {m} != I-5 stair max decode bs {i5_stair}")
        if m < REG["i5i6_pdmux_stair_min_m"]:
            why.append(f"stair M {m} < floor {REG['i5i6_pdmux_stair_min_m']}")
    else:
        missing = [b for b in range(lo, hi + 1) if b not in st]
        if missing:
            why.append(f"stair bs not visited: {missing[:10]}")
    cw = [r for r in dec if cap_window[0] <= float(r["t_host_start"]) <= cap_window[1]]
    if not cw:
        return {"verdict": UNRESOLVED, "why": ["no decode forward in the cap step"]}
    i6_max = max(int(r["bs_real"]) for r in cw)
    wins = [w for w in admission_rows if w.get("event") == "admission_window"
            and float(w.get("t_mono_end", 0)) >= cap_window[0] and float(w.get("t_mono_start", 0)) <= cap_window[1]]
    if not wins:
        return {"verdict": UNRESOLVED, "why": ["no admission window overlaps the cap step"]}
    i5_max = max((int(k) for w in wins for k in (w.get("decode_bs_hist") or {})), default=0)
    at_cap = sum(int(w.get("n_decode_iter_at_or_above_cap", 0)) for w in wins)
    cap_stops = sum(int((w.get("terminal_by_category") or {}).get("cap", 0)) for w in wins)
    if i6_max != cap:
        why.append(f"I-6 max bs_real {i6_max} != cap {cap}")
    if i5_max != cap:
        why.append(f"I-5 max decode bs {i5_max} != cap {cap}")
    if at_cap <= 0:
        why.append("I-5 saw no decode iteration at/above cap")
    if cap_stops <= 0:
        why.append("I-5 recorded no 'cap' terminal stop while cap+1 requests were offered")
    return {"verdict": FAIL if why else PASS, "why": why, "i6_max": i6_max, "i5_max": i5_max,
            "at_cap_iters": at_cap, "cap_stops": cap_stops, **stair_info}


def stall_boundary(itl_ms):
    """Diagnostic label (r2, never alters a verdict): fraction of token ITLs > REG stall_ms; STALL_BOUNDARY when it lies within
    stall_boundary_frac +- stall_boundary_halfwidth (the p95 of a two-peaked ITL distribution jumps there)."""
    xs = [float(v) for v in itl_ms if v is not None]
    if not xs:
        return {"stall_frac": None, "label": None, "n": 0}
    f = sum(1 for v in xs if v > REG["stall_ms"]) / len(xs)
    near = abs(f - REG["stall_boundary_frac"]) <= REG["stall_boundary_halfwidth"] + 1e-12
    return {"stall_frac": f, "label": "STALL_BOUNDARY" if near else None, "n": len(xs)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--print-reg", action="store_true")
    a = ap.parse_args(argv)
    if a.print_reg:
        print(json.dumps(REG, sort_keys=True, indent=1))
        print("REG_SHA256", reg_sha256())
    return 0


if __name__ == "__main__":
    sys.exit(main())
