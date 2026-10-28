#!/usr/bin/env python3
"""Verdict skeleton of PREREG_HD_HE0cell rev0.1 (NOT SEALED).  Pure functions over run records; stdlib only; CPU.

Run record (written by ph_runner): {"run_id", "kind": he0|hd|cap, "seed", "arm": {...}, "switch_count", "split_dwell",
  "segments": [{"segment_id", "phase", "round", "duration_s", "phase_len_s"?, "requests": [{"request_id", "phase",
  "ttft_ms", "token_itl_ms": [...], "error"}]}]}

Rules implemented (and mutation-tested in ph_mutants.py):
  * good request = no error AND TTFT <= SLO AND request-internal token-ITL p95 <= SLO      (gate 4; mean-ITL never used)
  * he0 goodput = sum(good) / sum(segment durations)                                         (gate 7: durations ADD, never max)
  * H-D phase goodput = good ARRIVALS of the phase / nominal phase length
  * CLIFF_ADJACENT: > 10 % of an (arm, phase)'s requests have TTFT within SLO +- 5 %          (gate 6)
  * paired CI: seed-level paired ln ratio, t(n-1) 95 %; NO percentile bootstrap anywhere     (CONSENSUS §3 item 56)
  * H-D: seed-8 cross-fit, widest of the two role CIs; selection per phase KIND, never per instance (rev4 §2-2)
"""
from __future__ import annotations

import math
import statistics
from collections import defaultdict

T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
        21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042}
MARGIN = math.log(1.03)
CLIFF_BAND, CLIFF_FRAC = 0.05, 0.10
CAP_ATTAIN = 0.90


# ---------------------------------------------------------------------------------------------------- basics
def percentile(values, q):
    if not values:
        return math.nan
    v = sorted(float(x) for x in values)
    pos = (len(v) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return v[lo] if lo == hi else v[lo] + (v[hi] - v[lo]) * (pos - lo)


def req_good(r, ttft_slo_ms, itl_slo_ms):
    if r.get("error") or r.get("ttft_ms") is None:
        return False
    if r["ttft_ms"] > ttft_slo_ms:
        return False
    itl = r.get("token_itl_ms") or []
    return (not itl) or percentile(itl, 0.95) <= itl_slo_ms


def paired_t_ci(diffs):
    """(mean, lo, hi, n) of seed-level paired differences, t(n-1) 95 %.  n < 2 -> NaN interval."""
    n = len(diffs)
    if n < 2:
        return (diffs[0] if n else math.nan), math.nan, math.nan, n
    m = statistics.fmean(diffs)
    sd = statistics.stdev(diffs)
    if n - 1 not in T975:
        raise ValueError(f"t table covers df 1..30, got {n - 1}")
    h = T975[n - 1] * sd / math.sqrt(n)
    return m, m - h, m + h, n


def cliff_fraction(requests, ttft_slo_ms):
    ts = [r for r in requests if r.get("ttft_ms") is not None]
    if not ts:
        return 0.0
    return sum(1 for r in ts if abs(r["ttft_ms"] - ttft_slo_ms) <= CLIFF_BAND * ttft_slo_ms) / len(ts)


def is_cliff(requests, ttft_slo_ms):
    return cliff_fraction(requests, ttft_slo_ms) > CLIFF_FRAC


def slo_of(phase, w3_ttft_ms, w1_ttft_ms, itl_ms):
    return (w3_ttft_ms if phase == "W3" else w1_ttft_ms), itl_ms


# ---------------------------------------------------------------------------------------------------- HE0 cell
def he0_goodput(run, ttft_slo_ms, itl_slo_ms, phase=None):
    """sum(good)/sum(duration) over the run's segments (phase filter LO / HI optional)."""
    good = dur = 0.0
    for s in run["segments"]:
        if phase is not None and s["phase"] != phase:
            continue
        good += sum(1 for r in s["requests"] if req_good(r, ttft_slo_ms, itl_slo_ms))
        dur += float(s["duration_s"])
    if dur <= 0:
        raise ValueError(f"{run['run_id']}: zero duration")
    return good / dur


def arm_requests(runs, phase=None):
    return [r for run in runs for s in run["segments"] if phase is None or s["phase"] == phase for r in s["requests"]]


def gate5_row(runs, ttft_slo_ms):
    """Gate 5 columns for one arm (pooled over its seeds): switch_count, dwell distribution, TTFT/ITL p50/p95/p99."""
    reqs = arm_requests(runs)
    tt = [r["ttft_ms"] for r in reqs if r.get("ttft_ms") is not None]
    il = [x for r in reqs for x in (r.get("token_itl_ms") or [])]
    dwell = defaultdict(float)
    for run in runs:
        for k, v in (run.get("split_dwell") or {}).items():
            dwell[k] += float(v)
    tot = sum(dwell.values())
    return {"switch_count": [run.get("switch_count", 0) for run in runs],
            "dwell_frac": {k: v / tot for k, v in sorted(dwell.items())} if tot > 0 else {},
            "ttft_ms": {f"p{int(q * 100)}": percentile(tt, q) for q in (.5, .95, .99)},
            "itl_ms": {f"p{int(q * 100)}": percentile(il, q) for q in (.5, .95, .99)}}


def he0_compare(runs, ttft_slo_ms, itl_slo_ms, ref_static=44, min_n=4):
    """runs: {arm_name: {seed: run}} with 'dyn' and 's<D>' arms.  Returns table + label (PREREG 5-2)."""
    dyn = runs["dyn"]
    statics = {a: v for a, v in runs.items() if a != "dyn"}
    seeds = sorted(set(dyn) & set.intersection(*[set(v) for v in statics.values()]))
    g = {a: {s: he0_goodput(v[s], ttft_slo_ms, itl_slo_ms) for s in seeds} for a, v in runs.items()}
    best = max(statics, key=lambda a: statistics.fmean(g[a].values()))
    out = {"seeds": seeds, "n": len(seeds), "best_static": best, "ref_a": f"s{ref_static}",
           "goodput_mean": {a: statistics.fmean(g[a].values()) for a in g}}
    res = {}
    for key, ref in (("a", f"s{ref_static}"), ("b", best)):
        d = [math.log(g["dyn"][s] / g[ref][s]) for s in seeds]
        res[key] = dict(zip(("mean", "lo", "hi", "n"), paired_t_ci(d)), ref=ref)
    out["ci"] = res
    cliff = {}
    for a, v in runs.items():
        for ph in ("LO", "HI"):
            cliff[f"{a}:{ph}"] = is_cliff(arm_requests(list(v.values()), ph), ttft_slo_ms)
    out["cliff"] = cliff
    involved = ["dyn", f"s{ref_static}", best]
    if len(seeds) < min_n:
        out["label"] = "HE0CELL_UNRESOLVED"
        out["why"] = f"n={len(seeds)} < {min_n}"
    elif any(cliff[f"{a}:{ph}"] for a in involved for ph in ("LO", "HI")):
        out["label"] = "HE0CELL_UNRESOLVED"
        out["why"] = "CLIFF_ADJACENT"
    elif res["a"]["lo"] > MARGIN and res["b"]["lo"] > MARGIN:
        out["label"] = "DYN_ABOVE_STATIC"
    elif res["b"]["hi"] < -MARGIN:
        out["label"] = "DYN_BELOW_STATIC"
    elif all(-MARGIN < res[k]["lo"] and res[k]["hi"] < MARGIN for k in ("a", "b")):
        out["label"] = "DYN_EQUIV_STATIC"
    else:
        out["label"] = "HE0CELL_UNRESOLVED"
        out["why"] = "interval straddles a margin"
    out["gate5"] = {a: gate5_row(list(v.values()), ttft_slo_ms) for a, v in runs.items()}
    return out


def select_tune(table, tie_frac=0.01):
    """table: {config_name: [{"goodput": float, "switch_count": int}, ...]} (tuning seeds only).  Highest mean goodput;
    within tie_frac of the best -> fewer mean switch_count; then name (deterministic)."""
    mean = {c: statistics.fmean(x["goodput"] for x in v) for c, v in table.items()}
    top = max(mean.values())
    near = [c for c in mean if mean[c] >= top * (1 - tie_frac)]
    sw = {c: statistics.fmean(x["switch_count"] for x in table[c]) for c in near}
    return min(near, key=lambda c: (sw[c], c)), {"mean_goodput": mean, "mean_switch": sw}


# ---------------------------------------------------------------------------------------------------- H-D
def hd_phase_goodput(run, phase, w1_ttft_ms, w3_ttft_ms, itl_ms):
    """good ARRIVALS of `phase` / nominal phase length (arrival-based; drain time not counted)."""
    reqs, plen = [], None
    for s in run["segments"]:
        if s["phase"] in (phase, "CONT"):
            plen = s["phase_len_s"]
            reqs += [r for r in s["requests"] if r["phase"] == phase]
    if plen is None:
        raise ValueError(f"{run['run_id']}: no segment for phase {phase}")
    t, i = slo_of(phase, w3_ttft_ms, w1_ttft_ms, itl_ms)
    return sum(1 for r in reqs if req_good(r, t, i)) / plen


def hd_table(runs, w1_ttft_ms, w3_ttft_ms, itl_ms):
    """runs: {(variant, plen): {D: {seed: run}}} -> {(variant, plen): {D: {seed: {'W1': g, 'W3': g, 'total': g}}}}"""
    out = {}
    for vk, byD in runs.items():
        out[vk] = {}
        for D, bys in byD.items():
            out[vk][D] = {}
            for s, run in bys.items():
                g1 = hd_phase_goodput(run, "W1", w1_ttft_ms, w3_ttft_ms, itl_ms)
                g3 = hd_phase_goodput(run, "W3", w1_ttft_ms, w3_ttft_ms, itl_ms)
                out[vk][D][s] = {"W1": g1, "W3": g3, "total": (g1 + g3) / 2.0}   # equal phase lengths -> length-weighted mean
    return out


def crossfit_H(tab, seeds_a, seeds_b):
    """tab: {D: {seed: {'W1','W3','total'}}}.  Two roles; selection on one half, evaluation on the other.
    Per role: D*_w = argmax_D mean selection-seed phase goodput (per phase KIND, pooled over seeds -- never per instance);
    G_best = single static D with max mean total goodput on the EVALUATION seeds (upward-biased, hard on PRESENT);
    h_s = ln(G_phase-oracle,s / G_best,s) on evaluation seeds.  Returns per-role stats and the widest-CI summary."""
    roles = []
    Ds = sorted(tab)
    for sel, ev in ((seeds_a, seeds_b), (seeds_b, seeds_a)):
        pick = {w: max(Ds, key=lambda D: statistics.fmean(tab[D][s][w] for s in sel)) for w in ("W1", "W3")}
        d_best = max(Ds, key=lambda D: statistics.fmean(tab[D][s]["total"] for s in ev))
        h = []
        for s in ev:
            oracle = (tab[pick["W1"]][s]["W1"] + tab[pick["W3"]][s]["W3"]) / 2.0
            h.append(math.log(oracle / tab[d_best][s]["total"]))
        m, lo, hi, n = paired_t_ci(h)
        roles.append({"pick": pick, "best": d_best, "H": m, "lo": lo, "hi": hi, "n": n, "h_s": h})
    summ = {"H": statistics.fmean(r["H"] for r in roles), "lo": min(r["lo"] for r in roles),
            "hi": max(r["hi"] for r in roles), "roles": roles}
    return summ


def hd_label(clean_by_len, cliff_by_len, qs_run=(1.0,), qs_required=(0.8, 1.0)):
    """clean_by_len: {plen: crossfit summary} for the clean-start variant at the run q; cliff_by_len: {plen: bool} (selected or
    best arm CLIFF_ADJACENT).  PRESENT: both lengths H >= ln 1.03 and widest lower > 0 (any one q suffices).  ABSENT: both
    lengths, ALL registered q, widest upper < ln 1.03 -- impossible when a registered q was not run (rev4 r2 N-2)."""
    lens = sorted(clean_by_len)
    if len(lens) < 2:
        return "HEADROOM_UNRESOLVED", "need both phase lengths"
    if any(cliff_by_len.get(p) for p in lens):
        return "HEADROOM_UNRESOLVED", "CLIFF_ADJACENT"
    if all(clean_by_len[p]["H"] >= MARGIN and clean_by_len[p]["lo"] > 0 for p in lens):
        return "DYNAMIC_HEADROOM_PRESENT", ""
    if all(clean_by_len[p]["hi"] < MARGIN for p in lens):
        if set(qs_required) <= set(qs_run):
            return "DYNAMIC_HEADROOM_ABSENT", ""
        return "HEADROOM_UNRESOLVED", "ABSENT not declarable: registered q not all run"
    return "HEADROOM_UNRESOLVED", ""


# ---------------------------------------------------------------------------------------------------- capacity
def capacity_per_D(cells, ladder):
    """cells: [{rate, n, good}] for one D (all seeds).  attainment per rate = sum(good)/sum(n).  Capacity = last ladder rate
    passing (>= 90 %) before the FIRST failing rate; bracket [last_pass, first_fail); a pass above the first fail is flagged."""
    att = {}
    for r in ladder:
        cs = [c for c in cells if abs(c["rate"] - r) < 1e-9]
        att[r] = (sum(c["good"] for c in cs) / sum(c["n"] for c in cs)) if cs and sum(c["n"] for c in cs) else None
    last = first_fail = None
    for r in ladder:
        a = att[r]
        if a is None:
            continue
        if a >= CAP_ATTAIN and first_fail is None:
            last = r
        elif a < CAP_ATTAIN and first_fail is None:
            first_fail = r
    nonmono = first_fail is not None and any(att[r] is not None and att[r] >= CAP_ATTAIN for r in ladder if r > first_fail)
    return {"capacity": last, "first_fail": first_fail, "attainment": att, "nonmonotone": nonmono}


def lambda_star_best_static(per_D):
    """per_D: {D: capacity_per_D result}.  Best static = largest capacity; None if no D passes any rate."""
    caps = {D: v["capacity"] for D, v in per_D.items() if v["capacity"] is not None}
    if not caps:
        return None, None
    D = max(caps, key=lambda k: caps[k])
    return caps[D], D
