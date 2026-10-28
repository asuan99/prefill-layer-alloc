#!/usr/bin/env python3
"""Glue: raw run records -> verdict inputs -> labels/addendum fragments.  Pure; no server.  See ph_verdict for the rules."""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import ph_spec as P
import ph_verdict as V


def load_runs(out):
    return [json.loads(p.read_text()) for p in sorted((Path(out) / "runs").glob("*.json"))]


def cap_addendum(runs, workload, floor_ms=None):
    """CAP Job -> {'lambda_star': {workload: x}, 'best_D': D, 'per_D': ...}.  W3 SLO = 3 x floor (same Job)."""
    if workload == "W3":
        if not floor_ms:
            raise RuntimeError("ABORT_NO_FLOOR: W3 capacity needs the Job's floor")
        ttft = P.W3_TTFT_K * floor_ms
    else:
        ttft = P.W1_TTFT_MS
    byD = {}
    for r in runs:
        good = sum(1 for s in r["segments"] for q in s["requests"] if V.req_good(q, ttft, P.ITL_MS))
        n = sum(len(s["requests"]) for s in r["segments"])
        byD.setdefault(r["arm"]["D"], []).append({"rate": r["spec"]["rate"], "n": n, "good": good})
    ladder = sorted({c["rate"] for v in byD.values() for c in v})
    per = {D: V.capacity_per_D(v, ladder) for D, v in byD.items()}
    lam, D, status = V.lambda_star_best_static(per)
    # stationarity of the capacity cell (S1/S2) at the accepted rate, argmax D, per seed
    stat = {}
    if lam is not None:
        for r in runs:
            if r["arm"]["D"] == D and abs(r["spec"]["rate"] - lam) < 1e-9:
                reqs = [q for s in r["segments"] for q in s["requests"]]
                stat[r["seed"]] = V.stationarity(reqs, ttft, P.ITL_MS)["label"]
    rmin = ladder[0]
    nxt = V.next_ladder_rmin(status, rmin, P.CAP_FACTOR[workload], P.CAP_N_RATES[workload])
    return {"lambda_star": {workload: lam}, "best_D": D, "status": status, "ttft_slo_ms": ttft, "per_D": per, "ladder": ladder,
            "stationarity_at_capacity": stat, "rerun_rmin": nxt}


def tune_pick(runs):
    table = {}
    for r in runs:
        g = V.he0_goodput(r, P.W1_TTFT_MS, P.ITL_MS)
        table.setdefault(r["arm"]["name"], []).append({"goodput": g, "switch_count": r.get("switch_count", 0)})
    if any(len(v) != len(P.TUNE_SEEDS) for v in table.values()):
        raise RuntimeError("ABORT_TUNE_INCOMPLETE: every config needs both tuning seeds")
    best, detail = V.select_tune(table)
    return {"tuned_config": next(r["arm"] for r in runs if r["arm"]["name"] == best), "detail": detail}


def eval_labels(runs):
    bad = eval_trace_problems(runs)
    if bad:
        raise RuntimeError(f"ABORT_TRACE_IDENTITY: arms of seeds {bad} saw different requests")
    by = {}
    for r in runs:
        name = "dyn" if r["arm"]["kind"] == "dynamic" else f"s{r['arm']['D']}"
        by.setdefault(name, {})[r["seed"]] = r
    return V.he0_compare(by, P.W1_TTFT_MS, P.ITL_MS)


def hd_labels(runs, w3_ttft_ms, qs_run=(1.0,)):
    """runs: all HD-k records (seeds 1..8).  Clean-start labels from the cross-fit; continuous variant descriptive only."""
    bad = hd_trace_problems(runs)
    if bad:
        raise RuntimeError(f"ABORT_TRACE_IDENTITY: arms saw different requests for {bad}")
    runs_by = {}
    for r in runs:
        key = (r["spec"]["variant"], r["spec"]["phase_len_s"])
        runs_by.setdefault(key, {}).setdefault(r["arm"]["D"], {})[r["seed"]] = r
    tab = V.hd_table(runs_by, P.W1_TTFT_MS, w3_ttft_ms, P.ITL_MS)
    # continuous variant: also the value with the first 10 % of every phase excluded (rev4 2-3), both reported
    warm = {}
    for vk in [k for k in runs_by if k[0] == "cont"]:
        warm[vk] = {D: {sd: {w: V.hd_phase_goodput(run, w, P.W1_TTFT_MS, w3_ttft_ms, P.ITL_MS, exclude_warm=0.10)
                             for w in ("W1", "W3")} for sd, run in bys.items()} for D, bys in runs_by[vk].items()}
        for D in warm[vk]:
            for sd in warm[vk][D]:
                warm[vk][D][sd]["total"] = (warm[vk][D][sd]["W1"] + warm[vk][D][sd]["W3"]) / 2.0
    sa, sb = [1, 2, 3, 4], [5, 6, 7, 8]
    out = {"variants": {}}
    clean, cliff = {}, {}
    for vk, t in tab.items():
        seeds = sorted({s for D in t.values() for s in D})
        if seeds != sorted(sa + sb):
            out["variants"][str(vk)] = {"error": f"seeds {seeds} != 1..8"}
            continue
        cf = V.crossfit_H(t, sa, sb)
        out["variants"][str(vk)] = {"H": cf["H"], "lo": cf["lo"], "hi": cf["hi"],
                                    "picks": [r["pick"] for r in cf["roles"]], "best": [r["best"] for r in cf["roles"]],
                                    "descriptive_only": vk[0] == "cont"}
        if vk[0] == "cont" and vk in warm:
            cfw = V.crossfit_H(warm[vk], sa, sb)
            out["variants"][str(vk)]["H_warm_excluded"] = {"H": cfw["H"], "lo": cfw["lo"], "hi": cfw["hi"]}
        out["variants"][str(vk)]["gate5"] = V.hd_gate5(runs_by[vk], cf["roles"][0]["pick"])
        if vk[0] == "clean":
            clean[vk[1]] = cf
            arms = {D for r in cf["roles"] for D in (*r["pick"].values(), r["best"])}
            bad = False
            for D in arms:
                for ph, slo in (("W1", P.W1_TTFT_MS), ("W3", w3_ttft_ms)):
                    reqs = [q for r in runs_by[vk][D].values() for s in r["segments"] if s["phase"] == ph for q in s["requests"]]
                    bad = bad or V.is_cliff(reqs, slo)
            cliff[vk[1]] = bad
    if len(clean) == 2:
        out["label"], out["why"] = V.hd_label(clean, cliff, qs_run)
    else:
        out["label"], out["why"] = "HEADROOM_UNRESOLVED", "clean variants missing"
    return out


def trace_identity_problems(runs, keyfn):
    """Runs that must have seen the SAME requests (same seed/spec, different arm) must share trace_hash (audit M7)."""
    groups = {}
    for r in runs:
        groups.setdefault(keyfn(r), set()).add(r["trace_hash"])
    return [k for k, h in groups.items() if len(h) != 1]


def eval_trace_problems(runs):
    return trace_identity_problems(runs, lambda r: r["seed"])


def hd_trace_problems(runs):
    return trace_identity_problems(runs, lambda r: (r["seed"], r["spec"]["variant"], r["spec"]["phase_len_s"]))


def cap_trace_problems(runs):
    return trace_identity_problems(runs, lambda r: (r["spec"]["rate"], r["seed"]))
