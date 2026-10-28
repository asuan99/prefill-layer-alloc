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
    lam, D = V.lambda_star_best_static(per)
    return {"lambda_star": {workload: lam}, "best_D": D, "ttft_slo_ms": ttft, "per_D": per, "ladder": ladder}


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
    by = {}
    for r in runs:
        name = "dyn" if r["arm"]["kind"] == "dynamic" else f"s{r['arm']['D']}"
        by.setdefault(name, {})[r["seed"]] = r
    return V.he0_compare(by, P.W1_TTFT_MS, P.ITL_MS)


def hd_labels(runs, w3_ttft_ms, qs_run=(1.0,)):
    """runs: all HD-k records (seeds 1..8).  Clean-start labels from the cross-fit; continuous variant descriptive only."""
    runs_by = {}
    for r in runs:
        key = (r["spec"]["variant"], r["spec"]["phase_len_s"])
        runs_by.setdefault(key, {}).setdefault(r["arm"]["D"], {})[r["seed"]] = r
    tab = V.hd_table(runs_by, P.W1_TTFT_MS, w3_ttft_ms, P.ITL_MS)
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
