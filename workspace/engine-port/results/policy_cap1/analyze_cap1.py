#!/usr/bin/env python3
"""CAP1 descriptive analysis using the sealed harness (cap_char/policy_harness).  Usage: analyze_cap1.py <raw_dir> <out_json>"""
import sys, json, statistics, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cap_char/policy_harness"))
import ph_spec as P, ph_verdict as V, ph_analyze as A
raw, outp = Path(sys.argv[1]), Path(sys.argv[2])
runs = A.load_runs(raw)
add = A.cap_addendum(runs, "W1")     # raises on trace identity failure
bad = A.cap_trace_problems(runs)
cells = collections.defaultdict(lambda: {"n": 0, "good": 0})
cliff = {}; stat = {}; pct = {}
for r in runs:
    D, rate = r["arm"]["D"], r["spec"]["rate"]
    reqs = [q for s in r["segments"] for q in s["requests"]]
    c = cells[(D, rate)]; c["n"] += len(reqs); c["good"] += sum(1 for q in reqs if V.req_good(q, P.W1_TTFT_MS, P.ITL_MS))
    cliff.setdefault((D, rate), []).extend(reqs)
    stat[(D, rate, r["seed"])] = V.stationarity(reqs, P.W1_TTFT_MS, P.ITL_MS)
res = {"n_runs": len(runs), "trace_problems": bad, "addendum": add, "cells": {}, "stationarity_all": {}}
for (D, rate), reqs in sorted(cliff.items()):
    tt = [q["ttft_ms"] for q in reqs if q.get("ttft_ms") is not None]
    res["cells"][f"D{D}_r{rate}"] = {"D": D, "rate": rate, "n": cells[(D, rate)]["n"], "good": cells[(D, rate)]["good"],
        "attain": cells[(D, rate)]["good"] / cells[(D, rate)]["n"], "cliff_frac": V.cliff_fraction(reqs, P.W1_TTFT_MS),
        "cliff_adjacent": V.is_cliff(reqs, P.W1_TTFT_MS), "ttft_p50": V.percentile(tt, 0.5), "ttft_p95": V.percentile(tt, 0.95), "ttft_p99": V.percentile(tt, 0.99),
        "n_ttft_missing": len(reqs) - len(tt)}
for k, v in stat.items():
    res["stationarity_all"][f"D{k[0]}_r{k[1]}_s{k[2]}"] = v
res["switch_counts"] = sorted({r.get("switch_count", 0) for r in runs})
json.dump(res, open(outp, "w"), indent=1, sort_keys=True, default=str)
print(json.dumps({k: add[k] for k in ("lambda_star", "best_D", "status", "rerun_rmin", "stationarity_at_capacity")}))
for D, v in sorted(add["per_D"].items()):
    print("D", D, "cap", v["capacity"], "first_fail", v["first_fail"], "nonmono", v["nonmonotone"], {round(k, 3): (round(a, 3) if a is not None else None) for k, a in v["attainment"].items()})
