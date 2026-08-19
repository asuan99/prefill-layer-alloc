#!/usr/bin/env python3
"""window_exists.py under leave-one-block-out (the design's own sec2-2 rule).

blk1 is a systematic ITL-axis outlier (d74 HI ITL pass 42.3% vs 96.5-98.8% in
blk2-4).  Since the whole rate-axis feasibility of an arm hinges on
q = P(ITL-p95 > 60 | saturated) >= 0.05, and blk1 supplies most of the failures,
the feasibility verdict must be reported LOBO.  Read-only.
"""
import json, math, statistics, sys
from pathlib import Path

ROOT = Path("/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port")
SLO = ROOT / "results/slo_sched"
sys.path.insert(0, str(ROOT / "benchmarks"))
from pdmux_eval.analyze import load_bench_serving_rounds, percentile

ARMS = ["d16", "d24", "d34", "d44", "d54", "d64", "d74"]
JOBS = [("blk1", "884336"), ("blk2", "884410"), ("blk3", "884411"), ("blk4", "884412")]
SAT_CUT = 45.0


def feas(vals):
    sat = [x for x in vals if x >= SAT_CUT]
    if not sat:
        return None
    q = sum(x > 60.0 for x in sat) / len(sat)
    b = sum(54.0 <= x <= 66.0 for x in sat) / len(sat)
    ok = (q >= 0.05) and (b / q <= 5.0 if q else False)
    return {"q": q, "b": b, "b_over_q": (b / q) if q else None, "feasible": ok,
            "w_min": (0.05 / q) if q else None, "w_max": (0.25 / b) if b else None}


def main():
    out = {"note": "ITL axis feasibility over ALL rates; full and leave-one-block-out",
           "arms": {}}
    for arm in ARMS:
        per_blk = {}
        for blk, job in JOBS:
            reqs, _ = load_bench_serving_rounds(SLO / f"g16_{blk}_{arm}_boot1_{job}_HI.jsonl")
            per_blk[blk] = [percentile(r.token_itl_ms, .95) if r.token_itl_ms else math.inf
                            for r in reqs]
        allv = [x for v in per_blk.values() for x in v]
        row = {"ALL4": feas(allv)}
        for drop in per_blk:
            row[f"drop_{drop}"] = feas([x for k, v in per_blk.items() if k != drop for x in v])
        row["verdict_stable_under_LOBO"] = len({row[k]["feasible"] for k in row if k != "verdict_stable_under_LOBO"}) == 1
        out["arms"][arm] = row
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
