#!/usr/bin/env python3
"""AUDIT P-1: is the d34 valley in g2_0_full phase A real?

Read-only. Uses ONLY the canonical loader/predicate (pdmux_eval.analyze),
same as g16_analyze.boot_estimands, so the scorer is not a new one.
"""
import json, math, sys, itertools, re
from pathlib import Path
import numpy as np

ROOT = Path("/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port")
sys.path.insert(0, str(ROOT / "benchmarks"))
from pdmux_eval.analyze import load_bench_serving_rounds, percentile

TTFT_SLO, ITL_SLO = 3000.0, 60.0
FULL = ROOT / "results/g2_0_full"
ARMS = ["d16", "d24", "d34", "d44", "d54"]
REPS = [1, 2, 3, 4]

def cell(phase, arm, rep):
    g = sorted(FULL.glob(f"g20{phase}_{arm}_rep{rep}_*.jsonl"))
    assert len(g) == 1, (phase, arm, rep, g)
    path = g[0]
    job = re.search(r"_(\d+)\.jsonl$", path.name).group(1)
    reqs, dur = load_bench_serving_rounds(path)
    n = len(reqs)
    ttft_ok = [r.ttft_ms <= TTFT_SLO for r in reqs]
    itl95 = [percentile(r.token_itl_ms, .95) if r.token_itl_ms else math.inf for r in reqs]
    itl_ok = [v <= ITL_SLO for v in itl95]
    joint = [a and b for a, b in zip(ttft_ok, itl_ok)]
    return dict(path=path.name, job=job, n=n, dur=dur,
                joint_pct=100.0*sum(joint)/n, ttft_pct=100.0*sum(ttft_ok)/n,
                itl_pct=100.0*sum(itl_ok)/n, n_good=sum(joint),
                goodput_req_s=sum(joint)/dur, thr_req_s=n/dur,
                ttft_p50=float(np.median([r.ttft_ms for r in reqs])),
                itl_p50=float(np.median([v for v in itl95 if math.isfinite(v)])),
                joint_vec=[int(x) for x in joint])

out = {}
for phase in ("A", "B"):
    out[phase] = {}
    for arm in ARMS:
        rows = [cell(phase, arm, r) for r in REPS]
        vals = np.array([r["joint_pct"] for r in rows])
        out[phase][arm] = dict(
            per_rep=[{k: v for k, v in r.items() if k != "joint_vec"} for r in rows],
            mean=float(vals.mean()), sd=float(vals.std(ddof=1)),
            vals=[float(v) for v in vals],
            n_good=[r["n_good"] for r in rows],
            jobs=[r["job"] for r in rows])

print(json.dumps(out, indent=1))
