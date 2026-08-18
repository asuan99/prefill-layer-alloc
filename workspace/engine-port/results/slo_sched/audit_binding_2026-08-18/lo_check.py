#!/usr/bin/env python3
"""R-4 check: is the LO row three numbers or one?  How much headroom is there?"""
import json, math, statistics, sys
from pathlib import Path
ROOT = Path("/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port")
SLO_DIR = ROOT / "results/slo_sched"
sys.path.insert(0, str(ROOT / "benchmarks"))
from pdmux_eval.analyze import load_bench_serving_rounds, percentile
ARMS = ["d16","d24","d34","d44","d54","d64","d74"]
JOBS = [("blk1","884336"),("blk2","884410"),("blk3","884411"),("blk4","884412")]
out={}
per_blk_itl={}
for arm in ARMS:
    T,I=[],[]; blk_itl=[]
    for blk,job in JOBS:
        reqs,_=load_bench_serving_rounds(SLO_DIR/f"g16_{blk}_{arm}_boot1_{job}_LO.jsonl")
        t=[r.ttft_ms for r in reqs]
        i=[percentile(r.token_itl_ms,.95) if r.token_itl_ms else math.inf for r in reqs]
        T+=t; I+=i
        blk_itl.append(100*sum(v<=60 for v in i)/len(i))
    per_blk_itl[arm]=blk_itl
    fin=[v for v in I if math.isfinite(v)]
    out[arm]={"n":len(T),"ttft_max_ms":max(T),"ttft_p99_ms":percentile(T,.99),
              "ttft_pass_pct":100*sum(v<=3000 for v in T)/len(T),
              "itl95_median_ms":statistics.median(fin),"itl95_p95_ms":percentile(fin,.95),
              "itl95_p99_ms":percentile(fin,.99),"itl95_max_ms":max(fin),
              "itl_pass_pct":100*sum(v<=60 for v in I)/len(I),
              "n_itl_fail":sum(v>60 for v in I),
              "per_block_itl_pass":blk_itl}
# block-paired contrasts on the ITL axis (the only non-degenerate LO axis)
def paired(a1,a2):
    d=[per_blk_itl[a1][k]-per_blk_itl[a2][k] for k in range(4)]
    m,s=statistics.fmean(d),statistics.stdev(d); h=3.182*s/2
    return {"diffs":d,"mean":m,"t3_ci":[m-h,m+h],"excludes_zero":(m-h)*(m+h)>0}
pc={f"{a}-d44":paired(a,"d44") for a in ARMS if a!="d44"}
print(json.dumps({"per_arm":out,"paired_vs_d44":pc},indent=1))
