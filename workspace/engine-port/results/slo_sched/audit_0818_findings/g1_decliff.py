#!/usr/bin/env python3
"""AUDIT G-1 / S-2 / P-2: g2_0_decliff per-cell reproduction + the
circularity check on the survey's selection rule (band mass is computed on the
ARM-POOLED distribution, so 'discriminative' == 'arms are far apart')."""
import json, math, sys, re
from pathlib import Path
import numpy as np
ROOT = Path("/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port")
sys.path.insert(0, str(ROOT / "benchmarks"))
from pdmux_eval.analyze import load_bench_serving_rounds, percentile
DC = ROOT/"results/g2_0_decliff"
TT, IT = 3000.0, 60.0

rows=[]
for p in sorted(DC.glob("g20dc[AB]_*.jsonl")):
    m=re.match(r"g20dc([AB])_(d\d+)_(\d+)_rA([\d.]+)B4_OB512_(\d+)\.jsonl", p.name)
    if not m: print("SKIP", p.name); continue
    ph, arm, rep, rA, job = m.groups()
    reqs,dur = load_bench_serving_rounds(p)
    n=len(reqs)
    i95=[percentile(r.token_itl_ms,.95) if r.token_itl_ms else math.inf for r in reqs]
    tok=[r.ttft_ms<=TT for r in reqs]; iok=[v<=IT for v in i95]
    j=[a and b for a,b in zip(tok,iok)]
    rows.append(dict(phase=ph,arm=arm,rep=int(rep),rateA=float(rA),job=job,n=n,dur=dur,
        joint=100*sum(j)/n, ttft=100*sum(tok)/n, itl=100*sum(iok)/n,
        i95=i95, ttft_p50=float(np.median([r.ttft_ms for r in reqs])),
        i95_p50=float(np.median([v for v in i95 if math.isfinite(v)]))))

def show(ph, rateA=None):
    sel=[r for r in rows if r["phase"]==ph and (rateA is None or r["rateA"]==rateA)]
    print(f"\n-- phase {ph}" + (f", rateA={rateA}" if rateA else " (all rateA)"))
    print(f"   {'arm':5}{'n_rep':>6}{'mean':>8}{'sd':>7}{'median':>8}   per-rep")
    for arm in ("d16","d44","d54"):
        v=np.array([r["joint"] for r in sel if r["arm"]==arm])
        if len(v)==0: continue
        print(f"   {arm:5}{len(v):6}{v.mean():8.2f}{v.std(ddof=1) if len(v)>1 else 0:7.2f}{np.median(v):8.2f}   {[round(x,1) for x in v]}")

print("### SPLIT_MOVES table reproduction (phase B, by rateA)")
for ra in (2.0,3.0,3.5,4.0): show("B", ra)
print("\n### PLATEAU_WIDTH decliff rows (pooled over rateA)")
show("A"); show("B")

print("\n### circularity check: band mass is POOLED OVER ARMS")
for ph in ("A","B"):
    print(f"  phase {ph}")
    pooled=[]
    for arm in ("d16","d44","d54"):
        v=[x for r in rows if r["phase"]==ph and r["arm"]==arm for x in r["i95"]]
        pooled+=v
        fin=[x for x in v if math.isfinite(x)]
        fail=100*sum(x>IT for x in v)/len(v); band=100*sum(54<=x<=66 for x in v)/len(v)
        print(f"    {arm}: n={len(v):5} p50={np.median(fin):7.2f}  fail%={fail:6.2f}  band%={band:6.2f}")
    fail=100*sum(x>IT for x in pooled)/len(pooled); band=100*sum(54<=x<=66 for x in pooled)/len(pooled)
    print(f"    POOLED: n={len(pooled):5} p50={np.median([x for x in pooled if math.isfinite(x)]):7.2f}  fail%={fail:6.2f}  band%={band:6.2f}")
allp=[x for r in rows for x in r["i95"]]
print(f"  ALL (A+B, all arms, all rates) n={len(allp)} fail%={100*sum(x>IT for x in allp)/len(allp):.2f} "
      f"band%={100*sum(54<=x<=66 for x in allp)/len(allp):.2f}")

print("\n### same for G16 HI, per arm (the 'cliff' campaign)")
SLO=ROOT/"results/slo_sched"
BLOCKS=[("blk1","884336"),("blk2","884410"),("blk3","884411"),("blk4","884412")]
pooled=[]
for arm in ("d16","d24","d34","d44","d54","d64","d74"):
    v=[]
    for blk,job in BLOCKS:
        reqs,_=load_bench_serving_rounds(SLO/f"g16_{blk}_{arm}_boot1_{job}_HI.jsonl")
        v+=[percentile(r.token_itl_ms,.95) if r.token_itl_ms else math.inf for r in reqs]
    pooled+=v
    fin=[x for x in v if math.isfinite(x)]
    print(f"    {arm}: n={len(v):5} p50={np.median(fin):7.2f}  fail%={100*sum(x>IT for x in v)/len(v):6.2f}"
          f"  band%={100*sum(54<=x<=66 for x in v)/len(v):6.2f}")
print(f"    POOLED HI: n={len(pooled)} fail%={100*sum(x>IT for x in pooled)/len(pooled):.2f} "
      f"band%={100*sum(54<=x<=66 for x in pooled)/len(pooled):.2f}")
