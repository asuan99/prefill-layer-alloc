#!/usr/bin/env python3
"""AUDIT D-2 decisive test: at the p95 functional, does the STALL FRACTION or
the STALL HEIGHT determine ITL pass?  G16 HI, 4 blocks."""
import math, sys
from pathlib import Path
import numpy as np
ROOT=Path("/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port")
SLO=ROOT/"results/slo_sched"; sys.path.insert(0,str(ROOT/"benchmarks"))
from pdmux_eval.analyze import load_bench_serving_rounds, percentile
ARMS=["d16","d24","d34","d44","d54","d64","d74"]
BLOCKS=[("blk1","884336"),("blk2","884410"),("blk3","884411"),("blk4","884412")]
print(f"{'arm':5}{'pass%':>8}{'frac>45 (per-req mean)':>24}{'stall_h p50':>13}"
      f"{'P(req frac>5%)':>16}{'P(p95 in stall band)':>22}")
rows={}
for a in ARMS:
    fr=[];hh=[];p95=[]
    for blk,job in BLOCKS:
        reqs,_=load_bench_serving_rounds(SLO/f"g16_{blk}_{a}_boot1_{job}_HI.jsonl")
        for r in reqs:
            v=list(r.token_itl_ms)
            if not v: continue
            hi=[x for x in v if x>45]
            fr.append(len(hi)/len(v)); p95.append(percentile(v,.95))
            if hi: hh.append(float(np.median(hi)))
    fr=np.array(fr);p95=np.array(p95);hh=np.array(hh)
    rows[a]=dict(pas=100*(p95<=60).mean(), frac=fr.mean(), h=float(np.median(hh)),
                 pfr=100*(fr>0.05).mean(), pin=100*(p95>45).mean())
    print(f"{a:5}{rows[a]['pas']:8.2f}{rows[a]['frac']:24.4f}{rows[a]['h']:13.3f}"
          f"{rows[a]['pfr']:16.2f}{rows[a]['pin']:22.2f}")
pas=np.array([rows[a]['pas'] for a in ARMS]); frac=np.array([rows[a]['frac'] for a in ARMS])
h=np.array([rows[a]['h'] for a in ARMS])
print(f"\n  r(pass%, stall FRACTION) = {np.corrcoef(pas,frac)[0,1]:+.4f}")
print(f"  r(pass%, stall HEIGHT)   = {np.corrcoef(pas,h)[0,1]:+.4f}")
pl=[ARMS.index(x) for x in ('d34','d44','d54','d64','d74')]
print(f"  plateau only: r(pass,frac)={np.corrcoef(pas[pl],frac[pl])[0,1]:+.4f}  "
      f"r(pass,height)={np.corrcoef(pas[pl],h[pl])[0,1]:+.4f}")
print("\n  COUNTERFACTUAL: hold the height at each arm's value and give every arm")
print("  d34's stall fraction -- does the pass rate move?  (p95 is in the stall")
print("  band whenever per-request frac>5%, which holds for:")
for a in ARMS: print(f"    {a}: {rows[a]['pfr']:.1f}% of requests")

print("\n### D-1 sec3.3: shadow price of tightening theta 60 -> 58.5 (joint, HI pooled)")
def joint(th_itl):
    tot=0;ok=0
    for a in ARMS:
        for blk,job in BLOCKS:
            reqs,_=load_bench_serving_rounds(SLO/f"g16_{blk}_{a}_boot1_{job}_HI.jsonl")
            for r in reqs:
                tot+=1
                if r.ttft_ms<=3000 and (percentile(r.token_itl_ms,.95) if r.token_itl_ms else math.inf)<=th_itl: ok+=1
    return 100*ok/tot
print("  (per-arm, not pooled)")
for a in ARMS:
    for th in (60.0,58.5):
        tot=0;ok=0
        for blk,job in BLOCKS:
            reqs,_=load_bench_serving_rounds(SLO/f"g16_{blk}_{a}_boot1_{job}_HI.jsonl")
            for r in reqs:
                tot+=1
                if r.ttft_ms<=3000 and (percentile(r.token_itl_ms,.95) if r.token_itl_ms else math.inf)<=th: ok+=1
        print(f"    {a} theta={th}: joint={100*ok/tot:6.2f}%", end="")
    print()

print("\n### D-1 sec3.4: d74 ITL pass per block")
for a in ("d34","d74"):
    for blk,job in BLOCKS:
        reqs,_=load_bench_serving_rounds(SLO/f"g16_{blk}_{a}_boot1_{job}_HI.jsonl")
        v=[percentile(r.token_itl_ms,.95) if r.token_itl_ms else math.inf for r in reqs]
        print(f"  {a} {blk}: itl_pass={100*np.mean([x<=60 for x in v]):6.2f}%  p50={np.median(v):7.3f}")
