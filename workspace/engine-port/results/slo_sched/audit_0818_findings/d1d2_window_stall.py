#!/usr/bin/env python3
"""AUDIT D-1 / D-2: threshold position, 'adjudicable window', stall height vs
stall fraction.  G16 HI, 4 blocks, canonical loader."""
import math, sys
from pathlib import Path
import numpy as np
ROOT=Path("/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port")
SLO=ROOT/"results/slo_sched"; sys.path.insert(0,str(ROOT/"benchmarks"))
from pdmux_eval.analyze import load_bench_serving_rounds, percentile
ARMS=["d16","d24","d34","d44","d54","d64","d74"]
BLOCKS=[("blk1","884336"),("blk2","884410"),("blk3","884411"),("blk4","884412")]

R={}; TOK={}
for arm in ARMS:
    i95=[]; tok=[]
    for blk,job in BLOCKS:
        reqs,_=load_bench_serving_rounds(SLO/f"g16_{blk}_{arm}_boot1_{job}_HI.jsonl")
        i95+= [percentile(r.token_itl_ms,.95) if r.token_itl_ms else math.inf for r in reqs]
        tok+= [v for r in reqs for v in r.token_itl_ms]
    R[arm]=np.array(i95); TOK[arm]=np.array(tok)

print("### D-1 table 1 reproduction + concentration of the mass just below 60")
print(f"{'arm':5}{'p50':>8}{'p90':>8}{'pass%':>8}{'in[56,60]%':>12}{'in[54,66]%':>12}{'>66%':>8}{'<54%':>8}")
for a in ARMS:
    v=R[a]; n=len(v)
    print(f"{a:5}{np.median(v[np.isfinite(v)]):8.2f}{np.percentile(v[np.isfinite(v)],90):8.2f}"
          f"{100*(v<=60).mean():8.2f}{100*((v>=56)&(v<=60)).mean():12.2f}"
          f"{100*((v>=54)&(v<=66)).mean():12.2f}{100*(v>66).mean():8.2f}{100*(v<54).mean():8.2f}")

print("\n### D-1 part 2: the 'adjudicable window'.  Two criteria, wide grid.")
def win(v, crit, lo=1.0, hi=400.0, npts=8000):
    grid=np.exp(np.linspace(np.log(lo),np.log(hi),npts))
    ok=[]
    for th in grid:
        p=(v<=th).mean()
        bm=((v>=0.9*th)&(v<=1.1*th)).mean()
        if crit=="doc":  c=(1-p)>=0.05 and bm<=0.25
        else:            c=(0.05<=p<=0.95) and bm<=0.25
        ok.append(c)
    ok=np.array(ok)
    if not ok.any(): return None
    # contiguous segments
    segs=[]; s=None
    for i,b in enumerate(ok):
        if b and s is None: s=i
        if (not b) and s is not None: segs.append((grid[s],grid[i-1])); s=None
    if s is not None: segs.append((grid[s],grid[-1]))
    return segs
for crit,label in (("doc","doc's stated: fail>=5% & band<=0.25"),
                   ("prereg","E-B1 prereg: 5%<=pass<=95% & band<=0.25")):
    print(f"  -- {label}")
    for a in ARMS:
        segs=win(R[a],crit)
        print(f"     {a}: " + " U ".join(f"[{x:.1f},{y:.1f}]" for x,y in segs))
    # 7-arm common
    grid=np.exp(np.linspace(np.log(1.0),np.log(400.0),8000))
    common=np.ones(len(grid),bool)
    for a in ARMS:
        v=R[a]
        p=np.array([(v<=t).mean() for t in grid]); bm=np.array([((v>=.9*t)&(v<=1.1*t)).mean() for t in grid])
        c=((1-p)>=0.05)&(bm<=0.25) if crit=="doc" else ((p>=0.05)&(p<=0.95)&(bm<=0.25))
        common&=c
    if common.any():
        print(f"     7-arm common: [{grid[common][0]:.2f}, {grid[common][-1]:.2f}]  (grid spans [1,400])")
    print(f"     -> at theta=60: " + ", ".join(
        f"{a}:fail={100*(R[a]>60).mean():.1f}%/band={100*((R[a]>=54)&(R[a]<=66)).mean():.1f}%" for a in ARMS))

print("\n### D-2: stall HEIGHT vs stall FRACTION (token-level ITL, G16 HI)")
print(f"{'arm':5}{'n_tok':>9}{'frac>45ms':>11}{'frac>60ms':>11}{'median|>45':>12}{'mean|>45':>10}{'p95|>45':>10}{'mean_all':>10}")
for a in ARMS:
    t=TOK[a]; s=t[t>45]
    print(f"{a:5}{len(t):9}{(t>45).mean():11.4f}{(t>60).mean():11.4f}"
          f"{np.median(s):12.2f}{s.mean():10.2f}{np.percentile(s,95):10.2f}{t.mean():10.3f}")
print("  doc D-2 quotes 0.154 -> 0.091 for '>45 ms token fraction'; check which arms bracket that.")
