#!/usr/bin/env python3
"""AUDIT G-1 / gate #6 + #12: are the g2_0 phase-B conclusions themselves on a
metric cliff at theta_ITL = 60 ms?  Threshold ladder + band mass per cell."""
import math, sys, re
from pathlib import Path
import numpy as np
ROOT=Path("/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port")
sys.path.insert(0,str(ROOT/"benchmarks"))
from pdmux_eval.analyze import load_bench_serving_rounds, percentile
def cells(glob_pat, key):
    out={}
    for p in sorted(glob_pat):
        k=key(p.name)
        if k is None: continue
        reqs,_=load_bench_serving_rounds(p)
        v=[percentile(r.token_itl_ms,.95) if r.token_itl_ms else math.inf for r in reqs]
        t=[r.ttft_ms for r in reqs]
        out.setdefault(k,[]).append((np.array(t),np.array(v)))
    return out
FULL=ROOT/"results/g2_0_full"
c=cells(FULL.glob("g20B_*.jsonl"), lambda n: re.match(r"g20B_(d\d+)_",n).group(1))
print("### g2_0_full phase B: band mass around theta_ITL=60 (gate #6) + ladder")
print(f"{'arm':5}{'i95_p50':>9}{'band[54,66]%':>14}   goodput% at theta_ITL =")
lad=[45,50,55,58,60,62,65,70,80,100]
print(f"{'':5}{'':9}{'':14}   " + " ".join(f"{x:>6}" for x in lad))
for arm in ("d16","d24","d34","d44","d54"):
    T=np.concatenate([x[0] for x in c[arm]]); V=np.concatenate([x[1] for x in c[arm]])
    band=100*np.mean((V>=54)&(V<=66))
    gp=[100*np.mean((T<=3000)&(V<=th)) for th in lad]
    print(f"{arm:5}{np.median(V):9.2f}{band:14.2f}   " + " ".join(f"{g:6.1f}" for g in gp))
print("\n  -> per-rep i95 medians for d34 (the cell that straddles 60):")
for i,(T,V) in enumerate(c["d34"]):
    print(f"     rep{i+1}: i95_p50={np.median(V):7.3f}  goodput@60={100*np.mean((T<=3000)&(V<=60)):6.2f}%")

DC=ROOT/"results/g2_0_decliff"
c2=cells(DC.glob("g20dcB_*.jsonl"), lambda n: re.match(r"g20dcB_(d\d+)_",n).group(1))
print("\n### g2_0_decliff phase B: same")
print(f"{'arm':5}{'i95_p50':>9}{'band[54,66]%':>14}   " + " ".join(f"{x:>6}" for x in lad))
for arm in ("d16","d44","d54"):
    T=np.concatenate([x[0] for x in c2[arm]]); V=np.concatenate([x[1] for x in c2[arm]])
    band=100*np.mean((V>=54)&(V<=66))
    gp=[100*np.mean((T<=3000)&(V<=th)) for th in lad]
    print(f"{arm:5}{np.median(V):9.2f}{band:14.2f}   " + " ".join(f"{g:6.1f}" for g in gp))
print("\n  -> is the 'd16 starved / d44,d54 ceiling' dichotomy stable over the ladder?")
print("     (yes iff the d16<->d44 gap stays large and d44/d54 stay tied at every theta)")
