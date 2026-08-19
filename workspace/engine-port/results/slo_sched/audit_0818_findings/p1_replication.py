#!/usr/bin/env python3
"""AUDIT P-1 REPLICATION: g2_0_hard DOES contain an independent phase-A rA5
replication of d34 vs d44 (phase-A workload is byte-identical across OB, per the
campaign's own hardened verdict 2026-07-25)."""
import math, sys, re
from pathlib import Path
import numpy as np
ROOT=Path("/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port")
sys.path.insert(0,str(ROOT/"benchmarks"))
from pdmux_eval.analyze import load_bench_serving_rounds, percentile
TT,IT=3000.0,60.0
def score(p):
    reqs,dur=load_bench_serving_rounds(p); n=len(reqs)
    i95=[percentile(r.token_itl_ms,.95) if r.token_itl_ms else math.inf for r in reqs]
    tok=[r.ttft_ms<=TT for r in reqs]
    j=[a and (b<=IT) for a,b in zip(tok,i95)]
    return 100*sum(j)/n, 100*sum(tok)/n, float(np.median([r.ttft_ms for r in reqs])), dur

print("### g2_0_hard phase A (in2048/o32 @ rA5) -- INDEPENDENT REPLICATION")
print(f"{'file':46}{'joint%':>8}{'ttft%':>8}{'ttft_p50':>10}{'dur':>7}")
HARD=ROOT/"results/g2_0_hard"
by={}
for p in sorted(HARD.glob("g20hA_*_rA5B4_*.jsonl")):
    m=re.match(r"g20hA_(d\d+)_rep(\d+)_rA5B4_(OB\d+)_",p.name)
    arm,rep,ob=m.groups()
    j,t,t50,dur=score(p)
    by.setdefault(arm,[]).append((j,t,ob))
    print(f"{p.name:46}{j:8.2f}{t:8.2f}{t50:10.1f}{dur:7.2f}")
print()
for arm in sorted(by):
    v=np.array([x[0] for x in by[arm]])
    print(f"  {arm}: n={len(v)} mean={v.mean():6.2f} sd={v.std(ddof=1):6.2f} vals={[round(x,1) for x in v]}")

print("\n### d34 vs d44 phase A rA5, g2_0_full ALONE vs g2_0_hard ALONE vs POOLED")
FULL=ROOT/"results/g2_0_full"
full={a:[score(p)[0] for p in sorted(FULL.glob(f"g20A_{a}_rep*_rA5B4_*.jsonl"))] for a in ("d34","d44","d54")}
hard={a:[x[0] for x in by.get(a,[])] for a in ("d34","d44","d54")}
def tcrit(df): return {1:12.706,2:4.303,3:3.182,4:2.776,5:2.571,6:2.447,7:2.365,8:2.306,9:2.262}.get(df,1.96)
for label,src in (("g2_0_full",full),("g2_0_hard",hard),("pooled",{a:full[a]+hard[a] for a in full})):
    x=np.array(src["d34"]); y=np.array(src["d44"])
    sx,sy=x.std(ddof=1),y.std(ddof=1); nx,ny=len(x),len(y)
    se=math.sqrt(sx**2/nx+sy**2/ny)
    d=y.mean()-x.mean()
    if se>0:
        df=(sx**2/nx+sy**2/ny)**2/((sx**2/nx)**2/(nx-1)+(sy**2/ny)**2/(ny-1))
        lo,hi=d-tcrit(int(round(df)))*se, d+tcrit(int(round(df)))*se
    else: lo=hi=d
    print(f"  {label:10}: d34={x.mean():6.2f}(n={nx}) d44={y.mean():6.2f}(n={ny})  "
          f"d44-d34={d:+6.2f}  CI=[{lo:+6.2f},{hi:+6.2f}]  {'EXCLUDES 0' if lo*hi>0 else 'includes 0'}")
    print(f"              d34 vals={[round(v,1) for v in x]}  d44 vals={[round(v,1) for v in y]}")

print("\n### the SIGN FLIP the 2026-07-25 hardened verdict recorded (d44 vs d54, phase A rA5)")
for label,src in (("g2_0_full",full),("g2_0_hard",hard)):
    print(f"  {label}: d44={np.mean(src['d44']):.2f} d54={np.mean(src['d54']):.2f} "
          f"-> winner={'d44' if np.mean(src['d44'])>np.mean(src['d54']) else 'd54'}")
