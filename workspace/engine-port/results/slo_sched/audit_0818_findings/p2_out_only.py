#!/usr/bin/env python3
"""AUDIT P-2: the ONLY contrast in which `out` moves alone.

g2_0_full phase A vs B changes out (32->512) AND rate (5->4): confounded.
g2_0_decliff at rateA=4 vs phase B (rB=4 always) changes out alone."""
import math, sys, re
from pathlib import Path
import numpy as np
ROOT=Path("/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port")
sys.path.insert(0,str(ROOT/"benchmarks"))
from pdmux_eval.analyze import load_bench_serving_rounds, percentile
DC=ROOT/"results/g2_0_decliff"; TT,IT=3000.0,60.0
rows={}
for p in sorted(DC.glob("g20dc[AB]_*.jsonl")):
    ph,arm,rep,rA,job=re.match(r"g20dc([AB])_(d\d+)_(\d+)_rA([\d.]+)B4_OB512_(\d+)\.jsonl",p.name).groups()
    reqs,dur=load_bench_serving_rounds(p); n=len(reqs)
    i95=[percentile(r.token_itl_ms,.95) if r.token_itl_ms else math.inf for r in reqs]
    j=[(r.ttft_ms<=TT) and (v<=IT) for r,v in zip(reqs,i95)]
    tk=100*np.mean([r.ttft_ms<=TT for r in reqs])
    rows.setdefault((ph,arm,float(rA)),[]).append((100*sum(j)/n, tk))
def plateau(m,thr=5):
    top=max(m.values()); return sorted(a for a in m if m[a]>=top-thr)
print("### decliff: out-ONLY contrast (both legs at rate 4, L=2048, same job/server)")
for ph,label in (("A","out=32  @rate4"),("B","out=512 @rate4")):
    key=4.0
    m={a: float(np.mean([x[0] for x in rows[(ph,a,key)]])) for a in ("d16","d44","d54")}
    sd={a: float(np.std([x[0] for x in rows[(ph,a,key)]],ddof=1)) for a in ("d16","d44","d54")}
    nn={a: len(rows[(ph,a,key)]) for a in ("d16","d44","d54")}
    print(f"  {label}: " + "  ".join(f"{a}={m[a]:6.2f}+-{sd[a]:.2f}(n={nn[a]})" for a in m)
          + f"   plateau(5pp)={plateau(m)}  width {len(plateau(m))}/3")
print("\n### the same contrast in g2_0_full is CONFOUNDED (out 32->512 AND rate 5->4)")
print("    g2_0_full A: random_output_len=32,  request_rate=5.0")
print("    g2_0_full B: random_output_len=512, request_rate=4.0")
print("\n### decliff phase A by rate (the campaign's own de-cliff selection kept ONLY rate 2)")
for key in (2.0,3.0,3.5,4.0):
    m={a: float(np.mean([x[0] for x in rows[("A",a,key)]])) for a in ("d16","d44","d54")}
    anom={a: sum(1 for x in rows[("A",a,key)] if x[1]<100) for a in ("d16","d44","d54")}
    nn={a: len(rows[("A",a,key)]) for a in ("d16","d44","d54")}
    print(f"  rateA={key}: " + "  ".join(f"{a}={m[a]:6.2f}(n={nn[a]},anom={anom[a]})" for a in m)
          + f"  plateau={plateau(m)}")
