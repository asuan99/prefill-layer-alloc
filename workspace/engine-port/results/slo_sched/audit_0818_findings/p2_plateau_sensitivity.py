#!/usr/bin/env python3
"""AUDIT P-2 / probe 3: is the '5 pp plateau' threshold a post-hoc knob, and
does the phase-A plateau survive (a) threshold changes (b) removal of the
5 run-level TTFT anomalies (c) rep-level uncertainty?"""
import json, math, sys, re
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=Path("/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port")
sys.path.insert(0,str(ROOT/"benchmarks"))
from pdmux_eval.analyze import load_bench_serving_rounds, percentile
D=json.load(open(HERE/"p1_valley.json"))
ARMS=["d16","d24","d34","d44","d54"]

def plateau(means, thr):
    m=max(means.values())
    return [a for a in means if means[a] >= m-thr]

print("### g2_0_full: plateau membership vs threshold, three aggregations")
for ph in ("A","B"):
    print(f"\n phase {ph}")
    agg={}
    agg["mean(all 4 reps)   "] = {a: float(np.mean(D[ph][a]["vals"])) for a in ARMS}
    agg["median(all 4 reps) "] = {a: float(np.median(D[ph][a]["vals"])) for a in ARMS}
    clean={}
    for a in ARMS:
        v=[r["joint_pct"] for r in D[ph][a]["per_rep"] if r["ttft_pct"]==100.0]
        clean[a]=float(np.mean(v)) if v else float('nan')
    agg["mean(non-anom reps)"]=clean
    for label,m in agg.items():
        print(f"  {label}: " + "  ".join(f"{a}={m[a]:6.2f}" for a in ARMS))
        for thr in (3,5,7,10):
            pl=plateau(m,thr)
            print(f"      thr={thr:2} pp -> width {len(pl)}/5  {sorted(pl)}")

print("\n### rep-level uncertainty: bootstrap over reps, P(arm is in the 5pp plateau)")
rng=np.random.default_rng(20260819)
for ph in ("A","B"):
    cnt={a:0 for a in ARMS}; B=20000
    vals={a:np.array(D[ph][a]["vals"]) for a in ARMS}
    for _ in range(B):
        m={a: float(vals[a][rng.integers(0,4,4)].mean()) for a in ARMS}
        for a in plateau(m,5): cnt[a]+=1
    print(f"  phase {ph}: " + "  ".join(f"{a}={cnt[a]/B:.3f}" for a in ARMS))

print("\n### which arms are pairwise SEPARABLE at n=4 (t(3), unpaired Welch-ish)?")
def tcrit(df): return {1:12.706,2:4.303,3:3.182,4:2.776,5:2.571,6:2.447}.get(df,1.96)
for ph in ("A","B"):
    print(f"  phase {ph}: pairs whose 95% CI on the difference EXCLUDES 0")
    found=[]
    for i,a in enumerate(ARMS):
        for b in ARMS[i+1:]:
            x=np.array(D[ph][a]["vals"]); y=np.array(D[ph][b]["vals"])
            sx,sy=x.std(ddof=1),y.std(ddof=1)
            se=math.sqrt(sx**2/4+sy**2/4)
            if se==0:
                if x.mean()!=y.mean(): found.append(f"{a}-{b} (both SD=0, diff={y.mean()-x.mean():+.2f})")
                continue
            df=(sx**2/4+sy**2/4)**2/((sx**2/4)**2/3+(sy**2/4)**2/3)
            lo=(y.mean()-x.mean())-tcrit(int(round(df)))*se; hi=(y.mean()-x.mean())+tcrit(int(round(df)))*se
            if lo*hi>0: found.append(f"{a}-{b} ({lo:+.1f},{hi:+.1f})")
    print("     " + ("; ".join(found) if found else "NONE"))

print("\n### decliff plateau, same treatment (n_rep=15 pooled over rateA)")
DC=ROOT/"results/g2_0_decliff"
rows={}
for p in sorted(DC.glob("g20dc[AB]_*.jsonl")):
    m=re.match(r"g20dc([AB])_(d\d+)_(\d+)_rA([\d.]+)B4_OB512_(\d+)\.jsonl",p.name)
    ph,arm,rep,rA,job=m.groups()
    reqs,dur=load_bench_serving_rounds(p); n=len(reqs)
    i95=[percentile(r.token_itl_ms,.95) if r.token_itl_ms else math.inf for r in reqs]
    j=[(r.ttft_ms<=3000) and (v<=60) for r,v in zip(reqs,i95)]
    tk=100*np.mean([r.ttft_ms<=3000 for r in reqs])
    rows.setdefault((ph,arm),[]).append((100*sum(j)/n, tk))
for ph in ("A","B"):
    m={a: float(np.mean([x[0] for x in rows[(ph,a)]])) for a in ("d16","d44","d54")}
    mc={a: float(np.mean([x[0] for x in rows[(ph,a)] if x[1]==100.0])) for a in ("d16","d44","d54")}
    print(f"  phase {ph} mean: " + "  ".join(f"{a}={m[a]:6.2f}" for a in m))
    print(f"  phase {ph} mean(non-anom): " + "  ".join(f"{a}={mc[a]:6.2f}" for a in mc))
    for thr in (3,5,7,10):
        print(f"     thr={thr:2} -> all-reps {len(plateau(m,thr))}/3 {sorted(plateau(m,thr))} | "
              f"non-anom {len(plateau(mc,thr))}/3 {sorted(plateau(mc,thr))}")
