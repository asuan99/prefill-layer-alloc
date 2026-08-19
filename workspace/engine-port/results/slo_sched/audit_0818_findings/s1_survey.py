#!/usr/bin/env python3
"""AUDIT S-1 (fast, from cache): full enumeration + resampling robustness."""
import math, pickle, random, re
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
C = pickle.load(open(HERE/"itl95_cache.pkl","rb"))
ITL=60.0; LO,HI=54.0,66.0

def stats(vals):
    n=len(vals); fin=[v for v in vals if math.isfinite(v)]
    return dict(n=n, p50=float(np.median(fin)) if fin else float('nan'),
                fail=100*sum(v>ITL for v in vals)/n, band=100*sum(LO<=v<=HI for v in vals)/n)
def agg(keys):
    v=[x for k in keys for x in C[k]]
    return (stats(v) | {"files":len(keys)}) if v else None

camp={}
for k in C: camp.setdefault(k.split("/")[0],[]).append(k)

print("### FULL ENUMERATION (all bench jsonl in each results/<dir>)")
print(f"  {'campaign':22}{'files':>6}{'n_req':>9}{'p50':>8}{'fail%':>8}{'band%':>8}")
full={}
for k in sorted(camp):
    s=agg(camp[k]); full[k]=s
    print(f"  {k:22}{s['files']:6}{s['n']:9}{s['p50']:8.2f}{s['fail']:8.2f}{s['band']:8.2f}")

print("\n### slo_sched split by sub-campaign prefix (it is NOT G16-only)")
sub={}
for k in camp["slo_sched"]:
    n=Path(k).name; pref=re.sub(r"[0-9]{5,}","",n).split("_")[0]
    sub.setdefault(pref,[]).append(k)
print(f"  {'prefix':22}{'files':>6}{'n_req':>9}{'p50':>8}{'fail%':>8}{'band%':>8}")
for p,v in sorted(sub.items(), key=lambda x:-len(x[1])):
    s=agg(v)
    print(f"  {p:22}{s['files']:6}{s['n']:9}{s['p50']:8.2f}{s['fail']:8.2f}{s['band']:8.2f}")

print("\n### G16 4-block only, by phase")
g16=[k for k in camp["slo_sched"] if Path(k).name.startswith("g16_blk")]
for tag in ("HI","LO"):
    s=agg([k for k in g16 if k.endswith(f"_{tag}.jsonl")])
    print(f"  g16 {tag}: files={s['files']} n={s['n']} p50={s['p50']:.2f} fail%={s['fail']:.2f} band%={s['band']:.2f}")
s=agg(g16); print(f"  g16 LO+HI: files={s['files']} n={s['n']} p50={s['p50']:.2f} fail%={s['fail']:.2f} band%={s['band']:.2f}")

print("\n### RESAMPLING robustness: 12 random files/dir, 2000 seeds")
rng=random.Random(20260819)
print(f"  {'campaign':22}{'band%p05':>10}{'p50':>9}{'p95':>9}{'full':>9}{'  P(band>25%)':>13}")
for k in sorted(camp):
    v=camp[k]; b=[]
    for _ in range(2000):
        s=agg(rng.sample(v,min(12,len(v))))
        b.append(s['band'])
    b=np.array(b)
    print(f"  {k:22}{np.percentile(b,5):10.2f}{np.percentile(b,50):9.2f}{np.percentile(b,95):9.2f}{full[k]['band']:9.2f}{(b>25).mean():13.3f}")
