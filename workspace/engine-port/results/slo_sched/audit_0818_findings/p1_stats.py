#!/usr/bin/env python3
"""AUDIT P-1 statistics: paired/unpaired CI on the alleged d34 valley."""
import json, math, itertools
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
d = json.load(open(HERE / "p1_valley.json"))

def tcrit(df):  # 95% two-sided t quantiles
    tab = {1:12.706,2:4.303,3:3.182,4:2.776,5:2.571,6:2.447,7:2.365,8:2.306,
           9:2.262,10:2.228,11:2.201,12:2.179,15:2.131,20:2.086,30:2.042}
    return tab.get(df, 1.96)

print("### Phase A, g2_0_full, n_req=32/cell, n_rep=4, canonical (3000ms, 60ms)")
A = d["A"]
arms = list(A)
for a in arms:
    v = np.array(A[a]["vals"]); n = len(v)
    se = v.std(ddof=1)/math.sqrt(n)
    print(f"  {a}: mean={v.mean():6.2f} sd={v.std(ddof=1):6.2f} "
          f"t{n-1}-CI=[{v.mean()-tcrit(n-1)*se:7.2f},{v.mean()+tcrit(n-1)*se:7.2f}] vals={list(v)}")

print("\n### d34 vs d44 (the 33.6 pp valley)")
x = np.array(A["d34"]["vals"]); y = np.array(A["d44"]["vals"])
diff = y - x                      # 'paired' by rep index (see caveat)
n = len(diff); sd = diff.std(ddof=1); se = sd/math.sqrt(n)
t = diff.mean()/se
print(f"  paired-by-rep-INDEX: mean_diff={diff.mean():.2f} sd={sd:.2f} t({n-1})={t:.3f} "
      f"CI=[{diff.mean()-tcrit(n-1)*se:.2f},{diff.mean()+tcrit(n-1)*se:.2f}]  diffs={list(diff)}")
# Welch
sx, sy = x.std(ddof=1), y.std(ddof=1)
sew = math.sqrt(sx**2/len(x) + sy**2/len(y))
dfw = (sx**2/len(x)+sy**2/len(y))**2 / ((sx**2/len(x))**2/(len(x)-1) + (sy**2/len(y))**2/(len(y)-1)) if sy>0 else len(x)-1
tw = (y.mean()-x.mean())/sew
print(f"  Welch: t={tw:.3f} df={dfw:.2f} CI=[{(y.mean()-x.mean())-tcrit(int(round(dfw)))*sew:.2f},"
      f"{(y.mean()-x.mean())+tcrit(int(round(dfw)))*sew:.2f}]")
# exact permutation (unpaired, n=4 vs 4)
pool = np.concatenate([x, y]); obs = abs(y.mean()-x.mean()); cnt = 0; tot = 0
for idx in itertools.combinations(range(8), 4):
    g1 = pool[list(idx)]; g2 = pool[[i for i in range(8) if i not in idx]]
    tot += 1
    if abs(g2.mean()-g1.mean()) >= obs - 1e-9: cnt += 1
print(f"  exact 2-sided permutation p = {cnt}/{tot} = {cnt/tot:.4f}")
# sign test paired
pos = int((diff > 0).sum()); ties = int((diff == 0).sum())
print(f"  sign test on rep-index pairs: {pos} positive, {ties} ties of {n} -> p >= 0.25 (min attainable with n=4)")

print("\n### run-level anomaly incidence (defined by TTFT pass < 100%, an arm-independent symptom)")
anom = {}
for a in arms:
    rows = A[a]["per_rep"]
    anom[a] = [r["ttft_pct"] < 100.0 for r in rows]
    print(f"  {a}: ttft_pass% = {[round(r['ttft_pct'],1) for r in rows]}  ttft_p50 = {[round(r['ttft_p50']) for r in rows]}"
          f"  dur = {[round(r['dur'],2) for r in rows]}")
tot_anom = sum(sum(v) for v in anom.values())
print(f"  total anomalous runs = {tot_anom} / 20")
# hypergeometric: P(a GIVEN arm of 4 runs gets >= 2 of the 4 anomalies)
from math import comb
p_ge2 = sum(comb(4,k)*comb(16,4-k) for k in (2,3,4))/comb(20,4)
print(f"  P(one prespecified arm draws >=2 of the 4 anomalies) = {p_ge2:.4f}")
print(f"  P(at least one of 5 arms draws >=2)  ~ {1-(1-p_ge2)**5:.4f} (indep. approx.)")

print("\n### the two non-anomalous d34 reps vs d44")
good34 = [r["joint_pct"] for r in A["d34"]["per_rep"] if r["ttft_pct"] == 100.0]
print(f"  d34 non-anomalous reps: {good34}  vs d44 all reps: {A['d44']['vals']}")

print("\n### mechanism check: is the phase-A landscape monotone in the sub-metrics?")
for a in arms:
    rows = A[a]["per_rep"]
    good = [r for r in rows if r["ttft_pct"] == 100.0]
    print(f"  {a}: (non-anomalous reps only) ttft_p50={np.mean([r['ttft_p50'] for r in good]):8.1f} ms  "
          f"itl_p50={np.mean([r['itl_p50'] for r in good]):6.2f} ms  n={len(good)}")
