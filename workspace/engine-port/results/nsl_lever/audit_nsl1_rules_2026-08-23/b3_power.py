#!/usr/bin/env python3
"""B3 -- independent re-derivation of NSL-1 sec4.4, plus what it left out."""
import math
from scipy.stats import nct, t as tdist

A, PW = 0.05, 0.80

def pw_unpaired(n, sd, delta):
    df, ncp = 2*n-2, delta/(sd*math.sqrt(2.0/n))
    c = tdist.ppf(1-A/2, df)
    return 1 - nct.cdf(c, df, ncp) + nct.cdf(-c, df, ncp)

def pw_paired(n, sd_diff, delta):
    df, ncp = n-1, delta/(sd_diff/math.sqrt(n))
    c = tdist.ppf(1-A/2, df)
    return 1 - nct.cdf(c, df, ncp) + nct.cdf(-c, df, ncp)

def detect_sd(n, gp, powfun, rel=0.03):
    lo, hi = 1e-4, 100.0
    for _ in range(200):
        m = (lo+hi)/2
        if powfun(n, m, rel*gp) >= PW: lo = m
        else: hi = m
    return lo

print("A) reproduce the design's table (unpaired, exact nct):")
for gp in (70.0, 50.0):
    row = {n: round(detect_sd(n, gp, pw_unpaired), 3) for n in (4,6,8,12)}
    print(f"   goodput={gp:.0f}pp  detectable SD (pp): {row}")

print("\nB) if the SAME cells were BLOCKED on job (paired), sd_diff = sqrt(2)*sd_resid:")
for gp in (70.0, 50.0):
    row = {}
    for n in (4,6,8,12):
        sdd = detect_sd(n, gp, pw_paired)
        row[n] = round(sdd/math.sqrt(2), 3)      # per-cell residual sd equivalent
    print(f"   goodput={gp:.0f}pp  tolerable RESIDUAL SD (pp) if job is a block: {row}")

print("\nC) n required at the repo's OWN measured goodput-fraction SDs")
print("   (interactive_slo_retune_plan.md:155-158, n=4, chat 300/50, goodput %):")
for name, sd in (("d44", 4.8), ("d34", 3.9), ("bind+GATE", 2.7), ("bind", 0.4)):
    n = next((k for k in range(2, 400) if pw_unpaired(k, sd, 0.03*70) >= PW), None)
    hrs = None if n is None else round(9*n*145.8/3600, 2)
    print(f"   sd={sd:4.1f}pp ({name:9s}) -> n/cell={n}, boot-only GPU-hr(9 cells)={hrs}")

print("\nD) multiplicity: the design's power is for ONE pairwise test, but the")
print("   estimand is an argmax over 9 cells.  Bonferroni over the 8 vs-best")
print("   comparisons (or the 6 cap-contrasts) at alpha=0.05:")
for k, lbl in ((6, "6 cap contrasts"), (8, "8 vs-best")):
    for gp in (70.0,):
        lo, hi = 1e-4, 100.0
        for _ in range(200):
            m = (lo+hi)/2
            df, ncp = 2*4-2, 0.03*gp/(m*math.sqrt(2.0/4))
            c = tdist.ppf(1-(A/k)/2, df)
            p = 1 - nct.cdf(c, df, ncp) + nct.cdf(-c, df, ncp)
            if p >= PW: lo = m
            else: hi = m
        print(f"   {lbl}: n=4 tolerable SD = {lo:.3f} pp   (design claims 0.882)")
