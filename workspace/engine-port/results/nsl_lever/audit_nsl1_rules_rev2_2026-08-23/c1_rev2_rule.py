#!/usr/bin/env python3
"""C1 -- simulate the rev2 sec4.1 decision rule EXACTLY as written.

Rule (DESIGN rev2 sec4.1):
  Delta(c) = goodput(D*, c) - goodput(D*, 48), c in {24, 96}, PAIRED over jobs
  two-sided t, Bonferroni alpha/2
  MATTERS      : some c has corrected CI excluding 0
  INERT        : BOTH c have CI inside +-delta        (TOST as written: CI-in-margin)
  UNDETERMINED : otherwise
No priority among the verdicts is registered.
"""
import numpy as np
from scipy.stats import t as tdist

rng = np.random.default_rng(20260824)
REPS = 200000
ALPHA = 0.05
NC = 2                       # two contrasts
GP = 70.0                    # design's assumed level
DELTA = 0.03 * GP            # equivalence margin = 3% relative = 2.1 pp


def run(true_d24, true_d96, n, sd_diff, gp=GP, delta=DELTA, reps=REPS):
    crit = tdist.ppf(1 - (ALPHA / NC) / 2, n - 1)
    out = {"MATTERS": 0, "INERT": 0, "BOTH": 0, "UNDET": 0,
           "dhat24": [], "dneg": 0}
    d = {}
    for name, mu in (("24", true_d24), ("96", true_d96)):
        x = mu + rng.normal(0, sd_diff, size=(reps, n))
        m = x.mean(axis=1)
        s = x.std(axis=1, ddof=1)
        hw = crit * s / np.sqrt(n)
        d[name] = (m, hw)
    m24, hw24 = d["24"]; m96, hw96 = d["96"]
    excl24 = np.abs(m24) > hw24
    excl96 = np.abs(m96) > hw96
    matters = excl24 | excl96
    inert = ((np.abs(m24) + hw24) < delta) & ((np.abs(m96) + hw96) < delta)
    both = matters & inert
    undet = ~matters & ~inert
    return (matters.mean(), inert.mean(), both.mean(), undet.mean(),
            float(np.mean(m24)), float(np.mean(m24 < 0)))


print(f"delta (equivalence margin) = {DELTA:.3f} pp   alpha/contrast={ALPHA/NC}")
print("\n== H0: cap COMPLETELY inert (true Delta = 0 for both contrasts) ==")
print(f"{'n':>3} {'sd_diff':>8} | {'P(MATTERS)':>10} {'P(INERT)':>9} "
      f"{'P(BOTH!)':>9} {'P(UNDET)':>9} | {'E[dhat24]':>10} {'P(dhat<0)':>10}")
for n in (4, 5, 8, 12):
    for sd in (0.5, 1.0, 2.0, 4.8):
        r = run(0.0, 0.0, n, sd)
        print(f"{n:>3} {sd:>8.2f} | {r[0]:>10.4f} {r[1]:>9.4f} {r[2]:>9.4f} "
              f"{r[3]:>9.4f} | {r[4]:>+10.4f} {r[5]:>10.3f}")

print("\n== H1: cap matters by exactly the registered target (+2.1pp at cap=96) ==")
for n in (4, 5, 8, 12):
    for sd in (0.5, 1.0, 2.0):
        r = run(0.0, 2.1, n, sd)
        print(f"{n:>3} {sd:>8.2f} | P(MATTERS)={r[0]:.4f} P(INERT)={r[1]:.4f} "
              f"P(BOTH)={r[2]:.4f} P(UNDET)={r[3]:.4f}")

print("\n== H1': cap matters by HALF the target (+1.05pp at cap=96) -- the overlap band ==")
for n in (4, 5, 8, 12, 20):
    for sd in (0.5, 1.0):
        r = run(0.0, 1.05, n, sd)
        print(f"{n:>3} {sd:>8.2f} | P(MATTERS)={r[0]:.4f} P(INERT)={r[1]:.4f} "
              f"★P(BOTH=ambiguous)={r[2]:.4f} P(UNDET)={r[3]:.4f}")
