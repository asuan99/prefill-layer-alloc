#!/usr/bin/env python3
"""C2 -- the INERT arm has no power analysis. Compute what it actually needs.

DESIGN rev2 sec4.4 tabulates n for the SUPERIORITY test (detect a 3% difference).
The verdict the design says it most likely wants (INERT = "informative negative
that narrows sec5-8(b)") is an EQUIVALENCE claim requiring the corrected CI of
BOTH contrasts to lie inside +-delta.  Two conditions must hold jointly
(intersection-union), so joint power is roughly the square of one-contrast power.
"""
import numpy as np
from scipy.stats import t as tdist

rng = np.random.default_rng(7)
REPS = 100000
ALPHA, NC = 0.05, 2
GP = 70.0
DELTA = 0.03 * GP


def p_inert(n, sd, delta=DELTA, reps=REPS):
    crit = tdist.ppf(1 - (ALPHA / NC) / 2, n - 1)
    ok = np.ones(reps, dtype=bool)
    for _ in range(2):                       # two contrasts, true Delta = 0
        x = rng.normal(0, sd, size=(reps, n))
        m, s = x.mean(1), x.std(1, ddof=1)
        ok &= (np.abs(m) + crit * s / np.sqrt(n)) < delta
    return ok.mean()


print(f"delta={DELTA:.2f}pp  target joint P(INERT|H0) >= 0.80")
print(f"{'sd_diff':>8} | {'n from DESIGN sec4.4':>21} | {'P(INERT|H0) at that n':>22}"
      f" | {'n actually needed':>18}")
DESIGN = {0.5: 4, 1.0: 5, 2.0: 12, 3.4: 28, 4.8: 53, 6.8: 103}
for sd, nd in DESIGN.items():
    got = p_inert(nd, sd)
    need = None
    for n in range(3, 400):
        if p_inert(n, sd, reps=20000) >= 0.80:
            need = n
            break
    print(f"{sd:>8.1f} | {nd:>21} | {got:>22.3f} | {str(need):>18}")
