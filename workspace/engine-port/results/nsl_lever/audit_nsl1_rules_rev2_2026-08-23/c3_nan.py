#!/usr/bin/env python3
"""C3 -- nsl1_power.py silently swallows NaN from scipy's noncentral t.

`got >= power` is False when got is NaN, so a cell where scipy fails is treated
as "power not achieved".  n_required_* then skips it; detectable_sd_* bisects a
NON-MONOTONE predicate and can converge to the NaN boundary instead of the
power boundary.
"""
import math
import numpy as np
from scipy.stats import nct, t as tdist
from scipy.integrate import quad

ALPHA, NC, POWER = 0.05, 2, 0.80


def pw_scipy(n, sd, delta):
    df, ncp = n - 1, delta / (sd / math.sqrt(n))
    c = tdist.ppf(1 - (ALPHA / NC) / 2, df)
    return 1 - nct.cdf(c, df, ncp) + nct.cdf(-c, df, ncp)


def pw_exact(n, sd, delta):
    """P(|T|>c), T~nct(df,ncp), by conditioning on the chi2 scale. NaN-free."""
    from scipy.stats import chi2, norm
    df, ncp = n - 1, delta / (sd / math.sqrt(n))
    c = tdist.ppf(1 - (ALPHA / NC) / 2, df)
    f = lambda v: chi2.pdf(v, df) * (
        norm.sf(c * math.sqrt(v / df) - ncp) + norm.cdf(-c * math.sqrt(v / df) - ncp))
    return quad(f, 1e-9, df + 40 * math.sqrt(2 * df), limit=400)[0]


print("A) NaN map for the design's own paired power call (goodput=70pp, delta=2.1pp)")
print(f"{'sd':>6} " + " ".join(f"n={n:<12}" for n in (2, 3, 4, 5, 6)))
for sd in (0.2, 0.3, 0.4, 0.5, 0.7732, 1.0, 2.0):
    cells = []
    for n in (2, 3, 4, 5, 6):
        s = pw_scipy(n, sd, 2.1)
        cells.append(f"{'NaN':>6}/{pw_exact(n,sd,2.1):.4f}" if math.isnan(s)
                     else f"{s:.4f}/{pw_exact(n,sd,2.1):.4f}")
    print(f"{sd:>6.4f} " + " ".join(f"{c:<14}" for c in cells))

print("\nB) detectable_sd_diff_pp_at_n as PUBLISHED vs NaN-free recomputation")
PUB = {50.0: {4: 0.552, 6: 0.880, 8: 1.120, 12: 1.488, 20: 2.030},
       70.0: {4: 0.377, 6: 1.231, 8: 1.568, 12: 2.083, 20: 2.841}}
for gp, row in PUB.items():
    for n, pub in row.items():
        lo, hi = 1e-4, 50.0
        for _ in range(200):
            m = (lo + hi) / 2
            if pw_exact(n, m, 0.03 * gp) >= POWER: lo = m
            else: hi = m
        flag = "  <-- WRONG" if abs(lo - pub) > 0.02 else ""
        print(f"  goodput={gp:.0f}pp n={n:>2}: published={pub:.3f}  correct={lo:.3f}{flag}")

print("\nC) n_required_by_sd_diff as PUBLISHED vs NaN-free")
PUBN = {50.0: {0.5: 4, 1.0: 7, 2.0: 20, 3.4: 52, 4.8: 100, 6.8: 198},
        70.0: {0.5: 4, 1.0: 5, 2.0: 12, 3.4: 28, 4.8: 53, 6.8: 103}}
for gp, row in PUBN.items():
    for sd, pub in row.items():
        need = next((n for n in range(2, 400)
                     if pw_exact(n, sd, 0.03 * gp) >= POWER), None)
        flag = "  <-- differs" if need != pub else ""
        print(f"  goodput={gp:.0f}pp sd_diff={sd}: published n={pub:>3}  correct n={need:>3}{flag}")
