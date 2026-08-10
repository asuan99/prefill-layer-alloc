#!/usr/bin/env python3
"""G5 (HOLB observer-effect) RE-SCORING with equivalence testing (TOST).

2026-08-09, result-analyst.  This is a RE-SCORE of already-collected data, not a
new measurement and not a new performance verdict.

WHY
---
The original G5 rule (`g2_holb_analyze.py:g5_verdict`) is
    PASS = |point estimate| < 3%  AND  95% CI contains 0
which is a NULL-ACCEPTANCE rule: a noisy cell gets a wide CI, the CI then
contains 0, and the cell PASSES *because* it is noisy.  Symmetrically a cell can
be recorded FAIL purely for lack of power.  This is the same error class that
killed rev1's R1/R3 in the Gate 2 design audit.

This script re-scores the SAME raw data with an explicit three-state verdict:
    EXCEEDANCE   -- |mu| > delta is statistically supported
    EQUIVALENT   -- TOST fires: |mu| < delta is statistically supported
    UNDETERMINED -- neither; the cell had no power to answer the question

TOOLS (no re-implementation)
----------------------------
  g2_holb_analyze.py   : collect_phase_b, paired_diffs_for_var, paired_t_ci
                         (canonical G5 input parsing + the 95% paired t-CI)
  g2_analyze.py        : tost_equivalence (canonical Gate-2 TOST, 90% CI in +-delta)
  verify_g2_tau40_lib.py : tost_power (canonical TOST power simulator)

METHODOLOGY GATE #14: n = 5 per cell (<= 8) -> bootstrap is FORBIDDEN; the
paired t-CI is used throughout.  No bootstrap is called anywhere in this file.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VERIFY = HERE.parent / "verify"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(VERIFY))

import g2_holb_analyze as HOLB          # canonical G5 parsing + paired t-CI
from g2_analyze import tost_equivalence, t_ppf, t_cdf   # canonical Gate-2 TOST
from verify_g2_tau40_lib import tost_power              # canonical TOST power

DELTA = 0.03          # PRE-REGISTERED margin, DIRECT_BLOCKING_DESIGN.md section 9
ALPHA = 0.05          # per side
ARMS = ["plain", "plainaux", "chunk512", "agnostic"]
NREP = 5

JOBS = [
    # (jobid, tag, model, rate, node)
    ("874602", "granite", "ibm-granite/granite-4.0-h-micro-base", 4.0, "gpu39"),
    ("874633", "granite", "ibm-granite/granite-4.0-h-micro-base", 4.0, "gpu37"),
    ("874635", "zamba2", "Zyphra/Zamba2-2.7B", 3.0, "gpu38"),
]


def exceedance_test(diffs, delta=DELTA, alpha=ALPHA):
    """One-sided test of H0: |mu| <= delta against the data-side alternative.

    Rejection at alpha per side is exactly the mirror image of TOST: the 90%
    two-sided t interval lies ENTIRELY OUTSIDE [-delta, +delta].
    Also reports the strictly-conservative 95% two-sided version.
    """
    n = len(diffs)
    if n < 2:
        return {"n": n, "verdict": "INSUFFICIENT_N"}
    mean = statistics.mean(diffs)
    sd = statistics.stdev(diffs)
    if sd == 0.0:
        return {"n": n, "mean": mean, "sd": 0.0, "verdict": "DEGENERATE-T"}
    se = sd / math.sqrt(n)
    df = n - 1
    lo90 = mean - t_ppf(1 - alpha, df) * se
    hi90 = mean + t_ppf(1 - alpha, df) * se
    lo95 = mean - t_ppf(0.975, df) * se
    hi95 = mean + t_ppf(0.975, df) * se
    p_up = 1.0 - t_cdf((mean - delta) / se, df)     # H0: mu <= +delta
    p_dn = t_cdf((mean + delta) / se, df)           # H0: mu >= -delta
    p_exceed = min(p_up, p_dn)
    exceed90 = (lo90 > delta) or (hi90 < -delta)
    exceed95 = (lo95 > delta) or (hi95 < -delta)
    return {"n": n, "mean": mean, "sd": sd, "se": se,
            "ci90_lo": lo90, "ci90_hi": hi90, "ci95_lo": lo95, "ci95_hi": hi95,
            "p_exceed": p_exceed, "exceed_alpha05": exceed90, "exceed_conservative": exceed95}


def delta_min_equivalence(diffs, alpha=ALPHA):
    """Smallest margin at which TOST WOULD have fired for this cell:
    delta_min = |mean| + t_{1-alpha,df} * se .  Purely descriptive."""
    n = len(diffs)
    if n < 2:
        return None
    mean = statistics.mean(diffs)
    sd = statistics.stdev(diffs)
    if sd == 0.0:
        return abs(mean)
    se = sd / math.sqrt(n)
    return abs(mean) + t_ppf(1 - alpha, n - 1) * se


def sd_max_for_tost(n, delta=DELTA, alpha=ALPHA):
    """Deterministic detectability ceiling: with a PERFECTLY centred estimate
    (mean = 0) the TOST 90% interval is +- t*sd/sqrt(n).  Equivalence at margin
    delta is ARITHMETICALLY IMPOSSIBLE at this n once sd >= this value."""
    return delta * math.sqrt(n) / t_ppf(1 - alpha, n - 1)


def n_for_power(sigma, target=0.80, delta=DELTA, nmax=4096, trials=8000):
    """Smallest n whose TOST power at true gap 0 reaches `target` (Gaussian).

    Power is monotone in n here, so a geometric bracket + bisection is used
    instead of a linear scan (same canonical `tost_power` estimator)."""
    if sigma <= 0:
        return 2
    lo, hi = 3, 4
    while hi <= nmax and tost_power(sigma, hi, delta, 0.0, trials=trials) < target:
        lo, hi = hi, hi * 2
    if hi > nmax:
        return None
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if tost_power(sigma, mid, delta, 0.0, trials=trials) >= target:
            hi = mid
        else:
            lo = mid
    return hi


def holm(pairs):
    m = len(pairs)
    order = sorted(range(m), key=lambda i: pairs[i][1])
    adj = [None] * m
    run = 0.0
    for rank, idx in enumerate(order):
        _, p = pairs[idx]
        a = min(1.0, (m - rank) * p)
        run = max(run, a)
        adj[idx] = run
    return [(pairs[i][0], pairs[i][1], adj[i]) for i in range(m)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(HERE))
    ap.add_argument("--json-out", default=str(HERE / "g2holb_g5_tost_rescore_2026-08-09.json"))
    args = ap.parse_args()
    outdir = Path(args.outdir)

    report = {
        "created": "2026-08-09",
        "kind": "RE-SCORE of existing G5 data (not a new measurement)",
        "delta_prereg": DELTA,
        "alpha_per_side": ALPHA,
        "n_per_cell": NREP,
        "statistic": "paired t (methodology gate #14: n<=8 -> no bootstrap)",
        "jobs": {},
    }

    ceiling = sd_max_for_tost(NREP)
    print("=" * 100)
    print("G5 OBSERVER-EFFECT RE-SCORE -- TOST equivalence at the pre-registered margin")
    print(f"  delta = +-{DELTA*100:.0f}%   alpha = {ALPHA} per side   n = {NREP} paired per cell")
    print(f"  statistic = paired t (methodology gate #14: n<=8 forbids bootstrap)")
    print(f"  ARITHMETIC CEILING: at n={NREP}, TOST cannot fire unless the rep-to-rep")
    print(f"  SD of the paired relative diff is < {ceiling*100:.3f}%  (even with mean exactly 0).")
    print("=" * 100)

    grand = {"EXCEEDANCE": 0, "EQUIVALENT": 0, "UNDETERMINED": 0, "DEGENERATE": 0}

    for jobid, tag, model, rate, node in JOBS:
        per_arm = HOLB.collect_phase_b(outdir, tag, jobid, ARMS, NREP)
        jrep = {"tag": tag, "model": model, "rate": rate, "node": node, "cells": {}}
        print(f"\n### job {jobid}  tag={tag}  model={model}  rate={rate}  node={node}")
        print(f"{'arm':<10} {'var':<20} {'n':>2} {'mean%':>9} {'SD%':>8} "
              f"{'95%CI':>22} {'90%CI':>22} {'p_TOST':>8} {'p_exc':>8} "
              f"{'dmin%':>8} {'pwr0':>6}  {'OLD':<5} {'NEW'}")
        pex_pairs = []
        for arm in ARMS:
            for var in HOLB.RESPONSE_VARS:
                diffs, reps = HOLB.paired_diffs_for_var(per_arm[arm], NREP, var)
                if len(diffs) < 2:
                    continue
                tci = HOLB.paired_t_ci(diffs)                 # canonical 95% t-CI
                old = HOLB.g5_verdict(tci)                    # ORIGINAL G5 rule
                tost = tost_equivalence(diffs, DELTA, ALPHA)  # canonical TOST
                exc = exceedance_test(diffs, DELTA, ALPHA)
                dmin = delta_min_equivalence(diffs)
                sd = tci["sd"]
                if sd == 0.0:
                    new = "DEGENERATE"
                    pwr = None
                    nneed = None
                else:
                    pwr = tost_power(sd, NREP, DELTA, 0.0, trials=20000)
                    nneed = n_for_power(sd)
                    if exc["exceed_alpha05"]:
                        new = "EXCEEDANCE"
                    elif tost["equivalent"]:
                        new = "EQUIVALENT"
                    else:
                        new = "UNDETERMINED"
                grand[new] += 1
                pex_pairs.append((f"{arm}/{var}", exc.get("p_exceed", 1.0)))
                print(f"{arm:<10} {var:<20} {len(diffs):>2} {tci['mean']*100:>+9.2f} {sd*100:>8.2f} "
                      f"[{tci['ci_lo']*100:>+8.2f},{tci['ci_hi']*100:>+8.2f}] "
                      f"[{tost['ci90_lo']*100:>+8.2f},{tost['ci90_hi']*100:>+8.2f}] "
                      f"{tost.get('p_tost', float('nan')):>8.3f} {exc.get('p_exceed', float('nan')):>8.3f} "
                      f"{(dmin*100 if dmin is not None else float('nan')):>8.2f} "
                      f"{(pwr if pwr is not None else float('nan')):>6.3f}  {old:<5} {new}")
                jrep["cells"][f"{arm}/{var}"] = {
                    "n": len(diffs), "reps": reps,
                    "mean": tci["mean"], "sd": sd, "se": tci["se"],
                    "ci95": [tci["ci_lo"], tci["ci_hi"]],
                    "ci90": [tost.get("ci90_lo"), tost.get("ci90_hi")],
                    "p_tost": tost.get("p_tost"), "p_exceed": exc.get("p_exceed"),
                    "delta_min_equivalence": dmin,
                    "tost_power_at_gap0_with_observed_sd": pwr,
                    "n_needed_for_80pct_power": nneed,
                    "old_g5_verdict": old, "rescored_verdict": new,
                }
        adj = holm(pex_pairs)
        n_exc_holm = sum(1 for _, _, a in adj if a < ALPHA)
        jrep["holm_exceedance_significant"] = n_exc_holm
        jrep["min_holm_adj_p_exceed"] = min(a for _, _, a in adj) if adj else None
        print(f"  -> Holm-adjusted over {len(adj)} cells: exceedance significant in "
              f"{n_exc_holm} cells (min adj p = {jrep['min_holm_adj_p_exceed']:.3f})")
        report["jobs"][jobid] = jrep

    report["grand_totals"] = grand
    report["sd_ceiling_for_tost_at_n5"] = ceiling
    print("\n" + "=" * 100)
    print(f"GRAND TOTALS over {sum(grand.values())} arm x var cells (3 jobs x 4 arms x 6 vars):")
    for k, v in grand.items():
        print(f"  {k:<14} {v}")
    print("=" * 100)

    Path(args.json_out).write_text(json.dumps(report, indent=2))
    print(f"\nJSON -> {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
