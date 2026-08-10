#!/usr/bin/env python3
"""Summary roll-up + power back-calculation for the 2026-08-09 G5 TOST re-score.

Reads g2holb_g5_tost_rescore_2026-08-09.json (produced by g2holb_g5_tost_rescore.py)
and answers: (a) how many cells changed verdict and in which direction, (b) was the
n=5 design ever able to discriminate the pre-registered +-3% margin, (c) what n
would have been required.

Also runs an EXPLORATORY pooled-granite (874602 + 874633) t-CI/TOST.  This is
POST-HOC, not pre-registered, and the two jobs re-use the SAME 5 seeds
(SEED = 9000 + 100*rep + rate, rate=4 in both), so the pooled n=10 is 10 machine
draws over only 5 workload realisations -- it is NOT 10 independent replicates.
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
VERIFY = HERE.parent / "verify"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(VERIFY))

import g2_holb_analyze as HOLB
from g2_analyze import tost_equivalence, t_ppf
from g2holb_g5_tost_rescore import exceedance_test, sd_max_for_tost, n_for_power, DELTA, ARMS, NREP

REP = json.loads((HERE / "g2holb_g5_tost_rescore_2026-08-09.json").read_text())

print("=" * 96)
print("G5 RE-SCORE ROLL-UP  (2026-08-09)")
print("=" * 96)

trans = defaultdict(int)
by_var = defaultdict(lambda: defaultdict(int))
sd_by_var = defaultdict(list)
nneed = defaultdict(list)
pwr_all = []

for jobid, j in REP["jobs"].items():
    for cell, c in j["cells"].items():
        arm, var = cell.split("/")
        trans[(c["old_g5_verdict"], c["rescored_verdict"])] += 1
        by_var[var][c["rescored_verdict"]] += 1
        sd_by_var[var].append(c["sd"])
        if c["n_needed_for_80pct_power"] is not None:
            nneed[var].append(c["n_needed_for_80pct_power"])
        if c["tost_power_at_gap0_with_observed_sd"] is not None:
            pwr_all.append(c["tost_power_at_gap0_with_observed_sd"])

print("\n--- verdict transition table (old G5 rule -> TOST re-score), 72 cells ---")
for (o, n), k in sorted(trans.items(), key=lambda kv: -kv[1]):
    print(f"  {o:<6} -> {n:<13} {k:>3}")

print("\n--- by response variable (12 cells each: 3 jobs x 4 arms) ---")
print(f"{'var':<20} {'EQUIV':>6} {'UNDET':>6} {'EXCEED':>7} {'medianSD%':>10} {'medianN80':>10}")
for var in HOLB.RESPONSE_VARS:
    d = by_var[var]
    med_sd = statistics.median(sd_by_var[var]) * 100
    med_n = statistics.median(nneed[var]) if nneed[var] else float("nan")
    print(f"{var:<20} {d['EQUIVALENT']:>6} {d['UNDETERMINED']:>6} {d['EXCEEDANCE']:>7} "
          f"{med_sd:>10.2f} {med_n:>10.0f}")

ceil5 = sd_max_for_tost(5)
n_over = sum(1 for v in sd_by_var.values() for sd in v if sd >= ceil5)
print(f"\n--- design-level answerability (methodology gate #14: n=5 -> paired t, no bootstrap) ---")
print(f"  arithmetic ceiling at n=5, delta=3%: TOST is IMPOSSIBLE once SD >= {ceil5*100:.3f}%")
print(f"  cells whose observed SD is at/over that ceiling: {n_over}/72  "
      f"({n_over/72*100:.0f}% could not have passed no matter how well centred)")
print(f"  TOST power at true gap 0 with each cell's own observed SD: "
      f"median {statistics.median(pwr_all):.3f}, mean {statistics.mean(pwr_all):.3f}, "
      f"cells with power >= 0.80: {sum(1 for p in pwr_all if p >= 0.8)}/72")
print(f"  NOTE: SD is itself estimated from n=5 (df=4).  Its own 95% interval is "
      f"[{math.sqrt(4/11.143):.2f}x, {math.sqrt(4/0.4844):.2f}x] the point value, so every "
      f"n-requirement below is order-of-magnitude only.")

print("\n--- n required for 80% TOST power at delta=3%, true gap 0 (Gaussian, observed SD) ---")
allneed = [x for v in nneed.values() for x in v]
allneed.sort()
print(f"  min {allneed[0]}   q1 {allneed[len(allneed)//4]}   median {allneed[len(allneed)//2]}   "
      f"q3 {allneed[3*len(allneed)//4]}   max {allneed[-1]}")

# ---------------------------------------------------------------- pooled granite
print("\n" + "=" * 96)
print("EXPLORATORY (post-hoc, NOT pre-registered): granite 874602 + 874633 pooled, n=10")
print("  caveat: same 5 seeds in both jobs -> 10 machine draws over 5 workload draws.")
print("=" * 96)
pooled_counts = defaultdict(int)
print(f"{'arm':<10} {'var':<20} {'n':>2} {'mean%':>9} {'SD%':>8} {'90%CI':>22} {'p_TOST':>8} {'p_exc':>8}  VERDICT")
for arm in ARMS:
    for var in HOLB.RESPONSE_VARS:
        diffs = []
        for jobid, tag in (("874602", "granite"), ("874633", "granite")):
            per_arm = HOLB.collect_phase_b(HERE, tag, jobid, ARMS, NREP)
            d, _ = HOLB.paired_diffs_for_var(per_arm[arm], NREP, var)
            diffs.extend(d)
        if len(diffs) < 2:
            continue
        tost = tost_equivalence(diffs, DELTA, 0.05)
        exc = exceedance_test(diffs, DELTA, 0.05)
        if exc["exceed_alpha05"]:
            v = "EXCEEDANCE"
        elif tost["equivalent"]:
            v = "EQUIVALENT"
        else:
            v = "UNDETERMINED"
        pooled_counts[v] += 1
        print(f"{arm:<10} {var:<20} {len(diffs):>2} {tost['mean']*100:>+9.2f} {tost['sd']*100:>8.2f} "
              f"[{tost['ci90_lo']*100:>+8.2f},{tost['ci90_hi']*100:>+8.2f}] "
              f"{tost['p_tost']:>8.3f} {exc['p_exceed']:>8.3f}  {v}")
print(f"\n  pooled granite totals: {dict(pooled_counts)}")
print(f"  arithmetic ceiling at n=10: SD < {sd_max_for_tost(10)*100:.3f}%")
