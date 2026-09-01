#!/usr/bin/env python3
"""Aggregate a g1_probe_<job>/ directory into the numbers PREREG_G1_PROBE_2026-08-28.md
sec 2/3 needs -- NOT a label.  This script computes:

  * the full arm x rate x seed achieved/attainment table
  * per (arm, rate) seed SD of attainment (df = n_seeds - 1)
  * a candidate value for each cp0_g1_rule.py axis (coverage, band_vs_sd,
    grid_brackets) SO THE MAIN SESSION CAN FEED cp0_g1_rule.py -- this script
    does not import that rule and does not print a verdict, per
    OVERRIDE_P1_SUBMIT_2026-08-28.md ("라벨을 네가 판정하지 마라").

Usage: python3 g1_probe_analyze.py <g1_probe_dir> [--registered-grid 2 4 8]
"""
import argparse
import glob
import json
import os
import statistics
import sys


def load_boots(d):
    boots = []
    for f in sorted(glob.glob(os.path.join(d, "g1_*_s*.json"))):
        with open(f) as fh:
            boots.append(json.load(fh))
    return boots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--registered-grid", nargs="+", type=int, default=[2, 4, 8])
    args = ap.parse_args()

    boots = load_boots(args.dir)
    if not boots:
        print(f"NO_BOOT_FILES_FOUND in {args.dir}")
        sys.exit(1)

    arms = sorted({b["arm"] for b in boots})
    seeds = sorted({b["seed"] for b in boots})
    all_rates = sorted({r["rate"] for b in boots for r in b.get("rates", [])})

    # ---- full table -------------------------------------------------------
    print(f"boots found: {len(boots)}  arms={arms}  seeds={seeds}  rates={all_rates}")
    print()
    header = f"{'arm':<15}{'rate':>6}{'seed':>6}{'completed':>11}{'duration_s':>12}{'achieved_rps':>14}{'attainment':>12}{'extraction':>12}"
    print(header)
    print("-" * len(header))
    cells = {}  # (arm, rate) -> {seed: attainment}
    for b in boots:
        arm, seed = b["arm"], b["seed"]
        if not b.get("boot_ok", True):
            print(f"{arm:<15}{'--':>6}{seed:>6}  BOOT_FAILED")
            continue
        for r in b.get("rates", []):
            att = r.get("attainment")
            print(f"{arm:<15}{r['rate']:>6}{seed:>6}{str(r.get('completed')):>11}"
                  f"{str(round(r['duration'],2) if r.get('duration') else None):>12}"
                  f"{str(round(r['achieved_rps'],3) if r.get('achieved_rps') else None):>14}"
                  f"{str(round(att,4) if att is not None else None):>12}{r.get('extraction'):>12}")
            cells.setdefault((arm, r["rate"]), {})[seed] = att

    # ---- coverage -----------------------------------------------------------
    n_expected = len(arms) * len(all_rates) * len(seeds)
    n_present = sum(1 for v in cells.values() for a in v.values() if a is not None)
    n_slots = len(cells) * len(seeds)
    if n_present == 0:
        coverage = "absent"
    elif n_present < n_expected:
        coverage = "partial"
    else:
        coverage = "complete"
    print()
    print(f"coverage: {n_present}/{n_expected} cell-seed attainment values present -> "
          f"AXIS CANDIDATE coverage={coverage}")

    # ---- per (arm, rate) seed SD --------------------------------------------
    print()
    print(f"{'arm':<15}{'rate':>6}{'n_seeds':>9}{'mean_att':>12}{'sd_att':>10}   values")
    print("-" * 70)
    sd_by_cell = {}
    for (arm, rate), seed_map in sorted(cells.items()):
        vals = [v for v in seed_map.values() if v is not None]
        if len(vals) >= 2:
            sd = statistics.stdev(vals)
        elif len(vals) == 1:
            sd = float("nan")
        else:
            sd = None
        sd_by_cell[(arm, rate)] = sd
        mean = statistics.mean(vals) if vals else None
        print(f"{arm:<15}{rate:>6}{len(vals):>9}"
              f"{str(round(mean,4) if mean is not None else None):>12}"
              f"{str(round(sd,4) if isinstance(sd,float) and sd==sd else sd):>10}   {vals}")

    # ---- knee band per arm: last still-climbing rate vs first plateau rate --
    # NOT a judgement -- just prints the mean-attainment trend per arm across
    # the swept rates so a human/result-analyst can site the knee band by eye,
    # same convention as e1_capacity_scan.sbatch's printed-table pattern.
    print()
    print("mean attainment trend by arm (for siting the knee band by eye):")
    for arm in arms:
        trend = []
        for rate in all_rates:
            vals = [v for v in cells.get((arm, rate), {}).values() if v is not None]
            trend.append((rate, round(statistics.mean(vals), 4) if vals else None))
        print(f"  {arm}: {trend}")

    print()
    print("registered grid:", args.registered_grid)
    print("swept rates this probe:", all_rates)
    print()
    print("NOTE: band_vs_sd and grid_brackets axis values require a human/"
          "result-analyst to site the knee band on the printed trend above "
          "(prereg sec 2: \"knee band = attainment gap between the last "
          "still-climbing rate and the first plateau rate\") -- this script "
          "does not auto-site it or call cp0_g1_rule.py.")


if __name__ == "__main__":
    main()
