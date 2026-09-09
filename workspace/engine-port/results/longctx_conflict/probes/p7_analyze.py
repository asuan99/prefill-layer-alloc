#!/usr/bin/env python3
"""Analysis for PREREG_P7_ITL_SIGMA_2026-09-09.md.

PURE INSTRUMENTATION.  There is no decision rule, no pre-registered prediction
and no threshold in this file.  Every number below is REPORTED, never judged.

Design (prereg sec 2): 4 rounds x 3 arms = 12 boots; each boot runs 3 MAIN
benches (one per seed in that round's seed set) at the common W, then 1 SAT
bench (prereg sec 3.1) used only to observe mu_p.  Seeds are redrawn per round
and shared across arms within a round, so:

    within-boot spread across seeds  -> arrival-realisation variance (random)
    between-boot spread at matched   -> BOOT random effect
      seed set

which is the decomposition rev2 blocker C4 said a fixed shared seed set cannot
give.  One-way ANOVA estimators (boots = groups, seeds = replicates):

    sigma_seed^2 = mean_b Var_within(b)                    (unbiased, ddof=1)
    sigma_boot^2 = Var_between(boot means, ddof=1) - sigma_seed^2 / n_seed
                   (clipped at 0; the clip is reported when it fires)

Primary quantities per MAIN bench:
    itl95_i      = np.percentile(itls[i], 95) * 1000        (per REQUEST, ms)
    itl95_cell   = 4 estimators over the per-request values:
                     median / mean / 10% trimmed mean / pooled-p95
    L_decode     = sum_i sum(itls_i) / duration             (Little, integral)
    span_hat     = duration - (ttfts[-1] + sum(itls[-1]))   (arrival span)
    F(T)         = empirical TTFT CDF

TTFT stochastic dominance is evaluated over the FULL observed T range with NO
domain restriction -- the self-defining domain that killed P6 (death cause Y2)
is deliberately absent.  Violations are reported as intervals and magnitudes,
not as a verdict.

Usage: python3 p7_analyze.py <run_dir> [--out result.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
from pathlib import Path

import numpy as np

ARMS = (16, 44, 54)
# Step 1 (job 905958) measured mu_p under pdmux_homog6.yml -- carried in only
# to report the ratio; nothing here depends on it.
MU_P_S1 = {16: 0.914318261526629, 44: 0.6666328631373765, 54: 0.5772205386595464}


def load_bench(path):
    d = json.loads(Path(path).read_text().strip().split("\n")[0])
    ok = [i for i, e in enumerate(d["errors"]) if e == ""]
    ttfts = np.array([d["ttfts"][i] for i in ok], dtype=float)
    itls = [np.asarray(d["itls"][i], dtype=float) for i in ok]
    return d, ttfts, itls


def itl95_estimators(itls):
    v = np.array([np.percentile(x, 95) * 1000.0 for x in itls if x.size])
    s = np.sort(v)
    k = int(0.1 * s.size)
    trimmed = s[k:s.size - k] if s.size - 2 * k > 0 else s
    return {
        "median": float(np.median(v)),
        "mean": float(v.mean()),
        "trim10": float(trimmed.mean()),
        "pooled_p95": float(np.percentile(np.concatenate(itls), 95) * 1000.0),
        "n_requests": int(v.size),
        "iqr_ms": float(np.percentile(v, 75) - np.percentile(v, 25)),
    }


def model_a_span_factor(seed, n):
    """Arrival-span factor predicted by the exponential stream alone.
    Validated against observation in audit_sweep_rev2 sec R5 (12 cells, mean
    |err| 0.077%).  Reported as an instrument check, never as a gate."""
    rs = np.random.RandomState(seed)
    e = rs.exponential(1.0, size=n - 1)
    return float(e.sum() / (n - 1))


def analyse_main(path, seed):
    d, ttfts, itls = load_bench(path)
    dur = d["duration"]
    n = len(d["ttfts"])
    sum_dec = float(sum(x.sum() for x in itls))
    L_dec = sum_dec / dur
    L_pre = float(ttfts.sum()) / dur
    reported = d.get("concurrency", float("nan"))
    span_hat = dur - (d["ttfts"][-1] + float(np.sum(d["itls"][-1])))
    nominal_span = (n - 1) / d["request_rate"]
    return {
        "seed": seed,
        "offered_rate": d["request_rate"],
        "achieved_rate": d["request_throughput"],
        "rho_vs_step1_mu_p": None,  # filled by caller (needs arm)
        "duration_s": dur,
        "completed": d["completed"],
        "n_errors": n - len(ttfts),
        "itl95": itl95_estimators(itls),
        "L_decode_exact": L_dec,
        "L_pre_exact": L_pre,
        "L_total_reported": reported,
        "crosscheck_rel_err": (abs(L_pre + L_dec - reported) / reported
                               if reported and reported == reported else float("nan")),
        "ttft_p50_s": float(np.percentile(ttfts, 50)),
        "ttft_p95_s": float(np.percentile(ttfts, 95)),
        "span_hat_s": span_hat,
        "span_factor_observed": span_hat / nominal_span,
        "span_factor_model_a": model_a_span_factor(seed, n),
        "_ttfts": ttfts.tolist(),
    }


def anova_decomp(boot_values):
    """boot_values: list (per boot) of lists (per seed) of floats."""
    boots = [np.asarray(b, dtype=float) for b in boot_values if len(b) >= 1]
    if len(boots) < 2:
        return {"note": "fewer than 2 boots -- decomposition undefined"}
    n_seed = min(len(b) for b in boots)
    within = [float(b.var(ddof=1)) for b in boots if b.size >= 2]
    s2_seed = float(np.mean(within)) if within else float("nan")
    means = np.array([b.mean() for b in boots])
    s2_between = float(means.var(ddof=1))
    raw = s2_between - (s2_seed / n_seed if n_seed and s2_seed == s2_seed else 0.0)
    clipped = raw < 0
    s2_boot = max(raw, 0.0)
    grand = float(means.mean())
    return {
        "n_boot": len(boots),
        "n_seed_per_boot": n_seed,
        "grand_mean": grand,
        "sigma_seed": math.sqrt(s2_seed) if s2_seed == s2_seed else None,
        "sigma_boot": math.sqrt(s2_boot),
        "sigma_boot_raw_variance": raw,
        "sigma_boot_clipped_at_zero": bool(clipped),
        "sigma_seed_rel_pct": (100 * math.sqrt(s2_seed) / grand) if grand and s2_seed == s2_seed else None,
        "sigma_boot_rel_pct": (100 * math.sqrt(s2_boot) / grand) if grand else None,
        "boot_means": means.tolist(),
    }


def dominance(cdf_by_arm, grid):
    """Report where F_16 >= F_44 >= F_54 fails.  FULL grid, no domain
    restriction (prereg sec 3 item 3).  Reported, not judged."""
    order = [16, 44, 54]
    F = [cdf_by_arm[a] for a in order]
    viol = np.zeros(grid.size, dtype=bool)
    worst = 0.0
    for i in range(len(order) - 1):
        bad = F[i] < F[i + 1] - 1e-12
        viol |= bad
        if bad.any():
            worst = max(worst, float(np.max(F[i + 1][bad] - F[i][bad])))
    segs = []
    if viol.any():
        idx = np.flatnonzero(viol)
        start = idx[0]
        for a, b in zip(idx[:-1], idx[1:]):
            if b != a + 1:
                segs.append([float(grid[start]), float(grid[a])])
                start = b
        segs.append([float(grid[start]), float(grid[idx[-1]])])
    return {
        "expected_order": "F_16 >= F_44 >= F_54  (over the full observed T range)",
        "grid_s": [float(grid[0]), float(grid[-1]), float(grid[1] - grid[0])],
        "n_grid": int(grid.size),
        "violation_frac": float(viol.mean()),
        "violation_intervals_s": segs,
        "max_violation_cdf_units": worst,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    R = Path(args.run_dir)

    res = {"probe": "P7_ITL_SIGMA",
           "prereg": "PREREG_P7_ITL_SIGMA_2026-09-09.md",
           "note": "pure instrumentation -- no decision rule, no threshold, no verdict",
           "boots": [], "missing": []}

    # main benches:  bench_r<round>_d<arm>_s<seed>.jsonl
    # sat benches:   sat_r<round>_d<arm>.jsonl
    per_arm_main = {a: [] for a in ARMS}     # list per boot of list per seed
    per_arm_itl = {a: [] for a in ARMS}      # same shape, itl95 median
    per_arm_ldec = {a: [] for a in ARMS}
    per_arm_sat = {a: [] for a in ARMS}
    ttfts_by_arm = {a: [] for a in ARMS}

    for rnd in (1, 2, 3, 4):
        for arm in ARMS:
            mains = sorted(glob.glob(str(R / f"bench_r{rnd}_d{arm}_s*.jsonl")))
            if not mains:
                res["missing"].append(f"round{rnd}/d{arm}: no main bench (UNRESOLVED)")
                continue
            boot = {"round": rnd, "arm": arm, "benches": []}
            itl_vals, ldec_vals = [], []
            for p in mains:
                seed = int(re.search(r"_s(\d+)\.jsonl$", p).group(1))
                b = analyse_main(p, seed)
                b["rho_vs_step1_mu_p"] = b["offered_rate"] / MU_P_S1[arm]
                ttfts_by_arm[arm].extend(b.pop("_ttfts"))
                boot["benches"].append(b)
                itl_vals.append(b["itl95"]["median"])
                ldec_vals.append(b["L_decode_exact"])
            satp = R / f"sat_r{rnd}_d{arm}.jsonl"
            if satp.exists():
                d, _, _ = load_bench(satp)
                boot["sat"] = {"offered_rate": d["request_rate"],
                               "mu_p_observed": d["request_throughput"],
                               "ratio_to_step1": d["request_throughput"] / MU_P_S1[arm],
                               "completed": d["completed"]}
                per_arm_sat[arm].append(d["request_throughput"])
            else:
                boot["sat"] = {"status": "UNRESOLVED (no sat bench)"}
            per_arm_itl[arm].append(itl_vals)
            per_arm_ldec[arm].append(ldec_vals)
            per_arm_main[arm].append(boot)
            res["boots"].append(boot)

    # variance decomposition
    res["variance"] = {}
    for arm in ARMS:
        res["variance"][arm] = {
            "itl95_median_ms": anova_decomp(per_arm_itl[arm]),
            "L_decode_exact": anova_decomp(per_arm_ldec[arm]),
            "mu_p_saturated": (
                {"n_boot": len(per_arm_sat[arm]),
                 "mean": float(np.mean(per_arm_sat[arm])),
                 "sd": float(np.std(per_arm_sat[arm], ddof=1)) if len(per_arm_sat[arm]) >= 2 else None,
                 "rel_sd_pct": (100 * float(np.std(per_arm_sat[arm], ddof=1)) / float(np.mean(per_arm_sat[arm]))
                                if len(per_arm_sat[arm]) >= 2 else None),
                 "step1_n_boot1": MU_P_S1[arm],
                 "values": per_arm_sat[arm]}
                if per_arm_sat[arm] else {"status": "UNRESOLVED"}),
        }

    # TTFT dominance over the full observed range (main benches only)
    if all(ttfts_by_arm[a] for a in ARMS):
        hi = max(max(ttfts_by_arm[a]) for a in ARMS)
        grid = np.arange(0.05, hi + 0.05, 0.05)
        cdf = {a: np.array([(np.asarray(ttfts_by_arm[a]) <= t).mean() for t in grid])
               for a in ARMS}
        res["ttft_dominance"] = dominance(cdf, grid)
        res["ttft_percentiles_s"] = {
            a: {q: float(np.percentile(ttfts_by_arm[a], q)) for q in (5, 50, 95)}
            for a in ARMS}
    else:
        res["ttft_dominance"] = {"status": "UNRESOLVED"}

    xs = [b["crosscheck_rel_err"] for bt in res["boots"] for b in bt["benches"]]
    res["crosscheck_rel_err_max"] = max(xs) if xs else None
    res["n_main_benches"] = len(xs)

    text = json.dumps(res, indent=2, sort_keys=True, ensure_ascii=False, default=str)
    print(text)
    if args.out:
        Path(args.out).write_text(text)


if __name__ == "__main__":
    main()
