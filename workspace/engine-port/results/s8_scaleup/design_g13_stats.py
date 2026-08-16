#!/usr/bin/env python3
"""Design inputs for "다음 실험 gate" #13 (job/node axis + batch-vs-job decomposition).

GPU spend: 0.  This script only re-reads artifacts that already exist.
It introduces **no new estimand** -- the C2 ratio estimand stays exactly where it
is (`s8_batch_matched.py` / `s8_c2r_score.py`); here we only

  (1) `bootvar`  : read per-boot leg medians out of `C2R_RESULTS_2026-08-16.json`
                   and report the *within-job between-boot* dispersion of r,
                   reproducing the canonical t(5) intervals as a positive control
                   (CONSENSUS.md §3 item 56(A): M8 [3.056,3.060], Ha8 [3.077,3.155]).
  (2) `reach`    : histogram realized `decode_running_batch_size` over
                   decode-active *snapshots* (no t0 join needed) for the ctx1024
                   jobs, to test whether gate #12's B-closure still binds here.
                   Positive control: the same filter is run through the canonical
                   tool `realized_pin_check.py` (subprocess) and the CO_RESIDENT /
                   DECODE_BATCH-median lines must match.
  (3) `power`    : closed-form precision of a between-job SD estimate (chi-square)
                   and the detectable between-job difference, as a function of
                   n_job / n_boot_per_job.

Usage:
    python3 design_g13_stats.py bootvar
    python3 design_g13_stats.py reach
    python3 design_g13_stats.py power
    python3 design_g13_stats.py all --out DESIGN_G13_STATS_2026-08-16.json
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import os
import re
import statistics as st
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
C2R_JSON = os.path.join(HERE, "C2R_RESULTS_2026-08-16.json")
PIN_TOOL = os.path.join(HERE, "realized_pin_check.py")

# CONSENSUS.md §3 item 56(A) -- canonical adopted intervals (claims-auditor).
CANON_T5 = {"M8": (3.056, 3.060), "Ha8": (3.077, 3.155)}
CANON_R = {"M8": 3.058, "Ha8": 3.114}

# Student-t 0.975 quantiles, df -> t
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 14: 2.145,
        15: 2.131, 19: 2.093, 23: 2.069, 29: 2.045, 39: 2.023, 47: 2.011}
# chi-square quantiles chi2_{q, df}
CHI2 = {  # df: (q0.025, q0.975)
    1: (0.000982, 5.024), 2: (0.0506, 7.378), 3: (0.2158, 9.348),
    4: (0.4844, 11.143), 5: (0.8312, 12.833), 6: (1.2373, 14.449),
    7: (1.6899, 16.013), 8: (2.1797, 17.535), 9: (2.7004, 19.023),
    10: (3.2470, 20.483), 11: (3.8157, 21.920), 12: (4.4038, 23.337),
    15: (6.2621, 27.488), 19: (8.9065, 32.852), 23: (11.689, 38.076),
}


# --------------------------------------------------------------------------
# (1) within-job, between-boot dispersion of r
# --------------------------------------------------------------------------
def bootvar() -> dict:
    d = json.load(open(C2R_JSON))
    out = {"source": os.path.basename(C2R_JSON), "arms": {}, "positive_control": {}}
    for arm in ("M8", "Ha8"):
        leg16 = d["cells"][f"{arm}/d16/SM16/b16"]
        leg92 = d["cells"][f"{arm}/d92/SM92/b16"]
        blk = lambda k: int(re.search(r"blk(\d+)$", k).group(1))  # noqa: E731
        m16 = {blk(k): v for k, v in leg16["per_boot_median"].items()}
        m92 = {blk(k): v for k, v in leg92["per_boot_median"].items()}
        blks = sorted(set(m16) & set(m92))
        ratios = [m16[b] / m92[b] for b in blks]
        n = len(ratios)
        mean_r = st.fmean(ratios)
        sd_r = st.stdev(ratios)
        se = sd_r / math.sqrt(n)
        t = T975[n - 1]
        ci = (mean_r - t * se, mean_r + t * se)
        rel = lambda xs: st.stdev(xs) / st.fmean(xs) * 100.0  # noqa: E731
        # --- variant B: unpaired delta-method on the two legs' boot means.
        # This is the path that reproduces the canonical t(5) intervals of
        # CONSENSUS.md §3 item 56(A); variant A (block-paired) is reported as a
        # secondary because it answers a different question (does the block
        # alternation induce a correlated boot effect across legs? -> it does not).
        v16 = [m16[b] for b in blks]
        v92 = [m92[b] for b in blks]
        r_hat = st.fmean(v16) / st.fmean(v92)
        rel_sd_comb = math.hypot(rel(v16), rel(v92))          # % per single boot-pair
        rel_se = rel_sd_comb / math.sqrt(n)
        ci_b = (r_hat * (1 - t * rel_se / 100.0), r_hat * (1 + t * rel_se / 100.0))
        out["arms"][arm] = {
            "unpaired_r_hat": r_hat,
            "unpaired_rel_sd_per_boot_pct": rel_sd_comb,
            "unpaired_rel_se_pct": rel_se,
            "unpaired_t5_ci95": ci_b,
            "n_boot_per_leg": n,
            "blocks": blks,
            "leg_sm16_boot_medians_ms": [m16[b] for b in blks],
            "leg_sm92_boot_medians_ms": [m92[b] for b in blks],
            "leg_sm16_rel_sd_pct": rel([m16[b] for b in blks]),
            "leg_sm92_rel_sd_pct": rel([m92[b] for b in blks]),
            "block_paired_r": ratios,
            "r_boot_mean": mean_r,
            "r_boot_sd": sd_r,
            "r_boot_rel_sd_pct": sd_r / mean_r * 100.0,
            "t5_ci95": ci,
            "pooled_r_from_json": d["ratios"][arm]["r"],
        }
        lo, hi = CANON_T5[arm]
        out["positive_control"][arm] = {
            "canon_t5_ci95": [lo, hi],
            "recomputed_t5_ci95_unpaired": [round(ci_b[0], 3), round(ci_b[1], 3)],
            "recomputed_t5_ci95_blockpaired": [round(ci[0], 3), round(ci[1], 3)],
            "match_3dp": (abs(round(ci_b[0], 3) - lo) <= 0.0005
                          and abs(round(ci_b[1], 3) - hi) <= 0.0005),
            "canon_point": CANON_R[arm],
            "pooled_point_3dp": round(d["ratios"][arm]["r"], 3),
            "point_match": abs(round(d["ratios"][arm]["r"], 3) - CANON_R[arm]) < 1e-9,
        }
    return out


# --------------------------------------------------------------------------
# (2) batch reachability, ctx1024 -- snapshot level (no t0 join required)
# --------------------------------------------------------------------------
def _snapshot_scan(path: str) -> dict:
    """Verbatim reuse of realized_pin_check.py's snapshot filter (lines 46-55),
    with (realized_sm, decode_bs) cross-tab added as an extra reporting axis."""
    tab = collections.defaultdict(collections.Counter)
    batches, co_resident, n_snap, n_active = [], 0, 0, 0
    for line in open(path):
        try:
            e = json.loads(line)
        except Exception:
            continue
        if e.get("event") != "runtime_snapshot" or e.get("phase") != "benchmark":
            continue
        n_snap += 1
        b = e.get("decode_running_batch_size", 0)
        if b <= 0:
            continue
        n_active += 1
        sm = e["decode_sms"] or 108
        tab[sm][b] += 1
        batches.append(b)
        if e.get("prefill_active_batch_size", 0) > 0:
            co_resident += 1
    return {
        "n_benchmark_snapshots": n_snap,
        "n_decode_active": n_active,
        "co_resident_frac": (co_resident / n_active) if n_active else None,
        "decode_batch_median": st.median(batches) if batches else None,
        "table": {str(k): dict(v) for k, v in tab.items()},
    }


def _pin_tool(path: str, expect: int) -> dict:
    txt = subprocess.run([sys.executable, PIN_TOOL, path, str(expect)],
                         capture_output=True, text=True).stdout
    got = {}
    m = re.search(r"CO_RESIDENT_frac: ([0-9.]+)", txt)
    if m:
        got["co_resident_frac"] = float(m.group(1))
    m = re.search(r"median=([0-9.]+)", txt)
    if m:
        got["decode_batch_median"] = float(m.group(1))
    return got


CTX1024_FILES = [
    # (job, arm, leg, expected realized SM, telemetry glob)
    ("865493", "M8", "d16", 16, "s8_deconf_M8_C1024_d16_865493_telemetry.jsonl"),
    ("865493", "M8", "d92", 92, "s8_deconf_M8_C1024_d92_865493_telemetry.jsonl"),
    ("865493", "Ha8", "d16", 16, "s8_deconf_Ha8_C1024_d16_865493_telemetry.jsonl"),
    ("865493", "Ha8", "d92", 92, "s8_deconf_Ha8_C1024_d92_865493_telemetry.jsonl"),
    ("865533", "M8", "d16", 16, "s8_deconf_M8_C1024_d16_865533_telemetry.jsonl"),
    ("865533", "M8", "d92", 92, "s8_deconf_M8_C1024_d92_865533_telemetry.jsonl"),
    ("865533", "Ha8", "d16", 16, "s8_deconf_Ha8_C1024_d16_865533_telemetry.jsonl"),
    ("865533", "Ha8", "d92", 92, "s8_deconf_Ha8_C1024_d92_865533_telemetry.jsonl"),
]
C2R_GLOBS = [
    ("883574", "M8", "d16", 16, "s8c2r_M8_C1024_d16_883574_blk*_telemetry.jsonl"),
    ("883574", "M8", "d92", 92, "s8c2r_M8_C1024_d92_883574_blk*_telemetry.jsonl"),
    ("883575", "Ha8", "d16", 16, "s8c2r_Ha8_C1024_d16_883575_blk*_telemetry.jsonl"),
    ("883575", "Ha8", "d92", 92, "s8c2r_Ha8_C1024_d92_883575_blk*_telemetry.jsonl"),
]
TARGET_B = (1, 4, 8, 9, 12, 15, 16, 20)


def _summarize(tab: dict, sm: int) -> dict:
    hist = {int(k): v for k, v in tab.get(str(sm), {}).items()}
    n = sum(hist.values())
    if not n:
        return {"n": 0}
    ordered = sorted(hist)
    out = {"n": n, "max_b": max(ordered), "modal_b": max(hist, key=hist.get)}
    for b in TARGET_B:
        out[f"n_b{b}"] = hist.get(b, 0)
        out[f"pct_b{b}"] = 100.0 * hist.get(b, 0) / n
    cum, p = 0, {}
    for b in ordered:
        cum += hist[b]
        for q in (10, 50, 90):
            if f"p{q}" not in p and cum >= q / 100 * n:
                p[f"p{q}"] = b
    out.update(p)
    out["hist"] = dict(sorted(hist.items()))
    return out


def reach() -> dict:
    rows, pc = {}, {}
    for job, arm, leg, sm, pat in CTX1024_FILES + C2R_GLOBS:
        files = sorted(glob.glob(os.path.join(HERE, pat))) if "*" in pat \
            else [os.path.join(HERE, pat)]
        files = [f for f in files if os.path.exists(f)]
        if not files:
            continue
        merged = collections.defaultdict(collections.Counter)
        agg = {"n_benchmark_snapshots": 0, "n_decode_active": 0}
        co_num = 0
        allb = []
        for f in files:
            s = _snapshot_scan(f)
            agg["n_benchmark_snapshots"] += s["n_benchmark_snapshots"]
            agg["n_decode_active"] += s["n_decode_active"]
            if s["n_decode_active"]:
                co_num += s["co_resident_frac"] * s["n_decode_active"]
            for k, v in s["table"].items():
                merged[k].update(v)
            for k, v in s["table"].items():
                for b, c in v.items():
                    allb.extend([int(b)] * c)
            if len(files) == 1:  # positive control only where the tool takes 1 file
                tool = _pin_tool(f, sm)
                pc[f"{job}/{arm}/{leg}"] = {
                    "tool": tool,
                    "mine": {"co_resident_frac": round(s["co_resident_frac"], 3),
                             "decode_batch_median": s["decode_batch_median"]},
                    "match": (abs(tool.get("co_resident_frac", -9)
                                  - round(s["co_resident_frac"], 3)) < 1e-9
                              and abs(tool.get("decode_batch_median", -9)
                                      - s["decode_batch_median"]) < 1e-9),
                }
        key = f"{job}/{arm}/{leg}"
        rows[key] = {
            "n_files": len(files),
            "expected_sm": sm,
            "n_decode_active": agg["n_decode_active"],
            "co_resident_frac": (co_num / agg["n_decode_active"]
                                 if agg["n_decode_active"] else None),
            "realized_sm_share": {k: sum(v.values()) / agg["n_decode_active"]
                                  for k, v in merged.items()} if agg["n_decode_active"] else {},
            "at_realized_sm": _summarize({k: dict(v) for k, v in merged.items()}, sm),
        }
    return {"rows": rows, "positive_control": pc}


# --------------------------------------------------------------------------
# (3) precision of a between-job SD estimate
# --------------------------------------------------------------------------
def _gammainc_reg(a: float, x: float) -> float:
    """Regularized lower incomplete gamma P(a, x).  No scipy in this env
    (methodology gate #22: state the numerical backend explicitly)."""
    if x <= 0:
        return 0.0
    if x < a + 1.0:                      # series
        term = 1.0 / a
        s = term
        n = 1
        while n < 10000:
            term *= x / (a + n)
            s += term
            if abs(term) < abs(s) * 1e-15:
                break
            n += 1
        return s * math.exp(-x + a * math.log(x) - math.lgamma(a))
    # continued fraction for Q(a,x)
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 10000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delt = d * c
        h *= delt
        if abs(delt - 1.0) < 1e-15:
            break
    q = math.exp(-x + a * math.log(x) - math.lgamma(a)) * h
    return 1.0 - q


def _chi2_lower(p: float, nu: float) -> float:
    """chi2_{p,nu} by bisection on the regularized incomplete gamma (exact to 1e-9)."""
    lo, hi = 1e-12, max(10.0, 4 * nu + 40)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _gammainc_reg(nu / 2.0, mid / 2.0) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def power(sigma_boot_pct=(0.061, 1.20), target_pct=5.1,
          sigma_job_prior_pct=0.57) -> dict:
    """One-way random-effects precision for the between-job SD of r.

    Model (on log r):  r_{jk} = mu + J_j + E_jk,  SD(J)=sigma_job (the estimand),
    SD(E)=sigma_boot (measured: bootvar).  k jobs x m boot-pairs per job.

    MS_B = m*sigma_job^2 + sigma_boot^2 (df k-1);  MS_W = sigma_boot^2 (df k(m-1)).
    sigma_job^2_hat = (MS_B - MS_W)/m.
    Var(sigma_job^2_hat) ~= (2/m^2)[ (m*sj^2+sb^2)^2/(k-1) + sb^4/(k(m-1)) ].
    Satterthwaite df_eff = 2*sj^4 / Var(sigma_job^2_hat);
    95% upper bound = sj * sqrt(df_eff / chi2_{0.05, df_eff}).
    """
    res = {"target_gap_pct": target_pct,
           "sigma_boot_pct_measured": {"M8": sigma_boot_pct[0], "Ha8": sigma_boot_pct[1]},
           "sigma_job_prior_pct": sigma_job_prior_pct,
           "grid": [], "sd_precision_ideal": []}
    for k in (2, 3, 4, 6, 8, 12, 16, 24):
        df = k - 1
        if df not in CHI2:
            continue
        lo, hi = CHI2[df]
        res["sd_precision_ideal"].append({
            "n_job": k, "df": df,
            "rel_se_of_sd_pct": 100.0 / math.sqrt(2 * df),
            "ci95_multiplier_low": math.sqrt(df / hi),
            "ci95_multiplier_high": math.sqrt(df / lo),
        })
    sj = sigma_job_prior_pct
    for arm, sb in (("M8", sigma_boot_pct[0]), ("Ha8", sigma_boot_pct[1])):
        for k in (3, 4, 6, 8, 12):
            for m in (3, 6, 12):
                if k * (m - 1) < 1:
                    continue
                var = (2.0 / m ** 2) * ((m * sj ** 2 + sb ** 2) ** 2 / (k - 1)
                                        + sb ** 4 / (k * (m - 1)))
                df_eff = max(1.0, 2 * sj ** 4 / var)
                ub = sj * math.sqrt(df_eff / _chi2_lower(0.05, df_eff))
                boots = k * m * 2          # two legs per boot-pair
                res["grid"].append({
                    "arm": arm, "sigma_boot_pct": round(sb, 4),
                    "n_job": k, "n_bootpair_per_job": m, "n_boots_total": boots,
                    "gpu_hr": round(boots * 144.0 / 3600.0, 2),  # measured: 3460s/24 boots
                    "sd_of_job_mean_pct": math.sqrt(sj ** 2 + sb ** 2 / m),
                    "boot_noise_share_of_jobmean_var":
                        (sb ** 2 / m) / (sj ** 2 + sb ** 2 / m),
                    "df_eff_satterthwaite": df_eff,
                    "rel_se_of_sigma_job_pct": 100.0 / math.sqrt(2 * df_eff),
                    "sigma_job_ub95_pct": ub,
                    "gap_over_ub95": target_pct / ub,
                    "gap_ge_3sigma_ub": target_pct >= 3 * ub,
                })
    return res


# --------------------------------------------------------------------------
# (4) empirical prior on the cross-job dispersion (the only one that exists)
# --------------------------------------------------------------------------
AUDIT_JSON = os.path.join(HERE, "audit_c2_job_composition_2026-08-15.json")


def prior() -> dict:
    """Cross-job dispersion from the ONLY two jobs that can be matched cell-wise
    (865493 gpu40 vs 865533 gpu38).  Both the leg-median axis and the ratio axis.
    df=1 -- this is a point contrast, not a variance estimate."""
    d = json.load(open(AUDIT_JSON))
    cj = d["cross_job_matched"]
    legs, by_ab = [], {}
    for key, v in cj.items():
        arm, sm, b = key.split("/")
        if sm == "SM108":            # unpartitioned; not a leg of r
            continue
        legs.append({"cell": key, "pct_diff": v["pct_diff_533_vs_493"]})
        by_ab.setdefault((arm, b), {})[sm] = v
    ratios = []
    for (arm, b), sms in sorted(by_ab.items()):
        if "SM16" in sms and "SM92" in sms:
            r493 = sms["SM16"]["median_865493_ms"] / sms["SM92"]["median_865493_ms"]
            r533 = sms["SM16"]["median_865533_ms"] / sms["SM92"]["median_865533_ms"]
            ratios.append({"arm": arm, "b": b, "r_865493": r493, "r_865533": r533,
                           "pct_diff": (r533 / r493 - 1) * 100.0})
    lp = [x["pct_diff"] for x in legs]
    rp = [x["pct_diff"] for x in ratios]
    out = {
        "n_matched_split_cells": len(legs),
        "leg_pct_diff_mean": st.fmean(lp), "leg_pct_diff_sd": st.stdev(lp),
        "leg_pct_diff_rms": math.sqrt(sum(x * x for x in lp) / len(lp)),
        "leg_pct_diff_absmax": max(abs(x) for x in lp),
        "ratio_cells": ratios,
        "ratio_pct_diff_rms": (math.sqrt(sum(x * x for x in rp) / len(rp))
                               if rp else None),
        "sigma_job_hat_leg_pct_from_rms": math.sqrt(sum(x * x for x in lp)
                                                    / len(lp)) / math.sqrt(2),
        "sigma_job_hat_ratio_pct_from_rms": (math.sqrt(sum(x * x for x in rp)
                                                       / len(rp)) / math.sqrt(2)
                                             if rp else None),
        "df": 1,
        "chi2_ci95_multiplier_df1": [math.sqrt(1 / CHI2[1][1]),
                                     math.sqrt(1 / CHI2[1][0])],
        "note": ("865493 vs 865533 differ in job AND node (gpu40/gpu38) AND "
                 "keepalive regime; the split cannot be attributed. df=1."),
    }
    return out


# --------------------------------------------------------------------------
# (5) how big is the batch term?  within-SINGLE-job r(b) steps that already exist
# --------------------------------------------------------------------------
def bslope() -> dict:
    """Within-one-job r(b) steps from `audit_c2_job_composition_2026-08-15.json`.

    NOT a finding -- a sizing input.  Every cell below is n_indep=1 (one server
    boot per (job,cell); reps are not independent, CONSENSUS §3 item 51) and most
    of them live in the collapsed-keepalive job 865533.  Arms are NOT ranked here
    (citation-stop (a) is about arm-wise epsilon / arm ordering; these are each
    arm's own b-response and are used only to size the b ladder)."""
    d = json.load(open(AUDIT_JSON))["ratios"]
    curves = {}
    for k, v in d.items():
        arm, b = k.split("/")
        for src, p in v.items():
            if not src.startswith("job_"):
                continue
            curves.setdefault((arm, src), {})[int(b[1:])] = {
                "r": p["point"], "n_lo": p["n_lo"], "n_hi": p["n_hi"],
                "cl_lo": p["n_clusters_lo"], "cl_hi": p["n_clusters_hi"]}
    steps = []
    for (arm, job), bs in sorted(curves.items()):
        ks = sorted(bs)
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                lo, hi = ks[i], ks[j]
                if lo < 8 or hi < 12:      # only the b12..b16 neighbourhood matters
                    continue
                steps.append({
                    "arm": arm, "job": job, "b_from": lo, "b_to": hi,
                    "r_from": bs[lo]["r"], "r_to": bs[hi]["r"],
                    "pct": (bs[hi]["r"] / bs[lo]["r"] - 1) * 100.0,
                    "clusters": [bs[lo]["cl_lo"], bs[lo]["cl_hi"],
                                 bs[hi]["cl_lo"], bs[hi]["cl_hi"]],
                })
    gaps = {
        "M8": {"canon_headline_b12": 2.9091, "c2r_b16": 3.057990028074119},
        "Ha8": {"canon_headline_b9": 2.8518, "c2r_b16": 3.11406133891256},
    }
    for a, g in gaps.items():
        old = [v for k, v in g.items() if k.startswith("canon")][0]
        g["pct_gap"] = (g["c2r_b16"] / old - 1) * 100.0
    return {"curves": {f"{a}/{j}": {str(b): round(x["r"], 4) for b, x in sorted(c.items())}
                       for (a, j), c in sorted(curves.items())},
            "steps_b_ge_8_to_b_ge_12": steps,
            "total_gaps_to_explain": gaps}


# ==========================================================================
# rev2 (2026-08-16) -- claims-auditor findings F1-F6 + recommendation C-g.
#
# The rev1 design was audited NO-GO.  Nothing below changes an estimand or a
# measurement; it changes the *decision rule layer* that the audit refuted:
#
#   F2  MS_B <= MS_W must be a MEASUREMENT FAILURE, never a PASS (the rev1 rule
#       auto-passes on a degenerate sigma_hat=0, i.e. it declares "job axis
#       controlled" exactly when boot noise swallowed the job signal).  The
#       Satterthwaite upper bound is replaced by an exact-F pivot whose coverage
#       is simulated here rather than asserted.
#   F3  every (k, m) row carries a simulated P(pass) and a simulated coverage;
#       the acceptance line is P(pass) >= 0.8 AND coverage >= 0.93.
#   F4  sigma_boot is the MEASURED per-boot-pair SD of r (`r_boot_rel_sd_pct`,
#       already computed by bootvar()), not the independence-assuming delta
#       method (`unpaired_rel_sd_per_boot_pct`) rev1 used.
#   F5  the prior is a BAND over three defensible cell subsets, not one point.
#   F6  the per-arm gap is wired from bslope(); rev1's main() left power() at
#       its 5.1 default so the JSON never produced the doc's Ha8 numbers.
#   C-g alternating the conc step INSIDE a boot makes Delta_batch a within-boot
#       paired contrast and cancels the sigma_boot term outright.
#
# Positive controls (methodology gate #9 -- a control must run the SAME branch
# as the headline, and must be able to FAIL): see `_rev2_positive_controls`.
# ==========================================================================
import random  # noqa: E402  (rev2 addition, kept next to its users)

MC_N = 20000
MC_SEED = 20260816
# Acceptance line, pre-registered here (methodology gate #37: an adversarial
# audit gets a single question AND its pass mark).
ACCEPT_P_PASS = 0.80
ACCEPT_COVERAGE = 0.93
# Audit values this script must reproduce independently (it does not import the
# audit code -- `audit_g13_independent_2026-08-16/` exists to stay independent).
AUDIT_TARGETS = {
    "sd_paired_pct": {"M8": 0.0371, "Ha8": 1.5684},
    "rho_leg": {"M8": 0.800, "Ha8": -0.767},
    "cover_satterthwaite_Ha8_k4m3": 0.638,
    "cover_exactF_Ha8_k4m3": 0.970,
}


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta (Lentz).  No scipy here."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delt = d * c
        h *= delt
        if abs(delt - 1.0) < 1e-14:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
             + a * math.log(x) + b * math.log(1.0 - x))
    front = math.exp(lbeta)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _f_cdf(f: float, d1: float, d2: float) -> float:
    if f <= 0:
        return 0.0
    return _betai(d1 / 2.0, d2 / 2.0, d1 * f / (d1 * f + d2))


def _f_quantile(p: float, d1: float, d2: float) -> float:
    """F_{p, d1, d2} by bisection on the CDF."""
    lo, hi = 1e-12, 1.0
    while _f_cdf(hi, d1, d2) < p and hi < 1e12:
        hi *= 2.0
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if _f_cdf(mid, d1, d2) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---- F4: the measured per-boot-pair dispersion, and the leg correlation ----
def pairsd() -> dict:
    """F4.  sigma_boot must be the SD of r actually realized per boot-pair.

    rev1 used `hypot(relSD(leg16), relSD(leg92))`, which assumes the two legs
    are independent within a boot-pair.  They are not: bootvar() already
    measured the paired quantity and the two disagree by 0.61x (M8) and 1.32x
    (Ha8) because the leg correlation is strongly positive for one arm and
    strongly negative for the other.  rho_leg is pre-registered as a REPORTING
    AXIS (audit F1) because it is what decides whether the ratio structure
    cancels a common-mode job effect or amplifies it.
    """
    bv = bootvar()
    out = {"note": ("sigma_boot(r) = measured SD of the per-boot-pair ratio; "
                    "rev1 used the independence-assuming delta method"),
           "arms": {}}
    for arm in ("M8", "Ha8"):
        a = bv["arms"][arm]
        v16, v92 = a["leg_sm16_boot_medians_ms"], a["leg_sm92_boot_medians_ms"]
        n = len(v16)
        m16, m92 = st.fmean(v16), st.fmean(v92)
        cov = sum((x - m16) * (y - m92) for x, y in zip(v16, v92)) / (n - 1)
        rho = cov / (st.stdev(v16) * st.stdev(v92))
        out["arms"][arm] = {
            "n_boot_pairs": n,
            "sigma_boot_paired_pct": a["r_boot_rel_sd_pct"],       # ADOPTED
            "sigma_boot_unpaired_delta_pct": a["unpaired_rel_sd_per_boot_pct"],
            "paired_over_unpaired": (a["r_boot_rel_sd_pct"]
                                     / a["unpaired_rel_sd_per_boot_pct"]),
            "rho_leg": rho,
            "per_boot_r": a["block_paired_r"],
        }
    return out


# ---- F5: the prior is a band, not a point --------------------------------
def priorband() -> dict:
    """F5.  Three defensible readings of the ONLY cross-job contrast that exists.

    All three come from the same 865493-vs-865533 pair (df=1 either way), but
    they differ in which cells are admissible:

      leg     : all 40 matched split-leg cells      -> 0.41%
      ratio   : the 3 cells where r exists in both  -> 0.57%   (rev1's point)
      ratio15 : b>=15 only, i.e. the batch regime the campaign will actually
                run in                              -> 0.95%

    The third one matters: rev1's headline row (M8 k=4, m=3) is a PASS at 0.57%
    and a FAIL at 0.95%.  Choosing the prior after seeing that would be a
    post-hoc knob (methodology gate #19), so the band is registered up front and
    every power row is reported at all three.
    """
    pr = prior()
    rc = pr["ratio_cells"]
    hi = [c for c in rc if int(c["b"][1:]) >= 15]
    rms = lambda xs: math.sqrt(sum(x * x for x in xs) / len(xs))  # noqa: E731
    band = {
        "leg_all_cells": {"pct": pr["sigma_job_hat_leg_pct_from_rms"],
                          "n_cells": pr["n_matched_split_cells"], "df": 1},
        "ratio_all_cells": {"pct": pr["sigma_job_hat_ratio_pct_from_rms"],
                            "n_cells": len(rc), "df": 1,
                            "cells": [f"{c['arm']}/{c['b']}" for c in rc]},
        "ratio_b_ge_15": {"pct": (rms([c["pct_diff"] for c in hi]) / math.sqrt(2)
                                  if hi else None),
                          "n_cells": len(hi), "df": 1,
                          "cells": [f"{c['arm']}/{c['b']}" for c in hi]},
    }
    vals = [v["pct"] for v in band.values() if v["pct"] is not None]
    band["band_pct"] = [min(vals), max(vals)]
    band["ratio_of_extremes"] = max(vals) / min(vals)
    band["warning"] = ("every entry is df=1 and confounded job x node x "
                       "keepalive-regime (prior()'s note); the band is a "
                       "sensitivity range, NOT a variance estimate")
    return band


# ---- F2/F3: the decision rule, and what it actually does ------------------
def _ub_exact_f(msb: float, msw: float, m: int,
                fq_lo: float, chi_w: float, df_w: int) -> float:
    """95% upper bound on sigma_job via the exact-F pivot.

    theta = sigma_job^2 / sigma_boot^2 is bounded by ((MS_B/MS_W)/F_{.05}) - 1)/m
    and sigma_boot^2 by MS_W*df_W/chi2_{.05,df_W}; the product bounds
    sigma_job^2.  Conservative by construction (two 95% bounds multiplied) --
    which is the point: the audit measured the rev1 Satterthwaite bound at 0.638
    coverage for Ha8 k=4/m=3, i.e. it was not a 95% bound at all.
    """
    theta = ((msb / msw) / fq_lo - 1.0) / m
    return math.sqrt(max(0.0, theta) * msw * df_w / chi_w)


def _oc(k: int, m: int, sigma_boot: float, sigma_job: float, gap: float,
        n: int = MC_N, seed: int = MC_SEED, sat_prior: float = None) -> dict:
    """Monte-Carlo operating characteristics of the rev2 decision rule.

    Draws the two mean squares from their exact sampling distributions and
    applies the rule verbatim, including the F2 guard.  Reports the guard rate
    separately so a design cannot buy P(pass) with degenerate draws.
    """
    rnd = random.Random(seed + 1000 * k + m)
    df_b, df_w = k - 1, k * (m - 1)
    fq_lo = _f_quantile(0.05, df_b, df_w)
    chi_w = _chi2_lower(0.05, df_w)
    # The Satterthwaite multiplier is a DESIGN constant: it is fixed by the
    # registered prior, not by the data.  `sat_prior` lets the caller register
    # it at one prior and then evaluate coverage at every prior in the band --
    # which is the only way to find out whether that registered constant is
    # still a 95% bound when the truth sits elsewhere in the band.
    sj_reg = sigma_job if sat_prior is None else sat_prior
    df_eff = max(1.0, 2 * sj_reg ** 4 / ((2.0 / m ** 2) * (
        (m * sj_reg ** 2 + sigma_boot ** 2) ** 2 / df_b
        + sigma_boot ** 4 / (k * (m - 1)))))
    mult_sat = math.sqrt(df_eff / _chi2_lower(0.05, df_eff))
    ev_b = m * sigma_job ** 2 + sigma_boot ** 2
    n_fail = n_pass = n_cov = n_cov_claim = n_claim = n_pass_sat = n_cov_sat = 0
    n_pass_sat_guarded = 0
    ubs = []
    for _ in range(n):
        msb = ev_b * rnd.gammavariate(df_b / 2.0, 2.0) / df_b
        msw = sigma_boot ** 2 * rnd.gammavariate(df_w / 2.0, 2.0) / df_w
        # --- rev1 rule, kept only so the two can be compared on one draw set
        v = (msb - msw) / m
        ub_sat = math.sqrt(max(0.0, v)) * mult_sat
        # NOTE: rev1 counted a pass here even when msb <= msw (ub_sat == 0),
        # which is finding F2.  `p_pass` below is the UNGUARDED rev1 rule, kept
        # verbatim so PC5 reproduces the audit; `p_pass_guarded` is the same
        # bound under the F2 guard, which is what rev2 would actually adopt.
        if gap >= 3 * ub_sat:
            n_pass_sat += 1
            if msb > msw:
                n_pass_sat_guarded += 1
        if ub_sat >= sigma_job:
            n_cov_sat += 1
        # --- rev2 rule
        ub = _ub_exact_f(msb, msw, m, fq_lo, chi_w, df_w)
        ubs.append(ub)
        if msb <= msw:                      # F2 guard: MEASUREMENT FAILURE
            n_fail += 1
        else:
            n_claim += 1
            if gap >= 3 * ub:
                n_pass += 1
            if ub >= sigma_job:
                n_cov_claim += 1
        if ub >= sigma_job:
            n_cov += 1
    ubs.sort()
    return {
        "p_measurement_failure": n_fail / n,
        "p_pass": n_pass / n,                       # unconditional; F2-guarded
        "coverage": n_cov / n,
        "coverage_given_claim": (n_cov_claim / n_claim) if n_claim else None,
        "ub_median_pct": ubs[n // 2],
        "ub_p90_pct": ubs[int(0.9 * n)],
        "mc_se_p_pass": math.sqrt(max(n_pass / n * (1 - n_pass / n), 1e-12) / n),
        "rev1_satterthwaite": {"p_pass": n_pass_sat / n, "coverage": n_cov_sat / n,
                               "p_pass_guarded": n_pass_sat_guarded / n,
                               "registered_at_prior_pct": sj_reg},
        "n_mc": n,
    }


def power2() -> dict:
    """F2/F3/F6.  The rev1 table, rebuilt with the rule the audit demands.

    Every row: its OWN arm gap (F6), the MEASURED paired sigma_boot (F4), all
    three priors (F5), and simulated P(pass)/coverage instead of an asserted
    3-sigma tick (F2/F3).
    """
    ps = pairsd()
    band = priorband()
    gaps = bslope()["total_gaps_to_explain"]
    priors = {k: v["pct"] for k, v in band.items()
              if isinstance(v, dict) and v.get("pct") is not None}
    plans = {
        "M8": [(4, 3, 60), (6, 3, 60), (8, 3, 60), (12, 3, 60)],
        "Ha8": [(4, 3, 60), (6, 3, 60), (6, 6, 60), (8, 12, 60), (12, 6, 60),
                (6, 3, 240), (4, 3, 240), (6, 3, 480)],
    }
    rows = []
    for arm, plan in plans.items():
        gap = gaps[arm]["pct_gap"]
        sb60_paired = ps["arms"][arm]["sigma_boot_paired_pct"]
        sb60_doc = ps["arms"][arm]["sigma_boot_unpaired_delta_pct"]
        for (k, m, win) in plan:
            # window scaling is an ASSUMPTION (sigma_boot ~ 1/sqrt(T)), flagged
            # as such; S0(a) is the measurement that decides whether it holds.
            scale = math.sqrt(60.0 / win)
            boots = k * m * 2
            sec_per_boot = 144.0 + (win - 60.0)      # measured 144 s at 60 s
            for pname, sj in sorted(priors.items()):
                for sbname, sb0 in (("paired_measured", sb60_paired),
                                    ("rev1_delta_method", sb60_doc)):
                    oc = _oc(k, m, sb0 * scale, sj, gap)
                    rows.append({
                        "arm": arm, "k_job": k, "m_bootpair": m,
                        "measure_window_s": win, "n_boots": boots,
                        "gpu_hr": round(boots * sec_per_boot / 3600.0, 3),
                        "gap_pct": gap, "prior": pname, "sigma_job_prior_pct": sj,
                        "sigma_boot_source": sbname,
                        "sigma_boot_pct": sb0 * scale,
                        "window_scaling_assumed": win != 60,
                        **oc,
                        "ACCEPT": (oc["p_pass"] >= ACCEPT_P_PASS
                                   and oc["coverage"] >= ACCEPT_COVERAGE),
                    })
    # A plan is only adoptable if it passes at EVERY prior in the band (F5).
    robust = {}
    for arm in plans:
        for (k, m, win) in plans[arm]:
            sel = [r for r in rows
                   if r["arm"] == arm and r["k_job"] == k and r["m_bootpair"] == m
                   and r["measure_window_s"] == win
                   and r["sigma_boot_source"] == "paired_measured"]
            robust[f"{arm}/k{k}/m{m}/{win}s"] = {
                "accept_at_all_priors": all(r["ACCEPT"] for r in sel),
                "accept_count": sum(r["ACCEPT"] for r in sel),
                "n_priors": len(sel),
                "gpu_hr": sel[0]["gpu_hr"] if sel else None,
                "worst_p_pass": min((r["p_pass"] for r in sel), default=None),
                "worst_coverage": min((r["coverage"] for r in sel), default=None),
                "max_p_measurement_failure": max(
                    (r["p_measurement_failure"] for r in sel), default=None),
            }
    return {"acceptance_rule": {
                "p_pass_min": ACCEPT_P_PASS, "coverage_min": ACCEPT_COVERAGE,
                "verdict_on_MSB_le_MSW": "MEASUREMENT_FAILURE (never PASS)",
                "upper_bound": "exact-F pivot (Satterthwaite reported alongside)"},
            "rows": rows, "robust_at_all_priors": robust}


REGISTERED_SAT_PRIOR = 0.5688257172317233   # priorband()["ratio_all_cells"]


def _eval_plan(k, m, sb, priors, gap, n, seed=MC_SEED):
    """Evaluate BOTH bounds over the whole prior band and pick by the
    pre-registered rule below.

    Bound-selection rule (registered BEFORE any data; it depends only on
    (k, m, sigma_boot, prior band), all of which are pre-data):

        adopt the Satterthwaite bound IF its simulated coverage is >= 0.93 at
        EVERY prior in the band; otherwise adopt the exact-F pivot.
        Either way the F2 guard applies: MS_B <= MS_W is a measurement failure.

    Why this is not a post-hoc knob: it is a function of the design, evaluated
    by simulation, fixed at pre-registration time.  Why it matters: the audit
    demonstrated the Satterthwaite bound's failure on Ha8 (coverage 0.638), and
    rev2's first draft over-corrected by forcing every arm onto the exact-F
    pivot -- whose coverage is 0.99, i.e. it buys safety M8 does not need and
    charges M8 ~10 GPU-hr for it.
    """
    ocs = [_oc(k, m, sb, sj, gap, n=n, seed=seed,
               sat_prior=REGISTERED_SAT_PRIOR) for sj in priors]
    sat_ok = all(o["rev1_satterthwaite"]["coverage"] >= ACCEPT_COVERAGE
                 for o in ocs)
    if sat_ok:
        chosen = "satterthwaite_registered"
        acc = all(o["rev1_satterthwaite"]["p_pass_guarded"] >= ACCEPT_P_PASS
                  for o in ocs)
        worst_p = min(o["rev1_satterthwaite"]["p_pass_guarded"] for o in ocs)
        worst_c = min(o["rev1_satterthwaite"]["coverage"] for o in ocs)
    else:
        chosen = "exact_F_pivot"
        acc = all(o["p_pass"] >= ACCEPT_P_PASS
                  and o["coverage"] >= ACCEPT_COVERAGE for o in ocs)
        worst_p = min(o["p_pass"] for o in ocs)
        worst_c = min(o["coverage"] for o in ocs)
    return {"bound": chosen, "accept": acc, "worst_p_pass": worst_p,
            "worst_coverage": worst_c,
            "max_p_measurement_failure": max(o["p_measurement_failure"]
                                             for o in ocs),
            "per_prior": ocs}


def plan_search_v2(n: int = 8000) -> dict:
    """Cheapest adoptable plan under the registered bound-selection rule."""
    ps = pairsd()
    band = priorband()
    gaps = bslope()["total_gaps_to_explain"]
    priors = sorted(v["pct"] for v in band.values()
                    if isinstance(v, dict) and v.get("pct") is not None)
    out = {"priors_pct": priors,
           "bound_selection_rule": _eval_plan.__doc__.strip().split("\n\n")[1].strip(),
           "registered_sat_prior_pct": REGISTERED_SAT_PRIOR, "arms": {}}
    for arm in ("M8", "Ha8"):
        gap = gaps[arm]["pct_gap"]
        sb60 = ps["arms"][arm]["sigma_boot_paired_pct"]
        feasible = []
        for win in (60, 120, 240):
            sb = sb60 * math.sqrt(60.0 / win)
            sec = 144.0 + (win - 60.0)
            for k in (4, 6, 8, 10, 12, 16, 20, 24):
                for m in (3, 6, 9, 12):
                    ev = _eval_plan(k, m, sb, priors, gap, n)
                    if ev["accept"]:
                        feasible.append({
                            "k_job": k, "m_bootpair": m, "measure_window_s": win,
                            "n_boots": k * m * 2, "bound": ev["bound"],
                            "gpu_hr": round(k * m * 2 * sec / 3600.0, 3),
                            "window_scaling_assumed": win != 60,
                            "worst_p_pass": ev["worst_p_pass"],
                            "worst_coverage": ev["worst_coverage"],
                            "max_p_measurement_failure":
                                ev["max_p_measurement_failure"]})
        feasible.sort(key=lambda r: r["gpu_hr"])
        # The sweep runs at n=8000, so its argmin is a noisy winner: the cheapest
        # plan is, by construction, the one whose MC error pushed it over the
        # line.  Walk UP the cost-sorted list at full MC on a different seed and
        # adopt the first plan that still accepts.  Every rejected step is kept
        # in `rejected_on_confirmation` -- a design that silently dropped them
        # would be reporting a boundary plan as if it had margin.
        confirmed, rejected = None, []
        for cand in feasible[:8]:
            sb = sb60 * math.sqrt(60.0 / cand["measure_window_s"])
            e = _eval_plan(cand["k_job"], cand["m_bootpair"], sb, priors, gap,
                           MC_N, seed=MC_SEED + 991)
            rec = {**cand, "confirm_bound": e["bound"],
                   "confirm_worst_p_pass": e["worst_p_pass"],
                   "confirm_worst_coverage": e["worst_coverage"],
                   "confirm_max_p_measurement_failure":
                       e["max_p_measurement_failure"]}
            if e["accept"]:
                confirmed = rec
                break
            rejected.append(rec)
        # Fallback that does NOT rest on the unverified sigma_boot ~ 1/sqrt(T)
        # assumption.  If the adopted plan needs a longer window, this is what
        # the campaign costs when S0(a) says the assumption fails.
        fb = None
        for cand in [f for f in feasible if not f["window_scaling_assumed"]][:8]:
            e = _eval_plan(cand["k_job"], cand["m_bootpair"], sb60, priors, gap,
                           MC_N, seed=MC_SEED + 991)
            if e["accept"]:
                fb = {**cand, "confirm_worst_p_pass": e["worst_p_pass"],
                      "confirm_worst_coverage": e["worst_coverage"],
                      "confirm_max_p_measurement_failure":
                          e["max_p_measurement_failure"]}
                break
        out["arms"][arm] = {
            "gap_pct": gap, "sigma_boot_60s_pct": sb60,
            "n_feasible_in_sweep": len(feasible),
            "cheapest_in_sweep": feasible[0] if feasible else None,
            "rejected_on_confirmation": rejected,
            "ADOPTED": confirmed,
            "fallback_60s_window_no_assumption": fb,
        }
    both = [v["ADOPTED"]["gpu_hr"] for v in out["arms"].values() if v["ADOPTED"]]
    out["total_gpu_hr_both_arms"] = round(sum(both), 2) if len(both) == 2 else None
    return out


def plan_search(n: int = 8000) -> dict:
    """Cheapest (k, m, window) that ACCEPTS at every prior in the band.

    power2() evaluates the rev1 menu; this searches outside it, because the
    rev1 menu turns out to contain no adoptable plan once the F2 guard, the
    exact-F bound and the F5 band are all applied at once.  Coarse MC (n=8000)
    for the sweep; the winner is re-run at full MC_N so the reported numbers
    are not the ones the search optimised over (a search over noisy estimates
    biases the winner upward -- re-running on a fresh seed removes that).
    """
    ps = pairsd()
    band = priorband()
    gaps = bslope()["total_gaps_to_explain"]
    priors = sorted(v["pct"] for v in band.values()
                    if isinstance(v, dict) and v.get("pct") is not None)
    out = {"priors_pct": priors, "arms": {},
           "search_note": ("winner re-evaluated at n=%d with a different seed "
                           "to undo winner's-curse from the sweep" % MC_N)}
    for arm in ("M8", "Ha8"):
        gap = gaps[arm]["pct_gap"]
        sb60 = ps["arms"][arm]["sigma_boot_paired_pct"]
        feasible = []
        for win in (60, 120, 240):
            sb = sb60 * math.sqrt(60.0 / win)
            sec = 144.0 + (win - 60.0)
            for k in (4, 6, 8, 10, 12, 16, 20, 24):
                for m in (3, 6, 9, 12):
                    ocs = [_oc(k, m, sb, sj, gap, n=n) for sj in priors]
                    if all(o["p_pass"] >= ACCEPT_P_PASS
                           and o["coverage"] >= ACCEPT_COVERAGE for o in ocs):
                        feasible.append({
                            "k_job": k, "m_bootpair": m, "measure_window_s": win,
                            "n_boots": k * m * 2,
                            "gpu_hr": round(k * m * 2 * sec / 3600.0, 3),
                            "window_scaling_assumed": win != 60,
                        })
        feasible.sort(key=lambda r: r["gpu_hr"])
        best = feasible[0] if feasible else None
        conf = None
        if best:
            sb = sb60 * math.sqrt(60.0 / best["measure_window_s"])
            conf = [{"prior_pct": sj,
                     **{kk: vv for kk, vv in
                        _oc(best["k_job"], best["m_bootpair"], sb, sj, gap,
                            n=MC_N, seed=MC_SEED + 991).items()
                        if kk in ("p_pass", "coverage", "p_measurement_failure",
                                  "ub_median_pct")}}
                    for sj in priors]
        out["arms"][arm] = {
            "gap_pct": gap, "sigma_boot_60s_pct": sb60,
            "n_feasible": len(feasible),
            "cheapest": best,
            "cheapest_confirmation_fresh_seed": conf,
            "confirmed": bool(conf) and all(
                c["p_pass"] >= ACCEPT_P_PASS and c["coverage"] >= ACCEPT_COVERAGE
                for c in conf),
            "next_cheapest": feasible[1:4],
        }
    both = [v["cheapest"]["gpu_hr"] for v in out["arms"].values() if v["cheapest"]]
    out["total_gpu_hr_both_arms"] = round(sum(both), 2) if len(both) == 2 else None
    return out


# ---- F5 second half: two-stage sequential ---------------------------------
def seq2(k1: int = 4, k2: int = 4, m: int = 3, n: int = MC_N) -> dict:
    """F5.  Pre-registered 2-stage design: k1 jobs, then k1+k2 if unresolved.

    Stopping rule fixed HERE, before any data: stop at stage 1 only if the
    stage-1 estimate is non-degenerate AND already passes.  A degenerate draw
    (MS_B <= MS_W) never stops the design -- that is the F2 guard applied to
    the sequential layer, where it matters most, because the cheap way to "win"
    a sequential design is to stop early on a lucky degenerate estimate.

    The MC below is the honest cost of that: selection inflates the final
    coverage error, so coverage is reported for the SEQUENTIAL procedure, not
    for its stages.
    """
    ps = pairsd()
    band = priorband()
    gaps = bslope()["total_gaps_to_explain"]
    out = {"design": {"k_stage1": k1, "k_stage2_added": k2, "m_bootpair": m,
                      "stop_rule": ("stage 1 stops iff MS_B > MS_W and "
                                    "gap >= 3*UB_exactF; otherwise escalate")},
           "rows": []}
    for arm in ("M8", "Ha8"):
        gap = gaps[arm]["pct_gap"]
        sb = ps["arms"][arm]["sigma_boot_paired_pct"]
        for pname, pv in sorted(band.items()):
            if not isinstance(pv, dict) or pv.get("pct") is None:
                continue
            sj = pv["pct"]
            rnd = random.Random(MC_SEED + 7)
            kk = k1 + k2
            # Stage-1 and pooled quantiles depend only on (k1, k2, m), so they
            # are hoisted out of the MC loop.  df_B pools as (k1-1)+(k2-1) and
            # df_W as (k1+k2)(m-1) -- the pooled design has ONE fewer df_B than
            # a single k1+k2 job run, because each stage estimates its own mean.
            fq1, cw1 = (_f_quantile(0.05, k1 - 1, k1 * (m - 1)),
                        _chi2_lower(0.05, k1 * (m - 1)))
            db = (k1 - 1) + (k2 - 1)
            dw = k1 * (m - 1) + k2 * (m - 1)
            fq2, cw2 = _f_quantile(0.05, db, dw), _chi2_lower(0.05, dw)
            ev = m * sj ** 2 + sb ** 2
            stop1 = passes = covers = 0
            cost = 0.0
            for _ in range(n):
                # stage 1
                msb1 = ev * rnd.gammavariate((k1 - 1) / 2.0, 2.0) / (k1 - 1)
                msw1 = sb ** 2 * rnd.gammavariate(k1 * (m - 1) / 2.0, 2.0) / (k1 * (m - 1))
                ub1 = _ub_exact_f(msb1, msw1, m, fq1, cw1, k1 * (m - 1))
                if msb1 > msw1 and gap >= 3 * ub1:
                    stop1 += 1
                    passes += 1
                    covers += (ub1 >= sj)
                    cost += k1 * m * 2
                    continue
                # stage 2: the extra jobs are fresh draws; the pooled mean
                # squares are the df-weighted combination.
                msb2 = ev * rnd.gammavariate((k2 - 1) / 2.0, 2.0) / (k2 - 1)
                msw2 = sb ** 2 * rnd.gammavariate(k2 * (m - 1) / 2.0, 2.0) / (k2 * (m - 1))
                msb = ((k1 - 1) * msb1 + (k2 - 1) * msb2) / db
                msw = (k1 * (m - 1) * msw1 + k2 * (m - 1) * msw2) / dw
                ub = _ub_exact_f(msb, msw, m, fq2, cw2, dw)
                cost += kk * m * 2
                if msb > msw and gap >= 3 * ub:
                    passes += 1
                covers += (ub >= sj)
            out["rows"].append({
                "arm": arm, "prior": pname, "sigma_job_prior_pct": sj,
                "sigma_boot_pct": sb, "gap_pct": gap,
                "p_stop_at_stage1": stop1 / n,
                "p_pass_overall": passes / n,
                "coverage_sequential": covers / n,
                "expected_boots": cost / n,
                "expected_gpu_hr": cost / n * 144.0 / 3600.0,
                "ACCEPT": (passes / n >= ACCEPT_P_PASS
                           and covers / n >= ACCEPT_COVERAGE),
                "n_mc": n,
            })
    return out


# ---- C-g: alternate the conc step INSIDE a boot ---------------------------
def paired_conc() -> dict:
    """Audit recommendation C-g, quantified.

    Today `s8_c2r.sbatch` fixes `--conc` for a whole boot, so Delta_batch =
    r(b_hi) - r(b_lo) is a BETWEEN-boot contrast and carries sigma_boot twice.
    Alternating the conc step inside one boot makes it a within-boot paired
    contrast; the boot effect cancels to the extent it is common to both steps.

    What this function can and cannot say (honesty gate): the cancellation
    factor depends on the within-boot residual, which HAS NOT BEEN MEASURED --
    that is exactly S0(a) (add a within-boot half-split reporting axis to
    `s8_c2r_score.py`, reusing its own bracket logic; re-deriving the estimand
    here would violate the no-new-estimand rule and gate #9).  So we report the
    ceiling (perfect cancellation) and the floor (none), and the boots needed
    for a target SE under each.
    """
    ps = pairsd()
    out = {"lever": "alternate --conc within a boot (s8_c2r.sbatch:303)",
           "unmeasured_input": ("within-boot residual SD of r; S0(a) measures it "
                                "via a half-split axis in s8_c2r_score.py"),
           "arms": {}}
    for arm in ("M8", "Ha8"):
        sb = ps["arms"][arm]["sigma_boot_paired_pct"]
        # between-boot contrast: SE = sb*sqrt(2/n_pairs); within-boot: SE =
        # sb_within*sqrt(2/n_pairs) with sb_within <= sb (equality = no
        # cancellation at all).
        need = lambda se_target, s: math.ceil(2 * (s / se_target) ** 2)  # noqa: E731
        out["arms"][arm] = {
            "sigma_boot_between_pct": sb,
            "boots_for_se_0p5pct_between": need(0.5, sb) * 2,
            "boots_for_se_0p5pct_within_if_half_cancels": need(0.5, sb / 2) * 2,
            "boots_for_se_0p5pct_within_if_full_cancels": (
                "bounded by the within-boot residual, unmeasured -- see S0(a)"),
            "note": ("the between-boot number is what a conc-fixed design pays; "
                     "the middle row is the illustrative half-cancellation case, "
                     "NOT a prediction"),
        }
    return out


def _rev2_positive_controls() -> dict:
    """Controls that can actually FAIL (gate #9, gate #44 -- an empty control set
    must not be able to pass).

    PC3  the paired sigma_boot and rho_leg reproduce the audit's independent
         reimplementation (`indep_audit.py`) to 3 dp.
    PC4  the F quantile reproduces TABLE values (a real external check).  The
         reciprocal identity F_{p,d1,d2} = 1/F_{1-p,d2,d1} is reported too but
         labelled for what it is: an identity, which validates the inversion and
         NOTHING about the CDF (gate #9 -- an identity is not evidence).
    PC5  the MC reproduces the audit's coverage numbers for the one cell the
         audit published (Ha8 k=4/m=3): 0.638 Satterthwaite, 0.970 exact-F.
    """
    ps = pairsd()
    checks = []
    for arm in ("M8", "Ha8"):
        got_sd = ps["arms"][arm]["sigma_boot_paired_pct"]
        got_rho = ps["arms"][arm]["rho_leg"]
        checks.append({"id": f"PC3/{arm}/sigma_boot_paired",
                       "target": AUDIT_TARGETS["sd_paired_pct"][arm],
                       "got": round(got_sd, 4),
                       "ok": abs(round(got_sd, 4)
                                 - AUDIT_TARGETS["sd_paired_pct"][arm]) <= 5e-4})
        checks.append({"id": f"PC3/{arm}/rho_leg",
                       "target": AUDIT_TARGETS["rho_leg"][arm],
                       "got": round(got_rho, 3),
                       "ok": abs(round(got_rho, 3)
                                 - AUDIT_TARGETS["rho_leg"][arm]) <= 5e-4})
    # PC4 -- external table values for F_{0.95, d1, d2}
    for (d1, d2, ref) in ((3, 8, 4.0662), (5, 10, 3.3258), (10, 20, 2.3479),
                          (1, 1, 161.448), (2, 30, 3.3158)):
        got = _f_quantile(0.95, d1, d2)
        checks.append({"id": f"PC4/table/F0.95({d1},{d2})", "target": ref,
                       "got": round(got, 4), "ok": abs(got - ref) < 5e-3})
    idc = _f_quantile(0.05, 8, 3) * _f_quantile(0.95, 3, 8)
    identity = {"id": "PC4/identity/F05(8,3)*F95(3,8)", "target": 1.0,
                "got": round(idc, 6), "ok": abs(idc - 1.0) < 1e-6,
                "IS_AN_IDENTITY": ("validates the inversion only; it cannot "
                                   "detect a wrong CDF (gate #9)")}
    # PC5 -- reproduce the audit's published coverage cell
    oc = _oc(4, 3, 1.1924, 0.5688, 9.196, n=40000)
    checks.append({"id": "PC5/cover_satterthwaite/Ha8_k4m3",
                   "target": AUDIT_TARGETS["cover_satterthwaite_Ha8_k4m3"],
                   "got": round(oc["rev1_satterthwaite"]["coverage"], 3),
                   "ok": abs(oc["rev1_satterthwaite"]["coverage"]
                             - AUDIT_TARGETS["cover_satterthwaite_Ha8_k4m3"]) < 0.012})
    checks.append({"id": "PC5/cover_exactF/Ha8_k4m3",
                   "target": AUDIT_TARGETS["cover_exactF_Ha8_k4m3"],
                   "got": round(oc["coverage"], 3),
                   "ok": abs(oc["coverage"]
                             - AUDIT_TARGETS["cover_exactF_Ha8_k4m3"]) < 0.012})
    if not checks:                       # gate #44: an empty control set FAILS
        return {"checks": [], "all_pass": False,
                "error": "empty positive-control set is not a pass"}
    return {"checks": checks, "identity_only": identity,
            "n_checks": len(checks),
            "all_pass": all(c["ok"] for c in checks),
            "note": ("PC5 uses the rev1 sigma_boot/prior on purpose -- it is "
                     "reproducing the AUDIT's published cell, not the rev2 "
                     "design point")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["bootvar", "reach", "power", "prior",
                                    "bslope", "all",
                                    "pairsd", "priorband", "power2", "seq2",
                                    "paired_conc", "controls", "plan_search",
                                    "rev2"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = {}
    if a.cmd in ("bootvar", "all"):
        res["bootvar"] = bootvar()
    if a.cmd in ("reach", "all"):
        res["reach"] = reach()
    if a.cmd in ("bslope", "all"):
        res["bslope"] = bslope()
    if a.cmd in ("prior", "all"):
        res["prior"] = prior()
    if a.cmd in ("power", "all"):
        bv = res.get("bootvar") or bootvar()
        # use the canonical (unpaired delta-method) boot dispersion -- the one
        # that reproduces CONSENSUS §3 item 56(A)'s t(5) intervals exactly.
        sb = (bv["arms"]["M8"]["unpaired_rel_sd_per_boot_pct"],
              bv["arms"]["Ha8"]["unpaired_rel_sd_per_boot_pct"])
        pr = res.get("prior") or prior()
        gp = bslope()["total_gaps_to_explain"]
        # rev2 F6: rev1's main() never passed target_pct, so the JSON silently
        # computed BOTH arms at 5.1% and could not produce the doc's Ha8 rows.
        # The per-arm run is emitted alongside the legacy call so the rev1
        # artifact stays reproducible and the defect stays visible.
        res["power"] = power(sigma_boot_pct=sb,
                             sigma_job_prior_pct=pr["sigma_job_hat_ratio_pct_from_rms"])
        res["power"]["stat_backend"] = "builtin incomplete-gamma bisection (no scipy)"
        res["power"]["F6_defect"] = (
            "this call uses target_gap_pct=5.1 for BOTH arms (rev1 behaviour, "
            "kept for reproducibility); Ha8's own gap is "
            f"{gp['Ha8']['pct_gap']:.3f}% -- see power_per_arm below")
        res["power_per_arm"] = {
            arm: power(sigma_boot_pct=sb, target_pct=gp[arm]["pct_gap"],
                       sigma_job_prior_pct=pr["sigma_job_hat_ratio_pct_from_rms"])
            for arm in ("M8", "Ha8")}
    # ---- rev2 (audit F1-F6 + C-g) ----------------------------------------
    if a.cmd in ("controls", "rev2"):
        res["positive_controls"] = _rev2_positive_controls()
    if a.cmd in ("pairsd", "rev2"):
        res["pairsd"] = pairsd()
    if a.cmd in ("priorband", "rev2"):
        res["priorband"] = priorband()
    if a.cmd in ("power2", "rev2"):
        res["power2"] = power2()
    if a.cmd in ("plan_search", "rev2"):
        res["plan_search_exactF_only"] = plan_search()
        res["plan_search"] = plan_search_v2()
    if a.cmd in ("seq2", "rev2"):
        res["seq2"] = seq2()
    if a.cmd in ("paired_conc", "rev2"):
        res["paired_conc"] = paired_conc()
    if a.cmd == "rev2":
        pc = res["positive_controls"]
        res["REV2_CONTROLS"] = "PASS" if pc["all_pass"] else "FAIL"
        if not pc["all_pass"]:
            # gate #42: a gate must not label its own failure a success, and the
            # overall verdict must actually REFERENCE every check.
            res["REV2_FAILED_CHECKS"] = [c for c in pc["checks"] if not c["ok"]]
    txt = json.dumps(res, indent=2, sort_keys=False, default=str)
    if a.out:
        open(os.path.join(HERE, a.out), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
