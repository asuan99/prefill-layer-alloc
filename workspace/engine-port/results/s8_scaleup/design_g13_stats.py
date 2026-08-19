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
# rev3: registered floor on the positive-control set, so that the gate #44
# "an empty control set is not a pass" branch is REACHABLE (rev2's was not).
MIN_POSITIVE_CONTROLS = 8
# Re-audit verdict (2026-08-19, section D): register the CHI-SQUARE UPPER band
# point.  Only that plan's collapse sigma_boot (3.892%) covers the whole n=6
# two-sided CI [0.979%, 3.847%]; mid's collapse point is 1.652%, i.e. 5% of
# headroom over the point estimate, and lo's is 1.164%, barely above the CI
# floor.  Over-estimating the design prior costs money only -- the campaign's
# own MS_W re-measures sigma_boot -- while under-estimating voids the campaign.
REGISTERED_SIGMA_BOOT_BAND_POINT = "hi_chi2_upper"
REGISTERED_PLAN_M8 = (16, 3, 60)     # k, m, window_s
REGISTERED_PLAN_HA8 = (16, 6, 60)
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


# ---- rev3 REPAIR (audit 2026-08-19, death cause B) ------------------------
# Boots that the SOURCE campaign (C2-R) had already named and quarantined.  A
# design that inherits C2-R's per-boot medians inherits this list too; rev2 did
# not, and so let one off-spec boot set the entire Ha8 cost.
C2R_NAMED_EXCLUSION = ("Ha8/d16/883575/blk2", "Ha8/d16/883575/blk5")


def sigma_boot_band() -> dict:
    """rev3.  sigma_boot is an n=6 ESTIMATE and must get the same band
    treatment sigma_job got in F5.  rev2 gave sigma_job three priors and
    sigma_boot a single point -- an asymmetry that mattered, because on the Ha8
    arm sigma_boot is the dominant cost driver (it alone decided 5.40 vs 8.64).

    Two facts the band has to carry:

    1. JACKKNIFE.  Dropping one boot moves Ha8's sigma_boot by 2.46x
       (1.5684% -> 0.6373%) while every other boot moves it 0.90-0.98x.  The
       whole of rho_leg = -0.767 lives in that one boot as well (-> -0.009).
       M8 is robust (0.89-1.30x over all six).
    2. THAT BOOT IS ALREADY QUARANTINED.  `C2R_SENSITIVITY_2026-08-16.json`
       lists `Ha8/d16/883575/blk5` in `named_exclusion.excluded`; C2-R
       diagnosed it off-spec (realization 0.508 vs 0.750-0.765,
       co_resident_frac_both 0.9359 -- the only cell below 1.0 in 24).
       C2-R survived it because C2-R took a POINT estimate (jackknife
       |delta| <= 0.43%).  rev2 took a VARIANCE from the same data, and a
       variance from n=6 is not robust to one outlier.

    The band is registered as a SENSITIVITY RANGE, not an estimate:
        lo   = jackknife MINIMUM -- the smallest sigma_boot obtainable by
               dropping any ONE boot.  B4 (re-audit): this is NOT "the value
               under a pre-registered filter".  C2-R's named_exclusion holds
               TWO boots (blk2 and blk5); dropping both gives 0.7232%, and
               dropping blk2 alone RAISES sigma_boot to 1.742%.  `lo` is
               therefore the most optimistic data-dependent choice available,
               and the re-audit ruled it NOT registrable on that ground
               (its collapse point, 1.164%, sits just above the CI floor).
               Reported as a sensitivity only.
        mid  = full-sample         (rev2's adopted value; keeps every boot)
        hi   = two-sided 95% chi2 upper on the full-sample estimate
    B6 (re-audit): the code does NOT require acceptance across the whole
    sigma_boot band -- `plan_search_v3` searches each band point INDEPENDENTLY
    and the pre-registration then buys ONE of them (see
    REGISTERED_SIGMA_BOOT_BAND_POINT).  `hi` is deliberately punitive: it is
    what "we do not know sigma_boot to better than n=6" actually costs, and
    the re-audit ruled it the point to register.
    """
    ps = pairsd()
    out = {"note": ("sigma_boot is an n=6 estimate; rev2 treated it as known. "
                    "Registered as a band for the same reason sigma_job is "
                    "(audit F5, applied symmetrically)."),
           "c2r_named_exclusion": list(C2R_NAMED_EXCLUSION), "arms": {}}
    for arm in ("M8", "Ha8"):
        r = ps["arms"][arm]["per_boot_r"]
        n = len(r)
        full = st.stdev(r) / st.fmean(r) * 100.0
        jack = []
        for i in range(n):
            sub = [x for j, x in enumerate(r) if j != i]
            jack.append({"dropped_index": i,
                         "dropped_label": f"{arm}/blk{i + 1}",
                         "sigma_boot_pct": st.stdev(sub) / st.fmean(sub) * 100.0})
        lo = min(j["sigma_boot_pct"] for j in jack)
        most_influential = min(jack, key=lambda j: j["sigma_boot_pct"])
        df = n - 1
        ci_lo = full * math.sqrt(df / _chi2_lower(0.975, df))
        ci_hi = full * math.sqrt(df / _chi2_lower(0.025, df))
        out["arms"][arm] = {
            "n_boot_pairs": n,
            "sigma_boot_full_pct": full,
            "jackknife": jack,
            "jackknife_min_pct": lo,
            "jackknife_max_ratio": full / lo,
            "most_influential_boot": most_influential["dropped_label"],
            # B4: query the list, do not hard-wire the answer.  The labels
            # here are `<arm>/blk<i>`; C2R_NAMED_EXCLUSION entries are full
            # run ids `<arm>/<leg>/<job>/blk<i>`, so match on (arm, blk).
            "most_influential_in_c2r_named_exclusion": any(
                e.split("/")[0] == arm
                and e.split("/")[-1] == most_influential["dropped_label"].split("/")[-1]
                for e in C2R_NAMED_EXCLUSION),
            "c2r_named_exclusion_for_this_arm": [
                e for e in C2R_NAMED_EXCLUSION if e.split("/")[0] == arm],
            "sigma_boot_excluding_all_named_pct": (
                (lambda keep: st.stdev(keep) / st.fmean(keep) * 100.0
                 if len(keep) >= 2 else None)(
                    [x for i, x in enumerate(r)
                     if not any(e.split("/")[0] == arm
                                and e.split("/")[-1] == f"blk{i + 1}"
                                for e in C2R_NAMED_EXCLUSION)])),
            "chi2_ci95_pct": [ci_lo, ci_hi],
            "BAND_PCT": {"lo_jackknife": lo, "mid_full": full, "hi_chi2_upper": ci_hi},
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
    n_guard_rev2 = 0            # rev3 diagnostic: how often rev2's threshold fired
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
        # --- rev3 REPAIR (audit 2026-08-19, death cause A) -------------------
        # rev2 guarded the exact-F branch at `msb <= msw`.  That is the wrong
        # threshold.  The exact-F bound degenerates when
        #     theta = ((MS_B/MS_W)/F_.05 - 1)/m  <=  0   <=>   MS_B/MS_W <= F_.05
        # and F_.05(df_B, df_W) < 1 for every plan here (0.3405 at (9,20),
        # 0.4472 at (15,32), 0.4067 at (11,96)).  So rev2 declared
        # MEASUREMENT_FAILURE over the whole interval (F_.05, 1], where the
        # bound is positive, finite and covering -- labelling a good
        # measurement as a failed one (methodology gate #21, reverse direction).
        # Each bound is now guarded at ITS OWN degeneracy point:
        #     Satterthwaite : MS_B <= MS_W          (UB == 0 -> automatic PASS,
        #                                            so F2 is exactly right here)
        #     exact-F       : MS_B <= F_.05 * MS_W  (theta <= 0 -> UB == 0)
        if msb <= msw:
            n_guard_rev2 += 1               # diagnostic only, not the rule
        if msb <= fq_lo * msw:              # rev3 guard: exact-F's own degeneracy
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
        "p_measurement_failure": n_fail / n,        # rev3 guard (exact-F own)
        "p_guard_rev2_msb_le_msw": n_guard_rev2 / n,   # what rev2 would have
        "guard_threshold_ratio": fq_lo,             # F_.05(df_B, df_W)
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
                    # rev3 REPAIR (death cause C): register the Satterthwaite
                    # multiplier at the REGISTERED prior, exactly as
                    # _eval_plan does.  rev2 called _oc without sat_prior here,
                    # so power2's Satterthwaite column was registered at the
                    # true prior -- a quantity the campaign cannot know.
                    oc = _oc(k, m, sb0 * scale, sj, gap,
                             sat_prior=REGISTERED_SAT_PRIOR)
                    rows.append({
                        "arm": arm, "k_job": k, "m_bootpair": m,
                        "measure_window_s": win, "n_boots": boots,
                        "gpu_hr": round(boots * sec_per_boot / 3600.0, 3),
                        "gap_pct": gap, "prior": pname, "sigma_job_prior_pct": sj,
                        "sigma_boot_source": sbname,
                        "sigma_boot_pct": sb0 * scale,
                        "window_scaling_assumed": win != 60,
                        **oc,
                        # rev3 REPAIR (death cause C): the row used to report
                        # ONLY the exact-F column while the registered rule may
                        # pick Satterthwaite.  Both are now printed side by
                        # side, and ACCEPT_registered is the one that matches
                        # the rule this design pre-registers.
                        "p_pass_satterthwaite_guarded":
                            oc["rev1_satterthwaite"]["p_pass_guarded"],
                        "coverage_satterthwaite":
                            oc["rev1_satterthwaite"]["coverage"],
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
            # rev3 REPAIR (death cause C): evaluate the plan under the rule the
            # design actually pre-registers, not under a hard-wired exact-F.
            sb60 = ps["arms"][arm]["sigma_boot_paired_pct"]
            ev = _eval_plan(k, m, sb60 * math.sqrt(60.0 / win),
                            sorted(priors.values()), gaps[arm]["pct_gap"], MC_N)
            robust[f"{arm}/k{k}/m{m}/{win}s"] = {
                "accept_at_all_priors": all(r["ACCEPT"] for r in sel),
                "accept_count": sum(r["ACCEPT"] for r in sel),
                "n_priors": len(sel),
                "gpu_hr": sel[0]["gpu_hr"] if sel else None,
                "worst_p_pass": min((r["p_pass"] for r in sel), default=None),
                "worst_coverage": min((r["coverage"] for r in sel), default=None),
                "max_p_measurement_failure": max(
                    (r["p_measurement_failure"] for r in sel), default=None),
                "REGISTERED_RULE": {
                    "bound": ev["bound"], "accept": ev["accept"],
                    "worst_p_pass": ev["worst_p_pass"],
                    "worst_coverage": ev["worst_coverage"],
                    "note": ("this is the plan's operating characteristic under "
                             "the pre-registered bound-selection rule; the "
                             "exact-F columns above are diagnostic only"),
                },
            }
    return {"acceptance_rule": {
                "p_pass_min": ACCEPT_P_PASS, "coverage_min": ACCEPT_COVERAGE,
                "verdict_on_degenerate_draw": (
                    "MEASUREMENT_FAILURE (never PASS), guarded at each bound's "
                    "OWN degeneracy point: Satterthwaite at MS_B <= MS_W, "
                    "exact-F at MS_B <= F_.05(df_B,df_W)*MS_W  [rev3 repair]"),
                "upper_bound": (
                    "pre-registered SELECTION rule (see _eval_plan): "
                    "Satterthwaite if its simulated coverage >= 0.93 at every "
                    "prior in the band, else exact-F pivot.  rev2 printed "
                    "exact-F unconditionally in this table, which is NOT the "
                    "registered rule -- see REGISTERED_RULE per plan.")},
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
    # B5 (re-audit): report the guard rate belonging to the SELECTED bound.
    # rev3 changed the guard's meaning to "each bound at its own degeneracy
    # point", so `p_measurement_failure` (exact-F's) is the wrong field to
    # quote when Satterthwaite was chosen -- there the guard is
    # `p_guard_rev2_msb_le_msw`, which is Satterthwaite's OWN degeneracy point.
    fail_key = ("p_guard_rev2_msb_le_msw" if chosen == "satterthwaite_registered"
                else "p_measurement_failure")
    return {"bound": chosen, "accept": acc, "worst_p_pass": worst_p,
            "worst_coverage": worst_c,
            "max_p_measurement_failure": max(o[fail_key] for o in ocs),
            "measurement_failure_field": fail_key,
            "max_p_measurement_failure_exact_f": max(
                o["p_measurement_failure"] for o in ocs),
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


def plan_search_v3(n: int = 8000) -> dict:
    """rev3.  Same search as v2, run SEPARATELY at each sigma_boot band point
    (death cause B), under the repaired guard (death cause A).

    B6 (re-audit): "adoptable across the band" would be a different and
    stronger rule; this function does not implement it.  Each band point gets
    its own cheapest plan, and the pre-registration buys one point.

    Reported per sigma_boot band point rather than collapsed to one number, so
    the pre-registration can state explicitly WHICH band point it is buying and
    what the other two would cost.  Nothing here picks that for you: choosing
    the band point after seeing the costs would be the post-hoc knob F5 exists
    to prevent.
    """
    ps = pairsd()
    band = priorband()
    sbb = sigma_boot_band()
    gaps = bslope()["total_gaps_to_explain"]
    priors = sorted(v["pct"] for v in band.values()
                    if isinstance(v, dict) and v.get("pct") is not None)
    out = {"priors_pct": priors,
           "guard": ("rev3: each bound guarded at its own degeneracy point "
                     "(exact-F at MS_B <= F_.05*MS_W, Satterthwaite at "
                     "MS_B <= MS_W)"),
           "registered_sat_prior_pct": REGISTERED_SAT_PRIOR, "arms": {}}
    for arm in ("M8", "Ha8"):
        gap = gaps[arm]["pct_gap"]
        per_band = {}
        for bname, sb60 in sbb["arms"][arm]["BAND_PCT"].items():
            feasible = []
            for win in (60, 120, 240):
                sb = sb60 * math.sqrt(60.0 / win)
                sec = 144.0 + (win - 60.0)
                for k in (4, 6, 8, 10, 12, 16, 20, 24):
                    for m in (3, 6, 9, 12):
                        ev = _eval_plan(k, m, sb, priors, gap, n)
                        if ev["accept"]:
                            feasible.append({
                                "k_job": k, "m_bootpair": m,
                                "measure_window_s": win, "n_boots": k * m * 2,
                                "bound": ev["bound"],
                                "gpu_hr": round(k * m * 2 * sec / 3600.0, 3),
                                "window_scaling_assumed": win != 60,
                                "worst_p_pass": ev["worst_p_pass"],
                                "worst_coverage": ev["worst_coverage"],
                                "max_p_measurement_failure":
                                    ev["max_p_measurement_failure"]})
            feasible.sort(key=lambda r: r["gpu_hr"])
            confirmed, rejected = None, []
            for cand in feasible[:8]:
                sb = sb60 * math.sqrt(60.0 / cand["measure_window_s"])
                e = _eval_plan(cand["k_job"], cand["m_bootpair"], sb, priors,
                               gap, MC_N, seed=MC_SEED + 991)
                rec = {**cand, "confirm_bound": e["bound"],
                       "confirm_worst_p_pass": e["worst_p_pass"],
                       "confirm_worst_coverage": e["worst_coverage"],
                       "confirm_max_p_measurement_failure":
                           e["max_p_measurement_failure"]}
                if e["accept"]:
                    confirmed = rec
                    break
                rejected.append(rec)
            # The 60 s plan matters on its own: if it exists, the campaign does
            # not need the unverified sigma_boot ~ 1/sqrt(T) window scaling at
            # all, and S0(a) leaves the critical path.
            no_assume = None
            for cand in [f for f in feasible if not f["window_scaling_assumed"]][:8]:
                e = _eval_plan(cand["k_job"], cand["m_bootpair"], sb60, priors,
                               gap, MC_N, seed=MC_SEED + 991)
                if e["accept"]:
                    no_assume = {**cand, "confirm_worst_p_pass": e["worst_p_pass"],
                                 "confirm_worst_coverage": e["worst_coverage"],
                                 "confirm_max_p_measurement_failure":
                                     e["max_p_measurement_failure"]}
                    break
            per_band[bname] = {
                "sigma_boot_60s_pct": sb60,
                "n_feasible_in_sweep": len(feasible),
                "rejected_on_confirmation": rejected,
                "ADOPTED": confirmed,
                "cheapest_60s_no_window_assumption": no_assume,
            }
        out["arms"][arm] = {"gap_pct": gap, "per_sigma_boot_band": per_band}
    # Totals per band point, using the 60 s (assumption-free) plan where one
    # exists -- that is the plan rev3 recommends registering.
    tot = {}
    for bname in ("lo_jackknife", "mid_full", "hi_chi2_upper"):
        vals = []
        for arm in ("M8", "Ha8"):
            e = out["arms"][arm]["per_sigma_boot_band"][bname]
            pick = e["cheapest_60s_no_window_assumption"] or e["ADOPTED"]
            vals.append(pick["gpu_hr"] if pick else None)
        tot[bname] = (round(sum(vals), 2) if all(v is not None for v in vals)
                      else None)
    out["total_gpu_hr_both_arms_60s_by_band"] = tot
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
                if msb1 > fq1 * msw1 and gap >= 3 * ub1:   # rev3 guard (B1)
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
                if msb > fq2 * msw and gap >= 3 * ub:      # rev3 guard (B1)
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


def _rev3_extra_controls() -> list:
    """rev3.  Controls the audit found MISSING from rev2 (its finding on
    criterion 3).

    PC6  `_chi2_lower` against EXTERNAL table values.  rev2 had NO control on
         this function at all, yet the whole M8 bound is
         sqrt(df_eff / _chi2_lower(0.05, df_eff)) -- 42% of the campaign budget
         resting on an unchecked special function.  rev1 claimed a 6-digit
         check but neither the code nor the JSON contains one (a reproduction
         path that does not exist is not a control).
    PC7  the ADOPTED plan's own operating characteristic, recomputed from a
         different seed.  rev2's only MC control (PC5) was on a cell that is
         not any adopted plan.
    PC8  the rev3 guard repair, stated as a falsifiable inequality:
         F_.05(df_B, df_W) must be < 1 for every plan in the search grid --
         that is exactly why rev2's `MS_B <= MS_W` threshold discarded a live
         region.  If this were >= 1 anywhere the repair would be wrong there.
    """
    checks = []
    # PC6 -- external chi-square table values (Abramowitz & Stegun / NIST)
    for (nu, p, ref) in ((5, 0.025, 0.8312), (5, 0.05, 1.1455),
                         (5, 0.975, 12.8325), (1, 0.05, 0.003932),
                         (10, 0.05, 3.9403), (32, 0.05, 20.072),
                         (96, 0.05, 74.397)):
        got = _chi2_lower(p, nu)
        tol = max(5e-3, abs(ref) * 2e-3)
        checks.append({"id": f"PC6/table/chi2_{p}({nu})", "target": ref,
                       "got": round(got, 5), "ok": abs(got - ref) < tol})
    # PC7 -- adopted-plan OC, fresh seed, at the BINDING corner.
    # B8 (re-audit): rev3 evaluated this at prior 0.4112 and sigma_boot 1.5684
    # -- the plan's EASIEST point (p_pass 0.968), so it could not fail.  The
    # binding corner is the worst prior (0.9542) at the REGISTERED band point
    # (hi_chi2_upper), which is where the acceptance decision actually lives.
    sbb = sigma_boot_band()
    sb_reg = sbb["arms"]["Ha8"]["BAND_PCT"][REGISTERED_SIGMA_BOOT_BAND_POINT]
    gap_ha8 = bslope()["total_gaps_to_explain"]["Ha8"]["pct_gap"]
    band = priorband()
    prior_worst = max(v["pct"] for v in band.values()
                      if isinstance(v, dict) and v.get("pct") is not None)
    oc = _oc(REGISTERED_PLAN_HA8[0], REGISTERED_PLAN_HA8[1], sb_reg,
             prior_worst, gap_ha8, n=40000, seed=MC_SEED + 4242,
             sat_prior=REGISTERED_SAT_PRIOR)
    checks.append({"id": "PC7/adopted_plan_oc_at_binding_corner/Ha8",
                   "target": (">=%.2f p_pass and >=%.2f coverage at prior %.4f, "
                              "sigma_boot %.4f (band point %s), plan k=%d m=%d"
                              % (ACCEPT_P_PASS, ACCEPT_COVERAGE, prior_worst,
                                 sb_reg, REGISTERED_SIGMA_BOOT_BAND_POINT,
                                 REGISTERED_PLAN_HA8[0], REGISTERED_PLAN_HA8[1])),
                   "got": {"p_pass": round(oc["p_pass"], 3),
                           "coverage": round(oc["coverage"], 3),
                           "p_measurement_failure":
                               round(oc["p_measurement_failure"], 3)},
                   "ok": (oc["p_pass"] >= ACCEPT_P_PASS
                          and oc["coverage"] >= ACCEPT_COVERAGE)})
    # PC8 was REMOVED from the pass set by the 2026-08-19 re-audit.
    #   "F_.05(d1,d2) < 1 for every (k,m)" is a THEOREM, not a control:
    #   F_.05(d1,d2) = 1/F_.95(d2,d1) and F_.95 > 1 always, so it cannot fail
    #   anywhere on any grid.  It was in `all_pass`, i.e. an identity was being
    #   counted as evidence -- methodology gate #9, 13th recurrence, and the
    #   very error rev2 got right by isolating its own F-inversion identity.
    #   The claim rev3 wrote next to it ("if it were >= 1 the repair would be
    #   wrong there") is also FALSE: the guard `msb <= fq_lo*msw` is exactly
    #   the degeneracy point whatever the size of F_.05.  It survives as a
    #   labelled identity only.  See `_rev3_identities()`.
    return checks


def _rev3_identities() -> list:
    """Statements that are TRUE BY CONSTRUCTION and therefore prove nothing.

    Kept visible (rev2's own good practice) but never counted in `all_pass`.
    """
    f05 = {f"({k - 1},{k * (m - 1)})": _f_quantile(0.05, k - 1, k * (m - 1))
           for k in (4, 12, 24) for m in (3, 12)}
    return [
        {"id": "IDENT/F05_lt_1_everywhere",
         "statement": "F_.05(d1,d2) < 1 for all df",
         "why_identity": ("F_.05(d1,d2) = 1/F_.95(d2,d1) and F_.95 > 1 always; "
                          "no grid can violate it"),
         "sample_values": f05},
        {"id": "IDENT/exact_f_ub_zero_iff_guarded",
         "statement": "UB_exactF <= 0  <=>  MS_B <= F_.05*MS_W",
         "why_identity": ("_ub_exact_f returns sqrt(max(0, theta)*...), so the "
                          "equivalence is the definition of max(0, .) -- it "
                          "cannot detect a typo in _oc's guard line")},
        {"id": "IDENT/satterthwaite_ub_zero_iff_msb_le_msw",
         "statement": "UB_sat == 0  <=>  MS_B <= MS_W",
         "why_identity": ("sqrt(max(0,(MS_B-MS_W)/m)) == 0 iff MS_B <= MS_W; "
                          "restating max(0,.) says nothing about whether _oc "
                          "still guards the Satterthwaite branch there")},
    ]


def _finalize_controls(checks: list, identity: dict = None) -> dict:
    """gate #44: a control set that is empty -- or below the registered floor --
    must FAIL rather than vacuously pass.

    rev3 REPAIR (audit criterion 3): in rev2 this logic sat inline inside
    `_rev2_positive_controls`, where `checks` was filled unconditionally by the
    loops above it.  The branch was therefore UNREACHABLE and proved nothing
    about itself (gate #9, weak recurrence: a defence that cannot be exercised
    is not a defence).  Pulling it out makes it callable -- and the self-test
    S1 now actually calls it with a short set and requires a FAIL.
    """
    if not checks or len(checks) < MIN_POSITIVE_CONTROLS:
        return {"checks": checks, "n_checks": len(checks), "all_pass": False,
                "error": ("positive-control set below the registered minimum "
                          "(%d < %d) -- not a pass"
                          % (len(checks), MIN_POSITIVE_CONTROLS))}
    out = {"checks": checks, "n_checks": len(checks),
           "all_pass": all(c["ok"] for c in checks),
           "note": ("PC5 uses the rev1 sigma_boot/prior on purpose -- it is "
                    "reproducing the AUDIT's published cell, not the rev2 "
                    "design point")}
    if identity is not None:
        out["identity_only"] = identity
    return out


def _rev3_selftest() -> dict:
    """rev3.  Tests of the REPAIRS themselves, each able to fail.

    S1  the gate #44 branch is REACHABLE: hand it a short control set and it
        must return all_pass=False with an error.  (In rev2 no input could
        reach that branch, so it never demonstrated anything.)
    S2  the guard repair changes what it is supposed to change and nothing
        else: at the Ha8 adopted basis the rev2 threshold fires far more often
        than the true degeneracy, and P(pass) rises accordingly.
    S3  the repaired guard is never LOOSER than the true degeneracy point --
        a draw with theta <= 0 must always be counted as a measurement failure,
        because there the bound is 0 and would auto-PASS.
    S4  Satterthwaite keeps rev2's guard: for that bound MS_B <= MS_W IS the
        degeneracy point, so the repair must not have moved it.
    """
    checks = []
    # S1 -- reachability of the gate #44 branch: CALL it, do not assert about it
    empty = _finalize_controls([])
    short = _finalize_controls([{"id": "probe", "ok": True}])
    full = _finalize_controls([{"id": f"probe{i}", "ok": True}
                               for i in range(MIN_POSITIVE_CONTROLS)])
    checks.append({"id": "S1/gate44_branch_reachable",
                   "target": ("empty and below-floor sets must FAIL; a set at "
                              "the floor must be able to pass"),
                   "got": {"empty_all_pass": empty["all_pass"],
                           "short_all_pass": short["all_pass"],
                           "at_floor_all_pass": full["all_pass"],
                           "floor": MIN_POSITIVE_CONTROLS},
                   "ok": (empty["all_pass"] is False
                          and short["all_pass"] is False
                          and full["all_pass"] is True)})
    # S2 -- the repair moves the guard, in the right direction and size
    oc = _oc(10, 3, 0.7842, 0.4112, 9.196, n=20000)
    checks.append({"id": "S2/guard_repair_effect/Ha8_k10m3",
                   "target": "rev2 guard rate >> rev3 guard rate, F_.05 < 1",
                   "got": {"rev2_rate": round(oc["p_guard_rev2_msb_le_msw"], 4),
                           "rev3_rate": round(oc["p_measurement_failure"], 4),
                           "F05": round(oc["guard_threshold_ratio"], 4)},
                   "ok": (oc["p_guard_rev2_msb_le_msw"]
                          > oc["p_measurement_failure"]
                          and oc["guard_threshold_ratio"] < 1.0)})
    # S3 (rebuilt after re-audit) -- the guard rate that _oc ACTUALLY produces
    # must equal the analytic 5% tail of F(df_B, df_W).
    #
    # rev3's first S3 was an identity: it re-implemented `msb <= fq_lo*msw` and
    # checked that UB<=0 agreed with it, which max(0,.) guarantees.  Worse, it
    # never called _oc, so a typo in the real guard line was invisible to it.
    # This version calls _oc and compares against a route _oc does not use:
    # at sigma_job = 0 the ratio MS_B/MS_W is exactly F(df_B, df_W), so the
    # guard must fire with probability 0.05 by the DEFINITION of the 0.05
    # quantile.  Revert the guard to `msb <= msw` and this reads ~0.4-0.5.
    for (k, m) in ((10, 3), (16, 3), (12, 9)):
        oc0 = _oc(k, m, 1.0, 0.0, 9.196, n=40000, seed=MC_SEED + 77)
        got = oc0["p_measurement_failure"]
        se = math.sqrt(0.05 * 0.95 / 40000)
        checks.append({"id": f"S3/guard_rate_matches_F_tail/k{k}m{m}",
                       "target": "P(guard fires | sigma_job=0) == 0.05 +- 4se",
                       "got": {"observed": round(got, 4),
                               "expected": 0.05, "se": round(se, 5)},
                       "ok": abs(got - 0.05) <= 4 * se})
    # S4 (rebuilt) -- the Satterthwaite branch must STILL guard at MS_B <= MS_W.
    # rev3's first S4 restated max(0,.) and never looked at _oc's Satterthwaite
    # counter at all.  At sigma_job = 0 that guard must fire with probability
    # P(F(df_B,df_W) <= 1), which _f_cdf gives independently of the MC.
    for (k, m) in ((10, 3), (16, 3)):
        oc0 = _oc(k, m, 1.0, 0.0, 9.196, n=40000, seed=MC_SEED + 78)
        expect = _f_cdf(1.0, k - 1, k * (m - 1))
        got = oc0["p_guard_rev2_msb_le_msw"]
        se = math.sqrt(max(expect * (1 - expect), 1e-9) / 40000)
        checks.append({"id": f"S4/satterthwaite_guard_unmoved/k{k}m{m}",
                       "target": "P(MS_B<=MS_W | sigma_job=0) == F_cdf(1) +- 4se",
                       "got": {"observed": round(got, 4),
                               "expected": round(expect, 4),
                               "se": round(se, 5)},
                       "ok": abs(got - expect) <= 4 * se})
    return {"checks": checks, "n_checks": len(checks),
            "all_pass": all(c["ok"] for c in checks)}


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
    checks = _rev3_extra_controls()
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
    return _finalize_controls(checks, identity)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["bootvar", "reach", "power", "prior",
                                    "bslope", "all",
                                    "pairsd", "priorband", "power2", "seq2",
                                    "paired_conc", "controls", "plan_search",
                                    "rev2",
                                    "sigma_boot_band", "plan_search_v3", "rev3",
                                    "selftest"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true",
                    help="allow --out to overwrite an existing file (B7)")
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
    # ---- rev3 (audit 2026-08-19: death causes A/B/C + criteria 3/4/5) -----
    if a.cmd in ("controls", "rev3"):
        res["positive_controls"] = _rev2_positive_controls()
    if a.cmd in ("sigma_boot_band", "rev3"):
        res["sigma_boot_band"] = sigma_boot_band()
    if a.cmd in ("pairsd", "rev3"):
        res["pairsd"] = pairsd()
    if a.cmd in ("priorband", "rev3"):
        res["priorband"] = priorband()
    if a.cmd in ("power2", "rev3"):
        res["power2"] = power2()
    if a.cmd in ("plan_search_v3", "rev3"):
        res["plan_search_v3"] = plan_search_v3()
    if a.cmd == "rev3":
        # B1 (re-audit): sec 2.4 justifies "no extra blocks" by citing seq2,
        # so seq2 must be IN this artifact -- and must be the repaired one.
        res["seq2"] = seq2()
        # B2: the discarded identities stay VISIBLE and labelled, never counted.
        # (A helper that is defined but never emitted is the same dead-code
        # pattern the re-audit flagged in C2R_NAMED_EXCLUSION -- so wire it.)
        res["identities_not_counted"] = _rev3_identities()
    if a.cmd == "selftest":
        res["selftest"] = _rev3_selftest()
    if a.cmd == "rev3":
        res["selftest"] = _rev3_selftest()
        pc = res["positive_controls"]
        res["REV3_CONTROLS"] = "PASS" if pc["all_pass"] else "FAIL"
        res["REV3_SELFTEST"] = ("PASS" if res["selftest"]["all_pass"]
                                else "FAIL")
        if not pc["all_pass"]:
            res["REV3_FAILED_CHECKS"] = [c for c in pc["checks"] if not c["ok"]]
        if not res["selftest"]["all_pass"]:
            res["REV3_FAILED_SELFTESTS"] = [c for c in res["selftest"]["checks"]
                                            if not c["ok"]]
    if a.cmd == "rev2":
        # rev3 REPAIR NOTICE: `_oc` (the guard) and `power2` (the bound) were
        # repaired on 2026-08-19 after the rule-layer audit.  Both are SHARED
        # with the rev2 path, so this subcommand no longer reproduces the
        # rev2-of-record.  The artifact of record for rev2 is the stored
        # DESIGN_G13_STATS_REV2_2026-08-17.json, kept unmodified; git history
        # holds the pre-repair source.  Re-running `rev2` today yields rev2's
        # LAYOUT with rev3's REPAIRED numbers -- do not cite it as rev2.
        res["REV2_REPRODUCTION_WARNING"] = (
            "NOT the rev2-of-record: _oc/power2 were repaired 2026-08-19 "
            "(death causes A and C) and are shared with this path. Cite "
            "DESIGN_G13_STATS_REV2_2026-08-17.json for rev2, or "
            "DESIGN_G13_STATS_REV3_2026-08-19.json for the repaired design.")
        pc = res["positive_controls"]
        res["REV2_CONTROLS"] = "PASS" if pc["all_pass"] else "FAIL"
        if not pc["all_pass"]:
            # gate #42: a gate must not label its own failure a success, and the
            # overall verdict must actually REFERENCE every check.
            res["REV2_FAILED_CHECKS"] = [c for c in pc["checks"] if not c["ok"]]
    txt = json.dumps(res, indent=2, sort_keys=False, default=str)
    if a.out:
        # B7 (re-audit): rev2's own doc prints
        #   `design_g13_stats.py rev2 --out DESIGN_G13_STATS_REV2_2026-08-17.json`
        # as its reproduction path.  Since rev3 repaired _oc/power2/seq2, which
        # the rev2 path SHARES, following that instruction would overwrite the
        # committed rev2-of-record with repaired numbers -- and the only
        # warning would sit inside the file just destroyed.  Refuse instead.
        dest = os.path.join(HERE, a.out)
        if os.path.exists(dest) and not a.force:
            sys.stderr.write(
                "REFUSING to overwrite an existing artifact: %s\n"
                "  rev3 repaired _oc/power2/seq2, which the rev2 path shares, so\n"
                "  re-running an old command would replace a committed record\n"
                "  with repaired numbers. Write to a NEW filename, or pass\n"
                "  --force if you truly intend to replace it.\n" % dest)
            raise SystemExit(2)
        open(dest, "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
