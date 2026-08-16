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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["bootvar", "reach", "power", "prior",
                                    "bslope", "all"])
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
        res["power"] = power(sigma_boot_pct=sb,
                             sigma_job_prior_pct=pr["sigma_job_hat_ratio_pct_from_rms"])
        res["power"]["stat_backend"] = "builtin incomplete-gamma bisection (no scipy)"
    txt = json.dumps(res, indent=2, sort_keys=False, default=str)
    if a.out:
        open(os.path.join(HERE, a.out), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
