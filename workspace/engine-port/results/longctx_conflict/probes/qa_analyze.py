#!/usr/bin/env python3
"""Registered outputs 1-10 of PREREG_QA_REGRET_REV7_2026-09-10.md  (`Q-A REGRET`).

GPU 0.  This file is a CALCULATOR.  It produces NO performance verdict, NO arm
ranking and NO policy statement.  Per rev5 sec 1 / rev6 6-13 the following are
STRUCTURALLY ABSENT and must never be added here:

    min / argmin / envelope / `E-hat` / ranking / "which arm is best"
    gap rank-invariance test (DELETED by rev7 sec 6 -- do not re-add)

------------------------------------------------------------------------------
WHAT IS REGISTERED, AND WHERE THE REGISTRATION IS AMBIGUOUS
------------------------------------------------------------------------------
(1) THE FOUR CELL ESTIMATORS.  rev5 sec 6 item 1 names them
    (`pooled_p95`/`median`/`mean`/`trim10`) but does not name the ARRAY they
    are functionals of.  There are two candidate axes in this repo:

      axis P  "pooled token"   : all four are functionals of the cell's
                                 TOKEN-LEVEL ITL array (ms).
      axis R  "per-request"    : `p7_analyze.itl95_estimators` -- median/mean/
                                 trim10 over the per-REQUEST itl95 values,
                                 pooled_p95 over the token array (mixed axes).

    ** Every registered forecast in rev6 sec 3 and in the rev6 verdict sec 2-(2)
    was computed on axis P **, verified here to 4 decimals for all four
    estimators and all three P7 pairs (see `selfcheck`).  Axis P is also the
    only axis on which rev6 sec 1(d)'s cliff quantiles are coherent: rev7 sec 2
    registers `u`'s domain as the token ITL array, and 0.50 / 0.10-0.90 / 0.95
    are exactly the quantiles of THAT array that `median` / `trim10` /
    `pooled_p95` sit on.  Axis P is therefore PRIMARY here.

    Axis R is NOT discarded: `p7_analyze.itl95_estimators` is imported verbatim
    and reported next to axis P in every cell, so the two can never silently
    diverge.  `pooled_p95` is identical on both axes by construction.
    !! This ambiguity is a PREREG WORDING DEFECT, not an analyst choice.  It is
    reported, not resolved.  There is deliberately no CLI flag to switch axes.

(2) `trim10` := mean of the values in [p10, p90] of the array (the parent
    task's wording, "p10-p90 절사").  This reproduces the verdict's
    +6.6642 / +6.2233 to 4 decimals; the count-based variant inside
    `itl95_estimators` (drop int(0.1n) from each end) differs in the 4th decimal
    whenever 0.1n is not an integer.  Both are reported.

(3) 1-SEED RUNGS.  rev7 sec 3 registers the cross-design SE as
    sqrt(sigma_boot^2/4 + sigma_seed^2/8 + sigma_req^2/8) with "(1시드 rung은
    n = 4)" and stops there.  Reading used here: on a 1-seed rung boot and seed
    are not separable, so sigma_cell^2 (variance over the 4 boot values) takes
    the place of both and n = 4:  sqrt(sigma_cell^2/4 + sigma_req^2/4), t_3.
    Flagged in the output as `variance_mode` = "sigma_cell_only".  No decision
    hinges on the split of a variance that is not identified.

(4) The registered SE ADDS sigma_req^2/8 on top of sigma_seed^2/8, but
    sigma_seed^2 (the within-boot variance across seeds) already contains the
    request-sampling component.  The registered form is computed as written and
    is primary; the sigma_req-free form (= rev6 sec 6's 0.0718 / 0.4922 ms) is
    reported next to it as `se_without_req_diagnostic` so the size of the
    double count is visible.  Not a correction -- a disclosure.

------------------------------------------------------------------------------
CROSS-CHECK STATUS (gate #9: a check that is an identity checks nothing)
------------------------------------------------------------------------------
  REAL   : L_pre + L_decode  vs the independently reported `concurrency`
           field (threshold 1e-4).
  IDENTITY, NOT A CHECK: `completed / duration` vs `request_throughput`.
           bench_serving.py defines request_throughput := completed / dur_s,
           so agreement to 0.0e+00 validates nothing.  Computed and reported
           as provenance only; never counted as corroboration.

Usage:
  python3 qa_analyze.py --selfcheck                    # reproduce known numbers
  python3 qa_analyze.py <run_dir> [--out result.json]  # campaign outputs 1-10
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# Imported VERBATIM (gate #9: do not reimplement the canonical instrument).
from p7_analyze import itl95_estimators, load_bench, model_a_span_factor  # noqa: E402
from span_factor import span_factor  # noqa: E402

# =============================================================================
# Registered constants -- PREREG_QA_REGRET rev5 sec 3/sec 5, rev6 sec 1, rev7
# =============================================================================
PRIMARY_AXIS = "pooled_token"          # see docstring (1).  No CLI override.
EST = ("pooled_p95", "median", "mean", "trim10")

T_WIN = 150.0                          # rev5 sec 2.1
LAMBDA_RUNGS = (0.09, 0.1162, 0.15, 0.25, 0.42, 0.70)          # rev5 sec 4
A_OF_LAMBDA = {0.09: (16, 44, 54, 92), 0.1162: (92,),
               0.15: (16, 44, 54, 92), 0.25: (16, 44, 54),
               0.42: (16, 44, 54), 0.70: (16,)}                # rev5 sec 4
SEEDS_OF_LAMBDA = {0.09: (70, 67), 0.1162: (70,), 0.15: (61, 68),
                   0.25: (63,), 0.42: (71,), 0.70: (61,)}      # rev5 sec 3
N_OF_LAMBDA = {lam: int(round(lam * T_WIN)) + 1 for lam in LAMBDA_RUNGS}

# rev5 sec 5 -- ONE table for grid, r_cap and rho.  Sources are NOT averaged.
MU_P_REF = {16: 0.8962, 44: 0.6632, 54: 0.5736, 92: 0.1885}
MU_P_REF_SRC = {16: "P7 saturation bench, n_boot=4",
                44: "P7 saturation bench, n_boot=4",
                54: "P7 saturation bench, n_boot=4",
                92: "Step 1 (job 905958), n_boot=1 -- repo's only measurement"}

# rev6 sec 1(a) / rev7 sec 1 -- mode extraction, fixed by fiat.
BIN_W_MS = 2.0                 # registered bin width; 1 and 4 are DIAGNOSTIC
BIN_W_DIAG_MS = (1.0, 4.0)
HIST_HI_MS = 200.0             # range [0, 200)
FAR_MS = 4.0                   # "4 ms 이상 떨어진 빈"

# rev6 sec 1(d) -- estimator-wise cliff labels.  `mean` has no cliff.
CLIFF = {"median": ((0.50,), 0.10),
         "trim10": ((0.10, 0.90), 0.05),
         "pooled_p95": ((0.95,), 0.05)}

BOOT_N = 10000                 # repo convention (pdmux_eval.analyze)
BOOT_SEED = 1                  # repo convention
CROSSCHECK_TOL = 1e-4          # rev5 sec 6 item 10

ITL_SOLO_MS = 13.03689880296588       # P2 job 905712, ctx 8192, concurrency 1
UNTREATED_TOL = 1.10                  # qa_ladder_loo convention

# rev6 sec 5 + rev7 sec 5 erratum 5.  SCOPE: ctx = 8192, arm-tagged bench files
# only.  `probes/p2_905712` runs a d44 configuration without an arm tag and
# reaches X = 0.3257 at ctx 16384 -- outside this scope, cited in the table.
REPO_MIN_X = {16: 0.4245, 44: 0.4213, 54: 0.4186}
REPO_MIN_X_SCOPE = ("ctx=8192, arm-tagged bench files only (rev7 sec 5 erratum 5); "
                    "p2_905712 (d44 config, no arm tag) measured X=0.3257 at ctx 16384")

# Mandatory companion sentences -- rev7 sec 8 (3 items, verbatim in results).
MANDATORY_COMPANIONS = [
    "untreated floor is arm-INDEPENDENT (per-request median-ITL min 13.016-13.079 ms, "
    "min TTFT 0.883-0.903 s across all arms; mechanism CONSENSUS sec 3 item 26 = "
    "multiplexing_mixin.py:952-953)",
    "|A(lambda)| is a strictly DECREASING function of treatment intensity: the 4-arm "
    "rungs are the two lowest loads (realised rho(d16) 0.098/0.113/0.160/0.163), "
    "3-arm 0.288/0.457, 1-arm 0.811",
    "low-load extrapolation exposure: 18/19 registered cells sit below the repo's "
    "lowest observed X (" + REPO_MIN_X_SCOPE + ")",
]

# =============================================================================
# Distribution quantiles without scipy (scipy is NOT importable on this node).
# Validated in `selfcheck` against the two values the repo already uses:
#   F_{0.05}(3,8) = 0.1131   (RESULT_P7 sec 2)
#   F_{0.95}(3,4) = 6.591    (rev6 sec 6)
#   t_{0.975}(3)  = 3.1824   (rev6 sec 6 / verdict sec 2-(3))
# =============================================================================
def _betacf(a, b, x, itmax=300, eps=3e-16):
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < 1e-300:
        d = 1e-300
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


def betai(a, b, x):
    """Regularised incomplete beta I_x(a,b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
             + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lbeta) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbeta) * _betacf(b, a, 1.0 - x) / b


def f_cdf(x, d1, d2):
    if x <= 0:
        return 0.0
    return betai(d1 / 2.0, d2 / 2.0, d1 * x / (d1 * x + d2))


def f_ppf(q, d1, d2, lo=1e-12, hi=1e12):
    for _ in range(400):
        mid = math.sqrt(lo * hi)
        if f_cdf(mid, d1, d2) < q:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


def t_ppf(q, df):
    """Two-sided-friendly Student-t quantile via the incomplete beta."""
    lo, hi = 0.0, 1e6
    for _ in range(400):
        mid = 0.5 * (lo + hi)
        cdf = 1.0 - 0.5 * betai(df / 2.0, 0.5, df / (df + mid * mid))
        if cdf < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# =============================================================================
# Estimators
# =============================================================================
def trim_p10_p90(v):
    """Registered `trim10`: mean of the values inside [p10, p90]."""
    v = np.asarray(v, dtype=float)
    lo, hi = np.percentile(v, 10), np.percentile(v, 90)
    sel = v[(v >= lo) & (v <= hi)]
    return float(sel.mean()) if sel.size else float("nan")


def pooled_token_estimators(itls):
    """axis P (PRIMARY): all four estimators as functionals of the cell's
    token-level ITL array, in ms.  `u` (rev7 sec 2) lives on this same axis,
    which is why the cliff quantiles 0.50 / 0.10-0.90 / 0.95 are coherent."""
    v = np.concatenate([np.asarray(x, dtype=float) for x in itls if x.size]) * 1000.0
    return {"median": float(np.percentile(v, 50)),
            "mean": float(v.mean()),
            "trim10": trim_p10_p90(v),
            "pooled_p95": float(np.percentile(v, 95)),
            "n_tokens": int(v.size),
            "axis": "pooled_token_itl_ms"}


def cell_estimators(itls):
    """Both axes, always.  Never reconciled, never silently resolved."""
    axis_r = itl95_estimators(itls)          # imported verbatim
    axis_r["axis"] = "per_request_itl95_ms (p7_analyze.itl95_estimators, VERBATIM)"
    return {"pooled_token_axis": pooled_token_estimators(itls),
            "per_request_itl95_axis": axis_r}


# =============================================================================
# Mode extraction (rev6 sec 1(a)(b) + rev7 sec 1) -- NO rank-invariance test.
# =============================================================================
def mode_extract(req_median_ms, bin_w=BIN_W_MS, far=FAR_MS, hi=HIST_HI_MS):
    """Histogram of the cell's PER-REQUEST median-ITL (a DIFFERENT axis from
    `u`'s -- rev7 sec 2).  Returns m_low/m_high/gap/valley_depth/c_valley, or
    u_defined=False when no bin >= `far` ms from the mode holds any sample
    (rev7 sec 1, the registered third branch)."""
    v = np.asarray(req_median_ms, dtype=float)
    nb = int(round(hi / bin_w))
    cnt, _ = np.histogram(v, bins=nb, range=(0.0, hi))
    ctr = (np.arange(nb) + 0.5) * bin_w
    n_out_of_range = int(((v < 0) | (v >= hi)).sum())
    if cnt.max() == 0:
        return {"u_defined": False, "reason": "histogram empty",
                "bin_width_ms": bin_w, "n_out_of_range": n_out_of_range}
    i1 = int(np.argmax(cnt))
    c1 = float(ctr[i1])
    ties_mode = [float(ctr[i]) for i in np.flatnonzero(cnt == cnt[i1])]
    far_mask = np.abs(ctr - c1) >= far
    if (not far_mask.any()) or cnt[far_mask].max() == 0:
        return {"u_defined": False,
                "reason": "no sample in any bin >= %.1f ms from the mode "
                          "(rev7 sec 1 registered branch)" % far,
                "mode_bin_center_ms": c1, "bin_width_ms": bin_w,
                "support_ms": [float(v.min()), float(v.max())],
                "far_bin_count": 0, "n_out_of_range": n_out_of_range}
    idx = np.flatnonzero(far_mask)
    j = int(idx[int(np.argmax(cnt[idx]))])
    c2 = float(ctr[j])
    ties_far = [float(ctr[i]) for i in idx if cnt[i] == cnt[j]]
    m_low, m_high = (c1, c2) if c1 <= c2 else (c2, c1)
    between = (ctr > m_low) & (ctr < m_high)
    vd = (1.0 - float(cnt[between].min()) / float(min(cnt[i1], cnt[j]))
          if between.any() else None)
    out = {"u_defined": True, "bin_width_ms": bin_w,
           "m_low": m_low, "m_high": m_high, "gap": m_high - m_low,
           "c_valley": (m_low + m_high) / 2.0, "valley_depth": vd,
           "mode_bin_count": int(cnt[i1]), "far_bin_count": int(cnt[j]),
           "far_bin_below_mode": bool(c2 < c1),
           "n_out_of_range": n_out_of_range}
    # UNREGISTERED point: bin-count ties.  Reported, never resolved -- the rev6
    # verdict row A pushed this and `u` was identical to 4 decimals in both
    # tie cells of the repo.  Default = lowest centre (np.argmax = first index).
    if len(ties_mode) > 1 or len(ties_far) > 1:
        alts = sorted({(min(a, b) + max(a, b)) / 2.0
                       for a in ties_mode for b in ties_far if abs(a - b) >= far})
        out["tie_unregistered"] = True
        out["tie_mode_centres_ms"] = ties_mode
        out["tie_far_centres_ms"] = ties_far
        out["c_valley_tie_alternatives"] = alts
    return out


def u_stat(token_itl_ms, c_valley):
    """rev7 sec 2 (T2): u := #{ITL < c_valley} / #ITL over the cell's TOKEN-unit
    ITL array -- NOT the per-request median-ITL array used for mode extraction."""
    v = np.asarray(token_itl_ms, dtype=float)
    return float((v < c_valley).mean())


def cliff_labels(u):
    """rev6 sec 1(d).  A label does NOT suppress the value (rev6 sec 2 / S2):
    it attaches the sentence 'this value's movement is a move between the two
    modes, not a change in latency itself'."""
    if u is None:
        return {e: None for e in EST} | {"note": "u undefined -> no label (rev7 sec 1)"}
    out = {}
    for e in EST:
        if e not in CLIFF:
            out[e] = False        # `mean` has no cliff
            continue
        anchors, tol = CLIFF[e]
        out[e] = bool(any(abs(u - a) < tol for a in anchors))
    out["note"] = ("mode-pinned = report the value AND the sentence; "
                   "NOT a citation ban (rev6 sec 2)")
    return out


# =============================================================================
# Request-sampling bootstrap (10000 samples, seed 1 -- repo convention)
# =============================================================================
def _ragged_pack(itls):
    arrs = [np.asarray(x, dtype=float) * 1000.0 for x in itls if x.size]
    lens = np.array([a.size for a in arrs])
    return arrs, lens, bool(len(set(lens.tolist())) == 1)


def _boot_estimates(arrs, lens, rectangular, idx, chunk=500):
    """Four axis-P estimators for each bootstrap resample of REQUESTS."""
    n, R = idx.shape
    out = np.empty((n, 4), dtype=float)
    if rectangular:
        M = np.stack(arrs)
        for a in range(0, n, chunk):
            sel = idx[a:a + chunk]
            s = np.sort(M[sel].reshape(sel.shape[0], -1), axis=1)
            lo = np.percentile(s, 10, axis=1)
            hi = np.percentile(s, 90, axis=1)
            tr = np.array([row[(row >= l) & (row <= h)].mean()
                           for row, l, h in zip(s, lo, hi)])
            out[a:a + sel.shape[0]] = np.stack(
                [np.percentile(s, 95, axis=1), np.percentile(s, 50, axis=1),
                 s.mean(axis=1), tr], axis=1)
    else:                       # ragged (rare: 1 token short on 1 request)
        for i in range(n):
            s = np.sort(np.concatenate([arrs[j] for j in idx[i]]))
            lo, hi = np.percentile(s, 10), np.percentile(s, 90)
            out[i] = (np.percentile(s, 95), np.percentile(s, 50), s.mean(),
                      s[(s >= lo) & (s <= hi)].mean())
    return out                  # column order = EST


def request_bootstrap(itls, n=BOOT_N, seed=BOOT_SEED):
    arrs, lens, rect = _ragged_pack(itls)
    rs = np.random.RandomState(seed)
    idx = rs.randint(0, len(arrs), size=(n, len(arrs)))
    b = _boot_estimates(arrs, lens, rect, idx)
    base = pooled_token_estimators(itls)
    hw, sd = {}, {}
    for i, e in enumerate(EST):
        lo, hi = np.percentile(b[:, i], 2.5), np.percentile(b[:, i], 97.5)
        hw[e] = float((hi - lo) / 2.0)
        sd[e] = float(b[:, i].std(ddof=1))
    return {"n_requests": int(len(arrs)), "n_boot_samples": n, "seed": seed,
            "rectangular": rect,
            "half_width_ms": hw, "sd_ms": sd,
            "half_width_pct": {e: 100.0 * hw[e] / base[e] for e in EST},
            "axis": "pooled_token (PRIMARY)"}


def paired_ttft_bootstrap(ttft_a, ttft_b, q=95, n=BOOT_N, seed=BOOT_SEED):
    """sigma_req of the paired difference of TTFT p_q, same resampled request
    indices for both arms."""
    a = np.asarray(ttft_a, dtype=float)
    b = np.asarray(ttft_b, dtype=float)
    m = min(a.size, b.size)
    rs = np.random.RandomState(seed)
    idx = rs.randint(0, m, size=(n, m))
    d = np.percentile(a[:m][idx], q, axis=1) - np.percentile(b[:m][idx], q, axis=1)
    return float(d.std(ddof=1))


def paired_request_bootstrap(itls_a, itls_b, n=BOOT_N, seed=BOOT_SEED):
    """sigma_req of the PAIRED difference: the same resampled request indices
    are used for both arms (the two benches share the arrival trace because
    they share the seed), so the request-sampling component is paired exactly
    the way the estimand is."""
    aa, la, ra = _ragged_pack(itls_a)
    ab, lb, rb = _ragged_pack(itls_b)
    m = min(len(aa), len(ab))
    if len(aa) != len(ab):
        note = ("UNEQUAL request counts (%d vs %d) -- paired on the first %d "
                "indices; index alignment across arms is only exact when both "
                "benches completed the same requests" % (len(aa), len(ab), m))
    else:
        note = None
    aa, ab = aa[:m], ab[:m]
    rs = np.random.RandomState(seed)
    idx = rs.randint(0, m, size=(n, m))
    d = (_boot_estimates(aa, la[:m], ra and len(set(la[:m].tolist())) == 1, idx)
         - _boot_estimates(ab, lb[:m], rb and len(set(lb[:m].tolist())) == 1, idx))
    out = {e: float(d[:, i].std(ddof=1)) for i, e in enumerate(EST)}
    if note:
        out["note"] = note
    return out


# =============================================================================
# Variance components -- gate #116 format
# =============================================================================
def variance_components(groups, alpha=0.05):
    """One-way random effects ANOVA: groups = per boot, list of per-seed values.

    gate #116: a negative raw variance component is NOT a point estimate of 0,
    it is 'UNDETECTED at this n'.  The quotable object is the F-test 95% upper
    bound   sigma_boot^2 <= sigma_seed^2 * (F / F_{0.05}(a-1, a(n-1)) - 1) / n
    (the convention RESULT_P7 sec 2 already uses: F_{0.05}(3,8) = 0.1131), and
    when F < F_{0.05} the bound itself is DEGENERATE and must not be quoted as
    '0'.
    """
    g = [np.asarray(x, dtype=float) for x in groups if len(x) >= 1]
    if len(g) < 2:
        return {"status": "fewer than 2 boots -- decomposition undefined"}
    a = len(g)
    n = min(len(x) for x in g)
    means = np.array([x.mean() for x in g])
    grand = float(means.mean())
    if n < 2:
        s2_cell = float(means.var(ddof=1))
        return {"variance_mode": "sigma_cell_only", "n_boot": a, "n_seed": 1,
                "grand_mean": grand, "sigma_cell": math.sqrt(s2_cell),
                "sigma_cell_rel_pct": (100 * math.sqrt(s2_cell) / grand) if grand else None,
                "note": "1-seed rung: boot and seed are NOT separable "
                        "(rev5 sec 3 caveat); sigma_cell carries both"}
    within = [float(x.var(ddof=1)) for x in g if x.size >= 2]
    s2_seed = float(np.mean(within))
    s2_between = float(means.var(ddof=1))
    raw = s2_between - s2_seed / n
    clipped = raw < 0
    s2_boot = max(raw, 0.0)
    ms_b = n * s2_between
    F = (ms_b / s2_seed) if s2_seed > 0 else float("inf")
    df1, df2 = a - 1, a * (n - 1)
    f_lo = f_ppf(alpha, df1, df2)
    ratio_ub = (F / f_lo - 1.0) / n if f_lo > 0 else float("nan")
    degenerate = not (ratio_ub > 0)
    ub = math.sqrt(max(ratio_ub, 0.0) * s2_seed)
    return {
        "variance_mode": "boot_seed_decomposed", "n_boot": a, "n_seed": n,
        "grand_mean": grand,
        "sigma_seed": math.sqrt(s2_seed),
        "sigma_seed_rel_pct": (100 * math.sqrt(s2_seed) / grand) if grand else None,
        "sigma_boot_point": math.sqrt(s2_boot),
        "sigma_boot_raw_variance": raw,
        "sigma_boot_status": ("UNDETECTED at this n (raw component negative -> "
                              "clipped to 0; NOT a point estimate of 0)"
                              if clipped else "positive raw component"),
        "F": F, "F_df": [df1, df2], "F_lower_crit_0.05": f_lo,
        "sigma_boot_95_upper": (None if degenerate else ub),
        "sigma_boot_95_upper_rel_pct": (None if degenerate else
                                        (100 * ub / grand if grand else None)),
        "upper_bound_degenerate": bool(degenerate),
        "upper_bound_note": ("F < F_{0.05} => the upper bound is DEGENERATE, "
                             "not 0.0% (RESULT_P7 sec 7)" if degenerate else None),
        "boot_means": means.tolist(),
    }


# =============================================================================
# Paired difference matrix + sign_agree (rev5 sec 1, rev6 sec 3, rev7 sec 3)
# =============================================================================
def rung_ci(cell_deltas, sigma_req, alpha=0.05):
    """rev7 sec 3 (T3): CI at RUNG level, cell index WITHOUT the seed.

        SE = sqrt(sigma_boot^2/n_boot + sigma_seed^2/n_cells + sigma_req^2/n_cells)
        t_3  (the boot component carries df = 3 at the registered n_boot = 4)

    `cell_deltas`: {boot: {seed: delta}}.  See docstring (3) and (4).
    """
    boots = sorted(cell_deltas)
    per_boot = [[cell_deltas[b][s] for s in sorted(cell_deltas[b])] for b in boots]
    vals = np.array([v for row in per_boot for v in row], dtype=float)
    n_cells = vals.size
    n_boot = len(per_boot)
    n_seed = min(len(r) for r in per_boot)
    means = np.array([np.mean(r) for r in per_boot])
    point = float(means.mean())          # balanced -> = mean of all cell deltas
    df_boot = n_boot - 1
    tcrit = t_ppf(1 - alpha / 2.0, df_boot)
    se_common = None
    if n_seed >= 2:
        s2_seed = float(np.mean([np.var(r, ddof=1) for r in per_boot if len(r) >= 2]))
        s2_between = float(means.var(ddof=1))
        s2_boot = max(s2_between - s2_seed / n_seed, 0.0)
        se = math.sqrt(s2_boot / n_boot + s2_seed / n_cells + sigma_req ** 2 / n_cells)
        se_no_req = math.sqrt(s2_boot / n_boot + s2_seed / n_cells)
        mode = "boot_seed_decomposed"
    else:
        s2_cell = float(means.var(ddof=1))
        s2_boot, s2_seed = float("nan"), float("nan")
        se = math.sqrt(s2_cell / n_boot + sigma_req ** 2 / n_boot)
        se_no_req = math.sqrt(s2_cell / n_boot)
        # rev9 sec 2 (P3): on a 1-seed rung all 4 boots share ONE arrival trace,
        # so the request-sampling component is COMMON across boots and dividing
        # it by n_boot understates the SE by up to 2x (anti-conservative, rev8
        # verdict 6-5 / sec 4-4).  The registered form (above) is NOT changed;
        # this symmetric diagnostic is carried alongside it and NEITHER is
        # selected -- choosing one would itself be a new free surface.
        se_common = math.sqrt(s2_cell / n_boot + sigma_req ** 2)
        mode = "sigma_cell_only"
    hw = tcrit * se
    out = {"point": point, "n_boot": n_boot, "n_seed_per_boot": n_seed,
           "n_cells": n_cells, "variance_mode": mode,
           "sigma_boot": (math.sqrt(s2_boot) if s2_boot == s2_boot else None),
           "sigma_seed": (math.sqrt(s2_seed) if s2_seed == s2_seed else None),
           "sigma_req": sigma_req,
           "se_registered": se, "se_without_req_diagnostic": se_no_req,
           "t_crit": tcrit, "df_boot": df_boot,
           "ci95": [point - hw, point + hw], "half_width": hw,
           "excludes_zero": bool((point - hw > 0) or (point + hw < 0)),
           "sd_of_boot_means": float(means.std(ddof=1)),
           "ci95_boot_level_diagnostic": [
               point - tcrit * means.std(ddof=1) / math.sqrt(n_boot),
               point + tcrit * means.std(ddof=1) / math.sqrt(n_boot)],
           }
    if se_common is not None:
        hw_c = tcrit * se_common
        out["se_common_trace_diagnostic"] = se_common
        out["ci95_common_trace_diagnostic"] = [point - hw_c, point + hw_c]
        out["half_width_common_trace_diagnostic"] = hw_c
        out["excludes_zero_common_trace_diagnostic"] = bool(
            (point - hw_c > 0) or (point + hw_c < 0))
        out["common_trace_note"] = (
            "rev9 sec 2 (P3): sqrt(sigma_cell^2/4 + sigma_req^2).  Carried NEXT TO "
            "the registered SE, never instead of it.  BOTH are reported; no rule "
            "picks between them (rev9 sec 2).")
    else:
        out["se_common_trace_diagnostic"] = None
        out["common_trace_note"] = ("not applicable: this rung has >= 2 seeds per "
                                    "boot, so the trace is not common across boots")
    return out


def sign_agree(ci_by_est):
    """rev6 sec 3 / rev7 sec 3: how many of the four estimators have a CI that
    excludes 0 AND share a sign.  Value only; the permission it carries
    ('sign statement allowed at 4') is the reader's, not this file's."""
    pos = [e for e in EST if ci_by_est[e]["excludes_zero"] and ci_by_est[e]["point"] > 0]
    neg = [e for e in EST if ci_by_est[e]["excludes_zero"] and ci_by_est[e]["point"] < 0]
    n = max(len(pos), len(neg))
    return {"sign_agree": n, "n_excluding_zero": len(pos) + len(neg),
            "positive": pos, "negative": neg,
            "sign_conflict": bool(pos and neg)}


# =============================================================================
# Per-cell analysis (output 1, 6, 7, 9, 10)
# =============================================================================
# Filename tokens.  `qa_regret.sbatch` writes
#   bench_r<R>_d<D>_s<A|B>_l<lambda>_seed<S>.jsonl
#   sat_r<R>_d<D>_s<A|B>.jsonl
#   tel_r<R>_d<D>.jsonl                    <-- ONE PER BOOT, not per cell
# The older probes (P7 / probe C / S1) use bench_r<R>_d<D>_s<seed>.jsonl and
# bench_d<D>_r<rung>_o<outlen>.jsonl; both are accepted so the self-check and
# the campaign run through the same parser.
_TOK = {"round": r"(?:^|_)r(\d+)(?=_|\.)",
        "arm": r"(?:^|_)d(\d+)(?=_|\.)",
        "seed": r"(?:^|_)s(?:eed)?(\d+)(?=_|\.)",
        "out": r"(?:^|_)o(\d+)(?=_|\.)",
        "lam": r"(?:^|_)l(?:am)?([0-9]+(?:\.[0-9]+)?)(?=_|\.jsonl|$)"}
_TOK_SHAPE = r"(?:^|_)s([AB])(?=_|\.)"


def parse_tokens(name):
    out = {}
    for k, pat in _TOK.items():
        m = re.search(pat, name)
        out[k] = (float(m.group(1)) if k == "lam" else int(m.group(1))) if m else None
    m = re.search(_TOK_SHAPE, name)
    out["shape_tag"] = m.group(1) if m else None
    return out


def match_rung(rate, tol=0.02):
    cands = [lam for lam in LAMBDA_RUNGS if abs(lam - rate) <= tol * max(lam, 1e-9)]
    return cands[0] if len(cands) == 1 else None


def analyse_cell(bench_path, arm=None, telemetry=None, seed=None, boot_n=BOOT_N):
    d, ttfts, itls = load_bench(bench_path)
    ok = [i for i, e in enumerate(d["errors"]) if e == ""]
    assert len(ok) == len(ttfts), "load_bench ok-predicate drifted"
    name = Path(bench_path).name
    tok = parse_tokens(name)
    arm = arm if arm is not None else tok["arm"]
    seed = seed if seed is not None else tok["seed"]
    n = len(d["ttfts"])
    dur = float(d["duration"])
    rate = float(d["request_rate"])
    tok_itl_ms = np.concatenate([x for x in itls if x.size]) * 1000.0
    req_med_ms = np.array([np.median(x) * 1000.0 for x in itls if x.size])
    req_itl95_ms = np.array([np.percentile(x, 95) * 1000.0 for x in itls if x.size])

    L_dec = float(sum(float(x.sum()) for x in itls)) / dur
    L_pre = float(ttfts.sum()) / dur
    reported = d.get("concurrency", float("nan"))
    x_recomputed = float(d["completed"]) / dur
    span_hat = dur - (d["ttfts"][-1] + float(np.sum(d["itls"][-1])))
    nominal_span = (n - 1) / rate                     # rev7 sec 5 erratum 4
    lam = tok["lam"] if tok["lam"] is not None else match_rung(rate)
    sf_pred = span_factor(seed, n) if seed is not None else None

    md = mode_extract(req_med_ms)
    u = u_stat(tok_itl_ms, md["c_valley"]) if md.get("u_defined") else None
    diag_modes = {("bin_%.0fms" % w): mode_extract(req_med_ms, bin_w=w)
                  for w in BIN_W_DIAG_MS}

    cell = {
        "bench_file": name, "arm": arm, "seed": seed, "round": tok["round"],
        "rung_lambda": lam, "shape_out_len": int(d["random_output_len"]),
        # shape is read from the DATA (random_output_len), then cross-checked
        # against the filename tag -- a disagreement is reported, not resolved.
        "shape": ("A" if d["random_output_len"] == 96 else
                  ("B" if d["random_output_len"] == 384 else "UNREGISTERED")),
        "shape_tag_in_filename": tok["shape_tag"],
        "shape_tag_agrees_with_out_len": (
            None if tok["shape_tag"] is None else
            tok["shape_tag"] == ("A" if d["random_output_len"] == 96 else
                                 ("B" if d["random_output_len"] == 384 else "?"))),
        "ctx_in_len": int(d["random_input_len"]),
        "offered_rate": rate, "completed": int(d["completed"]),
        "n_requests_ok": int(req_med_ms.size), "n_errors": int(n - len(ttfts)),
        "duration_s": dur,
        # --- output 1 ---
        "estimators": cell_estimators(itls),
        "ttft_p50_s": float(np.percentile(ttfts, 50)),
        "ttft_p95_s": float(np.percentile(ttfts, 95)),
        "ttft_p99_s": float(np.percentile(ttfts, 99)),
        "X_completed_over_duration": x_recomputed,
        "achieved_over_offered": x_recomputed / rate,
        "L_pre_exact": L_pre, "L_decode_exact": L_dec,
        "request_bootstrap_ci": request_bootstrap(itls, n=boot_n),
        "mode_extraction": md,
        "u": u,
        "u_domain": "token-unit ITL array of the cell (rev7 sec 2 T2) -- NOT the "
                    "per-request median-ITL array that mode extraction uses",
        "c_valley": md.get("c_valley"),
        "mode_pinned_labels": cliff_labels(u),
        "mode_extraction_diagnostic_bins": diag_modes,   # rev7 sec 6: DIAGNOSTIC only
        # --- output 6 ---
        "makespan_s": dur,
        # --- output 7 ---
        "span_hat_s": span_hat,
        "span_factor_observed": span_hat / nominal_span,
        "span_factor_predicted": sf_pred,
        "arrival_window_predicted_s": (nominal_span * sf_pred if sf_pred else None),
        "arrival_window_formula": "(N-1)/lambda * span_factor  (rev7 sec 5 erratum 4; "
                                  "NOT 150 * span_factor)",
        "span_factor_rel_err_pct_vs_pred": (
            abs(sf_pred - span_hat / nominal_span) / sf_pred * 100.0 if sf_pred else None),
        "span_factor_model_a": model_a_span_factor(seed, n) if seed is not None else None,
        # --- output 8 ---
        "low_load_extrapolation": {
            "repo_min_X_same_arm": REPO_MIN_X.get(arm),
            "below_repo_min": (bool(x_recomputed < REPO_MIN_X[arm])
                               if arm in REPO_MIN_X else None),
            "scope": REPO_MIN_X_SCOPE},
        # --- output 10 (REAL cross-check) ---
        "L_total_reported": reported,
        "crosscheck_rel_err": (abs(L_pre + L_dec - reported) / reported
                               if reported and reported == reported else float("nan")),
        "crosscheck_pass": (abs(L_pre + L_dec - reported) / reported < CROSSCHECK_TOL
                            if reported and reported == reported else None),
        # gate #9: identity, not a check.  Provenance only.
        "X_identity_vs_request_throughput": {
            "value": float(d["request_throughput"]),
            "rel_diff": abs(x_recomputed - d["request_throughput"]) / d["request_throughput"],
            "warning": "ALGEBRAIC IDENTITY (bench_serving defines request_throughput "
                       ":= completed/duration).  Never cite as corroboration."},
    }
    # --- output 9: treatment duty cycle w, TWO routes, never reconciled ---
    w = {"route_A_mixture_token_fraction_above_valley":
         (None if u is None else 1.0 - u),
         "route_A_note": "1 - u on the token axis; undefined when u is undefined",
         "route_A_request_untreated_fraction":
             float((req_itl95_ms <= UNTREATED_TOL * ITL_SOLO_MS).mean()),
         "route_A_untreated_threshold_ms": UNTREATED_TOL * ITL_SOLO_MS,
         "rho_gap_registered": 4.754,
         "rho_gap_note": "rho(d16)/rho(d92) = mu_p(d92)/mu_p(d16) is an algebraic "
                         "identity, equal on every rung (rev6 sec 4)",
         "f_time_3.8x": "UNVERIFIED (rev6 sec 4) -- do not cite",
         "route_B_scope": "PER BOOT, NOT PER CELL -- see `duty_cycle_route_B_by_boot`",
         "route_B_boot_key": [tok["round"], arm]}
    cell["duty_cycle_w"] = w
    cell["_itls"] = itls          # popped before serialisation
    cell["_ttfts"] = ttfts
    return cell


def _hist_get(hist, k):
    """`compute_decode_realized` returns `decode_hist` with INT keys; the same
    dict read back from a JSON file has STR keys.  Looking up only one of the
    two silently returns 0 -- which is a plausible-looking duty cycle.  Try
    both."""
    for kk in (k, str(k), int(k)):
        if kk in hist:
            return float(hist[kk])
    return 0.0


def duty_cycle_route_B(run_dirs):
    """Output 9, route B.  `qa_regret.sbatch` writes ONE telemetry file per BOOT
    (`tel_r<R>_d<D>.jsonl`), which spans that boot's 12 main benches AND its
    saturation bench(es).  There is no per-cell marker and no shared clock
    between the server's `timestamp_monotonic_s` and the client's bench window,
    so the `decode_hist` time fractions CANNOT be attributed to a single cell.

    They are therefore reported at BOOT level with that scope stated.  Splitting
    a boot-level aggregate across cells would be exactly the interval-imputation
    artifact this track has already been bitten by (CONSENSUS sec 3 item 120,
    gate #103, lesson item 100).  Route A (the mixture decomposition) is the
    per-cell quantity; the two are printed side by side and never reconciled.
    """
    out = []
    for rd in run_dirs:
        for p in sorted(glob.glob(str(Path(rd) / "tel_r*_d*.jsonl"))):
            t = parse_tokens(Path(p).name)
            entry = {"telemetry_file": Path(p).name, "run_dir": str(rd),
                     "round": t["round"], "arm": t["arm"],
                     "scope": "WHOLE BOOT (all main benches + saturation bench(es) "
                              "of this (round, arm)) -- NOT per cell"}
            try:
                import analyze_probes
                a = analyze_probes.analyze_one(p, int(t["arm"]))
                hist = a.get("decode_hist") or {}
                tot = float(sum(hist.values())) or float("nan")
                t108 = _hist_get(hist, 108)
                entry.update({
                    "decode_hist": {str(k): v for k, v in hist.items()},
                    "decode_hist_108_time_s": t108,
                    "decode_hist_total_time_s": tot,
                    "decode_hist_108_time_fraction": (t108 / tot if tot == tot else None),
                    "split_fired_time_fraction": (
                        1.0 - t108 / tot if tot == tot else None),
                    "f_time": a.get("f_time"), "f_count": a.get("f_count"),
                    "n_split_total": a.get("n_split_total"),
                    "n_split_pab0": a.get("n_split_pab0"),
                    "audit": a.get("audit")})
            except Exception as exc:
                entry["error"] = "%s: %s" % (type(exc).__name__, exc)
            out.append(entry)
    return out


def registration_conformance(cells):
    """Fact report (not a decision rule): does the executed cell set equal the
    registered grid?  rev5 sec 3 (seeds), sec 4 (A(lambda)), sec 2.1 (N).
    Missing and unregistered-extra cells are both listed; nothing is dropped."""
    executed = {(c["rung_lambda"], c["shape"], c["arm"], c["seed"], c["round"])
                for c in cells}
    expected = set()
    for lam, arms in A_OF_LAMBDA.items():
        for sh in ("A", "B"):                       # rev5 sec 4: A(lam,B) := A(lam,A)
            for arm in arms:
                for sd in SEEDS_OF_LAMBDA[lam]:
                    for rnd in (1, 2, 3, 4):        # rev7 sec 9: 16 boots / 4 rounds
                        expected.add((lam, sh, arm, sd, rnd))
    missing = sorted(expected - executed, key=str)
    extra = sorted(executed - expected, key=str)
    n_obs = {}
    for c in cells:
        n_obs.setdefault((c["rung_lambda"], c["arm"]), set()).add(c["n_requests_ok"])
    return {
        "n_expected_cells": len(expected), "n_executed_cells": len(executed),
        "n_missing": len(missing), "n_unregistered_extra": len(extra),
        "missing_cells_lam_shape_arm_seed_round": missing[:200],
        "unregistered_extra_cells": extra[:200],
        "N_registered_by_rung": {str(k): v for k, v in N_OF_LAMBDA.items()},
        "N_observed_ok_by_rung_arm": {str(k): sorted(v) for k, v in sorted(
            n_obs.items(), key=str)},
        "note": "reported as a fact; this function makes no decision and drops no cell",
    }


# =============================================================================
# Campaign driver (outputs 2-5)
# =============================================================================
def build_delta_matrix(cells, boot_n=BOOT_N):
    """Output 2: paired difference matrix.  Keys are ORDERED pairs; there is no
    min, no argmin, no envelope and no ranking anywhere in this function."""
    by_key = {}
    for c in cells:
        by_key[(c["rung_lambda"], c["shape"], c["round"], c["seed"], c["arm"])] = c
    rungs = sorted({k[0] for k in by_key if k[0] is not None})
    shapes = sorted({k[1] for k in by_key})
    matrix = []
    for lam in rungs:
        for sh in shapes:
            arms = sorted({k[4] for k in by_key if k[0] == lam and k[1] == sh})
            for D in arms:
                for Dp in arms:
                    if D == Dp:
                        continue
                    d_itl = {}
                    d_itl_R = {}
                    d_ttft = {}
                    d_make = {}
                    sreq = {e: [] for e in EST}
                    sreq_ttft = []
                    for (l2, s2, rnd, sd, a) in list(by_key):
                        if (l2, s2, a) != (lam, sh, D):
                            continue
                        other = by_key.get((lam, sh, rnd, sd, Dp))
                        if other is None:
                            continue
                        me = by_key[(lam, sh, rnd, sd, D)]
                        e1 = me["estimators"]["pooled_token_axis"]
                        e2 = other["estimators"]["pooled_token_axis"]
                        r1 = me["estimators"]["per_request_itl95_axis"]
                        r2 = other["estimators"]["per_request_itl95_axis"]
                        d_itl.setdefault(rnd, {})[sd] = {e: e1[e] - e2[e] for e in EST}
                        d_itl_R.setdefault(rnd, {})[sd] = {e: r1[e] - r2[e] for e in EST}
                        d_ttft.setdefault(rnd, {})[sd] = (me["ttft_p95_s"] - other["ttft_p95_s"])
                        d_make.setdefault(rnd, {})[sd] = (me["makespan_s"] - other["makespan_s"])
                        pb = paired_request_bootstrap(me["_itls"], other["_itls"], n=boot_n)
                        for e in EST:
                            sreq[e].append(pb[e] ** 2)
                        sreq_ttft.append(paired_ttft_bootstrap(
                            me["_ttfts"], other["_ttfts"], n=boot_n) ** 2)
                    if not d_itl:
                        continue
                    ci = {}
                    for e in EST:
                        per = {b: {s: d_itl[b][s][e] for s in d_itl[b]} for b in d_itl}
                        ci[e] = rung_ci(per, math.sqrt(float(np.mean(sreq[e]))))
                    ttft_ci = rung_ci(d_ttft, math.sqrt(float(np.mean(sreq_ttft))))
                    # makespan is ONE number per bench: there is no request-
                    # sampling component to bootstrap, hence sigma_req = 0.
                    make_ci = rung_ci(d_make, 0.0)
                    # Axis R carried ALONGSIDE, per rev8 sec 1 ("축 R 값은 전 셀에
                    # 병기해 두 축이 조용히 갈라지는 일이 불가능하게 한다").  POINT
                    # ESTIMATES ONLY: axis R is NOT the registered axis, so it gets
                    # no CI, no sign_agree and no permission of any kind.
                    axisR = {}
                    for e in EST:
                        per = [d_itl_R[b][s][e] for b in sorted(d_itl_R)
                               for s in sorted(d_itl_R[b])]
                        bm = [float(np.mean([d_itl_R[b][s][e] for s in sorted(d_itl_R[b])]))
                              for b in sorted(d_itl_R)]
                        axisR[e] = {"point": float(np.mean(bm)),
                                    "sd_of_boot_means": (float(np.std(bm, ddof=1))
                                                         if len(bm) >= 2 else None),
                                    "n_cells": len(per)}
                    axisR["note"] = ("DIAGNOSTIC companion (rev8 sec 1 A1).  PRIMARY_AXIS "
                                     "is pooled token (axis P).  No CI / no sign_agree / "
                                     "no sign statement is derived from axis R.")
                    matrix.append({
                        "rung_lambda": lam, "shape": sh, "pair_D_minus_Dprime": [D, Dp],
                        "delta_itl_by_estimator": ci,
                        "sign_agree": sign_agree(ci),
                        "delta_itl_axis_R_DIAGNOSTIC": axisR,
                        "delta_ttft95_s": ttft_ci,
                        "delta_makespan_s": make_ci,
                        "delta_makespan_note": "NEVER call this a throughput difference "
                                               "(rev5 sec 1.1)",
                        "cells_used": {b: sorted(d_itl[b]) for b in sorted(d_itl)},
                    })
    return matrix


# =============================================================================
# Output 7 -- provenance (rev5 sec 6 item 7, implemented per rev9 sec 3 / P4)
# =============================================================================
SERVER_INFO_KEYS = ("disable_cuda_graph", "enable_pdmux", "attention_backend",
                    "max_running_requests", "cuda_graph_max_bs",
                    "pdmux_config_path", "sm_group_num", "disable_radix_cache",
                    "mem_fraction_static", "context_length", "model_path",
                    "disable_overlap_schedule", "page_size", "random_seed",
                    "chunked_prefill_size", "max_prefill_tokens", "tp_size")


def provenance(run_dirs, cells):
    """Registered output 7.  rev9 sec 3 (P4): report the 4 rounds' manifest
    IDENTITY, the full `server_info`, numpy/torch versions, node/time/clock/
    temperature -- AS FACTS.

    ! There is deliberately NO rule that drops a cell on a mismatch.  Creating
    one would put a new free surface in the eligibility layer (rev9 sec 3,
    gate #119 family).  A mismatch is reported; the results document
    interprets it.
    """
    out = {"registered_output": 7,
           "rule": "facts only -- NO cell is dropped for any mismatch (rev9 sec 3 P4)"}
    # --- manifest identity across rounds ---
    man = {}
    for rd in run_dirs:
        p = Path(rd) / "runtime_source_manifest.sha256"
        if p.exists():
            body = p.read_text()
            import hashlib
            man[Path(rd).name] = {
                "manifest_of_manifest_sha256": hashlib.sha256(body.encode()).hexdigest(),
                "n_files": len([x for x in body.splitlines() if x.strip()]),
                "entries": dict(
                    (ln.split()[1], ln.split()[0]) for ln in body.splitlines() if ln.strip()),
            }
        else:
            man[Path(rd).name] = {"error": "missing runtime_source_manifest.sha256"}
    digests = {v.get("manifest_of_manifest_sha256") for v in man.values()}
    out["runtime_source_manifest"] = {
        "per_round": {k: {kk: vv for kk, vv in v.items() if kk != "entries"}
                      for k, v in man.items()},
        "identical_across_rounds": bool(len(digests) == 1 and None not in digests),
        "n_distinct_digests": len(digests),
        "entries_round1": next(iter(man.values())).get("entries"),
    }
    # --- boot-level metadata ---
    boots = []
    for rd in run_dirs:
        for p in sorted(glob.glob(str(Path(rd) / "meta_r*_d*.txt"))):
            t = parse_tokens(Path(p).name)
            lines = [x.rstrip("\n") for x in Path(p).read_text().splitlines()]
            e = {"meta_file": Path(p).name, "run_dir": Path(rd).name,
                 "round": t["round"], "arm": t["arm"], "raw": lines}
            for ln in lines:
                if ln.startswith("boot_start="):
                    for fld in ln.split():
                        if "=" in fld:
                            k, v = fld.split("=", 1)
                            e[k] = v
                elif ln.startswith("numpy "):
                    e["numpy"] = ln.split()[1]
                elif ln.startswith("torch "):
                    e["torch"] = ln.split()[1]
                elif "MHz" in ln and "," in ln:
                    a, b = ln.split(",", 1)
                    e["sm_clock_MHz_at_boot"] = a.replace("MHz", "").strip()
                    e["gpu_temp_C_at_boot"] = b.strip()
                elif ln.startswith("sm_counts"):
                    e["sm_counts"] = ln.split(":", 1)[1].strip()
            boots.append(e)
    out["boots"] = boots
    out["nodes_by_round"] = {}
    for e in boots:
        out["nodes_by_round"].setdefault(str(e.get("round")), set()).add(e.get("node"))
    out["nodes_by_round"] = {k: sorted(v) for k, v in out["nodes_by_round"].items()}
    out["distinct_nodes"] = sorted({e.get("node") for e in boots if e.get("node")})
    out["distinct_numpy"] = sorted({e.get("numpy") for e in boots if e.get("numpy")})
    out["distinct_torch"] = sorted({e.get("torch") for e in boots if e.get("torch")})
    out["distinct_sm_counts"] = sorted({e.get("sm_counts") for e in boots
                                        if e.get("sm_counts")})
    out["gpu_info_by_round"] = {
        Path(rd).name: (Path(rd) / "gpu_info.txt").read_text().strip().splitlines()
        for rd in run_dirs if (Path(rd) / "gpu_info.txt").exists()}
    # --- server_info, read from the bench files themselves ---
    si_seen, si_full = {}, None
    for rd in run_dirs:
        for p in sorted(glob.glob(str(Path(rd) / "bench_*.jsonl"))):
            try:
                d = json.loads(Path(p).read_text())
            except Exception:
                continue
            si = d.get("server_info") or {}
            sa = si.get("server_args") if isinstance(si.get("server_args"), dict) else si
            if si_full is None:
                si_full = sa
            sig = tuple((k, str(sa.get(k))) for k in SERVER_INFO_KEYS)
            si_seen.setdefault(sig, []).append(Path(p).name)
            rr = d.get("random_range_ratio")
            out.setdefault("_rr", set()).add(rr)
    out["random_range_ratio_distinct"] = sorted(out.pop("_rr", {None}), key=str)
    out["server_info_registered_fields"] = [
        {"fields": dict(sig), "n_benches": len(v), "example_bench": v[0]}
        for sig, v in si_seen.items()]
    out["server_info_identical_across_all_benches"] = bool(len(si_seen) == 1)
    out["server_info_full_verbatim_round1_first_bench"] = si_full
    # --- max_running_requests vs realised L_total ---
    cap = None
    if si_full is not None:
        try:
            cap = float(si_full.get("max_running_requests"))
        except (TypeError, ValueError):
            cap = None
    lt = [(c["L_total_reported"], c["bench_file"]) for c in cells
          if c["L_total_reported"] == c["L_total_reported"]]
    mx = max(lt) if lt else (None, None)
    out["concurrency_cap_check"] = {
        "max_running_requests": cap, "max_L_total_observed": mx[0],
        "at_bench": mx[1],
        "binding": (None if (cap is None or mx[0] is None) else bool(mx[0] >= 0.95 * cap)),
        "note": "if binding, X_max(.,B) is a function of the cap and that must be "
                "stated in place (rev5 sec 6 item 7)"}
    return out


def collect_saturation(run_dirs):
    """Output 5: X_max := mu_p from saturation benches, per-cell SOURCE tag.
    Measured and cited values are NEVER averaged together (rev5 sec 5)."""
    if isinstance(run_dirs, (str, Path)):
        run_dirs = [run_dirs]
    out = {"measured_this_campaign": [], "cited": []}
    for p in sorted(q for rd in run_dirs
                    for q in glob.glob(str(Path(rd) / "sat_*.jsonl"))):
        d, _, _ = load_bench(p)
        t = parse_tokens(Path(p).name)
        out["measured_this_campaign"].append({
            "file": Path(p).name, "arm": t["arm"], "round": t["round"],
            "shape_out_len": int(d["random_output_len"]),
            "offered_rate": float(d["request_rate"]),
            "mu_p_observed": float(d["request_throughput"]),
            "completed": int(d["completed"]), "duration_s": float(d["duration"]),
            "source": "THIS CAMPAIGN (saturation bench)"})
    for arm, v in MU_P_REF.items():
        out["cited"].append({"arm": arm, "mu_p_ref": v, "source": MU_P_REF_SRC[arm]})
    grouped = {}
    for r in out["measured_this_campaign"]:
        grouped.setdefault((r["arm"], r["shape_out_len"]), []).append(r["mu_p_observed"])
    out["mu_p_by_arm_shape"] = [
        {"arm": k[0], "shape_out_len": k[1], "n_boot": len(v),
         "mean": float(np.mean(v)),
         "sd": (float(np.std(v, ddof=1)) if len(v) >= 2 else None),
         "values": v, "source": "THIS CAMPAIGN"} for k, v in sorted(grouped.items())]
    pairs = []
    for (a1, s1), v1 in grouped.items():
        for (a2, s2), v2 in grouped.items():
            if a1 == a2 or s1 != s2:
                continue
            pairs.append({"shape_out_len": s1, "pair_Dprime_minus_D": [a2, a1],
                          "r_cap_req_s": float(np.mean(v2)) - float(np.mean(v1)),
                          "source_both": "THIS CAMPAIGN"})
    out["r_cap"] = pairs
    out["rule"] = ("capacity difference is spoken ONLY through r_cap (rev5 sec 1.1); "
                   "measured and cited mu_p are tagged per cell and never averaged")
    return out


def run(run_dirs, boot_n=BOOT_N):
    """`qa_regret.sbatch` writes ONE DIRECTORY PER ROUND (`qa_r<R>_<jobid>/`),
    so all four round directories are passed together and merged on the
    (round, arm, rung, shape, seed) key carried in each filename."""
    if isinstance(run_dirs, (str, Path)):
        run_dirs = [run_dirs]
    run_dirs = [Path(x) for x in run_dirs]
    cells = []
    seen = {}
    for rd in run_dirs:
        for p in sorted(glob.glob(str(rd / "bench_*.jsonl"))):
            c = analyse_cell(p, boot_n=boot_n)
            c["run_dir"] = str(rd)
            k = (c["rung_lambda"], c["shape"], c["round"], c["seed"], c["arm"])
            if k in seen:
                c["DUPLICATE_KEY_WARNING"] = "same (rung,shape,round,seed,arm) as %s" % seen[k]
            seen[k] = c["bench_file"]
            cells.append(c)
    res = {
        "analysis": "QA_REGRET outputs 1-10",
        "prereg": ["PREREG_QA_REGRET_REV7_2026-09-10.md (submitted version)",
                   "PREREG_QA_REGRET_REV8_2026-09-10.md (axis / 1-seed CI / sigma_req)",
                   "PREREG_QA_REGRET_REV9_2026-09-10.md (analysis constants a-d, "
                   "symmetric 1-seed diagnostic P3, output 7 P4)"],
        "verdict": ["audit_qa_rev6_2026-09-10/VERDICT.md (GO-with-caveats)",
                    "audit_qa_rev8_2026-09-10/VERDICT.md (GO-with-caveats)"],
        "inherited_citation_bans": 68,
        "primary_estimator_axis": PRIMARY_AXIS,
        "gpu_hours_spent_by_this_file": 0,
        "produces": "numbers only -- no performance verdict, no arm ranking, "
                    "no min/argmin/envelope/ranking, no policy statement",
        "mandatory_companions": MANDATORY_COMPANIONS,
        "n_cells": len(cells),
    }
    # output 3: variance components, all four estimators, per (rung, shape, arm)
    var = []
    key = {}
    for c in cells:
        key.setdefault((c["rung_lambda"], c["shape"], c["arm"]), []).append(c)
    for k, group in sorted(key.items(), key=lambda kv: str(kv[0])):
        boots = {}
        for c in group:
            boots.setdefault(c["round"], {})[c["seed"]] = c
        entry = {"rung_lambda": k[0], "shape": k[1], "arm": k[2],
                 "n_boot": len(boots)}
        for e in EST:
            entry[e] = variance_components(
                [[boots[b][s]["estimators"]["pooled_token_axis"][e]
                  for s in sorted(boots[b])] for b in sorted(boots)])
        # Registered output 3 covers the FOUR ESTIMATORS.  The two below are
        # extra instrumentation, keyed so they cannot be mistaken for a
        # registered output ("등록된 것만, 사후 추가 금지").
        entry["DIAGNOSTIC_beyond_registered_list"] = {
            "L_decode_exact": variance_components(
                [[boots[b][s]["L_decode_exact"] for s in sorted(boots[b])]
                 for b in sorted(boots)]),
            "ttft_p95_s": variance_components(
                [[boots[b][s]["ttft_p95_s"] for s in sorted(boots[b])]
                 for b in sorted(boots)])}
        var.append(entry)
    res["variance_components"] = var
    res["delta_matrix"] = build_delta_matrix(cells, boot_n=boot_n)
    res["saturation_and_r_cap"] = collect_saturation(run_dirs)
    res["duty_cycle_route_B_by_boot"] = duty_cycle_route_B(run_dirs)
    xs = [c["crosscheck_rel_err"] for c in cells]
    res["crosscheck"] = {
        "L_pre_plus_L_decode_vs_reported_concurrency": {
            "max_rel_err": (max(xs) if xs else None), "threshold": CROSSCHECK_TOL,
            "n_over_threshold": sum(1 for x in xs if x >= CROSSCHECK_TOL)},
        "throughput_identity_note": "completed/duration vs request_throughput is an "
                                    "ALGEBRAIC IDENTITY -- reported, never a check"}
    res["low_load_exposure_table"] = [
        {"bench_file": c["bench_file"], "arm": c["arm"], "rung_lambda": c["rung_lambda"],
         "shape": c["shape"], "X_realised": c["X_completed_over_duration"],
         "repo_min_X_same_arm": c["low_load_extrapolation"]["repo_min_X_same_arm"],
         "below_repo_min": c["low_load_extrapolation"]["below_repo_min"]}
        for c in cells]
    res["span_observed_vs_forecast"] = [
        {"bench_file": c["bench_file"], "seed": c["seed"],
         "span_factor_observed": c["span_factor_observed"],
         "span_factor_predicted": c["span_factor_predicted"],
         "rel_err_pct_vs_pred": c["span_factor_rel_err_pct_vs_pred"],
         "X_realised": c["X_completed_over_duration"]} for c in cells]
    res["registration_conformance"] = registration_conformance(cells)
    res["provenance_output_7"] = provenance(run_dirs, cells)
    for c in cells:
        c.pop("_itls", None)
        c.pop("_ttfts", None)
    res["cells"] = cells
    return res


# =============================================================================
# Pre-submission self-check against numbers this repo already published
# =============================================================================
P7_DIR = HERE / "p7_905994"
C_DIR = HERE / "c_905835"
S1_DIR = HERE / "s1_905958"
P2_DIR = HERE / "p2_905712"


def _p7_cells():
    out = {}
    for p in sorted(glob.glob(str(P7_DIR / "bench_r*_d*_s*.jsonl"))):
        t = parse_tokens(Path(p).name)
        _, _, itls = load_bench(p)
        out[(t["round"], t["arm"], t["seed"])] = itls
    return out


def selfcheck(boot_n=BOOT_N):
    rep = {"gpu_hours": 0, "checks": []}

    def add(name, got, want, ok, note=""):
        rep["checks"].append({"check": name, "got": got, "expected": want,
                              "reproduced": ok, "note": note})

    # --- 0. quantile helpers against the two values the repo already uses ---
    add("F_{0.05}(3,8)", round(f_ppf(0.05, 3, 8), 4), 0.1131,
        abs(f_ppf(0.05, 3, 8) - 0.1131) < 5e-5, "RESULT_P7 sec 2")
    add("F_{0.95}(3,4)", round(f_ppf(0.95, 3, 4), 3), 6.591,
        abs(f_ppf(0.95, 3, 4) - 6.591) < 5e-3, "rev6 sec 6")
    add("t_{0.975}(3)", round(t_ppf(0.975, 3), 6), 3.182446,
        abs(t_ppf(0.975, 3) - 3.182446305284263) < 1e-6, "verdict sec 2-(3)")

    # --- 1. P7 paired deltas (axis P) ---
    cells = _p7_cells()
    rounds = sorted({k[0] for k in cells})
    seeds_of = {r: sorted({k[2] for k in cells if k[0] == r and k[1] == 44}) for r in rounds}
    est = {k: pooled_token_estimators(v) for k, v in cells.items()}
    est_r = {k: itl95_estimators(v) for k, v in cells.items()}
    tgt = {(44, 54): {"pooled_p95": 1.5201, "median": 1.2854, "mean": 0.4935, "trim10": 0.4409},
           (16, 54): {"pooled_p95": 25.5196, "median": -0.2165, "mean": 7.9639, "trim10": 6.6642},
           (16, 44): {"pooled_p95": 23.9996, "median": -1.5019, "mean": 7.4704, "trim10": 6.2233}}
    t3 = t_ppf(0.975, 3)
    deltas = {}
    for (a1, a2), want in tgt.items():
        per = {r: {s: {e: est[(r, a1, s)][e] - est[(r, a2, s)][e] for e in EST}
                   for s in seeds_of[r]} for r in rounds}
        deltas[(a1, a2)] = per
        for e in EST:
            bm = np.array([np.mean([per[r][s][e] for s in seeds_of[r]]) for r in rounds])
            m, sd = float(bm.mean()), float(bm.std(ddof=1))
            hw = t3 * sd / math.sqrt(len(bm))
            ok = abs(m - want[e]) < 5e-4
            extra = ""
            if (a1, a2) == (44, 54) and e == "pooled_p95":
                extra = ("SD(boot means) %.4f (want 0.1217); boot-level t3 CI "
                         "[%.4f, %.4f] (want [1.3265, 1.7137])" % (sd, m - hw, m + hw))
                ok = ok and abs(sd - 0.1217) < 5e-5
            add("Delta(d%d,d%d) %s [axis P]" % (a1, a2, e), round(m, 4), want[e], ok, extra)
        # axis R for contrast (this is the axis that does NOT reproduce)
        axr = {e: float(np.mean([est_r[(r, a1, s)][e] - est_r[(r, a2, s)][e]
                                 for r in rounds for s in seeds_of[r]])) for e in EST}
        add("Delta(d%d,d%d) axis R (per-request itl95) -- CONTRAST" % (a1, a2),
            {e: round(axr[e], 4) for e in EST}, "n/a", None,
            "reported so the axis divergence is visible; pooled_p95 identical by "
            "construction, the other three are NOT")

    # --- 2. sign_agree under the REGISTERED rung-level cross-design CI ---
    for (a1, a2) in [(44, 54), (16, 54)]:
        ci = {}
        for e in EST:
            sreq = []
            for r in rounds:
                for s in seeds_of[r]:
                    pb = paired_request_bootstrap(cells[(r, a1, s)], cells[(r, a2, s)],
                                                  n=boot_n)
                    sreq.append(pb[e] ** 2)
            per = {r: {s: deltas[(a1, a2)][r][s][e] for s in seeds_of[r]} for r in rounds}
            ci[e] = rung_ci(per, math.sqrt(float(np.mean(sreq))))
        sa = sign_agree(ci)
        want = 4 if (a1, a2) == (44, 54) else 3
        add("sign_agree Delta(d%d,d%d)" % (a1, a2), sa["sign_agree"], want,
            sa["sign_agree"] == want,
            "REGISTERED rev7 sec 3 CI, n_boot=4 x n_seed=3 (P7 design, NOT the "
            "registered n_seed=2); half-widths " +
            json.dumps({e: round(ci[e]["half_width"], 4) for e in EST}) +
            "; se_without_req " +
            json.dumps({e: round(ci[e]["se_without_req_diagnostic"], 4) for e in EST}) +
            "; N=90 and P7 LOAD, not a registered N")
        # the diagnostic (boot-level) CI the verdict sec 2-(2) table used
        ci2 = {e: dict(ci[e]) for e in EST}
        for e in EST:
            lo, hi = ci[e]["ci95_boot_level_diagnostic"]
            ci2[e]["excludes_zero"] = bool(lo > 0 or hi < 0)
        add("sign_agree Delta(d%d,d%d) [boot-level diagnostic CI]" % (a1, a2),
            sign_agree(ci2)["sign_agree"], want, sign_agree(ci2)["sign_agree"] == want,
            "the CI the rev6 verdict sec 2-(2) table used")

    # --- 3. rev6 sec 6 cross-design SE.  ** This is a FORECAST FOR THE
    # REGISTERED DESIGN (4 boots x 2 seeds, n_cells = 8), not a P7 statistic **:
    # P7's variance components are plugged into the registered denominators.
    # `rung_ci` divides by the ACTUAL n_cells (12 for P7), which is why the two
    # differ -- the difference is the design, not the estimator.
    for (a1, a2), w, wh in [((44, 54), 0.0718, 0.228), ((16, 54), 0.4922, 1.566)]:
        per = {r: {s: deltas[(a1, a2)][r][s]["pooled_p95"] for s in seeds_of[r]}
               for r in rounds}
        got_actual = rung_ci(per, 0.0)
        bm = np.array([np.mean([per[r][s] for s in seeds_of[r]]) for r in rounds])
        s2_seed = float(np.mean([np.var([per[r][s] for s in seeds_of[r]], ddof=1)
                                 for r in rounds]))
        s2_boot = max(float(bm.var(ddof=1)) - s2_seed / 3.0, 0.0)
        proj = math.sqrt(s2_boot / 4.0 + s2_seed / 8.0)     # registered n_cells = 8
        add("rev6 sec 6 SE projected onto the REGISTERED design (4 boots x 2 seeds) "
            "Delta(d%d,d%d) pooled_p95" % (a1, a2), round(proj, 4), w,
            abs(proj - w) < 5e-5,
            "half-width t3*SE = %.3f (want %.3f); the SE of the ACTUAL P7 design "
            "(n_cells=12) is %.4f -- rung_ci uses the actual design, as it must"
            % (t3 * proj, wh, got_actual["se_without_req_diagnostic"]))

    # --- 4. L cross-check and span_hat reconstruction ---
    for tag, d in (("probe C", C_DIR), ("P7", P7_DIR)):
        xs, errs = [], []
        for p in sorted(glob.glob(str(d / "bench_*.jsonl"))):
            dd, ttfts, itls = load_bench(p)
            dur = dd["duration"]
            xs.append(abs(float(ttfts.sum()) / dur
                          + sum(float(x.sum()) for x in itls) / dur
                          - dd["concurrency"]) / dd["concurrency"])
            n = len(dd["ttfts"])
            sh = dur - (dd["ttfts"][-1] + float(np.sum(dd["itls"][-1])))
            obs = sh / ((n - 1) / dd["request_rate"])
            sd = parse_tokens(Path(p).name)["seed"]
            pred = span_factor(sd if sd is not None else 41, n)
            errs.append(abs(pred - obs) / pred * 100.0)
        if tag == "probe C":
            add("L_pre+L_decode vs concurrency, max rel err (probe C, 12 cells)",
                "%.4e" % max(xs), "1.845e-06", abs(max(xs) - 1.845e-06) < 1e-9,
                "the parent brief attributed 1.845e-06 to P7; it is probe C")
        else:
            add("L_pre+L_decode vs concurrency, max rel err (P7, 36 benches)",
                "%.4e" % max(xs), "2.66e-06", abs(max(xs) - 2.66e-06) < 1e-8,
                "RESULT_P7 sec 4")
            add("span_hat reconstruction, mean |err| (P7, 36 benches)",
                "%.4f%%" % float(np.mean(errs)), "0.207%",
                abs(float(np.mean(errs)) - 0.207) < 5e-4,
                "denominator = PREDICTED span factor (|pred-obs|/pred); the /obs "
                "denominator gives 0.2052%")

    # --- 4b. the registered mu_p_ref table is the P7 saturation mean (n_boot=4) ---
    sat = {}
    for p in sorted(glob.glob(str(P7_DIR / "sat_r*_d*.jsonl"))):
        dd, _, _ = load_bench(p)
        sat.setdefault(parse_tokens(Path(p).name)["arm"], []).append(
            float(dd["request_throughput"]))
    for arm in (16, 44, 54):
        got = float(np.mean(sat[arm]))
        add("mu_p_ref(d%d) from P7 saturation benches, n_boot=4" % arm, round(got, 4),
            MU_P_REF[arm], abs(round(got, 4) - MU_P_REF[arm]) < 5e-5,
            "rev5 sec 5; sd %.5f over %d boots" % (float(np.std(sat[arm], ddof=1)),
                                                   len(sat[arm])))

    # --- 5. mode extraction / u over the repo's 56 arm-tagged-or-not cells ---
    pop = []
    for pat, tag in [(P7_DIR / "bench_r*_d*_s*.jsonl", "P7"), (C_DIR / "bench_*.jsonl", "C"),
                     (S1_DIR / "bench_d*.jsonl", "S1"), (P2_DIR / "bench_c*.jsonl", "P2")]:
        for p in sorted(glob.glob(str(pat))):
            _, _, itls = load_bench(p)
            rm = np.array([np.median(x) * 1000.0 for x in itls if x.size])
            tk = np.concatenate([x for x in itls if x.size]) * 1000.0
            md = mode_extract(rm)
            pop.append((tag, Path(p).name, md,
                        (u_stat(tk, md["c_valley"]) if md.get("u_defined") else None),
                        (float((rm < md["c_valley"]).mean()) if md.get("u_defined") else None)))
    ndef = sum(1 for x in pop if x[2].get("u_defined"))
    add("cells with u defined (repo population 56)", "%d/56" % ndef, "47/56", ndef == 47,
        "the 47-cell population the rev6 verdict used")
    d92_bad = [x[1] for x in pop if "d92" in x[1] and not x[2].get("u_defined")]
    add("d92 cells with u = UNDEFINED (far bin empty)", len(d92_bad), 5, len(d92_bad) == 5,
        "rev7 sec 1 registered third branch; files " + ", ".join(d92_bad))
    gaps = [x[2]["gap"] for x in pop if x[2].get("u_defined")]
    add("min gap over the 47 cells (bin width 2 ms, REGISTERED)", min(gaps), 4.0,
        False, "verdict sec 0 says 'exactly 4.0'; at the REGISTERED 2 ms width the "
               "minimum is 6.0 (d54).  4.0 is reachable only at the 4 ms DIAGNOSTIC "
               "width (verdict sec 2-(6) itself reports d54 gap 5/6/4 for widths "
               "1/2/4).  The operative claim -- 'gap < 4 never fires' -- reproduces: "
               "0 cells below 4")
    add("cells with gap < 4 ms", sum(1 for g in gaps if g < 4), 0,
        sum(1 for g in gaps if g < 4) == 0, "the 'empty branch' finding (verdict sec 0)")
    ties = [(x[1], x[2].get("c_valley"), x[2].get("c_valley_tie_alternatives"), x[3])
            for x in pop if x[2].get("tie_unregistered")]
    add("unregistered bin-count tie cells", [(t[0], t[1], t[2], round(t[3], 4)) for t in ties],
        "2 cells: r2_d16_s44 c_valley 27<->29 u 0.6133; r3_d16_s49 27<->28 u 0.7281",
        len(ties) == 2 and all(abs(t[3] - v) < 5e-5 for t, v in zip(ties, (0.6133, 0.7281))),
        "verdict sec 1 row A: u identical to 4 decimals under either tie-break")
    flips = sum(1 for x in pop if x[2].get("u_defined")
                and {k: v for k, v in cliff_labels(x[3]).items() if k != "note"}
                != {k: v for k, v in cliff_labels(x[4]).items() if k != "note"})
    maxdu = max(abs(x[3] - x[4]) for x in pop if x[2].get("u_defined"))
    add("label flips if u used the per-request median axis instead", "%d/47" % flips,
        "6/47", flips == 6, "verdict sec 1-E; max |du| %.4f (want 0.0677)" % maxdu)
    u16 = [x[3] for x in pop if x[0] == "P7" and "_d16_" in x[1]]
    add("P7 d16 u range", [round(min(u16), 4), round(max(u16), 4)], [0.483, 0.728],
        abs(min(u16) - 0.483) < 1e-3 and abs(max(u16) - 0.728) < 1e-3, "verdict sec 4 item 4")
    fires = sum(1 for u in u16 if abs(u - 0.50) < 0.10)
    add("P7 d16 cells firing the `median` cliff |u-0.50|<0.10", "%d/12" % fires, "8/12",
        False, "NOT reproduced: 6/12 on the registered token axis, 7/12 on the "
               "per-request median axis.  The same computation reproduces the u range "
               "and the 6/47 flip count exactly, so the discrepancy is in the "
               "verdict's count, not in the extraction.  No registered output depends "
               "on it (verdict sec 4 is prose)")
    return rep


def _print_selfcheck(rep):
    print("=" * 100)
    print("qa_analyze.py pre-submission self-check   GPU 0 | performance verdicts 0 | "
          "arm rankings 0")
    print("=" * 100)
    for c in rep["checks"]:
        flag = {True: "OK  ", False: "MISS", None: "----"}[c["reproduced"]]
        print("[%s] %-58s got %s | want %s" % (flag, c["check"], c["got"], c["expected"]))
        if c["note"]:
            for i in range(0, len(c["note"]), 92):
                print("       %s" % c["note"][i:i + 92])
    n_ok = sum(1 for c in rep["checks"] if c["reproduced"] is True)
    n_miss = sum(1 for c in rep["checks"] if c["reproduced"] is False)
    print("-" * 100)
    print("reproduced %d | NOT reproduced %d | informational %d"
          % (n_ok, n_miss, sum(1 for c in rep["checks"] if c["reproduced"] is None)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", nargs="*", default=None,
                    help="one or more round directories (qa_r<R>_<jobid>/)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--boot-n", type=int, default=BOOT_N,
                    help="bootstrap samples (repo convention 10000, seed 1)")
    args = ap.parse_args()
    if args.selfcheck or not args.run_dir:
        rep = selfcheck(boot_n=args.boot_n)
        _print_selfcheck(rep)
        if args.out:
            Path(args.out).write_text(json.dumps(rep, indent=2, ensure_ascii=False,
                                                 default=str))
        return
    res = run(args.run_dir, boot_n=args.boot_n)
    text = json.dumps(res, indent=2, sort_keys=True, ensure_ascii=False, default=str)
    if args.out:
        Path(args.out).write_text(text)
        print("wrote %s (%d cells)" % (args.out, res["n_cells"]))
    else:
        print(text)


if __name__ == "__main__":
    main()
