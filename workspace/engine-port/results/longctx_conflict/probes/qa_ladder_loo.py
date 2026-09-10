#!/usr/bin/env python3
"""Q-A prereg §5-1 pre-submission reversal test: leave-one-rung-out (LOO)
interpolation error of the load curve  itl95(X).

GPU 0.  Re-analysis of existing probe C (job 905835) raw artifacts only.
NO performance verdict, NO arm ranking, NO policy comparison is produced here.

WHAT THIS MEASURES
------------------
Only the *grid sensitivity of the curve shape*, per arm.  Probe C ran with
n_boot = 1 and a single fixed --seed 41 for every cell (REPORT §5-B), so the
between-boot and between-seed variance components are ABSENT from every number
below.  The rungs are also arm-relative (ρ × μ_p(D)), so the X ranges of the
three arms do not overlap -- cross-arm comparison is undefined here and is not
attempted.

DEFINITIONS (identical to the track's canonical instrument)
-----------------------------------------------------------
  itl95_i     = np.percentile(itls_i, 95) * 1000                    [ms, per request]
  itl95_cell  = 4 estimators over {itl95_i}: median / mean / 10% trimmed mean
                / pooled-p95 (percentile of the concatenation of all ITLs)
                -- imported verbatim from p7_analyze.itl95_estimators so that
                this file cannot drift from RESULT_P7.
  X           = achieved request throughput [req/s].  Recomputed here as
                completed / duration and compared against (a) bench jsonl
                "request_throughput" and (b) cell_*.json "achieved_rate".
                !! WARNING: this comparison is an ALGEBRAIC IDENTITY, not an
                independent cross-check -- bench_serving.py:1093 defines
                request_throughput := completed / dur_s and
                c_capacity_analyze.py:93 copies that field into achieved_rate.
                Agreement to 0.0e+00 therefore validates nothing (gate #9).
  span_hat    = duration - (ttfts[-1] + sum(itls[-1]))   [s]   (arrival span,
                audit_sweep_rev2 §R5; reported as a covariate only)

LOO TEST (prereg §5-1)
----------------------
For each arm, hold out the MIDDLE rung r1 and predict itl95(X_r1) from the two
outer rungs (r0, r2), in two interpolation variants that Q-A §2.4 registers:
  (i)  linear in X          pred = y0 + (y2-y0) * (X1-X0)/(X2-X0)
  (ii) linear in log-log    log pred = log y0 + (log y2 - log y0)
                                        * (log X1 - log X0)/(log X2 - log X0)
Relative prediction error = |pred - obs| / obs * 100  [%].
Registered resolution to compare against: 7.2 % (Q-A §4: half-width of the 95 %
CI of a two-arm difference at n_boot = 4, t(3) = 3.182, σ upper bound 3.2 %).

AUXILIARY (NOT the registered test -- EXTRAPOLATION)
----------------------------------------------------
r2 of every arm sits at ach/off ≈ 0.77-0.79, i.e. ABOVE the knee, whereas the
Q-A ladder Λ is entirely ρ ≤ 0.95 (below the knee).  As a regime-matched
supplement we also fit the two below-knee rungs (r0, r1) and EXTRAPOLATE to
X_r2.  This is extrapolation, which Q-A §2.4 explicitly forbids, so it is a
diagnostic of curvature only and is labelled as such everywhere.

SUPPLEMENTARY DIAGNOSTICS (is the registered PASS informative at all?)
----------------------------------------------------------------------
A small LOO error is only evidence about grid density if the test could have
produced a large one.  Four degeneracy checks are therefore reported:

  D1 lever arm      (X1-X0)/(X2-X0).  If ~1 the "interpolation" is really an
                    evaluation at the outer anchor.
  D2 dynamic range  (max-min)/min of itl95_cell over the arm's three rungs.
                    If it is below the 7.2% resolution there is nothing to
                    resolve and every predictor passes.
  D3 null models    err of pred := y_r2 (nearest anchor) / (y_r0+y_r2)/2 /
                    y_r0 (flat).  If the interpolant is not better than the
                    nearest-anchor constant, the LOO measures nothing.
  D4 noise floor    within-cell 95% CI half-width of itl95_cell from
                    resampling REQUESTS with replacement (10000 samples,
                    seed 1 -- the repo's canonical bootstrap convention).
                    If the LOO error is inside this band it is indistinguish-
                    able from request-sampling noise.  NB this is only the
                    request-sampling component; boot and seed components are
                    structurally absent (n_boot=1, one fixed seed).

The same LOO is also repeated on the OFFERED-rate axis λ (the axis Λ is
actually registered in, and the axis whose 3-point geometry -- ratio ~1.47 --
resembles Λ's ratio 1.67).  Not a registered estimator; reported for diagnosis.

Usage:
  python3 qa_ladder_loo.py [--dir <probe C dir>] [--out result.json]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from p7_analyze import itl95_estimators, load_bench, model_a_span_factor  # noqa: E402

PROBE_C = HERE / "c_905835"
ARMS = ("d16", "d44", "d92")
RUNGS = ("r0", "r1", "r2")
SHAPE = "o96"
EST = ("median", "mean", "trim10", "pooled_p95")
RESOLUTION_PCT = 7.2          # Q-A §4, registered
SEED_FIXED = 41               # probe C: every cell (REPORT §5-B)
BOOT_N = 10000                # repo convention (pdmux_eval.analyze)
BOOT_SEED = 1                 # repo convention (pdmux_eval.analyze)
# Step 1 (job 905958) mu_p under pdmux_homog6.yml -- used ONLY to print where
# the registered ladder Lambda would fall in rho for each arm.  No verdict.
MU_P_S1 = {16: 0.914318261526629, 44: 0.6666328631373765,
           54: 0.5772205386595464, 92: 0.18854963732712318}   # S1_LABEL.json
LAMBDA_REGISTERED = (0.09, 0.15, 0.25, 0.42, 0.70)
T_WIN = 150.0                 # Q-A §2.1
# Uncontended decode ITL, ctx 8192 (P2 job 905712); the constant exact_L.py uses.
ITL_SOLO_MS = 13.03689880296588
UNTREATED_TOL = 1.10          # itl95_i <= 1.10 * ITL_SOLO_MS  =>  request never
                              # spent 5%+ of its decode under a fired split
# Q-A §2.1:  N(D,s,rho) := round(rate * T_win) + 1
N_REGISTERED = tuple(int(round(lam * T_WIN)) + 1 for lam in LAMBDA_REGISTERED)


def itl95_bootstrap_ci(itls, n=BOOT_N, seed=BOOT_SEED, chunk=500):
    """95% CI half-width of each itl95_cell estimator from resampling REQUESTS
    with replacement.  Only the request-sampling component -- boot/seed
    components are absent from probe C by construction."""
    per_req = np.array([np.percentile(x, 95) * 1000.0 for x in itls if x.size])
    m = per_req.size
    rs = np.random.RandomState(seed)
    idx = rs.randint(0, m, size=(n, m))
    med = np.median(per_req[idx], axis=1)
    mean = per_req[idx].mean(axis=1)
    s = np.sort(per_req[idx], axis=1)
    k = int(0.1 * m)
    trim = s[:, k:m - k].mean(axis=1) if m - 2 * k > 0 else s.mean(axis=1)
    lens = {x.size for x in itls}
    pooled = None
    if len(lens) == 1:                       # rectangular -> vectorisable
        M = np.stack([np.asarray(x, dtype=float) for x in itls])
        vals = []
        for a in range(0, n, chunk):
            sel = idx[a:a + chunk]
            vals.append(np.percentile(M[sel].reshape(sel.shape[0], -1),
                                      95, axis=1) * 1000.0)
        pooled = np.concatenate(vals)

    def hw(v):
        if v is None:
            return None
        lo, hi = np.percentile(v, 2.5), np.percentile(v, 97.5)
        return float((hi - lo) / 2.0)

    out = {"n_requests": int(m), "n_boot_samples": n, "seed": seed,
           "half_width_ms": {"median": hw(med), "mean": hw(mean),
                             "trim10": hw(trim), "pooled_p95": hw(pooled)}}
    base = {"median": float(np.median(per_req)), "mean": float(per_req.mean()),
            "trim10": float(np.sort(per_req)[k:m - k].mean() if m - 2 * k > 0
                            else per_req.mean()),
            "pooled_p95": float(np.percentile(np.concatenate(itls), 95) * 1000.0)}
    out["half_width_pct"] = {
        e: (100.0 * out["half_width_ms"][e] / base[e]
            if out["half_width_ms"][e] is not None else None) for e in EST}
    return out


def n_projection(itls, m_list, n=BOOT_N, seed=BOOT_SEED, n_pooled=2000, chunk=500):
    """m-out-of-n bootstrap: 95% CI half-width of itl95_cell if the cell had
    only m requests.  Used to project the noise floor onto Q-A's registered
    per-rung request counts N = round(lambda*T_win)+1.  First-order estimate:
    it inherits the load condition of THIS cell."""
    per_req = np.array([np.percentile(x, 95) * 1000.0 for x in itls if x.size])
    lens = {x.size for x in itls}
    M = (np.stack([np.asarray(x, dtype=float) for x in itls])
         if len(lens) == 1 else None)
    base_med = float(np.median(per_req))
    base_mean = float(per_req.mean())
    base_pool = float(np.percentile(np.concatenate(itls), 95) * 1000.0)
    out = {}
    for m in m_list:
        rs = np.random.RandomState(seed + m)
        idx = rs.randint(0, per_req.size, size=(n, m))
        smp = per_req[idx]
        k = int(0.1 * m)
        srt = np.sort(smp, axis=1)
        trim = srt[:, k:m - k].mean(axis=1) if m - 2 * k > 0 else srt.mean(axis=1)
        base_trim = float(np.sort(per_req)[int(0.1 * per_req.size):
                                           per_req.size - int(0.1 * per_req.size)].mean())

        def hw(v, base):
            lo, hi = np.percentile(v, 2.5), np.percentile(v, 97.5)
            return float(100.0 * (hi - lo) / 2.0 / base)

        row = {"median": hw(np.median(smp, axis=1), base_med),
               "mean": hw(smp.mean(axis=1), base_mean),
               "trim10": hw(trim, base_trim)}
        if M is not None:
            idx2 = rs.randint(0, per_req.size, size=(n_pooled, m))
            vals = []
            for a in range(0, n_pooled, chunk):
                sel = idx2[a:a + chunk]
                vals.append(np.percentile(M[sel].reshape(sel.shape[0], -1),
                                          95, axis=1) * 1000.0)
            row["pooled_p95"] = hw(np.concatenate(vals), base_pool)
            row["_n_boot_pooled"] = n_pooled
        out[m] = row
    return out


def mixture_diag(itls, cell):
    """D5: the per-request itl95 distribution is a MIXTURE of an untreated mode
    (split never fired for that request -> itl95 ~ ITL_SOLO_MS) and a treated
    mode.  Reported because the MEDIAN estimator is a STEP function of the
    untreated fraction at 0.5, which no interpolation density can reproduce."""
    v = np.sort(np.array([np.percentile(x, 95) * 1000.0 for x in itls if x.size]))
    thr = UNTREATED_TOL * ITL_SOLO_MS
    frac = float((v <= thr).mean())
    gaps = np.diff(v)
    j = int(np.argmax(gaps)) if gaps.size else 0
    tel = cell.get("telemetry", {})
    hist = tel.get("decode_hist", {})
    t108 = float(hist.get("108", 0.0))
    ttot = float(sum(hist.values())) or float("nan")
    return {
        "untreated_threshold_ms": thr,
        "untreated_fraction": frac,
        "median_sits_in": "untreated mode" if frac > 0.5 else "treated mode",
        "largest_gap_ms": float(gaps[j]) if gaps.size else float("nan"),
        "largest_gap_between_ms": [float(v[j]), float(v[j + 1])] if gaps.size else None,
        "untreated_mode_mean_ms": float(v[v <= thr].mean()) if (v <= thr).any() else None,
        "treated_mode_mean_ms": float(v[v > thr].mean()) if (v > thr).any() else None,
        "telemetry_f_time": tel.get("f_time"),
        "decode_time_frac_at_108sm_split_not_fired": t108 / ttot if ttot == ttot else None,
    }


def load_cell(d: Path, arm: str, rung: str):
    bench = d / f"bench_{arm}_{rung}_{SHAPE}.jsonl"
    cellf = d / f"cell_{arm}_{rung}_{SHAPE}.json"
    raw, ttfts, itls = load_bench(bench)
    cell = json.loads(cellf.read_text())
    n = len(raw["ttfts"])
    dur = float(raw["duration"])
    x_recomputed = float(raw["completed"]) / dur
    span_hat = dur - (raw["ttfts"][-1] + float(np.sum(raw["itls"][-1])))
    nominal_span = (n - 1) / raw["request_rate"]
    est = itl95_estimators(itls)
    return {
        "arm": arm,
        "rung": rung,
        "bench_file": bench.name,
        "cell_file": cellf.name,
        "offered_rate": float(raw["request_rate"]),
        "completed": int(raw["completed"]),
        "n_requests_ok": int(est["n_requests"]),
        "n_errors": int(n - len(ttfts)),
        "duration_s": dur,
        # --- X, three independent routes ---
        "X_recomputed": x_recomputed,
        "X_bench_field": float(raw["request_throughput"]),
        "X_cell_json": float(cell["achieved_rate"]),
        "X_relerr_vs_bench": abs(x_recomputed - raw["request_throughput"]) / raw["request_throughput"],
        "X_relerr_vs_cell": abs(x_recomputed - cell["achieved_rate"]) / cell["achieved_rate"],
        # --- regime markers ---
        "ach_over_off": x_recomputed / float(raw["request_rate"]),
        "above_knee": bool(x_recomputed / float(raw["request_rate"]) < 0.90),
        # --- covariates (reported, not used in the LOO) ---
        "span_hat_s": span_hat,
        "span_factor_observed": span_hat / nominal_span,
        "span_factor_model_a": model_a_span_factor(SEED_FIXED, n),
        "ttft_p50_s": float(np.percentile(ttfts, 50)),
        "ttft_p95_s": float(np.percentile(ttfts, 95)),
        "itl95": {k: est[k] for k in EST},
        "itl95_iqr_ms": est["iqr_ms"],
        "itl95_request_bootstrap": itl95_bootstrap_ci(itls),
        "itl95_N_projection": n_projection(itls, N_REGISTERED),
        "D5_mixture": mixture_diag(itls, cell),
    }


def interp_linear(x0, y0, x2, y2, xq):
    if x2 == x0:
        return math.nan
    return y0 + (y2 - y0) * (xq - x0) / (x2 - x0)


def interp_loglog(x0, y0, x2, y2, xq):
    if min(x0, x2, xq, y0, y2) <= 0 or x2 == x0:
        return math.nan
    b = (math.log(y2) - math.log(y0)) / (math.log(x2) - math.log(x0))
    return math.exp(math.log(y0) + b * (math.log(xq) - math.log(x0)))


def rel_err_pct(pred, obs):
    if not (obs and obs == obs) or pred != pred:
        return float("nan")
    return abs(pred - obs) / obs * 100.0


def run(dirpath: Path):
    cells = {(a, r): load_cell(dirpath, a, r) for a in ARMS for r in RUNGS}

    out = {
        "test": "QA_LADDER_LOO",
        "prereg": "PREREG_QA_REGRET_2026-09-10.md §5-1 / §6-0",
        "source": str(dirpath),
        "gpu_hours": 0.0,
        "scope": ("grid sensitivity of curve SHAPE per arm only; n_boot=1 and a "
                  "single fixed seed 41 for all cells, so between-boot and "
                  "between-seed components are absent; arm-relative rungs mean "
                  "the arms' X ranges do not overlap -- no cross-arm statement"),
        "resolution_pct_registered": RESOLUTION_PCT,
        "cells": [cells[(a, r)] for a in ARMS for r in RUNGS],
        "loo": [],
        "extrapolation_aux": [],
        "loo_lambda_axis_diagnostic": [],
        "conditioning": [],
    }

    xmax = max(c["X_relerr_vs_bench"] for c in out["cells"])
    xmax2 = max(c["X_relerr_vs_cell"] for c in out["cells"])
    out["X_crosscheck_max_relerr_vs_bench_field"] = xmax
    out["X_crosscheck_max_relerr_vs_cell_json"] = xmax2

    for arm in ARMS:
        c0, c1, c2 = (cells[(arm, r)] for r in RUNGS)
        X0, X1, X2 = c0["X_recomputed"], c1["X_recomputed"], c2["X_recomputed"]
        l0, l1, l2 = c0["offered_rate"], c1["offered_rate"], c2["offered_rate"]
        out["conditioning"].append({
            "arm": arm,
            "X": [X0, X1, X2],
            "offered": [l0, l1, l2],
            "dX_r0_r1": X1 - X0,
            "dX_r1_r2": X2 - X1,
            "X_span_r0_r2": X2 - X0,
            "lever_arm_frac_(X1-X0)/(X2-X0)": (X1 - X0) / (X2 - X0),
            "dlambda_r0_r1": l1 - l0,
            "dlambda_r1_r2": l2 - l1,
            "lever_arm_frac_lambda": (l1 - l0) / (l2 - l0),
            "ach_over_off": [c0["ach_over_off"], c1["ach_over_off"], c2["ach_over_off"]],
            "X_axis_collapse_ratio_dX_r1r2_over_dX_r0r1": (
                (X2 - X1) / (X1 - X0) if X1 != X0 else float("nan")),
            "dX_dlambda_r0_r1": (X1 - X0) / (l1 - l0),
            "dX_dlambda_r1_r2": (X2 - X1) / (l2 - l1),
            # D2 dynamic range of the curve over the arm's whole X range
            "dynamic_range_pct": {
                e: 100.0 * (max(c0["itl95"][e], c1["itl95"][e], c2["itl95"][e])
                            - min(c0["itl95"][e], c1["itl95"][e], c2["itl95"][e]))
                / min(c0["itl95"][e], c1["itl95"][e], c2["itl95"][e]) for e in EST},
            # local log-log slopes (curvature evidence)
            "loglog_slope_median_r0_r1": (
                math.log(c1["itl95"]["median"] / c0["itl95"]["median"])
                / math.log(X1 / X0)),
            "loglog_slope_median_r1_r2": (
                math.log(c2["itl95"]["median"] / c1["itl95"]["median"])
                / math.log(X2 / X1)),
            "rho_nominal_lambda_over_mu_p_probeC": [
                l0 / c2["X_recomputed"], l1 / c2["X_recomputed"], l2 / c2["X_recomputed"]],
            "rho_realized_approx": [
                l0 / c2["X_recomputed"] / c0["span_factor_observed"],
                l1 / c2["X_recomputed"] / c1["span_factor_observed"],
                l2 / c2["X_recomputed"] / c2["span_factor_observed"]],
        })

        for e in EST:
            y0, y1, y2 = c0["itl95"][e], c1["itl95"][e], c2["itl95"][e]
            # ---- registered LOO: hold out r1, interpolate from r0 & r2 ----
            p_lin = interp_linear(X0, y0, X2, y2, X1)
            p_log = interp_loglog(X0, y0, X2, y2, X1)
            out["loo"].append({
                "arm": arm, "held_out": "r1", "estimator": e,
                "obs_ms": y1, "outer_ms": [y0, y2],
                "pred_linear_X_ms": p_lin, "err_linear_X_pct": rel_err_pct(p_lin, y1),
                "pred_loglog_ms": p_log, "err_loglog_pct": rel_err_pct(p_log, y1),
                # --- D3 null models: does the interpolant beat a constant? ---
                "null_nearest_anchor_err_pct": rel_err_pct(y2, y1),
                "null_midpoint_err_pct": rel_err_pct(0.5 * (y0 + y2), y1),
                "null_flat_from_r0_err_pct": rel_err_pct(y0, y1),
                # --- D4 noise floor (request resampling only) ---
                "noise_floor_ci95_halfwidth_pct":
                    c1["itl95_request_bootstrap"]["half_width_pct"][e],
                "regime_note": "outer point r2 is ABOVE the knee (ach/off %.3f)"
                               % c2["ach_over_off"],
            })
            # ---- auxiliary: below-knee pair (r0,r1) EXTRAPOLATED to r2 ----
            e_lin = interp_linear(X0, y0, X1, y1, X2)
            e_log = interp_loglog(X0, y0, X1, y1, X2)
            out["extrapolation_aux"].append({
                "arm": arm, "target": "r2", "estimator": e,
                "obs_ms": y2, "from_ms": [y0, y1],
                "pred_linear_X_ms": e_lin, "err_linear_X_pct": rel_err_pct(e_lin, y2),
                "pred_loglog_ms": e_log, "err_loglog_pct": rel_err_pct(e_log, y2),
                "kind": "EXTRAPOLATION -- forbidden by Q-A §2.4; curvature diagnostic only",
            })
            # ---- supplementary: same LOO on the offered-rate axis ----
            q_lin = interp_linear(l0, y0, l2, y2, l1)
            q_log = interp_loglog(l0, y0, l2, y2, l1)
            out["loo_lambda_axis_diagnostic"].append({
                "arm": arm, "held_out": "r1", "estimator": e, "obs_ms": y1,
                "pred_linear_lambda_ms": q_lin, "err_linear_lambda_pct": rel_err_pct(q_lin, y1),
                "pred_loglog_lambda_ms": q_log, "err_loglog_lambda_pct": rel_err_pct(q_log, y1),
                "kind": "DIAGNOSTIC -- axis is offered rate, not the estimand's X axis",
            })

    # --- where would the REGISTERED ladder Lambda land, per arm? -------------
    sf = [c["span_factor_observed"] for c in out["cells"]]
    span_med, span_lo, span_hi = (float(np.median(sf)), float(min(sf)), float(max(sf)))
    ladder = []
    for D, mu in sorted(MU_P_S1.items()):
        exec_rungs = [lam for lam in LAMBDA_REGISTERED if lam <= 0.95 * mu]
        top = exec_rungs[-1] if exec_rungs else float("nan")
        ladder.append({
            "arm": f"d{D}",
            "mu_p_step1": mu,
            "exec_cap_0.95_mu_p": 0.95 * mu,
            "executed_rungs": exec_rungs,
            "n_executed": len(exec_rungs),
            "rho_nominal": [lam / mu for lam in exec_rungs],
            "top_rung_rho_realized_range": [top / mu / span_hi, top / mu / span_lo],
            "top_rung_realized_can_exceed_0.95": bool(top / mu / span_lo > 0.95),
            "rho_realized_if_span_factor_%.4f" % span_med:
                [lam / mu / span_med for lam in exec_rungs],
            "note": ("2-rung arm: linear and log-log fits COINCIDE (Q-A §2.4) "
                     "-> its curve is a straight line by construction"
                     if len(exec_rungs) <= 2 else ""),
        })
    out["registered_ladder_coverage"] = {
        "Lambda": list(LAMBDA_REGISTERED),
        "geometric_ratio": [LAMBDA_REGISTERED[i + 1] / LAMBDA_REGISTERED[i]
                            for i in range(len(LAMBDA_REGISTERED) - 1)],
        "span_factor_median_probeC": span_med,
        "span_factor_range_probeC": [span_lo, span_hi],
        "caveat": ("Q-A §2.1 states the execution condition in NOMINAL lambda; "
                   "probe C shows the realised arrival stream is ~1/span_factor "
                   "faster, so realised rho is ~%.0f%% higher than nominal"
                   % (100 * (1 / span_med - 1))),
        "arms": ladder,
    }

    # --- arithmetic behind the recommendations (RESULT doc sec 8) -----------
    def plan(name, lam, cap_frac, extra_cap_rungs_if_below=0):
        rows, total = [], 0
        for D, mu in sorted(MU_P_S1.items()):
            cap = cap_frac * mu
            ex = [x for x in lam if x <= cap]
            add = []
            if extra_cap_rungs_if_below and len(ex) < extra_cap_rungs_if_below:
                add = [cap, cap / 1.29]
                ex = sorted(set(ex) | set(add))
            rows.append({"arm": f"d{D}", "cap": cap, "n": len(ex),
                         "rungs": [round(v, 4) for v in ex],
                         "added_cap_rungs": [round(v, 4) for v in add]})
            total += len(ex)
        benches = total * 2 * 4                      # 2 shapes x 4 rounds
        h_bench = benches * 170.0 / 3600.0
        h_boot = 16 * 195.0 / 3600.0                 # Q-A sec 6
        return {"name": name, "cap_fraction_of_mu_p": cap_frac,
                "ladder": list(lam), "arms": rows,
                "cells_per_round_per_shape": total, "benches_total": benches,
                "gpu_h_bench": h_bench, "gpu_h_boot": h_boot,
                "gpu_h_stageA": 0.30,
                "gpu_h_total": h_bench + h_boot + 0.30}

    L7 = (0.09, 0.116, 0.15, 0.25, 0.42, 0.54, 0.70)
    out["recommendation_arithmetic"] = [
        plan("baseline: Lambda, nominal cap 0.95*mu_p", LAMBDA_REGISTERED, 0.95),
        plan("(ii) only: Lambda, realised-safe cap 0.785*mu_p", LAMBDA_REGISTERED, 0.785),
        plan("(ii)+(iii-a): + deterministic cap rungs for arms with <3",
             LAMBDA_REGISTERED, 0.785, extra_cap_rungs_if_below=3),
        plan("(iii-b): 7-point shared ladder, cap 0.785*mu_p", L7, 0.785),
    ]

    errs = [r[k] for r in out["loo"] for k in ("err_linear_X_pct", "err_loglog_pct")]
    out["summary"] = {
        "n_loo_entries": len(errs),
        "loo_err_max_pct": float(np.nanmax(errs)),
        "loo_err_median_pct": float(np.nanmedian(errs)),
        "loo_err_mean_pct": float(np.nanmean(errs)),
        "loo_err_min_pct": float(np.nanmin(errs)),
        "n_over_resolution": int(sum(1 for v in errs if v > RESOLUTION_PCT)),
        "verdict_grid_density": ("PASS (max < 7.2%)" if float(np.nanmax(errs)) < RESOLUTION_PCT
                                 else "FAIL (max >= 7.2%)"),
        "verdict_scope": "applies to curve shape only; see 'scope'",
        # degeneracy audit of the PASS itself
        "null_nearest_anchor_max_pct": float(np.nanmax(
            [r["null_nearest_anchor_err_pct"] for r in out["loo"]])),
        "null_nearest_anchor_median_pct": float(np.nanmedian(
            [r["null_nearest_anchor_err_pct"] for r in out["loo"]])),
        "null_flat_from_r0_max_pct": float(np.nanmax(
            [r["null_flat_from_r0_err_pct"] for r in out["loo"]])),
        "noise_floor_median_pct": float(np.nanmedian(
            [r["noise_floor_ci95_halfwidth_pct"] for r in out["loo"]])),
        "n_loo_err_below_own_noise_floor": int(sum(
            1 for r in out["loo"]
            for k in ("err_linear_X_pct", "err_loglog_pct")
            if r[k] < r["noise_floor_ci95_halfwidth_pct"])),
        "n_interpolant_beats_nearest_anchor_null": int(sum(
            1 for r in out["loo"]
            if min(r["err_linear_X_pct"], r["err_loglog_pct"])
            < r["null_nearest_anchor_err_pct"])),
        "n_loo_comparisons_vs_null": len(out["loo"]),
        "informativeness_caveat": (
            "A PASS is evidence about grid density only if the test could have "
            "FAILED.  Check lever arm (D1), dynamic range vs 7.2% (D2), the "
            "nearest-anchor null model (D3) and the request-sampling noise "
            "floor (D4) before citing this PASS."),
    }
    aux = [r[k] for r in out["extrapolation_aux"]
           for k in ("err_linear_X_pct", "err_loglog_pct")]
    out["summary_extrapolation_aux"] = {
        "err_max_pct": float(np.nanmax(aux)),
        "err_median_pct": float(np.nanmedian(aux)),
        "n_over_resolution": int(sum(1 for v in aux if v > RESOLUTION_PCT)),
    }
    lam = [r[k] for r in out["loo_lambda_axis_diagnostic"]
           for k in ("err_linear_lambda_pct", "err_loglog_lambda_pct")]
    out["summary_lambda_axis_diagnostic"] = {
        "err_max_pct": float(np.nanmax(lam)),
        "err_median_pct": float(np.nanmedian(lam)),
        "n_over_resolution": int(sum(1 for v in lam if v > RESOLUTION_PCT)),
    }
    return out


def print_tables(res):
    def f(v, n=4):
        return "nan" if v != v else f"{v:.{n}f}"

    print("=== cells (probe C, o96) ===")
    print(f"{'cell':>12} {'N':>4} {'lambda':>7} {'X':>9} {'ach/off':>8} {'med':>8} {'mean':>8} "
          f"{'trim10':>8} {'pool95':>8} {'span_f':>7} | 95%CI half-width % (req resample)")
    for c in res["cells"]:
        hw = c["itl95_request_bootstrap"]["half_width_pct"]
        print(f"{c['arm']+'_'+c['rung']:>12} {c['n_requests_ok']:>4} {c['offered_rate']:>7.3f} "
              f"{c['X_recomputed']:>9.6f} "
              f"{c['ach_over_off']:>8.4f} {c['itl95']['median']:>8.3f} {c['itl95']['mean']:>8.3f} "
              f"{c['itl95']['trim10']:>8.3f} {c['itl95']['pooled_p95']:>8.3f} "
              f"{c['span_factor_observed']:>7.4f} | " +
              " ".join(f"{e[:4]}=±{hw[e]:.3f}" for e in EST))
    print(f"\nX vs bench 'request_throughput'  max rel err : "
          f"{res['X_crosscheck_max_relerr_vs_bench_field']:.3e}   "
          f"(!! ALGEBRAIC IDENTITY, bench_serving.py:1093 -- validates nothing)")
    print(f"X vs cell_*.json 'achieved_rate' max rel err : "
          f"{res['X_crosscheck_max_relerr_vs_cell_json']:.3e}   (same identity)")

    print("\n=== D1/D2 conditioning of the X axis + dynamic range ===")
    for c in res["conditioning"]:
        dr = c["dynamic_range_pct"]
        print(f"  {c['arm']}: X={[round(v,6) for v in c['X']]}")
        print(f"        dX(r0,r1)={c['dX_r0_r1']:.6f}  dX(r1,r2)={c['dX_r1_r2']:.6f}  "
              f"collapse={c['X_axis_collapse_ratio_dX_r1r2_over_dX_r0r1']:.4f}")
        print(f"        D1 lever(X)={c['lever_arm_frac_(X1-X0)/(X2-X0)']:.4f}   "
              f"lever(lambda)={c['lever_arm_frac_lambda']:.4f}")
        print(f"        dX/dlambda: {c['dX_dlambda_r0_r1']:.4f} (r0->r1)  "
              f"{c['dX_dlambda_r1_r2']:.4f} (r1->r2)")
        print(f"        D2 dynamic range %: " +
              "  ".join(f"{e}={dr[e]:.2f}" for e in EST) +
              f"   [resolution {RESOLUTION_PCT}%]")
        print(f"        log-log slope (median): {c['loglog_slope_median_r0_r1']:.3f} (r0->r1) "
              f"-> {c['loglog_slope_median_r1_r2']:.3f} (r1->r2)")
        print(f"        rho realized approx: "
              f"{[round(v,3) for v in c['rho_realized_approx']]}")

    print("\n=== LOO (hold out r1; interpolate r0,r2) -- REGISTERED TEST ===")
    print(f"{'arm':>5} {'estimator':>11} {'obs':>8} {'pred_lin':>9} {'err%':>7} "
          f"{'pred_log':>9} {'err%':>7} | {'D3 near':>8} {'D3 mid':>7} {'D3 flat':>8} "
          f"| {'D4 noise':>8}")
    for r in res["loo"]:
        print(f"{r['arm']:>5} {r['estimator']:>11} {r['obs_ms']:>8.3f} "
              f"{r['pred_linear_X_ms']:>9.3f} {r['err_linear_X_pct']:>7.3f} "
              f"{r['pred_loglog_ms']:>9.3f} {r['err_loglog_pct']:>7.3f} | "
              f"{r['null_nearest_anchor_err_pct']:>8.3f} "
              f"{r['null_midpoint_err_pct']:>7.3f} "
              f"{r['null_flat_from_r0_err_pct']:>8.3f} | "
              f"{r['noise_floor_ci95_halfwidth_pct']:>8.3f}")
    s = res["summary"]
    print(f"\n  max {s['loo_err_max_pct']:.3f}%  median {s['loo_err_median_pct']:.3f}%  "
          f"mean {s['loo_err_mean_pct']:.3f}%  min {s['loo_err_min_pct']:.3f}%  "
          f"| over 7.2%: {s['n_over_resolution']}/{s['n_loo_entries']}  => {s['verdict_grid_density']}")
    print(f"  D3 nearest-anchor null: max {s['null_nearest_anchor_max_pct']:.3f}%  "
          f"median {s['null_nearest_anchor_median_pct']:.3f}%   "
          f"| flat-from-r0 null max {s['null_flat_from_r0_max_pct']:.3f}%")
    print(f"  D4 request-sampling noise floor (median 95% CI half-width): "
          f"{s['noise_floor_median_pct']:.3f}%   "
          f"| LOO errors inside their own noise floor: "
          f"{s['n_loo_err_below_own_noise_floor']}/{s['n_loo_entries']}")
    print(f"  interpolant beats the nearest-anchor constant in "
          f"{s['n_interpolant_beats_nearest_anchor_null']}/"
          f"{s['n_loo_comparisons_vs_null']} (arm,estimator) cases")

    print("\n=== D5 mixture structure (untreated mode = split never fired) ===")
    print(f"  untreated threshold = {UNTREATED_TOL} x ITL_SOLO({ITL_SOLO_MS:.3f} ms) "
          f"= {UNTREATED_TOL*ITL_SOLO_MS:.3f} ms")
    print(f"{'cell':>12} {'untreated':>10} {'median in':>16} {'gap ms':>8} {'gap between':>18} "
          f"{'f_time':>7} {'t@108SM':>8}")
    for c in res["cells"]:
        m = c["D5_mixture"]
        gb = m["largest_gap_between_ms"]
        print(f"{c['arm']+'_'+c['rung']:>12} {m['untreated_fraction']:>10.3f} "
              f"{m['median_sits_in']:>16} {m['largest_gap_ms']:>8.2f} "
              f"{('%.2f-%.2f' % (gb[0], gb[1])) if gb else '-':>18} "
              f"{m['telemetry_f_time']:>7.3f} "
              f"{m['decode_time_frac_at_108sm_split_not_fired']:>8.3f}")
    print("  !! the MEDIAN estimator jumps between modes as untreated_fraction "
          "crosses 0.5 -> y(X) has a STEP; no ladder density reproduces a step.")
    print(f"\n  mode decomposition  itl95_cell ~ (1-w)*untreated + w*treated")
    print(f"{'cell':>12} {'w=treated frac':>15} {'untreated mean':>15} {'treated mean':>13} "
          f"{'cell median':>12} {'cell mean':>10}")
    for c in res["cells"]:
        m = c["D5_mixture"]
        um = m["untreated_mode_mean_ms"]
        print(f"{c['arm']+'_'+c['rung']:>12} {1-m['untreated_fraction']:>15.3f} "
              f"{(f'{um:.3f}' if um else '-'):>15} {m['treated_mode_mean_ms']:>13.3f} "
              f"{c['itl95']['median']:>12.3f} {c['itl95']['mean']:>10.3f}")
    for arm in ARMS:
        tm = [c["D5_mixture"]["treated_mode_mean_ms"] for c in res["cells"]
              if c["arm"] == arm]
        print(f"  {arm}: treated-mode mean over the 3 rungs "
              f"{[round(v,3) for v in tm]}  -> range {100*(max(tm)-min(tm))/min(tm):.2f}% "
              f"(vs {RESOLUTION_PCT}% resolution)")

    print("\n=== D4b noise floor projected onto Q-A's registered per-rung N ===")
    print(f"  Lambda = {list(LAMBDA_REGISTERED)}  T_win = {T_WIN:.0f}s  =>  "
          f"N = round(lambda*T_win)+1 = {list(N_REGISTERED)}")
    print("  m-out-of-n bootstrap on each probe C cell; 95% CI half-width [%] of itl95_cell")
    hdr = "  ".join(f"N={m:>3}" for m in N_REGISTERED)
    for e in EST:
        print(f"  -- estimator: {e}")
        print(f"{'cell':>12} {'N_obs':>6}   {hdr}")
        for c in res["cells"]:
            proj = c["itl95_N_projection"]
            row = "  ".join(f"{proj[str(m)][e] if isinstance(list(proj)[0], str) else proj[m][e]:>5.2f}"
                            for m in N_REGISTERED)
            print(f"{c['arm']+'_'+c['rung']:>12} {c['n_requests_ok']:>6}   {row}")

    print("\n=== registered ladder Lambda coverage (Step 1 mu_p) ===")
    rl = res["registered_ladder_coverage"]
    print(f"  Lambda = {rl['Lambda']}  ratios {[round(v,3) for v in rl['geometric_ratio']]}")
    print(f"  {rl['caveat']}")
    key = [k for k in rl["arms"][0] if k.startswith("rho_realized_if")][0]
    for a in rl["arms"]:
        print(f"    {a['arm']}: mu_p={a['mu_p_step1']:.4f} cap={a['exec_cap_0.95_mu_p']:.4f} "
              f"n_exec={a['n_executed']}  rho_nom={[round(v,3) for v in a['rho_nominal']]}  "
              f"rho_real={[round(v,3) for v in a[key]]}")
        tr = a["top_rung_rho_realized_range"]
        print(f"          top rung realized rho over observed span factors "
              f"{rl['span_factor_range_probeC'][0]:.4f}-{rl['span_factor_range_probeC'][1]:.4f}: "
              f"{tr[0]:.3f}-{tr[1]:.3f}"
              + ("   ** CAN EXCEED the registered 0.95" if a["top_rung_realized_can_exceed_0.95"] else ""))
        if a["note"]:
            print(f"          ** {a['note']}")

    print("\n=== recommendation arithmetic (RESULT doc sec 8) ===")
    for p in res["recommendation_arithmetic"]:
        print(f"  {p['name']}")
        print("     " + "  ".join(f"{a['arm']}:{a['n']}" for a in p["arms"]) +
              f"   = {p['cells_per_round_per_shape']} cells/(round,shape)"
              f"  -> {p['benches_total']} benches"
              f"  -> {p['gpu_h_total']:.2f} GPU-h total"
              f" (bench {p['gpu_h_bench']:.2f} + boot {p['gpu_h_boot']:.2f} + StageA 0.30)")
        for a in p["arms"]:
            if a["added_cap_rungs"]:
                print(f"       {a['arm']} cap={a['cap']:.4f} rungs={a['rungs']} "
                      f"(added {a['added_cap_rungs']})")

    print("\n=== AUX: below-knee pair (r0,r1) EXTRAPOLATED to r2 (NOT the registered test) ===")
    print(f"{'arm':>5} {'estimator':>11} {'obs':>8} {'pred_lin':>9} {'err%':>8} "
          f"{'pred_log':>9} {'err%':>8}")
    for r in res["extrapolation_aux"]:
        print(f"{r['arm']:>5} {r['estimator']:>11} {r['obs_ms']:>8.3f} "
              f"{r['pred_linear_X_ms']:>9.3f} {r['err_linear_X_pct']:>8.3f} "
              f"{r['pred_loglog_ms']:>9.3f} {r['err_loglog_pct']:>8.3f}")
    a = res["summary_extrapolation_aux"]
    print(f"\n  max {a['err_max_pct']:.3f}%  median {a['err_median_pct']:.3f}%  "
          f"| over 7.2%: {a['n_over_resolution']}/24")

    print("\n=== DIAGNOSTIC: same LOO on the offered-rate axis lambda ===")
    print(f"{'arm':>5} {'estimator':>11} {'obs':>8} {'pred_lin':>9} {'err%':>7} "
          f"{'pred_log':>9} {'err%':>7}")
    for r in res["loo_lambda_axis_diagnostic"]:
        print(f"{r['arm']:>5} {r['estimator']:>11} {r['obs_ms']:>8.3f} "
              f"{r['pred_linear_lambda_ms']:>9.3f} {r['err_linear_lambda_pct']:>7.3f} "
              f"{r['pred_loglog_lambda_ms']:>9.3f} {r['err_loglog_lambda_pct']:>7.3f}")
    d = res["summary_lambda_axis_diagnostic"]
    print(f"\n  max {d['err_max_pct']:.3f}%  median {d['err_median_pct']:.3f}%  "
          f"| over 7.2%: {d['n_over_resolution']}/24")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(PROBE_C))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    res = run(Path(args.dir))
    print_tables(res)
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
