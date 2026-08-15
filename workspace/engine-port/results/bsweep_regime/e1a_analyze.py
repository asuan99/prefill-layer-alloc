#!/usr/bin/env python3
"""E-1a design pre-analysis (GPU 0).  Consumes e1a_repcells_2026-08-14.json.

Deliverables (T1-T4 of the tasking):
  T1  S(B) = eps_Ha8(B) - eps_T8(B), SIGNED, decomposed to rep level.
  T2  regression S(B) = k + b*dw(B) WITH INTERCEPT; t-CI on k and b.
  T3  measured rep-level sd(eps) -> MDE for b at n = 4/6/8 (t-CI; no bootstrap).
  T4  reachability: realized-SM-conditioned realized-B histograms; does B=24/40 exist?

Statistics: t-based only (methodology gates #14/#27 -- percentile bootstrap
forbidden).  No scipy in this venv, so the Student-t CDF/quantile are
implemented here from the regularized incomplete beta (Numerical Recipes
continued fraction) and unit-tested against known values at import time.

NOT BLIND: claims-auditor already computed S(8)=0.2058, S(16)=0.3016 for the
44->92 pair from the pooled producer table.  Reproducing those is a
cross-check, not an independent confirmation.
"""
import collections
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
IN = os.path.join(HERE, "e1a_repcells_2026-08-14.json")
OUT = os.path.join(HERE, "e1a_preanalysis_2026-08-14.json")

MIN_N = 50            # per-rep-cell interval count (producer's own threshold)
ARMS = ("Ha8", "T8")
SMS = (16, 44, 92)
PAIRS = ((16, 44), (16, 92), (44, 92))
CAPTURED_BS = [1, 2, 4, 8, 12, 16, 24, 32, 40, 48]   # from srv.log "Capture cuda graph bs"

# ---- traffic model inputs (auditor-corrected; TRAFFIC_ROOFLINE_DIAGNOSTIC 3.1/3.2) ----
GB = 1e9
MiB = 1048576.0
W = {"T8": 14.141 * GB, "Ha8": 22.061 * GB}
KV_PER_SEQ = {"T8": 70.1 * MiB, "Ha8": 455.7 * MiB}       # at L = 1282
STATE_PER_SEQ = {"T8": 0.0, "Ha8": 145.19 * MiB}          # L-invariant


def c_per_seq(arm, kappa):
    return KV_PER_SEQ[arm] + kappa * STATE_PER_SEQ[arm]


def w_share(arm, B, kappa):
    return 100.0 * W[arm] / (W[arm] + B * c_per_seq(arm, kappa))


def dw(B, kappa=2):
    """dw(B) = w_T8(B) - w_Ha8(B), in percentage points.  [C] = computed from
    config, contains NO measured quantity (gate #9 independence note)."""
    return w_share("T8", B, kappa) - w_share("Ha8", B, kappa)


def pad_bs(B):
    for s in CAPTURED_BS:
        if B <= s:
            return s
    return B


# ------------------------------ Student t ------------------------------------
def _betacf(a, b, x):
    MAXIT, EPS, FPMIN = 300, 3e-16, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < EPS:
            break
    return h


def betai(a, b, x):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
             + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lbeta) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbeta) * _betacf(b, a, 1.0 - x) / b


def t_cdf(t, df):
    x = df / (df + t * t)
    p = 0.5 * betai(df / 2.0, 0.5, x)
    return 1.0 - p if t > 0 else p


def t_ppf(p, df):
    lo, hi = -400.0, 400.0
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


assert abs(t_ppf(0.975, 2) - 4.30265) < 1e-3, t_ppf(0.975, 2)
assert abs(t_ppf(0.975, 3) - 3.18245) < 1e-3
assert abs(t_ppf(0.975, 10) - 2.22814) < 1e-3
assert abs(t_ppf(0.975, 1e6) - 1.95996) < 1e-3
assert abs(t_ppf(0.80, 4) - 0.94096) < 1e-3


def mean(v):
    return sum(v) / len(v)


def var(v):
    if len(v) < 2:
        return None
    m = mean(v)
    return sum((x - m) ** 2 for x in v) / (len(v) - 1)


# ------------------------------ load ----------------------------------------
D = json.load(open(IN))
rep_cells = D["rep_cells"]
pooled = {(c["arm"], c["sm"], c["bs"]): c for c in D["pooled_cells"]}

# (arm, sm, B) -> list of dicts (one per rep that clears MIN_N)
G = collections.defaultdict(list)
for c in rep_cells:
    if c["arm"] in ARMS and c["sm"] in SMS and c["n"] >= MIN_N:
        G[(c["arm"], c["sm"], c["bs"])].append(c)

report = {"generated": "2026-08-14", "blind": False,
          "blind_note": "claims-auditor already produced S(8)=0.2058, S(16)=0.3016 (44->92, pooled p50)",
          "min_n_per_rep_cell": MIN_N,
          "captured_cuda_graph_bs": CAPTURED_BS}

# ---------------------- SC1 producer-identity self-check ---------------------
sc1 = [{"cell": f"{a}/SM{s}/B{b}", "e1a_pooled_p50": round(pooled[(a, s, b)]["p50"], 2)}
       for (a, s, b) in [("M8", 16, 12), ("M8", 92, 12), ("Ha8", 44, 8), ("Ha8", 92, 8),
                         ("Ha8", 44, 16), ("Ha8", 92, 16), ("T8", 44, 8), ("T8", 92, 8),
                         ("T8", 44, 16), ("T8", 92, 16)] if (a, s, b) in pooled]
report["SC1_pooled_reproduces_s8_batch_matched"] = sc1

# ------------------------------ T1 -------------------------------------------
def eps_cell(arm, lo, hi, B, stat="p50"):
    """eps and its SE from rep-level log-ITL means (independent groups)."""
    gl, gh = G.get((arm, lo, B), []), G.get((arm, hi, B), [])
    if not gl or not gh:
        return None
    L = math.log(hi / lo)
    xl = [math.log(c[stat]) for c in gl]
    xh = [math.log(c[stat]) for c in gh]
    e = (mean(xl) - mean(xh)) / L
    vl, vh = var(xl), var(xh)
    out = {"arm": arm, "B": B, "pair": f"{lo}->{hi}", "eps": e,
           "n_rep_lo": len(xl), "n_rep_hi": len(xh),
           "itl_lo_repmean_ms": math.exp(mean(xl)), "itl_hi_repmean_ms": math.exp(mean(xh)),
           "n_itl_lo": sum(c["n"] for c in gl), "n_itl_hi": sum(c["n"] for c in gh),
           "jobs_lo": sorted({c["job"] for c in gl}), "jobs_hi": sorted({c["job"] for c in gh})}
    if vl is None or vh is None:
        out.update({"se": None, "df": None,
                    "scoreable": False, "why": "n_rep<2 on at least one side"})
        return out
    a_, b_ = vl / len(xl) / L / L, vh / len(xh) / L / L
    se = math.sqrt(a_ + b_)
    df = (a_ + b_) ** 2 / (a_ ** 2 / (len(xl) - 1) + b_ ** 2 / (len(xh) - 1))
    out.update({"se": se, "df": df, "scoreable": True,
                "sd_rep_lo_logitl": math.sqrt(vl), "sd_rep_hi_logitl": math.sqrt(vh),
                "sd_rep_eps": math.sqrt(vl + vh) / L})
    return out


def eps_pooled(arm, lo, hi, B, stat="p50"):
    a, b = pooled.get((arm, lo, B)), pooled.get((arm, hi, B))
    if not a or not b:
        return None
    return math.log(a[stat] / b[stat]) / math.log(hi / lo)


T1 = {}
for lo, hi in PAIRS:
    rows = []
    Bs = sorted({B for (arm, s, B) in G if s in (lo, hi)})
    for B in Bs:
        e = {arm: eps_cell(arm, lo, hi, B) for arm in ARMS}
        e95 = {arm: eps_cell(arm, lo, hi, B, "p95") for arm in ARMS}
        row = {"B": B, "dw_kappa2": dw(B, 2), "dw_kappa1": dw(B, 1),
               "B_padded": pad_bs(B), "dw_padded_kappa2": dw(pad_bs(B), 2)}
        for arm in ARMS:
            row[arm] = e[arm]
            row[arm + "_p95_eps"] = e95[arm]["eps"] if e95[arm] else None
            row[arm + "_eps_pooled"] = eps_pooled(arm, lo, hi, B)
        if e["Ha8"] and e["T8"]:
            row["S"] = e["Ha8"]["eps"] - e["T8"]["eps"]
            row["S_pooled_auditor_method"] = (row["Ha8_eps_pooled"] - row["T8_eps_pooled"]
                                              if row["Ha8_eps_pooled"] is not None
                                              and row["T8_eps_pooled"] is not None else None)
            row["S_p95"] = ((row["Ha8_p95_eps"] - row["T8_p95_eps"])
                            if row["Ha8_p95_eps"] is not None and row["T8_p95_eps"] is not None
                            else None)
            row["n_rep_min"] = min(e["Ha8"]["n_rep_lo"], e["Ha8"]["n_rep_hi"],
                                   e["T8"]["n_rep_lo"], e["T8"]["n_rep_hi"])
            if e["Ha8"]["scoreable"] and e["T8"]["scoreable"]:
                v = e["Ha8"]["se"] ** 2 + e["T8"]["se"] ** 2
                se = math.sqrt(v)
                df = v ** 2 / (e["Ha8"]["se"] ** 4 / e["Ha8"]["df"]
                               + e["T8"]["se"] ** 4 / e["T8"]["df"])
                tq = t_ppf(0.975, df)
                row.update({"S_se": se, "S_df": df, "S_scoreable": True,
                            "S_ci95": [row["S"] - tq * se, row["S"] + tq * se],
                            "sd_rep_S": math.sqrt(e["Ha8"]["sd_rep_eps"] ** 2
                                                  + e["T8"]["sd_rep_eps"] ** 2)})
            else:
                row.update({"S_se": None, "S_scoreable": False,
                            "S_status": "UNSCOREABLE (n_rep<2 on >=1 of the 4 groups)"})
        else:
            row["S"] = None
            row["S_status"] = "UNSCOREABLE (arm/SM/B cell empty)"
        rows.append(row)
    T1[f"{lo}->{hi}"] = rows
report["T1_S_of_B"] = T1

# ------------------------------ T2 -------------------------------------------
def wls(xs, ys, ses=None):
    """(weighted) least squares y = k + b x. Returns k,b and both CI flavours."""
    n = len(xs)
    wgt = [1.0] * n if ses is None else [1.0 / s ** 2 for s in ses]
    Sw = sum(wgt)
    xb = sum(w * x for w, x in zip(wgt, xs)) / Sw
    yb = sum(w * y for w, y in zip(wgt, ys)) / Sw
    Sxx = sum(w * (x - xb) ** 2 for w, x in zip(wgt, xs))
    Sxy = sum(w * (x - xb) * (y - yb) for w, x, y in zip(wgt, xs, ys))
    b = Sxy / Sxx
    k = yb - b * xb
    res = [y - (k + b * x) for x, y in zip(xs, ys)]
    df = n - 2
    out = {"n_points": n, "k": k, "b": b, "df_resid": df,
           "x_mean": xb, "Sxx": Sxx, "resid": res}
    if df >= 1:
        s2 = sum(w * r ** 2 for w, r in zip(wgt, res)) / df       # scaled residual var
        se_b = math.sqrt(s2 / Sxx)
        se_k = math.sqrt(s2 * (1.0 / Sw + xb ** 2 / Sxx))
        tq = t_ppf(0.975, df)
        out.update({"se_b_resid": se_b, "se_k_resid": se_k,
                    "b_ci95_resid": [b - tq * se_b, b + tq * se_b],
                    "k_ci95_resid": [k - tq * se_k, k + tq * se_k],
                    "t_b": b / se_b, "t_k": k / se_k,
                    "p_b": 2 * (1 - t_cdf(abs(b / se_b), df)),
                    "p_k": 2 * (1 - t_cdf(abs(k / se_k), df))})
    if ses is not None:
        se_b = math.sqrt(1.0 / Sxx)                                # known-variance
        se_k = math.sqrt(1.0 / Sw + xb ** 2 / Sxx)
        z = 1.959964
        out.update({"se_b_known": se_b, "se_k_known": se_k,
                    "b_ci95_known": [b - z * se_b, b + z * se_b],
                    "k_ci95_known": [k - z * se_k, k + z * se_k],
                    "z_b": b / se_b, "z_k": k / se_k})
    return out


T2 = {}
for pair, rows in T1.items():
    for kap in (1, 2):
        for xkey, tag in ((f"dw_kappa{kap}", f"kappa{kap}_rawB"),
                          ("dw_padded_kappa2", "kappa2_paddedB")):
            if tag == "kappa2_paddedB" and kap != 2:
                continue
            pts = [r for r in rows if r.get("S_scoreable")]
            if len(pts) < 3:
                T2[f"{pair}|{tag}"] = {"status": "UNSCOREABLE (fewer than 3 scoreable B points)",
                                       "n_points": len(pts),
                                       "B_scoreable": [r["B"] for r in pts]}
                continue
            xs = [r[xkey] for r in pts]
            ys = [r["S"] for r in pts]
            ses = [r["S_se"] for r in pts]
            T2[f"{pair}|{tag}"] = {
                "B_points": [r["B"] for r in pts], "x": xs, "S": ys, "S_se": ses,
                "OLS_unweighted": wls(xs, ys),
                "WLS_by_measurement_se": wls(xs, ys, ses),
                "ratio_R_naive": (ys[-1] / ys[0]) if ys[0] else None,
            }
report["T2_regression"] = T2

# ------------------------------ T3 -------------------------------------------
# rep-level sd of log ITL, split into within-job and between-job components.
sd_tab = []
for (arm, sm, B), cells in sorted(G.items()):
    xs = [math.log(c["p50"]) for c in cells]
    byjob = collections.defaultdict(list)
    for c in cells:
        byjob[c["job"]].append(math.log(c["p50"]))
    within = [v for j, vv in byjob.items() if len(vv) >= 2 for v in [var(vv)]]
    sd_tab.append({"arm": arm, "sm": sm, "B": B, "n_rep": len(xs),
                   "sd_log_p50_all": math.sqrt(var(xs)) if var(xs) else None,
                   "sd_log_p50_within_job": (math.sqrt(mean(within)) if within else None),
                   "jobs": {j: len(v) for j, v in sorted(byjob.items())},
                   "cv_pct_all": (100 * math.sqrt(var(xs)) if var(xs) else None)})
report["T3_sd_table"] = sd_tab

# headline sd(eps) per arm for the 44->92 pair, using cells with n_rep>=3
def sd_eps_summary(lo, hi):
    out = {}
    L = math.log(hi / lo)
    for arm in ARMS:
        per_side = collections.defaultdict(list)
        for r in sd_tab:
            if r["arm"] == arm and r["sm"] in (lo, hi) and r["n_rep"] >= 3 and r["B"] >= 4:
                per_side[r["sm"]].append(r)
        vals = {}
        for sm, rs in per_side.items():
            vals[sm] = {
                "cells": [(r["B"], r["n_rep"], round(r["sd_log_p50_all"], 4)) for r in rs],
                "pooled_sd_log_p50": math.sqrt(mean([r["sd_log_p50_all"] ** 2 for r in rs])),
                "pooled_sd_log_p50_within_job": (
                    math.sqrt(mean([r["sd_log_p50_within_job"] ** 2 for r in rs
                                    if r["sd_log_p50_within_job"] is not None]))
                    if any(r["sd_log_p50_within_job"] is not None for r in rs) else None),
            }
        if lo in vals and hi in vals:
            s_all = math.sqrt(vals[lo]["pooled_sd_log_p50"] ** 2
                              + vals[hi]["pooled_sd_log_p50"] ** 2) / L
            wj = [vals[s]["pooled_sd_log_p50_within_job"] for s in (lo, hi)]
            s_wj = (math.sqrt(wj[0] ** 2 + wj[1] ** 2) / L
                    if all(x is not None for x in wj) else None)
            out[arm] = {"per_sm": vals, "sd_rep_eps_all": s_all,
                        "sd_rep_eps_within_job": s_wj}
    if len(out) == 2:
        out["sd_rep_S_all"] = math.sqrt(out["Ha8"]["sd_rep_eps_all"] ** 2
                                        + out["T8"]["sd_rep_eps_all"] ** 2)
        if all(out[a]["sd_rep_eps_within_job"] is not None for a in ARMS):
            out["sd_rep_S_within_job"] = math.sqrt(
                out["Ha8"]["sd_rep_eps_within_job"] ** 2
                + out["T8"]["sd_rep_eps_within_job"] ** 2)
    return out


report["T3_sd_eps"] = {f"{lo}->{hi}": sd_eps_summary(lo, hi) for lo, hi in PAIRS}

# MDE for b, planned design = 4 B levels at captured sizes, n reps per cell.
def mde_b(sd_rep_S, B_levels, ns=(4, 6, 8), kappa=2, power=0.80):
    xs = [dw(B, kappa) for B in B_levels]
    xb = mean(xs)
    Sxx = sum((x - xb) ** 2 for x in xs)
    span = max(xs) - min(xs)
    rows = []
    for n in ns:
        se_b = (sd_rep_S / math.sqrt(n)) / math.sqrt(Sxx)
        df_resid = len(B_levels) - 2
        df_known = 4 * (n - 1) * len(B_levels)      # effective, measurement-error model
        for label, df in (("t(df=n_B-2), OLS-residual model", df_resid),
                          ("t(df=large), known-variance model", df_known)):
            mult = t_ppf(0.975, df) + t_ppf(power, df)
            rows.append({"n_rep_per_cell": n, "df_model": label, "df": df,
                         "se_b": se_b, "mde_b": mult * se_b,
                         "mde_S_change_over_dw_span": mult * se_b * span})
    return {"B_levels": B_levels, "dw": xs, "dw_span": span, "Sxx": Sxx,
            "sd_rep_S_used": sd_rep_S, "rows": rows}


report["T3_MDE"] = {}
for pair in ("44->92", "16->44", "16->92"):
    s = report["T3_sd_eps"].get(pair, {})
    if "sd_rep_S_all" in s:
        report["T3_MDE"][pair] = {
            "planned_B_12_24_40_48": mde_b(s["sd_rep_S_all"], [12, 24, 40, 48]),
            "planned_B_12_24_40_48_within_job_sd": (
                mde_b(s["sd_rep_S_within_job"], [12, 24, 40, 48])
                if s.get("sd_rep_S_within_job") else None),
            "reachable_today_B_8_12_16": mde_b(s["sd_rep_S_all"], [8, 12, 16]),
        }

# ---- T2 sensitivity: B=1 is a different regime (drain tail, sign flip) -------
T2s = {}
for pair, rows in T1.items():
    for label, keep in (("all_scoreable_B", lambda r: True),
                        ("exclude_B1", lambda r: r["B"] >= 2),
                        ("exclude_B1_and_B15", lambda r: r["B"] >= 2 and r["B"] != 15)):
        pts = [r for r in rows if r.get("S_scoreable") and keep(r)]
        key = f"{pair}|{label}"
        if len(pts) < 3:
            T2s[key] = {"status": "UNSCOREABLE (<3 points)", "B": [r["B"] for r in pts]}
            continue
        xs = [r["dw_kappa2"] for r in pts]
        ys = [r["S"] for r in pts]
        ses = [r["S_se"] for r in pts]
        T2s[key] = {"B": [r["B"] for r in pts], "dw": xs, "S": ys, "S_se": ses,
                    "OLS": wls(xs, ys), "WLS": wls(xs, ys, ses)}
report["T2_sensitivity"] = T2s

# ---- observed R = S(hi)/S(lo) with delta-method CI, and the prereg D1 verdict
def R_obs(pair, blo, bhi):
    rows = {r["B"]: r for r in T1[pair]}
    a, b = rows.get(blo), rows.get(bhi)
    if not a or not b or a.get("S") is None or b.get("S") is None:
        return None
    out = {"B_lo": blo, "B_hi": bhi, "S_lo": a["S"], "S_hi": b["S"],
           "R": b["S"] / a["S"],
           "R_pred_if_k0_kappa2": dw(bhi, 2) / dw(blo, 2),
           "R_pred_if_k0_kappa1": dw(bhi, 1) / dw(blo, 1)}
    if a.get("S_se") and b.get("S_se"):
        s = math.sqrt((a["S_se"] / a["S"]) ** 2 + (b["S_se"] / b["S"]) ** 2)
        df = min(a["S_df"], b["S_df"])
        tq = t_ppf(0.975, df)
        lr = math.log(out["R"])
        out.update({"se_logR": s, "df": df,
                    "R_ci95": [math.exp(lr - tq * s), math.exp(lr + tq * s)],
                    "logR_minus_log1.5": lr - math.log(1.5),
                    "D1_verdict": ("SUPPORTED" if out["R"] >= 1.5 and lr - tq * s > 0 else
                                   "REFUTED" if out["R"] < 1.5 and lr + tq * s < math.log(1.5)
                                   else "UNDETERMINED (INSUFFICIENT SEPARATION)")})
    else:
        out["D1_verdict"] = "UNSCOREABLE (no SE at one endpoint)"
    return out


report["T2_R_verdicts"] = {f"{blo}->{bhi}": R_obs("44->92", blo, bhi)
                           for blo, bhi in ((8, 16), (7, 16), (9, 16), (7, 15), (9, 15))}

# ---- k-attenuation with the MEASURED sign of k -------------------------------
def R_with_k(k, b_, blo, bhi, kappa=2):
    num, den = k + b_ * dw(bhi, kappa), k + b_ * dw(blo, kappa)
    return num / den if den else None


b_fit = T2s.get("44->92|exclude_B1", {}).get("OLS", {}).get("b")
b_prop = None
rows44 = {r["B"]: r for r in T1["44->92"]}
if rows44.get(16, {}).get("S"):
    b_prop = rows44[16]["S"] / dw(16, 2)      # b implied by pure proportionality at B=16
att = []
for k in (-0.10, -0.08, -0.057, -0.02, 0.0, 0.02, 0.057, 0.08):
    row = {"k": k}
    for tag, b_ in (("b_from_fit_exclB1", b_fit), ("b_from_proportional_at_B16", b_prop)):
        if b_ is None:
            continue
        row[tag] = {f"{lo}->{hi}": R_with_k(k, b_, lo, hi)
                    for lo, hi in ((9, 16), (12, 40), (12, 48), (9, 40))}
    att.append(row)
report["T2_k_attenuation"] = {"b_from_fit_exclB1": b_fit,
                              "b_from_proportional_at_B16": b_prop, "table": att}

# ---- cudagraph padding diagnostic: is the B axis quantized? ------------------
padd = {}
for arm in ARMS:
    for sm in (44, 92):
        ser = []
        for B in range(1, 25):
            g = G.get((arm, sm, B), [])
            if g:
                ser.append({"B": B, "pad": pad_bs(B), "n_rep": len(g),
                            "p50_repmean_ms": math.exp(mean([math.log(c["p50"]) for c in g]))})
        padd[f"{arm}/SM{sm}"] = ser
report["T2_padding_diagnostic"] = {
    "captured_bs": CAPTURED_BS,
    "note": ("decode runs under cudagraph replay with padding enabled "
             "(disable_cuda_graph_padding=False); realized B is padded up to the next "
             "captured size, so the executed batch -- and the traffic B*c -- is a STEP "
             "function of realized B."),
    "series": padd}

# ---- job composition of every cell used in the 44->92 fit --------------------
report["T2_job_confound"] = [
    {"B": r["B"], "S": r.get("S"),
     "Ha8_jobs_lo": r["Ha8"]["jobs_lo"] if r.get("Ha8") else None,
     "Ha8_jobs_hi": r["Ha8"]["jobs_hi"] if r.get("Ha8") else None,
     "T8_jobs_lo": r["T8"]["jobs_lo"] if r.get("T8") else None,
     "T8_jobs_hi": r["T8"]["jobs_hi"] if r.get("T8") else None}
    for r in T1["44->92"] if r.get("S") is not None]

# what does H_weight (k=0) predict for R = S(hi)/S(lo)?  And the k-attenuation.
def R_pred(B_lo, B_hi, k_over_bdw=None, kappa=2):
    return dw(B_hi, kappa) / dw(B_lo, kappa)


report["T2_threshold_feasibility"] = {
    "note": ("prereg D1 threshold R>=1.5 with R = S(B_hi)/S(B_lo). Under the prereg's own "
             "proportional model (k=0) R equals dw(B_hi)/dw(B_lo) exactly, so the threshold "
             "is only reachable if the dw ratio itself clears 1.5."),
    "dw_kappa2": {str(B): dw(B, 2) for B in (1, 4, 8, 9, 12, 16, 20, 24, 32, 40, 48, 64)},
    "dw_kappa1": {str(B): dw(B, 1) for B in (1, 4, 8, 9, 12, 16, 20, 24, 32, 40, 48, 64)},
    "R_max_if_k_is_0": {
        "B 9->16 (reachable in existing data)": R_pred(9, 16),
        "B 12->16": R_pred(12, 16),
        "B 12->24": R_pred(12, 24),
        "B 12->40": R_pred(12, 40),
        "B 12->48": R_pred(12, 48),
        "B 9->24": R_pred(9, 24),
        "B 9->40": R_pred(9, 40),
        "B 9->64 (prereg's own derivation)": R_pred(9, 64),
        "B 12->64": R_pred(12, 64),
    },
    "prereg_dw_numbers_recheck": {
        "prereg_says_dw_at_B9": 24.0, "actual_dw_at_B9_kappa2": dw(9, 2),
        "actual_dw_at_B12_kappa2": dw(12, 2),
        "prereg_says_dw_at_B64": 51.0, "actual_dw_at_B64_kappa2": dw(64, 2),
        "prereg_R_derivation": 51.0 / 24.0,
        "corrected_R_derivation_B9_to_B64": dw(64, 2) / dw(9, 2),
        "corrected_R_derivation_B12_to_B64": dw(64, 2) / dw(12, 2),
    },
}

# k-attenuation table: observed R given true b and an arm-common intercept k
# ------------------------------ T4 -------------------------------------------
hist = D["snapshot_bs_hist_by_realized_sm"]
T4 = {"max_B_anywhere_in_matched_intervals": max(c["bs"] for c in rep_cells),
      "max_B_in_decode_active_snapshots": max(
          int(k) for h in hist for k in h["bs_hist"]),
      "engine_caps": {"max_running_requests": 48,
                      "Ha8_max_mamba_cache_size": 48,
                      "Ha8_kv_tokens": 121556, "T8_kv_tokens": 910419,
                      "client_concurrency_in_C2": 16,
                      "context_length": 1792},
      "B24_or_B40_present": {}}
for target in (24, 40):
    n = sum(v for h in hist for k, v in h["bs_hist"].items() if int(k) >= target)
    T4["B24_or_B40_present"][f"decode_active_snapshots_with_B>={target}"] = n
sel = []
for h in hist:
    if h["arm"] in ARMS and h["sm"] in SMS:
        bs = {int(k): v for k, v in h["bs_hist"].items()}
        tot = sum(bs.values())
        sel.append({"arm": h["arm"], "cell": h["cell"], "job": h["job"], "realized_sm": h["sm"],
                    "n_snap": tot, "max_B": max(bs), "median_B": sorted(
                        [b for b, c in bs.items() for _ in range(c)])[tot // 2],
                    "frac_B_ge_12": sum(v for b, v in bs.items() if b >= 12) / tot,
                    "n_B_ge_12": sum(v for b, v in bs.items() if b >= 12),
                    "bs_hist": h["bs_hist"]})
T4["realized_sm_conditioned_B_hist"] = sel
T4["realized_pin_sm_hist"] = [r for r in D["snapshot_sm_hist"] if r["arm"] in ARMS]
report["T4_reachability"] = T4

json.dump(report, open(OUT, "w"), indent=1, default=str)
print("wrote", OUT)

# ============================ appended: design diagnostics ===================
# (a) identification of k for candidate B grids: se(k) has lever arm xbar/sqrt(Sxx)
def design_se(B_levels, sd_rep_S, n, kappa=2):
    xs = [dw(B, kappa) for B in B_levels]
    xb = mean(xs)
    Sxx = sum((x - xb) ** 2 for x in xs)
    s = sd_rep_S / math.sqrt(n)                    # sd of S at one B level
    se_b = s / math.sqrt(Sxx)
    se_k = s * math.sqrt(1.0 / len(xs) + xb ** 2 / Sxx)
    df2 = len(B_levels) - 2
    return {"B": B_levels, "dw": [round(x, 2) for x in xs], "n": n,
            "se_b": se_b, "se_k": se_k,
            "k_halfwidth_normal": 1.959964 * se_k,
            "k_halfwidth_t_dfNb2": (t_ppf(0.975, df2) * se_k) if df2 >= 1 else None,
            "b_halfwidth_normal": 1.959964 * se_b,
            "b_halfwidth_t_dfNb2": (t_ppf(0.975, df2) * se_b) if df2 >= 1 else None}


sdS = report["T3_sd_eps"]["44->92"]["sd_rep_S_all"]
report["T5_design_identification"] = {
    "sd_rep_S_used": sdS,
    "k_magnitudes_at_issue": {"fit_all_points": -0.0574, "fit_excl_B1": +0.0661,
                              "auditor_illustration": [0.057, 0.08]},
    "grids": {str(g): {n: design_se(g, sdS, n) for n in (4, 6, 8)}
              for g in ([12, 24, 40, 48], [8, 12, 24, 40, 48], [4, 8, 16, 32, 48],
                        [2, 4, 8, 16, 32, 48], [8, 16, 24, 40], [1, 8, 16, 32, 48])}}

# (b) predictor competition: is dw doing any work beyond "monotone in B"?
pts = [r for r in T1["44->92"] if r.get("S_scoreable") and r["B"] >= 2]
comp = {}
for name, fx in (("dw_kappa2", lambda B: dw(B, 2)),
                 ("dw_kappa1", lambda B: dw(B, 1)),
                 ("B", lambda B: float(B)),
                 ("log_B", lambda B: math.log(B)),
                 ("sqrt_B", lambda B: math.sqrt(B)),
                 ("dw_padded", lambda B: dw(pad_bs(B), 2)),
                 ("cache_share_Ha8", lambda B: 100 - w_share("Ha8", B, 2))):
    xs = [fx(r["B"]) for r in pts]
    ys = [r["S"] for r in pts]
    o = wls(xs, ys)
    yb = mean(ys)
    sst = sum((y - yb) ** 2 for y in ys)
    sse = sum(r ** 2 for r in o["resid"])
    comp[name] = {"k": o["k"], "b": o["b"], "R2": 1 - sse / sst, "sse": sse,
                  "p_b": o.get("p_b"), "p_k": o.get("p_k")}
report["T5_predictor_competition"] = {"B_points": [r["B"] for r in pts], "fits": comp,
                                      "note": ("all predictors are deterministic monotone "
                                               "functions of the same design variable B; a "
                                               "good fit of dw is not evidence that weight "
                                               "share is the mechanism")}

json.dump(report, open(OUT, "w"), indent=1, default=str)
print("appended design diagnostics ->", OUT)
