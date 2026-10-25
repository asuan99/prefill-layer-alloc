#!/usr/bin/env python3
"""Cost-term harness, stage 1 -- fits and labels (pure, stdlib only).

Fits (PREREG §1-1, §1-2): relative-error weighted least squares sum(((T - T_hat) / T)^2), leave-one-out, extrapolation
test at the sealed levels, F1 -> F2 -> F3 fallback, gamma_j across D, T2 quadratic per P and the shape index q.
Labels: only those registered in PREREG §9-1 / §4' for T1, T2, T4, J-H1.  The analytic lower bounds a0, b0, c0 and the
prefill alpha, gamma of `scripts/costmodel/analytic_terms.py` (decision 27 single source) are emitted as PRIOR COLUMNS
next to the fitted values; no label reads them (an unregistered verdict would be a new rule).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import ct_spec as S
from ct_extract import median, quantile

_ENG = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ENG / "scripts" / "costmodel"))
import analytic_terms as AT  # noqa: E402


# ============================================================================ linear algebra
def _solve(A, b):
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-300:
            return None
        M[col], M[piv] = M[piv], M[col]
        for r in range(n):
            if r != col:
                f = M[r][col] / M[col][col]
                for c in range(col, n + 1):
                    M[r][c] -= f * M[col][c]
    return [M[i][n] / M[i][i] for i in range(n)]


def rel_lstsq(X, y):
    """min sum(((y - X beta) / y)^2): rows scaled by 1/y; columns scaled for conditioning."""
    if len(X) < len(X[0]):
        return None
    p = len(X[0])
    scale = [max(abs(r[j]) for r in X) or 1.0 for j in range(p)]
    Z = [[r[j] / scale[j] / yi for j in range(p)] for r, yi in zip(X, y)]
    ones = [1.0] * len(y)
    A = [[sum(z[i] * z[j] for z in Z) for j in range(p)] for i in range(p)]
    bb = [sum(z[i] * o for z, o in zip(Z, ones)) for i in range(p)]
    sol = _solve(A, bb)
    return None if sol is None else [s / sc for s, sc in zip(sol, scale)]


# ============================================================================ T1 forms
def _feat(form, bs, ctx, beta=None):
    if form == "F1":
        return [1.0, bs, bs * ctx]
    if form == "F2":
        return [1.0, bs, bs * ctx, bs * bs]
    return [1.0, bs ** beta, bs * ctx]


def _btc(points):
    """Points are (bs, ctx, T_ms) or (bs, actual ctx, T_ms, nominal c) (rev3.4): the fit reads the first three."""
    return [(p[0], p[1], p[2]) for p in points]


def _level(p, axis):
    """Extrapolation-test level of a point: the NOMINAL c (cell label) on the ctx axis, bs on the bs axis."""
    if axis == "ctx":
        return p[3] if len(p) > 3 else p[1]
    return p[0]


def fit_form(points, form):
    """points = [(bs, ctx, T_ms[, c_nominal])] -> {coef, beta, pred(bs, ctx)} or None."""
    points = _btc(points)
    if form == "F3":
        best = None
        for beta in S.F3_BETA_GRID:
            co = rel_lstsq([_feat("F3", b, c, beta) for b, c, _ in points], [t for _, _, t in points])
            if co is None:
                continue
            sse = sum(((t - sum(x * y for x, y in zip(_feat("F3", b, c, beta), co))) / t) ** 2 for b, c, t in points)
            if best is None or sse < best[0]:
                best = (sse, beta, co)
        if best is None:
            return None
        _, beta, co = best
    else:
        beta = None
        co = rel_lstsq([_feat(form, b, c) for b, c, _ in points], [t for _, _, t in points])
        if co is None:
            return None
    return {"form": form, "coef": co, "beta": beta,
            "pred": (lambda b, c, co=co, beta=beta, form=form: sum(x * y for x, y in zip(_feat(form, b, c, beta), co)))}


N_PARAMS = {"F1": 3, "F2": 4, "F3": 4}


def rel_err(pred, t):
    return abs(pred - t) / t


def loo(points, form):
    errs = []
    if len(points) - 1 < N_PARAMS[form]:
        return None
    for i, (b, c, t) in enumerate(_btc(points)):
        f = fit_form(points[:i] + points[i + 1:], form)
        if f is None:
            return None
        errs.append(rel_err(f["pred"](b, c), t))
    return errs


def extrap_test(points, form, axis):
    """Drop one level of `axis` (registered order, fallback) and predict it.  NA if fewer than 2 levels remain or the
    remaining points cannot identify the form."""
    order = S.T1_EXTRAP_CTX_ORDER if axis == "ctx" else S.T1_EXTRAP_BS_ORDER
    key = lambda p: _level(p, axis)   # noqa: E731
    levels = {key(p) for p in points}
    for lev in order:
        if lev not in levels or len(levels - {lev}) < 2:
            continue
        keep = [p for p in points if key(p) != lev]
        drop = [p for p in points if key(p) == lev]
        if len(keep) < N_PARAMS[form]:
            continue
        f = fit_form(keep, form)
        if f is None:
            continue
        return {"level": lev, "max_err": max(rel_err(f["pred"](b, c), t) for b, c, t in _btc(drop)),
                "n_dropped": len(drop)}
    return {"level": None, "max_err": None, "NA": True}


def assess(points, form):
    f = fit_form(points, form)
    if f is None:
        return None
    resid = [rel_err(f["pred"](b, c), t) for b, c, t in _btc(points)]
    lo = loo(points, form)
    ex = {ax: extrap_test(points, form, ax) for ax in ("ctx", "bs")}
    ex_vals = [e["max_err"] for e in ex.values() if e.get("max_err") is not None]
    ok = (lo is not None and median(lo) <= S.T1_LOO_MEDIAN_MAX and quantile(lo, 0.9) <= S.T1_LOO_P90_MAX
          and all(v <= S.T1_EXTRAP_MAX for v in ex_vals))
    return {"form": form, "coef": f["coef"], "beta": f["beta"], "n": len(points),
            "resid_median": median(resid), "resid_max": max(resid),
            "loo_median": median(lo) if lo else None, "loo_p90": quantile(lo, 0.9) if lo else None,
            "extrap": ex, "accepted": ok}


def fit_t1(points, mode_label):
    """F1, then F2, then F3 (§1-1); FORM_INADEQUATE_T1_D{mode} if none accepted.  Always returns the F1 coefficients
    (gamma_j input)."""
    tried = []
    for form in S.T1_FORMS:
        a = assess(points, form)
        tried.append(a)
        if a and a["accepted"]:
            return {"chosen": a, "tried": tried, "label": None, "F1": tried[0]}
    return {"chosen": None, "tried": tried, "label": f"FORM_INADEQUATE_T1_{mode_label}", "F1": tried[0]}


def gamma_fit(coef_by_D, anchor):
    """kappa_j(D) = kappa_j(108) (108/D)^gamma_j: through-origin LS of ln(kappa_j(D)/kappa_j(anchor)) on ln(108/D)
    over the realized D axis (descriptive, §1-1).  coef_by_D = {D_realized: (a, b, c)}."""
    out = {}
    for j, name in enumerate("abc"):
        k0 = anchor[j]
        xs, ys = [], []
        for d, co in coef_by_D.items():
            if co[j] > 0 and k0 > 0:
                xs.append(math.log(S.TOTAL_SM / d))
                ys.append(math.log(co[j] / k0))
        den = sum(x * x for x in xs)
        out[name] = (sum(x * y for x, y in zip(xs, ys)) / den) if den > 0 else None
    return out


# ============================================================================ T2
def fit_t2(points):
    """points = [(L, T_ms)] -> quadratic u + v L + w L^2 (relative LS), leave-one-internal-L-out."""
    X = [[1.0, L, L * L] for L, _ in points]
    co = rel_lstsq(X, [t for _, t in points])
    if co is None:
        return None
    pred = lambda L, co=co: co[0] + co[1] * L + co[2] * L * L   # noqa: E731
    Ls = sorted(L for L, _ in points)
    internal = Ls[1:-1]
    errs = {}
    for L0 in internal:
        keep = [(L, t) for L, t in points if L != L0]
        c2 = rel_lstsq([[1.0, L, L * L] for L, _ in keep], [t for _, t in keep])
        t0 = next(t for L, t in points if L == L0)
        errs[L0] = None if c2 is None else rel_err(c2[0] + c2[1] * L0 + c2[2] * L0 * L0, t0)
    vals = [v for v in errs.values() if v is not None]
    return {"coef": co, "pred": pred, "loo_internal": errs, "loo_max": max(vals) if vals else None,
            "accepted": bool(vals) and max(vals) <= S.T2_LOO_MAX}


def q_index(points, fit=None):
    """q = [T(65536)/65536] / [T(8800)/8800]; measured points first, fitted quadratic when a point is missing."""
    d = dict(points)
    src = []
    def val(L):
        if L in d:
            src.append(f"measured {L}")
            return d[L]
        if fit is None:
            return None
        src.append(f"fit {L}")
        return fit["pred"](L)
    hi, lo = val(S.Q_L_HI), val(S.Q_L_LO)
    if hi is None or lo is None:
        return {"q": None, "source": src}
    return {"q": (hi / S.Q_L_HI) / (lo / S.Q_L_LO), "source": src, "from_fit": any(s.startswith("fit") for s in src)}


def q_label(qs):
    """qs = per-boot q values (A-15): interval [min, max]."""
    qs = [q for q in qs if q is not None]
    if not qs:
        return None
    lo, hi = min(qs), max(qs)
    if lo >= S.Q_THRESHOLD:
        return "PREFILL_SUPERLINEAR_SCOPED"
    if hi < S.Q_THRESHOLD:
        return "PREFILL_NEAR_LINEAR_SCOPED"
    return "Q_UNRESOLVED"


def j1_slope_label(points):
    pts = [(L, t) for L, t in points if L <= S.J1_SLOPE_L_MAX]
    if len(pts) < 2:
        return None, None
    n = len(pts)
    mx, my = sum(L for L, _ in pts) / n, sum(t for _, t in pts) / n
    sxx = sum((L - mx) ** 2 for L, _ in pts)
    slope = sum((L - mx) * (t - my) for L, t in pts) / sxx
    dev = abs(slope / S.J1_SLOPE_MS_PER_TOKEN - 1.0)
    return slope, ("T2_J1_SLOPE_DEVIATES" if dev > S.J1_SLOPE_DEV else None)


# ============================================================================ §1-7 agreement
def agree_key(w):
    """Audit r1 MAJOR-1: agreement is judged per (model, mode, cap, tp, bs) -- a TP2 boot, an N=1 anchor, a cap-48 and a
    cap-128 boot of the same (model, mode) are different servers and are never pooled."""
    return (w["model"], w["mode"], w.get("cap"), w.get("tp", 1), w["bs"])


def agreement_labels(windows):
    """windows: [{model, mode, cap, tp, bs, S_eng_ms, client_gap_ms}] (VALID engine windows).  Per agree_key:
    AGREE iff every window has |ln(client/S_eng)| <= ln 1.03 (A-12).  Missing S_eng -> NOT_COLLECTED."""
    groups = {}
    for w in windows:
        groups.setdefault(agree_key(w), []).append(w)
    out = {}
    for key, ws in sorted(groups.items(), key=lambda kv: str(kv[0])):
        if any(w.get("S_eng_ms") is None for w in ws):
            out[key] = {"label": "NOT_COLLECTED", "ln": None}
            continue
        lns = [math.log(w["client_gap_ms"] / w["S_eng_ms"]) if w.get("client_gap_ms") else float("inf") for w in ws]
        ok = all(abs(x) <= S.AGREE_LN for x in lns)
        out[key] = {"label": f"CLIENT_ENGINE_{'AGREE' if ok else 'MISMATCH'}_bs{key[4]}", "ln": lns}
    return out


def client_secondary_table(cells, agree):
    """T1 secondary estimand (client gap) per (boot key, bs, c), with MISMATCH (cap, tp, bs) EXCLUDED (§1-7).  Value =
    median over the cell's VALID windows of client_gap_ms (audit r1 MAJOR-1: the old table kept the last window and pooled
    boots of different cap / tp).  cells carry boot_key (= ct_runner._tkey of the boot)."""
    acc = {}
    for c in cells:
        lab = agree.get(agree_key(c), {}).get("label", "")
        if "MISMATCH" in lab:
            continue
        acc.setdefault((c.get("boot_key", c["model"] + "|" + c["mode"]), c["bs"], c["c"]), []).append((c, lab))
    out = {}
    for key, items in acc.items():
        vals = [c["client_gap_ms"] for c, _ in items if c.get("client_gap_ms") is not None]
        tag = "CLIENT_ONLY_UNVERIFIED" if items[0][1] == "NOT_COLLECTED" else None
        out[key] = {"client_gap_ms": median(vals) if vals else None, "n_windows": len(vals), "tag": tag}
    return out


# ============================================================================ cap invariance (J-A1)
def cap_invariance(cap48, cap128):
    """{(bs, c): S_eng_ms} at cap 48 vs cap 128 (D=44)."""
    common = sorted(set(cap48) & set(cap128))
    if len(common) != len(S.CAP_INV_CELLS):
        return {"label": None, "detail": f"{len(common)} common cells"}
    lns = {k: math.log(cap48[k] / cap128[k]) for k in common}
    return {"label": "CAP_INVARIANT" if all(abs(v) <= S.CAP_INV_LN for v in lns.values()) else "CAP_DEPENDENT",
            "ln": {f"{b}x{c}": v for (b, c), v in lns.items()}}


# ============================================================================ OFF diagnostic (§2-2' (2))
def instrument_effect(on, off, registered=None):
    """on/off = {key: client value}.  Descriptive; NEVER modifies T1-T4 (decision 22).  Audit r1 m1 (A-49): the verdict covers
    only the keys present in BOTH; `registered` (the OFF boot's registered keys) lets the output say how many of them were
    compared and which are missing, so INSTR_EFFECT_SMALL on 4 of 5 registered cells is not read as 5 of 5."""
    common = sorted(set(on) & set(off), key=str)
    lns = {str(k): math.log(on[k] / off[k]) for k in common}
    reg = sorted(set(registered), key=str) if registered is not None else sorted(off, key=str)
    out = {"cells_compared": len(common), "cells_registered": len(reg),
           "cells_missing": [str(k) for k in reg if k not in common]}
    if not lns:
        return dict(out, label=None, ln={})
    lab = "INSTR_EFFECT_SMALL" if all(abs(v) <= S.INSTR_SMALL_LN for v in lns.values()) else "INSTR_EFFECT_DISCLOSED"
    return dict(out, label=lab, ln=lns,
                scope=f"{lab} on {len(common)} of {len(reg)} registered cells" + (" (PARTIAL)" if len(common) < len(reg) else ""))


# ============================================================================ T4 labels (§1-4)
def t4_labels(fill, pred):
    """fill from ct_extract.t4_fill, pred from cmem_predictions (sealed before the fill)."""
    out = {"L": fill["L"], "cap": pred["cap"], "n_real": fill["n_real"], "n_A": pred["n_A"], "n_B": pred["n_B"],
           "labels": list(fill.get("labels", []))}
    if "FILL_INCOMPLETE" in out["labels"] or not fill["n_real"]:
        out["excluded"] = True
        return out
    rb, ra = fill["n_real"] / pred["n_B"], fill["n_real"] / pred["n_A"]
    out.update(ratio_B=rb, ratio_A=ra, excluded=False)
    out["labels"].append("CMEM_ACCT_HELD" if S.CMEM_HELD_LO <= rb <= S.CMEM_HELD_HI else "CMEM_ACCT_MISS")
    if ra < S.CMEM_ORIG_OVER:
        out["labels"].append("CMEM_ORIG_OVERPREDICTS")
    return out


def c_mem_drift(pools):
    """pools = {boot_id: pool_bytes} of the T4 boots: every pool within 2 % of their median."""
    m = median(list(pools.values()))
    bad = {b: p / m - 1 for b, p in pools.items() if abs(p / m - 1) > S.C_MEM_DRIFT_TOL}
    return {"label": "C_MEM_DRIFT" if bad else None, "median": m, "outliers": bad}


# ============================================================================ J-H1 (§4')
def w_indep(dummy, real):
    """dummy/real = {key: value} for one term (T1 S_eng per common cell, or T2 TTFT per L).  ln is dummy/real."""
    common = sorted(set(dummy) & set(real), key=str)
    if not common:
        return {"label": None, "ln": {}}
    lns = {str(k): math.log(dummy[k] / real[k]) for k in common}
    return {"label": "WEIGHT_INDEP_HELD" if all(abs(v) <= S.W_INDEP_LN for v in lns.values()) else "WEIGHT_DEPENDENT",
            "ln": lns}


def analytic_sk(n_attn):
    a = AT.nano9b_v2(n_attn, S.VARIANT_MAMBA[n_attn])
    return AT.state_bytes_per_request(a), AT.kv_bytes_per_token(a)


def compo_mem(n_attn, s_banner, k_banner, slots):
    """P1: banner s, k within +-2 % of the analytic table; s = 0 tolerates <= 0.01 GiB over all slots."""
    s0, k0 = analytic_sk(n_attn)
    ok_k = k_banner is not None and abs(k_banner / k0 - 1) <= S.COMPO_MEM_TOL
    if s0 == 0:
        ok_s = s_banner is not None and abs(s_banner) <= (0.01 * 1024 ** 3 / slots if slots else 0.0)
    else:
        ok_s = s_banner is not None and abs(s_banner / s0 - 1) <= S.COMPO_MEM_TOL
    # only the HELD label is registered (§4' P1); a miss is reported as held=False with no invented label
    return {"variant": f"a{n_attn}", "s_pred": s0, "k_pred": k0, "s_banner": s_banner, "k_banner": k_banner,
            "held": ok_s and ok_k, "label": "COMPO_MEM_HELD" if (ok_s and ok_k) else None}


def cmem_variant(n_attn, labelled_fills):
    use = [f for f in labelled_fills if not f.get("excluded")]
    if not use:
        return {"variant": f"a{n_attn}", "label": None}
    ok = all("CMEM_ACCT_HELD" in f["labels"] for f in use)
    return {"variant": f"a{n_attn}", "held": ok, "label": f"CMEM_ACCT_HELD_a{n_attn}" if ok else None,
            "n_fills": len(use)}


def ctx_slope_prop(c_by_n, wi_t1_label):
    """P3: c(n)/c(4) within (n/4) +- 25 %, every n != 4.  Not evaluated if T1 W-INDEP failed."""
    if wi_t1_label != "WEIGHT_INDEP_HELD":
        return {"label": None, "status": "NOT_EVALUATED (T1 W-INDEP not held)"}
    if 4 not in c_by_n:
        return {"label": None, "status": "no a4"}
    ratios = {n: c_by_n[n] / c_by_n[4] for n in c_by_n if n != 4}
    ok = all(abs(r / (n / 4) - 1) <= S.CTX_SLOPE_TOL for n, r in ratios.items())
    return {"label": "CTX_SLOPE_PROP_HELD" if ok else "CTX_SLOPE_PROP_MISS", "ratios": ratios}


def q_monotone(q_by_n, wi_t2_label):
    if wi_t2_label != "WEIGHT_INDEP_HELD":
        return {"label": None, "status": "NOT_EVALUATED (T2 W-INDEP not held)"}
    ns = sorted(q_by_n)
    qs = [q_by_n[n] for n in ns]
    if any(q is None for q in qs):
        return {"label": None, "status": "missing q"}
    ok = all(b > a for a, b in zip(qs, qs[1:]))
    return {"label": "PREFILL_Q_MONOTONE" if ok else "PREFILL_Q_NOT_MONOTONE", "q": dict(zip(ns, qs))}


# ============================================================================ priors (comparison columns only)
def priors(n_attn=4, bw=S.HBM_BYTES_PER_S):
    a = AT.nano9b_v2(n_attn, S.VARIANT_MAMBA[n_attn])
    dc = AT.decode_coefficients(a, bw)
    pf = AT.prefill_flops(a, S.Q_L_HI)
    alpha, gamma = pf["linear_per_token"], pf["quad_coeff"]
    q_flop = (alpha + gamma * S.Q_L_HI) / (alpha + gamma * S.Q_L_LO)
    return {"source": f"analytic_terms {AT.VERSION}, BW {bw:.4g} B/s [derived, comparison column, no verdict]",
            "a0_ms": dc["a0_s"] * 1e3, "b0_ms_per_req": dc["b0_s_per_req"] * 1e3,
            "c0_ms_per_req_token": dc["c0_s_per_req_token"] * 1e3, "alpha_flop_per_token": alpha,
            "gamma_flop_per_token2": gamma, "q_flop_prior": q_flop}
