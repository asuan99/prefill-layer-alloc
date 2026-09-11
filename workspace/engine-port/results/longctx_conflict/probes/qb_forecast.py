#!/usr/bin/env python3
"""Q-B' -- standardized composition decomposition of the paired mean decode-ITL
difference at common lambda (path cf), computed from EXISTING raw data.
EXPLORATORY (definitions fixed AFTER seeing the data; audit verdict
audit_qb_rules_2026-09-11/VERDICT.md, GO-with-caveats).  File/prereg name
`PREREG_QB_LOOPGAIN_2026-09-11.md` is kept only as an identifier; the original
Q-B question is exhausted by the Little identity (verdict sec 3(c), G-eta).

    PREREG_QB_LOOPGAIN_2026-09-11.md  sec 3 (Stage 0)  /  sec 2 (definitions)

GPU 0.  This file is a CALCULATOR.  It produces NO performance verdict, NO arm
ranking, NO argmin/argmax/envelope, NO "which arm is best", NO policy or
controller statement.  Those are STRUCTURALLY ABSENT and must not be added.

------------------------------------------------------------------------------
WHAT IT COMPUTES
------------------------------------------------------------------------------
(A) TOKEN TIMELINE (new client-side instrument, no telemetry).
    bench_serving.py:941-948 draws exactly one np.random.exponential(1/rate)
    per emitted request after `np.random.seed(seed)` (:1706); `span_factor.py`
    documents and validates this.  Hence request k's send time is
        a_0 = 0,   a_k = sum_{m<=k} Exp_m(1/rate)          (seed-reconstructed)
    and every client timestamp of request k follows from its own record:
        F_k = a_k + ttft_k  (first token),  t_kj = F_k + cumsum(itls_k)[:j].
    VALIDATION (a real check, not an identity): tokens of two requests that are
    in decode at the same time are produced by the SAME engine step and must
    coincide on the client clock.  The fraction of such "partnered" tokens whose
    nearest partner token lies within 2 ms / 5 ms is reported for all cells.

(B) TOKEN COVARIATES (registered, sec 2.2 of the prereg).
    For decode token j>=1 of request i, launch time  tau_ij := t_{i,j-1}.
      e_ij := 1[ tau_ij in U_{k != i} [a_k, F_k) ]   -- another request is in
              pre-first-token state (queue+prefill) when the step launches
              (mechanism: multiplexing_mixin.py:952-953, CONSENSUS sec 3 item 26:
              the split applies only while prefill is in flight)
      b_ij := #{k : F_k <= tau_ij < last_k}          -- decode population at launch
    Neither covariate uses the token's OWN itl value.

(C) ESTIMAND (prereg sec 1.3).  Statistic = pooled decode-token MEAN ITL
    (axis P; the only additive functional -- median/p95/trim10 do not
    decompose).  Matched pair (D < D', same rung, shape, round, seed = same
    arrival trace).  pi_c(e,b) = token share of arm c, mu_s(e,b) = mean ITL of
    arm s in cell (e,b).  2x2 counterfactual table  M[s,c] = sum pi_c * mu_s:
        S@c = M[D',c] - M[D,c]   supply effect (composition held at c)
        K@s = M[s,D'] - M[s,D]   composition effect (service curve held at s)
        I   = S@D' - S@D         interaction
        Delta = S@D' + K@D = S@D + K@D'                   (ACCOUNTING IDENTITY)
    PRIMARY path = (K@D, S@D'): it only needs M[D,D'], i.e. mu_D on D''s
    (e,b) support, which the higher-D arm's smaller populations keep inside D's
    support (off-support mass reported per counterfactual).  SECONDARY
    (extrapolation-dependent): S@D, K@D', I.  Placebo: the e=0 parts of S
    (both arms run at 108 SM there).  Also SECONDARY, both reported, neither
    chosen: the exposure/population split of the composition effect under the
    two hierarchies P(e)P(b|e) and P(b)P(e|b), with midpoint weights.
    Off-support (e,b) cells: missing mu filled with the other arm's mu
    (dmu := 0 there); alternative b-linear fill is a sensitivity row.
    !! The exact sums are ACCOUNTING IDENTITIES and are NOT counted as checks.

(D) CI -- inherited VERBATIM from qa_analyze.rung_ci (rev7 sec 3 + rev8 A2 +
    rev9 P3): 2-seed rungs sqrt(sb^2/4 + ss^2/8 + sreq^2/8), 1-seed rungs
    sqrt(sc^2/4 + sreq^2/4) with the common-trace diagnostic sqrt(sc^2/4+sreq^2)
    alongside, t_3.  sigma_req = paired request bootstrap, B = 10000, seed = 1
    (rev9 sec 1(a)(b)), token covariates held FIXED at their full-timeline values
    (a resample cannot rebuild a timeline).

(E) REVERSAL SELF-TESTS: every free surface of (A)-(C) pushed to its
    alternatives; reports max |change| and sign changes per component.
(F) LOAD COVARIATE (gate #117 / #138): R^2 of replicate components on the
    realised arrival-span factor.
(G) Cross-check vs qa_regret_result.json: our Delta = Q-A Delta_mean with the
    orientation flipped (independent loader path -> a real check of the loader).

Usage:  python3 qb_forecast.py [--boot 10000] [--out qb_forecast_result.json]
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# gate #9: do not reimplement the canonical instruments -- import them.
from qa_analyze import rung_ci, t_ppf, variance_components, ITL_SOLO_MS, MU_P_REF  # noqa: E402
from span_factor import span_factor  # noqa: E402

BOOT_N = 10000
BOOT_SEED = 1
ARMS = (16, 44, 54, 92)

# =============================================================================
# Loading & timeline
# =============================================================================
def load(path):
    return json.loads(Path(path).read_text().strip().split("\n")[0])


def arrivals(seed, rate, n, rescale_to=None):
    rs = np.random.RandomState(seed)
    e = rs.exponential(1.0 / rate, size=n - 1)
    a = np.concatenate([[0.0], np.cumsum(e)])
    if rescale_to is not None and a[-1] > 0:
        a = a * (rescale_to / a[-1])
    return a


def timeline(d, seed, arrival="seed_raw"):
    assert all(e == "" for e in d["errors"]), "failed request present"
    n = len(d["ttfts"])
    ttft = np.asarray(d["ttfts"], float)
    itls = [np.asarray(x, float) for x in d["itls"]]
    resc = None
    if arrival == "span_rescaled":
        resc = d["duration"] - (ttft[-1] + itls[-1].sum())
    a = arrivals(seed, d["request_rate"], n, resc)
    F = a + ttft
    toks = [F[k] + np.concatenate([[0.0], np.cumsum(itls[k])]) for k in range(n)]
    last = np.array([t[-1] for t in toks])
    return {"a": a, "F": F, "ttft": ttft, "itls": itls, "toks": toks, "last": last,
            "n": n, "duration": d["duration"], "rate": d["request_rate"],
            "out_len": d["random_output_len"]}


def alignment(tl):
    """Real check (A): partnered-token coincidence on the client clock."""
    toks, F, last = tl["toks"], tl["F"], tl["last"]
    res = []
    for i, t in enumerate(toks):
        x = t[1:]
        part = (F[None, :] < x[:, None] - 5e-4) & (last[None, :] > x[:, None] + 5e-4)
        part[:, i] = False
        for jx in np.where(part.any(axis=1))[0]:
            best = np.inf
            for k in np.where(part[jx])[0]:
                tk = toks[k]
                j = np.searchsorted(tk, x[jx])
                for jj in (j - 1, j):
                    if 0 <= jj < tk.size:
                        best = min(best, abs(tk[jj] - x[jx]))
            res.append(best)
    r = np.asarray(res) * 1000.0
    return {"n_partnered": int(r.size),
            "frac_lt_1ms": float((r < 1).mean()) if r.size else None,
            "frac_lt_2ms": float((r < 2).mean()) if r.size else None,
            "frac_lt_5ms": float((r < 5).mean()) if r.size else None}


def covariates(tl, point="launch", pre_start_lag=0.0, exposure_rule="point", batch="exact"):
    """Registered: point='launch', pre_start_lag=0, exposure_rule='point', batch='exact'.
    batch='none' collapses b to 1 (e-only conditioning) -- a DISCLOSURE variant
    only (verdict A2 / D8), outside the registered estimand."""
    a, F, last, toks, itls = tl["a"], tl["F"], tl["last"], tl["toks"], tl["itls"]
    n = tl["n"]
    lo = a + pre_start_lag
    per = []
    for i in range(n):
        t = toks[i]
        tau = t[:-1] if point == "launch" else (t[1:] if point == "emission" else 0.5 * (t[:-1] + t[1:]))
        oth = np.arange(n) != i
        if exposure_rule == "point":
            e = ((tau[:, None] >= lo[oth][None, :]) & (tau[:, None] < F[oth][None, :])).any(axis=1)
        else:  # 'overlap_half': fraction of [t_{j-1}, t_j] covered by the pre-union >= 1/2
            s0, s1 = t[:-1], t[1:]
            iv = sorted(zip(lo[oth], F[oth]))
            merged = []
            for x0, x1 in iv:
                if x1 <= x0:
                    continue
                if merged and x0 <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], x1)
                else:
                    merged.append([x0, x1])
            cov = np.zeros(s0.size)
            for x0, x1 in merged:
                cov += np.clip(np.minimum(s1, x1) - np.maximum(s0, x0), 0, None)
            e = cov / np.maximum(s1 - s0, 1e-12) >= 0.5
        bpt = t[:-1] if point != "emission" else t[1:]
        b = ((F[None, :] <= bpt[:, None]) & (last[None, :] > bpt[:, None])).sum(axis=1)
        b = np.maximum(b, 1)
        if batch == "none":
            b = np.ones_like(b)
        per.append((itls[i] * 1000.0, e.astype(int), b.astype(int)))
    return per


def class_matrix(per, bmax):
    n = len(per)
    C = np.zeros((n, 2 * bmax))
    S = np.zeros((n, 2 * bmax))
    for i, (itl, e, b) in enumerate(per):
        k = e * bmax + (np.minimum(b, bmax) - 1)
        np.add.at(C[i], k, 1.0)
        np.add.at(S[i], k, itl)
    return C, S


# =============================================================================
# Decomposition (vectorised over leading axes)
# =============================================================================
def decompose(Ca, Sa, Cb, Sb, bmax, hierarchy="exposure_first", offsupport="composition",
              weights="midpoint"):
    sh = Ca.shape[:-1]
    Ca = Ca.reshape(sh + (2, bmax)); Sa = Sa.reshape(sh + (2, bmax))
    Cb = Cb.reshape(sh + (2, bmax)); Sb = Sb.reshape(sh + (2, bmax))
    Na = Ca.sum(axis=(-1, -2), keepdims=True)
    Nb = Cb.sum(axis=(-1, -2), keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        mu_a = np.where(Ca > 0, Sa / np.where(Ca > 0, Ca, 1.0), np.nan)
        mu_b = np.where(Cb > 0, Sb / np.where(Cb > 0, Cb, 1.0), np.nan)
    if offsupport == "linear_in_b":
        mu_a = _fill_linear(mu_a, Ca); mu_b = _fill_linear(mu_b, Cb)
    mu_a_f = np.where(np.isnan(mu_a), mu_b, mu_a)
    mu_b_f = np.where(np.isnan(mu_b), mu_a, mu_b)
    both = np.isnan(mu_a_f)
    mu_a_f = np.where(both, 0.0, mu_a_f); mu_b_f = np.where(both, 0.0, mu_b_f)
    pi_a = Ca / Na; pi_b = Cb / Nb
    if weights == "midpoint":
        wa, wb = 0.5, 0.5
    elif weights == "D_weights":       # composition at D, rates at D'
        wa, wb = 1.0, 0.0
    else:                              # "Dp_weights"
        wa, wb = 0.0, 1.0
    # rate part uses composition weights (wa*pi_a + wb*pi_b); composition part
    # uses rates at (1-wa)*mu_a + (1-wb)*mu_b so the total stays exact.
    pibar = wa * pi_a + wb * pi_b
    mubar = (1 - wa) * mu_a_f + (1 - wb) * mu_b_f
    dmu = mu_b_f - mu_a_f
    C_sm = pibar * dmu
    C_sm0 = C_sm[..., 0, :].sum(-1); C_sm1 = C_sm[..., 1, :].sum(-1)
    comp = ((pi_b - pi_a) * mubar)            # total composition, per (e,b)
    # --- hierarchy E-first: P(e) * P(b|e)   (causal order e -> b)
    Pe_a = pi_a.sum(-1, keepdims=True); Pe_b = pi_b.sum(-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        Pbe_a = np.where(Pe_a > 0, pi_a / np.where(Pe_a > 0, Pe_a, 1), 0.0)
        Pbe_b = np.where(Pe_b > 0, pi_b / np.where(Pe_b > 0, Pe_b, 1), 0.0)
    Pbe_bar = 0.5 * (Pbe_a + Pbe_b); Pe_bar = 0.5 * (Pe_a + Pe_b)
    C_exp_e = ((Pe_b - Pe_a) * (Pbe_bar * mubar).sum(-1, keepdims=True)).sum(axis=(-1, -2))
    C_pop_e = (Pe_bar * ((Pbe_b - Pbe_a) * mubar)).sum(axis=(-1, -2))
    # --- hierarchy B-first: P(b) * P(e|b)
    Pb_a = pi_a.sum(-2, keepdims=True); Pb_b = pi_b.sum(-2, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        Peb_a = np.where(Pb_a > 0, pi_a / np.where(Pb_a > 0, Pb_a, 1), 0.0)
        Peb_b = np.where(Pb_b > 0, pi_b / np.where(Pb_b > 0, Pb_b, 1), 0.0)
    Peb_bar = 0.5 * (Peb_a + Peb_b); Pb_bar = 0.5 * (Pb_a + Pb_b)
    C_pop_b = ((Pb_b - Pb_a) * (Peb_bar * mubar).sum(-2, keepdims=True)).sum(axis=(-1, -2))
    C_exp_b = (Pb_bar * ((Peb_b - Peb_a) * mubar)).sum(axis=(-1, -2))
    if hierarchy == "exposure_first":
        C_exp, C_pop = C_exp_e, C_pop_e
    else:
        C_exp, C_pop = C_exp_b, C_pop_b
    resid = comp.sum(axis=(-1, -2)) - C_exp - C_pop   # 0 for midpoint (reported)
    # ---- 2x2 counterfactual table (PRIMARY): M[s,c] = sum pi_c * mu_s
    #      s = service-curve source, c = (e,b)-composition source; a = D, b = D'
    M_aa = (pi_a * mu_a_f).sum(axis=(-1, -2)); M_bb = (pi_b * mu_b_f).sum(axis=(-1, -2))
    M_ba = (pi_a * mu_b_f).sum(axis=(-1, -2)); M_ab = (pi_b * mu_a_f).sum(axis=(-1, -2))
    S_atD = M_ba - M_aa;  S_atDp = M_bb - M_ab       # supply effect at D / D' composition
    K_atD = M_ab - M_aa;  K_atDp = M_bb - M_ba       # composition effect at D / D' service
    S1_atD = (pi_a * (mu_b_f - mu_a_f))[..., 1, :].sum(-1)
    S1_atDp = (pi_b * (mu_b_f - mu_a_f))[..., 1, :].sum(-1)
    S0_atD = (pi_a * (mu_b_f - mu_a_f))[..., 0, :].sum(-1)   # PLACEBO parts
    S0_atDp = (pi_b * (mu_b_f - mu_a_f))[..., 0, :].sum(-1)
    m_a = Sa.sum(axis=(-1, -2)) / Na[..., 0, 0]
    m_b = Sb.sum(axis=(-1, -2)) / Nb[..., 0, 0]
    offsup = (((Ca > 0) != (Cb > 0)) * (0.5 * (pi_a + pi_b))).sum(axis=(-1, -2))
    # support needed by each counterfactual: M[D',D] needs mu_D' on D's support,
    # M[D,D'] needs mu_D on D''s support
    offsup_M_Dp_D = ((Ca > 0) & (Cb == 0)).astype(float).__mul__(pi_a).sum(axis=(-1, -2))
    offsup_M_D_Dp = ((Cb > 0) & (Ca == 0)).astype(float).__mul__(pi_b).sum(axis=(-1, -2))
    return {"delta_mean": m_b - m_a, "C_exp": C_exp, "C_pop": C_pop, "C_sm1": C_sm1,
            "C_sm0_placebo": C_sm0, "C_comp": comp.sum(axis=(-1, -2)),
            "C_exp_Efirst": C_exp_e, "C_pop_Efirst": C_pop_e,
            "C_exp_Bfirst": C_exp_b, "C_pop_Bfirst": C_pop_b,
            "M_DD": M_aa, "M_DpDp": M_bb, "M_serviceDp_compD": M_ba, "M_serviceD_compDp": M_ab,
            "S_atD": S_atD, "S_atDp": S_atDp, "K_atD": K_atD, "K_atDp": K_atDp,
            "I_interaction": S_atDp - S_atD,
            "S1_atD": S1_atD, "S1_atDp": S1_atDp, "S0_atD_placebo": S0_atD, "S0_atDp_placebo": S0_atDp,
            "composition_interaction_resid": resid,
            "offsupport_mass": offsup, "offsupport_M_serviceDp_compD": offsup_M_Dp_D,
            "offsupport_M_serviceD_compDp": offsup_M_D_Dp,
            "E_D": pi_a[..., 1, :].sum(-1), "E_Dp": pi_b[..., 1, :].sum(-1),
            "m_D": m_a, "m_Dp": m_b}


def _fill_linear(mu, C):
    """Alternative off-support rule: within (arm, e), fill missing b by a
    count-weighted straight line in b (only for the point estimate path)."""
    out = mu.copy()
    b = np.arange(1, mu.shape[-1] + 1, dtype=float)
    it = np.ndindex(mu.shape[:-1])
    for idx in it:
        m, c = mu[idx], C[idx]
        ok = ~np.isnan(m)
        if ok.sum() >= 2:
            w = c[ok]
            A = np.vstack([np.ones(ok.sum()), b[ok]]).T
            coef = np.linalg.lstsq(A * np.sqrt(w)[:, None], m[ok] * np.sqrt(w), rcond=None)[0]
            out[idx] = np.where(ok, m, coef[0] + coef[1] * b)
        elif ok.sum() == 1:
            out[idx] = np.where(ok, m, m[ok][0])
    return out


COMPONENTS = ("delta_mean", "S_atD", "S_atDp", "K_atD", "K_atDp", "I_interaction",
              "S1_atD", "S1_atDp", "S0_atD_placebo", "S0_atDp_placebo",
              "C_sm1", "C_comp", "C_sm0_placebo",
              "C_exp_Efirst", "C_pop_Efirst", "C_exp_Bfirst", "C_pop_Bfirst")


# =============================================================================
# Campaign indexing
# =============================================================================
def qa_cells():
    cells = {}
    for f in sorted(glob.glob(str(HERE / "qa_r*_9065*" / "bench_*.jsonl"))):
        m = re.search(r"bench_r(\d)_d(\d+)_s([AB])_l([\d.]+)_seed(\d+)", Path(f).name)
        r, D, s, lam, seed = int(m.group(1)), int(m.group(2)), m.group(3), float(m.group(4)), int(m.group(5))
        cells[("QA", lam, s, r, seed, D)] = f
    return cells


def p7_cells():
    cells = {}
    for f in sorted(glob.glob(str(HERE / "p7_905994" / "bench_r*_d*_s*.jsonl"))):
        m = re.search(r"bench_r(\d)_d(\d+)_s(\d+)\.jsonl", Path(f).name)
        r, D, seed = int(m.group(1)), int(m.group(2)), int(m.group(3))
        cells[("P7", 0.491, "A", r, seed, D)] = f
    return cells


def c_cells():
    cells = {}
    for f in sorted(glob.glob(str(HERE / "c_905835" / "bench_d*_r*_o*.jsonl"))):
        m = re.search(r"bench_d(\d+)_r(\d)_o(\d+)\.jsonl", Path(f).name)
        D, rung, out = int(m.group(1)), int(m.group(2)), int(m.group(3))
        cells[("C", rung, "A" if out == 96 else "B", 0, 41, D)] = f
    return cells


def pairs_of(cells):
    p = collections.defaultdict(list)
    for (src, lam, s, r, seed, D) in cells:
        if src == "C":
            continue          # probe C rungs are arm-relative: NO matched pairs
        for D2 in ARMS:
            if D2 > D and (src, lam, s, r, seed, D2) in cells:
                p[(src, lam, s, D, D2)].append((r, seed))
    return p


# =============================================================================
# Main pieces
# =============================================================================
_TL, _COV = {}, {}


def get_tl(cells, key, arrival="seed_raw"):
    k = (key, arrival)
    if k not in _TL:
        _TL[k] = timeline(load(cells[key]), key[4], arrival)
    return _TL[k]


def get_cov(cells, key, **kw):
    arrival = kw.pop("arrival", "seed_raw")
    k = (key, arrival, tuple(sorted(kw.items())))
    if k not in _COV:
        _COV[k] = covariates(get_tl(cells, key, arrival), **kw)
    return _COV[k]


def cell_summary(cells, key):
    tl = get_tl(cells, key)
    per = get_cov(cells, key)
    itl = np.concatenate([p[0] for p in per]); e = np.concatenate([p[1] for p in per])
    b = np.concatenate([p[2] for p in per])
    dur, n = tl["duration"], tl["n"]
    L_dec = sum(x.sum() for x in tl["itls"]) / dur
    X = n / dur
    # prefill-busy time fraction over the arrival window (client side)
    iv = sorted(zip(tl["a"], tl["F"])); U = 0.0; cur = None
    for x0, x1 in iv:
        if cur is None or x0 > cur[1]:
            if cur is not None:
                U += cur[1] - cur[0]
            cur = [x0, x1]
        else:
            cur[1] = max(cur[1], x1)
    U += cur[1] - cur[0]
    src, lam, s, r, seed, D = key
    sf = span_factor(seed, n)
    q = lambda v, p: float(np.percentile(v, p)) if v.size else None
    return {"key": list(key), "file": Path(cells[key]).name, "n_requests": n,
            "n_tokens": int(itl.size), "E_token_exposure": float(e.mean()),
            "prefill_busy_frac_U": U / dur,
            "mean_itl_ms": float(itl.mean()),
            "mu_unexposed_ms": float(itl[e == 0].mean()) if (e == 0).any() else None,
            "mu_exposed_ms": float(itl[e == 1].mean()) if (e == 1).any() else None,
            "p05_p50_p95_unexposed": [q(itl[e == 0], p) for p in (5, 50, 95)],
            "p05_p50_p95_exposed": [q(itl[e == 1], p) for p in (5, 50, 95)],
            "bbar_token_unexposed": float(b[e == 0].mean()) if (e == 0).any() else None,
            "bbar_token_exposed": float(b[e == 1].mean()) if (e == 1).any() else None,
            "L_decode_exact": L_dec, "X": X,
            "little_identity_L_over_X_outm1_mean": L_dec / (X * (tl["out_len"] - 1) * itl.mean() / 1000.0),
            "span_factor_pred": sf,
            "rho_realised": (lam / (sf * MU_P_REF[D])) if src == "QA" else None}


def pair_decomp(cells, pk, reps, boot_n, **kw):
    src, lam, s, D, D2 = pk
    by_rep = {}
    sreq = {c: [] for c in COMPONENTS}
    extra = {}
    for (r, seed) in sorted(reps):
        ka = (src, lam, s, r, seed, D); kb = (src, lam, s, r, seed, D2)
        dk = dict(kw)
        hier = dk.pop("hierarchy", "exposure_first"); offs = dk.pop("offsupport", "composition")
        wts = dk.pop("weights", "midpoint")
        pa = get_cov(cells, ka, **dk); pb = get_cov(cells, kb, **dk)
        bmax = int(max(max(x[2].max() for x in pa), max(x[2].max() for x in pb)))
        Ca, Sa = class_matrix(pa, bmax); Cb, Sb = class_matrix(pb, bmax)
        comp = decompose(Ca.sum(0), Sa.sum(0), Cb.sum(0), Sb.sum(0), bmax, hier, offs, wts)
        by_rep.setdefault(r, {})[seed] = {k: float(v) for k, v in comp.items()}
        if boot_n:
            n = min(Ca.shape[0], Cb.shape[0])
            rs = np.random.RandomState(BOOT_SEED)
            idx = rs.randint(0, n, size=(boot_n, n))
            W = np.zeros((boot_n, n))
            rows = np.arange(boot_n)
            for j in range(n):
                np.add.at(W, (rows, idx[:, j]), 1.0)
            bc = decompose(W @ Ca[:n], W @ Sa[:n], W @ Cb[:n], W @ Sb[:n], bmax, hier,
                           "composition" if offs == "linear_in_b" else offs, wts)
            for c in COMPONENTS:
                sreq[c].append(float(np.var(bc[c], ddof=1)))
    out = {"pair": pk, "orientation": f"Delta := y(d{D2}) - y(d{D})  (D -> D' = MORE decode SMs)",
           "n_replicates": sum(len(v) for v in by_rep.values()), "components": {}}
    for c in COMPONENTS:
        per = {r: {sd: by_rep[r][sd][c] for sd in by_rep[r]} for r in by_rep}
        sr = math.sqrt(float(np.mean(sreq[c]))) if sreq[c] else 0.0
        ci = rung_ci(per, sr)
        keep = ("point", "ci95", "half_width", "se_registered", "se_without_req_diagnostic",
                "sigma_boot", "sigma_seed", "sigma_req", "variance_mode",
                "ci95_common_trace_diagnostic", "half_width_common_trace_diagnostic",
                "n_boot", "n_seed_per_boot")
        out["components"][c] = {k: ci.get(k) for k in keep}
        vcomp = variance_components([[per[r][sd] for sd in sorted(per[r])] for r in sorted(per)])
        out["components"][c]["gate116"] = {k: (vcomp.get(k) if vcomp.get(k) is not None else "n/a") for k in (
            "variance_mode", "sigma_boot_status", "sigma_boot_raw_variance", "F",
            "sigma_boot_95_upper", "upper_bound_degenerate", "sigma_cell")}
    for k in ("offsupport_mass", "offsupport_M_serviceDp_compD", "offsupport_M_serviceD_compDp",
              "E_D", "E_Dp", "m_D", "m_Dp", "composition_interaction_resid"):
        out[k] = float(np.mean([by_rep[r][sd][k] for r in by_rep for sd in by_rep[r]]))
        out[k + "_max_over_replicates"] = float(np.max([by_rep[r][sd][k] for r in by_rep for sd in by_rep[r]]))
    out["_replicates"] = {f"r{r}_seed{sd}": by_rep[r][sd] for r in by_rep for sd in by_rep[r]}
    return out


# =============================================================================
# Forbidden-output selfcheck (verdict G-delta): the `does_not_produce` declaration
# is checked MECHANICALLY against every key path of the output.  Every declared
# entry must have a rule (else "unmapped"), and no key may match a rule.
# Mutation-tested in the prereg sec 10 (a check that cannot fail checks nothing).
# =============================================================================
FORBIDDEN_RULES = (
    ("arm ranking", r"(^|_)(rank|ranking|arm_order)(_|$)"),
    ("argmin/argmax/envelope", r"argmin|argmax|envelope|e_hat|ehat"),
    ("which arm is best", r"best|optimal|optimum|winner"),
    ("stake #1 statement (either direction)", r"stake|position_moves|split_moves"),
    ("policy/controller claim", r"policy|controller|recommend"),
    ("any sign statement on Delta (Q-A sign_agree regime, N-1)", "PATH:sign&delta"),
    ("G compared to 1 (== sign of Delta, N-1 / QB-1)", r"gt_?1|above_?1|exceeds_?1|g_vs_1|over_one"),
)


def _key_paths(o, pre=""):
    if isinstance(o, dict):
        for k_, v_ in o.items():
            p_ = f"{pre}/{k_}"
            yield p_, str(k_)
            yield from _key_paths(v_, p_)
    elif isinstance(o, list):
        for i_, v_ in enumerate(o):
            yield from _key_paths(v_, f"{pre}[{i_}]")


def forbidden_output_selfcheck(res):
    kp = [x for x in _key_paths(res) if not x[0].startswith("/L_forbidden_output_selfcheck")]
    out_rules = []
    for entry, pat in FORBIDDEN_RULES:
        if pat == "PATH:sign&delta":
            viol = [p_ for p_, k_ in kp if "sign" in p_.lower() and "delta" in p_.lower()]
            if "delta_mean" in res.get("C_sign_pattern", {}):
                viol.append("/C_sign_pattern/delta_mean")
        else:
            viol = [p_ for p_, k_ in kp if re.search(pat, k_.lower())]
        out_rules.append({"entry": entry, "pattern": pat, "violations": sorted(set(viol))})
    unmapped = [e_ for e_ in res.get("does_not_produce", [])
                if e_ not in [r_[0] for r_ in FORBIDDEN_RULES]]
    return {"n_key_paths_scanned": len(kp), "rules": out_rules, "unmapped_entries": unmapped,
            "passed": bool(not unmapped and all(not x["violations"] for x in out_rules))}



def selfcheck_mutation_test(res):
    """gate #53: the selfcheck must FAIL on mutants that re-introduce a forbidden output
    (M1 is exactly the revert of verdict D1).  Mutants are applied to deep copies."""
    import copy
    muts = (
        ("M1 re-add C_sign_pattern['delta_mean'] (= D1 revert)",
         lambda r: r["C_sign_pattern"].__setitem__("delta_mean", {"neg_point": 0})),
        ("M2 key 'delta_sign_count'", lambda r: r.__setitem__("delta_sign_count", 0)),
        ("M3 key 'argmin_arm'", lambda r: r.__setitem__("argmin_arm", 0)),
        ("M4 key 'best_split'", lambda r: r.__setitem__("best_split", 0)),
        ("M5 key 'G_gt_1_cells'", lambda r: r.__setitem__("G_gt_1_cells", 0)),
        ("M6 key 'arm_rank'", lambda r: r.__setitem__("arm_rank", 0)),
        ("M7 declared ban without a rule", lambda r: r["does_not_produce"].append("unchecked ban")),
        ("M8 key 'policy_recommendation'", lambda r: r.__setitem__("policy_recommendation", 0)),
        ("M9 key 'stake1_moves'", lambda r: r.__setitem__("stake1_moves", 0)),
    )
    base = {k: v for k, v in res.items() if k != "L_forbidden_output_selfcheck"}
    out = []
    for name, f in muts:
        r = copy.deepcopy(base)
        f(r)
        out.append({"mutant": name, "selfcheck_passed": forbidden_output_selfcheck(r)["passed"]})
    return {"mutants": out, "all_mutants_detected": all(not x["selfcheck_passed"] for x in out)}


SENSITIVITY = [
    ("REGISTERED", {}),
    ("point=emission", {"point": "emission"}),
    ("point=midpoint", {"point": "midpoint"}),
    ("exposure=overlap>=1/2", {"exposure_rule": "overlap_half"}),
    ("pre_start_lag=+50ms", {"pre_start_lag": 0.05}),
    ("arrival=span_rescaled", {"arrival": "span_rescaled"}),
    ("offsupport=linear_in_b", {"offsupport": "linear_in_b"}),
    ("batch=none(e-only)", {"batch": "none"}),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=BOOT_N)
    ap.add_argument("--out", default=str(HERE / "qb_forecast_result.json"))
    args = ap.parse_args()
    t0 = time.time()
    qa, p7, cc = qa_cells(), p7_cells(), c_cells()
    allc = {**qa, **p7, **cc}
    res = {"prereg": "PREREG_QB_LOOPGAIN_2026-09-11.md",
           "name": "Q-B' (exploratory, defined after seeing the data; file name kept as identifier)",
           "audit": "audit_qb_rules_2026-09-11/VERDICT.md -- GO-with-caveats",
           "gpu_hours_spent_by_this_file": 0,
           "produces": "timeline validation, token covariates, exposure/population/supply "
                       "decomposition of paired mean-ITL differences with inherited CIs, "
                       "sensitivity, load-covariate R^2",
           "does_not_produce": ["arm ranking", "argmin/argmax/envelope", "which arm is best",
                                "stake #1 statement (either direction)", "policy/controller claim",
                                "any sign statement on Delta (Q-A sign_agree regime, N-1)",
                                "G compared to 1 (== sign of Delta, N-1 / QB-1)"],
           "registered_constants": {
               "arrival": "seed_raw: a_k = cumsum(RandomState(seed).exponential(1/rate, N-1))",
               "covariate_point": "launch tau_ij = t_{i,j-1}",
               "pre_interval": "[a_k, F_k), k != i",
               "batch": "#{k: F_k <= tau < last_k} (incl. i)",
               "statistic": "pooled decode-token mean ITL (axis P)",
               "primary_estimand": "2x2 table M[s,c]; PRIMARY path K@D + S@D' (needs M[D,D'] only)",
               "secondary": "S@D, K@D', I (need M[D',D] -> extrapolation); exposure/population split "
                            "under BOTH hierarchies with midpoint weights (neither chosen)",
               "offsupport": "missing mu filled with other arm's mu (dmu := 0), mass reported per counterfactual",
               "orientation": "D < D', Delta := y(D') - y(D)",
               "boot": {"B": args.boot, "seed": BOOT_SEED, "resampling": "paired request, covariates fixed"},
               "ci": "qa_analyze.rung_ci VERBATIM (rev7 sec 3 / rev8 A2 / rev9 P3)"}}
    # (A) alignment
    al = {}
    for key in sorted(allc):
        al[Path(allc[key]).name + f"|{key[0]}"] = alignment(get_tl(allc, key))
    agg = collections.defaultdict(lambda: [0, 0.0, 0.0, 0.0, 0])
    for name, v in al.items():
        src = name.split("|")[1]
        if v["n_partnered"]:
            g = agg[src]; g[0] += v["n_partnered"]; g[1] += v["n_partnered"] * v["frac_lt_1ms"]
            g[2] += v["n_partnered"] * v["frac_lt_2ms"]; g[3] += v["n_partnered"] * v["frac_lt_5ms"]; g[4] += 1
    res["A_timeline_alignment"] = {
        src: {"cells": g[4], "partnered_tokens": g[0], "frac_lt_1ms": g[1] / g[0],
              "frac_lt_2ms": g[2] / g[0], "frac_lt_5ms": g[3] / g[0]} for src, g in agg.items()}
    res["A_timeline_alignment_per_cell"] = al
    # (B) per cell
    cs = [cell_summary(allc, k) for k in sorted(allc)]
    res["B_cells"] = cs
    # classifier validation per arm, QA only (matched design)
    val = {}
    for D in ARMS:
        un, ex = [], []
        for k in sorted(qa):
            if k[5] != D:
                continue
            per = get_cov(qa, k)
            itl = np.concatenate([p[0] for p in per]); e = np.concatenate([p[1] for p in per])
            un.append(itl[e == 0]); ex.append(itl[e == 1])
        un, ex = np.concatenate(un), np.concatenate(ex)
        val[f"d{D}"] = {"n_unexposed": int(un.size), "n_exposed": int(ex.size),
                        "unexposed_p05_p50_p95": [float(np.percentile(un, p)) for p in (5, 50, 95)],
                        "exposed_p05_p50_p95": [float(np.percentile(ex, p)) for p in (5, 50, 95)]}
    res["B_classifier_by_arm_QA"] = val
    # E vs rho (QA)
    Es = np.array([c["E_token_exposure"] for c in cs if c["key"][0] == "QA"])
    rh = np.array([c["rho_realised"] for c in cs if c["key"][0] == "QA"])
    Us = np.array([c["prefill_busy_frac_U"] for c in cs if c["key"][0] == "QA"])
    res["B_exposure_vs_rho_QA"] = {
        "corr_E_rho": float(np.corrcoef(Es, rh)[0, 1]), "corr_E_U": float(np.corrcoef(Es, Us)[0, 1]),
        "mean_E_over_U": float(np.mean(Es / Us)), "n": int(Es.size)}
    # (C)+(D) decomposition with CI
    pr = pairs_of({**qa, **p7})
    dec = []
    for pk in sorted(pr):
        dec.append(pair_decomp({**qa, **p7}, pk, pr[pk], args.boot))
    res["C_decomposition"] = dec
    # (G) cross-check vs Q-A delta_mean
    qres = json.loads((HERE / "qa_regret_result.json").read_text())
    qmap = {(m["rung_lambda"], m["shape"], tuple(m["pair_D_minus_Dprime"])):
            m["delta_itl_by_estimator"]["mean"]["point"] for m in qres["delta_matrix"]}
    diffs = []
    for d in dec:
        src, lam, s, D, D2 = d["pair"]
        if src != "QA":
            continue
        q = qmap.get((lam, s, (D2, D)))
        if q is not None:
            diffs.append(abs(d["components"]["delta_mean"]["point"] - q))
    res["G_crosscheck_vs_QA_delta_mean"] = {"n": len(diffs), "max_abs_diff_ms": max(diffs),
                                            "note": "independent loader path; Q-A orientation D-D' flipped"}
    # (E) sensitivity (point estimates only; no bootstrap).  Both hierarchies are
    # ALWAYS reported (C_*_Efirst / C_*_Bfirst), so hierarchy is not a variant here.
    sens = {}
    for name, kw in SENSITIVITY:
        rows = []
        for pk in sorted(pr):
            d = pair_decomp({**qa, **p7}, pk, pr[pk], 0, **kw)
            rows.append({"pair": list(pk), **{c: d["components"][c]["point"] for c in COMPONENTS},
                         "offsupport_mass": d["offsupport_mass"],
                         "offsupport_M_serviceD_compDp": d["offsupport_M_serviceD_compDp"],
                         "offsupport_M_serviceDp_compD": d["offsupport_M_serviceDp_compD"],
                         "offsupport_M_serviceD_compDp_max_over_replicates": d["offsupport_M_serviceD_compDp_max_over_replicates"],
                         "offsupport_M_serviceDp_compD_max_over_replicates": d["offsupport_M_serviceDp_compD_max_over_replicates"],
                         "resid": d["composition_interaction_resid"]})
        sens[name] = rows
    base = {tuple(r["pair"]): r for r in sens["REGISTERED"]}
    TRACK = ("S_atD", "S_atDp", "K_atD", "K_atDp", "I_interaction", "S0_atD_placebo",
             "S0_atDp_placebo", "C_sm1", "C_comp", "C_exp_Efirst", "C_pop_Efirst",
             "C_exp_Bfirst", "C_pop_Bfirst")
    summ = {}
    for name, rows in sens.items():
        if name == "REGISTERED":
            continue
        ch = {c: [] for c in TRACK}
        flips = {c: 0 for c in TRACK if "placebo" not in c}
        for r in rows:
            b0 = base[tuple(r["pair"])]
            for c in TRACK:
                ch[c].append(abs(r[c] - b0[c]))
                if c in flips and np.sign(r[c]) != np.sign(b0[c]) and abs(b0[c]) > 0:
                    flips[c] += 1
        summ[name] = {"max_abs_change_ms": {c: max(v) for c, v in ch.items()},
                      "median_abs_change_ms": {c: float(np.median(v)) for c, v in ch.items()},
                      "sign_flips_vs_registered": flips, "n_pairs": len(rows)}
    # hierarchy disagreement (registered: BOTH reported, NEITHER chosen)
    hd = {"C_pop_sign_disagree": 0, "C_exp_sign_disagree": 0,
          "exp_vs_pop_magnitude_order_disagree": 0, "n_pairs": len(base)}
    for r in base.values():
        if np.sign(r["C_pop_Efirst"]) != np.sign(r["C_pop_Bfirst"]):
            hd["C_pop_sign_disagree"] += 1
        if np.sign(r["C_exp_Efirst"]) != np.sign(r["C_exp_Bfirst"]):
            hd["C_exp_sign_disagree"] += 1
        if ((abs(r["C_exp_Efirst"]) > abs(r["C_pop_Efirst"]))
                != (abs(r["C_exp_Bfirst"]) > abs(r["C_pop_Bfirst"]))):
            hd["exp_vs_pop_magnitude_order_disagree"] += 1
    res["E_sensitivity"] = {"summary": summ, "hierarchy_disagreement": hd, "rows": sens}
    # G (derived, dimensionless) -- composition/supply ratio, path-specific, from the 2x2 table
    for d in dec:
        c = d["components"]
        sD, sDp, kD, kDp = (c["S_atD"]["point"], c["S_atDp"]["point"], c["K_atD"]["point"], c["K_atDp"]["point"])
        csm = c["C_sm1"]["point"] + c["C_sm0_placebo"]["point"]
        d["G_ratio"] = {
            "supply_first_path": (-kDp / sD) if sD else None,     # Delta = S_atD + K_atDp
            "composition_first_path": (-kD / sDp) if sDp else None,  # Delta = S_atDp + K_atD
            "midpoint": (-c["C_comp"]["point"] / csm) if csm else None,
            "definition": "-(composition component) / (supply component), dimensionless, "
                          "descriptive standardization; path label REQUIRED (cf = "
                          "composition_first_path, sf = supply_first_path)",
            "BAN": "see prereg sec 12 QB-1 / QB-8 / QB-14 (G vs 1 == sign of Delta; "
                   "no pooled median as a single number; no path-unlabelled G; "
                   "not a control-loop or actuator gain)"}
        d["K_sign_invariant_over_references"] = bool(np.sign(kD) == np.sign(kDp))
        d["S_sign_invariant_over_references"] = bool(np.sign(sD) == np.sign(sDp))
    # placebo summary (a REAL check of the classifier: both arms run at 108 SM on e=0 tokens)
    pl = ([d["components"]["S0_atD_placebo"] for d in dec]
          + [d["components"]["S0_atDp_placebo"] for d in dec])
    res["C_placebo_summary"] = {
        "n_placebo_estimates": len(pl), "n_pair_cells": len(dec),
        "max_abs_point_ms": max(abs(x["point"]) for x in pl),
        "n_ci_excluding_zero_registered": sum(1 for x in pl if x["ci95"][0] > 0 or x["ci95"][1] < 0),
        "n_ci_excluding_zero_common_trace": sum(
            1 for x in pl if x.get("ci95_common_trace_diagnostic")
            and (x["ci95_common_trace_diagnostic"][0] > 0 or x["ci95_common_trace_diagnostic"][1] < 0)),
        "expected_by_chance_at_alpha_0.05": 0.05 * len(pl)}
    # sign pattern of the decomposition components (deducible signs, QB-5).
    # delta_mean is EXCLUDED: a sign count of Delta is an arm order (N-1, QB-13).
    res["C_sign_pattern"] = {
        c: {"neg_ci": sum(1 for d in dec if d["components"][c]["ci95"][1] < 0),
            "pos_ci": sum(1 for d in dec if d["components"][c]["ci95"][0] > 0),
            "ci_contains_0": sum(1 for d in dec if d["components"][c]["ci95"][0] <= 0 <= d["components"][c]["ci95"][1]),
            "neg_point": sum(1 for d in dec if d["components"][c]["point"] < 0),
            "pos_point": sum(1 for d in dec if d["components"][c]["point"] > 0)}
        for c in COMPONENTS if c != "delta_mean"}
    res["C_sign_pattern_note"] = ("delta_mean EXCLUDED -- FORBIDDEN output: a sign count of Delta "
                                  "(= Q-A Delta_mean flipped) is an arm order on the mean "
                                  "(Q-A N-1; prereg sec 12 QB-13)")
    # variance components per component (for the power section)
    vc = {}
    for c in ("S_atD", "S_atDp", "K_atD", "K_atDp", "I_interaction", "C_sm1", "C_comp", "delta_mean"):
        rows = []
        for d in dec:
            x = d["components"][c]
            rows.append({"pair": d["pair"], "variance_mode": x["variance_mode"],
                         "sigma_boot": x["sigma_boot"], "sigma_seed": x["sigma_seed"],
                         "sigma_req": x["sigma_req"], "half_width": x["half_width"],
                         "point": x["point"], "gate116": x["gate116"]})
        vc[c] = rows
    res["D_variance_components"] = vc
    # (F) load covariate.  Q-A 2-seed rungs have only TWO distinct span values per
    # pair-cell (seed repeats across rounds) -> R^2 there = between-seed share.
    # P7 has 12 distinct seeds -> a genuine regression on span_factor.
    RC = ("S_atD", "S_atDp", "K_atD", "K_atDp", "I_interaction", "delta_mean")
    r2 = {"QA_2seed_between_seed_share": {c: [] for c in RC},
          "P7_12seed_regression": {c: [] for c in RC}}
    for d in dec:
        src, lam, s, D, D2 = d["pair"]
        reps = d["_replicates"]
        xs, ys = [], {c: [] for c in RC}
        for rk, v in reps.items():
            rr = int(rk[1]); seed = int(rk.split("seed")[1])
            n = get_tl({**qa, **p7}, (src, lam, s, rr, seed, D))["n"]
            xs.append(span_factor(seed, n))
            for c in ys:
                ys[c].append(v[c])
        xs = np.asarray(xs)
        if xs.size < 4 or np.ptp(xs) == 0:
            continue
        tag = "P7_12seed_regression" if src == "P7" else "QA_2seed_between_seed_share"
        for c in ys:
            y = np.asarray(ys[c]); A = np.vstack([np.ones_like(xs), xs]).T
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            ss_res = float(((y - A @ coef) ** 2).sum()); ss_tot = float(((y - y.mean()) ** 2).sum())
            r2[tag][c].append({"pair": d["pair"], "R2": (1 - ss_res / ss_tot) if ss_tot > 0 else None,
                               "slope_ms_per_unit_span": float(coef[1]),
                               "resid_sd": float(np.sqrt(ss_res / max(y.size - 2, 1))),
                               "raw_sd": float(np.std(y, ddof=1))})
    res["F_load_covariate"] = r2
    # service curves (per arm, within-arm only -- NO cross-arm use of probe C)
    sc = {}
    for key in sorted(allc):
        per = get_cov(allc, key)
        itl = np.concatenate([p[0] for p in per]); e = np.concatenate([p[1] for p in per])
        b = np.concatenate([p[2] for p in per])
        rows = {}
        for ee in (0, 1):
            for bb in sorted(set(b[e == ee].tolist())):
                sel = (e == ee) & (b == bb)
                rows[f"e{ee}_b{bb}"] = [int(sel.sum()), float(itl[sel].mean())]
        sc["|".join(map(str, key))] = rows
    res["H_service_curves_within_arm"] = sc
    # (I) OPTIONAL Stage-1 forecast (fresh-seed confirmatory replication, prereg sec 4).
    #     Seeds 81..96 UNSELECTED (no |span-1| rule): 4 rounds x 2 seeds, fresh per round.
    st = {"design": "arms {16,44,54}; lambda 0.42; shapes A,B; 4 rounds x 2 fresh seeds "
                    "(round r uses seeds 81+2(r-1), 82+2(r-1)); N = round(0.42*150)+1 = 64",
          "seeds": {}, "mu_p_ref": MU_P_REF}
    lam1, n1 = 0.42, int(round(0.42 * 150.0)) + 1
    for r in range(1, 5):
        for sd in (81 + 2 * (r - 1), 82 + 2 * (r - 1)):
            sf = span_factor(sd, n1)
            st["seeds"][str(sd)] = {"round": r, "span_factor": sf,
                                    "arrival_window_s": (n1 - 1) / lam1 * sf,
                                    "X_forecast_arrival_axis": lam1 / sf,
                                    "rho_realised_forecast": {f"d{D}": lam1 / (sf * MU_P_REF[D])
                                                              for D in (16, 44, 54)}}
    # trace-to-trace SD reference: P7 (12 fresh seeds) sigma_seed of the primary components
    ref = {}
    for d in dec:
        if d["pair"][0] != "P7":
            continue
        ref["d%d->d%d" % (d["pair"][3], d["pair"][4])] = {
            c: {k: d["components"][c][k] for k in ("sigma_boot", "sigma_seed", "sigma_req", "point", "half_width")}
            for c in ("K_atD", "S_atDp", "delta_mean")}
    st["trace_sd_reference_P7_W0.491"] = ref
    # projected registered half-width for n_boot = 4, n_seed = 2 fresh per boot, t_3,
    # using the Q-A lambda=0.42 sigma_cell as an upper proxy for (trace + boot) and the
    # P7 sigma_seed as the trace component (both are FORECAST inputs, not imports of claims)
    proj = {}
    tcrit = t_ppf(0.975, 3)
    for d in dec:
        if d["pair"][0] != "QA" or d["pair"][1] != 0.42:
            continue
        key = "%s d%d->d%d" % (d["pair"][2], d["pair"][3], d["pair"][4])
        proj[key] = {}
        for c in ("K_atD", "S_atDp", "delta_mean"):
            x = d["components"][c]
            p7k = "d%d->d%d" % (d["pair"][3], d["pair"][4])
            s_tr = ref.get(p7k, {}).get(c, {}).get("sigma_seed")
            s_req = x["sigma_req"]
            if s_tr is None:
                continue
            se = math.sqrt(s_tr ** 2 / 8.0 + s_req ** 2 / 8.0)
            proj[key][c] = {"existing_point": x["point"], "existing_half_width_registered": x["half_width"],
                            "existing_half_width_common_trace": x.get("half_width_common_trace_diagnostic"),
                            "projected_half_width_4boot_x_2freshseed": tcrit * se,
                            "inputs": {"sigma_trace_from_P7": s_tr, "sigma_req_from_QA_0.42": s_req,
                                       "sigma_boot": "undetected in P7 & Q-A -> set 0 (gate #116: NOT a point estimate)"}}
    st["projected_half_widths"] = proj
    res["I_stage1_forecast"] = st

    # ------------------------------------------------------------------
    # (N) prereg sec 3.6 proxy table -- ONE definition (verdict D4):
    #   per replicate r of a Q-A pair:  gap_D(r) = mu_exposed(D,r) - mu_unexposed(D,r)
    #   (cell-level means of arm D in that replicate),  x_q(r) = (q(D',r) - q(D,r)) * gap_D(r),
    #   pair-level x_q = mean_r x_q(r);  y = K@D point (= mean_r K@D(r)).  y = c*x, origin.
    #   !! the dE row is an ACCOUNTING IDENTITY (K@D == dE*gap + within-e population term):
    #   its R^2 only measures how small the population term is (gate #9, QB-12) -- NOT evidence.
    cm = {tuple(c_["key"]): c_ for c_ in cs}
    prox = []
    for d in dec:
        src_, lam_, s_, D_, D2_ = d["pair"]
        if src_ != "QA":
            continue
        per = []
        for rk in d["_replicates"]:
            rr = int(rk[1]); sd = int(rk.split("seed")[1])
            ca = cm[("QA", lam_, s_, rr, sd, D_)]; cb = cm[("QA", lam_, s_, rr, sd, D2_)]
            gap = ca["mu_exposed_ms"] - ca["mu_unexposed_ms"]
            per.append({"gap": gap, "dE": cb["E_token_exposure"] - ca["E_token_exposure"],
                        "drho": cb["rho_realised"] - ca["rho_realised"],
                        "dU": cb["prefill_busy_frac_U"] - ca["prefill_busy_frac_U"]})
        row = {"pair": d["pair"], "K_atD": d["components"]["K_atD"]["point"]}
        for k_ in ("drho", "dU", "dE"):
            row["x_" + k_] = float(np.mean([p[k_] * p["gap"] for p in per]))
            row["mean_" + k_] = float(np.mean([p[k_] for p in per]))
        prox.append(row)
    yv = np.array([x["K_atD"] for x in prox])

    def _fit(xv):
        c_ = float(xv @ yv / (xv @ xv)); rr_ = yv - c_ * xv
        return {"c": c_, "R2": float(1.0 - rr_ @ rr_ / ((yv - yv.mean()) @ (yv - yv.mean()))),
                "median_abs_rel_resid": float(np.median(np.abs(rr_) / np.abs(yv)))}
    mdE = np.array([x["mean_dE"] for x in prox]); mdr = np.array([x["mean_drho"] for x in prox])
    mdU = np.array([x["mean_dU"] for x in prox])
    res["N_proxy_table_sec3_6"] = {
        "definition": "per-replicate product (dq(r) * gap_D(r)), averaged over replicates; "
                      "gap_D(r) from arm D's cell in replicate r; y = K@D point; origin fit",
        "n_pairs": len(prox),
        "rho_proxy": _fit(np.array([x["x_drho"] for x in prox])),
        "U_proxy": _fit(np.array([x["x_dU"] for x in prox])),
        "dE_times_gap_IDENTITY_NOT_EVIDENCE": _fit(np.array([x["x_dE"] for x in prox])),
        "dE_alone": _fit(mdE), "drho_alone": _fit(mdr),
        "corr_dE_drho": float(np.corrcoef(mdE, mdr)[0, 1]),
        "corr_dE_dU": float(np.corrcoef(mdE, mdU)[0, 1]),
        "rows": prox}
    # ------------------------------------------------------------------
    # (O) path disclosure (verdict D8): registered layer vs b-linear fill vs e-only layer.
    #     Algebra (checked, count of Delta<0 NOT emitted): with S@D<0, S@D'<0,
    #     G_cf > G_sf  <=>  Delta * I > 0, i.e. if I<0 then G_cf > G_sf <=> Delta<0.
    def _paths(rows_):
        gcf = [-x["K_atD"] / x["S_atDp"] for x in rows_]
        gsf = [-x["K_atDp"] / x["S_atD"] for x in rows_]
        chk = [((-x["K_atD"] / x["S_atDp"]) > (-x["K_atDp"] / x["S_atD"]))
               == (x["delta_mean"] * x["I_interaction"] > 0)
               for x in rows_ if x["S_atD"] < 0 and x["S_atDp"] < 0]
        return {"G_cf_median_pooled_DISCLOSURE_ONLY": float(np.median(gcf)),
                "G_sf_median_pooled_DISCLOSURE_ONLY": float(np.median(gsf)),
                "K_atDp_negative_points": int(sum(1 for x in rows_ if x["K_atDp"] < 0)),
                "offsupport_M_serviceD_compDp_mean": float(np.mean([x["offsupport_M_serviceD_compDp"] for x in rows_])),
                "offsupport_M_serviceDp_compD_mean": float(np.mean([x["offsupport_M_serviceDp_compD"] for x in rows_])),
                "algebra_Gcf_gt_Gsf_iff_Delta_times_I_pos_holds": bool(all(chk)),
                "algebra_n_pairs_checked": len(chk)}
    res["O_path_disclosure"] = {
        "note": "pooled medians are a PATH-DEPENDENCE disclosure only (verdict D8); QB-1 forbids a "
                "pooled median as a single headline number",
        "registered": _paths(sens["REGISTERED"]),
        "offsupport_linear_in_b": _paths(sens["offsupport=linear_in_b"]),
        "e_only_batch_none": _paths(sens["batch=none(e-only)"])}
    # ------------------------------------------------------------------
    # (M) Little identity split of the probe-C candidate confound (prereg sec 0.2(b)) -- NOT an output
    def _LX(key):
        d_ = load(cc[key])
        return (sum(sum(x) for x in d_["itls"]) / d_["duration"], d_["completed"] / d_["duration"])
    lit = {}
    for tag, ka, kb in (("out96_r2", ("C", 2, "A", 0, 41, 16), ("C", 2, "A", 0, 41, 92)),
                        ("out384_r1", ("C", 1, "B", 0, 41, 16), ("C", 1, "B", 0, 41, 92))):
        (La, Xa), (Lb, Xb) = _LX(ka), _LX(kb)
        rL, rX, rS = La / Lb, Xa / Xb, (La / Xa) / (Lb / Xb)
        lit[tag] = {"cells": [Path(cc[ka]).name, Path(cc[kb]).name], "L_ratio": rL, "X_ratio": rX,
                    "sojourn_ratio": rS, "log_share_arrival": math.log(rX) / math.log(rL),
                    "log_share_sojourn": math.log(rS) / math.log(rL)}
    res["M_little_probeC"] = lit
    # ------------------------------------------------------------------
    # (K) operating-point provenance (server_info signature per source)
    def _sig(cells_):
        cnt = collections.Counter()
        for k_ in cells_:
            d_ = load(cells_[k_]); si = d_["server_info"]
            cnt[json.dumps({"disable_cuda_graph": si.get("disable_cuda_graph"),
                            "enable_pdmux": si.get("enable_pdmux"),
                            "attention_backend": si.get("attention_backend"),
                            "max_running_requests": si.get("max_running_requests"),
                            "chunked_prefill_size": si.get("chunked_prefill_size"),
                            "pdmux_config": str(si.get("pdmux_config_path")).split("/")[-1],
                            "context_length": si.get("context_length"),
                            "model": str(si.get("model_path")).split("/")[-1],
                            "random_range_ratio": d_.get("random_range_ratio"),
                            "random_input_len": d_.get("random_input_len")}, sort_keys=True)] += 1
        return [{"signature": json.loads(k_), "n_benches": v} for k_, v in cnt.items()]
    res["K_provenance"] = {"QA": _sig(qa), "P7": _sig(p7), "C": _sig(cc)}
    # ------------------------------------------------------------------
    # (J) every aggregate the prereg body cites (gate #159/#161, verdict D4)
    def _agg(v):
        v = np.asarray(v, float)
        return {"mean": float(v.mean()), "median": float(np.median(v)), "min": float(v.min()),
                "max": float(v.max()), "n": int(v.size)}
    paired = set()
    for d in dec:
        s0, l0, h0, D0, D20 = d["pair"]
        for rk in d["_replicates"]:
            rr = int(rk[1]); sd = int(rk.split("seed")[1])
            paired.add((s0, l0, h0, rr, sd, D0)); paired.add((s0, l0, h0, rr, sd, D20))
    armcell = collections.defaultdict(list)
    for key_ in paired:
        c_ = cm[key_]
        if key_[0] == "QA":
            armcell[(key_[1], key_[2], key_[5])].append((c_["mu_exposed_ms"], c_["mu_unexposed_ms"]))
    arm_rng = {}
    for D_ in ARMS:
        m1 = [np.mean([x[0] for x in v]) for (l_, h_, a_), v in armcell.items() if a_ == D_]
        m0 = [np.mean([x[1] for x in v]) for (l_, h_, a_), v in armcell.items() if a_ == D_]
        arm_rng[f"d{D_}"] = {"mu_exposed_min": float(min(m1)), "mu_exposed_max": float(max(m1)),
                             "mu_unexposed_min": float(min(m0)), "mu_unexposed_max": float(max(m0))}
    g116 = {}
    for c_ in ("K_atD", "S_atDp", "delta_mean", "I_interaction"):
        v = [d["components"][c_]["gate116"] for d in dec
             if d["components"][c_]["gate116"]["variance_mode"] == "boot_seed_decomposed"]
        g116[c_] = {"decomposable": len(v),
                    "undetected": sum(1 for x in v if str(x["sigma_boot_status"]).startswith("UNDETECTED")),
                    "degenerate": sum(1 for x in v if x["upper_bound_degenerate"] is True)}
    proj = res["I_stage1_forecast"]["projected_half_widths"]
    pr_rows = []
    for cell_, comps in proj.items():
        for comp_, x in comps.items():
            pr_rows.append({"cell": cell_, "component": comp_,
                            "proj_over_registered": x["projected_half_width_4boot_x_2freshseed"] / x["existing_half_width_registered"],
                            "proj_over_common_trace": x["projected_half_width_4boot_x_2freshseed"] / x["existing_half_width_common_trace"]})
    dur042 = [c_["n_requests"] / c_["X"] for c_ in cs if c_["key"][0] == "QA" and c_["key"][1] == 0.42]
    gcf_by = {s_: [d["G_ratio"]["composition_first_path"] for d in dec if d["pair"][0] == s_] for s_ in ("QA", "P7")}
    res["J_doc_summary"] = {
        "QA_cells_total": len(qa),
        "QA_cells_with_partnered_tokens": res["A_timeline_alignment"]["QA"]["cells"],
        "offsupport_M_serviceD_compDp": {**_agg([d["offsupport_M_serviceD_compDp"] for d in dec]),
                                         "max_over_replicates": max(d["offsupport_M_serviceD_compDp_max_over_replicates"] for d in dec)},
        "offsupport_M_serviceDp_compD": {**_agg([d["offsupport_M_serviceDp_compD"] for d in dec]),
                                         "max_over_replicates": max(d["offsupport_M_serviceDp_compD_max_over_replicates"] for d in dec)},
        "G_cf_by_source_path_cf": {s_: _agg(v) for s_, v in gcf_by.items()},
        "G_cf_range_all_39": [min(gcf_by["QA"] + gcf_by["P7"]), max(gcf_by["QA"] + gcf_by["P7"])],
        "K_atD_range": _agg([d["components"]["K_atD"]["point"] for d in dec]),
        "S_atDp_range": _agg([d["components"]["S_atDp"]["point"] for d in dec]),
        "I_interaction_range": _agg([d["components"]["I_interaction"]["point"] for d in dec]),
        "gate116_counts": g116,
        "QA_2seed_between_seed_share_median": {
            c_: float(np.median([x["R2"] for x in rows_ if x["R2"] is not None]))
            for c_, rows_ in res["F_load_covariate"]["QA_2seed_between_seed_share"].items()},
        "arm_level_means_QA_paired_cells": arm_rng,
        "bbar_exposed_max_per_cell_paired": float(max(cm[k_]["bbar_token_exposed"] for k_ in paired
                                                      if cm[k_]["bbar_token_exposed"] is not None)),
        "placebo_common_trace_denominator": sum(
            1 for d in dec for k_ in ("S0_atD_placebo", "S0_atDp_placebo")
            if d["components"][k_].get("ci95_common_trace_diagnostic")),
        "stage1_rho_realised_max": max(max(v["rho_realised_forecast"].values())
                                       for v in res["I_stage1_forecast"]["seeds"].values()),
        "stage1_projection_ratios": pr_rows,
        "QA_lambda0.42_bench_duration_s": _agg(dur042)}
    # ------------------------------------------------------------------
    # (L) forbidden-output selfcheck (verdict G-delta) -- see forbidden_output_selfcheck()
    res["L_forbidden_output_selfcheck"] = forbidden_output_selfcheck(res)
    res["L_forbidden_output_selfcheck"]["mutation_test"] = selfcheck_mutation_test(res)
    res["runtime_s"] = time.time() - t0
    for d in res["C_decomposition"]:
        d["pair"] = list(d["pair"])
    Path(args.out).write_text(json.dumps(res, indent=1, default=float))
    _print(res)
    if not (res["L_forbidden_output_selfcheck"]["passed"]
            and res["L_forbidden_output_selfcheck"]["mutation_test"]["all_mutants_detected"]):
        print("!! FORBIDDEN-OUTPUT SELFCHECK FAILED", file=sys.stderr)
        sys.exit(3)


def _print(res):
    print("== (A) timeline alignment (partnered tokens within 1/2/5 ms)")
    for k, v in res["A_timeline_alignment"].items():
        print(f"   {k:3s} cells {v['cells']:3d} tokens {v['partnered_tokens']:7d}  "
              f"{v['frac_lt_1ms']:.4f} / {v['frac_lt_2ms']:.4f} / {v['frac_lt_5ms']:.4f}")
    print("== (B) classifier by arm (Q-A): unexposed vs exposed ITL p05/p50/p95 [ms]")
    for k, v in res["B_classifier_by_arm_QA"].items():
        print(f"   {k}: unexp {['%.2f' % x for x in v['unexposed_p05_p50_p95']]} (n {v['n_unexposed']})"
              f"  exp {['%.2f' % x for x in v['exposed_p05_p50_p95']]} (n {v['n_exposed']})")
    print("   E vs rho:", {k: round(v, 4) if isinstance(v, float) else v for k, v in res["B_exposure_vs_rho_QA"].items()})
    print("== (C) 2x2 table: Delta = S_atD + K_atDp = S_atDp + K_atD ; I = S_atDp - S_atD   [ms point (CI95 hw)]")
    for d in res["C_decomposition"]:
        c = d["components"]
        f = lambda k: f"{c[k]['point']:+7.3f}({c[k]['half_width']:.3f})"
        g = d["G_ratio"]
        print(f"   {d['pair'][0]} l={d['pair'][1]:<6} {d['pair'][2]} d{d['pair'][3]}->d{d['pair'][4]:<2} n={d['n_replicates']:2d} "
              f"D={f('delta_mean')} S@D={f('S_atD')} S@D'={f('S_atDp')} K@D={f('K_atD')} K@D'={f('K_atDp')} I={f('I_interaction')} "
              f"pl {c['S0_atD_placebo']['point']:+.3f}/{c['S0_atDp_placebo']['point']:+.3f} | "
              f"Ef exp {c['C_exp_Efirst']['point']:+.2f} pop {c['C_pop_Efirst']['point']:+.2f} ; "
              f"Bf exp {c['C_exp_Bfirst']['point']:+.2f} pop {c['C_pop_Bfirst']['point']:+.2f} | "
              f"G sf {g['supply_first_path']:.2f} cf {g['composition_first_path']:.2f} mid {g['midpoint']:.2f} | E {d['E_D']:.3f}->{d['E_Dp']:.3f} off={d['offsupport_mass']:.3f}")
    print("== placebo:", res["C_placebo_summary"])
    print("== sign pattern (delta_mean excluded, QB-13):", json.dumps(res["C_sign_pattern"]))
    print("== (G)", res["G_crosscheck_vs_QA_delta_mean"])
    print("== (E) sensitivity summary (max |change| ms; sign flips)")
    for k, v in res["E_sensitivity"]["summary"].items():
        m = v["max_abs_change_ms"]
        print(f"   {k:26s} S@D {m['S_atD']:.3f} S@D' {m['S_atDp']:.3f} K@D {m['K_atD']:.3f} K@D' {m['K_atDp']:.3f} "
              f"I {m['I_interaction']:.3f} pl {max(m['S0_atD_placebo'], m['S0_atDp_placebo']):.3f} | flips {v['sign_flips_vs_registered']}")
    print("   hierarchy disagreement:", res["E_sensitivity"]["hierarchy_disagreement"])
    print("== (F) load covariate")
    for tag, dd in res["F_load_covariate"].items():
        for c, rows in dd.items():
            if rows:
                r2s = [x["R2"] for x in rows if x["R2"] is not None]
                print(f"   {tag:30s} {c:10s} n={len(rows)} R2 median {np.median(r2s):.3f} min {min(r2s):.3f} max {max(r2s):.3f}")
    print("== (N) sec 3.6 proxy:", {k: res["N_proxy_table_sec3_6"][k] for k in (
        "rho_proxy", "U_proxy", "dE_times_gap_IDENTITY_NOT_EVIDENCE", "dE_alone", "drho_alone",
        "corr_dE_drho", "corr_dE_dU")})
    print("== (O) path disclosure:", json.dumps(res["O_path_disclosure"]))
    print("== (M) Little probe C:", json.dumps(res["M_little_probeC"]))
    print("== (K) provenance:", json.dumps(res["K_provenance"]))
    print("== (L) forbidden-output selfcheck: passed =", res["L_forbidden_output_selfcheck"]["passed"],
          "| key paths scanned", res["L_forbidden_output_selfcheck"]["n_key_paths_scanned"],
          "| violations", sum(len(x["violations"]) for x in res["L_forbidden_output_selfcheck"]["rules"]),
          "| unmapped", res["L_forbidden_output_selfcheck"]["unmapped_entries"],
          "| mutants detected", sum(1 for x in res["L_forbidden_output_selfcheck"]["mutation_test"]["mutants"]
                                   if not x["selfcheck_passed"]), "/",
          len(res["L_forbidden_output_selfcheck"]["mutation_test"]["mutants"]))
    print(f"runtime {res['runtime_s']:.1f} s")


if __name__ == "__main__":
    main()
