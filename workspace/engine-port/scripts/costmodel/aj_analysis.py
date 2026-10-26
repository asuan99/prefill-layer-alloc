#!/usr/bin/env python3
"""AJ-1 ... AJ-4 derived verdicts of PREREG_COSTMODEL_TERMS_MEAS rev3.6' §1-9 (decision 31 + rule-layer audit r1 repair).

Input = the `analysis.json` that the SEALED harness writes (`ct_runner.py analyze RAW`), one file per Job:
J-A1 and J-A2 (AJ-1, AJ-2, AJ-3) and J-H1 (AJ-2 side table, AJ-4).  Nothing here re-fits T1/T2/T2' data the harness
already fitted, except the two refits §1-9 itself asks for (F1 on `decode_fwd_gpu_ms` for AJ-1 (iii); F1 on the J-H1
common 5 cells per variant for AJ-2), and both call the harness's own `ct_fit` functions (read-only import, no bytecode
written into the sealed harness directory).

Priors (AJ-1 W, AJ-2 2s/k, AJ-4 Delta_ideal) come ONLY from `scripts/costmodel/analytic_terms.py` (analytic-v1, decision 27).
Thresholds are the §1-9 values, fixed at the seal.  This file adds no verdict or metric beyond §1-9; every reading of an
ambiguous phrase is marked [해석] in a comment and listed in INTERPRETATIONS.

Committed BEFORE the J-A1 fetch (PREREG §8-6, §12 item 25).  It has never been run on measured data at commit time.

  python3 aj_analysis.py --ja1 J-A1/analysis.json --ja2 J-A2/analysis.json --jh1 J-H1/analysis.json --out aj.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True          # never write __pycache__ into the sealed harness directory
_ENG = Path(os.path.abspath(__file__)).parents[2]
HARNESS = _ENG / "results" / "cap_char" / "costterms_harness"
sys.path.insert(0, str(_ENG / "scripts" / "costmodel"))
sys.path.insert(0, str(HARNESS))
import analytic_terms as AT  # noqa: E402
import ct_fit as F  # noqa: E402  (harness fit functions, read-only)

VERSION = "aj-analysis-v1.1 (2026-10-27: AJ-2 J-H1 NA_FORM gate, result audit r1; PREREG rev3.6' §1-9)"

# ---------------------------------------------------------------- §1-9 sealed constants
D_STICKY = (16, 24, 34, 44, 64)
D_REF_AJ3 = 16
AJ1_RESID_MAX = 0.03                 # (i) max relative residual
S108_FLAT, S108_KNEE = 0.033, 0.101  # (ii) midpoints of the derived priors 0 / 0.067 / 0.136
KNEE_UNSTABLE_REL = 0.5              # AJ-2 two-Job difference > 50 %
THETA_LO, THETA_HI = 0.9, 1.1        # AJ-3
LAG_SHARE_MAX = 0.03                 # AJ-3
AJ3_L = (32768, 65536)               # AJ-3 pooled L
T2P_L_GRID = (1024, 4096, 8800, 16384, 32768, 65536)
LAG_T1_CELL = "1x2048"               # same-boot T1 (bs 1, c 2k)
PEAK_FLOPS = 312e12                  # bf16 dense @ P = 108 (AJ-4 sealed constant)
HBM_BPS = 2.039e12                   # AJ-4 sealed constant
VARIANTS = (4, 13, 22, 31)
N_LAYERS_AM = 31                     # n_M = 31 - n_A
JH_COMMON_CELLS = ((8, 2048), (8, 8192), (8, 32768), (32, 2048), (32, 8192))
JH_T2_L = (4096, 16384, 32768, 65536)
CAP_MAIN = "cap128"

INTERPRETATIONS = [
    "[해석] I-1 AJ-1/2: 'F1 계수' = harness fits.t1[key]['F1'].coef even when the accepted form is F2/F3; NA_FORM only when the "
    "harness label is FORM_INADEQUATE_* (no form accepted). The chosen form is reported per mode.",
    "[해석] I-2 AJ-1/3: a missing / unusable boot, fit or point (SM_NOT_REALIZED, < 4 cells, absent key) -> NA_POINTS "
    "(§1-9 names NA_POINTS for AJ-3/AJ-4 only; reused for AJ-1/AJ-2 missing inputs).",
    "[해석] I-3 AJ-1: the D axis of (i) is D_real (tables.t1[key].D_real, §1-8 'realized value'); a_16/a_64 are picked by the "
    "planned mode label D16/D64. J-A keys used: 5-part keys with cap128 only (cap-48 D44 boot, TP and resume segments excluded).",
    "[해석] I-4 AJ-1 (iii): BWD_UNRESOLVED / BWD_ESTIMATOR_SPLIT are evaluated separately for sub-verdict (i) and (ii); "
    "an NA on either side gives no comparison label (reason recorded), not a new label.",
    "[해석] I-5 AJ-1 (iii): the decode_fwd_gpu_ms F1 is refit with the harness ct_fit.fit_t1 on the cells that have both a "
    "VALID S_eng median and a gpu_ms median, x = actual ctx (same rule as the S_eng fit, >= 4 cells).",
    "[해석] I-6 AJ-1 (iv): W = analytic_terms.decode_step(nano9b_v2(4,27)).bytes_weights (weights + LM head, read once per step); "
    "BW_eff reported for every PD-mux mode present (D16..D64, 108u).",
    "[해석] I-7 AJ-2: c_knee reported for every J-A cap128 mode with an F1 fit (incl. 108u, 108v). 'difference > 50 %' = "
    "|x1 - x2| / min(|x1|, |x2|) > 0.5 (the stricter, more-flagging reading), separately for b and c; sign flip or a zero "
    "coefficient -> unstable.",
    "[해석] I-8 AJ-2 J-H1 side table and AJ-4 T1 are computed only when the J-H1 T1 W-INDEP label is WEIGHT_INDEP_HELD "
    "(AJ-4 per-term rule); slope of b(n) vs n_M = OLS over the 4 variants, all 4 required.",
    "[해석] I-9 AJ-3: strict NA_POINTS -- the reference (D16) and all four non-reference D must be CONFIRMED at BOTH L 32,768 and "
    "65,536 (10 points); otherwise no theta' label.",
    "[해석] I-10 AJ-3: lag_share is checked for every used point including the reference values; if any lag_share is NA "
    "(same-boot T1 (1,2k) missing) or > 3 %, the theta' label is withheld (THETA_LAG_BOUND_EXCEEDED flag in the latter case); "
    "a withheld label counts as NA for the two-Job comparison.",
    "[해석] I-11 AJ-3: no NA_FORM gate -- theta' is a log-ratio regression on points, it does not use the T2' quadratic fit.",
    "[해석] I-12 AJ-3 descriptive: per-L through-origin theta' and per-L free-intercept slope for every L of the T2' grid "
    "(non-reference D CONFIRMED at that L with the reference CONFIRMED at that L).",
    "[해석] I-13 AJ-4: Delta_ideal = OLS slope over n_A of the per-variant ideal time (equals the one-layer difference since "
    "analytic-v1 is linear in n_A at fixed n_A + n_M); prefill time = prefill_flops.total / 312 TFLOP/s; decode time = "
    "max(decode_step.flops / 312 TFLOP/s, decode_step.bytes_total / 2.039 TB/s) at the NOMINAL c (roofline; both parts reported).",
    "[해석] I-14 AJ-4: decode 'ctx 간 차의 차' = beta1(bs, c) - beta1(bs, 2k) per bs (and the same for Delta_ideal); the "
    "per-cell decode beta1 is reported but flagged level_includes_weight_term.",
    "[해석] I-15 J-H1 keys: dummy_a{n} / 108v / cap128 T1 table and dummy_a{n} / 108v T2 table; more than one matching key -> NA_POINTS.",
]

SCOPE = {
    "AJ-1": "D <= 64 1/D compatibility and 64-108 knee only; s_108 includes the 108u (sticky OFF) vs sticky-D stream-path "
            "difference; BWD_KNEE_64_108 is not 'saturation at 80 SMs'; 'HBM saturation' wording forbidden.",
    "AJ-2": "c_knee = ratio of the whole per-request term; not evidence that the state term is visible in decode time; "
            "no layer-type reading of D differences. No label (only KNEE_COEF_UNSTABLE tag).",
    "AJ-3": "theta' is a composite of one co-resident decode and SM reduction (L >= 32k); does not support 'prefill "
            "throughput proportional to P'.",
    "AJ-4": "beta1 = total effect of a layer swap (weight bytes and KV pool included), 4 points / 2 dof, descriptive; "
            "not a separation of eta_A, eta_M.",
}


# ============================================================================ small helpers
def _load(p):
    return json.loads(Path(p).read_text()) if p else None


def _ols(xs, ys):
    """Ordinary LS y = b0 + b1 x -> (b0, b1, residuals, r2) or None."""
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return None
    b1 = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    b0 = my - b1 * mx
    res = [y - (b0 + b1 * x) for x, y in zip(xs, ys)]
    sst = sum((y - my) ** 2 for y in ys)
    r2 = (1.0 - sum(r * r for r in res) / sst) if sst > 0 else None
    return b0, b1, res, r2


def _origin_slope(xs, ys):
    den = sum(x * x for x in xs)
    return (sum(x * y for x, y in zip(xs, ys)) / den) if den > 0 else None


def _split_key(k):
    return k.split("|")


def ja_groups(A, job):
    """{(round, tag): {mode: key}} over the 5-part cap128 T1 keys of `job` (I-3)."""
    out = {}
    for k in (A.get("tables", {}).get("t1") or {}):
        p = _split_key(k)
        if len(p) != 5 or p[0] != job or p[4] != CAP_MAIN:
            continue
        out.setdefault((p[1], p[2]), {})[p[3]] = k
    return out


# ============================================================================ AJ-1
def label_inv_d(max_resid, t0):
    return "BWD_INV_D_HELD_LE64" if (max_resid <= AJ1_RESID_MAX and t0 >= 0) else "BWD_INV_D_MISS_LE64"


def label_s108(s):
    if s < S108_FLAT:
        return "BWD_FLAT_BY_64"
    if s < S108_KNEE:
        return "BWD_KNEE_64_108"
    return "BWD_NO_KNEE_TO_108"


def shared_weight_bytes():
    return AT.decode_step(AT.nano9b_v2(4, 27), 1, 1)["bytes_weights"]


def _coef_from_fit(fit):
    """harness fits.t1 entry -> (status, F1 coef, chosen form)."""
    if not fit:
        return "NA_POINTS", None, None
    if str(fit.get("label") or "").startswith("FORM_INADEQUATE"):
        return "NA_FORM", None, None
    f1 = fit.get("F1") or {}
    if not f1.get("coef"):
        return "NA_POINTS", None, None
    return None, list(f1["coef"]), (fit.get("chosen") or {}).get("form")


def _gpu_fit(tab, mode):
    """I-5: F1 (harness rule F1 -> F2 -> F3 acceptance) on decode_fwd_gpu_ms cell medians, x = actual ctx."""
    cells, ctx, gpu = tab.get("cells") or {}, tab.get("ctx_actual") or {}, tab.get("gpu_ms") or {}
    pts = []
    for ck in sorted(set(cells) & set(ctx) & set(gpu)):
        if gpu[ck] is None or ctx[ck] is None:
            continue
        bs, c = (int(x) for x in ck.split("x"))
        pts.append((bs, ctx[ck], gpu[ck], c))
    if len(pts) < 4:
        return "NA_POINTS", None, None
    fit = F.fit_t1(pts, mode)
    if fit["label"]:
        return "NA_FORM", None, None
    if not fit["F1"] or not fit["F1"].get("coef"):
        return "NA_POINTS", None, None
    return None, list(fit["F1"]["coef"]), fit["chosen"]["form"]


def aj1_eval(entries):
    """entries = {mode: {'status': None|'NA_FORM'|'NA_POINTS', 'a': a_ms, 'D_real': int}} for one Job and one estimator."""
    W = shared_weight_bytes()
    out = {}
    sticky = [f"D{d}" for d in D_STICKY]
    st = [entries.get(m, {"status": "NA_POINTS"}) for m in sticky]
    bad = [e.get("status") for e in st if e.get("status") or e.get("a") is None or not e.get("D_real")]
    if bad:
        out["i"] = {"label": "NA_FORM" if "NA_FORM" in bad else "NA_POINTS",
                    "missing": [m for m, e in zip(sticky, st) if e.get("status") or e.get("a") is None or not e.get("D_real")]}
    else:
        Ds, ys = [e["D_real"] for e in st], [e["a"] for e in st]
        co = F.rel_lstsq([[1.0, 1.0 / d] for d in Ds], ys)
        if co is None:
            out["i"] = {"label": "NA_POINTS", "missing": [], "reason": "singular fit"}
        else:
            t0, kappa = co
            res = {m: (t0 + kappa / d - y) / y for m, d, y in zip(sticky, Ds, ys)}
            mx = max(abs(v) for v in res.values())
            out["i"] = {"label": label_inv_d(mx, t0), "t0_ms": t0, "kappa_ms_sm": kappa, "D_real": dict(zip(sticky, Ds)),
                        "rel_resid": res, "max_rel_resid": mx, "dof": len(Ds) - 2}
    need = ("D16", "D64", "108u")
    ne = {m: entries.get(m, {"status": "NA_POINTS"}) for m in need}
    nb = [e.get("status") or ("NA_POINTS" if e.get("a") is None else None) for e in ne.values()]
    nb = [x for x in nb if x]
    if nb:
        out["ii"] = {"label": "NA_FORM" if "NA_FORM" in nb else "NA_POINTS"}
    else:
        a16, a64, a108 = ne["D16"]["a"], ne["D64"]["a"], ne["108u"]["a"]
        den = a16 - a64
        if den == 0:
            out["ii"] = {"label": "NA_POINTS", "reason": "a_16 == a_64"}
        else:
            s = (a64 - a108) / den
            out["ii"] = {"label": label_s108(s), "s_108": s, "a_16": a16, "a_64": a64, "a_108u": a108}
    a16, a64 = (entries.get("D16") or {}).get("a"), (entries.get("D64") or {}).get("a")
    out["iv"] = {"descriptive": True,
                 "r_a": (a16 / a64) if (a16 is not None and a64) else None,
                 "W_bytes": W, "W_source": f"analytic_terms {AT.VERSION} decode_step.bytes_weights (a4)",
                 "BW_eff_bytes_per_s": {m: (W / (e["a"] * 1e-3) if e.get("a") else None)
                                        for m, e in entries.items() if (m.startswith("D") or m == "108u")
                                        and not e.get("status")}}
    return out


def aj1_job(A, job):
    """Per (round, tag) group of a J-A analysis.json: S_eng and gpu estimators + estimator split."""
    tabs, fits = A.get("tables", {}).get("t1") or {}, (A.get("fits") or {}).get("t1") or {}
    res = {}
    for (rnd, tag), modes in sorted(ja_groups(A, job).items()):
        ent = {"S_eng": {}, "gpu": {}}
        forms = {}
        for mode, key in modes.items():
            tab = tabs[key]
            stt, co, form = _coef_from_fit(fits.get(key))
            ent["S_eng"][mode] = {"status": stt, "a": co[0] if co else None, "D_real": tab.get("D_real"), "coef": co}
            forms[mode] = form
            gs, gco, gform = _gpu_fit(tab, mode)
            ent["gpu"][mode] = {"status": gs, "a": gco[0] if gco else None, "D_real": tab.get("D_real"), "coef": gco}
        r = {e: aj1_eval(v) for e, v in ent.items()}
        split = {}
        for sub in ("i", "ii"):
            ls, lg = r["S_eng"][sub]["label"], r["gpu"][sub]["label"]
            if ls.startswith("NA") or lg.startswith("NA"):
                split[sub] = {"label": None, "reason": f"S_eng {ls} / gpu {lg}: no comparison"}
            else:
                split[sub] = {"label": "BWD_ESTIMATOR_SPLIT" if ls != lg else None, "S_eng": ls, "gpu": lg}
        any_split = any(v["label"] for v in split.values())
        res[f"{job}|{rnd}|{tag}"] = {"S_eng": r["S_eng"], "gpu_ms": r["gpu"], "estimator_split": split,
                                     "S_eng_label_citable": not any_split, "chosen_form": forms,
                                     "coef_S_eng": {m: e["coef"] for m, e in ent["S_eng"].items()},
                                     "coef_gpu_ms": {m: e["coef"] for m, e in ent["gpu"].items()}}
    return res


def aj1_cross(g1, g2):
    """BWD_UNRESOLVED across J-A1 / J-A2 on the S_eng labels (values listed, never pooled)."""
    out = {}
    for sub in ("i", "ii"):
        l1, l2 = (g1 or {}).get("S_eng", {}).get(sub, {}).get("label"), (g2 or {}).get("S_eng", {}).get(sub, {}).get("label")
        if not l1 or not l2 or l1.startswith("NA") or l2.startswith("NA"):
            out[sub] = {"label": None, "J-A1": l1, "J-A2": l2, "reason": "NA on a side: no two-Job comparison"}
        else:
            out[sub] = {"label": "BWD_UNRESOLVED" if l1 != l2 else None, "J-A1": l1, "J-A2": l2}
    return out


# ============================================================================ AJ-2
def knee_prior_tokens():
    a = AT.nano9b_v2(4, 27)
    return 2 * AT.state_bytes_per_request(a) / AT.kv_bytes_per_token(a)


def aj2_job(A, job):
    prior = knee_prior_tokens()
    fits = (A.get("fits") or {}).get("t1") or {}
    out = {}
    for (rnd, tag), modes in sorted(ja_groups(A, job).items()):
        rows = {}
        for mode, key in sorted(modes.items()):
            stt, co, _ = _coef_from_fit(fits.get(key))
            if stt:
                rows[mode] = {"status": stt}
                continue
            b, c = co[1], co[2]
            ck = b / c if c else None
            rows[mode] = {"b_ms_per_req": b, "c_ms_per_req_token": c, "c_knee_tokens": ck,
                          "ratio_to_prior": (ck / prior) if ck is not None else None}
        out[f"{job}|{rnd}|{tag}"] = {"prior_2s_over_k_tokens": prior, "rows": rows, "descriptive": True}
    return out


def coef_unstable(x1, x2):
    """I-7: sign differs, a zero, or |x1 - x2| / min(|x1|, |x2|) > 0.5."""
    if x1 == 0 or x2 == 0 or (x1 > 0) != (x2 > 0):
        return True
    return abs(x1 - x2) / min(abs(x1), abs(x2)) > KNEE_UNSTABLE_REL


def aj2_cross(r1, r2):
    out = {}
    rows1, rows2 = (r1 or {}).get("rows", {}), (r2 or {}).get("rows", {})
    for m in sorted(set(rows1) & set(rows2)):
        a, b = rows1[m], rows2[m]
        if a.get("status") or b.get("status"):
            out[m] = {"tag": None, "reason": "NA on a side"}
            continue
        ub = coef_unstable(a["b_ms_per_req"], b["b_ms_per_req"])
        uc = coef_unstable(a["c_ms_per_req_token"], b["c_ms_per_req_token"])
        out[m] = {"tag": "KNEE_COEF_UNSTABLE" if (ub or uc) else None, "b_unstable": ub, "c_unstable": uc}
    return out


# ============================================================================ J-H1 inputs (AJ-2 side table, AJ-4)
def _jh1_key(tab, n, nparts, cap=None):
    ks = []
    for k in tab:
        p = _split_key(k)
        if len(p) == nparts and p[0] == "J-H1" and p[2] == f"dummy_a{n}" and p[3] == "108v" and (cap is None or p[4] == cap):
            ks.append(k)
    return ks[0] if len(ks) == 1 else None


def jh1_t1_cells(A, n):
    tab = A.get("tables", {}).get("t1") or {}
    k = _jh1_key(tab, n, 5, CAP_MAIN)
    return (tab[k] if k else None)


def jh1_t2_points(A, n):
    tab = A.get("tables", {}).get("t2") or {}
    k = _jh1_key(tab, n, 4)
    return ({int(L): t for L, t in (tab[k].get("points") or {}).items() if t is not None} if k else None)


def windep(A, term):
    return (((A.get("jh1") or {}).get("w_indep") or {}).get(term) or {}).get("label")


def aj2_jh1(A):
    """b(n) of the per-variant F1 on the common 5 cells, slope of b on n_M, b(a31) (descriptive, J-H1 only)."""
    wl = windep(A, "T1")
    if wl != "WEIGHT_INDEP_HELD":
        return {"status": f"NOT_EVALUATED (T1 W-INDEP {wl})"}
    b, na_form = {}, []
    fits_t1 = (A.get("fits") or {}).get("t1") or {}
    for n in VARIANTS:
        t = jh1_t1_cells(A, n)
        if not t:
            b[n] = None
            continue
        # §1-9 공통 규칙(result audit r1, 2026-10-27): the harness T1 fit of this variant is FORM_INADEQUATE_* -> NA_FORM
        fk = _jh1_key(fits_t1, n, 5, CAP_MAIN)
        if fk and str((fits_t1[fk] or {}).get("label") or "").startswith("FORM_INADEQUATE"):
            b[n] = None
            na_form.append(f"a{n}")
            continue
        pts = []
        for bs, c in JH_COMMON_CELLS:
            ck = f"{bs}x{c}"
            if (t.get("cells") or {}).get(ck) is not None and (t.get("ctx_actual") or {}).get(ck) is not None:
                pts.append((bs, t["ctx_actual"][ck], t["cells"][ck], c))
        ft = F.fit_form(pts, "F1") if len(pts) == len(JH_COMMON_CELLS) else None
        b[n] = ft["coef"][1] if ft else None
    full = all(b[n] is not None for n in VARIANTS)
    o = _ols([N_LAYERS_AM - n for n in VARIANTS], [b[n] for n in VARIANTS]) if full else None
    return {"b_by_variant": {f"a{n}": ("NA_FORM" if f"a{n}" in na_form else v) for n, v in b.items()},
            "slope_b_vs_nM": o[1] if o else None, "b_a31": b[31], "na_form": na_form,
            "status": "OK" if full else ("NA_FORM" if na_form else "NA_POINTS"), "descriptive": True}


# ============================================================================ AJ-3
def label_theta(th):
    if th < THETA_LO:
        return "T2P_COEXIST_EXP_SUB_SCOPED"
    if th > THETA_HI:
        return "T2P_COEXIST_EXP_SUPER_SCOPED"
    return "T2P_COEXIST_EXP_LINEAR_SCOPED"


def aj3_job(A, job):
    t2p = A.get("tables", {}).get("t2prime") or {}
    t1 = A.get("tables", {}).get("t1") or {}
    groups = {}
    for k, v in t2p.items():
        p = _split_key(k)
        if len(p) == 4 and p[0] == job and p[3].startswith("D"):
            groups.setdefault((p[1], p[2]), {})[p[3]] = v
    out = {}
    for (rnd, tag), byd in sorted(groups.items()):
        ref = byd.get(f"D{D_REF_AJ3}")
        nonref = [f"D{d}" for d in D_STICKY if d != D_REF_AJ3]

        def conf(mode, L):
            v = byd.get(mode)
            return None if not v else (v.get("points_confirmed") or {}).get(str(L))

        def lag(mode, L):
            s = ((t1.get(f"{job}|{rnd}|{tag}|{mode}|{CAP_MAIN}") or {}).get("cells") or {}).get(LAG_T1_CELL)
            t = conf(mode, L)
            return (2 * s / t) if (s is not None and t) else None

        r = {"P_max": (ref or {}).get("P"), "P": {m: (byd.get(m) or {}).get("P") for m in nonref}}
        missing = [f"{m}@L{L}" for L in AJ3_L for m in [f"D{D_REF_AJ3}"] + nonref
                   if conf(m, L) is None or not (byd.get(m) or {}).get("P")]
        if missing:
            r.update(label=None, status="NA_POINTS", missing=missing)
        else:
            xs, ys, lags = [], [], {}
            for L in AJ3_L:
                lags[f"D{D_REF_AJ3}@L{L}"] = lag(f"D{D_REF_AJ3}", L)
                for m in nonref:
                    xs.append(math.log(r["P_max"] / byd[m]["P"]))
                    ys.append(math.log(conf(m, L) / conf(f"D{D_REF_AJ3}", L)))
                    lags[f"{m}@L{L}"] = lag(m, L)
            th = _origin_slope(xs, ys)
            r.update(theta_prime=th, n_points=len(xs), lag_share=lags)
            if th is None:
                r.update(label=None, status="NA_POINTS", reason="P_max == P for all points")
            elif any(v is None for v in lags.values()):
                r.update(label=None, status="LAG_NA", reason="same-boot T1 (1,2k) S_eng missing: lag bound unchecked")
            elif any(v > LAG_SHARE_MAX for v in lags.values()):
                r.update(label=None, status="THETA_LAG_BOUND_EXCEEDED")
            else:
                r.update(label=label_theta(th), status="OK")
        desc = {}
        for L in T2P_L_GRID:
            t0 = conf(f"D{D_REF_AJ3}", L)
            xs, ys = [], []
            for m in nonref:
                t, P = conf(m, L), (byd.get(m) or {}).get("P")
                if t0 and t and P and r["P_max"]:
                    xs.append(math.log(r["P_max"] / P))
                    ys.append(math.log(t / t0))
            o = _ols(xs, ys) if len(xs) >= 2 else None
            desc[str(L)] = {"theta_origin": _origin_slope(xs, ys) if xs else None,
                            "slope_free_intercept": o[1] if o else None, "n": len(xs)}
        r["per_L_descriptive"] = desc
        out[f"{job}|{rnd}|{tag}"] = r
    return out


def aj3_cross(r1, r2):
    l1, l2 = (r1 or {}).get("label"), (r2 or {}).get("label")
    if l1 and l2:
        return {"label": "THETA_UNRESOLVED" if l1 != l2 else None, "J-A1": l1, "J-A2": l2,
                "theta_prime": {"J-A1": r1.get("theta_prime"), "J-A2": r2.get("theta_prime")}, "pooled": False}
    if l1 or l2:
        return {"label": "THETA_SINGLE_JOB", "J-A1": l1, "J-A2": l2}
    return {"label": None, "J-A1": l1, "J-A2": l2, "reason": "both NA"}


# ============================================================================ AJ-4
def _arch(n):
    return AT.nano9b_v2(n, N_LAYERS_AM - n)


def ideal_prefill_ms(n, L):
    return AT.prefill_flops(_arch(n), L)["total"] / PEAK_FLOPS * 1e3


def ideal_decode_ms(n, bs, c):
    d = AT.decode_step(_arch(n), bs, c)
    comp, mem = d["flops"] / PEAK_FLOPS * 1e3, d["bytes_total"] / HBM_BPS * 1e3
    return max(comp, mem), comp, mem


def delta_ideal_prefill(L):
    return _ols(list(VARIANTS), [ideal_prefill_ms(n, L) for n in VARIANTS])[1]


def delta_ideal_decode(bs, c):
    vals = [ideal_decode_ms(n, bs, c) for n in VARIANTS]
    s = lambda i: _ols(list(VARIANTS), [v[i] for v in vals])[1]   # noqa: E731
    return {"roofline": s(0), "compute": s(1), "memory": s(2)}


def regress4(vals):
    """vals = {n_A: T}; all four variants required (no 3-point regression)."""
    if any(vals.get(n) is None for n in VARIANTS):
        return {"status": "NA_POINTS", "missing": [f"a{n}" for n in VARIANTS if vals.get(n) is None]}
    b0, b1, res, r2 = _ols(list(VARIANTS), [vals[n] for n in VARIANTS])
    return {"status": "OK", "beta0": b0, "beta1": b1, "resid": dict(zip((f"a{n}" for n in VARIANTS), res)), "r2": r2}


def aj4(A):
    out = {"W_INDEP": {"T1": windep(A, "T1"), "T2": windep(A, "T2")}, "constants": {"peak_flops": PEAK_FLOPS, "hbm_Bps": HBM_BPS,
           "analytic": AT.VERSION}}
    if out["W_INDEP"]["T2"] == "WEIGHT_INDEP_HELD":
        pts = {n: jh1_t2_points(A, n) for n in VARIANTS}
        t2 = {}
        for L in JH_T2_L:
            r = regress4({n: (pts[n] or {}).get(L) for n in VARIANTS})
            d = delta_ideal_prefill(L)
            r["delta_ideal_ms"] = d
            r["ratio_beta1_to_ideal"] = (r["beta1"] / d) if r["status"] == "OK" and d else None
            t2[str(L)] = r
        out["T2"] = t2
    else:
        out["T2"] = {"status": f"NOT_EVALUATED (T2 W-INDEP {out['W_INDEP']['T2']})"}
    if out["W_INDEP"]["T1"] == "WEIGHT_INDEP_HELD":
        tabs = {n: jh1_t1_cells(A, n) for n in VARIANTS}
        t1 = {}
        for bs, c in JH_COMMON_CELLS:
            ck = f"{bs}x{c}"
            r = regress4({n: ((tabs[n] or {}).get("cells") or {}).get(ck) for n in VARIANTS})
            d = delta_ideal_decode(bs, c)
            r["delta_ideal_ms"] = d
            r["ratio_beta1_to_ideal"] = (r["beta1"] / d["roofline"]) if r["status"] == "OK" and d["roofline"] else None
            r["level_includes_weight_term"] = True
            t1[ck] = r
        dd = {}
        for bs in sorted({b for b, _ in JH_COMMON_CELLS}):
            cs = sorted(c for b, c in JH_COMMON_CELLS if b == bs)
            base = t1[f"{bs}x{cs[0]}"]
            for c in cs[1:]:
                cur = t1[f"{bs}x{c}"]
                ok = base["status"] == "OK" and cur["status"] == "OK"
                di = cur["delta_ideal_ms"]["roofline"] - base["delta_ideal_ms"]["roofline"]
                dd[f"{bs}x{c}-{bs}x{cs[0]}"] = {"beta1_diff": (cur["beta1"] - base["beta1"]) if ok else None,
                                                "delta_ideal_diff_ms": di, "status": "OK" if ok else "NA_POINTS"}
        out["T1"] = t1
        out["T1_ctx_diff_of_diff"] = dd
    else:
        out["T1"] = {"status": f"NOT_EVALUATED (T1 W-INDEP {out['W_INDEP']['T1']})"}
    return out


# ============================================================================ driver
def run(ja1=None, ja2=None, jh1=None):
    out = {"version": VERSION, "analytic": AT.VERSION, "interpretations": INTERPRETATIONS, "scope": SCOPE,
           "inputs_smoke": {k: bool((v or {}).get("smoke")) for k, v in (("J-A1", ja1), ("J-A2", ja2), ("J-H1", jh1)) if v}}
    ja = {j: A for j, A in (("J-A1", ja1), ("J-A2", ja2)) if A}
    aj1 = {j: aj1_job(A, j) for j, A in ja.items()}
    aj2 = {j: aj2_job(A, j) for j, A in ja.items()}
    aj3 = {j: aj3_job(A, j) for j, A in ja.items()}

    def only(d):
        return next(iter(d.values())) if d and len(d) == 1 else None    # one (round, tag) group per Job expected

    out["AJ-1"] = {"per_job": aj1, "cross_job": aj1_cross(only(aj1.get("J-A1")), only(aj1.get("J-A2")))}
    out["AJ-2"] = {"per_job": aj2, "cross_job": aj2_cross(only(aj2.get("J-A1")), only(aj2.get("J-A2"))),
                   "J-H1": aj2_jh1(jh1) if jh1 else None}
    out["AJ-3"] = {"per_job": aj3, "cross_job": aj3_cross(only(aj3.get("J-A1")), only(aj3.get("J-A2")))}
    out["AJ-4"] = aj4(jh1) if jh1 else None
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ja1")
    ap.add_argument("--ja2")
    ap.add_argument("--jh1")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    res = run(_load(a.ja1), _load(a.ja2), _load(a.jh1))
    txt = json.dumps(res, indent=1, sort_keys=True, default=str)
    if a.out:
        Path(a.out).write_text(txt)
    else:
        print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
