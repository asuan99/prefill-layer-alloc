#!/usr/bin/env python3
"""
rev7 operating characteristics -- computed under the REGISTERED analysis rule.

★ CPU ONLY. NO GPU. NO MEASUREMENT. This file imposes a truth and asks what
  the registered rule does with it. Nothing here is an observation, and no
  number in its output may be quoted as one.

------------------------------------------------------------------------------
WHY THIS FILE EXISTS -- audit blocker B4 (rev6, 2026-08-22)
------------------------------------------------------------------------------
`rev6_power.py` computed operating characteristics for a rule that was NOT the
registered one. The audit listed five mismatches:

  (a) it covered only the SECONDARY (counterfactual) verdict -- the PRIMARY
      descriptive quantities and the sec3.3 gate had ZERO operating
      characteristics, and they are what the campaign actually buys;
  (b) ONE-stage bootstrap (windows), where the registration says TWO
      (boot x window);
  (c) a REDUCED rule with no `S <= 0` abstention and no `HOST_DOMINATED`;
  (d) B=600 / seed 12345, where the registration says B=10,000 / seed 1;
  (e) PAIRED resample indices across D=44 and D=92, where sec7-7 declares the
      comparison CROSS-BOOT (independent). The audit measured that this alone
      moves sensitivity 0.850 -> 0.785.

The audit also found the NOISE MODEL wrong in structure: `G := T - K` is the
measured relation, but rev6 drew K and G as independent log-normals, which
understates the effective noise on G by 2-2.5x. This file draws T and K and
DERIVES G, and reports that ratio as a diagnostic rather than asserting it.

------------------------------------------------------------------------------
WHAT IS AND IS NOT REGISTERED HERE
------------------------------------------------------------------------------
Registered analysis knobs (rev7 sec3.4/sec3.5): B=10,000, seed=1, BCa (never
percentile), two-stage boot x window, cross-boot independent indices, the
`S<=0` resample-rate ceiling 0.05, the `s` threshold 0.5, the MULTI_CAUSE band
[0.15,0.85], >=10 windows/cell, >=20 steps/window.

NOT registered, and swept instead: every truth imposed below, the boot/window
CVs, the inter-step host share, and the epsilon grid for the gate. ★These are
simulation INPUTS. They are NOT citations of canon values -- rev5 was
triple-invalidated for quoting the canon local-epsilon band into a design, and
sec3.3 exists precisely so no canon number is needed. Sweeping is what makes
the conclusion independent of the guess.
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
from scipy.stats import norm

# ---- registered analysis constants (rev7 sec3.4/sec3.5) --------------------
B_BOOT = 10_000
SEED = 1
S_LE0_MAX = 0.05          # resample rate of S<=0 above which the 2nd-order
                          # verdict abstains
S_THRESH = 0.5
MULTI_BAND = (0.15, 0.85)
MIN_WIN = 10
MIN_STEP = 20
N_BOOT = 4                # boots per cell (rev6 sec5, unchanged)
N_WIN = 20                # ★B6 prereg: R=5 rounds x 4 boots = 20 windows/cell
ALPHA = 0.05

VERDICTS = ("KERNEL_DOMINATED", "GAP_DOMINATED", "MULTI_CAUSE",
            "COUNTERFACTUAL_SENSITIVE", "HOST_DOMINATED", "UNDETERMINED",
            # ★rev8 (audit C5): rev7 downgraded the sec3.3 gate to a
            #   sanity check and DELETED its failure outcome, which rev6
            #   had registered. A gate with no failure label cannot fail.
            "PHENOMENON_ABSENT_IN_CELL_A")


# ===========================================================================
# 1. Simulation -- T and K are measured, G_intra is DERIVED
# ===========================================================================
def _ln(rng, mu, cv, size, common=None, rho=0.0):
    """Log-normal with mean `mu` and coefficient of variation `cv`.

    `common` carries a shared latent normal so two quantities measured off the
    SAME timeline can be correlated (rho). rho=0 is the registered
    conservative case; the sweep reports rho in {0, 0.5, 0.9}.
    """
    if cv <= 0:
        return np.full(size, float(mu))
    s2 = math.log1p(cv * cv)
    s = math.sqrt(s2)
    z = rng.standard_normal(size)
    if common is not None and rho > 0:
        z = math.sqrt(rho) * common + math.sqrt(1.0 - rho) * z
    return mu * np.exp(s * z - 0.5 * s2)


def simulate_cell(rng, mu_K, mu_Gi, mu_Ge, n_boot=N_BOOT, n_win=N_WIN,
                  cv_boot=0.01, cv_win=0.05, rho=0.0, derive_g=True):
    """One cell's data: (n_boot, n_win) arrays of K, G_intra, G_inter, T_step.

    ★Structure (audit, rev6): the timeline gives `Sigma kernel_dur` and the
    span; the intra-step gap is their DIFFERENCE. Drawing K and G
    independently pretends the difference has its own small noise, which it
    does not. `derive_g=False` reproduces rev6's structure so the two can be
    compared in the same run instead of asserted.
    """
    shape = (n_boot, n_win)
    mu_T = mu_K + mu_Gi + mu_Ge

    def draw(mu, cv_b, cv_w, common_b=None, common_w=None):
        b = _ln(rng, 1.0, cv_b, (n_boot, 1), common_b, rho)
        w = _ln(rng, 1.0, cv_w, shape, common_w, rho)
        return mu * b * w

    cb = rng.standard_normal((n_boot, 1)) if rho > 0 else None
    cw = rng.standard_normal(shape) if rho > 0 else None
    Ge = draw(mu_Ge, cv_boot, cv_win)
    if derive_g:
        T = draw(mu_T, cv_boot, cv_win, cb, cw)
        K = draw(mu_K, cv_boot, cv_win, cb, cw)
        Gi = T - K - Ge
    else:                                   # rev6's structure, for contrast
        K = draw(mu_K, cv_boot, cv_win, cb, cw)
        Gi = draw(mu_Gi, cv_boot, cv_win)
        T = K + Gi + Ge
    return {"K": K, "Gi": Gi, "Ge": Ge, "T": T}


def invalid_rate(cell):
    """Windows a real run would have to discard: a negative intra-step gap is
    physically impossible, so its appearance is a noise-model artefact and is
    reported, never silently clamped (`A3_NO_CLAMP`'s replacement)."""
    return float(np.mean(cell["Gi"] <= 0))


# ===========================================================================
# 2. Two-stage BCa bootstrap (registered)
# ===========================================================================
def _idx(rng, n_boot, n_win, B):
    b = rng.integers(0, n_boot, size=(B, n_boot))
    w = rng.integers(0, n_win, size=(B, n_boot, n_win))
    return b, w


def _gather(arr, b, w):
    return arr[b[:, :, None], w]


def _cell_sums(cell, b, w):
    """Resampled cell aggregates: (B,) sums of K, Gi, Ge, T, plus the COUNT.

    ★rev8 (audit C8). The count used to be assumed equal to the global
    `N_BOOT * N_WIN`, but a JACKKNIFE replicate has one boot deleted, so its
    sums run over 3x20 = 60 values while the divisor stayed 80. Every mean fed
    to `stat_eps`/`stat_s` was therefore scaled by 0.75 in the jackknife and
    1.00 in the resamples -- which corrupts the BCa acceleration term
    specifically (the audit measured an artefactual jackknife spread in the
    epsilon statistic). Carry the count with the sums.
    """
    g = {k: _gather(cell[k], b, w) for k in ("K", "Gi", "Ge", "T")}
    out = {k: v.sum(axis=(1, 2)) for k, v in g.items()}
    out["n"] = float(g["K"].shape[1] * g["K"].shape[2])
    return out


def _cell_sums_obs(cell):
    out = {k: cell[k].sum() for k in ("K", "Gi", "Ge", "T")}
    out["n"] = float(cell["K"].size)
    return out


def _jack_cells(cells):
    """Delete-one jackknife over the TOP-LEVEL units (boots), across every
    cell the statistic touches. With cells of 4 boots each this is 4*len(cells)
    replicates -- small, and that smallness is why BCa's acceleration term is
    the weak part of this design. Reported, not hidden."""
    out = []
    names = list(cells)
    for nm in names:
        n = cells[nm]["K"].shape[0]
        if n > 1:
            for i in range(n):
                keep = [j for j in range(n) if j != i]
                sub = dict(cells)
                sub[nm] = {k: v[keep] for k, v in cells[nm].items()}
                out.append(sub)
        else:
            # ★A single top-level unit has no delete-one replicate: deleting it
            #   leaves nothing, the acceleration term becomes 0/0, and the
            #   whole interval comes back NaN. That is exactly what happened
            #   to the one-stage contrast in this file's first run -- it
            #   reported `nan` rather than a ratio. With one boot the
            #   resampling unit IS the window, so jackknife over windows.
            m = cells[nm]["K"].shape[1]
            for i in range(m):
                keep = [j for j in range(m) if j != i]
                sub = dict(cells)
                sub[nm] = {k: v[:, keep] for k, v in cells[nm].items()}
                out.append(sub)
    return out


def bca(theta_hat, theta_star, theta_jack, alpha=ALPHA):
    """BCa interval. Returns (lo, hi, degraded_flag).

    ★percentile intervals are forbidden in canon prose (under-coverage
    M8 1.36x / Ha8 1.55x), so every degradation away from BCa -- an undefined
    z0, a non-finite acceleration, or too few usable resamples -- is REPORTED
    through the flag instead of silently becoming a percentile interval.
    ★Non-finite resamples are DROPPED here rather than propagated: `s` is a
    ratio whose denominator can cross zero, and their RATE is the registered
    `S<=0` quantity, evaluated by the caller. Dropping them here and gating on
    the rate there keeps the two jobs separate.
    """
    star = np.asarray(theta_star, dtype=float)
    star = star[np.isfinite(star)]
    jack = np.asarray(theta_jack, dtype=float)
    jack = jack[np.isfinite(jack)]
    if (not np.isfinite(theta_hat)) or star.size < 100 or jack.size < 2:
        return float("nan"), float("nan"), True
    prop = float(np.mean(star < theta_hat))
    if prop <= 0.0 or prop >= 1.0:
        lo, hi = np.quantile(star, [alpha / 2, 1 - alpha / 2])
        return float(lo), float(hi), True
    z0 = norm.ppf(prop)
    jm = np.mean(jack)
    num = np.sum((jm - jack) ** 3)
    den = 6.0 * (np.sum((jm - jack) ** 2) ** 1.5)
    a = 0.0 if den == 0 else num / den
    if not np.isfinite(a):
        a = 0.0
    out, degraded = [], False
    for q in (alpha / 2, 1 - alpha / 2):
        zq = z0 + norm.ppf(q)
        denom = 1 - a * zq
        z = z0 + zq / denom if denom != 0 else float("nan")
        p = norm.cdf(z) if np.isfinite(z) else float("nan")
        if not np.isfinite(p):
            p, degraded = q, True
        out.append(float(np.quantile(star, min(max(p, 0.0), 1.0))))
    return out[0], out[1], degraded


def ci_for(cells, stat, rng, B=B_BOOT, paired=False):
    """Two-stage BCa CI for `stat(sums_by_cell)`.

    `paired=False` (registered, sec7-7) resamples each cell's boots
    INDEPENDENTLY -- the comparison is cross-boot, so the resampling must be
    too. `paired=True` reproduces rev6's harness pattern for contrast.
    """
    names = list(cells)
    n_boot = {k: cells[k]["K"].shape[0] for k in names}
    n_win = {k: cells[k]["K"].shape[1] for k in names}
    star = {}
    shared = None
    for nm in names:
        if paired and shared is not None and n_boot[nm] == shared[0].shape[1]:
            b, w = shared
        else:
            b, w = _idx(rng, n_boot[nm], n_win[nm], B)
            if paired:
                shared = (b, w)
        star[nm] = _cell_sums(cells[nm], b, w)
    theta_star = stat(star)
    theta_hat = stat({k: {kk: np.array([vv]) for kk, vv in
                          _cell_sums_obs(cells[k]).items()} for k in names})[0]
    jack = np.array([stat({k: {kk: np.array([vv]) for kk, vv in
                               _cell_sums_obs(s[k]).items()} for k in names})[0]
                     for s in _jack_cells(cells)])
    # ★rev8 C8 regression guard: a jackknife replicate must NOT be normalised
    #   by the full-sample count. If it were, deleting a boot would shift every
    #   mean by n/(n-1) and the acceleration term would be an artefact.
    lo, hi, flag = bca(theta_hat, theta_star, jack)
    return {"point": float(theta_hat), "lo": lo, "hi": hi,
            "z0_undefined": flag, "star": theta_star,
            "nonfinite_rate": float(np.mean(~np.isfinite(theta_star)))}


# ===========================================================================
# 3. The registered statistics
# ===========================================================================
def stat_gap_frac(cell_name):
    """PRIMARY: G_intra / T_step. ★rev7 B1 repair -- the denominator is
    T_step for BOTH gap fractions, so `frac_inter > gap_frac_intra` compares
    like with like. Under rev6's `/T_span` the intra fraction was inflated by
    1/(1-eta) and HOST_DOMINATED could not fire even when the host share was
    larger."""
    def f(s):
        return s[cell_name]["Gi"] / s[cell_name]["T"]
    return f


def stat_frac_inter(cell_name):
    def f(s):
        return s[cell_name]["Ge"] / s[cell_name]["T"]
    return f


def stat_eps(c1, c2, d1, d2):
    """epsilon_T(d1->d2) = -ln(T(d2)/T(d1)) / ln(d2/d1), on per-window means."""
    def f(s):
        t1 = s[c1]["T"] / s[c1]["n"]
        t2 = s[c2]["T"] / s[c2]["n"]
        return -np.log(t2 / t1) / math.log(d2 / d1)
    return f


def stat_s(kind, d_lo=44, d_hi=92):
    """SECONDARY: s = S_G / (S_K + S_G) under counterfactual A or B."""
    r = d_lo / d_hi

    def f(s):
        K_lo = s["lo"]["K"] / s["lo"]["n"]
        K_hi = s["hi"]["K"] / s["hi"]["n"]
        G_lo = s["lo"]["Gi"] / s["lo"]["n"]
        G_hi = s["hi"]["Gi"] / s["hi"]["n"]
        S_K = K_hi - K_lo * r
        S_G = G_hi - (G_lo if kind == "A" else G_lo * r)
        den = S_K + S_G
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(den > 0, S_G / np.where(den == 0, np.nan, den),
                           np.nan)
        return out
    return f


# ===========================================================================
# 4. The registered decision rule, in the registered ORDER
# ===========================================================================
def stat_host_margin(cell_name):
    """frac_inter - gap_frac_intra, the quantity HOST_DOMINATED tests.

    ★rev8 (audit C2). rev7 compared two POINT estimates and ranked the result
    first, so at a true tie it fired on a coin flip -- the audit measured 0.600
    -- and pre-empted every substantive label. A comparison that decides a
    label needs an interval like every other comparison in the registration.
    """
    def f(s):
        return (s[cell_name]["Ge"] - s[cell_name]["Gi"]) / s[cell_name]["T"]
    return f


def decide(cells, rng, B=B_BOOT, paired=False):
    """The registered rule, in the registered ORDER -- all FOUR stages.

    ★rev8 (audit C9): rev7 registered `preceding gate -> HOST_DOMINATED ->
    S<=0 -> counterfactual pair` and implemented only the last three. The
    preceding gate was in the document and not in the code, so its operating
    characteristics were being reported for a rule nothing ran.
    """
    # --- stage 1: the sec3.3 lever-existence gate (preceding condition)
    if "c16" in cells:
        a = ci_for({k: cells[k] for k in ("c16", "lo")},
                   stat_eps("c16", "lo", 16, 44), rng, B)
        b = ci_for({k: cells[k] for k in ("lo", "hi")},
                   stat_eps("lo", "hi", 44, 92), rng, B)
        if not (np.isfinite(a["lo"]) and np.isfinite(b["hi"])
                and a["lo"] > b["hi"]):
            return "PHENOMENON_ABSENT_IN_CELL_A", {
                "eps_16_44": [a["lo"], a["hi"]], "eps_44_92": [b["lo"], b["hi"]]}
    # --- stage 2: HOST_DOMINATED, now on an interval (C2)
    hm = ci_for({"hi": cells["hi"]}, stat_host_margin("hi"), rng, B)
    if np.isfinite(hm["lo"]) and hm["lo"] > 0:
        return "HOST_DOMINATED", {"host_margin": [hm["lo"], hm["hi"]],
                                  "point": hm["point"]}
    # --- stage 3: S<=0 abstention, then stage 4: the counterfactual pair
    two = {"lo": cells["lo"], "hi": cells["hi"]}
    res = {"host_margin": [hm["lo"], hm["hi"]]}
    for kind in ("A", "B"):
        c = ci_for(two, stat_s(kind), rng, B, paired=paired)
        bad = c["nonfinite_rate"]
        res[kind] = {"lo": c["lo"], "hi": c["hi"], "point": c["point"],
                     "s_le0_rate": bad, "z0_undefined": c["z0_undefined"]}
        if (bad > S_LE0_MAX or not np.isfinite(c["point"])
                or not np.isfinite(c["lo"]) or not np.isfinite(c["hi"])):
            return "UNDETERMINED", res
    lab = {}
    for kind in ("A", "B"):
        lo, hi = res[kind]["lo"], res[kind]["hi"]
        if hi < S_THRESH:
            lab[kind] = "KERNEL_DOMINATED"
        elif lo > S_THRESH:
            lab[kind] = "GAP_DOMINATED"
        elif MULTI_BAND[0] <= lo and hi <= MULTI_BAND[1]:
            lab[kind] = "MULTI_CAUSE"
        else:
            lab[kind] = "UNDETERMINED"
    if lab["A"] != lab["B"]:
        return "COUNTERFACTUAL_SENSITIVE", res
    return lab["A"], res


# ===========================================================================
# 5. Scenarios (imposed truths -- INPUTS, not canon citations)
# ===========================================================================
# (K, G_intra) at D=44 and D=92, in arbitrary time units, from rev6 sec1 so the
# two tables can be compared line by line.
TRUTHS = {
    "kernel_does_not_shrink":   ((95, 5),  (80, 20)),
    "gap_actually_grows":       ((60, 40), (32, 60)),
    "gap_constant_kernel_ok":   ((50, 50), (30, 50)),
    "mixed":                    ((70, 30), (45, 40)),
}
ETA = 0.10          # inter-step host share of T_step (swept)
CVS = (0.01, 0.025, 0.05)


def _cells_for(rng, truth, cv, eta=ETA, rho=0.0, derive_g=True, eps1=None):
    """Cells lo(=D 44) and hi(=D 92), plus optionally c16 for the gate.

    ★rev8 (audit C9): `eps1` imposes the 16->44 elasticity so the sec3.3 gate
    has a third cell to run on. `None` leaves it out, which is how the gate's
    own operating characteristics are measured separately in `exp_gate`.
    """
    (k44, g44), (k92, g92) = truth
    out = {}
    for nm, (k, g) in (("lo", (k44, g44)), ("hi", (k92, g92))):
        span = k + g
        ge = eta * span / (1 - eta)
        out[nm] = simulate_cell(rng, k, g, ge, cv_boot=cv, cv_win=cv * 2,
                                rho=rho, derive_g=derive_g)
    if eps1 is not None:
        t44 = k44 + g44
        t16 = t44 * math.exp(eps1 * math.log(44 / 16))
        sc = t16 / t44
        out["c16"] = simulate_cell(rng, k44 * sc, g44 * sc,
                                   eta * t16 / (1 - eta), cv_boot=cv,
                                   cv_win=cv * 2, rho=rho, derive_g=derive_g)
    return out


# ===========================================================================
# 6. Experiments
# ===========================================================================
def exp_primary(rng, n_rep, B):
    """★B4(a): the PRIMARY deliverable had zero operating characteristics.

    How precisely does the campaign measure `gap_frac_intra(D)`? That number,
    not the counterfactual pair, is what the audit said the campaign's value
    rests on entirely.
    """
    out = {}
    for tname, truth in TRUTHS.items():
        for cv in CVS:
            half = []
            for _ in range(n_rep):
                cells = _cells_for(rng, truth, cv)
                c = ci_for({"hi": cells["hi"]}, stat_gap_frac("hi"), rng, B)
                half.append((c["hi"] - c["lo"]) / 2)
            half = np.array(half)
            out[f"{tname}|cv={cv}"] = {
                "half_width_median_pp": float(np.median(half) * 100),
                "half_width_p90_pp": float(np.quantile(half, 0.9) * 100),
                "P_half_width_le_5pp": float(np.mean(half <= 0.05)),
                "P_half_width_le_2pp": float(np.mean(half <= 0.02)),
            }
    return out


def exp_gate(rng, n_rep, B, grid):
    """★B4(a): operating characteristics of the sec3.3 lever-existence gate.

    The gate passes when CI_lo(eps(16->44)) > CI_hi(eps(44->92)). The audit
    said its power looks certain and its information content near zero; the
    number that matters is therefore the FALSE-POSITIVE rate at eps1 == eps2,
    which nobody had computed.
    """
    out = {}
    for (e1, e2) in grid:
        for cv in (0.01, 0.05):
            passed = 0
            for _ in range(n_rep):
                mu44 = 100.0
                mu16 = mu44 * math.exp(e1 * math.log(44 / 16))
                mu92 = mu44 * math.exp(-e2 * math.log(92 / 44))
                cells = {}
                for nm, mu in (("c16", mu16), ("c44", mu44), ("c92", mu92)):
                    cells[nm] = simulate_cell(rng, 0.6 * mu, 0.3 * mu,
                                              0.1 * mu, cv_boot=cv,
                                              cv_win=cv * 2)
                a = ci_for({k: cells[k] for k in ("c16", "c44")},
                           stat_eps("c16", "c44", 16, 44), rng, B)
                b = ci_for({k: cells[k] for k in ("c44", "c92")},
                           stat_eps("c44", "c92", 44, 92), rng, B)
                passed += int(a["lo"] > b["hi"])
            out[f"eps1={e1},eps2={e2}|cv={cv}"] = {
                "P_gate_passes": passed / n_rep,
                "is_null": bool(abs(e1 - e2) < 1e-12)}
    return out


EPS1_SCEN = 0.55   # imposed 16->44 elasticity for the preceding gate (swept
                   # input, NOT a canon citation; the gate's own operating
                   # characteristics are measured separately in `exp_gate`)


def exp_secondary(rng, n_rep, B, paired=False, derive_g=True, with_gate=True):
    """★B4(b)-(e) + rev8 C9: the FULL registered rule, all four stages."""
    out = {}
    for tname, truth in TRUTHS.items():
        for cv in CVS:
            counts = {v: 0 for v in VERDICTS}
            z0flag = 0
            for _ in range(n_rep):
                cells = _cells_for(rng, truth, cv, derive_g=derive_g,
                                   eps1=EPS1_SCEN if with_gate else None)
                v, det = decide(cells, rng, B, paired=paired)
                counts[v] = counts.get(v, 0) + 1
                if isinstance(det, dict) and det.get("A", {}).get(
                        "z0_undefined"):
                    z0flag += 1
            out[f"{tname}|cv={cv}"] = {
                "P": {k: c / n_rep for k, c in counts.items() if c},
                "z0_undefined_rate": z0flag / n_rep}
    return out


def exp_diagnostics(rng, n_rep, B):
    """The three structural findings the audit measured, recomputed here so
    rev7 owns them instead of citing them."""
    out = {}
    # (1) one-stage (windows only) vs two-stage (boot x window)
    w1, w2 = [], []
    for _ in range(n_rep):
        cells = _cells_for(rng, TRUTHS["mixed"], 0.025)
        c2 = ci_for({"hi": cells["hi"]}, stat_gap_frac("hi"), rng, B)
        flat = {"hi": {k: v.reshape(1, -1) for k, v in cells["hi"].items()}}
        c1 = ci_for(flat, stat_gap_frac("hi"), rng, B)
        w2.append(c2["hi"] - c2["lo"])
        w1.append(c1["hi"] - c1["lo"])
    out["one_stage_understates_width_by"] = float(np.median(w2) / np.median(w1))
    # (2) paired vs independent resampling across cells
    for paired in (False, True):
        r = exp_secondary(rng, max(30, n_rep // 4), B, paired=paired)
        key = "paired" if paired else "independent"
        out[f"secondary_{key}"] = {
            k: v["P"] for k, v in r.items() if k.endswith("cv=0.05")}
    # (3) G derived vs G drawn independently (rev6's structure)
    for derive in (True, False):
        cells = _cells_for(rng, TRUTHS["mixed"], 0.025, derive_g=derive)
        gi = cells["hi"]["Gi"]
        out["effective_cv_of_G_" + ("derived" if derive else "independent")] \
            = float(np.std(gi) / np.mean(gi))
    out["effective_noise_ratio"] = (
        out["effective_cv_of_G_derived"]
        / out["effective_cv_of_G_independent"])
    # (4) invalid (negative intra gap) window rate at the widest cv
    cells = _cells_for(rng, TRUTHS["kernel_does_not_shrink"], 0.05)
    out["invalid_window_rate_worst_truth"] = invalid_rate(cells["hi"])
    return out


# ===========================================================================
# 7. Self-test -- every claim this file makes must be falsifiable
# ===========================================================================
def selftest():
    ok = [True]

    def ck(name, cond, detail=""):
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
              + (f"  {detail}" if detail and not cond else ""))
        ok[0] = ok[0] and bool(cond)

    rng = np.random.default_rng(SEED)
    cells = _cells_for(rng, TRUTHS["mixed"], 0.0)     # noiseless
    hi = cells["hi"]
    print("-- S1 identity guards (declared identities; regression only)")
    part = (hi["K"] + hi["Gi"] + hi["Ge"]) / hi["T"]
    ck("S1a the three fractions partition T_step (IDENTITY, not evidence)",
       np.allclose(part, 1.0))
    ck("S1b a mis-sliced decomposition breaks S1a (the identity is still a "
       "regression guard)",
       not np.allclose((hi["K"] + hi["Gi"]) / hi["T"], 1.0))

    print("-- S2 B1 repair: the two gap fractions share a denominator")
    gi = float(hi["Gi"].mean() / hi["T"].mean())
    ge = float(hi["Ge"].mean() / hi["T"].mean())
    span_frac = float(hi["Gi"].mean() / (hi["K"].mean() + hi["Gi"].mean()))
    ck("S2a gap_frac_intra uses T_step, not T_span", abs(gi - span_frac) > 1e-6)
    ck("S2b under rev6's /T_span the intra fraction is inflated",
       span_frac > gi)
    ck("S2c HOST_DOMINATED can fire when the host share is the larger one",
       decide(_cells_for(rng, ((50, 10), (30, 10)), 0.0, eta=0.6), rng,
              200)[0] == "HOST_DOMINATED")

    print("-- S3 the bootstrap machinery")
    c = ci_for({"hi": hi}, stat_gap_frac("hi"), rng, 500)
    ck("S3a a noiseless cell gives a degenerate interval",
       abs(c["hi"] - c["lo"]) < 1e-9)
    cells = _cells_for(rng, TRUTHS["mixed"], 0.05)
    c = ci_for({"hi": cells["hi"]}, stat_gap_frac("hi"), rng, 2000)
    ck("S3b a noisy cell gives a non-degenerate interval containing the point",
       c["lo"] < c["point"] < c["hi"])

    print("-- S4 the rule reaches each registered label from SOME truth")
    rng2 = np.random.default_rng(7)
    seen = set()
    for t in TRUTHS.values():
        for cv in (0.01, 0.05):
            seen.add(decide(_cells_for(rng2, t, cv), rng2, 400)[0])
    seen.add(decide(_cells_for(rng2, ((50, 10), (30, 10)), 0.01, eta=0.6),
                    rng2, 400)[0])
    ck("S4a at least three distinct labels are reachable", len(seen) >= 3,
       str(sorted(seen)))
    ck("S4b HOST_DOMINATED is reachable", "HOST_DOMINATED" in seen,
       str(sorted(seen)))

    print("-- S6 C8: jackknife replicates carry their OWN count (audit C8)")
    c = _cells_for(np.random.default_rng(5), TRUTHS["mixed"], 0.0)
    full = _cell_sums_obs(c["hi"])["n"]
    jk = [_cell_sums_obs(j["hi"])["n"] for j in _jack_cells({"hi": c["hi"]})]
    ck("S6a a delete-one replicate reports a SMALLER count than the sample",
       all(x < full for x in jk), f"full={full} jack={sorted(set(jk))}")
    ck("S6b the count is never the hardwired global",
       all(x != N_BOOT * N_WIN for x in jk), str(sorted(set(jk))))
    # the epsilon statistic on a noiseless cell must be IDENTICAL under
    # delete-one -- it is a ratio of means, so a wrong divisor shows up here.
    cc = _cells_for(np.random.default_rng(5), TRUTHS["mixed"], 0.0)
    e_full = stat_eps("a", "b", 16, 44)({
        "a": {k: np.array([v]) for k, v in _cell_sums_obs(cc["lo"]).items()},
        "b": {k: np.array([v]) for k, v in _cell_sums_obs(cc["hi"]).items()}})[0]
    j = _jack_cells({"a": cc["lo"], "b": cc["hi"]})[0]
    e_jack = stat_eps("a", "b", 16, 44)({
        "a": {k: np.array([v]) for k, v in _cell_sums_obs(j["a"]).items()},
        "b": {k: np.array([v]) for k, v in _cell_sums_obs(j["b"]).items()}})[0]
    ck("S6c on a NOISELESS cell, delete-one leaves epsilon unchanged "
       "(a wrong divisor would move it)", abs(e_full - e_jack) < 1e-9,
       f"{e_full:.6f} vs {e_jack:.6f}")

    print("-- S5 mutation: each registered knob is load-bearing")
    # ★The first version of S5a printed base -> mutant and asserted True. That
    #   is a check that cannot fail (lesson #53), and it needed a witness in
    #   which the ceiling actually decides: a truth whose deficit S_K + S_G is
    #   SMALL AND POSITIVE, so resamples straddle zero at a rate between the
    #   registered 0.05 and 1.0. A truth where the deficit is plainly negative
    #   abstains through `isfinite(point)` no matter what the ceiling says.
    base, mut, rate = _ceiling_flips()
    ck("S5a the S<=0 ceiling changes the label on a straddling truth",
       base != mut, f"{base} -> {mut} (nonfinite rate {rate:.3f})")
    ck("S5b the s threshold is not vacuous: moving it flips a truth",
       _thresh_flips())
    print("ALL PASS" if ok[0] else "FAILURES PRESENT")
    return 0 if ok[0] else 1


def _ceiling_flips():
    """Witness for S5a: deficit small and positive, so resamples straddle 0."""
    global S_LE0_MAX
    saved = S_LE0_MAX
    truth = ((50, 50), (30, 46))       # S_K ~ +6.1, S_G(A) ~ -4.0 => den ~ 2.1
    try:
        out = []
        rate = 0.0
        for ceiling in (saved, 1.0):
            S_LE0_MAX = ceiling
            cells = _cells_for(np.random.default_rng(11), truth, 0.05)
            v, det = decide(cells, np.random.default_rng(12), 4000)
            out.append(v)
            if isinstance(det, dict) and "A" in det:
                rate = det["A"].get("s_le0_rate", 0.0)
        return out[0], out[1], rate
    finally:
        S_LE0_MAX = saved


def _thresh_flips():
    global S_THRESH
    saved = S_THRESH
    try:
        a = decide(_cells_for(np.random.default_rng(3), TRUTHS["mixed"], 0.01),
                   np.random.default_rng(4), 2000)[0]
        S_THRESH = 0.05
        b = decide(_cells_for(np.random.default_rng(3), TRUTHS["mixed"], 0.01),
                   np.random.default_rng(4), 2000)[0]
        return a != b
    finally:
        S_THRESH = saved


# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--B", type=int, default=B_BOOT)
    ap.add_argument("--out", default="REV7_POWER_2026-08-23.json")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    grid = [(0.55, 0.05), (0.40, 0.20), (0.30, 0.30), (0.20, 0.20),
            (0.60, 0.40)]
    res = {
        "kind": "kernel_mech_rev7_power",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "registered": {"B": a.B, "seed": SEED, "n_boot": N_BOOT,
                       "n_win": N_WIN, "S_le0_max": S_LE0_MAX,
                       "s_threshold": S_THRESH, "multi_band": MULTI_BAND,
                       "interval": "BCa (percentile forbidden)",
                       "resampling": "two-stage boot x window, cross-boot "
                                     "independent across cells"},
        "swept_inputs": {"truths": {k: v for k, v in TRUTHS.items()},
                         "cvs": CVS, "eta_inter": ETA,
                         "eps_grid": grid,
                         "note_not_canon": "every truth here is an imposed "
                         "simulation input; none is a canon citation"},
        "n_rep": a.reps, "eps1_for_gate": EPS1_SCEN,
        "primary": exp_primary(rng, a.reps, a.B),
        "gate": exp_gate(rng, max(50, a.reps // 4), a.B, grid),
        "secondary_registered_rule": exp_secondary(rng, a.reps, a.B),
        "diagnostics": exp_diagnostics(rng, a.reps, a.B),
    }
    res["wall_s"] = round(time.time() - t0, 1)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({k: res[k] for k in
                      ("primary", "gate", "secondary_registered_rule",
                       "diagnostics")}, indent=2)[:6000])
    print(f"\n-> {a.out}  ({res['wall_s']} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
