"""Shared machinery for the Gate-2 rev3 submission precondition §11-2:
re-verification of delta=0.05 and TOST power under the tau=40 variance structure.

GPU 0.  Existing data only (873944 Zamba2 / 873945 Granite).

=========================  AGGREGATION UNITS (declared first)  =================
  scoring unit      = REQUEST.  Indicator  1{ max_i(itl_i) > tau }.
                      Requests with an EMPTY itls array are dropped from BOTH the
                      numerator and the denominator (prereg §4.1) and counted in
                      n_missing_itl.
  cell              = (model, arm, rate, rep).  X_tau = mean of the request
                      indicators inside that cell -> ONE SCALAR PER REP.
  replication unit  = REP.  n = 5 per (model, arm, rate).  Paired across arms on
                      the rep index (identical seed = identical prompt set and
                      arrival stream; verified in the C-1/C-2/C-3 pass).
  interval unit     = the 5 paired REP differences.  Requests are never pooled
                      across reps for any interval.
  descriptive-only  = per-token stall probability p = #(gaps > tau)/#(gaps),
                      pooled over the 5 reps of a cell (prereg §4.2 says this is
                      descriptive; it is used here only to interpret the scale of
                      delta, never as a decision variable).
===============================================================================

The t distribution is implemented locally (no scipy in this env) and checked
against the value t_{0.975,4} = 2.7764451051977987 that the C-2 pass inverted
numerically by Simpson integration -- i.e. two independent implementations.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from verify_load import REPS, load_cells  # noqa: E402

# --- the five rev3 decision cells (prereg §2) --------------------------------
DECISION_CELLS = (
    ("zamba2-27b", 2.0),
    ("zamba2-27b", 3.0),
    ("granite-40-h-micro-base", 3.0),
    ("granite-40-h-micro-base", 4.0),   # designated cell for R3' (prereg §5.3)
    ("granite-40-h-micro-base", 6.0),
)
DESIGNATED = ("granite-40-h-micro-base", 4.0)
SHORT = {"zamba2-27b": "Zamba2 r%g", "granite-40-h-micro-base": "Granite r%g"}

# arms actually present in 873944/873945.  A2/A3 DO NOT EXIST YET.
A1 = "plain"
A4 = "agnostic"

TAU_LADDER = (20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 60.0, 80.0, 100.0)
DELTA = 0.05
N_REQ_NOMINAL = 120


def cell_label(model: str, rate: float) -> str:
    return SHORT[model] % rate


# ------------------------------------------------------------------ statistics
def _betacf(a: float, b: float, x: float) -> float:
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
        de = d * c
        h *= de
        if abs(de - 1.0) < 3e-16:
            break
    return h


def betai(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbeta + b * math.log1p(-x) + a * math.log(x)) * _betacf(b, a, 1.0 - x) / b


def t_cdf(t: float, df: int) -> float:
    x = df / (df + t * t)
    tail = 0.5 * betai(df / 2.0, 0.5, x)
    return 1.0 - tail if t > 0 else tail


def t_ppf(q: float, df: int) -> float:
    lo, hi = -200.0, 200.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_cdf(mid, df) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ------------------------------------------------------------------ X_tau
def x_tau_table(cells, taus=TAU_LADDER):
    """-> {(model, arm, rate, rep, tau): dict(x=..., n_used=..., n_missing=..., n_viol=...)}

    plus a pooled per-token stall table {(model, arm, rate, tau): p_token}
    """
    per_rep = {}
    gaps_pool = {}
    for (model, arm, rate, rep), cell in cells.items():
        maxes = []
        n_missing = 0
        all_gaps = []
        for r in cell["requests"]:
            if not r.token_itl_ms:
                n_missing += 1
                continue
            maxes.append(max(r.token_itl_ms))
            all_gaps.extend(r.token_itl_ms)
        maxes = np.asarray(maxes)
        all_gaps = np.asarray(all_gaps)
        gaps_pool.setdefault((model, arm, rate), []).append(all_gaps)
        for tau in taus:
            n_viol = int((maxes > tau).sum())
            per_rep[(model, arm, rate, rep, tau)] = {
                "x": n_viol / len(maxes),
                "n_viol": n_viol,
                "n_used": len(maxes),
                "n_missing_itl": n_missing,
                "n_gaps": int(all_gaps.size),
            }
    p_token = {}
    for key, chunks in gaps_pool.items():
        g = np.concatenate(chunks)
        for tau in taus:
            p_token[key + (tau,)] = float((g > tau).mean())
    return per_rep, p_token


def arm_vector(per_rep, model, arm, rate, tau):
    return np.array([per_rep[(model, arm, rate, rep, tau)]["x"] for rep in REPS])


def descr(v):
    v = np.asarray(v, dtype=float)
    n = v.size
    mean = float(v.mean())
    sd = float(v.std(ddof=1))
    return {
        "mean": mean,
        "sd": sd,
        "cv": (sd / mean) if mean > 0 else float("nan"),
        "min": float(v.min()),
        "max": float(v.max()),
        "n_zero": int((v == 0.0).sum()),
        "n_one": int((v == 1.0).sum()),
        "values": [float(x) for x in v],
        "n": n,
    }


def paired_t_ci(diffs, conf=0.95):
    d = np.asarray(diffs, dtype=float)
    n = d.size
    mean = float(d.mean())
    sd = float(d.std(ddof=1))
    se = sd / math.sqrt(n)
    tc = t_ppf(0.5 + conf / 2.0, n - 1)
    return {"mean": mean, "sd": sd, "se": se, "lo": mean - tc * se, "hi": mean + tc * se,
            "tcrit": tc, "n": n}


# ------------------------------------------------------------------ TOST power
def tost_power(sigma, n, delta, gap, trials=200_000, shape=None, rng=None):
    """P(declare equivalence) for the paired-t TOST at alpha=0.05 per side
    (equivalently: two-sided 90% t interval contained in (-delta, +delta)).

    shape: None -> Gaussian errors.  Otherwise a 1-D array of STANDARDISED
    (mean 0, sd 1) residuals resampled with replacement -- lets the empirical
    shape of this campaign's rep-to-rep noise be used at a controlled scale.
    """
    rng = rng or np.random.default_rng(1)
    if shape is None:
        d = rng.normal(gap, sigma, size=(trials, n))
    else:
        idx = rng.integers(0, len(shape), size=(trials, n))
        d = gap + sigma * np.asarray(shape)[idx]
    mean = d.mean(axis=1)
    sd = d.std(axis=1, ddof=1)
    se = sd / math.sqrt(n)
    tc = t_ppf(0.95, n - 1)          # two one-sided 5% tests == 90% two-sided CI
    lo = mean - tc * se
    hi = mean + tc * se
    return float(((lo > -delta) & (hi < delta)).mean())


def standardised_shape(values):
    v = np.asarray(values, dtype=float)
    v = v - v.mean()
    s = v.std(ddof=1)
    if s == 0:
        return None
    return v / s


def pair_difference_shape(arm_values):
    """All n^2 differences x_i - x_j of an arm's rep values, standardised.

    This is the empirical distribution of a difference between two INDEPENDENT
    draws from the arm's rep-level law -- i.e. the shape that A3 - A4 would have
    if A3 and A4 were exchangeable and unpaired.  Scale is set by the caller.
    """
    v = np.asarray(arm_values, dtype=float)
    d = (v[:, None] - v[None, :]).ravel()
    s = d.std(ddof=1)
    if s == 0:
        return None
    return d / s


def moments(v):
    v = np.asarray(v, dtype=float)
    m = v.mean()
    s = v.std(ddof=0)
    if s == 0:
        return {"skew": float("nan"), "exkurt": float("nan")}
    z = (v - m) / s
    return {"skew": float((z ** 3).mean()), "exkurt": float((z ** 4).mean() - 3.0)}


def min_n_for_power(sigma, delta, gap, target=0.80, nmax=60, trials=60_000, shape=None, rng=None):
    rng = rng or np.random.default_rng(7)
    for n in range(3, nmax + 1):
        if tost_power(sigma, n, delta, gap, trials=trials, shape=shape, rng=rng) >= target:
            return n
    return None


def min_delta_for_power(sigma, n, gap, target=0.80, trials=60_000, shape=None, rng=None):
    rng = rng or np.random.default_rng(11)
    lo, hi = 1e-6, 2.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if tost_power(sigma, n, mid, gap, trials=trials, shape=shape, rng=rng) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ------------------------------------------------------------------ token scale
def token_p_from_x(x, n_gaps=95):
    """Independence-model inverse: X = 1-(1-p)^G  =>  p = 1-(1-X)^(1/G)."""
    x = min(max(x, 0.0), 1.0 - 1e-12)
    return 1.0 - (1.0 - x) ** (1.0 / n_gaps)


if __name__ == "__main__":
    print("t_{0.975,4} =", t_ppf(0.975, 4), " (C-2 Simpson value 2.7764451051977987)")
    print("t_{0.95,4}  =", t_ppf(0.95, 4))
    print("t_{0.95,9}  =", t_ppf(0.95, 9), " (tabulated 1.833113)")
