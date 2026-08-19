#!/usr/bin/env python3
"""E-B1 rules-layer audit, item 1: DOES THE ADJUDICABLE WINDOW EXIST AT ALL?

E-B1 asks for a (rate, arm) cell that simultaneously satisfies
  (A) band mass <= 0.25   : P(0.9*theta <= X <= 1.1*theta) <= 0.25      [prereg gate #6 / K2]
  (B) marginal in [5,95]% : 0.05 <= P(X <= theta) <= 0.95               [design sec1]
on BOTH axes (X = TTFT with theta=3000 ms, X = per-request ITL-p95 with theta=60 ms).

Key observation this script exploits: sweeping the LOAD RATE moves the metric
distribution relative to the FIXED threshold.  Under a scale family
(X_rate = c(rate) * X_0, c increasing in rate) the pair (A,B) evaluated at
threshold theta and rate `rate` equals the pair evaluated on the BASE
distribution at threshold theta/c.  So sweeping theta over the observed
distribution is EXACTLY equivalent to sweeping the rate, and the feasibility of
(A and B) can be decided BEFORE any GPU is spent.

Read-only: reads only G16 campaign artifacts through the canonical loader.
"""
import json, math, sys
from pathlib import Path

ROOT = Path("/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port")
SLO = ROOT / "results/slo_sched"
sys.path.insert(0, str(ROOT / "benchmarks"))
from pdmux_eval.analyze import load_bench_serving_rounds, percentile

ARMS = ["d16", "d24", "d34", "d44", "d54", "d64", "d74"]
JOBS = [("blk1", "884336"), ("blk2", "884410"), ("blk3", "884411"), ("blk4", "884412")]
BAND, BAND_MAX = 0.10, 0.25
PLO, PHI = 0.05, 0.95


def load(phase, arm):
    T, I = [], []
    for blk, job in JOBS:
        reqs, _ = load_bench_serving_rounds(SLO / f"g16_{blk}_{arm}_boot1_{job}_{phase}.jsonl")
        for r in reqs:
            T.append(r.ttft_ms)
            I.append(percentile(r.token_itl_ms, .95) if r.token_itl_ms else math.inf)
    return T, I


def cdf(xs_sorted, v):
    # P(X <= v)
    lo, hi = 0, len(xs_sorted)
    while lo < hi:
        mid = (lo + hi) // 2
        if xs_sorted[mid] <= v:
            lo = mid + 1
        else:
            hi = mid
    return lo / len(xs_sorted)


def probe(xs_sorted, theta):
    p = cdf(xs_sorted, theta)
    bm = cdf(xs_sorted, theta * (1 + BAND)) - cdf(xs_sorted, theta * (1 - BAND))
    return p, bm


def sweep(xs, n_grid=4000):
    """Sweep theta (equivalently: sweep the rate under a scale family).
    Returns the feasible theta interval(s) for (A) and (B) and their intersection."""
    fin = sorted(v for v in xs if math.isfinite(v))
    xs_s = sorted(xs, key=lambda v: (math.inf if not math.isfinite(v) else v))
    if not fin:
        return None
    lo, hi = fin[0] * 0.5, fin[-1] * 2.0
    grid = [lo * (hi / lo) ** (k / (n_grid - 1)) for k in range(n_grid)]
    feasA, feasB, feasAB = [], [], []
    rows = []
    for th in grid:
        p, bm = probe(xs_s, th)
        okA, okB = bm <= BAND_MAX, PLO <= p <= PHI
        if okA: feasA.append(th)
        if okB: feasB.append(th)
        if okA and okB: feasAB.append(th)
        rows.append((th, p, bm))
    def span(v):
        return None if not v else (min(v), max(v))
    return {"theta_range_A(bandmass<=.25)": span(feasA),
            "theta_range_B(marginal in [5,95])": span(feasB),
            "theta_range_A_and_B": span(feasAB),
            "n_grid_pts_A_and_B": len(feasAB),
            "frac_grid_A_and_B": len(feasAB) / n_grid,
            "rows": rows}


def stats(xs):
    fin = sorted(v for v in xs if math.isfinite(v))
    n = len(xs)
    m = sum(fin) / len(fin)
    sd = (sum((v - m) ** 2 for v in fin) / (len(fin) - 1)) ** .5
    q = lambda p: fin[min(len(fin) - 1, max(0, int(round(p * (len(fin) - 1)))))]
    return {"n": n, "n_finite": len(fin), "mean": m, "sd": sd, "cv": sd / m,
            "p05": q(.05), "p25": q(.25), "p50": q(.50), "p75": q(.75), "p95": q(.95),
            "iqr_ratio_p75_over_p25": q(.75) / q(.25) if q(.25) else None,
            "p95_over_p05": q(.95) / q(.05) if q(.05) else None}


def main():
    out = {"definitions": {"band": "+-10% around threshold, mass<=0.25 (K2/gate#6)",
                           "marginal_window": "[5%,95%] pass rate (design sec1)",
                           "equivalence": "sweeping theta == sweeping rate under a scale family"},
           "per_cell": {}}
    for phase in ("LO", "HI"):
        for arm in ARMS:
            T, I = load(phase, arm)
            cell = {}
            for name, xs, th in (("TTFT", T, 3000.0), ("ITLp95", I, 60.0)):
                p, bm = probe(sorted(xs, key=lambda v: (math.inf if not math.isfinite(v) else v)), th)
                sw = sweep(xs)
                cell[name] = {
                    "at_operating_threshold": {"theta": th, "pass_frac": p, "band_mass": bm,
                                               "okA_bandmass": bm <= BAND_MAX,
                                               "okB_marginal": PLO <= p <= PHI,
                                               "adjudicable": bm <= BAND_MAX and PLO <= p <= PHI},
                    "dist": stats(xs),
                    "theta_sweep": {k: v for k, v in sw.items() if k != "rows"},
                }
            out["per_cell"][f"{phase}/{arm}"] = cell
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
