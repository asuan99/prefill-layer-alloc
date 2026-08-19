#!/usr/bin/env python3
"""E-B1 audit: (1) the band-mass gate ALGEBRAICALLY CAPS the shadow price it is
supposed to license; (2) the epsilon (finite-difference width) is a free knob;
(3) block-level precision of the shadow price and its ratio.

(1)  joint(x-threshold t) = P(X<=t, Y<=y0).   For the +-10% probe,
     joint(1.1t) - joint(0.9t) = P(X in (0.9t,1.1t], Y<=y0)  <=  band_mass_X.
     S^(10%) = that difference / ln(1.1/0.9),  ln(1.1/0.9)=0.200671.
     => gate #6 (band_mass <= 0.25) IMPLIES S^(10%) <= 124.58 pp per log-unit.
     The gate is an upper bound on the estimand, on whichever axis has the cliff.

Read-only.
"""
import json, math, statistics, sys
from pathlib import Path

ROOT = Path("/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port")
SLO = ROOT / "results/slo_sched"
sys.path.insert(0, str(ROOT / "benchmarks"))
from pdmux_eval.analyze import load_bench_serving_rounds, percentile

ARMS = ["d16", "d24", "d34", "d44", "d54", "d64", "d74"]
JOBS = [("blk1", "884336"), ("blk2", "884410"), ("blk3", "884411"), ("blk4", "884412")]
T0, I0 = 3000.0, 60.0
EPS = [0.005, 0.01, 0.025, 0.05, 0.10]


def load_block(phase, arm, blk, job):
    reqs, _ = load_bench_serving_rounds(SLO / f"g16_{blk}_{arm}_boot1_{job}_{phase}.jsonl")
    T = [r.ttft_ms for r in reqs]
    I = [percentile(r.token_itl_ms, .95) if r.token_itl_ms else math.inf for r in reqs]
    return T, I


def metrics(T, I, eps):
    n = len(T)
    j = lambda t, i: sum(a <= t and b <= i for a, b in zip(T, I)) / n
    dl = math.log((1 + eps) / (1 - eps))
    sT = 100 * (j(T0 * (1 + eps), I0) - j(T0 * (1 - eps), I0)) / dl
    sI = 100 * (j(T0, I0 * (1 + eps)) - j(T0, I0 * (1 - eps))) / dl
    return sT, sI


def t_ci(v):
    n = len(v)
    m = statistics.fmean(v)
    s = statistics.stdev(v)
    tc = {2: 12.706, 3: 4.303, 4: 3.182}[n]
    return m, s, (m - tc * s / n ** .5, m + tc * s / n ** .5)


def main():
    out = {"gate_cap": {"ln(1.1/0.9)": math.log(1.1 / 0.9),
                        "S_max_pp_per_logunit_if_bandmass_0.25": 100 * 0.25 / math.log(1.1 / 0.9)},
           "cells": {}}
    for phase in ("HI", "LO"):
        for arm in ARMS:
            blocks = [load_block(phase, arm, b, jb) for b, jb in JOBS]
            T = [x for b in blocks for x in b[0]]
            I = [x for b in blocks for x in b[1]]
            n = len(T)
            pT = sum(a <= T0 for a in T) / n
            pI = sum(b <= I0 for b in I) / n
            bmT = sum(0.9 * T0 <= a <= 1.1 * T0 for a in T) / n
            bmI = sum(0.9 * I0 <= b <= 1.1 * I0 for b in I) / n
            eps_rows = {}
            for e in EPS:
                sT, sI = metrics(T, I, e)
                eps_rows[f"{e:.3f}"] = {"S_ttft": sT, "S_itl": sI,
                                        "ratio_itl_over_ttft": (sI / sT) if sT else None,
                                        "binding": ("ITL" if sI > sT else "TTFT")}
            # per-block at eps=0.025 (the audit's choice) and 0.10 (the gate's own band)
            per_blk = {}
            for e in (0.025, 0.10):
                sTs, sIs, ratios = [], [], []
                for (Tb, Ib) in blocks:
                    a, b = metrics(Tb, Ib, e)
                    sTs.append(a); sIs.append(b)
                    ratios.append(b / a if a else float("nan"))
                mT, sdT, ciT = t_ci(sTs)
                mI, sdI, ciI = t_ci(sIs)
                per_blk[f"{e:.3f}"] = {
                    "S_ttft_blocks": sTs, "S_ttft_mean": mT, "S_ttft_sd": sdT, "S_ttft_t3_ci": ciT,
                    "S_itl_blocks": sIs, "S_itl_mean": mI, "S_itl_sd": sdI, "S_itl_t3_ci": ciI,
                    "ratio_blocks": ratios,
                    "n_blocks_with_S_ttft_zero": sum(1 for x in sTs if x == 0),
                }
            out["cells"][f"{phase}/{arm}"] = {
                "n_requests": n, "pT": pT, "pI": pI,
                "band_mass_ttft": bmT, "band_mass_itl": bmI,
                "gate6_pass_ttft": bmT <= 0.25, "gate6_pass_itl": bmI <= 0.25,
                "marginal_window_pass_ttft": 0.05 <= pT <= 0.95,
                "marginal_window_pass_itl": 0.05 <= pI <= 0.95,
                "ADJUDICABLE_both_axes": (bmT <= .25 and bmI <= .25
                                          and 0.05 <= pT <= 0.95 and 0.05 <= pI <= 0.95),
                # cliff-ratio invariant  R = bandmass / (1 - pass);  feasible iff R <= 5
                "R_itl_bandmass_over_failfrac": (bmI / (1 - pI)) if pI < 1 else None,
                "R_ttft_bandmass_over_failfrac": (bmT / (1 - pT)) if pT < 1 else None,
                "eps_sweep": eps_rows,
                "per_block": per_blk,
            }
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
