#!/usr/bin/env python3
"""Which constraint is BINDING, under the shadow-price definition?

The main session's predicate was "the axis with the LOWER marginal pass rate
binds".  Under a full-relaxation counterfactual that predicate is an IDENTITY:
  gain if ITL made perfect  = P(TTFT pass) - joint
  gain if TTFT made perfect = P(ITL  pass) - joint
so "which relaxation gains more" reduces to "which marginal is lower" -- the same
statement, not independent evidence (methodology gate #40 / #9).

The economically meaningful notion is the LOCAL shadow price
  d(joint) / d(log threshold)
which is what a scheduler could actually trade.  This script computes both.
Read-only.
"""
import json, math, statistics, sys
from pathlib import Path

ROOT = Path("/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port")
SLO_DIR = ROOT / "results/slo_sched"
sys.path.insert(0, str(ROOT / "benchmarks"))
from pdmux_eval.analyze import load_bench_serving_rounds, percentile

ARMS = ["d16", "d24", "d34", "d44", "d54", "d64", "d74"]
JOBS = [("blk1", "884336"), ("blk2", "884410"), ("blk3", "884411"), ("blk4", "884412")]
T0, I0 = 3000.0, 60.0
EPS = 0.025      # +-2.5% symmetric probe, well inside gate #6's +-10% band


def load(phase, arm):
    T, I = [], []
    for blk, job in JOBS:
        reqs, _ = load_bench_serving_rounds(SLO_DIR / f"g16_{blk}_{arm}_boot1_{job}_{phase}.jsonl")
        for r in reqs:
            T.append(r.ttft_ms)
            I.append(percentile(r.token_itl_ms, .95) if r.token_itl_ms else math.inf)
    return T, I


def joint(T, I, t, i):
    return 100.0 * sum(a <= t and b <= i for a, b in zip(T, I)) / len(T)


def main():
    out = {"probe": {"eps_rel": EPS, "ttft_slo_ms": T0, "itl_slo_ms": I0}, "phases": {}}
    for phase in ("HI", "LO"):
        rows = {}
        for arm in ARMS:
            T, I = load(phase, arm)
            j0 = joint(T, I, T0, I0)
            pT = 100.0 * sum(a <= T0 for a in T) / len(T)
            pI = 100.0 * sum(b <= I0 for b in I) / len(I)
            jTp = joint(T, I, T0 * (1 + EPS), I0)
            jTm = joint(T, I, T0 * (1 - EPS), I0)
            jIp = joint(T, I, T0, I0 * (1 + EPS))
            jIm = joint(T, I, T0, I0 * (1 - EPS))
            dl = math.log((1 + EPS) / (1 - EPS))
            sT = (jTp - jTm) / dl          # pp per log-unit of TTFT threshold
            sI = (jIp - jIm) / dl
            rows[arm] = {
                "joint_pct": j0, "ttft_pass_pct": pT, "itl_p95_pass_pct": pI,
                "lower_marginal_axis": "ITL" if pI < pT else "TTFT",
                "full_relax_gain_if_ITL_perfect_pp": pT - j0,
                "full_relax_gain_if_TTFT_perfect_pp": pI - j0,
                "shadow_TTFT_pp_per_logunit": sT,
                "shadow_ITL_pp_per_logunit": sI,
                "shadow_ratio_ITL_over_TTFT": (sI / sT) if sT else float("inf"),
                "more_binding_by_shadow_price": "ITL" if sI > sT else "TTFT",
                "joint_at_itl_minus2.5pct": jIm, "joint_at_itl_plus2.5pct": jIp,
                "joint_at_ttft_minus2.5pct": jTm, "joint_at_ttft_plus2.5pct": jTp,
            }
        out["phases"][phase] = rows
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
