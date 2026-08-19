#!/usr/bin/env python3
"""E-B1 audit: the raw ratio S_itl/S_ttft is a NORMALISATION artefact.

S_x prices a +-eps RELATIVE move of threshold x.  Equivalently it prices a policy
that scales metric x by (1-eps).  But the policy set (7 decode-SM arms) can move
the two metrics by wildly different amounts.  Canonical statistics (g16_analyze:
M_ttft = median TTFT, M_itl = median per-request ITL-p95) over the HI grid:

    reachable dln(M_ttft) across the goodput plateau  >>  reachable dln(M_itl)

so pricing both axes with the SAME relative probe systematically flatters ITL.
This script recomputes the decision quantity on a reachable basis, and checks the
local linearisation against observed neighbour-arm goodput differences.
Read-only.
"""
import json, math, statistics, sys
from pathlib import Path

ROOT = Path("/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port")
SLO = ROOT / "results/slo_sched"
sys.path.insert(0, str(ROOT / "benchmarks"))
from pdmux_eval.analyze import load_bench_serving_rounds, percentile

ARMS = ["d16", "d24", "d34", "d44", "d54", "d64", "d74"]
PLATEAU = ["d34", "d44", "d54", "d64"]
JOBS = [("blk1", "884336"), ("blk2", "884410"), ("blk3", "884411"), ("blk4", "884412")]
T0, I0, EPS = 3000.0, 60.0, 0.025


def main():
    S, M, J = {}, {}, {}
    for a in ARMS:
        T, I = [], []
        for blk, job in JOBS:
            reqs, _ = load_bench_serving_rounds(SLO / f"g16_{blk}_{a}_boot1_{job}_HI.jsonl")
            for r in reqs:
                T.append(r.ttft_ms)
                I.append(percentile(r.token_itl_ms, .95) if r.token_itl_ms else math.inf)
        n = len(T)
        j = lambda t, i: 100.0 * sum(x <= t and y <= i for x, y in zip(T, I)) / n
        dl = math.log((1 + EPS) / (1 - EPS))
        S[a] = {"ttft": (j(T0 * (1 + EPS), I0) - j(T0 * (1 - EPS), I0)) / dl,
                "itl": (j(T0, I0 * (1 + EPS)) - j(T0, I0 * (1 - EPS))) / dl}
        M[a] = {"M_ttft": statistics.median(T),
                "M_itl": statistics.median([v for v in I])}
        J[a] = j(T0, I0)

    def rng(keys, k):
        v = [M[a][k] for a in keys]
        return math.log(max(v) / min(v))

    out = {"canonical_stats": "M_ttft = median TTFT, M_itl = median per-request ITL-p95 (g16_analyze:322-323)",
           "per_arm": {a: {**M[a], "joint_pct": J[a], **{f"S_{k}": v for k, v in S[a].items()}} for a in ARMS},
           "reachable_ln_range": {
               "plateau_arms": PLATEAU,
               "dln_M_ttft_plateau": rng(PLATEAU, "M_ttft"),
               "dln_M_itl_plateau": rng(PLATEAU, "M_itl"),
               "dln_M_ttft_all7": rng(ARMS, "M_ttft"),
               "dln_M_itl_all7": rng(ARMS, "M_itl"),
               "probe_ln_width_2eps": 2 * EPS,
           },
           "ratio_raw_vs_reachable": {}, "neighbour_check": []}
    dT, dI = rng(PLATEAU, "M_ttft"), rng(PLATEAU, "M_itl")
    for a in PLATEAU:
        raw = S[a]["itl"] / S[a]["ttft"] if S[a]["ttft"] else None
        out["ratio_raw_vs_reachable"][a] = {
            "raw_ratio_S_itl_over_S_ttft": raw,
            "reachable_pp_ttft": S[a]["ttft"] * dT,
            "reachable_pp_itl": S[a]["itl"] * dI,
            "reachable_ratio": (S[a]["itl"] * dI) / (S[a]["ttft"] * dT) if S[a]["ttft"] else None,
            "deflation_factor": (raw / ((S[a]["itl"] * dI) / (S[a]["ttft"] * dT))) if S[a]["ttft"] else None,
        }
    for a in PLATEAU:
        for b in PLATEAU:
            if a == b:
                continue
            pred = (S[a]["ttft"] * math.log(M[a]["M_ttft"] / M[b]["M_ttft"])
                    + S[a]["itl"] * math.log(M[a]["M_itl"] / M[b]["M_itl"]))
            out["neighbour_check"].append(
                {"from": a, "to": b, "pred_dJoint_pp": pred, "obs_dJoint_pp": J[b] - J[a],
                 "ttft_term": S[a]["ttft"] * math.log(M[a]["M_ttft"] / M[b]["M_ttft"]),
                 "itl_term": S[a]["itl"] * math.log(M[a]["M_itl"] / M[b]["M_itl"])})
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
