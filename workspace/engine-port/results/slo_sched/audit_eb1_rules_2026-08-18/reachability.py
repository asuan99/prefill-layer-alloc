#!/usr/bin/env python3
"""E-B1 audit: is the shadow price a price of anything the POLICY can buy?

S_x = d(joint)/d ln(threshold_x) is (to first order) also the return on a policy
that scales metric x down by 1%.  So it is decision-relevant only over the range
of x that the policy set can actually reach.

Test 1 (range): compare the +-2.5% probe width against the WHOLE observed spread
of each metric across the 7 arms.
Test 2 (falsification): if S were the marginal return to a policy move, then for
any arm pair a->b,
    joint(b) - joint(a)  ~  S_ttft(a)*ln(M_ttft(a)/M_ttft(b)) + S_itl(a)*ln(M_itl(a)/M_itl(b))
Compare predicted vs observed over all 21 ordered pairs.
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
T0, I0, EPS = 3000.0, 60.0, 0.025


def load(phase, arm):
    T, I = [], []
    for blk, job in JOBS:
        reqs, _ = load_bench_serving_rounds(SLO / f"g16_{blk}_{arm}_boot1_{job}_{phase}.jsonl")
        for r in reqs:
            T.append(r.ttft_ms)
            I.append(percentile(r.token_itl_ms, .95) if r.token_itl_ms else math.inf)
    return T, I


def main():
    phase = "HI"
    D = {a: load(phase, a) for a in ARMS}
    S, M, J = {}, {}, {}
    for a in ARMS:
        T, I = D[a]
        n = len(T)
        j = lambda t, i: 100.0 * sum(x <= t and y <= i for x, y in zip(T, I)) / n
        dl = math.log((1 + EPS) / (1 - EPS))
        S[a] = {"ttft": (j(T0 * (1 + EPS), I0) - j(T0 * (1 - EPS), I0)) / dl,
                "itl": (j(T0, I0 * (1 + EPS)) - j(T0, I0 * (1 - EPS))) / dl}
        fin = [v for v in I if math.isfinite(v)]
        M[a] = {"ttft_mean": statistics.fmean(T), "itl_mean": statistics.fmean(fin)}
        J[a] = j(T0, I0)

    rng_t = (min(M[a]["ttft_mean"] for a in ARMS), max(M[a]["ttft_mean"] for a in ARMS))
    rng_i = (min(M[a]["itl_mean"] for a in ARMS), max(M[a]["itl_mean"] for a in ARMS))
    U = ["d44", "d54", "d64", "d74"]
    rng_iU = (min(M[a]["itl_mean"] for a in U), max(M[a]["itl_mean"] for a in U))

    out = {"phase": phase, "eps": EPS,
           "probe_width_vs_policy_range": {
               "probe_half_width_rel": EPS,
               "ttft_full_range_ln": math.log(rng_t[1] / rng_t[0]),
               "ttft_full_range_rel_pct": 100 * (rng_t[1] / rng_t[0] - 1),
               "itl_full_range_ln": math.log(rng_i[1] / rng_i[0]),
               "itl_full_range_rel_pct": 100 * (rng_i[1] / rng_i[0] - 1),
               "itl_U_range_ln": math.log(rng_iU[1] / rng_iU[0]),
               "itl_U_range_rel_pct": 100 * (rng_iU[1] / rng_iU[0] - 1),
               "probe_over_itl_U_range": (2 * EPS) / (rng_iU[1] / rng_iU[0] - 1),
           },
           "per_arm": {a: {"S_ttft": S[a]["ttft"], "S_itl": S[a]["itl"],
                           "joint_pct": J[a], **M[a]} for a in ARMS},
           "reachable_price": {}, "pairs": []}

    for a in ARMS:
        # what the BEST reachable move on each axis is worth, from arm a
        best_t = min(M[b]["ttft_mean"] for b in ARMS)
        best_i = min(M[b]["itl_mean"] for b in ARMS)
        out["reachable_price"][a] = {
            "S_ttft_x_reachable_dlnTTFT_pp": S[a]["ttft"] * math.log(M[a]["ttft_mean"] / best_t),
            "S_itl_x_reachable_dlnITL_pp": S[a]["itl"] * math.log(M[a]["itl_mean"] / best_i),
            "raw_ratio_S_itl_over_S_ttft": S[a]["itl"] / S[a]["ttft"] if S[a]["ttft"] else None,
        }
    for a in ARMS:
        for b in ARMS:
            if a == b:
                continue
            pred = (S[a]["ttft"] * math.log(M[a]["ttft_mean"] / M[b]["ttft_mean"])
                    + S[a]["itl"] * math.log(M[a]["itl_mean"] / M[b]["itl_mean"]))
            obs = J[b] - J[a]
            out["pairs"].append({"from": a, "to": b, "pred_pp": pred, "obs_pp": obs,
                                 "pred_minus_obs": pred - obs})
    preds = [p["pred_pp"] for p in out["pairs"]]
    obss = [p["obs_pp"] for p in out["pairs"]]
    mp = statistics.fmean(preds); mo = statistics.fmean(obss)
    sp = statistics.pstdev(preds); so = statistics.pstdev(obss)
    cov = statistics.fmean([(p - mp) * (o - mo) for p, o in zip(preds, obss)])
    out["pair_summary"] = {"n": len(preds), "corr": cov / (sp * so),
                           "mean_abs_pred": statistics.fmean(abs(x) for x in preds),
                           "mean_abs_obs": statistics.fmean(abs(x) for x in obss),
                           "slope_obs_on_pred": cov / (sp * sp)}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
