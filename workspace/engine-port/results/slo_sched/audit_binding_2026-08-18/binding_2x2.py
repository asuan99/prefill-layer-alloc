#!/usr/bin/env python3
"""Adversarial audit of the G16 side-output 'binding constraint' reading.

Read-only.  Recomputes, from the raw per-request artifacts, the 2x2 joint
distribution of {TTFT fail} x {ITL-p95 fail} that the pass-rate table
(side_outputs.per_arm) marginalises away.

Canonical predicate (methodology gate #4, pdmux_eval.analyze.RequestResult.passes):
    pass  <=>  ttft_ms <= 3000  AND  percentile(token_itl_ms, .95) <= 60
A request with no recorded ITL is scored as an ITL FAILURE
(percentile(()) = nan; nan <= x is False)  == g16_analyze empty_itl_policy='inf'.
"""
import json, math, statistics, sys
from pathlib import Path
from collections import defaultdict

ROOT = Path("/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port")
SLO_DIR = ROOT / "results/slo_sched"
sys.path.insert(0, str(ROOT / "benchmarks"))
from pdmux_eval.analyze import load_bench_serving_rounds, percentile  # canon loader

TTFT_SLO, ITL_SLO = 3000.0, 60.0
ARMS = ["d16", "d24", "d34", "d44", "d54", "d64", "d74"]
JOBS = {"blk1": "884336", "blk2": "884410", "blk3": "884411", "blk4": "884412"}


def boot_cells(path, ttft_slo=TTFT_SLO, itl_slo=ITL_SLO):
    reqs, dur = load_bench_serving_rounds(Path(path))
    a = b = c = d = 0           # a: pass/pass  b: Tpass/Ifail  c: Tfail/Ipass  d: fail/fail
    empty = 0
    for r in reqs:
        tp = r.ttft_ms <= ttft_slo
        if r.token_itl_ms:
            ip = percentile(r.token_itl_ms, 0.95) <= itl_slo
        else:
            empty += 1
            ip = False          # canon: nan <= x is False
        if tp and ip:   a += 1
        elif tp:        b += 1
        elif ip:        c += 1
        else:           d += 1
    n = a + b + c + d
    return dict(a=a, b=b, c=c, d=d, n=n, empty=empty, duration_s=dur)


def phi_and_or(a, b, c, d):
    n = a + b + c + d
    r1, r2 = a + b, c + d       # TTFT pass / fail
    k1, k2 = a + c, b + d       # ITL  pass / fail
    denom = math.sqrt(r1 * r2 * k1 * k2)
    phi = (a * d - b * c) / denom if denom > 0 else float("nan")
    # Haldane-Anscombe corrected odds ratio (guards zero cells)
    orat = ((a + .5) * (d + .5)) / ((b + .5) * (c + .5))
    chi2 = n * phi * phi if denom > 0 else float("nan")
    return phi, orat, chi2


def t_ci(vals, conf=0.95):
    n = len(vals)
    if n < 2:
        return (float("nan"), float("nan"))
    m = statistics.fmean(vals)
    s = statistics.stdev(vals)
    tcrit = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571}[n]  # t(n-1), .975
    h = tcrit * s / math.sqrt(n)
    return (m - h, m + h)


def main():
    out = {"predicate": {"ttft_slo_ms": TTFT_SLO, "itl_slo_ms": ITL_SLO,
                         "itl_quantile": 0.95, "empty_itl": "FAIL (canon)",
                         "loader": "pdmux_eval.analyze.load_bench_serving_rounds"},
           "phases": {}}
    for phase in ("HI", "LO"):
        per_arm = {}
        for arm in ARMS:
            boots = []
            for blk, job in JOBS.items():
                p = SLO_DIR / f"g16_{blk}_{arm}_boot1_{job}_{phase}.jsonl"
                cells = boot_cells(p)
                cells["block"] = blk
                boots.append(cells)
            A = sum(x["a"] for x in boots); B = sum(x["b"] for x in boots)
            C = sum(x["c"] for x in boots); D = sum(x["d"] for x in boots)
            N = A + B + C + D
            phi, orat, chi2 = phi_and_or(A, B, C, D)
            pT, pI = (A + B) / N, (A + C) / N
            per_boot_phi = [phi_and_or(x["a"], x["b"], x["c"], x["d"])[0] for x in boots]
            per_boot_joint = [100.0 * x["a"] / x["n"] for x in boots]
            per_boot_ttft = [100.0 * (x["a"] + x["b"]) / x["n"] for x in boots]
            per_boot_itl = [100.0 * (x["a"] + x["c"]) / x["n"] for x in boots]
            per_arm[arm] = {
                "pooled_cells": {"TpassIpass": A, "TpassIfail": B,
                                 "TfailIpass": C, "TfailIfail": D, "n": N},
                "empty_itl_requests": sum(x["empty"] for x in boots),
                "ttft_pass_pct": 100 * pT, "itl_p95_pass_pct": 100 * pI,
                "joint_pass_pct": 100 * A / N,
                "product_of_marginals_pct": 100 * pT * pI,
                "joint_minus_product_pp": 100 * (A / N - pT * pI),
                "phi": phi, "odds_ratio_HA": orat, "chi2_1df": chi2,
                "P(Ifail|Tpass)": B / (A + B) if A + B else float("nan"),
                "P(Ifail|Tfail)": D / (C + D) if C + D else float("nan"),
                "P(Tfail|Ipass)": C / (A + C) if A + C else float("nan"),
                "P(Tfail|Ifail)": D / (B + D) if B + D else float("nan"),
                # counterfactual "fix one axis" gains, in pp of joint pass
                "gain_if_TTFT_perfect_pp": 100 * (pI - A / N),
                "gain_if_ITL_perfect_pp": 100 * (pT - A / N),
                "per_boot": {"block": [x["block"] for x in boots],
                             "joint_pct": per_boot_joint,
                             "ttft_pct": per_boot_ttft,
                             "itl_pct": per_boot_itl,
                             "phi": per_boot_phi},
                "per_boot_ci": {"joint": t_ci(per_boot_joint),
                                "ttft": t_ci(per_boot_ttft),
                                "itl": t_ci(per_boot_itl)},
                "per_boot_sd": {"joint": statistics.stdev(per_boot_joint),
                                "ttft": statistics.stdev(per_boot_ttft),
                                "itl": statistics.stdev(per_boot_itl)},
            }
        out["phases"][phase] = per_arm
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
