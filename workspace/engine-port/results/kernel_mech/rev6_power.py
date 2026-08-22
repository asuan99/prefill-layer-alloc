#!/usr/bin/env python3
"""rev6 -- pre-simulation of the decision rule's operating characteristics.

rev5 was refused partly because it asked "can we判別 with a CI?" while carrying
zero power analysis (audit C4).  This script answers that question BEFORE the
harness exists, with the two counterfactuals rev6 registers side by side.

Discipline:
  * sigma is swept as a RANGE, not imported as a point value.  The nearest
    measured boot-to-boot dispersion in this repo is a different model and a
    different workload (lesson #31), so it enters as a plausible band only.
  * The ratio s = S_G / (S_K + S_G) has a random denominator; percentile
    bootstrap is prohibited in canon body text (undercoverage), so the interval
    is BCa and the denominator-sign behaviour is reported, not hidden.
  * Every scenario is a TRUTH we impose; the output is P(the rule reaches that
    truth), i.e. the operating characteristic -- not evidence about the GPU.
"""
from __future__ import annotations
import math, statistics, json, sys
try:
    from scipy import stats as st
except Exception:                                    # pragma: no cover
    print("needs scipy (project venv)"); sys.exit(2)

RNG = 12345
D_LO, D_HI = 44.0, 92.0
SCALE = D_LO / D_HI                                  # ideal parallel shrink


def s_of(K44, G44, K92, G92, symmetric: bool):
    """Shortfall share attributable to the gap, under one counterfactual."""
    S_K = K92 - K44 * SCALE
    S_G = G92 - (G44 * SCALE if symmetric else G44)
    S = S_K + S_G
    return (S_G / S) if S != 0 else float("nan"), S


def bca(samples, theta_hat, jack):
    """BCa interval (95%).  Falls back to basic percentile ONLY if the
    acceleration is undefined, and says so in the output."""
    samples = sorted(x for x in samples if not math.isnan(x))
    if len(samples) < 50:
        return None, None, "too few finite resamples"
    n_less = sum(1 for x in samples if x < theta_hat)
    if n_less in (0, len(samples)):
        return samples[0], samples[-1], "z0 undefined (degenerate)"
    z0 = st.norm.ppf(n_less / len(samples))
    jbar = statistics.fmean(jack)
    num = sum((jbar - j) ** 3 for j in jack)
    den = 6.0 * (sum((jbar - j) ** 2 for j in jack) ** 1.5)
    a = num / den if den else 0.0
    out = []
    for q in (0.025, 0.975):
        z = st.norm.ppf(q)
        adj = z0 + (z0 + z) / (1 - a * (z0 + z))
        out.append(min(max(st.norm.cdf(adj), 0.001), 0.999))
    lo = samples[int(out[0] * (len(samples) - 1))]
    hi = samples[int(out[1] * (len(samples) - 1))]
    return lo, hi, "BCa"


def one_campaign(truth, n_boots, cv, rng, B=2000):
    """Simulate one campaign and return the verdict under both counterfactuals."""
    draws = {}
    for key, mu in truth.items():
        draws[key] = [mu * math.exp(rng.normalvariate(0, cv)) for _ in range(n_boots)]

    def point(idx=None, sym=False):
        take = (lambda v: statistics.fmean(v)) if idx is None else (lambda v: statistics.fmean([v[i] for i in idx]))
        return s_of(take(draws["K44"]), take(draws["G44"]),
                    take(draws["K92"]), take(draws["G92"]), sym)[0]

    res = {}
    for sym in (False, True):
        hat = point(sym=sym)
        boots = []
        for _ in range(B):
            idx = [rng.randrange(n_boots) for _ in range(n_boots)]
            boots.append(point(idx, sym))
        jack = [point([j for j in range(n_boots) if j != i], sym) for i in range(n_boots)]
        lo, hi, how = bca(boots, hat, jack)
        res["sym" if sym else "asym"] = {"s": hat, "lo": lo, "hi": hi, "how": how}
    return res


def verdict(lo, hi):
    if lo is None or math.isnan(lo) or math.isnan(hi):
        return "UNDETERMINED"
    if lo > 0.5:  return "GAP"
    if hi < 0.5:  return "KERNEL"
    if 0.15 <= lo and hi <= 0.85: return "MULTI_CAUSE"
    return "UNDETERMINED"


SCENARIOS = {
    # name: (K44, G44, K92, G92)  -- truths we impose
    "gap_truth   (간극 불변, 커널 잘 줄어듦)":      (50, 50, 30, 50),   # = 감사 CE-1
    "kernel_truth(커널이 안 줄어듦)":               (95,  5, 80, 20),   # = rev4 반례
    "mixed       (양쪽 절반씩)":                    (70, 30, 45, 40),
    "strong_gap  (간극이 실제로 커짐)":             (60, 40, 32, 60),
}

def main():
    import random
    out = {"frame": "rev6 operating characteristic. No GPU, no measurement. "
                    "Truths are imposed; output is P(rule reaches verdict).",
           "n_boots": 4, "reps": 400, "cv_sweep": [0.01, 0.025, 0.05], "scenarios": {}}
    for name, (K44, G44, K92, G92) in SCENARIOS.items():
        truth = {"K44": K44, "G44": G44, "K92": K92, "G92": G92}
        sA, _ = s_of(K44, G44, K92, G92, False)
        sB, _ = s_of(K44, G44, K92, G92, True)
        row = {"s_asym_true": round(sA, 4), "s_sym_true": round(sB, 4),
               "counterfactuals_agree": verdict(sA, sA) == verdict(sB, sB), "by_cv": {}}
        for cv in out["cv_sweep"]:
            rng = random.Random(RNG)
            tally = {}
            for _ in range(out["reps"]):
                r = one_campaign(truth, out["n_boots"], cv, rng, B=600)
                va = verdict(r["asym"]["lo"], r["asym"]["hi"])
                vb = verdict(r["sym"]["lo"], r["sym"]["hi"])
                key = va if va == vb else f"SENSITIVE({va}|{vb})"
                tally[key] = tally.get(key, 0) + 1
            row["by_cv"][f"cv={cv}"] = {k: round(v / out['reps'], 3)
                                        for k, v in sorted(tally.items(), key=lambda x: -x[1])}
        out["scenarios"][name] = row
    p = "/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/kernel_mech/REV6_POWER_2026-08-22.json"
    open(p, "w").write(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
