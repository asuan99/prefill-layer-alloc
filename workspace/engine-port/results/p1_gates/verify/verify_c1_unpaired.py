"""C-1 blast-radius addendum 2: coverage of ``unpaired_bootstrap_ci`` at the n
actually used in the canonical citation (n=4 per arm, CEILING_CENSORING_DIAG /
CONSENSUS 1-13 footnote), compared with Welch's t.

Same recovery-and-validate strategy as verify_c1_coverage.py: the tool's frozen
seed=1 makes the resampling design deterministic, and the resample means are
linear in the data, so the design can be recovered, checked for EXACT agreement
with the canonical function, and then evaluated at scale.
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/benchmarks")
from pdmux_eval.analyze import unpaired_bootstrap_ci  # noqa: E402

SAMPLES = 10000
LO_POS = (SAMPLES - 1) * 0.025
HI_POS = (SAMPLES - 1) * 0.975
# Welch/Student t_{0.975, df}
TCRIT = {2: 4.302652730, 3: 3.182446305, 4: 2.776445105, 5: 2.570581836,
         6: 2.446911851, 7: 2.364624252}


def recover_design(nb, np_, seed=1):
    """Replays the tool's interleaved draws: per sample, nb baseline then np_ proposed."""
    rng = random.Random(seed)
    cb = np.zeros((SAMPLES, nb))
    cp = np.zeros((SAMPLES, np_))
    poolb, poolp = list(range(nb)), list(range(np_))
    for s in range(SAMPLES):
        for _ in range(nb):
            cb[s, rng.choice(poolb)] += 1
        for _ in range(np_):
            cp[s, rng.choice(poolp)] += 1
    return cb, cp


def ci_fast(xb, xp, cb, cp):
    d = (xp @ cp.T) / xp.shape[1] - (xb @ cb.T) / xb.shape[1]
    lo_i, hi_i = int(math.floor(LO_POS)), int(math.floor(HI_POS))
    part = np.partition(d, (lo_i, lo_i + 1, hi_i, hi_i + 1), axis=1)
    lo = part[:, lo_i] + (part[:, lo_i + 1] - part[:, lo_i]) * (LO_POS - lo_i)
    hi = part[:, hi_i] + (part[:, hi_i + 1] - part[:, hi_i]) * (HI_POS - hi_i)
    return lo, hi


def main():
    out = {}
    for n in (4, 5):
        cb, cp = recover_design(n, n)
        rng = np.random.default_rng(11)
        xb = rng.normal(size=(200, n))
        xp = rng.normal(size=(200, n))
        lo, hi = ci_fast(xb, xp, cb, cp)
        worst = 0.0
        for i in range(200):
            ref = unpaired_bootstrap_ci(xb[i].tolist(), xp[i].tolist())
            worst = max(worst, abs(ref["ci95_low"] - lo[i]), abs(ref["ci95_high"] - hi[i]))
        print(f"n={n}: design validated against canonical unpaired_bootstrap_ci, "
              f"max abs diff {worst:.3e}")
        assert worst < 1e-12

        trials, cov_b, cov_t, wb, wt = 100000, 0, 0, 0.0, 0.0
        done = 0
        while done < trials:
            m = min(2000, trials - done)
            a = rng.normal(size=(m, n))
            b = rng.normal(size=(m, n))
            lo, hi = ci_fast(a, b, cb, cp)
            cov_b += int(((lo <= 0) & (hi >= 0)).sum())
            wb += float((hi - lo).sum())
            # Welch t
            va, vb = a.var(axis=1, ddof=1), b.var(axis=1, ddof=1)
            se = np.sqrt(va / n + vb / n)
            df = (va / n + vb / n) ** 2 / ((va / n) ** 2 / (n - 1) + (vb / n) ** 2 / (n - 1))
            tc = np.interp(df, list(TCRIT), [TCRIT[k] for k in TCRIT])
            half = tc * se
            diff = b.mean(axis=1) - a.mean(axis=1)
            cov_t += int(((diff - half <= 0) & (diff + half >= 0)).sum())
            wt += float((2 * half).sum())
            done += m
        out[n] = {"coverage_unpaired_bootstrap": cov_b / trials,
                  "coverage_welch_t": cov_t / trials,
                  "mean_width_bootstrap": wb / trials, "mean_width_welch": wt / trials,
                  "width_ratio": (wb / trials) / (wt / trials),
                  "mc_se": math.sqrt((cov_b / trials) * (1 - cov_b / trials) / trials)}
        print(f"   n={n} per arm: unpaired percentile bootstrap coverage "
              f"{out[n]['coverage_unpaired_bootstrap']:.4f} +- {out[n]['mc_se']:.4f} "
              f"(Welch t {out[n]['coverage_welch_t']:.4f}), width ratio "
              f"{out[n]['width_ratio']:.3f}")
    Path(Path(__file__).parent / "c1_unpaired.json").write_text(json.dumps(out, indent=1))
    print("wrote c1_unpaired.json")


if __name__ == "__main__":
    main()
