"""C-2 companion: exact p-values and two distribution-free / calibrated
alternatives to the percentile bootstrap, on the canonical predicate.

  * paired t (df=4) two-sided p, from a numerically integrated t density
  * exact paired randomisation (sign-flip) test over all 2^5 = 32 sign patterns
    -- assumption-free; its two-sided p FLOOR at n=5 is 2/32 = 0.0625
  * studentised (bootstrap-t) interval, the small-sample correction the
    canonical percentile bootstrap omits
"""
from __future__ import annotations

import itertools
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/benchmarks")
sys.path.insert(0, str(Path(__file__).parent))
from pdmux_eval.analyze import percentile  # noqa: E402
from verify_c2_c3 import TCRIT4, paired_t, t_cdf  # noqa: E402

HERE = Path(__file__).parent


def perm_p(diffs):
    obs = abs(statistics.fmean(diffs))
    n = len(diffs)
    hits = 0
    for signs in itertools.product((1, -1), repeat=n):
        m = abs(statistics.fmean(s * d for s, d in zip(signs, diffs)))
        if m >= obs - 1e-15:
            hits += 1
    return hits / 2 ** n


def studentised_ci(diffs, samples=10000, seed=1):
    rng = np.random.default_rng(seed)
    d = np.array(diffs, dtype=float)
    n = len(d)
    mean, sd = d.mean(), d.std(ddof=1)
    se = sd / math.sqrt(n)
    idx = rng.integers(0, n, size=(samples, n))
    b = d[idx]
    bm = b.mean(axis=1)
    bsd = b.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        tstar = (bm - mean) / (bsd / math.sqrt(n))
    tstar = tstar[np.isfinite(tstar)]
    lo_q = percentile(tstar.tolist(), 0.975)
    hi_q = percentile(tstar.tolist(), 0.025)
    return mean - lo_q * se, mean - hi_q * se, len(tstar) / samples


def main():
    rows = json.loads((HERE / "c2_c3_results.json").read_text())["c2"]
    out = []
    print(f"{'model':22s} {'rate':>4s} {'eff%':>9s} {'t':>7s} {'p_t':>8s} {'p_perm':>7s} "
          f"{'boot CI':>21s} {'t CI':>21s} {'studentised CI':>23s}")
    for r in rows:
        d = r["per_rep_diffs"]
        t = paired_t(d)
        p_t = 2 * (1 - t_cdf(abs(t["t"]), 4))
        pp = perm_p(d)
        slo, shi, frac = studentised_ci(d)
        out.append({**{k: r[k] for k in ("model", "rate", "effect_percent",
                                         "boot_lo", "boot_hi", "t_lo", "t_hi")},
                    "t_stat": t["t"], "p_t": p_t, "p_perm_exact": pp,
                    "studentised_lo": slo, "studentised_hi": shi,
                    "studentised_finite_frac": frac,
                    "n_pos_diffs": sum(1 for x in d if x > 0),
                    "per_rep_diffs": d})
        print(f"{r['model']:22s} {r['rate']:4.0f} {r['effect_percent']:+9.2f} {t['t']:7.3f} "
              f"{p_t:8.4f} {pp:7.4f} "
              f"[{r['boot_lo']:+8.4f},{r['boot_hi']:+8.4f}] "
              f"[{r['t_lo']:+8.4f},{r['t_hi']:+8.4f}] "
              f"[{slo:+9.4f},{shi:+9.4f}]")
    print("\nexact sign-flip randomisation p FLOOR at n=5 (all diffs same sign) = "
          f"{2 / 32:.4f}  -> no n=5 paired cell can reach p<0.05 distribution-free")
    (HERE / "c2_pvalues.json").write_text(json.dumps(out, indent=1))
    print("wrote c2_pvalues.json")


if __name__ == "__main__":
    main()
