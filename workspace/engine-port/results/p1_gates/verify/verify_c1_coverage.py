"""C-1: actual coverage of ``pdmux_eval.analyze.paired_bootstrap_ci`` at n=5.

Method note (why this is not a copy of the audited code):
  The canonical function is a PERCENTILE bootstrap of the mean with a HARD-CODED
  seed (seed=1) and a fixed sample count (10000).  Because ``random.Random(1)``
  and ``len(effects)==5`` fully determine the 50000 index draws, the tool applies
  the SAME fixed resampling design to every analysis in this repository, and the
  bootstrap mean vector is LINEAR in the 5 effect values:
        boot_means = (C @ effects) / 5,    C = 10000x5 resample count matrix.
  Step 1 recovers C by replaying the RNG, then VALIDATES it by requiring exact
  agreement with the canonical function's own ci95_low/high on random datasets.
  Everything after that is the canonical estimator evaluated fast, not a
  re-implementation of it.

Conditions (all with true mean 0; the percentile bootstrap of the mean is
location-equivariant, so coverage of 0 by centred data == coverage of mu):
  A  normal(0,1)                       -- the auditor's stated condition
  B  empirical: pooled within-cell standardised rep-differences of THIS campaign
  C  empirical per-cell: resample from that cell's own 5 centred differences
  D  t_3 (heavy tail) and centred lognormal (skew) stress conditions
Also measured: coverage when the seed is NOT frozen (fresh design per trial),
to separate "small n" from "one unlucky frozen design".
"""
from __future__ import annotations

import json
import math
import random
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/benchmarks")
sys.path.insert(0, str(Path(__file__).parent))
from pdmux_eval.analyze import paired_bootstrap_ci  # noqa: E402

N = 5
SAMPLES = 10000
TCRIT4 = 2.7764451051977987
LO_POS = (SAMPLES - 1) * 0.025      # 249.975
HI_POS = (SAMPLES - 1) * 0.975      # 9749.025


def recover_design(n=N, samples=SAMPLES, seed=1):
    rng = random.Random(seed)
    idx = np.empty((samples, n), dtype=np.int8)
    pool = list(range(n))
    for s in range(samples):
        for k in range(n):
            idx[s, k] = rng.choice(pool)
    counts = np.zeros((samples, n), dtype=np.float64)
    for j in range(n):
        counts[:, j] = (idx == j).sum(axis=1)
    return idx, counts


def boot_ci_fast(x, counts):
    """x: (T, n) data. -> (T,) lo, hi using the canonical percentile definition."""
    means = (x @ counts.T) / x.shape[1]           # (T, samples)
    lo_i = int(math.floor(LO_POS))
    hi_i = int(math.floor(HI_POS))
    part = np.partition(means, (lo_i, lo_i + 1, hi_i, hi_i + 1), axis=1)
    lo = part[:, lo_i] + (part[:, lo_i + 1] - part[:, lo_i]) * (LO_POS - lo_i)
    hi = part[:, hi_i] + (part[:, hi_i + 1] - part[:, hi_i]) * (HI_POS - hi_i)
    return lo, hi


def validate(counts, n_check=400, seed=20260806):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_check, N))
    lo, hi = boot_ci_fast(x, counts)
    max_lo = max_hi = 0.0
    for i in range(n_check):
        ref = paired_bootstrap_ci({j: 0.0 for j in range(N)},
                                  {j: float(x[i, j]) for j in range(N)})
        max_lo = max(max_lo, abs(ref["ci95_low"] - lo[i]))
        max_hi = max(max_hi, abs(ref["ci95_high"] - hi[i]))
    return max_lo, max_hi


def coverage(sampler, trials, counts, chunk=2000, label=""):
    cov_b = cov_t = 0
    wb = wt = 0.0
    wratio = []
    done = 0
    while done < trials:
        m = min(chunk, trials - done)
        x = sampler(m)
        lo, hi = boot_ci_fast(x, counts)
        mean = x.mean(axis=1)
        sd = x.std(axis=1, ddof=1)
        half = TCRIT4 * sd / math.sqrt(N)
        tlo, thi = mean - half, mean + half
        cov_b += int(((lo <= 0) & (hi >= 0)).sum())
        cov_t += int(((tlo <= 0) & (thi >= 0)).sum())
        wb += float((hi - lo).sum())
        wt += float((thi - tlo).sum())
        wratio.extend(((hi - lo) / np.where(thi - tlo > 0, thi - tlo, np.nan)).tolist())
        done += m
    wr = np.array(wratio, dtype=float)
    return {
        "label": label, "trials": trials,
        "coverage_bootstrap": cov_b / trials,
        "coverage_t": cov_t / trials,
        "mc_se": math.sqrt((cov_b / trials) * (1 - cov_b / trials) / trials),
        "mean_width_bootstrap": wb / trials,
        "mean_width_t": wt / trials,
        "width_ratio_of_means": (wb / trials) / (wt / trials),
        "mean_width_ratio": float(np.nanmean(wr)),
    }


def coverage_freshseed(sampler, trials, seed=7, chunk=1):
    """Same estimator but with a NEW resampling design per trial (seed not frozen)."""
    rng = np.random.default_rng(seed)
    cov = 0
    widths = 0.0
    for _ in range(trials):
        x = sampler(1)[0]
        idx = rng.integers(0, N, size=(SAMPLES, N))
        means = np.sort(x[idx].mean(axis=1))
        lo_i, hi_i = int(math.floor(LO_POS)), int(math.floor(HI_POS))
        lo = means[lo_i] + (means[lo_i + 1] - means[lo_i]) * (LO_POS - lo_i)
        hi = means[hi_i] + (means[hi_i + 1] - means[hi_i]) * (HI_POS - hi_i)
        cov += int(lo <= 0 <= hi)
        widths += hi - lo
    return {"trials": trials, "coverage_bootstrap_freshseed": cov / trials,
            "mean_width": widths / trials,
            "mc_se": math.sqrt((cov / trials) * (1 - cov / trials) / trials)}


def main():
    global TCRIT4
    out = {}
    idx, counts = recover_design()
    ml, mh = validate(counts)
    print(f"design recovery validation vs canonical paired_bootstrap_ci over 400 random "
          f"datasets: max |dlo| = {ml:.3e}, max |dhi| = {mh:.3e}")
    out["design_validation_max_abs_diff"] = [ml, mh]
    assert ml < 1e-12 and mh < 1e-12, "recovered design does not match the canonical tool"

    # how uniform is the frozen design?
    cnt = counts.astype(int)
    multiset = {}
    for row in cnt:
        key = tuple(sorted(row.tolist(), reverse=True))
        multiset[key] = multiset.get(key, 0) + 1
    total = sum(multiset.values())
    from math import comb, factorial

    def multinom_prob(shape):
        # probability of a resample whose sorted count shape == `shape`
        k = factorial(N)
        for c in set(shape):
            k //= factorial(shape.count(c))
        perms = k
        p = perms * factorial(N)
        for c in shape:
            p //= factorial(c)
        return p / (N ** N)

    print("\nfrozen (seed=1) design: empirical vs ideal multinomial shape frequencies")
    shapes = sorted(multiset, key=lambda s: -multiset[s])
    for s in shapes:
        print(f"  shape {s}: empirical {multiset[s] / total:.4f}  ideal {multinom_prob(s):.4f}")
    out["design_shape_freq"] = {str(k): v / total for k, v in multiset.items()}
    # fraction of resamples that are a single repeated value (CI-collapsing draws)
    out["frac_degenerate_resamples"] = multiset.get((5, 0, 0, 0, 0), 0) / total

    rng = np.random.default_rng(20260806)

    # ---- condition A: normal
    trials = 100000
    resA = coverage(lambda m: rng.normal(size=(m, N)), trials, counts, label="A normal(0,1)")
    print("\n", resA)
    out["A_normal"] = resA

    # ---- condition B: pooled standardised empirical rep-differences of this campaign
    diffs = json.loads((Path(__file__).parent / "c2_c3_results.json").read_text())["c2"]
    pool = []
    per_cell = {}
    for row in diffs:
        d = np.array(row["per_rep_diffs"], dtype=float)
        c = (d - d.mean()) / d.std(ddof=1)
        pool.extend(c.tolist())
        per_cell[f"{row['model']}|r{row['rate']:.0f}"] = (d - d.mean()).tolist()
    poolarr = np.array(pool)
    print(f"\nempirical pool: n={len(poolarr)} standardised rep-diffs, "
          f"skew {float(((poolarr - poolarr.mean())**3).mean() / poolarr.std()**3):+.3f}, "
          f"excess kurtosis {float(((poolarr - poolarr.mean())**4).mean() / poolarr.std()**4 - 3):+.3f}")
    out["empirical_pool"] = {"n": len(poolarr),
                             "skew": float(((poolarr - poolarr.mean())**3).mean() / poolarr.std()**3),
                             "excess_kurtosis": float(((poolarr - poolarr.mean())**4).mean() / poolarr.std()**4 - 3)}
    resB = coverage(lambda m: rng.choice(poolarr, size=(m, N), replace=True), trials, counts,
                    label="B pooled empirical rep-diffs (standardised)")
    print(" ", resB)
    out["B_empirical_pooled"] = resB

    # ---- condition C: per-cell empirical (the cell's own 5 centred diffs)
    out["C_per_cell"] = {}
    print("\nC per-cell empirical (resample from that cell's own 5 centred diffs):")
    for name, cd in per_cell.items():
        arr = np.array(cd)
        r = coverage(lambda m, a=arr: rng.choice(a, size=(m, N), replace=True), 40000, counts,
                     label=f"C {name}")
        out["C_per_cell"][name] = r
        print(f"  {name:28s} boot {r['coverage_bootstrap']:.4f}  t {r['coverage_t']:.4f}  "
              f"width ratio {r['width_ratio_of_means']:.3f}")

    # ---- condition D: heavy tail / skew
    resD1 = coverage(lambda m: rng.standard_t(3, size=(m, N)), trials, counts, label="D1 t_3")
    ln = lambda m: (np.exp(rng.normal(size=(m, N))) - math.exp(0.5))  # noqa: E731
    resD2 = coverage(ln, trials, counts, label="D2 centred lognormal")
    print("\n", resD1, "\n", resD2)
    out["D1_t3"] = resD1
    out["D2_lognormal"] = resD2

    # ---- fresh-seed control
    fs = coverage_freshseed(lambda m: rng.normal(size=(m, N)), 20000)
    print("\nfresh-seed control (design NOT frozen), normal:", fs)
    out["freshseed_normal"] = fs

    # ---- other n, for blast radius
    print("\nblast radius: coverage at other n (normal data, frozen seed=1 design)")
    out["other_n"] = {}
    for n in (3, 4, 5, 6, 8, 10):
        _, cn = recover_design(n=n)
        globals()["N"] = n  # coverage() uses module-level N for t df
        tc = {2: 12.706204736, 3: 4.302652730, 4: 3.182446305, 5: 2.776445105,
              6: 2.570581836, 7: 2.446911851, 8: 2.364624252, 9: 2.262157163}[n - 1]
        old = TCRIT4
        TCRIT4 = tc
        r = coverage(lambda m, k=n: rng.normal(size=(m, k)), 50000, cn, label=f"n={n}")
        TCRIT4 = old
        globals()["N"] = 5
        out["other_n"][n] = r
        print(f"  n={n:2d}: bootstrap {r['coverage_bootstrap']:.4f} (width {r['mean_width_bootstrap']:.3f})"
              f"  t {r['coverage_t']:.4f} (width {r['mean_width_t']:.3f})"
              f"  ratio {r['width_ratio_of_means']:.3f}")

    (Path(__file__).parent / "c1_coverage.json").write_text(json.dumps(out, indent=1, default=str))
    print("\nwrote c1_coverage.json")


if __name__ == "__main__":
    main()
