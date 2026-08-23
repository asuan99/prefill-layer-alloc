"""E4: sec1.4 row 3 claims paired resampling has a NON-UNIFORM direction
('GAP 0.84->0.96, KERNEL 0.80->0.68'), from n_rep = max(30, 200//4) = 50.
Binomial SE at p~0.8, n=50 is 5.7pp; the difference of two such is ~8pp, so a
0.12 swing is ~1.5 sigma.  Re-run at n_rep=200 with two seeds."""
import sys, numpy as np
sys.path.insert(0, "/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/kernel_mech")
import rev7_power as R

def run(seed, paired, n_rep=200, B=10000, cv=0.05):
    out = {}
    for tname, truth in R.TRUTHS.items():
        rng = np.random.default_rng(seed)
        counts = {}
        for _ in range(n_rep):
            cells = R._cells_for(rng, truth, cv)
            v, _ = R.decide(cells, rng, B, paired=paired)
            counts[v] = counts.get(v, 0) + 1
        out[tname] = {k: c/n_rep for k, c in counts.items()}
    return out

print("n_rep=200, cv=0.05  (shipped table used n_rep=50)")
for seed in (1, 202608):
    ind = run(seed, False); pai = run(seed, True)
    print("-- seed %d" % seed)
    for t in R.TRUTHS:
        key = {"kernel_does_not_shrink": "KERNEL_DOMINATED",
               "gap_actually_grows": "GAP_DOMINATED",
               "gap_constant_kernel_ok": "COUNTERFACTUAL_SENSITIVE",
               "mixed": "COUNTERFACTUAL_SENSITIVE"}[t]
        a = ind[t].get(key, 0.0); b = pai[t].get(key, 0.0)
        se = (a*(1-a)/200 + b*(1-b)/200) ** 0.5
        print("   %-24s %-26s indep %.3f -> paired %.3f   (delta %+0.3f, SE %.3f, %.1f sigma)"
              % (t, key, a, b, b-a, se, abs(b-a)/se if se else 0))
