"""E2: does the PRIMARY precision claim ('half-width 0.5-1.0 pp is usable')
survive when the TRUE intra-step gap is small?  The four registered truths all
put gap_frac_intra(D=92) between 18% and 59%.  A cudagraph-ON decode step whose
kernels replay back-to-back would sit far below that."""
import sys, numpy as np
sys.path.insert(0, "/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/kernel_mech")
import rev7_power as R

def gapfrac(k, g, eta=R.ETA):
    span = k + g; ge = eta*span/(1-eta); return g/(span+ge)

print("registered truths: gap_frac_intra(D=92) =",
      {n: round(gapfrac(*t[1]), 4) for n, t in R.TRUTHS.items()})

EXTRA = {
    "gap_small_4pct":  ((96, 4),  (96, 4)),
    "gap_small_2pct":  ((98, 2),  (98, 2)),
    "gap_tiny_0p5pct": ((99.5, 0.5), (99.5, 0.5)),
}
print("audit truths     : gap_frac_intra(D=92) =",
      {n: round(gapfrac(*t[1]), 4) for n, t in EXTRA.items()})
rng = np.random.default_rng(1)
for name, truth in EXTRA.items():
    for cv in (0.01, 0.025):
        half = []; inval = []
        for _ in range(60):
            cells = R._cells_for(rng, truth, cv)
            c = R.ci_for({"hi": cells["hi"]}, R.stat_gap_frac("hi"), rng, 10000)
            half.append((c["hi"]-c["lo"])/2)
            inval.append(R.invalid_rate(cells["hi"]))
        half = np.array(half)*100
        true_pp = gapfrac(*truth[1])*100
        print("  %-16s cv=%-6s true=%5.2f pp | half-width med=%5.2f pp "
              "(=%4.0f%% of the value) | P(CI excludes 0)=%.2f | neg-window rate=%.3f"
              % (name, cv, true_pp, np.median(half), 100*np.median(half)/true_pp,
                 np.mean(np.array(half) < true_pp), np.mean(inval)))
