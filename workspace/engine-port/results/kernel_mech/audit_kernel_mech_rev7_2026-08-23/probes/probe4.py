"""E3: sec1.3 claims the n-axis proposition ('buying more boots does not fix it')
survives unchanged, citing the rev6 audit's sweep -- which was run on the
REDUCED rule that sec1.3 itself just declared wrong.  Re-run the n sweep under
the REGISTERED rule."""
import sys, numpy as np
sys.path.insert(0, "/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/kernel_mech")
import rev7_power as R

def sweep(truthname, cv, n_boots, n_rep=60, B=4000):
    truth = R.TRUTHS[truthname]
    for nb in n_boots:
        rng = np.random.default_rng(1)
        counts = {}
        for _ in range(n_rep):
            cells = {}
            (k44, g44), (k92, g92) = truth
            for nm, (k, g) in (("lo", (k44, g44)), ("hi", (k92, g92))):
                span = k + g; ge = R.ETA*span/(1-R.ETA)
                cells[nm] = R.simulate_cell(rng, k, g, ge, n_boot=nb,
                                            cv_boot=cv, cv_win=cv*2)
            v, _ = R.decide(cells, rng, B)
            counts[v] = counts.get(v, 0) + 1
        print("  %-24s cv=%.3f n_boot=%2d -> %s" % (truthname, cv, nb,
              {k: round(c/n_rep, 3) for k, c in sorted(counts.items())}))

print("=== n sweep under the REGISTERED rule (B=4000, 60 reps) ===")
for t in ("gap_constant_kernel_ok", "kernel_does_not_shrink", "gap_actually_grows"):
    sweep(t, 0.05, (4, 8, 16))
