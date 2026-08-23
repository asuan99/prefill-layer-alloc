"""Probe D -- sec3.5-16/18 registers the Stage-0' decision on
"the boot-to-boot CV of gap_frac_intra(92)" with thresholds 0.025 / 0.05,
and tells the reader to look up sec1.3, whose columns are the simulator's
`cv` (= cv_boot of K/T/Ge, with cv_win = 2*cv_boot).  Are these the same
quantity?  If not, the registered branch reads the wrong column."""
import sys, numpy as np
sys.path.insert(0, "/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/kernel_mech")
import rev7_power as R

print("  truth                cv_boot | observed boot-to-boot CV of gap_frac_intra(92) | ratio")
for name, truth in R.TRUTHS.items():
    for cv in (0.01, 0.025, 0.05):
        vals = []
        for rep in range(200):
            rng = np.random.default_rng(1000 + rep)
            c = R._cells_for(rng, truth, cv)["hi"]
            per_boot = c["Gi"].sum(axis=1) / c["T"].sum(axis=1)   # 4 boot-level values
            vals.append(per_boot.std(ddof=1) / per_boot.mean())
        m = float(np.median(vals))
        print(f"  {name:22s} {cv:5.3f} | {m:8.4f}  (p10={np.quantile(vals,.1):.4f} p90={np.quantile(vals,.9):.4f}) | {m/cv:5.2f}x")
print()
print("  => sec3.5-18's thresholds are stated on the OBSERVED CV of gap_frac_intra(92);")
print("     sec1.3's columns are cv_boot.  A ratio far from 1.00 means the branch")
print("     'look up sec1.3 with the observed CV' reads the wrong column.")
