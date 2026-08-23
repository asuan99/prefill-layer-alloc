"""E1: is the two-cell BCa acceleration term computed from a corrupted jackknife?
Minimal fix: feed CELL MEANS instead of CELL SUMS.  Every registered statistic
(gap_frac, frac_inter, eps, s) is scale-invariant, so for full-size cells the
bootstrap replicates are IDENTICAL; only the delete-one jackknife changes,
because a deleted-boot cell has 3*20=60 windows but is divided by the GLOBAL
constant 4*20=80 in the shipped code."""
import sys, numpy as np
sys.path.insert(0, "/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/kernel_mech")
import rev7_power as R

KEYS = ("K", "Gi", "Ge", "T")
def cell_means(cell, b, w):
    return {k: R._gather(cell[k], b, w).mean(axis=(1, 2)) for k in KEYS}
def cell_means_obs(cell):
    return {k: float(cell[k].mean()) for k in KEYS}

def run(fixed, n_rep, B, what):
    if fixed:
        R._cell_sums, R._cell_sums_obs = cell_means, cell_means_obs
    else:
        R._cell_sums, R._cell_sums_obs = ORIG_S, ORIG_O
    rng = np.random.default_rng(1)
    if what == "gate":
        grid = [(0.55, 0.05), (0.40, 0.20), (0.30, 0.30), (0.20, 0.20), (0.60, 0.40)]
        return R.exp_gate(rng, n_rep, B, grid)
    return R.exp_secondary(rng, n_rep, B)

ORIG_S, ORIG_O = R._cell_sums, R._cell_sums_obs
import json
for what, nrep in (("gate", 50), ("secondary", 60)):
    a = run(False, nrep, 10000, what)
    b = run(True, nrep, 10000, what)
    print("=== %s : SHIPPED (sums) vs FIXED (means) jackknife, same seed ===" % what)
    for k in a:
        va = a[k] if what == "gate" else a[k]["P"]
        vb = b[k] if what == "gate" else b[k]["P"]
        if what == "gate":
            print("  %-32s  shipped P=%.3f   fixed P=%.3f" % (k, va["P_gate_passes"], vb["P_gate_passes"]))
        else:
            print("  %-32s\n      shipped %s\n      fixed   %s" % (k, {x: round(y,3) for x,y in va.items()}, {x: round(y,3) for x,y in vb.items()}))
