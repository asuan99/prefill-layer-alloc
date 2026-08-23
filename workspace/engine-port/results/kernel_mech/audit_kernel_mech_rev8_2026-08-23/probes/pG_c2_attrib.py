"""Probe G -- rev7 had HOST_DOMINATED 0.01 only in kernel_does_not_shrink|cv=0.05,
a cell whose gate margin (0.55-0.00) makes stage 1 pass ~always.  So is the
label's disappearance from the rev8 table due to C2 (interval) or C9 (gate)?"""
import sys, numpy as np
sys.path.insert(0, "/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/kernel_mech")
import rev7_power as R
for with_gate in (False, True):
    rng = np.random.default_rng(1); cnt = {}
    for _ in range(150):
        cells = R._cells_for(rng, R.TRUTHS["kernel_does_not_shrink"], 0.05,
                             eps1=R.EPS1_SCEN if with_gate else None)
        v = R.decide(cells, rng, 3000)[0]; cnt[v] = cnt.get(v,0)+1
    print(f"  with_gate={with_gate}: {cnt}")
print("  (rev7, point-estimate HOST_DOMINATED, no gate: HOST_DOMINATED 0.01)")
