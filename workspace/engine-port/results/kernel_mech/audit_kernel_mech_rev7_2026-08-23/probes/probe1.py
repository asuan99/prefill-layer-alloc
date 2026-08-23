import sys, math, numpy as np
sys.path.insert(0, "/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/kernel_mech")
import rev7_power as R

rng = np.random.default_rng(1)

# ---- P1: does the jackknife normalisation bug exist for multi-cell stats? ----
# stat_eps / stat_s divide cell SUMS by the GLOBAL constant N_BOOT*N_WIN (=80).
# _jack_cells deletes one boot from ONE cell -> that cell's sum covers 3*20=60
# windows but is still divided by 80.
cells = R._cells_for(rng, R.TRUTHS["mixed"], 0.01)
# two-cell eps stat
c16 = R.simulate_cell(rng, 60, 30, 10, cv_boot=0.01, cv_win=0.02)
c44 = R.simulate_cell(rng, 60, 30, 10, cv_boot=0.01, cv_win=0.02)
cc = {"c16": c16, "c44": c44}
stat = R.stat_eps("c16", "c44", 16, 44)
theta_hat = stat({k: {kk: np.array([vv]) for kk, vv in R._cell_sums_obs(cc[k]).items()} for k in cc})[0]
jack = np.array([stat({k: {kk: np.array([vv]) for kk, vv in R._cell_sums_obs(s[k]).items()} for k in cc})[0]
                 for s in R._jack_cells(cc)])
print("eps  theta_hat = %.6f" % theta_hat)
print("eps  jackknife values:", np.round(jack, 4))
print("eps  jack spread = %.4f  (artefact size ln(4/3)/ln(44/16) = %.4f)"
      % (jack.max()-jack.min(), math.log(4/3)/math.log(44/16)))
jm = jack.mean(); num = ((jm-jack)**3).sum(); den = 6*(((jm-jack)**2).sum()**1.5)
print("eps  BCa acceleration a = %.6f" % (num/den if den else 0.0))

# same for stat_s
two = {"lo": cells["lo"], "hi": cells["hi"]}
sA = R.stat_s("A")
th = sA({k: {kk: np.array([vv]) for kk, vv in R._cell_sums_obs(two[k]).items()} for k in two})[0]
jk = np.array([sA({k: {kk: np.array([vv]) for kk, vv in R._cell_sums_obs(s[k]).items()} for k in two})[0]
               for s in R._jack_cells(two)])
print("\ns_A  theta_hat = %.6f" % th)
print("s_A  jackknife values:", np.round(jk, 4))
jm = jk.mean(); num = ((jm-jk)**3).sum(); den = 6*(((jm-jk)**2).sum()**1.5)
print("s_A  BCa acceleration a = %.6f" % (num/den if den else 0.0))
