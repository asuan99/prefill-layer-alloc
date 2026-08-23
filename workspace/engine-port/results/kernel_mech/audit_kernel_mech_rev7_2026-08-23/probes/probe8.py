"""HOST_DOMINATED is decided by a POINT comparison (decide(): inter.point >
gap.point) inside a rule that otherwise forbids anything but BCa CIs -- and by
the registered ORDER it preempts every substantive label."""
import sys, numpy as np
sys.path.insert(0, "/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/kernel_mech")
import rev7_power as R

truth = R.TRUTHS["mixed"]           # (70,30) -> (45,40)
for eta in (0.10, 0.28, 0.30, 0.32):
    k, g = truth[1]; span = k + g; ge = eta*span/(1-eta); T = span + ge
    for cv in (0.01, 0.05):
        rng = np.random.default_rng(5)
        host = 0; n = 60
        for _ in range(n):
            cells = R._cells_for(rng, truth, cv, eta=eta)
            v, _ = R.decide(cells, rng, 4000)
            host += int(v == "HOST_DOMINATED")
        print("  eta=%.2f (true frac_inter=%.3f vs true gap_frac_intra=%.3f, "
              "delta=%+.3f)  cv=%.2f -> P(HOST_DOMINATED)=%.3f"
              % (eta, ge/T, g/T, ge/T - g/T, cv, host/n))
