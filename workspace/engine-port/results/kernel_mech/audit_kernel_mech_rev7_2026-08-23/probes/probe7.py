import sys, numpy as np
sys.path.insert(0, "/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/kernel_mech")
import rev7_power as R

print("=== S5a witness: is the S<=0 ceiling really load-bearing? ===")
base, mut, rate = R._ceiling_flips()
print("   ceiling=0.05 -> %s ;  ceiling=1.0 -> %s ;  observed nonfinite rate = %.4f"
      % (base, mut, rate))
print("   (load-bearing iff base != mut AND 0.05 < rate < 1.0):",
      base != mut and 0.05 < rate < 1.0)

print("\n=== A3_NEGATIVE_INTER inside the simulator: can G_inter go negative? ===")
rng = np.random.default_rng(1)
mn = 1e18
for t in R.TRUTHS.values():
    for cv in (0.01, 0.05, 0.2):
        c = R._cells_for(rng, t, cv)
        for nm in ("lo", "hi"):
            mn = min(mn, float(c[nm]["Ge"].min()))
print("   min G_inter over every simulated cell = %.4f  (log-normal draw => >0 by "
      "construction; the simulator can never exercise A3_NEGATIVE_INTER)" % mn)

print("\n=== S1b: does the partition identity catch a MIS-SLICED decomposition? ===")
# sec2 definitions applied to an explicit timeline.
def decompose(Kset, next_first_start):
    iv = sorted(Kset); merged = []
    for s, e in iv:
        if merged and s <= merged[-1][1]: merged[-1][1] = max(merged[-1][1], e)
        else: merged.append([s, e])
    sig = sum(e - s for s, e in merged)
    span = merged[-1][1] - merged[0][0]
    g_intra = span - sig
    g_inter = next_first_start - merged[-1][1]
    t_step = sig + g_intra + g_inter          # sec2: T_step is DEFINED as the sum
    return sig, g_intra, g_inter, t_step

true_k = [(0, 1), (2, 3)]              # kernels of step k
true_k1 = [(5, 6), (7, 8)]             # kernels of step k+1
sig, gi, ge, ts = decompose(true_k, true_k1[0][0])
print("   correct slice : sum=%.0f G_intra=%.0f G_inter=%.0f T_step=%.0f "
      "-> identity %s , gap_frac_intra=%.3f"
      % (sig, gi, ge, ts, abs(sig+gi+ge-ts) < 1e-12, gi/ts))
mis_k = true_k + [true_k1[0]]           # one kernel of k+1 mis-attributed to k
sig2, gi2, ge2, ts2 = decompose(mis_k, true_k1[1][0])
print("   MIS-sliced    : sum=%.0f G_intra=%.0f G_inter=%.0f T_step=%.0f "
      "-> identity %s , gap_frac_intra=%.3f"
      % (sig2, gi2, ge2, ts2, abs(sig2+gi2+ge2-ts2) < 1e-12, gi2/ts2))
print("   => the identity holds in BOTH, while the headline descriptive quantity "
      "moves %.3f -> %.3f" % (gi/ts, gi2/ts2))
print("   A5_CONTAINMENT on the mis-sliced case (all K(k) end < first start of K(k+1)):",
      max(e for _, e in mis_k) < true_k1[1][0], "  <- discard rule does NOT fire")
