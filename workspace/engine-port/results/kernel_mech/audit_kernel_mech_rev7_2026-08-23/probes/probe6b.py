"""E5 corrected: identical DATA and identical bootstrap seed for both orders."""
import sys, numpy as np
sys.path.insert(0, "/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/kernel_mech")
import rev7_power as R
def decide_host_last(cells, rng, B=10000):
    two = {"lo": cells["lo"], "hi": cells["hi"]}
    res = {}
    for kind in ("A", "B"):
        c = R.ci_for(two, R.stat_s(kind), rng, B)
        res[kind] = (c["lo"], c["hi"], c["point"], c["nonfinite_rate"])
        if (c["nonfinite_rate"] > R.S_LE0_MAX or not np.isfinite(c["point"])
                or not np.isfinite(c["lo"]) or not np.isfinite(c["hi"])):
            return "UNDETERMINED"
    lab = {}
    for kind in ("A", "B"):
        lo, hi, _, _ = res[kind]
        if hi < R.S_THRESH: lab[kind] = "KERNEL_DOMINATED"
        elif lo > R.S_THRESH: lab[kind] = "GAP_DOMINATED"
        elif R.MULTI_BAND[0] <= lo and hi <= R.MULTI_BAND[1]: lab[kind] = "MULTI_CAUSE"
        else: lab[kind] = "UNDETERMINED"
    v = "COUNTERFACTUAL_SENSITIVE" if lab["A"] != lab["B"] else lab["A"]
    g = R.ci_for({"hi": cells["hi"]}, R.stat_gap_frac("hi"), rng, B)
    i = R.ci_for({"hi": cells["hi"]}, R.stat_frac_inter("hi"), rng, B)
    return "HOST_DOMINATED" if i["point"] > g["point"] else v

rng_data = np.random.default_rng(99)
for tname in R.TRUTHS:
    diff = []; n = 60
    for r in range(n):
        cells = R._cells_for(rng_data, R.TRUTHS[tname], 0.05)
        a = R.decide(cells, np.random.default_rng(1000+r), 4000)[0]
        b = decide_host_last(cells, np.random.default_rng(1000+r), 4000)
        if a != b: diff.append((a, b))
    print("   %-24s label differs in %2d/%d reps  %s" % (tname, len(diff), n, diff[:4]))
