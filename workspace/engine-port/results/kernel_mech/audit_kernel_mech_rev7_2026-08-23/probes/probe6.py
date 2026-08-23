"""E5: (i) is the registered ORDER load-bearing in decide()?  (ii) does decide()
implement the FIRST element of the registered order (the sec3.3 lever gate)?"""
import sys, inspect, numpy as np
sys.path.insert(0, "/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/kernel_mech")
import rev7_power as R

src = inspect.getsource(R.decide)
print("decide() mentions eps / lever gate:", ("stat_eps" in src) or ("eps" in src))
print("decide() mentions HOST_DOMINATED  :", "HOST_DOMINATED" in src)
print("decide() mentions S_LE0_MAX       :", "S_LE0_MAX" in src)

def decide_host_last(cells, rng, B=10000):
    """Mutation: assign the substantive label FIRST, host condition LAST."""
    two = {"lo": cells["lo"], "hi": cells["hi"]}
    res = {}
    for kind in ("A", "B"):
        c = R.ci_for(two, R.stat_s(kind), rng, B)
        res[kind] = {"lo": c["lo"], "hi": c["hi"], "point": c["point"],
                     "s_le0_rate": c["nonfinite_rate"]}
        if (c["nonfinite_rate"] > R.S_LE0_MAX or not np.isfinite(c["point"])
                or not np.isfinite(c["lo"]) or not np.isfinite(c["hi"])):
            return "UNDETERMINED"
    lab = {}
    for kind in ("A", "B"):
        lo, hi = res[kind]["lo"], res[kind]["hi"]
        if hi < R.S_THRESH: lab[kind] = "KERNEL_DOMINATED"
        elif lo > R.S_THRESH: lab[kind] = "GAP_DOMINATED"
        elif R.MULTI_BAND[0] <= lo and hi <= R.MULTI_BAND[1]: lab[kind] = "MULTI_CAUSE"
        else: lab[kind] = "UNDETERMINED"
    v = "COUNTERFACTUAL_SENSITIVE" if lab["A"] != lab["B"] else lab["A"]
    g = R.ci_for({"hi": cells["hi"]}, R.stat_gap_frac("hi"), rng, B)
    i = R.ci_for({"hi": cells["hi"]}, R.stat_frac_inter("hi"), rng, B)
    return "HOST_DOMINATED" if i["point"] > g["point"] else v

print("\n-- order mutation (host test first = registered, vs host test last) --")
for tname in R.TRUTHS:
    for cv in (0.05,):
        diff = 0; n = 60
        rngA = np.random.default_rng(1); rngB = np.random.default_rng(1)
        for _ in range(n):
            cA = R._cells_for(rngA, R.TRUTHS[tname], cv)
            cB = R._cells_for(rngB, R.TRUTHS[tname], cv)
            a = R.decide(cA, rngA, 4000)[0]
            b = decide_host_last(cB, rngB, 4000)
            diff += int(a != b)
        print("   %-24s cv=%.2f : label differs in %d/%d reps" % (tname, cv, diff, n))
