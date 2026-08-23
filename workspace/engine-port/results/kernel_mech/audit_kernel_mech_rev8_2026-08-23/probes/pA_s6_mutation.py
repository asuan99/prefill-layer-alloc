"""Probe A -- is S6 (the C8 regression guard) a REAL mutation test?
Revert the C8 repair (hardwire the global count) and re-run the three S6 checks.
Read-only: monkeypatches an imported module in memory, touches no repo file."""
import sys, numpy as np
sys.path.insert(0, "/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/kernel_mech")
import rev7_power as R

def s6(tag):
    c = R._cells_for(np.random.default_rng(5), R.TRUTHS["mixed"], 0.0)
    full = R._cell_sums_obs(c["hi"])["n"]
    jk = [R._cell_sums_obs(j["hi"])["n"] for j in R._jack_cells({"hi": c["hi"]})]
    a = all(x < full for x in jk)
    b = all(x != R.N_BOOT * R.N_WIN for x in jk)
    cc = R._cells_for(np.random.default_rng(5), R.TRUTHS["mixed"], 0.0)
    def ev(cells, k1, k2):
        return R.stat_eps("a","b",16,44)({
            "a": {k: np.array([v]) for k, v in R._cell_sums_obs(cells[k1]).items()},
            "b": {k: np.array([v]) for k, v in R._cell_sums_obs(cells[k2]).items()}})[0]
    e_full = ev(cc, "lo", "hi")
    j = R._jack_cells({"a": cc["lo"], "b": cc["hi"]})[0]
    e_jack = ev(j, "a", "b")
    c_ok = abs(e_full - e_jack) < 1e-9
    print(f"[{tag}] S6a={'PASS' if a else 'FAIL'} (full={full} jack={sorted(set(jk))})")
    print(f"[{tag}] S6b={'PASS' if b else 'FAIL'}")
    print(f"[{tag}] S6c={'PASS' if c_ok else 'FAIL'}  eps_full={e_full:.6f} eps_jack={e_jack:.6f} d={e_full-e_jack:+.6f}")
    return a, b, c_ok

print("== SHIPPED ==");  base = s6("ship")

# --- MUTATION: undo C8 -- put the global count back
_orig_sums, _orig_obs = R._cell_sums, R._cell_sums_obs
def mut_sums(cell, b, w):
    d = _orig_sums(cell, b, w); d["n"] = float(R.N_BOOT * R.N_WIN); return d
def mut_obs(cell):
    d = _orig_obs(cell); d["n"] = float(R.N_BOOT * R.N_WIN); return d
R._cell_sums, R._cell_sums_obs = mut_sums, mut_obs
print("== MUTANT (C8 reverted: n hardwired to N_BOOT*N_WIN) ==");  mut = s6("mut")
print("\nVERDICT: S6 is a genuine mutation test iff every shipped PASS becomes FAIL.")
print("  shipped:", base, " mutant:", mut)
