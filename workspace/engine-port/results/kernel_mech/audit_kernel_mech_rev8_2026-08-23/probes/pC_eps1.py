"""Probe C -- the new sec1.3 column (PHENOMENON_ABSENT 3.5-21.5%) is a function of
EPS1_SCEN=0.55, an unswept constant that sec3.5's "exhaustive 19" does not list.
Also: is HOST_DOMINATED reachable under the FULL 4-stage rule with a
gate-PASSING host-heavy truth?"""
import sys, math, numpy as np
sys.path.insert(0, "/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/kernel_mech")
import rev7_power as R

def eps2_of(truth, eta=R.ETA):
    (k44,g44),(k92,g92) = truth
    t44 = (k44+g44)/(1-eta); t92 = (k92+g92)/(1-eta)
    return -math.log(t92/t44)/math.log(92/44)

print("== implied eps_T(44->92) of each imposed truth (vs EPS1_SCEN=0.55) ==")
for n,t in R.TRUTHS.items(): print(f"   {n:26s} eps2={eps2_of(t):.4f}  margin={0.55-eps2_of(t):+.4f}")

print("\n== sec1.3 PHENOMENON_ABSENT rate vs EPS1_SCEN (cv=0.05, n_rep=60, B=3000) ==")
print("   eps1 | " + " | ".join(f"{n[:12]:12s}" for n in R.TRUTHS))
for eps1 in (0.80, 0.55, 0.45, 0.35):
    row = []
    for n,t in R.TRUTHS.items():
        rng = np.random.default_rng(1); cnt = 0
        for _ in range(60):
            cells = R._cells_for(rng, t, 0.05, eps1=eps1)
            if R.decide(cells, rng, 3000)[0] == "PHENOMENON_ABSENT_IN_CELL_A": cnt += 1
        row.append(f"{cnt/60:12.3f}")
    print(f"   {eps1:4.2f} | " + " | ".join(row))

print("\n== HOST_DOMINATED under the FULL 4-stage rule, gate-PASSING host-heavy truth ==")
for truth, eta in ((((50,10),(45,10)), 0.6), (((50,10),(48,10)), 0.6)):
    print(f"   truth={truth} eta={eta} eps2={eps2_of(truth,eta):.3f}")
    rng = np.random.default_rng(21); lab = {}
    for _ in range(40):
        cells = R._cells_for(rng, truth, 0.01, eta=eta, eps1=0.55)
        v = R.decide(cells, rng, 2000)[0]; lab[v] = lab.get(v,0)+1
    print("      ->", lab)
