"""Probe B -- is HOST_DOMINATED reachable under the FULL registered 4-stage
rule (i.e. with the c16 cell present so stage 1 actually runs)?
S4b/S2c in --selftest call decide() WITHOUT c16, so stage 1 is skipped there."""
import sys, numpy as np
sys.path.insert(0, "/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/kernel_mech")
import rev7_power as R

print("-- (1) does the shipped selftest witness include the stage-1 gate? --")
c = R._cells_for(np.random.default_rng(7), ((50,10),(30,10)), 0.01, eta=0.6)
print("   S4b/S2c witness cells:", sorted(c))          # no 'c16' => stage 1 skipped
print("   verdict without gate:",
      R.decide(c, np.random.default_rng(7), 2000)[0])

print("\n-- (2) same witness truth, but WITH c16 (stage 1 runs) --")
for eps1 in (0.55, 0.30, 0.10):
    lab = {}
    for i in range(40):
        cells = R._cells_for(np.random.default_rng(100+i), ((50,10),(30,10)), 0.01,
                             eta=0.6, eps1=eps1)
        v = R.decide(cells, np.random.default_rng(500+i), 2000)[0]
        lab[v] = lab.get(v, 0) + 1
    print(f"   eps1={eps1}: {lab}")

print("\n-- (3) HOST_DOMINATED count across the 12 REGISTERED cells (the shipped run) --")
import json
j = json.load(open("/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/"
                   "results/kernel_mech/REV8_POWER_2026-08-23.json"))
tot = {}
for k, v in j["secondary_registered_rule"].items():
    for lab, p in v["P"].items():
        tot[lab] = tot.get(lab, 0) + p
print("   summed P over 12 cells:", {k: round(v,3) for k,v in sorted(tot.items())})
print("   HOST_DOMINATED present in the sec1.3 table? ->", "HOST_DOMINATED" in tot)

print("\n-- (4) eta is DECLARED SWEPT in the file docstring. Is it? --")
import inspect, re
src = inspect.getsource(R)
print("   ETA constant:", R.ETA)
print("   occurrences of 'eta=' overriding the default in experiments:",
      re.findall(r"eta=[^,\)]+", src))
print("   docstring line:", [l.strip() for l in R.__doc__.splitlines()
                             if "inter-step host share" in l])
