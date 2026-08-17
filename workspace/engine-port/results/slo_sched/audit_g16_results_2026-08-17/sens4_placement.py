"""H-7 audit: is the 'residency does not depend on placement' inference
supported, and is the t(6) over 7 arms a legitimate error term?"""
import json, os, math, statistics, itertools, random

RES = "/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/slo_sched/residency_scope_2026-08-17/residency_scope_2026-08-17.json"
d = json.load(open(RES))
boots = [b for b in d["boots"] if b.get("mode") == "full"]
ARMS = sorted({b["arm"] for b in boots}, key=lambda a: int(a[1:]))
BLKS = sorted({b["block"] for b in boots})
W = {(b["arm"], b["block"]): b["W_tim_all"] * 100 for b in boots}
print("n boots", len(boots), "arms", ARMS, "blocks", BLKS)

armmean = {a: statistics.fmean(W[(a, k)] for k in BLKS) for a in ARMS}
rel = {(a, k): 100 * (W[(a, k)] - armmean[a]) / armmean[a] for a in ARMS for k in BLKS}
T = {6: 2.447, 3: 3.182, 2: 4.303}

def paired(vals, dof):
    m, s = statistics.fmean(vals), statistics.stdev(vals)
    h = T[dof] * s / math.sqrt(len(vals))
    return m, s, m - h, m + h

print("\n-- block relative deviation, paired over the 7 arms (as the doc does) --")
for k in BLKS:
    m, s, lo, hi = paired([rel[(a, k)] for a in ARMS], 6)
    print(f"  {k}: {m:+.2f} +- {s:.2f}  t(6) CI [{lo:+.2f}, {hi:+.2f}]")

print("\n-- placement contrasts, paired over the 7 arms --")
c1 = [rel[(a, "blk4")] - (rel[(a, "blk2")] + rel[(a, "blk3")]) / 2 for a in ARMS]
c2 = [rel[(a, "blk1")] - statistics.fmean(rel[(a, k)] for k in ("blk2", "blk3", "blk4")) for a in ARMS]
c3 = [rel[(a, "blk2")] - rel[(a, "blk3")] for a in ARMS]
for name, c in (("blk4 - mean(blk2,blk3)  solo vs concurrent", c1),
                ("blk1 - mean(2,3,4)      gpu42 vs gpu41", c2),
                ("blk2 - blk3             same node, same clock", c3)):
    m, s, lo, hi = paired(c, 6)
    print(f"  {name}: {m:+.2f} +- {s:.2f}  t(6) CI [{lo:+.2f}, {hi:+.2f}]  |m|/CIhalf={abs(m)/(hi-m):.2f}")

print("\n-- PSEUDO-REPLICATION probe: are the 7 arms independent replicates of the")
print("   placement contrast, or 7 correlated readings of ONE placement event? --")
# If the 7 arms were independent replicates, the block deviations would be
# uncorrelated across arms.  Test: does the SIGN of the block deviation agree
# across arms far more often than chance (=> a common block factor)?
for k in BLKS:
    signs = [1 if rel[(a, k)] > 0 else 0 for a in ARMS]
    print(f"   {k}: sign pattern over arms {signs}  (#positive={sum(signs)}/7)")
# the true replication level for 'solo vs concurrent' is:
print("   distinct placement configurations observed:",
      sorted({(b['node'], b['gpu_uuid'][:12], b['block']) for b in boots}))
print("   -> solo-vs-concurrent contrasts available: blk4 vs {blk2,blk3} = ONE contrast")
print("      (blk1 is a different node AND first-in-time, so it cannot serve as a second)")

print("\n-- how big is the arm axis for comparison --")
print(f"   d74/d16 = {armmean['d74']/armmean['d16']:.3f}x  (+{100*(armmean['d74']/armmean['d16']-1):.0f}%)")
for a in ARMS:
    print(f"   {a}: {armmean[a]:.2f}%  (blocks: " + ", ".join(f"{W[(a,k)]:.2f}" for k in BLKS) + ")")

print("\n-- imbalance ratios d16->d74 on EVERY reported denominator/window --")
def ratio(key, phase=None):
    vals = {}
    for a in ARMS:
        xs = []
        for b in boots:
            if b["arm"] != a:
                continue
            xs.append((b["per_phase"][phase][key] if phase else b[key]) * 100)
        vals[a] = statistics.fmean(xs)
    return vals["d16"], vals["d74"], vals["d74"] / vals["d16"]
for key in ("W_tim_all", "W_cnt_da", "W_tim_da", "W_cnt_all"):
    for phase in (None, "LO", "HI"):
        lo, hi, r = ratio(key, phase)
        print(f"   {key:11s} {str(phase or 'whole-span'):10s}: {lo:6.2f}% -> {hi:6.2f}%   ratio {r:.2f}x")
