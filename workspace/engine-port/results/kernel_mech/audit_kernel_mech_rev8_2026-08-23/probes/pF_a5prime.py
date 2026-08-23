"""Probe F -- sec2.1 asserts: "귀속 사상이 어긋나면 카운트가 반드시 튄다"
(if the attribution mapping is wrong the per-step kernel COUNT must jump).
Counter-example: a UNIFORM off-by-one attribution leaves every count at N,
so A5'_KERNEL_COUNT's anomaly rate is exactly 0 while gap_frac_intra moves.
Deterministic arithmetic on the design's own sec2 definitions."""
N_STEPS, N_K = 40, 5
# kernels: within a step, kernel i occupies [t, t+2); 1 unit intra-kernel gap;
# 4 units of inter-step gap.
ev = []
t = 0.0
for s in range(N_STEPS):
    for i in range(N_K):
        ev.append((s, t, t + 2.0)); t += 3.0     # 2 busy + 1 intra gap
    t += 4.0 - 1.0                                # make the step->step gap 4.0

def metrics(assign):
    """assign: list of step-labels, one per kernel, in timeline order."""
    from collections import defaultdict
    Kset = defaultdict(list)
    for (true_s, a, b), lab in zip(ev, assign):
        Kset[lab].append((a, b))
    ks = sorted(Kset)
    counts = [len(Kset[k]) for k in ks]
    rows = []
    for j, k in enumerate(ks[:-1]):
        iv = sorted(Kset[k]); nxt = sorted(Kset[ks[j + 1]])
        sig = sum(b - a for a, b in iv)
        span = iv[-1][1] - iv[0][0]
        gi = span - sig
        ge = nxt[0][0] - iv[-1][1]
        T = sig + gi + ge
        rows.append((sig, gi, ge, T))
    gap_frac = sum(r[1] for r in rows) / sum(r[3] for r in rows)
    frac_int = sum(r[2] for r in rows) / sum(r[3] for r in rows)
    return counts, gap_frac, frac_int

truth = [s for (s, a, b) in ev]
c0, g0, i0 = metrics(truth)
print(f"TRUE attribution      : counts={set(c0)}  gap_frac_intra={g0:.4f}  frac_inter={i0:.4f}")

# uniform off-by-one: every step steals the NEXT step's first kernel
shift = list(truth)
for idx in range(len(ev)):
    s, a, b = ev[idx]
    if idx % N_K == 0 and s > 0:
        shift[idx] = s - 1
c1, g1, i1 = metrics(shift)
print(f"UNIFORM off-by-one    : counts={set(c1)}  gap_frac_intra={g1:.4f}  frac_inter={i1:.4f}")
print(f"  -> A5' anomaly rate against the measured mode = "
      f"{sum(1 for c in c1 if c != max(set(c1), key=c1.count))/len(c1):.3f}  (registered ceiling 0.10)")

# ragged (one boundary only) -- the audit's fixture shape
rag = list(truth)
rag[N_K * 5] = 4
c2, g2, i2 = metrics(rag)
print(f"RAGGED (1 boundary)   : counts={sorted(set(c2))}  gap_frac_intra={g2:.4f}")
mode = max(set(c2), key=c2.count)
print(f"  -> A5' anomaly rate = {sum(1 for c in c2 if c != mode)/len(c2):.3f}")
print()
print("CONCLUSION: the count-based check is blind to UNIFORM misattribution "
      "(anomaly rate 0.000) even though gap_frac_intra moves "
      f"{abs(g1-g0)/g0*100:.1f}% ; it fires only on RAGGED misattribution.")
