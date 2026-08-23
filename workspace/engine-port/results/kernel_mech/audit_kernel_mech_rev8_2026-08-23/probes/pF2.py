"""Probe F2 -- exact uniform off-by-one: every analysed step keeps exactly N kernels."""
N_STEPS, N_K = 42, 5
ev = []; t = 0.0
for s in range(N_STEPS):
    for i in range(N_K):
        ev.append([s, t, t + 2.0]); t += 3.0
    t += 3.0                                   # inter-step gap = 4.0 total
from collections import defaultdict
def metrics(assign, drop_edges=True):
    K = defaultdict(list)
    for (ts, a, b), lab in zip(ev, assign):
        if lab is not None: K[lab].append((a, b))
    ks = sorted(K)
    if drop_edges: ks = ks[1:-1]
    counts = [len(K[k]) for k in ks]
    rows = []
    for j, k in enumerate(ks[:-1]):
        iv = sorted(K[k]); nxt = sorted(K[ks[j+1]])
        sig = sum(b-a for a, b in iv); span = iv[-1][1]-iv[0][0]
        gi = span - sig; ge = nxt[0][0]-iv[-1][1]
        rows.append((sig, gi, ge, sig+gi+ge))
    return counts, sum(r[1] for r in rows)/sum(r[3] for r in rows), \
           sum(r[2] for r in rows)/sum(r[3] for r in rows)

truth = [e[0] for e in ev]
c0, g0, i0 = metrics(truth)
# exact uniform shift: kernel j -> step floor((j-1)/N_K)
uni = [None if j == 0 else (j-1)//N_K for j in range(len(ev))]
c1, g1, i1 = metrics(uni)
print(f"TRUE            counts={set(c0)}  gap_frac_intra={g0:.4f}  frac_inter={i0:.4f}")
print(f"UNIFORM shift   counts={set(c1)}  gap_frac_intra={g1:.4f}  frac_inter={i1:.4f}")
mode = max(set(c1), key=c1.count)
print(f"A5' anomaly rate under uniform misattribution = "
      f"{sum(1 for c in c1 if c != mode)/len(c1):.3f}   (registered ceiling 0.10 -> CELL ACCEPTED)")
print(f"gap_frac_intra error under that accepted cell  = {(g1-g0)/g0*100:+.1f}%")
