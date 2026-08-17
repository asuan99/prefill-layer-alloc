"""Prereg addendum A-3-2 threat, executed: a SYMMETRIC (curvature) position
drift is un-balanceable with 4 blocks and pushes the argmin toward the arm
that sits nearest the centre of the position axis.  d44 has the SMALLEST
sum (pos-3)^2 of all 7 arms (8 vs 10..26).  Does a within-arm estimate of the
curvature term survive, and does correcting for it move D_ttft off d44?
"""
import json, glob, os, re, statistics, math, random

SLO = "/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/slo_sched"
HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "indep_perboot.json")))
SMn = lambda a: int(a[1:])

# position of each (arm, block) from t_boot0
pos = {}
rows = []
for p in sorted(glob.glob(os.path.join(SLO, "g16_blk*_sidecar.json"))):
    s = json.load(open(p))
    m = re.match(r"g16_(blk\d)_(d\d+)_boot1_", os.path.basename(p))
    rows.append((m.group(1), m.group(2), s["t_boot0"]))
for b in sorted({r[0] for r in rows}):
    rs = sorted([r for r in rows if r[0] == b], key=lambda r: r[2])
    for i, r in enumerate(rs):
        pos[(r[1], r[0])] = i

def fit(phase, key, basis):
    grid = D[phase]
    arms = sorted(grid, key=SMn)
    blocks = sorted(grid["d44"])
    # within-arm (fixed-effect) slope on the chosen position basis
    num = den = 0.0
    for a in arms:
        xs = [basis(pos[(a, b)]) for b in blocks]
        ys = [grid[a][b][key] for b in blocks]
        xb, yb = statistics.fmean(xs), statistics.fmean(ys)
        for x, y in zip(xs, ys):
            num += (x - xb) * (y - yb)
            den += (x - xb) ** 2
    c = num / den
    xbar_all = statistics.fmean(basis(pos[(a, b)]) for a in arms for b in blocks)
    corr = {}
    for a in arms:
        xb = statistics.fmean(basis(pos[(a, b)]) for b in blocks)
        yb = statistics.fmean(grid[a][b][key] for b in blocks)
        corr[a] = yb - c * (xb - xbar_all)
    return c, corr

for phase in ("LO", "HI"):
    print("=" * 74, phase)
    for name, basis in (("linear  (pos-3)", lambda p: p - 3),
                        ("curvature (pos-3)^2", lambda p: (p - 3) ** 2)):
        for key in ("M_ttft", "M_itl"):
            c, corr = fit(phase, key, basis)
            raw = {a: statistics.fmean(D[phase][a][b][key] for b in sorted(D[phase][a]))
                   for a in D[phase]}
            am_raw = min(raw, key=lambda a: (raw[a], SMn(a)))
            am_cor = min(corr, key=lambda a: (corr[a], SMn(a)))
            print(f"  {name:22s} {key}: c={c:+.4f} per unit   argmin raw={am_raw}"
                  f"  argmin corrected={am_cor}")
            if key == "M_ttft":
                print("      corrected means: " + "  ".join(
                    f"{a}:{corr[a]:.1f}" for a in sorted(corr, key=SMn)))
            else:
                print("      corrected means: " + "  ".join(
                    f"{a}:{corr[a]:.3f}" for a in sorted(corr, key=SMn)))
    # how big would a curvature effect have to be to erase the d44 advantage?
    grid = D[phase]; blocks = sorted(grid["d44"])
    raw = {a: statistics.fmean(grid[a][b]["M_ttft"] for b in blocks) for a in grid}
    S = {a: statistics.fmean((pos[(a, b)] - 3) ** 2 for b in blocks) for a in grid}
    runner = min((a for a in raw if a != "d44"), key=lambda a: raw[a])
    need = (raw[runner] - raw["d44"]) / (S[runner] - S["d44"]) if S[runner] != S["d44"] else float("nan")
    print(f"  d44 vs runner-up {runner}: gap {raw[runner]-raw['d44']:+.1f} ms; "
          f"mean (pos-3)^2  d44={S['d44']:.2f} {runner}={S[runner]:.2f}; "
          f"curvature c needed to erase = {need:+.2f} ms/unit "
          f"(=> edge-vs-centre position penalty {9*need:+.0f} ms)")
    # within-arm SD of the estimate of c (block bootstrap)
    random.seed(7)
    cs = []
    for _ in range(4000):
        bs = [random.choice(blocks) for _ in blocks]
        num = den = 0.0
        for a in grid:
            xs = [(pos[(a, b)] - 3) ** 2 for b in bs]
            ys = [grid[a][b]["M_ttft"] for b in bs]
            xb, yb = statistics.fmean(xs), statistics.fmean(ys)
            for x, y in zip(xs, ys):
                num += (x - xb) * (y - yb); den += (x - xb) ** 2
        if den > 0:
            cs.append(num / den)
    cs.sort()
    print(f"  estimated curvature c (M_ttft, within-arm) = {fit(phase,'M_ttft',lambda p:(p-3)**2)[0]:+.2f}"
          f"  block-bootstrap 95% [{cs[int(.025*len(cs))]:+.2f}, {cs[int(.975*len(cs))]:+.2f}]")
