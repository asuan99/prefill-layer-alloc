"""Is the ordering that sec2.4 uses to REFUTE the pure-exposure model itself
resolvable?  (The same arms are declared 'indistinguishable within delta'.)"""
import json, os, statistics, random, itertools, math

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "indep_perboot.json")))
SMn = lambda a: int(a[1:])
U = ["d44", "d54", "d64", "d74"]          # exposure order w: d44<d54<d64<d74
T = {3: 3.182}

for phase in ("LO", "HI"):
    grid = D[phase]; blocks = sorted(grid["d44"])
    print("=" * 70, phase)
    # paired within-block contrasts among U
    for a, b in itertools.combinations(U, 2):
        diffs = [grid[b][k]["M_itl"] - grid[a][k]["M_itl"] for k in blocks]
        m, s = statistics.fmean(diffs), statistics.stdev(diffs)
        half = T[3] * s / math.sqrt(4)
        star = "  *" if abs(m) > half else ""
        print(f"  M_itl({b}) - M_itl({a}) = {m:+.4f} +- {s:.4f}  t(3) CI "
              f"[{m-half:+.4f}, {m+half:+.4f}]{star}")
    # bootstrap: how often is the observed ordering reproduced?
    random.seed(99); N = 20000
    obs_rank = tuple(sorted(U, key=lambda a: statistics.fmean(grid[a][k]["M_itl"] for k in blocks)))
    same = mono = 0
    perms = {}
    for _ in range(N):
        bs = [random.choice(blocks) for _ in blocks]
        mi = {a: statistics.fmean(grid[a][k]["M_itl"] for k in bs) for a in U}
        r = tuple(sorted(U, key=lambda a: mi[a]))
        perms[r] = perms.get(r, 0) + 1
        if r == obs_rank:
            same += 1
        if all(mi[U[i]] <= mi[U[i+1]] for i in range(3)):   # monotone in exposure w
            mono += 1
    top = sorted(perms.items(), key=lambda kv: -kv[1])[:5]
    print(f"  observed U ordering {obs_rank}: reproduced in {same/N:.3f} of block resamples")
    print(f"  P(M_itl monotone increasing in exposure w over U) = {mono/N:.4f}"
          f"   (24 orderings; 1/24 = 0.0417 if pure noise)")
    print("  top orderings:", [(''.join(x[1:] for x in k), round(v/N, 3)) for k, v in top])
