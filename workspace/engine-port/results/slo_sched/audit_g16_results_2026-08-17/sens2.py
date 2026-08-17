"""Second adversarial pass: (a) throughput vs TTFT donor, (b) reachability-edge
band = Delta identity, (c) upper-arm ordering robustness, (d) old-4-arm-grid
reproduction of the headline Delta_SLO."""
import json, math, os, glob, re, statistics, random

SLO = "/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/slo_sched"
HERE = os.path.dirname(os.path.abspath(__file__))
SMn = lambda a: int(a[1:])

# ---- per-boot throughput / duration straight from the artifacts -------------
thr = {"LO": {}, "HI": {}}
for phase in ("LO", "HI"):
    for path in sorted(glob.glob(os.path.join(SLO, f"g16_*_{phase}.jsonl"))):
        name = os.path.basename(path)
        m = re.match(r"^g16_(?P<block>[^_]+)_(?P<arm>d\d+)_boot", name)
        if not m or m.group("block").startswith("smoke"):
            continue
        dur = comp = 0.0
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            dur += float(r["duration"]); comp += float(r["completed"])
        thr[phase].setdefault(m.group("arm"), {})[m.group("block")] = comp / dur

print("== per-arm achieved throughput (req/s), n=4 blocks ==")
for phase in ("LO", "HI"):
    print(" ", phase)
    arms = sorted(thr[phase], key=SMn)
    for a in arms:
        v = [thr[phase][a][b] for b in sorted(thr[phase][a])]
        print(f"   {a}  mean {statistics.fmean(v):.4f}  sd {statistics.stdev(v):.4f}  {['%.4f'%x for x in v]}")
    best = max(arms, key=lambda a: statistics.fmean(thr[phase][a].values()))
    print("   argmax throughput =", best)
    # paired block bootstrap: P(d64 > d44)
    blocks = sorted(thr[phase]["d44"])
    diffs = [thr[phase]["d64"][b] - thr[phase]["d44"][b] for b in blocks]
    md = statistics.fmean(diffs); sd = statistics.stdev(diffs)
    print(f"   paired d64-d44 = {md:+.4f} +- {sd:.4f} req/s  "
          f"t(3) CI [{md-3.182*sd/2:+.4f}, {md+3.182*sd/2:+.4f}]  "
          f"rel {100*md/statistics.fmean(thr[phase]['d44'].values()):+.2f}%")

# ---- reachability edge == Delta identity + upper-arm order robustness -------
D = json.load(open(os.path.join(HERE, "indep_perboot.json")))
random.seed(4242)
for phase in ("LO", "HI"):
    grid = D[phase]; blocks = sorted(grid["d44"]); N = 20000
    edge_delta = {}
    d74_worst = 0; d74_gt_d64 = 0; itl_min_arm = {}
    argmin_all = {}
    for _ in range(N):
        bs = [random.choice(blocks) for _ in blocks]
        mi = {a: statistics.fmean(grid[a][b]["M_itl"] for b in bs) for a in grid}
        mt = {a: statistics.fmean(grid[a][b]["M_ttft"] for b in bs) for a in grid}
        dt = SMn(min(mt, key=lambda a: (mt[a], SMn(a))))
        di_arm = min(mi, key=lambda a: (mi[a], SMn(a)))
        itl_min_arm[di_arm] = itl_min_arm.get(di_arm, 0) + 1
        e = SMn(di_arm) - dt                      # Delta_SLO in the FIRST band
        edge_delta[e] = edge_delta.get(e, 0) + 1
        U = [a for a in mi if SMn(a) >= 44]
        if max(U, key=lambda a: mi[a]) == "d74":
            d74_worst += 1
        if mi["d74"] > mi["d64"]:
            d74_gt_d64 += 1
    print(f"== {phase}: Delta_SLO in the FIRST (reachability-edge) band == Delta, "
          f"distribution: { {k: round(v/N,4) for k,v in sorted(edge_delta.items())} }")
    print(f"   P(first-band Delta_SLO > 0) = {sum(v for k,v in edge_delta.items() if k>0)/N:.4f}")
    print(f"   argmin M_itl distribution   = { {k: round(v/N,4) for k,v in sorted(itl_min_arm.items(), key=lambda kv: SMn(kv[0]))} }")
    print(f"   P(d74 = worst of U) = {d74_worst/N:.4f}   P(M_itl(d74) > M_itl(d64)) = {d74_gt_d64/N:.4f}")

# ---- what the OLD 4-arm grid alone would have said --------------------------
print("== old 4-arm subgrid (d16..d44) of THIS campaign ==")
for phase in ("LO", "HI"):
    grid = D[phase]; blocks = sorted(grid["d44"])
    sub = {a: statistics.fmean(grid[a][b]["M_itl"] for b in blocks) for a in ("d16","d24","d34","d44")}
    subt = {a: statistics.fmean(grid[a][b]["M_ttft"] for b in blocks) for a in ("d16","d24","d34","d44")}
    dt = SMn(min(subt, key=lambda a: (subt[a], SMn(a))))
    s60 = min((SMn(a) for a in sub if sub[a] <= 60), default=None)
    print(f"  {phase}: D_ttft={dt}  S_itl(60)={s60}  Delta_SLO(60)={(s60 or 0)-dt}"
          f"   (full 7-arm grid gave the same? see above)")
