"""Adversarial sensitivity of the G16 headline decision quantities.

Reads only indep_perboot.json (produced here) -- no g16_analyze import, so
this is an INDEPENDENT re-derivation, not a re-run of the same code path.
"""
import json, math, os, random, statistics, itertools

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "indep_perboot.json")))
SM = lambda a: int(a[1:])

def arm_means(grid, key, blocks):
    return {a: statistics.fmean(grid[a][b][key] for b in blocks) for a in grid}

def s_itl(means, slo):
    ok = [a for a in means if means[a] <= slo]
    return min((SM(a) for a in ok), default=None)

def d_ttft(means):
    return SM(min(means, key=lambda a: (means[a], SM(a))))

for phase in ("LO", "HI"):
    grid = D[phase]
    blocks = sorted(grid["d44"])
    print("=" * 78, phase)
    mi = arm_means(grid, "M_itl", blocks)
    mt = arm_means(grid, "M_ttft", blocks)
    print("  full-grid  D_ttft =", d_ttft(mt), " S_itl(60) =", s_itl(mi, 60),
          " Delta_SLO(60) =", (s_itl(mi, 60) or 0) - d_ttft(mt))
    # per-block argmin TTFT / ITL
    for b in blocks:
        bm_t = {a: grid[a][b]["M_ttft"] for a in grid}
        bm_i = {a: grid[a][b]["M_itl"] for a in grid}
        print(f"   {b}: argmin TTFT={min(bm_t,key=lambda a:(bm_t[a],SM(a)))}"
              f"  argmin ITL={min(bm_i,key=lambda a:(bm_i[a],SM(a)))}"
              f"  S_itl(60)={s_itl(bm_i,60)}  Delta_SLO(60)="
              f"{(s_itl(bm_i,60) or 0)-SM(min(bm_t,key=lambda a:(bm_t[a],SM(a))))}")
    # exact S_itl transitions on full grid
    order = sorted(mi.items(), key=lambda kv: kv[1])
    print("   exact S_itl breakpoints (ms -> S_itl, and Delta_SLO):")
    cur = None
    prev = None
    for a, v in order:
        new = min([SM(x) for x in mi if mi[x] <= v] or [10**9])
        if new != cur:
            print(f"      SLO >= {v:8.4f} ms  ->  S_itl = {new:3d}"
                  f"  Delta_SLO = {new - d_ttft(mt):+4d}"
                  + (f"   [band width {v-prev:.4f} ms]" if prev is not None else ""))
            cur = new
            prev = v
    # ---- block bootstrap (resample the 4 blocks with replacement) ----
    random.seed(20260817)
    N = 20000
    cnt_ttft, cnt_s60, cnt_dslo, sign = {}, {}, {}, {"neg": 0, "zero": 0, "pos": 0}
    gap_up = []
    for _ in range(N):
        bs = [random.choice(blocks) for _ in blocks]
        m_i = {a: statistics.fmean(grid[a][b]["M_itl"] for b in bs) for a in grid}
        m_t = {a: statistics.fmean(grid[a][b]["M_ttft"] for b in bs) for a in grid}
        dt = d_ttft(m_t)
        s6 = s_itl(m_i, 60)
        cnt_ttft[dt] = cnt_ttft.get(dt, 0) + 1
        cnt_s60[s6] = cnt_s60.get(s6, 0) + 1
        if s6 is not None:
            d = s6 - dt
            cnt_dslo[d] = cnt_dslo.get(d, 0) + 1
            sign["neg" if d < 0 else "pos" if d > 0 else "zero"] += 1
        U = [a for a in m_i if SM(a) >= 44]
        gap_up.append(max(m_i[a] for a in U) - min(m_i[a] for a in U))
    print("   block-bootstrap (n=%d, resample 4 blocks w/ replacement):" % N)
    print("     D_ttft distribution :", {k: round(v / N, 4) for k, v in sorted(cnt_ttft.items())})
    print("     S_itl(60) distribution:", {k: round(v / N, 4) for k, v in sorted(cnt_s60.items(), key=lambda kv: (kv[0] is None, kv[0]))})
    print("     Delta_SLO(60) distr :", {k: round(v / N, 4) for k, v in sorted(cnt_dslo.items())})
    print("     sign of Delta_SLO(60):", {k: round(v / N, 4) for k, v in sign.items()})
    gap_up.sort()
    print("     gap_upper: point %.4f  pctile CI [%.4f, %.4f]  P(gap>1.0 ms)=%.4f"
          % (max(mi[a] for a in mi if SM(a) >= 44) - min(mi[a] for a in mi if SM(a) >= 44),
             gap_up[int(.025 * N)], gap_up[int(.975 * N)],
             sum(g > 1.0 for g in gap_up) / N))
