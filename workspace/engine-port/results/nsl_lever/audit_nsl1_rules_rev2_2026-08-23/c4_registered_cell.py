#!/usr/bin/env python3
"""C4 -- measure, from logs the repo ALREADY has, the three constants that
DESIGN rev2 sec4.4/sec7 import from elsewhere or leave blank:

  (1) mean in-system concurrency vs the cap=48 incumbent  (is cap binding?)
  (2) goodput LEVEL and BETWEEN-BOOT SD of the registered cell
      (Zamba2-2.7B, d44, change trace 3<->12, chat 300/50), under BOTH the
      harness's mean-ITL predicate and canon gate #4's p95-ITL predicate
  (3) the change-trace wall clock that sec7 says is "미확정"

Positive control: the same reader reproduces canon sec1-7's native metric
(g/dur, 3s/60ms) for this cell to three digits.

Read-only.  Run from results/slo_sched/.
"""
import json, glob, os, statistics as st

CELL = "d44"; TRACE = "L3H12"

def pctl(a, q):
    if not a: return 0.0
    a = sorted(a); return a[min(len(a) - 1, int(q * len(a)))]

def rounds(f):
    for ln in open(f):
        ln = ln.strip()
        if ln: yield json.loads(ln)

def score(f, ttft_slo, itl_slo, stat):
    g = n = 0
    for o in rounds(f):
        T = o.get("ttfts") or []; I = o.get("itls") or []
        for i, t in enumerate(T):
            il = I[i] if i < len(I) else []
            m = 9.0 if not il else (sum(il)/len(il) if stat == "mean" else pctl(il, .95))
            n += 1
            if t <= ttft_slo and m <= itl_slo: g += 1
    return g, n

def dur(f):  return sum((o.get("duration") or 0) for o in rounds(f))

def conc(f):
    """Little's law: mean in-system concurrency per round = sum(latency)/duration."""
    out = []
    for o in rounds(f):
        T = o.get("ttfts") or []; I = o.get("itls") or []; d = o.get("duration") or 0
        lat = [T[i] + sum(I[i] if i < len(I) else []) for i in range(len(T))]
        out.append(sum(lat) / d if d else 0)
    return out

pairs = []
for hi in sorted(glob.glob(f"sgptvHi_{CELL}_*_{TRACE}_*.jsonl")):
    lo = hi.replace("sgptvHi_", "sgptvLo_")
    if os.path.exists(lo): pairs.append((lo, hi))
assert pairs, "no d44 change-trace logs found -- run from results/slo_sched/"

print(f"n boots = {len(pairs)}  ({', '.join(os.path.basename(h).split('_')[-1][:-6] for _, h in pairs)})")

cl = [c for lo, _ in pairs for c in conc(lo)]
ch = [c for _, hi in pairs for c in conc(hi)]
print(f"\n(1) mean concurrency vs cap=48 (hardwired sharegpt_vary_bench.sbatch:53)")
print(f"    LO rate3 : {st.mean(cl):5.2f}  [{min(cl):.2f},{max(cl):.2f}]  n={len(cl)} rounds  -> cap grid inert")
print(f"    HI rate12: {st.mean(ch):5.2f}  [{min(ch):.2f},{max(ch):.2f}]  n={len(ch)} rounds  -> cap BINDS")
print(f"    Markov bound: P(conc>40 | HI) >= 1-(48-{st.mean(ch):.2f})/(48-40) = "
      f"{1-(48-st.mean(ch))/8:.2f}")

print(f"\n(2) level and between-boot SD of the registered cell")
for stat in ("mean", "p95"):
    tag = "harness predicate" if stat == "mean" else "canon gate #4"
    L = []; H = []; C = []
    for lo, hi in pairs:
        gl, nl = score(lo, .300, .050, stat); gh, nh = score(hi, .300, .050, stat)
        L.append(100*gl/nl); H.append(100*gh/nh); C.append(100*(gl+gh)/(nl+nh))
    print(f"    ITL={stat:<4} ({tag:<17}) chat 300/50 attainment%:  "
          f"LO {st.mean(L):6.2f}+-{st.stdev(L):.2f}   HI {st.mean(H):6.2f}+-{st.stdev(H):.2f}   "
          f"COMBINED {st.mean(C):6.2f}+-{st.stdev(C):.2f}")
gp = []
for lo, hi in pairs:
    gl, _ = score(lo, 3.0, .060, "mean"); gh, _ = score(hi, 3.0, .060, "mean")
    gp.append((gl+gh)/(dur(lo)+dur(hi)))
print(f"    POSITIVE CONTROL, canon sec1-7 native metric (g/dur, 3s/60ms mean-ITL): "
      f"{st.mean(gp):.3f} +- {st.stdev(gp):.3f} req/s   (canon: 3.220 +- 0.013)")

print(f"\n(3) change-trace wall clock (sec7 says '미확정'; sec4.4 costs BOOT ONLY at 145.8 s)")
tot = [dur(lo) + dur(hi) for lo, hi in pairs]
print(f"    bench seconds per cell (3 rounds) = {st.mean(tot):.1f}  [{min(tot):.1f},{max(tot):.1f}]")
print(f"    per-cell wall clock = {st.mean(tot):.1f} + boot;  sec4.4 assumes 145.8 total")
print(f"    understatement factor = {(st.mean(tot)+145.8)/145.8:.2f}x  (and 145.8 is the "
      f"gate #13 M8/Ha8 constant, which kernel_mech rev7:406 reads as boot+60 s window)")
