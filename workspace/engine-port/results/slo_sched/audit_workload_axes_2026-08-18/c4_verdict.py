#!/usr/bin/env python3
"""C-4 VERDICT computation.  Run from results/s8_scaleup/.

C-4 claims: "the DIRECTION (L/out up => co-residency up) is robust -- G16
1.8-18.3% vs s8 ctx4096 46-93% is an order of magnitude, no noise model flips it."

Two independent objections, both computed here:
 (1) The cross-campaign gap is compared ARM-BAND to ARM-BAND (G16 d16..d74 vs
     s8 d16..d92).  WORKLOAD_AXES W-4 itself rules that the arm band is a POLICY
     property, not a workload property.  Arm-matched (d44) it is 5.27% vs 65-93%.
 (2) The only L-ISOLATING contrast the repo owns is s8 ctx1024 vs ctx4096.  With
     the contaminated ctx1024 job (865533, keepalive_done=0 / errors=5943)
     excluded, the L effect is ~1.0x and its SIGN is not stable across arms.
     Sign test over the clean (arm,cell) pairs is reported.
"""
import json, glob, re, math, statistics as st, collections

def frac(p):
    n = c = 0
    for line in open(p, errors="replace"):
        try: e = json.loads(line)
        except Exception: continue
        if e.get("event") != "runtime_snapshot" or e.get("phase") != "benchmark": continue
        if e.get("decode_running_batch_size", 0) <= 0: continue
        n += 1
        if e.get("prefill_active_batch_size", 0) > 0: c += 1
    return c/n if n else None

v = collections.defaultdict(dict)
for p in sorted(glob.glob("s8_deconf_*_C*_*_telemetry.jsonl")):
    m = re.match(r"s8_deconf_(\w+?)_C(\d+)_(\w+?)_(\d+)(?:_blk\d)?_telemetry", p)
    if not m: continue
    f = frac(p)
    if f is not None: v[(m.group(1), m.group(3))][(int(m.group(2)), m.group(4))] = f

pos = neg = 0; ratios = []
print(f"{'arm':>5} {'cell':>5} {'ctx1024 CLEAN(865493)':>22} {'ctx4096(865311)':>17} {'ratio':>7} {'sign':>5}")
for (arm, cell), d in sorted(v.items()):
    a = d.get((1024, "865493")); b = d.get((4096, "865311"))
    if a is None or b is None: continue
    r = b/a; ratios.append(r)
    s = "+" if b > a else "-"
    pos += (b > a); neg += (b <= a)
    print(f"{arm:>5} {cell:>5} {a*100:21.1f}% {b*100:16.1f}% {r:7.2f} {s:>5}")
n = pos+neg
# exact two-sided binomial sign test
p2 = 2*sum(math.comb(n, k) for k in range(max(pos, neg), n+1))/2**n
print(f"\n  clean-pair ratio: median={st.median(ratios):.2f}x  range=[{min(ratios):.2f}, {max(ratios):.2f}]")
print(f"  sign test: {pos} up / {neg} down of {n}  -> exact two-sided p = {min(1.0,p2):.3f}")
print(f"  (and n_indep = 1 JOB per leg: 865493 gpu40 2026-07-27 22:45 vs 865311 gpu40 2026-07-27 21:02,")
print(f"   KEEPA_N = 8 vs 2 -- the sbatch itself calls KEEPA the prefill DUTY-CYCLE knob)")
print(f"\n  contaminated pooled ctx1024 (865493+865533) reproduces the note's 1.26-1.33x:")
for (arm, cell), d in sorted(v.items()):
    if cell != "d44": continue
    lo = [x for (c, j), x in d.items() if c == 1024]
    hi = d.get((4096, "865311"))
    if len(lo) == 2 and hi:
        print(f"    {arm} d44: pooled {st.fmean(lo)*100:.1f}% -> {hi*100:.1f}% = {hi/st.fmean(lo):.2f}x  "
              f"| clean-only {d[(1024,'865493')]*100:.1f}% -> {hi*100:.1f}% = {hi/d[(1024,'865493')]:.2f}x")
