#!/usr/bin/env python3
"""A8 -- how big is the thing G17-T1 is trying to sign?

The G16 payoff band is [58.533, 58.625) ms, width 0.092 ms.  Its SIGN is
decided by ONE pairwise ordering: M_itl(d64) < M_itl(d34).  (Below d34's mean
only d64 reaches the SLO -> S_itl = 64 > D_ttft = 44 -> TAX_POSITIVE; above it
d34 reaches -> S_itl = 34 < 44 -> NEGATIVE.)

So T1 = "is the d34-vs-d64 ordering real?".  This script prices that.
"""
import json, os, math, statistics

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "..", "audit_g16_results_2026-08-17", "indep_perboot.json")))
BLOCKS = ["blk1", "blk2", "blk3", "blk4"]

for phase in ("HI",):
    for a, b in (("d34", "d64"), ("d44", "d64"), ("d34", "d44"), ("d64", "d74")):
        da = [D[phase][a][k]["M_itl"] - D[phase][b][k]["M_itl"] for k in BLOCKS]
        m, s = statistics.fmean(da), statistics.stdev(da)
        se = s / math.sqrt(len(da))
        t = 3.182                                  # t(3) 95%
        n80 = (2.8 * s / abs(m)) ** 2 if m else float("inf")
        print(f"{phase} {a}-{b}: mean={m:+.4f} ms  sd(paired,block)={s:.4f}  "
              f"t(3) 95% CI=[{m-t*se:+.3f}, {m+t*se:+.3f}]  "
              f"blocks for 80% power = {n80:.0f}")
    print()
    # what the sticky lever could buy, at its STRUCTURAL ceiling (a4: 1/w_da(HI) = 3.0x)
    a, b = "d34", "d64"
    da = [D[phase][a][k]["M_itl"] - D[phase][b][k]["M_itl"] for k in BLOCKS]
    m, s = statistics.fmean(da), statistics.stdev(da)
    for amp, tag in ((1.0, "OFF (as measured)"), (3.0, "ceiling of sticky ON in HI (1/w_da)"),
                     (4.3, "the design's claimed 4.3x (uses the ALL denominator)")):
        print(f"  contrast x{amp:<4} [{tag}]: blocks for 80% power (noise unchanged) = "
              f"{(2.8*s/(amp*abs(m)))**2:.0f}"
              f"   |  if noise scales too: {(2.8*s/abs(m))**2:.0f} (unchanged -- scale-free)")
    print("\n  NOTE: d34 is NOT in the G17 S2 grid U={d44,d54,d64,d74}; the ON leg "
          "cannot evaluate this ordering at all.")
