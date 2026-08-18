#!/usr/bin/env python3
"""A7 -- can the G17 S1 probe (d64 x sticky{ON,OFF} x 2 boots) actually enforce
P2 ("M_ttft^ON within +20% of OFF"), and does P2 protect the sec1 reduction
("D_ttft is already identified, so T1 reduces to D_itl")?

Two separate questions:
  (a) FLIP MARGIN: how much ARM-DIFFERENTIAL change in M_ttft flips D_ttft?
  (b) GATE POWER: with n=2 boots per leg and the measured boot-to-boot spread,
      what does a "+20% or less" verdict actually exclude?
"""
import json, os, math, statistics

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "..", "audit_g16_results_2026-08-17", "indep_perboot.json")))
BLOCKS = ["blk1", "blk2", "blk3", "blk4"]
ARMS = ["d16", "d24", "d34", "d44", "d54", "d64", "d74"]
U = ["d44", "d54", "d64", "d74"]

for phase in ("HI", "LO"):
    mu = {a: statistics.fmean(D[phase][a][b]["M_ttft"] for b in BLOCKS) for a in ARMS}
    sd = {a: statistics.stdev([D[phase][a][b]["M_ttft"] for b in BLOCKS]) for a in ARMS}
    win = min(mu, key=mu.get)
    print(f"\n=== {phase}: D_ttft = {win} ({mu[win]:.1f} ms)")
    print(f"{'arm':>5} {'M_ttft':>9} {'sd(4 blk)':>10} {'cv%':>6} "
          f"{'margin over argmin':>19}")
    for a in ARMS:
        m = (mu[a] - mu[win]) / mu[win] * 100
        print(f"{a:>5} {mu[a]:9.1f} {sd[a]:10.1f} {100*sd[a]/mu[a]:6.2f} {m:18.2f}%")
    # (a) flip margin within U (the G17 S2 grid) and within the full grid
    for arms, tag in ((ARMS, "full 7-arm"), (U, "G17 S2 grid U")):
        w = min(arms, key=lambda a: mu[a])
        others = sorted((mu[a] for a in arms if a != w))
        print(f"  [{tag}] argmin={w}; DIFFERENTIAL drift that flips it = "
              f"{100*(others[0]-mu[w])/mu[w]:.2f}%  "
              f"(P2 envelope is +20% -> {20/ (100*(others[0]-mu[w])/mu[w]):.1f}x the flip margin)")
    # (b) power of a 2-boot-per-leg level check on d64
    s = sd["d64"]
    se2 = s * math.sqrt(1 / 2 + 1 / 2)      # difference of two 2-boot means
    t = 2.776                               # t(4) 95%
    print(f"  [P2 power] d64 boot-to-boot sd = {s:.1f} ms ({100*s/mu['d64']:.2f}%). "
          f"2 boots/leg -> 95% CI half-width on the ON-OFF difference = "
          f"{t*se2:.1f} ms = {100*t*se2/mu['d64']:.2f}% of M_ttft.")
    print(f"             a measured 'within +20%' verdict is compatible with a true "
          f"shift of up to {20 + 100*t*se2/mu['d64']:.1f}% (and with a true shift as "
          f"small as {max(0.0, 20 - 100*t*se2/mu['d64']):.1f}%).")
