#!/usr/bin/env python3
"""A6 -- WHAT DOES M_itl ACTUALLY MEASURE?  (decides whether the G17 lever can
move it at all)

M_itl = median_requests( p95(token_itl) ).  G16 sec2.4 and the G17 design both
treat it as a LINEAR two-state mixture  M = w*S + (1-w)*U  and derive the
"dilution factor 1/w_bar = 4.3x (HI)" that is G17's entire power argument.

A p95 is not a mean.  This script measures, from the already-collected G16
bench artifacts (GPU cost 0):
  - per-request p50 / mean / p95 of token ITL  (same artifacts, same loader
    convention: linear-interpolation percentile, per gate #4/#7)
  - the fraction of a request's tokens that exceed a stall threshold
  - where the p95 sits relative to the bimodal ITL distribution

Read-only.  This is a DIAGNOSTIC about the estimand's structure; it is not a
new performance claim and it does not re-score any policy comparison.
"""
import json, os, glob, statistics
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
SL = os.path.join(HERE, "..")
ARMS = ["d16", "d24", "d34", "d44", "d54", "d64", "d74"]
BLK = ["blk1", "blk2", "blk3", "blk4"]


def percentile(xs, q):                      # copy of pdmux_eval.analyze.percentile
    if not xs:
        return float("nan")
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * q
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def load(arm, blk, phase):
    pat = os.path.join(SL, f"g16_{blk}_{arm}_boot1_*_{phase}.jsonl")
    fs = [f for f in glob.glob(pat) if "smoke" not in f]
    reqs = []
    for f in fs:
        with open(f) as fh:
            for line in fh:
                e = json.loads(line)
                for itl in e["itls"]:
                    if itl:
                        reqs.append([1000.0 * x for x in itl])
    return reqs


print(f"{'arm':>5} {'phase':>5} | {'M_itl(p95)':>10} {'M_p50':>8} {'M_mean':>8} "
      f"| {'tok frac >45ms':>14} {'stall ms(p50 of>45)':>20} {'fast ms(p50 of<45)':>19}")
res = {}
for phase in ("HI", "LO"):
    for arm in ARMS:
        p95s, p50s, mns, fr, stall, fast = [], [], [], [], [], []
        for blk in BLK:
            for r in load(arm, blk, phase):
                p95s.append(percentile(r, 0.95))
                p50s.append(percentile(r, 0.50))
                mns.append(statistics.fmean(r))
                hi = [x for x in r if x > 45.0]
                lo = [x for x in r if x <= 45.0]
                fr.append(len(hi) / len(r))
                if hi:
                    stall.append(statistics.median(hi))
                if lo:
                    fast.append(statistics.median(lo))
        res[(arm, phase)] = (statistics.median(p95s), statistics.median(p50s),
                             statistics.median(mns), statistics.fmean(fr),
                             statistics.median(stall) if stall else float("nan"),
                             statistics.median(fast) if fast else float("nan"))
        v = res[(arm, phase)]
        print(f"{arm:>5} {phase:>5} | {v[0]:10.3f} {v[1]:8.3f} {v[2]:8.3f} "
              f"| {v[3]:14.4f} {v[4]:20.3f} {v[5]:19.3f}")
    print()

print("SPREAD ACROSS U = {d44,d54,d64,d74} (the K11 upper set):")
for phase in ("HI", "LO"):
    for i, name in ((0, "M_itl (p95)  [prereg estimand]"), (1, "per-req p50"),
                    (2, "per-req mean"), (4, "stall height"), (5, "fast-token level")):
        vals = [res[(a, phase)][i] for a in ("d44", "d54", "d64", "d74")]
        print(f"  {phase} {name:32s}: " + " ".join(f"{v:7.3f}" for v in vals)
              + f"   max-min = {max(vals)-min(vals):6.3f} ms"
              + f"   ({100*(max(vals)-min(vals))/min(vals):5.2f}%)")
    print()
