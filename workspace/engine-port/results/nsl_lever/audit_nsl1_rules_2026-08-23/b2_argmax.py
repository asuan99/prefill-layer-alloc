#!/usr/bin/env python3
"""B2 -- does NSL-1 sec4.1's decision rule manufacture its own verdict?

Grid: D in {d16,d44,d64} x cap in {24,48,96} = 9 cells, n boots each.
Rule (as written):
  (i)  argmax over the JOINT grid has cap != 48  -> ADMISSION_AXIS_MATTERS
  (ii) argmax over the JOINT grid has cap == 48  -> ADMISSION_AXIS_INERT
  (iii) "paired CI contains 0"                   -> UNDETERMINED  (undefined operand)

Also: Delta := max_joint - max_{cap=48 slice}.  Set inclusion => Delta >= 0 ALWAYS.
"""
import numpy as np

rng = np.random.default_rng(20260823)
CAPS = [24, 48, 96]
DS = ["d16", "d44", "d64"]
REPS = 200000


def sim(true_mu, n, sd, reps=REPS):
    """true_mu[d][cap] -> (P(cap!=48 at joint argmax), mean Delta, P(Delta>0))"""
    mu = np.array([[true_mu[d][c] for c in CAPS] for d in DS])   # 3x3
    # cell means over n boots
    obs = mu[None, :, :] + rng.normal(0, sd / np.sqrt(n), size=(reps, 3, 3))
    flat = obs.reshape(reps, 9)
    amax = flat.argmax(axis=1)
    cap_of = np.array([c for _ in DS for c in CAPS])
    p_not48 = float(np.mean(cap_of[amax] != 48))
    slice48 = obs[:, :, CAPS.index(48)]
    delta = flat.max(axis=1) - slice48.max(axis=1)
    return p_not48, float(delta.mean()), float(np.mean(delta > 0)), float(np.mean(delta < 0))


print("== H0: cap is COMPLETELY INERT (all 9 cells identical true goodput 70pp) ==")
h0 = {d: {c: 70.0 for c in CAPS} for d in DS}
for sd in (0.5, 0.88, 2.0, 4.8):
    for n in (4,):
        p, dmean, dpos, dneg = sim(h0, n, sd)
        print(f"  sd={sd:4.2f}pp n={n}: P(rule declares ADMISSION_AXIS_MATTERS)={p:.3f}"
              f"   E[Delta]={dmean:+.3f}pp  P(Delta>0)={dpos:.3f}  P(Delta<0)={dneg:.3f}")

print("\n== H0': cap inert but D matters (d44 best by 10pp), still all caps equal ==")
h0b = {"d16": {c: 60.0 for c in CAPS}, "d44": {c: 70.0 for c in CAPS},
       "d64": {c: 65.0 for c in CAPS}}
for sd in (0.5, 0.88, 2.0, 4.8):
    p, dmean, dpos, dneg = sim(h0b, 4, sd)
    print(f"  sd={sd:4.2f}pp n=4: P(MATTERS)={p:.3f}  E[Delta]={dmean:+.3f}pp  P(Delta>0)={dpos:.3f}")

print("\n== H1: cap DOES matter (+2.1pp = the registered 3% rel effect at cap=96, d44) ==")
h1 = {"d16": {24: 60.0, 48: 60.0, 96: 62.1}, "d44": {24: 70.0, 48: 70.0, 96: 72.1},
      "d64": {24: 65.0, 48: 65.0, 96: 67.1}}
for sd in (0.5, 0.88, 2.0, 4.8):
    p, dmean, dpos, dneg = sim(h1, 4, sd)
    print(f"  sd={sd:4.2f}pp n=4: P(MATTERS)={p:.3f} -> P(false INERT)={1-p:.3f}  E[Delta]={dmean:+.3f}pp")
