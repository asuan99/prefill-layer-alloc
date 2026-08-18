#!/usr/bin/env python3
"""A3b -- (1) K1's false-identification rate under EXACT ties at N=4, and
(2) the SCALE-INVARIANCE identity created by design sec4's own fix:
"delta_ON must be re-derived from the ON leg's own within-boot spread".

If the sticky lever scales the arm contrast AND the boot-to-boot spread by the
same factor (both are the same physical quantity seen through a longer
exposure), then gap/delta and P(K1) are INVARIANT -- the ON leg is guaranteed
to reproduce the OFF verdict regardless of the truth.  The design's power
argument silently assumes noise stays put.
"""
import json, os, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "..", "audit_g16_results_2026-08-17", "indep_perboot.json")))
BLOCKS = ["blk1", "blk2", "blk3", "blk4"]
U = ["d44", "d54", "d64", "d74"]
X = np.array([[D["HI"][a][b]["M_itl"] for b in BLOCKS] for a in U])
mu = X.mean(1); beta = X.mean(0) - X.mean()
sigma = math.sqrt(((X - mu[:, None] - beta[None, :]) ** 2).sum() / 9)
sdb = beta.std(ddof=1)
rng = np.random.default_rng(11)


def k1(Xs, nboot=2000, seed=0):
    A, N = Xs.shape
    if np.bincount(Xs.argmin(0), minlength=A).max() < math.ceil(0.75 * N):
        return False
    r = np.random.default_rng(seed)
    idx = r.integers(0, N, size=(nboot, N))
    m = Xs[:, idx].mean(2)
    return np.bincount(m.argmin(0), minlength=A).max() / nboot >= 0.80


def run(k, r_, N=4, nsim=1500):
    h = 0
    for s in range(nsim):
        b = rng.normal(0, sdb * r_, size=N)
        e = rng.normal(0, sigma * r_, size=(4, N))
        h += k1((mu.mean() + k * (mu - mu.mean()))[:, None] + b[None, :] + e, seed=s)
    return h / nsim


print(f"sigma={sigma:.3f} sd_block={sdb:.3f}")
print(f"FALSE identification under EXACT ties (k=0): N=4 {run(0,1.0):.3f}  N=6 {run(0,1.0,6):.3f}")
print("signal and noise scaled TOGETHER (scale-free -> no power gain):")
for k in (1, 2, 4.3, 8):
    print(f"  k={k:<4} sigma x{k:<4} -> P(K1)={run(k,k):.3f}")
print("signal only (the design's unstated assumption):")
for k in (1, 2, 3.0, 4.3, 6):
    print(f"  k={k:<4} sigma x1   -> P(K1)={run(k,1.0):.3f}")
