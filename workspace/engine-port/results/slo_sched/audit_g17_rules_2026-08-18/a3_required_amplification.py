#!/usr/bin/env python3
"""A3 -- how large must the CONDITIONAL contrast be for K1 to identify D_itl
on the G17 S2 sticky-ON leg (4 arms x 4 blocks)?

Same fitted model as a2 (mu from data, sigma = block-paired residual SD).
Scale the arm-mean deviations by k (the amplification the sticky lever is
supposed to buy) and, separately, scale the noise by r (unknown for the ON
leg -- reported as a sensitivity, not a knob).

K1 as preregistered = rank rule (>=3 of 4 blocks) AND bootstrap >= 0.80.
"""
import json, os, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "..", "audit_g16_results_2026-08-17", "indep_perboot.json")))
BLOCKS = ["blk1", "blk2", "blk3", "blk4"]
U = ["d44", "d54", "d64", "d74"]
rng = np.random.default_rng(7)


def fit(phase, arms):
    X = np.array([[D[phase][a][b]["M_itl"] for b in BLOCKS] for a in arms])
    mu = X.mean(1)
    beta = X.mean(0) - X.mean()
    resid = X - mu[:, None] - beta[None, :]
    sigma = math.sqrt((resid ** 2).sum() / ((X.shape[0] - 1) * (X.shape[1] - 1)))
    return mu, sigma, beta.std(ddof=1)


def k1(X, nboot=2000, seed=0):
    A, N = X.shape
    wins = np.bincount(X.argmin(0), minlength=A)
    if wins.max() < math.ceil(0.75 * N):
        return False
    r = np.random.default_rng(seed)
    idx = r.integers(0, N, size=(nboot, N))
    m = X[:, idx].mean(2)
    return np.bincount(m.argmin(0), minlength=A).max() / nboot >= 0.80


for phase in ("HI",):
    mu, sigma, sdb = fit(phase, U)
    c = mu.mean()
    print(f"{phase}: mu={np.round(mu,3)} sigma={sigma:.3f} sd_block={sdb:.3f}")
    print("  gap(best vs 2nd) at k=1 :", round(float(np.sort(mu)[1] - np.sort(mu)[0]), 3), "ms")
    print("  k = amplification of the arm contrast (sticky lever's promise)")
    for N in (4, 6, 8):
        line = []
        for k in (1, 2, 3, 4, 6, 8, 12):
            hits = 0
            nsim = 400
            for s in range(nsim):
                b = rng.normal(0, sdb, size=N)
                e = rng.normal(0, sigma, size=(len(U), N))
                X = (c + k * (mu - c))[:, None] + b[None, :] + e
                hits += k1(X, seed=s)
            line.append(f"k={k:2d}:{hits/nsim:.2f}")
        print(f"  N={N}: " + "  ".join(line))
    # noise sensitivity at N=4, k=4.3 (the design's claimed dilution factor)
    print("  noise sensitivity at N=4, k=4.3 (ON-leg sigma unknown):")
    for r_ in (0.5, 1.0, 1.5, 2.0, 3.0):
        hits = 0
        nsim = 400
        for s in range(nsim):
            b = rng.normal(0, sdb * r_, size=4)
            e = rng.normal(0, sigma * r_, size=(len(U), 4))
            X = (c + 4.3 * (mu - c))[:, None] + b[None, :] + e
            hits += k1(X, seed=s)
        print(f"    sigma_ON/sigma_OFF={r_:.1f} -> P(K1 identifies)={hits/nsim:.2f}")
