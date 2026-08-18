#!/usr/bin/env python3
"""A2 -- G17 sec2 claims "adding blocks needs TENS of blocks (n ~ (sd/gap)^2)".
Check that arithmetic against the real 4x7 block table.

Model (fitted from the data, no free knobs): X[a,b] = mu_a + beta_b + eps,
two-way additive, sigma = residual SD.  Truth = the observed arm means
(i.e. the OPTIMISTIC case: assume the observed ordering is real).
Then simulate N blocks and apply K1 exactly: rank rule ceil(0.75N)/N blocks
won AND block-bootstrap (10000 draws, prereg K3) frequency >= 0.80.

Also reports the paired-difference SD, which is the scale the sec2 formula
should have used (blocks are paired across arms; the marginal per-arm sd the
design quotes contains the block main effect, which cancels in K1).
"""
import json, os, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "..", "audit_g16_results_2026-08-17", "indep_perboot.json")))
BLOCKS = ["blk1", "blk2", "blk3", "blk4"]
FULL = ["d16", "d24", "d34", "d44", "d54", "d64", "d74"]
U = ["d44", "d54", "d64", "d74"]
rng = np.random.default_rng(1)


def table(phase, arms, key="M_itl"):
    return np.array([[D[phase][a][b][key] for b in BLOCKS] for a in arms])  # arms x blocks


def fit(X):
    mu = X.mean(1)                       # arm means
    beta = X.mean(0) - X.mean()          # block effects
    resid = X - mu[:, None] - beta[None, :]
    dof = (X.shape[0] - 1) * (X.shape[1] - 1)
    sigma = math.sqrt((resid ** 2).sum() / dof)
    sd_beta = beta.std(ddof=1)
    return mu, sigma, sd_beta


def k1_identified(X, nboot=10000, seed=0):
    """X: arms x blocks.  Returns True if K1 (rank AND bootstrap) passes."""
    A, N = X.shape
    per_block = X.argmin(0)
    need = math.ceil(0.75 * N)
    wins = np.bincount(per_block, minlength=A)
    if wins.max() < need:
        return False
    r = np.random.default_rng(seed)
    idx = r.integers(0, N, size=(nboot, N))
    means = X[:, idx].mean(2)            # arms x nboot
    win = means.argmin(0)
    frac = np.bincount(win, minlength=A).max() / nboot
    return frac >= 0.80


def power(phase, arms, Ns, nsim=400):
    X = table(phase, arms)
    mu, sigma, sd_beta = fit(X)
    print(f"\n--- {phase} arms={arms}")
    print(f"    fitted sigma(residual, block-paired) = {sigma:.3f} ms ; "
          f"sd(block effect) = {sd_beta:.3f} ms")
    print(f"    marginal per-arm sd across blocks = "
          + " ".join(f"{a}:{X[i].std(ddof=1):.3f}" for i, a in enumerate(arms)))
    gaps = np.sort(mu)
    print(f"    true-mean gap (best vs 2nd best) = {gaps[1]-gaps[0]:.3f} ms ; "
          f"paired-diff sd = {sigma*math.sqrt(2):.3f} ms")
    print(f"    naive n from sec2 formula (sd/gap)^2 with paired sd: "
          f"{(sigma*math.sqrt(2)/(gaps[1]-gaps[0]))**2:.1f}")
    for N in Ns:
        hits = 0
        for s in range(nsim):
            beta = rng.normal(0, sd_beta, size=N)
            eps = rng.normal(0, sigma, size=(len(arms), N))
            Xs = mu[:, None] + beta[None, :] + eps
            hits += k1_identified(Xs, nboot=2000, seed=s)
        print(f"    N={N:3d} blocks -> P(K1 identifies D_itl) = {hits/nsim:.3f} "
              f"({nsim} sims)   [~{N*1.0:.0f} GPU-hr]")


power("HI", FULL, [4, 8, 12, 16, 24, 32, 48])
power("HI", U, [4, 8, 12, 16, 24, 32, 48])
