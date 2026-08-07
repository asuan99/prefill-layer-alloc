"""Gate-2 rev3 §11-2, part 3: is delta=0.05 even ABOVE the metric's own systematic
sensitivity at tau=40?

(G) global-perturbation test.  Multiply every recorded ITL by (1+e) for small e and
    recompute X_tau.  If |dX| from a 1-3% engine perturbation exceeds delta=0.05, then
    delta is smaller than the threshold indicator's own systematic drift and NO amount
    of n rescues it (project methodology gate #6 / #12: metric cliff).
(H) delta expressed on three scales, per tau: absolute / fraction of the A1->A4 gap /
    multiple of the A4 level / per-token stall probability.
(I) exact required n at tau=40 including Holm-adjusted alpha, nmax raised to 250.
(J) TOST size at the boundary |Delta| = delta under the empirical (non-normal) shape.

Aggregation units: verify_g2_tau40_lib.py header.  GPU 0.  No new performance verdict.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from verify_load import REPS, load_cells  # noqa: E402
from verify_g2_tau40_lib import (  # noqa: E402
    A1, A4, DECISION_CELLS, DELTA, DESIGNATED, arm_vector, cell_label,
    min_n_for_power, pair_difference_shape, standardised_shape, t_ppf,
    token_p_from_x, tost_power, x_tau_table,
)

OUT = Path(__file__).parent
report = {}
rng = np.random.default_rng(1)


def hr(t):
    print("\n" + "=" * 104)
    print(t)
    print("=" * 104)


def x_perturbed(cells, model, arm, rate, rep, tau, eps):
    c = cells[(model, arm, rate, rep)]
    mx = [max(q.token_itl_ms) for q in c["requests"] if q.token_itl_ms]
    if not mx:
        return float("nan")
    k = 1.0 + eps
    return sum(1 for v in mx if v * k > tau) / len(mx)


def tost_power_alpha(sigma, n, delta, gap, alpha, trials=100_000, rng=None, shape=None):
    rng = rng or np.random.default_rng(3)
    if shape is None:
        d = rng.normal(gap, sigma, size=(trials, n))
    else:
        idx = rng.integers(0, len(shape), size=(trials, n))
        d = gap + sigma * np.asarray(shape)[idx]
    mean = d.mean(axis=1)
    se = d.std(axis=1, ddof=1) / math.sqrt(n)
    tc = t_ppf(1 - alpha, n - 1)
    return float(((mean - tc * se > -delta) & (mean + tc * se < delta)).mean())


def min_n_alpha(sigma, delta, gap, alpha, target=0.80, nmax=250, trials=40_000, rng=None):
    rng = rng or np.random.default_rng(5)
    for n in range(3, nmax + 1):
        if tost_power_alpha(sigma, n, delta, gap, alpha, trials=trials, rng=rng) >= target:
            return n
    return f">{nmax}"


def main():
    cells = load_cells()
    taus = (40.0, 45.0, 50.0, 60.0)
    per_rep, p_token = x_tau_table(cells, taus=taus)

    # ------------------------------------------------------------------ (G)
    hr("(G) global ITL perturbation -- how far does X_tau move if the engine is e%% slower?")
    print("all recorded ITLs scaled by (1+e); X recomputed; entry = mean over 5 reps of X(A4).")
    print("delta = 0.05 is the equivalence margin the prereg wants to defend.\n")
    eps_list = (-0.03, -0.01, 0.0, 0.01, 0.03)
    partG = {}
    for tau in taus:
        print(f"--- tau = {tau:.0f} ms")
        print(f"{'cell':14s} " + " ".join(f"{'e=%+.0f%%' % (100 * e):>9s}" for e in eps_list)
              + f" | {'|dX| @1%':>9s} {'|dX| @3%':>9s} {'vs delta(1%)':>13s} {'vs delta(3%)':>13s}")
        for m, r in DECISION_CELLS:
            xs = []
            for e in eps_list:
                v = [x_perturbed(cells, m, A4, r, rep, tau, e) for rep in REPS]
                xs.append(float(np.mean(v)))
            base = xs[2]
            d1 = max(abs(xs[1] - base), abs(xs[3] - base))
            d3 = max(abs(xs[0] - base), abs(xs[4] - base))
            partG[f"{m}|r{r:g}|t{tau:g}"] = {"eps": list(eps_list), "x": xs,
                                             "abs_dX_1pct": d1, "abs_dX_3pct": d3,
                                             "ratio_1pct_to_delta": d1 / DELTA,
                                             "ratio_3pct_to_delta": d3 / DELTA}
            print(f"{cell_label(m, r):14s} " + " ".join(f"{x:9.4f}" for x in xs)
                  + f" | {d1:9.4f} {d3:9.4f} {d1 / DELTA:12.2f}x {d3 / DELTA:12.2f}x")
        print()
    report["G_perturbation"] = partG

    # same thing for A1, to show the asymmetry
    print("A1 (plain) for contrast, tau=40:")
    for m, r in DECISION_CELLS:
        xs = [float(np.mean([x_perturbed(cells, m, A1, r, rep, 40.0, e) for rep in REPS]))
              for e in eps_list]
        print(f"{cell_label(m, r):14s} " + " ".join(f"{x:9.4f}" for x in xs)
              + f" | |dX|@3% = {max(abs(xs[0] - xs[2]), abs(xs[4] - xs[2])):.4f}")

    # ------------------------------------------------------------------ (H)
    hr("(H) what delta = 0.05 buys on four scales, per tau")
    print(f"{'cell':14s} {'tau':>4s} {'X(A4)':>7s} {'gap':>7s} | {'d/gap':>7s} {'d/X(A4)':>8s} "
          f"{'p(A4) ppm':>10s} {'p(A4+d) ppm':>12s} {'rel. token':>11s}")
    partH = {}
    for m, r in DECISION_CELLS:
        for tau in taus:
            a1 = arm_vector(per_rep, m, A1, r, tau)
            a4 = arm_vector(per_rep, m, A4, r, tau)
            x4, gap = float(a4.mean()), float(a1.mean() - a4.mean())
            G = float(np.mean([per_rep[(m, A4, r, rep, tau)]["n_gaps"]
                               / per_rep[(m, A4, r, rep, tau)]["n_used"] for rep in REPS]))
            p0 = token_p_from_x(x4, G)
            p1 = token_p_from_x(x4 + DELTA, G)
            partH[f"{m}|r{r:g}|t{tau:g}"] = {
                "x_a4": x4, "gap": gap, "delta_over_gap": DELTA / gap if gap > 0 else float("nan"),
                "delta_over_x_a4": DELTA / x4 if x4 > 0 else float("inf"),
                "p_a4_ppm": 1e6 * p0, "p_a4_plus_delta_ppm": 1e6 * p1,
                "relative_token_inflation": p1 / p0 if p0 > 0 else float("inf")}
            print(f"{cell_label(m, r):14s} {tau:4.0f} {x4:7.4f} {gap:7.4f} | "
                  f"{DELTA / gap if gap > 0 else float('nan'):7.3f} "
                  f"{DELTA / x4 if x4 > 0 else float('inf'):8.2f} {1e6 * p0:10.1f} "
                  f"{1e6 * p1:12.1f} {p1 / p0 if p0 > 0 else float('inf'):10.2f}x")
        print()
    report["H_delta_scales"] = partH

    # ------------------------------------------------------------------ (I)
    hr("(I) required n at tau=40 -- unadjusted and Holm-adjusted, target power 0.80 at gap 0")
    print("Holm over 4 replication cells: worst step alpha = 0.05/4 = 0.0125 per side.")
    print(f"{'cell':14s} {'sigma':>7s} {'n80 a=.05':>10s} {'n80 a=.025':>11s} {'n80 a=.0125':>12s} "
          f"{'n90 a=.05':>10s}")
    partI = {}
    for m, r in DECISION_CELLS:
        a4 = arm_vector(per_rep, m, A4, r, 40.0)
        sig = math.sqrt(2) * float(a4.std(ddof=1))
        row = {"sigma": sig,
               "n80_a05": min_n_alpha(sig, DELTA, 0.0, 0.05, 0.80, rng=rng),
               "n80_a025": min_n_alpha(sig, DELTA, 0.0, 0.025, 0.80, rng=rng),
               "n80_a0125": min_n_alpha(sig, DELTA, 0.0, 0.0125, 0.80, rng=rng),
               "n90_a05": min_n_alpha(sig, DELTA, 0.0, 0.05, 0.90, rng=rng)}
        partI[f"{m}|r{r:g}"] = row
        print(f"{cell_label(m, r):14s} {sig:7.4f} {str(row['n80_a05']):>10s} "
              f"{str(row['n80_a025']):>11s} {str(row['n80_a0125']):>12s} {str(row['n90_a05']):>10s}")
    report["I_required_n_holm"] = partI

    print("\nsame at tau=60 (the threshold the power criterion prefers):")
    per_rep60, _ = x_tau_table(cells, taus=(60.0,))
    partI60 = {}
    for m, r in DECISION_CELLS:
        a4 = arm_vector(per_rep60, m, A4, r, 60.0)
        sig = math.sqrt(2) * float(a4.std(ddof=1))
        row = {"sigma": sig,
               "n80_a05": min_n_alpha(sig, DELTA, 0.0, 0.05, 0.80, rng=rng),
               "n80_a0125": min_n_alpha(sig, DELTA, 0.0, 0.0125, 0.80, rng=rng)}
        partI60[f"{m}|r{r:g}"] = row
        print(f"{cell_label(m, r):14s} {sig:7.4f} n80(a=.05)={str(row['n80_a05']):>4s} "
              f"n80(a=.0125)={str(row['n80_a0125']):>4s}")
    report["I_required_n_tau60"] = partI60

    # ------------------------------------------------------------------ (J)
    hr("(J) TOST size at the boundary |Delta| = delta (consumer risk), tau=40")
    print("nominal <= 0.05.  Checked under normal errors and under two empirical shapes.")
    print(f"{'cell':14s} {'sigma P-a':>10s} {'normal':>8s} {'emp A4-pairs':>13s} "
          f"{'sigma P-b':>10s} {'normal':>8s} {'emp D14':>9s}")
    partJ = {}
    for m, r in DECISION_CELLS:
        a1 = arm_vector(per_rep, m, A1, r, 40.0)
        a4 = arm_vector(per_rep, m, A4, r, 40.0)
        sa = math.sqrt(2) * float(a4.std(ddof=1))
        sb = float((a1 - a4).std(ddof=1))
        sha, shb = pair_difference_shape(a4), standardised_shape(a1 - a4)
        vals = {
            "pa_normal": tost_power(sa, 5, DELTA, DELTA, trials=200_000, rng=rng),
            "pa_emp": tost_power(sa, 5, DELTA, DELTA, trials=200_000, shape=sha, rng=rng),
            "pb_normal": tost_power(sb, 5, DELTA, DELTA, trials=200_000, rng=rng),
            "pb_emp": (tost_power(sb, 5, DELTA, DELTA, trials=200_000, shape=shb, rng=rng)
                       if shb is not None else float("nan")),
        }
        partJ[f"{m}|r{r:g}"] = vals
        print(f"{cell_label(m, r):14s} {sa:10.4f} {vals['pa_normal']:8.4f} {vals['pa_emp']:13.4f} "
              f"{sb:10.4f} {vals['pb_normal']:8.4f} {vals['pb_emp']:9.4f}")
    report["J_boundary_size"] = partJ

    # ------------------------------------------------------------------ (K)
    hr("(K) how much of the A1->A4 gap must A3 close, as a function of delta")
    print("R3' fires when |mu(A3)-mu(A4)| < delta with 90% confidence.  In gap units that is")
    print("'A3 closed at least (1 - delta/gap) of the A1->A4 gap'.")
    print(f"{'delta':>7s} " + " ".join(f"{cell_label(m, r):>14s}" for m, r in DECISION_CELLS)
          + f" | {'designated n80@tau40':>21s} {'designated n80@tau60':>21s}")
    partK = {}
    per_rep_k, _ = x_tau_table(cells, taus=(40.0, 60.0))
    a4_40 = arm_vector(per_rep_k, *(DESIGNATED[0], A4, DESIGNATED[1]), 40.0) \
        if False else arm_vector(per_rep_k, DESIGNATED[0], A4, DESIGNATED[1], 40.0)
    a4_60 = arm_vector(per_rep_k, DESIGNATED[0], A4, DESIGNATED[1], 60.0)
    s40 = math.sqrt(2) * float(a4_40.std(ddof=1))
    s60 = math.sqrt(2) * float(a4_60.std(ddof=1))
    for d in (0.02, 0.05, 0.08, 0.10, 0.15, 0.19, 0.25):
        fr = []
        for m, r in DECISION_CELLS:
            a1 = arm_vector(per_rep_k, m, A1, r, 40.0)
            a4 = arm_vector(per_rep_k, m, A4, r, 40.0)
            gap = float(a1.mean() - a4.mean())
            fr.append(1 - d / gap)
        n40 = min_n_alpha(s40, d, 0.0, 0.05, 0.80, rng=rng)
        n60 = min_n_alpha(s60, d, 0.0, 0.05, 0.80, rng=rng)
        partK[f"{d:.2f}"] = {"gap_fraction_required": fr, "n80_tau40": n40, "n80_tau60": n60}
        print(f"{d:7.2f} " + " ".join(f"{100 * f:13.1f}%" for f in fr)
              + f" | {str(n40):>21s} {str(n60):>21s}")
    report["K_delta_vs_gapfraction"] = partK

    (OUT / "g2_cliff_sensitivity.json").write_text(json.dumps(report, indent=1, default=str))
    print(f"\nwrote {OUT / 'g2_cliff_sensitivity.json'}")


if __name__ == "__main__":
    main()
