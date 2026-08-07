"""Gate-2 rev3 §11-2, part 2:  IS tau=40 the right threshold, and what breaks?

  (A) fine tau sweep 28..80 ms: A4 level/SD, gap, TOST power -- per cell and min-over-cells
  (B) metric-cliff diagnosis: distribution of per-request max(ITL) around tau
  (C) Holm-adjusted TOST power for the 4 replication cells (prereg §5.3)
  (D) floor degeneracy: how often does SD(D)=0 make the paired t interval a point?
  (E) the Zamba2 r3 outlier rep that drives sigma up 5x at tau=40

Aggregation units: see verify_g2_tau40_lib.py header.  GPU 0.  No new performance verdict.
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from verify_load import REPS, load_cells  # noqa: E402
from verify_g2_tau40_lib import (  # noqa: E402
    A1, A4, DECISION_CELLS, DELTA, DESIGNATED, arm_vector, cell_label,
    min_n_for_power, t_ppf, tost_power, x_tau_table,
)

OUT = Path(__file__).parent
report = {}
rng = np.random.default_rng(1)


def hr(t):
    print("\n" + "=" * 104)
    print(t)
    print("=" * 104)


def tost_power_alpha(sigma, n, delta, gap, alpha, trials=100_000, rng=None):
    """TOST at a general per-side alpha (Holm-adjusted replication cells)."""
    rng = rng or np.random.default_rng(3)
    d = rng.normal(gap, sigma, size=(trials, n))
    mean = d.mean(axis=1)
    se = d.std(axis=1, ddof=1) / math.sqrt(n)
    tc = t_ppf(1 - alpha, n - 1)
    return float(((mean - tc * se > -delta) & (mean + tc * se < delta)).mean())


def main():
    cells = load_cells()
    fine = tuple(float(t) for t in range(28, 81, 1))
    per_rep, _ = x_tau_table(cells, taus=fine)

    # ------------------------------------------------------------------ (A)
    hr("(A) fine tau sweep -- TOST power at true gap 0, delta=0.05, n=5, sigma = sqrt(2)*SD(A4)")
    print("cols per cell: A4 mean / sqrt(2)*SD(A4) / gap(A1-A4) / P(equiv|gap=0)")
    print("a cell is marked V (vacuous) when delta >= 0.25*gap, i.e. the band swallows >25% of "
          "the A1->A4 gap\n")
    hdr = f"{'tau':>4s}"
    for m, r in DECISION_CELLS:
        hdr += f" | {cell_label(m, r):>26s}"
    hdr += f" | {'MIN pow':>7s} {'designated pow':>14s}"
    print(hdr)
    sweep = {}
    best = []
    for tau in fine:
        line = f"{tau:4.0f}"
        powers = {}
        row = {}
        for m, r in DECISION_CELLS:
            a1 = arm_vector(per_rep, m, A1, r, tau)
            a4 = arm_vector(per_rep, m, A4, r, tau)
            sig = math.sqrt(2) * float(a4.std(ddof=1))
            gap = float(a1.mean() - a4.mean())
            pw = tost_power(sig, 5, DELTA, 0.0, trials=40_000, rng=rng)
            vac = (gap <= 0) or (DELTA / gap > 0.25)
            powers[(m, r)] = -1.0 if vac else pw
            row[f"{m}|r{r:g}"] = {"a4_mean": float(a4.mean()), "sigma": sig, "gap": gap,
                                  "power_gap0": pw, "vacuous": bool(vac),
                                  "a4_n0": int((a4 == 0).sum()),
                                  "a1_n1": int((a1 == 1.0).sum())}
            line += (f" | {a4.mean():6.3f} {sig:6.4f} {gap:6.3f} "
                     + (f"{'V':>5s}" if vac else f"{pw:5.3f}"))
        mn = min(powers.values())
        dp = powers[DESIGNATED]
        line += f" | {mn:7.3f} {dp:14.3f}"
        print(line)
        sweep[f"t{tau:g}"] = {"cells": row, "min_power": mn, "designated_power": dp}
        best.append((mn, dp, tau))
    report["A_fine_sweep"] = sweep
    feasible = [b for b in best if b[0] >= 0]
    bmin = max(feasible, key=lambda b: (b[0], b[1]))
    bdes = max(feasible, key=lambda b: (b[1], b[0]))
    print(f"\nbest tau by MIN-over-5-cells power  : tau = {bmin[2]:.0f}  (min power {bmin[0]:.3f}, "
          f"designated {bmin[1]:.3f})")
    print(f"best tau by DESIGNATED-cell power   : tau = {bdes[2]:.0f}  (designated {bdes[1]:.3f}, "
          f"min {bdes[0]:.3f})")
    print(f"prereg choice tau=40                : min power {sweep['t40']['min_power']:.3f}, "
          f"designated {sweep['t40']['designated_power']:.3f}")
    report["A_best"] = {"by_min_power": bmin, "by_designated": bdes,
                        "tau40": [sweep["t40"]["min_power"], sweep["t40"]["designated_power"], 40.0]}

    # ------------------------------------------------------------------ (B)
    hr("(B) metric cliff -- where does per-request max(ITL) actually live?  (5 reps pooled)")
    print("If tau sits on a mode of the max-ITL distribution, a 1ms move in tau (or any 1ms")
    print("perturbation of the engine) moves X a lot -> maximal rep-to-rep variance.")
    print(f"\n{'cell':14s} {'arm':9s} " + " ".join(f"{'[%d,%d)' % (lo, lo + 5):>9s}"
                                                   for lo in range(25, 70, 5))
          + f" {'>=70':>8s} | {'dX/dtau@40':>11s} {'dX/dtau@50':>11s} {'dX/dtau@60':>11s}")
    partB = {}
    for m, r in DECISION_CELLS:
        for arm in (A4, A1):
            maxes = []
            for rep in REPS:
                for q in cells[(m, arm, r, rep)]["requests"]:
                    if q.token_itl_ms:
                        maxes.append(max(q.token_itl_ms))
            mx = np.asarray(maxes)
            n = mx.size
            hist = [int(((mx >= lo) & (mx < lo + 5)).sum()) for lo in range(25, 70, 5)]
            tail = int((mx >= 70).sum())

            def slope(t0):  # dX/dtau over a +-2ms window, per ms, in X units
                return -float(((mx > t0 - 2) & (mx <= t0 + 2)).sum()) / n / 4.0
            partB[f"{m}|r{r:g}|{arm}"] = {"n": n, "hist_25_70_by5": hist, "ge70": tail,
                                          "dXdtau_40": slope(40), "dXdtau_50": slope(50),
                                          "dXdtau_60": slope(60),
                                          "median_max_itl": float(np.median(mx)),
                                          "p90_max_itl": float(np.percentile(mx, 90))}
            print(f"{cell_label(m, r):14s} {arm:9s} "
                  + " ".join(f"{h / n:9.4f}" for h in hist)
                  + f" {tail / n:8.4f} | {slope(40):11.5f} {slope(50):11.5f} {slope(60):11.5f}")
    report["B_cliff"] = partB
    print("\n(dX/dtau is per ms, averaged over a +-2ms window; more negative = steeper cliff)")

    # ------------------------------------------------------------------ (C)
    hr("(C) Holm-adjusted TOST power for the 4 replication cells (prereg §5.3)")
    print("Holm over the 4 replication cells: the most extreme test is compared at alpha/4.")
    print("Reported = power at the WORST (alpha/4) and BEST (alpha/1) Holm step, gap=0, n=5, tau=40.")
    print(f"{'cell':14s} {'sigma':>7s} {'a=.050':>8s} {'a=.025':>8s} {'a=.0167':>8s} {'a=.0125':>8s}")
    per_rep40, _ = x_tau_table(cells, taus=(40.0,))
    partC = {}
    for m, r in DECISION_CELLS:
        a4 = arm_vector(per_rep40, m, A4, r, 40.0)
        sig = math.sqrt(2) * float(a4.std(ddof=1))
        vals = [tost_power_alpha(sig, 5, DELTA, 0.0, a, rng=rng)
                for a in (0.05, 0.025, 0.05 / 3, 0.0125)]
        partC[f"{m}|r{r:g}"] = {"sigma": sig, "alpha_ladder": vals}
        print(f"{cell_label(m, r):14s} {sig:7.4f} " + " ".join(f"{v:8.3f}" for v in vals))
    report["C_holm"] = partC

    # ------------------------------------------------------------------ (D)
    hr("(D) floor degeneracy -- does A4's discreteness break the paired t interval?")
    print("X lives on the lattice 1/120 = 0.008333.  delta=0.05 = exactly 6 lattice steps.")
    print("Null resample: draw BOTH arms' 5 rep values iid from the observed A4 rep vector")
    print("(exchangeability null, true gap 0) and score the paired-t TOST on the lattice.")
    print(f"{'cell':14s} {'tau':>4s} {'A4 n0':>6s} {'P(SD=0)':>8s} {'P(equiv)':>9s} "
          f"{'P(equiv|SD=0)':>14s}")
    partD = {}
    fine_taus = (40.0, 45.0, 50.0, 60.0)
    per_rep_d, _ = x_tau_table(cells, taus=fine_taus)
    tc = t_ppf(0.95, 4)
    for m, r in DECISION_CELLS:
        for tau in fine_taus:
            v = arm_vector(per_rep_d, m, A4, r, tau)
            draws_a = rng.choice(v, size=(100_000, 5))
            draws_b = rng.choice(v, size=(100_000, 5))
            d = draws_a - draws_b
            mean = d.mean(axis=1)
            sd = d.std(axis=1, ddof=1)
            se = sd / math.sqrt(5)
            eq = (mean - tc * se > -DELTA) & (mean + tc * se < DELTA)
            zero = sd == 0
            partD[f"{m}|r{r:g}|t{tau:g}"] = {
                "a4_n0": int((v == 0).sum()), "p_sd_zero": float(zero.mean()),
                "p_equiv": float(eq.mean()),
                "p_equiv_given_sd0": float(eq[zero].mean()) if zero.any() else float("nan")}
            print(f"{cell_label(m, r):14s} {tau:4.0f} {int((v == 0).sum()):6d} "
                  f"{zero.mean():8.4f} {eq.mean():9.4f} "
                  f"{(eq[zero].mean() if zero.any() else float('nan')):14.4f}")
    report["D_degeneracy"] = partD

    # ------------------------------------------------------------------ (E)
    hr("(E) the Zamba2 r3 rep that drives sigma up 5x at tau=40")
    print(f"{'arm':9s} {'rep':>4s} {'X_35':>7s} {'X_40':>7s} {'X_45':>7s} {'X_50':>7s} {'X_60':>7s} "
          f"{'medMax':>8s} {'p90Max':>8s} {'p99Max':>8s} {'maxMax':>9s} {'dur_s':>7s}")
    per_rep_e, _ = x_tau_table(cells, taus=(35.0, 40.0, 45.0, 50.0, 60.0))
    partE = {}
    for arm in (A4, A1):
        for rep in REPS:
            c = cells[("zamba2-27b", arm, 3.0, rep)]
            mx = np.array([max(q.token_itl_ms) for q in c["requests"] if q.token_itl_ms])
            row = {t: per_rep_e[("zamba2-27b", arm, 3.0, rep, t)]["x"]
                   for t in (35.0, 40.0, 45.0, 50.0, 60.0)}
            partE[f"{arm}|rep{rep}"] = {"x": row, "median_max": float(np.median(mx)),
                                        "p90_max": float(np.percentile(mx, 90)),
                                        "p99_max": float(np.percentile(mx, 99)),
                                        "max_max": float(mx.max()),
                                        "duration": c["duration"]}
            print(f"{arm:9s} {rep:4d} " + " ".join(f"{row[t]:7.4f}" for t in
                                                   (35.0, 40.0, 45.0, 50.0, 60.0))
                  + f" {np.median(mx):8.3f} {np.percentile(mx, 90):8.3f} "
                    f"{np.percentile(mx, 99):8.3f} {mx.max():9.2f} {c['duration']:7.2f}")
    report["E_zamba_r3"] = partE

    # ------------------------------------------------------------------ (F)
    hr("(F) required n at the tau that the power criterion prefers")
    for tau in (40.0, 45.0, 50.0, 60.0):
        prd, _ = x_tau_table(cells, taus=(tau,))
        out = []
        for m, r in DECISION_CELLS:
            a4 = arm_vector(prd, m, A4, r, tau)
            sig = math.sqrt(2) * float(a4.std(ddof=1))
            n80 = min_n_for_power(sig, DELTA, 0.0, 0.80, rng=rng)
            n80b = min_n_for_power(sig, DELTA, 0.01, 0.80, rng=rng)
            out.append((cell_label(m, r), sig, n80, n80b))
        print(f"tau={tau:.0f}: " + "   ".join(
            f"{lab}: sig {s:.4f} n80(g=0)={n0} n80(g=.01)={n1}" for lab, s, n0, n1 in out))
        report[f"F_required_n_tau{tau:g}"] = [
            {"cell": lab, "sigma": s, "n80_gap0": n0, "n80_gap001": n1} for lab, s, n0, n1 in out]

    (OUT / "g2_tau_choice.json").write_text(json.dumps(report, indent=1, default=str))
    print(f"\nwrote {OUT / 'g2_tau_choice.json'}")


if __name__ == "__main__":
    main()
