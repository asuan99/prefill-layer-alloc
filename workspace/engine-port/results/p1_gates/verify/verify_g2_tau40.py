"""Gate-2 rev3 submission precondition §11-2.

Q: is delta = 0.05 (and the n=5 TOST power behind it) still defensible after the
   primary threshold moved tau = 60 -> tau = 40?

GPU 0.  Only the existing 873944/873945 artifacts are used.  NO new performance
verdict is produced here; this is a power/scale calculation.

Aggregation units are declared in verify_g2_tau40_lib.py (top of file).
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
    A1, A4, DECISION_CELLS, DELTA, DESIGNATED, TAU_LADDER,
    arm_vector, cell_label, descr, min_delta_for_power, min_n_for_power, moments,
    pair_difference_shape, paired_t_ci, standardised_shape, t_ppf, token_p_from_x,
    tost_power, x_tau_table,
)

OUT = Path(__file__).parent
TRIALS = 200_000
GAPS = (0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10)
report = {}


def hr(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def main():
    cells = load_cells()
    per_rep, p_token = x_tau_table(cells)
    rng = np.random.default_rng(1)

    # ================================================================= PART 0
    hr("PART 0 -- what exists.  A2/A3 DO NOT EXIST.  Everything below is a PROXY.")
    print("arms present in 873944/873945 :", sorted({k[1] for k in cells}))
    print("prereg primary estimand       : Delta = mu(X_40(A3)) - mu(X_40(A4))")
    print("directly estimable here       : X_40(A1), X_40(A4), and D_14 = A1 - A4")
    print("=> Var(A3 - A4) is NOT identified by this data.  Proxies are enumerated in PART 2.")

    # ================================================================= PART 1
    hr("PART 1 -- X_40 per arm per rep, five rev3 decision cells")
    print(f"{'cell':14s} {'arm':9s} " + " ".join(f"{'rep%d' % r:>8s}" for r in REPS)
          + f" {'mean':>8s} {'SD':>8s} {'CV':>7s} {'n0':>3s} {'n1':>3s} {'binomSD':>8s}"
          + f" {'nUsed':>6s} {'nMiss':>6s}")
    part1 = {}
    for model, rate in DECISION_CELLS:
        for arm in (A1, A4):
            v = arm_vector(per_rep, model, arm, rate, 40.0)
            d = descr(v)
            n_used = [per_rep[(model, arm, rate, r, 40.0)]["n_used"] for r in REPS]
            n_miss = [per_rep[(model, arm, rate, r, 40.0)]["n_missing_itl"] for r in REPS]
            nu = float(np.mean(n_used))
            binom_sd = math.sqrt(max(d["mean"] * (1 - d["mean"]), 0.0) / nu)
            d.update(n_used=n_used, n_missing_itl=n_miss, binom_sd=binom_sd)
            part1[f"{model}|r{rate:g}|{arm}"] = d
            print(f"{cell_label(model, rate):14s} {arm:9s} "
                  + " ".join(f"{x:8.4f}" for x in v)
                  + f" {d['mean']:8.4f} {d['sd']:8.4f} {d['cv']:7.3f} {d['n_zero']:3d} "
                  f"{d['n_one']:3d} {binom_sd:8.4f} {sum(n_used):6d} {sum(n_miss):6d}")
    report["part1_x40"] = part1

    # ================================================================= PART 2
    hr("PART 2 -- variance proxies for SD(A3 - A4) at tau=40  (the missing quantity)")
    print("P-a  sqrt(2) * SD_rep(A4)        : upper bound IF A3 ~ A4 in law AND corr(A3,A4) >= 0")
    print("P-b  SD of the paired D = A1-A4  : the only DIRECTLY OBSERVED paired-difference SD")
    print("P-c  sqrt(2) * binomial SD(A4)   : sampling-only floor (120 requests/rep)")
    print("P-d  corr(A1,A4) across reps     : does pairing on the seed remove any variance at all?")
    print()
    print(f"{'cell':14s} {'SD(A4)':>8s} {'SD(A1)':>8s} {'binSD(A4)':>10s} {'SD(D14)':>9s} "
          f"{'sqrt2*SD(A4)':>13s} {'r(A1,A4)':>9s} {'varRed%':>8s} {'gap=A1-A4':>10s} "
          f"{'d/gap':>7s} {'d/A4':>7s}")
    part2 = {}
    for model, rate in DECISION_CELLS:
        a1 = arm_vector(per_rep, model, A1, rate, 40.0)
        a4 = arm_vector(per_rep, model, A4, rate, 40.0)
        d14 = a1 - a4
        sd1, sd4 = a1.std(ddof=1), a4.std(ddof=1)
        sd_d = d14.std(ddof=1)
        corr = float(np.corrcoef(a1, a4)[0, 1]) if sd1 > 0 and sd4 > 0 else float("nan")
        indep = math.sqrt(sd1 ** 2 + sd4 ** 2)
        var_red = 100 * (1 - (sd_d / indep) ** 2) if indep > 0 else float("nan")
        nu = float(np.mean([per_rep[(model, A4, rate, r, 40.0)]["n_used"] for r in REPS]))
        m4 = float(a4.mean())
        bin_sd = math.sqrt(max(m4 * (1 - m4), 0) / nu)
        gap = float(a1.mean() - a4.mean())
        row = {"sd_a1": float(sd1), "sd_a4": float(sd4), "sd_d14": float(sd_d),
               "binom_sd_a4": bin_sd, "sqrt2_sd_a4": math.sqrt(2) * float(sd4),
               "corr": corr, "var_reduction_pct_from_pairing": var_red,
               "gap_a1_a4": gap, "delta_over_gap": DELTA / gap if gap else float("nan"),
               "delta_over_a4_mean": DELTA / m4 if m4 else float("inf"),
               "d14_values": [float(x) for x in d14]}
        part2[f"{model}|r{rate:g}"] = row
        print(f"{cell_label(model, rate):14s} {sd4:8.4f} {sd1:8.4f} {bin_sd:10.4f} {sd_d:9.4f} "
              f"{math.sqrt(2) * sd4:13.4f} {corr:9.3f} {var_red:8.1f} {gap:10.4f} "
              f"{DELTA / gap if gap else float('nan'):7.3f} {DELTA / m4 if m4 else float('inf'):7.2f}")
    report["part2_variance_proxies"] = part2

    print("\nshape of the observed rep-level noise (standardised), tau=40:")
    print(f"{'cell':14s} {'skew(A4)':>9s} {'exkurt(A4)':>11s} {'skew(D14)':>10s} {'exkurt(D14)':>12s}")
    for model, rate in DECISION_CELLS:
        a4 = arm_vector(per_rep, model, A4, rate, 40.0)
        d14 = arm_vector(per_rep, model, A1, rate, 40.0) - a4
        m4, md = moments(a4), moments(d14)
        print(f"{cell_label(model, rate):14s} {m4['skew']:9.3f} {m4['exkurt']:11.3f} "
              f"{md['skew']:10.3f} {md['exkurt']:12.3f}")

    # ================================================================= PART 3
    hr("PART 3 -- TOST power at tau=40, n=5, delta=0.05, paired t (90% two-sided CI)")
    print("NOTE: percentile bootstrap is NOT used (n=5 coverage 0.840, canonical rev13 gate).")
    print(f"t_(0.95, df=4) = {t_ppf(0.95, 4):.6f}   trials = {TRIALS}\n")
    part3 = {}
    for model, rate in DECISION_CELLS:
        a4 = arm_vector(per_rep, model, A4, rate, 40.0)
        a1 = arm_vector(per_rep, model, A1, rate, 40.0)
        d14 = a1 - a4
        sig_a = math.sqrt(2) * float(a4.std(ddof=1))          # proxy P-a
        sig_b = float(d14.std(ddof=1))                        # proxy P-b
        shape_a = pair_difference_shape(a4)
        shape_b = standardised_shape(d14)
        lab = cell_label(model, rate)
        star = "  <== DESIGNATED CELL (prereg §5.3)" if (model, rate) == DESIGNATED else ""
        print(f"--- {lab}   sigma_P-a = {sig_a:.4f}   sigma_P-b = {sig_b:.4f}{star}")
        print(f"{'true gap':>9s} | {'P-a normal':>10s} {'P-a empir':>10s} | "
              f"{'P-b normal':>10s} {'P-b empir':>10s}")
        rows = {}
        for g in GAPS:
            pa_n = tost_power(sig_a, 5, DELTA, g, trials=TRIALS, rng=rng)
            pa_e = tost_power(sig_a, 5, DELTA, g, trials=TRIALS, shape=shape_a, rng=rng)
            pb_n = tost_power(sig_b, 5, DELTA, g, trials=TRIALS, rng=rng)
            pb_e = (tost_power(sig_b, 5, DELTA, g, trials=TRIALS, shape=shape_b, rng=rng)
                    if shape_b is not None else float("nan"))
            rows[f"{g:.2f}"] = {"pa_normal": pa_n, "pa_empirical": pa_e,
                                "pb_normal": pb_n, "pb_empirical": pb_e}
            print(f"{g:9.2f} | {pa_n:10.3f} {pa_e:10.3f} | {pb_n:10.3f} {pb_e:10.3f}")
        part3[f"{model}|r{rate:g}"] = {"sigma_pa": sig_a, "sigma_pb": sig_b, "power": rows}
        print()
    report["part3_power"] = part3

    print("reference: the 2nd-audit MC that justified delta=0.05 (tau=60 structure) reported")
    print("  gap 0 -> 0.975 | gap 0.05 -> 0.070 | gap 0.10 -> 0.002")
    print("its implied sigma is recovered in PART 3b.")

    # -- 3b: what sigma does the auditor's tau=60 MC correspond to?
    hr("PART 3b -- reverse-engineer the sigma behind the tau=60 audit MC (P(equiv|gap=0)=0.975)")
    lo, hi = 1e-5, 0.2
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if tost_power(mid, 5, DELTA, 0.0, trials=80_000, rng=rng) > 0.975:
            lo = mid
        else:
            hi = mid
    sig60 = 0.5 * (lo + hi)
    print(f"sigma implied by P(equiv | gap=0) = 0.975 at n=5, delta=0.05  ->  {sig60:.4f}")
    print(f"  cross-check: power at gap 0.05 = {tost_power(sig60, 5, DELTA, 0.05, trials=200_000, rng=rng):.3f} "
          f"(audit said 0.070);  at 0.10 = {tost_power(sig60, 5, DELTA, 0.10, trials=200_000, rng=rng):.3f} "
          f"(audit said 0.002)")
    print("\nobserved tau=60 SD(A4) and SD(D14) for comparison:")
    print(f"{'cell':14s} {'SD(A4)@60':>10s} {'sqrt2*SD':>9s} {'SD(D14)@60':>11s} | "
          f"{'SD(A4)@40':>10s} {'sqrt2*SD':>9s} {'SD(D14)@40':>11s} {'ratio 40/60':>12s}")
    part3b = {"sigma_implied_by_audit_mc": sig60, "cells": {}}
    for model, rate in DECISION_CELLS:
        r = {}
        for tau in (60.0, 40.0):
            a1 = arm_vector(per_rep, model, A1, rate, tau)
            a4 = arm_vector(per_rep, model, A4, rate, tau)
            r[tau] = (float(a4.std(ddof=1)), math.sqrt(2) * float(a4.std(ddof=1)),
                      float((a1 - a4).std(ddof=1)))
        ratio = (r[40.0][1] / r[60.0][1]) if r[60.0][1] > 0 else float("inf")
        part3b["cells"][f"{model}|r{rate:g}"] = {"tau60": r[60.0], "tau40": r[40.0],
                                                 "sqrt2sd_ratio_40_over_60": ratio}
        print(f"{cell_label(model, rate):14s} {r[60.0][0]:10.4f} {r[60.0][1]:9.4f} {r[60.0][2]:11.4f} | "
              f"{r[40.0][0]:10.4f} {r[40.0][1]:9.4f} {r[40.0][2]:11.4f} {ratio:12.2f}")
    report["part3b_tau60_vs_tau40"] = part3b

    # ================================================================= PART 4
    hr("PART 4 -- what does delta = 0.05 MEAN?  (scale audit, prereg §9-3)")
    print("(i) request scale: 0.05 = 6 requests out of 120 scored requests per rep.")
    print("(ii) per-token scale, independence model  p = 1 - (1-X)^(1/G), G = gaps/request")
    print("(iii) EMPIRICAL per-token stall rate (pooled 5 reps) -- tests the independence model")
    print()
    print(f"{'cell':14s} {'X40(A4)':>8s} {'X40(A1)':>8s} {'G':>5s} | "
          f"{'p(A4) model':>11s} {'p(A4) emp':>10s} {'clust':>6s} | "
          f"{'p(A4+d)':>9s} {'dp(ppm)':>9s} {'dp/p(A4)':>9s} | {'dp at A1(ppm)':>13s} {'ratio':>6s}")
    part4 = {}
    for model, rate in DECISION_CELLS:
        a1 = arm_vector(per_rep, model, A1, rate, 40.0)
        a4 = arm_vector(per_rep, model, A4, rate, 40.0)
        x4, x1 = float(a4.mean()), float(a1.mean())
        G = float(np.mean([per_rep[(model, A4, rate, r, 40.0)]["n_gaps"]
                           / per_rep[(model, A4, rate, r, 40.0)]["n_used"] for r in REPS]))
        p4 = token_p_from_x(x4, G)
        p4_emp = p_token[(model, A4, rate, 40.0)]
        p4d = token_p_from_x(min(x4 + DELTA, 0.999999), G)
        dp_at_a4 = p4d - p4
        p1 = token_p_from_x(x1, G)
        p1d = token_p_from_x(min(x1 + DELTA, 0.999999), G)
        dp_at_a1 = p1d - p1
        row = {"x40_a4": x4, "x40_a1": x1, "G": G,
               "p_token_model_a4": p4, "p_token_empirical_a4": p4_emp,
               "clustering_ratio_model_over_empirical": p4 / p4_emp if p4_emp else float("inf"),
               "delta_in_token_p_at_a4": dp_at_a4,
               "delta_in_token_p_at_a1": dp_at_a1,
               "relative_delta_at_a4": dp_at_a4 / p4 if p4 else float("inf"),
               "nonuniformity_a1_over_a4": dp_at_a1 / dp_at_a4 if dp_at_a4 else float("inf")}
        part4[f"{model}|r{rate:g}"] = row
        print(f"{cell_label(model, rate):14s} {x4:8.4f} {x1:8.4f} {G:5.1f} | "
              f"{p4:11.6f} {p4_emp:10.6f} {row['clustering_ratio_model_over_empirical']:6.2f} | "
              f"{p4d:9.6f} {1e6 * dp_at_a4:9.1f} {row['relative_delta_at_a4']:9.3f} | "
              f"{1e6 * dp_at_a1:13.1f} {row['nonuniformity_a1_over_a4']:6.2f}")
    span = [v["delta_in_token_p_at_a4"] for v in part4.values()]
    print(f"\ndelta=0.05 spans {1e6 * min(span):.1f} - {1e6 * max(span):.1f} ppm of per-token stall "
          f"probability across the 5 cells  =>  spread factor {max(span) / min(span):.2f}x")
    part4["_span_ppm"] = [1e6 * min(span), 1e6 * max(span), max(span) / min(span)]
    report["part4_delta_scale"] = part4

    # ---- 4b: what an "equivalent" verdict would tolerate, in gap units
    hr("PART 4b -- vacuity check: can TOST declare A3 == A4 while A3 is indistinguishable from A1?")
    print("If delta >= gap(A1,A4), then 'A3 == A4' is compatible with 'chunk512 did NOTHING'.")
    print(f"{'cell':14s} {'gap(A1-A4)':>11s} {'delta/gap':>10s} "
          f"{'A3=A1 in band?':>15s} {'max frac of gap A3 may miss':>29s}")
    part4b = {}
    for model, rate in DECISION_CELLS:
        g = part2[f"{model}|r{rate:g}"]["gap_a1_a4"]
        ratio = DELTA / g if g else float("nan")
        part4b[f"{model}|r{rate:g}"] = {"gap": g, "delta_over_gap": ratio,
                                        "a3_equals_a1_inside_band": bool(g <= DELTA)}
        print(f"{cell_label(model, rate):14s} {g:11.4f} {ratio:10.3f} "
              f"{('YES -- VACUOUS' if g <= DELTA else 'no'):>15s} {min(ratio, 1.0) * 100:28.1f}%")
    report["part4b_vacuity"] = part4b

    # ================================================================= PART 5
    hr("PART 5 -- required n (prereg §5.1: raise n, never delta)")
    print("target = P(declare equivalence | true gap = g) >= 0.80 and >= 0.90, delta=0.05, paired t")
    print(f"{'cell':14s} {'proxy':6s} {'sigma':>7s} | "
          + " ".join(f"{'n80@%.2f' % g:>10s}" for g in (0.0, 0.01, 0.02))
          + " | " + " ".join(f"{'n90@%.2f' % g:>10s}" for g in (0.0, 0.01, 0.02))
          + f" | {'delta_min@n5':>12s}")
    part5 = {}
    for model, rate in DECISION_CELLS:
        a4 = arm_vector(per_rep, model, A4, rate, 40.0)
        a1 = arm_vector(per_rep, model, A1, rate, 40.0)
        for pname, sig in (("P-a", math.sqrt(2) * float(a4.std(ddof=1))),
                           ("P-b", float((a1 - a4).std(ddof=1)))):
            n80 = [min_n_for_power(sig, DELTA, g, 0.80, rng=rng) for g in (0.0, 0.01, 0.02)]
            n90 = [min_n_for_power(sig, DELTA, g, 0.90, rng=rng) for g in (0.0, 0.01, 0.02)]
            dmin = min_delta_for_power(sig, 5, 0.0, 0.80, rng=rng)
            part5[f"{model}|r{rate:g}|{pname}"] = {
                "sigma": sig, "n80": n80, "n90": n90, "delta_min_at_n5_power80": dmin}
            print(f"{cell_label(model, rate):14s} {pname:6s} {sig:7.4f} | "
                  + " ".join(f"{str(x):>10s}" for x in n80) + " | "
                  + " ".join(f"{str(x):>10s}" for x in n90) + f" | {dmin:12.4f}")
    report["part5_required_n"] = part5

    print("\nSTRUCTURAL FLOOR (independent of power): the exact two-sided sign-flip permutation")
    print("p-value for n=5 paired has minimum 2/2^5 = 0.0625.  This does NOT bind the TOST")
    print("(equivalence is one-sided x2 and is run parametrically), but it DOES bind R4'")
    print("('A4 significantly better', §5.2) if a distribution-free test is ever demanded:")
    for n in range(4, 9):
        print(f"   n={n}: min two-sided permutation p = 2/2^{n} = {2 / 2 ** n:.4f}"
              + ("   <- first n reaching p<0.05" if 2 / 2 ** n < 0.05 <= 2 / 2 ** (n - 1) else ""))

    # ================================================================= PART 6
    hr("PART 6 -- is tau=40 actually the best choice?  (prereg §9-2 self-declared weakness)")
    print("prereg criterion  = 'A4 farthest from the floor'  (maximise X_40(A4))")
    print("TOST criterion    = minimise SD(A3-A4) at FIXED ABSOLUTE delta, subject to non-vacuity")
    print("These are OPPOSED: moving X away from 0 raises the binomial SD.\n")
    tau_show = (30.0, 40.0, 50.0, 60.0)
    print(f"{'cell':14s} {'tau':>5s} {'A4 mean':>8s} {'A4 SD':>7s} {'A4 CV':>7s} {'A4 n0':>6s} "
          f"{'A1 mean':>8s} {'A1 n1':>6s} {'gap':>7s} {'d/gap':>7s} {'sig P-a':>8s} "
          f"{'sig P-b':>8s} {'pow g=0':>8s} {'pow g=.02':>9s} {'vacuous?':>9s}")
    part6 = {}
    for model, rate in DECISION_CELLS:
        for tau in tau_show:
            a1 = arm_vector(per_rep, model, A1, rate, tau)
            a4 = arm_vector(per_rep, model, A4, rate, tau)
            sa = math.sqrt(2) * float(a4.std(ddof=1))
            sb = float((a1 - a4).std(ddof=1))
            gap = float(a1.mean() - a4.mean())
            pw0 = tost_power(sa, 5, DELTA, 0.0, trials=100_000, rng=rng)
            pw2 = tost_power(sa, 5, DELTA, 0.02, trials=100_000, rng=rng)
            vac = gap <= DELTA
            part6[f"{model}|r{rate:g}|t{tau:g}"] = {
                "a4_mean": float(a4.mean()), "a4_sd": float(a4.std(ddof=1)),
                "a4_cv": float(a4.std(ddof=1) / a4.mean()) if a4.mean() > 0 else float("nan"),
                "a4_n_zero": int((a4 == 0).sum()), "a1_mean": float(a1.mean()),
                "a1_n_one": int((a1 == 1.0).sum()), "gap": gap,
                "delta_over_gap": DELTA / gap if gap else float("nan"),
                "sigma_pa": sa, "sigma_pb": sb, "power_gap0": pw0, "power_gap002": pw2,
                "vacuous_delta_ge_gap": bool(vac),
                "a4_values": [float(x) for x in a4], "a1_values": [float(x) for x in a1]}
            cv = a4.std(ddof=1) / a4.mean() if a4.mean() > 0 else float("nan")
            print(f"{cell_label(model, rate):14s} {tau:5.0f} {a4.mean():8.4f} {a4.std(ddof=1):7.4f} "
                  f"{cv:7.3f} {int((a4 == 0).sum()):6d} {a1.mean():8.4f} {int((a1 == 1.0).sum()):6d} "
                  f"{gap:7.4f} {DELTA / gap if gap else float('nan'):7.3f} {sa:8.4f} {sb:8.4f} "
                  f"{pw0:8.3f} {pw2:9.3f} {('VACUOUS' if vac else '-'):>9s}")
        print()
    report["part6_tau_choice"] = part6

    # full ladder for the designated cell
    hr("PART 6b -- full tau ladder, DESIGNATED cell (Granite r4) + all cells summary")
    print(f"{'tau':>5s} {'A4 mean':>8s} {'A4 SD':>7s} {'A4 n0':>6s} {'A1 mean':>8s} {'A1 n1':>6s} "
          f"{'gap':>7s} {'d/gap':>7s} {'sigma P-a':>10s} {'pow g=0':>8s} {'n80 g=0':>8s}")
    model, rate = DESIGNATED
    part6b = {}
    for tau in TAU_LADDER:
        a1 = arm_vector(per_rep, model, A1, rate, tau)
        a4 = arm_vector(per_rep, model, A4, rate, tau)
        sa = math.sqrt(2) * float(a4.std(ddof=1))
        gap = float(a1.mean() - a4.mean())
        pw0 = tost_power(sa, 5, DELTA, 0.0, trials=100_000, rng=rng)
        n80 = min_n_for_power(sa, DELTA, 0.0, 0.80, rng=rng)
        part6b[f"t{tau:g}"] = {"a4_mean": float(a4.mean()), "a4_sd": float(a4.std(ddof=1)),
                               "a4_n0": int((a4 == 0).sum()), "a1_mean": float(a1.mean()),
                               "a1_n1": int((a1 == 1.0).sum()), "gap": gap,
                               "sigma_pa": sa, "power_gap0": pw0, "n80": n80}
        print(f"{tau:5.0f} {a4.mean():8.4f} {a4.std(ddof=1):7.4f} {int((a4 == 0).sum()):6d} "
              f"{a1.mean():8.4f} {int((a1 == 1.0).sum()):6d} {gap:7.4f} "
              f"{DELTA / gap if gap else float('nan'):7.3f} {sa:10.4f} {pw0:8.3f} {str(n80):>8s}")
    report["part6b_designated_ladder"] = part6b

    (OUT / "g2_tau40_power.json").write_text(json.dumps(report, indent=1, default=str))
    print(f"\nwrote {OUT / 'g2_tau40_power.json'}")


if __name__ == "__main__":
    main()
