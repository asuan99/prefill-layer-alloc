#!/usr/bin/env python3
"""seq3 -- the larger-step sequential re-search that rev3 left open (§6-6).

GPU spend: 0.  Nothing here touches a GPU or an engine; it re-uses the design
inputs that `design_g13_stats.py` already computed from committed artifacts.

WHY THIS EXISTS
---------------
rev3's re-audit registered the FIXED plan
    M8  k=16 x m=3  (96 boots, 3.84 GPU-hr)
    Ha8 k=16 x m=6 (192 boots, 7.68 GPU-hr)      total 11.52 GPU-hr
at the `hi_chi2_upper` sigma_boot band point, and then wrote down, as open
item §6-6, that

    "a sequential design with a LARGER STEP SIZE has never been re-searched.
     After the death-cause-A repair its E[GPU-hr] is 1.68-1.80, i.e. far below
     the registered fixed plan.  Closing that avenue with stale numbers while
     asking for 11.52 GPU-hr would be unfair -- but any sequential/adaptive
     design MUST re-simulate its own coverage, because under a data-dependent
     (k, m) the F pivot is no longer an F."

This script does exactly that and nothing else.  It does NOT close gate #13
(§5-0/§5-1 of rev3 are inherited verbatim), it does NOT change any estimand,
and it makes no performance claim.

WHAT IS PRE-SPECIFIED HERE, BEFORE LOOKING AT ANY OUTPUT
--------------------------------------------------------
1. Decision rule per stage = rev3's, verbatim (exact-F pivot, each bound cut at
   ITS OWN degeneracy point `MS_B <= F_.05 * MS_W`).
2. Stopping rule, fixed pre-data: stop at stage 1 iff the stage-1 estimate is
   non-degenerate AND already passes.  A degenerate draw never stops the design.
3. Acceptance = design_g13_stats.ACCEPT_P_PASS / ACCEPT_COVERAGE, required at
   the WORST sigma_job prior of the registered band, evaluated at the
   REGISTERED sigma_boot band point.  Other band points are sensitivity only.
4. The reported coverage is the SEQUENTIAL procedure's own coverage (self-
   coverage), not a stage's.  A design that buys cheapness with coverage < 0.93
   is rejected here, not reported as a saving.
5. Winner's curse: the sweep's argmin is re-evaluated at a FRESH seed and full
   MC_N before it is quoted.
6. A sequential design must be affordable at its MAXIMUM, not at its mean, so
   both E[GPU-hr] and max GPU-hr are reported and the comparison against the
   registered fixed plan is made on BOTH.

FALSIFIABLE SELF-CHECKS (methodology gate #9/#50 -- every check below must be
able to fail, and SC1/SC3 are mutation tests that MUST fail on a mutated copy):
  SC1  guard mutation: reverting the stage-1 guard to the rev2 threshold
       (`MS_B <= MS_W`) must MEASURABLY change early stopping and expected cost
       at a named cell.  Direction, stated before the run and CORRECTED after
       it: the rev2 threshold is STRICTER, not looser (F_.05 < 1), so it
       *suppresses* stopping -- the first draft of this check predicted the
       opposite and failed, which is the check doing its job.
  SC2  k2=0 reduction: with no second stage, seq3 must reproduce the AUDITED
       fixed-design machinery `design_g13_stats._oc` (p_pass, coverage) within
       MC error at a named cell.  Catches a mis-implemented pooling/df path.
  SC3  stopping-rule mutation: letting stage 1 stop on ANY pass, degenerate
       included, must MEASURABLY depress sequential coverage (a degenerate stop
       reports UB=0, which cannot cover a positive sigma_job).
  SC4  seq2 cross-check: at (k1,k2,m)=(4,4,3), mid sigma_boot, seq3 must
       reproduce `design_g13_stats.seq2()` within MC error on a DIFFERENT seed.

★HONESTY NOTE ON SC1/SC3 (gate #9, fourteenth-recurrence guard).  The SIGN of
both mutation deltas is structural, not evidence: with UB unchanged, the rev2
guard's accept set is a strict SUBSET of rev3's, and a degenerate stop reports
UB=0 which never covers.  So "the sign came out as predicted" would be an
identity.  What these checks actually test is falsifiable and was NOT knowable
in advance: (i) the mutation handle is live at all, and (ii) the effect is
larger than 5 x MC SE.  A dead handle or an immaterial effect fails them.
Materiality against a 5-percentage-point floor is reported SEPARATELY as a
measurement, never folded into `all_pass`.
Note on what is NOT a check: "E[boots] >= k1*m*2" is an identity (every path
pays stage 1) and is deliberately excluded from the pass set -- rev3's B2
finding (gate #9's thirteenth recurrence) was exactly this failure mode.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _load_g13():
    spec = importlib.util.spec_from_file_location(
        "design_g13_stats", os.path.join(HERE, "design_g13_stats.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


G = _load_g13()

SEC_PER_BOOT_60S = 144.0        # rev3 §6-8: harness-layer confirmation still owed
SWEEP_N = 4000                  # sweep resolution; winner re-run at G.MC_N
SEED_SWEEP = G.MC_SEED + 3100
SEED_CONFIRM = G.MC_SEED + 3191   # fresh seed, winner's-curse guard

# Pre-registered grid.  k1 is the first commitment (what is actually spent if
# the design stops early); k2 is the escalation.  m is boot-pairs per job.
GRID_K1 = (4, 6, 8, 10, 12)
GRID_K2 = (2, 4, 6, 8, 10, 12)
GRID_M = (3, 6, 9)


def _gpu_hr(boots: float) -> float:
    return boots * SEC_PER_BOOT_60S / 3600.0


def _df_eff(df_b: float, df_w: float, m: int, sigma_boot: float,
            sj_reg: float) -> float:
    """Satterthwaite effective df, verbatim from design_g13_stats._oc.

    Generalised from (k, m) to (df_B, df_W) so the POOLED stage of a sequential
    design can use it: the pooled design has df_B = (k1-1)+(k2-1), which is one
    less than a single k1+k2 run.
    """
    return max(1.0, 2 * sj_reg ** 4 / ((2.0 / m ** 2) * (
        (m * sj_reg ** 2 + sigma_boot ** 2) ** 2 / df_b
        + sigma_boot ** 4 / df_w)))


def _sat_mult(df_b: float, df_w: float, m: int, sigma_boot: float,
              sj_reg: float) -> float:
    de = _df_eff(df_b, df_w, m, sigma_boot, sj_reg)
    return math.sqrt(de / G._chi2_lower(0.05, de))


def _ub_sat(msb: float, msw: float, m: int, mult: float) -> float:
    return math.sqrt(max(0.0, (msb - msw) / m)) * mult


def _seq_oc(k1: int, k2: int, m: int, sigma_boot: float, sigma_job: float,
            gap: float, n: int, seed: int,
            bound: str = "exact_F_pivot",
            mutate_stage1_guard: bool = False,
            mutate_stop_on_any_pass: bool = False) -> dict:
    """Monte-Carlo operating characteristic of the 2-stage sequential design.

    `bound` is the registered bound-selection rule's OUTPUT, not a free knob --
    see `_seq_eval_plan`, which applies design_g13_stats._eval_plan's rule
    (Satterthwaite iff its simulated coverage >= 0.93 at every band prior,
    else the exact-F pivot) to the SEQUENTIAL procedure's own coverage.
    Each bound is guarded at ITS OWN degeneracy point (rev3 death cause A):
    Satterthwaite at MS_B <= MS_W, exact-F at MS_B <= F_.05 * MS_W.

    `mutate_*` are used ONLY by the self-checks; the registered rule has both
    False.  They are the mutation-test handles required by gate #50.
    """
    rnd = random.Random(seed + 1000 * k1 + 37 * k2 + m)
    ev = m * sigma_job ** 2 + sigma_boot ** 2
    sat = bound == "satterthwaite_registered"

    df_b1, df_w1 = k1 - 1, k1 * (m - 1)
    fq1 = G._f_quantile(0.05, df_b1, df_w1)
    cw1 = G._chi2_lower(0.05, df_w1)
    mult1 = _sat_mult(df_b1, df_w1, m, sigma_boot, G.REGISTERED_SAT_PRIOR)
    two_stage = k2 > 0
    if two_stage:
        # pooled design loses one df_B: each stage estimates its own mean.
        df_b2 = (k1 - 1) + (k2 - 1)
        df_w2 = k1 * (m - 1) + k2 * (m - 1)
        fq2 = G._f_quantile(0.05, df_b2, df_w2)
        cw2 = G._chi2_lower(0.05, df_w2)
        mult2 = _sat_mult(df_b2, df_w2, m, sigma_boot, G.REGISTERED_SAT_PRIOR)

    stop1 = passes = covers = undet = 0
    boots_total = 0.0
    for _ in range(n):
        msb1 = ev * rnd.gammavariate(df_b1 / 2.0, 2.0) / df_b1
        msw1 = sigma_boot ** 2 * rnd.gammavariate(df_w1 / 2.0, 2.0) / df_w1
        ub1 = (_ub_sat(msb1, msw1, m, mult1) if sat
               else G._ub_exact_f(msb1, msw1, m, fq1, cw1, df_w1))
        # each bound's own degeneracy point; the mutation handle forces rev2's.
        deg1 = msw1 if (sat or mutate_stage1_guard) else fq1 * msw1
        thr1 = deg1
        nondeg1 = msb1 > thr1
        stops = (gap >= 3 * ub1) and (mutate_stop_on_any_pass or nondeg1)
        if stops or not two_stage:
            boots_total += k1 * m * 2
            if not two_stage and not nondeg1:
                undet += 1
            else:
                if stops:
                    stop1 += 1
                    passes += 1
                elif gap >= 3 * ub1:
                    passes += 1
            covers += (ub1 >= sigma_job)
            continue
        msb2 = ev * rnd.gammavariate((k2 - 1) / 2.0, 2.0) / (k2 - 1)
        msw2 = sigma_boot ** 2 * rnd.gammavariate(k2 * (m - 1) / 2.0, 2.0) / (k2 * (m - 1))
        msb = ((k1 - 1) * msb1 + (k2 - 1) * msb2) / df_b2
        msw = (k1 * (m - 1) * msw1 + k2 * (m - 1) * msw2) / df_w2
        ub = (_ub_sat(msb, msw, m, mult2) if sat
              else G._ub_exact_f(msb, msw, m, fq2, cw2, df_w2))
        boots_total += (k1 + k2) * m * 2
        if msb <= (msw if sat else fq2 * msw):
            undet += 1
        elif gap >= 3 * ub:
            passes += 1
        covers += (ub >= sigma_job)
    return {
        "p_stop_at_stage1": stop1 / n,
        "p_pass_overall": passes / n,
        "coverage_sequential": covers / n,
        "p_undetermined": undet / n,
        "expected_boots": boots_total / n,
        "expected_gpu_hr": _gpu_hr(boots_total / n),
        "max_boots": (k1 + k2) * m * 2 if two_stage else k1 * m * 2,
        "max_gpu_hr": _gpu_hr(((k1 + k2) if two_stage else k1) * m * 2),
        "mc_se_p_pass": math.sqrt(max(passes / n * (1 - passes / n), 1e-12) / n),
        "mc_se_coverage": math.sqrt(max(covers / n * (1 - covers / n), 1e-12) / n),
        "n_mc": n,
    }


def _worst_over_priors(k1, k2, m, sb, priors, gap, n, seed,
                       bound="exact_F_pivot", **mut) -> dict:
    ocs = [(sj, _seq_oc(k1, k2, m, sb, sj, gap, n, seed, bound=bound, **mut))
           for sj in priors]
    worst_pass = min(o["p_pass_overall"] for _, o in ocs)
    worst_cov = min(o["coverage_sequential"] for _, o in ocs)
    # E[cost] is prior-dependent (a design stops early more often when the
    # truth is easy); the reservation figure is the max over the band.
    worst_cost = max(o["expected_gpu_hr"] for _, o in ocs)
    return {
        "bound": bound,
        "worst_p_pass": worst_pass,
        "worst_coverage": worst_cov,
        "worst_expected_gpu_hr": worst_cost,
        "max_gpu_hr": ocs[0][1]["max_gpu_hr"],
        "max_p_undetermined": max(o["p_undetermined"] for _, o in ocs),
        "min_p_stop_at_stage1": min(o["p_stop_at_stage1"] for _, o in ocs),
        "ACCEPT": (worst_pass >= G.ACCEPT_P_PASS and worst_cov >= G.ACCEPT_COVERAGE),
        "per_prior": {("%.4f" % sj): o for sj, o in ocs},
    }


def _seq_eval_plan(k1, k2, m, sb, priors, gap, n, seed) -> dict:
    """design_g13_stats._eval_plan's registered rule, applied to the SEQUENTIAL
    procedure's own coverage.

    This is the fix for a confound the first draft of this script had: it forced
    both arms onto the exact-F pivot, while the registered FIXED plan puts M8 on
    Satterthwaite.  Comparing a Satterthwaite-bounded fixed plan against an
    exact-F-bounded sequential one measures the BOUND, not sequentiality.
    """
    sat = _worst_over_priors(k1, k2, m, sb, priors, gap, n, seed,
                             bound="satterthwaite_registered")
    if sat["worst_coverage"] >= G.ACCEPT_COVERAGE:
        return sat
    return _worst_over_priors(k1, k2, m, sb, priors, gap, n, seed,
                              bound="exact_F_pivot")


def sweep() -> dict:
    ps = G.pairsd()
    band = G.priorband()
    gaps = G.bslope()["total_gaps_to_explain"]
    sbb = G.sigma_boot_band()
    priors = sorted(v["pct"] for v in band.values()
                    if isinstance(v, dict) and v.get("pct") is not None)
    reg_point = G.REGISTERED_SIGMA_BOOT_BAND_POINT
    out = {
        "what": "larger-step sequential re-search (rev3 open item 6-6)",
        "gpu_spend": 0,
        "registered_sigma_boot_band_point": reg_point,
        "sigma_job_prior_band_pct": priors,
        "accept": {"p_pass_min": G.ACCEPT_P_PASS, "coverage_min": G.ACCEPT_COVERAGE},
        "sec_per_boot_60s": SEC_PER_BOOT_60S,
        "fixed_reference_plan": {
            "M8": {"k": G.REGISTERED_PLAN_M8[0], "m": G.REGISTERED_PLAN_M8[1],
                   "gpu_hr": _gpu_hr(G.REGISTERED_PLAN_M8[0] * G.REGISTERED_PLAN_M8[1] * 2)},
            "Ha8": {"k": G.REGISTERED_PLAN_HA8[0], "m": G.REGISTERED_PLAN_HA8[1],
                    "gpu_hr": _gpu_hr(G.REGISTERED_PLAN_HA8[0] * G.REGISTERED_PLAN_HA8[1] * 2)},
            "total_gpu_hr": _gpu_hr(G.REGISTERED_PLAN_M8[0] * G.REGISTERED_PLAN_M8[1] * 2
                                    + G.REGISTERED_PLAN_HA8[0] * G.REGISTERED_PLAN_HA8[1] * 2),
        },
        "arms": {},
    }
    for arm in ("M8", "Ha8"):
        gap = gaps[arm]["pct_gap"]
        sb_band = sbb["arms"][arm]["BAND_PCT"]
        sb_reg = sb_band[reg_point]
        rows = []
        for k1 in GRID_K1:
            for k2 in GRID_K2:
                for m in GRID_M:
                    r = _seq_eval_plan(k1, k2, m, sb_reg, priors, gap,
                                       SWEEP_N, SEED_SWEEP)
                    rows.append({"k1": k1, "k2": k2, "m": m, "bound": r["bound"],
                                 "worst_p_pass": r["worst_p_pass"],
                                 "worst_coverage": r["worst_coverage"],
                                 "worst_expected_gpu_hr": r["worst_expected_gpu_hr"],
                                 "max_gpu_hr": r["max_gpu_hr"],
                                 "max_p_undetermined": r["max_p_undetermined"],
                                 "min_p_stop_at_stage1": r["min_p_stop_at_stage1"],
                                 "ACCEPT": r["ACCEPT"]})
        feasible = [r for r in rows if r["ACCEPT"]]
        arm_out = {
            "sigma_boot_registered_pct": sb_reg,
            "sigma_boot_band_pct": sb_band,
            "gap_pct": gap,
            "n_grid": len(rows),
            "n_feasible": len(feasible),
            "all_rows": rows,
        }
        if feasible:
            # two argmins: by expected cost and by RESERVATION (max) cost.
            by_exp = min(feasible, key=lambda r: (r["worst_expected_gpu_hr"], r["max_gpu_hr"]))
            by_max = min(feasible, key=lambda r: (r["max_gpu_hr"], r["worst_expected_gpu_hr"]))
            conf = {}
            for label, cand in (("argmin_expected", by_exp), ("argmin_reservation", by_max)):
                full = _worst_over_priors(cand["k1"], cand["k2"], cand["m"], sb_reg,
                                          priors, gap, G.MC_N, SEED_CONFIRM,
                                          bound=cand["bound"])
                conf[label] = {
                    "plan": {"k1": cand["k1"], "k2": cand["k2"], "m": cand["m"]},
                    "sweep": cand,
                    "confirmation_fresh_seed": {
                        "bound": full["bound"],
                        "worst_p_pass": full["worst_p_pass"],
                        "worst_coverage": full["worst_coverage"],
                        "worst_expected_gpu_hr": full["worst_expected_gpu_hr"],
                        "max_gpu_hr": full["max_gpu_hr"],
                        "max_p_undetermined": full["max_p_undetermined"],
                        "min_p_stop_at_stage1": full["min_p_stop_at_stage1"],
                        "ACCEPT": full["ACCEPT"],
                        "n_mc": G.MC_N, "seed": SEED_CONFIRM,
                    },
                    "survives_confirmation": full["ACCEPT"],
                }
            arm_out["confirmed"] = conf
        # sensitivity: the same argmin evaluated at the other band points
        if feasible and arm_out["confirmed"]["argmin_expected"]["survives_confirmation"]:
            p = arm_out["confirmed"]["argmin_expected"]["plan"]
            arm_out["sensitivity_other_band_points"] = {
                name: _seq_eval_plan(p["k1"], p["k2"], p["m"], val, priors, gap,
                                     SWEEP_N, SEED_CONFIRM)["ACCEPT"]
                for name, val in sb_band.items()
            }
        out["arms"][arm] = arm_out
    return out


# ---------------------------------------------------------------- self-checks
def selfcheck() -> dict:
    """Every check must be able to fail.  SC1/SC3 are mutation tests."""
    ps = G.pairsd()
    band = G.priorband()
    gaps = G.bslope()["total_gaps_to_explain"]
    sbb = G.sigma_boot_band()
    priors = sorted(v["pct"] for v in band.values()
                    if isinstance(v, dict) and v.get("pct") is not None)
    checks = []

    # SC1 -- guard mutation at a named cell.
    arm, k1, k2, m = "Ha8", 8, 8, 6
    sb = sbb["arms"][arm]["BAND_PCT"][G.REGISTERED_SIGMA_BOOT_BAND_POINT]
    gap = gaps[arm]["pct_gap"]
    sj = max(priors)
    base = _seq_oc(k1, k2, m, sb, sj, gap, G.MC_N, SEED_CONFIRM)
    mut = _seq_oc(k1, k2, m, sb, sj, gap, G.MC_N, SEED_CONFIRM,
                  mutate_stage1_guard=True)
    d_stop = mut["p_stop_at_stage1"] - base["p_stop_at_stage1"]
    d_cost = mut["expected_gpu_hr"] - base["expected_gpu_hr"]
    checks.append({
        "id": "SC1", "kind": "mutation",
        "what": ("rev2 guard (MS_B<=MS_W) at stage 1 must MEASURABLY change early "
                 "stopping; it is the STRICTER threshold (F_.05 < 1), so it "
                 "suppresses stopping and raises cost"),
        "sign_is_structural": ("rev2's accept set is a strict subset of rev3's, so "
                               "the sign is an identity; the magnitude is not"),
        "cell": f"{arm} k1={k1} k2={k2} m={m} sigma_job={sj:.4f}",
        "F_05_stage1": G._f_quantile(0.05, k1 - 1, k1 * (m - 1)),
        "base_p_stop": base["p_stop_at_stage1"], "mutant_p_stop": mut["p_stop_at_stage1"],
        "delta_p_stop": d_stop, "delta_expected_gpu_hr": d_cost,
        "threshold": "|delta_p_stop| > 5 * mc_se  (handle liveness)",
        "mc_se": base["mc_se_p_pass"],
        "ok": abs(d_stop) > 5 * max(base["mc_se_p_pass"], 1e-9),
        "material_at_5pp": abs(d_stop) >= 0.05,
    })

    # SC2 -- k2=0 reduction must reproduce the AUDITED fixed-design machinery.
    k, mm = 12, 3
    red = _seq_oc(k, 0, mm, sb, sj, gap, G.MC_N, SEED_CONFIRM)
    ref = G._oc(k, mm, sb, sj, gap, n=G.MC_N, seed=SEED_CONFIRM)
    dp = abs(red["p_pass_overall"] - ref["p_pass"])
    dc = abs(red["coverage_sequential"] - ref["coverage"])
    tol = 5 * math.sqrt(0.25 / G.MC_N)
    checks.append({
        "id": "SC2", "kind": "cross-implementation",
        "what": "k2=0 must reproduce design_g13_stats._oc (p_pass, coverage)",
        "cell": f"{arm} k={k} m={mm} sigma_job={sj:.4f}",
        "seq3": [red["p_pass_overall"], red["coverage_sequential"]],
        "audited_oc": [ref["p_pass"], ref["coverage"]],
        "abs_diff": [dp, dc], "tolerance_5se": tol,
        "ok": dp <= tol and dc <= tol,
    })

    # SC3 -- stopping-rule mutation must inflate P(pass).
    mut3 = _seq_oc(k1, k2, m, sb, sj, gap, G.MC_N, SEED_CONFIRM,
                   mutate_stop_on_any_pass=True)
    d_pass = mut3["p_pass_overall"] - base["p_pass_overall"]
    d_cov = mut3["coverage_sequential"] - base["coverage_sequential"]
    checks.append({
        "id": "SC3", "kind": "mutation",
        "what": ("stopping on ANY stage-1 pass (degenerate included) must MEASURABLY "
                 "depress sequential coverage -- a degenerate stop reports UB=0"),
        "sign_is_structural": ("UB=0 never covers a positive sigma_job, so the sign is "
                               "an identity; the magnitude is not"),
        "cell": f"{arm} k1={k1} k2={k2} m={m} sigma_job={sj:.4f}",
        "base_coverage": base["coverage_sequential"],
        "mutant_coverage": mut3["coverage_sequential"],
        "delta_coverage": d_cov,
        "delta_p_pass": d_pass,
        "threshold": "|delta_coverage| > 5 * mc_se_coverage  (handle liveness)",
        "mc_se_coverage": base["mc_se_coverage"],
        "ok": abs(d_cov) > 5 * max(base["mc_se_coverage"], 1e-9),
        "material_at_1pp": abs(d_cov) >= 0.01,
        "note_p_pass": ("P(pass) moves only +0.3pp because escalation recovers most "
                        "degenerate draws anyway -- the damage lands on coverage, "
                        "not on power"),
    })

    # SC4 -- reproduce seq2's rev3 numbers on a different seed.
    s2 = G.seq2(4, 4, 3, n=G.MC_N)
    row = [r for r in s2["rows"]
           if r["arm"] == "Ha8" and abs(r["sigma_job_prior_pct"] - max(priors)) < 1e-9][0]
    mine = _seq_oc(4, 4, 3, ps["arms"]["Ha8"]["sigma_boot_paired_pct"], max(priors),
                   gaps["Ha8"]["pct_gap"], G.MC_N, SEED_CONFIRM)
    d4 = abs(mine["p_pass_overall"] - row["p_pass_overall"])
    checks.append({
        "id": "SC4", "kind": "cross-implementation",
        "what": "seq3 must reproduce seq2 at (4,4,3), mid sigma_boot, different seed",
        "seq2_p_pass": row["p_pass_overall"], "seq3_p_pass": mine["p_pass_overall"],
        "abs_diff": d4, "tolerance_5se": tol, "ok": d4 <= tol,
    })

    # SC5 -- the NEW Satterthwaite branch must reproduce the audited machinery.
    # rev3's PC6 finding was that a function carrying 42% of the budget had zero
    # controls; adding a bound branch with zero controls repeats it.
    k5, m5 = 12, 3
    sb5 = sbb["arms"]["M8"]["BAND_PCT"][G.REGISTERED_SIGMA_BOOT_BAND_POINT]
    gap5, sj5 = gaps["M8"]["pct_gap"], max(priors)
    red5 = _seq_oc(k5, 0, m5, sb5, sj5, gap5, G.MC_N, SEED_CONFIRM,
                   bound="satterthwaite_registered")
    ref5 = G._oc(k5, m5, sb5, sj5, gap5, n=G.MC_N, seed=SEED_CONFIRM,
                 sat_prior=G.REGISTERED_SAT_PRIOR)["rev1_satterthwaite"]
    dp5 = abs(red5["p_pass_overall"] - ref5["p_pass_guarded"])
    dc5 = abs(red5["coverage_sequential"] - ref5["coverage"])
    checks.append({
        "id": "SC5", "kind": "cross-implementation",
        "what": ("k2=0 on the Satterthwaite branch must reproduce "
                 "design_g13_stats._oc's rev1_satterthwaite (guarded p_pass, coverage)"),
        "cell": f"M8 k={k5} m={m5} sigma_job={sj5:.4f}",
        "seq3": [red5["p_pass_overall"], red5["coverage_sequential"]],
        "audited_oc": [ref5["p_pass_guarded"], ref5["coverage"]],
        "abs_diff": [dp5, dc5], "tolerance_5se": tol,
        "ok": dp5 <= tol and dc5 <= tol,
    })

    return {"checks": checks, "all_pass": all(c["ok"] for c in checks),
            "n_checks": len(checks),
            "excluded_as_identity": [
                "E[boots] >= k1*m*2 (every path pays stage 1 -- unfalsifiable)",
                "max_gpu_hr == (k1+k2)*m*2*sec/3600 (definition, not a check)"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sweep", "selfcheck", "all"])
    ap.add_argument("--out")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.cmd == "sweep":
        res = sweep()
    elif a.cmd == "selfcheck":
        res = selfcheck()
    else:
        sc = selfcheck()
        res = {"selfcheck": sc, "sweep": sweep()}
        # gate #42/#44: the verdict must reference every check and must not
        # label its own failure a success.
        res["SELFCHECK"] = "PASS" if sc["all_pass"] else "FAIL"
        if not sc["all_pass"]:
            res["FAILED_CHECKS"] = [c for c in sc["checks"] if not c["ok"]]
    txt = json.dumps(res, indent=2, sort_keys=False, default=str)
    if a.out:
        dest = os.path.join(HERE, a.out)
        if os.path.exists(dest) and not a.force:
            sys.stderr.write("REFUSING to overwrite an existing artifact: %s\n" % dest)
            raise SystemExit(2)
        open(dest, "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
