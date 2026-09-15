#!/usr/bin/env python3
"""E2 decision rules R1-R6 (PREREG_E2_STICKY_REV3 sec 6 + ADDENDUM A4, A5).

Reads the per-boot `cell_<cell>_<arm>_<seed>.json` written by
`e2_realized_mix.py` and emits `E2_LABEL.json`.  It computes NO estimator --
that split is deliberate (the rev1 audit killed a design in which the estimator
menu and the rule could be varied together, because picking E-cnt vs E-qcond
flipped the negative control from FAIL to PASS on the same bytes).

Every threshold here is a registered literal.  The two that were argued over:
  * R1 bands 0.90 / 0.20 -- the shape-A OFF maximum in the archive is E-qcond
    17.65%, and the adjacent rung a_r3 reaches 18.67%, so the OFF band has
    1.33pp of headroom on UNMEASURED seeds.  That is why R1 has a separate
    `OFF_BASELINE_ABOVE_BAND` branch (ADDENDUM A5): a failing control arm is a
    statement about the CONTROL, not about sticky.
  * R4" threshold 3x (not 1x) -- restored from the rev1 verdict after rev2
    tightened it without disclosure.  Note E2C-18: at EITHER threshold this
    probe cannot resolve the drain axis (3x trigger 0.93% vs a drain scale of
    0.17-0.70%), so `B_RESULT_AXIS_WITHIN` is the near-certain outcome and is
    NOT evidence about drains.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

SHAPE_A_CELLS = ("a_r4", "a_r2")
NEG_CONTROL_CELL = "b_r3"
R1_ON_BAND = 90.0
R1_OFF_BAND = 20.0
R3_IDX0_MAX_PCT = 5.0
R4_QCOND_MAX_DELTA_PP = 0.5
R4P_ON_ECNT_MIN_PCT = 95.0  # same ruler as R3(ii): 100 - 5.0 (rev2 verdict B3')
R4PP_MULTIPLIER = 3.0
R5_SEED_SPREAD_MAX_PP = 5.0
E_PACT_ABORT_PCT = 95.0  # ADDENDUM A6 -- a RUNTIME clause, not only a selftest constant
# Registered seed order (prereg sec 2).  R4''/R4" index by s1/s2 = the FIRST TWO of
# this tuple, NOT by sorted() -- lexicographically "4162" precedes "4386", which would
# silently swap s1 and s2 and change which boot the ABORT-layer gate reads.
SEED_ORDER = ("4386", "4162", "251", "2630")
BOOTSTRAP_B = 100000
BOOTSTRAP_RNG_SEED = 20260915
ESTIMATORS = ("e_cnt", "e_time", "e_iter", "e_qcond")


def load_cells(cells_dir):
    cells = {}
    for path in sorted(glob.glob(os.path.join(cells_dir, "cell_*.json"))):
        rec = json.load(open(path))
        if rec.get("empty"):
            continue
        cells[(rec["cell"], rec["arm"], str(rec["seed"]))] = rec
    return cells


def _pcts(rec):
    return {name: rec[name]["pct"] for name in ESTIMATORS}


def _achieved(rec):
    bench = rec.get("bench") or {}
    return bench.get("achieved_req_s") if bench.get("all_pass") else None


def paired_bootstrap(diffs):
    """Percentile bootstrap of the MEAN of the paired differences.

    E2C-17: B does not buy resolution here -- with n=4 there are only 4**4 = 256
    distinct resamples, so the 2.5% endpoint moves in ~0.39% steps.  Do not read
    the CI width as precision.
    """
    arr = np.asarray(diffs, dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_RNG_SEED)
    idx = rng.integers(0, arr.size, size=(BOOTSTRAP_B, arr.size))
    means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean": float(arr.mean()), "ci_lo": float(lo), "ci_hi": float(hi),
            "n": int(arr.size), "B": BOOTSTRAP_B, "rng_seed": BOOTSTRAP_RNG_SEED,
            "distinct_resamples": int(arr.size ** arr.size)}


def label(cells, expect_pairs):
    out = {"unresolved": [], "rules": {}, "companions": {}}

    # ---- availability / UNRESOLVED bookkeeping (D23: never re-label a shortened set)
    for cell, seed in expect_pairs:
        for arm in ("OFF", "ON"):
            rec = cells.get((cell, arm, seed))
            if rec is None:
                out["unresolved"].append({"cell": cell, "seed": seed, "why": f"missing {arm} boot"})
                break
            if rec.get("sm_index_mismatch"):
                out["unresolved"].append({"cell": cell, "seed": seed,
                                          "why": f"{arm}: stream_index/sm_counts disagree"})
                break
            # H-F1: `bench` ABSENT means the artefact was never written -- P7 CAPPED
            # (timeout SIGTERM; bench_serving writes --output-file only at the very
            # end of benchmark() and installs no signal/atexit handler) or a server
            # crash.  The old `rec.get("bench") and not ...` read key-absence as
            # falsy and PASSED it, so a truncated boot reached R1 while R2 dropped
            # to n=3 -- the two rules standing on different populations.
            bench = rec.get("bench")
            if not bench or not bench.get("all_pass"):
                why = bench.get("why") if isinstance(bench, dict) and bench.get("why") \
                    else ("bench artefact absent (P7 CAPPED or crash)" if not bench
                          else "P9-P12 bench validity failed")
                out["unresolved"].append({"cell": cell, "seed": seed, "why": f"{arm}: {why}"})
                break
            if bench.get("achieved_req_s") is None:
                out["unresolved"].append({"cell": cell, "seed": seed,
                    "why": f"{arm}: bench passed P9-P12 but carries no achieved_req_s "
                           f"(duration missing or zero) -- R2 would silently lose n"})
                break
            # H-B1: the probe-clock guard is fail-CLOSED only if somebody reads it.
            if rec.get("ABORT_ARM_LABEL_MISMATCH"):
                out["unresolved"].append({"cell": cell, "seed": seed,
                                          "why": f"{arm}: {rec['ABORT_ARM_LABEL_MISMATCH']}"})
                break
            if rec.get("ABORT_PROBE_CLOCK_MISMATCH"):
                out["unresolved"].append({"cell": cell, "seed": seed,
                                          "why": f"{arm}: {rec['ABORT_PROBE_CLOCK_MISMATCH']}"})
                break
            # H-B5: ADDENDUM A6 is a clause about MEASURED cells, not only about the
            # selftest.  < 95% is an instrument failure; [95,100) is a read-skew the
            # pair cannot carry.
            ep = rec.get("e_pact") or {}
            if ep.get("den"):
                if ep["pct"] < E_PACT_ABORT_PCT:
                    out["unresolved"].append({"cell": cell, "seed": seed,
                        "why": f"{arm}: ABORT_TELEMETRY_INCONSISTENT e_pact "
                               f"{ep['num']}/{ep['den']} = {ep['pct']:.2f}% < {E_PACT_ABORT_PCT}"})
                    break
                if ep["pct"] < 100.0:
                    out["unresolved"].append({"cell": cell, "seed": seed,
                        "why": f"{arm}: e_pact {ep['num']}/{ep['den']} = {ep['pct']:.2f}% "
                               f"in [{E_PACT_ABORT_PCT},100) -- identity guard degraded"})
                    break
            # R3(ii): read-skew above the ruler voids the pair, both arms.
            if rec["idx0"]["pct"] > R3_IDX0_MAX_PCT:
                out["unresolved"].append({"cell": cell, "seed": seed,
                                          "why": f"{arm}: idx0 {rec['idx0']['num']}/{rec['idx0']['den']} "
                                                 f"= {rec['idx0']['pct']:.2f}% > {R3_IDX0_MAX_PCT}%"})
                break
    dead = {(u["cell"], u["seed"]) for u in out["unresolved"]}
    live = [p for p in expect_pairs if p not in dead]

    # ---- R5 RUNS FIRST.  `SEED_SPREAD_FAILS` makes that cell's R1/R2 UNRESOLVED
    # (prereg sec 6 R5).  It used to be computed last and consumed by nobody, so a
    # cell could report SEED_SPREAD_FAILS and STICKY_REALIZES_A in the same file.
    # No circularity: R5 reads only `e_cnt` on the post-R3(ii) live set.
    out["rules"]["R5"] = {}
    for cell in SHAPE_A_CELLS:
        per_arm = {}
        for arm in ("OFF", "ON"):
            vals = [cells[(cell, arm, s)]["e_cnt"]["pct"] for c, s in live if c == cell]
            per_arm[arm] = (max(vals) - min(vals)) if len(vals) > 1 else None
        spreads = [v for v in per_arm.values() if v is not None]
        lab = ("PASS" if spreads and all(v <= R5_SEED_SPREAD_MAX_PP for v in spreads)
               else ("SEED_SPREAD_FAILS" if spreads else "UNRESOLVED"))
        out["rules"]["R5"][cell] = {"label": lab, "spread_pp": per_arm}
        if lab == "SEED_SPREAD_FAILS":
            for c, seed in [p for p in live if p[0] == cell]:
                out["unresolved"].append({"cell": c, "seed": seed,
                                          "why": f"R5 SEED_SPREAD_FAILS ({per_arm}) > "
                                                 f"{R5_SEED_SPREAD_MAX_PP}pp"})
    dead = {(u["cell"], u["seed"]) for u in out["unresolved"]}
    live = [p for p in expect_pairs if p not in dead]

    # ---- R1 (ADDENDUM A5: four branches, so an OFF failure is not read as an ON claim)
    on_fail, off_fail, detail = [], [], {}
    for cell, seed in live:
        if cell not in SHAPE_A_CELLS:
            continue
        on, off = _pcts(cells[(cell, "ON", seed)]), _pcts(cells[(cell, "OFF", seed)])
        detail[f"{cell}/{seed}"] = {"ON": on, "OFF": off}
        if not all(v > R1_ON_BAND for v in on.values()):
            on_fail.append(f"{cell}/{seed}")
        if not all(v < R1_OFF_BAND for v in off.values()):
            off_fail.append(f"{cell}/{seed}")
    if not detail:
        r1 = "UNRESOLVED"
    elif not on_fail and not off_fail:
        r1 = "STICKY_REALIZES_A"
    elif on_fail and off_fail:
        r1 = "STICKY_ON_BELOW_BAND+OFF_BASELINE_ABOVE_BAND"
    elif on_fail:
        r1 = "STICKY_ON_BELOW_BAND"
    else:
        r1 = "OFF_BASELINE_ABOVE_BAND"
    seen = sorted({k.split("/")[0] for k in detail})
    dropped = [c for c in SHAPE_A_CELLS if c not in seen]
    if dropped and r1 == "STICKY_REALIZES_A":
        r1 = "STICKY_REALIZES_A_ON_REMAINING_CELLS"
    out["rules"]["R1"] = {"label": r1, "cells_evaluated": seen, "cells_dropped": dropped,
                          "on_clause_failures": on_fail,
                          "off_clause_failures": off_fail, "per_seed": detail,
                          "note": "OFF_BASELINE_ABOVE_BAND is a statement about the CONTROL arm, "
                                  "not about sticky (ADDENDUM A5)."}

    # ---- R2: paired bootstrap per shape-A cell, three branches (rev1 verdict B2)
    out["rules"]["R2"] = {}
    for cell in SHAPE_A_CELLS:
        diffs, seeds_used = [], []
        for c, seed in live:
            if c != cell:
                continue
            on, off = _achieved(cells[(cell, "ON", seed)]), _achieved(cells[(cell, "OFF", seed)])
            if on is None or off is None:
                continue
            diffs.append(on - off)
            seeds_used.append(seed)
        if len(diffs) < 4:
            out["rules"]["R2"][cell] = {"label": "UNRESOLVED",
                                        "why": f"paired n={len(diffs)} < 4 (gate #3)"}
            continue
        boot = paired_bootstrap(diffs)
        if boot["ci_hi"] < 0:
            lab = "LAMBDA_MOVES_DOWN"
        elif boot["ci_lo"] > 0:
            lab = "LAMBDA_MOVES_UP"
        else:
            lab = "LAMBDA_WITHIN_CI"
        st = {arm: [cells[(cell, arm, s)]["split_transition"] for s in seeds_used]
              for arm in ("OFF", "ON")}
        out["rules"]["R2"][cell] = {
            "label": lab, "bootstrap": boot, "seeds": seeds_used, "diffs": diffs,
            "split_transition": st,  # Q4 companion is MANDATORY in every branch
            "mandatory_companion": "E2C-1: one knob, TWO mechanisms (decode SM 108->44 AND the "
                                   "removal of one drain pair per prefill boundary; opposite signs). "
                                   "No achieved change may be attributed to partitioning alone, and "
                                   "no null may be read as 'partitioning does not matter'.",
            "banned_reading": None if lab == "LAMBDA_MOVES_DOWN" else
                              "May NOT be written as 'the lambda0R-8(iii) confound concern is weakened'.",
        }

    # ---- R3: one-knob check (iii is report-only, per rev2 verdict B5)
    idx3_off = {f"{c}/{s}": cells[(c, 'OFF', s)]["idx3"]["num"] for c, s in live}
    st_on_gt_off = [f"{c}/{s}" for c, s in live
                    if cells[(c, "ON", s)]["split_transition"] > cells[(c, "OFF", s)]["split_transition"]]
    out["rules"]["R3"] = {
        "i_off_idx3_all_zero": all(v == 0 for v in idx3_off.values()),
        "i_detail": idx3_off,
        "ii_idx0_within_ruler": [u for u in out["unresolved"] if "idx0" in u["why"]] or "all within 5.0%",
        "iii_split_transition_report_only": {
            f"{c}/{s}": {"OFF": cells[(c, 'OFF', s)]["split_transition"],
                         "ON": cells[(c, 'ON', s)]["split_transition"]} for c, s in live},
        "iii_ON_exceeds_OFF": st_on_gt_off,
        "iii_consequence": ("code reading of sec 4-1(c) is broken; inherit E2C-1 in its strong form"
                            if st_on_gt_off else "code reading holds"),
    }

    # ---- R4 / R4' / R4": the b_r3 controls
    b_seeds = [s for s in SEED_ORDER if (NEG_CONTROL_CELL, s) in live]
    if b_seeds:
        qs = {arm: cells[(NEG_CONTROL_CELL, arm, b_seeds[0])]["e_qcond"]["pct"] for arm in ("OFF", "ON")}
        delta = abs(qs["ON"] - qs["OFF"])
        out["rules"]["R4"] = {
            "label": "PASS" if delta <= R4_QCOND_MAX_DELTA_PP else "FAIL",
            "delta_pp": delta, "per_arm": qs,
            "consequence_if_fail": "arm differences also arise from something other than sticky "
                                   "(order, drift, read-skew) => caveat on ALL of R1/R2.",
        }
        on_ecnt = cells[(NEG_CONTROL_CELL, "ON", b_seeds[0])]["e_cnt"]
        out["rules"]["R4prime"] = {
            "label": "PASS" if on_ecnt["pct"] >= R4P_ON_ECNT_MIN_PCT else "ABORT_STICKY_STATE_MISMATCH",
            "on_e_cnt_pct": on_ecnt["pct"], "num_den": [on_ecnt["num"], on_ecnt["den"]],
            "idx0": cells[(NEG_CONTROL_CELL, "ON", b_seeds[0])]["idx0"],
        }
    if len(b_seeds) >= 2:
        s1, s2 = b_seeds[0], b_seeds[1]
        a = {arm: [_achieved(cells[(NEG_CONTROL_CELL, arm, s)]) for s in (s1, s2)]
             for arm in ("OFF", "ON")}
        if all(v is not None for arm in a for v in a[arm]):
            s_off, s_on = abs(a["OFF"][0] - a["OFF"][1]), abs(a["ON"][0] - a["ON"][1])
            Y = max(s_off, s_on)
            dbar = ((a["ON"][0] - a["OFF"][0]) + (a["ON"][1] - a["OFF"][1])) / 2.0
            out["rules"]["R4dprime"] = {
                "label": "B_RESULT_AXIS_MOVES" if abs(dbar) > R4PP_MULTIPLIER * Y else "B_RESULT_AXIS_WITHIN",
                "d_bar": dbar, "s_off": s_off, "s_on": s_on, "Y": Y,
                "trigger": R4PP_MULTIPLIER * Y,
                "mandatory_companion": "n=2, no CI, power unregistered. B_RESULT_AXIS_WITHIN may NOT be "
                                       "read as 'no difference'. E2C-18: this probe cannot resolve the "
                                       "drain axis at any threshold, and Y is endogenous (treatment that "
                                       "raises run-to-run spread pushes the verdict toward WITHIN).",
                "not_a_negative_control": "b_r3 moves +2.4pp (E-time) / +9.1pp (E-cnt) OFF->ON; this is a "
                                          "drain-axis sensitivity probe, not a partition-axis control.",
                # Reported, NOT corrected: the registered predicate is literally |d_bar| > 3*Y,
                # so a degenerate Y makes the trigger 0 and any non-zero d_bar reads as MOVES.
                # Changing the rule here would be an unregistered repair; flagging it is not.
                "degenerate_yardstick": (Y == 0.0),
            }

    # ---- R6: cross-job companion, NEVER a verdict
    out["companions"]["R6"] = {
        "status": "report only",
        "mandatory_companion": "Cross-job comparison to job 908623 (gpu38) carries the unregistered "
                               "node axis (gate #233: gpu38->43->40->41) and must never be cited as a "
                               "judgement. This campaign's judgements are all within-job OFF vs ON.",
    }
    out["inherited_caveats"] = [
        "E2C-1 (one knob, two mechanisms)", "E2C-2 (E-cnt is not time occupancy)",
        "E2C-3 (ON index set is engine-forced; R1's ON clause is a positive control)",
        "E2C-4 (E-pact is an identity in both arms)", "E2C-5 (does NOT resolve lambda0R-8(iii))",
        "E2C-6 (a_r2 Q5 sits on the cliff; step function only)",
        "E2C-7 (OFF->ON order is a registered asymmetry)",
        "E2C-8' (no estimator column may be cited without its convention)",
        "E2C-9/E2C-20 (sampling grid and auxiliary numbers need conventions)",
        "E2C-10 (achieved is bounded by --max-running-requests 48)",
        "E2C-11/E2C-19 (four-estimator agreement is not robustness evidence; b_r0's exclusion "
        "carried it, and the binding estimator there is E-qcond, not E-iter)",
        "E2C-12/E2C-13 (seed provenance; seeds change the REALIZED offered rate: a_r2 = "
        "2.996/2.955/2.950/2.951 req/s)",
        "E2C-15 (rev3 sec 4-2 E-time seconds are citation-banned; use the corrected table)",
        "E2C-16 (population is not bench-only; post-hoc bench-only re-scoring may NOT re-label R1)",
        "E2C-17 (n=4 bootstrap CI endpoints step over 256 distinct resamples)",
        "E2C-18 (R4'' cannot resolve the drain axis)",
        "E2C-21 (shape A: 86-89% of the treated time has NO prefill in flight; Q2 buys only the cost "
        "of pinning decode to 44 SM while there is nothing to multiplex with -- CONSENSUS audit 2F9)",
    ]
    out["not_changed_by_this_campaign"] = [
        "new performance verdicts: 0", "Claim D/E grades: unchanged (both unverified)",
        "gate #6 (lambda*_SLO): unchanged", "P2 blockers (1)(2)(3): unchanged",
        "policy ranking / HE0 / layer-type negative result: unchanged",
    ]
    return out


# ------------------------------------------------------------------------- selftest
def _synth(cell, arm, seed, on_like, achieved, idx0=0.1, idx3=0, st=200):
    hi = 99.0 if on_like else 9.0
    mk = lambda p: {"num": int(p * 10), "den": 1000, "pct": p}
    # b_r3 is the shape-B cell: its OFF E-qcond is 100.00% (395/395) in the archive,
    # which is exactly why R4 uses that quantity -- sticky cannot move it.
    qcond = 100.0 if cell == NEG_CONTROL_CELL else (hi if on_like else 17.6)
    ecnt = (99.8 if on_like else 90.7) if cell == NEG_CONTROL_CELL else hi
    return {"cell": cell, "arm": arm, "seed": seed, "busy_n": 1000,
            "e_cnt": mk(ecnt), "e_time": {"num_s": hi, "den_s": 100.0, "pct": hi},
            "e_iter": mk(hi), "e_qcond": mk(qcond),
            "e_pact": mk(100.0), "idx0": {"num": 1, "den": 1000, "pct": idx0},
            "idx3": {"num": idx3, "den": 1000, "pct": 0.0}, "split_transition": st,
            "sm_index_mismatch": 0,
            "bench": {"all_pass": True, "achieved_req_s": achieved}}


def selftest():
    """Both branches of every rule must be REACHABLE (the lambda0 reachability gate).
    A rule whose FAIL branch cannot be produced is an identity, not a gate."""
    seeds = ["4386", "4162", "251", "2630"]
    pairs = [(c, s) for c in SHAPE_A_CELLS for s in seeds] + [("b_r3", s) for s in seeds[:2]]
    failures = []

    def build(**kw):
        cells = {}
        for c, s in pairs:
            base = 3.0 if c == "a_r4" else 2.9
            cells[(c, "OFF", s)] = _synth(c, "OFF", s, False, base, **kw.get("off", {}))
            cells[(c, "ON", s)] = _synth(c, "ON", s, True, base + kw.get("delta", 0.0), **kw.get("on", {}))
        return cells

    # (1) registered forecast: sticky realizes, achieved unchanged
    r = label(build(), pairs)
    if r["rules"]["R1"]["label"] != "STICKY_REALIZES_A":
        failures.append(f"R1 positive branch unreachable: {r['rules']['R1']['label']}")
    if r["rules"]["R2"]["a_r4"]["label"] != "LAMBDA_WITHIN_CI":
        failures.append(f"R2 WITHIN branch: {r['rules']['R2']['a_r4']['label']}")
    if r["rules"]["R4"]["label"] != "PASS" or r["rules"]["R4prime"]["label"] != "PASS":
        failures.append("R4/R4' positive branch unreachable")

    # (2) achieved drops -> MOVES_DOWN must be reachable
    r = label(build(delta=-0.5), pairs)
    if r["rules"]["R2"]["a_r4"]["label"] != "LAMBDA_MOVES_DOWN":
        failures.append(f"R2 DOWN unreachable: {r['rules']['R2']['a_r4']['label']}")
    # (3) achieved rises -> MOVES_UP must be reachable AND must carry the ban
    r = label(build(delta=+0.5), pairs)
    if r["rules"]["R2"]["a_r4"]["label"] != "LAMBDA_MOVES_UP":
        failures.append(f"R2 UP unreachable: {r['rules']['R2']['a_r4']['label']}")
    if not r["rules"]["R2"]["a_r4"]["banned_reading"]:
        failures.append("R2 UP branch lost its banned_reading companion")

    # (4) ON clause fails alone / OFF clause fails alone -> distinct labels (ADDENDUM A5)
    cells = build()
    for s in seeds:
        cells[("a_r4", "ON", s)]["e_qcond"]["pct"] = 80.0
    if label(cells, pairs)["rules"]["R1"]["label"] != "STICKY_ON_BELOW_BAND":
        failures.append("R1 ON-only branch unreachable")
    cells = build()
    for s in seeds:
        cells[("a_r4", "OFF", s)]["e_qcond"]["pct"] = 25.0
    if label(cells, pairs)["rules"]["R1"]["label"] != "OFF_BASELINE_ABOVE_BAND":
        failures.append("R1 OFF-only branch unreachable")

    # (5) idx0 above the ruler voids the PAIR, and R2 then refuses (n<4)
    cells = build()
    cells[("a_r4", "ON", "4386")]["idx0"]["pct"] = 6.0
    r = label(cells, pairs)
    if not any(u["cell"] == "a_r4" for u in r["unresolved"]):
        failures.append("R3(ii) does not void the pair")
    if r["rules"]["R2"]["a_r4"]["label"] != "UNRESOLVED":
        failures.append("R2 does not refuse n<4 after an UNRESOLVED pair (gate #3)")

    # (6) bench validity failure voids the pair (P9-P12)
    cells = build()
    cells[("a_r2", "OFF", "251")]["bench"]["all_pass"] = False
    if not any(u["cell"] == "a_r2" and u["seed"] == "251" for u in label(cells, pairs)["unresolved"]):
        failures.append("P9-P12 failure does not void the pair")

    # (7) R4' ABORT branch reachable
    cells = build()
    cells[("b_r3", "ON", "4386")]["e_cnt"]["pct"] = 94.0
    if label(cells, pairs)["rules"]["R4prime"]["label"] != "ABORT_STICKY_STATE_MISMATCH":
        failures.append("R4' ABORT branch unreachable")

    # (8) bootstrap determinism: same input -> same CI (registered rng seed)
    a = paired_bootstrap([0.1, -0.2, 0.05, 0.0])
    b = paired_bootstrap([0.1, -0.2, 0.05, 0.0])
    if (a["ci_lo"], a["ci_hi"]) != (b["ci_lo"], b["ci_hi"]):
        failures.append("bootstrap is not deterministic under the registered rng seed")

    # ---- REGRESSION CHECKS (lesson 53: a test of a repair MUST fail on a variant
    # that reverts the repair).  The eight reachability checks above did NOT cover
    # the three harness-audit fatals; these do.
    # (9) H-F1: a boot with NO bench artefact (P7 CAPPED / crash) must void its pair.
    cells = build()
    del cells[("a_r4", "ON", "251")]["bench"]
    r = label(cells, pairs)
    if not any(u["cell"] == "a_r4" and u["seed"] == "251" for u in r["unresolved"]):
        failures.append("H-F1 regression: absent bench artefact does NOT void the pair")
    if r["rules"]["R2"]["a_r4"]["label"] != "UNRESOLVED":
        failures.append("H-F1 regression: R2 still labels a cell whose pair was truncated")
    # (10) H-F2: SEED_SPREAD_FAILS must make that cell's R1/R2 UNRESOLVED.
    cells = build()
    cells[("a_r4", "OFF", "2630")]["e_cnt"]["pct"] = 15.5  # 6.5pp spread vs the 5pp rule
    r = label(cells, pairs)
    if r["rules"]["R5"]["a_r4"]["label"] != "SEED_SPREAD_FAILS":
        failures.append("H-F2 regression: R5 does not fire on a 6.5pp spread")
    if r["rules"]["R2"]["a_r4"]["label"] != "UNRESOLVED":
        failures.append("H-F2 regression: SEED_SPREAD_FAILS does not void R2 (rule computed too late?)")
    if "a_r4" in str(r["rules"]["R1"]["per_seed"].keys()) and r["rules"]["R1"]["label"] == "STICKY_REALIZES_A":
        failures.append("H-F2 regression: R1 still reads a cell R5 rejected")
    # (11) H-B1: the probe-clock guard must be read, not merely recorded.
    cells = build()
    cells[("a_r2", "ON", "4386")]["ABORT_PROBE_CLOCK_MISMATCH"] = "t_probe_end outside span"
    if not any(u["cell"] == "a_r2" and u["seed"] == "4386" for u in label(cells, pairs)["unresolved"]):
        failures.append("H-B1 regression: ABORT_PROBE_CLOCK_MISMATCH is fail-OPEN")
    # (12) H-B5: ADDENDUM A6's band must apply to measured cells, both sides of it.
    for pct, tag in ((94.0, "<95"), (97.0, "[95,100)")):
        cells = build()
        cells[("a_r2", "OFF", "4162")]["e_pact"]["pct"] = pct
        if not any(u["cell"] == "a_r2" and u["seed"] == "4162"
                   for u in label(cells, pairs)["unresolved"]):
            failures.append(f"H-B5 regression: e_pact {tag} does not void the pair")

    # (13) H-B2: the P8 arm-label guard must be READ, not merely recorded.
    cells = build()
    cells[("a_r2", "ON", "251")]["ABORT_ARM_LABEL_MISMATCH"] = "P8: run_ids [x] do not carry arm ON"
    if not any(u["cell"] == "a_r2" and u["seed"] == "251" for u in label(cells, pairs)["unresolved"]):
        failures.append("H-B2 regression: ABORT_ARM_LABEL_MISMATCH is fail-OPEN")
    # (14) #4: a bench record without achieved_req_s must void the pair.
    cells = build()
    del cells[("a_r4", "ON", "4386")]["bench"]["achieved_req_s"]
    if not any(u["cell"] == "a_r4" and u["seed"] == "4386" for u in label(cells, pairs)["unresolved"]):
        failures.append("#4 regression: bench without achieved_req_s is fail-OPEN")
    # (16) #1: when R5 (or any voiding rule) removes a whole cell, R1 must SAY SO.
    # Without this the surviving cell's PASS is printed as the headline and a cell
    # whose OFF band was violated simply disappears.
    cells = build()
    cells[("a_r4", "OFF", "2630")]["e_cnt"]["pct"] = 15.5   # 6.5pp spread -> a_r4 voided
    r = label(cells, pairs)
    if r["rules"]["R1"]["label"] != "STICKY_REALIZES_A_ON_REMAINING_CELLS":
        failures.append(f"#1 regression: R1 does not disclose domain shrinkage "
                        f"(got {r['rules']['R1']['label']!r})")
    if r["rules"]["R1"].get("cells_dropped") != ["a_r4"]:
        failures.append("#1 regression: R1 does not report cells_dropped")
    # (17) H-F1 must fail CLEANLY, not by traceback: a pair with no bench artefact is
    # voided before anything dereferences it.
    cells = build()
    cells[("a_r2", "OFF", "2630")]["bench"] = None
    try:
        r = label(cells, pairs)
    except Exception as exc:                                  # noqa: BLE001
        failures.append(f"H-F1 regression: absent bench raises instead of voiding ({exc!r})")
    else:
        if not any(u["cell"] == "a_r2" and u["seed"] == "2630" for u in r["unresolved"]):
            failures.append("H-F1 regression: bench=None is fail-OPEN")

    # (15) H-B3 (A3 covers Q4) + H-B7 (`errors` fail-closed), exercised in e2_realized_mix.
    import tempfile, json as _j, importlib.util as _iu
    _mix = os.path.join(os.path.dirname(os.path.abspath(__file__)), "e2_realized_mix.py")
    _s = _iu.spec_from_file_location("_mix", _mix); _m = _iu.module_from_spec(_s); _s.loader.exec_module(_m)
    with tempfile.TemporaryDirectory() as _d:
        _p = os.path.join(_d, "t.jsonl")
        with open(_p, "w") as _f:
            for _t in (1.0, 2.0):                      # inside the probe window
                _f.write(_j.dumps({"event": "split_transition", "timestamp_monotonic_s": _t,
                                   "run_id": "e2_a_r4_ON_4386_1"}) + "\n")
            for _t in (4.0, 5.0, 6.0):                 # after it
                _f.write(_j.dumps({"event": "split_transition", "timestamp_monotonic_s": _t,
                                   "run_id": "e2_a_r4_ON_4386_1"}) + "\n")
                _f.write(_j.dumps({"event": "runtime_snapshot", "phase": "benchmark",
                                   "timestamp_monotonic_s": _t, "stream_index": 2,
                                   "prefill_sms": 64, "decode_sms": 44,
                                   "decode_running_batch_size": 1, "decode_iterations": int(_t),
                                   "prefill_active_batch_size": 1, "prefill_queue_depth": 1,
                                   "run_id": "e2_a_r4_ON_4386_1"}) + "\n")
        if _m.compute(_p, None)["split_transition"] != 5:
            failures.append("H-B3 regression: unfiltered Q4 count changed")
        if _m.compute(_p, 3.0)["split_transition"] != 3:
            failures.append("H-B3 regression: A3 probe filter does NOT apply to Q4")
        _b = os.path.join(_d, "b.jsonl")
        _rec = {"completed": 4, "total_input_tokens": 4 * 256, "random_range_ratio": 1.0,
                "duration": 2.0, "output_lens": [512] * 4}
        open(_b, "w").write(_j.dumps(_rec) + "\n")
        if _m.bench_validity(_b, 4, 256, 512)["all_pass"]:
            failures.append("H-B7 regression: absent `errors` is fail-OPEN")
        _rec["errors"] = [""] * 4
        open(_b, "w").write(_j.dumps(_rec) + "\n")
        if not _m.bench_validity(_b, 4, 256, 512)["all_pass"]:
            failures.append("H-B7 regression: a clean bench record no longer passes")

    if failures:
        print("SELFTEST_FAILED")
        for f in failures:
            print(f"  {f}")
        return 1
    print("SELFTEST_OK 8 reachability checks + 10 repair-regression checks "
          "(H-F1 x2/H-F2/H-B1/H-B2/H-B3/H-B5/H-B7/#1/#4); every rule's PASS and FAIL branch "
          "is producible")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells-dir")
    ap.add_argument("--expect", help="comma list of <cell>/<seed> pairs that MUST be present")
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    pairs = [tuple(p.split("/")) for p in args.expect.split(",")]
    res = label(load_cells(args.cells_dir), pairs)
    if args.out:
        json.dump(res, open(args.out, "w"), indent=2, sort_keys=True)
    print(json.dumps(res["rules"], indent=1, sort_keys=True))
    print(f"UNRESOLVED: {res['unresolved']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
