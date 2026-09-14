#!/usr/bin/env python3
"""lambda* (capacity) decision rule for the campaign's stage 0 -- rev4.

Fixed BEFORE data.  Inherited in substance from the already-audited probe C
capacity rule (`results/longctx_conflict/probes/c_capacity_label.py` R1/R5,
pre-registered in `PREREG_CAPACITY_2026-09-09.md`), keyed on the request SHAPE
instead of the arm, with the two changes the two audits forced:

  rev2 (rev1 audit, D2)  the ratio is `achieved / REALIZED offered`, because the
      NOMINAL ratio in an unsaturated cell is the arrival realization factor
      1/Ebar (probe C seed 41: 12/12 cells realized arrivals ~15% fast, so
      11/12 reported `ach/off > 1`).
  rev3 (rev2 audit, D16-D17)  ★that ratio is `(N/(N-1))*span/(span+L)` -- an
      IDENTITY in the run's timing -- so an UNSATURATED cell reads not 1 but
      `kappa = (N/(N-1))/(1+L/span)`.  On (256,512) with rev2's fixed 170 s
      window kappa is 0.91-0.95, i.e. BELOW the low-side threshold, and the
      low-side test silently became "is this arm's decode ITL under 20 ms?"
      (probe C measured 19.92-23.53 ms on this very arm).  So a cell may supply
      R1's LOW side only if the plan registered it as a low-side candidate
      (`kappa_pred >= 0.97`), and the certification is void if the measured
      drain blows past the registered prediction.

  R0 CELL_VALIDITY[shape]   every cell: arrival realization inside the
                            registered band (1/Ebar in [0.95,1.05]) AND
                            `--random-range-ratio 1.0` (necessary for the
                            offline arrival replay, rev2-audit D20).  A
                            violation is a MEASUREMENT failure -> UNRESOLVED.
                            ★rev4/E1: the drain-model check is NOT in R0.
  R1 KNEE_BRACKETED[shape]  there is a cell USABLE AS A LOW SIDE (the plan
                            certified kappa_pred >= 0.97 AND the measured drain
                            did not refute that ceiling) with
                            achieved/realized_offered >= 0.95, AND a strictly
                            higher-offered cell with <= 0.90.  No drain
                            condition is placed on the HIGH side: a drain past
                            prediction is evidence that the cell SATURATED.
  R2 LAMBDA_STAR[shape]     measurement, no threshold: the largest ACHIEVED
                            rate among SATURATED cells (ratio <= 0.90).
                            Reported only when R1 is KNEE_BRACKETED.
  R3 SEED_REPEAT[shape]     registered prediction: the second-seed repeat of the
                            TOP rung stays saturated AND reproduces lambda*
                            within 5%.  ★rev3/D21: when there is no bracket
                            this is `UNRESOLVED (not evaluated)`, never
                            `REFUTED` -- rev2 recorded an unevaluated forecast
                            as refuted (gate #21 / lesson 21 recurrence).
  Not bracketed             sub-labelled by DIRECTION so the single registered
                            redesign is deterministic: `LADDER_TOO_HIGH` (the
                            lowest rung is saturated -> divide by 4),
                            `LADDER_TOO_LOW` (NO rung reached capacity ->
                            multiply by 4), else `KNEE_NOT_BRACKETED`.
                            ★rev4: `LADDER_TOO_LOW` is "nothing saturated", not
                            "the top rung reads >= 0.95" -- the latter is
                            ceiling limited and was unreachable.
  Missing expected cell     -> UNRESOLVED for that shape (lesson 21).

★D4: the expected cell names are an ARGUMENT and the rule iterates over that
list -- it does NOT glob the directory.  Under rev1's glob a cell whose boot
failed vanished from the ladder, so `UNRESOLVED` was structurally unreachable.
`--mutation-missing-cell` reproduces that failure on demand.

No arm comparison is possible or permitted: every ladder is relative to the
reference arm's own predicted capacity, and the prior campaign measured the same
quantity differing 5x across splits (D16 0.933 vs D92 0.187 req/s).

usage:
  lambda0_label.py --cells-dir DIR --expect a_r0,...,b_r3
                   [--repeat a_r4_s2,b_r3_s2] [--out LABEL.json]
  lambda0_label.py --cells-dir DIR --lambda-inf-a A --lambda-inf-b B [--fallback]
  lambda0_label.py --selftest
  lambda0_label.py --mutation-missing-cell   # shows WHY --expect is required
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path

ACH_HI = 0.95
ACH_LO = 0.90
SEED_REPEAT_TOL = 0.05
REDESIGN_FACTOR = 4.0
RATIO_KEY = "achieved_over_realized"         # PRIMARY (rev2)
RATIO_KEY_NOMINAL = "achieved_over_offered"  # reported alongside, never used


def _cell_invalid(c):
    """R0 -- MEASUREMENT validity only.  Returns a reason, or None.

    ★rev4/E1: the drain-model check is NOT here any more.  rev3 put it in R0,
    applied to every cell the PLAN had certified as a possible low side, and a
    single violation voided the whole shape.  That made `KNEE_BRACKETED[B]`
    unreachable: shape B certifies 4/4 rungs, and R1's HIGH side
    (`ratio <= 0.90`) is -- by the registered identity -- exactly
    `drain >= ((N/(N-1))/0.90 - 1)*span` = 19.2-34.5 s, while the guard bit at
    2*Lhat = 8.3-8.5 s.  The evidence the bracket REQUIRES was being discarded
    as a measurement failure (independently recomputed: ratio 2.0-2.9x on every
    certified rung of every registered scenario).

    The fix is a granularity fix, not a new constant: a drain past prediction is
    evidence that THAT CELL SATURATED, which is exactly what a high-side rung is
    supposed to show.  It only invalidates the cell's LOW-SIDE certification,
    because the kappa ceiling was computed from the predicted drain.  So it now
    lives in `_usable_low` below.

    Fail-closed on ABSENT keys: a record that does not carry a guard came from
    an analyzer that never evaluated it, and must not be trusted by default.
    """
    if not c.get("ebar_guard_ok", False):
        return "arrival realization outside the registered band"
    if not c.get("range_ratio_ok", False):
        return ("random_range_ratio != 1.0 -- the offline arrival replay is "
                "invalid (benchmark/datasets/common.py:56-64 consumes the RNG "
                "unless the range is degenerate)")
    return None


def _usable_low(c):
    """May this cell supply R1's LOW side?

    Two conditions, both fail-closed on an absent key:
      low_side_candidate  the PLAN certified a ceiling kappa_pred >= 0.97, so a
                          reading >= 0.95 is attainable at this rung at all.
      drain_model_ok      the measured drain did not blow past the prediction
                          the ceiling was computed from.  If it did, the
                          certification is void FOR THIS CELL -- and note that
                          the same fact is a perfectly good high-side reading.
    """
    certified = c.get("low_side_candidate", False)
    ceiling_intact = c.get("drain_model_ok", False)
    return bool(certified and ceiling_intact)


def rule(cells, repeats=None, enforce_guards=True):
    """cells: {shape: [cell-record | None, ...]}, repeats: {shape: record|None}.

    A record needs `offered_rate`, `achieved_rate`, `achieved_over_realized` and
    `low_side_candidate`; with `enforce_guards` also `ebar_guard_ok`,
    `range_ratio_ok`, `drain_model_ok`.

    `enforce_guards=False` exists for exactly one caller: replaying probe C,
    whose seed 41 and range-ratio predate this registration.  That run is the
    artifact this stage removes, so of course it fails the guards; the point of
    replaying it is that the RATIO RULE still returns the audited verdicts.
    """
    repeats = repeats or {}
    out = {}
    for shape, seq in sorted(cells.items()):
        if not seq or any(c is None for c in seq):
            out[shape] = {
                "verdict": "UNRESOLVED",
                "reason": "missing expected cell(s) -- MEASUREMENT failure, not "
                          "a rule failure (gate #21)",
                "n_expected": len(seq), "n_present": sum(c is not None for c in seq),
                "lambda_star": None, "ladder": None, "seed_repeat": None,
            }
            continue
        if enforce_guards:
            bad = [(c.get("label", c["offered_rate"]), _cell_invalid(c))
                   for c in seq if _cell_invalid(c)]
            if bad:
                out[shape] = {
                    "verdict": "UNRESOLVED",
                    "reason": "cell validity (R0) failed -- MEASUREMENT failure, "
                              "not a rule failure (gate #21)",
                    "r0_violations": [{"cell": n, "why": w} for n, w in bad],
                    "lambda_star": None, "ladder": None, "seed_repeat": None,
                }
                continue

        ladder = sorted(((c["offered_rate"], c[RATIO_KEY], c["achieved_rate"],
                          _usable_low(c) if enforce_guards
                          else bool(c.get("low_side_candidate", False)),
                          c.get("kappa_pred"))
                         for c in seq), key=lambda t: t[0])
        # ★R1: only a cell that is USABLE AS A LOW SIDE may be the low side.
        # (registered ceiling AND a drain that did not refute that ceiling)
        bracketed = any(
            lo[3] and lo[1] >= ACH_HI and hi[1] <= ACH_LO and hi[0] > lo[0]
            for i, lo in enumerate(ladder) for hi in ladder[i + 1:])
        saturated = [a for _, r, a, _, _ in ladder if r <= ACH_LO]
        lam = max(saturated) if (bracketed and saturated) else None

        if bracketed:
            verdict = "KNEE_BRACKETED"
        elif ladder[0][1] <= ACH_LO:
            verdict = "LADDER_TOO_HIGH"      # even the lowest rung saturated
        elif not saturated:
            # ★rev4: "no rung ever reached capacity".  rev3 asked instead for
            # the TOP rung to read >= ACH_HI, and that is a CEILING-limited
            # quantity: the top rung runs at T=170 s where kappa is ~0.90-0.94,
            # so it can never read 0.95 and the branch was unreachable under the
            # anchored ladder.  Found by regenerating the reachability map
            # (E2) rather than by inspection; the rev3 audit's own replacement
            # claim -- "lambda*(A)=4.0 does yield LADDER_TOO_LOW" -- is not
            # reproducible either: the shipped code returns KNEE_BRACKETED with
            # a -6.4% underestimate there.  Saturation, unlike "reads >= 0.95",
            # is not ceiling limited, so this form is reachable.
            verdict = "LADDER_TOO_LOW"
        else:
            verdict = "KNEE_NOT_BRACKETED"

        # Reported, never used by the verdict.
        nom = sorted(((c["offered_rate"], c.get(RATIO_KEY_NOMINAL, float("nan")))
                      for c in seq), key=lambda t: t[0])
        nom_br = any(lo[1] >= ACH_HI and hi[1] <= ACH_LO and hi[0] > lo[0]
                     for i, lo in enumerate(nom) for hi in nom[i + 1:])

        rep = repeats.get(shape)
        if rep is None:
            seed_repeat = {"verdict": "UNRESOLVED", "reason": "repeat cell absent"}
        elif lam is None:
            # ★D21: an unevaluated forecast is not a refuted one.
            seed_repeat = {"verdict": "UNRESOLVED",
                           "reason": "not evaluated: no bracket, so there is no "
                                     "lambda* to reproduce (gate #21)"}
        else:
            still_sat = rep[RATIO_KEY] <= ACH_LO
            rel = abs(rep["achieved_rate"] - lam) / lam
            seed_repeat = {
                "verdict": ("SEED_REPEAT_HOLDS"
                            if (still_sat and rel <= SEED_REPEAT_TOL)
                            else "SEED_REPEAT_REFUTED"),
                "still_saturated": bool(still_sat),
                "rel_diff_vs_lambda_star": rel,
                "tolerance": SEED_REPEAT_TOL,
                "achieved_rate": rep["achieved_rate"],
                "ratio": rep[RATIO_KEY],
            }

        out[shape] = {
            "verdict": verdict,
            "lambda_star": lam,
            "n_saturated_cells": len(saturated),
            "redesign_factor": (REDESIGN_FACTOR if verdict == "LADDER_TOO_LOW" else
                                1.0 / REDESIGN_FACTOR if verdict == "LADDER_TOO_HIGH"
                                else None),
            "ladder": [{"offered": o, "ach_over_realized": r, "achieved": a,
                        "low_side_candidate": cand, "kappa_pred": k,
                        "slack_vs_ACH_HI": r - ACH_HI}
                       for o, r, a, cand, k in ladder],
            "blind_spot_cells": [o for o, r, _, _, _ in ladder if ACH_LO < r < ACH_HI],
            "n_usable_low_side": sum(1 for t in ladder if t[3]),
            "n_low_side_candidates": sum(
                1 for c in seq if c.get("low_side_candidate", False)),
            "drain_disqualified_low_side": [
                c.get("label", c["offered_rate"]) for c in seq
                if c.get("low_side_candidate", False)
                and not c.get("drain_model_ok", False)],
            "nominal_denominator_would_say":
                "KNEE_BRACKETED" if nom_br else "KNEE_NOT_BRACKETED",
            "nominal_agrees": bool(nom_br == (verdict == "KNEE_BRACKETED")),
            "seed_repeat": seed_repeat,
        }
    return out


def load(cells_dir, expect, repeat=()):
    """D4: iterate the EXPECTED names.  Absent file -> None -> UNRESOLVED."""
    d = Path(cells_dir)
    cells, repeats, missing = {}, {}, []

    def read(name):
        p = d / f"cell_{name}.json"
        if not p.exists():
            missing.append(name)
            return None
        try:
            return json.loads(p.read_text())
        except Exception as exc:
            missing.append(f"{name} (unparseable: {exc})")
            return None

    for name in expect:
        rec = read(name)
        shape = rec["shape"] if rec else name.split("_")[0].upper()
        cells.setdefault(shape, []).append(rec)
    for name in repeat:
        rec = read(name)
        shape = rec["shape"] if rec else name.split("_")[0].upper()
        repeats[shape] = rec
    return cells, repeats, missing


# --------------------------------------------------------------- selftest
HERE = Path(__file__).resolve().parent
# The archived probe C artifacts are committed in-repo.  Their ABSENCE is a
# misconfiguration, never a reason to skip: a selftest that silently drops its
# only comparison against audited data is an identity (lesson 9).  Overridable
# only so the mutation harness can run a patched copy from a temp dir.
PROBE_C = Path(os.environ.get(
    "LAMBDA0_PROBE_C_DIR",
    HERE.parent.parent / "longctx_conflict" / "probes" / "c_905835"))
PROBE_C_SEED = 41           # c_capacity.sbatch:73
PROBE_C_NP = {"d16_r0_o96": 110, "d16_r1_o96": 170, "d16_r2_o96": 170,
              "d44_r0_o96": 80, "d44_r1_o96": 120, "d44_r2_o96": 130,
              "d92_r0_o96": 40, "d92_r1_o96": 40, "d92_r2_o96": 45}


def _probe_c_records():
    """Read the ARCHIVED probe C cell JSONs (not a transcription) and add the
    realized-denominator ratio this stage judges on.

    D9: the audit required the selftest to compare against the audited
    artifacts themselves -- a hard-coded transcription of numbers the same
    session produced is an identity (gate #9 / lesson 9).
    """
    sys.path.insert(0, str(HERE))
    from lambda0_analyze import arrival_span  # noqa: E402
    out = {}
    for name, npr in PROBE_C_NP.items():
        p = PROBE_C / f"cell_{name}.json"
        assert p.exists(), (
            "archived probe C cell JSON missing: %s -- this selftest compares "
            "against the audited artifacts, it does not transcribe them, so a "
            "missing artifact is a hard failure (set LAMBDA0_PROBE_C_DIR only "
            "for the mutation harness)" % p)
        c = json.loads(p.read_text())
        span, _ = arrival_span(PROBE_C_SEED, npr, c["offered_rate"])
        out.setdefault(name[:3], []).append({
            "shape": name[:3], "label": name,
            "offered_rate": c["offered_rate"],
            "achieved_rate": c["achieved_rate"],
            "achieved_over_offered": c["achieved_over_offered"],
            "achieved_over_realized": c["achieved_rate"] / ((npr - 1) / span),
            # probe C predates this registration: seed 41 and its range ratio
            # were never registered here, so the guards are switched off for
            # this replay (and asserted to FIRE when they are not).
            "ebar_guard_ok": False, "range_ratio_ok": True,
            "drain_model_ok": True, "low_side_candidate": True,
        })
    return out


def _cell(offered, ratio, achieved=None, shape="x", cand=True, guard_ok=True,
          rr_ok=True, drain_ok=True, kappa=0.98):
    return {"shape": shape, "offered_rate": offered,
            "achieved_over_realized": ratio, "achieved_over_offered": ratio,
            "achieved_rate": achieved if achieved is not None else ratio * offered,
            "low_side_candidate": cand, "kappa_pred": kappa,
            "ebar_guard_ok": guard_ok, "range_ratio_ok": rr_ok,
            "drain_model_ok": drain_ok}


def selftest():
    # ---- (0) reproduce the AUDITED probe C verdicts from the archived JSONs.
    pc = _probe_c_records()
    got = rule(pc, enforce_guards=False)
    for arm, lam in (("d16", 0.9331188685063582),
                     ("d44", 0.675302644015995),
                     ("d92", 0.18668462822130108)):
        assert got[arm]["verdict"] == "KNEE_BRACKETED", (arm, got[arm])
        assert abs(got[arm]["lambda_star"] - lam) < 1e-12, (arm, got[arm])
    # NOTE (rev2 audit L3): d44 having TWO saturated cells here is partly a
    # drain artifact (0.8988 measured vs 0.9295 idealised), so it must NOT be
    # cited as the data basis for blocking mutation M1.  The synthetic twin in
    # (3b) is what blocks M1; this assert only records the archived fact.
    assert got["d44"]["n_saturated_cells"] == 2, got["d44"]

    # ---- (0b) the guards are LOAD BEARING: the same archived ladder judged
    # with guards armed (the shipped default) is UNRESOLVED, because seed 41 was
    # never registered here.
    g = rule({"d16": pc["d16"]})["d16"]
    assert g["verdict"] == "UNRESOLVED" and len(g["r0_violations"]) == 3, g

    # ---- (1) a ladder that never saturates: not bracketed, no lambda*, and the
    # registered redesign direction is UP.
    flat = [_cell(r, 1.00) for r in (1.0, 2.0, 4.0)]
    f = rule({"a": flat})["a"]
    assert f["verdict"] == "LADDER_TOO_LOW" and f["lambda_star"] is None, f
    assert f["redesign_factor"] == REDESIGN_FACTOR, f

    # ---- (2) everything saturated: redesign direction is DOWN, and R2 is NOT
    # reported even though saturated cells exist.
    high = [_cell(r, 0.70) for r in (4.0, 8.0)]
    h = rule({"a": high})["a"]
    assert h["verdict"] == "LADDER_TOO_HIGH" and h["lambda_star"] is None, h
    assert h["n_saturated_cells"] == 2 and h["redesign_factor"] == 0.25, h

    # ---- (3) ordering comes from the offered rate, not the input order.
    shuffled = [pc["d16"][2], pc["d16"][0], pc["d16"][1]]
    assert rule({"a": shuffled}, enforce_guards=False)["a"]["verdict"] \
        == "KNEE_BRACKETED"

    # ---- (3b) M1: TWO saturated rungs whose achieved rates differ, synthetic
    # so that `max(saturated)` stays load bearing without the archive.
    two_sat = [_cell(1.0, 1.00), _cell(2.0, 0.85, achieved=1.70),
               _cell(4.0, 0.45, achieved=1.80)]
    t = rule({"a": two_sat})["a"]
    assert t["verdict"] == "KNEE_BRACKETED" and t["n_saturated_cells"] == 2, t
    assert abs(t["lambda_star"] - 1.80) < 1e-12, t   # max, not min (1.70)

    # ---- (4) M10: lambda* is the max over SATURATED cells only.
    collapse = [_cell(1.0, 0.96, achieved=0.96), _cell(2.0, 0.40, achieved=0.80)]
    r4 = rule({"a": collapse})["a"]
    assert r4["verdict"] == "KNEE_BRACKETED" and abs(r4["lambda_star"] - 0.80) < 1e-12
    assert max(c["achieved_rate"] for c in collapse) > r4["lambda_star"], r4

    # ---- (5) M2: non-monotone and TIED ladders are not brackets.  The tie case
    # is not hypothetical: the F4 second-seed repeat runs at the SAME nominal
    # rate as its rung, so without `hi[0] > lo[0]` one rung could bracket itself.
    assert rule({"a": [_cell(1.0, 0.85), _cell(2.0, 0.98)]})["a"]["verdict"] \
        != "KNEE_BRACKETED"
    tie = [_cell(2.0, 0.98), _cell(2.0, 0.85)]
    assert rule({"a": tie})["a"]["verdict"] != "KNEE_BRACKETED", rule({"a": tie})

    # ---- (6) M5-M8: the thresholds are CLOSED at 0.95 and 0.90, and the open
    # interval between them is the blind spot, which is NEITHER side.
    edge = [_cell(1.0, 0.95), _cell(2.0, 0.90)]
    e = rule({"a": edge})["a"]
    assert e["verdict"] == "KNEE_BRACKETED" and abs(e["lambda_star"] - 1.80) < 1e-12
    blind = [_cell(1.0, 0.94), _cell(2.0, 0.91)]
    b = rule({"a": blind})["a"]
    assert b["blind_spot_cells"] == [1.0, 2.0], b
    assert b["lambda_star"] is None, b
    # nothing saturated anywhere -> the registered direction is UP.  (rev3 asked
    # the TOP rung to read >= 0.95 for this, which is ceiling limited and was
    # unreachable; see the verdict logic for why.)
    assert b["verdict"] == "LADDER_TOO_LOW", b

    # ---- (6b) KNEE_NOT_BRACKETED keeps a domain of its own: saturation WAS
    # observed, but no cell is usable as the low side (here the only certified
    # rung had its ceiling refuted by the drain).
    nb2 = [_cell(1.0, 0.99, drain_ok=False), _cell(2.0, 0.93, cand=False),
           _cell(4.0, 0.40, achieved=1.6, cand=False)]
    r6b = rule({"a": nb2})["a"]
    assert r6b["verdict"] == "KNEE_NOT_BRACKETED", r6b
    assert r6b["n_saturated_cells"] == 1 and r6b["n_usable_low_side"] == 0, r6b

    # ---- (7) ★rev3/D16: a cell that is NOT a registered low-side candidate may
    # not supply the low side, however good its reading.  This is the whole
    # repair: rev2's shape-A low rungs had a CEILING of 0.91-0.95, so a 0.99
    # reading there would have been an artifact of something else.
    notcand = [_cell(1.0, 0.99, cand=False, kappa=0.93),
               _cell(4.0, 0.50, achieved=2.0, cand=False)]
    nc = rule({"a": notcand})["a"]
    assert nc["verdict"] != "KNEE_BRACKETED", nc
    assert nc["n_low_side_candidates"] == 0, nc
    # ...and the SAME ladder with only the certification flipped does bracket,
    # so the restriction is what changed the answer and nothing else did.
    yes = [_cell(1.0, 0.99, cand=True, kappa=0.98),
           _cell(4.0, 0.50, achieved=2.0, cand=False)]
    assert rule({"a": yes})["a"]["verdict"] == "KNEE_BRACKETED", rule({"a": yes})
    # fail-closed on an ABSENT key.
    miss = [dict(_cell(1.0, 0.99)), _cell(4.0, 0.50, achieved=2.0)]
    miss[0].pop("low_side_candidate")
    assert rule({"a": miss})["a"]["verdict"] != "KNEE_BRACKETED", rule({"a": miss})

    # ---- (8) ★E1.  R0 = MEASUREMENT validity only (ebar, range ratio), each
    # independently load bearing and fail-closed on an absent key.
    for kw in ({"guard_ok": False}, {"rr_ok": False}):
        lad = [_cell(1.0, 0.99, **kw), _cell(4.0, 0.50, achieved=2.0)]
        assert rule({"a": lad})["a"]["verdict"] == "UNRESOLVED", (kw, rule({"a": lad}))
    for key in ("ebar_guard_ok", "range_ratio_ok"):
        lad = [dict(_cell(1.0, 0.99)), _cell(4.0, 0.50, achieved=2.0)]
        lad[0].pop(key)
        assert rule({"a": lad})["a"]["verdict"] == "UNRESOLVED", (key, rule({"a": lad}))

    # ---- (8b) ★E1 proper: a blown drain DISQUALIFIES the cell from the low
    # side; it does NOT void the shape.  This is the rev3 kill cause, inverted.
    lowbad = [_cell(1.0, 0.99, drain_ok=False),
              _cell(4.0, 0.50, achieved=2.0, cand=False)]
    lb = rule({"a": lowbad})["a"]
    assert lb["verdict"] != "UNRESOLVED", lb
    assert lb["n_usable_low_side"] == 0 and lb["n_low_side_candidates"] == 1, lb
    assert lb["drain_disqualified_low_side"] == [1.0], lb
    # fail-closed on the absent key, still without voiding the shape.
    miss_drain = [dict(_cell(1.0, 0.99)),
                  _cell(4.0, 0.50, achieved=2.0, cand=False)]
    miss_drain[0].pop("drain_model_ok")
    md = rule({"a": miss_drain})["a"]
    assert md["verdict"] != "UNRESOLVED" and md["n_usable_low_side"] == 0, md

    # ---- (8c) ★E1's other half: NO drain condition on the HIGH side.  A drain
    # past prediction is evidence the cell SATURATED -- which is what a high
    # rung is for.  rev3 discarded exactly this evidence, which is why
    # KNEE_BRACKETED[B] had an empty domain.
    for cand in (True, False):
        ok_high = [_cell(1.0, 0.99),
                   _cell(4.0, 0.50, achieved=2.0, cand=cand, drain_ok=False)]
        r8c = rule({"a": ok_high})["a"]
        assert r8c["verdict"] == "KNEE_BRACKETED", (cand, r8c)
        assert abs(r8c["lambda_star"] - 2.0) < 1e-12, r8c
    # ...and the shape-B shape of it: EVERY rung certified, top rung saturated
    # with a blown drain.  rev3 answered UNRESOLVED here for every lambda*(B)
    # in [0.3, 1.0], the anchor 0.675 included.
    allcert = [_cell(0.30, 1.00, achieved=0.30), _cell(0.47, 0.99, achieved=0.465),
               _cell(0.78, 0.87, achieved=0.675, drain_ok=False),
               _cell(1.15, 0.59, achieved=0.675, drain_ok=False)]
    ac = rule({"b": allcert})["b"]
    assert ac["verdict"] == "KNEE_BRACKETED", ac
    assert abs(ac["lambda_star"] - 0.675) < 1e-12, ac
    assert ac["n_low_side_candidates"] == 4 and ac["n_usable_low_side"] == 2, ac

    # ---- (8d) ★Y4: `load()` must take the shape from the RECORD, not from the
    # cell name.  A renderer/analyzer disagreement would silently split one
    # ladder into two and both halves would read as unbracketed.
    with tempfile.TemporaryDirectory() as td:
        rec = dict(_cell(1.0, 0.99, shape="B"), label="a_r0")
        Path(td, "cell_a_r0.json").write_text(json.dumps(rec))
        cells, _, _ = load(td, ["a_r0"])
        assert set(cells) == {"B"}, ("shape must come from the record", cells)

    # ---- (9) missing expected cell -> UNRESOLVED (in-memory form).
    assert rule({"a": [_cell(1.0, 1.0), None, _cell(4.0, 0.7)]})["a"]["verdict"] \
        == "UNRESOLVED"

    # ---- (10) R3 both ways, and ★D21: no bracket -> UNRESOLVED, not REFUTED.
    lad = [_cell(1.0, 1.00), _cell(4.0, 0.50, achieved=2.0)]
    assert rule({"a": lad}, repeats={"a": _cell(4.0, 0.51, achieved=2.05)}
                )["a"]["seed_repeat"]["verdict"] == "SEED_REPEAT_HOLDS"
    assert rule({"a": lad}, repeats={"a": _cell(4.0, 0.45, achieved=1.80)}
                )["a"]["seed_repeat"]["verdict"] == "SEED_REPEAT_REFUTED"
    nb = rule({"a": [_cell(r, 1.00) for r in (1.0, 2.0)]},
              repeats={"a": _cell(2.0, 1.00)})["a"]
    assert nb["lambda_star"] is None
    assert nb["seed_repeat"]["verdict"] == "UNRESOLVED", nb["seed_repeat"]
    assert "not evaluated" in nb["seed_repeat"]["reason"], nb["seed_repeat"]

    # ---- (11) ★X1: a BLIND-SPOT cell is not the high side.  (rev2's harness
    # never exercised this and the auditor's X1 escaped: with `hi <= ACH_HI`
    # instead of `<= ACH_LO`, a 0.93 reading would close a bracket.  shape A is
    # exactly where readings land in that band.)
    x1 = [_cell(1.0, 0.99), _cell(2.0, 0.93, cand=False)]
    r11 = rule({"a": x1})["a"]
    assert r11["verdict"] != "KNEE_BRACKETED", r11
    assert r11["lambda_star"] is None and r11["n_saturated_cells"] == 0, r11

    # ---- (12) ★X2: a BLIND-SPOT cell is not in the saturated set either, so it
    # cannot become lambda*.
    x2 = [_cell(1.0, 0.99, achieved=0.99), _cell(2.0, 0.93, achieved=1.86, cand=False),
          _cell(4.0, 0.45, achieved=1.80, cand=False)]
    r12 = rule({"a": x2})["a"]
    assert r12["verdict"] == "KNEE_BRACKETED", r12
    assert abs(r12["lambda_star"] - 1.80) < 1e-12, r12       # not 1.86
    assert r12["n_saturated_cells"] == 1, r12

    # ---- (13) ★END TO END (gate #181 / rev2-audit D22): the ANALYZER's real
    # output dict, not a hand-built one, must flow into `rule()` unchanged.
    _end_to_end()

    # ---- (14) D4 end to end: a DELETED cell file must reach UNRESOLVED.
    assert mutation_missing_cell(verbose=False) == "UNRESOLVED"

    print("SELFTEST OK (rev4: reproduces the archived probe-C verdicts "
          "D16/D44/D92 = KNEE_BRACKETED with lambda* 0.93312/0.67530/0.18668 "
          "read from the cell JSONs; low side restricted to registered kappa "
          "candidates and fail-closed on an absent flag; R0 guards ebar/"
          "range-ratio/drain each load bearing and fail-closed; ladder-too-high"
          "/-too-low carry the redesign direction; thresholds closed at "
          "0.95/0.90; a blown drain DISQUALIFIES a cell from the low side and "
          "never voids the shape [E1] and carries NO condition on the high "
          "side; LADDER_TOO_LOW = nothing saturated; unevaluated seed repeat "
          "is UNRESOLVED not REFUTED; deleted cell file -> UNRESOLVED)")


def _ulp_walk(x, k):
    """`x` moved `|k|` representable steps toward +inf (k>0) or 0 (k<0)."""
    for _ in range(abs(k)):
        x = math.nextafter(x, math.inf if k > 0 else 0.0)
    return x


def _synth_bench(path, n, rate, seed, in_len, out_len, itl_s, ttft_s,
                 lam_star=None):
    """A bench_serving --output-details record with a KNOWN arrival window and a
    KNOWN drain, so the analyzer's arithmetic can be checked against a value
    derived independently (from the definitions, not from the analyzer).

    `lam_star` switches on the saturation model the audits used and verified on
    the archived cells: `duration = max(span + drain_uncontended, N/lambda*)`.
    """
    import numpy as np
    np.random.seed(seed)
    iv = np.random.exponential(1.0 / rate, size=n - 1)
    span = float(iv.sum())
    duration = span + ttft_s + (out_len - 1) * itl_s      # last arrival drains
    if lam_star:
        duration = max(duration, n / lam_star)
    rec = {
        "errors": [""] * n,
        "ttfts": [ttft_s] * n,
        "itls": [[itl_s] * (out_len - 1)] * n,
        "duration": duration,
        "completed": n,
        "request_rate": rate,
        "request_throughput": n / duration,
        # ★deliberately WRONG, so a mutation that sources `achieved_rate` from
        # any other throughput field (auditor mutation Y1) is caught by the
        # numeric assert in `_end_to_end`, not merely by a KeyError.
        "output_throughput": 999.0,
        "total_throughput": 12345.0,
        "input_throughput": 6789.0,
        "random_input_len": in_len, "random_output_len": out_len,
        "random_range_ratio": 1.0,
        "total_input_tokens": n * in_len, "total_output_tokens": n * out_len,
        "concurrency": float("nan"),
    }
    Path(path).write_text(json.dumps(rec))
    return span, duration


def _end_to_end():
    """Wire `lambda0_analyze.analyze` -> `rule` with nothing in between.

    rev2's harness mutated `lambda0_label.py` only, so five of the auditor's six
    independent mutations escaped -- including one on the primary estimator
    itself (realized denominator N-1 -> N).  This closes the wiring.
    """
    import numpy as np
    sys.path.insert(0, str(HERE))
    from lambda0_analyze import analyze  # noqa: E402
    seed, n, rate = 4056, 320, 0.525
    with tempfile.TemporaryDirectory() as td:
        bp = Path(td, "bench_a_r0.jsonl")
        span, duration = _synth_bench(bp, n, rate, seed, 256, 512, 0.0230, 0.091)
        lhat = 0.091 + 511 * 0.0230
        lo = analyze(bp, "A", "a_r0", n, seed, kappa_pred=0.9839,
                     drain_pred_s=lhat, low_side_candidate=True, t_measure_s=600.0)

        # (a) the analyzer's estimator, re-derived here FROM THE DEFINITIONS.
        np.random.seed(seed)
        span_indep = float(np.random.exponential(1.0 / rate, size=n - 1).sum())
        off_real_indep = (n - 1) / span_indep          # <- N-1, not N
        want = (n / duration) / off_real_indep
        assert abs(lo["achieved_over_realized"] - want) < 1e-12, (
            "realized-denominator arithmetic drifted", lo["achieved_over_realized"], want)
        assert abs(lo["span_s"] - span) < 1e-9 and abs(lo["drain_s"] - (duration - span)) < 1e-9
        assert lo["shape"] == "A" and lo["label"] == "a_r0"
        assert lo["ebar_guard_ok"] and lo["range_ratio_ok"] and lo["drain_model_ok"]
        assert lo["input_len_exact"] and lo["output_len_exact"]
        # the reading must clear the low side, i.e. the extended window works.
        assert lo["achieved_over_realized"] >= ACH_HI, lo["achieved_over_realized"]

        # (b) a saturated high rung, built by giving it a drain far past its span.
        bp2 = Path(td, "bench_a_r4.jsonl")
        _synth_bench(bp2, 400, 4.2, seed, 256, 512, 0.0230, 60.0)
        hi = analyze(bp2, "A", "a_r4", 400, seed, kappa_pred=None,
                     drain_pred_s=None, low_side_candidate=False, t_measure_s=170.0)
        assert hi["achieved_over_realized"] <= ACH_LO, hi["achieved_over_realized"]

        res = rule({"A": [lo, hi]})["A"]
        assert res["verdict"] == "KNEE_BRACKETED", res
        assert abs(res["lambda_star"] - hi["achieved_rate"]) < 1e-12, res

        # (c) ★the guards must FIRE on analyzer output too, not only on hand
        # built dicts: the same cell analysed at an UNREGISTERED seed.
        bad = analyze(bp, "A", "a_r0", n, 41, kappa_pred=0.9839, drain_pred_s=lhat,
                      low_side_candidate=True, t_measure_s=600.0)
        assert not bad["ebar_guard_ok"], bad["inv_ebar"]
        assert rule({"A": [bad, hi]})["A"]["verdict"] == "UNRESOLVED"

        # (d) ★D20 on analyzer output: range ratio != 1.0 invalidates the replay.
        rr = json.loads(bp.read_text()); rr["random_range_ratio"] = 0.9
        bp3 = Path(td, "bench_rr.jsonl"); bp3.write_text(json.dumps(rr))
        rrc = analyze(bp3, "A", "a_r0", n, seed, kappa_pred=0.9839,
                      drain_pred_s=lhat, low_side_candidate=True, t_measure_s=600.0)
        assert not rrc["range_ratio_ok"], rrc["random_range_ratio"]
        assert rule({"A": [rrc, hi]})["A"]["verdict"] == "UNRESOLVED"

        # (e) ★the drain-model guard on a low-side candidate: it disqualifies
        # that cell from the LOW side and does NOT void the shape (E1).
        bp4 = Path(td, "bench_drain.jsonl")
        _synth_bench(bp4, n, rate, seed, 256, 512, 0.0230, 30.0)   # +30 s drain
        dc = analyze(bp4, "A", "a_r0", n, seed, kappa_pred=0.9839,
                     drain_pred_s=lhat, low_side_candidate=True, t_measure_s=600.0)
        assert not dc["drain_model_ok"], (dc["drain_s"], dc["drain_pred_s"])
        rdc = rule({"A": [dc, hi]})["A"]
        assert rdc["verdict"] != "UNRESOLVED", rdc
        assert rdc["n_usable_low_side"] == 0, rdc

        # (f) ★E4 -- the SHAPE B configuration, which rev3's e2e leg never ran:
        # EVERY rung certified (kappa >= 0.97 at 170 s because the drain is
        # small at out=64) and the top rung SATURATED.  Under rev3 this whole
        # branch answered UNRESOLVED; it is the kill cause reproduced end to
        # end through the shipped analyzer.
        recs = []
        lam_b = 0.675
        for name, x, npr, cert in (("b_r0", 0.304, 50, True), ("b_r1", 0.472, 80, True),
                                   ("b_r2", 0.776, 130, True), ("b_r3", 1.15, 200, True)):
            bpn = Path(td, f"bench_{name}.jsonl")
            # unsaturated rungs drain in ~4.2 s; saturated ones are queue bound.
            itl_b, ttft_b = 0.0200, 2.91
            _synth_bench(bpn, npr, x, seed, 8192, 64, itl_b, ttft_b,
                         lam_star=lam_b)
            recs.append(analyze(bpn, "B", name, npr, seed, kappa_pred=0.985,
                                drain_pred_s=ttft_b + 63 * itl_b,
                                low_side_candidate=cert, t_measure_s=170.0))
        rb = rule({"B": recs})["B"]
        assert rb["n_low_side_candidates"] == 4, rb
        assert rb["verdict"] == "KNEE_BRACKETED", rb
        assert abs(rb["lambda_star"] - lam_b) / lam_b < 0.02, rb
        # and the saturated rungs are exactly the ones the drain disqualified.
        assert set(rb["drain_disqualified_low_side"]) == {"b_r2", "b_r3"}, rb

        # (g) ★rev5/F5-2 -- the drain-model MULTIPLE GUARD's boundary DIRECTION.
        # `drain_model_ok` is `drain <= TOL * drain_pred_s`; the rev4 audit's Z19
        # flipped that `<=` to `<` and no registered mutation noticed, because
        # random data never lands exactly on the boundary.  So the boundary is
        # hit EXACTLY here: `drain/2.0` is exact in binary floating point and
        # `2.0 * (drain/2.0) == drain`, so predicting half the measured drain
        # puts the cell precisely ON the tolerance.  A closed guard must accept
        # it; one nudge outwards must reject it.  (This also pins TOL itself: at
        # TOL = 3.0 the `nudge` leg below would still pass.)
        from lambda0_analyze import DRAIN_MODEL_TOL  # noqa: E402
        raw = analyze(bp, "A", "a_r0", n, seed, kappa_pred=0.9839,
                      drain_pred_s=None, low_side_candidate=True,
                      t_measure_s=600.0)["drain_s"]
        # `raw / TOL` is exact when TOL is a power of two and within one ulp
        # otherwise, so walk a few ulps to land on `TOL * pred == raw` for ANY
        # registered tolerance (the leg must not silently depend on TOL = 2.0).
        base = raw / DRAIN_MODEL_TOL
        on_boundary = None
        for cand in [base] + [_ulp_walk(base, k) for k in
                              (1, -1, 2, -2, 3, -3, 4, -4)]:
            if DRAIN_MODEL_TOL * cand == raw:
                on_boundary = cand
                break
        assert on_boundary is not None, (
            "no float prediction puts the measured drain exactly on "
            "TOL * prediction; the boundary DIRECTION of the multiple guard "
            "cannot be tested without one", raw, DRAIN_MODEL_TOL)
        exact = analyze(bp, "A", "a_r0", n, seed, kappa_pred=0.9839,
                        drain_pred_s=on_boundary,
                        low_side_candidate=True, t_measure_s=600.0)
        assert exact["drain_s"] == DRAIN_MODEL_TOL * exact["drain_pred_s"], (
            "the boundary leg must land ON the tolerance, exactly",
            exact["drain_s"], exact["drain_pred_s"])
        assert exact["drain_model_ok"], (
            "the drain-model guard is CLOSED at TOL * prediction: a measured "
            "drain exactly equal to the tolerance is inside it", exact["drain_s"])
        nudge = analyze(bp, "A", "a_r0", n, seed, kappa_pred=0.9839,
                        drain_pred_s=on_boundary * (1 - 1e-9),
                        low_side_candidate=True, t_measure_s=600.0)
        assert not nudge["drain_model_ok"], (
            "one part in 1e9 past the tolerance must be OUTSIDE it -- otherwise "
            "the tolerance is not the tolerance", nudge["drain_s"],
            nudge["drain_pred_s"])
        # ...and the guard must be load bearing THROUGH the rule, not only in
        # the analyzer dict.
        assert rule({"A": [exact, hi]})["A"]["n_usable_low_side"] == 1
        assert rule({"A": [nudge, hi]})["A"]["n_usable_low_side"] == 0


def mutation_missing_cell(verbose=True):
    """Injection experiment for D4 (rev1 audit S9).

    Writes a 3-rung ladder whose top rung is SATURATED, deletes the top cell's
    file -- exactly what a boot failure on that rung produces -- and runs the
    shipped loader.  rev3 must answer UNRESOLVED.  rev1, which globbed the
    directory, answered KNEE_NOT_BRACKETED: a rule verdict for a measurement
    failure.
    """
    with tempfile.TemporaryDirectory() as td:
        names = ["a_r0", "a_r1", "a_r2"]
        recs = [_cell(1.0, 1.00, shape="A"), _cell(2.0, 0.99, shape="A"),
                _cell(4.0, 0.50, achieved=2.0, shape="A", cand=False)]
        for n, r in zip(names, recs):
            Path(td, f"cell_{n}.json").write_text(json.dumps(dict(r, label=n)))
        os.remove(Path(td, "cell_a_r2.json"))          # <- the boot failure
        cells, repeats, missing = load(td, names)
        res = rule(cells, repeats)
        v = res["A"]["verdict"]
        if verbose:
            print(json.dumps({"missing": missing, "result": res}, indent=2,
                             sort_keys=True))
            print("MUTATION(missing top cell) ->", v,
                  "[PASS]" if v == "UNRESOLVED" else "[FAIL: a measurement "
                  "failure was printed as a rule verdict]")
        return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells-dir")
    ap.add_argument("--expect", help="comma separated expected cell names")
    ap.add_argument("--repeat", default="", help="comma separated repeat cells")
    ap.add_argument("--lambda-inf-a", type=float)
    ap.add_argument("--lambda-inf-b", type=float)
    ap.add_argument("--fallback", action="store_true")
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--mutation-missing-cell", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        selftest()
        return 0
    if a.mutation_missing_cell:
        return 0 if mutation_missing_cell() == "UNRESOLVED" else 1
    if not a.cells_dir:
        ap.print_help()
        return 2

    if a.expect:
        expect = [s for s in a.expect.split(",") if s]
        repeat = [s for s in a.repeat.split(",") if s]
    elif a.lambda_inf_a is not None and a.lambda_inf_b is not None:
        sys.path.insert(0, str(HERE))
        import lambda0_plan
        lad = ({k: tuple(v) for k, v in lambda0_plan.FALLBACK_LADDER.items()}
               if a.fallback else None)
        p = lambda0_plan.plan(a.lambda_inf_a, a.lambda_inf_b, ladders=lad)
        expect = [c["name"] for s in ("A", "B") for c in p["shapes"][s]["cells"]]
        repeat = [p["shapes"][s]["seed_repeat_cell"] for s in ("A", "B")]
    else:
        ap.print_help()
        return 2

    cells, repeats, missing = load(a.cells_dir, expect, repeat)
    # ★E6: the anchor's provenance travels INTO the verdict file, so a reader
    # can tell which I3 artifacts (by digest) produced the ladder this label
    # was computed on.
    dec_path = Path(a.cells_dir) / "LAMBDA_INF_DECISION.json"
    decision = (json.loads(dec_path.read_text()) if dec_path.exists()
                else {"note": "no LAMBDA_INF_DECISION.json beside the cells"})
    result = {
        "stage": "LAMBDA0", "rev": 5,
        "prereg": "PREREG_LAMBDA0_REV5_2026-09-14.md",
        "expected_cells": expect,
        "expected_repeat_cells": repeat,
        "missing_cells": missing,
        "lambda_inf_decision": decision,
        "ratio_key": RATIO_KEY,
        "thresholds": {"ACH_HI": ACH_HI, "ACH_LO": ACH_LO,
                       "SEED_REPEAT_TOL": SEED_REPEAT_TOL},
        "R0_R1_R2_R3_by_shape": rule(cells, repeats),
    }
    text = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False,
                      default=str)
    print(text)
    if a.out:
        Path(a.out).write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
