#!/usr/bin/env python3
"""M4R -- THE REGISTERED DECISION RULE, fixed in code.  Rule layer only, NOT AUDITED.

RULE_REV = 1.  Gate #34 stage 1.  GPU spend 0 (this track reanalyses telemetry
that already exists; it submits no jobs).

THE QUESTION.  CONSENSUS s1-24 says the ITL tail on this substrate is dominated
by "the time monolithic prefill stopped decode", on the basis of one rate (2),
one arm (Ha8), one seed, 17 stall probes.  M4R asks whether the mechanism holds
across cells -- and it deliberately does NOT ask s1-24's own question, because
that question is close to vacuous (see IDENTITY NOTE below).

IDENTITY NOTE (gate #40, this is what changed the design):
  1.  s1-24's SIZE half is ~arithmetic.  Its own table follows
      t ~ 1/prefill_SM to within 2.5% across d24..d92 (elasticity 0.99), so
      "monotone in D" carries almost no information.  Demoted to CALIBRATION.
  2.  s1-24's ATTRIBUTION half ("a stall coincides with an in-flight prefill")
      is near-vacuous GIVEN the base rates: decode is busy ~72% of wall time and
      prefill-active time is a near-subset of decode-active time (measured:
      g16 d44 prefill 5.73% / BOTH 5.73%).  Coincidence is therefore expected.
  3.  So the estimand is neither of those.  It is the MAGNITUDE of the decode
      slowdown inside the window:  R = rate_outside / rate_inside.
  4.  SNAPSHOT COUNTS ARE NOT TIME.  Count-weighted activity fractions differ
      from time-weighted ones by up to 30x on the same file (g16 d44 decode:
      2.4% counted vs 72.5% time-weighted).  Every estimator here is
      TIME-WEIGHTED; a count-weighted variant is a registered FAILURE.

WHAT THIS FILE IS NOT: not a measurement, not a harness, not evidence.
Run:  python3 m4r_rule.py
"""
import hashlib, json, os, sys
from itertools import product

RULE_REV = 1

# ── registered labels ────────────────────────────────────────────────────────
ABSENT   = "TELEMETRY_ABSENT"        # file missing / unparseable / no benchmark phase
FLAT     = "ITERATIONS_FLAT"         # decode_iterations does not advance -> no rate
THIN     = "COVERAGE_THIN"           # confined time below the registered minimum
NOCONTR  = "NO_CONTRAST"             # unconfined-and-decode-busy population empty
CADENCE  = "CADENCE_SUSPECT"         # sampling interval correlates with activity
CONFIRM  = "CONFINEMENT_CONFIRMED"   # R >= R_MIN with CI excluding 1, multi-cell
SCOPED   = "CONFINEMENT_SCOPED"      # same, but only one model or one rate reached
ABSENTFX = "CONFINEMENT_ABSENT"      # R <= 1 -- decode is not slower inside
TIE      = "INCONCLUSIVE_TIE"        # 1 < R < R_MIN, or CI includes 1

MEASUREMENT = {ABSENT, FLAT, THIN, NOCONTR, CADENCE}
SUBSTANTIVE = {CONFIRM, SCOPED, ABSENTFX, TIE}
LABELS = MEASUREMENT | SUBSTANTIVE

# ── registered constants ─────────────────────────────────────────────────────
MIN_CONFINED_S   = 20.0   # per cell, time-weighted
MIN_UNCONFINED_S = 20.0
R_MIN            = 1.15   # a 15% decode slowdown inside the window; below this the
                          # mechanism cannot carry the ITL tail it is credited with
MAX_GAP_S        = 1.0    # inter-snapshot gaps above this are dropped (boot/drain)
CADENCE_MAX_RATIO = 3.0   # median sampling interval, confined vs unconfined
MIN_CELLS_MODEL  = 2      # distinct models required for CONFIRMED
MIN_CELLS_RATE   = 2      # distinct rate regimes required for CONFIRMED

# ── world axes (exhaustive) ──────────────────────────────────────────────────
AXES = {
    "telemetry": ("ok", "absent"),
    "iters":     ("advances", "flat"),
    "confined":  ("enough", "thin"),
    "unconf":    ("enough", "empty"),
    "cadence":   ("ok", "suspect"),
    "R":         ("ge_rmin", "between", "le_one"),
    "ci":        ("excludes_one", "includes_one"),
    "models":    ("multi", "single"),
    "rates":     ("multi", "single"),
}
AXIS_ORDER = tuple(AXES)
GUARD_ORDER = ("telemetry", "iters", "confined", "unconf", "cadence")


def worlds():
    for c in product(*(AXES[a] for a in AXIS_ORDER)):
        yield dict(zip(AXIS_ORDER, c))


def label(w, skip=(), order=GUARD_ORDER):
    for g in order:
        if g in skip:
            continue
        if g == "telemetry" and w["telemetry"] == "absent": return ABSENT
        if g == "iters"     and w["iters"] == "flat":       return FLAT
        if g == "confined"  and w["confined"] == "thin":    return THIN
        if g == "unconf"    and w["unconf"] == "empty":     return NOCONTR
        if g == "cadence"   and w["cadence"] == "suspect":  return CADENCE
    if w["R"] == "le_one":
        # a null claim: it needs the CI too, else it is just an underpowered tie
        return ABSENTFX if w["ci"] == "excludes_one" else TIE
    if w["R"] == "between" or w["ci"] == "includes_one":
        return TIE
    # R >= R_MIN and CI excludes 1
    if w["models"] == "multi" and w["rates"] == "multi":
        return CONFIRM
    return SCOPED


def run_tests():
    W = list(worlds()); L = [label(w) for w in W]
    res, fail = {}, []
    def check(n, ok, d=""):
        res[n] = {"pass": bool(ok), "detail": d}
        if not ok: fail.append(n)

    check("T1_totality", all(x in LABELS for x in L), f"{len(W)} worlds")
    marg = {k: L.count(k) for k in sorted(LABELS)}
    check("T2_reachable", all(v > 0 for v in marg.values()),
          f"unreachable={[k for k,v in marg.items() if v==0]}")
    check("T3_measurement_terminal",
          all(label(w) in MEASUREMENT for w in W
              if w["telemetry"] == "absent" or w["iters"] == "flat"
              or w["confined"] == "thin" or w["unconf"] == "empty"
              or w["cadence"] == "suspect"),
          "gate #21: a measurement failure is never scored as a verdict")
    # CONFIRMED requires BOTH scope axes, in both directions
    fwd = all(w["models"] == "multi" and w["rates"] == "multi"
              for w, x in zip(W, L) if x == CONFIRM)
    rev = any(x == CONFIRM for w, x in zip(W, L)
              if w["models"] == "multi" and w["rates"] == "multi"
              and w["R"] == "ge_rmin" and w["ci"] == "excludes_one")
    check("T4_confirmed_needs_both_scopes", fwd and rev)
    # a negative claim is gated by the CI exactly as a positive one is
    check("T5_null_needs_ci",
          all(label({**w, "ci": "includes_one"}) == TIE
              for w in W if label(w) == ABSENTFX),
          "R<=1 with a CI covering 1 is a TIE, not a refutation")
    # every guard must move labels when removed  (no hollow guard)
    for g in GUARD_ORDER:
        moved = sum(1 for w, x in zip(W, L) if label(w, skip=(g,)) != x)
        check(f"M_{g}_not_hollow", moved > 0, f"{moved} worlds change")
    # the guard order is a registered degree of freedom
    alt = ("telemetry", "iters", "cadence", "confined", "unconf")
    moved = sum(1 for w, x in zip(W, L) if label(w, order=alt) != x)
    check("M_order_is_load_bearing", moved > 0,
          f"cadence-first moves {moved} labels; order REGISTERED as {GUARD_ORDER}")
    # no inert axis
    idx = {tuple(w[a] for a in AXIS_ORDER): x for w, x in zip(W, L)}
    inert = []
    for i, a in enumerate(AXIS_ORDER):
        if not any(idx[tuple(w[b] for b in AXIS_ORDER)[:i] + (v,) +
                       tuple(w[b] for b in AXIS_ORDER)[i+1:]] != x
                   for w, x in zip(W, L) for v in AXES[a] if v != w[a]):
            inert.append(a)
    check("T6_no_inert_axis", not inert, f"inert={inert}")
    # SCOPED and CONFIRMED are separated by scope alone, nothing else
    check("T7_scope_is_the_only_difference",
          all(label({**w, "models": "multi", "rates": "multi"}) == CONFIRM
              for w in W if label(w) == SCOPED),
          "SCOPED becomes CONFIRMED iff both scope axes are satisfied")

    out = {"rule_rev": RULE_REV, "n_worlds": len(W),
           "axes": {k: list(v) for k, v in AXES.items()},
           "guard_order": list(GUARD_ORDER),
           "labels": {"MEASUREMENT": sorted(MEASUREMENT), "SUBSTANTIVE": sorted(SUBSTANTIVE)},
           "marginals": marg,
           "constants": {"MIN_CONFINED_S": MIN_CONFINED_S, "MIN_UNCONFINED_S": MIN_UNCONFINED_S,
                         "R_MIN": R_MIN, "MAX_GAP_S": MAX_GAP_S,
                         "CADENCE_MAX_RATIO": CADENCE_MAX_RATIO,
                         "MIN_CELLS_MODEL": MIN_CELLS_MODEL, "MIN_CELLS_RATE": MIN_CELLS_RATE},
           "tests": res, "all_pass": not fail, "failed": fail}
    return out


if __name__ == "__main__":
    out = run_tests()
    out["rule_sha256"] = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "m4r_rule_rev1.json")
    json.dump(out, open(p, "w"), indent=2, ensure_ascii=False, sort_keys=True)
    print(f"RULE_REV={RULE_REV} worlds={out['n_worlds']} sha={out['rule_sha256'][:16]}")
    for k, v in out["tests"].items():
        print(f"  {'PASS' if v['pass'] else 'FAIL'}  {k}  {v['detail']}")
    print("  marginals:", json.dumps(out["marginals"], ensure_ascii=False))
    print("ALL PASS" if out["all_pass"] else f"FAILED: {out['failed']}")
    sys.exit(0 if out["all_pass"] else 1)
