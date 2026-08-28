#!/usr/bin/env python3
"""M4R rev2 -- THE REGISTERED DECISION RULE, fixed in code.  NOT AUDITED.

RULE_REV = 2.  rev1 was audited NO-GO (F1-F4 fatal, B1-B16 blocking).  Two GPU-0
probes registered by that audit have since been bought and they decide the shape
of this revision:

  probes/alias_verdict.json     EXPOSURE_ALIASED, 28/28 cells,
                                P(decode_sms == D | confined) = 1.0000 exactly.
      => rev1's estimand R = rate_free/rate_confined is BARRED.  It contrasts
         108 SM against D SM and therefore measures the C2 lever, exactly as
         CONSENSUS s1-26(B) predicted in writing ("decode ran at D SM" and
         "prefill was in flight" are THE SAME EVENT on this substrate).
  probes/rmatched_verdict.json  RMATCHED_VIABLE, 7/7 cells at every drain cut,
                                clean free@D time 38-90 s per D pooled over blocks.
      => the SM-matched replacement is measurable without new GPU.

rev2 therefore REPLACES the estimand rather than widening it:

    R_matched = rate(free & decode_sms==D & not drain) / rate(confined & decode_sms==D)

Holding SM fixed removes the aliasing.  What survives is the question "does being
inside a prefill window cost decode anything BEYOND the SM reduction?"

WHAT rev2 DOES NOT DO: it does not test s1-24's own claim.  s1-24 says the ITL
tail is dominated by confinement, and the confinement mechanism IS the SM drop --
so matching SM removes the mechanism by construction.  rev2 measures the RESIDUAL
only, and the prereg says so in its title sentence.

Run:  python3 m4r_rule_rev2.py
"""
import hashlib, json, os, sys
from itertools import product

RULE_REV = 2

ABSENT   = "TELEMETRY_ABSENT"
FLAT     = "ITERATIONS_FLAT"
THIN     = "COVERAGE_THIN"          # clean free@D or confined@D below the floor
ALIASED  = "EXPOSURE_ALIASED"       # SM-matching failed -> estimand not identified
BATCHCF  = "BATCH_CONFOUNDED"       # decode batch differs across the contrast
CADENCE  = "CADENCE_SUSPECT"
AGGDEP   = "AGGREGATION_DEPENDENT"  # F3: per-cell and pooled disagree on the sign
RESID    = "RESIDUAL_PRESENT"       # R_matched >= R_MIN with CI excluding 1
RESNONE  = "RESIDUAL_ABSENT"        # R_matched <= 1 with CI excluding 1
RESTIE   = "RESIDUAL_INCONCLUSIVE"

MEASUREMENT = {ABSENT, FLAT, THIN, CADENCE}
DESIGN      = {ALIASED, BATCHCF, AGGDEP}
SUBSTANTIVE = {RESID, RESNONE, RESTIE}
LABELS = MEASUREMENT | DESIGN | SUBSTANTIVE

# ── constants (all measured or derived, none free) ───────────────────────────
MAX_GAP_S     = 1.0
DRAIN_MS      = 25      # probe 2: drain exclusion saturates at 25 ms; 25-200 identical
MIN_CLEAN_S   = 20.0
SM_MATCH_MIN  = 0.99    # both legs must be >= this fraction at decode_sms == D
# R_MIN is NOT the rev1 value.  rev1 derived 1.15 from a single cell's confinement
# fraction (~6%); the audit showed that fraction spans 3.7-91% across the grid
# (B14).  rev2 registers a PER-CELL threshold instead: the residual must be large
# enough that, at THAT cell's confined fraction f, it moves the ITL budget by 1%.
def r_min(f):
    """f = confined share of decode-busy time in that cell.  Residual R must
    satisfy  f * (R - 1) >= 0.01  ->  R >= 1 + 0.01/f."""
    return 1.0 + 0.01 / max(f, 1e-6)
# batch parity: the audit measured the confined batch LARGER in G16 (16.7 vs 12.2)
# and SMALLER in s8 (2.77 vs 12.07).  rev1 registered the wrong sign; rev2
# registers a two-sided parity band instead of a direction.
BATCH_PARITY_BAND = (0.80, 1.25)
SCOPE = ("Zamba2-2.7B, ShareGPT varying trace, cudagraph ON, G16 grid d16-d74, "
         "4 blocks. NOT multi-model and NOT multi-rate-regime: the audit showed "
         "CONFINEMENT_CONFIRMED was unreachable (F4), so rev2 has no such label.")

AXES = {
    "telemetry": ("ok", "absent"),
    "iters":     ("advances", "flat"),
    "coverage":  ("enough", "thin"),
    "sm_match":  ("ok", "failed"),
    "batch":     ("parity", "skewed"),
    "cadence":   ("ok", "suspect"),
    "agg":       ("agree", "disagree"),
    "R":         ("ge_rmin", "between", "le_one"),
    "ci":        ("excludes_one", "includes_one"),
}
AXIS_ORDER = tuple(AXES)
IX = {a: i for i, a in enumerate(AXIS_ORDER)}
GUARD_ORDER = ("telemetry", "iters", "coverage", "sm_match", "batch", "cadence", "agg")

def worlds(): return product(*(AXES[a] for a in AXIS_ORDER))

def label(w, skip=(), order=GUARD_ORDER):
    g = lambda a: w[IX[a]]
    for gd in order:
        if gd in skip: continue
        if gd == "telemetry" and g("telemetry") == "absent": return ABSENT
        if gd == "iters"     and g("iters") == "flat":       return FLAT
        if gd == "coverage"  and g("coverage") == "thin":    return THIN
        if gd == "sm_match"  and g("sm_match") == "failed":  return ALIASED
        if gd == "batch"     and g("batch") == "skewed":     return BATCHCF
        if gd == "cadence"   and g("cadence") == "suspect":  return CADENCE
        if gd == "agg"       and g("agg") == "disagree":     return AGGDEP
    R, ci = g("R"), g("ci")
    if ci == "includes_one": return RESTIE
    if R == "ge_rmin":       return RESID
    if R == "le_one":        return RESNONE
    return RESTIE

MUTANTS = {f"drop_{g}": dict(skip=(g,)) for g in GUARD_ORDER}
MUTANTS["reorder_sm_batch"] = dict(order=("telemetry","iters","coverage","batch",
                                          "sm_match","cadence","agg"))

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
          all(label(w) in MEASUREMENT | DESIGN for w in W
              if any(w[IX[a]] == b for a, b in (("telemetry","absent"),("iters","flat"),
                     ("coverage","thin"),("sm_match","failed"),("batch","skewed"),
                     ("cadence","suspect"),("agg","disagree")))),
          "gate #21: a measurement or design failure is never scored as a verdict")
    check("T4_null_needs_ci",
          all(label(tuple("includes_one" if a == "ci" else w[IX[a]] for a in AXIS_ORDER))
              == RESTIE for w in W if label(w) == RESNONE),
          "R<=1 with a CI covering 1 is a TIE, not a refutation")
    check("T5_alias_guard_is_the_point",
          all(label(w) == ALIASED for w in W
              if w[IX["telemetry"]]=="ok" and w[IX["iters"]]=="advances"
              and w[IX["coverage"]]=="enough" and w[IX["sm_match"]]=="failed"),
          "SM-matching failure blocks before any substantive reading")
    killed = {n: sum(1 for w, x in zip(W, L) if label(w, **kw) != x) for n, kw in MUTANTS.items()}
    check("T6_mutant_coverage", all(v > 0 for v in killed.values()),
          f"uncovered={[k for k,v in killed.items() if v==0]}")
    check("T7_every_guard_has_mutant", all(f"drop_{g}" in MUTANTS for g in GUARD_ORDER))
    # r_min must be a real function of f, not a constant in disguise
    check("T8_rmin_varies_with_f",
          abs(r_min(0.037) - r_min(0.91)) > 0.2,
          f"r_min(0.037)={r_min(0.037):.3f} vs r_min(0.91)={r_min(0.91):.3f} "
          "-- the audit's B14 (single global threshold over a 25x range) is answered")
    check("T9_no_confirmed_label", "CONFINEMENT_CONFIRMED" not in LABELS,
          "F4: the label the audit showed unreachable is not defined at all")
    out = {"rule_rev": RULE_REV, "n_worlds": len(W),
           "axes": {k: list(v) for k, v in AXES.items()},
           "guard_order": list(GUARD_ORDER), "scope": SCOPE,
           "labels": {"MEASUREMENT": sorted(MEASUREMENT), "DESIGN": sorted(DESIGN),
                      "SUBSTANTIVE": sorted(SUBSTANTIVE)},
           "marginals": marg, "mutants_killed": killed,
           "constants": {"MAX_GAP_S": MAX_GAP_S, "DRAIN_MS": DRAIN_MS,
                         "MIN_CLEAN_S": MIN_CLEAN_S, "SM_MATCH_MIN": SM_MATCH_MIN,
                         "BATCH_PARITY_BAND": list(BATCH_PARITY_BAND),
                         "r_min_formula": "1 + 0.01/f  (f = confined share of decode-busy time)"},
           "tests": res, "all_pass": not fail, "failed": fail}
    return out

if __name__ == "__main__":
    out = run_tests()
    out["rule_sha256"] = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
              "m4r_rule_rev2.json"), "w"), indent=2, ensure_ascii=False, sort_keys=True)
    print(f"RULE_REV={RULE_REV} worlds={out['n_worlds']} sha={out['rule_sha256'][:16]}")
    for k, v in out["tests"].items():
        print(f"  {'PASS' if v['pass'] else 'FAIL'}  {k}  {v['detail']}")
    print("  marginals:", json.dumps(out["marginals"], ensure_ascii=False))
    print("ALL PASS" if out["all_pass"] else f"FAILED: {out['failed']}")
    sys.exit(0 if out["all_pass"] else 1)
