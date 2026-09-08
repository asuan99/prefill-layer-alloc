"""D1: does the registered PAIR's goodput difference have a sign, or only a
curve?  RULE_REV = 2.

rev1 took this track's EIGHTH consecutive NO-GO
(`audit_d1_rules_2026-09-07/VERDICT.md`, death causes U1-U8).  Its single
judging answer was: **the eighth name of the free surface is the YARDSTICK.**
rev1 divided every arm by `plain`'s uncontended floor, and that choice was
measured, at GPU 0, to put OPPOSITE-SIGNED biases on the two grid axes --
TTFT -5.07% (tilting to `d44`), ITL p95 +5.97% (tilting to `cp2048`), both above
the registered 3% decision floor (`RESULT_D1_A1A2_2026-09-07.md` A1).  That is
the SHAPE of the crossing rev1 had registered as prediction 1.  A single
yardstick could manufacture the predicted answer under no load at all.

What rev2 changes
-----------------
U3  -> every grid point is scored TWICE (fixed yardstick / each arm's own
       measured floor) and the sign is the INTERSECTION
       (`d1_predicates.sign_intersect`).  A yardstick can now only REMOVE a
       sign, never create one.  Where the two scorings give OPPOSITE signs the
       yardstick sets the direction, and that has its own label
       (`YARDSTICK_DECIDES`) which fires as a GUARD, ahead of the outcome table.
U2  -> the firing rate is a MEASURED constant now (`d1_predicates.FIRING_RATE`,
       campaign tokenizer: `cps 2048` fires on 79.67%, not 100%).
U4  -> `d1_predicates.py` exists.  Every axis value is produced by a named fold
       on `PREDICATE_FOLDS`, including the two rev1 only named in prose
       (`bracket_ok`, the degenerate-end predicates) and the rule rev1 never
       wrote at all (`call_sign`'s `max(G) == 0` convention).
U5  -> `degenerate_high` is documented as a THROUGHPUT end, and the
       leave-one-out set INCLUDES the degenerate ends (excluding them would be a
       silent choice about which points count).
U6  -> Stage V is DELETED, not repaired.  `RESULT_D1_A1A2_2026-09-07.md` A2
       priced it: at n=2, t(1)=12.706 makes the minimum detectable |delta|
       70.98% at SD 7.9% -- the gate was statistically empty.  There is no
       interim label; `UNAFFORDABLE_PRECISION` is now emitted from the FULL
       campaign, where it means something: every point was a 0-call and the CI
       is what bound it.
U7/U8 -> carried in the pre-registration (sec 9 inheritance count, absolute
       class-SLO co-report), not here.
A2's second finding is registered in the FORK, not in prose: at n=4 the
resolvable object is the SIGN AT THE LADDER EXTREMES, not the crossing locus
(near a crossing |delta| is small by definition and falls under the MDE).  The
branch is named `answered_sign_flip_extremes_only` so the scope travels with it.

★ Regime transfer (the question this rule was asked to survive)
--------------------------------------------------------------
`REGIME_FREE` and `REGIME_BOUND` below split this design into the part that
moves to a long-context environment unchanged and the part that must be
re-measured there.  The self-test requires the two sets to PARTITION every
constant in both modules, so a new constant cannot be added without answering
"does this survive the move?".  See the pre-registration sec 10.
"""
import d1_predicates as P

RULE_REV = 2
PREDICATE_MODULE = "d1_predicates.py"
# Derived, never retyped (AF-1 4th-audit K7: a byte-identical copy that nothing
# compares is the data version of the duplicate-definition risk).
PREDICATE_FOLDS = P.PREDICATE_FOLDS

# ---------------------------------------------------------------- constants
MIN_BOOTS = 4          # gate #3 (n>=4): scored boots per arm, from distinct jobs
CI_LEVEL = 0.95
N_BOOTS = 10000
PRACTICAL_FLOOR = 0.03  # gate #3: under 3% relative is not a headline
# Derived, not chosen: W1 sec 3 observed a median TTFT of 18,559 ms against a
# p50 floor of 297.1 ms = a slowdown of 62.5x.  W1 reported medians only, so the
# per-request tail is unmeasured; 1024 is the smallest power of two above 16x
# that observed maximum, which is the headroom for the gap between a median and
# a tail.  Reaching the cap is a FINDING (`GRID_UNBRACKETABLE`), not a failure.
K_MAX = 1024

DECISION_CONSTANTS = ("MIN_BOOTS", "CI_LEVEL", "N_BOOTS", "PRACTICAL_FLOOR",
                      "K_MAX")

# The registered comparison pair.  Fixed before any D1 data exists.
# ★ rev1 justified `cp2048` with "the census puts 200/200 prompts above 2048 ->
# firing rate 100%".  That was REFUTED at GPU 0 (the census used the SELECTION
# tokenizer and its own limit 1 forbids transferring boundary counts).  The
# surviving justification is the measured one: with the CAMPAIGN tokenizer
# `cp2048` fires on 79.67% of the pool -- the highest firing rate of any cps
# whose uncontended floor AF-1 actually measured.
PRIMARY_PAIR = ("cp2048", "d44")
DIAGNOSTIC_ARMS = ("plain", "cp512")

# ★ The authoritative arm definitions.  rev1 described `plain` in one place as
# the engine default; it is not (`af1_probe.sbatch:81` boots it with
# `--chunked-prefill-size -1`, i.e. chunked prefill DISABLED, while the engine
# default is 8192).  The self-test now requires the pre-registration's arm table
# to quote these strings, so the two files cannot drift again (1st audit U1).
ARM_FLAGS = {
    "plain":  "--chunked-prefill-size -1",
    "cp512":  "--chunked-prefill-size 512",
    "cp2048": "--chunked-prefill-size 2048",
    "d44":    "--enable-pdmux --pdmux-config-path pdmux_d44.yml "
              "--chunked-prefill-size -1 --disable-overlap-schedule",
}

# ------------------------------------------------------------ regime split
# What moves to a long-context environment UNCHANGED: the decision layer.
REGIME_FREE = ("MIN_BOOTS", "CI_LEVEL", "N_BOOTS", "PRACTICAL_FLOOR",
               "AXES", "AXIS_ORDER", "OUTCOME", "FORK", "GUARDS", "COHERENCE",
               "SUBSTANTIVE", "NON_SUBSTANTIVE", "PREDICATE_FOLDS",
               "FORK_BRANCHES", "FORK_GRID_FRAGILE", "DELETED_IN_REV2")
# What must be RE-MEASURED before this design may be used in another regime.
# Every one of these is a measurement of (model, workload, tree), not a choice.
REGIME_BOUND = ("K_MAX", "PRIMARY_PAIR", "DIAGNOSTIC_ARMS", "ARM_FLAGS",
                "STRATA_TOK", "FLOOR_TTFT_MS", "FLOOR_ITLP95_MS",
                "FIRING_RATE", "YARDSTICK_ARM", "DEGEN_HIGH_TOL")


# ★ The deletion ledger, as code.  Gate #88: a deletion is not a registered
# condition, so grep cannot find what is no longer there -- the only defence is
# to ENUMERATE what was removed and let a check count it.  Every name here was
# live in rev1, is dead in rev2, and must be accounted for in the
# pre-registration's version-inheritance section.
DELETED_IN_REV2 = (
    "INTERIM_BOOTS", "INTERIM_CONTINUE",          # Stage V (U6)
    "GRID_AND_YARDSTICK_DECIDE",                  # folded into the guard
    "yardstick_fragile", "both_fragile",          # robustness values, ditto
    "answered_cp", "answered_pd",                 # renamed to carry scope
    "answered_slo_dependent", "answered_null",
)


def regime_transfer(name):
    """Which half of the design a constant belongs to.  Raises on an
    unclassified name so a new constant cannot dodge the question."""
    if name in REGIME_FREE:
        return "free"
    if name in REGIME_BOUND:
        return "bound"
    raise KeyError(f"{name!r} is classified neither free nor regime-bound")


# ------------------------------------------------------------------- axes
AXES = {
    "trace": ["matched", "mismatched", "unmeasured"],
    "coverage": ["sufficient", "insufficient"],
    "bracket": ["bracketed", "unbracketable", "unmeasured"],
    # Do the two yardstick scorings agree?  `strong_discordant` = some point
    # where they give OPPOSITE signs -- there the yardstick sets the direction.
    "estimator": ["concordant", "weak_discordant", "strong_discordant",
                  "unmeasured"],
    # Computed on the INTERSECTION of the two scorings.
    "sign_field": ["cp_dominant", "pd_dominant", "crossing", "null",
                   "unmeasured"],
    "robustness": ["stable", "grid_fragile", "unmeasured"],
    "band_binding": ["ci", "floor", "both", "none", "unmeasured"],
}
AXIS_ORDER = ["trace", "coverage", "bracket", "estimator", "sign_field",
              "robustness", "band_binding"]

_FIELD = ("estimator", "sign_field", "robustness", "band_binding")
_DOWN = ("bracket",) + _FIELD


# ------------------------------------------------------------- coherence
def _k_nothing(w):
    if w["trace"] == "unmeasured" and not (
            w["coverage"] == "insufficient"
            and all(w[k] == "unmeasured" for k in _DOWN)):
        return "nothing scorable was produced, so no arm has a scored boot"
    return None


def _k_mismatch(w):
    if w["trace"] == "mismatched" and not all(w[k] == "unmeasured" for k in _DOWN):
        return "the arms did not see the same realised trace, so the analyser does not run"
    return None


def _k_coverage(w):
    if w["coverage"] == "insufficient" and not all(
            w[k] == "unmeasured" for k in _DOWN):
        return "below MIN_BOOTS there is no between-boot spread and no paired CI"
    return None


def _k_sufficient(w):
    if (w["coverage"] == "sufficient" and w["trace"] == "matched"
            and w["bracket"] == "unmeasured"):
        return "a scored campaign always runs the (zero-GPU) bracketing step"
    return None


def _k_bracket(w):
    if w["bracket"] == "unbracketable" and not all(
            w[k] == "unmeasured" for k in _FIELD):
        return "a field that cannot be bracketed is reported as a curve, not scored"
    if w["bracket"] == "bracketed" and any(w[k] == "unmeasured" for k in _FIELD):
        return "the bracketed field is scored in one pass: all four readings or none"
    return None


def _k_field_together(w):
    meas = [w[k] != "unmeasured" for k in _FIELD]
    if any(meas) and not all(meas):
        return "estimator, sign_field, robustness and band_binding are one fold"
    return None


def _k_null_needs_zeros(w):
    if w["sign_field"] == "null" and w["band_binding"] == "none":
        return "`null` means every point is a 0-call, so something produced them"
    return None


def _k_discord_makes_zeros(w):
    # An opposite-signed point becomes a 0-call under `sign_intersect`, and
    # `intersect_reason` files it as `both`.  So a strong discordance cannot
    # co-exist with "there were no 0-calls".
    if w["estimator"] == "strong_discordant" and w["band_binding"] == "none":
        return "an opposite-signed point is intersected to a 0-call, so 0-calls exist"
    return None


COHERENCE = (_k_nothing, _k_mismatch, _k_coverage, _k_sufficient, _k_bracket,
             _k_field_together, _k_null_needs_zeros, _k_discord_makes_zeros)


def incoherent(w, clauses=None):
    for c in (COHERENCE if clauses is None else clauses):
        r = c(w)
        if r:
            return r
    return None


# --------------------------------------------------------------- outcomes
def _build_outcome():
    out = {}
    for sf in ("cp_dominant", "pd_dominant", "crossing", "null"):
        for rb in ("stable", "grid_fragile"):
            for bb in ("ci", "floor", "both", "none"):
                if sf == "null" and bb == "none":
                    continue          # incoherent, never registered
                if rb == "grid_fragile":
                    out[(sf, rb, bb)] = "GRID_RESOLUTION_DECIDES"
                elif sf == "cp_dominant":
                    out[(sf, rb, bb)] = "CP2048_DOMINANT"
                elif sf == "pd_dominant":
                    out[(sf, rb, bb)] = "D44_DOMINANT"
                elif sf == "crossing":
                    out[(sf, rb, bb)] = "SLO_DEPENDENT"
                else:                  # null, stable
                    # ★ the distinction rev1 could not make: zeros bound by the
                    # CI mean "not identifiable at this precision"; zeros bound
                    # by the 3% floor mean "the effect is smaller than what this
                    # project calls a result".  Those are different answers.
                    out[(sf, rb, bb)] = ("PAIR_INDISTINGUISHABLE" if bb == "floor"
                                         else "UNAFFORDABLE_PRECISION")
    return out


OUTCOME = _build_outcome()

SUBSTANTIVE = ("CP2048_DOMINANT", "D44_DOMINANT", "SLO_DEPENDENT",
               "PAIR_INDISTINGUISHABLE", "UNAFFORDABLE_PRECISION",
               "GRID_RESOLUTION_DECIDES", "YARDSTICK_DECIDES",
               "GRID_UNBRACKETABLE")
NON_SUBSTANTIVE = ("NOTHING_SCORED", "TRACE_INVALID", "COVERAGE_INSUFFICIENT",
                   "IMPOSSIBLE_WORLD")


class OutcomeGap(Exception):
    """A coherent world reached the outcome table and was not in it."""


# ------------------------------------------------------------------ guards
def _g_nothing(w):
    return "NOTHING_SCORED" if w["trace"] == "unmeasured" else None


def _g_trace(w):
    return "TRACE_INVALID" if w["trace"] == "mismatched" else None


def _g_coverage(w):
    return "COVERAGE_INSUFFICIENT" if w["coverage"] == "insufficient" else None


def _g_bracket(w):
    return "GRID_UNBRACKETABLE" if w["bracket"] == "unbracketable" else None


def _g_yardstick(w):
    # ★ ahead of the outcome table on purpose: if the two yardsticks disagree in
    # DIRECTION anywhere, no field computed from their intersection may be read
    # as the system's answer.
    return "YARDSTICK_DECIDES" if w["estimator"] == "strong_discordant" else None


GUARDS = (_g_nothing, _g_trace, _g_coverage, _g_bracket, _g_yardstick)


def label(w, clauses=None, rules=None):
    for k in AXIS_ORDER:
        if k not in w:
            raise KeyError(f"world is missing axis {k!r}")
        if w[k] not in AXES[k]:
            raise ValueError(f"axis {k!r} has no value {w[k]!r}")
    if incoherent(w, clauses) is not None:
        return "IMPOSSIBLE_WORLD"
    for g in (GUARDS if rules is None else rules):
        lab = g(w)
        if lab:
            return lab
    key = (w["sign_field"], w["robustness"], w["band_binding"])
    if key not in OUTCOME:
        raise OutcomeGap(f"no outcome registered for {key}")
    return OUTCOME[key]


# -------------------------------------------------------------------- fork
FORK = {
    "CP2048_DOMINANT": "answered_cp_pair",
    "D44_DOMINANT": "answered_pd_pair",
    # ★ A2 scope in the branch name: at MIN_BOOTS the MDE is 12.6% of max(G) at
    # SD 7.9%, so the resolvable object is the sign at the ladder EXTREMES.  The
    # crossing LOCUS is not purchasable here and this branch does not promise it.
    "SLO_DEPENDENT": "answered_sign_flip_extremes_only",
    "PAIR_INDISTINGUISHABLE": "answered_null_below_floor",
    "UNAFFORDABLE_PRECISION": "precision_bound",
    "YARDSTICK_DECIDES": "yardstick_report",
    "GRID_UNBRACKETABLE": "regime_report",
    "NOTHING_SCORED": "n/a",
    "TRACE_INVALID": "n/a",
    "COVERAGE_INSUFFICIENT": "n/a",
    "IMPOSSIBLE_WORLD": "n/a",
}
FORK_GRID_FRAGILE = {
    "none": "refine_grid",       # free: the SLO axis is re-scored offline
    "ci": "buy_boots",           # purchasable as 1/sqrt(n), against a cap
    "floor": "precision_bound",  # 3% is a registered decision, not noise
    "both": "precision_bound",
}
FORK_BRANCHES = ("answered_cp_pair", "answered_pd_pair",
                 "answered_sign_flip_extremes_only", "answered_null_below_floor",
                 "precision_bound", "yardstick_report", "regime_report",
                 "refine_grid", "buy_boots", "n/a")


def fork_branch(lab, band_binding="unmeasured"):
    if lab == "GRID_RESOLUTION_DECIDES":
        if band_binding not in FORK_GRID_FRAGILE:
            raise KeyError(f"GRID_RESOLUTION_DECIDES has no branch for "
                           f"band_binding={band_binding!r}")
        return FORK_GRID_FRAGILE[band_binding]
    return FORK[lab]
