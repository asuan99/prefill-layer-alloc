"""AF-1: does an arm-neutral, externally-anchored operating point EXIST?
RULE_REV = 6.

Why this probe exists
---------------------
CP-2 has failed six consecutive rule-layer audits.  The 6th
(`audit_cp2r2_rules_6th_2026-09-03/VERDICT.md`) found that half of rev2's repairs
were DELETIONS, and the most expensive was F2: the rule reads the estimand "at the
registered operating point" and four of its eight substantive labels carry a
direction in their name, while no (TTFT, ITL) coordinate is registered anywhere.
W1 had already measured that the sign of the comparison flips inside the
registered ladder, so that missing coordinate WAS the conclusion's direction.

`RESULT_POSITIONING_AND_WORKLOAD_2026-09-02.md` sec 2.1 left three routes open and
adopted none.  AF-1 buys the measurement route (a) needs -- an UNCONTENDED floor
on the campaign model and workload -- and registers the fork mapping BEFORE the
run: every outcome names which branch CP-2 rev3 takes.  A probe whose result
cannot change what happens next is not a probe, and a probe whose mapping is
written afterwards IS the free parameter it was meant to remove.

What rev6 changed (`audit_af1_rules_5th_2026-09-04/VERDICT.md`, NO-GO, ONE death
cause -- three sentences and one regex; the 6th pass then returned GO)
-----------------------------------------------------------------------------------
  Q1  The branch `b_more_boots`, deleted in rev5 and entered in the deletion
      ledger, was still handing out a PRESENT-TENSE prescription in the
      pre-registration's forbidden-sentence list.  One label carried two
      branches, and the check meant to catch that read only UPPERCASE tokens
      while every branch name in this design is lowercase.
      -> the surviving descriptions now say `b_precision_bound`, and C20 reads
         identifiers of either case.  The repair carries its own test: before it,
         the widened C20 flags exactly `b_more_boots` and nothing else.

What changed in rev4 (`audit_af1_rules_4th_2026-09-04/VERDICT.md`, NO-GO + a
CONDITIONAL GO list, K1-K8)
-----------------------------------------------------------------------------------
  K2  `screen_incomplete` fired on ANY unmeasured reading and sat ahead of the
      neutrality guards, so 21 of the 107 reachable worlds -- where the registered
      stratum's neutrality was FULLY measured -- were demoted to a measurement
      failure with no prescription, and the pre-registration's sec 6 anchor
      ("the neutrality guard precedes every value of stratum, anchor and ladder")
      became FALSE while the anchor machinery kept it verbatim.
      -> split: `screen_incomplete` covers the one-pass readings, and the new
         `stratum_unmeasured` sits AFTER the neutrality guards.
  K3  `b_more_boots` -- a branch name this file took from the 3rd audit's
      prescription -- is not purchasable: `boot_sd` is a SAMPLE SD, so the band
      converges to 2*sigma rather than to zero (verified: n=4..400 gives a band of
      37.5..39.9 ms at sigma=20), and sec 3.3 forbids the purchase anyway.
      -> `b_precision_bound`: the noise decided at THIS measurement precision, the
         lever is sigma (measurement design), and AF-1 does not buy it.
  K7  `PREDICATE_CONSTANTS` was a byte-identical, unchecked copy of the predicate
      module's own list -- the data version of the duplicate-definition risk.
      -> derived from the module, not retyped.
  K1/K5/K6/K8 are repaired in the predicates, the self-test and the
  pre-registration; K4 in the self-test.

What changed in rev4 (`audit_af1_rules_3rd_2026-09-04/VERDICT.md`, J1-J8)
-----------------------------------------------------------------------------------
  J3  The image was computed over PAIRS while `neutrality` was an axis, so its
      values were checked over the DECLARED product -- an identity, and fourteen
      worlds outside the screen's image carried a substantive label.
      -> `af1_predicates.reachable_triples()` and `_c_image` over triples.
  J1  `FLOOR_NEUTRALITY_BORDERLINE` was a new NAME on the old path: identical
      floors plus one noisy arm still bought exactly what a real arm difference
      bought (`b`).  The repair went to the dominating axis; the prescription did
      not follow it.
      -> its own branch `b_more_boots`.  Noise now buys "buy more boots", not
         "route (a) is closed".
  J4  A boot that died between strata left `coverage = sufficient` with one
      stratum missing -- IMPOSSIBLE_WORLD and a `fork_branch` KeyError.  And the
      two measurement-failure labels had no branch at all.
      -> `_c_one_pass` is split by stratum, `screen_incomplete` covers any
         unmeasured reading, and `fork_branch` answers "n/a" for them.
  J8  `variance` and `VARIANCE_UNMEASURED` were deleted in rev3 and no ledger
      recorded it, while a prohibition and a docstring still named them.
      -> deletion ledger (pre-registration sec 0.4), and the ghosts are gone.

What changed in rev3 (`audit_af1_rules_2nd_2026-09-04/VERDICT.md`, E1-E10)
-----------------------------------------------------------------------------------
  E1  The FORK -- this probe's entire output -- was not pinned by anything a
      reader sees.  Flipping every `b` to `a` passed with a two-file edit while
      the pre-registration's table still said `b`, and this file claimed the
      opposite ("a one-line edit here fails the oracle").  The paired anchors were
      the right device and rev2 simply did not apply it to the branch.
      -> every anchor is now ``world` -> `LABEL` -> `branch`: ...`, and the
         pre-registration's tables are written in that form, so changing a
         prescription requires editing the document an adversarial reader reads.
  E2  neutrality judged on BANDED sets (SD re-entered the dominating axis) and
      took the union of two failures; three documents said "SD-free".
      -> strict-only, with `borderline` for a strict/banded disagreement, and it
         takes `q` so `stratum_axis` compares it too.
  E3  `boot` had no predicate, and the slack `PLANNED_BOOTS > MIN_BOOTS` created
      an IMPOSSIBLE world with a `fork_branch` KeyError.
      -> `coverage_axis` (two values, real predicate, boundary vectors) and
         `SCREEN_INCOMPLETE` for "enough boots, screen still could not run".
  E4/E5/E6/E7/E8/E10 are repaired in af1_selftest.py, the oracle and the
  pre-registration; E9 (pre-register the roofline) is pre-registration sec 5.1.

What changed in rev2 (`audit_af1_rules_2026-09-04/VERDICT.md`, D1-D10)
--------------------------------------------------------------------------------
  D2  `OUTCOME` is now required to equal the EXACT IMAGE of the screen
      (`af1_predicates.reachable_pairs`), in both directions, and the image is
      derived from the constants rather than sampled on a chosen grid.  rev1
      registered `("none", "intact")`, which the screen cannot produce, and its
      reachability check enumerated the declared axis product -- an identity.
  D3  `FORK` values are now checked for CONTENT, not only totality: every hand
      oracle row carries the branch, and the pre-registration's table is written
      in the world-key + label format the oracle pins.
  D4  `anchor` and `ladder` gained the value `borderline`: the strict and banded
      screens are both computed and a disagreement is an outcome, not a silent
      choice of which to believe.
  D7  Prose anchors are PAIRED: each begins with its own world key and label, so
      swapping two outcomes forces an edit to the pre-registration.  rev1's
      anchors were a set-membership check and a swap passed untouched.
  D8  `stratum` became an axis, `PLANNED_BOOTS` = 5 (one boot of slack), and the
      worlds the pipeline can produce all have a fork branch.
  D10 The dangling `sec 10-2` reference is gone; the prefill-partition limit is
      written where the mitigation actually lives.

What ONE thing this rule decides
--------------------------------
Whether the uncontended floor of this model, on this workload, in this tree,
admits a coordinate that (i) comes from outside this campaign, (ii) is reachable
by every registered arm, (iii) does not depend on which stratum or which band is
believed -- and, if not, which of the two remaining designs rev3 is obliged to
adopt.  Nothing else.

★What it does NOT decide, because the candidate set is a chain
--------------------------------------------------------------
The five candidates are totally ordered (`af1_predicates.survey_is_chain`), so
`anchor = unique` can only ever be `batch_async`.  AF-1 buys ONE BIT plus its
robustness, never the identity of the survivor.

Scope this rule INHERITS and does not re-argue
----------------------------------------------
  * The engine forbids {pdmux ON, chunking ON} (`server_args.py` line 6131,
    "PD-Multiplexing is not compatible with chunked prefill."), so the arms differ
    in a BUNDLE.  AF-1 makes no mechanism claim and needs none: it asks whether a
    latency is physically reachable, not why.
  * Piecewise CUDA graph capture is OFF on every arm
    (`DECISION_PIECEWISE_OFF_2026-09-02.md`).
  * The workload is the ShareGPT LONG-PROMPT TAIL, and the file was selected with
    the Zamba2 tokenizer while the campaign model is Nemotron
    (`RESULT_SHAREGPT_CENSUS_2026-09-02.md` sec 5-1), so AF-1 re-tokenises and
    never cites the dataset metadata's token counts.
  * Correctness: 55/56 greedy outputs byte-identical, concurrency UNVERIFIED
    (`RESULT_CORRECTNESS_GATE_2026-09-03.md`).  AF-1 runs at concurrency 1, so it
    does not exercise the unverified part -- a limit of AF-1's reach, not evidence.
"""

RULE_REV = 6

PREDICATE_MODULE = "af1_predicates.py"
# ★Derived, not retyped.  rev4 kept a byte-identical copy that nothing compared,
# so dropping two names from it passed every check (4th-audit K7) -- the data
# version of the duplicate-definition risk C18 exists for.
from af1_predicates import LABEL_CONSTANTS as PREDICATE_CONSTANTS  # noqa: E402
# Every fold that produces a number or a class this probe will PRINT.  6th-audit
# lesson 6 ("함수도 상수처럼 분류하라"): rev2's `answerable_margin` produced the
# number that document advertised as its virtue and sat outside every check
# surface, so `return 0.001` still passed.  A fold not on this list may not appear
# in the result document.
PREDICATE_FOLDS = (
    "order_stat", "agg_within_boot", "agg_across_boots", "boot_sd",
    "screen_bound", "arm_bounds", "survey_is_chain", "undominated_survey",
    "critical_bound_values",
    "admissible", "admissible_survey", "anchor_class", "ladder_class",
    "anchor_axis", "ladder_axis", "stratum_axis", "coverage_axis",
    "neutrality_axis", "reachable_pairs", "reachable_triples", "margin_in_bands",
    "margin_identifiable", "roofline_prefill_ms", "stratum_floor_ms",
    "roofline_decode_ms",
    "screen_boundaries",
)

LABEL_CONSTANTS = ()   # every branch in this file is categorical

AXES = {
    # did every arm reach MIN_BOOTS SCORED boots?  Two values: losing one of the
    # PLANNED boots is slack being spent, not a new world (E3).
    "coverage": ["sufficient", "insufficient"],
    # do the arms admit the SAME candidate set, on the STRICT screen? (if not,
    # the survivor is set by the worst arm and is an arm choice wearing a
    # coordinate's clothes).  `borderline` = the band decides (E2).
    "neutrality": ["neutral", "borderline", "arm_specific", "unmeasured"],
    # does the registered stratum agree with the alternative one?
    "stratum": ["agree", "disagree", "unmeasured"],
    # how many EXTERNAL candidates are reachable -- `borderline` = the band decides
    "anchor": ["unique", "multiple", "none", "borderline", "unmeasured"],
    # what the same screen does to CP-2's own registered 36-point ladder
    "ladder": ["intact", "pruned", "empty", "borderline", "unmeasured"],
}

AXIS_ORDER = ["coverage", "neutrality", "stratum", "anchor", "ladder"]

# ★Six of these are MEASURED FINDINGS, not failures: `FLOOR_NOT_ARM_NEUTRAL`,
# `STRATUM_DEPENDENT` and the four `ANCHOR_BORDERLINE*` say "the premise of route
# (a) is false", which is an answer.  rev2 of CP-2 was right to make `UNAFFORDABLE`
# substantive for the same reason (6th-audit S5).  What does NOT belong here is
# `VARIANCE_UNMEASURED`: that is a measurement failure, and `CONSENSUS.md` sec 3
# item 35 has been re-opened eight times by promoting exactly that kind of label.
SUBSTANTIVE = [
    "ANCHOR_REGISTERED",
    "ANCHOR_REGISTERED_LADDER_PRUNED",
    "ANCHOR_REGISTERED_LADDER_EMPTY",
    "ANCHOR_REGISTERED_LADDER_BORDERLINE",
    "ANCHOR_UNDERDETERMINED",
    "ANCHOR_UNDERDETERMINED_LADDER_PRUNED",
    "ANCHOR_UNDERDETERMINED_LADDER_BORDERLINE",
    "ANCHOR_INFEASIBLE_LADDER_PRUNED",
    "ANCHOR_INFEASIBLE_LADDER_BORDERLINE",
    "NOTHING_FEASIBLE",
    "ANCHOR_BORDERLINE",
    "ANCHOR_BORDERLINE_LADDER_PRUNED",
    "ANCHOR_BORDERLINE_LADDER_EMPTY",
    "ANCHOR_BORDERLINE_LADDER_BORDERLINE",
    "FLOOR_NOT_ARM_NEUTRAL",
    "FLOOR_NEUTRALITY_BORDERLINE",
    "STRATUM_DEPENDENT",
]

# ★The fork mapping, registered here, in code, before the run.  Its CONTENT is
# checked: every hand oracle row carries the branch this dict assigns, and the
# pre-registration's table is written in the same world-key + label format, so a
# one-line edit here fails the oracle instead of passing silently (rev1's D3).
FORK = {
    "ANCHOR_REGISTERED": "a",
    "ANCHOR_REGISTERED_LADDER_PRUNED": "a",
    "ANCHOR_REGISTERED_LADDER_EMPTY": "a_with_ladder_rederivation",
    "ANCHOR_REGISTERED_LADDER_BORDERLINE": "a_with_ladder_rederivation",
    "ANCHOR_UNDERDETERMINED": "b",
    "ANCHOR_UNDERDETERMINED_LADDER_PRUNED": "b",
    "ANCHOR_UNDERDETERMINED_LADDER_BORDERLINE": "b",
    "ANCHOR_INFEASIBLE_LADDER_PRUNED": "b",
    "ANCHOR_INFEASIBLE_LADDER_BORDERLINE": "b",
    "NOTHING_FEASIBLE": "cp2_regime_void",
    "ANCHOR_BORDERLINE": "b",
    "ANCHOR_BORDERLINE_LADDER_PRUNED": "b",
    "ANCHOR_BORDERLINE_LADDER_EMPTY": "b",
    "ANCHOR_BORDERLINE_LADDER_BORDERLINE": "b",
    "FLOOR_NOT_ARM_NEUTRAL": "b",
    # ★J1 gave this its own branch; K3 fixed what that branch SAYS.  "buy more
    # boots" is not purchasable here -- `boot_sd` is a sample SD, so the band
    # converges to 2*sigma, not to zero, and sec 3.3 forbids the purchase.  What
    # the noise actually bought is a statement about THIS precision.
    "FLOOR_NEUTRALITY_BORDERLINE": "b_precision_bound",
    "STRATUM_DEPENDENT": "b",
}
FORK_BRANCHES = ("a", "a_with_ladder_rederivation", "b", "b_precision_bound",
                 "cp2_regime_void", "n/a")

# Labels the rule can emit that are NOT findings.  They have no prescription, and
# `fork_branch` answers "n/a" rather than raising -- rev3 raised a KeyError on the
# two most likely failure labels (3rd-audit J4).
NON_SUBSTANTIVE = ("COVERAGE_INSUFFICIENT", "SCREEN_INCOMPLETE",
                   "STRATUM_UNMEASURED", "IMPOSSIBLE_WORLD")

# One entry per (anchor, ladder) pair the screen can actually produce.  The key
# set is required to EQUAL `af1_predicates.reachable_pairs()` in both directions
# (self-test C11), so a pair cannot be registered that the screen cannot reach
# (rev1 registered `("none", "intact")`) and a reachable pair cannot be left out.
OUTCOME = {
    ("unique", "intact"): "ANCHOR_REGISTERED",
    ("unique", "pruned"): "ANCHOR_REGISTERED_LADDER_PRUNED",
    ("unique", "empty"): "ANCHOR_REGISTERED_LADDER_EMPTY",
    ("unique", "borderline"): "ANCHOR_REGISTERED_LADDER_BORDERLINE",
    ("multiple", "intact"): "ANCHOR_UNDERDETERMINED",
    ("multiple", "pruned"): "ANCHOR_UNDERDETERMINED_LADDER_PRUNED",
    ("multiple", "borderline"): "ANCHOR_UNDERDETERMINED_LADDER_BORDERLINE",
    ("none", "pruned"): "ANCHOR_INFEASIBLE_LADDER_PRUNED",
    ("none", "borderline"): "ANCHOR_INFEASIBLE_LADDER_BORDERLINE",
    ("none", "empty"): "NOTHING_FEASIBLE",
    ("borderline", "intact"): "ANCHOR_BORDERLINE",
    ("borderline", "pruned"): "ANCHOR_BORDERLINE_LADDER_PRUNED",
    ("borderline", "empty"): "ANCHOR_BORDERLINE_LADDER_EMPTY",
    ("borderline", "borderline"): "ANCHOR_BORDERLINE_LADDER_BORDERLINE",
}

_DOWNSTREAM = ("neutrality", "stratum", "anchor", "ladder")


def _c_coverage(w):
    # Below MIN_BOOTS on any arm there is no between-boot SD, and the band IS the
    # SD -- so the screen is not a tighter screen, it is no screen.
    if w["coverage"] == "insufficient" and not all(
        w[k] == "unmeasured" for k in _DOWNSTREAM
    ):
        return ("no arm reached the minimum boots: there is no between-boot SD, "
                "so the screen has no band and cannot run")
    return None


# The readings that come from the REGISTERED stratum in one pass.  `stratum` is
# NOT one of them: it needs the alternative stratum too, so a boot that dies
# between the two leaves `stratum` unmeasured while these three exist.  rev3
# lumped all four together and called that world impossible (3rd-audit J4).
_ONE_PASS = ("neutrality", "anchor", "ladder")


def _c_one_pass(w):
    measured = [k for k in _ONE_PASS if w[k] != "unmeasured"]
    if measured and len(measured) != len(_ONE_PASS):
        return ("neutrality, anchor and ladder are three readings of ONE pass at "
                "the registered stratum: they cannot be measured separately")
    return None


def _c_stratum_needs_pass(w):
    # The alternative stratum is compared against the registered one, so a
    # measured `stratum` presupposes the registered stratum's readings.
    if w["stratum"] != "unmeasured" and w["anchor"] == "unmeasured":
        return ("the stratum axis compares the alternative stratum against the "
                "registered one, which was not measured")
    return None


def _c_image(w):
    # The screen's image, computed from the constants.  ★Over TRIPLES since rev4:
    # `neutrality` is an axis and checking only the pair left fourteen worlds
    # outside the image carrying a substantive label (3rd-audit J3).
    from af1_predicates import reachable_triples
    trip = (w["neutrality"], w["anchor"], w["ladder"])
    if "unmeasured" in trip:
        return None
    if trip not in reachable_triples():
        return ("%s is not in the image of the screen: no per-arm floors and band "
                "can produce it from the registered candidate and ladder values"
                % (trip,))
    return None


# Separately removable clauses.  af1_selftest.py disables each in turn and
# requires at least one label to move -- 6th-audit S1 recorded per-clause removal
# as the one structure this track has never broken, so it is kept and made
# explicit rather than buried in one function body.
INCOHERENCE = (
    ("coverage", _c_coverage),
    ("one_pass", _c_one_pass),
    ("stratum_needs_pass", _c_stratum_needs_pass),
    ("image", _c_image),
)


def incoherent(w, clauses=None):
    """Worlds the pipeline cannot produce.  PIPELINE FACTS, not design choices."""
    for _name, fn in (INCOHERENCE if clauses is None else clauses):
        msg = fn(w)
        if msg:
            return msg
    return None


class OutcomeGap(Exception):
    """A coherent world with no registered outcome.

    Raised, not defaulted: `OUTCOME`'s key set is required to equal the screen's
    image, so reaching this means that equality has broken and the run must stop
    rather than pick a label.
    """


# Ordered guards.  ORDER IS A DECISION: report the earliest failure in the
# pipeline, because that is the one that explains the others.
#
# ★`neutrality` precedes `stratum`, and both precede the outcome.  Neutrality is
# about the ESTIMAND -- whether an arm-neutral coordinate is definable at all --
# while `stratum` is about the MEASUREMENT.  An estimand failure explains more,
# so it is reported first.  Neither may be overruled by a measured anchor: an
# anchor derived from an arm-specific floor, or one that exists only at the
# stratum we happened to register, is the free parameter this probe exists to
# remove.
RULES = [
    ("coverage_insufficient", lambda w: w["coverage"] == "insufficient",
     "COVERAGE_INSUFFICIENT"),
    # Enough boots were scored and the screen still produced nothing -- a missing
    # stratum cell, an ITL undefined on some arm.  rev2 called these worlds
    # impossible and answered them with a KeyError (E3).
    # ★K2: the ONE-PASS readings only.  rev4 fired this on any unmeasured reading
    # including `stratum`, which demoted 21 worlds whose neutrality was fully
    # measured at the registered stratum to a prescription-less failure and made
    # the pre-registration's sec 6 ordering anchor false.
    ("screen_incomplete",
     lambda w: any(w[k] == "unmeasured" for k in _ONE_PASS),
     "SCREEN_INCOMPLETE"),
    ("floor_not_neutral", lambda w: w["neutrality"] == "arm_specific",
     "FLOOR_NOT_ARM_NEUTRAL"),
    ("floor_neutrality_borderline", lambda w: w["neutrality"] == "borderline",
     "FLOOR_NEUTRALITY_BORDERLINE"),
    # ★K2: only the ALTERNATIVE stratum is missing.  It comes AFTER the neutrality
    # guards, so an estimand failure that IS measured still reports as itself.
    ("stratum_unmeasured", lambda w: w["stratum"] == "unmeasured",
     "STRATUM_UNMEASURED"),
    ("stratum_disagree", lambda w: w["stratum"] == "disagree",
     "STRATUM_DEPENDENT"),
]


def label(w, clauses=None, rules=None):
    if incoherent(w, clauses):
        return "IMPOSSIBLE_WORLD"
    for _name, pred, lab in (RULES if rules is None else rules):
        if pred(w):
            return lab
    pair = (w["anchor"], w["ladder"])
    if pair not in OUTCOME:
        raise OutcomeGap(
            "no outcome for %s, yet the image derivation admits it: OUTCOME and "
            "af1_predicates.reachable_pairs() have diverged" % (pair,))
    return OUTCOME[pair]


def fork_branch(lab):
    """Which CP-2 rev3 design the label obliges.  Truly unknown labels raise."""
    if lab in NON_SUBSTANTIVE:
        return "n/a"
    return FORK[lab]
