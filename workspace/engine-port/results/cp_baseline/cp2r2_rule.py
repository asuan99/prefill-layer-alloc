"""CP-2 rev2: chunked prefill vs PD-mux on the long-prompt regime.  RULE_REV = 1.

What changed from rev1, and why (5th audit `audit_cp2_rules_5th_2026-09-01/VERDICT.md`)
--------------------------------------------------------------------------------------
rev1 was `NO-GO` on eight kill-shots.  rev2 does not patch them one at a time; three
of them turned out to be the same defect seen from different sides, and the repairs
are structural:

  H1 (margin below its own noise floor, arithmetic deferred)
      -> the margin is FIXED and external (3%, CLAUDE.md); the N is DERIVED from a
         measured variance by `cp2r2_predicates.required_n`.  A `power` axis carries
         the result, and `UNAFFORDABLE` is a real, reachable outcome that says what
         the budget COULD have answered.
  H2/H3/H8 (the ITL SLO sat inside a bimodal mode; the cliff was already unstable;
            COMBINED was dominated by the saturated phase)
      -> these are one fact, and W1 measured it directly on this regime: the two
         families are limited by OPPOSITE SLO legs (pdmux flat in ITL, steep in TTFT;
         chunked prefill the reverse).  So rev2 does not pick a threshold at all.
         `ladder` (rule E) is a PRIMARY axis and `inconsistent` is SUBSTANTIVE --
         it is the finding that the answer is a function of the operating point.
  H6 (arm selection outside code, cross-job, n=2)
      -> there is no selection stage.  The registered pair is fixed in this file.
  H7 (the stop rule's scope missed sections)
      -> the pre-registration quantifies over SECTIONS, not over a list of tests.
  H4/H5 (a false provenance claim, and gate misattribution)
      -> every constant's provenance is checked in this session, and gate citations
         name FILE + opening phrase.  ★This repository runs FOUR numbering systems
         (CLAUDE.md, PROJECT_STATUS "방법론 게이트", CONSENSUS sec 3, and the memory
         topic file); rev1's convention note said "두" and that miscount is itself a
         registered lesson.  Nothing here cites a bare number.

What this rule decides
----------------------
ONE thing: on the long-prompt regime, at rates at or above the measured knee, does
the chunked-prefill family serve more, fewer, or indistinguishably many requests
within SLO than the PD-mux family -- AND is that answer the same everywhere on the
registered SLO ladder.  Nothing else.

Scope this rule INHERITS and does not re-argue
----------------------------------------------
  * The engine forbids {pdmux ON, chunking ON} (`server_args.py`, "PD-Multiplexing
    is not compatible with chunked prefill."), so the arms differ in a BUNDLE and no
    mechanism attribution is licensed.  With piecewise capture off on every arm
    (DECISION_PIECEWISE_OFF_2026-09-02.md) the bundle is THREE, not four.
  * Correctness: 55/56 greedy outputs byte-identical against the `plain` reference,
    the single exception non-monotone in chunk count
    (RESULT_CORRECTNESS_GATE_2026-09-03.md).  ★Accepted by the user for this design.
    UNVERIFIED and inherited as a limit: whether CONCURRENCY changes outputs -- the
    gate sent one request at a time and the campaign runs under load.
  * The workload is the ShareGPT LONG-PROMPT TAIL (2.1% of the corpus), not ShareGPT.
"""

RULE_REV = 1

LABEL_CONSTANTS = ()          # every branch here is categorical; the continuous
                              # decisions live in cp2r2_predicates.py

# Campaign parameters, declared NOT to affect the label map.  Rates come from W1's
# MEASURED knee (jobs 900784/900785), not from the canonical 3/12 grid -- that grid
# belongs to a different workload and transplanting it is 4th-audit G2.
RATE_A_TRACK = (1.0, 2.0, 3.0)      # knee measured at 1.0-2.0 req/s
RATE_B_TRACK = (0.25, 0.5, 0.75)    # knee measured at 0.25-0.5 req/s
CAMPAIGN_PARAMS = ("RATE_A_TRACK", "RATE_B_TRACK")

PREDICATE_MODULE = "cp2r2_predicates.py"
PREDICATE_CONSTANTS = ("EQUIV_MARGIN_REL", "T975", "LADDER_TTFT_MS", "LADDER_ITL_MS")
PREDICATE_FOLDS = ("power_axis", "contrast_axis", "ladder_axis", "coverage_axis")

AXES = {
    # did both arms boot and serve on every planned pair?
    "boot": ["ok", "failed"],
    # is the registered margin answerable at the derived n within the budget cap?
    "power": ["sufficient", "unaffordable", "unmeasured"],
    # did the campaign complete the DERIVED number of pairs (not a hard-coded 4)?
    "coverage": ["complete", "partial", "absent"],
    # rule E: is the contrast the same at all 36 ladder points?
    "ladder": ["consistent", "inconsistent", "unmeasured"],
    # the estimand at the registered operating point, by TOST
    "contrast": ["cp_better", "pdmux_better", "equivalent", "inconclusive", "unmeasured"],
}

AXIS_ORDER = ["boot", "power", "coverage", "ladder", "contrast"]

SUBSTANTIVE = [
    "CP_BETTER",
    "PDMUX_BETTER",
    "EQUIVALENT",
    "INCONCLUSIVE",
    # ★The four SLO_DEPENDENT_* labels carry the operating-point direction IN THEIR
    # NAME alongside the fact that it does not generalise.  Collapsing them into one
    # label would lose the direction; dropping the prefix would let someone quote
    # "CP better" without the caveat.  Both failures have precedent in this track.
    "SLO_DEPENDENT_CP_AT_OP",
    "SLO_DEPENDENT_PDMUX_AT_OP",
    "SLO_DEPENDENT_EQUIVALENT_AT_OP",
    "SLO_DEPENDENT_INCONCLUSIVE_AT_OP",
]

OUTCOME = {
    ("consistent", "cp_better"): "CP_BETTER",
    ("consistent", "pdmux_better"): "PDMUX_BETTER",
    ("consistent", "equivalent"): "EQUIVALENT",
    ("consistent", "inconclusive"): "INCONCLUSIVE",
    ("inconsistent", "cp_better"): "SLO_DEPENDENT_CP_AT_OP",
    ("inconsistent", "pdmux_better"): "SLO_DEPENDENT_PDMUX_AT_OP",
    ("inconsistent", "equivalent"): "SLO_DEPENDENT_EQUIVALENT_AT_OP",
    ("inconsistent", "inconclusive"): "SLO_DEPENDENT_INCONCLUSIVE_AT_OP",
}


def incoherent(w):
    """Worlds the pipeline cannot produce.  PIPELINE FACTS, not design choices.

    ★The converse is deliberately NOT asserted: a run can have a contrast and still
    fail to produce a ladder value (the re-score did not run).  Declaring that
    impossible would make skipping rule E the cheapest route to a substantive label.
    """
    if w["boot"] == "failed" and not (
        w["coverage"] == "absent" and w["ladder"] == "unmeasured"
        and w["contrast"] == "unmeasured"
    ):
        return "boot failed: no downstream observable can exist"
    if w["coverage"] == "absent" and not (
        w["ladder"] == "unmeasured" and w["contrast"] == "unmeasured"
    ):
        return "no completed pair: nothing can be scored"
    if w["contrast"] == "unmeasured" and w["ladder"] != "unmeasured":
        return ("the operating-point contrast is unmeasured, so its re-score across "
                "the ladder cannot exist")
    return None


# Ordered guards.  ORDER IS A DECISION: report the EARLIEST failure in the pipeline,
# because that is the one that explains the others.
RULES = [
    ("boot_failed", lambda w: w["boot"] == "failed", "BOOT_FAILED"),
    ("coverage_absent", lambda w: w["coverage"] == "absent", "NOT_MEASURED"),
    # ★POWER comes BEFORE coverage_partial on purpose.  "we ran fewer pairs than the
    # plan" and "the plan itself could not answer the question at any affordable n"
    # are different facts, and the second one is not fixed by running more pairs.
    ("power_unmeasured", lambda w: w["power"] == "unmeasured", "POWER_UNMEASURED"),
    ("power_unaffordable", lambda w: w["power"] == "unaffordable", "UNAFFORDABLE"),
    ("coverage_partial", lambda w: w["coverage"] == "partial", "COVERAGE_INCOMPLETE"),
    ("contrast_unmeasured", lambda w: w["contrast"] == "unmeasured", "CONTRAST_UNMEASURED"),
    ("ladder_unmeasured", lambda w: w["ladder"] == "unmeasured", "LADDER_UNMEASURED"),
]


def label(w):
    if incoherent(w):
        return "IMPOSSIBLE_WORLD"
    for _name, pred, lab in RULES:
        if pred(w):
            return lab
    return OUTCOME[(w["ladder"], w["contrast"])]
