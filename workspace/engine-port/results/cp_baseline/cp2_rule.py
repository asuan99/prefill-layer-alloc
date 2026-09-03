"""CP-2 head-to-head rule: chunked prefill vs PD-mux on the canonical change trace.

RULE_REV = 1.

What this rule decides
----------------------
ONE thing: on the repository's canonical time-varying ShareGPT trace, does the
chunked-prefill family serve more, fewer, or indistinguishably many requests
within SLO than the PD-mux family -- and does that answer survive both (i) moving
the SLO threshold inside its registered band and (ii) looking at the two phases
separately.  Nothing else.  No mechanism attribution, no arm ranking beyond the
registered pair, no capacity claim.

★Why no mechanism attribution is even POSSIBLE here (register this first)
-------------------------------------------------------------------------
The engine forbids the crossed cell.  `sglang/srt/server_args.py` (dev tree
/scratch/ehmoon/whlee/sglang_engine_dev/python), inside `if self.enable_pdmux:`:

    assert (
        self.chunked_prefill_size == -1
    ), "PD-Multiplexing is not compatible with chunked prefill."

So {PD-mux ON, chunking ON} DOES NOT EXIST in this engine.  This is not a confound
a better design can balance away -- it is a positivity violation, the exact shape
PROJECT_STATUS.md registered as gate 83 after Nemotron-H+triton refused to boot
and Zamba2+flashinfer killed the scheduler ("엔진이 강제하는 상호배타 지원영역은
설계 선택으로 못 넘는 교락이다").

Consequently the two arms differ in a BUNDLE of at least four things, three of
them engine-forced:

  |                         | chunked-prefill arm | PD-mux arm      | forced? |
  | chunked prefill         | ON  (cps = N)       | OFF (cps = -1)  | yes     |
  | SM partitioning         | none                | green ctx split | yes     |
  | overlap schedule        | ON  (default)       | OFF             | yes     |
  | piecewise CUDA graph    | captured            | capture list 0  | yes*    |

  *the fourth is the F3 mechanism this track confirmed by observation on
   2026-09-01 (job 900053): `--chunked-prefill-size` sets the prefill budget AND
   the piecewise capture range at once, and at cps -1 the realized capture list
   had len 0.  Scope of that observation: Zamba2-2.7B, this tree, n=1 boot.

⇒ CP-2's estimand is a POLICY-FAMILY OPERATING-POINT COMPARISON ("which of these
two deployable configurations serves this trace better"), which is a real and
answerable operational question, and it is the one a reviewer asks first.  It is
NOT "does chunking beat SM splitting", which this engine cannot be asked.  Section
9 of the pre-registration forbids the second sentence in every form.

Why the axes are what they are
------------------------------
`contrast` is the estimand.  The other four axes exist because this track has been
told four times what happens without them:

  boot / coverage  -- gate 21, "측정 실패를 게이트 실패로 라벨링 마라".  A campaign
                      that half-ran is not a finding about either family.
  cliff            -- CLAUDE.md gate 6 and CONSENSUS 1-20: a goodput verdict taken
                      at one SLO threshold, when that threshold sits in the mass of
                      the TTFT distribution, moved a published headline from +0.00%
                      to +19.75% under a +-10% sweep.  CP-2 refuses to publish a
                      verdict that does not survive its own band.
  phase            -- the change trace exists BECAUSE LO and HI want different
                      splits.  A COMBINED number that hides a phase reversal erases
                      the structure the trace was built to expose (gate 5 asks for
                      per-phase reporting; this makes it a decision axis, not a
                      footnote).

Every data -> axis map is a function in `cp2_predicates.py` with registered
vectors.  `label()` is an ordered guard list followed by a TOTAL 4x2 outcome table
so `cp2_selftest.py` can enumerate and mutate every branch mechanically.

Gate citation convention: this repository runs TWO DISJOINT gate numbering
schemes -- the 8 in /scratch/ehmoon/whlee/CLAUDE.md and the "방법론 게이트" #1-#87
in PROJECT_STATUS.md.  Every citation here names the FILE and the opening phrase,
never a bare number, because rev2 of CP-1 mis-attributed six CLAUDE.md gates to
PROJECT_STATUS (2nd audit K6).  No new PROJECT_STATUS line coordinate is
introduced by this file: `check_version_sweep.py` rule S3 requires every such
coordinate in this directory to agree, and the one already registered here (5819)
is cited from the CP-0 files.
"""

RULE_REV = 1

# ---- constants that AFFECT the label map -------------------------------------
LABEL_CONSTANTS = ()      # none: every branch of this rule is categorical.  The
                          # continuous decisions all live in cp2_predicates.py and
                          # are mutation-tested there against registered vectors.

# ---- campaign parameters, declared NOT to affect the label map ---------------
# The trace itself.  These are INHERITED from ../slo_sched/sharegpt_vary_bench.sbatch
# (LO 3 / HI 12, 3 rounds, 200 prompts per round-phase) rather than chosen by CP-2 --
# 4th audit G2: a portable threshold measured on a grid the design picked for itself
# moves the defect instead of closing it.  CP-2 owns no rate grid.
RATE_LO = 3
RATE_HI = 12
ROUNDS = 3
NUM_PROMPTS = 200
CAMPAIGN_PARAMS = ("RATE_LO", "RATE_HI", "ROUNDS", "NUM_PROMPTS")

PREDICATE_MODULE = "cp2_predicates.py"
PREDICATE_CONSTANTS = ("EQUIV_MARGIN_REL", "N_MIN_REPS",
                       "TTFT_SLO_BAND_MS", "ITL_SLO_BAND_MS")
PREDICATE_FOLDS = ("coverage_axis", "phase_reversal_axis", "cliff_axis")

AXES = {
    # did every registered arm boot and serve the trace?
    "boot": ["ok", "failed"],
    # did every registered arm reach N_MIN_REPS independent boots?
    "coverage": ["complete", "partial", "absent"],
    # is the contrast label identical at all 9 points of the registered SLO band?
    "cliff": ["stable", "unstable", "unmeasured"],
    # THE ESTIMAND: paired relative Delta in SLO-goodput, COMBINED over both phases
    # with durations SUMMED (CLAUDE.md gate 7 -- "여러 라운드 합칠 때 duration은
    # 합산", the harness bug that inflated every reported goodput ~3x).
    "contrast": ["cp_better", "pdmux_better", "equivalent", "inconclusive",
                 "unmeasured"],
    # do the two phases point in OPPOSITE significant directions?
    "phase": ["no_reversal", "reversal", "unmeasured"],
}

AXIS_ORDER = ["boot", "coverage", "cliff", "contrast", "phase"]

SUBSTANTIVE = [
    "CP_BETTER",
    "CP_BETTER_PHASE_REVERSAL",
    "PDMUX_BETTER",
    "PDMUX_BETTER_PHASE_REVERSAL",
    "EQUIVALENT",
    "EQUIVALENT_PHASE_REVERSAL",
    "INCONCLUSIVE",
    "INCONCLUSIVE_PHASE_REVERSAL",
]

# 4x2 outcome table over (contrast, phase).  Total on the product of the measured
# values, so no world falls through.  All eight labels are DISTINCT, which is what
# makes the permutation theorem in the self-test applicable (cp0's g1 rule had to
# note the opposite case).
OUTCOME = {
    ("cp_better", "no_reversal"): "CP_BETTER",
    ("cp_better", "reversal"): "CP_BETTER_PHASE_REVERSAL",
    ("pdmux_better", "no_reversal"): "PDMUX_BETTER",
    ("pdmux_better", "reversal"): "PDMUX_BETTER_PHASE_REVERSAL",
    # A COMBINED `equivalent` with opposite phase directions is not a null result --
    # it is the two phases CANCELLING, which is the single most interesting outcome
    # the change trace can produce and the one a COMBINED-only design would erase.
    ("equivalent", "no_reversal"): "EQUIVALENT",
    ("equivalent", "reversal"): "EQUIVALENT_PHASE_REVERSAL",
    ("inconclusive", "no_reversal"): "INCONCLUSIVE",
    ("inconclusive", "reversal"): "INCONCLUSIVE_PHASE_REVERSAL",
}


def incoherent(w):
    """Worlds the measurement pipeline cannot produce.

    These are PIPELINE FACTS, not design choices: the phase contrasts and the SLO-band
    re-scores are computed from the same scored artifacts as the combined contrast,
    so nothing downstream can exist when the thing upstream does not.  ★The converse
    is NOT asserted: a run can have a combined contrast and still fail to produce a
    phase value (a phase whose CI is undefined) or a cliff value (the band was not
    swept).  Those worlds are REAL and are guarded to their own labels below -- if
    they were declared impossible instead, skipping the expensive check would become
    the cheapest way to a substantive label.
    """
    if w["boot"] == "failed" and not (
        w["coverage"] == "absent" and w["cliff"] == "unmeasured"
        and w["contrast"] == "unmeasured" and w["phase"] == "unmeasured"
    ):
        return "boot failed: no downstream observable can exist"
    if w["coverage"] == "absent" and not (
        w["cliff"] == "unmeasured" and w["contrast"] == "unmeasured"
        and w["phase"] == "unmeasured"
    ):
        return "no completed cell: nothing can be scored"
    if w["contrast"] == "unmeasured" and not (
        w["cliff"] == "unmeasured" and w["phase"] == "unmeasured"
    ):
        return ("the combined contrast is unmeasured, so neither its per-phase "
                "decomposition nor its re-score across the SLO band can exist")
    return None


# Ordered guards.  Kept as DATA so the self-test can drop, negate and transpose each
# one mechanically (2nd audit K4: a hand-authored mutation list covers only what its
# author already had in mind, and 9 of 14 external mutants survived rev2).
#
# ORDER IS A DECISION, and it is this one: report the EARLIEST failure in the
# pipeline, because that is the one that explains the others.  A half-run campaign
# whose contrast could not be computed is a coverage failure, not a scoring failure.
RULES = [
    ("boot_failed", lambda w: w["boot"] == "failed", "BOOT_FAILED"),
    ("coverage_absent", lambda w: w["coverage"] == "absent", "NOT_MEASURED"),
    ("coverage_partial", lambda w: w["coverage"] == "partial", "COVERAGE_INCOMPLETE"),
    ("contrast_unmeasured", lambda w: w["contrast"] == "unmeasured", "CONTRAST_UNMEASURED"),
    ("phase_unmeasured", lambda w: w["phase"] == "unmeasured", "PHASE_UNMEASURED"),
    ("cliff_unmeasured", lambda w: w["cliff"] == "unmeasured", "CLIFF_UNMEASURED"),
    # ★CLIFF_UNSTABLE is deliberately NON-substantive.  "The verdict flips inside its
    # own SLO band" is real information and the pre-registration requires publishing
    # it, but it is not a statement about either policy family, and letting it count
    # as one would let instability buy a headline.  CLAUDE.md gate 6 exists because
    # this repository already published a number that did exactly that.
    ("cliff_unstable", lambda w: w["cliff"] == "unstable", "CLIFF_UNSTABLE"),
]


def label(w):
    if incoherent(w):
        return "IMPOSSIBLE_WORLD"
    for _name, pred, lab in RULES:
        if pred(w):
            return lab
    return OUTCOME[(w["contrast"], w["phase"])]
