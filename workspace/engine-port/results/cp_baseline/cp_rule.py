"""CP (chunked-prefill) baseline campaign -- decision rule, rev2.

STATUS: RULE_REV=2, **still not a pre-registration**.  rev1 (`cp_rule_draft.py`,
RULE_REV=1) was audited on 2026-08-28 and returned `NO-GO` with nine kill-shots
(`audit_cp_rules_2026-08-28/VERDICT.md`).  This file answers D1, D2, D3, D4, D7,
D8, D9 and L10; the campaign-shape answers (D5, D6) live in
`DESIGN_CP_BASELINE_REV2_2026-08-28.md` and `served_population.json`.

What changed, and which kill-shot each change answers
-----------------------------------------------------
D1  `ci_shape` was a set of prose predicates that the rule claimed were
    "exhaustive and mutually exclusive by construction" and were neither.  It is
    now a TOTAL FUNCTION of `(lo, hi, delta)` -- `ci_shape()` below -- built as
    the product of two total, mutually exclusive classifications (sign x
    magnitude).  `cp_rule_selftest.py` proves totality and exclusivity by
    counterexample search and kills every branch with a mutation test.
D2  The rev1 decision quantity was `goodput(best CP arm) - goodput(d44)`: a
    post-hoc argmax over three arms that was not an axis, so the reachability
    certificate could not see it.  There is no argmax now.  Four contrasts are
    registered by name (see CONTRASTS), all four are reported whatever they say,
    and family-wise error is controlled by Holm across the four.
D3  `ladder` read point-estimate SIGNS, so noise became the substantive label
    `LADDER_UNSTABLE` and a 0.001 flip at a secondary ladder point could delete a
    real result.  It now reads CIs (`concordant` / `conflicting` /
    `uninformative`), and the power check runs BEFORE it.  The null firing rate of
    the conflict screen is computed in the self-test, as CONSENSUS §3 item 38
    ("an `any()`-over-n screen has null firing rate 1-(1-alpha)^n") requires.
D4  rev1's single cross-substrate contrast changed pdmux, overlap-schedule and
    chunking at once, so the mechanism question (§3.2) was not identified.  The
    E1 family below is single-lever: `cpN` and `fused_mono` differ in
    `--chunked-prefill-size` and nothing else.
D7  `capacity` was a per-world scalar while the estimand spans two arms.  It is
    now defined on the CONTRAST and is parameter-free: `below` iff the offered HI
    rate is at or under the measured achieved throughput of BOTH arms.
D8  The CI estimator was unnamed.  It is the paired t-CI (df = n-1), not the
    percentile bootstrap: PROJECT_STATUS "next experiment gate" #8 registers that
    bootstrap's undercoverage (n=4 0.798 / n=6 0.859 / n=8 0.888, CONSENSUS.md
    :1472-1473) as a SUBMISSION PRECONDITION, and its prescription was to replace
    the estimator, not to raise n.
D9  CP-1b is unconditional, so no label gates a spend.  This file contains no
    spend condition at all.
L10 rev1 folded `pos_lt_delta` into a plain `CP_WINS`, erasing the reason delta
    exists (PROJECT_STATUS "methodology gate" #3: "a difference under 3% is not a
    headline").  Sub-delta wins now carry their own label, `CP_WINS_SUBDELTA`.

Registered thresholds -- every one a number, with where the number comes from
----------------------------------------------------------------------------
delta            = 0.03 * goodput(reference arm of the contrast).
                   PROJECT_STATUS "methodology gate" #3.
REQ_SPLIT_MIN_N  = 6 requests of the 200 served.
                   = ceil(delta * N) = ceil(0.03 * 200).  An arm whose
                   request-channel activity touches fewer requests than the
                   headline threshold itself cannot move the headline through
                   that channel.  Derived, not chosen: it is fixed by delta and
                   N, both of which were registered before `served_population.json`
                   existed.
BATCH_MARGIN_REL = 0.03 relative difference in mean extend tokens per prefill
                   forward vs `fused_mono`, paired CI excluding 0.  Reuses the
                   project's standing equivalence margin (the HOLB G5 / gate #3
                   3% convention) rather than inventing a new constant.
ALPHA            = 0.05 two-sided, Holm-corrected across the four contrasts.
N_REP            = 10 paired reps per arm.  Cost basis: `sacct` over this
                   harness (n=34 completed runs, mean 8.0 min, range 7.2-11.9),
                   NOT the sbatch `--time` ceiling -- rev1 back-computed from the
                   ceiling and overstated the budget ~3x, which is what made rev1
                   choose n=6 and make CP-1b conditional.

Gate citations in this file give number + document + opening phrase, because the
repository runs three separate numbering schemes (PROJECT_STATUS "methodology
gate" #N, CONSENSUS §3 item N, and the memory topic file's own item N) and rev1
mixed them inside the rule file itself (audit L1).
"""

import math

RULE_REV = 2

# ---------------------------------------------------------------- thresholds
DELTA_REL = 0.03          # PROJECT_STATUS "methodology gate" #3
REQ_SPLIT_MIN_N = 6       # = ceil(DELTA_REL * 200 served requests)
BATCH_MARGIN_REL = 0.03   # project standing equivalence margin
ALPHA = 0.05
N_REP = 10
LADDER_POINTS = 6         # TTFT {1.0, 3.0} s  x  ITL-p95 {50, 60, 80} ms

# ------------------------------------------------------------------ estimands
# Four contrasts, all registered by name, all reported regardless of outcome,
# Holm-corrected as one family.  No selection step anywhere -- that is D2.
#
#   family E1 (mechanism, single lever: only --chunked-prefill-size differs)
#     C1  cp512   - fused_mono        <- PRIMARY
#     C2  cp1024  - fused_mono
#   family E2 (deployable default: what a fused deployment actually gets)
#     C3  cp512   - fused_default     (fused_default = no cps flag => 8192 on A100-80GB)
#   family E3 (external baseline parity, CROSS-SUBSTRATE -- mandatory scope banner)
#     C4  cp512   - d44
CONTRASTS = {
    "C1": {"arm": "cp512", "ref": "fused_mono", "family": "E1", "primary": True},
    "C2": {"arm": "cp1024", "ref": "fused_mono", "family": "E1", "primary": False},
    "C3": {"arm": "cp512", "ref": "fused_default", "family": "E2", "primary": False},
    "C4": {"arm": "cp512", "ref": "d44", "family": "E3", "primary": False,
           "scope_banner": "cross-substrate: pdmux, --disable-overlap-schedule and "
                           "chunking all differ (server_args.py:6125-6131 forbids "
                           "pdmux with chunked prefill), so no single-mechanism "
                           "attribution may be read off this contrast"},
}

# ----------------------------------------------------------------------- axes
AXES = {
    # positive control channel 1, as a COUNT over the deterministic 200-request
    # served set (served_population.json).  Percentages are meaningless here:
    # --seed is fixed, so the served set is a constant, not a sample.
    "req_split_n_ge_min": ["yes", "no"],
    # positive control channel 2: does the per-batch token cap bind?
    "batch_budget": ["binds", "slack"],
    # property of the CONTRAST, not of one arm: `below` iff offered HI rate is at
    # or under the measured achieved throughput of BOTH arms (CP-0 stage (e)).
    "capacity": ["below", "above"],
    # total function of (lo, hi, delta); see ci_shape().
    "ci_shape": [
        "pos_ge_delta", "pos_subdelta", "pos_indet",
        "neg_ge_delta", "neg_subdelta", "neg_indet",
        "within_delta", "straddles",
    ],
    # CI-based, not sign-based.
    "ladder": ["concordant", "conflicting", "uninformative"],
}

AXIS_ORDER = ["req_split_n_ge_min", "batch_budget", "capacity", "ci_shape", "ladder"]

SUBSTANTIVE = [
    "LADDER_CONFLICT",
    "CP_WINS_RANK_ONLY", "CP_LOSES_RANK_ONLY", "CP_EQUIV_RANK_ONLY",
    "CP_WINS_GE_DELTA", "CP_WINS_SUBDELTA", "CP_WINS_SIZE_UNRESOLVED",
    "CP_LOSES_GE_DELTA", "CP_LOSES_SUBDELTA", "CP_LOSES_SIZE_UNRESOLVED",
    "CP_EQUIV_SIZED",
]

_SIGN_TO_DIR = {"pos": "WINS", "neg": "LOSES", "zero": "EQUIV"}
_MAG_SUFFIX = {"ge_delta": "GE_DELTA", "subdelta": "SUBDELTA", "indet": "SIZE_UNRESOLVED"}


# ------------------------------------------------------- the data -> axis map
def ci_shape(lo, hi, delta):
    """Map a paired CI to exactly one `ci_shape` axis value.  Total on
    {(lo, hi) : lo <= hi} for any delta > 0.

    Built as the product of two classifications that are each total and mutually
    exclusive, so totality and exclusivity are structural rather than asserted:

      sign       pos  : lo > 0            (strict -- a CI touching 0 establishes
                 neg  : hi < 0             no direction)
                 zero : otherwise (CI contains 0)

      magnitude  ge   : CI lies entirely outside (-delta, +delta)
                        i.e. lo >= delta or hi <= -delta   (non-strict: a lower
                        bound exactly at delta does establish |D| >= delta)
                 subdelta : CI lies entirely inside (-delta, +delta)
                        i.e. lo > -delta and hi < delta    (strict)
                 indet: neither -- the CI crosses a +/-delta boundary, so the
                        sign may be settled while the size is not

    (zero, ge_delta) is unreachable: a CI containing 0 meets (-delta, delta).  The
    remaining eight cells are the eight axis values.  `pos_indet` is the cell
    rev1 had no value for at all, and it is the shape of this project's most-cited
    effect: d44 - bind+GATE = 0.088 with CI [+0.068, +0.108] against
    delta = 0.03 * 3.220 = 0.0966 (CONSENSUS §1-7).
    """
    if not (lo <= hi):
        raise ValueError(f"malformed CI: lo={lo} > hi={hi}")
    if not (delta > 0):
        raise ValueError(f"delta must be positive, got {delta}")

    if lo > 0:
        sign = "pos"
    elif hi < 0:
        sign = "neg"
    else:
        sign = "zero"

    if lo >= delta or hi <= -delta:
        mag = "ge_delta"
    elif lo > -delta and hi < delta:
        mag = "subdelta"
    else:
        mag = "indet"

    if sign == "zero":
        # (zero, ge_delta) is unreachable; see docstring.
        return "within_delta" if mag == "subdelta" else "straddles"
    return f"{sign}_{mag}"


def ladder_shape(cis, delta):
    """Map the per-ladder-point CIs to exactly one `ladder` axis value.

    `cis` is the list of (lo, hi) over the LADDER_POINTS SLO settings, one of
    which is the primary operating point (TTFT 3.0 s, ITL-p95 60 ms).  Only CIs
    that exclude 0 vote, which is the D3 repair: rev1 compared point-estimate
    signs, so a 0.001 flip at a secondary point could delete a real result and a
    powerless arm could manufacture a substantive label.
    """
    pos = any(lo > 0 for lo, _ in cis)
    neg = any(hi < 0 for _, hi in cis)
    if pos and neg:
        return "conflicting"
    if pos or neg:
        return "concordant"
    return "uninformative"


def ladder_conflict_null_rate(n_points=LADDER_POINTS, alpha=ALPHA):
    """P(`conflicting` | true effect is 0 at every ladder point), assuming the
    points are independent.

    Registered because CONSENSUS §3 item 38 (PROJECT_STATUS "methodology gate"
    #24, "an `any()`-over-n screen has null firing rate 1-(1-alpha)^n and hands
    out citation eligibility essentially at random") requires an OR-shaped screen
    to publish its null firing rate before it is used.

    Independence is the CONSERVATIVE direction here: the ladder points are six
    re-scorings of the SAME reps, so they are strongly positively correlated, and
    positive correlation makes a sign CONFLICT rarer, not commoner.  So this is an
    upper reference, not an estimate.
    """
    p = alpha / 2.0
    return 1.0 - 2.0 * (1.0 - p) ** n_points + (1.0 - 2.0 * p) ** n_points


# ------------------------------------------------------------ world coherence
def incoherent(w):
    """Axis combinations that cannot co-occur, named rather than silently counted.

    The reachability tool takes the free product of the axes, so without this the
    grid contains worlds the experiment cannot produce and their counts would read
    as if they were possible outcomes.  The 3rd A1 audit's R1 finding was exactly
    a world model that did not cover its own design.

    Constraint: the primary operating point is ONE OF the ladder points.  So if no
    ladder point's CI excludes 0 (`uninformative`), the primary CI does not exclude
    0 either, and `ci_shape` must be `within_delta` or `straddles`.
    """
    if w["ladder"] == "uninformative" and w["ci_shape"] not in ("within_delta", "straddles"):
        return "primary point is a ladder point: it cannot exclude 0 while no ladder point does"
    return None


# ------------------------------------------------------------------ the label
def label(w):
    # 0. Worlds the design cannot produce are named, not counted as outcomes.
    if incoherent(w):
        return "IMPOSSIBLE_WORLD"

    # 1. Arm identity.  An arm that chunks on neither channel is `fused_mono`
    #    under another name, and scoring it would be scoring an identity.
    if w["req_split_n_ge_min"] == "no" and w["batch_budget"] == "slack":
        return "DEGENERATE_ARM"

    # 2. Power FIRST (D3).  rev1 checked the ladder before this, so a CI that
    #    separates nothing could still be cashed in as a substantive statement
    #    about the SLO threshold.  A contrast with no power says nothing about
    #    anything, including the metric.
    if w["ci_shape"] == "straddles":
        return "UNDERPOWERED"

    # 3. Direction disagreement across the ladder, now CI-based, so reaching here
    #    requires at least two ladder points that each exclude 0 with opposite
    #    signs.  That is a statement about the SLO threshold -- which is one of
    #    the two payoffs this campaign is built to buy (arm-dependent ITL cliff,
    #    CONSENSUS §3 item 89) -- and it blocks a policy verdict.
    if w["ladder"] == "conflicting":
        return "LADDER_CONFLICT"

    sign, _, mag = w["ci_shape"].partition("_")
    if w["ci_shape"] == "within_delta":
        direction, mag = "EQUIV", "subdelta"
    else:
        direction = _SIGN_TO_DIR[sign]

    # 4. Above the arms' achieved capacity the threshold-goodput MAGNITUDE is
    #    ill-posed (PROJECT_STATUS "methodology gate" #6, "measure goodput away
    #    from the metric cliff; measure capacity first"; CONSENSUS §1-32 records
    #    the canonical HI phase at offered/achieved 2.15-2.35x).  The ORDER
    #    survives -- canon cites the HE0 ranking from exactly this operating point
    #    (CONSENSUS §1-7) -- so capacity gates the size claim only.  Letting it
    #    gate the whole verdict makes the campaign NOTHING_PURCHASABLE before any
    #    datum arrives; that variant was built and tested
    #    (`cp_rule_rejected_capacity_gates_all.py`).
    if w["capacity"] == "above":
        return f"CP_{direction}_RANK_ONLY"

    if direction == "EQUIV":
        return "CP_EQUIV_SIZED"
    return f"CP_{direction}_{_MAG_SUFFIX[mag]}"
