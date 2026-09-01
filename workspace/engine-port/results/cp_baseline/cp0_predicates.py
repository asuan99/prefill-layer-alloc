"""CP-0 data -> axis maps, as total functions.  RULE_REV-independent module.

3rd audit F1 killed the prose versions of these:

    saturated  ==  "achieved request throughput < offered rate"      (no tolerance)
    binds      ==  "mean extend tokens per prefill forward differs by >= 3%"

The first is a NOISE INDICATOR, not a saturation test.  Under Poisson pacing the
benchmark duration spans the ramp and the drain tail, so achieved < offered even
far below capacity: on the repository's own capacity scan the predicate fires at
offered rate 1 (seed27 achieved 0.86).  The second was never coded at all --
`BATCH_MARGIN_REL` sat in the rule file as a `CAMPAIGN_PARAM` whose "does not
touch the label map" check was an identity, because nothing read it.

Both now live here, as functions, with their constants derived from measurements
that already exist in this repository and with the derivation written down.

Why a separate module
---------------------
The label functions in `cp0_*_rule.py` map AXIS VALUES to labels; these map DATA to
axis values.  Keeping them apart makes the two mutation surfaces distinct and both
testable: `cp0_selftest.py` mutates the label guards/outcome cells/orderings AND
mutates the constants here against registered vectors.  A constant that survives
its own mutation test is dead, and the test says so.
"""

# The module owns its own classification.  4th audit L-d: rev3 asked each RULE to
# name the predicate constants it used, so `ATTAINMENT_THRESHOLD` looked unclassified
# from the arm rule and `BATCH_MARGIN_REL` from the capacity rule, and the two infra
# constants were named nowhere.  The shared module declares them once.
LABEL_CONSTANTS = ("ATTAINMENT_THRESHOLD", "BATCH_MARGIN_REL")   # must be load-bearing
INFRA_CONSTANTS = ("FLOAT_TOL", "PROBE_RATES")                   # must NOT change a
# classification on their own: FLOAT_TOL only widens an exact boundary, PROBE_RATES
# selects which offered rates are measured and is consumed by the harness, not by
# these functions.  Both are checked for that property, not merely declared.

# ---------------------------------------------------------------- saturation
# The invariant we lean on is NOT achieved throughput (which is non-monotone near
# the plateau -- `plain` measures 4.39 / 4.30 / 4.23 at rates 8 / 10 / 12) but the
# ATTAINMENT RATIO achieved/offered, which is monotone decreasing in every arm of
# the repository's capacity scan:
#
#   g2ea PHASE 0 capscan, Zamba2-2.7B, mean over 3 seeds (jobs 875657/875661)
#   rate        1     2     3     4     5     6     8    10    12
#   agnostic  0.96  0.95  0.93  0.88  0.81  0.72  0.55  0.45  0.37
#   chunk512  0.96  0.95  0.87  0.71  0.56  0.48  0.36  0.29  0.24
#   plain     0.96  0.96  0.94  0.89  0.82  0.71  0.55  0.43  0.35
#
# Each arm's measured plateau (agnostic 4.46 / chunk512 2.89 / plain 4.39) is first
# reached at the rate where its ratio falls into [0.71, 0.72]; the last rate that is
# still climbing has ratio in [0.81, 0.87].  So the knee lies inside [0.72, 0.81]
# for ALL THREE arms, and 0.80 is the round value inside that band.
#
# SCOPE (must accompany any citation): that scan is `random-ids in2000/out96, N=60`
# -- a different workload from the ShareGPT varying trace.  The band is used to
# CHOOSE a threshold, never to predict a capacity.  A ratio-based threshold is
# workload-portable in a way an absolute req/s number is not, which is the reason
# for choosing this form.
ATTAINMENT_THRESHOLD = 0.80

# Probe rates.  Registered from the same scan plus canon: the slowest arm there
# plateaus at 2.89 req/s and the fastest at 4.46, while CONSENSUS §1-32 puts the
# canonical ShareGPT HI (12) at 2.15-2.35x achieved capacity, i.e. capacity ~5.3.
# {2, 4, 8} brackets every one of those: rate 2 is still tracking for the slowest
# arm measured (0.95) and rate 8 is past the knee for the fastest (0.55), with 4
# sitting where the two separate (chunk512 0.71 vs plain 0.89).
# ★The 3rd audit killed the previous grid {3, 6, 12}: rate 3 is at or past the knee
# for chunk512 (plateau 2.89), so the lowest probe rate could not be a tracking
# point and CAPACITY_BRACKETED was close to unreachable by construction.
PROBE_RATES = (2, 4, 8)


# Float safety.  The registered boundary vector (achieved 2.40, offered 3.0) is
# exactly at the threshold by construction, but 2.40 / 3.0 == 0.7999999999999999 in
# binary floating point, so a bare `>=` classifies the boundary as `saturated`.
# This is the same failure the repository already registered against NSL
# (`CONSENSUS.md` §3, the `1.0-0.80 == 0.19999999999999996` finding: a decision that
# turns on float representation is not a decision).  The comparison is therefore
# explicit and its tolerance registered.  ★This bug was caught by this module's own
# RATE_VECTORS on their first run, which is the reason those vectors include a
# boundary case at all.
FLOAT_TOL = 1e-9


def _ge(a, b):
    return a >= b - FLOAT_TOL


def attainment(achieved_rps, offered_rps):
    if offered_rps <= 0:
        raise ValueError(f"offered rate must be positive, got {offered_rps}")
    if achieved_rps < 0:
        raise ValueError(f"achieved rate cannot be negative, got {achieved_rps}")
    return achieved_rps / offered_rps


def rate_axis(achieved_rps, offered_rps, threshold=None):
    """Total map from one (achieved, offered) pair to a `cp0_capacity_rule` axis value.

    `tracking`  -- the server is still keeping up: ratio >= threshold
    `saturated` -- it is not
    Monotonicity of the ratio means these partition the probe rates into a tracking
    prefix and a saturated suffix, so 'bracketed' is well defined without needing a
    separate plateau test (which is what inverted on `plain`).
    """
    t = ATTAINMENT_THRESHOLD if threshold is None else threshold
    return "tracking" if _ge(attainment(achieved_rps, offered_rps), t) else "saturated"


# ------------------------------------------------------------- batch channel
# Relative difference in mean extend tokens per prefill forward, arm vs fused_mono,
# paired by rep.  The margin reuses the project's standing equivalence margin
# (/scratch/ehmoon/whlee/CLAUDE.md gate #3, "3% 미만 차이는 headline 아님") rather
# than inventing a constant.  `binds` additionally requires the paired CI to exclude
# zero, so noise alone cannot buy the label (the 3rd audit's F1(b) note that this
# mean is a rate-mixture is handled in the pre-registration by reporting it PER
# PROBE RATE, not by widening the margin).
BATCH_MARGIN_REL = 0.03


def batch_axis(arm_mean, ref_mean, ci_lo, ci_hi, margin=None):
    """Total map to a `cp0_arm_rule` batch_channel value.

    ci_lo / ci_hi bound the PAIRED difference (arm - ref) in the same units as the
    means.  `binds` needs both a relative difference at or over the margin and a CI
    that excludes 0; anything else is `slack`.
    """
    m = BATCH_MARGIN_REL if margin is None else margin
    if ref_mean <= 0:
        raise ValueError(f"reference mean must be positive, got {ref_mean}")
    if ci_lo > ci_hi:
        raise ValueError(f"malformed CI: {ci_lo} > {ci_hi}")
    rel = abs(arm_mean - ref_mean) / ref_mean
    excludes_zero = ci_lo > 0 or ci_hi < 0
    return "binds" if (_ge(rel, m) and excludes_zero) else "slack"


# --------------------------------------------------------- folding over rates
# 4th audit G6 / F1(b): sections 4.1 and 4.2 register the two channels PER PROBE
# RATE, but `cp0_arm_rule` carries ONE axis value per arm.  rev3 left the fold
# unwritten, which is a free parameter on the decision path.  It is written here,
# and the two channels fold differently ON PURPOSE:
#
#   channel 1 is an OBSERVATION, not a test.  "did any request get chunked" has no
#   null hypothesis and no sampling distribution -- it either happened or it did
#   not -- so a disjunction over rates carries no multiple-comparison inflation.
#   (Contrast PROJECT_STATUS.md "방법론 게이트" #24, "`any()` over n reps 형태의
#   스크린은 귀무 발화율이 1-(1-alpha)^n": that gate governs SCREENS BUILT ON
#   TESTS.  Counting is not a test.)
#
#   channel 2 IS a test (relative difference + paired CI), so a disjunction over
#   rates WOULD inflate.  It folds by CONJUNCTION instead: `binds` requires every
#   probe rate to bind.  A conjunction's null firing rate is bounded by the single-
#   test alpha, so no correction is needed -- the multiplicity is spent on being
#   conservative rather than on a Holm ladder.  The middle case is kept rather than
#   collapsed, because "the cap binds only under backlog" is the outcome the
#   mechanism actually predicts and folding it into `slack` would erase it.
def req_axis_fold(per_rate_counts):
    """channel 1: OBSERVATION.  `active` iff any probe rate chunked >= 1 request."""
    if not per_rate_counts or any(c is None for c in per_rate_counts):
        return "unmeasured"
    return "active" if any(c >= 1 for c in per_rate_counts) else "inactive"


def batch_axis_fold(per_rate_values):
    """channel 2: TEST.  Conjunction over probe rates; `mixed` is kept, not collapsed."""
    vals = list(per_rate_values)
    if not vals or any(v == "unmeasured" or v is None for v in vals):
        return "unmeasured"
    if all(v == "binds" for v in vals):
        return "binds"
    if all(v == "slack" for v in vals):
        return "slack"
    return "mixed"


# Registered vectors.  `cp0_selftest.py` asserts each expectation AND that
# perturbing the corresponding constant changes at least one of them -- that is how
# a constant is shown to be load-bearing rather than declared to be.
RATE_VECTORS = [
    # (achieved, offered, expected)              provenance
    (1.90, 2.0, "tracking"),    # capscan chunk512 @2  ratio 0.95
    (2.84, 4.0, "saturated"),   # capscan chunk512 @4  ratio 0.71, at plateau 2.89
    (3.57, 4.0, "tracking"),    # capscan plain    @4  ratio 0.89, still climbing
    (4.39, 8.0, "saturated"),   # capscan plain    @8  ratio 0.55, at plateau
    (2.40, 3.0, "tracking"),    # ratio 0.80 exactly -- boundary inclusive; 2.40/3.0
                                #   is 0.7999999999999999 in float, see FLOAT_TOL
]

BATCH_VECTORS = [
    # (arm_mean, ref_mean, ci_lo, ci_hi, expected)
    (400.0, 1000.0, -700.0, -500.0, "binds"),   # -60%, CI excludes 0
    (1020.0, 1000.0, 5.0, 35.0, "slack"),       # +2% < 3% margin, CI excludes 0
    (700.0, 1000.0, -600.0, 50.0, "slack"),     # -30% but CI contains 0
    (970.0, 1000.0, -55.0, -5.0, "binds"),      # -3% exactly, CI excludes 0
    # ★CI boundary (4th audit L-g: RATE_VECTORS had a boundary case and these did
    # not, so a mutant that read `excludes_zero` as `>= 0` survived).  A CI whose
    # lower bound IS zero does not exclude zero.
    (500.0, 1000.0, 0.0, 35.0, "slack"),
    (500.0, 1000.0, -35.0, 0.0, "slack"),
]

FOLD_VECTORS = [
    # (kind, per_rate, expected)
    ("req", [0, 0, 3], "active"),        # backlog only at the top rate -- still an observation
    ("req", [0, 0, 0], "inactive"),
    ("req", [1, None, 2], "unmeasured"),
    ("batch", ["binds", "binds", "binds"], "binds"),
    ("batch", ["slack", "slack", "slack"], "slack"),
    ("batch", ["slack", "binds", "binds"], "mixed"),      # binds only under backlog
    ("batch", ["binds", "unmeasured", "binds"], "unmeasured"),
]
