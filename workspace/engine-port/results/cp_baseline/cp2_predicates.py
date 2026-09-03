"""CP-2 data -> axis maps, as total functions.  RULE_REV-independent module.

CP-2 asks the question this track was opened for and has not yet measured once:
**on the canonical change trace, which policy family serves better -- chunked
prefill or PD-mux?**  It is a HEAD-TO-HEAD at an operating point.  It is not a
mechanism attribution, and section 1 of PREREG_CP2_2026-09-01.md registers why it
cannot be one (the engine forbids the crossed cell; see `FORCED_BUNDLE` below).

Why the axis maps live here and not in prose
--------------------------------------------
The 3rd audit killed CP-0's prose predicates (F1) and the 4th audit killed the
one that was left free (G6: "rate별 추정량에 추출 경로·접기 규칙·수용 시험이 없다").
Every data -> axis map CP-2 uses is a function in this file, with registered
vectors, and `cp2_selftest.py` mutates each constant against those vectors.

Provenance discipline (4th audit G2, the lesson that outlived the grid)
-----------------------------------------------------------------------
G2's finding was that making a THRESHOLD portable (a ratio) buys nothing if the
GRID that threshold is measured on is still in absolute units -- the defect moves,
it does not close.  CP-2 answers it by not owning a rate grid at all: it inherits
the canonical change trace (LO 3 / HI 12 req/s, 3 rounds), which is a REGISTERED
WORKLOAD of this repository rather than a grid CP-2 chose, and every constant
below is a RATIO or a count.  The one place absolute units appear -- the SLO band
-- is inherited verbatim from an existing canon document and is used only to ask
whether the verdict SURVIVES moving it, never to set it.

★None of the constants below were derived from CP-2's own data, because CP-2 has
no data yet.  That is the property the 2026-09-01 G1 result asked for in writing:
*"plateau 술어를 코드로 등록하는 것이 다음 판본의 선결조건이며, 그 술어는 결과를
보기 전에 고정돼야 한다"* (`RESULT_G1_900067_2026-09-01.md`).  This module is
registered before its campaign exists.
"""

# The module owns its own classification (4th audit L-d: classifying constants in
# the rule module only is how `FLOAT_TOL` and `PROBE_RATES` became free).
LABEL_CONSTANTS = ("EQUIV_MARGIN_REL", "N_MIN_REPS",
                   "TTFT_SLO_BAND_MS", "ITL_SLO_BAND_MS")   # must be load-bearing
INFRA_CONSTANTS = ("FLOAT_TOL",)                            # must NOT classify alone


# ---------------------------------------------------------------- the estimand
# Delta is the PAIRED RELATIVE difference in SLO-goodput between the two families:
#
#     Delta = (goodput(cp_arm) - goodput(pdmux_arm)) / goodput(pdmux_arm)
#
# so Delta > 0 means the chunked-prefill family served more requests within SLO.
# Goodput itself is NOT defined here: it is taken from the canonical scorer
# `../slo_sched/oracle_reanalysis_2026_08_16.py` (`score_run`, whose
# `RequestResult.passes` is "TTFT <= SLO AND p95 of that request's own token ITLs
# <= SLO").  CP-2 re-implements no estimator -- /scratch/ehmoon/whlee/CLAUDE.md
# gate 4 names that predicate as canonical and the repository already owns one
# implementation of it.  The harness's own inline scorer uses MEAN ITL and is
# therefore NOT the primary (it stays as a published secondary).

# Equivalence margin.  Reused, not invented: /scratch/ehmoon/whlee/CLAUDE.md
# "방법론 게이트" 3 -- "3% 미만 차이는 headline 아님".  A difference inside this band
# is not a finding in this repository, so the band is what TOST tests against.
EQUIV_MARGIN_REL = 0.03

# Minimum reps per arm.  Reused, not invented: the same gate 3 -- "베이스라인 분산
# 먼저 측정, n>=4 없이 정책 결론 금지".  `rep` here means an independent BOOT, which
# is how the repository's own sgptv arm map counts them (four jobs per arm in
# `../slo_sched/oracle_reanalysis_2026_08_16.py`, e.g. d16 -> rep1/rep44/rep45/rep46).
N_MIN_REPS = 4

# The SLO band the verdict must survive.  Inherited VERBATIM from
# `../../../../reports/CONSENSUS.md` 1-20, which records that sweeping the TTFT
# threshold +-10% (2700-3300 ms) moved a published headline between +0.00% and
# +19.75% and moved its STRUCTURE at 2 of 41 points.  That finding is why this axis
# exists: a goodput verdict taken at one threshold, when the threshold sits in the
# mass of the TTFT distribution, is a coin flip wearing a CI.
# ★These are used ONLY to re-score the SAME runs at 9 points and ask whether the
# contrast label is the same at all 9.  They never set the operating point (which
# stays 3000/60, the value every prior sgptv run used) and they are never tuned.
TTFT_SLO_BAND_MS = (2700.0, 3000.0, 3300.0)
ITL_SLO_BAND_MS = (54.0, 60.0, 66.0)


# Float safety.  Registered because CP-0 already lost a boundary vector to it
# (2.40/3.0 == 0.7999999999999999) and because the repository has the same finding
# against NSL (1.0-0.80 == 0.19999999999999996).  A decision that turns on float
# representation is not a decision.
FLOAT_TOL = 1e-9


def _gt(a, b):
    """Strictly greater, with the boundary EXCLUDED under tolerance."""
    return a > b + FLOAT_TOL


def _lt(a, b):
    return a < b - FLOAT_TOL


# ------------------------------------------------------------ contrast (TOST)
def contrast_axis(ci_lo, ci_hi, margin=None):
    """Total map from one paired relative-difference CI to a `cp2_rule` contrast value.

    `cp_better`    -- the whole CI is beyond +margin      (chunked prefill wins)
    `pdmux_better` -- the whole CI is beyond -margin       (PD-mux wins)
    `equivalent`   -- the whole CI is INSIDE (-margin, +margin): TOST.  Note this
                      is power-aware by construction -- a wide CI cannot fit inside
                      the band -- which is what keeps PROJECT_STATUS.md "방법론
                      게이트" 20 ("귀무 채택형 게이트는 노이즈를 보상한다") from
                      biting: noise buys `inconclusive`, never `equivalent`.
    `inconclusive` -- anything else (the CI crosses a margin boundary).

    The boundary is EXCLUDED on purpose: a CI that merely touches +margin has not
    cleared the repository's own "3% 미만은 headline 아님" bar.
    """
    m = EQUIV_MARGIN_REL if margin is None else margin
    if ci_lo > ci_hi:
        raise ValueError(f"malformed CI: {ci_lo} > {ci_hi}")
    if m <= 0:
        raise ValueError(f"margin must be positive, got {m}")
    if _gt(ci_lo, m):
        return "cp_better"
    if _lt(ci_hi, -m):
        return "pdmux_better"
    if _gt(ci_lo, -m) and _lt(ci_hi, m):
        return "equivalent"
    return "inconclusive"


# -------------------------------------------------------------- phase reversal
# The change trace exists because LO and HI want different splits (that is the
# whole premise of `../slo_sched/sharegpt_vary_bench.sbatch`).  A COMBINED verdict
# that hides a phase reversal would erase exactly the structure the trace was built
# to expose, so the reversal is carried as its own axis rather than folded away.
#
# ★Conservative by construction: only two OPPOSITE SIGNIFICANT directions count as
# a reversal.  `equivalent` or `inconclusive` in one phase does not manufacture one
# -- otherwise noise in the quiet phase would buy the more interesting label.
def phase_reversal_axis(lo_contrast, hi_contrast):
    """Total map from the two per-phase contrast values to a `cp2_rule` axis value."""
    vals = (lo_contrast, hi_contrast)
    if any(v is None or v == "unmeasured" for v in vals):
        return "unmeasured"
    known = {"cp_better", "pdmux_better", "equivalent", "inconclusive"}
    bad = [v for v in vals if v not in known]
    if bad:
        raise ValueError(f"unknown contrast value(s): {bad}")
    return "reversal" if set(vals) == {"cp_better", "pdmux_better"} else "no_reversal"


# ------------------------------------------------------------------- coverage
def coverage_axis(reps_per_arm, n_min=None):
    """Total map from {arm: completed rep count} to a `cp2_rule` coverage value.

    `absent`   -- nothing completed anywhere
    `complete` -- every registered arm reached n_min INDEPENDENT BOOTS
    `partial`  -- something completed but not that

    PROJECT_STATUS.md "방법론 게이트" 21 ("측정 실패를 게이트 실패로 라벨링 마라")
    is why `partial` and `absent` are distinct values and both are non-substantive:
    a campaign that half-ran is not a finding about either policy family.
    """
    n = N_MIN_REPS if n_min is None else n_min
    if not reps_per_arm:
        raise ValueError("no arms given")
    counts = list(reps_per_arm.values())
    if any(c < 0 for c in counts):
        raise ValueError(f"negative rep count in {reps_per_arm}")
    if all(c == 0 for c in counts):
        return "absent"
    return "complete" if all(c >= n for c in counts) else "partial"


# ---------------------------------------------------------------- metric cliff
def cliff_axis(contrast_by_slo_point):
    """Total map from the re-scored contrast labels to a `cp2_rule` cliff value.

    Input is the contrast label at EVERY point of the registered SLO band -- the
    full product TTFT_SLO_BAND_MS x ITL_SLO_BAND_MS, in any order.  A shorter or
    longer sequence is REFUSED rather than scored, because a cliff check run on a
    subset of the band is not the check that was registered (this is what makes the
    two band constants load-bearing rather than decorative).

    `stable`     -- the same contrast label at all 9 points
    `unstable`   -- not
    `unmeasured` -- the band was not swept, or a point could not be scored
    """
    vals = list(contrast_by_slo_point)
    n_expected = len(TTFT_SLO_BAND_MS) * len(ITL_SLO_BAND_MS)
    if len(vals) != n_expected:
        raise ValueError(
            f"cliff check needs the full registered band: expected {n_expected} "
            f"points ({len(TTFT_SLO_BAND_MS)} TTFT x {len(ITL_SLO_BAND_MS)} ITL), got {len(vals)}")
    if any(v is None or v == "unmeasured" for v in vals):
        return "unmeasured"
    return "stable" if len(set(vals)) == 1 else "unstable"


# ---------------------------------------------------------------------- vectors
# `cp2_selftest.py` asserts each expectation AND that perturbing the corresponding
# constant changes at least one of them.  A constant that survives its own mutation
# test is dead, and the test says so.
CONTRAST_VECTORS = [
    # (ci_lo, ci_hi, expected)                       why this vector is here
    (0.10, 0.20, "cp_better"),           # clean win, both ends past +3%
    (-0.20, -0.10, "pdmux_better"),      # clean loss
    (-0.02, 0.02, "equivalent"),         # TOST: CI inside the band
    (-0.05, 0.10, "inconclusive"),       # straddles both boundaries
    (-0.05, 0.02, "inconclusive"),       # crosses -margin only
    (0.02, 0.10, "inconclusive"),        # crosses +margin only
    # boundary cases (4th audit L-g: a rule with no boundary vector lets a mutant
    # that reads `>` as `>=` survive)
    (0.03, 0.20, "inconclusive"),        # ci_lo EXACTLY at +margin -- not beyond it
    (-0.20, -0.03, "inconclusive"),      # ci_hi EXACTLY at -margin
    (-0.03, 0.02, "inconclusive"),       # ci_lo exactly at -margin: not inside
    (0.031, 0.05, "cp_better"),          # just past
    (-0.029, 0.029, "equivalent"),       # just inside
]

PHASE_VECTORS = [
    ("cp_better", "cp_better", "no_reversal"),
    ("pdmux_better", "pdmux_better", "no_reversal"),
    ("cp_better", "pdmux_better", "reversal"),
    ("pdmux_better", "cp_better", "reversal"),        # order must not matter
    ("equivalent", "pdmux_better", "no_reversal"),    # quiet phase does not buy it
    ("inconclusive", "cp_better", "no_reversal"),     # noise does not buy it
    ("cp_better", "unmeasured", "unmeasured"),
]

COVERAGE_VECTORS = [
    ({"cp": 4, "pdmux": 4}, "complete"),
    ({"cp": 5, "pdmux": 4}, "complete"),
    ({"cp": 4, "pdmux": 3}, "partial"),      # one arm short of n>=4
    ({"cp": 0, "pdmux": 4}, "partial"),      # one arm never ran
    ({"cp": 0, "pdmux": 0}, "absent"),
]

CLIFF_VECTORS = [
    (["pdmux_better"] * 9, "stable"),
    (["pdmux_better"] * 8 + ["inconclusive"], "unstable"),   # ONE point flips
    (["cp_better"] * 4 + ["pdmux_better"] * 5, "unstable"),  # sign reverses in band
    (["equivalent"] * 8 + ["unmeasured"], "unmeasured"),
]
