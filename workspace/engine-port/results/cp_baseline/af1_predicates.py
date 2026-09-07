"""AF-1 predicates: the UNCONTENDED-FLOOR SCREEN, as code.  PRED_REV = 6.

What changed in rev3, rev4, rev5 and rev6 -- the short version
-----------------------------------------------------------------------------------
  rev3 (3rd audit, J1-J8)  neutrality judged on the STRICT screen with a
        `borderline` value and a `q` argument; `coverage_axis` replaced the
        predicate-less `boot` axis; `undominated_survey` restored.
  rev4 (4th audit, K1-K8)  the roofline became registered constants and folds on
        MEASURED strata; `screen_boundaries`; `MARGIN_MIN_BAND_MULT`;
        `INTERNAL_CONSTANTS` so private globals cannot escape classification.
  rev5 (5th audit, Q1)     the branch `b_more_boots` was retired for
        `b_precision_bound` -- a sample SD does not shrink with more boots, so
        "buy more boots" was never purchasable (4th-audit K3).
  rev6 (6th audit, GO)     the surviving descriptions of that retired branch were
        corrected, and the token check that should have caught them was widened
        past UPPERCASE-only.
  ★This header stopped at rev2 through four revisions while a ledger row claimed
  it had been updated (6th-audit 등재 권고 1) -- gates #67/#70: a change-history
  entry may only record what has actually been done.

What changed from rev2 (`audit_af1_rules_2nd_2026-09-04/VERDICT.md`, NO-GO, E1-E10)
-----------------------------------------------------------------------------------
  E2  `neutrality_axis` was NOT SD-free, though three documents said it was: it
      compared the arms' BANDED sets, so `floor + BAND_SD_MULT * SD` re-entered
      the axis that dominates every other one, and a single noisy arm could close
      route (a) with identical floors.  Every registered vector had `sd = 0.0`, so
      no check could see it.
      -> neutrality now compares the STRICT sets only.  A strict/banded
         disagreement is its own value `borderline`, exactly as `anchor` and
         `ladder` already treat it -- the rev2 repair applied to the axis that
         dominates, instead of only to the two it dominates.  It takes `q`, so the
         stratum axis compares it too.
  E3  The slack `PLANNED_BOOTS = 5 > MIN_BOOTS = 4` created a world -- one lost
      boot -- that the rule called IMPOSSIBLE and `fork_branch` answered with a
      KeyError.  `boot` was the only axis with no predicate, no vector and no
      prose definition, so "4 of 5" was the author's post-hoc reading.
      -> `coverage_axis` is a real predicate with vectors straddling MIN_BOOTS,
         it has TWO values (a lost boot is `sufficient`, not a new world), and
         `PLANNED_BOOTS` moves to DESIGN where a two-sided mutation proves it
         moves no label.  `SCREEN_INCOMPLETE` names the worlds where enough boots
         were scored but the screen could not run.
  E8  `undominated_survey()` was deleted in rev2 while the pre-registration kept
      citing it -- gate 88 firing on this probe's own anti-deletion machinery.
      -> restored, on PREDICATE_FOLDS, with a vector.
  E9  The roofline is now pre-registered (pre-registration sec 5.1), so the
      prediction is falsifiable rather than defended by "this cannot be derived
      at GPU 0", which was false.

What changed in rev2, from rev1 (`audit_af1_rules_2026-09-04/VERDICT.md`, D1-D10)
--------------------------------------------------------------------------------
  D1  The neutrality axis was a significance test with no effect-size floor
      (`spread > 3 * pooled_sd`), so a MORE precise measurement made route (a)
      MORE likely to die, and `pooled_sd` -- a 3-arm-SD-to-scalar reduction --
      was never registered, which is 6th-audit F1 in the same words.
      -> neutrality is DECISION-RELEVANT and judged on the STRICT screen: the arms
         are neutral when they admit the SAME set of candidates.  `pooled_sd` and
         the unsourced 3.0 are both gone.  ⚠️This is NOT "SD-free": the band still
         enters through the `borderline` value (4th-audit K5 -- rev4's ledger said
         this line had been retracted and the line was byte-identical).
  D2  `ANCHOR_INFEASIBLE` was not in the IMAGE of the screen: `none` needs a
      bound above 3000 ms or 200 ms and `intact` needs one at or below 500/40, so
      the pair could never occur.  The reachability check enumerated the DECLARED
      axis product, which is an identity.
      -> `reachable_pairs()` computes the image EXACTLY, from critical values
         derived from the constants (no probe grid to choose), and the self-test
         requires OUTCOME's key set to equal it in BOTH directions.
  D4  The band `floor + 2*SD` was called conservative in both directions.  It is
      not: widening the band makes it HARDER to admit a coordinate and therefore
      EASIER to conclude that route (a) is closed.
      -> both screens are computed, `strict` (floor only) and `banded`, and a
         disagreement is its own axis value `borderline` rather than a silent
         choice of which one to believe.  This is the metric-cliff gate
         (`CLAUDE.md`, "goodput처럼 SLO 임계 지시함수는 metric cliff") applied to
         this probe's own threshold.
  D5  The two-level aggregation (within a boot, then across boots) was unregistered
      and moved the bound by about 2 SD -- the same size as the band itself.
      -> `AGG_WITHIN_BOOT` and `AGG_ACROSS_BOOTS` are registered LABEL constants
         that the code dispatches on, and the sampling parameters are registered.
  D8  Planned boots equalled MIN_BOOTS, so a single lost boot triggered the stop
      rule with no slack.
      -> PLANNED_BOOTS = 5, MIN_BOOTS = 4.

What this module is for
-----------------------
CP-2 rev2 died on death cause F2: its rule reads `contrast` "at the registered
operating point" and four of its labels carry a DIRECTION in their name, while no
(TTFT, ITL) coordinate is registered anywhere -- so the direction was a free
parameter to be set after seeing the data.  W1 measured that the sign of the
comparison flips inside the registered ladder, which makes that parameter decisive.

`RESULT_POSITIONING_AND_WORKLOAD_2026-09-02.md` sec 2.1 handed rev3 three ways out
and adopted none: (a) derive the coordinate from workload/engine properties
without looking at arm data, (b) report the whole ladder and pick no threshold,
(c) drop the indicator and go continuous.  This module registers the DECISION
PROCEDURE for (a), and -- because a procedure that cannot select is as informative
as one that can -- registers in advance what happens when (a) fails.

The screen is MONOTONE and it can only REMOVE candidates.  It never ranks the
survivors, so it cannot smuggle a direction back in.

★What the screen does NOT buy, stated here because rev1 omitted it (D2)
-----------------------------------------------------------------------
The five candidate coordinates form a TOTAL ORDER under componentwise <=
(`survey_is_chain`), so the admissible set is always a SUFFIX of that chain.
Therefore `anchor = unique` can only ever mean `batch_async`, the chain's maximum.
AF-1 does not discover WHICH coordinate is reachable -- the survey table's
ordering already fixed that.  It buys ONE BIT: whether the largest production
budget is physically reachable on this model and workload, and whether the arms
agree about it.  A result document that presents the identity of the survivor as
a finding is misreading this module.

Provenance of the METHOD (both external to this campaign)
---------------------------------------------------------
  * `reports/serving_slo_survey.md` sec 2, "SLO를 *sweep*하는 것이 표준 방법론":
    DistServe defines "SLO scale" as a multiple of the UNCONTENDED single-request
    latency, so an uncontended floor is the standard anchor.
  * `reports/interactive_slo_retune_plan.md` line 77, "물리적 실현불가 (HT-neg)":
    this repository has already screened an SLO out this way -- line 110 records
    code(100/25) at 66.2% attainment UNCONTENDED -- and its prescription is
    literally "L−1 전에 무부하 attainment 체크".
"""

# --- constants, classified in three kinds ----------------------------------
#
# LABEL: read BY VALUE by a function below; moving one moves the label map or a
#        registered fold output.  af1_selftest.py substitutes VALUES here, never
#        tuple lengths (6th-audit F6), and checks statically that each name is
#        read by some registered fold (gate 87).
# DESIGN: decides what is BOUGHT, required to move nothing in EITHER direction
#        (6th-audit L3: a label-bearing constant filed as DESIGN passed only
#        because its mutation went the harmless way).
# INFRA: plumbing.
# Private module globals that are not constants of the design.  4th-audit K7:
# rev4's C1 exempted every `_`-prefixed name, so a private `_BAND = 2.0` could
# take over the label path with both gate-87 certificates still green.
INTERNAL_CONSTANTS = ("_IMAGE_CACHE", "_TRIPLE_CACHE")

LABEL_CONSTANTS = (
    "SURVEY_POINTS", "LADDER_TTFT_MS", "LADDER_ITL_MS", "SCREEN_STRATUM_Q",
    "SCREEN_STRATUM_Q_ALT", "UNCONTENDED_STAT_Q", "BAND_SD_MULT",
    "MARGIN_MIN_BAND_MULT", "MIN_BOOTS", "ARMS",
    "AGG_WITHIN_BOOT", "AGG_ACROSS_BOOTS",
    # The roofline constants are LABEL, not DESIGN: registered folds read them by
    # value, so moving one moves a registered output.  (The class is defined as
    # "moves the label map OR a registered fold output".)
    "STRATUM_TOKENS", "PARAM_COUNT", "PEAK_BF16_TFLOPS", "HBM_BW_GBPS", "SM_TOTAL",
)
# `PLANNED_BOOTS` is DESIGN, not LABEL, and that is the E3 repair: it decides how
# many boots are BOUGHT, and losing one must not create a world.  The two-sided
# mutation in af1_selftest.py proves it moves no label in either direction.
DESIGN_CONSTANTS = (
    "REPORT_STRATUM_QS", "REPEATS_PER_STRATUM", "PLANNED_BOOTS", "MAX_NEW_TOKENS",
    "TEMPERATURE", "IGNORE_EOS", "BOOT_SEEDS", "ARM_ORDER_BY_JOB",
)
INFRA_CONSTANTS = ("REQ_TIMEOUT_S",)

# Production interactive SLO budgets, P99, from `reports/serving_slo_survey.md`
# sec 1, "Spheron 2026 SLO Engineering 가이드의 use-case별 표(P99)".  External to
# this campaign, registered in this repository on 2026-07-18, and no arm datum
# took part in choosing them.
# ★"code panel" in that table is (300, 50), the same COORDINATE as "chat", so it
#   is deliberately absent: the axis counts coordinates, and listing one twice
#   would let `multiple` be reached by a duplicate rather than a second budget.
SURVEY_POINTS = (
    ("code_inline", 100.0, 25.0),
    ("voice", 150.0, 30.0),
    ("chat", 300.0, 50.0),
    ("rag", 400.0, 80.0),
    ("batch_async", 3000.0, 200.0),
)

# CP-2 rev2's registered re-scoring ladder, inherited VERBATIM from
# `cp2r2_predicates.py` lines 47-48.  af1_selftest.py check C14 re-parses that
# file and compares, because 6th-audit F6's successor mutation was "+1 ms on every
# cell" and prose cannot catch it.
LADDER_TTFT_MS = (500.0, 1000.0, 2000.0, 3000.0, 4000.0, 6000.0)
LADDER_ITL_MS = (40.0, 50.0, 60.0, 70.0, 80.0, 100.0)

# Which prompt-length stratum the screen is applied at.
# ⚠️SCOPE, and it is not the survey's: `reports/serving_slo_survey.md` sec 2 gives
# the practitioner rule "SLO를 median prompt length에 맞춰 잡지 말고 P99 prompt
# length로 prefill budget을 산정하라" for PROVISIONING, and its very next sentence
# separates the layers ("두 층위는 별개").  More decisively, this repository ran
# this same check once and moved OFF p99:
# `reports/interactive_slo_retune_plan.md` line 113 records
# "★무경쟁 p99=383ms > chat 300ms(긴 요청 탓) ⇒ P99-attainment 100%는 물리적으로
# 불가 → P90 기준" -- i.e. p99 on a long-prompt population measures the tail's
# physical limit, not the budget.  AF-1 therefore screens at BOTH and treats a
# disagreement as an outcome instead of picking one (see `stratum_axis`).
SCREEN_STRATUM_Q = 0.90
SCREEN_STRATUM_Q_ALT = 0.99
REPORT_STRATUM_QS = (0.10, 0.50, 0.90, 0.99)

# The within-(arm, stratum, boot) summary.  ★An ORDER STATISTIC, registered as one
# (`CONSENSUS.md` sec 3 item 82: an estimand that depends on evaluation order is a
# rank, not an attribution).  0.90 because `reports/serving_slo_survey.md` sec 1
# records "goodput/attainment도 통상 P90–P99" and p99 is not estimable here.
UNCONTENDED_STAT_Q = 0.90

# The two-level aggregation, registered because it was worth about 2 SD (D5).
AGG_WITHIN_BOOT = "order_stat"   # over REPEATS_PER_STRATUM observations
AGG_ACROSS_BOOTS = "mean"        # over the per-boot values

# The band, as an explicit MULTIPLE of the untreated SD
# (`PROJECT_STATUS.md` gate 86, "연속 축 위에 문턱을 등록하기 전에 그 통계량의
# 무처치(no-load) 분산을 먼저 재고").
# ⚠️DIRECTION (D4): widening this band makes admitting a coordinate HARDER and so
# makes "route (a) is closed" EASIER.  It is conservative about the claim
# "this budget is reachable", and liberal about the claim "it is not".  That is
# why `strict` and `banded` are both computed and a disagreement is `borderline`.
BAND_SD_MULT = 2.0

# `PROJECT_STATUS.md` gate 86 does not stop at "measure the untreated variance
# first": it goes on to "유도한 판정 밴드가 그 분산보다 넓은지 산술로 보여라 — 좁으면
# 그 축은 식별력이 없다".  rev3 quoted only the first half and satisfied it by
# DEFINING the band as 2*SD, which is an identity (3rd-audit J5).  The arithmetic
# the gate actually asks for is: how far is the measured floor from the nearest
# decision boundary, in units of the band?  Below this multiple the axis has no
# discriminating power at that boundary and the result document must say so.
MARGIN_MIN_BAND_MULT = 2.0

MIN_BOOTS = 4        # `CLAUDE.md`, "베이스라인 분산 먼저 측정, n≥4 없이"
PLANNED_BOOTS = 6    # slack 2.  Six jobs also make the arm rotation BALANCED --
                     # rev2 registered five of the six permutations and called it
                     # "전부 열거", which was false and left `d44` under-exposed in
                     # first position, the arm whose floor decides the whole axis
                     # (E10).
ARMS = ("plain", "cp2048", "d44")

# --- the roofline, as registered constants and folds (4th-audit K1) ----------
#
# rev4 wrote the roofline in prose and quoted the SELECTION tokenizer's token
# counts -- the very numbers sec 1 forbids citing -- so its prediction 2 rested on
# a number the pre-registration had banned, and that prediction is now FALSIFIED
# (see the pre-registration's sec 5.1).  The strata below are MEASURED with the
# campaign tokenizer (`af1_strata.json`, CPU only, GPU 0, no arm involved) and the
# arithmetic is a fold with vectors, so neither can drift into prose again.
STRATUM_TOKENS = {0.10: 1932.7, 0.50: 2514.0, 0.90: 3764.0, 0.99: 5925.3}

# Model weights, from the model's own `model.safetensors.index.json`:
# total_size 17,776,454,656 bytes / 2 (bf16) = 8.888227328e9 parameters.
PARAM_COUNT = 8.888227328e9

# A100 SXM 80GB.  ★Which card was NOT registered by rev4 and the ITL prediction
# needed it (4th-audit P3).  It is resolvable from an artefact this track already
# has: `cgate_902407/srv_plain.log` reports `avail mem=78.74 GB` at startup, which
# is the 80GB part.  Vendor spec values; no in-repo citation exists for either,
# and that is registered here rather than invented (4th-audit criterion 4).
PEAK_BF16_TFLOPS = 312.0
HBM_BW_GBPS = 2039.0
SM_TOTAL = 108

# All 3! = 6 orders, so every arm appears in every position exactly twice.
ARM_ORDER_BY_JOB = (
    ("plain", "cp2048", "d44"), ("plain", "d44", "cp2048"),
    ("cp2048", "plain", "d44"), ("cp2048", "d44", "plain"),
    ("d44", "plain", "cp2048"), ("d44", "cp2048", "plain"),
)

REPEATS_PER_STRATUM = 3
MAX_NEW_TOKENS = 64
TEMPERATURE = 0.0
IGNORE_EOS = True    # so that ITL p95 is defined on a fixed token count
BOOT_SEEDS = (11, 23, 37, 53, 67, 71)   # one per planned boot
REQ_TIMEOUT_S = 600


# --- primitives ------------------------------------------------------------

def order_stat(values, q=None):
    """The registered order statistic: linear interpolation between ranks.

    Named `order_stat`, not `percentile`, so no reader mistakes it for a
    distributional parameter.  Which rank is fixed by `UNCONTENDED_STAT_Q`.
    """
    q = UNCONTENDED_STAT_Q if q is None else q
    xs = sorted(float(v) for v in values)
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    pos = q * (len(xs) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (pos - lo) * (xs[hi] - xs[lo])


def agg_within_boot(observations):
    """Reduce one boot's repeats to one number.  Dispatches on AGG_WITHIN_BOOT."""
    xs = [float(v) for v in observations if v is not None]
    if not xs:
        return None
    if AGG_WITHIN_BOOT == "order_stat":
        return order_stat(xs)
    if AGG_WITHIN_BOOT == "mean":
        return sum(xs) / len(xs)
    if AGG_WITHIN_BOOT == "max":
        return max(xs)
    raise ValueError("unregistered AGG_WITHIN_BOOT: %r" % (AGG_WITHIN_BOOT,))


def agg_across_boots(per_boot):
    """Reduce the per-boot values to the floor estimate.  Dispatches on
    AGG_ACROSS_BOOTS.  `None` below MIN_BOOTS, so that a floor never exists
    without the SD that gives it a band."""
    xs = [float(v) for v in per_boot if v is not None]
    if len(xs) < MIN_BOOTS:
        return None
    if AGG_ACROSS_BOOTS == "mean":
        return sum(xs) / len(xs)
    if AGG_ACROSS_BOOTS == "median":
        return order_stat(xs, 0.5)
    if AGG_ACROSS_BOOTS == "max":
        return max(xs)
    raise ValueError("unregistered AGG_ACROSS_BOOTS: %r" % (AGG_ACROSS_BOOTS,))


def boot_sd(per_boot):
    """Sample SD (df = n-1) across boots.  None below MIN_BOOTS.

    None rather than 0.0 on purpose: a screen built on an unmeasured variance
    would silently become a screen with a ZERO band, which is the tightest screen
    there is.  `coverage_axis` turns that None into a label.  (rev3's docstring
    here still named a `variance` axis that rev3 had already removed -- 3rd-audit
    J8, gate 88 on this probe's own anti-deletion machinery.)
    """
    xs = [float(v) for v in per_boot if v is not None]
    if len(xs) < MIN_BOOTS:
        return None
    m = sum(xs) / len(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def screen_bound(floor_hat, sd, banded=True):
    """The value a candidate SLO must reach.  `banded` adds BAND_SD_MULT * SD."""
    if floor_hat is None:
        return None
    if not banded:
        return float(floor_hat)
    if sd is None:
        return None
    return float(floor_hat) + BAND_SD_MULT * float(sd)


def arm_bounds(strata_by_arm, banded=True, q=None):
    """{arm: (ttft_bound, ritl_bound)} at the screen stratum.

    `strata_by_arm` maps arm -> {q: (ttft_floor, ritl_floor, ttft_sd, ritl_sd)}.
    Exists so that SCREEN_STRATUM_Q is read by the LABEL path and not only by the
    harness -- a constant no rule function reads cannot be shown to be
    load-bearing (`PROJECT_STATUS.md` gate 87).
    """
    q = SCREEN_STRATUM_Q if q is None else q
    if not strata_by_arm:
        return {}
    out = {}
    for arm in ARMS:
        st = strata_by_arm.get(arm)
        if not st or st.get(q) is None:
            return {}
        tf, rf, tsd, rsd = st[q]
        out[arm] = (screen_bound(tf, tsd, banded), screen_bound(rf, rsd, banded))
    return out


# --- the screen ------------------------------------------------------------

def survey_is_chain():
    """Are the candidate coordinates a TOTAL ORDER under componentwise <=?

    Registered as a fold because the answer is what makes `unique` mean only ever
    the chain's maximum -- the fact rev1 failed to disclose (D2).  If a future
    edit breaks the chain, this flips and the self-test's disclosure check fails.
    """
    p = list(SURVEY_POINTS)
    return all(p[k][1] <= p[k + 1][1] and p[k][2] <= p[k + 1][2]
               for k in range(len(p) - 1))


def undominated_survey():
    """Candidate coordinates that NO ladder point dominates componentwise.

    ★Restored in rev3.  rev2 deleted it while the pre-registration kept citing it
    by name (E8) -- gate 88 ("삭제는 조건 등록이 아니라") firing on this probe's own
    anti-deletion machinery.  It is the constant-level fact behind sec 4.5: the
    chain's maximum is undominated, so a registered anchor is NOT scorable on
    CP-2's 36-point grid and branch `a` always carries a ladder extension.
    """
    out = []
    for name, t, i in SURVEY_POINTS:
        if not any(lt >= t and li >= i
                   for lt in LADDER_TTFT_MS for li in LADDER_ITL_MS):
            out.append(name)
    return out


def admissible(ttft_slo, itl_slo, bounds_by_arm):
    """Is this coordinate reachable UNCONTENDED by EVERY registered arm?

    Taking every arm rather than the reference arm alone is the conservative
    direction for ADMITTING a coordinate, and it is direction-free with respect
    to the two families: it removes coordinates where SOME arm is physically
    incapable, whichever arm that is.
    """
    if not bounds_by_arm:
        return None
    for arm in bounds_by_arm:
        tb, ib = bounds_by_arm[arm]
        if tb is None or ib is None:
            return None
        if tb > float(ttft_slo) or ib > float(itl_slo):
            return False
    return True


def admissible_survey(bounds_by_arm):
    """Names of the candidate coordinates every arm can reach."""
    out = []
    for name, t, i in SURVEY_POINTS:
        a = admissible(t, i, bounds_by_arm)
        if a is None:
            return None
        if a:
            out.append(name)
    return out


def anchor_class(bounds_by_arm):
    """unique / multiple / none, for ONE screen (strict or banded)."""
    surv = admissible_survey(bounds_by_arm)
    if surv is None:
        return "unmeasured"
    if len(surv) == 0:
        return "none"
    return "unique" if len(surv) == 1 else "multiple"


def ladder_class(bounds_by_arm):
    """intact / pruned / empty, for ONE screen."""
    total = ok = 0
    for t in LADDER_TTFT_MS:
        for i in LADDER_ITL_MS:
            a = admissible(t, i, bounds_by_arm)
            if a is None:
                return "unmeasured"
            total += 1
            ok += 1 if a else 0
    if ok == 0:
        return "empty"
    return "intact" if ok == total else "pruned"


def _both(strata_by_arm, cls, q=None):
    s = cls(arm_bounds(strata_by_arm, banded=False, q=q))
    b = cls(arm_bounds(strata_by_arm, banded=True, q=q))
    if s == "unmeasured" or b == "unmeasured":
        return "unmeasured"
    return s if s == b else "borderline"


def anchor_axis(strata_by_arm, q=None):
    """unique / multiple / none / borderline / unmeasured.

    ★No tie-break and no ordering code: `multiple` is a registered outcome
    meaning "this procedure cannot choose", which is the only thing stopping the
    coordinate from becoming a free parameter again.  `borderline` means the
    strict and banded screens disagree, i.e. the noise decides -- also not
    something an author may resolve (D4).
    """
    return _both(strata_by_arm, anchor_class, q)


def ladder_axis(strata_by_arm, q=None):
    """intact / pruned / empty / borderline / unmeasured."""
    return _both(strata_by_arm, ladder_class, q)


def coverage_axis(boots_by_arm):
    """sufficient / insufficient -- did every arm reach MIN_BOOTS SCORED boots?

    ★rev2 had a `boot` axis with no predicate, no vector and no prose definition:
    its only meaning was a comment saying "every planned boot", under which losing
    one of five boots gave `boot=failed` AND `variance=measured`, a combination the
    rule called impossible and `fork_branch` answered with a KeyError (E3).  Which
    reading to take would have been the author's choice AFTER the run.

    Two values, on purpose.  `PLANNED_BOOTS` buys slack; it does not create a
    world.  How many boots were actually scored is a MANDATORY OUTPUT
    (pre-registration sec 3.2), not a label -- reporting it is not the same as
    letting it move the verdict.
    """
    if not boots_by_arm:
        return "insufficient"
    for arm in ARMS:
        n = boots_by_arm.get(arm, 0)
        if n is None or n < MIN_BOOTS:
            return "insufficient"
    return "sufficient"


def _survey_sets(strata_by_arm, banded, q=None):
    """Per-arm admissible candidate sets, or None if anything is unmeasured."""
    b = arm_bounds(strata_by_arm, banded=banded, q=q)
    if not b:
        return None
    sets = []
    for arm in ARMS:
        if arm not in b:
            return None
        one = admissible_survey({arm: b[arm]})
        if one is None:
            return None
        sets.append(tuple(one))
    return sets


def neutrality_axis(strata_by_arm, q=None):
    """neutral / borderline / arm_specific / unmeasured.

    Do the arms admit the SAME candidate set?  If they disagree on the FLOORS
    themselves, the surviving coordinate is set by the worst arm and is an arm
    choice wearing a coordinate's clothes, so route (a)'s premise is false.

    ★Judged on the STRICT screen only.  rev2 judged on the banded sets as well and
    took the UNION of the two failures, so `floor + BAND_SD_MULT * SD` decided the
    axis that dominates every other one -- identical floors plus one noisy arm
    closed route (a) -- while three documents claimed "no SD enters here" (E2).
    A strict/banded disagreement is `borderline`: the noise decides, and that is an
    outcome, not a silent choice of which screen to believe.

    ⚠️NOT SD-free, and rev3 said it was in three places (3rd-audit J1).  The band
    still enters through `borderline`: identical floors plus one noisy arm gives
    `borderline`, not `neutral`.  What changed is the PRESCRIPTION -- `borderline`
    has its OWN branch, `b_precision_bound`: the noise decided at THIS measurement
    precision.  ★It is NOT "buy more boots".  rev4 named that branch
    `b_more_boots` and 4th-audit K3 showed the name was unpurchasable -- `boot_sd`
    is a SAMPLE SD, so the band converges to 2*sigma rather than to zero, and sec
    3.3 forbids the purchase anyway.  The lever is sigma (measurement design) and
    AF-1 does not buy it, so this label ends AF-1 and hands the coordinate
    question to CP-2 rev3.  (5th-audit Q1: this docstring was one of three places
    where the deleted name kept giving the old prescription.)
    """
    strict = _survey_sets(strata_by_arm, banded=False, q=q)
    if strict is None:
        return "unmeasured"
    if len(set(strict)) != 1:
        return "arm_specific"
    banded = _survey_sets(strata_by_arm, banded=True, q=q)
    if banded is None:
        return "unmeasured"
    if len(set(banded)) != 1:
        return "borderline"
    return "neutral"


def stratum_axis(strata_by_arm):
    """agree / disagree / unmeasured -- does the registered stratum say what the
    alternative one says, on ALL THREE readings?

    ★rev2 compared only `(anchor, ladder)`, so the dominating axis was again
    evaluated at one stratum (E2-ii).  The triple is compared here.
    """
    if not strata_by_arm:
        return "unmeasured"
    trip = []
    for q in (SCREEN_STRATUM_Q, SCREEN_STRATUM_Q_ALT):
        trip.append((neutrality_axis(strata_by_arm, q=q),
                     anchor_axis(strata_by_arm, q=q),
                     ladder_axis(strata_by_arm, q=q)))
    if any("unmeasured" in t for t in trip):
        return "unmeasured"
    return "agree" if trip[0] == trip[1] else "disagree"


def roofline_prefill_ms(tokens, sm_count, mfu):
    """Uncontended TTFT lower bound: dense-equivalent 2*N*P at a given MFU.

    An UPPER BOUND on speed, i.e. a LOWER bound on time.  Registered as a fold so
    the pre-registration's predictions are recomputed from constants instead of
    being retyped as prose (4th-audit K1).  Assumptions, all registered:
    dense-equivalent FLOPs for a hybrid Mamba-Transformer, vendor peak, and
    performance proportional to the green-context SM count.
    """
    if not tokens or not sm_count or not mfu:
        return None
    peak = PEAK_BF16_TFLOPS * 1e12 * (float(sm_count) / SM_TOTAL) * float(mfu)
    return 2.0 * PARAM_COUNT * float(tokens) / peak * 1000.0


def stratum_floor_ms(q, sm_count, mfu):
    """The prefill floor at a REGISTERED stratum, so `STRATUM_TOKENS` is read by
    the label path rather than passed in from prose (gate 87)."""
    t = STRATUM_TOKENS.get(q)
    if t is None:
        return None
    return roofline_prefill_ms(t, sm_count, mfu)


def roofline_decode_ms(mfu_bw=1.0):
    """Uncontended per-token ITL lower bound: weights / HBM bandwidth.

    Decode at batch 1 is weight-bandwidth bound, so this is the floor the ITL leg
    is screened against.  rev4 asserted "≈9.3 ms" in prose with no constant behind
    it and without registering which A100 (4th-audit P3).
    """
    if not mfu_bw:
        return None
    return PARAM_COUNT * 2.0 / (HBM_BW_GBPS * 1e9 * float(mfu_bw)) * 1000.0


def screen_boundaries(leg):
    """The decision boundaries a floor on this leg is screened against.

    `margin_in_bands` needs a boundary set, and rev4 defined it in prose only --
    so the result document would have chosen it, which is the free parameter the
    gate-86 device existed to remove (4th-audit K8).  It is the union of the
    candidate budgets and the ladder rows on that leg, and nothing else.
    """
    if leg == "ttft":
        return tuple(sorted({t for _n, t, _i in SURVEY_POINTS} | set(LADDER_TTFT_MS)))
    if leg == "itl":
        return tuple(sorted({i for _n, _t, i in SURVEY_POINTS} | set(LADDER_ITL_MS)))
    raise ValueError("unregistered leg: %r" % (leg,))


def margin_in_bands(floor_hat, sd, boundaries):
    """Distance from the floor to the NEAREST boundary, in units of the band.

    `boundaries` are the decision thresholds that this floor is screened against
    (candidate coordinates and ladder rows on the same leg).  Returns None when
    the SD is unmeasured, and `float("inf")` when no boundary lies above the
    floor.  A value below `MARGIN_MIN_BAND_MULT` means the screen cannot tell
    that boundary from noise, which is the half of gate 86 rev3 dropped.
    """
    if floor_hat is None or sd is None:
        return None
    band = BAND_SD_MULT * float(sd)
    above = [float(b) - float(floor_hat) for b in boundaries
             if float(b) > float(floor_hat)]
    if not above:
        return float("inf")
    if band == 0.0:
        return float("inf")
    return min(above) / band


def margin_identifiable(floor_hat, sd, boundaries):
    """Is the nearest boundary far enough from the floor to be told from noise?

    This is the fold that READS `MARGIN_MIN_BAND_MULT`, so the constant is on the
    label path instead of parked beside the prose (gate 87).  It is the half of
    gate 86 rev3 quoted and did not apply: "유도한 판정 밴드가 그 분산보다 넓은지
    산술로 보여라 — 좁으면 그 축은 식별력이 없다".  A False here does not change a
    label; it obliges the result document to say the screen could not tell that
    boundary from noise (pre-registration sec 3.2).
    """
    m = margin_in_bands(floor_hat, sd, boundaries)
    if m is None:
        return None
    return m >= MARGIN_MIN_BAND_MULT


# --- the exact image of the screen (D2) ------------------------------------

def critical_bound_values(values):
    """Values at which a `bound <= threshold` comparison can change, plus one
    strictly below all of them.  Derived from the constants, so no probe grid is
    chosen and no new free surface appears.

    ★Public and vectored in rev3.  It was private in rev2, so the derivation that
    `reachable_pairs()` rests on had no test of its own: changing `min(gap)/2` to
    `min(gap)` or dropping the below-all probe both passed (2nd-audit L2).
    """
    vs = sorted(set(float(v) for v in values))
    gaps = [b - a for a, b in zip(vs, vs[1:])]
    d = (min(gaps) / 2.0) if gaps else 1.0
    out = set()
    for v in vs:
        out.add(v)
        out.add(v + d)
    out.add(vs[0] - d)
    return sorted(out)


_IMAGE_CACHE = {}
_TRIPLE_CACHE = {}


def _survey_set_for(t_bound, i_bound):
    """The candidate set one arm admits at a given bound (a chain suffix)."""
    return tuple(n for n, t, i in SURVEY_POINTS
                 if t_bound <= float(t) and i_bound <= float(i))


def reachable_triples():
    """The EXACT set of (neutrality, anchor, ladder) triples the screen can produce.

    ★rev3 computed the image over PAIRS only, so when `neutrality` became an axis
    its values were checked over the DECLARED product -- an identity, and the very
    shape 1st-audit D2 named (3rd-audit J3).  Fourteen worlds were outside the
    image and still carried a substantive label.

    Exactness argument, so this is a derivation and not a sample:
      * `anchor` and `ladder` depend on the arms only through the INTERSECTION of
        their admissible sets, i.e. through the WORST arm.  Adding arms that admit
        MORE changes neither.
      * so enumerate the worst arm's (strict, banded) bounds at the critical
        values -- exactly the pair enumeration -- and ask which `neutrality`
        values a second arm can then realise:
          `neutral`      always (every arm equal to the worst);
          `arm_specific` iff the worst arm's STRICT set is not the whole chain
                         (otherwise no arm can differ by admitting more);
          `borderline`   iff the worst arm's BANDED set is a proper subset of its
                         STRICT set (so another arm can keep the strict set and
                         lose nothing to the band).
    """
    ck = (SURVEY_POINTS, LADDER_TTFT_MS, LADDER_ITL_MS)
    if ck in _TRIPLE_CACHE:
        return _TRIPLE_CACHE[ck]
    full = tuple(n for n, _t, _i in SURVEY_POINTS)
    Ts = critical_bound_values([t for _, t, _ in SURVEY_POINTS] + list(LADDER_TTFT_MS))
    Is = critical_bound_values([i for _, _, i in SURVEY_POINTS] + list(LADDER_ITL_MS))
    out = set()
    for ts in Ts:
        for tb in Ts:
            if tb < ts:
                continue
            for i_s in Is:
                for ib in Is:
                    if ib < i_s:
                        continue
                    bs = {a: (ts, i_s) for a in ARMS}
                    bb = {a: (tb, ib) for a in ARMS}
                    ac, ab = anchor_class(bs), anchor_class(bb)
                    lc, lb = ladder_class(bs), ladder_class(bb)
                    anchor = ac if ac == ab else "borderline"
                    ladder = lc if lc == lb else "borderline"
                    sw = _survey_set_for(ts, i_s)
                    bw = _survey_set_for(tb, ib)
                    out.add(("neutral", anchor, ladder))
                    if sw != full:
                        out.add(("arm_specific", anchor, ladder))
                    if bw != sw:
                        out.add(("borderline", anchor, ladder))
    _TRIPLE_CACHE[ck] = out
    return out


def reachable_pairs():
    """The EXACT set of (anchor, ladder) pairs the screen can produce.

    rev1 declared reachability over the axis PRODUCT, which is an identity
    (`CONSENSUS.md` sec 3 item 9, and 6th-audit F3 in the reachability layer).
    The classes are monotone step functions of the two bounds, so evaluating at
    the critical values is exhaustive, and taking strict <= banded over the same
    values covers `borderline` as well.  `anchor`/`ladder` depend on the arms only
    through the maximum bound, so a single (T, I) pair is fully general here.
    """
    ck = (SURVEY_POINTS, LADDER_TTFT_MS, LADDER_ITL_MS)
    if ck in _IMAGE_CACHE:                # pure in exactly these three constants;
        return _IMAGE_CACHE[ck]           # the self-test substitutes them, so the
                                          # key must name all three and nothing else
    Ts = critical_bound_values([t for _, t, _ in SURVEY_POINTS] + list(LADDER_TTFT_MS))
    Is = critical_bound_values([i for _, _, i in SURVEY_POINTS] + list(LADDER_ITL_MS))
    out = set()
    for ts in Ts:
        for tb in Ts:
            if tb < ts:
                continue
            for i_s in Is:
                for ib in Is:
                    if ib < i_s:
                        continue
                    st = {a: {0.0: (ts, i_s, 0.0, 0.0)} for a in ARMS}
                    bd = {a: {0.0: (tb, ib, 0.0, 0.0)} for a in ARMS}
                    ac = anchor_class(arm_bounds(st, banded=False, q=0.0))
                    ab = anchor_class(arm_bounds(bd, banded=False, q=0.0))
                    lc = ladder_class(arm_bounds(st, banded=False, q=0.0))
                    lb = ladder_class(arm_bounds(bd, banded=False, q=0.0))
                    out.add((ac if ac == ab else "borderline",
                             lc if lc == lb else "borderline"))
    _IMAGE_CACHE[ck] = out
    return out
