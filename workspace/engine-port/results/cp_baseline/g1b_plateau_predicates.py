"""G1-b: the PLATEAU predicate `cp0_g1_rule.py` needs, registered as code.

Why this file exists
--------------------
`cp0_g1_rule.py` (RULE_REV=1) carries an axis `band_vs_sd` whose definition is
"the attainment gap between the last rate that is still climbing and the first
rate at the plateau, against the measured per-cell seed SD".  The G1 probe ran
(job 900067) and that axis COULD NOT BE FILLED, because no predicate decided what
"the plateau" is.  The result document said so in writing rather than filling it:

    "`band_vs_sd`를 지금 채우면 미등록 기준으로 라벨을 만드는 것이고, 이 세션이
     반복해서 맞은 실패 형태다.  plateau 술어를 코드로 등록하는 것이 다음 판본의
     선결조건이며, 그 술어는 결과를 보기 전에 고정돼야 한다."
        -- RESULT_G1_900067_2026-09-01.md

This module is that predicate.  `cp0_g1_rule.py` is NOT modified: its label map,
its RULE_REV and its registered oracle entry all stay exactly as they are.  What
changes is that its axes now have registered producers.

★ DISCLOSURE, first and unhedged (this is the honest limit of this registration)
-------------------------------------------------------------------------------
The author of this file HAS ALREADY SEEN job 900067's table.  It is canon; it is
in `RESULT_G1_900067_2026-09-01.md`.  "Register the predicate before seeing the
results" is therefore no longer literally available for that job, and pretending
otherwise would be worse than saying it.  Three things are done instead, and the
pre-registration (`PREREG_G1B_PLATEAU_2026-09-01.md`) states them as binding:

  1. Every constant below is derived from a STATISTICAL TABLE or from an
     already-published repository measurement, never from 900067's numbers.  Each
     one carries its derivation inline.
  2. 900067 is NOT the scoring run.  The axis is scored on a FRESH ARRIVAL
     REALIZATION (new seeds), and 900067 is used only as a reference the result
     document may compare against.
  3. The one place 900067 unavoidably informed the design is the DIRECTION of the
     rate ladder's extension (upward), because 900067 published that throughput
     was still climbing at rate 8.  That is public canon rather than private
     knowledge, and it is declared here rather than hidden in a rationale.

★ What this module does NOT do (4th audit G2, the lesson that outlived the grid)
--------------------------------------------------------------------------------
G2 killed CP-0's capacity axis because a portable RATIO threshold was still
measured on an ABSOLUTE req/s grid, so the transferability defect merely MOVED
from the threshold to the grid.  The tempting repair -- re-express the grid in
offered tokens/s, since the two workloads differ 2096 vs 578 tok/req -- is NOT
taken here, and deliberately: the repository's own numbers do not support that
normalization either (the capscan arms plateau at 6.1k-9.3k tok/s while the
ShareGPT probe was still climbing at ~3.2k tok/s, so the two workloads do not
share a token-rate capacity).  Whether ANY normalization transfers is itself an
unmeasured question, and assuming one would be G2 in a new coat.

⇒ This module owns NO transferred grid.  It registers a STOPPING RULE instead: the
sweep continues up the ladder until the plateau predicate fires or a budget-derived
ceiling is hit.  A grid that is defined by the predicate cannot be the wrong grid;
it can only be too short, and running out of ladder has its own registered label
(`no_knee_in_range`), which is the honest name for that outcome.
"""

# ★THREE classes, not two.  `cp0_predicates.py` had only (label-bearing / infra),
# and that is exactly why `PROBE_RATES` -- a constant that decides WHICH DATA IS
# BOUGHT but not how it is labelled -- had to be filed as "infra" and verified with
# a check it could only ever pass.  A sweep constant is a real third thing:
#   LABEL  -- perturbing it MUST change an axis classification
#   DESIGN -- perturbing it must change NO axis classification, but MAY change the
#             stopping rule, because moving the sweep is what it is for
#   INFRA  -- perturbing it must change nothing at all
LABEL_CONSTANTS = ("T95_ONE_SIDED", "BAND_SD_MULT", "GRID_LO", "GRID_HI")
DESIGN_CONSTANTS = ("RATE_LADDER",)
INFRA_CONSTANTS = ("FLOAT_TOL",)


# ---------------------------------------------------------------- statistics
# One-sided 95% Student-t quantiles, df = n-1.  A TABLE, not a tuning knob: the
# 4th audit's third new lesson was "연속 축 등록 전에 그 통계량의 무처치 분산을 재고
# 유도 밴드가 그보다 넓은지 산술로 보여라", and the arithmetic it asks for is a
# one-sided test of "is this increment bigger than its own noise".  Values are the
# standard table (df=2: 2.920, df=3: 2.353, df=4: 2.132, df=5: 2.015, df=7: 1.895).
T95_ONE_SIDED = {2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015, 6: 1.943, 7: 1.895,
                 8: 1.860, 9: 1.833, 10: 1.812}

# The knee band is compared against ONE SD, not one standard error, and the choice
# is substantive rather than conservative-by-habit.  The question the axis asks is
# "could a THRESHOLD APPLIED TO ONE MEASUREMENT land inside this band" -- i.e. can a
# single realization's attainment tell the last climbing rate from the first plateau
# rate.  That is a question about the spread of individual measurements (SD), not
# about the precision of their mean (SE).  It is also the exact arithmetic the 4th
# audit performed when it killed the threshold: band 0.09 vs no-load SD 0.108.
BAND_SD_MULT = 1.0

# The registered grid whose bracketing `cp0_g1_rule.grid_brackets` asks about.
# NOT a grid this module sweeps -- it is the OBJECT of the question, inherited
# verbatim from `cp0_predicates.PROBE_RATES = (2, 4, 8)`.
GRID_LO = 2.0
GRID_HI = 8.0

# The ladder the sweep walks.  Its first five rungs ARE G1's own {2,3,4,6,8}; the
# extension is geometric (x1.5, rounded to the repository's habit of integer rates)
# and contains the canonical ShareGPT HI of 12.  It is a DESIGN constant, not a
# label-bearing one: the stopping rule decides where the sweep ends, and the ladder
# only says which rungs exist, so perturbing it must move the sweep and must NOT
# move any axis classification.  Running off the end of it produces
# `no_knee_in_range`, which the rule already carries as a label.
# ★The LADDER'S LENGTH IS THE BUDGET CEILING -- there is no second constant for it.
# The first draft of this module carried a separate `MAX_RATES = 8` "budget ceiling"
# guard, and this module's own mutation test showed it was DEAD: with a ladder of
# exactly 8 rungs, `len(done) >= MAX_RATES` can only fire when the ladder is already
# exhausted, so both branches returned None and no perturbation of it could change
# anything.  It is removed rather than declared, because "a constant that survives
# its own mutation test is dead, and the test says so" (`cp0_predicates.py`).
# The budget statement it carried is kept as a statement: at G1's measured ~55 s per
# rate per boot, 8 rungs x 3 seeds x 2 arms is ~0.75 GPU-hr.
RATE_LADDER = (2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 18.0, 27.0)

FLOAT_TOL = 1e-9


def _mean(xs):
    return sum(xs) / len(xs)


def _sd(xs):
    """Sample SD (n-1).  Refuses n<2: a spread cannot be estimated from one point."""
    n = len(xs)
    if n < 2:
        raise ValueError(f"need at least 2 observations for an SD, got {n}")
    mu = _mean(xs)
    return (sum((x - mu) ** 2 for x in xs) / (n - 1)) ** 0.5


def t95(n):
    """One-sided 95% t quantile for n paired observations (df = n-1)."""
    if n < 2:
        raise ValueError(f"need n>=2, got {n}")
    if n not in T95_ONE_SIDED:
        raise ValueError(f"no registered t quantile for n={n}; "
                         f"registered: {sorted(T95_ONE_SIDED)}")
    return T95_ONE_SIDED[n]


# ------------------------------------------------------------------- climbing
def climbing(achieved_lo, achieved_hi):
    """Is achieved throughput still RISING from one rate to the next?

    Inputs are the per-seed achieved throughputs at two adjacent rates, in the SAME
    SEED ORDER, so the increment is PAIRED.  Pairing matters: the arrival
    realization is the dominant nuisance here, and a paired test removes it instead
    of pooling it into the noise (the G1 result compared a raw increment against a
    single cell's SD, which is the unpaired -- and looser -- comparison).

    One-sided by construction: the alternative is "still climbing", never "falling".
    """
    if len(achieved_lo) != len(achieved_hi):
        raise ValueError(f"unpaired inputs: {len(achieved_lo)} vs {len(achieved_hi)}")
    n = len(achieved_lo)
    diffs = [hi - lo for lo, hi in zip(achieved_lo, achieved_hi)]
    sd = _sd(diffs)
    if sd == 0.0:
        # Degenerate but real (identical values across seeds).  A zero-spread rise is
        # climbing; a zero-spread non-rise is not.  Deciding it here keeps a
        # ZeroDivisionError from becoming an implicit label.
        return _mean(diffs) > FLOAT_TOL
    se = sd / (n ** 0.5)
    return _mean(diffs) > t95(n) * se + FLOAT_TOL


# -------------------------------------------------------------------- plateau
def knee(rates, achieved_by_rate):
    """Locate the knee: (last_climbing_rate, first_plateau_rate).

    `achieved_by_rate` maps rate -> per-seed achieved throughputs.  Walks the rates
    in ascending order and returns the first adjacent pair that stops climbing.
    Returns None when every pair is still climbing -- the `no_knee_in_range` case,
    which is a real outcome and not an error.
    """
    rs = sorted(rates)
    if len(rs) < 2:
        raise ValueError(f"need at least 2 rates to locate a knee, got {len(rs)}")
    missing = [r for r in rs if r not in achieved_by_rate]
    if missing:
        raise ValueError(f"no data for rate(s) {missing}")
    for lo, hi in zip(rs, rs[1:]):
        if not climbing(achieved_by_rate[lo], achieved_by_rate[hi]):
            return (lo, hi)
    return None


def band_vs_sd_axis(rates, achieved_by_rate, attainment_by_rate):
    """Produce the `cp0_g1_rule.band_vs_sd` axis value.

    `band_wider`       -- the attainment gap across the knee is wider than the
                          single-measurement noise, so a threshold can locate it
    `band_narrower`    -- it is not: the axis is NOT IDENTIFIED on this workload
    `no_knee_in_range` -- throughput never stopped climbing on the swept rates

    ★NOT CIRCULAR, and this is the property to check when auditing this file: the
    knee is located on ACHIEVED THROUGHPUT with a paired test of its own increments,
    while the band is measured in ATTAINMENT against the spread of individual
    attainment measurements.  Those are different statistics on different scales,
    so "we found a knee" does not arithmetically force either answer.
    """
    k = knee(rates, achieved_by_rate)
    if k is None:
        return "no_knee_in_range"
    lo, hi = k
    for r in (lo, hi):
        if r not in attainment_by_rate:
            raise ValueError(f"no attainment data for rate {r}")
    a_lo = attainment_by_rate[lo]
    a_hi = attainment_by_rate[hi]
    band = abs(_mean(a_lo) - _mean(a_hi))
    # Pooled single-measurement spread across the two cells that define the band.
    pooled_sd = ((_sd(a_lo) ** 2 + _sd(a_hi) ** 2) / 2) ** 0.5
    return "band_wider" if band > BAND_SD_MULT * pooled_sd + FLOAT_TOL else "band_narrower"


def grid_brackets_axis(rates, achieved_by_rate, grid_lo=None, grid_hi=None):
    """Produce the `cp0_g1_rule.grid_brackets` axis value.

    Asks the question the registered grid {2,4,8} was always meant to answer: does
    the knee actually lie inside it?  `yes` iff the located knee's plateau rate is
    within (GRID_LO, GRID_HI].  When no knee was located the rule declares the pair
    (`no_knee_in_range`, `yes`) incoherent, so this returns `no` -- the value that
    keeps the world coherent -- rather than inventing a third value.
    """
    g_lo = GRID_LO if grid_lo is None else grid_lo
    g_hi = GRID_HI if grid_hi is None else grid_hi
    if g_lo >= g_hi:
        raise ValueError(f"malformed grid: lo {g_lo} >= hi {g_hi}")
    k = knee(rates, achieved_by_rate)
    if k is None:
        return "no"
    _lo, plateau_rate = k
    return "yes" if (plateau_rate > g_lo + FLOAT_TOL
                     and plateau_rate <= g_hi + FLOAT_TOL) else "no"


def coverage_axis(rates, achieved_by_rate, n_seeds):
    """Produce the `cp0_g1_rule.coverage` axis value."""
    if n_seeds < 2:
        raise ValueError(f"the axis needs a spread, so n_seeds>=2; got {n_seeds}")
    got = [len(achieved_by_rate.get(r, [])) for r in rates]
    if all(g == 0 for g in got):
        return "absent"
    return "complete" if all(g == n_seeds for g in got) else "partial"


# -------------------------------------------------------------- stopping rule
def next_rate(rates_done, achieved_by_rate):
    """The sweep's stopping rule, in code (4th audit G2: own no transferred grid).

    Returns the next ladder rung to measure, or None when the sweep should stop.
    Stops when (a) the top two measured rates no longer climb -- the knee is
    bracketed -- or (b) the ladder / MAX_RATES budget is exhausted.
    """
    done = sorted(rates_done)
    if not done:
        return RATE_LADDER[0]
    if len(done) >= 2 and not climbing(achieved_by_rate[done[-2]],
                                       achieved_by_rate[done[-1]]):
        return None                      # knee bracketed: stop
    higher = [r for r in RATE_LADDER if r > done[-1] + FLOAT_TOL]
    # Ladder exhausted = budget exhausted: stop and let the caller report
    # `no_knee_in_range`, which is the honest name for "we ran out of ladder".
    return higher[0] if higher else None


# ---------------------------------------------------------------------- vectors
# Every vector is SYNTHETIC or drawn from a published repository measurement other
# than job 900067.  `g1b_plateau_selftest.py` asserts each expectation AND that
# perturbing the corresponding constant changes at least one of them.
CLIMBING_VECTORS = [
    # (achieved_lo, achieved_hi, expected)                      why
    ([1.0, 1.0, 1.0], [2.0, 2.0, 2.0], True),      # large, noiseless rise
    ([2.0, 2.0, 2.0], [2.0, 2.0, 2.0], False),     # flat
    ([2.0, 2.0, 2.0], [1.0, 1.0, 1.0], False),     # falling is not climbing
    ([1.0, 2.0, 3.0], [1.05, 2.5, 2.6], False),    # mean rises but paired sd swamps it
    ([1.0, 2.0, 3.0], [1.2, 2.2, 3.2], True),      # small but CONSISTENT paired rise
    # ★A vector whose verdict actually turns on the t quantile.  Without one, every
    # other climbing vector has either zero paired spread (decided by the sd==0
    # branch) or a spread so large the answer is the same at any quantile, and
    # `T95_ONE_SIDED` would pass its own load-bearing test while being dead.  Here
    # mean 0.500, sd 0.100, se 0.0577: the threshold at t=2.353 is 0.136 (climbing),
    # and any inflation of the table past t=8.66 flips it.
    ([1.0, 2.0, 3.0], [1.5, 2.4, 3.6], True),
]

KNEE_VECTORS = [
    # (rates, achieved_by_rate, expected knee)
    ((1.0, 2.0, 3.0),
     {1.0: [1.0, 1.0, 1.0], 2.0: [2.0, 2.0, 2.0], 3.0: [2.0, 2.0, 2.0]},
     (2.0, 3.0)),                                   # climbs then flattens
    ((1.0, 2.0, 3.0),
     {1.0: [1.0, 1.0, 1.0], 2.0: [2.0, 2.0, 2.0], 3.0: [3.0, 3.0, 3.0]},
     None),                                         # never stops climbing
    ((1.0, 2.0, 3.0),
     {1.0: [1.0, 1.0, 1.0], 2.0: [1.0, 1.0, 1.0], 3.0: [3.0, 3.0, 3.0]},
     (1.0, 2.0)),                                   # FIRST non-climbing pair wins
]

BAND_VECTORS = [
    # (rates, achieved_by_rate, attainment_by_rate, expected)
    ((1.0, 2.0, 3.0),
     {1.0: [1.0, 1.0, 1.0], 2.0: [2.0, 2.0, 2.0], 3.0: [2.0, 2.0, 2.0]},
     {1.0: [0.98, 0.98, 0.98], 2.0: [0.90, 0.91, 0.89], 3.0: [0.60, 0.61, 0.59]},
     "band_wider"),        # gap 0.30 vs spread ~0.01
    ((1.0, 2.0, 3.0),
     {1.0: [1.0, 1.0, 1.0], 2.0: [2.0, 2.0, 2.0], 3.0: [2.0, 2.0, 2.0]},
     {1.0: [0.98, 0.98, 0.98], 2.0: [0.80, 0.90, 0.70], 3.0: [0.78, 0.88, 0.68]},
     "band_narrower"),     # gap 0.02 vs spread 0.10 -- the 4th audit's arithmetic
    ((1.0, 2.0, 3.0),
     {1.0: [1.0, 1.0, 1.0], 2.0: [2.0, 2.0, 2.0], 3.0: [3.0, 3.0, 3.0]},
     {1.0: [0.98, 0.98, 0.98], 2.0: [0.95, 0.95, 0.95], 3.0: [0.92, 0.92, 0.92]},
     "no_knee_in_range"),
    # ★A vector whose verdict actually turns on BAND_SD_MULT.  The first draft had
    # none: vector 1's band is 30x its spread and vector 2's is 0.2x, so BOTH survive
    # a x10 perturbation and the constant passed its load-bearing test while being
    # untested.  Here the band is 0.05 against a pooled spread of 0.02 -- ratio 2.5,
    # so it is `band_wider` at multiplier 1 and `band_narrower` at anything over 2.5.
    ((1.0, 2.0, 3.0),
     {1.0: [1.0, 1.0, 1.0], 2.0: [2.0, 2.0, 2.0], 3.0: [2.0, 2.0, 2.0]},
     {1.0: [0.98, 0.98, 0.98], 2.0: [0.88, 0.90, 0.92], 3.0: [0.83, 0.85, 0.87]},
     "band_wider"),
]

GRID_VECTORS = [
    # (rates, achieved_by_rate, expected)
    ((2.0, 4.0, 8.0),
     {2.0: [1.9, 1.9, 1.9], 4.0: [3.5, 3.5, 3.5], 8.0: [3.5, 3.5, 3.5]},
     "yes"),               # plateau first seen at 8 -- inside (2, 8]
    ((2.0, 4.0, 8.0),
     {2.0: [1.9, 1.9, 1.9], 4.0: [3.5, 3.5, 3.5], 8.0: [7.0, 7.0, 7.0]},
     "no"),                # never plateaus inside the grid
    ((8.0, 12.0, 18.0),
     {8.0: [5.5, 5.5, 5.5], 12.0: [8.0, 8.0, 8.0], 18.0: [8.0, 8.0, 8.0]},
     "no"),                # plateau at 18 -- above the registered grid
]

COVERAGE_VECTORS = [
    ((2.0, 4.0), {2.0: [1, 2, 3], 4.0: [1, 2, 3]}, 3, "complete"),
    ((2.0, 4.0), {2.0: [1, 2, 3], 4.0: [1, 2]}, 3, "partial"),
    ((2.0, 4.0), {}, 3, "absent"),
]

NEXT_RATE_VECTORS = [
    # (rates_done, achieved_by_rate, expected)
    ((), {}, 2.0),                                             # start at the bottom rung
    ((2.0,), {2.0: [1.9, 1.9, 1.9]}, 3.0),                     # only one point: keep going
    ((2.0, 3.0), {2.0: [1.9, 1.9, 1.9], 3.0: [2.8, 2.8, 2.8]}, 4.0),   # still climbing
    ((2.0, 3.0), {2.0: [2.8, 2.8, 2.8], 3.0: [2.8, 2.8, 2.8]}, None),  # knee bracketed
    # ★Exercises the ladder END -- without this the only DESIGN constant moved
    # nothing on any registered vector, i.e. it was untested.
    ((2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 18.0),
     {2.0: [1.0, 1.1, 1.2], 3.0: [2.0, 2.1, 2.2], 4.0: [3.0, 3.1, 3.2],
      6.0: [4.0, 4.1, 4.2], 8.0: [5.0, 5.1, 5.2], 12.0: [6.0, 6.1, 6.2],
      18.0: [7.0, 7.1, 7.2]},
     27.0),
]
