"""D1 rev2 predicates: the DATA -> AXIS functions.  PRED_REV = 3.

rev1 had none of this.  The 1st audit's U4 was that `bracket_ok` and the
degenerate-end predicates existed only inside a docstring, so the
pre-registration's "✅ 닫힘" for free surfaces F1/F2 rested on functions that
were never written.  Every axis value the rule reads is produced here, and the
self-test exercises each one on fixtures.

★ What changed from rev1's DESIGN (1st audit, `audit_d1_rules_2026-09-07`)
--------------------------------------------------------------------------
U3 (the death cause that decided the verdict): scoring every arm by `plain`'s
uncontended floor puts OPPOSITE-SIGNED biases on the two grid axes.  Measured,
GPU 0 (`RESULT_D1_A1A2_2026-09-07.md`): occupancy-weighted differential bias on
the registered pair is TTFT -5.07% (tilts to `d44`) and ITL p95 +5.97% (tilts to
`cp2048`).  Both exceed `PRACTICAL_FLOOR = 3%`, and their SHAPE is exactly the
crossing rev1 had registered as its prediction.  A single yardstick could
manufacture the predicted answer at zero load.
  -> rev2 scores EVERY point TWICE (fixed yardstick / each arm's own floor) and
     takes the INTERSECTION: `sign_intersect`.  A yardstick can now only REMOVE
     a sign, never create one.  Where the two scorings give OPPOSITE signs the
     yardstick decides the direction, and that is its own label.

U2: rev1 asserted the treatment fires on 100% of the workload, citing a census
run with the SELECTION tokenizer.  Re-tokenised with the campaign tokenizer:
`cps 2048` fires on 79.67%, `cps 4096` on 7.42%, `cps 8192` on 0.00%.
  -> `FIRING_RATE` below, and it is a MEASURED constant, not a claim.
★ What changed in rev3 (2nd audit, `audit_d1_rules_2nd_2026-09-07`, V1-V10)
--------------------------------------------------------------------------
V2 (and the finding behind it): the sign call ran on a PERCENTILE BOOTSTRAP CI,
whose true-null false-positive rate at n=4 is ~20% against a nominal 5%.  This
was NOT a new discovery -- `PROJECT_STATUS.md` methodology gate **#14**
(2026-08-06) already registered it with coverage n=4 0.798 / n=5 0.840 /
n=6 0.859 / n=8 0.888 and the prescription *"at n<=8 do not use the
paired/unpaired bootstrap interval for a verdict; primary is the t-CI,
bootstrap is reported alongside only"*.  This session reproduced those
coverages independently (0.802 / 0.838 / 0.860 / 0.882; the t-CI measures
0.949-0.954).  rev2 walked into a gate that had been standing for a month.
  -> `paired_t_ci` is the primary and the only label-bearing interval.
     `paired_bootstrap_ci` survives as a REPORTED companion and may not decide
     a sign.  `permutation_p` is the distribution-free companion.
  -> `MIN_BOOTS` 4 -> 6, because the exact paired sign-flip test's two-sided p
     floor is 2/2^n: n=4 gives 0.125 and n=5 gives 0.0625, so neither can reach
     0.05 at all; n=6 gives 0.03125 and can.  (Same arithmetic gate #14 used to
     retire the project's n=5 paired cells.)

V1: `bracket_ok` forces a degenerate-violating point into the grid, `call_sign`
sent it to a fixed band, and that alone collapsed `band_binding` to two values
-- killing both purchasable branches and making the track's only termination
label unreachable.
  -> the degenerate ends are BRACKETING WITNESSES, not comparison points: they
     are excluded from the sign field and from the leave-one-out set
     (`interior_points`).  The satisfying end carries a THROUGHPUT difference,
     which is reported separately by `throughput_end_delta` rather than being
     silently mixed into an SLO sign.
  -> `band_binding` now asks the question that actually distinguishes the two
     null labels: is the interval PRECISE enough to assert "the effect is below
     the practical floor" (equivalence), or merely inconclusive?

"""
import ast  # noqa: F401 -- imported so the self-test can AST-scan this module
import math
import statistics

PRED_REV = 3

# ----------------------------------------------------------------- measured
# Nemotron-Nano-9B-v2 tokenizer over ShareGPT_long2048_cap8192.json (n=1,968).
# `RESULT_D1_A1A2_2026-09-07.md`.  A prompt splits iff len > cps.
# ★VALUES CORRECTED 2026-09-07 (2nd audit V3); ★CAUSAL CLAIM RETRACTED 2026-09-08
# (3rd audit W5, re-measured here).  rev2 registered {512: 0.9995, 2048: 0.7967} and
# both values were wrong.  The three conventions that fix the numbers are: the
# canonical prompt is turn 0 of a >=2-turn conversation (`af1_strata.py:59`), the
# tokenizer is called with `add_special_tokens=False` (only that reproduces
# `af1_strata.json`'s quantiles; the default adds exactly +1.0 tok to every one),
# and a prompt splits iff `len > cps`.  All three are registered together.
#
# ★ What this comment must NOT say, and said until 2026-09-08: that the `>=`
# hypothesis was measured and rejected.  At cps 2048 the two explanations are
# OBSERVATIONALLY EQUIVALENT -- (canonical, `>=`) and (default, `>`) both give
# 1568/1968 = 79.6748%, which is exactly rev2's registered number.  The earlier
# "measured: `len >= 2048` gives 79.73%" was `>=` evaluated under the WRONG
# convention, i.e. two variables moved at once to reject one of them (confound #10
# in its diagnostic form).  The values below stand; the causal attribution does not.
FIRING_RATE = {512: 1.0000, 1024: 0.9995, 2048: 0.7952, 4096: 0.0742, 8192: 0.0000}
# AF-1 strata, campaign tokenizer (`af1_strata.json`).  These are the ONLY
# lengths at which an uncontended floor was measured, so they are the buckets:
# no interpolation, no fitting (gate: the floor is measured, never modelled).
STRATA_TOK = (1932.7, 2514.0, 3764.0, 5925.3)
# `RESULT_AF1_2026-09-07.md` sec 2, n=6 boots, ms, in STRATA_TOK order.
FLOOR_TTFT_MS = {"plain":  (235.9, 297.1, 422.1, 645.9),
                 "cp2048": (235.9, 309.5, 450.0, 697.5),
                 "d44":    (229.3, 291.6, 419.4, 666.0)}
FLOOR_ITLP95_MS = {"plain":  (12.27, 12.25, 12.29, 12.30),
                   "cp2048": (12.27, 12.29, 12.31, 12.33),
                   "d44":    (13.07, 13.06, 13.09, 13.11)}
YARDSTICK_ARM = "plain"

# ------------------------------------------------------------ folds printed
# 6th-audit lesson 6: a fold that produces a number the result document prints
# must be on a list something checks.  A fold not here may not appear in the
# result document.
PREDICATE_FOLDS = (
    "bucket_of", "request_passes", "goodput", "degenerate_low",
    "degenerate_high", "bracket_ok", "extend_grid", "paired_bootstrap_ci",
    "paired_t_ci", "paired_t_p", "t_crit_for", "holm_calls",
    "permutation_p",
    "call_sign", "sign_intersect",
    "intersect_reason", "interior_points", "throughput_end_delta",
    "estimator_class", "sign_field_of",
    "band_binding_of", "loo_stability", "yardstick_bias", "variance_of",
)


# --------------------------------------------------------------- bucketing
def bucket_of(prompt_tok, strata=STRATA_TOK):
    """Nearest measured stratum in LOG length.  Outside the measured range the
    end bucket applies FLAT (no extrapolation of the floor curve)."""
    if prompt_tok <= 0:
        raise ValueError("prompt length must be positive")
    return min(range(len(strata)),
               key=lambda i: abs(math.log(prompt_tok) - math.log(strata[i])))


def yardstick_bias(arm, occupancy, floors):
    """Occupancy-weighted ratio floor[YARDSTICK]/floor[arm].  >1 = the arm is
    handed a LOOSER budget than its own physics; <1 = handicapped."""
    if abs(sum(occupancy) - 1.0) > 1e-6:
        raise ValueError("occupancy must sum to 1")
    return sum(o * (floors[YARDSTICK_ARM][i] / floors[arm][i])
               for i, o in enumerate(occupancy))


# ----------------------------------------------------------------- goodput
def request_passes(rec, k_t, k_i, floor_t, floor_i):
    """gate #4: TTFT <= budget AND the request's OWN token-ITL p95 <= budget.
    A request that never completed has `ttft_ms is None` and is a VIOLATION --
    it is not absent, it is a miss (otherwise starving requests looks good)."""
    if rec.get("ttft_ms") is None or rec.get("itl_p95_ms") is None:
        return False
    b = bucket_of(rec["prompt_tok"])
    return (rec["ttft_ms"] <= k_t * floor_t[b]
            and rec["itl_p95_ms"] <= k_i * floor_i[b])


def goodput(records, k_t, k_i, floor_t, floor_i, duration_s):
    """req/s.  `duration_s` is the SUM of segment durations (gate #7: max()
    inflated every reported goodput by ~3x once already)."""
    if duration_s <= 0:
        raise ValueError("duration must be positive; pass the SUM of segments")
    return sum(1 for r in records
               if request_passes(r, k_t, k_i, floor_t, floor_i)) / duration_s


# ------------------------------------------------------- degenerate ends
DEGEN_HIGH_TOL = 0.02   # within 2% of the arm's own achieved throughput


def degenerate_low(g_by_arm):
    """Every arm scores exactly zero: nothing meets the SLO."""
    return all(g == 0.0 for g in g_by_arm.values())


def degenerate_high(g_by_arm, achieved_by_arm, tol=DEGEN_HIGH_TOL):
    """Every arm's goodput has reached its own achieved throughput: the SLO has
    stopped binding, and any residual difference is a THROUGHPUT difference, not
    an SLO one.  (1st audit L5/U5: the sign at this end is a throughput sign;
    the result document must say so.)"""
    if not achieved_by_arm or any(a <= 0 for a in achieved_by_arm.values()):
        return False
    return all(g_by_arm[a] >= achieved_by_arm[a] * (1.0 - tol)
               for a in achieved_by_arm)


def interior_points(grid_scores, achieved_by_arm):
    """★V1 repair.  The degenerate ends are BRACKETING WITNESSES: at the
    violating end both arms are pinned to zero by construction (no information
    about their difference), and at the satisfying end the difference is a
    THROUGHPUT difference, a different estimand.  Neither is a comparison
    point, so neither enters the sign field or the leave-one-out set.  What the
    satisfying end does carry is reported by `throughput_end_delta`."""
    return tuple(sorted(p for p, g in grid_scores.items()
                        if not degenerate_low(g)
                        and not degenerate_high(g, achieved_by_arm)))


def throughput_end_delta(grid_scores, achieved_by_arm, arm_a, arm_b):
    """The difference at the degenerate-satisfying end, reported so that
    excluding it from the sign field does not hide it (2nd audit U5/L5)."""
    highs = [p for p, g in grid_scores.items()
             if degenerate_high(g, achieved_by_arm)]
    if not highs:
        return None
    p = min(highs)
    return grid_scores[p][arm_a] - grid_scores[p][arm_b]


def bracket_ok(grid_scores, achieved_by_arm):
    """`grid_scores`: {(k_t, k_i): {arm: goodput}}.  The grid brackets the
    question iff it contains at least one degenerate-violating point AND at
    least one degenerate-satisfying point."""
    lows = [p for p, g in grid_scores.items() if degenerate_low(g)]
    highs = [p for p, g in grid_scores.items()
             if degenerate_high(g, achieved_by_arm)]
    return bool(lows) and bool(highs)


def extend_grid(k_values, k_max):
    """Doubling extension.  Deterministic and registered, so widening the grid
    is part of the analyser, not a discretionary repair after seeing a result.
    Costs zero GPU: the arms are SLO-agnostic, so one realised run is re-scored."""
    nxt = k_values[-1] * 2
    if nxt > k_max:
        return None
    return tuple(k_values) + (nxt,)


# ------------------------------------------------------------- sign calling
def paired_t_ci(deltas, level, t_crit):
    """★PRIMARY interval (methodology gate #14).  Paired t on the differences.
    `t_crit` is passed in from the rule's registered table so the critical value
    is a DECISION constant, not something this module invents."""
    n = len(deltas)
    if n < 2:
        raise ValueError("a CI needs at least two paired boots")
    m = statistics.fmean(deltas)
    hw = t_crit * statistics.stdev(deltas) / math.sqrt(n)
    return m - hw, m + hw


def paired_bootstrap_ci(deltas, level, n_boots, rng):
    """COMPANION ONLY -- reported, never label-bearing.  Gate #14: at n<=8 the
    percentile bootstrap of the mean under-covers badly (measured true-null
    false-positive ~20% at n=4 against a nominal 5%).  `call_sign` does not
    accept this interval; the self-test asserts that it cannot."""
    n = len(deltas)
    if n < 2:
        raise ValueError("a CI needs at least two paired boots")
    means = sorted(sum(deltas[rng.randrange(n)] for _ in range(n)) / n
                   for _ in range(n_boots))
    lo = means[int((1 - level) / 2 * (n_boots - 1))]
    hi = means[int((1 + level) / 2 * (n_boots - 1))]
    return lo, hi


def permutation_p(deltas):
    """Exact paired sign-flip test, two-sided.  Distribution-free COMPANION.
    Its p floor is 2/2**n, so it is informative only from n=6 up -- that bound
    is why `MIN_BOOTS` is 6 and it is asserted in the self-test."""
    n = len(deltas)
    obs = abs(statistics.fmean(deltas))
    hits = 0
    for mask in range(2 ** n):
        flipped = [d if (mask >> i) & 1 else -d for i, d in enumerate(deltas)]
        if abs(statistics.fmean(flipped)) >= obs - 1e-15:
            hits += 1
    return hits / 2 ** n


# Numerical-method tolerances for the incomplete beta / bisection.  They are
# NOT decision constants -- no verdict depends on their value beyond convergence
# -- but they are named so that no threshold hides inside a function body
# (self-test C12d), and `NUMERICAL_TOLERANCES` must cover them exactly (C12e).
_BETACF_ITMAX = 300
_BETACF_EPS = 3e-14
_TINY = 1e-30
_TCRIT_HI = 10000.0
_TCRIT_ITERS = 200
NUMERICAL_TOLERANCES = ("_BETACF_ITMAX", "_BETACF_EPS", "_TINY",
                        "_TCRIT_HI", "_TCRIT_ITERS")


def _betacf(a, b, x, itmax=_BETACF_ITMAX, eps=_BETACF_EPS):
    """Lentz continued fraction for the incomplete beta.  Standard; no SciPy in
    this environment and the decision must not depend on an unpinned import."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < _TINY:
        d = _TINY
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        if abs(d) < _TINY:
            d = _TINY
        if abs(c) < _TINY:
            c = _TINY
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        if abs(d) < _TINY:
            d = _TINY
        if abs(c) < _TINY:
            c = _TINY
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


def _betai(a, b, x):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
             + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lbeta) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbeta) * _betacf(b, a, 1.0 - x) / b


def paired_t_p(deltas):
    """Two-sided p of the paired t on the differences.  Used only for the
    MULTIPLICITY correction (`holm_calls`); the CI remains the primary object."""
    n = len(deltas)
    if n < 2:
        raise ValueError("a t test needs at least two paired boots")
    sd = statistics.stdev(deltas)
    if sd == 0.0:
        return 0.0 if statistics.fmean(deltas) != 0.0 else 1.0
    t = statistics.fmean(deltas) / (sd / math.sqrt(n))
    nu = n - 1
    return _betai(nu / 2.0, 0.5, nu / (nu + t * t))


def t_crit_for(alpha, df, hi=_TCRIT_HI):
    """Two-sided t critical value at `alpha`, by bisection on `paired_t_p`'s own
    tail function -- so the critical value and the p-value can never disagree."""
    def tail(t):
        return _betai(df / 2.0, 0.5, df / (df + t * t))
    lo = 0.0
    for _ in range(_TCRIT_ITERS):
        mid = (lo + hi) / 2.0
        if tail(mid) > alpha:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def holm_calls(pvals, alpha):
    """Holm-Bonferroni step-down over the INTERIOR grid points.

    ★ Why this exists (2nd audit 5-A/B1): the sign field is a DISJUNCTION over
    the grid, so a per-point alpha is not the rate at which a direction appears.
    With ~40 interior points at alpha=0.05 the expected number of false calls is
    about two -- a directional label could be manufactured by multiplicity alone,
    which is the same failure mode as rev1's yardstick one level up.

    Returns a tuple of booleans, aligned with `pvals`."""
    m = len(pvals)
    if m == 0:
        return ()
    order = sorted(range(m), key=lambda i: pvals[i])
    out = [False] * m
    for rank, i in enumerate(order):
        if pvals[i] <= alpha / (m - rank):
            out[i] = True
        else:
            break                      # step-down: stop at the first failure
    return tuple(out)


def call_sign(deltas, max_g, practical_floor, level, t_crit):
    """(sign, band) for one grid point, from the PRIMARY interval.

    band:
      `called`    the t-CI excludes 0 AND |delta|/max(G) >= practical_floor
      `precise`   not called, but the CI half-width is BELOW the practical
                  floor -- the measurement is sharp enough to assert that the
                  effect is smaller than what this project calls a result
                  (an equivalence statement, and the only honest route to
                  `PAIR_INDISTINGUISHABLE`)
      `imprecise` not called and the half-width is at or above the floor --
                  this measurement cannot resolve a floor-sized effect, which
                  is what `buy_boots` is for

    ★ max(G) == 0 convention: both arms scored zero, so there is no difference
    to size and no boots can change that.  The band is `precise`."""
    m = statistics.fmean(deltas)
    lo, hi = paired_t_ci(deltas, level, t_crit)
    hw = (hi - lo) / 2.0
    if max_g <= 0:
        return 0, "precise"
    rel_hw = hw / max_g
    rel = abs(m) / max_g
    if ((lo > 0) or (hi < 0)) and rel >= practical_floor:
        return (1 if m > 0 else -1), "called"
    return 0, ("precise" if rel_hw < practical_floor else "imprecise")


def sign_intersect(sign_fixed, sign_own):
    """★ The U3 repair.  A sign survives only if BOTH scorings produce it.
    A yardstick can remove a sign; it can never create one."""
    if sign_fixed == sign_own:
        return sign_fixed
    return 0


def intersect_reason(s_fixed, s_own, r_fixed, r_own):
    """Why the INTERSECTED call came out the way it did.  Opposite signs ->
    `imprecise`: the two yardsticks disagree in direction, which no number of
    boots resolves, and the world it produces is caught by the
    `strong_discordant` guard before any label is read."""
    if s_fixed == s_own and s_fixed != 0:
        return "called"
    if s_fixed * s_own < 0:
        return "imprecise"
    return r_fixed if s_fixed == 0 else r_own


def estimator_class(pairs):
    """`pairs`: [(sign_fixed, sign_own), ...] over the grid.
    `strong_discordant` = some point where the two yardsticks give OPPOSITE
    signs -- there the yardstick, not the system, sets the direction."""
    if any(a * b < 0 for a, b in pairs):
        return "strong_discordant"
    if any(a != b for a, b in pairs):
        return "weak_discordant"
    return "concordant"


def sign_field_of(signs):
    pos, neg = any(s > 0 for s in signs), any(s < 0 for s in signs)
    if pos and neg:
        return "crossing"
    if pos:
        return "cp_dominant"
    if neg:
        return "pd_dominant"
    return "null"


def band_binding_of(reasons):
    """What the 0-calls were: `none` (there were none), `precise` (every one is
    an equivalence statement), `imprecise` (at least one could not resolve a
    floor-sized effect -- conservative, since one unresolved point is enough to
    stop the field being an equivalence claim)."""
    z = [r for r in reasons if r != "called"]
    if not z:
        return "none"
    return "imprecise" if any(r == "imprecise" for r in z) else "precise"


def loo_stability(signs_by_point):
    """Drop each grid point in turn: does `sign_field_of` change?
    ★ The degenerate ends are INCLUDED in the leave-one-out set.  They are grid
    points and dropping one may change the field; excluding them would be a
    silent choice about which points count."""
    pts = list(signs_by_point)
    if len(pts) < 2:
        raise ValueError("leave-one-out needs at least two grid points")
    base = sign_field_of([signs_by_point[p] for p in pts])
    for drop in pts:
        rest = [signs_by_point[p] for p in pts if p != drop]
        if sign_field_of(rest) != base:
            return "grid_fragile"
    return "stable"


def variance_of(deltas):
    """Between-boot SD of the paired difference, reported alongside every
    field (gate #3: baseline variance is measured, not assumed)."""
    return statistics.stdev(deltas) if len(deltas) > 1 else float("nan")
