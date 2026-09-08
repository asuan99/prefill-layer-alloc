"""D1 rev2 predicates: the DATA -> AXIS functions.  PRED_REV = 2.

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
"""
import math
import statistics

PRED_REV = 2

# ----------------------------------------------------------------- measured
# Nemotron-Nano-9B-v2 tokenizer over ShareGPT_long2048_cap8192.json (n=1,968).
# `RESULT_D1_A1A2_2026-09-07.md`.  A prompt splits iff len > cps.
# ★CORRECTED 2026-09-07 (2nd audit V3, independently re-verified).  rev2 registered
# {512: 0.9995, 2048: 0.7967} and BOTH were wrong.  The cause is NOT the `>` vs `>=`
# convention (the audit's diagnosis; measured: `len >= 2048` gives 79.73%, not 79.67%)
# -- it is the tokenizer's SPECIAL-TOKEN convention.  Only `add_special_tokens=False`
# reproduces `af1_strata.json`'s quantiles exactly (the default adds precisely +1.0 tok
# to every quantile), and the canonical prompt is turn 0 of a >=2-turn conversation
# (`af1_strata.py:59`).  Under that convention, with this module's own rule
# "a prompt splits iff len > cps":
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
    "call_sign", "sign_intersect", "intersect_reason",
    "estimator_class", "sign_field_of",
    "band_binding_of", "loo_stability", "yardstick_bias",
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
def paired_bootstrap_ci(deltas, level, n_boots, rng):
    """Percentile CI of the mean paired difference across boots."""
    n = len(deltas)
    if n < 2:
        raise ValueError("a CI needs at least two paired boots")
    means = []
    for _ in range(n_boots):
        means.append(sum(deltas[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int((1 - level) / 2 * (n_boots - 1))]
    hi = means[int((1 + level) / 2 * (n_boots - 1))]
    return lo, hi


def call_sign(delta_mean, ci, max_g, practical_floor):
    """(sign, reason).  A point is called only if the CI excludes 0 AND the
    relative size clears `practical_floor`.

    ★ max(G) == 0 convention (1st audit, U4's missing rule): both arms scored
    zero, so there is no difference to size.  The point is a 0-call and the
    binding reason is `floor` -- NOT `ci` -- because no amount of extra boots
    changes a difference that is identically zero."""
    lo, hi = ci
    ci_excludes_zero = (lo > 0) or (hi < 0)
    if max_g <= 0:
        return 0, "floor"
    rel = abs(delta_mean) / max_g
    clears_floor = rel >= practical_floor
    if ci_excludes_zero and clears_floor:
        return (1 if delta_mean > 0 else -1), "called"
    if not ci_excludes_zero and not clears_floor:
        return 0, "both"
    return 0, ("ci" if not ci_excludes_zero else "floor")


def sign_intersect(sign_fixed, sign_own):
    """★ The U3 repair.  A sign survives only if BOTH scorings produce it.
    A yardstick can remove a sign; it can never create one."""
    if sign_fixed == sign_own:
        return sign_fixed
    return 0


def intersect_reason(s_fixed, s_own, r_fixed, r_own):
    """Why the INTERSECTED call came out the way it did.
    Opposite signs -> `both`: the direction was set by the yardstick, and no
    number of boots buys past that (it is not a CI problem)."""
    if s_fixed == s_own and s_fixed != 0:
        return "called"
    if s_fixed * s_own < 0:
        return "both"
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
    """What produced the 0-calls.  `none` = there were none."""
    z = [r for r in reasons if r != "called"]
    if not z:
        return "none"
    has_ci, has_floor = "ci" in z, "floor" in z
    if "both" in z or (has_ci and has_floor):
        return "both"
    return "ci" if has_ci else "floor"


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
