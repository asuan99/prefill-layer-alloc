"""CP-2 rev2 data -> axis maps.  RULE_REV-independent module.

rev1 died on the 5th audit's H1: it registered a 3% equivalence margin without ever
measuring the estimator's untreated variance, so `equivalent` was unreachable and
the arithmetic was deferred (`PREREG_CP2 sec 12.6`) -- a free parameter on the
decision path.  rev2's answer is NOT to move the margin.  It is to split the two
things rev1 conflated:

    the MARGIN is the scientific question   -- fixed, externally justified, 3%
    the N      is the price of answering it -- DERIVED from measured variance

So `required_n()` below is a registered function, and the campaign either buys
that n or declares the question UNAFFORDABLE at the registered budget and says
what it could have answered instead.  Nothing here is chosen after seeing a
contrast.

★Three classes of constant, as in `g1b_plateau_predicates.py`:
   LABEL  -- perturbing MUST change an axis classification
   DESIGN -- perturbing must change NO axis classification (it moves what is bought)
   INFRA  -- perturbing must change nothing at all
"""

LABEL_CONSTANTS = ("EQUIV_MARGIN_REL", "T975", "LADDER_TTFT_MS", "LADDER_ITL_MS")
DESIGN_CONSTANTS = ("BUDGET_MAX_PAIRS",)
INFRA_CONSTANTS = ("FLOAT_TOL",)


# ---------------------------------------------------------------- the margin
# NOT re-derived and NOT tuned: /scratch/ehmoon/whlee/CLAUDE.md "방법론 게이트",
# opening phrase "3% 미만 차이는 headline 아님".  It is the size of difference this
# repository has already decided is not a finding, so it is the size the comparison
# must be powered for.  Moving it would change the QUESTION, and that change would
# have to be argued in the pre-registration, not here.
EQUIV_MARGIN_REL = 0.03

# Two-sided 95% Student-t, df = n-1.  A table.
T975 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365,
        9: 2.306, 10: 2.262, 12: 2.201, 14: 2.160, 16: 2.131, 18: 2.110,
        20: 2.093, 25: 2.064, 30: 2.045, 40: 2.023, 60: 2.001}

# The SLO ladder.  ITL leg is `CONSENSUS.md` sec 3 item 89's 40-100 ms ladder, which
# the 5th audit named as the replacement for rev1's single 60 ms threshold; the TTFT
# leg spans the range W1 measured the arms' TTFT distributions across.
# ★rev1 claimed its band was "inherited verbatim from canon" and that was FALSE for
# the ITL leg (5th audit H4: canon contains `54ms` zero times).  This one is cited to
# a document that exists and the citation was checked this session.
LADDER_TTFT_MS = (500.0, 1000.0, 2000.0, 3000.0, 4000.0, 6000.0)
LADDER_ITL_MS = (40.0, 50.0, 60.0, 70.0, 80.0, 100.0)

# Budget ceiling in PAIRS (one pair = one job holding both arms).  A DESIGN constant:
# it decides what is bought, never how a bought thing is labelled.  Basis: W1 measured
# ~45 min for a 4-arm x 4-rate job on this model; a 2-arm x 1-rate pair is ~15 min, so
# 24 pairs ~= 6 GPU-hr.  That is the largest single spend this track has proposed and
# it is registered as the cap rather than discovered afterwards.
BUDGET_MAX_PAIRS = 24

FLOAT_TOL = 1e-9


def _lt(a, b):
    return a < b - FLOAT_TOL


def _gt(a, b):
    return a > b + FLOAT_TOL


def half_width(sd, n, t_table=None):
    """Two-sided 95% half-width of the paired mean at this n."""
    t = T975 if t_table is None else t_table
    if n < 2:
        raise ValueError(f"need n>=2, got {n}")
    if sd < 0:
        raise ValueError(f"sd must be non-negative, got {sd}")
    if n not in t:
        raise ValueError(f"no registered t quantile for n={n}; registered: {sorted(t)}")
    return t[n] * sd / (n ** 0.5)


def required_n(sd, margin=None, cap=None):
    """Smallest registered n whose half-width clears the margin, or None if none does
    within the budget cap.  THE answer to H1: n is derived, the margin is not."""
    m = EQUIV_MARGIN_REL if margin is None else margin
    c = BUDGET_MAX_PAIRS if cap is None else cap
    if m <= 0:
        raise ValueError(f"margin must be positive, got {m}")
    for n in sorted(k for k in T975 if k <= c):
        if _lt(half_width(sd, n), m):
            return n
    return None


def answerable_margin(sd, cap=None):
    """If the registered margin is unaffordable, what margin IS affordable at the cap?
    Reported so that `UNAFFORDABLE` says what the spend COULD have bought rather than
    only what it could not."""
    c = BUDGET_MAX_PAIRS if cap is None else cap
    ns = [k for k in T975 if k <= c]
    if not ns:
        raise ValueError(f"budget cap {c} admits no registered n")
    return half_width(sd, max(ns))


def power_axis(sd, n_planned, margin=None, cap=None):
    """`sufficient` / `unaffordable` / `unmeasured` -> cp2r2_rule.power"""
    if sd is None or n_planned is None:
        return "unmeasured"
    need = required_n(sd, margin, cap)
    if need is None:
        return "unaffordable"
    return "sufficient" if n_planned >= need else "unaffordable"


# --------------------------------------------------------------- the contrast
def contrast_axis(ci_lo, ci_hi, margin=None):
    """TOST on the paired relative difference (cp - pdmux)/pdmux.

    Boundary EXCLUDED: a CI that merely touches the margin has not cleared the
    repository's own "3% 미만은 headline 아님" bar.
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


# ------------------------------------------------------------------ rule E
def ladder_axis(contrast_by_point):
    """Is the contrast the SAME at every point of the registered SLO ladder?

    Input: the contrast label at every (TTFT, ITL) point -- the full product, in any
    order.  A short or long sequence is REFUSED: a consistency claim over a subset of
    the ladder is not the claim that was registered (this is what makes the two ladder
    constants load-bearing rather than decorative).

    ★Why this is the primary axis in rev2 and was only a guard in rev1: W1 measured
    the two families being limited by OPPOSITE SLO legs (pdmux flat in ITL and steep
    in TTFT; chunked prefill the reverse), so "which is better" is a function of the
    SLO pair.  `inconsistent` is therefore a SUBSTANTIVE finding, not a failure --
    it says the answer does not exist independent of the operating point.
    """
    vals = list(contrast_by_point)
    n_expected = len(LADDER_TTFT_MS) * len(LADDER_ITL_MS)
    if len(vals) != n_expected:
        raise ValueError(f"rule E needs the FULL ladder: expected {n_expected} points "
                         f"({len(LADDER_TTFT_MS)} TTFT x {len(LADDER_ITL_MS)} ITL), "
                         f"got {len(vals)}")
    if any(v is None or v == "unmeasured" for v in vals):
        return "unmeasured"
    return "consistent" if len(set(vals)) == 1 else "inconsistent"


def coverage_axis(pairs_completed, n_required):
    """`complete` / `partial` / `absent` -> cp2r2_rule.coverage.
    n_required comes from `required_n`, NOT from a hard-coded 4 -- that is the whole
    point of rev2 (H1)."""
    if pairs_completed < 0:
        raise ValueError(f"negative pair count: {pairs_completed}")
    if n_required is None or n_required < 2:
        raise ValueError(f"n_required must be a registered n>=2, got {n_required}")
    if pairs_completed == 0:
        return "absent"
    return "complete" if pairs_completed >= n_required else "partial"


# ---------------------------------------------------------------------- vectors
POWER_VECTORS = [
    # (sd, n_planned, expected)
    (0.010, 4, "sufficient"),      # hw = 3.182*0.010/2 = 0.0159 < 0.03
    (0.020, 4, "unaffordable"),    # hw = 0.0318 > 0.03 -> needs n=5
    (0.020, 5, "sufficient"),      # hw = 2.776*0.020/2.236 = 0.0248 < 0.03
    (0.060, 20, "sufficient"),     # hw = 2.093*0.060/4.472 = 0.02808 < 0.03
    # ★A boundary vector, and the session's hand-arithmetic got it WRONG before the
    # code corrected it: hw(0.060, 18) = 2.110*0.060/4.243 = 0.02984, which is UNDER
    # 0.03, so required_n(0.060) is 18 and n_planned=18 is SUFFICIENT.  Kept exactly
    # because it sits 0.00016 from the margin -- if a later edit moves the t table or
    # the margin, this vector is the one that notices.
    (0.060, 18, "sufficient"),
    (0.060, 16, "unaffordable"),   # hw = 2.131*0.060/4.0 = 0.03197 > 0.03
    (0.150, 24, "unaffordable"),   # no registered n<=cap clears it
    (None, 4, "unmeasured"),
    (0.020, None, "unmeasured"),
]

CONTRAST_VECTORS = [
    (0.10, 0.20, "cp_better"),
    (-0.20, -0.10, "pdmux_better"),
    (-0.02, 0.02, "equivalent"),
    (-0.05, 0.10, "inconclusive"),
    (0.03, 0.20, "inconclusive"),      # ci_lo EXACTLY at +margin -- not beyond it
    (-0.20, -0.03, "inconclusive"),    # ci_hi EXACTLY at -margin
    (-0.03, 0.02, "inconclusive"),     # ci_lo exactly at -margin: not inside
    (0.031, 0.05, "cp_better"),
    (-0.029, 0.029, "equivalent"),
]

LADDER_VECTORS = [
    (["pdmux_better"] * 36, "consistent"),
    (["cp_better"] * 36, "consistent"),
    (["cp_better"] * 35 + ["pdmux_better"], "inconsistent"),   # ONE point flips
    (["cp_better"] * 18 + ["pdmux_better"] * 18, "inconsistent"),
    (["equivalent"] * 35 + ["unmeasured"], "unmeasured"),
]

# Exercises the DESIGN constant: the cap decides WHAT IS BOUGHT (which n are on the
# table), never how a bought thing is labelled.  Without these vectors
# BUDGET_MAX_PAIRS would pass its "moves no axis" check while being untested.
REQUIRED_N_VECTORS = [
    # (sd, cap, expected required_n)
    (0.010, 24, 3),      # hw(0.010, 2) = 0.0898 -> no;  n=3 -> 0.0248 -> yes
    (0.020, 24, 5),
    (0.060, 24, 18),     # ★not 20 -- n=18 clears it by 0.00016 (see POWER_VECTORS)
    (0.060, 10, None),   # same sd, smaller cap -> unanswerable
    (0.150, 60, None),   # no registered n clears it at all
    # ★cap=None means "use the module's BUDGET_MAX_PAIRS".  Without these two the
    # DESIGN constant is never read by any registered vector and its mutation test
    # passes vacuously -- the selftest caught exactly that and this is the repair.
    (0.020, None, 5),    # answerable well inside the default cap of 24
    (0.090, None, None), # NOT answerable at cap 24; a larger cap would reach n=40,
                         # which is what makes the cap load-bearing on what is bought
]

COVERAGE_VECTORS = [
    (4, 4, "complete"),
    (5, 4, "complete"),
    (3, 4, "partial"),
    (0, 4, "absent"),
    (19, 20, "partial"),
]
