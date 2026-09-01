#!/usr/bin/env python3
"""Self-test for `cp_rule.py` (RULE_REV=2).  Zero GPU.

Answers audit kill-shot D1's remediation clause literally:

    "`ci_shape` を (lo,hi,delta) の全域関数として固定し, (1) 実数平面
     {(lo,hi): lo<=hi} を6値が分割することを反例探索で証明, (2) 各分岐が
     発火する変異テスト, (3) `straddles` の定義を文字で確定,
     (4) 検定力検査を順位検査より前に置き, それが `pos_ge_delta` を
     飲み込まないことを証明"

and the D3 clause "register the null firing rate of the OR-shaped screen"
(CONSENSUS §3 item 38 / PROJECT_STATUS "methodology gate" #24).

A self-check that cannot fail on a broken rule is worthless -- CONSENSUS §3 item
53: "a check that validates a repair must FAIL on a mutant that undoes it".  So
every structural claim here is paired with a mutant, and the run aborts if any
mutant survives.

Usage:  python3 cp_rule_selftest.py
"""
import importlib.util, itertools, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("cp_rule", os.path.join(HERE, "cp_rule.py"))
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)

FAILURES = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


# ---------------------------------------------------------------------------
# Grid over (lo, hi).  Deliberately dense at the four boundaries the rule turns
# on (0, +/-delta) and at exact equality, because those are where an off-by-one
# in strictness hides.
# ---------------------------------------------------------------------------
DELTA = 0.0966                      # = 0.03 * 3.220, the canonical HE0 scale
def grid_points():
    marks = [-3, -1.5, -1.0000001, -1, -0.9999999, -0.5, -1e-9, 0, 1e-9,
             0.5, 0.9999999, 1, 1.0000001, 1.5, 3]
    vals = sorted({round(m * DELTA, 12) for m in marks} | {0.0})
    for lo in vals:
        for hi in vals:
            if lo <= hi:
                yield (lo, hi)


PTS = list(grid_points())
print(f"grid: {len(PTS)} (lo,hi) pairs with lo<=hi, delta={DELTA}")

# ---------------------------------------------------------------------------
# 1. TOTALITY + EXCLUSIVITY of ci_shape.
#    Structural argument: sign x magnitude, each total and exclusive.  Verified
#    here against an INDEPENDENT second implementation written as a flat decision
#    list, so an implementation slip in either one shows up as a disagreement.
# ---------------------------------------------------------------------------
AXIS = set(R.AXES["ci_shape"])


def ci_shape_independent(lo, hi, d):
    """Deliberately different shape: flat, ordered, no shared helper."""
    if lo >= d:
        return "pos_ge_delta"
    if hi <= -d:
        return "neg_ge_delta"
    if lo > 0 and hi < d:
        return "pos_subdelta"
    if hi < 0 and lo > -d:
        return "neg_subdelta"
    if lo > 0:
        return "pos_indet"
    if hi < 0:
        return "neg_indet"
    if lo > -d and hi < d:
        return "within_delta"
    return "straddles"


hit, total_ok, agree = set(), True, True
for lo, hi in PTS:
    try:
        v = R.ci_shape(lo, hi, DELTA)
    except Exception as e:                                  # noqa: BLE001
        total_ok, v = False, f"RAISED {e}"
    if v not in AXIS:
        total_ok = False
    else:
        hit.add(v)
    if v != ci_shape_independent(lo, hi, DELTA):
        agree = False

check("ci_shape is TOTAL on {lo<=hi} (returns an axis value, never raises)", total_ok)
check("ci_shape agrees with an independent implementation on every grid point", agree)
check("ci_shape is SURJECTIVE onto the axis (all 8 values attained)",
      hit == AXIS, f"attained {len(hit)}/8: {sorted(hit)}")
check("(sign=zero, magnitude=ge_delta) is unreachable, as the docstring claims",
      not any(R.ci_shape(lo, hi, DELTA) in ("pos_ge_delta", "neg_ge_delta")
              and lo <= 0 <= hi for lo, hi in PTS))
def _raises(fn):
    try:
        fn()
    except Exception:                                       # noqa: BLE001
        return True
    return False


check("ci_shape rejects a malformed CI (lo > hi)", _raises(lambda: R.ci_shape(1.0, 0.0, DELTA)))
check("ci_shape rejects delta <= 0", _raises(lambda: R.ci_shape(0.0, 1.0, 0.0)))

# ---------------------------------------------------------------------------
# 2. The cell rev1 had no value for at all.
# ---------------------------------------------------------------------------
he0 = R.ci_shape(0.068, 0.108, DELTA)
check("canonical HE0-scale CI [+0.068,+0.108] maps to a real axis value",
      he0 == "pos_indet",
      f"-> {he0}  (rev1: no value; sign settled, size not -- exactly the gap D1 named)")

# ---------------------------------------------------------------------------
# 3. MUTATION TESTS on ci_shape: every branch must be load-bearing.
# ---------------------------------------------------------------------------
def mutate_ci(kind):
    def f(lo, hi, d):
        if kind == "sign_pos_nonstrict":
            sign = "pos" if lo >= 0 else ("neg" if hi < 0 else "zero")
        elif kind == "sign_neg_nonstrict":
            sign = "pos" if lo > 0 else ("neg" if hi <= 0 else "zero")
        else:
            sign = "pos" if lo > 0 else ("neg" if hi < 0 else "zero")
        if kind == "ge_strict":
            ge = lo > d or hi < -d
        else:
            ge = lo >= d or hi <= -d
        if kind == "inside_nonstrict":
            ins = lo >= -d and hi <= d
        else:
            ins = lo > -d and hi < d
        mag = "ge_delta" if ge else ("subdelta" if ins else "indet")
        if kind == "drop_zero_case":
            return f"{sign}_{mag}"                      # leaks non-axis values
        if sign == "zero":
            return "within_delta" if mag == "subdelta" else "straddles"
        return f"{sign}_{mag}"
    return f


for kind in ("sign_pos_nonstrict", "sign_neg_nonstrict", "ge_strict",
             "inside_nonstrict", "drop_zero_case"):
    m = mutate_ci(kind)
    diff = [(lo, hi) for lo, hi in PTS if m(lo, hi, DELTA) != R.ci_shape(lo, hi, DELTA)]
    check(f"mutant `{kind}` is DETECTED", bool(diff),
          f"{len(diff)} point(s) differ, e.g. {diff[0] if diff else '-'}")

# ---------------------------------------------------------------------------
# 4. The label grid, and the world-coherence bookkeeping.
# ---------------------------------------------------------------------------
WORLDS = [dict(zip(R.AXIS_ORDER, c))
          for c in itertools.product(*(R.AXES[a] for a in R.AXIS_ORDER))]
labels = {}
for w in WORLDS:
    labels.setdefault(R.label(w), []).append(w)
print(f"\ngrid: {len(WORLDS)} worlds -> {len(labels)} labels")
for k in sorted(labels):
    mark = "  (substantive)" if k in R.SUBSTANTIVE else ""
    print(f"    {k:28s} {len(labels[k]):4d}{mark}")

check("every substantive label appears somewhere in the grid",
      set(R.SUBSTANTIVE) <= set(labels),
      f"missing: {sorted(set(R.SUBSTANTIVE) - set(labels)) or 'none'}")
check("no label is produced that is neither substantive nor a declared non-label",
      set(labels) <= set(R.SUBSTANTIVE) | {"IMPOSSIBLE_WORLD", "DEGENERATE_ARM", "UNDERPOWERED"},
      f"unexpected: {sorted(set(labels) - set(R.SUBSTANTIVE) - {'IMPOSSIBLE_WORLD','DEGENERATE_ARM','UNDERPOWERED'})}")
check("IMPOSSIBLE_WORLD is non-substantive and is actually used",
      "IMPOSSIBLE_WORLD" not in R.SUBSTANTIVE and "IMPOSSIBLE_WORLD" in labels)

# ---------------------------------------------------------------------------
# 5. MUTATION TESTS on label(): the two orderings the audit turned on.
# ---------------------------------------------------------------------------
def label_power_after_ladder(w):
    """rev1's ordering: ladder before power.  D3."""
    if R.incoherent(w):
        return "IMPOSSIBLE_WORLD"
    if w["req_split_n_ge_min"] == "no" and w["batch_budget"] == "slack":
        return "DEGENERATE_ARM"
    if w["ladder"] == "conflicting":
        return "LADDER_CONFLICT"
    if w["ci_shape"] == "straddles":
        return "UNDERPOWERED"
    return R.label(w)


def label_subdelta_folded(w):
    """L10 regression: sub-delta wins folded into plain wins."""
    out = R.label(w)
    return out.replace("_SUBDELTA", "_GE_DELTA")


def label_no_coherence(w):
    """Drop the impossible-world bookkeeping."""
    save = R.incoherent
    R.incoherent = lambda _w: None
    try:
        return R.label(w)
    finally:
        R.incoherent = save


for name, fn in (("power-check moved AFTER ladder (rev1 ordering)", label_power_after_ladder),
                 ("sub-delta wins folded into GE_DELTA (L10 regression)", label_subdelta_folded),
                 ("world-coherence check removed", label_no_coherence)):
    diff = [w for w in WORLDS if fn(w) != R.label(w)]
    check(f"mutant `{name}` is DETECTED", bool(diff), f"{len(diff)}/{len(WORLDS)} worlds differ")

# power-first must not swallow a decisive result
swallowed = [w for w in WORLDS
             if w["ci_shape"] in ("pos_ge_delta", "neg_ge_delta")
             and R.label(w) == "UNDERPOWERED"]
check("power-first does NOT swallow pos_ge_delta / neg_ge_delta", not swallowed,
      f"{len(swallowed)} swallowed")

# ---------------------------------------------------------------------------
# 6. ladder_shape: total, and the OR-screen's null firing rate.
# ---------------------------------------------------------------------------
CI_SET = [(-0.3, -0.1), (-0.2, 0.2), (0.1, 0.3), (-0.05, 0.05)]
lad_ok = True
for combo in itertools.product(CI_SET, repeat=3):
    if R.ladder_shape(list(combo), DELTA) not in set(R.AXES["ladder"]):
        lad_ok = False
check("ladder_shape is TOTAL over sampled CI tuples", lad_ok)
check("ladder_shape('conflicting') needs two OPPOSITE CIs that each exclude 0",
      R.ladder_shape([(0.1, 0.3), (-0.3, -0.1)], DELTA) == "conflicting"
      and R.ladder_shape([(0.1, 0.3), (-0.2, 0.2)], DELTA) == "concordant"
      and R.ladder_shape([(-0.2, 0.2), (-0.05, 0.05)], DELTA) == "uninformative")

rate = R.ladder_conflict_null_rate()
print(f"\nladder conflict screen, null firing rate (6 points, alpha=0.05, independence)"
      f" = {rate:.6f}  ({rate*100:.3f}%)")
print("  independence is the conservative direction: the 6 points re-score the SAME")
print("  reps, and positive correlation makes a sign conflict rarer, not commoner.")
print(f"  for contrast: a rev1-style point-estimate sign screen has no such bound at all.")
check("null firing rate of the ladder screen is registered and below alpha",
      rate < R.ALPHA, f"{rate:.4f} < {R.ALPHA}")

# ---------------------------------------------------------------------------
print()
if FAILURES:
    print(f"SELFTEST FAILED: {len(FAILURES)} check(s): {FAILURES}")
    sys.exit(1)
print("SELFTEST OK -- all checks passed and every mutant was detected.")
