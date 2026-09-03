#!/usr/bin/env python3
"""Self-test for the G1-b plateau predicate.  Zero GPU.

Lineage, named rather than left to be found: the structure is `cp0_selftest.py`'s
(registered vectors + constant classification + mutation), and the two repairs
this session made to `cp2_selftest.py` are carried here on purpose:

  * every registered-vector check is EXCEPTION-SAFE, so a predicate that refuses
    its own vectors is reported as a named failure instead of aborting the run;
  * the constants are classified into THREE classes, not two.  `cp0_predicates.py`
    had exactly two (label-bearing / infra) and that is why `PROBE_RATES` --
    which decides WHICH DATA IS BOUGHT but not how it is labelled -- had to be
    declared "infra" and verified with a check that could only ever pass.  A
    sweep constant is a real third thing and gets a real third check:
        LABEL  -> mutating it MUST change an axis classification
        DESIGN -> mutating it must change NO axis classification, though it may
                  change the stopping rule (that is what it is FOR)
        INFRA  -> mutating it must change nothing at all

★INTEGRATION is the point of this file, not a bonus: these predicates exist to
fill `cp0_g1_rule.py`'s axes, so the test asserts that every value they can emit
is a declared value of that rule's axes, and that the pair they jointly produce is
never one the rule declares IMPOSSIBLE.  A predicate that produced an incoherent
pair would have been invisible to a test that checked each axis alone.

Usage:  python3 g1b_plateau_selftest.py
"""
import importlib.util, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
FAILURES = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def load(fn):
    s = importlib.util.spec_from_file_location(fn[:-3], os.path.join(HERE, fn))
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def perturb(v):
    if isinstance(v, bool):
        return not v
    if isinstance(v, (int, float)):
        return v * 10
    if isinstance(v, dict):
        return {k: (x * 10 if isinstance(x, (int, float)) else x) for k, x in v.items()}
    if isinstance(v, tuple):
        return v[:-1] if len(v) > 1 else ()
    raise TypeError(f"no registered perturbation for {type(v)}")


P = load("g1b_plateau_predicates.py")
RULE = load("cp0_g1_rule.py")

print("=== g1b_plateau_predicates.py : registered vectors ===")


def vectors_ok(name, vectors, call):
    bad, raised = [], []
    for v in vectors:
        try:
            got = call(v)
        except Exception as e:
            raised.append((v, f"{type(e).__name__}: {e}"))
            continue
        if got != v[-1]:
            bad.append((v[:-1], "expected", v[-1], "got", got))
    detail = ""
    if raised:
        detail = f"{len(raised)} REFUSED its own registered vector, e.g. {raised[0]}"
    elif bad:
        detail = f"{len(bad)} misclassified, e.g. {bad[0]}"
    check(f"registered {name} classify as expected", not bad and not raised, detail)


def axis_outputs():
    """Every axis classification the registered vectors produce, as one tuple.
    A LABEL constant must move at least one of these; a DESIGN constant must move
    none of them.  Exceptions are captured as outcomes, not allowed to abort."""
    out = []
    for vs, call in (
        (P.CLIMBING_VECTORS, lambda v: P.climbing(v[0], v[1])),
        (P.KNEE_VECTORS, lambda v: P.knee(v[0], v[1])),
        (P.BAND_VECTORS, lambda v: P.band_vs_sd_axis(v[0], v[1], v[2])),
        (P.GRID_VECTORS, lambda v: P.grid_brackets_axis(v[0], v[1])),
        (P.COVERAGE_VECTORS, lambda v: P.coverage_axis(v[0], v[1], v[2])),
    ):
        for v in vs:
            try:
                out.append(call(v))
            except Exception as e:
                out.append(f"RAISED:{type(e).__name__}")
    return tuple(out)


vectors_ok("CLIMBING_VECTORS", P.CLIMBING_VECTORS, lambda v: P.climbing(v[0], v[1]))
vectors_ok("KNEE_VECTORS", P.KNEE_VECTORS, lambda v: P.knee(v[0], v[1]))
vectors_ok("BAND_VECTORS", P.BAND_VECTORS, lambda v: P.band_vs_sd_axis(v[0], v[1], v[2]))
vectors_ok("GRID_VECTORS", P.GRID_VECTORS, lambda v: P.grid_brackets_axis(v[0], v[1]))
vectors_ok("COVERAGE_VECTORS", P.COVERAGE_VECTORS,
           lambda v: P.coverage_axis(v[0], v[1], v[2]))
vectors_ok("NEXT_RATE_VECTORS", P.NEXT_RATE_VECTORS, lambda v: P.next_rate(v[0], v[1]))

print("\n=== integration with cp0_g1_rule.py (the axes these predicates fill) ===")
check("`band_vs_sd` outputs are declared values of the rule's axis",
      {v[-1] for v in P.BAND_VECTORS} <= set(RULE.AXES["band_vs_sd"]),
      f"rule declares {RULE.AXES['band_vs_sd']}")
check("`grid_brackets` outputs are declared values of the rule's axis",
      {v[-1] for v in P.GRID_VECTORS} <= set(RULE.AXES["grid_brackets"]))
check("`coverage` outputs are declared values of the rule's axis",
      {v[-1] for v in P.COVERAGE_VECTORS} <= set(RULE.AXES["coverage"]))
check("the rule's RULE_REV is unchanged by this registration (label map untouched)",
      RULE.RULE_REV == 1, f"RULE_REV={RULE.RULE_REV}")

# ★The pair check.  `cp0_g1_rule.incoherent` declares (no_knee_in_range, yes)
# impossible.  A predicate pair that produced it would be invisible to per-axis
# checks, so the two axes are run TOGETHER on the same datasets.
joint = []
for rates, achieved, attainment, _e in P.BAND_VECTORS:
    b = P.band_vs_sd_axis(rates, achieved, attainment)
    g = P.grid_brackets_axis(rates, achieved)
    w = {"coverage": "complete", "band_vs_sd": b, "grid_brackets": g}
    joint.append((b, g, RULE.incoherent(w), RULE.label(w)))
check("the two axes NEVER jointly produce a world the rule calls impossible",
      all(inc is None for _b, _g, inc, _l in joint),
      str([(b, g) for b, g, inc, _l in joint if inc]))
for b, g, _inc, lab in joint:
    print(f"  [note] ({b}, {g}) -> {lab}")

print("\n=== constants: three classes, each verified behaviourally ===")
consts = {k for k in vars(P)
          if k.isupper() and not k.startswith("_")
          and not k.endswith("_VECTORS")
          and k not in ("LABEL_CONSTANTS", "INFRA_CONSTANTS", "DESIGN_CONSTANTS")}
classified = set(P.LABEL_CONSTANTS) | set(P.INFRA_CONSTANTS) | set(P.DESIGN_CONSTANTS)
check("every module constant is classified", consts <= classified,
      f"unclassified: {sorted(consts - classified)}")
check("the three classes are disjoint",
      len(P.LABEL_CONSTANTS) + len(P.INFRA_CONSTANTS) + len(P.DESIGN_CONSTANTS)
      == len(classified))

ref_axes = axis_outputs()
ref_next = tuple(str(P.next_rate(v[0], v[1])) for v in P.NEXT_RATE_VECTORS)

for c in P.LABEL_CONSTANTS:
    saved = getattr(P, c)
    setattr(P, c, perturb(saved))
    changed = axis_outputs() != ref_axes
    setattr(P, c, saved)
    check(f"LABEL_CONSTANT `{c}` is load-bearing on an AXIS", changed)

for c in P.DESIGN_CONSTANTS:
    saved = getattr(P, c)
    setattr(P, c, perturb(saved))
    same_axes = axis_outputs() == ref_axes
    try:
        moved_sweep = tuple(str(P.next_rate(v[0], v[1]))
                            for v in P.NEXT_RATE_VECTORS) != ref_next
    except Exception:
        moved_sweep = True
    setattr(P, c, saved)
    check(f"DESIGN_CONSTANT `{c}` moves NO axis classification (declared, verified)",
          same_axes)
    # Not a failure if it does not move the sweep on these particular vectors --
    # but it is reported, because a design constant that moves nothing at all is
    # dead and this line is where that would show.
    print(f"  [note] DESIGN_CONSTANT `{c}` "
          f"{'does' if moved_sweep else 'does NOT'} move the stopping rule on the "
          f"registered NEXT_RATE_VECTORS")

for c in P.INFRA_CONSTANTS:
    saved = getattr(P, c)
    setattr(P, c, saved * 3 if isinstance(saved, (int, float)) else perturb(saved))
    same = axis_outputs() == ref_axes
    setattr(P, c, saved)
    check(f"INFRA_CONSTANT `{c}` changes nothing (declared, verified)", same)

print("\n=== malformed input must be REFUSED, not silently classified ===")
for bad, name in (
    (lambda: P.climbing([1.0, 2.0], [1.0]), "climbing(unpaired lengths)"),
    (lambda: P.climbing([1.0], [2.0]), "climbing(n=1, no SD)"),
    (lambda: P.t95(1), "t95(n=1)"),
    (lambda: P.t95(99), "t95(n outside the registered table)"),
    (lambda: P.knee((2.0,), {2.0: [1.0, 1.0]}), "knee(one rate)"),
    (lambda: P.knee((2.0, 4.0), {2.0: [1.0, 1.0]}), "knee(missing rate data)"),
    (lambda: P.grid_brackets_axis((2.0, 4.0), {2.0: [1.0] * 3, 4.0: [2.0] * 3},
                                  grid_lo=9.0, grid_hi=1.0), "grid_brackets(lo>=hi)"),
    (lambda: P.coverage_axis((2.0,), {2.0: [1.0]}, 1), "coverage(n_seeds=1)"),
):
    try:
        bad(); raised = False
    except Exception:
        raised = True
    check(f"{name} is refused", raised)

print()
if FAILURES:
    print(f"SELFTEST FAILED: {len(FAILURES)} check(s): {FAILURES}")
    sys.exit(1)
print("SELFTEST OK -- plateau predicate registered, mutation-tested, and coherent "
      "with cp0_g1_rule.py.")
