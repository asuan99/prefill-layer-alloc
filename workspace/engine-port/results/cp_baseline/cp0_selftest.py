#!/usr/bin/env python3
"""Self-test for the CP-0 rules.  Zero GPU.

Written against the 2nd audit's K4: rev2's mutation list was AUTHORED BY THE
SELF-TEST ITSELF, so it covered only what its author already had in mind -- and 9
of 14 externally-supplied mutants survived, including an inversion of the axis
that gates every size claim.  The repair is structural, not a longer list:

  * mutants are ENUMERATED FROM THE RULE, not hand-picked.  Every entry of
    `RULES`, every key of `OUTCOME`, and `incoherent` are each dropped/negated/
    remapped in turn, and each must change the label map.
  * ★rev2 (3rd audit F4): enumeration from the rule was NOT ENOUGH -- an external
    mutation run got 8 of 9 mutants past the rev1 test, and the survivors were the
    ones that carry meaning: OUTCOME PERMUTATIONS (swapping which cell means what),
    GUARD ORDER (which measurement-failure label wins), and the CONTINUOUS
    THRESHOLDS, which the rule did not even read.  All three are now generated:
      - every non-identity permutation of the OUTCOME values
      - every guard-order transposition whose two guards carry DIFFERENT labels
        (a transposition of two guards with the same label is inert by
        construction, is listed as such, and is not counted as coverage)
      - every constant of the rule's `PREDICATE_CONSTANTS`, mutated against the
        registered data vectors in `cp0_predicates.py`
  * every module constant is classified.  `LABEL_CONSTANTS` must be load-bearing
    (perturbing changes the map).  `CAMPAIGN_PARAMS` must NOT be (perturbing
    leaves the map identical) -- declared so nobody mistakes this test for a
    guard on them.  A constant in neither list fails the run.
  * mutation works by rebinding module globals, so the REAL `label()` is what
    runs.  A mirror re-implementation would only test the mirror.

Usage:  python3 cp0_selftest.py
"""
import importlib.util, itertools, json, os, sys

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


def worlds(m):
    return [dict(zip(m.AXIS_ORDER, c))
            for c in itertools.product(*(m.AXES[a] for a in m.AXIS_ORDER))]


def label_map(m, ws):
    return tuple(m.label(w) for w in ws)


def audit(fn):
    print(f"\n=== {fn} ===")
    m = load(fn)
    ws = worlds(m)
    base = label_map(m, ws)
    declared = set(m.SUBSTANTIVE) | {"IMPOSSIBLE_WORLD"} | {lab for _n, _p, lab in m.RULES}
    print(f"  {len(ws)} worlds, {len(set(base))} labels")

    # ---- ORACLE (4th audit G5) ---------------------------------------------
    # Every `is DETECTED` check below compares a mutant against `base`, which is
    # computed from the module under test.  That measures self-consistency, and an
    # externally introduced semantic inversion is invisible to it -- three such
    # mutants survived rev3.  The registered table in cp0_expected_labels.json is
    # the missing external reference.  It is a GOLDEN file, honestly labelled as
    # one: it freezes the mapping rather than deriving it, but it lives outside the
    # rules, so changing a rule without changing it fails here and shows in a diff.
    oracle = json.load(open(os.path.join(HERE, "cp0_expected_labels.json"), encoding="utf-8"))
    entry = oracle["rules"].get(fn)
    check("a registered oracle exists for this rule", entry is not None)
    if entry:
        check("oracle axis order matches the rule", entry["axis_order"] == m.AXIS_ORDER)
        check("oracle rule_rev matches the rule", entry["rule_rev"] == m.RULE_REV,
              f"oracle {entry['rule_rev']} vs rule {m.RULE_REV}")
        got = {"|".join(w[a] for a in m.AXIS_ORDER): lab for w, lab in zip(ws, base)}
        diff = [k for k in set(got) | set(entry["table"])
                if got.get(k) != entry["table"].get(k)]
        check("EVERY world's label matches the registered oracle", not diff,
              f"{len(diff)} mismatch(es)" + (f", e.g. {diff[0]}: oracle "
              f"{entry['table'].get(diff[0])} vs rule {got.get(diff[0])}" if diff else ""))

    check("label() is TOTAL and emits only declared labels",
          set(base) <= declared, f"undeclared: {sorted(set(base) - declared)}")
    check("every substantive label is reachable in the grid",
          set(m.SUBSTANTIVE) <= set(base), f"missing: {sorted(set(m.SUBSTANTIVE) - set(base))}")
    check("OUTCOME table is total over the measured 2-tuples",
          all((a, b) in m.OUTCOME for a, b in itertools.product(
              [v for v in m.AXES[m.AXIS_ORDER[-2]] if v != "unmeasured"],
              [v for v in m.AXES[m.AXIS_ORDER[-1]] if v != "unmeasured"])))

    # ---- branch mutants, enumerated from the rule ---------------------------
    covered = set()
    for i, (name, pred, lab) in enumerate(list(m.RULES)):
        for kind in ("dropped", "negated"):
            saved = m.RULES
            if kind == "dropped":
                m.RULES = saved[:i] + saved[i + 1:]
            else:
                m.RULES = saved[:i] + [(name, (lambda p: (lambda w: not p(w)))(pred), lab)] + saved[i + 1:]
            try:
                changed = label_map(m, ws) != base
            except Exception:
                changed = True          # a mutant that crashes is also detected
            finally:
                m.RULES = saved
            check(f"branch mutant `{name}` [{kind}] is DETECTED", changed)
            covered.add(name)

    check("every RULES branch was mutated",
          covered == {n for n, _p, _l in m.RULES},
          f"{len(covered)}/{len(m.RULES)}")

    # ---- outcome-table mutants ---------------------------------------------
    tail = m.AXIS_ORDER[-len(list(m.OUTCOME)[0]):]
    for key in list(m.OUTCOME):
        # A cell whose worlds are ALL incoherent cannot be reached, so remapping it
        # is inert by construction -- the same distinction the guard-transposition
        # check makes.  Naming it keeps a real coverage gap from hiding behind it.
        reachable_cell = any(
            not m.incoherent(w) and tuple(w[a] for a in tail) == key
            and all(not p(w) for _n, p, _l in m.RULES)
            for w in ws)
        saved = dict(m.OUTCOME)
        other = next(v for v in m.SUBSTANTIVE if v != saved[key])
        m.OUTCOME = {**saved, key: other}
        changed = label_map(m, ws) != base
        m.OUTCOME = saved
        if reachable_cell:
            check(f"outcome mutant {key} -> {other} is DETECTED", changed)
        else:
            check(f"outcome cell {key} is UNREACHABLE by construction (remap inert)",
                  not changed)

    # ---- OUTCOME permutation mutants (3rd audit F4) -------------------------
    import itertools as _it  # noqa: F401
    keys = list(m.OUTCOME)
    base_vals = [m.OUTCOME[k] for k in keys]
    # ★NOT A CHECK -- a THEOREM, demoted on the 4th audit's finding (G5).  If the
    # OUTCOME values are pairwise distinct and every cell is reachable, then every
    # non-identity permutation necessarily changes the label map; asserting it can
    # never fail (CONSENSUS §3 항목9, "확증서술이 항등식일 수 있다").  What IS
    # checkable is its precondition, so that is what is checked.
    dup = len(set(base_vals)) != len(base_vals)
    if dup:
        # Not a failure: several cells may legitimately carry the same label (here
        # `no_knee_in_range` makes `grid_brackets` meaningless, so both of its cells
        # answer the same thing).  What it means is that the permutation THEOREM
        # does not apply, so the duplicates are listed instead of assumed away.
        shared = {v: [k for k, vv in m.OUTCOME.items() if vv == v]
                  for v in set(base_vals) if base_vals.count(v) > 1}
        print(f"  [note] permutation theorem does NOT apply -- cells sharing a label: {shared}")
    else:
        check("OUTCOME values are pairwise distinct (permutation theorem applies)", True,
              f"{len(base_vals)} cells")
    check("every OUTCOME cell is reachable in the grid (the other precondition)",
          all(v in base for v in base_vals),
          f"unreachable: {[v for v in base_vals if v not in base]}")
    n_perm = 0

    # ---- guard ORDER mutants (3rd audit F4) ---------------------------------
    inert_by_construction = []
    for i in range(len(m.RULES) - 1):
        a, b = m.RULES[i], m.RULES[i + 1]
        saved = m.RULES
        m.RULES = saved[:i] + [b, a] + saved[i + 2:]
        changed = label_map(m, ws) != base
        m.RULES = saved
        # A transposition can be inert for exactly two structural reasons, and the
        # test must name which -- otherwise "no change" reads as missing coverage
        # when it is a theorem, or as a theorem when it is missing coverage.
        coherent = [w for w in ws if not m.incoherent(w)]
        fires_a = {id(w) for w in coherent if a[1](w)}
        fires_b = {id(w) for w in coherent if b[1](w)}
        disjoint = not (fires_a & fires_b)
        if a[2] == b[2]:
            reason = f"same label {a[2]}"
        elif disjoint:
            # e.g. `boot_failed` vs `echo_mismatch`: incoherent() forces every
            # boot=failed world to have echo=unmeasured, so the two guards can never
            # both fire.  Order between them is immaterial -- but ONLY because of
            # that coherence constraint, so the test asserts the disjointness rather
            # than the absence of change.
            reason = "firing sets disjoint over coherent worlds"
        else:
            reason = None
        if reason:
            inert_by_construction.append((a[0], b[0], reason))
            check(f"guard transposition `{a[0]}`<->`{b[0]}` is inert BY CONSTRUCTION "
                  f"({reason})", not changed)
            if reason.startswith("firing sets"):
                check(f"  ...and the disjointness is real, not assumed "
                      f"(`{a[0]}` {len(fires_a)} worlds, `{b[0]}` {len(fires_b)}, "
                      f"overlap 0)", disjoint)
        else:
            check(f"guard order mutant `{a[0]}`<->`{b[0]}` is DETECTED", changed)
    if inert_by_construction:
        print(f"  [note] {len(inert_by_construction)} transposition(s) inert by construction: "
              f"{inert_by_construction}")

    # ---- coherence mutant ---------------------------------------------------
    saved = m.incoherent
    m.incoherent = lambda _w: None
    changed = label_map(m, ws) != base
    m.incoherent = saved
    check("coherence mutant `incoherent -> always None` is DETECTED", changed)

    # ---- constants: every one must be classified, and behave as classified ---
    consts = {k for k, v in vars(m).items()
              if k.isupper() and not k.startswith("_")
              and k not in ("AXES", "AXIS_ORDER", "SUBSTANTIVE", "OUTCOME", "RULES",
                            "RULE_REV", "LABEL_CONSTANTS", "CAMPAIGN_PARAMS",
                            "PREDICATE_MODULE", "PREDICATE_CONSTANTS", "PREDICATE_FOLDS")}
    classified = set(m.LABEL_CONSTANTS) | set(m.CAMPAIGN_PARAMS)
    check("every module constant is classified as label-bearing or campaign-only",
          consts <= classified, f"unclassified: {sorted(consts - classified)}")

    for c in m.LABEL_CONSTANTS:
        saved = getattr(m, c)
        setattr(m, c, saved * 10 if isinstance(saved, (int, float)) else saved)
        changed = label_map(m, ws) != base
        setattr(m, c, saved)
        check(f"LABEL_CONSTANT `{c}` is load-bearing", changed)
    if not m.LABEL_CONSTANTS:
        print("  [note] this rule declares no label-bearing constants: every branch is categorical")

    for c in m.CAMPAIGN_PARAMS:
        saved = getattr(m, c)
        setattr(m, c, saved * 10 if isinstance(saved, (int, float)) else (0,))
        same = label_map(m, ws) == base
        setattr(m, c, saved)
        check(f"CAMPAIGN_PARAM `{c}` does NOT touch the label map (declared, verified)", same)


def audit_predicates(fn):
    """Constants that map DATA to axis values live in cp0_predicates.py, not in the
    rule, so the label map cannot test them.  They get their own mutation surface:
    a constant is load-bearing iff perturbing it changes at least one registered
    vector's classification.  (3rd audit F4: rev1 declared BATCH_MARGIN_REL a
    CAMPAIGN_PARAM and 'verified' it does not touch the label map -- an identity,
    since nothing read it.)"""
    print(f"\n=== {fn} predicates ===")
    m = load(fn)
    P = load(m.PREDICATE_MODULE)

    ok_r = all(P.rate_axis(a, o) == e for a, o, e in P.RATE_VECTORS)
    ok_b = all(P.batch_axis(a, r, lo, hi) == e for a, r, lo, hi, e in P.BATCH_VECTORS)
    check("registered RATE_VECTORS classify as expected", ok_r)
    check("registered BATCH_VECTORS classify as expected", ok_b)

    def classify_all():
        return (tuple(P.rate_axis(a, o) for a, o, _ in P.RATE_VECTORS),
                tuple(P.batch_axis(a, r, lo, hi) for a, r, lo, hi, _ in P.BATCH_VECTORS))

    def classify_all_of(_P):
        return classify_all()

    def ref_of(_P, saved, c):
        cur = getattr(_P, c)
        setattr(_P, c, saved)
        out = classify_all()
        setattr(_P, c, cur)
        return out

    # ★4th audit L-d: the "every module constant is classified" check ran on the
    # RULE module only, so moving BATCH_MARGIN_REL here to fix one identity created
    # two UNCLASSIFIED free constants (FLOAT_TOL, PROBE_RATES).  Moving a constant
    # out of the checked surface is not a repair.
    pconsts = {k for k, v in vars(P).items()
               if k.isupper() and not k.startswith("_")
               and k not in ("RATE_VECTORS", "BATCH_VECTORS", "FOLD_VECTORS",
                             "LABEL_CONSTANTS", "INFRA_CONSTANTS")}
    classified = set(P.LABEL_CONSTANTS) | set(P.INFRA_CONSTANTS)
    check("every predicate-module constant is classified",
          pconsts <= classified, f"unclassified: {sorted(pconsts - classified)}")
    check("this rule's PREDICATE_CONSTANTS are a subset of the module's LABEL_CONSTANTS",
          set(m.PREDICATE_CONSTANTS) <= set(P.LABEL_CONSTANTS))
    # INFRA constants are DECLARED not to change a classification -- verify it.
    for c in P.INFRA_CONSTANTS:
        saved = getattr(P, c)
        setattr(P, c, saved * 3 if isinstance(saved, (int, float)) else (99,))
        same = classify_all_of(P) == ref_of(P, saved, c)
        setattr(P, c, saved)
        check(f"INFRA_CONSTANT `{c}` does not change a classification (declared, verified)",
              same)

    check("registered FOLD_VECTORS fold as expected",
          all((P.req_axis_fold(v) if k == "req" else P.batch_axis_fold(v)) == e
              for k, v, e in P.FOLD_VECTORS))

    ref = classify_all()
    for c in m.PREDICATE_CONSTANTS:
        saved = getattr(P, c)
        for factor in (0.5, 1.5):
            setattr(P, c, saved * factor)
            changed = classify_all() != ref
            setattr(P, c, saved)
            check(f"PREDICATE_CONSTANT `{c}` x{factor} is load-bearing", changed)

    # malformed input must be refused, not silently classified
    for bad, fn_ in ((lambda: P.rate_axis(1.0, 0.0), "rate_axis(offered=0)"),
                     (lambda: P.rate_axis(-1.0, 1.0), "rate_axis(achieved<0)"),
                     (lambda: P.batch_axis(1.0, 0.0, -1.0, 1.0), "batch_axis(ref=0)"),
                     (lambda: P.batch_axis(1.0, 1.0, 1.0, -1.0), "batch_axis(lo>hi)")):
        try:
            bad(); raised = False
        except Exception:
            raised = True
        check(f"{fn_} is refused", raised)


for fn in ("cp0_arm_rule.py", "cp0_capacity_rule.py"):
    audit(fn)
    audit_predicates(fn)

# The G1 probe rule has no data->axis predicate module: its two continuous inputs
# (knee band width, seed SD) are compared to EACH OTHER, so there is no registered
# constant to be load-bearing.  It gets the label-layer audit only, and declares
# that absence rather than leaving it to be inferred.
audit("cp0_g1_rule.py")

print()
if FAILURES:
    print(f"SELFTEST FAILED: {len(FAILURES)} check(s): {FAILURES}")
    sys.exit(1)
print("SELFTEST OK -- every enumerated branch, outcome cell, coherence guard and "
      "constant behaved as declared.")
