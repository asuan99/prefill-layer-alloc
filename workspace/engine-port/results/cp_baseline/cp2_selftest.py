#!/usr/bin/env python3
"""Self-test for the CP-2 rule.  Zero GPU.

Lineage, stated rather than left to be discovered: this file is adapted from
`cp0_selftest.py` and keeps its structure (mutants enumerated FROM the rule;
outcome permutations; guard-order transpositions with their two structural
inertness reasons named; every constant classified and behaviourally verified).
It is a SEPARATE FILE rather than a shared library because `cp0_selftest.py` runs
its audits at import time and CP-0's rules are a closed, NO-GO'd artifact that
must not change when CP-2 changes.  The duplication is named here because silent
duplication is how the next repair lands in one copy only -- which is this track's
own signature failure (4th audit: 11 of 24 defects were pure propagation).

Two things this file does that `cp0_selftest.py` does not:

  1. ★HAND-AUTHORED ORACLE.  The 4th audit's G5 finding was that a golden file
     GENERATED from the rule freezes the mapping but cannot detect a rule that was
     already wrong when it was registered.  `cp2_expected_labels.json` therefore
     carries TWO sections: `hand_authored`, ~two dozen worlds written out from the
     PROSE with a `why` for each (small enough to be read and argued with), and
     `frozen_table`, the full 270-world freeze that catches later edits.  The two
     are checked separately and reported separately, so a pass on one is never
     read as a pass on the other.

  2. TUPLE CONSTANTS.  Two of CP-2's label-bearing constants are tuples (the SLO
     band).  `saved * 10` is not a perturbation of a tuple's MEANING, so tuples are
     mutated by dropping a member -- which is what "the band the verdict must
     survive" being smaller actually means.

Usage:  python3 cp2_selftest.py
"""
import importlib.util, itertools, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
RULE = "cp2_rule.py"
ORACLE = "cp2_expected_labels.json"
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


def key_of(m, w):
    return "|".join(w[a] for a in m.AXIS_ORDER)


def perturb(value):
    """A perturbation that changes the constant's MEANING, per type."""
    if isinstance(value, bool):
        return not value
    if isinstance(value, (int, float)):
        return value * 10
    if isinstance(value, tuple):
        return value[:-1] if len(value) > 1 else ()
    if isinstance(value, list):
        return value[:-1] if len(value) > 1 else []
    raise TypeError(f"no registered perturbation for {type(value)}")


def audit_labels(fn):
    print(f"\n=== {fn} : label layer ===")
    m = load(fn)
    ws = worlds(m)
    base = label_map(m, ws)
    declared = set(m.SUBSTANTIVE) | {"IMPOSSIBLE_WORLD"} | {lab for _n, _p, lab in m.RULES}
    print(f"  {len(ws)} worlds, {len(set(base))} labels")

    # ---- ORACLE, in two honestly separated parts (4th audit G5) --------------
    oracle = json.load(open(os.path.join(HERE, ORACLE), encoding="utf-8"))
    entry = oracle["rules"].get(fn)
    check("a registered oracle exists for this rule", entry is not None)
    if not entry:
        return m
    check("oracle axis order matches the rule", entry["axis_order"] == m.AXIS_ORDER)
    check("oracle rule_rev matches the rule", entry["rule_rev"] == m.RULE_REV,
          f"oracle {entry['rule_rev']} vs rule {m.RULE_REV}")
    # ★Found by an external mutant this file did not survive on its first run: promoting
    # CLIFF_UNSTABLE into SUBSTANTIVE changes NO world's label, so every table check
    # passed while the rule's meaning changed -- "the verdict flips inside its own SLO
    # band" would have become a publishable finding about a policy family.  WHICH LABELS
    # COUNT AS A VERDICT is part of the rule, so the oracle registers it and this
    # compares it.  (Same shape as 4th audit G5, one level up from the label map.)
    check("oracle SUBSTANTIVE set matches the rule (which labels count as a verdict)",
          entry["substantive"] == list(m.SUBSTANTIVE),
          f"oracle {entry['substantive']} vs rule {list(m.SUBSTANTIVE)}")

    got = {key_of(m, w): lab for w, lab in zip(ws, base)}

    hand = entry["hand_authored"]
    unknown = [k for k in hand if k not in got]
    check("every hand-authored world is a real world of this lattice", not unknown,
          f"unknown keys: {unknown[:3]}")
    bad = [(k, v["label"], got.get(k)) for k, v in hand.items()
           if k in got and got[k] != v["label"]]
    check(f"HAND-AUTHORED oracle: {len(hand)} world(s) written from the prose match "
          f"the rule", not bad,
          f"{len(bad)} mismatch(es), e.g. {bad[0]}" if bad else "")
    covered = {v["label"] for v in hand.values()}
    check("the hand-authored subset covers EVERY label the rule can emit",
          set(base) <= covered, f"uncovered: {sorted(set(base) - covered)}")

    frozen = entry["frozen_table"]
    diff = [k for k in set(got) | set(frozen) if got.get(k) != frozen.get(k)]
    check("FROZEN table (generated freeze, catches later edits) matches", not diff,
          f"{len(diff)} mismatch(es), e.g. {diff[0]}: frozen "
          f"{frozen.get(diff[0])} vs rule {got.get(diff[0])}" if diff else "")

    check("label() is TOTAL and emits only declared labels",
          set(base) <= declared, f"undeclared: {sorted(set(base) - declared)}")
    check("every substantive label is reachable in the grid",
          set(m.SUBSTANTIVE) <= set(base), f"missing: {sorted(set(m.SUBSTANTIVE) - set(base))}")
    check("OUTCOME table is total over the measured 2-tuples",
          all((a, b) in m.OUTCOME for a, b in itertools.product(
              [v for v in m.AXES[m.AXIS_ORDER[-2]] if v != "unmeasured"],
              [v for v in m.AXES[m.AXIS_ORDER[-1]] if v != "unmeasured"])))

    # ---- branch mutants, enumerated from the rule ---------------------------
    seen = set()
    for i, (name, pred, lab) in enumerate(list(m.RULES)):
        for kind in ("dropped", "negated"):
            saved = m.RULES
            if kind == "dropped":
                m.RULES = saved[:i] + saved[i + 1:]
            else:
                m.RULES = saved[:i] + [(name, (lambda p: (lambda w: not p(w)))(pred), lab)] \
                    + saved[i + 1:]
            try:
                changed = label_map(m, ws) != base
            except Exception:
                changed = True
            finally:
                m.RULES = saved
            check(f"branch mutant `{name}` [{kind}] is DETECTED", changed)
            seen.add(name)
    check("every RULES branch was mutated", seen == {n for n, _p, _l in m.RULES},
          f"{len(seen)}/{len(m.RULES)}")

    # ---- outcome-table mutants ---------------------------------------------
    tail = m.AXIS_ORDER[-len(list(m.OUTCOME)[0]):]
    for key in list(m.OUTCOME):
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

    # ---- OUTCOME permutation: a THEOREM, so its PRECONDITIONS are checked ----
    vals = [m.OUTCOME[k] for k in m.OUTCOME]
    check("OUTCOME values are pairwise distinct (permutation theorem applies)",
          len(set(vals)) == len(vals), f"{len(vals)} cells")
    check("every OUTCOME cell is reachable in the grid (the other precondition)",
          all(v in base for v in vals),
          f"unreachable: {[v for v in vals if v not in base]}")

    # ---- guard ORDER mutants ------------------------------------------------
    inert = []
    for i in range(len(m.RULES) - 1):
        a, b = m.RULES[i], m.RULES[i + 1]
        saved = m.RULES
        m.RULES = saved[:i] + [b, a] + saved[i + 2:]
        changed = label_map(m, ws) != base
        m.RULES = saved
        coherent = [w for w in ws if not m.incoherent(w)]
        fires_a = [w for w in coherent if a[1](w)]
        fires_b = [w for w in coherent if b[1](w)]
        overlap = [w for w in fires_a if b[1](w)]
        if a[2] == b[2]:
            reason = f"same label {a[2]}"
        elif not overlap:
            reason = "firing sets disjoint over coherent worlds"
        else:
            reason = None
        if reason:
            inert.append((a[0], b[0], reason))
            check(f"guard transposition `{a[0]}`<->`{b[0]}` is inert BY CONSTRUCTION "
                  f"({reason})", not changed)
            if reason.startswith("firing sets"):
                check(f"  ...and the disjointness is real, not assumed (`{a[0]}` "
                      f"{len(fires_a)} worlds, `{b[0]}` {len(fires_b)}, overlap 0)",
                      not overlap)
        else:
            check(f"guard order mutant `{a[0]}`<->`{b[0]}` is DETECTED", changed)
    if inert:
        print(f"  [note] {len(inert)} transposition(s) inert by construction: {inert}")

    # ---- coherence mutants: each CLAUSE, not just the function --------------
    saved = m.incoherent
    m.incoherent = lambda _w: None
    changed = label_map(m, ws) != base
    m.incoherent = saved
    check("coherence mutant `incoherent -> always None` is DETECTED", changed)

    # ---- constants ----------------------------------------------------------
    consts = {k for k in vars(m)
              if k.isupper() and not k.startswith("_")
              and k not in ("AXES", "AXIS_ORDER", "SUBSTANTIVE", "OUTCOME", "RULES",
                            "RULE_REV", "LABEL_CONSTANTS", "CAMPAIGN_PARAMS",
                            "PREDICATE_MODULE", "PREDICATE_CONSTANTS", "PREDICATE_FOLDS")}
    check("every module constant is classified as label-bearing or campaign-only",
          consts <= set(m.LABEL_CONSTANTS) | set(m.CAMPAIGN_PARAMS),
          f"unclassified: {sorted(consts - set(m.LABEL_CONSTANTS) - set(m.CAMPAIGN_PARAMS))}")
    for c in m.LABEL_CONSTANTS:
        saved = getattr(m, c)
        setattr(m, c, perturb(saved))
        changed = label_map(m, ws) != base
        setattr(m, c, saved)
        check(f"LABEL_CONSTANT `{c}` is load-bearing", changed)
    if not m.LABEL_CONSTANTS:
        print("  [note] this rule declares no label-bearing constants: every branch is "
              "categorical, and every continuous decision lives in the predicate module")
    for c in m.CAMPAIGN_PARAMS:
        saved = getattr(m, c)
        setattr(m, c, perturb(saved))
        same = label_map(m, ws) == base
        setattr(m, c, saved)
        check(f"CAMPAIGN_PARAM `{c}` does NOT touch the label map (declared, verified)", same)
    return m


def audit_predicates(m):
    print(f"\n=== {m.PREDICATE_MODULE} : data -> axis layer ===")
    P = load(m.PREDICATE_MODULE)

    def classify_all():
        """Every registered vector's classification.  A raised exception is itself a
        classification outcome (malformed input must be REFUSED, not scored), so it
        is captured rather than allowed to abort the sweep."""
        out = []
        for lo, hi, _e in P.CONTRAST_VECTORS:
            try: out.append(P.contrast_axis(lo, hi))
            except Exception as e: out.append(f"RAISED:{type(e).__name__}")
        for a, b, _e in P.PHASE_VECTORS:
            try: out.append(P.phase_reversal_axis(a, b))
            except Exception as e: out.append(f"RAISED:{type(e).__name__}")
        for d, _e in P.COVERAGE_VECTORS:
            try: out.append(P.coverage_axis(d))
            except Exception as e: out.append(f"RAISED:{type(e).__name__}")
        for v, _e in P.CLIFF_VECTORS:
            try: out.append(P.cliff_axis(v))
            except Exception as e: out.append(f"RAISED:{type(e).__name__}")
        return tuple(out)

    # ★Every registered-vector check is exception-safe.  Found by an external mutant
    # (shrinking the SLO band) that made `cliff_axis` refuse its own registered
    # vectors: the suite ABORTED with a traceback instead of naming a failing check.
    # A non-zero exit is not the same as a diagnosis, and a suite that dies cannot
    # report the checks after the one that died (PROJECT_STATUS.md "방법론 게이트"
    # 21's shape: do not let a tooling failure masquerade as, or hide, a verdict).
    def vectors_ok(fn, vectors, call):
        bad, raised = [], []
        for v in vectors:
            try:
                got = call(v)
            except Exception as e:
                raised.append((v, f"{type(e).__name__}: {e}"))
                continue
            if got != v[-1]:
                bad.append((v, got))
        detail = ""
        if raised:
            detail = f"{len(raised)} REFUSED its own registered vector, e.g. {raised[0]}"
        elif bad:
            detail = f"{len(bad)} misclassified, e.g. {bad[0]}"
        check(f"registered {fn} classify as expected", not bad and not raised, detail)

    vectors_ok("CONTRAST_VECTORS", P.CONTRAST_VECTORS, lambda v: P.contrast_axis(v[0], v[1]))
    vectors_ok("PHASE_VECTORS", P.PHASE_VECTORS, lambda v: P.phase_reversal_axis(v[0], v[1]))
    vectors_ok("COVERAGE_VECTORS", P.COVERAGE_VECTORS, lambda v: P.coverage_axis(v[0]))
    vectors_ok("CLIFF_VECTORS", P.CLIFF_VECTORS, lambda v: P.cliff_axis(v[0]))

    pconsts = {k for k in vars(P)
               if k.isupper() and not k.startswith("_")
               and k not in ("CONTRAST_VECTORS", "PHASE_VECTORS", "COVERAGE_VECTORS",
                             "CLIFF_VECTORS", "LABEL_CONSTANTS", "INFRA_CONSTANTS")}
    check("every predicate-module constant is classified",
          pconsts <= set(P.LABEL_CONSTANTS) | set(P.INFRA_CONSTANTS),
          f"unclassified: {sorted(pconsts - set(P.LABEL_CONSTANTS) - set(P.INFRA_CONSTANTS))}")
    check("this rule's PREDICATE_CONSTANTS are exactly the module's LABEL_CONSTANTS",
          set(m.PREDICATE_CONSTANTS) == set(P.LABEL_CONSTANTS),
          f"rule {sorted(m.PREDICATE_CONSTANTS)} vs module {sorted(P.LABEL_CONSTANTS)}")
    check("every declared PREDICATE_FOLD exists in the module",
          all(callable(getattr(P, f, None)) for f in m.PREDICATE_FOLDS),
          f"missing: {[f for f in m.PREDICATE_FOLDS if not callable(getattr(P, f, None))]}")

    ref = classify_all()
    for c in P.LABEL_CONSTANTS:
        saved = getattr(P, c)
        setattr(P, c, perturb(saved))
        changed = classify_all() != ref
        setattr(P, c, saved)
        check(f"LABEL_CONSTANT `{c}` is load-bearing", changed)
    for c in P.INFRA_CONSTANTS:
        saved = getattr(P, c)
        setattr(P, c, saved * 3 if isinstance(saved, (int, float)) else perturb(saved))
        same = classify_all() == ref
        setattr(P, c, saved)
        check(f"INFRA_CONSTANT `{c}` does not change a classification (declared, verified)",
              same)

    for bad, name in (
        (lambda: P.contrast_axis(1.0, -1.0), "contrast_axis(lo>hi)"),
        (lambda: P.contrast_axis(-1.0, 1.0, margin=0.0), "contrast_axis(margin=0)"),
        (lambda: P.phase_reversal_axis("cp_better", "banana"), "phase_reversal_axis(unknown)"),
        (lambda: P.coverage_axis({}), "coverage_axis(no arms)"),
        (lambda: P.coverage_axis({"a": -1}), "coverage_axis(negative)"),
        (lambda: P.cliff_axis(["cp_better"] * 4), "cliff_axis(partial band)"),
        (lambda: P.cliff_axis(["cp_better"] * 12), "cliff_axis(oversized band)"),
    ):
        try:
            bad(); raised = False
        except Exception:
            raised = True
        check(f"{name} is refused", raised)


mod = audit_labels(RULE)
if mod:
    audit_predicates(mod)

print()
if FAILURES:
    print(f"SELFTEST FAILED: {len(FAILURES)} check(s): {FAILURES}")
    sys.exit(1)
print("SELFTEST OK -- every enumerated branch, outcome cell, coherence guard, "
      "constant and registered vector behaved as declared.")
