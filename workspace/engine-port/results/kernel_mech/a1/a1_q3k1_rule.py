#!/usr/bin/env python3
"""A1 -- the ONLY new decision rules: Q3 (step-boundary separability) and K1.

Q1e/Q2ae/Q2be are scored by the A0 rule, unchanged: its World axes are
substrate-independent and it has already been through three rules-layer audits
with a registered hard stop.  Re-auditing it is out of scope.  What is new on
the engine substrate is (a) whether a graph-launch row separates decode steps
-- E1-(a), the thing that would remove the NVTX prerequisite -- and (b) what
the profiler costs.

Run:  python3 a1_q3k1_rule.py            # full self-test
"""
import hashlib, json, os, sys
from itertools import product

RULE_REV = 1

# --- Q3 labels ---------------------------------------------------------------
Q3_ABSENT   = "Q3_MEASUREMENT_ABSENT"
Q3_NOTEL    = "Q3_TELEMETRY_ABSENT"          # the independent channel is missing
Q3_THIN     = "SPLIT_WINDOW_INSUFFICIENT"    # design sec5: NOT "Q3 = no"
Q3_NOSEP    = "BOUNDARY_NOT_SEPARABLE"
Q3_PARTIAL  = "BOUNDARY_SEPARABLE_PARTIAL"
Q3_SEP      = "BOUNDARY_SEPARABLE"           # E1-(a) holds on the engine
Q3_LABELS = {Q3_ABSENT, Q3_NOTEL, Q3_THIN, Q3_NOSEP, Q3_PARTIAL, Q3_SEP}

# --- K1 labels ---------------------------------------------------------------
K1_UNMEAS = "K1_UNMEASURED"
K1_NEAR1  = "K1_NEAR_1"
K1_MOD    = "K1_MODERATE"
K1_HIGH   = "K1_HIGH"
K1_LABELS = {K1_UNMEAS, K1_NEAR1, K1_MOD, K1_HIGH}

# --- registered constants ----------------------------------------------------
N_MIN_SPLIT_STEPS = 200   # design sec5.  Below this the window, not the tool,
                          # is what failed.  Chosen from the reference trace:
                          # the split state is rare (130 of 59,646 snapshots),
                          # so a thin window is the EXPECTED failure, not a
                          # surprise, and must not be scorable as "Q3 = no".
K1_MOD_AT  = 1.10         # ratio of engine-telemetry median ITL, nsys ON / OFF
K1_HIGH_AT = 2.00         # rev8 registered this as the affordability stop


class Q3World:
    """One possible outcome of the Q3 measurement.

    steps      : decode steps the ENGINE reported inside split windows
    launches   : graph-launch RUNTIME rows found in those same windows
    partition  : do node rows partition cleanly by launch correlationId
    monotonic  : are launch host timestamps ordered and non-overlapping
    """

    def __init__(self, run_ok=True, export="ok", tel="ok", steps="ok",
                 launches="equal", partition="clean", monotonic="yes"):
        self.run_ok, self.export, self.tel = run_ok, export, tel
        self.steps = steps            # none | thin | ok
        self.launches = launches      # zero | fewer | equal | more
        self.partition = partition    # clean | orphans | overlap
        self.monotonic = monotonic    # yes | no

    def __repr__(self):
        return (f"Q3(run={self.run_ok},exp={self.export},tel={self.tel},"
                f"steps={self.steps},launch={self.launches},"
                f"part={self.partition},mono={self.monotonic})")


def q3_score(w, guards=frozenset()):
    def on(g):
        return g not in guards
    if on("g_measure") and (not w.run_ok or w.export == "fail"):
        return Q3_ABSENT
    # The engine channel is what makes this non-circular (design sec1).  With no
    # telemetry there is no independent statement of which windows were split,
    # so nothing here is scorable -- and it is NOT a fact about nsys.
    if on("g_tel") and w.tel != "ok":
        return Q3_NOTEL
    # design sec5: a thin window is a measurement condition.  This guard must
    # come BEFORE anything that reads the launch rows, or "few steps" turns
    # into "the boundary is not separable".
    if on("g_thin") and "g_order_thin_last" not in guards and w.steps != "ok":
        return Q3_THIN
    # ORDER MUTANT: the design's whole point is that the thin-window guard
    # comes FIRST.  With it last, a window with too few steps has too few
    # launch rows and scores BOUNDARY_NOT_SEPARABLE -- "the tool cannot
    # separate steps" -- which is a statement about nsys made out of a
    # workload shortfall.  A0's B3/N1 were this exact shape.
    if "g_order_thin_last" in guards and w.steps == "ok":
        pass
    if on("g_nosep") and (w.launches != "equal" or w.monotonic != "yes"
                          or w.partition == "overlap"):
        return Q3_NOSEP
    if on("g_partial") and w.partition == "orphans":
        return Q3_PARTIAL
    if "g_order_thin_last" in guards and w.steps != "ok":
        return Q3_THIN
    return Q3_SEP


Q3_AXES = dict(run_ok=[True, False], export=["ok", "partial", "fail"],
               tel=["ok", "absent"], steps=["none", "thin", "ok"],
               launches=["zero", "fewer", "equal", "more"],
               partition=["clean", "orphans", "overlap"],
               monotonic=["yes", "no"])
Q3_MUTANTS = ["g_measure", "g_tel", "g_thin", "g_nosep", "g_partial",
              "g_order_thin_last"]


def q3_worlds():
    keys = list(Q3_AXES)
    for c in product(*(Q3_AXES[k] for k in keys)):
        yield Q3World(**dict(zip(keys, c)))


class K1World:
    def __init__(self, on_ok=True, off_ok=True, ratio=1.0):
        self.on_ok, self.off_ok, self.ratio = on_ok, off_ok, ratio

    def __repr__(self):
        return f"K1(on={self.on_ok},off={self.off_ok},r={self.ratio})"


def k1_score(w, guards=frozenset()):
    def on(g):
        return g not in guards
    if on("h_measure") and not (w.on_ok and w.off_ok):
        return K1_UNMEAS
    hi = K1_HIGH_AT if on("h_hi_value") else 5.0
    mod = K1_MOD_AT if on("h_mod_value") else 1.5
    if on("h_hi") and w.ratio >= hi:
        return K1_HIGH
    if on("h_mod") and w.ratio >= mod:
        return K1_MOD
    return K1_NEAR1


K1_AXES = dict(on_ok=[True, False], off_ok=[True, False],
               # 1.09/1.10 and 1.99/2.00 straddle the two registered constants
               ratio=[0.95, 1.09, 1.10, 1.50, 1.99, 2.00, 12.0])
K1_MUTANTS = ["h_measure", "h_mod", "h_hi", "h_mod_value", "h_hi_value"]


def k1_worlds():
    keys = list(K1_AXES)
    for c in product(*(K1_AXES[k] for k in keys)):
        yield K1World(**dict(zip(keys, c)))


# ============================================================================
# SELF-TEST.  Same machinery the A0 rule converged on: every check must be
# broken by a named mutant (T8), every mutant must be caught by some check
# (T10), and no check may be entailed by another or bind nothing on its own
# (T19a/b) -- that last pair is what caught two hollow checks in A0.
# ============================================================================
def _q3_checks():
    """ONE check, not six.

    rev1 first wrote six per-label iff checks.  T19b then reported that FIVE of
    them had sole-binding 0 AND were not required for mutant coverage -- i.e.
    they were mutually redundant: on a total iff-partition of a six-label
    space, any mislabeled assignment violates several of them at once, so six
    checks carried exactly one check's worth of information.  Six checks that
    look thorough and are not is the shape this project keeps paying for, so
    they are collapsed here and the collapse is recorded rather than papered
    over.

    What keeps this from being an identity is that the partition is restated
    INDEPENDENTLY -- literal constants, different control flow from
    `q3_score()` -- and every mutant is proven to break it (T8/T10).
    """
    def c_partition(w, l):
        if not w.run_ok or w.export == "fail":
            want = Q3_ABSENT
        elif w.tel != "ok":
            want = Q3_NOTEL
        elif w.steps != "ok":                    # order: THIN before anything
            want = Q3_THIN                       # that reads the launch rows
        elif (w.launches != "equal" or w.monotonic != "yes"
              or w.partition == "overlap"):
            want = Q3_NOSEP
        elif w.partition == "orphans":
            want = Q3_PARTIAL
        else:
            want = Q3_SEP
        return l == want

    return {"Q3 label == the registered partition (independent restatement)":
            (c_partition, set(Q3_MUTANTS))}


def _k1_checks():
    def k_unm(w, l):
        return (l == K1_UNMEAS) == (not (w.on_ok and w.off_ok))

    def k_bucket(w, l):
        # HIGH and MODERATE share a boundary, so any threshold mutant breaks
        # both and neither binds alone (T19b caught it).  One check that
        # restates BOTH registered constants as literals.
        if not (w.on_ok and w.off_ok):
            return True
        want = (K1_HIGH if w.ratio >= 2.00 else
                K1_MOD if w.ratio >= 1.10 else K1_NEAR1)
        return l == want
    return {
        "K1-a UNMEASURED iff a leg is missing": (k_unm, {"h_measure"}),
        "K1-b buckets are exactly 1.10 / 2.00 (literals)":
            (k_bucket, {"h_mod", "h_hi", "h_mod_value", "h_hi_value"}),
    }


def _battery(name, worlds, score, mutants, checks, labels, out):
    W = list(worlds())
    base = [score(w) for w in W]
    fails = []

    def chk(n, c, d=""):
        print(f"  [{'PASS' if c else 'FAIL'}] {n} {d}")
        out[n] = bool(c)
        if not c:
            fails.append(n)

    chk(f"{name} T1 totality", all(l in labels for l in base))
    chk(f"{name} T1b determinism", base == [score(w) for w in W])
    from collections import Counter
    cnt = Counter(base)
    for lab in sorted(labels):
        chk(f"{name} T2 reachable: {lab}", cnt[lab] > 0, f"n={cnt[lab]}")
    mut = {m: [score(w, {m}) for w in W] for m in mutants}
    for m in mutants:
        d = sum(1 for a, b in zip(base, mut[m]) if a != b)
        chk(f"{name} T3 mutant {m} load-bearing", d > 0, f"n={d}")
    for n, (fn, _) in checks.items():
        chk(f"{name} {n} [DISCRIM]", all(fn(w, l) for w, l in zip(W, base)))
    print(f"  -- {name} T8 meta --")
    for n, (fn, ms) in checks.items():
        for m in sorted(ms):
            chk(f"{name} T8 '{n[:30]}' fails under {m}",
                any(not fn(w, l) for w, l in zip(W, mut[m])))
    print(f"  -- {name} T10 meta --")
    unc = [m for m in mutants
           if not any(any(not fn(w, l) for w, l in zip(W, mut[m]))
                      for fn, _ in checks.values())]
    chk(f"{name} T10 every mutant covered", not unc, f"uncovered={unc or 'none'}")
    print(f"  -- {name} T19 meta (entailment / sole-binding on reachable assignments) --")
    ASG = [(w, score(w, frozenset({m}) if m else frozenset()))
           for m in [None] + list(mutants) for w in W]
    fs = {n: {i for i, (w, l) in enumerate(ASG) if not fn(w, l)}
          for n, (fn, _) in checks.items()}
    ent = [f"{a}=>{b}" for a in checks for b in checks
           if a != b and fs[b] and fs[b] <= fs[a]]
    chk(f"{name} T19a no check entailed by another", not ent, f"{ent or 'none'}")
    # T19b -- "does this check earn its place".  A0 used SOLE BINDING for
    # this and it caught two hollow checks.  It does not transfer here: the Q3
    # checks are a TOTAL iff-partition of a six-label space, so any mislabeled
    # assignment violates at least two of them at once (the label it wrongly
    # got, and the one it should have got).  Sole binding is therefore
    # structurally unreachable, and demanding it would only push me to write
    # WEAKER (one-way) checks -- which is the defect it exists to prevent.
    #
    # So the criterion is a disjunction, and BOTH components are reported:
    #   (a) sole binding -- some assignment only this check rejects, or
    #   (b) deletion coverage -- removing it leaves some mutant uncovered.
    # A check failing both is dead weight.  ★The auditor should judge whether
    # (b) is an acceptable substitute here; the raw sole-binding result is
    # printed either way rather than hidden.
    sole = {n: len(fs[n] - set().union(*[fs[o] for o in checks if o != n]))
            for n in checks}
    def _covered(subset):
        return [m for m in mutants
                if not any(any(not checks[n][0](w, l) for w, l in zip(W, mut[m]))
                           for n in subset)]
    needed = {n: bool(_covered([o for o in checks if o != n])) for n in checks}
    dead = [n for n in checks if sole[n] == 0 and not needed[n]]
    print(f"     sole-binding: { {n[:14]: sole[n] for n in checks} }")
    print(f"     required-for-coverage: { {n[:14]: needed[n] for n in checks} }")
    chk(f"{name} T19b no check is dead weight (sole-binding OR needed for coverage)",
        not dead, f"{dead or 'none'}")
    return fails, len(W), dict(cnt)


def run():
    print(f"== A1 Q3/K1 rule (RULE_REV={RULE_REV}) ==")
    res = {}
    f1, n1, c1 = _battery("Q3", q3_worlds, q3_score, Q3_MUTANTS, _q3_checks(),
                          Q3_LABELS, res)
    f2, n2, c2 = _battery("K1", k1_worlds, k1_score, K1_MUTANTS, _k1_checks(),
                          K1_LABELS, res)
    ok = not (f1 or f2)
    print(f"\n== Q3 {n1:,} worlds / K1 {n2:,} worlds -> "
          f"{'ALL PASS' if ok else str(len(f1 + f2)) + ' FAILURES: ' + '; '.join((f1 + f2)[:4])} ==")
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.abspath(__file__), "rb") as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()
    with open(os.path.join(here, "selftest_a1_q3k1_2026-08-25.json"), "w") as fh:
        json.dump({"rule_rev": RULE_REV, "rule_sha256": sha, "date": "2026-08-25",
                   "gpu_hr": 0.0, "q3_worlds": n1, "k1_worlds": n2,
                   "q3_labels": c1, "k1_labels": c2, "checks": res,
                   "all_pass": ok}, fh, indent=1, sort_keys=True)
    print("   wrote selftest_a1_q3k1_2026-08-25.json")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(run())
