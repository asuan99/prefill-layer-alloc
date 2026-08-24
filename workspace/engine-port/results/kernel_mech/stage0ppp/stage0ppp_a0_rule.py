#!/usr/bin/env python3
"""Stage 0''' A0 -- THE REGISTERED DECISION RULE, fixed in code (gate #66).

RULE_REV = 2.  rev1 was audited NO-GO (audit_stage0ppp_a0_2026-08-24/VERDICT.md,
blockers X1-X13, no fatal).  This revision closes the five the audit named as
prerequisites to buying: X1 (thresholds absent from the rule), X3 (a world with
no green context scoring the top positive label), X4 (the companion reproducing
the parent audit's E2), X5 (partial/truncated traces leaking into substantive
labels), X13 (an asymmetric capture failure being flattened), plus X2's real
content (checks must cover every mutant, and the rev1 "identity" diagnosis was
itself wrong).

WHAT THIS FILE IS NOT: not a measurement, not a harness, not evidence.  Scoring
a world here says nothing about the substrate.  It never touches a GPU.

Run:  python3 stage0ppp_a0_rule.py            # full self-test
"""
from itertools import product

RULE_REV = 2

# --- registered labels -------------------------------------------------------
# Measurement conditions (gate #21: a measurement failure is NOT a verdict).
ABSENT   = "MEASUREMENT_ABSENT"            # nsys run or sqlite export failed
INVALID  = "PROBE_INVALID"                 # a CONTROL failed -> adjudicate nothing
TRUNC    = "TRACE_TRUNCATED"               # X5: the trace lost rows; not a Q1 answer
GREENBAD = "GREEN_PARTITION_MISMATCH"      # X3: realized partition != target
CAPGREEN = "CAPTURE_FAILS_ONLY_UNDER_GREEN"  # X13: cheap substantive fact
# Tool-limit facts (green-context-independent).
NONODE   = "NODE_TRACE_UNAVAILABLE"        # L2 has no node rows at all
NOGREEN  = "GREENCTX_INVISIBLE"            # green-ctx kernels absent from the trace
# Substantive outcomes.
UNCONSTR = "PRIMARY_ESTIMAND_UNCONSTRUCTIBLE"   # NOT "track closed" (parent E1)
EXCESS   = "NODE_ROWS_EXCEED_EXPECTATION"  # X5/axis-2: more rows than L2 predicts
NOATTR   = "KSET_NOT_ATTRIBUTABLE"
CONTRA   = "CONTRADICTORY_ATTRIBUTION"     # ctx says green, stream says not
CAPJOIN  = "KSET_JOINS_CAPTURE_NOT_REPLAY" # parent E5: field present, unusable
TIMEATTR = "KSET_NEEDS_TIME_ATTRIBUTION"   # parent E1-(d) path
PARTIAL  = "KSET_CONSTRUCTIBLE_PARTIAL"    # join rate below the registered floor
OK       = "KSET_CONSTRUCTIBLE"            # scoped; basis is carried, see below

LABELS = {ABSENT, INVALID, TRUNC, GREENBAD, CAPGREEN, NONODE, NOGREEN,
          UNCONSTR, EXCESS, NOATTR, CONTRA, CAPJOIN, TIMEATTR, PARTIAL, OK}
# Attribution BASIS is reported beside the label and never collapsed into it.
BASES = {"ctx+stream", "stream_only", None}

# --- registered constants (changing one is a protocol amendment) -------------
Q1_FRAC   = 0.90   # node rows observed / node rows EXPECTED (expectation from L2)
JOIN_HIGH = 0.95   # correlationId -> replay cudaGraphLaunch join success floor

# X5: the observed node-row shape across the 20 replays.  rev1 collapsed this to
# a scalar, so "every replay uniformly short" (which the L2-derived expectation
# cancels, gate/lesson #20) and "the tail of the trace was lost" (which aims
# straight at L4, the LAST leg) were the same world.
PROFILE_FRAC = {"full": 1.0, "uniform_short": 0.80, "tail_missing": 0.75,
                "excess": 1.30, "empty": 0.0}
PROFILE_UNIFORM = {"full", "uniform_short", "excess", "empty"}


class World:
    """One possible outcome of the legs + tool state.

    L1  full-GPU eager      control: does nsys see ANY kernel at all
    L2  full-GPU graph      control: node rows exist AND define the expectation
    L3  green-ctx eager     control: are green-ctx kernels visible at all
    L4  green-ctx graph     THE CONDITION UNDER TEST
    L2' full-GPU graph, AFTER L4  X5 control: did the trace survive to the end
    """

    def __init__(self, run_ok=True, export="ok", l1="ok", l2="ok", l3="ok",
                 l2_post="ok", capture="ok", green="matched", profile="full",
                 ctx="distinct", stream="match", join_target="replay",
                 join_rate=1.0):
        self.run_ok = run_ok
        self.export = export            # ok | partial | fail
        self.l1, self.l2, self.l3 = l1, l2, l3          # ok | zero
        self.l2_post = l2_post          # ok | zero   (X5 trailing control)
        self.capture = capture          # ok | fail_green_only | fail_all
        self.green = green              # matched | mismatched | absent
        self.profile = profile          # see PROFILE_FRAC
        self.ctx = ctx                  # null | zero | parent | distinct
        self.stream = stream            # match | mismatch
        self.join_target = join_target  # none | capture_only | replay
        self.join_rate = join_rate      # 0.0 | 0.90 | 0.95 | 1.0

    @property
    def frac(self):
        return PROFILE_FRAC[self.profile]

    def __repr__(self):
        return (f"W(export={self.export},l1={self.l1},l2={self.l2},l3={self.l3},"
                f"l2p={self.l2_post},cap={self.capture},green={self.green},"
                f"prof={self.profile},ctx={self.ctx},str={self.stream},"
                f"jt={self.join_target},jr={self.join_rate})")


def score(w, guards=frozenset()):
    """The registered rule.  Returns (label, basis).

    `guards` names guards to DISABLE or WEAKEN, for mutation testing.  Per
    lesson #70 the suite includes PARAMETRIC weakenings; per audit X1 it also
    includes VALUE mutants on the two registered thresholds, because rev1's
    suite passed unchanged with Q1_FRAC at 0.60, 0.70 or 0.99.
    """
    def on(g):
        return g not in guards

    # -- 1. measurement conditions come FIRST (gate #21) ----------------------
    if on("g_measurement") and (not w.run_ok or w.export == "fail"):
        return ABSENT, None
    # X13: "L2 captures but the GREEN capture fails" is a cheap substantive
    # fact.  rev1 flattened it into PROBE_INVALID because the capture branch sat
    # ahead of the control branches.  Keeping capture failure OUT of the
    # substantive labels is right; erasing which capture failed was not.
    if on("g_capture"):
        if w.capture == "fail_all":
            return INVALID, None
        if w.capture == "fail_green_only":
            return (CAPGREEN, None) if (on("g_capgreen") and w.l2 == "ok") \
                   else (INVALID, None)

    # -- 2. controls (parent E2) ---------------------------------------------
    if on("g_l1") and w.l1 == "zero":
        return INVALID, None            # nsys attached to nothing / wrong process
    if on("g_l2") and w.l2 == "zero":
        return NONODE, None             # node granularity gone -- green-independent
    if on("g_l3") and w.l3 == "zero":
        return NOGREEN, None            # P1-shaped fact; green kernels not traced

    # X3: the green context must be SHOWN to exist, not assumed.  This repo has
    # already been burnt by trusting a target instead of the realized value
    # (Stage 0 de-confound: the "D108" anchor was really 16 SM).  Without this
    # branch, a silent fallback to the default stream scores the top positive
    # label, because stream-based membership always succeeds.
    if on("g_green"):
        if w.green == "absent":
            return INVALID, None
        if w.green == "mismatched":
            return GREENBAD, None

    # X5: the trace must be shown to have survived to the end, and a partial
    # export is not a fact about the driver.
    if on("g_trunc"):
        if w.export == "partial":
            return TRUNC, None
        if w.l2_post == "zero":
            return TRUNC, None          # the trailing control died -> rows lost
        if w.profile not in PROFILE_UNIFORM:
            return TRUNC, None          # non-uniform loss = truncation, not Q1

    # -- 3. Q1, against an EXPECTATION MEASURED IN L2 ------------------------
    if on("g_excess") and w.frac > 1.0:
        return EXCESS, None             # more rows than L2 predicts -> not a pass
    floor = Q1_FRAC
    if not on("g_q1_floor"):
        floor = 0.50                    # parametric weakening
    if not on("g_q1_value"):
        floor = 0.70                    # X1: VALUE mutant on the threshold
    if on("g_q1") and w.frac < floor:
        return UNCONSTR, None           # parent E1: (b)(c)(d) survive

    # -- 4. attribution -------------------------------------------------------
    ctx_ok = w.ctx == "distinct" if on("g_ctx") else w.ctx in ("distinct", "parent")
    strm_ok = w.stream == "match"
    if on("g_contra") and ctx_ok and not strm_ok:
        return CONTRA, None             # two channels disagree -> adjudicate nothing
    if not ctx_ok and not strm_ok:
        return NOATTR, None
    basis = "ctx+stream" if (ctx_ok and strm_ok) else "stream_only"
    if not on("g_basis"):
        basis = "ctx+stream"            # mutant: the report drops the basis

    # -- 5. join: can a row be tied to THE REPLAY that produced it? ----------
    if on("g_capjoin") and w.join_target == "capture_only":
        return CAPJOIN, basis           # parent E5
    if on("g_join_none") and w.join_target == "none":
        return TIMEATTR, basis          # parent E1-(d)
    jfloor = JOIN_HIGH if on("g_join_value") else 0.10   # X1: VALUE mutant
    if w.join_rate < jfloor:
        return PARTIAL, basis
    return OK, basis


AXES = dict(
    run_ok=[True, False], export=["ok", "partial", "fail"],
    l1=["ok", "zero"], l2=["ok", "zero"], l3=["ok", "zero"],
    l2_post=["ok", "zero"],
    capture=["ok", "fail_green_only", "fail_all"],
    green=["matched", "mismatched", "absent"],
    profile=list(PROFILE_FRAC),
    ctx=["null", "zero", "parent", "distinct"],
    stream=["match", "mismatch"],
    join_target=["none", "capture_only", "replay"],
    # 0.94/0.95 straddle the registered floor so an off-by-one in the
    # comparison changes a world's label.
    join_rate=[0.0, 0.94, 0.95, 1.0],
)


def worlds():
    keys = list(AXES)
    for combo in product(*(AXES[k] for k in keys)):
        yield World(**dict(zip(keys, combo)))


MUTANTS = ["g_measurement", "g_capture", "g_capgreen", "g_l1", "g_l2", "g_l3",
           "g_green", "g_trunc", "g_excess", "g_q1", "g_q1_floor", "g_q1_value",
           "g_ctx", "g_contra", "g_capjoin", "g_join_none", "g_join_value",
           "g_basis"]


# ============================================================================
# COMPANION -- the parent E1-(b) fallback, bought in the same job.
#
# `nsys profile --help` (2025.3.2, read 2026-08-24, GPU 0): "If 'node' is
# selected, node activities will be collected, but CUDA graphs will NOT be
# traced as a whole."  The granularities are MUTUALLY EXCLUSIVE, so the primary
# `node` run cannot also answer E1-(b); the four legs are run under both.
#
# X4 (rev1 defect): the rev1 companion had NO full-GPU graph control, so
# "--cuda-graph-trace=graph emits no whole-graph rows ANYWHERE" -- a
# green-independent tool limit -- scored as E1B_DEAD, i.e. "the (b) path is
# dead under green context".  That is the parent audit's E2 reproduced inside
# the repair for E1.  The control leg is already being bought; it only had to
# be scored.
#
# NOT answered by the docs: the `[:<launch origin>]` suffix is 'host-only' |
# 'host-and-device' -- WHERE the launch originated (host vs device code).  It
# does NOT separate capture-time from replay-time launches, so it does not
# close the parent E5.
# ============================================================================
E1B_UNMEAS  = "E1B_UNMEASURED"
E1B_INVALID = "E1B_PROBE_INVALID"
E1B_NOTRACE = "E1B_GRAPH_TRACE_UNAVAILABLE"   # X4: green-INDEPENDENT tool limit
E1B_DEAD    = "E1B_DEAD"                      # whole-graph rows absent under green
E1B_STREAM  = "E1B_NEEDS_STREAM"              # rows present, ctx unusable
E1B_ALIVE   = "E1B_ALIVE"
E1B_LABELS = {E1B_UNMEAS, E1B_INVALID, E1B_NOTRACE, E1B_DEAD, E1B_STREAM, E1B_ALIVE}

G_AXES = dict(run_ok=[True, False], export=["ok", "fail"],
              l1g=["ok", "zero"], l2g=["ok", "zero"],
              capture=["ok", "fail_all"], green=["matched", "absent"],
              l4g_rows=["ok", "zero"],
              l4g_ctx=["null", "zero", "parent", "distinct"])


class GWorld:
    def __init__(self, run_ok=True, export="ok", l1g="ok", l2g="ok",
                 capture="ok", green="matched", l4g_rows="ok", l4g_ctx="distinct"):
        self.run_ok, self.export = run_ok, export
        self.l1g, self.l2g = l1g, l2g
        self.capture, self.green = capture, green
        self.l4g_rows, self.l4g_ctx = l4g_rows, l4g_ctx

    def __repr__(self):
        return (f"G(exp={self.export},l1g={self.l1g},l2g={self.l2g},"
                f"cap={self.capture},green={self.green},rows={self.l4g_rows},"
                f"ctx={self.l4g_ctx})")


def companion(g, guards=frozenset()):
    def on(x):
        return x not in guards
    if on("h_measure") and (not g.run_ok or g.export == "fail"):
        return E1B_UNMEAS
    if on("h_ctl") and (g.l1g == "zero" or g.capture == "fail_all"):
        return E1B_INVALID
    if on("h_green") and g.green == "absent":
        return E1B_INVALID
    if on("h_l2g") and g.l2g == "zero":
        return E1B_NOTRACE          # X4: green-independent -- NOT "(b) is dead"
    if on("h_rows") and g.l4g_rows == "zero":
        return E1B_DEAD
    if on("h_ctx") and g.l4g_ctx != "distinct":
        return E1B_STREAM
    return E1B_ALIVE


def gworlds():
    keys = list(G_AXES)
    for combo in product(*(G_AXES[k] for k in keys)):
        yield GWorld(**dict(zip(keys, combo)))


H_MUTANTS = ["h_measure", "h_ctl", "h_green", "h_l2g", "h_rows", "h_ctx"]


# ============================================================================
# SELF-TEST
#
# Every DISCRIMINATING check names mutants that MUST break it (T8), and every
# mutant must be broken by at least one discriminating check (T10).  rev1 had
# T8 but not T10, and its discriminating checks covered only 8 of 12 mutants --
# the four uncovered ones included the PRIMARY estimand's branch and its
# threshold.  A check that no mutant can break is an identity dressed as
# evidence (gate #9); a mutant that no check can catch is an unguarded degree
# of freedom.  Both are needed.
#
# NOTE on rev1's "self-detected identity": that diagnosis was WRONG (audit X2).
# The original T6b was broken by g_contra -- it was a check/mutant PAIRING
# failure, not an identity.  The claim is retracted; T10 is what rev1 actually
# needed.
# ============================================================================
CONTROL_FAILED = lambda w: (not w.run_ok or w.export != "ok" or w.capture != "ok"
                            or w.l1 == "zero" or w.l2 == "zero" or w.l3 == "zero"
                            or w.l2_post == "zero" or w.green != "matched"
                            or w.profile not in PROFILE_UNIFORM)
SUBSTANTIVE = {UNCONSTR, EXCESS, NOATTR, CONTRA, CAPJOIN, TIMEATTR, PARTIAL, OK}


def _reaches(w):
    """Worlds that get past every measurement/control branch of the intact rule."""
    return not CONTROL_FAILED(w) and w.frac <= 1.0


# Each entry: name -> (predicate over (worlds, labels), mutants that must break it)
def _mk_checks():
    def t4(W, L):
        return not any(l[0] in SUBSTANTIVE and CONTROL_FAILED(w)
                       for w, l in zip(W, L))

    def t6b(W, L):
        return not any(w.ctx in ("null", "zero", "parent") and l[1] == "ctx+stream"
                       for w, l in zip(W, L) if l[1] is not None)

    def t7(W, L):
        return not any(w.join_target == "capture_only" and l[0] == OK
                       for w, l in zip(W, L))

    def t9a(W, L):
        # X1: restate the threshold as a literal, independently of Q1_FRAC.
        return all((l[0] == UNCONSTR) == (w.frac < 0.90)
                   for w, l in zip(W, L) if _reaches(w))

    def t9b(W, L):
        # X1: JOIN_HIGH was dead code in rev1.  This is what makes it live.
        return all(w.join_rate >= 0.95 for w, l in zip(W, L) if l[0] == OK)

    def t11(W, L):
        return all((l[0] == CONTRA) == (w.ctx == "distinct" and w.stream == "mismatch")
                   for w, l in zip(W, L) if _reaches(w) and w.frac >= 0.90)

    def t12(W, L):
        return all(l[0] == TIMEATTR for w, l in zip(W, L)
                   if _reaches(w) and w.frac >= 0.90 and w.join_target == "none"
                   and not (w.ctx == "distinct" and w.stream == "mismatch")
                   and not (w.ctx not in ("distinct",) and w.stream == "mismatch"))

    def t14(W, L):
        return not any(w.green != "matched" and l[0] in SUBSTANTIVE
                       for w, l in zip(W, L))

    def t15(W, L):
        return not any((w.export == "partial" or w.l2_post == "zero"
                        or w.profile == "tail_missing") and l[0] in SUBSTANTIVE
                       for w, l in zip(W, L))

    def t16(W, L):
        return not any(w.frac > 1.0 and l[0] in (OK, PARTIAL, UNCONSTR)
                       for w, l in zip(W, L))

    def t17(W, L):
        return all(l[0] == CAPGREEN for w, l in zip(W, L)
                   if w.capture == "fail_green_only" and w.l2 == "ok"
                   and w.run_ok and w.export != "fail")

    def t18(W, L):
        return all(l[0] == ABSENT for w, l in zip(W, L)
                   if not w.run_ok or w.export == "fail")

    return {
        "T4  no substantive verdict on a failed control": (t4, {"g_l1"}),
        "T6b unusable ctx never reported on a ctx basis": (t6b, {"g_basis"}),
        "T7  capture-only join never CONSTRUCTIBLE":      (t7, {"g_capjoin"}),
        "T9a UNCONSTR <=> frac < 0.90 (literal)":         (t9a, {"g_q1_value", "g_q1_floor", "g_q1"}),
        "T9b OK => join_rate >= 0.95 (literal)":          (t9b, {"g_join_value"}),
        "T11 CONTRA <=> distinct ctx + mismatched stream": (t11, {"g_contra", "g_ctx"}),
        "T12 join_target=none => NEEDS_TIME_ATTRIBUTION": (t12, {"g_join_none"}),
        "T14 non-matched green never substantive":        (t14, {"g_green"}),
        "T15 truncation never substantive":               (t15, {"g_trunc"}),
        "T16 excess rows never OK/PARTIAL/UNCONSTR":      (t16, {"g_excess"}),
        "T17 green-only capture failure is preserved":    (t17, {"g_capgreen", "g_capture"}),
        "T18 run/export failure => MEASUREMENT_ABSENT":   (t18, {"g_measurement"}),
    }


def run():
    print(f"== Stage 0ppp A0 rule (RULE_REV={RULE_REV}) ==")
    W = list(worlds())
    print(f"   enumerating {len(W):,} worlds x {len(MUTANTS)} mutants ...")
    base = [score(w) for w in W]
    mut = {m: [score(w, {m}) for w in W] for m in MUTANTS}
    checks = _mk_checks()
    fails = []

    def chk(name, cond, detail=""):
        print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")
        if not cond:
            fails.append(name)

    chk("T1a totality", all(l in LABELS for l, _ in base))
    chk("T1b basis registered", all(b in BASES for _, b in base))
    chk("T1c determinism", base == [score(w) for w in W])

    from collections import Counter
    c = Counter(l for l, _ in base)
    for lab in sorted(LABELS):
        chk(f"T2 reachable: {lab}", c[lab] > 0, f"n={c[lab]}")

    for m in MUTANTS:
        d = sum(1 for a, b in zip(base, mut[m]) if a != b)
        chk(f"T3 mutant {m} is load-bearing", d > 0, f"n={d}")

    for name, (fn, _) in checks.items():
        chk(f"{name} [DISCRIM]", fn(W, base))

    # -- T8 META: each check must FAIL under each of its named mutants -------
    print("  -- T8 meta: does each DISCRIM check fail when broken? --")
    for name, (fn, muts) in checks.items():
        for m in muts:
            chk(f"T8 '{name[:38]}' fails under {m}", not fn(W, mut[m]))

    # -- T10 META (X2): every mutant must be caught by some check ------------
    print("  -- T10 meta: is every mutant covered by a discriminating check? --")
    uncovered = []
    for m in MUTANTS:
        if not any(not fn(W, mut[m]) for fn, _ in checks.values()):
            uncovered.append(m)
    chk("T10 every mutant is broken by >=1 discriminating check",
        not uncovered, f"uncovered={uncovered or 'none'}")

    # -- companion ----------------------------------------------------------
    G = list(gworlds())
    gbase = [companion(g) for g in G]
    chk(f"C1 totality ({len(G):,} graph-granularity worlds)",
        all(l in E1B_LABELS for l in gbase))
    gc = Counter(gbase)
    for lab in sorted(E1B_LABELS):
        chk(f"C2 reachable: {lab}", gc[lab] > 0, f"n={gc[lab]}")
    gmut = {m: [companion(g, {m}) for g in G] for m in H_MUTANTS}
    for m in H_MUTANTS:
        d = sum(1 for a, b in zip(gbase, gmut[m]) if a != b)
        chk(f"C3 mutant {m} is load-bearing", d > 0, f"n={d}")

    def c4(L):
        return not any(l == E1B_DEAD and (not g.run_ok or g.export == "fail"
                                          or g.l1g == "zero" or g.capture != "ok"
                                          or g.green != "matched")
                       for g, l in zip(G, L))

    def c5(L):   # X4's core
        return not any(l == E1B_DEAD and g.l2g == "zero" for g, l in zip(G, L))

    chk("C4 E1B_DEAD never on a failed control [DISCRIM]", c4(gbase))
    chk("C4 meta: fails under h_ctl", not c4(gmut["h_ctl"]))
    chk("C5 E1B_DEAD never when the graph-trace control is empty [DISCRIM]", c5(gbase))
    chk("C5 meta: fails under h_l2g", not c5(gmut["h_l2g"]))

    print(f"\n== {len(W):,} worlds - {len(LABELS)} labels - {len(MUTANTS)} mutants - "
          f"{len(checks)} discriminating checks | companion {len(G):,} worlds -> "
          f"{'ALL PASS' if not fails else str(len(fails)) + ' FAILURES: ' + '; '.join(fails[:4])} ==")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(run())
