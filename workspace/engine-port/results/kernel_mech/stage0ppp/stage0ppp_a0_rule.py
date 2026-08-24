#!/usr/bin/env python3
"""Stage 0''' A0 -- THE REGISTERED DECISION RULE, fixed in code (gate #66).

RULE_REV = 4.  rev2 was audited NO-GO (audit_stage0ppp_a0_rev2_2026-08-24, N1-N13,
no fatal).  Two of its findings were defects introduced by rev2 itself:

  N2  T14 and T15 -- registered as the VERIFICATION for X3 and X5 -- are
      entailed by T4.  Marginal binding: exactly zero worlds.  The change log
      cited their antecedent-set sizes as if they were new constraint.
  N1  Branch ORDER was an unguarded degree of freedom.  Four permutations moved
      3,840-18,816 labels and the suite still passed, and the new measurement
      labels masked each other: a trace truncated from L3 onward -- with the
      green partition ABSENT -- scored GREENCTX_INVISIBLE, i.e. "nsys cannot see
      green-context kernels", a statement about the tool.  Worse, T17 ENFORCED
      the bad order: fixing it made T17 fail.

rev3 reorders the guards, replaces both hollow checks with exclusivity forms,
and adds T19 -- no discriminating check may be entailed by another -- so this
class of defect cannot recur silently.

WHAT THIS FILE IS NOT: not a measurement, not a harness, not evidence.
Run:  python3 stage0ppp_a0_rule.py            # full self-test, writes JSON
"""
import hashlib, json, os, sys
from itertools import product

RULE_REV = 4

# --- registered labels -------------------------------------------------------
ABSENT   = "MEASUREMENT_ABSENT"
INVALID  = "PROBE_INVALID"
TRUNC    = "TRACE_TRUNCATED"
GREENBAD = "GREEN_PARTITION_MISMATCH"
CAPGREEN = "CAPTURE_FAILS_ONLY_UNDER_GREEN"
NONODE   = "NODE_TRACE_UNAVAILABLE"
NOGREEN  = "GREENCTX_INVISIBLE"
UNCONSTR = "PRIMARY_ESTIMAND_UNCONSTRUCTIBLE"
EXCESS   = "NODE_ROWS_EXCEED_EXPECTATION"
NOATTR   = "KSET_NOT_ATTRIBUTABLE"
CONTRA   = "CONTRADICTORY_ATTRIBUTION"
DISCONF  = "ATTRIBUTION_DISCONFIRMED"      # N4: positive evidence AGAINST green
CAPJOIN  = "KSET_JOINS_CAPTURE_NOT_REPLAY"
TIMEATTR = "KSET_NEEDS_TIME_ATTRIBUTION"
STREAMONLY = "KSET_STREAM_ONLY_ATTRIBUTION"  # N8: weak; NOT the primary estimand
EXPUNVER = "EXPECTATION_UNVERIFIABLE"      # R6: sec7 falls back to a mean denominator
PARTIAL  = "KSET_CONSTRUCTIBLE_PARTIAL"
OK       = "KSET_CONSTRUCTIBLE"            # N8: requires ctx+stream, not stream alone

MEASUREMENT = {ABSENT, INVALID, TRUNC, GREENBAD, CAPGREEN, EXPUNVER}
TOOLLIMIT   = {NONODE, NOGREEN}
SUBSTANTIVE = {UNCONSTR, EXCESS, NOATTR, CONTRA, DISCONF, CAPJOIN, TIMEATTR,
               STREAMONLY, PARTIAL, OK}
LABELS = MEASUREMENT | TOOLLIMIT | SUBSTANTIVE
BASES = {"ctx+stream", "stream_only", None}

# --- registered constants ----------------------------------------------------
Q1_FRAC   = 0.90
JOIN_HIGH = 0.95

# N3: rev2 pinned Q1_FRAC only to the band (0.80, 1.00] because no reachable
# world had a frac between them.  0.89/0.91 straddle it, the same trick rev2
# already used for join_rate and forgot to use here.
PROFILE_FRAC = {"full": 1.0, "uniform_short": 0.80, "just_below": 0.89,
                "just_above": 0.91, "tail_missing": 0.75, "excess": 1.30,
                "empty": 0.0, "eager_only": 0.0}
PROFILE_UNIFORM = {"full", "uniform_short", "just_below", "just_above",
                   "excess", "empty", "eager_only"}
# N9: a silent eager fallback -- the capture succeeded, the replay ran eagerly,
# so every row has graphNodeId == 0.  rev2 scored this KSET_NEEDS_TIME_
# ATTRIBUTION or (under the sec7 numerator predicate) PRIMARY_ESTIMAND_
# UNCONSTRUCTIBLE: a probe malfunction reported as a fact about the tool.
EAGER_ONLY = "eager_only"


class World:
    def __init__(self, run_ok=True, export="ok", l1="ok", l2="ok", l3="ok",
                 l2_post="ok", capture="ok", green="matched", profile="full",
                 ctx="distinct", stream="match", join_target="replay",
                 join_rate=1.0, l2_join="ok"):
        self.run_ok, self.export = run_ok, export
        self.l1, self.l2, self.l3, self.l2_post = l1, l2, l3, l2_post
        self.capture, self.green, self.profile = capture, green, profile
        self.ctx, self.stream = ctx, stream
        self.join_target, self.join_rate = join_target, join_rate
        self.l2_join = l2_join          # R6: does L2's join define the denominator?

    @property
    def frac(self):
        return PROFILE_FRAC[self.profile]

    def __repr__(self):
        return (f"W(exp={self.export},l1={self.l1},l2={self.l2},l3={self.l3},"
                f"l2p={self.l2_post},cap={self.capture},green={self.green},"
                f"prof={self.profile},ctx={self.ctx},str={self.stream},"
                f"jt={self.join_target},jr={self.join_rate},l2j={self.l2_join})")


def truncated(w):
    """R1: rev3 read `l2_post == "zero"` as truncation unconditionally, so the
    world that node-granularity failure PHYSICALLY forces -- L2 empty AND the
    trailing L2' empty -- was scored TRACE_TRUNCATED, and
    NODE_TRACE_UNAVAILABLE became reachable only in the self-contradictory
    world (L2 empty, L2' alive).  A trailing control can only have DROPPED if
    it was there to begin with."""
    return (w.export == "partial" or (w.l2 == "ok" and w.l2_post == "zero")
            or w.profile not in PROFILE_UNIFORM)


def score(w, guards=frozenset()):
    """The registered rule.  Returns (label, basis).

    ORDER IS PART OF THE RULE (N1).  It is registered as free parameter #26 and
    guarded by the exclusivity checks T20a-T20e, which fail if any guard is
    moved.  rev2's order was invisible to its own suite.
    """
    def on(g):
        return g not in guards

    if "g_order_capture_first" in guards and w.capture == "fail_all":
        return INVALID, None            # audit R4: measurement <-> capture swap
    # 1. measurement -- nothing was recorded
    if on("g_measurement") and (not w.run_ok or w.export == "fail"):
        return ABSENT, None
    # 2. the probe itself failed outright
    if on("g_capture") and w.capture == "fail_all":
        return INVALID, None
    # -- ORDER MUTANTS (N1).  Each reproduces rev2's ordering, which its own
    #    suite could not see: four permutations moved 3,840-18,816 labels and
    #    everything still passed.
    if "g_order_capgreen_early" in guards and w.capture == "fail_green_only" \
            and w.l2 == "ok":
        return CAPGREEN, None
    if "g_order_ctl_before_trunc" in guards:
        if w.l1 == "zero":
            return INVALID, None
        if w.l2 == "zero":
            return NONODE, None
        if w.l3 == "zero":
            return NOGREEN, None
    # 3. INTEGRITY before anything is read off the trace (N1).  rev2 put the
    #    existence controls first, so a truncated trace was read as "the tool
    #    cannot see green-context kernels".
    if on("g_trunc") and truncated(w):
        return TRUNC, None
    # 4. was the green partition actually realized?  (X3; still before the
    #    existence controls, so "green is absent" never reads as "green is
    #    invisible to nsys".)
    if "g_order_ctl_before_green" in guards:
        if w.l1 == "zero":
            return INVALID, None
        if w.l2 == "zero":
            return NONODE, None
        if w.l3 == "zero":
            return NOGREEN, None
    if on("g_green"):
        if w.green == "absent":
            return INVALID, None
        if w.green == "mismatched":
            return GREENBAD, None
    if "g_order_eager_first" in guards and w.profile == EAGER_ONLY:
        return INVALID, None            # audit R4: eager moved ahead of L1/L2/L3
    # 5. existence controls
    if on("g_l1") and w.l1 == "zero":
        return INVALID, None
    if on("g_l2") and w.l2 == "zero":
        return NONODE, None
    if on("g_l3") and w.l3 == "zero":
        return NOGREEN, None
    # 6. asymmetric capture failure -- LAST, so the label is clean by
    #    construction (X13 kept, N1 fixed: rev2 had it first and 23 of its 24
    #    reachable worlds carried some other failure too).
    if on("g_capgreen") and w.capture == "fail_green_only":
        return CAPGREEN, None
    # 7. N9: capture ok, replay ran eagerly -> no graph nodes exist to count.
    if on("g_eager") and w.profile == EAGER_ONLY:
        return INVALID, None

    # 8. R6: sec7 registers a fallback -- if L2's correlationId join fails the
    #    expectation becomes total/20 instead of the per-replay median.  Under
    #    that denominator a UNIFORM shortfall in both legs cancels and frac
    #    reads ~1.0 (lesson #20), so Q1 cannot be scored from it.
    if on("g_expbasis") and w.l2_join == "fail":
        return EXPUNVER, None
    if on("g_excess") and w.frac > 1.0:
        return EXCESS, None
    floor = Q1_FRAC
    if not on("g_q1_floor"):
        floor = 0.50
    if not on("g_q1_value"):
        floor = 0.85            # N3: now detectable, 0.89 straddles it
    if on("g_q1") and w.frac < floor:
        return UNCONSTR, None

    # 9. attribution
    ctx_green = w.ctx == "distinct"
    ctx_against = w.ctx == "parent"      # N4: evidence the rows are NOT green
    if not on("g_ctx"):
        ctx_green = w.ctx in ("distinct", "parent")
        ctx_against = False
    strm_ok = w.stream == "match"
    if on("g_contra") and ctx_green and not strm_ok:
        return CONTRA, None
    # N4 (mirror image): rev2 caught only "ctx says green, stream says not".
    # The other direction -- the rows are tagged with the PARENT context while
    # the stream matches -- scored KSET_CONSTRUCTIBLE/stream_only, i.e. positive
    # disconfirming evidence produced the top positive label.
    if on("g_disconf") and ctx_against:
        return DISCONF, None
    if not ctx_green and not strm_ok:
        return NOATTR, None
    basis = "ctx+stream" if (ctx_green and strm_ok) else "stream_only"
    if not on("g_basis"):
        basis = "ctx+stream"

    # 10. join
    if on("g_capjoin") and w.join_target == "capture_only":
        return CAPJOIN, basis
    if on("g_join_none") and w.join_target == "none":
        return TIMEATTR, basis
    jfloor = JOIN_HIGH if on("g_join_value") else 0.10
    if w.join_rate < jfloor:
        return PARTIAL, basis
    # N8: stream placement is registered (#15) so that L3 and L4 share the green
    # stream -- which makes `stream == "match"` STRUCTURAL, not evidence.  A
    # basis that rests on it alone therefore cannot carry the primary estimand.
    if on("g_streamonly") and basis == "stream_only":
        return STREAMONLY, basis
    return OK, basis


AXES = dict(
    run_ok=[True, False], export=["ok", "partial", "fail"],
    l1=["ok", "zero"], l2=["ok", "zero"], l3=["ok", "zero"],
    l2_post=["ok", "zero"], capture=["ok", "fail_green_only", "fail_all"],
    green=["matched", "mismatched", "absent"], profile=list(PROFILE_FRAC),
    ctx=["null", "zero", "parent", "distinct"], stream=["match", "mismatch"],
    join_target=["none", "capture_only", "replay"],
    join_rate=[0.0, 0.94, 0.95, 1.0], l2_join=["ok", "fail"],
)



def consistent(w):
    """Conservative physical-consistency model (R9).

    A world violating one of these cannot occur on the substrate, so a label
    reachable ONLY through such worlds is not reachable at all.  rev3's
    NODE_TRACE_UNAVAILABLE was exactly that (R1).  Conservative = when in doubt
    the world is kept, so this under-rejects rather than over-rejects.
    """
    if not w.run_ok and w.export != "fail":
        return False                       # a run that never happened exports nothing
    if w.export == "fail" and not (w.l1 == w.l2 == w.l3 == w.l2_post == "zero"
                                   and w.profile == "empty" and w.l2_join == "fail"):
        return False
    if w.l1 == "zero" and not (w.l2 == w.l3 == w.l2_post == "zero"
                               and w.profile == "empty"):
        return False                       # nsys saw nothing at all
    if w.capture != "ok" and w.profile != "empty":
        return False                       # no capture -> no graph rows on L4
    if w.l2 == "zero" and not (w.profile == "empty" and w.l2_post == "zero"
                               and w.l2_join == "fail"):
        return False                       # no node granularity -> no expectation, no L2'
    if w.profile != "empty" and (w.capture != "ok" or w.l2 == "zero"):
        return False
    if w.profile == EAGER_ONLY and w.join_target != "none":
        return False                       # eager replay has no graph launch to join
    if w.join_target != "none" and w.profile in ("empty", EAGER_ONLY):
        return False
    if w.green == "absent" and w.ctx == "distinct":
        return False                       # no green context -> no distinct greenContextId
    return True


def worlds():
    keys = list(AXES)
    for combo in product(*(AXES[k] for k in keys)):
        yield World(**dict(zip(keys, combo)))


MUTANTS = ["g_measurement", "g_capture", "g_trunc", "g_green", "g_l1", "g_l2",
           "g_l3", "g_capgreen", "g_eager", "g_excess", "g_q1", "g_q1_floor",
           "g_q1_value", "g_ctx", "g_contra", "g_disconf", "g_capjoin",
           "g_join_none", "g_join_value", "g_basis", "g_streamonly",
           "g_expbasis", "g_order_capgreen_early", "g_order_ctl_before_trunc",
           "g_order_ctl_before_green", "g_order_capture_first",
           "g_order_eager_first"]


# ============================================================================
# COMPANION -- E1-(b), granularity `graph`, bought in the same job.
# N5/N7: rev2's companion never inherited the X3/X5/X13 repairs and had three
# legs while the prereg registered five.  A truncated graph-granularity trace
# therefore scored E1B_DEAD -- "the (b) path is dead under green context".
# ============================================================================
E1B_UNMEAS  = "E1B_UNMEASURED"
E1B_INVALID = "E1B_PROBE_INVALID"
E1B_TRUNC   = "E1B_TRACE_TRUNCATED"
E1B_GREENBAD = "E1B_GREEN_PARTITION_MISMATCH"
E1B_NOTRACE = "E1B_GRAPH_TRACE_UNAVAILABLE"   # green-INDEPENDENT tool limit
E1B_DEAD    = "E1B_DEAD"
E1B_STREAM  = "E1B_NEEDS_STREAM"
E1B_ALIVE   = "E1B_ALIVE"
E1B_LABELS = {E1B_UNMEAS, E1B_INVALID, E1B_TRUNC, E1B_GREENBAD, E1B_NOTRACE,
              E1B_DEAD, E1B_STREAM, E1B_ALIVE}

G_AXES = dict(run_ok=[True, False], export=["ok", "partial", "fail"],
              l1g=["ok", "zero"], l2g=["ok", "zero"], l3g=["ok", "zero"],
              l2g_post=["ok", "zero"],
              capture=["ok", "fail_green_only", "fail_all"],
              green=["matched", "mismatched", "absent"],
              l4g_rows=["ok", "zero"],
              l4g_ctx=["null", "zero", "parent", "distinct"])


class GWorld:
    def __init__(self, run_ok=True, export="ok", l1g="ok", l2g="ok", l3g="ok",
                 l2g_post="ok", capture="ok", green="matched", l4g_rows="ok",
                 l4g_ctx="distinct"):
        self.run_ok, self.export = run_ok, export
        self.l1g, self.l2g, self.l3g, self.l2g_post = l1g, l2g, l3g, l2g_post
        self.capture, self.green = capture, green
        self.l4g_rows, self.l4g_ctx = l4g_rows, l4g_ctx

    def __repr__(self):
        return (f"G(exp={self.export},l1g={self.l1g},l2g={self.l2g},l3g={self.l3g},"
                f"l2gp={self.l2g_post},cap={self.capture},green={self.green},"
                f"rows={self.l4g_rows},ctx={self.l4g_ctx})")


def companion(g, guards=frozenset()):
    """Same guard ORDER as score() -- N1/N5: the two rules must not disagree
    about which failure dominates, or the same masking bug lives here."""
    def on(x):
        return x not in guards
    if on("h_measure") and (not g.run_ok or g.export == "fail"):
        return E1B_UNMEAS
    if on("h_capture") and g.capture == "fail_all":
        return E1B_INVALID
    if on("h_trunc") and (g.export == "partial" or g.l2g_post == "zero"):
        return E1B_TRUNC
    if on("h_green"):
        if g.green == "absent":
            return E1B_INVALID
        if g.green == "mismatched":
            return E1B_GREENBAD
    if on("h_ctl") and g.l1g == "zero":
        return E1B_INVALID
    if on("h_l2g") and g.l2g == "zero":
        return E1B_NOTRACE          # green-independent
    if on("h_l3g") and g.l3g == "zero":
        return E1B_NOTRACE
    if on("h_capgreen") and g.capture == "fail_green_only":
        return E1B_INVALID
    if on("h_rows") and g.l4g_rows == "zero":
        return E1B_DEAD
    if on("h_ctx") and g.l4g_ctx != "distinct":
        return E1B_STREAM
    return E1B_ALIVE


def gworlds():
    keys = list(G_AXES)
    for combo in product(*(G_AXES[k] for k in keys)):
        yield GWorld(**dict(zip(keys, combo)))


H_MUTANTS = ["h_measure", "h_capture", "h_trunc", "h_green", "h_ctl", "h_l2g",
             "h_l3g", "h_capgreen", "h_rows", "h_ctx"]


# ============================================================================
# SELF-TEST.  Three meta-checks, each guarding a way the suite can lie:
#   T8  a check that no mutant can break is an identity.
#   T10 a mutant that no check catches is an unguarded degree of freedom.
#   T19 a check ENTAILED by another adds nothing -- rev2's T14/T15 were of this
#       kind and were reported as the verification for X3 and X5 (N2).
# ============================================================================
def _pack(bits):
    b = bytearray((len(bits) + 7) // 8)
    for i, v in enumerate(bits):
        if v:
            b[i >> 3] |= 1 << (i & 7)
    return int.from_bytes(bytes(b), "little")


def _mk_checks():
    """name -> (per-world predicate, mutants that MUST break it).

    Every check is per-world so entailment between checks is decidable (T19).
    """
    def t4(w, l):
        fail = (not w.run_ok or w.export != "ok" or w.capture != "ok"
                or w.l1 == "zero" or w.l2 == "zero" or w.l3 == "zero"
                or w.l2_post == "zero" or w.green != "matched"
                or w.profile not in PROFILE_UNIFORM)
        return not (l[0] in SUBSTANTIVE and fail)

    def t6b(w, l):
        return not (w.ctx in ("null", "zero", "parent") and l[1] == "ctx+stream")

    def t7(w, l):
        return not (w.join_target == "capture_only" and l[0] == OK)

    def t9a(w, l):
        return (l[0] == UNCONSTR) == (_reaches(w) and 0.0 <= w.frac < 0.90)

    def t9b(w, l):          # N3: iff, so raising JOIN_HIGH is detected too
        return (l[0] == PARTIAL) == (_reaches_join(w) and w.join_target == "replay"
                                     and w.join_rate < 0.95)

    def t11(w, l):          # N4: both directions
        return (l[0] == CONTRA) == (_reaches_attr(w) and w.ctx == "distinct"
                                    and w.stream == "mismatch")

    def t11b(w, l):
        return (l[0] == DISCONF) == (_reaches_attr(w) and w.ctx == "parent")

    def t12(w, l):
        return (l[0] == TIMEATTR) == (_reaches_join(w) and w.join_target == "none")

    def t16(w, l):
        return not (w.frac > 1.0 and l[0] in (OK, PARTIAL, STREAMONLY, UNCONSTR))

    # -- N1/N2: exclusivity.  These say WHICH label a failure gets, not merely
    #    "not substantive" -- that weaker form is what T4 already entailed.
    # R4: iff, so a guard MOVED (not merely deleted) is caught.  rev3's
    # one-way forms could not see the five order permutations that survived.
    def _clean(w):
        return (w.run_ok and w.export != "fail" and w.capture != "fail_all"
                and not truncated(w) and w.green == "matched")

    def t20a(w, l):
        return (l[0] == NOGREEN) == (_clean(w) and w.l1 == "ok" and w.l2 == "ok"
                                     and w.l3 == "zero")

    def t20b(w, l):
        return (l[0] == NONODE) == (_clean(w) and w.l1 == "ok" and w.l2 == "zero")

    def t20c(w, l):
        return (l[0] == CAPGREEN) == (_clean(w) and w.capture == "fail_green_only"
                                      and w.l1 == "ok" and w.l2 == "ok"
                                      and w.l3 == "ok")

    def t20d(w, l):
        return (l[0] == TRUNC) == (truncated(w) and w.run_ok and w.export != "fail"
                                   and w.capture != "fail_all")

    def t20e(w, l):
        return (l[0] == GREENBAD) == (w.green == "mismatched" and w.run_ok
                                      and w.export == "ok" and not truncated(w)
                                      and w.capture != "fail_all")

    def t21(w, l):          # N8: the primary estimand never rests on stream alone
        return not (l[0] == OK and l[1] != "ctx+stream")

    def t18(w, l):
        # R4: iff, so moving the capture branch AHEAD of the measurement branch
        # is caught.  (An earlier edit of mine deleted this check outright; T10
        # is what surfaced its absence -- g_order_capture_first was uncovered.)
        return (l[0] == ABSENT) == (not w.run_ok or w.export == "fail")

    def t23(w, l):          # R6
        return (l[0] == EXPUNVER) == (_pre_eager(w) and w.profile != EAGER_ONLY
                                      and w.l2_join == "fail")

    # T22 REMOVED (R3 / T19b).  A dedicated check for the silent-eager guard
    # was registered in rev3 and re-derived in rev4; both times T19b showed its
    # sole-binding was zero -- T4 and T9a already fail under g_eager.  Keeping a
    # check that adds no constraint, while calling it "the verification for N9",
    # is the exact sin the rev2 audit named (T14/T15).  So it is deleted and the
    # honest record is: g_eager is detected by T4 and T9a.
    def t18(w, l):
        # R4: iff, so moving the capture branch AHEAD of the measurement branch
        # is caught.  (An earlier edit of mine deleted this check outright; T10
        # is what surfaced its absence -- g_order_capture_first was uncovered.)
        return (l[0] == ABSENT) == (not w.run_ok or w.export == "fail")

    def t23(w, l):          # R6
        return (l[0] == EXPUNVER) == (_pre_eager(w) and w.profile != EAGER_ONLY
                                      and w.l2_join == "fail")

    def t22(w, l):          # N9 / R3: rev3's one-way form had sole-binding 0
        return (_pre_eager(w) and l[0] == INVALID) == \
               (_pre_eager(w) and w.profile == EAGER_ONLY)

    return {
        "T4  failed control never substantive": (t4, {"g_l1"}),
        "T6b unusable ctx never on a ctx basis": (t6b, {"g_basis"}),
        "T7  capture-only join never OK":        (t7, {"g_capjoin"}),
        "T9a UNCONSTR iff frac < 0.90":          (t9a, {"g_q1", "g_q1_floor", "g_q1_value"}),
        "T9b PARTIAL iff join_rate < 0.95":      (t9b, {"g_join_value"}),
        "T11 CONTRA iff distinct+mismatch":      (t11, {"g_contra"}),
        "T11b DISCONF iff ctx=parent":           (t11b, {"g_disconf", "g_ctx"}),
        "T12 TIMEATTR iff join_target=none":     (t12, {"g_join_none"}),
        "T16 excess never OK/PARTIAL/UNCONSTR":  (t16, {"g_excess"}),
        "T20a GREENCTX_INVISIBLE is clean":      (t20a, {"g_trunc", "g_green", "g_order_ctl_before_trunc", "g_order_ctl_before_green"}),
        "T20b NODE_TRACE_UNAVAILABLE is clean":  (t20b, {"g_trunc", "g_green", "g_order_ctl_before_trunc", "g_order_eager_first"}),
        "T20c CAPGREEN is clean":                (t20c, {"g_order_capgreen_early"}),
        "T20d TRUNC iff truncated":              (t20d, {"g_trunc"}),
        "T20e GREENBAD iff mismatched":          (t20e, {"g_green"}),
        "T18 ABSENT iff run/export failed":      (t18, {"g_measurement", "g_order_capture_first"}),
        "T21 OK requires ctx+stream basis":      (t21, {"g_streamonly"}),
        "T23 EXPECTATION_UNVERIFIABLE iff l2_join fails": (t23, {"g_expbasis"}),
    }


def _pre_eager(w):
    return (w.run_ok and w.export == "ok" and w.capture == "ok"
            and w.green == "matched" and w.l1 == "ok" and w.l2 == "ok"
            and w.l3 == "ok" and not truncated(w))


def _reaches(w):
    return _pre_eager(w) and w.profile != EAGER_ONLY and w.l2_join == "ok"


def _reaches_attr(w):
    return _reaches(w) and 0.90 <= w.frac <= 1.0


def _reaches_join(w):   # reaches the JOIN STAGE (any join_target)
    if not _reaches_attr(w):
        return False
    if w.ctx == "distinct" and w.stream == "mismatch":
        return False
    if w.ctx == "parent":
        return False
    if w.ctx not in ("distinct",) and w.stream == "mismatch":
        return False
    return True


LABEL_PAIRS = [(l, b) for l in sorted(LABELS) for b in (None, "ctx+stream", "stream_only")]


def _reachable_assignments(W):
    """R3: rev3 quantified entailment over 17 labels x 3 bases attached to EVERY
    world -- a product space containing assignments `score()` can never emit.
    "No entailment" there is a strictly weaker claim, and it hid that T22's
    failures were all already caught by T4.  The honest space is the one the
    suite actually evaluates: base plus every mutant, on CONSISTENT worlds."""
    out = []
    for g in [frozenset()] + [frozenset({m}) for m in MUTANTS]:
        for w in W:
            out.append((w, score(w, g)))
    return out


def run(emit_json=True):
    here = os.path.dirname(os.path.abspath(__file__))
    print(f"== Stage 0ppp A0 rule (RULE_REV={RULE_REV}) ==")
    W = list(worlds())
    checks = _mk_checks()
    print(f"   {len(W):,} worlds x {len(MUTANTS)} mutants x {len(checks)} checks ...")
    base = [score(w) for w in W]
    fails, results = [], {}

    def chk(name, cond, detail=""):
        print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")
        results[name] = bool(cond)
        if not cond:
            fails.append(name)

    chk("T1a totality", all(l in LABELS for l, _ in base))
    chk("T1b basis registered", all(b in BASES for _, b in base))
    chk("T1c determinism", base == [score(w) for w in W])

    from collections import Counter
    c = Counter(l for l, _ in base)
    cons = [w for w in W if consistent(w)]
    cc = Counter(score(w)[0] for w in cons)
    for lab in sorted(LABELS):
        chk(f"T2 reachable: {lab}", c[lab] > 0, f"n={c[lab]}")
        # R1/R9: rev3's NODE_TRACE_UNAVAILABLE was reachable ONLY through
        # self-contradictory worlds.  Reachability must survive the model.
        chk(f"T2c reachable in a CONSISTENT world: {lab}", cc[lab] > 0, f"n={cc[lab]}")

    # base pass + how many worlds each check actually rejects under each mutant
    for name, (fn, _) in checks.items():
        bad = sum(1 for w, l in zip(W, base) if not fn(w, l))
        chk(f"{name} [DISCRIM]", bad == 0, f"" if bad == 0 else f"violations={bad}")

    mut_fail = {}          # (check, mutant) -> does the check fail?
    delta = {}
    for m in MUTANTS:
        lm = [score(w, {m}) for w in W]
        delta[m] = sum(1 for a, b in zip(base, lm) if a != b)
        # one pass per mutant, dropping each check as soon as it has failed
        pending = dict(checks)
        for w, l in zip(W, lm):
            if not pending:
                break
            for name in [n for n in pending]:
                if not pending[name][0](w, l):
                    mut_fail[(name, m)] = True
                    del pending[name]
        for name in checks:
            mut_fail.setdefault((name, m), False)
    for m in MUTANTS:
        chk(f"T3 mutant {m} is load-bearing", delta[m] > 0, f"n={delta[m]}")

    print("  -- T8 meta: each check fails under each of its named mutants --")
    for name, (fn, muts) in checks.items():
        for m in sorted(muts):
            chk(f"T8 '{name[:34]}' fails under {m}", mut_fail[(name, m)])

    print("  -- T10 meta: every mutant is caught by some check --")
    unc = [m for m in MUTANTS if not any(mut_fail[(n, m)] for n in checks)]
    chk("T10 every mutant broken by >=1 check", not unc, f"uncovered={unc or 'none'}")

    # -- T19 META (N2/R3): entailment AND sole-binding, on the space the
    #    suite actually evaluates (consistent worlds x base+mutants) ---------
    print("  -- T19 meta: entailment / sole-binding on reachable assignments --")
    CW = [w for w in W if consistent(w)]
    ASG = _reachable_assignments(CW)
    failset = {}
    for name, (fn, _) in checks.items():
        failset[name] = {i for i, (w, l) in enumerate(ASG) if not fn(w, l)}
    ent = [f"{a} => {b}" for a in checks for b in checks
           if a != b and failset[b] and failset[b] <= failset[a]]
    chk("T19a no check is entailed by another", not ent, f"entailed={ent or 'none'}")
    sole = {}
    for name in checks:
        others = set().union(*[failset[o] for o in checks if o != name]) if len(checks) > 1 else set()
        sole[name] = len(failset[name] - others)
    nosole = [n for n, v in sole.items() if v == 0]
    chk("T19b every check binds something on its own", not nosole,
        f"sole-binding=0 for {nosole or 'none'}")
    print(f"     ({len(CW):,} consistent worlds, {len(ASG):,} assignments)")

    # -- companion ------------------------------------------------------------
    G = list(gworlds())
    gbase = [companion(g) for g in G]
    chk(f"C1 totality ({len(G):,} graph worlds)", all(l in E1B_LABELS for l in gbase))
    gc = Counter(gbase)
    for lab in sorted(E1B_LABELS):
        chk(f"C2 reachable: {lab}", gc[lab] > 0, f"n={gc[lab]}")
    gmut = {m: [companion(g, {m}) for g in G] for m in H_MUTANTS}
    for m in H_MUTANTS:
        d = sum(1 for a, b in zip(gbase, gmut[m]) if a != b)
        chk(f"C3 mutant {m} is load-bearing", d > 0, f"n={d}")

    def c5(L):
        return not any(l == E1B_DEAD and g.l2g == "zero" for g, l in zip(G, L))

    def c6(L):   # N5: E1B_DEAD must be clean of every other failure
        return all(g.export == "ok" and g.l2g_post == "ok" and g.green == "matched"
                   and g.capture == "ok" and g.l1g == "ok" and g.l3g == "ok"
                   for g, l in zip(G, L) if l == E1B_DEAD)

    chk("C5 E1B_DEAD never when graph-trace control is empty", c5(gbase))
    chk("C5 meta: fails under h_l2g", not c5(gmut["h_l2g"]))
    chk("C6 E1B_DEAD is clean of every other failure", c6(gbase))
    chk("C6 meta: fails under h_trunc", not c6(gmut["h_trunc"]))

    ok = not fails
    print(f"\n== {len(W):,} worlds | {len(LABELS)} labels | {len(MUTANTS)} mutants | "
          f"{len(checks)} checks | companion {len(G):,} -> "
          f"{'ALL PASS' if ok else str(len(fails)) + ' FAILURES: ' + '; '.join(fails[:4])} ==")

    if emit_json:
        # N10: rev2's .json could not be produced by the committed script.
        with open(os.path.abspath(__file__), "rb") as f:
            sha = hashlib.sha256(f.read()).hexdigest()
        out = {"rule_rev": RULE_REV, "rule_sha256": sha, "date": "2026-08-24",
               "gpu_hr": 0.0, "n_worlds": len(W), "n_labels": len(LABELS),
               "n_mutants": len(MUTANTS), "n_checks": len(checks),
               "label_reachability": {k: c[k] for k in sorted(LABELS)},
               "mutant_delta": {m: delta[m] for m in MUTANTS},
               "mutant_coverage": {m: sorted(n for n in checks if mut_fail[(n, m)])
                                   for m in MUTANTS},
               "uncovered_mutants": unc, "entailed_pairs": ent,
               "sole_binding": sole, "n_consistent_worlds": len(CW),
               "n_reachable_assignments": len(ASG),
               "consistent_label_reachability": {k: cc[k] for k in sorted(LABELS)},
               "companion": {"n_worlds": len(G),
                             "labels": {k: gc[k] for k in sorted(E1B_LABELS)}},
               "checks": results, "all_pass": ok}
        p = os.path.join(here, f"selftest_rev{RULE_REV}_2026-08-24.json")
        with open(p, "w") as f:
            json.dump(out, f, indent=1, sort_keys=True)
        print(f"   wrote {os.path.basename(p)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(run())
