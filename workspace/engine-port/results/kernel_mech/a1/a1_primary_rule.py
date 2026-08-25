#!/usr/bin/env python3
"""A1 **primary** rule -- Q1e / Q2ae / Q2be on the BOOT-SEPARATED substrate.

WHY THIS IS A NEW FILE AND NOT `stage0ppp_a0_rule.py` rev5
----------------------------------------------------------
The A1 rev2 audit (`audit_a1_rev2_rules_2026-08-25`, B1') is right that the
primary estimand is still scored by the A0 rule, which encodes "four legs in ONE
report".  Its recommendation reads "A0 rule rev5".  It cannot be an edit:

    a0_verdict_892554.json / a0_verdict_892556.json pin
        rule_rev = 4,  rule_sha256 = 583c4ab08a1f2194d63178ac1dab40aa9625e601...

Editing that file in place would change the hash that a **completed, submitted**
experiment was scored under, and A0's own prereg carries a hard stop saying its
rules-layer audits are finished.  So rev4 stays frozen as the record of what
scored job 892556, and the boot-separated substrate gets its own rule here,
derived from it and stating every departure.

WHAT THE BOOT SEPARATION BREAKS IN rev4 (audit B1', item by item)
----------------------------------------------------------------
  `stream`      `match|mismatch` compared L3's streamId to L4's.  nsys streamIds
                are report-local, so across boots that comparison is not merely
                unavailable -- it is meaningless.  Replaced by TWO axes,
                `stream_single` and `stream_disjoint` (design sec 3.2).
  `consistent()` rev4 requires `l1 == "zero" => l2 == l3 == l2_post == "zero"`,
                i.e. "nsys saw nothing at all".  Across boots B-U's rows and
                B-G's rows have NO mutual implication: one boot can be empty
                while another is healthy.  Rewritten per boot.
  `l2_post`     the trailing control existed because both legs were in one
                report.  Replaced by `halves_s` -- node rows present in BOTH
                halves of B-S's replay order (design sec 2.1).
  `export`      one export state for one report becomes one PER BOOT.
  role          rev4 has no axis saying which boot a fact came from, so B-U
                cannot be scored at all: feed it a sticky-OFF boot and rev4's
                descendants answer `STICKY_NOT_REALIZED`.  `plumbing` and the
                per-boot row axes carry that here.
  N8 basis      in A0, `stream == "match"` was STRUCTURAL -- free parameter #15
                deliberately put L3 and L4 on the same green stream -- so it
                could not be evidence, and `OK` was made to require
                `ctx+stream` jointly.  That argument does not transfer: with
                the legs in different boots, #15 has nothing to arrange.
                ★The re-derivation is different and is stated in BASIS below.

WHAT THIS RULE DOES NOT DO
--------------------------
  * It does not score Q3 or K1 -- those are `a1_q3k1_rule.py`.
  * It does not claim the A1 fatal flaws are closed.  A2's time-window join
    requirement is gone; in its place sits ONE unmeasured premise (that B-U and
    B-S replay structurally identical graphs), and this rule makes that premise
    an explicit axis (`bu_quality`) rather than an assumption.
  * ★It is UNAUDITED.  "rev5 passed the rules layer" is a forbidden sentence.

Run:  python3 a1_primary_rule.py
"""
import hashlib, json, os, sys
from collections import Counter
from itertools import product

RULE_REV = 1                      # of THIS rule; derived from A0 rule rev4
DERIVED_FROM_SHA = "583c4ab08a1f2194d63178ac1dab40aa9625e601bc0f72d6ee516d8219082652"

# --- MEASUREMENT: the experiment did not deliver a scorable observation -------
ABSENT    = "MEASUREMENT_ABSENT"           # a boot never happened / export failed
TRUNC     = "TRACE_TRUNCATED"              # partial export, or one half empty
NOSTICKY  = "STICKY_NOT_REALIZED"          # B-S did not hold the division
UNSPLIT   = "UNSPLIT_NOT_REALIZED"         # ★B-U did not run unpartitioned
EXPAMBIG  = "EXPECTATION_AMBIGUOUS"        # ★B-U's node count is not unimodal
NOLEG     = "LEG_LABELLING_UNAVAILABLE"    # no decode-only kernel name
GREENBAD  = "GREEN_PARTITION_MISMATCH"     # driver says the green ctx is not D
MEASUREMENT = {ABSENT, TRUNC, NOSTICKY, UNSPLIT, EXPAMBIG, NOLEG, GREENBAD}

# --- TOOLLIMIT: nsys could not do the thing (a fact about the TOOL) ----------
# ★This layer is why A0 did not manufacture tool verdicts out of probe faults,
# and the rev2 audit's B2 is that the Q3 rule dropped it.  It is kept here.
NODEUNAVAIL = "NODE_TRACE_UNAVAILABLE"     # no node granularity anywhere
GREENINVIS  = "GREENCTX_INVISIBLE"         # green rows exist nowhere, controls fine
EAGERONLY   = "EAGER_REPLAY_ONLY"          # B-S replayed nothing to attribute
# ★NEW (re-audit P2).  THE modal negative of this whole experiment: the tool
# sees engine kernels (B-U alive) and sees them on a green stream when they run
# eager (B-G alive), and still emits no node rows for the green GRAPH REPLAY
# (B-S empty).  That is a statement about `nsys --cuda-graph-trace=node`, not
# about the estimand -- and the first version of this rule sent it to
# PRIMARY_ESTIMAND_UNCONSTRUCTIBLE, i.e. SUBSTANTIVE.  Gate #21, in the one
# world A1 exists to distinguish.  `GREENINVIS` did not catch it because that
# guard also requires B-G to be empty.
GRAPHINVIS  = "GRAPH_NODE_ROWS_INVISIBLE_UNDER_GREEN"
TOOLLIMIT = {NODEUNAVAIL, GREENINVIS, EAGERONLY, GRAPHINVIS}

# --- SUBSTANTIVE: an answer about the estimand -------------------------------
UNCONSTR  = "PRIMARY_ESTIMAND_UNCONSTRUCTIBLE"
CONTRA    = "CONTRADICTORY_ATTRIBUTION"
DISCONF   = "ATTRIBUTION_DISCONFIRMED"
NOATTR    = "KSET_NOT_ATTRIBUTABLE"
CAPJOIN   = "KSET_JOINS_AT_CAPTURE_ONLY"
STREAMONLY = "KSET_STREAM_ONLY_ATTRIBUTION"    # weak; NOT the primary estimand
OK        = "KSET_CONSTRUCTIBLE"
SUBSTANTIVE = {UNCONSTR, CONTRA, DISCONF, NOATTR, CAPJOIN, STREAMONLY, OK}

LABELS = MEASUREMENT | TOOLLIMIT | SUBSTANTIVE

# --- BASIS -------------------------------------------------------------------
# ★N8 RE-DERIVED (the audit's "N8 basis 근거 재작성").
#
# A0: `stream == "match"` was arranged by the harness (free parameter #15 put
# L3 and L4 on one green stream), so a match carried no information and `OK`
# was made to rest on `ctx+stream` jointly.
#
# Here nothing arranges it -- but `stream_single` ("all analysed rows sit on ONE
# streamId") is very nearly the same statement as `ctx == "distinct"` ("those
# rows carry one non-parent greenContextId"), because under sticky the decode
# stream IS the green division for the whole run.  Treating `single` as
# independent evidence would be an identity dressed as a second channel
# (gate #9).
#
# ⇒ the evidentiary content lives in `stream_disjoint`: the analysed rows sit on
#   a stream that carries NO prefill-only kernel, decided by KERNEL NAMES.
#
# ★SCOPE (re-audit P9), registered here because this is where the next revision
#   reads it.  Under sticky with FixedPolicy, prefill and decode sit on the two
#   green streams of ONE stream group -- so "the decode stream carries no
#   prefill-only kernel" is ARRANGED BY THE ENGINE SUBSTRATE, exactly as A0's
#   `stream == "match"` was arranged by free parameter #15.  N8's rule that
#   what is arranged is not evidence therefore still bites.
#   ⇒ what `disjoint` MEASURES is whether nsys' `streamId` column is FAITHFUL
#     to the engine's stream separation -- tool fidelity -- and NOT whether the
#     green-context attribution is correct.  That is necessary to establish and
#     it is weaker than "non-circular evidence for Q2ae".  Do not write the
#     stronger sentence.
#   So:
#       OK           requires ctx == "distinct" AND stream_disjoint == "yes"
#       STREAMONLY   is what ctx-without-disjoint gets, and is NOT primary
BASES = {"ctx+disjoint", "ctx_only", None}

# --- registered constants ----------------------------------------------------
Q1_FRAC = 0.90            # inherited from A0 prereg sec8 #12 (band (0.89,0.91])
JOIN_HIGH = 0.95          # inherited from A0 rev4
PROFILE_FRAC = {"full": 1.0, "just_above": 0.91, "just_below": 0.89,
                "eager_only": 0.0, "empty": 0.0}
EAGER_ONLY = "eager_only"


class World:
    """One possible outcome of the WHOLE boot matrix (design sec 2.2).

    plumbing        ok | boot_failed | partial_s | partial_ctl | fail
                    ★per-boot export, folded into one enumerated axis so the
                    world count stays near A0's.  `partial_s` and `partial_ctl`
                    are kept apart because a partial export in the CONDITION
                    boot and in a CONTROL boot are not the same failure.
    rows_s/u/g      node rows present in B-S / B-U / B-G
    halves_s        node rows in both halves of B-S's replay order (sec 2.1)
    sticky_s        B-S realized the division: ok | low | nolog
    bu_quality      ok | unsplit_low | expect_ambiguous   ★B-U's two gates
    profile         the Q1e fraction band in B-S
    ctx_s           greenContextId on B-S's rows: null | zero | parent | distinct
    stream_single   all B-S rows on ONE streamId
    stream_disjoint that stream carries no prefill-only kernel: yes|no|unanswerable
    green_s         DRIVER read-out (PDMUX_GREEN_READOUT), not nsys:
                    matched | mismatched | absent
    join_target     correlationId joins to: none | capture_only | replay
    join_rate       low (< JOIN_HIGH) | high
    """

    __slots__ = ("plumbing", "rows_s", "rows_u", "rows_g", "halves_s",
                 "sticky_s", "bu_quality", "profile", "ctx_s", "stream_single",
                 "stream_disjoint", "green_s", "join_target", "join_rate")

    def __init__(self, plumbing="ok", rows_s="ok", rows_u="ok", rows_g="ok",
                 halves_s="both", sticky_s="ok", bu_quality="ok",
                 profile="full", ctx_s="distinct", stream_single="yes",
                 stream_disjoint="yes", green_s="matched",
                 join_target="replay", join_rate="high"):
        self.plumbing, self.rows_s, self.rows_u, self.rows_g = plumbing, rows_s, rows_u, rows_g
        self.halves_s, self.sticky_s, self.bu_quality = halves_s, sticky_s, bu_quality
        self.profile, self.ctx_s = profile, ctx_s
        self.stream_single, self.stream_disjoint = stream_single, stream_disjoint
        self.green_s, self.join_target, self.join_rate = green_s, join_target, join_rate

    @property
    def frac(self):
        return PROFILE_FRAC[self.profile]

    def __repr__(self):
        return (f"W(pl={self.plumbing},s={self.rows_s},u={self.rows_u},"
                f"g={self.rows_g},hv={self.halves_s},st={self.sticky_s},"
                f"bu={self.bu_quality},pr={self.profile},ctx={self.ctx_s},"
                f"sg={self.stream_single},dj={self.stream_disjoint},"
                f"gr={self.green_s},jt={self.join_target},jr={self.join_rate})")


AXES = dict(
    plumbing=["ok", "boot_failed", "partial_s", "partial_ctl", "fail"],
    rows_s=["ok", "zero"], rows_u=["ok", "zero"], rows_g=["ok", "zero"],
    halves_s=["both", "one"],
    sticky_s=["ok", "low", "nolog"],
    bu_quality=["ok", "unsplit_low", "expect_ambiguous"],
    profile=list(PROFILE_FRAC),
    ctx_s=["null", "zero", "parent", "distinct"],
    stream_single=["yes", "no"],
    stream_disjoint=["yes", "no", "unanswerable"],
    green_s=["matched", "mismatched", "absent"],
    join_target=["none", "capture_only", "replay"],
    join_rate=["low", "high"],
)


def score(w, guards=frozenset()):
    """Returns (label, basis).  ORDER IS PART OF THE RULE (A0's N1 lesson).

    The three tiers are decided in this order and never mixed:
        MEASUREMENT  -- the experiment did not deliver an observation
        TOOLLIMIT    -- nsys could not do the thing (a fact about the TOOL)
        SUBSTANTIVE  -- an answer about the estimand
    ★The rev2 audit's B2 is that the Q3 rule collapsed the middle tier and then
    sent probe faults to substantive labels.  This rule keeps all three.
    """
    def on(g):
        return g not in guards

    def _basis(label, basis):
        # ADDITIVE MUTANT: emit the primary label with no basis recorded.
        if "g_basis_string" in guards and label == OK:
            return label, None
        return label, basis

    # ---- tier 1: MEASUREMENT ------------------------------------------------
    if on("g_absent") and w.plumbing in ("boot_failed", "fail"):
        return ABSENT, None
    # A0 harness audit / job 892554: a partial export must never reach a
    # substantive label, in EITHER the condition boot or a control boot.
    if on("g_trunc") and (w.plumbing in ("partial_s", "partial_ctl")
                          or w.halves_s != "both"):
        return TRUNC, None
    if on("g_sticky") and w.sticky_s != "ok":
        return NOSTICKY, None
    # ★NEW vs rev4: B-U has its own gates and they are decided BEFORE the
    # expectation is used, because the expectation is what B-U is for.
    if on("g_unsplit") and w.bu_quality == "unsplit_low":
        return UNSPLIT, None
    if on("g_expambig") and w.bu_quality == "expect_ambiguous":
        return EXPAMBIG, None
    # the driver read-out (design sec 3.1).  nsys is NOT consulted here.
    if on("g_green") and w.green_s == "mismatched":
        return GREENBAD, None
    # ---- ORDER MUTANT (A0 N1 shape): leg labelling decided after the rows are
    #      read turns "we cannot tell decode rows from prefill rows" into a
    #      statement about attribution.
    if "g_order_leg_late" not in guards:
        # ★`rows_s == "ok"` is REQUIRED here and its absence was a real defect
        # in the first run of this suite: with no rows in B-S, `disjoint` is
        # unanswerable BECAUSE THERE IS NO TRACE, which is a tool limit, not a
        # leg-labelling failure.  Without this conjunct NOLEG masked both
        # TOOLLIMIT labels and T2 reported them unreachable -- the A0 R1 shape
        # ("the most plausible negative comes out as a re-run signal").
        if on("g_leg") and w.rows_s == "ok" and w.stream_disjoint == "unanswerable":
            return NOLEG, None

    # ---- tier 2: TOOLLIMIT --------------------------------------------------
    # No node granularity anywhere: the tool, not the experiment, is the limit.
    if on("g_nodeunavail") and w.rows_s == "zero" and w.rows_u == "zero" \
            and w.rows_g == "zero":
        return NODEUNAVAIL, None
    # Controls healthy, condition empty, and nothing green was ever visible.
    if on("g_greeninvis") and w.rows_u == "ok" and w.rows_s == "zero" \
            and w.rows_g == "zero":
        return GREENINVIS, None
    if on("g_eager") and w.profile == EAGER_ONLY:
        return EAGERONLY, None
    # ★P2: controls alive on BOTH sides, condition empty.  Ordered after the
    # two narrower tool limits so it does not swallow them.
    if on("g_graphinvis") and w.rows_g == "ok" and w.rows_s == "zero":
        return GRAPHINVIS, None
    if "g_order_leg_late" in guards:
        if w.rows_s == "ok" and w.stream_disjoint == "unanswerable":
            return NOLEG, None

    # ---- tier 3: SUBSTANTIVE ------------------------------------------------
    # ★P2: `rows_s == "zero"` never reaches here any more -- the tool tier owns
    # it.  What remains is a condition boot that produced rows but no usable
    # fraction.
    if on("g_unconstr") and w.profile == "empty":
        return UNCONSTR, None
    if on("g_q1") and w.frac < (Q1_FRAC if on("g_q1_value") else 0.50):
        return UNCONSTR, None
    # attribution
    ctx_green = w.ctx_s == "distinct"
    # ★P10: the driver says there is no green context and nsys reports a
    # distinct one.  Two channels disagreeing is a harness/analyser fact.
    if on("g_contra") and w.green_s == "absent" and w.ctx_s == "distinct":
        return CONTRA, None
    if on("g_contra") and w.ctx_s == "parent" and w.stream_disjoint == "yes":
        # the ID says "not green" while the kernel-name channel says the stream
        # is a decode-only stream.  Two channels disagree -> harness/analyser.
        return CONTRA, None
    if on("g_disconf") and w.ctx_s in ("null", "zero") and w.green_s == "absent":
        return DISCONF, None
    if on("g_noattr") and not ctx_green:
        return NOATTR, None
    # join (Q2be)
    if on("g_join_none") and w.join_target == "none":
        return NOATTR, None
    if on("g_capjoin") and w.join_target == "capture_only":
        return CAPJOIN, None
    if on("g_joinrate") and w.join_rate == "low":
        return NOATTR, None
    # ★BASIS.  `single` alone is near-identity with `ctx == distinct`; the
    # independent channel is `disjoint` (kernel names).  See BASIS above.
    if on("g_basis") and w.stream_disjoint != "yes":
        return _basis(STREAMONLY, "ctx_only")
    if on("g_single") and w.stream_single != "yes":
        # rows spread over several streams: the set is not one stream's replays
        return NOATTR, None
    return _basis(OK, "ctx+disjoint")


# ★TWO MUTANTS DELETED, with the reason recorded rather than a check invented
# for them (A0 rev4 removed T22 the same way when sole binding came up 0 twice):
#   g_unconstr  removing it leaves `rows_s == "zero"` to `g_q1`, because
#               consistent() forces `profile == "empty"` there and
#               PROFILE_FRAC["empty"] = 0.0 < Q1_FRAC.  Same label either way,
#               so the mutant moved 0 worlds.  The two routes agreeing is
#               INTENDED -- an empty condition boot and a below-threshold
#               fraction are the same verdict -- so the redundancy is the
#               design, not a defect.
#   g_join_none removing it leaves `join_target == "none"` to `g_joinrate`,
#               because consistent() forces `join_rate == "low"` there.  Same
#               label, 0 worlds moved.
MUTANTS = ["g_absent", "g_trunc", "g_sticky", "g_unsplit", "g_expambig",
           "g_green", "g_leg", "g_nodeunavail", "g_greeninvis", "g_eager",
           "g_q1", "g_q1_value", "g_contra", "g_disconf",
           "g_noattr", "g_capjoin", "g_joinrate", "g_basis",
           "g_single", "g_order_leg_late", "g_basis_string", "g_graphinvis"]

# ★Which guard is registered as PRODUCING which label.  T25 uses this: a
# necessary-condition check cannot see a label that stopped appearing, so the
# removal mutants need their own meta-check (see _battery).
GUARD_LABEL = {
    "g_absent": ABSENT, "g_trunc": TRUNC, "g_sticky": NOSTICKY,
    "g_unsplit": UNSPLIT, "g_expambig": EXPAMBIG, "g_green": GREENBAD,
    "g_leg": NOLEG, "g_nodeunavail": NODEUNAVAIL, "g_greeninvis": GREENINVIS,
    "g_eager": EAGERONLY, "g_contra": CONTRA, "g_disconf": DISCONF,
    "g_capjoin": CAPJOIN, "g_basis": STREAMONLY,
    "g_graphinvis": GRAPHINVIS,
}


def consistent(w):
    """Physical consistency, REWRITTEN FOR BOOT SEPARATION (audit B1').

    rev4's model said `l1 == "zero" => l2 == l3 == l2_post == "zero"` -- "nsys
    saw nothing at all" -- because all four legs shared one report.  ★Across
    boots that implication is false: B-U can be empty while B-S is healthy, and
    a world where they differ is exactly the interesting one.  So NO
    cross-boot implication is asserted here.  What remains are WITHIN-boot
    facts and the few genuinely impossible combinations.

    Conservative: when in doubt the world is kept (A0's R9 discipline).
    """
    # a failed run exports nothing and has no rows anywhere
    if w.plumbing == "fail" and not (w.rows_s == w.rows_u == w.rows_g == "zero"
                                     and w.profile == "empty"):
        return False
    if w.plumbing == "boot_failed" and w.profile != "empty":
        return False
    # WITHIN B-S: no rows -> no fraction, no halves, no ids, no join
    if w.rows_s == "zero" and not (w.profile == "empty" and w.halves_s == "both"
                                   and w.ctx_s == "null"
                                   and w.stream_single == "no"
                                   and w.join_target == "none"):
        return False
    if w.rows_s == "ok" and w.profile == "empty":
        return False
    # eager replay has no graph launch to join to
    if w.profile == EAGER_ONLY and w.join_target != "none":
        return False
    if w.join_target == "none" and w.join_rate != "low":
        return False
    # ★P10 (re-audit): this exclusion is RETRACTED.  It said "no green context
    # anywhere means nsys cannot report a distinct greenContextId", but the two
    # sides are not the same observation: `green_s` is a ONE-SHOT driver
    # read-out at `init_pdmux`, and `ctx_s` is a property of the whole trace.
    # A boot where the read-out fired before the context existed, or asked the
    # wrong stream, is physically possible and is EXACTLY the world worth
    # catching.  The file's own R9 discipline says keep the world when in
    # doubt, and the mirror case (`ctx null/zero` + `green absent`) already had
    # a label -- so the disagreement space was being cut on one side only.
    # It is a label now, not an exclusion; see `g_contra`.
    # ★the kernel-name channel is a property of the TRACE, so it cannot be
    # answered when there is no trace in B-S
    if w.rows_s == "zero" and w.stream_disjoint != "unanswerable":
        return False
    # B-U's quality gates presuppose B-U produced something
    if w.rows_u == "zero" and w.bu_quality != "ok":
        return False
    return True


# ★★REGISTERED STOPS -- the design's sec 5 table, transcribed ONCE as data.
#
# WHY (re-audit P1).  The Q3 rule got this and this rule did not, and the
# re-audit showed exactly what that costs: apply the audit's own both-copies
# specification edit -- regress the C5 repair in `score()` AND in every check
# that reads it -- and all six checks pass while `plumbing == "partial_s"`
# comes back KSET_CONSTRUCTIBLE.  `T25` cannot see it either, because
# `g_trunc` still produces TRACE_TRUNCATED via `halves_s`, so the census does
# not move.  A registry is the only oracle that does not travel with the prose.
#
# ★LIMIT, stated here as it is in the Q3 rule: this raises a two-copy edit to a
# three-copy edit and makes the third one a visible change to what was
# REGISTERED.  A determined three-copy edit still passes.  Saying otherwise
# would be the overclaim the audit punished.
_CLEAN_EMPTY = dict(profile="empty", ctx_s="null", stream_single="no",
                    stream_disjoint="unanswerable", join_target="none",
                    join_rate="low")
REGISTERED_STOPS = (
    # (kwargs from an otherwise-clean world, expected label)
    (dict(plumbing="boot_failed", rows_s="zero", rows_u="zero", rows_g="zero",
          **_CLEAN_EMPTY), ABSENT),
    (dict(plumbing="fail", rows_s="zero", rows_u="zero", rows_g="zero",
          **_CLEAN_EMPTY), ABSENT),
    (dict(plumbing="partial_s"), TRUNC),                       # ★the C5 stop
    (dict(plumbing="partial_ctl"), TRUNC),
    (dict(halves_s="one"), TRUNC),
    (dict(sticky_s="low"), NOSTICKY),
    (dict(sticky_s="nolog"), NOSTICKY),
    (dict(bu_quality="unsplit_low"), UNSPLIT),
    (dict(bu_quality="expect_ambiguous"), EXPAMBIG),
    (dict(green_s="mismatched"), GREENBAD),
    (dict(stream_disjoint="unanswerable"), NOLEG),
    (dict(profile=EAGER_ONLY, join_target="none", join_rate="low"), EAGERONLY),
    (dict(rows_s="zero", rows_g="ok", **_CLEAN_EMPTY), GRAPHINVIS),   # ★P2
    (dict(rows_s="zero", rows_u="ok", rows_g="zero", **_CLEAN_EMPTY), GREENINVIS),
    (dict(rows_s="zero", rows_u="zero", rows_g="zero", **_CLEAN_EMPTY),
     NODEUNAVAIL),
    (dict(profile="just_below"), UNCONSTR),
    (dict(ctx_s="parent"), CONTRA),
    (dict(green_s="absent", ctx_s="distinct"), CONTRA),          # ★P10
    (dict(ctx_s="null", green_s="absent"), DISCONF),
    (dict(join_target="capture_only"), CAPJOIN),
    (dict(join_rate="low"), NOATTR),
    (dict(stream_disjoint="no"), STREAMONLY),
)


def worlds():
    keys = list(AXES)
    for combo in product(*(AXES[k] for k in keys)):
        yield World(**dict(zip(keys, combo)))


# =============================================================================
# CHECKS -- different cross-sections, per the rev2 audit's D2/B10 finding that
# three transcriptions of one branch order carry one check's worth of power.
# =============================================================================
def _checks():
    def c_tiers(w, l, b):
        """A. TIER DISCIPLINE: a plumbing fault never reaches a substantive
        label, and a tool limit never reaches one either.  Reads the axes, not
        the branch order."""
        broken = (w.plumbing != "ok" or w.halves_s != "both"
                  or w.sticky_s != "ok" or w.bu_quality != "ok"
                  or w.green_s == "mismatched"
                  or (w.rows_s == "ok" and w.stream_disjoint == "unanswerable"))
        if broken:
            return l in MEASUREMENT
        return l not in MEASUREMENT

    def c_basis(w, l, b):
        """B. BASIS: the primary label exists only on the two-channel basis, and
        the basis string never contradicts the label."""
        if l == OK:
            return b == "ctx+disjoint" and w.ctx_s == "distinct" \
                and w.stream_disjoint == "yes"
        if l == STREAMONLY:
            return b == "ctx_only"
        return b is None

    def c_toollimit(w, l, b):
        """C. TOOL LIMIT: 'nsys saw no node rows anywhere' is a statement about
        the tool and must be labelled as one -- never as an answer about the
        estimand.  ★This is the cross-section the Q3 rule was missing (B2)."""
        if (w.plumbing == "ok" and w.halves_s == "both" and w.sticky_s == "ok"
                and w.bu_quality == "ok" and w.green_s != "mismatched"
                and w.rows_s == "zero"):
            # ★P2 widened the guarded region: ANY world where the condition
            # boot is empty and the plumbing is clean is a statement about the
            # tool, whatever the controls did.
            return l in TOOLLIMIT
        return True

    def c_q1(w, l, b):
        """D. THRESHOLD: below the registered fraction the estimand is not
        constructible, stated with the literal."""
        if (w.plumbing == "ok" and w.halves_s == "both" and w.sticky_s == "ok"
                and w.bu_quality == "ok" and w.green_s != "mismatched"
                and w.stream_disjoint != "unanswerable" and w.rows_s == "ok"
                and w.profile not in ("empty", EAGER_ONLY)):
            if w.frac < 0.90:
                return l == UNCONSTR
        return True

    def c_witness(w, l, b):
        """E. WITNESS: every substantive label carries a NECESSARY condition in
        the axes.  One-directional on purpose -- an iff here would be the
        transcription the rev2 audit's B10 showed cannot catch a spec error."""
        if l == CAPJOIN:
            return w.join_target == "capture_only"
        if l == CONTRA:
            return ((w.ctx_s == "parent" and w.stream_disjoint == "yes")
                    or (w.green_s == "absent" and w.ctx_s == "distinct"))
        if l == DISCONF:
            return w.ctx_s in ("null", "zero") and w.green_s == "absent"
        if l == UNCONSTR:
            # ★P2: `rows_s == "zero"` is no longer a route to UNCONSTR -- it is
            # a tool limit.  Only the fraction can make the estimand
            # unconstructible now.
            return w.profile == "empty" or w.frac < 0.90
        if l == NOATTR:
            return (w.ctx_s != "distinct" or w.join_target == "none"
                    or w.join_rate == "low" or w.stream_single != "yes")
        if l == EAGERONLY:
            return w.profile == EAGER_ONLY
        if l == OK:
            # ★the join belongs HERE and not only in the basis string: without
            # it, dropping the capture-only or the join-rate guard produces a
            # KSET_CONSTRUCTIBLE whose basis string is still well formed, and
            # check B is satisfied by a world where nothing joined to a replay.
            return (w.join_target == "replay" and w.join_rate == "high"
                    and w.ctx_s == "distinct" and w.stream_disjoint == "yes"
                    and w.stream_single == "yes"
                    and w.profile not in ("empty", EAGER_ONLY)
                    and w.frac >= 0.90)
        return True

    def c_absent(w, l, b):
        """F. A boot that never happened is ABSENT and nothing else.  Narrow by
        design: it restates ONE registered fact, and it exists because check A
        only asks for "some measurement label", which any of six guards can
        satisfy -- so A alone never notices this one going missing."""
        if w.plumbing in ("boot_failed", "fail"):
            return l == ABSENT
        return True

    return {
        "E every substantive label has a witness": (c_witness, {
            # `g_eager` is NOT here: removing it lets an eager-only profile
            # fall through to the Q1 threshold, which returns UNCONSTR -- a
            # label whose witness is satisfied (frac 0.0 < 0.90).  E cannot see
            # it; T25 does, because EAGER_REPLAY_ONLY stops appearing.
            "g_capjoin", "g_joinrate", "g_single", "g_q1"}),
        "F a boot that never happened is ABSENT": (c_absent, {"g_absent"}),
        "A tier discipline (plumbing => measurement)": (c_tiers, {
            "g_absent", "g_trunc", "g_sticky", "g_unsplit", "g_expambig",
            "g_green", "g_leg"}),
        # `g_single` is NOT here: without that guard, OK appears with a
        # well-formed basis string, so B is satisfied.  E's OK witness is what
        # rejects it, because that witness carries `stream_single == "yes"`.
        "B basis is two-channel and matches the label": (c_basis, {
            "g_basis", "g_noattr", "g_basis_string"}),
        # `g_greeninvis` is NOT here: C's guarded region is "no rows in ANY
        # boot", and GREENCTX_INVISIBLE lives in "controls alive, condition
        # empty" -- outside it.  T25 covers that mutant.
        "C tool limits are labelled as tool limits": (c_toollimit, {
            "g_nodeunavail", "g_graphinvis"}),
        "D Q1_FRAC = 0.90 (literal)": (c_q1, {"g_q1", "g_q1_value"}),
    }


def _battery(out):
    W = [w for w in worlds() if consistent(w)]
    n_all = 1
    for v in AXES.values():
        n_all *= len(v)
    base = [score(w) for w in W]
    checks = _checks()
    fails = []

    def chk(n, c, d=""):
        print(f"  [{'PASS' if c else 'FAIL'}] {n} {d}")
        out[n] = bool(c)
        if not c:
            fails.append(n)

    chk("T1 totality", all(l in LABELS and b in BASES for l, b in base))
    chk("T1b determinism", base == [score(w) for w in W])
    cnt = Counter(l for l, _ in base)
    for lab in sorted(LABELS):
        chk(f"T2 reachable in consistent worlds: {lab}", cnt[lab] > 0,
            f"n={cnt[lab]}")
    mut = {m: [score(w, {m}) for w in W] for m in MUTANTS}
    for m in MUTANTS:
        d = sum(1 for a, b in zip(base, mut[m]) if a != b)
        chk(f"T3 mutant {m} load-bearing", d > 0, f"n={d}")
    for n, (fn, _) in checks.items():
        chk(f"{n} [DISCRIM]", all(fn(w, l, b) for w, (l, b) in zip(W, base)))
    print("  -- T8 meta (each check fails under its named mutants) --")
    for n, (fn, ms) in checks.items():
        for m in sorted(ms):
            chk(f"T8 '{n[:26]}' fails under {m}",
                any(not fn(w, l, b) for w, (l, b) in zip(W, mut[m])))
    print("  -- T10 meta (every mutant caught by some check) --")
    unc = [m for m in MUTANTS
           if not any(any(not fn(w, l, b) for w, (l, b) in zip(W, mut[m]))
                      for fn, _ in checks.values())]
    # ★T25 is a check, just not a per-assignment one, and the removal mutants
    # are exactly what it exists for.  Report the per-assignment fact as
    # INFORMATION and assert coverage against BOTH layers -- hiding the first
    # number would make the second look stronger than it is.
    t25_covers = {g for g, lab in GUARD_LABEL.items()
                  if Counter(l for l, _ in mut[g])[lab] < cnt[lab]}
    print(f"     not caught per-assignment: {unc or 'none'}")
    print(f"     of those, caught by T25:   "
          f"{[m for m in unc if m in t25_covers] or 'none'}")
    residual = [m for m in unc if m not in t25_covers]
    chk("T10 every mutant covered (per-assignment checks + T25)",
        not residual, f"uncovered={residual or 'none'}")
    print("  -- T19 meta (entailment / sole-binding) --")
    ASG = [(w, score(w, frozenset({m}) if m else frozenset()))
           for m in [None] + MUTANTS for w in W]
    fs = {n: {i for i, (w, (l, b)) in enumerate(ASG) if not fn(w, l, b)}
          for n, (fn, _) in checks.items()}
    ent = [f"{a}=>{b}" for a in checks for b in checks
           if a != b and fs[b] and fs[b] <= fs[a]]
    chk("T19a no check entailed by another", not ent, f"{ent or 'none'}")
    sole = {n: len(fs[n] - set().union(*[fs[o] for o in checks if o != n]))
            for n in checks}
    def _covered(subset):
        return [m for m in MUTANTS
                if not any(any(not checks[n][0](w, l, b) for w, (l, b) in zip(W, mut[m]))
                           for n in subset)]
    needed = {n: bool(_covered([o for o in checks if o != n])) for n in checks}
    dead = [n for n in checks if sole[n] == 0 and not needed[n]]
    print(f"     sole-binding: { {n[:12]: sole[n] for n in checks} }")
    print(f"     required-for-coverage: { {n[:12]: needed[n] for n in checks} }")
    chk("T19b no check is dead weight", not dead, f"{dead or 'none'}")
    # ★T26: each registered stop, exercised one axis at a time from an
    # otherwise-clean world.  The oracle is a declaration of what the DESIGN
    # registered, not a re-derivation of the rule -- which is what makes it
    # survive a both-copies specification edit.  See REGISTERED_STOPS for how
    # far that goes and exactly where it stops.
    print("  -- T26 meta (every registered stop fires on its own) --")
    for kwargs, want in REGISTERED_STOPS:
        got = score(World(**kwargs))[0]
        chk(f"T26 {sorted(kwargs.items())[:2]} -> {want}", got == want,
            f"got {got}")

    # ★T25: a NECESSARY-CONDITION check (E) constrains worlds that DO carry a
    # label; it is structurally blind to a label that stopped appearing.  The
    # first run of this suite showed that directly -- E was named on nine
    # removal mutants and caught none of them, because removing a guard makes
    # its label vacuously absent.  Sufficiency would fix it and would be the
    # whole-branch transcription the rev2 audit's B10 showed cannot catch a
    # specification error.  So removal is tested where it lives: at the label
    # census.
    print("  -- T25 meta (each guard's own label disappears without it) --")
    for g, lab in sorted(GUARD_LABEL.items()):
        before = cnt[lab]
        after = Counter(l for l, _ in mut[g])[lab]
        chk(f"T25 {g} is what produces {lab}", after < before,
            f"{before} -> {after}")

    # ★T24: no cross-boot implication is asserted (the B1' repair itself).
    print("  -- T24 meta (the boot-separation repair) --")
    cross = [w for w in worlds()
             if w.plumbing == "ok" and w.rows_u == "zero" and w.rows_s == "ok"
             and consistent(w)]
    chk("T24 a healthy B-S with an EMPTY B-U is a consistent world",
        len(cross) > 0, f"n={len(cross)}")
    cross2 = [w for w in worlds()
              if w.plumbing == "ok" and w.rows_g == "zero" and w.rows_s == "ok"
              and consistent(w)]
    chk("T24b a healthy B-S with an EMPTY B-G is a consistent world",
        len(cross2) > 0, f"n={len(cross2)}")
    return fails, len(W), n_all, dict(cnt), sole


def run():
    print(f"== A1 primary rule (RULE_REV={RULE_REV}, derived from A0 rev4) ==")
    res = {}
    fails, n_consistent, n_all, cnt, sole = _battery(res)
    ok = not fails
    print(f"\n== {n_all:,} worlds enumerated / {n_consistent:,} consistent -> "
          f"{'ALL PASS' if ok else str(len(fails)) + ' FAILURES: ' + '; '.join(fails[:4])} ==")
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.abspath(__file__), "rb") as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()
    with open(os.path.join(here, "selftest_a1_primary_2026-08-25.json"), "w") as fh:
        json.dump({"rule_rev": RULE_REV, "rule_sha256": sha,
                   "derived_from_a0_rev4_sha256": DERIVED_FROM_SHA,
                   "date": "2026-08-25", "gpu_hr": 0.0,
                   "worlds_enumerated": n_all, "worlds_consistent": n_consistent,
                   "labels": cnt, "sole_binding": sole,
                   "q1_frac": Q1_FRAC, "checks": res, "all_pass": ok}, fh,
                  indent=1, sort_keys=True)
    print("   wrote selftest_a1_primary_2026-08-25.json")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(run())
