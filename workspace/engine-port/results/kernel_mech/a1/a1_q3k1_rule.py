#!/usr/bin/env python3
"""A1 **rev3** -- the new decision rules: Q3 (step-boundary separability) and K1.

rev2 was audited `NO-GO` (`audit_a1_rev2_rules_2026-08-25`).  Its Q3 half had one
runkiller: **B2, gate #21 reopened**.

    launches == "zero"  -> BOUNDARY_NOT_SEPARABLE
    launches == "more"  -> BOUNDARY_NOT_SEPARABLE

`more` is the value rev2 ITSELF registered as *"the row population is not the
decode replays alone ... the denominator is wrong"*, and `zero` is a silent
eager fallback.  Both are facts about the measurement, and both were scored as
answers about the tool -- "the boundary is not separable".  A0 avoided that with
a THREE-tier label space (MEASUREMENT / TOOLLIMIT / SUBSTANTIVE) and rev2
collapsed the middle one.  ★And the impurity is not hypothetical:
`cuda_graph_runner.py:806-816` captures for EVERY stream-group index, so any
report contains capture activity from indices other than the one under test.

rev3 restores the three tiers, and takes the rest of the audit's Q3 list:

  B6  `disjoint` is its own axis.  rev2's non-circularity argument rested
      entirely on "the decode stream carries no prefill-only kernel", decided
      by KERNEL NAMES -- and that predicate had no axis, so it could not be
      enumerated.  What was left, `single`, is near-identity with
      `ctx == distinct` (gate #9), which is why the argument was hollow.
  B9  `UNSPLIT_NOT_REALIZED` and `EXPECTATION_AMBIGUOUS` are code, and a
      **`role` axis** exists so B-U can be scored at all.  rev2 had no way to
      say which boot a world described: feed it the unsplit reference boot and
      it answered `Q3_STICKY_NOT_REALIZED`.
  B10 a check that is not a transcription of `q3_score`'s branch order, plus
      the census meta-check (`T25`) that catches what a necessary condition
      cannot see.

Repairs carried over from rev2 (audited CONFIRMED, do not undo):
  C1  `N_MIN_DECODE_STEPS` is compared against an INTEGER axis and
      `g_nmin_value` proves it -- 5/40/63/65/128/200 all break the self-test.
  C2  the unit is decode STEPS (`Delta decode_iterations`), never snapshots.
  C5  `export == "partial"` is a MEASUREMENT condition.
  D7  `launches == "more"` keeps its registered meaning -- but now that meaning
      decides its TIER instead of contradicting it.
  D8  `dropped_events` is an axis.

Run:  python3 a1_q3k1_rule.py
"""
import hashlib, json, os, sys
from collections import Counter
from itertools import product

RULE_REV = 3

# --- Q3 tier 1: MEASUREMENT --------------------------------------------------
Q3_BOOT     = "Q3_BOOT_FAILED"
Q3_TRUNC    = "Q3_TRACE_TRUNCATED"
Q3_NOTEL    = "Q3_TELEMETRY_ABSENT"
Q3_NOSTICKY = "Q3_STICKY_NOT_REALIZED"       # B-S only
Q3_UNSPLIT  = "Q3_UNSPLIT_NOT_REALIZED"      # ★B-U only (audit B9)
Q3_EXPAMBIG = "Q3_EXPECTATION_AMBIGUOUS"     # ★B-U only (audit B9)
Q3_ZEROCH   = "Q3_CHANNEL_ZERO"
Q3_SHORT    = "Q3_SPAN_TOO_SHORT"
Q3_NOLEG    = "Q3_LEG_LABELLING_UNAVAILABLE"
Q3_MEASUREMENT_LABELS = {Q3_BOOT, Q3_TRUNC, Q3_NOTEL, Q3_NOSTICKY, Q3_UNSPLIT,
                         Q3_EXPAMBIG, Q3_ZEROCH, Q3_SHORT, Q3_NOLEG}

# --- Q3 tier 2: TOOLLIMIT ★NEW (the B2 repair) -------------------------------
Q3_NOLAUNCH = "Q3_LAUNCH_ROWS_ABSENT"        # `zero`: nothing to count
Q3_IMPURE   = "Q3_LAUNCH_POPULATION_IMPURE"  # `more`: the denominator is wrong
Q3_EAGER    = "Q3_EAGER_FALLBACK"            # the boot replayed nothing
# ★NEW (re-audit P12): `fewer` is the exact mirror of `more`.  If MORE rows than
# steps means the denominator is wrong, FEWER means the numerator is short --
# some decode steps produced no graph launch (a partial eager fallback for
# uncaptured batch sizes, or rows lost in the trace).  Both are facts about the
# measurement.  What is left in SUBSTANTIVE is `partition == "overlap"`, which
# really is an answer about separability.
Q3_SHORTLAUNCH = "Q3_LAUNCH_ROWS_SHORT"
Q3_TOOLLIMIT_LABELS = {Q3_NOLAUNCH, Q3_IMPURE, Q3_EAGER, Q3_SHORTLAUNCH}

# --- Q3 tier 3: SUBSTANTIVE --------------------------------------------------
Q3_STREAMAMB = "Q3_STREAM_AMBIGUOUS"
Q3_NOSEP     = "BOUNDARY_NOT_SEPARABLE"
Q3_PARTIAL   = "BOUNDARY_SEPARABLE_PARTIAL"
Q3_SEP       = "BOUNDARY_SEPARABLE"
Q3_SUBSTANTIVE_LABELS = {Q3_STREAMAMB, Q3_NOSEP, Q3_PARTIAL, Q3_SEP}

Q3_LABELS = (Q3_MEASUREMENT_LABELS | Q3_TOOLLIMIT_LABELS
             | Q3_SUBSTANTIVE_LABELS)

# --- K1 labels ---------------------------------------------------------------
K1_UNMEAS   = "K1_UNMEASURED"
K1_DEADCH   = "K1_CHANNEL_DEAD"
K1_PAIRBAD  = "K1_PAIR_INVALID"
K1_NEAR1    = "K1_NEAR_1"
K1_MOD      = "K1_MODERATE"
K1_HIGH     = "K1_HIGH"
K1_PROHIB   = "K1_PROHIBITIVE"
K1_LABELS = {K1_UNMEAS, K1_DEADCH, K1_PAIRBAD, K1_NEAR1, K1_MOD, K1_HIGH,
             K1_PROHIB}

# --- registered constants ----------------------------------------------------
# ★UNIT = decode steps, as `Delta decode_iterations` over the boot.  NOT
#   snapshots (C2): pooled over 8 boots only 2.3% of benchmark snapshot pairs
#   advance the counter and the mean is 0.156-0.409 steps per snapshot.
#   Value: half of one batch-synchronous round (out=128).  ★[ARBITRARY] as a
#   number; what is not arbitrary is that it is compared against an integer and
#   that `g_nmin_value` proves the comparison is live.
N_MIN_DECODE_STEPS = 64

K1_MOD_AT    = 1.10
K1_HIGH_AT   = 2.00   # ★[ARBITRARY -- provenance retracted, audit D5]
K1_PROHIB_AT = 10.00  # ★[ARBITRARY] -- the design's fourth band

# ★K1_CHANNEL -- registered 2026-08-25 from code + existing telemetry (GPU 0):
#   PRIMARY   per-step wall time = boot duration / `Delta decode_iterations`.
#             Neither term passes the outlier filter below.
#   SECONDARY `measured_itl_p95_ms` (p95 of <=128 raw per-iteration samples,
#             `src/multiplex/multiplexing_mixin.py:504-509`).
#   DEAD      `decode_last_tpot_ms` -- behind the `dual_worker_enabled` gate
#             that kills `decode_step_count`.  0 / 294,506 snapshots.
#   ★LIVENESS IS CONDITIONAL on `_slo_on` (`src/multiplex/multiplexing_mixin.py:1256`): measured non-zero in 4/4
#   s2_sticky boots and zero in 4/4 g16 boots.  A1 sets PDMUX_R2_POLICY=fixed,
#   so it is live -- a boot that omits it gets K1_CHANNEL_DEAD.
#   ★CLIPPING HAZARD: samples outside (0, max(3*EMA, 90ms)) are rejected
#   (`src/multiplex/multiplexing_mixin.py:1274-1276`), which biases the ITL
#   channel DOWNWARD exactly when overhead is
#   large.  That is why PRIMARY is the counter, not the ITL field.


class Q3World:
    """One boot's Q3 outcome.

    role        ★NEW (audit B9): "B_S" is the condition boot (sticky ON) and
                "B_U" is the unsplit reference.  rev2 had no such axis, so
                B-U was unscorable -- it answered STICKY_NOT_REALIZED.
    realized    the boot's own realization gate: for B_S "did sticky hold the
                division", for B_U "did decode stay unpartitioned".  Same three
                values, different meaning per role -- which is why `role` has
                to exist for the gate to be well posed.
    expect      B-U only: is the node-count-per-replay distribution unimodal.
    steps       ★INTEGER.  Delta decode_iterations over the boot.
    launches    graph-launch rows vs `steps`, at RUN level:
                  zero  : none found -> TOOL LIMIT, not an answer
                  fewer : fewer rows than steps -- some steps unlaunched
                  equal : 1:1, the thing Q3 asks about
                  more  : MORE rows than steps.  Registered meaning (D7): the
                          population is not the decode replays alone -- capture
                          activity for other stream-group indices is captured
                          by construction.  ⇒ the DENOMINATOR is wrong ⇒ TOOL
                          LIMIT.  rev2 scored this BOUNDARY_NOT_SEPARABLE.
    eager       the boot replayed nothing (cudagraph off / fallback)
    stream_single    all rows on ONE streamId
    stream_disjoint  ★NEW axis (audit B6): that stream carries no prefill-only
                     kernel -- yes | no | unanswerable.  Decided by KERNEL
                     NAMES.  This is where the non-circularity actually lives;
                     `single` is near-identity with `ctx == distinct`.
    partition   node rows partition by launch correlationId
    """

    __slots__ = ("role", "boot_ok", "export", "dropped", "tel", "realized",
                 "expect", "steps", "halves", "launches", "eager",
                 "stream_single", "stream_disjoint", "partition")

    def __init__(self, role="B_S", boot_ok=True, export="ok", dropped=0,
                 tel="ok", realized="ok", expect="unimodal", steps=3802,
                 halves="both", launches="equal", eager="no",
                 stream_single="yes", stream_disjoint="yes",
                 partition="clean"):
        self.role, self.boot_ok, self.export, self.dropped = role, boot_ok, export, dropped
        self.tel, self.realized, self.expect = tel, realized, expect
        self.steps, self.halves = steps, halves
        self.launches, self.eager = launches, eager
        self.stream_single, self.stream_disjoint = stream_single, stream_disjoint
        self.partition = partition

    def __repr__(self):
        return (f"Q3({self.role},boot={self.boot_ok},exp={self.export},"
                f"drop={self.dropped},tel={self.tel},real={self.realized},"
                f"exp2={self.expect},steps={self.steps},hv={self.halves},"
                f"lau={self.launches},eag={self.eager},sg={self.stream_single},"
                f"dj={self.stream_disjoint},part={self.partition})")


def q3_score(w, guards=frozenset()):
    """Boot-level, THREE TIERS, decided in order and never mixed.

    MEASUREMENT  the boot did not deliver an observation
    TOOLLIMIT    nsys/the capture path could not deliver a countable population
    SUBSTANTIVE  an answer about step-boundary separability
    """
    def on(g):
        return g not in guards

    # ---- tier 1: MEASUREMENT ------------------------------------------------
    if on("g_boot") and not w.boot_ok:
        return Q3_BOOT
    if on("g_trunc") and (w.export != "ok" or w.dropped > 0
                          or w.halves != "both"):
        return Q3_TRUNC
    if on("g_tel") and w.tel != "ok":
        return Q3_NOTEL
    # ★role-specific realization gates.  The same axis value means a different
    # thing per role, which is exactly why rev2 could not score B-U.
    if on("g_realized") and w.realized != "ok":
        # ADDITIVE MUTANT `g_role_confused`: answer the CONDITION boot's label
        # for every role.  ★This reproduces rev2's actual defect -- it had no
        # role axis, so the unsplit reference boot came back
        # STICKY_NOT_REALIZED -- which is what makes Q3-D load-bearing rather
        # than vacuous.  A removal mutant cannot exercise it: drop `g_realized`
        # and BOTH labels vanish, so the role check passes emptily.
        if "g_role_confused" in guards:
            return Q3_NOSTICKY
        return Q3_NOSTICKY if w.role == "B_S" else Q3_UNSPLIT
    if on("g_expambig") and w.role == "B_U" and w.expect != "unimodal":
        return Q3_EXPAMBIG
    if on("g_zero") and w.steps <= 0:
        return Q3_ZEROCH
    nmin = N_MIN_DECODE_STEPS if on("g_nmin_value") else 5
    if on("g_span") and "g_order_span_last" not in guards and w.steps < nmin:
        return Q3_SHORT
    # ★`launches != "zero"` is REQUIRED here.  With no rows at all, disjointness
    # is unanswerable BECAUSE THERE IS NOTHING TO LABEL -- a tool limit, not a
    # leg-labelling failure.  The A1 primary rule hit exactly this and its T2
    # reported two tool-limit labels unreachable.
    if on("g_leg") and w.launches != "zero" and w.stream_disjoint == "unanswerable":
        return Q3_NOLEG

    # ---- tier 2: TOOLLIMIT ★the B2 repair -----------------------------------
    if on("g_eager") and w.eager == "yes":
        return Q3_EAGER
    if on("g_nolaunch") and w.launches == "zero":
        return Q3_NOLAUNCH
    if on("g_impure") and w.launches == "more":
        return Q3_IMPURE
    if on("g_shortlaunch") and w.launches == "fewer":
        return Q3_SHORTLAUNCH

    # ---- tier 3: SUBSTANTIVE ------------------------------------------------
    if on("g_stream") and w.stream_single != "yes":
        return Q3_STREAMAMB
    if on("g_nosep") and w.partition == "overlap":
        return Q3_NOSEP
    if on("g_partial") and w.partition == "orphans":
        return Q3_PARTIAL
    if "g_order_span_last" in guards and w.steps < N_MIN_DECODE_STEPS:
        return Q3_SHORT
    # ★`single` alone is near-identity with the green-context id; the extra
    # channel is that the stream carries no prefill-only kernel, decided by
    # kernel names.
    # ★SCOPE (re-audit P9): under sticky with FixedPolicy, prefill and decode
    # sit on the two green streams of ONE stream group, so that separation is
    # ARRANGED BY THE ENGINE -- as A0's `stream == "match"` was arranged by
    # free parameter #15.  ⇒ `disjoint` measures whether nsys' `streamId`
    # column is FAITHFUL to the engine's separation (tool fidelity), NOT that
    # the green-context attribution is correct.  Do not write the stronger
    # sentence; the design's forbidden list carries the allowed reduced form.
    if on("g_disjoint") and w.stream_disjoint != "yes":
        return Q3_STREAMAMB
    return Q3_SEP


Q3_AXES = dict(
    role=["B_S", "B_U"],
    boot_ok=[True, False],
    export=["ok", "partial", "fail"],
    dropped=[0, 17],
    tel=["ok", "absent"],
    realized=["ok", "low", "nolog"],
    expect=["unimodal", "ambiguous"],
    steps=[0, 4, 63, 64, 3802],
    halves=["both", "one"],
    launches=["zero", "fewer", "equal", "more"],
    eager=["no", "yes"],
    stream_single=["yes", "no"],
    stream_disjoint=["yes", "no", "unanswerable"],
    partition=["clean", "orphans", "overlap"],
)
Q3_MUTANTS = ["g_boot", "g_trunc", "g_tel", "g_realized", "g_expambig",
              "g_zero", "g_span", "g_nmin_value", "g_leg", "g_eager",
              "g_nolaunch", "g_impure", "g_shortlaunch", "g_stream", "g_nosep",
              "g_partial",
              "g_disjoint", "g_order_span_last", "g_role_confused"]

# ★which guard is registered as producing which label.  T25 uses this: a
# necessary-condition check is structurally blind to a label that stopped
# appearing (the A1 primary rule demonstrated that on nine mutants at once).
Q3_GUARD_LABEL = {
    "g_boot": Q3_BOOT, "g_trunc": Q3_TRUNC, "g_tel": Q3_NOTEL,
    "g_expambig": Q3_EXPAMBIG, "g_zero": Q3_ZEROCH, "g_span": Q3_SHORT,
    "g_leg": Q3_NOLEG, "g_eager": Q3_EAGER, "g_nolaunch": Q3_NOLAUNCH,
    "g_impure": Q3_IMPURE, "g_shortlaunch": Q3_SHORTLAUNCH,
    "g_nosep": Q3_NOSEP, "g_partial": Q3_PARTIAL,
}


# ★★REGISTERED STOPS -- the design's sec 5 table transcribed ONCE, as data.
#
# WHY THIS EXISTS.  The rev2 audit's B10 showed that editing a specification in
# BOTH the rule and its check passes everything, and I verified the same
# mutation against rev3's redesigned checks: regressing the C5 repair in both
# copies (`export != "ok"` -> `export == "fail"`, in `q3_score` and in
# `_q3_plumbing_dirty`) STILL PASSES.  Three checks reading different
# cross-sections did not help, because check A consumes the same helper whose
# meaning was edited.  ★That is a limit of self-checking, not of check design:
# any oracle the same author derives from the same prose can be moved with it.
#
# What this table changes is not soundness but LEGIBILITY.  The oracle is now a
# declaration of what the DESIGN registered, keyed by (axis, value) -> label.
# A both-copies edit now needs a third edit, here, in a file whose only purpose
# is to state the registered stops -- so the edit becomes a visible act of
# changing the registration rather than an invisible change of behaviour.
# ★It does NOT make the suite proof against a determined three-copy edit, and
# claiming otherwise would be the same overclaim the audit just punished.
REGISTERED_STOPS = (
    # (role, axis, value, expected label)          design sec 5 row
    ("B_S", "boot_ok", False, Q3_BOOT),                       # 1
    ("B_S", "export", "partial", Q3_TRUNC),                   # 6  (C5)
    ("B_S", "export", "fail", Q3_TRUNC),                      # 6
    ("B_S", "dropped", 17, Q3_TRUNC),                         # 6  (D8)
    ("B_S", "halves", "one", Q3_TRUNC),                       # 6  (sec 2.1)
    ("B_S", "tel", "absent", Q3_NOTEL),
    ("B_S", "realized", "low", Q3_NOSTICKY),                  # 2
    ("B_S", "realized", "nolog", Q3_NOSTICKY),                # 2
    ("B_U", "realized", "low", Q3_UNSPLIT),                   # 3  (B9)
    ("B_U", "expect", "ambiguous", Q3_EXPAMBIG),              # 5  (B9)
    ("B_S", "steps", 0, Q3_ZEROCH),                           # 8
    ("B_S", "steps", 4, Q3_SHORT),                            # 4
    ("B_S", "stream_disjoint", "unanswerable", Q3_NOLEG),     # 7  (B6)
    ("B_S", "eager", "yes", Q3_EAGER),                        # ★B2
    ("B_S", "launches", "zero", Q3_NOLAUNCH),                 # ★B2
    ("B_S", "launches", "more", Q3_IMPURE),                   # ★B2
    ("B_S", "launches", "fewer", Q3_SHORTLAUNCH),             # ★P12
)


def q3_worlds():
    """★NOTE (re-audit P14): Q3 has **no** `consistent()` model, unlike the
    primary rule.  So its world count is the size of the AXIS PRODUCT, not the
    number of physically possible worlds, and its sole-binding numbers are NOT
    comparable with the primary rule's.  Enumerating impossible worlds is the
    conservative direction -- it can only make a check look weaker -- but the
    number must not be quoted as "possible worlds"."""
    keys = list(Q3_AXES)
    for c in product(*(Q3_AXES[k] for k in keys)):
        yield Q3World(**dict(zip(keys, c)))


def _q3_plumbing_dirty(w):
    """The registered MEASUREMENT conditions as a predicate, not a branch
    order -- this is what the tier check reads."""
    return (not w.boot_ok or w.export != "ok" or w.dropped > 0
            or w.halves != "both" or w.tel != "ok" or w.realized != "ok"
            or (w.role == "B_U" and w.expect != "unimodal")
            or w.steps < N_MIN_DECODE_STEPS
            or (w.launches != "zero" and w.stream_disjoint == "unanswerable"))


def _q3_toollimited(w):
    return w.eager == "yes" or w.launches in ("zero", "more", "fewer")


def _q3_checks():
    """Different CROSS-SECTIONS, not three transcriptions of one branch order.

    ★rev2's three checks were all literal restatements of `q3_score`, and the
    audit showed what that buys: changing a specification in BOTH copies -- the
    rule and the check -- passed everything, including a regression of the C5
    repair.  A transcription catches typos and cannot catch a specification
    error.  So:

      A  tier discipline : dirty plumbing => a MEASUREMENT label; tool-limited
                           => a TOOLLIMIT label.  Says nothing about WHICH.
      B  witness         : every SUBSTANTIVE label carries a necessary
                           condition in the axes.  One-directional on purpose.
      C  threshold       : SPAN_TOO_SHORT holds exactly on 0 < steps < 64.
      D  role            : ★a label that belongs to one boot role never appears
                           under the other.  A cross-section no branch-order
                           transcription contains, and the one that would have
                           caught rev2's B-U defect.

    Removal mutants are covered by T25, not by these -- see the A1 primary rule
    for why a necessary condition cannot see a label that stopped appearing.
    """
    def c_tiers(w, l):
        if _q3_plumbing_dirty(w):
            return l in Q3_MEASUREMENT_LABELS
        if _q3_toollimited(w):
            return l in Q3_TOOLLIMIT_LABELS
        return l in Q3_SUBSTANTIVE_LABELS

    def c_witness(w, l):
        if l == Q3_SEP:
            return (w.launches == "equal" and w.partition == "clean"
                    and w.stream_single == "yes" and w.stream_disjoint == "yes"
                    and not _q3_plumbing_dirty(w) and not _q3_toollimited(w))
        if l == Q3_PARTIAL:
            return w.partition == "orphans"
        if l == Q3_NOSEP:
            return w.partition == "overlap"
        if l == Q3_STREAMAMB:
            return w.stream_single != "yes" or w.stream_disjoint != "yes"
        if l == Q3_IMPURE:
            return w.launches == "more"
        if l == Q3_NOLAUNCH:
            return w.launches == "zero"
        if l == Q3_SHORTLAUNCH:
            return w.launches == "fewer"
        if l == Q3_EAGER:
            return w.eager == "yes"
        return True

    def c_threshold(w, l):
        if (not w.boot_ok or w.export != "ok" or w.dropped > 0
                or w.halves != "both" or w.tel != "ok" or w.realized != "ok"
                or (w.role == "B_U" and w.expect != "unimodal")):
            return True
        return (l == Q3_SHORT) == (0 < w.steps < 64)

    def c_role(w, l):
        """★D. ROLE: STICKY_NOT_REALIZED is a statement about the condition
        boot and UNSPLIT_NOT_REALIZED about the reference boot; neither may
        appear under the other role, and EXPECTATION_AMBIGUOUS is B-U's alone.
        rev2 had no role axis and answered STICKY_NOT_REALIZED for B-U."""
        if l == Q3_NOSTICKY:
            return w.role == "B_S"
        if l in (Q3_UNSPLIT, Q3_EXPAMBIG):
            return w.role == "B_U"
        return True

    return {
        "Q3-A tier discipline (plumbing/tool => that tier)": (c_tiers, {
            "g_boot", "g_trunc", "g_tel", "g_realized", "g_expambig", "g_span",
            "g_leg", "g_nmin_value", "g_eager", "g_nolaunch", "g_impure",
            "g_shortlaunch"}),
        "Q3-B every substantive label has a witness": (c_witness, {
            "g_stream", "g_disjoint"}),
        "Q3-C SPAN_TOO_SHORT iff 0 < steps < 64 (literal)": (c_threshold, {
            "g_span", "g_nmin_value", "g_order_span_last", "g_zero"}),
        "Q3-D role-specific labels stay in their role": (c_role, {
            "g_role_confused"}),
    }


# =============================================================================
# K1
# =============================================================================
class K1World:
    """nsys ON/OFF on ONE registered configuration -- a BOOT PAIR (audit D6).

    on_ok/off_ok : each leg produced a number
    pair         : matched | mismatched.  ★Registered: the two legs must differ
                   in nsys and nothing else (same model, D, workload, ctx,
                   rounds, mamba pool).  A mismatched pair is not a small
                   error, it is a different experiment.
    channel      : live | dead.  See K1_CHANNEL above -- dead when the boot
                   sets neither PDMUX_SLO_SCHED nor PDMUX_R2_POLICY, which is
                   what 4/4 g16 boots did.
    clipped      : no | suspect.  The ON leg's per-iteration samples may be
                   rejected by the (0, max(3*EMA, 90ms)) band, which biases the
                   ITL channel DOWNWARD exactly when overhead is large.
                   `suspect` forces the PRIMARY (per-step wall time) reading and
                   forbids quoting the ITL secondary.
    ratio        : per-step wall time, ON / OFF.
    """

    def __init__(self, on_ok=True, off_ok=True, pair="matched", channel="live",
                 clipped="no", ratio=1.0):
        self.on_ok, self.off_ok, self.pair = on_ok, off_ok, pair
        self.channel, self.clipped, self.ratio = channel, clipped, ratio

    def __repr__(self):
        return (f"K1(on={self.on_ok},off={self.off_ok},pair={self.pair},"
                f"ch={self.channel},clip={self.clipped},r={self.ratio})")


def k1_score(w, guards=frozenset()):
    def on(g):
        return g not in guards
    # ADDITIVE MUTANT (rev1's `g_order_*` pattern): a rule that let `clipped`
    # decide the label would be answering a measurement question by fiat.  The
    # first run of this suite flagged K1-c as dead weight precisely because
    # nothing could make it fail -- a vacuous check, the shape this project
    # keeps paying for.  This mutant is what makes it load-bearing.
    if "h_clip_verdict" in guards and w.clipped == "suspect":
        return K1_UNMEAS
    if on("h_measure") and not (w.on_ok and w.off_ok):
        return K1_UNMEAS
    if on("h_channel") and w.channel != "live":
        return K1_DEADCH
    if on("h_pair") and w.pair != "matched":
        return K1_PAIRBAD
    prohib = K1_PROHIB_AT if on("h_prohib_value") else 50.0
    hi = K1_HIGH_AT if on("h_hi_value") else 5.0
    mod = K1_MOD_AT if on("h_mod_value") else 1.5
    if on("h_prohib") and w.ratio >= prohib:
        return K1_PROHIB
    if on("h_hi") and w.ratio >= hi:
        return K1_HIGH
    if on("h_mod") and w.ratio >= mod:
        return K1_MOD
    return K1_NEAR1


K1_AXES = dict(
    on_ok=[True, False], off_ok=[True, False],
    pair=["matched", "mismatched"], channel=["live", "dead"],
    clipped=["no", "suspect"],
    # straddle all three registered constants from both sides
    ratio=[0.95, 1.09, 1.10, 1.50, 1.99, 2.00, 9.99, 10.00, 40.0],
)
K1_MUTANTS = ["h_measure", "h_channel", "h_pair", "h_mod", "h_hi", "h_prohib",
              "h_mod_value", "h_hi_value", "h_prohib_value", "h_clip_verdict"]


def k1_worlds():
    keys = list(K1_AXES)
    for c in product(*(K1_AXES[k] for k in keys)):
        yield K1World(**dict(zip(keys, c)))


def _k1_checks():
    def k_gates(w, l):
        """Cross-section 1: the three non-numeric gates and their PRECEDENCE."""
        if not (w.on_ok and w.off_ok):
            return l == K1_UNMEAS
        if w.channel != "live":
            return l == K1_DEADCH
        if w.pair != "matched":
            return l == K1_PAIRBAD
        return l not in {K1_UNMEAS, K1_DEADCH, K1_PAIRBAD}

    def k_bands(w, l):
        """Cross-section 2: the band edges as literals, on scorable worlds."""
        if not (w.on_ok and w.off_ok) or w.channel != "live" or w.pair != "matched":
            return True
        want = (K1_PROHIB if w.ratio >= 10.00 else
                K1_HIGH if w.ratio >= 2.00 else
                K1_MOD if w.ratio >= 1.10 else K1_NEAR1)
        return l == want

    return {
        "K1-a gates and their precedence": (k_gates, {"h_measure", "h_channel", "h_pair"}),
        "K1-b bands are exactly 1.10 / 2.00 / 10.00 (literals)":
            (k_bands, {"h_mod", "h_hi", "h_prohib", "h_mod_value", "h_hi_value",
                       "h_prohib_value"}),
    }


def _k1_flip_clipped(w):
    return K1World(w.on_ok, w.off_ok, w.pair, w.channel,
                   "no" if w.clipped == "suspect" else "suspect", w.ratio)


# ★INVARIANTS are properties of the scoring FUNCTION, not of a (world, label)
# pair, and this suite learned that the hard way in two runs.  Written as a
# per-assignment check, "`clipped` must not change the label" was first VACUOUS
# (no mutant could make it fail) and then, re-written to read the observed
# label, it swallowed every other check's failure set -- T19a reported it
# entailing both others and every sole-binding collapsed to 0.  Neither run
# tested the property.  It is tested here instead: score(w) == score(flip(w))
# under the base rule and under every mutant EXCEPT the ones registered as
# breaking it.
#   name -> (flip, breakers)
K1_INVARIANTS = {
    "`clipped` does not change the label": (_k1_flip_clipped, {"h_clip_verdict"}),
}


# =============================================================================
# SELF-TEST
# =============================================================================
def _battery(name, worlds, score, mutants, checks, labels, out, invariants=None,
             guard_label=None, registered_stops=None, world_cls=None):
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
    cnt = Counter(base)
    for lab in sorted(labels):
        chk(f"{name} T2 reachable: {lab}", cnt[lab] > 0, f"n={cnt[lab]}")
    mut = {m: [score(w, {m}) for w in W] for m in mutants}
    for m in mutants:
        d = sum(1 for a, b in zip(base, mut[m]) if a != b)
        chk(f"{name} T3 mutant {m} load-bearing", d > 0, f"n={d}")
    for n, (fn, _) in checks.items():
        chk(f"{name} {n} [DISCRIM]", all(fn(w, l) for w, l in zip(W, base)))
    print(f"  -- {name} T8 meta (each check fails under its named mutants) --")
    for n, (fn, ms) in checks.items():
        for m in sorted(ms):
            chk(f"{name} T8 '{n[:30]}' fails under {m}",
                any(not fn(w, l) for w, l in zip(W, mut[m])))
    print(f"  -- {name} T10 meta (every mutant caught by some check) --")
    unc = [m for m in mutants
           if not any(any(not fn(w, l) for w, l in zip(W, mut[m]))
                      for fn, _ in checks.values())]
    # ★T25 is a check too, just not a per-assignment one.  Print the raw
    # per-assignment fact and assert coverage over BOTH layers -- hiding the
    # first number would make the second look stronger than it is.
    t25_covers = {g for g, lab in (guard_label or {}).items()
                  if Counter(score(w, {g}) for w in W)[lab] < cnt[lab]}
    if unc:
        print(f"     not caught per-assignment: {unc}")
        print(f"     of those, caught by T25:   "
              f"{[m for m in unc if m in t25_covers] or 'none'}")
    residual = [m for m in unc if m not in t25_covers]
    chk(f"{name} T10 every mutant covered (checks + T25)", not residual,
        f"uncovered={residual or 'none'}")
    print(f"  -- {name} T19 meta (entailment / sole-binding) --")
    ASG = [(w, score(w, frozenset({m}) if m else frozenset()))
           for m in [None] + list(mutants) for w in W]
    fs = {n: {i for i, (w, l) in enumerate(ASG) if not fn(w, l)}
          for n, (fn, _) in checks.items()}
    ent = [f"{a}=>{b}" for a in checks for b in checks
           if a != b and fs[b] and fs[b] <= fs[a]]
    chk(f"{name} T19a no check entailed by another", not ent, f"{ent or 'none'}")
    # ★D3 CORRECTION.  rev1 wrote that sole binding is "structurally
    # unreachable" here.  That was FALSE and its own output disproved it -- the
    # audit measured sole = 954 / 21 / 7.  The criterion stays a disjunction
    # (sole binding OR required for mutant coverage) because a total
    # iff-partition can make sole binding scarce, but the claim of
    # impossibility is retracted and the raw numbers print either way.
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
    if registered_stops:
        # ★T26: each registered stop, exercised one axis at a time from an
        # otherwise-clean world.  This is the check that survives a both-copies
        # specification edit -- see REGISTERED_STOPS for exactly how far that
        # goes and where it stops.
        print(f"  -- {name} T26 meta (every registered stop fires on its own) --")
        for role, axis, value, want in registered_stops:
            w = world_cls(role=role, **{axis: value})
            got = score(w)
            chk(f"{name} T26 {axis}={value!r} ({role}) -> {want}", got == want,
                f"got {got}")
    if guard_label:
        # ★T25.  A necessary-condition check constrains worlds that DO carry a
        # label and is structurally blind to a label that STOPPED appearing --
        # remove a guard and its label is vacuously absent.  Sufficiency would
        # fix that and would be the branch-order transcription the rev2 audit
        # showed cannot catch a specification error.  So removal is tested at
        # the label census instead.
        print(f"  -- {name} T25 meta (each guard's own label disappears without it) --")
        for g, lab in sorted(guard_label.items()):
            before = cnt[lab]
            after = Counter(score(w, {g}) for w in W)[lab]
            chk(f"{name} T25 {g} produces {lab}", after < before,
                f"{before} -> {after}")
    if invariants:
        print(f"  -- {name} T23 meta (axis invariance of the scoring function) --")
        for inv, (flip, breakers) in invariants.items():
            holds = all(score(w) == score(flip(w)) for w in W)
            chk(f"{name} T23 invariant '{inv}' holds under the base rule", holds)
            for m in mutants:
                broken = any(score(w, {m}) != score(flip(w), {m}) for w in W)
                if m in breakers:
                    chk(f"{name} T23 invariant '{inv}' IS broken by {m}", broken)
                else:
                    chk(f"{name} T23 invariant '{inv}' survives {m}", not broken)
    return fails, len(W), dict(cnt), {n: sole[n] for n in checks}


def run():
    print(f"== A1 Q3/K1 rule (RULE_REV={RULE_REV}) ==")
    res = {}
    f1, n1, c1, s1 = _battery("Q3", q3_worlds, q3_score, Q3_MUTANTS,
                              _q3_checks(), Q3_LABELS, res,
                              guard_label=Q3_GUARD_LABEL,
                              registered_stops=REGISTERED_STOPS,
                              world_cls=Q3World)
    f2, n2, c2, s2 = _battery("K1", k1_worlds, k1_score, K1_MUTANTS,
                              _k1_checks(), K1_LABELS, res,
                              invariants=K1_INVARIANTS)
    ok = not (f1 or f2)
    print(f"\n== Q3 {n1:,} worlds / K1 {n2:,} worlds -> "
          f"{'ALL PASS' if ok else str(len(f1 + f2)) + ' FAILURES: ' + '; '.join((f1 + f2)[:4])} ==")
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.abspath(__file__), "rb") as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()
    with open(os.path.join(here, "selftest_a1_q3k1_2026-08-25.json"), "w") as fh:
        json.dump({"rule_rev": RULE_REV, "rule_sha256": sha, "date": "2026-08-25",
                   "gpu_hr": 0.0, "q3_worlds": n1, "k1_worlds": n2,
                   "q3_labels": c1, "k1_labels": c2,
                   "q3_sole_binding": s1, "k1_sole_binding": s2,
                   "n_min_decode_steps": N_MIN_DECODE_STEPS,
                   "k1_bands": [K1_MOD_AT, K1_HIGH_AT, K1_PROHIB_AT],
                   "checks": res, "all_pass": ok}, fh, indent=1, sort_keys=True)
    print("   wrote selftest_a1_q3k1_2026-08-25.json")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(run())
