#!/usr/bin/env python3
"""A1 **rev2** -- the new decision rules: Q3 (step-boundary separability) and K1.

WHAT CHANGED FROM rev1 (which was audited `NO-GO`, fatal flaws A1/A2):
  The bridges are split by BOOT now, not by a time window inside one run
  (`DESIGN_A1_REV2_STICKY_2026-08-25.md` sec 2).  So the axes here are
  boot-level, the stop rules are boot-level, and nothing joins across a clock.

  Repairs carried out against the audit, item by item:
    C1  `N_min` is an INTEGER axis compared against a literal that the
        self-test restates and a value mutant must break.  rev1's
        `N_MIN_SPLIT_STEPS = 200` was defined once, referenced zero times, and
        survived 200 -> 5 unchanged.  See `t_span_threshold`/`g_nmin_value`.
    C2  the unit is decode STEPS (`Delta decode_iterations`), never snapshots.
        Measured 2026-08-25: pooled over 8 boots only 2.3% of benchmark
        snapshot pairs advance the counter and the mean is 0.156-0.409 steps
        per snapshot, so no snapshot->step conversion is sound.  (The audit's
        own C2 arithmetic IS exact on ITS population -- the 130 split-realized
        snapshots all sit at an advancing interval and sum to exactly 2,064
        steps.  This rule does not contradict that; it removes the need for the
        conversion.)  Artefact: `a1_q3_channel_probe.json`.
    C5  `export == "partial"` is a MEASUREMENT condition and cannot reach a
        substantive label.  A0's job 892554 landed in exactly that world.
    D2  more than one discriminating check, and they read DIFFERENT
        cross-sections -- rev1's single check transcribed `q3_score`'s branch
        order and could not catch a specification error.
    D4  K1 has four bands.
    D5  `K1_HIGH_AT` provenance retracted, marked [ARBITRARY].
    D6  K1's measurement channel is registered, with the code facts that decide
        whether it is live at all (see K1_CHANNEL below).
    D7  `launches == "more"` has a registered meaning.
    D8  `dropped_events` is an axis.

  Q1e/Q2ae/Q2be are still scored by the A0 rule -- BUT the `stream` axis is
  redefined (design sec 3.2) because nsys `streamId` is report-local, so rev1's
  "L3 and L4 share a stream" is impossible once the legs are separate boots.
  ★That means a new A0 rule revision and ONE more rules-layer audit.  rev1
  sec 4's "we do not re-audit" is RETRACTED.

Run:  python3 a1_q3k1_rule.py            # full self-test
"""
import hashlib, json, os, sys
from collections import Counter
from itertools import product

RULE_REV = 2

# --- Q3 labels: measurement conditions ---------------------------------------
Q3_BOOT     = "Q3_BOOT_FAILED"
Q3_TRUNC    = "Q3_TRACE_TRUNCATED"           # export partial/fail, dropped, halves
Q3_NOTEL    = "Q3_TELEMETRY_ABSENT"
Q3_NOSTICKY = "Q3_STICKY_NOT_REALIZED"
Q3_ZEROCH   = "Q3_CHANNEL_ZERO"              # the counter never advanced
Q3_SHORT    = "Q3_SPAN_TOO_SHORT"            # advanced, but < N_MIN_DECODE_STEPS
Q3_NOLEG    = "Q3_LEG_LABELLING_UNAVAILABLE" # no decode-only kernel name
Q3_MEASUREMENT_LABELS = {Q3_BOOT, Q3_TRUNC, Q3_NOTEL, Q3_NOSTICKY, Q3_ZEROCH,
                         Q3_SHORT, Q3_NOLEG}
# --- Q3 labels: substantive ---------------------------------------------------
Q3_STREAMAMB = "Q3_STREAM_AMBIGUOUS"         # rows not on ONE stream (sec 3.2 i)
Q3_NOSEP     = "BOUNDARY_NOT_SEPARABLE"
Q3_PARTIAL   = "BOUNDARY_SEPARABLE_PARTIAL"
Q3_SEP       = "BOUNDARY_SEPARABLE"
Q3_SUBSTANTIVE_LABELS = {Q3_STREAMAMB, Q3_NOSEP, Q3_PARTIAL, Q3_SEP}
Q3_LABELS = Q3_MEASUREMENT_LABELS | Q3_SUBSTANTIVE_LABELS

# --- K1 labels ----------------------------------------------------------------
K1_UNMEAS   = "K1_UNMEASURED"                # a leg did not produce a number
K1_DEADCH   = "K1_CHANNEL_DEAD"              # the ITL/step channel is not live
K1_PAIRBAD  = "K1_PAIR_INVALID"              # the legs differ in more than nsys
K1_NEAR1    = "K1_NEAR_1"
K1_MOD      = "K1_MODERATE"
K1_HIGH     = "K1_HIGH"
K1_PROHIB   = "K1_PROHIBITIVE"
K1_LABELS = {K1_UNMEAS, K1_DEADCH, K1_PAIRBAD, K1_NEAR1, K1_MOD, K1_HIGH,
             K1_PROHIB}

# --- registered constants -----------------------------------------------------
# ★UNIT = decode steps, measured as `Delta decode_iterations` over the boot.
#   NOT snapshots (C2).  Value: half of one batch-synchronous round (out=128).
#   ★[ARBITRARY] -- it is a floor on "enough replays to divide by", not a
#   calibrated quantity.  What is NOT arbitrary is that it is compared against
#   an integer here and that `g_nmin_value` proves the comparison is live.
N_MIN_DECODE_STEPS = 64

K1_MOD_AT    = 1.10   # ratio of per-step wall time, nsys ON / OFF
K1_HIGH_AT   = 2.00   # ★[ARBITRARY -- provenance retracted, audit D5].  rev1
                      # attributed this to rev8 as "the affordability stop";
                      # rev8 contains "K1" ZERO times and records nsys overhead
                      # as UNMEASURED (:438).  Round number, nothing more.
K1_PROHIB_AT = 10.00  # ★[ARBITRARY] -- the design's fourth band (">=10").
                      # Registered so the band exists, not because it is known.

# ★K1_CHANNEL -- registered 2026-08-25 from code + existing telemetry (GPU 0):
#
#   PRIMARY   per-step wall time = boot duration / `Delta decode_iterations`.
#             Both terms are counters/timestamps.  Neither passes through the
#             outlier filter below, which is why this and not the ITL field is
#             primary.
#   SECONDARY `measured_itl_p95_ms` = p95 of the last <=128 RAW per-iteration
#             samples (`multiplexing_mixin.py:392-397`); `measured_itl_ewma_ms`
#             is the same signal EMA-smoothed (alpha `PDMUX_SLO_EMA`, 0.85).
#   DEAD      `decode_last_tpot_ms` -- written in `dual_worker.py:121` inside
#             `finish_step()`, behind the same `dual_worker_enabled` gate that
#             kills `decode_step_count`.  Measured 0 / 294,506 snapshots.
#
#   ★LIVENESS IS CONDITIONAL.  The ITL fields update only when
#   `_slo_on = bool(PDMUX_SLO_SCHED) or self.r2_policy is not None`
#   (`multiplexing_mixin.py:1009`).  A1's registered recipe sets
#   PDMUX_R2_POLICY=fixed, so they ARE live -- measured non-zero in 4/4
#   s2_sticky boots (64,788 snapshots) and zero in 4/4 g16 boots, which set
#   neither.  ⇒ a boot that omits the policy has a dead channel: K1_CHANNEL_DEAD.
#
#   ★CLIPPING HAZARD (why PRIMARY is not the ITL field).  The sampler accepts
#   `_dt` only inside `(0, max(3*EMA, 90ms))` (`:1029`).  K1 exists to measure
#   profiler overhead; an nsys-ON leg whose true per-iteration time leaves that
#   band has its samples REJECTED, so the ITL field would understate the
#   overhead it is supposed to reveal.  The `clipped` axis below carries that.

# =============================================================================
# Q3
# =============================================================================
class Q3World:
    """One possible outcome of the Q3 measurement, per BOOT.

    boot_ok   : the server came up and answered /health
    export    : nsys export state -- ok | partial | fail
    dropped   : CUPTI dropped_events -- 0 | positive
    tel       : engine telemetry present
    sticky    : realized-partition gate (design sec 5-2/5-3) -- ok | low | nolog
    steps     : ★INTEGER.  Delta decode_iterations over the analysed boot.
    halves    : node rows present in both halves of the replay order -- both|one
    legnames  : is there a kernel name unique to the decode span -- yes|no
    launches  : graph-launch rows vs `steps`, at RUN level.
                zero  : none found
                fewer : fewer rows than decode steps -- some steps unlaunched
                equal : 1:1, the thing Q3 asks about
                ★more : MORE rows than decode steps.  Registered meaning (D7):
                        the row population is not the decode replays alone
                        (capture-time launches, a second graph, another
                        stream's rows misattributed).  It is NOT "separable
                        with room to spare" -- it means the denominator is
                        wrong, so it scores NOT separable.
    stream    : do the rows sit on ONE streamId, disjoint from the prefill-only
                kernels' stream (design sec 3.2) -- single | multi
    partition : do node rows partition by launch correlationId -- clean|orphans|overlap
    """

    def __init__(self, boot_ok=True, export="ok", dropped=0, tel="ok",
                 sticky="ok", steps=3802, halves="both", legnames="yes",
                 launches="equal", stream="single", partition="clean"):
        self.boot_ok, self.export, self.dropped, self.tel = boot_ok, export, dropped, tel
        self.sticky, self.steps, self.halves = sticky, steps, halves
        self.legnames, self.launches = legnames, launches
        self.stream, self.partition = stream, partition

    def __repr__(self):
        return (f"Q3(boot={self.boot_ok},exp={self.export},drop={self.dropped},"
                f"tel={self.tel},sticky={self.sticky},steps={self.steps},"
                f"halves={self.halves},leg={self.legnames},"
                f"launch={self.launches},str={self.stream},part={self.partition})")


def q3_score(w, guards=frozenset()):
    """Boot-level.  MEASUREMENT conditions are decided before anything reads a
    substantive quantity -- gate #21: a plumbing failure must never be turned
    into a statement about the tool or the engine."""
    def on(g):
        return g not in guards

    if on("g_boot") and not w.boot_ok:
        return Q3_BOOT
    # C5: `partial` belongs HERE.  A0's first run (892554) had every axis clean
    # and still exported partially; letting that reach a substantive label is
    # the defect the A0 harness audit repaired and rev1 re-opened.
    if on("g_trunc") and (w.export != "ok" or w.dropped > 0 or w.halves != "both"):
        return Q3_TRUNC
    if on("g_tel") and w.tel != "ok":
        return Q3_NOTEL
    # design sec 5-2/5-3: what the boot REALIZED, not what it targeted.  The
    # D108 lesson: verify realized, never the label.
    if on("g_sticky") and w.sticky != "ok":
        return Q3_NOSTICKY
    # ★C1: the threshold is READ here, against an integer.  `g_nmin_value`
    # proves it: change the literal and the self-test must fail.
    if on("g_zero") and w.steps <= 0:
        return Q3_ZEROCH
    nmin = N_MIN_DECODE_STEPS if on("g_nmin_value") else 5
    if on("g_span") and "g_order_span_last" not in guards and w.steps < nmin:
        return Q3_SHORT
    if on("g_leg") and w.legnames != "yes":
        return Q3_NOLEG
    # substantive from here down
    if on("g_stream") and w.stream != "single":
        return Q3_STREAMAMB
    if on("g_nosep") and w.launches != "equal":
        return Q3_NOSEP
    if on("g_nosep") and w.partition == "overlap":
        return Q3_NOSEP
    if on("g_partial") and w.partition == "orphans":
        return Q3_PARTIAL
    # ORDER MUTANT (A0's N1 shape): with the span guard LAST, a boot with too
    # few steps has too few launch rows and scores BOUNDARY_NOT_SEPARABLE --
    # "the tool cannot separate steps" -- manufactured out of a short run.
    if "g_order_span_last" in guards and w.steps < N_MIN_DECODE_STEPS:
        return Q3_SHORT
    return Q3_SEP


# `steps` values straddle 0 and the registered literal in BOTH directions, and
# include 5 so that `g_nmin_value` (64 -> 5) actually changes labels.
Q3_AXES = dict(
    boot_ok=[True, False],
    export=["ok", "partial", "fail"],
    dropped=[0, 17],
    tel=["ok", "absent"],
    sticky=["ok", "low", "nolog"],
    steps=[0, 4, 63, 64, 3802],
    halves=["both", "one"],
    legnames=["yes", "no"],
    launches=["zero", "fewer", "equal", "more"],
    stream=["single", "multi"],
    partition=["clean", "orphans", "overlap"],
)
Q3_MUTANTS = ["g_boot", "g_trunc", "g_tel", "g_sticky", "g_zero", "g_span",
              "g_nmin_value", "g_leg", "g_stream", "g_nosep", "g_partial",
              "g_order_span_last"]


def q3_worlds():
    keys = list(Q3_AXES)
    for c in product(*(Q3_AXES[k] for k in keys)):
        yield Q3World(**dict(zip(keys, c)))


def _q3_plumbing_dirty(w):
    """The registered measurement conditions, stated as a PREDICATE rather than
    as a branch order -- this is what check A reads."""
    return (not w.boot_ok or w.export != "ok" or w.dropped > 0
            or w.halves != "both" or w.tel != "ok" or w.sticky != "ok"
            or w.steps < N_MIN_DECODE_STEPS or w.legnames != "yes")


def _q3_checks():
    """★D2: MORE THAN ONE check, and they read DIFFERENT cross-sections.

    rev1 had exactly one, and it was a transcription of `q3_score`'s branch
    order -- so T19a/T19b could not fail and a specification error was
    invisible.  rev1 got there because six per-label iff checks had collapsed
    to one check's worth of information on a total iff-partition (that
    diagnosis was right).  The way out is not "six again", it is checks whose
    FAILURE SETS are shaped differently.

      A  projection : plumbing dirty  =>  label is a MEASUREMENT label.
                      Says nothing about WHICH substantive label, and applies
                      to every world.  This is the gate #21 property.
      B  top-label  : SEP  <=>  everything clean AND equal AND single AND clean
                      partition.  Says nothing about worlds that are not SEP
                      beyond "not SEP".
      C  threshold  : with plumbing otherwise clean, SPAN_TOO_SHORT holds
                      exactly on 0 < steps < 64 (literal restated).  A single
                      integer cross-section; the C1 repair's own check.

    A does not entail B (A is silent on substantive labels), B does not entail
    A (B only rejects wrong SEPs), and C is a strictly narrower slice than
    either -- T19a/T19b are evaluated, not assumed, and both results print.
    """
    def c_projection(w, l):
        if _q3_plumbing_dirty(w):
            return l in Q3_MEASUREMENT_LABELS
        return l in Q3_SUBSTANTIVE_LABELS

    def c_toplabel(w, l):
        clean = (not _q3_plumbing_dirty(w) and w.launches == "equal"
                 and w.stream == "single" and w.partition == "clean")
        return (l == Q3_SEP) == clean

    def c_threshold(w, l):
        # only the slice where nothing else can claim the world first
        if (not w.boot_ok or w.export != "ok" or w.dropped > 0
                or w.halves != "both" or w.tel != "ok" or w.sticky != "ok"):
            return True
        return (l == Q3_SHORT) == (0 < w.steps < 64)

    return {
        "Q3-A plumbing dirty => measurement label (projection)":
            (c_projection, {"g_boot", "g_trunc", "g_tel", "g_sticky", "g_span",
                            "g_leg", "g_nmin_value"}),
        "Q3-B SEP iff every axis clean (top label)":
            (c_toplabel, {"g_stream", "g_nosep", "g_partial"}),
        # ★`g_zero` belongs HERE, not on Q3-B.  Removing the zero guard turns
        # CHANNEL_ZERO into SPAN_TOO_SHORT -- both measurement labels, neither
        # SEP -- so Q3-A and Q3-B are structurally blind to it and only the
        # threshold check sees `steps == 0` labelled SHORT.  The first run of
        # this suite named it on Q3-B and T8 failed; that is a check<->mutant
        # PAIRING failure, the same shape A0's rev2 audit found in X2, and it
        # is recorded rather than silently re-paired.
        "Q3-C SPAN_TOO_SHORT iff 0 < steps < 64 (literal)":
            (c_threshold, {"g_span", "g_nmin_value", "g_order_span_last",
                           "g_zero"}),
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
def _battery(name, worlds, score, mutants, checks, labels, out, invariants=None):
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
    chk(f"{name} T10 every mutant covered", not unc, f"uncovered={unc or 'none'}")
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
                              _q3_checks(), Q3_LABELS, res)
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
