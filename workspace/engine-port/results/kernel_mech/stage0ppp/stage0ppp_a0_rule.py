#!/usr/bin/env python3
"""Stage 0''' A0 -- THE REGISTERED DECISION RULE, fixed in code (gate #66).

Origin: audit_stage0pp_2026-08-23/VERDICT.md sec4 (Stage A0 redesign), blockers
E1/E2/E5/E10.  The Stage 0'' prereg stated its rule in PROSE ONLY and the audit
scored that E10 -- an IMMEDIATE relapse of the lesson registered the day before
("fix the decision rule in code and enumerate the world space exhaustively").
This file is that fix.  It answers ONE question per world and nothing else; it
never touches a file, a GPU, or the engine.

WHAT THIS FILE IS NOT: it is not a measurement, not a harness, and not evidence.
Scoring a world here says nothing about the substrate.

Run:  python3 stage0ppp_a0_rule.py            # full self-test
"""
from itertools import product

# --- registered labels -------------------------------------------------------
# Measurement conditions (gate #21: a measurement failure is NOT a verdict).
ABSENT   = "MEASUREMENT_ABSENT"            # nsys run or sqlite export failed
INVALID  = "PROBE_INVALID"                 # a CONTROL failed -> adjudicate nothing
# Tool-limit facts (green-context-independent).
NONODE   = "NODE_TRACE_UNAVAILABLE"        # L2 has no node rows at all
NOGREEN  = "GREENCTX_INVISIBLE"            # green-ctx kernels absent from the trace
# Substantive outcomes.
UNCONSTR = "PRIMARY_ESTIMAND_UNCONSTRUCTIBLE"   # ★NOT "track closed" (E1)
NOATTR   = "KSET_NOT_ATTRIBUTABLE"
CONTRA   = "CONTRADICTORY_ATTRIBUTION"     # ctx says green, stream says not
CAPJOIN  = "KSET_JOINS_CAPTURE_NOT_REPLAY" # ★E5: field present, membership unusable
TIMEATTR = "KSET_NEEDS_TIME_ATTRIBUTION"   # E1-(d) path
PARTIAL  = "KSET_CONSTRUCTIBLE_PARTIAL"    # join rate below the registered floor
OK       = "KSET_CONSTRUCTIBLE"            # scoped; basis is carried, see below

LABELS = {ABSENT, INVALID, NONODE, NOGREEN, UNCONSTR, NOATTR, CONTRA,
          CAPJOIN, TIMEATTR, PARTIAL, OK}
# Attribution BASIS is reported beside the label and never collapsed into it:
#   "ctx+stream" = greenContextId AND streamId agree
#   "stream_only"= E1-(c): greenContextId unusable, streamId alone carries it
BASES = {"ctx+stream", "stream_only", None}

# --- registered constants (changing one is a protocol amendment) -------------
Q1_FRAC   = 0.90   # node rows observed / node rows EXPECTED (expectation from L2)
JOIN_HIGH = 0.95   # correlationId -> replay cudaGraphLaunch join success floor


class World:
    """One possible outcome of the four legs + tool state.

    L1 full-GPU eager        (control: does nsys see ANY kernel at all)
    L2 full-GPU graph        (control: node rows exist AND define the expected
                              node count -- green-independent)
    L3 green-ctx eager       (control: are green-ctx kernels visible at all)
    L4 green-ctx graph       (★the condition under test)
    """

    def __init__(self, run_ok=True, export_ok=True, l1="ok", l2="ok", l3="ok",
                 capture="ok", l4_frac=1.0, ctx="distinct", stream="match",
                 join="replay_high"):
        self.run_ok, self.export_ok = run_ok, export_ok
        self.l1, self.l2, self.l3 = l1, l2, l3     # "ok" | "zero"
        self.capture = capture                      # "ok" | "fail"
        self.l4_frac = l4_frac                      # 0.0 | 0.5 | 0.9 | 1.0
        self.ctx = ctx        # "null" | "zero" | "parent" | "distinct"
        self.stream = stream  # "match" | "mismatch"
        self.join = join      # "none" | "capture_only" | "replay_low" | "replay_high"

    def __repr__(self):
        return (f"W(run={self.run_ok},exp={self.export_ok},l1={self.l1},l2={self.l2},"
                f"l3={self.l3},cap={self.capture},frac={self.l4_frac},ctx={self.ctx},"
                f"str={self.stream},join={self.join})")


def score(w, guards=frozenset()):
    """The registered rule.  Returns (label, basis).

    `guards` names guards to DISABLE or WEAKEN, for mutation testing.  Per
    lesson #70 the suite must include PARAMETRIC weakenings, not only
    deletions -- a deletion-only suite cannot see a loosened threshold.
    """
    def on(g):
        return g not in guards

    # -- 1. measurement conditions come FIRST (gate #21) ----------------------
    if on("g_measurement") and not (w.run_ok and w.export_ok):
        return ABSENT, None
    # ★A capture failure on the green leg is a MEASUREMENT condition, not the
    #   answer "graph node rows are unavailable under green context".  Stage
    #   0'' had no such branch: a broken capture would have been read as Q1=no
    #   and, under its own rule, escalated to a track-closing statement.
    if on("g_capture") and w.capture != "ok":
        return INVALID, None

    # -- 2. controls (E2: without these, every failure mode reads as Q1=no) ---
    if on("g_l1") and w.l1 == "zero":
        return INVALID, None            # nsys attached to nothing / wrong process
    if on("g_l2") and w.l2 == "zero":
        return NONODE, None             # node granularity unavailable -- green-independent
    if on("g_l3") and w.l3 == "zero":
        return NOGREEN, None            # P1-shaped fact; green kernels not in the trace

    # -- 3. Q1: are node rows there, against an EXPECTATION MEASURED IN L2 ----
    floor = Q1_FRAC if on("g_q1_floor") else 0.5     # parametric weakening
    if on("g_q1") and w.l4_frac < floor:
        return UNCONSTR, None           # ★E1: not "track closed"; (b)(c)(d) survive

    # -- 4. attribution: does the row belong to the green context? -----------
    ctx_ok = w.ctx == "distinct" if on("g_ctx") else w.ctx in ("distinct", "parent")
    strm_ok = w.stream == "match"
    if on("g_contra") and ctx_ok and not strm_ok:
        # Two independent channels disagree -> a bug in the probe or the
        # analyser, not a fact about the driver.  Adjudicate nothing.
        return CONTRA, None
    if not ctx_ok and not strm_ok:
        return NOATTR, None
    basis = "ctx+stream" if (ctx_ok and strm_ok) else "stream_only"
    if not on("g_basis"):
        basis = "ctx+stream"   # mutant: the report drops the basis

    # -- 5. join: can a row be tied to THE REPLAY that produced it? ----------
    # ★E5: the audit's mirror-image gap.  A correlationId that joins only to a
    #   CAPTURE-time launch has the field populated and is still useless for
    #   membership.  Stage 0'' would have scored that world "Q2 = yes".
    if on("g_capjoin") and w.join == "capture_only":
        return CAPJOIN, basis
    if on("g_join_none") and w.join == "none":
        return TIMEATTR, basis          # E1-(d) remains open
    if w.join == "replay_low":
        return PARTIAL, basis
    return OK, basis


def worlds():
    ax = dict(
        run_ok=[True, False], export_ok=[True, False],
        l1=["ok", "zero"], l2=["ok", "zero"], l3=["ok", "zero"],
        capture=["ok", "fail"],
        # ★0.9 sits exactly ON the registered floor: an off-by-one in the
        #   comparison (< vs <=) changes this world's label and is caught.
        l4_frac=[0.0, 0.5, 0.9, 1.0],
        ctx=["null", "zero", "parent", "distinct"],
        stream=["match", "mismatch"],
        join=["none", "capture_only", "replay_low", "replay_high"],
    )
    keys = list(ax)
    for combo in product(*(ax[k] for k in keys)):
        yield World(**dict(zip(keys, combo)))


MUTANTS = ["g_measurement", "g_capture", "g_l1", "g_l2", "g_l3", "g_q1",
           "g_q1_floor", "g_ctx", "g_contra", "g_capjoin", "g_join_none",
           "g_basis"]


# ============================================================================
# COMPANION MEASUREMENT -- the E1-(b) fallback, bought in the same job.
#
# * `nsys profile --help` (2025.3.2, read 2026-08-24, GPU 0): "If 'node' is
#   selected, node activities will be collected, but CUDA graphs will NOT be
#   traced as a whole."  The two granularities are MUTUALLY EXCLUSIVE, so the
#   primary rule's `node` run cannot also answer E1-(b).  Because the probe is
#   tiny, the four legs are run TWICE (node, graph) and the `graph` run is
#   scored by this companion rule.  Consequence: `PRIMARY_ESTIMAND_
#   UNCONSTRUCTIBLE` no longer leaves the track empty-handed -- the state of
#   the (b) path is already known when that label fires.
#
# * NOT answered by the docs: the `[:<launch origin>]` suffix is
#   'host-only' | 'host-and-device', i.e. WHERE the launch originated (host vs
#   device code).  It does NOT distinguish capture-time from replay-time
#   launches, so it does not close E5.  (Recorded because reading it as an E5
#   answer was the obvious mistake to make.)
# ============================================================================
E1B_UNMEAS = "E1B_UNMEASURED"
E1B_DEAD   = "E1B_DEAD"            # whole-graph rows absent under green ctx
E1B_STREAM = "E1B_NEEDS_STREAM"    # rows present, ctx unusable -> stream basis
E1B_ALIVE  = "E1B_ALIVE"
E1B_LABELS = {E1B_UNMEAS, E1B_DEAD, E1B_STREAM, E1B_ALIVE}


class GWorld:
    def __init__(self, run_ok=True, l1g="ok", l4g_rows="ok", l4g_ctx="distinct"):
        self.run_ok, self.l1g = run_ok, l1g
        self.l4g_rows, self.l4g_ctx = l4g_rows, l4g_ctx

    def __repr__(self):
        return (f"G(run={self.run_ok},l1g={self.l1g},rows={self.l4g_rows},"
                f"ctx={self.l4g_ctx})")


def companion(g, guards=frozenset()):
    def on(x):
        return x not in guards
    if on("h_measure") and not g.run_ok:
        return E1B_UNMEAS
    if on("h_ctl") and g.l1g == "zero":          # same control discipline as A0
        return E1B_UNMEAS
    if on("h_rows") and g.l4g_rows == "zero":
        return E1B_DEAD
    if on("h_ctx") and g.l4g_ctx != "distinct":
        return E1B_STREAM
    return E1B_ALIVE


def gworlds():
    for run_ok in (True, False):
        for l1g in ("ok", "zero"):
            for rows in ("ok", "zero"):
                for ctx in ("null", "zero", "parent", "distinct"):
                    yield GWorld(run_ok, l1g, rows, ctx)


H_MUTANTS = ["h_measure", "h_ctl", "h_rows", "h_ctx"]


def _hdead(sc):
    # DISCRIM: E1B_DEAD must never be reported when the control leg failed --
    # otherwise a mis-attached profiler is written up as "the (b) path is dead".
    return not [g for g in gworlds()
                if sc(g) == E1B_DEAD and (not g.run_ok or g.l1g == "zero")]


def run_companion(chk):
    G = list(gworlds())
    labs = [companion(g) for g in G]
    chk(f"C1 totality ({len(G)} graph-granularity worlds)",
        all(l in E1B_LABELS for l in labs))
    from collections import Counter
    cc = Counter(labs)
    for lab in sorted(E1B_LABELS):
        chk(f"C2 reachable: {lab}", cc[lab] > 0, f"n={cc[lab]}")
    for m in H_MUTANTS:
        d = sum(1 for g in G if companion(g) != companion(g, {m}))
        chk(f"C3 mutant {m} is load-bearing", d > 0, f"n={d}")
    chk("C4 E1B_DEAD never on a failed control [DISCRIM]", _hdead(companion))
    chk("C4 meta: fails under ['h_ctl']",
        not _hdead(lambda g: companion(g, {"h_ctl"})))


def _t4(sc):
    SUBST = {UNCONSTR, NOATTR, CONTRA, CAPJOIN, TIMEATTR, PARTIAL, OK}
    return not [w for w in worlds() if sc(w)[0] in SUBST
                and (not w.run_ok or not w.export_ok or w.capture != "ok"
                     or w.l1 == "zero" or w.l2 == "zero" or w.l3 == "zero")]


def _t6b(sc):
    # * The first form of this check was an IDENTITY and the T8 meta-check
    #   caught it on the first run: "stream_only => ctx != distinct" follows
    #   from the definition of basis, so NO mutant could break it.  The
    #   direction that carries real risk is the opposite one -- a world whose
    #   greenContextId is unusable must never be REPORTED on a ctx basis,
    #   because that is exactly how "stream attribution worked" becomes
    #   "greenContextId works" in a write-up (gate #61 / the R0
    #   label-vs-index overclaim).
    return all(not (w.ctx in ("null", "zero", "parent") and sc(w)[1] == "ctx+stream")
               for w in worlds() if sc(w)[1] is not None)


def _t7(sc):
    return not [w for w in worlds() if w.join == "capture_only" and sc(w)[0] == OK]


def _t5(sc):
    return all(w.l4_frac < Q1_FRAC for w in worlds() if sc(w)[0] == UNCONSTR)


# * Each DISCRIMINATING check names a mutant that MUST break it.  A check with
#   no such mutant is an IDENTITY dressed as evidence -- this project has been
#   bitten by that 16 times (gate #9), most recently by a self-repair check
#   that could not fail (lesson #53).  Checks that are structurally true under
#   the present rule are labelled REGRESS and are NOT counted as evidence of
#   anything; they exist to catch a future reordering edit (the P0-A round-2
#   C5 defect, where a check moved ahead of the capture branch and the whole
#   suite still passed).
DISCRIM = {
    "T4 no substantive verdict on a failed control": (_t4, {"g_l1"}),
    "T6b unusable ctx never reported on a ctx basis": (_t6b, {"g_basis"}),
    "T7 capture-only join never CONSTRUCTIBLE":      (_t7, {"g_capjoin"}),
}
REGRESS = {"T5 UNCONSTR reachable only via Q1": _t5}


def run():
    W = list(worlds())
    fails = []

    def chk(name, cond, detail=""):
        print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")
        if not cond:
            fails.append(name)

    print(f"== Stage 0ppp A0 rule -- exhaustive enumeration: {len(W):,} worlds ==")

    labs = [score(w) for w in W]
    chk("T1a totality (every world lands in a registered label)",
        all(l in LABELS for l, _ in labs))
    chk("T1b basis is registered", all(b in BASES for _, b in labs))
    chk("T1c determinism", labs == [score(w) for w in W])

    from collections import Counter
    c = Counter(l for l, _ in labs)
    for lab in sorted(LABELS):
        chk(f"T2 reachable: {lab}", c[lab] > 0, f"n={c[lab]}")

    for m in MUTANTS:
        diff = sum(1 for w in W if score(w) != score(w, {m}))
        chk(f"T3 mutant {m} is load-bearing", diff > 0, f"n={diff}")

    for name, (fn, mut) in DISCRIM.items():
        chk(f"{name} [DISCRIM]", fn(score))
    chk("T6 E1-(c) stream-only attribution is reachable",
        any(b == "stream_only" for _, b in labs),
        f"n={sum(1 for _, b in labs if b == 'stream_only')}")
    for name, fn in REGRESS.items():
        chk(f"{name} [REGRESS -- not evidence]", fn(score))

    run_companion(chk)

    # -- T8 *META: every DISCRIM check must FAIL on its named mutant --------
    print("  -- T8 meta: does each DISCRIM check actually fail when broken? --")
    for name, (fn, mut) in DISCRIM.items():
        chk(f"T8 '{name}' fails under {sorted(mut)}",
            not fn(lambda w: score(w, mut)))

    print(f"\n== {len(MUTANTS)} mutants, {len(DISCRIM)} discriminating checks, "
          f"{len(LABELS)} labels, {len(W):,} worlds -> "
          f"{'ALL PASS' if not fails else 'FAILURES: ' + ','.join(fails)} ==")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(run())
