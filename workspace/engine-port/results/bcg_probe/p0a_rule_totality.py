#!/usr/bin/env python3
"""P0-A: is the registered decision RULE total, unambiguous, and load-bearing?

WHY THIS EXISTS
---------------
Two audit rounds found holes created by fixing the rule in PROSE:
  * round 1 (B1): an empty census satisfied every set predicate vacuously.
  * round 2 (C1): a graph leg that is too NARROW also satisfies `E = ∅`.
    The `TOL` band that the rev1 rewrite deleted had TWO sides; only the lower
    one was restored.
  * round 2 (C5): the prose evaluation order made `NOCAP` unreachable.
Round 2 therefore recommended, out of scope, a "world enumerator". This is it.

It implements the registered rule as a PURE function of a world description and
enumerates the whole world space, asserting:
  T1 totality        every world gets exactly one label from the registered set
  T2 no vacuity      no world with a degenerate observation reaches a
                     SUBSTANTIVE label (PRESERVED / LOST_*)
  T3 reachability    every registered label is produced by at least one world
                     (a label no world can reach is dead prose -- C5's defect)
  T4 load-bearing    each guard, when deleted, changes at least one world's
                     label (mutation tests, methodology lesson #53)

★ This checks the RULE, not the harness. It is deliberately independent of
  `p0a_graph_sm_confinement.py` so that a harness bug cannot make the rule look
  total. The harness must later be shown to implement THIS function.
"""
import argparse
import itertools
import sys

# --- registered labels ------------------------------------------------------
PRESERVED = "CONFINEMENT_PRESERVED_THROUGH_GRAPH_REPLAY"
LOST_FULL = "CONFINEMENT_LOST_THROUGH_GRAPH_REPLAY (FULL)"
LOST_PART = "CONFINEMENT_LOST_THROUGH_GRAPH_REPLAY (PARTIAL)"
UNSEP = "UNDER_COVERAGE_IN_GRAPH_LEG (UNSEPARATED)"   # round-3 D5 rename
DISAGREE = "UNDETERMINED (LABEL SET DISAGREEMENT WITHIN TARGET)"  # round-3 D1(a)
ABSENT = "UNDETERMINED (MEASUREMENT ABSENT)"
NOSAT = "UNDETERMINED (COVERAGE NOT SATURATED)"
NOATT = "UNDETERMINED (MEASUREMENT ABSENT: GREEN CONTEXT NOT ATTACHED)"
NODEL = "GREEN PARTITION NOT DELIVERED (RESULT)"
NULLCH = "UNDETERMINED (DECISION QUANTITY'S NULL CHANNEL OPEN)"
NOCAP = "UNDETERMINED (GRAPH CAPTURE UNAVAILABLE ON GREEN STREAM)"
NOREPLAY = "UNDETERMINED (GRAPH REPLAY FAILED)"
SUBSTANTIVE = {PRESERVED, LOST_FULL, LOST_PART, UNSEP}
LABELS = SUBSTANTIVE | {ABSENT, NOSAT, NOATT, NODEL, NULLCH, NOCAP,
                        NOREPLAY, DISAGREE}

# ★ round-3 D8: these were literals with a comment claiming "from the producer".
#   Import them, and record loudly when the producer is unreachable -- a rule
#   file that silently substitutes its own constants is the R9 defect shape.
def _from_producer():
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(
        _os.path.abspath(__file__))), "smid_census"))
    import smid_l0_census as _CEN
    _sys.path.insert(0, "/scratch/ehmoon/whlee/sglang_engine_dev/python")
    from sglang.srt.multiplex.pdmux_context import get_arch_constraints as _gac
    return _CEN.MIN_HITS_REPORTED, _gac((8, 0))[1]


try:
    MIN_HITS, GRANULARITY = _from_producer()
    PRODUCER_OK, PRODUCER_ERR = True, None
except Exception as _e:            # noqa: BLE001
    MIN_HITS, GRANULARITY, PRODUCER_OK = 1, 2, False
    PRODUCER_ERR = repr(_e)


class World:
    """A world = what each leg observed + tool outcomes. Sets are label sets."""

    def __init__(self, S_ep, S_gp, S_eg, S_gg, d_sm=34,
                 min_hits=None, saturated=None, attached=True,
                 capture="ok", replay="ok", instrument=True):
        self.S = {"ep": set(S_ep), "gp": set(S_gp),
                  "eg": set(S_eg), "gg": set() if S_gg is None else set(S_gg)}
        self.d_sm = d_sm
        self.min_hits = {k: (MIN_HITS if self.S[k] else 0) for k in self.S}
        if min_hits:
            self.min_hits.update(min_hits)
        self.saturated = {k: True for k in self.S}
        if saturated:
            self.saturated.update(saturated)
        self.attached = attached
        self.capture = capture       # ok | partial | fail
        self.replay = replay         # ok | fail
        self.instrument = instrument  # P1: %smid site + spin back edge alive


def score(w, guards=frozenset()):
    """The registered rule. `guards` names guards to DISABLE (for mutation).

    ★Weakenings are expressed as separate guard names (round-3 D8: a
    deletion-only mutant suite could not see a WEAKENED guard, which the
    round-3 audit proved by weakening P5 and still getting ALL PASS).
    """
    S, D = w.S, w.S["ep"]

    def on(g):
        return g not in guards

    # --- control + baseline legs -------------------------------------------
    for leg in ("ep", "gp", "eg"):
        if on("P0") and (not S[leg] or w.min_hits[leg] < MIN_HITS):
            return ABSENT
    if on("P1") and not w.instrument:          # round-3 D3: P1 had been lost
        return ABSENT
    if on("P2") and not w.saturated["ep"]:
        return NOSAT
    if on("P3") and not w.saturated["gp"]:
        return NOSAT
    if on("NULL") and (S["gp"] ^ S["ep"]):
        return NULLCH
    if on("P4a") and not w.attached:
        return NOATT
    # ★ round-3 D1(b): TWO-SIDED. The upper bound alone let a baseline leg that
    #   realised only 8 of 34 labels through; a graph leg confined to the same
    #   8 then scored PRESERVED. The lower bound says the baseline must ACTUALLY
    #   realise the target partition before "confined to it" means anything.
    #   Both bounds come from the producer (d_sm, GRANULARITY).
    if on("P4b_hi") and len(S["eg"]) > w.d_sm + GRANULARITY:
        return NODEL
    if on("P4b_lo") and len(S["eg"]) < w.d_sm - GRANULARITY:
        return NODEL
    # --- tool outcomes, BEFORE any statement about the graph leg ------------
    if on("CAP") and w.capture != "ok":
        return NOCAP
    if on("REP") and w.replay != "ok":
        return NOREPLAY
    if on("P0g") and (not S["gg"] or w.min_hits["gg"] < MIN_HITS):
        return ABSENT
    if on("P5_eg") and not w.saturated["eg"]:
        return NOSAT
    if on("P5_gg") and not w.saturated["gg"]:
        return NOSAT
    # --- decision -----------------------------------------------------------
    E = S["gg"] - S["eg"]          # escape
    E_rev = S["eg"] - S["gg"]      # shortfall
    U = S["eg"] | S["gg"]          # ★ round-3 D1(a): the PAIR's reach
    # `LOST` requires the PAIR to reach outside a target-sized set. One label
    # seen by only one leg keeps the union inside the target: that is coverage
    # disagreement, not escape. The pre-repair rule turned it into the most
    # expensive verdict in the probe (immediate canon-review referral).
    if on("UNION"):
        if len(U) > w.d_sm + GRANULARITY:
            return LOST_FULL if S["gg"] >= D else LOST_PART
    elif E:                        # the pre-repair rule, kept for mutation
        return LOST_FULL if S["gg"] >= D else LOST_PART
    if E:
        return DISAGREE
    if on("REV") and E_rev:
        return UNSEP
    return PRESERVED


# ===========================================================================
# World space
# ===========================================================================
FULL = set(range(108))
GREEN = set(range(34))                      # the delivered decode half
NARROW = set(range(8))                      # round-2 C1's world
WIDE = GREEN | {40, 41, 42, 43}             # partial escape
ALL108 = set(FULL)
MISS1 = GREEN - {17}                        # ★round-3 D1(a): one label unseen


def worlds():
    gg_opts = {
        "empty": set(), "narrow": NARROW, "exact": GREEN,
        "escape_partial": WIDE, "escape_full": ALL108,
        "shifted": set(range(60, 94)),      # |S|=34 but different labels
        "miss1": MISS1,                     # ★round-3: graph missed one
    }
    # ★round-3 D1: the space had NO eager_green under-coverage world, so the
    #   rule's behaviour there had never been enumerated. The audit had to
    #   construct those worlds by hand -- which is exactly the failure this
    #   file exists to prevent.
    eg_opts = {"exact": GREEN, "empty": set(), "wide_107": set(range(107)),
               "miss1": MISS1, "narrow": NARROW}
    ctl = {"clean": (FULL, FULL), "gp_narrow": (FULL, set(range(100))),
           "ep_incomplete": (set(range(20)), set(range(20)))}   # ★round-3 D1
    ctl_sat = {"sat": (True, True), "ep_unsat": (False, True),
               "gp_unsat": (True, False)}
    for (gn, gg), (en, eg), (cn, (ep, gp)), (sn, (sep, sgp)) in itertools.product(
            gg_opts.items(), eg_opts.items(), ctl.items(), ctl_sat.items()):
        # ★round-3 D8 (second instance found by this file itself): `eg` and
        #   `gg` saturation used to move TOGETHER, so deleting either P5 half
        #   left the other firing and T4 saw no change. A bundled axis makes a
        #   guard look load-bearing-free when it is only redundant WITH ITS
        #   TWIN. Vary them independently.
        for att, cap, rep, s_eg, s_gg, ins in itertools.product(
                (True, False), ("ok", "fail"), ("ok", "fail"),
                (True, False), (True, False), (True, False)):
            name = (f"gg={gn},eg={en},ctl={cn},ctlsat={sn},att={att},"
                    f"cap={cap},rep={rep},sat_eg={s_eg},sat_gg={s_gg},instr={ins}")
            yield name, World(ep, gp, eg, gg, attached=att, capture=cap,
                              replay=rep, instrument=ins,
                              saturated={"gg": s_gg, "eg": s_eg,
                                         "ep": sep, "gp": sgp})


# ★ROUND-3 META-CONDITION. Every repair must be paired with a world in which a
#   mutant that REVERTS that repair produces a different (and wrong) label.
#   Three consecutive revisions had a repair create the next round's top
#   blocker; this table is the mechanical guard against a fourth.
#   (world, mutant-guard-set, label-with-repair, label-without-repair)
REPAIR_WITNESSES = [
    # ★The original round-1 B1 world (BOTH green legs empty) is now caught by
    #   TWO independent guards -- P0 and the new two-sided P4b_lo -- so it can
    #   no longer isolate P0. That is a strengthening, but it means the witness
    #   must be a world where ONLY the positivity floor stands between a
    #   measurement failure and a substantive label. An empty graph leg against
    #   a healthy baseline is that world: without P0g it is reported as the
    #   RESULT "the graph leg covered less", i.e. gate #21 exactly.
    ("round-1 B1 positivity floor",
     lambda: World(FULL, FULL, GREEN, set()), {"P0g"}, ABSENT, UNSEP),
    ("round-2 C1 symmetric difference (E_rev)",
     lambda: World(FULL, FULL, GREEN, GREEN - {1, 2, 3}), {"REV"}, UNSEP, PRESERVED),
    ("round-3 D1(a) union-based LOST",
     lambda: World(FULL, FULL, MISS1, GREEN), {"UNION"}, DISAGREE, LOST_PART),
    ("round-3 D1(b) two-sided P4b",
     lambda: World(FULL, FULL, NARROW, NARROW), {"P4b_lo"}, NODEL, PRESERVED),
    ("round-3 D3 P1 restored",
     lambda: World(FULL, FULL, GREEN, GREEN, instrument=False), {"P1"},
     ABSENT, PRESERVED),
]

# ★Guards must also be tested under WEAKENING, not only deletion (round-3 D8:
#   the audit weakened P5 to look at `gg` only and the suite still passed).
WEAKENINGS = [("P5 sees only the graph leg", {"P5_eg"}),
              ("P4b upper bound only", {"P4b_lo"}),
              ("P4b lower bound only", {"P4b_hi"}),
              ("decision ignores the shortfall", {"REV"}),
              ("decision ignores the union", {"UNION"})]


def run():
    ok = True

    def chk(c, m):
        nonlocal ok
        print(("  [PASS] " if c else "  [FAIL] ") + m)
        ok = ok and c

    chk(PRODUCER_OK, f"producer constants imported (MIN_HITS={MIN_HITS}, "
                     f"GRANULARITY={GRANULARITY})"
                     + ("" if PRODUCER_OK else f" -- FELL BACK: {PRODUCER_ERR}"))

    ws = list(worlds())
    print(f"-- enumerated {len(ws)} worlds")

    labels = {}
    bad = []
    for n, w in ws:
        lb = score(w)
        labels[n] = lb
        if lb not in LABELS:
            bad.append((n, lb))
    chk(not bad, f"T1 every world gets a registered label ({len(bad)} bad)")

    viol = [n for n, w in ws
            if (not w.S["gg"] or not w.S["eg"] or not w.saturated["gg"]
                or not w.saturated["eg"] or not w.attached or not w.instrument
                or w.capture != "ok" or w.replay != "ok")
            and labels[n] in SUBSTANTIVE]
    chk(not viol, f"T2 no degenerate world reaches a substantive label "
                  f"({len(viol)} leak(s))" + (f" e.g. {viol[0]}" if viol else ""))
    # ★round-3 D8: T2 must be shown CAPABLE of failing, inside the suite.
    ordering_break = sum(
        1 for n, w in ws
        if score(w, guards={"P5_eg", "P5_gg"}) in SUBSTANTIVE
        and (not w.saturated["gg"] or not w.saturated["eg"]))
    chk(ordering_break > 0,
        f"T2' T2 is capable of failing: dropping P5 leaks {ordering_break} "
        f"unsaturated world(s) into substantive labels")

    for nm, wf, mut, want, want_mut in REPAIR_WITNESSES:
        w = wf()
        chk(score(w) == want, f"T2r {nm}: witness world -> {want!r}")
        chk(score(w, guards=mut) == want_mut,
            f"T5  {nm}: reverting mutant -> {want_mut!r} (repair is load-bearing)")

    produced = set(labels.values())
    missing = LABELS - produced
    chk(not missing, f"T3 every registered label is reachable "
                     f"({len(missing)} dead)" + (f": {sorted(missing)}" if missing else ""))

    print("-- T4 mutation: each guard is load-bearing under DELETION")
    for g in ("P0", "P1", "P2", "P3", "NULL", "P4a", "P4b_hi", "P4b_lo",
              "CAP", "REP", "P0g", "P5_eg", "P5_gg", "UNION", "REV"):
        changed = sum(1 for n, w in ws if score(w, guards={g}) != labels[n])
        chk(changed > 0, f"   delete {g:7s} changes {changed} world(s)")

    print("-- T4' mutation: each guard is load-bearing under WEAKENING")
    for nm, gs in WEAKENINGS:
        changed = sum(1 for n, w in ws if score(w, guards=gs) != labels[n])
        chk(changed > 0, f"   weaken: {nm:38s} changes {changed} world(s)")

    print("ALL PASS" if ok else "RULE TOTALITY CHECK FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.parse_args()
    sys.exit(run())
