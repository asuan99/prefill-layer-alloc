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
UNDER = "UNDER_COVERAGE_IN_GRAPH_LEG (RESULT)"
ABSENT = "UNDETERMINED (MEASUREMENT ABSENT)"
NOSAT = "UNDETERMINED (COVERAGE NOT SATURATED)"
NOATT = "UNDETERMINED (MEASUREMENT ABSENT: GREEN CONTEXT NOT ATTACHED)"
NODEL = "GREEN PARTITION NOT DELIVERED (RESULT)"
NULLCH = "UNDETERMINED (DECISION QUANTITY'S NULL CHANNEL OPEN)"
NOCAP = "UNDETERMINED (GRAPH CAPTURE UNAVAILABLE ON GREEN STREAM)"
NOREPLAY = "UNDETERMINED (GRAPH REPLAY FAILED)"
SUBSTANTIVE = {PRESERVED, LOST_FULL, LOST_PART, UNDER}
LABELS = SUBSTANTIVE | {ABSENT, NOSAT, NOATT, NODEL, NULLCH, NOCAP, NOREPLAY}

MIN_HITS = 1          # CEN.MIN_HITS_REPORTED -- not a threshold, the 0/1 boundary
GRANULARITY = 2       # get_arch_constraints((8,0))[1] -- from the producer


class World:
    """A world = what each leg observed + tool outcomes. Sets are label sets."""

    def __init__(self, S_ep, S_gp, S_eg, S_gg, d_sm=34,
                 min_hits=None, saturated=None, attached=True,
                 capture="ok", replay="ok"):
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


def score(w, guards=frozenset()):
    """The registered rule. `guards` names guards to DISABLE (for mutation)."""
    S, D = w.S, w.S["ep"]

    def on(g):
        return g not in guards

    # --- order is part of the rule (round-2 C5) -----------------------------
    # Legs that exist regardless of the graph-capture outcome come first.
    for leg in ("ep", "gp", "eg"):
        if on("P0") and (not S[leg] or w.min_hits[leg] < MIN_HITS):
            return ABSENT
    if on("P2") and not w.saturated["ep"]:
        return NOSAT
    if on("P3") and not w.saturated["gp"]:
        return NOSAT
    # NULL CHANNEL: the decision function itself, run on the control pair,
    # where the answer must be the empty symmetric difference.
    if on("NULL") and (S["gp"] ^ S["ep"]):
        return NULLCH
    if on("P4a") and not w.attached:
        return NOATT
    if on("P4b") and not (len(S["eg"]) <= w.d_sm + GRANULARITY):
        return NODEL
    # --- tool outcomes: BEFORE any statement about the graph leg ------------
    if on("CAP") and w.capture != "ok":
        return NOCAP
    if on("REP") and w.replay != "ok":
        return NOREPLAY
    # --- now the graph leg may be spoken about ------------------------------
    if on("P0g") and (not S["gg"] or w.min_hits["gg"] < MIN_HITS):
        return ABSENT
    if on("P5") and not (w.saturated["eg"] and w.saturated["gg"]):
        return NOSAT
    # --- decision: the SYMMETRIC difference (round-2 C1) --------------------
    E = S["gg"] - S["eg"]          # escape: replay went outside
    E_rev = S["eg"] - S["gg"]      # shortfall: replay covered less
    if E:
        return LOST_FULL if S["gg"] >= D else LOST_PART
    if on("REV") and E_rev:
        return UNDER
    return PRESERVED


# ===========================================================================
# World space
# ===========================================================================
FULL = set(range(108))
GREEN = set(range(34))                      # the delivered decode half
NARROW = set(range(8))                      # round-2 C1's world
WIDE = GREEN | {40, 41, 42, 43}             # partial escape
ALL108 = set(FULL)


def worlds():
    gg_opts = {
        "empty": set(), "narrow": NARROW, "exact": GREEN,
        "escape_partial": WIDE, "escape_full": ALL108,
        "shifted": set(range(60, 94)),      # |S|=34 but different labels
    }
    eg_opts = {"exact": GREEN, "empty": set(), "wide_107": set(range(107))}
    ctl = {"clean": (FULL, FULL), "gp_narrow": (FULL, set(range(100)))}
    # the control legs must also be allowed to be UNSATURATED, or the P2/P3
    # guards are never exercised and T4 cannot see them (the world SPACE, not
    # the rule, was the defect the first time this file was run).
    ctl_sat = {"sat": (True, True), "ep_unsat": (False, True),
               "gp_unsat": (True, False)}
    for (gn, gg), (en, eg), (cn, (ep, gp)), (sn, (sep, sgp)) in itertools.product(
            gg_opts.items(), eg_opts.items(), ctl.items(), ctl_sat.items()):
        for att, cap, rep, sat in itertools.product(
                (True, False), ("ok", "fail"), ("ok", "fail"),
                (True, False)):
            name = (f"gg={gn},eg={en},ctl={cn},ctlsat={sn},att={att},"
                    f"cap={cap},rep={rep},sat={sat}")
            yield name, World(ep, gp, eg, gg, attached=att, capture=cap,
                              replay=rep,
                              saturated={"gg": sat, "eg": sat,
                                         "ep": sep, "gp": sgp})


def run():
    ok = True

    def chk(c, m):
        nonlocal ok
        print(("  [PASS] " if c else "  [FAIL] ") + m)
        ok = ok and c

    ws = list(worlds())
    print(f"-- enumerated {len(ws)} worlds")

    # T1 totality / single label
    labels = {}
    bad = []
    for n, w in ws:
        lb = score(w)
        labels[n] = lb
        if lb not in LABELS:
            bad.append((n, lb))
    chk(not bad, f"T1 every world gets a registered label ({len(bad)} bad)")

    # T2 no vacuity: degenerate observations must never reach a substantive label
    viol = [n for n, w in ws
            if (not w.S["gg"] or not w.S["eg"] or not w.saturated["gg"]
                or not w.saturated["eg"] or not w.attached
                or w.capture != "ok" or w.replay != "ok")
            and labels[n] in SUBSTANTIVE]
    chk(not viol, f"T2 no degenerate world reaches a substantive label "
                  f"({len(viol)} leak(s))" + (f" e.g. {viol[0]}" if viol else ""))

    # ★the two historical defects, named
    w_b1 = World(FULL, FULL, set(), None)                       # round-1 B1
    chk(score(w_b1) == ABSENT, "T2a round-1 B1 world (both green legs empty) -> ABSENT")
    w_c1 = World(FULL, FULL, GREEN, NARROW)                     # round-2 C1
    chk(score(w_c1) == UNDER,
        "T2b round-2 C1 world (graph leg too NARROW) -> UNDER_COVERAGE, not PRESERVED")
    w_sh = World(FULL, FULL, GREEN, set(range(60, 94)))         # |S| equal, disjoint
    chk(score(w_sh) == LOST_PART,
        "T2c equal-cardinality-but-disjoint world -> LOST (PARTIAL)")

    # T3 reachability: every label must be produced by some world
    produced = set(labels.values())
    missing = LABELS - produced
    chk(not missing, f"T3 every registered label is reachable "
                     f"({len(missing)} dead)" + (f": {sorted(missing)}" if missing else ""))

    # T4 load-bearing: deleting a guard must change some world's label
    print("-- T4 mutation: each guard is load-bearing")
    for g in ("P0", "P2", "P3", "NULL", "P4a", "P4b", "CAP", "REP", "P0g", "P5", "REV"):
        changed = sum(1 for n, w in ws if score(w, guards={g}) != labels[n])
        chk(changed > 0, f"   guard {g:5s} deletion changes {changed} world(s)")

    print("ALL PASS" if ok else "RULE TOTALITY CHECK FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.parse_args()
    sys.exit(run())
