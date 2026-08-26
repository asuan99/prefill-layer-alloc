#!/usr/bin/env python3
"""NSL **E-B** decision rule — what blocked admission, the cap path or the KV path.

SCOPE, and why it is smaller than the track's hypothesis
-------------------------------------------------------
NSL-1's registered hypothesis is (b)-class: *"tune the admission cap to buy SLO
goodput"*.  This campaign does **not** test that.  Audit B1 invalidated the point
predicate (*"cap is binding"*) because `running_bs >= cap` is not the firing
condition, and step ① could only bracket the true predicate to [0.080, 0.119].
What ②(per-site counters) + ③(knob purity) buy is **attribution**: of the
admission blocks that actually fired, which SITE fired.

★So the estimand is one fraction, at ONE cap:

    cap_share = (firings at the cap site) / (firings at cap site + KV site)

★**The cap AXIS is deliberately not in this campaign.**  A sweep over cap is
what the lever claim needs, it is what B3 contaminates, and it costs 3x the
boots.  Buying attribution first makes the expensive sweep *conditional on a
cheap answer* — if the KV path dominates at the canonical cell, the cap sweep
was never the experiment to run.

Registered inheritance: ③'s rules R1-R6
(`PREREG_NSL_STEP3_KNOB_PURITY_2026-08-25.md`) apply to every boot here — the
mamba pool is pinned, the banner triple is recorded, and a cell whose realized
cap differs from the requested one is dropped.  ★②+③ ship together or not at
all (`MEMO_NSL_STEP2_DECISION_2026-08-24.md` §4).

Run:  python3 nsl_eb_rule.py
"""
import hashlib, json, os, sys
from collections import Counter
from itertools import product

RULE_REV = 1

# --- tier 1: MEASUREMENT -----------------------------------------------------
BOOT      = "BOOT_FAILED"
CFGDRIFT  = "CONFIG_DRIFT"           # a site that must be closed in this build fired
CAPCLAMP  = "CAP_CLAMPED"            # realized cap != requested (③ R4)
KVUNEQUAL = "KV_BUDGET_UNEQUAL"      # max_total_num_tokens differs across cells (③ R5)
COUNTDEAD = "COUNTER_DEAD"           # the patch is off, or never fired
NOTEL     = "TELEMETRY_ABSENT"
SHORT     = "TOO_FEW_FIRINGS"       # ★renamed: `SPAN_TOO_SHORT` is the Q3
                                    # rule's string for a different quantity
                                    # (decode steps), and A1's B7 was exactly a
                                    # label reused with a changed meaning.
# ★NEW, applied BEFORE the audit from the A1 chain's recurring families.
# `NEITHER_PATH_FIRES` was substantive, so a run whose workload never reached
# the cap-bound regime -- an experiment-design fact -- read as an answer about
# which path binds.  That is gate #21 in this track's clothing, and the A1
# chain paid for the same shape three times (Q3 `launches`, primary `rows_s`,
# primary (matched, null/zero)).  ★Carrying a lesson across tracks instead of
# re-buying it is the whole point of naming it.
NOSAT     = "WORKLOAD_NOT_SATURATING"
MEASUREMENT = {BOOT, CFGDRIFT, CAPCLAMP, KVUNEQUAL, COUNTDEAD, NOTEL, SHORT,
               NOSAT}

# --- tier 2: TOOLLIMIT -------------------------------------------------------
# ★The instrument, not the engine: `batch_is_full` was observed True while no
# instrumented site incremented.  That means the site list is incomplete for
# this build -- a fact about the patch.  Scoring it as "neither path fired"
# would be gate #21 in this track.
INCOMPLETE = "SITES_INCOMPLETE"
TOOLLIMIT = {INCOMPLETE}

# --- tier 3: SUBSTANTIVE -----------------------------------------------------
CAPDOM   = "CAP_PATH_DOMINANT"
KVDOM    = "KV_PATH_DOMINANT"
MIXED    = "BOTH_PATHS_MATERIAL"
NOBLOCK  = "NEITHER_PATH_FIRES"      # admission was never blocked in this run
SUBSTANTIVE = {CAPDOM, KVDOM, MIXED, NOBLOCK}

LABELS = MEASUREMENT | TOOLLIMIT | SUBSTANTIVE

# --- registered constants ----------------------------------------------------
# ★[ARBITRARY].  A share this far from 0.5 is called dominant; between the two
# the run says BOTH paths are material.  Registered BEFORE any counter exists,
# which is the only reason it can be called pre-registered at all.
DOMINANT_AT = 0.80
# ★minimum firings before a share is computed.  A share of 3/4 is not a share.
N_MIN_FIRINGS = 100


class World:
    """One BOOT of the E-B campaign.

    boot_ok      the server came up
    tel          engine telemetry present
    counters     the per-site counter payload: ok | absent | zero
    closed_sites did a site that this build closes fire?  none | fired
                 (③/②: `:2369` chunked-prefill, `:2439` disaggregation,
                  `:2472` hierarchical cache are shut under --enable-pdmux)
    realized_cap requested | clamped          (③ R4)
    kv_budget    equal | unequal              (③ R5, across the cells compared)
    n_firings    ★INTEGER -- cap-site + KV-site firings in the analysed span
    saturated    did the run REACH the cap-bound regime at all?  yes | no
                 ★Read from telemetry independently of the counters:
                 `running_bs` reaching the realized cap at any point.  Without
                 this axis "nothing blocked" and "the workload never created
                 the condition" are the same label.
    unattributed were there `batch_is_full` observations no site explains?
    cap_share    the estimand, banded: 0.0 | 0.5 | 0.85 | 1.0
    """

    __slots__ = ("boot_ok", "tel", "counters", "closed_sites", "realized_cap",
                 "kv_budget", "saturated", "n_firings", "unattributed",
                 "cap_share")

    def __init__(self, boot_ok=True, tel="ok", counters="ok",
                 closed_sites="none", realized_cap="requested",
                 kv_budget="equal", saturated="yes", n_firings=4000,
                 unattributed="no", cap_share=1.0):
        self.boot_ok, self.tel, self.counters = boot_ok, tel, counters
        self.closed_sites, self.realized_cap = closed_sites, realized_cap
        self.kv_budget, self.saturated = kv_budget, saturated
        self.n_firings = n_firings
        self.unattributed, self.cap_share = unattributed, cap_share

    def __repr__(self):
        return (f"W(boot={self.boot_ok},tel={self.tel},cnt={self.counters},"
                f"closed={self.closed_sites},cap={self.realized_cap},"
                f"kv={self.kv_budget},sat={self.saturated},n={self.n_firings},"
                f"unattr={self.unattributed},share={self.cap_share})")


AXES = dict(
    boot_ok=[True, False], tel=["ok", "absent"],
    counters=["ok", "absent", "zero"],
    closed_sites=["none", "fired"],
    realized_cap=["requested", "clamped"],
    kv_budget=["equal", "unequal"],
    saturated=["yes", "no"],
    n_firings=[0, 4, 99, 100, 4000],
    unattributed=["no", "yes"],
    # ★0.6/0.75 sit BETWEEN the registered 0.80 and the mutant's 0.55, which
    # is the only reason `g_dom_value` can move a label at all.  The first
    # run of this suite had neither and reported the mutant inert.
    cap_share=[0.0, 0.15, 0.5, 0.6, 0.75, 0.85, 1.0],
)


def score(w, guards=frozenset()):
    """Three tiers, decided in order.  ★The middle tier is the one this project
    keeps dropping and then re-opening gate #21 -- see the A1 chain."""
    def on(g):
        return g not in guards

    if on("g_boot") and not w.boot_ok:
        return BOOT
    if on("g_tel") and w.tel != "ok":
        return NOTEL
    # ③'s boot rules, decided before anything is attributed
    if on("g_cfg") and w.closed_sites != "none":
        return CFGDRIFT
    if on("g_clamp") and w.realized_cap != "requested":
        return CAPCLAMP
    if on("g_kv") and w.kv_budget != "equal":
        return KVUNEQUAL
    if on("g_dead") and w.counters != "ok":
        return COUNTDEAD
    # ★the workload gate is TIER 1, ahead of the instrument tier: `saturated`
    # comes from telemetry (`running_bs` vs the realized cap) and does not
    # depend on the counters, so it is the more primitive fact.  If the regime
    # was never reached, nothing about the instrument is under test -- and
    # "nothing blocked" would be an answer manufactured out of a workload that
    # never created the condition.
    if on("g_nosat") and w.saturated != "yes":
        return NOSAT
    # ★the tool tier: an unexplained block means the SITE LIST is incomplete.
    # It must be decided BEFORE `n_firings`, or a build whose real blocking site
    # is uninstrumented reads as "few firings" and then as a share.
    if on("g_incomplete") and w.unattributed == "yes":
        return INCOMPLETE
    if on("g_noblock") and w.n_firings == 0:
        return NOBLOCK
    nmin = N_MIN_FIRINGS if on("g_nmin_value") else 2
    if on("g_short") and w.n_firings < nmin:
        return SHORT
    hi = DOMINANT_AT if on("g_dom_value") else 0.55
    if on("g_capdom") and w.cap_share >= hi:
        return CAPDOM
    if on("g_kvdom") and w.cap_share <= 1.0 - hi:
        return KVDOM
    return MIXED


MUTANTS = ["g_boot", "g_tel", "g_cfg", "g_clamp", "g_kv", "g_dead", "g_nosat",
           "g_incomplete", "g_noblock", "g_short", "g_nmin_value",
           "g_capdom", "g_kvdom", "g_dom_value"]

GUARD_LABEL = {"g_boot": BOOT, "g_tel": NOTEL, "g_cfg": CFGDRIFT,
               "g_clamp": CAPCLAMP, "g_kv": KVUNEQUAL, "g_dead": COUNTDEAD,
               "g_incomplete": INCOMPLETE, "g_noblock": NOBLOCK,
               "g_nosat": NOSAT,
               "g_short": SHORT, "g_capdom": CAPDOM, "g_kvdom": KVDOM}

# ★REGISTERED STOPS -- the prereg's stop table as data.  The A1 chain showed
# that checks derived from the same prose as the rule move with the prose; a
# registry is the oracle that does not.  ★It raises a two-copy edit to a
# three-copy edit and makes the third one a visible change to what was
# registered.  It does not make self-checking sound.
REGISTERED_STOPS = (
    (dict(boot_ok=False), BOOT),
    (dict(tel="absent"), NOTEL),
    (dict(closed_sites="fired"), CFGDRIFT),
    (dict(realized_cap="clamped"), CAPCLAMP),
    (dict(kv_budget="unequal"), KVUNEQUAL),
    (dict(counters="absent"), COUNTDEAD),
    (dict(counters="zero"), COUNTDEAD),
    (dict(unattributed="yes"), INCOMPLETE),
    (dict(saturated="no"), NOSAT),
    (dict(n_firings=0), NOBLOCK),
    (dict(n_firings=4), SHORT),
    (dict(n_firings=99), SHORT),
    (dict(cap_share=0.0), KVDOM),
    (dict(cap_share=0.5), MIXED),
    (dict(cap_share=0.85), CAPDOM),
)


def worlds():
    keys = list(AXES)
    for c in product(*(AXES[k] for k in keys)):
        yield World(**dict(zip(keys, c)))


def _plumbing_dirty(w):
    """The boot-level conditions only.  ★The firing floor is NOT here: it is
    decided after attribution completeness, so folding it in makes a check
    contradict the rule's registered order."""
    return (not w.boot_ok or w.tel != "ok" or w.closed_sites != "none"
            or w.realized_cap != "requested" or w.kv_budget != "equal"
            or w.counters != "ok" or w.saturated != "yes")


def _checks():
    def c_tier(w, l):
        """A. TWO projections, and deliberately silent about the firing floor.

        ★The first version folded the floor in and failed on a world that is
        short AND unattributed: the rule decides the tool tier first, on
        purpose, because a build whose real blocking site is uninstrumented
        otherwise reads as "few firings" and then as a share.  The check was
        wrong, not the rule.  The floor belongs to C.
        """
        if _plumbing_dirty(w):
            return l in MEASUREMENT
        if w.unattributed == "yes":
            return l not in SUBSTANTIVE
        return True

    def c_witness(w, l):
        """B. every substantive label carries a necessary condition."""
        if l == CAPDOM:
            return w.cap_share >= 0.80
        if l == KVDOM:
            return w.cap_share <= 0.20
        if l == MIXED:
            return 0.20 < w.cap_share < 0.80 and w.n_firings >= N_MIN_FIRINGS
        if l == NOBLOCK:
            # ★"nothing fired" only means something once the regime was reached
            return w.n_firings == 0 and w.saturated == "yes"
        return True

    def c_threshold(w, l):
        """C. the firing floor as a literal, on otherwise-clean worlds."""
        if (not w.boot_ok or w.tel != "ok" or w.closed_sites != "none"
                or w.realized_cap != "requested" or w.kv_budget != "equal"
                or w.counters != "ok" or w.saturated != "yes"
                or w.unattributed == "yes"):
            return True
        return (l == SHORT) == (0 < w.n_firings < 100)

    # ★A FOURTH CHECK WAS WRITTEN AND DELETED, with the reason recorded rather
    # than an artificial mutant invented for it (A0 rev4 removed T22 the same
    # way).  It asserted that mirroring `cap_share` mirrors the dominance
    # label -- a real property, and one that per-label witnesses already imply:
    # with a single scalar estimand and literal thresholds, any asymmetry that
    # breaks the mirror also violates one side's necessary condition, so B
    # entailed it (T19a) and its sole binding was 0.  Symmetry is worth
    # checking when the estimand is not a single scalar; here it is not.

    return {
        "A tier discipline": (c_tier, {"g_boot", "g_tel", "g_cfg", "g_clamp",
                                       "g_kv", "g_dead", "g_incomplete",
                                       "g_nosat"}),
        "B substantive witness": (c_witness, {"g_capdom", "g_kvdom",
                                              "g_dom_value"}),
        "C firing floor = 100 (literal)": (c_threshold, {"g_short",
                                                         "g_nmin_value",
                                                         "g_noblock"}),
    }


# =============================================================================
# CAMPAIGN LEVEL -- the prereg registers 4 boots per rate and says, in prose,
# that boots disagreeing is itself reportable.  ★Prose is where the A1 chain
# lost three audits: the world model tracked one boot while the design grew to
# five.  The aggregation is code here, with its own labels and its own checks.
# =============================================================================
N_MIN_BOOTS = 4                      # prereg sec 3; gate #3 (n>=4)

AGG_FEW      = "INSUFFICIENT_BOOTS"          # too few scorable boots
AGG_TOOL     = "TOOL_LIMIT_IN_CAMPAIGN"      # a tool limit does not average away
AGG_DISAGREE = "BOOTS_DISAGREE"
AGG_LABELS = {AGG_FEW, AGG_TOOL, AGG_DISAGREE} | SUBSTANTIVE


def aggregate(boot_labels, guards=frozenset()):
    """One cell's verdict from its boots' labels.

    ★Three rules, and the middle one is the one worth arguing about:
      1. MEASUREMENT boots delivered no observation -> DROP them, then require
         `N_MIN_BOOTS` of the rest.
      2. ★A TOOLLIMIT boot is NOT dropped.  "The instrument could not do it"
         does not become false by averaging it with boots where it could; it is
         a campaign-level fact and it is reported as one.  Dropping it would be
         the aggregation-layer version of gate #21.
      3. The surviving substantive boots must AGREE.  Disagreement is a result,
         not noise to be voted away -- this project has been burned by
         majority-style summaries of bimodal populations (metric cliff).
    """
    def on(g):
        return g not in guards
    labels = list(boot_labels)
    if on("a_tool") and any(l in TOOLLIMIT for l in labels):
        return AGG_TOOL
    scorable = [l for l in labels if l in SUBSTANTIVE]
    if on("a_few") and len(scorable) < N_MIN_BOOTS:
        return AGG_FEW
    if on("a_disagree") and len(set(scorable)) > 1:
        return AGG_DISAGREE
    # ★totality must survive the mutants too: with `a_few` removed, `scorable`
    # can be empty, and a rule that raises under its own mutant is not a rule.
    return scorable[0] if scorable else AGG_FEW


AGG_MUTANTS = ["a_tool", "a_few", "a_disagree"]


def _agg_selftest(chk):
    """Enumerate every multiset of 4-6 boot labels over a small alphabet."""
    from itertools import combinations_with_replacement as cwr
    alphabet = [CAPDOM, KVDOM, MIXED, NOBLOCK, BOOT, NOSAT, INCOMPLETE]
    cases = [list(c) for n in (3, 4, 5) for c in cwr(alphabet, n)]
    outs = [aggregate(c) for c in cases]
    chk("AGG T1 totality", all(o in AGG_LABELS for o in outs))
    from collections import Counter as _C
    cnt = _C(outs)
    for lab in sorted(AGG_LABELS):
        chk(f"AGG T2 reachable: {lab}", cnt[lab] > 0, f"n={cnt[lab]}")
    for m in AGG_MUTANTS:
        d = sum(1 for a, b in zip(outs, [aggregate(c, {m}) for c in cases])
                if a != b)
        chk(f"AGG T3 mutant {m} load-bearing", d > 0, f"n={d}")
    # ★the three registered properties, each as its own witness
    chk("AGG P1 a tool limit anywhere surfaces",
        all(aggregate(c) == AGG_TOOL for c in cases
            if any(l in TOOLLIMIT for l in c)))
    chk("AGG P2 measurement boots are dropped, not counted",
        aggregate([CAPDOM] * 4 + [BOOT, NOSAT]) == CAPDOM)
    chk("AGG P3 disagreement is a result, not a majority vote",
        aggregate([CAPDOM] * 5 + [KVDOM]) == AGG_DISAGREE)
    chk("AGG P4 n<4 scorable boots is insufficient",
        aggregate([CAPDOM] * 3 + [BOOT]) == AGG_FEW)


def run():
    print(f"== NSL E-B rule (RULE_REV={RULE_REV}) ==")
    W = list(worlds())
    base = [score(w) for w in W]
    checks = _checks()
    out, fails = {}, []

    def chk(n, c, d=""):
        print(f"  [{'PASS' if c else 'FAIL'}] {n} {d}")
        out[n] = bool(c)
        if not c:
            fails.append(n)

    chk("T1 totality", all(l in LABELS for l in base))
    chk("T1b determinism", base == [score(w) for w in W])
    cnt = Counter(base)
    for lab in sorted(LABELS):
        chk(f"T2 reachable: {lab}", cnt[lab] > 0, f"n={cnt[lab]}")
    mut = {m: [score(w, {m}) for w in W] for m in MUTANTS}
    for m in MUTANTS:
        d = sum(1 for a, b in zip(base, mut[m]) if a != b)
        chk(f"T3 mutant {m} load-bearing", d > 0, f"n={d}")
    for n, (fn, _) in checks.items():
        chk(f"{n} [DISCRIM]", all(fn(w, l) for w, l in zip(W, base)))
    print("  -- T8 meta --")
    for n, (fn, ms) in checks.items():
        for m in sorted(ms):
            chk(f"T8 '{n[:22]}' fails under {m}",
                any(not fn(w, l) for w, l in zip(W, mut[m])))
    print("  -- T26 meta (registered stops) --")
    for kwargs, want in REGISTERED_STOPS:
        got = score(World(**kwargs))
        chk(f"T26 {sorted(kwargs.items())[:1]} -> {want}", got == want, f"got {got}")
    print("  -- T25 meta (each guard's own label disappears without it) --")
    for g, lab in sorted(GUARD_LABEL.items()):
        after = Counter(mut[g])[lab]
        chk(f"T25 {g} produces {lab}", after < cnt[lab], f"{cnt[lab]} -> {after}")
    print("  -- T19 meta --")
    ASG = [(w, score(w, frozenset({m}) if m else frozenset()))
           for m in [None] + MUTANTS for w in W]
    fs = {n: {i for i, (w, l) in enumerate(ASG) if not fn(w, l)}
          for n, (fn, _) in checks.items()}
    ent = [f"{a}=>{b}" for a in checks for b in checks
           if a != b and fs[b] and fs[b] <= fs[a]]
    chk("T19a no check entailed by another", not ent, f"{ent or 'none'}")
    sole = {n: len(fs[n] - set().union(*[fs[o] for o in checks if o != n]))
            for n in checks}
    print(f"     sole-binding: { {n[:12]: sole[n] for n in checks} }")
    chk("T19b no check binds nothing", all(v > 0 for v in sole.values()),
        f"{sole}")

    print("  -- campaign aggregation --")
    _agg_selftest(chk)
    ok = not fails
    print(f"\n== {len(W):,} worlds -> "
          f"{'ALL PASS' if ok else str(len(fails)) + ' FAILURES: ' + '; '.join(fails[:4])} ==")
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.abspath(__file__), "rb") as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()
    with open(os.path.join(here, "selftest_nsl_eb_2026-08-25.json"), "w") as fh:
        json.dump({"rule_rev": RULE_REV, "rule_sha256": sha,
                   "date": "2026-08-25", "gpu_hr": 0.0, "worlds": len(W),
                   "labels": dict(cnt), "sole_binding": sole,
                   "dominant_at": DOMINANT_AT,
                   "n_min_firings": N_MIN_FIRINGS,
                   "checks": out, "all_pass": ok}, fh, indent=1, sort_keys=True)
    print("   wrote selftest_nsl_eb_2026-08-25.json")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(run())
