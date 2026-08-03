#!/usr/bin/env python3
"""M3 decision rule, IN CODE, written BEFORE the job runs.

DESIGN.md sec 4.3.8(c) pre-registers the verdict as a function of

    g(arm, block) = A_free(d16) / A_free(d54)

This file exists because of what M1 found on 2026-08-02: sec 4.3.5(b)'s
exclusion rule called itself "a mechanical function of the measured knee" while
the knee itself was hand-computed and lived in no script, so an auditor had to
reverse-engineer it. Repeating that here -- computing g by hand from the
M3PROBE lines -- would reproduce the exact defect this campaign just paid to
fix. The thresholds below are the pre-registered ones; they may not be re-cut
after seeing data (sec 4.3.8(c), last row of the verdict table).

Reads the `M3PROBE` lines emitted by e1_m3_control.sbatch's in-run gate.

Usage:
  python3 m3_analyze.py --dir . --job 87xxxx
"""
import argparse
import collections
import glob
import os
import random
import re
import math
import statistics

# --- PRE-REGISTERED (DESIGN.md sec 4.3.8(c)) --- do not edit after the run ---
G_LEVER = 1.50        # "responds to D"
G_FLAT = 1.15         # "does not respond to D"
NUM_CELL = "d16"      # numerator cell of g
DEN_CELL = "d54"      # denominator cell of g
CONTROL_ARM = "T8"    # positive control (pure Transformer)
TEST_ARM = "Ha8"      # arm under test (hybrid)
N_MIN_INDEP = 6       # refuse a verdict below this many independent blocks

# ★2026-08-02: the interval is a t-interval over BLOCKS, not a percentile
# bootstrap over seeds.  Two separate reasons, both measured before the run:
#
# 1. WRONG UNIT.  g = A_free(d16)/A_free(d54) pairs two cells measured in
#    DIFFERENT boots, so boot-to-boot variance enters g instead of cancelling.
#    Seeds sharing one boot are therefore not independent replicates of g.  The
#    replication unit is the BLOCK (one fresh boot of every cell at one seed).
#    The first draft of this campaign's design had 4 seeds inside 2 boot-pairs,
#    i.e. n_independent = 2, which tolerates sd(g) <= 0.021 -- it could never
#    have fired.
#
# 2. ANTI-CONSERVATIVE INTERVAL.  Simulated coverage of a nominal 95% interval
#    (4000 trials, normal data):
#        n= 4  percentile bootstrap 79.8% (width 1.52) | t 94.5% (width 2.93)
#        n= 6  85.0%                                   | t 95.0%
#        n= 8  89.0% (width 1.24)                      | t 95.7% (width 1.63)
#    At these n the percentile bootstrap is roughly half the width it should be,
#    so "the CIs are disjoint" would fire far too easily -- on a rule whose
#    whole job is to decide a campaign.  The bootstrap is still PRINTED for
#    continuity with the rest of the campaign, but the verdict reads the t
#    interval.
TCRIT = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447,
         8: 2.365, 9: 2.306, 10: 2.262, 12: 2.201, 16: 2.131}
N_BOOT = 10000
BOOT_SEED = 20260802

PROBE = re.compile(
    r"M3PROBE arm=(?P<arm>\w+) cell=(?P<cell>d\d+) block=(?P<block>\w+) "
    r"seed=(?P<seed>\d+) n_keep=(?P<n_keep>\d+) n_err=(?P<n_err>\d+) "
    r"conc=(?P<conc>\S+) cap=(?P<cap>\d+) "
    r"ttft_p50=(?P<ttft_p50>\S+) ttft_p95=(?P<ttft_p95>\S+) plateau=(?P<plateau>\S+) "
    r"A_all=(?P<A_all>\S+) A_free=(?P<A_free>\S+) "
    r"n_blocking_windows=(?P<nbw>\d+) "
    r"GATE_CONC=(?P<g_conc>\w+) GATE_OFFCLIFF=(?P<g_cliff>\w+) GATE_ERR=(?P<g_err>\w+)"
)


def load_pin(dirpath, job):
    """★2026-08-02 (job 872077 postmortem). Read m3_pin_check.sh's output and
    return {(arm, cell, block): passed}.

    This wiring was MISSING when 872077 landed, and the omission produced
    exactly the failure this campaign keeps paying for: the pin gate voided
    d16 -- g's numerator -- on both arms, and this analyzer, knowing nothing
    about it, printed a confident "ASYMMETRY ATTRIBUTABLE TO THE ARM" anyway.
    A cite-blocking gate that the deciding script cannot see is not a gate.

    Absence is NOT treated as success: with no pin file the verdict is refused,
    because "the gate was never run" and "the gate passed" must never look the
    same (PROJECT_STATUS.md methodology gate #4's lineage -- silent population
    substitution)."""
    path = os.path.join(dirpath, f"m3_pin_{job}.txt")
    if not os.path.exists(path):
        return None
    # ★2026-08-03: two formats. The CURRENT one is the conditional gate
    # (sec 4.3.8(h)); the LEGACY one is the identity gate that scored the
    # designed decode-empty auto-revert as a failure. Prefer conditional when
    # present, fall back to legacy, and say which was used -- silently reading
    # whichever matched is how the wrong gate voided M3 in the first place.
    cond = re.compile(r"\s+e1m3_(\w+?)_(d\d+)_(b\d+)_\d+\s+COND_PIN=([\d.]+)\s*->\s*(PASS|FAIL)")
    legacy = re.compile(r"\s+e1m3_(\w+?)_(d\d+)_(b\d+)_\d+\s+D=\d+\(P\d+\)"
                        r".*?(PASS|FAIL).*?pin_frac=([\d.]+)")
    out, mode = {}, None
    text = open(path).read().splitlines()
    for line in text:
        m = cond.match(line)
        if m:
            out[(m[1], m[2], m[3])] = (m[5] == "PASS", float(m[4])); mode = "CONDITIONAL"
    if not out:
        for line in text:
            m = legacy.match(line)
            if m:
                out[(m[1], m[2], m[3])] = (m[4] == "PASS", float(m[5])); mode = "LEGACY(identity)"
    if out:
        print(f"  [pin gate parsed from {os.path.basename(path)}: {mode}]")
    return out or None


def load(dirpath, job):
    rows = []
    for fn in sorted(glob.glob(os.path.join(dirpath, f"e1m3_*_{job}_result.txt"))):
        for line in open(fn):
            m = PROBE.search(line)
            if m:
                d = m.groupdict()
                for k in ("A_all", "A_free", "ttft_p50", "ttft_p95", "plateau"):
                    d[k] = float(d[k])
                for k in ("seed", "n_keep", "n_err", "cap", "nbw"):
                    d[k] = int(d[k])
                rows.append(d)
    return rows


def t_ci(vals):
    """95% t-interval of the mean over BLOCKS -- the interval the verdict
    reads. See the TCRIT comment for why this and not a percentile bootstrap."""
    n = len(vals)
    if n < 2:
        return (float("nan"), float("nan"))
    m = statistics.fmean(vals)
    h = TCRIT.get(n, 1.96) * statistics.stdev(vals) / math.sqrt(n)
    return (m - h, m + h)


def boot_ci(vals, n_boot=N_BOOT, seed=BOOT_SEED):
    """Percentile bootstrap over BLOCKS. Printed for continuity with the rest
    of the campaign; NOT read by the verdict (anti-conservative at these n)."""
    if len(vals) < 2:
        return (float("nan"), float("nan"))
    rnd = random.Random(seed)
    reps = sorted(statistics.fmean([vals[rnd.randrange(len(vals))]
                                    for _ in range(len(vals))])
                  for _ in range(n_boot))
    return (reps[int(0.025 * n_boot)], reps[min(n_boot - 1, int(0.975 * n_boot))])


def tolerable_sd(n, gap=G_FLAT - 0.960):
    """The largest sd(g) at which n blocks can still put the test arm's upper
    bound below G_FLAT. Reported so the reader can see whether a NO VERDICT was
    a real null or just too few blocks."""
    return gap * math.sqrt(n) / TCRIT.get(n, 1.96) if n >= 2 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--job", required=True)
    a = ap.parse_args()

    rows = load(a.dir, a.job)
    if not rows:
        print(f"no M3PROBE lines for job={a.job} under {a.dir}")
        return

    # --- gates first: a voided cell must never reach the decision rule -------
    voided = set()
    print("=== per-probe gates (sec 4.3.8(b)(c)) ===")
    for r in sorted(rows, key=lambda r: (r["arm"], r["cell"], r["seed"])):
        bad = [k for k, v in (("conc", r["g_conc"]), ("offcliff", r["g_cliff"]),
                              ("err", r["g_err"])) if v == "FAIL"]
        if bad:
            voided.add((r["arm"], r["cell"]))
            print(f"  VOID {r['arm']:>4} {r['cell']:>5} s{r['seed']} block={r['block']}"
                  f" -- failed {bad} (conc={r['conc']}/{r['cap']}, "
                  f"ttft_p50={r['ttft_p50']:.0f} vs 2x plateau {2*r['plateau']:.0f})")
    print(f"  voided cells: {sorted(voided) if voided else 'none'}")

    # --- ★the pin gate, sec 4.3.8(e) -- cite-blocking, and now actually read --
    pin = load_pin(a.dir, a.job)
    pin_void_blocks = set()
    if pin is None:
        print(f"\n=== PIN GATE (sec 4.3.8(e)) ===\n"
              f"  MISSING: no m3_pin_{a.job}.txt under {a.dir}.\n"
              f"  Run  ./m3_pin_check.sh {a.job}  first. The verdict is REFUSED --\n"
              f"  'not run' must never read the same as 'passed'.")
        pin_missing = True
    else:
        pin_missing = False
        nfail = sum(1 for v in pin.values() if not v[0])
        print(f"\n=== PIN GATE (sec 4.3.8(e)): {len(pin)-nfail}/{len(pin)} PASS ===")
        percell = collections.defaultdict(lambda: [0, 0])
        for (arm, cell, blk), (ok, pf) in pin.items():
            percell[(arm, cell)][0 if ok else 1] += 1
            if not ok:
                pin_void_blocks.add((arm, cell, blk))
        for k in sorted(percell, key=lambda k: (k[0], int(k[1][1:]))):
            p, f = percell[k]
            mark = "   <-- cell has unpinned blocks" if f else ""
            print(f"    {k[0]:>4} {k[1]:>5}  {p}/{p+f} PASS{mark}")

    # --- per-cell summary ----------------------------------------------------
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["arm"], r["cell"])].append(r)
    print("\n=== per-cell A_all / A_free (ms), mean +/- sd over blocks ===")
    print(f"  {'arm':>4} {'cell':>5} {'n':>2} {'A_all':>16} {'A_free':>16} "
          f"{'blocking win':>13} {'conc':>6}")
    for k in sorted(by, key=lambda k: (k[0], int(k[1][1:]))):
        rs = by[k]
        aa = [r["A_all"] for r in rs]
        af = [r["A_free"] for r in rs]
        sd_a = statistics.stdev(aa) if len(aa) > 1 else 0.0
        sd_f = statistics.stdev(af) if len(af) > 1 else 0.0
        mark = "  <-- VOIDED" if k in voided else ""
        print(f"  {k[0]:>4} {k[1]:>5} {len(rs):>2} "
              f"{statistics.fmean(aa):8.1f} +/-{sd_a:5.1f} "
              f"{statistics.fmean(af):8.1f} +/-{sd_f:5.1f} "
              f"{statistics.fmean([r['nbw'] for r in rs]):13.1f} "
              f"{statistics.fmean([float(r['conc']) for r in rs if r['conc'] != 'None']):6.1f}{mark}")

    # --- g, per arm ----------------------------------------------------------
    print(f"\n=== g = A_free({NUM_CELL})/A_free({DEN_CELL}), paired within BLOCK ===")
    g_by_arm = {}
    for arm in sorted({r["arm"] for r in rows}):
        if (arm, NUM_CELL) in voided or (arm, DEN_CELL) in voided:
            print(f"  {arm}: NO VERDICT -- a cell g depends on was voided")
            continue
        # ★paired WITHIN BLOCK, not within seed: a block is one fresh boot of
        # every cell at one seed, and that is the unit g is independent over.
        num = {r["block"]: r["A_free"] for r in rows
               if r["arm"] == arm and r["cell"] == NUM_CELL}
        den = {r["block"]: r["A_free"] for r in rows
               if r["arm"] == arm and r["cell"] == DEN_CELL}
        blocks = sorted(set(num) & set(den))
        # ★sec 4.3.8(e): a block whose d16 OR d54 did not hold its target
        # partition cannot contribute a g -- the label is not the realized
        # allocation there (the Stage-0 lesson, CONSENSUS 1-21/1-22).
        dropped = [b for b in blocks
                   if (arm, NUM_CELL, b) in pin_void_blocks
                   or (arm, DEN_CELL, b) in pin_void_blocks]
        blocks = [b for b in blocks if b not in dropped]
        if dropped:
            print(f"  {arm:>4}  pin-voided blocks dropped: {dropped} "
                  f"({len(blocks)} of {len(blocks)+len(dropped)} remain)")
        gs = [num[b] / den[b] for b in blocks if den[b]]
        if not gs:
            print(f"  {arm}: no paired blocks survive the pin gate")
            continue
        lo, hi = t_ci(gs)
        blo, bhi = boot_ci(gs)
        g_by_arm[arm] = (statistics.fmean(gs), lo, hi, gs)
        print(f"  {arm:>4}  g = {statistics.fmean(gs):.3f}  "
              f"t-CI95 [{lo:.3f}, {hi:.3f}]  (bootstrap [{blo:.3f}, {bhi:.3f}], "
              f"not read by the verdict)")
        print(f"        per-block {[round(x, 3) for x in gs]}  n_indep={len(gs)}  "
              f"sd={statistics.stdev(gs) if len(gs) > 1 else float('nan'):.3f}  "
              f"(this n tolerates sd <= {tolerable_sd(len(gs)):.3f})")

    # --- the pre-registered verdict -----------------------------------------
    print(f"\n=== VERDICT (thresholds fixed in DESIGN.md sec 4.3.8(c): "
          f"lever g>={G_LEVER}, flat g<={G_FLAT}) ===")
    if CONTROL_ARM not in g_by_arm or TEST_ARM not in g_by_arm:
        print("  NO VERDICT -- both arms are required and at least one is missing/voided.")
        return
    if pin_missing:
        print("  NO VERDICT -- the sec 4.3.8(e) pin gate was never run for this job.")
        return
    (gc, cl, ch, gcs), (gt, tl, th, gts) = g_by_arm[CONTROL_ARM], g_by_arm[TEST_ARM]
    n_indep = min(len(gcs), len(gts))
    if n_indep < N_MIN_INDEP:
        print(f"  NO VERDICT -- only {n_indep} independent blocks (need "
              f"{N_MIN_INDEP}). At this n the rule tolerates sd(g) <= "
              f"{tolerable_sd(n_indep):.3f}, against sd values of 0.10-0.22 "
              f"measured across this campaign. A null here would be "
              f"indistinguishable from insufficient replication.")
        return
    disjoint = (ch < tl) or (th < cl)

    # ★★2026-08-03 -- RESOLVING THE PRE-REGISTRATION INCONSISTENCY (sec 4.3.8(c)).
    #
    # sec 4.3.8(c)'s rule TEXT is on point estimates ("T8 g>=1.5 and Ha8
    # g<=1.15, CIs disjoint") and that is what the code below implements. The
    # SAME section's power calculation instead framed the test-arm side as an
    # UPPER BOUND below 1.15. On 872077 the two disagree (Ha8's t-CI upper
    # bound is 1.190), so the campaign's deciding rule was ambiguous exactly
    # where it mattered.
    #
    # It cannot be resolved for 872077 without choosing a rule after seeing the
    # data, so BOTH readings are printed for that job and NEITHER is adopted.
    # For every FUTURE job the rule is fixed here, before that data exists:
    #
    #   RULE_BOUNDS (pre-registered 2026-08-03, effective for jobs after this
    #   commit): each arm is judged on the CI bound FACING its threshold --
    #   control arm needs lower bound >= G_LEVER, test arm needs upper bound
    #   <= G_FLAT -- plus disjoint CIs.
    #
    # Rationale, and note it cuts AGAINST the earlier apparent result: the test
    # arm's side is the ACCEPTANCE OF A NULL ("Ha8 does not respond"), and a
    # null accepted on a point estimate alone is not evidence. Requiring the
    # bound facing the threshold makes both halves symmetric and makes the null
    # actually evidenced. Under RULE_BOUNDS 872077 would NOT fire -- which is
    # why adopting it now for 872077 would be indistinguishable from picking
    # the answer, and why it binds only going forward.
    bounds_fires = (cl >= G_LEVER) and (th <= G_FLAT) and disjoint
    point_fires = (gc >= G_LEVER) and (gt <= G_FLAT) and disjoint
    print(f"  [RULE_POINT  -- sec 4.3.8(c) text]   "
          f"{CONTROL_ARM} g={gc:.3f}>={G_LEVER}? {gc >= G_LEVER}; "
          f"{TEST_ARM} g={gt:.3f}<={G_FLAT}? {gt <= G_FLAT}; disjoint? {disjoint} "
          f"=> {'FIRES' if point_fires else 'does not fire'}")
    print(f"  [RULE_BOUNDS -- pre-registered 2026-08-03, FUTURE jobs only]   "
          f"{CONTROL_ARM} lower={cl:.3f}>={G_LEVER}? {cl >= G_LEVER}; "
          f"{TEST_ARM} upper={th:.3f}<={G_FLAT}? {th <= G_FLAT}; disjoint? {disjoint} "
          f"=> {'FIRES' if bounds_fires else 'does not fire'}")
    if point_fires != bounds_fires:
        print(f"  ⇒ THE TWO READINGS DISAGREE. For a job predating the "
              f"2026-08-03 fix, that is itself the verdict: NO VERDICT, because "
              f"the deciding rule was ambiguous where it mattered. Neither "
              f"reading may be adopted for it after the fact.")
        return

    if gc >= G_LEVER and gt <= G_FLAT and disjoint:
        v = (f"ASYMMETRY ATTRIBUTABLE TO THE ARM. {CONTROL_ARM} responds "
              f"(g={gc:.2f}), {TEST_ARM} does not (g={gt:.2f}), CIs disjoint. "
              f"This is the defensive asset vs DuetServe's Transformer result. "
              f"E1 proceeds, reported per arm.")
    elif gc >= G_LEVER and gt >= G_LEVER:
        v = (f"LEVER PRESENT IN BOTH (g={gc:.2f}, {gt:.2f}). Proceed to the E1 "
              f"main sweep (branch A), re-siting the ladder on the measured "
              f"A_free range.")
    elif gc <= G_FLAT and gt <= G_FLAT:
        v = (f"LEVER ABSENT AT SERVING CONCURRENCY IN BOTH (g={gc:.2f}, {gt:.2f}). "
              f"Close E1 as branch B -- and the REASON is 'the ITL term of "
              f"conjunctive goodput is not a function of the decode-SM lever "
              f"here', NOT 'the exclusion rule deletes the lever' (refuted by "
              f"sec 4.3.7 and the C2 log-range).")
    else:
        v = (f"NO VERDICT (g={gc:.2f} control, {gt:.2f} test, CIs "
              f"{'disjoint' if disjoint else 'OVERLAPPING'}). Report and stop. "
              f"Do NOT re-cut the thresholds after seeing the data.")
    print(f"  {v}")
    print("\n  Scope, stated in advance: one rate, one context regime, two arms; "
          "A_free is valid only over d16-d54. This does not decide whether the "
          "lever pays off in goodput.")
    print("  Not citable before claims-auditor.")


if __name__ == "__main__":
    main()
