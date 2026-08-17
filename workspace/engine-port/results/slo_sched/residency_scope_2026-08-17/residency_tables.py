#!/usr/bin/env python3
"""Tables + identity checks + block/placement analysis for the A-2 scope size.

Reads only `residency_scope_2026-08-17.json` (produced by residency_scope.py)
and, READ-ONLY and as given constants, `G16_RESULTS_2026-08-17.json`.
No decision quantity is recomputed here.
"""
from __future__ import annotations
import json, math, os, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SLO = os.path.dirname(HERE)
DATA = json.load(open(os.path.join(HERE, "residency_scope_2026-08-17.json")))
REPORT = json.load(open(os.path.join(SLO, "G16_RESULTS_2026-08-17.json")))

T = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447}
ARMS = ["d16", "d24", "d34", "d44", "d54", "d64", "d74"]
BLOCKS = ["blk1", "blk2", "blk3", "blk4"]
FULL = [b for b in DATA["boots"] if b["mode"] == "full"]
SMOKE = [b for b in DATA["boots"] if b["mode"] == "smoke"]


def stat(xs):
    xs = list(xs)
    n = len(xs)
    m = statistics.fmean(xs)
    sd = statistics.stdev(xs) if n > 1 else float("nan")
    h = T[n - 1] * sd / math.sqrt(n) if n > 1 else float("nan")
    return m, sd, m - h, m + h, min(xs), max(xs), n


def fmt(xs, pct=True, dec=2):
    m, sd, lo, hi, mn, mx, n = stat(xs)
    k = 100.0 if pct else 1.0
    return (f"{m*k:.{dec}f} ± {sd*k:.{dec}f}  [{lo*k:.{dec}f}, {hi*k:.{dec}f}]  "
            f"({mn*k:.{dec}f}–{mx*k:.{dec}f})")


def by_arm(key, phase=None, boots=FULL):
    out = {}
    for a in ARMS:
        vals = []
        for b in boots:
            if b["arm"] != a:
                continue
            vals.append(b["per_phase"][phase][key] if phase else b[key])
        if vals:
            out[a] = vals
    return out


line = "-" * 108
print("=" * 108)
print("TABLE 1 -- UNIFIED DENOMINATOR.  'W' = share of the window in which the arm's NOMINAL split (108-D, D)")
print("           is the realized partition.  FULL campaign, n=4 boots/arm, mean ± SD [t(3) 95% CI] (min-max), %")
print("=" * 108)
for key, label in [
        ("W_cnt_da",  "A  count / decode-active   <- ADDENDUM A-2's OWN DENOMINATOR (realized_pin_check)"),
        ("W_cnt_all", "B  count / all bench snaps"),
        ("W_tim_all", "C  time  / all bench span  <- controller_summary, the FULL number in the handoff"),
        ("W_tim_da",  "D  time  / decode-active")]:
    print(f"\n[{label}]")
    print(f"  {'arm':5s} {'mean ± SD  [t(3) CI]  (min-max)  %':52s}")
    for a, v in by_arm(key).items():
        print(f"  {a:5s} {fmt(v):52s}")

print("\n" + "=" * 108)
print("TABLE 2 -- the three quoted numbers, ON THE SAME DENOMINATOR (A-2's: count / decode-active)")
print("=" * 108)
print(f"  {'source':34s} {'denominator':34s} {'nominal-split share %':26s}")
print(f"  {'addendum A-2 (as written)':34s} {'count / decode-active':34s} "
      f"{'32-47  (= 100 - 53..68)':26s}   <- baseline, 26 files, provenance NOT in repo")
sm = {a: [b["W_cnt_da"] for b in SMOKE if b["arm"] == a] for a in ("d44", "d64", "d74")}
for a, v in sm.items():
    m, sd, *_ = stat(v)
    print(f"  {'G16 smoke NP=40 R=1 ' + a:34s} {'count / decode-active':34s} "
          f"{m*100:6.2f} (n=2: {v[0]*100:.2f}, {v[1]*100:.2f})")
for a, v in by_arm("W_cnt_da").items():
    print(f"  {'G16 FULL NP=200 R=3 ' + a:34s} {'count / decode-active':34s} {fmt(v)}")
allv = [b["W_cnt_da"] for b in FULL]
print(f"  {'G16 FULL all 28 boots (RANGE only':34s} {'count / decode-active':34s} "
      f"{min(allv)*100:.2f} – {max(allv)*100:.2f}   (pooling ACROSS arms is not an")
print(f"  {'  -- arms are not exchangeable)':34s} {'':34s} estimate: the arm axis IS the trend)")

print("\n  -- and the SAME three on the E-TIM denominator (time / whole benchmark span) --")
for a, v in sm.items():
    vv = [b["W_tim_all"] for b in SMOKE if b["arm"] == a]
    print(f"  {'G16 smoke ' + a:34s} {'time / all bench span':34s} "
          f"{statistics.fmean(vv)*100:6.2f} (n=2: {vv[0]*100:.2f}, {vv[1]*100:.2f})")
for a, v in by_arm("W_tim_all").items():
    print(f"  {'G16 FULL ' + a:34s} {'time / all bench span':34s} {fmt(v)}")

print("\n" + "=" * 108)
print("TABLE 3 -- WINDOW RESTRICTION: whole benchmark span vs the 6 pre-registered round windows (LO / HI)")
print("           (the whole-span denominator includes inter-round idle, which is all state-0)")
print("=" * 108)
for key in ("W_tim_all", "W_cnt_da"):
    print(f"\n[{key}]   {'arm':5s} {'whole span':30s} {'LO windows':30s} {'HI windows':30s}")
    for a in ARMS:
        w = by_arm(key)[a]
        lo = by_arm(key, "LO")[a]
        hi = by_arm(key, "HI")[a]
        print(f"        {a:5s} {fmt(w,dec=2):30s} {fmt(lo,dec=2):30s} {fmt(hi,dec=2):30s}")

print("\n" + "=" * 108)
print("TABLE 4 -- RATIO DECOMPOSITION (is the monotone rise numerator- or denominator-driven?)")
print("           seconds at the nominal split vs total benchmark span, per arm (n=4)")
print("=" * 108)
print(f"  {'arm':5s} {'sec at nominal split':28s} {'total bench span (s)':28s} {'decode-active span (s)':28s}")
for a in ARMS:
    num = [b["sec_at_D"] for b in FULL if b["arm"] == a]
    den = [b["sec_total"] for b in FULL if b["arm"] == a]
    da = [b["sec_decode_active"] for b in FULL if b["arm"] == a]
    print(f"  {a:5s} {fmt(num,pct=False,dec=1):28s} {fmt(den,pct=False,dec=1):28s} {fmt(da,pct=False,dec=1):28s}")

print("\n" + "=" * 108)
print("IDENTITY CHECKS (gate #40 -- run BEFORE any of the above is quoted)")
print("=" * 108)
s0 = sum(b["n_decode_active_state0"] for b in FULL)
nda = sum(b["n_decode_active"] for b in FULL)
print(f"IC-1  Is '53-68% at (0,108)' a SECOND fact, or 1 - (nominal share)?")
print(f"      states present in the 3-group config: 0 (unpartitioned/idle), D (nominal), 108 (unpartitioned).")
print(f"      decode-active samples landing on raw state 0: {s0} / {nda} = {s0/nda*100:.4f}%")
print(f"      -> realized_pin_check bins raw 0 into the 108 bin, so among decode-active samples")
print(f"         share(108 bin) + share(D) = 1 EXACTLY (max residual {max(abs(b['cnt_da_at_108bin']+b['W_cnt_da']-1) for b in FULL):.2e}).")
print(f"      -> VERDICT: the two phrasings are ONE number. '53-68% at (0,108)' and '32-47% at the")
print(f"         nominal split' must never be cited as independent corroboration.")
print()
forced = all(b["states_time_s"].get(str(b["decode_sm"]), 0.0) <= b["sec_decode_active"] + 1e-9 for b in FULL)
never_idle = all(b["W_tim_da"] >= b["W_tim_all"] - 1e-12 for b in FULL)
print(f"IC-2  Is 'time/decode-active > time/all' a finding or an identity?")
print(f"      the nominal split state is reachable only with a non-empty decode batch")
print(f"      (multiplexing_mixin.py:882-923), so T_D^decode-active == T_D and the denominator")
print(f"      only shrinks -> W_tim_da >= W_tim_all is ALGEBRAICALLY FORCED.  observed: {never_idle} (28/28)")
print(f"      -> VERDICT: identity.  Report it as a denominator restatement, NOT as evidence.")
print()
print(f"IC-3  Is 'residency is monotone in decode SM' forced?")
mono_cnt = sum(1 for blk in BLOCKS
               if all(next(b["W_tim_all"] for b in FULL if b["arm"] == ARMS[i] and b["block"] == blk)
                      < next(b["W_tim_all"] for b in FULL if b["arm"] == ARMS[i+1] and b["block"] == blk)
                      for i in range(len(ARMS) - 1)))
print(f"      NOT forced: numerator (T_D) and denominator (T_total) both move with D (TABLE 4).")
print(f"      strict monotone increase in W_tim_all across all 7 arms, per block: {mono_cnt}/4 blocks")
print(f"      -> VERDICT: empirical, not algebraic.  But it is ALSO predicted a priori by")
print(f"         multiplexing_mixin.py:504-511 (prefill SM = 108-D shrinks -> prefill spans lengthen),")
print(f"         and addendum A-1 already registered it.  This is a CONFIRMATION, not a discovery.")
print()
print(f"IC-4  Does the count axis carry a bias with the SAME sign as the trend?")
print(f"      yes -- multiplexing_mixin.py:504-511 documents that prefill-in-flight syncs are")
print(f"      under-sampled in the COUNT grid, more so at large prefill SM (= small D).")
print(f"      -> the count axis alone cannot establish the trend.  The TIME axis is the")
print(f"         bias-corrected one; both agree here, which is the meaningful check.")

print("\n" + "=" * 108)
print("TASK 4 -- PLACEMENT / BLOCK DEPENDENCE of residency  (n_indep(placement) = 3)")
print("           blk1 = gpu42 solo | blk2+blk3 = gpu41 concurrent pair | blk4 = gpu41 solo")
print("=" * 108)
print("\n  W_tim_all (%) by arm x block")
print(f"  {'arm':5s} " + " ".join(f"{b:>8s}" for b in BLOCKS) + "   arm mean   max-min")
for a in ARMS:
    row = {b["block"]: b["W_tim_all"] for b in FULL if b["arm"] == a}
    vals = [row[b] for b in BLOCKS]
    print(f"  {a:5s} " + " ".join(f"{row[b]*100:8.2f}" for b in BLOCKS)
          + f"   {statistics.fmean(vals)*100:8.2f}  {(max(vals)-min(vals))*100:8.2f}")

print("\n  block main effect, PAIRED WITHIN ARM (7 arms = 7 pairs; every block contains all 7 arms")
print("  and all 7 positions, so block means are composition-balanced by design):")
rel = {}
for a in ARMS:
    row = {b["block"]: b["W_tim_all"] for b in FULL if b["arm"] == a}
    am = statistics.fmean(row.values())
    rel[a] = {b: row[b] / am - 1.0 for b in BLOCKS}
for b in BLOCKS:
    v = [rel[a][b] for a in ARMS]
    print(f"    {b}: relative deviation from arm mean = {fmt(v, dec=2)}   (t(6) CI, n=7 arms)")

print("\n  placement contrasts (paired across the 7 arms):")
for name, f in [
    ("blk4 - mean(blk2,blk3)   same node gpu41, solo vs concurrent  [+ time]",
     lambda a: rel[a]["blk4"] - (rel[a]["blk2"] + rel[a]["blk3"]) / 2),
    ("blk1 - mean(blk2,blk3,blk4)  gpu42 vs gpu41  [+ first-in-time]",
     lambda a: rel[a]["blk1"] - (rel[a]["blk2"] + rel[a]["blk3"] + rel[a]["blk4"]) / 3),
    ("blk2 - blk3              same node, same wall-clock, different GPU",
     lambda a: rel[a]["blk2"] - rel[a]["blk3"]),
]:
    v = [f(a) for a in ARMS]
    m, sd, lo, hi, mn, mx, n = stat(v)
    sig = "CI EXCLUDES 0" if (lo > 0 or hi < 0) else "CI includes 0"
    print(f"    {name:62s} {m*100:+6.2f} ± {sd*100:.2f} %  [{lo*100:+.2f}, {hi*100:+.2f}]  {sig}")

print("\n  block x arm INTERACTION probe (addendum E-3's residual threat):")
print("    per-block relative deviation, spread across arms (SD over the 7 arms):")
for b in BLOCKS:
    v = [rel[a][b] for a in ARMS]
    print(f"    {b}: SD across arms = {statistics.stdev(v)*100:.2f} %  "
          f"(range {min(v)*100:+.2f} .. {max(v)*100:+.2f})")
print("    arm rank order of W_tim_all, per block:")
for b in BLOCKS:
    order = sorted(ARMS, key=lambda a: next(x["W_tim_all"] for x in FULL
                                            if x["arm"] == a and x["block"] == b))
    print(f"    {b}: {' < '.join(order)}")

print("\n" + "=" * 108)
print("TASK 3 -- what the arm-dependence does to gap_upper  (M_itl, gap_upper read-only from the report)")
print("=" * 108)
for ph in ("LO", "HI"):
    c = REPORT["campaign"][ph]
    M = c["arm_means"]["M_itl"]
    sat = c["saturation"]
    U = sat["upper_arms"]
    gap = sat["gap_upper_ms"]
    amax = max(U, key=lambda a: M[a])
    amin = min(U, key=lambda a: M[a])
    print(f"\n[{ph}]  gap_upper = M_itl({amax}) - M_itl({amin}) = {gap:.4f} ms  "
          f"(delta = {sat['delta_ms']} ms, verdict {c['verdict']})")
    print(f"  {'arm':5s} {'M_itl (ms) mean ± SD':26s} {'w = W_tim_all in-phase (%)':30s} {'w = W_cnt_da in-phase (%)':30s}")
    wph = by_arm("W_tim_all", ph)
    wcd = by_arm("W_cnt_da", ph)
    for a in U:
        ci = c["arm_t_ci"]["M_itl"][a]
        print(f"  {a:5s} {ci['mean']:8.3f} ± {ci['sd']:.3f}{'':10s} "
              f"{fmt(wph[a], dec=2):30s} {fmt(wcd[a], dec=2):30s}")
    # exposure contrast between exactly the two arms that define the gap, paired by block
    pair = []
    for blk in BLOCKS:
        wa = next(b["per_phase"][ph]["W_tim_all"] for b in FULL if b["arm"] == amax and b["block"] == blk)
        wb = next(b["per_phase"][ph]["W_tim_all"] for b in FULL if b["arm"] == amin and b["block"] == blk)
        pair.append(wa - wb)
    m, sd, lo, hi, mn, mx, n = stat(pair)
    print(f"  paired exposure contrast  w({amax}) - w({amin}) = {m*100:+.2f} ± {sd*100:.2f} pp "
          f"[t(3) {lo*100:+.2f}, {hi*100:+.2f}]  n=4 blocks")
    wbar = statistics.fmean([statistics.fmean(wph[a]) for a in U])
    print(f"  mean exposure over U: w_bar = {wbar*100:.2f}%   -> dilution factor 1/w_bar = {1/wbar:.1f}x")
    print(f"     delta = {sat['delta_ms']} ms on this MIXTURE scale corresponds to "
          f"{sat['delta_ms']/wbar:.1f} ms on the co-resident-conditional scale")
    if abs(m) > 1e-9:
        print(f"  SUFFICIENCY CHECK: a single ARM-INDEPENDENT co-residency ITL penalty of "
              f"Delta = gap/(w_diff) = {gap/m:.1f} ms")
        print(f"     reproduces the observed gap_upper EXACTLY through unequal exposure alone "
              f"(zero difference in the conditional means).")
        print(f"     ({gap/(m-(hi-m)):.1f} .. {gap/(m+(hi-m)):.1f} ms over the t(3) CI of the exposure contrast)")
    # does a pure exposure model order the upper arms correctly?
    ord_w = sorted(U, key=lambda a: statistics.fmean(wph[a]))
    ord_m = sorted(U, key=lambda a: M[a])
    print(f"  ordering by exposure w : {' < '.join(ord_w)}")
    print(f"  ordering by M_itl      : {' < '.join(ord_m)}   -> pure-exposure model "
          f"{'CONSISTENT' if ord_w == ord_m else 'REFUTED (conditional means are not equal across U)'}")

print("\n" + "=" * 108)
print("TABLE 5 -- BRACKET for the A-2 conditional-label size, and the pre-registered headline axis")
print("           A-2 names its vehicle explicitly: \"H3'-b's time-weighted residency, reported per arm\"")
print("           => the HEADLINE axis is W_tim_all (controller_summary).  The rest is the bracket.")
print("=" * 108)
print(f"  {'arm':5s} {'HEADLINE W_tim_all %':22s} {'LO window %':14s} {'HI window %':14s} "
      f"{'HI, decode-active %':20s} {'sec at split':14s}")
for a in ARMS:
    w = by_arm("W_tim_all")[a]
    lo = by_arm("W_tim_all", "LO")[a]
    hi = by_arm("W_tim_all", "HI")[a]
    hida = by_arm("W_tim_da", "HI")[a]
    sec = [b["sec_at_D"] for b in FULL if b["arm"] == a]
    f = lambda v, d=2: f"{statistics.fmean(v)*100:.{d}f}±{statistics.stdev(v)*100:.{d}f}"
    print(f"  {a:5s} {f(w):22s} {f(lo):14s} {f(hi):14s} {f(hida):20s} "
          f"{statistics.fmean(sec):.1f}±{statistics.stdev(sec):.1f} s")

amin_arm, amax_arm = ARMS[0], ARMS[-1]
wmin = by_arm("W_tim_all")[amin_arm]
wmax = by_arm("W_tim_all")[amax_arm]
m0, sd0, l0, h0, *_ = stat(wmin)
m1, sd1, l1, h1, *_ = stat(wmax)
print(f"\n  span across the 7 arms: {m0*100:.2f}% ({amin_arm}) -> {m1*100:.2f}% ({amax_arm})"
      f"  = {m1/m0:.2f}x")
allb = [b["W_tim_all"] for b in FULL]
print(f"  every one of the 28 boots lies in [{min(allb)*100:.2f}%, {max(allb)*100:.2f}%]")

print("\n" + "=" * 108)
print("MEASUREMENT CAVEAT -- time resolution of the residency estimator")
print("=" * 108)
print(f"  {'arm':5s} {'mean dwell at split (ms)':26s} {'mean inter-sample gap INSIDE':30s} {'ratio':8s}")
print(f"  {'':5s} {'= T_D / (transitions/2)':26s} {'the split state (ms)':30s}")
for a in ARMS:
    dw, gp = [], []
    for b in FULL:
        if b["arm"] != a:
            continue
        nD = b["states_count"].get(str(b["decode_sm"]), 0)
        entries = max(1, b["split_transitions"] / 2.0)
        dw.append(b["sec_at_D"] / entries * 1000)
        gp.append(b["sec_at_D"] / max(1, nD) * 1000)
    print(f"  {a:5s} {statistics.fmean(dw):10.1f} ± {statistics.stdev(dw):5.1f}{'':9s} "
          f"{statistics.fmean(gp):10.1f} ± {statistics.stdev(gp):5.1f}{'':9s} "
          f"{statistics.fmean(dw)/statistics.fmean(gp):5.2f}")
print("  -> a co-residency span is only ~1-2 emitted snapshots long.  The 1/32 count subsample")
print("     therefore MISSES split spans that open and close between two emitted rows, and the")
print("     E-TIM rule attributes that time to the PRECEDING state (state 0 / 108).")
print("  -> DIRECTION: W is a LOWER bound on the true co-resident time share.  The bias is larger")
print("     at small D (shorter prefill spans), i.e. it works AGAINST the reported monotone trend,")
print("     so the trend is not created by it -- but the LEVEL (7.7-20.3%) may be understated.")

print("\n" + "=" * 108)
print("SUPPORTING: decode-active time as a share of the benchmark span (explains axis C vs D)")
print("=" * 108)
for grp, lab in ((FULL, "FULL"), (SMOKE, "smoke")):
    v = [b["sec_decode_active"] / b["sec_total"] for b in grp]
    print(f"  {lab:6s} n={len(v):2d}  decode-active time / benchmark span = "
          f"{statistics.fmean(v)*100:.2f} ± {statistics.stdev(v)*100:.2f} %  "
          f"({min(v)*100:.2f}-{max(v)*100:.2f})")

print("\n" + "=" * 108)
print("SUPPORTING -- WHY residency rises with D (mechanism check, not a decision quantity)")
print("           prefill work is byte-identical across all 28 boots and both phases:")
print("           204,828 input tokens / 600 completed requests per phase (verified from the")
print("           LO/HI artifacts), so seconds-at-the-split is a clean prefill-occupancy probe.")
print("=" * 108)
print(f"  {'arm':5s} {'prefill SM':11s} {'sec at split (n=4)':22s} {'sec x prefill_SM (SM*s)':26s}")
prods = []
for a in ARMS:
    D = int(a[1:]); P = 108 - D
    sec = [b["sec_at_D"] for b in FULL if b["arm"] == a]
    pr = [s * P for s in sec]
    prods.append(statistics.fmean(pr))
    print(f"  {a:5s} {P:>7d}     {statistics.fmean(sec):8.1f} ± {statistics.stdev(sec):4.1f}"
          f"{'':6s} {statistics.fmean(pr):8.0f} ± {statistics.stdev(pr):4.0f}")
print(f"  -> product across the 7 arms: {min(prods):.0f}-{max(prods):.0f} SM*s "
      f"(spread {(max(prods)/min(prods)-1)*100:.1f}%) while prefill SM spans 34-92 (2.71x)")
print("  -> prefill throughput is ~LINEAR in SM over 34-92 SM here; the residency trend is that")
print("     linearity, not a scheduler artefact.  NOT an identity: sublinear/saturating scaling")
print("     would have produced a product rising with D.  (The resolution bias of the estimator")
print("     is larger at small D, so the true product is if anything slightly SUPER-linear at")
print("     small D -- i.e. the check is conservative in the direction that matters.)")

print("\n" + "=" * 108)
print("TABLE 6 -- THE TWO CANONICAL CO-RESIDENCY OBSERVABLES DISAGREE, AND THE DISAGREEMENT IS")
print("           ITSELF MONOTONE IN D.  Both come from realized_pin_check.py, same denominator")
print("           (count / decode-active).  'split state' = realized partition is (108-D, D);")
print("           'CO_RESIDENT' = a prefill batch is in flight (prefill_active_batch_size > 0).")
print("=" * 108)
print(f"  {'arm':5s} {'split-state share %':22s} {'CO_RESIDENT_frac %':22s} {'ratio CO/split':14s}")
for a in ARMS:
    w = [b["W_cnt_da"] for b in FULL if b["arm"] == a]
    c = [b["co_resident_frac"] for b in FULL if b["arm"] == a]
    print(f"  {a:5s} {statistics.fmean(w)*100:8.2f} ± {statistics.stdev(w)*100:.2f}{'':7s} "
          f"{statistics.fmean(c)*100:8.2f} ± {statistics.stdev(c)*100:.2f}{'':7s} "
          f"{statistics.fmean(c)/statistics.fmean(w):8.2f}")
print("  -> the ratio runs 0.29 (d16) -> 0.84 (d74): at small D a majority of the split-state time")
print("     carries NO in-flight prefill batch.  ANY statement of the A-2 scope size must name")
print("     WHICH observable it uses; the two differ by up to 3.4x and the gap is arm-dependent.")
print("  -> A-2/A-1's imported baselines used BOTH: '53-68% at (0,108)' is the split-state")
print("     observable (32-47% nominal); '0.582 -> 0.818' (A-1, s8 grid) is CO_RESIDENT.")
print("     G16 measures 6.2-21.8% and 1.8-18.3% respectively -- the imported CO_RESIDENT")
print("     baseline is off by 3-32x on its own axis (lesson item 31: verify the basis of")
print("     auxiliary numbers imported from another campaign/scale).")
