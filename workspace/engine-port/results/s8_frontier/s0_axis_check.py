#!/usr/bin/env python3
"""s0_axis_check.py -- put BOTH sides of DESIGN.md sec 4.3.13 sec 0 on one
recorded, bin-conditional estimand computed by one script.

Pre-registration: `PREREG_S0_AXIS_2026-08-03.md` (written before any output).
STATUS: **DIAGNOSTIC ONLY.**  Nothing here licenses a performance claim, a
`G_LEVER`/`G_FLAT` value, or any statement about decode-SM elasticity.

WHY IT EXISTS
-------------
sec 0 asserts a 2.6x disagreement between two direct measurements of
"decode at 16 SM" per-token ITL p50:

    C2 865493, SPLIT population, batch bin 5   ->  28.79 ms
    872077 (E1 grid) d16, decode batch ~4.5    ->  11.09 ms

Decode batch was *already* matched by the auditor (bin 5 vs ~4.5), so batch
mismatch is not an available third explanation.  What is missing is that the
E1-side number **has no stored artifact** -- `grep` over `results/` finds
11.09 only in `DESIGN.md` prose -- so whether it is the bin-5 *conditional*
or a *pooled* quantile over a batch distribution whose mean is 4.5 is not
recorded.  Those are different estimands (methodology gates #4/#5).  This
script computes the bin-conditional form on both sides with the same rule.

It CANNOT adjudicate sec 0's (i) vs (ii) -- see PREREG sec 6.

Run:
  python3 s0_axis_check.py                      # T8 primary + Ha8 secondary
  python3 s0_axis_check.py --arms T8
"""
import argparse
import collections
import glob
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c2_anchor as c2          # noqa: E402  (C2 side, audited reconstruction)
import m3_conditional as m3     # noqa: E402  (E1 side, audited estimator)

HERE = os.path.dirname(os.path.abspath(__file__))
C2_JOB = "865493"
E1_JOB = "872077"
CELL = "d16"
MIN_N = 50                      # PREREG sec 2: same floor as c2_anchor step [4]
C2_GATE_BIN, C2_GATE_P50, C2_GATE_TOL = 5, 28.79, 0.05   # PREREG sec 4 gate 2


# --------------------------------------------------------------- E1 side
def label_probe_with_batch(d, arm, cell, block, seed, job, rate=m3.RATE_DEFAULT):
    """m3_conditional.label_probe + ONE added field: duration-weighted mean
    in-interval `decode_running_batch_size`.

    Everything else -- warm-up rule, error filter, interval construction,
    `split_frac` definition, `a_free` window rule -- is copied verbatim so
    the equivalence gate (PREREG sec 4 gate 1) can assert element-wise
    identity against the audited estimator.  Do not "improve" this loop.
    """
    tag = f"e1m3_{arm}_{cell}_b{block}_{job}"
    o = json.loads(open(f"{d}/{tag}_s{seed}.jsonl").read().strip().split("\n")[0])
    t0b, rows = m3.load_telemetry(f"{d}/{tag}_telemetry.jsonl")
    Dsm = int(cell[1:])

    tt, il = o["ttfts"], o["itls"]
    inp = o["input_lens"]
    err = o.get("errors") or []
    arr = m3.replay_arrivals(seed, rate, len(tt))
    lag, r = m3.align(rows, t0b, arr, tt, il)

    ct = np.array([row[0] - t0b - lag for row in rows])
    dsm = np.array([row[1] if row[1] is not None else -1 for row in rows])
    drb = np.array([row[2] for row in rows], dtype=float)
    pab = np.array([row[4] for row in rows], dtype=float)

    def _span(a, b):
        i0 = max(np.searchsorted(ct, a, side="right") - 1, 0)
        i1 = max(np.searchsorted(ct, b, side="right") - 1, 0)
        return i0, min(i1 + 1, len(ct) - 1)

    def frac(a, b, mask):
        if b <= a:
            return 0.0
        i0, i1 = _span(a, b)
        tot = hit = 0.0
        for k in range(i0, i1):
            s, e = max(a, ct[k]), min(b, ct[k + 1])
            if e <= s:
                continue
            tot += e - s
            if mask[k]:
                hit += e - s
        return (hit / tot) if tot > 0 else 0.0

    def wmean(a, b, vals):
        """Duration-weighted mean of a per-snapshot numeric array over [a,b].

        Same step-function integration as frac(); returns nan when the
        interval covers no telemetry span (so it is dropped, never binned to
        an invented 0)."""
        if b <= a:
            return float("nan")
        i0, i1 = _span(a, b)
        tot = acc = 0.0
        for k in range(i0, i1):
            s, e = max(a, ct[k]), min(b, ct[k + 1])
            if e <= s:
                continue
            tot += e - s
            acc += vals[k] * (e - s)
        return (acc / tot) if tot > 0 else float("nan")

    m_split = (dsm == Dsm)
    m_pref = (pab > 0)

    nw = int(math.ceil(m3.WARMUP_S * rate))
    keep = [i for i in range(nw, len(tt)) if (err[i] if i < len(err) else "") == ""]
    win = [(arr[j], arr[j] + tt[j]) for j in range(len(tt))
           if j < len(inp) and inp[j] >= m3.PREFILL_BLOCK_TOK]

    recs = []
    for i in keep:
        if not il[i]:
            continue
        t = arr[i] + tt[i]
        for dd in il[i]:
            a, b = t, t + dd
            a_free = not any(b > w0 and a < w1 for w0, w1 in win)
            recs.append((dd * 1000.0, a_free, frac(a, b, m_split),
                         frac(a, b, m_pref), i, wmean(a, b, drb)))
            t = b
    return dict(arm=arm, cell=cell, block=block, lag=lag, align_r=r, recs=recs)


def e1_side(arms, verbose=True):
    """-> (rows_by_arm, diag_by_arm).  rows = (itl_ms, split_frac, pref_frac, mbs)"""
    out, diag = {}, {}
    for arm in arms:
        rows, dg = [], []
        for b in m3.BLOCKS:
            mine = label_probe_with_batch(HERE, arm, CELL, b, b, E1_JOB)
            ref = m3.label_probe(HERE, arm, CELL, b, b, E1_JOB)
            # ---- PREREG sec 4 gate 1: element-wise identity vs the audited estimator
            assert len(mine["recs"]) == len(ref["recs"]), (
                f"GATE-E1 FAIL {arm} b{b}: n {len(mine['recs'])} != {len(ref['recs'])}")
            for x, y in zip(mine["recs"], ref["recs"]):
                assert abs(x[0] - y[0]) < 1e-9 and abs(x[2] - y[2]) < 1e-12, (
                    f"GATE-E1 FAIL {arm} b{b}: rec mismatch {x[:3]} vs {y[:3]}")
            rows.extend([(r[0], r[2], r[3], r[5]) for r in mine["recs"]
                         if not math.isnan(r[5])])
            dg.append(dict(block=b, align_r=mine["align_r"], lag=mine["lag"],
                           n=len(mine["recs"]), w_time=ref["w_time"]))
            if verbose:
                print(f"  [E1] {arm} {CELL} b{b}: n={len(mine['recs'])} "
                      f"align_r={mine['align_r']:.3f} w_time={ref['w_time']:.4f}",
                      file=sys.stderr)
        out[arm], diag[arm] = rows, dg
    return out, diag


# --------------------------------------------------------------- C2 side
def c2_side(arms):
    """-> (rows_by_arm, resid).  rows = (itl_ms, split_frac, pref_frac=nan, mbs)

    `c2_anchor.collect` returns (itl_ms, split_frac, un_frac, mean_bs, dur);
    it does not carry prefill co-residency, so that descriptive is computed
    separately in `c2_prefill_coresidency`.

    ⚠️ DATA AVAILABILITY (measured 2026-08-03): in job 865493 the client
    `t0_monotonic_s` anchor -- which `c2_anchor.collect` needs to place raw
    client chunk times on the telemetry clock -- exists for
    **T8 {d16,d24,d44,d92,np}** and **Hs8 {d16,d24,d44,d92,np}**, but for
    **Ha8 only {d24,d44,d92,np} (d16 is MISSING)** and for **M8 none at
    all**.  So the C2 side of the sec 0 comparison is reconstructible for
    **T8 and Hs8 only**.  An arm with no anchor yields ZERO rows here -- that
    is an absent measurement, NOT a measured zero, and the report labels it
    as such.
    """
    data, resid, _meta = c2.collect(C2_JOB, "1024")
    out = {}
    for arm in arms:
        rows = []
        for (a, cell, jb, _rep), rs in data.items():
            if a == arm and cell == CELL and jb == C2_JOB:
                rows.extend([(r[0], r[1], float("nan"), r[3]) for r in rs])
        out[arm] = rows
    return out, resid


def c2_prefill_coresidency(arm):
    """Time-weighted share of decode-busy telemetry time with prefill in
    flight, for the C2 cell.  Descriptive only (PREREG sec 6.3)."""
    p = os.path.join(c2.C2, f"s8_deconf_{arm}_C1024_{CELL}_{C2_JOB}_telemetry.jsonl")
    if not os.path.exists(p):
        return float("nan")
    ts, busy, pref = [], [], []
    for line in open(p):
        try:
            e = json.loads(line)
        except Exception:
            continue
        if e.get("event") != "runtime_snapshot":
            continue
        ts.append(e["timestamp_monotonic_s"])
        busy.append(e.get("decode_running_batch_size", 0) > 0)
        pref.append(e.get("prefill_active_batch_size", 0) > 0)
    tot = hit = 0.0
    for k in range(len(ts) - 1):
        dt = ts[k + 1] - ts[k]
        if dt > c2.GUARD_S or not busy[k]:
            continue
        tot += dt
        if pref[k]:
            hit += dt
    return hit / tot if tot else float("nan")


# --------------------------------------------------------------- reporting
def binned(rows, split_only=True):
    """-> {bin: [itl_ms, ...]} keyed by round(mean in-interval decode batch)."""
    by = collections.defaultdict(list)
    for itl, sf, _pf, mbs in rows:
        if split_only and sf < c2.SPLIT_HI:
            continue
        by[int(round(mbs))].append(itl)
    return by


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="T8,Ha8")
    a = ap.parse_args()
    arms = [x for x in a.arms.split(",") if x]

    print("=" * 78)
    print("s0_axis_check -- DESIGN.md sec 4.3.13 sec 0, one bin-conditional estimand")
    print("unit = ITL interval (per token) | label = REALIZED, duration-weighted,")
    print(f"SPLIT >= {c2.SPLIT_HI} | bin = round(duration-weighted mean decode batch)")
    print(f"primary stat = p50 (sec 0's own statistic; the sticky campaign primary")
    print("is p95 and is NOT changed by this file) | DIAGNOSTIC ONLY")
    print("=" * 78)

    print("\n[loading E1 side, with equivalence gate against m3_conditional]",
          file=sys.stderr)
    e1_rows, e1_diag = e1_side(arms)
    print("[loading C2 side via c2_anchor.collect]", file=sys.stderr)
    c2_rows, c2_resid = c2_side(arms)

    # ---------------------------------------------------------- gates
    print("\n[GATE 1] E1 labelling == m3_conditional.label_probe : PASS "
          "(asserted element-wise for every block)")

    gate2 = {}
    for arm in arms:
        b = binned(c2_rows[arm])
        v = c2.pct(b.get(C2_GATE_BIN, []), 0.50)
        gate2[arm] = v
    ref = gate2.get("T8", float("nan"))
    ok = abs(ref - C2_GATE_P50) <= C2_GATE_TOL
    print(f"[GATE 2] C2 T8 {CELL} bin-{C2_GATE_BIN} SPLIT p50 = {ref:.2f} ms "
          f"(target {C2_GATE_P50} +-{C2_GATE_TOL}) : {'PASS' if ok else 'FAIL'}")
    if not ok:
        print("         => C2 reconstruction differs from the audited one; "
              "per PREREG sec 4 the numbers below are VOID.")

    # ---------------------------------------------------------- table A
    print("\n[A] PER-BIN SPLIT per-token ITL, both jobs, cell " + CELL)
    print(f"    bins with n < {MIN_N} are printed as UNDERPOWERED, never dropped")
    print(f"{'arm':>4} {'bin':>4} | {'C2 p50':>8} {'C2 p95':>8} {'C2 n':>7} | "
          f"{'E1 p50':>8} {'E1 p95':>8} {'E1 n':>7} | {'r(p50)':>7} {'r(p95)':>7}")
    ratios = {}
    for arm in arms:
        cb, eb = binned(c2_rows[arm]), binned(e1_rows[arm])
        for bn in sorted(set(cb) | set(eb)):
            cv, ev = cb.get(bn, []), eb.get(bn, [])
            c50, c95 = c2.pct(cv, .50), c2.pct(cv, .95)
            e50, e95 = c2.pct(ev, .50), c2.pct(ev, .95)
            r50 = c50 / e50 if cv and ev and e50 else float("nan")
            r95 = c95 / e95 if cv and ev and e95 else float("nan")
            mark = "" if min(len(cv), len(ev)) >= MIN_N else "  UNDERPOWERED"
            print(f"{arm:>4} {bn:>4} | {c50:8.2f} {c95:8.2f} {len(cv):7d} | "
                  f"{e50:8.2f} {e95:8.2f} {len(ev):7d} | {r50:7.3f} {r95:7.3f}{mark}")
            if min(len(cv), len(ev)) >= MIN_N:
                ratios.setdefault(arm, []).append((bn, r50, r95))
        print()

    # ---------------------------------------------------------- readouts
    print("[B] PRE-REGISTERED READOUTS (PREREG sec 5)")
    for arm in arms:
        if not c2_rows[arm]:
            print(f"  {arm}: C2 side NOT RECONSTRUCTIBLE -- job {C2_JOB} has no "
                  f"client t0 anchor for this arm/{CELL} (absent measurement, "
                  "not a measured zero). Cross-job readout skipped.")
        eb = binned(e1_rows[arm])
        e_bin5 = c2.pct(eb.get(C2_GATE_BIN, []), 0.50)
        pooled = c2.pct([r[0] for r in e1_rows[arm] if r[1] >= c2.SPLIT_HI], 0.50)
        mean_b = (np.mean([r[3] for r in e1_rows[arm] if r[1] >= c2.SPLIT_HI])
                  if e1_rows[arm] else float("nan"))
        print(f"  {arm}: E1 bin-{C2_GATE_BIN} p50 = {e_bin5:.2f} ms | "
              f"E1 pooled-SPLIT p50 = {pooled:.2f} ms "
              f"(mean batch {mean_b:.2f}, n={len(eb.get(C2_GATE_BIN, []))})")
        if arm == "T8":
            for name, val in (("bin-5 conditional", e_bin5), ("pooled", pooled)):
                if not math.isnan(val):
                    d = abs(val - 11.09) / 11.09
                    print(f"       vs sec 0's 11.09 ms: {name} differs by "
                          f"{d*100:5.1f}%  -> {'R1 (within 15%)' if d <= .15 else 'R2 (>15%)'}")
    print("\n  [R3] ratio trend across shared bins (n >= %d):" % MIN_N)
    for arm, rs in ratios.items():
        if len(rs) >= 2:
            bs = [x[0] for x in rs]
            r50 = [x[1] for x in rs]
            slope = np.polyfit(bs, r50, 1)[0] if len(rs) >= 2 else float("nan")
            print(f"    {arm}: bins {bs} r(p50) "
                  f"{['%.2f' % x for x in r50]} | OLS slope {slope:+.3f}/bin")
        else:
            print(f"    {arm}: only {len(rs)} shared bin(s) at n>={MIN_N} "
                  "-> no trend readable")

    # ---------------------------------------------------------- table C
    print("\n[C] POPULATION DESCRIPTIVES (PREREG sec 6.3 -- reported, NOT controls)")
    print("    'C2 n_tot = 0' means the anchor is absent, not that no tokens ran")
    print(f"{'arm':>4} | {'C2 n_tot':>9} {'C2 SPLIT share':>15} {'C2 pref@busy':>13} "
          f"{'C2 realized@D':>14} | {'E1 SPLIT share':>15} {'E1 pref|SPLIT':>14} "
          f"{'E1 realized@D':>14} {'align_r':>9}")
    for arm in arms:
        c_all, e_all = c2_rows[arm], e1_rows[arm]
        c_sp = (sum(1 for r in c_all if r[1] >= c2.SPLIT_HI) / len(c_all)
                if c_all else float("nan"))
        e_sp = sum(1 for r in e_all if r[1] >= c2.SPLIT_HI) / max(len(e_all), 1)
        e_pref = np.mean([r[2] for r in e_all if r[1] >= c2.SPLIT_HI]) \
            if any(r[1] >= c2.SPLIT_HI for r in e_all) else float("nan")
        cres = c2_resid.get((arm, CELL, C2_JOB), {}).get("frac_target", float("nan"))
        eres = np.mean([d["w_time"] for d in e1_diag[arm]])
        arng = (min(d["align_r"] for d in e1_diag[arm]),
                max(d["align_r"] for d in e1_diag[arm]))
        print(f"{arm:>4} | {len(c_all):9d} {c_sp:15.3f} "
              f"{c2_prefill_coresidency(arm):13.3f} "
              f"{cres:14.3f} | {e_sp:15.3f} {e_pref:14.3f} {eres:14.4f} "
              f"{arng[0]:.2f}-{arng[1]:.2f}")

    print("\n" + "=" * 78)
    print("DIAGNOSTIC ONLY.  This check cannot adjudicate sec 0's (i) vs (ii):")
    print("  (i) 872077's decode_sms==16 is not a real 16-SM hardware execution")
    print("  (ii) C2's 28-31ms is a property of the [16,16,76-idle] + saturated-")
    print("       keepalive cell composition")
    print("Both remain open regardless of any number above.  The sec 4.3.11")
    print("residual (green-context creation vs hardware SM grant, IN ENGINE")
    print("CONTEXT) is untouched -- job 864669 answered only the isolated API")
    print("layer.  Separating (i)/(ii) needs a direct measurement (engine-context")
    print("SM-grant probe, or gate S1).  Result goes to claims-auditor before any")
    print("canon edit: this script and its pre-registration share an author.")
    print("=" * 78)


if __name__ == "__main__":
    main()
