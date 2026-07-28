#!/usr/bin/env python3
"""Analysis for the s8p_prefill sweep (mirror of s8_scaleup/s8_analyze.py +
s8_batch_matched.py, adapted to prefill probes).

PRIMARY metric (2026-07-28 revision, coordinator directive): a `max_new_tokens=1`
request's client-measured latency is
    admission_wait + prefill_forward(L) + one_decode_step + detok/network
and every term except prefill_forward is a large SM-axis-independent constant
(no per-prefill-forward-duration telemetry field exists -- DESIGN.md sec 2).
The absolute latency RATIO between two SM cells is therefore biased toward 1
by that constant (it systematically UNDER-estimates SM sensitivity, unlike
s8_scaleup's steady-state decode ITL, where the analogous constant is
negligible). The fix: regress per-cell TTFT p50 against L. The constant lands
in the INTERCEPT; the SLOPE (us/token) is the reciprocal of prefill compute
throughput and is what should be compared across SM cells. Slope ratio
(SM16/SM92) is the headline; absolute-latency ratio (the old table) is kept as
an auxiliary secondary check, consistent with the pre-2026-07-28 harness.

Regression is done PER REP, not on rep-pooled samples -- claims-auditor
flagged rep-pooling (no per-rep CI) as a residual confound in an earlier
version of this analyzer's sibling (s8_batch_matched.py's report style). Each
rep yields one slope sample and one intercept sample; n>=4 reps -> n>=4 slope
samples, mean +/- sd reported (methodology gate: n>=4).

L-dependence of the slope (curvature) is the model-discriminating signal this
campaign was built for: attention is O(L^2) so its per-token marginal cost
should RISE with L; SSD is O(L) so it should stay flat. Reported as
consecutive-L-pair segment slopes (3 segments from 4 L points; a full
quadratic fit is deferred to a future L_LIST extension to 8192 -- see
DESIGN.md sec 7 note added 2026-07-28), each aggregated the same way (n>=4
reps, mean +/- sd per segment).

Intercept stability across SM cells is also reported: if the fitted intercept
swings a lot between P16..P92 for the same arm, that means the assumed-SM-
independent constant is NOT actually SM-independent (e.g. admission/
scheduling contention differs by cell) and the slope-ratio headline should be
read with that caveat -- flagged explicitly, not silently averaged over.

Usage: s8p_analyze.py [--dir <campaign dir>]
"""
import argparse
import bisect
import collections
import glob
import json
import os
import re
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
CELL_SM = {"p16": 16, "p24": 24, "p44": 44, "p92": 92}
GUARD = 3.0
MIN_SAMPLES_PER_L = 3   # per (rep, L): below this the rep's median for that L is not trusted


def telemetry_timeline(path):
    ts, psm, pab, dbs = [], [], [], []
    for line in open(path):
        try:
            e = json.loads(line)
        except Exception:
            continue
        if e.get("event") != "runtime_snapshot" or e.get("phase") != "benchmark":
            continue
        ts.append(e["timestamp_monotonic_s"])
        psm.append(e.get("prefill_sms"))
        pab.append(e.get("prefill_active_batch_size", 0))
        dbs.append(e.get("decode_running_batch_size", 0))
    return ts, psm, pab, dbs


def linreg(xs, ys):
    """OLS slope+intercept. None if underdetermined or degenerate."""
    n = len(xs)
    if n < 2:
        return None
    mx, my = st.fmean(xs), st.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = my - slope * mx
    return slope, intercept


def mean_sd(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return float("nan"), float("nan"), 0
    m = st.fmean(xs)
    sd = st.stdev(xs) if len(xs) > 1 else 0.0
    return m, sd, len(xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=HERE)
    args = ap.parse_args()

    pat = re.compile(
        r"s8p_(?P<arm>\w+?)_(?P<cell>p16|p24|p44|p92)_(?P<job>\d+)_telemetry\.jsonl$"
    )
    cells = []
    for fn in sorted(os.listdir(args.dir)):
        m = pat.search(fn)
        if m:
            cells.append((m.groupdict(), os.path.join(args.dir, fn)))
    if not cells:
        print(f"no s8p telemetry under {args.dir}")
        return

    # Auxiliary (secondary) table: absolute latency at matched realized SM,
    # rep-pooled -- kept for continuity with the pre-revision harness and as
    # a sanity cross-check, NOT the headline (see module docstring).
    lat_by = collections.defaultdict(list)   # (arm, L, realized_psm) -> [latency_ms]
    conf_l = collections.defaultdict(collections.Counter)      # (arm, cell) -> L -> n
    conf_batch = collections.defaultdict(list)                 # (arm, cell) -> [prefill_active_batch_size]
    no_t0 = set()
    rows = []
    # PRIMARY: per-rep regression inputs. (arm, cell) -> rep -> {L: [latency_ms]}
    # (only samples attributed to the cell's OWN target prefill_sms -- escapes
    # or cross-partition samples are dropped from the regression, same
    # filtering the realized-pin gate itself applies).
    rep_l_lat = collections.defaultdict(lambda: collections.defaultdict(
        lambda: collections.defaultdict(list)))

    for g, tpath in cells:
        arm, cell, job = g["arm"], g["cell"], g["job"]
        want = CELL_SM[cell]
        ts, psm, pab, dbs = telemetry_timeline(tpath)
        act = [i for i, b in enumerate(pab) if b > 0]
        hist = collections.Counter(psm[i] for i in act)
        n_act = sum(hist.values())
        frac = hist[want] / n_act if n_act else float("nan")
        cores = (sum(1 for i in act if dbs[i] > 0) / n_act) if n_act else float("nan")
        batch = st.fmean(pab[i] for i in act) if n_act else float("nan")

        stem = tpath[: -len("_telemetry.jsonl")]
        t0_by_rep = {}
        for res in sorted(glob.glob(os.path.join(args.dir, f"s8p_{arm}_{job}_result.txt"))):
            for line in open(res):
                if not line.startswith("{"):
                    continue
                try:
                    s = json.loads(line)
                except Exception:
                    continue
                tag = s.get("tag", "")
                if tag.startswith(f"{arm}_{cell}_r") and "t0_monotonic_s" in s:
                    t0_by_rep[int(tag.rsplit("_r", 1)[1])] = s["t0_monotonic_s"]

        for raw in sorted(glob.glob(f"{stem}_rep*_raw.jsonl")):
            rep = int(re.search(r"_rep(\d+)_raw", raw).group(1))
            t0 = t0_by_rep.get(rep)
            if t0 is None:
                no_t0.add((arm, cell))
                continue
            for line in open(raw):
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                a, b = r["t_start_s"] + t0, r["t_end_s"] + t0
                L = r["L"]
                conf_l[(arm, cell)][L] += 1
                i = bisect.bisect_right(ts, a) - 1
                j = bisect.bisect_left(ts, b)
                if i < 0 or j >= len(ts) or psm[i] != psm[j]:
                    continue
                if (a - ts[i]) > GUARD or (ts[j] - b) > GUARD:
                    continue
                conf_batch[(arm, cell)].append(pab[i])
                lat_by[(arm, L, psm[i])].append(r["latency_ms"])
                if psm[i] == want:
                    rep_l_lat[(arm, cell)][rep][L].append(r["latency_ms"])

        rows.append(dict(arm=arm, cell=cell, want=want, frac=frac, cores=cores,
                          batch=batch, n_act=n_act, hist=dict(hist.most_common(3))))

    rows.sort(key=lambda r: (r["arm"], r["want"]))
    print("=== per-cell REALIZED prefill-partition / co-residency (all reps pooled) ===")
    hdr = (f"{'arm':>4} {'cell':>5} {'P':>4} | {'realized':>9} {'co-res':>7} "
           f"{'pbatch':>7} | realized hist")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['arm']:>4} {r['cell']:>5} {r['want']:>4} | "
              f"{r['frac']*100:8.1f}% {r['cores']*100:6.1f}% {r['batch']:7.2f} | {r['hist']}")
        if not (r["frac"] >= 0.80):
            print(f"     ^^ REALIZED-PREFILL-PIN FAIL: cell {r['cell']} did not run at P{r['want']}")
        if not (r["cores"] >= 0.80):
            print(f"     ^^ CO-RESIDENCY FAIL: cell {r['cell']} co-res {r['cores']:.3f} < 0.80")

    if no_t0:
        print()
        print(f"NOTE: {len(no_t0)} cell(s) have no attributable t0_monotonic_s: "
              f"{sorted(f'{a}/{c}' for a, c in no_t0)}")

    print()
    print("=== gate #4: does observed L or prefill_active_batch_size covary with the SM cell? ===")
    for (arm, cell), counter in sorted(conf_l.items()):
        total = sum(counter.values())
        dist = ", ".join(f"L{L}:{n}({n/total*100:.0f}%)" for L, n in sorted(counter.items()))
        bvals = conf_batch.get((arm, cell), [])
        bstat = (f"mean={st.fmean(bvals):.2f} median={st.median(bvals):.1f}"
                 if bvals else "no matched intervals")
        print(f"  {arm:>4} {cell:>5}: L-dist [{dist}]  prefill_active_batch [{bstat}]")

    # ------------------------------------------------------------------
    # PRIMARY: per-rep TTFT ~ L regression.
    # ------------------------------------------------------------------
    ls_global = sorted({L for byrep in rep_l_lat.values() for byL in byrep.values()
                         for L in byL})
    seg_pairs = list(zip(ls_global, ls_global[1:])) if len(ls_global) > 1 else []

    reg = {}   # (arm, cell) -> dict(slope_mean, slope_sd, slope_n, intercept_mean/sd, seg={pair: (mean,sd,n)}, per_rep=[...])
    for (arm, cell), byrep in sorted(rep_l_lat.items()):
        slopes, intercepts, per_rep_rows = [], [], []
        seg_samples = collections.defaultdict(list)
        for rep in sorted(byrep):
            byL = byrep[rep]
            xs, ys, n_by_l = [], [], {}
            for L in sorted(byL):
                vals = byL[L]
                n_by_l[L] = len(vals)
                if len(vals) >= MIN_SAMPLES_PER_L:
                    xs.append(L)
                    ys.append(st.median(vals))
            fit = linreg(xs, ys) if len(xs) >= 2 else None
            slope, intercept = fit if fit else (None, None)
            if slope is not None:
                slopes.append(slope)
                intercepts.append(intercept)
            med_by_l = dict(zip(xs, ys))
            for L0, L1 in seg_pairs:
                if L0 in med_by_l and L1 in med_by_l:
                    seg_samples[(L0, L1)].append(
                        (med_by_l[L1] - med_by_l[L0]) / (L1 - L0))
            per_rep_rows.append(dict(rep=rep, n_by_l=n_by_l, slope=slope,
                                      intercept=intercept))
        sm, ssd, sn = mean_sd(slopes)
        im, isd, in_ = mean_sd(intercepts)
        seg_stats = {pair: mean_sd(vals) for pair, vals in seg_samples.items()}
        reg[(arm, cell)] = dict(
            slope_mean=sm, slope_sd=ssd, slope_n=sn,
            intercept_mean=im, intercept_sd=isd, intercept_n=in_,
            seg=seg_stats, per_rep=per_rep_rows,
        )

    print()
    print("=== PRIMARY: per-rep TTFT(L) regression -- slope (us/token) and intercept (ms) ===")
    print("(slope = d(latency_ms)/d(L) * 1000 -> us/token; regression run separately per rep, "
          "no rep-pooling; n = number of reps with a valid fit)")
    hdr2 = (f"{'arm':>4} {'cell':>5} {'P':>4} | {'slope(us/tok)':>18} {'n':>3} | "
            f"{'intercept(ms)':>16} {'n':>3}")
    print(hdr2); print("-" * len(hdr2))
    for (arm, cell), d in sorted(reg.items(), key=lambda kv: (kv[0][0], CELL_SM[kv[0][1]])):
        want = CELL_SM[cell]
        slope_us = d["slope_mean"] * 1000.0 if d["slope_mean"] == d["slope_mean"] else float("nan")
        slope_sd_us = d["slope_sd"] * 1000.0 if d["slope_sd"] == d["slope_sd"] else float("nan")
        print(f"{arm:>4} {cell:>5} {want:>4} | "
              f"{slope_us:9.3f} +/- {slope_sd_us:6.3f} {d['slope_n']:>3} | "
              f"{d['intercept_mean']:8.2f} +/- {d['intercept_sd']:5.2f} {d['intercept_n']:>3}")
        if d["slope_n"] < 4:
            print(f"     ^^ n<4 slope samples for {arm}/{cell} -- do not cite (methodology gate)")

    if seg_pairs:
        print()
        print("=== curvature check: segment slope (us/token) by consecutive L pair ===")
        print("(rising across segments -> superlinear/attention-like; flat -> SSD-like O(L))")
        for (arm, cell), d in sorted(reg.items(), key=lambda kv: (kv[0][0], CELL_SM[kv[0][1]])):
            parts = []
            for pair in seg_pairs:
                sm_, ssd_, sn_ = d["seg"].get(pair, (float("nan"), float("nan"), 0))
                sm_us = sm_ * 1000.0 if sm_ == sm_ else float("nan")
                ssd_us = ssd_ * 1000.0 if ssd_ == ssd_ else float("nan")
                parts.append(f"{pair[0]}-{pair[1]}:{sm_us:7.3f}+/-{ssd_us:5.3f}(n={sn_})")
            print(f"  {arm:>4} {cell:>5}: " + "  ".join(parts))
        print("  NOTE: only 4 L points (512/1024/2048/4096) -> 3 segments, no independent "
              "quadratic-coefficient fit attempted. Extending L_LIST to include 8192 "
              "(DESIGN.md sec 7) would let curvature separate from sampling noise more cleanly.")

    print()
    print("=== intercept stability across prefill-SM cells (same arm) -- contamination check ===")
    by_arm = collections.defaultdict(dict)
    for (arm, cell), d in reg.items():
        by_arm[arm][cell] = d["intercept_mean"]
    for arm, cellmap in sorted(by_arm.items()):
        vals = [v for v in cellmap.values() if v == v]
        if len(vals) < 2:
            continue
        spread = (max(vals) - min(vals)) / st.fmean(vals) if st.fmean(vals) else float("nan")
        flag = " <-- LARGE SWING: admission/scheduling constant is NOT SM-independent, " \
               "read slope-ratio headline with this caveat" if spread > 0.30 else ""
        detail = ", ".join(f"{c}={v:.2f}ms" for c, v in sorted(cellmap.items(),
                                                                 key=lambda kv: CELL_SM[kv[0]]))
        print(f"  {arm:>4}: [{detail}]  spread={spread*100:.1f}%{flag}")

    print()
    print("=== HEADLINE: prefill-SM sensitivity = slope ratio (min-P cell / max-P cell) ===")
    cells_by_arm = collections.defaultdict(set)
    for arm, cell in reg:
        cells_by_arm[arm].add(cell)
    for arm, cs in sorted(cells_by_arm.items()):
        lo_cell = min(cs, key=lambda c: CELL_SM[c])
        hi_cell = max(cs, key=lambda c: CELL_SM[c])
        lo, hi = reg[(arm, lo_cell)], reg[(arm, hi_cell)]
        if lo["slope_n"] and hi["slope_n"] and hi["slope_mean"]:
            ratio = lo["slope_mean"] / hi["slope_mean"]
            cite_ok = lo["slope_n"] >= 4 and hi["slope_n"] >= 4
            tag = "" if cite_ok else "  [n<4 on one side -- do not cite]"
            print(f"  {arm:>4}: P{CELL_SM[lo_cell]}/P{CELL_SM[hi_cell]} slope ratio = "
                  f"{ratio:5.2f}x  (n={lo['slope_n']}/{hi['slope_n']}){tag}")

    # ------------------------------------------------------------------
    # AUXILIARY (secondary): rep-pooled absolute latency at matched SM.
    # ------------------------------------------------------------------
    if not lat_by:
        print("\nNo attributable intervals -- skipping the auxiliary matched-SM x L table.")
        return

    arms = sorted({a for a, _, _ in lat_by})
    ls = sorted({l for _, l, _ in lat_by})
    sms = sorted({s for _, _, s in lat_by if s is not None})
    print()
    print("=== AUXILIARY (secondary, rep-pooled, biased toward 1x by the constant admission/"
          "decode-step/network term -- see module docstring): probe latency p50 (ms) at "
          "MATCHED realized prefill-SM, by L ===")
    print(f"{'arm':>4} {'L':>6} | " + " ".join(f"{('P'+str(s)):>18}" for s in sms))
    table = {}
    for arm in arms:
        for L in ls:
            cellsr = []
            for s in sms:
                v = lat_by.get((arm, L, s), [])
                table[(arm, L, s)] = st.median(v) if v else None
                cellsr.append(f"{st.median(v):7.2f}(n={len(v):4d})" if v else " " * 18)
            print(f"{arm:>4} {L:>6} | " + " ".join(cellsr))

    print()
    print("=== AUXILIARY: absolute-latency ratio at matched exposure, P-min/P-max ===")
    for arm in arms:
        for L in ls:
            lo = table.get((arm, L, min(sms)))
            hi = table.get((arm, L, max(sms)))
            if lo and hi:
                print(f"  {arm:>4} L{L:<6} P{min(sms)}/P{max(sms)} = "
                      f"{lo:7.2f}/{hi:6.2f} = {lo/hi:5.2f}x")


if __name__ == "__main__":
    main()
