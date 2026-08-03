#!/usr/bin/env python3
"""C2 (results/s8_scaleup) re-analysis -> principled anchor for sticky `G_LEVER`.

WHY THIS EXISTS
---------------
The sticky pre-registration (`results/s8_frontier/DESIGN.md` Sec 4.3.12(d)) records
`G_LEVER`/`G_FLAT` as UNDETERMINED: the old 1.5/1.15 were scaled for the retired
`A_free` statistic, and the replacement estimator (Sec 4.3.10) is a *per-token ITL
quantile*.  The single live argument for a principled threshold is that the new
estimator sits on the SAME AXIS as C2 (per-token ITL quantiles), so C2's measured
decode-SM elasticity can anchor `G_LEVER`.  C2's headline (2.36-2.91x) is the
16->92 endpoint ratio; the sticky grid runs 16->54, so the PARTIAL ratio is what
is needed.

AGGREGATION UNITS (fixed BEFORE looking at any output; methodology gates #4/#5)
------------------------------------------------------------------------------
  unit          : one ITL interval = one emitted token (NO per-request aggregation)
                  -- matches Sec 4.3.10(1) exactly.
  label         : REALIZED partition, time-weighted over the interval
                  split_frac(a,b) = (time in [a,b] with decode_sms == D)/(b-a)
                  SPLIT := >= SPLIT_HI (0.90) / UNSPLIT108 := frac(decode_sms==108)
                  >= SPLIT_HI / AMBIGUOUS := neither (never force-classified).
                  Same rule and same 0.90 constant as Sec 4.3.10(1).
                  NOT target: Stage 0 lost a headline to a target-vs-realized
                  mix-up, and C2's own NOTES_865311_865312.md Sec 1 documents the
                  same auto-revert here.
  statistic     : direct per-token quantiles of the SPLIT population.
                  primary p95 (== Sec 4.3.10 primary), secondary p50 / mean.
  gate weighting: TIME-weighted residency (gate #4 corollary: gates are
                  time-weighted, attribution is per-interval).
  replicate     : rep index within a job (n=4).  Ratios are paired by rep index.
                  *** reps within a cell share ONE server boot, and d16/d44 are
                  DIFFERENT boots, so rep-level sd is pseudo-replication and
                  UNDERSTATES between-boot variance.  The between-job (v2 865493
                  vs v3 865533) spread is the only boot-level replication here.
                  This is the M3/`n_indep` lesson; it is reported, not hidden.

IDENTITY CHECK (gate #6): the quantity used for the ratio (per-token ITL inside
SPLIT intervals) is not definitionally tied to `decode_sms`; `decode_sms` is a
persisted state variable written by the arbiter, and the ITL is client-side
wall-clock.  The residency gate (fraction of decode-busy TIME at the target) and
the estimand (ITL magnitude) are likewise logically independent -- residency
could be 1.0 with any ITL.

Run:
  python3 c2_anchor.py                 # full report
  python3 c2_anchor.py --job 865493    # one job only
"""
import argparse
import bisect
import collections
import glob
import json
import math
import os
import re
import statistics as st

C2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "s8_scaleup")
C2 = os.path.normpath(C2)

PAT = re.compile(r"s8_deconf_(?P<arm>\w+?)_C(?P<ctx>\d+)_(?P<cell>d16|d24|d44|d54|d92|np)_"
                 r"(?P<job>\d+(?:_blk\d+)?)_telemetry\.jsonl$")
CELL_SM = {"d16": 16, "d24": 24, "d44": 44, "d54": 54, "d92": 92}
# d54 + the "<slurm_id>_blk<N>" job-field extension added 2026-08-03 for the
# C2 D=54 anchor block campaign (s8_sweep_d54.sbatch). Purely additive/
# mechanical: existing d16/d24/d44/d92/np filenames and plain-digit job ids
# still match unchanged. No SPLIT_HI/unit/statistic/gate definition touched.
# Each "<slurm_id>_blk<N>" string is treated as its own "job" by collect()/
# ratio_block(), which is exactly the desired unit here: one fresh boot pair
# of (d16, d54) = one block = one replicate for the r=stat(d16)/stat(d54)
# ratio (see d54_block_ratio.py).
SPLIT_HI = 0.90          # identical constant to DESIGN.md Sec 4.3.10(1)
GUARD_S = 3.0            # max telemetry gap tolerated inside an interval
WARMUP_S, MEASURE_S = 20.0, 60.0   # s8_sweep.sbatch defaults
TCRIT = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365}


def pct(xs, q):
    if not xs:
        return float("nan")
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * q
    lo = math.floor(k)
    hi = math.ceil(k)
    return s[lo] if lo == hi else s[lo] + (s[hi] - s[lo]) * (k - lo)


def load_telemetry(path):
    ts, sm, bs = [], [], []
    for line in open(path):
        try:
            e = json.loads(line)
        except Exception:
            continue
        if e.get("event") != "runtime_snapshot":
            continue
        ts.append(e["timestamp_monotonic_s"])
        sm.append(e.get("decode_sms"))
        bs.append(e.get("decode_running_batch_size", 0))
    return ts, sm, bs


def seg_fracs(ts, sm, bs, a, b):
    """Time-weighted fractions + mean batch over [a,b] on the telemetry step fn.

    Returns None when the interval is not fully covered or a telemetry gap
    inside it exceeds GUARD_S (we refuse to guess through a hole).
    """
    if not ts or a < ts[0] or b > ts[-1] or b <= a:
        return None
    i = bisect.bisect_right(ts, a) - 1
    acc = collections.defaultdict(float)
    bacc = 0.0
    k = i
    while k < len(ts) - 1 and ts[k] < b:
        seg_lo, seg_hi = max(ts[k], a), min(ts[k + 1], b)
        if ts[k + 1] - ts[k] > GUARD_S and seg_hi > seg_lo:
            return None
        if seg_hi > seg_lo:
            acc[sm[k]] += seg_hi - seg_lo
            bacc += bs[k] * (seg_hi - seg_lo)
        k += 1
    tot = b - a
    return {k2: v / tot for k2, v in acc.items()}, bacc / tot


def collect(job=None, ctx="1024"):
    """-> data[(arm, cell, rep)] = list of (itl_ms, split_frac, un_frac, mean_bs)"""
    data = collections.defaultdict(list)
    resid = {}
    meta = {}
    for fn in sorted(os.listdir(C2)):
        m = PAT.search(fn)
        if not m:
            continue
        g = m.groupdict()
        if g["ctx"] != ctx or (job and g["job"] != job):
            continue
        arm, cell, jb = g["arm"], g["cell"], g["job"]
        if cell not in CELL_SM:
            continue
        D = CELL_SM[cell]
        ts, sm, bs = load_telemetry(os.path.join(C2, fn))
        if not ts:
            continue
        # --- realized-residency gate, TIME-weighted, conditional on decode busy
        busy_t, tgt_t, un_t, other_t, bw = 0.0, 0.0, 0.0, 0.0, 0.0
        for k in range(len(ts) - 1):
            dt = ts[k + 1] - ts[k]
            if dt > GUARD_S or bs[k] <= 0:
                continue
            busy_t += dt
            bw += bs[k] * dt
            if sm[k] == D:
                tgt_t += dt
            elif sm[k] == 108:
                un_t += dt
            else:
                other_t += dt
        resid[(arm, cell, jb)] = dict(
            busy_s=busy_t,
            frac_target=tgt_t / busy_t if busy_t else float("nan"),
            frac_108=un_t / busy_t if busy_t else float("nan"),
            frac_other=other_t / busy_t if busy_t else float("nan"),
            mean_bs_busy=bw / busy_t if busy_t else float("nan"),
        )
        # --- client t0 per rep
        t0s = {}
        for res in glob.glob(os.path.join(C2, f"s8_deconf_{arm}_C{ctx}_{jb}_result.txt")):
            for line in open(res):
                if not line.startswith("{"):
                    continue
                try:
                    s = json.loads(line)
                except Exception:
                    continue
                tag = s.get("tag", "")
                if tag.startswith(f"deconf_{arm}_C{ctx}_{cell}_r") and "t0_monotonic_s" in s:
                    r = int(tag.rsplit("_r", 1)[1])
                    t0s[r] = s["t0_monotonic_s"]
                    meta[(arm, cell, jb, r)] = dict(
                        occ=s.get("occupancy_mean_in_window"),
                        cell_p95=s.get("itl_ms_p95"), cell_p50=s.get("itl_ms_p50"),
                        cell_mean=s.get("itl_ms_mean"), n=s.get("itl_samples"))
        stem = os.path.join(C2, fn[: -len("_telemetry.jsonl")])
        for raw in sorted(glob.glob(f"{stem}_rep*_raw.jsonl")):
            rep = int(re.search(r"_rep(\d+)_raw", raw).group(1))
            t0 = t0s.get(rep)
            if t0 is None:
                continue
            lo, hi = WARMUP_S, WARMUP_S + MEASURE_S
            for line in open(raw):
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                times = r.get("chunk_times_s", [])
                # index >= 2 mirrors s0dc_client.py: times[1]-times[0] is the
                # first post-prefill gap, not a steady-state ITL.
                for i in range(2, len(times)):
                    if not (lo <= times[i] <= hi):
                        continue
                    a, b = times[i - 1] + t0, times[i] + t0
                    got = seg_fracs(ts, sm, bs, a, b)
                    if got is None:
                        continue
                    fr, mbs = got
                    data[(arm, cell, jb, rep)].append(
                        ((b - a) * 1000.0, fr.get(D, 0.0), fr.get(108, 0.0), mbs, b - a))
    return data, resid, meta


def stats_of(vals):
    return dict(p95=pct(vals, 0.95), p50=pct(vals, 0.50),
                mean=st.fmean(vals) if vals else float("nan"), n=len(vals))


def ci_t(xs):
    n = len(xs)
    if n < 2:
        return (float("nan"), float("nan"), float("nan"), float("nan"))
    m, sd = st.fmean(xs), st.stdev(xs)
    t = TCRIT.get(n, 2.0)
    h = t * sd / math.sqrt(n)
    return (m, sd, m - h, m + h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", default=None)
    ap.add_argument("--ctx", default="1024")
    ap.add_argument("--lo-cell", default="d16")
    ap.add_argument("--hi-cell", default="d44")
    a = ap.parse_args()

    data, resid, meta = collect(a.job, a.ctx)
    jobs = sorted({k[2] for k in data})
    arms = sorted({k[0] for k in data})
    cells = [c for c in ("d16", "d24", "d44", "d92") if any(k[1] == c for k in data)]

    print("=" * 78)
    print("C2 -> G_LEVER anchor.  unit=ITL interval (per token) | label=REALIZED,")
    print(f"time-weighted, SPLIT>= {SPLIT_HI} | primary stat = per-token p95")
    print("=" * 78)

    # Estimand-matched gate: time-weighted over the ITL intervals that actually
    # contribute (in-window), computed BEFORE any SPLIT labeling, so it is not
    # circular (gate #6).  The all-busy telemetry version below it is the
    # as-run number and includes warmup + inter-rep drain, where decode runs
    # unsplit at 108 with no keepalive prefill present -- that is why the two
    # columns differ, and why only the in-window one answers "did the measured
    # tokens run at D".
    pooled_all = {}
    for (arm, cell, jb, rep), rows in sorted(data.items()):
        pooled_all.setdefault((arm, cell, jb), []).extend(rows)

    print("\n[1] REALIZED-PARTITION GATE")
    print("    in-window = time-weighted over contributing ITL intervals (estimand-matched)")
    print("    all-busy  = time-weighted over ALL decode-busy telemetry (incl. warmup/drain)")
    print(f"{'job':>7} {'arm':>4} {'cell':>5} | {'inwin@D':>8} {'inwin@108':>10} | "
          f"{'allbusy@D':>10} {'allbusy@108':>12} {'mean_bs':>8} | gate(in-win>=0.90)")
    for (arm, cell, jb), rows in sorted(pooled_all.items(),
                                        key=lambda x: (x[0][2], x[0][0], x[0][1])):
        v = resid.get((arm, cell, jb), {})
        tot = sum(r[4] for r in rows)
        wD = sum(r[1] * r[4] for r in rows) / tot if tot else float("nan")
        w108 = sum(r[2] * r[4] for r in rows) / tot if tot else float("nan")
        ok = "PASS" if wD >= 0.90 else "FAIL"
        print(f"{jb:>7} {arm:>4} {cell:>5} | {wD:8.3f} {w108:10.3f} | "
              f"{v.get('frac_target', float('nan')):10.3f} {v.get('frac_108', float('nan')):12.3f} "
              f"{v.get('mean_bs_busy', float('nan')):8.2f} | {ok}")

    print("\n[2] INTERVAL LABEL YIELD (per cell, pooled over reps)")
    print(f"{'job':>7} {'arm':>4} {'cell':>5} {'n_tot':>7} {'SPLIT':>7} {'UN108':>7} "
          f"{'AMBIG':>7} {'amb_frac':>9}")
    pooled = {}
    for (arm, cell, jb, rep), rows in sorted(data.items()):
        pooled.setdefault((arm, cell, jb), []).extend(rows)
    for (arm, cell, jb), rows in sorted(pooled.items(), key=lambda x: (x[0][2], x[0][0], x[0][1])):
        if cell not in (a.lo_cell, a.hi_cell):
            continue
        sp = [r for r in rows if r[1] >= SPLIT_HI]
        un = [r for r in rows if r[2] >= SPLIT_HI]
        amb = len(rows) - len(sp) - len(un)
        print(f"{jb:>7} {arm:>4} {cell:>5} {len(rows):7d} {len(sp):7d} {len(un):7d} "
              f"{amb:7d} {amb/max(len(rows),1):9.3f}")

    def ratio_block(lo_cell, hi_cell, sel, tag):
        print(f"\n  r = {tag}({lo_cell}) / {tag}({hi_cell})")
        print(f"{'job':>7} {'arm':>4} {'stat':>5} | {'lo':>19} | {'hi':>19} | "
              f"{'r pooled':>9} | {'r rep mean+-sd':>16} | {'t95 CI':>18}")
        for jb in jobs:
            for arm in arms:
                lo_rows = [r for r in pooled.get((arm, lo_cell, jb), []) if sel(r)]
                hi_rows = [r for r in pooled.get((arm, hi_cell, jb), []) if sel(r)]
                if len(lo_rows) < 100 or len(hi_rows) < 100:
                    continue
                slo = stats_of([r[0] for r in lo_rows])
                shi = stats_of([r[0] for r in hi_rows])
                for stat in ("p95", "p50", "mean"):
                    rp = slo[stat] / shi[stat]
                    rr = []
                    for rep in (1, 2, 3, 4):
                        lv = [r[0] for r in data.get((arm, lo_cell, jb, rep), []) if sel(r)]
                        hv = [r[0] for r in data.get((arm, hi_cell, jb, rep), []) if sel(r)]
                        if len(lv) >= 30 and len(hv) >= 30:
                            rr.append(stats_of(lv)[stat] / stats_of(hv)[stat])
                    m, sd, cl, ch = ci_t(rr)
                    print(f"{jb:>7} {arm:>4} {stat:>5} | {slo[stat]:9.2f}ms n={slo['n']:<6d} | "
                          f"{shi[stat]:9.2f}ms n={shi['n']:<6d} | {rp:9.3f} | "
                          f"{m:7.3f}+-{sd:6.3f} | [{cl:6.3f},{ch:6.3f}] n={len(rr)}")

    is_split = lambda r: r[1] >= SPLIT_HI          # noqa: E731
    is_un = lambda r: r[2] >= SPLIT_HI             # noqa: E731

    print(f"\n[3] PARTIAL RATIO, SPLIT population (label = realized decode_sms == D)")
    for hc in cells:
        if hc == a.lo_cell:
            continue
        ratio_block(a.lo_cell, hc, is_split, "SPLIT")

    print("\n[3b] NEGATIVE CONTROL: same contrast on the UNSPLIT(108) population.")
    print("     Both cells sit at 108 SM there, so the decode-SM contrast is zero")
    print("     by construction; r != 1 means the difference was not made by decode SM.")
    for hc in cells:
        if hc == a.lo_cell:
            continue
        ratio_block(a.lo_cell, hc, is_un, "UNSPLIT108")

    print("\n[4] BATCH-MATCHED PARTIAL RATIO (mean in-interval decode batch binned to int)")
    print("    C2's headline conditioned on batch; the new estimator does NOT.")
    print(f"{'job':>7} {'arm':>4} {'bin':>4} {'stat':>5} | {'lo':>9} {'hi':>9} | {'r':>7} | n_lo n_hi")
    for jb in jobs:
        for arm in arms:
            lo_rows = [r for r in pooled.get((arm, a.lo_cell, jb), []) if r[1] >= SPLIT_HI]
            hi_rows = [r for r in pooled.get((arm, a.hi_cell, jb), []) if r[1] >= SPLIT_HI]
            lob = collections.defaultdict(list)
            hib = collections.defaultdict(list)
            for r in lo_rows:
                lob[int(round(r[3]))].append(r[0])
            for r in hi_rows:
                hib[int(round(r[3]))].append(r[0])
            shared = sorted(set(lob) & set(hib),
                            key=lambda b: -min(len(lob[b]), len(hib[b])))
            for b in shared[:3]:
                if min(len(lob[b]), len(hib[b])) < 50:
                    continue
                for stat in ("p95", "p50"):
                    rl, rh = stats_of(lob[b])[stat], stats_of(hib[b])[stat]
                    print(f"{jb:>7} {arm:>4} {b:>4} {stat:>5} | {rl:9.2f} {rh:9.2f} | "
                          f"{rl/rh:7.3f} | {len(lob[b]):5d} {len(hib[b]):5d}")

    print("\n[5] UNCONDITIONED CELL SUMMARY (client-side, mixture of D and 108) "
          "-- contamination check")
    print(f"{'job':>7} {'arm':>4} {'cell':>5} {'occ':>6} {'p95':>9} {'p50':>9} {'n':>7}")
    for (arm, cell, jb, rep), v in sorted(meta.items(), key=lambda x: (x[0][2], x[0][0], x[0][1], x[0][3])):
        if cell not in (a.lo_cell, a.hi_cell) or rep != 1:
            continue
        agg = [meta[(arm, cell, jb, r)] for r in (1, 2, 3, 4) if (arm, cell, jb, r) in meta]
        print(f"{jb:>7} {arm:>4} {cell:>5} {st.fmean([x['occ'] for x in agg]):6.1f} "
              f"{st.fmean([x['cell_p95'] for x in agg]):9.2f} "
              f"{st.fmean([x['cell_p50'] for x in agg]):9.2f} "
              f"{int(st.fmean([x['n'] for x in agg])):7d}   (rep-mean, n=4)")


if __name__ == "__main__":
    main()
