#!/usr/bin/env python3
"""m3_conditional.py -- partition-CONDITIONAL per-token ITL estimator for the
E1/M3 grid, plus the client<->telemetry clock alignment it needs.

STATUS: DIAGNOSTIC / DESCRIPTIVE.  Produced by claims-auditor on 2026-08-03 while
auditing the "dilution attenuation" claim about job 872077.  Nothing here is a
verdict, and per DESIGN.md sec 4.3.8(c)'s forward-only discipline an estimator
chosen AFTER seeing 872077 may not be used to adopt a verdict FOR 872077 (that is
the re-score-vs-re-tune confound wearing a new hat).  Thresholds must be
pre-registered before the sticky-partition run and bind only on it.

WHY IT EXISTS
-------------
On this substrate `decode_sms == D`  <=>  `split_prefill_batch in flight AND
decode batch non-empty` (multiplex/multiplexing_mixin.py:773,792-794; the revert
target `real_sm_group_num - 1` is hard-coded to the plain (0,108) stream by
multiplex/pdmux_context.py:initialize_stream_groups).  So a cell label is NOT a
sustained allocation (CONSENSUS sec 1-25), and any per-cell ITL statistic mixes
tokens that ran at D with tokens that ran at 108.  This script labels every
individual ITL interval with the partition that was actually realized while that
token was produced, and reports statistics on the labelled sub-populations.

It also replaces `A_free`'s prompt-size blocking heuristic with a direct
prefill-overlap label (see --prefill-overlap).

USAGE
  python3 m3_conditional.py --job 872077                 # everything
  python3 m3_conditional.py --job 872077 --only align    # alignment table only
  python3 m3_conditional.py --job 872077 --cache mc.pkl  # reuse the labelling

The labelling pass costs ~4 min for 64 probes; it is cached.
"""
import argparse, collections, json, math, os, pickle, statistics, sys
import numpy as np

# ---------------------------------------------------------------- constants
RATE_DEFAULT   = 2.0     # e1_m3_control.sbatch RATE
WARMUP_S       = 3.0     # e1_m3_control.sbatch WARMUP_S
PREFILL_BLOCK_TOK = 1024 # e1_m3_control.sbatch's A_free window threshold
ARMS  = ("Ha8", "T8")
CELLS = ("d16", "d24", "d44", "d54")
BLOCKS = tuple(range(1, 9))
NUM_CELL, DEN_CELL = "d16", "d54"

# label bands on split_frac (duration-weighted share of the ITL interval spent
# at decode_sms == D).  The band is deliberately WIDE: everything in between is
# dropped as AMBIGUOUS rather than forced into a class.
SPLIT_HI  = 0.90
UNSPLIT_LO = 0.10

# alignment gate (see sec 2 of the handoff): report-only, never a silent drop.
ALIGN_R_MIN = 0.95

# same table, same keying (by n, not n-1), as m3_analyze.py:60
TCRIT = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447,
         8: 2.365, 9: 2.306, 10: 2.262, 12: 2.201, 16: 2.131}


def t_ci(vals):
    n = len(vals)
    if n < 2:
        return (float("nan"), float("nan"))
    m = statistics.fmean(vals)
    h = TCRIT.get(n, 1.96) * statistics.stdev(vals) / math.sqrt(n)
    return (m - h, m + h)


def pctl(a, q):
    """Same interpolation rule as e1_m3_control.sbatch:257-260 (q in [0,1])."""
    if not len(a):
        return float("nan")
    a = sorted(a)
    p = q * (len(a) - 1)
    lo = int(p)
    hi = min(lo + 1, len(a) - 1)
    return a[lo] + (a[hi] - a[lo]) * (p - lo)


# ------------------------------------------------------------- 1. alignment
def load_telemetry(path):
    """Returns (t0_benchmark_marker, rows) with rows sorted by monotonic ts.

    NOTE the phase filter: runtime_snapshots are taken WITHOUT filtering on
    phase == 'benchmark'.  The 'benchmark' marker fires at the first request the
    server sees, which is bench_serving's WARM-UP request, so the marker is only
    an anchor for the lag search -- not a probe boundary.
    """
    t0b, rows = None, []
    for line in open(path):
        try:
            e = json.loads(line)
        except Exception:
            continue
        ev = e.get("event")
        if ev == "phase_marker" and e.get("phase") == "benchmark" and t0b is None:
            t0b = e["timestamp_monotonic_s"]
        elif ev == "runtime_snapshot":
            rows.append((e["timestamp_monotonic_s"],
                         e.get("decode_sms"),
                         e.get("decode_running_batch_size", 0),
                         e.get("prefill_sms"),
                         e.get("prefill_active_batch_size", 0)))
    rows.sort(key=lambda r: r[0])
    return t0b, rows


def replay_arrivals(seed, rate, n):
    """BIT-IDENTICAL to e1_m3_control.sbatch:284-287 -- same RNG, same order.
    Do not 'improve' this: it must reproduce the harness's own reconstruction,
    including its known limitations (it is the DRAWN schedule, not the client's
    realised send times; see CONSENSUS sec 3-14 on arrival_rps)."""
    np.random.seed(int(seed))
    iv = [np.random.exponential(1.0 / rate) for _ in range(n)]
    return np.cumsum([0.0] + iv[:-1])


def align(rows, t0b, arr, ttfts, itls):
    """Find the lag L (seconds) such that client time t corresponds to
    telemetry time t0b + L + t, by maximising Pearson r between the
    client-side in-flight count and telemetry decode_running_batch_size.

    Returns (lag, r).  Coarse scan then refine -- exactly the two-stage search
    used for the audit numbers.
    """
    ev = []
    for i in range(len(ttfts)):
        ev.append((arr[i], 1))
        ev.append((arr[i] + ttfts[i] + sum(itls[i]), -1))
    ev.sort(key=lambda x: x[0])
    evt = np.array([x[0] for x in ev])
    cum = np.cumsum([x[1] for x in ev])
    times = np.array([r[0] - t0b for r in rows])
    drb = np.array([r[2] for r in rows], dtype=float)

    def inflight_at(ts):
        idx = np.searchsorted(evt, ts, side="right") - 1
        return np.where(idx >= 0, cum[np.clip(idx, 0, len(cum) - 1)], 0).astype(float)

    best = (None, -2.0)
    for lag in np.arange(-5, 60, 0.5):
        a = inflight_at(times - lag)
        if a.std() == 0:
            continue
        r = float(np.corrcoef(a, drb)[0, 1])
        if r > best[1]:
            best = (float(lag), r)
    for lag in np.arange(best[0] - 0.5, best[0] + 0.5, 0.02):
        a = inflight_at(times - lag)
        if a.std() == 0:
            continue
        r = float(np.corrcoef(a, drb)[0, 1])
        if r > best[1]:
            best = (float(lag), r)
    return best


# ------------------------------------------------------- 2. per-token labels
def label_probe(d, arm, cell, block, seed, job, rate=RATE_DEFAULT):
    """Returns dict with recs = [(itl_ms, a_free_kept, split_frac,
    prefill_overlap_frac, req_index), ...] for every post-warmup, error-free
    ITL interval, plus alignment + engagement diagnostics."""
    tag = f"e1m3_{arm}_{cell}_b{block}_{job}"
    o = json.loads(open(f"{d}/{tag}_s{seed}.jsonl").read().strip().split("\n")[0])
    t0b, rows = load_telemetry(f"{d}/{tag}_telemetry.jsonl")
    Dsm = int(cell[1:])

    tt, il = o["ttfts"], o["itls"]
    inp = o["input_lens"]
    err = o.get("errors") or []
    arr = replay_arrivals(seed, rate, len(tt))
    lag, r = align(rows, t0b, arr, tt, il)

    ct = np.array([row[0] - t0b - lag for row in rows])          # client clock
    dsm = np.array([row[1] if row[1] is not None else -1 for row in rows])
    drb = np.array([row[2] for row in rows], dtype=float)
    pab = np.array([row[4] for row in rows], dtype=float)

    # E1_DECODE_REALIZED, recomputed here so the estimator and the gate come
    # from ONE pass (e1_pin_check.compute_decode_realized is the reference; this
    # reproduces it to 3 decimals on all 64 cell-blocks of 872077).
    t_tot = t_targ = 0.0
    for i in range(len(ct) - 1):
        if drb[i] > 0:
            dt = max(0.0, ct[i + 1] - ct[i])
            t_tot += dt
            if dsm[i] == Dsm:
                t_targ += dt
    w_time = (t_targ / t_tot) if t_tot else float("nan")

    def frac(a, b, mask):
        """duration-weighted share of [a,b] in which `mask` (a per-snapshot
        boolean array) holds, using the snapshot step function."""
        if b <= a:
            return 0.0
        i0 = max(np.searchsorted(ct, a, side="right") - 1, 0)
        i1 = max(np.searchsorted(ct, b, side="right") - 1, 0)
        tot = hit = 0.0
        for k in range(i0, min(i1 + 1, len(ct) - 1)):
            s = max(a, ct[k]); e = min(b, ct[k + 1])
            if e <= s:
                continue
            tot += e - s
            if mask[k]:
                hit += e - s
        return (hit / tot) if tot > 0 else 0.0

    m_split = (dsm == Dsm)
    m_pref = (pab > 0)

    nw = int(math.ceil(WARMUP_S * rate))
    keep = [i for i in range(nw, len(tt)) if (err[i] if i < len(err) else "") == ""]
    win = [(arr[j], arr[j] + tt[j]) for j in range(len(tt))
           if j < len(inp) and inp[j] >= PREFILL_BLOCK_TOK]

    recs = []
    for i in keep:
        if not il[i]:
            continue
        t = arr[i] + tt[i]
        for dd in il[i]:
            a, b = t, t + dd
            a_free = not any(b > w0 and a < w1 for w0, w1 in win)
            recs.append((dd * 1000.0, a_free, frac(a, b, m_split),
                         frac(a, b, m_pref), i))
            t = b
    return dict(arm=arm, cell=cell, block=block, seed=seed, Dsm=Dsm,
                lag=lag, align_r=r, w_time=w_time, n_keep=len(keep),
                n_blocking_windows=len(win), recs=recs)


def build(d, job, cache):
    if cache and os.path.exists(cache):
        return pickle.load(open(cache, "rb"))
    out = {}
    for arm in ARMS:
        for cell in CELLS:
            for b in BLOCKS:
                out[(arm, cell, b)] = label_probe(d, arm, cell, b, b, job)
                print(f"  labelled {arm} {cell} b{b}", file=sys.stderr)
    if cache:
        pickle.dump(out, open(cache, "wb"))
    return out


# ------------------------------------------------------------ 3. estimators
def sel_split(rec):    return rec[2] >= SPLIT_HI
def sel_unsplit(rec):  return rec[2] <= UNSPLIT_LO
def sel_all(rec):      return True


def tokens(recs, sel, a_free_only=True, block_free=None):
    """The PRIMARY population: individual ITL values (ms), one per emitted
    token.  No per-request inner aggregation -- that is the whole point."""
    out = []
    for rec in recs:
        if a_free_only and not rec[1]:
            continue
        if block_free is not None and rec[3] > block_free:
            continue
        if sel(rec):
            out.append(rec[0])
    return out


def afree_form(recs, sel):
    """`A_free` in its EXACT harness form (p95 over requests of per-request
    p95), restricted to a labelled sub-population.  Reported only so the new
    estimator can be compared against the old one on the same data."""
    by = collections.defaultdict(list)
    for rec in recs:
        if rec[1] and sel(rec):
            by[rec[4]].append(rec[0])
    per = [pctl(v, 0.95) for v in by.values() if v]
    return pctl(per, 0.95)


# ------------------------------------------------------------- 4. reporting
def report_align(O):
    print("\n=== [1] CLOCK ALIGNMENT (per probe) ===")
    print("  lag = client t=0 sits at telemetry (benchmark_marker + lag).  The"
          " marker fires at\n  bench_serving's WARM-UP request, hence lag >> 0.")
    print(f"  {'arm':>4} {'cell':>5} {'blk':>3} {'lag_s':>7} {'pearson_r':>10}  flag")
    weak = []
    for arm in ARMS:
        for cell in CELLS:
            for b in BLOCKS:
                p = O[(arm, cell, b)]
                fl = "" if p["align_r"] >= ALIGN_R_MIN else "ALIGN-WEAK"
                if fl:
                    weak.append((arm, cell, b, p["align_r"]))
                print(f"  {arm:>4} {cell:>5} {b:>3} {p['lag']:7.2f} {p['align_r']:10.3f}  {fl}")
    print(f"  ALIGN_R_MIN={ALIGN_R_MIN}; ALIGN-WEAK probes: {weak if weak else 'none'}")


def report_engagement(O):
    print("\n=== [2] ENGAGEMENT + LABEL YIELD (mean over 8 blocks) ===")
    print(f"  {'arm':>4} {'cell':>5} {'w_time':>7} {'w_token':>8} {'amb_frac':>9} "
          f"{'n_split':>8} {'n_unspl':>8} {'a_free_drop':>12}")
    for arm in ARMS:
        for cell in CELLS:
            wt, wk, am, ns, nu, ad = [], [], [], [], [], []
            for b in BLOCKS:
                p = O[(arm, cell, b)]
                rc = [r for r in p["recs"] if r[1]]
                n = len(rc)
                wt.append(p["w_time"])
                wk.append(sum(1 for r in rc if r[2] >= 0.5) / n)
                am.append(sum(1 for r in rc if UNSPLIT_LO < r[2] < SPLIT_HI) / n)
                ns.append(sum(1 for r in rc if sel_split(r)))
                nu.append(sum(1 for r in rc if sel_unsplit(r)))
                ad.append(1 - n / len(p["recs"]))
            print(f"  {arm:>4} {cell:>5} {np.mean(wt):7.3f} {np.mean(wk):8.3f} "
                  f"{np.mean(am):9.3f} {np.mean(ns):8.0f} {np.mean(nu):8.0f} {np.mean(ad):12.3f}")


def report_conditional(O):
    print("\n=== [3] PRIMARY: partition-conditional per-token ITL (pooled over 8 blocks) ===")
    print(f"  {'arm':>4} {'cell':>5} | {'n_sp':>7} {'sp_p50':>7} {'sp_p95':>7} {'sp_mean':>8} "
          f"| {'n_un':>7} {'un_p50':>7} {'un_p95':>7} {'un_mean':>8}")
    for arm in ARMS:
        for cell in CELLS:
            SP, UN = [], []
            for b in BLOCKS:
                rc = O[(arm, cell, b)]["recs"]
                SP += tokens(rc, sel_split)
                UN += tokens(rc, sel_unsplit)
            print(f"  {arm:>4} {cell:>5} | {len(SP):7d} {pctl(SP,.50):7.2f} {pctl(SP,.95):7.2f} "
                  f"{np.mean(SP):8.2f} | {len(UN):7d} {pctl(UN,.50):7.2f} {pctl(UN,.95):7.2f} "
                  f"{np.mean(UN):8.2f}")

    print("\n=== [4] PAIRED-WITHIN-BLOCK RATIO d16/d54, block-clustered t-CI (n=8) ===")
    stats = (("sp_p95", sel_split, lambda a: pctl(a, .95)),
             ("un_p95", sel_unsplit, lambda a: pctl(a, .95)),
             ("sp_p50", sel_split, lambda a: pctl(a, .50)),
             ("un_p50", sel_unsplit, lambda a: pctl(a, .50)),
             ("sp_mean", sel_split, np.mean),
             ("un_mean", sel_unsplit, np.mean))
    for arm in ARMS:
        for name, sel, agg in stats:
            num, den = [], []
            for b in BLOCKS:
                num.append(agg(tokens(O[(arm, NUM_CELL, b)]["recs"], sel)))
                den.append(agg(tokens(O[(arm, DEN_CELL, b)]["recs"], sel)))
            rr = [num[i] / den[i] for i in range(len(BLOCKS))]
            lo, hi = t_ci(rr)
            print(f"  {arm:>4} {name:>8}: {np.mean(rr):.3f} [{lo:.3f},{hi:.3f}] "
                  f"sd={statistics.stdev(rr):.3f}  d16={np.mean(num):7.2f} d54={np.mean(den):7.2f}")

    print("\n=== [5] SAME CONTRAST IN THE OLD `A_free` FORM (for comparison only) ===")
    for arm in ARMS:
        for name, sel in (("ALL", sel_all), ("SPLIT-only", sel_split),
                          ("UNSPLIT-only", sel_unsplit)):
            num = [afree_form(O[(arm, NUM_CELL, b)]["recs"], sel) for b in BLOCKS]
            den = [afree_form(O[(arm, DEN_CELL, b)]["recs"], sel) for b in BLOCKS]
            rr = [num[i] / den[i] for i in range(len(BLOCKS))]
            lo, hi = t_ci(rr)
            print(f"  {arm:>4} {name:>13}: g={np.mean(rr):.3f} [{lo:.3f},{hi:.3f}]  "
                  f"d16={np.mean(num):7.2f} d54={np.mean(den):7.2f}")


def report_leave_one_out(O):
    print("\n=== [6] LEAVE-ONE-BLOCK-OUT on the primary ratio (align robustness) ===")
    for arm in ARMS:
        num = [pctl(tokens(O[(arm, NUM_CELL, b)]["recs"], sel_split), .95) for b in BLOCKS]
        den = [pctl(tokens(O[(arm, DEN_CELL, b)]["recs"], sel_split), .95) for b in BLOCKS]
        rr = [num[i] / den[i] for i in range(len(BLOCKS))]
        outs = []
        for k in range(len(BLOCKS)):
            sub = [rr[i] for i in range(len(rr)) if i != k]
            outs.append(f"b{BLOCKS[k]}:{np.mean(sub):.3f}")
        print(f"  {arm:>4} sp_p95 full={np.mean(rr):.3f}  LOO -> " + " ".join(outs))


def report_tail(O):
    print("\n=== [7] IS THE TAIL ENRICHED IN SPLIT TOKENS? ===")
    for arm in ARMS:
        for cell in CELLS:
            fr, ov = [], []
            for b in BLOCKS:
                rc = [r for r in O[(arm, cell, b)]["recs"] if r[1]]
                v = sorted(rc, key=lambda x: -x[0])
                k = max(1, int(0.05 * len(v)))
                fr.append(sum(1 for x in v[:k] if x[2] >= 0.5) / k)
                ov.append(sum(1 for x in rc if x[2] >= 0.5) / len(rc))
            print(f"  {arm:>4} {cell}: split share of top-5% tail={np.mean(fr):.3f} "
                  f"overall={np.mean(ov):.3f}  enrichment x{np.mean(fr)/np.mean(ov):.1f}")


def report_prefill_overlap(O):
    print("\n=== [8] REPLACEMENT FOR THE 1024-TOKEN BLOCKING FILTER ===")
    print("  BLOCK-FREE := prefill_overlap_frac <= 0.0 (no prefill executing during the")
    print("  interval), measured from prefill_active_batch_size, NOT from prompt size.")
    print(f"  {'arm':>4} {'cell':>5} {'n_all':>8} {'A_free keeps':>13} {'overlap-free keeps':>19} "
          f"{'p95_all':>8} {'p95_ofree':>10}")
    for arm in ARMS:
        for cell in CELLS:
            na, ka, kb, pa, pb = [], [], [], [], []
            for b in BLOCKS:
                rc = O[(arm, cell, b)]["recs"]
                na.append(len(rc))
                ka.append(sum(1 for r in rc if r[1]) / len(rc))
                of = [r[0] for r in rc if r[3] <= 0.0]
                kb.append(len(of) / len(rc))
                pa.append(pctl([r[0] for r in rc], .95))
                pb.append(pctl(of, .95))
            print(f"  {arm:>4} {cell:>5} {np.mean(na):8.0f} {np.mean(ka):13.3f} "
                  f"{np.mean(kb):19.3f} {np.mean(pa):8.2f} {np.mean(pb):10.2f}")


def report_deengagement(O, seed=0):
    """Diagnostic that quantified how much of A_free the labelled split tokens
    can possibly be responsible for.  ORDER-DEPENDENT: one rng is drawn across
    the whole nested loop, exactly as in the audit run, so the loop order below
    must not be changed if the audit's printed numbers are to reproduce."""
    print("\n=== [9] DE-ENGAGEMENT DIAGNOSTIC (order-dependent RNG; see docstring) ===")
    rng = np.random.default_rng(seed)
    for arm in ARMS:
        for cell in (NUM_CELL, DEN_CELL):
            print(f"  {arm} {cell}:")
            for f in (1.0, 0.75, 0.5, 0.25, 0.0):
                vals = []
                for b in BLOCKS:
                    rc = [r for r in O[(arm, cell, b)]["recs"] if r[1]]
                    un = np.array([r[0] for r in rc if r[2] <= UNSPLIT_LO])
                    by = collections.defaultdict(list)
                    for v, _fr, sf, _po, i in rc:
                        if sf >= 0.5 and rng.random() > f:
                            v = float(rng.choice(un))
                        by[i].append(v)
                    vals.append(pctl([pctl(x, .95) for x in by.values() if x], .95))
                print(f"     w x {f:4.2f}: A_free = {np.mean(vals):7.2f}")


def report_threshold_sweep(d, job, rate=RATE_DEFAULT):
    """Is `A_free`'s blocking filter fixable by re-tuning PREFILL_BLOCK_TOK?
    Sweeps the threshold using ONLY client-side data (windows are
    [arrival, arrival+TTFT] of prompts with input_len >= TH), so this table is
    independent of telemetry AND of the clock alignment -- it is immune to every
    instrument objection that can be raised against reports [3]-[7].

    ADDED 2026-08-03 in the handoff turn.  NOT part of the audited set: it has
    not been through an adversarial pass and must not enter canon on its own.
    """
    print("\n=== [10] BLOCKING-THRESHOLD SWEEP (client-side only; no telemetry) ===")
    print("  entries are  keep_frac / pooled-p95 / A_free-form   (ms)")
    hdr = ("1024 (as-run)", "512", "256", "0 (all prompts)")
    print(f"  {'arm':>4} {'cell':>5} " + " ".join(f"{h:>22}" for h in hdr))
    for arm in ARMS:
        for cell in (NUM_CELL, DEN_CELL):
            cols = []
            for TH in (1024, 512, 256, 0):
                ks, ps, afs = [], [], []
                for b in BLOCKS:
                    tag = f"e1m3_{arm}_{cell}_b{b}_{job}"
                    o = json.loads(open(f"{d}/{tag}_s{b}.jsonl").read().strip().split("\n")[0])
                    tt, il, inp = o["ttfts"], o["itls"], o["input_lens"]
                    err = o.get("errors") or []
                    arr = replay_arrivals(b, rate, len(tt))
                    win = [(arr[j], arr[j] + tt[j]) for j in range(len(tt))
                           if j < len(inp) and inp[j] >= TH]
                    nw = int(math.ceil(WARMUP_S * rate))
                    keep = [i for i in range(nw, len(tt))
                            if (err[i] if i < len(err) else "") == ""]
                    tot, kept = 0, []
                    by = collections.defaultdict(list)
                    for i in keep:
                        t = arr[i] + tt[i]
                        for dd in il[i]:
                            a, bb = t, t + dd
                            tot += 1
                            if not any(bb > w0 and a < w1 for w0, w1 in win):
                                kept.append(dd * 1000.0)
                                by[i].append(dd * 1000.0)
                            t = bb
                    ks.append(len(kept) / tot)
                    ps.append(pctl(kept, .95))
                    afs.append(pctl([pctl(v, .95) for v in by.values() if v], .95))
                cols.append(f"{np.mean(ks):.3f}/{np.mean(ps):6.2f}/{np.mean(afs):7.2f}")
            print(f"  {arm:>4} {cell:>5} " + " ".join(f"{c:>22}" for c in cols))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--job", required=True)
    ap.add_argument("--cache", default=None,
                    help="pickle path for the labelling pass (recommended)")
    ap.add_argument("--only", default="all",
                    choices=["all", "align", "engagement", "conditional",
                             "loo", "tail", "overlap", "deengage", "thresh"])
    a = ap.parse_args()
    O = build(a.dir, a.job, a.cache)
    f = a.only
    if f in ("all", "align"):       report_align(O)
    if f in ("all", "engagement"):  report_engagement(O)
    if f in ("all", "conditional"): report_conditional(O)
    if f in ("all", "loo"):         report_leave_one_out(O)
    if f in ("all", "tail"):        report_tail(O)
    if f in ("all", "overlap"):     report_prefill_overlap(O)
    if f in ("all", "deengage"):    report_deengagement(O)
    if f in ("all", "thresh"):      report_threshold_sweep(a.dir, a.job)
    print("\nDIAGNOSTIC ONLY -- no verdict may be adopted for job 872077 from this "
          "estimator\n(chosen after seeing the data).  Pre-register thresholds before "
          "the sticky run.")


if __name__ == "__main__":
    main()
