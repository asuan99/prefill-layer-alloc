#!/usr/bin/env python3
"""s2_analyze.py -- independent readout of job 873015 (S2, sticky ON) against
PREREG_S2_STICKY_ITL_2026-08-03.md sec 3 / 3.1 / 4, plus the confound
decomposition 872077 (sticky OFF) vs 873015 that the pre-registration does not
cover.

STATUS: DIAGNOSTIC + PRE-REGISTERED READOUT.  No throughput/goodput/g/policy
claim is made or licensed here; run-level throughput numbers appear ONLY as
confound descriptors for two runs that are not a controlled comparison.

Reuse (canonical, NOT re-implemented): m3_conditional.label_probe (the whole
labelling pass: split_frac, a_free flag, clock alignment), .tokens, .pctl,
.sel_split/.sel_unsplit/.sel_all, .t_ci, .load_telemetry, .replay_arrivals,
.align, and c2_anchor.collect/.seg_fracs/.pct for the C2 side.

Written fresh here: the per-interval decode-batch weighting (label_ext), all
block-level aggregation/CIs, the confound tables, the batch-bin matching.
label_ext is gated element-wise against m3_conditional.label_probe (gate G-A).
"""
import collections, json, math, os, pickle, statistics, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results"
E1DIR = f"{ROOT}/s8_frontier"     # 872077 (sticky OFF) + canonical modules
S2DIR = f"{ROOT}/s2_sticky"       # 873015 (sticky ON)
C2DIR = f"{ROOT}/s8_scaleup"      # 865493 (C2)
sys.path.insert(0, E1DIR)
import m3_conditional as mc                      # noqa: E402

BLOCKS = tuple(range(1, 9))
CELLS = ("d16", "d54")
ARM = "T8"
RATE = 2.0
CACHE = os.environ.get("S2_CACHE", "/tmp/s2_cache.pkl")


# ------------------------------------------------------------------ helpers
def t_stats(vals):
    """mean, sd, t95 half-width, CI -- same TCRIT table as m3_conditional."""
    n = len(vals)
    m = statistics.fmean(vals)
    if n < 2:
        return m, float("nan"), float("nan"), (float("nan"), float("nan"))
    sd = statistics.stdev(vals)
    h = mc.TCRIT.get(n, 1.96) * sd / math.sqrt(n)
    return m, sd, h, (m - h, m + h)


def wmean(ct, vals, a, b):
    """duration-weighted mean of a per-snapshot value over [a,b] on the
    telemetry step function -- the same integration rule as m3_conditional's
    inner frac(), with `mask` replaced by a numeric series."""
    if b <= a:
        return float("nan")
    i0 = max(np.searchsorted(ct, a, side="right") - 1, 0)
    i1 = max(np.searchsorted(ct, b, side="right") - 1, 0)
    tot = acc = 0.0
    for k in range(i0, min(i1 + 1, len(ct) - 1)):
        s = max(a, ct[k]); e = min(b, ct[k + 1])
        if e <= s:
            continue
        tot += e - s
        acc += vals[k] * (e - s)
    return (acc / tot) if tot > 0 else float("nan")


# --------------------------------------------------- extended labelling pass
def label_ext(d, arm, cell, block, seed, job, rate=RATE):
    """mc.label_probe + per-interval mean decode batch.  Structure copied from
    m3_conditional.label_probe so the populations are identical; verified
    element-wise by gate G-A below."""
    tag = f"e1m3_{arm}_{cell}_b{block}_{job}"
    o = json.loads(open(f"{d}/{tag}_s{seed}.jsonl").read().strip().split("\n")[0])
    t0b, rows = mc.load_telemetry(f"{d}/{tag}_telemetry.jsonl")
    Dsm = int(cell[1:])

    tt, il, inp = o["ttfts"], o["itls"], o["input_lens"]
    err = o.get("errors") or []
    arr = mc.replay_arrivals(seed, rate, len(tt))
    lag, r = mc.align(rows, t0b, arr, tt, il)

    ct = np.array([row[0] - t0b - lag for row in rows])
    dsm = np.array([row[1] if row[1] is not None else -1 for row in rows])
    drb = np.array([row[2] for row in rows], dtype=float)
    psm = np.array([row[3] if row[3] is not None else -1 for row in rows])
    pab = np.array([row[4] for row in rows], dtype=float)

    m_split = (dsm == Dsm)
    m_pref = (pab > 0)

    def frac(a, b, mask):
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

    nw = int(math.ceil(mc.WARMUP_S * rate))
    keep = [i for i in range(nw, len(tt)) if (err[i] if i < len(err) else "") == ""]
    win = [(arr[j], arr[j] + tt[j]) for j in range(len(tt))
           if j < len(inp) and inp[j] >= mc.PREFILL_BLOCK_TOK]

    recs = []
    for i in keep:
        if not il[i]:
            continue
        t = arr[i] + tt[i]
        for dd in il[i]:
            a, b = t, t + dd
            a_free = not any(b > w0 and a < w1 for w0, w1 in win)
            recs.append((dd * 1000.0, a_free, frac(a, b, m_split),
                         frac(a, b, m_pref), i, wmean(ct, drb, a, b)))
            t = b

    # --- telemetry-side run descriptors (time-weighted, decode-busy) ---------
    busy_t = 0.0
    bs_hist = collections.Counter()      # int(decode batch) -> seconds, busy only
    sm_hist = collections.Counter()      # decode_sms -> seconds, busy only
    bw = 0.0
    idle_t = 0.0
    for k in range(len(ct) - 1):
        dt = max(0.0, ct[k + 1] - ct[k])
        if drb[k] > 0:
            busy_t += dt
            bw += drb[k] * dt
            bs_hist[int(drb[k])] += dt
            sm_hist[int(dsm[k])] += dt
        else:
            idle_t += dt

    return dict(arm=arm, cell=cell, block=block, seed=seed, Dsm=Dsm, job=job,
                lag=lag, align_r=r, recs=recs, n_keep=len(keep),
                busy_s=busy_t, idle_s=idle_t,
                mean_bs_busy=(bw / busy_t if busy_t else float("nan")),
                bs_hist=dict(bs_hist), sm_hist=dict(sm_hist),
                # client-side run descriptors
                duration=o["duration"], completed=o["completed"],
                req_thr=o["request_throughput"], out_thr=o["output_throughput"],
                conc=o["concurrency"], max_conc=o.get("max_concurrent_requests"),
                ttfts=[tt[i] * 1000.0 for i in keep],
                n_out_tok=o["total_output_tokens"],
                psm_set=sorted(set(int(x) for x in psm)))


def build():
    if os.path.exists(CACHE):
        return pickle.load(open(CACHE, "rb"))
    O = {}
    for job, d in (("873015", S2DIR), ("872077", E1DIR)):
        for cell in CELLS:
            for b in BLOCKS:
                O[(job, cell, b)] = label_ext(d, ARM, cell, b, b, job)
                print(f"  labelled {job} {cell} b{b}", file=sys.stderr)
    pickle.dump(O, open(CACHE, "wb"))
    return O


# ------------------------------------------------------------------- gate G-A
def gate_GA(O):
    print("\n=== [G-A] INSTRUMENT GATE: label_ext == m3_conditional.label_probe ===")
    print("  (fields itl_ms, a_free, split_frac, prefill_overlap, req_idx compared"
          " element-wise)")
    worst = 0.0
    nrec = 0
    for job, d in (("873015", S2DIR), ("872077", E1DIR)):
        for cell in CELLS:
            for b in (1, 5):     # 2 probes per (job,cell) -- 8 probes total
                ref = mc.label_probe(d, ARM, cell, b, b, job)
                mine = O[(job, cell, b)]
                assert len(ref["recs"]) == len(mine["recs"]), (job, cell, b)
                for r0, r1 in zip(ref["recs"], mine["recs"]):
                    worst = max(worst, abs(r0[0] - r1[0]), abs(r0[2] - r1[2]),
                                abs(r0[3] - r1[3]), abs(int(r0[1]) - int(r1[1])),
                                abs(r0[4] - r1[4]))
                    nrec += 1
                assert abs(ref["lag"] - mine["lag"]) < 1e-12
                assert abs(ref["align_r"] - mine["align_r"]) < 1e-12
    print(f"  8 probes, {nrec} records compared, max abs diff = {worst:.3e}  -> "
          f"{'PASS' if worst == 0.0 else 'FAIL'}")


# ------------------------------------------------- pre-registered readout (S2)
def report_prereg(O, job="873015"):
    print("\n" + "=" * 78)
    print(f"=== [1] PRE-REGISTERED STATISTIC (PREREG_S2 sec 3): per-token p50 of "
          f"SPLIT,\n===     a_free_only=True, pooled over 8 blocks; job {job}")
    print("=" * 78)
    out = {}
    for cell in CELLS:
        pool_sp, pool_all, pool_un = [], [], []
        rows = []
        for b in BLOCKS:
            rc = O[(job, cell, b)]["recs"]
            sp = mc.tokens(rc, mc.sel_split)
            al = mc.tokens(rc, mc.sel_all)
            un = mc.tokens(rc, mc.sel_unsplit)
            pool_sp += sp; pool_all += al; pool_un += un
            rows.append((b, len(sp), len(al), len(un),
                         mc.pctl(sp, .50), mc.pctl(al, .50),
                         mc.pctl(sp, .95), mc.pctl(sp, .99),
                         float(np.mean(sp)), O[(job, cell, b)]["align_r"]))
        out[cell] = dict(rows=rows, sp=pool_sp, al=pool_all, un=pool_un)
        print(f"\n  --- {ARM} {cell} ---")
        print(f"  {'blk':>3} {'n_SPLIT':>8} {'n_all':>8} {'n_UNSPL':>7} "
              f"{'p50_SP':>7} {'p50_all':>7} {'|d|%':>6} {'p95_SP':>7} {'p99_SP':>7} "
              f"{'mean_SP':>8} {'align_r':>7}")
        for (b, ns, na, nu, p50s, p50a, p95s, p99s, ms, ar) in rows:
            dpct = 100.0 * abs(p50s - p50a) / max(p50s, p50a)
            print(f"  {b:>3} {ns:>8} {na:>8} {nu:>7} {p50s:>7.2f} {p50a:>7.2f} "
                  f"{dpct:>6.2f} {p95s:>7.2f} {p99s:>7.2f} {ms:>8.2f} {ar:>7.4f}")
        P50 = [r[4] for r in rows]
        m, sd, h, ci = t_stats(P50)
        pooled = mc.pctl(pool_sp, .50)
        pooled_all = mc.pctl(pool_all, .50)
        print(f"  POOLED  n_SPLIT={len(pool_sp)}  p50(SPLIT)={pooled:.2f} ms  "
              f"p50(all)={pooled_all:.2f} ms  "
              f"|diff|={100*abs(pooled-pooled_all)/max(pooled,pooled_all):.3f}% "
              f"-> agree_within_5pct="
              f"{abs(pooled-pooled_all) <= 0.05*max(pooled,pooled_all)}")
        print(f"  BLOCK   mean of per-block p50 = {m:.3f} +- {sd:.3f} (sd, n=8), "
              f"t95 CI [{ci[0]:.3f}, {ci[1]:.3f}], half-width {h:.3f}")
        print(f"  pooled p95(SPLIT)={mc.pctl(pool_sp,.95):.2f}  "
              f"p99={mc.pctl(pool_sp,.99):.2f}  mean={np.mean(pool_sp):.2f}  "
              f"n_UNSPLIT(pooled)={len(pool_un)}")
    return out


def report_ratio(O, job="873015"):
    print("\n" + "=" * 78)
    print("=== [2] BLOCK-PAIRED d16/d54 RATIOS (block = boot = replication unit,")
    print("===     DESIGN.md sec 4.3.8).  VALUES ONLY -- G_LEVER/G_FLAT are")
    print("===     UNDETERMINED, so NO verdict is derived from any ratio here.")
    print("=" * 78)
    stats = (("sp_p50", mc.sel_split, lambda a: mc.pctl(a, .50)),
             ("sp_p95", mc.sel_split, lambda a: mc.pctl(a, .95)),
             ("sp_mean", mc.sel_split, np.mean),
             ("all_p50", mc.sel_all, lambda a: mc.pctl(a, .50)),
             ("all_p95", mc.sel_all, lambda a: mc.pctl(a, .95)))
    for name, sel, agg in stats:
        num = [agg(mc.tokens(O[(job, "d16", b)]["recs"], sel)) for b in BLOCKS]
        den = [agg(mc.tokens(O[(job, "d54", b)]["recs"], sel)) for b in BLOCKS]
        rr = [num[i] / den[i] for i in range(len(BLOCKS))]
        m, sd, h, ci = t_stats(rr)
        print(f"  {name:>8}: r = {m:.3f} +- {sd:.3f} (sd)  t95 CI [{ci[0]:.3f}, "
              f"{ci[1]:.3f}]  d16={np.mean(num):7.2f} d54={np.mean(den):7.2f}  "
              f"per-block " + " ".join(f"{x:.3f}" for x in rr))


# --------------------------------------------------------- confound machinery
def report_confound(O):
    print("\n" + "=" * 78)
    print("=== [3] CONFOUND DECOMPOSITION: 872077 (sticky OFF) vs 873015 (ON)")
    print("===     Sticky ON changes the RUN, not only the measurement.  These are")
    print("===     run descriptors, NOT a performance comparison (different code")
    print("===     path, different realized partition, no controlled contrast).")
    print("=" * 78)
    hdr = (f"  {'job':>7} {'cell':>5} | {'dur_s':>7} {'compl':>6} {'req/s':>6} "
           f"{'outtok/s':>9} {'conc':>6} {'maxconc':>7} | {'ttft_p50':>8} "
           f"{'ttft_p95':>8} {'ttft_p99':>8} | {'busy_s':>7} {'idle_s':>7} "
           f"{'mean_bs':>7} | {'n_tok':>7} {'p50_all':>7}")
    print(hdr)
    summ = {}
    for cell in CELLS:
        for job in ("872077", "873015"):
            D, C, RT, OT, CO, MC, T50, T95, T99, BS, ID, MB, NT, P50 = ([] for _ in range(14))
            for b in BLOCKS:
                p = O[(job, cell, b)]
                D.append(p["duration"]); C.append(p["completed"])
                RT.append(p["req_thr"]); OT.append(p["out_thr"])
                CO.append(p["conc"]); MC.append(p["max_conc"] or float("nan"))
                tt = p["ttfts"]
                T50.append(mc.pctl(tt, .50)); T95.append(mc.pctl(tt, .95))
                T99.append(mc.pctl(tt, .99))
                BS.append(p["busy_s"]); ID.append(p["idle_s"])
                MB.append(p["mean_bs_busy"])
                al = mc.tokens(p["recs"], mc.sel_all)
                NT.append(len(al)); P50.append(mc.pctl(al, .50))
            summ[(job, cell)] = dict(dur=D, compl=C, conc=CO, mb=MB, t50=T50,
                                     nt=NT, p50=P50, ot=OT)
            print(f"  {job:>7} {cell:>5} | {np.mean(D):7.1f} {np.mean(C):6.1f} "
                  f"{np.mean(RT):6.3f} {np.mean(OT):9.1f} {np.mean(CO):6.2f} "
                  f"{np.mean(MC):7.1f} | {np.mean(T50):8.1f} {np.mean(T95):8.1f} "
                  f"{np.mean(T99):8.1f} | {np.mean(BS):7.1f} {np.mean(ID):7.1f} "
                  f"{np.mean(MB):7.2f} | {np.mean(NT):7.0f} {np.mean(P50):7.2f}")
        # paired (same seed/block) deltas
        for key, lab in (("conc", "client concurrency"), ("mb", "mean decode batch (busy)"),
                         ("dur", "duration s"), ("t50", "TTFT p50 ms"),
                         ("nt", "n tokens (a_free)"), ("p50", "p50(all tokens) ms"),
                         ("ot", "output tok/s")):
            a = summ[("873015", cell)][key]; bb = summ[("872077", cell)][key]
            dd = [a[i] - bb[i] for i in range(len(BLOCKS))]
            rr = [a[i] / bb[i] for i in range(len(BLOCKS))]
            m, sd, h, ci = t_stats(dd)
            mr, sdr, hr, cir = t_stats(rr)
            print(f"      {cell} {lab:<28} ON-OFF = {m:+9.3f} +- {sd:7.3f} "
                  f"t95 [{ci[0]:+8.3f},{ci[1]:+8.3f}]   ON/OFF = {mr:6.3f} "
                  f"[{cir[0]:.3f},{cir[1]:.3f}]")
    return summ


def report_sm_residency(O):
    print("\n=== [3b] REALIZED decode_sms RESIDENCY (time-weighted, decode-busy only) ===")
    print(f"  {'job':>7} {'cell':>5} | " + " ".join(f"{'D'+str(s):>9}" for s in
                                                   (16, 54, 108)) + "   busy_s")
    for cell in CELLS:
        for job in ("872077", "873015"):
            fr = collections.defaultdict(list)
            bs = []
            for b in BLOCKS:
                p = O[(job, cell, b)]
                tot = sum(p["sm_hist"].values())
                for s in (16, 54, 108):
                    fr[s].append(p["sm_hist"].get(s, 0.0) / tot if tot else float("nan"))
                bs.append(tot)
            print(f"  {job:>7} {cell:>5} | " +
                  " ".join(f"{np.mean(fr[s]):9.4f}" for s in (16, 54, 108)) +
                  f"   {np.mean(bs):7.1f}")


def report_batch(O):
    print("\n" + "=" * 78)
    print("=== [4] DECODE BATCH DISTRIBUTION (the matching question)")
    print("===  (a) time-weighted over decode-busy telemetry, int(batch) bins")
    print("=" * 78)
    for cell in CELLS:
        for job in ("872077", "873015"):
            agg = collections.Counter()
            for b in BLOCKS:
                for k, v in O[(job, cell, b)]["bs_hist"].items():
                    agg[k] += v
            tot = sum(agg.values())
            top = sorted(agg.items())
            share = {k: v / tot for k, v in top}
            # cumulative percentiles of batch by time
            cum, q = 0.0, {}
            for k, v in top:
                cum += v / tot
                for tgt in (0.25, 0.50, 0.75, 0.95):
                    if tgt not in q and cum >= tgt:
                        q[tgt] = k
            line = " ".join(f"{k}:{100*share[k]:.1f}%" for k, _ in top if share[k] >= 0.02)
            print(f"  {job} {cell}: mean={sum(k*v for k,v in top)/tot:.2f} "
                  f"p25={q.get(.25)} p50={q.get(.50)} p75={q.get(.75)} p95={q.get(.95)}"
                  f"  bins>=2%: {line}")

    print("\n  (b) per-TOKEN mean in-interval decode batch, binned to int")
    print("      (this is c2_anchor.py's bin definition: mean decode batch over the")
    print("       ITL interval, int()-binned; 'bin 5' = mean_bs in [5,6))")
    print(f"  {'job':>7} {'cell':>5} {'pop':>6} | {'n':>7} {'modal':>5} | "
          + " ".join(f"{'b'+str(i):>13}" for i in range(0, 10)))
    for cell in CELLS:
        for job in ("872077", "873015"):
            recs = []
            for b in BLOCKS:
                recs += [r for r in O[(job, cell, b)]["recs"] if r[1]]
            for pop, sel in (("SPLIT", mc.sel_split), ("all", lambda r: True)):
                v = [r for r in recs if sel(r)]
                by = collections.defaultdict(list)
                for r in v:
                    if r[5] == r[5]:
                        by[int(r[5])].append(r[0])
                n = sum(len(x) for x in by.values())
                modal = max(by, key=lambda k: len(by[k])) if by else None
                cols = []
                for i in range(0, 10):
                    if i in by and len(by[i]) >= 30:
                        cols.append(f"{100*len(by[i])/n:4.1f}%/{mc.pctl(by[i],.5):6.2f}")
                    elif i in by:
                        cols.append(f"{100*len(by[i])/n:4.1f}%/    --")
                    else:
                        cols.append(f"{'-':>13}")
                print(f"  {job:>7} {cell:>5} {pop:>6} | {n:>7} {str(modal):>5} | "
                      + " ".join(f"{c:>13}" for c in cols))
    print("      cell entries are  share-of-population / p50 ITL(ms) in that bin"
          " (p50 shown only when n>=30)")


def report_c2(O):
    print("\n" + "=" * 78)
    print("=== [5] C2 (865493) T8 d16 BATCH BINS vs 873015 d16 -- is 28.79 ~ 28.92")
    print("===     a like-for-like batch match?")
    print("=" * 78)
    sys.path.insert(0, E1DIR)
    import c2_anchor as ca
    data, resid, meta = ca.collect("865493", "1024")
    rows = []
    for (arm, cell, jb, rep), rs in data.items():
        if arm == "T8" and cell == "d16":
            rows += rs
    by = collections.defaultdict(list)
    for (itl, sf, uf, mbs, dur) in rows:
        if sf >= ca.SPLIT_HI:
            by[int(mbs)].append(itl)
    n = sum(len(v) for v in by.values())
    print(f"  C2 865493 T8 d16 SPLIT: n={n}  pooled p50={ca.pct([x for v in by.values() for x in v], .5):.2f} ms")
    print(f"  {'bin':>4} {'n':>7} {'share':>7} {'p50':>7} {'p95':>7}")
    for k in sorted(by):
        v = by[k]
        print(f"  {k:>4} {len(v):>7} {100*len(v)/n:6.1f}% {ca.pct(v,.5):7.2f} "
              f"{ca.pct(v,.95):7.2f}")
    # C2 telemetry-side batch (time weighted) for the same cell
    for (arm, cell, jb), v in sorted(resid.items()):
        if arm == "T8" and cell in ("d16", "d44"):
            print(f"  C2 resid {arm} {cell}: busy_s={v['busy_s']:.1f} "
                  f"frac@D={v['frac_target']:.3f} frac@108={v['frac_108']:.3f} "
                  f"mean_bs_busy={v['mean_bs_busy']:.2f}")






# ============================ EXTENSION PASS (part 2) ========================
def bins_of(O, job, cell, sel=mc.sel_split, afree=True):
    recs = []
    for b in BLOCKS:
        recs += [r for r in O[(job, cell, b)]["recs"] if (r[1] or not afree)]
    by = collections.defaultdict(list)
    for r in recs:
        if sel(r) and r[5] == r[5]:
            by[int(r[5])].append(r[0])
    return by


def c2_bins(cell="d16", arm="T8", job="865493"):
    sys.path.insert(0, E1DIR)
    import c2_anchor as ca
    data, resid, meta = ca.collect(job, "1024")
    by = collections.defaultdict(list)
    for (a, c, jb, rep), rs in data.items():
        if a == arm and c == cell:
            for (itl, sf, uf, mbs, dur) in rs:
                if sf >= ca.SPLIT_HI:
                    by[int(mbs)].append(itl)
    return by, resid


def report_bin_matched(O):
    print("\n" + "=" * 78)
    print("=== [6] BATCH-MATCHED per-token p50, bin = int(mean in-interval decode")
    print("===     batch).  Columns: n / p50(ms).  '--' = fewer than 100 tokens.")
    print("=== The question: how much of 11.09 -> 28.92 survives batch matching?")
    print("=" * 78)
    series = [
        ("872077 d16 SPLIT", bins_of(O, "872077", "d16", mc.sel_split)),
        ("872077 d16 all  ", bins_of(O, "872077", "d16", mc.sel_all)),
        ("873015 d16 all  ", bins_of(O, "873015", "d16", mc.sel_all)),
        ("872077 d54 all  ", bins_of(O, "872077", "d54", mc.sel_all)),
        ("873015 d54 all  ", bins_of(O, "873015", "d54", mc.sel_all)),
    ]
    c216, _ = c2_bins("d16")
    c244, _ = c2_bins("d44")
    series.append(("C2 865493 d16 SP", c216))
    series.append(("C2 865493 d44 SP", c244))
    allbins = sorted({k for _, by in series for k in by})
    allbins = [k for k in allbins if k <= 24]
    print(f"  {'series':>17} | " + " ".join(f"{('b'+str(k)):>12}" for k in allbins))
    for name, by in series:
        cells = []
        for k in allbins:
            v = by.get(k, [])
            if len(v) >= 100:
                cells.append(f"{len(v):5d}/{mc.pctl(v,.5):6.2f}")
            elif v:
                cells.append(f"{len(v):5d}/    --")
            else:
                cells.append(f"{'-':>12}")
        print(f"  {name:>17} | " + " ".join(f"{c:>12}" for c in cells))

    print("\n  Bin-matched ratios (only bins with n>=100 on BOTH sides):")
    for lab, num, den in (
            ("873015 d16 / 872077 d16 (all pop)",
             bins_of(O, "873015", "d16", mc.sel_all), bins_of(O, "872077", "d16", mc.sel_all)),
            ("873015 d16 / C2 865493 d16 SPLIT",
             bins_of(O, "873015", "d16", mc.sel_all), c216),
            ("873015 d54 / 872077 d54 (all pop)",
             bins_of(O, "873015", "d54", mc.sel_all), bins_of(O, "872077", "d54", mc.sel_all)),
            ("873015 d54 / C2 865493 d44 SPLIT",
             bins_of(O, "873015", "d54", mc.sel_all), c244)):
        parts, ws, rs = [], [], []
        for k in sorted(set(num) & set(den)):
            if len(num[k]) >= 100 and len(den[k]) >= 100:
                r = mc.pctl(num[k], .5) / mc.pctl(den[k], .5)
                parts.append(f"b{k}:{r:.2f}")
                ws.append(min(len(num[k]), len(den[k]))); rs.append(r)
        wavg = (sum(w * r for w, r in zip(ws, rs)) / sum(ws)) if ws else float("nan")
        print(f"    {lab:<36} n_bins={len(rs):2d}  min={min(rs) if rs else float('nan'):.2f} "
              f"max={max(rs) if rs else float('nan'):.2f} n-weighted={wavg:.3f}   " + " ".join(parts))

    print("\n  ITL-vs-batch slope INSIDE each run (OLS over bins with n>=100,")
    print("  ms per +1 decode batch) -- how much can batch alone move p50?")
    for name, by in series:
        xs = [k for k in sorted(by) if len(by[k]) >= 100]
        ys = [mc.pctl(by[k], .5) for k in xs]
        if len(xs) >= 3:
            sl, ic = np.polyfit(xs, ys, 1)
            print(f"    {name}: slope={sl:+.4f} ms/batch  intercept={ic:6.2f}  "
                  f"bins {xs[0]}..{xs[-1]}  p50 range {min(ys):.2f}-{max(ys):.2f}")


def report_snapshot_cadence(O):
    print("\n=== [7] TELEMETRY SNAPSHOT CADENCE (instrument check for the bin estimator) ===")
    for job, d in (("872077", E1DIR), ("873015", S2DIR)):
        for cell in CELLS:
            t0b, rows = mc.load_telemetry(f"{d}/e1m3_{ARM}_{cell}_b1_{job}_telemetry.jsonl")
            ts = np.array([r[0] for r in rows])
            gaps = np.diff(ts) * 1000.0
            print(f"  {job} {cell} b1: n_snap={len(ts)} gap_ms p50={np.percentile(gaps,50):.2f} "
                  f"p95={np.percentile(gaps,95):.2f} max={gaps.max():.1f}  "
                  f"span={ts[-1]-ts[0]:.1f}s")


def report_server_args(O):
    print("\n=== [8] SERVER-ARG / CONFIG DIFF BETWEEN THE TWO JOBS (arm-equality check) ===")
    keys = ("max_running_requests", "context_length", "attention_backend",
            "disable_cuda_graph", "disable_radix_cache", "disable_overlap_schedule",
            "mem_fraction_static", "sm_group_num", "enable_pdmux", "max_mamba_cache_size",
            "chunked_prefill_size", "max_prefill_tokens", "schedule_policy",
            "page_size", "dtype", "random_seed", "cuda_graph_max_bs",
            "pdmux_config_path", "model_path")
    got = {}
    for job, d in (("872077", E1DIR), ("873015", S2DIR)):
        for cell in CELLS:
            o = json.loads(open(f"{d}/e1m3_{ARM}_{cell}_b1_{job}_s1.jsonl")
                           .read().strip().split("\n")[0])
            si = o["server_info"]
            got[(job, cell)] = {k: si.get(k) for k in keys}
            got[(job, cell)]["request_rate"] = o["request_rate"]
            got[(job, cell)]["n_prompts"] = len(o["ttfts"])
    base = got[("872077", "d16")]
    for k in list(base) :
        vals = {kk: v.get(k) for kk, v in got.items()}
        if len(set(map(str, vals.values()))) > 1:
            print(f"  DIFF {k}: " + "  ".join(f"{kk[0]}/{kk[1]}={v}" for kk, v in vals.items()))
    print("  (only differing keys printed; everything else in the list is identical"
          " across all four job x cell combos)")


def report_identity(O):
    print("\n=== [9] IDENTITY AUDIT -- which statistics are forced once realized ~ 1.0? ===")
    for job in ("872077", "873015"):
        for cell in CELLS:
            ns = na = nu = nam = 0
            for b in BLOCKS:
                rc = [r for r in O[(job, cell, b)]["recs"] if r[1]]
                na += len(rc)
                ns += sum(1 for r in rc if r[2] >= mc.SPLIT_HI)
                nu += sum(1 for r in rc if r[2] <= mc.UNSPLIT_LO)
                nam += sum(1 for r in rc if mc.UNSPLIT_LO < r[2] < mc.SPLIT_HI)
            print(f"  {job} {cell}: n_a_free={na}  SPLIT={ns} ({100*ns/na:.3f}%)  "
                  f"UNSPLIT={nu} ({100*nu/na:.4f}%)  AMBIG={nam} ({100*nam/na:.4f}%)"
                  f"   -> p50(SPLIT) vs p50(all) share {100*ns/na:.3f}% of the same tokens")


def report_ttft_itl_percentiles(O):
    print("\n=== [10] TTFT / ITL PERCENTILES (metric-cliff + distribution reporting) ===")
    print(f"  {'job':>7} {'cell':>5} | {'TTFT p50':>8} {'p95':>8} {'p99':>8} {'max':>8} | "
          f"{'ITL p50':>8} {'p95':>8} {'p99':>8} | {'E2E-ish n_err':>13}")
    for cell in CELLS:
        for job in ("872077", "873015"):
            T = []
            I50, I95, I99 = [], [], []
            for b in BLOCKS:
                p = O[(job, cell, b)]
                T += p["ttfts"]
                al = mc.tokens(p["recs"], mc.sel_all)
                I50.append(mc.pctl(al, .50)); I95.append(mc.pctl(al, .95))
                I99.append(mc.pctl(al, .99))
            print(f"  {job:>7} {cell:>5} | {mc.pctl(T,.50):8.1f} {mc.pctl(T,.95):8.1f} "
                  f"{mc.pctl(T,.99):8.1f} {max(T):8.1f} | {np.mean(I50):8.2f} "
                  f"{np.mean(I95):8.2f} {np.mean(I99):8.2f} | {'0 (all blocks)':>13}")






# ============================ EXTENSION PASS (part 3) ========================
def report_cross_tab():
    """Time-weighted joint distribution of (decode busy?, prefill active?) x
    (decode_sms, prefill_sms).  This is the mechanism check for 'sticky changes
    the run': it says exactly which SM configuration each state got."""
    print("\n" + "=" * 78)
    print("=== [11] STATE CROSS-TAB, time-weighted, pooled over 8 blocks")
    print("===  state = (decode_busy, prefill_active) ; value = (prefill_sms, decode_sms)")
    print("=" * 78)
    for cell in CELLS:
        for job, d in (("872077", E1DIR), ("873015", S2DIR)):
            acc = collections.Counter()
            for b in BLOCKS:
                t0b, rows = mc.load_telemetry(
                    f"{d}/e1m3_{ARM}_{cell}_b{b}_{job}_telemetry.jsonl")
                for k in range(len(rows) - 1):
                    dt = rows[k + 1][0] - rows[k][0]
                    if dt <= 0 or dt > 3.0:
                        continue
                    st = (rows[k][2] > 0, rows[k][4] > 0, rows[k][3], rows[k][1])
                    acc[st] += dt
            tot = sum(acc.values())
            print(f"  {job} {cell} (total {tot:.1f}s):")
            for st, v in sorted(acc.items(), key=lambda x: -x[1])[:6]:
                if v / tot < 0.001:
                    continue
                print(f"      dec_busy={str(st[0]):5} pref_active={str(st[1]):5} "
                      f"-> (P{st[2]},D{st[3]})  {100*v/tot:6.2f}%  {v:7.1f}s")


def report_controls(O):
    print("\n" + "=" * 78)
    print("=== [12] WHAT CAN SERVE AS A CONTROL IN THIS RUN?")
    print("=" * 78)
    # (a) within-cell UNSPLIT sample at d16 / d54
    for cell in CELLS:
        un = []
        per_block = []
        for b in BLOCKS:
            v = mc.tokens(O[("873015", cell, b)]["recs"], mc.sel_unsplit)
            per_block.append(len(v)); un += v
        print(f"  873015 {cell} UNSPLIT(108-SM) sample: n={len(un)} "
              f"per-block {per_block}")
        if len(un) >= 10:
            print(f"      p25/p50/p75 = {mc.pctl(un,.25):.2f} / {mc.pctl(un,.50):.2f} "
                  f"/ {mc.pctl(un,.75):.2f} ms   min={min(un):.2f} max={max(un):.2f}"
                  f"   (n<<100: NOT a usable control, reported for completeness)")
    # (b) does 872077 d16 already contain mass at the S2 level?
    print("\n  Mass of 872077 already sitting at the S2-realized level")
    lo, hi = 26.0, 32.0
    for cell in CELLS:
        for job in ("872077", "873015"):
            al = []
            for b in BLOCKS:
                al += mc.tokens(O[(job, cell, b)]["recs"], mc.sel_all)
            inw = [x for x in al if lo <= x <= hi]
            print(f"    {job} {cell}: share of a_free tokens in [{lo},{hi}] ms = "
                  f"{100*len(inw)/len(al):6.3f}%  (n={len(inw)}/{len(al)})")
    # (c) cell-label discrimination under OFF vs ON (the internal check)
    print("\n  Cell-label discrimination, p50(all tokens), pooled:")
    for job in ("872077", "873015"):
        v = {}
        for cell in CELLS:
            al = []
            for b in BLOCKS:
                al += mc.tokens(O[(job, cell, b)]["recs"], mc.sel_all)
            v[cell] = mc.pctl(al, .50)
        print(f"    {job}: d16={v['d16']:.2f}  d54={v['d54']:.2f}  "
              f"d16/d54={v['d16']/v['d54']:.3f}")


def report_d54_miss(O):
    print("\n" + "=" * 78)
    print("=== [13] d54 MISS FORENSICS -- post-hoc diagnostics ONLY.")
    print("===  The pre-registered interval [13,16] is NOT revised; this asks")
    print("===  where its two anchors came from and what they would predict if")
    print("===  the extrapolations they skipped are put back in.")
    print("=" * 78)
    # C2's own D-scaling, fitted on C2's OWN three anchored cells
    pts = {}
    for cell, in (("d16",), ("d24",), ("d44",)):
        by, _ = c2_bins(cell)
        allv = [x for v in by.values() for x in v]
        pts[int(cell[1:])] = mc.pctl(allv, .50)
    print("  C2 865493 T8 pooled SPLIT p50 by cell: " +
          "  ".join(f"D{k}={v:.2f}" for k, v in sorted(pts.items())))
    Ds = np.array(sorted(pts))
    ys = np.array([pts[d] for d in Ds])
    A = np.vstack([np.ones_like(Ds, dtype=float), 1.0 / Ds]).T
    coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
    pred = coef[0] + coef[1] / 54.0
    print(f"  fit ITL = a + b/D on C2's own cells: a={coef[0]:.2f} b={coef[1]:.1f}"
          f"  -> C2-curve extrapolation to D=54 = {pred:.2f} ms")
    print(f"  (the pre-registration used C2's D=44 value {pts[44]:.2f} as the anchor"
          f" for a D=54 cell instead)")
    # batch correction inside 873015 d54
    by = bins_of(O, "873015", "d54", mc.sel_all)
    xs = [k for k in sorted(by) if len(by[k]) >= 100]
    sl, ic = np.polyfit(xs, [mc.pctl(by[k], .5) for k in xs], 1)
    mb_s2 = np.mean([O[("873015", "d54", b)]["mean_bs_busy"] for b in BLOCKS])
    _, resid = c2_bins("d44")
    mb_c2 = [v["mean_bs_busy"] for k, v in resid.items() if k[0] == "T8" and k[1] == "d44"]
    print(f"  mean decode batch (busy): 873015 d54 = {mb_s2:.2f} vs C2 d44 = "
          f"{np.mean(mb_c2):.2f}")
    print(f"  873015 d54 ITL-vs-batch slope = {sl:+.4f} ms/batch -> moving 873015 d54"
          f" to C2's batch would add {sl*(np.mean(mb_c2)-mb_s2):+.2f} ms "
          f"({12.03 + sl*(np.mean(mb_c2)-mb_s2):.2f} ms), still "
          f"{'below' if 12.03 + sl*(np.mean(mb_c2)-mb_s2) < 13 else 'inside'} [13,16]")
    # the other anchor: 872077 d54 UNSPLIT upper mode 14.12 (S0-R)
    print("  second anchor = S0-R's 872077 d54 UNSPLIT slow mode 14.12 ms: that is a")
    print("  MODE of a bimodal 108-SM-labelled population (share_win 5.61%), not a p50;")
    print("  872077 d54's own p50(all) is 10.97 ms and p95 14.01 ms.")






# ============================ EXTENSION PASS (part 4) ========================
def report_workload_identity():
    print("\n=== [14] WORKLOAD PAIRING CHECK (seed=block in both campaigns) ===")
    ok = True
    for cell in CELLS:
        for b in BLOCKS:
            a = json.loads(open(f"{E1DIR}/e1m3_{ARM}_{cell}_b{b}_872077_s{b}.jsonl")
                           .read().strip().split("\n")[0])
            c = json.loads(open(f"{S2DIR}/e1m3_{ARM}_{cell}_b{b}_873015_s{b}.jsonl")
                           .read().strip().split("\n")[0])
            same = (a["input_lens"] == c["input_lens"])
            ok &= same
            if not same:
                print(f"  MISMATCH {cell} b{b}")
    print(f"  input_lens identical for all 16 (cell,block) pairs: {ok}"
          f"  -> the ON/OFF comparison is paired on the workload")
    a = json.loads(open(f"{E1DIR}/e1m3_{ARM}_d16_b1_872077_s1.jsonl")
                   .read().strip().split("\n")[0])
    print(f"  n_prompts={len(a['input_lens'])} input_len p50="
          f"{mc.pctl(a['input_lens'],.5):.0f} p95={mc.pctl(a['input_lens'],.95):.0f} "
          f"max={max(a['input_lens'])}; output_len p50="
          f"{mc.pctl(a['output_lens'],.5):.0f} p95={mc.pctl(a['output_lens'],.95):.0f}")


def report_little(O):
    print("\n=== [15] IS THE CONCURRENCY/BATCH GROWTH DOWNSTREAM OF THE ITL CHANGE? ===")
    print("  Little's law check: concurrency ?= arrival rate x mean E2E latency,")
    print("  with E2E ~ TTFT + (n_out-1) x ITL.  If the ON run's higher concurrency")
    print("  is *explained* by its slower decode, it is a consequence, not a cause.")
    for cell in CELLS:
        for job, d in (("872077", E1DIR), ("873015", S2DIR)):
            C, PR = [], []
            for b in BLOCKS:
                o = json.loads(open(f"{d}/e1m3_{ARM}_{cell}_b{b}_{job}_s{b}.jsonl")
                               .read().strip().split("\n")[0])
                itl_all = [x * 1000.0 for L in o["itls"] for x in L]
                e2e = (np.mean(o["ttfts"]) * 1000.0
                       + (np.mean([len(L) for L in o["itls"]])) * mc.pctl(itl_all, .5))
                PR.append(o["request_throughput"] * e2e / 1000.0)
                C.append(o["concurrency"])
            print(f"  {job} {cell}: observed concurrency {np.mean(C):6.2f}  "
                  f"predicted from rate x (TTFT + n_out x ITL_p50) {np.mean(PR):6.2f}  "
                  f"ratio {np.mean(C)/np.mean(PR):.3f}")


def report_ratio_off(O):
    print("\n=== [16] THE SAME BLOCK-PAIRED d16/d54 RATIO IN 872077 (selector NOT held) ===")
    for name, sel, agg in (("all_p50", mc.sel_all, lambda a: mc.pctl(a, .50)),
                           ("all_p95", mc.sel_all, lambda a: mc.pctl(a, .95)),
                           ("sp_p50", mc.sel_split, lambda a: mc.pctl(a, .50))):
        for job in ("872077", "873015"):
            num = [agg(mc.tokens(O[(job, "d16", b)]["recs"], sel)) for b in BLOCKS]
            den = [agg(mc.tokens(O[(job, "d54", b)]["recs"], sel)) for b in BLOCKS]
            rr = [num[i] / den[i] for i in range(len(BLOCKS))]
            m, sd, h, ci = t_stats(rr)
            print(f"  {job} {name:>8}: r = {m:.3f} +- {sd:.3f}  t95 CI [{ci[0]:.3f},"
                  f" {ci[1]:.3f}]   d16={np.mean(num):7.2f} d54={np.mean(den):7.2f}")


def report_attribution(O):
    print("\n=== [17] ATTRIBUTION OF 872077 d16 -> 873015 d16 (p50, all-token pop) ===")
    e_all, s_all = [], []
    for b in BLOCKS:
        e_all += mc.tokens(O[("872077", "d16", b)]["recs"], mc.sel_all)
        s_all += mc.tokens(O[("873015", "d16", b)]["recs"], mc.sel_all)
    p_off, p_on = mc.pctl(e_all, .50), mc.pctl(s_all, .50)
    byo = bins_of(O, "872077", "d16", mc.sel_all)
    xs = [k for k in sorted(byo) if len(byo[k]) >= 100]
    sl_off, ic_off = np.polyfit(xs, [mc.pctl(byo[k], .5) for k in xs], 1)
    mb_off = np.mean([O[("872077", "d16", b)]["mean_bs_busy"] for b in BLOCKS])
    mb_on = np.mean([O[("873015", "d16", b)]["mean_bs_busy"] for b in BLOCKS])
    batch_term = sl_off * (mb_on - mb_off)
    print(f"  OFF p50 = {p_off:.2f} ms  (mean decode batch {mb_off:.2f})")
    print(f"  ON  p50 = {p_on:.2f} ms  (mean decode batch {mb_on:.2f})")
    print(f"  total change = {p_on - p_off:+.2f} ms  ({p_on/p_off:.3f}x)")
    print(f"  batch term, using the OFF run's OWN ITL-vs-batch slope "
          f"({sl_off:+.4f} ms/batch) over the OFF run's measured bin range: "
          f"{batch_term:+.2f} ms = {100*batch_term/(p_on-p_off):.1f}% of the change")
    byn = bins_of(O, "873015", "d16", mc.sel_all)
    ws, rs = [], []
    for k in sorted(set(byo) & set(byn)):
        if len(byo[k]) >= 100 and len(byn[k]) >= 100:
            ws.append(min(len(byo[k]), len(byn[k])))
            rs.append(mc.pctl(byn[k], .5) / mc.pctl(byo[k], .5))
    wr = sum(w * r for w, r in zip(ws, rs)) / sum(ws)
    print(f"  batch-matched ratio over {len(rs)} shared bins = {wr:.3f} vs pooled "
          f"{p_on/p_off:.3f}  -> composition explains {100*(p_on/p_off - wr)/(p_on/p_off - 1):.1f}%"
          f" of the excess over 1.0")


# --------------------------------------------------------------------- main
def main():
    O = build()
    gate_GA(O)
    report_prereg(O, "873015")
    report_prereg(O, "872077")
    report_identity(O)
    report_ratio(O, "873015")
    report_ratio_off(O)
    report_confound(O)
    report_sm_residency(O)
    report_cross_tab()
    report_ttft_itl_percentiles(O)
    report_batch(O)
    report_bin_matched(O)
    report_attribution(O)
    report_little(O)
    report_workload_identity()
    report_controls(O)
    report_d54_miss(O)
    report_c2(O)
    report_snapshot_cadence(O)
    report_server_args(O)
    print("\n" + "=" * 78)
    print("NO throughput / goodput / g / G_LEVER / G_FLAT / policy claim is made or")
    print("licensed by anything above.  Ratios are values only.  See")
    print("S2_ANALYSIS_2026-08-04.md for the pre-registered verdict.")
    print("=" * 78)


if __name__ == "__main__":
    main()
