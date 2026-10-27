#!/usr/bin/env python3
"""Lockstep-stall share diagnostic (result-analyst, 2026-10-27, CPU only).

Descriptive values only -- no performance verdict.  Reads PDMUX_PHASE_EVENTS
records (prefill_span_start / prefill_span_end / decode_iteration) from archived
telemetry and decomposes the server-side decode token interval by what the
event loop was doing in that interval.

Event-loop facts used (src/multiplex/multiplexing_mixin.py, event_loop_pdmux):
  loop k:  recv -> update_split_prefill_batch [span_start emitted here]
           -> adjust/drain (prefill_stream.sync + decode_stream.sync)
           -> decode run_batch -> t_launch_k (host)
           -> prefill chunk run_batch (host launch of forward_count layers)
           -> decode_stream.synchronize() -> t_sync_k (host)
           -> process decode result -> [if last chunk done: span_end emitted,
              merge] -> loop k+1
  Token interval of iteration k (server side) := I_k = t_sync_k - t_sync_{k-1}
  (consecutive iter_seq only).  Pre-launch part P_k = t_launch_k - t_sync_{k-1}
  (merge of k-1, recv, admission, drain, decode host launch); host part
  H_k = t_sync_k - t_launch_k (chunk host launch + residual decode GPU).
Classes of iteration k:
  SOLO  prefill_in_flight False
  CHUNK prefill_in_flight True  and prefill_kernel_done_wait False (a chunk is
        issued in this loop)
  WAIT  prefill_in_flight True  and prefill_kernel_done_wait True (all chunks
        issued, waiting for the GPU to finish the last one)
  flags: ADMIT (a span_start in (t_sync_{k-1}, t_launch_k]),
         MERGE (a span_end   in (t_sync_{k-1}, t_launch_k]).
All times are host CLOCK_MONOTONIC (perf_counter / time.monotonic): there is
no GPU timestamp in these events.

Usage:
  python3 workspace/engine-port/results/lockstep_diag/lockstep_diag.py \
      --out workspace/engine-port/results/lockstep_diag/lockstep_diag_values.json --with-profile
  (imports pdmux_eval.analyze.percentile from workspace/engine-port/benchmarks)
"""
from __future__ import annotations

import argparse
import bisect
import glob
import json
import math
import os
import random
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # workspace/engine-port
sys.path.insert(0, str(ROOT / "benchmarks"))
from pdmux_eval.analyze import percentile  # noqa: E402  (canonical percentile)

RES = ROOT / "results"
BUDGET = 65536  # split_forward_token_budget (upstream default, all these runs)
N_LAYERS = 56  # Nemotron-Nano-9B-v2; confirmed by 7 raw CHUNK iterations per 8192-token span (fwd 8)


# ----------------------------------------------------------------- loading
def load_tel(path):
    iters, starts, ends = [], {}, {}
    dropped = 0
    with open(path) as fh:
        for line in fh:
            if '"decode_iteration"' not in line and '"prefill_span_' not in line:
                if '"dropped_events": 0' not in line and '"dropped_events"' in line:
                    try:
                        dropped = max(dropped, int(json.loads(line).get("dropped_events", 0)))
                    except Exception:
                        pass
                continue
            d = json.loads(line)
            dropped = max(dropped, int(d.get("dropped_events", 0) or 0))
            e = d["event"]
            if e == "decode_iteration":
                iters.append(d)
            elif e == "prefill_span_start":
                starts.setdefault(d["span_seq"], d)
            elif e == "prefill_span_end":
                ends.setdefault(d["span_seq"], d)
    iters.sort(key=lambda x: x["iter_seq"])
    spans = []
    for s, st in sorted(starts.items()):
        en = ends.get(s)
        if en is None:
            continue
        spans.append(dict(seq=s, t0=st["timestamp_monotonic_s"], t1=en["timestamp_monotonic_s"],
                          extend=int(st["extend_tokens"]), n_new=int(st["n_new"]),
                          n_done=int(en.get("n_done", 0)), running_bs=int(st["running_bs"]),
                          queue_len=int(st["queue_len"])))
    return iters, spans, dropped


def read_marks(path):
    out = {}
    if not os.path.exists(path):
        return out
    for line in open(path):
        p = line.split()
        if len(p) >= 2:
            try:
                out.setdefault(p[0], []).append((p[1:-1], float(p[-1])))
            except ValueError:
                pass
    return out


# ----------------------------------------------------------------- core
def build_intervals(iters, spans, lo, hi):
    """Return per-iteration records for consecutive iter_seq inside [lo, hi]."""
    st_ts = sorted(s["t0"] for s in spans)
    en_ts = sorted(s["t1"] for s in spans)
    recs = []
    for a, b in zip(iters, iters[1:]):
        if b["iter_seq"] != a["iter_seq"] + 1:
            continue
        if not (lo <= a["t_sync"] and b["t_sync"] <= hi):
            continue
        I = b["t_sync"] - a["t_sync"]
        P = b["t_launch"] - a["t_sync"]
        H = b["t_sync"] - b["t_launch"]
        if b["prefill_in_flight"]:
            cls = "WAIT" if b["prefill_kernel_done_wait"] else "CHUNK"
        else:
            cls = "SOLO"
        lo_w, hi_w = a["t_sync"], b["t_launch"]
        admit = bisect.bisect_right(st_ts, hi_w) - bisect.bisect_right(st_ts, lo_w) > 0
        merge = bisect.bisect_right(en_ts, hi_w) - bisect.bisect_right(en_ts, lo_w) > 0
        # requests present in BOTH iterations (token interval exists only for them):
        # bs_b minus the requests merged in this interval (n_done of spans ending here)
        merged = sum(s["n_done"] for s in spans if lo_w < s["t1"] <= hi_w)
        carried = max(0, int(b["bs"]) - merged)
        recs.append(dict(I=I * 1e3, P=P * 1e3, H=H * 1e3, cls=cls, admit=admit, merge=merge,
                         carried=carried, merged=merged,
                         bs=int(b["bs"]), dsm=int(b["decode_sms"]), psm=int(b["prefill_sms"]),
                         t_launch=b["t_launch"], t_sync=b["t_sync"], queue_len=int(b["queue_len"])))
    return recs


def label(r):
    if r["merge"]:
        return "MERGE"          # interval holds the end-of-span merge (+drain of next loop)
    if r["admit"]:
        return "ADMIT"          # interval holds an admission (+partition switch drain)
    return r["cls"]


def tw_quantiles(recs, qs=(0.5, 0.95, 0.99)):
    """Token-weighted quantiles of I; weight = requests present in both iterations."""
    vals = []
    for r in recs:
        vals.extend([r["I"]] * r["carried"])
    return {f"p{int(q*100)}": percentile(vals, q) for q in qs}, vals


def tail_share(recs, q):
    vals = []
    for r in recs:
        vals.extend([(r["I"], label(r))] * r["carried"])
    if not vals:
        return None
    thr = percentile([v for v, _ in vals], q)
    tail = [lab for v, lab in vals if v >= thr]
    out = {"threshold_ms": thr, "n_tail_tokens": len(tail)}
    for lab in ("SOLO", "CHUNK", "WAIT", "ADMIT", "MERGE"):
        out[lab] = sum(1 for x in tail if x == lab) / len(tail)
    out["nonSOLO"] = 1.0 - out["SOLO"]
    return out


def span_table(spans, recs, lo, hi):
    rows = []
    for s in spans:
        if not (lo <= s["t0"] and s["t1"] <= hi):
            continue
        inside = [r for r in recs if s["t0"] < r["t_launch"] <= s["t1"]]
        # pure classes only (no admission/merge in the interval) and a carried request
        ch = [r for r in inside if label(r) == "CHUNK" and r["carried"] > 0]
        wt = [r for r in inside if label(r) == "WAIT" and r["carried"] > 0]
        n_chunk_raw = sum(1 for r in inside if r["cls"] == "CHUNK")
        fwd = max(1, BUDGET // s["extend"]) if s["extend"] > 0 else None
        last_chunk = max((r["t_sync"] for r in ch), default=None)
        rows.append(dict(seq=s["seq"], extend=s["extend"], n_new=s["n_new"], fwd_layers=fwd,
                         W_ms=(s["t1"] - s["t0"]) * 1e3, n_chunk_it=len(ch), n_chunk_raw=n_chunk_raw,
                         n_chunks_expected=(-(-N_LAYERS // fwd)) if fwd else None, n_wait_it=len(wt),
                         sumI_chunk=sum(r["I"] for r in ch), sumI_wait=sum(r["I"] for r in wt),
                         medI_chunk=statistics.median([r["I"] for r in ch]) if ch else None,
                         medH_chunk=statistics.median([r["H"] for r in ch]) if ch else None,
                         tail_after_last_chunk_ms=((s["t1"] - last_chunk) * 1e3) if last_chunk else None,
                         psm=statistics.median([r["psm"] for r in ch]) if ch else None,
                         dsm=statistics.median([r["dsm"] for r in ch]) if ch else None,
                         bs=statistics.median([r["bs"] for r in ch]) if ch else None))
    return rows


def ols(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return math.nan, math.nan
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return b, my - b * mx


def slope_ci(xs, ys, samples=10000, seed=1):
    b, a = ols(xs, ys)
    rng = random.Random(seed)
    n = len(xs)
    bs = []
    for _ in range(samples):
        idx = [rng.randrange(n) for _ in range(n)]
        bx, by = [xs[i] for i in idx], [ys[i] for i in idx]
        bb, _ = ols(bx, by)
        if not math.isnan(bb):
            bs.append(bb)
    bs.sort()
    return dict(slope=b, intercept=a, n=n, ci95_low=percentile(bs, 0.025),
                ci95_high=percentile(bs, 0.975))


# ----------------------------------------------------------------- per-dataset drivers
def analyse_cell(tel, lo, hi, meta, bench=None):
    iters, spans, dropped = load_tel(tel)
    recs = build_intervals(iters, spans, lo, hi)
    if not recs:
        return None
    q, _ = tw_quantiles(recs)
    by = {}
    for lab in ("SOLO", "CHUNK", "WAIT", "ADMIT", "MERGE"):
        rr = [r for r in recs if label(r) == lab]
        if rr:
            v = [r["I"] for r in rr]
            rc = [r for r in rr if r["carried"] > 0]
            by[lab + "_carried"] = dict(n_iter=len(rc), n_tok=sum(r["carried"] for r in rc),
                                        p50=percentile([r["I"] for r in rc], 0.5) if rc else None,
                                        max=max((r["I"] for r in rc), default=None))
            by[lab] = dict(n_iter=len(rr), n_tok=sum(r["carried"] for r in rr),
                           p50=percentile(v, 0.5), p95=percentile(v, 0.95), max=max(v),
                           medP=percentile([r["P"] for r in rr], 0.5),
                           medH=percentile([r["H"] for r in rr], 0.5))
    out = dict(meta, dropped_events=dropped, n_iter=len(recs), tw_I=q, by_class=by,
               tail_p95=tail_share(recs, 0.95), tail_p99=tail_share(recs, 0.99),
               spans=span_table(spans, recs, lo, hi))
    solo_med = by.get("SOLO", {}).get("p50")
    if solo_med:
        big = [r for r in recs if r["I"] > 3 * solo_med and r["carried"] > 0]
        out["stall_gt3x_solo"] = dict(n=len(big), labels={lab: sum(1 for r in big if label(r) == lab)
                                                         for lab in ("SOLO", "CHUNK", "WAIT", "ADMIT", "MERGE")},
                                      max_ms=max((r["I"] for r in big), default=None))
    if bench is not None:
        out["client"] = client_compare(bench, recs, spans, lo, hi)
    return out


def client_compare(bench_path, recs, spans, lo, hi):
    d = None
    for line in open(bench_path):
        d = json.loads(line)
    if d is None:
        return None
    itls = [x * 1e3 for req in d["itls"] for x in req]
    ttfts = [x * 1e3 for x in d["ttfts"]]
    q, _ = tw_quantiles(recs)
    res = dict(n_req=d["completed"], client_itl=dict(p50=percentile(itls, .5), p95=percentile(itls, .95),
                                                     p99=percentile(itls, .99)),
               server_tw_I=q,
               ttft=dict(p50=percentile(ttfts, .5), p95=percentile(ttfts, .95), p99=percentile(ttfts, .99),
                         mean=statistics.fmean(ttfts)))
    # TTFT decomposition by order matching (valid only if every span admitted one request)
    sp = [s for s in spans if lo <= s["t0"] and s["t1"] <= hi]
    if sp and all(s["n_new"] == 1 for s in sp) and len(sp) == len(ttfts):
        W = [(s["t1"] - s["t0"]) * 1e3 for s in sp]
        resid = [t - w for t, w in zip(ttfts, W)]
        res["ttft_decomp"] = dict(matched=len(sp), mean_ttft=statistics.fmean(ttfts),
                                  mean_span_W=statistics.fmean(W), mean_resid=statistics.fmean(resid),
                                  min_resid=min(resid), share_W=statistics.fmean(W) / statistics.fmean(ttfts))
    else:
        res["ttft_decomp"] = dict(matched=0, reason="span/request count mismatch or n_new>1",
                                  n_spans=len(sp), n_req=len(ttfts),
                                  n_new_hist={k: sum(1 for s in sp if s["n_new"] == k) for k in {s["n_new"] for s in sp}})
        if sp:
            W = [(s["t1"] - s["t0"]) * 1e3 for s in sp]
            res["ttft_decomp"]["mean_span_W"] = statistics.fmean(W)
            res["ttft_decomp"]["mean_ttft"] = statistics.fmean(ttfts)
    return res


def n3_cells():
    out = []
    for job in sorted(glob.glob(str(RES / "n3_capacity" / "n3b_*"))):
        for tel in sorted(glob.glob(job + "/tel_*.jsonl")):
            cell = Path(tel).stem[4:]
            marks = read_marks(f"{job}/marks_{cell}.txt")
            bench = f"{job}/bench_{cell}.jsonl"
            if "bench_start" not in marks:
                continue
            lo, hi = marks["bench_start"][0][1], marks["bench_end"][0][1]
            out.append((tel, lo, hi, dict(dataset="N3", job=Path(job).name, cell=cell),
                        bench if os.path.exists(bench) else None))
    return out


def n2_cells(jobglob, dataset):
    out = []
    for job in sorted(glob.glob(str(RES / "n2_static_grid" / jobglob))):
        for tel in sorted(glob.glob(job + "/tel_*.jsonl")):
            boot = Path(tel).stem[4:]
            m = read_marks(f"{job}/marks_{boot}.txt")
            for (args, t0), (_, t1) in zip(m.get("cell_start", []), m.get("cell_end", [])):
                cid = args[0]
                co = [t for a, t in m.get("coload_sent", []) if a and a[0] == cid]
                out.append((tel, t0, t1, dict(dataset=dataset, job=Path(job).name, boot=boot, cell=cid,
                                              coload_sent=co[0] if co else None), None))
    return out


# ----------------------------------------------------------------- N2-EA client-aligned


def n2ea_client(job, boot):
    """Client per-token absolute times (CLOCK_MONOTONIC) aligned with server spans."""
    tel = f"{job}/tel_{boot}.jsonl"
    cl = f"{job}/client_{boot}.jsonl"
    if not (os.path.exists(tel) and os.path.exists(cl)):
        return []
    iters, spans, dropped = load_tel(tel)
    out = []
    for line in open(cl):
        c = json.loads(line)
        tco = c.get("t_coload")
        co_sp = sorted([s for s in spans if tco and s["t0"] >= tco and s["extend"] == s["n_new"] * 8192
                        and s["t1"] <= c["t_end"]], key=lambda s: s["t1"])
        if not co_sp:
            continue
        w0, w1 = co_sp[0]["t0"], co_sp[-1]["t1"]
        st = sorted(s["t0"] for s in spans)
        en = sorted(s["t1"] for s in spans)
        ints = []  # (ms, label)
        for rq in c["requests"]:
            ts = rq["times"]
            for x, y in zip(ts, ts[1:]):
                if y <= x:
                    continue
                inside = (w0 <= x and y <= w1)
                has_end = bisect.bisect_left(en, y) - bisect.bisect_right(en, x) > 0
                has_start = bisect.bisect_left(st, y) - bisect.bisect_right(st, x) > 0
                lab = ("CO_MERGE" if has_end else "CO_ADMIT" if has_start else "CO") if inside else \
                      ("ALONE_EDGE" if (has_end or has_start) else "ALONE")
                ints.append(((y - x) * 1e3, lab, x, y))
        if not ints:
            continue
        allv = [v for v, *_ in ints]
        res = dict(dataset="N2EA_client", job=Path(job).name, boot=boot, cell=c["cell_id"], d=c["d"], b=c["b"],
                   c=c["c"], K=c["K"], n_co_spans=len(co_sp), co_window_s=w1 - w0, dropped_events=dropped,
                   all=dict(n=len(allv), p50=percentile(allv, .5), p95=percentile(allv, .95),
                            p99=percentile(allv, .99), max=max(allv)))
        for lab in ("ALONE", "ALONE_EDGE", "CO", "CO_ADMIT", "CO_MERGE"):
            v = [x for x, l, *_ in ints if l == lab]
            if v:
                res[lab] = dict(n=len(v), p50=percentile(v, .5), p95=percentile(v, .95),
                                p99=percentile(v, .99), max=max(v))
        co_all = [x for x, l, *_ in ints if l.startswith("CO")]
        if co_all:
            res["CO_all"] = dict(n=len(co_all), p50=percentile(co_all, .5), p95=percentile(co_all, .95),
                                 p99=percentile(co_all, .99), max=max(co_all))
        for q in (0.95, 0.99):
            thr = percentile(allv, q)
            tail = [l for v, l, *_ in ints if v >= thr]
            res[f"tail_p{int(q*100)}"] = {l: tail.count(l) / len(tail) for l in set(tail)}
        # per co span: decode tokens per decode request emitted inside the span vs span length
        per_span = []
        for s in co_sp:
            toks = [sum(1 for t in rq["times"] if s["t0"] < t <= s["t1"]) for rq in c["requests"]]
            fwd = max(1, BUDGET // s["extend"])
            nch = -(-N_LAYERS // fwd)
            vin = [v for v, l, x, y in ints if s["t0"] <= x and y <= s["t1"]]
            per_span.append(dict(W_ms=(s["t1"] - s["t0"]) * 1e3, extend=s["extend"], n_chunks=nch,
                                 tok_per_req=statistics.median(toks) if toks else None,
                                 med_int_ms=statistics.median(vin) if vin else None))
        res["co_spans"] = per_span
        # TTFT decomposition of the co-load requests (all sent at t_coload)
        recv = sorted(r["times"][0] for r in c["co_requests"] if r["times"])
        assign = []
        for s in co_sp:
            assign += [s] * s["n_done"]
        if len(assign) == len(recv):
            q_ = [(s["t0"] - tco) * 1e3 for s in assign]
            p_ = [(s["t1"] - s["t0"]) * 1e3 for s in assign]
            o_ = [(r - s["t1"]) * 1e3 for r, s in zip(recv, assign)]
            tt = [(r - tco) * 1e3 for r in recv]
            res["co_ttft"] = dict(n=len(tt), mean_ttft=statistics.fmean(tt), p50_ttft=percentile(tt, .5),
                                  p99_ttft=percentile(tt, .99),
                                  mean_queue=statistics.fmean(q_), mean_prefill_W=statistics.fmean(p_),
                                  mean_post=statistics.fmean(o_), min_post=min(o_), max_post=max(o_),
                                  share_queue=sum(q_) / sum(tt), share_prefill=sum(p_) / sum(tt),
                                  share_post=sum(o_) / sum(tt))
        else:
            res["co_ttft"] = dict(n=0, reason=f"assign {len(assign)} != recv {len(recv)}")
        out.append(res)
    return out


def costterms_t3_summary():
    """Derived only: the costterms T3 co-residence interval arrays kept in git
    (t3_iterations.json; raw telemetry is on the VESSL volume /data/costterms)."""
    out = []
    for f in sorted(glob.glob(str(RES / "cap_char" / "costterms_runs" / "*" / "t3_iterations.json"))):
        d = json.load(open(f))
        for key, cells in d.items():
            for cell, reps in cells.items():
                for rep, x in reps.items():
                    co = x.get("co_ms") or []
                    if not co:
                        continue
                    solo = x.get("solo_ms")
                    if isinstance(solo, list):
                        solo = percentile(solo, .5) if solo else None
                    out.append(dict(job=Path(f).parent.name, key=key, cell=cell, rep=rep, n=len(co),
                                    co_p50=percentile(co, .5), co_p99=percentile(co, .99), co_max=max(co),
                                    solo_med=solo, max_over_solo=(max(co) / solo) if solo else None,
                                    n_gt5x_solo=sum(1 for v in co if solo and v > 5 * solo)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "lockstep_diag_values.json"))
    ap.add_argument("--with-profile", action="store_true", help="also N2 Job-PROFILE a5de2d6")
    a = ap.parse_args()
    todo = n3_cells() + n2_cells("n2p_*7e09cc3*", "N2EA")
    if a.with_profile:
        todo += n2_cells("n2p_*a5de2d6*", "N2PROFILE")
    results, results_client = [], []
    for tel, lo, hi, meta, bench in todo:
        r = analyse_cell(tel, lo, hi, meta, bench)
        if r is not None:
            results.append(r)
    for job in sorted(glob.glob(str(RES / "n2_static_grid" / "n2p_*7e09cc3*"))):
        for tel in sorted(glob.glob(job + "/tel_*.jsonl")):
            results_client.extend(n2ea_client(job, Path(tel).stem[4:]))
    # client-aligned regression: median client interval inside a co span vs W / n_chunks
    xs, ys = [], []
    for r in results_client:
        for s in r["co_spans"]:
            if s["med_int_ms"] is not None:
                xs.append(s["W_ms"] / s["n_chunks"]); ys.append(s["med_int_ms"])
    reg_client = dict(pooled=slope_ci(xs, ys), x_range=[min(xs), max(xs)],
                      y_range=[min(ys), max(ys)]) if len(xs) >= 5 else {}
    for key in sorted({(r["d"], r["b"]) for r in results_client}):
        xs, ys = [], []
        for r in results_client:
            if (r["d"], r["b"]) != key:
                continue
            for s_ in r["co_spans"]:
                if s_["med_int_ms"] is not None:
                    xs.append(s_["W_ms"] / s_["n_chunks"]); ys.append(s_["med_int_ms"])
        if len(xs) >= 5:
            reg_client[f"d{key[0]}_b{key[1]}"] = dict(slope_ci(xs, ys), x_range=[min(xs), max(xs)],
                                                      y_median=statistics.median(ys),
                                                      x_median=statistics.median(xs))
    # pooled regressions per dataset: per span, y = median I(CHUNK) , x = W / n_chunks
    reg = {}
    for ds in sorted({r["dataset"] for r in results}):
        xs, ys, xs2, ys2 = [], [], [], []
        for r in results:
            if r["dataset"] != ds:
                continue
            solo = r["by_class"].get("SOLO", {}).get("p50")
            for s in r["spans"]:
                if s["n_chunk_it"] >= 2 and s["medI_chunk"] is not None and s["n_chunks_expected"]:
                    nch = s["n_chunks_expected"]
                    xs.append(s["W_ms"] / nch)
                    ys.append(s["medI_chunk"])
                    if solo:
                        xs2.append(s["W_ms"] / nch)
                        ys2.append(s["medI_chunk"] - solo)
        if len(xs) >= 5:
            reg[ds] = dict(y_medIchunk_vs_x_Wpertchunk=slope_ci(xs, ys),
                           y_excess_vs_x=slope_ci(xs2, ys2) if len(xs2) >= 5 else None,
                           x_range=[min(xs), max(xs)], y_range=[min(ys), max(ys)])
    json.dump(dict(results=results, regressions=reg, results_client=results_client,
                   regression_client=reg_client, costterms_t3=costterms_t3_summary()), open(a.out, "w"), indent=1, default=float)
    print("cells", len(results), "->", a.out)


if __name__ == "__main__":
    main()
