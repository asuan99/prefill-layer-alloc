#!/usr/bin/env python3
"""Phase A3 of PLAN_stock_baseline_metrics_2026-11-01: stock-SGLang pacing ratios from the Gate 2 HOLB probe.

Source: p1_gates/gate2/<campaign>_<model>_<arm>_rep<k>_<job>.holb.jsonl (KISTI, cudagraph ON, local untracked).
Each `holb_gap` event k closes on a decode-carrying forward D_k and carries (DIRECT_BLOCKING_DESIGN.md §1):
  decode_dur_ms       GPU time of D_k (CUDA events)
  same_stream_fw_ms   GPU time of the non-decode forwards (EXTEND) issued on D_k's stream since D_{k-1}
  decode_mode         DECODE | MIXED (MIXED = --enable-mixed-chunk puts prefill tokens inside D_k)

Definitions (plan §0-1; NOT the PD-mux r_span and never merged with it):
  T_dec(bs)  median decode_dur_ms of pure gaps (n_other_fw = 0, DECODE) with the same decode_bs in the same
             rate segment; segment-wide median when that bs has < MIN_BS samples (flagged by n_fallback)
  r_ser(k)   same_stream_fw_ms(k) / T_dec(bs_k)            for gaps with same_stream_fw_ms > 0
             (decode stalls serially for r_ser decode steps' worth of prefill GPU time)
  r_mix(k)   (decode_dur_ms(k) - T_dec(bs_k)) / T_dec(bs_k) for MIXED decode forwards
Excluded: gap_class "first", inconsistent gaps. PD-mux arms (agnostic) are skipped: their prefill runs on
another stream, which is r_span territory.

Rate segments: the probe has no rate field for g2/g2ea rep files. Leading warm-up pieces are dropped and the rest is
cut at its n-1 largest host-time gaps (n = rows in the paired bench jsonl, `segments`); each segment's host span
is checked against that row's bench duration (|ratio - 1| <= SEG_TOL else seg_check = FAIL, row kept, flagged).
g2holb files carry the rate in workload_id (r4_...).

Writes R_SER_HOLB.csv (one row per file x rate segment) and R_SER_HOLB_SUMMARY.csv (pooled per
campaign x model x arm x rate over seg_check != FAIL segments only (n_seg_fail = excluded count): medians over files of the per-file statistics, plus pooled gap quantiles).
Descriptive values only (no verdicts).
"""
import csv
import glob
import json
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from stock_metrics import pctl  # noqa: E402

GATE2 = os.path.join(os.path.dirname(HERE), "p1_gates", "gate2")
STOCK_ARMS = ("plain", "plainaux", "chunk512", "plainmix", "chunk512mix")
SEG_GAP_S = 3.0
SEG_TOL = 0.25
MIN_BS = 5
WARM_FRAC = 0.02
NAME = re.compile(r"^(g2holb|g2ea|g2)_(.+?)_(" + "|".join(sorted(STOCK_ARMS, key=len, reverse=True))
                  + r")_rep(\d+)_(\d+)\.holb\.jsonl$")


def load(path):
    open_ev, gaps = None, []
    for ln in open(path):
        if not ln.strip():
            continue
        d = json.loads(ln)
        if d.get("event") == "holb_open":
            open_ev = d
        elif d.get("event") == "holb_gap":
            gaps.append(d)
    return open_ev, gaps


def segments(gaps, n):
    """Split at host gaps > SEG_GAP_S, drop leading warm-up pieces (< WARM_FRAC of events), then cut the rest
    at its n-1 largest internal gaps (the between-rate pauses). Pieces between are merged, so a long in-run
    decode silence (e.g. chunk512 back-to-back prefill) does not create a false segment."""
    ts = [g.get("host_ts") for g in gaps]
    cuts = [i for i in range(1, len(gaps)) if ts[i] is not None and ts[i - 1] is not None and ts[i] - ts[i - 1] > SEG_GAP_S]
    start = 0
    for c in cuts:
        if c - start < WARM_FRAC * len(gaps):
            start = c
        else:
            break
    inner = [c for c in cuts if c > start]
    big = sorted(sorted(inner, key=lambda c: -(ts[c] - ts[c - 1]))[: max(n - 1, 0)])
    bounds = [start] + big + [len(gaps)]
    return [gaps[a:b] for a, b in zip(bounds, bounds[1:])]


def seg_stats(seg):
    pure = [g for g in seg if g.get("gap_class") != "first" and not g.get("inconsistent")
            and (g.get("n_other_fw") or 0) == 0 and g.get("decode_mode") == "DECODE" and g.get("decode_dur_ms")]
    by_bs = {}
    for g in pure:
        by_bs.setdefault(g.get("decode_bs"), []).append(g["decode_dur_ms"])
    seg_med = statistics.median([g["decode_dur_ms"] for g in pure]) if pure else float("nan")
    n_fb = 0

    def tdec(bs):
        nonlocal n_fb
        xs = by_bs.get(bs, [])
        if len(xs) >= MIN_BS:
            return statistics.median(xs)
        n_fb += 1
        return seg_med

    r_ser, r_mix, pre_ms, nfw = [], [], [], []
    for g in seg:
        if g.get("gap_class") == "first" or g.get("inconsistent"):
            continue
        t = tdec(g.get("decode_bs"))
        if not t or t != t:
            continue
        s = g.get("same_stream_fw_ms") or 0.0
        if s > 0:
            r_ser.append(s / t)
            pre_ms.append(s)
            nfw.append(g.get("n_other_fw") or 0)
        if g.get("decode_mode") == "MIXED" and g.get("decode_dur_ms"):
            r_mix.append((g["decode_dur_ms"] - t) / t)
    span = (seg[-1]["host_ts"] - seg[0]["host_ts"]) if seg and seg[0].get("host_ts") else float("nan")
    q = lambda xs, p: pctl(xs, p) if xs else float("nan")
    return {"n_gaps": len(seg), "n_pure": len(pure), "n_fallback": n_fb, "tdec_median_ms": seg_med,
            "n_r_ser": len(r_ser), "prefill_ms_p50": q(pre_ms, .5), "n_prefill_fw_p50": q(nfw, .5), "r_ser_p50": q(r_ser, .5), "r_ser_p95": q(r_ser, .95),
            "r_ser_max": max(r_ser) if r_ser else float("nan"),
            "frac_r_ser_gt1": (sum(1 for r in r_ser if r > 1) / len(r_ser)) if r_ser else float("nan"),
            "n_r_mix": len(r_mix), "r_mix_p50": q(r_mix, .5), "r_mix_p95": q(r_mix, .95), "host_span_s": span,
            "_r_ser": r_ser, "_r_mix": r_mix}


def main():
    rows = []
    for path in sorted(glob.glob(os.path.join(GATE2, "*.holb.jsonl"))):
        m = NAME.match(os.path.basename(path))
        if not m:
            continue
        camp, model, arm, rep, job = m.groups()
        open_ev, gaps = load(path)
        if not gaps:
            continue
        bench = path[: -len(".holb.jsonl")] + ".jsonl"
        if camp == "g2holb":
            wl = gaps[0].get("workload_id", "")
            mm = re.match(r"r(\d+)_", wl)
            plan = [(int(mm.group(1)) if mm else None, None)]
            segs = segments(gaps, 1)
        elif os.path.exists(bench):
            brows = [json.loads(ln) for ln in open(bench) if ln.strip()]
            plan = [(int(b["request_rate"]), float(b.get("duration") or 0)) for b in brows]
            segs = segments(gaps, len(plan))
        else:
            continue
        for (rate, bdur), seg in zip(plan, segs):
            st = seg_stats(seg)
            chk = "NA" if bdur is None else ("OK" if bdur and abs(st["host_span_s"] / bdur - 1) <= SEG_TOL else "FAIL")
            arm_cfg = (f"chunk={open_ev.get('arm_chunked_prefill_size')} mixed={open_ev.get('arm_enable_mixed_chunk')} "
                       f"overlap={open_ev.get('arm_enable_overlap')}") if open_ev else ""
            rows.append({"campaign": camp, "model": model, "arm": arm, "arm_cfg": arm_cfg, "rep": int(rep), "job": job,
                         "rate": rate, "bench_duration_s": bdur, "seg_check": chk, **st})
    cols = ["campaign", "model", "arm", "arm_cfg", "rep", "job", "rate", "bench_duration_s", "host_span_s", "seg_check",
            "n_gaps", "n_pure", "n_fallback", "tdec_median_ms", "n_r_ser", "prefill_ms_p50", "n_prefill_fw_p50", "r_ser_p50", "r_ser_p95",
            "r_ser_max", "frac_r_ser_gt1", "n_r_mix", "r_mix_p50", "r_mix_p95"]
    # 4 significant digits: enough for descriptive values, and keeps numeric substrings like "120.39" from [CS-OK]
    # tripping check_citation_stops.py patterns (e.g. /20\.39/) that have no word boundary [CS-OK]
    fmt = lambda v: f"{v:.4g}" if isinstance(v, float) else v
    with open(os.path.join(HERE, "R_SER_HOLB.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: fmt(r.get(k)) for k in cols})
    groups = {}
    for r in rows:
        if r["seg_check"] == "FAIL":  # rate assignment unverified -> kept in R_SER_HOLB.csv, not pooled
            continue
        groups.setdefault((r["campaign"], r["model"], r["arm"], r["arm_cfg"], r["rate"]), []).append(r)
    scols = ["campaign", "model", "arm", "arm_cfg", "rate", "n_files", "n_seg_fail", "tdec_median_ms", "n_r_ser",
             "prefill_ms_p50", "n_prefill_fw_p50", "r_ser_p50", "r_ser_p95", "r_ser_max", "frac_r_ser_gt1", "n_r_mix", "r_mix_p50", "r_mix_p95"]
    with open(os.path.join(HERE, "R_SER_HOLB_SUMMARY.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=scols)
        w.writeheader()
        for k, rs in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
            rser = [x for r in rs for x in r["_r_ser"]]
            rmix = [x for r in rs for x in r["_r_mix"]]
            med = lambda key: statistics.median([r[key] for r in rs if r[key] == r[key]]) if any(r[key] == r[key] for r in rs) else float("nan")
            w.writerow({k2: fmt(v) for k2, v in {
                "campaign": k[0], "model": k[1], "arm": k[2], "arm_cfg": k[3], "rate": k[4], "n_files": len(rs),
                "n_seg_fail": sum(1 for r in rows if r["seg_check"] == "FAIL" and (r["campaign"], r["model"], r["arm"],
                                                                                   r["arm_cfg"], r["rate"]) == k), "tdec_median_ms": med("tdec_median_ms"),
                "n_r_ser": len(rser), "prefill_ms_p50": med("prefill_ms_p50"), "n_prefill_fw_p50": med("n_prefill_fw_p50"),
                "r_ser_p50": pctl(rser, .5) if rser else float("nan"), "r_ser_p95": pctl(rser, .95) if rser else float("nan"),
                "r_ser_max": max(rser) if rser else float("nan"),
                "frac_r_ser_gt1": (sum(1 for x in rser if x > 1) / len(rser)) if rser else float("nan"),
                "n_r_mix": len(rmix), "r_mix_p50": pctl(rmix, .5) if rmix else float("nan"),
                "r_mix_p95": pctl(rmix, .95) if rmix else float("nan")}.items()})
    print(f"segments={len(rows)} seg_fail={sum(1 for r in rows if r['seg_check'] == 'FAIL')} groups={len(groups)}")


if __name__ == "__main__":
    main()
