#!/usr/bin/env python3
"""Label / estimand code of PREREG_MULTIGPU_2026-10-12.md (rev6) §4-3…§4-6.  CPU only, stdlib+numpy.

  per cell   Q_c(t), g, CLIENT_ERROR, SATURATION_UNRESOLVED, send-lag p99, cgroup throttle ratio,
             HARNESS_LIMITED, FRONTEND_SATURATED (+ per-thread max, MGP-30), achieved throughput
  per key    (config, arm, shape): HI_SATURATED / LO_UNSATURATED / KNEE_BRACKETED / LADDER_TOO_LOW,
             lambda* = median of the 4 HI achieved values, PARTITION_NOT_EXERCISED
  per config MEMORY_MISMATCH (G5(c)), GPU_CONTAMINATED, FRONTEND_SATURATED, M2g exclusions
  estimands  G_sym (primary) = lambda*_agnostic / lambda*_fused_sym ; G_def, G_d44,sym, S (auxiliary)
             seed-paired bootstrap CI (10000, seed 1), trend slope of log G on log2 N (seed index
             resampled JOINTLY over all N and both arms), magnitude rule, ANCHOR_DRIFT,
             GPU_HETEROGENEOUS, P-DP check, undefined-cell count + reasons beside every trend (MGP-34/25)

H9 CONVENTION (explicit; PREREG §4-3 says only "TTFT가 없는 오류 요청은 Q_c에서 뺀다"):
  * an ERROR REQUEST is a record with a non-empty `error`, OR `ttft_ms is None`.  All of them count
    toward the 1 % CLIENT_ERROR test, whether or not a first token arrived.
  * An error request WITHOUT ttft is removed from Q_c (it never has a first-token time) and counted.
  * An error request WITH ttft (the stream failed AFTER the first token) KEEPS its Q_c interval
    [sent, sent+ttft): it really waited for its first token and then left the queue; dropping it
    would understate Q_c exactly when the server is slow.  It is NOT a successful completion: it is
    excluded from the achieved-throughput numerator (and trace_loadgen's own summary excludes it).
  * n in r = n/T is ALL requests sent in the window (errors included): the offered load.

The estimators the PREREG forbids (§4-3 mutation list) live below as _MUT_* functions and are run by
`--selftest` against synthetic scenarios; each must be KILLED and the unmutated control must not be.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import math
import os
import re
import statistics
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mg_const as C  # noqa: E402

# =========================================================================== requests → Q_c, g
def load_requests(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def is_error(r):
    return bool(r.get("error")) or r.get("ttft_ms") is None


def qc_intervals(reqs):
    """[(sent, first_token_time)] of the requests that live in Q_c (H9 convention above)."""
    return [(float(r["sent_s"]), float(r["sent_s"]) + float(r["ttft_ms"]) / 1000.0)
            for r in reqs if r.get("ttft_ms") is not None]


def qc_value(reqs, t):
    """Q_c(t) = #{sent <= t} - #{sent + ttft <= t}  (literal PREREG §4-3 formula; for tests)."""
    iv = qc_intervals(reqs)
    return sum(1 for s, _ in iv if s <= t) - sum(1 for _, d in iv if d <= t)


def _integral(iv, a, b):
    """∫_a^b Q_c(t) dt = Σ_i |[s_i,d_i) ∩ [a,b]|  (Q_c is a sum of indicator functions)."""
    if not iv:
        return 0.0
    s = np.fromiter((x[0] for x in iv), float, len(iv))
    d = np.fromiter((x[1] for x in iv), float, len(iv))
    return float(np.clip(np.minimum(d, b) - np.maximum(s, a), 0.0, None).sum())


def window_of(reqs):
    sent = [float(r["sent_s"]) for r in reqs]
    return (min(sent), max(sent)) if sent else (0.0, 0.0)


def qc_thirds(reqs, intervals=None):
    """Time-integral mean of Q_c over the three equal thirds of the SEND window."""
    t0, t1 = window_of(reqs)
    T = t1 - t0
    iv = qc_intervals(reqs) if intervals is None else intervals
    if T <= 0:
        return T, [0.0, 0.0, 0.0]
    edges = [t0 + T * k / 3.0 for k in range(4)]
    means = [_integral(iv, edges[k], edges[k + 1]) / (T / 3.0) for k in range(3)]
    return T, means


def g_from_means(means, n, T):
    """g = (Q3 - Q1) / ((2T/3) r),  r = n/T."""
    if T <= 0 or n <= 0:
        return None
    r = n / T
    return (means[2] - means[0]) / ((2.0 * T / 3.0) * r)


def g_of(reqs):
    T, means = qc_thirds(reqs)
    return g_from_means(means, len(reqs), T), T, means


# =========================================================================== per-cell harness stats
def percentile(vals, q):
    vals = sorted(vals)
    if not vals:
        return None
    k = (len(vals) - 1) * q / 100.0
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    return vals[lo] + (vals[hi] - vals[lo]) * (k - lo)


def send_lag_p99_ms(reqs):
    lags = [(float(r["sent_s"]) - float(r["scheduled_arrival_s"])) * 1000.0
            for r in reqs if r.get("scheduled_arrival_s") is not None]
    return percentile(lags, 99.0), len(lags)


def parse_cpu_stat(text):
    d = {}
    for line in (text or "").splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].lstrip("-").isdigit():
            d[parts[0]] = int(parts[1])
    return d


def throttle_ratio(before_text, after_text):
    """Δnr_throttled / Δnr_periods over the cell (MGP-31).  None when Δnr_periods == 0 / unreadable."""
    b, a = parse_cpu_stat(before_text), parse_cpu_stat(after_text)
    if "nr_periods" not in b or "nr_periods" not in a:
        return None, None, None
    dp = a["nr_periods"] - b["nr_periods"]
    dt = a.get("nr_throttled", 0) - b.get("nr_throttled", 0)
    return (dt / dp if dp > 0 else None), dt, dp


def frontend_stats(csv_path, win_mono):
    """FRONTEND_SATURATED inputs from proc_samples.csv (columns t_mono,role,pid,tid,ticks; tid=-1 is the
    process total = utime+stime of /proc/<pid>/stat).  For each role: fraction of consecutive-sample
    intervals (≈1 s) inside the send window whose core-use ≥ 0.90.  Per-thread maxima are recorded too
    (MGP-30) but never enter the label.  Roles: tokenizer_manager, detokenizer, dp_controller."""
    if not csv_path or not os.path.exists(csv_path):
        return None
    clk = float(C.CLK_TCK)
    series = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            key = (row["role"], int(row["pid"]), int(row["tid"]))
            series.setdefault(key, []).append((float(row["t_mono"]), int(row["ticks"])))
    lo, hi = win_mono
    per_role, per_thread = {}, {}
    for (role, pid, tid), pts in series.items():
        pts.sort()
        for (ta, ka), (tb, kb) in zip(pts, pts[1:]):
            if tb <= ta or ta < lo or tb > hi:
                continue
            use = (kb - ka) / clk / (tb - ta)
            if tid == -1:
                per_role.setdefault(role, []).append(use)
            else:
                per_thread[role] = max(per_thread.get(role, 0.0), use)
    out = {"roles": {}, "per_thread_max_cores": per_thread}
    flagged = False
    for role, uses in per_role.items():
        frac = sum(1 for u in uses if u >= C.FRONTEND_CORE_FRAC) / len(uses)
        out["roles"][role] = {"seconds": len(uses), "frac_seconds_ge_0.90": frac, "max_cores": max(uses)}
        flagged = flagged or frac >= C.FRONTEND_SECOND_FRAC
    out["frontend_saturated"] = flagged
    out["n_roles"] = len(per_role)
    return out


# =========================================================================== server-log (descriptive only)
_TS = re.compile(r"(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?)")
_DEC = re.compile(r"Decode batch.*?#running-req: *(\d+).*?#queue-req: *(\d+)")
_PRE = re.compile(r"Prefill batch.*?#running-req: *(\d+).*?#queue-req: *(\d+)")


def _epoch(ts):
    ts = ts.replace("T", " ")
    fmt = "%Y-%m-%d %H:%M:%S.%f" if "." in ts else "%Y-%m-%d %H:%M:%S"
    return _dt.datetime.strptime(ts, fmt).replace(tzinfo=_dt.timezone.utc).timestamp()


def parse_server_log(text):
    dec, pre = [], []
    for line in text.splitlines():
        m = _TS.search(line)
        if not m:
            continue
        t = _epoch(m.group(1))
        d = _DEC.search(line)
        if d:
            dec.append((t, int(d.group(1)), int(d.group(2))))
            continue
        p = _PRE.search(line)
        if p:
            pre.append((t, int(p.group(1)), int(p.group(2))))
    return dec, pre


def server_log_descriptive(text, cap, win=None):
    """Descriptive only (MGP-17): never an input to a label.  win=(epoch_lo, epoch_hi) or None."""
    dec, pre = parse_server_log(text)
    if win:
        dec = [x for x in dec if win[0] <= x[0] <= win[1]]
        pre = [x for x in pre if win[0] <= x[0] <= win[1]]
    f = lambda xs, i: [x[i] for x in xs]
    out = {"decode_lines": len(dec), "prefill_lines": len(pre)}
    if dec:
        run, q = f(dec, 1), f(dec, 2)
        out.update(decode_running_mean=statistics.fmean(run), decode_running_max=max(run),
                   decode_frac_running_eq_cap=sum(1 for v in run if v == cap) / len(run),
                   decode_queue_mean=statistics.fmean(q), decode_queue_max=max(q))
    if pre:
        out.update(prefill_queue_mean=statistics.fmean(f(pre, 2)), prefill_queue_max=max(f(pre, 2)))
    return out


# =========================================================================== mutants (§4-3 list) — NOT used for labels
def _class_of_g(g):
    if g is None:
        return "NA"
    return "HI" if g >= C.G_HI else ("LO" if g <= C.G_LO else "MID")


def _MUT_a_server_decode_lines(text, win_epoch, n, T):
    """(a) "서버 Decode 줄만 쓰는 추정량": g from #queue-req of Decode lines (sample mean per third)."""
    dec, _ = parse_server_log(text)
    lo, hi = win_epoch
    thirds = [[], [], []]
    for t, _run, q in dec:
        if lo <= t <= hi:
            thirds[min(2, int((t - lo) / ((hi - lo) / 3.0)))].append(q)
    means = [statistics.fmean(x) if x else 0.0 for x in thirds]
    return g_from_means(means, n, T)


def _MUT_b_sample_mean(reqs):
    """(b) "표본 평균(시간 적분 아님)": mean of Q_c sampled at the event instants inside each third."""
    iv = qc_intervals(reqs)
    t0, t1 = window_of(reqs)
    T = t1 - t0
    ev = sorted([s for s, _ in iv] + [d for _, d in iv])
    thirds = [[], [], []]
    for t in ev:
        if t0 <= t <= t1:
            k = min(2, int((t - t0) / (T / 3.0)))
            thirds[k].append(sum(1 for s, d in iv if s <= t < d))
    means = [statistics.fmean(x) if x else 0.0 for x in thirds]
    return g_from_means(means, len(reqs), T)


def _MUT_c_keep_errors(reqs):
    """(c) "오류 요청을 Q_c에 남김": an error request without ttft never leaves the queue."""
    t0, t1 = window_of(reqs)
    iv = qc_intervals(reqs) + [(float(r["sent_s"]), float("inf")) for r in reqs if r.get("ttft_ms") is None]
    T = t1 - t0
    edges = [t0 + T * k / 3.0 for k in range(4)]
    means = [_integral([(s, min(d, 1e18)) for s, d in iv], edges[k], edges[k + 1]) / (T / 3.0) for k in range(3)]
    return g_from_means(means, len(reqs), T)


def _MUT_d_running_eq_cap(text, cap):
    """(d) "running==cap 술어": saturated iff ≥ half of the Decode lines show #running-req == cap."""
    dec, _ = parse_server_log(text)
    if not dec:
        return None
    frac = sum(1 for _, r, _q in dec if r == cap) / len(dec)
    return 1.0 if frac >= 0.5 else 0.0


# --------------------------------------------------------------------------- synthetic scenarios
def _log_line(t, kind, running, queue):
    ts = _dt.datetime.fromtimestamp(t, _dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    if kind == "D":
        return (f"[{ts} TP0] Decode batch, #running-req: {running}, #token: 1000, token usage: 0.10, "
                f"gen throughput (token/s): 1000.00, #queue-req: {queue}, ")
    return (f"[{ts} TP0] Prefill batch, #new-seq: 1, #new-token: 8192, #cached-token: 0, "
            f"token usage: 0.10, #running-req: {running}, #queue-req: {queue}, ")


def _fifo_cell(x, lam, T, base_ttft=0.2):
    """Uniform arrivals at rate x, single FIFO server of rate lam: ttft = wait + base."""
    n = int(round(x * T))
    reqs, free = [], 0.0
    for i in range(n):
        s = i / x
        start = max(s, free)
        free = start + 1.0 / lam
        reqs.append({"sent_s": s, "scheduled_arrival_s": s, "ttft_ms": (start - s + base_ttft) * 1000.0,
                     "error": "", "completion_s": free + 5.0})
    return reqs


def _log_for(reqs, epoch0, running, decode_frac=1.0, queue_fn=None, cap=48):
    """Decode lines every 1 s over the first `decode_frac` of the window; Prefill lines in the rest."""
    t0, t1 = window_of(reqs)
    lines, T = [], t1 - t0
    t = t0
    while t <= t1:
        q = queue_fn(t) if queue_fn else 0
        lines.append(_log_line(epoch0 + t, "D" if (t - t0) <= decode_frac * T else "P", running, q))
        t += 1.0
    return "\n".join(lines)


def _queue_fn_of(reqs):
    return lambda t: qc_value(reqs, t)


def synthetic_scenarios():
    """name -> dict(reqs, log, epoch0, expected class, cap)."""
    cap, epoch0 = 48, 1.7e9
    sc = {}
    lam, T = 4.0, 120.0
    r = _fifo_cell(1.6 * lam, lam, T)
    sc["S1_saturated_fifo"] = dict(reqs=r, log=_log_for(r, epoch0, 20, 1.0, _queue_fn_of(r)), expected="HI")
    r = _fifo_cell(0.6 * lam, lam, T)
    sc["S2_unsaturated"] = dict(reqs=r, log=_log_for(r, epoch0, 20, 1.0, _queue_fn_of(r)), expected="LO")
    # S3: quiet third, then a 60-request burst that drains in ~1 s.  Time-integral g ~ 0.02; event-sample mean >> 0.10
    r = [{"sent_s": float(i), "scheduled_arrival_s": float(i), "ttft_ms": 200.0, "error": "", "completion_s": i + 5.0}
         for i in range(100)]
    r += [{"sent_s": 90.0 + 0.5 * k / 60.0, "scheduled_arrival_s": 90.0 + 0.5 * k / 60.0, "ttft_ms": 1000.0,
           "error": "", "completion_s": 95.0} for k in range(60)]
    sc["S3_burst_unsaturated"] = dict(reqs=r, log=_log_for(r, epoch0, 20, 1.0, lambda t: 0), expected="LO")
    # S4: fused prefill priority -- queue grows (S1), but the server prints Decode lines for the first 25 % only
    r = _fifo_cell(1.6 * lam, lam, T)
    sc["S4_prefill_priority_saturated"] = dict(reqs=r, log=_log_for(r, epoch0, 20, 0.25, lambda t: 1), expected="HI")
    # S5: server full (running == cap in every Decode line) but the client queue is flat
    r = _fifo_cell(0.6 * lam, lam, T)
    sc["S5_cap_bound_unsaturated"] = dict(reqs=r, log=_log_for(r, epoch0, cap, 1.0, lambda t: 0), expected="LO")
    # S6: saturated while running stays far below cap (shape-B-like)
    r = _fifo_cell(1.6 * lam, lam, T)
    sc["S6_saturated_low_running"] = dict(reqs=r, log=_log_for(r, epoch0, 2, 1.0, _queue_fn_of(r)), expected="HI")
    # S7: g just under 0.05 plus 4 late error requests without ttft (1 % of n = 400)
    n, c = 400, None
    for c in np.arange(6.0, 30.0, 0.05):
        base = [{"sent_s": float(i), "scheduled_arrival_s": float(i),
                 "ttft_ms": (1.0 if i < n / 3 else (c if i >= 2 * n / 3 else 1.0)) * 1000.0,
                 "error": "", "completion_s": i + 5.0} for i in range(n)]
        g, _, _ = g_of(base)
        if g is not None and g >= 0.040:
            break
    errs = [{"sent_s": float(i), "scheduled_arrival_s": float(i), "ttft_ms": None, "error": "ReadTimeout",
             "completion_s": i + 1.0} for i in (270, 271, 272, 273)]
    for i, e in zip((270, 271, 272, 273), errs):
        base[i] = e
    sc["S7_errors_near_threshold"] = dict(reqs=base, log=_log_for(base, epoch0, 20, 1.0, lambda t: 0), expected="LO")
    for v in sc.values():
        v["epoch0"], v["cap"] = epoch0, cap
    return sc


def correct_class(s):
    g, _, _ = g_of(s["reqs"])
    return _class_of_g(g)


def mutant_classes(s):
    reqs, log, e0, cap = s["reqs"], s["log"], s["epoch0"], s["cap"]
    t0, t1 = window_of(reqs)
    T, n = t1 - t0, len(reqs)
    return {
        "a_server_decode_lines": _class_of_g(_MUT_a_server_decode_lines(log, (e0 + t0, e0 + t1), n, T)),
        "b_sample_mean": _class_of_g(_MUT_b_sample_mean(reqs)),
        "c_keep_errors_in_Qc": _class_of_g(_MUT_c_keep_errors(reqs)),
        "d_running_eq_cap": _class_of_g(_MUT_d_running_eq_cap(log, cap)),
        "CONTROL_unmutated": correct_class(s),
    }


def run_mutation_checks():
    """Lesson 53 + 254: each mutant must DIFFER from the expected class on >= 1 scenario (KILLED);
    the no-mutation control must match on ALL scenarios (a control that fails makes the scenario set invalid)."""
    sc = synthetic_scenarios()
    verdict, per = {}, {}
    for name, s in sc.items():
        per[name] = mutant_classes(s)
        per[name]["expected"] = s["expected"]
    for m in ("a_server_decode_lines", "b_sample_mean", "c_keep_errors_in_Qc", "d_running_eq_cap", "CONTROL_unmutated"):
        killed_on = [n for n, p in per.items() if p[m] != p["expected"]]
        verdict[m] = {"killed_on": killed_on, "killed": bool(killed_on)}
    ok = (all(verdict[m]["killed"] for m in verdict if m != "CONTROL_unmutated")
          and not verdict["CONTROL_unmutated"]["killed"])
    return ok, verdict, per


# =========================================================================== cell analysis
def analyze_cell(reqs, meta, cpu_before="", cpu_after="", proc_csv=None, win_mono=None, etime=None):
    """meta: cell.json dict.  Returns the per-cell record used by the aggregation."""
    n = len(reqs)
    errs = [r for r in reqs if is_error(r)]
    err_no_ttft = [r for r in errs if r.get("ttft_ms") is None]
    rec = {"cell_id": meta.get("cell_id"), "n_sent": n, "n_errors": len(errs),
           "n_errors_without_ttft": len(err_no_ttft), "n_errors_with_ttft": len(errs) - len(err_no_ttft),
           "truncated": bool(meta.get("truncated"))}
    if n == 0:
        rec.update(status="EMPTY", g=None)
        return rec
    t0, t1 = window_of(reqs)
    T = t1 - t0
    g, _, means = g_of(reqs)
    rec.update(T=T, r_realized=(n / T if T > 0 else None), q_means=means, g=g)
    if len(errs) / n > C.CLIENT_ERROR_FRAC:
        rec["status"] = "CLIENT_ERROR"
    elif n < C.MIN_WINDOW_REQS:
        rec["status"] = "SATURATION_UNRESOLVED"
    elif rec["truncated"]:
        rec["status"] = "TRUNCATED"
    else:
        rec["status"] = "OK"
    ok = [r for r in reqs if not is_error(r) and r.get("completion_s") is not None]
    if ok:
        span = max(float(r["completion_s"]) for r in ok) - t0
        rec["achieved"] = len(ok) / span if span > 0 else None
    else:
        rec["achieved"] = None
    lag, nlag = send_lag_p99_ms(reqs)
    rec["send_lag_p99_ms"] = lag
    tr, dt, dp = throttle_ratio(cpu_before, cpu_after)
    rec.update(throttle_ratio=tr, nr_throttled_delta=dt, nr_periods_delta=dp)
    rec["throttle_unmeasured"] = tr is None
    rec["harness_limited"] = bool((lag is not None and lag > C.LAG_P99_MS) or (tr is not None and tr > C.THROTTLE_FRAC))
    fe = frontend_stats(proc_csv, win_mono) if (proc_csv and win_mono) else None
    rec["frontend"] = fe
    rec["frontend_saturated"] = bool(fe and fe["frontend_saturated"])
    rec["frontend_unmeasured"] = fe is None
    if etime is not None:
        rec["etime"] = etime
    return rec


def load_cell(cell_dir):
    cell_dir = Path(cell_dir)
    meta = json.loads((cell_dir / "cell.json").read_text())
    meta.setdefault("cell_id", cell_dir.name)
    req_path = cell_dir / "requests.jsonl"
    reqs = load_requests(req_path) if req_path.exists() else []
    rd = lambda n: (cell_dir / n).read_text() if (cell_dir / n).exists() else ""
    win_mono = None
    clk = cell_dir / "loadgen_clock.json"
    if clk.exists() and reqs:
        c = json.loads(clk.read_text())
        t0, t1 = window_of(reqs)
        win_mono = (c["perf_at_call"] + t0, c["perf_at_call"] + t1)
    et = None
    if (cell_dir / "etime.json").exists():
        et = json.loads((cell_dir / "etime.json").read_text())
    rec = analyze_cell(reqs, meta, rd("cpu_stat_before.txt"), rd("cpu_stat_after.txt"),
                       str(cell_dir / "proc_samples.csv"), win_mono, et)
    rec["meta"] = meta
    rec["achieved_by_seed_key"] = (meta.get("seed"), rec.get("achieved"))
    return rec


# =========================================================================== key-level labels
def _key(m):
    return (m["config"], m["arm"], m["shape"], int(m.get("round", 1)))


def _et(c):
    """E-time value of a cell: mg_realized_mix.cell_e_time writes `e_time` (older drafts `etime`)."""
    e = c.get("etime") or {}
    return e.get("e_time", e.get("etime"))


def key_labels(cells):
    """cells: list of analyzed-cell records of ONE key.  Final HI set = the HI cells with the largest `ext`."""
    hi_all = [c for c in cells if c["meta"]["rung"] == "HI"]
    lo_all = [c for c in cells if c["meta"]["rung"] == "LO"]
    ext = max((int(c["meta"].get("ext", 0)) for c in hi_all), default=0)
    hi = [c for c in hi_all if int(c["meta"].get("ext", 0)) == ext]
    lo_ext = max((int(c["meta"].get("ext", 0)) for c in lo_all), default=0)
    lo = [c for c in lo_all if int(c["meta"].get("ext", 0)) == lo_ext]
    out = {"ext": ext, "n_hi": len(hi), "n_lo": len(lo), "reasons": []}
    byseed = {int(c["meta"]["seed"]): c for c in hi}
    ok_hi = [byseed.get(s) for s in C.HI_SEEDS]
    bad = [s for s, c in zip(C.HI_SEEDS, ok_hi) if c is None or c["status"] != "OK"]
    out["hi_bad_seeds"] = {int(s): (None if byseed.get(s) is None else byseed[s]["status"]) for s in bad}
    out["g_hi"] = {int(s): (byseed[s]["g"] if s in byseed else None) for s in C.HI_SEEDS}
    if bad:
        out["hi_label"] = "HI_NOT_JUDGED"
        out["reasons"].append("HI_NOT_JUDGED:" + ",".join(f"{s}={out['hi_bad_seeds'][s] or 'MISSING'}" for s in map(int, bad)))
    elif all(c["g"] >= C.G_HI for c in ok_hi):
        out["hi_label"] = "HI_SATURATED"
    else:
        out["hi_label"] = "LADDER_TOO_LOW" if ext >= 1 else "HI_NOT_SATURATED"
        out["reasons"].append(out["hi_label"])
    # extension trigger (§4-2): a JUDGED HI cell with g < 0.10; CLIENT_ERROR / UNRESOLVED never trigger
    out["ext_trigger"] = any(c["status"] == "OK" and c["g"] is not None and c["g"] < C.G_HI for c in hi)
    lo_by = {int(c["meta"]["seed"]): c for c in lo}
    lo_c = [lo_by.get(s) for s in C.LO_SEEDS]
    lo_bad = [s for s, c in zip(C.LO_SEEDS, lo_c) if c is None or c["status"] != "OK"]
    out["g_lo"] = {int(s): (lo_by[s]["g"] if s in lo_by else None) for s in C.LO_SEEDS}
    if lo_bad:
        out["lo_label"] = "LO_NOT_JUDGED"
        out["reasons"].append("LO_NOT_JUDGED:" + ",".join(f"{s}={(lo_by[s]['status'] if s in lo_by else 'MISSING')}" for s in map(int, lo_bad)))
    elif all(c["g"] <= C.G_LO for c in lo_c):
        out["lo_label"] = "LO_UNSATURATED"
    else:
        out["lo_label"] = "LO_ABOVE_0.05"
        out["reasons"].append("LO_ABOVE_0.05 (MGP-34: not an upper bound on lambda*)")
    out["knee"] = ("KNEE_BRACKETED" if out["hi_label"] == "HI_SATURATED" and out["lo_label"] == "LO_UNSATURATED"
                   else "KNEE_NOT_BRACKETED")
    ach = [byseed[s]["achieved"] if (s in byseed and byseed[s]["status"] == "OK") else None for s in C.HI_SEEDS]
    out["achieved_by_seed"] = ach
    if all(a is not None for a in ach):
        out["lambda_star"] = statistics.median(ach)
        out["lambda_star_range"] = [min(ach), max(ach)]
    else:
        out["lambda_star"] = None
        out["reasons"].append("LAMBDA_STAR_UNRESOLVED_N_BELOW_4")
    allc = hi + lo
    out["harness_limited"] = any(c.get("harness_limited") for c in allc)
    out["truncated"] = any(c.get("truncated") for c in allc)
    out["throttle_unmeasured_cells"] = sum(1 for c in allc if c.get("throttle_unmeasured"))
    out["frontend_saturated_hi_cells"] = [c["cell_id"] for c in hi if c.get("frontend_saturated")]
    ets = [_et(c) for c in hi if c.get("etime") and c["etime"].get("status") == "OK" and _et(c) is not None]
    if any(c.get("etime") and c["etime"].get("status") == "MIX_FAILED" for c in hi):
        out["mix_failed"] = True
    if ets and len(ets) == len(hi) and statistics.median(ets) < C.PARTITION_ETIME_MIN:
        out["partition_not_exercised"] = True
    out["etime_median"] = statistics.median(ets) if ets else None
    return out


# =========================================================================== G5(c) memory + contamination (config level)
def memory_check(arm_facts, shapes=("A", "B")):
    """arm_facts: arm -> dict(mamba=[...], max_total=[...], eff_cap=int, mode, N).  PREREG §3 G5(c) (i)-(iii)."""
    reasons = []
    mt_min = {}
    for arm, f in arm_facts.items():
        if any(x != f["eff_cap"] for x in f.get("mamba", [])) or not f.get("mamba"):
            reasons.append(f"{arm}: mamba cache slots {f.get('mamba')} != effective cap {f['eff_cap']}")
        need = max(f["eff_cap"] * sum(C.SHAPE_IO[s]) for s in shapes)
        if not f.get("max_total") or min(f["max_total"]) < need:
            reasons.append(f"{arm}: max_total_num_tokens {f.get('max_total')} < cap*(ISL+OSL)={need}")
        if f.get("max_total"):
            mt_min[arm] = min(f["max_total"])
    if len(mt_min) >= 2:
        hi, lo = max(mt_min.values()), min(mt_min.values())
        if (hi - lo) / hi > C.MEMORY_REL_TOL:
            reasons.append(f"arm max_total_num_tokens relative spread {(hi - lo) / hi:.4f} > {C.MEMORY_REL_TOL}")
    return {"memory_mismatch": bool(reasons), "reasons": reasons}


# =========================================================================== estimands
PAIRS = {"G_sym": ("agnostic", "fused_sym"), "G_def": ("agnostic", "fused"),
         "G_d44sym": ("d44", "fused_sym"), "S": ("fused_sym", "fused")}
PRIMARY = "G_sym"


def _med_idx(arr, idx):
    return np.median(arr[idx], axis=-1)


def paired_ratio_ci(num, den, reps=None, seed=None):
    """num/den: 4 achieved values indexed by HI_SEEDS order (seed-paired).  ratio of medians per
    resample of the SEED INDEX.  Returns (point, lo, hi)."""
    reps = reps or C.BOOTSTRAP_REPS
    rng = np.random.default_rng(C.BOOTSTRAP_SEED if seed is None else seed)
    num, den = np.asarray(num, float), np.asarray(den, float)
    idx = rng.integers(0, len(num), size=(reps, len(num)))
    g = np.median(num[idx], axis=1) / np.median(den[idx], axis=1)
    lo, hi = np.percentile(g, C.CI_PCT)
    return float(np.median(num) / np.median(den)), float(lo), float(hi)


def trend(points, reps=None):
    """points: list of (N, num[4], den[4]).  slope of log G on log2 N, SEED INDEX resampled jointly over all N
    and both arms (MGP-10).  Returns dict."""
    reps = reps or C.BOOTSTRAP_REPS
    pts = sorted(points, key=lambda p: p[0])
    if len(pts) < C.TREND_MIN_POINTS:
        return {"label": "TREND_UNDEFINED", "n_points": len(pts), "Ns": [p[0] for p in pts]}
    x = np.log2([p[0] for p in pts])
    xc = x - x.mean()
    denom = float((xc ** 2).sum())
    nums = np.array([p[1] for p in pts], float)     # (nN, 4)
    dens = np.array([p[2] for p in pts], float)
    k = nums.shape[1]
    rng = np.random.default_rng(C.BOOTSTRAP_SEED)
    idx = rng.integers(0, k, size=(reps, k))
    logg = np.stack([np.log(np.median(nums[i][idx], axis=1) / np.median(dens[i][idx], axis=1))
                     for i in range(len(pts))], axis=1)           # (reps, nN)
    slopes = (logg * xc).sum(axis=1) / denom
    point_logg = np.log(np.median(nums, axis=1) / np.median(dens, axis=1))
    s = float((point_logg * xc).sum() / denom)
    lo, hi = (float(v) for v in np.percentile(slopes, C.CI_PCT))
    label = "NO_TREND_DETECTED" if lo <= 0.0 <= hi else ("G_INCREASES_WITH_N" if lo > 0 else "G_DECREASES_WITH_N")
    # magnitude rule (§4-4): G(N_max)/G(1); needs N=1
    out = {"label": label, "slope": s, "ci": [lo, hi], "n_points": len(pts), "Ns": [p[0] for p in pts],
           "G_by_N": {int(p[0]): float(np.exp(v)) for p, v in zip(pts, point_logg)}}
    if pts[0][0] == 1:
        ratio = float(np.exp(point_logg[-1] - point_logg[0]))
        rb = np.exp(logg[:, -1] - logg[:, 0])
        out.update(N_max=int(pts[-1][0]), magnitude_ratio=ratio,
                   magnitude_ci=[float(v) for v in np.percentile(rb, C.CI_PCT)],
                   direction_citable=abs(ratio - 1.0) >= C.MAGNITUDE_BAND)
        out["magnitude"] = "OK" if out["direction_citable"] else "DIRECTION_NOT_CITABLE_UNDER_3PCT"
    else:
        out.update(magnitude="MAGNITUDE_UNDEFINED", direction_citable=None)
    return out


def aggregate(cells, pass_table=None, config_flags=None):
    """cells: analyzed-cell records (load_cell).  Returns the full label document."""
    pass_table = pass_table or {}
    config_flags = config_flags or {}
    groups = {}
    for c in cells:
        m = c["meta"]
        if m.get("block", "main") in ("main", "anchor") and m.get("rung") in ("HI", "LO"):
            groups.setdefault(_key(m), []).append(c)      # gpu_dev / numa / nsys / m2g blocks are descriptive
    keys = {k: key_labels(cs) for k, cs in groups.items()}
    doc = {"prereg": f"{C.PREREG} rev{C.PREREG_REV}", "keys": {}, "configs": {}, "estimands": {}, "trends": {},
           "anchor": {}, "blocks": {}}
    for (cfg, arm, shape, rnd), lab in sorted(keys.items()):
        excl = pass_table.get(f"{cfg}:{arm}")
        if excl is not None and not excl.get("included", True):
            lab["m2g_excluded"] = excl.get("reason", "excluded")
        doc["keys"][f"{cfg}|{arm}|{shape}|r{rnd}"] = lab
    cfgs = sorted({k[0] for k in keys})
    for cfg in cfgs:
        fl = dict(config_flags.get(cfg, {}))
        fl["frontend_saturated"] = any(lab["frontend_saturated_hi_cells"]
                                       for (c2, _a, _s, r), lab in keys.items() if c2 == cfg and r == 1)
        doc["configs"][cfg] = fl

    def defined(cfg, shape, num_arm, den_arm, rnd=1):
        reasons = []
        fl = doc["configs"].get(cfg, {})
        for flag in ("memory_mismatch", "gpu_contaminated"):
            if fl.get(flag):
                reasons.append(flag.upper())
        for arm in (num_arm, den_arm):
            lab = keys.get((cfg, arm, shape, rnd))
            if lab is None:
                reasons.append(f"{arm}:NOT_RUN")
                continue
            if lab.get("m2g_excluded"):
                reasons.append(f"{arm}:M2G_EXCLUDED({lab['m2g_excluded']})")
            if lab["knee"] != "KNEE_BRACKETED":
                reasons.append(f"{arm}:KNEE_NOT_BRACKETED[{'; '.join(lab['reasons']) or 'n/a'}]")
            if lab["harness_limited"]:
                reasons.append(f"{arm}:HARNESS_LIMITED")
            if lab["truncated"]:
                reasons.append(f"{arm}:TRUNCATED")
            if lab["lambda_star"] is None:
                reasons.append(f"{arm}:LAMBDA_STAR_UNRESOLVED")
        return (not reasons), reasons

    for name, (na, da) in PAIRS.items():
        for cfg in cfgs:
            for shape in ("A", "B"):
                ok, why = defined(cfg, shape, na, da)
                ent = {"defined": ok, "undefined_reasons": why}
                if ok:
                    num = keys[(cfg, na, shape, 1)]["achieved_by_seed"]
                    den = keys[(cfg, da, shape, 1)]["achieved_by_seed"]
                    p, lo, hi = paired_ratio_ci(num, den)
                    ent.update(point=p, ci=[lo, hi], num=num, den=den,
                               ratio_of_lambda_star=keys[(cfg, na, shape, 1)]["lambda_star"] /
                               keys[(cfg, da, shape, 1)]["lambda_star"])
                    ent["frontend_saturated_do_not_cite_as_scaling"] = bool(doc["configs"][cfg].get("frontend_saturated"))
                    pne = [a for a in (na, da) if keys[(cfg, a, shape, 1)].get("partition_not_exercised")]
                    if pne and a_is_pdmux(pne):
                        ent["PARTITION_NOT_EXERCISED"] = pne
                doc["estimands"].setdefault(name, {}).setdefault(cfg, {})[shape] = ent
    # trends: per quantity, mode, shape.  N=1 config is shared by TP and DP modes.
    for name in PAIRS:
        if name == "S":
            continue
        for mode in ("TP", "DP"):
            for shape in ("A", "B"):
                pts, undefined = [], []
                for cfg in ["N1"] + [f"{mode}{n}" for n in (2, 4, 8)]:
                    ent = doc["estimands"].get(name, {}).get(cfg, {}).get(shape)
                    n_gpus = 1 if cfg == "N1" else int(cfg[2:])
                    if ent and ent["defined"]:
                        pts.append((n_gpus, ent["num"], ent["den"]))
                    else:
                        undefined.append({"config": cfg, "reasons": (ent or {"undefined_reasons": ["NOT_RUN"]})["undefined_reasons"]})
                t = trend(pts)
                t["undefined_count"] = len(undefined)
                t["undefined"] = undefined          # MGP-34 / MGP-25: always printed beside the label
                t["primary"] = (name == PRIMARY)
                doc["trends"][f"{name}|{mode}|{shape}"] = t
    # P-DP (§4-4) and MGP-13
    broken = [k for k, t in doc["trends"].items()
              if k.startswith("G_sym|DP|") and t["label"] in ("G_INCREASES_WITH_N", "G_DECREASES_WITH_N")]
    doc["P_DP"] = {"broken": bool(broken), "shapes": broken,
                   "MGP13_TP_labels_not_citable": bool(broken)}
    # ANCHOR_DRIFT (§4-5): round 2 of N1 vs round 1
    a = {}
    for arm in ("fused", "fused_sym"):
        k1, k2 = ("N1", arm, "A", 1), ("N1", arm, "A", 2)
        if k1 in keys and k2 in keys and keys[k1]["lambda_star"] and keys[k2]["lambda_star"]:
            a[f"lambda_star_{arm}_A"] = {"r1": keys[k1]["lambda_star"], "r2": keys[k2]["lambda_star"],
                                         "rel": keys[k2]["lambda_star"] / keys[k1]["lambda_star"] - 1.0}
    g1 = _g_sym_round("N1", keys, 1)
    g2 = _g_sym_round("N1", keys, 2)
    if g1 and g2:
        a["G_sym_1_A"] = {"r1": g1, "r2": g2, "rel": g2 / g1 - 1.0}
    doc["anchor"] = {"checked": a, "ANCHOR_DRIFT": any(abs(v["rel"]) > C.ANCHOR_DRIFT_TOL for v in a.values()),
                     "G_sym_drift_checked": "G_sym_1_A" in a,
                     "note": ("G_sym(1) drift NOT checked (fused_sym/agnostic anchor missing) -- "
                              "'no ANCHOR_DRIFT' must not be read as 'no G_sym drift' (MGP-38)") if "G_sym_1_A" not in a else ""}
    # GPU heterogeneity (§4-6) and NUMA (descriptive)
    gd = [c for c in cells if c["meta"].get("block") == "gpu_dev" and c["status"] == "OK" and c.get("achieved")]
    if gd:
        v = [c["achieved"] for c in gd]
        spread = (max(v) - min(v)) / min(v)
        doc["blocks"]["gpu_dev"] = {"n_gpus": len(v), "spread": spread,
                                    "GPU_HETEROGENEOUS": spread > C.GPU_DEV_TOL,
                                    "by_gpu": {str(c["meta"].get("gpu")): c["achieved"] for c in gd}}
    nu = [c for c in cells if c["meta"].get("block") == "numa" and c["status"] == "OK" and c.get("achieved")]
    if nu:
        by = {}
        for c in nu:
            by.setdefault(c["meta"]["arm"], []).append(c["achieved"])
        if "fused" in by and "agnostic" in by:
            doc["blocks"]["numa"] = {"G_def_cross_numa_median": statistics.median(by["agnostic"]) / statistics.median(by["fused"]),
                                     "n_per_arm": {a: len(x) for a, x in by.items()},
                                     "note": "descriptive; compare with the main TP4 G_def (no judgement)"}
    return doc


def a_is_pdmux(arms):
    return any(a in C.PDMUX_ARMS for a in arms)


def _g_sym_round(cfg, keys, rnd):
    a, f = keys.get((cfg, "agnostic", "A", rnd)), keys.get((cfg, "fused_sym", "A", rnd))
    if a and f and a["lambda_star"] and f["lambda_star"]:
        return a["lambda_star"] / f["lambda_star"]
    return None


def render_text(doc):
    L = [f"# multi-GPU labels ({doc['prereg']})", ""]
    for k, t in sorted(doc["trends"].items()):
        mark = "PRIMARY" if t["primary"] else "auxiliary"
        s = f"{k:20s} [{mark}] {t['label']}"
        if "slope" in t:
            s += f" slope={t['slope']:+.4f} CI=[{t['ci'][0]:+.4f},{t['ci'][1]:+.4f}] Ns={t['Ns']} magnitude={t.get('magnitude')}"
        s += f"  UNDEFINED_CELLS={t['undefined_count']}"
        L.append(s)
        for u in t["undefined"]:
            L.append(f"      undefined {u['config']}: {'; '.join(u['reasons'])}")
    L.append("")
    L.append(f"P_DP broken={doc['P_DP']['broken']} (MGP-13 TP labels not citable={doc['P_DP']['MGP13_TP_labels_not_citable']})")
    L.append(f"ANCHOR_DRIFT={doc['anchor']['ANCHOR_DRIFT']} {doc['anchor'].get('note', '')}")
    for k, v in doc["blocks"].items():
        L.append(f"block {k}: {json.dumps(v, sort_keys=True)}")
    return "\n".join(L) + "\n"


# =========================================================================== CLI + selftest
def find_cells(root):
    return sorted(p.parent for p in Path(root).rglob("cell.json"))


def build_config_flags(root, cells):
    """Per config: memory_mismatch (G5(c)) from boot-level cell_meta.json, gpu_contaminated from contam.json."""
    by_cfg_arm = {}
    contaminated = set()
    for c in cells:
        m = c["meta"]
        boot = Path(root) / m["boot_dir"] if m.get("boot_dir") else None
        if not boot:
            continue
        if (boot / "contam.json").exists() and json.loads((boot / "contam.json").read_text()).get("gpu_contaminated"):
            contaminated.add(m["config"])
        cm = boot / "cell_meta.json"
        if cm.exists() and m.get("block", "main") == "main":
            f = json.loads(cm.read_text())
            si = f.get("server_info") or {}
            eff = f.get("max_running_requests_requested") or si.get("max_running_requests")
            by_cfg_arm.setdefault(m["config"], {})[m["arm"]] = dict(
                mamba=f.get("max_mamba_cache_size", []), max_total=f.get("max_total_num_tokens", []), eff_cap=eff)
    flags = {}
    for cfg, arms in by_cfg_arm.items():
        flags[cfg] = memory_check(arms)
        flags[cfg]["memory_mismatch"] = flags[cfg]["memory_mismatch"]
    for cfg in contaminated:
        flags.setdefault(cfg, {})["gpu_contaminated"] = True
    return flags


def selftest():
    fails = []
    ok, verdict, per = run_mutation_checks()
    for m, v in verdict.items():
        print(f"mutant {m:28s} killed={v['killed']} on {v['killed_on']}")
    if not ok:
        fails.append("mutation checks")
    # Q_c literal formula vs integral estimator on a tiny hand case
    reqs = [{"sent_s": 0.0, "ttft_ms": 2000.0, "error": ""}, {"sent_s": 1.0, "ttft_ms": 2000.0, "error": ""},
            {"sent_s": 3.0, "ttft_ms": None, "error": "x"}, {"sent_s": 3.0, "ttft_ms": 500.0, "error": "midstream"}]
    assert qc_value(reqs, 1.5) == 2 and qc_value(reqs, 2.5) == 1 and qc_value(reqs, 3.2) == 1
    T, means = qc_thirds(reqs)
    assert abs(T - 3.0) < 1e-9 and abs(means[0] - 1.0) < 1e-9 and abs(means[1] - 2.0) < 1e-9 and abs(means[2] - 1.0) < 1e-9, (T, means)
    # bootstrap determinism + CI contains point for a clean ratio
    p, lo, hi = paired_ratio_ci([2.0, 2.1, 1.9, 2.0], [1.0, 1.0, 1.0, 1.0])
    assert lo <= p <= hi and abs(p - 2.0) < 0.06
    t = trend([(1, [2, 2, 2, 2], [1, 1, 1, 1]), (2, [2, 2, 2, 2], [1, 1, 1, 1]), (4, [2, 2, 2, 2], [1, 1, 1, 1])])
    assert t["label"] == "NO_TREND_DETECTED" and t["magnitude"] == "DIRECTION_NOT_CITABLE_UNDER_3PCT"
    t = trend([(1, [1, 1, 1, 1.01], [1, 1, 1, 1]), (2, [1.5, 1.5, 1.5, 1.5], [1, 1, 1, 1]),
               (4, [2, 2, 2, 2.02], [1, 1, 1, 1])])
    assert t["label"] == "G_INCREASES_WITH_N", t
    assert trend([(2, [1] * 4, [1] * 4), (4, [1] * 4, [1] * 4)])["label"] == "TREND_UNDEFINED"
    assert trend([(2, [1] * 4, [1] * 4), (4, [1] * 4, [1] * 4), (8, [1] * 4, [1] * 4)])["magnitude"] == "MAGNITUDE_UNDEFINED"
    print("selftest:", "PASS" if not fails else f"FAIL {fails}")
    return 0 if not fails else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", help="directory scanned recursively for cell.json")
    ap.add_argument("--pass-table", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not a.root:
        ap.error("--root or --selftest")
    cells = [load_cell(d) for d in find_cells(a.root)]
    pt = json.loads(Path(a.pass_table).read_text()) if a.pass_table else {}
    doc = aggregate(cells, pt.get("cells") if isinstance(pt, dict) and "cells" in pt else pt,
                    build_config_flags(a.root, cells))
    doc["n_cells"] = len(cells)
    if a.out:
        Path(a.out).write_text(json.dumps(doc, indent=1, sort_keys=True, default=float) + "\n")
    sys.stdout.write(render_text(doc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
