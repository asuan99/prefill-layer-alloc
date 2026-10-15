#!/usr/bin/env python3
"""Label / estimand code of PREREG_MULTIGPU_2026-10-12.md (rev10) §4-3…§4-6, §15-3, §16.  CPU only, stdlib+numpy.

  per cell   Q_c(t), g, CLIENT_ERROR, SATURATION_UNRESOLVED, send-lag p99, cgroup throttle ratio,
             HARNESS_LIMITED, FRONTEND_SATURATED (+ per-thread max, MGP-30), achieved throughput
  per key    (config, arm, shape): HI_SATURATED / LO_UNSATURATED / KNEE_BRACKETED / LADDER_TOO_LOW,
             lambda* = median of the 4 HI achieved values, PARTITION_NOT_EXERCISED
  per config MEMORY_MISMATCH (G5(c)), GPU_CONTAMINATED, FRONTEND_SATURATED, M2g exclusions
  estimands  G_sym (primary) = lambda*_agnostic / lambda*_fused_sym ; G_def, G_d44,sym, S (auxiliary)
             seed-paired bootstrap CI (10000, seed 1), trend slope of log G on log2 N (seed index
             resampled JOINTLY over all N and both arms), magnitude rule, GPU_HETEROGENEOUS, P-DP check,
             undefined-cell count + reasons beside every trend (MGP-34/25)
  rev9       ONE Job = ONE N.  `aggregate` labels one Job's cells (keys, estimands, N=1 anchor summary); `aggregate_jobs`
             joins the four Jobs: JOB_EFFECT check (anchor vs J1 boot1, §15-3), corrected trend, P9-a/b/c, P-DP.

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

HERE = Path(os.path.abspath(__file__)).parent
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
    """(config, arm, shape, round).  A J_N N=1 anchor boot (block == "anchor") is config ANCHOR_CONFIG: it must never merge with, or
    become, the N=1 point of a trend (the N=1 point is J1's)."""
    cfg = C.ANCHOR_CONFIG if m.get("block", "main") == "anchor" else m["config"]
    return (cfg, m["arm"], m["shape"], int(m.get("round", 1)))


def _et(c):
    """E-time value of a cell: mg_realized_mix.cell_e_time writes `e_time` (older drafts `etime`)."""
    e = c.get("etime") or {}
    return e.get("e_time", e.get("etime"))


def _gaps_of(hi, lo):
    """F13: measurements that did NOT happen are listed, never read as 'false'.  throttle/send-lag over HI+LO cells (they feed
    HARNESS_LIMITED), FRONTEND (sampler rows, registered roles) over HI cells (the registered FRONTEND_SATURATED window)."""
    allc = hi + lo
    gaps = []
    n_thr = sum(1 for c in allc if c.get("throttle_unmeasured"))
    if n_thr:
        gaps.append(f"THROTTLE_UNMEASURED({n_thr}/{len(allc)} cells)")
    n_lag = sum(1 for c in allc if c.get("send_lag_p99_ms") is None)
    if n_lag:
        gaps.append(f"SENDLAG_UNMEASURED({n_lag}/{len(allc)} cells)")
    n_fe = sum(1 for c in hi if c.get("frontend_unmeasured"))
    if n_fe:
        gaps.append(f"FRONTEND_UNMEASURED({n_fe}/{len(hi)} HI cells)")
    missing = set()
    for c in hi:
        fe = c.get("frontend")
        if not fe:
            continue
        need = C.REQUIRED_FRONTEND_ROLES.get(str((c.get("meta") or {}).get("mode", "N1")).upper(), ())
        missing |= {r for r in need if r not in fe.get("roles", {})}
    for r in sorted(missing):
        gaps.append(f"FRONTEND_ROLE_MISSING:{r}")
    return gaps


def key_labels(cells, ext_fired=False):
    """cells: list of analyzed-cell records of ONE key.  Final HI set = the HI cells with the largest `ext`.
    ext_fired (F10): the JOINT extension fired for this (config, shape, round) -- some arm triggered it or ran an extension.
    A key that has NO extension HI cells then is HI_NOT_JUDGED(EXT_NOT_RUN): its first-pass HI set is not a judged ladder and
    its lambda* is undefined (the first-pass verdict of a not-extended arm must not be paired with extended arms)."""
    hi_all = [c for c in cells if c["meta"]["rung"] == "HI"]
    lo_all = [c for c in cells if c["meta"]["rung"] == "LO"]
    ext = max((int(c["meta"].get("ext", 0)) for c in hi_all), default=0)
    hi = [c for c in hi_all if int(c["meta"].get("ext", 0)) == ext]
    lo_ext = max((int(c["meta"].get("ext", 0)) for c in lo_all), default=0)
    lo = [c for c in lo_all if int(c["meta"].get("ext", 0)) == lo_ext]
    out = {"ext": ext, "max_ext": ext, "n_hi": len(hi), "n_lo": len(lo), "reasons": []}
    ext_missing = bool(ext_fired and ext == 0)
    out["ext_not_run"] = ext_missing
    out["ext0_trigger"] = any(c["status"] == "OK" and c["g"] is not None and c["g"] < C.G_HI
                              for c in hi_all if int(c["meta"].get("ext", 0)) == 0)
    byseed = {int(c["meta"]["seed"]): c for c in hi}
    ok_hi = [byseed.get(s) for s in C.HI_SEEDS]
    bad = [s for s, c in zip(C.HI_SEEDS, ok_hi) if c is None or c["status"] != "OK"]
    out["hi_bad_seeds"] = {int(s): (None if byseed.get(s) is None else byseed[s]["status"]) for s in bad}
    out["g_hi"] = {int(s): (byseed[s]["g"] if s in byseed else None) for s in C.HI_SEEDS}
    # rev9 §15-3 / §16-1: the Job-effect REFERENCE is the J1 boot1 value at the 1.6x rung (ext == 0), never the extension rung.
    by0 = {int(c["meta"]["seed"]): c for c in hi_all if int(c["meta"].get("ext", 0)) == 0}
    ach0 = [by0[s]["achieved"] if (s in by0 and by0[s]["status"] == "OK") else None for s in C.HI_SEEDS]
    out["boot1_achieved_by_seed"] = ach0
    out["boot1_lambda_star"] = statistics.median(ach0) if all(a is not None for a in ach0) else None
    if any(a is None for a in ach0):
        out["boot1_hi_label"] = "HI_NOT_JUDGED"
    else:
        out["boot1_hi_label"] = "HI_SATURATED" if all(by0[s]["g"] is not None and by0[s]["g"] >= C.G_HI for s in C.HI_SEEDS) else "HI_NOT_SATURATED"
    out["client_error_hi"] = any(c["status"] == "CLIENT_ERROR" for c in hi)
    # rev11 M-2: the Job-effect reference flags are read from the SAME cells as boot1_lambda_star / boot1_hi_label -- the ext-0 HI cells --
    # never from the final (extension-rung) set, and never from LO cells.
    hi0 = [c for c in hi_all if int(c["meta"].get("ext", 0)) == 0]
    out["boot1_harness_limited"] = any(c.get("harness_limited") for c in hi0)
    out["boot1_client_error"] = any(c["status"] == "CLIENT_ERROR" for c in hi0)
    if ext_missing:
        out["hi_label"] = "HI_NOT_JUDGED"
        out["reasons"].append("HI_NOT_JUDGED(EXT_NOT_RUN)")
    elif bad:
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
    if all(a is not None for a in ach) and not ext_missing:
        out["lambda_star"] = statistics.median(ach)
        out["lambda_star_range"] = [min(ach), max(ach)]
    else:
        out["lambda_star"] = None
        out["reasons"].append("LAMBDA_STAR_UNRESOLVED_EXT_NOT_RUN" if ext_missing else "LAMBDA_STAR_UNRESOLVED_N_BELOW_4")
    allc = hi + lo
    out["harness_limited"] = any(c.get("harness_limited") for c in allc)
    out["truncated"] = any(c.get("truncated") for c in allc)
    out["throttle_unmeasured_cells"] = sum(1 for c in allc if c.get("throttle_unmeasured"))
    # §16-3 covariate: the per-cell cgroup throttle ratio (dnr_throttled / dnr_periods) is printed beside every key
    thr = {c["cell_id"]: c.get("throttle_ratio") for c in allc if c.get("throttle_ratio") is not None}
    out["throttle_ratio_cells"] = thr
    out["throttle_ratio_max"] = max(thr.values()) if thr else None
    out["throttle_ratio_median"] = statistics.median(thr.values()) if thr else None
    out["frontend_saturated_hi_cells"] = [c["cell_id"] for c in hi if c.get("frontend_saturated")]
    out["measurement_gaps"] = _gaps_of(hi, lo)                                       # F13
    # ---- E-time / partition (F12).  Only PD-mux arms have an E-time; MIX_FAILED / missing E-time = E_TIME_UNDEFINED (MGH-12:
    # "분할이 실행됐는지 미정"), carried to the estimand items -- it is neither 'exercised' nor 'not exercised'.
    arm = (cells[0]["meta"].get("arm") if cells else None)
    ets = [_et(c) for c in hi if c.get("etime") and c["etime"].get("status") == "OK" and _et(c) is not None]
    mix_failed = any(c.get("etime") and c["etime"].get("status") == "MIX_FAILED" for c in hi)
    if mix_failed:
        out["mix_failed"] = True
    if arm in C.PDMUX_ARMS:
        if len(hi) == len(C.HI_SEEDS) and len(ets) == len(hi) and not mix_failed:
            out["partition"] = "PARTITION_NOT_EXERCISED" if statistics.median(ets) < C.PARTITION_ETIME_MIN else "PARTITION_EXERCISED"
        else:
            out["partition"] = "E_TIME_UNDEFINED"
    else:
        out["partition"] = "NOT_APPLICABLE"
    if out["partition"] == "PARTITION_NOT_EXERCISED":
        out["partition_not_exercised"] = True
    out["etime_median"] = statistics.median(ets) if ets else None
    return out


# =========================================================================== G5(c) memory + contamination (config level)
def memory_check(arm_facts, shapes=("A", "B"), _mut=None):
    """arm_facts: arm -> dict(mamba=[...], max_total=[...], eff_cap=int, mode, N).  PREREG §3 G5(c) (i)-(ii) (rev9 §16-2 item 1, rev10 H-5).
    Rule (iii) (arm-to-arm max_total_num_tokens relative difference > 1 %) was RETIRED by user decision 2026-10-13 -- a POST-HOC change made
    after the M2g round-2 values were seen.  It is no longer a MEMORY_MISMATCH reason.  The arm-to-arm relative difference is still recorded
    under BOTH denominators (fused_sym; the minimum arm) and `MEMORY_RULE_III_COUNTERFACTUAL` says, per denominator, whether rule (iii) would
    have flagged the config, so the result document can disclose the cells it would have made undefined.
    `_mut` (tests only): 'iii' = rule (iii) still present; 'ii' = rule (ii) removed."""
    reasons = []
    mt_min = {}
    for arm, f in arm_facts.items():
        if any(x != f["eff_cap"] for x in f.get("mamba", [])) or not f.get("mamba"):
            reasons.append(f"{arm}: mamba cache slots {f.get('mamba')} != effective cap {f['eff_cap']}")
        need = max(f["eff_cap"] * sum(C.SHAPE_IO[s]) for s in shapes)
        if _mut != "ii" and (not f.get("max_total") or min(f["max_total"]) < need):
            reasons.append(f"{arm}: max_total_num_tokens {f.get('max_total')} < cap*(ISL+OSL)={need}")
        if f.get("max_total"):
            mt_min[arm] = min(f["max_total"])
    diff = {}
    if len(mt_min) >= 2:
        if "fused_sym" in mt_min:
            ref = mt_min["fused_sym"]
            diff["fused_sym"] = max(abs(v - ref) for v in mt_min.values()) / ref
        lo = min(mt_min.values())
        diff["min_arm"] = (max(mt_min.values()) - lo) / lo
    cf = {f"denominator_{k}": v > C.MEMORY_REL_DESCRIPTIVE for k, v in diff.items()}
    if _mut == "iii" and any(cf.values()):
        reasons.append(f"arm max_total_num_tokens relative difference {diff} > {C.MEMORY_REL_DESCRIPTIVE}")
    return {"memory_mismatch": bool(reasons), "reasons": reasons, "arm_rel_diff": diff,
            "MEMORY_RULE_III_COUNTERFACTUAL": cf, "rule_iii_would_have_flagged": any(cf.values())}


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


G1_UNDEFINED = "G1_UNDEFINED_NO_JOB_CHECK"       # P9-a
TREND_JOB_SENSITIVE = "TREND_JOB_SENSITIVE"       # P9-b
DIRECTIONAL = ("G_INCREASES_WITH_N", "G_DECREASES_WITH_N")


def _slope_stats(Ns, logg, point_logg):
    """slope of log G on log2 N + percentile CI over the joint seed-index bootstrap (logg: reps x nN) + the magnitude rule
    G(N_max)/G(1) (needs N=1: the caller guarantees it, P9-a)."""
    x = np.log2(Ns)
    xc = x - x.mean()
    denom = float((xc ** 2).sum())
    slopes = (logg * xc).sum(axis=1) / denom
    s = float((point_logg * xc).sum() / denom)
    lo, hi = (float(v) for v in np.percentile(slopes, C.CI_PCT))
    label = "NO_TREND_DETECTED" if lo <= 0.0 <= hi else ("G_INCREASES_WITH_N" if lo > 0 else "G_DECREASES_WITH_N")
    out = {"label": label, "slope": s, "ci": [lo, hi], "n_points": len(Ns), "Ns": [int(n) for n in Ns],
           "G_by_N": {int(n): float(np.exp(v)) for n, v in zip(Ns, point_logg)}}
    if int(Ns[0]) == 1:
        ratio = float(np.exp(point_logg[-1] - point_logg[0]))
        rb = np.exp(logg[:, -1] - logg[:, 0])
        out.update(N_max=int(Ns[-1]), magnitude_ratio=ratio, magnitude_ci=[float(v) for v in np.percentile(rb, C.CI_PCT)],
                   direction_citable=abs(ratio - 1.0) >= C.MAGNITUDE_BAND)
        out["magnitude"] = "OK" if out["direction_citable"] else "DIRECTION_NOT_CITABLE_UNDER_3PCT"
    else:
        out.update(magnitude="MAGNITUDE_UNDEFINED", direction_citable=None)
    return out


def trend(points, reps=None, anchors=None, job_flags=None, ref_g1=None, _mut=None):
    """points: list of (N, num[4], den[4]).  slope of log G on log2 N, SEED INDEX resampled jointly over all N and both arms (MGP-10).

    rev9/rev10:
      P9-a  no N=1 point (G(1) undefined) -> label G1_UNDEFINED_NO_JOB_CHECK, direction never citable (the rev8 'MAGNITUDE_UNDEFINED, direction
            label only' sentence is retired).
      P9-b  `job_flags` {N: FIRED|CLEAR|UNTESTED...} (JOB_EFFECT of the Job that measured N), `anchors` {N: (anchor_num[4], anchor_den[4])}
            (that SAME Job's own N=1 anchor, shape A; only Jobs whose anchor is usable are present) and `ref_g1` (J1 boot1 agnostic / fused_sym
            achieved arrays, the SAME 1.6x rung as the anchors).  If any JOB_EFFECT FIRED the corrected trend is computed with the factor
            G_ref(boot1) / G_anchor(J_N), the seed index resampled JOINTLY over trend points, reference and anchor arrays.  The direction is
            citable only if the RAW and the CORRECTED trend both pass the 3 % magnitude rule AND carry the same label, else TREND_JOB_SENSITIVE.
      H-4   points of a J_N whose anchor is UNTESTED are EXCLUDED from the corrected trend; with fewer than 3 corrected points it is
            TREND_UNDEFINED and P9-b fails (TREND_JOB_SENSITIVE).
    `_mut` (tests only): 'i' = keep the retired G(1)-undefined path; 'iv' = drop the magnitude rule from the corrected trend;
    'h3' = correction ratio from the extension-rung G(1) instead of the boot1 reference; 'h4' = raw value substituted for excluded points."""
    reps = reps or C.BOOTSTRAP_REPS
    pts = sorted(points, key=lambda p: p[0])
    if len(pts) < C.TREND_MIN_POINTS:
        return {"label": "TREND_UNDEFINED", "n_points": len(pts), "Ns": [p[0] for p in pts], "direction_citable": False}
    Ns = [p[0] for p in pts]
    nums = np.array([p[1] for p in pts], float)     # (nN, 4)
    dens = np.array([p[2] for p in pts], float)
    k = nums.shape[1]
    rng = np.random.default_rng(C.BOOTSTRAP_SEED)
    idx = rng.integers(0, k, size=(reps, k))
    logg = np.stack([np.log(np.median(nums[i][idx], axis=1) / np.median(dens[i][idx], axis=1)) for i in range(len(pts))], axis=1)
    point_logg = np.log(np.median(nums, axis=1) / np.median(dens, axis=1))
    raw = _slope_stats(Ns, logg, point_logg)
    if Ns[0] != 1:
        if _mut == "i":                                              # the retired rev8 rule: direction label only
            return dict(raw, direction_citable=None, magnitude="MAGNITUDE_UNDEFINED")
        return dict(raw, label=G1_UNDEFINED, raw_label=raw["label"], direction_citable=False, magnitude=G1_UNDEFINED)
    flags = job_flags or {}
    anchors = anchors or {}
    fired = sorted(n for n in Ns if n >= 2 and flags.get(n) == "FIRED")
    out = dict(raw, raw_label=raw["label"], job_effect_fired_N=fired, corrected=None)
    if not fired:
        return out
    if ref_g1 is None and _mut != "h3":
        return dict(out, label=TREND_JOB_SENSITIVE, direction_citable=False,
                    job_sensitive_reason="CORRECTION_UNDEFINED(J1 boot1 reference arrays absent)")
    if _mut == "h3" or ref_g1 is None:
        log_ref, point_ref = logg[:, 0], point_logg[0]            # mutant: the extension-rung G(1) (what the trend uses for N=1)
    else:
        rn, rd = (np.asarray(v, float) for v in ref_g1)
        log_ref, point_ref = np.log(np.median(rn[idx], axis=1) / np.median(rd[idx], axis=1)), float(np.log(np.median(rn) / np.median(rd)))
    keep = [i for i, n in enumerate(Ns) if n == 1 or n in anchors] if _mut != "h4" else list(range(len(Ns)))
    cl, cpl = logg.copy(), point_logg.copy()
    for i, n in enumerate(Ns):
        if n >= 2 and n in anchors:
            an, ad = (np.asarray(v, float) for v in anchors[n])
            cl[:, i] = logg[:, i] + log_ref - np.log(np.median(an[idx], axis=1) / np.median(ad[idx], axis=1))
            cpl[i] = point_logg[i] + point_ref - np.log(np.median(an) / np.median(ad))
    excluded = [n for i, n in enumerate(Ns) if i not in keep]
    if len(keep) < C.TREND_MIN_POINTS:
        corr = {"label": "TREND_UNDEFINED", "n_points": len(keep), "Ns": [Ns[i] for i in keep], "direction_citable": False}
        return dict(out, corrected=corr, label=TREND_JOB_SENSITIVE, direction_citable=False,
                    job_sensitive_reason=f"CORRECTED_TREND_UNDEFINED(<{C.TREND_MIN_POINTS} points; excluded anchor-UNTESTED N={excluded})")
    corr = _slope_stats([Ns[i] for i in keep], cl[:, keep], cpl[keep])
    corr["excluded_N"] = excluded
    out["corrected"] = corr
    if _mut == "iv":                                                 # mutant: corrected trend not held to the magnitude rule
        ok = bool(raw["direction_citable"]) and raw["label"] == corr["label"]
    else:
        ok = bool(raw["direction_citable"]) and bool(corr["direction_citable"]) and raw["label"] == corr["label"]
    if ok:
        return dict(out, direction_citable=True)
    why = ("RAW_AND_CORRECTED_LABELS_DIFFER" if raw["label"] != corr["label"] else "MAGNITUDE_RULE_FAILED(" +
           ",".join(n for n, d in (("raw", raw), ("corrected", corr)) if not d["direction_citable"]) + ")")
    return dict(out, label=TREND_JOB_SENSITIVE, direction_citable=False, job_sensitive_reason=why)


def p_dp_status(trends, _mut=None):
    """rev10 P10 / H-1: P-DP is evaluated PER SHAPE and combined BROKEN > UNTESTED > HELD (any BROKEN fires MGP-13; another shape's UNTESTED
    cannot suppress it).  Shape s is BROKEN iff its G_sym DP trend is defined, G_sym(1, s) is defined and the label is directional
    (G_INCREASES/G_DECREASES -- for the ANCHOR shape A a TREND_JOB_SENSITIVE label is not; for shape B the RAW label decides, rev11 §18).  UNTESTED (P9-c) = TREND_JOB_SENSITIVE / TREND_UNDEFINED /
    G1_UNDEFINED_NO_JOB_CHECK / G(1) undefined / no DP trend.  HELD = NO_TREND_DETECTED.  UNTESTED is not 'not broken'; it does NOT fire MGP-13
    and obliges the 'P-DP 미검정' disclosure on every TP label.   `_mut` 'untested_first' (tests only): UNTESTED outranks BROKEN; 'same_shape_only' = the rev10 reading (cross-shape correction lowers shape B to UNTESTED);
    'raw_unconditional' = the rev11 fail-open (D-A: a raw NO_TREND_DETECTED replaces TREND_JOB_SENSITIVE -> HELD)."""
    dp = {k: t for k, t in trends.items() if k.startswith("G_sym|DP|")}
    per, via = {}, {}
    for k, t in dp.items():
        lab = t["label"]
        g1 = t.get("g1_defined", True)
        shape = k.rsplit("|", 1)[1]
        # rev11 §18 (M-4, user decision, fail-closed): the cross-shape correction (shape-A anchors, G_sym) is NOT a same-shape correction.
        # For a shape other than the anchor's, P-DP reads the RAW DP trend: defined + G(1, shape) defined + directional raw label -> BROKEN,
        # even when the corrected trend turned the published label into TREND_JOB_SENSITIVE.  The anchor shape keeps its same-shape correction.
        # rev11 D-A (harness delta audit 2026-10-13): the raw label is substituted ONLY when it is directional (it can only RAISE UNTESTED to
        # BROKEN).  A raw NO_TREND_DETECTED / TREND_UNDEFINED under a TREND_JOB_SENSITIVE published label stays UNTESTED (P9-c), it is not HELD.
        # `raw_label_used` is recorded only when the substitution promoted the shape to BROKEN.
        if shape != C.ANCHOR_SHAPE and _mut != "same_shape_only" and lab == TREND_JOB_SENSITIVE:
            raw = t.get("raw_label", lab)
            if _mut == "raw_unconditional":                       # mutant (D-A): the rev11 fail-open -- raw label whatever it is
                lab = raw
            elif raw in DIRECTIONAL:
                lab = raw
            if lab in DIRECTIONAL and g1 and lab != t["label"]:
                via[k] = "RAW_LABEL(cross-shape correction does not lower BROKEN)"
        if lab in DIRECTIONAL and g1:
            per[k] = "BROKEN"
        elif lab == "NO_TREND_DETECTED" and g1:
            per[k] = "HELD"
        else:
            per[k] = "UNTESTED"
    vals = set(per.values())
    if not per:
        status = "UNTESTED"
    elif _mut == "untested_first":
        status = "UNTESTED" if "UNTESTED" in vals else ("BROKEN" if "BROKEN" in vals else "HELD")
    elif "BROKEN" in vals:
        status = "BROKEN"
    elif "UNTESTED" in vals:
        status = "UNTESTED"
    else:
        status = "HELD"
    labels = {k: t["label"] for k, t in dp.items()}
    why = sorted(k for k, v in per.items() if v == "UNTESTED")
    return {"status": status, "broken": status == "BROKEN", "per_shape": per, "shapes": sorted(k for k, v in per.items() if v == "BROKEN"),
            "labels": labels, "raw_label_used": via, "untested_because": why, "MGP13_fired": status == "BROKEN",
            "MGP13_TP_labels_not_citable": status == "BROKEN",
            "TP_labels_need_P_DP_UNTESTED_disclosure": status == "UNTESTED"}


def aggregate(cells, pass_table=None, config_flags=None, m2g_configs=None, job=None):
    """ONE Job's labels (rev9: one Job = one N).  cells: analyzed-cell records (load_cell).  pass_table: M2G table `cells` dict.  config_flags: build_config_flags output.
    m2g_configs: M2G table `configs` dict (F14: TP_NUMERIC_DIVERGENT, G2_NOT_TESTED, <arm>:G1_UNINFORMATIVE, MEMORY_MISMATCH)."""
    pass_table = pass_table or {}
    config_flags = config_flags or {}
    m2g_configs = m2g_configs or {}
    groups = {}
    for c in cells:
        m = c["meta"]
        if m.get("block", "main") in ("main", "anchor") and m.get("rung") in ("HI", "LO"):
            groups.setdefault(_key(m), []).append(c)      # gpu_dev / numa / nsys / m2g blocks are descriptive
    first = {k: key_labels(cs) for k, cs in groups.items()}
    # F10: the extension is JOINT over the four arms of a (config, shape, round); one arm triggering (or running) it makes
    # every arm of that group without extension cells HI_NOT_JUDGED(EXT_NOT_RUN).
    fired = {(k[0], k[2], k[3]) for k, lab in first.items() if k[0] != C.ANCHOR_CONFIG and (lab["ext0_trigger"] or lab["max_ext"] >= 1)}   # an anchor never extends
    keys = {k: key_labels(cs, ext_fired=((k[0], k[2], k[3]) in fired)) for k, cs in groups.items()}
    doc = {"prereg": f"{C.PREREG} rev{C.PREREG_REV}", "job": job, "keys": {}, "configs": {}, "estimands": {}, "anchor": {}, "ref": {},
           "blocks": {}}
    for (cfg, arm, shape, rnd), lab in sorted(keys.items()):
        excl = pass_table.get(f"{'N1' if cfg == C.ANCHOR_CONFIG else cfg}:{arm}")
        if excl is not None and not excl.get("included", True):
            lab["m2g_excluded"] = excl.get("reason", "excluded")
        doc["keys"][f"{cfg}|{arm}|{shape}|r{rnd}"] = lab
    cfgs = sorted({k[0] for k in keys if k[0] != C.ANCHOR_CONFIG})
    for cfg in cfgs:
        fl = dict(config_flags.get(cfg, {}))
        fl["frontend_saturated"] = any(lab["frontend_saturated_hi_cells"]
                                       for (c2, _a, _s, r), lab in keys.items() if c2 == cfg and r == 1)
        mc = m2g_configs.get(cfg) or {}
        mflags = list(mc.get("flags", []))
        fl["m2g_flags"] = mflags
        fl["tp_numeric_divergent"] = bool(mc.get("tp_numeric_divergent")) or "TP_NUMERIC_DIVERGENT" in mflags
        fl["g2_not_tested"] = "G2_NOT_TESTED" in mflags
        fl["g1_uninformative_arms"] = sorted(f.split(":")[0] for f in mflags if f.endswith(":G1_UNINFORMATIVE"))
        fl["m2g_memory_mismatch"] = bool(mc.get("memory_mismatch")) or any(f.startswith("MEMORY_MISMATCH") for f in mflags)
        if fl["m2g_memory_mismatch"]:
            fl["memory_mismatch"] = True                    # M2g G5(c) FAIL / NOT_TESTED carries into M3 (OR, never overwritten)
        mc_iii = bool(mc.get("rule_iii_would_have_flagged"))
        fl["rule_iii_would_have_flagged"] = bool(fl.get("rule_iii_would_have_flagged")) or mc_iii       # rule (iii) retired, disclosed
        if mc.get("MEMORY_RULE_III_COUNTERFACTUAL") and not fl.get("MEMORY_RULE_III_COUNTERFACTUAL"):
            fl["MEMORY_RULE_III_COUNTERFACTUAL"], fl["arm_rel_diff"] = mc["MEMORY_RULE_III_COUNTERFACTUAL"], mc.get("arm_rel_diff")
        doc["configs"][cfg] = fl
    doc["memory_rule_iii"] = {
        "status": "RETIRED_POST_HOC (user decision 2026-10-13, after the M2g round-2 max_total_num_tokens values were seen)",
        "configs_iii_would_have_made_undefined": sorted(c for c, f in doc["configs"].items() if f.get("rule_iii_would_have_flagged")),
        "MEMORY_RULE_III_COUNTERFACTUAL_by_config": {c: {"flags": f.get("MEMORY_RULE_III_COUNTERFACTUAL"), "arm_rel_diff": f.get("arm_rel_diff")}
                                                     for c, f in doc["configs"].items() if f.get("MEMORY_RULE_III_COUNTERFACTUAL")}}

    def disclosures(cfg, arms):
        fl = doc["configs"].get(cfg, {})
        d = []
        if fl.get("tp_numeric_divergent"):
            d.append("TP_NUMERIC_DIVERGENT")
        if fl.get("g2_not_tested"):
            d.append("G2_NOT_TESTED")
        d += [f"{a}:G1_UNINFORMATIVE" for a in fl.get("g1_uninformative_arms", []) if a in arms]
        if fl.get("contam_unverified"):
            d.append("CONTAM_UNVERIFIED")
        return d

    def defined(cfg, shape, num_arm, den_arm, rnd=1):
        reasons = []
        fl = doc["configs"].get(cfg, {})
        if fl.get("memory_mismatch"):
            reasons.append("MEMORY_MISMATCH")
        if fl.get("gpu_contaminated"):
            # R4: scope = the pairs that contain a contaminated boot's arm.  Flags without `contam_arms` (older layout) keep the
            # conservative whole-config reading.
            ca = fl.get("contam_arms")
            if ca is None:
                reasons.append("GPU_CONTAMINATED")
            else:
                reasons += [f"{a}:GPU_CONTAMINATED" for a in (num_arm, den_arm) if a in ca]
        for arm in (num_arm, den_arm):                     # R3: a pair arm whose memory G5(c) was not tested
            if arm in (fl.get("memory_not_tested_arms") or []):
                reasons.append(f"{arm}:MEMORY_NOT_TESTED")
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
                    # F12: the partition status of every PD-mux arm of the pair -- E_TIME_UNDEFINED / MIX_FAILED are carried, not dropped
                    ent["partition"] = {a: keys[(cfg, a, shape, 1)]["partition"] for a in (na, da)
                                        if keys[(cfg, a, shape, 1)]["partition"] != "NOT_APPLICABLE"}
                    ent["mix_failed_arms"] = [a for a in (na, da) if keys[(cfg, a, shape, 1)].get("mix_failed")]
                    ent["partition_exercised_unknown"] = any(v == "E_TIME_UNDEFINED" for v in ent["partition"].values())
                    # F13: measurements that did not happen (throttle / send-lag / FRONTEND / roles) -- explicit, per arm
                    ent["measurement_gaps"] = {a: keys[(cfg, a, shape, 1)]["measurement_gaps"] for a in (na, da)
                                               if keys[(cfg, a, shape, 1)]["measurement_gaps"]}
                    ent["disclosures"] = disclosures(cfg, (na, da))                       # F14
                doc["estimands"].setdefault(name, {}).setdefault(cfg, {})[shape] = ent
    # ---- rev9: the Job-effect reference (J1 boot1, ext 0) and the N=1 anchor (J2/J4/J8) as plain facts of THIS Job; aggregate_jobs joins them.
    doc["ref"] = {arm: ref_entry(keys.get(("N1", arm, C.ANCHOR_SHAPE, 1))) for arm in C.ANCHOR_ARMS} if any(k[0] == "N1" for k in keys) else {}
    afl = ((config_flags.get("N1") or {}).get("by_group") or {}).get("r1/anchor") or {}
    doc["anchor"] = {"arms": {arm: anchor_entry(keys.get((C.ANCHOR_CONFIG, arm, C.ANCHOR_SHAPE, 1))) for arm in C.ANCHOR_ARMS},
                     "gpu_contaminated_arms": afl.get("contam_arms", []), "contam_unverified": bool(afl.get("contam_unverified"))} \
        if any(k[0] == C.ANCHOR_CONFIG for k in keys) else {}
    # GPU heterogeneity (§4-6, J8 only; descriptive under 8 concurrent servers, no outcome branch -- MGP-47)
    gd = [c for c in cells if c["meta"].get("block") == "gpu_dev" and c["status"] == "OK" and c.get("achieved")]
    if gd:
        v = [c["achieved"] for c in gd]
        spread = (max(v) - min(v)) / min(v)
        doc["blocks"]["gpu_dev"] = {"n_gpus": len(v), "spread": spread,
                                    "GPU_HETEROGENEOUS": spread > C.GPU_DEV_TOL,
                                    "by_gpu": {str(c["meta"].get("gpu")): c["achieved"] for c in gd},
                                    "note": "descriptive, measured with 8 servers running concurrently; no result branch in rev9 (MGP-47); "
                                            "the GPU 0 anchor is NOT this node's N=1 representative"}
    return doc


def ref_entry(lab):
    """What the Job-effect check reads from a J1 key: the boot1 (ext 0, 1.6x rung) value.  `final_*` is the extension-rung value that the
    trend uses -- it is carried only so that a test can show the check never reads it (mutant ii)."""
    if lab is None:
        return {"present": False}
    return {"present": True, "boot1_lambda_star": lab.get("boot1_lambda_star"), "boot1_hi_label": lab.get("boot1_hi_label"),
            "boot1_harness_limited": bool(lab.get("boot1_harness_limited")), "boot1_client_error": bool(lab.get("boot1_client_error")),
            "harness_limited": bool(lab.get("harness_limited")), "client_error": bool(lab.get("client_error_hi")),    # FINAL set: carried only so a test can show the check never reads it
            "final_lambda_star": lab.get("lambda_star"), "final_ext": lab.get("ext"),
            "boot1_achieved_by_seed": lab.get("boot1_achieved_by_seed")}


def anchor_entry(lab):
    """What the Job-effect check reads from a J_N anchor key (shape A, HI, 4 seeds, ext 0)."""
    if lab is None:
        return {"present": False}
    return {"present": True, "lambda_star": lab.get("lambda_star"), "hi_label": lab.get("hi_label"),
            "achieved_by_seed": lab.get("achieved_by_seed"), "harness_limited": bool(lab.get("harness_limited")),
            "client_error": bool(lab.get("client_error_hi")), "truncated": bool(lab.get("truncated")),
            "throttle_ratio_max": lab.get("throttle_ratio_max"), "ext": lab.get("ext")}


# =========================================================================== rev9: Job effect (§15-3, §16-1) and the cross-Job join
def job_effect(ref, anc, tol=None, _mut=None):
    """JOB_EFFECT(J_N) status from J1's boot1 reference `ref` (doc["ref"] of J1) and J_N's anchor `anc` (doc["anchor"] of J_N).
    CLEAR / FIRED / UNTESTED.  UNTESTED when the anchor cell is not HI_SATURATED, is HARNESS_LIMITED / CLIENT_ERROR / TRUNCATED /
    contaminated, lacks an arm or a lambda*, or when the reference lacks one.  FIRED iff |G_anchor/G_ref - 1| > 3 % or
    |lambda*_anchor(fused_sym)/lambda*_ref(fused_sym) - 1| > 3 %.
    `_mut` 'ii' (tests only) = compare against J1's FINAL (extension-rung) value instead of boot1; 'flags_final' = read harness_limited /
    client_error of J1's FINAL cell set (rev11 M-2 mutant)."""
    tol = C.JOB_EFFECT_TOL if tol is None else tol
    why = []
    arms = (anc or {}).get("arms") or {}
    for a in C.ANCHOR_ARMS:
        e = arms.get(a) or {"present": False}
        if not e.get("present"):
            why.append(f"{a}:ANCHOR_ABSENT")
            continue
        if e.get("hi_label") != "HI_SATURATED":
            why.append(f"{a}:ANCHOR_{e.get('hi_label')}")
        if e.get("harness_limited"):
            why.append(f"{a}:ANCHOR_HARNESS_LIMITED")
        if e.get("client_error"):
            why.append(f"{a}:ANCHOR_CLIENT_ERROR")
        if e.get("truncated"):
            why.append(f"{a}:ANCHOR_TRUNCATED")
        if e.get("lambda_star") is None:
            why.append(f"{a}:ANCHOR_LAMBDA_STAR_UNRESOLVED")
        if a in ((anc or {}).get("gpu_contaminated_arms") or []):
            why.append(f"{a}:ANCHOR_GPU_CONTAMINATED")
    refkey = "final_lambda_star" if _mut == "ii" else "boot1_lambda_star"
    for a in C.ANCHOR_ARMS:                                      # H-3: the J1 boot1 reference cell must itself be a clean saturated cell
        e = (ref or {}).get(a) or {"present": False}
        if not e.get("present") or e.get(refkey) is None:
            why.append(f"{a}:J1_REFERENCE_UNDEFINED")
            continue
        if e.get("boot1_hi_label") != "HI_SATURATED":
            why.append(f"{a}:J1_REFERENCE_{e.get('boot1_hi_label')}")
        hl_k, ce_k = (("harness_limited", "client_error") if _mut == "flags_final" else ("boot1_harness_limited", "boot1_client_error"))
        if hl_k not in e or ce_k not in e:                       # fail-closed: a reference without the boot1 flags is not a clean cell
            why.append(f"{a}:J1_REFERENCE_BOOT1_FLAGS_MISSING")
            continue
        if e.get(hl_k):
            why.append(f"{a}:J1_REFERENCE_HARNESS_LIMITED")
        if e.get(ce_k):
            why.append(f"{a}:J1_REFERENCE_CLIENT_ERROR")
    if why:
        return {"status": "UNTESTED", "reasons": why}
    lam = lambda d, a, k: d[a][k]
    g_ref = ref["agnostic"][refkey] / ref["fused_sym"][refkey]
    g_anc = arms["agnostic"]["lambda_star"] / arms["fused_sym"]["lambda_star"]
    g_rel = g_anc / g_ref - 1.0
    l_rel = arms["fused_sym"]["lambda_star"] / ref["fused_sym"][refkey] - 1.0
    fired = abs(g_rel) > tol or abs(l_rel) > tol
    return {"status": "FIRED" if fired else "CLEAR", "reasons": [], "G_anchor": g_anc, "G_ref": g_ref, "G_rel": g_rel,
            "lambda_fused_sym_rel": l_rel, "tolerance": tol}


def _g_of_arrays(anc_arms):
    return anc_arms["agnostic"]["achieved_by_seed"], anc_arms["fused_sym"]["achieved_by_seed"]


_BOOT_ID = re.compile(r"^m3_(?P<cfg>N1|TP\d+|DP\d+)_(?P<arm>fused_sym|fused|agnostic|d44)_(?P<kind>[A-Za-z0-9]+)")
HARD_CAP_REASONS = ("NOT_ATTEMPTED_BUDGET", "RECONSTRUCTED_NO_BOOT_RECORD")      # unconditional "the hard cap cut this boot"
EXT_INCOMPLETE = "NOT_ATTEMPTED_EXT_INCOMPLETE"                                   # CONDITIONAL (rev11 D-C): boot2 not entered because the extension was incomplete
CUT_REASONS = HARD_CAP_REASONS + (EXT_INCOMPLETE,)


def driver_cut_arms(doc, _mut=None):
    """rev11 M-1 / D-C: the (config, arm) units the Job's HARD CAP cut, read from the DRIVER'S OWN RECORDS -- never from an estimand's reason text:
      * driver.not_attempted with reason NOT_ATTEMPTED_BUDGET (admission refused) or RECONSTRUCTED_NO_BOOT_RECORD (reconstructed driver: a planned
        main boot with no boots/<id>/boot.json -- fail-closed);
      * driver.partial_boots (a boot with a NOT_STARTED_DEADLINE step / rc 6, mg_boot.py);
      * a cell the runner flagged TRUNCATED (doc["keys"][...]["truncated"]);
      * D-C: NOT_ATTEMPTED_EXT_INCOMPLETE (boot2 not entered) counts ONLY when the SAME config's extension boot (kind boot1ext) has a hard-cap record --
        a NOT_ATTEMPTED_BUDGET / a partial boot of that extension boot, or a TRUNCATED_CELL of a key with extension cells (max_ext >= 1).  An extension
        that is incomplete because a boot FAILED TO START has no such record: not a truncation (registered section 17-2(b) / P9-c path).
    Only main-block boots count (an anchor / GPU-deviation boot is not a trend point); TP_NOT_TESTED / M2g exclusions are not truncation.
    `_mut` 'ext_unconditional' (tests only) = the rev11 predicate (EXT_INCOMPLETE always cuts).  Returns {(config, arm): sorted sources}."""
    cut = {}
    ext_evidence = set()

    def add(cfg, arm, why):
        cut.setdefault((cfg, arm), set()).add(why)

    drv = (doc or {}).get("driver") or {}

    def unit(x):
        cfg, arm, block = x.get("config"), x.get("arm"), x.get("block")
        if cfg is None or arm is None:
            m = _BOOT_ID.match(str(x.get("boot_id", "")))
            if not m:
                return None
            cfg, arm = m.group("cfg"), m.group("arm")
            block = block or ("anchor" if m.group("kind") == "anchor" else "main")
        return (cfg, arm) if block in (None, "main") else None

    def is_ext(x):
        m = _BOOT_ID.match(str(x.get("boot_id", "")))
        return bool(m and m.group("kind") == "boot1ext")

    for x in drv.get("not_attempted") or []:
        u = unit(x)
        if x.get("reason") in HARD_CAP_REASONS and u:
            add(*u, x["reason"])
            if is_ext(x):
                ext_evidence.add(u[0])
    for x in drv.get("partial_boots") or []:
        u = unit(x)
        if u:
            add(*u, "PARTIAL_BOOT")
            if is_ext(x):
                ext_evidence.add(u[0])
    for key, lab in ((doc or {}).get("keys") or {}).items():
        if lab.get("truncated"):
            cfg, arm = key.split("|")[:2]
            if cfg != C.ANCHOR_CONFIG:
                add(cfg, arm, "TRUNCATED_CELL")
                if int(lab.get("max_ext", 1) or 0) >= 1:
                    ext_evidence.add(cfg)
    for x in drv.get("not_attempted") or []:
        u = unit(x)
        if x.get("reason") == EXT_INCOMPLETE and u and (u[0] in ext_evidence or _mut == "ext_unconditional"):
            add(*u, EXT_INCOMPLETE)
    return {k: sorted(v) for k, v in cut.items()}


def _read_driver_log(root):
    out = []
    lg = Path(root) / "DRIVER_LOG.jsonl"
    if lg.exists():
        for ln in lg.read_text().splitlines():
            try:
                ev = json.loads(ln)
            except ValueError:
                continue
            if isinstance(ev, dict):
                out.append(ev)
    return out


def reconstruct_driver(root, job, excluded=(), _mut=None):
    """rev11 D-B: a Job directory WITHOUT a `driver` record in LABELS.json (the driver died before finish()) is RECONSTRUCTED, not refused:
      * DRIVER_LOG.jsonl events NOT_ATTEMPTED_BUDGET / NOT_ATTEMPTED_EXT_INCOMPLETE (key `boot` = boot_id) -> driver.not_attempted, the boot_id split
        by the same regex as driver_cut_arms;
      * partial_boots_of(root) (boot.json steps) -> driver.partial_boots;
      * fail-closed: a PLANNED main boot (mg_plan: boot1 of every config the log does not show as EXCLUDED / TP_NOT_TESTED, arms not excluded by the
        pass table) -- or a boot2 the config's logged decision (x_lo non-empty, or no decision logged at all) implies, or any boot with a specs/<id>.json --
        that has no boots/<id>/boot.json is a TRUNCATION (reason RECONSTRUCTED_NO_BOOT_RECORD).  A boot that HAS a boot.json (e.g. a failed start)
        is not.   `excluded`: pass-table 'cfg:arm' keys.   `_mut` 'no_missing_boot_cut' (tests only) drops the last rule."""
    import mg_plan as P
    root = Path(root)
    ex = set(excluded)
    events = _read_driver_log(root)
    na, seen = [], set()
    skipped, decisions = set(), {}
    for ev in events:
        st, bid = ev.get("status"), ev.get("boot")
        if st in ("NOT_ATTEMPTED_BUDGET", EXT_INCOMPLETE) and bid and bid not in seen:
            m = _BOOT_ID.match(str(bid))
            if not m:
                continue
            seen.add(bid)
            na.append({"boot_id": bid, "config": m.group("cfg"), "arm": m.group("arm"),
                       "block": "anchor" if m.group("kind") == "anchor" else "main", "reason": st, "reconstructed": True})
        if ev.get("config") and st in ("EXCLUDED_BY_M2G_PASS_TABLE", "TP_NOT_TESTED"):
            skipped.add(ev["config"])
        if ev.get("config") and isinstance(ev.get("decision"), dict):
            decisions[ev["config"]] = ev["decision"]
    ext_cfg = {x["config"] for x in na if x["reason"] == EXT_INCOMPLETE}
    have = lambda bid: (root / "boots" / bid / "boot.json").exists()

    def miss(cfg, arm, kind):
        bid = f"m3_{cfg}_{arm}_{kind}"
        if bid in seen or have(bid) or _mut == "no_missing_boot_cut":
            return
        seen.add(bid)
        na.append({"boot_id": bid, "config": cfg, "arm": arm, "block": "main", "reason": "RECONSTRUCTED_NO_BOOT_RECORD", "reconstructed": True})

    for cfg in P.config_order(job):
        if cfg in skipped:
            continue
        arms = [a for a in P.arm_order(cfg) if f"{cfg}:{a}" not in ex]
        for a in arms:
            miss(cfg, a, "boot1")
        dec = decisions.get(cfg)
        boot2_implied = (dec is None) or bool((dec or {}).get("x_lo"))
        for a in arms:
            if (root / "specs" / f"m3_{cfg}_{a}_boot1ext.json").exists():
                miss(cfg, a, "boot1ext")
            if (root / "specs" / f"m3_{cfg}_{a}_boot2.json").exists() or (boot2_implied and cfg not in ext_cfg):
                miss(cfg, a, "boot2")
    return {"decisions": {"reconstructed": True, "from": "DRIVER_LOG.jsonl + boots/ + specs/", "config_decisions": decisions},
            "not_attempted": na, "partial_boots": partial_boots_of(root), "truncated_cells": [], "reconstructed": True}


def partial_boots_of(root):
    """Boots of one Job directory that the deadline interrupted: boots/<id>/boot.json has a NOT_STARTED_DEADLINE step (mg_boot.py).
    Joined with specs/<id>.json for (config, arm, block).  Written into LABELS.json driver.partial_boots by mg_drive.finish."""
    root = Path(root)
    out = []
    for bj in sorted((root / "boots").glob("*/boot.json")) if (root / "boots").exists() else []:
        try:
            b = json.loads(bj.read_text())
        except ValueError:
            continue
        if not any(st.get("status") == "NOT_STARTED_DEADLINE" for st in b.get("steps") or []):
            continue
        bid = bj.parent.name
        rec = {"boot_id": bid}
        sp = root / "specs" / f"{bid}.json"
        if sp.exists():
            try:
                sd = json.loads(sp.read_text())
                rec.update(config=sd.get("config"), arm=sd.get("arm"), block=sd.get("block", "main"))
            except ValueError:
                pass
        out.append(rec)
    return out


def aggregate_jobs(docs, job_meta=None, _mut=None, not_joined=None):
    """Join the per-Job documents {J1: doc, J2: doc, ...} into trends.  N=1 points come from J1 ONLY; N=2/4/8 from the Job of that size.
    P9-a  G_sym(1, shape) undefined -> that shape's every trend is G1_UNDEFINED_NO_JOB_CHECK, every J_N JOB_EFFECT_UNTESTED, P-DP UNTESTED.
    P9-b / P9-c: see trend() and p_dp_status().
    The anchor of J_N is used ONLY for J_N's own point (mutant iii borrows another Job's anchor).
    rev11 D-B: `not_joined` {J: reason} = Jobs the caller explicitly left out (CLI `J8=NOT_SUBMITTED`).  Every Job with no document is listed in
    out["JOB_NOT_JOINED"] and every trend carries the disclosure JOB_NOT_JOINED(<Js>) -- a trend over a subset of the registered Ns is never silent."""
    job_meta = job_meta or {}
    out = {"prereg": f"{C.PREREG} rev{C.PREREG_REV}", "jobs": {}, "trends": {}, "job_effect": {}, "P_DP": None}
    nj = {J: (dict(not_joined or {}).get(J) or "NO_DOCUMENT") for J in C.JOBS if docs.get(J) is None}
    out["JOB_NOT_JOINED"] = nj
    j1 = docs.get("J1")
    ref = (j1 or {}).get("ref") or {}
    _cut_cache = {}

    def cut_of(d):
        if id(d) not in _cut_cache:
            _cut_cache[id(d)] = driver_cut_arms(d, _mut="ext_unconditional" if _mut == "ext_unconditional" else None)
        return _cut_cache[id(d)]
    g1_def = lambda name, shape: bool(j1 and ((((j1.get("estimands") or {}).get(name) or {}).get("N1") or {}).get(shape) or {}).get("defined"))
    for J in C.JOBS:
        d = docs.get(J)
        out["jobs"][J] = {"present": d is not None, "n_cells": (d or {}).get("n_cells"), **(job_meta.get(J) or {})}
        if J in nj:
            out["jobs"][J]["JOB_NOT_JOINED"] = nj[J]
    # ---- Job effect per Job (shape A anchor).  P9-a: G_sym(1, A) undefined -> every Job UNTESTED.
    status = {}
    for J in C.JOBS[1:]:
        d = docs.get(J)
        if not g1_def("G_sym", C.ANCHOR_SHAPE):
            status[J] = {"status": "UNTESTED", "reasons": [G1_UNDEFINED + " (P9-a: G_sym(1, A) undefined)"]}
        elif d is None:
            status[J] = {"status": "UNTESTED", "reasons": ["JOB_NOT_RUN"]}
        else:
            status[J] = job_effect(ref, d.get("anchor"), _mut=_mut)
    out["job_effect"] = status
    ra, rf = (ref.get("agnostic") or {}).get("boot1_achieved_by_seed"), (ref.get("fused_sym") or {}).get("boot1_achieved_by_seed")
    ref_g1 = (ra, rf) if (ra and rf and all(v is not None for v in list(ra) + list(rf))) else None      # J1 boot1 arrays (H-3)
    anchors = {}
    for J in C.JOBS[1:]:
        d = docs.get(J)
        arms = ((d or {}).get("anchor") or {}).get("arms") or {}
        ok = all((arms.get(a) or {}).get("present") and (arms.get(a) or {}).get("hi_label") == "HI_SATURATED"
                 and not (arms.get(a) or {}).get("harness_limited") and not (arms.get(a) or {}).get("client_error")
                 and (arms.get(a) or {}).get("achieved_by_seed") and all(v is not None for v in (arms[a]["achieved_by_seed"]))
                 and a not in (d["anchor"].get("gpu_contaminated_arms") or []) for a in C.ANCHOR_ARMS)
        if ok:
            anchors[J] = _g_of_arrays(d["anchor"]["arms"])
    for name in PAIRS:
        if name == "S":
            continue
        for mode in ("TP", "DP"):
            for shape in ("A", "B"):
                pts, undefined, disc, pnotes, gaps, flags, anc_by_N, jobs_of = [], [], set(), [], {}, {}, {}, {}
                for J in C.JOBS:
                    n_gpus = C.JOB_N[J]
                    if J == "J1" and mode == "TP":
                        cfg = "N1"
                    elif J == "J1":
                        cfg = "N1"
                    else:
                        cfg = f"{mode}{n_gpus}"
                    d = docs.get(J)
                    ent = (((d or {}).get("estimands") or {}).get(name) or {}).get(cfg, {}).get(shape) if d else None
                    if ent and ent.get("defined"):
                        pts.append((n_gpus, ent["num"], ent["den"]))
                        jobs_of[n_gpus] = J
                        disc |= set(ent.get("disclosures", []))
                        pnotes += [f"{cfg}:{a}:{v}" for a, v in ent.get("partition", {}).items() if v != "PARTITION_EXERCISED"]
                        if ent.get("measurement_gaps"):
                            gaps[cfg] = ent["measurement_gaps"]
                        if J != "J1":
                            st = status[J]["status"]
                            flags[n_gpus] = st
                            ju = J if _mut != "iii" else (sorted(anchors) or [J])[0]      # mutant iii: another Job's anchor
                            if ju in anchors:
                                anc_by_N[n_gpus] = anchors[ju]
                    else:
                        rs = (ent or {"undefined_reasons": ["NOT_RUN"]})["undefined_reasons"]
                        pair = PAIRS[name]
                        src = {a: cut_of(d).get((cfg, a), []) for a in pair}
                        trunc = any(src.values())                  # M-1: (cfg, arm) cut by the hard cap, from the driver's records
                        if _mut == "trunc_reason":                 # mutant: the rev10 predicate -- reason TEXT ("TRUNCATED" / "NOT_RUN" substring)
                            na = [x for x in (((d or {}).get("driver") or {}).get("not_attempted") or [])
                                  if str(x.get("boot_id", "")).startswith(f"m3_{cfg}_") and x.get("reason") in CUT_REASONS]
                            trunc = any("TRUNCATED" in r for r in rs) or (any("NOT_RUN" in r for r in rs) and bool(na))
                        undefined.append({"config": cfg, "job": J, "reasons": rs, "truncated": trunc,
                                          "cut_by": {a: v for a, v in src.items() if v} if _mut != "trunc_reason" else {}})
                own_g1 = any(p[0] == 1 for p in pts)
                g1_ok = own_g1 and g1_def("G_sym", shape)
                if not g1_ok:
                    pts = [p for p in pts if p[0] != 1]            # trend() then returns G1_UNDEFINED_NO_JOB_CHECK (needs >= 3 points)
                # scope of the Job-effect check for this series (MGP-46): the anchor is shape A / G_sym; elsewhere a CLEAR is UNTESTED-scope
                flags_series = {}
                for n, st in flags.items():
                    flags_series[n] = ("UNTESTED_SCOPE" if (st == "CLEAR" and not (name == PRIMARY and shape == C.ANCHOR_SHAPE)) else st)
                if not g1_ok:
                    flags_series = {n: "UNTESTED" for n in flags_series}
                t = trend(pts, anchors=anc_by_N, job_flags=flags_series, ref_g1=ref_g1, _mut=_mut)
                if not g1_ok and len([p for p in pts]) < C.TREND_MIN_POINTS and t["label"] == "TREND_UNDEFINED":
                    t = dict(t, label=G1_UNDEFINED, raw_label="TREND_UNDEFINED", direction_citable=False)   # G(1) missing outranks "too few points"
                t["undefined_count"] = len(undefined)
                t["undefined"] = undefined          # MGP-34 / MGP-25: always printed beside the label
                t["primary"] = (name == PRIMARY)
                t["disclosures"] = sorted(disc)
                t["partition_notes"] = pnotes
                t["partition_exercised_unknown"] = any(n.endswith(":E_TIME_UNDEFINED") for n in pnotes)
                t["measurement_gaps"] = gaps
                t["job_effect_by_N"] = {int(n): flags_series.get(n, "n/a") for n in sorted(jobs_of) if n != 1}
                untested = sorted(f"{jobs_of[n]}" for n, st in flags_series.items() if st.startswith("UNTESTED"))
                if untested:                                       # §15-3: the point stays in the trend, the disclosure is mandatory
                    t["disclosures"] = sorted(set(t["disclosures"]) | {"JOB_EFFECT_UNTESTED(" + ",".join(untested) + ")"})
                if not g1_ok:
                    t["disclosures"] = sorted(set(t["disclosures"]) | {G1_UNDEFINED})
                if nj:
                    t["disclosures"] = sorted(set(t["disclosures"]) | {"JOB_NOT_JOINED(" + ",".join(sorted(nj)) + ")"})
                t["g1_defined"] = g1_ok
                t["jobs"] = {int(n): j for n, j in sorted(jobs_of.items())}
                out["trends"][f"{name}|{mode}|{shape}"] = t
    dx = sorted(J for J in C.JOBS if ((docs.get(J) or {}).get("driver") or {}).get("exception"))
    if dx and _mut != "no_exception_disclosure":                    # rev11 D-D: every trend of a Job whose driver died by exception says so
        for t in out["trends"].values():
            t["disclosures"] = sorted(set(t["disclosures"]) | {"JOB_DRIVER_EXCEPTION(" + ",".join(dx) + ")"})
    out["JOB_DRIVER_EXCEPTION"] = dx
    out["P_DP"] = p_dp_status(out["trends"], _mut=(_mut if _mut in ("same_shape_only", "raw_unconditional") else None))
    # ---- rev10 P11 (b) / H-2: a DP trend that is TREND_UNDEFINED BECAUSE OF TRUNCATED cells, with P-DP UNTESTED as the consequence, forbids
    # citing the TP direction labels even with the disclosure.  A non-truncation reason keeps the P9-c path (disclose, then cite).
    # The reverse (TP cut by truncation) leaves the DP labels for the P-DP verdict only.
    isund = lambda t: "TREND_UNDEFINED" in (t["label"], t.get("raw_label"))
    cut = lambda t: isund(t) and any(u.get("truncated") for u in t["undefined"])
    dp_trunc = [k for k, t in out["trends"].items() if k.startswith("G_sym|DP|") and cut(t)]
    tp_trunc = [k for k, t in out["trends"].items() if k.startswith("G_sym|TP|") and cut(t)]
    out["truncation"] = {"dp_trend_undefined_by_truncation": dp_trunc, "tp_trend_undefined_by_truncation": tp_trunc}
    if dp_trunc and out["P_DP"]["status"] == "UNTESTED" and _mut != "trunc":
        for k, t in out["trends"].items():
            if "|TP|" in k and t["label"] not in ("TREND_UNDEFINED", G1_UNDEFINED):
                t.update(label_before_truncation_rule=t["label"], label="TP_UNCONTROLLED_TRUNCATION", direction_citable=False)
                t["disclosures"] = sorted(set(t["disclosures"]) | {"TP_UNCONTROLLED_TRUNCATION"})
        out["P_DP"]["TP_UNCONTROLLED_TRUNCATION"] = True
    if tp_trunc:
        for k, t in out["trends"].items():
            if "|DP|" in k:
                t["used_for_p_dp_only"] = True
    for k, t in out["trends"].items():
        if "|TP|" in k:
            t["P_DP"] = out["P_DP"]["status"]
            if out["P_DP"]["status"] == "UNTESTED":
                t["disclosures"] = sorted(set(t["disclosures"]) | {"P-DP 미검정"})        # MGH-8 / P9-c: no TP citation without it
            elif out["P_DP"]["status"] == "BROKEN":
                t["disclosures"] = sorted(set(t["disclosures"]) | {"P-DP BROKEN: MGP-13 TP labels not citable"})
    out["memory_rule_iii"] = {"status": "RETIRED_POST_HOC (user decision 2026-10-13; the M2g round-2 values had been seen)",
                              "configs_iii_would_have_made_undefined": {J: (docs[J].get("memory_rule_iii") or {}).get("configs_iii_would_have_made_undefined", [])
                                                                        for J in C.JOBS if J in docs},
                              "MEMORY_RULE_III_COUNTERFACTUAL": {J: (docs[J].get("memory_rule_iii") or {}).get("MEMORY_RULE_III_COUNTERFACTUAL_by_config", {})
                                                                 for J in C.JOBS if J in docs}}
    out["gpu_dev"] = (docs.get("J8") or {}).get("blocks", {}).get("gpu_dev")
    return out


def a_is_pdmux(arms):
    return any(a in C.PDMUX_ARMS for a in arms)


def _g_sym_round(cfg, keys, rnd):
    a, f = keys.get((cfg, "agnostic", "A", rnd)), keys.get((cfg, "fused_sym", "A", rnd))
    if a and f and a["lambda_star"] and f["lambda_star"]:
        return a["lambda_star"] / f["lambda_star"]
    return None


def job_covariates(out_dir):
    """§16-3: the Job's cgroup cpu.max (and visible cores) from its preflight record SUBSTRATE_CPU.txt -- a covariate printed beside every result."""
    p = Path(out_dir) / "SUBSTRATE_CPU.txt"
    cov = {"cpu_max": None, "nproc": None}
    if p.exists():
        for ln in p.read_text(errors="replace").splitlines():
            if ln.startswith("cgroup_cpu_max="):
                cov["cpu_max"] = ln.split("=", 1)[1].strip()
            elif ln.startswith("nproc="):
                cov["nproc"] = ln.split("=", 1)[1].split()[0]
    return cov


def render_text(doc):
    """One Job's document (aggregate)."""
    L = [f"# multi-GPU labels, Job {doc.get('job')} ({doc['prereg']})", ""]
    cov = doc.get("covariates") or {}
    L.append(f"covariate cpu.max={cov.get('cpu_max')} nproc={cov.get('nproc')}")
    for name in PAIRS:
        for cfg, per in sorted((doc["estimands"].get(name) or {}).items()):
            for shape, e in sorted(per.items()):
                s = f"{name:9s} {cfg:4s} {shape}: "
                s += (f"G={e['point']:.4f} CI=[{e['ci'][0]:.4f},{e['ci'][1]:.4f}]" if e.get("defined") else "UNDEFINED " + "; ".join(e["undefined_reasons"]))
                if e.get("disclosures"):
                    s += "  DISCLOSE=" + ",".join(e["disclosures"])
                L.append(s)
    for k, lab in sorted(doc["keys"].items()):
        L.append(f"key {k}: knee={lab['knee']} lambda*={lab['lambda_star']} harness_limited={lab['harness_limited']} "
                 f"throttle max={lab.get('throttle_ratio_max')} median={lab.get('throttle_ratio_median')}")
    if doc.get("anchor"):
        L.append("anchor " + json.dumps(doc["anchor"], sort_keys=True, default=str))
    for cfg, fl in sorted(doc["configs"].items()):
        extra = [k for k in ("memory_mismatch", "gpu_contaminated", "contam_unverified", "tp_numeric_divergent", "g2_not_tested",
                             "m2g_memory_mismatch", "frontend_saturated") if fl.get(k)]
        extra += [f"{a}:G1_UNINFORMATIVE" for a in fl.get("g1_uninformative_arms", [])]
        if extra:
            L.append(f"config {cfg}: {', '.join(extra)}")
    mr = doc.get("memory_rule_iii") or {}
    L.append(f"memory rule (iii) {mr.get('status')}: configs it would have made undefined = {mr.get('configs_iii_would_have_made_undefined')} "
             f"MEMORY_RULE_III_COUNTERFACTUAL={json.dumps(mr.get('MEMORY_RULE_III_COUNTERFACTUAL_by_config'), sort_keys=True)}")
    for k, v in doc["blocks"].items():
        L.append(f"block {k}: {json.dumps(v, sort_keys=True)}")
    return "\n".join(L) + "\n"


def render_jobs_text(doc):
    """The cross-Job document (aggregate_jobs)."""
    L = [f"# multi-GPU labels, Jobs joined ({doc['prereg']})", ""]
    if doc.get("JOB_NOT_JOINED"):
        L.append(f"JOB_NOT_JOINED {doc['JOB_NOT_JOINED']} -- the trends below are computed WITHOUT these Ns (P-DP status and TP citability can differ if they had run)")
    for J, m in doc["jobs"].items():
        L.append(f"job {J}: present={m['present']} cpu.max={m.get('cpu_max')} nproc={m.get('nproc')} n_cells={m.get('n_cells')}")
    for J, st in doc["job_effect"].items():
        L.append(f"JOB_EFFECT {J}: {st['status']} {('; '.join(st['reasons'])) if st['reasons'] else ''} "
                 + (f"G_rel={st['G_rel']:+.4f} lambda_fused_sym_rel={st['lambda_fused_sym_rel']:+.4f}" if "G_rel" in st else ""))
    L.append("")
    for k, t in sorted(doc["trends"].items()):
        mark = "PRIMARY" if t["primary"] else "auxiliary"
        s = f"{k:20s} [{mark}] {t['label']}"
        if "slope" in t:
            s += f" slope={t['slope']:+.4f} CI=[{t['ci'][0]:+.4f},{t['ci'][1]:+.4f}] Ns={t['Ns']} magnitude={t.get('magnitude')}"
        if t.get("corrected"):
            c = t["corrected"]
            s += f"  CORRECTED {c['label']} slope={c['slope']:+.4f} CI=[{c['ci'][0]:+.4f},{c['ci'][1]:+.4f}] magnitude={c.get('magnitude')}"
        if t.get("job_sensitive_reason"):
            s += f"  ({t['job_sensitive_reason']})"
        s += f"  direction_citable={t.get('direction_citable')}  UNDEFINED_CELLS={t['undefined_count']}"
        if t.get("disclosures"):
            s += "  DISCLOSE=" + ",".join(t["disclosures"])
        L.append(s)
        for u in t["undefined"]:
            L.append(f"      undefined {u['config']} ({u['job']}): {'; '.join(u['reasons'])}")
    L.append("")
    pd = doc["P_DP"]
    L.append(f"P_DP status={pd['status']} per_shape={pd['per_shape']} broken={pd['broken']} untested_because={pd['untested_because']} "
             f"(MGP-13 TP labels not citable={pd['MGP13_TP_labels_not_citable']}; TP labels need 'P-DP 미검정' disclosure={pd['TP_labels_need_P_DP_UNTESTED_disclosure']})")
    mr = doc.get("memory_rule_iii") or {}
    L.append(f"memory rule (iii) {mr.get('status')}")
    for J, cl in (mr.get("configs_iii_would_have_made_undefined") or {}).items():
        L.append(f"   cells rule (iii) would have made undefined, {J}: {cl}")
    if doc.get("gpu_dev"):
        L.append("block gpu_dev: " + json.dumps(doc["gpu_dev"], sort_keys=True))
    return "\n".join(L) + "\n"


# =========================================================================== rev9 mutation harness (Job layer; lesson 53 + 254)
def _syn_ent(G, noise=0.0):
    num = [G * (1.0 + noise * k) for k in range(4)]
    return {"defined": True, "num": num, "den": [1.0] * 4, "undefined_reasons": [], "disclosures": [], "partition": {}}


def _syn_undef(*reasons):
    return {"defined": False, "undefined_reasons": list(reasons) or ["NOT_RUN"]}


def _syn_anchor(G, lam_fs=1.0, **over):
    arm = lambda lam: dict(present=True, lambda_star=lam, hi_label="HI_SATURATED", achieved_by_seed=[lam] * 4, harness_limited=False,
                           client_error=False, truncated=False)
    d = {"arms": {"fused_sym": arm(lam_fs), "agnostic": arm(G * lam_fs)}, "gpu_contaminated_arms": []}
    d["arms"]["fused_sym"].update(over)
    return d


def _syn_ref(g_ref=1.0, final_mult=1.0, **over):
    """J1 boot1 reference: fused_sym lambda* 1.0, agnostic lambda* g_ref; `final_mult` scales the extension-rung (final) lambda*."""
    arm = lambda lam: dict(present=True, boot1_lambda_star=lam, final_lambda_star=lam * final_mult, boot1_hi_label="HI_SATURATED",
                           harness_limited=False, client_error=False, boot1_harness_limited=False, boot1_client_error=False,
                           boot1_achieved_by_seed=[lam] * 4)
    r = {"fused_sym": arm(1.0), "agnostic": arm(g_ref)}
    for a, kv in over.items():
        r[a].update(kv)
    return r


def _syn_docs(g1, series, anchors, shape="A", ref=None, drivers=None, series_b=None, g1_b=1.0):
    """g1: G(1) (extension-rung/final value the trend uses) or None (undefined).  series: {cfg: G | ('undef', reason...)} e.g. {'DP2': 1.0}.
    anchors: {N: G_anchor | None (anchor absent)}.  drivers: {N: driver record of that Job} (rev11 M-1: truncation is read from it).
    series_b: {cfg: G} -- the same configs' shape-B estimands (rev11 M-4); N=1 shape-B point = g1_b."""
    ests = {}
    series_b = dict(series) if series_b == "same" else series_b          # "same": shape B mirrors shape A (a real aggregate() always has both shapes)
    for cfg, v in series.items():
        ests.setdefault(int("".join(ch for ch in cfg if ch.isdigit())), {})[cfg] = _syn_undef(*v[1:]) if isinstance(v, tuple) else _syn_ent(v)
    n1 = {shape: _syn_ent(g1)} if g1 is not None else {}
    if series_b is not None:
        n1["B"] = _syn_ent(g1_b)
    docs = {"J1": {"estimands": {"G_sym": ({"N1": n1} if n1 else {})}, "ref": ref or _syn_ref(), "n_cells": 1, "anchor": {}}}
    for n, cfgs in ests.items():
        J = f"J{n}"
        docs[J] = {"estimands": {"G_sym": {cfg: {shape: e} for cfg, e in cfgs.items()}}, "ref": {}, "n_cells": 1,
                   "anchor": ({} if anchors.get(n) is None else _syn_anchor(anchors[n]))}
        if series_b is not None:
            for cfg in cfgs:
                if cfg in series_b:
                    v = series_b[cfg]
                    docs[J]["estimands"]["G_sym"][cfg]["B"] = _syn_undef(*v[1:]) if isinstance(v, tuple) else _syn_ent(v)
        if drivers and n in drivers:
            docs[J]["driver"] = drivers[n]
    return docs


def _syn_noisy(docs, noise, seed):
    """Deterministic per-seed noise on every defined numerator (shape-A DP/N1 at `noise`, TP at noise/3) so a bootstrap CI has a width."""
    rng = np.random.default_rng(seed)
    for d in docs.values():
        for per in (d.get("estimands") or {}).values():
            for cfg, shp in per.items():
                for e in shp.values():
                    if e.get("defined"):
                        sc = noise if (cfg.startswith("DP") or cfg == "N1") else noise / 3
                        e["num"] = [v * (1 + sc * float(rng.uniform(-1, 1))) for v in e["num"]]
    return docs


def _syn_cut(*cfg_arm_boot, reason="NOT_ATTEMPTED_BUDGET"):
    """A driver record: boots (cfg, arm, kind) the hard cap refused -- the shape mg_drive.run_boot writes (boot_id / need_s / reason)."""
    return {"not_attempted": [{"boot_id": f"m3_{c}_{a}_{k}", "need_s": 5000.0, "remaining_s": 100.0, "reason": reason} for c, a, k in cfg_arm_boot]}


_LO_REFUSED = "KNEE_NOT_BRACKETED[LO_NOT_JUDGED:4385=MISSING,4386=MISSING,4387=MISSING,4388=MISSING]"     # what a refused boot2 looks like in aggregate()
_lab = lambda t: t["label"]
JOB_MUT_SCENARIOS = {
    # name -> (docs, series key, expected outcome, outcome extractor, {mutant tag: kw for aggregate_jobs _mut})
    "S_clean_direction": (_syn_docs(1.0, {"DP2": 1.04, "DP4": 1.08}, {2: 1.0, 4: 1.0}), "G_sym|DP|A", "G_INCREASES_WITH_N", _lab),
    "S_i_g1_undefined_J8_plus2pct": (_syn_docs(None, {"DP2": 1.00, "DP4": 1.00, "DP8": 1.02}, {2: 1.0, 4: 1.0, 8: 1.0}), "G_sym|DP|A", G1_UNDEFINED, _lab),
    "S_iii_other_jobs_anchor": (_syn_docs(1.0, {"DP2": 1.00, "DP4": 1.08}, {2: 1.0, 4: 1.08}), "G_sym|DP|A", TREND_JOB_SENSITIVE, _lab),
    "S_iv_corrected_below_3pct": (_syn_docs(1.0, {"DP2": 1.04, "DP4": 1.08}, {2: 1.0, 4: 1.05}), "G_sym|DP|A", TREND_JOB_SENSITIVE, _lab),
    # H-3: J1 boot1 G_ref = 1.0 but the extension-rung G(1) the trend uses is 1.10 -> the correction ratio must come from boot1
    "S_h3_boot1_vs_extension_G1": (_syn_docs(1.10, {"DP2": 1.10, "DP4": 1.20}, {2: 1.0, 4: 1.10}), "G_sym|DP|A", TREND_JOB_SENSITIVE, _lab),
    # H-4: the J2 anchor is absent (UNTESTED) -> J2 leaves the corrected trend, 2 points remain -> TREND_UNDEFINED -> P9-b fails
    "S_h4_untested_anchor_point_excluded": (_syn_docs(1.0, {"DP2": 1.08, "DP4": 1.08}, {2: None, 4: 1.04}), "G_sym|DP|A", TREND_JOB_SENSITIVE, _lab),
    # H-2 / rev11 M-1: DP boot2 admission refused by the hard cap (driver: NOT_ATTEMPTED_BUDGET) -> aggregate() reports only
    # KNEE_NOT_BRACKETED[LO_NOT_JUDGED..] (no "TRUNCATED" text) -> DP TREND_UNDEFINED -> P-DP UNTESTED -> TP direction not citable
    "S_h2_truncated_dp_blocks_tp": (_syn_docs(1.0, {"DP2": 1.0, "DP4": ("undef", "agnostic:" + _LO_REFUSED), "DP8": ("undef", "agnostic:" + _LO_REFUSED),
                                                     "TP2": 1.04, "TP4": 1.08, "TP8": 1.12}, {2: 1.0, 4: 1.0, 8: 1.0},
                                    drivers={4: _syn_cut(("DP4", "agnostic", "boot2")), 8: _syn_cut(("DP8", "agnostic", "boot2"))}, series_b="same"),
                                    "G_sym|TP|A", "TP_UNCONTROLLED_TRUNCATION", _lab),
    # rev11 M-1 (b): TP boot2 refused (TP4/TP8) while DP is fine -> DP labels are used for the P-DP verdict only
    "S_h2_tp_cut_marks_dp_p_dp_only": (_syn_docs(1.0, {"DP2": 1.0, "DP4": 1.0, "DP8": 1.0,
                                                       "TP2": 1.04, "TP4": ("undef", "agnostic:" + _LO_REFUSED), "TP8": ("undef", "agnostic:" + _LO_REFUSED)},
                                                 {2: 1.0, 4: 1.0, 8: 1.0},
                                                 drivers={4: _syn_cut(("TP4", "agnostic", "boot2")), 8: _syn_cut(("TP8", "agnostic", "boot2"))}, series_b="same"),
                                       "G_sym|DP|A", True, lambda t: bool(t.get("used_for_p_dp_only"))),
    # a cut of an arm OUTSIDE the pair (d44 is not in G_sym) is not a truncation of G_sym
    "S_h2_cut_arm_outside_pair": (_syn_docs(1.0, {"DP2": 1.0, "DP4": ("undef", "agnostic:" + _LO_REFUSED), "DP8": ("undef", "agnostic:" + _LO_REFUSED),
                                                  "TP2": 1.04, "TP4": 1.08, "TP8": 1.12}, {2: 1.0, 4: 1.0, 8: 1.0},
                                  drivers={4: _syn_cut(("DP4", "d44", "boot2")), 8: _syn_cut(("DP8", "d44", "boot2"))}, series_b="same"),
                                  "G_sym|TP|A", "G_INCREASES_WITH_N", _lab),
    # H-2 (control side): the same gap for a NON-truncation reason (no driver cut record) keeps the P9-c path (disclose, then cite)
    "S_h2_nontruncation_dp_gap_cites_tp": (_syn_docs(1.0, {"DP2": 1.0, "DP4": ("undef", "agnostic:KNEE_NOT_BRACKETED"), "DP8": ("undef", "agnostic:KNEE_NOT_BRACKETED"),
                                                           "TP2": 1.04, "TP4": 1.08, "TP8": 1.12}, {2: 1.0, 4: 1.0, 8: 1.0}, series_b="same"),
                                           "G_sym|TP|A", "G_INCREASES_WITH_N", _lab),
    # rev11 M-4 (user decision, fail-closed): the verdict-audit reversal scenario.  Anchors J2 1.00 CLEAR, J4 1.10 FIRED, J8 1.21 FIRED;
    # DP-A 1,1,1,1 / DP-B 1,1,1.10,1.21 / TP-A 1,1.2,1.5,2.0.  The cross-shape (shape-A anchor) correction turns DP-B into TREND_JOB_SENSITIVE,
    # but P-DP reads the RAW shape-B trend -> BROKEN (MGP-13 fires), not UNTESTED.
    "S_m4_shape_b_raw_increase_cross_shape_corrected": (
        _syn_docs(1.0, {"DP2": 1.0, "DP4": 1.0, "DP8": 1.0, "TP2": 1.2, "TP4": 1.5, "TP8": 2.0}, {2: 1.00, 4: 1.10, 8: 1.21},
                  series_b={"DP2": 1.0, "DP4": 1.10, "DP8": 1.21}),
        None, ("BROKEN", True), lambda d: (d["P_DP"]["status"], d["P_DP"]["MGP13_fired"])),
    # rev11 D-A (harness delta audit): A HELD + B TJS with RAW NO_TREND -> UNTESTED, TP carries 'P-DP 미검정'.  Anchors J2 1.00 CLEAR, J4 0.90 FIRED,
    # J8 0.80 FIRED; DP-A 1,1,.95,.90 (raw NO_TREND under seed noise), DP-B flat (raw NO_TREND, 3 % magnitude rule fails -> TJS), TP-A 1,1.05,1.10,1.20.
    "S_da_shape_b_tjs_raw_no_trend_is_untested": (
        _syn_noisy(_syn_docs(1.0, {"DP2": 1.0, "DP4": .95, "DP8": .90, "TP2": 1.05, "TP4": 1.10, "TP8": 1.20}, {2: 1.00, 4: .90, 8: .80},
                             series_b={"DP2": 1.0, "DP4": 1.0, "DP8": 1.0}), 0.15, 4),
        None, ("UNTESTED", True), lambda d: (d["P_DP"]["status"], "P-DP 미검정" in d["trends"]["G_sym|TP|A"]["disclosures"])),
    # rev11 D-C: NOT_ATTEMPTED_EXT_INCOMPLETE with NO hard-cap record on the config's extension boot (the extension is incomplete because a boot
    # failed to START) is not a truncation -> P9-c path: the DP gap keeps the TP direction citable (with the disclosure).
    "S_dc_ext_incomplete_start_failure_is_not_truncation": (
        _syn_docs(1.0, {"DP2": 1.0, "DP4": ("undef", "agnostic:" + _LO_REFUSED), "DP8": ("undef", "agnostic:" + _LO_REFUSED),
                        "TP2": 1.04, "TP4": 1.08, "TP8": 1.12}, {2: 1.0, 4: 1.0, 8: 1.0},
                  drivers={4: _syn_cut(("DP4", "agnostic", "boot2"), reason="NOT_ATTEMPTED_EXT_INCOMPLETE"),
                           8: _syn_cut(("DP8", "agnostic", "boot2"), reason="NOT_ATTEMPTED_EXT_INCOMPLETE")}, series_b="same"),
        "G_sym|TP|A", "G_INCREASES_WITH_N", _lab),
    # ... and the same EXT_INCOMPLETE WITH a hard-cap record on the same config's extension boot (NOT_ATTEMPTED_BUDGET of boot1ext) IS a truncation
    "S_dc_ext_incomplete_with_ext_boot_cut_is_truncation": (
        _syn_docs(1.0, {"DP2": 1.0, "DP4": ("undef", "agnostic:" + _LO_REFUSED), "DP8": ("undef", "agnostic:" + _LO_REFUSED),
                        "TP2": 1.04, "TP4": 1.08, "TP8": 1.12}, {2: 1.0, 4: 1.0, 8: 1.0},
                  drivers={4: {"not_attempted": _syn_cut(("DP4", "fused_sym", "boot1ext"))["not_attempted"]
                                                + _syn_cut(("DP4", "agnostic", "boot2"), reason="NOT_ATTEMPTED_EXT_INCOMPLETE")["not_attempted"]},
                           8: {"not_attempted": _syn_cut(("DP8", "fused_sym", "boot1ext"))["not_attempted"]
                                                + _syn_cut(("DP8", "agnostic", "boot2"), reason="NOT_ATTEMPTED_EXT_INCOMPLETE")["not_attempted"]}}, series_b="same"),
        "G_sym|TP|A", "TP_UNCONTROLLED_TRUNCATION", _lab),
}
JOB_MUT_TAGS = {"i": "i", "iii": "iii", "iv": "iv", "h3": "h3", "h4": "h4", "h2_ignore_truncation": "trunc", "h2_reason_text": "trunc_reason",
                "m4_same_shape_only": "same_shape_only", "da_raw_unconditional": "raw_unconditional",
                "dc_ext_unconditional": "ext_unconditional"}

# H-1: per-shape P-DP.  name -> (inputs {shape: (label, g1_defined)}, expected status)
PDP_SCENARIOS = {
    "P_A_job_sensitive_B_citable_increase": ({"A": ("TREND_JOB_SENSITIVE", True), "B": ("G_INCREASES_WITH_N", True)}, "BROKEN"),
    "P_A_g1_undefined_B_increase": ({"A": (G1_UNDEFINED, False), "B": ("G_INCREASES_WITH_N", True)}, "BROKEN"),
    "P_both_no_trend": ({"A": ("NO_TREND_DETECTED", True), "B": ("NO_TREND_DETECTED", True)}, "HELD"),
    "P_A_no_trend_B_job_sensitive": ({"A": ("NO_TREND_DETECTED", True), "B": ("TREND_JOB_SENSITIVE", True, "NO_TREND_DETECTED")}, "UNTESTED"),
    # D-A: raw NO_TREND under TJS stays UNTESTED; a raw label that is itself TREND_UNDEFINED likewise
    "P_A_no_trend_B_tjs_raw_undefined": ({"A": ("NO_TREND_DETECTED", True), "B": ("TREND_JOB_SENSITIVE", True, "TREND_UNDEFINED")}, "UNTESTED"),
    "P_A_undefined_B_no_trend": ({"A": ("TREND_UNDEFINED", True), "B": ("NO_TREND_DETECTED", True)}, "UNTESTED"),
    # rev11 M-4: shape B corrected to TREND_JOB_SENSITIVE by the cross-shape correction, raw directional -> BROKEN; shape A keeps its same-shape correction
    "P_B_raw_increase_corrected_job_sensitive": ({"A": ("NO_TREND_DETECTED", True), "B": ("TREND_JOB_SENSITIVE", True, "G_INCREASES_WITH_N")}, "BROKEN"),
    "P_A_raw_increase_corrected_job_sensitive": ({"A": ("TREND_JOB_SENSITIVE", True, "G_INCREASES_WITH_N"), "B": ("NO_TREND_DETECTED", True)}, "UNTESTED"),
}

# H-5: rule (iii) removed, rule (ii) kept.  name -> (arm facts, expected memory_mismatch)
def _mem_arms(mt):
    return {a: dict(mamba=[48], max_total=[v], eff_cap=48) for a, v in mt.items()}


MEM_SCENARIOS = {
    "M_iii_spread_only": (_mem_arms({"fused": 10_000_000, "fused_sym": 10_000_000, "agnostic": 9_866_000, "d44": 9_866_000}), False),   # 1.34 % spread
    "M_ii_kv_bound_violated": (_mem_arms({"fused": 10_000, "fused_sym": 10_000, "agnostic": 10_000, "d44": 10_000}), True),
}


def run_job_mutation_checks():
    """Each rev9/rev10 mutant must DIFFER from the expected outcome on >= 1 scenario (KILLED); the unmutated control must match on ALL.
    i  G(1)-undefined direction cited · ii Job-effect reference = extension rung · iii correction by another Job's anchor ·
    iv corrected trend without the 3 % rule · h3 correction ratio from the extension-rung G(1) · h4 raw value substituted for anchor-UNTESTED
    points · h2_ignore_truncation TP cited despite truncation · untested_first (H-1) · mem_iii_present / mem_ii_removed (H-5) ·
    (v gate-name partial match: mg_substrate.run_mutation_checks; config-order initial [DP, TP]: mg_plan selftest)."""
    per, verdict = {}, {}
    for name, (docs, key, expected, fn) in JOB_MUT_SCENARIOS.items():
        pick = (lambda d, key=key: d["trends"][key]) if key else (lambda d: d)         # key None: the extractor reads the whole joined document
        row = {"expected": expected, "CONTROL": fn(pick(aggregate_jobs(docs)))}
        for tag, m in JOB_MUT_TAGS.items():
            row[tag] = fn(pick(aggregate_jobs(docs, _mut=m)))
        per[name] = row
    ref, anc = _syn_ref(1.0, final_mult=1.3), _syn_anchor(1.0)
    # rev11 D-B: --jobs must name every Job; an omission is the explicit <J>=NOT_SUBMITTED, never silent
    def _join(kvs, mut=None):
        try:
            return "ACCEPTED" if parse_jobs_args(kvs, _mut=mut) else "ACCEPTED"
        except JoinRefused:
            return "REFUSED"
    sub = ["J1=a", "J2=b", "J4=c"]
    per["S_db_silent_subset_refused"] = {"expected": "REFUSED", "CONTROL": _join(sub), "db_subset_ok": _join(sub, "subset_ok")}
    per["S_db_explicit_omission_accepted"] = {"expected": "ACCEPTED", "CONTROL": _join(sub + ["J8=NOT_SUBMITTED"]),
                                              "db_subset_ok": _join(sub + ["J8=NOT_SUBMITTED"], "subset_ok")}
    per["S_db_full_set_accepted"] = {"expected": "ACCEPTED", "CONTROL": _join(sub + ["J8=d"]), "db_subset_ok": _join(sub + ["J8=d"], "subset_ok")}
    per["S_ii_extension_rung_reference"] = {"expected": "CLEAR", "CONTROL": job_effect(ref, anc)["status"], "ii": job_effect(ref, anc, _mut="ii")["status"]}
    for nm, kv in (("not_saturated", dict(boot1_hi_label="HI_NOT_SATURATED")), ("harness_limited", dict(boot1_harness_limited=True)), ("client_error", dict(boot1_client_error=True))):
        per[f"S_h3_reference_cell_{nm}"] = {"expected": "UNTESTED", "CONTROL": job_effect(_syn_ref(1.0, fused_sym=kv), _syn_anchor(1.0))["status"]}
    # rev11 M-2: J1 boot1 cells are HARNESS_LIMITED / CLIENT_ERROR, the FINAL (extension-rung) set is clean -> UNTESTED; reading the final set -> CLEAR
    for nm, kv in (("harness_limited", dict(boot1_harness_limited=True)), ("client_error", dict(boot1_client_error=True))):
        rf = _syn_ref(1.0, fused_sym=dict(kv, harness_limited=False, client_error=False), agnostic=dict(kv, harness_limited=False, client_error=False))
        per[f"S_m2_boot1_flag_{nm}_final_set_clean"] = {"expected": "UNTESTED", "CONTROL": job_effect(rf, _syn_anchor(1.0))["status"],
                                                        "m2_flags_final": job_effect(rf, _syn_anchor(1.0), _mut="flags_final")["status"]}
    for name, (inp, expected) in PDP_SCENARIOS.items():
        tr = {}
        for sh, v in inp.items():
            tr[f"G_sym|DP|{sh}"] = {"label": v[0], "g1_defined": v[1], **({"raw_label": v[2]} if len(v) > 2 else {})}
        per[name] = {"expected": expected, "CONTROL": p_dp_status(tr)["status"], "untested_first": p_dp_status(tr, _mut="untested_first")["status"],
                     "m4_same_shape_only": p_dp_status(tr, _mut="same_shape_only")["status"],
                     "da_raw_unconditional": p_dp_status(tr, _mut="raw_unconditional")["status"]}
    for name, (facts, expected) in MEM_SCENARIOS.items():
        cf = memory_check(facts)
        row = {"expected": expected, "CONTROL": cf["memory_mismatch"], "mem_iii_present": memory_check(facts, _mut="iii")["memory_mismatch"],
               "mem_ii_removed": memory_check(facts, _mut="ii")["memory_mismatch"]}
        if name == "M_iii_spread_only":
            row["CONTROL"] = (cf["memory_mismatch"], cf["MEMORY_RULE_III_COUNTERFACTUAL"] == {"denominator_fused_sym": True, "denominator_min_arm": True})
            row["expected"] = (False, True)
            row["mem_iii_present"] = (row["mem_iii_present"], True)
            row["mem_ii_removed"] = (row["mem_ii_removed"], True)
        per[name] = row
    for m in ("i", "ii", "iii", "iv", "h3", "h4", "h2_ignore_truncation", "h2_reason_text", "m4_same_shape_only", "da_raw_unconditional", "dc_ext_unconditional", "db_subset_ok", "m2_flags_final", "untested_first",
              "mem_iii_present", "mem_ii_removed", "CONTROL"):
        killed_on = [n for n, r in per.items() if m in r and r[m] != r["expected"]]
        verdict[m] = {"killed_on": killed_on, "killed": bool(killed_on)}
    ok = all(v["killed"] for k, v in verdict.items() if k != "CONTROL") and not verdict["CONTROL"]["killed"]
    return ok, verdict, per


# =========================================================================== CLI + selftest
def find_cells(root):
    return sorted(p.parent for p in Path(root).rglob("cell.json"))


def _round_of(steps):
    rs = [int(st.get("round", 1)) for st in steps or [] if st.get("type") == "cell"]
    return max(rs) if rs else 1


def build_config_flags(root, cells, excluded=()):
    """Per config flags from the boot artifacts (F5, F11, F14).

    Scope (F11): a boot belongs to a (config, round, block) group -- block main / anchor / gpu_dev / numa / nsys -- taken from the
    driver's specs/<boot>.json (fallback: its cells).  Flags are OR-summed WITHIN a group and never leak ACROSS groups: a dirty
    gpu_dev GPU, a NUMA boot or the round-2 anchor cannot erase the round-1 main G of the same config.  The top level of
    `flags[cfg]` is the (round 1, block main) group -- the one the estimands read; every group is under `by_group`.
    memory_mismatch = G5(c) over the arms that were expected (not M2g-excluded); an expected arm without boot facts is listed in
    `memory_not_tested_arms` (NOT_TESTED, distinct from FAIL; the M2g table carries its own NOT_TESTED=MEMORY_MISMATCH row).
    contam_unverified (NVML unavailable at the pre-boot check) is a disclosure, not an exclusion."""
    root = Path(root)
    boots = {}
    spec_dir = root / "specs"
    if spec_dir.exists():
        for sp in sorted(spec_dir.glob("*.json")):
            try:
                sd = json.loads(sp.read_text())
            except ValueError:
                continue
            boots[sd["boot_id"]] = dict(config=sd["config"], arm=sd["arm"], block=sd.get("block", "main"),
                                        round=_round_of(sd.get("steps")))
    for c in cells:
        m = c["meta"]
        bid = Path(m["boot_dir"]).name if m.get("boot_dir") else None
        if bid and bid not in boots:
            boots[bid] = dict(config=m["config"], arm=m["arm"], block=m.get("block", "main"), round=int(m.get("round", 1)))
    groups = {}
    for bid, b in boots.items():
        bd = root / "boots" / bid
        g = groups.setdefault((b["config"], f"r{b['round']}/{b['block']}"),
                              {"contaminated": False, "contam_boots": [], "contam_arms": [], "contam_unverified": False, "unverified_boots": [],
                               "arms": {}})
        cx = bd / "contam.json"
        if cx.exists() and json.loads(cx.read_text()).get("gpu_contaminated"):
            g["contaminated"] = True
            g["contam_boots"].append(bid)
            if b["arm"] not in g["contam_arms"]:
                g["contam_arms"].append(b["arm"])
        unv = False
        cpre = bd / "contam_pre.json"
        if cpre.exists():
            v = (json.loads(cpre.read_text()).get("verdict") or {})
            unv = bool(v.get("contam_unverified") or v.get("verified") is False)
        if (bd / "boot.json").exists() and json.loads((bd / "boot.json").read_text()).get("contam_status") == "CONTAM_UNVERIFIED":
            unv = True
        if unv:
            g["contam_unverified"] = True
            g["unverified_boots"].append(bid)
        cm = bd / "cell_meta.json"
        if cm.exists() and b["block"] in ("main", "anchor"):
            f = json.loads(cm.read_text())
            si = f.get("server_info") or {}
            eff = f.get("max_running_requests_requested") or si.get("max_running_requests")
            a = g["arms"].setdefault(b["arm"], dict(mamba=[], max_total=[], eff_cap=eff))
            a["mamba"] = a["mamba"] + list(f.get("max_mamba_cache_size", []) or [])
            a["max_total"] = a["max_total"] + list(f.get("max_total_num_tokens", []) or [])
            a["eff_cap"] = a["eff_cap"] if a["eff_cap"] is not None else eff
    flags = {}
    for (cfg, gname), g in sorted(groups.items()):
        arms = {a: v for a, v in g["arms"].items() if v["mamba"] or v["max_total"] or v["eff_cap"] is not None}
        mem = memory_check(arms) if arms else {"memory_mismatch": False, "reasons": []}
        expected = [a for a in C.ARMS if f"{cfg}:{a}" not in set(excluded)]
        not_tested = [a for a in expected if a not in arms] if gname.endswith("/main") else []
        gf = {"memory_mismatch": mem["memory_mismatch"], "reasons": mem["reasons"],
              "arm_rel_diff": mem.get("arm_rel_diff"), "MEMORY_RULE_III_COUNTERFACTUAL": mem.get("MEMORY_RULE_III_COUNTERFACTUAL"),
              "rule_iii_would_have_flagged": bool(mem.get("rule_iii_would_have_flagged")),
              "memory_not_tested_arms": not_tested, "gpu_contaminated": g["contaminated"], "contam_boots": g["contam_boots"],
              "contam_arms": sorted(g["contam_arms"]),
              "contam_unverified": g["contam_unverified"], "unverified_boots": g["unverified_boots"]}
        fl = flags.setdefault(cfg, {"by_group": {}})
        fl["by_group"][gname] = gf
    for cfg, fl in flags.items():
        main = fl["by_group"].get("r1/main")
        if main:
            fl.update({k: v for k, v in main.items()})
        else:
            fl.update(memory_mismatch=False, reasons=[], memory_not_tested_arms=[], gpu_contaminated=False, contam_boots=[], contam_arms=[],
                      contam_unverified=False, unverified_boots=[])
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
    ok2, verdict2, per2 = run_job_mutation_checks()
    for m, v in verdict2.items():
        print(f"job-layer mutant {m:22s} killed={v['killed']} on {v['killed_on']}")
    if not ok2:
        fails.append("job-layer mutation checks")
    t = trend([(1, [2, 2, 2, 2], [1, 1, 1, 1]), (2, [2, 2, 2, 2], [1, 1, 1, 1]), (4, [2, 2, 2, 2], [1, 1, 1, 1])])
    assert t["label"] == "NO_TREND_DETECTED" and t["magnitude"] == "DIRECTION_NOT_CITABLE_UNDER_3PCT"
    t = trend([(1, [1, 1, 1, 1.01], [1, 1, 1, 1]), (2, [1.5, 1.5, 1.5, 1.5], [1, 1, 1, 1]),
               (4, [2, 2, 2, 2.02], [1, 1, 1, 1])])
    assert t["label"] == "G_INCREASES_WITH_N", t
    assert trend([(2, [1] * 4, [1] * 4), (4, [1] * 4, [1] * 4)])["label"] == "TREND_UNDEFINED"
    assert trend([(2, [1] * 4, [1] * 4), (4, [1] * 4, [1] * 4), (8, [1] * 4, [1] * 4)])["label"] == G1_UNDEFINED
    print("selftest:", "PASS" if not fails else f"FAIL {fails}")
    return 0 if not fails else 1


class JoinRefused(Exception):
    """rev11 M-3: a Job directory is not admissible to the cross-Job join (no / unverified pass table, wrong Job, aborted)."""


def _sha256_file(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def job_dir_admission(J, d, committed_sha_dir=None, _mut=None):
    """rev11 M-3 (decision: the `--jobs` join re-loads each Job's own M2G_PASS_TABLE.used.json -- the table the Job itself ran against -- and
    labels the Job with it; it does NOT consume LABELS.json).  Admission, all of which must hold:
      * `<d>/BUDGET.txt` exists and its `job=` field equals the CLI's J (MGH-29: the Job name carries no J; BUDGET.txt `job=` is the attribution),
        and `kind=m3`;
      * no `<d>/ABORT`, no `<d>/DRYRUN` marker;
      * `<d>/M2G_PASS_TABLE.used.json` exists and its sha256 equals the COMMITTED `results/multigpu/M2G_PASS_TABLE_<J>.json.sha256`
        (`committed_sha_dir`, default this directory) -- a table that was edited after the Job, or the wrong Job's table, is refused;
      * the table parses and has a `cells` dict;
      * (rev11 D-B) a missing `driver` record in `<d>/LABELS.json` is reconstructed, not refused (reconstruct_driver).
    Returns (path of the used table, parsed table).  `_mut` 'no_table' (tests only) skips the table (the rev10 behaviour)."""
    d = Path(d)
    if not d.is_dir():
        raise JoinRefused(f"{J}: {d} is not a directory")
    if (d / "ABORT").exists():
        raise JoinRefused(f"{J}: {d}/ABORT present ({(d / 'ABORT').read_text().strip()[:120]})")
    if (d / "DRYRUN").exists():
        raise JoinRefused(f"{J}: {d}/DRYRUN present (a dry run is not a measurement)")
    bt = d / "BUDGET.txt"
    if not bt.exists():
        raise JoinRefused(f"{J}: {d}/BUDGET.txt missing (cannot attribute the directory to a Job)")
    kv = dict(tok.split("=", 1) for tok in bt.read_text().split() if "=" in tok)
    if kv.get("kind") != "m3":
        raise JoinRefused(f"{J}: BUDGET.txt kind={kv.get('kind')!r}, an M3 Job directory is required")
    if kv.get("job") != J:
        raise JoinRefused(f"{J}: BUDGET.txt job={kv.get('job')!r} != CLI {J}")
    if _mut == "require_driver":                 # mutant (D-B): the rev11 refusal of a Job directory without a driver record
        try:
            has_driver = isinstance(json.loads((d / "LABELS.json").read_text()).get("driver"), dict)
        except (OSError, ValueError):
            has_driver = False
        if not has_driver:
            raise JoinRefused(f"{J}: {d}/LABELS.json missing or without a `driver` record")
    # rev11 D-B: no driver record is NOT a refusal -- label_job_dir RECONSTRUCTS it from DRIVER_LOG.jsonl + boots/ + specs/ (reconstruct_driver)
    if _mut == "no_table":
        return None, {}
    used = d / "M2G_PASS_TABLE.used.json"
    if not used.exists():
        raise JoinRefused(f"{J}: {used} missing -- no pass table, no join")
    csd = Path(committed_sha_dir) if committed_sha_dir else HERE
    cs = csd / f"M2G_PASS_TABLE_{J}.json.sha256"
    if not cs.exists():
        raise JoinRefused(f"{J}: committed {cs} missing -- cannot verify the used table")
    want = cs.read_text().split()[:1]
    got = _sha256_file(used)
    if want != [got]:
        raise JoinRefused(f"{J}: sha256(M2G_PASS_TABLE.used.json)={got[:16]}.. != committed .sha256 {(want or ['<empty>'])[0][:16]}..")
    try:
        t = json.loads(used.read_text())
        assert isinstance(t.get("cells"), dict)
    except (ValueError, AssertionError):
        raise JoinRefused(f"{J}: {used} is not a pass table (no `cells` dict)")
    return used, t


NOT_JOINED_TOKENS = ("NOT_SUBMITTED",)


def parse_jobs_args(kvs, _mut=None):
    """rev11 D-B: `--jobs` must name EVERY registered Job (J1, J2, J4, J8) exactly once.  A Job may be left out only by an explicit argument
    `J8=NOT_SUBMITTED` (its value is a token, not a directory); the join then reports JOB_NOT_JOINED.  A silent subset is refused: a trend over
    {1,2,4} is not the registered {1,2,4,8} trend (rev11 audit D-B: DP-A 1,1,1,1.10 BROKEN became 1,1,1 HELD).
    Returns ({J: dir}, {J: token}).  `_mut` 'subset_ok' (tests only) = the rev11 behaviour (any subset accepted)."""
    dirs, nj = {}, {}
    for kv in kvs:
        if "=" not in kv:
            raise JoinRefused(f"bad or duplicate Job argument {kv!r}")
        J, d = kv.split("=", 1)
        if J not in C.JOBS or J in dirs or J in nj:
            raise JoinRefused(f"bad or duplicate Job id {J!r}")
        if d in NOT_JOINED_TOKENS:
            nj[J] = d
        else:
            dirs[J] = d
    missing = [J for J in C.JOBS if J not in dirs and J not in nj]
    if missing and _mut != "subset_ok":
        raise JoinRefused(f"--jobs omits {missing}: every Job of {list(C.JOBS)} is required; an intended omission is the explicit argument "
                          f"<J>=NOT_SUBMITTED")
    return dirs, nj


def label_job_dir(root, job=None, pass_table=None, _mut=None):
    """One Job directory -> aggregate() document (+ covariates).  pass_table: path or None."""
    cells = [load_cell(d) for d in find_cells(root)]
    pt = json.loads(Path(pass_table).read_text()) if pass_table else {}
    ptc = pt.get("cells") if isinstance(pt, dict) and "cells" in pt else pt
    excl = {k for k, v in (ptc or {}).items() if isinstance(v, dict) and not v.get("included", True)}
    excl_arms = set(excl)
    doc = aggregate(cells, ptc, build_config_flags(root, cells, excl), pt.get("configs") if isinstance(pt, dict) else None, job=job)
    doc["n_cells"] = len(cells)
    doc["covariates"] = job_covariates(root)
    drv = Path(root) / "LABELS.json"
    if drv.exists():
        try:
            doc["driver"] = json.loads(drv.read_text()).get("driver")
        except ValueError:
            pass
    if not isinstance(doc.get("driver"), dict):                                          # D-B: the driver died before finish() -> reconstruct (fail-closed)
        doc["driver"] = reconstruct_driver(root, job or doc.get("job") or "J2", excluded=excl_arms)
        if any(ev.get("status") == "DRIVER_EXCEPTION_FINISH_IN_FINALLY" for ev in _read_driver_log(root)):
            doc["driver"]["exception"] = True                                            # D-D R-1: finish() died before writing LABELS.json -> still disclose
    elif isinstance(doc.get("driver"), dict) and (doc["driver"].get("exception") or any(ev.get("status") == "DRIVER_EXCEPTION_FINISH_IN_FINALLY" for ev in _read_driver_log(root))):
        # rev11 D-D: finish() written from `finally` after an exception is NOT a complete record -- boots the driver never reached are absent from not_attempted.
        # Reconstruct and ADD (never replace) the RECONSTRUCTED_NO_BOOT_RECORD entries, de-duplicated by boot_id.
        rec = reconstruct_driver(root, job or doc.get("job") or "J2", excluded=excl_arms, _mut=("no_missing_boot_cut" if _mut == "no_missing_boot_cut" else None))
        have = {x.get("boot_id") for x in doc["driver"].get("not_attempted") or [] if x.get("boot_id")}
        add = []
        for x in rec["not_attempted"]:
            if x.get("reason") == "RECONSTRUCTED_NO_BOOT_RECORD" and x.get("boot_id") not in have:
                have.add(x["boot_id"])
                add.append(x)
        if _mut != "trust_finally":
            doc["driver"]["not_attempted"] = list(doc["driver"].get("not_attempted") or []) + add
        doc["driver"]["exception"] = True
        doc["driver"]["exception_merged_boot_ids"] = [x["boot_id"] for x in add]
    if isinstance(doc.get("driver"), dict) and "partial_boots" not in doc["driver"]:     # M-1: older LABELS.json -> recompute from the boot records
        doc["driver"]["partial_boots"] = partial_boots_of(root)
    return doc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", help="ONE Job directory scanned recursively for cell.json")
    ap.add_argument("--job", default=None, choices=C.JOBS)
    ap.add_argument("--jobs", nargs="*", default=[], help="J1=<dir> J2=<dir> ... : join the Jobs (aggregate_jobs)")
    ap.add_argument("--pass-table", default="")
    ap.add_argument("--committed-sha-dir", default="", help="--jobs: directory holding the committed M2G_PASS_TABLE_<J>.json.sha256 (default: results/multigpu)")
    ap.add_argument("--out", default="")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.jobs:
        docs, meta = {}, {}
        try:
            dirs, nj = parse_jobs_args(a.jobs)
        except JoinRefused as e:
            if "bad or duplicate" in str(e):
                ap.error(f"--jobs: {e}")
            sys.stderr.write(f"JOIN_REFUSED {e}\n")
            return 3
        for J, d in dirs.items():
            try:
                used, _t = job_dir_admission(J, d, a.committed_sha_dir or None)
            except JoinRefused as e:
                sys.stderr.write(f"JOIN_REFUSED {e}\n")
                return 3
            docs[J] = label_job_dir(d, J, used)
            meta[J] = {k: v for k, v in docs[J]["covariates"].items()}
        doc = aggregate_jobs(docs, meta, not_joined=nj)
        if a.out:
            Path(a.out).write_text(json.dumps(doc, indent=1, sort_keys=True, default=float) + "\n")
        sys.stdout.write(render_jobs_text(doc))
        return 0
    if not a.root:
        ap.error("--root, --jobs or --selftest")
    doc = label_job_dir(a.root, a.job, a.pass_table or None)
    if a.out:
        Path(a.out).write_text(json.dumps(doc, indent=1, sort_keys=True, default=float) + "\n")
    sys.stdout.write(render_text(doc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
