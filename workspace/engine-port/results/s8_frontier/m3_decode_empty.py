#!/usr/bin/env python3
"""m3_decode_empty.py -- DIAGNOSTIC: where does decode-empty time come from,
and does it explain the E1_DECODE_REALIZED dilution?  (job 872077)

★★ READ THIS BEFORE USING ANY NUMBER THIS SCRIPT PRINTS ★★

METHODOLOGY GATE #6 (do not use an identity as evidence) -- FIELD AUDIT.
Verified in code and re-verified on every file this script reads (the audit is
re-run per file and printed; it is not a claim, it is a check):

  * `decode_ready_queue_depth`  == 0 IDENTICALLY.  `DecodeWorker.ready_queue`
    (multiplex/dual_worker.py:104,110) is only ever appended to inside
    `_dual_worker_prefill_ready`, which is guarded by
    `if getattr(self, "dual_worker_enabled", False)`
    (multiplex/multiplexing_mixin.py:490-492).  Job 872077 ran with
    PDMUX_DUAL_WORKER unset (architecture=="legacy" in every record), so the
    list is never filled and the field is a constant 0.
    ==> THE STATED PRIMARY PREDICTOR OF H-STARVE IS NOT INSTRUMENTED.
        "decode work is ready but not running" cannot be read off this
        telemetry, in either direction.  Any decomposition that treats
        `decode_ready_queue_depth == 0` as *evidence for* H-arrival is
        circular.  This script therefore reports H-starve as UNMEASURABLE and
        decomposes only what is independently observable.
  * `active_decode_sequences`   == `decode_running_batch_size` IDENTICALLY
    (both are `running_batch.batch_size()` computed in the same payload build:
    multiplexing_mixin.py:238 and dual_worker.py:110/617).  Zero information
    about "is there decode work that is not in the batch".
  * `running_batch_occupancy`   == min(1, decode_running_batch_size / cap).
  * `decode_idle_ratio`, `prefill_idle_ratio` == 0.0 IDENTICALLY (declared with
    a default in controller.py:55 and never assigned).
  * `prefill_admission_blocked` == (prefill_queue_depth > 0 AND
    prefill_active_batch_size == 0) exactly (multiplexing_mixin.py, the
    `observe_scheduler` post-step).  It is a *derived* flag, not a new
    observation; and inside the decode-empty population its `reason` string is
    forced to "scheduler_admission_pending" because the alternative branch
    tests `running_batch.batch_is_full`, which is False when decode is empty.

  SURVIVING INDEPENDENT SIGNALS while decode is empty:
    prefill_active_batch_size, prefill_queue_depth, oldest_prefill_age_ms,
    kv_*_occupancy, decode_iterations, timestamps.

AGGREGATION UNIT (methodology gate #5).  Snapshots are a COUNT subsample of
scheduler sync calls (PDMUX_DUAL_WORKER_TRACE_EVERY=32,
multiplexing_mixin.py:91), so the sampling grid is dense in wall-clock exactly
where sync calls are cheap and sparse where a long GPU op runs between syncs.
The estimand here is "what FRACTION OF TIME is decode empty, and why", so the
count grid is biased against long stalls by construction.  TIME-WEIGHTED
(sum of dt to the next snapshot) is the matching unit; EPISODE counts answer a
different, also-useful question ("how many stalls, how long each").  All three
are computed and printed so the reader can see whether the unit changes the
answer.  Interval dt statistics are printed for the same reason.

Reuses the conventions of e1_pin_check.py verbatim: `event=="runtime_snapshot"`
and `phase=="benchmark"` only, sorted by `timestamp_monotonic_s`, dt = gap to
the next retained snapshot, last snapshot dropped (no dt).

ANALYSIS WINDOW (gate #5, second guise).  The telemetry file spans SERVER
lifetime, not LOAD lifetime.  `bench_serving` is launched with
`warmup_requests=1`; that warmup request flips `pdmux_experiment_phase` to
"benchmark" (multiplexing_mixin.py:387-398) and THEN the client spends 12-31 s
tokenizing/preparing the ShareGPT dataset before the real 200-prompt load
starts, and the server keeps tracing 3-7 s after the last completion until it
is killed.  Both stretches are decode-empty AND prefill-empty and they dominate
raw decode-empty time.  Measured on the full file they would answer
"why is decode empty?" with "because the client was not sending anything",
which is true and useless.  So every decode-empty statistic here is reported
BOTH over the full file and over the LOAD WINDOW (leading/trailing decode-empty
episodes of >= MIN_EDGE_S trimmed), and the load window is cross-validated
against the client's own reported `duration` in the per-request jsonl.

Usage:
  python3 m3_decode_empty.py [--job 872077] [--csv out.csv]
"""

from __future__ import annotations

import glob
import json
import math
import os
import re
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = 48  # --max-running-requests as run (e1_m3_control.sbatch)

# stream groups as configured for this campaign: idx0=(108,0) prefill-only /
# idle, idx1=(108-D, D) the cell's target split, idx2=(0,108) decode-only.
FNAME_RE = re.compile(r"^e1m3_(?P<arm>[A-Za-z0-9]+)_d(?P<d>\d+)_b(?P<block>\d+)_(?P<job>\d+)_telemetry\.jsonl$")

FIELDS = (
    "timestamp_monotonic_s", "prefill_sms", "decode_sms", "stream_index",
    "decode_running_batch_size", "active_decode_sequences",
    "decode_ready_queue_depth", "decode_idle_ratio", "running_batch_occupancy",
    "prefill_active_batch_size", "prefill_queue_depth", "prefill_queue_age_ms",
    "oldest_prefill_age_ms", "prefill_admission_blocked",
    "prefill_admission_block_reason", "kv_occupancy", "decode_iterations",
)


def load_rows(path):
    """Return (rows, audit).  rows = list of dicts sorted by monotonic ts."""
    rows = []
    audit = dict(n=0, drqd_nonzero=0, ads_ne_drb=0, didle_nonzero=0,
                 rbo_ne_ident=0, adm_ne_ident=0, arch=set())
    for line in open(path):
        try:
            e = json.loads(line)
        except Exception:
            continue
        if e.get("event") != "runtime_snapshot" or e.get("phase") != "benchmark":
            continue
        audit["n"] += 1
        audit["arch"].add(e.get("architecture"))
        drb = int(e.get("decode_running_batch_size", 0) or 0)
        pab = int(e.get("prefill_active_batch_size", 0) or 0)
        pqd = int(e.get("prefill_queue_depth", 0) or 0)
        if int(e.get("decode_ready_queue_depth", 0) or 0) != 0:
            audit["drqd_nonzero"] += 1
        if int(e.get("active_decode_sequences", 0) or 0) != drb:
            audit["ads_ne_drb"] += 1
        if float(e.get("decode_idle_ratio", 0.0) or 0.0) != 0.0:
            audit["didle_nonzero"] += 1
        if abs(float(e.get("running_batch_occupancy", 0.0) or 0.0)
               - min(1.0, drb / CAP)) > 1e-9:
            audit["rbo_ne_ident"] += 1
        if bool(e.get("prefill_admission_blocked", False)) != (pqd > 0 and pab == 0):
            audit["adm_ne_ident"] += 1
        rows.append(dict(ts=float(e["timestamp_monotonic_s"]), drb=drb, pab=pab,
                         pqd=pqd, dsm=e.get("decode_sms"), psm=e.get("prefill_sms"),
                         idx=e.get("stream_index"),
                         age=float(e.get("oldest_prefill_age_ms", 0.0) or 0.0),
                         kv=float(e.get("kv_occupancy", 0.0) or 0.0),
                         it=int(e.get("decode_iterations", 0) or 0)))
    rows.sort(key=lambda r: r["ts"])
    return rows, audit


def pctl(a, q):
    if not a:
        return float("nan")
    a = sorted(a)
    p = q * (len(a) - 1)
    lo = int(p)
    hi = min(lo + 1, len(a) - 1)
    return a[lo] + (a[hi] - a[lo]) * (p - lo)


def classify(r):
    """EXCLUSIVE + EXHAUSTIVE class of one decode-empty snapshot interval.

    The partition is over the two independent 'is there work in the system'
    signals that survive the gate-#6 audit.  Naming is deliberately mechanistic,
    NOT hypothesis-labelled, because (see module docstring) the field that would
    adjudicate H-starve does not exist in this telemetry:

      A_no_work        : nothing prefilling, nothing waiting  -> consistent with
                         H-arrival (a genuine gap in demand) AND with "the last
                         decode finished and nothing has arrived".
      B_prefill_only   : a prefill batch is in flight, queue empty -> the GPU is
                         busy on prefill; no request has reached decode yet.
      C_prefill_queued : prefill in flight AND more requests waiting -> same,
                         under backlog.
      D_queue_stalled  : requests WAITING but NO prefill batch in flight and no
                         decode -> the engine is holding admission while nothing
                         runs.  This is the only class where work provably
                         exists and provably nothing is executing.
    """
    if r["pab"] > 0:
        return "C_prefill_queued" if r["pqd"] > 0 else "B_prefill_only"
    return "D_queue_stalled" if r["pqd"] > 0 else "A_no_work"


CLASSES = ("A_no_work", "B_prefill_only", "C_prefill_queued", "D_queue_stalled")


MIN_EDGE_S = 1.0   # an edge decode-empty stretch shorter than this is treated
                   # as a real in-load gap, not a window artifact.


def load_window(rows, client_duration=None):
    """★CLIENT-ANCHORED LOAD WINDOW.  The telemetry file spans server lifetime;
    the estimand is the served load.  Anchor on the LAST decode-busy snapshot
    (= last completion) and go back exactly `client_duration` seconds, the
    `duration` bench_serving itself reports (first send -> last completion).
    Both clocks are on the same node and only a DIFFERENCE is used, so no
    epoch alignment is needed -- this is the same anchoring technique
    e1_pin_check.compute_episode_gate uses for its `t1_anchor`.

    Falls back to trimming the leading/trailing decode-empty stretches when the
    client duration is unavailable.  Returns (i0, i1_exclusive, head_s, tail_s).
    """
    n = len(rows)
    if client_duration and client_duration == client_duration:
        j = n - 1
        while j >= 0 and rows[j]["drb"] == 0:
            j -= 1
        if j >= 1:
            t_end = rows[j]["ts"]
            t_start = t_end - client_duration
            i0 = 0
            while i0 < j and rows[i0]["ts"] < t_start:
                i0 += 1
            i1 = min(j + 2, n)
            return i0, i1, rows[i0]["ts"] - rows[0]["ts"], rows[-1]["ts"] - rows[j]["ts"]
    i0 = 0
    if rows[0]["drb"] == 0:
        j = 0
        while j < n and rows[j]["drb"] == 0:
            j += 1
        if j < n and (rows[j]["ts"] - rows[0]["ts"]) >= MIN_EDGE_S:
            i0 = j
    i1 = n
    if rows[-1]["drb"] == 0:
        j = n - 1
        while j >= 0 and rows[j]["drb"] == 0:
            j -= 1
        if j >= 0 and (rows[-1]["ts"] - rows[j]["ts"]) >= MIN_EDGE_S:
            i1 = min(j + 2, n)
    head = rows[i0]["ts"] - rows[0]["ts"] if i0 else 0.0
    tail = rows[-1]["ts"] - rows[min(i1, n) - 1]["ts"] if i1 < n else 0.0
    return i0, i1, head, tail


def analyze_file(path, target_d, client=None):
    all_rows, audit = load_rows(path)
    out = dict(path=os.path.basename(path), audit=audit, n_bench=len(all_rows))
    if len(all_rows) < 3:
        out["ok"] = False
        return out
    out["ok"] = True
    i0, i1, head_s, tail_s = load_window(
        all_rows, (client or {}).get("duration"))
    out["head_gap_s"] = head_s
    out["tail_gap_s"] = tail_s
    out["file_span_s"] = all_rows[-1]["ts"] - all_rows[0]["ts"]

    # full-file decode-empty time share (for the unit/window comparison)
    tf_e = tf_b = 0.0
    for i in range(len(all_rows) - 1):
        dt = max(0.0, all_rows[i + 1]["ts"] - all_rows[i]["ts"])
        if all_rows[i]["drb"] > 0:
            tf_b += dt
        else:
            tf_e += dt
    out["frac_time_dempty_FULLFILE"] = tf_e / (tf_e + tf_b) if (tf_e + tf_b) else float("nan")

    rows = all_rows[i0:i1]
    if len(rows) < 3:
        out["ok"] = False
        return out
    out["n_bench_window"] = len(rows)
    dts = [rows[i + 1]["ts"] - rows[i]["ts"] for i in range(len(rows) - 1)]
    out["dt_median_ms"] = pctl(dts, 0.50) * 1e3
    out["dt_p95_ms"] = pctl(dts, 0.95) * 1e3
    out["dt_max_ms"] = max(dts) * 1e3
    out["t_total_s"] = sum(dts)

    t_by_class = defaultdict(float)
    n_by_class = defaultdict(int)
    t_dempty = t_dbusy = 0.0
    n_dempty = n_dbusy = 0
    t_pa = t_overlap = t_decode_target = 0.0
    t_dbusy_pa = 0.0           # decode busy AND prefill active
    t_dbusy_dsm108 = 0.0
    drb_time_w = 0.0
    # literal H-starve probe (expected to be structurally impossible to see)
    n_starve_literal = 0
    # episodes
    episodes = []              # (duration, majority_class) for decode-empty
    pa_episodes = []           # prefill-active episode durations
    cur_ep = None
    cur_pa = None

    for i in range(len(rows) - 1):
        r = rows[i]
        dt = max(0.0, rows[i + 1]["ts"] - r["ts"])
        if r["pab"] > 0:
            t_pa += dt
        if r["drb"] > 0:
            t_dbusy += dt
            n_dbusy += 1
            drb_time_w += r["drb"] * dt
            if r["dsm"] == target_d:
                t_decode_target += dt
            if r["dsm"] == 108:
                t_dbusy_dsm108 += dt
            if r["pab"] > 0:
                t_dbusy_pa += dt
                t_overlap += dt
        else:
            t_dempty += dt
            n_dempty += 1
            c = classify(r)
            t_by_class[c] += dt
            n_by_class[c] += 1
            # literal starve: ready decode work reported while batch empty
            if r.get("drqd_nonzero_flag"):
                n_starve_literal += 1
        # decode-empty episodes
        if r["drb"] == 0:
            if cur_ep is None:
                cur_ep = dict(dur=0.0, cls=defaultdict(float))
            cur_ep["dur"] += dt
            cur_ep["cls"][classify(r)] += dt
        else:
            if cur_ep is not None:
                episodes.append(cur_ep)
                cur_ep = None
        # prefill-active episodes
        if r["pab"] > 0:
            cur_pa = (cur_pa or 0.0) + dt
        else:
            if cur_pa is not None:
                pa_episodes.append(cur_pa)
                cur_pa = None
    if cur_ep is not None:
        episodes.append(cur_ep)
    if cur_pa is not None:
        pa_episodes.append(cur_pa)

    out.update(
        t_dempty_s=t_dempty, t_dbusy_s=t_dbusy,
        frac_time_dempty=t_dempty / (t_dempty + t_dbusy) if (t_dempty + t_dbusy) else float("nan"),
        frac_count_dempty=n_dempty / (n_dempty + n_dbusy) if (n_dempty + n_dbusy) else float("nan"),
        n_dempty_snap=n_dempty, n_dbusy_snap=n_dbusy,
        t_prefill_active_s=t_pa,
        frac_time_prefill_active=t_pa / sum(dts) if dts else float("nan"),
        t_overlap_s=t_overlap,
        decode_realized_frac=(t_decode_target / t_dbusy) if t_dbusy else float("nan"),
        dbusy_prefill_active_frac=(t_dbusy_pa / t_dbusy) if t_dbusy else float("nan"),
        dbusy_dsm108_frac=(t_dbusy_dsm108 / t_dbusy) if t_dbusy else float("nan"),
        mean_drb_timeweighted=(drb_time_w / t_dbusy) if t_dbusy else float("nan"),
        n_dempty_episodes=len(episodes),
        dempty_ep_mean_ms=(sum(e["dur"] for e in episodes) / len(episodes) * 1e3) if episodes else float("nan"),
        dempty_ep_median_ms=pctl([e["dur"] for e in episodes], 0.50) * 1e3 if episodes else float("nan"),
        dempty_ep_p95_ms=pctl([e["dur"] for e in episodes], 0.95) * 1e3 if episodes else float("nan"),
        n_pa_episodes=len(pa_episodes),
        pa_ep_mean_ms=(sum(pa_episodes) / len(pa_episodes) * 1e3) if pa_episodes else float("nan"),
        n_starve_literal=n_starve_literal,
    )
    for c in CLASSES:
        out["t_" + c] = t_by_class[c]
        out["ftime_" + c] = (t_by_class[c] / t_dempty) if t_dempty else float("nan")
        out["fcount_" + c] = (n_by_class[c] / n_dempty) if n_dempty else float("nan")
    # episode-unit view: an episode is attributed to the class holding the most
    # of its time (reported ONLY as a third unit; the time shares above are the
    # estimand-matched statistic).
    ep_major = defaultdict(int)
    for e in episodes:
        ep_major[max(e["cls"].items(), key=lambda kv: kv[1])[0]] += 1
    for c in CLASSES:
        out["fep_" + c] = (ep_major[c] / len(episodes)) if episodes else float("nan")

    # ---- client-side cross-checks (independent instrument) -----------------
    if client:
        out["client_duration_s"] = client["duration"]
        out["window_minus_client_s"] = out["t_total_s"] - client["duration"]
        out["n_requests"] = client["n"]
        out["sum_ttft_s"] = client["sum_ttft_s"]
        out["ttft_mean_ms"] = client["ttft_mean_ms"]
        # PREFILL-OCCUPANCY COVERAGE.  sum(TTFT) is an UPPER bound on total
        # time-with-a-prefill-batch-in-flight (TTFT = queue wait + prefill
        # span, and the queue-wait part is not prefill-active).  The telemetry
        # grid samples 1 in PDMUX_DUAL_WORKER_TRACE_EVERY=32 scheduler loop
        # iterations, so a prefill batch living fewer than ~32 iterations can
        # be missed entirely.  These two numbers say how much of prefill
        # occupancy the grid actually sees -- and they are the reason the
        # E1_DECODE_REALIZED gradient cannot be read as physics.
        out["pa_coverage_vs_ttft"] = (out["t_prefill_active_s"] / client["sum_ttft_s"]
                                      if client["sum_ttft_s"] else float("nan"))
        out["pa_episode_detect_frac"] = (out["n_pa_episodes"] / client["n"]
                                         if client["n"] else float("nan"))
        # ceiling on the realized fraction that ANY estimator could report,
        # given that the split is engaged only while prefill is in flight.
        out["realized_ceiling"] = (client["sum_ttft_s"] / out["t_dbusy_s"]
                                   if out["t_dbusy_s"] else float("nan"))
    return out


def load_client(path):
    """Per-request bench_serving jsonl -> the independent instrument."""
    try:
        o = json.loads(open(path).read().strip().split("\n")[0])
    except Exception:
        return None
    tt = o.get("ttfts") or []
    err = o.get("errors") or []
    keep = [i for i in range(len(tt)) if (err[i] if i < len(err) else "") == ""]
    if not keep:
        return None
    v = [tt[i] for i in keep]
    return dict(n=len(keep), sum_ttft_s=sum(v),
                ttft_mean_ms=1e3 * sum(v) / len(v),
                duration=float(o.get("duration") or float("nan")))


def mean_sd(vals):
    vals = [v for v in vals if v == v]
    if not vals:
        return float("nan"), float("nan"), 0
    m = sum(vals) / len(vals)
    if len(vals) < 2:
        return m, float("nan"), len(vals)
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))
    return m, sd, len(vals)


def fmt(m, sd, prec=3):
    if m != m:
        return "  n/a "
    if sd != sd:
        return f"{m:.{prec}f}"
    return f"{m:.{prec}f}+-{sd:.{prec}f}"


def main():
    argv = sys.argv[1:]
    job = "872077"
    if "--job" in argv:
        job = argv[argv.index("--job") + 1]
    csv_path = argv[argv.index("--csv") + 1] if "--csv" in argv else None

    files = sorted(glob.glob(os.path.join(HERE, f"e1m3_*_{job}_telemetry.jsonl")))
    if not files:
        print(f"no telemetry for job={job} under {HERE}")
        return 1

    per_cell = defaultdict(list)
    audit_tot = defaultdict(int)
    n_files = 0
    for path in files:
        m = FNAME_RE.match(os.path.basename(path))
        if not m:
            continue
        arm, d, block = m["arm"], int(m["d"]), int(m["block"])
        cpath = path.replace("_telemetry.jsonl", f"_s{block}.jsonl")
        client = load_client(cpath) if os.path.exists(cpath) else None
        res = analyze_file(path, d, client)
        if not res["ok"]:
            print(f"  SKIP (too few benchmark snapshots): {res['path']}")
            continue
        n_files += 1
        a = res["audit"]
        for k in ("n", "drqd_nonzero", "ads_ne_drb", "didle_nonzero",
                  "rbo_ne_ident", "adm_ne_ident"):
            audit_tot[k] += a[k]
        res["arm"], res["d"], res["block"] = arm, d, block
        per_cell[(arm, d)].append(res)
        print(f"  read {res['path']}: {res['n_bench']} bench snapshots", flush=True)

    print()
    print("=" * 78)
    print(f"GATE #6 FIELD AUDIT (job {job}, {n_files} files, "
          f"{audit_tot['n']} benchmark runtime_snapshots)")
    print("=" * 78)
    print(f"  decode_ready_queue_depth != 0        : {audit_tot['drqd_nonzero']}  "
          "(0 => field is a CONSTANT, H-starve NOT INSTRUMENTED)")
    print(f"  active_decode_sequences != decode_running_batch_size : "
          f"{audit_tot['ads_ne_drb']}  (0 => IDENTITY, no evidence)")
    print(f"  decode_idle_ratio != 0.0             : {audit_tot['didle_nonzero']}  "
          "(0 => never assigned, no evidence)")
    print(f"  running_batch_occupancy != min(1,drb/{CAP}) : {audit_tot['rbo_ne_ident']}  "
          "(0 => IDENTITY)")
    print(f"  prefill_admission_blocked != (pqd>0 & pab==0) : "
          f"{audit_tot['adm_ne_ident']}  (0 => DERIVED, not an observation)")

    order = [("T8", d) for d in (16, 24, 44, 54)] + [("Ha8", d) for d in (16, 24, 44, 54)]

    def cell_stat(key, cell, prec=3):
        return fmt(*mean_sd([r[key] for r in per_cell[cell]])[:2], prec=prec)

    print()
    print("=" * 78)
    print("TABLE 1 -- IS DECODE EMPTY AT ALL?  (mean +- sd over n blocks)")
    print("=" * 78)
    print(f"{'arm cell':>10} {'n':>2} {'frac_time_dempty':>18} {'frac_count_dempty':>18} "
          f"{'#episodes':>11} {'ep_mean_ms':>11} {'ep_p95_ms':>10}")
    for cell in order:
        rs = per_cell.get(cell)
        if not rs:
            continue
        print(f"{cell[0]+' d'+str(cell[1]):>10} {len(rs):>2} "
              f"{cell_stat('frac_time_dempty', cell, 4):>18} "
              f"{cell_stat('frac_count_dempty', cell, 4):>18} "
              f"{cell_stat('n_dempty_episodes', cell, 1):>11} "
              f"{cell_stat('dempty_ep_mean_ms', cell, 1):>11} "
              f"{cell_stat('dempty_ep_p95_ms', cell, 1):>10}")

    print()
    print("=" * 78)
    print("TABLE 2 -- DECODE-EMPTY TIME DECOMPOSITION (time-weighted shares of")
    print("           decode-empty time; exclusive & exhaustive, sums to 1.000)")
    print("=" * 78)
    print(f"{'arm cell':>10} {'n':>2} {'A_no_work':>16} {'B_prefill_only':>16} "
          f"{'C_prefill_queued':>17} {'D_queue_stalled':>16}")
    for cell in order:
        rs = per_cell.get(cell)
        if not rs:
            continue
        row = f"{cell[0]+' d'+str(cell[1]):>10} {len(rs):>2} "
        for c in CLASSES:
            row += f"{cell_stat('ftime_' + c, cell, 4):>16} " if c != "C_prefill_queued" \
                   else f"{cell_stat('ftime_' + c, cell, 4):>17} "
        print(row)

    print()
    print("TABLE 2b -- same decomposition, SNAPSHOT-COUNT unit (unit-sensitivity)")
    print(f"{'arm cell':>10} {'A_no_work':>16} {'B_prefill_only':>16} "
          f"{'C_prefill_queued':>17} {'D_queue_stalled':>16}")
    for cell in order:
        if not per_cell.get(cell):
            continue
        row = f"{cell[0]+' d'+str(cell[1]):>10} "
        for c in CLASSES:
            row += f"{cell_stat('fcount_' + c, cell, 4):>16} " if c != "C_prefill_queued" \
                   else f"{cell_stat('fcount_' + c, cell, 4):>17} "
        print(row)

    print()
    print("TABLE 2c -- same decomposition, EPISODE unit (majority class per episode)")
    print(f"{'arm cell':>10} {'A_no_work':>16} {'B_prefill_only':>16} "
          f"{'C_prefill_queued':>17} {'D_queue_stalled':>16}")
    for cell in order:
        if not per_cell.get(cell):
            continue
        row = f"{cell[0]+' d'+str(cell[1]):>10} "
        for c in CLASSES:
            row += f"{cell_stat('fep_' + c, cell, 4):>16} " if c != "C_prefill_queued" \
                   else f"{cell_stat('fep_' + c, cell, 4):>17} "
        print(row)

    print()
    print("=" * 78)
    print("TABLE 3 -- WHAT ACTUALLY DILUTES E1_DECODE_REALIZED")
    print("  decode_realized      = frac of decode-BUSY time with decode_sms==D")
    print("  dbusy_prefill_active = frac of decode-BUSY time with a prefill batch")
    print("                         in flight (the split can only be engaged then)")
    print("  dbusy_dsm108         = frac of decode-BUSY time at the decode-only")
    print("                         group (0,108): prefill absent -> no split")
    print("=" * 78)
    print(f"{'arm cell':>10} {'n':>2} {'decode_realized':>18} {'dbusy_prefill_active':>22} "
          f"{'dbusy_dsm108':>16} {'t_prefill_active_s':>19} {'pa_ep_mean_ms':>14}")
    for cell in order:
        rs = per_cell.get(cell)
        if not rs:
            continue
        print(f"{cell[0]+' d'+str(cell[1]):>10} {len(rs):>2} "
              f"{cell_stat('decode_realized_frac', cell, 4):>18} "
              f"{cell_stat('dbusy_prefill_active_frac', cell, 4):>22} "
              f"{cell_stat('dbusy_dsm108_frac', cell, 4):>16} "
              f"{cell_stat('t_prefill_active_s', cell, 2):>19} "
              f"{cell_stat('pa_ep_mean_ms', cell, 2):>14}")

    print()
    print("=" * 78)
    print("TABLE 3b -- CAN THE GRID EVEN SEE PREFILL?  (the estimator's own bias)")
    print("  pa_episode_detect_frac = detected prefill-active episodes / requests")
    print("  pa_coverage_vs_ttft    = telemetry t_prefill_active / sum(client TTFT)")
    print("      sum(TTFT) UPPER-bounds true prefill-in-flight time (TTFT = queue")
    print("      wait + prefill span), so this ratio is a LOWER bound on coverage.")
    print("  realized_ceiling       = sum(TTFT)/t_decode_busy = the largest value")
    print("      E1_DECODE_REALIZED could take even with a perfect instrument.")
    print("=" * 78)
    print(f"{'arm cell':>10} {'n':>2} {'pa_detect_frac':>17} {'pa_cov_vs_ttft':>17} "
          f"{'realized_ceiling':>18} {'decode_realized':>17}")
    for cell in order:
        if not per_cell.get(cell):
            continue
        print(f"{cell[0]+' d'+str(cell[1]):>10} {len(per_cell[cell]):>2} "
              f"{cell_stat('pa_episode_detect_frac', cell, 4):>17} "
              f"{cell_stat('pa_coverage_vs_ttft', cell, 4):>17} "
              f"{cell_stat('realized_ceiling', cell, 4):>18} "
              f"{cell_stat('decode_realized_frac', cell, 4):>17}")

    print()
    print("=" * 78)
    print("TABLE 5 -- WINDOW VALIDATION (load window vs the client's own duration)")
    print("=" * 78)
    print(f"{'arm cell':>10} {'n':>2} {'file_span_s':>16} {'head_gap_s':>16} "
          f"{'tail_gap_s':>16} {'window_s':>16} {'window-client_s':>17} "
          f"{'dempty_FULLFILE':>17}")
    for cell in order:
        if not per_cell.get(cell):
            continue
        print(f"{cell[0]+' d'+str(cell[1]):>10} {len(per_cell[cell]):>2} "
              f"{cell_stat('file_span_s', cell, 2):>16} "
              f"{cell_stat('head_gap_s', cell, 2):>16} "
              f"{cell_stat('tail_gap_s', cell, 2):>16} "
              f"{cell_stat('t_total_s', cell, 2):>16} "
              f"{cell_stat('window_minus_client_s', cell, 2):>17} "
              f"{cell_stat('frac_time_dempty_FULLFILE', cell, 4):>17}")

    print()
    print("=" * 78)
    print("TABLE 4 -- CONTEXT / CONFOUNDS")
    print("=" * 78)
    print(f"{'arm cell':>10} {'n':>2} {'t_total_s':>14} {'mean_drb(tw)':>14} "
          f"{'dt_med_ms':>11} {'dt_p95_ms':>11} {'dt_max_ms':>11}")
    for cell in order:
        rs = per_cell.get(cell)
        if not rs:
            continue
        print(f"{cell[0]+' d'+str(cell[1]):>10} {len(rs):>2} "
              f"{cell_stat('t_total_s', cell, 2):>14} "
              f"{cell_stat('mean_drb_timeweighted', cell, 2):>14} "
              f"{cell_stat('dt_median_ms', cell, 2):>11} "
              f"{cell_stat('dt_p95_ms', cell, 2):>11} "
              f"{cell_stat('dt_max_ms', cell, 1):>11}")

    if csv_path:
        import csv as _csv
        keys = [k for k in sorted(per_cell[order[0]][0].keys())
                if k not in ("audit",)]
        with open(csv_path, "w", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for cell in order:
                for r in per_cell.get(cell, []):
                    w.writerow(r)
        print(f"\nper-block rows written to {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
