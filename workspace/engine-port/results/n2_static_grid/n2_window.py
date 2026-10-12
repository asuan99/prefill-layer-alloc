#!/usr/bin/env python3
"""N2(3-a) cell cutting + replicate validity (PREREG_N2_3A rev16 §6, §7, §9-1, §11-4; §15 `n2_window.py`). CPU only.

HARNESS-SEAL file. SHARED WINDOW IMPLEMENTATION (§15, rev15 R14-8): every window, window status, S-per interval,
A_on interpolation and PE fraction comes from the RULE-SEALED reducer n2_probe_reduce (loaded by file path):
`cell_windows`, `s_per_intervals`, `a_on_interp`, `_pe`, `_pe_time`. This module adds ONLY what §15 assigns to it:
cutting a boot's telemetry by the runner's cell marks (§6-4), the cell-interval Δdropped increment with the F-6
after-record rule, the boot-level R7 marker, the per-cell cudagraph evidence (§3-2 R-5 / F-5), RETRACT, bench
validity, the §9-1 `measurement_ok` order, the OBS client window (§11-4), CTX_MISMATCH (§10-2, Job-EA) and the
reason combination. No second window implementation.

measurement_ok order (§9-1): NOT_ATTEMPTED_BUDGET (ledger) > BENCH_INVALID > [E-alpha: COLOAD_PAYLOAD_INVALID] >
RETRACT > RECORDER_UNPROVEN (boot R7) > RECORDER_UNPROVEN (cell Δdropped != 0 or no record after the cell) >
CG_UNPROVEN. Then PROFILE: the window status of cell_windows(W) (§9-1 alone_status); OBS: the client window;
EALPHA (Job-EA, `ealpha_replicate`): the A1 / A2 / C states of cell_windows plus, in the §9-1 C order, the co-span
composition (§7-2 (c)) and the JOINED signature (§7-2 (f), n2_signature).
EA HARNESS READINGS (reported, not yet in §18-2-d): a cell_abort with reason coload_payload_invalid is
COLOAD_PAYLOAD_INVALID (not BENCH_INVALID); the client co-load record (K responses, completion_tokens 1, length) is
re-checked after bench validity; a completed E-alpha cell without coload_sent is BENCH_INVALID; a co-span composition
failure is a C-window WINDOW_SHORT; S2 reads the PRE-FILTER first 128 iterations after the last co span end.

HARNESS READINGS (unregistered details, reported): bench validity under U-1 (b) cancellation = every one of the b
requests answered without an exception, prompt_tokens == c_in, >= 1 token, and either finished at max_new_tokens
(finish_reason length) or was still running when the registered cancel rule fired; RETRACT = a server-log line in the
cell's wall seconds [floor(start), ceil(end)] carrying `#retracted_reqs: N` with N > 0 (boundary seconds attributed to
BOTH neighbours -- fail-closed); cells never started because their boot aborted = BENCH_INVALID with sub-reason
`not_attempted_boot_aborted`; log timestamps are converted with the boot record's tz_gmtoff_s.
"""
from __future__ import annotations

import calendar
import importlib.util
import json
import math
import os
import re
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REDUCER_FILE = os.path.join(HERE, "n2_probe_reduce.py")


def _load_reducer():
    spec = importlib.util.spec_from_file_location("_n2_window_reducer", REDUCER_FILE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_n2_window_reducer"] = mod
    spec.loader.exec_module(mod)
    return mod


RED = _load_reducer()
sys.path.insert(0, HERE)
import n2_signature as SG    # noqa: E402  (§7-2 (f) S1-S3, Job-EA)
PAIR = RED.PAIR
WIN = RED.WINDOW_ITERS
CTX_MISMATCH_FRAC = 0.1                       # §10-2 (Job-EA)
STALL_MULTS = (5.0, 10.0)                     # §7-1 (R) descriptive stall counts
R7_MARK = "PDMUX_PHASE_EVENTS disabled after failure"
LOG_TS = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] (.*)$")
RETRACT_RE = re.compile(r"#retracted_reqs: *(\d+)")

MEASUREMENT_FAILURE = ("NOT_ATTEMPTED_BUDGET", "BENCH_INVALID", "COLOAD_PAYLOAD_INVALID", "RETRACT",
                       "RECORDER_UNPROVEN", "CG_UNPROVEN", "CO_WINDOW_SHORT")          # §9-1 부류 (F-3)
PREDICATE_FAILURE = ("CTX_UNREALIZED", "WINDOW_SHORT", "D_UNREALIZED", "BS_UNREALIZED", "COLOC_UNREALIZED",
                     "CTX_MISMATCH")


def quantile(xs, q):
    """numpy default 'linear' percentile (§6-3)."""
    v = sorted(xs)
    if not v:
        return None
    pos = (len(v) - 1) * q
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


# ------------------------------------------------------------------------------------------- boot loading
def _read(path):
    try:
        with open(path) as fh:
            return fh.read()
    except OSError:
        return None


def load_boot(out, name):
    """Everything n2_window / n2_analyze need from one boot directory entry. Never raises on a missing file: the
    absence is recorded and the replicate rules fail closed."""
    b = dict(name=name, problems=[])
    tel = os.path.join(out, f"tel_{name}.jsonl")
    try:
        b["records"], b["truncated"] = RED._read_telemetry(tel)
    except (OSError, ValueError) as e:                        # corrupt telemetry: nothing in this boot is provable
        b["records"], b["truncated"] = [], False
        b["problems"].append(f"telemetry unreadable: {e!r}")
        b["telemetry_corrupt"] = True
    lines = [ln for ln in (_read(os.path.join(out, f"marks_{name}.txt")) or "").splitlines() if ln.strip()]
    warm = {}
    b["cells"], b["calls"], b["plan"], b["boot_aborts"], b["mark_issues"] = RED.parse_marks(lines, warm)
    b["warmups"] = warm
    b["log"] = _read(os.path.join(out, f"srv_{name}.log")) or ""
    b["boot"] = json.loads(_read(os.path.join(out, f"boot_{name}.json")) or "{}")
    b["green"] = json.loads(_read(os.path.join(out, f"green_{name}.json")) or "{}")
    b["greedy"] = json.loads(_read(os.path.join(out, f"greedy_{name}.json")) or "null")
    b["client"] = {}
    for ln in (_read(os.path.join(out, f"client_{name}.jsonl")) or "").splitlines():
        if ln.strip():
            rec = json.loads(ln)
            b["client"][rec["cell_id"]] = rec
    b["wall"] = {}
    for ln in (_read(os.path.join(out, f"wall_{name}.txt")) or "").splitlines():
        f = ln.split()
        if len(f) == 4:
            b["wall"][(f[0], f[1])] = (float(f[2]), float(f[3]))
    b["cpu"] = _read(os.path.join(out, f"cpu_{name}.txt")) or ""
    gm = b["boot"].get("tz_gmtoff_s")
    b["tz_gmtoff_s"] = time.localtime().tm_gmtoff if gm is None else gm
    b["log_lines"] = parse_log(b["log"], b["tz_gmtoff_s"])
    return b


def parse_log(text, gmtoff):
    out = []
    for ln in text.splitlines():
        m = LOG_TS.match(ln)
        if not m:
            continue
        st = time.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
        out.append((calendar.timegm(st) - gmtoff, m.group(2)))
    return out


# ------------------------------------------------------------------------------------------- §6-4 cutting
def _ts(e):
    return RED._ts(e)


def cell_slice(records, t0, t1):
    return [e for e in records if t0 <= _ts(e) <= t1]


def delta_dropped(records, t0, t1):
    """§6-4: (first record after the cell)'s dropped_events - (last record before the cell)'s; None when no record
    follows the cell (F-6 -> RECORDER_UNPROVEN)."""
    before = [e for e in records if _ts(e) < t0]
    after = [e for e in records if _ts(e) > t1]
    if not after:
        return None
    first_after = min(after, key=_ts)
    last_before = max(before, key=_ts) if before else None
    base = int(last_before.get("dropped_events", 0) or 0) if last_before else 0
    return int(first_after.get("dropped_events", 0) or 0) - base


def r7(boot):
    return R7_MARK in boot["log"]


def cell_bounds(boot, cell_id):
    cm = boot["cells"].get(cell_id) or {}
    if "cell_start" not in cm or "cell_end" not in cm:
        return None
    return cm["cell_start"][1], cm["cell_end"]


def wall_offset(boot, cell_id):
    """(time.time - CLOCK_MONOTONIC) of the runner's wall pair at the cell start (G-11)."""
    pr = boot["wall"].get(("cell_start", cell_id))
    return None if pr is None else pr[0] - pr[1]


def to_wall(boot, cell_id, t):
    """§3-2: map a CLOCK_MONOTONIC time of this cell to wall seconds with the runner's (time.time, monotonic) pairs
    taken at the cell boundaries (G-11): linear between the cell_start and cell_end pairs, the start pair's offset
    when only that one exists. None without a start pair."""
    return pair_to_wall(boot["wall"].get(("cell_start", cell_id)), boot["wall"].get(("cell_end", cell_id)), t)


def pair_to_wall(a, z, t):
    """The §3-2 / G-11 mapping of a CLOCK_MONOTONIC time `t` to wall seconds from two (time.time, monotonic) pairs
    `a` (start) and `z` (end): linear between them, `a`'s offset alone when `z` is missing; None without `a`."""
    if a is None:
        return None
    if z is None or z[1] == a[1]:
        return t + (a[0] - a[1])
    return a[0] + (t - a[1]) * (z[0] - a[0]) / (z[1] - a[1])


def warmup_cg_canary(log_lines, wall_start, wall_end, t0, t1):
    """§18-2-d #16 CG canary (rev17 H17-3, Job abort ABORT_CG_CANARY -- cost protection, NOT a judgment input; the
    per-cell `cg_evidence` / CG_UNPROVEN is unchanged). `log_lines` = parse_log(server log, boot tz_gmtoff_s);
    `wall_start` / `wall_end` = the runner's (time.time, monotonic) pairs taken around the warm-up; t0 / t1 = the
    `warmup_start` / `warmup_end` mark times (monotonic). The warm-up seconds use the same boundary convention as
    cg_evidence: [ceil(wall(t0)), floor(wall(t1)) - 1]. Returns (ok, n_decode_lines, n_not_true, lo, hi) with
    ok := at least ONE `Decode batch` line in those seconds AND every one of them carries `cuda graph: True`."""
    w0, w1 = pair_to_wall(wall_start, wall_end, t0), pair_to_wall(wall_start, wall_end, t1)
    if w0 is None or w1 is None:
        return False, 0, 0, None, None
    lo, hi = math.ceil(w0), math.floor(w1) - 1
    canary = [txt for ts, txt in log_lines if txt.startswith("Decode batch") and lo <= ts <= hi]
    n_bad = sum(1 for txt in canary if "cuda graph: True" not in txt)
    return (bool(canary) and n_bad == 0), len(canary), n_bad, lo, hi


def cg_evidence(boot, cell_id):
    """§3-2 R-5 / F-5: `Decode batch` lines only, in the wall seconds [ceil(wall(start)), floor(wall(end)) - 1].
    Returns (proven, n_lines, n_false)."""
    bd = cell_bounds(boot, cell_id)
    if bd is None or to_wall(boot, cell_id, bd[0]) is None:
        return False, 0, 0
    lo, hi = math.ceil(to_wall(boot, cell_id, bd[0])), math.floor(to_wall(boot, cell_id, bd[1])) - 1
    dec = [txt for ts, txt in boot["log_lines"] if lo <= ts <= hi and txt.startswith("Decode batch")]
    n_true = sum("cuda graph: True" in t for t in dec)
    return (len(dec) >= 1 and n_true == len(dec)), len(dec), len(dec) - n_true


def retract_count(boot, cell_id):
    bd = cell_bounds(boot, cell_id)
    if bd is None or to_wall(boot, cell_id, bd[0]) is None:
        return 0
    lo, hi = math.floor(to_wall(boot, cell_id, bd[0])), math.ceil(to_wall(boot, cell_id, bd[1]))
    n = 0
    for ts, txt in boot["log_lines"]:
        if lo <= ts <= hi:
            for m in RETRACT_RE.findall(txt):
                n += int(m)
    return n


def bench_valid(client, cell):
    """(ok, why). HARNESS READING under U-1 (b) cancellation (module docstring)."""
    if client is None:
        return False, "no client record"
    if client.get("aborted"):
        return False, f"cell aborted: {client['aborted']}"
    reqs = client.get("requests") or []
    if len(reqs) != cell["b"]:
        return False, f"{len(reqs)} requests != b {cell['b']}"
    for q in reqs:
        if q.get("prompt_tokens") is None or int(q["prompt_tokens"]) != cell["c_in"]:      # missing = fail-closed
            return False, f"{q['rid']}: prompt_tokens {q['prompt_tokens']} != c_in {cell['c_in']}"
        if q.get("n", 0) < 1:
            return False, f"{q['rid']}: no token"
        finished = q.get("done") and q.get("n") == cell["O"] and q.get("finish") == "length"
        if not (finished or (client.get("cancel_fired") and not q.get("done"))):
            return False, f"{q['rid']}: ended with {q.get('n')} tokens (neither O={cell['O']} nor cancelled)"
    return True, ""


def coload_payload_ok(client, cell):
    """§9-1 coload_payload_ok (F3): the client record holds exactly K co-load responses, each finished with
    completion_tokens == 1 and finish_reason length."""
    co = (client or {}).get("co_requests")
    if co is None:
        return False, "no co-load record"
    if len(co) != cell["K"]:
        return False, f"{len(co)} co-load responses != K {cell['K']}"
    for q in co:
        if not (q.get("done") and q.get("completion_tokens") == 1 and q.get("n") == 1 and q.get("finish") == "length"):
            return False, f"{q.get('rid')}: completion_tokens {q.get('completion_tokens')} finish {q.get('finish')}"
    return True, ""


def measurement_ok(boot, cell, not_attempted=False):
    """§9-1 measurement stage (all cell kinds), first failing reason or None; plus a detail string."""
    cid = cell["cell_id"]
    if not_attempted:
        return "NOT_ATTEMPTED_BUDGET", "stop rule (§14-3)"
    cm = boot["cells"].get(cid) or {}
    if "cell_start" not in cm:
        return "BENCH_INVALID", "not_attempted_boot_aborted"
    ea = cell.get("kind") == "EALPHA"
    if ea and (cm.get("cell_abort") or ("",))[0] == "coload_payload_invalid":
        return "COLOAD_PAYLOAD_INVALID", "cell_abort coload_payload_invalid (F3: the client aborted the cell)"
    if "cell_abort" in cm:
        return "BENCH_INVALID", f"cell_abort {cm['cell_abort'][0]}"
    ok, why = bench_valid(boot["client"].get(cid), cell)
    if not ok:
        return "BENCH_INVALID", why
    if ea:
        ok, why = coload_payload_ok(boot["client"].get(cid), cell)
        if not ok:
            return "COLOAD_PAYLOAD_INVALID", why
    if retract_count(boot, cid) > 0:
        return "RETRACT", "retracted requests in the cell's log seconds"
    if r7(boot):
        return "RECORDER_UNPROVEN", "boot R7 marker (D-6: every cell of the boot)"
    if boot.get("telemetry_corrupt"):
        return "RECORDER_UNPROVEN", "telemetry corrupt"
    bd = cell_bounds(boot, cid)
    dd = delta_dropped(boot["records"], *bd)
    if dd is None:
        return "RECORDER_UNPROVEN", "no record after the cell (F-6)"
    if dd != 0:
        return "RECORDER_UNPROVEN", f"cell Δdropped {dd}"
    proven, n, nf = cg_evidence(boot, cid)
    if not proven:
        return "CG_UNPROVEN", f"Decode batch lines {n}, cuda graph False {nf}"
    return None, ""


# ------------------------------------------------------------------------------------------- windows (shared)
def profile_windows(boot, cell, pi_alone):
    bd = cell_bounds(boot, cell["cell_id"])
    recs = cell_slice(boot["records"], *bd)
    return recs, RED.cell_windows(recs, {}, (cell.get("D_mark"), cell["b"], cell["c"]), pi_alone, coload=False)


def window_stats(win_iters, recs, b):
    """Descriptive per-window values (§6-1, §6-3, §7-1 (R), §12-1): S-per quantiles, S-dev, stalls, ctx spread."""
    s = RED.s_per_intervals(win_iters)
    med = statistics.median(s)
    lo, hi = win_iters[0]["t_launch"], win_iters[-1]["t_sync"]
    snaps = [e for e in recs if e.get("event") == "runtime_snapshot" and lo <= _ts(e) <= hi
             and e.get("active_decode_sequences") == b]
    spread95 = [e["context_p95"] - e["context_p50"] for e in snaps if "context_p95" in e]
    spreadmx = [e["context_max"] - e["context_p50"] for e in snaps if "context_max" in e]
    return dict(n_intervals=len(s), s_per_p50=quantile(s, 0.5), s_per_p95=quantile(s, 0.95),
                s_per_p99=quantile(s, 0.99), s_per_median=med,
                s_dev_median_ms=statistics.median((e["t_sync"] - e["t_launch"]) * 1000.0 for e in win_iters),
                stalls_gt5=sum(1 for x in s if x > STALL_MULTS[0] * med),
                stalls_gt10=sum(1 for x in s if x > STALL_MULTS[1] * med),
                ctx_spread_p95_max=max(spread95) if spread95 else None,
                ctx_spread_max_max=max(spreadmx) if spreadmx else None, ctx_snapshots=len(snaps),
                t_lo=lo, t_hi=hi, iter_seq=(win_iters[0]["iter_seq"], win_iters[-1]["iter_seq"]),
                intervals=s)


def profile_replicate(boot, unit, cell, not_attempted=False):
    """One PROFILE replicate (§9-1 replicate_status for cell.kind == PROFILE) + descriptives."""
    pi_alone = PAIR[unit["alone_D"]]
    out = dict(boot=boot["name"], unit=unit["name"], cell_id=cell["cell_id"], kind=cell["kind"], D=unit["D"],
               alone_D=unit["alone_D"], b=cell["b"], c=cell["c"], round=unit["round"], boot_kind=unit["kind"])
    m, why = measurement_ok(boot, cell, not_attempted)
    if m:
        out.update(status=m, reason=why)
        return out
    cm = boot["cells"][cell["cell_id"]]
    cell = dict(cell, D_mark=cm["cell_start"][0][0])
    recs, win = profile_windows(boot, cell, pi_alone)
    out.update(status=win["status"]["W"], reason=win["detail"]["W"], ctx_mid=win["ctx_mid"].get("A1"),
               split_transitions=sum(1 for e in recs if e.get("event") == "split_transition"))
    w = win["W"]
    if w:
        # N2C-75 report (rev16 seal audit): the two conventions not bound by a rule test -- a window whose PE-time
        # weight sum is 0 (no iter_seq-consecutive pair) and prefill span events exactly on the §7-1 (c) endpoints
        seqs = {e["iter_seq"] for e in w}
        out["n2c75_zero_weight"] = not any(q + 1 in seqs for q in seqs)
        ends = (w[0]["t_launch"], w[-1].get("t_sync"))
        out["n2c75_endpoint_prefill_events"] = sum(
            1 for e in recs if e.get("event") in ("prefill_span_start", "prefill_span_end")
            and e["timestamp_monotonic_s"] in ends)
        out["pe_iter"] = RED._pe(w, pi_alone)
        out["pe_time"] = RED._pe_time(w, lambda e: RED._pair(e) == pi_alone)
        out["pe_iter_idx5"] = RED._pe(w, PAIR[108])
    if len(w) == WIN and out["status"] in ("VALID", "D_UNREALIZED"):
        try:
            out["stats"] = window_stats(w, recs, cell["b"])
        except RED.MeasurementRejected as e:                    # no consecutive pair: descriptive only
            out["stats_error"] = str(e)
    out["window_len"] = len(w)
    out["_window"] = w
    out["_records"] = recs
    return out


def obs_client_window(client):
    """§11-4: per request, the token intervals from the moment ALL b requests have their first token until just
    before the first request's termination (end of stream or cancel); >= 128 intervals per request, else
    WINDOW_SHORT. C-itl = median over requests of the per-request p95 interval (§6-1). Returns (c_itl_ms, min_n)."""
    reqs = (client or {}).get("requests") or []
    if not reqs or any(not q.get("times") for q in reqs):
        return None, 0
    t_all_first = max(q["times"][0] for q in reqs)
    ends = [q["times"][-1] for q in reqs]
    if client.get("t_cancel") is not None:
        ends.append(client["t_cancel"])
    t_end = min(ends)
    per, nmin = [], None
    for q in reqs:
        ts = [t for t in q["times"] if t_all_first <= t < t_end]
        iv = [(b_ - a_) * 1000.0 for a_, b_ in zip(ts, ts[1:])]
        nmin = len(iv) if nmin is None else min(nmin, len(iv))
        if len(iv) < WIN:
            return None, nmin
        per.append(quantile(iv, 0.95))
    return statistics.median(per), nmin


def c_itl_in_window(client, t_lo, t_hi):
    """§6-1 / §12-3 (descriptive, reconstruction-based): per request, intervals of the tokens received inside
    [first t_launch, last t_sync] of the engine window; p95 per request, median over requests."""
    per = []
    for q in (client or {}).get("requests") or []:
        ts = [t for t in q.get("times") or [] if t_lo <= t <= t_hi]
        iv = [(b_ - a_) * 1000.0 for a_, b_ in zip(ts, ts[1:])]
        if len(iv) >= 2:
            per.append(quantile(iv, 0.95))
    return statistics.median(per) if per else None


def obs_replicate(boot, unit, cell, not_attempted=False):
    out = dict(boot=boot["name"], unit=unit["name"], cell_id=cell["cell_id"], kind="OBS", D=unit["D"],
               alone_D=unit["alone_D"], b=cell["b"], c=cell["c"], round=unit["round"], boot_kind=unit["kind"])
    m, why = measurement_ok(boot, cell, not_attempted)
    if m:
        out.update(status=m, reason=why)
        return out
    citl, nmin = obs_client_window(boot["client"].get(cell["cell_id"]))
    out.update(status="VALID" if citl is not None else "WINDOW_SHORT", c_itl_ms=citl, client_min_intervals=nmin)
    return out


def ctx_mismatch(ctx_c, ctx_a1, ctx_a2, c):
    """§10-2 (Job-EA): ctx_mid(C) outside [ctx_mid(A1), ctx_mid(A2)] by more than 0.1 x c tokens."""
    lo, hi = min(ctx_a1, ctx_a2), max(ctx_a1, ctx_a2)
    return ctx_c < lo - CTX_MISMATCH_FRAC * c or ctx_c > hi + CTX_MISMATCH_FRAC * c


def window_wall_range(boot, cell_id, t_lo, t_hi):
    lo = to_wall(boot, cell_id, t_lo)
    return None if lo is None else (lo, to_wall(boot, cell_id, t_hi))


def cpu_delta(boot, cell_id):
    """cgroup cpu.stat increments across the cell (§3-3, descriptive): nr_throttled, throttled_usec."""
    blocks = re.split(r"^# ", boot["cpu"], flags=re.M)
    vals = {}
    for blk in blocks:
        head, _, body = blk.partition("\n")
        for tag in ("before", "after"):
            if head.startswith(f"{tag} {cell_id} "):
                vals[tag] = dict(ln.split() for ln in body.splitlines() if len(ln.split()) == 2)
    if "before" not in vals or "after" not in vals:
        return None
    out = {}
    for k in ("nr_throttled", "throttled_usec", "usage_usec"):
        try:
            out[k] = int(vals["after"][k]) - int(vals["before"][k])
        except (KeyError, ValueError):
            out[k] = None
    return out


# ------------------------------------------------------------------------------------------- Job-EA (§7-2, §9-1, §10)
CO_INPUT = 8192
JOINABLE_C = ("BS_UNREALIZED", "CO_WINDOW_SHORT", "COLOC_UNREALIZED", "VALID")   # §9-1: JOINED sits after RECORDER /
#                                                                                   non-co span, before B_FINISHED


def co_span_composition(records, t_co, k):
    """§7-2 (c) co-span composition (engine edit 4): span_seq unique among the starts and among the ends of the cell,
    every co span (start at/after coload_sent, extend_tokens == n_new x 8192) paired with its end, each co span
    n_new <= 2, sum n_new == K. Returns the problems (empty = as registered). HARNESS READING (EA, reported): a
    violation is a C-window WINDOW_SHORT (predicate failure) placed after RECORDER / non-co span in the §9-1 order --
    the registration lists the condition but no label for its failure."""
    starts = [e for e in records if e.get("event") == "prefill_span_start"]
    ends = [e for e in records if e.get("event") == "prefill_span_end"]
    bad = []
    for what, evs in (("start", starts), ("end", ends)):
        seqs = [e["span_seq"] for e in evs]
        if len(seqs) != len(set(seqs)):
            bad.append(f"duplicate span_seq among prefill_span_{what}")
    end_seqs = {e["span_seq"] for e in ends}
    co = [s for s in starts if s["timestamp_monotonic_s"] >= t_co and s.get("n_new", 0) >= 1
          and s.get("extend_tokens") == s["n_new"] * CO_INPUT]
    unpaired = [s["span_seq"] for s in co if s["span_seq"] not in end_seqs]
    if unpaired:
        bad.append(f"co span(s) without prefill_span_end: {unpaired[:5]}")
    big = [s["n_new"] for s in co if s["n_new"] > 2]
    if big:
        bad.append(f"co span n_new > 2: {big[:5]}")
    tot = sum(s["n_new"] for s in co)
    if tot != k:
        bad.append(f"sum n_new {tot} != K {k}")
    return bad


def _intervals_desc(w):
    """DESCRIPTIVE ONLY (§10-2 / §12-4 whole-W_co and front/back-half rho): the S-per consecutive-iter_seq rule of
    RED.s_per_intervals without its 128-iteration minimum (a half of the co-located set can be shorter). Never a label
    input -- the judgment values (C, A1, A2) always go through RED.s_per_intervals."""
    if len(w) >= WIN:
        return RED.s_per_intervals(w)
    by = {e["iter_seq"]: e for e in w}
    return [(by[q + 1]["t_launch"] - e["t_launch"]) * 1000.0 for q, e in sorted(by.items()) if q + 1 in by]


def _window_values(w, pi, ctx_mid):
    if len(w) != WIN:
        return dict(n=len(w))
    base = dict(n=len(w), ctx_mid=ctx_mid, pe_iter=RED._pe(w, pi), pe_iter_idx5=RED._pe(w, PAIR[108]),
                t_lo=w[0]["t_launch"], t_hi=w[-1].get("t_sync", w[-1]["t_launch"]),
                iter_seq=(w[0]["iter_seq"], w[-1]["iter_seq"]))
    try:
        iv = RED.s_per_intervals(w)
    except RED.MeasurementRejected as e:
        return dict(base, error=str(e))
    med = statistics.median(iv)
    return dict(base, s_per_median=med, s_per_p50=quantile(iv, 0.5), s_per_p95=quantile(iv, 0.95),
                n_intervals=len(iv), stalls_gt5=sum(1 for x in iv if x > STALL_MULTS[0] * med),
                stalls_gt10=sum(1 for x in iv if x > STALL_MULTS[1] * med))


def ealpha_replicate(boot, unit, cell, not_attempted=False):
    """One E-alpha replicate (§9-1 replicate_status for cell.kind == EALPHA): the measurement stage, then the window
    states A1 / A2 / C from the RULE-SEALED cell_windows (π per window, §7-1: ON A1/A2 = idx(d), OFF A1/A2 = idx5;
    C = idx(d)), with the two harness-side C steps in the §9-1 order: co-span composition (§7-2 (c)) and the JOINED
    signature (§7-2 (f): S1 over co span ends, S2 over the PRE-FILTER W_co and the first 128 PRE-FILTER iterations
    after the last co span end -- the filtered A2 window would make S2 vacuous). Values for rho (§10-2) are the
    S-per medians and ctx_mid of A1 / A2 / C; descriptives (§12-4) are kept apart."""
    d, b, c = unit["D"], cell["b"], cell["c"]
    off = unit["kind"] == "OFF"
    pi_alone, pi_coloc = (PAIR[108] if off else PAIR[d]), PAIR[d]
    out = dict(boot=boot["name"] if boot else unit["name"], unit=unit["name"], cell_id=cell["cell_id"], kind="EALPHA",
               D=d, alone_D=108 if off else d, b=b, c=c, round=unit["round"], boot_kind=unit["kind"], K=cell["K"],
               cell_key=f"{d},{b},{c}")
    m, why = measurement_ok(boot, cell, not_attempted) if boot is not None else ("BENCH_INVALID", "no boot")
    if m:
        out.update(status=m, reason=why, measurement=m, windows={k: m for k in ("A1", "A2", "C")}, details={})
        return out
    cm = boot["cells"][cell["cell_id"]]
    t_co = cm.get("coload_sent")
    if t_co is None:                    # HARNESS READING (EA): a completed cell without coload_sent = client protocol
        out.update(status="BENCH_INVALID", reason="coload_sent mark missing", measurement="BENCH_INVALID",
                   windows={k: "BENCH_INVALID" for k in ("A1", "A2", "C")}, details={})
        return out
    recs = cell_slice(boot["records"], *cell_bounds(boot, cell["cell_id"]))
    win = RED.cell_windows(recs, {"coload_sent": t_co}, (cm["cell_start"][0][0], b, c), pi_alone, pi_coloc,
                           coload=True)
    st = {"A1": win["status"]["W"], "A2": win["status"]["A2"], "C": win["status"]["C"]}
    det = {"A1": win["detail"]["W"], "A2": win["detail"]["A2"], "C": win["detail"]["C"]}
    comp = co_span_composition(recs, t_co, cell["K"])
    if comp and st["C"] not in ("RECORDER_UNPROVEN", "WINDOW_SHORT"):
        st["C"], det["C"] = "WINDOW_SHORT", "co span composition: " + "; ".join(comp)
    iters = sorted((e for e in recs if e.get("event") == "decode_iteration"), key=lambda e: e["iter_seq"])
    co_sp = SG.co_spans(recs, t_co)
    sig = None
    if co_sp:
        t_last = max(en["timestamp_monotonic_s"] for _s, en in co_sp)
        a2_pre = [e for e in iters if e["t_launch"] > t_last][:WIN]
        sig = SG.signature(recs, t_co, b, win["W_co"], a2_pre)
        if sig["signature"] == "JOINED" and st["C"] in JOINABLE_C:
            st["C"], det["C"] = "BS_UNREALIZED", f"JOINED (S1 {sig['s1']}, S2 bs>b {sig['s2_bs_gt_b']})"
    snaps = [e for e in recs if e.get("event") == "runtime_snapshot"]
    ctx = RED.ctx_of(iters, snaps, b)
    vals = {"A1": _window_values(win["W"], pi_alone, win["ctx_mid"].get("A1")),
            "A2": _window_values(win["A2"], pi_alone, win["ctx_mid"].get("A2")),
            "C": _window_values(win["C"], pi_coloc, win["ctx_mid"].get("C"))}
    wco = win["W_co"]
    bs_eq = [e for e in wco if e["bs"] == b]
    call = win["C_all"]
    desc = dict(w_co=len(wco), n_colocated_all=len(call), l_lag_iters=win.get("l_lag_iters"),
                bs_ne_b=sum(1 for e in wco if e["bs"] != b),
                bs_ne_b_hist={str(k): sum(1 for e in wco if e["bs"] == k) for k in sorted({e["bs"] for e in wco
                                                                                          if e["bs"] != b})},
                pe_coloc_iter=(len(call) / len(bs_eq)) if bs_eq else None,
                pe_coloc_time=RED._pe_time(bs_eq, lambda e: e["prefill_in_flight"] and RED._pair(e) == pi_coloc)
                if bs_eq else None,
                kernel_done_wait_frac=(sum(1 for e in win["C"] if e.get("prefill_kernel_done_wait")) / len(win["C"]))
                if win["C"] else None,
                co_spans=len(co_sp), n_new=[s["n_new"] for s, _e in co_sp],
                backlog_realized=all((en.get("queue_len") or 0) > 0 for _s, en in co_sp[:-1]) if co_sp else None,
                composition_problems=comp, delta_l=cm.get("delta_l"), fix_s=cm.get("fix_s"))
    if call:
        iv = _intervals_desc(call)
        desc["c_all"] = dict(n=len(call), s_per_median=statistics.median(iv) if iv else None,
                             ctx_mid=RED._ctx_mid(ctx, call))
        h = len(call) // 2
        for name, part in (("front", call[:h]), ("back", call[h:])):
            iv = _intervals_desc(part)
            desc[name] = dict(n=len(part), s_per_median=statistics.median(iv) if iv else None,
                              ctx_mid=RED._ctx_mid(ctx, part))
    cl = boot["client"].get(cell["cell_id"]) or {}
    desc.update(cancel_fired=cl.get("cancel_fired"), t_cancel=cl.get("t_cancel"), base=cl.get("base"))
    out.update(status="MEASURED", reason="", measurement=None, windows=st, details=det, values=vals, desc=desc,
               signature=sig, co_span_present=bool(co_sp),
               window_len={"A1": len(win["W"]), "A2": len(win["A2"]), "C": len(win["C"])},
               split_transitions=sum(1 for e in recs if e.get("event") == "split_transition"),
               _records=recs)
    return out
