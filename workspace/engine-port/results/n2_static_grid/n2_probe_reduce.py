"""N2(3-a) reducer: raw probe telemetry -> probe.json (CPU only). PREREG_N2_3A rev14 (R7-1, R8-1/2/3,
R9-1/2/5/6, R10-1/2/5, R11-1/2/3/5).

rev14 (A') STATUS: the U-2 probe is an OPTIONAL post-seal, NON-GATING harness smoke. Nothing this module computes is a
judgment input or a registered constant source (K is the registration-time constant n2_feasibility.K_CO; L_lag and
delta are fixed allowances). What REMAINS registered here is the shared window implementation that the main-measurement
analyzer (n2_analyze / n2_window) imports: extract_cell_windows (alone W/A1, W_co, the C VALUE WINDOW = FIRST 128
co-located iterations in iter_seq order [rev14 truncation], A2), s_per_intervals (iter_seq-adjacent pairs), ctx_of,
a_on_interp (§10-2). The probe is NOT a measurement and is never cited (N2C-35, N2C-40).

theta (prefill throughput) sample composition (R7-1, registered):
  * a "16,384-token span" = prefill_span_start/_end pair (same span_seq) with extend_tokens == 16384;
  * EXCLUDED, and recorded in probe.json["theta_excluded"]: the FIRST 16,384-token span of each boot (telemetry
    file) and the FIRST 16,384-token span starting inside each load call window (one client loadgen invocation);
  * kept only if realized-split evidence: >= 1 decode_iteration inside [span_start, span_end] whose (prefill_sms,
    decode_sms) is a split (0 < prefill_sms < 108, decode_sms > 0) -- entry labels are NOT used (lesson 271);
    p := that iteration's prefill_sms (all such iterations must agree, else the span is dropped and recorded);
  * decode-load condition: kept only if max decode_iteration.bs inside the span <= BIND_BS (32 = binding cells),
    so the probe's co-located decode load is not heavier than the binding E-alpha cells (N2C-43);
  * value = extend_tokens / (span_end_ts - span_start_ts); n >= 20 per p is counted AFTER exclusion (calculator).
rev10 (R9-1/R9-5): reduce_probe() reads the telemetry and marks FILES itself (raw sha256 from the bytes), enumerates
load calls and probe cells from the marks, and records every measurement as REDUCED or rejected with its registered
class (KAPPA_CENSORED / VALUE_DEPENDENT / VALUE_INDEPENDENT); probe_status follows mechanically. Completeness:
marked cells == reduced + rejected. tau (R9-2) = the A1 window of every kappa measurement.
tau / kappa (rev9, R8-1): the windows themselves are extracted HERE from raw PHASE_EVENTS + runtime_snapshot + the
runner's cell marks (extract_cell_windows): alone window W/A1 (§6-1/§6-2: after the last burst span, bs == b,
not in flight, trim 16, ctx_p50 band [c-64, c+63], 128 iterations, PE-iter >= 0.99), W_co / C (§6-1/§7-2: co spans
after 'coload_sent' with extend_tokens == n_new*8192; C = W_co with bs == b, in flight, target pair), A2 (after the
last co span, trim 16, first 128). S-per uses only consecutive iter_seq pairs both inside the window (R8-2).
kappa_i (one measurement) = median S-per(C) / A_on(ctx_mid(C)), A_on = §10-2 linear interpolation of the A1 and A2
medians (containment guard, R9-6); the cell's kappa = EQUAL-WEIGHT median of kappa_i over its measurements (rev10).
JSON input is parsed with a duplicate-key rejecting object_pairs_hook.
"""
from __future__ import annotations

import json
import statistics
import sys

BIND_BS = 32
SPAN_TOKENS = 16384
WINDOW_ITERS = 128


class ReduceRejected(ValueError):
    pass


def no_duplicate_keys(pairs):
    keys = [k for k, _ in pairs]
    if len(keys) != len(set(keys)):
        raise ReduceRejected("duplicate JSON keys")
    return dict(pairs)


def loads_strict(text):
    return json.loads(text, object_pairs_hook=no_duplicate_keys)


def read_jsonl_strict(path):
    with open(path) as fh:
        return [loads_strict(line) for line in fh if line.strip()]


def theta_from_boot(records, load_calls, boot_id="boot"):
    """records: PHASE_EVENTS dicts of ONE boot; load_calls: [(t0, t1)] client loadgen windows (same clock).
    Returns (values_by_p, excluded, dropped)."""
    starts, ends, iters = {}, {}, []
    for e in records:
        ev = e.get("event")
        if ev == "prefill_span_start":
            starts[e["span_seq"]] = e
        elif ev == "prefill_span_end":
            ends[e["span_seq"]] = e
        elif ev == "decode_iteration":
            iters.append(e)
    spans = sorted((starts[k]["timestamp_monotonic_s"], ends[k]["timestamp_monotonic_s"], k)
                   for k in starts if k in ends and starts[k].get("extend_tokens") == SPAN_TOKENS)
    first = set()
    if spans:
        first.add(spans[0][2])
    for a, b in load_calls:
        inside = [s for s in spans if a <= s[0] <= b]
        if inside:
            first.add(inside[0][2])
    out, excluded, dropped = {}, [], []
    for t0, t1, k in spans:
        rec = dict(boot=boot_id, span_seq=k, t0=t0, t1=t1)
        if t1 <= t0:
            dropped.append(dict(rec, reason="non-positive duration")); continue
        val = SPAN_TOKENS / (t1 - t0)
        if k in first:
            excluded.append(dict(rec, tok_s=val, reason="first 16384-token span of boot/load call")); continue
        its = [e for e in iters if t0 <= e["t_launch"] <= t1]
        split = [e for e in its if 0 < e["prefill_sms"] < 108 and e["decode_sms"] > 0]
        if not split:
            dropped.append(dict(rec, tok_s=val, reason="no realized split decode iteration")); continue
        ps = {e["prefill_sms"] for e in split}
        if len(ps) != 1:
            dropped.append(dict(rec, tok_s=val, reason=f"mixed splits {sorted(ps)}")); continue
        if max(e["bs"] for e in its) > BIND_BS:
            dropped.append(dict(rec, tok_s=val, reason=f"decode load above binding bs {BIND_BS}")); continue
        out.setdefault(ps.pop(), []).append(val)
    return out, excluded, dropped


def reduce_theta(boots):
    """boots: [(boot_id, records, load_calls)] -> (theta_spans {p: [...]}, excluded, dropped)"""
    theta, excluded, dropped = {}, [], []
    for boot_id, records, calls in boots:
        vals, ex, dr = theta_from_boot(records, calls, boot_id)
        for p, lst in vals.items():
            theta.setdefault(p, []).extend(lst)
        excluded += ex
        dropped += dr
    return theta, excluded, dropped


# ---- R8-1 / R8-2: registered window extraction from RAW PHASE_EVENTS (§6-1, §6-2, §7-2, §10-2) ------
TRIM = 16
BAND_LO, BAND_HI = 64, 63            # centered window band [c - 64, c + 63]
CO_INPUT = 8192
PAIR = {16: (92, 16), 24: (84, 24), 34: (74, 34), 44: (64, 44), 108: (0, 108)}   # A-agn realized pairs
ALONE_PE_MIN = 0.99
PROBE_D = (16, 44)                   # smoke cells are sticky-ON binding-type cells only (R9-7; smoke, not judgment)

# R9-1 measurement-level reject classes (registered):
KAPPA_CENSORED = "KAPPA_CENSORED"        # C < 128: kappa above the probe's own ceiling kappa* -> no seal, NO re-probe
VALUE_DEPENDENT = "VALUE_DEPENDENT"      # window / A2 / non-co span / PE / A_on containment -> probe rejected, NO re-probe
VALUE_INDEPENDENT = "VALUE_INDEPENDENT"  # missing / malformed marks, harness defect -> probe rejected, re-probe allowed


class MeasurementRejected(ReduceRejected):
    def __init__(self, klass, msg, detail=None, code="UNSPECIFIED"):
        super().__init__(msg)
        self.klass = klass
        self.detail = detail or {}
        self.code = code


SEVERITY = {KAPPA_CENSORED: 3, VALUE_DEPENDENT: 2, VALUE_INDEPENDENT: 1}
# rev14: the three constants below describe the OPTIONAL SMOKE only (not judgment inputs). K_PROBE is the smoke's own
# co-load size and is NOT the E-alpha K (that is n2_feasibility.K_CO = {16: 33, 44: 15}, a registration constant).
PROBE_CALLS = {16: 7, 44: 14}        # smoke call (= kappa cell) counts, enforced from the marks + raw (R10-2)
K_PROBE = {16: 18, 44: 8}            # smoke co-load size (sum n_new per smoke cell)
BIND_CELL = (32, 8192)               # the only smoke (b, c)
# R10-1 / H-1 / H-5: registered cell_abort / boot_abort reasons -> class. Any other reason is VALUE_DEPENDENT.
ABORT_REASONS = {"server_down": VALUE_INDEPENDENT, "http_connection_error": VALUE_INDEPENDENT,
                 "cell_timeout": VALUE_DEPENDENT, "coload_payload_invalid": VALUE_DEPENDENT,
                 "stream_cancel_failed": VALUE_DEPENDENT, "runner_stop": VALUE_DEPENDENT,
                 "client_exception": VALUE_DEPENDENT, "wiring_check_failed": VALUE_DEPENDENT,
                 "boot_health_timeout": VALUE_DEPENDENT}
# rev13 G-8: boot_health_timeout is VALUE_INDEPENDENT only as a boot_abort of a boot that started NO cell (no
# measurement exists to hide); anywhere else it is VALUE_DEPENDENT. G-9: client_crash / node_failure removed (the
# harness can never write them).
BOOT_ONLY_VI_REASONS = frozenset({"boot_health_timeout"})
# R11-5: the registered, stable reason_code enumeration (the re-probe stop rule compares these, never free text)
REASON_CODES = frozenset({
    "WINDOW_SHORT", "WINDOW_NO_PAIR", "WINDOW_EMPTY", "ALONE_PE_LOW", "COLOAD_SENT_MISSING",
    "NON_CO_SPAN_AFTER_TRIGGER", "NO_CO_SPAN", "CENTER_WINDOW", "C_CENSORED", "C_CENSORED_ABORTED", "A2_SHORT",
    "AON_CONTAINMENT", "RAW_MISSING", "TELEMETRY_TRUNCATED", "REDUCER_FAILURE", "MARK_MALFORMED", "MARK_DUPLICATE",
    "CELL_START_MISSING", "CELL_END_MISSING", "SCALAR_MARK_MISSING", "PROBE_D_INVALID", "CELL_BC_INVALID",
    "COLOAD_SIZE_MISMATCH", "CALL_WRAP", "CALL_CELLS", "CALL_DUPLICATE", "CALL_UNPAIRED", "CO_SPAN_CELLS",
    "BOOT_PLAN_MISSING", "BOOT_PLAN_SUM", "CELL_COUNT_EXCESS", "CELL_COUNT_SHORT", "CELL_COUNT_SHORT_DEATH",
    "ABORT_UNREGISTERED", "UNSPECIFIED"} | {f"ABORT_{k.upper()}" for k in ABORT_REASONS})


def abort_code(reason):
    return f"ABORT_{reason.upper()}" if reason in ABORT_REASONS else "ABORT_UNREGISTERED"


def _ts(e):
    t = e.get("timestamp_monotonic_s")
    return e["t_launch"] if t is None else t


def _pair(e):
    return (e["prefill_sms"], e["decode_sms"])


def s_per_intervals(window_iters):
    """R8-2: S-per = t_launch[i+1] - t_launch[i] ONLY for pairs with consecutive iter_seq that are BOTH in the window
    (the window is the already-filtered set). A window needs >= 128 iterations."""
    if len(window_iters) < WINDOW_ITERS:
        raise MeasurementRejected(VALUE_DEPENDENT, f"window has {len(window_iters)} < {WINDOW_ITERS} iterations",
                                  code="WINDOW_SHORT")
    by_seq = {e["iter_seq"]: e for e in window_iters}
    out = []
    for q, e in sorted(by_seq.items()):
        n = by_seq.get(q + 1)
        if n is not None:
            out.append((n["t_launch"] - e["t_launch"]) * 1000.0)
    if not out:
        raise MeasurementRejected(VALUE_DEPENDENT, "window has no consecutive iter_seq pair", code="WINDOW_NO_PAIR")
    return out


def pooled_median_ms(windows):
    pooled = []
    for w in windows:
        pooled += s_per_intervals(w)
    return statistics.median(pooled)


def ctx_of(iters, snapshots, b):
    """§6-2: ctx_p50(iteration) = latest valid snapshot context_p50 + (iter_seq - snapshot.decode_iterations).
    Valid snapshot: active_decode_sequences == b (excludes post-merge, pre-filter snapshots, engine edit 5)."""
    snaps = sorted((s_["decode_iterations"], s_["context_p50"]) for s_ in snapshots
                   if s_.get("active_decode_sequences") == b)
    out = {}
    k = -1
    for e in sorted(iters, key=lambda x: x["iter_seq"]):
        while k + 1 < len(snaps) and snaps[k + 1][0] <= e["iter_seq"]:
            k += 1
        if k >= 0:
            out[e["iter_seq"]] = snaps[k][1] + (e["iter_seq"] - snaps[k][0])
    return out


def _alone(iters, b, t_after, t_before, pair):
    seg = [e for e in iters if t_after < e["t_launch"] < t_before and e["bs"] == b and not e["prefill_in_flight"]]
    seg.sort(key=lambda e: e["iter_seq"])
    seg = seg[TRIM:]
    return seg


def _check_alone_pair(win, pair, what):
    if not win:
        raise MeasurementRejected(VALUE_DEPENDENT, f"{what}: empty window", code="WINDOW_EMPTY")
    frac = sum(1 for e in win if _pair(e) == pair) / len(win)
    if frac < ALONE_PE_MIN:
        raise MeasurementRejected(VALUE_DEPENDENT, f"{what}: PE-iter {frac:.3f} < {ALONE_PE_MIN} at {pair} (D_UNREALIZED)",
                                  code="ALONE_PE_LOW")


def extract_cell_windows(records, marks, cell):
    """records: PHASE_EVENTS + runtime_snapshot of ONE probe cell (sliced by the reducer from the cell marks);
    marks: dict with 'coload_sent' (client CLOCK_MONOTONIC); cell = (d, b, c), sticky ON, d in PROBE_D.
    Returns the windows W(=A1), W_co, C, A2, their realized ctx means (ctx_mid) and iter_seq ranges, and L_lag.
    Raises MeasurementRejected with the registered class (R9-1)."""
    d, b, c = cell
    iters = [e for e in records if e.get("event") == "decode_iteration"]
    snaps = [e for e in records if e.get("event") == "runtime_snapshot"]
    starts = {e["span_seq"]: e for e in records if e.get("event") == "prefill_span_start"}
    ends = {e["span_seq"]: e for e in records if e.get("event") == "prefill_span_end"}
    spans = sorted((starts[k]["timestamp_monotonic_s"], ends[k]["timestamp_monotonic_s"], starts[k])
                   for k in starts if k in ends)
    t_co = marks.get("coload_sent")
    if t_co is None:
        raise MeasurementRejected(VALUE_DEPENDENT, "kappa cell needs a coload_sent mark", code="COLOAD_SENT_MISSING")
    burst = [sp for sp in spans if sp[0] < t_co]
    co = [sp for sp in spans if sp[0] >= t_co and sp[2]["extend_tokens"] == sp[2]["n_new"] * CO_INPUT]
    if any(sp for sp in spans if sp[0] >= t_co and sp not in co):
        raise MeasurementRejected(VALUE_DEPENDENT, "non-co span after coload_sent (A1 not complete before trigger)",
                                  code="NON_CO_SPAN_AFTER_TRIGGER")
    if not co:
        raise MeasurementRejected(VALUE_DEPENDENT, "no co span", code="NO_CO_SPAN")
    t_burst_end = max((sp[1] for sp in burst), default=float("-inf"))
    t_co0 = min(sp[0] for sp in co)
    t_co1 = max(sp[1] for sp in co)
    pair = PAIR[d]
    ctx = ctx_of(iters, snaps, b)
    seg = _alone(iters, b, t_burst_end, t_co0, pair)
    w = [e for e in seg if e["iter_seq"] in ctx and c - BAND_LO <= ctx[e["iter_seq"]] <= c + BAND_HI]
    if len(w) != WINDOW_ITERS:
        raise MeasurementRejected(VALUE_DEPENDENT, f"centered window has {len(w)} != {WINDOW_ITERS} iterations",
                                  code="CENTER_WINDOW")
    _check_alone_pair(w, pair, "W/A1")
    wco = [e for e in iters if t_co0 <= e["t_launch"] <= t_co1]           # W_co: before the bs filter (R5-7)
    call = sorted((e for e in wco if e["bs"] == b and e["prefill_in_flight"] and _pair(e) == pair),
                  key=lambda e: e["iter_seq"])                              # all co-located iterations (descriptive)
    cwin = call[:WINDOW_ITERS]                                              # rev14 (A'): C VALUE WINDOW = first 128
    if len(cwin) < WINDOW_ITERS:
        bs_eq = [e for e in wco if e["bs"] == b]
        inflt = [e for e in bs_eq if e["prefill_in_flight"]]
        detail = dict(w_co=len(wco), c=len(cwin), loss_bs_ne_b=len(wco) - len(bs_eq),
                      loss_not_in_flight=len(bs_eq) - len(inflt), loss_pair_mismatch=len(inflt) - len(cwin))
        raise MeasurementRejected(KAPPA_CENSORED, f"C has {len(cwin)} < {WINDOW_ITERS} iterations (W_co {len(wco)}; "
                                                  f"censored at the probe K, R9-1 / R10-5)", detail, code="C_CENSORED")
    a2 = _alone(iters, b, t_co1, float("inf"), pair)[:WINDOW_ITERS]
    if len(a2) < WINDOW_ITERS:
        raise MeasurementRejected(VALUE_DEPENDENT, "A2 window short", code="A2_SHORT")
    _check_alone_pair(a2, pair, "A2")
    out = dict(W=w, W_co=wco, C=cwin, C_all=call, A2=a2)
    out["ctx_mid"] = {k: statistics.mean(ctx[e["iter_seq"]] for e in ws if e["iter_seq"] in ctx)
                      for k, ws in (("A1", w), ("A2", a2), ("C", cwin), ("C_all", call))}
    out["ranges"] = {k: (min(e["iter_seq"] for e in out[k]), max(e["iter_seq"] for e in out[k]))
                     for k in ("W", "W_co", "C", "C_all", "A2")}
    out["n_colocated_all"] = len(call)                                      # descriptive: whole-W_co co-located count
    out["l_lag_iters"] = sum(1 for e in iters if t_co <= e["t_launch"] < t_co0)
    return out


def a_on_interp(a1_ms, x1, a2_ms, x2, x):
    """§10-2: A_on(x) = linear interpolation of the A1 and A2 S-per medians at their ctx_mid.
    R9-6: x must lie in [min(x1, x2), max(x1, x2)] (containment guard, no constant) -- otherwise the measurement
    is rejected (VALUE_DEPENDENT); by construction (A1 < C < A2 in time, ctx monotone) it lies inside."""
    if not min(x1, x2) <= x <= max(x1, x2):
        raise MeasurementRejected(VALUE_DEPENDENT, f"ctx_mid(C) {x} outside [ctx_mid(A1), ctx_mid(A2)] (A_on containment)",
                                  code="AON_CONTAINMENT")
    if x2 == x1:
        return (a1_ms + a2_ms) / 2.0
    return a1_ms + (a2_ms - a1_ms) * (x - x1) / (x2 - x1)


def kappa_parts(win):
    """one kappa measurement: (median S-per(C), A_on at ctx_mid(C)) with C = the FIRST-128 value window (rev14);
    its kappa_i = c / a_on. Descriptive whole-W_co variants are computed by the analyzer, not here."""
    c_med = statistics.median(s_per_intervals(win["C"]))
    a1 = statistics.median(s_per_intervals(win["W"]))
    a2 = statistics.median(s_per_intervals(win["A2"]))
    m = win["ctx_mid"]
    return c_med, a_on_interp(a1, m["A1"], a2, m["A2"], m["C"])


# ---- R9-1 / R9-5: the reducer reads the files itself, enumerates calls and cells from the marks ------------------
def sha256_file(path):
    import hashlib
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def self_sha256():
    return sha256_file(__file__)


def load_calls_from_marks(marks_lines):
    """R8-3: load-call windows ONLY from client CLOCK_MONOTONIC marks 'load_call_start <id> <t>' /
    'load_call_end <id> <t>' (N3 bench mark format 'bench_start <t>' / 'bench_end <t>' is the legacy one-call case)."""
    starts, ends = {}, {}
    for line in marks_lines:
        parts = line.split()
        if len(parts) == 3 and parts[0] == "load_call_start":
            starts[parts[1]] = float(parts[2])
        elif len(parts) == 3 and parts[0] == "load_call_end":
            ends[parts[1]] = float(parts[2])
    if set(starts) != set(ends):
        raise ReduceRejected("unpaired load_call marks")
    return [(starts[k], ends[k]) for k in sorted(starts, key=lambda k: starts[k])]


def parse_marks(marks_lines):
    """MARK FORMAT (§14-2, registered; H-1, rev12 R11-1/2/3). One mark per line, single-space fields, times are
    CLOCK_MONOTONIC seconds of the same container as the engine (time.monotonic / perf_counter):
      boot_plan <d> <n>                       runner, at boot start, once per d it will run (R11-2)
      cell_start <cell_id> <d> <b> <c> <t>    runner, before the cell's burst is sent
      load_call_start <call_id> <t>           client, before the call's first request (one call per cell, unique id)
      coload_sent <cell_id> <t>               client, read immediately BEFORE the first co-load request is sent
      cell_abort <cell_id> <reason> <t>       runner, in `finally` when the cell did not complete (reason table)
      abort_detail <cell_id> <exception_class>  runner, with cell_abort (descriptive, H-4)
      cell_end <cell_id> <t>                  runner, in `finally` after the cell is drained (always written)
      load_call_end <call_id> <t>             client, in `finally` after the last response / cancellation
      delta_l <cell_id> <s> / fix_s <cell_id> <s>   client-measured scalars, after the cell
      boot_abort <reason> <t>                 runner, in `finally` when the boot died or was stopped (reason table)
    Returns (cells, calls, plan, boot_aborts, issues). Duplicates and malformed lines are issues, never dropped or
    overwritten. issues: (klass, code, text)."""
    cells, calls, plan, boot_aborts, issues = {}, {}, {}, [], []
    arity = {"cell_start": 5, "cell_end": 2, "coload_sent": 2, "delta_l": 2, "fix_s": 2, "cell_abort": 3,
             "abort_detail": 2, "load_call_start": 2, "load_call_end": 2, "boot_plan": 2, "boot_abort": 2}
    for line in marks_lines:
        parts = line.split()
        if not parts:
            continue
        tag, rest = parts[0], parts[1:]
        if tag not in arity or len(rest) != arity[tag]:
            issues.append((VALUE_INDEPENDENT, "MARK_MALFORMED", f"malformed mark: {line}")); continue
        try:
            if tag == "boot_plan":
                d, n = int(rest[0]), int(rest[1])
                if d in plan:
                    issues.append((VALUE_DEPENDENT, "MARK_DUPLICATE", f"duplicate boot_plan for d={d}")); continue
                plan[d] = n
            elif tag == "boot_abort":
                boot_aborts.append((rest[0], float(rest[1])))
            elif tag in ("load_call_start", "load_call_end"):
                cl = calls.setdefault(rest[0], {})
                if tag in cl:
                    issues.append((VALUE_DEPENDENT, "CALL_DUPLICATE", f"duplicate {tag} for call id {rest[0]}"))
                    cl["dup"] = True
                    continue
                cl[tag] = float(rest[1])
            else:
                cm = cells.setdefault(rest[0], {})
                if tag in cm:
                    issues.append((VALUE_DEPENDENT, "MARK_DUPLICATE", f"duplicate {tag} for cell id {rest[0]}"))
                    cm["dup"] = True
                    continue
                if tag == "cell_start":
                    cm[tag] = ((int(rest[1]), int(rest[2]), int(rest[3])), float(rest[4]))
                elif tag == "cell_abort":
                    cm[tag] = (rest[1], float(rest[2]))
                elif tag == "abort_detail":
                    cm[tag] = rest[1]
                else:
                    cm[tag] = float(rest[1])
        except ValueError:
            issues.append((VALUE_INDEPENDENT, "MARK_MALFORMED", f"malformed mark: {line}"))
    return cells, calls, plan, boot_aborts, issues


def _slice(records, t0, t1):
    return [e for e in records if t0 <= _ts(e) <= t1]


def _read_telemetry(path):
    """Strict JSONL; a non-parsable LAST line is a truncated write (boot death) and is dropped and reported.
    Any other unparsable line is corruption (ReduceRejected)."""
    with open(path) as fh:
        lines = [ln for ln in fh.read().split("\n") if ln.strip()]
    out, truncated = [], False
    for k, ln in enumerate(lines):
        try:
            out.append(loads_strict(ln))
        except (ValueError, ReduceRejected):
            if k == len(lines) - 1:
                truncated = True
            else:
                raise ReduceRejected(f"corrupt telemetry line {k + 1}")
    return out, truncated


def _co_spans(records, t_co):
    starts = {e["span_seq"]: e for e in records if e.get("event") == "prefill_span_start"}
    ends = {e["span_seq"]: e for e in records if e.get("event") == "prefill_span_end"}
    return [(starts[k], ends.get(k)) for k in starts
            if starts[k]["timestamp_monotonic_s"] >= t_co and starts[k].get("n_new", 0) >= 1
            and starts[k].get("extend_tokens") == starts[k]["n_new"] * CO_INPUT]


def _c_count(records, cell, co):
    """|W_co| and |C| from the co spans alone (no A1/A2 needed) -- used to promote an aborted cell (R11-3)."""
    d, b, c = cell
    t0 = min(s["timestamp_monotonic_s"] for s, _e in co)
    t1 = max(e["timestamp_monotonic_s"] for _s, e in co)
    wco = [e for e in records if e.get("event") == "decode_iteration" and t0 <= e["t_launch"] <= t1]
    return len(wco), sum(1 for e in wco if e["bs"] == b and e["prefill_in_flight"] and _pair(e) == PAIR[d])


def _measure(cid, cm, records, calls):
    """R10-1 / R11-1 / R11-3: every applicable check runs; the cell's class is the MOST severe (censored > VD > VI).
    Issues are (klass, reason_code, text[, detail])."""
    issues, out = [], None
    if cm.get("dup"):
        issues.append((VALUE_DEPENDENT, "MARK_DUPLICATE", "duplicate cell id"))
    if "cell_start" not in cm:
        issues.append((VALUE_DEPENDENT, "CELL_START_MISSING", "marks for a cell id without cell_start"))
        return issues, None
    cell, t0 = cm["cell_start"]
    if cell[0] not in PROBE_D:
        issues.append((VALUE_INDEPENDENT, "PROBE_D_INVALID", f"probe cell d={cell[0]} not in {PROBE_D}"))
    if tuple(cell[1:]) != BIND_CELL:                                            # R11-1
        issues.append((VALUE_DEPENDENT, "CELL_BC_INVALID", f"probe cell (b, c)={tuple(cell[1:])} != {BIND_CELL}"))
    if "coload_sent" in cm:
        n_calls = sum(1 for a, b in calls if a <= cm["coload_sent"] <= b)
        if n_calls != 1:
            issues.append((VALUE_DEPENDENT, "CALL_WRAP", f"coload_sent inside {n_calls} load calls (must be exactly 1)"))
    co = None
    if cell[0] in PROBE_D and "coload_sent" in cm:
        t_end = cm.get("cell_end", float("inf"))
        co = _co_spans(_slice(records, t0, t_end), cm["coload_sent"])
        if "cell_end" in cm and sum(s["n_new"] for s, _e in co) != K_PROBE[cell[0]]:          # R11-1
            issues.append((VALUE_DEPENDENT, "COLOAD_SIZE_MISMATCH",
                           f"co-load sum n_new {sum(s['n_new'] for s, _e in co)} != K_PROBE {K_PROBE[cell[0]]}"))
    abort = cm.get("cell_abort")
    if abort is not None:                         # aborted: reason class, plus censoring promotion (R11-3)
        reason, t_ab = abort
        issues.append((ABORT_REASONS.get(reason, VALUE_DEPENDENT), abort_code(reason), f"cell_abort {reason}"))
        if (co and "cell_end" in cm and sum(s["n_new"] for s, _e in co) == K_PROBE[cell[0]]
                and all(e is not None and e["timestamp_monotonic_s"] <= t_ab for _s, e in co)):
            w_co, n_c = _c_count(_slice(records, t0, cm["cell_end"]), cell, co)
            if n_c < WINDOW_ITERS:
                issues.append((KAPPA_CENSORED, "C_CENSORED_ABORTED", f"aborted after a complete co-load with C {n_c}",
                               dict(w_co=w_co, c=n_c)))
        return issues, None
    for key, code in (("cell_end", "CELL_END_MISSING"), ("coload_sent", "COLOAD_SENT_MISSING")):
        if key not in cm:                         # without cell_abort a missing cell_end / coload_sent is VD
            issues.append((VALUE_DEPENDENT, code, f"{key} mark missing without cell_abort"))
    if cell[0] in PROBE_D and "cell_end" in cm and "coload_sent" in cm:
        try:
            win = extract_cell_windows(_slice(records, t0, cm["cell_end"]), {"coload_sent": cm["coload_sent"]}, cell)
            c_med, a_on = kappa_parts(win)
            out = dict(win=win, c_med=c_med, a_on=a_on)
        except MeasurementRejected as e:
            issues.append((e.klass, e.code, str(e)) + ((e.detail,) if e.detail else ()))
    for key in ("delta_l", "fix_s"):              # evaluated AFTER extraction (R10-1)
        if key not in cm:
            issues.append((VALUE_INDEPENDENT, "SCALAR_MARK_MISSING", f"{key} mark missing"))
    return issues, out


def _issue(boot, klass, code, text):
    return dict(boot=boot, status=klass, reason_code=code, reason=text)


def reduce_probe(boots, expected_calls=None):
    """boots: [dict(boot_id=str, telemetry=path, marks=path)] -- the ONLY registered path raw files -> probe.json.
    Raw-file sha256 from the bytes (R9-5); calls, cells and the boot plan from the marks (H-1 format); every
    measurement REDUCED or recorded with its most severe class and reason_code (R10-1, R11-5); completeness against
    the RAW co spans, unique ids, call<->cell 1:1, per-boot plan and death (R10-2, R11-1, R11-2); probe_status = most
    severe issue anywhere; reason_codes = the codes of that class. expected_calls defaults to PROBE_CALLS and is
    recorded (the calculator re-reduces with the default, R10-3)."""
    expected = dict(PROBE_CALLS if expected_calls is None else expected_calls)
    raw_files, theta_boots, measurements, boot_issues, tau_w, kap = {}, [], [], [], {}, {}
    l_lag, delta_l, fix_s = [], [], []
    plan_total, plans_known, n_cells = {d: 0 for d in PROBE_D}, True, 0
    for bt in boots:
        bid = bt["boot_id"]
        try:
            for k in ("telemetry", "marks"):
                raw_files[bt[k]] = sha256_file(bt[k])
        except OSError as e:
            boot_issues.append(_issue(bid, VALUE_INDEPENDENT, "RAW_MISSING", f"raw file missing: {e}"))
            plans_known = False
            continue
        try:
            records, truncated = _read_telemetry(bt["telemetry"])
            with open(bt["marks"]) as fh:
                mlines = [ln for ln in fh.read().splitlines() if ln.strip()]
        except Exception as e:                                                  # R10-1: not raw absence -> VD
            boot_issues.append(_issue(bid, VALUE_DEPENDENT, "REDUCER_FAILURE", f"reducer failure: {e!r}"))
            plans_known = False
            continue
        cells, calls_d, plan, boot_aborts, mark_issues = parse_marks(mlines)
        for cls, code, why in mark_issues:
            boot_issues.append(_issue(bid, cls, code, why))
        calls = []
        for cid_, cl in calls_d.items():
            if "load_call_start" in cl and "load_call_end" in cl:
                calls.append((cl["load_call_start"], cl["load_call_end"]))
            else:
                boot_issues.append(_issue(bid, VALUE_DEPENDENT, "CALL_UNPAIRED", f"unpaired load_call marks {cid_}"))
        calls.sort()
        death_t = None
        if truncated:
            boot_issues.append(_issue(bid, VALUE_INDEPENDENT, "TELEMETRY_TRUNCATED", "telemetry truncated (boot death)"))
            death_t = max((_ts(e) for e in records), default=float("-inf"))
        any_cell = any("cell_start" in cm for cm in cells.values())
        for ab, t_ab in boot_aborts:
            cls = ABORT_REASONS.get(ab, VALUE_DEPENDENT)
            if ab in BOOT_ONLY_VI_REASONS and not any_cell:
                cls = VALUE_INDEPENDENT
            boot_issues.append(_issue(bid, cls, abort_code(ab), f"boot_abort {ab}"))
            if cls == VALUE_INDEPENDENT:
                death_t = t_ab if death_t is None else min(death_t, t_ab)
        theta_boots.append((bid, records, calls))
        # R10-2: every raw co-type span must start inside exactly one marked cell window
        wins = [(cm["cell_start"][1], cm.get("cell_end", float("inf"))) for cm in cells.values() if "cell_start" in cm]
        for e in (x for x in records if x.get("event") == "prefill_span_start"):
            if e.get("n_new", 0) >= 1 and e.get("extend_tokens") == e["n_new"] * CO_INPUT:
                n_in = sum(1 for a, b in wins if a <= e["timestamp_monotonic_s"] <= b)
                if n_in != 1:
                    boot_issues.append(_issue(bid, VALUE_DEPENDENT, "CO_SPAN_CELLS",
                                              f"co span {e['span_seq']} inside {n_in} marked cells"))
        # R11-1: each load call wraps exactly one cell's coload_sent
        sent = [cm["coload_sent"] for cm in cells.values() if "coload_sent" in cm]
        for a, b in calls:
            k = sum(1 for t in sent if a <= t <= b)
            if k != 1:
                boot_issues.append(_issue(bid, VALUE_DEPENDENT, "CALL_CELLS", f"load call [{a}, {b}] wraps {k} cells"))
        # R11-2: per-boot plan and per-boot death
        if not plan:
            boot_issues.append(_issue(bid, VALUE_DEPENDENT, "BOOT_PLAN_MISSING", "no boot_plan mark"))
        started = {d: [cm["cell_start"][1] for cm in cells.values()
                       if "cell_start" in cm and cm["cell_start"][0][0] == d] for d in PROBE_D}
        for d in PROBE_D:
            plan_total[d] += plan.get(d, 0)
            n_started, n_plan = len(started[d]), plan.get(d, 0)
            if n_started > n_plan:
                boot_issues.append(_issue(bid, VALUE_DEPENDENT, "CELL_COUNT_EXCESS",
                                          f"d={d}: {n_started} cells started, boot_plan {n_plan}"))
            elif n_started < n_plan:
                last = max((t for dd in PROBE_D for t in started[dd]), default=float("-inf"))
                if death_t is not None and last <= death_t:
                    boot_issues.append(_issue(bid, VALUE_INDEPENDENT, "CELL_COUNT_SHORT_DEATH",
                                              f"d={d}: {n_started} of {n_plan} cells, boot died after its last cell"))
                else:
                    boot_issues.append(_issue(bid, VALUE_DEPENDENT, "CELL_COUNT_SHORT",
                                              f"d={d}: {n_started} of {n_plan} cells started"))
        n_cells += len(cells)
        for cid in sorted(cells, key=lambda k: cells[k].get("cell_start", ((), float("inf")))[1]):
            cm = cells[cid]
            issues, out = _measure(cid, cm, records, calls)
            rec = dict(boot=bid, cell_id=cid, cell=list(cm["cell_start"][0]) if "cell_start" in cm else [])
            if "abort_detail" in cm:
                rec["abort_detail"] = cm["abort_detail"]
            if issues:
                worst = max(issues, key=lambda x: SEVERITY[x[0]])
                measurements.append(dict(rec, status=worst[0], reason_code=worst[1], reason=worst[2],
                                         all_issues=[list(x[:3]) for x in issues],
                                         **({"censor_detail": worst[3]} if len(worst) > 3 else {})))
                continue
            win, cell = out["win"], tuple(cm["cell_start"][0])
            tau_w.setdefault(cell, []).append(win["W"])                           # R9-2: tau from the kappa A1 window
            kap.setdefault(cell, []).append(out["c_med"] / out["a_on"])
            l_lag.append(win["l_lag_iters"]); delta_l.append(cm["delta_l"]); fix_s.append(cm["fix_s"])
            measurements.append(dict(rec, status="REDUCED", kappa=out["c_med"] / out["a_on"], c_median_ms=out["c_med"],
                                     a_on_ms=out["a_on"], ctx_mid=win["ctx_mid"],
                                     ranges={k: list(v) for k, v in win["ranges"].items()},
                                     l_lag_iters=win["l_lag_iters"]))
    if len(measurements) != n_cells:                                            # reducer-internal no-drop check
        raise ReduceRejected(f"internal: {n_cells} marked cells but {len(measurements)} recorded")
    if plans_known:                                                             # R11-2: plans add up to PROBE_CALLS
        for d in PROBE_D:
            if plan_total[d] != expected.get(d, 0):
                boot_issues.append(_issue(None, VALUE_DEPENDENT, "BOOT_PLAN_SUM",
                                          f"d={d}: boot plans sum to {plan_total[d]}, registered {expected.get(d, 0)}"))
    theta, excluded, dropped = reduce_theta(theta_boots)
    tau = {cell: {"s_per_median_ms": pooled_median_ms(ws), "window_iters": min(len(w) for w in ws)}
           for cell, ws in tau_w.items()}
    kappa = {cell: {"kappa_median": statistics.median(ks), "n_measurements": len(ks),
                    "window_iters": WINDOW_ITERS, "a_on_window_iters": WINDOW_ITERS} for cell, ks in kap.items()}
    issues_all = [(m["status"], m["reason_code"]) for m in measurements if m["status"] != "REDUCED"] + \
                 [(m["status"], m["reason_code"]) for m in boot_issues]
    statuses = {s for s, _c in issues_all}
    status = next((s for s in (KAPPA_CENSORED, VALUE_DEPENDENT, VALUE_INDEPENDENT) if s in statuses), "OK")
    return {"theta_spans": theta, "theta_excluded": excluded, "theta_dropped": dropped,
            "tau_cells": tau, "kappa_cells": kappa, "l_lag_iters": l_lag, "delta_l_s": delta_l, "fix_s": fix_s,
            "probe_status": status, "reason_codes": sorted({c for s, c in issues_all if s == status}),
            "measurements": measurements, "boot_issues": boot_issues,
            "provenance": dict(reducer_sha256=self_sha256(), raw_files=raw_files, expected_calls=expected,
                               boots=[dict(boot_id=b["boot_id"], telemetry=b["telemetry"], marks=b["marks"])
                                      for b in boots],
                               load_calls={bid: [list(w) for w in calls] for bid, _r, calls in theta_boots})}


def to_json(probe):
    def key(k):
        return json.dumps(list(k)) if isinstance(k, tuple) else str(k)
    out = {}
    for k, v in probe.items():
        out[k] = {key(kk): vv for kk, vv in v.items()} if isinstance(v, dict) else v
    return json.dumps(out, indent=1, sort_keys=True)


def n3_theta_positive_control(n3_dir):
    """Positive control on N3 run-3 raw telemetry (design recount, cite-forbidden): bench window = load call."""
    import glob
    import os
    boots = []
    for path in sorted(glob.glob(os.path.join(n3_dir, "tel_g_*.jsonl"))):
        unit = os.path.basename(path)[4:-6]
        with open(os.path.join(n3_dir, f"marks_{unit}.txt")) as fh:
            marks = dict(line.split() for line in fh if line.strip())
        boots.append((unit, read_jsonl_strict(path), [(float(marks["bench_start"]), float(marks["bench_end"]))]))
    return reduce_theta(boots)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--n3-theta"]:
        th, ex, dr = n3_theta_positive_control(sys.argv[2])
        for p, lst in th.items():
            print(f"p={p} kept={len(lst)} median={statistics.median(lst):.1f}")
        print(f"excluded={len(ex)} dropped={len(dr)}")
