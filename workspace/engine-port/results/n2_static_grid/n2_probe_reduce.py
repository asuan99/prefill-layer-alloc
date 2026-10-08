"""N2(3-a) registered reducer: raw probe telemetry -> probe.json (CPU only). PREREG_N2_3A rev9 (R7-1, R8-1/2/3).

The U-2 (a) Workspace probe is NOT a measurement and is never cited (N2C-35, N2C-40). This tool is the ONLY
registered path from raw PHASE_EVENTS telemetry to the aggregates that n2_feasibility.apply_probe accepts.

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
tau / kappa (rev9, R8-1): the windows themselves are extracted HERE from raw PHASE_EVENTS + runtime_snapshot + the
runner's cell marks (extract_cell_windows): alone window W/A1 (§6-1/§6-2: after the last burst span, bs == b,
not in flight, trim 16, ctx_p50 band [c-64, c+63], 128 iterations, PE-iter >= 0.99), W_co / C (§6-1/§7-2: co spans
after 'coload_sent' with extend_tokens == n_new*8192; C = W_co with bs == b, in flight, target pair), A2 (after the
last co span, trim 16, first 128). S-per uses only consecutive iter_seq pairs both inside the window (R8-2).
kappa element = median S-per(C) / A_on(ctx_mean(C)), A_on = §10-2 linear interpolation of the A1 and A2 medians.
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


def _pair(e):
    return (e["prefill_sms"], e["decode_sms"])


def s_per_intervals(window_iters):
    """R8-2: S-per = t_launch[i+1] - t_launch[i] ONLY for pairs with consecutive iter_seq that are BOTH in the window
    (the window is the already-filtered set). A window needs >= 128 iterations."""
    if len(window_iters) < WINDOW_ITERS:
        raise ReduceRejected(f"window has {len(window_iters)} < {WINDOW_ITERS} iterations")
    by_seq = {e["iter_seq"]: e for e in window_iters}
    out = []
    for q, e in sorted(by_seq.items()):
        n = by_seq.get(q + 1)
        if n is not None:
            out.append((n["t_launch"] - e["t_launch"]) * 1000.0)
    if not out:
        raise ReduceRejected("window has no consecutive iter_seq pair")
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
        raise ReduceRejected(f"{what}: empty window")
    frac = sum(1 for e in win if _pair(e) == pair) / len(win)
    if frac < ALONE_PE_MIN:
        raise ReduceRejected(f"{what}: PE-iter {frac:.3f} < {ALONE_PE_MIN} at {pair} (D_UNREALIZED)")


def extract_cell_windows(records, marks, cell, kind, d_alone):
    """records: PHASE_EVENTS + runtime_snapshot of ONE cell (sliced by the runner's cell_start/cell_end marks);
    marks: dict with 'coload_sent' (E-alpha only, client CLOCK_MONOTONIC); cell = (d, b, c); kind 'TAU' | 'KAPPA';
    d_alone: D of the alone windows (ON boot: d; for the probe tau/kappa cells this is always d).
    Returns dict of windows (lists of decode_iteration records) and their iter_seq ranges."""
    d, b, c = cell
    iters = [e for e in records if e.get("event") == "decode_iteration"]
    snaps = [e for e in records if e.get("event") == "runtime_snapshot"]
    starts = {e["span_seq"]: e for e in records if e.get("event") == "prefill_span_start"}
    ends = {e["span_seq"]: e for e in records if e.get("event") == "prefill_span_end"}
    spans = sorted((starts[k]["timestamp_monotonic_s"], ends[k]["timestamp_monotonic_s"], starts[k])
                   for k in starts if k in ends)
    t_co = marks.get("coload_sent") if kind == "KAPPA" else None
    if kind == "KAPPA" and t_co is None:
        raise ReduceRejected("kappa cell needs a coload_sent mark")
    burst = [sp for sp in spans if t_co is None or sp[0] < t_co]
    co = [sp for sp in spans if t_co is not None and sp[0] >= t_co and sp[2]["extend_tokens"] == sp[2]["n_new"] * CO_INPUT]
    if any(sp for sp in spans if t_co is not None and sp[0] >= t_co and sp not in co):
        raise ReduceRejected("non-co span after coload_sent (A1 not complete before trigger)")
    t_burst_end = max((sp[1] for sp in burst), default=float("-inf"))
    t_co0 = min((sp[0] for sp in co), default=float("inf"))
    t_co1 = max((sp[1] for sp in co), default=float("inf"))
    pair_alone = PAIR[d_alone]
    ctx = ctx_of(iters, snaps, b)
    seg = _alone(iters, b, t_burst_end, t_co0, pair_alone)
    w = [e for e in seg if e["iter_seq"] in ctx and c - BAND_LO <= ctx[e["iter_seq"]] <= c + BAND_HI]
    if len(w) != WINDOW_ITERS:
        raise ReduceRejected(f"centered window has {len(w)} != {WINDOW_ITERS} iterations")
    _check_alone_pair(w, pair_alone, "W/A1")
    out = dict(W=w)
    if kind == "KAPPA":
        if not co:
            raise ReduceRejected("no co span")
        wco = [e for e in iters if t_co0 <= e["t_launch"] <= t_co1]           # W_co: before the bs filter (R5-7)
        cwin = [e for e in wco if e["bs"] == b and e["prefill_in_flight"] and _pair(e) == PAIR[d]]
        a2 = _alone(iters, b, t_co1, float("inf"), pair_alone)[:WINDOW_ITERS]
        if len(a2) < WINDOW_ITERS:
            raise ReduceRejected("A2 window short")
        _check_alone_pair(a2, pair_alone, "A2")
        out.update(C=cwin, A2=a2, W_co=wco)
        out["ctx"] = {k: ctx.get(e["iter_seq"]) for k, ws in (("A1", w), ("A2", a2), ("C", cwin)) for e in ws[:1]}
        out["ctx_mean"] = {k: statistics.mean(ctx[e["iter_seq"]] for e in ws if e["iter_seq"] in ctx)
                           for k, ws in (("A1", w), ("A2", a2), ("C", cwin))}
    out["ranges"] = {k: (min(e["iter_seq"] for e in v), max(e["iter_seq"] for e in v))
                     for k, v in out.items() if k in ("W", "A2", "C") and v}
    return out


def a_on_interp(a1_ms, x1, a2_ms, x2, x):
    """§10-2: A_on(x) = linear interpolation of the A1 and A2 S-per medians at their mean ctx."""
    if x2 == x1:
        return (a1_ms + a2_ms) / 2.0
    return a1_ms + (a2_ms - a1_ms) * (x - x1) / (x2 - x1)


def kappa_parts(win):
    """one kappa measurement: (C intervals, A_on at ctx_mean(C))."""
    c_iv = s_per_intervals(win["C"])
    a1 = statistics.median(s_per_intervals(win["W"]))
    a2 = statistics.median(s_per_intervals(win["A2"]))
    m = win["ctx_mean"]
    return c_iv, a_on_interp(a1, m["A1"], a2, m["A2"], m["C"])


def build_probe(theta_boots, cell_measurements, l_lag_iters, delta_l_s, fix_s, raw_files=None, reducer_sha256=None):
    """cell_measurements: [dict(cell=(D,b,c), kind='TAU'|'KAPPA', records=[...], marks={...})] -- RAW per cell.
    TAU: pooled S-per of all W windows of that cell. KAPPA: c = pooled C intervals of all measurements of the cell,
    a_on = median over measurements of the interpolated A_on(ctx_mean(C)). Provenance is recorded (R8-3)."""
    theta, excluded, dropped = reduce_theta(theta_boots)
    tau_w, kap_c, kap_a, ranges = {}, {}, {}, []
    for m in cell_measurements:
        cell, kind = tuple(m["cell"]), m["kind"]
        win = extract_cell_windows(m["records"], m.get("marks", {}), cell, kind, cell[0])
        ranges.append(dict(cell=list(cell), kind=kind, **{k: list(v) for k, v in win["ranges"].items()}))
        if kind == "TAU":
            tau_w.setdefault(cell, []).append(win["W"])
        else:
            c_iv, a_on = kappa_parts(win)
            kap_c.setdefault(cell, []).extend(c_iv)
            kap_a.setdefault(cell, []).append(a_on)
    tau = {cell: {"s_per_median_ms": pooled_median_ms(ws), "window_iters": min(len(w) for w in ws)}
           for cell, ws in tau_w.items()}
    kappa = {cell: {"c_median_ms": statistics.median(kap_c[cell]), "a_on_ms": statistics.median(kap_a[cell]),
                    "window_iters": WINDOW_ITERS, "a_on_window_iters": WINDOW_ITERS} for cell in kap_c}
    return {"theta_spans": theta, "theta_excluded": excluded, "theta_dropped": dropped,
            "tau_cells": tau, "kappa_cells": kappa,
            "l_lag_iters": list(l_lag_iters), "delta_l_s": list(delta_l_s), "fix_s": list(fix_s),
            "provenance": dict(reducer_sha256=reducer_sha256 or self_sha256(), raw_files=dict(raw_files or {}),
                               load_calls={bid: [list(w) for w in calls] for bid, _r, calls in theta_boots},
                               cell_windows=ranges)}


def self_sha256():
    import hashlib
    with open(__file__, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


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
