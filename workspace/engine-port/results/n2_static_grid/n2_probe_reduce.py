"""N2(3-a) registered reducer: raw probe telemetry -> probe.json (CPU only). PREREG_N2_3A rev8 R7-1.

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
tau / kappa: one element per cell = median S-per over the registered window(s); several measurements of the same
cell are pooled (median of ALL S-per intervals of all measurements of that cell).
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


def s_per_intervals(window_iters):
    """window_iters: decode_iteration records of ONE registered window (already filtered by the harness)."""
    if len(window_iters) < WINDOW_ITERS:
        raise ReduceRejected(f"window has {len(window_iters)} < {WINDOW_ITERS} iterations")
    ts = sorted(e["t_launch"] for e in window_iters)
    return [(b - a) * 1000.0 for a, b in zip(ts, ts[1:])]


def pooled_median_ms(windows):
    pooled = []
    for w in windows:
        pooled += s_per_intervals(w)
    return statistics.median(pooled)


def build_probe(theta_boots, tau_windows, kappa_windows, l_lag_iters, delta_l_s, fix_s):
    """tau_windows: {(D,b,c): [window, ...]} (several measurements -> pooled median);
    kappa_windows: {(d,b,c): {"C": [window, ...], "A_on": [window, ...]}}."""
    theta, excluded, dropped = reduce_theta(theta_boots)
    tau = {cell: {"s_per_median_ms": pooled_median_ms(ws), "window_iters": min(len(w) for w in ws)}
           for cell, ws in tau_windows.items()}
    kappa = {}
    for cell, d in kappa_windows.items():
        kappa[cell] = {"c_median_ms": pooled_median_ms(d["C"]), "a_on_ms": pooled_median_ms(d["A_on"]),
                       "window_iters": min(len(w) for w in d["C"]),
                       "a_on_window_iters": min(len(w) for w in d["A_on"])}
    return {"theta_spans": theta, "theta_excluded": excluded, "theta_dropped": dropped,
            "tau_cells": tau, "kappa_cells": kappa,
            "l_lag_iters": list(l_lag_iters), "delta_l_s": list(delta_l_s), "fix_s": list(fix_s)}


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
