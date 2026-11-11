#!/usr/bin/env python3
"""Realized decode SM count of the PD-mux bundle (motivation harness, 2026-11-11).  CPU only.

Measurement boots carry no PD-mux instrumentation, so the realized split is read in ONE separate boot per Job (role "split_check",
the bundle arm plus the env below; its runs never enter a verdict table).  Two channels, both inside the server process:

  scheduler : telemetry `decode_iteration` events (PDMUX_TELEMETRY_PATH + PDMUX_PHASE_EVENTS=1) in the run's load window
              -> which decode SM count each decode step used, split by "prefill in flight" (co-resident) or not
  hardware  : green-context read-out (PDMUX_GREEN_READOUT=1): the driver's SM count of each stream group's green context, asked through
              the stream (src/multiplex/green_readout.py; the channel that exposed the Stage 0 "D108 was 16 SM" error)

matches_target = every co-resident decode step used the registered D, every stream group those steps used has a green context within
SM_TOL of (D, 108 - D), and no read-out field is missing.  Non-sticky static (decision 49): steps without a prefill in flight run on all
SMs and are reported, not judged.
"""
from __future__ import annotations

import json
from pathlib import Path

TOTAL_SM = 108
SM_TOL = 4                      # same tolerance as cap_char/costterms_harness/ct_spec.SM_TOL (§1-8)
MIN_COSHARE = 0.99              # share of co-resident decode steps that must sit on the registered D
SPLIT_CHECK_ENV = {"PDMUX_TELEMETRY_PATH": "{dir}/tel.jsonl", "PDMUX_PHASE_EVENTS": "1", "PDMUX_GREEN_READOUT": "1",
                   "PDMUX_GREEN_READOUT_PATH": "{dir}/green.json"}


def read_events(path, kinds=("decode_iteration", "split_transition")):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("event") in kinds:
                out.append(r)
    return out


def dwell(events, t0, t1, target_d):
    """Decode steps whose event time is in [t0, t1] (time.monotonic of the same host; client and server share the clock)."""
    it = [e for e in events if e.get("event") == "decode_iteration" and t0 <= float(e.get("timestamp_monotonic_s", -1)) <= t1]
    tr = [e for e in events if e.get("event") == "split_transition" and t0 <= float(e.get("timestamp_monotonic_s", -1)) <= t1]

    def step_s(e):
        a, b = e.get("t_issue_start"), e.get("t_sync")
        return (float(b) - float(a)) if a is not None and b is not None else 0.0

    by = {}
    for e in it:
        k = ("co" if e.get("prefill_in_flight") else "alone", int(e.get("decode_sms", -1)))
        d = by.setdefault(k, {"steps": 0, "time_s": 0.0, "stream_idx": set()})
        d["steps"] += 1
        d["time_s"] += step_s(e)
        d["stream_idx"].add(e.get("stream_idx"))
    co = [e for e in it if e.get("prefill_in_flight")]
    on_target = sum(1 for e in co if int(e.get("decode_sms", -1)) == int(target_d))
    return {"n_steps": len(it), "n_coresident": len(co), "coresident_share_on_target": (on_target / len(co)) if co else None,
            "coresident_stream_idx": sorted({e.get("stream_idx") for e in co}, key=lambda x: (x is None, x)),
            "n_split_transitions": len(tr),
            "by_state_decode_sms": [{"state": s, "decode_sms": k, "steps": v["steps"], "time_s": round(v["time_s"], 6),
                                     "stream_idx": sorted(v["stream_idx"], key=lambda x: (x is None, x))}
                                    for (s, k), v in sorted(by.items())]}


def green_groups(green, stream_idx, target_d, tol=SM_TOL):
    """Green read-out of the stream groups used by co-resident steps -> (rows, ok, missing_reason)."""
    if not green or green.get("error"):
        return [], False, f"green read-out missing or failed: {(green or {}).get('error', 'no file')}"
    groups = {g.get("stream_index"): g for g in green.get("groups", [])}
    rows, ok, why = [], True, None
    for i in stream_idx:
        g = groups.get(i)
        if g is None:
            rows.append({"stream_idx": i, "decode_sm": None, "prefill_sm": None})
            ok, why = False, f"stream group {i} absent from the read-out"
            continue
        dr = ((g.get("decode") or {}).get("green_sm") or {}).get("smCount")
        pr = ((g.get("prefill") or {}).get("green_sm") or {}).get("smCount")
        rows.append({"stream_idx": i, "decode_sm": dr, "prefill_sm": pr, "target_decode_sm": g.get("target_decode_sm"),
                     "target_prefill_sm": g.get("target_prefill_sm")})
        if dr is None or pr is None:
            ok, why = False, f"stream group {i}: green SM count not answered"
        elif abs(dr - target_d) > tol or abs(pr - (TOTAL_SM - target_d)) > tol or dr + pr > TOTAL_SM:
            ok = False
    return rows, ok, why


def realized(bdir, run, target_d):
    """Split-check run -> realized_split record (schema: {"available": bool, ...})."""
    bdir = Path(bdir)
    tel, gp = bdir / "tel.jsonl", bdir / "green.json"
    if not tel.exists():
        return {"available": False, "target_D": target_d, "reason": "split-check boot wrote no telemetry (tel.jsonl)"}
    dw = dwell(read_events(tel), run["t_load_start_mono"], run["t_load_end_mono"], target_d)
    green = json.loads(gp.read_text()) if gp.exists() else None
    rows, gok, gwhy = green_groups(green, [i for i in dw["coresident_stream_idx"] if i is not None], target_d)
    if dw["n_coresident"] == 0:
        match, why = None, "no decode step with a prefill in flight in the load window"
    else:
        sched_ok = dw["coresident_share_on_target"] >= MIN_COSHARE
        match = bool(sched_ok and gok)
        why = gwhy or (None if sched_ok else f"co-resident share on D{target_d} = {dw['coresident_share_on_target']:.3f} < {MIN_COSHARE}")
    return {"available": True, "target_D": target_d, "matches_target": match, "reason": why, "scheduler": dw, "green": rows,
            "rule": f"co-resident share on D >= {MIN_COSHARE} and every used group's green SM within {SM_TOL} of (D, {TOTAL_SM}-D)"}
