#!/usr/bin/env python3
"""Server-less calibration of the I-4 CUDA-event bracket on a green-context stream (audit r1 MAJOR 1(2), 2026-10-20).

Question: does a [start.record ... end.record] bracket on a green-context stream measure what the I-4 probe claims
(GPU interval of the forward, INCLUDING host gaps inside the bracket, EXCLUDING host work after the end event), and does
the green stream differ from the default stream?  The CPU tests cannot show this; this unit is its only check.

2 x 2 design (REGISTERED BEFORE ANY DATA; do not edit the constants after a GPU run -- a FAIL goes to a USER DECISION,
the tolerance is never adjusted to the data):
  order  pre  : start.record(); host sleep SLEEP_MS; matmul;            end.record()      -> sleep is INSIDE the bracket
         post : start.record(); matmul;            end.record(); host sleep SLEEP_MS     -> sleep is AFTER the bracket
  stream green (sgl_kernel.spatial green-context stream, GREEN_SM SMs) | default (torch default stream)
Each cell: REPS repetitions, median.  Reference per stream: ref_ms = median host wall time of [matmul; stream.synchronize()]
(launch latency included), same stream, same shapes.

Registered predictions (all must hold for PASS):
  P1 pre  event ms - post event ms >= PRE_INCLUDES_FRAC * SLEEP_MS     (host gap inside the bracket is included)
  P2 post event ms <= POST_EXCLUDES_FACTOR * ref_ms                    (host work after the end event is not included)
  P3 |post event ms / ref_ms - 1| <= CALIB_REL_TOL on BOTH streams     (event timing agrees with the host-synchronised reference)
  P4 |post_green / ref_green - post_default / ref_default| <= CALIB_REL_TOL   (green vs default: no systematic difference)
If the green stream cannot be created the cell is UNRESOLVED (never PASS).

usage: green_event_calib_unit.py --out calib.json [--green-sm 74] [--device cuda]
       green_event_calib_unit.py --score-json cells.json     (pure scorer on supplied numbers; CPU test)
Exit 0 PASS, 1 FAIL, 4 UNRESOLVED.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

SLEEP_MS = 5.0
REPS = 30
PRE_INCLUDES_FRAC = 0.8
POST_EXCLUDES_FACTOR = 1.2
CALIB_REL_TOL = 0.02          # the coordinator's "<= 2 %"
MATMUL_N = 8192
GREEN_SM_DEFAULT = 74
CONSTANTS = {"SLEEP_MS": SLEEP_MS, "REPS": REPS, "PRE_INCLUDES_FRAC": PRE_INCLUDES_FRAC,
             "POST_EXCLUDES_FACTOR": POST_EXCLUDES_FACTOR, "CALIB_REL_TOL": CALIB_REL_TOL,
             "MATMUL_N": MATMUL_N}


def score(cells):
    """cells = {"green": {"pre": ms, "post": ms, "ref": ms}, "default": {...}} (medians).  A missing stream -> UNRESOLVED."""
    why = []
    for s in ("green", "default"):
        c = cells.get(s)
        if not c or any(c.get(k) is None for k in ("pre", "post", "ref")):
            return {"verdict": "UNRESOLVED", "why": [f"{s} stream cell missing"], "constants": CONSTANTS}
    for s in ("green", "default"):
        c = cells[s]
        if c["pre"] - c["post"] < PRE_INCLUDES_FRAC * SLEEP_MS:
            why.append(f"P1 {s}: pre-post {c['pre'] - c['post']:.3f} ms < {PRE_INCLUDES_FRAC * SLEEP_MS} ms")
        if c["post"] > POST_EXCLUDES_FACTOR * c["ref"]:
            why.append(f"P2 {s}: post {c['post']:.3f} ms > {POST_EXCLUDES_FACTOR} x ref {c['ref']:.3f}")
        if abs(c["post"] / c["ref"] - 1.0) > CALIB_REL_TOL:
            why.append(f"P3 {s}: |post/ref-1| = {abs(c['post'] / c['ref'] - 1.0):.4f} > {CALIB_REL_TOL}")
    g, d = cells["green"], cells["default"]
    if abs(g["post"] / g["ref"] - d["post"] / d["ref"]) > CALIB_REL_TOL:
        why.append("P4 green vs default calibration differs by more than the tolerance")
    return {"verdict": "FAIL" if why else "PASS", "why": why, "cells": cells, "constants": CONSTANTS,
            "on_fail": "USER DECISION (the tolerance is not adjusted after seeing data)"}


def _measure(torch, stream, device):
    a = torch.randn(MATMUL_N, MATMUL_N, device=device, dtype=torch.bfloat16)
    b = torch.randn(MATMUL_N, MATMUL_N, device=device, dtype=torch.bfloat16)
    out = {"pre": [], "post": [], "ref": []}
    with torch.cuda.stream(stream):
        for _ in range(5):
            torch.matmul(a, b)
        stream.synchronize()
        for _ in range(REPS):
            s0, e0 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s0.record()
            time.sleep(SLEEP_MS / 1000.0)
            torch.matmul(a, b)
            e0.record()
            stream.synchronize()
            out["pre"].append(s0.elapsed_time(e0))
            s1, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s1.record()
            torch.matmul(a, b)
            e1.record()
            time.sleep(SLEEP_MS / 1000.0)
            stream.synchronize()
            out["post"].append(s1.elapsed_time(e1))
            t = time.perf_counter()
            torch.matmul(a, b)
            stream.synchronize()
            out["ref"].append((time.perf_counter() - t) * 1000.0)
    return {k: statistics.median(v) for k, v in out.items()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out")
    ap.add_argument("--green-sm", type=int, default=GREEN_SM_DEFAULT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--score-json")
    a = ap.parse_args(argv)
    if a.score_json:
        res = score(json.loads(Path(a.score_json).read_text()))
    else:
        import torch
        cells = {}
        cells["default"] = _measure(torch, torch.cuda.current_stream(), a.device)
        try:
            from sgl_kernel.spatial import create_greenctx_stream_by_value
            total = torch.cuda.get_device_properties(0).multi_processor_count
            green, _rest = create_greenctx_stream_by_value(a.green_sm, total - a.green_sm, 0)
            cells["green"] = _measure(torch, green, a.device)
        except Exception as exc:  # noqa: BLE001
            cells["green"] = None
            cells["green_error"] = repr(exc)[:300]
        res = score({k: v for k, v in cells.items() if k in ("green", "default")})
        res["green_error"] = cells.get("green_error")
        res["green_sm"] = a.green_sm
    text = json.dumps(res, indent=1, sort_keys=True)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n")
    return {"PASS": 0, "FAIL": 1}.get(res["verdict"], 4)


if __name__ == "__main__":
    sys.exit(main())
