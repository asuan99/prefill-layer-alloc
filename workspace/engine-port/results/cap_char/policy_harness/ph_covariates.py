#!/usr/bin/env python3
"""Substrate covariates for TUNE / HE0 Jobs (decision 57, CAP2 audit r1 issue 4): DESCRIPTIVE ONLY, no label, no verdict, never an
input to he0_compare / select_tune / any decision.  Different Jobs run on different GPUs (CAP1 / CAP1-R / CAP2 UUIDs all differ) and
the SW power cap is a load-induced limit, so Job-to-Job throttle differences are recorded, never interpreted as drift.

Recorded: Job level -- GPU UUID(s), driver, hostname (meta/substrate.json); per run and per segment (LO/HI) -- the throttle fraction
(any reason except GpuIdle), power median/p95, temperature median, SM clock median over the segment's wall window (clk.csv of this Job),
the completion rate (completed requests / segment duration) and the drain time (duration - last scheduled arrival).
Block rule for the analysis: the Job is the BLOCK (HE0 n=4 = 2 Jobs x 2 seeds; seeds nest inside Jobs) -- see ph_analyze.job_blocks.
"""
from __future__ import annotations

import datetime as dt
import json
import statistics
import sys
from pathlib import Path

import ph_verdict as V


def parse_clk(text):
    """clk.csv rows -> [(epoch, sm_mhz, power_w, temp_c, throttled_bool)]; header / malformed lines skipped."""
    rows = []
    for ln in text.splitlines():
        p = [c.strip() for c in ln.split(",")]
        if len(p) < 5 or not p[0].startswith("20"):
            continue
        try:
            t = dt.datetime.strptime(p[0], "%Y/%m/%d %H:%M:%S.%f").timestamp()
            sm = float(p[1].split()[0])
            w = float(p[2].split()[0])
            temp = float(p[3])
            thr = (int(p[4], 16) & ~0x1) != 0
        except (ValueError, IndexError):
            continue
        rows.append((t, sm, w, temp, thr))
    return rows


def window_stats(rows, t0, t1):
    sel = [r for r in rows if t0 <= r[0] < t1]
    if not sel:
        return {"n_samples": 0}
    pw = [r[2] for r in sel]
    return {"n_samples": len(sel), "throttle_frac": sum(1 for r in sel if r[4]) / len(sel),
            "power_w_median": statistics.median(pw), "power_w_p95": V.percentile(pw, 0.95),
            "temp_c_median": statistics.median(r[3] for r in sel), "sm_mhz_median": statistics.median(r[1] for r in sel)}


def segment_stats(seg):
    """completion rate and drain time of one replayed segment (post-hoc, from its records)."""
    reqs = seg["requests"]
    ok = [r for r in reqs if not r.get("error") and r.get("ttft_ms") is not None]
    dur = float(seg["duration_s"])
    last_arrival = max((r.get("scheduled_arrival_s", 0.0) for r in reqs), default=0.0)
    return {"phase": seg["phase"], "round": seg.get("round"), "n": len(reqs), "n_completed": len(ok), "duration_s": dur,
            "completion_rate_req_s": (len(ok) / dur) if dur > 0 else None, "drain_s": max(0.0, dur - last_arrival)}


def run_covariates(run, rows):
    segs = []
    for s in run["segments"]:
        d = segment_stats(s)
        if "wall_start_epoch" in s:
            d.update(window_stats(rows, s["wall_start_epoch"], s["wall_end_epoch"]))
        segs.append(d)
    ws = [s["wall_start_epoch"] for s in run["segments"] if "wall_start_epoch" in s]
    we = [s["wall_end_epoch"] for s in run["segments"] if "wall_end_epoch" in s]
    return {"run_id": run["run_id"], "seed": run["seed"], "arm": run["arm"].get("name"), "segments": segs,
            "run_window": window_stats(rows, min(ws), max(we)) if ws else {"n_samples": 0}}


def job_covariates(substrate):
    g = substrate.get("gpu") or {}
    return {"gpu_uuids": substrate.get("gpu_uuids") or [g.get("uuid")], "gpu_name": g.get("name"),
            "driver_version": g.get("driver_version"), "hostname": substrate.get("hostname"), "job_name": substrate.get("job_name")}


def cov_report(runs, clk_text, substrate):
    rows = parse_clk(clk_text or "")
    return {"DESCRIPTIVE_ONLY": "no label, no verdict; not an input to any decision", "job": job_covariates(substrate or {}),
            "runs": [run_covariates(r, rows) for r in runs]}


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--clk", default="")
    ap.add_argument("--substrate", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    runs = [json.loads(p.read_text()) for p in sorted((Path(a.raw) / "runs").glob("*.json"))]
    clk = Path(a.clk).read_text(errors="replace") if a.clk and Path(a.clk).exists() else ""
    sub = json.loads(Path(a.substrate).read_text()) if a.substrate and Path(a.substrate).exists() else {}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(cov_report(runs, clk, sub), indent=1, sort_keys=True))
    print("ph_covariates:", len(runs), "runs")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.exit(main())
