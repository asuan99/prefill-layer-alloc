#!/usr/bin/env python3
"""lambda* sustainability block (decision 56; DESCRIPTIVE ONLY, no label, no verdict, never an input to CAP2).

CAP1-R fixed lambda*_W1 = 7.29491 req/s from 120-request (~16 s) cells.  This block asks, in the same Job/node as CAP2 and AFTER it,
what happens at that rate for 600 s: static D44 and D64 (non-sticky, decision 49) x seed {201, 202}, n = round(7.29491 x 600) = 4377
requests at the fixed rate.  Reported per run and per 60 s ARRIVAL window: attainment (good/arrived), TTFT p50/p95, plus the
request-level S1/S2 stationarity numbers and the SW-power-cap throttle fraction of the same window (clk.csv of this Job).
PHASE/diag instrumentation is OFF (a normal measurement boot).  Output goes to a directory of its own (the caller passes
${summary}/sustain); CAP2 reads only its own raw tree, and ph_analyze.cap_addendum refuses to mix Jobs (run ids start with SUSTAIN).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(os.path.abspath(__file__)).parent
sys.path.insert(0, str(HERE))
import ph_spec as P  # noqa: E402
import ph_verdict as V  # noqa: E402

SUSTAIN_D = (44, 64)
SUSTAIN_SEEDS = (201, 202)
DURATION_S = 600
WINDOW_S = 60.0
LAMBDA_STAR_W1 = 7.29491          # CAP1-R (Job pdmux-policy-cap1r-4bef3cf-20261029003539), input value registered in PREREG §20


def n_requests(rate, duration_s=DURATION_S):
    return int(round(rate * duration_s))


def window_table(requests, ttft_slo_ms=P.W1_TTFT_MS, itl_slo_ms=P.ITL_MS, window_s=WINDOW_S):
    """Per ARRIVAL window of one run: n, good, attainment, TTFT p50/p95 (descriptive)."""
    by = {}
    for r in requests:
        by.setdefault(int(r["scheduled_arrival_s"] // window_s), []).append(r)
    out = []
    for w in sorted(by):
        rs = by[w]
        tt = [x["ttft_ms"] for x in rs if x.get("ttft_ms") is not None]
        good = sum(1 for x in rs if V.req_good(x, ttft_slo_ms, itl_slo_ms))
        out.append({"window": w, "t0_s": w * window_s, "n": len(rs), "good": good, "attainment": good / len(rs),
                    "ttft_p50_ms": V.percentile(tt, 0.5), "ttft_p95_ms": V.percentile(tt, 0.95)})
    return out


def _parse_clk_ts(ts):
    return dt.datetime.strptime(ts.strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp()


def throttle_by_window(clk_text, t0_epoch, n_windows, window_s=WINDOW_S):
    """Fraction of clk.csv samples per window of the run with any throttle reason other than GpuIdle (bit 0x1), as job_entry counts."""
    cnt = [[0, 0] for _ in range(n_windows)]
    for ln in clk_text.splitlines():
        p = [c.strip() for c in ln.split(",")]
        if len(p) < 5 or not p[0][:2] == "20":
            continue
        try:
            t = _parse_clk_ts(p[0])
            active = (int(p[4], 16) & ~0x1) != 0
        except ValueError:
            continue
        w = int((t - t0_epoch) // window_s)
        if 0 <= w < n_windows:
            cnt[w][0] += 1
            cnt[w][1] += int(active)
    return [(c[1] / c[0]) if c[0] else None for c in cnt]


def summarize_run(requests, clk_text, t0_epoch, ttft_slo_ms=P.W1_TTFT_MS, itl_slo_ms=P.ITL_MS):
    wt = window_table(requests, ttft_slo_ms, itl_slo_ms)
    thr = throttle_by_window(clk_text or "", t0_epoch, (wt[-1]["window"] + 1) if wt else 0)
    for row in wt:
        row["throttle_frac"] = thr[row["window"]] if row["window"] < len(thr) else None
    st = V.stationarity(requests, ttft_slo_ms, itl_slo_ms)
    return {"windows": wt, "S1": st["S1"], "S2": st["S2"], "descriptive_only": True}


def plan(seed_order_key="sustain"):
    import random
    rng = random.Random(int.from_bytes(seed_order_key.encode(), "big") % (2 ** 32))
    ds = list(SUSTAIN_D)
    rng.shuffle(ds)
    out = []
    for i, D in enumerate(ds):
        seeds = list(SUSTAIN_SEEDS)
        rng.shuffle(seeds)
        out.append({"boot_id": f"SUSTAIN_b{i}_s{D}", "arm": P.static_arm(D),
                    "runs": [{"run_id": f"SUSTAIN_D{D}_s{s}", "seed": s, "rate": LAMBDA_STAR_W1, "n_requests": n_requests(LAMBDA_STAR_W1)}
                             for s in seeds]})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--w1-lengths", required=True)
    ap.add_argument("--clk", default="")
    a = ap.parse_args(argv)
    import ph_runner as RU
    import ph_trace as T
    R = RU.R
    w1 = json.load(open(a.w1_lengths))
    out = Path(a.out)
    (out / "runs").mkdir(parents=True, exist_ok=True)
    rc = 0
    for boot in plan():
        bdir = out / "boots" / boot["boot_id"]
        bdir.mkdir(parents=True, exist_ok=True)
        env = RU.boot_env(boot["arm"], bdir, run_id=os.environ.get("PDMUX_JOB_NAME", "sustain"))
        srv = R.ProcessServer(bdir, RU.server_argv(boot["arm"]), env)
        if srv.start() != "ok":
            srv.stop()
            rc = 17
            continue
        try:
            endpoint = f"http://127.0.0.1:{srv.port}/generate"
            for run in boot["runs"]:
                segs = T.cap_trace("W1", w1, run["rate"], run["seed"], run["n_requests"], run["run_id"])
                t0 = time.time()
                seg = RU.replay_segment(RU.default_replay, endpoint, segs[0], out / "seg")
                t1 = time.time()
                clk = Path(a.clk).read_text(errors="replace") if a.clk and Path(a.clk).exists() else ""
                rec = {"run_id": run["run_id"], "arm": boot["arm"], "seed": run["seed"], "rate": run["rate"], "n_requests": run["n_requests"],
                       "wall_start_epoch": t0, "wall_end_epoch": t1, "duration_s": seg["duration_s"],
                       "summary": summarize_run(seg["requests"], clk, t0), "NOTE": "SUSTAIN_DESCRIPTIVE: no label, no verdict"}
                json.dump(rec, open(out / "runs" / f"{run['run_id']}.json", "w"), indent=1, sort_keys=True)
        finally:
            srv.stop()
    json.dump({"status": "done" if rc == 0 else "BOOT_FAILURES", "rc": rc, "lambda_star_W1": LAMBDA_STAR_W1}, open(out / "SUSTAIN_DONE.json", "w"))
    print("ph_sustain: done rc", rc)
    return rc


if __name__ == "__main__":
    sys.exit(main())
