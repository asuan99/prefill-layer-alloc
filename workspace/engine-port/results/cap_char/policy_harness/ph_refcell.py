#!/usr/bin/env python3
"""Job-start / Job-end reference cell (decision 57; DESCRIPTIVE ONLY, no label, no verdict, never a decision input).

Same cell every time so Job-to-Job and start-to-end substrate differences can be READ next to the TUNE / HE0 data:
W1, static D44 (non-sticky), 7.29491 req/s (CAP1-R lambda*), 120 requests, seed 201 -- a ~17 s load after the standard boot + 60 s warm-up.
Writes <out>/REFCELL_<tag>.json with attainment, TTFT p50/p95, the wall window and the clk.csv statistics of that window.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(os.path.abspath(__file__)).parent
sys.path.insert(0, str(HERE))
import ph_spec as P  # noqa: E402
import ph_verdict as V  # noqa: E402

REF_D, REF_RATE, REF_N, REF_SEED = 44, 7.29491, 120, 201


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--w1-lengths", required=True)
    ap.add_argument("--tag", required=True, choices=["start", "end"])
    ap.add_argument("--clk", default="")
    a = ap.parse_args(argv)
    import ph_covariates as C
    import ph_runner as RU
    import ph_trace as T
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    w1 = json.load(open(a.w1_lengths))
    arm = P.static_arm(REF_D)
    bdir = out / f"boot_{a.tag}"
    bdir.mkdir(exist_ok=True)
    env = RU.boot_env(arm, bdir, run_id=os.environ.get("PDMUX_JOB_NAME", "refcell"))
    srv = RU.R.ProcessServer(bdir, RU.server_argv(arm), env)
    if srv.start() != "ok":
        srv.stop()
        json.dump({"tag": a.tag, "status": "BOOT_FAILED"}, open(out / f"REFCELL_{a.tag}.json", "w"))
        return 17
    try:
        endpoint = f"http://127.0.0.1:{srv.port}/generate"
        ok, why = RU.warmup_and_check(endpoint, RU.default_replay, w1, bdir, f"ref_{a.tag}")
        seg = T.cap_trace("W1", w1, REF_RATE, REF_SEED, REF_N, f"REF_{a.tag}")[0]
        t0 = time.time()
        res = RU.replay_segment(RU.default_replay, endpoint, seg, bdir / "seg")
        t1 = time.time()
    finally:
        srv.stop()
    good = sum(1 for r in res["requests"] if V.req_good(r, P.W1_TTFT_MS, P.ITL_MS))
    tt = [r["ttft_ms"] for r in res["requests"] if r.get("ttft_ms") is not None]
    clk = Path(a.clk).read_text(errors="replace") if a.clk and Path(a.clk).exists() else ""
    rec = {"tag": a.tag, "status": "done", "warmup_correct": ok, "D": REF_D, "rate": REF_RATE, "n": REF_N, "seed": REF_SEED,
           "attainment": good / len(res["requests"]), "ttft_p50_ms": V.percentile(tt, 0.5), "ttft_p95_ms": V.percentile(tt, 0.95),
           "duration_s": res["duration_s"], "wall_start_epoch": t0, "wall_end_epoch": t1,
           "substrate_window": C.window_stats(C.parse_clk(clk), t0, t1), "NOTE": "REFCELL_DESCRIPTIVE: no label, no verdict"}
    json.dump(rec, open(out / f"REFCELL_{a.tag}.json", "w"), indent=1, sort_keys=True)
    print("ph_refcell:", a.tag, "attainment", rec["attainment"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
