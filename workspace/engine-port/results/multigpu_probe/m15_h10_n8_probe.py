#!/usr/bin/env python3
"""H10 -- client capacity at the N=8 shape-A HI scale (PREREG_MULTIGPU rev6 harness condition H10; MGP-34 companion).

The rev5 delta audit forecast the peak simultaneous connections of the HI rung: N=4 shape A ≈ 1,410, N=8 ≈ 1,880
(extension ×2 ≈ 3,130).  M1.5 (m15_client_capacity_probe.py) only reached ~100 rps with a FIXED-latency dummy, i.e. a few
hundred open streams.  This probe emulates the real shape of an overloaded engine instead:

  * a dummy SSE backend with `slots` = 48 x N running slots (the registered cap).  A request that finds no free slot
    WAITS with no bytes sent (like an SGLang queue), then streams `output_tokens` cumulative-text SSE chunks at
    `itl_ms` per token after a `ttft_base_ms` prefill -- so the number of open connections grows exactly as in a
    saturated cell (queue ~ (x - lambda) t) and ~slots streams carry token events.
  * the REAL pdmux_eval.trace_loadgen.replay (via mg_loadgen: RLIMIT_NOFILE raise, connector limit 0, sock_read 600),
    unmodified, is the client.
  * measured: send lag (sent_s - scheduled), asyncio heartbeat overshoot, client CPU seconds / wall, peak open
    connections (client side, from the request intervals), token-ITL inflation over the dummy's own spacing.

LOCAL UPPER BOUND ONLY: on VESSL the client shares an ~11-CPU cgroup quota with N schedulers, the TokenizerManager and the
detokenizer (lesson 268), so a PASS here is necessary, not sufficient.  The thresholds are the M1.5 CANDIDATE ones
(sched lag p99 <= 5 ms = the registered HARNESS_LIMITED lag; ITL p95 inflation <= 2 ms), not new registrations.
"""

import argparse
import asyncio
import json
import math
import multiprocessing as mp
import os
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENGINE = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ENGINE / "benchmarks"))
sys.path.insert(0, str(ENGINE / "scripts" / "multigpu"))
sys.path.insert(0, str(ENGINE / "results" / "multigpu"))
import m15_client_capacity_probe as M15  # noqa: E402
import mg_const as C  # noqa: E402
import mg_loadgen  # noqa: E402


def _worker(port, slots, ttft_base_s, itl_s):
    from aiohttp import web
    meta_fields = {"id": "x" * 32, "finish_reason": None, "prompt_tokens": 256, "weight_version": "default",
                   "cached_tokens": 0, "e2e_latency": 0.0, "spec_verify_ct": 0, "queue_time": 0.0,
                   "forward_entry_time": 0.0}
    sem = asyncio.Semaphore(slots)

    async def generate(request):
        body = await request.json()
        n = int(body["sampling_params"]["max_new_tokens"])
        async with sem:                       # queue: connection open, nothing sent
            resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await resp.prepare(request)
            await asyncio.sleep(ttft_base_s)
            for k in range(1, n + 1):
                chunk = {"text": "tok " * k, "meta_info": dict(meta_fields, completion_tokens=k)}
                await resp.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
                if k < n:
                    await asyncio.sleep(itl_s)
            await resp.write(b"data: [DONE]\n\n")
            await resp.write_eof()
        return resp

    app = web.Application()
    app.router.add_post("/generate", generate)
    web.run_app(app, host="127.0.0.1", port=port, reuse_port=True, print=None, access_log=None, handle_signals=True)


def peak_open(recs):
    ev = []
    for r in recs:
        ev.append((r["sent_s"], 1))
        ev.append((r["completion_s"], -1))
    cur = peak = 0
    for _, d in sorted(ev):
        cur += d
        peak = max(peak, cur)
    return peak


def run_point(args, n_gpus, rate_mult, ext):
    port = 18000 + n_gpus * 10 + ext
    slots_total = C.CAP_PER_GPU * n_gpus
    workers = args.server_workers
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_worker, args=(port, slots_total // workers, args.ttft_base_ms / 1000.0, args.itl_ms / 1000.0))
             for _ in range(workers)]
    for p in procs:
        p.start()
    import urllib.error
    import urllib.request
    for _ in range(100):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/nothing", timeout=0.5)
        except urllib.error.HTTPError:
            break
        except Exception:  # noqa: BLE001
            time.sleep(0.1)
    try:
        x = C.HI_MULT * n_gpus * C.PRIOR_PER_GPU["A"] * (C.EXT_MULT ** ext)
        n_req = max(40, min(4000, int(round(x * C.WINDOW_S / 10.0) * 10)))
        items = M15.make_trace(x, n_req / x, 256, C.SHAPE_IO["A"][1], 7 + n_gpus, f"h10n{n_gpus}e{ext}")
        out = Path(tempfile.mkdtemp(prefix="h10_")) / "requests.jsonl"
        fd = mg_loadgen.raise_fd_limit()
        t0, cpu, lags = asyncio.run(M15.run_point(f"http://127.0.0.1:{port}/generate", items, out, C.CONNECTOR_LIMIT))
        recs = [json.loads(ln) for ln in out.read_text().splitlines() if ln.strip()]
        ok = [r for r in recs if r["ttft_ms"] is not None and not r["error"]]
        sched = [(r["sent_s"] - r["scheduled_arrival_s"]) * 1000.0 for r in recs]
        itls = [v for r in ok for v in r["token_itl_ms"]]
        return {"n_gpus": n_gpus, "ext": ext, "offered_req_s": round(x, 3), "n_req": len(recs), "slots": slots_total,
                "peak_open_connections_client": peak_open(recs), "errors": len(recs) - len(ok),
                "sched_lag_ms": {"p50": M15.pct(sched, .5), "p99": M15.pct(sched, .99), "max": max(sched)},
                "loop_lag_ms": {"p50": M15.pct(lags, .5), "p99": M15.pct(lags, .99), "max": max(lags) if lags else math.nan},
                "client_cpu_util": cpu,
                "itl_ms": {"p50": M15.pct(itls, .5), "p95": M15.pct(itls, .95), "p99": M15.pct(itls, .99)},
                "itl_p95_inflation_ms": M15.pct(itls, .95) - args.itl_ms,
                "fd_limit": fd, "send_window_s": max(r["sent_s"] for r in recs) - min(r["sent_s"] for r in recs)}
    finally:
        import signal
        for p in procs:
            if p.pid:
                os.kill(p.pid, signal.SIGTERM)
        for p in procs:
            p.join(timeout=20)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--points", default="4:0,8:0,8:1", help="n_gpus:ext list (ext=1 = x2 extension rung)")
    ap.add_argument("--ttft-base-ms", type=float, default=200.0)
    ap.add_argument("--itl-ms", type=float, default=32.0)
    ap.add_argument("--server-workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    pts = []
    for tok in a.points.split(","):
        n, e = (int(v) for v in tok.split(":"))
        r = run_point(a, n, 1, e)
        r["PASS_candidate"] = bool(r["errors"] == 0 and r["sched_lag_ms"]["p99"] <= 5.0 and r["itl_p95_inflation_ms"] <= 2.0)
        print(json.dumps(r, sort_keys=True), flush=True)
        pts.append(r)
    Path(a.out).write_text(json.dumps({"params": vars(a), "host_cpus": os.cpu_count(), "points": pts,
                                       "note": "local upper bound; VESSL client shares an ~11-CPU quota"}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
