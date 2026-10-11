#!/usr/bin/env python3
"""M1.5 harness capacity probe (multi-GPU track; design reports/multigpu_scaling_design_2026-10-12.md
section 6 M1.5, audit C-4): can the trace-replay CLIENT sustain >= 1.5x the largest registered
offered rate without distorting what it measures?

WHAT IS MEASURED (client side only; no GPU, no model):
  Mode `dummy` (default): the REAL `pdmux_eval.trace_loadgen.replay` is run, unmodified, against a
  local SGLang `/generate` SSE look-alike (a pool of aiohttp processes that emit
  `completion_tokens`-bearing SSE chunks at a fixed TTFT and per-token ITL).  Because the dummy's
  latencies are known, every excess seen by the client is client/harness overhead:
    sched_lag   sent_s - scheduled arrival (client event-loop issue lag; exact)
    wire_lag    dummy-server receive time - scheduled arrival (includes connector queueing; the
                client's own `sent_s` is taken BEFORE session.post, so connector waits are invisible
                there).  Offset-calibrated on the lowest-rate run.
    ttft_excess client TTFT minus the baseline-run median TTFT
    itl_excess  client token-ITL p95 minus the baseline-run p95 (ITL tail = SLO leg)
    loop_lag    asyncio heartbeat overshoot inside the client loop (saturation indicator)
    cpu_util    client process CPU seconds / wall seconds (>= ~0.9 means one core is the bottleneck)
  `--connector-limit N` is trace_loadgen's `connector_limit` (0 = unlimited); the default None is the
  stock aiohttp connector (limit=100), which is what every earlier campaign used.

  Mode `endpoint` (GPU SKELETON -- NOT VALIDATED, needs a real server): replays 1-output-token,
  short-prompt requests against --endpoint so the prefill/decode work is negligible; the TTFT excess
  over the lowest rate then isolates the TokenizerManager/frontend queue (single process, GIL).  The
  same sched_lag / loop_lag / cpu_util columns apply.  Run it inside a short Job per the M1.5 row.

THRESHOLDS: the PASS flags below use CANDIDATE thresholds (sched/wire/ttft p99 <= 5 ms, ITL p95
excess <= 2 ms, no errors).  They are not registered; the PREREG owns the real numbers.

CAVEAT: local results are an UPPER BOUND for the client.  On VESSL the same cores (cgroup quota ~11
CPUs, lesson 268) are shared with N schedulers, the TokenizerManager and the detokenizer.
"""

import argparse
import asyncio
import json
import math
import multiprocessing as mp
import os
import random
import resource
import statistics
import sys
import tempfile
import time
from pathlib import Path

ENGINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ENGINE / "benchmarks"))

CAND = {"lag_p99_ms": 5.0, "ttft_excess_p99_ms": 5.0, "itl_excess_p95_ms": 2.0}


# --------------------------------------------------------------------------- dummy backend
def _server_worker(port, ttft_s, itl_s, out_path):
    from aiohttp import web

    recv = {}
    meta_fields = {"id": "x" * 32, "finish_reason": None, "prompt_tokens": 1000,
                   "weight_version": "default", "cached_tokens": 0, "e2e_latency": 0.0,
                   "spec_verify_ct": 0, "queue_time": 0.0, "forward_entry_time": 0.0}

    async def generate(request):
        t_recv = time.perf_counter()
        body = await request.json()
        rid = body.get("rid", "?")
        recv[rid] = t_recv
        n = int(body["sampling_params"]["max_new_tokens"])
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        await asyncio.sleep(ttft_s)
        for k in range(1, n + 1):
            # SGLang streams the cumulative text each step; payload grows with k
            chunk = {"text": "tok " * k, "meta_info": dict(meta_fields, completion_tokens=k)}
            await resp.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
            if k < n:
                await asyncio.sleep(itl_s)
        await resp.write(b"data: [DONE]\n\n")
        await resp.write_eof()
        return resp

    async def dump(_app):
        Path(out_path).write_text(json.dumps(recv))

    app = web.Application()
    app.router.add_post("/generate", generate)
    app.on_cleanup.append(dump)
    web.run_app(app, host="127.0.0.1", port=port, reuse_port=True, print=None,
                access_log=None, handle_signals=True)


class DummyBackend:
    def __init__(self, port, workers, ttft_s, itl_s, tmp):
        self.port, self.tmp = port, Path(tmp)
        ctx = mp.get_context("spawn")
        self.procs = [ctx.Process(target=_server_worker,
                                  args=(port, ttft_s, itl_s, str(self.tmp / f"recv{i}.json")))
                      for i in range(workers)]

    def start(self):
        for p in self.procs:
            p.start()
        import urllib.request
        for _ in range(100):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{self.port}/nothing", timeout=0.5)
            except urllib.error.HTTPError:
                return
            except Exception:  # noqa: BLE001
                time.sleep(0.1)
        raise RuntimeError("dummy backend did not come up")

    def stop(self):
        import signal
        for p in self.procs:
            if p.pid:
                os.kill(p.pid, signal.SIGTERM)
        for p in self.procs:
            p.join(timeout=20)
        recv = {}
        for i in range(len(self.procs)):
            f = self.tmp / f"recv{i}.json"
            if f.exists():
                recv.update(json.loads(f.read_text()))
        return recv


# --------------------------------------------------------------------------- trace + stats
def make_trace(rate, duration_s, input_tokens, output_tokens, seed, tag):
    """Poisson arrivals at `rate` req/s for `duration_s`."""
    rng = random.Random(seed)
    t, items, i = 0.0, [], 0
    while True:
        t += rng.expovariate(rate)
        if t >= duration_s:
            return items
        items.append({"record": "request", "request_id": f"{tag}-{i:06d}", "phase": "p",
                      "arrival_s": t, "input_tokens": input_tokens,
                      "output_tokens": output_tokens})
        i += 1


def pct(values, q):
    if not values:
        return math.nan
    v = sorted(values)
    pos = (len(v) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


async def heartbeat(period_s, lags):
    while True:
        t = time.perf_counter()
        await asyncio.sleep(period_s)
        lags.append((time.perf_counter() - t - period_s) * 1000.0)


async def run_point(endpoint, items, out_path, connector_limit):
    import aiohttp  # noqa: F401  -- imported BEFORE t0: replay() imports it lazily, which would
    from pdmux_eval import trace_loadgen as TL  # shift the first run's t0 offset by ~60 ms

    lags = []
    hb = asyncio.create_task(heartbeat(0.005, lags))
    cpu0, w0 = time.process_time(), time.perf_counter()
    t0 = time.perf_counter()
    try:
        await TL.replay(endpoint, items, out_path, 1e12, 1e12, connector_limit)
    finally:
        hb.cancel()
    wall = time.perf_counter() - w0
    return t0, (time.process_time() - cpu0) / wall, lags


def point_stats(rate, items, out_path, t0, cpu_util, loop_lags, recv, args, in_flight_nominal):
    recs = [json.loads(x) for x in Path(out_path).read_text().splitlines() if x.strip()]
    ok = [r for r in recs if r["ttft_ms"] is not None and not r["error"]]
    sched = [(r["sent_s"] - r["scheduled_arrival_s"]) * 1000.0 for r in recs]
    wire = []
    for r in recs:
        if r["request_id"] in recv:
            wire.append((recv[r["request_id"]] - (t0 + r["scheduled_arrival_s"])) * 1000.0)
    itls = [v for r in ok for v in r["token_itl_ms"]]
    return {
        "rate_req_s": rate, "requests": len(recs), "errors": len(recs) - len(ok),
        "nominal_in_flight_streams": in_flight_nominal,
        "nominal_token_events_per_s": rate * args.output_tokens,
        "sched_lag_ms": {"p50": pct(sched, .5), "p99": pct(sched, .99), "max": max(sched)},
        "wire_raw_ms": {"p50": pct(wire, .5), "p99": pct(wire, .99)},
        "ttft_ms": {"p50": pct([r["ttft_ms"] for r in ok], .5),
                    "p99": pct([r["ttft_ms"] for r in ok], .99)},
        "itl_ms": {"p50": pct(itls, .5), "p95": pct(itls, .95), "p99": pct(itls, .99)},
        "loop_lag_ms": {"p50": pct(loop_lags, .5), "p99": pct(loop_lags, .99),
                        "max": max(loop_lags) if loop_lags else math.nan},
        "cpu_util": cpu_util,
    }


def judge(points):
    base = points[0]
    wire_off = base["wire_raw_ms"]["p50"]
    for p in points:
        p["wire_lag_p99_ms"] = (p["wire_raw_ms"]["p99"] - wire_off
                                if not math.isnan(p["wire_raw_ms"]["p99"]) else math.nan)
        p["ttft_excess_p99_ms"] = p["ttft_ms"]["p99"] - base["ttft_ms"]["p50"]
        p["itl_excess_p95_ms"] = p["itl_ms"]["p95"] - base["itl_ms"]["p95"]
        lag = max(p["sched_lag_ms"]["p99"],
                  p["wire_lag_p99_ms"] if not math.isnan(p["wire_lag_p99_ms"]) else 0.0)
        p["PASS_candidate"] = bool(
            p["errors"] == 0 and lag <= CAND["lag_p99_ms"]
            and p["ttft_excess_p99_ms"] <= CAND["ttft_excess_p99_ms"]
            and p["itl_excess_p95_ms"] <= CAND["itl_excess_p95_ms"])
    return points


def sweep(args, endpoint):
    """Run every rate once; return raw per-rate records (stats are computed after the backend
    has dumped its receive times)."""
    raw = []
    for k, rate in enumerate(float(x) for x in args.rates.split(",")):
        tag = f"r{int(rate)}"
        items = make_trace(rate, args.duration_s, args.input_tokens, args.output_tokens,
                           args.seed + k, tag)
        out = Path(args.workdir) / f"requests_{tag}.jsonl"
        t0, cpu, lags = asyncio.run(run_point(endpoint, items, out, args.connector_limit))
        raw.append({"rate": rate, "items": items, "out": out, "t0": t0, "cpu": cpu, "lags": lags})
        recs = [json.loads(x) for x in out.read_text().splitlines() if x.strip()]
        sched = [(r["sent_s"] - r["scheduled_arrival_s"]) * 1000.0 for r in recs]
        print(f"[m15] rate {rate:g}: n={len(recs)} sched_lag_p99={pct(sched, .99):.2f}ms "
              f"cpu={cpu:.2f} loop_lag_p99={pct(lags, .99):.2f}ms", flush=True)
    return raw


def finalize(raw, recv, args):
    points = [point_stats(r["rate"], r["items"], r["out"], r["t0"], r["cpu"], r["lags"], recv, args,
                          r["rate"] * (args.ttft_ms + args.itl_ms * (args.output_tokens - 1)) / 1000)
              for r in raw]
    return judge(points)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=("dummy", "endpoint"), default="dummy")
    ap.add_argument("--endpoint", default="", help="endpoint mode: http://host:port/generate")
    ap.add_argument("--rates", default="2,8,16,32,48,64,96,128,192")
    ap.add_argument("--duration-s", type=float, default=15.0)
    ap.add_argument("--input-tokens", type=int, default=1000)
    ap.add_argument("--output-tokens", type=int, default=128)
    ap.add_argument("--ttft-ms", type=float, default=10.0)
    ap.add_argument("--itl-ms", type=float, default=25.0)
    ap.add_argument("--server-workers", type=int, default=4)
    ap.add_argument("--connector-limit", type=int, default=None,
                    help="None = trace_loadgen stock (aiohttp default 100); 0 = unlimited")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workdir", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    args.workdir = args.workdir or tempfile.mkdtemp(prefix="m15_")
    Path(args.workdir).mkdir(parents=True, exist_ok=True)

    if args.mode == "dummy":
        port = 18000 + os.getpid() % 2000
        be = DummyBackend(port, args.server_workers, args.ttft_ms / 1000, args.itl_ms / 1000,
                          args.workdir)
        be.start()
        try:
            raw = sweep(args, f"http://127.0.0.1:{port}/generate")
        finally:
            recv = be.stop()
        points = finalize(raw, recv, args)
    else:
        # GPU SKELETON (unvalidated).  1-token, short-prompt requests make prefill/decode work
        # negligible, so TTFT excess over the lowest rate = frontend (TokenizerManager) queueing.
        if not args.endpoint:
            raise SystemExit("--endpoint required in endpoint mode")
        args.output_tokens, args.input_tokens = 1, min(args.input_tokens, 16)
        raw = sweep(args, args.endpoint)
        points = finalize(raw, {}, args)       # no server-side receive times on a real server
    result = {"mode": args.mode, "connector_limit": args.connector_limit,
              "params": {k: getattr(args, k) for k in
                         ("rates", "duration_s", "input_tokens", "output_tokens", "ttft_ms",
                          "itl_ms", "server_workers", "seed")},
              "host_cpus": os.cpu_count(), "candidate_thresholds": CAND,
              "points": points,
              "max_passing_rate_req_s": max((p["rate_req_s"] for p in points
                                             if all(q["PASS_candidate"] for q in points
                                                    if q["rate_req_s"] <= p["rate_req_s"])),
                                            default=None)}
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"[m15] max rate with all-lower-rates PASS (candidate thresholds): "
          f"{result['max_passing_rate_req_s']} req/s -> {args.out}")


if __name__ == "__main__":
    main()
