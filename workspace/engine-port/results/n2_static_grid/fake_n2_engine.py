"""CPU-only FAKE SGLang + PD-mux engine for the N2 probe harness tests (test_n2_harness.py). NOT engine code.

An aiohttp app serving /health, /generate (SSE, SGLang meta_info shape) and /abort_request, driven by a simulated
sticky-ON legacy loop that writes PHASE_EVENTS-shaped telemetry (prefill_span_start/_end, decode_iteration,
runtime_snapshot) with time.monotonic() -- the same CLOCK_MONOTONIC the client marks use -- so the harness output can
be fed to the registered reducer end to end. Prefill: one split-prefill span at a time, packing requests up to 16,384
tokens (co-load 8192 -> 2 per span); a request with max_new_tokens == 1 finishes at its span end and never joins the
decode batch (ENGINE_CHECK (a)). Decode: one iteration per step on PAIR[d] (sticky ON), slower by `kappa` while a
prefill span is in flight. Failure modes for the exception tests: hang, bad_coload, ignore_abort, drop_after.
"""
from __future__ import annotations

import asyncio
import json
import time

PAIR = {16: (92, 16), 24: (84, 24), 34: (74, 34), 44: (64, 44)}
IDX = {16: 1, 24: 2, 34: 3, 44: 4}


class _R:
    def __init__(self, rid, n_in, max_new):
        self.rid, self.n_in, self.max_new = rid, n_in, max_new
        self.gen = 0
        self.q = asyncio.Queue()
        self.finished = False
        self.aborted = False


class FakeEngine:
    def __init__(self, d=44, tau_s=0.001, kappa=1.3, burst_span_s=0.002, co_span_s=0.1, snapshot_every=32,
                 idle_snapshot_s=0.01, chunk_every=1, mode=None, drop_after=None, sticky_log=True,
                 co_response_delay_s=0.0):
        self.d, self.tau_s, self.kappa = d, tau_s, kappa
        self.burst_span_s, self.co_span_s = burst_span_s, co_span_s
        self.snapshot_every, self.idle_snapshot_s, self.chunk_every = snapshot_every, idle_snapshot_s, chunk_every
        self.mode = set(mode or ())
        self.drop_after = drop_after
        self.sticky_log = sticky_log
        self.co_response_delay_s = co_response_delay_s
        self.waiting, self.running = [], []
        self.prefill = None
        self.span_seq = 0
        self.iter_seq = 0
        self.tel = None
        self.arrivals = {}                  # rid -> monotonic arrival time at the server
        self.coload_arrivals = []
        self._task = None
        self.exit_code = None
        self.log_lines = []
        self.stopped = False

    # ---- telemetry
    def emit(self, event, **f):
        rec = dict(event=event, timestamp_monotonic_s=time.monotonic(), dropped_events=0, phase="benchmark", **f)
        self.tel.write(json.dumps(rec) + "\n")
        self.tel.flush()

    def snapshot(self):
        ctx = sorted(r.n_in + r.gen for r in self.running)
        p50 = ctx[int((len(ctx) - 1) * 0.5)] if ctx else 1
        self.emit("runtime_snapshot", decode_iterations=self.iter_seq, active_decode_sequences=len(self.running),
                  decode_running_batch_size=len(self.running), context_p50=p50,
                  context_p95=ctx[int((len(ctx) - 1) * 0.95)] if ctx else 1, context_max=max(ctx, default=1),
                  prefill_queue_depth=len(self.waiting))

    # ---- loop
    async def loop(self):
        pr, de = PAIR[self.d]
        last_idle = 0.0
        while not self.stopped:
            now = time.monotonic()
            if self.prefill is not None and now >= self.prefill[1]:
                reqs = self.prefill[0]
                self.emit("prefill_span_end", span_seq=self.span_seq, n_done=len(reqs), running_bs=len(self.running),
                          queue_len=len(self.waiting), stream_idx=IDX[self.d])
                for r in reqs:
                    if r.aborted:
                        continue
                    r.gen = 1
                    if r.max_new == 1:
                        ct = 2 if "bad_coload" in self.mode else 1
                        r.finished = True
                        if self.co_response_delay_s > 0:
                            asyncio.get_running_loop().call_later(self.co_response_delay_s, r.q.put_nowait, (ct, True))
                        else:
                            r.q.put_nowait((ct, True))
                    else:
                        r.q.put_nowait((1, False))
                        self.running.append(r)
                self.prefill = None
            if self.prefill is None and self.waiting:
                take, tok = [], 0
                while self.waiting and (not take or tok + self.waiting[0].n_in <= 16384):
                    r = self.waiting.pop(0)
                    if r.aborted:
                        continue
                    take.append(r); tok += r.n_in
                if take:
                    self.span_seq += 1
                    self.emit("prefill_span_start", span_seq=self.span_seq, n_new=len(take), extend_tokens=tok,
                              running_bs=len(self.running), queue_len=len(self.waiting), cap=48,
                              stream_idx=IDX[self.d], prefill_sms=pr, decode_sms=de)
                    co = all(r.n_in == 8192 for r in take)
                    self.prefill = (take, time.monotonic() + (self.co_span_s if co else self.burst_span_s))
            if self.running:
                in_flight = self.prefill is not None
                bs = len(self.running)
                t_launch = time.monotonic()
                await asyncio.sleep(self.tau_s * (self.kappa if in_flight else 1.0))
                t_sync = time.monotonic()
                self.iter_seq += 1
                for r in list(self.running):
                    if r.aborted and "ignore_abort" not in self.mode:
                        self.running.remove(r)
                        continue
                    r.gen += 1
                    done = r.gen >= r.max_new
                    if done or r.gen % self.chunk_every == 0:
                        r.q.put_nowait((r.gen, done))
                    if done:
                        r.finished = True
                        self.running.remove(r)
                self.emit("decode_iteration", iter_seq=self.iter_seq, t_launch=t_launch, t_sync=t_sync, bs=bs,
                          stream_idx=IDX[self.d], prefill_sms=pr, decode_sms=de, prefill_in_flight=in_flight,
                          prefill_kernel_done_wait=False, queue_len=len(self.waiting))
                if self.iter_seq % self.snapshot_every == 0:
                    self.snapshot()
            else:
                if "no_idle_snapshot" not in self.mode and time.monotonic() - last_idle >= self.idle_snapshot_s:
                    self.snapshot()
                    last_idle = time.monotonic()
                await asyncio.sleep(0.001)

    # ---- http
    async def health(self, request):
        from aiohttp import web
        return web.Response(text="ok")

    async def abort(self, request):
        from aiohttp import web
        body = await request.json()
        for r in self.running + self.waiting:
            if r.rid == body.get("rid"):
                r.aborted = True
        return web.Response(text="")

    async def generate(self, request):
        from aiohttp import web
        body = await request.json()
        rid = body["rid"]
        self.arrivals[rid] = time.monotonic()
        mx = int(body["sampling_params"]["max_new_tokens"])
        r = _R(rid, len(body["input_ids"]), mx)
        if mx == 1:
            self.coload_arrivals.append(self.arrivals[rid])
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        if "hang" in self.mode:
            await asyncio.sleep(3600)
        self.waiting.append(r)
        sent = 0
        try:
            while True:
                ct, done = await r.q.get()
                if self.drop_after is not None and ct >= self.drop_after:
                    request.transport.close()          # mid-response disconnect (server "crash")
                    self.exit_code = 1 if "exit_on_drop" in self.mode else None
                    return resp
                meta = {"completion_tokens": ct, "prompt_tokens": r.n_in, "id": rid,
                        "finish_reason": ({"type": "length", "length": ct} if done else None)}
                await resp.write(f"data: {json.dumps({'text': '', 'meta_info': meta})}\n\n".encode())
                sent += 1
                if done:
                    break
            await resp.write(b"data: [DONE]\n\n")
        except asyncio.CancelledError:
            r.aborted = True
            raise
        except Exception:                                  # client went away (cancel): stop generating for it
            r.aborted = True
        return resp


class FakeServerHandle:
    """ServerHandle for n2_probe_run.run_boot: runs a FakeEngine app in the current event loop."""

    def __init__(self, engine_kw=None, fail_start=None):
        self.engine_kw = engine_kw or {}
        self.fail_start = fail_start
        self.engine = None
        self.runner = None
        self.port = None
        self.stop_called_at = None
        self._log = None

    @property
    def base(self):
        return f"http://127.0.0.1:{self.port}"

    def start(self, env, log_path):
        self.env = env
        self._log = log_path
        d = int(env["PDMUX_R2_FIXED_DSM"])
        self.engine = FakeEngine(d=d, **self.engine_kw)
        self.engine.tel = open(env["PDMUX_TELEMETRY_PATH"], "a")
        with open(log_path, "x") as fh:
            if self.engine.sticky_log and env.get("PDMUX_STICKY_PARTITION") == "1":
                fh.write(f"PD-mux sticky partition ENABLED: decode holds its division while decode is busy "
                         f"(fixed target index={IDX[d]}); ...\n")

    async def wait_healthy(self, session):
        from aiohttp import web
        if self.fail_start == "exit":
            self.engine.exit_code = 1
            return False
        if self.fail_start == "timeout":                   # never healthy, process still alive
            return False
        app = web.Application()
        app.router.add_get("/health", self.engine.health)
        app.router.add_post("/generate", self.engine.generate)
        app.router.add_post("/abort_request", self.engine.abort)
        self.runner = web.AppRunner(app, shutdown_timeout=0.1)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        self.port = site._server.sockets[0].getsockname()[1]
        self.engine._task = asyncio.ensure_future(self.engine.loop())
        return True

    def exit_code(self, grace_s=0.0):
        return None if self.engine is None else self.engine.exit_code

    def log_text(self):
        with open(self._log) as fh:
            return fh.read()

    async def stop(self):
        self.stop_called_at = time.monotonic()
        if self.engine is None:
            return
        self.engine.stopped = True
        if self.engine.exit_code is None:
            self.engine.exit_code = -15                    # like the runner's SIGTERM: an exit code now exists
        if self.engine._task is not None:
            await asyncio.gather(self.engine._task, return_exceptions=True)
        if self.runner is not None:
            await self.runner.cleanup()
        self.engine.tel.close()
