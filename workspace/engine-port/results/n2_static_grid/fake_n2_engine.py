"""CPU-only FAKE SGLang + PD-mux engine for the N2 harness tests (test_n2_harness.py, test_n2_job.py). NOT engine code.

An aiohttp app serving /health, /generate (SSE, SGLang meta_info shape; non-stream JSON for the §11-2 greedy probe)
and /abort_request, driven by a simulated legacy PD-mux loop that writes PHASE_EVENTS-shaped telemetry
(prefill_span_start/_end, decode_iteration, runtime_snapshot) with time.monotonic() -- the same CLOCK_MONOTONIC the
client marks use -- so the harness output can be fed to the registered reducer / windows end to end. Prefill: one
split-prefill span at a time, packing requests up to 16,384 tokens (co-load 8192 -> 2 per span); a request with
max_new_tokens == 1 finishes at its span end and never joins the decode batch (ENGINE_CHECK (a)). Decode: one iteration
per step, slower by `kappa` while a prefill span is in flight.

Partition model (A-agn, §3-1 / §8-1): sticky ON -> every decode iteration on PAIR[d] (stream idx(d)); sticky OFF ->
decode on PAIR[d] only while a prefill span is in flight, else on the unpartitioned (0,108) group (idx 5).
Job extensions (rev17 harness, all default-off so the smoke tests are unchanged): server log lines (`Decode batch ...
cuda graph: True` every `decode_log_every` iterations with a wall-clock stamp from `wall`, `Prefill batch ... cuda
graph: False`, P3 banners, capture-bs line), green read-out JSON, PHASE_EVENTS off (snapshots only), recorder defects
(`drop_iters`: iter_seqs whose decode_iteration is dropped and counted in dropped_events; `r7_at_iter`: PHASE_EVENTS
disabled after that iteration with the R7 log line), `retract_at_iter`, `cg_false_at_iter`, `cuda_graph=False` (every Decode batch line `cuda graph: False` -- CG canary
§18-2-d #16), greedy modes (`empty`, `diverge`, `no_ids` = text without output_ids, H17-1).
Failure modes for the exception tests: hang, bad_coload, ignore_abort, drop_after, exit_on_drop, no_idle_snapshot.
EA semantic canary (§18-2-d #32) defect / shape modes: co_no_end (no prefill_span_end for a co-load span),
no_inflight_flag (decode_iteration never says prefill_in_flight), co_one_per_span ({1, 1, 1} co spans), and the
co_extend_delta / coloc_idx / co_gather_s knobs.
"""
from __future__ import annotations

import asyncio
import json
import time

PAIR = {16: (92, 16), 24: (84, 24), 34: (74, 34), 44: (64, 44)}
IDX = {16: 1, 24: 2, 34: 3, 44: 4}
PAIR108, IDX108 = (0, 108), 5
GROUPS = [(108, 0), (92, 16), (84, 24), (74, 34), (64, 44), (0, 108)]
CAPTURE_BS = [1, 2, 4, 8, 12, 16, 24, 32, 40, 48]


class _R:
    def __init__(self, rid, n_in, max_new, ids=None):
        self.rid, self.n_in, self.max_new = rid, n_in, max_new
        self.ids = ids or []
        self.gen = 0
        self.q = asyncio.Queue()
        self.finished = False
        self.aborted = False


def green_readout(sticky_idx):
    groups = []
    for i, (p, d) in enumerate(GROUPS):
        division = 1 <= i <= len(GROUPS) - 2
        row = dict(stream_index=i, target_prefill_sm=p, target_decode_sm=d,
                   is_sticky_target=(sticky_idx is not None and i == sticky_idx))
        for side, sm in (("prefill", p), ("decode", d)):
            row[side] = ({"green_ctx_is_null": False, "green_sm": {"smCount": sm}} if division
                         else {"green_ctx_is_null": True, "green_sm": None})
        groups.append(row)
    return {"probe": "pdmux_green_readout", "sticky_target_index": sticky_idx, "groups": groups}


class FakeEngine:
    def __init__(self, d=44, tau_s=0.001, kappa=1.3, burst_span_s=0.002, co_span_s=0.1, snapshot_every=32,
                 idle_snapshot_s=0.01, chunk_every=1, mode=None, drop_after=None, sticky_log=True,
                 co_response_delay_s=0.0, sticky=True, phase_events=True, wall=time.time, decode_log_every=40,
                 drop_iters=(), r7_at_iter=None, retract_at_iter=None, cg_false_at_iter=None, greedy_mode=None,
                 alone_tau_s=None, cuda_graph=True, co_extend_delta=0, coloc_idx=None, co_gather_s=0.0):
        self.d, self.tau_s, self.kappa = d, tau_s, kappa
        self.alone_tau_s = tau_s if alone_tau_s is None else alone_tau_s
        self.burst_span_s, self.co_span_s = burst_span_s, co_span_s
        self.snapshot_every, self.idle_snapshot_s, self.chunk_every = snapshot_every, idle_snapshot_s, chunk_every
        self.mode = set(mode or ())
        self.drop_after = drop_after
        self.sticky_log = sticky_log
        self.co_response_delay_s = co_response_delay_s
        self.sticky, self.phase_events = sticky, phase_events
        self.wall, self.decode_log_every = wall, decode_log_every
        self.drop_iters, self.r7_at_iter = set(drop_iters), r7_at_iter
        self.retract_at_iter, self.cg_false_at_iter = retract_at_iter, cg_false_at_iter
        self.greedy_mode = greedy_mode
        self.cuda_graph = cuda_graph            # False: EVERY Decode batch line says `cuda graph: False` (CG canary)
        # EA semantic canary defects (§18-2-d #32): co_extend_delta added to a co span's extend_tokens (the 8192 x
        # n_new identification breaks), coloc_idx = the stream_idx reported by in-flight decode iterations (None =
        # idx(d)), co_gather_s = hold a co-load head until it has waited this long (packs concurrent co requests).
        self.co_extend_delta, self.coloc_idx, self.co_gather_s = co_extend_delta, coloc_idx, co_gather_s
        self.dropped = 0
        self.waiting, self.running = [], []
        self.prefill = None
        self.span_seq = 0
        self.iter_seq = 0
        self.tel = None
        self.log = None
        self.arrivals = {}                  # rid -> monotonic arrival time at the server
        self.coload_arrivals = []
        self._task = None
        self.exit_code = None
        self.log_lines = []
        self.stopped = False

    # ---- telemetry / log
    def emit(self, event, **f):
        if not self.phase_events and event in ("decode_iteration", "prefill_span_start", "prefill_span_end"):
            return
        rec = dict(event=event, timestamp_monotonic_s=time.monotonic(), dropped_events=self.dropped,
                   phase="benchmark", **f)
        self.tel.write(json.dumps(rec) + "\n")
        self.tel.flush()

    def logline(self, text):
        if self.log is None:
            return
        self.log.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.wall()))}] {text}\n")
        self.log.flush()

    def _decode_pair(self, in_flight):
        if self.sticky or in_flight:
            return PAIR[self.d], IDX[self.d]
        return PAIR108, IDX108

    def snapshot(self):
        ctx = sorted(r.n_in + r.gen for r in self.running)
        p50 = ctx[int((len(ctx) - 1) * 0.5)] if ctx else 1
        if self.running:
            pair, idx = self._decode_pair(self.prefill is not None)
        else:
            pair, idx = (108, 0), 0
        self.emit("runtime_snapshot", decode_iterations=self.iter_seq, active_decode_sequences=len(self.running),
                  decode_running_batch_size=len(self.running), context_p50=p50,
                  context_p95=ctx[int((len(ctx) - 1) * 0.95)] if ctx else 1, context_max=max(ctx, default=1),
                  prefill_queue_depth=len(self.waiting), prefill_active_batch_size=(len(self.prefill[0])
                                                                                    if self.prefill else 0),
                  prefill_sms=pair[0], decode_sms=pair[1], stream_index=idx)

    # ---- loop
    async def loop(self):
        last_idle = 0.0
        while not self.stopped:
            now = time.monotonic()
            if self.prefill is not None and now >= self.prefill[1]:
                reqs = self.prefill[0]
                if not ("co_no_end" in self.mode and all(r.n_in == 8192 for r in reqs)):
                    self.emit("prefill_span_end", span_seq=self.span_seq, n_done=len(reqs),
                              running_bs=len(self.running), queue_len=len(self.waiting), stream_idx=IDX[self.d])
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
            hold = (self.co_gather_s > 0 and self.waiting and self.waiting[0].max_new == 1
                    and now - self.arrivals.get(self.waiting[0].rid, now) < self.co_gather_s)
            if self.prefill is None and self.waiting and not hold:
                take, tok = [], 0
                one = "co_one_per_span" in self.mode and self.waiting[0].n_in == 8192
                while self.waiting and (not take or (tok + self.waiting[0].n_in <= 16384 and not one)):
                    r = self.waiting.pop(0)
                    if r.aborted:
                        continue
                    take.append(r); tok += r.n_in
                if take:
                    self.span_seq += 1
                    pr, de = PAIR[self.d] if self.running else (108, 0)
                    co_ext = self.co_extend_delta if all(r.n_in == 8192 for r in take) else 0
                    self.emit("prefill_span_start", span_seq=self.span_seq, n_new=len(take), extend_tokens=tok + co_ext,
                              running_bs=len(self.running), queue_len=len(self.waiting), cap=48,
                              stream_idx=IDX[self.d] if self.running else 0, prefill_sms=pr, decode_sms=de)
                    self.logline(f"Prefill batch, #new-seq: {len(take)}, #new-token: {tok}, #cached-token: 0, "
                                 f"#running-req: {len(self.running)}, #queue-req: {len(self.waiting)}, "
                                 f"cuda graph: False, input throughput (token/s): 0.00")
                    co = all(r.n_in == 8192 for r in take)
                    self.prefill = (take, time.monotonic() + (self.co_span_s if co else self.burst_span_s))
            if self.running:
                in_flight = self.prefill is not None
                bs = len(self.running)
                t_launch = time.monotonic()
                await asyncio.sleep(self.tau_s * self.kappa if in_flight else self.alone_tau_s)
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
                pair, idx = self._decode_pair(in_flight)
                if in_flight and self.coloc_idx is not None:
                    idx = self.coloc_idx
                if "no_inflight_flag" in self.mode:
                    in_flight = False
                if self.iter_seq in self.drop_iters:
                    self.dropped += 1                          # async writer queue drop: no record, counter +1
                else:
                    self.emit("decode_iteration", iter_seq=self.iter_seq, t_launch=t_launch, t_sync=t_sync, bs=bs,
                              stream_idx=idx, prefill_sms=pair[0], decode_sms=pair[1], prefill_in_flight=in_flight,
                              prefill_kernel_done_wait=False, queue_len=len(self.waiting))
                if self.r7_at_iter is not None and self.iter_seq == self.r7_at_iter and self.phase_events:
                    self.phase_events = False
                    self.logline("PDMUX_PHASE_EVENTS disabled after failure: RuntimeError('fake')")
                if self.iter_seq % self.decode_log_every == 0:
                    cg = "False" if (not self.cuda_graph or (self.cg_false_at_iter is not None
                                                             and self.iter_seq >= self.cg_false_at_iter)) else "True"
                    retr = (", #retracted_reqs: 1" if (self.retract_at_iter is not None
                                                        and self.iter_seq >= self.retract_at_iter
                                                        and self.iter_seq - self.decode_log_every < self.retract_at_iter)
                            else "")
                    self.logline(f"Decode batch, #running-req: {bs}, #full token: 0, full token usage: 0.00, "
                                 f"mamba num: {bs}, mamba usage: 0.00{retr}, cuda graph: {cg}, "
                                 f"gen throughput (token/s): 0.00, #queue-req: {len(self.waiting)}")
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

    def _greedy(self, ids, mx):
        """Deterministic 'greedy' continuation (a function of the prompt only, like T = 0 decoding)."""
        if self.greedy_mode == "empty":
            return []
        base = sum(ids) + (7 if self.greedy_mode == "diverge" else 0)
        return [(base * 31 + 17 * k) % 50000 + 1000 for k in range(mx)]

    async def generate(self, request):
        from aiohttp import web
        body = await request.json()
        rid = body["rid"]
        self.arrivals[rid] = time.monotonic()
        mx = int(body["sampling_params"]["max_new_tokens"])
        if body.get("stream") is False:                    # §11-2 greedy probe (no telemetry, no batching)
            out = self._greedy(list(body["input_ids"]), mx)
            if self.greedy_mode == "no_ids":                   # a server that returns text but no output_ids
                return web.json_response({"text": " ".join(str(x) for x in out),
                                          "meta_info": {"prompt_tokens": len(body["input_ids"]),
                                                        "completion_tokens": len(out), "id": rid,
                                                        "finish_reason": {"type": "length", "length": len(out)}}})
            return web.json_response({"text": " ".join(str(x) for x in out), "output_ids": out,
                                      "meta_info": {"prompt_tokens": len(body["input_ids"]),
                                                    "completion_tokens": len(out), "id": rid,
                                                    "finish_reason": {"type": "length", "length": len(out)}}})
        r = _R(rid, len(body["input_ids"]), mx)
        if mx == 1:
            self.coload_arrivals.append(self.arrivals[rid])
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        if "hang" in self.mode:
            await asyncio.sleep(3600)
        self.waiting.append(r)
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
    """ServerHandle for n2_probe_run.run_boot / n2_job.run_boot: runs a FakeEngine app in the current event loop.
    job_mode=True: read sticky / PHASE_EVENTS from the boot env exactly as the engine does (unset = off), write the
    P3 banners, the capture-bs line and the green read-out JSON (the smoke tests keep job_mode=False)."""

    def __init__(self, engine_kw=None, fail_start=None, job_mode=False, green_override=None, log_extra=""):
        self.engine_kw = engine_kw or {}
        self.fail_start = fail_start
        self.job_mode = job_mode
        self.green_override = green_override
        self.log_extra = log_extra
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
        kw = dict(self.engine_kw)
        sticky = env.get("PDMUX_STICKY_PARTITION") == "1"
        if self.job_mode:
            kw.setdefault("sticky", sticky)
            kw.setdefault("phase_events", env.get("PDMUX_PHASE_EVENTS") == "1")
        self.engine = FakeEngine(d=d, **kw)
        self.engine.tel = open(env["PDMUX_TELEMETRY_PATH"], "a")
        with open(log_path, "x") as fh:
            if self.job_mode:
                fh.write("[fake] Mamba Cache is allocated. max_mamba_cache_size: 48, conv_state size: 0.09GB\n")
                fh.write(f"[fake] Capture cuda graph bs {CAPTURE_BS}\n")
                fh.write("[fake] max_total_num_tokens=2721290, chunked_prefill_size=-1, max_prefill_tokens=16384\n")
            if self.engine.sticky_log and sticky:
                fh.write(f"PD-mux sticky partition ENABLED: decode holds its division while decode is busy "
                         f"(fixed target index={IDX[d]}); ...\n")
            fh.write(self.log_extra)
        if self.job_mode:
            self.engine.log = open(log_path, "a")
            gp = env.get("PDMUX_GREEN_READOUT_PATH")
            if gp:
                with open(gp, "w") as fh:
                    json.dump(self.green_override or green_readout(IDX[d] if sticky else None), fh)

    async def wait_healthy(self, session):
        from aiohttp import web
        if self.fail_start == "exit":
            self.engine.exit_code = 1
            return False
        if self.fail_start == "timeout":                   # never healthy, process still alive
            return False
        app = web.Application(client_max_size=64 * 1024 * 1024)
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
        if self.engine.log is not None:
            self.engine.log.close()
