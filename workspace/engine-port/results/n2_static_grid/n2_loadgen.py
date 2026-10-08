"""N2(3-a) load generator (trace_loadgen extension, PREREG_N2_3A rev12 §4-2, §14-2, §15 `n2_loadgen.py`) -- the U-2
probe cell client. CPU-testable against a fake server (test_n2_harness.py). HARNESS MARK WRITER (H-2): together with
n2_marks.py this file writes every client mark (load_call_*, coload_sent, delta_l, fix_s) and decides every
cell_abort reason (H-4), so it belongs to the frozen file list.

One probe cell = one load call = one kappa measurement (R11-1), sticky-ON binding cell (d, 32, 8192):
  cell_start -> load_call_start -> b decode requests sent at once (input c_in, max_new_tokens O, ignore_eos, T = 0)
  -> client trigger: lower median L_(int((b-1)*0.5)) of L_i = c_in + tokens received >= c + 71 (§4-2 step 4)
  -> coload_sent <t read BEFORE the first co request> -> K co-load requests (input 8192, max_new_tokens = 1) all at
     once, no pacing (asyncio, one shared unlimited connector) -> every co response asserted completion_tokens == 1
     and finish_reason length (F3, else coload_payload_invalid) -> cancel when the lower median of the token counts
     has grown by 16 + 128 + delta since the moment the last co response arrived (§4-2 step 5, E-alpha rule; U-1 (b))
  -> load_call_end (finally) -> drain: cancel streams, POST /abort_request, wait for engine evidence that the
     running batch is empty (a runtime_snapshot after the close with decode_running_batch_size == 0 and no
     decode_iteration after it, §4-2 step 6) -> cell_end (finally, always) -> delta_l, fix_s.
Exception -> registered reason (H-4, first match): see classify_exception.
"""
from __future__ import annotations

import asyncio
import bisect
import hashlib
import json
import math
import os
import random
import statistics
import time

import n2_marks

CO_INPUT = 8192
TRIGGER_EXTRA = 71                       # c + 63 + 8 (§4-2 step 4)
EA_CANCEL_TOKENS = 16 + 128              # A2 trim + window; + delta (§4-2 step 5)
SOCK_CONNECT_S = 60.0                    # = trace_loadgen.py ClientTimeout
SOCK_READ_S = 600.0                      # = trace_loadgen.py ClientTimeout (aiohttp ServerTimeoutError -> TimeoutError)
# ---- harness constants NOT in the registration (reported as contract gaps; values are harness choices) ----------
TOKEN_LO, TOKEN_HI = 1000, 100000        # seed-token id range (§4-2 registers the seed, not the generator)
CELL_TIMEOUT_S = 1800.0                  # per-cell wall cap -> TimeoutError -> cell_timeout (VD)
DRAIN_TIMEOUT_S = 120.0                  # engine-idle evidence wait -> StreamCancelFailed -> stream_cancel_failed (VD)
SERVER_EXIT_GRACE_S = 5.0                # wait for an exit code before rule (3) server_down is decided


# ------------------------------------------------------------------------------- seeds / payloads (§4-2)
def seed_of(text):
    """§4-2: seed = the first 8 bytes of SHA-256(text) as an unsigned integer (big-endian)."""
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def seed_tokens(seed, n, lo=TOKEN_LO, hi=TOKEN_HI):
    rnd = random.Random(seed)
    return [rnd.randrange(lo, hi) for _ in range(int(n))]


def decode_prompt(kind, b, c, r, c_in):
    """Every decode request of the cell gets the first c_in tokens of the cell's seed sequence (§4-2 literal)."""
    return seed_tokens(seed_of(f"n2|{kind}|{b}|{c}|{r}"), c_in)


def coload_prompt(b, c, r, j):
    return seed_tokens(seed_of(f"n2|COLOAD|{b}|{c}|{r}|{j}"), CO_INPUT)


class TraceInvalid(ValueError):
    """ABORT_TRACE (§3-1 co-load payload row): co-load max_new_tokens must be exactly 1."""


def payload(input_ids, max_new_tokens, rid, coload=False):
    if coload and (max_new_tokens is None or int(max_new_tokens) != 1):
        raise TraceInvalid(f"co-load max_new_tokens must be 1 (got {max_new_tokens!r}) -- ABORT_TRACE")
    if not coload and int(max_new_tokens) < 1:
        raise TraceInvalid("decode max_new_tokens must be >= 1")
    return {"input_ids": list(input_ids),
            "sampling_params": {"max_new_tokens": int(max_new_tokens), "temperature": 0.0, "ignore_eos": True},
            "stream": True, "rid": str(rid)}


# ------------------------------------------------------------------------------- exceptions + H-4 mapping
class RequestFailed(Exception):
    """A request's own failure; `inner` is the original exception, `bytes_received` its response bytes so far."""

    def __init__(self, rid, inner, bytes_received):
        super().__init__(f"request {rid}: {type(inner).__name__}: {inner}")
        self.rid, self.inner, self.bytes_received = rid, inner, bytes_received


class CoLoadPayloadInvalid(Exception):
    pass


class StreamCancelFailed(Exception):
    pass


class RunnerStop(Exception):
    """Intentional stop (budget / deadline / operator) -- H-5: recorded as runner_stop (VALUE_DEPENDENT)."""


class HarnessError(Exception):
    """A harness-side impossibility (e.g. streams ended before the trigger) -- unregistered name -> VD."""


class ServerExited(Exception):
    """Boot level: the server process exited (before /health) -- classified by rule (3) once the exit code is seen."""


class BootHealthTimeout(Exception):
    """Boot level: /health never answered 200 and no exit code -- unregistered name 'boot_health_timeout' (VD)."""


class WiringCheckFailed(Exception):
    """Boot level: the sticky wiring line is absent from the server log -- 'wiring_check_failed' (VD)."""


def _connector_errors():
    try:
        import aiohttp
        return (aiohttp.ClientConnectorError, ConnectionRefusedError)
    except ImportError:                                   # pragma: no cover
        return (ConnectionRefusedError,)


def classify_exception(exc, server_exit_code=lambda: None):
    """H-4 (registered order, first match). (0) RunnerStop -> runner_stop (H-5); (1) isinstance(root, TimeoutError)
    -> cell_timeout, FIRST (asyncio.TimeoutError is TimeoutError on 3.11+; aiohttp ServerTimeoutError subclasses both
    ClientConnectionError and TimeoutError); (2) connection establishment failure before ANY response byte of that
    request (aiohttp.ClientConnectorError / refused) -> http_connection_error; (3) ONLY if the server process exit code
    is observed -> server_down; (4) co-load assertion (F3) -> coload_payload_invalid; (5) cell-boundary cancel failure
    -> stream_cancel_failed; (6) anything else (incl. mid-response disconnects) -> 'client_exception' (unregistered
    name -> VALUE_DEPENDENT, reason_code ABORT_UNREGISTERED)."""
    if isinstance(exc, (n2_marks.CellAborted, n2_marks.Classified)):
        return exc.reason
    if isinstance(exc, RunnerStop):
        return "runner_stop"
    if isinstance(exc, WiringCheckFailed):
        return "wiring_check_failed"
    root = n2_marks._root(exc)
    if isinstance(root, TimeoutError):
        return "cell_timeout"
    nbytes = getattr(exc, "bytes_received", 0)
    if isinstance(root, _connector_errors()) and not nbytes:
        return "http_connection_error"
    if server_exit_code() is not None:
        return "server_down"
    if isinstance(root, CoLoadPayloadInvalid):
        return "coload_payload_invalid"
    if isinstance(root, StreamCancelFailed):
        return "stream_cancel_failed"
    if isinstance(root, BootHealthTimeout):
        return "boot_health_timeout"
    return "client_exception"


# ------------------------------------------------------------------------------- streaming request
class Req:
    def __init__(self, rid, n_input, coload=False):
        self.rid, self.n_input, self.coload = rid, n_input, coload
        self.n = 0                     # tokens received (meta_info.completion_tokens)
        self.times = []                # receive time of each token (CLOCK_MONOTONIC)
        self.bytes = 0
        self.meta = None               # last meta_info
        self.done = False
        self.t_done = None
        self.cancelled = False
        self.resp = None


async def stream(session, url, req, body, on_token=None, clock=time.monotonic):
    """POST one streaming /generate request; record per-token receive times. Raises RequestFailed on any error
    except a cancellation requested by the controller (task.cancel())."""
    try:
        async with session.post(url, json=body) as resp:
            req.resp = resp
            resp.raise_for_status()
            while True:
                raw = await resp.content.readline()
                if not raw:
                    break
                req.bytes += len(raw)
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                chunk = json.loads(data)
                meta = chunk.get("meta_info") or {}
                req.meta = meta
                ct = int(meta.get("completion_tokens", req.n + 1))
                if ct > req.n:
                    now = clock()
                    req.times.extend([now] * (ct - req.n))
                    req.n = ct
                    if on_token is not None:
                        on_token(req)
        req.done, req.t_done = True, clock()
        if on_token is not None:
            on_token(req)
    except asyncio.CancelledError:
        req.cancelled = True
        raise
    except Exception as e:
        raise RequestFailed(req.rid, e, req.bytes) from e


def lower_median(values):
    v = sorted(values)
    return v[int((len(v) - 1) * 0.5)]


async def _wait_or_fail(event, tasks, what):
    """Wait until `event` is set; a failed task raises; if every task finishes first, return False."""
    waiter = asyncio.ensure_future(event.wait())
    try:
        while not event.is_set():
            live = [t for t in tasks if not t.done()]
            for t in tasks:
                if t.done() and not t.cancelled() and t.exception() is not None:
                    raise t.exception()
            if not live:
                return False
            await asyncio.wait([waiter, *live], return_when=asyncio.FIRST_COMPLETED)
        for t in tasks:
            if t.done() and not t.cancelled() and t.exception() is not None:
                raise t.exception()
        return True
    finally:
        waiter.cancel()


# ------------------------------------------------------------------------------- telemetry tail (engine evidence)
class TelemetryTail:
    """Reads complete JSONL lines appended to the engine telemetry file (same container, same clock).
    `read(start, kinds)` re-reads from a byte offset; `Cursor` follows the file incrementally."""

    KEEP = ('"decode_iteration"', '"prefill_span_start"', '"prefill_span_end"', '"runtime_snapshot"')

    def __init__(self, path):
        self.path = path

    def offset(self):
        try:
            return os.path.getsize(self.path)
        except OSError:
            return 0

    @staticmethod
    def _parse(data, kinds):
        out = []
        for ln in data.split(b"\n"):
            s = ln.decode("utf-8", errors="replace")
            if not s.strip() or not any(k in s for k in kinds):
                continue
            try:
                out.append(json.loads(s))
            except ValueError:
                continue
        return out

    def _chunk(self, start):
        """(complete-line bytes from `start`, next offset). A partial first line (start inside a line) is skipped."""
        try:
            with open(self.path, "rb") as fh:
                if start > 0:
                    fh.seek(start - 1)
                    prev = fh.read(1)
                else:
                    prev = b"\n"
                data = fh.read()
        except OSError:
            return b"", start
        skip = 0
        if prev != b"\n":
            nl = data.find(b"\n")
            if nl < 0:
                return b"", start
            skip = nl + 1
        end = data.rfind(b"\n")
        if end < skip:
            return b"", start + skip
        return data[skip:end + 1], start + end + 1

    def read(self, start, kinds=KEEP):
        return self._parse(self._chunk(start)[0], kinds)

    def cursor(self, start):
        return _Cursor(self, start)


class _Cursor:
    def __init__(self, tail, pos):
        self.tail, self.pos = tail, pos

    def poll(self, kinds):
        data, self.pos = self.tail._chunk(self.pos)
        return self.tail._parse(data, kinds)


async def wait_engine_idle(tail, start_offset, t_closed, timeout_s=DRAIN_TIMEOUT_S, poll_s=0.25, settle_s=0.5,
                           clock=time.monotonic):
    """§4-2 step 6 (engine evidence that the running batch is empty): runtime_snapshots stamped after t_closed with
    decode_running_batch_size == 0 and active_decode_sequences == 0, spanning >= settle_s, and no decode_iteration
    launched after the first of them. (Assumes the engine emits runtime_snapshot on idle loop syncs, as the N3 run-3
    telemetry shows; if it does not, this times out -> stream_cancel_failed, fail-closed.)
    Raises StreamCancelFailed on timeout."""
    deadline = clock() + timeout_s
    cur = tail.cursor(start_offset)
    zero_first, zero_last, iters_after = None, None, []
    while True:
        for e in cur.poll(('"decode_iteration"', '"runtime_snapshot"')):
            if e.get("event") == "decode_iteration":
                iters_after.append(e.get("t_launch", 0.0))
                continue
            ts = e.get("timestamp_monotonic_s", 0.0)
            if ts <= t_closed:
                continue
            idle = (e.get("decode_running_batch_size", e.get("active_decode_sequences")) == 0
                    and e.get("active_decode_sequences", 0) == 0)
            if idle:
                zero_first = ts if zero_first is None else zero_first
                zero_last = ts
            else:
                zero_first = zero_last = None
        if zero_first is not None and any(t > zero_first for t in iters_after):
            zero_first = zero_last = None
            iters_after = [t for t in iters_after if t > t_closed]
        if zero_first is not None and zero_last - zero_first >= settle_s:
            return zero_first
        if clock() > deadline:
            raise StreamCancelFailed(f"no idle runtime_snapshot run of {settle_s} s within {timeout_s} s after close")
        await asyncio.sleep(poll_s)


def delta_l_from(events, decode_req, t_co_recv, t_co_sent):
    """Δℓ = ℓ_d − ℓ_co (§8-4, R4-3), seconds. HARNESS OPERATIONALISATION (not registered -- contract gap G-5):
    ℓ_co = receive time of the last co response − engine time of the last co span's prefill_span_end;
    ℓ_d  = median over the lower-median decode request's tokens received after t_co_recv of
           (receive time − t_sync of the latest decode_iteration with t_sync <= receive time).
    Returns None if the engine evidence is missing."""
    starts = {e["span_seq"]: e for e in events if e.get("event") == "prefill_span_start"}
    ends = {e["span_seq"]: e for e in events if e.get("event") == "prefill_span_end"}
    co_end = [ends[k]["timestamp_monotonic_s"] for k, s in starts.items()
              if k in ends and s["timestamp_monotonic_s"] >= t_co_sent and s.get("n_new", 0) >= 1
              and s.get("extend_tokens") == s["n_new"] * CO_INPUT]
    syncs = sorted(e["t_sync"] for e in events if e.get("event") == "decode_iteration" and "t_sync" in e)
    if not co_end or not syncs or decode_req is None:
        return None
    l_co = t_co_recv - max(co_end)
    lags = []
    for tr in (t for t in decode_req.times if t > t_co_recv):
        k = bisect.bisect_right(syncs, tr) - 1
        if k >= 0:
            lags.append(tr - syncs[k])
    if not lags:
        return None
    return statistics.median(lags) - l_co


# ------------------------------------------------------------------------------- the probe cell
class CellSpec:
    def __init__(self, d, b, c, c_in, o_ea, k, delta, kind="PROBE", r=1):
        self.d, self.b, self.c, self.c_in, self.o_ea, self.k, self.delta = d, b, c, c_in, int(o_ea), k, delta
        self.kind, self.r = kind, r


async def _abort_requests(session, base, rids):
    for rid in rids:
        try:
            async with session.post(f"{base}/abort_request", json={"rid": rid}) as resp:
                await resp.read()
        except Exception:
            pass


async def run_probe_cell(session, base, mw, spec, cell_id, call_id, classify, tail=None, clock=time.monotonic,
                         cell_timeout_s=CELL_TIMEOUT_S, drain_timeout_s=DRAIN_TIMEOUT_S, drain_poll_s=0.25,
                         drain_settle_s=0.5):
    """Run ONE probe cell (= one load call = one kappa measurement). Writes, in order: cell_start, load_call_start,
    coload_sent, load_call_end (finally), [cell_abort + abort_detail], cell_end (finally), then delta_l and fix_s for
    a completed cell. Returns a dict of client-side facts; raises n2_marks.CellAborted on abort."""
    url = f"{base}/generate"
    info = dict(cell_id=cell_id, call_id=call_id, d=spec.d)
    reqs, co_reqs, tasks, co_tasks = [], [], [], []
    st = dict(trigger=asyncio.Event(), cancel=asyncio.Event(), base=None, t_co_recv=None, t_first_send=None,
              t_active_end=None, t_coload=None)
    off0 = tail.offset() if tail is not None else 0

    def on_decode_token(_req):
        counts = [r.n for r in reqs]
        if not st["trigger"].is_set() and spec.c_in + lower_median(counts) >= spec.c + TRIGGER_EXTRA:
            st["trigger"].set()
        if st["base"] is not None and lower_median(counts) >= st["base"] + EA_CANCEL_TOKENS + spec.delta:
            st["cancel"].set()

    def on_co_token(_req):
        if all(r.done for r in co_reqs) and st["base"] is None:
            st["t_co_recv"] = max(r.t_done for r in co_reqs)
            st["base"] = lower_median([r.n for r in reqs])

    async def body():
        prompt = decode_prompt(spec.kind, spec.b, spec.c, spec.r, spec.c_in)
        bodies = [payload(prompt, spec.o_ea, f"{cell_id}-d{i}") for i in range(spec.b)]
        co_bodies = [payload(coload_prompt(spec.b, spec.c, spec.r, j), 1, f"{cell_id}-co{j}", coload=True)
                     for j in range(spec.k)]                                   # built (and asserted) BEFORE sending
        mw.load_call_start(call_id)
        try:
            st["t_first_send"] = clock()
            for i, bd in enumerate(bodies):
                r = Req(bd["rid"], spec.c_in)
                reqs.append(r)
                tasks.append(asyncio.ensure_future(stream(session, url, r, bd, on_decode_token, clock)))
            if not await _wait_or_fail(st["trigger"], tasks, "trigger"):
                raise HarnessError("every decode stream ended before the co-load trigger")
            t_co = clock()                                                     # read BEFORE the first co request
            st["t_coload"] = t_co
            mw.coload_sent(cell_id, t_co)
            for bd in co_bodies:                                               # K at once, no pacing
                r = Req(bd["rid"], CO_INPUT, coload=True)
                co_reqs.append(r)
                co_tasks.append(asyncio.ensure_future(stream(session, url, r, bd, on_co_token, clock)))
            done_ev = asyncio.Event()

            def _co_done(_):
                if all(t.done() for t in co_tasks):
                    done_ev.set()
            for t in co_tasks:
                t.add_done_callback(_co_done)
            if not await _wait_or_fail(done_ev, tasks + co_tasks, "co-load"):
                raise HarnessError("streams ended before the co-load completed")
            for t in co_tasks:
                if t.exception() is not None:
                    raise t.exception()
            bad = [r.rid for r in co_reqs if not _co_payload_ok(r)]
            if bad:
                raise CoLoadPayloadInvalid(f"co-load responses violate completion_tokens == 1 / length: {bad[:3]}")
            if st["base"] is None:                                             # all co done but callback order
                on_co_token(None)
            await _wait_or_fail(st["cancel"], tasks, "cancel")                 # False = all streams ended (O reached)
            st["t_active_end"] = clock()
            st["median_at_cancel"] = lower_median([r.n for r in reqs])
        finally:
            for t in tasks + co_tasks:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*tasks, *co_tasks, return_exceptions=True)
            mw.load_call_end(call_id)

    async def drain(aborted):
        off_close = tail.offset() if tail is not None else 0
        for t in tasks + co_tasks:
            if not t.done():
                t.cancel()
        await asyncio.gather(*tasks, *co_tasks, return_exceptions=True)
        await _abort_requests(session, base, [r.rid for r in reqs + co_reqs if not r.done])
        st["t_closed"] = clock()
        if tail is not None and not aborted:          # after an abort the boot ends; the reason already stands
            await wait_engine_idle(tail, off_close, st["t_closed"], drain_timeout_s, drain_poll_s, drain_settle_s,
                                   clock)

    async with mw.cell(cell_id, spec.d, spec.b, spec.c, classify) as cell:
        cell.drain = drain
        await asyncio.wait_for(body(), timeout=cell_timeout_s)
    # completed cell: client scalars (written after cell_end, "셀 뒤")
    info.update(t_start=cell.t_start, t_end=cell.t_end, t_first_send=st["t_first_send"], t_coload=st["t_coload"],
                t_co_recv=st["t_co_recv"], t_active_end=st["t_active_end"],
                tokens=[r.n for r in reqs], co_tokens=[r.n for r in co_reqs], base=st["base"],
                median_at_cancel=st.get("median_at_cancel"))
    fix = (cell.t_end - cell.t_start) - (st["t_active_end"] - st["t_first_send"])
    mw.fix_s(cell_id, fix)
    info["fix_s"] = fix
    if tail is not None:
        med = sorted(reqs, key=lambda r: r.n)[int((len(reqs) - 1) * 0.5)] if reqs else None
        dl = delta_l_from(tail.read(off0, kinds=TelemetryTail.KEEP[:3]), med, st["t_co_recv"], st["t_coload"])
        if dl is not None and math.isfinite(dl):
            mw.delta_l(cell_id, dl)
            info["delta_l"] = dl
    return info


def _co_payload_ok(r):
    meta = r.meta or {}
    fr = meta.get("finish_reason")
    ftype = fr.get("type") if isinstance(fr, dict) else fr
    return r.done and int(meta.get("completion_tokens", -1)) == 1 and r.n == 1 and ftype == "length"


async def simple_request(session, base, input_ids, max_new_tokens, rid):
    """Non-measured request (warm-up, F-6 trailing record request)."""
    r = Req(rid, len(input_ids))
    await stream(session, f"{base}/generate", r, payload(input_ids, max_new_tokens, rid))
    return r


def new_session(**kw):
    import aiohttp
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=kw.pop("sock_connect", SOCK_CONNECT_S),
                                    sock_read=kw.pop("sock_read", SOCK_READ_S))
    return aiohttp.ClientSession(timeout=timeout, connector=aiohttp.TCPConnector(limit=0, limit_per_host=0))
