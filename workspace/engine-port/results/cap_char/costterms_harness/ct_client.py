#!/usr/bin/env python3
"""Cost-term harness, stage 1 -- spec-driven cell client (T1 burst, T2 single-request TTFT, T2' collection, T4 fill).

The programs below talk to a TRANSPORT with five methods -- now(), start(rid, ids, max_new) -> handle,
wait(pred, timeout), abort(handle), sleep(dt) (+ greedy(ids, n)) -- so the same client code runs against the real
server (HttpTransport: one thread per streamed request, CLOCK_MONOTONIC via time.perf_counter, the clock the engine's
decode_forward t_host_start uses) and against the fake engine's deterministic VirtualTransport.

Every request is `input_ids` of exactly the registered length (prompt seed §2-3), `ignore_eos`, temperature 0, streamed.
Cell records carry the per-request chunk times (t, cumulative tokens); nothing is reduced here (ct_extract does that).
"""
from __future__ import annotations

import hashlib
import http.client
import json
import threading
import time

import ct_spec as S


# ============================================================================ HTTP transport
class HttpHandle:
    def __init__(self, rid, n_in, max_new, t_send):
        self.rid, self.n_in, self.max_new, self.t_send = rid, n_in, max_new, t_send
        self.chunks, self.done, self.finish, self.error, self.aborted = [], False, None, None, False
        self.prompt_tokens = None
        self.conn = None

    def record(self):
        return {"rid": self.rid, "t_send": self.t_send, "chunks": [list(c) for c in self.chunks],
                "t_first": self.chunks[0][0] if self.chunks else None,
                "t_done": self.chunks[-1][0] if self.done and self.chunks else None,
                "prompt_tokens": self.prompt_tokens, "completion_tokens": self.chunks[-1][1] if self.chunks else 0,
                "finish": self.finish, "error": self.error, "aborted": self.aborted}


class HttpTransport:
    def __init__(self, port, host="127.0.0.1", timeout=1800.0):
        self.port, self.host, self.timeout = port, host, timeout

    @staticmethod
    def now():
        return time.perf_counter()

    def _run(self, h, ids):
        body = json.dumps({"input_ids": list(ids), "rid": h.rid, "stream": True,
                           "sampling_params": {"max_new_tokens": int(h.max_new), "temperature": 0.0, "ignore_eos": True}})
        try:
            h.conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
            h.conn.request("POST", "/generate", body, {"Content-Type": "application/json"})
            resp = h.conn.getresponse()
            if resp.status != 200:
                h.error = f"HTTP {resp.status}"
                return
            while True:
                line = resp.readline()
                if not line:
                    break
                s = line.decode("utf-8", "replace").strip()
                if not s.startswith("data:"):
                    continue
                s = s[5:].strip()
                if s == "[DONE]":
                    break
                t = time.perf_counter()
                ev = json.loads(s)
                mi = ev.get("meta_info") or {}
                if mi.get("prompt_tokens") is not None:
                    h.prompt_tokens = mi["prompt_tokens"]
                ct = mi.get("completion_tokens")
                if ct is not None and (not h.chunks or ct > h.chunks[-1][1]):
                    h.chunks.append((t, int(ct)))
                fr = mi.get("finish_reason")
                if fr is not None:
                    h.finish = fr.get("type") if isinstance(fr, dict) else str(fr)
                if "error" in ev:
                    h.error = str(ev["error"])[:300]
        except Exception as exc:  # noqa: BLE001
            if not h.aborted:
                h.error = repr(exc)[:300]
        finally:
            h.done = True
            try:
                h.conn.close()
            except Exception:  # noqa: BLE001
                pass

    def start(self, rid, ids, max_new):
        h = HttpHandle(rid, len(ids), max_new, time.perf_counter())
        threading.Thread(target=self._run, args=(h, ids), daemon=True).start()
        return h

    def abort(self, h):
        h.aborted = True
        try:
            c = http.client.HTTPConnection(self.host, self.port, timeout=30)
            c.request("POST", "/abort_request", json.dumps({"rid": h.rid}), {"Content-Type": "application/json"})
            c.getresponse().read()
            c.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            h.conn.sock.close()
        except Exception:  # noqa: BLE001
            pass
        t0 = time.perf_counter()
        while not h.done and time.perf_counter() - t0 < 30:
            time.sleep(0.005)
        h.finish = h.finish or "abort"

    def wait(self, pred, timeout=1800.0):
        t0 = time.perf_counter()
        while not pred():
            if time.perf_counter() - t0 > timeout:
                return False
            time.sleep(0.002)
        return True

    @staticmethod
    def sleep(dt):
        time.sleep(dt)

    def health(self):
        try:
            c = http.client.HTTPConnection(self.host, self.port, timeout=5)
            c.request("GET", "/health")
            ok = c.getresponse().status == 200
            c.close()
            return ok
        except Exception:  # noqa: BLE001
            return False

    def greedy(self, ids, max_new):
        try:
            c = http.client.HTTPConnection(self.host, self.port, timeout=600)
            c.request("POST", "/generate", json.dumps({"input_ids": list(ids), "sampling_params": {
                "max_new_tokens": int(max_new), "temperature": 0.0}}), {"Content-Type": "application/json"})
            r = c.getresponse()
            data = json.loads(r.read())
            c.close()
            if isinstance(data, list):
                data = data[0]
            return {"ok": r.status == 200, "output_ids": data.get("output_ids") or []}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "output_ids": [], "error": repr(exc)}


# ============================================================================ programs
def _all_done(hs):
    return lambda: all(h.done for h in hs)


def cancel_burst(tr, hs):
    """Abort every still-running request of a burst and wait until each stream has ended (HTTP /abort_request; the
    next cell must start on an engine with no residual request -- checked again by the extractor, FOREIGN_RESIDUAL)."""
    for h in hs:
        if not h.done:
            tr.abort(h)
    tr.wait(_all_done(hs), 120.0)


def run_t1_cell(tr, cell_id, bs, c, rnd, O, timeout=1800.0, mark=None):
    """§2-1 burst (rev3.4, decision 28 A-1): bs requests at once, c_in = c - X, max_new = O (registered per cell:
    O_BASE + ramp bound), ignore_eos.  The client waits until every request has received ABORT_AT_TOKENS tokens (the
    ramp-aware window r0 + TRIM + WINDOW has then been executed) and aborts the burst."""
    c_in = c - S.X_TOKENS
    ids = S.prompt_ids("T1", bs, c, rnd, c_in)
    rec = {"kind": "t1", "cell_id": cell_id, "bs": bs, "c": c, "c_in": c_in, "O": int(O), "round": rnd}
    if mark:
        mark("cell_start", cell_id)
    rec["t_first_send"] = tr.now()
    hs = [tr.start(f"{cell_id}-q{i}", ids, int(O)) for i in range(bs)]

    def tokens(h):
        return h.chunks[-1][1] if h.chunks else 0
    rec["window_reached"] = tr.wait(lambda: all(h.done or tokens(h) >= S.ABORT_AT_TOKENS for h in hs), timeout)
    joined = [h.chunks[0][0] for h in hs if h.chunks]
    rec["t_reach_client"] = max(joined) if len(joined) == bs else None
    rec["n_ended_before_abort"] = sum(1 for h in hs if h.done)
    rec["t_abort"] = tr.now()
    cancel_burst(tr, hs)
    rec["completed"] = all(h.done for h in hs)
    rec["t_end"] = tr.now()
    rec["requests"] = [h.record() for h in hs]
    if mark:
        mark("cell_end", cell_id)
    return rec


def run_t2(tr, seq_id, L, rnd, reps=S.T2_REPS, timeout=1800.0, mark=None, term="T2"):
    """§1-2: one request at a time, exactly L input tokens, 1 output token; every rep recorded (rep 0 = warm-up,
    excluded by the extractor, kept here)."""
    ids = S.prompt_ids(term, 1, L, rnd, L)
    rec = {"kind": "t2", "seq_id": seq_id, "L": L, "round": rnd, "reps": []}
    if mark:
        mark("cell_start", seq_id)
    rec["t_first_send"] = tr.now()
    for i in range(reps):
        h = tr.start(f"{seq_id}-p{i}", ids, 1)
        tr.wait(lambda: h.done, timeout)
        r = h.record()
        r.update(rep=i, ttft_s=(r["t_first"] - r["t_send"]) if r["t_first"] is not None else None,
                 ok=(r["error"] is None and r["prompt_tokens"] == L and r["completion_tokens"] == 1 and r["finish"] == "length"))
        rec["reps"].append(r)
    rec["t_end"] = tr.now()
    if mark:
        mark("cell_end", seq_id)
    return rec


def run_t2prime(tr, seq_id, L, rnd, reps=S.T2_REPS, timeout=1800.0, mark=None):
    """§1-8 T2': the same single-request TTFT while ONE bs=1 decode request (ctx 2k) is running.  Each rep records
    whether the decode stream was demonstrably alive across the TTFT interval (a chunk before the send and after
    the first token), then the decode request is aborted."""
    dec_ids = S.prompt_ids("T2P_DEC", 1, S.T2P_DECODE_CTX, rnd, S.T2P_DECODE_CTX)
    ids = S.prompt_ids("T2P", 1, L, rnd, L)
    rec = {"kind": "t2p", "seq_id": seq_id, "L": L, "round": rnd, "reps": []}
    if mark:
        mark("cell_start", seq_id)
    rec["t_first_send"] = tr.now()
    d = tr.start(f"{seq_id}-dec", dec_ids, S.T2P_DECODE_MAX_NEW)
    tr.wait(lambda: d.done or (len(d.chunks) and d.chunks[-1][1] >= S.T2P_DECODE_MIN_TOKENS), timeout)
    for i in range(reps):
        h = tr.start(f"{seq_id}-p{i}", ids, 1)
        tr.wait(lambda: h.done, timeout)
        r = h.record()
        t_s, t_f = r["t_send"], r["t_first"]
        # decode alive across the TTFT: a decode token arrived before the send and another after the first token
        tr.wait(lambda: d.done or (d.chunks and t_f is not None and d.chunks[-1][0] > t_f), 60.0)
        before = any(t <= t_s for t, _ in d.chunks)
        after = t_f is not None and any(t > t_f for t, _ in d.chunks)
        r.update(rep=i, ttft_s=(t_f - t_s) if t_f is not None else None, coexist=bool(before and after and not d.done),
                 ok=(r["error"] is None and r["prompt_tokens"] == L and r["completion_tokens"] == 1 and r["finish"] == "length"))
        rec["reps"].append(r)
    tr.abort(d)
    rec["decode"] = d.record()
    rec["t_end"] = tr.now()
    if mark:
        mark("cell_end", seq_id)
    return rec


def run_t4_fill(tr, fill_id, L, R, N, rnd=1, timeout=7200.0, mark=None):
    """§1-4 (b): N = cap + 8 requests of length L at once, output R (= the reservation), ignore_eos."""
    ids = S.prompt_ids("T4", N, L, rnd, L)
    rec = {"kind": "t4", "fill_id": fill_id, "L": L, "R": R, "N": N}
    if mark:
        mark("cell_start", fill_id)
    rec["t_first_send"] = tr.now()
    hs = [tr.start(f"{fill_id}-q{i}", ids, R) for i in range(N)]
    rec["completed"] = tr.wait(_all_done(hs), timeout)
    rec["t_end"] = tr.now()
    rec["requests"] = [h.record() for h in hs]
    if mark:
        mark("cell_end", fill_id)
    return rec


def run_warmup(tr, boot_id, timeout=600.0):
    """§2-2 warm-up: WARMUP_N concurrent requests of one shape (A-? harness reading), then a greedy correctness probe."""
    ids = S.prompt_ids("WARM", S.WARMUP_N, S.WARMUP_IN, 0, S.WARMUP_IN)
    hs = [tr.start(f"{boot_id}-warm{i}", ids, S.WARMUP_OUT) for i in range(S.WARMUP_N)]
    ok = tr.wait(_all_done(hs), timeout)
    g = tr.greedy(S.prompt_ids("GREEDY", 1, S.GREEDY_IN, 0, S.GREEDY_IN), S.GREEDY_OUT)
    return {"warmup_ok": ok and all(h.error is None and h.finish == "length" for h in hs),
            "greedy_ok": bool(g.get("ok")) and len(g.get("output_ids") or []) == S.GREEDY_OUT,
            "greedy_ids_sha256": hashlib.sha256(json.dumps(g.get("output_ids") or []).encode()).hexdigest()}


def run_trailer(tr, boot_id, timeout=600.0):
    """§2-2 trailing non-measurement request, then idle 2 x flush."""
    h = tr.start(f"{boot_id}-trail", S.prompt_ids("TRAIL", 1, 256, 0, 256), 16)
    tr.wait(lambda: h.done, timeout)
    tr.sleep(S.IDLE_FLUSHES * S.PROBE_WINDOW_S + 0.1)
    return h.error is None
