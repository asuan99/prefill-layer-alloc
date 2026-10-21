#!/usr/bin/env python3
"""Fake sglang server for the CPU tests of engine_gate_driver.py (NOT an engine, no GPU, no model).

Speaks /health and /generate (JSON and SSE) with deterministic outputs, paces STREAMED requests (FAKE_TOK_S per token),
and, when the probe env paths are set, writes decode_probe / admission / PD-mux phase records shaped like the real probes
(schemas imported from src/multiplex).  Behaviour switches (env FAKE_MODE, comma list):
  host_time       gpu_ms := host_ms (the I-4 'host clock' mutant)           diverge_on  ON boots emit different tokens
  pristine_survive  the un-patched conv file does not crash the server (outputs differ instead)
  pristine_same   ... and outputs equal the patched ones (falsification must FAIL)
  no_enabled_log  never log 'probe ENABLED'        off_writes_probe  OFF boots also write probe files
  no_pad          bs_padded := bs_real             die_after_stream  exit(3) after the first streamed request
FAKE_GATE_CONV = the engine tree's conv file: without the int64 marker and input >= 94,800 tokens the server dies
(illegal memory access + causal_conv1d traceback on stdout), like the pristine kernel is predicted to.
"""
import hashlib, http.server, importlib.util, json, os, socketserver, sys, threading, time
from pathlib import Path

ENG = Path(__file__).resolve().parents[1]
MODE = set(filter(None, os.environ.get("FAKE_MODE", "").split(",")))
TOK_S = float(os.environ.get("FAKE_TOK_S", "0.002"))
ROW_S = 0.002
CAP = 48
CONV = os.environ.get("FAKE_GATE_CONV", "")
MARK = "int64: token offsets"


def load(name):
    spec = importlib.util.spec_from_file_location("fake_" + name, ENG / "src" / "multiplex" / f"{name}.py")
    m = importlib.util.module_from_spec(spec); sys.modules["fake_" + name] = m; spec.loader.exec_module(m); return m


DP, AP = load("decode_probe"), load("admission_probe")
args = sys.argv[1:]
PORT = int(args[args.index("--port") + 1])
EAGER = "--disable-cuda-graph" in args
PDMUX = "--enable-pdmux" in args
CAPTURE = [] if EAGER else [1, 2, 4, 8, 12, 16, 24, 32, 40, 48]
DEC_P, ADM_P, TEL_P = (os.environ.get(k) for k in ("PDMUX_DECODE_PROBE_PATH", "PDMUX_ADMISSION_PATH", "PDMUX_TELEMETRY_PATH"))
ON = bool(DEC_P)
if "off_writes_probe" in MODE and not DEC_P:
    DEC_P = os.environ.get("FAKE_STRAY_PATH", "") or None
lock = threading.Lock()
state = {"active": int(os.environ.get("FAKE_BG_ACTIVE", "0") or 0), "waiting": 0, "cap_stops": 0, "streamed": 0}   # FAKE_BG_ACTIVE: constant background decode batch (decomp tests: the cells there never reach this server)
slots = threading.Semaphore(CAP)
files = {}


def emit(path, row):
    if not path:
        return
    with lock:
        f = files.setdefault(path, open(path, "a", buffering=1))
        f.write(json.dumps(row) + "\n")


EV = os.environ.get("PDMUX_DECODE_PROBE_GPU_EVENTS", "1") != "0"
if "no_enabled_log" not in MODE:
    if os.environ.get("PDMUX_DECODE_PROBE_PATH"):
        print(f"decode probe ENABLED (I-4 events={EV}, I-6; all arms; fake)", flush=True)
    if ADM_P:
        print("admission probe ENABLED (all arms; fake)", flush=True)
print("max_total_num_tokens=500000 context_len=131072 available_gpu_mem=10.0 GB", flush=True)
emit(DEC_P, {"event": "decode_probe_boot", "record_schema": DP.DECODE_SCHEMA, "capture_bs": CAPTURE,
             "disable_cuda_graph_padding": False, "arm_enable_pdmux": PDMUX})
seq = [0]


def pad(b):
    if EAGER:
        return b
    if "no_pad" in MODE:
        return b
    return next((c for c in CAPTURE if c >= b), b)


def recorder():
    nxt, last_win, n_dec, hist, atcap = time.perf_counter(), time.perf_counter(), 0, {}, 0
    pend = None          # (t, bs): finalised with the REAL gap to the next tick so gpu_ms never exceeds the bracket
    while True:
        nxt += ROW_S
        while time.perf_counter() < nxt:
            time.sleep(0.0002)
        t = time.perf_counter()
        if pend is not None and (DEC_P or TEL_P):
            t0, b0 = pend
            gap = t - t0
            seq[0] += 1
            host = 0.1
            row = {"event": "decode_forward", "record_schema": DP.DECODE_SCHEMA, "seq": seq[0], "mode": "decode",
                   "bs_real": b0, "bs_padded": pad(b0), "graph": "eager" if EAGER else "hit", "bs_pad_expected": pad(b0),
                   "pad_check": "ok", "eager_reason": "graph_runner_absent" if EAGER else None,
                   "t_host_start": t0, "decode_fwd_host_ms": host,
                   "decode_fwd_gpu_ms": None if not EV else (host if "host_time" in MODE else 0.9 * gap * 1000.0)}
            emit(DEC_P, row)
            n_dec += 1
            hist[b0] = hist.get(b0, 0) + 1
            atcap += 1 if b0 >= CAP else 0
            if TEL_P:
                emit(TEL_P, {"event": "decode_iteration", "t_launch": t0 + 0.0002, "t_sync": t0 + 0.95 * gap})
        with lock:
            b, w = state["active"], state["waiting"]
        pend = (t, b) if b > 0 else None
        if w > 0:
            with lock:
                state["cap_stops"] += 1
        if t - last_win >= 0.25 and ADM_P:
            cats = {c: 0 for c in AP.TERMINAL_CATEGORIES}
            with lock:
                cs = state["cap_stops"]; state["cap_stops"] = 0
            cats["cap"] = cs
            emit(ADM_P, {"event": "admission_window", "record_schema": AP.ADMISSION_SCHEMA, "t_mono_start": last_win,
                         "t_mono_end": t, "n_decisions": cs, "terminal_by_category": cats,
                         "n_decode_iter": sum(hist.values()), "n_decode_iter_at_or_above_cap": atcap,
                         "decode_bs_hist": {str(k): v for k, v in hist.items()}})
            emit(DEC_P, {"event": "decode_probe_window", "record_schema": DP.DECODE_SCHEMA, "n_decode": n_dec, "n_pending": 0})
            last_win, n_dec, hist, atcap = t, 0, {}, 0


threading.Thread(target=recorder, daemon=True).start()


def out_ids(req, n):
    key = json.dumps(req.get("input_ids") if req.get("input_ids") is not None else req.get("text"), sort_keys=True)
    h = int(hashlib.sha256(key.encode()).hexdigest(), 16)
    ids = [(h >> (3 * i)) % 50000 + 1 for i in range(n)] if n < 80 else [((h >> (i % 200)) + i * 7919) % 50000 + 1 for i in range(n)]
    if ("diverge_on" in MODE and ON) and ids:
        ids[-1] += 1
    conv_ok = (not CONV) or MARK in Path(CONV).read_text()
    n_in = len(req.get("input_ids") or [])
    if not conv_ok and n_in >= 94800:
        if "pristine_survive" in MODE and ids:
            ids[0] += 1
        elif "pristine_same" in MODE:
            pass
        else:
            print("CUDA error: an illegal memory access was encountered", flush=True)
            if os.environ.get("CUDA_LAUNCH_BLOCKING") == "1":     # async (non-blocking) crashes do not name the kernel file (G-beta round 1)
                print("Traceback (most recent call last):\n  File \"causal_conv1d_triton.py\", line 83, in causal_conv1d_fn", flush=True)
            os._exit(1)
    return ids


class H(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(data)))
        self.end_headers(); self.wfile.write(data)

    def do_GET(self):
        self._send(200, b"ok", "text/plain") if self.path == "/health" else self._send(404, {})

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        sp = req.get("sampling_params") or {}
        n = int(sp.get("max_new_tokens", 16))
        n_in = len(req.get("input_ids") or [1] * 4)
        ids = out_ids(req, n)
        mi = {"prompt_tokens": n_in, "completion_tokens": n, "finish_reason": {"type": "length"}}
        if not req.get("stream"):
            return self._send(200, {"text": "", "output_ids": ids, "meta_info": mi})
        if "die_after_stream" in MODE and state["streamed"] > 0:
            os._exit(3)
        state["streamed"] += 1
        with lock:
            state["waiting"] += 1
        slots.acquire()
        with lock:
            state["waiting"] -= 1
            state["active"] += 1
        self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def chunk(b):
            self.wfile.write(f"{len(b):x}\r\n".encode() + b + b"\r\n"); self.wfile.flush()
        try:
            for k in range(1, n + 1):
                time.sleep(TOK_S)
                m = dict(mi, completion_tokens=k)
                if k < n:
                    m["finish_reason"] = None
                chunk(("data: " + json.dumps({"text": "", "meta_info": m}) + "\n\n").encode())
            chunk(b"data: [DONE]\n\n"); chunk(b"")
        except Exception:
            pass
        finally:
            with lock:
                state["active"] -= 1
            slots.release()


class S(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    request_queue_size = 256


S(("127.0.0.1", PORT), H).serve_forever()
