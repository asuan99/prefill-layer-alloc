"""trace_loadgen request-outcome classification + engine_bench_runner context pre-flight.

2026-10-16, M3 PREREG rev12 audit (VERDICT_m3_rev12_draft_r1_2026-10-15.md section 4, R-2/R-3),
shared by M3 and G0.  CPU only: fake SSE servers (aiohttp.web) and a fake venv; no GPU, no model.

What is pinned here:
  * an SSE `{"error": ...}` chunk (SGLang's post-acceptance refusal on a streaming request,
    http_server.generate_request.stream_results) is a SERVER-side error, never a token, and its
    message is kept;
  * an ignore_eos request that ends cleanly with fewer tokens than requested is OUTPUT_SHORT;
  * a normal stream is still a success with error "" (old consumers unchanged);
  * every error kind leaves `error` non-empty, so `not r["error"]` / mg_label.is_error exclude it;
  * the runner refuses (exit 5, ABORT_CONTEXT_TOO_SHORT) a trace that does not fit the context
    length it would launch with, before any server start;
  * mutation tests (lesson 53): "error chunk counted as a token" and "OUTPUT_SHORT check removed"
    must make this suite fail (KILLED); the unmutated control must pass.
"""

import asyncio
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE / "benchmarks"))
sys.path.insert(0, str(ENGINE / "results" / "multigpu"))

from pdmux_eval import context_limit as CL      # noqa: E402
from pdmux_eval import trace_loadgen as TL      # noqa: E402

LOADGEN_SRC = ENGINE / "benchmarks" / "pdmux_eval" / "trace_loadgen.py"
RUNNER = ENGINE / "scripts" / "r2_eval" / "engine_bench_runner.sh"

try:
    import aiohttp  # noqa: F401
    from aiohttp import web
    HAVE_AIOHTTP = True
except ImportError:                                # pragma: no cover
    HAVE_AIOHTTP = False


def sse(obj):
    return b"data: " + (obj if isinstance(obj, bytes) else json.dumps(obj).encode()) + b"\n"


def token_chunk(k, finish=None):
    return sse({"text": "t" * k, "meta_info": {"completion_tokens": k, "finish_reason": finish}})


# Fake SGLang refusal, byte-shaped like http_server.stream_results.
REFUSAL_MSG = ("Requested token count exceeds the model's maximum context length of 16384 tokens. "
               "You requested a total of 16400 tokens: 16000 tokens from the input messages and "
               "400 tokens for the completion.")
SSE_ERROR_LINES = [sse({"error": {"message": REFUSAL_MSG}}), b"\n", sse(b"[DONE]")]


def feed(mod, lines):
    st = mod.StreamState()
    t = [0.0]

    def clock():
        t[0] += 0.001
        return t[0]
    for raw in lines:
        st.feed_line(raw, clock)
    return st


# --------------------------------------------------------------------------- core checks
# Written as plain functions over a loadgen MODULE so the same checks run on the real module
# and on mutants (TestMutantsKilled).  Each returns None on pass, a string on failure.

def check_sse_error(mod):
    st = feed(mod, SSE_ERROR_LINES)
    err, kind, side = st.classify(400, True)
    if len(st.token_times) != 0:
        return f"error chunk counted as {len(st.token_times)} token(s)"
    if kind != "SSE_ERROR" or side != "server":
        return f"kind/side {kind!r}/{side!r}"
    if REFUSAL_MSG not in err:
        return "server message not preserved"
    return None


def check_output_short(mod):
    lines = [token_chunk(k) for k in (1, 2, 3)] + [sse(b"[DONE]")]
    st = feed(mod, lines)
    err, kind, side = st.classify(5, True)
    if len(st.token_times) != 3:
        return f"observed {len(st.token_times)}"
    if kind != "OUTPUT_SHORT" or side != "server" or not err:
        return f"kind/side/err {kind!r}/{side!r}/{err!r}"
    return None


def check_normal(mod):
    lines = [token_chunk(k) for k in range(1, 6)]
    lines[-1] = token_chunk(5, {"type": "length", "length": 5})
    st = feed(mod, lines + [sse(b"[DONE]")])
    res = st.classify(5, True)
    if res != ("", "", ""):
        return f"normal stream classified {res!r}"
    if len(st.token_times) != 5:
        return f"observed {len(st.token_times)}"
    return None


CORE_CHECKS = (check_sse_error, check_output_short, check_normal)


def run_core(mod):
    return [f"{c.__name__}: {r}" for c in CORE_CHECKS if (r := c(mod)) is not None]


class TestStreamStateClassification(unittest.TestCase):
    def test_core_checks_pass(self):
        self.assertEqual(run_core(TL), [])

    def test_sse_error_with_empty_message_still_an_error(self):
        st = feed(TL, [sse({"error": {"message": ""}}), sse(b"[DONE]")])
        err, kind, side = st.classify(10, True)
        self.assertEqual((kind, side), ("SSE_ERROR", "server"))
        self.assertTrue(err)

    def test_sse_error_after_tokens_wins_over_short(self):
        st = feed(TL, [token_chunk(1), token_chunk(2), sse({"error": "boom"}), sse(b"[DONE]")])
        err, kind, _ = st.classify(8, True)
        self.assertEqual(kind, "SSE_ERROR")
        self.assertIn("boom", err)
        self.assertEqual(len(st.token_times), 2)

    def test_server_abort_finish_reason(self):
        st = feed(TL, [token_chunk(1, {"type": "abort", "message": "Input length (16380 tokens) exceeds",
                                       "status_code": 400}), sse(b"[DONE]")])
        err, kind, side = st.classify(1, True)       # 1 token observed == requested, still an abort
        self.assertEqual((kind, side), ("SERVER_ABORT", "server"))
        self.assertIn("Input length", err)

    def test_abort_with_zero_tokens(self):
        line = sse({"text": "", "meta_info": {"completion_tokens": 0,
                                              "finish_reason": {"type": "abort", "message": "x"}}})
        st = feed(TL, [line, sse(b"[DONE]")])
        self.assertEqual(len(st.token_times), 0)
        self.assertEqual(st.classify(4, True)[1], "SERVER_ABORT")

    def test_output_long(self):
        st = feed(TL, [token_chunk(1), token_chunk(3)])
        self.assertEqual(st.classify(2, True)[1], "OUTPUT_LONG")

    def test_short_is_not_flagged_without_ignore_eos(self):
        st = feed(TL, [token_chunk(1), token_chunk(2)])
        self.assertEqual(st.classify(5, False), ("", "", ""))

    def test_client_exception_precedence(self):
        st = feed(TL, SSE_ERROR_LINES)
        self.assertEqual(st.classify(4, True, client_error="TimeoutError()")[1:],
                         ("CLIENT_EXCEPTION", "client"))
        self.assertEqual(st.classify(4, True, client_error="ClientResponseError(503)", http_status=503)[1:],
                         ("HTTP_STATUS", "server"))

    def test_every_kind_has_a_side_and_nonempty_error(self):
        for kind, side in TL.SIDE_OF_KIND.items():
            self.assertEqual(bool(kind), bool(side), kind)

    def test_chunk_without_meta_info_keeps_legacy_count(self):
        # Legacy behaviour for a non-error chunk without meta_info is unchanged (+1 token).
        st = feed(TL, [sse({"text": "a"}), sse({"text": "ab"})])
        self.assertEqual(len(st.token_times), 2)


# --------------------------------------------------------------------------- replay() end to end
def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class _FakeServer:
    """SSE /generate whose behaviour is chosen by the request id prefix."""

    def __init__(self):
        self.port = _free_port()
        self.loop = asyncio.new_event_loop()
        self.ready = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    async def generate(self, req):
        body = await req.json()
        rid, n = body["rid"], int(body["sampling_params"]["max_new_tokens"])
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(req)
        if rid.startswith("err"):
            for line in SSE_ERROR_LINES:
                await resp.write(line)
        else:
            m = max(1, n - 2) if rid.startswith("short") else n
            for k in range(1, m + 1):
                fin = {"type": "length", "length": m} if k == m else None
                await resp.write(token_chunk(k, fin) + b"\n")
                await asyncio.sleep(0.001)
            await resp.write(sse(b"[DONE]") + b"\n")
        await resp.write_eof()
        return resp

    def _run(self):
        asyncio.set_event_loop(self.loop)
        app = web.Application()
        app.router.add_post("/generate", self.generate)
        self.runner = web.AppRunner(app, access_log=None)
        self.loop.run_until_complete(self.runner.setup())
        site = web.TCPSite(self.runner, "127.0.0.1", self.port)
        self.loop.run_until_complete(site.start())
        self.ready.set()
        self.loop.run_forever()

    def __enter__(self):
        self.thread.start()
        self.ready.wait(10)
        return self

    def __exit__(self, *a):
        async def stop():
            await self.runner.cleanup()
        asyncio.run_coroutine_threadsafe(stop(), self.loop).result(10)
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(10)


@unittest.skipUnless(HAVE_AIOHTTP, "aiohttp not installed")
class TestReplayEndToEnd(unittest.TestCase):
    def test_records_and_summary(self):
        trace = [{"request_id": rid, "phase": "decode", "arrival_s": 0.01 * i,
                  "input_tokens": 4, "output_tokens": 6}
                 for i, rid in enumerate(("ok0", "ok1", "err0", "short0"))]
        with tempfile.TemporaryDirectory() as tmp, _FakeServer() as srv:
            out = Path(tmp) / "requests.jsonl"
            summary = asyncio.run(TL.replay(f"http://127.0.0.1:{srv.port}/generate", trace, out,
                                            3000.0, 50.0, 0))
            recs = {r["request_id"]: r for r in map(json.loads, out.read_text().splitlines())}
        for rid in ("ok0", "ok1"):
            r = recs[rid]
            self.assertEqual((r["error"], r["error_kind"], r["error_side"]), ("", "", ""), rid)
            self.assertEqual(r["observed_output_tokens"], 6)
            self.assertIsNotNone(r["ttft_ms"])
        e = recs["err0"]
        self.assertEqual((e["error_kind"], e["error_side"]), ("SSE_ERROR", "server"))
        self.assertIn(REFUSAL_MSG, e["error"])
        self.assertEqual(e["observed_output_tokens"], 0)
        self.assertIsNone(e["ttft_ms"])
        s = recs["short0"]
        self.assertEqual((s["error_kind"], s["error_side"]), ("OUTPUT_SHORT", "server"))
        self.assertEqual(s["observed_output_tokens"], 4)
        self.assertEqual(summary["requests"], 2.0)            # successes only
        self.assertEqual(summary["failed_requests"], 2.0)
        self.assertEqual(summary["server_error_requests"], 2.0)
        self.assertEqual(summary["client_error_requests"], 0.0)
        # existing consumers see the two as errors with no code change
        import mg_label
        self.assertEqual(sorted(rid for rid, r in recs.items() if mg_label.is_error(r)),
                         ["err0", "short0"])
        from pdmux_eval import analyze_concurrent  # noqa: F401  (import only: module unchanged)


# --------------------------------------------------------------------------- mutation tests
MUTANTS = {
    # (lesson 53) the pre-2026-10-16 behaviour: an error chunk falls through to the
    # "no meta_info -> last + 1" token count.
    "error_chunk_counted_as_token": (
        '        if isinstance(chunk, dict) and "error" in chunk:\n',
        '        if False and isinstance(chunk, dict) and "error" in chunk:\n',
    ),
    "output_short_check_removed": (
        "        if ignore_eos and observed != int(requested_output_tokens):\n",
        "        if False:\n",
    ),
}


def load_variant(name, src):
    mod = types.ModuleType(f"pdmux_eval._mut_{name}")
    mod.__package__ = "pdmux_eval"
    mod.__file__ = f"<mutant {name}>"
    sys.modules[mod.__name__] = mod
    try:
        exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    finally:
        del sys.modules[mod.__name__]
    return mod


class TestMutantsKilled(unittest.TestCase):
    def test_control_unmutated_survives(self):
        self.assertEqual(run_core(load_variant("control", LOADGEN_SRC.read_text())), [])

    def test_each_mutant_is_killed(self):
        src = LOADGEN_SRC.read_text()
        for name, (old, new) in MUTANTS.items():
            with self.subTest(mutant=name):
                self.assertEqual(src.count(old), 1, f"mutation site for {name} not unique/absent")
                failures = run_core(load_variant(name, src.replace(old, new, 1)))
                self.assertTrue(failures, f"mutant {name} SURVIVED")


# --------------------------------------------------------------------------- context pre-flight
def write_trace(path, pairs):
    with open(path, "w") as f:
        f.write(json.dumps({"record": "header"}) + "\n")
        for i, (inp, out) in enumerate(pairs):
            f.write(json.dumps({"record": "request", "request_id": f"r{i}", "phase": "decode",
                                "arrival_s": float(i), "input_tokens": inp, "output_tokens": out}) + "\n")


class TestFitPredicate(unittest.TestCase):
    def test_boundaries_at_16384(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "t.jsonl"
            cases = {(16000, 382): True,     # total 16382 = ctx-2: fits
                     (16000, 383): False,    # total 16383 = ctx-1: scheduler clamps by one
                     (16000, 384): False,    # total 16384 = ctx: tokenizer refuses (SSE error)
                     (16377, 1): True,       # input = ctx-7
                     (16378, 1): False}      # input = ctx-6: scheduler input bound
            for (inp, out), ok in cases.items():
                with self.subTest(inp=inp, out=out):
                    write_trace(p, [(100, 10), (inp, out)])
                    self.assertEqual(not CL.fit_problems(16384, [p]), ok)

    def test_missing_and_empty_traces_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            empty = Path(tmp) / "e.jsonl"
            write_trace(empty, [])
            self.assertTrue(CL.fit_problems(16384, [Path(tmp) / "none.jsonl"]))
            self.assertTrue(CL.fit_problems(16384, [empty]))
            self.assertTrue(CL.fit_problems(16384, []))

    def test_cli_fit_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "t.jsonl"
            write_trace(p, [(16000, 600)])
            env = dict(os.environ, PYTHONPATH=str(ENGINE / "benchmarks"))
            bad = subprocess.run([sys.executable, "-m", "pdmux_eval.context_limit", "--fit-only",
                                  "--context-length", "16384", "--trace", str(p)],
                                 capture_output=True, text=True, env=env)
            self.assertEqual(bad.returncode, 2)
            self.assertIn("ABORT_CONTEXT_TOO_SHORT", bad.stderr)
            self.assertIn("need context_length >= 16602", bad.stderr)
            good = subprocess.run([sys.executable, "-m", "pdmux_eval.context_limit", "--fit-only",
                                   "--context-length", "20480", "--trace", str(p)],
                                  capture_output=True, text=True, env=env)
            self.assertEqual((good.returncode, good.stdout.strip()), (0, "20480"))


PREFLIGHT_BLOCK_START = "# ---- context-length pre-flight"


def _runner_env(tmp, ctx):
    venv = Path(tmp) / "venv" / "bin"
    venv.mkdir(parents=True)
    (venv / "activate").write_text("# fake venv (test)\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PDMUX_", "SLURM_"))}
    env.pop("PYTHONPATH", None)
    env.update(PDMUX_VENV=str(Path(tmp) / "venv"), PDMUX_CONTEXT_LENGTH=str(ctx),
               PDMUX_ENGINE_MODE="fused", PDMUX_DRY_RUN="0",
               SGLANG_ENGINE_DEV=str(Path(tmp) / "no_engine_tree"))
    # `python` must exist for the runner; point it at this interpreter.
    shim = Path(tmp) / "shim"
    shim.mkdir()
    (shim / "python").symlink_to(sys.executable)
    env["PATH"] = f"{shim}:{env.get('PATH', '/usr/bin:/bin')}"
    return env


def run_runner(runner, tmp, pairs, ctx):
    trace = Path(tmp) / "trace.jsonl"
    write_trace(trace, pairs)
    rec = Path(tmp) / "rec.json"
    rec.write_text(json.dumps({"trace_path": str(trace), "server_seed": 1, "max_running_requests": 48,
                               "ttft_slo_ms": 3000, "itl_slo_ms": 50, "workload": "w"}))
    run_dir = Path(tmp) / "run"
    p = subprocess.run(["bash", str(runner), str(rec), str(run_dir)], capture_output=True, text=True,
                       env=_runner_env(tmp, ctx), timeout=120)
    return p, run_dir


class TestRunnerPreflight(unittest.TestCase):
    def test_refuses_before_boot(self):
        with tempfile.TemporaryDirectory() as tmp:
            p, run_dir = run_runner(RUNNER, tmp, [(100, 10), (16000, 384)], 16384)
            self.assertEqual(p.returncode, 5, p.stderr[-2000:])
            self.assertIn("ABORT_CONTEXT_TOO_SHORT", p.stderr)
            self.assertFalse((run_dir / "server.log").exists(), "server was launched")

    def test_fitting_trace_reaches_the_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            p, run_dir = run_runner(RUNNER, tmp, [(100, 10), (16000, 384)], 20480)
            self.assertNotIn("ABORT_CONTEXT_TOO_SHORT", p.stderr)
            self.assertTrue((run_dir / "server.log").exists(), p.stderr[-2000:])   # launch attempted
            self.assertNotEqual(p.returncode, 0)                                    # no sglang here

    def test_mutant_preflight_removed_is_killed(self):
        src = RUNNER.read_text()
        start = src.index(PREFLIGHT_BLOCK_START)
        end = src.index("\ncleanup() {", start)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "eng"
            (root / "scripts" / "r2_eval").mkdir(parents=True)
            (root / "benchmarks").symlink_to(ENGINE / "benchmarks")
            (root / "scripts" / "r2_eval" / "multigpu_cell_meta.py").symlink_to(
                ENGINE / "scripts" / "r2_eval" / "multigpu_cell_meta.py")
            for name, text in (("control", src), ("mutant", src[:start] + src[end:])):
                mut = root / "scripts" / "r2_eval" / "engine_bench_runner.sh"
                mut.write_text(text)
                with tempfile.TemporaryDirectory() as t2:
                    p, run_dir = run_runner(mut, t2, [(16000, 384)], 16384)
                refused = p.returncode == 5 and "ABORT_CONTEXT_TOO_SHORT" in p.stderr
                self.assertEqual(refused, name == "control", f"{name}: rc={p.returncode}")


if __name__ == "__main__":
    unittest.main()
