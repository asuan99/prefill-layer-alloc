"""abs_mono_loadgen: trace_loadgen.replay + absolute CLOCK_MONOTONIC client stamps (diag Job item (v), 2026-10-28).

Pinned:
  * trace_loadgen.replay is called UNCHANGED (its record keys are a subset, its values untouched);
  * the anchor is exact: the first perf_counter read inside replay is `benchmark_started` (source pin), and the
    reconstructed token stamps equal the perf_counter values StreamState actually recorded (+ the one offset);
  * the shim is removed afterwards (trace_loadgen.time is the real module again);
  * a mutant that anchors on a perf_counter read taken BEFORE replay (the obvious shortcut) is detected.
CPU only: the fake SSE server of test_loadgen_outcome.
"""

import ast
import asyncio
import inspect
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import test_loadgen_outcome as LO  # noqa: E402
from pdmux_eval import abs_mono_loadgen as AM  # noqa: E402

TL = LO.TL
NEW_KEYS = {"t_sent_mono", "t_finished_mono", "token_t_mono"}


def _trace():
    return [{"request_id": f"ok{i}", "phase": "decode", "arrival_s": 0.01 * i, "input_tokens": 4,
             "output_tokens": 6} for i in range(3)] + [
        {"request_id": "err0", "phase": "decode", "arrival_s": 0.0, "input_tokens": 4, "output_tokens": 6}]


def _run(fn):
    seen = []  # one entry per classify() call; an id()-keyed dict collapses when CPython reuses ids of freed StreamState objects
    real_state = TL.StreamState

    class Spy(real_state):
        def classify(self, *a, **k):
            seen.append(list(self.token_times))
            return super().classify(*a, **k)

    with tempfile.TemporaryDirectory() as tmp, LO._FakeServer() as srv, mock.patch.object(TL, "StreamState", Spy):
        out = Path(tmp) / "requests.jsonl"
        summary = asyncio.run(fn(f"http://127.0.0.1:{srv.port}/generate", _trace(), out, 3000.0, 50.0, 0))
        recs = [json.loads(line) for line in out.read_text().splitlines()]
    return summary, recs, sorted(seen, key=lambda v: (len(v), v[:1]))


class SourcePinTest(unittest.TestCase):
    def test_first_perf_counter_in_replay_is_benchmark_started(self):
        src = inspect.getsource(TL.replay)
        tree = ast.parse(src.lstrip())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "perf_counter"]
        first = min(calls, key=lambda n: (n.lineno, n.col_offset))
        line = src.lstrip().splitlines()[first.lineno - 1].strip()
        self.assertEqual(line, "benchmark_started = time.perf_counter()")

    def test_loadgen_is_not_modified_by_the_wrapper(self):
        self.assertNotIn("abs_mono", Path(TL.__file__).read_text())


@unittest.skipUnless(LO.HAVE_AIOHTTP, "aiohttp not installed")
class ReplayTest(unittest.TestCase):
    def test_plain_replay_unchanged_and_wrapper_adds_exact_stamps(self):
        plain_summary, plain, _ = _run(TL.replay)
        summary, recs, actual = _run(AM.replay_abs_mono)
        self.assertIs(TL.time, time)                                   # shim removed
        self.assertEqual({k for r in plain for k in r} & NEW_KEYS, set())
        self.assertEqual(set(summary) - set(plain_summary), {"benchmark_started_mono", "abs_mono_offset_s"})
        self.assertLess(abs(summary["abs_mono_offset_s"]), 1e-3)
        off = summary["abs_mono_offset_s"]
        got = sorted(([t - off for t in r["token_t_mono"]] for r in recs), key=lambda v: (len(v), v[:1]))
        self.assertEqual(len(got), len(actual))
        for g, a in zip(got, actual):
            self.assertEqual(len(g), len(a))
            for x, y in zip(g, a):
                self.assertAlmostEqual(x, y, delta=1e-6)               # the very perf_counter stamps
        for r in recs:
            self.assertTrue(NEW_KEYS <= set(r))
            if r["token_t_mono"]:
                self.assertLessEqual(r["t_sent_mono"], r["token_t_mono"][0])
                self.assertLessEqual(r["token_t_mono"][-1], r["t_finished_mono"] + 1e-9)
            self.assertGreaterEqual(r["t_sent_mono"], summary["benchmark_started_mono"] - 1e-9)

    def test_mutant_anchor_before_replay_is_detected(self):
        # the shortcut: anchor on a perf_counter read taken before replay instead of replay's own first read
        async def mutant(endpoint, requests, output_path, a, b, c):
            t_before = time.perf_counter()
            time.sleep(0.05)
            s = await TL.replay(endpoint, requests, output_path, a, b, c)
            lines = []
            for line in Path(output_path).read_text().splitlines():
                rec = json.loads(line)
                rec.update(AM.abs_fields(rec, t_before, 0.0))
                lines.append(json.dumps(rec))
            Path(output_path).write_text("".join(x + "\n" for x in lines))
            s["abs_mono_offset_s"] = 0.0
            return s

        _s, recs, actual = _run(mutant)
        got = sorted((r["token_t_mono"] for r in recs), key=lambda v: (len(v), v[:1]))
        worst = max(abs(x - y) for g, a in zip(got, actual) for x, y in zip(g, a))
        self.assertGreater(worst, 1e-3)                                # the exactness check above would fail


if __name__ == "__main__":
    unittest.main()
