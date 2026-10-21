"""CPU gate for the I-4 (decode forward GPU time) + I-6 (cudagraph padding) probe.

What is pinned here (NO GPU -- nothing here says the probe is neutral, that ON/OFF
outputs match, or that CUDA events on a green-context stream time correctly;
those are GPU gates G-alpha / G-gamma, results/engine_gate/GATE_SPLIT_DESIGN_2026-10-20.md):

  A. ``expected_padded_bs`` / ``graph_fields`` against HAND-WRITTEN tables.
  B. The probe on a fake GPU timeline: I-4 is the event interval, not host time;
     resolution is FIFO and NON-BLOCKING (fake ``synchronize`` raises); pending
     overflow is counted; mode filter; capture guard; events-off mode; engine
     exceptions propagate; probe failures never break the forward; threads.
  C. Mutation (lesson 53): source-level mutants of decode_probe.py -- host-time
     I-4, start event after the forward, blocking resolution, bs_padded := bs_real,
     graph hit from bs <= max_bs, mixed batches dropped -- each FAILS the scenario
     table the unmutated module passes.
  D. Lockstep neutrality, static: the module calls no synchronize / wait_event /
     wait_stream, records events only with ``record()`` (current stream), and the
     scanner itself catches an injected ``synchronize``.
  E. Default OFF, env parsing, no torch/sglang import at module scope, schema (d),
     JSONL through the async writer.
  F. Arm symmetry + upstream assumptions, against the INSTALLED runtime source:
     every event loop reaches ``ModelRunner.forward`` via
     ``forward_batch_generation``; ``can_run_graph``, ``graph_runner``,
     ``capture_bs``/``max_bs``/``disable_padding`` and the ``bisect_left`` padding rule
     exist as the probe assumes.
  G. The scheduler hook patch: applies after the admission hook, reverses exactly,
     lands in ``run_scheduler_process``, moves no Scheduler class line; sync wiring
     and manifest entry 30.
"""

import ast
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ENG = Path(__file__).resolve().parents[1]
PROBE_PY = ENG / "src" / "multiplex" / "decode_probe.py"
PATCH = ENG / "src" / "patches" / "decode_probe_scheduler_hook.patch"
ADM_PATCH = ENG / "src" / "patches" / "admission_probe_scheduler_hook.patch"
SYNC = ENG / "scripts" / "bootstrap" / "sync_engine_tree.sh"
RUNTIME = Path(
    os.environ.get("SGLANG_ENGINE_DEV")
    or os.path.join(
        os.environ.get("PDMUX_ROOT", str(Path(__file__).resolve().parents[4])),
        "sglang_engine_dev",
        "python",
    )
)
SRT = RUNTIME / "sglang" / "srt"
sys.modules.pop("profile", None)

SRC = PROBE_PY.read_text(encoding="utf-8")


def _load(source=SRC, name="decode_probe_under_test"):
    mod = types.ModuleType(name)
    mod.__file__ = str(PROBE_PY)
    sys.modules[name] = mod  # dataclasses resolve annotations via sys.modules
    exec(compile(source, str(PROBE_PY), "exec"), mod.__dict__)
    return mod


dp = _load()


def mutate(old, new):
    assert SRC.count(old) == 1, f"mutation target not unique/absent: {old!r}"
    return _load(SRC.replace(old, new), name=f"decode_probe_mutant_{abs(hash(old)) % 10**8}")


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------

class FakeGPU:
    """One in-order stream.  ``enqueue(ms)`` adds GPU work; ``complete_until(t)``
    advances the completion frontier.  Events time-stamp at the enqueue cursor."""

    def __init__(self):
        self.cursor = 0.0
        self.done = 0.0

    def enqueue(self, ms):
        self.cursor += ms

    def complete_all(self):
        self.done = self.cursor


class BlockingCall(AssertionError):
    pass


class FakeEvent:
    def __init__(self, gpu):
        self.gpu = gpu
        self.ts = None

    def record(self, *args, **kwargs):
        if args or kwargs:
            raise AssertionError("event recorded on an explicit stream")
        self.ts = self.gpu.cursor

    def query(self):
        return self.ts is not None and self.gpu.done >= self.ts

    def elapsed_time(self, other):
        if not (self.query() and other.query()):
            raise RuntimeError("elapsed_time on an incomplete event")
        return other.ts - self.ts

    def synchronize(self):
        raise BlockingCall("probe blocked on an event")

    def wait(self, *a):
        raise BlockingCall("probe issued a cross-stream wait")


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


class FM:
    def __init__(self, name):
        self.name = name.upper()

    def is_decode(self):
        return self.name == "DECODE"

    def is_mixed(self):
        return self.name == "MIXED"


CAPTURE = [1, 2, 4, 8, 12, 16, 24, 32, 40, 48]


class FakeGraphRunner:
    def __init__(self, capture_bs=CAPTURE, disable_padding=False):
        self.capture_bs = list(capture_bs)
        self.max_bs = max(capture_bs)
        self.disable_padding = disable_padding
        self.bs = None


class FakeRunner:
    """Mimics ModelRunner.forward: the engine decides can_run_graph; replay sets
    graph_runner.bs by bisect (the real replay_prepare rule)."""

    def __init__(self, gpu, clock, graph_runner=None, gpu_ms=20.0, host_ms=1.5, refuse_graph=False):
        self.gpu, self.clock = gpu, clock
        self.graph_runner = graph_runner
        self.gpu_ms, self.host_ms = gpu_ms, host_ms
        self.refuse_graph = refuse_graph
        self.forward_pass_id = 0
        self.raise_next = False

    def forward(self, forward_batch, skip_attn_backend_init=False, pp_proxy_tensors=None,
                reinit_attn_backend=False, split_forward_count=1):
        self.forward_pass_id += 1
        if self.raise_next:
            self.raise_next = False
            raise ValueError("engine failure")
        gr = self.graph_runner
        can = bool(forward_batch.forward_mode.is_decode() and gr is not None and not self.refuse_graph
                   and forward_batch.batch_size <= gr.max_bs)
        if can:
            gr.bs = next(b for b in sorted(gr.capture_bs) if b >= forward_batch.batch_size)
        self.gpu.enqueue(self.gpu_ms)
        self.clock.t += self.host_ms / 1000.0
        return SimpleNamespace(logits_output="L", can_run_graph=can)


def FB(bs, mode="decode"):
    return SimpleNamespace(batch_size=bs, forward_mode=FM(mode))


def make(mod=dp, graph_runner="default", events=True, capturing=False, max_pending=4096,
         gpu_ms=20.0, host_ms=1.5, refuse_graph=False, stream_idx_fn=None):
    gpu, clock = FakeGPU(), Clock()
    gr = FakeGraphRunner() if graph_runner == "default" else graph_runner
    runner = FakeRunner(gpu, clock, gr, gpu_ms=gpu_ms, host_ms=host_ms, refuse_graph=refuse_graph)
    boot = {"capture_bs": list(gr.capture_bs) if gr else [],
            "disable_cuda_graph_padding": bool(gr.disable_padding) if gr else False}
    probe = mod.DecodeProbe(None, window_s=1e9, events_enabled=events, boot=boot, clock=clock,
                            event_factory=(lambda: FakeEvent(gpu)) if events else None,
                            capturing_fn=lambda: capturing, stream_idx_fn=stream_idx_fn,
                            max_pending=max_pending)
    mod.set_probe(probe)
    mod.install_on_runner(runner)
    return SimpleNamespace(gpu=gpu, clock=clock, runner=runner, probe=probe, mod=mod)


def fwd_records(probe):
    return [r for r in probe.records if r["event"] == "decode_forward"]


# ---------------------------------------------------------------------------
# A. pure tables
# ---------------------------------------------------------------------------

class TestPureTables(unittest.TestCase):
    def test_expected_padded_bs(self):
        f = dp.expected_padded_bs
        cases = [  # (capture, raw, disable_padding) -> expected, written by hand
            (CAPTURE, 1, False, 1), (CAPTURE, 3, False, 4), (CAPTURE, 5, False, 8),
            (CAPTURE, 8, False, 8), (CAPTURE, 9, False, 12), (CAPTURE, 33, False, 40),
            (CAPTURE, 48, False, 48), (CAPTURE, 49, False, None), (CAPTURE, 0, False, None),
            ([48, 1, 8], 5, False, 8),  # unsorted input
            (CAPTURE, 5, True, None), (CAPTURE, 8, True, 8), ([], 4, False, None),
        ]
        for cb, raw, dpad, exp in cases:
            self.assertEqual(f(cb, raw, dpad), exp, (cb, raw, dpad))

    def test_graph_fields(self):
        g = dp.graph_fields
        r = g("decode", True, 5, True, 8, CAPTURE, False)
        self.assertEqual((r["graph"], r["bs_padded"], r["bs_pad_expected"], r["pad_check"], r["eager_reason"]),
                         ("hit", 8, 8, "ok", None))
        r = g("decode", True, 5, True, 12, CAPTURE, False)  # engine padded differently than predicted
        self.assertEqual((r["bs_padded"], r["pad_check"]), (12, "mismatch"))
        r = g("decode", False, 5, False, None, [], False)
        self.assertEqual((r["graph"], r["bs_padded"], r["eager_reason"], r["pad_check"]),
                         ("eager", 5, "graph_runner_absent", "na"))
        r = g("decode", False, 60, True, 48, CAPTURE, False)
        self.assertEqual((r["bs_padded"], r["eager_reason"]), (60, "bs_over_max"))
        r = g("decode", False, 5, True, None, CAPTURE, False)
        self.assertEqual(r["eager_reason"], "can_run_false")
        r = g("mixed", False, 5, True, None, CAPTURE, False)
        self.assertEqual(r["eager_reason"], "mode_not_capturable")

    def test_forward_mode_kind(self):
        self.assertEqual(dp.forward_mode_kind(FM("decode")), "decode")
        self.assertEqual(dp.forward_mode_kind(FM("mixed")), "mixed")
        for m in ("extend", "split_prefill", "idle", "target_verify"):
            self.assertIsNone(dp.forward_mode_kind(FM(m)), m)
        self.assertIsNone(dp.forward_mode_kind(None))


# ---------------------------------------------------------------------------
# B + C. scenario table (unmutated passes; every mutant fails)
# ---------------------------------------------------------------------------

def scenario_failures(mod):
    """Run the registered scenarios on ``mod``; return a list of failure strings."""
    fails = []

    def expect(cond, msg):
        if not cond:
            fails.append(msg)

    try:
        # S1: I-4 = event interval (20 ms GPU) not host time (1.5 ms); resolved non-blocking, FIFO
        s = make(mod)
        s.runner.forward(FB(5))
        expect(fwd_records(s.probe) == [], "S1 record emitted before GPU completion")
        s.gpu.complete_all()
        s.runner.forward(FB(9))           # resolves the first at entry, without blocking
        recs = fwd_records(s.probe)
        expect(len(recs) == 1, f"S1 expected 1 resolved record, got {len(recs)}")
        if recs:
            r = recs[0]
            expect(abs(r["decode_fwd_gpu_ms"] - 20.0) < 1e-9, f"S1 gpu_ms {r['decode_fwd_gpu_ms']}")
            expect(abs(r["decode_fwd_host_ms"] - 1.5) < 1e-6, f"S1 host_ms {r['decode_fwd_host_ms']}")
            expect((r["bs_real"], r["bs_padded"], r["graph"], r["pad_check"]) == (5, 8, "hit", "ok"),
                   f"S1 I-6 fields {r}")
        s.gpu.complete_all()
        s.probe.maybe_flush(force=True)
        recs = fwd_records(s.probe)
        expect([r["bs_padded"] for r in recs] == [8, 12], f"S1 padded sequence {[r['bs_padded'] for r in recs]}")
        expect([r["seq"] for r in recs] == [1, 2], "S1 FIFO order")

        # S2: --disable-cuda-graph (no graph runner) -> eager, unpadded, reason graph_runner_absent
        s = make(mod, graph_runner=None)
        s.runner.forward(FB(5))
        s.gpu.complete_all()
        s.probe.maybe_flush(force=True)
        r = fwd_records(s.probe)
        expect(len(r) == 1 and (r[0]["graph"], r[0]["bs_padded"], r[0]["eager_reason"]) ==
               ("eager", 5, "graph_runner_absent"), f"S2 {r}")

        # S3: graph runner present but the engine refuses the graph (bs <= max): eager can_run_false
        s = make(mod, refuse_graph=True)
        s.runner.forward(FB(5))
        s.gpu.complete_all()
        s.probe.maybe_flush(force=True)
        r = fwd_records(s.probe)
        expect(len(r) == 1 and (r[0]["graph"], r[0]["bs_padded"], r[0]["eager_reason"]) ==
               ("eager", 5, "can_run_false"), f"S3 {r}")

        # S4: bs above the capture list -> eager bs_over_max
        s = make(mod)
        s.runner.forward(FB(60))
        s.gpu.complete_all()
        s.probe.maybe_flush(force=True)
        r = fwd_records(s.probe)
        expect(len(r) == 1 and (r[0]["graph"], r[0]["eager_reason"], r[0]["bs_padded"]) ==
               ("eager", "bs_over_max", 60), f"S4 {r}")

        # S5: mixed batch recorded as mode mixed; extend / split_prefill / idle pass through, counted
        s = make(mod)
        s.runner.forward(FB(6, "mixed"))
        for m in ("extend", "split_prefill", "idle"):
            s.runner.forward(FB(3, m))
        s.gpu.complete_all()
        s.probe.maybe_flush(force=True)
        r = fwd_records(s.probe)
        expect(len(r) == 1 and r[0]["mode"] == "mixed" and r[0]["eager_reason"] == "mode_not_capturable",
               f"S5 {r}")
        win = [x for x in s.probe.records if x["event"] == "decode_probe_window"][-1]
        expect(win["n_other_modes"] == {"extend": 1, "split_prefill": 1, "idle": 1}, f"S5 window {win}")
        expect(win["n_mixed"] == 1 and win["n_decode"] == 0, f"S5 counts {win}")

        # S6: never blocks -- incomplete GPU work stays pending, overflow is dropped and counted
        s = make(mod, max_pending=2)
        for _ in range(3):
            s.runner.forward(FB(4))
        s.probe.maybe_flush(force=True)
        win = [x for x in s.probe.records if x["event"] == "decode_probe_window"][-1]
        expect(fwd_records(s.probe) == [], "S6 resolved without completion")
        expect((win["n_pending"], win["n_dropped_pending"]) == (2, 1), f"S6 window {win}")

        # S7: graph capture in progress -> passthrough, counted, no event recorded
        s = make(mod, capturing=True)
        s.runner.forward(FB(4))
        s.probe.maybe_flush(force=True)
        win = [x for x in s.probe.records if x["event"] == "decode_probe_window"][-1]
        expect(win["n_skipped_capturing"] == 1 and fwd_records(s.probe) == [], f"S7 {win}")
    except BlockingCall as exc:
        fails.append(f"blocking call: {exc}")
    finally:
        mod.set_probe(None)
        mod.uninstall_all()
    return fails


class TestScenarios(unittest.TestCase):
    def test_unmutated_passes(self):
        self.assertEqual(scenario_failures(dp), [])


class TestMutants(unittest.TestCase):
    MUTANTS = {
        "host_time_i4": ("item.fields[I4_FIELD] = gpu_ms",
                         "item.fields[I4_FIELD] = item.fields[I4_HOST_FIELD]"),
        "start_after_forward": ("        if pair is not None:\n            pair[0].record()\n        out = orig(*args, **kwargs)\n",
                                "        out = orig(*args, **kwargs)\n        if pair is not None:\n            pair[0].record()\n"),
        "blocking_resolution": ("        if not head.end.query():\n            break\n",
                                "        head.end.synchronize()\n"),
        "pad_equals_real": ('            "bs_padded": padded,\n', '            "bs_padded": int(bs_real),\n'),
        "hit_from_max_bs": ("    if can_run_graph:\n        exp = expected_padded_bs",
                            "    if runner_present and capture_bs and int(bs_real) <= max(capture_bs):\n        exp = expected_padded_bs"),
        "mixed_dropped": ("    if _truth(forward_mode, \"is_mixed\"):\n        return MODE_MIXED\n", ""),
        "eager_padded_to_capture": ('        "bs_padded": int(bs_real),\n        "bs_pad_expected": int(bs_real),\n',
                                    '        "bs_padded": expected_padded_bs(capture_bs, bs_real, disable_padding) or int(bs_real),\n        "bs_pad_expected": int(bs_real),\n'),
    }

    def test_every_mutant_fails(self):
        for name, (old, new) in self.MUTANTS.items():
            with self.subTest(mutant=name):
                fails = scenario_failures(mutate(old, new))
                self.assertTrue(fails, f"mutant {name} survived the scenario table")


class TestBehaviour(unittest.TestCase):
    def tearDown(self):
        dp.set_probe(None)
        dp.uninstall_all()

    def test_events_off_records_i6_immediately(self):
        s = make(events=False)
        s.runner.forward(FB(5))
        r = fwd_records(s.probe)
        self.assertEqual(len(r), 1)
        self.assertIsNone(r[0]["decode_fwd_gpu_ms"])
        self.assertEqual(r[0]["bs_padded"], 8)
        self.assertFalse(s.probe.events_enabled)

    def test_engine_exception_propagates_and_probe_survives(self):
        s = make()
        s.runner.raise_next = True
        with self.assertRaises(ValueError):
            s.runner.forward(FB(4))
        self.assertFalse(s.probe.disabled)
        s.runner.forward(FB(4))
        s.gpu.complete_all()
        s.probe.maybe_flush(force=True)
        self.assertEqual(len(fwd_records(s.probe)), 1)

    def test_probe_failure_never_breaks_forward(self):
        def boom():
            raise KeyError("stream idx")
        s = make(stream_idx_fn=boom)
        out = s.runner.forward(FB(4))
        self.assertEqual(out.logits_output, "L")
        self.assertTrue(s.probe.disabled)
        out = s.runner.forward(FB(4))  # disabled -> pure passthrough
        self.assertEqual(out.logits_output, "L")

    def test_off_probe_none_is_passthrough(self):
        s = make()
        dp.set_probe(None)
        s.runner.forward(FB(4))
        self.assertEqual(fwd_records(s.probe), [])

    def test_uninstall_restores_class_method(self):
        s = make()
        self.assertIn("forward", s.runner.__dict__)
        dp.uninstall_all()
        self.assertNotIn("forward", s.runner.__dict__)
        self.assertIs(type(s.runner).forward, FakeRunner.forward)

    def test_instance_level_only(self):
        s = make()
        other = FakeRunner(s.gpu, s.clock, FakeGraphRunner())
        other.forward(FB(4))  # same class, not instrumented
        s.gpu.complete_all()
        s.probe.maybe_flush(force=True)
        self.assertEqual(fwd_records(s.probe), [])

    def test_threads(self):
        s = make(gpu_ms=1.0, host_ms=0.0)
        s.gpu.done = float("inf")
        def work():
            for _ in range(200):
                s.runner.forward(FB(4))
        ts = [threading.Thread(target=work) for _ in range(2)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        s.probe.maybe_flush(force=True)
        seqs = [r["seq"] for r in fwd_records(s.probe)]
        self.assertEqual(sorted(seqs), list(range(1, 401)))

    def test_stream_idx_recorded(self):
        s = make(stream_idx_fn=lambda: 3)
        s.runner.forward(FB(4))
        s.gpu.complete_all()
        s.probe.maybe_flush(force=True)
        self.assertEqual(fwd_records(s.probe)[0]["pdmux_stream_idx"], 3)


# ---------------------------------------------------------------------------
# D. static lockstep neutrality
# ---------------------------------------------------------------------------

FORBIDDEN_ATTRS = {"synchronize", "wait_event", "wait_stream", "wait"}


def neutrality_violations(source):
    v = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in FORBIDDEN_ATTRS:
                v.append(f"line {node.lineno}: .{node.func.attr}()")
            if node.func.attr == "record" and (node.args or node.keywords):
                v.append(f"line {node.lineno}: record() with an explicit stream")
    return v


class TestNeutralityStatic(unittest.TestCase):
    def test_module_is_clean(self):
        self.assertEqual(neutrality_violations(SRC), [])

    def test_scanner_catches_injected_sync(self):
        bad = SRC.replace("            pair[1].record()\n",
                          "            pair[1].record()\n            pair[1].synchronize()\n", 1)
        self.assertNotEqual(bad, SRC)
        self.assertTrue(neutrality_violations(bad))
        bad2 = SRC.replace("            pair[0].record()\n", "            pair[0].record(stream)\n", 1)
        self.assertTrue(neutrality_violations(bad2))

    def test_no_torch_or_sglang_at_module_scope(self):
        tree = ast.parse(SRC)
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
                for n in names:
                    self.assertFalse(n.startswith(("torch", "sglang")), n)


# ---------------------------------------------------------------------------
# E. default OFF, env, schema, async writer
# ---------------------------------------------------------------------------

class TestDefaultOffAndSchema(unittest.TestCase):
    def tearDown(self):
        dp.set_probe(None)
        dp.uninstall_all()

    def test_unset_installs_nothing(self):
        runner = FakeRunner(FakeGPU(), Clock(), FakeGraphRunner())
        sched = SimpleNamespace(tp_worker=SimpleNamespace(model_runner=runner))
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PDMUX_DECODE_PROBE_PATH", None)
            self.assertIsNone(dp.maybe_install_decode_probe(sched))
        self.assertNotIn("forward", runner.__dict__)
        self.assertIsNone(dp.get_probe())

    def test_gpu_events_flag_strict(self):
        for raw, exp in (("", True), ("1", True), ("0", False)):
            with mock.patch.dict(os.environ, {"X_FLAG": raw}):
                self.assertEqual(dp._env_flag_strict("X_FLAG", True), exp)
        with mock.patch.dict(os.environ, {"X_FLAG": "yes"}):
            with self.assertRaises(ValueError):
                dp._env_flag_strict("X_FLAG", True)

    def test_validate_record(self):
        good = {"record_schema": dp.DECODE_SCHEMA, "event": "decode_forward", "mode": "decode",
                "bs_real": 5, "bs_padded": 8, "graph": "hit", "pad_check": "ok",
                "decode_fwd_gpu_ms": 1.0, "decode_fwd_host_ms": 0.1}
        self.assertEqual(dp.validate_record(good), [])
        self.assertTrue(dp.validate_record({**good, "record_schema": "pdmux.decode_probe/v0"}))
        self.assertTrue(dp.validate_record({**good, "graph": "eager"}))  # eager must be unpadded
        self.assertTrue(dp.validate_record({**good, "bs_padded": 4}))
        bad = dict(good)
        bad.pop("decode_fwd_gpu_ms")
        self.assertTrue(dp.validate_record(bad))

    def test_i4_field_is_new_not_phase_event_fields(self):
        self.assertEqual(dp.I4_FIELD, "decode_fwd_gpu_ms")
        self.assertNotIn(dp.I4_FIELD, ("t_launch", "t_sync"))

    def test_jsonl_through_async_writer(self):
        sys.path.insert(0, str(ENG / "src" / "multiplex"))
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "dec.jsonl"
                gpu, clock = FakeGPU(), Clock()
                runner = FakeRunner(gpu, clock, FakeGraphRunner())
                probe = dp.DecodeProbe(path, run_id="r", workload_id="w", window_s=1e9,
                                       boot={"capture_bs": CAPTURE}, clock=clock,
                                       event_factory=lambda: FakeEvent(gpu))
                dp.set_probe(probe)
                dp.install_on_runner(runner)
                runner.forward(FB(5))
                gpu.complete_all()
                probe.close()
                rows = [json.loads(x) for x in path.read_text().splitlines()]
        finally:
            sys.path.remove(str(ENG / "src" / "multiplex"))
        events = [r["event"] for r in rows]
        self.assertEqual(events[0], "decode_probe_boot")
        self.assertIn("decode_forward", events)
        self.assertIn("decode_probe_close", events)
        for r in rows:
            if r["event"] in ("decode_forward", "decode_probe_window", "decode_probe_boot"):
                self.assertEqual(dp.validate_record(r), [], r)
        f = next(r for r in rows if r["event"] == "decode_forward")
        self.assertEqual((f["bs_padded"], f["decode_fwd_gpu_ms"]), (8, 20.0))


# ---------------------------------------------------------------------------
# F. arm symmetry + upstream assumptions (installed runtime source)
# ---------------------------------------------------------------------------

def _fn_src(path, cls, fn):
    src = Path(path).read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == fn:
                    return ast.get_source_segment(src, item), item
    raise AssertionError(f"{cls}.{fn} not found in {path}")


class TestUpstreamAssumptions(unittest.TestCase):
    def setUp(self):
        if not (SRT / "model_executor" / "model_runner.py").is_file():
            self.skipTest(f"runtime tree absent: {SRT}")

    def test_every_loop_reaches_model_runner_forward(self):
        s, _ = _fn_src(SRT / "managers/tp_worker.py", "TpModelWorker", "forward_batch_generation")
        self.assertIn("self.model_runner.forward(", s)
        s, _ = _fn_src(SRT / "managers/tp_worker.py", "TpModelWorker", "forward_batch_split_prefill")
        self.assertIn("self.model_runner.forward(", s)
        s, _ = _fn_src(SRT / "managers/scheduler.py", "Scheduler", "run_batch")
        self.assertIn("forward_batch_generation(", s)
        for loop in ("event_loop_normal", "event_loop_overlap"):
            s, _ = _fn_src(SRT / "managers/scheduler.py", "Scheduler", loop)
            self.assertIn("self.run_batch(", s, loop)
        mixin = ENG / "src/multiplex/multiplexing_mixin.py"
        s, _ = _fn_src(mixin, "SchedulerMultiplexMixin", "event_loop_pdmux")
        self.assertIn("self.run_batch(self.running_batch)", s)

    def test_model_runner_contract(self):
        s, node = _fn_src(SRT / "model_executor/model_runner.py", "ModelRunner", "forward")
        self.assertEqual(node.args.args[1].arg, "forward_batch")
        raw, _ = _fn_src(SRT / "model_executor/model_runner.py", "ModelRunner", "_forward_raw")
        self.assertIn("self.graph_runner.can_run(forward_batch)", raw)
        self.assertIn("can_run_graph=can_run_graph", raw)
        mr = (SRT / "model_executor/model_runner.py").read_text()
        self.assertRegex(mr, r"class ModelRunnerOutput:\n    logits_output: .*\n    can_run_graph: bool")

    def test_graph_runner_padding_rule(self):
        p = SRT / "model_executor/cuda_graph_runner.py"
        rp, _ = _fn_src(p, "CudaGraphRunner", "replay_prepare")
        self.assertIn("index = bisect.bisect_left(self.capture_bs, raw_bs)", rp)
        self.assertIn("bs = self.capture_bs[index]", rp)
        self.assertIn("self.bs = bs", rp)
        init, _ = _fn_src(p, "CudaGraphRunner", "__init__")
        for attr in ("self.disable_padding =", "self.max_bs =", "self.capture_bs, self.compile_bs ="):
            self.assertIn(attr, init)
        cr, _ = _fn_src(p, "CudaGraphRunner", "can_run")
        self.assertIn("graph_key in self.graphs\n            if self.disable_padding\n            else cuda_graph_bs <= self.max_bs", cr)

    def test_capture_does_not_go_through_model_runner_forward(self):
        p = SRT / "model_executor/cuda_graph_runner.py"
        src = p.read_text()
        cap, _ = _fn_src(p, "CudaGraphRunner", "capture_one_batch_size")
        self.assertNotIn("self.model_runner.forward(", cap)
        # the captured callable is patch_model(self.model_runner.model, ...) -> model.forward
        whole, _ = _fn_src(p, "CudaGraphRunner", "capture")
        self.assertIn("with patch_model(\n                    self.model_runner.model,", whole)
        self.assertNotIn("self.model_runner.forward", whole)
        tree = ast.parse(src)
        pm = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "patch_model")
        self.assertIn("model.forward", ast.get_source_segment(src, pm))

    def test_forward_mode_methods(self):
        p = SRT / "model_executor/forward_batch_info.py"
        s = p.read_text()
        self.assertIn("    def is_decode(self):\n        return self == ForwardMode.DECODE\n", s)
        self.assertIn("    def is_mixed(self):\n        return self == ForwardMode.MIXED\n", s)


# ---------------------------------------------------------------------------
# G. hook patch + sync wiring
# ---------------------------------------------------------------------------

class TestHookPatch(unittest.TestCase):
    def setUp(self):
        self.installed = SRT / "managers/scheduler.py"
        if not self.installed.is_file():
            self.skipTest(f"runtime tree absent: {self.installed}")
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.target = Path(self.tmp) / "sglang/srt/managers/scheduler.py"
        self.target.parent.mkdir(parents=True)
        shutil.copy(self.installed, self.target)
        text = self.target.read_text()
        if "maybe_install_decode_probe" in text:
            self.skipTest("runtime already carries the decode hook")
        if "maybe_install_admission_probe" not in text:
            self._patch(ADM_PATCH, "--forward")
        self.base = self.target.read_text()

    def _patch(self, pf, direction):
        with open(pf, "rb") as fh:
            r = subprocess.run(["patch", direction, "--batch", "-p1", "-d", self.tmp],
                               stdin=fh, capture_output=True)
        self.assertEqual(r.returncode, 0, r.stderr.decode() + r.stdout.decode())
        return r.stdout.decode()

    def test_forward_then_reverse_is_identity(self):
        out = self._patch(PATCH, "--forward")
        self.assertNotIn("fuzz", out)
        self.assertIn("maybe_install_decode_probe", self.target.read_text())
        self._patch(PATCH, "--reverse")
        self.assertEqual(self.target.read_text(), self.base)

    def test_after_admission_inside_run_scheduler_process_class_unmoved(self):
        self._patch(PATCH, "--forward")
        new = self.target.read_text()
        tree = ast.parse(new)
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_scheduler_process")
        seg = ast.get_source_segment(new, fn)
        self.assertIn("scheduler.decode_probe = maybe_install_decode_probe(scheduler)", seg)
        self.assertLess(seg.index("maybe_install_admission_probe(scheduler)"),
                        seg.index("maybe_install_decode_probe(scheduler)"))
        self.assertLess(seg.index("maybe_install_decode_probe(scheduler)"), seg.index("pipe_writer.send("))
        old_cls = next(n for n in ast.parse(self.base).body if isinstance(n, ast.ClassDef) and n.name == "Scheduler")
        new_cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Scheduler")
        self.assertEqual((old_cls.lineno, old_cls.end_lineno), (new_cls.lineno, new_cls.end_lineno))
        self.assertEqual(self.base.splitlines()[: old_cls.end_lineno], new.splitlines()[: new_cls.end_lineno])

    def test_without_admission_only_fuzzy(self):
        # Order matters: the hunk's context is the admission hook.  Without it GNU patch only
        # applies by fuzz (default fuzz 2) -- sync therefore applies it strictly after admission.
        self._patch(ADM_PATCH, "--reverse")
        with open(PATCH, "rb") as fh:
            r = subprocess.run(["patch", "--dry-run", "--forward", "--batch", "--fuzz=0", "-p1", "-d", self.tmp],
                               stdin=fh, capture_output=True)
        self.assertNotEqual(r.returncode, 0)


class TestSyncWiring(unittest.TestCase):
    def setUp(self):
        self.sync = SYNC.read_text()

    def test_installed_and_guarded_after_admission_before_conv(self):
        self.assertIn('"${track_root}/src/multiplex/decode_probe.py"; do', self.sync)
        a = self.sync.index('admission_patch="${track_root}/src/patches/admission_probe_scheduler_hook.patch"')
        d = self.sync.index('decode_patch="${track_root}/src/patches/decode_probe_scheduler_hook.patch"')
        c = self.sync.index('conv_int64_patch=')
        self.assertLess(a, d)
        self.assertLess(d, c)
        self.assertIn('if ! grep -q "maybe_install_decode_probe"', self.sync)

    def test_manifest_entry_30_appended_last(self):
        sys.path.insert(0, str(ENG / "scripts" / "bootstrap"))
        try:
            spec = importlib.util.spec_from_file_location("engine_digest_t", ENG / "scripts/bootstrap/engine_digest.py")
            ed = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(ed)
        finally:
            sys.path.pop(0)
        rels = ed.registered_relpaths(SYNC)
        self.assertEqual(len(rels), 30)
        self.assertEqual(rels[-1], "sglang/srt/multiplex/decode_probe.py")
        self.assertEqual(rels[27:29], ["sglang/srt/multiplex/admission_probe.py",
                                       "sglang/srt/layers/attention/mamba/causal_conv1d_triton.py"])


if __name__ == "__main__":
    unittest.main()
