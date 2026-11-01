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
  H. EXTEND timing flag ``PDMUX_PROBE_EXTEND_TIME`` (decision 62 item 4): OFF reproduces
     the pre-flag probe's records (frozen golden); ON emits ``extend_forward`` records;
     ``pacing_ratios`` r_ser / r_mix equal hand-computed values and
     ``results/stock_baseline/r_ser_holb.py``; mutants (mean denominator, MIXED counted
     as EXTEND, ``is_extend()`` classifier, no same-stream filter, no seq-hole check, ...)
     each fail.
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
import textwrap
import statistics
import inspect
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

    def test_decode_iter_seq_recorded(self):
        # diag audit r1 M4: PD-mux decode records carry the loop's decode iteration (join key), None otherwise
        it = {"n": 0}
        s = make()
        s.probe.decode_iter_fn = lambda: it["n"]
        for k in (1, 2, 3):
            it["n"] = k
            s.runner.forward(FB(4))
        s.gpu.complete_all()
        s.probe.maybe_flush(force=True)
        self.assertEqual([r["decode_iter_seq"] for r in fwd_records(s.probe)], [1, 2, 3])
        s2 = make()
        s2.runner.forward(FB(4))
        s2.gpu.complete_all()
        s2.probe.maybe_flush(force=True)
        self.assertIsNone(fwd_records(s2.probe)[0]["decode_iter_seq"])

    def test_install_wires_decode_iter_for_pdmux_only(self):
        src = PROBE_PY.read_text()
        self.assertIn('decode_iter_fn = lambda: getattr(scheduler, "_r2_decode_iterations", None)', src)
        self.assertIn("decode_iter_fn=decode_iter_fn,", src)

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



# ---------------------------------------------------------------------------
# H. EXTEND timing flag (decision 62 item 4, 2026-11-01): OFF golden, ON records,
#    r_ser / r_mix on synthetic input, mutation (lesson 53).  CPU only: nothing
#    here says EXTEND events time correctly on a GPU or that the ON boot is cheap.
# ---------------------------------------------------------------------------

GOLDEN = ENG / "tests" / "decode_probe_off_golden_2026-11-01.json"
GOLDEN_MASK = ("t_realtime", "t_realtime_end")  # wall clock: the only non-deterministic fields


class FMReal(FM):
    """FM + the v0.5.10 ``is_extend()`` truth table (forward_batch_info.py ForwardMode.is_extend)."""

    def is_extend(self, include_draft_extend_v2=False):
        return self.name in ("EXTEND", "MIXED", "DRAFT_EXTEND", "TARGET_VERIFY", "SPLIT_PREFILL", "DLLM_EXTEND")


def XB(lens, mode="extend", prefix=None):
    """An EXTEND / MIXED forward batch with the host-side lists the probe reads."""
    return SimpleNamespace(batch_size=len(lens), forward_mode=FMReal(mode), extend_num_tokens=sum(lens),
                           extend_seq_lens_cpu=list(lens),
                           extend_prefix_lens_cpu=list(prefix) if prefix is not None else [0] * len(lens))


def DB(bs):
    return SimpleNamespace(batch_size=bs, forward_mode=FMReal("decode"))


# (batch, gpu_ms) in issue order.  Shared by the OFF golden and the ON scenario.
TRACE = (
    [(DB(4), 10.0)]                                         # 1  first gap (excluded)
    + [(DB(4), ms) for ms in (10.0, 12.0, 10.0, 30.0, 10.0)]  # 2-6 pure bs4 (median 10, mean 14.4)
    + [(DB(8), 20.0)] * 5                                    # 7-11 pure bs8
    + [(XB([2000]), 105.0),                                  # E
       (DB(4), 10.0),                                        # gap: r_ser = 105 / T_dec(4)=10
       (XB([700]), 50.0), (XB([600, 100], prefix=[0, 400]), 60.0),  # E, E
       (DB(5), 10.0),                                        # gap: bs5 has no pure -> fallback T_all = 20
       (XB([300, 1, 1, 1, 1], mode="mixed", prefix=[0, 50, 60, 70, 80]), 40.0),  # MIXED, decode part 4
       (DB(4), 10.0),                                        # pure
       (SimpleNamespace(batch_size=2, forward_mode=FMReal("split_prefill")), 7.0),  # never timed
       (SimpleNamespace(batch_size=2, forward_mode=FMReal("idle")), 0.0),
       (DB(4), 11.0),                                        # pure
       (DB(60), 25.0)]                                       # eager bs_over_max; pure (fallback)
)
# Hand computation: pure gaps = bs4 {10,12,10,30,10,10,11} (median 10, mean 14.71), bs8 {20 x5}, bs60 {25};
# all 13 sorted -> 7th = 20 (mean 16.77).  T_dec(4)=10, T_dec(8)=20, bs5 / bs60 fall back to 20 (n_fallback 2).
# r_ser = [105/10, (50+60)/20]; r_mix = (40 - T_dec(4)) / T_dec(4) with the mixed decode part = 4.
EXPECT_R_SER = [10.5, 110.0 / 20.0]
EXPECT_R_MIX = [3.0]


def run_trace(mod, trace=TRACE, window_s=0.004, **probe_kw):
    """Deterministic probe run (fake clock / GPU); returns the probe's records."""
    gpu, clock = FakeGPU(), Clock()
    gr = FakeGraphRunner()
    runner = FakeRunner(gpu, clock, gr)
    probe = mod.DecodeProbe(None, run_id="g", workload_id="g", window_s=window_s,
                            boot={"capture_bs": list(gr.capture_bs), "disable_cuda_graph_padding": False},
                            clock=clock, event_factory=lambda: FakeEvent(gpu), **probe_kw)
    mod.set_probe(probe)
    mod.install_on_runner(runner)
    try:
        for i, (fb, ms) in enumerate(trace):
            runner.gpu_ms = ms
            runner.forward(fb)
            if i % 3 == 2:
                gpu.complete_all()  # resolution happens at a later entry / flush, as on the GPU
        gpu.complete_all()
        probe.close()
    finally:
        mod.set_probe(None)
        mod.uninstall_all()
    return probe.records


def masked(records):
    return [{k: v for k, v in r.items() if k not in GOLDEN_MASK} for r in records]


def run_on(mod=None, trace=TRACE, **kw):
    mod = mod or dp
    return run_trace(mod, trace, extend_time=True, stream_key_fn=lambda: 7, **kw)


class TestExtendOffGolden(unittest.TestCase):
    """OFF == the pre-flag probe, byte for byte (records generated from the pre-flag source,
    sha256 092e5a10..., and frozen in GOLDEN)."""

    def setUp(self):
        self.golden = json.loads(GOLDEN.read_text(encoding="utf-8"))

    def test_default_matches_golden(self):
        got = masked(run_trace(dp))
        self.assertEqual(json.dumps(got, sort_keys=False), json.dumps(self.golden["records"], sort_keys=False))

    def test_explicit_false_matches_golden(self):
        got = masked(run_trace(dp, extend_time=False, stream_key_fn=lambda: 7))
        self.assertEqual(got, self.golden["records"])

    def test_golden_counts_extend_only_as_other_mode(self):
        wins = [r for r in self.golden["records"] if r["event"] == "decode_probe_window"]
        other = Counter_sum(w["n_other_modes"] for w in wins)
        self.assertEqual(other, {"extend": 3, "split_prefill": 1, "idle": 1})
        self.assertFalse(any(r["event"] == dp.EXTEND_EVENT for r in self.golden["records"]))

    def test_golden_detects_an_off_change(self):
        # the golden is not a tautology: an OFF-path change (stream_key always written) breaks it
        m = mutate('                if self.extend_time:  # ON only: OFF records stay byte-identical\n'
                   '                    fields["stream_key"] = skey\n',
                   '                fields["stream_key"] = None\n                if self.extend_time:\n'
                   '                    fields["stream_key"] = skey\n')
        self.assertNotEqual(masked(run_trace(m)), self.golden["records"])


def Counter_sum(dicts):
    out = {}
    for d in dicts:
        for k, v in d.items():
            out[k] = out.get(k, 0) + v
    return out


def extend_failures(mod):
    """ON scenario + r_ser / r_mix on it; the unmutated module passes, each mutant fails."""
    fails = []

    def expect(cond, msg):
        if not cond:
            fails.append(msg)

    try:
        recs = run_on(mod)
        ext = sorted((r for r in recs if r["event"] == mod.EXTEND_EVENT), key=lambda r: r["seq"])
        expect([r["extend_gpu_ms"] for r in ext] == [105.0, 50.0, 60.0], f"extend gpu ms {[r['extend_gpu_ms'] for r in ext]}")
        expect([(r["n_reqs"], r["extend_tokens"], r["extend_len_max"], r["extend_prefix_tokens"]) for r in ext]
               == [(1, 2000, 2000, 0), (1, 700, 700, 0), (2, 700, 600, 400)], f"extend shapes {ext}")
        expect(all(r["mode"] == "extend" and r["stream_key"] == 7 and mod.validate_record(r) == [] for r in ext),
               "extend record fields / schema")
        expect(all(abs(r["extend_host_ms"] - 1.5) < 1e-6 for r in ext), "extend host ms")
        dec = sorted((r for r in recs if r["event"] == "decode_forward"), key=lambda r: r["seq"])
        mixed = [r for r in dec if r["mode"] == "mixed"]
        expect(len(mixed) == 1 and mixed[0].get("mixed_decode_bs_est") == 4 and mixed[0].get("extend_tokens") == 304,
               f"mixed record {mixed}")
        expect(all(r.get("stream_key") == 7 for r in dec), "decode stream_key")
        seqs = sorted(r["seq"] for r in ext + dec)
        expect(seqs == list(range(1, len(seqs) + 1)), f"seq not contiguous {seqs}")
        expect(len(seqs) == len(TRACE) - 2, f"timed forwards {len(seqs)} (split_prefill/idle must not be timed)")
        win = [r for r in recs if r["event"] == "decode_probe_window"]
        expect(sum(w["n_extend_timed"] for w in win) == 3 and sum(w["n_extend_resolved"] for w in win) == 3,
               "window extend counters")
        expect(sum(w["n_decode"] for w in win) == 16 and sum(w["n_mixed"] for w in win) == 1,
               "decode window counts must not include extend")
        expect(Counter_sum(w["n_other_modes"] for w in win) == {"extend": 3, "split_prefill": 1, "idle": 1},
               "n_other_modes meaning unchanged ON")
        boot = [r for r in recs if r["event"] == "decode_probe_boot"][0]
        expect(boot.get("extend_time_enabled") is True, "boot flag")

        pr = mod.pacing_ratios(recs)
        expect(all(abs(a - b) < 1e-9 for a, b in zip(pr["r_ser"], EXPECT_R_SER)) and len(pr["r_ser"]) == 2,
               f"r_ser {pr['r_ser']} != {EXPECT_R_SER}")
        expect(len(pr["r_mix"]) == 1 and abs(pr["r_mix"][0] - EXPECT_R_MIX[0]) < 1e-9, f"r_mix {pr['r_mix']}")
        expect(pr["tdec_by_bs"] == {4: 10.0, 8: 20.0} and pr["tdec_all_ms"] == 20.0,
               f"T_dec {pr['tdec_by_bs']} {pr['tdec_all_ms']}")
        expect((pr["n_first"], pr["n_incomplete"], pr["n_fallback"], pr["n_pure"]) == (1, 0, 2, 13),
               f"counts {pr['n_first'], pr['n_incomplete'], pr['n_fallback'], pr['n_pure']}")

        # direct records: other-stream extend is not a serial stall; a seq hole excludes the gap
        boot = {"event": "decode_probe_boot", "extend_time_enabled": True}

        def d(seq, ms, bs=4, sk=1, mode="decode"):
            return {"event": "decode_forward", "seq": seq, "mode": mode, "bs_real": bs, "stream_key": sk,
                    "decode_fwd_gpu_ms": ms, "t_host_start": float(seq)}

        def e(seq, ms, sk=1):
            return {"event": "extend_forward", "seq": seq, "mode": "extend", "extend_gpu_ms": ms, "stream_key": sk,
                    "extend_tokens": 100, "n_reqs": 1, "t_host_start": float(seq)}

        base = [boot] + [d(i, 10.0) for i in range(1, 7)]
        p2 = mod.pacing_ratios(base + [e(7, 40.0, sk=2), d(8, 10.0)], min_bs=5)
        expect(p2["r_ser"] == [] and p2["gaps"][-1]["other_stream_ms"] == 40.0, f"other stream counted {p2['r_ser']}")
        p3 = mod.pacing_ratios(base + [e(8, 40.0), d(9, 10.0)], min_bs=5)  # seq 7 dropped
        expect(p3["n_incomplete"] == 1 and p3["r_ser"] == [], f"seq hole not excluded {p3['n_incomplete']} {p3['r_ser']}")
        p4 = mod.pacing_ratios(base + [e(7, 40.0), d(8, 10.0)], min_bs=5, t_window=(2.0, 8.0))
        expect(p4["r_ser"] == [4.0] and p4["n_first"] == 1 and p4["n_pure"] == 4, f"t_window {p4}")
        try:
            mod.pacing_ratios([{"event": "decode_probe_boot"}] + base[1:])
            fails.append("OFF file accepted by pacing_ratios")
        except ValueError:
            pass
    except BlockingCall as exc:
        fails.append(f"blocking call: {exc}")
    except Exception as exc:  # a mutant may crash: that is a detection too
        fails.append(f"exception {type(exc).__name__}: {exc}")
    finally:
        mod.set_probe(None)
        mod.uninstall_all()
    return fails


class TestExtendOn(unittest.TestCase):
    def tearDown(self):
        dp.set_probe(None)
        dp.uninstall_all()

    def test_unmutated_passes(self):
        self.assertEqual(extend_failures(dp), [])

    def test_extend_requires_events(self):
        with self.assertRaises(ValueError):
            dp.DecodeProbe(None, events_enabled=False, extend_time=True)

    def test_extend_pending_does_not_hold_back_decode(self):
        # separate deque: an unresolved extend never blocks resolution of decode records
        gpu, clock = FakeGPU(), Clock()
        runner = FakeRunner(gpu, clock, FakeGraphRunner())
        probe = dp.DecodeProbe(None, window_s=1e9, boot={"capture_bs": CAPTURE}, clock=clock,
                               event_factory=lambda: FakeEvent(gpu), extend_time=True, max_pending=2)
        dp.set_probe(probe)
        dp.install_on_runner(runner)
        for _ in range(3):
            runner.forward(XB([100]))
        probe.maybe_flush(force=True)
        win = [r for r in probe.records if r["event"] == "decode_probe_window"][-1]
        self.assertEqual((win["n_extend_pending"], win["n_extend_dropped_pending"], win["n_pending"]), (2, 1, 0))
        self.assertEqual([r for r in probe.records if r["event"] == dp.EXTEND_EVENT], [])

    def test_capture_guard_applies_to_extend(self):
        recs = run_trace(dp, [(XB([100]), 5.0)], extend_time=True, capturing_fn=lambda: True)
        self.assertFalse(any(r["event"] == dp.EXTEND_EVENT for r in recs))

    def test_validate_extend_record(self):
        rec = next(r for r in run_on() if r["event"] == dp.EXTEND_EVENT)
        self.assertEqual(dp.validate_record(rec), [])
        self.assertTrue(dp.validate_record({**rec, "mode": "mixed"}))
        bad = dict(rec)
        bad.pop("extend_gpu_ms")
        self.assertTrue(dp.validate_record(bad))

    def test_is_plain_extend_table(self):
        self.assertTrue(dp.is_plain_extend(FMReal("extend")))
        for m in ("mixed", "split_prefill", "target_verify", "draft_extend", "dllm_extend", "decode", "idle"):
            self.assertFalse(dp.is_plain_extend(FMReal(m)), m)
        self.assertFalse(dp.is_plain_extend(None))

    def test_env_flag(self):
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = SimpleNamespace(Event=lambda enable_timing=True: object(),
                                          is_current_stream_capturing=lambda: False,
                                          current_stream=lambda: SimpleNamespace(cuda_stream=11))
        runner = FakeRunner(FakeGPU(), Clock(), FakeGraphRunner())
        sched = SimpleNamespace(tp_worker=SimpleNamespace(model_runner=runner))
        sys.path.insert(0, str(ENG / "src" / "multiplex"))
        try:
            with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(sys.modules, {"torch": fake_torch}):
                env = {"PDMUX_DECODE_PROBE_PATH": str(Path(tmp) / "d.jsonl")}
                for bad_env in ({dp.EXTEND_ENV: "1", "PDMUX_DECODE_PROBE_PATH": ""},
                                {**env, dp.EXTEND_ENV: "1", "PDMUX_DECODE_PROBE_GPU_EVENTS": "0"},
                                {**env, dp.EXTEND_ENV: "yes"}):
                    with mock.patch.dict(os.environ, bad_env):
                        with self.assertRaises(ValueError, msg=bad_env):
                            dp.maybe_install_decode_probe(sched)
                for val, exp in (("", False), ("0", False), ("1", True)):
                    with mock.patch.dict(os.environ, {**env, dp.EXTEND_ENV: val, "PDMUX_DECODE_PROBE_GPU_EVENTS": ""}):
                        p = dp.maybe_install_decode_probe(sched)
                        self.assertEqual(p.extend_time, exp, val)
                        if exp:
                            self.assertEqual(p.stream_key_fn(), 11)
                        else:
                            self.assertIsNone(p.stream_key_fn)
                        p.close()
                        dp.set_probe(None)
                        dp.uninstall_all()
                with mock.patch.dict(os.environ, {"PDMUX_DECODE_PROBE_PATH": "", dp.EXTEND_ENV: "0"}):
                    self.assertIsNone(dp.maybe_install_decode_probe(sched))
        finally:
            sys.path.remove(str(ENG / "src" / "multiplex"))

    def test_docstring_states_instrumentation_boot_only(self):
        doc = ast.get_docstring(ast.parse(SRC))
        self.assertIn("INSTRUMENTATION BOOT ONLY", doc)
        self.assertIn("never goodput / latency", doc)


class TestPacingMatchesPhaseA3(unittest.TestCase):
    """``pacing_ratios`` == the Gate 2 HOLB re-analysis definition (results/stock_baseline/r_ser_holb.py
    ``seg_stats``, PLAN sec 0-1) on the same gaps, so Phase A3 (KISTI HOLB) and Phase B (this probe) share r_ser."""

    def test_same_values_as_r_ser_holb(self):
        sb = ENG / "results" / "stock_baseline"
        if not (sb / "r_ser_holb.py").is_file():
            self.skipTest("r_ser_holb.py absent")
        sys.path.insert(0, str(sb))
        try:
            spec = importlib.util.spec_from_file_location("r_ser_holb_t", sb / "r_ser_holb.py")
            rh = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(rh)
        finally:
            sys.path.remove(str(sb))
        pr = dp.pacing_ratios(run_on())
        self.assertEqual(rh.MIN_BS, dp.PACING_MIN_BS)
        holb = [{"gap_class": "first", "decode_mode": "DECODE", "decode_bs": 4, "decode_dur_ms": 10.0,
                 "n_other_fw": 0, "same_stream_fw_ms": 0.0, "host_ts": 0.0}]
        holb += [{"gap_class": "strict", "decode_mode": g["mode"].upper(), "decode_bs": g["bs"],
                  "decode_dur_ms": g["decode_ms"], "n_other_fw": g["n_extend"],
                  "same_stream_fw_ms": g["same_stream_ms"], "host_ts": float(i + 1)} for i, g in enumerate(pr["gaps"])]
        st = rh.seg_stats(holb)
        self.assertEqual(st["_r_ser"], pr["r_ser"])
        self.assertEqual(st["_r_mix"], pr["r_mix"])
        self.assertEqual(st["tdec_median_ms"], pr["tdec_all_ms"])
        self.assertEqual(st["n_pure"], pr["n_pure"])


class TestExtendMutants(unittest.TestCase):
    MUTANTS = {
        # T_dec denominator = mean instead of median
        "tdec_mean": ("    tdec_by_bs = {bs: statistics.median(xs) for bs, xs in by_bs.items() if len(xs) >= int(min_bs)}\n",
                      "    tdec_by_bs = {bs: statistics.mean(xs) for bs, xs in by_bs.items() if len(xs) >= int(min_bs)}\n"),
        "tdec_all_mean": ('    tdec_all = statistics.median([g["decode_ms"] for g in pure]) if pure else None\n',
                          '    tdec_all = statistics.mean([g["decode_ms"] for g in pure]) if pure else None\n'),
        # MIXED counted as EXTEND (analysis side)
        "mixed_as_extend_rec": ("    return r.get(\"event\") == EXTEND_EVENT\n",
                                "    return r.get(\"event\") == EXTEND_EVENT or r.get(\"mode\") == MODE_MIXED\n"),
        # MIXED / SPLIT_PREFILL timed as EXTEND (probe side: the natural is_extend() bug)
        "classify_is_extend": ("    return forward_mode is not None and mode_name(forward_mode) == MODE_EXTEND\n",
                               "    return forward_mode is not None and _truth(forward_mode, \"is_extend\")\n"),
        "mixed_routed_to_extend": ("    if _truth(forward_mode, \"is_mixed\"):\n        return MODE_MIXED\n", ""),
        "mixed_bs_is_bs_real": ("    if r.get(\"mode\") == MODE_MIXED:\n        return r.get(\"mixed_decode_bs_est\")\n", ""),
        "no_same_stream_filter": ('                same = [e for e in open_ext if e.get("stream_key") == r.get("stream_key")]\n',
                                  '                same = list(open_ext)\n'),
        "no_seq_hole_check": ("            complete = got == expect and r.get(I4_FIELD)", "            complete = True and r.get(I4_FIELD)"),
        "pure_includes_extend_gaps": ('    pure = [g for g in gaps if g["mode"] == MODE_DECODE and g["n_extend"] == 0]\n',
                                      '    pure = [g for g in gaps if g["mode"] == MODE_DECODE]\n'),
        "first_gap_kept": ("        if prev is None:\n            n_first += 1\n        else:\n",
                           "        if prev is None:\n            n_first += 1\n            prev = {\"seq\": int(r[\"seq\"]) - 1 - len(open_ext)}\n        if True:\n"),
        "extend_host_time": ("                item.fields[EXT_GPU_FIELD] = gpu_ms\n",
                             "                item.fields[EXT_GPU_FIELD] = item.fields[EXT_HOST_FIELD]\n"),
        "off_file_accepted": ('    if not boots or not all(b.get("extend_time_enabled") is True for b in boots):\n',
                              '    if False:\n'),
    }

    def test_every_mutant_fails(self):
        for name, (old, new) in self.MUTANTS.items():
            with self.subTest(mutant=name):
                fails = extend_failures(mutate(old, new))
                self.assertTrue(fails, f"mutant {name} survived the EXTEND scenario")


class TestLoadWindowContamination(unittest.TestCase):
    """PREREG STOCK-BASE 5-1 mutation test: records outside the registered load window (warmup / smoke
    prefills before t_load_start) must not enter r_ser.  The policy_implfix PC reduction read the whole boot
    telemetry (warmup + guard probes = ~70 % of spans) and its labels did not measure the registered load
    (result audit r1, 2026-11-01).  Here the same contamination is built on purpose: a long warmup extend
    before the window would push r_ser p50 up; applying ``t_window`` must remove it, and a mutant that
    ignores the window (no filtering) must give a different p50 -- otherwise the window is an identity."""

    @staticmethod
    def _recs():
        boot = {"event": "decode_probe_boot", "extend_time_enabled": True}

        def d(seq, ms, t, bs=4):
            return {"event": "decode_forward", "seq": seq, "mode": "decode", "bs_real": bs, "stream_key": 1,
                    "decode_fwd_gpu_ms": ms, "t_host_start": t}

        def e(seq, ms, t):
            return {"event": "extend_forward", "seq": seq, "mode": "extend", "extend_gpu_ms": ms, "stream_key": 1,
                    "extend_tokens": 2048, "n_reqs": 1, "t_host_start": t}

        # warmup (t < 100): decodes + one huge prefill (r = 50); load window [100, 200]: decodes + two prefills (r = 2, 4)
        recs = [boot] + [d(i, 10.0, float(i)) for i in range(1, 7)] + [e(7, 500.0, 7.0), d(8, 10.0, 8.0)]
        recs += [d(i, 10.0, 100.0 + i) for i in range(9, 15)] + [e(15, 20.0, 115.0), d(16, 10.0, 116.0),
                                                                  e(17, 40.0, 117.0), d(18, 10.0, 118.0)]
        return recs

    def test_window_excludes_warmup_and_is_not_identity(self):
        recs = self._recs()
        inside = dp.pacing_ratios(recs, min_bs=5, t_window=(100.0, 200.0))
        self.assertEqual(sorted(inside["r_ser"]), [2.0, 4.0], inside["r_ser"])
        whole = dp.pacing_ratios(recs, min_bs=5)            # the mutant behaviour: no window = boot-wide reduce
        self.assertEqual(sorted(whole["r_ser"]), [2.0, 4.0, 50.0], whole["r_ser"])
        med = lambda xs: statistics.median(xs)
        self.assertNotEqual(med(inside["r_ser"]), med(whole["r_ser"]),
                            "window filter is an identity on this input -- mutation undetectable")

    def test_window_mutant_detected(self):
        # the same as pc_reduce's defect: filtering silently dropped -> label must change
        src = inspect.getsource(dp.pacing_ratios)
        self.assertIn("if t_window is not None:", src)
        mutated = src.replace("if t_window is not None:", "if False:")
        ns = dict(vars(dp))
        exec(textwrap.dedent(mutated), ns)
        r = ns["pacing_ratios"](self._recs(), min_bs=5, t_window=(100.0, 200.0))
        self.assertNotEqual(sorted(r["r_ser"]), [2.0, 4.0], "window-ignoring mutant survived")


if __name__ == "__main__":
    unittest.main()
