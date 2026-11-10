"""CPU gate for four default-OFF observation records.

  forward interior   decode_probe, PDMUX_PROBE_FORWARD_INTERIOR
  request admission  admission_probe, PDMUX_ADMISSION_REQ_RECORDS
  mixed forward      admission_probe, PDMUX_ADMISSION_MIXED_RATIO
  decode SM share    multiplexing_mixin, PDMUX_PHASE_DECODE_SHARE

Nothing here is GPU evidence.  It does not show that the host stamps land where a
real forward spends its time, that ON and OFF serve the same output, or what the
observer cost is.  It pins: the record contents against hand-written values, that
each flag OFF leaves behaviour unchanged, that a probe failure never reaches the
serving path, that the upstream call structure the shadows rely on exists, and
that source-level mutants of the new code fail (lesson 53).
"""

import os
import re
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_admission_probe as tap  # noqa: E402
import test_decode_probe as tdp  # noqa: E402
import test_phase_events as tpe  # noqa: E402
from pdmux_loop_fakes import LoopTestBase, mixin_module, run_loop  # noqa: E402

dp = tdp.dp
ap = tap.ap
SRT = Path(tdp.SRT)
CAP = tap.EngineFixture.CAP
CLIP = tap.CLIP_MAX_NEW_TOKENS


# ---------------------------------------------------------------------------
# Forward interior: a fake runner with the real call structure
# ---------------------------------------------------------------------------
class Linear:
    def __init__(self, clock, ms):
        self.clock, self.ms = clock, ms

    def init_forward_metadata(self, fb):
        self.clock.t += self.ms / 1000.0


class Hybrid:
    def __init__(self, clock, full_ms, ssm_ms, replay_ms):
        self.clock, self.full_ms, self.replay_ms = clock, full_ms, replay_ms
        self.linear_attn_backend = Linear(clock, ssm_ms)

    def init_forward_metadata(self, fb):
        self.clock.t += self.full_ms / 1000.0
        self.linear_attn_backend.init_forward_metadata(fb)

    def init_forward_metadata_replay_cuda_graph(self, *a, **k):
        self.clock.t += self.replay_ms / 1000.0


class Model:
    def __init__(self, clock, ms):
        self.clock, self.ms, self.raise_next = clock, ms, False

    def forward(self, *a, **k):
        if self.raise_next:
            self.raise_next = False
            raise ValueError("engine failure")
        self.clock.t += self.ms / 1000.0
        return "logits"


class Graph(tdp.FakeGraphRunner):
    def __init__(self, runner, prep_ms):
        super().__init__()
        self.runner, self.prep_ms = runner, prep_ms

    def replay_prepare(self, fb):
        self.runner.clock.t += self.prep_ms / 1000.0
        self.runner.attn_backend.init_forward_metadata_replay_cuda_graph()
        self.bs = next(b for b in sorted(self.capture_bs) if b >= fb.batch_size)

    def replay(self, fb):
        self.replay_prepare(fb)
        self.runner.clock.t += 0.2 / 1000.0  # graph replay itself


class Runner:
    """Mirrors ModelRunner._forward_raw for the three paths the probe must see."""

    PRE_MS, POST_MS = 1.0, 0.5

    def __init__(self, clock, piecewise, use_graph=True):
        self.clock = clock
        self.piecewise = piecewise
        self.use_graph = use_graph
        self.attn_backend = Hybrid(clock, 2.0, 3.0, 0.7)
        self.model = Model(clock, 10.0)
        self.graph_runner = Graph(self, 0.4)
        self.forward_pass_id = 0

    def forward(self, fb, **kw):
        self.forward_pass_id += 1
        self.clock.t += self.PRE_MS / 1000.0
        fm = fb.forward_mode
        if fm.is_decode() and self.use_graph:
            self.graph_runner.replay(fb)
            can = True
        elif fm.is_decode():
            self.attn_backend.init_forward_metadata(fb)
            self.model.forward(fb)
            can = False
        else:
            # piecewise ON: PiecewiseCudaGraphRunner.replay; OFF: forward_extend.
            # Both call the same two objects.
            self.attn_backend.init_forward_metadata(fb)
            self.model.forward(fb)
            can = self.piecewise
        self.clock.t += self.POST_MS / 1000.0
        return SimpleNamespace(logits_output="L", can_run_graph=can)


def interior_probe(mod, piecewise=False, use_graph=True, capturing_fn=None, extend_time=False):
    gpu, clock = tdp.FakeGPU(), tdp.Clock()
    runner = Runner(clock, piecewise, use_graph)
    kw = {"extend_time": True, "stream_key_fn": lambda: 7} if extend_time else {}
    probe = mod.DecodeProbe(None, window_s=1e9, boot={"capture_bs": tdp.CAPTURE}, clock=clock,
                            event_factory=lambda: tdp.FakeEvent(gpu),
                            capturing_fn=capturing_fn or (lambda: False), interior=True, **kw)
    mod.set_probe(probe)
    mod.install_on_runner(runner)
    mod.install_interior(runner)
    return SimpleNamespace(probe=probe, runner=runner, gpu=gpu, clock=clock)


def interior_records(probe):
    return [r for r in probe.records if r["event"] == dp.INTERIOR_EVENT]


def close(a, b):
    return a is not None and b is not None and abs(a - b) < 1e-6


def interior_failures(mod):
    """Returns failures; empty = pass.  Also used by the mutants."""
    fails = []

    def check(name, cond, detail=None):
        if not cond:
            fails.append((name, detail))

    def segments_partition(r):
        parts = [r["pre_meta_ms"], r["meta_ms"], r["meta_to_launch_ms"], r["launch_to_return_ms"]]
        return None not in parts and close(sum(parts), r["forward_ms"])

    try:
        for piecewise in (True, False):
            s = interior_probe(mod, piecewise=piecewise)
            s.runner.forward(tdp.XB([500]))
            recs = interior_records(s.probe)
            check("extend/one", len(recs) == 1, recs)
            if recs:
                r = recs[0]
                check("extend/mode", r["mode"] == "extend", r)
                check("extend/graph", r["graph"] == ("hit" if piecewise else "eager"), r)
                check("extend/pre", close(r["pre_meta_ms"], 1.0), r)
                check("extend/meta", close(r["meta_ms"], 5.0), r)
                check("extend/ssm", close(r["ssm_meta_ms"], 3.0), r)
                check("extend/launch_gap", close(r["meta_to_launch_ms"], 0.0), r)
                check("extend/launch_to_return", close(r["launch_to_return_ms"], 10.5), r)
                check("extend/source", r["launch_source"] == dp.LAUNCH_MODEL, r)
                check("extend/model_return", r["t_model_return"] is not None, r)
                check("extend/segments", segments_partition(r), r)
                check("extend/counts", (r["n_meta_calls"], r["n_model_calls"]) == (1, 1), r)
            mod.set_probe(None)
            mod.uninstall_all()

        # MIXED goes down the same path and joins decode_forward by seq
        s = interior_probe(mod)
        s.runner.forward(tdp.XB([300, 1, 1], mode="mixed"))
        s.gpu.complete_all()
        s.probe.poll()
        fwd = tdp.fwd_records(s.probe)
        recs = interior_records(s.probe)
        check("mixed/mode", len(recs) == 1 and recs[0]["mode"] == "mixed", recs)
        check("mixed/seq_join", len(fwd) == 1 and recs and fwd[0]["seq"] == recs[0]["seq"], (fwd, recs))
        check("mixed/segments", recs and segments_partition(recs[0]), recs)
        mod.set_probe(None)
        mod.uninstall_all()

        # decode graph hit: replay metadata, launch at replay_prepare return, no model call
        s = interior_probe(mod)
        s.runner.forward(tdp.FB(4))
        recs = interior_records(s.probe)
        check("graph/one", len(recs) == 1, recs)
        if recs:
            r = recs[0]
            check("graph/source", r["launch_source"] == dp.LAUNCH_GRAPH, r)
            check("graph/meta", close(r["meta_ms"], 0.7), r)
            check("graph/no_ssm", r["ssm_meta_ms"] is None, r)
            check("graph/no_model", r["t_model_return"] is None and r["n_model_calls"] == 0, r)
            check("graph/hit", r["graph"] == "hit", r)
            check("graph/segments", segments_partition(r), r)
        mod.set_probe(None)
        mod.uninstall_all()

        # decode eager
        s = interior_probe(mod, use_graph=False)
        s.runner.forward(tdp.FB(4))
        r = interior_records(s.probe)[0]
        check("eager/source", r["launch_source"] == dp.LAUNCH_MODEL and r["graph"] == "eager", r)
        check("eager/segments", segments_partition(r), r)
        mod.set_probe(None)
        mod.uninstall_all()

        # engine exception propagates and closes the thread-local
        s = interior_probe(mod)
        s.runner.model.raise_next = True
        raised = False
        try:
            s.runner.forward(tdp.XB([10]))
        except ValueError:
            raised = True
        check("exc/propagates", raised)
        check("exc/stamps_closed", getattr(s.probe._tls, "stamps", None) is None)
        mod.set_probe(None)
        mod.uninstall_all()

        # EXTEND with extend timing ON shares seq with extend_forward
        s = interior_probe(mod, extend_time=True)
        s.runner.forward(tdp.XB([10]))
        s.gpu.complete_all()
        s.probe.poll()
        ext = [r for r in s.probe.records if r["event"] == dp.EXTEND_EVENT]
        recs = interior_records(s.probe)
        check("ext/join", len(ext) == 1 and len(recs) == 1 and ext[0]["seq"] == recs[0]["seq"], (ext, recs))
        mod.set_probe(None)
        mod.uninstall_all()
    except Exception as exc:  # a broken mutant counts as failing, not as a crash
        fails.append(("exception", repr(exc)))
    finally:
        mod.set_probe(None)
        mod.uninstall_all()
    return fails


class TestForwardInterior(unittest.TestCase):
    def tearDown(self):
        dp.set_probe(None)
        dp.uninstall_all()

    def test_scenarios(self):
        self.assertEqual(interior_failures(dp), [])

    def test_stamps_outside_a_recorded_forward_are_ignored(self):
        s = interior_probe(dp)
        self.assertEqual(s.runner.model.forward(None), "logits")
        s.runner.attn_backend.init_forward_metadata(None)
        self.assertEqual(interior_records(s.probe), [])
        self.assertEqual(s.probe.n_interior_errors, 0)

    def test_capturing_forward_records_nothing(self):
        s = interior_probe(dp, capturing_fn=lambda: True)
        s.runner.forward(tdp.XB([10]))
        self.assertEqual(interior_records(s.probe), [])

    def test_capture_inside_a_forward_is_not_stamped(self):
        # recapture_if_needed runs inside replay_prepare, on a forward already open
        flags = {"capturing": False}
        s = interior_probe(dp, capturing_fn=lambda: flags["capturing"])
        real = s.runner.attn_backend.init_forward_metadata

        def with_capture(fb):
            flags["capturing"] = True
            s.runner.model.forward(fb)
            flags["capturing"] = False
            return real(fb)

        s.runner.attn_backend.__dict__["init_forward_metadata"] = with_capture
        s.runner.forward(tdp.XB([10]))
        r = interior_records(s.probe)[0]
        self.assertEqual(r["n_model_calls"], 1)  # the real call only
        self.assertTrue(close(r["launch_to_return_ms"], 10.5), r)

    def test_probe_failure_never_reaches_the_forward(self):
        calls = {"n": 0}

        def bad():
            calls["n"] += 1
            if calls["n"] == 1:  # the around_forward capture check
                return False
            raise RuntimeError("probe bug")

        s = interior_probe(dp, capturing_fn=bad)
        out = s.runner.forward(tdp.FB(4))
        self.assertEqual(out.logits_output, "L")
        self.assertFalse(s.probe.interior)
        self.assertEqual(s.probe.n_interior_errors, 1)
        self.assertFalse(s.probe.disabled)
        s.gpu.complete_all()
        s.probe.poll()
        self.assertEqual(len(tdp.fwd_records(s.probe)), 1)  # decode_forward unaffected
        self.assertEqual(interior_records(s.probe), [])

    def test_unshadowable_site_is_reported_not_raised(self):
        class NoSet:
            __slots__ = ()

            def init_forward_metadata(self, fb):
                return None

        runner = SimpleNamespace(attn_backend=NoSet(), model=None, graph_runner=None)
        self.assertEqual(dp.install_interior(runner), [])

    def test_uninstall_restores_every_site(self):
        s = interior_probe(dp)
        backend, model, graph = s.runner.attn_backend, s.runner.model, s.runner.graph_runner
        self.assertIn("forward", vars(model))
        dp.set_probe(None)
        dp.uninstall_all()
        self.assertNotIn("forward", vars(model))
        self.assertNotIn("init_forward_metadata", vars(backend))
        self.assertNotIn("init_forward_metadata", vars(backend.linear_attn_backend))
        self.assertNotIn("replay_prepare", vars(graph))

    def test_flag_off_matches_the_frozen_golden(self):
        golden = tdp.json.loads(tdp.GOLDEN.read_text(encoding="utf-8"))["records"]
        self.assertEqual(tdp.masked(tdp.run_trace(dp)), golden)
        self.assertEqual(tdp.masked(tdp.run_trace(dp, interior=False)), golden)

    def test_flag_on_changes_no_existing_record(self):
        def strip(records):
            out = []
            for r in records:
                if r["event"] == dp.INTERIOR_EVENT:
                    continue
                out.append({k: v for k, v in r.items()
                            if k not in ("forward_interior_enabled", "n_interior_records", "n_interior_errors")})
            return tdp.masked(out)

        base = dict(extend_time=True, stream_key_fn=lambda: 7)
        off = tdp.run_trace(dp, **base)
        on = tdp.run_trace(dp, interior=True, **base)
        self.assertEqual(strip(on), strip(off))
        self.assertGreater(sum(1 for r in on if r["event"] == dp.INTERIOR_EVENT), 0)

    def test_flag_off_installs_no_shadow(self):
        s = tdp.make(dp)
        self.assertFalse(s.probe.interior)
        with mock.patch.dict(os.environ, {"PDMUX_PROBE_FORWARD_INTERIOR": ""}, clear=False):
            os.environ.pop("PDMUX_DECODE_PROBE_PATH", None)
            self.assertIsNone(dp.maybe_install_decode_probe(SimpleNamespace()))
        dp.set_probe(None)
        dp.uninstall_all()

    def test_env_is_strict_and_needs_a_path(self):
        with mock.patch.dict(os.environ, {"PDMUX_PROBE_FORWARD_INTERIOR": "1"}, clear=False):
            os.environ.pop("PDMUX_DECODE_PROBE_PATH", None)
            with self.assertRaises(ValueError):
                dp.maybe_install_decode_probe(SimpleNamespace())
        with mock.patch.dict(os.environ, {"PDMUX_PROBE_FORWARD_INTERIOR": "yes"}, clear=False):
            with self.assertRaises(ValueError):
                dp.maybe_install_decode_probe(SimpleNamespace())

    def test_interior_seq_holes_are_refused_by_pacing(self):
        s = interior_probe(dp)
        s.runner.forward(tdp.XB([10]))
        s.runner.forward(tdp.FB(4))
        s.gpu.complete_all()
        s.probe.close()
        with self.assertRaises(ValueError):
            dp.pacing_ratios(s.probe.records)


class TestUpstreamCallStructure(unittest.TestCase):
    """The shadows assume these calls.  Files verified byte-equal to v0.5.10 by the
    tree manifest are named in the commit; model_runner.py is checked for the calls."""

    def read(self, rel):
        return (SRT / rel).read_text(encoding="utf-8")

    def test_prepare_mixed_has_the_item_sync(self):
        text = self.read("layers/attention/mamba/mamba2_metadata.py")
        body = text[text.index("def prepare_mixed"):]
        self.assertRegex(body, r"torch\.any\(has_initial_states\[:num_prefills\]\)\.item\(\)")

    def test_hybrid_metadata_reaches_the_linear_backend(self):
        text = self.read("layers/attention/hybrid_linear_attn_backend.py")
        self.assertIn("self.linear_attn_backend = linear_attn_backend", text)
        self.assertRegex(text, r"self\.attn_backend_list = \[full_attn_backend, linear_attn_backend\]")
        self.assertRegex(text, r"Mamba2Metadata\.prepare_mixed\(")

    def test_piecewise_replay_calls_the_shadowed_objects(self):
        text = self.read("model_executor/piecewise_cuda_graph_runner.py")
        body = text[text.index("def replay("):]
        self.assertIn("self.model_runner.attn_backend.init_forward_metadata(forward_batch)", body)
        self.assertIn("self.model_runner.model.forward(", body)

    def test_eager_extend_and_decode_call_the_shadowed_objects(self):
        text = self.read("model_executor/model_runner.py")
        ext = text[text.index("def forward_extend("):text.index("def forward_idle(")]
        self.assertIn("self.attn_backend.init_forward_metadata(forward_batch)", ext)
        self.assertIn("self.model.forward(", ext)
        self.assertIn("self.piecewise_cuda_graph_runner.replay(", ext)
        dec = text[text.index("def forward_decode("):text.index("def forward_extend(")]
        self.assertIn("self.decode_attn_backend.init_forward_metadata(forward_batch)", dec)
        self.assertIn("self.attn_backend.init_forward_metadata(forward_batch)", dec)

    def test_graph_replay_prepare_precedes_replay(self):
        text = self.read("model_executor/cuda_graph_runner.py")
        body = text[text.index("    def replay(\n"):]
        self.assertLess(body.index("self.replay_prepare(forward_batch, pp_proxy_tensors)"),
                        body.index("self.graphs[graph_key].replay()"))
        self.assertIn("attn_backend.init_forward_metadata_replay_cuda_graph(", text)

    def test_decode_attn_backend_group_exists(self):
        text = self.read("model_executor/cuda_graph_runner.py")
        self.assertIn("self.model_runner.decode_attn_backend_group[stream_idx]", text)


# ---------------------------------------------------------------------------
# Request-level admission records + mixed opportunities
# ---------------------------------------------------------------------------
class MixedBatch(tap.FakeNewBatch):
    def __init__(self, reqs, **kw):
        super().__init__(reqs)
        self.batch_is_full = kw.get("batch_is_full", False)

    def mix_with_running(self, running):
        self.forward_mode = SimpleNamespace(is_mixed=lambda: True, name="MIXED")


def adm_probe(mod, req_records=True, mixed_ratio=True, boot=None):
    p = mod.AdmissionProbe(None, window_s=1e9, boot=boot or {"pp_max_micro_batch_size": CAP},
                           clock=lambda: 5.0, req_records=req_records, mixed_ratio=mixed_ratio)
    p.clip_max_new_tokens = CLIP
    mod.set_probe(p)
    return p


def reqs_of(p, mod):
    return [r for r in p.records if r["event"] == mod.REQUEST_EVENT]


def admission_failures(mod):
    fails = []

    def check(name, cond, detail=None):
        if not cond:
            fails.append((name, detail))

    mod.install_wrappers(tap.PrefillAdder, tap.FakeScheduler, CLIP)
    try:
        # cap: one seat; a admitted, b blocked by the loop cap and named
        p = adm_probe(mod)
        s = tap.FakeScheduler(cap=CAP, running=CAP - 1, kv_pool=1000,
                              waiting=[tap.FakeReq("a", 10), tap.FakeReq("b", 10)], mamba_avail=1)
        s.get_new_batch_prefill()
        rows = reqs_of(p, mod)
        check("cap/n", len(rows) == 2, rows)
        if len(rows) == 2:
            a, b = rows
            check("cap/a", (a["rid"], a["outcome"], a["stop_reason"]) == ("a", "admitted", None), a)
            # 1000 free - 47 running x 16 reserved output tokens = 248, before a's own 26 are taken
            check("cap/a_rem_pre_call", a["rem_total"] == 248.0, a)
            check("cap/a_required", (a["required_tokens"], a["required_basis"]) == (26, "add_total"), a)
            check("cap/b", (b["rid"], b["outcome"], b["stop_reason"], b["stop_sub"]) == ("b", "blocked", "cap", "loop"), b)
            check("cap/b_required", (b["required_tokens"], b["required_basis"]) == (26, "queue_estimate"), b)
            check("cap/b_running", (b["running_bs"], b["t_mono"]) == (CAP - 1, 5.0), b)
            check("cap/b_not_inherited", b["stop_inherited"] is False, b)
            check("cap/decision_seq", a["decision_seq"] == b["decision_seq"] == 1, rows)

        # the next decision stops on the sticky flag: inherited cap, no adder
        s.get_new_batch_prefill()
        rows = reqs_of(p, mod)
        check("sticky/n", len(rows) == 3, rows)
        if len(rows) == 3:
            c = rows[2]
            check("sticky/fields", (c["rid"], c["stop_reason"], c["stop_sub"], c["stop_inherited"])
                  == ("b", "cap", "sticky", True), c)
            check("sticky/no_adder", c["rem_total"] is None, c)

        # kv: the first request does not fit; KV, not cap
        p = adm_probe(mod)
        s = tap.FakeScheduler(cap=CAP, running=1, kv_pool=20, waiting=[tap.FakeReq("x", 10)])
        s.get_new_batch_prefill()
        rows = reqs_of(p, mod)
        check("kv/n", len(rows) == 1, rows)
        if rows:
            x = rows[0]
            check("kv/fields", (x["rid"], x["outcome"], x["stop_reason"], x["stop_sub"])
                  == ("x", "rejected", "kv", "add"), x)
            # 20 free - 1 running x 16 reserved output tokens = 4
            check("kv/values", (x["rem_total"], x["required_tokens"]) == (4.0, 26), x)

        # ignore_eos, early size check
        p = adm_probe(mod)
        s = tap.FakeScheduler(cap=CAP, running=1, kv_pool=50, waiting=[tap.FakeReq("y", 100, ignore_eos=True)])
        s.get_new_batch_prefill()
        rows = reqs_of(p, mod)
        check("eos/fields", len(rows) == 1 and (rows[0]["stop_sub"], rows[0]["required_tokens"], rows[0]["required_basis"])
              == ("ignore_eos_early", 100, "ignore_eos_input"), rows)

        # PD-mux pre-gate: seat occupied, head request named
        p = adm_probe(mod)
        s = tap.FakeScheduler(cap=CAP, running=2, kv_pool=1000, waiting=[tap.FakeReq("h", 10)])
        s.split_prefill_batch = SimpleNamespace(reqs=[1])
        s.update_split_prefill_batch(0)
        rows = reqs_of(p, mod)
        check("seat/fields", len(rows) == 1 and (rows[0]["rid"], rows[0]["stop_reason"], rows[0]["outcome"])
              == ("h", "prefill_seat", "blocked"), rows)

        # no waiting request: nothing to name
        p = adm_probe(mod)
        s = tap.FakeScheduler(cap=CAP, running=1, waiting=[])
        s.get_new_batch_prefill()
        check("empty/none", reqs_of(p, mod) == [], reqs_of(p, mod))

        # mixed opportunity
        with mock.patch.dict(tap.ENGINE, {"ScheduleBatch": MixedBatch}):
            p = adm_probe(mod)
            s = tap.FakeScheduler(cap=CAP, running=2, kv_pool=10**6, waiting=[tap.FakeReq("m", 10)])
            s.is_mixed_chunk = True
            s.running_batch.return_logprob = False
            s.get_new_batch_prefill()
            rows = [r for r in p.records if r["event"] == mod.MIXED_EVENT]
            check("mixed/row", len(rows) == 1 and rows[0]["mixed"] is True and rows[0]["running_bs"] == 2
                  and rows[0]["forward_mode"] == "MIXED", rows)
            p = adm_probe(mod)
            s = tap.FakeScheduler(cap=CAP, running=0, kv_pool=10**6, waiting=[tap.FakeReq("n", 10)])
            s.is_mixed_chunk = True
            s.get_new_batch_prefill()
            rows = [r for r in p.records if r["event"] == mod.MIXED_EVENT]
            check("mixed/no_running", len(rows) == 1 and rows[0]["mixed"] is False and rows[0]["running_bs"] == 0, rows)
    except Exception as exc:
        fails.append(("exception", repr(exc)))
    finally:
        mod.uninstall_wrappers()
        mod.set_probe(None)
    return fails


def ratio_failures(mod):
    boot = {"event": "admission_boot", "mixed_ratio_enabled": True, "arm_enable_mixed_chunk": True}
    row = lambda t, bs, m: {"event": mod.MIXED_EVENT, "t_mono": t, "running_bs": bs, "mixed": m}  # noqa: E731
    rows = [boot, row(1.0, 3, True), row(2.0, 3, False), row(3.0, 0, False), row(4.0, 1, True), row(9.0, 5, True)]
    fails = []
    got = mod.mixed_forward_ratio(rows)
    want = (3, 4, 0.75, 1, 0)
    if (got["n_mixed"], got["n_opportunities"], got["ratio"], got["n_no_running"], got["n_outside_window"]) != want:
        fails.append(("all", got))
    got = mod.mixed_forward_ratio(rows, t_window=(1.5, 4.5))
    want = (1, 2, 0.5, 1, 2)
    if (got["n_mixed"], got["n_opportunities"], got["ratio"], got["n_no_running"], got["n_outside_window"]) != want:
        fails.append(("window", got))
    return fails


class TestAdmissionRecords(unittest.TestCase):
    def tearDown(self):
        ap.uninstall_wrappers()
        ap.set_probe(None)

    def test_scenarios(self):
        self.assertEqual(admission_failures(ap), [])

    def test_ratio_table(self):
        self.assertEqual(ratio_failures(ap), [])

    def test_ratio_none_with_no_opportunity(self):
        boot = {"event": "admission_boot", "mixed_ratio_enabled": True}
        self.assertIsNone(ap.mixed_forward_ratio([boot])["ratio"])

    def test_ratio_refuses_a_boot_without_the_flag(self):
        with self.assertRaises(ValueError):
            ap.mixed_forward_ratio([{"event": "admission_boot"}])
        with self.assertRaises(ValueError):
            ap.mixed_forward_ratio([])

    def test_no_slot_reason_is_ever_written(self):
        ap.install_wrappers(tap.PrefillAdder, tap.FakeScheduler, CLIP)
        p = adm_probe(ap, boot={"pp_max_micro_batch_size": CAP, "cap_slot_relation": "IDENTICAL"})
        for running, kv in ((CAP - 1, 1000), (1, 20), (CAP, 1000)):
            s = tap.FakeScheduler(cap=CAP, running=running, kv_pool=kv, mamba_avail=max(CAP - running, 0),
                                  waiting=[tap.FakeReq("a", 10), tap.FakeReq("b", 10)])
            s.get_new_batch_prefill()
            s.get_new_batch_prefill()
        reasons = {r["stop_reason"] for r in reqs_of(p, ap)}
        self.assertTrue({"cap", "kv"} <= reasons, reasons)
        self.assertNotIn(ap.MAMBA_SLOT, reasons)

    def test_required_tokens_uses_the_pre_call_length(self):
        snap = ap.AddSnapshot(added=True, n_before=0, rem_total=1, cur_rem=1, rem_input=1, rem_chunk=64,
                              extend_input_len=64, max_new_tokens=16, n_output=0, page_size=1)
        self.assertEqual(ap.required_tokens(snap, 1000, "budget_after_add"), (1016, "add_total"))

    def _scenario(self, req_records, mixed_ratio):
        ap.install_wrappers(tap.PrefillAdder, tap.FakeScheduler, CLIP)
        p = adm_probe(ap, req_records=req_records, mixed_ratio=mixed_ratio)
        admitted = []
        for running, kv, max_new in ((CAP - 1, 1000, 16), (1, 20, 16), (2, 10**6, 8)):
            s = tap.FakeScheduler(cap=CAP, running=running, kv_pool=kv,
                                  waiting=[tap.FakeReq("a", 10, max_new), tap.FakeReq("b", 10, max_new)])
            b = s.get_new_batch_prefill()
            admitted.append([r.rid for r in b.reqs] if b is not None else None)
            admitted.append([r.rid for r in s.waiting_queue])
            admitted.append(s.running_batch.batch_is_full)
        p.close()
        return p, admitted

    def test_flag_off_writes_no_new_record_and_same_decisions(self):
        off, dec_off = self._scenario(False, False)
        ap.uninstall_wrappers()
        on, dec_on = self._scenario(True, True)
        self.assertEqual(dec_off, dec_on)
        self.assertEqual({r["event"] for r in off.records}, {"admission_boot", "admission_window"})
        self.assertNotIn("req_records_enabled", off.records[0])
        self.assertNotIn("mixed_ratio_enabled", off.records[0])
        on_core = [r for r in on.records if r["event"] in ("admission_boot", "admission_window")]
        strip = lambda rs: [{k: v for k, v in r.items()  # noqa: E731
                             if k not in ("req_records_enabled", "mixed_ratio_enabled",
                                          "t_realtime", "t_realtime_end")} for r in rs]
        self.assertEqual(strip(off.records), strip(on_core))

    def test_flag_off_installs_nothing(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            for k in ("PDMUX_ADMISSION_PATH", ap.REQ_RECORDS_ENV, ap.MIXED_RATIO_ENV):
                os.environ.pop(k, None)
            self.assertIsNone(ap.maybe_install_admission_probe(SimpleNamespace()))
        self.assertIsNone(ap.get_probe())

    def test_env_is_strict_and_needs_a_path(self):
        for name in (ap.REQ_RECORDS_ENV, ap.MIXED_RATIO_ENV):
            with mock.patch.dict(os.environ, {name: "1"}, clear=False):
                os.environ.pop("PDMUX_ADMISSION_PATH", None)
                with self.assertRaises(ValueError):
                    ap.maybe_install_admission_probe(SimpleNamespace())
            with mock.patch.dict(os.environ, {name: "true"}, clear=False):
                with self.assertRaises(ValueError):
                    ap.maybe_install_admission_probe(SimpleNamespace())

    def test_record_failure_never_reaches_admission(self):
        ap.install_wrappers(tap.PrefillAdder, tap.FakeScheduler, CLIP)
        p = adm_probe(ap)
        real = p._emit

        def bad(event, **fields):
            if event == ap.REQUEST_EVENT:
                raise RuntimeError("probe bug")
            return real(event, **fields)

        p._emit = bad
        s = tap.FakeScheduler(cap=CAP, running=CAP - 1, kv_pool=1000,
                              waiting=[tap.FakeReq("a", 10), tap.FakeReq("b", 10)], mamba_avail=1)
        b = s.get_new_batch_prefill()
        self.assertEqual([r.rid for r in b.reqs], ["a"])
        self.assertFalse(p.req_records)
        self.assertEqual(p.n_record_errors, 1)
        self.assertEqual(p.acct.terminal.get("cap/loop"), 1)  # aggregate accounting continues


# ---------------------------------------------------------------------------
# Decode SM time-share
# ---------------------------------------------------------------------------
def share_records(mod=None):
    def row(i, d, issue, sync, dropped=0):
        return {"event": "decode_iteration", "iter_seq": i, "decode_sms": d, "t_issue_start": issue,
                "t_launch": issue + 0.001, "t_sync": sync, "dropped_events": dropped}

    return [row(1, 108, 10.000, 10.010), row(2, 44, 10.010, 10.040), row(3, 44, 10.040, 10.070),
            row(5, 108, 10.080, 10.090, dropped=2), row(6, 0, 10.090, 10.095)]


def share_failures(mod):
    got = mod.decode_sm_time_share(share_records())
    fails = []

    def check(name, cond):
        if not cond:
            fails.append((name, got))

    check("shares", abs(got["shares"][44] - 0.75) < 1e-9 and abs(got["shares"][108] - 0.25) < 1e-9)
    check("busy", abs(got["busy_ms"][44] - 60.0) < 1e-6 and abs(got["busy_ms"][108] - 20.0) < 1e-6)
    check("steps", got["n_steps"] == {44: 2, 108: 2})
    check("holes", got["n_missing_iters"] == 1)
    check("zero", got["n_decode_sms_zero"] == 1 and abs(got["decode_sms_zero_ms"] - 5.0) < 1e-6)
    check("drops", got["max_dropped_events"] == 2)
    check("coverage", abs(got["coverage"] - 85.0 / 95.0) < 1e-6)
    win = mod.decode_sm_time_share(share_records(), t_window=(10.005, 10.075))
    if win["n_steps"] != {44: 2}:
        fails.append(("window", win))
    return fails


class TestDecodeShare(LoopTestBase):
    def test_table(self):
        self.assertEqual(share_failures(dp), [])

    def test_refuses_a_boot_without_the_flag(self):
        rows = [{"event": "decode_iteration", "iter_seq": 1, "decode_sms": 44, "t_launch": 1.0, "t_sync": 2.0}]
        with self.assertRaises(ValueError):
            dp.decode_sm_time_share(rows)

    def test_resolve(self):
        r = mixin_module.resolve_phase_decode_share
        self.assertFalse(r(False, {}))
        self.assertFalse(r(True, {"PDMUX_PHASE_DECODE_SHARE": "0"}))
        self.assertTrue(r(True, {"PDMUX_PHASE_DECODE_SHARE": "1"}))
        with self.assertRaises(RuntimeError):
            r(False, {"PDMUX_PHASE_DECODE_SHARE": "1"})
        with self.assertRaises(ValueError):
            r(True, {"PDMUX_PHASE_DECODE_SHARE": "yes"})

    def _run(self, share):
        sched = self.install(tpe._build(True))
        if share:
            sched.pdmux_phase_decode_share = True
            sched._phase_share_t_issue = None
            sched._phase_share_install()
        run_loop(sched)
        return sched

    def test_flag_off_keeps_the_record_and_the_trace(self):
        off = self._run(False)
        on = self._run(True)
        keys_off = {tuple(sorted(e[2])) for e in off.events("decode_iteration")}
        keys_on = {tuple(sorted(e[2])) for e in on.events("decode_iteration")}
        self.assertEqual(len(keys_off), 1)
        self.assertNotIn("t_issue_start", next(iter(keys_off)))
        self.assertEqual({tuple(k for k in ks if k != "t_issue_start") for ks in keys_on}, keys_off)
        self.assertEqual(tpe._scheduling_trace(off), tpe._scheduling_trace(on))
        self.assertEqual(off.steps, on.steps)
        self.assertFalse(getattr(off, "pdmux_phase_decode_share", False))

    def test_flag_off_installs_no_run_batch_shadow(self):
        class Sched(mixin_module.SchedulerMultiplexMixin):
            pdmux_phase_events = True

            def run_batch(self, batch):
                return "r"

        for env in ({}, {"PDMUX_PHASE_DECODE_SHARE": "0"}):
            with mock.patch.dict(os.environ, env, clear=False):
                if not env:
                    os.environ.pop("PDMUX_PHASE_DECODE_SHARE", None)
                sched = Sched()
                sched._phase_share_init()
                self.assertFalse(sched.pdmux_phase_decode_share)
                self.assertNotIn("run_batch", vars(sched))

    def test_issue_stamp_brackets_the_step(self):
        on = self._run(True)
        rows = [dict(e[2], event="decode_iteration") for e in on.events("decode_iteration")]
        self.assertGreater(len(rows), 0)
        for f in rows:
            self.assertIsNotNone(f["t_issue_start"])
            self.assertLessEqual(f["t_issue_start"], f["t_launch"])
            self.assertLessEqual(f["t_launch"], f["t_sync"])
        got = dp.decode_sm_time_share(rows)
        self.assertEqual(got["n_kept"], len(rows))
        self.assertEqual(got["n_missing_iters"], 0)
        self.assertAlmostEqual(sum(got["shares"].values()), 1.0)

    def test_prefill_batch_does_not_stamp(self):
        sched = self.install(tpe._build(True))
        sched.pdmux_phase_decode_share = True
        sched.running_batch = SimpleNamespace()
        sched._phase_share_t_issue = None
        sched._phase_share_stamp(SimpleNamespace())
        self.assertIsNone(sched._phase_share_t_issue)
        sched._phase_share_stamp(sched.running_batch)
        self.assertIsNotNone(sched._phase_share_t_issue)

    def test_class_level_wrappers_stay_in_the_path(self):
        calls = []

        class Base(mixin_module.SchedulerMultiplexMixin):
            running_batch = None

            def run_batch(self, batch):
                calls.append("class")
                return "r"

        sched = Base()
        sched.pdmux_phase_decode_share = True
        sched._phase_share_install()
        Base.run_batch = lambda self, batch: (calls.append("later"), "r2")[1]  # installed after the shadow
        self.assertEqual(sched.run_batch(None), "r2")
        self.assertEqual(calls, ["later"])

    def test_stamp_failure_stays_inside_the_probe(self):
        sched = self.install(tpe._build(True))
        sched.pdmux_phase_decode_share = True
        del sched.running_batch  # the stamp cannot read it
        type(sched)._phase_share_stamp(sched, SimpleNamespace())
        self.assertFalse(sched.pdmux_phase_decode_share)


# ---------------------------------------------------------------------------
# Mutants of the new code: each must fail its scenario table (lesson 53)
# ---------------------------------------------------------------------------
def mutate_admission(old, new):
    src = tap.PROBE_PY.read_text(encoding="utf-8")
    if old not in src:
        raise AssertionError(f"mutant pattern missing: {old[:60]}")
    name = "admission_probe_mutant"
    mod = types.ModuleType(name)
    mod.__file__ = str(tap.PROBE_PY)
    sys.modules[name] = mod
    exec(compile(src.replace(old, new), str(tap.PROBE_PY), "exec"), mod.__dict__)
    return mod


class TestMutants(unittest.TestCase):
    INTERIOR = {
        "launch_stamped_at_model_return": ('"forward", "t_launch", "t_model_return", "n_model_calls"',
                                           '"forward", "t_model_return", "t_launch", "n_model_calls"'),
        "stamps_left_open": ("            finally:\n                tls.stamps = None\n",
                             "            finally:\n                pass\n"),
        "metadata_shadow_replay_only": ('for name in ("init_forward_metadata", "init_forward_metadata_replay_cuda_graph"):',
                                        'for name in ("init_forward_metadata_replay_cuda_graph",):'),
        "ssm_interval_on_top_backend": ('if _shadow(linear, "init_forward_metadata", "t_ssm_meta_enter", "t_ssm_meta_exit"):',
                                        'if False:'),
        "wrong_seq_on_decode": ('self._emit_interior(stamps, fields["seq"], kind,',
                                'self._emit_interior(stamps, fields["seq"] + 1, kind,'),
        "graph_launch_ignored": ('if t_launch is None and stamps.get("t_graph_launch") is not None:',
                                 'if False:'),
    }
    ADMISSION = {
        "kv_written_as_cap": ("                None if cat == CONTINUE else cat, None if cat == CONTINUE else sub, False,",
                              "                None if cat == CONTINUE else CAP, None if cat == CONTINUE else sub, False,"),
        "rem_total_read_after_the_call": ("            rem_total, input_len = pre\n",
                                          "            rem_total, input_len = float(adder.rem_total_tokens), pre[1]\n"),
        "blocked_names_the_queue_head": ("self._record_blocked(self._entry_queue, t.n_add_calls, adder,",
                                         "self._record_blocked(self._entry_queue, 0, adder,"),
        "sticky_not_recorded": ("        if cat == CAP or inherited:\n", "        if cat == CAP and not inherited:\n"),
        "required_without_clip_or_output": ("    return int(extend_input_len) + output, BASIS_ADD\n",
                                            "    return int(extend_input_len), BASIS_ADD\n"),
        "mixed_flag_inverted": ("mixed=bool(mixed), forward_mode=", "mixed=not bool(mixed), forward_mode="),
        "pregate_unrecorded": ("        if waiting and self.req_records:\n", "        if False:\n"),
    }
    RATIO = {
        "ratio_counts_empty_running": ('if int(r["running_bs"]) >= 1]', 'if int(r["running_bs"]) >= 0]'),
        "ratio_ignores_window": ('inside = [r for r in rows if lo <= float(r["t_mono"]) <= hi]', 'inside = list(rows)'),
    }
    SHARE = {
        "share_uses_host_launch": ('ms = (float(r["t_sync"]) - float(r["t_issue_start"])) * 1000.0',
                                   'ms = (float(r["t_sync"]) - float(r["t_launch"])) * 1000.0'),
        "zero_sms_in_shares": ("        if d == 0:\n", "        if False:\n"),
        "holes_not_counted": ("            n_missing += seq - prev - 1\n", "            pass\n"),
    }

    def test_interior_mutants(self):
        for name, (old, new) in self.INTERIOR.items():
            with self.subTest(mutant=name):
                self.assertTrue(interior_failures(tdp.mutate(old, new)), f"{name} survived")

    def test_admission_mutants(self):
        for name, (old, new) in self.ADMISSION.items():
            with self.subTest(mutant=name):
                self.assertTrue(admission_failures(mutate_admission(old, new)), f"{name} survived")

    def test_ratio_mutants(self):
        for name, (old, new) in self.RATIO.items():
            with self.subTest(mutant=name):
                self.assertTrue(ratio_failures(mutate_admission(old, new)), f"{name} survived")

    def test_share_mutants(self):
        for name, (old, new) in self.SHARE.items():
            with self.subTest(mutant=name):
                self.assertTrue(share_failures(tdp.mutate(old, new)), f"{name} survived")


class TestNoBlockingInNewCode(unittest.TestCase):
    def test_new_code_never_synchronizes(self):
        for path, start, end in (
            (tdp.PROBE_PY, "    # -- forward interior\n", "    # -- the wrapped call\n"),
            (tap.PROBE_PY, "    # -- request-level records", "    # -- decision bracket\n"),
        ):
            text = path.read_text(encoding="utf-8")
            body = text[text.index(start):text.index(end)]
            self.assertIsNone(re.search(r"synchronize|wait_stream|wait_event|\.item\(\)", body), path)


if __name__ == "__main__":
    unittest.main()
