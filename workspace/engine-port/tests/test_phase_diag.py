"""Gate for PDMUX_PHASE_DIAG (diag Job instrumentation, default OFF; decision 40, 2026-10-28).

Requirement: results/lockstep_diag/DIAG_lockstep_stall_share_2026-10-27.md sec 5 and
PREREG_DIAG_stall_drain_rev0_2026-10-28.md P-1 (i)-(v).

The real `event_loop_pdmux` is driven on CPU through pdmux_loop_fakes (legacy loop).
CUDA events are fakes whose completion LAGS the host (an event completes only after
`lag` further records), so the non-blocking resolution path is exercised, and whose
`synchronize` / `wait` raise, so any blocking wait is a test failure.  What is pinned:

  resolve       unset / "0" -> OFF; only "1" accepted; "1" needs PDMUX_PHASE_EVENTS=1;
                refused with PDMUX_TRUE_DUAL_WORKER; accepted beside PDMUX_DECODE_PROBE_PATH
                (item iv: the decode probe is a separate, independent instrument).
  OFF           `_init_phase_diag` OFF creates exactly one attribute; no diag record, no
                `rids`; PHASE_EVENTS records byte-equal to a pre-patch run's; scheduling
                trace and stream synchronize counts identical to the ON run.
  ON (i)        one prefill_chunk per legacy chunk, split_index / forward_count / span_seq
                match what the forward saw, chunk_gpu_ms == the event pair's elapsed time,
                both events recorded on the chunk's own prefill stream.
  ON (ii)       one drain per drain branch taken; separate prefill/decode sync host ms;
                partition chain consistent; unresolved-chunk count at drain start equals
                the count reconstructed from chunk records.
  ON (iii)      prefill_span_start.rids == the admitted batch.
  no blocking   no synchronize / wait / .item() is called by the instrument (fakes raise),
                and a source scan of the diag block finds none.
  isolation     a failing event factory disables the diag only; scheduling unchanged.

Negative controls (lesson 53): `MutantTest` runs this module against mutated copies of
the mixin (PDMUX_MIXIN_UNDER_TEST, subprocess) and requires each to FAIL; an unmutated
copy must PASS (lesson 254).
"""

import os
import re
import subprocess
import sys
import tempfile
import time
import unittest
import unittest.mock
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pdmux_loop_fakes import (  # noqa: E402
    MIXIN_SOURCE,
    FakeReq,
    FakeStream,
    LoopScheduler,
    LoopTestBase,
    mixin_module,
    run_loop,
)

PHASE = ("prefill_span_start", "prefill_span_end", "decode_iteration")
DIAG = ("prefill_chunk", "drain", "phase_diag_boot", "phase_diag_close")


class BlockingCall(AssertionError):
    pass


class FakeGPU:
    """Event completion lags the host by `lag` records (a GPU running behind)."""

    def __init__(self, lag=0):
        self.lag = lag
        self.n_records = 0
        self.n_queries = 0
        self.record_log = []   # (event, stream) in record order
        self.forced_upto = 0   # stream order: a later event observed complete completes every earlier record


class FakeTimedEvent:
    def __init__(self, gpu):
        self.gpu = gpu
        self.stream = None
        self.at = None
        self.idx = None

    def record(self, stream=None):
        self.gpu.n_records += 1
        self.idx = self.gpu.n_records
        self.stream = stream
        self.at = time.perf_counter()
        self.gpu.record_log.append((self, stream))

    def query(self):
        self.gpu.n_queries += 1
        return self.idx is not None and (self.gpu.n_records - self.idx >= self.gpu.lag
                                         or self.idx <= self.gpu.forced_upto)

    def elapsed_time(self, end):
        if not end.query():
            raise BlockingCall("elapsed_time before completion")
        return (end.at - self.at) * 1000.0 + 0.25  # +0.25 ms: distinguishable from host ms

    def synchronize(self):
        raise BlockingCall("instrument blocked on an event")

    def wait(self, *_a, **_k):
        raise BlockingCall("instrument added a stream wait")


class ExeEvent:
    """`prefill_stream.record_event()` of the loop.  Its query() is True (as the shared fake's) and, being later in
    stream order, completes every chunk event recorded before it -- the real-GPU semantics that make all chunks of a
    span resolvable at prefill_span_end.  It does not count as a record, so OFF and ON dynamics stay identical."""

    def __init__(self, gpu):
        self.gpu, self.upto = gpu, gpu.n_records

    def query(self):
        self.gpu.forced_upto = max(self.gpu.forced_upto, self.upto)
        return True

    def synchronize(self):
        return None


class SlowStream(FakeStream):
    """Prefill streams sleep in synchronize so the two drain timings are distinguishable."""

    def __init__(self, name, delay_s, gpu=None):
        super().__init__(name)
        self.delay_s = delay_s
        self.gpu = gpu

    def record_event(self):
        return ExeEvent(self.gpu) if self.gpu is not None else super().record_event()

    def synchronize(self):
        self.syncs += 1
        if self.delay_s:
            time.sleep(self.delay_s)


def _arrivals():
    arr = {}
    rid = 0
    for it in (0, 1, 3, 6, 7, 12, 20, 21, 22, 30):
        arr.setdefault(it, []).append(FakeReq(f"r{rid}", max_new=6 + (rid % 4)))
        rid += 1
    return arr


def _build(phase_on, diag_on, lag=0, admit=1, prefill_sync_delay_s=0.0, slo=False):
    sched = LoopScheduler(policy=None, policy_name="", arrivals=_arrivals(),
                          max_iters=60, admit_per_batch=admit, max_running_requests=48)
    sched.gpu = FakeGPU(lag)
    sched.stream_groups = [
        (SlowStream(f"p{i}", prefill_sync_delay_s, sched.gpu), FakeStream(f"d{i}"))
        for i in range(len(sched.sm_counts))
    ]
    sched.pdmux_phase_events = phase_on
    sched._phase_span_seq = 0
    sched._phase_events_error_logged = False
    sched._phase_decode_mark = None
    if diag_on:
        sched._phase_diag_setup(lambda _g=sched.gpu: FakeTimedEvent(_g))
    else:
        sched.pdmux_phase_diag = False
    sched.steps = []
    real_run = sched.run_batch

    def run_batch(batch, _real=real_run, _s=sched):
        mode = "prefill" if batch.forward_mode == mixin_module.ForwardMode.SPLIT_PREFILL else "decode"
        _s.steps.append((_s.iteration, mode))
        return _real(batch)

    sched.run_batch = run_batch
    sched.backend_switches = []
    sched.tp_worker.model_runner.update_decode_attn_backend = sched.backend_switches.append
    if slo:
        # PDMUX_SLO_SCHED path: a scripted controller that alternates its target, so
        # the `_tgt != stream_idx` drain branch is taken repeatedly.
        targets = [1, 4]
        state = {"n": 0}

        def decide(_s=sched, _t=targets, _st=state):
            _st["n"] += 1
            return _t[(_st["n"] // 2) % 2]

        sched._slo_decide_idx = decide
    return sched


def _scheduling_trace(sched):
    return [e for e in sched.log if not (e[0] == "telemetry" and e[1] in PHASE + DIAG)]


def _phase_records_without_rids(sched):
    out = []
    for e in sched.log:
        if e[0] == "telemetry" and e[1] in PHASE:
            f = {k: v for k, v in e[2].items() if k != "rids"}
            out.append((e[1], f))
    return out


def _strip_times(recs):
    keys = ("t_launch", "t_sync")
    return [(n, {k: v for k, v in f.items() if k not in keys}) for n, f in recs]


def _diag_block_source():
    text = Path(MIXIN_SOURCE).read_text()
    start = text.index("# PDMUX_PHASE_DIAG  (default OFF")
    end = text.index("# Multi-GPU (TP>1 / DP>1) wiring", start)   # the diag block ends where M2 begins
    return text[start:end]


class ResolveTest(unittest.TestCase):
    def test_default_off_and_refusals(self):
        r = mixin_module.resolve_phase_diag
        ok = {"PDMUX_PHASE_EVENTS": "1"}
        self.assertFalse(r("", {}))
        self.assertFalse(r("/x.jsonl", {}))
        self.assertFalse(r("/x.jsonl", dict(ok)))                      # PHASE on, DIAG unset
        self.assertFalse(r("/x.jsonl", dict(ok, PDMUX_PHASE_DIAG="0")))
        self.assertTrue(r("/x.jsonl", dict(ok, PDMUX_PHASE_DIAG="1")))
        for bad in ("true", "yes", "on", "2", "True", "1 1"):
            with self.assertRaises(ValueError, msg=bad):
                r("/x.jsonl", dict(ok, PDMUX_PHASE_DIAG=bad))
        with self.assertRaises(RuntimeError):                          # needs PHASE_EVENTS
            r("/x.jsonl", {"PDMUX_PHASE_DIAG": "1"})
        with self.assertRaises(RuntimeError):                          # needs a telemetry path
            r("", dict(ok, PDMUX_PHASE_DIAG="1"))
        with self.assertRaises(RuntimeError):                          # lockstep only
            r("/x.jsonl", dict(ok, PDMUX_PHASE_DIAG="1", PDMUX_TRUE_DUAL_WORKER="1"))
        with self.assertRaises(RuntimeError):                          # inherited LA_COORD refusal
            r("/x.jsonl", dict(ok, PDMUX_PHASE_DIAG="1", PDMUX_LA_COORD="1"))

    def test_coexists_with_decode_probe(self):
        # item (iv): I-4 decode_fwd_gpu_ms comes from the decode probe; the two
        # instruments read disjoint env and neither refuses the other.
        env = {"PDMUX_PHASE_EVENTS": "1", "PDMUX_PHASE_DIAG": "1",
               "PDMUX_DECODE_PROBE_PATH": "/x/decode.jsonl"}
        self.assertTrue(mixin_module.resolve_phase_diag("/x.jsonl", env))
        probe_src = (Path(MIXIN_SOURCE).parent / "decode_probe.py").read_text()
        env_read = r"environ(?:\.get\(|\[)\s*[\"']"
        self.assertIsNone(re.search(env_read + "PDMUX_PHASE", probe_src))
        self.assertIsNone(re.search(env_read + "PDMUX_DECODE_PROBE", _diag_block_source()))
        self.assertIsNone(re.search(r"get\(\s*[\"']PDMUX_DECODE_PROBE", _diag_block_source()))

    def test_init_off_creates_one_attribute(self):
        class Bare:
            pass

        b = Bare()
        with unittest.mock.patch.dict(os.environ, {"PDMUX_PHASE_EVENTS": "1"}, clear=False):
            os.environ.pop("PDMUX_PHASE_DIAG", None)
            mixin_module.SchedulerMultiplexMixin._init_phase_diag(b, "/x.jsonl")
        self.assertEqual(vars(b), {"pdmux_phase_diag": False})

    def test_installed_runtime_has_the_hooks(self):
        text = Path(MIXIN_SOURCE).read_text()
        for needle in (
            "self.pdmux_phase_events = resolve_phase_events(trace_path); self._init_phase_diag(trace_path)",
            "prefill_result = self.run_batch(self.split_prefill_batch); self._phase_chunk_end(_pdc)",
            'self._phase_on_drain("adjust", stream_idx)',
            'self._phase_on_drain("slo_switch", _tgt)',
            "def resolve_phase_diag",
        ):
            self.assertIn(needle, text)


class DiagLoopTest(LoopTestBase):
    def _run(self, phase_on, diag_on, **kw):
        slo = kw.pop("slo", False)
        if slo:
            os.environ["PDMUX_SLO_SCHED"] = "1"
        sched = self.install(_build(phase_on, diag_on, slo=slo, **kw))
        run_loop(sched)
        return sched

    # -- OFF -------------------------------------------------------------
    def test_off_is_inert_and_on_does_not_steer(self):
        for slo in (False, True):
            with self.subTest(slo=slo):
                off = self._run(True, False, lag=3, slo=slo)
                on = self._run(True, True, lag=3, slo=slo)
                self.assertEqual([e for e in off.log if e[0] == "telemetry" and e[1] in DIAG], [])
                self.assertFalse(any("rids" in e[2] for e in off.events("prefill_span_start")))
                self.assertEqual(off.gpu.n_records, 0)
                self.assertGreater(len(on.events("prefill_chunk")), 0)
                self.assertEqual(_scheduling_trace(off), _scheduling_trace(on))
                self.assertEqual(off.steps, on.steps)
                self.assertEqual(off.backend_switches, on.backend_switches)
                # no new synchronisation: every stream synchronised exactly as often
                self.assertEqual(
                    [(p.syncs, d.syncs) for p, d in off.stream_groups],
                    [(p.syncs, d.syncs) for p, d in on.stream_groups])
                # PHASE_EVENTS content is unchanged apart from the added rids
                self.assertEqual(_strip_times(_phase_records_without_rids(off)),
                                 _strip_times(_phase_records_without_rids(on)))

    def test_phase_off_diag_absent_matches_pre_patch_shape(self):
        both_off = self._run(False, False)
        self.assertEqual([e for e in both_off.log if e[0] == "telemetry"
                          and e[1] in PHASE + DIAG], [])

    # -- (i) prefill_chunk ------------------------------------------------
    def test_one_chunk_record_per_chunk_with_gpu_time(self):
        on = self._run(True, True, lag=3)
        chunks = on.prefill_chunks()               # what the fake forward saw
        recs = sorted((e[2] for e in on.events("prefill_chunk")), key=lambda f: f["chunk_seq"])
        close = on.events("phase_diag_close")
        self.assertEqual(close, [])                # loop ends by exception, not close
        on._phase_diag_close()
        close = on.events("phase_diag_close")[-1][2]
        self.assertEqual(close["n_chunks_issued"], len(chunks))
        self.assertEqual(close["n_chunks_resolved"] + close["n_chunks_unresolved"], len(chunks))
        self.assertEqual(close["n_chunks_dropped"], 0)
        self.assertEqual(len(recs), close["n_chunks_resolved"])
        self.assertGreater(len(recs), 0)
        self.assertEqual([f["chunk_seq"] for f in recs], list(range(1, len(recs) + 1)))
        spans = {e[2]["span_seq"]: e[2] for e in on.events("prefill_span_start")}
        for f, c in zip(recs, chunks):
            _, rids, observed, start, end, _thread = c
            self.assertEqual(f["split_index"], observed)
            self.assertEqual(f["forward_count"], end - start)
            self.assertEqual(f["n_reqs"], len(rids))
            self.assertEqual(spans[f["span_seq"]]["rids"], list(rids))
            self.assertEqual(f["num_layers"], on.model_config.num_hidden_layers)
            self.assertLessEqual(f["t_issue_start"], f["t_issue_end"])
            self.assertLessEqual(f["t_issue_end"], f["t_resolved_host"])
            self.assertIsNotNone(f["chunk_gpu_ms"])
            self.assertAlmostEqual(f["chunk_gpu_ms"], f["issue_host_ms"] + 0.25, delta=0.05)
            self.assertEqual((f["prefill_sms"], f["decode_sms"]), tuple(on.sm_counts[f["stream_idx"]]))
        # GPU lag really exercised the deferred path: some chunk resolved after a later issue
        later = [f for f in recs if any(g["t_issue_start"] > f["t_issue_end"]
                                        and g["t_issue_start"] < f["t_resolved_host"] for g in recs)]
        self.assertGreater(len(later), 0)

    def test_events_recorded_on_the_chunks_prefill_stream(self):
        on = self._run(True, True, lag=0)
        recs = sorted((e[2] for e in on.events("prefill_chunk")), key=lambda f: f["chunk_seq"])
        self.assertGreater(len(recs), 0)
        self.assertEqual(len(recs), len(on.prefill_chunks()))
        self.assertTrue(len({r["stream_idx"] for r in recs}) > 1, "fixture must use >1 partition")
        log = on.gpu.record_log
        self.assertEqual(len(log), 2 * len(recs))          # only the chunk pairs record events
        for k, f in enumerate(recs):
            want = on.stream_groups[f["stream_idx"]][0]
            self.assertIs(log[2 * k][1], want)
            self.assertIs(log[2 * k + 1][1], want)

    def test_chunk_pairs_with_decode_iteration(self):
        on = self._run(True, True)
        dec = {e[2]["iter_seq"]: e[2] for e in on.events("decode_iteration")}
        recs = [e[2] for e in on.events("prefill_chunk")]
        paired = [f for f in recs if f["decode_launched"]]
        self.assertGreater(len(paired), 0)
        for f in paired:
            d = dec[f["decode_iter_seq"]]
            self.assertTrue(d["prefill_in_flight"])
            self.assertLessEqual(d["t_launch"], f["t_issue_start"])
            self.assertLessEqual(f["t_issue_end"], d["t_sync"])
        unpaired = [f for f in recs if not f["decode_launched"]]
        for f in unpaired:
            d = dec.get(f["decode_iter_seq"])
            if d is not None:
                self.assertLess(d["t_sync"], f["t_issue_start"])

    # -- (ii) drain -------------------------------------------------------
    def _check_drains(self, on, reasons):
        drains = [e[2] for e in on.events("drain")]
        self.assertGreater(len(drains), 0)
        self.assertEqual(len(drains), len(on.backend_switches))     # one per branch taken
        self.assertEqual([d["drain_seq"] for d in drains], list(range(1, len(drains) + 1)))
        self.assertEqual({d["reason"] for d in drains}, set(reasons))
        prev = 0
        for d, to in zip(drains, on.backend_switches):
            self.assertEqual(d["stream_idx_from"], prev)            # partition chain
            self.assertEqual(d["stream_idx_to"], to)
            prev = to
            self.assertEqual((d["prefill_sms_from"], d["decode_sms_from"]),
                             tuple(on.sm_counts[d["stream_idx_from"]]))
            self.assertEqual((d["prefill_sms_to"], d["decode_sms_to"]),
                             tuple(on.sm_counts[d["stream_idx_to"]]))
            self.assertLessEqual(d["t_drain_start"], d["t_prefill_sync_end"])
            self.assertLessEqual(d["t_prefill_sync_end"], d["t_drain_end"])
            self.assertAlmostEqual(d["prefill_sync_ms"],
                                   (d["t_prefill_sync_end"] - d["t_drain_start"]) * 1000.0)
            if d["split_present"]:
                self.assertTrue(0 <= d["split_index"] <= d["num_layers"])
            else:
                self.assertIsNone(d["split_index"])
            if d["reason"] == "slo_switch":
                self.assertEqual(d["controller"], "slo_sched")
                self.assertNotEqual(d["stream_idx_from"], d["stream_idx_to"])
            else:
                self.assertIsNone(d["controller"])
        return drains

    def test_adjust_drains_time_each_sync_separately(self):
        on = self._run(True, True, lag=3, prefill_sync_delay_s=0.004)
        drains = self._check_drains(on, {"adjust"})
        for d in drains:
            self.assertGreaterEqual(d["prefill_sync_ms"], 3.5)    # the sleeping prefill sync
            self.assertLess(d["decode_sync_ms"], d["prefill_sync_ms"])

    def test_slo_switch_drains(self):
        on = self._run(True, True, lag=3, slo=True)
        drains = self._check_drains(on, {"adjust", "slo_switch"})
        self.assertTrue(any(d["reason"] == "slo_switch" and d["split_present"] for d in drains))

    def test_unresolved_chunks_at_drain_match_chunk_records(self):
        for lag in (1, 3, 5):
            with self.subTest(lag=lag):
                on = self._run(True, True, lag=lag, slo=True)
                resolved = [e[2] for e in on.events("prefill_chunk")]
                never = [item.fields for item in on._phase_diag_pending]   # still in flight at loop end
                self.assertEqual(len(resolved) + len(never), on._phase_chunk_seq)
                drains = [e[2] for e in on.events("drain")]
                if lag >= 3:
                    self.assertTrue(any(d["prefill_chunks_unresolved"] > 0 for d in drains),
                                    "fixture must leave chunks in flight at some drain")
                for d in drains:
                    t0 = d["t_drain_start"]
                    expect = sum(1 for f in resolved if f["t_issue_end"] <= t0 < f["t_resolved_host"])
                    expect += sum(1 for f in never if f["t_issue_end"] <= t0)
                    self.assertEqual(d["prefill_chunks_unresolved"], expect)

    # -- (iii) rids -------------------------------------------------------
    def test_span_start_carries_rids(self):
        on = self._run(True, True, admit=2)
        starts = on.events("prefill_span_start")
        self.assertGreater(len(starts), 0)
        admitted = [e[1] for e in on.log if e[0] == "admit"]
        self.assertEqual([r for s in starts for r in s[2]["rids"]], admitted)
        for s in starts:
            self.assertEqual(len(s[2]["rids"]), s[2]["n_new"])

    def test_boot_record_once_with_clock_pair(self):
        on = self._run(True, True)
        boots = on.events("phase_diag_boot")
        self.assertEqual(len(boots), 1)
        b = boots[0][2]
        for k in ("t_perf", "t_monotonic", "t_realtime", "perf_counter_impl", "num_layers",
                  "split_forward_token_budget"):
            self.assertIn(k, b)
        diag_idx = [i for i, e in enumerate(on.log) if e[0] == "telemetry" and e[1] in DIAG]
        self.assertEqual(on.log[diag_idx[0]][1], "phase_diag_boot")
        for e in on.log:
            if e[0] == "telemetry" and e[1] in DIAG:
                self.assertEqual(e[2]["record_schema"], mixin_module.PHASE_DIAG_SCHEMA)

    # -- no blocking ------------------------------------------------------
    def test_never_blocks(self):
        # fakes raise on synchronize / wait / premature elapsed_time; any such call
        # would disable the diag (caught) -- so "still enabled" + records == no call.
        on = self._run(True, True, lag=7, slo=True)
        self.assertTrue(on.pdmux_phase_diag)
        self.assertFalse(on._phase_diag_error_logged)
        self.assertGreater(on.gpu.n_queries, 0)

    def test_source_has_no_blocking_call(self):
        src = _diag_block_source()
        code = "\n".join(l.split("#", 1)[0] for l in src.splitlines())
        code = re.sub(r'"""[\s\S]*?"""', "", code)
        for pat in (r"\.synchronize\(", r"\.item\(", r"\.wait\(", r"\.wait_event\(",
                    r"\.wait_stream\(", r"cuda\.synchronize"):
            self.assertIsNone(re.search(pat, code), pat)

    # -- isolation --------------------------------------------------------
    def test_event_failure_disables_diag_only(self):
        base = self._run(True, False)
        sched = self.install(_build(True, True))

        def boom():
            raise RuntimeError("no events")

        sched._phase_diag_event_factory = boom
        run_loop(sched)
        self.assertFalse(sched.pdmux_phase_diag)
        self.assertTrue(sched._phase_diag_error_logged)
        self.assertTrue(sched.pdmux_phase_events)
        self.assertEqual(_scheduling_trace(base), _scheduling_trace(sched))
        self.assertEqual(len(sched.events("decode_iteration")), len(base.events("decode_iteration")))


# ------------------------------------------------------------------- mutants
_MUTANTS = {
    "chunk_hook_removed": (
        "_pdc = self._phase_chunk_begin(prefill_stream, stream_idx, forward_count); prefill_result = self.run_batch(self.split_prefill_batch); self._phase_chunk_end(_pdc)",
        "prefill_result = self.run_batch(self.split_prefill_batch)",
    ),
    "adjust_drain_begin_removed": (
        "\n                self._phase_drain_begin(stream_idx, wait_prefill_kernel_done); prefill_stream.synchronize(); self._phase_drain_mark()",
        "\n                prefill_stream.synchronize(); self._phase_drain_mark()",
    ),
    "slo_drain_emit_removed": (
        'set_current_stream_idx(_tgt); self._phase_on_drain("slo_switch", _tgt)',
        "set_current_stream_idx(_tgt)",
    ),
    "decode_sync_mark_removed": (
        "                decode_stream.synchronize(); self._phase_drain_mark()  # PDMUX_PHASE_DIAG\n                if getattr(self, \"true_dual_worker_runtime\", None) is not None:\n                    if not self.true_dual_worker_runtime.arbiter.safe_to_switch():\n                        raise RuntimeError(\n                            \"unsafe automatic",
        "                decode_stream.synchronize()\n                if getattr(self, \"true_dual_worker_runtime\", None) is not None:\n                    if not self.true_dual_worker_runtime.arbiter.safe_to_switch():\n                        raise RuntimeError(\n                            \"unsafe automatic",
    ),
    "flag_forced_on": (
        '    raw = (env.get("PDMUX_PHASE_DIAG", "") or "").strip()\n',
        '    raw = (env.get("PDMUX_PHASE_DIAG", "1") or "1").strip()\n',
    ),
    "rids_dropped": (
        '            fields["rids"] = [str(req.rid) for req in batch.reqs]\n',
        "            pass\n",
    ),
    "blocking_end_event": (
        "            pair[1].record(stream)\n",
        "            pair[1].record(stream); pair[1].synchronize()\n",
    ),
    "gpu_ms_is_host_ms": (
        '                item.fields["chunk_gpu_ms"] = gpu_ms\n',
        '                item.fields["chunk_gpu_ms"] = item.fields["issue_host_ms"]\n',
    ),
    "unresolved_counted_after_drain": (
        "                prefill_chunks_unresolved=len(self._phase_diag_pending),\n",
        "                prefill_chunks_unresolved=0,\n",
    ),
}


def _run_self_against(path):
    env = dict(os.environ, PDMUX_MIXIN_UNDER_TEST=str(path), PHASE_DIAG_NO_MUTANTS="1",
               PYTHONDONTWRITEBYTECODE="1")
    proc = subprocess.run(
        [sys.executable, "-m", "unittest", "-q", "test_phase_diag"],
        cwd=os.path.dirname(os.path.abspath(__file__)), env=env,
        capture_output=True, text=True, timeout=600,
    )
    return proc.returncode, proc.stderr[-2000:]


@unittest.skipIf(os.environ.get("PHASE_DIAG_NO_MUTANTS"), "inside a mutant run")
class MutantTest(unittest.TestCase):
    def test_mutants_are_killed(self):
        src = Path(MIXIN_SOURCE).read_text()
        # the mixin's loose-file fallback imports decode_probe as a sibling module
        with tempfile.TemporaryDirectory() as td:
            sib = Path(MIXIN_SOURCE).parent / "decode_probe.py"
            (Path(td) / "decode_probe.py").write_text(sib.read_text())
            control = Path(td) / "control.py"
            control.write_text(src)
            rc, err = _run_self_against(control)
            self.assertEqual(rc, 0, "unmutated copy must pass:\n" + err)
            for name, (old, new) in _MUTANTS.items():
                self.assertEqual(src.count(old), 1, f"{name}: anchor not unique/absent")
                p = Path(td) / f"{name}.py"
                p.write_text(src.replace(old, new))
                rc, _ = _run_self_against(p)
                self.assertNotEqual(rc, 0, f"mutant {name} SURVIVED")


if __name__ == "__main__":
    unittest.main()
