"""CPU fakes for driving the real `event_loop_pdmux` (shared test helper).

Not a test module (no `test` prefix, so `unittest discover` does not collect
it).  Used by test_r2_admission_latch.py and
test_true_dual_prefill_ownership.py.

WHAT IS FAKED.  Streams, CUDA events and the TP group are fakes;
`torch.cuda.stream` / `torch.cuda.device` / `torch.cuda.empty_cache` are
patched to no-ops for the duration of `run_loop`; a scripted
`recv_requests` ends the infinite loop after a fixed number of iterations.
Everything else -- the event loop, `update_split_prefill_batch`,
`adjust_stream_groups`, `_r2_decide_idx`, the true dual-worker runtime and
its arbiter -- is the real code under test.

SPLIT PREFILL IS MODELLED, NOT STUBBED.  `run_batch` on a SPLIT_PREFILL batch
follows `tp_worker.forward_batch_split_prefill` (tp_worker.py:536-546): it
builds `split_forward_batch` only when the ScheduleBatch reads
`split_index == 0`, and otherwise hands the existing one -- possibly None --
to the model forward, which then fails exactly as `model_runner._forward_raw`
does on None (`'NoneType' object has no attribute 'forward_mode'`,
model_runner.py:2819).  The forward advances the ForwardBatch's own
`split_index` by `split_forward_count` (model_runner.py:2715-2727).  Every
chunk records what the ScheduleBatch showed the forward, so a test can check
that the forward saw the batch as it was when the chunk was submitted.

MIXIN UNDER TEST.  Imported from the installed runtime tree by default, so a
stale dev tree is caught; set PDMUX_MIXIN_UNDER_TEST=/path/to/
multiplexing_mixin.py to load a specific file instead (used to show that a
test fails on pre-fix code without touching the shared dev tree).
"""

import contextlib
import importlib.util
import os
import sys
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

# test_profile_controller.py installs src/multiplex/profile.py as
# sys.modules["profile"], which shadows the stdlib module and breaks the
# sglang import chain below (see test_trace_force_prefill.py).
sys.modules.pop("profile", None)

import torch

from sglang.srt.multiplex import pdmux_context

_UNDER_TEST = os.environ.get("PDMUX_MIXIN_UNDER_TEST", "")
if _UNDER_TEST:
    _spec = importlib.util.spec_from_file_location(
        "pdmux_mixin_under_test", _UNDER_TEST
    )
    mixin_module = importlib.util.module_from_spec(_spec)
    assert _spec.loader is not None
    _spec.loader.exec_module(mixin_module)
else:
    from sglang.srt.multiplex import multiplexing_mixin as mixin_module

SchedulerMultiplexMixin = mixin_module.SchedulerMultiplexMixin
ForwardMode = mixin_module.ForwardMode
TrueDualWorkerRuntime = mixin_module.TrueDualWorkerRuntime
ExecutionContext = mixin_module.ExecutionContext
MIXIN_SOURCE = mixin_module.__file__

# The canonical R2 state set (benchmarks/configs/pdmux_r2.yml): plain
# prefill group, four whole-phase divisions, plain decode group.
SM_COUNTS = [(108, 0), (92, 16), (84, 24), (74, 34), (64, 44), (0, 108)]
MANUAL_DIVISIONS = [[92, 16, 0], [84, 24, 0], [74, 34, 0], [64, 44, 0]]
NUM_LAYERS = 4
TOKEN_BUDGET = 100  # with 100-token prompts: one layer per iteration

# Env the loop or the snapshot reads; cleared so a developer shell cannot
# change the scripted dynamics.
ENV_KEYS = (
    "PDMUX_SLO_SCHED",
    "PDMUX_SLO_SPAN_TYPE",
    "PDMUX_STICKY_PARTITION",
    "PDMUX_TPOT_SLO_MS",
    "PDMUX_TTFT_SLO_MS",
    "PDMUX_SLO_EMA",
)


class LoopDone(Exception):
    """Raised by the scripted `recv_requests` to leave the infinite loop."""


class FakeEvent:
    def query(self):
        return True

    def synchronize(self):
        return None


class FakeStream:
    def __init__(self, name):
        self.name = name
        self.syncs = 0

    def synchronize(self):
        self.syncs += 1

    def record_event(self):
        return FakeEvent()


class FakeGroup:
    def allreduce(self, _tensor, _op):
        return SimpleNamespace(wait=lambda: None)


class FakeReq:
    def __init__(self, rid, max_new, prompt_len=100):
        self.rid = rid
        self.origin_input_ids = [0] * prompt_len
        self.output_ids = []
        self.sampling_params = SimpleNamespace(max_new_tokens=max_new)
        self.time_stats = SimpleNamespace(wait_queue_entry_time=0.0)

    def finished(self):
        return len(self.output_ids) >= self.sampling_params.max_new_tokens


class FakeBatch:
    """ScheduleBatch stand-in.  Like ScheduleBatch it defines neither
    __len__ nor __bool__, so any non-None batch is truthy, and the split
    fields default as in schedule_batch.py:1395-1399."""

    def __init__(self, reqs=(), prefill=False):
        self.reqs = list(reqs)
        self.forward_mode = None
        self.split_index = 0
        self.split_forward_count = 1
        self.split_prefill_finished = False
        self.split_forward_batch = None
        self.seq_lens_cpu_cache = None
        self.extend_num_tokens = (
            sum(len(r.origin_input_ids) for r in self.reqs) if prefill else 0
        )

    def is_empty(self):
        return not self.reqs

    def batch_size(self):
        return len(self.reqs)

    def merge_batch(self, other):
        self.reqs.extend(other.reqs)


class RecordingTelemetry:
    """AsyncJsonlTelemetry's emit() surface, appending to a shared log."""

    def __init__(self, log):
        self.log = log

    def emit(self, event, phase, request_id="", **fields):
        self.log.append(("telemetry", event, dict(fields)))
        return True

    def mark_phase(self, phase):
        return self.emit("phase_marker", phase)

    def close(self, timeout_s=5.0):
        return None


class LoopScheduler(SchedulerMultiplexMixin):
    """Just enough Scheduler for `event_loop_pdmux` to run on CPU.

    `runtime_factory(sched)` (optional) builds the true dual-worker runtime;
    it receives the scheduler so it can pass `sched._activate_role_context`.
    `prefill_read_delay_s` delays the split-prefill forward BEFORE it reads
    the batch -- a stand-in for worker-thread wake-up latency.
    """

    def __init__(self, policy, policy_name, arrivals, max_iters,
                 admit_per_batch=1, max_running_requests=48,
                 runtime_factory=None, prefill_read_delay_s=0.0):
        self.gpu_id = 0
        self.sm_counts = list(SM_COUNTS)
        self.stream_groups = [
            (FakeStream(f"p{i}"), FakeStream(f"d{i}")) for i in range(len(SM_COUNTS))
        ]
        self.real_sm_group_num = len(SM_COUNTS)
        self.pdmux_config = SimpleNamespace(
            manual_divisions=[list(row) for row in MANUAL_DIVISIONS],
            split_forward_token_budget=TOKEN_BUDGET,
            decode_bs_divisor=36,
        )
        self.model_config = SimpleNamespace(num_hidden_layers=NUM_LAYERS)
        self.tp_worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                update_decode_attn_backend=lambda _idx: None,
                model=SimpleNamespace(),
                decode_attn_backend=None,
            )
        )
        self.tp_cpu_group = FakeGroup()
        self.tp_size = 1
        self.waiting_queue = []
        self.running_batch = FakeBatch()
        self.split_prefill_batch = None
        self.max_running_requests = max_running_requests
        self.token_to_kv_pool_allocator = None
        self.req_to_token_pool = None
        self.dual_worker_enabled = False
        self.dual_worker_trace_path = ""
        self.sticky_partition_enabled = False
        self._sticky_fixed_idx = None
        self.new_token_ratio = 0.0
        self.init_new_token_ratio = 0.0
        self.log = []
        self.pdmux_telemetry = RecordingTelemetry(self.log)
        self.pdmux_experiment_phase = "benchmark"
        self.r2_policy = policy
        self.r2_policy_name = policy_name
        self.r2_admission_limited = False
        self._arrivals = arrivals
        self._max_iters = max_iters
        self._admit_per_batch = admit_per_batch
        self._prefill_read_delay_s = prefill_read_delay_s
        self.iteration = -1
        self.true_dual_worker_runtime = (
            runtime_factory(self) if runtime_factory is not None else None
        )
        self.true_dual_worker_enabled = self.true_dual_worker_runtime is not None

    # --- scheduler surface used by event_loop_pdmux -----------------------
    def recv_requests(self):
        if self.iteration + 1 >= self._max_iters:
            raise LoopDone
        self.iteration += 1
        return list(self._arrivals.get(self.iteration, ()))

    def process_input_requests(self, reqs):
        self.waiting_queue.extend(reqs)

    def get_new_batch_prefill(self):
        if not self.waiting_queue:
            return None
        take = self.waiting_queue[: self._admit_per_batch]
        del self.waiting_queue[: self._admit_per_batch]
        for req in take:
            self.log.append(
                ("admit", req.rid, self.iteration, self.running_batch.batch_size())
            )
        return FakeBatch(take, prefill=True)

    def update_running_batch(self, batch):
        batch.reqs = [req for req in batch.reqs if not req.finished()]
        if batch.reqs:
            batch.forward_mode = ForwardMode.DECODE  # as prepare_for_decode does
        return batch

    def check_memory(self):
        return None

    def check_tree_cache(self):
        return None

    def maybe_sleep_on_idle(self):
        return None

    def run_batch(self, batch):
        if batch.forward_mode == ForwardMode.SPLIT_PREFILL:
            return self._forward_split_prefill(batch)
        return SimpleNamespace(batch=batch)

    def _forward_split_prefill(self, batch):
        # tp_worker.forward_batch_split_prefill, tp_worker.py:536-546
        if self._prefill_read_delay_s:
            time.sleep(self._prefill_read_delay_s)
        observed = batch.split_index
        if observed == 0:
            batch.split_forward_batch = SimpleNamespace(split_index=0)
            batch.seq_lens_cpu_cache = [len(r.origin_input_ids) for r in batch.reqs]
        forward_batch = batch.split_forward_batch
        if forward_batch is None:  # model_runner._forward_raw, model_runner.py:2819
            raise AttributeError("'NoneType' object has no attribute 'forward_mode'")
        # model_runner.forward_split_prefill, model_runner.py:2715-2727
        start = forward_batch.split_index
        forward_batch.split_index = min(start + batch.split_forward_count, NUM_LAYERS)
        self.log.append((
            "prefill_chunk", tuple(r.rid for r in batch.reqs), observed, start,
            forward_batch.split_index, threading.current_thread().name,
        ))
        return SimpleNamespace(batch=batch)

    def process_batch_result(self, batch, _result):
        # decode: one token per running request; prefill completion: first token
        for req in batch.reqs:
            if not req.finished():
                req.output_ids.append(1)

    # --- helpers for the assertions ---------------------------------------
    def admissions(self, rid):
        return [e for e in self.log if e[0] == "admit" and e[1] == rid]

    def decisions(self):
        return [e for e in self.log if e[0] == "telemetry" and e[1] == "controller_decision"]

    def events(self, name):
        return [e for e in self.log if e[0] == "telemetry" and e[1] == name]

    def prefill_chunks(self):
        return [e for e in self.log if e[0] == "prefill_chunk"]


def run_loop(sched):
    with mock.patch.object(torch.cuda, "stream", lambda _s: contextlib.nullcontext()), \
            mock.patch.object(torch.cuda, "device", lambda _d: contextlib.nullcontext()), \
            mock.patch.object(torch.cuda, "empty_cache", lambda: None):
        try:
            sched.event_loop_pdmux()
        except LoopDone:
            pass
        finally:
            runtime = getattr(sched, "true_dual_worker_runtime", None)
            if runtime is not None:
                runtime.close()
    return sched


class LoopTestBase(unittest.TestCase):
    def setUp(self):
        saved = (
            pdmux_context.STREAM_GROUPS,
            pdmux_context.CURRENT_STREAM_IDX,
            pdmux_context.CURRENT_STREAM_GROUP,
        )

        def _restore():
            (
                pdmux_context.STREAM_GROUPS,
                pdmux_context.CURRENT_STREAM_IDX,
                pdmux_context.CURRENT_STREAM_GROUP,
            ) = saved

        self.addCleanup(_restore)
        env = mock.patch.dict(os.environ, {}, clear=False)
        env.start()
        self.addCleanup(env.stop)
        for key in ENV_KEYS:
            os.environ.pop(key, None)

    def install(self, sched):
        pdmux_context.STREAM_GROUPS = sched.stream_groups
        pdmux_context.CURRENT_STREAM_IDX = 0
        pdmux_context.CURRENT_STREAM_GROUP = sched.stream_groups[0]
        return sched
