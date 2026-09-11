"""Gate for the R2 admission-limit latch (`r2_admission_limited`).

THE BUG (reports/r2_decoupling_review_2026-07-24.md, item 4).  The latch is
written only by `_r2_decide_idx`, which `event_loop_pdmux` calls only while a
split prefill batch is in flight, and it is read only by
`update_split_prefill_batch`, which reaches that check only when NO split
prefill batch is in flight.  Once the last in-span decision said "limited"
and that prefill drained, nothing could write the latch again, so prefill
admission stopped for the rest of the run.

WHY THE REAL EVENT LOOP IS DRIVEN.  The defect is in the loop's gate
structure, not in any one helper, so a test that called helpers directly
could not fail on the pre-fix code for the right reason.  These tests run the
unmodified `event_loop_pdmux` on CPU: streams, CUDA events and the TP group
are fakes, `torch.cuda.stream` is a null context, and a scripted
`recv_requests` ends the loop after a fixed number of iterations.

WHAT IS PINNED, AND WHICH VARIANT EACH TEST KILLS.
  liveness  -- after the policy stops reporting "limited", a waiting prefill
               is admitted while decode is still running.  Kills the pre-fix
               code and a backstop-only fix.
  safety    -- no prefill is admitted while the most recent policy decision
               reported "limited".  Kills a clear-on-drain "fix", which
               silently deletes the R2 admission limit.
  backstop  -- a limit the policy never lifts is released once the running
               batch is empty.  Kills the pre-fix code and a gate-only fix.
  generic   -- end to end with the real GenericDynamicPolicy and
               CoarseGrainedController: overload (running-batch occupancy
               >= 0.9) sets the limit and the drain below 0.9 lifts it.
  fixed     -- FixedPolicy never sets the limit, so the fixed-split arms
               (P1/P2, Claim D) are outside this bug both before and after
               the fix.

RUNNING AGAINST A VARIANT.  By default the mixin is imported from the
installed runtime tree (as the other mixin tests do), so a stale dev tree is
caught.  Set PDMUX_MIXIN_UNDER_TEST=/path/to/multiplexing_mixin.py to load a
specific file instead -- used to show that the tests fail on the pre-fix code
and on the rejected fix variants, without touching the shared dev tree.
"""

import contextlib
import importlib.util
import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

# See test_trace_force_prefill.py: test_profile_controller.py installs
# src/multiplex/profile.py as sys.modules["profile"], which shadows the stdlib
# module and breaks the sglang import chain below.
sys.modules.pop("profile", None)

import torch

from sglang.srt.multiplex import pdmux_context
from sglang.srt.multiplex.controller import (
    CoarseGrainedController,
    FixedPolicy,
    GenericDynamicPolicy,
    RuntimeSnapshot,
    SplitDecision,
)

_UNDER_TEST = os.environ.get("PDMUX_MIXIN_UNDER_TEST", "")
if _UNDER_TEST:
    _spec = importlib.util.spec_from_file_location(
        "pdmux_mixin_under_test", _UNDER_TEST
    )
    _mixin_module = importlib.util.module_from_spec(_spec)
    assert _spec.loader is not None
    _spec.loader.exec_module(_mixin_module)
else:
    from sglang.srt.multiplex import multiplexing_mixin as _mixin_module

SchedulerMultiplexMixin = _mixin_module.SchedulerMultiplexMixin
MIXIN_SOURCE = _mixin_module.__file__

# The canonical R2 state set (benchmarks/configs/pdmux_r2.yml): plain
# prefill group, four whole-phase divisions, plain decode group.
SM_COUNTS = [(108, 0), (92, 16), (84, 24), (74, 34), (64, 44), (0, 108)]
MANUAL_DIVISIONS = [[92, 16, 0], [84, 24, 0], [74, 34, 0], [64, 44, 0]]
NUM_LAYERS = 4
TOKEN_BUDGET = 100  # with 100-token prompts: one layer per iteration

# Env the loop or the snapshot reads; cleared so a developer shell cannot
# change the scripted dynamics.
_ENV_KEYS = (
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
    __len__ nor __bool__, so any non-None batch is truthy."""

    def __init__(self, reqs=(), prefill=False):
        self.reqs = list(reqs)
        self.forward_mode = None
        self.split_index = 0
        self.split_forward_count = 0
        self.split_prefill_finished = False
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
    """Just enough Scheduler for `event_loop_pdmux` to run on CPU."""

    def __init__(self, policy, policy_name, arrivals, max_iters,
                 admit_per_batch=1, max_running_requests=48):
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
        self.true_dual_worker_enabled = False
        self.true_dual_worker_runtime = None
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
        self.iteration = -1

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
        return batch

    def check_memory(self):
        return None

    def check_tree_cache(self):
        return None

    def maybe_sleep_on_idle(self):
        return None

    def run_batch(self, batch):
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


class ScriptedPolicy:
    """R2 policy whose `admission_limited` answer is scripted.

    `limited(sched, call_index)` decides the answer; every decision targets
    D44 (a whole-phase division), like a fixed-split arm would.
    """

    def __init__(self, sched_ref, limited):
        self._sched_ref = sched_ref
        self._limited = limited
        self.calls = 0

    def decide(self, snapshot, current_decode_sms, safe_boundary):
        limited = bool(self._limited(self._sched_ref(), self.calls))
        self.calls += 1
        return SplitDecision(
            current_decode_sms=current_decode_sms,
            target_decode_sms=44,
            reason="scripted",
            safe=safe_boundary,
            admission_limited=limited,
        )


def run_loop(sched):
    with mock.patch.object(torch.cuda, "stream", lambda _s: contextlib.nullcontext()), \
            mock.patch.object(torch.cuda, "empty_cache", lambda: None):
        try:
            sched.event_loop_pdmux()
        except LoopDone:
            pass
    return sched


def scripted_scheduler(limited, arrivals, max_iters):
    holder = {}
    policy = ScriptedPolicy(lambda: holder["s"], limited)
    sched = LoopScheduler(policy, "scripted", arrivals, max_iters)
    holder["s"] = sched
    return sched, policy


def generic_policy():
    # Zero epoch/dwell: every call is an evaluation, so the latch reflects
    # the overload streak alone (the controller's HOLD path is not exercised).
    states = (16, 24, 34, 44)
    return GenericDynamicPolicy(
        states,
        emergency_state=108,
        controller=CoarseGrainedController(
            states,
            108,
            min_epoch_s=0.0,
            min_epoch_iterations=0,
            min_dwell_s=0.0,
            min_dwell_iterations=0,
        ),
    )


class _LoopTestBase(unittest.TestCase):
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
        for key in _ENV_KEYS:
            os.environ.pop(key, None)

    def install(self, sched):
        pdmux_context.STREAM_GROUPS = sched.stream_groups
        pdmux_context.CURRENT_STREAM_IDX = 0
        pdmux_context.CURRENT_STREAM_GROUP = sched.stream_groups[0]
        return sched

    def assert_no_admission_while_limited(self, sched, after_rid):
        """Safety: from `after_rid`'s admission on, every admission must be
        preceded by a decision that did NOT report admission_limited (or by
        an explicit release)."""
        seen_anchor = False
        last_limited = None
        for entry in sched.log:
            if entry[0] == "admit" and entry[1] == after_rid:
                seen_anchor = True
                continue
            if not seen_anchor:
                continue
            if entry[0] == "telemetry" and entry[1] == "controller_decision":
                last_limited = bool(entry[2]["admission_limited"])
            elif entry[0] == "telemetry" and entry[1] == "r2_admission_released":
                last_limited = False
            elif entry[0] == "admit":
                self.assertFalse(
                    last_limited,
                    f"prefill {entry[1]} admitted at iteration {entry[2]} while the "
                    f"most recent R2 decision reported admission_limited=True "
                    f"(the admission limit was dropped, not honoured) [{MIXIN_SOURCE}]",
                )


class ScriptedLatchTest(_LoopTestBase):
    """The latch plumbing with a scripted policy (controller logic removed)."""

    # A decodes for a long time (so the running batch never empties inside
    # the window), B is the prefill whose span sets the limit, C waits on it.
    ARRIVALS = {
        0: [FakeReq("A", max_new=400)],
        6: [FakeReq("B", max_new=3)],
        8: [FakeReq("C", max_new=3)],
    }

    def test_limit_is_lifted_when_policy_stops_limiting(self):
        # Overload is reported only while a prefill competes with decode.
        sched, policy = scripted_scheduler(
            lambda s, _i: s.split_prefill_batch is not None, self.ARRIVALS, 60
        )
        run_loop(self.install(sched))

        in_span = [d for d in sched.decisions() if d[2]["admission_limited"]]
        self.assertGreaterEqual(
            len(in_span), 1, "scenario broken: the in-span policy never limited"
        )
        admitted = sched.admissions("C")
        self.assertTrue(
            admitted,
            "C was never admitted: the admission latch stayed True after the "
            "limiting prefill drained and the policy was never asked again "
            f"(stale latch) [{MIXIN_SOURCE}]",
        )
        _, _, iteration, running_at_admit = admitted[0]
        self.assertGreater(
            running_at_admit,
            0,
            "C was admitted only after decode drained -- that is the empty-batch "
            "backstop, not the policy lifting the limit",
        )
        self.assertLessEqual(iteration, 14, f"C admitted late (iteration {iteration})")
        self.assertFalse(sched.r2_admission_limited)

    def test_limit_still_refuses_while_policy_reports_overload(self):
        # Limited while B is in flight AND for the first three decode-only
        # re-evaluations after it drains; lifted afterwards.
        state = {"after_drain": 0}

        def limited(s, _i):
            if s.split_prefill_batch is not None:
                return True
            state["after_drain"] += 1
            return state["after_drain"] <= 3

        sched, _policy = scripted_scheduler(limited, self.ARRIVALS, 60)
        run_loop(self.install(sched))

        limited_in_span = [
            d for d in sched.decisions() if d[2]["admission_limited"]
        ]
        self.assertGreaterEqual(len(limited_in_span), 2, "scenario broken")
        self.assert_no_admission_while_limited(sched, after_rid="B")

    def test_empty_running_batch_releases_a_limit_the_policy_never_lifts(self):
        arrivals = {
            0: [FakeReq("A", max_new=6)],
            6: [FakeReq("B", max_new=3)],
            8: [FakeReq("C", max_new=3)],
        }
        sched, _policy = scripted_scheduler(lambda _s, _i: True, arrivals, 40)
        run_loop(self.install(sched))

        admitted = sched.admissions("C")
        self.assertTrue(
            admitted,
            "C was never admitted although the server went idle: nothing "
            f"releases the latch once decode is empty [{MIXIN_SOURCE}]",
        )
        self.assertEqual(
            admitted[0][3], 0, "C should be admitted only once decode is empty"
        )
        released = sched.events("r2_admission_released")
        self.assertEqual(len(released), 1)
        self.assertEqual(released[0][2]["reason"], "running_batch_empty")


class GenericPolicyLatchTest(_LoopTestBase):
    """End to end with the real GenericDynamicPolicy + CoarseGrainedController."""

    def test_overload_sets_the_limit_and_the_drain_lifts_it(self):
        # cap 10: nine long decodes put running-batch occupancy at 0.9, X's
        # prefill span is where the controller sees it, Y waits on the limit.
        arrivals = {
            0: [FakeReq(f"W{i}", max_new=20 + i) for i in range(9)],
            6: [FakeReq("X", max_new=50)],
            8: [FakeReq("Y", max_new=3)],
        }
        sched = LoopScheduler(
            generic_policy(), "generic", arrivals, 80,
            admit_per_batch=9, max_running_requests=10,
        )
        run_loop(self.install(sched))

        limited = [d for d in sched.decisions() if d[2]["admission_limited"]]
        self.assertGreaterEqual(
            len(limited), 1, "scenario broken: generic policy never limited"
        )
        admitted = sched.admissions("Y")
        self.assertTrue(
            admitted,
            f"Y was never admitted: stale admission latch [{MIXIN_SOURCE}]",
        )
        _, _, _, running_at_admit = admitted[0]
        self.assertGreater(
            running_at_admit, 0,
            "Y was admitted only after decode fully drained (backstop), not when "
            "the controller stopped reporting overload",
        )
        self.assertLessEqual(
            running_at_admit, 8,
            "Y admitted while running-batch occupancy was still >= 0.9",
        )
        self.assert_no_admission_while_limited(sched, after_rid="X")


class FixedPolicyTest(_LoopTestBase):
    """The fixed-split arms (P1/P2, Claim D) cannot set the latch at all."""

    def test_fixed_policy_never_reports_admission_limited(self):
        policy = FixedPolicy(44)
        for occupancy in (0.0, 0.5, 0.85, 0.9, 1.0):
            for itl in (10.0, 60.0, 500.0):
                for current in (16, 24, 34, 44, 108):
                    for safe in (True, False):
                        snapshot = RuntimeSnapshot(
                            timestamp_s=0.0, decode_iterations=0,
                            active_decode_sequences=48, decode_batch_size=48,
                            context_p50=4096, context_p95=4096, context_max=4096,
                            expected_remaining_output_p90=512,
                            prefill_queue_depth=64, oldest_prefill_age_ms=1e5,
                            measured_itl_ewma_ms=itl, measured_itl_p95_ms=itl,
                            itl_slo_ms=60.0, ttft_risk=1.0,
                            kv_occupancy=occupancy,
                            running_batch_occupancy=occupancy,
                        )
                        decision = policy.decide(snapshot, current, safe)
                        self.assertFalse(decision.admission_limited)

    def test_fixed_policy_loop_never_latches_or_rechecks(self):
        arrivals = {
            0: [FakeReq(f"W{i}", max_new=20 + i) for i in range(9)],
            6: [FakeReq("X", max_new=50)],
            8: [FakeReq("Y", max_new=3)],
        }
        sched = LoopScheduler(
            FixedPolicy(44), "fixed", arrivals, 40,
            admit_per_batch=9, max_running_requests=10,
        )
        run_loop(self.install(sched))
        self.assertTrue(sched.decisions(), "scenario broken: no R2 decision")
        self.assertFalse(any(d[2]["admission_limited"] for d in sched.decisions()))
        self.assertEqual(sched.events("r2_admission_recheck"), [])
        self.assertEqual(sched.events("r2_admission_released"), [])
        # Y waits only for X's prefill to finish (single split batch at a time).
        self.assertTrue(sched.admissions("Y"))
        self.assertLessEqual(sched.admissions("Y")[0][2], 10)


if __name__ == "__main__":
    unittest.main()
