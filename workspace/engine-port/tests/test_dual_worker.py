import importlib.util
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


TRACKED_MODULE_PATH = Path(__file__).parents[1] / "src" / "multiplex" / "dual_worker.py"
RUNTIME_MODULE_PATH = (
    Path(os.environ["SGLANG_ENGINE_DEV"])
    / "sglang"
    / "srt"
    / "multiplex"
    / "dual_worker.py"
    if os.environ.get("SGLANG_ENGINE_DEV")
    else TRACKED_MODULE_PATH
)
MODULE_PATH = RUNTIME_MODULE_PATH if RUNTIME_MODULE_PATH.is_file() else TRACKED_MODULE_PATH
SPEC = importlib.util.spec_from_file_location("dual_worker", MODULE_PATH)
dual_worker = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules["dual_worker"] = dual_worker
SPEC.loader.exec_module(dual_worker)


class FakeBatch:
    def __init__(self, size, batch_is_full=False):
        self.reqs = [object() for _ in range(size)]
        self.batch_is_full = batch_is_full

    def batch_size(self):
        return len(self.reqs)


class DualWorkerTest(unittest.TestCase):
    def test_phase_coordinator_keeps_prefill_and_decode_transitions_explicit(self):
        req = SimpleNamespace(rid="r0")
        coordinator = dual_worker.PhaseCoordinator()

        coordinator.register_waiting(req)
        self.assertEqual(
            coordinator.request_phase["r0"],
            dual_worker.RequestPhase.PREFILL_WAITING,
        )
        coordinator.start_prefill([req])
        coordinator.complete_prefill([req])
        coordinator.start_decode([req])
        self.assertEqual(
            coordinator.request_phase["r0"],
            dual_worker.RequestPhase.DECODE_RUNNING,
        )
        coordinator.complete([req])
        self.assertEqual(
            coordinator.request_phase["r0"], dual_worker.RequestPhase.COMPLETE
        )

    def test_phase_coordinator_prunes_requests_removed_from_scheduler(self):
        req = SimpleNamespace(rid="r0")
        other = SimpleNamespace(rid="r1")
        coordinator = dual_worker.PhaseCoordinator()
        coordinator.register_waiting(req)
        coordinator.register_waiting(other)

        coordinator.prune([req])

        self.assertIn("r0", coordinator.request_phase)
        self.assertNotIn("r1", coordinator.request_phase)

    def test_observe_scheduler_marks_finished_requests_complete(self):
        finished = SimpleNamespace(rid="done", finished=lambda: True)
        scheduler = SimpleNamespace(
            waiting_queue=[],
            split_prefill_batch=None,
            running_batch=FakeBatch(0),
        )
        scheduler.running_batch.reqs = [finished]
        state = dual_worker.DualWorkerState.from_sm_counts([(0, 108)])

        state.coordinator.start_decode([finished])
        state.observe_scheduler(scheduler, stream_index=0, now=100.0)

        self.assertEqual(
            state.coordinator.request_phase["done"],
            dual_worker.RequestPhase.COMPLETE,
        )

    def test_prefill_snapshot_is_not_derived_from_decode_batch_occupancy(self):
        req = SimpleNamespace(
            rid="r0",
            time_stats=SimpleNamespace(wait_queue_entry_time=99.0),
        )
        scheduler = SimpleNamespace(
            waiting_queue=[req],
            split_prefill_batch=None,
            running_batch=FakeBatch(8, batch_is_full=True),
        )
        state = dual_worker.DualWorkerState.from_sm_counts([(0, 108), (84, 24)])

        state.observe_scheduler(scheduler, stream_index=1, now=100.0)

        self.assertEqual(state.prefill.snapshot.queue_depth, 1)
        self.assertAlmostEqual(state.prefill.snapshot.queue_age_ms, 1000.0)
        self.assertEqual(state.decode.snapshot.active_batch_size, 8)
        self.assertEqual(
            state.prefill.snapshot.admission_block_reason, "shared_batch_capacity"
        )

    def test_arbiter_allows_two_roles_only_on_the_same_partition(self):
        arbiter = dual_worker.SharedGpuArbiter([(0, 108), (84, 24)])
        arbiter.select_partition(1)
        prefill_lease = arbiter.acquire(dual_worker.WorkerRole.PREFILL)
        decode_lease = arbiter.acquire(dual_worker.WorkerRole.DECODE)

        self.assertEqual(prefill_lease.prefill_sms, 84)
        self.assertEqual(decode_lease.decode_sms, 24)
        with self.assertRaises(RuntimeError):
            arbiter.select_partition(0)

        arbiter.release(dual_worker.WorkerRole.PREFILL)
        arbiter.release(dual_worker.WorkerRole.DECODE)
        arbiter.select_partition(0)
        self.assertEqual(arbiter.stream_index, 0)

    def test_worker_timing_metrics_are_role_local(self):
        prefill = dual_worker.PrefillWorker()
        prefill.begin_admission(now=10.0)
        prefill.finish_admission(now=10.025)
        self.assertAlmostEqual(prefill.snapshot.admission_latency_ms, 25.0)

        decode = dual_worker.DecodeWorker()
        decode.begin_step(now=20.0)
        decode.finish_step(now=20.004)
        self.assertAlmostEqual(decode.snapshot.last_tpot_ms, 4.0)
        self.assertEqual(decode.snapshot.step_count, 1)


if __name__ == "__main__":
    unittest.main()
