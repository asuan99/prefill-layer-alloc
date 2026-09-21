"""Gate for PDMUX_TRACE_FORCE_PREFILL (prefill-in-flight forced telemetry).

The plain count subsampling in `_dual_worker_sync` is blind to cells whose
prefill is fast (E1/T8/d44: prefill active 4.9% of wall time, 0.07% of
sampled snapshots), which makes partition-pin verification impossible at
exactly one end of the [108-D, D] frontier.  These tests pin the three
properties the fix has to have:

  1. default OFF, and OFF is byte-identical to the pre-patch emitter
     (same record set, and no new field),
  2. ON adds records only, never removes or shifts one -- the scheduled
     grid (`sample_index == 1 or % trace_every == 0`) is unchanged, so a
     force-mode run can be re-scored exactly like a legacy run by
     filtering `trace_forced != True`,
  3. tracing is OBSERVATION: no scheduler state is mutated by the emitter.

The mixin is loaded from the installed runtime tree when SGLANG_ENGINE_DEV
is set (same convention as test_dual_worker.py), so this also checks that
sync_engine_tree.sh actually installed the patch.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

# test_profile_controller.py loads src/multiplex/profile.py as sys.modules
# ["profile"] (load-bearing: controller.py's CPU-only fallback imports it by
# that bare name), which shadows the STDLIB `profile` and makes the import
# below fail with `module 'profile' has no attribute 'run'` via
# sglang.utils -> IPython -> cProfile.  Drop the shadow before touching the
# installed runtime; nothing looks it up in sys.modules after load time.
sys.modules.pop("profile", None)

from sglang.srt.multiplex.dual_worker import DualWorkerState
from sglang.srt.multiplex.multiplexing_mixin import SchedulerMultiplexMixin
from sglang.srt.multiplex.telemetry import AsyncJsonlTelemetry

SM_COUNTS = [(108, 0), (92, 16), (0, 108)]
TRACE_EVERY = 32


class FakeBatch:
    def __init__(self, size):
        self.reqs = [SimpleNamespace(time_stats=None, rid=f"r{i}") for i in range(size)]

    def batch_size(self):
        return len(self.reqs)

    def is_empty(self):
        return not self.reqs


class FakeScheduler(SchedulerMultiplexMixin):
    """Minimal stand-in exposing only what `_dual_worker_sync` touches."""

    def __init__(self, path, force):
        self.waiting_queue = []
        self.split_prefill_batch = None
        self.running_batch = FakeBatch(4)
        self.max_running_requests = 48
        self.token_to_kv_pool_allocator = None
        self.req_to_token_pool = None
        self.dual_worker_enabled = False
        self.true_dual_worker_enabled = False
        self.true_dual_worker_runtime = None
        self.dual_worker_state = DualWorkerState.from_sm_counts(SM_COUNTS)
        self.dual_worker_trace_path = str(path)
        self.dual_worker_trace_every = TRACE_EVERY
        self.dual_worker_trace_count = 0
        self.dual_worker_trace_error_logged = False
        self.dual_worker_trace_force_prefill = force
        self.dual_worker_trace_forced_count = 0
        self.pdmux_experiment_phase = "benchmark"
        self.pdmux_telemetry = AsyncJsonlTelemetry(
            Path(path), run_id="test", workload_id="test"
        )


def run_loop(path, force, n_iters=200, prefill_iters=frozenset(range(40, 46))):
    """Drive n_iters synthetic loop iterations; prefill is in flight on some.

    Two syncs per iteration, matching event_loop_pdmux (one after
    process_input_requests, one at the end of the iteration).
    """
    sched = FakeScheduler(path, force)
    for i in range(n_iters):
        sched.split_prefill_batch = FakeBatch(2) if i in prefill_iters else None
        sched._dual_worker_sync(1)
        sched._dual_worker_sync(1)
    sched.pdmux_telemetry.close()
    records = [
        json.loads(line)
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]
    return sched, [r for r in records if r.get("event") == "runtime_snapshot"]


class TraceForcePrefillTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _path(self, name):
        return Path(self.tmp.name) / name

    def test_runtime_default_is_off(self):
        """The installed runtime must default the flag OFF (past-campaign parity)."""
        self.assertNotIn("PDMUX_TRACE_FORCE_PREFILL", os.environ)
        src = Path(
            os.environ.get("SGLANG_ENGINE_DEV")
            or os.path.join(
                os.environ.get("PDMUX_ROOT", str(Path(__file__).resolve().parents[4])),
                "sglang_engine_dev", "python",
            )
        ) / "sglang/srt/multiplex/multiplexing_mixin.py"
        text = src.read_text()
        self.assertIn('os.environ.get(\n            "PDMUX_TRACE_FORCE_PREFILL", "0"\n        )', text)

    def test_off_matches_pure_count_subsampling(self):
        _, recs = run_loop(self._path("off.jsonl"), force=False)
        idx = [r["sample_index"] for r in recs]
        expected = [
            n for n in range(1, 401) if n == 1 or n % TRACE_EVERY == 0
        ]
        self.assertEqual(idx, expected)
        # no new field in legacy mode: a run with the flag off writes exactly
        # what the pre-patch emitter wrote.
        self.assertTrue(all("trace_forced" not in r for r in recs))
        # and the bias this patch exists to fix is reproduced: prefill was in
        # flight on 6 of 200 iterations yet no sample landed inside it.
        self.assertEqual(
            sum(1 for r in recs if r["prefill_active_batch_size"] > 0), 0
        )

    def test_on_adds_records_without_disturbing_the_scheduled_grid(self):
        _, off = run_loop(self._path("a_off.jsonl"), force=False)
        sched, on = run_loop(self._path("b_on.jsonl"), force=True)
        scheduled = [r for r in on if not r["trace_forced"]]
        self.assertEqual(
            [r["sample_index"] for r in scheduled],
            [r["sample_index"] for r in off],
        )
        forced = [r for r in on if r["trace_forced"]]
        self.assertGreater(len(forced), 0)
        self.assertEqual(len(forced), sched.dual_worker_trace_forced_count)
        # every forced record is a prefill-in-flight record (the point of it)
        self.assertTrue(all(r["prefill_active_batch_size"] > 0 for r in forced))
        # sample_index is still the raw sync counter, so the sampling interval
        # of a forced record is recoverable for time weighting
        self.assertEqual(len(set(r["sample_index"] for r in on)), len(on))

    def test_forced_sampling_makes_prefill_windows_observable(self):
        _, off = run_loop(self._path("c_off.jsonl"), force=False)
        _, on = run_loop(self._path("c_on.jsonl"), force=True)
        seen_off = sum(1 for r in off if r["prefill_active_batch_size"] > 0)
        seen_on = sum(1 for r in on if r["prefill_active_batch_size"] > 0)
        self.assertEqual(seen_off, 0)
        # 6 prefill iterations x 2 syncs, minus any that already fell on the grid
        self.assertGreaterEqual(seen_on, 10)

    def test_tracing_does_not_mutate_scheduler_state(self):
        for force in (False, True):
            sched = FakeScheduler(self._path(f"d_{force}.jsonl"), force)
            batch = FakeBatch(2)
            sched.split_prefill_batch = batch
            running = sched.running_batch
            queue = sched.waiting_queue
            for _ in range(TRACE_EVERY * 3):
                sched._dual_worker_sync(1)
            sched.pdmux_telemetry.close()
            self.assertIs(sched.split_prefill_batch, batch)
            self.assertIs(sched.running_batch, running)
            self.assertIs(sched.waiting_queue, queue)
            self.assertEqual(len(batch.reqs), 2)
            # counter advances once per sync regardless of the flag: the
            # scheduled grid cannot drift between arms.
            self.assertEqual(sched.dual_worker_trace_count, TRACE_EVERY * 3)


if __name__ == "__main__":
    unittest.main()
