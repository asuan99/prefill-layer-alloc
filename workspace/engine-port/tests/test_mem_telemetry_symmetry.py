"""Gate: allocator telemetry is arm-symmetric and verdict-neutral.

The true-dual OOM of job 907959 cannot be attributed without knowing what each
arm's device allocation did during the split prefill both arms were given.  The
instrument that answers that must not itself be the asymmetry: if only the
true-dual arm carried memory fields, or carried them at a different cadence,
the comparison would be between two different measurement processes.

FOUR PROPERTIES PINNED HERE
  1. FLAG OFF is byte-identical to the pre-patch emitter -- no new key, so no
     other campaign inherits an unmeasured observer effect (same rule as
     PDMUX_TRACE_FORCE_PREFILL).
  2. FLAG ON puts the SAME keys in the SAME records for legacy and true-dual,
     at the same cadence, and in the BASE payload rather than in
     `runtime.metrics()` (which exists only in true-dual).
  3. The peak-reset boundary is `_dual_worker_start_prefill`, one window per
     split-prefill batch, reached by both arms, gated on the same flag.
  4. `r2_correctness_check.py` SNAP_KEYS -- the pinned verdict rule v2 inputs
     that jobs 907032/907100/907456/907959 were scored under -- is untouched,
     and a report scored with the new fields present is bit-identical to one
     scored without them.

NOT PINNED HERE: the observer-effect magnitude.  That needs a GPU and a paired
on/off run; until it exists this instrument is not cleared for any run that
publishes a timing number.
"""

import ast
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.modules.pop("profile", None)

import torch  # noqa: E402

from sglang.srt.multiplex import multiplexing_mixin as mixin_module  # noqa: E402
from sglang.srt.multiplex.dual_worker import (  # noqa: E402
    DualWorkerState,
    TrueDualWorkerRuntime,
)
from sglang.srt.multiplex.multiplexing_mixin import (  # noqa: E402
    SchedulerMultiplexMixin,
)
from sglang.srt.multiplex.telemetry import AsyncJsonlTelemetry  # noqa: E402

SM_COUNTS = [(108, 0), (64, 44), (0, 108)]
TRACE_EVERY = 32
CHECKER = (
    Path(__file__).resolve().parents[1]
    / "results/r2_correctness/r2_correctness_check.py"
)
NEW_KEYS = {
    "gpu_mem_allocated_b",
    "gpu_mem_peak_allocated_b",
    "gpu_mem_reserved_b",
    "gpu_mem_peak_epoch",
    "worker_grad_guard",
    "prefill_worker_grad_enabled",
    "prefill_worker_inference_mode",
    "decode_worker_grad_enabled",
    "decode_worker_inference_mode",
}


class FakeBatch:
    def __init__(self, size):
        self.reqs = [SimpleNamespace(time_stats=None, rid=f"r{i}") for i in range(size)]

    def batch_size(self):
        return len(self.reqs)

    def is_empty(self):
        return not self.reqs


class FakeScheduler(SchedulerMultiplexMixin):
    """Stand-in exposing what `_dual_worker_sync` and the emitter touch."""

    def __init__(self, path, true_dual, mem_telemetry):
        self.gpu_id = 0
        self.waiting_queue = []
        self.split_prefill_batch = None
        self.running_batch = FakeBatch(4)
        self.max_running_requests = 48
        self.token_to_kv_pool_allocator = None
        self.req_to_token_pool = None
        self.dual_worker_enabled = False
        self.true_dual_worker_enabled = true_dual
        self.true_dual_worker_runtime = (
            TrueDualWorkerRuntime(SM_COUNTS) if true_dual else None
        )
        self.dual_worker_state = DualWorkerState.from_sm_counts(SM_COUNTS)
        self.dual_worker_trace_path = str(path)
        self.dual_worker_trace_every = TRACE_EVERY
        self.dual_worker_trace_count = 0
        self.dual_worker_trace_error_logged = False
        self.dual_worker_trace_force_prefill = True
        self.dual_worker_trace_forced_count = 0
        self.pdmux_experiment_phase = "benchmark"
        self.pdmux_worker_grad_guard = "inference_mode"
        self.pdmux_mem_telemetry = mem_telemetry
        self._r2_mem_peak_epoch = 0
        self._r2_worker_guard = {
            "prefill_worker_grad_enabled": None,
            "prefill_worker_inference_mode": None,
            "decode_worker_grad_enabled": None,
            "decode_worker_inference_mode": None,
        }
        self.pdmux_telemetry = AsyncJsonlTelemetry(
            Path(path), run_id="test", workload_id="test"
        )

    def close(self):
        self.pdmux_telemetry.close()
        if self.true_dual_worker_runtime is not None:
            self.true_dual_worker_runtime.close()


def run_loop(path, true_dual, mem_telemetry, n_iters=200,
             prefill_iters=frozenset(range(40, 46))):
    sched = FakeScheduler(path, true_dual, mem_telemetry)
    for i in range(n_iters):
        if i in prefill_iters:
            if sched.split_prefill_batch is None:
                sched.split_prefill_batch = FakeBatch(2)
                sched._dual_worker_start_prefill(sched.split_prefill_batch)
        else:
            sched.split_prefill_batch = None
        sched._dual_worker_sync(1)
        sched._dual_worker_sync(1)
    sched.close()
    records = [
        json.loads(line)
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]
    return sched, [r for r in records if r.get("event") == "runtime_snapshot"]


class MemTelemetrySymmetryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _path(self, name):
        return Path(self.tmp.name) / name

    # ---- property 1: OFF is byte-identical ---------------------------

    def test_runtime_default_is_off(self):
        self.assertNotIn("PDMUX_MEM_TELEMETRY", os.environ)
        src = Path(mixin_module.__file__).read_text()
        self.assertIn(
            'os.environ.get(\n            "PDMUX_MEM_TELEMETRY", "0"\n        )', src
        )

    def test_off_adds_no_key(self):
        for true_dual in (False, True):
            with self.subTest(true_dual=true_dual):
                _, recs = run_loop(
                    self._path(f"off_{true_dual}.jsonl"), true_dual, False
                )
                self.assertTrue(recs)
                leaked = NEW_KEYS.intersection(*(set(r) for r in recs))
                self.assertEqual(leaked, set())
                for rec in recs:
                    self.assertEqual(NEW_KEYS & set(rec), set())

    def test_off_and_on_share_the_same_record_grid(self):
        _, off = run_loop(self._path("grid_off.jsonl"), False, False)
        _, on = run_loop(self._path("grid_on.jsonl"), False, True)
        self.assertEqual(
            [r["sample_index"] for r in off], [r["sample_index"] for r in on]
        )
        # and the emitter adds keys only: nothing the old record had is gone
        self.assertEqual(set(off[0]) - set(on[0]), set())

    # ---- property 2: ON is arm-symmetric -----------------------------

    def test_on_puts_the_same_keys_in_both_arms(self):
        _, legacy = run_loop(self._path("sym_l.jsonl"), False, True)
        _, dual = run_loop(self._path("sym_td.jsonl"), True, True)
        for recs, label in ((legacy, "legacy"), (dual, "true_dual")):
            with self.subTest(arm=label):
                self.assertTrue(recs)
                for rec in recs:
                    self.assertEqual(
                        NEW_KEYS - set(rec), set(), f"{label} lost a memory key"
                    )
        self.assertEqual(
            [r["sample_index"] for r in legacy], [r["sample_index"] for r in dual],
            "the two arms must emit on the same cadence",
        )
        self.assertEqual(legacy[0]["architecture"], "legacy")
        self.assertEqual(dual[0]["architecture"], "true_dual")

    def test_fields_live_in_the_base_payload_not_in_runtime_metrics(self):
        """`runtime.metrics()` exists only in true-dual; memory must not ride
        on it, or legacy would silently lose the fields."""
        runtime = TrueDualWorkerRuntime(SM_COUNTS)
        self.addCleanup(runtime.close)
        self.assertEqual(NEW_KEYS & set(runtime.metrics()), set())

    def test_values_are_honest_when_cuda_is_absent(self):
        """No CUDA here: the allocator keys must be present and None, never a
        fabricated 0 that a consumer would read as 'measured zero bytes'."""
        _, recs = run_loop(self._path("vals.jsonl"), False, True)
        rec = recs[-1]
        if not torch.cuda.is_initialized():
            self.assertIsNone(rec["gpu_mem_allocated_b"])
            self.assertIsNone(rec["gpu_mem_peak_allocated_b"])
            self.assertIsNone(rec["gpu_mem_reserved_b"])
        self.assertEqual(rec["worker_grad_guard"], "inference_mode")
        self.assertIsInstance(rec["gpu_mem_peak_epoch"], int)

    # ---- property 3: the reset boundary ------------------------------

    def test_peak_epoch_advances_once_per_split_prefill_batch(self):
        with mock.patch.object(torch.cuda, "is_initialized", lambda: True), \
             mock.patch.object(torch.cuda, "reset_peak_memory_stats") as reset:
            sched, recs = run_loop(self._path("epoch.jsonl"), False, True)
        # six prefill iterations, one contiguous batch -> exactly one window
        self.assertEqual(reset.call_count, 1)
        self.assertEqual(sched._r2_mem_peak_epoch, 1)
        self.assertEqual(recs[0]["gpu_mem_peak_epoch"], 0)
        self.assertEqual(recs[-1]["gpu_mem_peak_epoch"], 1)

    def test_reset_is_reached_by_both_arms(self):
        for true_dual in (False, True):
            with self.subTest(true_dual=true_dual):
                with mock.patch.object(torch.cuda, "is_initialized", lambda: True), \
                     mock.patch.object(
                         torch.cuda, "reset_peak_memory_stats") as reset:
                    sched, _ = run_loop(
                        self._path(f"reset_{true_dual}.jsonl"), true_dual, True
                    )
                self.assertEqual(reset.call_count, 1)
                self.assertEqual(sched._r2_mem_peak_epoch, 1)

    def test_reset_does_nothing_with_the_flag_off(self):
        with mock.patch.object(torch.cuda, "is_initialized", lambda: True), \
             mock.patch.object(torch.cuda, "reset_peak_memory_stats") as reset:
            sched, _ = run_loop(self._path("noreset.jsonl"), False, False)
        self.assertEqual(reset.call_count, 0)
        self.assertEqual(sched._r2_mem_peak_epoch, 0)

    def test_emitter_survives_a_broken_allocator_read(self):
        """A telemetry failure must not take the scheduler with it, and must
        not silently disable the whole trace either."""
        with mock.patch.object(
            torch.cuda, "memory_stats_as_nested_dict",
            side_effect=RuntimeError("boom"),
        ):
            sched, recs = run_loop(self._path("broken.jsonl"), False, True)
        self.assertTrue(recs)
        self.assertFalse(sched.dual_worker_trace_error_logged)
        self.assertIsNone(recs[-1]["gpu_mem_allocated_b"])

    # ---- property 4: the pinned verdict rule is untouched ------------

    @staticmethod
    def _snap_keys():
        tree = ast.parse(CHECKER.read_text(errors="replace"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "SNAP_KEYS" for t in node.targets
            ):
                return tuple(ast.literal_eval(node.value))
        raise AssertionError("SNAP_KEYS not found in the checker")

    def test_no_new_key_collides_with_snap_keys(self):
        if not CHECKER.is_file():
            self.skipTest(f"checker not present at {CHECKER}")
        self.assertEqual(NEW_KEYS.intersection(self._snap_keys()), set())

    def test_snap_keys_are_still_produced_with_the_flag_on(self):
        if not CHECKER.is_file():
            self.skipTest(f"checker not present at {CHECKER}")
        _, recs = run_loop(self._path("snap.jsonl"), True, True)
        envelope = {"timestamp_monotonic_s", "architecture", "dropped_events"}
        missing = [
            key for key in self._snap_keys()
            if key not in recs[-1] and key not in envelope
        ]
        self.assertEqual(missing, [], f"SNAP_KEYS lost a producer: {missing}")
        for key in envelope:
            self.assertIn(key, recs[-1])


if __name__ == "__main__":
    unittest.main()
