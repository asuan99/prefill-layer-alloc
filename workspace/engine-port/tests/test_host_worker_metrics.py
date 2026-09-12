"""Harness defects H2-H5 in the dual-worker telemetry fields.

The repairs are documented in the NOTE at the end of `src/multiplex/dual_worker.py`.
Each test below pins one of them, and the SNAP_KEYS test pins the constraint
that made them awkward: `results/r2_correctness/r2_correctness_check.py` scores
jobs 907032 / 907100 / 907456 on a fixed set of telemetry keys, so a repair may
add names but may never remove or redefine one of those.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib.util
import os
import sys
import tempfile
import threading
import unittest
from dataclasses import asdict
from pathlib import Path

TRACK_ROOT = Path(__file__).resolve().parents[1]
CHECKER = TRACK_ROOT / "results" / "r2_correctness" / "r2_correctness_check.py"


SOURCE = TRACK_ROOT / "src" / "multiplex"


def _load(name: str):
    """Load a tracked multiplex module standalone, under its own bare name.

    The bare name matters: `controller.py`'s CPU-only fallback does
    `from profile import ...`, so `sys.modules["profile"]` has to be this tree's
    profile module while controller loads.  That shadows the stdlib `profile`
    and would break a later `import sglang` in the same `unittest discover`
    run, so the shadow is undone below -- the same dance
    test_controller_defaults.py documents.
    """
    spec = importlib.util.spec_from_file_location(name, SOURCE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_had_profile = sys.modules.get("profile")
_load("profile")
controller = _load("controller")
if _had_profile is not None:
    sys.modules["profile"] = _had_profile
else:
    sys.modules.pop("profile", None)

# dual_worker has no intra-package imports, so it loads cleanly on its own.
# Prefer the INSTALLED copy when SGLANG_ENGINE_DEV is set, the way
# test_dual_worker.py does, so a stale sync is caught here too.
_dev = os.environ.get("SGLANG_ENGINE_DEV")
_dev_dual = Path(_dev) / "sglang" / "srt" / "multiplex" / "dual_worker.py" if _dev else None
_dual_path = _dev_dual if _dev_dual and _dev_dual.is_file() else SOURCE / "dual_worker.py"
_spec = importlib.util.spec_from_file_location("dual_worker", _dual_path)
dual_worker = importlib.util.module_from_spec(_spec)
sys.modules["dual_worker"] = dual_worker
assert _spec.loader is not None
_spec.loader.exec_module(dual_worker)


class TestH2MetricsPublishedBeforeFuture(unittest.TestCase):
    """H2: `task_count` used to rise in a `finally` that ran AFTER set_result.

    A `Future` runs its done-callbacks inside `set_result`, so a callback is the
    sharpest possible observer of "what could a waiter see the instant it was
    woken".  Under the defect it saw 0 completed tasks after 1 had completed.
    """

    def _observe_at_resolution(self, callback):
        worker = dual_worker.RoleWorkerThread(dual_worker.WorkerRole.DECODE)
        self.addCleanup(worker.close)
        seen = {}
        done = threading.Event()
        future = worker.submit(object(), callback)

        def watch(_f):
            seen["task_count"] = worker.task_count
            seen["intervals"] = len(worker.intervals())
            seen["busy_time_s"] = worker.busy_time_s
            done.set()

        future.add_done_callback(watch)
        self.assertTrue(done.wait(10.0), "future never resolved")
        return future, seen

    def test_success_path_counts_the_task_before_waking_the_waiter(self):
        future, seen = self._observe_at_resolution(lambda _ctx: "ok")
        self.assertEqual(future.result(timeout=5), "ok")
        self.assertEqual(seen["task_count"], 1)
        self.assertEqual(seen["intervals"], 1)
        self.assertGreater(seen["busy_time_s"], 0.0)

    def test_failure_path_also_counts_before_waking_the_waiter(self):
        def boom(_ctx):
            raise ValueError("nope")

        future, seen = self._observe_at_resolution(boom)
        with self.assertRaises(ValueError):
            future.result(timeout=5)
        self.assertEqual(seen["task_count"], 1)
        self.assertEqual(seen["intervals"], 1)

    def test_failure_still_records_last_error(self):
        worker = dual_worker.RoleWorkerThread(dual_worker.WorkerRole.PREFILL)
        self.addCleanup(worker.close)

        def boom(_ctx):
            raise RuntimeError("recorded")

        with self.assertRaises(RuntimeError):
            worker.submit(object(), boom).result(timeout=5)
        self.assertIn("recorded", worker.last_error)

    def test_many_sequential_tasks_never_lag(self):
        worker = dual_worker.RoleWorkerThread(dual_worker.WorkerRole.PREFILL)
        self.addCleanup(worker.close)
        for expected in range(1, 25):
            observed = []
            done = threading.Event()
            future = worker.submit(object(), lambda _ctx: None)
            future.add_done_callback(
                lambda _f: (observed.append(worker.task_count), done.set())
            )
            self.assertTrue(done.wait(10.0))
            self.assertEqual(observed[0], expected)


class TestH3OverlapWindow(unittest.TestCase):
    """H3: the lifetime-diluted, silently-truncated overlap ratio."""

    @staticmethod
    def _legacy_ratio(prefill, decode, elapsed):
        """The pre-repair formula, re-implemented independently.

        This is the pin that the OLD key's definition did not move: jobs
        907032 / 907100 were scored on it.
        """
        overlap = 0.0
        i = j = 0
        while i < len(prefill) and j < len(decode):
            p_start, p_end = prefill[i]
            d_start, d_end = decode[j]
            overlap += max(0.0, min(p_end, d_end) - max(p_start, d_start))
            if p_end <= d_end:
                i += 1
            else:
                j += 1
        return min(1.0, overlap / elapsed)

    def test_old_key_value_is_bit_for_bit_the_old_formula(self):
        prefill = [(0.0, 1.0), (2.0, 3.5), (5.0, 6.0)]
        decode = [(0.5, 2.5), (3.0, 4.0), (5.5, 7.0)]
        for elapsed in (1.0, 7.0, 1000.0):
            got = dual_worker._host_overlap_metrics(prefill, decode, elapsed, 3, 3)
            self.assertAlmostEqual(
                got["host_worker_overlap_ratio"],
                self._legacy_ratio(prefill, decode, elapsed),
                places=12,
                msg=f"elapsed={elapsed}",
            )

    def test_disjoint_intervals_report_no_overlap(self):
        got = dual_worker._host_overlap_metrics(
            [(0.0, 1.0)], [(2.0, 3.0)], 10.0, 1, 1
        )
        self.assertEqual(got["host_worker_overlap_s"], 0.0)
        self.assertEqual(got["host_worker_overlap_window_ratio"], 0.0)

    def test_window_ratio_is_not_diluted_by_boot_lifetime(self):
        """The defect: identical concurrency, longer boot, smaller old number."""
        prefill = [(100.0, 101.0)]
        decode = [(100.0, 101.0)]
        short = dual_worker._host_overlap_metrics(prefill, decode, 2.0, 1, 1)
        long = dual_worker._host_overlap_metrics(prefill, decode, 2000.0, 1, 1)
        self.assertGreater(
            short["host_worker_overlap_ratio"], long["host_worker_overlap_ratio"]
        )
        self.assertAlmostEqual(short["host_worker_overlap_window_ratio"], 1.0)
        self.assertAlmostEqual(long["host_worker_overlap_window_ratio"], 1.0)

    def test_window_span_is_the_jointly_observed_period(self):
        # decode's retained evidence starts late and ends early; the span is
        # decode's, not the union.
        prefill = [(0.0, 10.0)]
        decode = [(4.0, 6.0)]
        got = dual_worker._host_overlap_metrics(prefill, decode, 10.0, 1, 1)
        self.assertAlmostEqual(got["host_worker_overlap_window_span_s"], 2.0)
        self.assertAlmostEqual(got["host_worker_overlap_window_s"], 2.0)
        self.assertAlmostEqual(got["host_worker_overlap_window_ratio"], 1.0)

    def test_truncation_is_reported_not_hidden(self):
        prefill = [(0.0, 1.0)]
        decode = [(0.0, 1.0)]
        self.assertEqual(
            dual_worker._host_overlap_metrics(prefill, decode, 1.0, 1, 1)[
                "host_worker_overlap_truncated"
            ],
            0,
        )
        self.assertEqual(
            dual_worker._host_overlap_metrics(prefill, decode, 1.0, 5000, 1)[
                "host_worker_overlap_truncated"
            ],
            1,
        )

    def test_a_one_task_read_race_does_not_raise_the_truncation_flag(self):
        """`intervals()` and `task_count` are read under separate lock takes.

        A task completing between them differs the counts by exactly one on an
        untruncated worker; without the tolerance the flag would be stuck at 1
        on every busy boot and would therefore mean nothing.
        """
        prefill = decode = [(0.0, 1.0)]
        self.assertEqual(
            dual_worker._host_overlap_metrics(prefill, decode, 1.0, 2, 2)[
                "host_worker_overlap_truncated"
            ],
            0,
        )

    def test_empty_sides_are_safe(self):
        got = dual_worker._host_overlap_metrics([], [], 1.0, 0, 0)
        self.assertEqual(got["host_worker_overlap_ratio"], 0.0)
        self.assertEqual(got["host_worker_overlap_window_span_s"], 0.0)
        self.assertEqual(got["host_worker_overlap_window_ratio"], 0.0)

    def test_runtime_metrics_exposes_every_window_key(self):
        runtime = dual_worker.TrueDualWorkerRuntime([(0, 108), (84, 24)])
        self.addCleanup(runtime.close)
        metrics = runtime.metrics()
        for key in (
            "host_worker_overlap_ratio",
            "host_worker_overlap_s",
            "host_worker_overlap_window_s",
            "host_worker_overlap_window_span_s",
            "host_worker_overlap_window_ratio",
            "host_worker_overlap_truncated",
        ):
            self.assertIn(key, metrics)

    def test_real_runtime_overlap_is_visible_in_the_window_key(self):
        """Two role threads genuinely running at once must not report 0.0."""
        runtime = dual_worker.TrueDualWorkerRuntime([(84, 24)])
        self.addCleanup(runtime.close)
        barrier = threading.Barrier(2, timeout=10.0)

        def work(_ctx):
            barrier.wait()
            threading.Event().wait(0.05)

        futures = [
            runtime.submit(dual_worker.WorkerRole.PREFILL, work),
            runtime.submit(dual_worker.WorkerRole.DECODE, work),
        ]
        for future in futures:
            future.result(timeout=10)
        metrics = runtime.metrics()
        self.assertGreater(metrics["host_worker_overlap_s"], 0.0)
        self.assertGreater(metrics["host_worker_overlap_window_ratio"], 0.0)


class TestH4DecodeStepCount(unittest.TestCase):
    def test_dead_constant_zero_key_is_no_longer_emitted(self):
        state = dual_worker.DualWorkerState.from_sm_counts([(0, 108), (84, 24)])
        self.assertNotIn("decode_step_count", state.metrics())

    def test_the_r1_observer_instrument_itself_is_intact(self):
        """Removing the EMISSION must not remove the counter it reported."""
        decode = dual_worker.DecodeWorker()
        decode.begin_step(now=20.0)
        decode.finish_step(now=20.01)
        self.assertEqual(decode.snapshot.step_count, 1)
        self.assertAlmostEqual(decode.snapshot.last_tpot_ms, 10.0, places=6)

    def test_finish_step_without_begin_step_stays_a_no_op(self):
        """This is exactly why the key was 0 in every boot."""
        decode = dual_worker.DecodeWorker()
        decode.finish_step(now=1.0)
        self.assertEqual(decode.snapshot.step_count, 0)


class TestH5WorkerOverlapRatio(unittest.TestCase):
    def test_dead_snapshot_field_is_gone(self):
        fields = {
            f
            for f in controller.RuntimeSnapshot.__dataclass_fields__
        }
        self.assertNotIn("worker_overlap_ratio", fields)

    def test_the_live_sibling_still_exists_under_its_own_name(self):
        runtime = dual_worker.TrueDualWorkerRuntime([(84, 24)])
        self.addCleanup(runtime.close)
        self.assertIn("host_worker_overlap_ratio", runtime.metrics())

    def test_known_dead_idle_ratios_are_deliberately_left_alone(self):
        """`controller.py:55` is cited as the evidence that these are dead.

        Removing them would delete the record those citations point at, which is
        a different defect from the one being repaired.  If a later change does
        remove them, this test fails and forces the citations to be handled.
        """
        fields = controller.RuntimeSnapshot.__dataclass_fields__
        self.assertIn("prefill_idle_ratio", fields)
        self.assertIn("decode_idle_ratio", fields)


class TestVerdictRuleInputsUnbroken(unittest.TestCase):
    """★The constraint all four repairs had to respect.

    `r2_correctness_check.py` is the pinned judging rule (v2) that jobs 907032,
    907100 and 907456 were scored under, and it is not editable here.  Read its
    SNAP_KEYS textually and prove every key it consumes is still produced.
    """

    @staticmethod
    def _snap_keys():
        tree = ast.parse(CHECKER.read_text(errors="replace"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                if any(
                    isinstance(t, ast.Name) and t.id == "SNAP_KEYS"
                    for t in node.targets
                ):
                    return tuple(ast.literal_eval(node.value))
        raise AssertionError("SNAP_KEYS not found in the checker")

    def test_every_snap_key_still_has_a_producer(self):
        if not CHECKER.is_file():
            self.skipTest(f"checker not present at {CHECKER}")
        runtime = dual_worker.TrueDualWorkerRuntime([(0, 108), (84, 24)])
        self.addCleanup(runtime.close)
        state = dual_worker.DualWorkerState.from_sm_counts([(0, 108), (84, 24)])
        produced = set(runtime.metrics()) | set(state.metrics())
        # Supplied by the telemetry envelope / scheduler, not by these two.
        envelope = {"timestamp_monotonic_s", "architecture", "dropped_events"}
        missing = [
            key for key in self._snap_keys()
            if key not in produced and key not in envelope
        ]
        self.assertEqual(missing, [], f"SNAP_KEYS lost a producer: {missing}")

    def test_repairs_did_not_shrink_the_emitted_key_set(self):
        """Every key except the two deliberately dropped ones must survive."""
        runtime = dual_worker.TrueDualWorkerRuntime([(0, 108), (84, 24)])
        self.addCleanup(runtime.close)
        state = dual_worker.DualWorkerState.from_sm_counts([(0, 108), (84, 24)])
        produced = set(runtime.metrics()) | set(state.metrics())
        produced |= set(asdict(_reference_runtime_snapshot()))
        dropped = {"decode_step_count", "worker_overlap_ratio"}
        for key in _KEYS_BEFORE_REPAIR:
            if key in dropped:
                self.assertNotIn(key, produced)
            else:
                self.assertIn(key, produced, f"{key} disappeared")


def _reference_runtime_snapshot():
    return controller.RuntimeSnapshot(
        timestamp_s=0.0,
        decode_iterations=0,
        active_decode_sequences=0,
        decode_batch_size=1,
        context_p50=1,
        context_p95=1,
        context_max=1,
        expected_remaining_output_p90=0,
        prefill_queue_depth=0,
        oldest_prefill_age_ms=0.0,
        measured_itl_ewma_ms=0.0,
        measured_itl_p95_ms=0.0,
        itl_slo_ms=60.0,
        ttft_risk=0.0,
        kv_occupancy=0.0,
        running_batch_occupancy=0.0,
    )


# The `runtime_snapshot` payload key set as of HEAD 2eb66d1, before H2-H5.
_KEYS_BEFORE_REPAIR = (
    "prefill_host_queue_depth", "decode_host_queue_depth", "prefill_host_tasks",
    "decode_host_tasks", "prefill_host_busy_ms", "decode_host_busy_ms",
    "prefill_host_idle_ratio", "decode_host_idle_ratio",
    "host_worker_overlap_ratio", "active_leases", "pending_gpu_events",
    "partition_generation", "owned_prefill_waiting", "owned_prefill_active",
    "owned_decode_ready", "owned_decode_running",
    "prefill_queue_depth", "prefill_queue_age_ms", "prefill_active_batch_size",
    "prefill_chunk_progress", "prefill_admission_latency_ms",
    "prefill_admission_blocked", "prefill_admission_block_reason",
    "decode_ready_queue_depth", "decode_running_batch_size",
    "decode_last_tpot_ms", "decode_step_count", "stream_index", "prefill_sms",
    "decode_sms", "active_worker_leases",
    "timestamp_s", "decode_iterations", "active_decode_sequences",
    "decode_batch_size", "context_p50", "context_p95", "context_max",
    "expected_remaining_output_p90", "oldest_prefill_age_ms",
    "measured_itl_ewma_ms", "measured_itl_p95_ms", "itl_slo_ms", "ttft_risk",
    "kv_occupancy", "running_batch_occupancy", "prefill_idle_ratio",
    "decode_idle_ratio", "worker_overlap_ratio",
)


# ---------------------------------------------------------------------------
# NEGATIVE CONTROLS
# ---------------------------------------------------------------------------
# A test that passes on the unrepaired source certifies nothing (lesson #53).
# Each mutant below UNDOES one repair by text substitution on the CURRENT file
# -- not by reading an old commit, which would stop being the "pre-fix" source
# the moment this work is committed -- and the defect must then reappear.

_H2_REPAIRED = """            finally:
                finished = time.perf_counter()
                with self._metrics_lock:   # H2, see NOTE at end of file
                    self.busy_time_s += finished - started
                    self.task_count += 1
                    self._intervals = (self._intervals + [(started, finished)])[-1024:]
                setter = task.future.set_result if error is None else task.future.set_exception
                setter(result if error is None else error)"""

_H2_PRE_FIX = """            finally:
                if error is not None:
                    task.future.set_exception(error)
                else:
                    task.future.set_result(result)
                finished = time.perf_counter()
                with self._metrics_lock:
                    self.busy_time_s += finished - started
                    self.task_count += 1
                    self._intervals = (self._intervals + [(started, finished)])[-1024:]"""

_H4_REPAIRED = "            # H4: `decode_step_count` dropped -- NOTE at end of file."
_H4_PRE_FIX = '            "decode_step_count": d.step_count,'

_H3_WINDOW_RATIO = "min(1.0, window_s / span_s) if span_s > 0.0 else 0.0"
_H3_DILUTED = "min(1.0, window_s / elapsed_s) if elapsed_s > 0.0 else 0.0"

_H5_REPAIRED = (
    "    # H5: `worker_overlap_ratio` removed 2026-09-12 -- see dual_worker.py NOTE."
)
_H5_PRE_FIX = "    worker_overlap_ratio: float = 0.0"


def _load_mutant(tmpdir, name, substitutions):
    """Copy the whole multiplex source, mutate one file, import the copy."""
    import shutil

    target = Path(tmpdir) / "multiplex"
    if not target.exists():
        shutil.copytree(SOURCE, target, ignore=shutil.ignore_patterns("__pycache__"))
    path = target / (name + ".py")
    text = path.read_text(encoding="utf-8")
    for old, new in substitutions:
        assert old in text, "mutation anchor not found in %s.py: %r" % (name, old[:60])
        text = text.replace(old, new, 1)
    path.write_text(text, encoding="utf-8")
    modname = "mutant_%s_%d" % (name, abs(hash(repr(substitutions))) % 10 ** 8)
    spec = importlib.util.spec_from_file_location(modname, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[modname] = module
    spec.loader.exec_module(module)
    return module


class TestNegativeControls(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name
        self.addCleanup(self._tmp.cleanup)

    def test_h2_mutant_lags_by_one(self):
        mutant = _load_mutant(self.tmp, "dual_worker", [(_H2_REPAIRED, _H2_PRE_FIX)])
        worker = mutant.RoleWorkerThread(mutant.WorkerRole.DECODE)
        self.addCleanup(worker.close)
        seen, done = [], threading.Event()
        future = worker.submit(object(), lambda _ctx: None)
        future.add_done_callback(
            lambda _f: (seen.append(worker.task_count), done.set())
        )
        self.assertTrue(done.wait(10.0))
        self.assertEqual(seen[0], 0, "the pre-fix ordering must show the -1 lag")

    def test_h3_mutant_reintroduces_lifetime_dilution(self):
        mutant = _load_mutant(
            self.tmp, "dual_worker", [(_H3_WINDOW_RATIO, _H3_DILUTED)]
        )
        prefill = decode = [(100.0, 101.0)]
        short = mutant._host_overlap_metrics(prefill, decode, 2.0, 1, 1)
        long_ = mutant._host_overlap_metrics(prefill, decode, 2000.0, 1, 1)
        self.assertGreater(
            short["host_worker_overlap_window_ratio"],
            long_["host_worker_overlap_window_ratio"],
            "the diluted mutant must show the decay the window key removes",
        )

    def test_h4_mutant_emits_the_constant_zero_key(self):
        mutant = _load_mutant(self.tmp, "dual_worker", [(_H4_REPAIRED, _H4_PRE_FIX)])
        state = mutant.DualWorkerState.from_sm_counts([(0, 108), (84, 24)])
        metrics = state.metrics()
        self.assertIn("decode_step_count", metrics)
        self.assertEqual(
            metrics["decode_step_count"], 0,
            "and it is the constant 0 that made it a dead field",
        )

    def test_h5_mutant_reintroduces_the_dead_field(self):
        had = sys.modules.get("profile")
        try:
            profile_mutant = _load_mutant(self.tmp, "profile", [])
            sys.modules["profile"] = profile_mutant
            mutant = _load_mutant(
                self.tmp, "controller", [(_H5_REPAIRED, _H5_PRE_FIX)]
            )
        finally:
            if had is not None:
                sys.modules["profile"] = had
            else:
                sys.modules.pop("profile", None)
        fields = mutant.RuntimeSnapshot.__dataclass_fields__
        self.assertIn("worker_overlap_ratio", fields)
        required = {
            name: 0.0
            for name, spec in fields.items()
            if spec.default is dataclasses.MISSING
            and spec.default_factory is dataclasses.MISSING
        }
        snapshot = mutant.RuntimeSnapshot(**required)
        self.assertEqual(
            asdict(snapshot)["worker_overlap_ratio"], 0.0,
            "and it ships as a hard 0.0 into every telemetry record",
        )


if __name__ == "__main__":
    unittest.main()
