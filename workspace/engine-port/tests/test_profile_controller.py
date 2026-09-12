import importlib.util
import math
import sys
import tempfile
import unittest
from pathlib import Path


SOURCE = Path(__file__).parents[1] / "src" / "multiplex"


def load(name):
    # NOTE: the bare name is load-bearing -- controller.py's CPU-only fallback
    # does `from profile import ...` (controller.py:17), so this registration
    # must stay `sys.modules["profile"]`.  Side effect: it SHADOWS the stdlib
    # `profile` module for the rest of the interpreter, which breaks a later
    # `import sglang` in the same `unittest discover` run
    # (sglang.utils -> IPython -> cProfile -> `import profile`).  Test modules
    # that import the installed runtime must drop the shadow first; see
    # test_trace_force_prefill.py.
    spec = importlib.util.spec_from_file_location(name, SOURCE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


profile = load("profile")
controller = load("controller")
telemetry = load("telemetry")


def make_profile():
    # `engine_source_hash` became a REQUIRED compatibility axis on 2026-09-12
    # (roadmap "Controller defaults" declared canonical over the code): engine
    # identity is compared on source content instead of `engine_commit`, and an
    # empty hash is fail-closed on purpose, so a profile built without one is
    # never compatible.  Before that change this fixture passed with no hash at
    # all, which is exactly the silent "unknown == unknown" match the axis
    # removes; the axis itself is gated in test_controller_defaults.py.
    environment = profile.RuntimeEnvironment(
        engine_commit="abc",
        gpu_name="A100",
        gpu_sm_count=108,
        cuda_driver="13.0",
        attention_backend="triton",
        cuda_graph=True,
        engine_source_hash="f" * 64,
    )
    points = []
    for decode_sms, base in ((16, 70.0), (24, 50.0), (34, 40.0), (44, 30.0), (108, 20.0)):
        for batch, batch_cost in ((1, 0.0), (16, 10.0)):
            for context, context_cost in ((256, 0.0), (4096, 10.0)):
                value = base + batch_cost + context_cost
                points.append(
                    profile.DecodeLatencyPoint(
                        decode_sms,
                        batch,
                        context,
                        value - 5.0,
                        value,
                        value + 5.0,
                        repeats=5,
                        measured_steps=30,
                        residual_p95_ms=1.0,
                    )
                )
    return profile.HybridModelProfileV1(
        model_id="test/hybrid",
        model_revision="r1",
        model_config_hash="hash",
        environment=environment,
        attention_layers=8,
        ssm_layers=24,
        attention_kind="GQA",
        num_attention_heads=32,
        num_kv_heads=8,
        parameter_count=1_000_000,
        points=points,
        heldout_residual_p95_ms=2.0,
    )


class ProfileTest(unittest.TestCase):
    def test_profile_round_trip_and_compatibility(self):
        item = make_profile()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            item.save(path)
            loaded = profile.HybridModelProfileV1.load(path)
        self.assertEqual(loaded.model_id, item.model_id)
        self.assertAlmostEqual(loaded.attention_ratio, 0.25)
        self.assertEqual(loaded.is_compatible(item.environment), (True, []))

    def test_floor_uses_upper_bound_and_safety_margin(self):
        estimator = profile.ConservativeDecodeFloorEstimator(make_profile())
        estimate = estimator.estimate(1, 256, itl_slo_ms=60.0)
        self.assertEqual(estimate.decode_sms, 24)
        self.assertLessEqual(estimate.upper_bound_itl_ms, 54.0)
        self.assertEqual(estimate.safety_margin_ms, 6.0)

    def test_environment_mismatch_uses_conservative_fallback(self):
        item = make_profile()
        estimator = profile.ConservativeDecodeFloorEstimator(item)
        wrong = profile.RuntimeEnvironment(
            **{**item.environment.__dict__, "cuda_graph": False}
        )
        estimate = estimator.estimate(1, 256, 60.0, runtime=wrong)
        self.assertTrue(estimate.fallback)
        self.assertEqual(estimate.decode_sms, 44)
        self.assertTrue(math.isinf(estimate.upper_bound_itl_ms))


class ControllerTest(unittest.TestCase):
    @staticmethod
    def snapshot(now, iteration, itl=30.0, kv=0.1):
        return controller.RuntimeSnapshot(
            timestamp_s=now,
            decode_iterations=iteration,
            active_decode_sequences=8,
            decode_batch_size=8,
            context_p50=1024,
            context_p95=1024,
            context_max=1024,
            expected_remaining_output_p90=128,
            prefill_queue_depth=4,
            oldest_prefill_age_ms=100.0,
            measured_itl_ewma_ms=itl,
            measured_itl_p95_ms=itl,
            itl_slo_ms=60.0,
            ttft_risk=0.1,
            kv_occupancy=kv,
            running_batch_occupancy=0.2,
        )

    def test_downshift_requires_three_epochs_and_dwell(self):
        ctl = controller.CoarseGrainedController(
            min_epoch_s=0.0,
            min_epoch_iterations=0,
            min_dwell_s=0.0,
            min_dwell_iterations=0,
        )
        estimate = profile.DecodeFloorEstimate(24, 20, 30, 6, 0.95, False, "test")
        self.assertEqual(
            ctl.stabilize(estimate, self.snapshot(1, 1), 44, True).target_decode_sms,
            44,
        )
        self.assertEqual(
            ctl.stabilize(estimate, self.snapshot(2, 2), 44, True).target_decode_sms,
            44,
        )
        self.assertEqual(
            ctl.stabilize(estimate, self.snapshot(3, 3), 44, True).target_decode_sms,
            24,
        )

    def test_underprediction_upshifts_but_unsafe_boundary_blocks(self):
        ctl = controller.CoarseGrainedController(
            min_epoch_s=0.0, min_epoch_iterations=0
        )
        estimate = profile.DecodeFloorEstimate(24, 50, 55, 6, 0.8, False, "test")
        decision = ctl.stabilize(
            estimate, self.snapshot(1, 1, itl=70), 24, safe_boundary=False
        )
        self.assertFalse(decision.safe)
        self.assertEqual(decision.target_decode_sms, 24)
        self.assertEqual(decision.requested_decode_sms, 34)


class TelemetryTest(unittest.TestCase):
    def test_async_writer_emits_phase_and_close_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            writer = telemetry.AsyncJsonlTelemetry(path, "run", "W1", flush_every=1)
            writer.mark_phase("warmup")
            writer.emit("runtime_snapshot", "benchmark", queue_depth=2)
            writer.close()
            lines = path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 3)
        self.assertIn('"phase":"warmup"', lines[0])
        self.assertIn('"event":"telemetry_close"', lines[-1])


if __name__ == "__main__":
    unittest.main()
