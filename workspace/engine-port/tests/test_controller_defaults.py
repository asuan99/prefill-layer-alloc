"""Gates for the three roadmap-vs-code mismatches fixed on 2026-09-12.

The roadmap's "Controller defaults" section (reports/paper/EXPERIMENT_ROADMAP.md,
registered 2026-09-12) was declared canonical over the code on three points.
Each one gets a test here that FAILS on the pre-fix source:

  (1) admission-limit persistence -- a limit set by an evaluation survives the
      non-evaluation (HOLD) iterations in between, instead of being reset to the
      `SplitDecision` default on the next iteration.  Mutant that fails:
      HOLD returning `SplitDecision(current, current, HOLD)`.
      Degeneration guard: the limit must still LIFT (a later evaluation, or the
      runtime's `release_admission_limit`), so "inherit" must not become
      "latch forever".  Mutant that fails: `admission_limited or ...`.
  (2) evaluation cadence -- `max(4 decode iterations, 100 ms)`, i.e. BOTH
      thresholds, matching `_dwell_satisfied`.  Mutant that fails: the `or`
      (first-to-arrive) form of `evaluation_due`.
  (3) engine identity axis -- profiles are compared on engine SOURCE CONTENT
      (`engine_source_hash`), not on `engine_commit`; a missing hash on either
      side is FAIL-CLOSED.  Mutants that fail: `engine_commit` back in the
      strict field set; treating an empty hash as a match.

RUNNING AGAINST A VARIANT.  Set PDMUX_MULTIPLEX_SRC=/path/to/multiplex to load
`profile.py`/`controller.py` from another copy of the tracked source; the
pre-fix variants of all three changes fail here.

NO GPU, NO PERFORMANCE CLAIM.  These are determinism/plumbing gates only: they
say nothing about whether any policy is faster, and the hybrid/generic policies
they exercise still have zero GPU measurements.

WHY THE `profile` SHADOW IS POPPED.  controller.py's CPU-only fallback does
`from profile import ...` (controller.py:17), so loading it requires
`sys.modules["profile"]` to be this source tree's profile module -- which
shadows the stdlib `profile` and breaks any later `import sglang` in the same
`unittest discover` run (see test_profile_controller.py's note).  This module
sorts BEFORE the sglang-importing test modules, so it removes the shadow again
as soon as the two modules are loaded.
"""

import csv
import importlib.util
import io
import json
import math
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from unittest import mock
from pathlib import Path


# Tracked source by default; PDMUX_MULTIPLEX_SRC=/path/to/multiplex loads
# another copy instead -- used to show that these tests FAIL on the pre-fix
# source without touching the tracked tree (same role as
# PDMUX_MIXIN_UNDER_TEST in pdmux_loop_fakes.py).
SOURCE = Path(
    os.environ.get("PDMUX_MULTIPLEX_SRC", "")
    or Path(__file__).parents[1] / "src" / "multiplex"
)


def _load(name, directory=SOURCE):
    spec = importlib.util.spec_from_file_location(name, Path(directory) / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_had_profile = sys.modules.get("profile")
profile = _load("profile")
controller = _load("controller")
# Restore/remove the stdlib shadow: see the module docstring.
if _had_profile is not None:
    sys.modules["profile"] = _had_profile
else:
    sys.modules.pop("profile", None)


def snapshot(now, iteration, itl=30.0, kv=0.10, running=0.20):
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
        running_batch_occupancy=running,
    )


def overloaded_estimate():
    """A desired floor that makes `stabilize`'s overload condition true.

    `decode_sms=108` is the emergency state and `upper_bound_itl_ms=inf` is what
    the estimator returns on an incompatible profile, which is exactly the
    fallback that (3) interacts with.
    """
    return profile.DecodeFloorEstimate(108, math.nan, math.inf, 6.0, 0.0, True, "t")


def calm_estimate():
    return profile.DecodeFloorEstimate(24, 20.0, 30.0, 6.0, 0.95, False, "t")


def environment(**overrides):
    fields = dict(
        engine_commit="commit-a",
        gpu_name="A100",
        gpu_sm_count=108,
        cuda_driver="13.0",
        attention_backend="triton",
        cuda_graph=True,
        piecewise_cuda_graph=False,
        engine_source_hash="c0ffee" * 10,
    )
    fields.update(overrides)
    return profile.RuntimeEnvironment(**fields)


def make_profile(**environment_overrides):
    points = [
        profile.DecodeLatencyPoint(
            decode_sms, batch, context, value - 5.0, value, value + 5.0,
            repeats=5, measured_steps=30, residual_p95_ms=1.0,
        )
        for decode_sms, base in ((16, 70.0), (24, 50.0), (34, 40.0), (44, 30.0), (108, 20.0))
        for batch, batch_cost in ((1, 0.0), (16, 10.0))
        for context, context_cost in ((256, 0.0), (4096, 10.0))
        for value in (base + batch_cost + context_cost,)
    ]
    return profile.HybridModelProfileV1(
        model_id="test/hybrid",
        model_revision="r1",
        model_config_hash="hash",
        environment=environment(**environment_overrides),
        attention_layers=8,
        ssm_layers=24,
        attention_kind="GQA",
        num_attention_heads=32,
        num_kv_heads=8,
        parameter_count=1_000_000,
        points=points,
        heldout_residual_p95_ms=2.0,
    )


class EvaluationCadenceTest(unittest.TestCase):
    """(2) `max(4 decode iterations, 100 ms)` = BOTH thresholds."""

    def controller(self):
        return controller.CoarseGrainedController(
            min_epoch_s=0.100, min_epoch_iterations=4
        )

    def test_first_call_is_due_from_the_initial_sentinels(self):
        ctl = self.controller()
        self.assertTrue(ctl.evaluation_due(snapshot(0.0, 0)))
        decision = ctl.stabilize(calm_estimate(), snapshot(0.0, 0), 44, True)
        self.assertNotEqual(decision.reason, controller.DecisionReason.HOLD.value)

    def test_both_thresholds_are_required(self):
        ctl = self.controller()
        ctl.stabilize(calm_estimate(), snapshot(10.0, 100), 44, True)  # evaluation
        # Time met, iterations not -> NOT due (the `or` form fires here).
        self.assertFalse(ctl.evaluation_due(snapshot(10.50, 102)))
        # Iterations met, time not -> NOT due (the `or` form fires here too).
        self.assertFalse(ctl.evaluation_due(snapshot(10.05, 120)))
        # Neither -> not due.
        self.assertFalse(ctl.evaluation_due(snapshot(10.05, 101)))
        # Both -> due.
        self.assertTrue(ctl.evaluation_due(snapshot(10.20, 104)))

    def test_bucket_change_still_forces_an_evaluation(self):
        ctl = self.controller()
        ctl.stabilize(calm_estimate(), snapshot(10.0, 100), 44, True)
        self.assertTrue(
            ctl.evaluation_due(snapshot(10.01, 100), bucket_changed=True)
        )

    def test_hold_holds_the_split_between_evaluations(self):
        ctl = self.controller()
        ctl.stabilize(calm_estimate(), snapshot(10.0, 100), 44, True)
        decision = ctl.stabilize(calm_estimate(), snapshot(10.50, 102), 44, True)
        self.assertEqual(decision.reason, controller.DecisionReason.HOLD.value)
        self.assertEqual(decision.target_decode_sms, 44)

    def test_cadence_matches_the_dwell_operator(self):
        """`_dwell_satisfied` already means `max(...)`; the two must agree."""
        ctl = controller.CoarseGrainedController(
            min_epoch_s=0.100, min_epoch_iterations=4,
            min_dwell_s=0.100, min_dwell_iterations=4,
        )
        ctl.last_evaluation_s = ctl.last_transition_s = 10.0
        ctl.last_evaluation_iteration = ctl.last_transition_iteration = 100
        for now, iteration in ((10.05, 101), (10.50, 102), (10.05, 120), (10.20, 104)):
            probe = snapshot(now, iteration)
            self.assertEqual(
                ctl.evaluation_due(probe),
                ctl._dwell_satisfied(probe),
                f"cadence and dwell disagree at t={now}, it={iteration}",
            )


class AdmissionLimitPersistenceTest(unittest.TestCase):
    """(1) the limit lives between evaluations, and still lifts."""

    def controller(self):
        # One evaluation per 10 s AND 4 iterations: the iterations in between
        # are HOLDs, which is where the pre-fix code dropped the limit.
        return controller.CoarseGrainedController(
            min_epoch_s=10.0, min_epoch_iterations=4
        )

    def latched(self, ctl):
        """Drive two overloaded evaluations (the roadmap's 2 epochs)."""
        first = ctl.stabilize(overloaded_estimate(), snapshot(0.0, 0), 44, True)
        self.assertFalse(first.admission_limited, "one epoch must not latch")
        second = ctl.stabilize(overloaded_estimate(), snapshot(20.0, 10), 108, True)
        self.assertTrue(second.admission_limited, "two epochs must latch")
        return ctl

    def test_hold_carries_the_limit_set_by_the_last_evaluation(self):
        ctl = self.latched(self.controller())
        for now, iteration in ((20.1, 11), (21.0, 12), (25.0, 13)):
            decision = ctl.stabilize(
                overloaded_estimate(), snapshot(now, iteration), 108, True
            )
            self.assertEqual(decision.reason, controller.DecisionReason.HOLD.value)
            self.assertTrue(
                decision.admission_limited,
                f"HOLD at t={now} dropped the admission limit set 0.1-5 s earlier "
                "(roadmap EXPERIMENT_ROADMAP.md:979: it persists while the "
                "overload condition holds)",
            )

    def test_hold_does_not_become_a_permanent_limit(self):
        ctl = self.latched(self.controller())
        lifted = ctl.stabilize(calm_estimate(), snapshot(40.0, 20), 108, True)
        self.assertFalse(
            lifted.admission_limited,
            "an evaluation with no overload must lift the limit",
        )
        self.assertFalse(
            ctl.stabilize(
                calm_estimate(), snapshot(40.1, 21), 44, True
            ).admission_limited,
            "the HOLD after the lifting evaluation re-asserted the limit "
            "(inheritance degenerated into a latch)",
        )

    def test_runtime_release_clears_the_carried_verdict(self):
        ctl = self.latched(self.controller())
        ctl.release_admission_limit()
        self.assertFalse(
            ctl.stabilize(
                overloaded_estimate(), snapshot(20.1, 11), 108, True
            ).admission_limited,
            "HOLD re-asserted a limit the runtime had already released "
            "(the empty-running-batch backstop, multiplexing_mixin "
            "`_r2_admission_holds`)",
        )

    def test_release_also_resets_the_overload_streak(self):
        """After a release, a new limit needs two FRESH overloaded epochs."""
        ctl = self.latched(self.controller())
        ctl.release_admission_limit()
        self.assertFalse(
            ctl.stabilize(
                overloaded_estimate(), snapshot(40.0, 20), 108, True
            ).admission_limited,
            "one overloaded epoch after a release re-latched immediately",
        )
        self.assertTrue(
            ctl.stabilize(
                overloaded_estimate(), snapshot(60.0, 30), 108, True
            ).admission_limited
        )

    def test_non_overloaded_runs_never_limit(self):
        ctl = self.controller()
        for index in range(6):
            decision = ctl.stabilize(
                calm_estimate(), snapshot(20.0 * index, 10 * index), 44, True
            )
            self.assertFalse(decision.admission_limited)


class EngineSourceAxisTest(unittest.TestCase):
    """(3) engine identity = source content, not repo commit."""

    @staticmethod
    def write_tree(directory, bodies):
        paths = {}
        for name, body in bodies.items():
            path = Path(directory) / f"{name}.py"
            path.write_text(body, encoding="utf-8")
            paths[f"pkg.{name}"] = path
        return paths

    def test_hash_is_content_addressed_and_path_independent(self):
        bodies = {"alpha": "A = 1\n", "beta": "B = 2\n"}
        with tempfile.TemporaryDirectory() as one, tempfile.TemporaryDirectory() as two:
            first = profile.hash_engine_source_files(self.write_tree(one, bodies))
            second = profile.hash_engine_source_files(self.write_tree(two, bodies))
            self.assertEqual(len(first), 64)
            self.assertEqual(
                first, second, "the same sources under another root must hash equal"
            )
            moved = self.write_tree(two, {"alpha": "A = 1\n", "beta": "B = 3\n"})
            self.assertNotEqual(
                first,
                profile.hash_engine_source_files(moved),
                "a one-byte source change must change the hash",
            )

    def test_hash_binds_content_to_module_name(self):
        """Swapping which module holds which body must change the hash."""
        with tempfile.TemporaryDirectory() as one, tempfile.TemporaryDirectory() as two:
            straight = self.write_tree(one, {"alpha": "A = 1\n", "beta": "B = 2\n"})
            swapped = self.write_tree(two, {"alpha": "B = 2\n", "beta": "A = 1\n"})
            self.assertNotEqual(
                profile.hash_engine_source_files(straight),
                profile.hash_engine_source_files(swapped),
            )

    def test_loaded_modules_resolve_from_sys_modules(self):
        resolved = profile.resolve_engine_source_files(["unittest", "json"])
        self.assertEqual(set(resolved), {"unittest", "json"})
        self.assertTrue(all(path.is_file() for path in resolved.values()))
        self.assertEqual(
            profile.engine_source_hash(["unittest", "json"]),
            profile.engine_source_hash(["unittest", "json"]),
            "the hash must be deterministic across calls",
        )

    def test_unresolvable_module_is_fail_closed_not_a_crash(self):
        self.assertIsNone(
            profile.resolve_engine_source_files(["pdmux.not.a.module"])
        )
        self.assertEqual(
            profile.engine_source_hash(["pdmux.not.a.module"]),
            profile.ENGINE_SOURCE_HASH_UNAVAILABLE,
        )

    def test_axis_covers_the_serving_modules_and_excludes_probes(self):
        axis = set(profile.ENGINE_SOURCE_MODULES)
        for required in (
            "sglang.srt.multiplex.multiplexing_mixin",
            "sglang.srt.multiplex.controller",
            "sglang.srt.multiplex.profile",
            "sglang.srt.multiplex.dual_worker",
            "sglang.srt.distributed.parallel_state",
        ):
            self.assertIn(required, axis)
        for excluded in ("holb_probe", "chunk_probe", "green_readout"):
            self.assertNotIn(
                f"sglang.srt.multiplex.{excluded}",
                axis,
                "default-OFF probes are deliberately outside the axis",
            )

    def test_commit_only_change_keeps_the_profile_compatible(self):
        item = make_profile()
        runtime = environment(engine_commit="commit-b-docs-only")
        compatible, reasons = item.is_compatible(runtime)
        self.assertTrue(
            compatible,
            f"a docs-only commit invalidated the profile: {reasons}",
        )
        self.assertEqual(reasons, [])
        estimate = profile.ConservativeDecodeFloorEstimator(item).estimate(
            1, 256, 60.0, runtime=runtime
        )
        self.assertFalse(estimate.fallback)
        self.assertLess(estimate.upper_bound_itl_ms, math.inf)

    def test_source_change_makes_the_profile_incompatible(self):
        item = make_profile()
        runtime = environment(engine_source_hash="dead" * 16)
        compatible, reasons = item.is_compatible(runtime)
        self.assertFalse(compatible)
        self.assertTrue(any("engine_source_hash" in reason for reason in reasons))
        estimate = profile.ConservativeDecodeFloorEstimator(item).estimate(
            1, 256, 60.0, runtime=runtime
        )
        self.assertTrue(estimate.fallback)
        self.assertTrue(math.isinf(estimate.upper_bound_itl_ms))

    def test_profile_without_the_hash_field_fails_closed(self):
        """A profile JSON written before 2026-09-12 has no hash at all."""
        raw = make_profile().to_dict()
        del raw["environment"]["engine_source_hash"]
        legacy = profile.HybridModelProfileV1.from_dict(raw)
        self.assertEqual(legacy.environment.engine_source_hash, "")
        compatible, reasons = legacy.is_compatible(environment())
        self.assertFalse(
            compatible,
            "a profile with no engine_source_hash was accepted (silent match)",
        )
        self.assertTrue(
            any("missing in profile" in reason for reason in reasons), reasons
        )
        estimate = profile.ConservativeDecodeFloorEstimator(legacy).estimate(
            1, 256, 60.0, runtime=environment()
        )
        self.assertTrue(estimate.fallback)
        self.assertIn("incompatible_profile", estimate.reason)

    def test_unknown_runtime_hash_fails_closed(self):
        item = make_profile()
        compatible, reasons = item.is_compatible(environment(engine_source_hash=""))
        self.assertFalse(compatible)
        self.assertTrue(any("could not be computed" in reason for reason in reasons))

    def test_two_unknown_hashes_are_not_a_match(self):
        """`"" == ""` must not read as "same engine"."""
        item = make_profile(engine_source_hash="")
        compatible, reasons = item.is_compatible(environment(engine_source_hash=""))
        self.assertFalse(compatible)
        self.assertEqual(len(reasons), 2, reasons)

    def test_engine_commit_is_kept_as_provenance(self):
        raw = make_profile().to_dict()
        self.assertEqual(raw["environment"]["engine_commit"], "commit-a")
        self.assertIn("engine_source_hash", raw["environment"])

    def test_round_trip_preserves_the_hash(self):
        item = make_profile()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            item.save(path)
            loaded = profile.HybridModelProfileV1.load(path)
        self.assertEqual(
            loaded.environment.engine_source_hash,
            item.environment.engine_source_hash,
        )
        self.assertEqual(loaded.is_compatible(item.environment), (True, []))


class FallbackDrivesTheLatchTest(unittest.TestCase):
    """(3) x (1): fail-closed fallback -> inf -> overload -> carried limit."""

    def test_incompatible_profile_latches_admission_and_hold_keeps_it(self):
        item = make_profile(engine_source_hash="")  # pre-axis profile
        estimator = profile.ConservativeDecodeFloorEstimator(item)
        ctl = controller.CoarseGrainedController(
            min_epoch_s=10.0, min_epoch_iterations=4
        )
        policy = controller.HybridInformedPolicy(
            estimator, controller=ctl, runtime_environment=environment()
        )
        estimate = estimator.estimate(8, 1024, 60.0, runtime=environment())
        self.assertTrue(estimate.fallback)
        self.assertTrue(math.isinf(estimate.upper_bound_itl_ms))

        # ITL already over SLO at the top steady state -> emergency target.
        first = policy.decide(snapshot(0.0, 0, itl=90.0), 44, True)
        self.assertEqual(first.target_decode_sms, 108)
        self.assertFalse(first.admission_limited)
        second = policy.decide(snapshot(20.0, 10, itl=90.0), 108, True)
        self.assertTrue(second.admission_limited)
        held = policy.decide(snapshot(20.2, 11, itl=90.0), 108, True)
        self.assertEqual(held.reason, controller.DecisionReason.HOLD.value)
        self.assertTrue(
            held.admission_limited,
            "the fallback-driven limit vanished on the next iteration",
        )


class FixedPolicyIsUntouchedTest(unittest.TestCase):
    """Claim D / P1-P2 arms (PDMUX_R2_POLICY unset or fixed) see no change."""

    def test_fixed_policy_has_no_controller_and_never_limits(self):
        policy = controller.FixedPolicy(44)
        self.assertIsNone(
            getattr(policy, "controller", None),
            "FixedPolicy must not acquire controller state: the mixin's "
            "release hook looks up `r2_policy.controller`",
        )
        for now, iteration in ((0.0, 0), (0.001, 1), (100.0, 1000)):
            for safe in (True, False):
                decision = policy.decide(
                    snapshot(now, iteration, itl=500.0, kv=1.0, running=1.0),
                    44 if safe else 16,
                    safe,
                )
                self.assertFalse(decision.admission_limited)
                self.assertFalse(decision.emergency)
                self.assertEqual(decision.requested_decode_sms, 44)
                self.assertEqual(
                    decision.reason,
                    controller.DecisionReason.FIXED.value
                    if safe
                    else controller.DecisionReason.UNSAFE.value,
                )

    def test_fixed_policy_ignores_evaluation_cadence(self):
        """FixedPolicy never calls `evaluation_due`, so (2) cannot reach it."""
        policy = controller.FixedPolicy(24)
        for iteration in range(5):
            decision = policy.decide(snapshot(0.0, iteration), 44, True)
            self.assertEqual(decision.target_decode_sms, 24)
            self.assertEqual(decision.reason, controller.DecisionReason.FIXED.value)


class ProfileBuilderTest(unittest.TestCase):
    """`profile_cli build` must not invent an engine_source_hash."""

    def setUp(self):
        benchmarks = str(Path(__file__).parents[1] / "benchmarks")
        if benchmarks not in sys.path:
            sys.path.insert(0, benchmarks)
        from pdmux_eval import profile_cli

        # The CLI would otherwise import the INSTALLED engine profile module
        # (and all of sglang with it); this module is about the tracked source.
        self.cli = profile_cli
        patch = mock.patch.object(
            profile_cli, "_load_profile_module", lambda: profile
        )
        patch.start()
        self.addCleanup(patch.stop)

    def build(self, environment_extra):
        item = make_profile()
        metadata = {
            key: value
            for key, value in item.to_dict().items()
            if key not in ("points", "environment")
        }
        metadata["environment"] = {
            **item.to_dict()["environment"],
            **environment_extra,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            with (root / "points.csv").open("w", newline="", encoding="utf-8") as sink:
                writer = csv.DictWriter(
                    sink,
                    fieldnames=[
                        "decode_sms", "batch_size", "context_tokens",
                        "itl_p50_ms", "itl_p95_ms", "itl_p99_ms",
                        "repeats", "measured_steps", "residual_p95_ms",
                    ],
                )
                writer.writeheader()
                writer.writerow({
                    "decode_sms": 44, "batch_size": 1, "context_tokens": 256,
                    "itl_p50_ms": 20.0, "itl_p95_ms": 25.0, "itl_p99_ms": 30.0,
                    "repeats": 5, "measured_steps": 30, "residual_p95_ms": 0.5,
                })
            warnings = io.StringIO()
            with redirect_stderr(warnings):
                self.cli.build(
                    root / "metadata.json", root / "points.csv", root / "out.json"
                )
            built = profile.HybridModelProfileV1.load(root / "out.json")
        return built, warnings.getvalue()

    def test_missing_hash_is_warned_about_and_left_empty(self):
        built, warning = self.build({"engine_source_hash": ""})
        self.assertEqual(built.environment.engine_source_hash, "")
        self.assertIn("engine_source_hash", warning)
        self.assertIn("fail-closed", warning)
        self.assertFalse(built.is_compatible(environment())[0])

    def test_supplied_hash_is_preserved_without_warning(self):
        built, warning = self.build({"engine_source_hash": "c0ffee" * 10})
        self.assertEqual(built.environment.engine_source_hash, "c0ffee" * 10)
        self.assertEqual(warning, "")
        self.assertTrue(built.is_compatible(environment())[0])


if __name__ == "__main__":
    unittest.main()
