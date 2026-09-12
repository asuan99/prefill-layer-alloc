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

2026-09-12(3) adds four more, from the two tensions (1) left open:

  (4) the roadmap's "immediate safe-boundary upshift" (:976) is a SEPARATE,
      OFF-CADENCE clause, not something that waits for :975's
      `max(4 iterations, 100 ms)`.  Mutant that fails: dropping the
      `_live_underprediction` disjunct from `evaluation_due` (the reaction is
      then held to the next epoch).
  (5) the overload streak counts EPOCHS, not evaluations, so (4) cannot arm
      the admission limit in two consecutive decode iterations (~27-85 ms)
      instead of the roadmap's two epochs.  Mutant that fails: the
      unconditional `overload_streak + 1 if overloaded else 0`.
  (6) :979's trigger is a DISJUNCTION -- "D108 risk" OR "occupancy 90%".  Two
      mutants fail: the old flat `AND` (occupancy alone can then never limit),
      and a flat `OR` over all three terms (the fail-closed `inf` upper bound
      of an incompatible profile would then throttle admission at any
      occupancy).
  (7) `off_cadence` is carried on every evaluation decision, so a future GPU
      run can be audited for whether (4) ever fires.  It is emitted with the
      `controller_decision` record; that end is pinned in
      test_r2_admission_persistence.py.

`bucket_changed` is deliberately NOT implemented: the serving path never sets
it (`_r2_decide_idx`), and nothing in src/multiplex defines a bucket.  The
parameter stays for API/test compatibility and the docstrings say so.

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


def fallback_estimate(decode_sms=44):
    """What the estimator returns with NO usable profile.

    `upper_bound_itl_ms=inf` is the fail-closed bound and the default
    `fallback_decode_sms` is 44 -- BELOW the emergency state.  This is the
    shape that a flat-`OR` :979 trigger would throttle unconditionally, and
    the reason the D108-risk term stays conjoined with `target >=
    emergency_state`.
    """
    return profile.DecodeFloorEstimate(
        decode_sms, math.nan, math.inf, 6.0, 0.0, True, "incompatible_profile"
    )


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
        # The HOLD iteration has to be CALM.  Since 2026-09-12(3) an iteration
        # that is STILL violating the ITL SLO is an EVALUATION by design
        # (gate (4)), so the persistence being pinned here is the one that
        # matters in the runtime: the limit set at the last evaluation stays in
        # force through the non-evaluation iterations, including after the
        # instantaneous reading has fallen back under the trigger.
        held = policy.decide(snapshot(20.2, 11), 108, True)
        self.assertEqual(held.reason, controller.DecisionReason.HOLD.value)
        self.assertTrue(
            held.admission_limited,
            "the fallback-driven limit vanished on the next iteration",
        )

    def test_a_still_violating_iteration_is_an_evaluation_not_a_hold(self):
        """Companion to the above: the calm snapshot is not a workaround."""
        item = make_profile(engine_source_hash="")
        estimator = profile.ConservativeDecodeFloorEstimator(item)
        policy = controller.HybridInformedPolicy(
            estimator,
            controller=controller.CoarseGrainedController(
                min_epoch_s=10.0, min_epoch_iterations=4
            ),
            runtime_environment=environment(),
        )
        policy.decide(snapshot(0.0, 0, itl=90.0), 44, True)
        still_violating = policy.decide(snapshot(0.2, 1, itl=90.0), 108, True)
        self.assertNotEqual(
            still_violating.reason, controller.DecisionReason.HOLD.value
        )
        self.assertTrue(still_violating.off_cadence)


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

    def test_the_emergency_path_is_a_no_op_for_fixed_policy(self):
        """(4)-(6) live on `CoarseGrainedController`; FixedPolicy has none.

        Claim D / P1-P2 arms run `PDMUX_R2_POLICY` unset or `fixed`, so an
        off-cadence emergency evaluation, the epoch-gated overload streak and
        the :979 disjunction are all unreachable for them.
        """
        policy = controller.FixedPolicy(44)
        for attribute in (
            "controller", "evaluation_due", "_live_underprediction",
            "_cadence_due", "_overload_epoch_elapsed", "overload_streak",
        ):
            self.assertFalse(
                hasattr(policy, attribute),
                f"FixedPolicy acquired controller machinery: {attribute}",
            )
        for now, iteration in ((0.0, 0), (0.001, 1), (0.002, 2), (0.003, 3)):
            decision = policy.decide(
                snapshot(now, iteration, itl=500.0, kv=0.99, running=0.99),
                44,
                True,
            )
            self.assertEqual(decision.target_decode_sms, 44)
            self.assertEqual(
                decision.reason, controller.DecisionReason.FIXED.value
            )
            self.assertFalse(decision.admission_limited)
            self.assertFalse(
                decision.off_cadence,
                "FixedPolicy has no cadence to be off",
            )

    def test_fixed_policy_ignores_evaluation_cadence(self):
        """FixedPolicy never calls `evaluation_due`, so (2) cannot reach it."""
        policy = controller.FixedPolicy(24)
        for iteration in range(5):
            decision = policy.decide(snapshot(0.0, iteration), 44, True)
            self.assertEqual(decision.target_decode_sms, 24)
            self.assertEqual(decision.reason, controller.DecisionReason.FIXED.value)


class OffCadenceEmergencyUpshiftTest(unittest.TestCase):
    """(4) `:976` is an IMMEDIATE clause, independent of the `:975` cadence."""

    def controller(self, **overrides):
        """Cadence deliberately far away (10 s AND 100 iterations), so every
        evaluation these tests see can only come from the emergency clause."""
        fields = dict(min_epoch_s=10.0, min_epoch_iterations=100)
        fields.update(overrides)
        return controller.CoarseGrainedController(**fields)

    def settled(self, ctl):
        """One on-cadence evaluation, so the next call is inside an epoch."""
        first = ctl.stabilize(calm_estimate(), snapshot(0.0, 0), 24, True)
        self.assertFalse(first.off_cadence, "the first call IS on cadence")
        self.assertEqual(first.target_decode_sms, 24)
        return ctl

    def test_itl_violation_upshifts_on_the_same_iteration(self):
        ctl = self.settled(self.controller())
        probe = snapshot(0.010, 1, itl=90.0)
        self.assertFalse(ctl._cadence_due(probe), "scenario broken: on cadence")
        reacted = ctl.stabilize(calm_estimate(), probe, 24, True)
        self.assertEqual(
            reacted.reason,
            controller.DecisionReason.LIVE_UNDERPREDICTION.value,
            "the ITL violation waited for the next epoch instead of "
            "upshifting immediately (roadmap :976)",
        )
        self.assertEqual(reacted.target_decode_sms, 34)
        self.assertTrue(reacted.off_cadence)

    def test_occupancy_violations_are_immediate_too(self):
        for label, extra in (
            ("kv", {"kv": 0.85}),
            ("running_batch", {"running": 0.85}),
        ):
            with self.subTest(label):
                ctl = self.settled(self.controller())
                reacted = ctl.stabilize(
                    calm_estimate(), snapshot(0.010, 1, **extra), 24, True
                )
                self.assertEqual(reacted.target_decode_sms, 34)
                self.assertTrue(reacted.off_cadence)

    def test_the_same_predicate_drives_the_trigger_and_the_action(self):
        """Single definition site: whenever the emergency makes an iteration an
        evaluation, that evaluation IS the upshift, and vice versa.  Two copies
        of the predicate could disagree here."""
        cases = (
            (90.0, 0.10, 0.20, True),
            (30.0, 0.85, 0.20, True),
            (30.0, 0.10, 0.85, True),
            (60.0, 0.84, 0.84, False),  # exactly AT the SLO is not over it
            (30.0, 0.10, 0.20, False),
        )
        for itl, kv, running, fires in cases:
            with self.subTest(itl=itl, kv=kv, running=running):
                ctl = self.settled(self.controller())
                probe = snapshot(0.010, 1, itl=itl, kv=kv, running=running)
                self.assertFalse(ctl._cadence_due(probe))
                self.assertEqual(ctl._live_underprediction(probe), fires)
                self.assertEqual(ctl.evaluation_due(probe), fires)
                decision = ctl.stabilize(calm_estimate(), probe, 24, True)
                self.assertEqual(
                    decision.reason,
                    controller.DecisionReason.LIVE_UNDERPREDICTION.value
                    if fires
                    else controller.DecisionReason.HOLD.value,
                )
                self.assertEqual(decision.off_cadence, fires)

    def test_an_unsafe_boundary_still_blocks_the_off_cadence_upshift(self):
        """`:976` says "immediate SAFE-BOUNDARY upshift".  An off-cadence
        evaluation at an unsafe boundary must not switch -- and, because the
        emergency clause re-fires while the violation lasts, it must retry on
        the NEXT iteration rather than wait out the epoch it just consumed."""
        ctl = self.settled(self.controller())
        blocked = ctl.stabilize(
            calm_estimate(), snapshot(0.010, 1, itl=90.0), 24, False
        )
        self.assertEqual(blocked.reason, controller.DecisionReason.UNSAFE.value)
        self.assertEqual(blocked.target_decode_sms, 24)
        self.assertEqual(blocked.requested_decode_sms, 34)
        self.assertFalse(blocked.safe)
        self.assertTrue(blocked.off_cadence)
        retried = ctl.stabilize(
            calm_estimate(), snapshot(0.020, 2, itl=90.0), 24, True
        )
        self.assertEqual(
            retried.target_decode_sms,
            34,
            "the blocked emergency consumed its epoch and the retry had to "
            "wait for the cadence",
        )

    def test_the_cadence_resumes_after_the_violation_ends(self):
        """An emergency evaluation RESTARTS the epoch clock: the next normal
        evaluation is at least one full epoch after it, not after the last
        on-cadence one."""
        ctl = controller.CoarseGrainedController(
            min_epoch_s=0.100, min_epoch_iterations=4
        )
        ctl.stabilize(calm_estimate(), snapshot(0.0, 0), 24, True)
        emergency = ctl.stabilize(
            calm_estimate(), snapshot(0.010, 1, itl=90.0), 24, True
        )
        self.assertTrue(emergency.off_cadence)
        for now, iteration in ((0.020, 2), (0.060, 3), (0.105, 4)):
            calm = ctl.stabilize(calm_estimate(), snapshot(now, iteration), 34, True)
            self.assertEqual(
                calm.reason,
                controller.DecisionReason.HOLD.value,
                f"a normal evaluation ran at t={now}: the epoch clock was "
                "measured from the last ON-cadence evaluation, not from the "
                "emergency one",
            )
        resumed = ctl.stabilize(calm_estimate(), snapshot(0.111, 5), 34, True)
        self.assertNotEqual(resumed.reason, controller.DecisionReason.HOLD.value)
        self.assertFalse(resumed.off_cadence)

    def test_bucket_changed_is_not_set_by_the_serving_path(self):
        """(7) The parameter still exists and still forces an evaluation, but
        nothing in the runtime passes it -- so the immediacy comes from the
        emergency clause.  Pinned so a later reader does not "restore" a bucket
        implementation that never existed."""
        import inspect

        ctl = self.settled(self.controller())
        self.assertTrue(
            ctl.evaluation_due(snapshot(0.010, 1), bucket_changed=True)
        )
        for target in (
            controller.CoarseGrainedController.stabilize,
            controller.HybridInformedPolicy.decide,
        ):
            self.assertIn(
                "bucket_changed",
                inspect.signature(target).parameters,
                "the parameter was deleted; the roadmap clause it documents "
                "is still unimplemented",
            )
            self.assertIn(
                "NOT",
                target.__doc__ or "",
                "the docstring must say the serving path does not set it",
            )


class OverloadStreakCountsEpochsTest(unittest.TestCase):
    """(5) the streak counts EPOCHS, so (4) cannot arm the limit in 2 steps."""

    def controller(self):
        return controller.CoarseGrainedController(
            min_epoch_s=10.0, min_epoch_iterations=4
        )

    def test_violating_iterations_inside_one_epoch_do_not_limit(self):
        ctl = self.controller()
        first = ctl.stabilize(calm_estimate(), snapshot(0.0, 0, kv=0.92), 24, True)
        self.assertFalse(first.admission_limited)
        self.assertEqual(ctl.overload_streak, 1)
        for now, iteration in ((0.020, 1), (0.040, 2), (0.060, 3)):
            step = ctl.stabilize(
                calm_estimate(), snapshot(now, iteration, kv=0.92), 24, True
            )
            self.assertTrue(step.off_cadence, "scenario broken: on cadence")
            self.assertFalse(
                step.admission_limited,
                f"admission was limited at t={now}, {now * 1000:.0f} ms after "
                "the overload started: the streak counted decode iterations "
                "instead of the roadmap's 2 epochs (:979)",
            )
            self.assertEqual(ctl.overload_streak, 1)
        second = ctl.stabilize(calm_estimate(), snapshot(10.5, 10, kv=0.92), 24, True)
        self.assertEqual(ctl.overload_streak, 2)
        self.assertTrue(
            second.admission_limited,
            "the epoch gate also blocked the LEGITIMATE second epoch",
        )

    def test_a_calm_evaluation_still_resets_the_streak_immediately(self):
        ctl = self.controller()
        ctl.stabilize(calm_estimate(), snapshot(0.0, 0, kv=0.92), 24, True)
        self.assertEqual(ctl.overload_streak, 1)
        ctl.stabilize(calm_estimate(), snapshot(10.5, 10), 24, True)
        self.assertEqual(
            ctl.overload_streak, 0, "the reset must not be epoch-gated"
        )

    def test_zero_cadence_configuration_counts_every_evaluation(self):
        """`min_epoch_s=0, min_epoch_iterations=0` is what the event-loop tests
        configure; the gate must be a no-op there (`x - y >= 0` is true for a
        non-decreasing clock and iteration counter, including for the very same
        timestamp), so that arm behaves exactly as before."""
        ctl = controller.CoarseGrainedController(
            (16, 24, 34, 44),
            108,
            min_epoch_s=0.0,
            min_epoch_iterations=0,
            min_dwell_s=0.0,
            min_dwell_iterations=0,
        )
        first = ctl.stabilize(calm_estimate(), snapshot(0.0, 0, kv=0.92), 24, True)
        self.assertFalse(first.admission_limited)
        second = ctl.stabilize(calm_estimate(), snapshot(0.0, 0, kv=0.92), 24, True)
        self.assertTrue(
            second.admission_limited,
            "the epoch gate changed the zero-cadence arm, where every call is "
            "an epoch by construction",
        )

    def test_release_restarts_the_epoch_clock_as_well(self):
        ctl = self.controller()
        ctl.stabilize(calm_estimate(), snapshot(0.0, 0, kv=0.92), 24, True)
        ctl.stabilize(calm_estimate(), snapshot(10.5, 10, kv=0.92), 24, True)
        ctl.release_admission_limit()
        self.assertEqual(ctl.overload_streak, 0)
        self.assertFalse(
            ctl.stabilize(
                calm_estimate(), snapshot(10.6, 11, kv=0.92), 24, True
            ).admission_limited,
            "one overloaded evaluation after a release re-latched immediately",
        )
        self.assertTrue(
            ctl.stabilize(
                calm_estimate(), snapshot(30.0, 30, kv=0.92), 24, True
            ).admission_limited
        )


class OverloadTriggerDisjunctionTest(unittest.TestCase):
    """(6) :979 = "D108 risk" OR "occupancy 90%", with the inf-fallback guard."""

    def controller(self):
        return controller.CoarseGrainedController(
            min_epoch_s=10.0, min_epoch_iterations=4
        )

    def test_occupancy_alone_limits_after_two_epochs(self):
        """The half that the old `AND` composition made unreachable."""
        for label, extra in (
            ("kv", {"kv": 0.92}),
            ("running_batch", {"running": 0.92}),
        ):
            with self.subTest(label):
                ctl = self.controller()
                first = ctl.stabilize(
                    calm_estimate(), snapshot(0.0, 0, **extra), 16, True
                )
                self.assertFalse(first.admission_limited)
                second = ctl.stabilize(
                    calm_estimate(), snapshot(10.0, 10, **extra), 24, True
                )
                for decision in (first, second):
                    self.assertLess(
                        decision.target_decode_sms,
                        108,
                        "scenario broken: the target reached the emergency "
                        "state, so this no longer isolates the occupancy half",
                    )
                self.assertTrue(
                    second.admission_limited,
                    "90% occupancy for two epochs never limited admission: "
                    "the occupancy half of :979 is dead code",
                )

    def test_inf_fallback_does_not_limit_at_low_occupancy(self):
        """The rejected flat-`OR` variant.  `inf > slo` is ALWAYS true, so a
        hybrid arm booted without a compatible profile would throttle its own
        admission at any occupancy -- silently, and only in that arm."""
        ctl = self.controller()
        for index in range(6):
            decision = ctl.stabilize(
                fallback_estimate(), snapshot(20.0 * index, 10 * index), 44, True
            )
            self.assertFalse(
                decision.admission_limited,
                "an incompatible profile throttled admission on an idle "
                f"engine (iteration {index})",
            )
            self.assertEqual(ctl.overload_streak, 0)

    def test_inf_fallback_still_limits_at_the_emergency_state(self):
        """The D108-risk half stays alive: unbounded prediction AT D108 is
        exactly the condition :979 names."""
        ctl = self.controller()
        first = ctl.stabilize(fallback_estimate(108), snapshot(0.0, 0), 44, True)
        self.assertEqual(first.target_decode_sms, 108)
        self.assertFalse(first.admission_limited)
        second = ctl.stabilize(
            fallback_estimate(108), snapshot(20.0, 10), 108, True
        )
        self.assertTrue(second.admission_limited)


class DownshiftIsUnchangedTest(unittest.TestCase):
    """(4)/(5) must not touch the downshift side: hysteresis and dwell."""

    def controller(self):
        return controller.CoarseGrainedController(
            min_epoch_s=10.0,
            min_epoch_iterations=4,
            min_dwell_s=0.200,
            min_dwell_iterations=8,
            downshift_epochs=3,
        )

    def test_the_emergency_path_never_lowers_the_target(self):
        for itl, kv, running in (
            (90.0, 0.10, 0.20),
            (30.0, 0.92, 0.20),
            (30.0, 0.10, 0.92),
        ):
            with self.subTest(itl=itl, kv=kv, running=running):
                ctl = self.controller()
                held = ctl.stabilize(calm_estimate(), snapshot(0.0, 0), 44, True)
                self.assertEqual(held.target_decode_sms, 44)
                reacted = ctl.stabilize(
                    calm_estimate(),
                    snapshot(0.010, 1, itl=itl, kv=kv, running=running),
                    44,
                    True,
                )
                self.assertGreaterEqual(
                    reacted.target_decode_sms,
                    44,
                    "the off-cadence emergency path downshifted",
                )
                self.assertEqual(ctl.downshift_streak, 0)

    def test_hysteresis_still_takes_three_epochs(self):
        ctl = self.controller()
        for index, (now, iteration) in enumerate(
            ((0.0, 0), (10.0, 10), (20.0, 20)), start=1
        ):
            out = ctl.stabilize(calm_estimate(), snapshot(now, iteration), 44, True)
            if index < 3:
                self.assertEqual(
                    out.reason,
                    controller.DecisionReason.DOWNSHIFT_HYSTERESIS.value,
                )
                self.assertEqual(out.target_decode_sms, 44)
            else:
                self.assertEqual(out.target_decode_sms, 24)
                self.assertEqual(
                    out.reason, controller.DecisionReason.PROFILE_FLOOR.value
                )

    def test_dwell_still_blocks_a_downshift_after_a_recent_transition(self):
        ctl = self.controller()
        ctl.stabilize(calm_estimate(), snapshot(0.0, 0), 44, True)
        ctl.stabilize(calm_estimate(), snapshot(10.0, 10), 44, True)
        ctl.last_transition_s, ctl.last_transition_iteration = 19.95, 19
        blocked = ctl.stabilize(calm_estimate(), snapshot(20.0, 20), 44, True)
        self.assertEqual(blocked.reason, controller.DecisionReason.DWELL.value)
        self.assertEqual(blocked.target_decode_sms, 44)
        freed = ctl.stabilize(calm_estimate(), snapshot(30.0, 30), 44, True)
        self.assertEqual(freed.target_decode_sms, 24)


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
