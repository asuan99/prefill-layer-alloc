"""The admission limit in the REAL event loop: it persists, and it lifts.

Companion to test_r2_admission_latch.py (which pins the 2026-09-11 liveness
fix).  This module pins the 2026-09-12 roadmap-canonical behaviour on the same
machinery:

  * a limit set by a controller EVALUATION survives the non-evaluation (HOLD)
    iterations until the next evaluation -- before this, `stabilize`'s HOLD
    return used the `SplitDecision` default `admission_limited=False` and
    `_r2_decide_idx` overwrote `r2_admission_limited` with it, so the limit
    lived about one iteration (EXPERIMENT_ROADMAP.md:979 asks for a limit that
    lasts while the overload condition does);
  * it still LIFTS by the policy, with decode still running -- "persist" must
    not degenerate into the absorbing state that 2026-09-11 removed;
  * the empty-running-batch backstop clears the CONTROLLER's carried verdict
    too, not just the scheduler flag;
  * (2026-09-12(3)) the `off_cadence` audit flag reaches the
    `controller_decision` record through the real `_r2_decide_idx`, so a later
    GPU run can be checked for whether the off-cadence emergency path
    (EXPERIMENT_ROADMAP.md:976) ever fires.  Mutant that fails: dropping the
    field from the emit in `multiplexing_mixin._r2_decide_idx`.

WHAT IS REAL HERE.  The unmodified `event_loop_pdmux`,
`update_split_prefill_batch`, `_r2_decide_idx`, `_r2_admission_recheck`,
`_r2_admission_holds`, the real `GenericDynamicPolicy` and the real
`CoarseGrainedController`.  Streams/events/TP group are the shared CPU fakes
(pdmux_loop_fakes.py).  Cadence in the scenarios is `min_epoch_iterations=4`
with `min_epoch_s=0.0`, so "evaluation due" depends only on the scripted decode
iteration count and the tests are wall-clock independent.

RUNNING AGAINST A VARIANT.  PDMUX_MIXIN_UNDER_TEST=/path/to/
multiplexing_mixin.py swaps the mixin (see pdmux_loop_fakes.py);
PDMUX_MULTIPLEX_SRC=/path/to/multiplex swaps controller.py/profile.py.  The
pre-fix controller fails `test_hold_iterations_keep_the_limit` and
`test_waiting_prefill_waits_for_the_policy_to_lift_the_limit`; a mixin without
the controller-release hook fails `ControllerReleaseHookTest`.

NO GPU, NO PERFORMANCE CLAIM.  generic/hybrid have zero GPU measurements; this
is plumbing determinism only.
"""

import importlib.util
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pdmux_loop_fakes import (  # noqa: E402  (after the sys.path insert)
    MIXIN_SOURCE,
    FakeBatch,
    FakeReq,
    LoopScheduler,
    LoopTestBase,
    run_loop,
)

_VARIANT_SRC = os.environ.get("PDMUX_MULTIPLEX_SRC", "")
if _VARIANT_SRC:
    # Load profile.py first and register it as `profile`: controller.py's
    # CPU-only fallback imports it by bare name (controller.py:17).  The
    # shadow is removed again right away -- it breaks later `import sglang`.
    def _load(name):
        spec = importlib.util.spec_from_file_location(
            name, os.path.join(_VARIANT_SRC, f"{name}.py")
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    _load("profile")  # only to satisfy controller.py's bare-name import
    controller_module = _load("controller")
    sys.modules.pop("profile", None)
    sys.modules.pop("controller", None)
else:
    from sglang.srt.multiplex import controller as controller_module  # noqa: E402

# The profile module is ALWAYS the installed one, even under
# PDMUX_MULTIPLEX_SRC: `ProfileEnvironmentRecordTest` patches the symbol that
# the installed mixin's local import resolves
# (`multiplexing_mixin._r2_runtime_environment`), so patching a standalone copy
# would silently test nothing.  The hash-axis SEMANTICS are pinned separately,
# against the tracked source, in test_controller_defaults.py.
from sglang.srt.multiplex import profile as profile_module  # noqa: E402

CONTROLLER_SOURCE = controller_module.__file__
CoarseGrainedController = controller_module.CoarseGrainedController
FixedPolicy = controller_module.FixedPolicy
GenericDynamicPolicy = controller_module.GenericDynamicPolicy

STATES = (16, 24, 34, 44)


def generic_policy(min_epoch_iterations=4):
    """Real policy + controller, evaluating every `min_epoch_iterations`.

    `min_epoch_s=0.0` removes wall-clock from the cadence, so every iteration
    that is not a multiple of the iteration threshold is a HOLD -- the
    iterations this module is about.
    """
    return GenericDynamicPolicy(
        STATES,
        emergency_state=108,
        controller=CoarseGrainedController(
            STATES,
            108,
            min_epoch_s=0.0,
            min_epoch_iterations=min_epoch_iterations,
            min_dwell_s=0.0,
            min_dwell_iterations=0,
        ),
    )


class AdmissionLimitSurvivesHoldIterationsTest(LoopTestBase):
    """cap 10; nine long decodes hold occupancy at/above 0.9.

    Q's prefill span is where the controller reaches two overloaded
    evaluations; R arrives while the limit is on and must wait for a later
    evaluation to lift it.
    """

    # Built fresh per run: the loop MUTATES the requests (output_ids), so a
    # shared class-level dict would make every test after the first one run
    # with already-finished requests.
    @staticmethod
    def arrivals():
        return {
            0: [FakeReq(f"W{i}", max_new=14 + i) for i in range(9)],
            6: [FakeReq("X", max_new=60)],
            14: [FakeReq("Q", max_new=3)],
            17: [FakeReq("R", max_new=3)],
        }

    def scenario(self):
        sched = LoopScheduler(
            generic_policy(), "generic", self.arrivals(), 70,
            admit_per_batch=9, max_running_requests=10,
        )
        run_loop(self.install(sched))
        return sched

    @staticmethod
    def limited_holds(sched):
        return [
            d
            for d in sched.decisions()
            if d[2]["reason"] == "hold" and d[2]["admission_limited"]
        ]

    def test_hold_iterations_keep_the_limit(self):
        sched = self.scenario()
        evaluated = [
            d for d in sched.decisions()
            if d[2]["reason"] != "hold" and d[2]["admission_limited"]
        ]
        self.assertGreaterEqual(
            len(evaluated), 1, "scenario broken: no evaluation ever limited"
        )
        self.assertGreaterEqual(
            len(self.limited_holds(sched)),
            2,
            "every HOLD iteration reported admission_limited=False, so the "
            "limit an evaluation had just set survived about one iteration "
            f"[{CONTROLLER_SOURCE}]",
        )

    def test_waiting_prefill_waits_for_the_policy_to_lift_the_limit(self):
        sched = self.scenario()
        admitted = sched.admissions("R")
        self.assertTrue(admitted, f"R was never admitted [{MIXIN_SOURCE}]")
        _, _, iteration, running_at_admit = admitted[0]
        self.assertGreater(
            iteration,
            17,
            "R was admitted on the iteration it arrived, although the limit "
            "was on",
        )
        # R arrived at 17; with the limit honoured it waits for the evaluation
        # that clears the overload streak (iteration 23 in this scenario).
        self.assertGreaterEqual(
            iteration,
            20,
            "R was admitted during a HOLD iteration: the limit did not survive "
            f"the iterations between two evaluations [{CONTROLLER_SOURCE}]",
        )
        self.assertGreater(
            running_at_admit,
            0,
            "R was admitted only after decode drained -- that is the "
            "empty-batch backstop, not the policy lifting the limit",
        )
        self.assertEqual(
            sched.events("r2_admission_released"),
            [],
            "the backstop fired: this scenario must be resolved by the policy",
        )

    def test_the_limit_is_not_permanent(self):
        sched = self.scenario()
        decisions = sched.decisions()
        self.assertTrue(
            any(d[2]["admission_limited"] for d in decisions), "scenario broken"
        )
        self.assertFalse(
            decisions[-1][2]["admission_limited"],
            "the run ended with the limit still asserted: inheritance "
            "degenerated into a latch",
        )
        self.assertFalse(sched.r2_admission_limited)

    def test_off_cadence_evaluations_are_visible_in_the_telemetry(self):
        """Audit hook, NOT a performance signal.

        The generic/hybrid policies have zero GPU measurements; this only says
        that IF the off-cadence emergency path fires, a `controller_decision`
        record says so.  In this scenario occupancy sits at/above 0.85 for a
        stretch, which is exactly the roadmap's :976 trigger.
        """
        records = [decision[2] for decision in self.scenario().decisions()]
        self.assertTrue(records, "scenario broken: no decisions")
        missing = [r for r in records if "off_cadence" not in r]
        self.assertEqual(
            missing,
            [],
            "controller_decision records carry no off_cadence field, so a GPU "
            f"run could not be audited for this path [{MIXIN_SOURCE}]",
        )
        self.assertTrue(
            any(r["off_cadence"] for r in records),
            "scenario broken: the off-cadence path never fired, so this test "
            "would pass with the field hard-wired to False",
        )
        self.assertFalse(
            any(r["off_cadence"] for r in records if r["reason"] == "hold"),
            "a HOLD iteration was reported as an off-cadence EVALUATION; "
            "HOLDs are not evaluations",
        )

    def test_the_recheck_path_keeps_consulting_the_policy_while_limited(self):
        """Liveness precondition: the limit is re-examined every iteration."""
        sched = self.scenario()
        self.assertTrue(
            sched.events("r2_admission_recheck"),
            "the policy was never consulted on a decode-only iteration, so "
            "only the empty-batch backstop could ever lift the limit",
        )


class ControllerReleaseHookTest(LoopTestBase):
    """`_r2_admission_holds` must clear the controller's carried verdict."""

    def scheduler(self, policy, policy_name):
        sched = LoopScheduler(policy, policy_name, {}, 1)
        return self.install(sched)

    def test_empty_batch_backstop_clears_the_controller_state(self):
        policy = generic_policy()
        sched = self.scheduler(policy, "generic")
        policy.controller.admission_limited = True
        policy.controller.overload_streak = 5
        sched.r2_admission_limited = True
        sched.running_batch = FakeBatch()  # drained

        self.assertFalse(sched._r2_admission_holds())
        self.assertFalse(sched.r2_admission_limited)
        self.assertFalse(
            policy.controller.admission_limited,
            "the scheduler flag was cleared but the controller still carries "
            "the verdict, so the next HOLD decision re-asserts the limit the "
            f"backstop just released [{MIXIN_SOURCE}]",
        )
        self.assertEqual(policy.controller.overload_streak, 0)
        self.assertEqual(
            sched.events("r2_admission_released")[0][2]["reason"],
            "running_batch_empty",
        )

    def test_backstop_does_not_release_while_decode_is_running(self):
        policy = generic_policy()
        sched = self.scheduler(policy, "generic")
        policy.controller.admission_limited = True
        sched.r2_admission_limited = True
        sched.running_batch = FakeBatch([FakeReq("A", max_new=9)])

        self.assertTrue(sched._r2_admission_holds())
        self.assertTrue(sched.r2_admission_limited)
        self.assertTrue(policy.controller.admission_limited)
        self.assertEqual(sched.events("r2_admission_released"), [])

    def test_fixed_policy_exposes_no_controller_to_release(self):
        """Claim D arms: the release hook is a no-op lookup, not an error."""
        policy = FixedPolicy(44)
        sched = self.scheduler(policy, "fixed")
        self.assertIsNone(getattr(policy, "controller", None))
        sched.r2_admission_limited = True  # cannot happen via FixedPolicy
        sched.running_batch = FakeBatch()
        self.assertFalse(sched._r2_admission_holds())
        self.assertFalse(sched.r2_admission_limited)


class ProfileEnvironmentRecordTest(LoopTestBase):
    """The engine-source hash reaches the run record (telemetry)."""

    HASH = "a" * 64

    def profile(self, **environment_overrides):
        fields = dict(
            engine_commit="commit-a",
            gpu_name="A100",
            gpu_sm_count=108,
            cuda_driver="13.0",
            attention_backend="triton",
            cuda_graph=True,
            piecewise_cuda_graph=False,
            engine_source_hash=self.HASH,
        )
        fields.update(environment_overrides)
        return profile_module.HybridModelProfileV1(
            model_id="test/hybrid",
            model_revision="r1",
            model_config_hash="hash",
            environment=profile_module.RuntimeEnvironment(**fields),
            attention_layers=8,
            ssm_layers=24,
            attention_kind="GQA",
            num_attention_heads=32,
            num_kv_heads=8,
            parameter_count=1_000_000,
            points=[
                profile_module.DecodeLatencyPoint(
                    44, 1, 256, 20.0, 25.0, 30.0, repeats=5, measured_steps=30
                )
            ],
        )

    def build(self, profile, runtime_hash, **environment_overrides):
        sched = self.install(LoopScheduler(generic_policy(), "hybrid", {}, 1))
        fields = dict(
            engine_commit="commit-b-docs-only",
            gpu_name="A100",
            gpu_sm_count=108,
            cuda_driver="13.0",
            attention_backend="triton",
            cuda_graph=True,
            piecewise_cuda_graph=False,
        )
        fields.update(environment_overrides)
        with mock.patch.object(
            profile_module, "engine_source_hash", lambda *a, **k: runtime_hash
        ):
            environment = sched._r2_runtime_environment(profile, **fields)
        record = sched.events("r2_profile_environment")
        self.assertEqual(len(record), 1, "the provenance record is missing")
        return environment, record[0][2]

    def test_hash_is_recorded_and_commit_is_kept_as_provenance(self):
        environment, record = self.build(self.profile(), self.HASH)
        self.assertEqual(environment.engine_source_hash, self.HASH)
        self.assertEqual(environment.engine_commit, "commit-b-docs-only")
        self.assertEqual(record["engine_source_hash"], self.HASH)
        self.assertEqual(record["profile_engine_source_hash"], self.HASH)
        self.assertEqual(record["engine_commit"], "commit-b-docs-only")
        self.assertEqual(record["profile_engine_commit"], "commit-a")
        self.assertTrue(
            record["compatible"],
            "a docs-only commit difference made the profile incompatible",
        )
        self.assertEqual(record["reasons"], "")

    def test_source_mismatch_is_recorded_as_incompatible(self):
        _environment, record = self.build(self.profile(), "b" * 64)
        self.assertFalse(record["compatible"])
        self.assertIn("engine_source_hash", record["reasons"])

    def test_unavailable_hash_is_recorded_and_fails_closed(self):
        _environment, record = self.build(self.profile(), "")
        self.assertEqual(record["engine_source_hash"], "unavailable")
        self.assertFalse(record["compatible"])
        self.assertIn("fail-closed", record["reasons"])

    def test_legacy_profile_without_hash_is_recorded_as_incompatible(self):
        _environment, record = self.build(
            self.profile(engine_source_hash=""), self.HASH
        )
        self.assertEqual(record["profile_engine_source_hash"], "unavailable")
        self.assertFalse(record["compatible"])
        self.assertIn("missing in profile", record["reasons"])


if __name__ == "__main__":
    unittest.main()
