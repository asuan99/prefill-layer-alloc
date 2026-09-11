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

RUNNING AGAINST A VARIANT.  The loop fakes and the mixin import live in
pdmux_loop_fakes.py.  By default the mixin is imported from the installed
runtime tree (as the other mixin tests do), so a stale dev tree is caught.
Set PDMUX_MIXIN_UNDER_TEST=/path/to/multiplexing_mixin.py to load a specific
file instead -- used to show that the tests fail on the pre-fix code and on
the rejected fix variants, without touching the shared dev tree.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pdmux_loop_fakes import (  # noqa: E402  (after the sys.path insert)
    MIXIN_SOURCE,
    FakeReq,
    LoopScheduler,
    LoopTestBase,
    run_loop,
)
from sglang.srt.multiplex.controller import (  # noqa: E402
    CoarseGrainedController,
    FixedPolicy,
    GenericDynamicPolicy,
    RuntimeSnapshot,
    SplitDecision,
)


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


class _LoopTestBase(LoopTestBase):
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
