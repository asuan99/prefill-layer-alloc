"""Coarse-grained PD-mux controllers shared by all architecture baselines."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Protocol, Sequence

try:
    from .profile import (
        ConservativeDecodeFloorEstimator,
        DecodeFloorEstimate,
        RuntimeEnvironment,
    )
except ImportError:  # Loaded directly by CPU-only tests.
    from profile import (
        ConservativeDecodeFloorEstimator,
        DecodeFloorEstimate,
        RuntimeEnvironment,
    )


class DecisionReason(str, Enum):
    HOLD = "hold"
    FIXED = "fixed"
    PROFILE_FLOOR = "profile_floor"
    GENERIC_PRESSURE = "generic_pressure"
    LIVE_UNDERPREDICTION = "live_underprediction"
    OVERLOAD = "overload"
    DOWNSHIFT_HYSTERESIS = "downshift_hysteresis"
    DWELL = "minimum_dwell"
    UNSAFE = "unsafe_transition"


@dataclass(frozen=True)
class RuntimeSnapshot:
    timestamp_s: float
    decode_iterations: int
    active_decode_sequences: int
    decode_batch_size: int
    context_p50: int
    context_p95: int
    context_max: int
    expected_remaining_output_p90: int
    prefill_queue_depth: int
    oldest_prefill_age_ms: float
    measured_itl_ewma_ms: float
    measured_itl_p95_ms: float
    itl_slo_ms: float
    ttft_risk: float
    kv_occupancy: float
    running_batch_occupancy: float
    prefill_idle_ratio: float = 0.0
    decode_idle_ratio: float = 0.0
    # H5: `worker_overlap_ratio` removed 2026-09-12 -- see dual_worker.py NOTE.


@dataclass(frozen=True)
class SplitDecision:
    current_decode_sms: int
    target_decode_sms: int
    reason: str
    predicted_itl_ms: float = math.nan
    upper_bound_itl_ms: float = math.nan
    confidence: float = 0.0
    safe: bool = True
    admission_limited: bool = False
    emergency: bool = False
    requested_decode_sms: int = 0
    # AUDIT ONLY.  True when this decision came from an evaluation that the
    # emergency disjunct forced BEFORE the normal cadence was due
    # (`CoarseGrainedController.evaluation_due`, roadmap
    # EXPERIMENT_ROADMAP.md:976).  It exists so a later GPU run can be checked
    # for whether that path ever fires -- generic/hybrid have zero GPU
    # measurements, so today it fires only in CPU tests.  HOLD decisions carry
    # False: they are not evaluations.
    off_cadence: bool = False


class MultiplexingPolicy(Protocol):
    def decide(
        self, snapshot: RuntimeSnapshot, current_decode_sms: int, safe_boundary: bool
    ) -> SplitDecision:
        ...


class FixedPolicy:
    def __init__(self, decode_sms: int):
        self.decode_sms = int(decode_sms)

    def decide(
        self, snapshot: RuntimeSnapshot, current_decode_sms: int, safe_boundary: bool
    ) -> SplitDecision:
        target = self.decode_sms if safe_boundary else current_decode_sms
        return SplitDecision(
            current_decode_sms=current_decode_sms,
            target_decode_sms=target,
            reason=DecisionReason.FIXED.value if safe_boundary else DecisionReason.UNSAFE.value,
            safe=safe_boundary,
            requested_decode_sms=self.decode_sms,
        )


class CoarseGrainedController:
    """Stability envelope around a policy-specific desired decode floor."""

    def __init__(
        self,
        states: Sequence[int] = (16, 24, 34, 44),
        emergency_state: int = 108,
        min_epoch_s: float = 0.100,
        min_epoch_iterations: int = 4,
        min_dwell_s: float = 0.200,
        min_dwell_iterations: int = 8,
        downshift_epochs: int = 3,
    ):
        self.states = tuple(sorted({int(state) for state in states}))
        self.emergency_state = int(emergency_state)
        self.min_epoch_s = float(min_epoch_s)
        self.min_epoch_iterations = int(min_epoch_iterations)
        self.min_dwell_s = float(min_dwell_s)
        self.min_dwell_iterations = int(min_dwell_iterations)
        self.downshift_epochs = int(downshift_epochs)
        self.last_evaluation_s = -math.inf
        self.last_evaluation_iteration = -10**12
        self.last_transition_s = -math.inf
        self.last_transition_iteration = -10**12
        self.downshift_streak = 0
        # Counts overload EPOCHS, not evaluations.  Once `evaluation_due` can
        # fire off-cadence (roadmap :976), "2 consecutive evaluations" would
        # mean 2 consecutive decode iterations (~27-85 ms), not the roadmap's
        # "2 epochs" (>= 100 ms AND >= 4 iterations apart at the defaults).
        # That matters because the admission limit is the measured
        # TTFT-explosion lever in this project (CONSENSUS.md HE0: decode
        # starvation -> ITL up -> batch stalls -> admission blocked -> TTFT
        # blows up), so arming it an order of magnitude early is not a
        # conservative error.  `last_overload_epoch_*` gate the increment to at
        # most one per cadence epoch; the sentinels keep the FIRST overloaded
        # evaluation counting.
        self.overload_streak = 0
        self.last_overload_epoch_s = -math.inf
        self.last_overload_epoch_iteration = -10**12
        # Verdict of the LAST EVALUATION, carried by the HOLD decisions in
        # between (roadmap "Controller defaults", EXPERIMENT_ROADMAP.md:979:
        # the limit persists while the 2-epoch overload condition holds, and
        # is lifted by a later evaluation -- not by the next iteration).
        self.admission_limited = False

    @staticmethod
    def _live_underprediction(snapshot: RuntimeSnapshot) -> bool:
        """The roadmap's immediate-upshift predicate (EXPERIMENT_ROADMAP.md:976,
        "immediate safe-boundary upshift on ITL violation or 85% KV/batch
        occupancy").

        SINGLE DEFINITION SITE.  Two callers need it -- `evaluation_due`, so
        the upshift does not have to wait for the cadence, and `stabilize`, to
        pick the upshifted target.  A second copy of these three comparisons
        would be the same-name-two-implementations defect (PROJECT_STATUS.md
        gate #166 family), where the trigger and the action can drift apart.
        """
        return (
            snapshot.measured_itl_p95_ms > snapshot.itl_slo_ms
            or snapshot.kv_occupancy >= 0.85
            or snapshot.running_batch_occupancy >= 0.85
        )

    def _cadence_due(self, snapshot: RuntimeSnapshot) -> bool:
        """The NORMAL cadence alone: `max(4 decode iterations, 100 ms)`
        (EXPERIMENT_ROADMAP.md:975).  BOTH thresholds must be met, which is the
        same AND that `_dwell_satisfied` already uses for `max(8 steps,
        200 ms)`.  The `-inf` / `-10**12` initial sentinels keep the FIRST call
        due.
        """
        return (
            snapshot.timestamp_s - self.last_evaluation_s >= self.min_epoch_s
            and snapshot.decode_iterations - self.last_evaluation_iteration
            >= self.min_epoch_iterations
        )

    def _overload_epoch_elapsed(self, snapshot: RuntimeSnapshot) -> bool:
        """Whether a new CADENCE EPOCH has passed since the overload streak was
        last incremented -- the same `max(min_epoch_iterations, min_epoch_s)`
        AND as `_cadence_due`, kept on its own markers so that off-cadence
        emergency evaluations cannot inflate the streak.  With
        `min_epoch_s=0` and `min_epoch_iterations=0` (the zero-cadence
        configuration used by the loop tests) this is always True, so every
        evaluation counts exactly as it did before the gate existed.
        """
        return (
            snapshot.timestamp_s - self.last_overload_epoch_s >= self.min_epoch_s
            and snapshot.decode_iterations - self.last_overload_epoch_iteration
            >= self.min_epoch_iterations
        )

    def evaluation_due(self, snapshot: RuntimeSnapshot, bucket_changed: bool = False) -> bool:
        """Whether this iteration re-evaluates the split.

        TWO SEPARATE ROADMAP CLAUSES, not one:

          * `:975` "evaluate every `max(4 decode iterations, 100 ms)` or bucket
            change" -- the normal cadence (`_cadence_due`);
          * `:976` "immediate safe-boundary upshift on ITL violation or 85%
            KV/batch occupancy" -- an IMMEDIATE clause, so it has to be able to
            fire BETWEEN cadence points.  Folding it into the cadence (the
            2026-09-12(2) state, where it was only computed inside
            `stabilize`'s evaluation branch) made the worst-case reaction to an
            SLO violation one full epoch.

        `bucket_changed` is NOT SET by the serving path -- see `stabilize`.
        """
        return (
            bucket_changed
            or self._live_underprediction(snapshot)
            or self._cadence_due(snapshot)
        )

    def release_admission_limit(self) -> None:
        """Drop a limit that the RUNTIME released outside the controller.

        Called when the serving loop has already decided the limit cannot
        apply any more (`multiplexing_mixin._r2_admission_holds`: the running
        batch is empty, so the decode the limit protects has drained).  Both
        the carried verdict and the overload streak are cleared, so a new
        limit again needs two fresh overloaded evaluations; without clearing
        them the next HOLD decision would re-assert the limit the runtime just
        released (stale-True, the 2026-09-11 bug class).
        """
        self.admission_limited = False
        self.overload_streak = 0
        self.last_overload_epoch_s = -math.inf
        self.last_overload_epoch_iteration = -10**12

    def _dwell_satisfied(self, snapshot: RuntimeSnapshot) -> bool:
        return (
            snapshot.timestamp_s - self.last_transition_s >= self.min_dwell_s
            and snapshot.decode_iterations - self.last_transition_iteration
            >= self.min_dwell_iterations
        )

    def stabilize(
        self,
        desired: DecodeFloorEstimate,
        snapshot: RuntimeSnapshot,
        current_decode_sms: int,
        safe_boundary: bool,
        bucket_changed: bool = False,
    ) -> SplitDecision:
        """One controller step: cadence gate, then target, then admission.

        `bucket_changed` IS NOT IMPLEMENTED in this runtime.  Its only serving
        call site, `_r2_decide_idx` (`multiplexing_mixin.py:436-440`), never
        passes it, and `src/multiplex` contains no bucket definition at all, so
        it is False in every served iteration -- the roadmap's "or bucket
        change" (:975) has no implementation behind it.  The parameter is kept
        because the API and the unit tests use it, and because deleting it
        would erase the fact that the clause is unimplemented.  The IMMEDIACY
        the roadmap asks for is supplied here by the emergency disjunct of
        `evaluation_due` (:976) instead.  Do NOT invent a bucket definition in
        passing: that would be a new, unregistered policy knob, not a repair.
        """
        current = int(current_decode_sms)
        if not self.evaluation_due(snapshot, bucket_changed):
            # Not an evaluation: hold the split AND the admission verdict the
            # last evaluation produced.  Returning the dataclass default
            # (False) here made a limit live ~1 iteration, because
            # `_r2_decide_idx` writes `r2_admission_limited` from every
            # decision (multiplexing_mixin.py:430-441).
            return SplitDecision(
                current,
                current,
                DecisionReason.HOLD.value,
                admission_limited=self.admission_limited,
            )
        # Read BEFORE the markers move: True means the :976 emergency (or a
        # bucket change) brought us here with the :975 cadence not yet due.
        off_cadence = not self._cadence_due(snapshot)
        self.last_evaluation_s = snapshot.timestamp_s
        self.last_evaluation_iteration = snapshot.decode_iterations

        if self._live_underprediction(snapshot):
            target = max(current, desired.decode_sms)
            target = next(
                (
                    state
                    for state in self.states
                    if state >= target and state > current
                ),
                self.emergency_state,
            )
            reason = DecisionReason.LIVE_UNDERPREDICTION.value
            self.downshift_streak = 0
        else:
            target = desired.decode_sms
            reason = DecisionReason.PROFILE_FLOOR.value

        # :979 "D108 risk 또는 occupancy 90%가 2 epochs 지속되면 admission
        # 제한" is a DISJUNCTION of two independent triggers, and the split
        # below is deliberate in BOTH directions:
        #   * the old flat `AND` made the occupancy half unreachable -- with
        #     `target < emergency_state` no KV/batch pressure could ever arm
        #     the limit, so half of :979 was dead code;
        #   * a flat `OR` over all three terms would be wrong the other way.
        #     An incompatible profile makes `desired.upper_bound_itl_ms = inf`
        #     (profile.py's fail-closed fallback), and `inf > slo` is always
        #     true, so a hybrid arm running without a matching profile would
        #     throttle admission at ANY occupancy.  Today that is masked only
        #     by the accident that the fallback floor (44) is below
        #     `emergency_state`; keeping the D108-risk term conjoined with
        #     `target >= emergency_state` is what makes it safe on purpose.
        overloaded = (
            (
                target >= self.emergency_state
                and desired.upper_bound_itl_ms > snapshot.itl_slo_ms
            )
            or snapshot.kv_occupancy >= 0.90
            or snapshot.running_batch_occupancy >= 0.90
        )
        if not overloaded:
            self.overload_streak = 0
        elif self._overload_epoch_elapsed(snapshot):
            # At most one increment per cadence epoch: see the
            # `last_overload_epoch_*` note in `__init__`.
            self.overload_streak += 1
            self.last_overload_epoch_s = snapshot.timestamp_s
            self.last_overload_epoch_iteration = snapshot.decode_iterations
        admission_limited = self.overload_streak >= 2
        self.admission_limited = admission_limited

        if target < current:
            enough_slack = desired.upper_bound_itl_ms <= 0.75 * snapshot.itl_slo_ms
            self.downshift_streak = self.downshift_streak + 1 if enough_slack else 0
            if self.downshift_streak < self.downshift_epochs:
                target = current
                reason = DecisionReason.DOWNSHIFT_HYSTERESIS.value
            elif not self._dwell_satisfied(snapshot):
                target = current
                reason = DecisionReason.DWELL.value
        else:
            self.downshift_streak = 0

        if target != current and not safe_boundary:
            requested = target
            return SplitDecision(
                current,
                current,
                DecisionReason.UNSAFE.value,
                desired.predicted_itl_ms,
                desired.upper_bound_itl_ms,
                desired.confidence,
                safe=False,
                admission_limited=admission_limited,
                emergency=target >= self.emergency_state,
                requested_decode_sms=requested,
                off_cadence=off_cadence,
            )
        if target != current:
            self.last_transition_s = snapshot.timestamp_s
            self.last_transition_iteration = snapshot.decode_iterations
        return SplitDecision(
            current,
            target,
            DecisionReason.OVERLOAD.value if admission_limited else reason,
            desired.predicted_itl_ms,
            desired.upper_bound_itl_ms,
            desired.confidence,
            safe=True,
            admission_limited=admission_limited,
            emergency=target >= self.emergency_state,
            requested_decode_sms=target,
            off_cadence=off_cadence,
        )


class HybridInformedPolicy:
    def __init__(
        self,
        estimator: ConservativeDecodeFloorEstimator,
        controller: Optional[CoarseGrainedController] = None,
        runtime_environment: Optional[RuntimeEnvironment] = None,
    ):
        self.estimator = estimator
        self.runtime_environment = runtime_environment
        self.controller = controller or CoarseGrainedController(
            estimator.allowed_decode_sms, estimator.emergency_decode_sms
        )

    def decide(
        self,
        snapshot: RuntimeSnapshot,
        current_decode_sms: int,
        safe_boundary: bool,
        bucket_changed: bool = False,
    ) -> SplitDecision:
        """`bucket_changed` is accepted and forwarded, but NOTHING in the
        serving runtime sets it (`multiplexing_mixin.py:436-440` calls
        `decide(snapshot, current, safe_boundary=...)`); see
        `CoarseGrainedController.stabilize`."""
        estimate = self.estimator.estimate(
            snapshot.decode_batch_size,
            snapshot.context_p95,
            snapshot.itl_slo_ms,
            runtime=self.runtime_environment,
        )
        return self.controller.stabilize(
            estimate, snapshot, current_decode_sms, safe_boundary, bucket_changed
        )


class GenericDynamicPolicy:
    """Profile-free control baseline using only live queue/SLO pressure."""

    def __init__(
        self,
        states: Sequence[int] = (16, 24, 34, 44),
        emergency_state: int = 108,
        controller: Optional[CoarseGrainedController] = None,
    ):
        self.states = tuple(sorted(states))
        self.emergency_state = emergency_state
        self.controller = controller or CoarseGrainedController(
            states, emergency_state
        )

    def decide(
        self, snapshot: RuntimeSnapshot, current_decode_sms: int, safe_boundary: bool
    ) -> SplitDecision:
        slack = snapshot.itl_slo_ms - snapshot.measured_itl_p95_ms
        if snapshot.kv_occupancy >= 0.90 or snapshot.running_batch_occupancy >= 0.90:
            target = self.emergency_state
        elif slack < 0:
            target = next(
                (state for state in self.states if state > current_decode_sms),
                self.emergency_state,
            )
        elif slack > 0.35 * snapshot.itl_slo_ms and snapshot.prefill_queue_depth:
            target = max(
                self.states[0],
                max(
                    (state for state in self.states if state < current_decode_sms),
                    default=current_decode_sms,
                ),
            )
        else:
            target = current_decode_sms
        estimate = DecodeFloorEstimate(
            target,
            snapshot.measured_itl_p95_ms,
            snapshot.measured_itl_p95_ms,
            0.0,
            0.5,
            False,
            DecisionReason.GENERIC_PRESSURE.value,
        )
        return self.controller.stabilize(
            estimate, snapshot, current_decode_sms, safe_boundary
        )
