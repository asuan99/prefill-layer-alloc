"""
Mixin class providing multiplexing scheduling logic
"""

from __future__ import annotations

import logging
import atexit
import json
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import torch
import torch.distributed as dist
from torch.cuda.streams import ExternalStream

import os

from sglang.srt.distributed.parallel_state import set_pdmux_status
try:
    from sglang.srt.distributed.parallel_state import pdmux_role_is_thread_local
except ImportError:
    def pdmux_role_is_thread_local() -> bool:
        return False
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.multiplex.pdmux_context import (
    get_current_stream_idx,
    get_sm_counts,
    get_stream_groups,
    initialize_stream_groups,
    load_pdmux_config,
    set_current_stream_idx,
)
from sglang.srt.multiplex.green_readout import maybe_read_green_contexts
from sglang.srt.multiplex.dual_worker import (
    DualWorkerState,
    ExecutionContext,
    TrueDualWorkerRuntime,
    WorkerRole,
)
from sglang.srt.multiplex.controller import (
    FixedPolicy,
    GenericDynamicPolicy,
    HybridInformedPolicy,
    RuntimeSnapshot,
)
from sglang.srt.multiplex.profile import (
    ConservativeDecodeFloorEstimator,
    HybridModelProfileV1,
    RuntimeEnvironment,
)
from sglang.srt.multiplex.telemetry import AsyncJsonlTelemetry

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)


class SchedulerMultiplexMixin:

    def init_pdmux(self: Scheduler):
        # The current split prefill batch
        self.split_prefill_batch: Optional[ScheduleBatch] = None

        # for pd_multiplexing, Init stream_groups, exclude normal stream for prefill only and decode only
        self.pdmux_config = load_pdmux_config(self.server_args.pdmux_config_path)
        initialize_stream_groups(self.gpu_id, self.pdmux_config)
        self.stream_groups = get_stream_groups()
        self.sm_counts = get_sm_counts()
        self.real_sm_group_num = len(self.stream_groups)
        self.dual_worker_enabled = os.environ.get("PDMUX_DUAL_WORKER", "0") in (
            "1", "true", "True"
        )
        self.true_dual_worker_enabled = os.environ.get(
            "PDMUX_TRUE_DUAL_WORKER", "0"
        ) in ("1", "true", "True")
        if self.true_dual_worker_enabled and not pdmux_role_is_thread_local():
            raise RuntimeError(
                "PDMUX_TRUE_DUAL_WORKER requires the thread-local PD-mux role "
                "patch; run scripts/bootstrap/sync_engine_tree.sh"
            )
        trace_path = os.environ.get(
            "PDMUX_TELEMETRY_PATH",
            os.environ.get("PDMUX_DUAL_WORKER_TRACE", ""),
        )
        try:
            trace_every = max(1, int(os.environ.get("PDMUX_DUAL_WORKER_TRACE_EVERY", "32")))
        except ValueError:
            trace_every = 32
        self.dual_worker_trace_path = trace_path
        self.dual_worker_trace_every = trace_every
        self.dual_worker_trace_count = 0
        self.dual_worker_trace_error_logged = False
        # PDMUX_TRACE_FORCE_PREFILL=1 (default OFF): additionally emit a
        # runtime_snapshot on EVERY sync while a prefill batch is in flight,
        # bypassing the count subsampling below.  Default OFF so telemetry from
        # past campaigns stays comparable event-for-event; see
        # `_dual_worker_sync` for why the pure-count grid is blind to fast
        # prefill cells and what the forced samples do (and do not) change.
        self.dual_worker_trace_force_prefill = os.environ.get(
            "PDMUX_TRACE_FORCE_PREFILL", "0"
        ) in ("1", "true", "True")
        self.dual_worker_trace_forced_count = 0
        self.dual_worker_state = DualWorkerState.from_sm_counts(self.sm_counts)
        self.true_dual_worker_runtime = (
            TrueDualWorkerRuntime(self.sm_counts, self._activate_role_context)
            if self.true_dual_worker_enabled
            else None
        )
        self.pdmux_telemetry = AsyncJsonlTelemetry(
            Path(trace_path) if trace_path else None,
            run_id=os.environ.get("PDMUX_RUN_ID", "unlabeled"),
            workload_id=os.environ.get("PDMUX_WORKLOAD_ID", "unlabeled"),
        )
        self.pdmux_experiment_phase = "startup"
        self.r2_policy_name = os.environ.get("PDMUX_R2_POLICY", "").strip().lower()
        self.r2_policy = self._build_r2_policy(self.r2_policy_name)
        self.r2_admission_limited = False
        self._init_sticky_partition()
        self._maybe_emit_green_readout()
        logger.info(
            f"PD-Multiplexing enabled with {self.real_sm_group_num} stream groups, sm_counts (prefill_sm, decode_sm): {self.sm_counts}"
        )
        if self.dual_worker_enabled:
            logger.info(
                "PD-mux R1 observer path enabled (not execution-state separation)"
            )
        if self.true_dual_worker_enabled:
            logger.info(
                "PD-mux true dual-worker enabled: two host issue threads with "
                "role-local streams and thread-local PD-mux role state"
            )
        self.pdmux_telemetry.mark_phase("startup")
        atexit.register(self._close_pdmux_runtime)

    def _build_r2_policy(self: Scheduler, name: str):
        if not name:
            return None
        decode_states = tuple(
            sorted(
                {
                    int(decode_sms)
                    for _prefill_sms, decode_sms in self.sm_counts
                    if int(decode_sms) in (16, 24, 34, 44)
                }
            )
        )
        if not decode_states:
            raise RuntimeError(
                "PDMUX_R2_POLICY requires stream groups with D16/D24/D34/D44 states"
            )
        if name == "fixed":
            return FixedPolicy(
                int(os.environ.get("PDMUX_R2_FIXED_DSM", str(decode_states[-1])))
            )
        if name == "generic":
            return GenericDynamicPolicy(decode_states, emergency_state=108)
        if name == "hybrid":
            profile_path = os.environ.get("PDMUX_MODEL_PROFILE", "")
            if not profile_path:
                raise RuntimeError(
                    "PDMUX_R2_POLICY=hybrid requires PDMUX_MODEL_PROFILE"
                )
            profile = HybridModelProfileV1.load(Path(profile_path))
            estimator = ConservativeDecodeFloorEstimator(
                profile,
                allowed_decode_sms=decode_states,
                emergency_decode_sms=108,
                fallback_decode_sms=int(
                    os.environ.get("PDMUX_R2_FALLBACK_DSM", "44")
                ),
            )
            device_properties = torch.cuda.get_device_properties(self.gpu_id)
            cuda_graph_enabled = not bool(
                getattr(self.server_args, "disable_cuda_graph", False)
            )
            runtime_environment = self._r2_runtime_environment(profile,
                engine_commit=os.environ.get("PDMUX_ENGINE_COMMIT", "unknown"),
                gpu_name=device_properties.name,
                gpu_sm_count=int(device_properties.multi_processor_count),
                cuda_driver=str(torch.version.cuda or "unknown"),
                attention_backend=str(
                    getattr(self.server_args, "attention_backend", "unknown")
                ),
                cuda_graph=cuda_graph_enabled,
                piecewise_cuda_graph=(
                    cuda_graph_enabled
                    and not bool(
                        getattr(
                            self.server_args,
                            "disable_piecewise_cuda_graph",
                            False,
                        )
                    )
                ),
            )
            return HybridInformedPolicy(
                estimator, runtime_environment=runtime_environment
            )
        raise RuntimeError(
            f"unsupported PDMUX_R2_POLICY={name!r}; use fixed, generic or hybrid"
        )

    # ------------------------------------------------------------------
    # PDMUX_GREEN_READOUT (default OFF)
    # ------------------------------------------------------------------
    # A1 rev2 (DESIGN_A1_REV2_STICKY_2026-08-25.md sec 3.1) registers the
    # `green` axis' decision channel: the driver, asked INSIDE this process,
    # what green context each stream group's streams are attached to and how
    # many SM that context holds.  nsys is deliberately not consulted (that
    # would make the axis circular with Q2ae) and `realizedprobe` is not an
    # answer (separate process, after the server is dead -- the Stage 0 D108
    # failure mode).
    #
    # DEFAULT-OFF GUARANTEE.  Without PDMUX_GREEN_READOUT the helper returns
    # None before touching ctypes and this method emits nothing, so a run with
    # the flag unset is unchanged.  Nothing here feeds scheduling, partition
    # selection, batching or admission: it is a one-shot startup observation.
    def _maybe_emit_green_readout(self: Scheduler) -> None:
        try:
            readout = maybe_read_green_contexts(
                self.stream_groups,
                self.sm_counts,
                sticky_idx=getattr(self, "_sticky_fixed_idx", None),
            )
        except Exception as exc:  # noqa: BLE001 - an observation must never
            logger.warning("green read-out failed: %s", exc)  # kill a boot
            return
        if readout is None:
            return
        self._green_readout = readout
        logger.info("PD-mux green read-out: %s", readout)
        path = os.environ.get("PDMUX_GREEN_READOUT_PATH", "")
        if not path:
            return
        try:
            with open(path, "w") as handle:
                json.dump(readout, handle, indent=2, sort_keys=True)
        except (OSError, TypeError, ValueError) as exc:
            logger.warning("green read-out not written to %s: %s", path, exc)

    # ------------------------------------------------------------------
    # PDMUX_STICKY_PARTITION (default OFF)
    # ------------------------------------------------------------------
    # WHAT IT CHANGES.  `initialize_stream_groups` always appends a plain
    # (non-green-context) group labelled (0, total_sm) after the real
    # divisions, and `adjust_stream_groups` falls back to it whenever decode
    # is busy but no prefill batch is in flight.  Because a prefill batch is
    # in flight only for a small fraction of the run, the cell's decode
    # division was realized over only 4-19% of decode-active time in job
    # 872077 (T8 d16 0.038 -> d54 0.093, Ha8 0.104 -> 0.187); the remaining
    # 81-96% ran unpartitioned at 108 SM.  A cell label was therefore a
    # TARGET, not an allocation -- the decode-axis analogue of the Stage 0
    # D108 error -- and the dilution factor differed per cell, so it was a
    # confound INSIDE any estimand built from those cells.
    #
    # With the flag ON, "decode is busy" alone keeps the target division, so
    # decode runs at D SM continuously and prefill SM may sit idle.  That is
    # the budget-constrained quantity C2 measured directly (prefill pinned,
    # decode continuously at D).
    #
    # WHAT IT DOES NOT CHANGE.  The flag decides only WHICH stream group is
    # current; it adds no switch point, removes no drain, and does not touch
    # batching, admission or telemetry emission.  CUDA graphs are captured
    # per stream-group index (cuda_graph_runner.capture, key
    # f"{stream_idx}_{bs}"), so holding a division index replays a captured
    # graph exactly as the fallback index did -- no eager fallback.
    #
    # DEFAULT-OFF GUARANTEE.  Every predicate added below short-circuits on
    # `self.sticky_partition_enabled` being False, leaving the pre-patch
    # branch and the pre-patch value in every case; see
    # tests/test_sticky_partition.py, which asserts OFF equivalence against a
    # re-implementation of the pre-patch selector.
    def _init_sticky_partition(self: Scheduler) -> None:
        self.sticky_partition_enabled = os.environ.get(
            "PDMUX_STICKY_PARTITION", "0"
        ) in ("1", "true", "True")
        # Resolved target index for a FixedPolicy run (None => fall back to
        # the engine's own decode-batch-size selector, unchanged).
        self._sticky_fixed_idx: Optional[int] = None
        if not self.sticky_partition_enabled:
            return

        # Undefined combinations are REJECTED rather than silently given some
        # behaviour, because each of them owns the partition index through a
        # path this flag does not touch.
        if os.environ.get("PDMUX_LA_COORD"):
            raise RuntimeError(
                "PDMUX_STICKY_PARTITION is not defined for the coordinated "
                "per-layer-type event loop (PDMUX_LA_COORD); that loop selects "
                "stream indices itself and is a documented negative result"
            )
        if os.environ.get("PDMUX_SLO_SCHED"):
            raise RuntimeError(
                "PDMUX_STICKY_PARTITION with PDMUX_SLO_SCHED is undefined: the "
                "SLO controller only decides on prefill spans, and extending "
                "its decisions into decode-only spans changes its dynamics"
            )
        if os.environ.get("PDMUX_FIXED_DECODE_SM_FILE"):
            raise RuntimeError(
                "PDMUX_STICKY_PARTITION with PDMUX_FIXED_DECODE_SM_FILE is "
                "undefined: the model-side per-layer decode pin picks its own "
                "streams inside the forward pass"
            )
        if self.r2_policy_name not in ("", "fixed"):
            raise RuntimeError(
                "PDMUX_STICKY_PARTITION is only defined for "
                f"PDMUX_R2_POLICY in (unset, fixed); got {self.r2_policy_name!r}"
            )
        if self.real_sm_group_num < 3:
            raise RuntimeError(
                "PDMUX_STICKY_PARTITION requires at least one green-context "
                f"division; got {self.real_sm_group_num} stream groups"
            )

        if isinstance(self.r2_policy, FixedPolicy):
            target_d = int(self.r2_policy.decode_sms)
            matches = [
                index
                for index, (_prefill_sms, decode_sms) in enumerate(self.sm_counts)
                if int(decode_sms) == target_d
            ]
            # Index 0 is the plain prefill-only group and the last index is the
            # plain unpartitioned group; neither is a green-context division,
            # so neither is a legal sticky target.
            divisions = [
                index for index in matches if 1 <= index <= self.real_sm_group_num - 2
            ]
            if not divisions:
                raise RuntimeError(
                    f"PDMUX_STICKY_PARTITION: no green-context division with "
                    f"decode_sms={target_d} in sm_counts={self.sm_counts}"
                )
            self._sticky_fixed_idx = divisions[0]

        logger.info(
            "PD-mux sticky partition ENABLED: decode holds its division while "
            "decode is busy (fixed target index=%s); the unpartitioned "
            "(0, total_sm) group is used only when the decode batch is empty",
            self._sticky_fixed_idx,
        )

    def _r2_runtime_snapshot(self: Scheduler) -> RuntimeSnapshot:
        batch = getattr(self, "running_batch", None)
        batch_size = batch.batch_size() if batch is not None else 0
        context_lengths = []
        remaining_output_lengths = []
        for req in list(getattr(batch, "reqs", ()) or ()):
            try:
                context_lengths.append(
                    len(getattr(req, "origin_input_ids", ()))
                    + len(getattr(req, "output_ids", ()))
                )
                sampling = getattr(req, "sampling_params", None)
                requested = int(getattr(sampling, "max_new_tokens", 0) or 0)
                remaining_output_lengths.append(
                    max(0, requested - len(getattr(req, "output_ids", ())))
                )
            except TypeError:
                continue
        context_lengths.sort()
        remaining_output_lengths.sort()

        def _percentile(values, fraction, default=1):
            if not values:
                return default
            return values[min(len(values) - 1, int((len(values) - 1) * fraction))]

        capacity = max(1, int(getattr(self, "max_running_requests", 1)))
        kv_occupancy, _full_occupancy, _mamba_occupancy = (
            self._r2_cache_occupancy()
        )
        return RuntimeSnapshot(
            timestamp_s=time.perf_counter(),
            decode_iterations=int(getattr(self, "_r2_decode_iterations", 0)),
            active_decode_sequences=batch_size,
            decode_batch_size=max(1, batch_size),
            context_p50=_percentile(context_lengths, 0.50),
            context_p95=_percentile(context_lengths, 0.95),
            context_max=max(context_lengths, default=1),
            expected_remaining_output_p90=_percentile(
                remaining_output_lengths, 0.90, default=0
            ),
            prefill_queue_depth=len(self.waiting_queue),
            oldest_prefill_age_ms=self._slo_prefill_age_ms(),
            measured_itl_ewma_ms=float(getattr(self, "_slo_tpot_ema", 0.0)),
            measured_itl_p95_ms=float(
                _percentile(
                    sorted(getattr(self, "_r2_itl_samples", ())),
                    0.95,
                    default=getattr(self, "_slo_tpot_ema", 0.0),
                )
            ),
            itl_slo_ms=float(os.environ.get("PDMUX_TPOT_SLO_MS", "60")),
            ttft_risk=min(
                1.0,
                self._slo_prefill_age_ms()
                / max(1.0, float(os.environ.get("PDMUX_TTFT_SLO_MS", "3000"))),
            ),
            kv_occupancy=kv_occupancy,
            running_batch_occupancy=min(1.0, batch_size / capacity),
        )

    def _r2_cache_occupancy(self: Scheduler):
        def _occupancy(pool) -> float:
            if pool is None:
                return 0.0
            try:
                return max(
                    0.0,
                    min(
                        1.0,
                        1.0 - float(pool.available_size()) / max(1, int(pool.size)),
                    ),
                )
            except (AttributeError, TypeError, ZeroDivisionError):
                return 0.0

        allocator = getattr(self, "token_to_kv_pool_allocator", None)
        full = _occupancy(allocator)
        req_pool = getattr(self, "req_to_token_pool", None)
        mamba = _occupancy(getattr(req_pool, "mamba_pool", None))
        return max(full, mamba), full, mamba

    def _r2_decide_idx(self: Scheduler, current_idx: int) -> int:
        snapshot = self._r2_runtime_snapshot()
        safe = (
            self.true_dual_worker_runtime is None
            or self.true_dual_worker_runtime.arbiter.safe_to_switch()
        )
        decision = self.r2_policy.decide(
            snapshot,
            int(self.sm_counts[current_idx][1]),
            safe_boundary=safe,
        )
        self.r2_admission_limited = decision.admission_limited
        self.pdmux_telemetry.emit(
            "controller_decision",
            self.pdmux_experiment_phase,
            policy=self.r2_policy_name,
            current_decode_sms=decision.current_decode_sms,
            target_decode_sms=decision.target_decode_sms,
            reason=decision.reason,
            predicted_itl_ms=decision.predicted_itl_ms,
            upper_bound_itl_ms=decision.upper_bound_itl_ms,
            confidence=decision.confidence, safe=decision.safe,
            off_cadence=decision.off_cadence,  # :976 fired before the cadence
            admission_limited=decision.admission_limited,
        )
        if decision.target_decode_sms != decision.current_decode_sms:
            self.pdmux_telemetry.emit(
                "split_transition",
                self.pdmux_experiment_phase,
                current_decode_sms=decision.current_decode_sms,
                target_decode_sms=decision.target_decode_sms,
                reason=decision.reason,
                safe=decision.safe,
            )
        elif not decision.safe:
            self.pdmux_telemetry.emit(
                "blocked_unsafe_transition",
                self.pdmux_experiment_phase,
                current_decode_sms=decision.current_decode_sms,
                requested_decode_sms=(
                    decision.requested_decode_sms or decision.target_decode_sms
                ),
                reason=decision.reason,
            )
        matches = [
            index
            for index, (_prefill_sms, decode_sms) in enumerate(self.sm_counts)
            if int(decode_sms) == decision.target_decode_sms
        ]
        if not matches:
            logger.warning(
                "R2 policy requested unavailable D%d; holding D%d",
                decision.target_decode_sms,
                self.sm_counts[current_idx][1],
            )
            return current_idx
        return matches[0]

    def _close_pdmux_runtime(self: Scheduler) -> None:
        """Best-effort shutdown for writer and role threads."""
        telemetry = getattr(self, "pdmux_telemetry", None)
        if telemetry is not None:
            telemetry.close()
        runtime = getattr(self, "true_dual_worker_runtime", None)
        if runtime is not None:
            try:
                runtime.close()
            except TimeoutError as exc:
                logger.error("PD-mux worker shutdown timed out: %s", exc)

    @contextmanager
    def _activate_role_context(self: Scheduler, context: ExecutionContext):
        """Activate CUDA stream and thread-local PD-mux role for one host task."""
        if context.stream is None:
            raise RuntimeError("true dual-worker task has no CUDA stream")
        with torch.cuda.device(self.gpu_id), torch.cuda.stream(context.stream):
            set_pdmux_status(context.role is WorkerRole.PREFILL)
            yield

    def _dual_worker_sync(self, stream_idx: Optional[int] = None) -> None:
        """Refresh role-owned views without changing legacy scheduler state."""
        if not (
            getattr(self, "dual_worker_enabled", False)
            or getattr(self, "dual_worker_trace_path", "")
        ):
            return
        self.dual_worker_state.observe_scheduler(
            self, get_current_stream_idx() if stream_idx is None else stream_idx
        )
        runtime = getattr(self, "true_dual_worker_runtime", None)
        if runtime is not None:
            runtime.register_waiting(list(self.waiting_queue))
            live = list(self.waiting_queue)
            for batch in (
                getattr(self, "split_prefill_batch", None),
                getattr(self, "running_batch", None),
            ):
                live.extend(getattr(batch, "reqs", ()) or ())
            runtime.prune_to_live(live)
        if (
            self.pdmux_experiment_phase == "startup"
            and (
                len(self.waiting_queue)
                or getattr(self, "split_prefill_batch", None) is not None
                or (
                    getattr(self, "running_batch", None) is not None
                    and not self.running_batch.is_empty()
                )
            )
        ):
            self.pdmux_experiment_phase = "benchmark"
            self.pdmux_telemetry.mark_phase("benchmark")
        self.dual_worker_trace_count += 1
        # The scheduled grid is a pure COUNT subsample of sync calls.  The event
        # loop keeps spinning when idle, so a cell whose prefill is fast (large
        # prefill SM share, e.g. [92,16]) occupies very few sync calls and can
        # miss the grid entirely: measured on E1/T8/d44, prefill was active
        # 4.9% of wall time but only 0.07% of *sampled* snapshots.  That bias is
        # systematic in the direction of the experiment (the more SM prefill
        # gets, the less observable it is), so partition-pin verification is
        # impossible at exactly one end of the frontier.  With
        # PDMUX_TRACE_FORCE_PREFILL=1 every prefill-in-flight sync also emits.
        # OBSERVATION ONLY: nothing below this line feeds scheduling, partition
        # selection or batch composition, and `dual_worker_trace_count` still
        # advances once per sync either way, so the scheduled grid (and hence
        # the pre-patch record set) is bit-for-bit unchanged -- forced records
        # are strictly ADDITIONAL and are tagged `trace_forced=true`.
        #
        # CAVEAT for consumers: in force mode the emitted population is
        # deliberately OVER-sampled on prefill-in-flight syncs, so any
        # "fraction of snapshots" statistic (the `*_frac_count_based` audit
        # values in results/s8_frontier/e1_pin_check.py) is biased toward
        # prefill and is NOT comparable across the flag.  TIME-WEIGHTED
        # statistics (weight each snapshot by the gap to the next one) and
        # per-episode gates are unaffected -- denser sampling only makes the
        # Riemann sum more accurate.  Recompute a legacy-comparable count
        # statistic by filtering to `trace_forced != true`.
        if self.dual_worker_trace_path:
            scheduled = (
                self.dual_worker_trace_count == 1
                or self.dual_worker_trace_count % self.dual_worker_trace_every == 0
            )
            forced = (
                not scheduled
                and getattr(self, "dual_worker_trace_force_prefill", False)
                and getattr(self, "split_prefill_batch", None) is not None
            )
            if scheduled or forced:
                if forced:
                    self.dual_worker_trace_forced_count += 1
                self._write_dual_worker_trace(forced=forced)

    def _write_dual_worker_trace(self: Scheduler, forced: bool = False) -> None:
        """Enqueue symmetric sampled state without scheduler-thread file I/O."""
        try:
            payload = {
                "sample_index": self.dual_worker_trace_count,
                "architecture": (
                    "true_dual"
                    if getattr(self, "true_dual_worker_enabled", False)
                    else ("r1_observer" if self.dual_worker_enabled else "legacy")
                ),
                **self.dual_worker_state.metrics(),
                **asdict(self._r2_runtime_snapshot()),
            }
            runtime = getattr(self, "true_dual_worker_runtime", None)
            if runtime is not None:
                payload.update(runtime.metrics())
            kv_total, kv_full, kv_mamba = self._r2_cache_occupancy()
            payload.update(
                {
                    "kv_total_occupancy": kv_total,
                    "kv_full_occupancy": kv_full,
                    "kv_mamba_occupancy": kv_mamba,
                }
            )
            if getattr(self, "dual_worker_trace_force_prefill", False):
                # Emitted ONLY in force mode, so a run with the flag off writes
                # byte-identical records to a pre-patch run.  In force mode,
                # `trace_forced=false` is exactly the legacy scheduled-grid
                # population (sample_index == 1 or % trace_every == 0) and
                # `trace_forced=true` is the additional prefill-in-flight one.
                payload["trace_forced"] = forced
            self.pdmux_telemetry.emit(
                "runtime_snapshot",
                self.pdmux_experiment_phase,
                **payload,
            )
        except (OSError, TypeError, ValueError, RuntimeError) as exc:
            if not self.dual_worker_trace_error_logged:
                logger.warning("dual-worker telemetry disabled after write failure: %s", exc)
                self.dual_worker_trace_error_logged = True

    def _dual_worker_start_prefill(self, batch: ScheduleBatch) -> None:
        if getattr(self, "dual_worker_enabled", False):
            self.dual_worker_state.prefill.active_batch = batch
            self.dual_worker_state.coordinator.start_prefill(batch.reqs)
        runtime = getattr(self, "true_dual_worker_runtime", None)
        if runtime is not None:
            runtime.start_prefill(batch.reqs)

    def _dual_worker_prefill_ready(self, batch: ScheduleBatch) -> None:
        if getattr(self, "dual_worker_enabled", False):
            state = self.dual_worker_state
            state.coordinator.complete_prefill(batch.reqs)
            state.decode.ready_queue.extend(batch.reqs)
        runtime = getattr(self, "true_dual_worker_runtime", None)
        if runtime is not None:
            runtime.complete_prefill(batch.reqs)

    def _dual_worker_start_decode(self, batch: ScheduleBatch) -> None:
        if getattr(self, "dual_worker_enabled", False):
            state = self.dual_worker_state
            state.coordinator.start_decode(batch.reqs)
            rids = {getattr(req, "rid", id(req)) for req in batch.reqs}
            state.decode.ready_queue = [
                req for req in state.decode.ready_queue
                if getattr(req, "rid", id(req)) not in rids
            ]
        runtime = getattr(self, "true_dual_worker_runtime", None)
        if runtime is not None:
            runtime.start_decode(batch.reqs)

    def _dual_worker_complete_decode(self, batch: ScheduleBatch) -> None:
        completed = []
        for req in getattr(batch, "reqs", ()):
            finished = getattr(req, "finished", None)
            if callable(finished) and finished():
                completed.append(req)
        if getattr(self, "dual_worker_enabled", False):
            self.dual_worker_state.coordinator.complete(completed)
            self.dual_worker_state.decode.finish_step()
        runtime = getattr(self, "true_dual_worker_runtime", None)
        if runtime is not None:
            runtime.complete_requests(completed)

    def _slo_context_len_signal(self: Scheduler) -> int:
        """Step D (SKETCH): characteristic context length (tokens) of imminent prefill work =
        feedforward signal for _slo_decide_idx. max over the current split-prefill batch and the
        head of the waiting queue. Defensive (attrs may vary by sglang version); returns >=1."""
        _l = 0
        _b = getattr(self, "split_prefill_batch", None)
        try:
            _c = getattr(_b, "seq_lens_cpu_cache", None) if _b is not None else None
            if _c is not None and len(_c):
                _l = max(_l, int(max(_c)))
        except Exception:
            pass
        try:
            for _r in list(self.waiting_queue)[:32]:
                _l = max(_l, len(_r.origin_input_ids))
        except Exception:
            pass
        return _l or 1

    def _slo_prefill_age_ms(self: Scheduler) -> float:
        """Step E: TTFT-risk signal = age (ms) of the OLDEST prefill request that has NOT yet
        produced its first token = max over the waiting queue AND the IN-PROGRESS split-prefill
        batch (a long request that is admitted & chunking has left the waiting queue but is still
        accumulating TTFT -> must be counted, else pf_age reads 0). now - wait_queue_entry_time
        (perf_counter, same clock as TPOT signal). 0 if none."""
        import time as _t
        _now = _t.perf_counter()
        _oldest = _now
        try:
            _cands = list(self.waiting_queue)[:64]
            _spb = getattr(self, "split_prefill_batch", None)
            if _spb is not None and getattr(_spb, "reqs", None):
                _cands = _cands + list(_spb.reqs)[:64]
            for _r in _cands:
                _e = getattr(getattr(_r, "time_stats", None), "wait_queue_entry_time", 0.0) or 0.0
                if _e and _e < _oldest:
                    _oldest = _e
        except Exception:
            return 0.0
        return max(0.0, (_now - _oldest) * 1000.0)

    def _slo_knee_itl(self: Scheduler, d_sm: int) -> float:
        """Step G: profiled decode-forward time (ms) at a given decode-SM, used for its SHAPE only.
        Source: results/r0c/knee_result_835571.txt (Zamba2-2.7B) = 9*per-attn + 54*per-mamba.
        ABSOLUTES DO NOT TRANSFER (profiled at ctx3600/no-cudagraph: d24=127ms vs ~29ms observed on
        ShareGPT/cudagraph), so callers must rescale by the live TPOT at the current split."""
        _pts = ((8, 358.4), (16, 186.3), (24, 127.4), (44, 70.7), (108, 47.7))
        if d_sm <= _pts[0][0]:
            return _pts[0][1]
        if d_sm >= _pts[-1][0]:
            return _pts[-1][1]
        for (x0, y0), (x1, y1) in zip(_pts, _pts[1:]):
            if x0 <= d_sm <= x1:
                return y0 + (y1 - y0) * (d_sm - x0) / float(x1 - x0)
        return _pts[-1][1]

    def _slo_feasible(self: Scheduler, idx_cur: int, idx_new: int) -> bool:
        """Step G feasibility gate (design: reports/realtrace_findings_and_open_branches.md §3.4).

        Gates ONLY prefill-ward moves (D_sm down). Asymmetry: starving decode PROPAGATES
        (residency up -> running batch saturates -> prefill admission blocked -> TTFT explodes =
        the positive-feedback trap), while starving prefill does not propagate back into decode.

        Two guards (both must pass):
          (1) CONGESTION [primary, directly observable]: the decode batch must have headroom.
              Little's law: required_concurrency = arrival_rate * output_len * ITL; when that
              reaches max_running_requests the batch caps admission and TTFT collapses. Measured
              on ShareGPT r8: d44 N~42 (TTFT 0.12s) / d24 N~50 (1.21s) / d16 N~57 (7.24s).
              ** This is the guard that actually catches the trap: d16's ITL (33.5ms p50) PASSES
              the 60ms SLO, so an ITL-only gate would let the fatal move through. **
          (2) ITL [secondary]: predicted ITL at the candidate split stays under the TPOT SLO,
              using the knee SHAPE rescaled by the live TPOT-EMA at the current split.
        """
        if idx_new >= idx_cur:
            return True  # decode-ward or no move: cannot trigger this failure mode
        _occ = float(os.environ.get("PDMUX_SLO_FEAS_OCC", "0.85"))
        _mar = float(os.environ.get("PDMUX_SLO_FEAS_MARGIN", "0.9"))
        _slo = float(os.environ.get("PDMUX_TPOT_SLO_MS", "60"))
        # (1) congestion guard
        try:
            _cap = int(getattr(self, "max_running_requests", 0) or 0)
            _bs = self.running_batch.batch_size() if self.running_batch is not None else 0
            if _cap > 0 and _bs >= _cap * _occ:
                self._slo_feas_refused = getattr(self, "_slo_feas_refused", 0) + 1
                return False
        except Exception:
            pass
        # (2) ITL guard (knee shape anchored to the live measurement)
        _tpot = getattr(self, "_slo_tpot_ema", 0.0)
        try:
            if _tpot > 0:
                _k_cur = self._slo_knee_itl(self.sm_counts[idx_cur][1])
                if _k_cur > 0:
                    _pred = _tpot * (self._slo_knee_itl(self.sm_counts[idx_new][1]) / _k_cur)
                    if _pred > _slo * _mar:
                        self._slo_feas_refused = getattr(self, "_slo_feas_refused", 0) + 1
                        return False
        except Exception:
            pass
        return True

    def _slo_decide_idx_binding(self: Scheduler, _tpot_slo: float, _lo: int, _hi: int) -> int:
        """Step E (design §E): binding-signal-first dual-slack controller. Treat prefill-slack
        (TTFT_SLO - oldest-waiting age) and decode-slack (TPOT_SLO - TPOT-EMA) as PEERS; push the
        split toward whichever is more-binding & urgent; when both have slack, drift to a tuned-
        static ANCHOR (resting point that already wins in stationary). Fixes the v7b flaw where the
        prefill path was gated behind `tpot<lo` (missed TTFT-bound cudagraph regimes)."""
        _ttft_slo = float(os.environ.get("PDMUX_TTFT_SLO_MS", "3000"))
        _anchor = max(_lo, min(_hi, int(os.environ.get("PDMUX_SLO_ANCHOR_IDX", str((_lo + _hi) // 2)))))
        _hi_frac = float(os.environ.get("PDMUX_TPOT_HI", "0.85"))
        _pf_urg = float(os.environ.get("PDMUX_SLO_PF_URGENCY", "0.5"))  # prefill urgent if slack-frac below this
        _idx = getattr(self, "_slo_idx", _anchor)
        _tpot = getattr(self, "_slo_tpot_ema", 0.0)
        _dwell = getattr(self, "_slo_dwell", 0)
        _pf_age = self._slo_prefill_age_ms()
        _dec_slack = (_tpot_slo - _tpot) / _tpot_slo           # decode headroom (fraction of SLO)
        _pf_slack = (_ttft_slo - _pf_age) / _ttft_slo          # prefill headroom (fraction of SLO)
        # Step F (predictive-hybrid saturation-hold): saturation = no split satisfies both SLOs ->
        # chasing the more-binding side thrashes (flip-flop -> green-ctx drain), so converge to the
        # tuned-static anchor. PREDICT it (backlog grows over a window even at MAX prefill = we've
        # exhausted the prefill lever and still lose ground) instead of only reacting to both-slacks-
        # negative (lagging). both_neg kept as a safety-net fallback.
        _sat_win = max(2, int(os.environ.get("PDMUX_SLO_SAT_WIN", "4")))
        _sat_margin = float(os.environ.get("PDMUX_SLO_SAT_MARGIN", "0"))
        _sat_deep = float(os.environ.get("PDMUX_SLO_SAT_DEEP", "0.5"))
        _hist = (getattr(self, "_slo_pf_hist", []) + [_pf_age])[-_sat_win:]
        self._slo_pf_hist = _hist
        _sat_pred = (_idx <= _lo) and (len(_hist) >= _sat_win) and (_hist[-1] > _hist[0]) and (_pf_slack < _pf_urg)
        # saturation if EITHER constraint is DEEPLY (unrecoverably) violated -- chasing it is futile, and
        # the split itself drives the other slack, so "both<0" flickers (idx=lo starves decode -> dec_slack<0
        # -> hold -> idx=anchor frees decode -> dec_slack>0 -> chase prefill -> back). A deep one-sided
        # violation is a STABLE saturation signal -> converge to anchor & stay. OR both near boundary, OR pred.
        _sat_fb = (_pf_slack < -_sat_deep) or (_dec_slack < -_sat_deep) or ((_pf_slack < _sat_margin) and (_dec_slack < _sat_margin))
        _sat = _sat_pred or _sat_fb
        # slacks are split-coupled and settle at the SLO boundary under saturation -> the INSTANTANEOUS
        # _sat flickers no matter the threshold (proven: margin 0/0.15, deep 0.5 all thrash near a
        # boundary-hovering signal). Latch the HOLD STATE (hysteresis on the regime, not the signal):
        # once saturated, stay in hold for LATCH steps -> converge to anchor and STAY (degenerate to
        # static under sustained saturation, so no green-ctx thrash).
        _latch = getattr(self, "_slo_sat_latch", 0)
        if _sat:
            _latch = int(os.environ.get("PDMUX_SLO_SAT_LATCH", "20"))
        self._slo_sat_latch = max(0, _latch - 1)
        _hold = _sat or (_latch > 0)
        if _dwell > 0:
            self._slo_dwell = _dwell - 1
            _new = _idx
        elif _hold:
            _new = _idx + (1 if _idx < _anchor else (-1 if _idx > _anchor else 0))  # saturation -> converge to anchor (no chase)
        elif _pf_slack < _dec_slack and _pf_slack < _pf_urg:
            _new = max(_lo, _idx - 1)                          # TTFT more-binding & urgent -> more prefill SM
        elif _dec_slack < _pf_slack and _dec_slack < (1.0 - _hi_frac):
            _new = min(_hi, _idx + 1)                          # TPOT more-binding & urgent -> more decode SM
        elif _idx != _anchor:
            _new = _idx + (1 if _idx < _anchor else -1)        # both slack -> drift to tuned-static anchor
        else:
            _new = _idx                                        # at anchor, no urgency -> hold
        # Step G feasibility gate (env PDMUX_SLO_FEAS_GATE; off => byte-identical to Step E/F).
        # Refuses a prefill-ward move whose candidate split decode cannot afford -> blocks entry
        # into the positive-feedback trap (pf_age GROWS after moving prefill-ward, re-triggering
        # the chase: 2800->4743ms observed). See _slo_feasible.
        _feas = 1
        if os.environ.get("PDMUX_SLO_FEAS_GATE") and _new != _idx:
            if not self._slo_feasible(_idx, _new):
                _feas = 0
                _new = _idx  # refuse the move, hold current split
        if _new != _idx:
            self._slo_dwell = int(os.environ.get("PDMUX_SLO_DWELL", "3"))
            logger.info(
                "SLO-BIND %d->%d dec_sm=%d pf_age=%.0fms tpot=%.1fms pfslack=%.2f decslack=%.2f sat=%d anchor=%d",
                _idx, _new, self.sm_counts[_new][1], _pf_age, _tpot, _pf_slack, _dec_slack, int(_sat), _anchor,
            )
        elif _feas == 0:
            logger.info(
                "SLO-FEAS refused idx=%d->%d dec_sm=%d->%d pf_age=%.0fms tpot=%.1fms bs=%s refused_total=%d",
                _idx, _new, self.sm_counts[_idx][1], self.sm_counts[max(_lo, _idx - 1)][1], _pf_age, _tpot,
                (self.running_batch.batch_size() if self.running_batch is not None else -1),
                getattr(self, "_slo_feas_refused", 0),
            )
        self._slo_idx = _new
        return _new

    def _slo_decide_idx(self: Scheduler) -> int:
        """SLO-aware controller (v7): return the target split idx from the latency signal
        (measured TPOT-EMA + prefill backlog). Called PER prefill-layer-span from the event
        loop; the caller drains+switches only when the returned idx differs from the current
        one (agnostic latency-adaptive rarely switches once converged -> few green-ctx drains).
        Config divisions must be ordered by INCREASING decode SM (idx+1 = more decode SM)."""
        _pin = os.environ.get("PDMUX_SLO_PIN")
        if _pin:
            return int(_pin)
        _tpot_slo = float(os.environ.get("PDMUX_TPOT_SLO_MS", "60"))
        _qtarget = float(os.environ.get("PDMUX_QDEPTH_TARGET", "4"))
        _hi_frac = float(os.environ.get("PDMUX_TPOT_HI", "0.85"))
        _lo_frac = float(os.environ.get("PDMUX_TPOT_LO", "0.65"))
        _lo, _hi = 1, self.real_sm_group_num - 2
        # Step E (env-gated PDMUX_SLO_MODE=binding): binding-signal-first dual-slack controller.
        # Off -> the v7b path below (byte-identical).
        if os.environ.get("PDMUX_SLO_MODE") == "binding":
            return self._slo_decide_idx_binding(_tpot_slo, _lo, _hi)
        # Step D (SKETCH, env-gated PDMUX_SLO_LFF): context-length FEEDFORWARD. attn-prefill is
        # O(L^2) so a long-context request needs prefill SM preemptively (L known at admission,
        # unlike the feedback TPOT/queue signals). Shift the resting split center by L; feedback
        # still handles urgency. OFF (unset) => ff_center = neutral & deadband = hold = byte-identical to v7b.
        _ff_center = (_lo + _hi) // 2                    # NEUTRAL (1 step from either optimum)
        _ff_on = bool(os.environ.get("PDMUX_SLO_LFF"))
        if _ff_on:
            _g = float(os.environ.get("PDMUX_SLO_LFF", "1"))
            _lref = float(os.environ.get("PDMUX_L_REF_TOK", "3600"))
            _lsig = self._slo_context_len_signal()
            # long L -> negative offset -> lower idx -> more prefill SM; short L -> higher idx (more decode SM)
            _ff_center = int(round((_lo + _hi) / 2.0 - _g * (_lsig / _lref - 1.0)))
            _ff_center = max(_lo, min(_hi, _ff_center))
        _idx = getattr(self, "_slo_idx", _ff_center)
        _tpot = getattr(self, "_slo_tpot_ema", 0.0)
        _qd = len(self.waiting_queue)
        _dwell = getattr(self, "_slo_dwell", 0)
        if _dwell > 0:
            self._slo_dwell = _dwell - 1                 # min dwell after a switch (anti-oscillation)
            _new = _idx
        elif _tpot > _hi_frac * _tpot_slo:
            _new = min(_hi, _idx + 1)                    # TPOT tight -> more decode SM
        elif _tpot < _lo_frac * _tpot_slo and _qd > _qtarget:  # TPOT has real margin & prefill backlogged
            _new = max(_lo, _idx - 1)                    # -> more prefill SM (stop before decode gets tight)
        elif _ff_on and _idx != _ff_center:
            _new = _idx + (1 if _idx < _ff_center else -1)   # deadband -> drift to feedforward resting point
        else:
            _new = _idx                                  # deadband -> hold
        if _new != _idx:
            self._slo_dwell = int(os.environ.get("PDMUX_SLO_DWELL", "3"))
            logger.info(
                "SLO-SCHED %d->%d decode_sm=%d tpot=%.1fms qd=%d",
                _idx, _new, self.sm_counts[_new][1], _tpot, _qd,
            )
        self._slo_idx = _new
        return _new

    # TODO(jason-fxz): This is a temporary demo
    def adjust_stream_groups(
        self: Scheduler,
    ) -> tuple[int, tuple[ExternalStream, ExternalStream]]:
        if (
            os.environ.get("PDMUX_SLO_SCHED")
            and not self.running_batch.is_empty()
            and self.split_prefill_batch
        ):
            _idx = self._slo_decide_idx()
            set_current_stream_idx(_idx)
            self.tp_worker.model_runner.update_decode_attn_backend(_idx)
            return _idx, self.stream_groups[_idx]
        # PDMUX_STICKY_PARTITION (default OFF, see `_init_sticky_partition`):
        # with the flag ON, "decode busy" alone takes this branch, so decode
        # keeps its green-context division while prefill is idle instead of
        # falling back to the plain unpartitioned group below.  With the flag
        # OFF `self.sticky_partition_enabled` is False and the disjunction
        # collapses to the pre-patch `and self.split_prefill_batch`.
        if not self.running_batch.is_empty() and (
            self.split_prefill_batch or self.sticky_partition_enabled
        ):
            if self.sticky_partition_enabled and self._sticky_fixed_idx is not None:
                # Under PDMUX_R2_POLICY=fixed the target is a constant, and it
                # is the constant the v7 controller block installs on prefill
                # spans (`_r2_decide_idx`).  Using it here makes the whole run
                # sit on ONE division: without it, a config carrying a
                # guard-satisfier row (e.g. pdmux_e1_d54.yml) would drift to the
                # guard row whenever decode_bs crossed its threshold.
                stream_idx = self._sticky_fixed_idx
            else:
                decode_bs = self.running_batch.batch_size()
                manual_divisions = self.pdmux_config.manual_divisions
                if manual_divisions:
                    for i in range(len(manual_divisions)):
                        _, _, threshold = manual_divisions[i]
                        if decode_bs >= threshold:
                            stream_idx = i + 1
                else:
                    stream_idx = max(
                        1,
                        min(
                            self.real_sm_group_num - 2,
                            decode_bs
                            * (self.real_sm_group_num - 2)
                            // self.pdmux_config.decode_bs_divisor,
                        ),
                    )
            set_current_stream_idx(stream_idx)
        elif not self.running_batch.is_empty():
            set_current_stream_idx(self.real_sm_group_num - 1)
        else:
            # Decode batch EMPTY.  Sticky deliberately does NOT hold the
            # division here: there is no decode work to protect, holding it
            # would strand D SM that prefill could use, and the decode-active
            # time weighting of E1_DECODE_REALIZED gives this interval zero
            # weight either way -- so holding it could only alter the prefill
            # axis without being able to improve the gate it exists to serve.
            # This is also the state the `stream_idx > 0 and
            # running_batch.is_empty()` trigger in `event_loop_pdmux` exists to
            # reach, so leaving both alone keeps the two consistent.
            set_current_stream_idx(0)

        stream_idx = get_current_stream_idx()

        self.tp_worker.model_runner.update_decode_attn_backend(stream_idx)
        return stream_idx, self.stream_groups[stream_idx]

    def update_split_prefill_batch(self: Scheduler, sm_count: int) -> bool:
        if self.split_prefill_batch:
            return False
        if getattr(self, "r2_admission_limited", False) and self._r2_admission_holds():
            return False

        # add new request
        if getattr(self, "dual_worker_enabled", False):
            self.dual_worker_state.prefill.begin_admission()
        batch = self.get_new_batch_prefill()
        if getattr(self, "dual_worker_enabled", False):
            self.dual_worker_state.prefill.finish_admission()
        if batch and not batch.is_empty():
            batch.forward_mode = (
                ForwardMode.SPLIT_PREFILL
            )  # Set forward mode for split prefill
            self.split_prefill_batch = batch
            self._dual_worker_start_prefill(batch)
            return True
        return False

    @torch.inference_mode()
    def event_loop_pdmux(self: Scheduler):
        """A scheduler loop for pd multiplexing."""
        decode_done = False
        prefill_done = False
        wait_prefill_kernel_done = False
        adjust_stream_group = False
        stream_idx = get_current_stream_idx()
        stream_group = self.stream_groups[stream_idx]
        prefill_stream = stream_group[0]
        decode_stream = stream_group[1]
        torch.cuda.empty_cache()

        logger.debug("Starting event loop for pd multiplexing...")
        self._dual_worker_sync(stream_idx)

        import time as _time
        _slo_on = bool(os.environ.get("PDMUX_SLO_SCHED")) or self.r2_policy is not None
        self._slo_last_t = _time.perf_counter()
        while True:
            decode_future = None
            prefill_future = None
            if _slo_on:
                # measure per-iteration wall time = TPOT (one token/iter when decode active);
                # v3: skip the iterations right after a partition switch — the switch drains the
                # GPU (2x synchronize) and that drain would otherwise be misread as a huge TPOT
                # spike, which drove the v1/v2 oscillation.
                _now = _time.perf_counter()
                _dt = (_now - self._slo_last_t) * 1000.0
                self._slo_last_t = _now
                _ema = getattr(self, "_slo_tpot_ema", 0.0)
                # v4: reject outlier spikes. The adjust block does 2x synchronize (drain) at
                # EVERY prefill boundary (frequent in prefill-bound), which shows up as a ~200ms+
                # spike vs the true ~40ms TPOT. Accept _dt only within a band around the running
                # EMA, else the signal is poisoned and the controller oscillates.
                _cap = (3.0 * _ema) if _ema else 1000.0
                _a = float(os.environ.get("PDMUX_SLO_EMA", "0.85"))  # v7b: smoother signal (was 0.7)
                if (not self.running_batch.is_empty()) and 0.0 < _dt < max(_cap, 90.0):
                    self._slo_tpot_ema = (_a * _ema + (1.0 - _a) * _dt) if _ema else _dt
                    self._r2_itl_samples = (
                        getattr(self, "_r2_itl_samples", []) + [_dt]
                    )[-128:]
            with torch.cuda.stream(decode_stream):
                set_pdmux_status(False)
                recv_reqs = self.recv_requests()
                self.process_input_requests(recv_reqs)
                self._dual_worker_sync(stream_idx)

            with torch.cuda.stream(prefill_stream):
                set_pdmux_status(True)
                sm_count = self.sm_counts[stream_idx][0]
                if not wait_prefill_kernel_done:
                    adjust_stream_group = (
                        self.update_split_prefill_batch(sm_count) or adjust_stream_group
                    )

            with torch.cuda.stream(decode_stream):
                set_pdmux_status(False)
                self.running_batch = self.update_running_batch(self.running_batch)
                # Second half of the fallback mechanism.  This fires only once
                # the decode batch has gone EMPTY, and all it does is make
                # `adjust_stream_groups` run so the index can be released to 0
                # (full-SM prefill).  PDMUX_STICKY_PARTITION deliberately
                # leaves it as-is -- see the decode-empty note in
                # `adjust_stream_groups` -- so the two stay consistent.
                adjust_stream_group = adjust_stream_group or (
                    stream_idx > 0 and self.running_batch.is_empty()
                )
                if self.running_batch.is_empty() and self.split_prefill_batch is None:
                    self.check_memory()
                    self.check_tree_cache()
                    self.new_token_ratio = self.init_new_token_ratio
                    self.maybe_sleep_on_idle()

            if (
                _slo_on
                and not self.running_batch.is_empty()
                and (self.split_prefill_batch or self._r2_admission_recheck())
                and not wait_prefill_kernel_done
            ):
                # v7: SLO layer-span evaluation — decide the target split EVERY prefill span,
                # but drain+switch only when it actually changes (converged -> rare -> few drains).
                # Open item (2): direct cost of RUNNING the controller on the single-process
                # event loop. Measured here (the live path) rather than inferred from goodput,
                # which the bench noise swamps. Microsecond means => hypothesis dead.
                import time as _ct
                _ct0 = _ct.perf_counter()
                _tgt = (
                    self._r2_decide_idx(stream_idx)
                    if self.r2_policy is not None
                    else self._slo_decide_idx()
                )
                _cdt = (_ct.perf_counter() - _ct0) * 1000.0
                self._slo_ctl_n = getattr(self, "_slo_ctl_n", 0) + 1
                self._slo_ctl_ms = getattr(self, "_slo_ctl_ms", 0.0) + _cdt
                self._slo_ctl_max = max(getattr(self, "_slo_ctl_max", 0.0), _cdt)
                if self._slo_ctl_n % 200 == 0:
                    logger.info(
                        "SLO-CTLCOST n=%d mean=%.4fms max=%.3fms cum=%.1fms",
                        self._slo_ctl_n, self._slo_ctl_ms / self._slo_ctl_n,
                        self._slo_ctl_max, self._slo_ctl_ms,
                    )
                if _tgt != stream_idx:
                    if getattr(self, "dual_worker_enabled", False):
                        self.dual_worker_state.arbiter.active_roles.clear()
                    prefill_stream.synchronize()
                    decode_stream.synchronize()
                    if getattr(self, "true_dual_worker_runtime", None) is not None:
                        if not self.true_dual_worker_runtime.arbiter.safe_to_switch():
                            raise RuntimeError(
                                "unsafe split transition after stream drain"
                            )
                    set_current_stream_idx(_tgt)
                    self.tp_worker.model_runner.update_decode_attn_backend(_tgt)
                    stream_idx = _tgt
                    stream_group = self.stream_groups[_tgt]
                    prefill_stream = stream_group[0]
                    decode_stream = stream_group[1]
                adjust_stream_group = False
            elif adjust_stream_group:
                if getattr(self, "dual_worker_enabled", False):
                    self.dual_worker_state.arbiter.active_roles.clear()
                prefill_stream.synchronize()
                decode_stream.synchronize()
                if getattr(self, "true_dual_worker_runtime", None) is not None:
                    if not self.true_dual_worker_runtime.arbiter.safe_to_switch():
                        raise RuntimeError(
                            "unsafe automatic split transition after stream drain"
                        )
                stream_idx, stream_group = self.adjust_stream_groups()
                prefill_stream = stream_group[0]
                decode_stream = stream_group[1]
                adjust_stream_group = False
                logger.debug(
                    f"Adjusting stream groups: {stream_idx}, prefill sm: {self.sm_counts[stream_idx][0]}, decode sm: {self.sm_counts[stream_idx][1]}"
                )

            with torch.cuda.stream(decode_stream):
                set_pdmux_status(False)
                # process decode batch
                if self.running_batch and not self.running_batch.is_empty():
                    self._r2_decode_iterations = (
                        getattr(self, "_r2_decode_iterations", 0) + 1
                    )
                    if getattr(self, "dual_worker_enabled", False):
                        self.dual_worker_state.arbiter.select_partition(stream_idx)
                        self.dual_worker_state.arbiter.acquire(WorkerRole.DECODE)
                        self.dual_worker_state.decode.begin_step()
                    if getattr(self, "true_dual_worker_runtime", None) is not None:
                        self.true_dual_worker_runtime.select_partition(stream_idx)
                        decode_batch = self.running_batch
                        decode_future = self.true_dual_worker_runtime.submit(
                            WorkerRole.DECODE,
                            lambda _context, _batch=decode_batch: (
                                self.run_batch(_batch),
                                _context.stream.record_event(),
                            ),
                            stream=decode_stream,
                            decode_backend=getattr(
                                self.tp_worker.model_runner,
                                "decode_attn_backend",
                                None,
                            ),
                        )
                    else:
                        decode_result = self.run_batch(self.running_batch)
                    decode_done = True
                else:
                    decode_done = False
            with torch.cuda.stream(prefill_stream):
                set_pdmux_status(True)
                if (
                    self.split_prefill_batch
                    and not self.split_prefill_batch.is_empty()
                    and not wait_prefill_kernel_done
                ):
                    prefill_done = True
                    forward_count = (
                        max(
                            1,
                            self.pdmux_config.split_forward_token_budget
                            // self.split_prefill_batch.extend_num_tokens,
                        )
                        if self.split_prefill_batch.extend_num_tokens > 0
                        else self.model_config.num_hidden_layers
                    )
                    next_split_index = min(
                        self.split_prefill_batch.split_index + forward_count,
                        self.model_config.num_hidden_layers,
                    )
                    _m = self.tp_worker.model_runner.model
                    if os.environ.get("PDMUX_SLO_SPAN_TYPE") and hasattr(_m, "la_coord_windows"):
                        # type-aware span sizing: cut the prefill span at the current attn/ssm
                        # type-run boundary so each span is layer-type-HOMOGENEOUS (shorten mixed
                        # spans). Layer-type re-enters as SPAN BOUNDARIES, not per-window SM switching.
                        _cur = self.split_prefill_batch.split_index
                        for _s, _e, _a in _m.la_coord_windows():
                            if _s <= _cur < _e:
                                next_split_index = min(next_split_index, _e)
                                break
                    forward_count = (
                        next_split_index - self.split_prefill_batch.split_index
                    )

                    self.split_prefill_batch.split_forward_count = forward_count
                    if getattr(self, "dual_worker_enabled", False):
                        self.dual_worker_state.arbiter.select_partition(stream_idx)
                        self.dual_worker_state.arbiter.acquire(WorkerRole.PREFILL)
                    prefill_is_final = (
                        next_split_index == self.model_config.num_hidden_layers
                    )
                    if getattr(self, "true_dual_worker_runtime", None) is not None:
                        self.true_dual_worker_runtime.select_partition(stream_idx)
                        prefill_batch = self.split_prefill_batch

                        def _run_prefill(
                            _context: ExecutionContext,
                            _batch=prefill_batch,
                        ):
                            _result = self.run_batch(_batch)
                            _event = _context.stream.record_event()
                            return _result, _event

                        prefill_future = self.true_dual_worker_runtime.submit(
                            WorkerRole.PREFILL,
                            _run_prefill,
                            stream=prefill_stream,
                        )
                    else:
                        prefill_result = self.run_batch(self.split_prefill_batch)
                    # A submitted batch belongs to the prefill task until its
                    # future resolves: run_batch reads `split_index` to decide
                    # whether to build `split_forward_batch`
                    # (tp_worker.forward_batch_split_prefill), so advancing it
                    # here raced the worker and crashed job 907032.  The
                    # true-dual path advances it after `prefill_future.result()`
                    # below; the legacy statements and their order are unchanged.
                    if prefill_future is None:
                        if prefill_is_final:
                            self.split_prefill_batch.split_prefill_finished = True
                            prefill_exe_done = prefill_stream.record_event()
                        self.split_prefill_batch.split_index = next_split_index

                elif wait_prefill_kernel_done:
                    prefill_done = True
                else:
                    prefill_done = False

            with torch.cuda.stream(decode_stream):
                set_pdmux_status(False)
                if decode_future is not None:
                    decode_result, threaded_decode_event = decode_future.result()
                    self.true_dual_worker_runtime.record_inflight_event(
                        WorkerRole.DECODE, threaded_decode_event
                    )
                decode_stream.synchronize()
                if decode_done:
                    self.process_batch_result(self.running_batch, decode_result)
                    self._dual_worker_complete_decode(self.running_batch)
                    if getattr(self, "dual_worker_enabled", False):
                        self.dual_worker_state.arbiter.release(WorkerRole.DECODE)

            with torch.cuda.stream(prefill_stream):
                set_pdmux_status(True)
                if prefill_future is not None:
                    prefill_result, threaded_prefill_event = prefill_future.result()
                    self.true_dual_worker_runtime.record_inflight_event(
                        WorkerRole.PREFILL, threaded_prefill_event
                    )
                    # The task has returned the batch: apply the advance that
                    # the legacy path applies right after its synchronous
                    # run_batch (see the ownership note above).
                    if prefill_is_final:
                        self.split_prefill_batch.split_prefill_finished = True
                    self.split_prefill_batch.split_index = next_split_index
                    if self.split_prefill_batch.split_prefill_finished:
                        prefill_exe_done = threaded_prefill_event
                if prefill_done and self.split_prefill_batch.split_prefill_finished:
                    wait_prefill_kernel_done = True
                    prefill_exe_done_flag = prefill_exe_done.query()
                    flags = (
                        torch.ones(1, device="cpu", dtype=torch.int32)
                        if prefill_exe_done_flag
                        else torch.zeros(1, device="cpu", dtype=torch.int32)
                    )

                    self.tp_cpu_group.allreduce(flags, dist.ReduceOp.SUM).wait()
                    if flags.item() == self.tp_size:
                        self._dual_worker_prefill_ready(self.split_prefill_batch)
                        self.process_batch_result(
                            self.split_prefill_batch, prefill_result
                        )
                        if self.running_batch and not self.running_batch.is_empty():
                            self.running_batch.merge_batch(self.split_prefill_batch)
                        else:
                            self.running_batch = self.split_prefill_batch

                        self.split_prefill_batch = None
                        self._dual_worker_start_decode(self.running_batch)
                        if getattr(self, "dual_worker_enabled", False):
                            self.dual_worker_state.arbiter.release(WorkerRole.PREFILL)
                        wait_prefill_kernel_done = False
                        adjust_stream_group = True

            self._dual_worker_sync(stream_idx)

    @torch.inference_mode()
    def event_loop_pdmux_coord(self: Scheduler):
        """Coordinated per-type layer-aware event loop (R0c/A, PDMUX_LA_COORD=1).

        Same as event_loop_pdmux, except when BOTH a decode step and a concurrent
        prefill are active, the decode step is run in layer-TYPE windows: attn windows
        run on a decode-HEAVY coordinated pair (protect the SM-sensitive no-GQA attn),
        mamba windows on a decode-LIGHT pair (release SM), and prefill advances a share
        on the *complementary* prefill-half of the SAME pair per window. So releasing an
        insensitive layer's SM hands prefill the exact complement, coordinated (no overlap).
        """
        from sglang.srt.managers.utils import GenerationBatchResult

        decode_done = False
        prefill_done = False
        wait_prefill_kernel_done = False
        adjust_stream_group = False
        # persist across iterations (like event_loop_pdmux): when wait_prefill_kernel_done
        # is set but the allreduce flag wasn't ready, the next iteration's completion step
        # reuses the prefill_result/prefill_exe_done from when the prefill actually ran.
        decode_result = None
        prefill_result = None
        prefill_exe_done = None
        stream_idx = get_current_stream_idx()
        stream_group = self.stream_groups[stream_idx]
        prefill_stream = stream_group[0]
        decode_stream = stream_group[1]
        torch.cuda.empty_cache()

        # LIGHT = min decode-SM middle group (mamba windows), HEAVY = max (attn windows)
        _mids = list(range(1, self.real_sm_group_num - 1))
        LIGHT = min(_mids, key=lambda i: self.sm_counts[i][1])
        HEAVY = max(_mids, key=lambda i: self.sm_counts[i][1])
        logger.info(
            f"PD-Multiplexing COORD per-type la: LIGHT idx{LIGHT} {self.sm_counts[LIGHT]} "
            f"(mamba), HEAVY idx{HEAVY} {self.sm_counts[HEAVY]} (attn)"
        )
        MR = self.tp_worker.model_runner
        # (a) substrate-isolation variant: PDMUX_LA_COORD_OPT removes the two known
        # implementation overheads — per-window CPU synchronize (-> GPU-side wait_stream/
        # events) and decode SM-pinning (decode-only windows after prefill -> full-SM
        # normal stream). Leaves the *structural* cost (prefill overlaps window 0 only).
        coord_opt = bool(os.environ.get("PDMUX_LA_COORD_OPT"))
        # version-4: faithful sim mechanism — prefill SLICED across mamba (release) windows
        # via lightweight MR.forward (no run_batch), overlapping decode in EVERY release
        # window (not just window 0). Directly attacks the single-window-overlap killer.
        coord_v4 = bool(os.environ.get("PDMUX_LA_COORD_V4"))
        # prefill-type-aware boundary: shift the prefill<->decode SM split by the PREFILL's
        # current layer type (not decode's, as v4/R0d did). attn-prefill (more SM-sensitive)
        # -> prefill-heavy partition (protect prefill); mamba-prefill -> decode-heavy (release
        # prefill SM to decode). Prefill chunk is windowed by its type; decode is sliced across.
        coord_pf = bool(os.environ.get("PDMUX_LA_COORD_PF"))
        FULL = self.real_sm_group_num - 1  # normal full-SM decode stream (0,108)

        while True:
            with torch.cuda.stream(decode_stream):
                set_pdmux_status(False)
                recv_reqs = self.recv_requests()
                self.process_input_requests(recv_reqs)

            with torch.cuda.stream(prefill_stream):
                set_pdmux_status(True)
                sm_count = self.sm_counts[stream_idx][0]
                if not wait_prefill_kernel_done:
                    adjust_stream_group = (
                        self.update_split_prefill_batch(sm_count) or adjust_stream_group
                    )

            with torch.cuda.stream(decode_stream):
                set_pdmux_status(False)
                self.running_batch = self.update_running_batch(self.running_batch)
                adjust_stream_group = adjust_stream_group or (
                    stream_idx > 0 and self.running_batch.is_empty()
                )
                if self.running_batch.is_empty() and self.split_prefill_batch is None:
                    self.check_memory()
                    self.check_tree_cache()
                    self.new_token_ratio = self.init_new_token_ratio
                    self.maybe_sleep_on_idle()

            if adjust_stream_group:
                prefill_stream.synchronize()
                decode_stream.synchronize()
                stream_idx, stream_group = self.adjust_stream_groups()
                prefill_stream = stream_group[0]
                decode_stream = stream_group[1]
                adjust_stream_group = False

            decode_active = bool(self.running_batch) and not self.running_batch.is_empty()
            prefill_active = (
                self.split_prefill_batch is not None
                and not self.split_prefill_batch.is_empty()
                and not wait_prefill_kernel_done
            )

            if decode_active and prefill_active and coord_pf:
                # ===== prefill-type-aware boundary shift =====
                # Window the PREFILL chunk by prefill-layer-type; the (P,D) split follows
                # PREFILL's type: attn-prefill -> LIGHT pair (prefill 92 / decode 16, protect the
                # more-sensitive prefill), mamba-prefill -> HEAVY pair (prefill 34 / decode 74,
                # release prefill SM to decode). Decode (whole 54-layer step) is sliced across the
                # prefill windows on the complementary decode-half. GPU-ordered.
                num_layers = self.model_config.num_hidden_layers
                decode_mwb = self.running_batch.get_model_worker_batch()
                decode_fb = ForwardBatch.init_new(decode_mwb, MR)
                MR.init_decode_metadata_coord(decode_fb, HEAVY)
                if getattr(self.split_prefill_batch, "split_forward_batch", None) is None:
                    _mwb = self.split_prefill_batch.get_model_worker_batch()
                    self.split_prefill_batch.split_forward_batch = ForwardBatch.init_new(_mwb, MR)
                    self.split_prefill_batch.seq_lens_cpu_cache = _mwb.seq_lens_cpu
                pfb = self.split_prefill_batch.split_forward_batch
                ext = self.split_prefill_batch.extend_num_tokens
                fwd_total = (
                    max(1, self.pdmux_config.split_forward_token_budget // ext)
                    if ext > 0 else num_layers
                )
                _pf_chunk = os.environ.get("PDMUX_PF_CHUNK")  # fix #2: cap prefill layers/step
                if _pf_chunk:                                 # (match prefill workload to its low SM)
                    fwd_total = max(1, min(fwd_total, int(_pf_chunk)))
                p_start = self.split_prefill_batch.split_index
                k_total = min(fwd_total, num_layers - p_start)
                # prefill windows (by prefill-layer-type) over the chunk [p_start, p_start+k_total)
                pf_windows = [
                    (max(s, p_start), min(e, p_start + k_total), a)
                    for (s, e, a) in MR.model.la_coord_windows()
                    if e > p_start and s < p_start + k_total
                ]
                n_pw = max(1, len(pf_windows))
                # fix #1: distribute decode ∝ each window's decode-SM (more decode work in the
                # high-decode-SM mamba windows, less in the low-SM attn windows) — match decode
                # workload to its SM instead of the flawed even split.
                win_dsm = [
                    (self.sm_counts[LIGHT][1] if a else self.sm_counts[HEAVY][1])
                    for (_s, _e, a) in pf_windows
                ]
                total_dsm = sum(win_dsm) or 1
                cum_dsm = 0
                logits_output = None
                plog = None
                prefill_done = True
                prev_d = prev_p = None
                d_done = 0
                for wi, (ps, pe, is_attn) in enumerate(pf_windows):
                    idx = LIGHT if is_attn else HEAVY  # prefill-attn->prefill-heavy(92); mamba->decode-heavy(74)
                    p_s, d_s = self.stream_groups[idx][0], self.stream_groups[idx][1]
                    with torch.cuda.stream(p_s):
                        set_pdmux_status(True)
                        if prev_d is not None:
                            p_s.wait_stream(prev_d)
                        if prev_p is not None:
                            p_s.wait_stream(prev_p)
                        _pout = MR.forward(pfb, split_forward_count=(pe - ps))
                        if _pout.logits_output is not None:
                            plog = _pout.logits_output
                    cum_dsm += win_dsm[wi]
                    d_target = num_layers if wi == n_pw - 1 else min(
                        num_layers, round(num_layers * cum_dsm / total_dsm)
                    )
                    if d_target > d_done:
                        with torch.cuda.stream(d_s):
                            set_pdmux_status(False)
                            if prev_d is not None:
                                d_s.wait_stream(prev_d)
                            if prev_p is not None:
                                d_s.wait_stream(prev_p)
                            r = MR.forward_split_decode(decode_fb, (d_done, d_target))
                            if r is not None:
                                logits_output = r
                        d_done = d_target
                    prev_d, prev_p = d_s, p_s
                if d_done < num_layers:  # safety: finish decode
                    d_s = self.stream_groups[HEAVY][1]
                    with torch.cuda.stream(d_s):
                        set_pdmux_status(False)
                        if prev_d is not None:
                            d_s.wait_stream(prev_d)
                        if prev_p is not None:
                            d_s.wait_stream(prev_p)
                        r = MR.forward_split_decode(decode_fb, (d_done, num_layers))
                        if r is not None:
                            logits_output = r
                    prev_d = d_s
                if prev_d is not None:
                    prev_d.synchronize()
                if prev_p is not None:
                    prev_p.synchronize()
                next_ids = MR.sample(logits_output, decode_fb)
                self.running_batch.output_ids = next_ids
                decode_result = GenerationBatchResult(logits_output=logits_output)
                decode_result.next_token_ids = next_ids
                decode_done = True
                self.split_prefill_batch.split_index = pfb.split_index
                if pfb.split_index >= num_layers:
                    self.split_prefill_batch.split_prefill_finished = True
                    prefill_exe_done = (prev_p or prev_d).record_event()
                    pf_next = MR.sample(plog, pfb) if plog is not None else None
                    prefill_result = GenerationBatchResult(logits_output=plog)
                    prefill_result.next_token_ids = pf_next
                    self.split_prefill_batch.output_ids = pf_next
                stream_idx = HEAVY
                set_current_stream_idx(HEAVY)
                stream_group = self.stream_groups[HEAVY]
                prefill_stream = stream_group[0]
                decode_stream = stream_group[1]
            elif decode_active and prefill_active and coord_v4:
                # ===== version-4: sim's multi-window overlap, done cheaply =====
                # Prefill is SLICED across the mamba (release) windows via a lightweight
                # MR.forward on a PERSISTENT split_forward_batch (NO run_batch, NO per-window
                # get_model_worker_batch -> avoids inefficient_v1's 698ms). In each mamba window
                # decode runs the mamba layers on the decode-half (16 SM) while prefill advances a
                # slice on the disjoint prefill-half (92 SM); attn windows protect decode (74 SM)
                # and pause prefill. Windows serialize on GPU (consecutive green-ctx partitions
                # share physical SMs); within a window decode-slice ∥ prefill-slice = sim's vision.
                num_layers = self.model_config.num_hidden_layers
                windows = MR.model.la_coord_windows()
                decode_mwb = self.running_batch.get_model_worker_batch()
                decode_fb = ForwardBatch.init_new(decode_mwb, MR)
                MR.init_decode_metadata_coord(decode_fb, HEAVY)
                # persistent prefill forward batch (threads hidden state across windows & steps)
                if getattr(self.split_prefill_batch, "split_forward_batch", None) is None:
                    _mwb = self.split_prefill_batch.get_model_worker_batch()
                    self.split_prefill_batch.split_forward_batch = ForwardBatch.init_new(_mwb, MR)
                    self.split_prefill_batch.seq_lens_cpu_cache = _mwb.seq_lens_cpu
                pfb = self.split_prefill_batch.split_forward_batch
                ext = self.split_prefill_batch.extend_num_tokens
                fwd_total = (
                    max(1, self.pdmux_config.split_forward_token_budget // ext)
                    if ext > 0 else num_layers
                )
                remaining_pf = min(fwd_total, num_layers - self.split_prefill_batch.split_index)
                mamba_wins = sum(1 for (_s, _e, _a) in windows if not _a) or 1
                per_win = -(-remaining_pf // mamba_wins)   # ceil divide over release windows
                logits_output = None
                plog = None
                prefill_done = True
                prev_d = prev_p = None
                for wi, (s, e, is_attn) in enumerate(windows):
                    idx = HEAVY if is_attn else LIGHT
                    set_current_stream_idx(idx)
                    p_s, d_s = self.stream_groups[idx][0], self.stream_groups[idx][1]
                    with torch.cuda.stream(d_s):
                        set_pdmux_status(False)
                        if prev_d is not None:
                            d_s.wait_stream(prev_d)     # serialize windows (shared physical SMs)
                        if prev_p is not None:
                            d_s.wait_stream(prev_p)
                        r = MR.forward_split_decode(decode_fb, (s, e))
                        if r is not None:
                            logits_output = r
                    k = min(per_win, remaining_pf) if (not is_attn and remaining_pf > 0) else 0
                    if k > 0:
                        with torch.cuda.stream(p_s):
                            set_pdmux_status(True)
                            if prev_d is not None:
                                p_s.wait_stream(prev_d)
                            if prev_p is not None:
                                p_s.wait_stream(prev_p)
                            _pout = MR.forward(pfb, split_forward_count=k)
                            if _pout.logits_output is not None:
                                plog = _pout.logits_output
                        remaining_pf -= k
                        prev_p = p_s
                    prev_d = d_s
                if prev_d is not None:
                    prev_d.synchronize()
                if prev_p is not None:
                    prev_p.synchronize()
                # sample decode (after final window)
                next_ids = MR.sample(logits_output, decode_fb)
                self.running_batch.output_ids = next_ids
                decode_result = GenerationBatchResult(logits_output=logits_output)
                decode_result.next_token_ids = next_ids
                decode_done = True
                # prefill completion bookkeeping (mirror baseline; steps 7-8 consume these)
                self.split_prefill_batch.split_index = pfb.split_index
                if pfb.split_index >= num_layers:
                    self.split_prefill_batch.split_prefill_finished = True
                    prefill_exe_done = (prev_p or prev_d).record_event()
                    pf_next = MR.sample(plog, pfb) if plog is not None else None
                    prefill_result = GenerationBatchResult(logits_output=plog)
                    prefill_result.next_token_ids = pf_next
                    # run_batch normally sets batch.output_ids (merge_batch cats it into the
                    # running_batch); we bypass run_batch for prefill, so set it here — exactly
                    # like the decode side sets self.running_batch.output_ids above.
                    self.split_prefill_batch.output_ids = pf_next
                # settle stream refs on HEAVY pair for steps 7-8
                stream_idx = HEAVY
                set_current_stream_idx(HEAVY)
                stream_group = self.stream_groups[HEAVY]
                prefill_stream = stream_group[0]
                decode_stream = stream_group[1]
            elif decode_active and prefill_active:
                # ===== coordinated per-type windowed interleave =====
                num_layers = self.model_config.num_hidden_layers
                windows = MR.model.la_coord_windows()
                n_win = len(windows)
                decode_mwb = self.running_batch.get_model_worker_batch()
                decode_fb = ForwardBatch.init_new(decode_mwb, MR)
                MR.init_decode_metadata_coord(decode_fb, HEAVY)  # attn windows share heavy backend

                ext = self.split_prefill_batch.extend_num_tokens
                fwd_total = (
                    max(1, self.pdmux_config.split_forward_token_budget // ext)
                    if ext > 0
                    else num_layers
                )
                logits_output = None
                prefill_done = True
                full_d = self.stream_groups[FULL][1]   # normal full-SM decode stream
                prev_d = None                          # GPU-side window ordering (opt)
                prefill_evt = None
                for wi, (s, e, is_attn) in enumerate(windows):
                    if coord_opt and wi > 0:
                        # pin removal: prefill already ran in window 0, so decode-only
                        # windows run on the full-SM (unmasked) stream instead of the
                        # LIGHT/HEAVY sub-partition. attn backend (HEAVY) is stream-agnostic.
                        p_s = self.stream_groups[LIGHT][0]
                        d_s = full_d
                    else:
                        idx = HEAVY if is_attn else LIGHT
                        set_current_stream_idx(idx)
                        p_s, d_s = self.stream_groups[idx][0], self.stream_groups[idx][1]
                    with torch.cuda.stream(d_s):
                        set_pdmux_status(False)
                        if coord_opt:
                            if prev_d is not None:
                                d_s.wait_stream(prev_d)      # order windows on GPU (no CPU block)
                            if wi == 1 and prefill_evt is not None:
                                d_s.wait_event(prefill_evt)  # full-SM decode waits prefill-tail done
                        r = MR.forward_split_decode(decode_fb, (s, e))
                        if r is not None:
                            logits_output = r
                    prev_d = d_s
                    # advance the WHOLE prefill chunk once, in the first (mamba/LIGHT) window
                    # where prefill gets the large complementary partition (avoids 19x run_batch).
                    if wi == 0 and self.split_prefill_batch.split_index < num_layers:
                        k = min(fwd_total, num_layers - self.split_prefill_batch.split_index)
                        self.split_prefill_batch.split_forward_count = k
                        with torch.cuda.stream(p_s):
                            set_pdmux_status(True)
                            prefill_result = self.run_batch(self.split_prefill_batch)
                            if coord_opt:
                                prefill_evt = p_s.record_event()
                        nxt = self.split_prefill_batch.split_index + k
                        if nxt >= num_layers:
                            self.split_prefill_batch.split_prefill_finished = True
                            prefill_exe_done = p_s.record_event()
                        self.split_prefill_batch.split_index = nxt
                    if not coord_opt:
                        d_s.synchronize()
                        if wi == 0:
                            p_s.synchronize()
                if coord_opt and prev_d is not None:
                    prev_d.synchronize()                     # single final decode sync before sample

                # sample decode (after final window)
                next_ids = MR.sample(logits_output, decode_fb)
                # run_batch normally sets batch.output_ids; next prepare_for_decode reads it
                # as input_ids. We bypass run_batch for decode, so set it here.
                self.running_batch.output_ids = next_ids
                decode_result = GenerationBatchResult(logits_output=logits_output)
                decode_result.next_token_ids = next_ids
                decode_done = True
                # settle stream refs on HEAVY pair for steps 7-8
                stream_idx = HEAVY
                set_current_stream_idx(HEAVY)
                stream_group = self.stream_groups[HEAVY]
                prefill_stream = stream_group[0]
                decode_stream = stream_group[1]
            else:
                # ---- fallback: decode-only or prefill-only (original behavior) ----
                with torch.cuda.stream(decode_stream):
                    set_pdmux_status(False)
                    if decode_active:
                        decode_result = self.run_batch(self.running_batch)
                        decode_done = True
                    else:
                        decode_done = False
                with torch.cuda.stream(prefill_stream):
                    set_pdmux_status(True)
                    if prefill_active:
                        prefill_done = True
                        ext = self.split_prefill_batch.extend_num_tokens
                        forward_count = (
                            max(1, self.pdmux_config.split_forward_token_budget // ext)
                            if ext > 0
                            else self.model_config.num_hidden_layers
                        )
                        next_split_index = min(
                            self.split_prefill_batch.split_index + forward_count,
                            self.model_config.num_hidden_layers,
                        )
                        forward_count = next_split_index - self.split_prefill_batch.split_index
                        self.split_prefill_batch.split_forward_count = forward_count
                        prefill_result = self.run_batch(self.split_prefill_batch)
                        if next_split_index == self.model_config.num_hidden_layers:
                            self.split_prefill_batch.split_prefill_finished = True
                            prefill_exe_done = prefill_stream.record_event()
                        self.split_prefill_batch.split_index = next_split_index
                    elif wait_prefill_kernel_done:
                        prefill_done = True
                    else:
                        prefill_done = False

            with torch.cuda.stream(decode_stream):
                set_pdmux_status(False)
                decode_stream.synchronize()
                if decode_done:
                    self.process_batch_result(self.running_batch, decode_result)

            with torch.cuda.stream(prefill_stream):
                set_pdmux_status(True)
                if prefill_done and self.split_prefill_batch.split_prefill_finished:
                    wait_prefill_kernel_done = True
                    prefill_exe_done_flag = prefill_exe_done.query()
                    flags = (
                        torch.ones(1, device="cpu", dtype=torch.int32)
                        if prefill_exe_done_flag
                        else torch.zeros(1, device="cpu", dtype=torch.int32)
                    )
                    self.tp_cpu_group.allreduce(flags, dist.ReduceOp.SUM).wait()
                    if flags.item() == self.tp_size:
                        self.process_batch_result(
                            self.split_prefill_batch, prefill_result
                        )
                        if self.running_batch and not self.running_batch.is_empty():
                            self.running_batch.merge_batch(self.split_prefill_batch)
                        else:
                            self.running_batch = self.split_prefill_batch
                        self.split_prefill_batch = None
                        wait_prefill_kernel_done = False
                        adjust_stream_group = True

    # ------------------------------------------------------------------
    # R2 admission-limit liveness (2026-09-11)
    # ------------------------------------------------------------------
    # Placed at the END of the class on purpose: the two call sites
    # (`update_split_prefill_batch` and the R2 gate in `event_loop_pdmux`)
    # were edited in place, so no earlier line of this module moved and every
    # snapshotted `multiplexing_mixin.py:NNN` citation
    # (scripts/discipline/line_citations.json) stays valid.
    #
    # THE BUG (reports/r2_decoupling_review_2026-07-24.md, item 4).
    # `r2_admission_limited` is WRITTEN only by `_r2_decide_idx`, which the
    # event loop calls only while a split prefill batch is in flight, and it
    # is READ only by `update_split_prefill_batch`, which reaches that check
    # only when NO split prefill batch is in flight.  The write set and the
    # read set are disjoint: once the last in-span decision said "limited"
    # and that prefill drained, nothing could write the latch again, so
    # "latched with no prefill in flight" was an absorbing state and prefill
    # admission stopped for the rest of the run.  Before this fix the latch
    # had exactly two possible effects -- none, or a permanent stall -- and
    # never the temporary limit the controller asks for.
    #
    # WHY NOT "CLEAR ON DRAIN".  Every read happens after a drain, so clearing
    # the latch where the split batch is set to None would make it False at
    # every read: that deletes the R2 proactive admission limit instead of
    # fixing it.  tests/test_r2_admission_latch.py runs such a variant and it
    # fails the safety test.
    #
    # THE FIX (minimal; the policy's admission semantics are unchanged).
    #   1. While the latch is set and no prefill is in flight, the R2 policy
    #      is ALSO consulted on decode-only iterations
    #      (`_r2_admission_recheck` widens the existing gate).  Its decision
    #      goes through the same drain/switch code as an in-span decision, so
    #      the controller's view of the current partition stays true and no
    #      telemetry record describes a transition that was not applied.
    #   2. With the running batch EMPTY that gate cannot run (it needs decode
    #      state) and there is no decode left to protect, so the limit is
    #      released at the admission point (`_r2_admission_holds`).  Without
    #      this, a controller that keeps reporting overload from stale ITL
    #      samples could still stall an idle server.
    #
    # SCOPE.  FixedPolicy never sets `admission_limited` (controller.py:68,
    # :84-94), so under PDMUX_R2_POLICY unset/fixed the latch stays False,
    # `_r2_admission_holds` is never called, `_r2_admission_recheck` returns
    # False without emitting anything, and the loop is unchanged.  Only
    # generic/hybrid (CoarseGrainedController, controller.py:182-183) can
    # set the latch.
    def _r2_admission_recheck(self: Scheduler) -> bool:
        """True iff the R2 admission limit is set with no prefill in flight.

        Only caller: the R2 gate in `event_loop_pdmux`, after its
        `self.split_prefill_batch` operand is falsy.  Emits
        `r2_admission_recheck` so a consumer can tell the `controller_decision`
        that follows from an in-span one (that record's schema is left as is).
        """
        if getattr(self, "r2_policy", None) is None or not getattr(
            self, "r2_admission_limited", False
        ):
            return False
        self.pdmux_telemetry.emit(
            "r2_admission_recheck",
            self.pdmux_experiment_phase,
            policy=self.r2_policy_name,
            running_batch_size=self.running_batch.batch_size(),
            prefill_queue_depth=len(self.waiting_queue),
        )
        return True

    def _r2_admission_holds(self: Scheduler) -> bool:
        """Whether a set R2 admission limit still applies at an admission point.

        Only caller: `update_split_prefill_batch`, and only when the latch is
        set.  Releases the limit when the running batch is empty: the limit
        exists to let decode drain, and an empty batch means it has.
        """
        running = getattr(self, "running_batch", None)
        if running is not None and not running.is_empty():
            return True
        self.r2_admission_limited = False
        # The controller carries its own copy of the verdict across
        # non-evaluation iterations (controller.py `stabilize` HOLD branch), so
        # clearing only the scheduler flag would let the very next HOLD
        # decision re-assert the limit this backstop just released.
        controller = getattr(getattr(self, "r2_policy", None), "controller", None)
        release = getattr(controller, "release_admission_limit", None)
        if callable(release):
            release()
        self.pdmux_telemetry.emit(
            "r2_admission_released",
            self.pdmux_experiment_phase,
            policy=self.r2_policy_name,
            reason="running_batch_empty",
        )
        return False

    # ------------------------------------------------------------------
    # Hybrid-profile runtime environment + provenance (2026-09-12)
    # ------------------------------------------------------------------
    # Appended at the END of the class, and reached by changing exactly ONE
    # existing line in `_build_r2_policy` (`RuntimeEnvironment(` ->
    # `self._r2_runtime_environment(profile,`), so no earlier line of this
    # module moves and every snapshotted `multiplexing_mixin.py:NNN` citation
    # (scripts/discipline/line_citations.json) stays valid -- same reason as
    # the R2 admission-liveness helpers above.
    #
    # WHY THIS EXISTS.  `HybridModelProfileV1.is_compatible` compares engine
    # identity on `engine_source_hash` (engine SOURCE CONTENT) instead of
    # `engine_commit` (repo HEAD), because a docs-only commit moved HEAD and
    # dropped every hybrid arm into the conservative fallback
    # (`upper_bound_itl_ms=inf`).  `engine_commit` is still recorded, as
    # provenance only.  The hash is FAIL-CLOSED: if it cannot be computed the
    # profile is treated as incompatible, so the record below is the only place
    # that says why a run fell back, and it is written once at startup whether
    # the profile matched or not.
    #
    # SCOPE.  Hybrid policy construction only (`PDMUX_R2_POLICY=hybrid`).
    # FixedPolicy/generic never build a RuntimeEnvironment, never consult a
    # profile, and never reach this method.
    def _r2_runtime_environment(
        self: Scheduler, profile: HybridModelProfileV1, **fields
    ) -> RuntimeEnvironment:
        # Local import: adding a name to the module's import block would shift
        # every line below it and invalidate the snapshotted citations above.
        from sglang.srt.multiplex.profile import engine_source_hash

        source_hash = engine_source_hash()
        environment = RuntimeEnvironment(engine_source_hash=source_hash, **fields)
        compatible, reasons = profile.is_compatible(environment)
        self.pdmux_telemetry.emit(
            "r2_profile_environment",
            self.pdmux_experiment_phase,
            policy=self.r2_policy_name,
            engine_commit=environment.engine_commit,
            engine_source_hash=source_hash or "unavailable",
            profile_engine_commit=profile.environment.engine_commit,
            profile_engine_source_hash=(
                profile.environment.engine_source_hash or "unavailable"
            ),
            profile_model_id=profile.model_id,
            compatible=compatible,
            reasons="; ".join(reasons),
        )
        if not compatible:
            logger.warning(
                "PD-mux hybrid profile is NOT compatible with this runtime; the "
                "decode-floor estimator will return its conservative fallback "
                "(upper_bound_itl_ms=inf) for every decision: %s",
                "; ".join(reasons),
            )
        return environment
