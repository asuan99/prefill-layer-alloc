"""Role-local state for the experimental PD-mux dual-worker path.

This module deliberately contains no CUDA or SGLang scheduler imports. The
workers are cooperative objects owned by the scheduler (one process and one
CUDA context), not Python threads. That makes queue ownership explicit while
leaving GPU execution and request/KV-cache lifetime in the scheduler until the
new path has passed correctness checks.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, List, MutableSequence, Optional


class WorkerRole(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"


class RequestPhase(str, Enum):
    PREFILL_WAITING = "prefill_waiting"
    PREFILL_RUNNING = "prefill_running"
    DECODE_READY = "decode_ready"
    DECODE_RUNNING = "decode_running"
    COMPLETE = "complete"


@dataclass
class WorkerSnapshot:
    queue_depth: int = 0
    queue_age_ms: float = 0.0
    active_batch_size: int = 0
    admission_latency_ms: float = 0.0
    phase_time_ms: float = 0.0
    step_count: int = 0
    last_tpot_ms: float = 0.0
    admission_blocked: bool = False
    admission_block_reason: str = ""


def _batch_size(batch: Any) -> int:
    if batch is None:
        return 0
    try:
        return int(batch.batch_size())
    except (AttributeError, TypeError):
        return len(getattr(batch, "reqs", ()))


def _request_age_ms(req: Any, now: float) -> float:
    stats = getattr(req, "time_stats", None)
    entry = getattr(stats, "wait_queue_entry_time", 0.0) if stats else 0.0
    if not entry:
        return 0.0
    return max(0.0, (now - float(entry)) * 1000.0)


@dataclass
class PrefillWorker:
    """Owns prefill waiting/active state and TTFT-side telemetry."""

    waiting_queue: Optional[MutableSequence[Any]] = None
    active_batch: Any = None
    chunk_progress: int = 0
    snapshot: WorkerSnapshot = field(default_factory=WorkerSnapshot)
    _admission_started_at: Optional[float] = field(default=None, init=False)

    def bind_queue(self, queue: MutableSequence[Any]) -> None:
        self.waiting_queue = queue

    def observe(self, now: Optional[float] = None) -> WorkerSnapshot:
        now = time.perf_counter() if now is None else now
        queue = self.waiting_queue or ()
        ages = [_request_age_ms(req, now) for req in list(queue)[:128]]
        self.snapshot.queue_depth = len(queue)
        self.snapshot.queue_age_ms = max(ages, default=0.0)
        self.snapshot.active_batch_size = _batch_size(self.active_batch)
        return self.snapshot

    def begin_admission(self, now: Optional[float] = None) -> None:
        self._admission_started_at = time.perf_counter() if now is None else now

    def finish_admission(self, now: Optional[float] = None) -> None:
        if self._admission_started_at is None:
            return
        now = time.perf_counter() if now is None else now
        self.snapshot.admission_latency_ms = max(
            0.0, (now - self._admission_started_at) * 1000.0
        )
        self._admission_started_at = None


@dataclass
class DecodeWorker:
    """Owns the decode-ready/running state and TPOT-side telemetry."""

    ready_queue: List[Any] = field(default_factory=list)
    running_batch: Any = None
    snapshot: WorkerSnapshot = field(default_factory=WorkerSnapshot)
    _step_started_at: Optional[float] = field(default=None, init=False)

    def observe(self) -> WorkerSnapshot:
        self.snapshot.queue_depth = len(self.ready_queue)
        self.snapshot.active_batch_size = _batch_size(self.running_batch)
        return self.snapshot

    def begin_step(self, now: Optional[float] = None) -> None:
        self._step_started_at = time.perf_counter() if now is None else now

    def finish_step(self, now: Optional[float] = None) -> None:
        if self._step_started_at is None:
            return
        now = time.perf_counter() if now is None else now
        self.snapshot.last_tpot_ms = max(
            0.0, (now - self._step_started_at) * 1000.0
        )
        self.snapshot.step_count += 1
        self._step_started_at = None


@dataclass(frozen=True)
class ResourceLease:
    role: WorkerRole
    stream_index: int
    prefill_sms: int
    decode_sms: int
    generation: int


class SharedGpuArbiter:
    """Tracks one shared whole-phase GPU partition for both workers.

    Both roles may hold the same lease concurrently, representing overlap on
    the two streams. A second stream index cannot be acquired until the current
    iteration is drained, which prevents accidental SM oversubscription.
    """

    def __init__(self, sm_counts: Iterable[tuple[int, int]]):
        self.sm_counts = tuple((int(p), int(d)) for p, d in sm_counts)
        self.stream_index = 0
        self.generation = 0
        self.active_roles: set[WorkerRole] = set()

    def select_partition(self, stream_index: int) -> None:
        stream_index = int(stream_index)
        if not 0 <= stream_index < len(self.sm_counts):
            raise IndexError(f"invalid PD-mux stream index: {stream_index}")
        if self.active_roles and stream_index != self.stream_index:
            raise RuntimeError("cannot switch PD-mux partition while leases are active")
        if stream_index != self.stream_index:
            self.stream_index = stream_index
            self.generation += 1

    def acquire(self, role: WorkerRole) -> ResourceLease:
        self.active_roles.add(role)
        prefill_sms, decode_sms = self.sm_counts[self.stream_index]
        return ResourceLease(
            role, self.stream_index, prefill_sms, decode_sms, self.generation
        )

    def release(self, role: WorkerRole) -> None:
        self.active_roles.discard(role)

    def assert_no_oversubscription(self) -> None:
        if len(self.active_roles) > 2:
            raise AssertionError("more than two PD-mux worker leases are active")


class PhaseCoordinator:
    """Coordinates request phase transitions and minimal shared metadata."""

    def __init__(self):
        self.request_phase: dict[Any, RequestPhase] = {}

    @staticmethod
    def _rid(req: Any) -> Any:
        return getattr(req, "rid", id(req))

    def register_waiting(self, req: Any) -> None:
        self.request_phase.setdefault(self._rid(req), RequestPhase.PREFILL_WAITING)

    def start_prefill(self, reqs: Iterable[Any]) -> None:
        for req in reqs:
            self.request_phase[self._rid(req)] = RequestPhase.PREFILL_RUNNING

    def complete_prefill(self, reqs: Iterable[Any]) -> None:
        for req in reqs:
            self.request_phase[self._rid(req)] = RequestPhase.DECODE_READY

    def start_decode(self, reqs: Iterable[Any]) -> None:
        for req in reqs:
            self.request_phase[self._rid(req)] = RequestPhase.DECODE_RUNNING

    def complete(self, reqs: Iterable[Any]) -> None:
        for req in reqs:
            self.request_phase[self._rid(req)] = RequestPhase.COMPLETE

    def prune(self, reqs: Iterable[Any]) -> None:
        """Drop request metadata that is no longer owned by the scheduler."""
        live = {self._rid(req) for req in reqs}
        for rid in tuple(self.request_phase):
            if rid not in live:
                del self.request_phase[rid]


@dataclass
class DualWorkerState:
    """Composition root for the experimental single-process dual-worker path."""

    prefill: PrefillWorker
    decode: DecodeWorker
    arbiter: SharedGpuArbiter
    coordinator: PhaseCoordinator

    @classmethod
    def from_sm_counts(cls, sm_counts: Iterable[tuple[int, int]]) -> "DualWorkerState":
        return cls(
            prefill=PrefillWorker(),
            decode=DecodeWorker(),
            arbiter=SharedGpuArbiter(sm_counts),
            coordinator=PhaseCoordinator(),
        )

    def observe_scheduler(
        self, scheduler: Any, stream_index: int, now: Optional[float] = None
    ) -> None:
        self.prefill.bind_queue(scheduler.waiting_queue)
        self.prefill.active_batch = getattr(scheduler, "split_prefill_batch", None)
        self.prefill.chunk_progress = int(
            getattr(self.prefill.active_batch, "split_index", 0) or 0
        )
        self.decode.running_batch = getattr(scheduler, "running_batch", None)
        for req in list(scheduler.waiting_queue):
            self.coordinator.register_waiting(req)

        live_reqs = list(scheduler.waiting_queue)
        for batch in (self.prefill.active_batch, self.decode.running_batch):
            live_reqs.extend(getattr(batch, "reqs", ()) or ())
        finished_reqs = []
        for req in live_reqs:
            finished = getattr(req, "finished", None)
            if callable(finished) and finished():
                finished_reqs.append(req)
        self.coordinator.complete(finished_reqs)
        self.coordinator.prune(live_reqs)

        self.prefill.observe(now=now)
        self.decode.observe()
        if self.prefill.snapshot.queue_depth and self.prefill.snapshot.active_batch_size == 0:
            self.prefill.snapshot.admission_blocked = True
            running = getattr(scheduler, "running_batch", None)
            self.prefill.snapshot.admission_block_reason = (
                "shared_batch_capacity"
                if bool(getattr(running, "batch_is_full", False))
                else "scheduler_admission_pending"
            )
        else:
            self.prefill.snapshot.admission_blocked = False
            self.prefill.snapshot.admission_block_reason = ""
        self.arbiter.select_partition(stream_index)

    def metrics(self) -> dict[str, Any]:
        """Return queue/progress/resource metrics for experiment trace writers."""
        p = self.prefill.snapshot
        d = self.decode.snapshot
        prefill_sms, decode_sms = self.arbiter.sm_counts[self.arbiter.stream_index]
        return {
            "prefill_queue_depth": p.queue_depth,
            "prefill_queue_age_ms": p.queue_age_ms,
            "prefill_active_batch_size": p.active_batch_size,
            "prefill_chunk_progress": self.prefill.chunk_progress,
            "prefill_admission_latency_ms": p.admission_latency_ms,
            "prefill_admission_blocked": p.admission_blocked,
            "prefill_admission_block_reason": p.admission_block_reason,
            "decode_ready_queue_depth": d.queue_depth,
            "decode_running_batch_size": d.active_batch_size,
            "decode_last_tpot_ms": d.last_tpot_ms,
            "decode_step_count": d.step_count,
            "stream_index": self.arbiter.stream_index,
            "prefill_sms": prefill_sms,
            "decode_sms": decode_sms,
            "active_worker_leases": len(self.arbiter.active_roles),
        }
