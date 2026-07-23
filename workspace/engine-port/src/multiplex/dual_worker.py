"""Role-local state and host executors for PD-mux.

``DualWorkerState`` retains the R1 observer path for reproducibility.
``TrueDualWorkerRuntime`` is the R2 architecture: two long-lived host threads,
two role-owned task queues, explicit execution contexts, and a central
whole-phase resource arbiter.  CUDA/SGLang are injected by the integration
    layer so these concurrency primitives remain CPU-testable.
"""

from __future__ import annotations

import time
import queue
import threading
from concurrent.futures import Future
from contextlib import nullcontext
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, ContextManager, Iterable, List, MutableSequence, Optional


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


@dataclass(frozen=True)
class ExecutionContext:
    """Immutable role context passed to every host-thread task."""

    role: WorkerRole
    stream_index: int
    prefill_sms: int
    decode_sms: int
    generation: int
    stream: Any = None
    decode_backend: Any = None


class SharedGpuArbiter:
    """Tracks one shared whole-phase GPU partition for both workers.

    Both roles may hold the same lease concurrently, representing overlap on
    the two streams. A second stream index cannot be acquired until the current
    iteration is drained, which prevents accidental SM oversubscription.
    """

    def __init__(self, sm_counts: Iterable[tuple[int, int]]):
        self.sm_counts = tuple((int(p), int(d)) for p, d in sm_counts)
        if not self.sm_counts:
            raise ValueError("at least one PD-mux partition is required")
        self.stream_index = 0
        self.generation = 0
        self.active_roles: set[WorkerRole] = set()
        self.inflight_events: dict[WorkerRole, Any] = {}
        self._condition = threading.Condition()

    def select_partition(self, stream_index: int) -> None:
        stream_index = int(stream_index)
        with self._condition:
            if not 0 <= stream_index < len(self.sm_counts):
                raise IndexError(f"invalid PD-mux stream index: {stream_index}")
            if stream_index != self.stream_index and not self.safe_to_switch():
                raise RuntimeError(
                    "cannot switch PD-mux partition while leases/events are active"
                )
            if stream_index != self.stream_index:
                self.stream_index = stream_index
                self.generation += 1
                self._condition.notify_all()

    def acquire(self, role: WorkerRole, strict: bool = False) -> ResourceLease:
        with self._condition:
            if strict and role in self.active_roles:
                raise RuntimeError(f"{role.value} already holds a GPU lease")
            self.active_roles.add(role)
            prefill_sms, decode_sms = self.sm_counts[self.stream_index]
            return ResourceLease(
                role, self.stream_index, prefill_sms, decode_sms, self.generation
            )

    def release(self, role: WorkerRole) -> None:
        with self._condition:
            self.active_roles.discard(role)
            self._condition.notify_all()

    def safe_to_switch(self) -> bool:
        with self._condition:
            if self.active_roles:
                return False
            for event in self.inflight_events.values():
                try:
                    if not event.query():
                        return False
                except AttributeError:
                    return False
            return True

    def record_inflight_event(self, role: WorkerRole, event: Any) -> None:
        with self._condition:
            self.inflight_events[role] = event

    def wait_until_idle(self, timeout_s: Optional[float] = None) -> bool:
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        with self._condition:
            while not self.safe_to_switch():
                remaining = (
                    None if deadline is None else max(0.0, deadline - time.monotonic())
                )
                if remaining == 0.0:
                    return False
                self._condition.wait(
                    0.005 if remaining is None else min(0.005, remaining)
                )
            return True

    def assert_no_oversubscription(self) -> None:
        with self._condition:
            if len(self.active_roles) > 2:
                raise AssertionError("more than two PD-mux worker leases are active")


class PhaseCoordinator:
    """Coordinates request phase transitions and minimal shared metadata."""

    def __init__(self):
        self.request_phase: dict[Any, RequestPhase] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _rid(req: Any) -> Any:
        return getattr(req, "rid", id(req))

    def register_waiting(self, req: Any) -> None:
        with self._lock:
            self.request_phase.setdefault(self._rid(req), RequestPhase.PREFILL_WAITING)

    def start_prefill(self, reqs: Iterable[Any]) -> None:
        with self._lock:
            for req in reqs:
                self.request_phase[self._rid(req)] = RequestPhase.PREFILL_RUNNING

    def complete_prefill(self, reqs: Iterable[Any]) -> None:
        with self._lock:
            for req in reqs:
                self.request_phase[self._rid(req)] = RequestPhase.DECODE_READY

    def start_decode(self, reqs: Iterable[Any]) -> None:
        with self._lock:
            for req in reqs:
                self.request_phase[self._rid(req)] = RequestPhase.DECODE_RUNNING

    def complete(self, reqs: Iterable[Any]) -> None:
        with self._lock:
            for req in reqs:
                self.request_phase[self._rid(req)] = RequestPhase.COMPLETE

    def prune(self, reqs: Iterable[Any]) -> None:
        """Drop request metadata that is no longer owned by the scheduler."""
        live = {self._rid(req) for req in reqs}
        with self._lock:
            for rid in tuple(self.request_phase):
                if rid not in live:
                    del self.request_phase[rid]


@dataclass
class _RoleTask:
    context: ExecutionContext
    callback: Callable[[ExecutionContext], Any]
    future: Future


class RoleWorkerThread:
    """A long-lived role thread with an explicit condition-backed work queue."""

    def __init__(
        self,
        role: WorkerRole,
        activate: Optional[Callable[[ExecutionContext], ContextManager[Any]]] = None,
    ):
        self.role = role
        self._activate = activate or (lambda context: nullcontext())
        self._tasks: "queue.Queue[Optional[_RoleTask]]" = queue.Queue()
        self._thread = threading.Thread(
            target=self._loop,
            name=f"pdmux-{role.value}-worker",
            daemon=True,
        )
        self._started = False
        self._closed = False
        self.busy_time_s = 0.0
        self.task_count = 0
        self.last_error = ""
        self._intervals: List[tuple[float, float]] = []
        self._metrics_lock = threading.Lock()

    def start(self) -> None:
        if not self._started:
            self._thread.start()
            self._started = True

    def submit(
        self,
        context: ExecutionContext,
        callback: Callable[[ExecutionContext], Any],
    ) -> Future:
        if self._closed:
            raise RuntimeError(f"{self.role.value} worker is closed")
        self.start()
        future: Future = Future()
        self._tasks.put(_RoleTask(context, callback, future))
        return future

    def _loop(self) -> None:
        while True:
            task = self._tasks.get()
            if task is None:
                return
            if not task.future.set_running_or_notify_cancel():
                continue
            started = time.perf_counter()
            try:
                with self._activate(task.context):
                    result = task.callback(task.context)
            except BaseException as exc:
                self.last_error = repr(exc)
                task.future.set_exception(exc)
            else:
                task.future.set_result(result)
            finally:
                finished = time.perf_counter()
                with self._metrics_lock:
                    self.busy_time_s += finished - started
                    self.task_count += 1
                    self._intervals = (self._intervals + [(started, finished)])[-1024:]

    def close(self, timeout_s: float = 5.0) -> None:
        if self._closed:
            return
        self._closed = True
        if self._started:
            self._tasks.put(None)
            self._thread.join(timeout_s)
            if self._thread.is_alive():
                raise TimeoutError(f"{self.role.value} worker did not stop")

    @property
    def queue_depth(self) -> int:
        return self._tasks.qsize()

    def intervals(self) -> List[tuple[float, float]]:
        with self._metrics_lock:
            return list(self._intervals)


class TrueDualWorkerRuntime:
    """Two host issue loops sharing one whole-phase partition.

    The integration layer supplies role activation (CUDA device/stream and
    thread-local PD-mux role).  A task acquires its role lease before enqueue
    and releases the host lease when its future completes. The integration
    records the CUDA completion event, so partition transitions are rejected
    while either a host issue task or GPU event remains in flight.
    """

    def __init__(
        self,
        sm_counts: Iterable[tuple[int, int]],
        activate: Optional[Callable[[ExecutionContext], ContextManager[Any]]] = None,
    ):
        self.arbiter = SharedGpuArbiter(sm_counts)
        self.prefill = RoleWorkerThread(WorkerRole.PREFILL, activate)
        self.decode = RoleWorkerThread(WorkerRole.DECODE, activate)
        self.coordinator = PhaseCoordinator()
        self._request_lock = threading.RLock()
        self.prefill_waiting: dict[Any, Any] = {}
        self.prefill_active: dict[Any, Any] = {}
        self.decode_ready: dict[Any, Any] = {}
        self.decode_running: dict[Any, Any] = {}
        self.started_at_s = time.perf_counter()
        self._closed = False

    def select_partition(self, stream_index: int) -> None:
        self.arbiter.select_partition(stream_index)

    def record_inflight_event(self, role: WorkerRole, event: Any) -> None:
        self.arbiter.record_inflight_event(role, event)

    @staticmethod
    def _rid(req: Any) -> Any:
        return getattr(req, "rid", id(req))

    def register_waiting(self, reqs: Iterable[Any]) -> None:
        with self._request_lock:
            for req in reqs:
                rid = self._rid(req)
                if rid not in self.prefill_active:
                    self.prefill_waiting[rid] = req
                self.coordinator.register_waiting(req)

    def start_prefill(self, reqs: Iterable[Any]) -> None:
        with self._request_lock:
            reqs = list(reqs)
            for req in reqs:
                rid = self._rid(req)
                self.prefill_waiting.pop(rid, None)
                self.prefill_active[rid] = req
            self.coordinator.start_prefill(reqs)

    def complete_prefill(self, reqs: Iterable[Any]) -> None:
        with self._request_lock:
            reqs = list(reqs)
            for req in reqs:
                rid = self._rid(req)
                self.prefill_active.pop(rid, None)
                self.decode_ready[rid] = req
            self.coordinator.complete_prefill(reqs)

    def start_decode(self, reqs: Iterable[Any]) -> None:
        with self._request_lock:
            reqs = list(reqs)
            for req in reqs:
                rid = self._rid(req)
                self.decode_ready.pop(rid, None)
                self.decode_running[rid] = req
            self.coordinator.start_decode(reqs)

    def complete_requests(self, reqs: Iterable[Any]) -> None:
        with self._request_lock:
            reqs = list(reqs)
            for req in reqs:
                rid = self._rid(req)
                self.prefill_waiting.pop(rid, None)
                self.prefill_active.pop(rid, None)
                self.decode_ready.pop(rid, None)
                self.decode_running.pop(rid, None)
            self.coordinator.complete(reqs)

    def prune_to_live(self, reqs: Iterable[Any]) -> None:
        reqs = list(reqs)
        live = {self._rid(req) for req in reqs}
        with self._request_lock:
            for owned in (
                self.prefill_waiting,
                self.prefill_active,
                self.decode_ready,
                self.decode_running,
            ):
                for rid in tuple(owned):
                    if rid not in live:
                        owned.pop(rid, None)
            self.coordinator.prune(reqs)

    def submit(
        self,
        role: WorkerRole,
        callback: Callable[[ExecutionContext], Any],
        stream: Any = None,
        decode_backend: Any = None,
    ) -> Future:
        if self._closed:
            raise RuntimeError("true dual-worker runtime is closed")
        lease = self.arbiter.acquire(role, strict=True)
        context = ExecutionContext(
            role=role,
            stream_index=lease.stream_index,
            prefill_sms=lease.prefill_sms,
            decode_sms=lease.decode_sms,
            generation=lease.generation,
            stream=stream,
            decode_backend=decode_backend,
        )
        worker = self.prefill if role is WorkerRole.PREFILL else self.decode
        try:
            future = worker.submit(context, callback)
        except BaseException:
            self.arbiter.release(role)
            raise
        future.add_done_callback(lambda _: self.arbiter.release(role))
        return future

    def close(self, timeout_s: float = 5.0) -> None:
        if self._closed:
            return
        self.prefill.close(timeout_s)
        self.decode.close(timeout_s)
        self._closed = True

    def metrics(self) -> dict[str, Any]:
        pending_events = 0
        for event in self.arbiter.inflight_events.values():
            try:
                pending_events += int(not event.query())
            except AttributeError:
                pending_events += 1
        elapsed = max(1e-9, time.perf_counter() - self.started_at_s)
        prefill_intervals = self.prefill.intervals()
        decode_intervals = self.decode.intervals()
        overlap_s = 0.0
        i = j = 0
        while i < len(prefill_intervals) and j < len(decode_intervals):
            p_start, p_end = prefill_intervals[i]
            d_start, d_end = decode_intervals[j]
            overlap_s += max(0.0, min(p_end, d_end) - max(p_start, d_start))
            if p_end <= d_end:
                i += 1
            else:
                j += 1
        with self._request_lock:
            request_counts = {
                "owned_prefill_waiting": len(self.prefill_waiting),
                "owned_prefill_active": len(self.prefill_active),
                "owned_decode_ready": len(self.decode_ready),
                "owned_decode_running": len(self.decode_running),
            }
        return {
            "prefill_host_queue_depth": self.prefill.queue_depth,
            "decode_host_queue_depth": self.decode.queue_depth,
            "prefill_host_tasks": self.prefill.task_count,
            "decode_host_tasks": self.decode.task_count,
            "prefill_host_busy_ms": self.prefill.busy_time_s * 1000.0,
            "decode_host_busy_ms": self.decode.busy_time_s * 1000.0,
            "prefill_host_idle_ratio": max(
                0.0, 1.0 - self.prefill.busy_time_s / elapsed
            ),
            "decode_host_idle_ratio": max(
                0.0, 1.0 - self.decode.busy_time_s / elapsed
            ),
            "host_worker_overlap_ratio": min(1.0, overlap_s / elapsed),
            "active_leases": len(self.arbiter.active_roles),
            "pending_gpu_events": pending_events,
            "partition_generation": self.arbiter.generation,
            **request_counts,
        }


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
