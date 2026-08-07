"""Direct head-of-line-blocking (HOLB) instrumentation for **every** serving arm.

WHAT THIS MEASURES
------------------
The GPU-timeline milliseconds during which *decode work exists but no decode
forward is executing*::

    T_window = decode_fw_ms + stall_ms + stall_ambiguous_ms

The same three buckets partition the same wall-clock window in all four Gate-2
arms (``plain`` / ``plain+aux`` / ``plain+chunk512`` / ``agnostic``), because
every GPU forward in every arm goes through ``Scheduler.run_batch()``.  Nothing
in this module knows which arm it is running under.

WHY GPU EVENTS AND NOT THE HOST CLOCK
-------------------------------------
``event_loop_overlap`` (the fused arms A1/A3) decouples host time from GPU
time: the wall clock of loop iteration *i* corresponds to awaiting batch *i-1*,
not to executing batch *i*.  A host-timeline definition would therefore make
decode look concurrent with prefill on the overlap arms even though the GPU
serialises them -- i.e. it would understate blocking exactly on the arms whose
blocking is the question.  CUDA events are recorded in stream order and time
what the GPU actually did, so the definition means the same thing in all arms.

WHAT IS *NOT* A VALID PRIMARY
-----------------------------
``blocking_fw_ms`` (time spent inside non-decode forwards while decode was
pending) is **structurally zero for the pdmux arm**, whose prefill runs on a
second stream and can never occupy the decode timeline.  It is an identity, not
evidence.  It is reported as a *diagnostic only*.  The primary is ``stall_ms``
(the union: decode pending, no decode forward running), which is non-trivially
non-zero in every arm -- for pdmux it captures the green-context drains and the
per-iteration host overhead, i.e. pdmux's own known costs.

SCOPE OF THE ENV NAMESPACE
--------------------------
``PDMUX_*`` is this project's environment namespace, **not** a scope
restriction.  This probe is active for fused arms too.  Default is OFF: with
``PDMUX_HOLB_PATH`` unset, ``maybe_create_holb_probe`` returns ``None`` and the
call sites collapse to two ``is not None`` checks per forward -- no CUDA calls,
no allocation, no clock reads, no I/O.

See ``results/p1_gates/gate2/DIRECT_BLOCKING_DESIGN.md`` for the full
definition, the arm-symmetry argument, the known biases and the honest verdict
on whether this can serve as a Gate-2 primary.
"""

from __future__ import annotations

import atexit
import logging
import math
import os
import time
import traceback
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional

import torch

from sglang.srt.multiplex.telemetry import AsyncJsonlTelemetry

logger = logging.getLogger(__name__)

HOLB_SCHEMA = "pdmux.holb/v1"

# gap classification labels (no thresholds: every label is an observed state)
GAP_FIRST = "first"          # no preceding decode forward in this process
GAP_STRICT = "strict"        # every observation inside the gap had pending > 0
GAP_AMBIGUOUS = "ambiguous"  # some observation inside the gap had pending == 0
GAP_INVALID = "invalid"      # elapsed_time unusable (negative / NaN / raised)

_BUBBLE_TOL_MS = 0.05  # slack for float32 cudaEventElapsedTime when checking gap >= same_stream


def advances_decode(forward_mode: Any) -> bool:
    """Does this forward move at least one decode request forward by a token?

    DECODE -> yes.  MIXED -> yes (it contains decode requests; see
    DIRECT_BLOCKING_DESIGN.md section 3 for the pre-registered handling and the
    bias it introduces).  EXTEND / SPLIT_PREFILL / IDLE / everything else -> no.
    """
    return bool(forward_mode.is_decode()) or bool(forward_mode.is_mixed())


@dataclass
class SpanRecord:
    """A resolved forward span: CUDA already read out, pure data from here on."""

    kind: str  # "decode" | "other"
    mode: str  # ForwardMode.name
    pending: int  # decode-pending requests observed when the forward was issued
    fw_bs: int  # batch_size() of the forward itself
    stream_key: int
    dur_ms: float
    host_ts: float
    seq: int
    gap_ms: Optional[float] = None  # decode spans only: prev decode end -> this start


class HolbAccountant:
    """Pure-Python accounting for the HOLB definition.  No CUDA, no I/O.

    Fed with already-resolved spans in *program order* (the order the scheduler
    issued them).  Emits one gap event per decode forward.
    """

    def __init__(self, episode_cap: int = 200_000):
        self.episode_cap = int(episode_cap)
        self.episodes: List[float] = []  # strict gap durations, capped
        self.episodes_truncated = 0
        self.mode_counts: Counter = Counter()
        self._open: List[SpanRecord] = []
        self._seen_decode = False
        self.c: Dict[str, float] = {
            "n_decode_fw": 0,
            "n_other_fw": 0,
            "n_mixed": 0,
            "decode_fw_ms": 0.0,
            "mixed_fw_ms": 0.0,
            "other_fw_ms": 0.0,
            # primary
            "stall_ms": 0.0,
            "stall_req_ms": 0.0,
            "n_stall_episodes": 0,
            # excluded-but-reported
            "stall_ambiguous_ms": 0.0,
            "n_ambiguous_gaps": 0,
            # diagnostics (NOT decision quantities)
            "blocking_fw_ms": 0.0,
            "same_stream_fw_ms": 0.0,
            "other_stream_fw_ms": 0.0,
            "bubble_ms": 0.0,
            # health
            "n_invalid_gaps": 0,
            "n_inconsistent_gaps": 0,
            "n_pre_first_decode_fw": 0,
            "n_trailing_other_fw": 0,
        }
        self.first_decode_host_ts: Optional[float] = None
        self.last_decode_host_ts: Optional[float] = None

    # -- feed ---------------------------------------------------------------
    def record(self, span: SpanRecord) -> Optional[Dict[str, Any]]:
        self.mode_counts[span.mode] += 1
        if span.kind != "decode":
            self._open.append(span)
            self.c["n_other_fw"] += 1
            self.c["other_fw_ms"] += span.dur_ms
            if span.pending > 0:
                # The design-doc "recommended definition", kept verbatim as a
                # diagnostic.  Identically zero for pdmux -> never a primary.
                self.c["blocking_fw_ms"] += span.dur_ms
            return None

        self.c["n_decode_fw"] += 1
        self.c["decode_fw_ms"] += span.dur_ms
        if span.mode == "MIXED":
            self.c["n_mixed"] += 1
            self.c["mixed_fw_ms"] += span.dur_ms
        if self.first_decode_host_ts is None:
            self.first_decode_host_ts = span.host_ts
        self.last_decode_host_ts = span.host_ts

        same_ms = 0.0
        other_ms = 0.0
        pend_min = span.pending
        for s in self._open:
            if s.stream_key == span.stream_key:
                same_ms += s.dur_ms
            else:
                other_ms += s.dur_ms
            if s.pending < pend_min:
                pend_min = s.pending

        n_other = len(self._open)
        gap_ms = span.gap_ms
        bubble_ms = None
        inconsistent = False

        if not self._seen_decode:
            gap_class = GAP_FIRST
            self.c["n_pre_first_decode_fw"] += n_other
        elif gap_ms is None or not math.isfinite(gap_ms) or gap_ms < 0.0:
            gap_class = GAP_INVALID
            self.c["n_invalid_gaps"] += 1
        else:
            # Only same-stream forwards can occupy the decode timeline; a
            # different stream runs concurrently.  This rule is data-driven
            # (it reads the recorded stream key), not arm-specific.
            bubble_ms = gap_ms - same_ms
            if bubble_ms < -_BUBBLE_TOL_MS:
                inconsistent = True
                self.c["n_inconsistent_gaps"] += 1
            self.c["same_stream_fw_ms"] += min(same_ms, gap_ms)
            self.c["other_stream_fw_ms"] += other_ms
            self.c["bubble_ms"] += max(0.0, bubble_ms)
            if pend_min > 0:
                gap_class = GAP_STRICT
                self.c["stall_ms"] += gap_ms
                self.c["stall_req_ms"] += gap_ms * span.pending
                self.c["n_stall_episodes"] += 1
                if len(self.episodes) < self.episode_cap:
                    self.episodes.append(gap_ms)
                else:
                    self.episodes_truncated += 1
            else:
                gap_class = GAP_AMBIGUOUS
                self.c["stall_ambiguous_ms"] += gap_ms
                self.c["n_ambiguous_gaps"] += 1

        self._seen_decode = True
        self._open = []

        return {
            "seq": span.seq,
            "host_ts": span.host_ts,
            "gap_ms": gap_ms,
            "gap_class": gap_class,
            "decode_dur_ms": span.dur_ms,
            "decode_bs": span.pending,
            "decode_fw_bs": span.fw_bs,
            "decode_mode": span.mode,
            "stream_key": span.stream_key,
            "n_other_fw": n_other,
            "same_stream_fw_ms": same_ms,
            "other_stream_fw_ms": other_ms,
            "bubble_ms": bubble_ms,
            "pend_min_in_gap": pend_min,
            "inconsistent": inconsistent,
        }

    def finalize(self) -> None:
        """Account for non-decode forwards issued after the last decode forward."""
        self.c["n_trailing_other_fw"] += len(self._open)
        self._open = []

    # -- read ---------------------------------------------------------------
    @staticmethod
    def _pct(sorted_vals: List[float], frac: float) -> Optional[float]:
        if not sorted_vals:
            return None
        if len(sorted_vals) == 1:
            return sorted_vals[0]
        idx = frac * (len(sorted_vals) - 1)
        lo = int(math.floor(idx))
        hi = int(math.ceil(idx))
        if lo == hi:
            return sorted_vals[lo]
        return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (idx - lo)

    def summary(self) -> Dict[str, Any]:
        out: Dict[str, Any] = dict(self.c)
        eps = sorted(self.episodes)
        out["episode_p50_ms"] = self._pct(eps, 0.50)
        out["episode_p95_ms"] = self._pct(eps, 0.95)
        out["episode_p99_ms"] = self._pct(eps, 0.99)
        out["episode_max_ms"] = eps[-1] if eps else None
        out["episodes_truncated"] = self.episodes_truncated
        # GPU-timeline window reconstructed from the chain, and the host span
        # over the same forwards.  Their ratio is the instrumentation self-check
        # (see DIRECT_BLOCKING_DESIGN.md section 4.2).
        timeline_ms = (
            self.c["decode_fw_ms"] + self.c["stall_ms"] + self.c["stall_ambiguous_ms"]
        )
        out["timeline_ms"] = timeline_ms
        if self.first_decode_host_ts is not None and self.last_decode_host_ts is not None:
            host_span_ms = (self.last_decode_host_ts - self.first_decode_host_ts) * 1000.0
        else:
            host_span_ms = None
        out["host_span_ms"] = host_span_ms
        out["timeline_over_host"] = (
            timeline_ms / host_span_ms if host_span_ms and host_span_ms > 0 else None
        )
        out["stall_frac"] = (
            (self.c["stall_ms"] / timeline_ms) if timeline_ms > 0 else None
        )
        out["mode_counts"] = dict(self.mode_counts)
        return out


# ---------------------------------------------------------------------------
# CUDA side
# ---------------------------------------------------------------------------


@dataclass
class _PendingSpan:
    kind: str
    mode: str
    pending: int
    fw_bs: int
    stream_key: int
    stream: Any
    host_ts: float
    seq: int
    start_event: Any
    end_event: Any = None


def _default_event_factory():  # pragma: no cover - trivial, needs a GPU
    return torch.cuda.Event(enable_timing=True)


class HolbProbe:
    """Records a CUDA event pair per forward and drains them without syncing.

    Same lazy-readout pattern as upstream ``sglang/srt/utils/device_timer.py``
    (``DeviceTimer._report``): only events whose ``query()`` is already True are
    read, so the scheduler thread never blocks on the GPU.
    """

    def __init__(
        self,
        path: Optional[Path],
        run_id: str = "unlabeled",
        workload_id: str = "unlabeled",
        *,
        event_factory: Callable[[], Any] = _default_event_factory,
        telemetry: Optional[AsyncJsonlTelemetry] = None,
        max_inflight: int = 4096,
        drain_budget: int = 64,
        summary_every_s: float = 5.0,
        emit_gaps: bool = True,
        tags: Optional[Dict[str, Any]] = None,
    ):
        self.acct = HolbAccountant()
        self._event_factory = event_factory
        self._pool: List[Any] = []
        self._inflight: Deque[_PendingSpan] = deque()
        self._held_decode_end: Any = None  # kept until the next decode resolves
        self._seq = 0
        self.max_inflight = int(max_inflight)
        self.drain_budget = int(drain_budget)
        self.summary_every_s = float(summary_every_s)
        self.emit_gaps = bool(emit_gaps)
        self.tags = dict(tags or {})
        self.dropped_spans = 0
        self.n_errors = 0
        self._error_logged = False
        self._closed = False
        self._last_summary_ts = time.time()
        self.telemetry = (
            telemetry
            if telemetry is not None
            else AsyncJsonlTelemetry(path, run_id=run_id, workload_id=workload_id)
        )
        self.telemetry.emit("holb_open", "startup", schema_holb=HOLB_SCHEMA, **self.tags)

    # -- event pool ---------------------------------------------------------
    def _acquire_event(self) -> Any:
        if self._pool:
            return self._pool.pop()
        return self._event_factory()

    def _release_event(self, ev: Any) -> None:
        if ev is not None and len(self._pool) < 256:
            self._pool.append(ev)

    # -- scheduler-facing API ----------------------------------------------
    def begin(self, batch: Any, running_batch: Any) -> Optional[_PendingSpan]:
        if self._closed:
            return None
        try:
            if len(self._inflight) >= self.max_inflight:
                self.dropped_spans += 1
                return None
            mode = batch.forward_mode
            stream = torch.cuda.current_stream()
            start = self._acquire_event()
            start.record(stream)
            span = _PendingSpan(
                kind="decode" if advances_decode(mode) else "other",
                mode=mode.name,
                pending=_pending_decode_count(running_batch),
                fw_bs=len(getattr(batch, "reqs", ()) or ()),
                stream_key=int(stream.cuda_stream),
                stream=stream,
                host_ts=time.time(),
                seq=self._seq,
                start_event=start,
            )
            self._seq += 1
            return span
        except Exception:  # instrumentation must never kill a serving run
            self._on_error()
            return None

    def end(self, span: Optional[_PendingSpan]) -> None:
        if span is None or self._closed:
            return
        try:
            end_event = self._acquire_event()
            end_event.record(span.stream)
            span.end_event = end_event
            self._inflight.append(span)
            self._drain()
        except Exception:
            self._on_error()

    # -- drain --------------------------------------------------------------
    def _drain(self, force: bool = False) -> None:
        # `force` only lifts the per-call budget.  It deliberately does NOT
        # synchronize: this path also runs from atexit, where the CUDA context
        # may already be tearing down and a blocking wait is not recoverable.
        # Whatever is still in flight is reported as `n_unresolved_at_close`;
        # the periodic summary bounds how much that can hide.
        budget = len(self._inflight) if force else self.drain_budget
        processed = 0
        while self._inflight and processed < budget:
            span = self._inflight[0]
            if span.end_event is None:
                break
            if not span.end_event.query():
                break
            self._inflight.popleft()
            processed += 1
            self._resolve(span)
        now = time.time()
        if now - self._last_summary_ts >= self.summary_every_s:
            self._last_summary_ts = now
            self.emit_summary("running")

    def _resolve(self, span: _PendingSpan) -> None:
        gap_ms: Optional[float] = None
        if span.kind == "decode" and self._held_decode_end is not None:
            try:
                gap_ms = float(self._held_decode_end.elapsed_time(span.start_event))
            except Exception:
                gap_ms = None
                self._on_error()
        try:
            dur_ms = float(span.start_event.elapsed_time(span.end_event))
        except Exception:
            self._on_error()
            self._release_event(span.start_event)
            self._release_event(span.end_event)
            if span.kind == "decode":
                # Break the chain rather than silently measure the next gap
                # from a two-steps-back anchor: the next decode gap becomes
                # GAP_INVALID and is counted, not absorbed.
                self._release_event(self._held_decode_end)
                self._held_decode_end = None
            return

        rec = SpanRecord(
            kind=span.kind,
            mode=span.mode,
            pending=span.pending,
            fw_bs=span.fw_bs,
            stream_key=span.stream_key,
            dur_ms=dur_ms,
            host_ts=span.host_ts,
            seq=span.seq,
            gap_ms=gap_ms,
        )
        event = self.acct.record(rec)

        self._release_event(span.start_event)
        if span.kind == "decode":
            self._release_event(self._held_decode_end)
            self._held_decode_end = span.end_event
        else:
            self._release_event(span.end_event)

        if event is not None and self.emit_gaps:
            self.telemetry.emit("holb_gap", "measure", **event)

    # -- output -------------------------------------------------------------
    def emit_summary(self, phase: str) -> Dict[str, Any]:
        s = self.acct.summary()
        s["dropped_spans"] = self.dropped_spans
        s["n_errors"] = self.n_errors
        s["n_inflight"] = len(self._inflight)
        if phase == "shutdown":
            s["n_unresolved_at_close"] = len(self._inflight)
        s.update(self.tags)
        self.telemetry.emit("holb_summary", phase, **s)
        return s

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._drain(force=True)
            self.acct.finalize()
            self.emit_summary("shutdown")
        except Exception:
            self._on_error()
        finally:
            self._closed = True
            try:
                self.telemetry.close()
            except Exception:
                pass

    # -- errors -------------------------------------------------------------
    def _on_error(self) -> None:
        self.n_errors += 1
        if not self._error_logged:
            self._error_logged = True
            logger.error("HOLB probe error (instrumentation only):\n%s", traceback.format_exc())


def _pending_decode_count(running_batch: Any) -> int:
    """Requests that finished prefill and are waiting for their next token.

    Same field (`Scheduler.running_batch`) in all four arms.  Requests that have
    not been prefilled yet (waiting_queue / split_prefill_batch / chunked_req)
    are deliberately excluded: mixing them would fold admission queueing into
    head-of-line blocking.
    """
    if running_batch is None:
        return 0
    if getattr(running_batch, "is_prefill_only", False):
        return 0
    reqs = getattr(running_batch, "reqs", None)
    if not reqs:
        return 0
    n = 0
    for r in reqs:
        if not r.finished():
            n += 1
    return n


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


def maybe_create_holb_probe(scheduler: Any) -> Optional[HolbProbe]:
    """Return a probe iff ``PDMUX_HOLB_PATH`` is set; otherwise ``None``.

    Default OFF.  When OFF this function is the only code that runs, once, at
    scheduler construction.
    """
    path_str = os.environ.get("PDMUX_HOLB_PATH", "").strip()
    if not path_str:
        return None
    if os.environ.get("PDMUX_TRUE_DUAL_WORKER", "0") in ("1", "true", "True"):
        # The probe keeps an unsynchronised FIFO (event pool, in-flight deque,
        # decode chain anchor) and assumes `run_batch` is called from one
        # thread.  True dual-worker issues forwards from two host threads, so
        # the ordering the accountant relies on would silently be wrong.
        # Fail fast rather than emit a plausible-looking wrong number.
        raise RuntimeError(
            "PDMUX_HOLB_PATH is not supported together with "
            "PDMUX_TRUE_DUAL_WORKER (the probe assumes single-threaded forward "
            "issue). Unset one of them."
        )
    path = Path(path_str)
    tp_rank = int(getattr(scheduler, "tp_rank", 0) or 0)
    tp_size = int(getattr(scheduler, "tp_size", 1) or 1)
    if tp_size > 1:
        path = path.with_name(f"{path.stem}.tp{tp_rank}{path.suffix}")
    enable_pdmux = bool(getattr(scheduler, "enable_pdmux", False))
    enable_overlap = bool(getattr(scheduler, "enable_overlap", False))
    server_args = getattr(scheduler, "server_args", None)
    tags = {
        "arm_enable_pdmux": enable_pdmux,
        "arm_enable_overlap": enable_overlap,
        "arm_chunked_prefill_size": getattr(server_args, "chunked_prefill_size", None),
        "arm_enable_mixed_chunk": getattr(server_args, "enable_mixed_chunk", None),
        "tp_rank": tp_rank,
    }
    probe = HolbProbe(
        path,
        run_id=os.environ.get(
            "PDMUX_HOLB_RUN_ID", os.environ.get("PDMUX_RUN_ID", "unlabeled")
        ),
        workload_id=os.environ.get(
            "PDMUX_HOLB_WORKLOAD_ID", os.environ.get("PDMUX_WORKLOAD_ID", "unlabeled")
        ),
        max_inflight=_env_int("PDMUX_HOLB_MAX_INFLIGHT", 4096),
        drain_budget=_env_int("PDMUX_HOLB_DRAIN_BUDGET", 64),
        summary_every_s=_env_float("PDMUX_HOLB_SUMMARY_EVERY_S", 5.0),
        emit_gaps=os.environ.get("PDMUX_HOLB_EMIT_GAPS", "1") not in ("0", "false", "False"),
        tags=tags,
    )
    atexit.register(probe.close)
    logger.info(
        "HOLB probe ENABLED (all arms; pdmux=%s overlap=%s chunked_prefill_size=%s) -> %s",
        enable_pdmux,
        enable_overlap,
        tags["arm_chunked_prefill_size"],
        path,
    )
    return probe
