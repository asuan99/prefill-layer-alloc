"""I-4 decode forward GPU time + I-6 cudagraph padding probe (SPEC_common_measurement_v0 sec 9).

WHAT THIS RECORDS
-----------------
One ``decode_forward`` record per DECODE forward (and, labelled separately, per
MIXED forward -- fused ``--enable-mixed-chunk`` puts the decode tokens inside an
extend batch) of the scheduler's target ``ModelRunner``:

  I-4  ``decode_fwd_gpu_ms``   GPU time of ``ModelRunner.forward`` measured by two
                               CUDA events recorded on the stream the forward is
                               issued on (PD-mux: the green-context decode stream;
                               fused: the scheduler's forward stream).  A NEW field;
                               the PD-mux ``decode_iteration`` host stamps
                               ``t_launch``/``t_sync`` (PDMUX_PHASE_EVENTS) are not
                               touched and keep their meaning.
       ``decode_fwd_host_ms``  host duration of the same call, same bracket.  Kept
                               on purpose: gate (c) evaluates the "I-4 uses host
                               time" mutant by substituting this field (the mutant's
                               output is exactly this number, see the gate design).
  I-6  ``bs_real``             ``forward_batch.batch_size``
       ``bs_padded``           graph hit: the captured batch size replayed
                               (``CudaGraphRunner.bs``, set by ``replay_prepare``);
                               eager: ``bs_real`` (no padding happens).
       ``graph``               ``hit`` | ``eager`` = ``ModelRunnerOutput.can_run_graph``
       ``bs_pad_expected``     independent prediction from the boot capture list
                               (``bisect_left``), ``pad_check`` = ok | mismatch | na
       ``eager_reason``        graph_runner_absent | bs_over_max | mode_not_capturable
                               | can_run_false

ARM SYMMETRY (audit MAJOR 7, PREREG rev2 sec 1-7)
-------------------------------------------------
The hook is ONE wrapper on the target ``ModelRunner`` instance, installed from
``run_scheduler_process`` (the common entry of ``event_loop_normal``,
``event_loop_overlap`` and ``event_loop_pdmux``).  Every loop reaches the model
through ``TpModelWorker.forward_batch_generation`` -> ``self.model_runner.forward``
(PD-mux decode: mixin ``run_batch(self.running_batch)`` -> the same call), so
vanilla fused (Nano and Llama) and PD-mux arms are recorded by the same code.
The ``decode_iteration`` events, by contrast, exist only in the PD-mux mixin.

LOCKSTEP NEUTRALITY (why the CUDA event placement does not change sync)
-----------------------------------------------------------------------
* Both events are recorded on the CURRENT stream at call time, i.e. the stream
  the forward's own kernels go to.  ``cudaEventRecord`` on that stream adds no
  cross-stream dependency: no ``wait_event``/``wait_stream`` is issued, so the
  stream DAG the PD-mux loop builds (and its ``decode_stream.synchronize()`` /
  ``prefill_exe_done.query()`` points) is unchanged.
* The probe NEVER blocks: no ``Event.synchronize``, ``Stream.synchronize`` or
  ``torch.cuda.synchronize``.  Completion is polled with ``Event.query()``
  (``cudaEventQuery``, non-blocking) at the entry of the NEXT recorded forward,
  at window flushes and at close; ``elapsed_time`` is read only after the end
  event reported complete.  Unresolved pairs stay pending (bounded; overflow is
  counted, never waited for).
* Graph capture is never instrumented: if the current stream is capturing, the
  call passes through and is counted (``n_skipped_capturing``).
* The hook is installed AFTER ``Scheduler.__init__`` (cudagraph capture and
  warm-up are finished), so capture shapes, the capture list and the eager/graph
  decision are untouched -- the probe reads ``can_run_graph``, it does not decide it.
The observer cost is NOT claimed to be zero (two event records + one query + one
queued dict per decode forward); gate (d) measures it (paired TOST).

DEFAULT OFF
-----------
Without ``PDMUX_DECODE_PROBE_PATH`` ``maybe_install_decode_probe`` returns None
and installs nothing: ``ModelRunner.forward`` is the pristine method.
``PDMUX_DECODE_PROBE_GPU_EVENTS=0`` keeps I-6 and drops the CUDA events (I-4
fields null).  ``PDMUX_DECODE_PROBE_WINDOW_S`` (default 1.0) sets the summary
window.  Records go through ``AsyncJsonlTelemetry`` (no file I/O on the
scheduler/worker thread).

The pure part (above ``# ---- engine glue``) imports no torch / sglang, so CPU
tests load this file loose.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from bisect import bisect_left
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

DECODE_SCHEMA = "pdmux.decode_probe/v1"
DECODE_SCHEMA_VERSION = 1

MODE_DECODE = "decode"
MODE_MIXED = "mixed"
GRAPH_HIT = "hit"
GRAPH_EAGER = "eager"
EAGER_NO_RUNNER = "graph_runner_absent"
EAGER_OVER_MAX = "bs_over_max"
EAGER_MODE = "mode_not_capturable"
EAGER_OTHER = "can_run_false"
PAD_OK = "ok"
PAD_MISMATCH = "mismatch"
PAD_NA = "na"

I4_FIELD = "decode_fwd_gpu_ms"
I4_HOST_FIELD = "decode_fwd_host_ms"
MAX_PENDING_DEFAULT = 4096


def _truth(obj: Any, name: str) -> bool:
    fn = getattr(obj, name, None)
    try:
        return bool(fn()) if callable(fn) else False
    except Exception:
        return False


def forward_mode_kind(forward_mode: Any) -> Optional[str]:
    """``decode`` | ``mixed`` | None (not recorded: extend, split_prefill, idle, ...).

    ``ForwardMode.MIXED.is_decode()`` is False in SGLang v0.5.10
    (forward_batch_info.py), so the order of the two tests does not matter; it is
    kept decode-first for readability.
    """
    if forward_mode is None:
        return None
    if _truth(forward_mode, "is_decode"):
        return MODE_DECODE
    if _truth(forward_mode, "is_mixed"):
        return MODE_MIXED
    return None


def mode_name(forward_mode: Any) -> str:
    return str(getattr(forward_mode, "name", forward_mode)).lower()


def expected_padded_bs(capture_bs: Sequence[int], raw_bs: int, disable_padding: bool) -> Optional[int]:
    """The batch size ``CudaGraphRunner.replay_prepare`` pads ``raw_bs`` to.

    Mirrors cuda_graph_runner.py ``index = bisect.bisect_left(self.capture_bs, raw_bs);
    bs = self.capture_bs[index]`` (non-DP-attention branch) and ``can_run``'s
    ``disable_padding`` rule (exact key only).  None = no graph can serve it.
    """
    cb = sorted(int(b) for b in (capture_bs or ()))
    raw = int(raw_bs)
    if not cb or raw <= 0:
        return None
    if disable_padding:
        return raw if raw in cb else None
    i = bisect_left(cb, raw)
    return cb[i] if i < len(cb) else None


def graph_fields(
    kind: str,
    can_run_graph: bool,
    bs_real: int,
    runner_present: bool,
    runner_bs: Optional[int],
    capture_bs: Sequence[int],
    disable_padding: bool,
) -> Dict[str, Any]:
    """I-6 fields of one forward.  ``can_run_graph`` is the engine's own decision."""
    if can_run_graph:
        exp = expected_padded_bs(capture_bs, bs_real, disable_padding)
        padded = int(runner_bs) if runner_bs is not None else None
        ok = padded is not None and exp is not None and padded == exp
        return {
            "graph": GRAPH_HIT,
            "bs_padded": padded,
            "bs_pad_expected": exp,
            "pad_check": PAD_OK if ok else PAD_MISMATCH,
            "eager_reason": None,
        }
    if kind != MODE_DECODE:
        reason = EAGER_MODE
    elif not runner_present:
        reason = EAGER_NO_RUNNER
    elif capture_bs and int(bs_real) > max(int(b) for b in capture_bs):
        reason = EAGER_OVER_MAX
    else:
        reason = EAGER_OTHER
    return {
        "graph": GRAPH_EAGER,
        "bs_padded": int(bs_real),
        "bs_pad_expected": int(bs_real),
        "pad_check": PAD_NA,
        "eager_reason": reason,
    }


@dataclass
class PendingForward:
    fields: Dict[str, Any]
    start: Any = None
    end: Any = None


def drain_ready(pending: Deque[PendingForward]) -> List[Tuple[PendingForward, float]]:
    """Pop completed pairs in FIFO order WITHOUT blocking.

    ``end.query()`` is the only completion test (non-blocking); ``elapsed_time``
    is read only after it returned True.  Stops at the first incomplete head:
    recorded forwards of one probe share a stream order in every loop this
    project runs (one decode issuer at a time), so a later pair cannot be
    complete while an earlier one is not, and stopping there costs nothing.
    """
    out = []
    while pending:
        head = pending[0]
        if not head.end.query():
            break
        pending.popleft()
        out.append((head, float(head.start.elapsed_time(head.end))))
    return out


@dataclass
class DecodeAccountant:
    t_start: float = 0.0
    n_by_kind: Counter = field(default_factory=Counter)
    n_other_modes: Counter = field(default_factory=Counter)
    n_graph: Counter = field(default_factory=Counter)
    n_eager_reason: Counter = field(default_factory=Counter)
    n_pad_mismatch: int = 0
    pad_tokens: int = 0
    n_resolved: int = 0
    n_dropped_pending: int = 0
    n_skipped_capturing: int = 0
    n_probe_errors: int = 0

    def reset(self, t: float) -> None:
        self.__init__(t_start=t)  # type: ignore[misc]

    def window_record(self, t_end: float, n_pending: int) -> Dict[str, Any]:
        return {
            "record_schema": DECODE_SCHEMA,
            "t_host_start": self.t_start,
            "t_host_end": float(t_end),
            "n_decode": self.n_by_kind.get(MODE_DECODE, 0),
            "n_mixed": self.n_by_kind.get(MODE_MIXED, 0),
            "n_other_modes": dict(sorted(self.n_other_modes.items())),
            "n_graph_hit": self.n_graph.get(GRAPH_HIT, 0),
            "n_graph_eager": self.n_graph.get(GRAPH_EAGER, 0),
            "eager_reason": dict(sorted(self.n_eager_reason.items())),
            "n_pad_mismatch": self.n_pad_mismatch,
            "pad_tokens": self.pad_tokens,
            "n_resolved": self.n_resolved,
            "n_pending": int(n_pending),
            "n_dropped_pending": self.n_dropped_pending,
            "n_skipped_capturing": self.n_skipped_capturing,
            "n_probe_errors": self.n_probe_errors,
        }


def validate_record(rec: Dict[str, Any]) -> List[str]:
    """Schema gate (d): consumers refuse records they do not know."""
    errs = []
    if rec.get("record_schema") != DECODE_SCHEMA:
        errs.append(f"record_schema {rec.get('record_schema')!r} != {DECODE_SCHEMA!r}")
    if rec.get("event") == "decode_forward":
        for k in ("mode", "bs_real", "bs_padded", "graph", "pad_check", I4_FIELD, I4_HOST_FIELD):
            if k not in rec:
                errs.append(f"decode_forward without {k}")
        if rec.get("graph") == GRAPH_EAGER and rec.get("bs_padded") != rec.get("bs_real"):
            errs.append("eager forward with bs_padded != bs_real")
        if rec.get("graph") == GRAPH_HIT and rec.get("bs_padded") is not None and rec.get("bs_real") is not None:
            if int(rec["bs_padded"]) < int(rec["bs_real"]):
                errs.append("graph hit with bs_padded < bs_real")
    return errs


class DecodeProbe:
    """Instance state.  Thread-safe: true-dual decode tasks call it from a worker
    thread; everything mutable is under ``_lock`` (event records are not)."""

    def __init__(
        self,
        path: Optional[Path],
        run_id: str = "unlabeled",
        workload_id: str = "unlabeled",
        window_s: float = 1.0,
        events_enabled: bool = True,
        boot: Optional[Dict[str, Any]] = None,
        static_fields: Optional[Dict[str, Any]] = None,
        clock: Callable[[], float] = time.perf_counter,
        event_factory: Optional[Callable[[], Any]] = None,
        capturing_fn: Optional[Callable[[], bool]] = None,
        stream_idx_fn: Optional[Callable[[], Optional[int]]] = None,
        max_pending: int = MAX_PENDING_DEFAULT,
        decode_iter_fn: Optional[Callable[[], Optional[int]]] = None,
    ) -> None:
        self.boot = dict(boot or {})
        self.window_s = float(window_s)
        self.events_enabled = bool(events_enabled) and event_factory is not None
        self.clock = clock
        self.event_factory = event_factory
        self.capturing_fn = capturing_fn or (lambda: False)
        self.stream_idx_fn = stream_idx_fn
        self.decode_iter_fn = decode_iter_fn
        self.max_pending = max(1, int(max_pending))
        self.capture_bs = [int(b) for b in (self.boot.get("capture_bs") or [])]
        self.disable_padding = bool(self.boot.get("disable_cuda_graph_padding", False))
        self.acct = DecodeAccountant()
        self.pending: Deque[PendingForward] = deque()
        self._pool: List[Tuple[Any, Any]] = []
        self._lock = threading.Lock()
        self._seq = 0
        self.disabled = False
        self.records: List[Dict[str, Any]] = []  # kept only when path is None (tests)
        self._tel = None
        if path is not None:
            try:
                from sglang.srt.multiplex.telemetry import AsyncJsonlTelemetry
            except ImportError:  # loose-file CPU load
                from telemetry import AsyncJsonlTelemetry  # type: ignore[no-redef]
            self._tel = AsyncJsonlTelemetry(
                path, run_id=run_id, workload_id=workload_id, static_fields=static_fields
            )
        self.acct.reset(self.clock())
        self._emit("decode_probe_boot", schema_version=DECODE_SCHEMA_VERSION,
                   t_realtime=time.time(), t_host=self.clock(),
                   events_enabled=self.events_enabled, **self.boot)

    # -- output
    def _emit(self, event: str, **fields: Any) -> None:
        fields.setdefault("record_schema", DECODE_SCHEMA)
        if self._tel is not None:
            self._tel.emit(event, "decode_probe", **fields)
        else:
            rec = dict(fields)
            rec["event"] = event
            self.records.append(rec)

    def _disable(self, exc: BaseException) -> None:
        if not self.disabled:
            logger.warning("decode probe disabled after failure: %r", exc)
        self.disabled = True

    # -- events
    def _take_pair(self) -> Optional[Tuple[Any, Any]]:
        if not self.events_enabled:
            return None
        with self._lock:
            if self._pool:
                return self._pool.pop()
        return (self.event_factory(), self.event_factory())

    def _drain_locked(self) -> None:
        for item, gpu_ms in drain_ready(self.pending):
            item.fields[I4_FIELD] = gpu_ms
            self.acct.n_resolved += 1
            self._pool.append((item.start, item.end))
            self._emit("decode_forward", **item.fields)

    def poll(self) -> None:
        """Non-blocking resolution of completed pairs (also called by flushes)."""
        with self._lock:
            self._drain_locked()

    # -- the wrapped call
    def around_forward(self, runner: Any, orig: Callable, args: tuple, kwargs: dict) -> Any:
        if self.disabled:
            return orig(*args, **kwargs)
        try:
            fb = args[0] if args else kwargs.get("forward_batch")
            fm = getattr(fb, "forward_mode", None)
            kind = forward_mode_kind(fm)
            if kind is None:
                with self._lock:
                    self.acct.n_other_modes[mode_name(fm)] += 1
                return orig(*args, **kwargs)
            if self.capturing_fn():
                with self._lock:
                    self.acct.n_skipped_capturing += 1
                return orig(*args, **kwargs)
            self.poll()
            pair = self._take_pair()
            bs_real = int(getattr(fb, "batch_size", 0) or 0)
            sidx = self.stream_idx_fn() if self.stream_idx_fn is not None else None
            diter = self.decode_iter_fn() if self.decode_iter_fn is not None else None
        except Exception as exc:  # probe-side failure: never break the forward
            self._disable(exc)
            return orig(*args, **kwargs)

        t0 = self.clock()
        if pair is not None:
            pair[0].record()
        out = orig(*args, **kwargs)
        if pair is not None:
            pair[1].record()
        t1 = self.clock()

        try:
            gr = getattr(runner, "graph_runner", None)
            gf = graph_fields(
                kind,
                bool(getattr(out, "can_run_graph", False)),
                bs_real,
                gr is not None,
                getattr(gr, "bs", None) if gr is not None else None,
                self.capture_bs,
                self.disable_padding,
            )
            with self._lock:
                self._seq += 1
                fields = {
                    "seq": self._seq,
                    "forward_pass_id": getattr(runner, "forward_pass_id", None),
                    "mode": kind,
                    "bs_real": bs_real,
                    **gf,
                    "pdmux_stream_idx": sidx,
                    # PD-mux only (2026-10-28, diag audit r1 M4): the loop's decode iteration counter at this
                    # forward == `decode_iteration.iter_seq` == `prefill_chunk.decode_iter_seq` of the same step,
                    # so I-4 joins the phase events without a host-time join.  None on fused arms.
                    "decode_iter_seq": diter,
                    "t_host_start": t0,
                    I4_HOST_FIELD: (t1 - t0) * 1000.0,
                    I4_FIELD: None,
                    "thread": threading.current_thread().name,
                }
                self.acct.n_by_kind[kind] += 1
                self.acct.n_graph[gf["graph"]] += 1
                if gf["eager_reason"]:
                    self.acct.n_eager_reason[gf["eager_reason"]] += 1
                if gf["pad_check"] == PAD_MISMATCH:
                    self.acct.n_pad_mismatch += 1
                if gf["bs_padded"] is not None:
                    self.acct.pad_tokens += int(gf["bs_padded"]) - bs_real
                if pair is None:
                    self._emit("decode_forward", **fields)
                else:
                    self.pending.append(PendingForward(fields, pair[0], pair[1]))
                    while len(self.pending) > self.max_pending:
                        self.pending.popleft()  # never waited for; counted
                        self.acct.n_dropped_pending += 1
            self.maybe_flush()
        except Exception as exc:
            with self._lock:
                self.acct.n_probe_errors += 1
            self._disable(exc)
        return out

    def maybe_flush(self, force: bool = False) -> None:
        now = self.clock()
        with self._lock:
            if not force and now - self.acct.t_start < self.window_s:
                return
            self._drain_locked()
            rec = self.acct.window_record(now, len(self.pending))
            rec["t_realtime_end"] = time.time()
            rec["probe_disabled"] = self.disabled
            self._emit("decode_probe_window", **rec)
            self.acct.reset(now)

    def close(self) -> None:
        self.maybe_flush(force=True)
        with self._lock:
            self._emit("decode_probe_close", n_unresolved=len(self.pending))
        if self._tel is not None:
            self._tel.close()


# ---- engine glue --------------------------------------------------------
# Everything below touches the scheduler / model runner but still imports no
# torch at module scope.

_PROBE: Optional[DecodeProbe] = None
_WRAPPED: List[Any] = []


def set_probe(probe: Optional[DecodeProbe]) -> Optional[DecodeProbe]:
    global _PROBE
    prev, _PROBE = _PROBE, probe
    return prev


def get_probe() -> Optional[DecodeProbe]:
    return _PROBE


def install_on_runner(runner: Any) -> None:
    """Shadow ``runner.forward`` with an INSTANCE attribute.  Idempotent.

    Instance-level (not class-level) so that a draft/other ``ModelRunner`` of the
    same class in the process is not instrumented, and uninstall is ``del``.
    """
    if "forward" in getattr(runner, "__dict__", {}):
        return
    orig = runner.forward  # bound pristine method

    def forward(*args, **kwargs):
        p = _PROBE
        if p is None:
            return orig(*args, **kwargs)
        return p.around_forward(runner, orig, args, kwargs)

    forward.__wrapped__ = orig  # type: ignore[attr-defined]
    runner.forward = forward
    _WRAPPED.append(runner)


def uninstall_all() -> None:
    while _WRAPPED:
        r = _WRAPPED.pop()
        if "forward" in r.__dict__:
            del r.__dict__["forward"]


def build_boot_record(scheduler: Any, runner: Any) -> Dict[str, Any]:
    """Arm tags + the capture list the I-6 prediction uses.  Read, not assumed."""
    sa = getattr(runner, "server_args", None) or getattr(scheduler, "server_args", None)
    gr = getattr(runner, "graph_runner", None)
    cfg = getattr(getattr(runner, "model_config", None), "hf_config", None)
    archs = getattr(cfg, "architectures", None)
    return {
        "arm_enable_pdmux": bool(getattr(scheduler, "enable_pdmux", False)),
        "arm_enable_overlap": bool(getattr(scheduler, "enable_overlap", False)),
        "arm_chunked_prefill_size": getattr(sa, "chunked_prefill_size", None),
        "arm_enable_mixed_chunk": getattr(sa, "enable_mixed_chunk", None),
        "disable_cuda_graph": getattr(sa, "disable_cuda_graph", None),
        "graph_runner_class": type(gr).__name__ if gr is not None else None,
        "capture_bs": [int(b) for b in (getattr(gr, "capture_bs", None) or [])],
        "graph_max_bs": getattr(gr, "max_bs", None),
        "disable_cuda_graph_padding": bool(getattr(gr, "disable_padding", False)),
        "require_mlp_tp_gather": bool(getattr(gr, "require_mlp_tp_gather", False)),
        "model_architectures": list(archs) if archs else None,
        "runner_class": type(runner).__name__,
    }


def _env_flag_strict(name: str, default: bool) -> bool:
    raw = (os.environ.get(name, "") or "").strip()
    if raw == "":
        return default
    if raw in ("0", "1"):
        return raw == "1"
    raise ValueError(f"{name}={raw!r}: only unset, '0' or '1' are defined")


def maybe_install_decode_probe(scheduler: Any) -> Optional[DecodeProbe]:
    """Return a probe iff ``PDMUX_DECODE_PROBE_PATH`` is set; otherwise None
    (nothing installed, nothing imported beyond this module)."""
    path_str = (os.environ.get("PDMUX_DECODE_PROBE_PATH", "") or "").strip()
    if not path_str:
        return None
    events = _env_flag_strict("PDMUX_DECODE_PROBE_GPU_EVENTS", True)
    import torch

    runner = scheduler.tp_worker.model_runner
    path = Path(path_str)
    tp_size = int(getattr(scheduler, "tp_size", 1) or 1)
    dp_size = int(getattr(scheduler, "dp_size", 1) or 1)
    static: Dict[str, Any] = {}
    if tp_size > 1 or dp_size > 1:
        from sglang.srt.multiplex.multigpu_guard import rank_fields, rank_suffix

        tp_rank = getattr(scheduler, "tp_rank", 0)
        dp_rank = getattr(scheduler, "dp_rank", None)
        path = path.with_name(f"{path.stem}{rank_suffix(tp_size, tp_rank, dp_size, dp_rank)}{path.suffix}")
        static = rank_fields(tp_size, tp_rank, dp_size, dp_rank)
    stream_idx_fn = None
    decode_iter_fn = None
    if bool(getattr(scheduler, "enable_pdmux", False)):
        from sglang.srt.multiplex.pdmux_context import get_current_stream_idx

        stream_idx_fn = get_current_stream_idx
        # incremented by event_loop_pdmux right BEFORE the decode run_batch, read inside that forward
        decode_iter_fn = lambda: getattr(scheduler, "_r2_decode_iterations", None)  # noqa: E731
    boot = build_boot_record(scheduler, runner)
    probe = DecodeProbe(
        path,
        run_id=os.environ.get("PDMUX_RUN_ID", "unlabeled"),
        workload_id=os.environ.get("PDMUX_WORKLOAD_ID", "unlabeled"),
        window_s=float(os.environ.get("PDMUX_DECODE_PROBE_WINDOW_S", "1.0") or 1.0),
        events_enabled=events,
        boot=boot,
        static_fields=static,
        event_factory=(lambda: torch.cuda.Event(enable_timing=True)) if events else None,
        capturing_fn=torch.cuda.is_current_stream_capturing,
        stream_idx_fn=stream_idx_fn,
        decode_iter_fn=decode_iter_fn,
    )
    set_probe(probe)
    install_on_runner(runner)
    logger.info(
        "decode probe ENABLED (I-4 events=%s, I-6; all arms; pdmux=%s capture_bs=%s) -> %s",
        events, boot["arm_enable_pdmux"], boot["capture_bs"], path,
    )
    return probe
