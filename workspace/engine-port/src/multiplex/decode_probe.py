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

EXTEND TIMING (decision 62 item 4, 2026-11-01; default OFF)
------------------------------------------------------------
``PDMUX_PROBE_EXTEND_TIME=1`` (strict: unset | 0 | 1; requires
``PDMUX_DECODE_PROBE_PATH`` and GPU events, else ``ValueError`` at install) also
brackets every plain ``ForwardMode.EXTEND`` forward with the same CUDA-event pair
and emits one ``extend_forward`` record: ``extend_gpu_ms``, ``extend_host_ms``,
``extend_tokens`` (``extend_num_tokens``), ``n_reqs`` (``batch_size``),
``extend_len_max``, ``extend_prefix_tokens``, ``stream_key``, ``seq``,
``decode_iter_seq``.  Only the name EXTEND is timed: ``ForwardMode.is_extend()``
is NOT used because in v0.5.10 it is also True for MIXED / SPLIT_PREFILL /
TARGET_VERIFY / DRAFT_EXTEND / DLLM_EXTEND (MIXED stays a ``decode_forward``
record).  With the flag ON, decode/mixed records additionally carry
``stream_key`` and MIXED records ``extend_tokens``/``n_reqs``/
``mixed_decode_bs_est`` (trailing extend length 1 run = the running decode
requests that ``mix_with_running`` appends; an estimate -- a prefill chunk of
length 1 right before them would be counted too).  ``seq`` is shared by all
timed kinds, so ON a hole in ``seq`` = a dropped record, which
``pacing_ratios`` turns into an excluded gap (never a silent under-count).
Extend pairs use their own pending deque so a long prefill never holds back
decode resolution.  ``pacing_ratios`` computes r_ser / r_mix
(PLAN_stock_baseline_metrics_2026-11-01 sec 0-1, same definitions as the Gate 2
HOLB re-analysis ``results/stock_baseline/r_ser_holb.py``); it is defined for the
stock single-stream path, NOT for the PD-mux r_span.

OFF (flag unset or 0) is byte-identical to the pre-flag probe: no extend record,
no new field in any record, extend forwards pass through and are counted in
``n_other_modes`` exactly as before (pinned by tests/test_decode_probe.py golden).

OBSERVER COST -- INSTRUMENTATION BOOT ONLY.  Turning the flag ON adds two event
records + one query + one queued dict per EXTEND forward and one
``current_stream`` call per timed forward.  Its cost is not measured and is not
claimed to be zero.  It is to be enabled ONLY in a dedicated instrumentation
boot that is separate from every measurement arm (PC-boot precedent C5);
numbers from such a boot are r_ser / r_mix inputs, never goodput / latency
evidence, and measurement arms run with the flag OFF.

FORWARD INTERIOR TIMING (default OFF)
-------------------------------------
``PDMUX_PROBE_FORWARD_INTERIOR=1`` (strict: unset | 0 | 1; requires
``PDMUX_DECODE_PROBE_PATH``, else ``ValueError`` at install) emits one
``forward_interior`` record per DECODE, MIXED and plain EXTEND forward, joined to
``decode_forward`` / ``extend_forward`` by ``seq``.  All stamps use the probe's host
clock, the same clock as ``t_host_start``:

  t_forward_enter / t_forward_return   the bracket of ``ModelRunner.forward``
  t_meta_enter / t_meta_exit           ``init_forward_metadata`` on the attention
                                       backend (first enter, last exit), or
                                       ``init_forward_metadata_replay_cuda_graph``
                                       on the decode-graph path
  t_ssm_meta_enter / t_ssm_meta_exit   the same call on the hybrid backend's
                                       ``linear_attn_backend``.  That sub-interval
                                       holds ``Mamba2Metadata.prepare_mixed`` and its
                                       ``.item()`` (mamba2_metadata.py:201) on EXTEND
                                       and MIXED forwards
  t_launch, launch_source              host clock at the model call: entry of
                                       ``model.forward`` (eager and piecewise), or
                                       return of ``graph_runner.replay_prepare`` (the
                                       decode graph replays right after it)
  t_model_return                       return of ``model.forward``; None on the
                                       decode-graph path

``t_launch`` IS NOT A CUDA API CALLBACK.  It is the last host point before the model's
kernels are issued.  Metadata preparation issues small kernels of its own before it,
so "first kernel of the forward" would land inside the metadata interval.
``t_meta_*`` is host time: a ``.item()`` that waits for earlier GPU work on the same
stream shows up there, and these stamps cannot split waiting from Python work.

The piecewise and eager paths reach the same two objects
(``runner.attn_backend.init_forward_metadata`` and ``runner.model.forward``), so the
same instance-level shadows serve both; ``graph`` plus ``mode`` say which path ran.
SPLIT_PREFILL (PD-mux) is not recorded and stays in ``n_other_modes``.  Capture is
skipped.  EXTEND forwards take a ``seq`` while this flag is ON, even without
``PDMUX_PROBE_EXTEND_TIME``, so decode ``seq`` values have holes in that case; those
files are rejected by ``pacing_ratios`` through its boot check.  A failure inside
this code turns off interior records only and leaves the forward and ``decode_forward``
records alone.

OBSERVER COST IS UNMEASURED.  ON adds up to 6 clock reads, up to 3
``is_current_stream_capturing`` calls, a thread-local lookup per shadowed call and
one queued record per forward.  Same rule as EXTEND timing: dedicated instrumentation
boot only, never a measurement arm.

The pure part (above ``# ---- engine glue``) imports no torch / sglang, so CPU
tests load this file loose.
"""

from __future__ import annotations

import logging
import os
import statistics
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

# EXTEND timing (decision 62 item 4; default OFF, instrumentation boot only)
MODE_EXTEND = "extend"
EXTEND_EVENT = "extend_forward"
EXT_GPU_FIELD = "extend_gpu_ms"
EXT_HOST_FIELD = "extend_host_ms"
EXTEND_ENV = "PDMUX_PROBE_EXTEND_TIME"
INTERIOR_ENV = "PDMUX_PROBE_FORWARD_INTERIOR"
INTERIOR_EVENT = "forward_interior"
INTERIOR_BOOT_EVENT = "forward_interior_boot"
LAUNCH_MODEL = "model_forward"
LAUNCH_GRAPH = "graph_replay_prepare"
PACING_MIN_BS = 5  # == results/stock_baseline/r_ser_holb.py MIN_BS (per-bs T_dec needs >= 5 pure samples)


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


def is_plain_extend(forward_mode: Any) -> bool:
    """True only for ``ForwardMode.EXTEND`` (by name).  Deliberately not
    ``is_extend()``: that is also True for MIXED / SPLIT_PREFILL / TARGET_VERIFY /
    DRAFT_EXTEND / DLLM_EXTEND in v0.5.10, none of which is a stock prefill forward."""
    return forward_mode is not None and mode_name(forward_mode) == MODE_EXTEND


def _int_list(xs: Any) -> List[int]:
    if xs is None:
        return []
    if hasattr(xs, "tolist"):
        xs = xs.tolist()
    return [int(x) for x in xs]


def extend_batch_fields(fb: Any, mixed: bool = False) -> Dict[str, Any]:
    """Host-side batch shape of an EXTEND (or MIXED) forward; reads CPU lists only."""
    lens = _int_list(getattr(fb, "extend_seq_lens_cpu", None))
    prefix = _int_list(getattr(fb, "extend_prefix_lens_cpu", None))
    ent = getattr(fb, "extend_num_tokens", None)
    out = {
        "n_reqs": int(getattr(fb, "batch_size", 0) or 0),
        "extend_tokens": int(ent) if ent is not None else (sum(lens) if lens else None),
        "extend_len_max": max(lens) if lens else None,
        "extend_prefix_tokens": sum(prefix) if prefix else (0 if lens else None),
    }
    if mixed:
        k = 0
        for x in reversed(lens):
            if x != 1:
                break
            k += 1
        out["mixed_decode_bs_est"] = k if lens else None
    return out


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
    # EXTEND timing only (written into the window record only when the flag is ON)
    n_extend_timed: int = 0
    n_extend_resolved: int = 0
    n_extend_dropped_pending: int = 0

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
    if rec.get("event") == EXTEND_EVENT:
        for k in ("seq", "mode", "n_reqs", "extend_tokens", "stream_key", EXT_GPU_FIELD, EXT_HOST_FIELD):
            if k not in rec:
                errs.append(f"{EXTEND_EVENT} without {k}")
        if rec.get("mode") != MODE_EXTEND:
            errs.append(f"{EXTEND_EVENT} with mode {rec.get('mode')!r}")
    return errs


# ---- forward interior ----------------------------------------------------

def _ms(a: Optional[float], b: Optional[float]) -> Optional[float]:
    return None if a is None or b is None else (b - a) * 1000.0


def interior_fields(stamps: Dict[str, Any], t_enter: float, t_return: float) -> Dict[str, Any]:
    """Stamps of one forward -> record fields.

    The four segments partition the forward when every stamp exists:
    ``pre_meta_ms`` (host work before metadata), ``meta_ms``, ``meta_to_launch_ms``
    and ``launch_to_return_ms``.  A missing stamp gives None, never a zero.
    """
    t_meta_enter = stamps.get("t_meta_enter")
    t_meta_exit = stamps.get("t_meta_exit")
    t_launch = stamps.get("t_launch")
    source = LAUNCH_MODEL if t_launch is not None else None
    if t_launch is None and stamps.get("t_graph_launch") is not None:
        t_launch, source = stamps["t_graph_launch"], LAUNCH_GRAPH
    return {
        "t_forward_enter": t_enter,
        "t_forward_return": t_return,
        "t_meta_enter": t_meta_enter,
        "t_meta_exit": t_meta_exit,
        "t_ssm_meta_enter": stamps.get("t_ssm_meta_enter"),
        "t_ssm_meta_exit": stamps.get("t_ssm_meta_exit"),
        "t_launch": t_launch,
        "launch_source": source,
        "t_model_return": stamps.get("t_model_return"),
        "n_meta_calls": int(stamps.get("n_meta_calls", 0)),
        "n_model_calls": int(stamps.get("n_model_calls", 0)),
        "forward_ms": _ms(t_enter, t_return),
        "pre_meta_ms": _ms(t_enter, t_meta_enter),
        "meta_ms": _ms(t_meta_enter, t_meta_exit),
        "ssm_meta_ms": _ms(stamps.get("t_ssm_meta_enter"), stamps.get("t_ssm_meta_exit")),
        "meta_to_launch_ms": _ms(t_meta_exit, t_launch),
        "launch_to_return_ms": _ms(t_launch, t_return),
    }


# ---- realized decode SM time-share (PDMUX_PHASE_DECODE_SHARE) --------------

def decode_sm_time_share(
    records: Sequence[Dict[str, Any]],
    t_window: Optional[Tuple[float, float]] = None,
) -> Dict[str, Any]:
    """Share of decode step time per ``decode_sms``, from ``decode_iteration`` records.

    A step's weight is ``t_sync - t_issue_start``.  ``t_window`` (perf_counter clock)
    keeps steps whose ``t_issue_start`` falls inside it.  Steps with ``decode_sms == 0``
    (stream index 0) stay out of the shares and are counted.  Nothing is imputed:
    ``n_missing_iters`` counts ``iter_seq`` holes inside the kept range, and
    ``coverage`` = summed step time / (last sync - first issue).  Refuses a file with
    no ``t_issue_start`` (a boot without the flag), which would otherwise read as empty.
    """
    its = [r for r in records if r.get("event") == "decode_iteration"]
    if not any(r.get("t_issue_start") is not None for r in its):
        raise ValueError("decode_sm_time_share needs records from a PDMUX_PHASE_DECODE_SHARE=1 boot")
    kept = [r for r in its if r.get("t_issue_start") is not None]
    n_no_stamp = len(its) - len(kept)
    if t_window is not None:
        lo, hi = float(t_window[0]), float(t_window[1])
        kept = [r for r in kept if lo <= float(r["t_issue_start"]) <= hi]
    kept.sort(key=lambda r: int(r["iter_seq"]))
    busy: Dict[int, float] = {}
    steps: Dict[int, int] = {}
    n_zero = 0
    zero_ms = 0.0
    n_missing = 0
    prev = None
    for r in kept:
        seq = int(r["iter_seq"])
        if prev is not None and seq > prev + 1:
            n_missing += seq - prev - 1
        prev = seq
        d = int(r["decode_sms"])
        ms = (float(r["t_sync"]) - float(r["t_issue_start"])) * 1000.0
        if d == 0:
            n_zero += 1
            zero_ms += ms
            continue
        busy[d] = busy.get(d, 0.0) + ms
        steps[d] = steps.get(d, 0) + 1
    total = sum(busy.values())
    span_ms = None
    coverage = None
    if kept:
        span_ms = (max(float(r["t_sync"]) for r in kept) - min(float(r["t_issue_start"]) for r in kept)) * 1000.0
        coverage = (total + zero_ms) / span_ms if span_ms > 0 else None
    return {
        "shares": {d: (ms / total) for d, ms in sorted(busy.items())} if total > 0 else {},
        "busy_ms": dict(sorted(busy.items())),
        "n_steps": dict(sorted(steps.items())),
        "n_kept": len(kept),
        "n_no_issue_stamp": n_no_stamp,
        "n_missing_iters": n_missing,
        "n_decode_sms_zero": n_zero,
        "decode_sms_zero_ms": zero_ms,
        "span_ms": span_ms,
        "coverage": coverage,
        "max_dropped_events": max((int(r.get("dropped_events", 0) or 0) for r in its), default=0),
    }


# ---- r_ser / r_mix (PLAN_stock_baseline_metrics_2026-11-01 sec 0-1) -------

def _is_extend_rec(r: Dict[str, Any]) -> bool:
    return r.get("event") == EXTEND_EVENT


def _is_closer(r: Dict[str, Any]) -> bool:
    """A decode-carrying forward (DECODE or MIXED) closes a gap."""
    return r.get("event") == "decode_forward" and r.get("mode") in (MODE_DECODE, MODE_MIXED)


def _closer_bs(r: Dict[str, Any]) -> Optional[int]:
    """Decode batch size of a closer: DECODE -> bs_real; MIXED -> the decode part only."""
    if r.get("mode") == MODE_MIXED:
        return r.get("mixed_decode_bs_est")
    return r.get("bs_real")


def pacing_ratios(
    records: Sequence[Dict[str, Any]],
    min_bs: int = PACING_MIN_BS,
    t_window: Optional[Tuple[float, float]] = None,
) -> Dict[str, Any]:
    """r_ser and r_mix of one probe file (stock SGLang, single forward stream).

    Gap k = the timed forwards strictly between decode-carrying forward D_{k-1}
    and D_k (``seq`` order = host issue order = stream order on the stock path).

      T_dec(bs)  MEDIAN ``decode_fwd_gpu_ms`` of PURE gaps: D_k is DECODE, the gap
                 holds no extend forward, it is not the first gap and it is
                 complete; per ``bs_real`` when that bs has >= ``min_bs`` samples,
                 else the median over all pure gaps (counted in ``n_fallback``).
      r_ser(k)   sum of ``extend_gpu_ms`` of gap k's EXTEND forwards on D_k's
                 stream / T_dec(bs of D_k), for gaps with that sum > 0.
      r_mix(k)   (``decode_fwd_gpu_ms`` of MIXED D_k - T_dec(b)) / T_dec(b),
                 b = ``mixed_decode_bs_est`` (decode part of the mixed batch).

    Excluded (counted, never silently merged): the first gap (no D_{k-1}), gaps
    with a ``seq`` hole (dropped / unresolved record) or a null GPU time.
    Extend forwards on another stream than D_k are reported in ``other_stream_ms``
    only.  ``t_window`` (probe host clock, ``t_host_start``) filters records first.
    Refuses input without an extend-timing boot record: an OFF file has no
    extend records and would otherwise read as "no stalls".
    """
    boots = [r for r in records if r.get("event") == "decode_probe_boot"]
    if not boots or not all(b.get("extend_time_enabled") is True for b in boots):
        raise ValueError("pacing_ratios needs a probe file booted with PDMUX_PROBE_EXTEND_TIME=1")
    fw = [r for r in records if _is_extend_rec(r) or r.get("event") == "decode_forward"]
    if t_window is not None:
        lo, hi = float(t_window[0]), float(t_window[1])
        fw = [r for r in fw if lo <= float(r.get("t_host_start", float("nan"))) <= hi]
    fw.sort(key=lambda r: int(r["seq"]))

    gaps: List[Dict[str, Any]] = []
    open_ext: List[Dict[str, Any]] = []
    prev: Optional[Dict[str, Any]] = None
    n_first = n_incomplete = n_trailing = 0
    for r in fw:
        if _is_extend_rec(r):
            open_ext.append(r)
            continue
        if not _is_closer(r):
            continue
        if prev is None:
            n_first += 1
        else:
            expect = list(range(int(prev["seq"]) + 1, int(r["seq"])))
            got = [int(e["seq"]) for e in open_ext]
            complete = got == expect and r.get(I4_FIELD) is not None and all(
                e.get(EXT_GPU_FIELD) is not None for e in open_ext)
            if not complete:
                n_incomplete += 1
            else:
                same = [e for e in open_ext if e.get("stream_key") == r.get("stream_key")]
                gaps.append({
                    "seq": int(r["seq"]),
                    "mode": r.get("mode"),
                    "bs": _closer_bs(r),
                    "decode_ms": float(r[I4_FIELD]),
                    "n_extend": len(open_ext),
                    "same_stream_ms": sum(float(e[EXT_GPU_FIELD]) for e in same),
                    "other_stream_ms": sum(float(e[EXT_GPU_FIELD]) for e in open_ext
                                           if e.get("stream_key") != r.get("stream_key")),
                    "extend_tokens": sum(int(e.get("extend_tokens") or 0) for e in same),
                    "extend_n_reqs": sum(int(e.get("n_reqs") or 0) for e in same),
                })
        prev = r
        open_ext = []
    n_trailing = len(open_ext)

    pure = [g for g in gaps if g["mode"] == MODE_DECODE and g["n_extend"] == 0]
    by_bs: Dict[Any, List[float]] = {}
    for g in pure:
        by_bs.setdefault(g["bs"], []).append(g["decode_ms"])
    tdec_all = statistics.median([g["decode_ms"] for g in pure]) if pure else None
    tdec_by_bs = {bs: statistics.median(xs) for bs, xs in by_bs.items() if len(xs) >= int(min_bs)}
    n_fallback = 0

    def tdec(bs: Any) -> Optional[float]:
        nonlocal n_fallback
        if bs in tdec_by_bs:
            return tdec_by_bs[bs]
        n_fallback += 1
        return tdec_all

    r_ser: List[float] = []
    r_mix: List[float] = []
    for g in gaps:
        t = tdec(g["bs"])
        g["tdec_ms"] = t
        g["r_ser"] = g["r_mix"] = None
        if not t:
            continue
        if g["same_stream_ms"] > 0:
            g["r_ser"] = g["same_stream_ms"] / t
            r_ser.append(g["r_ser"])
        if g["mode"] == MODE_MIXED:
            g["r_mix"] = (g["decode_ms"] - t) / t
            r_mix.append(g["r_mix"])
    return {
        "gaps": gaps,
        "r_ser": r_ser,
        "r_mix": r_mix,
        "tdec_by_bs": tdec_by_bs,
        "tdec_all_ms": tdec_all,
        "n_pure": len(pure),
        "n_fallback": n_fallback,
        "n_first": n_first,
        "n_incomplete": n_incomplete,
        "n_trailing_extend": n_trailing,
    }


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
        extend_time: bool = False,
        stream_key_fn: Optional[Callable[[], Any]] = None,
        interior: bool = False,
    ) -> None:
        self.boot = dict(boot or {})
        self.window_s = float(window_s)
        self.events_enabled = bool(events_enabled) and event_factory is not None
        self.clock = clock
        self.event_factory = event_factory
        self.capturing_fn = capturing_fn or (lambda: False)
        self.stream_idx_fn = stream_idx_fn
        self.decode_iter_fn = decode_iter_fn
        # EXTEND timing: instrumentation boot only (see module docstring, OBSERVER COST)
        self.extend_time = bool(extend_time)
        if self.extend_time and not self.events_enabled:
            raise ValueError(f"{EXTEND_ENV}=1 needs GPU events (PDMUX_DECODE_PROBE_GPU_EVENTS unset or 1)")
        self.stream_key_fn = stream_key_fn
        # forward interior timing: instrumentation boot only (module docstring)
        self.interior = bool(interior)
        self._tls = threading.local()
        self.n_interior_records = 0
        self.n_interior_errors = 0
        self.ext_pending: Deque[PendingForward] = deque()
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
                   events_enabled=self.events_enabled, **self.boot,
                   **({"extend_time_enabled": True} if self.extend_time else {}),
                   **({"forward_interior_enabled": True} if self.interior else {}))

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
        if self.extend_time:
            for item, gpu_ms in drain_ready(self.ext_pending):
                item.fields[EXT_GPU_FIELD] = gpu_ms
                self.acct.n_extend_resolved += 1
                self._pool.append((item.start, item.end))
                self._emit(EXTEND_EVENT, **item.fields)

    def poll(self) -> None:
        """Non-blocking resolution of completed pairs (also called by flushes)."""
        with self._lock:
            self._drain_locked()

    # -- forward interior
    def _interior_fail(self, exc: BaseException) -> None:
        if self.interior:
            logger.warning("forward interior timing disabled after failure: %r", exc)
        self.interior = False
        self.n_interior_errors += 1

    def _interior_bind(self, orig: Callable, stamps: Dict[str, Any]) -> Callable:
        """``orig`` with ``stamps`` open on this thread for exactly its duration."""
        tls = self._tls

        def call(*args, **kwargs):
            tls.stamps = stamps
            try:
                return orig(*args, **kwargs)
            finally:
                tls.stamps = None

        return call

    def interior_call(self, orig: Callable, args: tuple, kwargs: dict,
                      enter: Optional[str], exit_: Optional[str], count: Optional[str] = None) -> Any:
        """Run one shadowed engine call, stamping it when a recorded forward is open.

        ``enter`` keeps the first entry, ``exit_`` the last exit.  Engine exceptions
        propagate unchanged; a probe-side failure turns interior timing off.
        """
        st = None
        try:
            st = getattr(self._tls, "stamps", None)
            if st is not None and (not self.interior or self.capturing_fn()):
                st = None
            if st is not None:
                if enter is not None and enter not in st:
                    st[enter] = self.clock()
                if count is not None:
                    st[count] = st.get(count, 0) + 1
        except Exception as exc:
            st = None
            self._interior_fail(exc)
        out = orig(*args, **kwargs)
        if st is not None and exit_ is not None:
            try:
                st[exit_] = self.clock()
            except Exception as exc:
                self._interior_fail(exc)
        return out

    def _emit_interior(self, stamps: Optional[Dict[str, Any]], seq: Optional[int], kind: str, runner: Any,
                       out: Any, sidx: Any, diter: Any, t0: float, t1: float) -> None:
        if stamps is None or seq is None or not self.interior:
            return
        try:
            self._emit(
                INTERIOR_EVENT,
                seq=seq,
                mode=kind,
                graph=GRAPH_HIT if bool(getattr(out, "can_run_graph", False)) else GRAPH_EAGER,
                forward_pass_id=getattr(runner, "forward_pass_id", None),
                pdmux_stream_idx=sidx,
                decode_iter_seq=diter,
                thread=threading.current_thread().name,
                **interior_fields(stamps, t0, t1),
            )
            self.n_interior_records += 1
        except Exception as exc:
            self._interior_fail(exc)

    def _next_seq(self) -> int:
        with self._lock:
            self._seq += 1
            return self._seq

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
                if not ((self.extend_time or self.interior) and is_plain_extend(fm)):
                    return orig(*args, **kwargs)
                kind = MODE_EXTEND
            if self.capturing_fn():
                with self._lock:
                    self.acct.n_skipped_capturing += 1
                return orig(*args, **kwargs)
            self.poll()
            pair = self._take_pair() if (kind != MODE_EXTEND or self.extend_time) else None
            bs_real = int(getattr(fb, "batch_size", 0) or 0)
            sidx = self.stream_idx_fn() if self.stream_idx_fn is not None else None
            diter = self.decode_iter_fn() if self.decode_iter_fn is not None else None
            if self.extend_time:
                skey = self.stream_key_fn() if self.stream_key_fn is not None else None
                ext = extend_batch_fields(fb, mixed=(kind == MODE_MIXED)) if kind != MODE_DECODE else None
            stamps = {} if self.interior else None
            if stamps is not None:
                orig = self._interior_bind(orig, stamps)
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

        if kind == MODE_EXTEND:
            seq = self._record_extend(runner, out, pair, ext, skey, sidx, diter, t0, t1) if self.extend_time else self._next_seq()
            self._emit_interior(stamps, seq, kind, runner, out, sidx, diter, t0, t1)
            return out

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
                if self.extend_time:  # ON only: OFF records stay byte-identical
                    fields["stream_key"] = skey
                    if ext is not None:
                        fields.update(ext)
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
            self._emit_interior(stamps, fields["seq"], kind, runner, out, sidx, diter, t0, t1)
            self.maybe_flush()
        except Exception as exc:
            with self._lock:
                self.acct.n_probe_errors += 1
            self._disable(exc)
        return out

    def _record_extend(self, runner: Any, out: Any, pair: Any, ext: Dict[str, Any], skey: Any,
                       sidx: Any, diter: Any, t0: float, t1: float) -> Optional[int]:
        """EXTEND record (flag ON only).  Not counted in the decode/graph window counters.
        Returns the record's ``seq``, or None if the record could not be built."""
        seq = None
        try:
            with self._lock:
                self._seq += 1
                fields = {
                    "seq": self._seq,
                    "forward_pass_id": getattr(runner, "forward_pass_id", None),
                    "mode": MODE_EXTEND,
                    **ext,
                    "graph": GRAPH_HIT if bool(getattr(out, "can_run_graph", False)) else GRAPH_EAGER,
                    "stream_key": skey,
                    "pdmux_stream_idx": sidx,
                    "decode_iter_seq": diter,
                    "t_host_start": t0,
                    EXT_HOST_FIELD: (t1 - t0) * 1000.0,
                    EXT_GPU_FIELD: None,
                    "thread": threading.current_thread().name,
                }
                self.acct.n_extend_timed += 1
                self.ext_pending.append(PendingForward(fields, pair[0], pair[1]))
                while len(self.ext_pending) > self.max_pending:
                    self.ext_pending.popleft()  # never waited for; counted (seq hole -> excluded gap)
                    self.acct.n_extend_dropped_pending += 1
                seq = fields["seq"]
            self.maybe_flush()
        except Exception as exc:
            with self._lock:
                self.acct.n_probe_errors += 1
            self._disable(exc)
        return seq

    def maybe_flush(self, force: bool = False) -> None:
        now = self.clock()
        with self._lock:
            if not force and now - self.acct.t_start < self.window_s:
                return
            self._drain_locked()
            rec = self.acct.window_record(now, len(self.pending))
            rec["t_realtime_end"] = time.time()
            rec["probe_disabled"] = self.disabled
            if self.extend_time:
                rec.update(n_extend_timed=self.acct.n_extend_timed, n_extend_resolved=self.acct.n_extend_resolved,
                           n_extend_pending=len(self.ext_pending),
                           n_extend_dropped_pending=self.acct.n_extend_dropped_pending)
            if self.interior or self.n_interior_errors:
                rec.update(n_interior_records=self.n_interior_records, n_interior_errors=self.n_interior_errors)
            self._emit("decode_probe_window", **rec)
            self.acct.reset(now)

    def close(self) -> None:
        self.maybe_flush(force=True)
        with self._lock:
            if self.extend_time:
                self._emit("decode_probe_close", n_unresolved=len(self.pending),
                           n_extend_unresolved=len(self.ext_pending))
            else:
                self._emit("decode_probe_close", n_unresolved=len(self.pending))
        if self._tel is not None:
            self._tel.close()


# ---- engine glue --------------------------------------------------------
# Everything below touches the scheduler / model runner but still imports no
# torch at module scope.

_PROBE: Optional[DecodeProbe] = None
_WRAPPED: List[Any] = []
_INTERIOR_WRAPPED: List[Tuple[Any, str]] = []


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
    while _INTERIOR_WRAPPED:
        owner, name = _INTERIOR_WRAPPED.pop()
        if name in owner.__dict__:
            del owner.__dict__[name]


def _shadow(owner: Any, name: str, enter: Optional[str], exit_: Optional[str], count: Optional[str] = None) -> bool:
    """Instance-level shadow of ``owner.name`` that reads ``_PROBE`` on every call.

    Returns False, without raising, when the site cannot be shadowed: the boot record
    lists the sites that were, so a missing site is visible in the data.
    """
    try:
        if owner is None or name in vars(owner) or not callable(getattr(owner, name, None)):
            return False
        orig = getattr(owner, name)

        def shadow(*args, **kwargs):
            p = _PROBE
            if p is None:
                return orig(*args, **kwargs)
            return p.interior_call(orig, args, kwargs, enter, exit_, count)

        shadow.__wrapped__ = orig  # type: ignore[attr-defined]
        setattr(owner, name, shadow)
    except (AttributeError, TypeError) as exc:
        logger.warning("forward interior: cannot shadow %s.%s: %r", type(owner).__name__, name, exc)
        return False
    _INTERIOR_WRAPPED.append((owner, name))
    return True


def _attention_backends(runner: Any) -> List[Any]:
    """Every attention backend a recorded forward can reach, without duplicates."""
    found = [getattr(runner, "attn_backend", None), getattr(runner, "decode_attn_backend", None)]
    found.extend(getattr(runner, "decode_attn_backend_group", None) or [])
    out, seen = [], set()
    for b in found:
        if b is not None and id(b) not in seen:
            seen.add(id(b))
            out.append(b)
    return out


def install_interior(runner: Any) -> List[str]:
    """Shadow the interior sites of ``runner``; returns the sites wrapped."""
    sites: List[str] = []
    for backend in _attention_backends(runner):
        label = type(backend).__name__
        for name in ("init_forward_metadata", "init_forward_metadata_replay_cuda_graph"):
            if _shadow(backend, name, "t_meta_enter", "t_meta_exit", "n_meta_calls"):
                sites.append(f"{label}.{name}")
        linear = getattr(backend, "linear_attn_backend", None)
        if _shadow(linear, "init_forward_metadata", "t_ssm_meta_enter", "t_ssm_meta_exit"):
            sites.append(f"{type(linear).__name__}.init_forward_metadata")
    if _shadow(getattr(runner, "model", None), "forward", "t_launch", "t_model_return", "n_model_calls"):
        sites.append("model.forward")
    if _shadow(getattr(runner, "graph_runner", None), "replay_prepare", None, "t_graph_launch"):
        sites.append("graph_runner.replay_prepare")
    return sites


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
    extend_time = _env_flag_strict(EXTEND_ENV, False)
    interior = _env_flag_strict(INTERIOR_ENV, False)
    if not path_str:
        if extend_time:  # fail fast: an instrumentation boot that would silently record nothing
            raise ValueError(f"{EXTEND_ENV}=1 without PDMUX_DECODE_PROBE_PATH")
        if interior:
            raise ValueError(f"{INTERIOR_ENV}=1 without PDMUX_DECODE_PROBE_PATH")
        return None
    events = _env_flag_strict("PDMUX_DECODE_PROBE_GPU_EVENTS", True)
    if extend_time and not events:
        raise ValueError(f"{EXTEND_ENV}=1 with PDMUX_DECODE_PROBE_GPU_EVENTS=0: EXTEND timing needs CUDA events")
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
        **({"extend_time": True, "stream_key_fn": lambda: int(torch.cuda.current_stream().cuda_stream)}
           if extend_time else {}),
        **({"interior": True} if interior else {}),
    )
    set_probe(probe)
    install_on_runner(runner)
    if interior:
        sites = install_interior(runner)
        probe._emit(INTERIOR_BOOT_EVENT, sites=sites,
                    piecewise_runner_class=type(getattr(runner, "piecewise_cuda_graph_runner", None)).__name__,
                    graph_runner_class=type(getattr(runner, "graph_runner", None)).__name__,
                    disable_piecewise_cuda_graph=getattr(getattr(runner, "server_args", None),
                                                         "disable_piecewise_cuda_graph", None))
        logger.warning("decode probe forward interior ON (%s=1): instrumentation boot only, not a measurement arm; "
                       "sites=%s", INTERIOR_ENV, sites)
    logger.info(
        "decode probe ENABLED (I-4 events=%s, I-6; all arms; pdmux=%s capture_bs=%s) -> %s",
        events, boot["arm_enable_pdmux"], boot["capture_bs"], path,
    )
    if extend_time:
        logger.warning("decode probe EXTEND timing ON (%s=1): instrumentation boot only, not a measurement arm",
                       EXTEND_ENV)
    return probe
