"""Chunked-prefill instrumentation for **every** serving arm (CP-0 prerequisite P1).

WHY THIS EXISTS
---------------
Before this module the repository had *no* way to observe chunked prefill on a
non-pdmux code path::

    grep -rn "num_chunked|chunk_count|chunked_prefill_count" sglang/srt/   -> 0 hits

and the only place that exported ``extend_num_tokens`` as telemetry was
``sglang/srt/multiplex/multiplexing_mixin.py``, which runs **only** under
``--enable-pdmux``.  ``PREREG_CP0_2026-08-28.md`` section 2 registers the two
counters this module produces and the acceptance tests that must pass before
the CP-0 campaign may be submitted.

THE TWO REGISTERED COUNTERS (prereg section 2.1, verbatim definitions)
---------------------------------------------------------------------
``n_requests_chunked``
    the number of *distinct* ``rid`` values that were truncated **at least
    once** during this boot.
``n_chunk_events``
    the number of *times* a truncation happened.

The prereg's table row for the second counter says "한 요청이 여러 번 잘릴 수
있다 ... ①보다 크거나 같다" -- one request may be cut several times, so (2) is
>= (1).  That is only realisable if the *continuation* truncation is counted;
see MECHANISM LIST below, item M3.  The two admission-time sites named in the
prereg text can fire **at most once per request**, so counting only those makes
``n_chunk_events == n_requests_chunked`` an identity.  This module therefore
reports the total *and* its two additive parts, so that either reading can be
recovered from the artifact and the identity is visible rather than hidden::

    n_chunk_events = n_chunk_events_admission + n_chunk_events_continuation
    n_chunk_events_admission == n_requests_chunked      (an identity -- see M3)

MECHANISM LIST (not a site list)
--------------------------------
Every way a request's prefill can be cut short by the *token budget* in
``sglang/srt/managers/schedule_policy.py`` (v0.5.10 dev tree, 2026-08-28):

M1  ``PrefillAdder.add_one_req``           final ``else:`` branch.
    ``trunc_len = self.rem_chunk_tokens // self.page_size * self.page_size``
    then ``req.set_extend_input_len(trunc_len)`` and ``self.new_chunked_req =
    req``.  Reached when ``req.sampling_params.ignore_eos`` is False *or* the
    tree cache is a real prefix cache.  -> ``MECH_ADMISSION_STD``

M2  ``PrefillAdder.add_one_req_ignore_eos`` final ``else:`` branch.
    ``trunc_len = self.rem_chunk_tokens`` (no page alignment) then the same two
    statements.  Reached from ``add_one_req`` when
    ``req.sampling_params.ignore_eos and getattr(self.tree_cache, "disable",
    True)``.  ``sglang.bench_serving`` sends ``ignore_eos = not
    args.disable_ignore_eos`` (bench_serving.py:535,612), and
    ``--disable-radix-cache`` makes the cache a ``ChunkCache`` whose ``disable``
    property is hardcoded ``True`` (chunk_cache.py:50-52), so **this is the
    mechanism the CP-0 harness actually rides**.  -> ``MECH_ADMISSION_IGNORE_EOS``

M3  ``PrefillAdder.add_chunked_req``.  The *continuation* of a request that is
    already ``Scheduler.chunked_req``.  ``truncated = req.extend_input_len >
    _rem_tokens`` then ``req.set_extend_input_len(min(...))``; the method
    returns the request iff it was truncated again.  A 3000-token prompt at
    ``cps 512`` is cut once by M1/M2 and five more times here.  Not named in the
    prereg's fixed site list; counted because without it the prereg's own
    definition of ``n_chunk_events`` is unrealisable (see above).
    -> ``MECH_CONTINUATION``

M4  ``PrefillAdder._add_dllm_req`` / ``PrefillAdder.add_dllm_staging_req``.
    Diffusion-LLM block truncation, live only when ``dllm_config is not None``.
    Structurally inert for the CP-0 arms.  Counted into a **separate**
    contamination counter (``n_dllm_trunc_events``) that must stay 0; it is not
    folded into either registered counter.

NOT a truncation mechanism, enumerated so the list is closed:
  * ``add_one_req`` line 784 ``req.set_extend_input_len(len(req.fill_ids) -
    len(req.prefix_indices))`` -- hierarchical-cache load-back *recompute*, can
    only grow the extend length.
  * ``AddReqResult.OTHER`` / ``NO_TOKEN`` returns (``max_prefill_tokens``,
    ``max_running_requests``, KV pressure) -- these **defer** a request to a
    later batch, they do not split it.
  * ``ForwardMode.SPLIT_PREFILL`` (pdmux) -- splits one prefill across *layer*
    groups, not across tokens; ``extend_input_len`` is unchanged.

HOW THE MECHANISMS ARE DETECTED
-------------------------------
The counters are **not** placed inside the branches.  They are placed in
wrappers around the three methods, and they read the engine's own truncation
postcondition:

  * M1/M2: ``PrefillAdder.new_chunked_req`` transitioned from something else to
    this request during the call.  ``new_chunked_req`` is assigned in exactly
    two places in the whole file (schedule_policy.py:723 and :839) and both are
    the truncation branches, so the postcondition is unique to them
    (``:413`` initialises it to ``None``).
  * M3: ``add_chunked_req`` returns non-``None`` iff its own local
    ``truncated`` is True.

The wrapper form is what makes "default OFF" mean *zero added code*: when
``PDMUX_CHUNK_PROBE_PATH`` is unset no wrapper is installed at all, so the
scheduler runs the pristine functions and there is not even an ``is not None``
test on the admission path.  ``maybe_install_chunk_probe`` is then the only
code that ever runs, once, at scheduler construction.

WHY THE SCHEDULER LAYER AND NOT THE FORWARD
-------------------------------------------
``server_args.py:1254-1259`` derives ``piecewise_cuda_graph_max_tokens`` from
``chunked_prefill_size``; ``_generate_piecewise_cuda_graph_tokens`` (:1397-1415)
keeps only capture sizes ``<= max_tokens``, which for ``--chunked-prefill-size
-1`` is the **empty list**; and ``model_executor/model_runner.py:2486-2491``
then disables piecewise CUDA graph entirely (2026-09-01: the path component is
part of the citation because a bare ``model_runner.py`` is ambiguous in this
tree -- ``hardware_backend/mlx/model_runner.py`` also matches -- and the range
now ends on the ``return`` that actually performs the disabling, which
``:2486-2490`` stopped one line short of).  The negative-control arm is therefore the one
arm whose prefill runs eager.  A counter living inside a graph-captured prefill
path would make that arm report zero *because the path did not run*, not
because nothing was chunked.  Everything here lives in ``schedule_policy`` /
``Scheduler.run_batch``, both of which are plain host Python that executes
identically in all six arms.  ``PREREG_CP0_2026-08-28.md`` acceptance test P1-g
requires the piecewise banner to be recorded so that this can be checked from
the artifact instead of trusted.

OUTPUT
------
JSONL at ``PDMUX_CHUNK_PROBE_PATH`` via the same async writer every other arm
uses (``sglang.srt.multiplex.telemetry.AsyncJsonlTelemetry``):

  ``chunk_open``      one line, boot configuration echo (P1-g fields)
  ``chunk_event``     one line per truncation: rid, mechanism, lengths, host_ts
  ``prefill_forward`` one line per PREFILL forward carrying extend tokens
  ``nonprefill_forward``
                      one line per NON-prefill forward that nevertheless
                      carries a positive ``extend_num_tokens`` (the stale-field
                      symptom below).  ``is_prefill: false``; excluded from the
                      aggregates, kept in the raw data.
  ``chunk_summary``   periodic + shutdown roll-up

WHICH FORWARDS CHANNEL 2 COUNTS (repair 1, 2026-09-01, after job 899768)
------------------------------------------------------------------------
``ScheduleBatch.extend_num_tokens`` is not cleared when the batch object is
reused for a decode step, so ``extend_num_tokens > 0`` is NOT the predicate
"this forward was a prefill".  On job 899768's cps512 boot 101 of the 116
``prefill_forward`` lines had ``forward_mode == "DECODE"`` -- each carrying the
extend count of the preceding prefill (``mode=DECODE, extend_num_tokens=7,
batch_size=1`` is verbatim from that file) -- and the pre-registered P1-e
expectations, which are counts of *prefill* forwards, could not be compared
with the counter at all.  Channel 2 now requires BOTH a prefill
``forward_mode`` (``PREFILL_FORWARD_MODES``, checked member-by-member against
the engine's own ``ForwardMode.is_extend``) and a positive extend count.  The
excluded records are still written and still counted, in their own event and
their own cells; nothing is silently corrected.

DURABILITY (2026-09-01, after job 899768).  A boot that serves a handful of
requests never reaches the writer's ``flush_every`` threshold, and the
scheduler process is SIGKILLed by its own parent, so ``atexit`` never runs:
before this date such a boot wrote a file of *exactly zero bytes*, not even the
``chunk_open`` line.  Two mechanisms now cover that, and neither of them
touches the emission path (P1-f neutrality): the writer flushes on a wall-clock
interval as well as on a record count, and ``install_shutdown_handlers`` closes
the probe on SIGTERM as well as on ``atexit``.

Rate-segment decomposition (prereg section 4.1 requires channel 1 to be
reported at each of the three probe rates separately) is done offline by
``results/cp_baseline/analyze_chunk_probe.py`` from the ``host_ts`` field and
the harness's own segment boundaries; the engine is never told what rate it is
serving.

TWO CONFIGURATIONS THAT CHANGE WHAT CHANNEL 2 MEANS
---------------------------------------------------
Both are recorded in the boot config echo so the reader can see them; neither is
used by the CP-0 arms, and neither is silently corrected here.

``--enable-mixed-chunk``
    ``ScheduleBatch.prepare_for_extend`` adds the decode batch size to
    ``extend_num_tokens`` (schedule_batch.py:1899), so on a MIXED batch the
    per-forward extend count is prefill tokens **plus** one token per running
    decode request.  Channel 2 would then not be "prefill tokens per forward".
    Echoed as ``enable_mixed_chunk``.

``PDMUX_TRUE_DUAL_WORKER=1``
    forwards are issued from two host threads, so the channel-2 counters (plain
    read-modify-write increments) can lose updates.  Channel 1 is unaffected --
    admission still runs on the single scheduler thread.  Echoed as
    ``true_dual_worker`` and logged as a warning at install time.  This probe
    does NOT fail fast on it (unlike ``holb_probe``, whose accountant depends on
    single-threaded *ordering* and would produce a plausible wrong number); here
    the risk is a bounded undercount on one channel, and refusing to boot would
    surprise tracks that have nothing to do with CP-0.
"""

from __future__ import annotations

import atexit
import logging
import os
import signal
import threading
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional

from sglang.srt.multiplex.telemetry import AsyncJsonlTelemetry

logger = logging.getLogger(__name__)

# v2 (repair 1, 2026-09-01): channel 2 changed POPULATION -- it counts prefill
# forwards only, and non-prefill forwards carrying a stale extend count are
# written under their own event.  The version is load bearing: it is how
# analyze_chunk_probe.py tells "these archived totals were computed with the old
# rule" (an expected overcount) from "the two passes disagree" (a defect).
CHUNK_SCHEMA = "pdmux.chunk-probe/v2"

# Mechanism labels (see MECHANISM LIST in the module docstring).
MECH_ADMISSION_STD = "admission_std"          # M1: add_one_req else-branch
MECH_ADMISSION_IGNORE_EOS = "admission_ignore_eos"  # M2: add_one_req_ignore_eos
MECH_CONTINUATION = "continuation"            # M3: add_chunked_req truncated
MECH_DLLM = "dllm"                            # M4: contamination detector only

ADMISSION_MECHS = (MECH_ADMISSION_STD, MECH_ADMISSION_IGNORE_EOS)

# -- forward-mode classification for channel 2 (repair 1, 2026-09-01) --------
#
# ``ScheduleBatch.extend_num_tokens`` is NOT cleared when the batch object is
# reused for a decode step, so "``extend_num_tokens > 0``" does not mean "this
# forward was a prefill".  In the engine the field is written by
# ``prepare_for_extend`` (``managers/schedule_batch.py:1612``) and zeroed ONLY
# by ``prepare_for_idle`` (:2049); ``prepare_for_decode`` (:2062) sets the
# forward mode and leaves the extend count standing (verified 2026-09-01).  Job 899768's cps512 boot proved it: 101 of its 116
# ``prefill_forward`` lines had ``forward_mode == "DECODE"``, each carrying the
# extend count of the preceding prefill.  Channel 2 therefore classifies by
# ``forward_mode`` and counts the prefill modes only.
#
# The sets mirror ``ForwardMode.is_extend`` / its complement
# (``model_executor/forward_batch_info.py``); ``test_chunk_probe.py`` checks
# them member-by-member against the engine's own predicate and fails if SGLang
# grows a mode that lands in neither set.  DRAFT_EXTEND_V2 is on the prefill
# side here although the engine's default ``is_extend()`` excludes it (it is
# admitted only under ``include_draft_extend_v2=True``): the question channel 2
# asks is "did this forward carry extend tokens", and that one does.  No CP-0
# arm enables speculative decoding, so the divergence is unreachable there.
PREFILL_FORWARD_MODES = frozenset({
    "EXTEND",
    "MIXED",
    "DRAFT_EXTEND",
    "DRAFT_EXTEND_V2",
    "TARGET_VERIFY",
    "SPLIT_PREFILL",
    "DLLM_EXTEND",
})
NONPREFILL_FORWARD_MODES = frozenset({"DECODE", "IDLE", "PREBUILT"})

CLASS_PREFILL = "prefill"
CLASS_NONPREFILL = "nonprefill"
# Neither set matched.  NOT folded into either one: a mode the probe cannot
# name is a hole in the instrument, and the roll-up warns when this is nonzero.
CLASS_UNKNOWN = "unknown"


def classify_forward_mode(mode_name) -> str:
    """Name of a ``ForwardMode`` (or None) -> one of the three classes above."""
    if mode_name is None:
        return CLASS_UNKNOWN
    name = str(mode_name)
    if name in PREFILL_FORWARD_MODES:
        return CLASS_PREFILL
    if name in NONPREFILL_FORWARD_MODES:
        return CLASS_NONPREFILL
    return CLASS_UNKNOWN

# How long a signal handler waits for the shutdown close to finish before it
# lets the process die anyway.  See install_shutdown_handlers.
CLOSE_ON_SIGNAL_TIMEOUT_S = 3.0

# Bound on the distinct-rid set so a long-lived server cannot grow it without
# limit.  Well above the CP-0 workload (a deterministic 200-request set).
_RID_SET_CAP = 500_000


class ChunkAccountant:
    """Pure-Python accounting.  No engine imports, no I/O, no CUDA.

    Fed with already-extracted scalars so that the arithmetic can be tested on
    CPU independently of the wrappers that produce them.
    """

    def __init__(self, rid_cap: int = _RID_SET_CAP):
        self.rid_cap = int(rid_cap)
        self._rids: set = set()
        self.rid_set_truncated = 0

        # channel 1
        self.n_requests_chunked = 0
        self.n_chunk_events = 0
        self.n_chunk_events_admission = 0
        self.n_chunk_events_continuation = 0
        self.mech_counts: Counter = Counter()
        self.n_dllm_trunc_events = 0

        # channel 2.  `n_prefill_forwards` is the ONLY population the
        # extend-token statistics are computed over; `n_nonprefill_forwards` is
        # its complement in `n_forwards`, so the two always sum to the total.
        # `forward_cells` enumerates the world that decides which is which:
        # (mode class) x (does this forward carry extend tokens).
        self.n_forwards = 0
        self.n_prefill_forwards = 0
        self.n_nonprefill_forwards = 0
        self.n_unknown_mode_forwards = 0
        self.n_nonprefill_forwards_with_extend_tokens = 0
        self.sum_extend_tokens = 0
        self.sum_extend_tokens_not_counted = 0
        self.max_extend_tokens = 0
        self.min_extend_tokens: Optional[int] = None
        self.forward_mode_counts: Counter = Counter()
        self.forward_cells: Counter = Counter()

        # health / anomalies -- reported, never silently absorbed
        self.n_events_without_shrink = 0
        self.n_events_unknown_rid = 0

    # -- channel 1 ----------------------------------------------------------
    def record_chunk(
        self,
        rid: Any,
        mechanism: str,
        len_before: Optional[int],
        len_after: Optional[int],
    ) -> Dict[str, Any]:
        self.mech_counts[mechanism] += 1
        if mechanism == MECH_DLLM:
            self.n_dllm_trunc_events += 1
            return {
                "mechanism": mechanism,
                "rid": rid,
                "len_before": len_before,
                "len_after": len_after,
                "first_for_rid": False,
            }

        self.n_chunk_events += 1
        if mechanism in ADMISSION_MECHS:
            self.n_chunk_events_admission += 1
        elif mechanism == MECH_CONTINUATION:
            self.n_chunk_events_continuation += 1

        first = False
        if rid is None:
            self.n_events_unknown_rid += 1
        elif rid in self._rids:
            pass
        elif len(self._rids) < self.rid_cap:
            self._rids.add(rid)
            self.n_requests_chunked += 1
            first = True
        else:
            self.rid_set_truncated += 1

        if (
            len_before is not None
            and len_after is not None
            and not (len_after < len_before)
        ):
            # A truncation that did not shorten anything.  Never expected; it
            # would mean the postcondition this probe reads is not the
            # truncation.  Counted, not swallowed.
            self.n_events_without_shrink += 1

        return {
            "mechanism": mechanism,
            "rid": rid,
            "len_before": len_before,
            "len_after": len_after,
            "first_for_rid": first,
        }

    # -- channel 2 ----------------------------------------------------------
    def record_forward(self, mode_name: str, extend_num_tokens: Optional[int]) -> bool:
        """Account one ``Scheduler.run_batch`` call.  Returns is-prefill.

        ★2026-09-01 (repair 1).  The test used to be ``extend_num_tokens > 0``
        alone.  ``ScheduleBatch.extend_num_tokens`` is not cleared when the
        batch is reused for a decode step, so that read a STALE field and
        counted decode steps into channel 2: on job 899768's cps512 boot 101 of
        116 accounted forwards were ``forward_mode == "DECODE"``, each carrying
        the extend count of the preceding prefill.  Both conditions are now
        required, and every forward is filed into one of the six cells of
        (mode class) x (carries extend tokens) so the classification is visible
        in the artifact rather than implied by a difference of two numbers.
        """
        self.n_forwards += 1
        self.forward_mode_counts[mode_name] += 1
        mode_class = classify_forward_mode(mode_name)
        if mode_class == CLASS_UNKNOWN:
            self.n_unknown_mode_forwards += 1

        ext = extend_num_tokens
        has_ext = ext is not None and ext > 0
        self.forward_cells[f"{mode_class}/{'ext' if has_ext else 'noext'}"] += 1

        if not (mode_class == CLASS_PREFILL and has_ext):
            self.n_nonprefill_forwards += 1
            if has_ext:
                # Recorded, not swallowed: a non-prefill forward carrying a
                # positive extend count is exactly the stale-field symptom, and
                # its size is the magnitude of the old defect.
                self.n_nonprefill_forwards_with_extend_tokens += 1
                self.sum_extend_tokens_not_counted += int(ext)
            return False

        ext = int(ext)
        self.n_prefill_forwards += 1
        self.sum_extend_tokens += ext
        if ext > self.max_extend_tokens:
            self.max_extend_tokens = ext
        if self.min_extend_tokens is None or ext < self.min_extend_tokens:
            self.min_extend_tokens = ext
        return True

    # -- read ---------------------------------------------------------------
    def summary(self) -> Dict[str, Any]:
        mean = (
            self.sum_extend_tokens / self.n_prefill_forwards
            if self.n_prefill_forwards
            else None
        )
        return {
            # channel 1 -- the two registered counters
            "n_requests_chunked": self.n_requests_chunked,
            "n_chunk_events": self.n_chunk_events,
            # additive decomposition of n_chunk_events (see module docstring)
            "n_chunk_events_admission": self.n_chunk_events_admission,
            "n_chunk_events_continuation": self.n_chunk_events_continuation,
            "mech_counts": dict(self.mech_counts),
            "n_dllm_trunc_events": self.n_dllm_trunc_events,
            # channel 2
            "extend_tokens_per_forward": {
                "max": self.max_extend_tokens if self.n_prefill_forwards else None,
                "min": self.min_extend_tokens,
                "mean": mean,
                "sum": self.sum_extend_tokens,
                "n_forwards": self.n_prefill_forwards,
            },
            "n_forwards_total": self.n_forwards,
            "n_prefill_forwards": self.n_prefill_forwards,
            # the complement of n_prefill_forwards in n_forwards_total: every
            # forward that channel 2 did NOT count, for any of the three
            # reasons enumerated in forward_cells
            "n_nonprefill_forwards": self.n_nonprefill_forwards,
            "n_unknown_mode_forwards": self.n_unknown_mode_forwards,
            "n_nonprefill_forwards_with_extend_tokens":
                self.n_nonprefill_forwards_with_extend_tokens,
            "sum_extend_tokens_not_counted": self.sum_extend_tokens_not_counted,
            "forward_cells": dict(self.forward_cells),
            "forward_mode_counts": dict(self.forward_mode_counts),
            # health
            "n_events_without_shrink": self.n_events_without_shrink,
            "n_events_unknown_rid": self.n_events_unknown_rid,
            "rid_set_truncated": self.rid_set_truncated,
        }


class ChunkProbe:
    """Accountant + async JSONL emission + periodic roll-up."""

    def __init__(
        self,
        path: Optional[Path],
        run_id: str = "unlabeled",
        workload_id: str = "unlabeled",
        arm: str = "unlabeled",
        *,
        telemetry: Optional[AsyncJsonlTelemetry] = None,
        emit_events: bool = True,
        emit_forwards: bool = True,
        summary_every_s: float = 5.0,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.acct = ChunkAccountant()
        self.arm = arm
        self.emit_events = bool(emit_events)
        self.emit_forwards = bool(emit_forwards)
        self.summary_every_s = float(summary_every_s)
        self.config = dict(config or {})
        self.n_errors = 0
        self._error_logged = False
        self._closed = False
        self._last_summary_ts = time.time()
        # Incremented on every accounted chunk event.  The wrappers use it to
        # detect that an inner wrapper already accounted for this truncation
        # (``add_one_req`` delegates to ``add_one_req_ignore_eos``), so the
        # innermost frame owns the attribution and nothing is double counted.
        self.fire_count = 0
        self.telemetry = (
            telemetry
            if telemetry is not None
            else AsyncJsonlTelemetry(path, run_id=run_id, workload_id=workload_id)
        )
        self.telemetry.emit(
            "chunk_open",
            "startup",
            host_ts=time.time(),
            schema_chunk=CHUNK_SCHEMA,
            arm=self.arm,
            **self.config,
        )

    # -- scheduler-facing API ----------------------------------------------
    def on_chunk(
        self,
        rid: Any,
        mechanism: str,
        len_before: Optional[int],
        len_after: Optional[int],
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        try:
            rec = self.acct.record_chunk(rid, mechanism, len_before, len_after)
            if mechanism != MECH_DLLM:
                self.fire_count += 1
            if self.emit_events:
                if extra:
                    rec.update(extra)
                rec["host_ts"] = time.time()
                rec["arm"] = self.arm
                self.telemetry.emit(
                    "chunk_event", "measure", request_id=str(rid), **rec
                )
            self._maybe_summary()
        except Exception:
            self._on_error()

    def on_forward(self, mode_name: str, extend_num_tokens: Optional[int], bs: int,
                   has_chunked_req: bool) -> None:
        """Account, and write the line, for one forward.

        ★2026-09-01 (repair 1).  A forward that carries extend tokens but is
        not a prefill is written as ``nonprefill_forward`` with
        ``is_prefill: false`` -- KEPT in the raw JSONL, and excluded only from
        the aggregates.  Those records are the evidence that the stale-field
        defect existed (they are what the 899768 diagnosis was read off), so
        the repair may not delete them.

        A forward with no extend tokens at all still writes nothing: an idle
        server polls ``run_batch`` and a line per poll would be a telemetry
        flood with no diagnostic content.  It is counted in the accountant's
        cells either way, so the artifact never loses the total.
        """
        try:
            is_prefill = self.acct.record_forward(mode_name, extend_num_tokens)
            ext = extend_num_tokens
            has_ext = ext is not None and ext > 0
            if self.emit_forwards and has_ext:
                self.telemetry.emit(
                    "prefill_forward" if is_prefill else "nonprefill_forward",
                    "measure",
                    host_ts=time.time(),
                    arm=self.arm,
                    forward_mode=mode_name,
                    mode_class=classify_forward_mode(mode_name),
                    is_prefill=bool(is_prefill),
                    extend_num_tokens=int(ext),
                    batch_size=int(bs),
                    has_chunked_req=bool(has_chunked_req),
                )
            self._maybe_summary()
        except Exception:
            self._on_error()

    # -- output -------------------------------------------------------------
    def _maybe_summary(self) -> None:
        now = time.time()
        if now - self._last_summary_ts >= self.summary_every_s:
            self._last_summary_ts = now
            self.emit_summary("running")

    def emit_summary(self, phase: str) -> Dict[str, Any]:
        s = self.acct.summary()
        s["host_ts"] = time.time()
        s["arm"] = self.arm
        s["n_errors"] = self.n_errors
        s["schema_chunk"] = CHUNK_SCHEMA
        s.update(self.config)
        self.telemetry.emit("chunk_summary", phase, **s)
        return s

    def close(self) -> None:
        if self._closed:
            return
        try:
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
            logger.error(
                "chunk probe error (instrumentation only):\n%s", traceback.format_exc()
            )


# ---------------------------------------------------------------------------
# Wrapper installation.  Nothing below runs unless PDMUX_CHUNK_PROBE_PATH is set.
# ---------------------------------------------------------------------------

_PROBE: Optional[ChunkProbe] = None
_INSTALLED = False
_ORIGINALS: Dict[str, Any] = {}


def _rid_of(req: Any) -> Any:
    rid = getattr(req, "rid", None)
    return rid if rid is not None else None


def _origin_len(req: Any) -> Optional[int]:
    ids = getattr(req, "origin_input_ids", None)
    try:
        return len(ids) if ids is not None else None
    except TypeError:
        return None


def _prefix_len(req: Any) -> Optional[int]:
    idx = getattr(req, "prefix_indices", None)
    try:
        return len(idx) if idx is not None else None
    except TypeError:
        return None


def _make_admission_wrapper(orig, mechanism: str):
    """Wrap an admission method; account iff `new_chunked_req` gained `req`."""

    def wrapper(self, req, *args, **kwargs):
        probe = _PROBE
        if probe is None:  # pragma: no cover - uninstall path
            return orig(self, req, *args, **kwargs)
        fired0 = probe.fire_count
        before = self.new_chunked_req
        len_entry = getattr(req, "extend_input_len", None)
        res = orig(self, req, *args, **kwargs)
        if probe.fire_count == fired0:
            after = self.new_chunked_req
            if after is not None and after is not before:
                probe.on_chunk(
                    _rid_of(after),
                    mechanism,
                    len_entry,
                    getattr(after, "extend_input_len", None),
                    extra={
                        "rem_chunk_tokens_after": getattr(self, "rem_chunk_tokens", None),
                        "page_size": getattr(self, "page_size", None),
                        "n_origin_input_ids": _origin_len(after),
                        "n_prefix_indices": _prefix_len(after),
                        "n_in_batch": len(getattr(self, "can_run_list", ()) or ()),
                    },
                )
        return res

    wrapper.__name__ = getattr(orig, "__name__", "wrapped")
    wrapper.__qualname__ = getattr(orig, "__qualname__", "wrapped")
    wrapper.__doc__ = getattr(orig, "__doc__", None)
    wrapper._chunk_probe_wrapped = True  # type: ignore[attr-defined]
    wrapper._chunk_probe_mechanism = mechanism  # type: ignore[attr-defined]
    return wrapper


def _make_continuation_wrapper(orig):
    """Wrap ``add_chunked_req``; it returns the request iff it was re-truncated."""

    def wrapper(self, req, *args, **kwargs):
        probe = _PROBE
        if probe is None:  # pragma: no cover - uninstall path
            return orig(self, req, *args, **kwargs)
        len_entry = getattr(req, "extend_input_len", None)
        res = orig(self, req, *args, **kwargs)
        if res is not None:
            probe.on_chunk(
                _rid_of(req),
                MECH_CONTINUATION,
                len_entry,
                getattr(req, "extend_input_len", None),
                extra={
                    "rem_chunk_tokens_after": getattr(self, "rem_chunk_tokens", None),
                    "page_size": getattr(self, "page_size", None),
                    "n_origin_input_ids": _origin_len(req),
                    "n_prefix_indices": _prefix_len(req),
                    "n_in_batch": len(getattr(self, "can_run_list", ()) or ()),
                },
            )
        return res

    wrapper.__name__ = getattr(orig, "__name__", "wrapped")
    wrapper.__qualname__ = getattr(orig, "__qualname__", "wrapped")
    wrapper.__doc__ = getattr(orig, "__doc__", None)
    wrapper._chunk_probe_wrapped = True  # type: ignore[attr-defined]
    wrapper._chunk_probe_mechanism = MECH_CONTINUATION  # type: ignore[attr-defined]
    return wrapper


def _make_dllm_wrapper(orig):
    """Contamination detector for the diffusion-LLM block truncation (M4)."""

    def wrapper(self, req, *args, **kwargs):
        probe = _PROBE
        if probe is None:  # pragma: no cover - uninstall path
            return orig(self, req, *args, **kwargs)
        len_entry = getattr(req, "extend_input_len", None)
        res = orig(self, req, *args, **kwargs)
        probe.on_chunk(
            _rid_of(req), MECH_DLLM, len_entry,
            getattr(req, "extend_input_len", None), extra=None,
        )
        return res

    wrapper.__name__ = getattr(orig, "__name__", "wrapped")
    wrapper.__qualname__ = getattr(orig, "__qualname__", "wrapped")
    wrapper._chunk_probe_wrapped = True  # type: ignore[attr-defined]
    wrapper._chunk_probe_mechanism = MECH_DLLM  # type: ignore[attr-defined]
    return wrapper


def _make_run_batch_wrapper(orig):
    """Channel 2: extend tokens of every forward, at the scheduler layer."""

    def wrapper(self, batch, *args, **kwargs):
        probe = _PROBE
        if probe is not None:
            try:
                mode = getattr(batch, "forward_mode", None)
                probe.on_forward(
                    getattr(mode, "name", str(mode)),
                    getattr(batch, "extend_num_tokens", None),
                    len(getattr(batch, "reqs", ()) or ()),
                    getattr(batch, "chunked_req", None) is not None,
                )
            except Exception:  # never kill a serving run
                probe._on_error()
        return orig(self, batch, *args, **kwargs)

    wrapper.__name__ = getattr(orig, "__name__", "wrapped")
    wrapper.__qualname__ = getattr(orig, "__qualname__", "wrapped")
    wrapper.__doc__ = getattr(orig, "__doc__", None)
    wrapper._chunk_probe_wrapped = True  # type: ignore[attr-defined]
    return wrapper


def install_wrappers(prefill_adder_cls, scheduler_cls) -> Dict[str, Any]:
    """Install the five wrappers.  Idempotent; returns the originals."""
    global _INSTALLED, _ORIGINALS
    if _INSTALLED:
        return _ORIGINALS
    originals: Dict[str, Any] = {}

    originals["add_one_req"] = prefill_adder_cls.add_one_req
    prefill_adder_cls.add_one_req = _make_admission_wrapper(
        prefill_adder_cls.add_one_req, MECH_ADMISSION_STD
    )

    originals["add_one_req_ignore_eos"] = prefill_adder_cls.add_one_req_ignore_eos
    prefill_adder_cls.add_one_req_ignore_eos = _make_admission_wrapper(
        prefill_adder_cls.add_one_req_ignore_eos, MECH_ADMISSION_IGNORE_EOS
    )

    originals["add_chunked_req"] = prefill_adder_cls.add_chunked_req
    prefill_adder_cls.add_chunked_req = _make_continuation_wrapper(
        prefill_adder_cls.add_chunked_req
    )

    for name in ("_add_dllm_req", "add_dllm_staging_req"):
        fn = getattr(prefill_adder_cls, name, None)
        if fn is not None:
            originals[name] = fn
            setattr(prefill_adder_cls, name, _make_dllm_wrapper(fn))

    if scheduler_cls is not None:
        originals["run_batch"] = scheduler_cls.run_batch
        scheduler_cls.run_batch = _make_run_batch_wrapper(scheduler_cls.run_batch)

    _ORIGINALS = originals
    _INSTALLED = True
    return originals


def uninstall_wrappers(prefill_adder_cls, scheduler_cls) -> None:
    """Restore the originals.  Used by the CPU unit tests, not by the engine."""
    global _INSTALLED, _ORIGINALS
    if not _INSTALLED:
        return
    for name, fn in _ORIGINALS.items():
        target = scheduler_cls if name == "run_batch" else prefill_adder_cls
        if target is not None:
            setattr(target, name, fn)
    _ORIGINALS = {}
    _INSTALLED = False


def set_probe(probe: Optional[ChunkProbe]) -> Optional[ChunkProbe]:
    """Bind the module-level probe the wrappers read.  Returns the previous one."""
    global _PROBE
    prev = _PROBE
    _PROBE = probe
    return prev


def get_probe() -> Optional[ChunkProbe]:
    return _PROBE


def install_shutdown_handlers(probe: ChunkProbe, signums=(signal.SIGTERM,)):
    """Close the probe on ``atexit`` **and** on SIGTERM.  Returns the signums wired.

    WHY BOTH, AND WHY NEITHER IS THE LOAD-BEARING FIX
    -------------------------------------------------
    ``atexit`` alone loses the whole boot: the scheduler process is reaped with
    SIGKILL by its own parent (``sglang/launch_server.py``'s ``finally:
    kill_process_tree(os.getpid(), include_parent=False)`` ->
    ``srt/utils/common.py:1054`` ``child.kill()``), so it never runs an exit
    hook, and Python's default SIGTERM disposition would not run one either.
    That is how CP-0 P1 job 899768 produced two 0-byte probe files.

    The repair that actually covers the SIGKILL case is the writer's interval
    flush (``AsyncJsonlTelemetry.flush_interval_s``).  This handler covers the
    *other* case -- a harness that signals the scheduler directly -- and gives
    the shutdown summary (the roll-up's ``cross_check: "full"``) a chance to be
    written.  Both are needed because they fail in different directions.

    The previous disposition is chained, never replaced: a callable predecessor
    is called, ``SIG_DFL`` is restored and re-raised so the process still dies
    exactly as it would have, and ``SIG_IGN`` is honoured.  Installed only when
    the probe is ON, so the "default OFF means zero added code" property of this
    module is unchanged.

    ★THE CLOSE RUNS ON A HELPER THREAD, ON PURPOSE.  A Python signal handler
    runs *inside* the interrupted thread, and the thread this probe interrupts
    is the one that emits telemetry.  Calling ``close()`` directly would
    re-enter ``queue.Queue.put``; if the interrupted frame was already inside
    ``put`` it holds a non-reentrant lock and the process would deadlock
    against itself, with no timeout, forever.  Handing the close to a helper
    thread and joining it with a bound turns that worst case into "the shutdown
    summary is skipped" -- and the records themselves are already on disk via
    the writer's interval flush, so nothing is lost that this handler was the
    only way to save.
    """
    atexit.register(probe.close)
    wired = []
    for signum in signums:
        try:
            previous = signal.getsignal(signum)
        except (ValueError, OSError):  # pragma: no cover - exotic platforms
            continue

        def handler(sig, frame, _prev=previous):
            try:
                closer = threading.Thread(
                    target=probe.close, name="chunk-probe-close", daemon=True
                )
                closer.start()
                closer.join(timeout=CLOSE_ON_SIGNAL_TIMEOUT_S)
            except Exception:  # pragma: no cover - never block a shutdown
                probe._on_error()
            if callable(_prev):
                return _prev(sig, frame)
            if _prev == signal.SIG_IGN:
                return None
            signal.signal(sig, signal.SIG_DFL)
            os.kill(os.getpid(), sig)
            return None

        try:
            signal.signal(signum, handler)
        except (ValueError, OSError) as exc:
            # signal.signal only works on the main thread.  A probe installed
            # from a worker thread keeps atexit and the interval flush; say so
            # rather than pretending the handler is there.
            logger.warning(
                "chunk probe: could not install a %s handler (%s); relying on "
                "the telemetry interval flush alone", signum, exc
            )
            continue
        wired.append(signum)
    return wired


def _env_flag(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip() not in ("0", "false", "False", "")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


def _piecewise_echo(server_args: Any) -> Dict[str, Any]:
    """P1-g fields: the piecewise CUDA-graph banner, read from server_args."""
    toks = getattr(server_args, "piecewise_cuda_graph_tokens", None)
    out: Dict[str, Any] = {
        "disable_piecewise_cuda_graph": getattr(
            server_args, "disable_piecewise_cuda_graph", None
        ),
        "enforce_piecewise_cuda_graph": getattr(
            server_args, "enforce_piecewise_cuda_graph", None
        ),
        "piecewise_cuda_graph_max_tokens": getattr(
            server_args, "piecewise_cuda_graph_max_tokens", None
        ),
        "piecewise_cuda_graph_compiler": getattr(
            server_args, "piecewise_cuda_graph_compiler", None
        ),
    }
    if toks is None:
        out["piecewise_cuda_graph_tokens_len"] = None
        out["piecewise_cuda_graph_tokens_min"] = None
        out["piecewise_cuda_graph_tokens_max"] = None
    else:
        try:
            vals = list(toks)
        except TypeError:
            vals = []
        out["piecewise_cuda_graph_tokens_len"] = len(vals)
        out["piecewise_cuda_graph_tokens_min"] = min(vals) if vals else None
        out["piecewise_cuda_graph_tokens_max"] = max(vals) if vals else None
    return out


def build_config_echo(scheduler: Any) -> Dict[str, Any]:
    server_args = getattr(scheduler, "server_args", None)
    tree_cache = getattr(scheduler, "tree_cache", None)
    cfg: Dict[str, Any] = {
        # the arm's lever, as requested and as realised
        "requested_chunked_prefill_size": getattr(
            server_args, "chunked_prefill_size", None
        ),
        # Scheduler.init_chunked_prefill maps <= 0 to None (scheduler.py:890-891):
        # None means chunking is OFF for this boot.
        "realized_chunked_prefill_size": getattr(
            scheduler, "chunked_prefill_size", None
        ),
        "page_size": getattr(scheduler, "page_size", None),
        "max_prefill_tokens": getattr(scheduler, "max_prefill_tokens", None),
        "max_running_requests": getattr(scheduler, "max_running_requests", None),
        "enable_mixed_chunk": getattr(server_args, "enable_mixed_chunk", None),
        "enable_dynamic_chunking": getattr(scheduler, "enable_dynamic_chunking", None),
        "disable_radix_cache": getattr(server_args, "disable_radix_cache", None),
        # which admission mechanism can be live: M2 needs tree_cache.disable
        "tree_cache_class": type(tree_cache).__name__ if tree_cache is not None else None,
        "tree_cache_disable": bool(getattr(tree_cache, "disable", False))
        if tree_cache is not None
        else None,
        "truncation_align_size": getattr(scheduler, "truncation_align_size", None),
        # arm identity / graph state
        "enable_pdmux": bool(getattr(scheduler, "enable_pdmux", False)),
        "true_dual_worker": os.environ.get("PDMUX_TRUE_DUAL_WORKER", "0")
        in ("1", "true", "True"),
        "enable_overlap": bool(getattr(scheduler, "enable_overlap", False)),
        "disable_cuda_graph": getattr(server_args, "disable_cuda_graph", None),
        "tp_rank": int(getattr(scheduler, "tp_rank", 0) or 0),
    }
    cfg.update(_piecewise_echo(server_args))
    return cfg


def maybe_install_chunk_probe(scheduler: Any) -> Optional[ChunkProbe]:
    """Return a probe iff ``PDMUX_CHUNK_PROBE_PATH`` is set; otherwise ``None``.

    Default OFF.  When OFF this function is the only code that ever runs, once,
    at scheduler construction: no wrapper is installed, so the admission path
    and ``run_batch`` are the pristine engine functions with no added branch.
    """
    path_str = os.environ.get("PDMUX_CHUNK_PROBE_PATH", "").strip()
    if not path_str:
        return None

    from sglang.srt.managers.schedule_policy import PrefillAdder

    path = Path(path_str)
    tp_rank = int(getattr(scheduler, "tp_rank", 0) or 0)
    tp_size = int(getattr(scheduler, "tp_size", 1) or 1)
    if tp_size > 1:
        path = path.with_name(f"{path.stem}.tp{tp_rank}{path.suffix}")

    cfg = build_config_echo(scheduler)
    probe = ChunkProbe(
        path,
        run_id=os.environ.get(
            "PDMUX_CHUNK_PROBE_RUN_ID", os.environ.get("PDMUX_RUN_ID", "unlabeled")
        ),
        workload_id=os.environ.get(
            "PDMUX_CHUNK_PROBE_WORKLOAD_ID",
            os.environ.get("PDMUX_WORKLOAD_ID", "unlabeled"),
        ),
        arm=os.environ.get("PDMUX_CHUNK_PROBE_ARM", "unlabeled"),
        emit_events=_env_flag("PDMUX_CHUNK_PROBE_EMIT_EVENTS"),
        emit_forwards=_env_flag("PDMUX_CHUNK_PROBE_EMIT_FORWARDS"),
        summary_every_s=_env_float("PDMUX_CHUNK_PROBE_SUMMARY_EVERY_S", 5.0),
        config=cfg,
    )
    set_probe(probe)
    install_wrappers(PrefillAdder, type(scheduler))
    if cfg.get("true_dual_worker"):
        logger.warning(
            "chunk probe: PDMUX_TRUE_DUAL_WORKER is set. Forwards are issued "
            "from two host threads, so the per-forward extend counters "
            "(channel 2) may undercount. Channel 1 is unaffected."
        )
    if cfg.get("enable_mixed_chunk"):
        logger.warning(
            "chunk probe: --enable-mixed-chunk is set, so extend_num_tokens "
            "includes one token per running decode request "
            "(schedule_batch.py:1899). Channel 2 is not 'prefill tokens per "
            "forward' on this boot."
        )
    install_shutdown_handlers(probe)
    logger.info(
        "chunk probe ENABLED (all arms; requested_cps=%s realized_cps=%s "
        "tree_cache=%s disable_piecewise=%s) -> %s",
        cfg["requested_chunked_prefill_size"],
        cfg["realized_chunked_prefill_size"],
        cfg["tree_cache_class"],
        cfg["disable_piecewise_cuda_graph"],
        path,
    )
    return probe
