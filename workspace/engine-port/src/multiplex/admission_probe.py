"""I-5 admission-reason probe (SPEC_common_measurement_v0 sec 9) -- every arm.

WHAT THIS RECORDS
-----------------
For every admission opportunity ("decision") the scheduler takes, the reason
admission STOPPED, plus the shadow state of the three resources that can
bound the running set (KV tokens, the running cap, mamba state slots):

  * decisions inside ``Scheduler.get_new_batch_prefill`` -- shared verbatim
    by the fused loops (``event_loop_normal`` / ``event_loop_overlap`` via
    ``get_next_batch_to_run``) and the PD-mux loop (via
    ``update_split_prefill_batch``);
  * PD-mux pre-gates that return before ``get_new_batch_prefill`` is reached
    (prefill seat occupied, R2 admission latch) -- structurally zero on fused
    arms, which is itself recorded (reachability is per arm, not per probe);
  * retract events (``_add_request_to_queue(..., is_retracted=True)``, the
    only re-queue path of ``update_running_batch``'s retract branch);
  * the running batch size of every decode iteration
    (``update_running_batch`` return value -- shared by both loop families)
    and of every EXTEND / SPLIT_PREFILL batch (R8-4 denominator).

STOP-REASON TAXONOMY (``CATEGORIES``)
-------------------------------------
Rejections, in the spec's vocabulary: ``kv``, ``cap``, ``mamba_slot``,
``prefill_seat``, ``max_prefill_tokens``, ``r2_latch``; plus ``other`` (closed
sub-list: chunk budget, prefill_max_requests, prefill delayer, CP single
request, dLLM, truncation align, TEST_RETRACT) and the two non-rejection
outcomes ``no_waiting`` and ``admitted_all``.  Anything the classifier cannot
reproduce is ``unclassified`` and also raises ``classifier_mismatch`` -- the
classifier is checked against the engine's own ``AddReqResult`` on every call,
so a drift between this mirror and ``schedule_policy.py`` is counted, not
silently mislabelled.

``mamba_slot`` IS NEVER AN ENGINE-FIRING STOP REASON IN SGLang v0.5.10.
The admission path (``PrefillAdder`` / ``_get_new_batch_prefill_raw``) has no
mamba-pool check at all.  Slots are taken later, in
``mem_cache/common.py:alloc_req_slots`` -> ``HybridReqToTokenPool.alloc``,
where exhaustion is a ``RuntimeError`` / ``AssertionError`` (process death),
not a soft rejection.  The bound the slots impose reaches admission only
through the cap predicate's constant: ``_resolve_max_num_reqs`` clamps
``max_running_requests`` to ``max_mamba_cache_size // ratio`` at boot, and with
``--disable-radix-cache`` the req pool and the mamba pool are allocated and
freed together 1:1.  Therefore the ``mamba_slot`` column is reported as a
SHADOW (slot headroom at the stop point) and a BOOT VERDICT
(``cap_slot_relation``), never as a terminal count.  See
``cap_slot_relation`` for when the two are identical by construction.

COST
----
Default OFF: without ``PDMUX_ADMISSION_PATH`` ``maybe_install_admission_probe``
returns None and installs nothing (same convention as chunk_probe.py), so the
pristine engine functions run with no added branch.  When ON, per decision:
a handful of attribute reads and dict increments; per ``add_one_req`` call: a
~12-number snapshot.  Records are WINDOW AGGREGATES (default 1 s) written
through ``AsyncJsonlTelemetry`` (no file I/O on the scheduler thread); only
rare events (retract, invariant violations) are per-event.  Per-EXTEND events
are opt-in (``PDMUX_ADMISSION_EMIT_EXTEND=1``).

The pure part of this module (everything above ``# ---- engine glue``) has
no torch / sglang import so CPU tests can load it as a loose file.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

ADMISSION_SCHEMA = "pdmux.admission/v1"
ADMISSION_SCHEMA_VERSION = 1

# ---------------------------------------------------------------- categories
KV = "kv"
CAP = "cap"
MAMBA_SLOT = "mamba_slot"
PREFILL_SEAT = "prefill_seat"
MAX_PREFILL_TOKENS = "max_prefill_tokens"
R2_LATCH = "r2_latch"
OTHER = "other"
NO_WAITING = "no_waiting"
ADMITTED_ALL = "admitted_all"
UNCLASSIFIED = "unclassified"
CONTINUE = "continue"  # an add that did not stop the loop (never terminal)

REJECTION_CATEGORIES = (KV, CAP, MAMBA_SLOT, PREFILL_SEAT, MAX_PREFILL_TOKENS, R2_LATCH, OTHER)
CATEGORIES = REJECTION_CATEGORIES + (NO_WAITING, ADMITTED_ALL, UNCLASSIFIED)
# Categories that the ENGINE can return as the reason a decision stopped.
# MAMBA_SLOT is deliberately absent (module docstring).
TERMINAL_CATEGORIES = tuple(c for c in CATEGORIES if c != MAMBA_SLOT)

# ``other`` sub-reasons (closed list)
OTHER_SUBS = (
    "chunk_budget",
    "prefill_max_requests",
    "prefill_delayer",
    "cp_single",
    "dllm",
    "trunc_align",
    "test_retract",
)

# AddReqResult names (strings, so this part needs no sglang import)
RES_CONTINUE = "CONTINUE"
RES_NO_TOKEN = "NO_TOKEN"
RES_OTHER = "OTHER"


def ceil_paged(tokens: int, page_size: int) -> int:
    """``PrefillAdder.ceil_paged_tokens`` (schedule_policy.py:499-500)."""
    return -(-int(tokens) // int(page_size)) * int(page_size)


@dataclass
class AddSnapshot:
    """What the classifier needs from one ``PrefillAdder.add_one_req`` call.

    ``rem_*`` are read AFTER the call.  For a rejected request that equals the
    pre-call value, because every rejection in add_one_req returns before
    ``_update_prefill_budget`` mutates the adder; for an admitted request it is
    exactly the state ``budget_state()`` saw.
    """

    added: bool
    n_before: int  # len(can_run_list) before the call
    rem_total: float
    cur_rem: float
    rem_input: int
    rem_chunk: Optional[int]
    extend_input_len: int
    host_hit_length: int = 0
    max_new_tokens: int = 0
    n_output: int = 0
    page_size: int = 1
    clip_max_new_tokens: int = 4096
    ignore_eos_path: bool = False
    delayer_present: bool = False
    cp_single: bool = False
    prefill_max_requests: Optional[int] = None
    dllm: bool = False
    truncation_align_size: Optional[int] = None


def classify_add(snap: AddSnapshot) -> Tuple[str, str, str]:
    """Return ``(category, sub, predicted_AddReqResult_name)``.

    Mirrors ``PrefillAdder.add_one_req`` (schedule_policy.py:728-843) and
    ``add_one_req_ignore_eos`` (:641-726) in engine order.  The caller compares
    ``predicted`` with the engine's actual result; a disagreement is
    ``unclassified`` + ``classifier_mismatch`` (never silently relabelled).
    """
    s = snap
    if s.added:
        # The request is in; the result came from budget_state() (:502-516).
        if s.rem_total <= 0 or s.cur_rem <= 0:
            return KV, "budget_after_add", RES_NO_TOKEN
        if s.rem_input <= 0:
            return MAX_PREFILL_TOKENS, "budget_after_add", RES_OTHER
        if s.dllm:
            # rem_dllm_tokens is not snapshotted; the engine decides.
            return OTHER, "dllm", RES_OTHER
        if s.rem_chunk is not None and s.rem_chunk <= 0:
            return OTHER, "chunk_budget", RES_OTHER
        return CONTINUE, "-", RES_CONTINUE

    # Not added: a pre-mutation rejection, in engine order.
    if s.delayer_present:
        # negotiate_should_allow_prefill (:730-739) runs first and has side
        # effects, so it cannot be re-evaluated; any OTHER is ambiguous.
        return OTHER, "prefill_delayer", RES_OTHER
    if s.cp_single and s.n_before >= 1:
        return OTHER, "cp_single", RES_OTHER
    if s.prefill_max_requests is not None and s.n_before >= s.prefill_max_requests:
        return OTHER, "prefill_max_requests", RES_OTHER
    if s.ignore_eos_path:
        # add_one_req_ignore_eos (:641-726).  NO_TOKEN there is KV in both
        # branches (early size check, then the decode-growth projection).
        ext = ceil_paged(s.extend_input_len, s.page_size)
        if ext > min(s.cur_rem, s.rem_total):
            return KV, "ignore_eos_early", RES_NO_TOKEN
        if s.dllm:
            return OTHER, "dllm", RES_OTHER
        if s.rem_chunk is not None and s.extend_input_len > s.rem_chunk and s.rem_chunk <= 0:
            return OTHER, "chunk_budget", RES_OTHER
        # Remaining NO_TOKEN path: the projection loop (:683-694), which the
        # classifier does not re-run (it needs every running request).
        return KV, "ignore_eos_projection", RES_NO_TOKEN

    total_tokens = s.extend_input_len + min(
        max(s.max_new_tokens - s.n_output, 0), s.clip_max_new_tokens
    )
    if total_tokens >= s.rem_total:
        return KV, "add", RES_NO_TOKEN
    real_input = ceil_paged(s.extend_input_len - s.host_hit_length, s.page_size)
    if real_input >= s.rem_input and s.n_before != 0:
        return MAX_PREFILL_TOKENS, "add", RES_OTHER
    input_tokens = ceil_paged(s.extend_input_len, s.page_size)
    if input_tokens >= s.rem_input and s.n_before != 0:
        return MAX_PREFILL_TOKENS, "add_post_lock", RES_OTHER
    if s.dllm:
        return OTHER, "dllm", RES_OTHER
    if s.rem_chunk is None or input_tokens <= s.rem_chunk:
        # The engine would have admitted it.
        return UNCLASSIFIED, "expected_admit", RES_CONTINUE
    trunc_len = s.rem_chunk // s.page_size * s.page_size
    if trunc_len <= 0:
        return OTHER, "chunk_budget", RES_OTHER
    if s.truncation_align_size is not None and trunc_len < s.truncation_align_size:
        return OTHER, "trunc_align", RES_OTHER
    return UNCLASSIFIED, "expected_chunk_admit", RES_CONTINUE


def cap_fired(can_run_len: int, allocatable: int) -> bool:
    """The loop cap predicate, scheduler.py (pristine :2421) -- note the
    constant inside ``allocatable`` is ``pp_max_micro_batch_size``, not
    ``max_running_requests`` (scheduler.py pristine :2284-2288, :655-659)."""
    return int(can_run_len) >= int(allocatable)


def chunk_guard_fired(allocatable: int, chunked_present: bool, priority_preemption: bool) -> bool:
    """The pre-adder guard, scheduler.py (pristine :2352-2358)."""
    return int(allocatable) <= 0 and bool(chunked_present) and not bool(priority_preemption)


@dataclass
class DecisionTrace:
    """Everything observed during ONE ``get_new_batch_prefill`` call."""

    waiting_n: int = 0
    sticky_full: bool = False
    chunked_present: bool = False
    test_retract: bool = False
    adder_built: bool = False
    guard_fired: bool = False
    # loop events in program order: ("cap", fired) / ("add", category, sub)
    events: List[Tuple] = field(default_factory=list)
    n_add_calls: int = 0


def resolve_terminal(trace: DecisionTrace, last_full_reason: Optional[str]) -> Tuple[str, str, bool]:
    """Why did admission stop?  Returns ``(category, sub, inherited)``.

    The LAST loop event decides, because the engine loop breaks on exactly two
    things -- a fired cap check (``batch_is_full`` -> break) or a non-CONTINUE
    add -- and otherwise runs off the end of the waiting queue.
    ``inherited`` marks an early return on a ``batch_is_full`` flag that an
    EARLIER decision set (the steady state at saturation): it is attributed to
    the reason that set the flag, flagged so an analysis can drop it.
    """
    t = trace
    if t.guard_fired:
        return CAP, "chunked_guard", False
    if not t.adder_built:
        if t.chunked_present:
            # can only happen if TEST_RETRACT returned between guard and adder
            return (OTHER, "test_retract", False) if t.test_retract else (UNCLASSIFIED, "no_adder", False)
        if t.waiting_n == 0:
            return NO_WAITING, "-", False
        if t.sticky_full:
            if last_full_reason in (CAP, KV):
                return last_full_reason, "sticky", True
            return UNCLASSIFIED, "sticky_unknown", True
        if t.test_retract:
            return OTHER, "test_retract", False
        return UNCLASSIFIED, "no_adder", False
    if t.events:
        last = t.events[-1]
        if last[0] == "cap" and last[1]:
            return CAP, "loop", False
        if last[0] == "add" and last[1] != CONTINUE:
            return last[1], last[2], False
    if t.waiting_n == 0:
        return NO_WAITING, "chunk_continuation" if t.chunked_present else "-", False
    return ADMITTED_ALL, "-", False


def sets_batch_full(category: str, sub: str, hierarchical_cache: bool) -> bool:
    """Does this terminal leave ``running_batch.batch_is_full = True``?

    scheduler.py pristine :2357 (guard), :2422 (loop cap), :2457-2464 (NO_TOKEN;
    under hierarchical cache it is conditional, so we do not claim it).
    """
    if category == CAP and sub in ("loop", "chunked_guard"):
        return True
    if category == KV and sub != "sticky" and not hierarchical_cache:
        return True
    return False


def slot_vs_cap(
    mamba_avail: Optional[int], cap_headroom_before: int, n_new: int, n_reuse: int = 0
) -> Optional[str]:
    """Shadow comparison at the stop point: slot headroom vs cap headroom.

    ``cap_headroom_before`` = ``pp_max_micro_batch_size - running_bs`` at entry;
    ``mamba_avail`` = free mamba slots at entry (before this decision's batch
    allocates); ``n_new`` = admitted requests that take a fresh slot;
    ``n_reuse`` = admitted requests that already hold one (the continuing
    chunked request, which holds a slot while it is NOT in running_batch but
    IS counted by the cap predicate through ``can_run_list``).
      identical    -- equal headroom: the two constraints are one constraint
      slot_looser  -- the slot pool cannot bind before the cap does (e.g.
                      overlap-loop frees landing before filter_batch)
      slot_tighter -- slots run out first: the engine would hit the
                      HybridReqToTokenPool.alloc assert, not reject softly
    """
    if mamba_avail is None:
        return None
    slot_h = int(mamba_avail) - int(n_new)
    cap_h = int(cap_headroom_before) - int(n_new) - int(n_reuse)
    if slot_h == cap_h:
        return "identical"
    return "slot_looser" if slot_h > cap_h else "slot_tighter"


def cap_slot_relation(
    *,
    is_hybrid: bool,
    tree_supports_mamba: bool,
    extra_buffer: bool,
    pp_max_micro_batch_size: int,
    req_pool_size: Optional[int],
    mamba_pool_size: Optional[int],
) -> str:
    """Boot verdict: can the mamba-slot constraint be told apart from cap?

    With ``--disable-radix-cache`` every request holds exactly one mamba slot
    and one req slot, both taken in ``alloc_req_slots`` and both freed in
    ``release_kv_cache`` (mem_cache/common.py pristine :297-325, :507-515), and
    ``_resolve_max_num_reqs`` (model_runner_kv_cache_mixin.py pristine
    :843-859) makes the req pool = the effective cap = min(user cap,
    mamba_size // 1).  Hence:

      IDENTICAL         slot count == req pool == cap constant.  "cap" counts
                        are "slot" counts; the split is not identifiable from
                        scheduler state (R16-2 A-11 class identity).
      SLOT_LOOSER       more slots than the cap constant: slots cannot bind.
      SLOT_TIGHTER      fewer slots than the cap constant: the engine would
                        crash at alloc, not reject (should not occur after
                        _resolve_max_num_reqs; reported as a defect).
      RADIX_MAMBA       radix cache manages mamba states (ratio 3, eviction in
                        alloc_req_slots): slots can bind independently of
                        cap, but via eviction pressure, still invisible at
                        admission.  Out of this probe's identification scope.
      EXTRA_BUFFER      mamba extra (ping-pong) buffer: not analysed.
      NOT_HYBRID        no mamba pool.
    """
    if not is_hybrid or mamba_pool_size is None:
        return "NOT_HYBRID"
    if tree_supports_mamba:
        return "RADIX_MAMBA"
    if extra_buffer:
        return "EXTRA_BUFFER"
    cap = int(pp_max_micro_batch_size)
    slots = int(mamba_pool_size)
    if slots == cap and (req_pool_size is None or int(req_pool_size) == cap):
        return "IDENTICAL"
    if slots > cap:
        return "SLOT_LOOSER"
    return "SLOT_TIGHTER"


def cap_source(
    *,
    user_max_running_requests: Optional[int],
    dp_size: int,
    estimated: Optional[int],
    mamba_cap: Optional[int],
    realized: int,
) -> str:
    """Which input set the realized cap (``_resolve_max_num_reqs``)?

    ``mamba_cap`` = ``max_mamba_cache_size // ratio`` (None for non-hybrid).
    Ties are reported jointly (e.g. ``user+mamba``) because then the inputs
    are not separable -- which is the identity case.
    """
    cands = {}
    if user_max_running_requests is not None:
        cands["user"] = int(user_max_running_requests) // max(1, int(dp_size))
    if estimated is not None:
        cands["estimate"] = int(estimated)
    if mamba_cap is not None:
        cands["mamba"] = int(mamba_cap)
    hits = sorted(k for k, v in cands.items() if v == int(realized))
    return "+".join(hits) if hits else "unexplained"


class AdmissionAccountant:
    """Window aggregator.  Pure Python, no clock of its own (caller passes t)."""

    def __init__(self) -> None:
        self.reset_window(0.0)
        self.cum = Counter()
        self.cum_decisions = 0

    def reset_window(self, t: float) -> None:
        self.t_start = float(t)
        self.n_decisions = 0
        self.terminal = Counter()  # "cat/sub" -> n
        self.inherited = Counter()  # cat -> n (subset of terminal)
        self.cobind = Counter()  # "kv" / "cap" / "slot" shadow at a rejection stop
        self.slot_rel = Counter()  # identical / slot_looser / slot_tighter
        self.decode_bs = Counter()  # bs -> n decode iterations
        self.n_decode_at_cap = 0
        self.extend_bs = Counter()  # running_bs at each EXTEND / SPLIT_PREFILL batch
        self.n_extend = 0
        self.n_extend_running_nonempty = 0
        self.n_mixed = 0
        self.n_retract_events = 0
        self.n_retracted_reqs = 0
        self.classifier_mismatch = 0

    def record_decision(self, category: str, sub: str, inherited: bool) -> None:
        key = f"{category}/{sub}"
        self.terminal[key] += 1
        self.cum[category] += 1
        self.cum_decisions += 1
        self.n_decisions += 1
        if inherited:
            self.inherited[category] += 1

    def record_cobind(self, kinds) -> None:
        for k in kinds:
            self.cobind[k] += 1

    def record_slot_relation(self, rel: Optional[str]) -> None:
        if rel is not None:
            self.slot_rel[rel] += 1

    def record_decode(self, bs: int, cap_constant: int) -> None:
        self.decode_bs[int(bs)] += 1
        if int(bs) >= int(cap_constant):
            self.n_decode_at_cap += 1

    def record_extend(self, running_bs: int, mixed: bool) -> None:
        self.n_extend += 1
        self.extend_bs[int(running_bs)] += 1
        if running_bs > 0:
            self.n_extend_running_nonempty += 1
        if mixed:
            self.n_mixed += 1

    def record_retract(self, n_reqs: int) -> None:
        self.n_retract_events += 1
        self.n_retracted_reqs += int(n_reqs)

    def window_record(self, t_end: float) -> Dict[str, Any]:
        n_dec = sum(self.decode_bs.values())
        by_cat = Counter()
        for key, n in self.terminal.items():
            by_cat[key.split("/", 1)[0]] += n
        return {
            "record_schema": ADMISSION_SCHEMA,
            "t_mono_start": self.t_start,
            "t_mono_end": float(t_end),
            "n_decisions": self.n_decisions,
            "terminal_by_category": {c: by_cat.get(c, 0) for c in TERMINAL_CATEGORIES},
            "terminal_by_site": dict(sorted(self.terminal.items())),
            "inherited_by_category": dict(sorted(self.inherited.items())),
            "shadow_cobind": dict(sorted(self.cobind.items())),
            "slot_vs_cap": dict(sorted(self.slot_rel.items())),
            "n_decode_iter": n_dec,
            "n_decode_iter_at_or_above_cap": self.n_decode_at_cap,
            "decode_bs_hist": {str(k): v for k, v in sorted(self.decode_bs.items())},
            "n_extend": self.n_extend,
            "n_extend_running_nonempty": self.n_extend_running_nonempty,
            "n_mixed": self.n_mixed,
            "extend_running_bs_hist": {str(k): v for k, v in sorted(self.extend_bs.items())},
            "n_retract_events": self.n_retract_events,
            "n_retracted_reqs": self.n_retracted_reqs,
            "classifier_mismatch": self.classifier_mismatch,
            "cum_decisions": self.cum_decisions,
            "cum_by_category": dict(sorted(self.cum.items())),
        }


def binding_rate(window_records: List[Dict[str, Any]]) -> Optional[float]:
    """R16-2 b = (decode iterations with running >= cap) / (decode iterations),
    over the windows given (the caller selects the cell window).  None if no
    decode iteration (-> CAP_BINDING_UNRESOLVED upstream)."""
    num = sum(int(r.get("n_decode_iter_at_or_above_cap", 0)) for r in window_records)
    den = sum(int(r.get("n_decode_iter", 0)) for r in window_records)
    if den <= 0:
        return None
    return num / den


def validate_record(rec: Dict[str, Any]) -> List[str]:
    """Schema gate (d): a consumer must refuse records it does not know."""
    errs = []
    if rec.get("record_schema") != ADMISSION_SCHEMA:
        errs.append(f"record_schema {rec.get('record_schema')!r} != {ADMISSION_SCHEMA!r}")
    if rec.get("event") == "admission_window":
        cats = rec.get("terminal_by_category", {})
        if MAMBA_SLOT in cats:
            errs.append("mamba_slot must not appear as a terminal category")
        if sum(cats.values()) != rec.get("n_decisions"):
            errs.append("terminal_by_category does not sum to n_decisions")
    return errs


# ---- engine glue --------------------------------------------------------
# Everything below touches scheduler / adder objects but still imports no
# torch at module scope.

_INSTALLED = False
_ORIGINALS: Dict[str, Tuple[Any, str, Any]] = {}
_PROBE: Optional["AdmissionProbe"] = None


class AdmissionProbe:
    def __init__(
        self,
        path: Optional[Path],
        run_id: str = "unlabeled",
        workload_id: str = "unlabeled",
        window_s: float = 1.0,
        emit_extend: bool = False,
        boot: Optional[Dict[str, Any]] = None,
        static_fields: Optional[Dict[str, Any]] = None,
        clock=time.monotonic,
    ) -> None:
        self.acct = AdmissionAccountant()
        self.window_s = float(window_s)
        self.emit_extend = bool(emit_extend)
        self.boot = dict(boot or {})
        self.clock = clock
        self.cap_constant = int(self.boot.get("pp_max_micro_batch_size") or 0) or 1 << 30
        self.hierarchical_cache = bool(self.boot.get("enable_hierarchical_cache", False))
        self.priority_preemption = bool(self.boot.get("enable_priority_preemption", False))
        self.clip_max_new_tokens = 4096
        self.last_full_reason: Optional[str] = None
        self.trace: Optional[DecisionTrace] = None
        self.current_adder = None
        self._entry_queue: List[Any] = []
        self._entry: Dict[str, Any] = {}
        self._inner_called = False
        self._thread_id: Optional[int] = None
        self.thread_mismatch = 0
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
        self.acct.reset_window(self.clock())
        self._emit("admission_boot", schema_version=ADMISSION_SCHEMA_VERSION,
                   t_realtime=time.time(), t_mono=self.clock(), **self.boot)

    # -- output
    def _emit(self, event: str, **fields: Any) -> None:
        fields.setdefault("record_schema", ADMISSION_SCHEMA)
        if self._tel is not None:
            self._tel.emit(event, "admission", **fields)
        else:
            rec = dict(fields)
            rec["event"] = event
            self.records.append(rec)

    def maybe_flush(self, force: bool = False) -> None:
        now = self.clock()
        if force or now - self.acct.t_start >= self.window_s:
            rec = self.acct.window_record(now)
            rec["t_realtime_end"] = time.time()
            rec["thread_mismatch"] = self.thread_mismatch
            self._emit("admission_window", **rec)
            self.acct.reset_window(now)

    def close(self) -> None:
        self.maybe_flush(force=True)
        if self._tel is not None:
            self._tel.close()

    def _check_thread(self) -> None:
        tid = threading.get_ident()
        if self._thread_id is None:
            self._thread_id = tid
        elif tid != self._thread_id:
            self.thread_mismatch += 1

    # -- decision bracket
    def begin_decision(self, sched: Any) -> None:
        self._check_thread()
        self._inner_called = True
        rb = sched.running_batch
        sticky = bool(getattr(rb, "batch_is_full", False))
        if not sticky:
            self.last_full_reason = None
        running_bs = len(getattr(rb, "reqs", []) or [])
        self.trace = DecisionTrace(
            waiting_n=len(sched.waiting_queue),
            sticky_full=sticky,
            chunked_present=getattr(sched, "chunked_req", None) is not None,
            test_retract=bool(self.boot.get("test_retract", False)),
        )
        self.current_adder = None
        pool = getattr(sched, "req_to_token_pool", None)
        mpool = getattr(pool, "mamba_pool", None)
        # Reference, not a copy: the engine REPLACES waiting_queue after the loop
        # (scheduler.py "Update waiting queue"), it does not mutate this list.
        self._entry_queue = sched.waiting_queue
        self._entry = {
            "running_bs": running_bs,
            "mamba_avail": (mpool.available_size() if mpool is not None else None),
            "cap_headroom": self.cap_constant - running_bs,
        }

    def end_decision(self, sched: Any, new_batch: Any) -> None:
        t = self.trace
        self.trace = None
        if t is None:
            return
        cat, sub, inherited = resolve_terminal(t, self.last_full_reason)
        if cat == UNCLASSIFIED and sub in ("expected_admit", "expected_chunk_admit"):
            self.acct.classifier_mismatch += 1
        self.acct.record_decision(cat, sub, inherited)
        if sets_batch_full(cat, sub, self.hierarchical_cache):
            self.last_full_reason = cat
        adder = self.current_adder
        can_run = list(getattr(adder, "can_run_list", []) or []) if adder is not None else []
        # radix OFF: the only admitted request that already holds a slot is the
        # continuing chunked request, appended by add_chunked_req before the loop.
        n_reuse = 1 if (t.chunked_present and t.adder_built and can_run) else 0
        n_new = len(can_run) - n_reuse
        rel = slot_vs_cap(self._entry.get("mamba_avail"), self._entry["cap_headroom"], n_new, n_reuse)
        self.acct.record_slot_relation(rel)
        if rel == "slot_tighter":
            self._emit("admission_invariant", kind="slot_tighter", **self._entry,
                       n_can_run=len(can_run), n_new=n_new)
        if cat in REJECTION_CATEGORIES and not inherited and cat not in (PREFILL_SEAT, R2_LATCH):
            self.acct.record_cobind(self._cobind(sched, t, cat, adder, len(can_run), n_new))
        if new_batch is not None:
            fm = getattr(new_batch, "forward_mode", None)
            mixed = bool(fm is not None and getattr(fm, "is_mixed", lambda: False)())
            self.acct.record_extend(self._entry["running_bs"], mixed)
            if self.emit_extend:
                self._emit("admission_extend", running_bs=self._entry["running_bs"],
                           n_can_run=len(can_run), mixed=mixed, t_mono=self.clock())
        self.current_adder = None
        self._entry_queue = []
        self.maybe_flush()

    def _cobind(self, sched: Any, t: DecisionTrace, cat: str, adder: Any, n_can_run: int, n_new: int) -> List[str]:
        """Which of the three resources was ALSO exhausted at this stop?

        Shadow, descriptive only.  The engine's first-firing reason is the
        terminal; this says whether a different reason would have stopped the
        same decision (the masking question, e.g. cap firing first hides KV).
          cap  -- running_bs + |can_run| >= pp_max_micro_batch_size
          kv   -- the next waiting request would fail the total-token check
                  (extend estimated as origin+output tokens, radix OFF: no
                  prefix; page rounding ignored) -- approximate
          slot -- free mamba slots minus fresh admissions <= 0
        """
        kinds = []
        if self._entry["cap_headroom"] - n_can_run <= 0:
            kinds.append(CAP)
        if cat == KV:
            kinds.append(KV)
        elif adder is not None:
            try:
                q = self._entry_queue
                if t.n_add_calls < len(q):
                    r = q[t.n_add_calls] if cat == CAP else q[max(0, t.n_add_calls - 1)]
                    ext = len(r.origin_input_ids) + len(r.output_ids)
                    tot = ext + min(max(int(r.sampling_params.max_new_tokens) - len(r.output_ids), 0),
                                    int(self.clip_max_new_tokens))
                    if tot >= float(adder.rem_total_tokens):
                        kinds.append(KV)
            except Exception:  # descriptive field; never break admission
                pass
        m = self._entry.get("mamba_avail")
        if m is not None and int(m) - n_new <= 0:
            kinds.append("slot")
        return kinds

    # -- events inside the bracket
    def on_adder_built(self, adder: Any) -> None:
        if self.trace is not None:
            self.trace.adder_built = True
            self.current_adder = adder

    def on_allocatable(self, sched: Any, allocatable: int) -> None:
        t = self.trace
        if t is None:
            return
        if not t.adder_built:
            if chunk_guard_fired(allocatable, t.chunked_present, self.priority_preemption):
                t.guard_fired = True
            return
        adder = self.current_adder
        n = len(adder.can_run_list) if adder is not None else 0
        t.events.append(("cap", cap_fired(n, allocatable)))

    def on_add(self, adder: Any, req: Any, res_name: str, snap: AddSnapshot) -> None:
        t = self.trace
        if t is None or adder is not self.current_adder:
            return
        t.n_add_calls += 1
        cat, sub, predicted = classify_add(snap)
        if predicted != res_name:
            self.acct.classifier_mismatch += 1
            cat, sub = UNCLASSIFIED, f"mismatch:{predicted}->{res_name}"
        t.events.append(("add", cat, sub))

    # -- outside get_new_batch_prefill
    def on_pdmux_pregate(self, sched: Any, seat_busy: bool, latch_before: bool) -> None:
        if self._inner_called:
            return  # get_new_batch_prefill ran; that decision is already counted
        self._check_thread()
        waiting = len(sched.waiting_queue)
        if waiting == 0:
            self.acct.record_decision(NO_WAITING, "seat_busy" if seat_busy else "pregate", False)
        elif seat_busy:
            self.acct.record_decision(PREFILL_SEAT, "pdmux", False)
        elif latch_before:
            self.acct.record_decision(R2_LATCH, "pdmux", False)
        else:
            self.acct.record_decision(UNCLASSIFIED, "pregate", False)
        self.maybe_flush()

    def on_decode_batch(self, batch: Any) -> None:
        if batch is None:
            return
        reqs = getattr(batch, "reqs", None)
        bs = len(reqs) if reqs is not None else 0
        if bs > 0:
            self.acct.record_decode(bs, self.cap_constant)
        self.maybe_flush()

    def on_retract(self, n_reqs: int, sched: Any) -> None:
        self.acct.record_retract(n_reqs)


def _snapshot(adder: Any, req: Any, n_before: int, added: bool, clip: int,
              truncation_align_size: Optional[int]) -> AddSnapshot:
    sp = req.sampling_params
    return AddSnapshot(
        added=added,
        n_before=n_before,
        rem_total=float(adder.rem_total_tokens),
        cur_rem=float(adder.cur_rem_tokens),
        rem_input=int(adder.rem_input_tokens),
        rem_chunk=adder.rem_chunk_tokens,
        extend_input_len=int(req.extend_input_len),
        host_hit_length=int(getattr(req, "host_hit_length", 0) or 0),
        max_new_tokens=int(sp.max_new_tokens),
        n_output=len(req.output_ids),
        page_size=int(adder.page_size),
        clip_max_new_tokens=int(clip),
        ignore_eos_path=bool(sp.ignore_eos) and bool(getattr(adder.tree_cache, "disable", True)),
        delayer_present=getattr(adder, "prefill_delayer_single_pass", None) is not None,
        cp_single=bool(getattr(adder, "nsa_prefill_cp_in_seq_split", False)
                       or getattr(adder, "prefill_context_parallel_enabled", False)),
        prefill_max_requests=getattr(adder, "prefill_max_requests", None),
        dllm=getattr(adder, "dllm_config", None) is not None,
        truncation_align_size=truncation_align_size,
    )


def install_wrappers(prefill_adder_cls: Any, scheduler_cls: Any, clip_max_new_tokens: int) -> None:
    """Wrap the admission entry points.  Idempotent.  Reads ``_PROBE``."""
    global _INSTALLED
    if _INSTALLED:
        return

    _save = _save_mro

    orig_init = _save(prefill_adder_cls, "__init__")

    def adder_init(self, *a, **kw):
        orig_init(self, *a, **kw)
        p = _PROBE
        if p is not None:
            p.on_adder_built(self)

    prefill_adder_cls.__init__ = adder_init

    orig_add = _save(prefill_adder_cls, "add_one_req")

    def add_one_req(self, req, has_chunked_req, truncation_align_size):
        p = _PROBE
        if p is None or p.trace is None:
            return orig_add(self, req, has_chunked_req, truncation_align_size)
        n_before = len(self.can_run_list)
        res = orig_add(self, req, has_chunked_req, truncation_align_size)
        added = len(self.can_run_list) > n_before and self.can_run_list[-1] is req
        snap = _snapshot(self, req, n_before, added, clip_max_new_tokens, truncation_align_size)
        p.on_add(self, req, getattr(res, "name", str(res)), snap)
        return res

    prefill_adder_cls.add_one_req = add_one_req

    if scheduler_cls is None:
        _INSTALLED = True
        return

    orig_gnbp = _save(scheduler_cls, "get_new_batch_prefill")

    def get_new_batch_prefill(self, *a, **kw):
        p = _PROBE
        if p is None:
            return orig_gnbp(self, *a, **kw)
        p.begin_decision(self)
        ret = None
        try:
            ret = orig_gnbp(self, *a, **kw)
        finally:
            p.end_decision(self, ret)
        return ret

    scheduler_cls.get_new_batch_prefill = get_new_batch_prefill

    orig_alloc = _save(scheduler_cls, "get_num_allocatable_reqs")

    def get_num_allocatable_reqs(self, running_bs):
        res = orig_alloc(self, running_bs)
        p = _PROBE
        if p is not None:
            p.on_allocatable(self, res)
        return res

    scheduler_cls.get_num_allocatable_reqs = get_num_allocatable_reqs

    orig_urb = _save(scheduler_cls, "update_running_batch")

    def update_running_batch(self, batch):
        ret = orig_urb(self, batch)
        p = _PROBE
        if p is not None:
            p.on_decode_batch(ret)
        return ret

    scheduler_cls.update_running_batch = update_running_batch

    orig_q = _save(scheduler_cls, "_add_request_to_queue")

    def _add_request_to_queue(self, req, is_retracted=False):
        r = orig_q(self, req, is_retracted=is_retracted)
        p = _PROBE
        if p is not None and is_retracted:
            p.on_retract(1, self)
        return r

    scheduler_cls._add_request_to_queue = _add_request_to_queue

    if "update_split_prefill_batch" in dir(scheduler_cls):
        orig_uspb = _save(scheduler_cls, "update_split_prefill_batch")

        def update_split_prefill_batch(self, sm_count):
            p = _PROBE
            if p is None:
                return orig_uspb(self, sm_count)
            seat = bool(getattr(self, "split_prefill_batch", None))
            latch = bool(getattr(self, "r2_admission_limited", False))
            p._inner_called = False
            r = orig_uspb(self, sm_count)
            p.on_pdmux_pregate(self, seat, latch)
            return r

        scheduler_cls.update_split_prefill_batch = update_split_prefill_batch

    _INSTALLED = True


def _save_mro(owner: Any, name: str):
    """Save an attribute that may be inherited (mixin); restored by delattr."""
    if name in owner.__dict__:
        _ORIGINALS[f"{owner.__name__}.{name}"] = (owner, name, owner.__dict__[name])
    else:
        _ORIGINALS[f"{owner.__name__}.{name}"] = (owner, name, None)
    return getattr(owner, name)


def uninstall_wrappers() -> None:
    """Restore the originals (CPU tests)."""
    global _INSTALLED
    for owner, name, fn in _ORIGINALS.values():
        if fn is None:
            if name in owner.__dict__:
                delattr(owner, name)
        else:
            setattr(owner, name, fn)
    _ORIGINALS.clear()
    _INSTALLED = False


def set_probe(probe: Optional[AdmissionProbe]) -> Optional[AdmissionProbe]:
    global _PROBE
    prev, _PROBE = _PROBE, probe
    return prev


def get_probe() -> Optional[AdmissionProbe]:
    return _PROBE


def build_boot_record(scheduler: Any) -> Dict[str, Any]:
    """Realized capacities + arm tags.  Every value is read, none assumed."""
    sa = getattr(scheduler, "server_args", None)
    g = None
    try:
        from sglang.srt.server_args import get_global_server_args

        g = get_global_server_args()
    except Exception:
        g = None
    pp_max = getattr(g, "pp_max_micro_batch_size", None) if g is not None else None
    pool = getattr(scheduler, "req_to_token_pool", None)
    mpool = getattr(pool, "mamba_pool", None)
    tree = getattr(scheduler, "tree_cache", None)
    tree_mamba = bool(getattr(tree, "supports_mamba", lambda: False)()) if tree is not None else False
    extra_buffer = False
    if sa is not None and hasattr(sa, "enable_mamba_extra_buffer"):
        try:
            extra_buffer = bool(sa.enable_mamba_extra_buffer())
        except Exception:
            extra_buffer = False
    ratio = 1 if getattr(sa, "disable_radix_cache", False) else None
    mamba_size = getattr(mpool, "size", None) if mpool is not None else None
    mrr = int(getattr(scheduler, "max_running_requests", 0) or 0)
    ctx = getattr(getattr(scheduler, "model_config", None), "context_len", None)
    mtt = getattr(scheduler, "max_total_num_tokens", None)
    estimated = None
    if ctx and mtt:
        estimated = max(min(int(mtt / ctx * 512), 4096), 2048)
    dp = int(getattr(scheduler, "dp_size", 1) or 1)
    boot = {
        "arm_enable_pdmux": bool(getattr(scheduler, "enable_pdmux", False)),
        "arm_enable_overlap": bool(getattr(scheduler, "enable_overlap", False)),
        "arm_chunked_prefill_size": getattr(sa, "chunked_prefill_size", None),
        "arm_enable_mixed_chunk": getattr(sa, "enable_mixed_chunk", None),
        "disable_radix_cache": getattr(sa, "disable_radix_cache", None),
        "tree_cache_class": type(tree).__name__ if tree is not None else None,
        "tree_supports_mamba": tree_mamba,
        "enable_hierarchical_cache": bool(getattr(scheduler, "enable_hierarchical_cache", False)),
        "enable_priority_preemption": bool(getattr(scheduler, "enable_priority_preemption", False)),
        "prefill_delayer": getattr(scheduler, "prefill_delayer", None) is not None,
        "disaggregation_mode": str(getattr(scheduler, "disaggregation_mode", "")),
        "test_retract": _env_flag("SGLANG_TEST_RETRACT"),
        "user_max_running_requests": getattr(sa, "max_running_requests", None),
        "max_running_requests": mrr,
        "pp_max_micro_batch_size": pp_max,
        "pp_size": int(getattr(scheduler, "pp_size", 1) or 1),
        "dp_size": dp,
        "req_pool_size": getattr(pool, "size", None),
        "mamba_pool_size": mamba_size,
        "max_mamba_cache_size": getattr(sa, "max_mamba_cache_size", None),
        "max_total_num_tokens": mtt,
        "max_prefill_tokens": getattr(scheduler, "max_prefill_tokens", None),
        "context_len": ctx,
        "cap_slot_relation": cap_slot_relation(
            is_hybrid=mpool is not None,
            tree_supports_mamba=tree_mamba,
            extra_buffer=extra_buffer,
            pp_max_micro_batch_size=int(pp_max or mrr or 0),
            req_pool_size=getattr(pool, "size", None),
            mamba_pool_size=mamba_size,
        ),
        "cap_source": cap_source(
            user_max_running_requests=getattr(sa, "max_running_requests", None),
            dp_size=dp,
            estimated=estimated,
            mamba_cap=(mamba_size // ratio) if (mamba_size is not None and ratio) else None,
            realized=mrr,
        ),
    }
    return boot


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes")


def maybe_install_admission_probe(scheduler: Any) -> Optional[AdmissionProbe]:
    """Return a probe iff ``PDMUX_ADMISSION_PATH`` is set; otherwise None.

    Default OFF: when unset this is the only code that runs, once, and no
    wrapper is installed.
    """
    path_str = os.environ.get("PDMUX_ADMISSION_PATH", "").strip()
    if not path_str:
        return None
    from sglang.srt.managers.schedule_policy import CLIP_MAX_NEW_TOKENS, PrefillAdder

    path = Path(path_str)
    tp_size = int(getattr(scheduler, "tp_size", 1) or 1)
    dp_size = int(getattr(scheduler, "dp_size", 1) or 1)
    static = {}
    if tp_size > 1 or dp_size > 1:
        from sglang.srt.multiplex.multigpu_guard import rank_fields, rank_suffix

        path = path.with_name(
            f"{path.stem}{rank_suffix(tp_size, getattr(scheduler, 'tp_rank', 0), dp_size, getattr(scheduler, 'dp_rank', None))}{path.suffix}"
        )
        static = rank_fields(tp_size, getattr(scheduler, "tp_rank", 0), dp_size, getattr(scheduler, "dp_rank", None))
    boot = build_boot_record(scheduler)
    probe = AdmissionProbe(
        path,
        run_id=os.environ.get("PDMUX_RUN_ID", "unlabeled"),
        workload_id=os.environ.get("PDMUX_WORKLOAD_ID", "unlabeled"),
        window_s=float(os.environ.get("PDMUX_ADMISSION_WINDOW_S", "1.0") or 1.0),
        emit_extend=_env_flag("PDMUX_ADMISSION_EMIT_EXTEND"),
        boot=boot,
        static_fields=static,
    )
    probe.clip_max_new_tokens = int(CLIP_MAX_NEW_TOKENS)
    set_probe(probe)
    install_wrappers(PrefillAdder, type(scheduler), CLIP_MAX_NEW_TOKENS)
    logger.info(
        "admission probe ENABLED (all arms; cap_slot_relation=%s cap_source=%s) -> %s",
        boot["cap_slot_relation"], boot["cap_source"], path,
    )
    return probe
