#!/usr/bin/env python3
"""Roll one boot's chunk-probe JSONL up into the CP-0 boot record.

PREREG_CP0_2026-08-28.md section 4.1 requires channel 1 to be reported "probe 3점
각각에서 따로" -- separately at each registered rate -- because "backlog가 없으면
배치 채널은 잠들어 있고, 그 사실 자체가 결과다".  The engine is deliberately never
told what rate it is serving: the probe stamps every record with `host_ts` and this
script bins those stamps into the windows the harness recorded.

INPUTS
    --jsonl     the probe output (PDMUX_CHUNK_PROBE_PATH of one boot)
    --segments  JSON written by the harness:
                {"segments":[{"segment_id":"rate3","rate":3,
                              "t_start":<unix s>,"t_end":<unix s>}, ...], ...}
                Any extra keys are copied through into the boot record.

OUTPUT (stdout, or --out)
    one JSON object per boot with the schema PREREG section 2 asks for:
    run_id / arm / realized_chunked_prefill_size / per-segment
    {n_requests_chunked, n_chunk_events, extend_tokens_per_forward:{max,mean,
    n_forwards}} / disable_piecewise_cuda_graph / piecewise_cuda_graph_tokens.

INTEGRITY, NOT DECORATION
    The probe writes each counter twice by two independent paths: the per-event
    lines, and the accountant's own in-memory totals in the shutdown summary.
    This script recomputes the boot totals from the event lines and compares
    them with the summary.  A mismatch, or any dropped telemetry event, sets
    `status = "EXTRACTION_UNRELIABLE"`, which under PREREG section 2.3 is
    failure class (ii) EXTRACTION_EMPTY -- a defect of the instrument, not a
    finding about the arm.

NOT ADDITIVE, ON PURPOSE
    `n_requests_chunked` counts DISTINCT rids, so the per-segment values do not
    have to sum to the boot value: a request may be chunked in two segments.
    The script reports both and asserts only what is true, rather than
    manufacturing an identity.

    ★2026-09-01.  "Only what is true" was previously enforced in ONE direction:
    the script complained when `sum(per-segment) < boot` and said nothing when
    the sum came out larger.  Job 899768 walked straight through that hole --
    its segments summed to 8 distinct rids against a boot total of 5, and to 9
    chunk events against 6, and the record said `status: OK`.  The cause was a
    real defect (`p1_accept_workload.py` registers `t_end + 1.0`, so the phase-S
    window overlapped the start of phase B and every event in the overlap was
    counted twice), and it is exactly the kind of thing the roll-up exists to
    catch.  The checks are now stated the way the quantities actually behave:

      * the registered windows must be DISJOINT.  Overlapping windows make
        every additive per-segment number ambiguous, so the overlap itself is
        the problem, reported before any count is interpreted;
      * ADDITIVE counters (chunk events, forwards, token sums) over disjoint
        windows must satisfy `sum(per-segment) <= boot`.  More than the boot
        total means double counting;
      * DISTINCT-rid counts are compared as SETS, not as a sum: the union of
        the per-segment rid sets must equal the boot's rid set (a shortfall
        means the windows do not cover the traffic; an excess would mean a
        segment saw an rid the boot roll-up did not, which is impossible).
        `sum(per-segment) > |union|` is the legitimate case the old comment was
        about -- one request chunked in two windows -- and it is now reported as
        a counted warning instead of being the reason the check was one-sided.

    ★2026-09-01, repair 1.  Channel 2 was reading the WRONG POPULATION.  The
    probe emitted one `prefill_forward` line per forward whose
    `extend_num_tokens` was positive, but `ScheduleBatch` does not clear that
    field when the batch object is reused for a decode step: on job 899768's
    cps512 boot 101 of the 116 lines had `forward_mode == "DECODE"`, each
    carrying the extend count of the preceding prefill (`mode=DECODE,
    extend_num_tokens=7, batch_size=1`, verbatim).  `extend_tokens_per_forward`
    was therefore not "prefill tokens per prefill forward", and the
    pre-registered P1-e numbers -- which are per prefill forward -- were being
    compared with a different quantity (phase S: expected 7 forwards, "got 43").

    The reader now classifies every forward line by `forward_mode` and counts
    the prefill modes only.  This is deliberately done on BOTH sides: the probe
    filters what it counts, and this script re-derives the classification from
    the mode written on each line, so that pre-repair artifacts -- in which a
    decode forward appears under the `prefill_forward` event name -- are re-read
    correctly instead of inheriting the defect.

    Nothing is deleted.  The excluded lines are counted
    (`n_nonprefill_forwards_with_extend_tokens`), their tokens are summed
    (`sum_extend_tokens_not_counted`), the raw per-mode counts stay in
    `forward_mode_counts`, and the exclusion is a warning on the record.  A
    line whose mode is in neither set is `unknown`: not counted into channel 2
    (a future mode could be another stale-field decode) and not dropped either
    (that would hide the hole), and warned about, because a nonzero unknown
    count means this reader's mode partition is behind the engine's enum.

This script makes no performance statement and ranks nothing (PREREG section 9).
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

SCHEMA = "cp0.chunk-boot-record/v1"

# The writer's schema.  Bumped by repair 1 (2026-09-01) because the meaning of
# the channel-2 counters changed: a v1 artifact's accountant counted every
# forward with a positive `extend_num_tokens`, including decode steps holding a
# stale value.  The reader below uses this to tell "the archived totals were
# computed with the old rule" (expected, and only ever an OVERcount) from "the
# two passes over the same event stream disagree" (an instrument defect).
WRITER_SCHEMA_CURRENT = "pdmux.chunk-probe/v2"
WRITER_SCHEMA_PRE_REPAIR = "pdmux.chunk-probe/v1"

# -- which forwards channel 2 counts (repair 1, 2026-09-01) ------------------
#
# `ScheduleBatch.extend_num_tokens` is not cleared when the batch object is
# reused for a decode step, so `extend_num_tokens > 0` does not mean "this was
# a prefill": the field is written by `prepare_for_extend`
# (`managers/schedule_batch.py:1612`) and zeroed only by `prepare_for_idle`
# (:2049), while `prepare_for_decode` (:2062) sets the mode and leaves the
# extend count standing (verified 2026-09-01).  Job 899768's cps512 boot: 101 of 116 `prefill_forward` lines had
# `forward_mode == "DECODE"`, each carrying the extend count of the preceding
# prefill, and the pre-registered P1-e numbers -- counts of PREFILL forwards --
# were being compared with a different quantity.
#
# The classification is by `forward_mode`, on the READER side as well, because
# the reader must also be able to re-read pre-repair artifacts in which such a
# forward was written under the `prefill_forward` event name.  This is a second
# copy of the sets in `sglang/srt/multiplex/chunk_probe.py` (this script runs
# offline, on hosts where the engine need not be installed); the copies are
# pinned to each other by tests/test_cp0_p1_tools.py and to the engine's own
# `ForwardMode.is_extend` by tests/test_chunk_probe.py.
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
CLASS_UNKNOWN = "unknown"

FORWARD_EVENTS = ("prefill_forward", "nonprefill_forward")

# Cap on the recorded per-forward token multiset.  The CP-0 phases have single
# digits of prefill forwards and the multiset is how the hand derivation in
# p1e_expectations.json is checked; a long boot would otherwise put an unbounded
# list in the record.  Truncation is flagged, never silent.
_MULTISET_CAP = 4096


def classify_forward_mode(mode_name):
    """Name of a ForwardMode (or None) -> prefill / nonprefill / unknown."""
    if mode_name is None:
        return CLASS_UNKNOWN
    name = str(mode_name)
    if name in PREFILL_FORWARD_MODES:
        return CLASS_PREFILL
    if name in NONPREFILL_FORWARD_MODES:
        return CLASS_NONPREFILL
    return CLASS_UNKNOWN

# Fields echoed once per boot in the `chunk_open` record.  P1-g requires the last
# three; the rest identify the arm's lever and which admission mechanism is live.
CONFIG_FIELDS = (
    "requested_chunked_prefill_size",
    "realized_chunked_prefill_size",
    "page_size",
    "max_prefill_tokens",
    "max_running_requests",
    "enable_mixed_chunk",
    "enable_dynamic_chunking",
    "disable_radix_cache",
    "tree_cache_class",
    "tree_cache_disable",
    "truncation_align_size",
    "enable_pdmux",
    "enable_overlap",
    "disable_cuda_graph",
    "tp_rank",
    "true_dual_worker",
    "disable_piecewise_cuda_graph",
    "enforce_piecewise_cuda_graph",
    "piecewise_cuda_graph_max_tokens",
    "piecewise_cuda_graph_compiler",
    "piecewise_cuda_graph_tokens_len",
    "piecewise_cuda_graph_tokens_min",
    "piecewise_cuda_graph_tokens_max",
)


def read_jsonl(path):
    rows = []
    bad = 0
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
    return rows, bad


def _blank_bucket():
    return {
        "n_chunk_events": 0,
        "n_chunk_events_admission": 0,
        "n_chunk_events_continuation": 0,
        "n_dllm_trunc_events": 0,
        "mech_counts": Counter(),
        "rids": set(),
        "n_forwards": 0,
        "sum_extend": 0,
        "max_extend": None,
        "min_extend": None,
        "forward_modes": Counter(),
        "n_nonprefill_lines": 0,
        "n_nonprefill_lines_with_ext": 0,
        "n_unknown_lines": 0,
        "sum_extend_not_counted": 0,
        "multiset": [],
        "multiset_truncated": 0,
    }


def _finish_bucket(b):
    mean = b["sum_extend"] / b["n_forwards"] if b["n_forwards"] else None
    return {
        "n_requests_chunked": len(b["rids"]),
        "n_chunk_events": b["n_chunk_events"],
        "n_chunk_events_admission": b["n_chunk_events_admission"],
        "n_chunk_events_continuation": b["n_chunk_events_continuation"],
        "n_dllm_trunc_events": b["n_dllm_trunc_events"],
        "mech_counts": dict(b["mech_counts"]),
        "extend_tokens_per_forward": {
            "max": b["max_extend"],
            "min": b["min_extend"],
            "mean": mean,
            "sum": b["sum_extend"],
            "n_forwards": b["n_forwards"],
            # the values themselves, so a hand derivation can be compared
            # against the observation and not only against its total
            "observed_multiset": list(b["multiset"]),
            "observed_multiset_truncated": b["multiset_truncated"],
        },
        # Forwards this window saw that channel 2 did NOT count, and the tokens
        # they claimed.  Excluded from the statistics, never dropped from the
        # record: on a pre-repair artifact these are the stale-field forwards
        # that used to be counted as prefill.
        "n_nonprefill_forwards": b["n_nonprefill_lines"],
        "n_nonprefill_forwards_with_extend_tokens": b["n_nonprefill_lines_with_ext"],
        "n_unknown_mode_forwards": b["n_unknown_lines"],
        "sum_extend_tokens_not_counted": b["sum_extend_not_counted"],
        # Every forward mode on every forward line in this window, counted as
        # written.  Recorded, never corrected.
        "forward_mode_counts": dict(b["forward_modes"]),
    }


def _add_chunk(b, row):
    mech = row.get("mechanism")
    b["mech_counts"][mech] += 1
    if mech == "dllm":
        b["n_dllm_trunc_events"] += 1
        return
    b["n_chunk_events"] += 1
    if mech == "continuation":
        b["n_chunk_events_continuation"] += 1
    else:
        b["n_chunk_events_admission"] += 1
    rid = row.get("rid")
    if rid is not None:
        b["rids"].add(rid)


def _add_forward(b, row):
    """File one forward line.  Channel 2 counts PREFILL forwards only.

    The event NAME is not trusted: pre-repair artifacts wrote decode forwards
    carrying a stale `extend_num_tokens` under `prefill_forward`, which is the
    defect this classification exists to undo, so the mode is what decides.
    """
    ext = row.get("extend_num_tokens")
    if ext is None:
        return
    ext = int(ext)
    mode = row.get("forward_mode")
    b["forward_modes"][mode] += 1
    mode_class = classify_forward_mode(mode)
    if mode_class == CLASS_UNKNOWN:
        b["n_unknown_lines"] += 1
    # Both halves of the writer's predicate: a prefill MODE and a positive
    # extend count.  Dropping the second half would let a zero-token line into
    # the statistics (it would become the `min`) and the reader would no longer
    # be running the same rule as the probe.
    if mode_class != CLASS_PREFILL or ext <= 0:
        b["n_nonprefill_lines"] += 1
        if ext > 0:
            b["n_nonprefill_lines_with_ext"] += 1
            b["sum_extend_not_counted"] += ext
        return
    b["n_forwards"] += 1
    b["sum_extend"] += ext
    b["max_extend"] = ext if b["max_extend"] is None else max(b["max_extend"], ext)
    b["min_extend"] = ext if b["min_extend"] is None else min(b["min_extend"], ext)
    if len(b["multiset"]) < _MULTISET_CAP:
        b["multiset"].append(ext)
    else:
        b["multiset_truncated"] += 1


def build_record(rows, segments_doc, bad_lines=0):
    segs = list(segments_doc.get("segments", []))
    opens = [r for r in rows if r.get("event") == "chunk_open"]
    summaries = [r for r in rows if r.get("event") == "chunk_summary"]
    shutdown = [r for r in summaries if r.get("phase") == "shutdown"]
    chunk_events = [r for r in rows if r.get("event") == "chunk_event"]
    # BOTH forward events.  A pre-repair artifact has only `prefill_forward`,
    # and some of those lines are decode forwards; `_add_forward` classifies by
    # mode, so which event a line arrived under does not change the counting.
    forwards = [r for r in rows if r.get("event") in FORWARD_EVENTS]

    problems = []
    if not opens:
        problems.append("no chunk_open record: the probe never started")
    if bad_lines:
        problems.append(f"{bad_lines} unparseable JSONL line(s)")

    cfg = {k: opens[0].get(k) if opens else None for k in CONFIG_FIELDS}

    writer_schema = None
    for r in (opens + summaries):
        if r.get("schema_chunk"):
            writer_schema = r["schema_chunk"]
            break
    pre_repair_writer = writer_schema == WRITER_SCHEMA_PRE_REPAIR

    boot_b = _blank_bucket()
    for r in chunk_events:
        _add_chunk(boot_b, r)
    for r in forwards:
        _add_forward(boot_b, r)
    boot_from_events = _finish_bucket(boot_b)

    seg_out, seg_buckets = [], []
    for s in segs:
        b = _blank_bucket()
        t0, t1 = float(s["t_start"]), float(s["t_end"])
        for r in chunk_events:
            ts = r.get("host_ts")
            if ts is not None and t0 <= float(ts) < t1:
                _add_chunk(b, r)
        for r in forwards:
            ts = r.get("host_ts")
            if ts is not None and t0 <= float(ts) < t1:
                _add_forward(b, r)
        rec = dict(s)
        rec.update(_finish_bucket(b))
        seg_out.append(rec)
        seg_buckets.append(b)

    # cross-check: the accountant's own totals vs. the per-event stream.
    #
    # The shutdown summary is written from an atexit hook.  If the scheduler
    # process is SIGKILLed the hook does not run, and that is a plumbing
    # artifact, not evidence about the counters -- calling it an instrument
    # defect would be the mislabelling PROJECT_STATUS.md "방법론 게이트" #21
    # forbids.  So a missing shutdown summary downgrades the cross-check to the
    # last periodic summary (monotone: it can only be behind the event stream)
    # and is reported as a WARNING; only an actual disagreement is a problem.
    warnings = []
    summary_totals = None
    cross_check = "none"
    last_summary = shutdown[-1] if shutdown else (summaries[-1] if summaries else None)
    if last_summary is not None:
        partial = not shutdown
        cross_check = "full" if shutdown else "partial_no_shutdown_summary"
        if partial:
            warnings.append(
                "no shutdown summary (the scheduler did not run its atexit "
                "hook); cross-checking against the last periodic summary only"
            )
        sd = last_summary
        summary_totals = {
            "n_requests_chunked": sd.get("n_requests_chunked"),
            "n_chunk_events": sd.get("n_chunk_events"),
            "n_chunk_events_admission": sd.get("n_chunk_events_admission"),
            "n_chunk_events_continuation": sd.get("n_chunk_events_continuation"),
            "n_dllm_trunc_events": sd.get("n_dllm_trunc_events"),
            "extend_tokens_per_forward": sd.get("extend_tokens_per_forward"),
            "forward_mode_counts": sd.get("forward_mode_counts"),
            "n_forwards_total": sd.get("n_forwards_total"),
            "n_events_without_shrink": sd.get("n_events_without_shrink"),
            "n_events_unknown_rid": sd.get("n_events_unknown_rid"),
            "rid_set_truncated": sd.get("rid_set_truncated"),
            "n_errors": sd.get("n_errors"),
        }
        for key in ("n_requests_chunked", "n_chunk_events",
                    "n_chunk_events_admission", "n_chunk_events_continuation"):
            got, want = summary_totals[key], boot_from_events[key]
            if got == want:
                continue
            msg = (f"{key}: event stream says {want}, accountant says {got}")
            if partial and got is not None and got <= want:
                warnings.append(msg + " (partial cross-check: the periodic "
                                       "summary predates the last events)")
            else:
                problems.append(msg)
        se = summary_totals["extend_tokens_per_forward"] or {}
        be = boot_from_events["extend_tokens_per_forward"]
        for key in ("n_forwards", "sum", "max"):
            got, want = se.get(key), be.get(key)
            if got == want:
                continue
            msg = (f"extend_tokens_per_forward.{key}: event stream says "
                   f"{want}, accountant says {got}")
            if partial and got is not None and want is not None and got <= want:
                warnings.append(msg + " (partial cross-check)")
            elif (
                pre_repair_writer
                and got is not None
                and want is not None
                and got >= want
            ):
                # A pre-repair (v1) accountant counted every forward with a
                # positive extend count, decode steps holding a stale value
                # included, so its channel-2 totals are an OVERcount of what
                # this reader now counts.  That is a known property of the
                # archived instrument, not a disagreement between two passes
                # over the same stream, and labelling it an instrument defect
                # would be the mislabelling PROJECT_STATUS.md "방법론 게이트" #21
                # forbids.  The amnesty is narrow ON PURPOSE: it applies only to
                # the v1 schema, only to channel 2 (channel 1 is untouched by
                # the repair), and only in the direction the old rule could err.
                warnings.append(
                    msg + " (pre-repair writer: the archived accountant counted "
                    "non-prefill forwards into channel 2; re-read totals are "
                    "computed from the event stream)"
                )
            else:
                problems.append(msg)
        if summary_totals["n_errors"]:
            problems.append(f"probe reported {summary_totals['n_errors']} internal error(s)")
        if summary_totals["rid_set_truncated"]:
            problems.append("rid set hit its cap: n_requests_chunked is a lower bound")
    else:
        problems.append(
            "no summary record at all: the accountant's totals cannot be "
            "cross-checked against the event stream"
        )

    dropped = max(
        [int(r.get("dropped_events") or 0) for r in rows] or [0]
    )
    if dropped:
        problems.append(f"{dropped} telemetry event(s) dropped by the async writer")

    # Configuration caveats that change what channel 2 means.  Surfaced as
    # warnings, not corrections: the probe records what the engine did.
    if cfg.get("enable_mixed_chunk"):
        warnings.append(
            "--enable-mixed-chunk: extend_num_tokens includes one token per "
            "running decode request (schedule_batch.py:1899), so channel 2 is "
            "not 'prefill tokens per forward' on this boot"
        )
    if cfg.get("true_dual_worker"):
        warnings.append(
            "PDMUX_TRUE_DUAL_WORKER: forwards are issued from two host threads, "
            "so channel 2 may undercount (channel 1 is unaffected)"
        )
    if cfg.get("enable_dynamic_chunking"):
        warnings.append(
            "--enable-dynamic-chunking: the per-batch chunk budget is not the "
            "boot's chunked_prefill_size (`managers/scheduler.py:2394-2400`.  "
            "Until 2026-09-01 this cited lines 2383-2387 of the same file, "
            "which are the priority-queue call and the TEST_RETRACT branch; "
            "the stale numbers are written without backticks so the citation "
            "checker does not read them as a live citation)"
        )

    # -- segment <-> boot consistency, in BOTH directions --------------------
    # See "NOT ADDITIVE, ON PURPOSE" in the module docstring.  Until 2026-09-01
    # only the shortfall direction was checked, and job 899768's overlapping
    # windows (segment sums of 8 rids / 9 events against boot totals of 5 / 6)
    # passed silently.
    seg_consistency = None
    if seg_out:
        # (a) the windows themselves.  Everything below assumes disjointness.
        overlaps = []
        ordered = sorted(
            range(len(seg_out)), key=lambda i: float(seg_out[i]["t_start"])
        )
        for a, b_ix in zip(ordered, ordered[1:]):
            a_end = float(seg_out[a]["t_end"])
            b_start = float(seg_out[b_ix]["t_start"])
            if b_start < a_end:
                overlaps.append({
                    "a": seg_out[a].get("segment_id"),
                    "b": seg_out[b_ix].get("segment_id"),
                    "seconds": a_end - b_start,
                })
        if overlaps:
            problems.append(
                "registered segment windows overlap "
                + "; ".join(
                    f"{o['a']} ends {o['seconds']:.3f}s after {o['b']} starts"
                    for o in overlaps
                )
                + " -- every event in the overlap is counted in two segments, so "
                "no per-segment number means what it says"
            )

        # (b) additive counters: disjoint windows are subsets of the boot.
        additive = {
            "n_chunk_events": lambda s: s["n_chunk_events"],
            "n_chunk_events_admission": lambda s: s["n_chunk_events_admission"],
            "n_chunk_events_continuation": lambda s: s["n_chunk_events_continuation"],
            "n_dllm_trunc_events": lambda s: s["n_dllm_trunc_events"],
            "n_prefill_forwards": lambda s: s["extend_tokens_per_forward"]["n_forwards"],
            "sum_extend_tokens": lambda s: s["extend_tokens_per_forward"]["sum"],
        }
        boot_additive = {
            "n_chunk_events": boot_from_events["n_chunk_events"],
            "n_chunk_events_admission": boot_from_events["n_chunk_events_admission"],
            "n_chunk_events_continuation":
                boot_from_events["n_chunk_events_continuation"],
            "n_dllm_trunc_events": boot_from_events["n_dllm_trunc_events"],
            "n_prefill_forwards":
                boot_from_events["extend_tokens_per_forward"]["n_forwards"],
            "sum_extend_tokens":
                boot_from_events["extend_tokens_per_forward"]["sum"],
        }
        sums = {k: sum(f(s) for s in seg_out) for k, f in additive.items()}
        for key, got in sums.items():
            want = boot_additive[key]
            if got > want:
                problems.append(
                    f"{key}: the segments sum to {got} but the boot saw {want}; "
                    "an additive counter cannot exceed the whole it partitions "
                    "(overlapping or double-counted windows)"
                )

        # (c) distinct rids: compared as SETS, which is what they are.
        seg_rid_union = set()
        for b in seg_buckets:
            seg_rid_union |= b["rids"]
        boot_rids = boot_b["rids"]
        seg_rid_sum = sum(s["n_requests_chunked"] for s in seg_out)
        missing = boot_rids - seg_rid_union
        extra = seg_rid_union - boot_rids
        if missing:
            problems.append(
                f"{len(missing)} chunked rid(s) fall in no registered window; "
                "the segments do not cover the served traffic"
            )
        if extra:
            problems.append(
                f"{len(extra)} rid(s) appear in a segment but not in the boot "
                "roll-up; the two passes over the same event stream disagree"
            )
        n_multi = seg_rid_sum - len(seg_rid_union)
        if n_multi > 0 and not overlaps:
            warnings.append(
                f"{n_multi} rid-segment pair(s) beyond the first: a request was "
                "chunked in more than one window, so the per-segment distinct-rid "
                "counts legitimately sum to more than the boot total"
            )

        # (d) coverage, reported as numbers rather than judged: the greedy set
        #     and the flush ping land outside every window on purpose.
        def _outside(rows):
            n = 0
            for r in rows:
                ts = r.get("host_ts")
                if ts is None:
                    continue
                ts = float(ts)
                if not any(
                    float(s["t_start"]) <= ts < float(s["t_end"]) for s in seg_out
                ):
                    n += 1
            return n

        seg_consistency = {
            "windows_overlap": overlaps,
            "additive_sums": sums,
            "additive_boot": boot_additive,
            "n_distinct_rids_boot": len(boot_rids),
            "n_distinct_rids_union_of_segments": len(seg_rid_union),
            "sum_of_per_segment_distinct_rids": seg_rid_sum,
            "n_rid_segment_pairs_beyond_the_first": n_multi,
            "n_chunk_events_outside_every_segment": _outside(chunk_events),
            "n_prefill_forwards_outside_every_segment": _outside([
                r for r in forwards
                if classify_forward_mode(r.get("forward_mode")) == CLASS_PREFILL
                and (r.get("extend_num_tokens") or 0) > 0
            ]),
            "n_forward_lines_outside_every_segment": _outside(forwards),
        }

    # Forward lines channel 2 did not count, and why.  Reported with their
    # counts and their token sum: the exclusion is a repair (2026-09-01), and a
    # repair that cannot be seen in the artifact cannot be audited.
    mode_counts = boot_from_events.get("forward_mode_counts") or {}
    n_excluded = boot_from_events["n_nonprefill_forwards_with_extend_tokens"]
    if n_excluded:
        excluded_modes = {
            m: n for m, n in mode_counts.items()
            if classify_forward_mode(m) != CLASS_PREFILL
        }
        warnings.append(
            f"{n_excluded} forward line(s) whose forward_mode is not a prefill "
            f"mode carried extend tokens ({excluded_modes}, "
            f"{boot_from_events['sum_extend_tokens_not_counted']} tokens in "
            "total); ScheduleBatch does not clear extend_num_tokens on a decode "
            "step, so these are excluded from channel 2 and kept in the raw data"
        )
    n_unknown = boot_from_events["n_unknown_mode_forwards"]
    if n_unknown:
        unknown_modes = {
            m: n for m, n in mode_counts.items()
            if classify_forward_mode(m) == CLASS_UNKNOWN
        }
        warnings.append(
            f"{n_unknown} forward line(s) carry an unknown forward mode "
            f"({unknown_modes}); they are counted separately and excluded from "
            "channel 2 -- if this is nonzero the reader's mode partition is "
            "behind the engine's ForwardMode enum"
        )

    return {
        "schema": SCHEMA,
        "status": "OK" if not problems else "EXTRACTION_UNRELIABLE",
        "problems": problems,
        "warnings": warnings,
        "cross_check": cross_check,
        "writer_schema": writer_schema,
        "writer_is_pre_repair": pre_repair_writer,
        "run_id": opens[0].get("run_id") if opens else None,
        "workload_id": opens[0].get("workload_id") if opens else None,
        "arm": opens[0].get("arm") if opens else None,
        **cfg,
        "piecewise_cuda_graph_tokens": {
            "len": cfg["piecewise_cuda_graph_tokens_len"],
            "min": cfg["piecewise_cuda_graph_tokens_min"],
            "max": cfg["piecewise_cuda_graph_tokens_max"],
        },
        "segments": seg_out,
        "segment_consistency": seg_consistency,
        "boot_total_from_events": boot_from_events,
        "boot_total_from_accountant": summary_totals,
        "health": {
            "dropped_events": dropped,
            "n_periodic_summaries": len(summaries) - len(shutdown),
            "n_chunk_event_lines": len(chunk_events),
            "n_forward_lines": len(forwards),
            "n_prefill_forward_lines":
                boot_from_events["extend_tokens_per_forward"]["n_forwards"],
            "n_nonprefill_forward_lines":
                boot_from_events["n_nonprefill_forwards"],
            "n_unknown_mode_forward_lines":
                boot_from_events["n_unknown_mode_forwards"],
            "bad_jsonl_lines": bad_lines,
        },
        "harness": {k: v for k, v in segments_doc.items() if k != "segments"},
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--segments", default=None,
                    help="harness segment windows; omit to report the boot only")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    rows, bad = read_jsonl(args.jsonl)
    segdoc = json.loads(Path(args.segments).read_text()) if args.segments else {}
    rec = build_record(rows, segdoc, bad_lines=bad)
    text = json.dumps(rec, indent=2, sort_keys=False)
    if args.out:
        Path(args.out).write_text(text + "\n")
    else:
        sys.stdout.write(text + "\n")
    return 0 if rec["status"] == "OK" else 3


if __name__ == "__main__":
    sys.exit(main())
