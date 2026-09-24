#!/usr/bin/env python3
"""PART-residency census -- DESCRIPTIVE AGGREGATION ONLY.

*** NOT A VERDICT.  NOT WIRED TO ANY DECISION RULE.  NO ARM RANKING. ***
This file answers exactly one descriptive question, per telemetry file:
"how much of the run actually sat in a PARTITIONED green-context state, and
how much of that was co-resident with prefill work?"  It emits numbers and the
convention each number was computed under.  It emits no labels, thresholds or
comparisons.  (Deliberate: an estimator menu and a decision rule that can be
varied together is a dead design here -- cf. the "NOT A LABEL" split between
`e2_realized_mix.py` and `e2_label.py`.)

CANONICAL REUSE (do not re-implement what exists; a gate that copies the code
it checks is an identity).  The sample predicates are IMPORTED from
`../r2_eval/e2_sticky_prereg/e2_realized_mix.py`:
  * `_is_snapshot`    -- event == "runtime_snapshot" and phase != "startup"
  * `_is_decode_busy` -- decode_running_batch_size > 0   (that module's
        ADDENDUM A2 literal; the neighbouring `decode_batch_size` is >= 1 on
        idle spin, and selecting it once flipped an E2 clause)
This module GENERALISES those from "fraction that is stream_index == 2" to
"PART vs FULL over the realized (prefill_sms, decode_sms) pair", and STREAMS
(some archives are single 800 MB files).

REGISTERED CONVENTION NAMES -- a percentage without its convention name is not
citable (E2C-8': one cell read 13.5% or 13.64% depending on convention, and the
convention-less value became citation-banned).

  population A = all non-startup runtime_snapshot records in the file
  population B = population A restricted to `_is_decode_busy`

  R-cnt-all    uniform weight over population A
  R-time-all   forward-gap weight over population A.  Weight of snapshot i is
               t(i+1) - t(i) on the non-startup sequence; the file's LAST
               snapshot has no defined weight and is DROPPED.
  R-cnt-busy   uniform weight over population B         (e2 `E-cnt` family)
  R-time-busy  forward-gap weight, numerator AND denominator restricted to
               decode-busy LEFT snapshots               (e2 `E-time` family)
  R-iter-busy  weight = delta(decode_iterations) between adjacent non-startup
               snapshots, attributed to the LEFT snapshot, gated on the left
               snapshot being decode-busy               (e2 `E-iter` family)

  WHY FIVE.  A count statistic rides a NON-UNIFORM sampling grid (measured by
  the E2 authors: 532 ms vs 208 ms between decode-busy snapshots on one cell),
  and a time-weighted statistic IMPOSES an interval on an INSTANTANEOUS state
  (`CONSENSUS sec 3 item 120`, the C-R confound: 64 instantaneous states x
  0.406 s once inflated a figure to 25.70 s).  Neither is safe alone.  All five
  are reported side by side; disagreement among them is itself the record.

  STATE PREDICATE.  PART <=> prefill_sms > 0 AND decode_sms > 0, read from the
  REALIZED per-snapshot fields, never from a target/config value (lesson 246 /
  Stage 0: the "D108 anchor" was realized 16 SM; the lambda0 "fixed D44" cell
  was realized D44 only a minority of the time).  FULL <=> exactly one side is
  0, i.e. 108/0 or 0/108.  This is a SNAPSHOT-LEVEL PROXY for the engine's
  per-step `split_prefill_batch is not None` condition
  (`multiplexing_mixin.py:880-893`) and cannot resolve states shorter than the
  sampling interval; `median_gap_s` is reported so that limit stays visible.

  COHABITATION, among PART-and-decode-busy snapshots (uniform weight only):
    P-act : prefill_active_batch_size > 0
    P-q   : prefill_queue_depth       > 0
  NOTE (E2C-4): in builds whose controller installs a split index on every
  prefill span, P-act is near-100% BY CONSTRUCTION.  It is an instrument
  reading, not evidence about the engine.

  MULTI-FILE AGGREGATION.  Denominators and spans are SUMMED, never max()'d --
  the max() form was a real harness bug that inflated a duration ~3x.  Pinned
  by `selftest_sum_not_max`.
"""

import argparse
import json
import os
import sys
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "r2_eval", "e2_sticky_prereg"))
import e2_realized_mix as CANON  # noqa: E402  -- canonical predicates live there

IS_SNAPSHOT = CANON._is_snapshot
IS_DECODE_BUSY = CANON._is_decode_busy

CONVENTIONS = ("R_cnt_all", "R_time_all", "R_cnt_busy", "R_time_busy", "R_iter_busy")


def realized_sm(rec):
    return (rec.get("prefill_sms"), rec.get("decode_sms"))


def is_part(rec):
    p, d = realized_sm(rec)
    return bool(p) and bool(d)


def _pct(num, den):
    return (100.0 * num / den) if den else None


def census_file(path, is_busy=IS_DECODE_BUSY, uniform_gap=False, part_pred=is_part,
                track_index=None):
    """One streaming pass.

    `is_busy`/`uniform_gap`/`part_pred` are injection points that exist ONLY so
    `selftest` can perturb exactly one convention at a time.  Production callers
    pass none of them.  `track_index` (int) additionally reports the share of a
    single stream_index, which is what lets the selftest compare against the
    externally-registered D44 table in `e2_realized_mix.EXPECTED`.
    """
    acc = {c: [0.0, 0.0] for c in CONVENTIONS}          # [part_num, den]
    trk = {c: [0.0, 0.0] for c in CONVENTIONS}          # [track_idx_num, den]
    idx_cnt_all, idx_time_all = defaultdict(int), defaultdict(float)
    idx_cnt_busy, idx_time_busy = defaultdict(int), defaultdict(float)
    idx_iter_busy = defaultdict(int)
    sm_cnt_all, sm_time_all = defaultdict(int), defaultdict(float)
    idx_to_sm = {}
    part_busy_n = part_busy_pact = part_busy_pq = 0
    n_all = n_busy = 0
    split_transitions = negative_gaps = 0
    gaps = []
    run_ids, workload_ids, archs = set(), set(), set()
    first_t = last_t = None
    prev = None
    dwell_part, dwell_full = [], []
    run_is_part, run_start_t = None, None

    with open(path, "r", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("event") == "split_transition":
                split_transitions += 1
            if rec.get("run_id"):
                run_ids.add(rec["run_id"])
            if not IS_SNAPSHOT(rec):
                continue
            if rec.get("workload_id"):
                workload_ids.add(rec["workload_id"])
            if rec.get("architecture"):
                archs.add(rec["architecture"])
            t = rec.get("timestamp_monotonic_s")
            i = rec.get("stream_index")
            sm = realized_sm(rec)
            idx_to_sm.setdefault(i, sm)
            p = bool(part_pred(rec))
            busy = bool(is_busy(rec))
            hit_trk = (track_index is not None and i == track_index)

            n_all += 1
            idx_cnt_all[i] += 1
            sm_cnt_all[sm] += 1
            acc["R_cnt_all"][1] += 1
            acc["R_cnt_all"][0] += 1 if p else 0
            trk["R_cnt_all"][1] += 1
            trk["R_cnt_all"][0] += 1 if hit_trk else 0
            if busy:
                n_busy += 1
                idx_cnt_busy[i] += 1
                acc["R_cnt_busy"][1] += 1
                acc["R_cnt_busy"][0] += 1 if p else 0
                trk["R_cnt_busy"][1] += 1
                trk["R_cnt_busy"][0] += 1 if hit_trk else 0
                if p:
                    part_busy_n += 1
                    if (rec.get("prefill_active_batch_size") or 0) > 0:
                        part_busy_pact += 1
                    if (rec.get("prefill_queue_depth") or 0) > 0:
                        part_busy_pq += 1
            if first_t is None:
                first_t = t
            last_t = t

            if prev is not None:
                gap = t - prev["t"]
                if gap < 0:
                    negative_gaps += 1
                gaps.append(gap)
                w = 1.0 if uniform_gap else gap
                idx_time_all[prev["i"]] += w
                sm_time_all[prev["sm"]] += w
                acc["R_time_all"][1] += w
                acc["R_time_all"][0] += w if prev["part"] else 0.0
                trk["R_time_all"][1] += w
                trk["R_time_all"][0] += w if prev["trk"] else 0.0
                if prev["busy"]:
                    idx_time_busy[prev["i"]] += w
                    acc["R_time_busy"][1] += w
                    acc["R_time_busy"][0] += w if prev["part"] else 0.0
                    trk["R_time_busy"][1] += w
                    trk["R_time_busy"][0] += w if prev["trk"] else 0.0
                    di = (rec.get("decode_iterations") or 0) - prev["it"]
                    if di > 0:
                        idx_iter_busy[prev["i"]] += di
                        acc["R_iter_busy"][1] += di
                        acc["R_iter_busy"][0] += di if prev["part"] else 0
                        trk["R_iter_busy"][1] += di
                        trk["R_iter_busy"][0] += di if prev["trk"] else 0

            if run_is_part is None:
                run_is_part, run_start_t = p, t
            elif p != run_is_part:
                (dwell_part if run_is_part else dwell_full).append(t - run_start_t)
                run_is_part, run_start_t = p, t
            prev = {"t": t, "i": i, "sm": sm, "part": p, "busy": busy, "trk": hit_trk,
                    "it": rec.get("decode_iterations") or 0}
    # the final run has no defined end -> dropped, same rule as the final gap

    gs = sorted(gaps)
    out = {
        "path": path,
        "run_ids": sorted(run_ids),
        "workload_ids": sorted(workload_ids),
        "architectures": sorted(archs),
        "snapshots_all": n_all,
        "snapshots_busy": n_busy,
        "span_s": (last_t - first_t) if (first_t is not None and last_t is not None) else None,
        "median_gap_s": (gs[len(gs) // 2] if gs else None),
        "p95_gap_s": (gs[int(0.95 * (len(gs) - 1))] if gs else None),
        "negative_gaps": negative_gaps,
        "split_transition_events": split_transitions,
        "index_to_realized_sm": {str(k): f"{v[0]}/{v[1]}" for k, v in
                                 sorted(idx_to_sm.items(), key=lambda kv: (kv[0] is None, kv[0]))},
        "part_indices": sorted(str(k) for k, v in idx_to_sm.items() if v[0] and v[1]),
        "index_cnt_all": {str(k): v for k, v in sorted(idx_cnt_all.items(), key=lambda kv: (kv[0] is None, kv[0]))},
        "index_time_all_s": {str(k): round(v, 4) for k, v in sorted(idx_time_all.items(), key=lambda kv: (kv[0] is None, kv[0]))},
        "index_cnt_busy": {str(k): v for k, v in sorted(idx_cnt_busy.items(), key=lambda kv: (kv[0] is None, kv[0]))},
        "index_time_busy_s": {str(k): round(v, 4) for k, v in sorted(idx_time_busy.items(), key=lambda kv: (kv[0] is None, kv[0]))},
        "index_iter_busy": {str(k): v for k, v in sorted(idx_iter_busy.items(), key=lambda kv: (kv[0] is None, kv[0]))},
        "realized_sm_cnt_all": {f"{k[0]}/{k[1]}": v for k, v in sorted(sm_cnt_all.items(), key=lambda kv: (str(kv[0])))},
        "realized_sm_time_all_s": {f"{k[0]}/{k[1]}": round(v, 4) for k, v in sorted(sm_time_all.items(), key=lambda kv: (str(kv[0])))},
        "cohab_part_busy_n": part_busy_n,
        "cohab_P_act": {"num": part_busy_pact, "den": part_busy_n, "pct": _pct(part_busy_pact, part_busy_n)},
        "cohab_P_q": {"num": part_busy_pq, "den": part_busy_n, "pct": _pct(part_busy_pq, part_busy_n)},
        "dwell_part_runs": len(dwell_part),
        "dwell_full_runs": len(dwell_full),
        "dwell_part_mean_s": (round(sum(dwell_part) / len(dwell_part), 4) if dwell_part else None),
        "dwell_part_median_s": (round(sorted(dwell_part)[len(dwell_part) // 2], 4) if dwell_part else None),
        "dwell_full_mean_s": (round(sum(dwell_full) / len(dwell_full), 4) if dwell_full else None),
    }
    for c in CONVENTIONS:
        num, den = acc[c]
        out[c] = {"num": round(num, 6), "den": round(den, 6), "pct": _pct(num, den)}
    if track_index is not None:
        out["track_index"] = track_index
        for c in CONVENTIONS:
            num, den = trk[c]
            out["TRK_" + c] = {"num": round(num, 6), "den": round(den, 6), "pct": _pct(num, den)}
    obs_runs = len(dwell_part) + len(dwell_full)
    out["transition_alias_ratio"] = (split_transitions / obs_runs) if obs_runs else None
    return out


# ----------------------------------------------------------------- aggregation
TIME_CONVENTIONS = ("R_time_all", "R_time_busy")


def aggregate(rows, root="", by=("campaign", "architecture", "workload_id")):
    """Group and SUM, never max().

    VALIDITY QUARANTINE.  A file whose snapshot timestamps are not monotonically
    non-decreasing has NO defined forward-gap weight (its `span_s` comes out
    negative), so it is excluded from the two TIME conventions and counted in
    `files_time_axis_nonmonotonic`.  It still contributes to the count- and
    iteration-weighted conventions, for which ordering is irrelevant.  Silently
    averaging such a file is how a weighting convention becomes unciteable.
    """
    g, members = defaultdict(lambda: defaultdict(float)), defaultdict(list)
    for r in rows:
        rel = os.path.relpath(r["path"], root) if root else r["path"]
        camp = rel.split(os.sep)[0]
        parts = {"campaign": camp,
                 "architecture": ",".join(r["architectures"]) or "(none)",
                 "workload_id": ",".join(r["workload_ids"]) or "(none)"}
        k = tuple(parts[b] for b in by)
        members[k].append(rel)
        a = g[k]
        a["files"] += 1
        a["snapshots_all"] += r["snapshots_all"]
        a["snapshots_busy"] += r["snapshots_busy"]
        if (r["span_s"] or 0.0) >= 0:
            a["span_s_sum"] += r["span_s"] or 0.0      # SUM, not max()
        a["split_transition_events"] += r["split_transition_events"]
        a["negative_gaps"] += r["negative_gaps"]
        nonmono = r["negative_gaps"] > 0
        a["files_time_axis_nonmonotonic"] += 1 if nonmono else 0
        for c in CONVENTIONS:
            if nonmono and c in TIME_CONVENTIONS:
                continue
            a[c + "_num"] += r[c]["num"]
            a[c + "_den"] += r[c]["den"]
        a["cohab_part_busy_n"] += r["cohab_part_busy_n"]
        a["cohab_P_act_num"] += r["cohab_P_act"]["num"]
        a["cohab_P_q_num"] += r["cohab_P_q"]["num"]
        a["dwell_part_runs"] += r["dwell_part_runs"]
        a["dwell_full_runs"] += r["dwell_full_runs"]
        if not nonmono:   # dwell SECONDS come from the same time axis as R_time_all
            a["dwell_part_runs_mono"] += r["dwell_part_runs"]
        a["snapshot_busy_share_num"] += r["snapshots_busy"]
    out = []
    for k, a in sorted(g.items()):
        row = dict(zip(by, k))
        row.update({
               "files": int(a["files"]),
               "files_time_axis_nonmonotonic": int(a["files_time_axis_nonmonotonic"]),
               "snapshots_all": int(a["snapshots_all"]),
               "snapshots_busy": int(a["snapshots_busy"]),
               "span_s_sum": round(a["span_s_sum"], 3),
               "negative_gaps": int(a["negative_gaps"]),
               "split_transition_events": int(a["split_transition_events"])})
        for c in CONVENTIONS:
            row[c + "_pct"] = (round(100.0 * a[c + "_num"] / a[c + "_den"], 4)
                               if a[c + "_den"] else None)
            row[c + "_den"] = round(a[c + "_den"], 4)
        n = a["cohab_part_busy_n"]
        row["cohab_part_busy_n"] = int(n)
        row["cohab_P_act_pct"] = round(100.0 * a["cohab_P_act_num"] / n, 4) if n else None
        row["cohab_P_q_pct"] = round(100.0 * a["cohab_P_q_num"] / n, 4) if n else None
        row["dwell_part_runs"] = int(a["dwell_part_runs"])
        row["dwell_full_runs"] = int(a["dwell_full_runs"])
        row["busy_share_of_snapshots_pct"] = (
            round(100.0 * a["snapshot_busy_share_num"] / a["snapshots_all"], 4)
            if a["snapshots_all"] else None)
        row["mean_part_dwell_s_gridlimited"] = (
            round(a["R_time_all_num"] / a["dwell_part_runs_mono"], 4)
            if a["dwell_part_runs_mono"] else None)
        obs = row["dwell_part_runs"] + row["dwell_full_runs"]
        row["transition_alias_ratio"] = round(row["split_transition_events"] / obs, 4) if obs else None
        row["members"] = sorted(members[k])
        out.append(row)
    return out


# --------------------------------------------------------------------- selftest
# Two independent tiers, both with an UNMUTATED CONTROL ARM (lesson 254: mutation
# testing without a no-mutation control cannot tell "the mutation was caught" from
# "the harness always fails"; note that the control still cannot catch a FALSE
# SURVIVAL from an equivalent mutation).
#
#  TIER 1  synthetic fixture, archive-independent.  Expectations are hand-derived
#          from the CONVENTION PROSE in this module's docstring, arithmetic shown
#          inline below, NOT read back from the implementation.
#  TIER 2  the lambda0 908623 archive.  Expectations come from
#          `e2_realized_mix.EXPECTED`, whose own provenance is EXTERNAL verdict
#          documents (see that module's docstring).  This tier is a
#          CROSS-IMPLEMENTATION agreement check: a second, streaming
#          implementation must reproduce the registered D44 table.

_FIX = [
    # (t, idx, prefill_sms, decode_sms, decode_running_batch_size, decode_iterations,
    #  prefill_active_batch_size, prefill_queue_depth)
    (0.0, 0, 108, 0, 0, 0, 0, 0),
    (1.0, 2, 64, 44, 2, 10, 1, 1),
    (3.0, 2, 64, 44, 2, 20, 0, 2),
    (4.0, 4, 0, 108, 3, 35, 0, 0),
    (8.0, 2, 64, 44, 1, 50, 1, 0),
    (10.0, 0, 108, 0, 1, 60, 0, 0),
]
# Hand-derived from the docstring (gaps 1,2,1,4,2; the last snapshot is dropped):
#   R-cnt-all   PART = s1,s2,s4          -> 3/6            = 50.0
#   R-time-all  PART gaps 2+1+2 = 5      -> 5/10           = 50.0
#   R-cnt-busy  busy = s1..s5, PART 3    -> 3/5            = 60.0
#   R-time-busy busy-left gaps 2+1+4+2=9 -> 5/9            = 55.5556
#   R-iter-busy deltas 10,15,15,10 = 50; PART-left 10+15+10=35 -> 70.0
#   cohab P-act 2/3, P-q 2/3            -> 66.6667 each
#   dwell part runs 2 (3.0 s, 2.0 s), full runs 2 (1.0 s, 4.0 s); 5th run dropped
FIX_EXPECT = {
    "snapshots_all": 6, "snapshots_busy": 5, "span_s": 10.0, "median_gap_s": 2.0,
    "R_cnt_all_pct": 50.0, "R_time_all_pct": 50.0, "R_cnt_busy_pct": 60.0,
    "R_time_busy_pct": 55.5556, "R_iter_busy_pct": 70.0,
    "cohab_P_act_pct": 66.6667, "cohab_P_q_pct": 66.6667,
    "dwell_part_runs": 2, "dwell_full_runs": 2,
    "dwell_part_mean_s": 2.5, "dwell_full_mean_s": 2.5,
    "split_transition_events": 3,
}


def _write_fixture(path):
    with open(path, "w") as fh:
        # a startup snapshot that MUST be excluded by `_is_snapshot`
        fh.write(json.dumps({"event": "runtime_snapshot", "phase": "startup",
                             "timestamp_monotonic_s": -1.0, "stream_index": 0,
                             "prefill_sms": 108, "decode_sms": 0,
                             "decode_running_batch_size": 9, "decode_batch_size": 4,
                             "decode_iterations": 0, "run_id": "fixture"}) + "\n")
        for (t, i, ps, ds, drbs, it, pab, pqd) in _FIX:
            fh.write(json.dumps({
                "event": "runtime_snapshot", "phase": "benchmark",
                "timestamp_monotonic_s": t, "stream_index": i,
                "prefill_sms": ps, "decode_sms": ds,
                "decode_running_batch_size": drbs,
                # NOTE: >0 on EVERY record incl. the idle one -- this is exactly the
                # field whose accidental use flipped an E2 clause, so M2 uses it.
                "decode_batch_size": 4,
                "decode_iterations": it, "prefill_active_batch_size": pab,
                "prefill_queue_depth": pqd, "run_id": "fixture",
                "workload_id": "FIXTURE", "architecture": "fixture"}) + "\n")
        for _ in range(3):
            fh.write(json.dumps({"event": "split_transition",
                                 "timestamp_monotonic_s": 5.0}) + "\n")


def _check_fixture(res):
    bad = []
    for k, want in FIX_EXPECT.items():
        got = res[k[:-4]]["pct"] if k.endswith("_pct") and k[:-4] in res else res.get(k)
        if got is None and k.endswith("_pct"):
            got = res.get(k)
        if isinstance(want, float):
            ok = got is not None and abs(round(got, 4) - want) < 1e-4
        else:
            ok = got == want
        if not ok:
            bad.append(f"{k}: got {got!r}, expected {want!r}")
    return bad


def selftest_tier1(verbose=True):
    import tempfile
    tmp = os.path.join(tempfile.mkdtemp(), "fixture.jsonl")
    _write_fixture(tmp)
    arms = [
        ("M0 control (no mutation)", {}, True),
        ("M1 uniform weight replaces forward-gap", {"uniform_gap": True}, False),
        ("M2 decode-busy := decode_batch_size>0",
         {"is_busy": lambda r: (r.get("decode_batch_size") or 0) > 0}, False),
        ("M3 PART := always true", {"part_pred": lambda r: True}, False),
        ("M4 PART := stream_index != 4 (target-ish, ignores 108/0)",
         {"part_pred": lambda r: r.get("stream_index") != 4}, False),
    ]
    fails = []
    for name, kw, should_pass in arms:
        bad = _check_fixture(census_file(tmp, **kw))
        passed = not bad
        verdict = "SURVIVED" if passed else "KILLED"
        if passed != should_pass:
            fails.append(f"{name}: {verdict} but expected "
                         f"{'SURVIVED' if should_pass else 'KILLED'}")
        if verbose:
            print(f"  [{'OK ' if passed == should_pass else 'BAD'}] {verdict:9} {name}"
                  + (f"   ({bad[0]})" if bad else ""))
    # cross-path identity: the PART numerator must equal the sum of the index
    # histogram over the indices whose REALIZED sm pair has both sides non-zero.
    res = census_file(tmp)
    for pop, hist in (("R_cnt_all", "index_cnt_all"), ("R_cnt_busy", "index_cnt_busy")):
        s = sum(v for k, v in res[hist].items() if k in res["part_indices"])
        if abs(res[pop]["num"] - s) > 1e-9:
            fails.append(f"cross-path identity {pop} {res[pop]['num']} != hist {s}")
    return fails


def selftest_sum_not_max(verbose=True):
    """The duration-aggregation gate.  Three rounds of the SAME run must give 3x
    the span; the historical harness bug took max() and under-counted 3x."""
    import tempfile
    tmp = os.path.join(tempfile.mkdtemp(), "fixture.jsonl")
    _write_fixture(tmp)
    r = census_file(tmp)
    agg = aggregate([dict(r), dict(r), dict(r)])
    fails = []
    if abs(agg[0]["span_s_sum"] - 3 * r["span_s"]) > 1e-6:
        fails.append(f"span_s_sum {agg[0]['span_s_sum']} != 3 x {r['span_s']}")
    if abs(agg[0]["span_s_sum"] - r["span_s"]) < 1e-6:
        fails.append("span_s_sum equals a single round -- max() semantics")
    if agg[0]["R_cnt_all_den"] != 3 * r["R_cnt_all"]["den"]:
        fails.append("R_cnt_all denominator was not summed across rounds")
    if verbose:
        print(f"  [{'OK ' if not fails else 'BAD'}] 3 rounds -> span_s_sum="
              f"{agg[0]['span_s_sum']} (single round {r['span_s']}); "
              f"max() would give {r['span_s']}")
    return fails


def selftest_tier2(archive=None, verbose=True):
    """Cross-implementation check against `e2_realized_mix.EXPECTED` (D44 table).
    Returns (fails, status) where status 2 = archive absent (NOT a failure)."""
    archive = archive or os.path.normpath(CANON.ARCHIVE)
    missing = [c for c in CANON.EXPECTED
               if not os.path.exists(os.path.join(archive, f"tel_{c}.jsonl"))]
    if missing:
        if verbose:
            print(f"  [SKIP] archive {archive} lacks {missing} -- tier 2 not run")
        return [], 2
    fails = []
    for cell, exp in CANON.EXPECTED.items():
        e_busy, e_cnt, e_time, e_iter, e_q, e_i0, e_i3, e_st = exp
        got = census_file(os.path.join(archive, f"tel_{cell}.jsonl"),
                          track_index=CANON.D44_STREAM_INDEX)
        checks = [
            ("busy_n", got["snapshots_busy"], e_busy),
            ("E-cnt.num", got["TRK_R_cnt_busy"]["num"], e_cnt[0]),
            ("E-cnt.den", got["TRK_R_cnt_busy"]["den"], e_cnt[1]),
            ("E-time.num_s", round(got["TRK_R_time_busy"]["num"], 2), e_time[0]),
            ("E-time.den_s", round(got["TRK_R_time_busy"]["den"], 2), e_time[1]),
            ("E-iter.pct", round(got["TRK_R_iter_busy"]["pct"], 2), e_iter),
            ("idx0.num", got["index_cnt_busy"].get("0", 0), e_i0[0]),
            ("idx3.num", got["index_cnt_busy"].get("3", 0), e_i3),
            ("split_transition", got["split_transition_events"], e_st),
        ]
        bad = [f"{cell}.{n}: got {a!r}, registered {w!r}" for n, a, w in checks if a != w]
        fails += bad
        if verbose:
            print(f"  [{'OK ' if not bad else 'BAD'}] {cell:9} busy={got['snapshots_busy']:5d} "
                  f"D44 E-cnt={got['TRK_R_cnt_busy']['pct']:6.2f}%  "
                  f"E-time={got['TRK_R_time_busy']['pct']:6.2f}%  "
                  f"E-iter={got['TRK_R_iter_busy']['pct']:6.2f}%  "
                  f"| PART R-time-all={got['R_time_all']['pct']:6.2f}%")
    # resolution guard (lesson 232): an estimator that reproduces a table but
    # cannot separate the two registered regimes is an identity.
    a = census_file(os.path.join(archive, "tel_a_r4.jsonl"), track_index=2)["TRK_R_cnt_busy"]["pct"]
    b = census_file(os.path.join(archive, "tel_b_r3.jsonl"), track_index=2)["TRK_R_cnt_busy"]["pct"]
    if not (a < 20.0 < b):
        fails.append(f"resolution guard: a_r4={a:.2f}% and b_r3={b:.2f}% do not straddle 20%")
    return fails, 0


def selftest(archive=None):
    print("# TIER 1 -- synthetic fixture, mutation arms (M0 is the no-mutation control)")
    f1 = selftest_tier1()
    print("# TIER 1b -- multi-round aggregation must SUM, not max()")
    f1b = selftest_sum_not_max()
    print("# TIER 2 -- cross-implementation vs e2_realized_mix.EXPECTED (external provenance)")
    f2, status = selftest_tier2(archive)
    fails = f1 + f1b + f2
    if fails:
        print("\nSELFTEST_FAILED")
        for x in fails:
            print("  " + x)
        return 1
    print("\nSELFTEST_OK" + ("  (tier 2 SKIPPED: archive absent)" if status == 2 else ""))
    return 0


# -------------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--archive", help="lam0_908623 directory for tier 2")
    ap.add_argument("--scan-root", help="results/ root to census")
    ap.add_argument("--file-list", help="newline-separated telemetry paths")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--out-json"), ap.add_argument("--out-csv"), ap.add_argument("--out-files-json")
    args = ap.parse_args()

    if args.selftest:
        return selftest(args.archive)
    if not args.file_list:
        ap.error("--file-list required unless --selftest")
    paths = [p.strip() for p in open(args.file_list) if p.strip()]
    root = args.scan_root or ""
    if root:
        paths = [p if os.path.isabs(p) else os.path.join(root, p) for p in paths]

    from multiprocessing import Pool
    with Pool(args.jobs) as pool:
        rows = pool.map(_safe, paths, chunksize=1)
    rows = [r for r in rows if r]
    agg = aggregate(rows, root=root)
    if args.out_files_json:
        json.dump(rows, open(args.out_files_json, "w"), indent=1, sort_keys=True)
    if args.out_json:
        json.dump({"_banner": BANNER, "conventions": list(CONVENTIONS),
                   "n_files": len(rows), "groups": agg},
                  open(args.out_json, "w"), indent=1, sort_keys=True)
    if args.out_csv:
        _write_csv(args.out_csv, agg)
        _write_csv(args.out_csv.replace(".csv", "_by_campaign.csv"),
                   aggregate(rows, root=root, by=("campaign",)))
        json.dump({"_banner": BANNER,
                   "by_campaign": sm_rollup(rows, root=root, by=("campaign",)),
                   "by_group": sm_rollup(rows, root=root,
                                         by=("campaign", "architecture", "workload_id"))},
                  open(args.out_csv.replace(".csv", "_realized_sm.json"), "w"),
                  indent=1, sort_keys=True)
    print(f"censused {len(rows)} files into {len(agg)} groups")
    return 0


def sm_rollup(rows, root="", by=("campaign",)):
    """REALIZED (prefill_sms, decode_sms) distribution -- never the target.

    Two conventions again: `cnt` (uniform over population A) and `time`
    (forward-gap over population A, non-monotonic files excluded).  A cell whose
    configured split does not dominate this table is the `target != realized`
    situation that has bitten this repo three times (Stage 0's "D108" anchor was
    realized 16 SM; the lambda0 "fixed D44" cell realized D44 only a minority).
    """
    cnt, tim = defaultdict(lambda: defaultdict(int)), defaultdict(lambda: defaultdict(float))
    for r in rows:
        rel = os.path.relpath(r["path"], root) if root else r["path"]
        parts = {"campaign": rel.split(os.sep)[0],
                 "architecture": ",".join(r["architectures"]) or "(none)",
                 "workload_id": ",".join(r["workload_ids"]) or "(none)"}
        k = tuple(parts[b] for b in by)
        for sm, v in (r.get("realized_sm_cnt_all") or {}).items():
            cnt[k][sm] += v
        if r["negative_gaps"] == 0:
            for sm, v in (r.get("realized_sm_time_all_s") or {}).items():
                tim[k][sm] += v
    out = []
    for k in sorted(set(cnt) | set(tim)):
        tc, tt = sum(cnt[k].values()), sum(tim[k].values())
        out.append({
            **dict(zip(by, k)),
            "snapshots": tc, "weighted_s": round(tt, 3),
            "sm_share_cnt_pct": {sm: round(100.0 * v / tc, 4) for sm, v in
                                 sorted(cnt[k].items(), key=lambda kv: -kv[1])} if tc else {},
            "sm_share_time_pct": {sm: round(100.0 * v / tt, 4) for sm, v in
                                  sorted(tim[k].items(), key=lambda kv: -kv[1])} if tt else {},
        })
    return out


def _write_csv(path, agg):
    cols = [k for k in agg[0] if k != "members"] if agg else []
    with open(path, "w") as fh:
        fh.write(",".join(cols) + "\n")
        for r in agg:
            fh.write(",".join("" if r[c] is None else str(r[c]) for c in cols) + "\n")


BANNER = ("DESCRIPTIVE REFERENCE AGGREGATION ONLY -- not a verdict, not wired to any "
          "decision rule, no arm ranking, GPU spend 0, 0 new performance claims.")


def _safe(p):
    try:
        return census_file(p)
    except Exception as exc:  # a single corrupt archive must not void the census
        return {"path": p, "ERROR": repr(exc), "run_ids": [], "workload_ids": [],
                "architectures": [], "snapshots_all": 0, "snapshots_busy": 0,
                "span_s": 0.0, "median_gap_s": None, "p95_gap_s": None,
                "negative_gaps": 0, "split_transition_events": 0,
                "cohab_part_busy_n": 0,
                "cohab_P_act": {"num": 0, "den": 0, "pct": None},
                "cohab_P_q": {"num": 0, "den": 0, "pct": None},
                "dwell_part_runs": 0, "dwell_full_runs": 0,
                **{c: {"num": 0, "den": 0, "pct": None} for c in CONVENTIONS}}


if __name__ == "__main__":
    sys.exit(main())
