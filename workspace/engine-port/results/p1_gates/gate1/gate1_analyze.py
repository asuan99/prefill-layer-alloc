#!/usr/bin/env python3
"""Gate 1 realized-partition analysis (see PREREG_GATE1_2026-08-06.md).

Answers ONE question: for agnostic v1 (Zamba2-2.7B, cudagraph-ON, job-873944
replica), in the population "decode-busy AND prefill in-flight", what
TIME-WEIGHTED fraction of that time has realized (prefill_sms, decode_sms) in
{(74,34), (54,54)}? Reports the same statistic, TIME- and COUNT-weighted, for
two negative-control populations (decode-idle; decode-busy-but-prefill-idle),
plus cadence/episode diagnostics needed to judge whether a PASS is meaningful
or a near-identity of the code path (see PREREG sec "게이트가 항등식인가").

NO latency/throughput/goodput number is read or printed by this script --
only `runtime_snapshot` telemetry fields. This script makes NO performance
claim; it only prints GATE1_* lines and a final verdict per the pre-registered
decision rule. Exit status is always 0.

★2026-08-06 COVERAGE GUARD (addendum to PREREG_GATE1_2026-08-06.md, added
after job 874465 revealed that a telemetry writer-thread backlog can silently
truncate the TAIL of a window while `dropped_events` stays 0 the whole time --
the counter is carried BY the very events that get dropped once the queue is
permanently full, so it cannot observe its own failure. Job 874465's rate=2
window looked "fully captured" (rows present throughout, `dropped_events==0`
in every row) but was in fact only 80.1% covered: telemetry stopped 22.96s
before the window's own t1, i.e. exactly the highest-load tail where a
`(54,54)` transition was most likely. This is a VALIDITY PRECONDITION, not a
decision-rule change (methodology gate #8): the >=0.90 threshold, population
definitions, aggregation unit, and target sets are all unchanged. It can only
make the verdict STRICTER (push toward UNMEASURABLE), never push a result
toward UNLOCK -- a window with good coverage is scored exactly as before; a
window with bad coverage that would otherwise have been scored is now
correctly flagged as not measured, rather than being scored as if it were
complete.
"""
import json
import sys

TARGET_A = {(74, 34), (54, 54)}   # decode-busy AND prefill in-flight -> expect these
TARGET_B = {(108, 0)}             # decode-idle -> expect full-prefill fallback (Sec 1-22)
TARGET_C = {(0, 108)}             # decode-busy, prefill NOT in-flight -> expect full-decode
MIN_TOTAL_S = 0.5
MIN_EPISODES = 2
MIN_COVERAGE = 0.98               # see module docstring "COVERAGE GUARD"


def load_rows(path, t0=None, t1=None):
    rows = []
    for line in open(path):
        try:
            e = json.loads(line)
        except Exception:
            continue
        if e.get("event") != "runtime_snapshot" or e.get("phase") != "benchmark":
            continue
        ts = e.get("timestamp_monotonic_s")
        if ts is None:
            continue
        if t0 is not None and ts < t0:
            continue
        if t1 is not None and ts > t1:
            continue
        rows.append(e)
    rows.sort(key=lambda e: e["timestamp_monotonic_s"])
    return rows


def compute_coverage(rows, t0, t1):
    """telemetry_coverage = (min(last_row_ts,t1) - max(first_row_ts,t0)) /
    (t1-t0). Measures what SPAN of the requested window is actually spanned
    by recorded telemetry, regardless of density within that span -- this is
    exactly the statistic that catches a truncated TAIL (rows present but
    stopping early), which a per-row `dropped_events` counter cannot (see
    module docstring). Returns None if not computable (no window bounds, or
    non-positive window span, or no rows) -- callers must treat None as
    "coverage indeterminate", not as passing."""
    if t0 is None or t1 is None or not rows:
        return None
    span = t1 - t0
    if span <= 0:
        return None
    first_ts = rows[0]["timestamp_monotonic_s"]
    last_ts = rows[-1]["timestamp_monotonic_s"]
    covered = min(last_ts, t1) - max(first_ts, t0)
    return max(0.0, covered) / span


def classify(e):
    """Returns 'A', 'B', or 'C' per PREREG population definitions."""
    p_active = (e.get("prefill_active_batch_size", 0) or 0) > 0
    d_active = (e.get("decode_running_batch_size", 0) or 0) > 0
    if not d_active:
        return "B"
    if p_active:
        return "A"
    return "C"


def analyze_window(rows, t1_bound):
    """One pass over time-ordered rows -> per-population (time-weighted hist,
    count-weighted hist, total_time, n_snapshots, n_episodes, sample_index
    deltas within the population, for cadence diagnostics)."""
    pops = {k: dict(time_hist={}, count_hist={}, t_total=0.0, n=0,
                     episodes=0, in_episode=False, sample_idx_gaps=[])
            for k in ("A", "B", "C")}
    prev_sample_idx = {"A": None, "B": None, "C": None}

    n = len(rows)
    for i, e in enumerate(rows):
        ts = e["timestamp_monotonic_s"]
        nxt = rows[i + 1]["timestamp_monotonic_s"] if i + 1 < n else t1_bound
        if nxt is None:
            continue
        dt = max(0.0, nxt - ts)
        pop = classify(e)
        d = pops[pop]
        key = (e.get("prefill_sms"), e.get("decode_sms"))
        d["time_hist"][key] = d["time_hist"].get(key, 0.0) + dt
        d["count_hist"][key] = d["count_hist"].get(key, 0) + 1
        d["t_total"] += dt
        d["n"] += 1
        si = e.get("sample_index")
        if si is not None and prev_sample_idx[pop] is not None:
            d["sample_idx_gaps"].append(si - prev_sample_idx[pop])
        if si is not None:
            prev_sample_idx[pop] = si
        d["in_episode"] = True

    # episode count: maximal contiguous runs of the SAME population label
    # over the FULL (unfiltered) row sequence -- must walk all rows in order,
    # not the per-population subsets, to see true adjacency.
    cur_pop = None
    for e in rows:
        pop = classify(e)
        if pop != cur_pop:
            pops[pop]["episodes"] += 1
            cur_pop = pop
    return pops


def frac_in(hist, total_time, target_set):
    if total_time <= 0:
        return float("nan")
    hit = sum(t for k, t in hist.items() if k in target_set)
    return hit / total_time


def fmt_hist(hist, total, topn=6):
    if not total:
        return "(none)"
    items = sorted(hist.items(), key=lambda kv: -kv[1])[:topn]
    return ", ".join(f"P{p}D{d}:{v:.3f}s({v/total*100:.0f}%)" for (p, d), v in items)


def median(xs):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    n = len(xs)
    mid = n // 2
    return xs[mid] if n % 2 else (xs[mid - 1] + xs[mid]) / 2.0


def print_pop_block(label, pop_name, d, target_set):
    tw = frac_in(d["time_hist"], d["t_total"], target_set)
    cw_total = sum(d["count_hist"].values())
    cw = (sum(c for k, c in d["count_hist"].items() if k in target_set) / cw_total
          ) if cw_total else float("nan")
    gap_med = median(d["sample_idx_gaps"])
    print(f"GATE1_POP {label}({pop_name}): n_snapshots={d['n']} n_episodes={d['episodes']} "
          f"t_total_s={d['t_total']:.3f} target={sorted(target_set)}")
    print(f"GATE1_FRAC {label} time_weighted={tw:.4f} count_weighted={cw:.4f} "
          f"(count secondary only, force-prefill biases it -- see PREREG)")
    print(f"GATE1_HIST {label} (time-weighted): {fmt_hist(d['time_hist'], d['t_total'])}")
    print(f"GATE1_CADENCE {label}: median_sample_index_gap={gap_med} "
          f"n_gaps={len(d['sample_idx_gaps'])} (event-loop iterations skipped between "
          f"recorded snapshots WITHIN this population -- large values = coarse cadence, "
          f"see S0/S2 sampling-cadence mechanism)")
    return dict(time_weighted=tw, count_weighted=cw, t_total=d["t_total"],
                n_episodes=d["episodes"], n_snapshots=d["n"])


def severity(v):
    """Strictness order used for the stricter-wins rule (PREREG rule 5),
    generalized to any 'UNMEASURABLE(...)' variant (e.g. partial coverage,
    empty window) -- all UNMEASURABLE-prefixed labels are equally strict
    (severity 1), same as the module docstring's coverage-guard requirement."""
    if v == "NO_UNLOCK":
        return 0
    if isinstance(v, str) and v.startswith("UNMEASURABLE"):
        return 1
    return 2  # UNLOCK / UNLOCK-eligible


def main():
    argv = sys.argv[1:]
    if not argv:
        print("usage: gate1_analyze.py <telemetry.jsonl> --window RATE T0 T1 [...]")
        sys.exit(0)
    telemetry_path = argv[0]
    windows = []  # (rate, t0, t1)
    i = 1
    while i < len(argv):
        if argv[i] == "--window":
            windows.append((argv[i + 1], float(argv[i + 2]), float(argv[i + 3])))
            i += 4
        else:
            i += 1
    if not windows:
        windows = [("ALL", None, None)]

    pooled_A = dict(time_hist={}, t_total=0.0)
    pooled_B = dict(time_hist={}, t_total=0.0)
    pooled_C = dict(time_hist={}, t_total=0.0)
    pooled_episodes = dict(A=0, B=0, C=0)
    pooled_n = dict(A=0, B=0, C=0)
    per_rate_verdict_inputs = []  # (rate, a-or-None, coverage-or-None)

    for rate, t0, t1 in windows:
        rows = load_rows(telemetry_path, t0, t1)
        cov = compute_coverage(rows, t0, t1)
        cov_str = f"{cov*100:.2f}%" if cov is not None else "N/A(no window bounds or empty)"
        first_ts = rows[0]["timestamp_monotonic_s"] if rows else None
        last_ts = rows[-1]["timestamp_monotonic_s"] if rows else None
        missing_tail_s = (t1 - last_ts) if (t1 is not None and last_ts is not None) else None
        print(f"GATE1_COVERAGE rate={rate}: telemetry_coverage={cov_str} "
              f"(window=[{t0},{t1}] span={(t1-t0) if (t0 is not None and t1 is not None) else None}, "
              f"first_row_ts={first_ts}, last_row_ts={last_ts}, "
              f"missing_tail_s={missing_tail_s}) -- ALWAYS printed regardless of pass/fail "
              f"(2026-08-06 coverage-guard addendum, see module docstring)")
        print(f"=== GATE1 window rate={rate} t0={t0} t1={t1} n_rows={len(rows)} ===")
        if not rows:
            print(f"GATE1_EMPTY_WINDOW rate={rate} -- no benchmark-phase runtime_snapshot rows")
            per_rate_verdict_inputs.append((rate, None, cov))
            continue
        pops = analyze_window(rows, t1)
        a = print_pop_block(f"rate{rate}_A", "decode-busy AND prefill in-flight", pops["A"], TARGET_A)
        b = print_pop_block(f"rate{rate}_B", "decode-idle", pops["B"], TARGET_B)
        c = print_pop_block(f"rate{rate}_C", "decode-busy, prefill NOT in-flight", pops["C"], TARGET_C)
        for k, v in pops["A"]["time_hist"].items():
            pooled_A["time_hist"][k] = pooled_A["time_hist"].get(k, 0.0) + v
        pooled_A["t_total"] += pops["A"]["t_total"]
        for k, v in pops["B"]["time_hist"].items():
            pooled_B["time_hist"][k] = pooled_B["time_hist"].get(k, 0.0) + v
        pooled_B["t_total"] += pops["B"]["t_total"]
        for k, v in pops["C"]["time_hist"].items():
            pooled_C["time_hist"][k] = pooled_C["time_hist"].get(k, 0.0) + v
        pooled_C["t_total"] += pops["C"]["t_total"]
        pooled_episodes["A"] += pops["A"]["episodes"]
        pooled_episodes["B"] += pops["B"]["episodes"]
        pooled_episodes["C"] += pops["C"]["episodes"]
        pooled_n["A"] += pops["A"]["n"]
        pooled_n["B"] += pops["B"]["n"]
        pooled_n["C"] += pops["C"]["n"]
        per_rate_verdict_inputs.append((rate, a, cov))

    print("=== GATE1 POOLED (sum of durations across windows, time-weighted) ===")
    print("GATE1_POOLED_COVERAGE_NOTE: pooled histograms below are computed from "
          "WHATEVER rows exist per window (same as before the coverage guard) -- "
          "the coverage guard acts on the PER-RATE VERDICT (via the stricter-wins "
          "rule), not by silently dropping/reweighting pooled data. A pooled "
          "number sitting on a partial-coverage window is a biased estimate of "
          "the full window and must be read together with GATE1_VERDICT, which "
          "will already reflect any coverage failure.")
    tw_A = frac_in(pooled_A["time_hist"], pooled_A["t_total"], TARGET_A)
    tw_B = frac_in(pooled_B["time_hist"], pooled_B["t_total"], TARGET_B)
    tw_C = frac_in(pooled_C["time_hist"], pooled_C["t_total"], TARGET_C)
    print(f"GATE1_POOLED_A: t_total_s={pooled_A['t_total']:.3f} n_episodes={pooled_episodes['A']} "
          f"n_snapshots={pooled_n['A']} frac_target={tw_A:.4f} hist={fmt_hist(pooled_A['time_hist'], pooled_A['t_total'])}")
    print(f"GATE1_POOLED_B: t_total_s={pooled_B['t_total']:.3f} n_episodes={pooled_episodes['B']} "
          f"n_snapshots={pooled_n['B']} frac_target={tw_B:.4f} hist={fmt_hist(pooled_B['time_hist'], pooled_B['t_total'])}")
    print(f"GATE1_POOLED_C: t_total_s={pooled_C['t_total']:.3f} n_episodes={pooled_episodes['C']} "
          f"n_snapshots={pooled_n['C']} frac_target={tw_C:.4f} hist={fmt_hist(pooled_C['time_hist'], pooled_C['t_total'])}")

    # --- Pre-registered verdict (PREREG_GATE1_2026-08-06.md "사전등록 판정 규칙") ---
    def unmeasurable(pop_t_total, pop_episodes):
        return pop_t_total < MIN_TOTAL_S or pop_episodes < MIN_EPISODES

    pooled_verdict = None
    reasons = []
    if unmeasurable(pooled_A["t_total"], pooled_episodes["A"]):
        pooled_verdict = "UNMEASURABLE"
        reasons.append(f"pooled A t_total_s={pooled_A['t_total']:.3f} (<{MIN_TOTAL_S}) or "
                        f"n_episodes={pooled_episodes['A']} (<{MIN_EPISODES})")
    else:
        a_pass = tw_A >= 0.90
        b_pass = (tw_B >= 0.90) if pooled_B["t_total"] > 0 else True
        c_pass = (tw_C >= 0.90) if pooled_C["t_total"] > 0 else True
        if a_pass and b_pass and c_pass:
            pooled_verdict = "UNLOCK"
        else:
            pooled_verdict = "NO_UNLOCK"
            if not a_pass:
                reasons.append(f"pooled A frac_target={tw_A:.4f} < 0.90")
            if not b_pass:
                reasons.append(f"pooled B (decode-idle) frac_target={tw_B:.4f} < 0.90 "
                                "-> negative control failed, A pass invalidated per PREREG")
            if not c_pass:
                reasons.append(f"pooled C (prefill-idle) frac_target={tw_C:.4f} < 0.90 "
                                "-> negative control failed, A pass invalidated per PREREG")

    # per-rate verdict, NOW GATED ON COVERAGE FIRST (2026-08-06 addendum):
    # a window with coverage < MIN_COVERAGE is UNMEASURABLE(partial coverage)
    # regardless of what its (possibly truncated, hence biased) A/B/C stats
    # would otherwise have said -- this is checked BEFORE the pre-existing
    # t_total/n_episodes/frac checks, and is strictly stricter (can only ADD
    # UNMEASURABLE verdicts, never remove one or produce UNLOCK-eligible where
    # the old logic would not have).
    per_rate_verdicts = []
    for rate, a, cov in per_rate_verdict_inputs:
        if a is None:
            per_rate_verdicts.append((rate, "UNMEASURABLE(empty window)"))
            continue
        if cov is not None and cov < MIN_COVERAGE:
            per_rate_verdicts.append((rate, f"UNMEASURABLE(partial coverage: {cov*100:.2f}%)"))
            continue
        if cov is None:
            per_rate_verdicts.append((rate, "UNMEASURABLE(coverage indeterminate -- no window bounds)"))
            continue
        if unmeasurable(a["t_total"], a["n_episodes"]):
            per_rate_verdicts.append((rate, "UNMEASURABLE"))
        elif a["time_weighted"] >= 0.90:
            per_rate_verdicts.append((rate, "UNLOCK-eligible"))
        else:
            per_rate_verdicts.append((rate, "NO_UNLOCK"))
    print(f"GATE1_PER_RATE: {per_rate_verdicts}")

    # Final verdict = strictest (min severity) among {pooled_verdict} union
    # {every per-rate verdict}. Ties toward the pooled label when pooled and
    # all per-rate windows agree at the top severity (UNLOCK/UNLOCK-eligible).
    candidates = [("POOLED", pooled_verdict)] + [(f"rate{r}", v) for r, v in per_rate_verdicts]
    worst = min(candidates, key=lambda kv: severity(kv[1]))
    if severity(worst[1]) < 2:
        verdict = worst[1]
    else:
        verdict = pooled_verdict  # both pooled and every per-rate window cleared -90%
    for src, v in candidates:
        if severity(v) < 2:
            reasons.append(f"{src} verdict={v}")

    print(f"GATE1_VERDICT: {verdict}")
    for r in reasons:
        print(f"GATE1_REASON: {r}")
    print("GATE1_CAVEAT: PASS/UNLOCK does NOT establish hardware-level SM partitioning "
          "(selector-level only, S3 hardware probe not run). See PREREG sec "
          "'게이트가 항등식인가' for the code-path identity risk that MUST be cited "
          "alongside any use of this verdict.")


if __name__ == "__main__":
    main()
