#!/usr/bin/env python3
"""G1-b realized-partition analysis (see PREREG_G1B_2026-08-07.md).

Extends gate1_analyze.py (which is left UNMODIFIED -- this is a separate
script so job 874478's analysis path/output are never touched) with the two
things G1-b's decision rule needs that Gate 1 did not compute:

  1. max(decode_running_batch_size) per rate window (over ALL benchmark-phase
     rows in the window, not just population A) -- this is an EMPIRICAL fact
     about the workload, not a code identity (contrast with Gate 1's main
     condition, which the audit confirmed is a near-identity of the selector
     code path; see PREREG_G1B "이 게이트가 Gate 1 본체와 다른 점").
  2. grid-completeness diagnostic: for `trace_forced==False` rows, checks
     `sample_index == 1 OR sample_index % TRACE_EVERY == 0` (a per-row
     invariant -- violations indicate a sampler bug) AND reports how many of
     the EXPECTED scheduled grid points (multiples of TRACE_EVERY spanning
     [min_sample_index, max_sample_index] observed in the window, plus
     sample_index==1 if in range) are MISSING from the observed
     trace_forced==False set. This catches mid-run drops that a tail-only
     coverage statistic cannot (see PREREG_GATE1 rev14 Gate-1 block: "이번
     '유실 없음'의 실제 근거는 coverage가 아니라 이 검정이었음"). NOTE:
     sample_index (`dual_worker_trace_count`) is a single monotonic counter
     for the ENTIRE server process, not reset per rate window -- the expected
     grid for a window must therefore be anchored to that window's own
     [min_sample_index, max_sample_index], not to 0..max_sample_index (an
     earlier draft of this check used the wrong anchor and wrongly reported
     thousands of "missing" points that were never in this window's range;
     validated by reproducing gate1_result_874478.txt's "missing 0/22,252"
     figure with the GLOBAL variant, and 0/11,959 (rate2) + 0/5,683 (rate3)
     with the per-window variant, on the existing 874478 telemetry file
     before this script was ever pointed at new data).

Population definitions (A/B/C), TARGET_A/B/C, aggregation unit (time-weighted
primary), coverage guard (>=0.98), episode/MIN_TOTAL_S/MIN_EPISODES gates are
copied VERBATIM from gate1_analyze.py -- this script does not change Gate 1's
decision rule, it only adds the G1-b-specific quantities layered on top.

NO latency/throughput/goodput number is read or printed by this script --
only `runtime_snapshot` telemetry fields. Exit status is always 0.
"""
import json
import sys

TARGET_A = {(74, 34), (54, 54)}
TARGET_B = {(108, 0)}
TARGET_C = {(0, 108)}
MIN_TOTAL_S = 0.5
MIN_EPISODES = 2
MIN_COVERAGE = 0.98
TRACE_EVERY = 32          # matches PDMUX_DUAL_WORKER_TRACE_EVERY default, unchanged from 874478
DECODE_BS_DIVISOR = 36    # pdmux_a100_smoke.yml -- threshold at which (54,54) becomes reachable
WITHDRAWAL_THRESHOLD = 0.01  # pre-registered: pop-A time-weighted frac of (54,54) >= 1% -> WITHDRAW


def load_rows(path, t0=None, t1=None, require_phase="benchmark"):
    rows = []
    for line in open(path):
        try:
            e = json.loads(line)
        except Exception:
            continue
        if e.get("event") != "runtime_snapshot":
            continue
        if require_phase is not None and e.get("phase") != require_phase:
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
    p_active = (e.get("prefill_active_batch_size", 0) or 0) > 0
    d_active = (e.get("decode_running_batch_size", 0) or 0) > 0
    if not d_active:
        return "B"
    if p_active:
        return "A"
    return "C"


def analyze_window(rows, t1_bound):
    pops = {k: dict(time_hist={}, count_hist={}, t_total=0.0, n=0,
                     episodes=0, sample_idx_gaps=[])
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

    cur_pop = None
    for e in rows:
        pop = classify(e)
        if pop != cur_pop:
            pops[pop]["episodes"] += 1
            cur_pop = pop
    return pops


def max_decode_bs(rows):
    vals = [e.get("decode_running_batch_size", 0) or 0 for e in rows]
    return max(vals) if vals else None


def grid_completeness(rows, trace_every=TRACE_EVERY):
    """Per-row invariant + missing-scheduled-point diagnostic. See module
    docstring for why the expected grid is anchored to THIS window's own
    [min_sample_index, max_sample_index], not a global 0-based range."""
    sis = [e.get("sample_index") for e in rows if e.get("sample_index") is not None]
    if not sis:
        return dict(expected=0, missing=0, violations=0, min_si=None, max_si=None,
                     n_scheduled_obs=0, missing_sample=[])
    min_si, max_si = min(sis), max(sis)
    scheduled_observed = {e["sample_index"] for e in rows if e.get("trace_forced") is False}
    violations = sum(
        1 for e in rows
        if e.get("trace_forced") is False
        and not (e["sample_index"] == 1 or e["sample_index"] % trace_every == 0)
    )
    expected = set()
    if min_si <= 1 <= max_si:
        expected.add(1)
    start = ((min_si + trace_every - 1) // trace_every) * trace_every
    expected |= set(range(start, max_si + 1, trace_every))
    missing = expected - scheduled_observed
    return dict(expected=len(expected), missing=len(missing), violations=violations,
                min_si=min_si, max_si=max_si, n_scheduled_obs=len(scheduled_observed),
                missing_sample=sorted(missing)[:10])


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
    print(f"G1B_POP {label}({pop_name}): n_snapshots={d['n']} n_episodes={d['episodes']} "
          f"t_total_s={d['t_total']:.3f} target={sorted(target_set)}")
    print(f"G1B_FRAC {label} time_weighted={tw:.4f} count_weighted={cw:.4f} "
          f"(count secondary only, force-prefill biases it -- see PREREG)")
    print(f"G1B_HIST {label} (time-weighted): {fmt_hist(d['time_hist'], d['t_total'])}")
    print(f"G1B_CADENCE {label}: median_sample_index_gap={gap_med} n_gaps={len(d['sample_idx_gaps'])}")
    return dict(time_weighted=tw, count_weighted=cw, t_total=d["t_total"],
                n_episodes=d["episodes"], n_snapshots=d["n"],
                frac_5454=frac_in(d["time_hist"], d["t_total"], {(54, 54)}) if d["t_total"] > 0 else float("nan"))


def main():
    argv = sys.argv[1:]
    if not argv:
        print("usage: gate1b_analyze.py <telemetry.jsonl> --window RATE T0 T1 [...]")
        sys.exit(0)
    telemetry_path = argv[0]
    windows = []
    i = 1
    while i < len(argv):
        if argv[i] == "--window":
            windows.append((argv[i + 1], float(argv[i + 2]), float(argv[i + 3])))
            i += 4
        else:
            i += 1
    if not windows:
        windows = [("ALL", None, None)]

    per_rate = {}
    for rate, t0, t1 in windows:
        rows = load_rows(telemetry_path, t0, t1)
        cov = compute_coverage(rows, t0, t1)
        cov_str = f"{cov*100:.2f}%" if cov is not None else "N/A(no window bounds or empty)"
        print(f"G1B_COVERAGE rate={rate}: telemetry_coverage={cov_str} "
              f"(window=[{t0},{t1}] n_rows={len(rows)})")
        if not rows:
            print(f"G1B_EMPTY_WINDOW rate={rate}")
            per_rate[rate] = dict(verdict="UNMEASURABLE(empty window)")
            continue

        mdbs = max_decode_bs(rows)
        print(f"G1B_MAX_DECODE_BS rate={rate}: max(decode_running_batch_size)={mdbs} "
              f"(threshold for (54,54) reachability = decode_bs_divisor={DECODE_BS_DIVISOR})")

        gc = grid_completeness(rows)
        print(f"G1B_GRID_COMPLETENESS rate={rate}: expected={gc['expected']} "
              f"observed_scheduled={gc['n_scheduled_obs']} missing={gc['missing']} "
              f"violations={gc['violations']} min_si={gc['min_si']} max_si={gc['max_si']} "
              f"missing_sample={gc['missing_sample']}")

        pops = analyze_window(rows, t1)
        a = print_pop_block(f"rate{rate}_A", "decode-busy AND prefill in-flight", pops["A"], TARGET_A)
        b = print_pop_block(f"rate{rate}_B", "decode-idle", pops["B"], TARGET_B)
        c = print_pop_block(f"rate{rate}_C", "decode-busy, prefill NOT in-flight", pops["C"], TARGET_C)

        cov_ok = cov is not None and cov >= MIN_COVERAGE
        a_measurable = a["t_total"] >= MIN_TOTAL_S and a["n_episodes"] >= MIN_EPISODES
        if not cov_ok:
            verdict = f"UNMEASURABLE(coverage={cov_str})"
        elif not a_measurable:
            verdict = "UNMEASURABLE(pop A too sparse)"
        else:
            verdict = "MEASURED"
        per_rate[rate] = dict(verdict=verdict, coverage=cov, max_decode_bs=mdbs,
                               frac_5454=a["frac_5454"], a=a, b=b, c=c, grid=gc)
        print(f"G1B_RATE_VERDICT rate={rate}: {verdict}")

    # --- Positive control: rate 2 and 3 must reproduce 874478 ---
    print("=== G1B POSITIVE CONTROL (rate 2, 3 vs 874478 anchor: pop A frac_target=1.0000, "
          "max_decode_bs approx 23) ===")
    posctrl_ok = True
    for r in ("2", "3"):
        info = per_rate.get(r)
        if info is None or info["verdict"] != "MEASURED":
            print(f"G1B_POSCTRL rate={r}: FAIL (not measurable: "
                  f"{info['verdict'] if info else 'window not run'})")
            posctrl_ok = False
            continue
        a_frac = info["a"]["time_weighted"]
        mdbs = info["max_decode_bs"]
        ok = a_frac >= 0.90 and mdbs is not None and mdbs <= 30
        print(f"G1B_POSCTRL rate={r}: pop_A_time_weighted_frac={a_frac:.4f} "
              f"max_decode_bs={mdbs} -> {'PASS' if ok else 'FAIL'}")
        posctrl_ok = posctrl_ok and ok
    print(f"G1B_POSCTRL_OVERALL: {'PASS' if posctrl_ok else 'FAIL -- rate 4/6 numbers below are NOT interpreted per PREREG_G1B'}")

    # --- Withdrawal / extension rule for rate 4, 6 ---
    print("=== G1B WITHDRAWAL/EXTENSION RULE (rate 4, 6; pre-registered, PREREG_G1B_2026-08-07.md) ===")
    for r in ("4", "6"):
        info = per_rate.get(r)
        if info is None or info["verdict"] != "MEASURED":
            print(f"G1B_RULE rate={r}: UNMEASURABLE -- "
                  f"{info['verdict'] if info else 'window not run'} -- no withdrawal/extension claim possible for this rate")
            continue
        mdbs = info["max_decode_bs"]
        frac5454 = info["frac_5454"]
        if not posctrl_ok:
            print(f"G1B_RULE rate={r}: max_decode_bs={mdbs} frac_5454_time_weighted={frac5454:.4f} "
                  f"-- POSITIVE CONTROL FAILED, this rate's numbers are NOT interpreted (PREREG_G1B rule)")
            continue
        if frac5454 >= WITHDRAWAL_THRESHOLD:
            verdict = f"WITHDRAW -- (54,54) time-weighted frac={frac5454:.4f} >= {WITHDRAWAL_THRESHOLD}"
        elif frac5454 > 0:
            verdict = f"TRACE -- (54,54) observed but frac={frac5454:.4f} < {WITHDRAWAL_THRESHOLD} (not a withdrawal, add caveat)"
        else:
            verdict = "NO_EVIDENCE -- (54,54) frac=0.0000, extend rule applies if max_decode_bs<36"
        print(f"G1B_RULE rate={r}: max_decode_bs={mdbs} frac_5454_time_weighted={frac5454:.4f} -> {verdict}")

    print("G1B_CAVEAT: selector-level, behavioural only (no hardware SM probe, S3 not run). "
          "Population C absolute fractions are NOT to be quoted per Gate 1 audit "
          "(few-snapshot, dt inflation) -- count/episode only. See PREREG_G1B_2026-08-07.md.")


if __name__ == "__main__":
    main()
