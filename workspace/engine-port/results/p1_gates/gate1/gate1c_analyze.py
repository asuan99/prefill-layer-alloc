#!/usr/bin/env python3
"""G1-c realized-partition analysis for Granite (see PREREG_G1C_2026-08-11.md).

Sibling of gate1b_analyze.py (which analyzed Zamba2/job 875293, replicating
873944) -- this script analyzes Granite/job <this job>, replicating 873945.
gate1_analyze.py and gate1b_analyze.py are left UNMODIFIED (this is a
separate script so 874478's and 875293's analysis paths/outputs are never
touched).

Population classification, aggregation unit (time-weighted primary), coverage
guard (>=0.98), sparse guard (MIN_TOTAL_S/MIN_EPISODES), grid-completeness
diagnostic, and the withdrawal threshold (WITHDRAWAL_THRESHOLD=0.01) are all
imported VERBATIM from gate1b_analyze.py -- zero new free parameters
(methodology gate #9: a gate that reuses the producer's own thresholds is not
circular here because the THING being computed -- Granite's realized
partition trajectory -- was never computed by any prior script; the
THRESHOLD is what's reused, not the computation).

What differs from gate1b_analyze.py, and why (both pre-registered in
PREREG_G1C_2026-08-11.md, fixed BEFORE this job runs):

  1. No Zamba2-style empirical anchor exists to reproduce for Granite --
     873945 (the campaign this job replicates) never captured telemetry
     (grep of p1op_run.sbatch confirms no PDMUX_TELEMETRY_PATH is set), so
     there is no prior Granite max_decode_bs number to match. The positive
     control here is instead the MODEL-AGNOSTIC structural condition Gate 1's
     own audit established (CONSENSUS.md sec1-1 Gate 1 block): pop A
     (decode-busy AND prefill-in-flight) almost always resolves to
     idx in {1,2} (i.e. TARGET_A = {(74,34),(54,54)}), because the selector
     code path that produces this is not conditioned on model identity. If
     this near-identity FAILS for Granite, that is evidence the telemetry/
     selector pipeline itself misbehaves for this model (harness bug), NOT a
     real behavioural difference -- and per PREREG_G1C, rates that fail this
     check are NOT interpreted for max_decode_bs/frac_5454.

  2. A new block (G1C_GATE2S_PREMISE) evaluates the specific downstream
     question this job exists to answer: for Gate 2-S's Granite r3 and r4
     cells (PREREG_GATE2S_2026-08-09.md sec8.9/sec8.9.1), was the label
     "unverified" upgradeable to VERIFIED or REFUTED? Gate 2-S's own in-job
     sec8.9.1 falsifier is STRUCTURALLY incapable of ever emitting VERIFIED
     (one-way by design, sparse PDMUX_TRACE_FORCE_PREFILL=0 telemetry -- see
     PREREG_GATE2S sec8.9.1). This job uses the DENSE forced-sampling
     protocol (PDMUX_TRACE_FORCE_PREFILL=1, matching Gate 1/G1-b) that is
     what let Gate 1 assert "verified" for Zamba2 in the first place, so it
     CAN supply a VERIFIED verdict (in addition to being able to REFUTE, like
     Gate 2-S's own weaker check).

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
TRACE_EVERY = 32          # matches PDMUX_DUAL_WORKER_TRACE_EVERY default, unchanged from 874478/G1-b
DECODE_BS_DIVISOR = 36    # pdmux_a100_smoke.yml -- threshold at which (54,54) becomes reachable (model-independent, config file is identical, see PREREG_G1C manifest check)
WITHDRAWAL_THRESHOLD = 0.01  # pre-registered: pop-A time-weighted frac of (54,54) >= 1% -> WITHDRAW/REFUTE
POSCTRL_TARGET_A_MIN = 0.90  # pre-registered: structural near-identity floor (imported from Gate 1's own main-condition threshold), NOT a Granite-specific empirical anchor -- see module docstring point 1
GATE2S_CELLS = ("3", "4")   # PREREG_GATE2S_2026-08-09.md sec3: Granite cells actually used = rate {3,4}


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
    """Per-row invariant + missing-scheduled-point diagnostic. Anchored to
    THIS window's own [min_sample_index, max_sample_index] (copied verbatim
    from gate1b_analyze.py -- see that file's docstring for the earlier
    global-anchor bug this avoids)."""
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
    print(f"G1C_POP {label}({pop_name}): n_snapshots={d['n']} n_episodes={d['episodes']} "
          f"t_total_s={d['t_total']:.3f} target={sorted(target_set)}")
    print(f"G1C_FRAC {label} time_weighted={tw:.4f} count_weighted={cw:.4f} "
          f"(count secondary only, force-prefill biases it -- see PREREG)")
    print(f"G1C_HIST {label} (time-weighted): {fmt_hist(d['time_hist'], d['t_total'])}")
    print(f"G1C_CADENCE {label}: median_sample_index_gap={gap_med} n_gaps={len(d['sample_idx_gaps'])}")
    return dict(time_weighted=tw, count_weighted=cw, t_total=d["t_total"],
                n_episodes=d["episodes"], n_snapshots=d["n"],
                frac_5454=frac_in(d["time_hist"], d["t_total"], {(54, 54)}) if d["t_total"] > 0 else float("nan"))


def main():
    argv = sys.argv[1:]
    if not argv:
        print("usage: gate1c_analyze.py <telemetry.jsonl> --window RATE T0 T1 [...]")
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
        print(f"G1C_COVERAGE rate={rate}: telemetry_coverage={cov_str} "
              f"(window=[{t0},{t1}] n_rows={len(rows)})")
        if not rows:
            print(f"G1C_EMPTY_WINDOW rate={rate}")
            per_rate[rate] = dict(verdict="UNMEASURABLE(empty window)")
            continue

        mdbs = max_decode_bs(rows)
        print(f"G1C_MAX_DECODE_BS rate={rate}: max(decode_running_batch_size)={mdbs} "
              f"(threshold for (54,54) reachability = decode_bs_divisor={DECODE_BS_DIVISOR})")

        gc = grid_completeness(rows)
        print(f"G1C_GRID_COMPLETENESS rate={rate}: expected={gc['expected']} "
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
        print(f"G1C_RATE_VERDICT rate={rate}: {verdict}")

    # --- Positive control: structural near-identity, ALL 4 rates ---
    # (This is NOT a Zamba2-style empirical-magnitude reproduction -- 873945
    # captured no telemetry, so there is no prior Granite number to match.
    # It is Gate 1's own MAIN condition, which the claims-auditor confirmed
    # is a near-identity of the selector code path (multiplexing_mixin.py
    # is not conditioned on model identity) -- see module docstring point 1.
    # A rate that fails this is a harness-integrity flag, not a behavioural
    # finding, and its max_decode_bs/frac_5454 are NOT interpreted below.)
    print(f"=== G1C POSITIVE CONTROL (structural near-identity, ALL rates, "
          f"pop_A_time_weighted_frac(TARGET_A) >= {POSCTRL_TARGET_A_MIN} expected by code audit "
          f"regardless of model -- see CONSENSUS.md sec1-1 Gate 1 block) ===")
    posctrl_ok = {}
    for r in ("2", "3", "4", "6"):
        info = per_rate.get(r)
        if info is None or info["verdict"] != "MEASURED":
            print(f"G1C_POSCTRL rate={r}: FAIL (not measurable: "
                  f"{info['verdict'] if info else 'window not run'})")
            posctrl_ok[r] = False
            continue
        a_frac = info["a"]["time_weighted"]
        ok = a_frac >= POSCTRL_TARGET_A_MIN
        print(f"G1C_POSCTRL rate={r}: pop_A_time_weighted_frac(TARGET_A)={a_frac:.4f} "
              f"-> {'PASS' if ok else 'FAIL'}")
        posctrl_ok[r] = ok

    # --- Withdrawal/extension-style rule per rate (mechanical, diagnostic; not a Granite CONSENSUS headline -- none exists yet) ---
    print("=== G1C PER-RATE PARTITION-TRAJECTORY RULE (pre-registered, PREREG_G1C_2026-08-11.md; "
          "mechanical output, not a performance judgment) ===")
    for r in ("2", "3", "4", "6"):
        info = per_rate.get(r)
        if info is None or info["verdict"] != "MEASURED":
            print(f"G1C_RULE rate={r}: UNMEASURABLE -- "
                  f"{info['verdict'] if info else 'window not run'} -- no claim possible for this rate")
            continue
        mdbs = info["max_decode_bs"]
        frac5454 = info["frac_5454"]
        if not posctrl_ok.get(r, False):
            print(f"G1C_RULE rate={r}: max_decode_bs={mdbs} frac_5454_time_weighted={frac5454:.4f} "
                  f"-- POSITIVE CONTROL FAILED at this rate, numbers NOT interpreted (PREREG_G1C rule)")
            continue
        if frac5454 >= WITHDRAWAL_THRESHOLD:
            verdict = f"(54,54) observed -- time-weighted frac={frac5454:.4f} >= {WITHDRAWAL_THRESHOLD}"
        elif frac5454 > 0:
            verdict = f"(54,54) observed but frac={frac5454:.4f} < {WITHDRAWAL_THRESHOLD} (trace-level only)"
        else:
            verdict = "NO_EVIDENCE -- (54,54) frac=0.0000"
        print(f"G1C_RULE rate={r}: max_decode_bs={mdbs} frac_5454_time_weighted={frac5454:.4f} -> {verdict}")

    # --- The reason this job exists: Gate 2-S sec8.9 Granite r3/r4 premise label ---
    print(f"=== G1C_GATE2S_PREMISE (PREREG_GATE2S_2026-08-09.md sec8.9/sec8.9.1, cells={GATE2S_CELLS}) ===")
    print("Rule (pre-registered, fixed before this job ran, NOT changed after seeing results):")
    print("  positive control FAIL at that rate       -> UNMEASURABLE (cannot evaluate premise)")
    print(f"  frac_5454 (pop A, time-weighted) >= {WITHDRAWAL_THRESHOLD} -> PREMISE REFUTED "
          f"(same criterion as Gate 2-S's own sec8.9.1 in-job falsifier, but this job's denser "
          f"forced-sampling telemetry makes any refutation higher-confidence than sec8.9.1's own "
          f"sparse PDMUX_TRACE_FORCE_PREFILL=0 check could)")
    print(f"  frac_5454 < {WITHDRAWAL_THRESHOLD} AND max_decode_bs < {DECODE_BS_DIVISOR} -> "
          f"PREMISE VERIFIED (upgrade from Gate 2-S's 'unverified' -- this is the label sec8.9.1 "
          f"is STRUCTURALLY incapable of ever emitting, by its own one-way design)")
    for r in GATE2S_CELLS:
        info = per_rate.get(r)
        if info is None or info["verdict"] != "MEASURED":
            print(f"G1C_GATE2S_PREMISE rate={r}: UNMEASURABLE -- "
                  f"{info['verdict'] if info else 'window not run'} -- Granite r{r} premise stays 'unverified'")
            continue
        if not posctrl_ok.get(r, False):
            print(f"G1C_GATE2S_PREMISE rate={r}: UNMEASURABLE -- positive control failed at this rate "
                  f"-- Granite r{r} premise stays 'unverified' (harness-integrity concern, not a partition finding)")
            continue
        mdbs = info["max_decode_bs"]
        frac5454 = info["frac_5454"]
        if frac5454 >= WITHDRAWAL_THRESHOLD:
            print(f"G1C_GATE2S_PREMISE rate={r}: max_decode_bs={mdbs} frac_5454={frac5454:.4f} "
                  f">= {WITHDRAWAL_THRESHOLD} -> PREMISE REFUTED")
        else:
            print(f"G1C_GATE2S_PREMISE rate={r}: max_decode_bs={mdbs} frac_5454={frac5454:.4f} "
                  f"< {WITHDRAWAL_THRESHOLD}, max_decode_bs<{DECODE_BS_DIVISOR}={mdbs is not None and mdbs < DECODE_BS_DIVISOR} "
                  f"-> PREMISE VERIFIED")

    print("G1C_CAVEAT: selector-level, behavioural only (no hardware SM probe, S3 not run). "
          "Population C absolute fractions are NOT to be quoted per Gate 1 audit "
          "(few-snapshot, dt inflation) -- count/episode only. n=1 (diagnostic pipeline, not "
          "a performance sample) -- no TTFT/TPOT/ITL/goodput number from this job's "
          "bench_serving JSONL output may ever be quoted. See PREREG_G1C_2026-08-11.md.")


if __name__ == "__main__":
    main()
