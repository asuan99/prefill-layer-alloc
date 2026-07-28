#!/usr/bin/env python3
"""REALIZED-partition gate (replaces Stage 0's target-only PIN_CHECK).

Stage 0 verified pinning by histogramming `controller_decision.target_decode_sms`
-- the policy's *request*.  That missed the legacy `adjust_stream_groups` path
entirely, and its D108 anchor silently ran decode at 16 SM in 82-97% of
decode-active samples (results/s0_deconfound/PARTITION_RESIDENCY_STAGE0.md).

This gate reads the *realized* partition instead: `runtime_snapshot` rows carry
sm_counts[arbiter.stream_index] as (prefill_sms, decode_sms) (dual_worker.py:608).
Stream group 0 is (108,0) = unpartitioned prefill-side stream and the last group
is (0,108) = unpartitioned decode-side stream, so decode_sms == 0 means decode
ran unrestricted and is scored as 108.

Also reports the realized decode occupancy, because run 865098 showed decode
batch co-varying with decode-SM and that confound is invisible unless measured.

Usage: realized_pin_check.py <telemetry.jsonl> <expected_decode_sm> [min_frac]
Exit status is always 0; the verdict line is what the campaign log records.
"""
import collections
import json
import statistics as st
import sys


def main():
    path = sys.argv[1]
    expect = int(sys.argv[2])
    min_frac = float(sys.argv[3]) if len(sys.argv) > 3 else 0.80

    realized = collections.Counter()
    co_resident = 0
    batches = []
    targets = collections.Counter()
    n_snap = 0
    for line in open(path):
        try:
            e = json.loads(line)
        except Exception:
            continue
        ev = e.get("event")
        if ev == "controller_decision":
            targets[e.get("target_decode_sms")] += 1
            continue
        if ev != "runtime_snapshot" or e.get("phase") != "benchmark":
            continue
        n_snap += 1
        if e.get("decode_running_batch_size", 0) <= 0:
            continue
        sm = e["decode_sms"] or 108
        realized[sm] += 1
        batches.append(e["decode_running_batch_size"])
        if e.get("prefill_active_batch_size", 0) > 0:
            co_resident += 1

    n = sum(realized.values())
    print(f"TARGET_hist: {dict(targets) or 'NONE'}")
    if not n:
        print(f"REALIZED_PIN D={expect}: FAIL -> no decode-active snapshots "
              f"({n_snap} benchmark snapshots) -> INVALID")
        return
    frac = realized[expect] / n
    dist = ", ".join(f"{k}SM:{v}({v/n*100:.0f}%)" for k, v in realized.most_common())
    print(f"REALIZED_hist(decode-active n={n}): {dist}")
    print(f"CO_RESIDENT_frac: {co_resident/n:.3f}  "
          f"(share of decode-active samples with a prefill batch in flight)")
    print(f"DECODE_BATCH: mean={st.fmean(batches):.2f} median={st.median(batches):.1f} "
          f"p10={sorted(batches)[len(batches)//10]:.0f} p90={sorted(batches)[9*len(batches)//10]:.0f}")
    verdict = "PASS" if frac >= min_frac else "FAIL -> measured at the wrong SM, INVALID"
    print(f"REALIZED_PIN D={expect}: {verdict} (frac={frac:.3f}, gate={min_frac})")


if __name__ == "__main__":
    main()
