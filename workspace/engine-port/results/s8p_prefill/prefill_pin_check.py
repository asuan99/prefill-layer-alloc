#!/usr/bin/env python3
"""REALIZED prefill-partition gate for the s8p_prefill sweep (mirror of
s8_scaleup/realized_pin_check.py with prefill/decode roles swapped).

s8_scaleup's gate read `sm_counts[stream_index][1]` (decode_sms) among
DECODE-active samples, because that campaign swept decode SM. This campaign
sweeps PREFILL SM with decode pinned, so the gate instead reads the
`prefill_sms` field (dual_worker.py:608/622, already emitted per-row in
runtime_snapshot -- see DESIGN.md sec 2) among PREFILL-active samples
(`prefill_active_batch_size > 0`).

Per the pre-registered gates (DESIGN.md sec 5):
  1. realized prefill-SM pin >= min_frac (default 0.80), judged on the
     REALIZED prefill_sms field, never on the policy's target (Stage 0's
     mistake: results/s0_deconfound/PARTITION_RESIDENCY_STAGE0.md).
  2. co-residency: among prefill-active samples, the fraction with
     decode_running_batch_size > 0 (the reverse of s8_scaleup's co-residency,
     which conditioned on decode-active samples instead).

Usage: prefill_pin_check.py <telemetry.jsonl> <expected_prefill_sm> [min_frac]
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
        if e.get("prefill_active_batch_size", 0) <= 0:
            continue
        sm = e.get("prefill_sms")
        if sm is None:
            continue
        realized[sm] += 1
        batches.append(e["prefill_active_batch_size"])
        if e.get("decode_running_batch_size", 0) > 0:
            co_resident += 1

    n = sum(realized.values())
    print(f"TARGET_hist(decode_sms, informational only): {dict(targets) or 'NONE'}")
    if not n:
        print(f"REALIZED_PREFILL_PIN P={expect}: FAIL -> no prefill-active "
              f"snapshots ({n_snap} benchmark snapshots) -> INVALID")
        return
    frac = realized[expect] / n
    dist = ", ".join(f"{k}SM:{v}({v/n*100:.0f}%)" for k, v in realized.most_common())
    print(f"REALIZED_hist(prefill-active n={n}): {dist}")
    print(f"CO_RESIDENT_frac: {co_resident/n:.3f}  "
          f"(share of prefill-active samples with a decode batch in flight)")
    print(f"PREFILL_BATCH: mean={st.fmean(batches):.2f} median={st.median(batches):.1f} "
          f"p10={sorted(batches)[len(batches)//10]:.0f} "
          f"p90={sorted(batches)[9*len(batches)//10]:.0f}")
    verdict = "PASS" if frac >= min_frac else "FAIL -> measured at the wrong SM, INVALID"
    print(f"REALIZED_PREFILL_PIN P={expect}: {verdict} (frac={frac:.3f}, gate={min_frac})")
    cores_verdict = ("PASS" if (co_resident / n) >= min_frac
                      else "FAIL -> low co-residency, cite with caution")
    print(f"CO_RESIDENCY_GATE: {cores_verdict} (frac={co_resident/n:.3f}, gate={min_frac})")


if __name__ == "__main__":
    main()
