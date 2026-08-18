#!/usr/bin/env python3
"""A5 -- FEASIBILITY: can the state-conditional decode step time be estimated
from the ALREADY-COLLECTED G16 telemetry (GPU cost 0)?

G16_RESULTS sec5-3 says closing the gap_upper non-identification "is not a
matter of re-measuring state-conditional ITL (that would be a new estimand)"
and prices a 1.5-2 GPU-hr sticky campaign.  But runtime_snapshot already
carries decode_iterations (a monotone decode-step counter), decode_sms (the
realised partition) and a monotonic timestamp.  Consecutive same-state
snapshots therefore bound the per-step time IN EACH STATE.

This is a DIAGNOSTIC, not a replacement estimand: scheduler-side step time is
not the client-side per-request p95 token ITL that M_itl is defined on.
Straddled intervals (a state flip between two samples) bias the two states
toward each other, i.e. this UNDERSTATES any conditional contrast.
"""
import json, os, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
SL = os.path.join(HERE, "..")


def probe(fn, phase_want="benchmark"):
    ts = []
    with open(fn) as fh:
        for line in fh:
            if '"runtime_snapshot"' not in line:
                continue
            e = json.loads(line)
            if e.get("phase") != phase_want:
                continue
            ts.append((e["timestamp_monotonic_s"], e["decode_sms"],
                       e["decode_iterations"], e.get("decode_running_batch_size", 0)))
    agg = defaultdict(lambda: [0.0, 0, 0, 0.0])   # state -> [dt, diter, npairs, dt*bs]
    for (t0, s0, i0, b0), (t1, s1, i1, b1) in zip(ts, ts[1:]):
        if s0 != s1:
            continue
        dt, di = t1 - t0, i1 - i0
        if dt <= 0 or di <= 0 or dt > 1.0:
            continue
        a = agg[s0]
        a[0] += dt; a[1] += di; a[2] += 1; a[3] += dt * b0
    return ts, agg


for run in sys.argv[1:] or ["g16_blk1_d16_boot1_884336", "g16_blk1_d44_boot1_884336",
                            "g16_blk1_d64_boot1_884336", "g16_blk1_d74_boot1_884336"]:
    fn = os.path.join(SL, run + "_telemetry.jsonl")
    if not os.path.exists(fn):
        print("missing", fn); continue
    ts, agg = probe(fn)
    print(f"\n{run}: {len(ts)} benchmark snapshots")
    print(f"  {'state':>6} {'pairs':>7} {'sum dt(s)':>10} {'sum steps':>10} "
          f"{'ms/step':>9} {'mean bs':>8}")
    for s in sorted(agg):
        dt, di, n, dtb = agg[s]
        print(f"  {s:>6} {n:>7} {dt:>10.2f} {di:>10} {1000*dt/di:>9.2f} "
              f"{dtb/dt:>8.2f}")
    if 108 in agg and any(k not in (0, 108) for k in agg):
        D = [k for k in agg if k not in (0, 108)][0]
        r = (agg[D][0] / agg[D][1]) / (agg[108][0] / agg[108][1])
        print(f"  ==> per-step time ratio  state({D} SM) / state(108 SM) = {r:.3f}x "
              f"(straddle-contaminated LOWER bound on the contrast)")
