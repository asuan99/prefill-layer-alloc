#!/usr/bin/env python3
"""Segment the G16 telemetry into benchmark ROUNDS (idle gaps where decode is
empty for >2s) and report, per round: wall span, decode forwards (delta of the
engine-side `decode_iterations` counter), time-mean decode batch, and the
CO_RESIDENT_frac.  This lets W-3's axis-2 ratio be evaluated per HI round
instead of via the mean-ITL proxy."""
import json, glob, sys, statistics as st

def rounds(path, idle_gap=2.0):
    rows = []
    for line in open(path, errors="replace"):
        try: e = json.loads(line)
        except Exception: continue
        if e.get("event") != "runtime_snapshot" or e.get("phase") != "benchmark": continue
        rows.append((e["timestamp_monotonic_s"], e.get("decode_running_batch_size", 0),
                     e.get("decode_iterations", 0), e.get("prefill_active_batch_size", 0)))
    rows.sort()
    segs, cur, last_active = [], [], None
    for r in rows:
        if r[1] > 0:
            if last_active is not None and r[0] - last_active > idle_gap and cur:
                segs.append(cur); cur = []
            last_active = r[0]
            cur.append(r)
        elif cur:
            cur.append(r)
    if cur: segs.append(cur)
    return [s for s in segs if sum(1 for x in s if x[1] > 0) > 50]

for p in sorted(glob.glob(sys.argv[1])):
    print(f"\n=== {p}")
    for i, s in enumerate(rounds(p)):
        act = [x for x in s if x[1] > 0]
        span = act[-1][0] - act[0][0]
        dfwd = act[-1][2] - act[0][2]
        b = [x[1] for x in act]
        co = sum(1 for x in act if x[3] > 0) / len(act)
        print(f"  round{i}: span={span:6.1f}s  DECODE_FORWARDS={dfwd:6d}  "
              f"decode-active snaps={len(act):6d}  mean B={st.fmean(b):5.2f} med B={st.median(b):4.1f} "
              f"max B={max(b):3d}  CO_RESIDENT_frac={co:.4f}")
