#!/usr/bin/env python3
"""Per-arm ITL at MATCHED realized SM partition.

The Stage 0 client means mix partitions: an arm that spends more time in the
decode-only (0,108) stream looks faster for reasons that have nothing to do
with the model.  Client raw jsonl carries per-token CLOCK_MONOTONIC timestamps
and telemetry carries timestamp_monotonic_s from the same clock domain, so each
inter-token interval can be attributed to the partition that was in force.

Attribution rule (conservative): an interval [t0,t1] counts only if the last
snapshot at or before t0 and the first snapshot at or after t1 report the SAME
realized decode-SM, and both are within GUARD seconds of the interval (3.0 s: a 250 ms guard
 silently dropped every slow-arm interval, biasing the table toward fast ones).  Intervals that
straddle a partition change are dropped.
"""
import bisect
import glob
import json
import os
import re
import statistics as st
from collections import defaultdict

DIR = "/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/stage0_xctrl"
PAT = re.compile(r"s0_(?P<arm>[MHT])_C(?P<ctx>\d+)_D(?P<d>\d+)_(?P<job>\d+)_telemetry\.jsonl$")
GUARD = 3.0


def load_timeline(path):
    ts, sm, bs = [], [], []
    for line in open(path):
        try:
            e = json.loads(line)
        except Exception:
            continue
        if e.get("event") != "runtime_snapshot" or e.get("phase") != "benchmark":
            continue
        ts.append(e["timestamp_monotonic_s"])
        sm.append(108 if e["decode_sms"] == 0 else e["decode_sms"])
        bs.append(e["decode_running_batch_size"])
    return ts, sm, bs


def main():
    out = defaultdict(list)      # (arm, ctx, realized_sm) -> itls
    out_bs = defaultdict(list)   # (arm, ctx, realized_sm, batch) -> itls
    dropped = defaultdict(int)
    for fn in sorted(os.listdir(DIR)):
        m = PAT.search(fn)
        if not m:
            continue
        g = m.groupdict()
        arm, ctx, D, job = g["arm"], int(g["ctx"]), int(g["d"]), g["job"]
        ts, sm, bs = load_timeline(os.path.join(DIR, fn))
        if not ts:
            continue
        stem = f"s0_{arm}_C{ctx}_D{D}_{job}"
        for raw in sorted(glob.glob(os.path.join(DIR, f"{stem}_rep*_raw.jsonl"))):
            for line in open(raw):
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                tt = r.get("token_ts") or []
                for a, b in zip(tt, tt[1:]):
                    i = bisect.bisect_right(ts, a) - 1
                    j = bisect.bisect_left(ts, b)
                    if i < 0 or j >= len(ts):
                        dropped[(arm, ctx, D)] += 1
                        continue
                    if sm[i] != sm[j] or (a - ts[i]) > GUARD or (ts[j] - b) > GUARD:
                        dropped[(arm, ctx, D)] += 1
                        continue
                    itl = (b - a) * 1000.0
                    out[(arm, ctx, sm[i])].append(itl)
                    out_bs[(arm, ctx, sm[i], bs[i])].append(itl)

    print("ITL p50 (ms) at MATCHED realized decode-SM  [n = attributed inter-token intervals]")
    print(f"{'arm':>3} {'ctx':>6} | " + " ".join(f"{('SM'+str(s)):>17}" for s in (16, 44, 92, 108)))
    print("-" * 90)
    rows = {}
    for arm in "MHT":
        for ctx in (4096, 8192, 16384):
            cells = []
            for s in (16, 44, 92, 108):
                v = out.get((arm, ctx, s), [])
                rows[(arm, ctx, s)] = st.median(v) if v else None
                cells.append(f"{st.median(v):7.2f} (n={len(v):5d})" if v else " " * 17)
            print(f"{arm:>3} {ctx:>6} | " + " ".join(cells))

    print()
    print("cross-model ratios at matched realized SM (p50)")
    print(f"{'ctx':>6} {'SM':>4} | {'H/M':>7} {'H/T':>7} {'T/M':>7}")
    for ctx in (4096, 8192, 16384):
        for s in (16, 44, 92, 108):
            h, mm, t = rows[("H", ctx, s)], rows[("M", ctx, s)], rows[("T", ctx, s)]
            if h and mm and t:
                print(f"{ctx:>6} {s:>4} | {h/mm:>7.2f} {h/t:>7.2f} {t/mm:>7.2f}")

    print()
    print("SM-sensitivity at matched exposure: p50(SM16)/p50(SM108) within each arm")
    for arm in "MHT":
        for ctx in (4096, 8192, 16384):
            a, b = rows[(arm, ctx, 16)], rows[(arm, ctx, 108)]
            if a and b:
                print(f"  {arm} C{ctx:<6} {a:7.2f} / {b:7.2f} = {a/b:5.2f}x")

    print()
    print("batch-stratified p50 at realized SM=16 and SM=108 (batch >= 5 samples)")
    for s in (16, 108):
        print(f"  --- realized SM={s} ---")
        for ctx in (4096, 8192, 16384):
            for b in range(1, 9):
                cells = []
                for arm in "MHT":
                    v = out_bs.get((arm, ctx, s, b), [])
                    cells.append(f"{arm}={st.median(v):6.2f}(n{len(v):4d})" if len(v) >= 5 else f"{arm}=     -      ")
                if any("-" not in c for c in cells):
                    print(f"    C{ctx:<6} bs={b} | " + "  ".join(cells))


if __name__ == "__main__":
    main()
