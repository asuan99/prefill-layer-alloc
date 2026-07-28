#!/usr/bin/env python3
"""Batch-matched ITL at matched realized SM (the last confound in this campaign).

s8_analyze.py already conditions on the realized partition, but decode batch
still drifts between buckets (b5..b16), and ITL is strongly batch-dependent --
that is precisely what invalidated run 865098's cell means. Here every
inter-token interval is bucketed by (realized SM, decode batch) and arms are
compared only inside a shared batch bin.
"""
import bisect
import collections
import glob
import json
import os
import re
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
GUARD = 3.0
PAT = re.compile(r"s8_(?P<mode>\w+?)_(?P<arm>\w+?)_C(?P<ctx>\d+)_(?P<cell>d16|d24|d44|d92|np)_"
                 r"(?P<job>\d+)_telemetry\.jsonl$")


def main():
    by = collections.defaultdict(list)   # (arm, sm, batch) -> itls
    for fn in sorted(os.listdir(HERE)):
        m = PAT.search(fn)
        if not m:
            continue
        g = m.groupdict()
        if g["mode"] != "deconf" or g["ctx"] != "1024":
            continue
        arm, job, cell, ctx = g["arm"], g["job"], g["cell"], g["ctx"]
        ts, sm, bs, pf = [], [], [], []
        for line in open(os.path.join(HERE, fn)):
            try:
                e = json.loads(line)
            except Exception:
                continue
            if e.get("event") != "runtime_snapshot" or e.get("phase") != "benchmark":
                continue
            ts.append(e["timestamp_monotonic_s"]); sm.append(e["decode_sms"] or 108)
            bs.append(e["decode_running_batch_size"]); pf.append(e["prefill_active_batch_size"])
        if not ts:
            continue
        t0s = {}
        for res in glob.glob(os.path.join(HERE, f"s8_deconf_{arm}_C{ctx}_{job}_result.txt")):
            for line in open(res):
                if not line.startswith("{"):
                    continue
                try:
                    s = json.loads(line)
                except Exception:
                    continue
                tag = s.get("tag", "")
                if tag.startswith(f"deconf_{arm}_C{ctx}_{cell}_r") and "t0_monotonic_s" in s:
                    t0s[int(tag.rsplit("_r", 1)[1])] = s["t0_monotonic_s"]
        stem = os.path.join(HERE, fn[: -len("_telemetry.jsonl")])
        for raw in sorted(glob.glob(f"{stem}_rep*_raw.jsonl")):
            t0 = t0s.get(int(re.search(r"_rep(\d+)_raw", raw).group(1)))
            if t0 is None:
                continue
            for line in open(raw):
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                for a, b in zip(r.get("chunk_times_s", []), r.get("chunk_times_s", [])[1:]):
                    a, b = a + t0, b + t0
                    i = bisect.bisect_right(ts, a) - 1
                    j = bisect.bisect_left(ts, b)
                    if i < 0 or j >= len(ts) or sm[i] != sm[j] or bs[i] != bs[j]:
                        continue
                    if (a - ts[i]) > GUARD or (ts[j] - b) > GUARD:
                        continue
                    by[(arm, sm[i], bs[i])].append((b - a) * 1000.0)

    arms = sorted({a for a, _, _ in by})
    sms = sorted({s for _, s, _ in by})
    print("ITL p50 (ms) at matched (realized SM, decode batch); cells with n>=50 only")
    for batch in sorted({b for _, _, b in by}):
        rowsy = {}
        for arm in arms:
            cellsr = {}
            for s in sms:
                v = by.get((arm, s, batch), [])
                if len(v) >= 50:
                    cellsr[s] = st.median(v)
            if len(cellsr) >= 2:
                rowsy[arm] = cellsr
        if len(rowsy) < 2:
            continue
        print(f"\n--- decode batch = {batch} ---")
        print(f"{'arm':>4} | " + " ".join(f"{('SM'+str(s)):>9}" for s in sms) + " | SM16/SM108")
        for arm in arms:
            c = rowsy.get(arm)
            if not c:
                continue
            line = " ".join(f"{c[s]:9.2f}" if s in c else " " * 9 for s in sms)
            lo, hi = c.get(min(sms)), c.get(108)
            ratio = f"{lo/hi:9.2f}x" if lo and hi else ""
            print(f"{arm:>4} | {line} | {ratio}")


if __name__ == "__main__":
    main()
