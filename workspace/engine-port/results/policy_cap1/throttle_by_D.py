#!/usr/bin/env python3
"""Throttle / SM-clock distribution per D boot.  Window per boot = [first run file mtime - its duration, last run file mtime] (UTC; tar preserved mtimes).  Descriptive."""
import sys, json, csv, glob, datetime as dt, collections, statistics
from pathlib import Path
raw = Path(sys.argv[1]); out = {}
rows = []
for f in glob.glob(str(raw / "clk/*.csv")):
    for r in csv.reader(open(f)):
        if not r or r[0].startswith("timestamp"): continue
        t = dt.datetime.strptime(r[0].strip(), "%Y/%m/%d %H:%M:%S.%f")
        rows.append((t, float(r[1].split()[0]), int(r[4].strip(), 16)))
rows.sort()
for b in sorted((raw / "boots").iterdir()):
    D = json.load(open(b / "boot.json"))["arm"]["D"]
    ends = []
    for rid in json.load(open(b / "DONE.json"))["runs"]:
        p = raw / "runs" / f"{rid}.json"
        d = json.load(open(p)); dur = sum(s["duration_s"] for s in d["segments"])
        e = dt.datetime.utcfromtimestamp(p.stat().st_mtime)
        ends.append((e - dt.timedelta(seconds=dur), e))
    lo, hi = min(a for a, _ in ends), max(e for _, e in ends)
    w = [r for r in rows if lo <= r[0] <= hi]
    busy = [r for r in w if r[1] > 300]
    bits = collections.Counter()
    for r in w:
        for k in (0x4, 0x8, 0x20, 0x40, 0x80, 0x100): 
            if r[2] & k: bits[hex(k)] += 1
    thr = sum(1 for r in w if r[2] & ~0x1)
    out[f"D{D}"] = {"window_utc": [lo.isoformat(), hi.isoformat()], "samples": len(w), "throttle_samples": thr, "throttle_frac": thr / len(w) if w else None,
                    "reason_bits": dict(bits), "sm_clk_median_MHz": statistics.median(r[1] for r in busy) if busy else None,
                    "sm_clk_p05_MHz": sorted(r[1] for r in busy)[int(.05 * len(busy))] if busy else None}
print(json.dumps(out, indent=1, sort_keys=True))
json.dump(out, open(sys.argv[2], "w"), indent=1, sort_keys=True)
