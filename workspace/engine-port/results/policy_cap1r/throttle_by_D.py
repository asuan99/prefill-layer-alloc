#!/usr/bin/env python3
"""CAP1-R throttle per D boot.  Run-file mtimes are copy times here, so boot windows come from the clk trace itself: the 4 longest
idle (SM clk < 300 MHz) gaps after the first model load separate the 5 boots (boot order = boots/ index).  Descriptive."""
import sys, json, csv, glob, collections, statistics, datetime as dt
from pathlib import Path
raw = Path(sys.argv[1]); rows = []
for f in glob.glob(str(raw / "clk/*.csv")):
    for r in csv.reader(open(f)):
        if r and not r[0].startswith("timestamp"):
            rows.append((dt.datetime.strptime(r[0].strip(), "%Y/%m/%d %H:%M:%S.%f"), float(r[1].split()[0]), int(r[4].strip(), 16)))
rows.sort(); gaps = []; s = None
for i, (t, c, _) in enumerate(rows):
    if c < 300:
        s = i if s is None else s
    elif s is not None:
        gaps.append((s, i - 1)); s = None
boots = sorted(p.name for p in (raw / "boots").iterdir()); n = len(boots)
first_busy = gaps[0][1] + 1 if gaps and gaps[0][0] == 0 else 0
inner = [g for g in gaps if g[0] > first_busy + 3000]          # exclude the initial load idle (and the post-first-load stretch)
inner = sorted(sorted(inner, key=lambda g: g[1] - g[0])[-(n - 1):])
cuts = [first_busy] + [g[0] for g in inner] + [len(rows)]
# the first low gap after load (~31 s) is boot-0 server start; start boot 0 after it
out = {}
for k, b in enumerate(boots):
    lo = cuts[k] if k else (gaps[1][1] + 1 if len(gaps) > 1 and gaps[1][0] < 3000 else first_busy)
    hi = cuts[k + 1]; w = rows[lo:hi]; busy = [r for r in w if r[1] > 300]
    D = json.load(open(raw / "boots" / b / "boot.json"))["arm"]["D"]
    thr = sum(1 for r in w if r[2] & ~0x1); bits = collections.Counter()
    for r in w:
        for m in (0x4, 0x8, 0x20, 0x40, 0x80, 0x100):
            if r[2] & m: bits[hex(m)] += 1
    out[f"D{D}"] = {"boot": b, "window": [w[0][0].isoformat(), w[-1][0].isoformat()], "samples": len(w), "throttle_samples": thr,
                    "throttle_frac": thr / len(w), "reason_bits": dict(bits), "sm_clk_median_MHz": statistics.median(r[1] for r in busy)}
print(json.dumps(out, indent=1, sort_keys=True)); json.dump(out, open(sys.argv[2], "w"), indent=1, sort_keys=True)
