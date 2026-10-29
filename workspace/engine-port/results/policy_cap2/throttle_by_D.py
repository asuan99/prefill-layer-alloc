#!/usr/bin/env python3
"""CAP2 throttle per D boot.  Boot windows = clk-trace idle gaps (4 boundaries); the LAST CAP2 boot is followed by the sustain stage with no
detectable gap, so its end is estimated as start + sum(run duration) + (mean per-run overhead measured on the gap-delimited boots).  Descriptive."""
import sys, json, csv, glob, collections, statistics, datetime as dt
from pathlib import Path
raw = Path(sys.argv[1]); rows = []
for f in glob.glob(str(raw / "clk/*.csv")):
    for r in csv.reader(open(f)):
        if r and not r[0].startswith("timestamp"):
            rows.append((dt.datetime.strptime(r[0].strip(), "%Y/%m/%d %H:%M:%S.%f"), float(r[1].split()[0]), int(r[4].strip(), 16)))
rows.sort(); gaps = []; s = None
for i, (t, c, _) in enumerate(rows):
    if c < 300: s = i if s is None else s
    elif s is not None:
        if (rows[i-1][0] - rows[s][0]).total_seconds() > 5: gaps.append((s, i - 1))
        s = None
boots = sorted(p.name for p in (raw / "boots").iterdir()); n = len(boots)
first = gaps[1][1] + 1                                   # after initial load idle(s)
cuts = [first] + [g[0] for g in gaps[2:2 + n - 1]]
def dur(b):
    return sum(sum(x["duration_s"] for x in json.load(open(raw / "runs" / f"{r}.json"))["segments"]) for r in json.load(open(raw / "boots" / b / "DONE.json"))["runs"]), len(json.load(open(raw / "boots" / b / "DONE.json"))["runs"])
wins = []
for k in range(n - 1): wins.append((cuts[k], cuts[k + 1]))
ovh = statistics.mean(((rows[hi][0] - rows[lo][0]).total_seconds() - dur(boots[k])[0]) / dur(boots[k])[1] for k, (lo, hi) in enumerate(wins))
lo = cuts[n - 1]; end = rows[lo][0] + dt.timedelta(seconds=dur(boots[-1])[0] + ovh * dur(boots[-1])[1])
hi = max(i for i, r in enumerate(rows) if r[0] <= end); wins.append((lo, hi))
out = {}
for b, (lo, hi) in zip(boots, wins):
    w = rows[lo:hi]; busy = [r for r in w if r[1] > 300]
    D = json.load(open(raw / "boots" / b / "boot.json"))["arm"]["D"]
    thr = sum(1 for r in w if r[2] & ~0x1); bits = collections.Counter()
    for r in w:
        for m in (0x4, 0x8, 0x20, 0x40, 0x80, 0x100):
            if r[2] & m: bits[hex(m)] += 1
    out[f"D{D}"] = {"boot": b, "window": [w[0][0].isoformat(), w[-1][0].isoformat()], "samples": len(w), "throttle_frac": thr / len(w),
                    "reason_bits": dict(bits), "sm_clk_median_MHz": statistics.median(r[1] for r in busy), "last_boot_end_estimated": b == boots[-1]}
json.dump(out, open(sys.argv[2], "w"), indent=1, sort_keys=True)
for k, v in sorted(out.items()): print(k, v["boot"], v["window"][0][11:19], v["window"][1][11:19], v["samples"], round(v["throttle_frac"], 4), v["reason_bits"], v["sm_clk_median_MHz"])
