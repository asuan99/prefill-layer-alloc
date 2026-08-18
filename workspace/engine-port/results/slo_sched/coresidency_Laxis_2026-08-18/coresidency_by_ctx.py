#!/usr/bin/env python3
"""L-axis co-residency re-aggregation (GPU 0) -- no new estimand.

Reuses `s8_scaleup/realized_pin_check.py`'s CO_RESIDENT definition VERBATIM:
  denominator = benchmark runtime_snapshots with decode_running_batch_size > 0
  numerator   = those with prefill_active_batch_size > 0
This is "prefill active while decode active", which is a DIFFERENT observable
from G16's residency_fraction[D] ("realized partition == nominal split").
CONSENSUS already records that the two diverge; they are never mixed here.
"""
import json, glob, re, sys, collections, statistics as st

def scan(path):
    n_dec = co = 0; batches = []
    with open(path, errors="replace") as fh:
        for line in fh:
            try: e = json.loads(line)
            except Exception: continue
            if e.get("event") != "runtime_snapshot" or e.get("phase") != "benchmark":
                continue
            if e.get("decode_running_batch_size", 0) <= 0:
                continue
            n_dec += 1
            batches.append(e["decode_running_batch_size"])
            if e.get("prefill_active_batch_size", 0) > 0:
                co += 1
    return n_dec, co, batches

rows = collections.defaultdict(list)
for p in sorted(glob.glob("*_telemetry.jsonl")):
    m = re.search(r"_(\w+?)_C(\d+)_d(\d+)_", p)
    if not m: continue
    arm, ctx, d = m.group(1), int(m.group(2)), int(m.group(3))
    n_dec, co, b = scan(p)
    if n_dec == 0: continue
    rows[(ctx, arm, d)].append((co / n_dec, st.median(b), max(b), n_dec))

print(f"{'ctx':>6} {'arm':>5} {'D':>4} {'boots':>6} {'CO_RESIDENT_frac':>18} {'med B':>7} {'max B':>6} {'snaps':>7}")
agg = {}
for k in sorted(rows, key=lambda x: (x[0], x[1], x[2])):
    v = rows[k]
    fr = [x[0] for x in v]
    print(f"{k[0]:6d} {k[1]:>5} {k[2]:4d} {len(v):6d} "
          f"{st.fmean(fr):10.4f} ± {(st.stdev(fr) if len(fr)>1 else 0):.4f} "
          f"{st.fmean([x[1] for x in v]):7.1f} {max(x[2] for x in v):6d} {sum(x[3] for x in v):7d}")
    agg[k] = st.fmean(fr)

print("\n=== ctx=1024 -> 4096 (L 4x, out=512 고정) ===")
for arm in sorted({k[1] for k in agg}):
    for d in sorted({k[2] for k in agg if k[1] == arm}):
        a, b = agg.get((1024, arm, d)), agg.get((4096, arm, d))
        if a and b:
            print(f"  {arm} d{d}: {a:.4f} -> {b:.4f}   ({b/a:.2f}x)")
