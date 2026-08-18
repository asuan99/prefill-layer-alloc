#!/usr/bin/env python3
"""Same CO_RESIDENT definition as realized_pin_check.py, applied to G16.
Puts G16 on the SAME observable as the s8 L-axis grid so the two can be
compared.  G16's residency_fraction[D] is a DIFFERENT observable and is not
mixed in here."""
import json, glob, re, collections, statistics as st

def scan(p):
    n=co=0; b=[]
    for line in open(p, errors="replace"):
        try: e=json.loads(line)
        except Exception: continue
        if e.get("event")!="runtime_snapshot" or e.get("phase")!="benchmark": continue
        if e.get("decode_running_batch_size",0)<=0: continue
        n+=1; b.append(e["decode_running_batch_size"])
        if e.get("prefill_active_batch_size",0)>0: co+=1
    return n,co,b

rows=collections.defaultdict(list)
for p in sorted(glob.glob("g16_blk*_d*_boot*_telemetry.jsonl")):
    m=re.search(r"g16_(blk\d)_d(\d+)_",p)
    if not m: continue
    n,co,b=scan(p)
    if n: rows[int(m.group(2))].append((co/n, st.median(b), max(b), n))

print("G16 (Zamba2-2.7B, ShareGPT L~341 / out~237, LO+HI 합산 boot)")
print(f"{'D':>4} {'boots':>6} {'CO_RESIDENT_frac':>20} {'med B':>7} {'max B':>6} {'snaps':>7}")
for d in sorted(rows):
    v=rows[d]; fr=[x[0] for x in v]
    sd=st.stdev(fr) if len(fr)>1 else 0.0
    print(f"{d:4d} {len(v):6d} {st.fmean(fr):11.4f} ± {sd:.4f} "
          f"{st.fmean([x[1] for x in v]):7.1f} {max(x[2] for x in v):6d} {sum(x[3] for x in v):7d}")
