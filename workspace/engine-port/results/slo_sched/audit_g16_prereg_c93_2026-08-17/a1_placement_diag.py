#!/usr/bin/env python3
"""Audit (iv): block x node x concurrency placement diagnostics.

NON-DECISION-QUANTITY ONLY.  Reads sidecars (t_boot0/t_boot1/t_healthy/node/
gpu_uuid) and controller_summary residency; NEVER opens *_LO.jsonl / *_HI.jsonl
and never computes M_ttft / M_itl / donors / Delta / S_itl.
"""
import glob, json, os, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
side = {}
for p in sorted(HERE.glob("g16_blk*_sidecar.json")):
    d = json.loads(p.read_text())
    side[p.name[:-len("_sidecar.json")]] = d

def key(run): return (side[run]["block"], side[run]["arm"])

print("== per-boot wall structure (monotonic clock, seconds) ==")
print(f"{'block':6} {'arm':5} {'pos':3} {'boot_s':>8} {'bench_s':>8} {'total_s':>8} {'node':6} {'gpu_uuid8':10}")
byblock = {}
for run, d in side.items():
    byblock.setdefault(d["block"], []).append((d["t_boot0"], run, d))
order = {}
for blk in sorted(byblock):
    rows = sorted(byblock[blk])
    order[blk] = [r[2]["arm"] for r in rows]
    for pos, (t0, run, d) in enumerate(rows):
        boot_s = d["t_healthy"] - d["t_boot0"]
        total = d["t_boot1"] - d["t_boot0"]
        print(f"{blk:6} {d['arm']:5} {pos:3d} {boot_s:8.1f} {total-boot_s:8.1f} {total:8.1f} "
              f"{d['node']:6} {d['gpu_uuid'][4:12]:10}")
print()
print("== arm order per block ==")
for blk in sorted(order): print(" ", blk, order[blk])
print()
print("== mean position per arm (K5 check) ==")
arms = sorted({d['arm'] for d in side.values()}, key=lambda a: int(a[1:]))
for a in arms:
    pos = [order[b].index(a) for b in sorted(order)]
    print(f"  {a}: positions {pos}  mean {sum(pos)/len(pos):.3f}")
print()
print("== blk2 x blk3 concurrency overlap (same node gpu41, different GPUs) ==")
b2 = [(d["t_boot0"], d["t_boot1"], d["arm"]) for r, d in side.items() if d["block"] == "blk2"]
b3 = [(d["t_boot0"], d["t_boot1"], d["arm"]) for r, d in side.items() if d["block"] == "blk3"]
for s2, e2, a2 in sorted(b2):
    tot = e2 - s2
    parts = []
    for s3, e3, a3 in sorted(b3):
        ov = min(e2, e3) - max(s2, s3)
        if ov > 0:
            parts.append((a3, ov / tot))
    txt = ", ".join(f"{a}:{f*100:.1f}%" for a, f in parts)
    print(f"  blk2 {a2:4} overlapped by blk3 [{txt}]")
print()
print("== blocks: span and gaps ==")
for blk in sorted(byblock):
    rows = sorted(byblock[blk])
    t0 = rows[0][2]["t_boot0"]; t1 = rows[-1][2]["t_boot1"]
    print(f"  {blk}: span {t1-t0:7.1f}s  node {rows[0][2]['node']}  gpu {rows[0][2]['gpu_uuid'][4:12]}")
print()
print("== residency_fraction (controller_summary; A-2 conditional-label baseline) ==")
for run in sorted(side, key=lambda r: (side[r]["block"], int(side[r]["arm"][1:]))):
    p = HERE / f"{run}_controller_summary.json"
    if not p.exists():
        print(f"  {run}: MISSING controller_summary"); continue
    s = json.loads(p.read_text())
    rf = s.get("residency_fraction") or {}
    top = sorted(rf.items(), key=lambda kv: -kv[1])[:3]
    print(f"  {side[run]['block']:6} {side[run]['arm']:5} transitions={s.get('split_transitions')} "
          f"top_residency={[(k, round(v,3)) for k,v in top]}")
