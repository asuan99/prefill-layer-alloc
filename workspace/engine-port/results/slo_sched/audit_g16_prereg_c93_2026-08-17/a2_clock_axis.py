#!/usr/bin/env python3
"""Audit (iv)/(i): is `t_boot0` on a COMMON time axis across nodes?

`drift_corrected_means` regresses the estimand on the sidecar's `t_boot0`,
which the harness fills from a per-NODE monotonic clock.  blk1 ran on gpu42 and
blk2-4 on gpu41, so a per-node clock origin difference would enter the A-3
drift correction as a constant shift on every blk1 point.  Measured here by
pairing each boot's `t_boot0` with the ABSOLUTE UTC timestamp of the co-tenancy
snapshot taken in the same boot iteration.

NON-DECISION-QUANTITY: no *_LO.jsonl / *_HI.jsonl is opened.
"""
import datetime as dt, glob, json, re, statistics
from pathlib import Path
HERE = Path(__file__).resolve().parent.parent
rows = []
for p in sorted(HERE.glob("g16_blk*_sidecar.json")):
    run = p.name[:-len("_sidecar.json")]
    d = json.loads(p.read_text())
    m = re.match(r"^g16_(blk\d)_(d\d+)_boot(\d+)_(\d+)$", run)
    blk, arm, boot, job = m.groups()
    pre = HERE / f"g16_{blk}_{arm}_boot{boot}_pre_cotenancy_{job}.txt"
    ts = re.search(r"TIMESTAMP=(\S+)", pre.read_text()).group(1)
    utc = dt.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc).timestamp()
    rows.append((d["node"], blk, arm, d["t_boot0"], utc, utc - d["t_boot0"]))
by_node = {}
for node, blk, arm, t0, utc, off in rows:
    by_node.setdefault(node, []).append(off)
print("offset = (absolute UTC of pre-cotenancy snapshot) - t_boot0, per node")
for node, offs in sorted(by_node.items()):
    print(f"  {node}: n={len(offs)} mean={statistics.fmean(offs):.3f} "
          f"min={min(offs):.3f} max={max(offs):.3f} spread={max(offs)-min(offs):.3f}")
nodes = sorted(by_node)
if len(nodes) == 2:
    d0 = statistics.fmean(by_node[nodes[0]]); d1 = statistics.fmean(by_node[nodes[1]])
    print(f"\ncross-node clock-origin difference ({nodes[0]} - {nodes[1]}) = {d0-d1:.3f} s")
    print("  (snapshot is written within ~1 s of t_boot0, so a |difference| much")
    print("   larger than the within-node spread means the two node clocks do NOT")
    print("   share an origin and A-3's pooled regression on t_boot0 is corrupted.)")
