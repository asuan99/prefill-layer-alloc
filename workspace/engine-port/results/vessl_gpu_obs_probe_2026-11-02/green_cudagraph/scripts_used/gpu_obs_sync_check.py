#!/usr/bin/env python3
"""Does torch.cuda.synchronize() wait for work on this stream? After launch + device sync, stream.synchronize()
times the residual: residual near the work's duration means the device sync returned early. Untraced."""
import json
import statistics as st
import sys
import time

import torch
from sgl_kernel.spatial import create_greenctx_stream_by_value

a = torch.randn(2048, 2048, device="cuda")
b = torch.randn(2048, 2048, device="cuda")
plain = torch.cuda.Stream()
sp, sd = create_greenctx_stream_by_value(16, 92, 0)
REPS = 20


def body():
    return torch.matmul(a, b)


def cell(stream, launch, sync_inside):
    ds, rs = [], []
    for _ in range(REPS):
        stream.synchronize()
        t0 = time.perf_counter()
        with torch.cuda.stream(stream):
            launch()
            if sync_inside:
                torch.cuda.synchronize()
        if not sync_inside:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        stream.synchronize()
        t2 = time.perf_counter()
        ds.append((t1 - t0) * 1e3)
        rs.append((t2 - t1) * 1e3)
    return {"device_sync_ms": round(st.median(ds), 3), "residual_ms": round(st.median(rs), 3)}


out = {}
for name, s in (("plain", plain), ("green_p16", sp), ("green_d92", sd)):
    with torch.cuda.stream(s):
        body()
    s.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        body()
    s.synchronize()
    for label, launch in (("eager", body), ("graph", g.replay)):
        for inside in (True, False):
            out[f"{name}/{label}/sync_{'inside' if inside else 'outside'}_stream_ctx"] = cell(s, launch, inside)
print(json.dumps(out, indent=0))
json.dump(out, open(sys.argv[1], "w"), indent=1)
