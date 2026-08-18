#!/usr/bin/env python3
"""AUDIT of W-3: measure axis-2 (sequential forward count) DIRECTLY instead of
deriving it from mean ITL / "1 chunk per request".

decode forwards  <- telemetry `decode_iterations` counter (monotone, engine-side)
prefill forwards <- srv.log 'Prefill batch' lines + the real layer-split rule
                    ceil(n_layers / max(1, 65536 // extend_num_tokens))

Rounds are segmented by wall-clock gaps in the srv.log prefill stream and cross
checked against the bench JSONL durations (LO 3x~70.0s @rate3, HI 3x~36.3s @rate12).
"""
import json, re, math, glob, sys, datetime as dt, statistics as st

N_LAYERS, BUDGET = 54, 65536
pat = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\] Prefill batch, #new-seq: (\d+), #new-token: (\d+)")

def prefill_events(path):
    out = []
    for line in open(path, errors="replace"):
        m = pat.match(line)
        if not m: continue
        t = dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").timestamp()
        n, tok = int(m.group(2)), int(m.group(3))
        lpf = max(1, BUDGET // tok) if tok > 0 else N_LAYERS
        out.append((t, n, tok, math.ceil(N_LAYERS / lpf)))
    return out

def segment(ev, gap=3.0):
    segs, cur = [], [ev[0]]
    for e in ev[1:]:
        if e[0] - cur[-1][0] > gap:
            segs.append(cur); cur = [e]
        else:
            cur.append(e)
    segs.append(cur)
    return segs

def decode_iters(tel):
    lo = hi = None
    for line in open(tel, errors="replace"):
        try: e = json.loads(line)
        except Exception: continue
        if e.get("event") != "runtime_snapshot" or e.get("phase") != "benchmark": continue
        v = e.get("decode_iterations")
        if v is None: continue
        lo = v if lo is None else min(lo, v)
        hi = v if hi is None else max(hi, v)
    return lo, hi

for srv in sorted(glob.glob(sys.argv[1])):
    tel = srv.replace("_srv.log", "_telemetry.jsonl")
    ev = prefill_events(srv)
    segs = [s for s in segment(ev) if len(s) > 30]
    print(f"\n=== {srv}")
    for i, s in enumerate(segs):
        span = s[-1][0] - s[0][0]
        print(f"  seg{i}: span~{span:5.0f}s  batches={len(s):4d} seqs={sum(x[1] for x in s):4d} "
              f"tokens={sum(x[2] for x in s):7d} PREFILL_FORWARDS={sum(x[3] for x in s):4d} "
              f"| fwd/seq={sum(x[3] for x in s)/max(1,sum(x[1] for x in s)):.3f}")
    lo, hi = decode_iters(tel)
    print(f"  telemetry decode_iterations: min={lo} max={hi} -> DECODE_FORWARDS(whole bench)={hi-lo}")
