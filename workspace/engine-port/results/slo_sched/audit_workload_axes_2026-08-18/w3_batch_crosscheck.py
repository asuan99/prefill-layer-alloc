#!/usr/bin/env python3
"""Cross-check of WORKLOAD_AXES 2.1 "batch tokens (derived) 32.7 matches the
independently reconstructed split-state decode batch 33.04 +- 0.86".

Recomputes, per arm and per load phase (LO rate3 rounds vs HI rate12 rounds):
  meanB_all   = time-mean decode_running_batch_size over decode-active snapshots
  meanB_split = same, conditional on realized decode_sms == D (nominal split)
  tokens/decode-forward = total_output_tokens / delta(decode_iterations)
"""
import json, glob, re, statistics as st, collections

def rounds(path, idle_gap=2.0):
    rows = []
    for line in open(path, errors="replace"):
        try: e = json.loads(line)
        except Exception: continue
        if e.get("event") != "runtime_snapshot" or e.get("phase") != "benchmark": continue
        rows.append(e)
    rows.sort(key=lambda e: e["timestamp_monotonic_s"])
    segs, cur, last = [], [], None
    for e in rows:
        b = e.get("decode_running_batch_size", 0)
        if b > 0:
            if last is not None and e["timestamp_monotonic_s"] - last > idle_gap and cur:
                segs.append(cur); cur = []
            last = e["timestamp_monotonic_s"]; cur.append(e)
        elif cur: cur.append(e)
    if cur: segs.append(cur)
    return [s for s in segs if sum(1 for x in s if x.get("decode_running_batch_size",0) > 0) > 50]

OUT_TOK = 47376   # per round, identical across rounds/arms (same ShareGPT trace)
res = collections.defaultdict(lambda: collections.defaultdict(list))
for p in sorted(glob.glob("g16_blk*_d*_boot1_*_telemetry.jsonl")):
    D = int(re.search(r"_d(\d+)_", p).group(1))
    for s in rounds(p):
        act = [x for x in s if x.get("decode_running_batch_size",0) > 0]
        span = act[-1]["timestamp_monotonic_s"] - act[0]["timestamp_monotonic_s"]
        phase = "HI" if span < 50 else "LO"
        dfwd = act[-1]["decode_iterations"] - act[0]["decode_iterations"]
        b = [x["decode_running_batch_size"] for x in act]
        bs = [x["decode_running_batch_size"] for x in act if x.get("decode_sms") == D]
        res[D][phase].append((st.fmean(b), st.fmean(bs) if bs else float("nan"),
                              OUT_TOK/dfwd if dfwd else float("nan"), dfwd, len(bs)))

print(f"{'D':>4} {'ph':>3} {'nrnd':>5} {'meanB_all':>10} {'meanB_SPLIT':>12} {'tok/dec-fwd':>12} {'dec_fwd':>9} {'n_split_snap':>12}")
for D in sorted(res):
    for ph in ("LO","HI"):
        v = res[D][ph]
        if not v: continue
        f = lambda i: st.fmean([x[i] for x in v])
        sd = lambda i: (st.stdev([x[i] for x in v]) if len(v)>1 else 0)
        print(f"{D:4d} {ph:>3} {len(v):5d} {f(0):10.2f} {f(1):8.2f}±{sd(1):.2f} "
              f"{f(2):12.2f} {f(3):9.0f} {sum(x[4] for x in v):12d}")
