#!/usr/bin/env python3
"""Realized decode-SM exposure per Stage 0 arm.

For every runtime_snapshot in the benchmark phase we know the stream group in
effect (dual_worker.py:608 -> sm_counts[stream_index]).  We ask, restricted to
samples where decode work was in flight (decode_running_batch_size > 0):
  - what decode-SM value was actually in force,
  - how that splits by whether a prefill batch was co-resident,
  - and the time-weighted version of the same.
decode_step_count is dead (always 0), so no step-weighted variant is possible.
"""
import json
import os
import re
from collections import Counter, defaultdict

DIR = "/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/stage0_xctrl"
PAT = re.compile(r"s0_(?P<arm>[MHT])_C(?P<ctx>\d+)_D(?P<d>\d+)_(?P<job>\d+)_telemetry\.jsonl$")

rows = []
for fn in sorted(os.listdir(DIR)):
    m = PAT.search(fn)
    if not m:
        continue
    g = m.groupdict()
    dec_sm = Counter()          # decode-active samples by realized decode SM
    dec_sm_cor = Counter()      # ... restricted to co-resident (prefill batch active)
    t_dec = defaultdict(float)  # wall time attributed to realized decode SM, decode-active only
    prev = None
    for line in open(os.path.join(DIR, fn)):
        try:
            e = json.loads(line)
        except Exception:
            continue
        if e.get("event") != "runtime_snapshot" or e.get("phase") != "benchmark":
            continue
        if e["decode_running_batch_size"] > 0:
            # stream group 0 == (108,0): decode is unsplit -> effective 108
            eff = 108 if e["decode_sms"] == 0 else e["decode_sms"]
            dec_sm[eff] += 1
            if e["prefill_active_batch_size"] > 0:
                dec_sm_cor[eff] += 1
            if prev is not None:
                dt = e["timestamp_monotonic_s"] - prev
                if 0 <= dt < 5.0:
                    t_dec[eff] += dt
        prev = e["timestamp_monotonic_s"]
    rows.append((g["arm"], int(g["ctx"]), int(g["d"]), dec_sm, dec_sm_cor, dict(t_dec)))

rows.sort(key=lambda r: (r[0], r[1], r[2]))
print(f"{'arm':>3} {'ctx':>6} {'nominal':>8} | {'realized decode-SM (decode-active samples)':<45} | "
      f"{'co-resident share':>17} | {'mean realized SM':>16}")
print("-" * 120)
for arm, ctx, d, dec_sm, dec_cor, t_dec in rows:
    n = sum(dec_sm.values())
    dist = ", ".join(f"{k}SM:{v} ({v/n*100:.0f}%)" for k, v in sorted(dec_sm.items(), key=lambda kv: -kv[1]))
    cor = sum(dec_cor.values())
    mean_sm = sum(k * v for k, v in dec_sm.items()) / n if n else float("nan")
    print(f"{arm:>3} {ctx:>6} {('D'+str(d)):>8} | {dist:<45} | {cor}/{n} ({cor/n*100:>4.0f}%) | {mean_sm:>15.1f}")

print()
print("=== D16 vs D108 arms side by side (the Stage 0 'no-contention anchor' contrast) ===")
by = {(a, c, d): (s, cor, t) for a, c, d, s, cor, t in rows}
print(f"{'arm':>3} {'ctx':>6} | {'D16 realized':>28} | {'D108 realized':>28}")
for arm in ("M", "H", "T"):
    for ctx in (4096, 8192, 16384):
        out = []
        for d in (16, 108):
            s, cor, t = by[(arm, ctx, d)]
            n = sum(s.values())
            mean_sm = sum(k * v for k, v in s.items()) / n
            top = max(s, key=s.get)
            out.append(f"{mean_sm:6.1f} SM mean, mode {top:3d} ({s[top]/n*100:3.0f}%)")
        print(f"{arm:>3} {ctx:>6} | {out[0]:>28} | {out[1]:>28}")
