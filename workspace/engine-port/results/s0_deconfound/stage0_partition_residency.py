#!/usr/bin/env python3
"""Stage 0 re-aggregation: how much of the decode work actually ran on the
sub-108 green-context partition?

PIN_CHECK in stage0_pdmux_capture.sbatch only histogrammed `target_decode_sms`
(the FixedPolicy *target*). The runtime_snapshot rows carry the REALIZED
partition: dual_worker.py:608 sets prefill_sms/decode_sms from
sm_counts[arbiter.stream_index], i.e. the stream group in effect for that
event-loop iteration.  stream_index 0 == (108, 0) == unsplit full-GPU stream.

Three weightings, all reported (they answer different questions):
  1. iteration-sample count  (biased toward cheap idle iterations)
  2. wall-time between samples (residency in seconds)
  3. decode-step attribution  (Delta decode_step_count between samples)
Sampling is every PDMUX_DUAL_WORKER_TRACE_EVERY=32 syncs (~16 iterations), so
step attribution is only unambiguous when consecutive samples agree on the
config; disagreeing intervals are reported separately as `ambiguous`.
"""
import json
import os
import re
import sys
from collections import Counter, defaultdict

DIR = "/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/stage0_xctrl"
PAT = re.compile(r"s0_(?P<arm>[MHT])_C(?P<ctx>\d+)_D(?P<d>\d+)_(?P<job>\d+)_telemetry\.jsonl$")


def analyze(path):
    snaps = []
    ctl = Counter()
    trans = 0
    for line in open(path):
        try:
            e = json.loads(line)
        except Exception:
            continue
        ev = e.get("event")
        if ev == "runtime_snapshot":
            if e.get("phase") != "benchmark":
                continue
            snaps.append(e)
        elif ev == "controller_decision":
            ctl[e.get("target_decode_sms")] += 1
        elif ev == "split_transition":
            trans += 1

    cnt = Counter()            # samples by (p_sm, d_sm)
    cnt_dec = Counter()        # samples with decode work in flight
    cnt_dec_pf = Counter()     # decode work AND prefill batch active
    time_w = defaultdict(float)
    step_w = defaultdict(int)
    ambiguous_steps = 0
    total_steps = 0
    dec_active = 0

    prev = None
    for s in snaps:
        key = (s["prefill_sms"], s["decode_sms"])
        cnt[key] += 1
        if s["decode_running_batch_size"] > 0:
            dec_active += 1
            cnt_dec[key] += 1
            if s["prefill_active_batch_size"] > 0:
                cnt_dec_pf[key] += 1
        if prev is not None:
            dt = s["timestamp_monotonic_s"] - prev["timestamp_monotonic_s"]
            pkey = (prev["prefill_sms"], prev["decode_sms"])
            if 0 <= dt < 5.0:  # drop gaps across client reps / server stalls
                time_w[pkey] += dt
            dstep = s["decode_step_count"] - prev["decode_step_count"]
            if dstep > 0:
                total_steps += dstep
                if pkey == key:
                    step_w[key] += dstep
                else:
                    ambiguous_steps += dstep
        prev = s

    return dict(
        n_snap=len(snaps), n_dec=dec_active, cnt=cnt, cnt_dec=cnt_dec,
        cnt_dec_pf=cnt_dec_pf, time_w=dict(time_w), step_w=dict(step_w),
        ambiguous_steps=ambiguous_steps, total_steps=total_steps,
        ctl=dict(ctl), transitions=trans,
    )


def frac_split(counter):
    """fraction of mass on a SPLIT partition (decode_sms not 0 and not 108)."""
    tot = sum(counter.values())
    if not tot:
        return float("nan"), 0
    split = sum(v for (p, d), v in counter.items() if 0 < d < 108)
    return split / tot, tot


def main():
    rows = []
    for fn in sorted(os.listdir(DIR)):
        m = PAT.search(fn)
        if not m:
            continue
        g = m.groupdict()
        r = analyze(os.path.join(DIR, fn))
        rows.append((g["arm"], int(g["ctx"]), int(g["d"]), g["job"], r))

    rows.sort(key=lambda x: (x[0], x[1], x[2]))
    hdr = (f"{'arm':>3} {'ctx':>6} {'D':>4} {'job':>7} | {'n_snap':>6} {'n_dec':>6} | "
           f"{'split%samp':>10} {'split%dec':>9} {'split%time':>10} {'split%step':>10} | "
           f"{'amb.steps%':>10} {'switch':>6} | realized configs (decode-active samples)")
    print(hdr)
    print("-" * len(hdr))
    for arm, ctx, d, job, r in rows:
        f_samp, _ = frac_split(r["cnt"])
        f_dec, _ = frac_split(r["cnt_dec"])
        f_time, _ = frac_split(Counter(r["time_w"]))
        f_step, _ = frac_split(Counter(r["step_w"]))
        amb = (r["ambiguous_steps"] / r["total_steps"] * 100) if r["total_steps"] else float("nan")
        cfgs = ", ".join(f"{p}/{dd}:{v}" for (p, dd), v in sorted(r["cnt_dec"].items(), key=lambda kv: -kv[1]))
        print(f"{arm:>3} {ctx:>6} {d:>4} {job:>7} | {r['n_snap']:>6} {r['n_dec']:>6} | "
              f"{f_samp*100:>9.2f}% {f_dec*100:>8.2f}% {f_time*100:>9.2f}% {f_step*100:>9.2f}% | "
              f"{amb:>9.1f}% {r['transitions']:>6} | {cfgs}")

    print()
    print("decode-active samples, split by whether a prefill batch was co-resident")
    print(f"{'arm':>3} {'ctx':>6} {'D':>4} | {'dec&pf samples':>14} {'of which split':>15} | "
          f"{'dec,no-pf samples':>18} {'of which split':>15}")
    for arm, ctx, d, job, r in rows:
        dec_pf = sum(r["cnt_dec_pf"].values())
        dec_pf_split = sum(v for (p, dd), v in r["cnt_dec_pf"].items() if 0 < dd < 108)
        dec_all = sum(r["cnt_dec"].values())
        dec_nopf = dec_all - dec_pf
        dec_split_all = sum(v for (p, dd), v in r["cnt_dec"].items() if 0 < dd < 108)
        dec_nopf_split = dec_split_all - dec_pf_split
        pf_pct = (dec_pf_split / dec_pf * 100) if dec_pf else float("nan")
        nopf_pct = (dec_nopf_split / dec_nopf * 100) if dec_nopf else float("nan")
        print(f"{arm:>3} {ctx:>6} {d:>4} | {dec_pf:>14} {pf_pct:>14.2f}% | "
              f"{dec_nopf:>18} {nopf_pct:>14.2f}%")

    print()
    print("controller target histogram (what PIN_CHECK saw) vs realized split residency")
    for arm, ctx, d, job, r in rows:
        f_dec, n = frac_split(r["cnt_dec"])
        print(f"{arm:>3} C{ctx:<6} D{d:<4} target_hist={r['ctl']}  realized_split_frac(decode-active)={f_dec*100:.2f}% (n={n})")


if __name__ == "__main__":
    sys.exit(main())
