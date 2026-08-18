#!/usr/bin/env python3
"""AUDIT of CORESIDENCY_L_AXIS C-4 ("direction is robust: G16 1.8-18.3% vs
s8 ctx4096 46-93% is an order-of-magnitude gap no noise model can flip")
and of the premise "s8 ctx{1024,4096} is an L-ONLY contrast".

s8_sweep.sbatch:104-112 names KEEPA_N/KEEPA_REPS explicitly as "the duty-cycle
knob" for the keepalive PREFILL generator.  CO_RESIDENT_frac IS a prefill duty
cycle.  This script extracts, per job, the keepalive settings actually used and
the realized keepalive throughput, next to the CO_RESIDENT_frac the L-axis note
attributes to L.
"""
import json, glob, re, collections, statistics as st

# ---- per-job keepalive health from the campaign .out logs -------------------
print("=== keepalive configuration / health, per job (from s8sweep*.out) ===")
for out in sorted(glob.glob("s8sweep*.out")):
    ctx = keepn = mode = None
    done, errs, occ, comp = [], [], [], []
    for line in open(out, errors="replace"):
        m = re.search(r"MODE=(\w+) ARM=\S+ .*CTX=(\d+) .*keepN=(\d+)", line)
        if m: mode, ctx, keepn = m.group(1), int(m.group(2)), int(m.group(3))
        if line.startswith("{"):
            try: e = json.loads(line)
            except Exception: continue
            if "keepalive_done" in e:
                done.append(e["keepalive_done"]); errs.append(e["keepalive_errors"])
                occ.append(e.get("occupancy_mean_in_window", 0))
                comp.append(e.get("requests_completed", 0))
    if done:
        print(f"{out:24s} MODE={mode:8s} CTX={ctx:5d} KEEPA_N={keepn}  "
              f"keepalive_done: med={st.median(done):7.1f} min={min(done)} max={max(done)}  "
              f"keepalive_errors: med={st.median(errs):8.1f}  "
              f"occupancy med={st.median(occ):5.2f}  reqs_completed med={st.median(comp):5.1f}  n_reps={len(done)}")

# ---- per-JOB CO_RESIDENT_frac for the cells the note pools -----------------
def co(path):
    n = c = 0; b = []
    for line in open(path, errors="replace"):
        try: e = json.loads(line)
        except Exception: continue
        if e.get("event") != "runtime_snapshot" or e.get("phase") != "benchmark": continue
        if e.get("decode_running_batch_size", 0) <= 0: continue
        n += 1; b.append(e["decode_running_batch_size"])
        if e.get("prefill_active_batch_size", 0) > 0: c += 1
    return (c/n, st.median(b), n) if n else (None, None, 0)

print("\n=== CO_RESIDENT_frac per FILE (job is in the filename) -- deconf cells ===")
print(f"{'arm':>5} {'ctx':>5} {'D':>4} {'job':>8} {'CO_RESIDENT':>12} {'medB':>6} {'snaps':>7}")
rows = collections.defaultdict(dict)
for p in sorted(glob.glob("s8_deconf_*_telemetry.jsonl")):
    m = re.match(r"s8_deconf_(\w+?)_C(\d+)_(\w+?)_(\d+)(?:_blk\d)?_telemetry", p)
    if not m: continue
    arm, ctx, cell, job = m.group(1), int(m.group(2)), m.group(3), m.group(4)
    f, mb, n = co(p)
    if f is None: continue
    print(f"{arm:>5} {ctx:5d} {cell:>4} {job:>8} {f:12.4f} {mb:6.1f} {n:7d}")
    rows[(arm, cell)][(ctx, job)] = f

print("\n=== the note's 'L-only' ratio, decomposed by which ctx1024 job you use ===")
for (arm, cell), d in sorted(rows.items()):
    if cell != "d44": continue
    hi = [v for (c, j), v in d.items() if c == 4096]
    for (c, j), v in sorted(d.items()):
        if c == 1024 and hi:
            print(f"  {arm} {cell}: ctx1024(job {j})={v:.4f} -> ctx4096={hi[0]:.4f}  ratio={hi[0]/v:.2f}x")
