#!/usr/bin/env python3
"""AUDIT of CORESIDENCY_L_AXIS C-2 (positive control), C-3 (k identity),
C-5 (L elasticity), C-6 (B collapse).  Run from results/s8_scaleup/.

C-2: runs the ORIGINAL tool realized_pin_check.py as a SUBPROCESS on s8 cells
     and compares to the note's inline reimplementation -> tests the s8 leg,
     which the note's G16-only control does NOT cover.
C-3: shows the k-recalibration is a 1-parameter fit to 1 point (residual == 0
     by construction) and that the "2.8-10.3x" misfit is NOT constant across the
     two s8 points -- i.e. it carries exponent information, contradicting C-5.
C-5: recomputes the L elasticity using ONLY the clean ctx1024 job (865493),
     excluding the keepalive-dead job 865533 that CONSENSUS already flags.
C-6: recomputes med/max decode batch by ctx.
"""
import json, glob, re, subprocess, sys, math, statistics as st, collections

def scan(path):
    n = c = 0; b = []
    for line in open(path, errors="replace"):
        try: e = json.loads(line)
        except Exception: continue
        if e.get("event") != "runtime_snapshot" or e.get("phase") != "benchmark": continue
        if e.get("decode_running_batch_size", 0) <= 0: continue
        n += 1; b.append(e["decode_running_batch_size"])
        if e.get("prefill_active_batch_size", 0) > 0: c += 1
    return n, c, b

# ---------------- C-2: independent-tool control on the S8 leg ----------------
print("=== C-2 control, s8 leg: inline reimplementation vs realized_pin_check.py subprocess ===")
mism = 0
for p in sorted(glob.glob("s8_deconf_*_C*_d*_telemetry.jsonl"))[:12]:
    D = int(re.search(r"_d(\d+)_", p).group(1))
    n, c, b = scan(p)
    out = subprocess.run([sys.executable, "realized_pin_check.py", p, str(D)],
                         capture_output=True, text=True).stdout
    m = re.search(r"CO_RESIDENT_frac: ([0-9.]+)", out)
    tool = float(m.group(1)) if m else float("nan")
    ok = abs(tool - c/n) < 5e-4
    mism += (not ok)
    print(f"  {p[:52]:52s} inline={c/n:.4f} tool={tool:.4f} {'OK' if ok else '**MISMATCH**'}")
print(f"  -> mismatches: {mism}")

# ---------------- C-3: the k identity ---------------------------------------
print("\n=== C-3: is the k recalibration an identity? ===")
w_g16 = 0.0527; Lo_g16 = 341/237
k = (w_g16/(1-w_g16))/Lo_g16
print(f"  k fitted on ONE point (G16 d44): k={k:.4f}; residual at that point = "
      f"{(w_g16/(1-w_g16)) - k*Lo_g16:.2e}  <- zero BY CONSTRUCTION (1 param, 1 obs, df=0)")
for name, Lo, obs in (("s8 ctx1024 d44", 2.0, (0.506, 0.737)), ("s8 ctx4096 d44", 8.0, (0.652, 0.929))):
    odds = k*Lo; pred = odds/(1+odds)
    print(f"  {name}: L/out={Lo}  pred w={pred*100:.1f}%  obs {obs[0]*100:.1f}-{obs[1]*100:.1f}%  "
          f"ratio {obs[0]/pred:.1f}-{obs[1]/pred:.1f}x")
print("  -> the misfit ratio FALLS from ~7-10x (L/out=2) to ~2.8-3.9x (L/out=8) within ONE campaign,")
print("     with the SAME k.  A pure 'k does not transfer' story predicts an IDENTICAL ratio at both")
print("     points.  The decline IS the exponent: it says odds grows far SLOWER than (L/out)^1.")

# ---------------- C-5: elasticity, contaminated vs clean --------------------
print("\n=== C-5: L elasticity, pooled(865493+865533) vs clean job only (865493) ===")
vals = collections.defaultdict(dict)
for p in sorted(glob.glob("s8_deconf_*_C*_*_telemetry.jsonl")):
    m = re.match(r"s8_deconf_(\w+?)_C(\d+)_(\w+?)_(\d+)(?:_blk\d)?_telemetry", p)
    if not m: continue
    arm, ctx, cell, job = m.group(1), int(m.group(2)), m.group(3), m.group(4)
    n, c, b = scan(p)
    if n: vals[(arm, cell)][(ctx, job)] = (c/n, st.median(b), max(b))
odds = lambda w: w/(1-w)
print(f"{'arm':>5} {'cell':>5} {'w1024 pooled':>13} {'w1024 CLEAN':>12} {'w4096':>8} "
      f"{'alpha pooled':>13} {'alpha CLEAN':>12}")
for (arm, cell), d in sorted(vals.items()):
    if cell not in ("d16", "d24", "d44", "d92", "np"): continue
    lo = [v[0] for (c_, j), v in d.items() if c_ == 1024]
    clean = [v[0] for (c_, j), v in d.items() if c_ == 1024 and j == "865493"]
    hi = [v[0] for (c_, j), v in d.items() if c_ == 4096]
    if not (lo and hi and clean): continue
    lo_m, cl_m, hi_m = st.fmean(lo), clean[0], hi[0]
    ap = math.log(odds(hi_m)/odds(lo_m))/math.log(4)
    ac = math.log(odds(hi_m)/odds(cl_m))/math.log(4)
    print(f"{arm:>5} {cell:>5} {lo_m*100:12.1f}% {cl_m*100:11.1f}% {hi_m*100:7.1f}% "
          f"{ap:13.2f} {ac:12.2f}")

# ---------------- C-6: decode batch by ctx ---------------------------------
print("\n=== C-6: decode batch by ctx (deconf cells), split by job ===")
agg = collections.defaultdict(list)
for (arm, cell), d in vals.items():
    for (ctx, job), v in d.items():
        agg[(ctx, job)].append((v[1], v[2]))
for k2 in sorted(agg):
    v = agg[k2]
    print(f"  ctx={k2[0]:5d} job={k2[1]:>8}  med B: {min(x[0] for x in v):.0f}-{max(x[0] for x in v):.0f}"
          f"   max B: {min(x[1] for x in v)}-{max(x[1] for x in v)}   (n_cells={len(v)})")
