#!/usr/bin/env python3
"""M4R prerequisite probe 1 -- is the exposure variable aliased with realized SM?

GPU 0.  Registered by audit_m4r_rules_2026-08-27/VERDICT.md sec "반드시 먼저 살 프로브".

Blocking condition (registered BEFORE running):
    if  P(decode_sms == D | confined) > 0.90  AND  P(decode_sms == 108 | free) > 0.90
    then the label `confined` is a PROXY FOR THE PARTITION, R is not identified,
    and M4R's registered estimand is BARRED.   Label: EXPOSURE_ALIASED.

All estimators are TIME-WEIGHTED (M4R sec3(c)); gaps > MAX_GAP_S are dropped.
"""
import json, glob, os, re, sys
from collections import defaultdict

MAX_GAP_S = 1.0
ALIAS_BLOCK = 0.90
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "alias_verdict.json")

def cell_D(path):
    m = re.search(r'_d(\d+)_', os.path.basename(path))
    return int(m.group(1)) if m else None

def scan(path):
    # field-level extraction: full json.loads on ~60k lines/file x 28 files is
    # too slow, and only five fields are used.  Same values, same order.
    F = {k: re.compile(r'"%s":\s*(-?[\d.]+)' % k) for k in
         ("timestamp_monotonic_s", "prefill_active_batch_size",
          "decode_running_batch_size", "decode_sms", "decode_iterations")}
    rows = []
    for ln in open(path):
        if '"runtime_snapshot"' not in ln or '"benchmark"' not in ln:
            continue
        v = {}
        for k, rx in F.items():
            m = rx.search(ln)
            v[k] = float(m.group(1)) if m else None
        if v["timestamp_monotonic_s"] is None:
            continue
        rows.append((v["timestamp_monotonic_s"],
                     v["prefill_active_batch_size"] or 0,
                     v["decode_running_batch_size"] or 0,
                     int(v["decode_sms"]) if v["decode_sms"] is not None else None,
                     v["decode_iterations"]))
    rows.sort()
    acc = defaultdict(float); it = defaultdict(float)
    for (t0,p0,d0,sm0,i0),(t1,_,_,_,i1) in zip(rows, rows[1:]):
        dt = t1 - t0
        if dt <= 0 or dt > MAX_GAP_S or d0 <= 0:
            continue
        grp = "confined" if p0 > 0 else "free"
        di = (i1 - i0) if (i0 is not None and i1 is not None and i1 >= i0) else 0
        acc[(grp, "T")] += dt
        it[(grp, "N")] += di
        acc[(grp, f"sm{sm0}")] += dt
    return acc, it, len(rows)

results = []
for path in sorted(glob.glob("/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/"
                             "engine-port/results/slo_sched/g16_blk*_d*_boot1_*_telemetry.jsonl")):
    D = cell_D(path)
    acc, it, n = scan(path)
    Tc, Tf = acc[("confined","T")], acc[("free","T")]
    if Tc <= 0 or Tf <= 0:
        results.append({"file": os.path.basename(path), "D": D, "status": "no_contrast"}); continue
    p_conf_D  = acc[("confined", f"sm{D}")] / Tc
    p_free_108 = acc[("free", "sm108")] / Tf
    r_conf = it[("confined","N")] / Tc
    r_free = it[("free","N")] / Tf
    # R_matched needs free-time AT D SM; report how much of it exists at all
    T_free_D = acc[("free", f"sm{D}")]
    results.append({
        "file": os.path.basename(path), "D": D, "snapshots": n,
        "T_confined_s": round(Tc,2), "T_free_s": round(Tf,2),
        "P_decode_sms_eq_D_given_confined": round(p_conf_D,4),
        "P_decode_sms_eq_108_given_free":   round(p_free_108,4),
        "T_free_at_D_s": round(T_free_D,2),
        "R_registered": round((r_free/r_conf),4) if r_conf > 0 else None,
    })

ok = [r for r in results if "P_decode_sms_eq_D_given_confined" in r]
aliased = [r for r in ok
           if r["P_decode_sms_eq_D_given_confined"] > ALIAS_BLOCK
           and r["P_decode_sms_eq_108_given_free"] > ALIAS_BLOCK]
out = {"probe": "M4R-alias", "gpu_hours": 0.0, "max_gap_s": MAX_GAP_S,
       "alias_block_threshold": ALIAS_BLOCK, "cells": results,
       "n_cells": len(ok), "n_aliased": len(aliased),
       "verdict": ("EXPOSURE_ALIASED" if ok and len(aliased) == len(ok)
                   else "PARTIALLY_ALIASED" if aliased
                   else "NOT_ALIASED" if ok else "PROBE_INVALID")}
out["reading"] = {
 "EXPOSURE_ALIASED": "The `confined` label is a proxy for the partition in EVERY cell. "
   "M4R rev1's registered R is not identified; it measures the C2 SM lever. "
   "The registered estimand is BARRED and rev2 must move to an SM-matched contrast.",
}.get(out["verdict"], "see cells")
json.dump(out, open(OUT,"w"), indent=2, ensure_ascii=False, sort_keys=True)
print(f"verdict={out['verdict']}  aliased {out['n_aliased']}/{out['n_cells']} cells\n")
print(f"{'cell':6} {'T_conf':>8} {'T_free':>8} {'P(D|conf)':>10} {'P(108|free)':>12} {'T_free@D':>9} {'R_reg':>7}")
for r in sorted(ok, key=lambda x: x["D"] or 0):
    print(f"d{r['D']:<5} {r['T_confined_s']:8.1f} {r['T_free_s']:8.1f} "
          f"{r['P_decode_sms_eq_D_given_confined']:10.4f} {r['P_decode_sms_eq_108_given_free']:12.4f} "
          f"{r['T_free_at_D_s']:9.2f} {r['R_registered']:7.3f}")
