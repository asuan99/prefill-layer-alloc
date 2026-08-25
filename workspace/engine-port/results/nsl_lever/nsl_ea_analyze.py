#!/usr/bin/env python3
"""E-A scorer.  Implements PREREG_NSL_EA_CAPBANNER_2026-08-25.md sec3 exactly.

Reads the ENGINE's banner, never the CLI flag (capbind audit B7).  Produces no
performance number -- the job it scores sends no requests.
"""
import argparse, glob, json, os, re, subprocess, sys

CELLS = {"c24": 24, "c48": 48, "c96": 96, "c48m96": 48}
PATS = {"max_total_num_tokens": r"max_total_num_tokens=(\d+)",
        "max_running_requests": r"max_running_requests=(\d+)",
        "max_mamba_cache_size": r"max_mamba_cache_size:\s*(\d+)"}


def read_cell(d, tag, job):
    log = os.path.join(d, f"ea_{tag}_{job}_srv.log")
    out = {"log": os.path.basename(log), "present": os.path.exists(log)}
    if not out["present"]:
        return out
    with open(log, errors="ignore") as f:
        txt = f.read()
    for k, p in PATS.items():
        m = re.search(p, txt)
        out[k] = int(m.group(1)) if m else None
    m = re.search(r"mamba\s*\(\s*(\d+)\s*\)\s*([0-9.]+)\s*GB", txt)
    out["mamba_gb"] = float(m.group(2)) if m else None
    for pat, key in ((r"Not enough memory", "boot_refused_not_enough_memory"),
                     (r"CUDA out of memory", "boot_oom")):
        out[key] = bool(re.search(pat, txt))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cells = {t: read_cell(a.dir, t, a.tag) for t in CELLS}
    got = {t: c for t, c in cells.items()
           if c.get("max_total_num_tokens") and c.get("max_mamba_cache_size") is not None}
    sweep = ["c24", "c48", "c96"]

    # --- the registered predicates (prereg sec3), evaluated verbatim --------
    P1 = all(t in got and got[t]["max_mamba_cache_size"] == CELLS[t] for t in sweep)
    P2 = (all(t in got for t in sweep) and
          got["c24"]["max_total_num_tokens"] > got["c48"]["max_total_num_tokens"]
          > got["c96"]["max_total_num_tokens"])
    P3 = all(t in got and got[t]["max_running_requests"] == CELLS[t] for t in sweep)
    C1 = False
    if "c48m96" in got and "c96" in got:
        C1 = (got["c48m96"]["max_mamba_cache_size"] == 96 and
              abs(got["c48m96"]["max_total_num_tokens"]
                  - got["c96"]["max_total_num_tokens"])
              / float(got["c96"]["max_total_num_tokens"]) < 0.01)

    if len(got) < len(CELLS):
        label = "MEASUREMENT_ABSENT"
    elif not P1:
        label = "DERIVATION_BROKEN"
    elif not P3:
        label = "CAP_SILENTLY_CLAMPED"
    elif not P2:
        label = "KV_BUDGET_NOT_OPPOSED"
    elif not C1:
        label = "CONTROL_DISAGREES"
    else:
        label = "ARITHMETIC_CONFIRMED"

    def sh(c):
        try:
            return subprocess.run(c, capture_output=True, text=True,
                                  timeout=30).stdout.strip()
        except Exception as e:
            return f"unavailable: {e}"

    v = {"kind": "nsl_ea_banner", "tag": a.tag,
         "prereg": "PREREG_NSL_EA_CAPBANNER_2026-08-25.md",
         "label": label, "P1_derivation": P1, "P2_opposite_direction": P2,
         "P3_realized_equals_requested": P3, "C1_control": C1,
         "cells": cells, "requested_caps": CELLS,
         "git_head": sh(["git", "-C", "/scratch/ehmoon/whlee/prefill-layer-alloc",
                         "rev-parse", "HEAD"]),
         "node": os.environ.get("SLURMD_NODENAME", ""),
         "gpu": sh(["nvidia-smi", "--query-gpu=name,driver_version",
                    "--format=csv,noheader"]),
         "NOT_A_PERFORMANCE_RESULT": (
             "no requests were sent; this job cannot produce latency, "
             "throughput or goodput. It does NOT answer whether the cap binds "
             "(capbind audit B1) and does NOT close B3."),
         }
    with open(a.out, "w") as f:
        json.dump(v, f, indent=1, sort_keys=True)
    print(f"[ea] label={label}  P1={P1} P2={P2} P3={P3} C1={C1}")
    for t in CELLS:
        c = cells[t]
        print(f"   {t:8s} mamba={c.get('max_mamba_cache_size')} "
              f"kv_tokens={c.get('max_total_num_tokens')} "
              f"realized_cap={c.get('max_running_requests')}")
    print(f"[ea] wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
