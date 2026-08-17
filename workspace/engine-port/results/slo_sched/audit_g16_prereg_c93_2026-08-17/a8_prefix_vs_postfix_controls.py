#!/usr/bin/env python3
"""Audit (iii)(b): C-5(1) claims PC-C / PC-D verdicts, `identified` and the
bootstrap fractions are IDENTICAL before and after the tie guard.  Verified
here by monkey-patching `identify_donor` back to the pre-fix version and
re-running the same controls.  Old grids only -- no campaign artifact touched.
"""
import json, sys
sys.path.insert(0, "/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/slo_sched")
import g16_analyze as G
sys.path.insert(0, "/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/slo_sched/audit_g16_prereg_c93_2026-08-17")
from a3_tie_guard_adversarial import identify_donor_prefix
from pathlib import Path
HERE = Path("/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/slo_sched")
G20 = G.ENGINE_PORT / "results" / "g2_0_hard"

def snapshot(controls):
    out = {}
    d = controls["PC_C"]["decision"]
    out["PC-C"] = (d["verdict"], d["D_itl"]["identified"],
                   round(d["D_itl"]["bootstrap_rule"]["fraction"], 6),
                   d["D_ttft"]["identified"],
                   round(d["D_ttft"]["bootstrap_rule"]["fraction"], 6))
    for ph in ("A", "B"):
        p = controls["PC_D"]["phases"][ph]
        out[f"PC-D {ph}"] = (p["verdict"], p["D_itl"]["identified"],
                             round(p["D_itl"]["bootstrap_rule"]["fraction"], 6),
                             p["D_ttft"]["identified"],
                             round(p["D_ttft"]["bootstrap_rule"]["fraction"], 6))
    out["all_passed"] = controls["all_passed"]
    return out

post = snapshot(G.run_controls(HERE, G20))
_real = G.identify_donor
def prefix_wrapper(grid, key):
    r = identify_donor_prefix(grid, key)
    return {"identified": r["identified"], "arm": r["rank_arm"] if r["identified"] else None,
            "decode_sm": G.arm_sm(r["rank_arm"]) if r["identified"] and r["rank_arm"] else None,
            "rank_rule": {"winner": r["rank_arm"], "blocks_won": 0, "n_blocks": 0,
                          "passes": r["rank_ok"], "per_block": {}, "per_block_tied": {},
                          "n_blocks_tied": 0},
            "bootstrap_rule": {"winner": r["boot_arm"], "fraction": r["boot_frac"],
                               "passes": r["boot_frac"] >= G.K1_BOOTSTRAP_MIN_FRAC,
                               "tied_fraction": 0.0, "distribution": {}},
            "rules_agree": r["rank_arm"] == r["boot_arm"], "note": "PRE-FIX"}
G.identify_donor = prefix_wrapper
pre = snapshot(G.run_controls(HERE, G20))
G.identify_donor = _real
print("cell                 post-fix                                pre-fix")
same = True
for k in post:
    flag = "" if post[k] == pre[k] else "   <-- DIFFERS"
    if post[k] != pre[k]:
        same = False
    print(f"{k:20} {str(post[k]):38}  {str(pre[k])}{flag}")
print("\nC-5(1) 'identical before/after on PC-C/PC-D':", "CONFIRMED" if same else "REFUTED")
