#!/usr/bin/env python3
"""Audit (i): two candidate NEW defects introduced/left by F1 and F3.

D1  the Delta_SLO ladder (primary-C) emits sec6 verdict labels
    (TAX_POSITIVE / NO_TAX / NEGATIVE_INTERIOR) with NO K1 identification
    conjunct, although sec6's table conditions every one of them on
    "donor identified".  Synthetic proof below.
D2  `side_outputs["residency_fraction_by_arm"]` is a dict comprehension keyed by
    ARM over a 28-record grid, so 3 of every 4 blocks are silently overwritten.
    add. A-2 / D-4 require the residency baseline to be REPORTED per arm.
NON-DECISION-QUANTITY: synthetic fixtures + a firewalled campaign shape check.
"""
import sys
from pathlib import Path
sys.path.insert(0, "/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/slo_sched")
import g16_analyze as G
HERE = Path(__file__).resolve().parent.parent


def rec(arm, block, ttft, itl):
    return G.BootRecord(arm=arm, decode_sm=G.arm_sm(arm), block=block, boot=1,
                        phase="HI", path="synthetic",
                        est={"M_ttft": ttft, "M_itl": itl})


# D1: TTFT argmin rotates across blocks -> D_ttft is NOT identified (K1 fails),
#     but the bare argmin of the pooled means is still d34 (interior).
print("=== D1: ladder rung verdicts without the K1 conjunct ===")
grid = []
ttfts = {"blk1": {"d16": 300, "d34": 100, "d54": 200},
         "blk2": {"d16": 100, "d34": 305, "d54": 200},
         "blk3": {"d16": 200, "d34": 100, "d54": 300},
         "blk4": {"d16": 100, "d34": 200, "d54": 300}}
itls = {"d16": 90.0, "d34": 70.0, "d54": 50.0}
for blk, row in ttfts.items():
    for arm, t in row.items():
        grid.append(rec(arm, blk, t, itls[arm] + (0.0 if blk == "blk1" else 0.0)))
d = G.decide(grid, label="synthetic unidentified-TTFT")
print("  D_ttft identified:", d["D_ttft"]["identified"],
      "| bare argmin arm:", d["D_ttft_arm"], "| delta_citable:", d["delta_citable"])
print("  overall verdict:", d["verdict"])
rungs = [r for r in d["delta_slo_ladder"] if r["verdict"] in
         ("TAX_POSITIVE", "NO_TAX", "NEGATIVE_INTERIOR")]
print(f"  ladder rungs carrying a sec6 'donor identified' verdict anyway: {len(rungs)}"
      f"  e.g. {rungs[0] if rungs else None}")
print("  operating point (60 ms):", {k: d["operating_point"][k]
                                     for k in ("S_itl", "delta_slo", "verdict")})
print("  -> the rung/operating-point verdicts carry NO citation gate of their own;")
print("     the reader must conjoin D_ttft.identified by hand.")

# D2: residency reporting collapse
print()
print("=== D2: residency_fraction_by_arm keeps only ONE block per arm ===")
CONST = {k: 1.0 for k in (
    "M_ttft", "M_itl", "requests", "duration_s", "throughput_req_s", "goodput_req_s",
    "ttft_pass_pct", "itl_p95_pass_pct", "joint_pass_pct", "band_mass_ttft",
    "band_mass_itl", "empty_itl_requests", "ttft_p50_ms", "ttft_p95_ms",
    "ttft_p99_ms", "token_itl_p50_ms", "token_itl_p95_ms", "token_itl_p99_ms")}
G._headline_estimands = lambda path: dict(CONST)
cgrid, _ = G.load_campaign_grid(HERE, "HI")
collapsed = {r.arm: r.residency_fraction for r in cgrid if r.residency_fraction}
print(f"  boots with residency in grid: {sum(1 for r in cgrid if r.residency_fraction)}")
print(f"  entries emitted by side_outputs['residency_fraction_by_arm']: {len(collapsed)}"
      f"  (one per arm -- {sum(1 for r in cgrid if r.residency_fraction) - len(collapsed)} boots dropped silently)")
print(f"  split_transitions_by_boot entries (keyed by filename, correct): "
      f"{len({Path(r.path).name for r in cgrid})}")
