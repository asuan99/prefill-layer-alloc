#!/usr/bin/env python3
"""Audit (i)/(ii): does the campaign loader contract hold on the real 4 blocks?

DELIBERATE CONTAMINATION FIREWALL: `_headline_estimands` is monkey-patched to a
constant so that NO decision quantity (M_ttft / M_itl / donor / Delta / S_itl)
is ever computed here.  What is exercised is the FILTERING path only: glob,
filename regex, sidecar presence, add. B-5(a)2 adoption rule, H3'-a exact
deactivation, H17 smoke exclusion, controller_summary attachment, and F3 grid
hygiene.

Part 2 checks the canonical loader can parse every artifact and reports ONLY
structural counts (requests per file, rounds per file, duration > 0, number of
requests with an EMPTY token_itl_ms list).  The empty-ITL count decides whether
`boot_estimands`' `inf` branch -- and therefore the difference between literal
K11 (U = SM>=44) and the implemented `finite`-filtered U -- is live on this
data.  Counts are aggregated over ALL boots; no per-arm value is printed.
"""
import sys, statistics
from pathlib import Path
sys.path.insert(0, "/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/slo_sched")
import g16_analyze as G

HERE = Path(__file__).resolve().parent.parent
G._headline_estimands = lambda path: {"M_ttft": 1.0, "M_itl": 1.0}   # firewall

print("=== part 1: filtering contract (estimands firewalled) ===")
for phase in ("LO", "HI"):
    grid, prov = G.load_campaign_grid(HERE, phase)
    hyg = G._grid_hygiene(grid)
    print(f"  phase {phase}: adopted={prov['n_adopted']} "
          f"arms_deactivated_by_exact={prov['arms_deactivated_by_exact']} "
          f"boots_without_controller_summary={len(prov['boots_without_controller_summary'])}")
    print(f"     blocks={hyg['blocks']} arms={hyg['arms']} balanced={hyg['balanced']} "
          f"sufficient_blocks={hyg['sufficient_blocks']} flags={hyg['flags']}")
    print(f"     boots_per_arm={hyg['boots_per_arm']}")
    why = {}
    for r in prov["rejected"]:
        why[r["why"]] = why.get(r["why"], 0) + 1
    print(f"     rejected: {why}")
    print(f"     t_boot0 present on all adopted: "
          f"{all(r.t_boot0 is not None for r in grid)}")
    print(f"     residency present on all adopted: "
          f"{all(r.residency_fraction for r in grid)}")
    # add. D-4 mandated provenance fields
    missing = [f for f in ("node", "gpu_uuid", "git_commit", "manifest_sha")
               if f not in G.BootRecord.__dataclass_fields__]
    print(f"     D-4 provenance fields absent from BootRecord: {missing}")

print()
print("=== part 2: canonical loader on every artifact (structural counts only) ===")
import pdmux_eval.analyze as pdmux
nreq, nrounds, durs, empty = [], [], [], 0
files = sorted(list(HERE.glob("g16_blk*_LO.jsonl")) + list(HERE.glob("g16_blk*_HI.jsonl")))
for p in files:
    requests, duration = pdmux.load_bench_serving_rounds(p)
    nreq.append(len(requests)); durs.append(duration)
    nrounds.append(sum(1 for line in p.open() if line.strip()))
    empty += sum(1 for r in requests if not r.token_itl_ms)
print(f"  files={len(files)}  requests/file: min={min(nreq)} max={max(nreq)}  "
      f"(K6 NP*ROUNDS = {G.K6_NP * G.K6_ROUNDS})")
print(f"  rounds/file: min={min(nrounds)} max={max(nrounds)} (K6 ROUNDS={G.K6_ROUNDS})")
print(f"  summed duration>0 on every file: {all(d > 0 for d in durs)}  "
      f"[min={min(durs):.1f}s max={max(durs):.1f}s over ALL boots+phases]")
print(f"  requests with EMPTY token_itl_ms, summed over ALL {len(files)} files: {empty}")
print("  -> empty==0 means boot_estimands' `inf` branch is DEAD on this data, so")
print("     the implemented `finite`-filter cannot make the K11 set U differ from")
print("     the literal 'decode SM >= 44 AND exact-surviving' definition.")
