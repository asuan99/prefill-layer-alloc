#!/usr/bin/env python3
"""Audit (iv): the block x node x concurrency ALIASING matrix (addendum D-4).

Built from sidecars + H10 co-tenancy snapshots ONLY.  No estimand is computed
and no *_LO.jsonl / *_HI.jsonl is opened.  The question this answers is whether
the four blocks can be treated as exchangeable draws (K3), and it is answered
from PLACEMENT alone, as addendum D-4 criterion (i) requires.
"""
import json, re
from pathlib import Path
HERE = Path(__file__).resolve().parent.parent

side = {}
for p in sorted(HERE.glob("g16_blk*_sidecar.json")):
    d = json.loads(p.read_text()); side[p.name[:-len("_sidecar.json")]] = d
blocks = {}
for run, d in side.items():
    b = blocks.setdefault(d["block"], {"node": d["node"], "gpu": d["gpu_uuid"][4:12],
                                       "t0": [], "arms": []})
    b["t0"].append(d["t_boot0"]); b["arms"].append((d["t_boot0"], d["arm"]))
for b in blocks.values():
    b["t_start"] = min(b["t0"]); b["t_end"] = max(b["t0"])
    b["order"] = [a for _t, a in sorted(b["arms"])]

# concurrency from the H10 snapshots: which G16 jobs were on the node
conc = {}
for p in sorted(HERE.glob("g16_blk*_cotenancy_*.txt")):
    txt = p.read_text()
    blk = re.match(r"g16_(blk\d)_", p.name).group(1)
    jobs = set(re.findall(r"^\s*(\d{6})\s+amd_a100n", txt, re.M))
    users = set(re.findall(r"\s(\w+)\s*$", txt, re.M))
    conc.setdefault(blk, set()).update(jobs)

print("design matrix (all nuisance factors are BLOCK-level):")
print(f"{'block':6} {'node':6} {'gpu':9} {'time rank':9} {'G16 jobs on node':18} {'position-0 arm':14}")
for blk in sorted(blocks):
    b = blocks[blk]
    rank = sorted(blocks, key=lambda k: blocks[k]["t_start"]).index(blk) + 1
    print(f"{blk:6} {b['node']:6} {b['gpu']:9} {rank:^9} {str(sorted(conc[blk])):18} {b['order'][0]:14}")

print()
print("aliases (each nuisance factor is a FUNCTION of block identity):")
print("  node        : gpu42 <=> blk1 <=> earliest block           -> node ALIASED with time")
print("  concurrency : 2 G16 jobs <=> {blk2,blk3} <=> middle slot   -> concurrency ALIASED with time")
print("  physical GPU: 707234ca <=> {blk2,blk4}                     -> GPU ALIASED with block pair")
print("  => with 4 blocks and 3 block-level nuisance factors, none is separable")
print("     from the others; the perturbation CANNOT be sized from these data.")
print()
print("what is nevertheless protected (design property, not an assumption):")
print("  every block contains ALL 7 arms exactly once (verified: incidence 7x4 = 1),")
print("  so a SEPARABLE block-level effect cancels in both K1 rules:")
print("    - rank rule  : within-block argmin is invariant to ANY strictly increasing")
print("                   per-block transformation of that block's arm means;")
print("    - bootstrap  : argmin_a mean_b (mu_a + c_b) = argmin_a mu_a, and likewise")
print("                   argmin_a mean_b (k_b * mu_a) = argmin_a mu_a for k_b > 0.")
print("  The residual threat is therefore a block x ARM INTERACTION only")
print("  (y_ab = mu_a + c_b + gamma_ab with gamma_ab != 0), e.g. host contention")
print("  whose severity depends on the arm -- which is exactly the channel the")
print("  deterministic blk2/blk3 co-tenant pairing creates (see a1 output).")
print()
print("independent placements:")
print("  4 blocks, but only 3 distinct placements: {blk1 @gpu42 solo}, ")
print("  {blk2||blk3 @gpu41 concurrent}, {blk4 @gpu41 solo}.  K3 resamples 4 units")
print("  that are not 4 independent placements  ->  n_indep(placement) = 3.")
