#!/usr/bin/env python3
"""C-tier null control analysis -- registered before the B3 jobs run.

The question this answers
-------------------------
job 907100 showed 3 of 32 concurrent-load (C tier) requests splitting exactly
along the arm boundary ({L1,L2} vs {TD1,TD2}), and job 907456 showed 0.  Neither
number can be read, because the C tier has no denominator: we have never
measured how often FOUR BOOTS OF THE SAME ARM split into that shape by chance.
Two same-arm jobs (`R2C_ORDER="L L L L"` and `"TD TD TD TD"`) give that
denominator directly: relabel the four boots by POSITION, so positions {1,3}
play the role legacy held in `L TD L TD` and positions {2,4} the role true-dual
held, then count how often the 2-2 partition lands on that boundary with no arm
difference present at all.

Estimands (fixed before any B3 data exists)
  E1  units differing        -- how many of the 32 C units are not one class
  E2  sum(classes - 1)       -- total disagreement, which unit COUNTS hide
                                (907100 and 907456 both = 10 with 8 vs 6 units)
  E3  pairwise mismatches    -- all 6 boot pairs
  E4  position-partition hits-- units whose classes are exactly {1,3} vs {2,4},
                                the same-arm analogue of "arm-separated"
  E5  all three 2-2 shapes   -- {1,3}/{2,4}, {1,2}/{3,4}, {1,4}/{2,3}; E4 is
                                only meaningful against its two siblings
  S/O tiers                  -- mismatches among all same-arm pairs (expected 0
                                from 907100/907456; a non-zero here would mean
                                the reference itself is not reproducible)

This script never emits a verdict.  The C tier licenses no gate, mechanism or
policy claim (R2C-3), and a null control cannot change that -- it only tells us
whether the numbers we already reported were distinguishable from noise.

usage:
  c_tier_null_control.py <job_dir> [<job_dir> ...]
  c_tier_null_control.py --selftest
"""
import itertools
import json
import sys
from pathlib import Path


def load(job):
    job = Path(job)
    labels = Path(job / "boots.txt").read_text().split()
    gen = {}
    for lb in labels:
        f = job / f"gen_{lb}.json"
        if f.exists():
            gen[lb] = json.load(open(f))
    return labels, gen


def tier_units(gen_one, tier):
    if tier == "C":
        return {r["id"]: tuple(r.get("output_ids") or ()) for r in gen_one.get("phase_c", [])}
    if tier == "S":
        return {r["id"]: tuple(r.get("output_ids") or ()) for r in gen_one.get("phase_s", [])}
    return {r["id"]: tuple((r.get("probe") or {}).get("output_ids") or ())
            for r in gen_one.get("phase_o", [])}


def partitions(labels, gen, tier):
    """unit -> list of boot-label groups that agree on the output."""
    per = {lb: tier_units(gen[lb], tier) for lb in labels if lb in gen}
    if not per:
        return {}
    ids = set.intersection(*(set(v) for v in per.values()))
    out = {}
    for uid in sorted(ids):
        groups = {}
        for lb in per:
            groups.setdefault(per[lb][uid], []).append(lb)
        out[uid] = sorted((sorted(v) for v in groups.values()), key=lambda g: g[0])
    return out


def shape_of(groups):
    """Canonical 2-2 shape as a frozenset of frozensets, or None."""
    if len(groups) != 2 or sorted(len(g) for g in groups) != [2, 2]:
        return None
    return frozenset(frozenset(g) for g in groups)


def report(job):
    labels, gen = load(job)
    print(f"=== {Path(job).name}  boots={labels}  present={sorted(gen)} ===")
    if len(gen) < 4:
        print(f"  NOT-RUN: needs four boots, found {len(gen)} "
              f"(a partial job prints a false zero otherwise)")
        return
    order = [lb for lb in labels if lb in gen]          # submission order
    pos = {lb: i + 1 for i, lb in enumerate(order)}
    for tier in ("S", "O", "C"):
        parts = partitions(order, gen, tier)
        differing = {u: g for u, g in parts.items() if len(g) > 1}
        excess = sum(len(g) - 1 for g in parts.values())
        pair = {f"{a}-{b}": sum(1 for u, g in parts.items()
                                if not any(a in grp and b in grp for grp in g))
                for a, b in itertools.combinations(order, 2)}
        print(f"  [{tier}] units={len(parts)} differing(E1)={len(differing)} "
              f"sum(classes-1)(E2)={excess}")
        print(f"       pairwise(E3)={pair}")
        if tier != "C":
            continue
        shapes = {}
        for u, g in parts.items():
            s = shape_of(g)
            if s is not None:
                shapes.setdefault(s, []).append(u)
        named = {
            "{1,3}|{2,4}": frozenset({frozenset({order[0], order[2]}),
                                      frozenset({order[1], order[3]})}),
            "{1,2}|{3,4}": frozenset({frozenset({order[0], order[1]}),
                                      frozenset({order[2], order[3]})}),
            "{1,4}|{2,3}": frozenset({frozenset({order[0], order[3]}),
                                      frozenset({order[1], order[2]})}),
        }
        print(f"       2-2 shapes(E5): " + ", ".join(
            f"{name}={len(shapes.get(sh, []))} {sorted(shapes.get(sh, []))}"
            for name, sh in named.items()))
        print(f"       position-partition (E4, the same-arm analogue of "
              f"'arm-separated') = {len(shapes.get(named['{1,3}|{2,4}'], []))}")
        print(f"       boot positions: " + ", ".join(f"{lb}={pos[lb]}" for lb in order))


def selftest():
    import tempfile
    labels = ["L1", "L2", "L3", "L4"]
    # C00 agrees; C01 splits on positions {1,3} vs {2,4}; C02 splits 3-1.
    outs = {"L1": {"C00": [1], "C01": [7], "C02": [5]},
            "L2": {"C00": [1], "C01": [9], "C02": [5]},
            "L3": {"C00": [1], "C01": [7], "C02": [5]},
            "L4": {"C00": [1], "C01": [9], "C02": [6]}}
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "boots.txt").write_text(" ".join(labels))
        for lb in labels:
            json.dump({"phase_s": [], "phase_o": [],
                       "phase_c": [{"id": u, "output_ids": v}
                                   for u, v in outs[lb].items()]},
                      open(d / f"gen_{lb}.json", "w"))
        parts = partitions(labels, {lb: json.load(open(d / f"gen_{lb}.json"))
                                    for lb in labels}, "C")
        assert len(parts) == 3
        assert sum(len(g) - 1 for g in parts.values()) == 2, "E2 must count 1+1"
        assert shape_of(parts["C01"]) == frozenset(
            {frozenset({"L1", "L3"}), frozenset({"L2", "L4"})}), "E4 shape"
        assert shape_of(parts["C02"]) is None, "3-1 split is not a 2-2 shape"
        report(d)
        (d / "gen_L4.json").unlink()
        report(d)          # must print NOT-RUN, never a false zero
    print("\nSELFTEST OK (E2 counts excess classes, E4 detects the position "
          "partition, 3-1 splits are not 2-2, partial job -> NOT-RUN)")


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--selftest":
        selftest()
    elif len(sys.argv) >= 2:
        for j in sys.argv[1:]:
            report(j)
    else:
        print(__doc__)
        sys.exit(2)
