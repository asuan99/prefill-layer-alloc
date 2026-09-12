#!/usr/bin/env python3
"""Order-swap (OS) shape analysis -- registered before the OS job runs.

The question
------------
In job 907100 the boot order was `L TD L TD`, so legacy held positions 1,3 and
true-dual held 2,4.  "Split along the arm boundary" and "split along the
position parity {1,3}|{2,4}" are therefore THE SAME PARTITION -- perfectly
aliased.  The 3/32 C-tier units that looked arm-separated (C01, C10, C19) are
equally consistent with an order/drift effect.

Running `R2C_ORDER="L TD TD L"` de-aliases them inside ONE job:

    position :   1     2     3     4
    label    :   L1    TD1   TD2   L2
    arm boundary      = {L1,L2}  vs {TD1,TD2} = positions {1,4}|{2,3}
    position parity   = {L1,TD2} vs {TD1,L2}  = positions {1,3}|{2,4}
    adjacency         = {L1,TD1} vs {TD2,L2}  = positions {1,2}|{3,4}

All three are distinct 2-2 shapes, so each differing unit votes for at most one.

Hardening carried from the B3 rules audit (2026-09-12, NO-GO on that design)
  D3  admissibility is checked at RECORD level (S=16, O=8, C=32 in every boot),
      not by counting gen files: an empty `phase_c` printed a false zero before.
  D5  the selftest asserts the SHAPE-NAME WIRING itself.  A mutant that wires
      "arm" to the wrong partition passed the old selftest while printing 0
      instead of 3 on job 907100 -- the checked-out thing was the conclusion.
  G-B3-3  every threshold is reported with its false-alarm rate under the
      uniform 1/3 null that this project already uses.

This script emits no verdict.  The C tier licenses no gate, mechanism or policy
claim (R2C-3); it can only say which shape the disagreements fall on.

usage:  os_shape_analysis.py <job_dir> [<job_dir> ...]   |   --selftest
"""
import itertools
import json
import sys
from pathlib import Path

EXPECTED = {"S": 16, "O": 8, "C": 32}
FOCUS = ("C01", "C10", "C19")          # pre-named in 907100, fixed before data


def tier_units(gen_one, tier):
    if tier == "S":
        return {r["id"]: tuple(r.get("output_ids") or ()) for r in gen_one.get("phase_s", [])}
    if tier == "O":
        return {r["id"]: tuple((r.get("probe") or {}).get("output_ids") or ())
                for r in gen_one.get("phase_o", [])}
    return {r["id"]: tuple(r.get("output_ids") or ()) for r in gen_one.get("phase_c", [])}


def load(job):
    job = Path(job)
    labels = (job / "boots.txt").read_text().split()
    gen = {}
    for lb in labels:
        f = job / f"gen_{lb}.json"
        if f.exists():
            gen[lb] = json.load(open(f))
    return labels, gen


def admissible(labels, gen):
    """Record-level (D3).  Returns (ok, reasons)."""
    reasons = []
    present = [lb for lb in labels if lb in gen]
    if len(present) != 4:
        reasons.append(f"boots present={present} (need 4)")
    for lb in present:
        for tier, want in EXPECTED.items():
            got = tier_units(gen[lb], tier)
            if len(got) != want:
                reasons.append(f"{lb} {tier}: {len(got)} records (need {want})")
            empty = [u for u, v in got.items() if not v]
            if empty:
                reasons.append(f"{lb} {tier}: {len(empty)} empty outputs {empty[:4]}")
    return (not reasons), reasons


def partitions(order, gen, tier):
    per = {lb: tier_units(gen[lb], tier) for lb in order}
    ids = set.intersection(*(set(v) for v in per.values()))
    out = {}
    for uid in sorted(ids):
        groups = {}
        for lb in order:
            groups.setdefault(per[lb][uid], []).append(lb)
        out[uid] = sorted((sorted(v) for v in groups.values()), key=lambda g: g[0])
    return out


def named_shapes(order):
    """ALL THREE 2-2 shapes, named by position, plus which one is the arm split.

    Naming every shape by position (never by meaning) is deliberate: with only
    "arm", "parity" and "adjacency" as names, job 907100's C24 -- a 2-2 unit on
    {1,4}|{2,3} -- matched no name and silently vanished from both buckets,
    while C01/C10/C19 were counted twice because arm and parity are the SAME
    partition in that order.  Position names are exhaustive and disjoint; the
    arm boundary is then reported as a pointer to one of them.
    """
    p = {i + 1: lb for i, lb in enumerate(order)}

    def sh(a, b, c, d):
        return frozenset({frozenset({p[a], p[b]}), frozenset({p[c], p[d]})})

    shapes = {"{1,3}|{2,4}": sh(1, 3, 2, 4),
              "{1,2}|{3,4}": sh(1, 2, 3, 4),
              "{1,4}|{2,3}": sh(1, 4, 2, 3)}
    arm_l = sorted(lb for lb in order if not lb.startswith("TD"))
    arm_t = sorted(lb for lb in order if lb.startswith("TD"))
    arm = (frozenset({frozenset(arm_l), frozenset(arm_t)})
           if len(arm_l) == 2 and len(arm_t) == 2 else None)
    arm_name = next((n for n, v in shapes.items() if v == arm), None)
    return shapes, arm_name


def shape_of(groups):
    if len(groups) != 2 or sorted(len(g) for g in groups) != [2, 2]:
        return None
    return frozenset(frozenset(g) for g in groups)


def report(job):
    labels, gen = load(job)
    order = [lb for lb in labels if lb in gen]
    print(f"=== {Path(job).name}  boots.txt={labels}  order={order} ===")
    ok, reasons = admissible(labels, gen)
    if not ok:
        print("  INADMISSIBLE -- no numbers are printed (D3):")
        for r in reasons:
            print(f"    - {r}")
        return
    names, arm_name = named_shapes(order)
    print(f"  shape map: " + " | ".join(f"{n}={sorted(tuple(sorted(g)) for g in s)}"
                                        for n, s in names.items()))
    print(f"  arm boundary (L|TD) = {arm_name}"
          + ("  ★ALIASED with the position parity {1,3}|{2,4} -- this boot order "
             "cannot tell 'arm' from 'order/drift'"
             if arm_name == "{1,3}|{2,4}" else
             "  (de-aliased from the position parity {1,3}|{2,4})"))
    for tier in ("S", "O", "C"):
        parts = partitions(order, gen, tier)
        differing = {u: g for u, g in parts.items() if len(g) > 1}
        excess = sum(len(g) - 1 for g in parts.values())
        pair = {f"{a}-{b}": sum(1 for u, g in parts.items()
                                if not any(a in grp and b in grp for grp in g))
                for a, b in itertools.combinations(order, 2)}
        print(f"  [{tier}] units={len(parts)} differing={len(differing)} "
              f"sum(classes-1)={excess}")
        print(f"       pairwise={pair}")
        if tier != "C":
            continue
        hit = {n: [] for n in names}
        other, unnamed = [], []
        for u, g in parts.items():
            s = shape_of(g)
            if s is None:
                if len(g) > 1:
                    other.append((u, [len(x) for x in g]))
                continue
            match = [n for n, sh in names.items() if s == sh]
            if not match:                      # must never happen: names are exhaustive
                unnamed.append(u)
            for n in match:
                hit[n].append(u)
        n22 = len({u for v in hit.values() for u in v})     # UNIQUE units, not sum
        assert not unnamed, f"2-2 unit(s) matched no position shape: {unnamed}"
        print(f"       2-2 units n_22={n22} (arm boundary = {arm_name}): " + ", ".join(
            f"{n}={len(v)} {sorted(v)}" + ("  <-ARM" if n == arm_name else "")
            for n, v in hit.items()))
        print(f"       non-2-2 differing units={len(other)} {other}")
        print(f"       focus units (pre-named from 907100):")
        for u in FOCUS:
            if u not in parts:
                print(f"         {u}: ABSENT")
                continue
            g = parts[u]
            s = shape_of(g)
            lab = next((n for n, sh in names.items() if s == sh), None)
            tag = lab or ("non-2-2" if len(g) > 1 else "identical")
            if lab and lab == arm_name:
                tag += " = ARM"
            print(f"         {u}: classes={g} shape={tag}")


def selftest():
    import tempfile
    order = ["L1", "TD1", "TD2", "L2"]                 # the OS order
    names, arm_name = named_shapes(order)
    # D5(ii): the wiring itself is the thing under test.
    assert names["{1,3}|{2,4}"] == frozenset({frozenset({"L1", "TD2"}),
                                              frozenset({"TD1", "L2"})})
    assert names["{1,2}|{3,4}"] == frozenset({frozenset({"L1", "TD1"}),
                                              frozenset({"TD2", "L2"})})
    assert names["{1,4}|{2,3}"] == frozenset({frozenset({"L1", "L2"}),
                                              frozenset({"TD1", "TD2"})})
    assert len(set(names.values())) == 3, "the three position shapes are distinct"
    assert arm_name == "{1,4}|{2,3}", "in L TD TD L the arm split is the diagonal"
    _, old_arm = named_shapes(["L1", "TD1", "L2", "TD2"])   # 907100 order
    assert old_arm == "{1,3}|{2,4}", "907100 order aliases arm with the parity"

    def mk(cvals):
        return {"phase_s": [{"id": f"S{i:02d}", "output_ids": [i]} for i in range(16)],
                "phase_o": [{"id": f"O{i:02d}", "probe": {"output_ids": [i]}}
                            for i in range(8)],
                "phase_c": [{"id": f"C{i:02d}", "output_ids": cvals.get(f"C{i:02d}", [0])}
                            for i in range(32)]}

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "boots.txt").write_text(" ".join(order))
        # C01 splits on the ARM shape; C10 on parity; C19 into three classes.
        per = {"L1": {"C01": [1], "C10": [1], "C19": [1]},
               "TD1": {"C01": [2], "C10": [2], "C19": [2]},
               "TD2": {"C01": [2], "C10": [1], "C19": [3]},
               "L2": {"C01": [1], "C10": [2], "C19": [3]}}
        for lb in order:
            json.dump(mk(per[lb]), open(d / f"gen_{lb}.json", "w"))
        gen = {lb: json.load(open(d / f"gen_{lb}.json")) for lb in order}
        parts = partitions(order, gen, "C")
        assert shape_of(parts["C01"]) == names[arm_name]
        assert shape_of(parts["C10"]) == names["{1,3}|{2,4}"]
        assert shape_of(parts["C19"]) is None and len(parts["C19"]) == 3
        # D5(i): a 3-class unit makes "units differing" != sum(classes-1).
        assert len([u for u, g in parts.items() if len(g) > 1]) == 3
        assert sum(len(g) - 1 for g in parts.values()) == 4, "E2 counts excess classes"
        report(d)
        # D3: a single missing C record must block ALL numbers.
        g = json.load(open(d / "gen_TD2.json"))
        g["phase_c"] = g["phase_c"][:-1]
        json.dump(g, open(d / "gen_TD2.json", "w"))
        assert not admissible(order, {lb: json.load(open(d / f"gen_{lb}.json"))
                                      for lb in order})[0]
        report(d)
        # D3: an empty output must also block (the B3 false-zero path).
        g["phase_c"] = json.load(open(d / "gen_L1.json"))["phase_c"]
        g["phase_c"][0]["output_ids"] = []
        json.dump(g, open(d / "gen_TD2.json", "w"))
        assert not admissible(order, {lb: json.load(open(d / f"gen_{lb}.json"))
                                      for lb in order})[0]
    print("\nSELFTEST OK (shape-name wiring asserted; OS de-aliases, 907100 does "
          "not; 3-class unit separates E2 from unit count; missing record and "
          "empty output both -> INADMISSIBLE)")


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--selftest":
        selftest()
    elif len(sys.argv) >= 2:
        for j in sys.argv[1:]:
            report(j)
    else:
        print(__doc__)
        sys.exit(2)
