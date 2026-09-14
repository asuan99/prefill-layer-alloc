#!/usr/bin/env python3
"""Render `plan.json` as the cell spec lines `lambda0.sbatch` loops over.

Kept as a FILE rather than an inline heredoc so that (a) the sbatch stays
readable, (b) a unit test can assert the spec it produces matches the plan, and
(c) the shape/kappa/low-side columns cannot drift apart from `lambda0_plan.py`
without something failing.

One line per cell, whitespace separated, in the order the sbatch reads:

    name shape input_len output_len offered_rate num_prompts seed
         kappa_pred drain_pred_s low_side_candidate t_measure_s

`kappa_pred` and `drain_pred_s` are the literal placeholder `-` for rungs the
drain model calls saturated (no ceiling applies to a high-side rung); the sbatch
turns `-` back into an empty string and then omits the analyzer flag.

★The placeholder is NOT cosmetic.  bash's `read` with the default IFS collapses
runs of whitespace, so an EMPTY column does not occupy a field: every column to
its right would shift left and a saturated cell would be analysed with
`low_side_candidate` parsed out of the kappa slot.  The selftest pins the field
count for every row, saturated ones included.

usage: lambda0_cells.py <plan.json>   |   --selftest
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _row(c, shape, seed):
    return " ".join(str(x) for x in (
        c["name"], shape, c["input_len"], c["output_len"], c["offered_nominal"],
        c["num_prompts"], seed,
        "-" if c["kappa_pred"] is None else c["kappa_pred"],
        "-" if c["drain_pred_s"] is None else c["drain_pred_s"],
        int(bool(c["low_side_candidate"])), c["T_measure_s"]))


def rows(plan):
    out = []
    for s in ("A", "B"):
        d = plan["shapes"][s]
        for c in d["cells"]:
            out.append(_row(c, s, plan["seed1"]))
        rep = dict(d["cells"][-1], name=d["seed_repeat_cell"])
        out.append(_row(rep, s, plan["seed2"]))
    return out


def selftest():
    sys.path.insert(0, str(HERE))
    import lambda0_plan
    p = lambda0_plan.plan(2.10, 0.675)
    rs = rows(p)
    assert len(rs) == p["n_boot"] == 11, (len(rs), p["n_boot"])
    for r in rs:
        f = r.split()
        assert len(f) == 11, ("every row needs 11 whitespace-separated fields, "
                              "because the sbatch `read -r` unpacks exactly "
                              "that many", r)
        assert f[1] in ("A", "B") and f[9] in ("0", "1"), r
    # the repeat rows carry the SECOND seed and the top rung's sizing.
    reps = [r for r in rs if r.split()[0].endswith("_s2")]
    assert len(reps) == 2, reps
    for r in reps:
        assert r.split()[6] == str(p["seed2"]), r
    # ★a saturated rung must STILL occupy 11 fields under whitespace-collapsing
    # `read`.  This is the assertion that catches the placeholder being dropped:
    # with empty strings the row splits into 9 fields and every column right of
    # kappa shifts left.
    sat = [c for c in p["shapes"]["A"]["cells"] if c["kappa_pred"] is None]
    assert sat, "the point-estimate scenario must contain a saturated rung"
    for c in sat:
        f = _row(c, "A", p["seed1"]).split()      # split() == bash read's IFS
        assert len(f) == 11, ("a saturated row lost a field -- bash `read` would "
                              "misalign low_side_candidate into the kappa slot", f)
        assert f[7] == "-" and f[8] == "-", f
        assert f[9] in ("0", "1") and float(f[10]) > 0, f
    # ★Y5: the renderer must carry the PLAN's certification, not a constant.
    # `int(bool(...)) -> 1` would certify every rung and replicate the rev3
    # kill cause into shape A, and rev3's harness passed it untouched because
    # this file was outside MODULES.
    certified = {r.split()[0] for r in rs if r.split()[9] == "1"}
    expected = {c["name"] for sh in ("A", "B")
                for c in p["shapes"][sh]["cells"] if c["low_side_candidate"]}
    expected |= {p["shapes"][sh]["seed_repeat_cell"] for sh in ("A", "B")
                 if p["shapes"][sh]["cells"][-1]["low_side_candidate"]}
    assert certified == expected, ("the rendered certification must equal the "
                                   "plan's", sorted(certified), sorted(expected))
    assert any(r.split()[9] == "0" for r in rs), (
        "at least one rung must be UNcertified, else the column is a constant")

    # ★Y6: kappa and low-side must not swap columns.  Column 7 is a ceiling in
    # (0,1] or the placeholder; column 9 is a 0/1 flag; column 10 is seconds.
    for r in rs:
        f = r.split()
        assert f[7] == "-" or 0.0 < float(f[7]) <= 1.0, ("kappa column", r)
        assert f[9] in ("0", "1"), ("low-side column", r)
        assert float(f[10]) >= 170.0, ("window column", r)

    print("CELLS SELFTEST OK (11 rows for the registered plan; repeat rows use "
          "seed2; saturated rungs emit the `-` placeholder so bash `read` keeps 11 "
          "fields; the certification column equals the plan's and is not a "
          "constant; kappa/low-side/window columns cannot swap)")


def main() -> int:
    if len(sys.argv) == 2 and sys.argv[1] == "--selftest":
        selftest()
        return 0
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    for r in rows(json.loads(Path(sys.argv[1]).read_text())):
        print(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
