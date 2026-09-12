#!/usr/bin/env python3
"""X1 cross-job comparator -- DIAGNOSTIC ONLY, registered before X1 has run.

rev2 (2026-09-12) applies the rules-layer audit conditions D2, D8, D11.

The primary X1 verdict stays with r2_correctness_check.py (rule v2, UNMODIFIED,
run inside the job).  This script answers a different, explicitly NON-verdict
question: does the greedy token comparison notice a reduction-order perturbation
at all?  Cross-job comparison is diagnostic because the two jobs differ in node,
clock and triton cache as well as in the perturbation (VERDICT §7.3).

What rev2 added, and why
  D2(i)  the denominator is printed and checked: the registered F2 sentence is
         only readable at 96 unit-pairs with all four boots present.  rev1
         printed the same interpretation sentence at compared=32.
  D2(ii) a mismatch at output_ids[0] is NOT attributable to the perturbation:
         index 0 comes out of prefill (forward_extend never reads
         num_kv_splits), which this perturbation leaves bit-identical.  Those
         are counted and reported separately.
  D2(iii) S and O are reported separately -- S is bs=1 throughout and has a
         cross-job null control (907032 -> 907100); O has neither.
  D8     the C-tier block prints NOT-RUN unless all four boots are present.
         rev1 printed "units differing = 0" for job 907032, which has only two
         boots -- a false zero (its real L1-L2 C mismatch is 7/32).
  D11    inputs are verified by prompt_sha256 AND prompt_tokens.

Usage:
  x1_cross_job_compare.py <baseline_job_dir> <x1_job_dir>
  x1_cross_job_compare.py --selftest
"""
import json
import sys
from pathlib import Path

LABELS = ["L1", "TD1", "L2", "TD2"]
EXPECTED_PAIRS = 96  # 4 boot labels x (16 S + 8 O)


def units(gen):
    """id -> (tier, prompt_sha256, prompt_tokens, output_ids)."""
    out = {}
    for r in gen.get("phase_s", []):
        out[r["id"]] = ("S", r.get("prompt_sha256"), r.get("prompt_tokens"),
                        r.get("output_ids"))
    for rec in gen.get("phase_o", []):
        p = rec.get("probe", {})
        out[rec["id"]] = ("O", p.get("prompt_sha256"), p.get("prompt_tokens"),
                          p.get("output_ids"))
    return out


def c_classes(gen_by_label):
    ids, per = None, {}
    for lab, gen in gen_by_label.items():
        per[lab] = {r["id"]: tuple(r.get("output_ids") or ())
                    for r in gen.get("phase_c", [])}
        ids = set(per[lab]) if ids is None else (ids & set(per[lab]))
    rows = {}
    for uid in sorted(ids or ()):
        groups = {}
        for lab in gen_by_label:
            groups.setdefault(per[lab][uid], []).append(lab)
        rows[uid] = sorted((sorted(v) for v in groups.values()), key=lambda g: g[0])
    return rows


def first_div(a, b):
    for i, (x, y) in enumerate(zip(a or (), b or ())):
        if x != y:
            return i
    if (a or []) != (b or []):
        return min(len(a or []), len(b or []))
    return None


def compare(base_dir, x1_dir):
    base_dir, x1_dir = Path(base_dir), Path(x1_dir)
    print(f"baseline = {base_dir}")
    print(f"X1       = {x1_dir}")
    tot = {"S": [0, 0], "O": [0, 0]}          # tier -> [compared, mismatching]
    idx0 = []
    bad_all, missing = [], []
    for lab in LABELS:
        fb, fx = base_dir / f"gen_{lab}.json", x1_dir / f"gen_{lab}.json"
        if not (fb.exists() and fx.exists()):
            missing.append(lab)
            print(f"{lab}: MISSING gen file (baseline={fb.exists()} x1={fx.exists()})"
                  f" -- excluded")
            continue
        ub, ux = units(json.load(open(fb))), units(json.load(open(fx)))
        per = {"S": [0, 0], "O": [0, 0]}
        mism, bad = [], []
        for uid in sorted(set(ub) & set(ux)):
            tier, sha_b, ptok_b, ids_b = ub[uid]
            _, sha_x, ptok_x, ids_x = ux[uid]
            if sha_b is None or sha_b != sha_x or ptok_b != ptok_x:
                bad.append(uid)
                continue
            per[tier][0] += 1
            d = first_div(ids_b, ids_x)
            if d is not None:
                per[tier][1] += 1
                mism.append((uid, d))
                if d == 0:
                    idx0.append((lab, uid))
        for t in ("S", "O"):
            tot[t][0] += per[t][0]
            tot[t][1] += per[t][1]
        bad_all += [(lab, u) for u in bad]
        print(f"{lab}: S compared={per['S'][0]} mismatch={per['S'][1]} | "
              f"O compared={per['O'][0]} mismatch={per['O'][1]}"
              f"{'  INPUT_MISMATCH=' + str(bad) if bad else ''}"
              f"{'  [' + ', '.join(f'{u}@{d}' for u, d in mism[:8]) + ']' if mism else ''}")
    compared = tot["S"][0] + tot["O"][0]
    mismatching = tot["S"][1] + tot["O"][1]
    print()
    print(f"TOTAL compared unit-pairs = {compared} (expected {EXPECTED_PAIRS})"
          f"  mismatching = {mismatching}"
          f"  [S {tot['S'][1]}/{tot['S'][0]}, O {tot['O'][1]}/{tot['O'][0]}]"
          f"  excluded(input) = {len(bad_all)}")
    if compared != EXPECTED_PAIRS or missing:
        print("!! DENOMINATOR NOT INTACT -- the registered F2 sentence does not "
              "apply to this run (pre-registration D2(i)).  Restate the "
              "denominator and the exclusions explicitly instead.")
        if missing:
            print(f"   missing boot labels: {missing}")
        if bad_all:
            print(f"   input-excluded units: {bad_all}")
    else:
        print("reading (F2): mismatch>=1 with first_divergence>=1 -> the S/O token "
              "comparison is sensitive to a decode-attention reduction-order "
              "perturbation, on the units where it landed.  mismatch==0 -> "
              "sensitivity not demonstrated; a PASS certifies absence of "
              "argmax-flipping corruption only (VERDICT §1.6).  Neither outcome "
              "is a gate result and neither licenses a Claim D statement.")
    if idx0:
        print(f"!! {len(idx0)} mismatch(es) at output_ids[0]: {idx0}")
        print("   index 0 is a prefill (extend) output and this perturbation does "
              "not touch forward_extend -- NOT attributable to the perturbation "
              "(pre-registration D2(ii)); investigate separately.")
    for tag, d in (("baseline", base_dir), ("X1", x1_dir)):
        gens = {}
        for lab in LABELS:
            f = d / f"gen_{lab}.json"
            if f.exists():
                gens[lab] = json.load(open(f))
        if len(gens) != 4:
            print(f"\nC tier ({tag}, diagnostic only): NOT-RUN -- needs all four "
                  f"boots, found {sorted(gens)} (D8: a partial job would print a "
                  f"false zero here)")
            continue
        rows = c_classes(gens)
        split = {u: g for u, g in rows.items() if len(g) > 1}
        arm = {u: g for u, g in split.items()
               if sorted(map(sorted, g)) == [["L1", "L2"], ["TD1", "TD2"]]}
        print(f"\nC tier ({tag}, diagnostic only): units differing = {len(split)}"
              f" {sorted(split)}")
        print(f"  arm-separated ({{L1,L2}} vs {{TD1,TD2}}) = {len(arm)} {sorted(arm)}")
        print("  C licenses no gate, mechanism or policy claim (R2C-3); a drop in "
              "the total is an EXPECTED side effect of making decode attention "
              "batch-invariant per row.")
    return mismatching


def selftest():
    """A comparator that cannot see a one-token change is useless.  Each case
    below must behave; see D2/D8/D11 for why the later ones exist."""
    import copy
    import tempfile

    def mk(sid_tok=6, oid_tok=1437):
        return {"bg_max_new": 160,
                "phase_s": [{"id": "S00", "prompt_sha256": "a",
                             "prompt_tokens": sid_tok, "output_ids": [1, 2, 3]},
                            {"id": "S01", "prompt_sha256": "c",
                             "prompt_tokens": 25, "output_ids": [7, 7, 7]}],
                "phase_o": [{"id": "O00", "bg": {"prompt_tokens": 17},
                             "probe": {"prompt_sha256": "b", "prompt_tokens": oid_tok,
                                       "output_ids": [4, 5, 6]}}],
                "phase_c": [{"id": "C00", "output_ids": [7, 8]}]}

    with tempfile.TemporaryDirectory() as td:
        d1, d2 = Path(td) / "a", Path(td) / "b"
        for d in (d1, d2):
            d.mkdir()
            for lab in LABELS:
                json.dump(mk(), open(d / f"gen_{lab}.json", "w"))
        assert compare(d1, d2) == 0, "identical jobs must report 0 mismatches"

        g = mk(); g["phase_o"][0]["probe"]["output_ids"] = [4, 9, 6]
        json.dump(g, open(d2 / "gen_TD1.json", "w"))
        assert compare(d1, d2) == 1, "one flipped probe token must be detected"

        g = mk(); g["phase_s"][0]["output_ids"] = [9, 2, 3]
        json.dump(g, open(d2 / "gen_TD1.json", "w"))
        assert compare(d1, d2) == 1, "a flipped S token must be detected too"

        g = mk(); g["phase_s"][0]["prompt_sha256"] = "zzz"
        json.dump(g, open(d2 / "gen_TD1.json", "w"))
        assert compare(d1, d2) == 0, "input sha mismatch must be excluded, not counted"

        g = mk(sid_tok=7)          # D11: same sha, different token count
        json.dump(g, open(d2 / "gen_TD1.json", "w"))
        assert compare(d1, d2) == 0, "prompt_tokens mismatch must be excluded"

        json.dump(mk(), open(d2 / "gen_TD1.json", "w"))
        (d2 / "gen_TD2.json").unlink()     # D8: partial job
        assert compare(d1, d2) == 0
    print("\nSELFTEST OK (identical=0; S and O flips detected; sha and "
          "prompt_tokens mismatches excluded; partial job handled)")


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--selftest":
        selftest()
    elif len(sys.argv) == 3:
        compare(sys.argv[1], sys.argv[2])
    else:
        print(__doc__)
        sys.exit(2)
