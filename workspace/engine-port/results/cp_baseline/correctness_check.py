#!/usr/bin/env python3
"""Correctness gate check: do the arms compute the same function?  GPU 0.

Compares every arm's greedy output against the `plain` reference (no chunking, no
pdmux), prompt by prompt.  Three verdicts, and the middle one exists because
collapsing it into either neighbour is how a real defect would get filed as noise:

  IDENTICAL   -- byte-identical output on every prompt.  The scheduling change
                 provably did not change the computed function on this set.
  PREFIX_DIV  -- outputs share a prefix and then diverge.  This is what float
                 nondeterminism looks like (different reduction order -> a
                 near-tie argmax flips -> everything after differs).  It is NOT
                 automatically benign: a divergence at token 1 is a different
                 animal from one at token 40, so the divergence POSITION is
                 reported rather than thresholded.
  DIVERGENT   -- differs from the first token, or one side is empty/errored.

★No threshold turns these into PASS/FAIL here.  This file reports the structure;
the pre-registration decides what is acceptable.  A checker that silently picked
a threshold would be choosing the answer (this track's signature failure).
"""
import json, sys
from pathlib import Path

REF = __import__("os").environ.get("CGATE_REF", "plain")


def common_prefix_len(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def degenerate(t):
    """Non-degeneracy sanity: empty, or a single token repeated to the end."""
    s = t.strip()
    if not s:
        return "empty"
    toks = s.split()
    if len(toks) >= 8 and len(set(toks)) == 1:
        return f"single token x{len(toks)}"
    if len(set(s)) <= 2 and len(s) >= 20:
        return f"<=2 distinct chars over {len(s)}"
    return None


def main(outdir):
    out = Path(outdir)
    arms = {}
    for f in sorted(out.glob("gen_*.json")):
        d = json.loads(f.read_text())
        arms[d["arm"]] = {r["id"]: r for r in d["results"]}
    if REF not in arms:
        print(f"NO REFERENCE ARM `{REF}` -- cannot run the gate (arms present: "
              f"{sorted(arms)})")
        return 1

    ref = arms[REF]
    report = {"_what_this_is": "Correctness gate: greedy-output agreement against the "
                               "`plain` reference. Reports structure, sets no "
                               "pass/fail threshold.",
              "reference": REF, "arms": {}}

    print(f"reference arm: {REF}   prompts: {len(ref)}   other arms: "
          f"{sorted(a for a in arms if a != REF)}\n")

    # degeneracy first: an arm producing garbage everywhere would otherwise show up
    # as "identical garbage" and pass an agreement test.
    print("--- non-degeneracy (does any arm emit empty / repeated output?) ---")
    for arm in sorted(arms):
        bad = [(i, degenerate(r["text"])) for i, r in arms[arm].items()
               if r.get("error") or degenerate(r["text"])]
        errs = [i for i, r in arms[arm].items() if r.get("error")]
        print(f"  {arm:<8} degenerate/errored: {len(bad)}/{len(arms[arm])}"
              + (f"   e.g. {bad[0]}" if bad else ""))
        report["arms"].setdefault(arm, {})["degenerate"] = [b[0] for b in bad]
        report["arms"][arm]["errors"] = errs

    print("\n--- agreement with the reference (per arm) ---")
    for arm in sorted(a for a in arms if a != REF):
        rows = []
        for pid, r in ref.items():
            o = arms[arm].get(pid)
            if o is None:
                rows.append((pid, "MISSING", None, None)); continue
            if r.get("error") or o.get("error"):
                rows.append((pid, "ERROR", None, None)); continue
            a, b = r["text"], o["text"]
            if a == b:
                rows.append((pid, "IDENTICAL", len(a), None)); continue
            cp = common_prefix_len(a, b)
            rows.append((pid, "PREFIX_DIV" if cp > 0 else "DIVERGENT", len(a), cp))
        n_id = sum(1 for _, v, _, _ in rows if v == "IDENTICAL")
        print(f"\n  {arm}:  IDENTICAL {n_id}/{len(rows)}")
        for pid, v, ln, cp in rows:
            if v != "IDENTICAL":
                k = ref[pid]["kind"] if pid in ref else "?"
                print(f"    {pid:<10} {k:<16} {v:<11}"
                      + (f" common prefix {cp}/{ln} chars" if cp is not None else ""))
        report["arms"][arm]["agreement"] = [
            {"id": p, "verdict": v, "ref_len": ln, "common_prefix": cp}
            for p, v, ln, cp in rows]
        report["arms"][arm]["n_identical"] = n_id
        report["arms"][arm]["n_prompts"] = len(rows)

    (out / "correctness_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(f"\nwrote {out / 'correctness_report.json'}")

    print("\n--- reference outputs (eyeball: are these sane continuations?) ---")
    for pid, r in ref.items():
        if r.get("error"):
            print(f"  {pid:<10} ERROR {r['error'][:60]}"); continue
        print(f"  {pid:<10} {r['kind']:<16} -> {repr(r['text'][:70])}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
