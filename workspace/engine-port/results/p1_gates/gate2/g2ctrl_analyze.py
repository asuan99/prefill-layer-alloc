#!/usr/bin/env python3
"""Score the G-2 negative-control campaign (PREREG_G2CTRL_2026-08-07.md).

Usage:
  python3 g2ctrl_analyze.py <summary_json> [<summary_json> ...]

Each <summary_json> is a g2ctrl_<tag>_summary_<jobid>.json produced by
g2ctrl_run.sbatch (one line per arm, JSONL). Pass the summary files for both
models (Zamba2 + Granite) to get the combined verdict; a single file gives
the per-model-only verdict.

No performance numbers anywhere in this script's output -- correctness
counts and mismatch positions only.
"""
import json
import math
import sys
from collections import defaultdict


def fisher_exact_2x2(a, b, c, d):
    """Two-sided Fisher exact test p-value for a 2x2 table
        [[a, b],
         [c, d]]
    without scipy (venv has no scipy). Exact hypergeometric enumeration --
    fine at this sample size (n <= ~200 per cell).
    """
    n = a + b + c + d
    row1 = a + b
    row2 = c + d
    col1 = a + c
    col2 = b + d

    def hyper_p(x):
        # P(A=x | fixed margins row1,row2,col1,col2), x = count in cell (0,0)
        lo = max(0, row1 - col2)
        hi = min(row1, col1)
        if x < lo or x > hi:
            return 0.0
        log_p = (
            math.lgamma(row1 + 1) - math.lgamma(x + 1) - math.lgamma(row1 - x + 1)
            + math.lgamma(row2 + 1) - math.lgamma(col1 - x + 1) - math.lgamma(row2 - col1 + x + 1)
            - (math.lgamma(n + 1) - math.lgamma(col1 + 1) - math.lgamma(col2 + 1))
        )
        return math.exp(log_p)

    lo = max(0, row1 - col2)
    hi = min(row1, col1)
    p_obs = hyper_p(a)
    total = 0.0
    eps = 1e-10
    for x in range(lo, hi + 1):
        px = hyper_p(x)
        if px <= p_obs * (1 + eps):
            total += px
    return min(1.0, total)


def load_rows(paths):
    rows = []  # (tag, arm_row_dict)
    for p in paths:
        tag = None
        # tag is embedded in filename g2ctrl_<tag>_summary_<jobid>.json
        base = p.split("/")[-1]
        if base.startswith("g2ctrl_") and "_summary_" in base:
            tag = base[len("g2ctrl_"):base.index("_summary_")]
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                rows.append((tag or p, d))
    return rows


def agg_self(rows, arm):
    n_total = 0
    n_mismatch = 0
    n_undetermined_reps = 0
    positions = []
    per_tag = defaultdict(lambda: [0, 0])  # tag -> [n_total, n_mismatch]
    for tag, d in rows:
        if d.get("arm") != arm or not d.get("self_baseline_ok"):
            continue
        for rep in d.get("reps", []):
            if rep["verdict_self"] == "UNDETERMINED":
                n_undetermined_reps += 1
                continue
            n_total += rep["n_ok"]
            n_mismatch += rep["n_mismatch_self"]
            per_tag[tag][0] += rep["n_ok"]
            per_tag[tag][1] += rep["n_mismatch_self"]
            for m in rep["self_mismatch_positions"]:
                positions.append((tag, m["first_diff_token_index"]))
    return {
        "n_total": n_total, "n_mismatch": n_mismatch,
        "n_undetermined_reps": n_undetermined_reps,
        "per_tag": dict(per_tag), "positions": positions,
    }


def agg_cross(rows, arm="chunk512"):
    n_total = 0
    n_mismatch = 0
    n_undetermined_reps = 0
    positions = []
    per_tag = defaultdict(lambda: [0, 0])
    for tag, d in rows:
        if d.get("arm") != arm or not d.get("self_baseline_ok"):
            continue
        for rep in d.get("reps", []):
            c = rep.get("cross")
            if c is None:
                continue
            if c["verdict_cross"] == "UNDETERMINED":
                n_undetermined_reps += 1
                continue
            n_total += c["n_ok"]
            n_mismatch += c["n_mismatch_cross"]
            per_tag[tag][0] += c["n_ok"]
            per_tag[tag][1] += c["n_mismatch_cross"]
            for m in rep["cross_mismatch_positions"] or []:
                positions.append((tag, m["first_diff_token_index"]))
    return {
        "n_total": n_total, "n_mismatch": n_mismatch,
        "n_undetermined_reps": n_undetermined_reps,
        "per_tag": dict(per_tag), "positions": positions,
    }


def fisher_or_note(a_mismatch, a_total, b_mismatch, b_total):
    a_ok = a_total - a_mismatch
    b_ok = b_total - b_mismatch
    if a_total == 0 or b_total == 0:
        return None, "one side has n=0 -- cannot test"
    p = fisher_exact_2x2(a_mismatch, a_ok, b_mismatch, b_ok)
    return p, None


def main():
    paths = sys.argv[1:]
    if not paths:
        print(__doc__)
        return 1
    rows = load_rows(paths)
    if not rows:
        print("NO ROWS LOADED -- check paths")
        return 1

    arms = ["plain", "plainaux", "chunk512", "agnostic"]
    self_agg = {a: agg_self(rows, a) for a in arms}
    cross_agg = agg_cross(rows, "chunk512")

    print("=" * 78)
    print("G-2 NEGATIVE CONTROL -- SELF-BASELINE MISMATCH RATES (16-concurrent vs own batch=1)")
    print("=" * 78)
    for a in arms:
        s = self_agg[a]
        rate = (s["n_mismatch"] / s["n_total"]) if s["n_total"] else float("nan")
        print(f"  arm={a:10s} n_total={s['n_total']:4d} n_mismatch={s['n_mismatch']:3d} "
              f"rate={rate:.4f} undetermined_reps={s['n_undetermined_reps']} per_tag={s['per_tag']}")

    print()
    print("=" * 78)
    print("A3 (chunk512) CROSS-BASELINE (vs A1 plain batch=1) -- reproduction of 875346's original G-2")
    print("=" * 78)
    c = cross_agg
    rate = (c["n_mismatch"] / c["n_total"]) if c["n_total"] else float("nan")
    print(f"  n_total={c['n_total']} n_mismatch={c['n_mismatch']} rate={rate:.4f} "
          f"undetermined_reps={c['n_undetermined_reps']} per_tag={c['per_tag']}")

    print()
    print("=" * 78)
    print("PRE-REGISTERED DECISION RULE (PREREG_G2CTRL_2026-08-07.md section 2)")
    print("=" * 78)
    a1 = self_agg["plain"]
    a3 = self_agg["chunk512"]
    p, note = fisher_or_note(a1["n_mismatch"], a1["n_total"], a3["n_mismatch"], a3["n_total"])
    if note:
        print(f"  Fisher exact test NOT computed: {note}")
        verdict = "INDETERMINATE (no test statistic -- see counts above; consult claims-auditor)"
    else:
        print(f"  Fisher exact test A1-self-mismatch vs A3-self-mismatch: p={p:.4g}")
        if a1["n_mismatch"] == 0 and a1["n_total"] > 0 and a3["n_mismatch"] > 0:
            verdict = "(a) chunk512-specific -- A1 perfect (0 mismatches), A3 mismatches. PREREG_GATE2 section 2.1 rule applies (A3 exclusion under consideration)."
        elif p is not None and p >= 0.05:
            verdict = "(b) ordinary batch-composition non-determinism -- A1 vs A3 self-mismatch rates NOT distinguishable (p >= 0.05). The 875346 G-2 FAIL is NOT evidence specific to chunk512; PREREG_GATE2 section 2.1's 'mismatch -> discard A3' rule does not apply to this evidence."
        elif p is not None and p < 0.05 and a1["n_mismatch"] > 0:
            verdict = "INDETERMINATE -- both arms mismatch but rates differ significantly. Report counts + p-value, do not force into (a) or (b)."
        else:
            verdict = "INDETERMINATE -- see counts above."
    print(f"  VERDICT: {verdict}")

    print()
    print("=" * 78)
    print("FIRST-DIVERGING-TOKEN POSITIONS (diagnostic, not a verdict)")
    print("=" * 78)
    for a in arms:
        pos = self_agg[a]["positions"]
        if pos:
            print(f"  arm={a} self-mismatch positions (tag, first_diff_idx): {pos}")
    if cross_agg["positions"]:
        print(f"  arm=chunk512 cross-mismatch positions (tag, first_diff_idx): {cross_agg['positions']}")

    print()
    print("NOTE: this script issues a correctness/availability VERDICT only per the")
    print("pre-registered rule above. It does not compute goodput/latency and must")
    print("not be cited for performance claims (experiment-runner scope; performance")
    print("interpretation is result-analyst's / claims-auditor's).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
