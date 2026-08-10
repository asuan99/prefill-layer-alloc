#!/usr/bin/env python3
"""Score the T4-1 deterministic-inference reconfirmation campaign
(g2det_run.sbatch, task brief 2026-08-09).

Usage:
  python3 g2det_analyze.py <summary_off_json> <summary_on_json>

Applies the pre-registered decision rule fixed in g2det_run.sbatch's header
comment (recorded before the job ran):

  OFF reproduces (chunk512 self-mismatch >= 1 across 3 reps / 48 trials)
    AND ON has ZERO self-mismatches in all 3 reps (48/48 match)
      => CONFIRMED (reduction-order attribution)
  ON still shows >= 1 self-mismatch in any rep
      => REFUTED (reduction-order is not the (sole) cause)
  OFF does NOT reproduce (0 mismatches in OFF's chunk512 arm)
      => NO VERDICT (precondition not reproduced, ON uninterpretable)

Guard added 2026-08-10 (job-876699 post-mortem, methodology gate #21): each of
the three branches above presupposes that the arm was actually MEASURED. If an
arm produced zero scored trials -- boot failure, a self-baseline greedy_call
that TIMED OUT or errored, or all reps UNDETERMINED -- the verdict is
  NO VERDICT (MEASUREMENT ABSENT),
never REFUTED. Before this guard, an ON chunk512 arm with no data at all fell
through `on_clean = False` straight into "REFUTED", i.e. a measurement failure
would have been reported as a substantive negative result. That is exactly the
hazard job 876699 created: its ON chunk512 self-baseline call hung for ~52 min
and the arm never ran.

No performance numbers anywhere in this script's output -- correctness
counts and mismatch positions only. This is a diagnostic verdict about a
reproducibility defect, not a performance claim.
"""
import json
import math
import sys
from collections import defaultdict


def fisher_exact_2x2(a, b, c, d):
    """Two-sided Fisher exact test p-value for [[a,b],[c,d]] (no scipy)."""
    n = a + b + c + d
    row1, row2 = a + b, c + d
    col1, col2 = a + c, b + d

    def hyper_p(x):
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

    lo, hi = max(0, row1 - col2), min(row1, col1)
    p_obs = hyper_p(a)
    total, eps = 0.0, 1e-10
    for x in range(lo, hi + 1):
        px = hyper_p(x)
        if px <= p_obs * (1 + eps):
            total += px
    return min(1.0, total)


def load_rows(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def agg_self(rows, arm):
    n_total = n_mismatch = n_undetermined_reps = 0
    positions = []
    per_rep_mismatch = []
    for d in rows:
        if d.get("arm") != arm or not d.get("self_baseline_ok"):
            continue
        for rep in d.get("reps", []):
            if rep["verdict_self"] == "UNDETERMINED":
                n_undetermined_reps += 1
                per_rep_mismatch.append(None)
                continue
            n_total += rep["n_ok"]
            n_mismatch += rep["n_mismatch_self"]
            per_rep_mismatch.append(rep["n_mismatch_self"])
            for m in rep["self_mismatch_positions"]:
                positions.append(m["first_diff_token_index"])
    return {
        "n_total": n_total, "n_mismatch": n_mismatch,
        "n_undetermined_reps": n_undetermined_reps,
        "per_rep_mismatch": per_rep_mismatch, "positions": positions,
    }


def no_data_rows(off_rows, on_rows, arms):
    """(detmode, arm) cells that produced NO scored trial, with the reason.

    A cell can be data-less for four distinct reasons, none of which is a
    correctness finding:
      * no summary row was written at all            -- the job died (or was
        killed by SLURM) before reaching that arm. This is what job 876699 did:
        one greedy_call hung ~52 min, the wall clock ran out, and ON/chunk512 +
        ON/agnostic never ran.
      * the server never booted                       (boot_ok false)
      * the batch=1 self-baseline greedy_call failed  (self_baseline_ok false),
        including self_baseline_status=TIMEOUT -- the job-876699 hang class
      * every rep came back UNDETERMINED from the concurrent gate
    Reported explicitly so "we did not measure this" is never silently read as
    "we measured this and it was dirty" (methodology gate #21).
    """
    out = []
    for detmode, rows in (("off", off_rows), ("on", on_rows)):
        seen = set()
        for d in rows:
            arm = d.get("arm", "?")
            seen.add(arm)
            if not d.get("boot_ok"):
                out.append({"detmode": detmode, "arm": arm, "reason": "boot_failed"})
                continue
            if not d.get("self_baseline_ok"):
                st = d.get("self_baseline_status", "UNRECORDED")
                extra = ""
                if st == "TIMEOUT":
                    extra = (" (greedy_call exceeded G2_GREEDY_MAX_TIME -- MEASUREMENT FAILURE,"
                             " job-876699 hang class, NOT a mismatch)")
                out.append({"detmode": detmode, "arm": arm,
                            "reason": f"self_baseline_not_built:{st}{extra}"})
                continue
            reps = d.get("reps", [])
            if reps and all(r.get("verdict_self") == "UNDETERMINED" for r in reps):
                out.append({"detmode": detmode, "arm": arm, "reason": "all_reps_UNDETERMINED"})
            elif not reps:
                out.append({"detmode": detmode, "arm": arm, "reason": "no_reps_recorded"})
        for arm in arms:
            if arm not in seen:
                out.append({"detmode": detmode, "arm": arm,
                            "reason": "no_summary_row_written (arm never ran -- job ended early;"
                                      " NOT a measured result)"})
    return out


def agg_cross(rows, arm="chunk512"):
    n_total = n_mismatch = n_undetermined_reps = 0
    positions = []
    for d in rows:
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
            for m in rep["cross_mismatch_positions"] or []:
                positions.append(m["first_diff_token_index"])
    return {"n_total": n_total, "n_mismatch": n_mismatch,
            "n_undetermined_reps": n_undetermined_reps, "positions": positions}


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 1
    off_path, on_path = sys.argv[1], sys.argv[2]
    off_rows = load_rows(off_path)
    on_rows = load_rows(on_path)
    if not off_rows or not on_rows:
        print(f"NO ROWS LOADED (off={len(off_rows)} on={len(on_rows)}) -- check paths")
        return 1

    arms = ["plain", "plainaux", "chunk512", "agnostic"]

    print("=" * 78)
    print("T4-1 -- SELF-BASELINE MISMATCH RATES BY DETMODE (16-concurrent vs own batch=1)")
    print("=" * 78)
    off_self = {a: agg_self(off_rows, a) for a in arms}
    on_self = {a: agg_self(on_rows, a) for a in arms}
    for detmode, agg in (("OFF", off_self), ("ON", on_self)):
        for a in arms:
            s = agg[a]
            rate = (s["n_mismatch"] / s["n_total"]) if s["n_total"] else float("nan")
            print(f"  detmode={detmode:3s} arm={a:10s} n_total={s['n_total']:4d} "
                  f"n_mismatch={s['n_mismatch']:3d} rate={rate:.4f} "
                  f"per_rep_mismatch={s['per_rep_mismatch']} "
                  f"undetermined_reps={s['n_undetermined_reps']}")

    print()
    print("=" * 78)
    print("chunk512 CROSS-BASELINE (vs A1 plain batch=1, same detmode) -- reproduction check")
    print("=" * 78)
    off_cross = agg_cross(off_rows, "chunk512")
    on_cross = agg_cross(on_rows, "chunk512")
    for detmode, c in (("OFF", off_cross), ("ON", on_cross)):
        rate = (c["n_mismatch"] / c["n_total"]) if c["n_total"] else float("nan")
        print(f"  detmode={detmode:3s} n_total={c['n_total']} n_mismatch={c['n_mismatch']} "
              f"rate={rate:.4f} undetermined_reps={c['n_undetermined_reps']} positions={c['positions']}")

    print()
    print("=" * 78)
    print("FIRST-DIVERGING-TOKEN POSITIONS (diagnostic)")
    print("=" * 78)
    for detmode, agg in (("OFF", off_self), ("ON", on_self)):
        for a in arms:
            pos = agg[a]["positions"]
            if pos:
                print(f"  detmode={detmode} arm={a} self-mismatch first-diff-token-indices: {pos}")

    print()
    print("=" * 78)
    print("PRE-REGISTERED DECISION RULE (g2det_run.sbatch header comment, fixed pre-run)")
    print("=" * 78)
    off_c512 = off_self["chunk512"]
    on_c512 = on_self["chunk512"]
    # "measured" = this detmode/arm produced at least one scored trial. Absence
    # of measurement (server never booted, self-baseline greedy_call TIMEOUT/
    # ERROR, all reps UNDETERMINED) is NOT evidence about the hypothesis. See
    # no_data_rows() below and methodology gate #21.
    off_measured = off_c512["n_total"] > 0
    on_measured = on_c512["n_total"] > 0
    off_reproduced = off_measured and off_c512["n_mismatch"] >= 1
    on_clean = on_measured and on_c512["n_mismatch"] == 0 and all(
        m == 0 for m in on_c512["per_rep_mismatch"] if m is not None
    ) and on_c512["n_undetermined_reps"] == 0

    print(f"  OFF chunk512 self: n_total={off_c512['n_total']} n_mismatch={off_c512['n_mismatch']} "
          f"per_rep={off_c512['per_rep_mismatch']} -> measured={'YES' if off_measured else 'NO'} "
          f"reproduced={'YES' if off_reproduced else 'NO'}")
    print(f"  ON  chunk512 self: n_total={on_c512['n_total']} n_mismatch={on_c512['n_mismatch']} "
          f"per_rep={on_c512['per_rep_mismatch']} -> measured={'YES' if on_measured else 'NO'} "
          f"all_reps_clean={'YES' if on_clean else 'NO'}")

    nodata = no_data_rows(off_rows, on_rows, arms)
    if nodata:
        print("  ARMS WITH NO DATA (measurement absent -- excluded from the rule, never scored"
              " as a mismatch or as 'not clean'):")
        for row in nodata:
            print(f"    detmode={row['detmode']:3s} arm={row['arm']:10s} reason={row['reason']}")

    if not off_measured:
        verdict = ("NO VERDICT (MEASUREMENT ABSENT) -- the OFF chunk512 arm produced no scored "
                   "trials at all (see 'ARMS WITH NO DATA' above: boot failure, greedy_call "
                   "TIMEOUT/ERROR, or all reps UNDETERMINED). This is a harness/measurement "
                   "failure, NOT a finding about reduction order. Re-run that arm; do not report "
                   "a verdict from this file.")
    elif not off_reproduced:
        verdict = "NO VERDICT -- OFF did not reproduce the original G-2 failure (0 self-mismatches). The failure's preconditions were not present in this boot; the ON-arm result below is uninterpretable and must NOT be read as evidence either way."
    elif not on_measured:
        # The pre-registered REFUTED condition is "ON still shows >= 1
        # self-mismatch in any rep". With zero scored trials there is no such
        # observation, so REFUTED is not reachable here -- routing no-data to
        # REFUTED (which the pre-2026-08-10 code did, via on_clean=False) would
        # be labelling a measurement failure as a gate failure. This branch
        # does not relax the rule; it refuses to evaluate it without data.
        verdict = ("NO VERDICT (MEASUREMENT ABSENT) -- OFF reproduced the failure, but the ON "
                   "chunk512 arm produced no scored trials at all (see 'ARMS WITH NO DATA' above; "
                   "the job-876699 failure mode was exactly this: the ON chunk512 self-baseline "
                   "greedy_call hung and the arm never ran). The pre-registered REFUTED condition "
                   "requires OBSERVING >=1 self-mismatch under ON; zero observations is not that. "
                   "Re-run the ON chunk512 arm.")
    elif on_clean:
        verdict = "CONFIRMED -- reduction-order attribution. OFF reproduced the failure, ON shows 0 self-mismatches across all reps. Recommend (to doc-steward/claims-auditor) lifting the inheritance-discard rule for the Gate 2 rev4 Granite cells and the E-A cmp2 Granite cells."
    else:
        verdict = "REFUTED -- reduction-order is not the (sole) cause. ON still shows self-mismatches. Keep the inheritance-discard rule."
    print(f"  VERDICT: {verdict}")

    print()
    print("=" * 78)
    print("SUPPLEMENTARY: Fisher exact test OFF-chunk512 vs ON-chunk512 self-mismatch rate")
    print("(diagnostic magnitude context only -- the decision rule above is count-based, not p-based)")
    print("=" * 78)
    if off_c512["n_total"] and on_c512["n_total"]:
        a_mis, a_ok = off_c512["n_mismatch"], off_c512["n_total"] - off_c512["n_mismatch"]
        b_mis, b_ok = on_c512["n_mismatch"], on_c512["n_total"] - on_c512["n_mismatch"]
        p = fisher_exact_2x2(a_mis, a_ok, b_mis, b_ok)
        print(f"  Fisher exact p (OFF vs ON chunk512 self-mismatch) = {p:.4g}")
    else:
        print("  not computed -- one side has n=0")

    print()
    print("=" * 78)
    print("NEGATIVE CONTROL CHECK (plain/plainaux/agnostic, both detmodes)")
    print("=" * 78)
    for detmode, agg in (("OFF", off_self), ("ON", on_self)):
        for a in ("plain", "plainaux", "agnostic"):
            s = agg[a]
            if s["n_total"] == 0:
                flag = "  <-- NO DATA (not measured; not a clean result either)"
            elif s["n_mismatch"] == 0:
                flag = ""
            else:
                flag = "  <-- UNEXPECTED NONZERO"
            print(f"  detmode={detmode:3s} arm={a:10s} n_mismatch={s['n_mismatch']}/{s['n_total']}{flag}")

    print()
    print("NOTE: this script issues a correctness/reproducibility VERDICT only, per")
    print("the pre-registered rule above. It does not compute goodput/latency and")
    print("must not be cited for performance claims (experiment-runner scope;")
    print("performance interpretation is result-analyst's / claims-auditor's).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
