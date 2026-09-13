#!/usr/bin/env python3
"""lambda* (capacity) decision rule for the campaign's stage 0 -- fixed before data.

The rule is inherited VERBATIM in substance from the already-audited probe C
capacity rule (`results/longctx_conflict/probes/c_capacity_label.py` R1/R5,
pre-registered in `PREREG_CAPACITY_2026-09-09.md`), with one change: the key is
the REQUEST SHAPE, not the arm, because this stage measures one arm (the
reference B1 = legacy, fixed D44) across shapes instead of one shape across arms.

  R1 KNEE_BRACKETED[shape]  there exists a tested offered rate with
                            achieved/offered >= 0.95 AND a strictly higher
                            tested rate with achieved/offered <= 0.90.
  R2 LAMBDA_STAR[shape]     measurement, no threshold: the largest ACHIEVED
                            rate among saturated cells (achieved/offered <=
                            0.90).  Reported only when R1 is KNEE_BRACKETED.
  A cell whose JSON is missing (boot or bench failure) is UNRESOLVED, which is
  a MEASUREMENT failure and never a rule failure (de-confound lesson 21).

No arm comparison is possible or permitted: every ladder is relative to the
reference arm's own predicted capacity, and the prior campaign measured the same
quantity differing 5x across splits (D16 0.933 vs D92 0.187 req/s).

usage: lambda0_label.py <cells_dir>   |   --selftest
"""
import json
import sys
from pathlib import Path

ACH_HI = 0.95
ACH_LO = 0.90


def rule(cells):
    """cells: {shape_key: [ {offered_rate, achieved_rate, achieved_over_offered}, ... ]}"""
    out = {}
    for shape, seq in sorted(cells.items()):
        if any(c is None for c in seq) or not seq:
            out[shape] = {"verdict": "UNRESOLVED", "reason": "missing cell(s)",
                          "lambda_star": None, "ladder": None}
            continue
        ladder = sorted(((c["offered_rate"], c["achieved_over_offered"],
                          c["achieved_rate"]) for c in seq), key=lambda t: t[0])
        bracketed = any(
            lo[1] >= ACH_HI and hi[1] <= ACH_LO and hi[0] > lo[0]
            for i, lo in enumerate(ladder) for hi in ladder[i + 1:])
        saturated = [a for _, r, a in ladder if r <= ACH_LO]
        out[shape] = {
            "verdict": "KNEE_BRACKETED" if bracketed else "KNEE_NOT_BRACKETED",
            "lambda_star": (max(saturated) if (bracketed and saturated) else None),
            "ladder": [{"offered": o, "ach_over_off": r, "achieved": a}
                       for o, r, a in ladder],
        }
    return out


def selftest():
    # Reproduce the published, audited probe-C ladders with this code (job
    # 905835, RESULT_CAPACITY_PRECOMP / ccap_905835.out).  If the rule here does
    # not reproduce the verdicts that were already audited, it is the wrong rule.
    d16 = [{"offered_rate": 0.56, "achieved_over_offered": 1.17, "achieved_rate": 0.655},
           {"offered_rate": 0.85, "achieved_over_offered": 1.08, "achieved_rate": 0.918},
           {"offered_rate": 1.22, "achieved_over_offered": 0.765, "achieved_rate": 0.933}]
    d44 = [{"offered_rate": 0.40, "achieved_over_offered": 1.14, "achieved_rate": 0.456},
           {"offered_rate": 0.60, "achieved_over_offered": 1.09, "achieved_rate": 0.654},
           {"offered_rate": 0.86, "achieved_over_offered": 0.785, "achieved_rate": 0.675}]
    got = rule({"d16": d16, "d44": d44})
    assert got["d16"]["verdict"] == "KNEE_BRACKETED", got["d16"]
    assert abs(got["d16"]["lambda_star"] - 0.933) < 1e-9, got["d16"]
    assert got["d44"]["verdict"] == "KNEE_BRACKETED"
    assert abs(got["d44"]["lambda_star"] - 0.675) < 1e-9, got["d44"]

    # A ladder that never saturates must NOT be called bracketed, and must not
    # yield a lambda* -- the failure mode this stage is most likely to hit on
    # the unanchored (256,512) shape.
    flat = [{"offered_rate": r, "achieved_over_offered": 1.05, "achieved_rate": r}
            for r in (1.0, 2.0, 4.0)]
    assert rule({"a": flat})["a"]["verdict"] == "KNEE_NOT_BRACKETED"
    assert rule({"a": flat})["a"]["lambda_star"] is None

    # Saturation with no >=0.95 cell below it is also not bracketed (the knee is
    # below the ladder, i.e. the ladder was aimed too high).
    high = [{"offered_rate": r, "achieved_over_offered": 0.7, "achieved_rate": 0.7 * r}
            for r in (4.0, 8.0)]
    assert rule({"a": high})["a"]["verdict"] == "KNEE_NOT_BRACKETED"

    # Ordering must come from the offered rate, not from input order.
    shuffled = [d16[2], d16[0], d16[1]]
    assert rule({"a": shuffled})["a"]["verdict"] == "KNEE_BRACKETED"

    # A missing cell is UNRESOLVED, never a rule failure.
    assert rule({"a": [d16[0], None, d16[2]]})["a"]["verdict"] == "UNRESOLVED"
    print("SELFTEST OK (reproduces the audited probe-C verdicts D16/D44 = "
          "KNEE_BRACKETED with lambda* 0.933/0.675; non-saturating and "
          "aimed-too-high ladders are NOT bracketed; order-independent; "
          "missing cell -> UNRESOLVED)")


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--selftest":
        selftest()
    elif len(sys.argv) == 2:
        d = Path(sys.argv[1])
        cells = {}
        for f in sorted(d.glob("cell_*.json")):
            rec = json.load(open(f))
            cells.setdefault(rec["shape"], []).append(rec)
        print(json.dumps(rule(cells), indent=2, sort_keys=True))
    else:
        print(__doc__)
        sys.exit(2)
