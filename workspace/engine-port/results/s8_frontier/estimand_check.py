#!/usr/bin/env python3
"""M2 (2026-08-02) -- is per-request ITL-p95 measuring what E1's decision rule
needs it to measure?

★★ SCOPE LIMIT, PRE-REGISTERED BEFORE RUNNING THIS ★★
This script RE-SCORES probes that were already collected. PROJECT_STATUS.md's
methodology gate #8 ("when the SLO changes, do NOT re-score the existing
controller -- re-tune to that SLO and measure directly") forbids turning any
number below into a performance claim. Its ONLY licensed use is ESTIMAND
SELECTION: deciding which quantity the main sweep should be pre-registered to
measure, before that sweep runs. Nothing here may be cited as a result, and no
cell ranking produced here is evidence about the lever.

WHY. E1's conjunctive goodput requires, per request, "token-ITL p95 <= SLO"
(CLAUDE.md gate #4). The 2026-08-02 claims-auditor argued that on this workload
that quantity is not decode step time at all:

  - ShareGPT output lengths are short. For a request emitting k tokens there
    are k-1 ITLs, so at k <= 20 the per-request p95 IS (or is adjacent to) the
    per-request MAX -- one stall anywhere in that request's lifetime becomes
    its whole ITL-p95.
  - Server-wide stalls exist even at rate 1: many DIFFERENT requests share the
    same maximum ITL to within ~1ms, which can only happen if the server
    stopped for all of them at once.

If both hold, the ITL term of the decision rule is dominated by "worst stall a
short request happened to sit through", which is not a function of the decode
SM partition in the way the rule assumes -- and no amount of extra reps fixes
that, because it is a bias, not variance.

This script measures both, plus three candidate replacement estimands, so the
choice among them is made on data and written down BEFORE the sweep.

Usage:
  python3 estimand_check.py --dir . --job 870296 --rate 2
  python3 estimand_check.py --dir . --job 867231,870295,870296,870297 --rate 2
"""
import argparse
import collections
import json
import math
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
CELL_D = {"d16": 16, "d24": 24, "d44": 44, "d54": 54, "d92": 92}
PAT = re.compile(
    r"e1cap_(?P<arm>\w+?)_(?P<cell>d16|d24|d44|d54|d92)_(?P<job>\d+)"
    r"_r(?P<rate>[0-9.]+)(?:_s(?P<seed>\d+))?\.jsonl$"
)
STALL_TOL_MS = 2.0    # two requests "share" a max ITL if within this
MIN_SHARERS = 3       # >=3 requests sharing one max => server-wide, not per-request
SHORT_OUTPUT = 20     # ShareGPT's short-output mass
LONG_FILTER = 40      # candidate estimand C's minimum output length

# ★2026-08-02, found while running this script -- the FIRST version of the
# stall test fired on 38/40 probes, which was an artifact of my own threshold,
# not a finding. "N requests share a max ITL to within 2ms" is vacuous when the
# max ITL IS the typical ITL: T8 d92 reported "23 requests share max=18.1ms",
# but 18ms is that cell's ordinary decode step time, so of course many requests
# top out there. An absolute tolerance cannot be read without a scale.
#
# A stall must therefore be defined RELATIVE to the probe's own normal step
# time: the shared value has to be far above it. Same failure family as
# PROJECT_STATUS.md methodology gate #4/#6 -- a statistic that is guaranteed to
# fire tells you nothing about the server.
STALL_MULT = 5.0      # shared max must exceed 5x the probe's typical ITL


def pctl(xs, q):
    if not xs:
        return float("nan")
    ys = sorted(xs)
    pos = q * (len(ys) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ys) - 1)
    return ys[lo] + (ys[hi] - ys[lo]) * (pos - lo)


def load(path):
    o = json.loads(open(path).read().strip().split("\n")[0])
    itls = [[x * 1000.0 for x in v] for v in (o.get("itls") or [])]
    return dict(itls=itls,
                out=o.get("output_lens") or [],
                inp=o.get("input_lens") or [],
                ttft=[t * 1000.0 for t in (o.get("ttfts") or [])],
                err=o.get("errors") or [])


def analyze(d, rate, warmup_s=3.0):
    n_warm = math.ceil(warmup_s * rate)
    idx = [i for i in range(len(d["itls"]))
           if i >= n_warm and d["itls"][i]
           and (d["err"][i] if i < len(d["err"]) else "") == ""]
    if not idx:
        return None
    out = [d["out"][i] if i < len(d["out"]) else 0 for i in idx]
    per_p95 = [pctl(d["itls"][i], 0.95) for i in idx]
    per_p90 = [pctl(d["itls"][i], 0.90) for i in idx]
    per_med = [pctl(d["itls"][i], 0.50) for i in idx]
    per_max = [max(d["itls"][i]) for i in idx]

    # (1) how often is per-request p95 just the max?
    same_as_max = sum(1 for a, b in zip(per_p95, per_max) if abs(a - b) < 1e-9)
    short = sum(1 for k in out if k <= SHORT_OUTPUT)

    # (2) server-wide stall signature: distinct requests sharing one max ITL,
    # AND that shared value far above the probe's own normal step time (see
    # STALL_MULT -- the first version of this test had no scale and fired
    # everywhere).
    typical = pctl(per_med, 0.50)
    top = max(per_max)
    sharers = sum(1 for m in per_max if abs(m - top) < STALL_TOL_MS)
    is_stall = (sharers >= MIN_SHARERS and typical == typical
                and top > STALL_MULT * typical)

    # (3) which requests set the across-request p95, and how long are they?
    order = sorted(range(len(idx)), key=lambda j: per_p95[j])
    k = max(0, math.ceil(0.95 * len(order)) - 1)
    tail = [(round(per_p95[j], 1), out[j]) for j in order[k:]][:5]

    # candidate estimands
    long_idx = [j for j, k_ in enumerate(out) if k_ >= LONG_FILTER]
    return dict(
        n=len(idx), frac_short=short / len(idx), frac_p95_is_max=same_as_max / len(idx),
        stall_ms=top, stall_sharers=sharers, typical_ms=typical, is_stall=is_stall,
        stall_ratio=(top / typical if typical else float("nan")),
        A_p95_of_p95=pctl(per_p95, 0.95),            # as-registered
        B_p95_of_med=pctl(per_med, 0.95),            # median ITL per request
        C_p95_of_p95_long=pctl([per_p95[j] for j in long_idx], 0.95) if long_idx else float("nan"),
        C_n=len(long_idx),
        D_p95_of_p90=pctl(per_p90, 0.95),
        med_of_p95=pctl(per_p95, 0.50),
        tail=tail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=HERE)
    ap.add_argument("--job", required=True, help="comma-separated job ids (one per arm)")
    ap.add_argument("--rate", type=float, required=True)
    ap.add_argument("--warmup-s", type=float, default=3.0)
    a = ap.parse_args()
    jobs = a.job.split(",")

    print("=== M2 estimand diagnostic (2026-08-02) -- ESTIMAND SELECTION ONLY ===")
    print("gate #8: nothing here is a performance claim or a cell ranking.\n")
    print(f"A = p95 across requests of per-request ITL-p95   [AS PRE-REGISTERED]")
    print(f"B = p95 across requests of per-request MEDIAN ITL")
    print(f"C = A, restricted to requests with output_len >= {LONG_FILTER}")
    print(f"D = p95 across requests of per-request ITL-p90\n")

    rows = []
    for job in jobs:
        for fn in sorted(os.listdir(a.dir)):
            m = PAT.search(fn)
            if not m or m["job"] != job or abs(float(m["rate"]) - a.rate) > 1e-9:
                continue
            r = analyze(load(os.path.join(a.dir, fn)), a.rate, a.warmup_s)
            if r:
                rows.append((m["arm"], m["cell"], int(m["seed"] or 0), r))

    print(f"{'arm':>4} {'cell':>5} {'sd':>2} {'n':>3} {'short':>6} {'p95=max':>8} "
          f"{'stall(ms)':>10} {'shared':>7} | {'A':>7} {'B':>7} {'C':>7} {'D':>7} "
          f"{'med(A)':>7} {'A/B':>5}")
    for arm, cell, seed, r in sorted(rows, key=lambda t: (t[0], CELL_D[t[1]], t[2])):
        stall = ("  <-- SERVER-WIDE STALL" if r["is_stall"] else "")
        ab = r["A_p95_of_p95"] / r["B_p95_of_med"] if r["B_p95_of_med"] else float("nan")
        print(f"{arm:>4} {cell:>5} {seed:>2} {r['n']:>3} {r['frac_short']:6.1%} "
              f"{r['frac_p95_is_max']:8.1%} {r['stall_ms']:10.1f} {r['stall_sharers']:7d} | "
              f"{r['A_p95_of_p95']:7.1f} {r['B_p95_of_med']:7.1f} {r['C_p95_of_p95_long']:7.1f} "
              f"{r['D_p95_of_p90']:7.1f} {r['med_of_p95']:7.1f} {ab:5.2f}{stall}")

    print("\n--- what sets A: (per-request ITL-p95 ms, that request's output_len) "
          "for the 5 requests at the top of the A distribution ---")
    for arm, cell, seed, r in sorted(rows, key=lambda t: (t[0], CELL_D[t[1]], t[2])):
        print(f"  {arm:>4} {cell:>5} s{seed}: {r['tail']}")

    agg = collections.defaultdict(list)
    for _, _, _, r in rows:
        agg["short"].append(r["frac_short"])
        agg["ismax"].append(r["frac_p95_is_max"])
        agg["shared"].append(r["stall_sharers"])
        agg["isstall"].append(r["is_stall"])
        agg["ratio"].append(r["stall_ratio"])
        if r["B_p95_of_med"]:
            agg["ab"].append(r["A_p95_of_p95"] / r["B_p95_of_med"])
    n_stall = sum(1 for s in agg["isstall"] if s)
    print(f"\n--- summary over {len(rows)} probes at rate {a.rate} ---")
    print(f"  output_len <= {SHORT_OUTPUT}:                 "
          f"{min(agg['short']):.1%} - {max(agg['short']):.1%}")
    print(f"  requests whose ITL-p95 IS their max ITL: "
          f"{min(agg['ismax']):.1%} - {max(agg['ismax']):.1%}")
    print(f"  probes with a server-wide stall (>= {MIN_SHARERS} requests sharing one max "
          f"AND that max > {STALL_MULT:g}x the probe's typical ITL): {n_stall}/{len(rows)}")
    print(f"  max/typical ITL ratio across probes: "
          f"{min(agg['ratio']):.1f}x - {max(agg['ratio']):.1f}x")
    print(f"  A/B ratio (tail-sensitivity of the registered estimand): "
          f"{min(agg['ab']):.2f}x - {max(agg['ab']):.2f}x")


if __name__ == "__main__":
    main()
