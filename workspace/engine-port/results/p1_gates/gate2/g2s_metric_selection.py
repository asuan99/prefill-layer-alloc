#!/usr/bin/env python3
"""Gate 2-S rev2 §4.1 — primary-metric selection evidence (CPU only, no GPU, read-only).

Reproduces every number in PREREG_GATE2S_2026-08-09.md §4.1 tables A/B/C and §4.1.1,
plus §5.2.0 table (TTFT p95 paired) and §2.3 (KV pool / cudagraph capture) from the
EXISTING rev4 artifacts (jobs 875344 / 875346).  Writes nothing, runs no server.

    cd workspace/engine-port/results/p1_gates/gate2 && python3 g2s_metric_selection.py

WHY THIS FILE EXISTS.  rev1 asserted that `ITLp95p90` was "a continuous quantity with
no censoring".  That was false and the claims-auditor caught it.  rev2 replaces the
primary metric, and the replacement must be auditable without trusting either the
auditor's numbers or mine -- hence this script.

TWO TRAPS IT PINS DOWN (both are correctness gates in the prereg, §6.3-9):

  1. UNITS.  `itls` / `ttfts` in bench_serving jsonl are SECONDS, not ms.  Reading them
     as ms makes every `ITL > 60ms` spike count come out 0.  Asserted below.
  2. PERCENTILE CONVENTION.  With 95 ITL samples per request, linear-interpolation p95
     and nearest-rank p95 straddle the bimodal gap: the per-request p95 median is
     60.1 ms under linear and 154.3 ms under nearest-rank.  BOTH are "correct"; the
     prereg fixes nearest-rank.  Reported side by side below.
"""

from __future__ import annotations

import glob
import json
import math
import os
import re
import statistics as st
from collections import defaultdict

# Prereg grid ONLY (§3): Granite r6 is excluded (§3.2 self-awareness risk #1).
# Including it changes the agnostic MEAN CV (e.g. pooled_p99 4.9% -> 6.2%) while leaving
# the max and both winners unchanged -- so the grid must match the prereg exactly.
CELLS = {"zamba2-27b": ("875344", (2.0, 3.0)), "granite-40-h-micro-base": ("875346", (3.0, 4.0))}
DECISION_ARMS = ("agnostic", "plainaux")  # agnostic == the T/C surrogate (both pdmux)
TAU_MS = 60.0


# ---------------------------------------------------------------- percentiles
def p_linear(a, f):
    a = sorted(a)
    i = (len(a) - 1) * f
    lo, hi = math.floor(i), math.ceil(i)
    return a[lo] if lo == hi else a[lo] + (a[hi] - a[lo]) * (i - lo)


def p_nearest(a, f):
    a = sorted(a)
    return a[max(0, math.ceil(f * len(a)) - 1)]


def kth_largest(a, k):
    return sorted(a, reverse=True)[min(k - 1, len(a) - 1)]


# ---------------------------------------------------------------- metrics
def row_metrics(row, inner=p_nearest):
    """All candidate primary metrics for one bench_serving run (one rate).

    `inner` = percentile convention for the PER-REQUEST p95.  rev3 §4.1.1 requires both:
    Granite r4 plainaux ITLp95p90 CV is 0.7% (nearest) vs 42.2% (linear) -- a 60x flip
    on convention alone, which is exactly what invalidated the rev2 R2 rebuttal.
    """
    itls = [x for x in row["itls"] if x]           # seconds
    per_req_p95 = [inner(x, 0.95) * 1000.0 for x in itls]
    pooled = [v * 1000.0 for x in itls for v in x]
    ttft = [t * 1000.0 for t in row["ttfts"]]
    return {
        "ITLp95p90": p_linear(per_req_p95, 0.90),      # rev1 primary (REJECTED)
        "mean_ITLp95": sum(per_req_p95) / len(per_req_p95),  # rev2 primary
        "med_ITLp95": p_linear(per_req_p95, 0.50),
        "pooled_p95": p_linear(pooled, 0.95),
        "pooled_p99": p_linear(pooled, 0.99),          # rev2 automatic substitute
        "mean_ITL": sum(pooled) / len(pooled),
        "TTFTp95": p_linear(ttft, 0.95),
        "X60": sum(1 for x in itls if max(x) * 1000.0 > TAU_MS) / len(itls),
    }


def assert_units(row):
    """§6.3-9: itls must be seconds.  Fails loudly rather than silently reporting 0 spikes."""
    flat = [v for x in row["itls"] if x for v in x]
    got = (sum(flat) / len(flat)) * 1000.0
    assert abs(got - row["mean_itl_ms"]) < 1e-6, (
        f"UNIT ASSERT FAILED: mean(itls)*1000={got} != mean_itl_ms={row['mean_itl_ms']}"
    )


def iter_reps(tag, job):
    for f in sorted(glob.glob(f"g2_{tag}_*_rep*_{job}.jsonl")):
        if ".holb." in f:
            continue
        base = os.path.basename(f)
        arm = base.split(f"g2_{tag}_")[1].rsplit("_rep", 1)[0]
        rep = int(base.rsplit("_rep", 1)[1].split("_")[0])
        for line in open(f):
            yield arm, rep, json.loads(line)


# ---------------------------------------------------------------- reports
def table_A_inner_structure():
    print("\n=== Table A -- per-request ITL structure (Zamba2 plainaux rep1, rate 3) ===")
    rows = [json.loads(l) for l in open("g2_zamba2-27b_plainaux_rep1_875344.jsonl")]
    row = next(r for r in rows if r["request_rate"] == 3.0)
    assert_units(row)
    itls = [x for x in row["itls"] if x]
    lens = [len(x) for x in itls]
    spikes = sorted(sum(1 for v in x if v * 1000.0 > TAU_MS) for x in itls)
    print(f"  requests={len(itls)}  random_output_len={row['random_output_len']}")
    print(f"  ITL samples/request: min={min(lens)} max={max(lens)} median={st.median(lens)}")
    print(f"  => nearest-rank p95 selects rank ceil(.95*95)=91st smallest = 5th LARGEST")
    print(f"  spikes(ITL>{TAU_MS:.0f}ms)/request: q25={p_linear(spikes,.25):.0f} "
          f"q50={p_linear(spikes,.5):.0f} q75={p_linear(spikes,.75):.0f} "
          f"mean={sum(spikes)/len(spikes):.2f} max={max(spikes)}")
    print("  ^ the rank p95 picks (5th largest) EQUALS the median spike count (5)")

    print("\n=== Table B -- both percentile conventions reproduced ===")
    lin = [p_linear(x, 0.95) * 1000 for x in itls]
    nea = [p_nearest(x, 0.95) * 1000 for x in itls]
    print(f"  {'stat':<18}{'linear':>12}{'nearest-rank':>15}")
    for nm, f in (("per-req p95 q25", .25), ("per-req p95 q50", .50), ("per-req p95 q75", .75)):
        print(f"  {nm:<18}{p_linear(lin,f):>12.1f}{p_linear(nea,f):>15.1f}")
    print(f"  {'=> ITLp95p90':<18}{p_linear(lin,.90):>12.1f}{p_linear(nea,.90):>15.1f}")


def table_C_instability():
    print("\n=== Table C -- unstable_frac (rank +-1 crosses the mode gap) and boundary_frac ===")
    print("  unstable_r := |k4-k6|/k5 > 0.50 ;  boundary_r := 0.75*tau <= k5 <= 1.33*tau")
    print(f"  {'model':<26}{'arm':<10}{'rate':>5}{'unstable':>11}{'boundary':>11}")
    for tag, (job, _) in CELLS.items():
        agg = defaultdict(lambda: ([], []))
        for arm, _rep, row in iter_reps(tag, job):
            if arm not in DECISION_ARMS:
                continue
            for x in row["itls"]:
                if len(x) < 10:
                    continue
                k4, k5, k6 = (kth_largest(x, k) * 1000 for k in (4, 5, 6))
                agg[(arm, row["request_rate"])][0].append(abs(k4 - k6) / k5 > 0.50 if k5 else False)
                agg[(arm, row["request_rate"])][1].append(0.75 * TAU_MS <= k5 <= 1.33 * TAU_MS)
        for arm, rate in sorted(agg):
            u, b = agg[(arm, rate)]
            print(f"  {tag:<26}{arm:<10}{rate:>5.0f}{sum(u)/len(u):>11.3f}{sum(b)/len(b):>11.3f}")
    print("  ^ REBUTTAL R1: boundary_frac is 0.000 almost everywhere -> the auditor's")
    print("    'boundary proximity' diagnostic cannot fire.  unstable_frac is the live one.")


def table_cv(inner=p_nearest, label="inner=NEAREST"):
    """rev3 §4.1.1: run under BOTH inner conventions -- the rev2 R2 rebuttal died here."""
    print(f"\n=== §4.1.1 [{label}] -- rep-to-rep CV (n=10).  DECISION ROWS = 'agnostic' ===")
    keys = ["ITLp95p90", "mean_ITLp95", "med_ITLp95", "pooled_p95", "pooled_p99", "mean_ITL"]
    agn = defaultdict(list)
    for tag, (job, rates) in CELLS.items():
        data = defaultdict(lambda: defaultdict(list))
        for arm, _rep, row in iter_reps(tag, job):
            if row["request_rate"] not in rates:
                continue
            m = row_metrics(row, inner=inner)
            for k in keys:
                data[(arm, row["request_rate"])][k].append(m[k])
        print(f"\n  --- {tag} ({job}) ---")
        print(f"  {'arm':<10}{'rate':>5}{'n':>4}" + "".join(f"{k:>14}" for k in keys))
        for arm, rate in sorted(data):
            m = data[(arm, rate)]
            cells = []
            for k in keys:
                v = m[k]
                mu = sum(v) / len(v)
                cv = 100 * st.stdev(v) / mu if mu > 0 else float("nan")
                cells.append(f"{cv:>13.1f}%" if mu > 0 else f"{'n/a':>14}")
                if arm == "agnostic" and mu > 0:
                    agn[k].append(cv)
            print(f"  {arm:<10}{rate:>5.0f}{len(m[keys[0]]):>4}" + "".join(cells))
    print(f"  {'AGNOSTIC max':<19}" + "".join(f"{max(agn[k]):>13.1f}%" for k in keys))
    print(f"  {'AGNOSTIC mean':<19}" + "".join(f"{sum(agn[k])/len(agn[k]):>13.1f}%" for k in keys))
    print(f"  worst-case winner: {min(keys, key=lambda k: max(agn[k]))}"
          f"   |   mean winner: {min(keys, key=lambda k: sum(agn[k]))}")
    print("  ^ §4.1.1.1: THE CRITERION PICKS THE WINNER.  rev3 therefore reports BOTH")
    print("    (primary-alpha = mean_ITLp95, primary-beta = pooled_p99); no criterion is chosen.")


def table_ttft():
    print("\n=== §5.2.0 -- TTFT p95, agnostic vs plainaux, paired by rep ===")
    print(f"  {'model':<26}{'rate':>5}{'plainaux':>11}{'agnostic':>11}{'delta':>9}{'95% CI':>22}")
    for tag, (job, _) in CELLS.items():
        d = defaultdict(dict)
        for arm, rep, row in iter_reps(tag, job):
            if arm in DECISION_ARMS:
                d[(row["request_rate"], rep)][arm] = p_linear([t * 1000 for t in row["ttfts"]], 0.95)
        for rate in sorted({k[0] for k in d}):
            pairs = [(v["plainaux"], v["agnostic"]) for (r, _), v in sorted(d.items())
                     if r == rate and len(v) == 2]
            pa = [p for p, _ in pairs]
            ag = [a for _, a in pairs]
            diff = [a - p for p, a in pairs]
            n = len(diff)
            m, se = sum(diff) / n, st.stdev(diff) / math.sqrt(n)
            t = {9: 2.262, 8: 2.306, 7: 2.365, 4: 2.776}.get(n - 1, 2.262)
            mp = sum(pa) / n
            ci = f"[{100*(m-t*se)/mp:+.1f}%, {100*(m+t*se)/mp:+.1f}%]"
            print(f"  {tag:<26}{rate:>5.0f}{mp:>11.1f}{sum(ag)/n:>11.1f}"
                  f"{100*m/mp:>8.1f}%{ci:>22}")
    print("  ^ every CI excludes both 0 and +10% -> rev1's '+10% TTFT gate' was a sealed null.")


def table_kv_pool():
    print("\n=== §2.3 -- KV pool + cudagraph capture per arm (server logs) ===")
    print(f"  {'model':<26}{'arm':<10}{'KV #tokens':>13}{'capture':>28}")
    kv_re = re.compile(r"KV Cache is allocated\. #tokens: (\d+)")
    cap_re = re.compile(r"Capture cuda graph end\. Time elapsed: ([\d.]+) s\. mem usage=([\d.]+) GB")
    for tag, (job, _) in CELLS.items():
        for arm in ("plain", "plainaux", "chunk512", "agnostic"):
            path = f"g2_{tag}_srv_{arm}_rep1_{job}.log"
            if not os.path.exists(path):
                continue
            txt = open(path, errors="ignore").read()
            kv = kv_re.search(txt)
            cap = cap_re.search(txt)
            capstr = f"{cap.group(1)}s / {cap.group(2)}GB" if cap else "-"
            print(f"  {tag:<26}{arm:<10}{kv.group(1) if kv else '-':>13}{capstr:>28}")
    print("  ^ pdmux KV pool is ~0.031% smaller than fused (extra cudagraph memory:")
    print("    4 stream groups vs 1).  T and C share one yml -> must be IDENTICAL (§6.3-8).")
    print("  ^ /get_server_info's max_total_tokens is null in ALL 114 rev4 flags files")
    print("    (rev2 said 56 -- that was a zamba2-only count; corrected in rev4 SS2.3-b),")
    print("    so parsing the server log is the only route (§2.3-b).")


def check_two_rates_per_file():
    print("\n=== §12-4 -- rows per rep file (duration must be SUMMED, never max()) ===")
    f = "g2_zamba2-27b_plainaux_rep1_875344.jsonl"
    rows = [json.loads(l) for l in open(f)]
    print(f"  {f}: {len(rows)} rows -> tags {[r['tag'] for r in rows]}")
    print(f"  durations {[round(r['duration'],1) for r in rows]}  "
          f"sum={sum(r['duration'] for r in rows):.1f}  max={max(r['duration'] for r in rows):.1f}")


if __name__ == "__main__":
    table_A_inner_structure()
    table_C_instability()
    table_cv(p_linear, "inner=LINEAR")
    table_cv(p_nearest, "inner=NEAREST")
    table_ttft()
    table_kv_pool()
    check_two_rates_per_file()
    print("\nDone.  No files written, no GPU used.")
