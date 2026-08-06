#!/usr/bin/env python3
"""P1 operating-point (cudagraph-ON) contrast: post-hoc measurement extraction.

Consumes raw --output-details JSONL from p1op_run.sbatch (Phase 0 capacity scan
+ Phase 1 n=5 paired measurement) and prints:
  - Phase 0 capacity-scan table + pre-registered cliff flags (A/B/C, see PREREG.md)
  - Phase 1 per (model, policy, rate) cell: goodput mean +/- sd (n), TTFT p50/p95/p99,
    TPOT p50/p95
  - paired (agnostic - plain) goodput diff per rate, using same-repeat pairing
  - mechanical pre-registered decision-rule verdict (shrink / demote / confirm)

This script computes MEASUREMENTS and applies the pre-registered mechanical rule
only. It does not interpret results -- that is result-analyst / claims-auditor's
job (see PREREG.md and the experiment-runner charter).

Usage:
  python3 p1op_analyze.py --dir <p1_opint dir> --tag <model tag> [--tag <tag2> ...]
"""
import argparse
import glob
import json
import math
import os
import statistics
import sys

SLO_TTFT = 3.0
SLO_TPOT = 0.06


def pctl(vals, p):
    if not vals:
        return float("nan")
    s = sorted(vals)
    k = (len(s) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] + (s[c] - s[f]) * (k - f)


def load_jsonl(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            rows.append(json.loads(ln))
    return rows


def cell_metrics(row):
    """Given one bench_serving --output-details result row, compute
    per-request TPOT (mean ITL), TTFT array, goodput, and basic counts."""
    ttfts = row.get("ttfts") or []
    itls = row.get("itls") or []
    errors = row.get("errors") or []
    dur = row.get("duration") or 0.0
    n_total = len(ttfts)
    tpots = []
    good = 0
    for i in range(n_total):
        err = errors[i] if i < len(errors) else None
        if err:
            continue
        tt = ttfts[i]
        il = itls[i] if i < len(itls) else []
        tp = (sum(il) / len(il)) if il else float("inf")
        tpots.append(tp)
        if tt is not None and tt <= SLO_TTFT and tp <= SLO_TPOT:
            good += 1
    n_completed = row.get("completed", sum(1 for e in errors if not e) if errors else n_total)
    return {
        "request_rate": row.get("request_rate"),
        "n_total": n_total,
        "n_completed": n_completed,
        "duration": dur,
        "good": good,
        "goodput": (good / dur) if dur else float("nan"),
        "request_throughput": row.get("request_throughput"),
        "ttft_p50": pctl([t for t in ttfts if t is not None], 0.50),
        "ttft_p95": pctl([t for t in ttfts if t is not None], 0.95),
        "ttft_p99": pctl([t for t in ttfts if t is not None], 0.99),
        "tpot_p50": pctl(tpots, 0.50),
        "tpot_p95": pctl(tpots, 0.95),
    }


def phase0_report(outdir, tag):
    print(f"\n=== Phase 0 capacity scan: tag={tag} ===")
    scan_rates = [1, 2, 3, 4, 5, 6, 8, 10, 12]
    main_rates = [2, 3, 4, 6]
    for pol in ("plain", "agnostic"):
        paths = sorted(glob.glob(os.path.join(outdir, f"p1op_{tag}_capscan_{pol}_*.jsonl")))
        if not paths:
            print(f"  policy={pol}: NO capscan file found")
            continue
        rows = load_jsonl(paths[-1])
        by_rate = {}
        for r in rows:
            m = cell_metrics(r)
            by_rate[m["request_rate"]] = m
        print(f"  policy={pol}  file={paths[-1]}")
        print(f"  {'rate':>5} {'compl/total':>12} {'ttft_p50(s)':>12} {'goodput':>9} {'req_thpt':>9} {'cliffA':>7} {'cliffB':>7} {'cliffC':>7}")
        prev_ttft_p50 = None
        for rate in scan_rates:
            m = by_rate.get(rate)
            if not m:
                print(f"  {rate:>5}  (missing)")
                continue
            ratio = m["n_completed"] / m["n_total"] if m["n_total"] else float("nan")
            flagA = ratio < 1.0
            flagB = prev_ttft_p50 is not None and m["ttft_p50"] > 2 * prev_ttft_p50
            rt = m["request_throughput"] or float("nan")
            flagC = (m["goodput"] / rt) < 0.95 if rt else True
            mark = " <== scored" if rate in main_rates else ""
            print(
                f"  {rate:>5} {m['n_completed']:>5}/{m['n_total']:<5} "
                f"{m['ttft_p50']:>12.3f} {m['goodput']:>9.3f} {rt:>9.3f} "
                f"{str(flagA):>7} {str(flagB):>7} {str(flagC):>7}{mark}"
            )
            prev_ttft_p50 = m["ttft_p50"]


def phase1_report(outdir, tag, nrep=5):
    print(f"\n=== Phase 1 n={nrep} paired measurement: tag={tag} ===")
    main_rates = [2, 3, 4, 6]
    # cell[(pol,rate)] = list of metrics dicts across reps
    cell = {(pol, r): [] for pol in ("plain", "agnostic") for r in main_rates}
    for pol in ("plain", "agnostic"):
        for rep in range(1, nrep + 1):
            paths = sorted(glob.glob(os.path.join(outdir, f"p1op_{tag}_{pol}_rep{rep}_*.jsonl")))
            if not paths:
                print(f"  MISSING: policy={pol} rep={rep}")
                continue
            rows = load_jsonl(paths[-1])
            by_rate = {}
            for r in rows:
                m = cell_metrics(r)
                by_rate[m["request_rate"]] = m
            for rate in main_rates:
                m = by_rate.get(rate)
                if m:
                    m["rep"] = rep
                    cell[(pol, rate)].append(m)
                else:
                    print(f"  MISSING CELL: policy={pol} rep={rep} rate={rate}")

    print(f"\n  {'policy':>9} {'rate':>5} {'n':>3} {'goodput mean':>13} {'sd':>7} {'ttft_p50':>9} {'ttft_p95':>9} {'ttft_p99':>9} {'tpot_p50':>9} {'tpot_p95':>9}")
    summary = {}
    for pol in ("plain", "agnostic"):
        for rate in main_rates:
            ms = cell[(pol, rate)]
            gp = [m["goodput"] for m in ms]
            if not gp:
                print(f"  {pol:>9} {rate:>5}  NO DATA")
                continue
            mean_gp = statistics.mean(gp)
            sd_gp = statistics.stdev(gp) if len(gp) > 1 else 0.0
            ttft50 = statistics.mean(m["ttft_p50"] for m in ms)
            ttft95 = statistics.mean(m["ttft_p95"] for m in ms)
            ttft99 = statistics.mean(m["ttft_p99"] for m in ms)
            tpot50 = statistics.mean(m["tpot_p50"] for m in ms)
            tpot95 = statistics.mean(m["tpot_p95"] for m in ms)
            summary[(pol, rate)] = {"goodput": gp, "mean": mean_gp, "sd": sd_gp}
            print(
                f"  {pol:>9} {rate:>5} {len(gp):>3} {mean_gp:>13.3f} {sd_gp:>7.3f} "
                f"{ttft50:>9.3f} {ttft95:>9.3f} {ttft99:>9.3f} {tpot50*1000:>8.1f}m {tpot95*1000:>8.1f}m"
            )

    print(f"\n  paired diff (agnostic - plain), same-rep pairing:")
    print(f"  {'rate':>5} {'n_pairs':>7} {'mean_diff':>10} {'%diff_of_plain':>15} {'sd_diff':>8}")
    verdicts = {}
    for rate in main_rates:
        pln = {m["rep"]: m["goodput"] for m in cell[("plain", rate)]}
        agn = {m["rep"]: m["goodput"] for m in cell[("agnostic", rate)]}
        reps = sorted(set(pln) & set(agn))
        if not reps:
            print(f"  {rate:>5}  NO PAIRED DATA")
            continue
        diffs = [agn[r] - pln[r] for r in reps]
        mean_diff = statistics.mean(diffs)
        sd_diff = statistics.stdev(diffs) if len(diffs) > 1 else 0.0
        plain_mean = statistics.mean(pln[r] for r in reps)
        pct = (mean_diff / plain_mean * 100) if plain_mean else float("nan")
        print(f"  {rate:>5} {len(reps):>7} {mean_diff:>10.4f} {pct:>14.2f}% {sd_diff:>8.4f}")
        verdicts[rate] = pct

    print("\n  === mechanical pre-registered decision rule (PREREG.md) ===")
    print("  (measurement + rule application only -- no interpretation)")
    le4 = [verdicts[r] for r in main_rates if r <= 4 and r in verdicts]
    r6 = verdicts.get(6)
    if le4:
        max_abs_le4 = max(abs(v) for v in le4)
        print(f"  max |%diff| for rate<=4: {max_abs_le4:.2f}%  -> {'below' if max_abs_le4 < 3 else 'at/above'} 3% threshold")
    if r6 is not None:
        print(f"  |%diff| for rate=6: {abs(r6):.2f}%  -> {'below' if abs(r6) < 3 else 'at/above'} 3% threshold")
    print("  Note: CI computation (paired bootstrap/t) and final verdict selection deferred to result-analyst.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--tag", action="append", required=True)
    ap.add_argument("--nrep", type=int, default=5)
    args = ap.parse_args()
    for tag in args.tag:
        phase0_report(args.dir, tag)
        phase1_report(args.dir, tag, args.nrep)


if __name__ == "__main__":
    main()
