#!/usr/bin/env python3
"""AUDIT (2026-08-15): job composition of the C2 headline cells.

`s8_batch_matched.py` pools every deconf/ctx1024 telemetry file that matches its
filename pattern into one `(arm, realized_sm, decode_batch)` bucket.  Both v2
(job 865493) and v3 (job 865533) runs exist for all four arms and all five
cells, so the pooled buckets can silently be composed of a single job -- and
job 865533's keepalive prompt was 1793 tokens against a 1792-token context
limit, i.e. its prefill co-residency (the condition under which the green
context partition is realized at all) is degraded.

This script re-runs the *identical* interval attribution as
`s8_batch_matched.py` (same GUARD, same both-endpoints-match rule, same
`decode_sms or 108` fallback) but keys the buckets by
`(arm, job, cell, realized_sm, decode_batch)` and also records, per interval,
the run file (rep) it came from and whether a prefill batch was in flight at
both endpoints.  Nothing about the estimator changes; only the reporting axis.

Outputs JSON to stdout (or --out) with:
  * per (arm, sm, batch) pooled median + n, and the per-(job, cell, rep) split
  * CO_RESIDENT_frac per (arm, job, cell) using realized_pin_check.py's
    definition (share of decode-active benchmark snapshots with prefill in
    flight), plus an interval-level co-residency share for the used intervals
  * rep-clustered bootstrap CI (seed=1, 10000 resamples) for the SM16/SM92
    ratio, per job and pooled

Usage:  audit_c2_job_composition.py [--out FILE]
"""
import argparse
import bisect
import collections
import glob
import json
import os
import random
import re
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
GUARD = float(os.environ.get("PDMUX_AUDIT_GUARD_S", "3.0"))  # 3.0 = s8_batch_matched.py default
# Telemetry snapshots arrive in ~2 ms bursts that cover only ~17 s of a ~390 s
# benchmark span, with 0.3-0.8 s (max ~8 s) gaps between bursts.  GUARD=3.0 s
# therefore admits intervals whose endpoints straddle an unobserved gap; setting
# PDMUX_AUDIT_GUARD_S=0.05 restricts the sample to intervals fully inside a
# dense burst and is used as a robustness check on the headline ratios.
MIN_N = 50   # identical to s8_batch_matched.py's n>=50 filter
PAT = re.compile(r"s8_(?P<mode>\w+?)_(?P<arm>\w+?)_C(?P<ctx>\d+)_(?P<cell>d16|d24|d44|d92|np)_"
                 r"(?P<job>\d+)_telemetry\.jsonl$")


def collect():
    """Return intervals keyed by (arm, sm, batch) -> list of records."""
    by = collections.defaultdict(list)
    snap_stats = {}   # (arm, job, cell) -> dict
    for fn in sorted(os.listdir(HERE)):
        m = PAT.search(fn)
        if not m:
            continue
        g = m.groupdict()
        if g["mode"] != "deconf" or g["ctx"] != "1024":
            continue
        arm, job, cell, ctx = g["arm"], g["job"], g["cell"], g["ctx"]
        ts, sm, bs, pf = [], [], [], []
        for line in open(os.path.join(HERE, fn)):
            try:
                e = json.loads(line)
            except Exception:
                continue
            if e.get("event") != "runtime_snapshot" or e.get("phase") != "benchmark":
                continue
            ts.append(e["timestamp_monotonic_s"])
            sm.append(e["decode_sms"] or 108)
            bs.append(e["decode_running_batch_size"])
            pf.append(e.get("prefill_active_batch_size", 0) or 0)
        if not ts:
            continue
        # realized_pin_check.py definition, recomputed here so the audit is self-contained
        dec_active = [i for i in range(len(ts)) if bs[i] > 0]
        snap_stats[(arm, job, cell)] = {
            "n_benchmark_snapshots": len(ts),
            "n_decode_active": len(dec_active),
            "co_resident_frac": (sum(1 for i in dec_active if pf[i] > 0) / len(dec_active))
            if dec_active else None,
            "realized_sm_hist": dict(collections.Counter(sm[i] for i in dec_active)),
            "decode_bs_median": st.median([bs[i] for i in dec_active]) if dec_active else None,
        }
        t0s = {}
        for res in glob.glob(os.path.join(HERE, f"s8_deconf_{arm}_C{ctx}_{job}_result.txt")):
            for line in open(res):
                if not line.startswith("{"):
                    continue
                try:
                    s = json.loads(line)
                except Exception:
                    continue
                tag = s.get("tag", "")
                if tag.startswith(f"deconf_{arm}_C{ctx}_{cell}_r") and "t0_monotonic_s" in s:
                    t0s[int(tag.rsplit("_r", 1)[1])] = s["t0_monotonic_s"]
        stem = os.path.join(HERE, fn[: -len("_telemetry.jsonl")])
        for raw in sorted(glob.glob(f"{stem}_rep*_raw.jsonl")):
            rep = int(re.search(r"_rep(\d+)_raw", raw).group(1))
            t0 = t0s.get(rep)
            if t0 is None:
                continue
            for line in open(raw):
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                ct = r.get("chunk_times_s", [])
                for idx, (a, b) in enumerate(zip(ct, ct[1:])):
                    a, b = a + t0, b + t0
                    i = bisect.bisect_right(ts, a) - 1
                    j = bisect.bisect_left(ts, b)
                    if i < 0 or j >= len(ts) or sm[i] != sm[j] or bs[i] != bs[j]:
                        continue
                    if (a - ts[i]) > GUARD or (ts[j] - b) > GUARD:
                        continue
                    by[(arm, sm[i], bs[i])].append({
                        "itl_ms": (b - a) * 1000.0,
                        "job": job, "cell": cell, "rep": rep,
                        "tok_idx": idx,        # decode position of the interval
                        # how far the nearest bracketing snapshot is from the
                        # interval endpoints; the (SM, batch) label is only
                        # observed at those snapshots, never inside the interval
                        "slack_s": max(a - ts[i], ts[j] - b),
                        "cores": bool(pf[i] > 0 and pf[j] > 0),
                    })
    return by, snap_stats


def cluster_bootstrap_ratio(lo_recs, hi_recs, n_boot=10000, seed=1):
    """Ratio of medians with rep-clustered resampling (clusters = (job,cell,rep)).

    Clustering matters: intervals inside one 60 s rep are heavily autocorrelated,
    so an interval-level bootstrap would report a CI ~sqrt(n_interval) too tight.
    With only 2-3 surviving clusters per leg the CI is itself coarse -- that is a
    finding about the design, not a defect of the estimator, and the cluster
    count is reported alongside every interval.
    """
    import numpy as np
    if not lo_recs or not hi_recs:
        return None

    def clusters(recs):
        d = collections.defaultdict(list)
        for r in recs:
            d[(r["job"], r["cell"], r["rep"])].append(r["itl_ms"])
        return [np.asarray(v) for v in d.values()]

    lo_c, hi_c = clusters(lo_recs), clusters(hi_recs)
    rng = np.random.default_rng(seed)
    point = (float(np.median([r["itl_ms"] for r in lo_recs]))
             / float(np.median([r["itl_ms"] for r in hi_recs])))
    draws = np.empty(n_boot)
    for k in range(n_boot):
        lo = np.concatenate([lo_c[i] for i in rng.integers(0, len(lo_c), len(lo_c))])
        hi = np.concatenate([hi_c[i] for i in rng.integers(0, len(hi_c), len(hi_c))])
        draws[k] = np.median(lo) / np.median(hi)
    draws.sort()
    return {
        "point": point,
        "ci95_low": float(draws[int(0.025 * n_boot)]),
        "ci95_high": float(draws[int(0.975 * n_boot)]),
        "n_clusters_lo": len(lo_c), "n_clusters_hi": len(hi_c),
    }


def split(recs):
    d = collections.defaultdict(list)
    for r in recs:
        d[(r["job"], r["cell"])].append(r)
    out = {}
    for (job, cell), v in sorted(d.items()):
        reps = collections.Counter(r["rep"] for r in v)
        out[f"{job}/{cell}"] = {
            "n": len(v),
            "median_ms": round(st.median([r["itl_ms"] for r in v]), 3),
            "reps": {str(k): reps[k] for k in sorted(reps)},
            "interval_coresident_frac": round(sum(r["cores"] for r in v) / len(v), 3),
            "tok_idx_median": st.median([r["tok_idx"] for r in v]),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--pickle", default=None,
                    help="also dump the raw interval records here (re-analysis without re-parsing)")
    args = ap.parse_args()

    by, snap_stats = collect()
    if args.pickle:
        import pickle
        with open(args.pickle, "wb") as fh:
            pickle.dump({"by": dict(by), "snap": snap_stats}, fh)
    arms = sorted({a for a, _, _ in by})

    report = {"guard_s": GUARD, "min_n": MIN_N, "snapshot_stats": {}, "cells": {}, "ratios": {}}
    for k, v in sorted(snap_stats.items()):
        report["snapshot_stats"]["/".join(k)] = v

    for arm in arms:
        for batch in sorted({b for a, _, b in by if a == arm}):
            for smv in sorted({s for a, s, b in by if a == arm and b == batch}):
                recs = by[(arm, smv, batch)]
                report["cells"][f"{arm}/SM{smv}/b{batch}"] = {
                    "n": len(recs),
                    "median_ms": round(st.median([r["itl_ms"] for r in recs]), 3),
                    "passes_n50": len(recs) >= MIN_N,
                    "by_job_cell": split(recs),
                }

    # SM16/SM92 ratios at every batch where both legs pass the n>=50 filter,
    # pooled and restricted to each single job.
    for arm in arms:
        for batch in sorted({b for a, _, b in by if a == arm}):
            lo = by.get((arm, 16, batch), [])
            hi = by.get((arm, 92, batch), [])
            entry = {}
            if len(lo) >= MIN_N and len(hi) >= MIN_N:
                entry["pooled"] = cluster_bootstrap_ratio(lo, hi, args.boot)
                entry["pooled"]["n_lo"] = len(lo)
                entry["pooled"]["n_hi"] = len(hi)
            for job in sorted({r["job"] for r in lo} | {r["job"] for r in hi}):
                jlo = [r for r in lo if r["job"] == job]
                jhi = [r for r in hi if r["job"] == job]
                if len(jlo) >= MIN_N and len(jhi) >= MIN_N:
                    e = cluster_bootstrap_ratio(jlo, jhi, args.boot)
                    e["n_lo"], e["n_hi"] = len(jlo), len(jhi)
                    entry[f"job_{job}"] = e
            if entry:
                report["ratios"][f"{arm}/b{batch}"] = entry

    # Cross-job replicate test: wherever BOTH jobs supply >=MIN_N intervals to the
    # SAME (arm, realized SM, decode batch) state, how far apart are their medians?
    # This is the only direct handle on "does the keepalive collapse move the
    # conditional ITL, or only the state occupancy?"  It is not an identity: the
    # two jobs are separate server launches on different nodes/times.
    report["cross_job_matched"] = {}
    for (arm, smv, batch), recs in sorted(by.items()):
        a493 = [r["itl_ms"] for r in recs if r["job"] == "865493"]
        a533 = [r["itl_ms"] for r in recs if r["job"] == "865533"]
        if len(a493) >= MIN_N and len(a533) >= MIN_N:
            m1, m2 = st.median(a493), st.median(a533)
            report["cross_job_matched"][f"{arm}/SM{smv}/b{batch}"] = {
                "n_865493": len(a493), "n_865533": len(a533),
                "median_865493_ms": round(m1, 3), "median_865533_ms": round(m2, 3),
                "pct_diff_533_vs_493": round((m2 - m1) / m1 * 100.0, 2),
            }

    # Attribution-guard scan.  The (SM, batch) label is read from the snapshots
    # bracketing the interval, which sit up to GUARD seconds away; telemetry
    # arrives in ~2 ms bursts covering only ~4-5% of wall time, so almost every
    # accepted interval has several hundred ms of unobserved state on one side.
    # If the headline ratio moved as the guard tightens, the label would be
    # doing the work.  It does not (<=0.6% down to 0.25-0.5 s).
    report["guard_scan"] = {}
    for arm in arms:
        for batch in sorted({b for a, _, b in by if a == arm}):
            lo, hi = by.get((arm, 16, batch), []), by.get((arm, 92, batch), [])
            if len(lo) < MIN_N or len(hi) < MIN_N:
                continue
            row = {"slack_p50_lo_s": round(st.median([r["slack_s"] for r in lo]), 3),
                   "slack_p50_hi_s": round(st.median([r["slack_s"] for r in hi]), 3)}
            for g in (0.25, 0.5, 1.0, 2.0, 3.0):
                l = [r["itl_ms"] for r in lo if r["slack_s"] <= g]
                h = [r["itl_ms"] for r in hi if r["slack_s"] <= g]
                row[f"guard_{g}"] = ({"n_lo": len(l), "n_hi": len(h),
                                      "ratio": round(st.median(l) / st.median(h), 4)}
                                     if len(l) >= MIN_N and len(h) >= MIN_N
                                     else {"n_lo": len(l), "n_hi": len(h), "ratio": None})
            report["guard_scan"][f"{arm}/b{batch}"] = row

    text = json.dumps(report, indent=1, sort_keys=True)
    if args.out:
        open(args.out, "w").write(text + "\n")
        print(f"wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
