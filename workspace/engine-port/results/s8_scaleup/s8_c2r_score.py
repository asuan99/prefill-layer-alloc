#!/usr/bin/env python3
"""C2-R scorer (prereg rev2, PREREG_C2R_RULES_REV2_2026-08-15.md).

The interval definition is a VERBATIM re-use of s8_batch_matched.py:30-76 --
`_intervals()` below is that loop with nothing added but drop accounting and a
parameterised GUARD (for the C7 sensitivity scan).  Nothing about the estimand
is re-derived here (methodology gate #16):

    r_a = ITL_pooled_median(a, realized decode_sms=16,  decode_bs=b)
        / ITL_pooled_median(a, realized decode_sms=92, decode_bs=b)

Independent unit = one server boot.  Legacy (s8_deconf_*, jobs 865493/865533)
has one boot per (job, arm, cell) with reps inside it -- that is exactly the
n_indep=1 defect C2-R exists to fix.  C2-R (s8c2r_*) has one boot per
(job, arm, cell, block).

Subcommands
  poscontrol : reproduce the canonical batch-matched table (caveat C5 gate)
  c2r        : D1/D2/D3 for the new campaign + boot-cluster bootstrap CI
  snaphist   : decode_bs snapshot histograms (caveat C2 / C10)
"""
from __future__ import annotations

import argparse
import bisect
import collections
import glob
import json
import os
import re
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
GUARD = 3.0                      # s8_batch_matched.py:19
BOOT_BOOTSTRAP_SEED = 1          # project convention (analyze.py paired bootstrap)
BOOT_BOOTSTRAP_DRAWS = 10000

LEGACY_PAT = re.compile(r"s8_(?P<mode>\w+?)_(?P<arm>\w+?)_C(?P<ctx>\d+)_"
                        r"(?P<cell>d16|d24|d44|d92|np)_(?P<job>\d+)_telemetry\.jsonl$")
C2R_PAT = re.compile(r"s8c2r_(?P<arm>\w+?)_C(?P<ctx>\d+)_(?P<cell>d16|d24|d44|d92|np)_"
                     r"(?P<job>\d+)_blk(?P<blk>\d+)_telemetry\.jsonl$")


# --------------------------------------------------------------------------
# verbatim estimand
# --------------------------------------------------------------------------
def _snapshots(path):
    """s8_batch_matched.py:34-45 (verbatim filter + fields)."""
    ts, sm, bs, pf = [], [], [], []
    for line in open(path):
        try:
            e = json.loads(line)
        except Exception:
            continue
        if e.get("event") != "runtime_snapshot" or e.get("phase") != "benchmark":
            continue
        ts.append(e["timestamp_monotonic_s"])
        sm.append(e["decode_sms"] or 108)
        bs.append(e["decode_running_batch_size"])
        pf.append(e.get("prefill_active_batch_size"))
    return ts, sm, bs, pf


def _intervals(raw_path, t0, ts, sm, bs, guard=GUARD, keep_slack=False, pf=None):
    """s8_batch_matched.py:63-76 (verbatim), + drop accounting.

    `keep_slack=True` records max(a-ts[i], ts[j]-b) per surviving interval and
    defers the guard test.  The guard test is independent of, and strictly after,
    the state-match test in the canonical loop, so post-filtering on
    slack <= g is identical to re-running with GUARD=g (caveat C7 scan).
    """
    out = []
    drops = collections.Counter()
    if keep_slack:
        guard = float("inf")
    for line in open(raw_path):
        try:
            r = json.loads(line)
        except Exception:
            continue
        ct = r.get("chunk_times_s", [])
        for a, b in zip(ct, ct[1:]):
            a, b = a + t0, b + t0
            i = bisect.bisect_right(ts, a) - 1
            j = bisect.bisect_left(ts, b)
            if i < 0 or j >= len(ts):
                drops["edge"] += 1
                continue
            if sm[i] != sm[j] or bs[i] != bs[j]:
                drops["state_mismatch"] += 1
                continue
            slack = max(a - ts[i], ts[j] - b)
            if slack > guard:
                drops["guard"] += 1
                continue
            extra = ()
            if keep_slack:
                # (slack, prefill co-resident at either bracketing snapshot, i, j).
                # C10: a matched decode_bs does not imply a matched batch
                # COMPOSITION.  (i, j) exposes how many DISTINCT snapshot
                # brackets the thousands of pooled intervals actually rest on.
                co = ((pf[i] or 0), (pf[j] or 0)) if pf else (None, None)
                extra = (slack, co, i, j)
            out.append((sm[i], bs[i], (b - a) * 1000.0) + extra)
    return out, drops


# --------------------------------------------------------------------------
# file discovery -> boot records
# --------------------------------------------------------------------------
def _summary_map(result_glob, tag_prefix):
    """rep -> client summary dict (the line carrying t0_monotonic_s)."""
    out = {}
    for res in glob.glob(result_glob):
        for line in open(res):
            if not line.startswith("{"):
                continue
            try:
                s = json.loads(line)
            except Exception:
                continue
            tag = s.get("tag", "")
            if tag.startswith(tag_prefix) and "t0_monotonic_s" in s:
                out[int(tag.rsplit("_r", 1)[1])] = s
    return out


def _t0_map(result_glob, tag_prefix):
    return {k: v["t0_monotonic_s"] for k, v in _summary_map(result_glob, tag_prefix).items()}


def discover(kind, jobs=None, arms=None, ctx="1024"):
    """-> list of boot dicts (unopened; telemetry parsed lazily by score())."""
    boots = []
    for fn in sorted(os.listdir(HERE)):
        if kind == "legacy":
            m = LEGACY_PAT.search(fn)
            if not m:
                continue
            g = m.groupdict()
            if g["mode"] != "deconf" or g["ctx"] != ctx:
                continue
            blk = None
            res_glob = os.path.join(HERE, f"s8_deconf_{g['arm']}_C{ctx}_{g['job']}_result.txt")
            tag_prefix = f"deconf_{g['arm']}_C{ctx}_{g['cell']}_r"
            stem = os.path.join(HERE, fn[: -len("_telemetry.jsonl")])
        else:
            m = C2R_PAT.search(fn)
            if not m:
                continue
            g = m.groupdict()
            if g["ctx"] != ctx:
                continue
            blk = int(g["blk"])
            res_glob = os.path.join(HERE, f"s8c2r_{g['arm']}_C{ctx}_{g['job']}_result.txt")
            tag_prefix = f"c2r_{g['arm']}_C{ctx}_{g['cell']}_blk{blk}_r"
            stem = os.path.join(HERE, fn[: -len("_telemetry.jsonl")])
        if jobs and g["job"] not in jobs:
            continue
        if arms and g["arm"] not in arms:
            continue
        boots.append(dict(kind=kind, arm=g["arm"], cell=g["cell"], job=g["job"], blk=blk,
                          ctx=ctx, telemetry=os.path.join(HERE, fn), stem=stem,
                          res_glob=res_glob, tag_prefix=tag_prefix,
                          boot_id=f"{g['arm']}/{g['cell']}/{g['job']}" +
                                  (f"/blk{blk}" if blk else "")))
    return boots


def score(boots, guard=GUARD, keep_slack=False):
    """Attach matched intervals + snapshot stats to every boot record."""
    for bt in boots:
        ts, sm, bs, pf = _snapshots(bt["telemetry"])
        bt["n_snapshots"] = len(ts)
        bt["snapshot_bs_hist"] = collections.Counter(bs)
        bt["snapshot_sm_hist"] = collections.Counter(sm)
        bt["snapshot_active_sm_hist"] = collections.Counter(
            s for s, b in zip(sm, bs) if b and b > 0)
        bt["intervals"] = []
        bt["drops"] = collections.Counter()
        bt["reps_used"] = []
        if not ts:
            continue
        summ = _summary_map(bt["res_glob"], bt["tag_prefix"])
        t0s = {k: v["t0_monotonic_s"] for k, v in summ.items()}
        bt["n_t0"] = len(t0s)
        bt["client_summary"] = {k: {kk: v.get(kk) for kk in
                                    ("itl_ms_p50", "itl_ms_p95", "itl_ms_p99", "itl_ms_mean",
                                     "itl_samples", "occupancy_mean_in_window",
                                     "requests_completed", "errors", "keepalive_done",
                                     "keepalive_errors")} for k, v in summ.items()}
        for raw in sorted(glob.glob(f"{bt['stem']}_rep*_raw.jsonl")):
            rep = int(re.search(r"_rep(\d+)_raw", raw).group(1))
            t0 = t0s.get(rep)
            if t0 is None:
                bt["drops"]["no_t0_rep"] += 1
                continue
            iv, dr = _intervals(raw, t0, ts, sm, bs, guard, keep_slack, pf)
            bt["intervals"] += iv
            bt["drops"] += dr
            bt["reps_used"].append(rep)
    return boots


def cell_intervals(boots, arm, sm_want, bs_want, max_slack=None):
    """-> {boot_id: [itl_ms, ...]} for one (arm, realized SM, decode batch) cell."""
    out = collections.OrderedDict()
    for bt in boots:
        if bt["arm"] != arm:
            continue
        v = [t[2] for t in bt["intervals"]
             if t[0] == sm_want and t[1] == bs_want
             and (max_slack is None or t[3] <= max_slack)]
        if v:
            out[bt["boot_id"]] = v
    return out


def pooled_median(per_boot):
    allv = [x for v in per_boot.values() for x in v]
    return (st.median(allv) if allv else None), len(allv)


def pooled_percentiles(per_boot, qs=(5, 25, 50, 75, 95, 99)):
    """Nearest-rank percentiles of the pooled accepted intervals (reporting only --
    the estimand is the median).  Shows how thin a slice the median describes."""
    allv = sorted(x for v in per_boot.values() for x in v)
    if not allv:
        return {}
    return {f"p{q}": allv[min(len(allv) - 1, int(q / 100 * len(allv)))] for q in qs}


# --------------------------------------------------------------------------
# boot-cluster bootstrap
# --------------------------------------------------------------------------
def cluster_bootstrap_ratio(num_boots, den_boots, draws=BOOT_BOOTSTRAP_DRAWS,
                            seed=BOOT_BOOTSTRAP_SEED):
    """Cluster (= boot) bootstrap of pooled-median ratio.

    The two legs are separate boots, so they are resampled independently.
    """
    import random
    rng = random.Random(seed)
    nk = list(num_boots.values())
    dk = list(den_boots.values())
    if not nk or not dk:
        return None
    reps = []
    for _ in range(draws):
        a = [x for _ in nk for x in nk[rng.randrange(len(nk))]]
        b = [x for _ in dk for x in dk[rng.randrange(len(dk))]]
        if not a or not b:
            continue
        reps.append(st.median(a) / st.median(b))
    reps.sort()
    lo = reps[int(0.025 * len(reps))]
    hi = reps[int(0.975 * len(reps)) - 1]
    return dict(draws=len(reps), seed=seed, ci95_low=lo, ci95_high=hi,
                boot_sd=st.pstdev(reps), n_clusters_num=len(nk), n_clusters_den=len(dk))


# --------------------------------------------------------------------------
# subcommands
# --------------------------------------------------------------------------
def cmd_poscontrol(args):
    """Caveat C5: reproduce T8 2.388 / Hs8 2.687 (b=16, SM16/SM92)."""
    out = {}
    for label, jobs in (("pooled_865493+865533", None), ("865493_only", {"865493"}),
                        ("865533_only", {"865533"})):
        boots = score(discover("legacy", jobs=jobs))
        rows = {}
        for arm in sorted({b["arm"] for b in boots}):
            for batch in (16,):
                n16 = cell_intervals(boots, arm, 16, batch)
                n92 = cell_intervals(boots, arm, 92, batch)
                m16, c16 = pooled_median(n16)
                m92, c92 = pooled_median(n92)
                rows[arm] = dict(
                    itl_p50_sm16=m16, n_sm16=c16, n_boots_sm16=len(n16),
                    itl_p50_sm92=m92, n_sm92=c92, n_boots_sm92=len(n92),
                    ratio=(m16 / m92 if (m16 and m92 and c16 >= 50 and c92 >= 50) else None))
        out[label] = rows
        print(f"\n=== {label} (decode batch = 16, n>=50 enforced) ===")
        print(f"{'arm':>5} | {'ITL p50 SM16':>13} {'n':>7} {'boots':>5} | "
              f"{'ITL p50 SM92':>13} {'n':>7} {'boots':>5} | {'SM16/SM92':>9}")
        for arm, r in rows.items():
            f = lambda v, d="": (f"{v:.4f}" if isinstance(v, float) else d)
            print(f"{arm:>5} | {f(r['itl_p50_sm16']):>13} {r['n_sm16']:>7} "
                  f"{r['n_boots_sm16']:>5} | {f(r['itl_p50_sm92']):>13} {r['n_sm92']:>7} "
                  f"{r['n_boots_sm92']:>5} | {f(r['ratio']):>9}")
    targets = {"T8": 2.388, "Hs8": 2.687}
    print("\n--- C5 verdict (target = canonical s8_batch_matched.py table, b=16) ---")
    verdict = {}
    for arm, tgt in targets.items():
        for label in out:
            got = out[label].get(arm, {}).get("ratio")
            if got is None:
                continue
            ok = abs(round(got, 3) - tgt) < 5e-4
            verdict[f"{arm}@{label}"] = dict(target=tgt, got=got, match_3dp=ok)
            print(f"{arm:>5} {label:>22}: {got:.4f} (target {tgt}) "
                  f"-> {'MATCH@3dp' if ok else 'differs'}")
    out["_verdict"] = verdict
    if args.output:
        json.dump(out, open(args.output, "w"), indent=1, default=str)
    return out


def _bracket_stats(boots, arm, cell, smw, batch):
    """How many DISTINCT (i, j) snapshot brackets carry a cell's pooled intervals,
    and what share of those intervals is prefill co-resident (C10)."""
    out = {}
    for bt in boots:
        if bt["arm"] != arm or bt["cell"] != cell:
            continue
        sel = [t for t in bt["intervals"] if t[0] == smw and t[1] == batch]
        if not sel or len(sel[0]) < 7:
            continue
        out[bt["boot_id"]] = dict(
            n_intervals=len(sel),
            n_distinct_brackets=len({(t[5], t[6]) for t in sel}),
            n_distinct_snapshots=len({t[5] for t in sel} | {t[6] for t in sel}),
            co_resident_frac_either=sum(1 for t in sel if max(t[4]) > 0) / len(sel),
            co_resident_frac_both=sum(1 for t in sel if min(t[4]) > 0) / len(sel),
            prefill_batch_mean=st.fmean([(t[4][0] + t[4][1]) / 2 for t in sel]))
    return out


def cmd_c2r(args):
    jobs = set(args.jobs.split(",")) if args.jobs else None
    # one extended pass (guard deferred), then filter at args.guard -- identical to
    # running the canonical loop with GUARD=args.guard (see _intervals docstring).
    boots = score(discover("c2r", jobs=jobs), keep_slack=True)
    for bt in boots:
        bt["intervals"] = [t for t in bt["intervals"] if t[3] <= args.guard]
    batch = args.batch
    res = {"guard_s": args.guard, "batch": batch, "cells": {}, "ratios": {},
           "boots": [], "bootstrap": {}}
    for bt in boots:
        res["boots"].append(dict(
            boot_id=bt["boot_id"], arm=bt["arm"], cell=bt["cell"], job=bt["job"],
            blk=bt["blk"], n_snapshots=bt["n_snapshots"], reps_used=bt["reps_used"],
            client_summary=bt.get("client_summary"),
            n_intervals_total=len(bt["intervals"]),
            n_intervals_cell=sum(1 for t in bt["intervals"]
                                 if t[0] == (16 if bt["cell"] == "d16" else 92)
                                 and t[1] == batch),
            drops=dict(bt["drops"]),
            snapshot_bs_hist={str(k): v for k, v in sorted(bt["snapshot_bs_hist"].items())},
            snapshot_active_sm_hist={str(k): v for k, v in
                                     sorted(bt["snapshot_active_sm_hist"].items())}))
    for arm in sorted({b["arm"] for b in boots}):
        legs = {}
        for cell, smw in (("d16", 16), ("d92", 92)):
            sub = [b for b in boots if b["cell"] == cell]
            per = cell_intervals(sub, arm, smw, batch)
            med, n = pooled_median(per)
            legs[cell] = dict(
                per_boot_n={k: len(v) for k, v in per.items()},
                per_boot_median={k: st.median(v) for k, v in per.items()},
                per_boot_brackets=_bracket_stats(sub, arm, cell, smw, batch),
                pooled_percentiles_ms=pooled_percentiles(per),
                n_boots_accepted=len(per), n_pooled=n, pooled_median_ms=med)
            res["cells"][f"{arm}/{cell}/SM{smw}/b{batch}"] = legs[cell]
        m16, m92 = legs["d16"]["pooled_median_ms"], legs["d92"]["pooled_median_ms"]
        r = m16 / m92 if (m16 and m92) else None
        res["ratios"][arm] = dict(
            r=r, n_pooled_sm16=legs["d16"]["n_pooled"], n_pooled_sm92=legs["d92"]["n_pooled"],
            n_boots_sm16=legs["d16"]["n_boots_accepted"],
            n_boots_sm92=legs["d92"]["n_boots_accepted"],
            d1_pass=(legs["d16"]["n_boots_accepted"] >= 4 and
                     legs["d92"]["n_boots_accepted"] >= 4 and
                     legs["d16"]["n_pooled"] >= 50 and legs["d92"]["n_pooled"] >= 50))
        if r:
            num = cell_intervals([b for b in boots if b["cell"] == "d16"], arm, 16, batch)
            den = cell_intervals([b for b in boots if b["cell"] == "d92"], arm, 92, batch)
            res["bootstrap"][arm] = cluster_bootstrap_ratio(num, den, args.draws, args.seed)
        # context only: every OTHER batch with both legs n>=50 (the estimand is b=16)
        grid = {}
        for bq in sorted({t[1] for b in boots if b["arm"] == arm for t in b["intervals"]}):
            n16 = cell_intervals([b for b in boots if b["cell"] == "d16"], arm, 16, bq)
            n92 = cell_intervals([b for b in boots if b["cell"] == "d92"], arm, 92, bq)
            m16, c16 = pooled_median(n16)
            m92, c92 = pooled_median(n92)
            if c16 >= 50 and c92 >= 50:
                grid[bq] = dict(r=m16 / m92, n16=c16, n92=c92,
                                boots16=len(n16), boots92=len(n92))
        res.setdefault("batch_grid", {})[arm] = grid
    if args.output:
        json.dump(res, open(args.output, "w"), indent=1, default=str)
    print(json.dumps({k: res[k] for k in ("guard_s", "batch", "ratios", "bootstrap")},
                     indent=1, default=str))
    return res


def cmd_guardscan(args):
    """Caveat C7: how much does the answer move with the 3.0s guard?

    One pass with the guard deferred (slack recorded), then post-filtered --
    identical to re-running the canonical loop at each GUARD value.
    """
    jobs = set(args.jobs.split(",")) if args.jobs else None
    arms = set(args.arms.split(",")) if args.arms else None
    boots = score(discover(args.kind, jobs=jobs, arms=arms), keep_slack=True)
    grid = [float(g) for g in args.grid.split(",")]
    res = {"grid_s": grid, "batch": args.batch, "arms": {}}
    for arm in sorted({b["arm"] for b in boots}):
        rows = []
        for g in grid:
            num = cell_intervals([b for b in boots if b["cell"] == "d16"], arm, 16,
                                 args.batch, max_slack=g)
            den = cell_intervals([b for b in boots if b["cell"] == "d92"], arm, 92,
                                 args.batch, max_slack=g)
            m16, n16 = pooled_median(num)
            m92, n92 = pooled_median(den)
            rows.append(dict(guard_s=g, itl_p50_sm16=m16, n_sm16=n16, boots_sm16=len(num),
                             itl_p50_sm92=m92, n_sm92=n92, boots_sm92=len(den),
                             r=(m16 / m92 if (m16 and m92) else None)))
        res["arms"][arm] = rows
        print(f"\n=== {arm} guard scan (b={args.batch}) ===")
        print(f"{'guard_s':>7} {'p50_SM16':>9} {'n16':>8} {'bt16':>4} "
              f"{'p50_SM92':>9} {'n92':>8} {'bt92':>4} {'r':>8}")
        for r in rows:
            f = lambda v: (f"{v:.3f}" if isinstance(v, float) else "-")
            print(f"{r['guard_s']:>7.2f} {f(r['itl_p50_sm16']):>9} {r['n_sm16']:>8} "
                  f"{r['boots_sm16']:>4} {f(r['itl_p50_sm92']):>9} {r['n_sm92']:>8} "
                  f"{r['boots_sm92']:>4} {f(r['r']):>8}")
    # slack distribution (the C7 cliff diagnostic itself)
    for arm in res["arms"]:
        for cell, smw in (("d16", 16), ("d92", 92)):
            sl = [t[3] for b in boots if b["arm"] == arm and b["cell"] == cell
                  for t in b["intervals"] if t[0] == smw and t[1] == args.batch]
            if sl:
                sl.sort()
                res.setdefault("slack_s", {})[f"{arm}/{cell}"] = dict(
                    n=len(sl), p50=sl[len(sl) // 2], p95=sl[int(0.95 * len(sl))], max=sl[-1])
    print("\nslack (s) of accepted intervals:", json.dumps(res.get("slack_s", {}), indent=1))
    if args.output:
        json.dump(res, open(args.output, "w"), indent=1, default=str)
    return res


def cmd_sens(args):
    """SECONDARY diagnostics only (gate #19): the pre-registered acceptance rule is
    H2+H3+H7, and no boot may be dropped post hoc.  Reported: leave-one-boot-out
    jackknife per leg, and an explicit named-exclusion probe."""
    jobs = set(args.jobs.split(",")) if args.jobs else None
    boots = score(discover("c2r", jobs=jobs), keep_slack=True)
    for bt in boots:
        bt["intervals"] = [t for t in bt["intervals"] if t[3] <= args.guard]
    out = {}
    for arm in sorted({b["arm"] for b in boots}):
        num = cell_intervals([b for b in boots if b["cell"] == "d16"], arm, 16, args.batch)
        den = cell_intervals([b for b in boots if b["cell"] == "d92"], arm, 92, args.batch)
        base = pooled_median(num)[0] / pooled_median(den)[0]
        jack = {}
        for leg, d in (("d16", num), ("d92", den)):
            for k in list(d):
                sub = {kk: v for kk, v in d.items() if kk != k}
                r = (pooled_median(sub)[0] / pooled_median(den)[0] if leg == "d16"
                     else pooled_median(num)[0] / pooled_median(sub)[0])
                jack[f"drop {k}"] = dict(r=r, delta_pct=100 * (r - base) / base)
        out[arm] = dict(r_all_boots=base, jackknife=jack)
        if args.exclude:
            ex = set(args.exclude.split(","))
            n2 = {k: v for k, v in num.items() if not any(e in k for e in ex)}
            d2 = {k: v for k, v in den.items() if not any(e in k for e in ex)}
            if n2 and d2:
                r = pooled_median(n2)[0] / pooled_median(d2)[0]
                out[arm]["named_exclusion"] = dict(
                    excluded=sorted(ex), r=r, delta_pct=100 * (r - base) / base,
                    n_boots_d16=len(n2), n_boots_d92=len(d2),
                    bootstrap=cluster_bootstrap_ratio(n2, d2, args.draws, args.seed))
    print(json.dumps(out, indent=1, default=str))
    if args.output:
        json.dump(out, open(args.output, "w"), indent=1, default=str)
    return out


def cmd_snaphist(args):
    """Snapshot-level decode_bs histograms (caveat C2/C10) -- no t0 needed."""
    kind = args.kind
    jobs = set(args.jobs.split(",")) if args.jobs else None
    arms = set(args.arms.split(",")) if args.arms else None
    out = {}
    for bt in discover(kind, jobs=jobs, arms=arms):
        ts, sm, bs, pf = _snapshots(bt["telemetry"])
        h = collections.Counter((s, b) for s, b in zip(sm, bs))
        out[bt["boot_id"]] = {f"sm{s}_bs{b}": n for (s, b), n in sorted(h.items())}
        out[bt["boot_id"]]["_n_snapshots"] = len(ts)
    if args.output:
        json.dump(out, open(args.output, "w"), indent=1)
    print(json.dumps(out, indent=1)[:4000])
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("poscontrol"); p.add_argument("--output"); p.set_defaults(fn=cmd_poscontrol)
    p = sub.add_parser("c2r")
    p.add_argument("--jobs"); p.add_argument("--output")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--guard", type=float, default=GUARD)
    p.add_argument("--draws", type=int, default=BOOT_BOOTSTRAP_DRAWS)
    p.add_argument("--seed", type=int, default=BOOT_BOOTSTRAP_SEED)
    p.set_defaults(fn=cmd_c2r)
    p = sub.add_parser("guardscan")
    p.add_argument("--jobs"); p.add_argument("--output"); p.add_argument("--arms")
    p.add_argument("--kind", default="c2r", choices=["legacy", "c2r"])
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--grid", default="0.25,0.5,0.75,1.0,1.5,2.0,2.5,3.0")
    p.set_defaults(fn=cmd_guardscan)
    p = sub.add_parser("sens")
    p.add_argument("--jobs"); p.add_argument("--output"); p.add_argument("--exclude")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--guard", type=float, default=GUARD)
    p.add_argument("--draws", type=int, default=BOOT_BOOTSTRAP_DRAWS)
    p.add_argument("--seed", type=int, default=BOOT_BOOTSTRAP_SEED)
    p.set_defaults(fn=cmd_sens)
    p = sub.add_parser("snaphist")
    p.add_argument("--kind", default="legacy", choices=["legacy", "c2r"])
    p.add_argument("--jobs"); p.add_argument("--arms"); p.add_argument("--output")
    p.set_defaults(fn=cmd_snaphist)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
