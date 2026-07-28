#!/usr/bin/env python3
"""Analysis for the 7-8B scale-up sweep.

Reports, per cell, both the client-side number and the two things the 2.7B
campaigns got wrong:
  - the REALIZED decode partition (Stage 0's 108 anchor was really 16 SM), and
  - the realized decode occupancy (run 865098's decode batch co-varied with D).

Then re-derives ITL at MATCHED realized SM by attributing every inter-token
interval to the partition in force at that wall-clock time (client
chunk_times_s and telemetry timestamp_monotonic_s share CLOCK_MONOTONIC).

Usage: s8_analyze.py [--mode deconf|coupled] [--dir <campaign dir>]
"""
import argparse
import bisect
import collections
import glob
import json
import os
import re
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
CELL_SM = {"d16": 16, "d24": 24, "d44": 44, "d92": 92, "np": 108}
GUARD = 3.0


def telemetry_timeline(path):
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
        pf.append(e["prefill_active_batch_size"])
    return ts, sm, bs, pf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="deconf")
    ap.add_argument("--dir", default=HERE)
    args = ap.parse_args()

    pat = re.compile(
        rf"s8_{args.mode}_(?P<arm>\w+?)_C(?P<ctx>\d+)_(?P<cell>d16|d24|d44|d92|np)_"
        rf"(?P<job>\d+)_telemetry\.jsonl$"
    )
    cells = []
    for fn in sorted(os.listdir(args.dir)):
        m = pat.search(fn)
        if m:
            cells.append((m.groupdict(), os.path.join(args.dir, fn)))
    if not cells:
        print(f"no {args.mode} telemetry under {args.dir}")
        return

    itl_by_sm = collections.defaultdict(list)     # (arm, ctx, realized_sm) -> itls
    no_t0 = set()
    rows = []
    for g, tpath in cells:
        arm, ctx, cell = g["arm"], int(g["ctx"]), g["cell"]
        want = CELL_SM[cell]
        ts, sm, bs, pf = telemetry_timeline(tpath)
        act = [i for i, b in enumerate(bs) if b > 0]
        hist = collections.Counter(sm[i] for i in act)
        n_act = sum(hist.values())
        frac = hist[want] / n_act if n_act else float("nan")
        cores = (sum(1 for i in act if pf[i] > 0) / n_act) if n_act else float("nan")
        batch = st.fmean(bs[i] for i in act) if n_act else float("nan")

        stem = tpath[: -len("_telemetry.jsonl")]
        p50s, occs, t0_by_rep = [], [], {}
        for res in sorted(glob.glob(os.path.join(args.dir, f"s8_{args.mode}_{arm}_C{ctx}_*result.txt"))):
            for line in open(res):
                if not line.startswith("{"):
                    continue
                try:
                    s = json.loads(line)
                except Exception:
                    continue
                tag = s.get("tag", "")
                if tag.startswith(f"{args.mode}_{arm}_C{ctx}_{cell}_r"):
                    p50s.append(s["itl_ms_p50"])
                    occs.append(s.get("occupancy_mean_in_window", float("nan")))
                    # chunk_times_s are perf_counter deltas from the client's
                    # t0; telemetry is absolute CLOCK_MONOTONIC. Runs before the
                    # 2026-07-27 client patch have no t0 and cannot be attributed.
                    if "t0_monotonic_s" in s:
                        t0_by_rep[int(tag.rsplit("_r", 1)[1])] = s["t0_monotonic_s"]

        for raw in sorted(glob.glob(f"{stem}_rep*_raw.jsonl")):
            rep = int(re.search(r"_rep(\d+)_raw", raw).group(1))
            t0 = t0_by_rep.get(rep)
            if t0 is None:
                no_t0.add((arm, ctx, cell))
                continue
            for line in open(raw):
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                tt = r.get("chunk_times_s")
                tt = [t + t0 for t in tt] if tt else (r.get("token_ts") or [])
                for a, b in zip(tt, tt[1:]):
                    i = bisect.bisect_right(ts, a) - 1
                    j = bisect.bisect_left(ts, b)
                    if i < 0 or j >= len(ts) or sm[i] != sm[j]:
                        continue
                    if (a - ts[i]) > GUARD or (ts[j] - b) > GUARD:
                        continue
                    # The NP cell reports 108 SM whether decode was time-sharing
                    # with prefill or running alone, so co-residency has to be a
                    # separate axis -- the isolated 108 anchor is the ISO bucket.
                    iso = "ISO" if (pf[i] == 0 and pf[j] == 0) else "CO"
                    itl_by_sm[(arm, ctx, sm[i], iso)].append(((b - a) * 1000.0, bs[i]))

        rows.append(dict(arm=arm, ctx=ctx, cell=cell, want=want, frac=frac,
                         cores=cores, batch=batch, n_act=n_act,
                         p50=st.fmean(p50s) if p50s else float("nan"),
                         sd=st.stdev(p50s) if len(p50s) > 1 else 0.0,
                         nrep=len(p50s),
                         occ=st.fmean(occs) if occs else float("nan"),
                         hist=dict(hist.most_common(3))))

    rows.sort(key=lambda r: (r["arm"], r["ctx"], r["want"]))
    print(f"=== {args.mode}: per-cell client ITL + REALIZED partition/occupancy ===")
    hdr = (f"{'arm':>4} {'ctx':>6} {'cell':>5} {'D':>4} | {'ITL p50 (n reps)':>22} | "
           f"{'realized':>9} {'co-res':>7} {'batch':>6} {'occ':>6} | realized hist")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['arm']:>4} {r['ctx']:>6} {r['cell']:>5} {r['want']:>4} | "
              f"{r['p50']:8.2f} +/- {r['sd']:5.2f} (n={r['nrep']}) | "
              f"{r['frac']*100:8.1f}% {r['cores']*100:6.1f}% {r['batch']:6.2f} {r['occ']:6.2f} | "
              f"{r['hist']}")
        if not (r["frac"] >= 0.80):
            print(f"     ^^ REALIZED-PIN FAIL: cell {r['cell']} did not run at D{r['want']}")

    arms = sorted({r["arm"] for r in rows})
    ctxs = sorted({r["ctx"] for r in rows})
    sms = sorted({s for (_, _, s, _) in itl_by_sm})
    if no_t0:
        print()
        print(f"NOTE: {len(no_t0)} cell(s) predate the client t0_monotonic_s patch "
              f"and cannot be partition-attributed: "
              f"{sorted(f'{a}/C{c}/{ce}' for a, c, ce in no_t0)[:6]}...")
    if not sms:
        print("\nNo attributable intervals -- skipping the matched-SM tables.")
        return
    table = {}
    for iso in ("CO", "ISO"):
        print()
        label = ("decode CO-RESIDENT with prefill" if iso == "CO"
                 else "decode ISOLATED (no prefill batch in flight)")
        print(f"=== ITL p50 at MATCHED realized decode-SM -- {label} ===")
        print(f"{'arm':>4} {'ctx':>6} | " + " ".join(f"{('SM'+str(s)):>21}" for s in sms))
        for arm in arms:
            for ctx in ctxs:
                cellsr = []
                for s in sms:
                    v = itl_by_sm.get((arm, ctx, s, iso), [])
                    vals = [x for x, _ in v]
                    table[(arm, ctx, s, iso)] = st.median(vals) if vals else None
                    # decode batch is printed alongside because ITL is strongly
                    # batch-dependent and run 865098 showed it co-varying with D.
                    cellsr.append(
                        f"{st.median(vals):7.2f} b{st.median([b for _, b in v]):.0f}(n={len(v):5d})"
                        if vals else " " * 21)
                print(f"{arm:>4} {ctx:>6} | " + " ".join(cellsr))

    print()
    print("=== SM sensitivity (matched exposure) and cross-arm ratios ===")
    for iso in ("CO", "ISO"):
        for arm in arms:
            for ctx in ctxs:
                lo = table.get((arm, ctx, min(sms), iso))
                hi = table.get((arm, ctx, 108, iso))
                if lo and hi:
                    print(f"  [{iso}] {arm:>4} C{ctx:<6} SM{min(sms)}/SM108 = "
                          f"{lo:7.2f}/{hi:6.2f} = {lo/hi:5.2f}x")
    for iso in ("CO", "ISO"):
        for ctx in ctxs:
            for s in sms:
                vals = {a: table.get((a, ctx, s, iso)) for a in arms}
                base = vals.get("M8")
                if base and all(vals.values()):
                    r = "  ".join(f"{a}/M8={vals[a]/base:.2f}" for a in arms if a != "M8")
                    print(f"  [{iso}] C{ctx:<6} SM{s:<4} M8={base:6.2f}ms   {r}")


if __name__ == "__main__":
    main()
