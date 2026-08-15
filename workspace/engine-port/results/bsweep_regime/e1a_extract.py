#!/usr/bin/env python3
"""E-1a rep-level extraction of ITL conditioned on (realized SM, realized B).

GPU 0. Reads only existing C2 artifacts (jobs 865493 / 865533, ctx1024 deconf).

Producer invariance (methodology gate #9): the canonical producer
``s8_scaleup/s8_batch_matched.py`` is **not modified**. Its two decision
constants (``PAT``, ``GUARD``) and its interval-attribution rule are imported /
replicated verbatim here; the only change is the **key**: this script keys every
matched inter-token interval by ``(arm, realized_sm, realized_bs, job, cell,
rep)`` instead of ``(arm, realized_sm, realized_bs)``.  Pooling this script's
per-rep buckets must reproduce ``s8_batch_matched.py`` exactly -- that identity
is asserted in ``e1a_analyze.py`` (self-check SC1).

★ realized conditioning carries the load: the C2 REALIZED_PIN check reports
``D=16: FAIL ... frac=0.317`` (68% of decode-active time ran at 108 SM, i.e. the
requested partition was NOT realized). Every number produced here is
conditioned on ``sm[i] == sm[j] == <realized SM>`` AND ``bs[i] == bs[j] ==
<realized B>`` at both endpoints of the interval, exactly as the producer does.

Output: e1a_repcells_2026-08-14.json
"""
import bisect
import collections
import glob
import importlib.util
import json
import os
import re
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
S8 = os.path.abspath(os.path.join(HERE, "..", "s8_scaleup"))
OUT = os.path.join(HERE, "e1a_repcells_2026-08-14.json")

# --- import the canonical producer (read-only) to inherit its constants -------
_spec = importlib.util.spec_from_file_location(
    "s8_batch_matched", os.path.join(S8, "s8_batch_matched.py"))
_s8bm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_s8bm)          # module body only defines PAT/GUARD/main
PAT, GUARD = _s8bm.PAT, _s8bm.GUARD


def percentile(v, q):
    """Nearest-rank-free linear interpolation percentile (statistics-free helper).

    Matches numpy's default 'linear' method so p95 is comparable to the
    project's other percentile reports.
    """
    if not v:
        return None
    s = sorted(v)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def main():
    # (arm, sm, bs, job, cell, rep) -> [itl_ms]
    by_rep = collections.defaultdict(list)
    # (arm, cell, job, sm) -> Counter(bs)  over decode-active telemetry snapshots
    snap_hist = collections.defaultdict(collections.Counter)
    # (arm, cell, job) -> Counter(sm) over decode-active snapshots (realized-pin audit)
    snap_sm = collections.defaultdict(collections.Counter)
    # (arm, realized sm, realized B) -> [prefill_active_batch_size]
    pf_by = collections.defaultdict(list)
    files = 0

    for fn in sorted(os.listdir(S8)):
        m = PAT.search(fn)
        if not m:
            continue
        g = m.groupdict()
        if g["mode"] != "deconf" or g["ctx"] != "1024":
            continue
        arm, job, cell, ctx = g["arm"], g["job"], g["cell"], g["ctx"]
        ts, sm, bs, pf = [], [], [], []
        for line in open(os.path.join(S8, fn)):
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
        if not ts:
            continue
        files += 1
        for s_, b_, p_ in zip(sm, bs, pf):
            if b_ and b_ > 0:                      # decode-active snapshots only
                snap_hist[(arm, cell, job, s_)][b_] += 1
                snap_sm[(arm, cell, job)][s_] += 1
                pf_by[(arm, s_, b_)].append(p_ or 0)

        t0s = {}
        for res in glob.glob(os.path.join(S8, f"s8_deconf_{arm}_C{ctx}_{job}_result.txt")):
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

        stem = os.path.join(S8, fn[: -len("_telemetry.jsonl")])
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
                for a, b in zip(ct, ct[1:]):
                    a, b = a + t0, b + t0
                    i = bisect.bisect_right(ts, a) - 1
                    j = bisect.bisect_left(ts, b)
                    if i < 0 or j >= len(ts) or sm[i] != sm[j] or bs[i] != bs[j]:
                        continue
                    if (a - ts[i]) > GUARD or (ts[j] - b) > GUARD:
                        continue
                    by_rep[(arm, sm[i], bs[i], job, cell, rep)].append((b - a) * 1000.0)

    cells = []
    for (arm, s_, b_, job, cell, rep), v in sorted(by_rep.items()):
        cells.append({
            "arm": arm, "sm": s_, "bs": b_, "job": job, "cell": cell, "rep": rep,
            "n": len(v),
            "p50": st.median(v),
            "p95": percentile(v, 0.95),
            "mean": st.fmean(v),
        })

    # SC1 self-check payload: re-pool across (job, cell, rep) -> must reproduce
    # s8_batch_matched.py's printed p50 table exactly.
    pooled_raw = collections.defaultdict(list)
    for (arm, s_, b_, job, cell, rep), v in by_rep.items():
        pooled_raw[(arm, s_, b_)].extend(v)
    pooled = [{"arm": a, "sm": s_, "bs": b_, "n": len(v),
               "p50": st.median(v), "p95": percentile(v, 0.95)}
              for (a, s_, b_), v in sorted(pooled_raw.items()) if len(v) >= 50]

    hist = []
    for (arm, cell, job, s_), c in sorted(snap_hist.items()):
        hist.append({"arm": arm, "cell": cell, "job": job, "sm": s_,
                     "n_snap": sum(c.values()),
                     "bs_hist": {str(k): c[k] for k in sorted(c)}})
    smh = [{"arm": a, "cell": c, "job": j, "sm_hist": {str(k): v[k] for k in sorted(v)},
            "n_snap": sum(v.values())}
           for (a, c, j), v in sorted(snap_sm.items())]

    cores = [{"arm": a, "sm": s_, "bs": b_, "n_snap": len(v),
              "mean_prefill_active_bs": st.fmean(v),
              "frac_prefill_active_gt0": sum(1 for x in v if x > 0) / len(v)}
             for (a, s_, b_), v in sorted(pf_by.items()) if len(v) >= 10]

    json.dump({
        "generated": "2026-08-14",
        "source_dir": S8,
        "producer_imported": "s8_batch_matched.py (unmodified; PAT/GUARD reused)",
        "guard_s": GUARD,
        "telemetry_files": files,
        "rep_cells": cells,
        "pooled_cells": pooled,
        "snapshot_bs_hist_by_realized_sm": hist,
        "snapshot_sm_hist": smh,
        "prefill_coresidency_by_realized_sm_and_B": cores,
    }, open(OUT, "w"), indent=1)
    print(f"wrote {OUT}: {len(cells)} rep-cells from {files} telemetry files")


if __name__ == "__main__":
    sys.exit(main())
