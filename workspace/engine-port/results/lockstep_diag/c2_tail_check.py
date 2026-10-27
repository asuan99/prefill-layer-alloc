#!/usr/bin/env python3
"""C2 (s8_scaleup deconf ctx1024, KISTI A100) -- did the adopted ITL intervals include
long stalls?  Mirrors the adoption rule of results/s8_scaleup/s8_batch_matched.py
(both bracketing runtime_snapshots agree on realized decode SM and decode batch,
GUARD 3.0 s) but keeps EVERY client interval and flags whether it was adopted.
No PHASE_EVENTS exist in these runs (KISTI, 2026-07), so spans cannot be attributed;
this only measures the tail of the adopted vs rejected interval populations.
Descriptive only.  Output: c2_tail_check.json next to this file.
"""
import bisect
import glob
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from pdmux_eval.analyze import percentile  # noqa: E402

S8 = ROOT / "results" / "s8_scaleup"
GUARD = 3.0
PAT = re.compile(r"s8_deconf_(?P<arm>\w+?)_C1024_(?P<cell>d16|d24|d44|d92|np)_(?P<job>\d+)_telemetry\.jsonl$")


def q(v):
    if not v:
        return None
    med = percentile(v, .5)
    return dict(n=len(v), p50=med, p95=percentile(v, .95), p99=percentile(v, .99), max=max(v),
                frac_gt3x_p50=sum(1 for x in v if x > 3 * med) / len(v))


def main():
    out = {}
    for fn in sorted(os.listdir(S8)):
        m = PAT.search(fn)
        if not m:
            continue
        arm, cell, job = m["arm"], m["cell"], m["job"]
        ts, sm, bs, pf = [], [], [], []
        for line in open(S8 / fn):
            try:
                e = json.loads(line)
            except Exception:
                continue
            if e.get("event") != "runtime_snapshot" or e.get("phase") != "benchmark":
                continue
            ts.append(e["timestamp_monotonic_s"]); sm.append(e["decode_sms"] or 108)
            bs.append(e["decode_running_batch_size"]); pf.append(e.get("prefill_active_batch_size", 0))
        if not ts:
            continue
        t0s = {}
        for res in glob.glob(str(S8 / f"s8_deconf_{arm}_C1024_{job}_result.txt")):
            for line in open(res):
                if line.startswith("{"):
                    try:
                        s = json.loads(line)
                    except Exception:
                        continue
                    tag = s.get("tag", "")
                    if tag.startswith(f"deconf_{arm}_C1024_{cell}_r") and "t0_monotonic_s" in s:
                        t0s[int(tag.rsplit("_r", 1)[1])] = s["t0_monotonic_s"]
        adopted, rejected = [], []
        for raw in sorted(glob.glob(str(S8 / fn.replace("_telemetry.jsonl", "_rep*_raw.jsonl")))):
            t0 = t0s.get(int(re.search(r"_rep(\d+)_raw", raw).group(1)))
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
                    v = (b - a) * 1e3
                    i = bisect.bisect_right(ts, a) - 1
                    j = bisect.bisect_left(ts, b)
                    ok = not (i < 0 or j >= len(ts) or sm[i] != sm[j] or bs[i] != bs[j]
                              or (a - ts[i]) > GUARD or (ts[j] - b) > GUARD)
                    (adopted if ok else rejected).append(v)
        if adopted or rejected:
            out[f"{arm}/{cell}/{job}"] = dict(adopted=q(adopted), rejected=q(rejected),
                                              adopted_frac=len(adopted) / max(1, len(adopted) + len(rejected)))
    json.dump(out, open(HERE / "c2_tail_check.json", "w"), indent=1)
    for k, v in out.items():
        a, r = v["adopted"] or {}, v["rejected"] or {}
        print(k, "adopt%.2f" % v["adopted_frac"],
              "A p50 %.1f p99 %.1f max %.0f gt3x %.4f" % (a.get("p50", 0), a.get("p99", 0), a.get("max", 0), a.get("frac_gt3x_p50", 0)),
              "| R n %d p50 %.1f p99 %.1f max %.0f gt3x %.4f" % (r.get("n", 0), r.get("p50", 0) or 0, r.get("p99", 0) or 0, r.get("max", 0) or 0, r.get("frac_gt3x_p50", 0) or 0))


if __name__ == "__main__":
    main()
