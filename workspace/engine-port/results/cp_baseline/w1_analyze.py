#!/usr/bin/env python3
"""W1 analysis: where is the knee, and where do the TTFT / ITL modes sit?  GPU 0.

Descriptive statistics only.  PREREG_W1_2026-09-02.md sec 1 and sec 8:
no arm ranking, no goodput comparison, no operating point is chosen here.
The SLO ladder (rule E) is REPORTED in full precisely so that no single threshold
becomes a primary by default.
"""
import json, statistics as st, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "benchmarks"))
from pdmux_eval.analyze import load_bench_serving_rounds, percentile  # noqa: E402

TTFT_LADDER = (500., 1000., 2000., 3000., 4000., 6000.)
ITL_LADDER = (40., 50., 60., 70., 80., 100.)


def main(outdir="."):
    out = Path(outdir)
    rows = [json.loads(l) for l in (out / "w1_summary.jsonl").read_text().splitlines() if l.strip()]
    res = {"_what_this_is": "W1 descriptive statistics: knee location and distribution "
                            "positions on the long-prompt tail. No arm ranking, no "
                            "goodput comparison, no operating point chosen.",
           "cells": []}

    for r in rows:
        if not r.get("file"):
            res["cells"].append({**r, "scored": False, "why": "boot failed"})
            continue
        try:
            reqs, dur = load_bench_serving_rounds(Path(r["file"]))
        except Exception as e:
            res["cells"].append({**r, "scored": False, "why": f"{type(e).__name__}: {e}"})
            continue
        if not reqs or dur <= 0:
            res["cells"].append({**r, "scored": False, "why": "no requests / zero duration"})
            continue
        tt = [x.ttft_ms for x in reqs]
        ip = [percentile(x.token_itl_ms, 0.95) for x in reqs]
        ladder = {f"{T:.0f}/{I:.0f}":
                  sum(1 for a, b in zip(tt, ip) if a <= T and b <= I) / dur
                  for T in TTFT_LADDER for I in ITL_LADDER}
        res["cells"].append({
            **r, "scored": True,
            "n": len(reqs), "duration_s": dur,
            "achieved_rps": len(reqs) / dur,
            "attainment": (len(reqs) / dur) / r["rate"],
            "ttft_ms": {f"p{int(q*100)}": percentile(tt, q) for q in (.1, .5, .9, .95, .99)},
            "req_itl_p95_ms": {f"p{int(q*100)}": percentile(ip, q) for q in (.1, .5, .9, .95)},
            "slo_ladder_goodput_req_s": ladder,
        })

    (out / "w1_analysis.json").write_text(json.dumps(res, indent=1))

    print("\n=== W1: achieved throughput / attainment (knee 위치) ===")
    print(f"{'arm':>10} {'rate':>6} {'np':>5} {'n':>5} {'dur_s':>7} {'achieved':>9} {'attain':>7}")
    for c in res["cells"]:
        if not c.get("scored"):
            print(f"{c.get('arm','?'):>10} {str(c.get('rate','-')):>6}  -- not scored: {c.get('why')}")
            continue
        print(f"{c['arm']:>10} {c['rate']:>6} {c['np']:>5} {c['n']:>5} {c['duration_s']:>7.1f} "
              f"{c['achieved_rps']:>9.3f} {c['attainment']:>7.3f}")

    print("\n=== 분포 위치 (모드가 어디 있는가) ===")
    print(f"{'arm':>10} {'rate':>6} | {'TTFT p50':>9} {'p90':>8} {'p95':>8} | "
          f"{'ITLp95 p10':>10} {'p50':>8} {'p90':>8}")
    for c in res["cells"]:
        if not c.get("scored"):
            continue
        t, i = c["ttft_ms"], c["req_itl_p95_ms"]
        print(f"{c['arm']:>10} {c['rate']:>6} | {t['p50']:>9.0f} {t['p90']:>8.0f} {t['p95']:>8.0f} | "
              f"{i['p10']:>10.1f} {i['p50']:>8.1f} {i['p90']:>8.1f}")

    print("\n=== SLO 사다리 goodput (req/s) -- 규칙 E: 단일 임계 없음, 전체 보고 ===")
    for c in res["cells"]:
        if not c.get("scored"):
            continue
        print(f"\n  {c['arm']} @ rate {c['rate']}   (achieved {c['achieved_rps']:.3f} req/s)")
        print("        ITL:" + "".join(f"{I:>8.0f}" for I in ITL_LADDER))
        for T in TTFT_LADDER:
            print(f"  TTFT {T:>5.0f}:" + "".join(
                f"{c['slo_ladder_goodput_req_s'][f'{T:.0f}/{I:.0f}']:>8.3f}" for I in ITL_LADDER))
    print(f"\nwrote {out / 'w1_analysis.json'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
