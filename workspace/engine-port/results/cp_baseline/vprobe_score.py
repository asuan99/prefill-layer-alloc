#!/usr/bin/env python3
"""Score one V-probe job: the UNTREATED variance of the paired goodput estimator.

Calls the CANONICAL scorer -- `../slo_sched/oracle_reanalysis_2026_08_16.score_run`,
whose predicate is `RequestResult.passes` = "TTFT <= SLO AND p95 of that request's
own token ITLs <= SLO" (CLAUDE.md gate 4).  Nothing here re-implements goodput.

What it computes, and nothing else:
  * per boot: COMBINED goodput = (good_LO + good_HI) / (dur_LO + dur_HI).
    Durations SUMMED, never max() (CLAUDE.md gate 7 -- the harness bug that
    inflated every reported goodput by the round count).
  * within this job (ONE ARM): the null Delta between boots, (b2 - b1) / b1.
    This is the quantity whose SD sets any future equivalence margin.
  * the same numbers re-scored across a 9-point SLO grid, as DIAGNOSTICS for the
    cliff-axis redesign the 5th audit asked for -- reported, never labelled.

★It does NOT compute any cross-arm quantity.  One job holds one arm, so there is
nothing to compare against; that is the design, not an instruction.

Usage:  PYTHONPATH=<repo>/workspace/engine-port/benchmarks python3 vprobe_score.py <outdir>
"""
import importlib.util, json, math, os, statistics, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORACLE = HERE.parents[0] / "slo_sched" / "oracle_reanalysis_2026_08_16.py"
if not ORACLE.exists():                       # results/cp_baseline -> results/slo_sched
    ORACLE = HERE.parent / "slo_sched" / "oracle_reanalysis_2026_08_16.py"

spec = importlib.util.spec_from_file_location("oracle_reanalysis", ORACLE)
ORC = importlib.util.module_from_spec(spec)
# ★Must be registered BEFORE exec: the module defines dataclasses, and
# `dataclasses._is_type` resolves a class's `__module__` through `sys.modules`.
# Without this the import dies with "'NoneType' object has no attribute '__dict__'"
# -- caught here by running it, not by reading it.  `design_reachability.py` does
# the same thing for the same reason.
sys.modules[spec.name] = ORC
spec.loader.exec_module(ORC)                  # safe: main() runs only under __main__
score_run = ORC.score_run

# Operating point: the value every prior sgptv run used.  The band around it is a
# DIAGNOSTIC grid, not a set of operating points to choose among.
TTFT_OP, ITL_OP = 3000.0, 60.0
TTFT_BAND = (2700.0, 3000.0, 3300.0)
ITL_BAND = (54.0, 60.0, 66.0)


def combined(lo_path, hi_path, ttft, itl):
    """COMBINED goodput over both phases, durations summed (gate 7)."""
    a = score_run(Path(lo_path), ttft, itl)
    b = score_run(Path(hi_path), ttft, itl)
    dur = a["duration_s"] + b["duration_s"]
    good = a["good"] + b["good"]
    return {
        "combined_goodput_req_s": good / dur,
        "combined_good": good,
        "combined_duration_s": dur,
        "lo": a,
        "hi": b,
    }


def main(outdir):
    out = Path(outdir)
    rows = [json.loads(l) for l in (out / "vprobe_summary.jsonl").read_text().splitlines() if l.strip()]
    boots = [r for r in rows if r.get("boot_ok")]
    arms = {r["arm"] for r in rows}
    assert len(arms) <= 1, f"a V-probe job must hold ONE arm; found {arms}"

    scored = []
    for r in boots:
        try:
            c = combined(r["lo"], r["hi"], TTFT_OP, ITL_OP)
        except Exception as e:
            print(f"  SCORE_FAILED boot={r['boot']}: {type(e).__name__}: {e}")
            continue
        band = {}
        for t in TTFT_BAND:
            for i in ITL_BAND:
                try:
                    band[f"{t:.0f}/{i:.0f}"] = combined(r["lo"], r["hi"], t, i)["combined_goodput_req_s"]
                except Exception:
                    band[f"{t:.0f}/{i:.0f}"] = None
        scored.append({
            "arm": r["arm"], "seed": r["seed"], "boot": r["boot"],
            "combined_goodput_req_s": c["combined_goodput_req_s"],
            "combined_good": c["combined_good"],
            "combined_duration_s": c["combined_duration_s"],
            "lo_goodput_req_s": c["lo"]["slo_goodput_req_s"],
            "hi_goodput_req_s": c["hi"]["slo_goodput_req_s"],
            "lo_ttft_p50_ms": c["lo"]["ttft_p50_ms"], "lo_ttft_p95_ms": c["lo"]["ttft_p95_ms"],
            "hi_ttft_p50_ms": c["hi"]["ttft_p50_ms"], "hi_ttft_p95_ms": c["hi"]["ttft_p95_ms"],
            "lo_itl_p95_ms": c["lo"]["token_itl_p95_ms"], "hi_itl_p95_ms": c["hi"]["token_itl_p95_ms"],
            "lo_req_itl_p95_p50_ms": c["lo"]["req_itl_p95_p50_ms"],
            "hi_req_itl_p95_p50_ms": c["hi"]["req_itl_p95_p50_ms"],
            "legacy_mean_itl_combined_req_s":
                (c["lo"]["legacy_good"] + c["hi"]["legacy_good"]) / c["combined_duration_s"],
            "slo_band_combined_goodput": band,
        })

    result = {
        "_what_this_is": "V-probe: untreated variance of the paired change-trace "
                         "goodput estimator. ONE ARM. No policy verdict, no arm "
                         "ranking, no cross-arm quantity (PREREG_VPROBE_2026-09-01.md).",
        "arm": (list(arms) or [None])[0],
        "operating_point": {"ttft_slo_ms": TTFT_OP, "itl_slo_ms": ITL_OP},
        "boots_requested": len(rows), "boots_ok": len(boots), "boots_scored": len(scored),
        "per_boot": scored,
    }

    # The one derived number this job exists to buy.  Reported only when both boots
    # of the pair scored -- a single boot has no null Delta, and calling that "0" is
    # the measurement-failure-as-verdict error (gate 21).
    if len(scored) >= 2:
        g = [s["combined_goodput_req_s"] for s in scored]
        deltas = [(g[i] - g[0]) / g[0] for i in range(1, len(g))]
        result["null_delta_rel"] = deltas
        result["null_delta_rel_abs_max"] = max(abs(d) for d in deltas)
        result["combined_goodput_mean"] = statistics.fmean(g)
        if len(g) >= 2:
            result["combined_goodput_sd"] = statistics.stdev(g)
            result["combined_goodput_cv"] = statistics.stdev(g) / statistics.fmean(g)
        # band stability of the RAW numbers (not a label -- cp2_rule is not applied)
        band_deltas = {}
        for k in scored[0]["slo_band_combined_goodput"]:
            vals = [s["slo_band_combined_goodput"].get(k) for s in scored]
            if all(v for v in vals):
                band_deltas[k] = (vals[1] - vals[0]) / vals[0]
        result["null_delta_rel_by_slo_point"] = band_deltas
    else:
        result["null_delta_rel"] = None
        result["_why_no_delta"] = ("fewer than two boots scored: a null Delta needs a "
                                   "pair. This is a measurement outcome, not a zero.")

    p = out / "vprobe_scores.json"
    p.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(f"\n=== V-PROBE SCORES  arm={result['arm']}  ({result['boots_scored']}/"
          f"{result['boots_requested']} boots scored) ===")
    for s in scored:
        print(f"  boot {s['boot']}: COMBINED {s['combined_goodput_req_s']:.4f} req/s "
              f"(LO {s['lo_goodput_req_s']:.4f} / HI {s['hi_goodput_req_s']:.4f})  "
              f"TTFT p95 LO/HI {s['lo_ttft_p95_ms']:.0f}/{s['hi_ttft_p95_ms']:.0f} ms  "
              f"req-ITL-p95 p50 LO/HI {s['lo_req_itl_p95_p50_ms']:.1f}/"
              f"{s['hi_req_itl_p95_p50_ms']:.1f} ms")
    if result.get("null_delta_rel"):
        print(f"  ★NULL DELTA (same arm, boot-to-boot): "
              f"{['%+.4f' % d for d in result['null_delta_rel']]}  "
              f"| CV {result.get('combined_goodput_cv', float('nan')):.4f}")
        print("  (one job = one pair. The SD this campaign buys is computed ACROSS "
              "jobs by vprobe_aggregate.py, not here.)")
    print(f"  wrote {p}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
