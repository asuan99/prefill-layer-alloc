#!/usr/bin/env python3
"""Gate-2 HOLB-probe GPU-gate scorer: G3 (correctness) / G4 (instrumentation
validity) / G5 (observer effect, paired-t).

This is a VALIDITY-GATE scorer, not a performance-campaign scorer.  It makes
no claim about which arm is faster or whether pdmux helps -- see
DIRECT_BLOCKING_DESIGN.md section 10 for why stall_ms is not (yet) a
performance primary.  The only judgments this script makes are:

  G3  did PDMUX_HOLB_PATH change greedy output?          (must be NO)
  G4  is the instrumentation internally consistent?       (three thresholds,
                                                             pre-registered in
                                                             DIRECT_BLOCKING_DESIGN.md
                                                             sections 4.2/4.4)
  G5  does the probe change measured performance by >=3%? (paired t, n=NREP,
                                                             per arm, per
                                                             response variable)

Also reports (not scored, "free" from the same run, per the task instructions):
  stall_ms, stall_req_ms, episode length distribution (p50/p95/p99/max),
  n_mixed, n_invalid, decode_fw_ms, stall_ambiguous_ms, and the diagnostic
  same_stream_fw_ms / other_stream_fw_ms / bubble_ms decomposition, PLUS the
  rep-to-rep variance of stall_ms per arm (needed for the still-unregistered
  rev3 delta -- see DIRECT_BLOCKING_DESIGN.md section 10.2 item 1).

Usage:
  python3 g2_holb_analyze.py --outdir DIR --tag TAG --jobid JOBID \
      --arms plain plainaux chunk512 agnostic --nrep 5
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from scipy import stats as _scipy_stats
except Exception:  # pragma: no cover - scipy is present in this project's venv
    _scipy_stats = None

RESPONSE_VARS = ["request_throughput", "ttft_p50", "ttft_p95", "itl_p50", "itl_p95", "mean_e2e_ms"]

G4_TIMELINE_OVER_HOST_TOL = 0.02  # |T/host - 1| <= 0.02
G4_INVALID_FRAC_MAX = 0.01  # n_invalid_gaps / n_decode_fw <= 0.01
G5_EFFECT_THRESHOLD = 0.03  # |point estimate| < 3%


def percentile(sorted_vals: List[float], frac: float) -> Optional[float]:
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    idx = frac * (len(sorted_vals) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return sorted_vals[lo]
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (idx - lo)


def t_crit_975(df: int) -> float:
    if _scipy_stats is not None:
        return float(_scipy_stats.t.ppf(0.975, df))
    # Fallback table (two-sided 95%), only used if scipy import fails.
    table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
             7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
    if df in table:
        return table[df]
    return 1.96  # asymptotic


def paired_t_ci(diffs: List[float]) -> Dict[str, Any]:
    n = len(diffs)
    if n < 2:
        return {"n": n, "mean": (diffs[0] if n == 1 else None), "sd": None,
                "ci_lo": None, "ci_hi": None, "note": "n<2, no CI"}
    mean = statistics.mean(diffs)
    sd = statistics.stdev(diffs)  # sample sd, ddof=1
    se = sd / math.sqrt(n)
    tcrit = t_crit_975(n - 1)
    return {
        "n": n, "mean": mean, "sd": sd, "se": se,
        "ci_lo": mean - tcrit * se, "ci_hi": mean + tcrit * se,
        "tcrit_df": n - 1,
    }


def g5_verdict(ci: Dict[str, Any]) -> str:
    if ci.get("mean") is None:
        return "NO_DATA"
    if ci.get("ci_lo") is None:
        # n=1: can only report the point estimate, no interval judgement possible.
        return "UNDERPOWERED_N1"
    ci_contains_zero = ci["ci_lo"] <= 0.0 <= ci["ci_hi"]
    point_ok = abs(ci["mean"]) < G5_EFFECT_THRESHOLD
    if point_ok and ci_contains_zero:
        return "PASS"
    return "FAIL"


# ---------------------------------------------------------------------------
# Phase A (G3) parsing
# ---------------------------------------------------------------------------

def load_correctness(outdir: Path, tag: str, jobid: str) -> Optional[Dict[str, Any]]:
    """Reads g2holb_<tag>_correctness_<jobid>.json, which is JSONL (one record
    per arm, written incrementally by the sbatch's Phase-A loop) despite the
    .json extension. Returns {arm: record}."""
    p = outdir / f"g2holb_{tag}_correctness_{jobid}.json"
    if not p.exists():
        return None
    out: Dict[str, Any] = {}
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        out[rec["arm"]] = rec
    return out


# ---------------------------------------------------------------------------
# Phase B (G5) parsing
# ---------------------------------------------------------------------------

def load_bench_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    lines = [l for l in path.read_text().splitlines() if l.strip()]
    if not lines:
        return None
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError:
        return None


def response_vars(d: Dict[str, Any]) -> Dict[str, Optional[float]]:
    ttfts = sorted(x for x in (d.get("ttfts") or []) if x is not None)
    out = {
        "request_throughput": d.get("request_throughput"),
        "ttft_p50": d.get("median_ttft_ms"),
        "ttft_p95": percentile(ttfts, 0.95),
        "itl_p50": d.get("median_itl_ms"),
        "itl_p95": d.get("p95_itl_ms"),
        "mean_e2e_ms": d.get("mean_e2e_latency_ms"),
        "completed": d.get("completed"),
        "n_err": sum(1 for e in (d.get("errors") or []) if e),
    }
    return out


def collect_phase_b(outdir: Path, tag: str, jobid: str, arms: List[str], nrep: int) -> Dict[str, Any]:
    per_arm: Dict[str, Any] = {}
    for arm in arms:
        rows = {"on": {}, "off": {}}
        for rep in range(1, nrep + 1):
            for onoff in ("on", "off"):
                fp = outdir / f"g2holb_{tag}_{arm}_{onoff}_rep{rep}_{jobid}.jsonl"
                d = load_bench_json(fp)
                rows[onoff][rep] = response_vars(d) if d is not None else None
        per_arm[arm] = rows
    return per_arm


def paired_diffs_for_var(rows: Dict[str, Any], nrep: int, var: str) -> Tuple[List[float], List[int]]:
    diffs = []
    used_reps = []
    for rep in range(1, nrep + 1):
        on = rows["on"].get(rep)
        off = rows["off"].get(rep)
        if on is None or off is None:
            continue
        on_v = on.get(var)
        off_v = off.get(var)
        if on_v is None or off_v is None:
            continue
        if off_v == 0:
            continue
        diffs.append((on_v - off_v) / off_v)
        used_reps.append(rep)
    return diffs, used_reps


# ---------------------------------------------------------------------------
# HOLB summary parsing (G4 + free diagnostics)
# ---------------------------------------------------------------------------

def load_last_holb_summary(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    last = None
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("event") == "holb_summary":
                last = r
    return last


def collect_holb_summaries(outdir: Path, tag: str, jobid: str, arms: List[str], nrep: int) -> Dict[str, List[Dict[str, Any]]]:
    per_arm: Dict[str, List[Dict[str, Any]]] = {}
    for arm in arms:
        summaries = []
        for rep in range(1, nrep + 1):
            fp = outdir / f"g2holb_{tag}_{arm}_rep{rep}_{jobid}.holb.jsonl"
            s = load_last_holb_summary(fp)
            if s is not None:
                s["_rep"] = rep
            summaries.append(s)
        per_arm[arm] = summaries
    return per_arm


def g4_check(summary: Dict[str, Any]) -> Dict[str, Any]:
    n_decode = max(1, summary.get("n_decode_fw", 0) or 0)
    invalid_frac = (summary.get("n_invalid_gaps", 0) or 0) / n_decode
    timeline_over_host = summary.get("timeline_over_host")
    dropped = summary.get("dropped_spans", 0) or 0
    checks = {
        "timeline_over_host": timeline_over_host,
        "timeline_over_host_ok": (
            timeline_over_host is not None
            and abs(timeline_over_host - 1.0) <= G4_TIMELINE_OVER_HOST_TOL
        ),
        "invalid_frac": invalid_frac,
        "invalid_frac_ok": invalid_frac <= G4_INVALID_FRAC_MAX,
        "dropped_spans": dropped,
        "dropped_spans_ok": dropped == 0,
        "n_inconsistent_gaps": summary.get("n_inconsistent_gaps", 0),
        "n_errors": summary.get("n_errors", 0),
    }
    checks["G4_PASS"] = (
        checks["timeline_over_host_ok"] and checks["invalid_frac_ok"] and checks["dropped_spans_ok"]
    )
    return checks


DIAG_FIELDS = [
    "stall_ms", "stall_req_ms", "n_stall_episodes",
    "stall_ambiguous_ms", "n_ambiguous_gaps",
    "decode_fw_ms", "n_decode_fw", "n_mixed", "mixed_fw_ms",
    "blocking_fw_ms", "same_stream_fw_ms", "other_stream_fw_ms", "bubble_ms",
    "n_invalid_gaps", "n_inconsistent_gaps",
    "episode_p50_ms", "episode_p95_ms", "episode_p99_ms", "episode_max_ms",
    "timeline_ms", "host_span_ms", "timeline_over_host", "stall_frac",
    "dropped_spans", "n_errors", "mode_counts",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--jobid", required=True)
    ap.add_argument("--arms", nargs="+", default=["plain", "plainaux", "chunk512", "agnostic"])
    ap.add_argument("--nrep", type=int, default=5)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    report: Dict[str, Any] = {"tag": args.tag, "jobid": args.jobid, "arms": args.arms, "nrep": args.nrep}

    # ---- G3 --------------------------------------------------------------
    # Three-way, not two-way: PASS (measured, ON==OFF) / FAIL (measured, ON!=OFF
    # or self-repro unstable) / UNDETERMINED (could not be measured -- boot
    # failure or a response-schema extraction error). Collapsing UNDETERMINED
    # into FAIL is exactly the bug that mislabelled job 874601 ("the probe
    # changed output") when the real cause was a KeyError in the sha
    # extractor (empty-string shas trivially "matched" and were then reported
    # as FAIL because sha_off/sha_on were both None, not because ON and OFF
    # actually differed). Measurement absence and gate violation are different
    # events -- see g2_holb_phaseA_lib.sh's header for the full post-mortem.
    corr = load_correctness(outdir, args.tag, args.jobid)
    g3_status = None
    if corr is not None:
        results = {a: corr.get(a, {}).get("result") for a in args.arms}
        if any(r == "FAIL" for r in results.values()):
            g3_status = "FAIL"
        elif any(r is None or r == "UNDETERMINED" for r in results.values()):
            g3_status = "UNDETERMINED"
        elif all(r == "PASS" for r in results.values()):
            g3_status = "PASS"
        else:
            g3_status = "UNDETERMINED"  # unrecognized value -- treat conservatively
    report["G3"] = {"raw": corr, "STATUS": g3_status}

    print("=" * 70)
    print(f"GATE 2 HOLB SCORING  tag={args.tag} jobid={args.jobid}")
    print("=" * 70)
    print("\n--- G3 correctness (probe ON/OFF byte-identical greedy output) ---")
    if corr is None:
        print("  NO g2holb_*_correctness_*.json found -- falling back to stdout grep is required.")
    else:
        for a in args.arms:
            row = corr.get(a, {})
            print(f"  arm={a:10s} result={row.get('result')} "
                  f"status_off={row.get('status_off')} status_on={row.get('status_on')} "
                  f"method_off={row.get('method_off')} method_on={row.get('method_on')} "
                  f"sha_off={str(row.get('sha_off'))[:12]} sha_on={str(row.get('sha_on'))[:12]} "
                  f"self_repro_off={row.get('self_repro_off')} self_repro_on={row.get('self_repro_on')}")
            if row.get("result") == "UNDETERMINED":
                print(f"    detail_off={row.get('detail_off')} detail_on={row.get('detail_on')}")
        print(f"  G3 STATUS = {g3_status}")

    if g3_status in ("FAIL", "UNDETERMINED", None):
        if g3_status == "FAIL":
            print("\n*** G3 FAILED (measured: at least one arm's ON/OFF output genuinely differs, or")
            print("*** is self-repro unstable). Per DIRECT_BLOCKING_DESIGN.md: STOP HERE. G4/G5 below")
            print("*** (if any) are NOT meaningful -- the probe changes engine output. ***")
        elif g3_status == "UNDETERMINED":
            print("\n*** G3 UNDETERMINED (at least one arm could not be measured -- boot failure or a")
            print("*** response-schema extraction error; see the per-arm detail_off/detail_on above and")
            print("*** the g2holb_*_A_*_call*_*.raw.json dumps). This is NOT evidence the probe changed")
            print("*** output -- do not report it as a G3 failure. Re-run after fixing the harness. ***")
        else:
            print("\n*** G3 status could not be determined (no correctness artifact). STOP HERE. ***")
        report["STOPPED_AFTER_G3"] = g3_status or "NO_ARTIFACT"
        (outdir / f"g2holb_report_{args.tag}_{args.jobid}.json").write_text(json.dumps(report, indent=2, default=str))
        return 1

    # ---- G4 + free diagnostics -------------------------------------------
    holb = collect_holb_summaries(outdir, args.tag, args.jobid, args.arms, args.nrep)
    print("\n--- G4 instrumentation validity + free diagnostics (per arm, per rep) ---")
    g4_all_pass = True
    report["G4"] = {}
    report["diagnostics"] = {}
    for arm in args.arms:
        summaries = holb[arm]
        present = [s for s in summaries if s is not None]
        report["G4"][arm] = []
        report["diagnostics"][arm] = []
        stall_ms_series = []
        for rep, s in zip(range(1, args.nrep + 1), summaries):
            if s is None:
                print(f"  arm={arm:10s} rep={rep} NO HOLB SUMMARY (server/rep may have failed)")
                report["G4"][arm].append({"rep": rep, "missing": True})
                continue
            chk = g4_check(s)
            g4_all_pass = g4_all_pass and chk["G4_PASS"]
            print(f"  arm={arm:10s} rep={rep} G4={'PASS' if chk['G4_PASS'] else 'FAIL'} "
                  f"timeline_over_host={chk['timeline_over_host']} "
                  f"invalid_frac={chk['invalid_frac']:.5f} dropped_spans={chk['dropped_spans']} "
                  f"n_inconsistent={chk['n_inconsistent_gaps']} n_errors={chk['n_errors']}")
            chk["rep"] = rep
            report["G4"][arm].append(chk)
            diag = {k: s.get(k) for k in DIAG_FIELDS}
            diag["rep"] = rep
            report["diagnostics"][arm].append(diag)
            if s.get("stall_ms") is not None:
                stall_ms_series.append(s["stall_ms"])
        n_present = len(present)
        if n_present < len(summaries):
            print(f"  arm={arm:10s} *** {len(summaries) - n_present}/{len(summaries)} reps missing HOLB summary ***")
        if len(stall_ms_series) >= 2:
            sm = statistics.mean(stall_ms_series)
            sd = statistics.stdev(stall_ms_series)
            cv = sd / sm if sm else None
            print(f"  arm={arm:10s} stall_ms across reps: n={len(stall_ms_series)} "
                  f"mean={sm:.3f} sd={sd:.3f} cv={cv:.4f} values={['%.3f' % v for v in stall_ms_series]}")
            report["diagnostics"][arm + "_stall_ms_rep_variance"] = {
                "n": len(stall_ms_series), "mean": sm, "sd": sd, "cv": cv, "values": stall_ms_series,
            }
        elif len(stall_ms_series) == 1:
            print(f"  arm={arm:10s} stall_ms: only 1 rep with data ({stall_ms_series[0]:.3f}), no variance")
    print(f"\n  G4 (ALL arm x rep PASS) = {g4_all_pass}")
    report["G4_ALL_PASS"] = g4_all_pass

    # ---- G5 observer effect ------------------------------------------------
    phaseb = collect_phase_b(outdir, args.tag, args.jobid, args.arms, args.nrep)
    print("\n--- G5 observer effect (paired t, n<=nrep, per arm x response var) ---")
    report["G5"] = {}
    g5_all_pass = True
    for arm in args.arms:
        rows = phaseb[arm]
        report["G5"][arm] = {}
        for var in RESPONSE_VARS:
            diffs, used_reps = paired_diffs_for_var(rows, args.nrep, var)
            ci = paired_t_ci(diffs)
            verdict = g5_verdict(ci)
            g5_all_pass = g5_all_pass and (verdict == "PASS")
            ci_str = (f"[{ci['ci_lo']*100:+.2f}%, {ci['ci_hi']*100:+.2f}%]"
                      if ci.get("ci_lo") is not None else "n/a")
            mean_str = f"{ci['mean']*100:+.2f}%" if ci.get("mean") is not None else "n/a"
            print(f"  arm={arm:10s} var={var:20s} n={ci['n']} mean={mean_str:>9s} "
                  f"CI95={ci_str:>22s} verdict={verdict} reps={used_reps}")
            ci["verdict"] = verdict
            ci["diffs"] = diffs
            ci["used_reps"] = used_reps
            report["G5"][arm][var] = ci
    print(f"\n  G5 (ALL arm x var PASS, |effect|<{G5_EFFECT_THRESHOLD*100:.0f}% AND CI ni 0) = {g5_all_pass}")
    report["G5_ALL_PASS"] = g5_all_pass

    # ---- overall -----------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"SUMMARY  G3={g3_status}  G4={g4_all_pass}  G5={g5_all_pass}")
    print("=" * 70)
    report["overall"] = {"G3": g3_status, "G4": g4_all_pass, "G5": g5_all_pass}

    out_json = outdir / f"g2holb_report_{args.tag}_{args.jobid}.json"
    out_json.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nFull report -> {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
