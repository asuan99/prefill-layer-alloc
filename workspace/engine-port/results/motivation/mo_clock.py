#!/usr/bin/env python3
"""GPU clock / power / clock-limit report per run and per arm (motivation harness, 2026-11-11).  CPU only.

Input: the Job's meta/clk.csv, written by scripts/vessl/job_entry.sh (nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,
temperature.gpu,clocks_throttle_reasons.active,index --format=csv -lms 100; container clock = UTC) and the run records' load windows
(t_load_start_epoch .. t_load_end_epoch).  A sample is "limited" under the job_entry.sh rule: any active reason bit other than 0x1
(GpuIdle).  The report is a covariate shown next to the results; it changes no label and no capacity.

  python3 mo_clock.py RAW_DIR CLK_CSV [--out clock_report.json]
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

IDLE_BIT = 0x1
REASON_BITS = {0x2: "applications_clocks", 0x4: "sw_power_cap", 0x8: "hw_slowdown", 0x10: "sync_boost", 0x20: "sw_thermal",
               0x40: "hw_thermal", 0x80: "hw_power_brake", 0x100: "display_clocks"}


def _num(cell):
    """'1410 MHz' / '57.08 W' / '27' -> float; '[N/A]' -> None."""
    try:
        return float(cell.strip().split()[0])
    except (ValueError, IndexError):
        return None


def _epoch(ts):
    """nvidia-smi timestamp '2026/11/11 04:21:46.773' (UTC in the container) -> epoch seconds."""
    return datetime.strptime(ts.strip(), "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=timezone.utc).timestamp()


def read_clk(path, gpu_index=None):
    """-> sorted list of samples {t, sm_mhz, power_w, temp_c, reasons, limited, gpu}.  Malformed rows are skipped and counted."""
    out, bad = [], 0
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    for r in rows[1:]:
        if len(r) < 5:
            bad += 1
            continue
        try:
            t, reasons = _epoch(r[0]), int(r[4].strip(), 16)
        except ValueError:
            bad += 1
            continue
        gpu = r[5].strip() if len(r) >= 6 else "0"
        if gpu_index is not None and gpu != str(gpu_index):
            continue
        out.append({"t": t, "sm_mhz": _num(r[1]), "power_w": _num(r[2]), "temp_c": _num(r[3]), "reasons": reasons,
                    "limited": (reasons & ~IDLE_BIT) != 0, "gpu": gpu})
    out.sort(key=lambda s: s["t"])
    return out, bad


def _q(xs, p):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def summarize(samples):
    """Samples of one window -> report row.  n = 0 -> every value None (a window shorter than the 100 ms period)."""
    n = len(samples)
    reasons = 0
    for s in samples:
        reasons |= s["reasons"]
    lim = sum(s["limited"] for s in samples)
    return {"n_samples": n, "limited_frac": (lim / n) if n else None, "n_limited": lim,
            "sm_mhz_p50": _q([s["sm_mhz"] for s in samples], 50), "sm_mhz_p10": _q([s["sm_mhz"] for s in samples], 10),
            "power_w_p50": _q([s["power_w"] for s in samples], 50), "power_w_max": max((s["power_w"] for s in samples if s["power_w"] is not None),
                                                                                       default=None),
            "temp_c_max": max((s["temp_c"] for s in samples if s["temp_c"] is not None), default=None),
            "reasons_seen": [name for bit, name in sorted(REASON_BITS.items()) if reasons & bit]}


def window(samples, t0, t1):
    return [s for s in samples if t0 <= s["t"] <= t1]


def report(runs, samples):
    """runs: run records (mo_runner) -> {"per_run": [...], "per_arm": {arm: row}, "spread": {...}}.
    per_arm pools the samples of all its runs (not a mean of fractions), so long runs weigh more, like the load itself."""
    per_run, by_arm = [], {}
    for r in sorted(runs, key=lambda r: r["t_load_start_epoch"]):
        w = window(samples, r["t_load_start_epoch"], r["t_load_end_epoch"])
        per_run.append({"run_id": r["run_id"], "arm": r["arm"], "boot_id": r.get("boot_id"), "kind": r.get("kind"),
                        "t0": r["t_load_start_epoch"], "t1": r["t_load_end_epoch"], **summarize(w)})
        by_arm.setdefault(r["arm"], []).extend(w)
    per_arm = {a: summarize(w) for a, w in sorted(by_arm.items())}
    fr = [v["limited_frac"] for v in per_arm.values() if v["limited_frac"] is not None]
    spread = {"limited_frac_min": min(fr) if fr else None, "limited_frac_max": max(fr) if fr else None,
              "limited_frac_range": (max(fr) - min(fr)) if fr else None}
    return {"rule": "limited = any active clocks_throttle_reasons bit other than 0x1 (GpuIdle), as scripts/vessl/job_entry.sh",
            "use": "covariate shown next to results; changes no label", "per_run": per_run, "per_arm": per_arm, "spread": spread}


def table_rows(rep):
    """Short per-arm rows for the analyzer's text table."""
    return [{"arm": a, "limited_frac": v["limited_frac"], "sm_mhz_p50": v["sm_mhz_p50"], "power_w_p50": v["power_w_p50"],
             "reasons_seen": ",".join(v["reasons_seen"]) or "-"} for a, v in rep["per_arm"].items()]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("clk")
    ap.add_argument("--gpu", default=None)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    runs = [json.loads(p.read_text()) for p in sorted((Path(a.raw) / "runs").glob("*.json"))]
    samples, bad = read_clk(a.clk, a.gpu)
    rep = report(runs, samples)
    rep["clk_csv"], rep["n_samples_total"], rep["n_rows_skipped"] = str(a.clk), len(samples), bad
    txt = json.dumps(rep, indent=1)
    if a.out:
        Path(a.out).write_text(txt + "\n")
    print(json.dumps(table_rows(rep), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
