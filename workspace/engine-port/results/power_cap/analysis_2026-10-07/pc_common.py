#!/usr/bin/env python3
"""Shared helpers for the power-cap PASSIVE re-analysis (CPU only, 0 GPU-h).

Scope (2026-10-07, plan `power_cap`): descriptive re-aggregation of clock/power traces that the
VESSL harness already records (`scripts/vessl/job_entry.sh:155-157`, 100 ms `nvidia-smi`
sampler) and of the KISTI-era 5 s `*_clocks.csv` sweeps.  Nothing here is a decision rule.
Rules LVR-2 / VC-2 / N3C-16 (no arm-delta attribution to power cap) stay in force.

Metric functions are IMPORTED from `benchmarks/pdmux_eval/analyze.py` (not copied -- lesson 9,
identity gates).  Clock parsing mirrors `results/n3_capacity/n3_analyze.py:593-619 clk_windows`
and extends it with the power column (r[2]); `pc_energy.py --regress` cross-checks the 0x4 %
against the N3 analyzer's own output on the same windows.

Caveats carried into every output:
  * NVML `power.draw` on A100 is a ~1 s moving average; 100 ms samples are not independent.
  * SW power cap (bit 0x4) exposure is ENDOGENOUS to offered load (9 -> 46 % across N3 rates).
  * KISTI `*_clocks.csv` have no timestamps (5 s cadence); the busy mask is `utilization.gpu`.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import math
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENG = HERE.parents[2]                      # .../workspace/engine-port
RES = ENG / "results"
sys.path.insert(0, str(ENG / "benchmarks" / "pdmux_eval"))
from analyze import (  # noqa: E402  (imported, not copied)
    RequestResult, load_bench_serving_rounds, percentile, summarize_requests,
)

BIT_SW_POWER_CAP = 0x4
BITS = {0x1: "GpuIdle", 0x2: "AppClocksSetting", 0x4: "SwPowerCap", 0x8: "HwSlowdown", 0x10: "SyncBoost",
        0x20: "SwThermalSlowdown", 0x40: "HwThermalSlowdown", 0x80: "HwPowerBrakeSlowdown",
        0x100: "DisplayClockSetting"}
A100_SXM_POWER_LIMIT_W = 400.0            # substrate.json power_limit on VESSL; KISTI A100-SXM4-80GB 400.00 W
CLK_TS_FMT = "%Y/%m/%d %H:%M:%S.%f"

# ---------------------------------------------------------------- VESSL campaigns with a local clk.csv
VESSL = {
    "n3_r3": {
        "out": RES / "n3_capacity" / "n3b_pdmux-n3-capacity-38ed3f4-20261006092813",
        "clk": RES / "n3_capacity" / "vessl" / "pdmux-n3-capacity-38ed3f4-20261006092813" / "meta" / "clk.csv",
        "timing_key": "unit", "windows": "timing",
        "slo": {"ttft_ms": "T_K.json:k=5", "itl_ms": 50.0},
        "note": "N3 shape B 3rd run (rev6); PHASE_EVENTS=1 on grid/obs_on/floor units",
    },
    "e2_r2": {
        "out": RES / "r2_eval" / "e2_sticky_vessl" / "e2v_pdmux-e2-sticky-vessl-22c8c64-20261006020317",
        "clk": RES / "e2_sticky_vessl" / "vessl" / "pdmux-e2-sticky-vessl-22c8c64-20261006020317" / "meta" / "clk.csv",
        "timing_key": "boot", "windows": "timing",
        "slo": {"ttft_ms": 3000.0, "itl_ms": 50.0},
        "note": "E2 sticky OFF/ON contrast r2 (TTFT 3000 / ITL 50 as in e2v_r2_result.py:36)",
    },
    "lam0": {
        "out": RES / "r2_eval" / "lambda0_vessl" / "lam0v_pdmux-lambda0-vessl-1d47d3e-20261004084945",
        "clk": RES / "lambda0_vessl" / "vessl" / "pdmux-lambda0-vessl-1d47d3e-20261004084945" / "meta" / "clk.csv",
        "windows": "srv_log",
        "slo": {"ttft_ms": 3000.0, "itl_ms": 50.0},
        "note": "lambda0 VESSL re-anchor; bench window = last srv.log timestamp - duration_s "
                "(lam0v_descriptive.py:95-96 convention, +-1-2 s)",
    },
}
LAM0_CELLS = ["a_r0", "a_r1", "a_r2", "a_r3", "a_r4", "a_r4_s2", "b_r0", "b_r1", "b_r2", "b_r3", "b_r3_s2"]
LAM0_VARIANCE = ["a_r4_s3", "b_r3_s3", "a_r4_s4", "b_r3_s4"]


# ---------------------------------------------------------------- clk.csv (VESSL, 100 ms)
def parse_clk(path: Path):
    """-> list of (epoch_s, sm_mhz, power_w, temp_c, reasons_bits), sorted by time.

    Same field conventions as n3_analyze.clk_windows / lam0v_descriptive.py; rows that fail to
    parse are skipped (header, partial last line)."""
    rows = []
    with open(path) as f:
        rd = csv.reader(f)
        next(rd, None)
        for r in rd:
            try:
                t = dt.datetime.strptime(r[0].strip(), CLK_TS_FMT).replace(tzinfo=dt.timezone.utc).timestamp()
                sm = int(r[1].strip().split()[0])
                pw = float(r[2].strip().split()[0])
                tmp = int(r[3].strip())
                v = int(r[4].strip(), 16)
            except (ValueError, IndexError):
                continue
            rows.append((t, sm, pw, tmp, v))
    rows.sort(key=lambda x: x[0])
    return rows


def window(rows, a: float, b: float):
    return [x for x in rows if a <= x[0] <= b]


def _pct(sorted_vals, p):
    return percentile(sorted_vals, p)


def window_stats(w, dt_cap_s: float = 0.5) -> dict:
    """Descriptive stats of one window of clk rows.

    energy_J = sum_i P_i * dt_i with dt_i = t_{i+1} - t_i clipped at dt_cap_s (sampler gaps are not
    integrated as if power were constant across them); the last sample gets the median dt."""
    n = len(w)
    if n == 0:
        return {"n": 0}
    dts = [w[i + 1][0] - w[i][0] for i in range(n - 1)]
    med = statistics.median(dts) if dts else 0.1
    e = 0.0
    for i in range(n):
        d = dts[i] if i < n - 1 else med
        e += w[i][2] * min(max(d, 0.0), dt_cap_s)
    pw = sorted(x[2] for x in w)
    sm = sorted(x[1] for x in w)
    cap = sum(1 for x in w if x[4] & BIT_SW_POWER_CAP)
    nonidle = sum(1 for x in w if x[4] & ~0x1)
    other = {}
    for bit, name in BITS.items():
        if bit in (0x1, BIT_SW_POWER_CAP):
            continue
        c = sum(1 for x in w if x[4] & bit)
        if c:
            other[name] = 100.0 * c / n
    return {
        "n": n, "span_s": w[-1][0] - w[0][0], "dt_median_s": med,
        "energy_J": e, "mean_W": statistics.fmean(pw),
        "p50_W": _pct(pw, 0.50), "p95_W": _pct(pw, 0.95), "max_W": pw[-1],
        "frac_W_ge_0p95_limit": sum(1 for p in pw if p >= 0.95 * A100_SXM_POWER_LIMIT_W) / n,
        "sw_power_cap_pct": 100.0 * cap / n, "nonidle_reason_pct": 100.0 * nonidle / n,
        "other_reason_pct": other,
        "sm_mhz_mean": statistics.fmean(sm), "sm_mhz_p5": _pct(sm, 0.05), "sm_mhz_min": sm[0],
        "temp_max_C": max(x[3] for x in w),
    }


def read_timing(path: Path, key: str) -> dict:
    """TIMING.tsv -> {unit: {bench_start_epoch, bench_end_epoch, ...}} (N3 key 'unit', E2 key 'boot')."""
    out = {}
    lines = Path(path).read_text().splitlines()
    hdr = lines[0].split("\t")
    for ln in lines[1:]:
        f = dict(zip(hdr, ln.split("\t")))
        try:
            a, b = float(f["bench_start_epoch"]), float(f["bench_end_epoch"])
        except (KeyError, ValueError):
            continue
        out[f[key]] = {"a": a, "b": b, "row": f}
    return out


SRV_TS = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\]")


def srv_last_ts(path: Path):
    last = None
    for line in open(path, errors="replace"):
        if "Decode batch" in line or "Prefill batch" in line:
            m = SRV_TS.match(line)
            if m:
                last = dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc).timestamp()
    return last


def lam0_windows(out: Path) -> dict:
    """lambda0 VESSL cells: bench window = [last srv timestamp - duration_s, last srv timestamp]."""
    w = {}
    for c in LAM0_CELLS + LAM0_VARIANCE:
        d = out / "variance" if c in LAM0_VARIANCE else out
        cj = d / f"cell_{c}.json"
        srv = d / f"srv_{c}.log"
        if not cj.exists() or not srv.exists():
            continue
        cell = json.load(open(cj))
        last = srv_last_ts(srv)
        if last is None:
            continue
        w[c] = {"a": last - float(cell["duration_s"]), "b": last, "row": cell, "dir": d}
    return w


# ---------------------------------------------------------------- bench_serving artifact
def bench_stats(bench_path: Path, ttft_slo_ms: float, itl_slo_ms: float) -> dict:
    reqs, dur = load_bench_serving_rounds(bench_path)
    rec = json.loads(open(bench_path).readline())
    out_tok = rec.get("total_output_tokens")
    if out_tok is None and rec.get("output_lens"):
        out_tok = sum(rec["output_lens"])
    in_tok = rec.get("total_input_tokens")
    s = summarize_requests(reqs, 0.0, dur, ttft_slo_ms, itl_slo_ms) if dur > 0 else {}
    good = sum(r.passes(ttft_slo_ms, itl_slo_ms) for r in reqs)
    return {"completed": rec.get("completed"), "n_loaded": len(reqs), "duration_s": dur,
            "output_tokens": out_tok, "input_tokens": in_tok, "good": good,
            "ttft_slo_ms": ttft_slo_ms, "itl_slo_ms": itl_slo_ms,
            "slo_goodput_req_s": s.get("slo_goodput_req_s"),
            "ttft_p95_ms": s.get("ttft_p95_ms"), "token_itl_p95_ms": s.get("token_itl_p95_ms"),
            "request_rate": rec.get("request_rate")}


# ---------------------------------------------------------------- unit-name metadata
def unit_meta(campaign: str, name: str) -> dict:
    if campaign == "n3_r3":
        m = re.match(r"(g|off|br|floor)_?(?:r(\d+)|b_r\d|head|tail)?_?(\d+)?", name)
        kind = {"g": "grid(PEV=1)", "off": "obs_off(PEV=0)", "br": "bridge(PEV=0)", "floor": "floor(PEV=1)"}[m.group(1)]
        rate = int(m.group(2)) / 100.0 if m.group(2) else (1.15 if m.group(1) == "br" else None)
        return {"kind": kind, "rate": rate, "seed": m.group(3), "shape": "B"}
    if campaign == "e2_r2":
        m = re.match(r"([ab])_r(\d)_(OFF|ON)_(\d+)", name)
        return {"shape": m.group(1).upper(), "cell": f"{m.group(1)}_r{m.group(2)}", "arm": f"sticky_{m.group(3)}",
                "seed": m.group(4)}
    if campaign == "lam0":
        m = re.match(r"([ab])_r(\d)(?:_s(\d))?", name)
        return {"shape": m.group(1).upper(), "rung": int(m.group(2)), "seed_rep": m.group(3) or "1"}
    return {}


def md_table(rows, cols, fmt=None) -> str:
    fmt = fmt or {}
    def cell(r, c):
        v = r.get(c)
        if v is None:
            return "-"
        if c in fmt:
            return fmt[c].format(v)
        if isinstance(v, float):
            return f"{v:.3g}" if abs(v) < 1 else f"{v:.1f}"
        return str(v)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(cell(r, c) for c in cols) + " |" for r in rows]
    return "\n".join(lines)


def dump(obj, path: Path):
    path.write_text(json.dumps(obj, indent=1, default=str, ensure_ascii=False))


# ---------------------------------------------------------------- energy.csv (NVML counter, job_entry >= 2026-10-08)
def parse_energy(path: Path):
    """-> sorted list of (epoch_s, energy_mJ) from meta/energy.csv (NVML total energy counter)."""
    rows = []
    with open(path) as f:
        next(f, None)
        for ln in f:
            try:
                t, e = ln.strip().split(",")
                rows.append((float(t), int(e)))
            except ValueError:
                continue
    rows.sort()
    return rows


def _counter_at(rows, t):
    import bisect
    ts = [r[0] for r in rows]
    i = bisect.bisect_left(ts, t)
    if i == 0 or i >= len(rows):
        return None                      # outside the sampled range: no extrapolation
    (t0, e0), (t1, e1) = rows[i - 1], rows[i]
    return e0 + (e1 - e0) * (t - t0) / (t1 - t0) if t1 > t0 else float(e0)


def counter_window_J(rows, a: float, b: float):
    """Board energy in [a, b] from the counter, linearly interpolated; None if [a, b] is not covered."""
    ea, eb = _counter_at(rows, a), _counter_at(rows, b)
    if ea is None or eb is None:
        return None
    return (eb - ea) / 1000.0
