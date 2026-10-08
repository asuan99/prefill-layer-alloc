#!/usr/bin/env python3
"""Rung A(1): energy per request / per good request / per output token from existing clk.csv.

    python3 pc_energy.py --all            # n3_r3, e2_r2, lam0 -> out/pc_energy.{json,md}
    python3 pc_energy.py --regress        # 0x4 % and SM MHz vs the campaigns' own analyzers
    python3 pc_energy.py --selftest

Descriptive only.  "good" uses the project predicate (analyze.RequestResult.passes: TTFT <= SLO AND
per-request token-ITL p95 <= SLO) with the SLO each campaign's analyzer used (N3: T_5 from T_K.json
and ITL 50 ms; E2/lambda0: 3000 / 50 ms).  J/good is undefined (None) when good == 0.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pc_common as C  # noqa: E402

CAVEATS = [
    "NVML power.draw (A100) is a ~1 s moving average; 100 ms samples are smoothed, window edges +-1 s.",
    "SW power cap exposure (bit 0x4) rises with offered rate (endogenous); no arm-delta attribution (LVR-2/VC-2/N3C-16).",
    "lambda0 windows are inferred from srv.log (second resolution) - window-level only.",
    "J/good depends on the SLO predicate; it is a secondary descriptive metric, not a decision input.",
    "Energy is whole-GPU; no per-partition (prefill vs decode) attribution is possible from NVML.",
]


def idle_reference_W(rows):
    idle = sorted(x[2] for x in rows if x[4] == 0x1)
    return C.percentile(idle, 0.05) if idle else None


def run_campaign(key: str, ttft_override=None, itl_override=None) -> dict:
    cfg = C.VESSL[key]
    out = cfg["out"]
    rows = C.parse_clk(cfg["clk"])
    epath = Path(cfg["clk"]).with_name("energy.csv")
    erows = C.parse_energy(epath) if epath.exists() else None
    idle_W = idle_reference_W(rows)
    if cfg["windows"] == "timing":
        wins = C.read_timing(out / "TIMING.tsv", cfg["timing_key"])
    else:
        wins = C.lam0_windows(out)
    ttft = cfg["slo"]["ttft_ms"]
    if isinstance(ttft, str):                       # N3: T_5 = 5 x floor TTFT
        tk = json.load(open(out / "T_K.json"))
        ttft = 1000.0 * float(tk["T_k_s"]["5"])
    if ttft_override is not None:
        ttft = ttft_override
    itl = itl_override if itl_override is not None else cfg["slo"]["itl_ms"]
    units = {}
    for u, w in wins.items():
        d = w.get("dir", out)
        bench = d / f"bench_{u}.jsonl"
        ws = C.window_stats(C.window(rows, w["a"], w["b"]))
        rec = {"meta": C.unit_meta(key, u), "window": {"a": w["a"], "b": w["b"], "len_s": w["b"] - w["a"]}, "clk": ws}
        if erows:
            ec = C.counter_window_J(erows, w["a"], w["b"])
            rec["energy_counter_J"] = ec
            if ec and ws.get("n"):
                rec["integral_vs_counter_pct"] = 100.0 * ws["energy_J"] / ec
        if bench.exists() and ws.get("n"):
            bs = C.bench_stats(bench, ttft, itl)
            e = rec.get("energy_counter_J") or ws["energy_J"]     # counter preferred when the Job has energy.csv
            rec["energy_source"] = "nvml_counter" if rec.get("energy_counter_J") else "power.draw_integral"
            rec["bench"] = bs
            rec["energy"] = {
                "J_per_request": e / bs["completed"] if bs["completed"] else None,
                "J_per_good_request": e / bs["good"] if bs["good"] else None,
                "J_per_output_token": e / bs["output_tokens"] if bs["output_tokens"] else None,
                "J_per_total_token": (e / (bs["output_tokens"] + bs["input_tokens"])
                                      if bs["output_tokens"] and bs["input_tokens"] else None),
                "mean_W": ws["mean_W"],
                "mean_W_above_idle": (ws["mean_W"] - idle_W) if idle_W is not None else None,
                "good_frac": bs["good"] / bs["n_loaded"] if bs["n_loaded"] else None,
            }
        units[u] = rec
    return {"campaign": key, "note": cfg["note"], "clk_path": str(cfg["clk"]), "n_clk_rows": len(rows),
            "energy_csv": str(epath) if erows else None,
            "idle_reference_W_p5": idle_W, "slo": {"ttft_ms": ttft, "itl_ms": itl}, "units": units}


def to_md(res: dict) -> str:
    parts = ["# pc_energy — energy per request/good/token (descriptive, 0 GPU-h)", ""]
    parts += [f"- caveat: {c}" for c in CAVEATS] + [""]
    for key, r in res.items():
        parts.append(f"## {key} — {r['note']}")
        parts.append(f"idle reference (p5 of GpuIdle samples): {r['idle_reference_W_p5']:.1f} W; "
                     f"SLO TTFT {r['slo']['ttft_ms']:.0f} ms / ITL {r['slo']['itl_ms']:.0f} ms; clk rows {r['n_clk_rows']}")
        rows = []
        for u, rec in r["units"].items():
            if "energy" not in rec:
                continue
            m, ws, bs, e = rec["meta"], rec["clk"], rec["bench"], rec["energy"]
            rows.append({"unit": u, "meta": "/".join(str(v) for v in m.values() if v is not None),
                         "win_s": rec["window"]["len_s"], "completed": bs["completed"], "good": bs["good"],
                         "req_rate": bs.get("request_rate"),
                         "mean_W": ws["mean_W"], "p95_W": ws["p95_W"], "cap_pct": ws["sw_power_cap_pct"],
                         "sm_MHz": ws["sm_mhz_mean"], "E_kJ": ws["energy_J"] / 1000.0,
                         "J/req": e["J_per_request"], "J/good": e["J_per_good_request"],
                         "J/out_tok": e["J_per_output_token"], "J/tok": e["J_per_total_token"]})
        def sk(x):
            m = r["units"][x["unit"]]["meta"]
            return (str(m.get("shape", "")), str(m.get("cell", m.get("rung", ""))), float(m.get("rate") or 0), x["unit"])
        rows.sort(key=sk)
        parts.append(C.md_table(rows, ["unit", "meta", "win_s", "completed", "good", "req_rate", "mean_W", "p95_W",
                                       "cap_pct", "sm_MHz", "E_kJ", "J/req", "J/good", "J/out_tok", "J/tok"],
                                {"win_s": "{:.0f}", "mean_W": "{:.0f}", "p95_W": "{:.0f}", "cap_pct": "{:.1f}",
                                 "sm_MHz": "{:.0f}", "E_kJ": "{:.1f}", "J/req": "{:.0f}", "J/good": "{:.0f}",
                                 "J/out_tok": "{:.2f}", "J/tok": "{:.3f}"}))
        parts.append("")
    return "\n".join(parts)


def regress(res: dict) -> dict:
    """Cross-check 0x4 % / SM MHz against the campaigns' own analyzers (same windows, same method)."""
    out = {}
    n3 = C.VESSL["n3_r3"]["out"] / "N3_RESULT.json"
    if n3.exists() and "n3_r3" in res:
        ref = json.load(open(n3)).get("substrate", {}).get("clk") or {}
        diffs = []
        for u, v in ref.items():
            mine = res["n3_r3"]["units"].get(u, {}).get("clk", {})
            if mine.get("n"):
                diffs.append((u, abs(v["sw_power_cap_pct"] - mine["sw_power_cap_pct"]),
                              abs(v["sm_mhz_mean"] - mine["sm_mhz_mean"])))
        out["n3_r3_vs_N3_RESULT"] = {"n_units": len(diffs),
                                     "max_abs_diff_cap_pct": max((d[1] for d in diffs), default=None),
                                     "max_abs_diff_sm_mhz": max((d[2] for d in diffs), default=None)}
    lam = C.VESSL["lam0"]["out"].parent / "analysis_2026-10-04" / "lam0v_descriptive.json"
    if lam.exists() and "lam0" in res:
        ref = json.load(open(lam))
        diffs = []
        for c, v in ref["clk"]["cells"].items():
            bw = v["bench_window"]
            mine = res["lam0"]["units"].get(c, {}).get("clk", {})
            if bw.get("n") and mine.get("n"):
                diffs.append((c, abs(bw["bit_pct"].get("SwPowerCap", 0.0) - mine["sw_power_cap_pct"]),
                              abs(bw["power_w"]["p50"] - mine["p50_W"]), abs(bw["n"] - mine["n"])))
        out["lam0_vs_lam0v_descriptive"] = {"n_cells": len(diffs),
                                            "max_abs_diff_cap_pct": max((d[1] for d in diffs), default=None),
                                            "max_abs_diff_p50_W": max((d[2] for d in diffs), default=None),
                                            "max_abs_diff_n_samples": max((d[3] for d in diffs), default=None)}
    return out


# ---------------------------------------------------------------- selftest
def _synth(td: Path):
    import datetime as dt
    t0 = dt.datetime(2026, 10, 7, 0, 0, 0, tzinfo=dt.timezone.utc)
    lines = ["timestamp, clocks.current.sm [MHz], power.draw [W], temperature.gpu, clocks_event_reasons.active"]
    for i in range(300):                               # 30 s at 100 ms
        t = t0 + dt.timedelta(milliseconds=100 * i)
        if i < 100:
            sm, pw, v = 1410, 200.0, "0x0000000000000000"
        elif i < 200:
            sm, pw, v = 1300, 100.0, "0x0000000000000004"
        else:
            sm, pw, v = 1410, 150.0, "0x0000000000000010"   # SyncBoost: not a power cap
        lines.append(f"{t.strftime('%Y/%m/%d %H:%M:%S.%f')[:-3]}, {sm} MHz, {pw:.2f} W, 40, {v}")
    (td / "clk.csv").write_text("\n".join(lines) + "\n")
    e0 = t0.timestamp()
    (td / "TIMING.tsv").write_text("unit\tbench_start_epoch\tbench_end_epoch\n"
                                   f"w1\t{e0}\t{e0 + 9.95}\nw2\t{e0 + 10}\t{e0 + 19.95}\nw3\t{e0 + 20}\t{e0 + 29.95}\n")
    rec = {"duration": 10.0, "completed": 2, "ttfts": [0.1, 5.0], "itls": [[0.01] * 5, [0.01] * 5],
           "total_output_tokens": 12, "total_input_tokens": 100, "request_rate": 1.0}
    for u in ("w1", "w2", "w3"):
        (td / f"bench_{u}.jsonl").write_text(json.dumps(rec) + "\n")
    return e0


def selftest() -> int:
    fails = []
    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        _synth(td)
        rows = C.parse_clk(td / "clk.csv")
        wins = C.read_timing(td / "TIMING.tsv", "unit")
        s1 = C.window_stats(C.window(rows, wins["w1"]["a"], wins["w1"]["b"]))
        s2 = C.window_stats(C.window(rows, wins["w2"]["a"], wins["w2"]["b"]))
        s3 = C.window_stats(C.window(rows, wins["w3"]["a"], wins["w3"]["b"]))
        if not (abs(s1["energy_J"] - 2000.0) < 40):
            fails.append(f"energy w1 {s1['energy_J']:.1f} != 2000+-40")
        if not (abs(s2["energy_J"] - 1000.0) < 20):
            fails.append(f"energy w2 {s2['energy_J']:.1f} != 1000+-20")
        if not (s1["sw_power_cap_pct"] == 0.0 and s2["sw_power_cap_pct"] == 100.0 and s3["sw_power_cap_pct"] == 0.0):
            fails.append(f"cap pct {s1['sw_power_cap_pct']},{s2['sw_power_cap_pct']},{s3['sw_power_cap_pct']}")
        if "SyncBoost" not in s3["other_reason_pct"]:
            fails.append("SyncBoost bit not reported as other reason")
        if s2["sm_mhz_mean"] != 1300:
            fails.append("sm mean w2")
        bs = C.bench_stats(td / "bench_w1.jsonl", 3000.0, 50.0)
        if bs["good"] != 1 or bs["completed"] != 2:
            fails.append(f"bench good/completed {bs['good']}/{bs['completed']}")
        jg = s1["energy_J"] / bs["good"]
        jr = s1["energy_J"] / bs["completed"]
        if not (abs(jg - 2 * jr) < 1e-6):
            fails.append("J/good != 2 x J/req with 1 of 2 passing")
        # NVML counter path: 250 W constant for 30 s, sampled every 0.1 s -> window [5, 15] = 2500 J
        e0 = wins["w1"]["a"]
        (td / "energy.csv").write_text("epoch_s,energy_mJ\n" + "".join(f"{e0 + 0.1*i:.3f},{int(250_000 * 0.1 * i)}\n" for i in range(301)))
        er = C.parse_energy(td / "energy.csv")
        cj = C.counter_window_J(er, e0 + 5.0, e0 + 15.0)
        if cj is None or abs(cj - 2500.0) > 1.0:
            fails.append(f"counter window {cj} != 2500")
        if C.counter_window_J(er, e0 - 1.0, e0 + 5.0) is not None:
            fails.append("counter window outside sampled range must be None (no extrapolation)")
        cj2 = C.counter_window_J(er, e0 + 5.05, e0 + 15.05)
        if cj2 is None or abs(cj2 - 2500.0) > 1.0:
            fails.append(f"interpolated counter window {cj2} != 2500")
        # negative control: a window with no samples must not produce energy
        if C.window_stats([]) != {"n": 0}:
            fails.append("empty window produced stats")
        # mutation: swap power and temperature columns -> energy must change (parser is column-bound)
        txt = (td / "clk.csv").read_text().splitlines()
        mut = [txt[0]] + [", ".join([p[0], p[1], f"{p[3]}.00 W", str(int(float(p[2].split()[0]))), p[4]]) for p in
                          (ln.split(", ") for ln in txt[1:])]
        (td / "clk_mut.csv").write_text("\n".join(mut) + "\n")
        m1 = C.window_stats(C.window(C.parse_clk(td / "clk_mut.csv"), wins["w1"]["a"], wins["w1"]["b"]))
        if abs(m1["energy_J"] - s1["energy_J"]) < 100:
            fails.append("mutant (swapped columns) not detected")
    print("SELFTEST", "PASS" if not fails else "FAIL", fails)
    return 0 if not fails else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--campaign", choices=list(C.VESSL), action="append")
    ap.add_argument("--regress", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--ttft-slo-ms", type=float)
    ap.add_argument("--itl-slo-ms", type=float)
    ap.add_argument("--out-dir", default=str(C.HERE / "out"))
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    keys = list(C.VESSL) if a.all or not a.campaign else a.campaign
    res = {k: run_campaign(k, a.ttft_slo_ms, a.itl_slo_ms) for k in keys}
    od = Path(a.out_dir); od.mkdir(parents=True, exist_ok=True)
    payload = {"caveats": CAVEATS, "results": res}
    if a.regress:
        payload["regress"] = regress(res)
        print(json.dumps(payload["regress"], indent=1))
    C.dump(payload, od / "pc_energy.json")
    (od / "pc_energy.md").write_text(to_md(res) + ("\n\n## regress\n```\n" + json.dumps(payload.get("regress"), indent=1) + "\n```\n" if a.regress else ""))
    for k, r in res.items():
        print(k, "units", len(r["units"]), "with energy", sum(1 for u in r["units"].values() if "energy" in u))


if __name__ == "__main__":
    main()
