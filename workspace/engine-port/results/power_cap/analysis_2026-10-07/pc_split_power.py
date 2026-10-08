#!/usr/bin/env python3
"""Rung A(2): power / clock vs SM split, descriptive.

(a) KISTI 5 s sweeps (`*_clocks.csv`: `clocks.sm, power.draw, utilization.gpu`, no timestamps) across
    s0_deconfound / s8_scaleup / s8p_prefill / s8_frontier -- split comes from the file name;
    busy mask = utilization.gpu >= BUSY_UTIL.  Realised partition is NOT verified here (Stage-0
    lesson: pin is verified by realized, not target -- see CONSENSUS §3); a partitioned arm that
    draws ~400 W is itself a hint that the partition did not hold.
(b) VESSL lambda0 cells: bench-window power vs realised D44 time share (mix_<cell>.json e_time).
(c) VESSL E2 r2: sticky OFF vs ON per (cell, seed): power/cap %/SM MHz, paired by seed.

    python3 pc_split_power.py --all | --selftest
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pc_common as C  # noqa: E402

BUSY_UTIL = 90
NEAR_CAP_W = 0.95 * C.A100_SXM_POWER_LIMIT_W
ARM_LEGEND = {
    # s0_deconfound (s0dc_sweep.sbatch:35-37, ARM_LABELS=(M H T)), 3B-class
    "M": "pure-Mamba2 (s0 negative control)", "H": "Zamba2 hybrid (s0 target)", "T": "Qwen2.5-3B attn (s0 positive control)",
    # s8 / s8p / e1 (s8_sweep.sbatch:22-23, ARM_LABELS=(M8 Ha8 Hs8 T8)), 7-8B-class
    "M8": "Mamba-Codestral-7B pure-Mamba2", "Ha8": "Zamba2-7B additive hybrid",
    "Hs8": "Nemotron-H-8B substitutive hybrid", "T8": "Qwen2.5-7B attention",
}
PATTERNS = [
    ("s0_deconfound", re.compile(r"s0dc_(?P<arm>[A-Z]+)_C(?P<ctx>\d+)_P(?P<p>\d+)D(?P<d>\d+)_(?P<job>\d+)_rep(?P<rep>\d+)_clocks\.csv$")),
    ("s8_scaleup", re.compile(r"s8_(?P<mode>coupled|deconf)_(?P<arm>[A-Za-z0-9]+)_C(?P<ctx>\d+)_(?P<split>d\d+|np)_(?P<job>\d+)_(?:blk(?P<blk>\d+)_)?rep(?P<rep>\d+)_clocks\.csv$")),
    ("s8_scaleup", re.compile(r"s8c2r_(?P<arm>[A-Za-z0-9]+)_C(?P<ctx>\d+)_(?P<split>d\d+)_(?P<job>\d+)_blk(?P<blk>\d+)_clocks\.csv$")),
    ("s8p_prefill", re.compile(r"s8p_(?P<arm>[A-Za-z0-9]+)_p(?P<p>\d+)_(?P<job>\d+)_rep(?P<rep>\d+)_clocks\.csv$")),
    ("s8_frontier", re.compile(r"e1_(?P<arm>[A-Za-z0-9]+)_d(?P<d>\d+)_(?P<job>\d+)_clocks\.csv$")),
]
KISTI_DIRS = ["s0_deconfound", "s8_scaleup", "s8p_prefill", "s8_frontier"]


def classify(path: Path):
    name = path.name
    for camp, rx in PATTERNS:
        m = rx.match(name)
        if m:
            g = m.groupdict()
            if name.startswith("s0dc"):
                split, mode = f"P{g['p']}/D{g['d']}", "deconf(prefill pinned)"
            elif name.startswith("s8c2r"):
                split, mode = f"P16/D{g['split'][1:]}", "c2r(re-measure, prefill pinned 16)"
            elif name.startswith("s8_"):
                if g["split"] == "np":
                    split = "np(108, unpartitioned)"
                elif g["mode"] == "coupled":
                    split = f"P{108 - int(g['split'][1:])}/D{g['split'][1:]}"
                else:
                    split = f"P16/D{g['split'][1:]}"
                mode = g["mode"]
            elif name.startswith("s8p"):
                split, mode = f"P{g['p']}/D16", "prefill-axis(decode pinned 16)"
            else:
                split, mode = f"D{g['d']}", "e1 frontier"
            return {"campaign": camp, "arm": g["arm"], "ctx": g.get("ctx", "-"), "split": split, "mode": mode,
                    "job": g["job"], "file": name}
    return None


def read_kisti(path: Path, busy_util: int = BUSY_UTIL):
    rows = []
    for ln in open(path):
        p = [x.strip() for x in ln.split(",")]
        if len(p) < 3:
            continue
        try:
            rows.append((int(float(p[0])), float(p[1]), int(float(p[2]))))
        except ValueError:
            continue
    busy = [r for r in rows if r[2] >= busy_util]
    out = {"n": len(rows), "n_busy": len(busy)}
    if busy:
        pw = sorted(r[1] for r in busy); sm = sorted(r[0] for r in busy)
        out |= {"busy_mean_W": statistics.fmean(pw), "busy_max_W": pw[-1], "busy_p95_W": C.percentile(pw, 0.95),
                "busy_min_MHz": sm[0], "busy_frac_near_cap": sum(1 for p in pw if p >= NEAR_CAP_W) / len(pw),
                "busy_frac_MHz_lt_1400": sum(1 for s in sm if s < 1400) / len(sm)}
    return out


def kisti_table(res_root: Path = C.RES, busy_util: int = BUSY_UTIL):
    groups = defaultdict(list)
    unmatched = []
    for d in KISTI_DIRS:
        for f in sorted((res_root / d).glob("*_clocks.csv")):
            meta = classify(f)
            if meta is None:
                unmatched.append(f.name); continue
            st = read_kisti(f, busy_util)
            if st["n_busy"] == 0:
                continue
            groups[(meta["campaign"], meta["mode"], meta["arm"], meta["ctx"], meta["split"])].append(st)
    rows = []
    for (camp, mode, arm, ctx, split), sts in sorted(groups.items()):
        pooled_near = sum(s["busy_frac_near_cap"] * s["n_busy"] for s in sts) / sum(s["n_busy"] for s in sts)
        pooled_lt = sum(s["busy_frac_MHz_lt_1400"] * s["n_busy"] for s in sts) / sum(s["n_busy"] for s in sts)
        rows.append({"campaign": camp, "mode": mode, "arm": arm, "arm_legend": ARM_LEGEND.get(arm, "?"), "ctx": ctx,
                     "split": split, "n_files": len(sts), "n_busy": sum(s["n_busy"] for s in sts),
                     "busy_mean_W": statistics.fmean(s["busy_mean_W"] for s in sts),
                     "busy_max_W": max(s["busy_max_W"] for s in sts),
                     "busy_min_MHz": min(s["busy_min_MHz"] for s in sts),
                     "frac_near_cap": pooled_near, "frac_MHz_lt_1400": pooled_lt})
    return rows, unmatched


def lam0_table():
    cfg = C.VESSL["lam0"]
    rows_clk = C.parse_clk(cfg["clk"])
    wins = C.lam0_windows(cfg["out"])
    rows = []
    for c, w in wins.items():
        ws = C.window_stats(C.window(rows_clk, w["a"], w["b"]))
        mix = w["dir"] / f"mix_{c}.json"
        mj = json.load(open(mix)) if mix.exists() else {}
        cell = w["row"]
        rows.append({"cell": c, "shape": cell.get("shape"), "offered": cell.get("offered_rate"),
                     "achieved": cell.get("achieved_rate"), "e_time_D44_pct": (mj.get("e_time") or {}).get("pct"),
                     "idx0_pct": (mj.get("idx0") or {}).get("pct"),
                     "n": ws.get("n"), "mean_W": ws.get("mean_W"), "p50_W": ws.get("p50_W"), "p95_W": ws.get("p95_W"),
                     "cap_pct": ws.get("sw_power_cap_pct"), "sm_MHz": ws.get("sm_mhz_mean"),
                     "ttft_p95_s": cell.get("ttft_p95_s"), "itl_p95_s": cell.get("itl_p95_s")})
    rows.sort(key=lambda r: (r["shape"], r["offered"] or 0, r["cell"]))
    return rows


def e2_table():
    cfg = C.VESSL["e2_r2"]
    rows_clk = C.parse_clk(cfg["clk"])
    wins = C.read_timing(cfg["out"] / "TIMING.tsv", "boot")
    per = {}
    for b, w in wins.items():
        ws = C.window_stats(C.window(rows_clk, w["a"], w["b"]))
        m = C.unit_meta("e2_r2", b)
        mix = cfg["out"] / f"mix_{b}.json"
        mj = json.load(open(mix)) if mix.exists() else {}
        per[b] = {"boot": b, **m, "n": ws.get("n"), "mean_W": ws.get("mean_W"), "p50_W": ws.get("p50_W"),
                  "p95_W": ws.get("p95_W"), "cap_pct": ws.get("sw_power_cap_pct"), "sm_MHz": ws.get("sm_mhz_mean"),
                  "e_time_D44_pct": (mj.get("e_time") or {}).get("pct")}
    pairs = []
    for b, r in per.items():
        if r["arm"] != "sticky_OFF":
            continue
        on = per.get(b.replace("_OFF_", "_ON_"))
        if on and r["n"] and on["n"]:
            pairs.append({"cell": r["cell"], "seed": r["seed"], "OFF_mean_W": r["mean_W"], "ON_mean_W": on["mean_W"],
                          "dW_ON_minus_OFF": on["mean_W"] - r["mean_W"], "OFF_cap_pct": r["cap_pct"],
                          "ON_cap_pct": on["cap_pct"], "OFF_sm_MHz": r["sm_MHz"], "ON_sm_MHz": on["sm_MHz"],
                          "OFF_e_time": r["e_time_D44_pct"], "ON_e_time": on["e_time_D44_pct"]})
    pairs.sort(key=lambda p: (p["cell"], p["seed"]))
    return list(per.values()), pairs


def to_md(k_rows, unmatched, l_rows, e_rows, e_pairs) -> str:
    out = ["# pc_split_power — power/clock vs SM split (descriptive, 0 GPU-h)", "",
           f"KISTI busy mask: utilization.gpu >= {BUSY_UTIL}; near-cap = power >= {NEAR_CAP_W:.0f} W "
           f"(0.95 x 400 W, A100-SXM4-80GB on both substrates). KISTI files have NO timestamps (5 s cadence, "
           "~17-20 rows per file); realised partition is not verified by this script.", ""]
    out.append("## (a) KISTI sweeps — busy-sample power by split")
    out.append(C.md_table(k_rows, ["campaign", "mode", "arm", "ctx", "split", "n_files", "n_busy", "busy_mean_W",
                                   "busy_max_W", "busy_min_MHz", "frac_near_cap", "frac_MHz_lt_1400"],
                          {"busy_mean_W": "{:.0f}", "busy_max_W": "{:.0f}", "frac_near_cap": "{:.2f}", "frac_MHz_lt_1400": "{:.2f}"}))
    out.append(f"\nunmatched file names: {len(unmatched)}" + (f" e.g. {unmatched[:5]}" if unmatched else ""))
    out.append("\narm legend: " + "; ".join(f"{k}={v}" for k, v in ARM_LEGEND.items()))
    out.append("\n## (b) VESSL lambda0 cells — bench-window power vs realised D44 time share")
    out.append(C.md_table(l_rows, ["cell", "shape", "offered", "achieved", "e_time_D44_pct", "idx0_pct", "n", "mean_W", "p50_W",
                                   "p95_W", "cap_pct", "sm_MHz", "ttft_p95_s", "itl_p95_s"],
                          {"mean_W": "{:.0f}", "p50_W": "{:.0f}", "p95_W": "{:.0f}", "cap_pct": "{:.1f}", "sm_MHz": "{:.0f}",
                           "e_time_D44_pct": "{:.1f}", "idx0_pct": "{:.1f}", "achieved": "{:.3f}"}))
    out.append("\n## (c) VESSL E2 r2 — sticky OFF vs ON, paired by (cell, seed)")
    out.append(C.md_table(e_pairs, ["cell", "seed", "OFF_mean_W", "ON_mean_W", "dW_ON_minus_OFF", "OFF_cap_pct", "ON_cap_pct",
                                    "OFF_sm_MHz", "ON_sm_MHz", "OFF_e_time", "ON_e_time"],
                          {"OFF_mean_W": "{:.0f}", "ON_mean_W": "{:.0f}", "dW_ON_minus_OFF": "{:+.0f}", "OFF_cap_pct": "{:.1f}",
                           "ON_cap_pct": "{:.1f}", "OFF_sm_MHz": "{:.0f}", "ON_sm_MHz": "{:.0f}", "OFF_e_time": "{:.1f}",
                           "ON_e_time": "{:.1f}"}))
    out.append("\nForbidden readings: arm-delta attribution to power cap (LVR-2/VC-2/N3C-16); 'SM split partitions power' "
               "as a mechanism claim (no per-partition power attribution exists); any policy statement.")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------- selftest
def selftest() -> int:
    fails = []
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        for sub in KISTI_DIRS:
            (root / sub).mkdir()
        def w(name, rows):
            (root / "s8_scaleup" / name).write_text("".join(f"{sm}, {pw:.2f}, {u}\n" for sm, pw, u in rows))
        w("s8_deconf_Hs8_C1024_d16_111111_rep1_clocks.csv", [(1155, 70, 0)] + [(1410, 200, 99)] * 10)
        w("s8_deconf_Hs8_C1024_d16_111111_rep2_clocks.csv", [(1155, 70, 3)] + [(1410, 220, 100)] * 10)
        w("s8_deconf_Hs8_C1024_np_111111_rep1_clocks.csv", [(1155, 70, 0)] + [(1335, 390, 100)] * 5 + [(1410, 385, 99)] * 5)
        w("s8_deconf_Hs8_C1024_d44_111111_rep1_clocks.csv", [(1155, 70, 0)] * 12)      # never busy -> excluded
        (root / "s8_scaleup" / "garbage_clocks.csv").write_text("1410, 100, 100\n")
        rows, unmatched = kisti_table(root)
        by = {r["split"]: r for r in rows}
        if set(by) != {"P16/D16", "np(108, unpartitioned)"}:
            fails.append(f"groups {sorted(by)}")
        else:
            if abs(by["P16/D16"]["busy_mean_W"] - 210.0) > 1e-6 or by["P16/D16"]["n_files"] != 2:
                fails.append("d16 mean/n_files")
            if by["P16/D16"]["frac_near_cap"] != 0.0 or by["np(108, unpartitioned)"]["frac_near_cap"] != 1.0:
                fails.append("near-cap fractions")
            if by["np(108, unpartitioned)"]["busy_min_MHz"] != 1335 or by["np(108, unpartitioned)"]["frac_MHz_lt_1400"] != 0.5:
                fails.append("np clock stats")
        if unmatched != ["garbage_clocks.csv"]:
            fails.append(f"unmatched {unmatched}")
        # mutation: dropping the busy mask must pull idle rows into the mean (mask is load-bearing)
        rows0, _ = kisti_table(root, busy_util=0)
        m0 = {r["split"]: r for r in rows0}["P16/D16"]["busy_mean_W"]
        if abs(m0 - 210.0) < 1.0:
            fails.append("busy mask mutation not detected")
        # classify() coverage of every real naming pattern seen on 2026-10-07
        for n in ["s0dc_H_C4096_P16D16_865006_rep1_clocks.csv", "s8_coupled_T8_C4096_np_865312_rep2_clocks.csv",
                  "s8_deconf_T8_C1024_d54_870001_blk2_rep1_clocks.csv", "s8c2r_Ha8_C1024_d92_883575_blk6_clocks.csv",
                  "s8p_M8_p44_865973_rep3_clocks.csv", "e1_T8_d92_865832_clocks.csv"]:
            if classify(Path(n)) is None:
                fails.append(f"classify miss {n}")
        if classify(Path("s8_coupled_T8_C4096_d92_865312_rep2_clocks.csv"))["split"] != "P16/D92":
            fails.append("coupled split arithmetic")
    print("SELFTEST", "PASS" if not fails else "FAIL", fails)
    return 0 if not fails else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out-dir", default=str(C.HERE / "out"))
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    k_rows, unmatched = kisti_table()
    l_rows = lam0_table()
    e_rows, e_pairs = e2_table()
    od = Path(a.out_dir); od.mkdir(parents=True, exist_ok=True)
    C.dump({"kisti": k_rows, "kisti_unmatched": unmatched, "lam0": l_rows, "e2_per_boot": e_rows, "e2_pairs": e_pairs},
           od / "pc_split_power.json")
    (od / "pc_split_power.md").write_text(to_md(k_rows, unmatched, l_rows, e_rows, e_pairs))
    print("kisti groups", len(k_rows), "unmatched", len(unmatched), "| lam0 cells", len(l_rows), "| e2 pairs", len(e_pairs))


if __name__ == "__main__":
    main()
