"""
projection_analysis.py — 실측 불가능한 end-to-end 이득의 PROJECTION.

Stage 1 saturation + Stage 2 overhead + Stage 2.2 coexec overlap 으로
layer-type-aware SM 할당의 prefill throughput 이득 상한을 projection 한다.

중요: 이 스크립트의 모든 출력은 PROJECTION 이다 — 측정값이 아니다.
  - 모든 figure 제목, caption, 파일명에 "PROJECTION" 하드코딩.
  - 측정값(measured)과 같은 figure에 혼재 금지.
  - 점선(dashed) + 회색 계열 + "(projected)" 라벨로 스타일링.

입력:
  S1.3  free_sm_zone_{model}.csv   (free_sm_fraction, decode_sm_sensitivity)
  Stage2 ctx_switch_{model}.csv    (ctx_switch_overhead_us) 또는 decision_matrix.json
  S2.2  coexec_{model}.csv         (overlap_ratio per sm_split, backend)

projection 공식 및 가정 (모두 리포트에 출력):

  Assumption A1: prefill kernel 의 SM 이득이 선형.
    speedup_raw = 1 / (1 - free_sm_fraction)   [상한]
    실제로는 비선형이며 saturation 이후 plateau. 따라서 이 projection 은 상한.

  Assumption A2: free SM 이 decode 에 완전히 재사용됨.
    decode 부담이 free SM 에 이상적으로 배분된다고 가정.
    현실에서는 context switch overhead 로 인해 일부 손실 발생.

  Assumption A3: overlap_ratio 가 단일 (seq_len, batch, sm_split) 설정에서
    전체 serving 워크로드를 대표함.
    실제로는 혼합 워크로드에서 다를 수 있음.

  net_speedup_prefill = speedup_raw × overlap_ratio_gc
  overhead_penalty    = ctx_switch_overhead_us / (prefill_layer_latency_us)
  net_gain_fraction   = net_speedup_prefill - 1.0 - overhead_penalty

  G1-gate: net_gain_fraction > 0 이어야 SM 재할당이 worthwhile.

Usage:
    # 기본: demo 데이터로 figure 생성
    python stage3_hm_eval/projection_analysis.py --demo

    # 실측 데이터 사용
    python stage3_hm_eval/projection_analysis.py \\
        --free-sm-csv  results/stage1/free_sm_zone_zamba2_a100.csv \\
        --coexec-csv   results/stage3/coexec_zamba2_a100.csv \\
        --stage2-json  results/stage2/decision_matrix.json \\
        --model zamba2

Output (모두 파일명에 PROJECTION 포함):
    results/stage3/projection_speedup_{model}.png
    results/stage3/projection_net_gain_{model}.png
    results/stage3/projection_report_{model}.txt
"""

from __future__ import annotations

import sys
import os
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "../.."))
sys.path.insert(0, os.path.join(_here, ".."))

import argparse
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Assumptions list (printed in every report)
# ---------------------------------------------------------------------------

ASSUMPTIONS = [
    (
        "A1: Prefill SM speedup is linear in free_sm_fraction (upper bound).\n"
        "    speedup_raw = 1 / (1 - free_sm_fraction). Real behaviour saturates\n"
        "    after SM count exceeds compute saturation; this projection is an upper bound."
    ),
    (
        "A2: Free SMs are fully reallocated to decode with zero fragmentation.\n"
        "    In reality, GPC-granular allocation rounds SM counts upward;\n"
        "    partial SMs are wasted."
    ),
    (
        "A3: Measured overlap_ratio (from coexec_microbench, single isolated kernel pair)\n"
        "    is representative of the full serving workload mixture.\n"
        "    Real workloads have variable seq_len and batch; overlap may differ."
    ),
    (
        "A4: ctx_switch_overhead is the only latency penalty per layer boundary.\n"
        "    Other serialization costs (e.g. stream synchronization, TLB flush) ignored."
    ),
    (
        "A5: SSM layer fraction is fixed (from model config). In practice,\n"
        "    layer mix during chunked prefill may vary with batch composition."
    ),
]


# ---------------------------------------------------------------------------
# Demo data generators
# ---------------------------------------------------------------------------

def _demo_free_sm(seq_lens, model_name="zamba2"):
    """Synthetic free_sm_fraction vs seq_len for SSM and Attn."""
    rows = []
    for sl in seq_lens:
        # SSM: memory-BW bound → large free zone, saturates early
        ssm_free = max(0.05, 0.75 - 0.25 * math.log2(max(1, sl) / 256) / 8)
        # Attn: compute-bound at long seq → small free zone
        attn_free = max(0.02, 0.40 - 0.35 * math.log2(max(1, sl) / 256) / 8)
        rows.append({
            "model_name": model_name, "seq_len": sl,
            "layer_type": "ssm",  "free_ratio": round(ssm_free, 4),
            "decode_sm_sensitivity": 0.3,
        })
        rows.append({
            "model_name": model_name, "seq_len": sl,
            "layer_type": "attn", "free_ratio": round(attn_free, 4),
            "decode_sm_sensitivity": 0.15,
        })
    return rows


def _demo_overlap(sm_splits, seq_lens):
    """Synthetic overlap_ratio for green_ctx vs two_stream."""
    rows = []
    for sl in seq_lens:
        solo_sum = 10.0 + sl / 400  # ms
        for spl in sm_splits:
            # Green Context: better overlap, especially at balanced splits
            overlap_gc = 0.55 + 0.20 * abs(spl - 0.5) * 2
            # Two-stream: near 1.0 (hardware can't always co-schedule)
            overlap_ts = min(0.95, 0.80 + 0.15 * abs(spl - 0.5))
            rows.append({
                "seq_len": sl, "sm_split": spl,
                "backend": "green_ctx",  "overlap_ratio": round(overlap_gc, 4),
                "solo_sum_ms": round(solo_sum, 2),
            })
            rows.append({
                "seq_len": sl, "sm_split": spl,
                "backend": "two_stream", "overlap_ratio": round(overlap_ts, 4),
                "solo_sum_ms": round(solo_sum, 2),
            })
    return rows


def _demo_overhead():
    """Synthetic overhead data."""
    return {"ctx_switch_overhead_us": 12.5, "prefill_layer_latency_ms": 2.5}


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_free_sm_csv(path: Path) -> list[dict]:
    """Load free_sm_zone CSV from plot_saturation.py or Stage 1 sweep."""
    try:
        import pandas as pd
        df = pd.read_csv(path)
        return df.to_dict("records")
    except Exception as e:
        print(f"  [WARN] Could not load free-SM CSV {path}: {e}")
        return []


def load_coexec_csv(path: Path) -> list[dict]:
    """Load coexec_*.csv from coexec_microbench.py."""
    try:
        import pandas as pd
        df = pd.read_csv(path)
        return df.to_dict("records")
    except Exception as e:
        print(f"  [WARN] Could not load coexec CSV {path}: {e}")
        return []


def load_stage2_overhead(json_path: Optional[Path], csv_glob: Optional[str]) -> dict:
    """Load Stage 2 ctx-switch overhead from decision_matrix.json or CSV.

    Returns dict with 'ctx_switch_overhead_us' and 'prefill_layer_latency_ms'.
    """
    result = {"ctx_switch_overhead_us": None, "prefill_layer_latency_ms": None}

    if json_path is not None and json_path.exists():
        try:
            with open(json_path) as f:
                dm = json.load(f)
            result["ctx_switch_overhead_us"] = dm.get("ctx_switch_overhead_us")
            result["prefill_layer_latency_ms"] = dm.get("prefill_layer_latency_ms")
            print(f"  Loaded Stage 2 overhead from {json_path}")
        except Exception as e:
            print(f"  [WARN] Could not parse {json_path}: {e}")

    if csv_glob:
        import glob
        import pandas as pd
        files = glob.glob(csv_glob)
        if files:
            dfs = [pd.read_csv(f) for f in files]
            df = pd.concat(dfs, ignore_index=True)
            if "ctx_switch_overhead_us" in df.columns:
                result["ctx_switch_overhead_us"] = float(df["ctx_switch_overhead_us"].median())
            if "layer_latency_ms" in df.columns:
                result["prefill_layer_latency_ms"] = float(df["layer_latency_ms"].median())

    return result


# ---------------------------------------------------------------------------
# Projection core
# ---------------------------------------------------------------------------

def compute_projections(
    free_sm_rows: list[dict],
    coexec_rows: list[dict],
    overhead: dict,
) -> list[dict]:
    """Compute projection rows from inputs.

    Returns list of dicts, one per (seq_len, layer_type, sm_split, backend).
    All numeric values are projections based on the assumptions listed above.
    """
    import pandas as pd

    if not free_sm_rows or not coexec_rows:
        return []

    df_free = pd.DataFrame(free_sm_rows)
    df_coex = pd.DataFrame(coexec_rows)

    ctx_overhead_us = overhead.get("ctx_switch_overhead_us") or 0.0
    layer_lat_ms    = overhead.get("prefill_layer_latency_ms") or 1.0
    layer_lat_us    = layer_lat_ms * 1000.0

    # Per-layer overhead penalty (fraction of layer time lost to ctx switch)
    overhead_penalty = ctx_overhead_us / max(layer_lat_us, 1.0) if layer_lat_us > 0 else 0.0

    # Column name normalisation
    free_col = (
        "free_ratio" if "free_ratio" in df_free.columns
        else "free_sm_fraction" if "free_sm_fraction" in df_free.columns
        else None
    )
    if free_col is None:
        print("  [WARN] free_sm CSV missing 'free_ratio'/'free_sm_fraction' column")
        return []

    type_col = (
        "layer_type" if "layer_type" in df_free.columns
        else "ssm_type" if "ssm_type" in df_free.columns
        else None
    )
    if type_col is None:
        df_free["layer_type"] = "ssm"
        type_col = "layer_type"

    rows = []
    seq_lens = sorted(df_free["seq_len"].unique()) if "seq_len" in df_free.columns else [2048]

    for seq_len in seq_lens:
        free_sub = df_free[df_free["seq_len"] == seq_len] if "seq_len" in df_free.columns else df_free

        for _, free_row in free_sub.iterrows():
            layer_type = free_row.get(type_col, "ssm")
            free_frac  = float(free_row[free_col])
            decode_sens = float(free_row.get("decode_sm_sensitivity", 0.3))

            # A1: raw speedup upper bound
            speedup_raw = 1.0 / max(1.0 - free_frac, 0.01)

            # A3: use overlap_ratio from coexec for green_ctx backend
            coex_sub = df_coex[
                (df_coex.get("seq_len", df_coex.iloc[:, 0]) == seq_len)
                if "seq_len" in df_coex.columns else pd.Series(True, index=df_coex.index)
            ].copy() if "seq_len" in df_coex.columns else df_coex.copy()

            for backend in ["green_ctx", "two_stream"]:
                be_sub = coex_sub[coex_sub["backend"] == backend] if "backend" in coex_sub.columns else coex_sub

                if be_sub.empty:
                    overlap_vals = [(0.5, 0.5)]  # default if no data
                    sm_splits    = [0.5]
                else:
                    overlap_vals = list(zip(
                        be_sub["sm_split"].values if "sm_split" in be_sub.columns else [0.5],
                        be_sub["overlap_ratio"].values if "overlap_ratio" in be_sub.columns else [0.5],
                    ))
                    sm_splits = [v[0] for v in overlap_vals]

                for sm_split, overlap_ratio in overlap_vals:
                    # Net speedup: raw speedup × overlap gain
                    net_speedup = speedup_raw * (1.0 / max(overlap_ratio, 0.01))
                    # (overlap_ratio < 1 means concurrent time < solo_sum → speedup)

                    # Overhead penalty subtracts from net gain
                    net_gain_fraction = net_speedup - 1.0 - overhead_penalty

                    # G1-gate score: free_frac × decode_sensitivity
                    g1_score = free_frac * decode_sens

                    rows.append({
                        "seq_len":          seq_len,
                        "layer_type":       layer_type,
                        "sm_split":         round(sm_split, 3),
                        "backend":          backend,
                        "free_sm_fraction": round(free_frac, 4),
                        "decode_sm_sensitivity": round(decode_sens, 4),
                        "speedup_raw_PROJ": round(speedup_raw, 4),
                        "overlap_ratio":    round(overlap_ratio, 4),
                        "overhead_penalty": round(overhead_penalty, 4),
                        "net_speedup_PROJ": round(net_speedup, 4),
                        "net_gain_PROJ":    round(net_gain_fraction, 4),
                        "g1_score":         round(g1_score, 4),
                        "worthwhile":       net_gain_fraction > 0,
                    })

    return rows


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(
    proj_rows: list[dict],
    overhead: dict,
    output_dir: Path,
    model_name: str,
) -> Path:
    """Write projection report text file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"projection_report_{model_name}.txt"

    lines = [
        "=" * 72,
        f"PROJECTION REPORT — {model_name}",
        "(All values in this report are PROJECTIONS, not measurements.)",
        "=" * 72,
        "",
        "ASSUMPTIONS",
        "-" * 40,
    ]
    for i, a in enumerate(ASSUMPTIONS, 1):
        lines.append(f"{i}. {a}")
        lines.append("")

    lines += [
        "STAGE 2 OVERHEAD INPUT",
        "-" * 40,
        f"  ctx_switch_overhead_us:     {overhead.get('ctx_switch_overhead_us', 'N/A')}",
        f"  prefill_layer_latency_ms:   {overhead.get('prefill_layer_latency_ms', 'N/A')}",
        "",
    ]

    if proj_rows:
        import pandas as pd
        df = pd.DataFrame(proj_rows)

        lines += ["PROJECTION SUMMARY (green_ctx, all seq_lens)", "-" * 40]
        gc = df[df["backend"] == "green_ctx"] if "backend" in df.columns else df
        if not gc.empty and "worthwhile" in gc.columns:
            worthwhile_pct = gc["worthwhile"].mean() * 100
            best = gc.loc[gc["net_gain_PROJ"].idxmax()] if "net_gain_PROJ" in gc.columns else None
            lines.append(
                f"  Configs where net_gain > 0 (worthwhile): {worthwhile_pct:.0f}%"
            )
            if best is not None:
                lines.append(
                    f"  Best PROJECTION: seq={best.get('seq_len')} "
                    f"sm_split={best.get('sm_split')} "
                    f"type={best.get('layer_type')} "
                    f"net_gain={best.get('net_gain_PROJ'):.3f}"
                )
        lines.append("")

    lines += [
        "IMPORTANT",
        "-" * 40,
        "  These projections assume ideal conditions (A1–A5 above).",
        "  Actual end-to-end gains will differ due to:",
        "    - Non-linear SM scaling past saturation",
        "    - GPC-granular SM allocation overhead",
        "    - Variable workload composition",
        "    - Thermal throttling under sustained load",
        "  Treat as ORDER-OF-MAGNITUDE estimates only.",
        "=" * 72,
    ]

    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"  Saved: {out}")
    return out


# ---------------------------------------------------------------------------
# Figures  — all labeled PROJECTION
# ---------------------------------------------------------------------------

def plot_speedup(proj_rows: list[dict], output_dir: Path, model_name: str) -> None:
    """PROJECTION: speedup_raw and net_speedup vs seq_len per layer_type."""
    if not proj_rows:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        print("  matplotlib not available — skipping projection speedup figure")
        return

    df = pd.DataFrame(proj_rows)
    # Use green_ctx backend and median sm_split
    if "backend" in df.columns:
        df = df[df["backend"] == "green_ctx"]
    if df.empty:
        return

    # Median over sm_splits
    agg = df.groupby(["seq_len", "layer_type"])[
        ["speedup_raw_PROJ", "net_speedup_PROJ"]
    ].median().reset_index()

    fig, ax = plt.subplots(figsize=(9, 5))

    colors    = {"ssm": "#4CAF50", "attn": "#FF9800", "other": "#9E9E9E"}
    ls_raw    = "--"
    ls_net    = "-"

    for lt in agg["layer_type"].unique():
        g = agg[agg["layer_type"] == lt].sort_values("seq_len")
        c = colors.get(lt, "#666666")
        ax.plot(
            g["seq_len"], g["speedup_raw_PROJ"],
            color=c, linestyle=ls_raw, linewidth=1.5, marker="o",
            label=f"{lt} — raw upper bound (PROJECTED)",
        )
        ax.plot(
            g["seq_len"], g["net_speedup_PROJ"],
            color=c, linestyle=ls_net, linewidth=1.5, marker="s",
            label=f"{lt} — net w/ overlap (PROJECTED)",
        )

    ax.axhline(1.0, color="black", linestyle=":", linewidth=1.2,
               label="No speedup (1.0×)")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Prefill sequence length")
    ax.set_ylabel("Speedup vs. fixed SM baseline  (PROJECTED)")
    ax.set_title(
        f"PROJECTION: Prefill Speedup from Layer-Type-Aware SM Allocation\n"
        f"Model: {model_name}  |  All values PROJECTED, not measured.\n"
        f"Dashed = raw upper bound (A1); Solid = net with overlap (A1×A3)",
        fontsize=10, fontweight="bold",
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.text(
        0.02, 0.03,
        "(PROJECTED — see projection_report for assumptions)",
        transform=ax.transAxes, fontsize=7, color="gray", style="italic",
    )

    plt.tight_layout()
    out = output_dir / f"projection_speedup_{model_name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_net_gain(proj_rows: list[dict], output_dir: Path, model_name: str) -> None:
    """PROJECTION: net_gain_fraction vs sm_split for best seq_len."""
    if not proj_rows:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        print("  matplotlib not available — skipping projection net-gain figure")
        return

    df = pd.DataFrame(proj_rows)
    if df.empty or "net_gain_PROJ" not in df.columns:
        return

    # Pick the seq_len with highest median net_gain for illustration
    if "seq_len" in df.columns:
        best_sl = df.groupby("seq_len")["net_gain_PROJ"].median().idxmax()
        df_sl = df[df["seq_len"] == best_sl]
    else:
        df_sl = df
        best_sl = "N/A"

    fig, ax = plt.subplots(figsize=(9, 5))

    backends = df_sl["backend"].unique() if "backend" in df_sl.columns else ["green_ctx"]
    colors   = {"green_ctx": "#4CAF50", "two_stream": "#9E9E9E"}
    ls_map   = {"green_ctx": "-", "two_stream": "--"}

    for backend in backends:
        sub = df_sl[df_sl["backend"] == backend] if "backend" in df_sl.columns else df_sl
        for lt in sub["layer_type"].unique() if "layer_type" in sub.columns else ["ssm"]:
            g = sub[sub["layer_type"] == lt] if "layer_type" in sub.columns else sub
            g = g.sort_values("sm_split") if "sm_split" in g.columns else g
            ax.plot(
                g["sm_split"] * 100, g["net_gain_PROJ"],
                color=colors.get(backend, "#333333"),
                linestyle=ls_map.get(backend, "-"),
                marker="o", markersize=4, linewidth=1.5,
                label=f"{backend} / {lt} (PROJECTED)",
            )

    ax.axhline(0.0, color="red", linestyle=":", linewidth=1.2,
               label="Break-even (gain = 0)")
    ax.fill_between(
        [0, 100], 0, -0.5,
        alpha=0.08, color="red", label="Net loss zone",
    )
    ax.set_xlabel("Prefill SM allocation (% of total)")
    ax.set_ylabel("Net gain fraction (PROJECTED)")
    ax.set_title(
        f"PROJECTION: Net Prefill Gain vs SM Allocation Split\n"
        f"Model: {model_name}  |  seq_len={best_sl}  |  "
        f"All values PROJECTED, not measured.",
        fontsize=10, fontweight="bold",
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.text(
        0.02, 0.03,
        "(PROJECTED — dashed lines and (projected) labels are mandatory per S3 policy)",
        transform=ax.transAxes, fontsize=7, color="gray", style="italic",
    )

    plt.tight_layout()
    out = output_dir / f"projection_net_gain_{model_name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def save_projection_csv(proj_rows: list[dict], output_dir: Path, model_name: str) -> Path:
    import csv
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"projection_data_{model_name}.csv"
    if not proj_rows:
        return out
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(proj_rows[0].keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(proj_rows)
    print(f"  Saved: {out}  ({len(proj_rows)} rows)")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PROJECTION analysis for layer-type-aware SM allocation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model", default="zamba2")
    p.add_argument(
        "--free-sm-csv", type=Path, default=None,
        help="Stage 1 free_sm_zone_{model}.csv (from plot_saturation.py --save-free-sm-csv)",
    )
    p.add_argument(
        "--coexec-csv", type=Path, default=None,
        help="coexec_{model}.csv from coexec_microbench.py (overlap_ratio per sm_split)",
    )
    p.add_argument(
        "--stage2-json", type=Path, default=None,
        help="results/stage2/decision_matrix.json (ctx_switch_overhead_us)",
    )
    p.add_argument(
        "--stage2-csv-glob", default=None,
        help="Glob pattern for Stage 2 CSV files (alternative to --stage2-json)",
    )
    p.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).parent.parent / "results" / "stage3",
    )
    p.add_argument(
        "--demo", action="store_true",
        help="Use synthetic demo data (no real CSV required)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("\n" + "=" * 60)
    print("  PROJECTION ANALYSIS — all outputs are projections.")
    print("  Not measurements. See assumptions in report.")
    print("=" * 60)

    seq_lens  = [256, 512, 1024, 2048, 4096, 8192, 16384]
    sm_splits = [0.3, 0.4, 0.5, 0.6, 0.7]

    if args.demo:
        print("\n  [DEMO MODE] Using synthetic data.")
        free_sm_rows = _demo_free_sm(seq_lens, args.model)
        coexec_rows  = _demo_overlap(sm_splits, seq_lens)
        overhead     = _demo_overhead()
    else:
        free_sm_rows = load_free_sm_csv(args.free_sm_csv) if args.free_sm_csv else []
        coexec_rows  = load_coexec_csv(args.coexec_csv)  if args.coexec_csv  else []
        overhead     = load_stage2_overhead(args.stage2_json, args.stage2_csv_glob)

        if not free_sm_rows:
            print(
                "\n  [WARN] No free-SM data. Run with --demo for demo figures,\n"
                "  or provide --free-sm-csv (from plot_saturation.py)."
            )
        if not coexec_rows:
            print(
                "\n  [WARN] No coexec data. Run with --demo for demo figures,\n"
                "  or provide --coexec-csv (from coexec_microbench.py)."
            )

    proj_rows = compute_projections(free_sm_rows, coexec_rows, overhead)

    if not proj_rows:
        print("\n  No projection rows — input data missing or incompatible.")
        if not args.demo:
            print("  Use --demo to generate synthetic demo figures.")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    save_projection_csv(proj_rows, args.output_dir, args.model)
    report_path = write_report(proj_rows, overhead, args.output_dir, args.model)
    plot_speedup(proj_rows, args.output_dir, args.model)
    plot_net_gain(proj_rows, args.output_dir, args.model)

    print(
        f"\n  PROJECTION outputs:\n"
        f"    report:    {report_path}\n"
        f"    speedup:   {args.output_dir}/projection_speedup_{args.model}.png\n"
        f"    net_gain:  {args.output_dir}/projection_net_gain_{args.model}.png\n"
        f"\n  REMINDER: All figures are PROJECTED bounds, not measurements."
    )


if __name__ == "__main__":
    main()
