"""
Stage 1 visualization: SM scaling curves and Free-SM zone.

Generates two figures from stage1 CSV results:

  Figure 1 — SM Scaling Curve
    x: SM count (or ratio), y: normalized throughput vs SM=100%
    Lines per (layer_type × seq_len), with saturation point markers.
    Saturation: first SM step where throughput improvement < 3% per 10% SM.

  Figure 2 — Free SM Zone
    Bar chart of saturation SM count per (model, seq_len, batch_size).
    "Free SM" = SM_total - SM_saturation, annotated as decode budget.

Usage:
    python stage1_sm_scaling/plot_saturation.py --results-dir results/stage1
    python stage1_sm_scaling/plot_saturation.py --model zamba2 --batch-size 1
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns


# ---------------------------------------------------------------------------
# Saturation detection
# ---------------------------------------------------------------------------

SATURATION_THRESHOLD = 0.03   # < 3% throughput gain per 10% SM → saturated
LAYER_COLORS = {
    "ssm":        "#2196F3",
    "ssm_triton": "#1565C0",   # darker blue  — analytical wave model
    "ssm_torch":  "#42A5F5",   # lighter blue — direct PyTorch scan
    "attn":       "#FF5722",
    "mlp":        "#4CAF50",
}
LAYER_LABELS = {
    "ssm":        "SSM (Mamba-2)",
    "ssm_triton": "SSM Triton (analytical)",
    "ssm_torch":  "SSM PyTorch (direct)",
    "attn":       "Attention",
    "mlp":        "MLP/FFN",
}

# SSM variants that share the same Free-SM zone semantics (prefer triton when available)
_SSM_TYPES = ("ssm_triton", "ssm", "ssm_torch")


def compute_throughput(df: pd.DataFrame) -> pd.DataFrame:
    """Add normalized_throughput column (relative to max-SM config)."""
    df = df.copy()
    # Throughput ∝ 1 / latency (tokens/sec per layer)
    df["throughput"] = 1.0 / df["latency_ms"]

    groups = ["model_name", "layer_type", "seq_len", "batch_size"]
    max_tp = df.groupby(groups)["throughput"].transform("max")
    df["normalized_throughput"] = df["throughput"] / max_tp
    return df


def find_saturation_sm(group: pd.DataFrame) -> int:
    """Return SM count at which throughput gain drops below threshold.

    Detection: for each step, compute marginal throughput improvement
    relative to the previous step (as fraction of max throughput).
    Saturation = first step where marginal improvement < 3% per 10% SM.
    """
    group = group.sort_values("sm_ratio")
    sm_ratios = group["sm_ratio"].values
    tps = group["normalized_throughput"].values

    for i in range(1, len(sm_ratios)):
        delta_sm = sm_ratios[i] - sm_ratios[i - 1]
        delta_tp = tps[i] - tps[i - 1]
        if delta_sm <= 0:
            continue
        # Normalize: throughput gain per 10% SM increase
        gain_per_10pct = (delta_tp / delta_sm) * 0.10
        if gain_per_10pct < SATURATION_THRESHOLD:
            return int(group["sm_count"].iloc[i - 1])

    # No saturation detected → full SM
    return int(group["sm_count"].max())


# ---------------------------------------------------------------------------
# Figure 1: SM Scaling Curves
# ---------------------------------------------------------------------------

def plot_scaling_curves(
    df: pd.DataFrame,
    output_dir: Path,
    model_name: str,
    batch_size: int,
) -> None:
    """Throughput vs seq_len, one line per SM count.

    X-axis : seq_len (workload size)
    Lines  : SM count (allocation)
    Marker : ✕ on the saturation SM count line at each seq_len
             (first SM allocation where adding more SMs yields < 3% gain per 10% SM)
    """
    df_model = df[(df["model_name"] == model_name) & (df["batch_size"] == batch_size)]
    if df_model.empty:
        print(f"  No data for {model_name} batch_size={batch_size}")
        return

    seq_lens   = sorted(df_model["seq_len"].unique())
    sm_counts  = sorted(df_model["sm_count"].unique())
    layer_types = sorted(df_model["layer_type"].unique())
    total_sm   = df_model["sm_count"].max()

    n_cols = len(layer_types)
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5), sharey=True)
    if n_cols == 1:
        axes = [axes]

    fig.suptitle(
        f"Throughput vs Sequence Length — {model_name}, batch_size={batch_size}",
        fontsize=13, fontweight="bold",
    )

    cmap = plt.cm.plasma
    sm_colors = {sm: cmap(i / max(len(sm_counts) - 1, 1))
                 for i, sm in enumerate(sm_counts)}

    for ax, lt in zip(axes, layer_types):
        df_lt = df_model[df_model["layer_type"] == lt]

        # Per-seq_len saturation SM count (to mark on the correct line)
        sat_sm_per_seq = {}
        for sl in seq_lens:
            grp = df_lt[df_lt["seq_len"] == sl].sort_values("sm_count")
            if not grp.empty:
                sat_sm_per_seq[sl] = find_saturation_sm(grp)

        # Plot one line per SM count
        for sm in sm_counts:
            grp = df_lt[df_lt["sm_count"] == sm].sort_values("seq_len")
            if grp.empty:
                continue
            ratio = sm / total_sm
            ax.plot(
                grp["seq_len"], grp["normalized_throughput"],
                color=sm_colors[sm],
                linewidth=2,
                marker="o", markersize=4,
                label=f"sm={sm} ({ratio:.0%})",
            )

        # Saturation markers: ✕ on the saturation line at each seq_len
        for sl, sat_sm in sat_sm_per_seq.items():
            row = df_lt[(df_lt["seq_len"] == sl) & (df_lt["sm_count"] == sat_sm)]
            if row.empty:
                continue
            tp = row["normalized_throughput"].values[0]
            ax.scatter([sl], [tp], color=sm_colors[sat_sm],
                       s=100, zorder=6, marker="x", linewidths=2.5)

        ax.set_title(LAYER_LABELS.get(lt, lt), fontsize=11)
        ax.set_xlabel("Sequence Length")
        ax.set_xscale("log", base=2)
        ax.set_xticks(seq_lens)
        ax.set_xticklabels(seq_lens, rotation=30)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=1.0, color="black", linestyle=":", linewidth=1, alpha=0.5)

        if ax == axes[0]:
            ax.set_ylabel("Normalized Throughput (SM=100% → 1.0)")

    handles = [
        mpatches.Patch(color=sm_colors[sm], label=f"sm={sm} ({sm/total_sm:.0%})")
        for sm in sm_counts
    ]
    n_legend_cols = min(len(sm_counts), 8)
    fig.legend(handles=handles, loc="lower center", ncol=n_legend_cols,
               bbox_to_anchor=(0.5, -0.04))

    fig.text(
        0.5, -0.10,
        "✕ = saturation point (first SM count where throughput gain < 3% per 10% SM increase)",
        ha="center", fontsize=9, style="italic",
    )

    plt.tight_layout()
    out_path = output_dir / f"fig1_scaling_{model_name}_bs{batch_size}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 2: Free SM Zone
# ---------------------------------------------------------------------------

def plot_free_sm_zone(
    df: pd.DataFrame,
    output_dir: Path,
    model_name: str,
) -> None:
    # Prefer ssm_triton (production kernel), fall back to generic ssm or ssm_torch
    ssm_type = next(
        (t for t in _SSM_TYPES if t in df["layer_type"].values),
        None,
    )
    if ssm_type is None:
        print(f"  No SSM data for {model_name}")
        return
    df_ssm = df[(df["model_name"] == model_name) & (df["layer_type"] == ssm_type)]
    if df_ssm.empty:
        print(f"  No SSM data for {model_name}")
        return

    total_sm = df_ssm["sm_count"].max()
    records = []

    for (sl, bs), grp in df_ssm.groupby(["seq_len", "batch_size"]):
        sat_sm = find_saturation_sm(grp)
        free_sm = total_sm - sat_sm
        records.append({
            "seq_len": sl,
            "batch_size": bs,
            "saturation_sm": sat_sm,
            "free_sm": free_sm,
            "sat_ratio": sat_sm / total_sm,
            "free_ratio": free_sm / total_sm,
            "label": f"seq={sl}\nbs={bs}",
        })

    rec_df = pd.DataFrame(records).sort_values(["seq_len", "batch_size"])

    fig, ax = plt.subplots(figsize=(max(10, len(records) * 0.8), 5))
    x = np.arange(len(rec_df))
    width = 0.6

    bars_sat = ax.bar(x, rec_df["saturation_sm"], width,
                      label=f"SSM saturation SM", color="#2196F3", alpha=0.8)
    bars_free = ax.bar(x, rec_df["free_sm"], width,
                       bottom=rec_df["saturation_sm"],
                       label="Free SM (decode budget)", color="#FF9800", alpha=0.8)

    # Annotate free SM fraction
    for i, row in rec_df.iterrows():
        idx = rec_df.index.get_loc(i)
        if row["free_sm"] > 0:
            ax.text(
                idx,
                row["saturation_sm"] + row["free_sm"] / 2,
                f"{row['free_ratio']:.0%}",
                ha="center", va="center", fontsize=8, fontweight="bold",
                color="white"
            )

    ax.axhline(y=total_sm, color="red", linestyle="--", linewidth=1.5,
               label=f"Total SM = {total_sm}")
    ax.set_xticks(x)
    ax.set_xticklabels(rec_df["label"], fontsize=9)
    ax.set_ylabel("SM Count")
    ax.set_ylim(0, total_sm * 1.15)
    ax.set_title(
        f"Free SM Zone — {model_name}  [{LAYER_LABELS.get(ssm_type, ssm_type)}]\n"
        f"(Free SM = SMs available for decode during SSM prefill)",
        fontsize=11, fontweight="bold"
    )
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / f"fig2_free_sm_{model_name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 3: SM Utilization vs SM count
# ---------------------------------------------------------------------------

def plot_sm_util_curves(
    df: pd.DataFrame,
    output_dir: Path,
    model_name: str,
    batch_size: int,
) -> None:
    """Plot ncu SM utilization, wave efficiency, and occupancy vs SM count.

    Three rows × N layer_type columns:
      Row 0: sm_util_per_sm_pct  — active cycles / elapsed cycles × 100
             Captures wave quantization (last-wave idle SMs) and warp stalls.
      Row 1: wave_efficiency_pct — grid_size / (n_waves × sm_count) × 100
             Pure scheduling loss from wave quantization.
      Row 2: achieved_occupancy_pct — avg resident warps / max warps × 100
             smsp__warps_active.sum / (sm__cycles_elapsed.sum × max_warps_per_sm)
             Reflects register/shared-memory pressure limiting concurrent warps.

    Skipped for models with no ncu data (run run_ncu_profile.py first).
    """
    sm_eff_col  = "sm_util_per_sm_pct"
    wave_col    = "wave_efficiency_pct"
    occ_col     = "achieved_occupancy_pct"

    required = [sm_eff_col, wave_col]
    if not all(c in df.columns for c in required):
        print("  No ncu SM utilization columns found — skipping Figure 3")
        print("  (run run_ncu_profile.py to generate ncu_*.csv files)")
        return

    df_model = df[(df["model_name"] == model_name) & (df["batch_size"] == batch_size)]
    if df_model.empty:
        return

    if df_model[sm_eff_col].isna().all():
        print(f"  No ncu data for {model_name} — skipping Figure 3")
        print(f"  (run: python stage1_sm_scaling/run_ncu_profile.py --model {model_name})")
        return

    layer_types = sorted(df_model["layer_type"].unique())
    seq_lens    = sorted(df_model["seq_len"].unique())
    n_cols      = len(layer_types)
    has_occ     = occ_col in df_model.columns and not df_model[occ_col].isna().all()
    n_rows      = 3 if has_occ else 2

    row_specs = [
        (sm_eff_col,
         "SM Utilization\n(active / elapsed cycles × 100)\n"
         "Wave quantization + warp stalls",
         (0, 110), 100, "Utilization (%)"),
        (wave_col,
         "Wave Efficiency\n(grid_size / n_waves / sm_count × 100)\n"
         "Pure scheduling loss from wave quantization",
         (0, 110), 100, "Efficiency (%)"),
    ]
    if has_occ:
        row_specs.append((
            occ_col,
            "Achieved Occupancy\n(avg resident warps / max warps × 100)\n"
            "Register / shared-mem pressure limit",
            (0, 110), None, "Occupancy (%)"),
        )

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(6 * n_cols, 4 * n_rows),
        sharey="row",
    )
    # Normalise axes shape to always (n_rows, n_cols)
    if n_cols == 1 and n_rows == 1:
        axes = np.array([[axes]])
    elif n_cols == 1:
        axes = axes.reshape(n_rows, 1)
    elif n_rows == 1:
        axes = axes.reshape(1, n_cols)

    fig.suptitle(
        f"ncu SM Utilization / Wave Efficiency / Occupancy — {model_name}, batch_size={batch_size}",
        fontsize=13, fontweight="bold",
    )

    total_sm  = df_model["sm_count"].max()
    sm_counts = sorted(df_model["sm_count"].unique())
    cmap      = plt.cm.plasma
    sm_colors = {sm: cmap(i / max(len(sm_counts) - 1, 1)) for i, sm in enumerate(sm_counts)}

    for col_idx, lt in enumerate(layer_types):
        df_lt = df_model[df_model["layer_type"] == lt]

        for row_idx, (col_name, row_label, ylim, hline_y, ylabel) in enumerate(row_specs):
            ax = axes[row_idx][col_idx]

            for sm in sm_counts:
                grp = df_lt[df_lt["sm_count"] == sm].sort_values("seq_len")
                if grp.empty or col_name not in grp.columns:
                    continue
                vals = grp[col_name].values.astype(float)
                if np.all(np.isnan(vals)):
                    continue
                ratio = sm / total_sm
                ax.plot(
                    grp["seq_len"], vals,
                    color=sm_colors[sm], linewidth=2,
                    marker="o", markersize=4,
                    label=f"sm={sm} ({ratio:.0%})",
                )

            ax.set_xscale("log", base=2)
            ax.set_xticks(seq_lens)
            ax.set_xticklabels(seq_lens, rotation=30)
            ax.set_ylim(*ylim)
            if hline_y is not None:
                ax.axhline(y=hline_y, color="gray", linestyle=":", linewidth=1, alpha=0.5)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("Sequence Length")

            if col_idx == 0:
                ax.set_ylabel(ylabel)
                ax.annotate(
                    row_label,
                    xy=(-0.22, 0.5), xycoords="axes fraction",
                    fontsize=7, va="center", ha="right",
                    rotation=90,
                )

            if row_idx == 0:
                ax.set_title(LAYER_LABELS.get(lt, lt), fontsize=11)

    handles = [
        mpatches.Patch(color=sm_colors[sm], label=f"sm={sm} ({sm/total_sm:.0%})")
        for sm in sm_counts
    ]
    n_legend_cols = min(len(sm_counts), 8)
    fig.legend(handles=handles, loc="lower center", ncol=n_legend_cols,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    out_path = output_dir / f"fig3_ncu_sm_util_{model_name}_bs{batch_size}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(results_dir: Path) -> pd.DataFrame:
    # --- Latency CSVs (ssm_scaling_*, attn_scaling_*, mlp_scaling_*) ---
    latency_files = [
        f for f in results_dir.glob("*.csv")
        if not f.name.startswith("ncu_")
    ]
    if not latency_files:
        raise FileNotFoundError(f"No latency CSV files found in {results_dir}")

    dfs = []
    for f in latency_files:
        df = pd.read_csv(f)
        if "layer_type" not in df.columns:
            for lt in ["ssm", "attn", "mlp"]:
                if lt in f.stem:
                    df["layer_type"] = lt
                    break
        # Distinguish ssm_triton (analytical wave model) vs ssm_torch (direct torchscan)
        # based on the output filename convention used by run_ssm_prefill_sweep.py.
        if "_torchscan" in f.stem:
            df["layer_type"] = df["layer_type"].replace({"ssm": "ssm_torch"})
        elif f.stem.startswith("ssm_"):
            df["layer_type"] = df["layer_type"].replace({"ssm": "ssm_triton"})
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    df = compute_throughput(df)

    # --- ncu CSVs (ncu_ssm_*, ncu_attn_*, ncu_mlp_*) ---
    # Merge SM utilization and wave stats from ncu profiling results.
    # ncu CSVs use "model" column (or omit it entirely); latency CSVs use "model_name".
    # Filename format: ncu_{layer_type}_{model}_{device}.csv — used as fallback.
    ncu_files = list(results_dir.glob("ncu_*.csv"))
    if ncu_files:
        ncu_dfs = []
        for f in ncu_files:
            ndf = pd.read_csv(f)
            if "model" in ndf.columns and "model_name" not in ndf.columns:
                ndf = ndf.rename(columns={"model": "model_name"})

            # Infer model_name / layer_type from filename when the CSV omits them.
            # Filename: ncu_{layer_type}_{model}_{device...}.csv
            stem_parts = f.stem.split("_")   # ['ncu','ssm','zamba2','a100',...]
            if "layer_type" not in ndf.columns and len(stem_parts) >= 2:
                raw_lt = stem_parts[1]   # ssm / attn / mlp
                # ncu profiles use the Triton SSD path → tag as ssm_triton
                ndf["layer_type"] = "ssm_triton" if raw_lt == "ssm" else raw_lt
            if "model_name" not in ndf.columns:
                for known in ("zamba2", "falcon_h1"):
                    if f"_{known}_" in f.name or f.name.endswith(f"_{known}.csv"):
                        ndf["model_name"] = known
                        break

            ncu_dfs.append(ndf)

        ncu_df = pd.concat(ncu_dfs, ignore_index=True)

        # Keep only columns that are merge keys or new (not already in latency df)
        merge_keys = ["model_name", "layer_type", "sm_count", "seq_len", "batch_size"]
        ncu_cols = [c for c in ncu_df.columns
                    if c in merge_keys or c not in df.columns]
        # Only dedup on keys that actually exist in ncu_df
        valid_dedup = [k for k in merge_keys if k in ncu_df.columns]
        ncu_df = ncu_df[ncu_cols].drop_duplicates(subset=valid_dedup or None)

        # Merge on the intersection of merge_keys present in both frames
        valid_on = [k for k in merge_keys if k in ncu_df.columns and k in df.columns]
        df = df.merge(ncu_df, on=valid_on, how="left")
        print(f"  Merged {len(ncu_files)} ncu CSV(s) → "
              f"SM util / wave stats columns added.")

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Plot Stage 1 SM scaling results")
    parser.add_argument(
        "--results-dir", type=Path,
        default=Path(__file__).parent.parent / "results" / "stage1"
    )
    parser.add_argument("--model", default=None, help="Filter by model name")
    parser.add_argument(
        "--batch-sizes", nargs="+", type=int, default=None,
        help="Batch sizes to plot (default: all found in data)"
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).parent.parent / "results" / "stage1"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print(f"Loading results from {args.results_dir} …")
    df = load_results(args.results_dir)

    models = [args.model] if args.model else sorted(df["model_name"].unique().tolist())
    batch_sizes = args.batch_sizes if args.batch_sizes else sorted(df["batch_size"].unique().tolist())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for model in models:
        print(f"\nPlotting {model} …")
        # Fig 2: one per model (SSM saturation, independent of batch_size axis)
        plot_free_sm_zone(df, args.output_dir, model)

        for bs in batch_sizes:
            print(f"  batch_size={bs}")
            plot_scaling_curves(df, args.output_dir, model, bs)
            plot_sm_util_curves(df, args.output_dir, model, bs)

    print("\nDone.")
