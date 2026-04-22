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
LAYER_COLORS = {"ssm": "#2196F3", "attn": "#FF5722", "mlp": "#4CAF50"}
LAYER_LABELS = {"ssm": "SSM (Mamba-2)", "attn": "Attention", "mlp": "MLP/FFN"}


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
    df_model = df[(df["model_name"] == model_name) & (df["batch_size"] == batch_size)]
    if df_model.empty:
        print(f"  No data for {model_name} batch_size={batch_size}")
        return

    seq_lens = sorted(df_model["seq_len"].unique())
    layer_types = sorted(df_model["layer_type"].unique())

    n_cols = len(layer_types)
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5), sharey=True)
    if n_cols == 1:
        axes = [axes]

    fig.suptitle(
        f"SM Scaling Curves — {model_name}, batch_size={batch_size}",
        fontsize=13, fontweight="bold"
    )

    total_sm = df_model["sm_count"].max()
    cmap = plt.cm.viridis
    seq_colors = {sl: cmap(i / max(len(seq_lens) - 1, 1))
                  for i, sl in enumerate(seq_lens)}

    for ax, lt in zip(axes, layer_types):
        df_lt = df_model[df_model["layer_type"] == lt]
        sat_points = []

        for sl in seq_lens:
            grp = df_lt[df_lt["seq_len"] == sl].sort_values("sm_count")
            if grp.empty:
                continue

            ax.plot(
                grp["sm_count"], grp["normalized_throughput"],
                color=seq_colors[sl],
                linewidth=2,
                marker="o", markersize=4,
                label=f"seq={sl}",
            )

            sat_sm = find_saturation_sm(grp)
            sat_tp = grp.loc[grp["sm_count"] == sat_sm, "normalized_throughput"]
            if not sat_tp.empty:
                ax.axvline(
                    x=sat_sm, color=seq_colors[sl],
                    linestyle="--", linewidth=1, alpha=0.6
                )
                ax.scatter([sat_sm], [sat_tp.values[0]],
                           color=seq_colors[sl], s=80, zorder=5,
                           marker="x", linewidths=2)
                sat_points.append(sat_sm)

        ax.set_title(LAYER_LABELS.get(lt, lt), fontsize=11)
        ax.set_xlabel("SM Count")
        ax.set_xlim(0, total_sm * 1.05)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=1.0, color="black", linestyle=":", linewidth=1, alpha=0.5)

        if ax == axes[0]:
            ax.set_ylabel("Normalized Throughput (SM=100% → 1.0)")

    # Shared legend for seq_lens
    handles = [
        mpatches.Patch(color=seq_colors[sl], label=f"seq_len={sl}")
        for sl in seq_lens
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(seq_lens),
               bbox_to_anchor=(0.5, -0.02))

    # Add saturation annotation note
    fig.text(
        0.5, -0.06,
        "✕ = saturation point (throughput gain < 3% per 10% SM increase)  "
        "-- = saturation SM count",
        ha="center", fontsize=9, style="italic"
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
    df_ssm = df[(df["model_name"] == model_name) & (df["layer_type"] == "ssm")]
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
        f"Free SM Zone — {model_name} SSM Prefill\n"
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
# Data loading
# ---------------------------------------------------------------------------

def load_results(results_dir: Path) -> pd.DataFrame:
    csv_files = list(results_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {results_dir}")

    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        # Infer layer_type from filename if column missing
        if "layer_type" not in df.columns:
            fname = f.stem
            for lt in ["ssm", "attn", "mlp"]:
                if lt in fname:
                    df["layer_type"] = lt
                    break
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    df = compute_throughput(df)
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
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).parent.parent / "results" / "stage1"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print(f"Loading results from {args.results_dir} …")
    df = load_results(args.results_dir)

    models = [args.model] if args.model else df["model_name"].unique().tolist()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for model in models:
        print(f"\nPlotting {model} …")
        plot_scaling_curves(df, args.output_dir, model, args.batch_size)
        plot_free_sm_zone(df, args.output_dir, model)

    print("\nDone.")
