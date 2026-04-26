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
# Figure 3: SM Utilization vs SM count
# ---------------------------------------------------------------------------

def plot_sm_util_curves(
    df: pd.DataFrame,
    output_dir: Path,
    model_name: str,
    batch_size: int,
) -> None:
    """Plot ncu SM utilization and wave efficiency vs SM count.

    Two subplots per layer type:
      Top:    sm_util_per_sm_pct  — active SM cycles / elapsed SM cycles × 100
              Source: ncu sm__cycles_active.sum / sm__cycles_elapsed.sum.
              Captures wave quantization (idle SMs in last wave) and warp stalls.
      Bottom: wave_efficiency_pct = grid_size / (n_waves × sm_count) × 100
              Analytical wave efficiency from ncu grid size capture.

    Skipped with a diagnostic message if ncu columns are absent.
    These columns come from the ncu CSV (run_ncu_profile.py), not the latency CSV.
    """
    eff_col = "wave_efficiency_pct"
    sm_eff_col = "sm_util_per_sm_pct"
    if eff_col not in df.columns or sm_eff_col not in df.columns:
        print("  No ncu SM utilization columns found — skipping Figure 3")
        print("  (run run_ncu_profile.py to generate ncu_*.csv files)")
        return

    df_model = df[(df["model_name"] == model_name) & (df["batch_size"] == batch_size)]
    if df_model.empty:
        return

    # Skip if all ncu columns are NaN for this model (no ncu CSV was run for it)
    if df_model[sm_eff_col].isna().all():
        print(f"  No ncu data for {model_name} — skipping Figure 3")
        print(f"  (run: python stage1_sm_scaling/run_ncu_profile.py --model {model_name})")
        return

    layer_types = sorted(df_model["layer_type"].unique())
    seq_lens = sorted(df_model["seq_len"].unique())
    n_cols = len(layer_types)

    fig, axes = plt.subplots(2, n_cols, figsize=(6 * n_cols, 8), sharey="row")
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    fig.suptitle(
        f"ncu SM Utilization & Wave Efficiency vs SM Count — {model_name}, batch_size={batch_size}",
        fontsize=13, fontweight="bold"
    )

    total_sm = df_model["sm_count"].max()
    cmap = plt.cm.viridis
    seq_colors = {sl: cmap(i / max(len(seq_lens) - 1, 1)) for i, sl in enumerate(seq_lens)}

    row_labels = [
        "ncu SM utilization  (active cycles / elapsed cycles × 100)\n"
        "Captures wave quantization + warp stalls. NOT binary on/off.",
        "Wave efficiency = grid_size / (n_waves × sm_count) × 100\n"
        "= fraction of SM slots actually used per wave  [from ncu grid capture]",
    ]

    for col_idx, lt in enumerate(layer_types):
        df_lt = df_model[df_model["layer_type"] == lt]

        for row_idx, (col_name, row_label) in enumerate(
            zip([sm_eff_col, eff_col], row_labels)
        ):
            ax = axes[row_idx][col_idx]

            for sl in seq_lens:
                grp = df_lt[df_lt["seq_len"] == sl].sort_values("sm_count")
                if grp.empty or col_name not in grp.columns:
                    continue
                vals = grp[col_name].values
                if np.all(np.isnan(vals)):
                    continue
                ax.plot(
                    grp["sm_count"], vals,
                    color=seq_colors[sl], linewidth=2,
                    marker="o", markersize=4,
                    label=f"seq={sl}",
                )

            ax.set_xlim(0, total_sm * 1.05)
            ax.set_ylim(0, 110)
            ax.axhline(y=100, color="gray", linestyle=":", linewidth=1, alpha=0.5)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("SM Count")

            if col_idx == 0:
                ax.set_ylabel("Utilization (%)")

            if row_idx == 0:
                ax.set_title(LAYER_LABELS.get(lt, lt), fontsize=11)

            # Add row label on leftmost column only
            if col_idx == 0:
                ax.annotate(
                    row_label,
                    xy=(-0.18, 0.5), xycoords="axes fraction",
                    fontsize=7, va="center", ha="right",
                    rotation=90, wrap=True,
                )

    # Shared legend
    handles = [
        mpatches.Patch(color=seq_colors[sl], label=f"seq_len={sl}")
        for sl in seq_lens
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(seq_lens),
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
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    df = compute_throughput(df)

    # --- ncu CSVs (ncu_ssm_*, ncu_attn_*, ncu_mlp_*) ---
    # Merge SM utilization and wave stats from ncu profiling results.
    # ncu CSVs use "model" column; latency CSVs use "model_name".
    ncu_files = list(results_dir.glob("ncu_*.csv"))
    if ncu_files:
        ncu_dfs = []
        for f in ncu_files:
            ndf = pd.read_csv(f)
            # Rename model → model_name to match latency schema
            if "model" in ndf.columns and "model_name" not in ndf.columns:
                ndf = ndf.rename(columns={"model": "model_name"})
            ncu_dfs.append(ndf)

        ncu_df = pd.concat(ncu_dfs, ignore_index=True)

        # Keep only the columns that don't already exist in the latency df
        # plus the merge keys
        merge_keys = ["model_name", "layer_type", "sm_count", "seq_len", "batch_size"]
        ncu_cols = [c for c in ncu_df.columns
                    if c in merge_keys or c not in df.columns]
        ncu_df = ncu_df[ncu_cols].drop_duplicates(subset=merge_keys)

        df = df.merge(ncu_df, on=merge_keys, how="left")
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
        plot_sm_util_curves(df, args.output_dir, model, args.batch_size)

    print("\nDone.")
