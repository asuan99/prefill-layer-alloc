"""
Stage 1 visualization: All-module comparison — attn, ssm_triton, ssm_torch, ssm_chunked.

Generates seven figures (Figs 4–10):

  Figure 4 — Module Latency Comparison
    Grid of (batch_size × seq_len) panels.  Each panel plots latency vs SM
    allocation ratio for all three modules on shared axes, with saturation
    markers.  Answers "which module is the bottleneck at each workload point?"

  Figure 5 — SSM Wave-Model Validation
    Scatter of ssm_triton analytical latency vs ssm_torch directly-measured
    latency for matched (sm_count, seq_len, batch_size) triples.  Points on
    y = x confirm the wave model is accurate; the plot also reports RMSE and
    MAPE.

  Figure 6 — Saturation SM Ratio Heatmap
    Three heatmaps (one per module) over seq_len × batch_size.  Each cell
    shows the saturation SM ratio (0 → 1), i.e. the minimum SM fraction that
    captures ≥ 97 % of the peak throughput.  Lighter cells = saturate earlier
    = more SM budget available for decode.

  Figure 7 — SM Sensitivity (latency degradation at reduced SM)
    Grouped bar chart: for each module the mean latency increase (%)
    when SM allocation is halved to 75 %, 50 %, or 25 % of the total.
    Shows which module is most sensitive to SM under-provisioning.

  Figure 8 — Chunked Prefill Cooperative Safety Map  [requires --ssm-chunked-csv]
    Heatmap over sm_count × prefill_chunk_tokens, one panel per batch_size.
    Green = cooperative_safe (n_blocks_per_call ≤ SM count).
    Red = deadlock risk (n_blocks_per_call > SM count).
    Dashed line = theoretical boundary: chunk_tokens = 4 × sm_count / batch_size.

  Figure 9 — Chunked Prefill: Latency vs SM Allocation  [requires --ssm-chunked-csv]
    Grid of (batch_size × seq_len) panels showing safe chunked-prefill configs.
    Each line corresponds to one prefill_chunk_tokens value.
    SSM Triton full-sequence curve overlaid as a dashed reference.

  Figure 10 — Chunked Prefill: Latency vs Kernel Call Count  [requires --ssm-chunked-csv]
    Scatter of latency vs n_kernel_calls (= seq_len / chunk_tokens), with a
    linear fit per SM count per batch_size.  Reveals per-call overhead.

Usage:
    python stage1_sm_scaling/plot_compare_modules.py
    python stage1_sm_scaling/plot_compare_modules.py --model zamba2 --batch-sizes 1 4 16
    python stage1_sm_scaling/plot_compare_modules.py --results-dir results/stage1
    python stage1_sm_scaling/plot_compare_modules.py \\
        --ssm-chunked-csv results/stage1/ssm_chunked_zamba2_a100-sxm4-80gb.csv
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.colors as mcolors

from stage1_sm_scaling.plot_saturation import (
    find_saturation_sm,
    compute_throughput,
    SATURATION_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

MODULE_ORDER = ["attn", "ssm_triton", "ssm_torch", "ssm_chunked"]

MODULE_COLORS = {
    "attn":        "#FF5722",   # red-orange
    "ssm_triton":  "#1565C0",   # deep blue  — analytical wave model
    "ssm_torch":   "#42A5F5",   # light blue — direct PyTorch scan
    "ssm_chunked": "#2E7D32",   # dark green — chunked prefill direct measurement
}
MODULE_LABELS = {
    "attn":        "Attention",
    "ssm_triton":  "SSM Triton (analytical)",
    "ssm_torch":   "SSM PyTorch (direct)",
    "ssm_chunked": "SSM Chunked Prefill (direct)",
}
MODULE_LINESTYLES = {
    "attn":        "-.",
    "ssm_triton":  "-",
    "ssm_torch":   "--",
    "ssm_chunked": ":",
}
MODULE_MARKERS = {
    "attn":        "^",
    "ssm_triton":  "o",
    "ssm_torch":   "s",
    "ssm_chunked": "D",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _has_analytical_col(path: Path) -> bool:
    """Return True if the CSV contains a populated 'analytical' column (wave-model file)."""
    try:
        df = pd.read_csv(path, nrows=2)
        return "analytical" in df.columns and df["analytical"].notna().any()
    except Exception:
        return False


def _pick_best_csv(files: list[Path], layer_type: str) -> Path:
    """Select the single best CSV when multiple files exist for a layer type.

    Selection rules (applied in order):
      ssm_triton → prefer files that have the 'analytical' column
                   (wave-model synthesis is more reliable than old direct measurements)
      all types  → among remaining candidates, take alphabetically last
                   (matches find_stage1_csv convention: picks most specific name)
    """
    candidates = sorted(files)
    if layer_type == "ssm_triton":
        analytical = [f for f in candidates if _has_analytical_col(f)]
        if analytical:
            candidates = analytical
    return candidates[-1]


def load_chunked_csv(chunked_csv: Path, model: str, filter_unsafe: bool = True) -> pd.DataFrame:
    """Load a chunked prefill CSV and normalise it to the common module schema.

    chunked CSV columns (from run_chunked_ssm_sweep.py):
        model, device, seq_len, batch_size, sm_count, sm_ratio_pct,
        prefill_chunk_tokens, n_kernel_calls, n_blocks_per_call,
        cooperative_safe, latency_ms, latency_std_ms

    Added columns for compatibility with plot_compare_modules pipeline:
        sm_ratio, layer_type='ssm_chunked', model_name, latency_p99_ms,
        achieved_bandwidth_GBs, theoretical_bw_GBs, bw_utilization_pct
    """
    df = pd.read_csv(chunked_csv)

    # Filter by model if column exists
    if "model" in df.columns:
        df = df[df["model"] == model].copy()
        df = df.rename(columns={"model": "model_name"})
    else:
        df = df.copy()
        df["model_name"] = model

    # Normalise sm_ratio
    if "sm_ratio_pct" in df.columns and "sm_ratio" not in df.columns:
        df["sm_ratio"] = df["sm_ratio_pct"] / 100.0

    # Drop cooperative_safe=False rows (deadlock risk, latency likely NaN)
    if "cooperative_safe" in df.columns and filter_unsafe:
        df = df[df["cooperative_safe"].astype(bool)].copy()

    df["layer_type"]             = "ssm_chunked"
    df["latency_p99_ms"]         = df["latency_ms"]    # no p99 in chunked CSV
    df["achieved_bandwidth_GBs"] = float("nan")
    df["theoretical_bw_GBs"]     = float("nan")
    df["bw_utilization_pct"]     = float("nan")

    return df


def load_all_modules(
    results_dir: Path,
    model: str,
    device: str | None = None,
    chunked_csv: Path | None = None,
) -> pd.DataFrame:
    """Load attn, ssm_triton, ssm_torch CSVs and tag them with explicit layer_type.

    File naming conventions (from run_*_prefill_sweep.py):
      attn_scaling_{model}_*.csv              → layer_type = attn
      ssm_scaling_{model}_*.csv               → layer_type = ssm_triton  (no _torchscan)
      ssm_scaling_{model}_*_torchscan.csv     → layer_type = ssm_torch

    When multiple CSVs match (e.g. measurements from different GPU SKUs), exactly
    ONE file is selected per layer type via _pick_best_csv().  Pass ``device`` to
    pin a specific hardware tag (e.g. ``"a100-sxm4-80gb"``).

    Mixing CSVs from different hardware produces jagged latency curves because:
      • Different SM sweep step sets interleave when sorted → sawtooth on the x-axis
      • Different full-SM baseline latencies → wave-model predictions diverge at each SM step
      • Green Context effectiveness differs across hardware → some files show flat ~0ms
        curves (restriction ineffective) next to files with genuine SM scaling
    """
    attn_files = sorted(results_dir.glob(f"attn_scaling_{model}_*.csv"))
    ssm_files  = sorted(
        f for f in results_dir.glob(f"ssm_scaling_{model}_*.csv")
        if "_torchscan" not in f.name
    )
    torch_files = sorted(results_dir.glob(f"ssm_scaling_{model}_*_torchscan.csv"))

    # Apply device filter when specified
    if device:
        attn_files  = [f for f in attn_files  if device in f.name]
        ssm_files   = [f for f in ssm_files   if device in f.name]
        torch_files = [f for f in torch_files if device in f.name]

    specs = [
        ("attn",       attn_files),
        ("ssm_triton", ssm_files),
        ("ssm_torch",  torch_files),
    ]

    dfs = []
    for module_type, files in specs:
        if not files:
            continue
        if len(files) > 1:
            chosen = _pick_best_csv(files, module_type)
            skipped = [f.name for f in files if f != chosen]
            print(
                f"  [load_all_modules] {module_type}: {len(files)} CSV files found — "
                f"using '{chosen.name}'\n"
                f"    Skipped: {skipped}\n"
                f"    (use --device to pin a specific hardware tag, e.g. "
                f"--device {chosen.stem.split('_', 3)[-1]})"
            )
        else:
            chosen = files[0]
        df = pd.read_csv(chosen)
        df["layer_type"] = module_type
        df["_source_file"] = chosen.name
        dfs.append(df)

    # chunked prefill CSV (optional — passed explicitly via --ssm-chunked-csv)
    if chunked_csv is not None and chunked_csv.exists():
        ch_df = load_chunked_csv(chunked_csv, model)
        if not ch_df.empty:
            ch_df["_source_file"] = chunked_csv.name
            dfs.append(ch_df)
            print(
                f"  [load_all_modules] ssm_chunked: loaded '{chunked_csv.name}' "
                f"({len(ch_df)} rows, cooperative_safe=True only)"
            )
        else:
            print(
                f"  [load_all_modules] ssm_chunked: '{chunked_csv.name}' loaded but "
                f"no cooperative_safe=True rows for model={model!r}"
            )

    if not dfs:
        raise FileNotFoundError(
            f"No attn / ssm / ssm_torchscan CSVs for model={model!r} in {results_dir}"
        )

    combined = pd.concat(dfs, ignore_index=True)

    # Drop error rows
    if "error" in combined.columns:
        combined = combined[
            combined["error"].isna() | (combined["error"] == "")
        ].copy()

    # Normalize model_name column
    if "model_name" not in combined.columns and "model" in combined.columns:
        combined = combined.rename(columns={"model": "model_name"})

    combined = compute_throughput(combined)
    return combined


# ---------------------------------------------------------------------------
# Figure 4: Module Latency Comparison
# ---------------------------------------------------------------------------

def plot_module_latency_comparison(
    df: pd.DataFrame,
    output_dir: Path,
    model: str,
    batch_sizes: list[int],
    seq_lens: list[int],
) -> None:
    """Fig 4: Latency vs SM allocation for all modules on one panel per (bs, seq_len).

    Grid: rows = batch_sizes, cols = seq_lens.
    Lines: attn (dash-dot ▲), ssm_triton (solid ●), ssm_torch (dashed ■).
    ✕ markers on each line at its saturation point.
    """
    df_m = df[df["model_name"] == model].copy()
    modules = [m for m in MODULE_ORDER if m in df_m["layer_type"].unique()]

    if not modules:
        print(f"  [Fig 4] No data for model={model}")
        return

    n_rows = len(batch_sizes)
    n_cols = len(seq_lens)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.4 * n_cols, 3.8 * n_rows),
        squeeze=False,
        layout="constrained",
    )

    model_label = model.upper().replace("_", "-")
    fig.suptitle(
        f"{model_label} — Module Latency vs SM Allocation"
        f"  (✕ = saturation, < {SATURATION_THRESHOLD*100:.0f}% gain per 10% SM)",
        fontsize=12, fontweight="bold",
    )

    for ri, bs in enumerate(batch_sizes):
        for ci, sl in enumerate(seq_lens):
            ax = axes[ri][ci]
            ax.set_title(f"seq={sl}, bs={bs}", fontsize=8)
            has_data = False

            for mod in modules:
                grp = df_m[
                    (df_m["layer_type"] == mod) &
                    (df_m["batch_size"] == bs) &
                    (df_m["seq_len"] == sl)
                ].sort_values("sm_ratio")
                if grp.empty:
                    continue
                has_data = True

                ax.plot(
                    grp["sm_ratio"] * 100,
                    grp["latency_ms"],
                    color=MODULE_COLORS[mod],
                    linestyle=MODULE_LINESTYLES[mod],
                    marker=MODULE_MARKERS[mod],
                    linewidth=2, markersize=3,
                    label=MODULE_LABELS[mod],
                )

                sat_sm = find_saturation_sm(grp)
                sat_row = grp[grp["sm_count"] == sat_sm]
                if not sat_row.empty:
                    ax.scatter(
                        [sat_row["sm_ratio"].values[0] * 100],
                        [sat_row["latency_ms"].values[0]],
                        color=MODULE_COLORS[mod],
                        marker="x", s=90, linewidths=2.5, zorder=6,
                    )

            if not has_data:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8, color="gray")

            ax.set_xlabel("SM Allocation (%)", fontsize=8)
            ax.set_ylabel("Latency (ms)", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 105)

    # Figure-level legend
    handles = [
        mlines.Line2D(
            [], [],
            color=MODULE_COLORS[m],
            linestyle=MODULE_LINESTYLES[m],
            marker=MODULE_MARKERS[m],
            linewidth=2, markersize=5,
            label=MODULE_LABELS[m],
        )
        for m in modules
    ]
    handles.append(
        mlines.Line2D([], [], color="gray", marker="x", linestyle="None",
                      markersize=7, linewidth=2, label="Saturation"),
    )
    fig.legend(
        handles=handles, loc="lower center",
        ncol=len(handles), fontsize=9,
    )

    out_path = output_dir / f"fig4_module_latency_{model}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 5: SSM Wave-Model Validation
# ---------------------------------------------------------------------------

def plot_ssm_validation(
    df: pd.DataFrame,
    output_dir: Path,
    model: str,
) -> None:
    """Fig 5: ssm_triton analytical vs ssm_torch direct-measurement scatter.

    Matched on (sm_count, seq_len, batch_size).  Identity line y = x represents
    perfect agreement.  RMSE and MAPE are annotated.
    Color: SM ratio (plasma).  Marker shape: batch_size.
    Two panels: linear and log scales for complementary views.
    """
    triton = df[(df["model_name"] == model) & (df["layer_type"] == "ssm_triton")].copy()
    torch_ = df[(df["model_name"] == model) & (df["layer_type"] == "ssm_torch")].copy()

    if triton.empty:
        print(f"  [Fig 5] No ssm_triton data for model={model} — skipping")
        return
    if torch_.empty:
        print(f"  [Fig 5] No ssm_torch data for model={model} — skipping")
        return

    merge_keys = ["sm_count", "seq_len", "batch_size"]
    merged = triton[merge_keys + ["latency_ms", "sm_ratio"]].merge(
        torch_[merge_keys + ["latency_ms"]].rename(columns={"latency_ms": "lat_torch"}),
        on=merge_keys,
        how="inner",
    )

    if merged.empty:
        print(
            f"  [Fig 5] No matching (sm_count, seq_len, batch_size) between "
            f"ssm_triton and ssm_torch for model={model} — skipping"
        )
        return

    sm_ratios   = sorted(merged["sm_ratio"].unique())
    batch_sizes = sorted(merged["batch_size"].unique())

    cmap = plt.cm.plasma
    ratio_colors = {
        r: cmap(i / max(len(sm_ratios) - 1, 1))
        for i, r in enumerate(sm_ratios)
    }
    bs_marker_list = ["o", "s", "D", "^", "v", "P", "X", "*"]
    bs_markers = {bs: bs_marker_list[i % len(bs_marker_list)]
                  for i, bs in enumerate(batch_sizes)}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, xscale in zip(axes, ["linear", "log"]):
        for _, row in merged.iterrows():
            ax.scatter(
                row["latency_ms"], row["lat_torch"],
                color=ratio_colors[row["sm_ratio"]],
                marker=bs_markers.get(int(row["batch_size"]), "o"),
                s=45, alpha=0.82, linewidths=0.5, edgecolors="white",
                zorder=4,
            )

        lo = min(merged["latency_ms"].min(), merged["lat_torch"].min()) * 0.88
        hi = max(merged["latency_ms"].max(), merged["lat_torch"].max()) * 1.12
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.2, alpha=0.55, label="y = x (perfect)")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

        if xscale == "log":
            ax.set_xscale("log")
            ax.set_yscale("log")

        ax.set_xlabel("SSM Triton — analytical latency (ms)", fontsize=10)
        ax.set_ylabel("SSM PyTorch — direct latency (ms)", fontsize=10)
        ax.set_title(f"{'Log scale' if xscale == 'log' else 'Linear scale'}", fontsize=10)
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=8)

    fig.suptitle(
        f"SSM Wave-Model Validation — {model}\n"
        f"Triton analytical  vs  PyTorch direct measurement\n"
        f"(points on y = x → wave model is accurate)",
        fontsize=11, fontweight="bold",
    )

    # SM ratio color legend
    sm_handles = [
        mpatches.Patch(color=ratio_colors[r], label=f"SM {r:.0%}")
        for r in sm_ratios
    ]
    bs_handles = [
        mlines.Line2D([], [], color="gray", marker=bs_markers[bs],
                      linestyle="None", markersize=7, label=f"bs={bs}")
        for bs in batch_sizes
    ]
    fig.legend(
        handles=sm_handles,
        title="SM ratio", loc="lower center",
        ncol=min(len(sm_ratios), 9),
        bbox_to_anchor=(0.35, -0.06),
        fontsize=8, title_fontsize=8,
    )
    axes[-1].legend(
        handles=bs_handles,
        title="batch_size", loc="lower right",
        fontsize=8, title_fontsize=8,
    )

    # Summary statistics
    rmse = float(np.sqrt(((merged["latency_ms"] - merged["lat_torch"]) ** 2).mean()))
    mape = float(
        ((merged["latency_ms"] - merged["lat_torch"]).abs()
         / merged["lat_torch"].replace(0, np.nan) * 100).mean()
    )
    fig.text(
        0.5, -0.09,
        f"RMSE = {rmse:.3f} ms  |  MAPE = {mape:.2f}%  "
        f"({len(merged)} matched points across {len(sm_ratios)} SM steps)",
        ha="center", fontsize=9, style="italic", color="#333",
    )

    plt.tight_layout()
    out_path = output_dir / f"fig5_ssm_validation_{model}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}  [RMSE={rmse:.3f}ms  MAPE={mape:.2f}%  n={len(merged)}]")


# ---------------------------------------------------------------------------
# Figure 6: Saturation SM Ratio Heatmap
# ---------------------------------------------------------------------------

def plot_saturation_heatmap(
    df: pd.DataFrame,
    output_dir: Path,
    model: str,
) -> None:
    """Fig 6: Heatmap of saturation_SM_ratio over seq_len × batch_size per module.

    Three panels (attn, ssm_triton, ssm_torch).  Cell value = fraction of total
    SMs required to reach saturation.  Lower = saturates earlier = more free SM
    available for concurrent decode tasks.
    """
    df_m = df[df["model_name"] == model].copy()
    modules = [m for m in MODULE_ORDER if m in df_m["layer_type"].unique()]

    if not modules:
        print(f"  [Fig 6] No data for model={model}")
        return

    fig, axes = plt.subplots(1, len(modules), figsize=(5.2 * len(modules), 4.2))
    if len(modules) == 1:
        axes = [axes]

    fig.suptitle(
        f"Saturation SM Ratio Heatmap — {model}\n"
        f"Cell value = fraction of total SMs at saturation  "
        f"(lower → more decode budget)",
        fontsize=11, fontweight="bold",
    )

    for ax, mod in zip(axes, modules):
        df_mod = df_m[df_m["layer_type"] == mod]
        seq_lens    = sorted(df_mod["seq_len"].unique())
        batch_sizes = sorted(df_mod["batch_size"].unique())
        total_sm    = int(df_mod["sm_count"].max())

        mat = np.full((len(batch_sizes), len(seq_lens)), np.nan)
        for i, bs in enumerate(batch_sizes):
            for j, sl in enumerate(seq_lens):
                grp = df_mod[
                    (df_mod["batch_size"] == bs) & (df_mod["seq_len"] == sl)
                ]
                if len(grp) < 2:
                    continue
                sat_sm = find_saturation_sm(grp)
                mat[i, j] = sat_sm / total_sm

        im = ax.imshow(mat, vmin=0.0, vmax=1.0, cmap="RdYlGn_r", aspect="auto")

        ax.set_xticks(range(len(seq_lens)))
        ax.set_xticklabels(seq_lens, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(batch_sizes)))
        ax.set_yticklabels(batch_sizes, fontsize=8)
        ax.set_xlabel("Sequence Length", fontsize=9)
        ax.set_ylabel("Batch Size", fontsize=9)
        ax.set_title(MODULE_LABELS.get(mod, mod), fontsize=10, fontweight="bold")

        for i in range(len(batch_sizes)):
            for j in range(len(seq_lens)):
                v = mat[i, j]
                if np.isfinite(v):
                    text_color = "white" if v > 0.62 else "black"
                    ax.text(j, i, f"{v:.0%}", ha="center", va="center",
                            fontsize=7, color=text_color, fontweight="bold")

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label="Saturation SM ratio")

    plt.tight_layout()
    out_path = output_dir / f"fig6_saturation_heatmap_{model}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 7: SM Sensitivity (latency degradation at reduced SM)
# ---------------------------------------------------------------------------

def plot_sm_sensitivity(
    df: pd.DataFrame,
    output_dir: Path,
    model: str,
    batch_sizes: list[int],
    target_ratios: list[float] | None = None,
) -> None:
    """Fig 7: Mean latency increase (%) per module when SM is reduced.

    For each (module, seq_len, batch_size):
      sensitivity(r) = (lat_at_r − lat_at_100%) / lat_at_100% × 100

    SM count at ratio r is obtained by linear interpolation over the measured
    SM steps.  One subplot per target SM ratio; within each subplot, grouped
    bars show modules side-by-side, with seq_len on the x-axis.
    """
    if target_ratios is None:
        target_ratios = [0.75, 0.50, 0.25]

    df_m = df[df["model_name"] == model].copy()
    modules = [m for m in MODULE_ORDER if m in df_m["layer_type"].unique()]

    if not modules:
        print(f"  [Fig 7] No data for model={model}")
        return

    seq_lens = sorted(df_m["seq_len"].unique())

    n_ratios = len(target_ratios)
    fig, axes = plt.subplots(1, n_ratios, figsize=(6.5 * n_ratios, 5), squeeze=False)
    axes = axes[0]

    model_label = model.upper().replace("_", "-")
    fig.suptitle(
        f"{model_label} — SM Sensitivity\n"
        f"Mean latency increase (%) when SM is reduced to target ratio\n"
        f"(averaged over batch_sizes {batch_sizes})",
        fontsize=12, fontweight="bold",
    )

    for ax, r_target in zip(axes, target_ratios):
        records: list[dict] = []
        for mod in modules:
            df_mod = df_m[df_m["layer_type"] == mod]
            for bs in batch_sizes:
                for sl in seq_lens:
                    grp = df_mod[
                        (df_mod["batch_size"] == bs) & (df_mod["seq_len"] == sl)
                    ].sort_values("sm_count")
                    if len(grp) < 2:
                        continue

                    sm_full = float(grp["sm_count"].max())
                    lat_full_rows = grp[grp["sm_count"] == sm_full]["latency_ms"].values
                    if len(lat_full_rows) == 0:
                        continue
                    lat_full = float(lat_full_rows[0])

                    sm_target = r_target * sm_full
                    lat_target = float(
                        np.interp(sm_target,
                                  grp["sm_count"].values.astype(float),
                                  grp["latency_ms"].values.astype(float))
                    )
                    sensitivity = (lat_target - lat_full) / lat_full * 100.0
                    records.append({
                        "module": mod,
                        "batch_size": bs,
                        "seq_len": sl,
                        "sensitivity": sensitivity,
                    })

        if not records:
            ax.set_visible(False)
            continue

        rec_df = pd.DataFrame(records)

        x = np.arange(len(seq_lens))
        n_mods = len(modules)
        width = 0.72 / n_mods

        for mi, mod in enumerate(modules):
            mod_df = rec_df[rec_df["module"] == mod]
            means = [
                mod_df[mod_df["seq_len"] == sl]["sensitivity"].mean()
                if not mod_df[mod_df["seq_len"] == sl].empty else 0.0
                for sl in seq_lens
            ]
            offset = (mi - (n_mods - 1) / 2) * width
            ax.bar(
                x + offset, means, width,
                color=MODULE_COLORS[mod], alpha=0.85,
                label=MODULE_LABELS[mod],
            )

        ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("Sequence Length", fontsize=9)
        ax.set_ylabel("Latency increase (%)", fontsize=9)
        ax.set_title(f"SM reduced to {r_target:.0%}", fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(seq_lens, rotation=30, fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        if ax is axes[0]:
            ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = output_dir / f"fig7_sm_sensitivity_{model}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 8: Chunked Prefill Cooperative Safety Map
# ---------------------------------------------------------------------------

def _pcolormesh_edges(vals: list) -> np.ndarray:
    """Return bin-edge array suitable for pcolormesh from a sorted list of centers."""
    v = np.array(sorted(set(vals)), dtype=float)
    if len(v) == 1:
        return np.array([v[0] * 0.8, v[0] * 1.2])
    d = np.diff(v) / 2.0
    return np.concatenate([[v[0] - d[0]], v[:-1] + d, [v[-1] + d[-1]]])


def plot_chunked_safety_map(
    chunked_df: pd.DataFrame,
    output_dir: Path,
    model: str,
) -> None:
    """Fig 8: sm_count × prefill_chunk_tokens safety heatmap, one panel per batch_size.

    Green = cooperative_safe=True (n_blocks_per_call ≤ sm_count).
    Red   = cooperative_safe=False (cooperative barrier deadlock risk).
    Dashed line = theoretical boundary: chunk_tokens = 4 × sm_count / batch_size.
    """
    if chunked_df.empty:
        print(f"  [Fig 8] No chunked data for model={model} — skipping")
        return
    required = {"prefill_chunk_tokens", "cooperative_safe", "sm_count", "batch_size"}
    if not required.issubset(chunked_df.columns):
        missing = required - set(chunked_df.columns)
        print(f"  [Fig 8] Missing columns {missing} — skipping")
        return

    df_m = (
        chunked_df[chunked_df["model_name"] == model].copy()
        if "model_name" in chunked_df.columns
        else chunked_df.copy()
    )
    if df_m.empty:
        print(f"  [Fig 8] No rows for model={model} — skipping")
        return

    batch_sizes = sorted(df_m["batch_size"].unique())
    n_panels = len(batch_sizes)
    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(5.5 * n_panels, 5.2),
        squeeze=False,
        layout="constrained",
    )
    axes = axes[0]

    model_label = model.upper().replace("_", "-")
    fig.suptitle(f"{model_label} — Cooperative Safety Map", fontsize=13, fontweight="bold")
    fig.supxlabel(
        "Green = safe (n_blocks_per_call ≤ SM count)  |  Red = deadlock risk  |  Dashed = safety boundary",
        fontsize=9, color="#555",
    )

    safe_cmap = mcolors.ListedColormap(["#EF5350", "#66BB6A"])  # red=0, green=1

    for ax, bs in zip(axes, batch_sizes):
        sub = df_m[df_m["batch_size"] == bs]
        chunk_vals = sorted(sub["prefill_chunk_tokens"].unique())
        sm_vals    = sorted(sub["sm_count"].unique())

        mat = np.full((len(sm_vals), len(chunk_vals)), np.nan)
        for i, sm in enumerate(sm_vals):
            for j, ch in enumerate(chunk_vals):
                cell = sub[(sub["sm_count"] == sm) & (sub["prefill_chunk_tokens"] == ch)]
                if not cell.empty:
                    mat[i, j] = float(cell["cooperative_safe"].astype(float).iloc[0])

        im = ax.pcolormesh(
            _pcolormesh_edges(chunk_vals),
            _pcolormesh_edges(sm_vals),
            mat,
            cmap=safe_cmap, vmin=0.0, vmax=1.0,
        )

        # Safety boundary derived from actual n_blocks_per_call data (model-agnostic).
        # n_blocks scales linearly with chunk_tokens; boundary: chunk = sm * min_chunk / n_blocks_at_min
        legend_handles = []
        if "n_blocks_per_call" in sub.columns:
            min_chunk = float(chunk_vals[0])
            ref_rows = sub[sub["prefill_chunk_tokens"] == chunk_vals[0]]
            if not ref_rows.empty:
                n_blocks_at_min = float(ref_rows["n_blocks_per_call"].iloc[0])
                sm_arr = np.array(sm_vals, dtype=float)
                boundary_ch = sm_arr * min_chunk / n_blocks_at_min

                ch_lo = _pcolormesh_edges(chunk_vals)[0]
                ch_hi = _pcolormesh_edges(chunk_vals)[-1]
                visible = (boundary_ch >= ch_lo) & (boundary_ch <= ch_hi)

                if visible.any():
                    ax.plot(
                        boundary_ch[visible], sm_arr[visible],
                        "k--", linewidth=2.0, zorder=5,
                    )
                    legend_handles.append(
                        mlines.Line2D([], [], color="black", linestyle="--",
                                      linewidth=2, label="Safety boundary")
                    )
                else:
                    # Boundary is outside chart — annotate with required SM count
                    min_safe_sm = int(np.ceil(n_blocks_at_min))
                    ax.text(
                        0.5, 0.96,
                        f"⚠ All configs unsafe\n(requires SM ≥ {min_safe_sm} for chunk={int(min_chunk)})",
                        ha="center", va="top", transform=ax.transAxes,
                        fontsize=7.5, color="#B71C1C",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFCDD2", alpha=0.9),
                    )

        ax.set_xlabel("Prefill Chunk Tokens", fontsize=9)
        ax.set_ylabel("SM Count", fontsize=9)
        ax.set_title(f"batch_size = {bs}", fontsize=10, fontweight="bold")
        ax.set_xticks(chunk_vals)
        ax.set_xticklabels([str(c) for c in chunk_vals], rotation=30, ha="right", fontsize=8)
        ax.set_yticks(sm_vals)
        ax.set_yticklabels([str(s) for s in sm_vals], fontsize=8)
        if legend_handles:
            ax.legend(handles=legend_handles, fontsize=8, loc="upper left")

    # Shared colorbar
    sm_mappable = plt.cm.ScalarMappable(
        cmap=safe_cmap, norm=mcolors.Normalize(vmin=0, vmax=1)
    )
    cbar = fig.colorbar(sm_mappable, ax=list(axes), fraction=0.02, pad=0.01)
    cbar.set_ticks([0.25, 0.75])
    cbar.set_ticklabels(["Unsafe", "Safe"], fontsize=10)

    out_path = output_dir / f"fig8_chunked_safety_{model}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 9: Chunked Prefill SM Scaling
# ---------------------------------------------------------------------------

def plot_chunked_sm_scaling(
    chunked_df: pd.DataFrame,
    main_df: pd.DataFrame,
    output_dir: Path,
    model: str,
    seq_lens: list[int],
    batch_sizes: list[int],
) -> None:
    """Fig 9: Latency vs SM allocation ratio, lines per chunk size (all configs).

    Grid: rows = batch_sizes, cols = seq_lens.
    Filled markers = cooperative_safe=True; hollow markers = cooperative_safe=False.
    SSM Triton full-sequence reference overlaid as a gray dashed curve.
    """
    if chunked_df.empty:
        print(f"  [Fig 9] No chunked data for model={model} — skipping")
        return
    if "prefill_chunk_tokens" not in chunked_df.columns:
        print(f"  [Fig 9] Missing prefill_chunk_tokens column — skipping")
        return

    df_m = (
        chunked_df[chunked_df["model_name"] == model].copy()
        if "model_name" in chunked_df.columns
        else chunked_df.copy()
    )
    if df_m.empty:
        print(f"  [Fig 9] No rows for model={model} — skipping")
        return

    avail_bs = sorted(df_m["batch_size"].unique())
    avail_sl = sorted(df_m["seq_len"].unique())
    plot_bs = [b for b in batch_sizes if b in avail_bs][:4]
    plot_sl = [s for s in seq_lens if s in avail_sl][:5]
    if not plot_bs or not plot_sl:
        print(f"  [Fig 9] No matching (batch_size, seq_len) in chunked data — skipping")
        return

    chunk_vals = sorted(df_m["prefill_chunk_tokens"].unique())
    cmap = plt.cm.viridis
    chunk_colors = {
        ch: cmap(i / max(len(chunk_vals) - 1, 1))
        for i, ch in enumerate(chunk_vals)
    }

    # SSM Triton reference
    triton_df = main_df[
        (main_df.get("model_name", pd.Series([model] * len(main_df), index=main_df.index)) == model)
        & (main_df["layer_type"] == "ssm_triton")
    ] if "model_name" in main_df.columns else main_df[main_df["layer_type"] == "ssm_triton"]

    n_rows, n_cols = len(plot_bs), len(plot_sl)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.4 * n_cols, 3.8 * n_rows),
        squeeze=False,
        layout="constrained",
    )

    has_safe_col = "cooperative_safe" in df_m.columns
    model_label = model.upper().replace("_", "-")
    safe_note = "  ●=safe  ○=unsafe" if has_safe_col else ""
    fig.suptitle(
        f"{model_label} — Chunked Prefill: Latency vs SM Allocation",
        fontsize=12, fontweight="bold",
    )
    fig.supxlabel(
        f"Colored lines = chunk sizes (all configs).{safe_note}  "
        "Gray dashed = SSM Triton full-seq ref.",
        fontsize=9, color="#555",
    )

    for ri, bs in enumerate(plot_bs):
        for ci, sl in enumerate(plot_sl):
            ax = axes[ri][ci]
            ax.set_title(f"seq={sl}, bs={bs}", fontsize=8)
            has_data = False

            ref = triton_df[
                (triton_df["batch_size"] == bs) & (triton_df["seq_len"] == sl)
            ].sort_values("sm_ratio")
            if not ref.empty:
                ax.plot(
                    ref["sm_ratio"] * 100, ref["latency_ms"],
                    color="gray", linestyle="--", linewidth=1.5, alpha=0.7,
                    label="SSM Triton (ref)",
                )

            for ch in chunk_vals:
                grp = df_m[
                    (df_m["batch_size"] == bs) &
                    (df_m["seq_len"] == sl) &
                    (df_m["prefill_chunk_tokens"] == ch)
                ].sort_values("sm_ratio")
                if grp.empty:
                    continue
                has_data = True
                # Connecting line
                ax.plot(
                    grp["sm_ratio"] * 100, grp["latency_ms"],
                    color=chunk_colors[ch], linestyle="-", linewidth=1.5, alpha=0.6,
                )
                # Safety-aware markers: filled = safe, hollow = unsafe
                if has_safe_col:
                    safe_mask = grp["cooperative_safe"].astype(bool)
                    for pts, facecolor in [
                        (grp[safe_mask],  chunk_colors[ch]),
                        (grp[~safe_mask], "none"),
                    ]:
                        if not pts.empty:
                            ax.scatter(
                                pts["sm_ratio"] * 100, pts["latency_ms"],
                                s=18, color=chunk_colors[ch],
                                facecolors=facecolor,
                                edgecolors=chunk_colors[ch],
                                linewidths=1.2, zorder=4,
                            )
                else:
                    ax.scatter(grp["sm_ratio"] * 100, grp["latency_ms"],
                               s=18, color=chunk_colors[ch], zorder=4)

            if not has_data:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8, color="gray")

            ax.set_xlabel("SM Allocation (%)", fontsize=7)
            ax.set_ylabel("Latency (ms)", fontsize=7)
            ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 105)

    chunk_handles = [
        mlines.Line2D([], [], color=chunk_colors[ch], marker="o", linestyle="-",
                      markersize=5, linewidth=2, label=f"chunk={ch}")
        for ch in chunk_vals
    ]
    chunk_handles.append(
        mlines.Line2D([], [], color="gray", linestyle="--", linewidth=1.5,
                      label="SSM Triton (ref)")
    )
    if has_safe_col:
        chunk_handles += [
            mlines.Line2D([], [], color="black", marker="o", linestyle="None",
                          markersize=6, label="safe (●)"),
            mlines.Line2D([], [], color="black", marker="o", linestyle="None",
                          markersize=6, markerfacecolor="none", label="unsafe (○)"),
        ]
    fig.legend(handles=chunk_handles, loc="outside lower center",
               ncol=min(len(chunk_handles), 6), fontsize=9)

    out_path = output_dir / f"fig9_chunked_sm_scaling_{model}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 10: Chunked Prefill Kernel Overhead
# ---------------------------------------------------------------------------

def plot_chunked_kernel_overhead(
    chunked_df: pd.DataFrame,
    output_dir: Path,
    model: str,
) -> None:
    """Fig 10: Latency vs n_kernel_calls with linear fit per SM count, one panel per batch_size.

    n_kernel_calls = seq_len / prefill_chunk_tokens.
    A linear fit slope estimates per-call latency overhead;
    y-intercept estimates fixed dispatch overhead.
    """
    if chunked_df.empty:
        print(f"  [Fig 10] No chunked data for model={model} — skipping")
        return
    if "n_kernel_calls" not in chunked_df.columns:
        print(f"  [Fig 10] Missing n_kernel_calls column — skipping")
        return

    df_m = (
        chunked_df[chunked_df["model_name"] == model].copy()
        if "model_name" in chunked_df.columns
        else chunked_df.copy()
    )
    if df_m.empty:
        print(f"  [Fig 10] No rows for model={model} — skipping")
        return

    sm_vals    = sorted(df_m["sm_count"].unique())
    batch_sizes = sorted(df_m["batch_size"].unique())
    n_panels = len(batch_sizes)

    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(5.8 * n_panels, 5.0),
        squeeze=False,
        layout="constrained",
    )
    axes = axes[0]

    has_safe_col = "cooperative_safe" in df_m.columns
    model_label = model.upper().replace("_", "-")
    safe_note = "  ●=safe  ○=unsafe" if has_safe_col else ""
    fig.suptitle(
        f"{model_label} — Chunked Prefill: Latency vs Kernel Call Count",
        fontsize=13, fontweight="bold",
    )
    fig.supxlabel(
        f"Points = one (seq_len, chunk_size) config.{safe_note}  "
        "Lines = linear fit; slope shown as annotation.",
        fontsize=9, color="#555",
    )

    cmap = plt.cm.plasma
    sm_colors = {sm: cmap(i / max(len(sm_vals) - 1, 1)) for i, sm in enumerate(sm_vals)}

    for ax, bs in zip(axes, batch_sizes):
        sub = df_m[df_m["batch_size"] == bs]

        for sm in sm_vals:
            grp = sub[sub["sm_count"] == sm].sort_values("n_kernel_calls")
            if grp.empty:
                continue

            # Safety-aware markers: filled = safe, hollow = unsafe
            if has_safe_col:
                safe_mask = grp["cooperative_safe"].astype(bool)
                for pts, facecolor in [
                    (grp[safe_mask],  sm_colors[sm]),
                    (grp[~safe_mask], "none"),
                ]:
                    if not pts.empty:
                        ax.scatter(
                            pts["n_kernel_calls"], pts["latency_ms"],
                            s=40, color=sm_colors[sm],
                            facecolors=facecolor,
                            edgecolors=sm_colors[sm],
                            linewidths=1.2, alpha=0.85, zorder=4,
                        )
            else:
                ax.scatter(
                    grp["n_kernel_calls"], grp["latency_ms"],
                    color=sm_colors[sm], s=40, alpha=0.82,
                    edgecolors="white", linewidths=0.5, zorder=4,
                )

            if len(grp) >= 2:
                x = grp["n_kernel_calls"].values.astype(float)
                y = grp["latency_ms"].values.astype(float)
                coeffs = np.polyfit(x, y, 1)
                x_fit = np.linspace(x.min(), x.max(), 60)
                y_fit = np.polyval(coeffs, x_fit)
                ax.plot(x_fit, y_fit, color=sm_colors[sm], linewidth=1.5,
                        alpha=0.72, label=f"SM={sm}")
                # Slope annotation at the midpoint of the fit line
                mid = len(x_fit) // 2
                ax.annotate(
                    f"{coeffs[0]:.2f}ms/call",
                    xy=(x_fit[mid], y_fit[mid]),
                    xytext=(3, 3), textcoords="offset points",
                    fontsize=6, color=sm_colors[sm], alpha=0.85,
                )

        ax.set_xlabel("Kernel Calls  (seq_len / chunk_tokens)", fontsize=9)
        ax.set_ylabel("Latency (ms)", fontsize=9)
        ax.set_title(f"batch_size = {bs}", fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3)

    # Shared figure-level legend (one per SM count)
    handles_leg = [
        mlines.Line2D([], [], color=sm_colors[sm], linewidth=2, label=f"SM={sm}")
        for sm in sm_vals
    ]
    if has_safe_col:
        handles_leg += [
            mpatches.Patch(color="none", label=""),   # spacer
            mlines.Line2D([], [], color="gray", marker="o", linestyle="None",
                          markersize=6, label="safe (●)"),
            mlines.Line2D([], [], color="gray", marker="o", linestyle="None",
                          markersize=6, markerfacecolor="none", label="unsafe (○)"),
        ]
    fig.legend(handles=handles_leg, loc="outside right upper",
               ncol=1, fontsize=8, title="SM count", title_fontsize=8)
    out_path = output_dir / f"fig10_chunked_overhead_{model}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Stage 1 all-module comparison: "
            "attn, ssm_triton (analytical), ssm_torch (direct), ssm_chunked (direct)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--results-dir", type=Path,
        default=Path(__file__).parent.parent / "results" / "stage1",
    )
    parser.add_argument("--model", default=None,
                        help="Model name (default: auto-detect from CSV filenames)")
    parser.add_argument(
        "--device", default=None,
        help=(
            "Hardware tag to filter CSVs (e.g. 'a100-sxm4-80gb'). "
            "When multiple CSVs exist for a model, the best one is auto-selected "
            "but --device lets you pin a specific run."
        ),
    )
    parser.add_argument(
        "--ssm-chunked-csv", type=Path, default=None,
        metavar="CSV",
        help=(
            "SSM CSV from direct chunked-prefill measurement "
            "(output of run_chunked_ssm_sweep.py, not yet implemented). "
            "When provided, overlays ssm_chunked on top of the wave-model ssm_triton curves."
        ),
    )
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=None)
    parser.add_argument("--seq-lens",    nargs="+", type=int, default=None,
                        help="Subset of seq_lens for the latency-comparison grid plot")
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).parent.parent / "results" / "stage1",
    )
    parser.add_argument(
        "--sensitivity-ratios", nargs="+", type=float, default=[0.75, 0.50, 0.25],
        metavar="R",
        help="SM ratios for sensitivity analysis (default: 0.75 0.50 0.25)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect models from CSV filenames
    results_dir = args.results_dir
    available_models: set[str] = set()
    for known in ("zamba2", "falcon_h1"):
        if (
            list(results_dir.glob(f"attn_scaling_{known}_*.csv"))
            or list(results_dir.glob(f"ssm_scaling_{known}_*.csv"))
        ):
            available_models.add(known)

    models = [args.model] if args.model else sorted(available_models)
    if not models:
        print("No model data found in results/stage1.  Run stage1 sweeps first.")
        raise SystemExit(1)

    for model in models:
        print(f"\n{'='*60}")
        print(f"Model: {model}")
        print(f"{'='*60}")

        chunked_csv = getattr(args, "ssm_chunked_csv", None)

        try:
            df = load_all_modules(
                results_dir, model,
                device=args.device,
                chunked_csv=chunked_csv,
            )
        except FileNotFoundError as e:
            print(f"  {e}")
            continue

        found_modules = sorted(df["layer_type"].unique())
        print(f"  Modules     : {found_modules}")
        print(f"  seq_lens    : {sorted(df['seq_len'].unique())}")
        print(f"  batch_sizes : {sorted(df['batch_size'].unique())}")
        print(f"  SM steps    : {sorted(df['sm_count'].unique())}")

        batch_sizes = args.batch_sizes or sorted(df["batch_size"].unique().tolist())
        seq_lens    = args.seq_lens    or sorted(df["seq_len"].unique().tolist())

        # For the grid figure, cap rows/cols to keep the figure readable
        grid_bs = batch_sizes[:3]
        grid_sl = seq_lens[:4]

        print(f"\n  [Fig 4] Module latency comparison grid  "
              f"(bs={grid_bs}, seq={grid_sl}) …")
        plot_module_latency_comparison(df, args.output_dir, model, grid_bs, grid_sl)

        print(f"\n  [Fig 5] SSM wave-model validation scatter …")
        plot_ssm_validation(df, args.output_dir, model)

        print(f"\n  [Fig 6] Saturation SM ratio heatmap …")
        plot_saturation_heatmap(df, args.output_dir, model)

        print(f"\n  [Fig 7] SM sensitivity analysis …")
        plot_sm_sensitivity(
            df, args.output_dir, model,
            batch_sizes, args.sensitivity_ratios,
        )

        # Chunked-prefill figures — only when --ssm-chunked-csv is provided
        if chunked_csv is not None and chunked_csv.exists():
            print(f"\n  [Fig 8] Chunked prefill cooperative safety map …")
            chunked_df_all = load_chunked_csv(chunked_csv, model, filter_unsafe=False)
            plot_chunked_safety_map(chunked_df_all, args.output_dir, model)

            print(f"\n  [Fig 9] Chunked prefill SM scaling (all configs, safety-coded) …")
            plot_chunked_sm_scaling(
                chunked_df_all, df, args.output_dir, model, grid_sl, batch_sizes,
            )

            print(f"\n  [Fig 10] Chunked prefill kernel call overhead …")
            plot_chunked_kernel_overhead(chunked_df_all, args.output_dir, model)

    print("\nDone.")
