"""
SM Split Latency Trade-off Plot.

For a fixed total SM budget, models SSM and Attention layers as concurrent
(BulletServe/MuxWise style): SM allocation ratio r_ssm goes to SSM,
r_attn = 1 - r_ssm goes to Attention.

  latency_ssm(r)   = measured latency at sm_count = r × total_sm  (interpolated)
  latency_attn(r)  = measured latency at sm_count = (1−r) × total_sm  (interpolated)
  latency_total(r) = max(latency_ssm, latency_attn)     ← concurrent bottleneck

Minimum of latency_total occurs at the crossover point where
latency_ssm(r*) ≈ latency_attn(r*).

Input CSV files from Stage 1 sweeps:
  results/stage1/ssm_scaling_{model}_{device}.csv
  results/stage1/attn_scaling_{model}_{device}.csv

Usage:
    python stage1_sm_scaling/plot_sm_split.py
    python stage1_sm_scaling/plot_sm_split.py --model zamba2 --seq-lens 1024 4096
    python stage1_sm_scaling/plot_sm_split.py --batch-sizes 1 4 16 --output-dir results/stage1
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def _load_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def find_stage1_csv(results_dir: Path, layer_type: str, model: str) -> Path:
    """Glob for a stage-1 latency CSV.

    Supported layer_type values:
      ssm / ssm_triton  →  ssm_scaling_{model}_*.csv  (no _torchscan suffix)
      ssm_torch         →  ssm_scaling_{model}_*_torchscan.csv
      attn / mlp        →  {layer_type}_scaling_{model}_*.csv
    """
    if layer_type in ("ssm", "ssm_triton"):
        matches = sorted(
            f for f in results_dir.glob(f"ssm_scaling_{model}_*.csv")
            if "_torchscan" not in f.name
        )
        desc = f"ssm_scaling_{model}_*.csv (non-torchscan)"
    elif layer_type == "ssm_torch":
        matches = sorted(results_dir.glob(f"ssm_scaling_{model}_*_torchscan.csv"))
        desc = f"ssm_scaling_{model}_*_torchscan.csv"
    else:
        prefix = f"{layer_type}_scaling_{model}_"
        matches = sorted(results_dir.glob(f"{prefix}*.csv"))
        desc = f"{prefix}*.csv"

    if not matches:
        sweep_script = layer_type.split("_")[0]   # attn / ssm / mlp
        raise FileNotFoundError(
            f"No {layer_type} CSV found in {results_dir}.\n"
            f"  Expected pattern: {desc}\n"
            f"  Run stage1 sweep first:\n"
            f"    python stage1_sm_scaling/run_{sweep_script}_prefill_sweep.py --model {model}"
        )
    return matches[-1]  # pick most recent if multiple


def load_latency_table(
    path: Path,
    seq_len: int,
    batch_size: int,
    latency_col: str = "latency_ms",
) -> tuple[np.ndarray, np.ndarray]:
    """Return (sm_counts, latencies) arrays for fixed (seq_len, batch_size)."""
    rows = _load_csv(path)
    filtered = [
        r for r in rows
        if int(r["seq_len"]) == seq_len and int(r["batch_size"]) == batch_size
    ]
    if not filtered:
        available_seqs = sorted({int(r["seq_len"]) for r in rows})
        available_bs   = sorted({int(r["batch_size"]) for r in rows})
        raise ValueError(
            f"No data for seq_len={seq_len}, batch_size={batch_size} in {path.name}.\n"
            f"  Available seq_lens: {available_seqs}\n"
            f"  Available batch_sizes: {available_bs}"
        )
    sm_counts = np.array([int(r["sm_count"]) for r in filtered], dtype=float)
    latencies = np.array([float(r[latency_col]) for r in filtered], dtype=float)
    order = np.argsort(sm_counts)
    return sm_counts[order], latencies[order]


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------

def interp_latency(
    sm_query: np.ndarray,
    sm_measured: np.ndarray,
    lat_measured: np.ndarray,
) -> np.ndarray:
    """Linear interpolation with linear extrapolation at boundaries.

    Clamping to boundary values is wrong at the low-SM end for Attn:
    when r_ssm is high, Attn gets fewer SMs than the minimum measured
    (11 for Zamba2 Attn), so we must extrapolate upward (slower) not clamp.
    """
    result = np.interp(sm_query, sm_measured, lat_measured,
                       left=lat_measured[0], right=lat_measured[-1])

    # Extrapolate below minimum measured SM using left-edge slope
    if len(sm_measured) >= 2:
        below = sm_query < sm_measured[0]
        if np.any(below):
            slope = (lat_measured[1] - lat_measured[0]) / (sm_measured[1] - sm_measured[0])
            result[below] = lat_measured[0] + slope * (sm_query[below] - sm_measured[0])
            result[below] = np.maximum(result[below], 0.0)

        # Extrapolate above maximum measured SM using right-edge slope
        above = sm_query > sm_measured[-1]
        if np.any(above):
            slope = (lat_measured[-1] - lat_measured[-2]) / (sm_measured[-1] - sm_measured[-2])
            result[above] = lat_measured[-1] + slope * (sm_query[above] - sm_measured[-1])
            result[above] = np.maximum(result[above], 0.0)

    return result


def find_crossover(x: np.ndarray, y1: np.ndarray, y2: np.ndarray) -> float | None:
    """Return x-value where y1 and y2 cross (y1 - y2 changes sign)."""
    diff = y1 - y2
    sign_changes = np.where(np.diff(np.sign(diff)))[0]
    if len(sign_changes) == 0:
        return None
    idx = sign_changes[0]
    # Linear interpolation between idx and idx+1
    x0, x1 = x[idx], x[idx + 1]
    d0, d1 = diff[idx], diff[idx + 1]
    if d1 == d0:
        return float(x0)
    return float(x0 - d0 * (x1 - x0) / (d1 - d0))


# ---------------------------------------------------------------------------
# Single panel
# ---------------------------------------------------------------------------

def _plot_panel(
    ax: plt.Axes,
    total_sm: int,
    sm_ssm: np.ndarray,
    lat_ssm: np.ndarray,
    sm_attn: np.ndarray,
    lat_attn: np.ndarray,
    r_grid: np.ndarray,
    seq_len: int,
    batch_size: int,
    annotate_crossover: bool = True,
    log_scale: bool = False,
) -> None:
    """Draw the three latency curves on a single Axes.

    Bottom x-axis: r_ssm (SM ratio to SSM).
    Top x-axis: corresponding Attn SM count = (1-r) × total_sm.
    """
    sm_ssm_q  = r_grid * total_sm        # SSM SM count
    sm_attn_q = (1.0 - r_grid) * total_sm  # Attn SM count (complement)

    lat_ssm_curve  = interp_latency(sm_ssm_q,  sm_ssm,  lat_ssm)
    lat_attn_curve = interp_latency(sm_attn_q, sm_attn, lat_attn)
    lat_total      = np.maximum(lat_ssm_curve, lat_attn_curve)

    ax.plot(r_grid, lat_ssm_curve,  color="#1f77b4", lw=2,   label="SSM latency", zorder=3)
    ax.plot(r_grid, lat_attn_curve, color="#ff7f0e", lw=2,   label="Attn latency", zorder=3)
    ax.plot(r_grid, lat_total,      color="#2ca02c", lw=2,   label="Total = max(SSM, Attn)", ls="--", zorder=2)

    # Minimum of total latency
    min_idx = np.argmin(lat_total)
    r_opt   = r_grid[min_idx]
    lat_opt = lat_total[min_idx]
    ax.axvline(r_opt, color="#d62728", lw=1.2, ls=":", alpha=0.8, zorder=4)
    ax.scatter([r_opt], [lat_opt], color="#d62728", s=60, zorder=5)

    # Crossover point (SSM = Attn)
    r_cross = find_crossover(r_grid, lat_ssm_curve, lat_attn_curve)

    if annotate_crossover:
        sm_ssm_opt  = int(round(r_opt * total_sm))
        sm_attn_opt = total_sm - sm_ssm_opt
        label_x = r_opt + 0.04 if r_opt < 0.65 else r_opt - 0.32
        ax.annotate(
            f"opt r={r_opt:.2f}\n"
            f"SSM:{sm_ssm_opt}SM Attn:{sm_attn_opt}SM\n"
            f"{lat_opt:.2f} ms",
            xy=(r_opt, lat_opt),
            xytext=(label_x, lat_opt * (1.3 if not log_scale else 2.0)),
            fontsize=6,
            arrowprops=dict(arrowstyle="->", color="#d62728", lw=0.8),
            color="#d62728",
            zorder=6,
        )
        if r_cross is not None:
            lat_cross = np.interp(r_cross, r_grid, lat_total)
            ax.scatter([r_cross], [lat_cross], color="black", s=25,
                       marker="x", zorder=6, linewidths=1.2)

    ax.set_title(f"seq={seq_len}, bs={batch_size}", fontsize=9)
    ax.set_xlabel("SM ratio → SSM  (r_ssm)", fontsize=8)
    ax.set_ylabel("Latency (ms)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlim(0, 1)
    if log_scale:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # Secondary x-axis: Attn SM count (top)
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    attn_ticks_sm = np.array([int(round((1 - r) * total_sm))
                               for r in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]])
    attn_ticks_r  = 1.0 - attn_ticks_sm / total_sm
    ax2.set_xticks(attn_ticks_r)
    ax2.set_xticklabels([str(v) for v in attn_ticks_sm], fontsize=6)
    ax2.set_xlabel("Attn SMs", fontsize=7, labelpad=2)


# ---------------------------------------------------------------------------
# Main figure builder
# ---------------------------------------------------------------------------

def plot_sm_split(
    model: str,
    seq_lens: list[int],
    batch_sizes: list[int],
    results_dir: Path,
    output_dir: Path,
    total_sm: int | None = None,
    r_steps: int = 200,
    log_scale: bool = False,
) -> None:
    ssm_path  = find_stage1_csv(results_dir, "ssm",  model)
    attn_path = find_stage1_csv(results_dir, "attn", model)

    print(f"SSM  data: {ssm_path.name}")
    print(f"Attn data: {attn_path.name}")

    # Infer total SM from the max sm_count in either CSV
    if total_sm is None:
        ssm_rows  = _load_csv(ssm_path)
        attn_rows = _load_csv(attn_path)
        total_sm = max(
            max(int(r["sm_count"]) for r in ssm_rows),
            max(int(r["sm_count"]) for r in attn_rows),
        )
    print(f"Total SM : {total_sm}")

    r_grid = np.linspace(0.01, 0.99, r_steps)   # exclude 0/1 (degenerate)

    n_rows = len(batch_sizes)
    n_cols = len(seq_lens)
    fig_w  = max(5, 3.2 * n_cols)
    fig_h  = max(4, 2.8 * n_rows)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(fig_w, fig_h),
        squeeze=False,
    )

    model_label = model.upper().replace("_", "-")
    scale_tag = " [log scale]" if log_scale else ""
    fig.suptitle(
        f"{model_label} — SM Split Latency Trade-off  (total SM={total_sm}){scale_tag}\n"
        r"$r_{\rm SSM}$ → SSM,  $(1-r_{\rm SSM})$ → Attn  |  "
        r"$L_{\rm total} = \max(L_{\rm SSM},\, L_{\rm Attn})$  [concurrent]  |  "
        r"$\bullet$ = optimal split",
        fontsize=10,
        y=1.01,
    )

    for ri, batch_size in enumerate(batch_sizes):
        for ci, seq_len in enumerate(seq_lens):
            ax = axes[ri][ci]
            try:
                sm_ssm,  lat_ssm  = load_latency_table(ssm_path,  seq_len, batch_size)
                sm_attn, lat_attn = load_latency_table(attn_path, seq_len, batch_size)
            except ValueError as e:
                ax.text(0.5, 0.5, str(e), ha="center", va="center",
                        transform=ax.transAxes, fontsize=7, color="red", wrap=True)
                ax.set_title(f"seq={seq_len}, bs={batch_size}", fontsize=9)
                continue

            _plot_panel(
                ax, total_sm,
                sm_ssm, lat_ssm,
                sm_attn, lat_attn,
                r_grid, seq_len, batch_size,
                annotate_crossover=True,
                log_scale=log_scale,
            )

            # Only show legend on first panel
            if ri == 0 and ci == 0:
                ax.legend(fontsize=7, loc="upper right")

    # Shared legend at figure level (from first valid axes)
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        for ax in axes.flat:
            leg = ax.get_legend()
            if leg:
                leg.remove()
        fig.legend(
            handles, labels,
            loc="lower center",
            ncol=3,
            fontsize=8,
            frameon=True,
            bbox_to_anchor=(0.5, -0.03),
        )

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_log" if log_scale else ""
    out_path = output_dir / f"sm_split_{model}{suffix}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


def plot_sm_split_per_batch(
    model: str,
    seq_lens: list[int],
    batch_sizes: list[int],
    results_dir: Path,
    output_dir: Path,
    total_sm: int | None = None,
    r_steps: int = 200,
    log_scale: bool = False,
) -> None:
    """One figure per batch_size, seq_lens as columns."""
    ssm_path  = find_stage1_csv(results_dir, "ssm",  model)
    attn_path = find_stage1_csv(results_dir, "attn", model)

    if total_sm is None:
        ssm_rows  = _load_csv(ssm_path)
        attn_rows = _load_csv(attn_path)
        total_sm = max(
            max(int(r["sm_count"]) for r in ssm_rows),
            max(int(r["sm_count"]) for r in attn_rows),
        )

    r_grid = np.linspace(0.01, 0.99, r_steps)
    model_label = model.upper().replace("_", "-")
    output_dir.mkdir(parents=True, exist_ok=True)

    for batch_size in batch_sizes:
        n_cols = len(seq_lens)
        fig, axes = plt.subplots(1, n_cols, figsize=(3.5 * n_cols, 3.5), squeeze=False)

        scale_tag = " [log scale]" if log_scale else ""
        fig.suptitle(
            f"{model_label}  |  batch_size={batch_size}  |  total SM={total_sm}{scale_tag}\n"
            r"$r_{\rm SSM}$ → SSM,  $(1-r_{\rm SSM})$ → Attn  "
            r"|  $L_{\rm total}=\max(L_{\rm SSM}, L_{\rm Attn})$  |  $\bullet$ = optimal split",
            fontsize=10, y=1.03,
        )

        for ci, seq_len in enumerate(seq_lens):
            ax = axes[0][ci]
            try:
                sm_ssm,  lat_ssm  = load_latency_table(ssm_path,  seq_len, batch_size)
                sm_attn, lat_attn = load_latency_table(attn_path, seq_len, batch_size)
            except ValueError as e:
                ax.text(0.5, 0.5, str(e), ha="center", va="center",
                        transform=ax.transAxes, fontsize=7, color="red")
                continue

            _plot_panel(
                ax, total_sm,
                sm_ssm, lat_ssm,
                sm_attn, lat_attn,
                r_grid, seq_len, batch_size,
                annotate_crossover=True,
                log_scale=log_scale,
            )

        # Figure-level legend
        handles, labels = axes[0][0].get_legend_handles_labels()
        if handles:
            for ax in axes.flat:
                leg = ax.get_legend()
                if leg:
                    leg.remove()
            fig.legend(handles, labels, loc="lower center", ncol=3,
                       fontsize=8, frameon=True, bbox_to_anchor=(0.5, -0.06))

        fig.tight_layout(rect=[0, 0.06, 1, 1])
        suffix = "_log" if log_scale else ""
        out_path = output_dir / f"sm_split_{model}_bs{batch_size}{suffix}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path}")
        plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _get_available_values(csv_path: Path, col: str) -> list[int]:
    rows = _load_csv(csv_path)
    return sorted({int(r[col]) for r in rows})


def parse_args():
    parser = argparse.ArgumentParser(
        description="SM split latency trade-off plot (SSM vs Attn crossover)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model", choices=["zamba2", "falcon_h1"], default="zamba2",
    )
    parser.add_argument(
        "--seq-lens", nargs="+", type=int, default=None,
        help="Sequence lengths (default: all available in CSV)",
    )
    parser.add_argument(
        "--batch-sizes", nargs="+", type=int, default=None,
        help="Batch sizes (default: all available in CSV)",
    )
    parser.add_argument(
        "--total-sm", type=int, default=None,
        help="Override total SM count (default: max in CSV)",
    )
    parser.add_argument(
        "--results-dir", type=Path,
        default=Path(__file__).parent.parent / "results" / "stage1",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).parent.parent / "results" / "stage1",
    )
    parser.add_argument(
        "--per-batch", action="store_true",
        help="Save one figure per batch_size instead of a combined grid",
    )
    parser.add_argument(
        "--log-scale", action="store_true",
        help="Use log y-axis (useful when SSM and Attn latencies differ by orders of magnitude)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    results_dir = args.results_dir
    ssm_path = find_stage1_csv(results_dir, "ssm", args.model)

    seq_lens    = args.seq_lens    or _get_available_values(ssm_path, "seq_len")
    batch_sizes = args.batch_sizes or _get_available_values(ssm_path, "batch_size")

    print(f"Model      : {args.model}")
    print(f"seq_lens   : {seq_lens}")
    print(f"batch_sizes: {batch_sizes}")

    if args.per_batch:
        plot_sm_split_per_batch(
            model=args.model,
            seq_lens=seq_lens,
            batch_sizes=batch_sizes,
            results_dir=results_dir,
            output_dir=args.output_dir,
            total_sm=args.total_sm,
            log_scale=args.log_scale,
        )
    else:
        plot_sm_split(
            model=args.model,
            seq_lens=seq_lens,
            batch_sizes=batch_sizes,
            results_dir=results_dir,
            output_dir=args.output_dir,
            total_sm=args.total_sm,
            log_scale=args.log_scale,
        )
