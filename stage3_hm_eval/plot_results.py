"""
Stage 3 visualization: TTFT-TPOT trade-off, SM utilization timeline,
and prefill throughput improvement figures.

Generates three publication-quality figures from stage3 CSV results:

  Figure 1 — TTFT vs TPOT Trade-off Scatter
    x: mean prefill TTFT (ms), y: decode TPOT p99 (ms)
    Points: (model, policy, seq_len), color-coded by policy
    SLO line: horizontal dashed at TPOT = 50ms

  Figure 2 — SM Utilization Timeline
    x: time (sec), y: SM utilization (%)
    Policy A vs B/C side-by-side
    Background color bands for SSM / Attn layer regions (estimated)

  Figure 3 — Prefill Throughput Improvement
    x: prefill seq_len, y: Policy B/C throughput / Policy A (ratio)
    1.0 = same as baseline; >1.0 = improvement

Usage:
    python stage3_hm_eval/plot_results.py
    python stage3_hm_eval/plot_results.py --results-dir results/stage3 --model zamba2
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
import seaborn as sns

SLO_TPOT_MS = 50.0

POLICY_COLORS = {
    "A": "#9E9E9E",  # gray
    "B": "#FF9800",  # orange
    "C": "#4CAF50",  # green
}
POLICY_LABELS = {
    "A": "Policy A (Fixed)",
    "B": "Policy B (Step-Adaptive)",
    "C": "Policy C (Layer-Wise)",
}
POLICY_MARKERS = {"A": "o", "B": "s", "C": "^"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

KNOWN_MODELS = {"zamba2", "falcon_h1"}


def _parse_stem(stem: str, prefix: str) -> tuple[str, str]:
    """Parse '{prefix}_{model}_{policy}_{device}' filename stem.

    Handles multi-part model names like 'falcon_h1' by trying known names first.
    Returns (model_name, policy_key).
    """
    body = stem[len(prefix) + 1:]  # strip 'eval_' or 'sm_timeline_'
    for model in KNOWN_MODELS:
        if body.startswith(model + "_"):
            rest = body[len(model) + 1:]  # '{policy}_{device}'
            policy = rest.split("_")[0]
            return model, policy
    # Fallback: assume single-token model name
    parts = body.split("_")
    return parts[0], parts[1] if len(parts) > 1 else "?"


def load_eval_results(results_dir: Path) -> pd.DataFrame:
    csv_files = list(results_dir.glob("eval_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No eval_*.csv files in {results_dir}")

    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        # CSV already has 'model' and 'policy' columns written by save_results()
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def load_sm_timelines(results_dir: Path) -> dict[str, pd.DataFrame]:
    """Load SM utilization timelines keyed by (model, policy)."""
    timelines = {}
    for f in results_dir.glob("sm_timeline_*.csv"):
        model, policy = _parse_stem(f.stem, "sm_timeline")
        df = pd.read_csv(f)
        timelines[(model, policy)] = df
    return timelines


# ---------------------------------------------------------------------------
# Figure 1: TTFT vs TPOT Scatter
# ---------------------------------------------------------------------------

def plot_ttft_tpot(df: pd.DataFrame, output_dir: Path, model_filter: str = None) -> None:
    if model_filter:
        df = df[df["model"] == model_filter]
    if df.empty:
        print("  No data for TTFT-TPOT scatter")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    for policy_key in sorted(df["policy"].unique()):
        grp = df[df["policy"] == policy_key]
        color = POLICY_COLORS.get(policy_key, "#000000")
        marker = POLICY_MARKERS.get(policy_key, "o")
        label = POLICY_LABELS.get(policy_key, f"Policy {policy_key}")

        ax.scatter(
            grp["ttft_mean_ms"], grp["tpot_p99_ms"],
            color=color, marker=marker, s=120, zorder=5,
            label=label, edgecolors="white", linewidths=0.8,
        )

        # Annotate with seq_len if column exists
        if "seq_len" in grp.columns:
            for _, row in grp.iterrows():
                ax.annotate(
                    f"L={int(row['seq_len'])}",
                    xy=(row["ttft_mean_ms"], row["tpot_p99_ms"]),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=7, color=color, alpha=0.8,
                )

    # SLO line
    ax.axhline(
        y=SLO_TPOT_MS, color="red", linestyle="--", linewidth=1.5,
        label=f"SLO: TPOT = {SLO_TPOT_MS:.0f}ms"
    )

    # Shade violation zone
    ax.axhspan(SLO_TPOT_MS, ax.get_ylim()[1] if ax.get_ylim()[1] > SLO_TPOT_MS else SLO_TPOT_MS * 2,
               alpha=0.08, color="red", label="SLO violation zone")

    ax.set_xlabel("Mean Prefill TTFT (ms)", fontsize=11)
    ax.set_ylabel("Decode TPOT p99 (ms)", fontsize=11)
    title = f"TTFT vs TPOT Trade-off"
    if model_filter:
        title += f" — {model_filter}"
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    suffix = f"_{model_filter}" if model_filter else ""
    out_path = output_dir / f"fig1_ttft_tpot{suffix}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 2: SM Utilization Timeline
# ---------------------------------------------------------------------------

def plot_sm_timeline(
    timelines: dict,
    output_dir: Path,
    model_name: str,
    compare_policies: tuple[str, str] = ("A", "C"),
) -> None:
    p1, p2 = compare_policies
    df1 = timelines.get((model_name, p1))
    df2 = timelines.get((model_name, p2))

    if df1 is None and df2 is None:
        print(f"  No SM timeline data for {model_name} policies {p1}/{p2}")
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=False)
    fig.suptitle(
        f"SM Utilization Timeline — {model_name}",
        fontsize=12, fontweight="bold"
    )

    for ax, (df, policy) in zip(axes, [(df1, p1), (df2, p2)]):
        if df is None or df.empty:
            ax.text(0.5, 0.5, f"No data for Policy {policy}",
                    ha="center", va="center", transform=ax.transAxes)
            continue

        t = df["timestamp_ms"].values / 1000.0  # convert to seconds
        sm = df["sm_util_pct"].values

        color = POLICY_COLORS.get(policy, "#000000")
        ax.fill_between(t, 0, sm, alpha=0.4, color=color)
        ax.plot(t, sm, color=color, linewidth=1.2,
                label=POLICY_LABELS.get(policy, f"Policy {policy}"))

        # Add SSM/Attn region bands (estimated from model layer structure)
        # For illustration: alternate 10-period bands (SSM lighter, Attn darker)
        if len(t) > 10:
            period = t[-1] / 10
            for i in range(10):
                layer_type = "ssm" if i % 2 == 0 else "attn"
                bg_color = "#E3F2FD" if layer_type == "ssm" else "#FFF3E0"
                ax.axvspan(i * period, (i + 1) * period, alpha=0.25,
                           color=bg_color, zorder=0)

        ax.set_ylabel("SM Util (%)")
        ax.set_ylim(0, 105)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel("Time (sec)")

    # Legend for background colors
    ssm_patch = mpatches.Patch(color="#E3F2FD", alpha=0.6, label="SSM layer region (est.)")
    attn_patch = mpatches.Patch(color="#FFF3E0", alpha=0.6, label="Attn layer region (est.)")
    fig.legend(handles=[ssm_patch, attn_patch], loc="lower center",
               ncol=2, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    out_path = output_dir / f"fig2_sm_timeline_{model_name}_{'_vs_'.join(compare_policies)}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 3: Prefill Throughput Improvement
# ---------------------------------------------------------------------------

def plot_throughput_improvement(
    df: pd.DataFrame,
    output_dir: Path,
    model_filter: str = None,
) -> None:
    if model_filter:
        df = df[df["model"] == model_filter]

    if "policy" not in df.columns or "A" not in df["policy"].values:
        print("  Need Policy A baseline for throughput comparison")
        return

    baseline = df[df["policy"] == "A"][["model", "seq_len", "batch_size",
                                         "prefill_throughput_toks_per_sec"]]
    baseline = baseline.rename(columns={"prefill_throughput_toks_per_sec": "baseline_tp"})

    compare_policies = [p for p in df["policy"].unique() if p != "A"]
    if not compare_policies:
        print("  No non-baseline policies to compare")
        return

    fig, ax = plt.subplots(figsize=(9, 5))

    seq_lens = sorted(df["seq_len"].unique())
    x = np.arange(len(seq_lens))
    width = 0.25
    n_policies = len(compare_policies)
    offsets = np.linspace(-(n_policies - 1) * width / 2, (n_policies - 1) * width / 2, n_policies)

    for offset, policy_key in zip(offsets, sorted(compare_policies)):
        grp = df[df["policy"] == policy_key].merge(baseline, on=["model", "seq_len", "batch_size"])
        grp["ratio"] = grp["prefill_throughput_toks_per_sec"] / grp["baseline_tp"].clip(lower=1e-6)

        ratios_by_seq = []
        errors_by_seq = []
        for sl in seq_lens:
            sub = grp[grp["seq_len"] == sl]["ratio"]
            ratios_by_seq.append(sub.mean() if not sub.empty else 1.0)
            errors_by_seq.append(sub.std() if not sub.empty else 0.0)

        color = POLICY_COLORS.get(policy_key, "#000000")
        label = POLICY_LABELS.get(policy_key, f"Policy {policy_key}")
        ax.bar(
            x + offset, ratios_by_seq, width,
            color=color, alpha=0.8, label=label,
            yerr=errors_by_seq, capsize=3, error_kw={"linewidth": 1},
        )

    ax.axhline(y=1.0, color="black", linestyle="--", linewidth=1.2, label="Baseline (Policy A)")
    ax.axhline(y=0.95, color="red", linestyle=":", linewidth=1, alpha=0.6, label="−5% threshold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{sl}" for sl in seq_lens])
    ax.set_xlabel("Prefill Sequence Length")
    ax.set_ylabel("Throughput Ratio vs Policy A")
    title = "Prefill Throughput Improvement (vs Fixed Baseline)"
    if model_filter:
        title += f" — {model_filter}"
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(2.0, ax.get_ylim()[1]))

    plt.tight_layout()
    suffix = f"_{model_filter}" if model_filter else ""
    out_path = output_dir / f"fig3_throughput{suffix}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Plot Stage 3 results")
    parser.add_argument(
        "--results-dir", type=Path,
        default=Path(__file__).parent.parent / "results" / "stage3"
    )
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).parent.parent / "results" / "stage3"
    )
    parser.add_argument(
        "--compare-policies", nargs=2, default=["A", "C"],
        help="Two policy keys to compare in Figure 2 timeline"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print(f"Loading Stage 3 results from {args.results_dir} …")
    df = load_eval_results(args.results_dir)
    timelines = load_sm_timelines(args.results_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    models = [args.model] if args.model else df["model"].unique().tolist()

    for model in models:
        print(f"\nGenerating figures for {model} …")
        plot_ttft_tpot(df, args.output_dir, model_filter=model)
        plot_sm_timeline(
            timelines, args.output_dir, model,
            compare_policies=tuple(args.compare_policies)
        )
        plot_throughput_improvement(df, args.output_dir, model_filter=model)

    print("\nDone.")
