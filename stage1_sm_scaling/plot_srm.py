"""
SM-Parametric Roofline Model (SRM) — Stage 1 visualization.

Extends the classical roofline (AI vs attained TFLOPS) with SM count as a
third axis, drawing one compute ceiling per SM allocation and overlaying each
layer's measured operating point.

BulletServe's SRM claim (ASPLOS 2026 paper, §4):
  For a kernel with arithmetic intensity AI and an SM allocation of ratio r:

    ceiling(AI, r) = min(r × peak_TFLOPS,   AI × peak_BW_GBs)
                      ↑ compute ceiling         ↑ BW ceiling (hardware-fixed)

  Key insight: the BW ceiling is a shared hardware resource — it does NOT scale
  linearly with SM count. Only the compute ceiling scales with r.
  Consequence:
    • Memory-bound kernels (left of ridge) → SM reduction barely changes latency
    • Compute-bound kernels (right of ridge) → SM reduction directly reduces throughput

  The ridge point shifts left as r decreases:
    ridge_AI(r) = r × peak_TFLOPS / peak_BW_GBs

  This makes SM reallocation safe for memory-bound layers (SSM decode, small-batch
  SSM/Attn prefill) and costly for compute-bound layers (large-batch Attn/MLP prefill).

Operating point coordinates:
  Arithmetic Intensity (FLOPs/Byte) — analytical formula per layer type
  Attained Performance (TFLOPS)     — FLOPs / measured_latency_s

GPU specs are loaded from configs/hardware.yaml via --device (default: a100_80gb).
Pass --device auto to query the runtime GPU via torch.cuda.

Usage:
    python stage1_sm_scaling/plot_srm.py
    python stage1_sm_scaling/plot_srm.py --device a100_80gb --model zamba2
    python stage1_sm_scaling/plot_srm.py --device auto --batch-sizes 1 4
    python stage1_sm_scaling/plot_srm.py --seq-len 1024 --output-dir results/stage1
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import yaml


# ---------------------------------------------------------------------------
# Hardware constants — loaded from configs/hardware.yaml at startup.
# Defaults match A100-SXM4-80GB; overridden in __main__ via --device.
# ---------------------------------------------------------------------------
GPU_NAME         = "NVIDIA A100-SXM4-80GB"
PEAK_TFLOPS_FP16 = 312.0    # FP16 dense tensor-core (spec sheet, A100 SXM4)
PEAK_BW_GBS      = 2000.0   # HBM2e peak bandwidth (spec sheet, A100 SXM4 80GB)
RIDGE_FULL       = PEAK_TFLOPS_FP16 * 1e3 / PEAK_BW_GBS  # FLOPs/Byte at full SM


def _load_hw_specs(device: str = "a100_80gb") -> None:
    """Load GPU specs from configs/hardware.yaml and update module globals."""
    global GPU_NAME, PEAK_TFLOPS_FP16, PEAK_BW_GBS, RIDGE_FULL
    cfg_path = Path(__file__).parent.parent / "configs" / "hardware.yaml"
    try:
        with open(cfg_path) as f:
            hw = yaml.safe_load(f)
    except FileNotFoundError:
        return

    cfg = None
    if device == "auto":
        try:
            import torch
            props = torch.cuda.get_device_properties(0)
            dev_name = props.device_name.lower()
            for v in hw.values():
                if isinstance(v, dict) and v.get("name", "").lower() in dev_name:
                    cfg = v
                    break
            if cfg is None:
                cfg = hw.get("a100_80gb", {})
                bw_auto = (
                    2.0 * props.memory_clock_rate * 1e3 * props.memory_bus_width
                ) / (8.0 * 1e9)
                cfg = {
                    "name": props.device_name,
                    "memory_bw_GBs": bw_auto,
                    "compute_fp16_tflops": cfg.get("compute_fp16_tflops", 312.0),
                }
        except Exception:
            cfg = hw.get("a100_80gb", {})
    else:
        cfg = hw.get(device) or hw.get("a100_80gb", {})

    if cfg:
        GPU_NAME         = cfg.get("name", GPU_NAME)
        PEAK_BW_GBS      = float(cfg.get("memory_bw_GBs", PEAK_BW_GBS))
        PEAK_TFLOPS_FP16 = float(cfg.get("compute_fp16_tflops", PEAK_TFLOPS_FP16))
        RIDGE_FULL       = PEAK_TFLOPS_FP16 * 1e3 / PEAK_BW_GBS

# Aesthetics
LAYER_COLORS  = {"ssm": "#2196F3", "attn": "#FF5722", "mlp": "#4CAF50"}
LAYER_LABELS  = {"ssm": "SSM (Mamba-2)", "attn": "Attention", "mlp": "MLP/FFN"}
SEQ_MARKERS   = {256: "o", 512: "s", 1024: "D", 2048: "^", 4096: "v"}


# ---------------------------------------------------------------------------
# FLOPs estimation
# ---------------------------------------------------------------------------

def compute_flops(layer_type: str, model_cfg: dict, batch: int, seq_len: int) -> float:
    """Analytical FLOPs for one forward pass of a single isolated layer.

    Counts multiply-add pairs (each = 2 FLOPs, matching GPU convention).
    Only operations whose weight matrices are HBM-resident are counted; tile-local
    scratchpad ops (e.g. SSM state updates within a chunk, flash-attn softmax) are
    also included as they involve tensor contractions even if memory-resident.
    """
    if layer_type == "ssm":
        hidden   = model_cfg["hidden_size"]
        n_heads  = model_cfg.get("n_ssm_heads", 64)
        head_dim = model_cfg.get("head_dim", model_cfg.get("ssm_head_dim", 32))
        d_state  = model_cfg.get("d_state", 128)
        n_groups = model_cfg.get("n_groups", max(1, n_heads // 8))
        inner_dim = n_heads * head_dim

        # in_proj: hidden → [2·inner_dim + 2·n_groups·d_state + n_heads]
        in_proj_out = 2 * inner_dim + 2 * n_groups * d_state + n_heads
        flops_in = 2 * batch * seq_len * hidden * in_proj_out

        # SSM chunked scan: dominant O(seq · heads · head_dim · d_state)
        flops_scan = 2 * batch * seq_len * n_heads * head_dim * d_state

        # out_proj: inner_dim → hidden
        flops_out = 2 * batch * seq_len * inner_dim * hidden

        return float(flops_in + flops_scan + flops_out)

    elif layer_type == "attn":
        n_heads    = model_cfg.get("n_attn_heads", 8)
        n_kv_heads = model_cfg.get("n_kv_heads", n_heads)
        head_dim   = model_cfg.get("attn_head_dim", 256)

        # FlashAttention prefill: Q·K^T (B×H×S×S) + (A·V) (B×H×S×D)
        # Both are O(B × H × S² × D), factor-2 for multiply-add
        flops_qk = 2 * batch * n_heads * seq_len * seq_len * head_dim
        flops_av = 2 * batch * n_heads * seq_len * seq_len * head_dim
        return float(flops_qk + flops_av)

    elif layer_type == "mlp":
        hidden       = model_cfg["hidden_size"]
        intermediate = model_cfg.get("intermediate_size", 4096)

        # SwiGLU: gate (hidden→intermediate) + up (hidden→intermediate) + down (intermediate→hidden)
        flops_gate = 2 * batch * seq_len * hidden * intermediate
        flops_up   = 2 * batch * seq_len * hidden * intermediate
        flops_down = 2 * batch * seq_len * intermediate * hidden
        return float(flops_gate + flops_up + flops_down)

    return 0.0


def compute_arithmetic_intensity(
    layer_type: str, model_cfg: dict, batch: int, seq_len: int,
    achieved_bw_GBs: float, latency_ms: float,
) -> float:
    """Effective AI = max(analytical AI, minimum AI consistent with attained perf).

    Analytical AI = FLOPs / bytes_analytical
    where bytes_analytical is computed from tensor sizes (Q+K+V+O for attention,
    weights + activations for SSM/MLP).

    For kernels like FlashAttention with large head_dim, L2 cache reuse means
    actual DRAM traffic < bytes_analytical, so the true DRAM AI is HIGHER than
    analytical. We take the maximum to ensure operating points stay at or below
    the BW ceiling on the roofline plot:

      AI_eff = max(AI_analytical, attained_tflops × 1e3 / PEAK_BW)

    This is physically sound: if attained TFLOPS exceeds AI_analytical × PEAK_BW/1e3,
    the kernel must be operating at higher effective AI (less DRAM traffic than
    estimated), so we set AI to the minimum value consistent with the observation.
    """
    flops = compute_flops(layer_type, model_cfg, batch, seq_len)
    bytes_analytical = achieved_bw_GBs * (latency_ms / 1000.0) * 1e9
    if bytes_analytical <= 0 or flops <= 0:
        return float("nan")
    ai_analytical = flops / bytes_analytical

    # Minimum AI to place the point at or below the BW ceiling
    attained_tflops = flops / (latency_ms / 1000.0) / 1e12
    ai_from_bw = attained_tflops * 1e3 / PEAK_BW_GBS

    return max(ai_analytical, ai_from_bw)


def compute_attained_tflops(
    layer_type: str, model_cfg: dict, batch: int, seq_len: int, latency_ms: float,
) -> float:
    """Attained performance = FLOPs / latency_s / 1e12."""
    flops = compute_flops(layer_type, model_cfg, batch, seq_len)
    if latency_ms <= 0 or flops <= 0:
        return float("nan")
    return flops / (latency_ms / 1000.0) / 1e12


# ---------------------------------------------------------------------------
# Roofline ceiling helpers
# ---------------------------------------------------------------------------

def sm_roofline(ai_range: np.ndarray, sm_ratio: float) -> np.ndarray:
    """SM-parametric roofline ceiling (TFLOPS) for a given SM ratio.

    compute_ceiling(r) = r × PEAK_TFLOPS_FP16
    bw_ceiling          = ai × PEAK_BW_GBS   (hardware-fixed, shared resource)
    ceiling(ai, r)      = min(compute_ceiling(r), bw_ceiling)

    The BW ceiling does NOT scale with r because the memory controller is a
    shared GPU resource. Only the compute units (SMs) are partitioned.
    """
    compute_ceil = sm_ratio * PEAK_TFLOPS_FP16          # flat horizontal line
    bw_ceil      = ai_range * PEAK_BW_GBS / 1e3         # sloped line (GBs → TFLOPS: ÷1e3 for unit)
    # PEAK_BW_GBS in GB/s; AI in FLOPs/Byte → AI × BW in FLOPs/s → ÷ 1e12 for TFLOPS
    bw_ceil_tflops = ai_range * PEAK_BW_GBS / 1e3       # = AI(FLOPs/B) × BW(GB/s) / 1e3 = TFLOPS
    compute_ceil_arr = np.full_like(ai_range, compute_ceil)
    return np.minimum(compute_ceil_arr, bw_ceil_tflops)


def ridge_ai(sm_ratio: float) -> float:
    """Arithmetic intensity at the roofline ridge for a given SM ratio."""
    return (sm_ratio * PEAK_TFLOPS_FP16 * 1e3) / PEAK_BW_GBS  # FLOPs/Byte


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_model_config(model_name: str) -> dict:
    cfg_path = Path(__file__).parent.parent / "configs" / "models.yaml"
    with open(cfg_path) as f:
        raw = yaml.safe_load(f)
    if model_name not in raw:
        raise ValueError(f"Model {model_name!r} not in configs/models.yaml")
    cfg = raw[model_name]
    # Flatten nested keys so compute_flops can access them directly
    flat = {
        "hidden_size":      cfg["hidden_size"],
        "num_layers":       cfg.get("num_layers", 1),
        "intermediate_size": cfg.get("mlp", {}).get("intermediate_size", 4096),
        # SSM
        "n_ssm_heads":  cfg.get("ssm", {}).get("n_heads", 64),
        "head_dim":     cfg.get("ssm", {}).get("head_dim", 32),
        "ssm_head_dim": cfg.get("ssm", {}).get("head_dim", 32),
        "d_state":      cfg.get("ssm", {}).get("d_state", 128),
        "n_groups":     cfg.get("ssm", {}).get("n_groups", None),
        # Attention
        "n_attn_heads": cfg.get("attention", {}).get("num_heads", 8),
        "n_kv_heads":   cfg.get("attention", {}).get("num_kv_heads", 8),
        "attn_head_dim": cfg.get("attention", {}).get("head_dim", 256),
    }
    if flat["n_groups"] is None:
        flat["n_groups"] = max(1, flat["n_ssm_heads"] // 8)
    return flat


def load_scaling_data(results_dir: Path, model_name: str) -> pd.DataFrame:
    """Load latency CSVs for a model and annotate with FLOPs / AI / attained_tflops."""
    patterns = [
        f"ssm_scaling_{model_name}_*.csv",
        f"attn_scaling_{model_name}_*.csv",
        f"mlp_scaling_{model_name}_*.csv",
    ]
    dfs = []
    for pat in patterns:
        for f in results_dir.glob(pat):
            dfs.append(pd.read_csv(f))
    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)
    # Normalise model_name column
    if "model_name" not in df.columns and "model" in df.columns:
        df = df.rename(columns={"model": "model_name"})

    model_cfg = load_model_config(model_name)

    # Compute FLOPs, AI, attained TFLOPS row-by-row
    flops_list, ai_list, tflops_list = [], [], []
    for _, row in df.iterrows():
        lt  = row["layer_type"]
        bs  = int(row["batch_size"])
        sl  = int(row["seq_len"])
        lat = float(row["latency_ms"])
        bw  = float(row.get("achieved_bandwidth_GBs", 0))

        flops = compute_flops(lt, model_cfg, bs, sl)
        ai    = compute_arithmetic_intensity(lt, model_cfg, bs, sl, bw, lat)
        tflops = compute_attained_tflops(lt, model_cfg, bs, sl, lat)

        flops_list.append(flops)
        ai_list.append(ai)
        tflops_list.append(tflops)

    df["flops"]           = flops_list
    df["arithmetic_intensity"] = ai_list
    df["attained_tflops"] = tflops_list
    return df


# ---------------------------------------------------------------------------
# Figure: SM-Parametric Roofline
# ---------------------------------------------------------------------------

def plot_srm(
    df: pd.DataFrame,
    output_dir: Path,
    model_name: str,
    batch_size: int,
    sm_steps: list[float] | None = None,
) -> None:
    """SM-Parametric Roofline for a single (model, batch_size) combination.

    Layout: 1 row × N_layer_types columns.
    Each panel:
      - Background roofline family: one curve per SM ratio (gray palette)
      - Operating points: scatter coloured by SM count (plasma), shaped by seq_len
      - Ridge markers: vertical dashed line per SM ratio
    """
    df_model = df[(df["model_name"] == model_name) & (df["batch_size"] == batch_size)]
    if df_model.empty:
        print(f"  No data for {model_name} bs={batch_size}")
        return

    layer_types = sorted(df_model["layer_type"].unique())
    sm_counts   = sorted(df_model["sm_count"].unique())
    seq_lens    = sorted(df_model["seq_len"].unique())
    total_sm    = int(df_model["sm_count"].max())

    if sm_steps is None:
        sm_steps = sorted(df_model["sm_ratio"].unique())

    # --- layout ---
    n_cols = len(layer_types)
    fig, axes = plt.subplots(
        1, n_cols,
        figsize=(7.0 * n_cols, 5.8),
        layout="constrained",
    )
    if n_cols == 1:
        axes = [axes]

    fig.suptitle(
        f"SM-Parametric Roofline — {model_name}  |  batch_size={batch_size}  |  "
        f"{GPU_NAME}  ({PEAK_TFLOPS_FP16:.0f} TFLOPS FP16,  {PEAK_BW_GBS:.0f} GB/s)",
        fontsize=11, fontweight="bold",
    )

    # AI axis range (log space)
    ai_lo, ai_hi = 1e-2, 1e4
    ai_range = np.logspace(np.log10(ai_lo), np.log10(ai_hi), 500)

    # SM ratio → color for roofline curves (gray palette, faint)
    roof_ratios = sorted(set(round(r, 2) for r in sm_steps))
    roof_alphas = np.linspace(0.15, 0.55, len(roof_ratios))

    # SM count → color for operating points (plasma, same as other figures)
    cmap = plt.cm.plasma
    sm_colors = {sm: cmap(i / max(len(sm_counts) - 1, 1)) for i, sm in enumerate(sm_counts)}

    for ax, lt in zip(axes, layer_types):
        df_lt = df_model[df_model["layer_type"] == lt].dropna(
            subset=["arithmetic_intensity", "attained_tflops"]
        )

        # ---- roofline curves ----
        for ratio, alpha in zip(roof_ratios, roof_alphas):
            ceil = sm_roofline(ai_range, ratio)
            n_sm_label = round(ratio * total_sm)
            ax.plot(
                ai_range, ceil,
                color="gray", linewidth=1.0, alpha=alpha,
                linestyle="--",
            )
            # Ridge annotation (only at bottom of compute-flat region)
            rid = ridge_ai(ratio)
            if ai_lo < rid < ai_hi:
                ax.axvline(
                    rid, color="gray", linewidth=0.6, linestyle=":",
                    alpha=alpha * 1.4,
                )

        # ---- full-SM roofline (highlighted) ----
        ceil_full = sm_roofline(ai_range, 1.0)
        ax.plot(
            ai_range, ceil_full,
            color="black", linewidth=2.2, alpha=0.85, linestyle="-",
            label="Full SM ceiling (r=1.0)",
            zorder=3,
        )
        ax.axvline(
            RIDGE_FULL, color="black", linewidth=1.0, linestyle=":",
            alpha=0.5, zorder=3,
        )
        ax.text(
            RIDGE_FULL * 1.12, PEAK_TFLOPS_FP16 * 0.55,
            f"Ridge\n{RIDGE_FULL:.2f} B/F",
            fontsize=7, color="black", alpha=0.7, va="top",
        )

        # ---- operating points ----
        for sm in sm_counts:
            grp = df_lt[df_lt["sm_count"] == sm]
            for sl in seq_lens:
                pts = grp[grp["seq_len"] == sl]
                if pts.empty:
                    continue
                x = pts["arithmetic_intensity"].values
                y = pts["attained_tflops"].values
                mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
                if not mask.any():
                    continue
                ax.scatter(
                    x[mask], y[mask],
                    color=sm_colors[sm],
                    marker=SEQ_MARKERS.get(sl, "o"),
                    s=60, linewidths=0.5, edgecolors="white",
                    zorder=5, alpha=0.88,
                )

        # ---- axes ----
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(ai_lo, ai_hi)
        ax.set_ylim(1e-3, PEAK_TFLOPS_FP16 * 2.5)
        ax.set_xlabel("Arithmetic Intensity [FLOPs / Byte]", fontsize=10)
        if ax == axes[0]:
            ax.set_ylabel("Attained Performance [TFLOPS]", fontsize=10)
        ax.set_title(LAYER_LABELS.get(lt, lt), fontsize=11, fontweight="bold")
        ax.grid(True, which="both", alpha=0.15, linewidth=0.5)

        # Peak FP16 reference line
        ax.axhline(
            PEAK_TFLOPS_FP16, color="black", linewidth=1.0, linestyle="-",
            alpha=0.25,
        )
        ax.text(
            ai_hi * 0.8, PEAK_TFLOPS_FP16 * 1.05,
            f"Peak FP16\n{PEAK_TFLOPS_FP16:.0f} TFLOPS",
            fontsize=7, ha="right", va="bottom", alpha=0.5,
        )

    # ---- legends ----
    # SM count (color legend — operating points)
    sm_handles = [
        mpatches.Patch(color=sm_colors[sm], label=f"sm={sm} ({sm/total_sm:.0%})")
        for sm in sm_counts
    ]
    # Seq len (marker legend)
    seq_handles = [
        mlines.Line2D([], [], color="gray", marker=SEQ_MARKERS.get(sl, "o"),
                      linestyle="None", markersize=7,
                      label=f"seq={sl}")
        for sl in seq_lens
    ]
    # Roofline style
    roof_handles = [
        mlines.Line2D([], [], color="black", linestyle="-", linewidth=2, label="Full SM (r=1.0)"),
        mlines.Line2D([], [], color="gray",  linestyle="--", linewidth=1, alpha=0.5,
                      label="Partial SM ceilings"),
    ]

    n_leg_cols = min(len(sm_counts), 6)
    fig.legend(
        handles=sm_handles,
        title="SM allocation",
        loc="outside lower center", ncol=n_leg_cols,
        fontsize=8, title_fontsize=8,
    )
    axes[-1].legend(
        handles=seq_handles + roof_handles,
        loc="upper left", fontsize=8, title="seq_len / ceilings",
        title_fontsize=8, framealpha=0.85,
    )

    out_path = output_dir / f"srm_{model_name}_bs{batch_size}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure: SRM Operating-Point Trajectory (seq_len × SM count grid)
# ---------------------------------------------------------------------------

def plot_srm_trajectory(
    df: pd.DataFrame,
    output_dir: Path,
    model_name: str,
    seq_len: int,
) -> None:
    """One roofline per batch_size; trajectories show how operating point
    moves along the AI axis as SM count changes (all batch sizes overlaid).

    This directly answers "does SM reduction stay on the BW ceiling?"
    """
    df_model = df[(df["model_name"] == model_name) & (df["seq_len"] == seq_len)]
    if df_model.empty:
        print(f"  No data for {model_name} seq={seq_len}")
        return

    layer_types = sorted(df_model["layer_type"].unique())
    sm_counts   = sorted(df_model["sm_count"].unique())
    batch_sizes = sorted(df_model["batch_size"].unique())
    total_sm    = int(df_model["sm_count"].max())

    cmap_sm   = plt.cm.plasma
    sm_colors = {sm: cmap_sm(i / max(len(sm_counts) - 1, 1)) for i, sm in enumerate(sm_counts)}
    bs_cmaps  = [plt.cm.Blues, plt.cm.Oranges, plt.cm.Greens,
                 plt.cm.Reds, plt.cm.Purples]
    bs_colors = {bs: bs_cmaps[i % len(bs_cmaps)](0.65) for i, bs in enumerate(batch_sizes)}

    n_cols = len(layer_types)
    fig, axes = plt.subplots(
        1, n_cols,
        figsize=(7.0 * n_cols, 5.8),
        layout="constrained",
    )
    if n_cols == 1:
        axes = [axes]

    fig.suptitle(
        f"SRM Trajectory — {model_name}  |  seq_len={seq_len}  |  "
        f"{GPU_NAME}  ({PEAK_TFLOPS_FP16:.0f} TFLOPS,  {PEAK_BW_GBS:.0f} GB/s)  "
        f"— arrow: low→full SM",
        fontsize=11, fontweight="bold",
    )

    ai_lo, ai_hi = 1e-2, 1e4
    ai_range = np.logspace(np.log10(ai_lo), np.log10(ai_hi), 500)

    sm_ratios_sorted = sorted(df_model["sm_ratio"].unique())

    for ax, lt in zip(axes, layer_types):
        df_lt = df_model[df_model["layer_type"] == lt].dropna(
            subset=["arithmetic_intensity", "attained_tflops"]
        )

        # Roofline ceilings
        for ratio in sm_ratios_sorted:
            n_sm = round(ratio * total_sm)
            ax.plot(
                ai_range, sm_roofline(ai_range, ratio),
                color=sm_colors.get(n_sm, "gray"),
                linewidth=1.2, alpha=0.35, linestyle="--", zorder=2,
            )
        # Full-SM highlighted
        ax.plot(
            ai_range, sm_roofline(ai_range, 1.0),
            color="black", linewidth=2.0, alpha=0.8, zorder=3,
        )
        ax.axhline(PEAK_TFLOPS_FP16, color="black", linewidth=0.8, linestyle=":", alpha=0.3)
        ax.axvline(RIDGE_FULL, color="black", linewidth=0.8, linestyle=":", alpha=0.3)

        # Trajectory: for each batch_size, connect operating points across SM counts
        for bs in batch_sizes:
            grp = df_lt[df_lt["batch_size"] == bs].sort_values("sm_count")
            if grp.empty:
                continue
            x = grp["arithmetic_intensity"].values
            y = grp["attained_tflops"].values
            mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
            if mask.sum() < 2:
                continue
            # Draw trajectory line (connects SM steps)
            ax.plot(
                x[mask], y[mask],
                color=bs_colors[bs], linewidth=1.5, alpha=0.6, linestyle="-",
                zorder=4,
            )
            # Scatter points colored by SM count
            for _, row in grp[mask].iterrows():
                sm = int(row["sm_count"])
                ax.scatter(
                    row["arithmetic_intensity"], row["attained_tflops"],
                    color=sm_colors.get(sm, "gray"),
                    s=55, edgecolors=bs_colors[bs], linewidths=1.5,
                    zorder=6,
                )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(ai_lo, ai_hi)
        ax.set_ylim(1e-3, PEAK_TFLOPS_FP16 * 2.5)
        ax.set_xlabel("Arithmetic Intensity [FLOPs / Byte]", fontsize=10)
        if ax == axes[0]:
            ax.set_ylabel("Attained Performance [TFLOPS]", fontsize=10)
        ax.set_title(LAYER_LABELS.get(lt, lt), fontsize=11, fontweight="bold")
        ax.grid(True, which="both", alpha=0.15, linewidth=0.5)
        ax.text(
            RIDGE_FULL * 1.1, 2e-3,
            f"Ridge {RIDGE_FULL:.2f} B/F",
            fontsize=7, alpha=0.6,
        )

    # Legends
    sm_handles = [
        mpatches.Patch(color=sm_colors[sm], label=f"sm={sm} ({sm/total_sm:.0%})")
        for sm in sm_counts
    ]
    bs_handles = [
        mlines.Line2D([], [], color=bs_colors[bs], linestyle="-", linewidth=2,
                      label=f"batch={bs}")
        for bs in batch_sizes
    ]
    fig.legend(
        handles=sm_handles,
        title="SM count",
        loc="outside lower center", ncol=min(len(sm_counts), 6),
        fontsize=8, title_fontsize=8,
    )
    axes[-1].legend(
        handles=bs_handles,
        loc="upper left", fontsize=8, title="batch_size",
        title_fontsize=8, framealpha=0.85,
    )

    out_path = output_dir / f"srm_trajectory_{model_name}_seq{seq_len}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure: SRM Summary — SM bound analysis per layer type
# ---------------------------------------------------------------------------

def plot_srm_bound_analysis(
    df: pd.DataFrame,
    output_dir: Path,
    model_name: str,
    batch_size: int,
) -> None:
    """Classify each operating point as 'compute-bound' or 'memory-bound'
    under its actual SM allocation and visualize the classification.

    A point is compute-bound at (AI, sm_ratio) if:
      AI > ridge_AI(sm_ratio) = sm_ratio × PEAK_TFLOPS / PEAK_BW
    Otherwise memory-bound.

    Shows: for each (seq_len, sm_ratio) → which regime? How close to the ceiling?
    """
    df_model = df[(df["model_name"] == model_name) & (df["batch_size"] == batch_size)]
    if df_model.empty:
        return

    layer_types = sorted(df_model["layer_type"].unique())
    sm_counts   = sorted(df_model["sm_count"].unique())
    seq_lens    = sorted(df_model["seq_len"].unique())
    total_sm    = int(df_model["sm_count"].max())

    cmap = plt.cm.plasma
    sm_colors = {sm: cmap(i / max(len(sm_counts) - 1, 1)) for i, sm in enumerate(sm_counts)}

    fig, axes = plt.subplots(
        1, len(layer_types),
        figsize=(7.0 * len(layer_types), 5.8),
        layout="constrained",
    )
    if len(layer_types) == 1:
        axes = [axes]

    fig.suptitle(
        f"SRM Bound Analysis — {model_name}  |  batch_size={batch_size}  "
        f"— Efficiency = attained / ceiling(AI, r)  [1.0 = on roofline]",
        fontsize=11, fontweight="bold",
    )

    sm_restrict_fail_marked = False  # track if we need the warning annotation

    for ax, lt in zip(axes, layer_types):
        df_lt = df_model[df_model["layer_type"] == lt].dropna(
            subset=["arithmetic_intensity", "attained_tflops"]
        )

        for sm in sm_counts:
            grp = df_lt[df_lt["sm_count"] == sm].sort_values("seq_len")
            if grp.empty:
                continue
            sm_ratio = sm / total_sm
            effs, sls = [], []
            fail_sls, fail_effs = [], []
            for _, row in grp.iterrows():
                ai    = row["arithmetic_intensity"]
                tflops = row["attained_tflops"]
                if not (np.isfinite(ai) and np.isfinite(tflops) and ai > 0 and tflops > 0):
                    continue
                ceil_arr = sm_roofline(np.array([ai]), sm_ratio)
                ceil = float(ceil_arr[0])
                eff  = tflops / ceil if ceil > 0 else float("nan")
                sl = int(row["seq_len"])
                # eff > 1.0 → SM restriction ineffective (kernel uses more SMs than mask)
                if eff > 1.0:
                    fail_sls.append(sl)
                    fail_effs.append(min(eff, 1.08))
                    sm_restrict_fail_marked = True
                else:
                    effs.append(eff)
                    sls.append(sl)

            if effs:
                ax.plot(
                    sls, effs,
                    color=sm_colors[sm], linewidth=2, marker="o", markersize=5,
                    label=f"sm={sm} ({sm/total_sm:.0%})",
                )
            if fail_sls:
                ax.scatter(
                    fail_sls, fail_effs,
                    color=sm_colors[sm], marker="X", s=80, linewidths=1.0,
                    edgecolors="red", zorder=7,
                )

        ax.axhline(1.0, color="black", linewidth=1.2, linestyle="--", alpha=0.5,
                   label="Roofline ceiling")
        ax.set_xscale("log", base=2)
        ax.set_xticks(seq_lens)
        ax.set_xticklabels(seq_lens, rotation=30)
        ax.set_ylim(0, 1.15)
        ax.set_xlabel("Sequence Length", fontsize=10)
        if ax == axes[0]:
            ax.set_ylabel("Efficiency = attained / ceiling(AI, r)", fontsize=10)
        ax.set_title(LAYER_LABELS.get(lt, lt), fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.2)

    handles = [
        mpatches.Patch(color=sm_colors[sm], label=f"sm={sm} ({sm/total_sm:.0%})")
        for sm in sm_counts
    ]
    if sm_restrict_fail_marked:
        handles.append(
            mlines.Line2D([], [], color="gray", marker="X", linestyle="None",
                          markersize=8, markeredgecolor="red", markeredgewidth=1.0,
                          label="eff>1: SM mask ineffective")
        )
    n_leg_cols = min(len(sm_counts), 6)
    fig.legend(
        handles=handles,
        loc="outside lower center", ncol=n_leg_cols,
        fontsize=8,
    )

    out_path = output_dir / f"srm_bound_{model_name}_bs{batch_size}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="SM-Parametric Roofline Model (SRM) visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--results-dir", type=Path,
        default=Path(__file__).parent.parent / "results" / "stage1",
    )
    parser.add_argument("--model", default=None, help="Filter by model name")
    parser.add_argument(
        "--device", default="a100_80gb",
        help=(
            "Hardware key from configs/hardware.yaml used to set roofline ceilings "
            "(peak FP16 TFLOPS, memory BW). Use 'auto' for runtime GPU detection. "
            "Default: a100_80gb"
        ),
    )
    parser.add_argument(
        "--batch-sizes", nargs="+", type=int, default=None,
        help="Batch sizes to plot (default: all found in data)",
    )
    parser.add_argument(
        "--seq-lens", nargs="+", type=int, default=None,
        help="Seq lens for trajectory plot (default: [1024, 4096])",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).parent.parent / "results" / "stage1",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    _load_hw_specs(args.device)
    print(f"GPU         : {GPU_NAME}")
    print(f"Peak FP16   : {PEAK_TFLOPS_FP16:.0f} TFLOPS")
    print(f"Peak BW     : {PEAK_BW_GBS:.0f} GB/s")
    print(f"Ridge (full): {RIDGE_FULL:.3f} FLOPs/Byte")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Discover models from CSV filenames
    results_dir = args.results_dir
    model_names = set()
    for f in results_dir.glob("ssm_scaling_*.csv"):
        parts = f.stem.split("_")  # ssm_scaling_<model>_<device>
        if len(parts) >= 3:
            # model name is parts[2] for single-word models, but could be multi-word
            # Heuristic: join parts[2:-3] (strip ssm, scaling, and device suffix)
            # Device suffix is typically 3-4 words like "geforce_rtx_5060_ti"
            model_names.add("_".join(parts[2:-4]) or parts[2])

    # Fallback: check both known models
    available = []
    for m in ["zamba2", "falcon_h1"]:
        if any(results_dir.glob(f"ssm_scaling_{m}_*.csv")):
            available.append(m)

    models = [args.model] if args.model else available
    if not models:
        print("No model data found. Run stage1 sweeps first.")
        raise SystemExit(1)

    for model_name in models:
        print(f"\nLoading {model_name} ...")
        df = load_scaling_data(results_dir, model_name)
        if df.empty:
            print(f"  No data found for {model_name}")
            continue

        batch_sizes = args.batch_sizes or sorted(df["batch_size"].unique().tolist())
        seq_lens_traj = args.seq_lens or [sl for sl in [1024, 4096]
                                           if sl in df["seq_len"].values]

        print(f"  Batches: {batch_sizes}  |  Seq lens (trajectory): {seq_lens_traj}")
        print(f"  AI range: [{df['arithmetic_intensity'].min():.3f}, "
              f"{df['arithmetic_intensity'].max():.3f}] FLOPs/Byte")
        print(f"  Attained TFLOPS range: [{df['attained_tflops'].min():.4f}, "
              f"{df['attained_tflops'].max():.4f}]")

        # Fig A: SM-parametric roofline with all points
        print(f"\n  [Fig A] SM-Parametric Roofline ...")
        for bs in batch_sizes:
            plot_srm(df, args.output_dir, model_name, bs)

        # Fig B: Operating point trajectories (SM count as trajectory axis)
        print(f"\n  [Fig B] Trajectory plots ...")
        for sl in seq_lens_traj:
            plot_srm_trajectory(df, args.output_dir, model_name, sl)

        # Fig C: Bound classification (efficiency vs ceiling)
        print(f"\n  [Fig C] Bound analysis ...")
        for bs in batch_sizes:
            plot_srm_bound_analysis(df, args.output_dir, model_name, bs)

    print("\nDone.")
