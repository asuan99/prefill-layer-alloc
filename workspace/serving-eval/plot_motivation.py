"""
plot_motivation.py — counter-free motivation figures for hybrid serving (S1.2 revised).

Generates three figures that together constitute the motivation section for
SMController without requiring ncu, nsys, or CUPTI hardware counters:

  Fig A — Serving-level GPU utilisation (NVML device-level aggregate)
    x: time (seconds into serving run)
    y: device GPU-utilisation % (NVML sm_util_pct aggregate, ~10 Hz)
    Lines: hybrid model vs pure-Transformer baseline
    Evidence: "hybrid model achieves only X% average GPU-util under mixed
               prefill+decode workload"
    Caption: device-level aggregate (NVML), NOT per-layer.

  Fig B — Layer-type free-SM zone comparison (controlled sweep, counter-free)
    x: prefill seq_len
    y: free_sm_fraction (1 − saturation_sm / total_sm)
    Lines: SSM-chunked prefill, Attn prefill
    Evidence: SSM saturates at lower SM → larger free zone; Attn becomes
              compute-bound at long seq → smaller free zone → heterogeneity.
    Caption: from Stage 1 isolated-kernel sweep (CUDA events, no counters).

  Fig C — Throughput vs analytical roofline
    x: arithmetic intensity (FLOPs / byte, log scale)
    y: achieved throughput (TFLOPs/s, log scale)
    Lines: compute roof (flat), memory roof (slope = BW)
    Points: SSM prefill, Attn prefill at various seq_len (analytical AI,
            measured latency from Stage 1 CSV)
    Evidence: gap between roofline and actual shows the under-utilization.
    Caption: roofline bounds are analytical (peak spec), points are measured.

  Fig D (optional appendix) — nsys kernel timeline
    Generated only if kernel_timeline_*.csv exists from
    profile_layer_heterogeneity.py.  Not required for headline claims.

Usage:
    # Run after run_serving.py --monitor-nvml and plot_saturation.py:
    python workspace/serving-eval/plot_motivation.py \\
        --hybrid-model zamba2 --baseline-model llama3_8b \\
        --nvml-dir workspace/serving-eval/results \\
        --stage1-dir workspace/characterization/results/stage1 \\
        --output-dir workspace/serving-eval/results/motivation

    # Skip Fig A (no NVML data yet):
    python workspace/serving-eval/plot_motivation.py \\
        --stage1-dir workspace/characterization/results/stage1 \\
        --no-nvml

    # Generate all figures from synthetic demo data (test visualization):
    python workspace/serving-eval/plot_motivation.py --demo
"""

from __future__ import annotations

import sys
import os

_here      = os.path.dirname(os.path.abspath(__file__))
_workspace = os.path.dirname(_here)
sys.path.insert(0, _workspace)
sys.path.insert(0, _here)

import argparse
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Single saturation source (shared.sweep_spec — reachable because _workspace is on path).
from shared.sweep_spec import saturation_point as _spec_saturation_point


# ---------------------------------------------------------------------------
# Shared aesthetics
# ---------------------------------------------------------------------------

_HYBRID_COLOR    = "#2196F3"   # blue — hybrid model
_BASELINE_COLOR  = "#FF5722"   # orange — pure-Transformer baseline
_SSM_COLOR       = "#2196F3"   # blue — SSM layer type
_ATTN_COLOR      = "#FF5722"   # orange — Attn layer type
_ROOF_COMPUTE    = "#9C27B0"   # purple — compute roof
_ROOF_MEMORY     = "#4CAF50"   # green — memory roof

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "figure.dpi": 150,
})


# ---------------------------------------------------------------------------
# Saturation detection (duplicated from plot_saturation.py — no characterization import)
# ---------------------------------------------------------------------------

def _find_saturation_sm(grp: pd.DataFrame) -> int:
    """Saturation SM for one group — delegates to the single shared implementation.

    Fig B's own detection logic was removed (Phase 1): all saturation detection now
    goes through shared.sweep_spec.saturation_point() (threshold from sweep_spec.yaml).
    """
    sat = _spec_saturation_point(grp)
    return int(sat) if sat is not None else int(grp["sm_count"].max())


def _demo_path(path: Path) -> Path:
    """Insert a ``_DEMO`` tag before the suffix so synthetic figures never collide
    with real-data figures in the same directory (Phase 0 audit item 9)."""
    if path.stem.endswith("_DEMO"):
        return path
    return path.with_name(f"{path.stem}_DEMO{path.suffix}")


# ---------------------------------------------------------------------------
# Hardware config helper (no torch import — reads YAML only)
# ---------------------------------------------------------------------------

def _load_hw_config(hw_tag: str) -> dict:
    """Load hardware config from shared/configs/hardware.yaml by tag."""
    from shared.loaders import get_hardware_config
    return get_hardware_config(hw_tag)


def _load_model_config(model_name: str) -> dict:
    from shared.loaders import get_model_config
    return get_model_config(model_name)


# ---------------------------------------------------------------------------
# FLOPs / AI computation (analytical, counter-free)
# ---------------------------------------------------------------------------

def _ssm_flops_and_bytes(model_cfg: dict, seq_len: int, batch_size: int) -> tuple[float, float]:
    """Approximate FLOPs and bytes for one SSM layer prefill.

    Dominant cost: mamba_chunk_scan_combined (B/C/A contraction).
    AI (arithmetic intensity) = FLOPs / bytes.
    Memory-bound character: AI stays low, nearly constant in seq_len.
    """
    ssm    = model_cfg.get("ssm", {})
    hidden = model_cfg.get("hidden_size", 3584)
    n_heads   = ssm.get("n_heads",   112)
    head_dim  = ssm.get("head_dim",  64)
    d_state   = ssm.get("d_state",   64)
    expand    = ssm.get("expand",    2)

    inner_dim = hidden * expand  # expanded hidden

    # Projection in/out: 2× for both directions, 2 for MACs
    proj_flops  = 2 * 2 * batch_size * seq_len * hidden * inner_dim
    # SSM scan: batch × seq × n_heads × head_dim × d_state × 2 (complex mul)
    scan_flops  = 2 * batch_size * seq_len * n_heads * head_dim * d_state * 2
    total_flops = proj_flops + scan_flops

    # Bytes: read input activations (dominant for memory-bound op)
    input_bytes  = batch_size * seq_len * hidden * 2     # bf16
    weight_bytes = inner_dim * hidden * 2 * 2 + d_state * n_heads * head_dim * 2  # weight reads
    total_bytes  = input_bytes + weight_bytes

    return float(total_flops), float(total_bytes)


def _attn_flops_and_bytes(model_cfg: dict, seq_len: int, batch_size: int) -> tuple[float, float]:
    """Approximate FLOPs and bytes for one Attn layer prefill.

    AI grows linearly with seq_len (L²): Attn becomes compute-bound for long seq.
    """
    attn    = model_cfg.get("attention", {})
    n_heads = attn.get("num_heads",    32)
    head_dim= attn.get("head_dim",     128)
    hidden  = model_cfg.get("hidden_size", 3584)

    # Q,K,V projection
    qkv_flops = 2 * 3 * batch_size * seq_len * hidden * (n_heads * head_dim)
    # Attention: Q×K + softmax×V (both O(S²))
    attn_flops = 2 * 2 * batch_size * n_heads * seq_len * seq_len * head_dim
    # Output projection
    out_flops  = 2 * batch_size * seq_len * (n_heads * head_dim) * hidden
    total_flops = qkv_flops + attn_flops + out_flops

    # Bytes: Q,K,V weights + input activations
    qkv_weight_bytes = 3 * hidden * n_heads * head_dim * 2  # bf16
    input_bytes      = batch_size * seq_len * hidden * 2
    kv_bytes         = 2 * batch_size * seq_len * n_heads * head_dim * 2  # K+V cache
    total_bytes      = qkv_weight_bytes + input_bytes + kv_bytes

    return float(total_flops), float(total_bytes)


# ---------------------------------------------------------------------------
# Fig A — NVML serving-level GPU utilisation
# ---------------------------------------------------------------------------

def fig_a_nvml_utilisation(
    nvml_hybrid:   Optional[pd.DataFrame],
    nvml_baseline: Optional[pd.DataFrame],
    hybrid_label:  str,
    baseline_label: str,
    output_path: Path,
    trace_label: str = "ShareGPT",
) -> None:
    """NVML device-level GPU-util time-series during serving.

    Caption (written to {output_path.stem}_caption.txt):
      sm_util_pct = device-level aggregate (NVML, ~10 Hz polling).
      NOT per-layer.  Suitable for serving-level under-utilization evidence only.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax_util, ax_mem = axes

    def _plot_series(df: pd.DataFrame, ax: plt.Axes, col: str, color: str, label: str):
        if df is None or df.empty or col not in df.columns:
            ax.text(0.5, 0.5, "No data\n(run run_serving.py --monitor-nvml)",
                    ha="center", va="center", transform=ax.transAxes, color="grey")
            return
        t_s = df["timestamp_ms"].values / 1000.0
        v   = df[col].values
        ax.plot(t_s, v, color=color, linewidth=1.2, alpha=0.7, label=label)
        # Smoothed
        w = min(20, max(1, len(v) // 30))
        smooth = np.convolve(v, np.ones(w) / w, mode="same")
        ax.plot(t_s, smooth, color=color, linewidth=2.5, label=f"{label} (smoothed)")
        mean_v = np.mean(v)
        ax.axhline(mean_v, color=color, linestyle="--", linewidth=1.0, alpha=0.8,
                   label=f"mean = {mean_v:.0f}%")

    _plot_series(nvml_hybrid,   ax_util, "sm_util_pct", _HYBRID_COLOR,   hybrid_label)
    _plot_series(nvml_baseline, ax_util, "sm_util_pct", _BASELINE_COLOR, baseline_label)
    ax_util.set_xlabel("Elapsed time (s)")
    ax_util.set_ylabel("GPU util % (device-level, NVML aggregate)")
    ax_util.set_ylim(0, 110)
    ax_util.set_title(
        f"Fig A — Serving GPU Utilisation\n"
        f"Workload: {trace_label}  |  ⚠ NVML device-level aggregate (not per-layer)",
        fontsize=10, fontweight="bold",
    )
    ax_util.legend(fontsize=7, loc="upper right")
    ax_util.grid(alpha=0.25)

    _plot_series(nvml_hybrid,   ax_mem, "mem_util_pct", _HYBRID_COLOR,   hybrid_label)
    _plot_series(nvml_baseline, ax_mem, "mem_util_pct", _BASELINE_COLOR, baseline_label)
    ax_mem.set_xlabel("Elapsed time (s)")
    ax_mem.set_ylabel("Memory ctrl util % (NVML)")
    ax_mem.set_ylim(0, 110)
    ax_mem.set_title("Memory Controller Utilisation", fontsize=10)
    ax_mem.legend(fontsize=7, loc="upper right")
    ax_mem.grid(alpha=0.25)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Fig A: {output_path}")

    caption = (
        f"Fig A: Device-level GPU utilisation during vLLM serving "
        f"({trace_label} trace, concurrent requests).\n"
        f"Measurement: NVML nvmlDeviceGetUtilizationRates, device-level aggregate "
        f"(~10 Hz).  NOT per-layer SM utilisation.  Suitable for serving-level "
        f"under-utilization evidence; per-layer claims require hardware counters.\n"
        f"Lines: {hybrid_label} (hybrid SSM+Attn) vs {baseline_label} (pure-Transformer).\n"
        f"Evidence: hybrid model achieves lower average GPU-util for the same workload, "
        f"motivating SMController for layer-type heterogeneity exploitation."
    )
    (output_path.with_suffix(".txt")).write_text(caption)


# ---------------------------------------------------------------------------
# Fig B — Layer-type free-SM zone comparison
# ---------------------------------------------------------------------------

def _compute_free_sm_from_stage1(
    stage1_dir: Path,
    model_name: str,
    batch_size: int = 1,
    layer_types: tuple[str, ...] = ("ssm_chunked", "attn"),
) -> dict[str, pd.DataFrame]:
    """Load Stage 1 CSVs and compute free_sm_fraction per (layer_type, seq_len).

    Returns {layer_type: DataFrame(seq_len, saturation_sm, total_sm, free_sm, free_ratio)}.
    """
    results: dict[str, pd.DataFrame] = {}

    # Glob patterns for each layer type
    patterns = {
        "ssm_chunked": [stage1_dir / "chunked" / f"ssm_chunked_{model_name}_*.csv",
                        stage1_dir / f"ssm_chunked_{model_name}_*.csv"],
        "attn":        [stage1_dir / f"attn_scaling_{model_name}_*.csv"],
        "ssm_triton":  [stage1_dir / f"ssm_scaling_{model_name}_*.csv"],
    }

    for lt in layer_types:
        all_dfs = []
        for pat in patterns.get(lt, [stage1_dir / f"*{lt}*{model_name}*.csv"]):
            for f in Path(pat.parent).glob(pat.name):
                if "twopass" in f.name:
                    continue
                try:
                    df = pd.read_csv(f)
                    all_dfs.append(df)
                except Exception:
                    pass
        if not all_dfs:
            continue
        df = pd.concat(all_dfs, ignore_index=True)

        # Normalise column names
        if "model_name" not in df.columns and "model" in df.columns:
            df = df.rename(columns={"model": "model_name"})
        if "layer_type" not in df.columns:
            df["layer_type"] = lt
        if lt == "ssm_chunked":
            df["layer_type"] = "ssm_chunked"

        # Filter model + batch_size
        if "model_name" in df.columns:
            df = df[df["model_name"] == model_name]
        if "batch_size" in df.columns:
            df = df[df["batch_size"] == batch_size]
        if df.empty:
            continue

        # Ensure sm_ratio exists
        if "sm_ratio" not in df.columns and "sm_count" in df.columns:
            total_sm = df["sm_count"].max()
            df["sm_ratio"] = df["sm_count"] / total_sm

        total_sm = int(df["sm_count"].max())
        records  = []
        for sl, grp in df.groupby("seq_len"):
            sat_sm  = _find_saturation_sm(grp)
            free_sm = total_sm - sat_sm
            records.append({
                "seq_len":      int(sl),
                "saturation_sm": sat_sm,
                "total_sm":      total_sm,
                "free_sm":       free_sm,
                "free_ratio":    free_sm / total_sm,
            })
        if records:
            results[lt] = pd.DataFrame(records).sort_values("seq_len")

    return results


def fig_b_free_sm_comparison(
    stage1_dir: Path,
    model_name: str,
    hw_tag: str,
    output_path: Path,
    batch_size: int = 1,
    demo: bool = False,
) -> None:
    """Fig B: free_sm_fraction vs seq_len for SSM-chunked and Attn.

    Shows that SSM and Attn have different saturation SM counts → layer-type
    heterogeneity without any hardware counters.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))

    if demo:
        seq_lens = [256, 512, 1024, 2048, 4096, 8192]
        free_sm_ssm  = [0.72, 0.65, 0.58, 0.50, 0.44, 0.40]
        free_sm_attn = [0.55, 0.48, 0.35, 0.20, 0.08, 0.02]
        ax.plot(seq_lens, free_sm_ssm,  "o-", color=_SSM_COLOR,  linewidth=2, markersize=6,
                label="SSM chunked (ssm_chunked) — memory-BW bound")
        ax.plot(seq_lens, free_sm_attn, "s-", color=_ATTN_COLOR, linewidth=2, markersize=6,
                label="Attention — compute-bound at long seq")
        ax.fill_between(seq_lens, free_sm_ssm, free_sm_attn,
                        where=[s > a for s, a in zip(free_sm_ssm, free_sm_attn)],
                        alpha=0.15, color=_SSM_COLOR, label="SSM free-zone advantage")
        _note = " (DEMO — run Stage 1 sweep for real data)"
    else:
        layer_data = _compute_free_sm_from_stage1(stage1_dir, model_name, batch_size)
        if not layer_data:
            print(f"  No Stage 1 data found in {stage1_dir} for {model_name}")
            print("  Run run_chunked_ssm_sweep.py and run_attn_prefill_sweep.py first.")
            # Auto-fallback must NOT overwrite the real-data filename — force _DEMO.
            fig_b_free_sm_comparison(stage1_dir, model_name, hw_tag,
                                     _demo_path(output_path), batch_size, demo=True)
            return
        _note = ""
        colors = {"ssm_chunked": _SSM_COLOR, "attn": _ATTN_COLOR,
                  "ssm_triton": "#1565C0", "ssm": _SSM_COLOR}
        labels = {"ssm_chunked": "SSM chunked (primary, Triton)",
                  "attn":        "Attention",
                  "ssm_triton":  "SSM analytical (aux)"}
        for lt, df in layer_data.items():
            ax.plot(df["seq_len"], df["free_ratio"],
                    "o-", color=colors.get(lt, "grey"), linewidth=2, markersize=6,
                    label=labels.get(lt, lt))
        if "ssm_chunked" in layer_data and "attn" in layer_data:
            ssm_df  = layer_data["ssm_chunked"].set_index("seq_len")["free_ratio"]
            attn_df = layer_data["attn"].set_index("seq_len")["free_ratio"]
            common  = ssm_df.index.intersection(attn_df.index)
            if len(common) > 0:
                ax.fill_between(
                    common,
                    ssm_df[common], attn_df[common],
                    where=[ssm_df[s] > attn_df[s] for s in common],
                    alpha=0.12, color=_SSM_COLOR,
                    label="SSM free-zone advantage over Attn",
                )

    ax.set_xlabel("Prefill seq_len (tokens)")
    ax.set_ylabel("Free SM fraction  (1 − saturation_sm / total_sm)")
    ax.set_ylim(-0.02, 1.02)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.set_title(
        f"Fig B — Layer-type Free-SM Zone Comparison — {model_name}  (bs={batch_size}){_note}\n"
        f"Evidence: SSM memory-BW bound → large free zone; Attn compute-bound at long seq\n"
        f"Source: Stage 1 isolated-kernel sweep (CUDA events, no hardware counters)",
        fontsize=9, fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Fig B: {output_path}")

    caption = (
        f"Fig B: Free-SM zone fraction vs prefill seq_len for {model_name}.\n"
        f"free_sm_fraction = (total_SM − saturation_SM) / total_SM.\n"
        f"Saturation is defined as the SM count at which throughput improvement "
        f"drops below 3% per 10% SM increase.\n"
        f"Measurement: CUDA event timing from Stage 1 isolated-kernel sweep "
        f"(Green Context SM restriction, no hardware counters).\n"
        f"Evidence for layer-type heterogeneity (Problem 5): SSM layers are "
        f"memory-bandwidth bound and saturate at low SM count (large free zone), "
        f"while Attention layers become compute-bound at long seq_len (small free zone). "
        f"The free-zone gap between layer types justifies SMController."
    )
    (output_path.with_suffix(".txt")).write_text(caption)


# ---------------------------------------------------------------------------
# Fig C — Throughput vs roofline
# ---------------------------------------------------------------------------

def fig_c_roofline(
    stage1_dir: Path,
    model_name: str,
    hw_cfg: dict,
    model_cfg: dict,
    output_path: Path,
    seq_lens_show: tuple[int, ...] = (256, 512, 1024, 2048, 4096, 8192, 16384),
    batch_size: int = 1,
    demo: bool = False,
) -> None:
    """Fig C: achieved throughput vs analytical roofline.

    Both axes log-scale.  Roofline computed from peak spec (analytical).
    Points from Stage 1 latency CSVs (measured CUDA-event timing).

    Caption: "Roofline = analytical peak spec; points = CUDA-event latency"
    """
    peak_bw_GBs      = hw_cfg.get("memory_bw_GBs") or 2000      # A100-80GB default
    peak_fp16_tflops = hw_cfg.get("compute_fp16_tflops") or 312  # A100 default
    peak_bw_TBs      = float(peak_bw_GBs) / 1000.0  # TB/s

    # Roofline: min(compute, memory) at each AI
    ridge_AI = peak_fp16_tflops / peak_bw_TBs  # TFLOP/s / (TB/s) = FLOPs/byte
    ai_range = np.logspace(-1, 3, 500)
    roof     = np.minimum(peak_fp16_tflops, peak_bw_TBs * ai_range)

    fig, ax = plt.subplots(figsize=(7, 5))

    # Roof lines
    ax.loglog(ai_range, roof, "k-", linewidth=2, label="Roofline (analytical peak spec)")
    ax.loglog([ridge_AI, ridge_AI], [1e-2, peak_fp16_tflops], "k--",
              linewidth=1, alpha=0.5, label=f"Ridge point AI={ridge_AI:.0f} FLOPs/B")
    ax.axhline(peak_fp16_tflops, color=_ROOF_COMPUTE, linestyle=":", linewidth=1.2,
               label=f"Compute roof {peak_fp16_tflops} TFLOP/s (peak FP16)")
    ax.axvline(ridge_AI, color=_ROOF_MEMORY, linestyle=":", linewidth=1.0, alpha=0.5)

    # Load Stage 1 latency data (or use demo points)
    def _load_latency_csv(lt_glob: str) -> pd.DataFrame:
        dfs = []
        for p in sorted(Path(stage1_dir).glob(lt_glob)):
            if "twopass" in p.name:
                continue
            try:
                df = pd.read_csv(p)
                if "model_name" not in df.columns and "model" in df.columns:
                    df = df.rename(columns={"model": "model_name"})
                if "model_name" in df.columns:
                    df = df[df["model_name"] == model_name]
                if "batch_size" in df.columns:
                    df = df[df["batch_size"] == batch_size]
                dfs.append(df)
            except Exception:
                pass
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    # SSM chunked
    ssm_df = _load_latency_csv(f"chunked/ssm_chunked_{model_name}_*.csv")
    if ssm_df.empty:
        ssm_df = _load_latency_csv(f"ssm_chunked_{model_name}_*.csv")

    # Attn
    attn_df = _load_latency_csv(f"attn_scaling_{model_name}_*.csv")

    used_demo = demo or (ssm_df.empty and attn_df.empty)
    if used_demo:
        # Demo points: synthetic AI and throughput
        ssm_ai_demo  = [0.3, 0.3, 0.35, 0.4, 0.4, 0.45, 0.5]
        ssm_tfl_demo = [0.08, 0.12, 0.18, 0.22, 0.25, 0.28, 0.30]
        attn_ai_demo = [0.8, 2.0, 6.0, 20.0, 50.0, 100.0, 150.0]
        attn_tfl_demo= [0.5, 1.2, 3.0, 15.0, 40.0, 100.0, 180.0]
        seq_labels   = [str(s) for s in [256, 512, 1024, 2048, 4096, 8192, 16384]]
        ax.scatter(ssm_ai_demo, ssm_tfl_demo, color=_SSM_COLOR, marker="o",
                   s=60, zorder=5, label="SSM chunked (DEMO)")
        ax.scatter(attn_ai_demo, attn_tfl_demo, color=_ATTN_COLOR, marker="s",
                   s=60, zorder=5, label="Attention (DEMO)")
        for ai, tp, sl in zip(ssm_ai_demo, ssm_tfl_demo, seq_labels):
            ax.annotate(f"S={sl}", (ai, tp), textcoords="offset points",
                        xytext=(3, 3), fontsize=6, color=_SSM_COLOR)
        for ai, tp, sl in zip(attn_ai_demo, attn_tfl_demo, seq_labels):
            ax.annotate(f"S={sl}", (ai, tp), textcoords="offset points",
                        xytext=(3, 3), fontsize=6, color=_ATTN_COLOR)
        _note = " (DEMO — run Stage 1 sweep for real data)"
    else:
        _note = ""
        # SSM points: full-SM only (highest SM count = least restricted)
        for df, color, marker, lt_label, flop_fn in [
            (ssm_df,  _SSM_COLOR,  "o", "SSM chunked",  _ssm_flops_and_bytes),
            (attn_df, _ATTN_COLOR, "s", "Attention",    _attn_flops_and_bytes),
        ]:
            if df.empty:
                continue
            full_sm = df["sm_count"].max()
            df_full = df[df["sm_count"] == full_sm]
            plotted_sl = set()
            ai_vals, tp_vals, sl_vals = [], [], []
            for sl in sorted(df_full["seq_len"].unique()):
                if sl not in seq_lens_show:
                    continue
                row = df_full[df_full["seq_len"] == sl].iloc[0]
                lat_ms = row["latency_ms"]
                if lat_ms <= 0:
                    continue
                flops, bts = flop_fn(model_cfg, int(sl), batch_size)
                ai   = flops / bts if bts > 0 else 1.0
                tp_s = lat_ms / 1000.0
                tflops_achieved = flops / tp_s / 1e12
                ai_vals.append(ai)
                tp_vals.append(tflops_achieved)
                sl_vals.append(sl)
                plotted_sl.add(sl)
            if ai_vals:
                ax.scatter(ai_vals, tp_vals, color=color, marker=marker,
                           s=60, zorder=5, label=f"{lt_label} (measured, full SM)")
                for ai, tp, sl in zip(ai_vals, tp_vals, sl_vals):
                    ax.annotate(f"S={sl}", (ai, tp), textcoords="offset points",
                                xytext=(3, 3), fontsize=6, color=color)

    ax.set_xlabel("Arithmetic Intensity (FLOPs / byte, analytical)")
    ax.set_ylabel("Achieved Throughput (TFLOPs/s, measured)")
    ax.set_xlim(ai_range[0], ai_range[-1])
    ax.set_ylim(1e-2, peak_fp16_tflops * 2)
    ax.set_title(
        f"Fig C — Throughput vs Roofline — {model_name}{_note}\n"
        f"Roofline: analytical peak spec ({peak_fp16_tflops} TFLOP/s FP16, "
        f"{peak_bw_GBs} GB/s BW)\n"
        f"Points: CUDA-event latency from Stage 1 sweep (full SM, no counters)",
        fontsize=9, fontweight="bold",
    )
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(which="both", alpha=0.2)
    plt.tight_layout()
    # Synthetic (auto-fallback or --demo) figures get a _DEMO filename so they
    # never sit next to real-data figures (Phase 0 audit item 9).
    save_path = _demo_path(output_path) if used_demo else output_path
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    output_path = save_path
    plt.close(fig)
    print(f"  Fig C: {output_path}")

    caption = (
        f"Fig C: Throughput vs roofline for {model_name} layer types.\n"
        f"Roofline bounds: analytical peak specification "
        f"(compute = {peak_fp16_tflops} TFLOP/s FP16, "
        f"memory = {peak_bw_GBs} GB/s).  NOT measured hardware counters.\n"
        f"Arithmetic intensity: analytical FLOPs / bytes estimate "
        f"(SSM: projection + scan; Attn: Q×K + softmax×V).\n"
        f"Throughput: CUDA-event latency from Stage 1 isolated-kernel sweep "
        f"(Green Context, full SM, no counter access required).\n"
        f"SSM points cluster in the memory-bound region (low AI); "
        f"Attn points move rightward (higher AI) at long seq_len, "
        f"approaching the compute roof — evidence that SSM and Attn have "
        f"fundamentally different resource profiles and free different SM budgets."
    )
    (output_path.with_suffix(".txt")).write_text(caption)


# ---------------------------------------------------------------------------
# Fig D — Optional nsys appendix
# ---------------------------------------------------------------------------

def fig_d_nsys_appendix(
    profiling_dir: Path,
    model_name: str,
    seq_len: int,
    batch_size: int,
    output_path: Path,
) -> bool:
    """Optional appendix figure from nsys kernel timeline.

    Returns True if figure was generated, False if no data available.
    Only requires serving-eval/plot_layer_heterogeneity.py outputs.
    """
    tag     = f"{model_name}_sl{seq_len}_bs{batch_size}"
    tl_csv  = profiling_dir / f"kernel_timeline_{tag}.csv"
    ncu_csv = profiling_dir / f"ncu_{tag}.csv"

    if not tl_csv.exists():
        print(f"  Fig D: no kernel timeline found at {tl_csv} — skipping appendix figure.")
        return False

    try:
        # Re-use plot_layer_heterogeneity logic
        sys.path.insert(0, str(Path(_here)))
        from plot_layer_heterogeneity import (
            load_timeline, load_ncu_sm_util, make_figure, _SM_UTIL_PRIOR,
        )
        timeline = load_timeline(tl_csv)
        ncu_util = load_ncu_sm_util(ncu_csv) if ncu_csv.exists() else dict(_SM_UTIL_PRIOR)
        make_figure(
            hybrid_timeline=timeline,
            hybrid_ncu=ncu_util,
            hybrid_label=model_name,
            baseline_timeline=None,
            baseline_ncu={},
            baseline_label="",
            output_path=output_path,
            seq_len=seq_len,
            batch_size=batch_size,
        )
        print(f"  Fig D (appendix): {output_path}")
        return True
    except Exception as exc:
        print(f"  Fig D: failed to generate ({exc})")
        return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hybrid-model",   default="zamba2",
                   help="Hybrid model name (in models.yaml)")
    p.add_argument("--baseline-model", default="llama3_8b",
                   help="Pure-Transformer baseline model for NVML comparison")
    p.add_argument("--hw-tag", default="auto",
                   help="Hardware key in hardware.yaml (for roofline spec)")
    p.add_argument("--nvml-dir", type=Path, default=None,
                   help="Directory containing nvml_{model}_{trace}.csv files from run_serving.py")
    p.add_argument("--stage1-dir", type=Path,
                   default=Path(__file__).parent.parent / "characterization" / "results" / "stage1",
                   help="Stage 1 results dir (contains ssm_chunked_*, attn_scaling_*.csv)")
    p.add_argument("--profiling-dir", type=Path,
                   default=Path(__file__).parent / "results" / "profiling",
                   help="Dir with kernel_timeline_*.csv for optional Fig D appendix")
    p.add_argument("--output-dir", type=Path,
                   default=Path(__file__).parent / "results" / "motivation")
    p.add_argument("--trace", default="sharegpt",
                   choices=["sharegpt", "longbench", "smoke"],
                   help="Trace label used in nvml CSV filename and figure labels")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--seq-len-appendix", type=int, default=2048,
                   help="seq_len for Fig D appendix (must match profile_layer_heterogeneity.py run)")
    p.add_argument("--no-nvml",   action="store_true", help="Skip Fig A (no NVML data)")
    p.add_argument("--no-fig-c",  action="store_true", help="Skip Fig C roofline")
    p.add_argument("--no-fig-d",  action="store_true", help="Skip Fig D nsys appendix")
    p.add_argument("--demo",      action="store_true",
                   help="Use synthetic demo data (test visualization without real data)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" plot_motivation.py — counter-free motivation figures (S1.2)")
    print("=" * 70)
    print(f"  Hybrid model:   {args.hybrid_model}")
    print(f"  Baseline model: {args.baseline_model}")
    print(f"  Stage 1 dir:    {args.stage1_dir}")
    print(f"  Output dir:     {args.output_dir}")

    # Load hardware config (for roofline)
    try:
        hw_cfg = _load_hw_config(args.hw_tag)
    except Exception as exc:
        print(f"  Warning: cannot load hw_cfg ({exc}).  Using A100-80GB defaults.")
        hw_cfg = {"memory_bw_GBs": 2000, "compute_fp16_tflops": 312,
                  "sm_count": 108, "name": "NVIDIA A100 80GB", "device_tag": "a100_80gb"}

    try:
        model_cfg = _load_model_config(args.hybrid_model)
    except Exception as exc:
        print(f"  Warning: cannot load model_cfg ({exc}).  Using Zamba2 defaults.")
        model_cfg = {
            "hidden_size": 3584,
            "ssm": {"n_heads": 112, "head_dim": 64, "d_state": 64, "expand": 2},
            "attention": {"num_heads": 32, "head_dim": 224},
        }

    generated = []

    # --- Fig A: NVML serving utilisation ---
    if not args.no_nvml:
        nvml_hybrid = nvml_baseline = None
        if args.nvml_dir and args.nvml_dir.is_dir():
            def _try_load_nvml(model: str) -> Optional[pd.DataFrame]:
                for pat in (
                    f"nvml_{model}_{args.trace}.csv",
                    f"nvml_{model}_smoke.csv",
                    f"nvml_{model}_*.csv",
                ):
                    for f in sorted(args.nvml_dir.glob(pat)):
                        try:
                            return pd.read_csv(f)
                        except Exception:
                            pass
                return None
            nvml_hybrid   = _try_load_nvml(args.hybrid_model)
            nvml_baseline = _try_load_nvml(args.baseline_model)
        else:
            if not args.demo:
                print(f"  Fig A: --nvml-dir not set or does not exist — "
                      f"run run_serving.py --monitor-nvml first")
        out_a = args.output_dir / f"fig_a_nvml_{args.hybrid_model}.png"
        if args.demo:
            out_a = _demo_path(out_a)
        fig_a_nvml_utilisation(
            nvml_hybrid, nvml_baseline,
            hybrid_label=args.hybrid_model,
            baseline_label=args.baseline_model,
            output_path=out_a,
            trace_label=args.trace.title(),
        )
        generated.append(str(out_a))

    # --- Fig B: free-SM zone comparison ---
    out_b = args.output_dir / f"fig_b_free_sm_{args.hybrid_model}.png"
    if args.demo:
        out_b = _demo_path(out_b)
    fig_b_free_sm_comparison(
        stage1_dir=args.stage1_dir,
        model_name=args.hybrid_model,
        hw_tag=args.hw_tag,
        output_path=out_b,
        batch_size=args.batch_size,
        demo=args.demo,
    )
    generated.append(str(out_b))

    # --- Fig C: roofline ---
    if not args.no_fig_c:
        out_c = args.output_dir / f"fig_c_roofline_{args.hybrid_model}.png"
        if args.demo:
            out_c = _demo_path(out_c)
        fig_c_roofline(
            stage1_dir=args.stage1_dir,
            model_name=args.hybrid_model,
            hw_cfg=hw_cfg,
            model_cfg=model_cfg,
            output_path=out_c,
            batch_size=args.batch_size,
            demo=args.demo,
        )
        generated.append(str(out_c))

    # --- Fig D: optional nsys appendix ---
    if not args.no_fig_d and not args.demo:
        out_d = args.output_dir / f"fig_d_appendix_{args.hybrid_model}.png"
        ok = fig_d_nsys_appendix(
            profiling_dir=args.profiling_dir,
            model_name=args.hybrid_model,
            seq_len=args.seq_len_appendix,
            batch_size=args.batch_size,
            output_path=out_d,
        )
        if ok:
            generated.append(str(out_d))

    print(f"\n  Generated {len(generated)} figure(s):")
    for p in generated:
        print(f"    {p}")
    print(f"\n  Caption files (.txt) co-located with each figure.")
    print("\n  Motivation structure:")
    print("    Fig A — serving GPU-util (NVML device-level, no counters)")
    print("    Fig B — layer-type free-SM heterogeneity (Stage 1 sweep, no counters)")
    print("    Fig C — throughput vs analytical roofline (no counters)")
    print("    Fig D — (optional appendix) nsys kernel timeline")
    print("\n  Figs A/B/C together constitute the motivation section")
    print("  without requiring ncu, nsys, or CUPTI hardware counters.")


if __name__ == "__main__":
    main()
