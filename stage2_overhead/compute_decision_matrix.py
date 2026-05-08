"""
Stage 2: Strategy decision matrix computation.

Synthesizes Stage 1 (SM saturation) and Stage 2 (overhead) results
to determine which allocation strategy to execute in Stage 3.

Decision formula:
  overhead_ratio = ctx_switch_overhead_us / (layer_latency_ms * 1000)
  free_sm_fraction = (SM_total - SM_saturation) / SM_total
  potential_gain = free_sm_fraction × ssm_layer_fraction × decode_sm_sensitivity

Strategy thresholds:
  overhead_ratio < 0.05  → 'layer_wise'    (Policy C)
  overhead_ratio < 0.20  → 'step_adaptive' (Policy B)
  otherwise              → 'fixed'         (Policy A)

Outputs:
  results/stage2/decision_matrix.json
  results/stage2/decision_matrix.html

Usage:
    python stage2_overhead/compute_decision_matrix.py
    python stage2_overhead/compute_decision_matrix.py --decode-sm-sensitivity 0.8
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import json
import glob
from pathlib import Path

import pandas as pd
import yaml

# Strategy thresholds
THRESHOLD_LAYER_WISE = 0.05
THRESHOLD_STEP_ADAPTIVE = 0.20

# Decode SM sensitivity: how much TPOT improves per 10% additional SM
# Estimated from mixer-alloc or literature; 0.5 = moderate sensitivity
DEFAULT_DECODE_SM_SENSITIVITY = 0.5


def load_stage1_saturation(stage1_dir: Path) -> dict:
    """Load SSM saturation SM counts from Stage 1 CSVs.

    Returns: {(model_name, seq_len, batch_size): saturation_sm}
    """
    import numpy as np

    saturation = {}
    csv_files = list(stage1_dir.glob("ssm_scaling_*.csv"))
    if not csv_files:
        print(f"  WARNING: No SSM scaling CSVs in {stage1_dir}")
        return saturation

    for f in csv_files:
        df = pd.read_csv(f)
        if "normalized_throughput" not in df.columns:
            # Recompute throughput normalization
            df["throughput"] = 1.0 / df["latency_ms"]
            groups = ["model_name", "seq_len", "batch_size"]
            max_tp = df.groupby(groups)["throughput"].transform("max")
            df["normalized_throughput"] = df["throughput"] / max_tp

        for (model, sl, bs), grp in df.groupby(["model_name", "seq_len", "batch_size"]):
            grp = grp.sort_values("sm_ratio")
            sm_ratios = grp["sm_ratio"].values
            tps = grp["normalized_throughput"].values
            total_sm = grp["sm_count"].max()

            sat_sm = total_sm  # default: no saturation
            for i in range(1, len(sm_ratios)):
                delta_sm = sm_ratios[i] - sm_ratios[i - 1]
                delta_tp = tps[i] - tps[i - 1]
                if delta_sm <= 0:
                    continue
                gain_per_10pct = (delta_tp / delta_sm) * 0.10
                if gain_per_10pct < 0.03:
                    sat_sm = int(grp["sm_count"].iloc[i - 1])
                    break

            saturation[(model, sl, bs)] = {
                "saturation_sm": sat_sm,
                "total_sm": int(total_sm),
                "free_sm": int(total_sm) - sat_sm,
                "free_sm_fraction": (int(total_sm) - sat_sm) / int(total_sm),
            }

    return saturation


def load_stage2_overhead(stage2_dir: Path) -> dict:
    """Load SM-switch overhead from Stage 2 JSON files.

    Reads ctx_switch_overhead_*.json produced by measure_ctx_switch_latency.py.

    Returns: {'mean_us': ..., 'p99_us': ..., 'backend': ...}
    """
    json_files = sorted(stage2_dir.glob("ctx_switch_overhead_*.json"))
    if not json_files:
        print(f"  WARNING: No overhead JSON in {stage2_dir} "
              f"(checked ctx_switch_overhead_*.json)")
        return {
            "mean_us": 50.0,
            "p99_us": 120.0,
            "backend": "unknown",
            "device": "unknown",
        }

    latest = json_files[-1]
    with open(latest) as f:
        data = json.load(f)

    # Extract representative single-transition latency (ssm→attn with sync)
    transitions = data.get("single_transitions", {})
    key = "ssm→attn_sync_yes"
    if key in transitions:
        mean_us = transitions[key]["mean_us"]
        p99_us = transitions[key]["p99_us"]
    else:
        # Fallback: average across all sync=yes transitions, then any transitions
        sync_means = [v["mean_us"] for k, v in transitions.items()
                      if "sync_yes" in k and "mean_us" in v]
        sync_p99s  = [v["p99_us"]  for k, v in transitions.items()
                      if "sync_yes" in k and "p99_us"  in v]
        if sync_means:
            mean_us = sum(sync_means) / len(sync_means)
            p99_us  = max(sync_p99s)
        else:
            means = [v["mean_us"] for v in transitions.values() if "mean_us" in v]
            p99s  = [v["p99_us"]  for v in transitions.values() if "p99_us"  in v]
            mean_us = sum(means) / len(means) if means else 50.0
            p99_us  = max(p99s)               if p99s  else 120.0

    return {
        "mean_us": mean_us,
        "p99_us": p99_us,
        "backend": data.get("meta", {}).get("backend", "unknown"),
        "device": data.get("meta", {}).get("device", "unknown"),
        "source_file": str(latest.name),
        "raw": data,
    }


def load_layer_latencies(stage2_dir: Path) -> dict:
    """Load baseline layer latencies from Stage 2 CSVs.

    Returns: {(model_name, layer_type, seq_len, batch_size): latency_ms}
    """
    latencies = {}
    for f in stage2_dir.glob("layer_latency_*.csv"):
        df = pd.read_csv(f)
        for _, row in df.iterrows():
            key = (row["model_name"], row["layer_type"], int(row["seq_len"]), int(row["batch_size"]))
            latencies[key] = row["latency_ms"]
    return latencies


def load_model_configs() -> dict:
    cfg_path = Path(__file__).parent.parent / "configs" / "models.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def compute_strategy(overhead_ratio: float) -> str:
    if overhead_ratio < THRESHOLD_LAYER_WISE:
        return "layer_wise"
    elif overhead_ratio < THRESHOLD_STEP_ADAPTIVE:
        return "step_adaptive"
    else:
        return "fixed"


def build_decision_matrix(
    saturation: dict,
    overhead: dict,
    layer_latencies: dict,
    model_configs: dict,
    decode_sm_sensitivity: float,
) -> list[dict]:
    rows = []

    # Get unique models from saturation data
    models = set(k[0] for k in saturation.keys())
    if not models:
        models = {"zamba2", "falcon_h1"}

    for model_name in sorted(models):
        ssm_layer_fraction = (
            model_configs.get(model_name, {})
            .get("ssm", {})
            .get("ssm_layer_fraction", 0.75)
        )

        # Use SSM layer latency as representative denominator
        for (m, lt, sl, bs), lat_ms in sorted(layer_latencies.items()):
            if m != model_name or lt != "ssm":
                continue

            sat_key = (model_name, sl, bs)
            sat_data = saturation.get(sat_key, {})
            free_sm_fraction = sat_data.get("free_sm_fraction", 0.0)
            sat_sm = sat_data.get("saturation_sm", "N/A")
            total_sm = sat_data.get("total_sm", "N/A")

            overhead_ratio = overhead["mean_us"] / (lat_ms * 1000.0)
            overhead_ratio_p99 = overhead["p99_us"] / (lat_ms * 1000.0)

            potential_gain = (
                free_sm_fraction
                * ssm_layer_fraction
                * decode_sm_sensitivity
            )

            strategy = compute_strategy(overhead_ratio)

            rows.append({
                "model": model_name,
                "seq_len": sl,
                "batch_size": bs,
                "ssm_layer_latency_ms": round(lat_ms, 4),
                "ctx_switch_overhead_us": round(overhead["mean_us"], 2),
                "ctx_switch_overhead_p99_us": round(overhead["p99_us"], 2),
                "overhead_ratio": round(overhead_ratio, 4),
                "overhead_ratio_p99": round(overhead_ratio_p99, 4),
                "saturation_sm": sat_sm,
                "total_sm": total_sm,
                "free_sm_fraction": round(free_sm_fraction, 3),
                "ssm_layer_fraction": ssm_layer_fraction,
                "decode_sm_sensitivity": decode_sm_sensitivity,
                "potential_gain": round(potential_gain, 4),
                "strategy": strategy,
                "backend": overhead["backend"],
            })

    return rows


def render_html(rows: list[dict], out_path: Path) -> None:
    """Render decision matrix as styled HTML table."""
    df = pd.DataFrame(rows)

    def color_strategy(val):
        colors = {
            "layer_wise": "background-color: #c8e6c9",      # green
            "step_adaptive": "background-color: #fff9c4",   # yellow
            "fixed": "background-color: #ffccbc",            # orange
        }
        return colors.get(val, "")

    styled = df.style.map(color_strategy, subset=["strategy"])
    styled = styled.format({
        "overhead_ratio": "{:.2%}",
        "overhead_ratio_p99": "{:.2%}",
        "free_sm_fraction": "{:.1%}",
        "potential_gain": "{:.1%}",
    })

    html = f"""<!DOCTYPE html>
<html>
<head>
<title>Stage 2 Decision Matrix</title>
<style>
  body {{ font-family: monospace; font-size: 12px; padding: 20px; }}
  h1 {{ font-size: 16px; }}
  .legend {{ margin-bottom: 16px; }}
  .legend span {{ padding: 4px 8px; border-radius: 3px; margin-right: 8px; }}
  .lw {{ background: #c8e6c9; }} .sa {{ background: #fff9c4; }} .fx {{ background: #ffccbc; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #ccc; padding: 4px 8px; text-align: right; }}
  th {{ background: #f0f0f0; }}
</style>
</head>
<body>
<h1>Stage 2 Decision Matrix: SM Reconfiguration Strategy</h1>
<div class="legend">
  <span class="lw">layer_wise (overhead &lt; 5%)</span>
  <span class="sa">step_adaptive (5–20%)</span>
  <span class="fx">fixed (&gt; 20%)</span>
</div>
<p><b>overhead_ratio</b> = ctx_switch_overhead_μs / (ssm_layer_latency_ms × 1000)<br>
   <b>potential_gain</b> = free_sm_fraction × ssm_layer_fraction × decode_sm_sensitivity</p>
{styled.to_html()}
</body>
</html>
"""
    out_path.write_text(html)


def parse_args():
    parser = argparse.ArgumentParser(description="Compute Stage 2 decision matrix")
    parser.add_argument(
        "--stage1-dir", type=Path,
        default=Path(__file__).parent.parent / "results" / "stage1"
    )
    parser.add_argument(
        "--stage2-dir", type=Path,
        default=Path(__file__).parent.parent / "results" / "stage2"
    )
    parser.add_argument("--decode-sm-sensitivity", type=float,
                        default=DEFAULT_DECODE_SM_SENSITIVITY)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("Loading Stage 1 saturation data …")
    saturation = load_stage1_saturation(args.stage1_dir)
    print(f"  {len(saturation)} (model, seq_len, batch_size) entries")

    print("Loading Stage 2 overhead data …")
    overhead = load_stage2_overhead(args.stage2_dir)
    print(
        f"  smctrl overhead: mean={overhead['mean_us']:.1f}μs  "
        f"p99={overhead['p99_us']:.1f}μs  backend={overhead['backend']}"
    )

    print("Loading layer latency baselines …")
    layer_latencies = load_layer_latencies(args.stage2_dir)
    print(f"  {len(layer_latencies)} latency entries")

    model_configs = load_model_configs()

    print("\nComputing decision matrix …")
    rows = build_decision_matrix(
        saturation, overhead, layer_latencies, model_configs,
        decode_sm_sensitivity=args.decode_sm_sensitivity,
    )

    if not rows:
        print("WARNING: No data to build decision matrix. Run Stage 1 & 2 first.")
        exit(0)

    # Summary
    strategies = [r["strategy"] for r in rows]
    from collections import Counter
    counts = Counter(strategies)
    print(f"\nStrategy distribution: {dict(counts)}")
    # Dominant strategy (used by Stage 3 entrypoint)
    dominant = counts.most_common(1)[0][0]
    print(f"Dominant strategy: {dominant}")

    output_dir = args.stage2_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON output
    out_json = output_dir / "decision_matrix.json"
    with open(out_json, "w") as f:
        json.dump({"dominant_strategy": dominant, "rows": rows}, f, indent=2)
    print(f"\nSaved: {out_json}")

    # HTML output
    out_html = output_dir / "decision_matrix.html"
    render_html(rows, out_html)
    print(f"Saved: {out_html}")
