"""
Stage 2: Baseline layer latency for overhead ratio computation.

Collects SSM and Attention prefill latency at full SM count for the
(model, seq_len, batch_size) combinations used in compute_decision_matrix.py.

This data is the denominator in:
  overhead_ratio = smctrl_overhead_us / (layer_latency_ms × 1000)

Usage:
    python stage2_overhead/measure_layer_latency.py --model zamba2
    python stage2_overhead/measure_layer_latency.py --model falcon_h1 --device auto
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import csv
from pathlib import Path
from itertools import product
from tqdm import tqdm

import torch

from src.models.layer_runner import LayerRunner
from stage1_sm_scaling.run_ssm_prefill_sweep import load_hardware_config, device_tag


# Measurement matrix for decision matrix denominator
SEQ_LENS = [512, 1024, 2048]
BATCH_SIZES = [1, 4]
LAYER_TYPES = ["ssm", "attn"]


def run_measurements(
    model_name: str,
    hw_cfg: dict,
    seq_lens: list[int],
    batch_sizes: list[int],
    n_warmup: int = 20,
    n_measure: int = 100,
    output_dir: Path = None,
) -> list[dict]:
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "results" / "stage2"
    output_dir.mkdir(parents=True, exist_ok=True)

    total_sm = hw_cfg["sm_count"]
    runner = LayerRunner(device="cuda", total_sm_count=total_sm)
    tag = device_tag(hw_cfg)

    results = []
    combos = list(product(LAYER_TYPES, seq_lens, batch_sizes))

    print(f"\n=== Layer Latency Baseline: {model_name} on {hw_cfg['name']} ===")
    print(f"  Full SM={total_sm}, n_warmup={n_warmup}, n_measure={n_measure}")

    for layer_type, seq_len, batch_size in tqdm(combos, desc="layer latency"):
        try:
            if layer_type == "ssm":
                row = runner.run_ssm_layer(
                    model_name=model_name,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    sm_count=total_sm,  # full SM — baseline
                    n_warmup=n_warmup,
                    n_measure=n_measure,
                )
            else:
                row = runner.run_attn_layer(
                    model_name=model_name,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    sm_count=total_sm,
                    n_warmup=n_warmup,
                    n_measure=n_measure,
                )
            results.append(row)
            tqdm.write(
                f"  {layer_type:5s}  seq={seq_len:5d}  bs={batch_size}  "
                f"lat={row['latency_ms']:.3f}ms"
            )
        except Exception as e:
            tqdm.write(f"  ERROR {layer_type} seq={seq_len} bs={batch_size}: {e}")

    if results:
        out_csv = output_dir / f"layer_latency_{model_name}_{tag}.csv"
        fieldnames = [
            "model_name", "layer_type", "seq_len", "batch_size",
            "sm_count", "sm_ratio",
            "latency_ms", "latency_p99_ms",
            "achieved_bandwidth_GBs", "theoretical_bw_GBs", "bw_utilization_pct",
        ]
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
        print(f"\nSaved: {out_csv}")

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Measure baseline layer latency for Stage 2")
    parser.add_argument("--model", choices=["zamba2", "falcon_h1"], required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seq-lens", nargs="+", type=int, default=SEQ_LENS)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=BATCH_SIZES)
    parser.add_argument("--n-warmup", type=int, default=20)
    parser.add_argument("--n-measure", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required")

    hw_cfg = load_hardware_config(args.device)

    run_measurements(
        model_name=args.model,
        hw_cfg=hw_cfg,
        seq_lens=args.seq_lens,
        batch_sizes=args.batch_sizes,
        n_warmup=args.n_warmup,
        n_measure=args.n_measure,
    )
