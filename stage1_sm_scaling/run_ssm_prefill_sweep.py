"""
Stage 1: SSM prefill layer SM scaling curve measurement.

Sweeps SM count across 10%–100% of GPU capacity for each combination of
(model, seq_len, batch_size) and records latency + bandwidth utilization.

Usage:
    python stage1_sm_scaling/run_ssm_prefill_sweep.py --model zamba2 --device auto
    python stage1_sm_scaling/run_ssm_prefill_sweep.py --model falcon_h1 --device a100_40gb
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import csv
import json
from pathlib import Path
from itertools import product
from tqdm import tqdm

import torch
import yaml

from src.models.layer_runner import LayerRunner
from src.profiling.metrics import BandwidthEstimator


# ---------------------------------------------------------------------------
# Hardware config helpers
# ---------------------------------------------------------------------------

def load_hardware_config(device_key: str) -> dict:
    cfg_path = Path(__file__).parent.parent / "configs" / "hardware.yaml"
    with open(cfg_path) as f:
        hw = yaml.safe_load(f)

    if device_key == "auto" or device_key not in hw:
        props = torch.cuda.get_device_properties(0)
        n_sm = props.multi_processor_count
        steps = compute_sm_steps(n_sm)
        return {
            "name": torch.cuda.get_device_name(0),
            "sm_count": n_sm,
            "sm_sweep_steps": steps,
            "memory_bw_GBs": None,
        }

    cfg = hw[device_key]
    if cfg["sm_sweep_steps"] is None:
        cfg["sm_sweep_steps"] = compute_sm_steps(cfg["sm_count"])
    return cfg


def compute_sm_steps(total_sm: int, n_steps: int = 8) -> list[int]:
    """Compute n_steps SM counts from ~10% to 100% of total_sm."""
    steps = []
    for i in range(1, n_steps + 1):
        sm = max(1, round(total_sm * i / n_steps))
        if sm not in steps:
            steps.append(sm)
    return sorted(steps)


def device_tag(hw_cfg: dict) -> str:
    name = hw_cfg["name"].lower()
    name = name.replace("nvidia ", "").replace(" ", "_")
    return name


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_sweep(
    model_name: str,
    hw_cfg: dict,
    seq_lens: list[int],
    batch_sizes: list[int],
    n_warmup: int = 100,
    n_measure: int = 200,
    output_dir: Path = None,
    use_fallback: bool = False,
) -> list[dict]:
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "results" / "stage1"
    output_dir.mkdir(parents=True, exist_ok=True)

    sm_steps = hw_cfg["sm_sweep_steps"]
    total_sm = hw_cfg["sm_count"]
    bw_est = BandwidthEstimator()
    if hw_cfg.get("memory_bw_GBs"):
        bw_est.set_theoretical_bw(hw_cfg["memory_bw_GBs"])

    runner = LayerRunner(
        device="cuda",
        total_sm_count=total_sm,
    )

    results = []
    combos = list(product(sm_steps, seq_lens, batch_sizes))
    tag = device_tag(hw_cfg)

    print(f"\n=== SSM Prefill SM Sweep: {model_name} on {hw_cfg['name']} ===")
    print(f"  SM steps: {sm_steps}")
    print(f"  seq_lens: {seq_lens}")
    print(f"  batch_sizes: {batch_sizes}")
    print(f"  Total configs: {len(combos)}\n")

    for sm_count, seq_len, batch_size in tqdm(combos, desc="sweep"):
        try:
            row = runner.run_ssm_layer(
                model_name=model_name,
                batch_size=batch_size,
                seq_len=seq_len,
                sm_count=sm_count,
                n_warmup=n_warmup,
                n_measure=n_measure,
                use_fallback_kernel=use_fallback,
            )
            row["sm_ratio"] = sm_count / total_sm
            row["theoretical_bw_GBs"] = hw_cfg.get("memory_bw_GBs") or row.get("theoretical_bw_GBs", 0)
            row["bw_utilization"] = (
                row["achieved_bandwidth_GBs"] / row["theoretical_bw_GBs"]
                if row["theoretical_bw_GBs"] > 0 else float("nan")
            )
            results.append(row)
            tqdm.write(
                f"  sm={sm_count:3d} ({row['sm_ratio']:.0%})  "
                f"seq={seq_len:5d}  bs={batch_size}  "
                f"lat={row['latency_ms']:.3f}ms  "
                f"bw={row['achieved_bandwidth_GBs']:.1f}GB/s"
            )
        except Exception as e:
            tqdm.write(f"  ERROR sm={sm_count} seq={seq_len} bs={batch_size}: {e}")

    # Save CSV
    if results:
        out_csv = output_dir / f"ssm_scaling_{model_name}_{tag}.csv"
        fieldnames = [
            "sm_count", "sm_ratio", "seq_len", "batch_size",
            "latency_ms", "latency_p99_ms",
            "achieved_bandwidth_GBs", "theoretical_bw_GBs", "bw_utilization",
            "model_name", "layer_type",
        ]
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
        print(f"\nSaved: {out_csv}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="SSM prefill SM scaling sweep")
    parser.add_argument("--model", choices=["zamba2", "falcon_h1"], default="zamba2")
    parser.add_argument(
        "--device", default="auto",
        help="Hardware key from configs/hardware.yaml, or 'auto' for runtime detection"
    )
    parser.add_argument(
        "--seq-lens", nargs="+", type=int,
        default=[256, 512, 1024, 2048, 4096],
    )
    parser.add_argument(
        "--batch-sizes", nargs="+", type=int,
        default=[1, 4, 16],
    )
    parser.add_argument("--n-warmup", type=int, default=10)
    parser.add_argument("--n-measure", type=int, default=50)
    parser.add_argument(
        "--fallback", action="store_true",
        help="Use fallback kernel (no HuggingFace model download required)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required")

    hw_cfg = load_hardware_config(args.device)

    run_sweep(
        model_name=args.model,
        hw_cfg=hw_cfg,
        seq_lens=args.seq_lens,
        batch_sizes=args.batch_sizes,
        n_warmup=args.n_warmup,
        n_measure=args.n_measure,
        use_fallback=args.fallback,
    )
