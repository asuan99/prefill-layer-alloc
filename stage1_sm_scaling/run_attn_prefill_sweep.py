"""
Stage 1: Attention prefill layer SM scaling curve measurement.

Identical sweep structure to run_ssm_prefill_sweep.py but targets the
Attention layer. KV cache is pre-filled to simulate realistic context.

Usage:
    python stage1_sm_scaling/run_attn_prefill_sweep.py --model zamba2 --device auto
    python stage1_sm_scaling/run_attn_prefill_sweep.py --model falcon_h1 --context-len 512
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
from src.profiling.nvtx_markers import NVTXMarker
from stage1_sm_scaling.run_ssm_prefill_sweep import (
    load_hardware_config, device_tag
)


def run_sweep(
    model_name: str,
    hw_cfg: dict,
    seq_lens: list[int],
    batch_sizes: list[int],
    context_len: int = 0,
    n_warmup: int = 100,
    n_measure: int = 200,
    output_dir: Path = None,
    skip_verify: bool = False,
) -> list[dict]:
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "results" / "stage1"
    output_dir.mkdir(parents=True, exist_ok=True)

    sm_steps = hw_cfg["sm_sweep_steps"]
    total_sm = hw_cfg["sm_count"]
    theoretical_bw = hw_cfg.get("memory_bw_GBs")

    runner = LayerRunner(
        device="cuda",
        total_sm_count=total_sm,
        theoretical_bw_GBs=theoretical_bw,
    )

    if not skip_verify:
        ok = runner.verify_sm_control(verbose=True)
        if not ok:
            print(
                "WARNING: proceeding with sweep despite SM control failure.\n"
                "  sm_count column will be logged but latency will NOT vary with it.\n"
            )

    results = []
    combos = list(product(sm_steps, seq_lens, batch_sizes))
    tag = device_tag(hw_cfg)

    print(f"\n=== Attention Prefill SM Sweep: {model_name} on {hw_cfg['name']} ===")
    print(f"  SM steps     : {sm_steps}")
    print(f"  seq_lens     : {seq_lens}")
    print(f"  batch_sizes  : {batch_sizes}")
    print(f"  context_len  : {context_len}")
    print(f"  theoretical BW: {runner.bw_estimator.theoretical_bw_GBs:.1f} GB/s")
    print(f"  Total configs: {len(combos)}\n")

    for sm_count, seq_len, batch_size in tqdm(combos, desc="sweep"):
        try:
            with NVTXMarker.config_range("attn", sm_count, seq_len, batch_size, total_sm):
                row = runner.run_attn_layer(
                    model_name=model_name,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    sm_count=sm_count,
                    context_len=context_len,
                    n_warmup=n_warmup,
                    n_measure=n_measure,
                )
            results.append(row)
            tqdm.write(
                f"  sm={sm_count:3d} ({row['sm_ratio']:.0%})  "
                f"seq={seq_len:5d}  bs={batch_size}  "
                f"lat={row['latency_ms']:.3f}ms  "
                f"bw={row['achieved_bandwidth_GBs']:.1f}/{row['theoretical_bw_GBs']:.0f}GB/s "
                f"({row['bw_utilization_pct']:.1f}%)"
            )
        except Exception as e:
            tqdm.write(f"  ERROR sm={sm_count} seq={seq_len} bs={batch_size}: {e}")

    if results:
        out_csv = output_dir / f"attn_scaling_{model_name}_{tag}.csv"
        fieldnames = [
            "sm_count", "sm_ratio", "seq_len", "batch_size", "context_len",
            "latency_ms", "latency_p99_ms",
            "achieved_bandwidth_GBs", "theoretical_bw_GBs", "bw_utilization_pct",
            "model_name", "layer_type",
        ]
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
        print(f"\nSaved: {out_csv}")

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Attention prefill SM scaling sweep")
    parser.add_argument("--model", choices=["zamba2", "falcon_h1"], default="zamba2")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seq-lens", nargs="+", type=int, default=[256, 512, 1024, 2048, 4096])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4, 16])
    parser.add_argument("--context-len", type=int, default=0,
                        help="Pre-filled KV cache length (0 = empty context)")
    parser.add_argument("--n-warmup", type=int, default=100)
    parser.add_argument("--n-measure", type=int, default=200)
    parser.add_argument("--skip-verify", action="store_true")
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
        context_len=args.context_len,
        n_warmup=args.n_warmup,
        n_measure=args.n_measure,
    )
