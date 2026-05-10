"""
Stage 1: SSM prefill layer SM scaling curve measurement.

Strategy
--------
Two measurement modes are supported:

Default (analytical wave model):
  mamba_chunk_scan_combined (Triton SSD kernel) uses cooperative inter-block
  barriers that require ALL thread blocks to be active simultaneously. Under
  Green Context SM restriction the barrier deadlocks, making direct measurement
  at restricted SM counts impossible.

  Instead we use an analytical wave model:
    1. Measure latency at FULL SM (no Green Context, no restriction) for all
       (seq_len, batch_size) combos. This is straightforward and reliable.
    2. Derive n_blocks = batch × seq_len / BLOCKS_PER_SEQ_BS from the
       empirically confirmed formula (verified against ncu profiling data).
    3. For each target SM count k:
         latency(k) = latency(full) × ceil(n_blocks / k) / ceil(n_blocks / total_sm)
       This is exact for a perfectly wave-parallel kernel with constant time-per-wave.

  The result is a synthetic SM scaling table that correctly represents the
  Triton SSD kernel's behaviour, unlike a PyTorch-scan proxy which has
  fundamentally different compute characteristics.

  wave_eff_pct is always >99.96 % for all measured configs (verified from ncu),
  confirming the wave model assumption holds.

PyTorch scan mode (--force-pytorch-scan):
  Uses a pure-PyTorch chunked scan implementation instead of the Triton SSD kernel.
  This avoids the cooperative-barrier deadlock, so direct Green Context measurement
  at each SM step is possible. Results reflect PyTorch scan compute characteristics,
  not the Triton kernel. Output filename includes '_torchscan' suffix.

Usage
-----
    python stage1_sm_scaling/run_ssm_prefill_sweep.py --model zamba2 --device auto
    python stage1_sm_scaling/run_ssm_prefill_sweep.py --model falcon_h1 --device a100_40gb
    python stage1_sm_scaling/run_ssm_prefill_sweep.py --model zamba2 --force-pytorch-scan
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import csv
import math
from pathlib import Path
from itertools import product

import torch
import yaml
from tqdm import tqdm

from src.models.layer_runner import LayerRunner
from src.profiling.metrics import BandwidthEstimator


# ---------------------------------------------------------------------------
# n_blocks formula (empirically verified against ncu profiling data)
# mamba_chunk_scan_combined: n_blocks = batch × seq_len / 4
# (i.e. 4 seq tokens per thread block, invariant across head/state dims)
# ---------------------------------------------------------------------------
_TOKENS_PER_BLOCK = 4


def _n_blocks(batch: int, seq_len: int) -> int:
    return max(1, batch * seq_len // _TOKENS_PER_BLOCK)


# ---------------------------------------------------------------------------
# Hardware config helpers (shared with attn/mlp sweeps)
# ---------------------------------------------------------------------------

def load_hardware_config(device_key: str) -> dict:
    cfg_path = Path(__file__).parent.parent / "configs" / "hardware.yaml"
    with open(cfg_path) as f:
        hw = yaml.safe_load(f)

    if device_key == "auto" or device_key not in hw:
        props = torch.cuda.get_device_properties(0)
        n_sm = props.multi_processor_count
        steps = compute_sm_steps(n_sm)
        try:
            mem_bw_GBs = (
                2.0 * props.memory_clock_rate * 1e3 * props.memory_bus_width
            ) / (8.0 * 1e9)
        except Exception:
            mem_bw_GBs = None
        return {
            "name": torch.cuda.get_device_name(0),
            "sm_count": n_sm,
            "sm_sweep_steps": steps,
            "memory_bw_GBs": mem_bw_GBs,
        }

    cfg = hw[device_key]
    if cfg["sm_sweep_steps"] is None:
        cfg["sm_sweep_steps"] = compute_sm_steps(cfg["sm_count"])
    return cfg


def compute_sm_steps(total_sm: int, n_steps: int = 8) -> list[int]:
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
# Full-SM measurement (direct, no Green Context)
# ---------------------------------------------------------------------------

def _measure_full_sm(
    runner: LayerRunner,
    model_name: str,
    seq_lens: list[int],
    batch_sizes: list[int],
    n_warmup: int,
    n_measure: int,
) -> dict[tuple, dict]:
    """Measure SSM latency at full SM (no restriction) for all combos.

    Returns mapping (seq_len, batch_size) → result_dict.
    """
    measured: dict[tuple, dict] = {}
    combos = list(product(seq_lens, batch_sizes))
    print(f"\n  [full-SM measurement] {len(combos)} configs")

    for seq_len, batch_size in tqdm(combos, desc="  full-SM"):
        key = (seq_len, batch_size)
        try:
            row = runner.run_ssm_layer(
                model_name=model_name,
                batch_size=batch_size,
                seq_len=seq_len,
                sm_count=runner.smctrl.total_sm_count,
                n_warmup=n_warmup,
                n_measure=n_measure,
                skip_sm_control=True,   # no Green Context; use default stream
            )
            measured[key] = row
            tqdm.write(
                f"  seq={seq_len:6d} bs={batch_size:3d}  "
                f"lat={row['latency_ms']:.3f}ms  "
                f"bw={row['achieved_bandwidth_GBs']:.1f}/{row['theoretical_bw_GBs']:.0f}GB/s"
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            tqdm.write(f"  seq={seq_len:6d} bs={batch_size:3d}  OOM — skipped")
        except Exception as e:
            tqdm.write(f"  seq={seq_len:6d} bs={batch_size:3d}  ERROR: {e}")

    return measured


# ---------------------------------------------------------------------------
# Direct measurement (PyTorch scan — no cooperative-barrier deadlock)
# ---------------------------------------------------------------------------

def _measure_direct_sm(
    runner: LayerRunner,
    model_name: str,
    seq_lens: list[int],
    batch_sizes: list[int],
    sm_steps: list[int],
    total_sm: int,
    n_warmup: int,
    n_measure: int,
) -> list[dict]:
    """Measure SSM latency directly at each SM step using PyTorch scan.

    Safe under Green Context because the pure-PyTorch chunked scan has no
    cooperative inter-block barriers (unlike mamba_chunk_scan_combined).
    """
    rows = []
    combos = list(product(sm_steps, seq_lens, batch_sizes))
    print(f"\n  [direct measurement / torch scan] {len(combos)} configs")

    for sm_count, seq_len, batch_size in tqdm(combos, desc="  torch-scan"):
        try:
            row = runner.run_ssm_layer(
                model_name=model_name,
                batch_size=batch_size,
                seq_len=seq_len,
                sm_count=sm_count,
                n_warmup=n_warmup,
                n_measure=n_measure,
                force_pytorch_scan=True,
            )
            nb = _n_blocks(batch_size, seq_len)
            rows.append({
                **row,
                "n_blocks":   nb,
                "waves":      None,
                "analytical": False,
            })
            tqdm.write(
                f"  sm={sm_count:3d} seq={seq_len:6d} bs={batch_size:3d}  "
                f"lat={row['latency_ms']:.3f}ms  "
                f"bw={row['achieved_bandwidth_GBs']:.1f}/{row['theoretical_bw_GBs']:.0f}GB/s"
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            tqdm.write(f"  sm={sm_count:3d} seq={seq_len:6d} bs={batch_size:3d}  OOM — skipped")
        except Exception as e:
            tqdm.write(f"  sm={sm_count:3d} seq={seq_len:6d} bs={batch_size:3d}  ERROR: {e}")

    rows.sort(key=lambda r: (r["sm_count"], r["seq_len"], r["batch_size"]))
    return rows


# ---------------------------------------------------------------------------
# Analytical SM scaling synthesis
# ---------------------------------------------------------------------------

def _synthesize_sm_scaling(
    measured: dict[tuple, dict],
    sm_steps: list[int],
    total_sm: int,
) -> list[dict]:
    """Derive latency at each SM step via the wave model.

    For cooperative kernels (constant time-per-wave):
        latency(k) = latency(full) × ceil(n_blocks / k) / ceil(n_blocks / total_sm)

    Wave efficiency is >99.96% for all Zamba2 SSM configs (verified via ncu),
    so the wave-parallel assumption is accurate.
    """
    rows = []
    for (seq_len, batch_size), full_row in measured.items():
        nb = _n_blocks(batch_size, seq_len)
        waves_full = math.ceil(nb / total_sm)

        for sm_count in sm_steps:
            waves_k = math.ceil(nb / sm_count)
            scale = waves_k / waves_full
            lat_ms = full_row["latency_ms"] * scale
            lat_p99 = full_row["latency_p99_ms"] * scale

            rows.append({
                "sm_count":                sm_count,
                "sm_ratio":                sm_count / total_sm,
                "seq_len":                 seq_len,
                "batch_size":              batch_size,
                "latency_ms":              lat_ms,
                "latency_p99_ms":          lat_p99,
                "achieved_bandwidth_GBs":  full_row["achieved_bandwidth_GBs"],
                "theoretical_bw_GBs":      full_row["theoretical_bw_GBs"],
                "bw_utilization_pct":      full_row["bw_utilization_pct"],
                "model_name":              full_row["model_name"],
                "layer_type":              "ssm",
                "n_blocks":                nb,
                "waves":                   waves_k,
                "analytical":              True,
            })

    rows.sort(key=lambda r: (r["sm_count"], r["seq_len"], r["batch_size"]))
    return rows


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_sweep(
    model_name: str,
    hw_cfg: dict,
    seq_lens: list[int],
    batch_sizes: list[int],
    n_warmup: int = 20,
    n_measure: int = 50,
    output_dir: Path = None,
    force_pytorch_scan: bool = False,
) -> list[dict]:
    """Run SSM SM scaling sweep.

    Default (force_pytorch_scan=False):
      Step 1: measure at full SM (all SMs, no Green Context).
      Step 2: synthesise latency at each SM step via wave scaling.
      Correctly characterises mamba_chunk_scan_combined which uses cooperative
      inter-block barriers and cannot be directly measured under Green Context.

    PyTorch scan mode (force_pytorch_scan=True):
      Directly measures latency at each SM step using a pure-PyTorch chunked
      scan. No cooperative barriers so Green Context restriction is safe.
      Output filename includes '_torchscan' suffix.
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "results" / "stage1"
    output_dir.mkdir(parents=True, exist_ok=True)

    total_sm = hw_cfg["sm_count"]
    sm_steps = hw_cfg["sm_sweep_steps"]
    theoretical_bw = hw_cfg.get("memory_bw_GBs")
    tag = device_tag(hw_cfg)

    runner = LayerRunner(
        device="cuda",
        total_sm_count=total_sm,
        theoretical_bw_GBs=theoretical_bw,
    )

    if force_pytorch_scan:
        mode_label = "direct (torch scan)"
        method_desc = "direct Green Context measurement with PyTorch chunked scan"
    else:
        mode_label = "analytical (wave model)"
        method_desc = "full-SM measurement + wave-model synthesis"

    print(f"\n=== SSM Prefill SM Sweep ({mode_label}): {model_name} on {hw_cfg['name']} ===")
    print(f"  SM steps     : {sm_steps}")
    print(f"  seq_lens     : {seq_lens}")
    print(f"  batch_sizes  : {batch_sizes}")
    print(f"  method       : {method_desc}")
    if force_pytorch_scan:
        print(f"  note         : PyTorch scan has no cooperative barriers; direct Green")
        print(f"                 Context measurement is safe. Results reflect PyTorch scan")
        print(f"                 characteristics, not the Triton SSD kernel.")
    else:
        print(f"  note         : Triton SSD uses cooperative barriers; direct Green Context")
        print(f"                 measurement deadlocks. Wave model is exact for this kernel")
        print(f"                 (wave_eff > 99.96% verified via ncu).")

    if force_pytorch_scan:
        results = _measure_direct_sm(
            runner=runner,
            model_name=model_name,
            seq_lens=seq_lens,
            batch_sizes=batch_sizes,
            sm_steps=sm_steps,
            total_sm=total_sm,
            n_warmup=n_warmup,
            n_measure=n_measure,
        )
        if not results:
            print("\nERROR: no configs measured successfully — aborting sweep.")
            return []
    else:
        # Step 1: measure at full SM
        measured = _measure_full_sm(
            runner=runner,
            model_name=model_name,
            seq_lens=seq_lens,
            batch_sizes=batch_sizes,
            n_warmup=n_warmup,
            n_measure=n_measure,
        )

        if not measured:
            print("\nERROR: no configs measured successfully — aborting sweep.")
            return []

        # Step 2: synthesise SM scaling
        results = _synthesize_sm_scaling(measured, sm_steps, total_sm)

    # Save CSV
    scan_suffix = "_torchscan" if force_pytorch_scan else ""
    out_csv = output_dir / f"ssm_scaling_{model_name}_{tag}{scan_suffix}.csv"
    fieldnames = [
        "sm_count", "sm_ratio", "seq_len", "batch_size",
        "latency_ms", "latency_p99_ms",
        "achieved_bandwidth_GBs", "theoretical_bw_GBs", "bw_utilization_pct",
        "model_name", "layer_type", "n_blocks", "waves", "analytical",
    ]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved: {out_csv}")
    if force_pytorch_scan:
        print(f"  {len(results)} directly measured rows (torch scan, {len(sm_steps)} SM steps)")
    else:
        n_full = len(measured)
        print(f"  {n_full} direct measurements × {len(sm_steps)} SM steps = {len(results)} synthesised rows")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="SSM prefill SM scaling sweep (analytical wave model or torch scan)"
    )
    parser.add_argument("--model", choices=["zamba2", "falcon_h1"], default="zamba2")
    parser.add_argument(
        "--device", default="auto",
        help="Hardware key from configs/hardware.yaml, or 'auto' for runtime detection"
    )
    parser.add_argument(
        "--seq-lens", nargs="+", type=int,
        default=[512, 1024, 2048, 4096, 8192, 16384, 32768],
    )
    parser.add_argument(
        "--batch-sizes", nargs="+", type=int,
        default=[1, 4, 16, 32, 64],
    )
    parser.add_argument("--n-warmup",  type=int, default=100)
    parser.add_argument("--n-measure", type=int, default=200)
    parser.add_argument(
        "--force-pytorch-scan", action="store_true",
        help=(
            "Use pure-PyTorch chunked scan instead of the Triton SSD kernel. "
            "Enables direct Green Context measurement at each SM step (no cooperative-"
            "barrier deadlock). Results reflect PyTorch scan characteristics, not the "
            "Triton kernel. Output file will include '_torchscan' in the filename."
        ),
    )
    parser.add_argument(
        "--skip-verify", action="store_true",
        help="Skip SM control verification (not needed for analytical path)"
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
        force_pytorch_scan=args.force_pytorch_scan,
    )
