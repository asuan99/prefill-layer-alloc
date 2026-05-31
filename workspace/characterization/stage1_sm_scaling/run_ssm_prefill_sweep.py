"""
Stage 1: SSM prefill SM scaling (analytical wave model) — CROSS-VALIDATION AUX.

**역할 강등 (S0.2):** 이 스크립트는 primary SSM 측정이 아니다.  Primary 측정은
``run_chunked_ssm_sweep.py`` (chunked Triton 직접 측정)이고, 여기서 얻은
``ssm_scaling_*.csv`` / ``ssm_scaling_*_torchscan.csv``는 교차검증 보조 용도로만
사용된다.  plot_saturation.py / plot_compare_modules.py 의 기본 SSM 곡선에서는
이 CSV들이 제외되며, ``--show-analytical`` 플래그로만 표시된다.

Analytical wave model (기본 모드):
  전체 SM에서 측정한 뒤 wave 모델로 SM-scaled latency를 합성.
  cooperative barrier 문제를 우회하지만 실측이 아닌 모델 추정값.

PyTorch scan mode (--force-pytorch-scan):
  Triton SSD 대신 PyTorch scan을 직접 Green Context로 측정.
  cooperative barrier가 없어 직접 측정 가능하나, kernel이 다르다.

Usage
-----
    python stage1_sm_scaling/run_ssm_prefill_sweep.py --model zamba2 --device auto
    python stage1_sm_scaling/run_ssm_prefill_sweep.py --model falcon_h1 --device a100_40gb
    python stage1_sm_scaling/run_ssm_prefill_sweep.py --model zamba2 --force-pytorch-scan
"""

import sys
import os
_here = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_here, "../.."))  # workspace/ — for shared.*
sys.path.insert(0, os.path.join(_here, ".."))     # characterization/ — for src.*

import argparse
import csv
import math
from pathlib import Path
from itertools import product

import torch
from tqdm import tqdm

from shared.loaders import get_hardware_config, device_tag  # noqa: F401 (re-export)
from shared.loaders import get_model_config
from src.models.layer_runner import LayerRunner
from src.profiling.metrics import BandwidthEstimator


# ---------------------------------------------------------------------------
# n_blocks formula (empirically verified against ncu profiling data)
#
# mamba_chunk_scan_combined grid:
#   nchunks  = seq_len // chunk_size
#   n_blocks = batch × nchunks × n_heads
#
# For Zamba2  (n_heads=112, chunk_size=256): n_blocks = batch × seq_len × 112/256
# For Falcon-H1 (n_heads=24, chunk_size=256): n_blocks = batch × seq_len × 24/256
#
# The old simplified formula (batch × seq_len // 4) was model-agnostic and incorrect.
# ---------------------------------------------------------------------------


def _n_blocks(batch: int, seq_len: int, n_heads: int, chunk_size: int = 256) -> int:
    nchunks = max(1, seq_len // chunk_size)
    return max(1, batch * nchunks * n_heads)


# ---------------------------------------------------------------------------
# Hardware config helpers — load_hardware_config replaced by shared.loaders.
# get_hardware_config and device_tag are imported above and re-exported so
# callers that do:
#   from stage1_sm_scaling.run_ssm_prefill_sweep import load_hardware_config, device_tag
# continue to work via the shared implementation.
# ---------------------------------------------------------------------------

# Backward-compat alias — callers should migrate to shared.loaders directly.
load_hardware_config = get_hardware_config


def compute_sm_steps(total_sm: int, n_steps: int = 8) -> list:
    steps = []
    for i in range(1, n_steps + 1):
        sm = max(1, round(total_sm * i / n_steps))
        if sm not in steps:
            steps.append(sm)
    return sorted(steps)


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
    n_heads: int,
    chunk_size: int,
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
                use_pytorch_fallback=True,
            )
            nb = _n_blocks(batch_size, seq_len, n_heads, chunk_size)
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
    n_heads: int,
    chunk_size: int,
) -> list[dict]:
    """Derive latency at each SM step via the wave model.

    For cooperative kernels (constant time-per-wave):
        latency(k) = latency(full) × ceil(n_blocks / k) / ceil(n_blocks / total_sm)

    Wave efficiency is >99.96% for all Zamba2 SSM configs (verified via ncu),
    so the wave-parallel assumption is accurate.
    """
    rows = []
    for (seq_len, batch_size), full_row in measured.items():
        nb = _n_blocks(batch_size, seq_len, n_heads, chunk_size)
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

    # Load model-specific SSM config for the correct n_blocks formula.
    model_cfg = get_model_config(model_name)
    ssm_cfg   = model_cfg.get("ssm", {})
    n_heads_ssm   = ssm_cfg["n_heads"]
    chunk_size_ssm = ssm_cfg["chunk_size"]

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
            n_heads=n_heads_ssm,
            chunk_size=chunk_size_ssm,
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
        results = _synthesize_sm_scaling(
            measured, sm_steps, total_sm,
            n_heads=n_heads_ssm, chunk_size=chunk_size_ssm,
        )

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
