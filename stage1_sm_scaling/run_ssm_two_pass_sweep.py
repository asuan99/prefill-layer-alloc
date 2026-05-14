"""
Stage 1: SSM two-pass kernel SM scaling curve measurement.

Uses TwoPassSSMKernel (ssd_chunk_scan_twopass) which decomposes the cooperative
Triton SSD kernel into three sequential phases with no inter-block grid barriers.
This allows direct latency measurement at each SM step under CUDA Green Context
without the deadlock that kills mamba_chunk_scan_combined.

Comparison context
------------------
  run_ssm_prefill_sweep.py (default mode):
    Analytical wave model — measures at full SM, synthesises SM-scaled latency.
    Output: ssm_scaling_{model}_{tag}.csv  (analytical=True)

  run_ssm_prefill_sweep.py --force-pytorch-scan:
    Pure-PyTorch chunked scan, direct measurement.  Different compute profile
    from the Triton kernel (no d_state B/C contraction).
    Output: ssm_scaling_{model}_{tag}_torchscan.csv  (analytical=False)

  THIS SCRIPT (two-pass):
    Two-pass three-kernel decomposition of the full SSD algorithm.  Same
    arithmetic as mamba_chunk_scan_combined but no cooperative barrier.  Direct
    measurement at each SM step.
    Output: ssm_twopass_{model}_{tag}.csv  (analytical=False)

The two-pass output can be compared directly against the wave-model CSV to
validate the analytical approximation and to find any configs where wave_eff < 99%.

Usage
-----
    python stage1_sm_scaling/run_ssm_two_pass_sweep.py --model zamba2 --device auto
    python stage1_sm_scaling/run_ssm_two_pass_sweep.py --model falcon_h1 --device a100_40gb
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import csv
import json
import subprocess
from pathlib import Path
from itertools import product

import torch
import yaml
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Hardware config helpers (identical to run_ssm_prefill_sweep.py)
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
# Per-SM subprocess runner
# ---------------------------------------------------------------------------

def _run_sm_subprocess(
    model_name: str,
    sm_count: int,
    seq_lens: list[int],
    batch_sizes: list[int],
    total_sm: int,
    bw_gbs: Optional[float],
    n_warmup: int,
    n_measure: int,
    worker_script: Path,
) -> list[dict]:
    """Launch _two_pass_worker.py in a subprocess for one SM level.

    Returns list of result dicts (empty on subprocess failure).
    """
    cmd = [
        sys.executable, str(worker_script),
        "--model",      model_name,
        "--sm-count",   str(sm_count),
        "--seq-lens",   *[str(s) for s in seq_lens],
        "--batch-sizes", *[str(b) for b in batch_sizes],
        "--total-sm",   str(total_sm),
        "--n-warmup",   str(n_warmup),
        "--n-measure",  str(n_measure),
    ]
    if bw_gbs is not None:
        cmd += ["--bw-gbs", str(bw_gbs)]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if proc.stderr:
            for line in proc.stderr.strip().splitlines():
                tqdm.write(f"  [worker] {line}")
        if proc.returncode != 0:
            tqdm.write(f"  [worker] exited with code {proc.returncode} for sm={sm_count}")
            return []
        rows = json.loads(proc.stdout.strip() or "[]")
        return rows
    except subprocess.TimeoutExpired:
        tqdm.write(f"  [worker] timeout for sm={sm_count}")
        return []
    except Exception as e:
        tqdm.write(f"  [worker] error for sm={sm_count}: {e}")
        return []


# ---------------------------------------------------------------------------
# Optional type hint import
# ---------------------------------------------------------------------------

from typing import Optional


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
    isolate: bool = True,
) -> list[dict]:
    """Run two-pass SSM SM scaling sweep.

    Each SM level is measured in an isolated subprocess (_two_pass_worker.py)
    to contain any residual CUDA context state.

    Output CSV: ssm_twopass_{model}_{tag}.csv
    Columns: sm_count, sm_ratio, seq_len, batch_size, latency_ms,
             latency_p99_ms, achieved_bandwidth_GBs, theoretical_bw_GBs,
             bw_utilization_pct, model_name, layer_type, n_blocks, analytical
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "results" / "stage1"
    output_dir.mkdir(parents=True, exist_ok=True)

    total_sm = hw_cfg["sm_count"]
    sm_steps = hw_cfg["sm_sweep_steps"]
    bw_gbs   = hw_cfg.get("memory_bw_GBs")
    tag      = device_tag(hw_cfg)

    worker_script = Path(__file__).parent / "_two_pass_worker.py"

    print(f"\n=== SSM Two-Pass SM Sweep: {model_name} on {hw_cfg['name']} ===")
    print(f"  SM steps    : {sm_steps}")
    print(f"  seq_lens    : {seq_lens}")
    print(f"  batch_sizes : {batch_sizes}")
    print(f"  mode        : {'subprocess isolation' if isolate else 'in-process'}")
    print(f"  note        : Two-pass kernel has no cooperative barriers.")
    print(f"                Direct Green Context measurement is safe.")
    print(f"                Compare against ssm_scaling_{model_name}_{tag}.csv (wave model).")

    all_rows: list[dict] = []

    for sm_count in tqdm(sm_steps, desc="SM levels"):
        tqdm.write(f"\n  SM {sm_count}/{total_sm}:")

        if isolate:
            rows = _run_sm_subprocess(
                model_name=model_name,
                sm_count=sm_count,
                seq_lens=seq_lens,
                batch_sizes=batch_sizes,
                total_sm=total_sm,
                bw_gbs=bw_gbs,
                n_warmup=n_warmup,
                n_measure=n_measure,
                worker_script=worker_script,
            )
        else:
            from src.models.zamba2_two_pass import TwoPassLayerRunner
            runner = TwoPassLayerRunner(
                device="cuda",
                total_sm_count=total_sm,
                theoretical_bw_GBs=bw_gbs,
            )
            rows = []
            for seq_len, batch_size in product(seq_lens, batch_sizes):
                try:
                    row = runner.run_ssm_layer(
                        model_name=model_name,
                        batch_size=batch_size,
                        seq_len=seq_len,
                        sm_count=sm_count,
                        n_warmup=n_warmup,
                        n_measure=n_measure,
                    )
                    rows.append(row)
                    tqdm.write(
                        f"  OK  sm={sm_count:3d} seq={seq_len:6d} bs={batch_size:3d} "
                        f"lat={row['latency_ms']:.3f}ms"
                    )
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    tqdm.write(f"  OOM sm={sm_count:3d} seq={seq_len:6d} bs={batch_size:3d}")
                except Exception as e:
                    tqdm.write(f"  ERR sm={sm_count:3d} seq={seq_len:6d} bs={batch_size:3d}: {e}")

        # Annotate with sweep metadata not set by the worker
        for row in rows:
            row["analytical"] = False
            # n_blocks: two-pass doesn't have a fixed blocks-per-token formula
            # but we record the original wave model's formula for reference
            row["n_blocks"] = max(1, row["batch_size"] * row["seq_len"] // 4)

        all_rows.extend(rows)

    all_rows.sort(key=lambda r: (r["sm_count"], r["seq_len"], r["batch_size"]))

    out_csv = output_dir / f"ssm_twopass_{model_name}_{tag}.csv"
    fieldnames = [
        "sm_count", "sm_ratio", "seq_len", "batch_size",
        "latency_ms", "latency_p99_ms",
        "achieved_bandwidth_GBs", "theoretical_bw_GBs", "bw_utilization_pct",
        "model_name", "layer_type", "n_blocks", "analytical",
    ]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nSaved: {out_csv}")
    print(f"  {len(all_rows)} rows ({len(sm_steps)} SM levels × up to "
          f"{len(seq_lens) * len(batch_sizes)} configs each)")
    if all_rows:
        n_analytical = output_dir / f"ssm_scaling_{model_name}_{tag}.csv"
        if n_analytical.exists():
            print(f"  Compare with wave model: {n_analytical}")

    return all_rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "SSM two-pass SM scaling sweep — direct measurement with no "
            "cooperative barrier deadlock"
        )
    )
    parser.add_argument("--model", choices=["zamba2", "falcon_h1"], default="zamba2")
    parser.add_argument(
        "--device", default="auto",
        help="Hardware key from configs/hardware.yaml, or 'auto' for runtime detection",
    )
    parser.add_argument(
        "--seq-lens", nargs="+", type=int,
        default=[512, 1024, 2048, 4096],
        help=(
            "Sequence lengths to sweep. Default is conservative (≤4096) because the "
            "pure-Python sequential scan is ~30-50× slower than the Triton kernel. "
            "Large seq (16384+) with large batch will hit the subprocess timeout. "
            "Use --seq-lens 512 1024 2048 for a quick smoke test."
        ),
    )
    parser.add_argument(
        "--batch-sizes", nargs="+", type=int,
        default=[1, 4, 16, 32],
    )
    parser.add_argument("--n-warmup",  type=int, default=2,
                        help="Warmup iterations per config. Keep small (2-5) for two-pass.")
    parser.add_argument("--n-measure", type=int, default=5,
                        help="Measurement iterations per config. Keep small (5-10) for two-pass.")
    parser.add_argument(
        "--no-isolate", action="store_true",
        help=(
            "Run all SM levels in the same process instead of per-SM subprocesses. "
            "Faster but CUDA context corruption at one SM level will poison subsequent levels."
        ),
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
        isolate=not args.no_isolate,
    )
