"""
DEPRECATED (S0.3): deadlock 우회는 chunked prefill(chunked_ssm_runner.py)로 단일화됨.
본 two-pass 경로는 보관용이며 측정 파이프라인에서 제외됨.
사유: chunked 가 프로덕션 Triton kernel을 그대로 사용하고 Falcon-H1 CP/Marconi와 정합.
출력 CSV(ssm_twopass_*.csv)는 plot/analyze 스크립트의 기본 입력 glob에서 제외됨.

---

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
_here = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_here, "../.."))  # workspace/
sys.path.insert(0, os.path.join(_here, ".."))     # characterization/

import argparse
import csv
import json
import subprocess
from pathlib import Path
from itertools import product
from typing import Optional

import torch
from tqdm import tqdm

from shared.loaders import get_hardware_config, device_tag
from shared.sweep_spec import n_blocks as spec_n_blocks
from stage1_sm_scaling.run_ssm_prefill_sweep import compute_sm_steps

# Backward-compat alias
load_hardware_config = get_hardware_config


# ---------------------------------------------------------------------------
# Per-config subprocess runner
# ---------------------------------------------------------------------------

def _run_single_config(
    model_name: str,
    sm_count: int,
    seq_len: int,
    batch_size: int,
    total_sm: int,
    bw_gbs: Optional[float],
    n_warmup: int,
    n_measure: int,
    worker_script: Path,
    timeout: int = 600,
) -> Optional[dict]:
    """Launch _two_pass_worker.py for a single (sm_count, seq_len, batch_size).

    One subprocess per config avoids per-SM-level timeouts when some configs
    are much slower than others (e.g. seq=32768, batch=64).

    Returns the result dict, or None on failure/timeout.
    """
    cmd = [
        sys.executable, str(worker_script),
        "--model",       model_name,
        "--sm-count",    str(sm_count),
        "--seq-lens",    str(seq_len),
        "--batch-sizes", str(batch_size),
        "--total-sm",    str(total_sm),
        "--n-warmup",    str(n_warmup),
        "--n-measure",   str(n_measure),
    ]
    if bw_gbs is not None:
        cmd += ["--bw-gbs", str(bw_gbs)]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if proc.stderr:
            for line in proc.stderr.strip().splitlines():
                tqdm.write(f"  [worker] {line}")
        if proc.returncode != 0:
            tqdm.write(
                f"  [worker] exited with code {proc.returncode} "
                f"for sm={sm_count} seq={seq_len} bs={batch_size}"
            )
            return None
        rows = json.loads(proc.stdout.strip() or "[]")
        return rows[0] if rows else None
    except subprocess.TimeoutExpired:
        tqdm.write(
            f"  [worker] timeout ({timeout}s) for sm={sm_count} "
            f"seq={seq_len} bs={batch_size}"
        )
        return None
    except Exception as e:
        tqdm.write(
            f"  [worker] error for sm={sm_count} seq={seq_len} bs={batch_size}: {e}"
        )
        return None


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
    print(f"  mode        : {'per-config subprocess' if isolate else 'in-process'}")
    print(f"  note        : Two-pass kernel has no cooperative barriers.")
    print(f"                Direct Green Context measurement is safe.")
    print(f"                Compare against ssm_scaling_{model_name}_{tag}.csv (wave model).")

    all_rows: list[dict] = []

    n_configs = len(sm_steps) * len(seq_lens) * len(batch_sizes)
    pbar = tqdm(total=n_configs, desc="configs")

    if not isolate:
        from src.models.zamba2_two_pass import TwoPassLayerRunner
        runner = TwoPassLayerRunner(
            device="cuda",
            total_sm_count=total_sm,
            theoretical_bw_GBs=bw_gbs,
        )

    for sm_count in sm_steps:
        tqdm.write(f"\n  SM {sm_count}/{total_sm}:")

        for seq_len, batch_size in product(seq_lens, batch_sizes):
            if isolate:
                row = _run_single_config(
                    model_name=model_name,
                    sm_count=sm_count,
                    seq_len=seq_len,
                    batch_size=batch_size,
                    total_sm=total_sm,
                    bw_gbs=bw_gbs,
                    n_warmup=n_warmup,
                    n_measure=n_measure,
                    worker_script=worker_script,
                )
            else:
                row = None
                try:
                    row = runner.run_ssm_layer(
                        model_name=model_name,
                        batch_size=batch_size,
                        seq_len=seq_len,
                        sm_count=sm_count,
                        n_warmup=n_warmup,
                        n_measure=n_measure,
                    )
                    tqdm.write(
                        f"  OK  sm={sm_count:3d} seq={seq_len:6d} bs={batch_size:3d} "
                        f"lat={row['latency_ms']:.3f}ms"
                    )
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    tqdm.write(f"  OOM sm={sm_count:3d} seq={seq_len:6d} bs={batch_size:3d}")
                except Exception as e:
                    tqdm.write(f"  ERR sm={sm_count:3d} seq={seq_len:6d} bs={batch_size:3d}: {e}")

            pbar.update(1)

            if row is not None:
                row["analytical"] = False
                # Single n_blocks source (shared.sweep_spec). Replaces the old
                # model-agnostic `batch × seq // 4` formula (Phase 0 audit item 8).
                row["n_blocks"] = spec_n_blocks(
                    row["model_name"], row["batch_size"], row["seq_len"]
                )
                all_rows.append(row)

    pbar.close()

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
    print(f"  {len(all_rows)} rows out of "
          f"{len(sm_steps) * len(seq_lens) * len(batch_sizes)} attempted "
          f"({len(sm_steps)} SM levels × {len(seq_lens) * len(batch_sizes)} configs)")
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
        default=[512, 1024, 2048, 4096, 8192, 16384, 32768],
        help=(
            "Sequence lengths to sweep. Matches the wave-model sweep for direct comparison. "
            "Large configs use per-config subprocesses so no single subprocess exceeds the "
            "timeout. Use --seq-lens 512 1024 2048 for a quick smoke test."
        ),
    )
    parser.add_argument(
        "--batch-sizes", nargs="+", type=int,
        default=[1, 4, 16, 32, 64],
    )
    parser.add_argument("--n-warmup",  type=int, default=10,
                        help="Warmup iterations per config.")
    parser.add_argument("--n-measure", type=int, default=20,
                        help="Measurement iterations per config.")
    parser.add_argument(
        "--no-isolate", action="store_true",
        help=(
            "Run all configs in the same process instead of one subprocess per config. "
            "Faster but CUDA context corruption at one config will affect subsequent ones."
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
