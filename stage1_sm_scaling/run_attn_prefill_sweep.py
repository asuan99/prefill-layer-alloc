"""
Stage 1: Attention prefill layer SM scaling curve measurement.

Each SM level runs in an isolated subprocess so a CUDA OOM or context error
at one SM count cannot contaminate subsequent SM levels. Failed configs are
retried once in a fresh subprocess (fresh CUDA context).

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
from stage1_sm_scaling.run_ssm_prefill_sweep import load_hardware_config, device_tag
from stage1_sm_scaling._sweep_worker import run_sm_level_subprocess


# ---------------------------------------------------------------------------
# CSV field names (attn-specific, includes context_len and error)
# ---------------------------------------------------------------------------
_FIELDNAMES = [
    "sm_count", "sm_ratio", "seq_len", "batch_size", "context_len",
    "latency_ms", "latency_p99_ms",
    "achieved_bandwidth_GBs", "theoretical_bw_GBs", "bw_utilization_pct",
    "model_name", "layer_type", "error",
]


def _save_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDNAMES, extrasaction="ignore", restval="")
        writer.writeheader()
        writer.writerows(rows)


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
    subprocess_timeout: int = 3600,
) -> list[dict]:
    """Run Attention SM scaling sweep with subprocess isolation per SM level.

    Each SM level runs in its own subprocess. If a CUDA OOM or context error
    occurs, the subprocess exits cleanly and the parent retries unreached
    configs in a fresh process. Results are saved to CSV after each SM level.
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "results" / "stage1"
    output_dir.mkdir(parents=True, exist_ok=True)

    sm_steps = hw_cfg["sm_sweep_steps"]
    total_sm = hw_cfg["sm_count"]
    theoretical_bw = hw_cfg.get("memory_bw_GBs")
    tag = device_tag(hw_cfg)
    out_csv = output_dir / f"attn_scaling_{model_name}_{tag}.csv"

    # Lightweight runner only for SM control verification; no layer runs in parent.
    runner = LayerRunner(
        device="cuda",
        total_sm_count=total_sm,
        theoretical_bw_GBs=theoretical_bw,
    )

    if not skip_verify:
        ok = runner.verify_sm_control(verbose=True)
        if not ok:
            print(
                "WARNING: SM control verification failed — latency will NOT vary with sm_count.\n"
                "         Proceeding anyway; check Green Context support on this device.\n"
            )

    n_configs = len(sm_steps) * len(seq_lens) * len(batch_sizes)
    print(f"\n=== Attention Prefill SM Sweep: {model_name} on {hw_cfg['name']} ===")
    print(f"  SM steps          : {sm_steps}")
    print(f"  seq_lens          : {seq_lens}")
    print(f"  batch_sizes       : {batch_sizes}")
    print(f"  context_len       : {context_len}")
    print(f"  n_warmup/n_measure: {n_warmup}/{n_measure}")
    print(f"  Total configs     : {n_configs}")
    print(f"  Isolation         : subprocess per SM level (retry once on CUDA error)\n")

    all_results: list[dict] = []
    n_ok = n_err = n_oom = 0

    for sm_count in tqdm(sm_steps, desc="SM levels", unit="SM"):
        tqdm.write(f"\n── sm={sm_count} ({sm_count/total_sm:.0%}) ──")

        sm_rows = run_sm_level_subprocess(
            layer_type="attn",
            model_name=model_name,
            sm_count=sm_count,
            seq_lens=seq_lens,
            batch_sizes=batch_sizes,
            total_sm=total_sm,
            bw_gbs=theoretical_bw,
            n_warmup=n_warmup,
            n_measure=n_measure,
            context_len=context_len,
            timeout=subprocess_timeout,
            max_retries=1,
        )

        # inject context_len into error rows that don't have it
        for row in sm_rows:
            row.setdefault("context_len", context_len)
            if row.get("error"):
                n_oom += 1 if row["error"] == "OOM" else 0
                n_err += 1 if row["error"] not in ("OOM", "") else 0
            else:
                n_ok += 1

        all_results.extend(sm_rows)

        # Intermediate save after each SM level so partial results are not lost.
        _save_csv(all_results, out_csv)
        tqdm.write(
            f"  sm={sm_count}: {sum(1 for r in sm_rows if not r.get('error'))} OK  "
            f"{sum(1 for r in sm_rows if r.get('error')=='OOM')} OOM  "
            f"{sum(1 for r in sm_rows if r.get('error') not in ('', 'OOM', None))} err  "
            f"→ running total saved to {out_csv.name}"
        )

    print(f"\nSaved: {out_csv}")
    print(f"  {n_ok} OK  {n_oom} OOM  {n_err} CUDA/other errors  "
          f"out of {n_configs} total configs")

    return all_results


def parse_args():
    parser = argparse.ArgumentParser(description="Attention prefill SM scaling sweep")
    parser.add_argument("--model", choices=["zamba2", "falcon_h1"], default="zamba2")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--seq-lens", nargs="+", type=int,
        default=[512, 1024, 2048, 4096, 8192, 16384],
    )
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4, 16, 32])
    parser.add_argument("--context-len", type=int, default=4096,
                        help="Pre-filled KV cache length (0 = empty context)")
    parser.add_argument("--n-warmup",  type=int, default=100)
    parser.add_argument("--n-measure", type=int, default=200)
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--subprocess-timeout", type=int, default=3600,
                        help="Per-SM-level subprocess timeout in seconds")
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
        skip_verify=args.skip_verify,
        subprocess_timeout=args.subprocess_timeout,
    )
