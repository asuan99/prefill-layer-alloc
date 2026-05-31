"""
profile_layer_heterogeneity.py — nsys + ncu profiling (OPTIONAL APPENDIX).

S1.2 revised: the headline motivation figures (A/B/C) are counter-free.
This script provides supplementary data for Fig D (nsys kernel timeline
appendix) only.  Headline claims are complete without running this script.

Run this ONLY if nsys/ncu are available with sufficient permissions.
Output feeds plot_motivation.py --profiling-dir for Fig D.

Original S1.2 description:
profile_layer_heterogeneity.py — nsys + ncu profiling for layer-type
heterogeneity measurement.

Orchestrates two profiling steps for a hybrid model:
  1. nsys profile — captures per-kernel duration timeline (CUDA trace).
     Kernels are classified by name into: ssm_scan, attn, gemm, other.
  2. ncu profile  — captures SM active cycles / elapsed cycles per kernel class.
     Runs on isolated single-layer inputs at the same (seq_len, batch_size).

Both steps also run for a pure-Transformer baseline (Llama-3.1-8B) so that
Figure 1 of plot_layer_heterogeneity.py can contrast hybrid vs pure-Attn
SM-util distributions.

Outputs (written to --output-dir):
  nsys_{model}_{seq_len}b{batch_size}.sqlite  — nsys report (requires nsys ≥ 2023.1)
  ncu_{model}_{seq_len}b{batch_size}.csv       — per-kernel SM metrics
  kernel_timeline_{model}_{seq_len}b{batch_size}.csv — parsed + classified timeline

Usage:
    # Profile Zamba2 on a single shape:
    python workspace/serving-eval/profile_layer_heterogeneity.py \\
        --model zamba2 --seq-len 2048 --batch-size 1

    # Profile both hybrid + baseline for the comparison figure:
    python workspace/serving-eval/profile_layer_heterogeneity.py \\
        --model zamba2 --baseline llama3_8b --seq-len 2048 --batch-size 1

    # Skip ncu (fast, timeline only):
    python workspace/serving-eval/profile_layer_heterogeneity.py \\
        --model zamba2 --no-ncu

Hardware guard: A100 required for numeric conclusions.
"""

from __future__ import annotations

import sys
import os

_here      = os.path.dirname(os.path.abspath(__file__))
_workspace = os.path.dirname(_here)
sys.path.insert(0, _workspace)
sys.path.insert(0, _here)

import argparse
import csv
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import torch


# ---------------------------------------------------------------------------
# Hardware guard
# ---------------------------------------------------------------------------

_A100_KW = frozenset({"a100", "a800"})


def _is_a100() -> bool:
    return torch.cuda.is_available() and any(
        kw in torch.cuda.get_device_name(0).lower() for kw in _A100_KW
    )


def _hardware_warn() -> None:
    if not _is_a100():
        print("=" * 70)
        print("  WARNING: Non-A100 GPU — PIPELINE VALIDATION ONLY.")
        print("  Numeric SM-util conclusions from this run are not valid.")
        print("  Re-run on A100 for paper-quality results.")
        print("=" * 70)


# ---------------------------------------------------------------------------
# Kernel classification patterns
# ---------------------------------------------------------------------------

# Each pattern is tested against the CUDA kernel mangled name (lowercase).
# Order matters: first match wins.
_KERNEL_CLASSES: list[tuple[str, re.Pattern]] = [
    ("ssm_scan", re.compile(
        r"mamba_chunk|ssd_chunk|selective_scan|ssm_chunk|chunk_scan"
    )),
    ("attn", re.compile(
        r"flash_fwd|flash_attn|fmha|fused_attn|xformers.*attn"
    )),
    ("gemm", re.compile(
        r"ampere.*gemm|volta.*gemm|cutlass.*gemm|s16816gemm|sgemm|hgemm|bgemm"
    )),
    ("elementwise", re.compile(
        r"elementwise_|vectorized_elem|fused_bias|layer_norm|rms_norm"
    )),
]


def classify_kernel(kernel_name: str) -> str:
    kl = kernel_name.lower()
    for cls, pat in _KERNEL_CLASSES:
        if pat.search(kl):
            return cls
    return "other"


# ---------------------------------------------------------------------------
# nsys profiling
# ---------------------------------------------------------------------------

def run_nsys(
    model: str,
    seq_len: int,
    batch_size: int,
    output_prefix: Path,
    n_warmup: int = 2,
    extra_env: Optional[dict] = None,
) -> Optional[Path]:
    """Run nsys profile on _forward_pass.py and return .sqlite path.

    Returns None if nsys is not available.
    """
    if shutil.which("nsys") is None:
        print("  nsys not found — skipping timeline capture.")
        print("  Install CUDA toolkit nsys to enable: nsys profile ...")
        return None

    forward_script = Path(_here) / "_forward_pass.py"
    sqlite_out = output_prefix.with_suffix(".sqlite")

    cmd = [
        "nsys", "profile",
        "--trace=cuda,nvtx",
        "--capture-range=nvtx",
        "--capture-range-data=profile_run",  # capture only the profiled pass
        "--output",          str(output_prefix),
        "--force-overwrite=true",
        sys.executable, str(forward_script),
        "--model",      model,
        "--seq-len",    str(seq_len),
        "--batch-size", str(batch_size),
        "--n-warmup",   str(n_warmup),
    ]

    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)

    print(f"  nsys: {' '.join(cmd[:6])} ...")
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, env=env, timeout=600,
        )
        if result.returncode != 0:
            print(f"  nsys failed (code {result.returncode}):")
            print(f"    {result.stderr[:400]}")
            return None
        print(f"  nsys report: {sqlite_out}")
        return sqlite_out
    except subprocess.TimeoutExpired:
        print("  nsys timed out (600s)")
        return None


# ---------------------------------------------------------------------------
# nsys → kernel timeline CSV
# ---------------------------------------------------------------------------

def parse_nsys_timeline(
    sqlite_path: Path,
    output_csv: Path,
) -> Optional[Path]:
    """Extract kernel timeline from nsys sqlite and write classified CSV.

    Uses ``nsys stats --report cuda_gpu_trace`` to dump kernel events,
    then classifies each kernel and writes:
      kernel_name, start_ns, duration_ns, layer_class

    Returns output_csv path, or None on failure.
    """
    if shutil.which("nsys") is None:
        return None

    cmd = [
        "nsys", "stats",
        "--report", "cuda_gpu_trace",
        "--format", "csv",
        str(sqlite_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            print(f"  nsys stats failed: {result.stderr[:200]}")
            return None
    except subprocess.TimeoutExpired:
        print("  nsys stats timed out")
        return None

    lines = result.stdout.strip().splitlines()
    if not lines:
        print("  nsys stats produced no output")
        return None

    # Parse CSV: find header row
    header = None
    rows = []
    for line in lines:
        if header is None:
            if "Name" in line or "CorrId" in line:
                header = [h.strip().strip('"') for h in line.split(",")]
            continue
        fields = [f.strip().strip('"') for f in line.split(",")]
        if len(fields) == len(header):
            rows.append(dict(zip(header, fields)))

    if not rows:
        print("  No kernel rows parsed from nsys output")
        return None

    # Write classified timeline CSV
    fieldnames = ["kernel_name", "start_ns", "duration_ns", "layer_class"]
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            name  = row.get("Name", row.get("name", ""))
            start = row.get("Start (ns)", row.get("start_ns", "0"))
            dur   = row.get("Duration (ns)", row.get("duration_ns", "0"))
            writer.writerow({
                "kernel_name": name,
                "start_ns":    start,
                "duration_ns": dur,
                "layer_class": classify_kernel(name),
            })
    print(f"  Timeline CSV: {output_csv}  ({len(rows)} kernels)")
    return output_csv


# ---------------------------------------------------------------------------
# ncu profiling
# ---------------------------------------------------------------------------

_NCU_METRICS = [
    "sm__active_cycles_sum",
    "sm__cycles_elapsed_sum",
    "sm__warps_active.sum",
    "l1tex__t_bytes.sum",
    "lts__t_bytes.sum",
]

# Kernel name regex for ncu --kernel-name filter
_NCU_KERNEL_FILTER = (
    r"mamba_chunk.*|ssd_chunk.*|selective_scan.*"
    r"|flash_fwd.*|flash_attn.*"
    r"|ampere.*gemm|cutlass.*gemm|volta.*gemm"
)


def run_ncu(
    model: str,
    seq_len: int,
    batch_size: int,
    output_csv: Path,
    n_warmup: int = 1,
) -> Optional[Path]:
    """Run ncu on _forward_pass.py and save per-kernel SM metrics to CSV.

    Returns output_csv path, or None if ncu is unavailable / fails.
    """
    if shutil.which("ncu") is None:
        print("  ncu not found — skipping SM-util measurement.")
        print("  Install CUDA toolkit ncu to enable: ncu --target-processes all ...")
        return None

    forward_script = Path(_here) / "_forward_pass.py"
    metrics_str = ",".join(_NCU_METRICS)

    cmd = [
        "ncu",
        "--target-processes", "all",
        "--kernel-name",      f"regex:{_NCU_KERNEL_FILTER}",
        "--metrics",          metrics_str,
        "--csv",
        "--page",             "raw",
        sys.executable, str(forward_script),
        "--model",      model,
        "--seq-len",    str(seq_len),
        "--batch-size", str(batch_size),
        "--n-warmup",   str(n_warmup),
    ]

    print(f"  ncu: {' '.join(cmd[:4])} ...")
    print("  (This may require CAP_SYS_ADMIN or ncu sudo privileges.)")
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=1800,
        )
        if result.returncode != 0 and "ERR_NVGPUCTRPERM" in result.stderr:
            print("  ncu: ERR_NVGPUCTRPERM — insufficient permissions.")
            print("  Try: sudo ncu ...  or set /proc/driver/nvidia/params NVreg_RestrictProfilingToAdminUsers=0")
            return None
        if result.returncode != 0:
            print(f"  ncu failed (code {result.returncode}): {result.stderr[:200]}")
            return None
        with open(output_csv, "w") as f:
            f.write(result.stdout)
        print(f"  ncu CSV: {output_csv}")
        return output_csv
    except subprocess.TimeoutExpired:
        print("  ncu timed out (1800s)")
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="zamba2",
                   choices=["zamba2", "nemotron_h", "falcon_h1"],
                   help="Hybrid model to profile")
    p.add_argument("--baseline", default="llama3_8b",
                   choices=["llama3_8b", "none"],
                   help="Pure-Transformer baseline for comparison (none = skip)")
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--n-warmup", type=int, default=2,
                   help="Warmup passes before profiled pass")
    p.add_argument("--output-dir", type=Path,
                   default=Path(_here) / "results" / "profiling")
    p.add_argument("--no-ncu", action="store_true",
                   help="Skip ncu step (timeline only)")
    p.add_argument("--no-nsys", action="store_true",
                   help="Skip nsys step (ncu only)")
    return p.parse_args()


def _profile_one(
    model: str,
    seq_len: int,
    batch_size: int,
    output_dir: Path,
    n_warmup: int,
    run_ncu_step: bool,
    run_nsys_step: bool,
) -> dict:
    """Profile one model, return paths to output files."""
    tag = f"{model}_sl{seq_len}_bs{batch_size}"
    output_dir.mkdir(parents=True, exist_ok=True)

    out = {"model": model, "seq_len": seq_len, "batch_size": batch_size}

    if run_nsys_step:
        nsys_prefix  = output_dir / f"nsys_{tag}"
        sqlite_path  = run_nsys(model, seq_len, batch_size, nsys_prefix, n_warmup)
        timeline_csv = None
        if sqlite_path and sqlite_path.exists():
            timeline_csv = parse_nsys_timeline(
                sqlite_path,
                output_dir / f"kernel_timeline_{tag}.csv",
            )
        out["nsys_sqlite"]    = str(sqlite_path) if sqlite_path else None
        out["timeline_csv"]   = str(timeline_csv) if timeline_csv else None

    if run_ncu_step:
        ncu_csv = run_ncu(
            model, seq_len, batch_size,
            output_dir / f"ncu_{tag}.csv",
            n_warmup=1,
        )
        out["ncu_csv"] = str(ncu_csv) if ncu_csv else None

    return out


def main() -> None:
    args = parse_args()
    _hardware_warn()

    print(f"\n  Model:      {args.model}")
    print(f"  seq_len:    {args.seq_len}    batch_size: {args.batch_size}")
    print(f"  output_dir: {args.output_dir}")
    print(f"  nsys: {'yes' if not args.no_nsys else 'skip'}  "
          f"ncu: {'yes' if not args.no_ncu else 'skip'}")

    results = []

    print(f"\n--- Profiling hybrid model: {args.model} ---")
    r = _profile_one(
        args.model, args.seq_len, args.batch_size, args.output_dir, args.n_warmup,
        run_ncu_step=not args.no_ncu,
        run_nsys_step=not args.no_nsys,
    )
    results.append(r)

    if args.baseline != "none":
        print(f"\n--- Profiling baseline: {args.baseline} ---")
        r2 = _profile_one(
            args.baseline, args.seq_len, args.batch_size, args.output_dir, args.n_warmup,
            run_ncu_step=not args.no_ncu,
            run_nsys_step=not args.no_nsys,
        )
        results.append(r2)

    print("\n=== Profiling complete ===")
    for r in results:
        print(f"  {r['model']}:")
        print(f"    timeline_csv : {r.get('timeline_csv')}")
        print(f"    ncu_csv      : {r.get('ncu_csv')}")
    print(f"\nNext step: python workspace/serving-eval/plot_layer_heterogeneity.py "
          f"--results-dir {args.output_dir}")


if __name__ == "__main__":
    main()
