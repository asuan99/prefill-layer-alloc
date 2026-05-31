"""
_forward_pass.py — standalone forward pass for nsys / ncu profiling.

Run this script UNDER nsys or ncu to capture kernel traces for
profile_layer_heterogeneity.py.  Do NOT run it directly for benchmarking.

Under nsys (timeline capture):
    nsys profile \\
        --trace=cuda,nvtx \\
        --output={output_prefix} \\
        python workspace/serving-eval/_forward_pass.py \\
            --model zamba2 --seq-len 2048 --batch-size 1

Under ncu (SM-utilization capture):
    ncu \\
        --target-processes all \\
        --kernel-name regex:"mamba_chunk.*|flash.*fwd.*|ampere.*gemm.*|cutlass.*" \\
        --metrics sm__active_cycles_sum,sm__cycles_elapsed_sum \\
        --csv \\
        python workspace/serving-eval/_forward_pass.py \\
            --model zamba2 --seq-len 2048 --batch-size 1 \\
        > ncu_kernels.csv

The script warms up (--n-warmup runs), then marks a single profiled run with
NVTX ranges so nsys can label the regions:
    "warmup"         — initial passes (ncu/nsys should skip these)
    "profile_run"    — single measurement pass (ncu/nsys targets this)
    Within profile_run, each transformer block emits:
      "block_{i}_ssm", "block_{i}_attn", "block_{i}_mlp"
    (requires model to expose block-level NVTX tagging — see below)
"""

from __future__ import annotations

import sys
import os

_here      = os.path.dirname(os.path.abspath(__file__))
_workspace = os.path.dirname(_here)
sys.path.insert(0, _workspace)
sys.path.insert(0, _here)

import argparse
import time
from typing import Optional


# ---------------------------------------------------------------------------
# NVTX helpers (graceful if NVTX unavailable)
# ---------------------------------------------------------------------------

try:
    import torch.cuda.nvtx as nvtx
    _NVTX = True
except ImportError:
    _NVTX = False


class _NVTXRange:
    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        if _NVTX:
            nvtx.range_push(self.name)
        return self

    def __exit__(self, *_):
        if _NVTX:
            nvtx.range_pop()


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

_HF_REPOS = {
    "zamba2":     "Zyphra/Zamba2-7B-Instruct",
    "nemotron_h": "nvidia/Nemotron-H-8B-Base",
    "falcon_h1":  "tiiuae/Falcon-H1-7B-Instruct",
    # pure-Transformer baseline for comparison (S1.2)
    "llama3_8b":  "meta-llama/Meta-Llama-3.1-8B",
}


def _load_model(model_name: str, device: str = "cuda", dtype_str: str = "bfloat16"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(dtype_str, torch.bfloat16)

    try:
        from shared.loaders import get_model_config
        cfg = get_model_config(model_name)
        hf_repo = cfg.get("hf_repo") or _HF_REPOS[model_name]
    except Exception:
        hf_repo = _HF_REPOS.get(model_name)
        if not hf_repo:
            raise ValueError(f"Unknown model: {model_name}")

    print(f"  Loading {hf_repo} onto {device} ...")
    model = AutoModelForCausalLM.from_pretrained(
        hf_repo,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    print(f"  Model loaded ({sum(p.numel() for p in model.parameters()) / 1e9:.1f}B params)")
    return model, hf_repo


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

def run_forward(
    model,
    seq_len: int,
    batch_size: int,
    device: str = "cuda",
    vocab_size: int = 32000,
    label: str = "profile_run",
) -> float:
    """Run one forward pass under NVTX label and return wall time in ms."""
    import torch
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with _NVTXRange(label):
        with torch.no_grad():
            _ = model(input_ids)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="zamba2", choices=list(_HF_REPOS),
                   help="Model to profile")
    p.add_argument("--seq-len", type=int, default=2048,
                   help="Sequence length (tokens)")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--n-warmup", type=int, default=2,
                   help="Warmup passes (not profiled)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    return p.parse_args()


def main() -> None:
    import torch
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required")

    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Shape: batch={args.batch_size}  seq_len={args.seq_len}")

    model, hf_repo = _load_model(args.model, device=args.device, dtype_str=args.dtype)
    vocab_size = model.config.vocab_size

    # Warmup (excluded from nsys/ncu capture by tool defaults; NVTX labeled "warmup")
    print(f"  Warmup ({args.n_warmup} passes) ...")
    for i in range(args.n_warmup):
        run_forward(model, args.seq_len, args.batch_size, args.device, vocab_size, label="warmup")

    # Profile run (nsys/ncu capture this pass)
    print("  Profile run ...")
    lat_ms = run_forward(
        model, args.seq_len, args.batch_size, args.device, vocab_size,
        label="profile_run",
    )
    print(f"  Forward pass: {lat_ms:.1f} ms  "
          f"(seq={args.seq_len}, bs={args.batch_size}, model={args.model})")
    print("  Done.  nsys/ncu will capture the 'profile_run' NVTX range.")


if __name__ == "__main__":
    main()
