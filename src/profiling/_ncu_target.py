"""
ncu subprocess target — thin single-kernel runner for Nsight Compute profiling.

This script is spawned BY ncu (not called directly). ncu intercepts CUDA kernel
launches and attaches hardware performance counters.

Usage (via NCURunner, not directly):
    ncu --metrics sm__active_cycles_sum,... \\
        --launch-count 3 --launch-skip 10 \\
        --csv python src/profiling/_ncu_target.py \\
        --layer-type ssm --model zamba2 \\
        --sm-count 27 --seq-len 1024 --batch-size 1

The script runs n_warmup kernels (to warm GPU state), then the measured kernel.
ncu's --launch-skip skips the warmup launches automatically.
"""

import sys
import os
import argparse

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="ncu kernel target")
    parser.add_argument("--layer-type", choices=["ssm", "attn", "mlp"], required=True)
    parser.add_argument("--model", choices=["zamba2", "falcon_h1"], required=True)
    parser.add_argument("--sm-count", type=int, required=True)
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--context-len", type=int, default=0)
    parser.add_argument("--n-warmup", type=int, default=10,
                        help="Warmup launches before the measured kernel")
    parser.add_argument("--n-measure", type=int, default=3,
                        help="Measured launches (ncu --launch-count should match)")
    parser.add_argument("--dtype", default="bfloat16")
    return parser.parse_args()


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    from src.smctrl.libsmctrl_wrapper import SMController
    from src.models.layer_runner import LayerRunner

    smctrl = SMController()
    runner = LayerRunner(device="cuda", dtype=dtype)

    # Apply SM restriction
    smctrl.set_sm_count(args.sm_count)

    # Build the kernel function
    if args.layer_type == "ssm":
        extractor = runner._get_extractor(args.model)
        try:
            layer = extractor.get_ssm_layer()
        except Exception:
            layer = runner._build_fallback_ssm(args.model)
        inputs = extractor.make_ssm_inputs(args.batch_size, args.seq_len)
        hidden_states = inputs["hidden_states"]

        def kernel():
            with torch.no_grad():
                layer(hidden_states)

    elif args.layer_type == "attn":
        extractor = runner._get_extractor(args.model)
        cfg = extractor.get_model_config()
        n_heads = cfg.get("n_attn_heads", 8)
        n_kv_heads = cfg.get("n_kv_heads", 8)
        head_dim = cfg.get("attn_head_dim", 256)
        query = torch.randn(
            args.batch_size, args.seq_len, n_heads, head_dim,
            device="cuda", dtype=dtype
        )
        total_kv_len = args.context_len + args.seq_len
        kv_cache = torch.randn(
            args.batch_size, total_kv_len, 2, n_kv_heads, head_dim,
            device="cuda", dtype=dtype
        )
        key = kv_cache[:, :, 0]
        value = kv_cache[:, :, 1]
        kernel = runner._build_attn_fn(query, key, value, n_heads, n_kv_heads, head_dim,
                                       use_flashinfer=True)

    else:  # mlp
        extractor = runner._get_extractor(args.model)
        cfg = extractor.get_model_config()
        hidden_size = cfg["hidden_size"]
        intermediate_size = cfg.get("intermediate_size", 4096)
        try:
            layer = extractor.get_mlp_layer()
        except Exception:
            layer = runner._build_fallback_mlp(hidden_size, intermediate_size)
        inputs = torch.randn(
            args.batch_size, args.seq_len, hidden_size, device="cuda", dtype=dtype
        )

        def kernel():
            with torch.no_grad():
                layer(inputs)

    # Warmup — outside NVTX range, ncu --nvtx-include skips these
    for _ in range(args.n_warmup):
        kernel()
    torch.cuda.synchronize()

    # Measured launches — inside "ncu_measure" NVTX range
    # ncu --nvtx --nvtx-include "ncu_measure" profiles only these kernels
    torch.cuda.nvtx.range_push("ncu_measure")
    for _ in range(args.n_measure):
        kernel()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

    smctrl.reset()


if __name__ == "__main__":
    main()
