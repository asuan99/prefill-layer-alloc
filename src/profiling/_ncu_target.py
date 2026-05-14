"""
ncu subprocess target — thin single-kernel runner for Nsight Compute profiling.

This script is spawned BY ncu (not called directly). ncu intercepts CUDA kernel
launches and attaches hardware performance counters.

Usage (via NCURunner, not directly):
    ncu --metrics sm__active_cycles_sum,... --csv \\
        python src/profiling/_ncu_target.py \\
        --layer-type ssm --model zamba2 \\
        --sm-count 27 --seq-len 1024 --batch-size 1

The script runs n_warmup kernels (to warm GPU caches and JIT-compile Triton
kernels), then n_measure measured launches. ncu captures all kernel launches;
NCURunner picks the dominant kernel by sm__cycles_active.sum.
"""

import sys
import os
import argparse

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="ncu kernel target")
    parser.add_argument("--layer-type", choices=["ssm", "chunked_ssm", "attn", "mlp"], required=True)
    parser.add_argument("--model", choices=["zamba2", "falcon_h1"], required=True)
    parser.add_argument("--sm-count", type=int, required=True)
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--context-len", type=int, default=0)
    parser.add_argument("--prefill-chunk-tokens", type=int, default=0,
                        help="Tokens per kernel call for chunked_ssm (0 = full seq_len)")
    parser.add_argument("--n-warmup", type=int, default=10,
                        help="Warmup launches before the measured kernel")
    parser.add_argument("--n-measure", type=int, default=3,
                        help="Measured launches after warmup")
    parser.add_argument("--dtype", default="bfloat16")
    return parser.parse_args()


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    from src.smctrl import SMController
    from src.models.layer_runner import LayerRunner

    smctrl = SMController()
    runner = LayerRunner(device="cuda", dtype=dtype, smctrl=smctrl)

    # Apply SM restriction — must use runner.smctrl.get_stream() for kernels
    smctrl.set_sm_count(args.sm_count)
    _stream = smctrl.get_stream()

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

    elif args.layer_type == "chunked_ssm":
        # Profile the SSD scan kernel for one prefill chunk (prefill_chunk_tokens tokens).
        # This captures the per-call grid size used in chunked-prefill SSM, which is
        # smaller than the full-sequence grid and avoids cooperative barrier deadlock.
        import math
        import yaml
        from pathlib import Path
        from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

        cfg_path = Path(__file__).parent.parent.parent / "configs" / "models.yaml"
        with open(cfg_path) as _f:
            _raw = yaml.safe_load(_f)[args.model]
        _ssm = _raw.get("ssm", {})
        n_heads   = _ssm["n_heads"]
        head_dim  = _ssm["head_dim"]
        d_state   = _ssm["d_state"]
        n_groups  = _ssm.get("n_groups", _ssm.get("expand", 2))
        ssd_chunk = _ssm["chunk_size"]

        # Use prefill_chunk_tokens if provided, otherwise fall back to full seq_len.
        pct = args.prefill_chunk_tokens if args.prefill_chunk_tokens > 0 else args.seq_len

        x  = torch.randn(args.batch_size, pct, n_heads, head_dim, device="cuda", dtype=dtype)
        dt = torch.ones(args.batch_size, pct, n_heads, device="cuda", dtype=dtype) * 0.1
        A  = -torch.ones(n_heads, device="cuda", dtype=dtype)
        B  = torch.randn(args.batch_size, pct, n_groups, d_state, device="cuda", dtype=dtype)
        C  = torch.randn(args.batch_size, pct, n_groups, d_state, device="cuda", dtype=dtype)
        D  = torch.ones(n_heads, device="cuda", dtype=dtype)
        dt_bias = torch.zeros(n_heads, device="cuda", dtype=dtype)

        def kernel():
            with torch.no_grad():
                mamba_chunk_scan_combined(
                    x, dt, A, B, C,
                    chunk_size=ssd_chunk,
                    D=D,
                    dt_bias=dt_bias,
                    dt_softplus=True,
                )

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

    # Warmup — warms GPU caches and JIT-compiles Triton kernels.
    with torch.cuda.stream(_stream):
        for _ in range(args.n_warmup):
            kernel()
    torch.cuda.synchronize()

    # Measured launches — ncu captures all kernel launches from this process
    # and dominant-kernel selection (max sm__cycles_active.sum) identifies the
    # primary compute kernel regardless of warmup inclusion.
    with torch.cuda.stream(_stream):
        for _ in range(args.n_measure):
            kernel()
    torch.cuda.synchronize()

    smctrl.reset()


if __name__ == "__main__":
    main()
