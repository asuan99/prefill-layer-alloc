"""
OPTIONAL — ncu subprocess target (run_ncu_profile.py 전용).

이 스크립트는 ncu (NCURunner) 가 spawn 하는 child process 로만 쓰인다.
직접 실행하지 않는다. run_ncu_profile.py 가 OPTIONAL ENRICHMENT 이므로
이 파일도 헤드라인 critical path 에 속하지 않는다.

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

# Allow imports from characterization/ and workspace/
_here = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_here, "../../.."))  # workspace/
sys.path.insert(0, os.path.join(_here, "../.."))     # characterization/

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

    # Do NOT call smctrl.set_sm_count() here.
    # CUPTI (the API ncu uses for hardware counter collection) is incompatible with
    # CUDA Green Contexts. Restricting SMs via Green Context while ncu is attached
    # causes "Failed to prepare kernel for profiling / Unknown Error on device 0."
    #
    # Design intent: ncu profiling always runs at full GPU.  args.sm_count is the
    # *target allocation* passed through to _derive_sm_util / _add_analytical_wave,
    # where n_waves = ceil(measured_grid_size / sm_count) is computed analytically.
    # Sweeping sm_count therefore produces different wave estimates from the same
    # physical kernel run, which is exactly what we want for allocation analysis.
    _stream = torch.cuda.current_stream()

    # Build the kernel function
    if args.layer_type == "ssm":
        # Profile mamba_chunk_scan_combined directly (not through the full model layer).
        # Using the full layer (in_proj + scan + out_proj) causes two problems under ncu:
        #   1. Multiple kernels captured — dominant selection by sm__cycles_active.sum can
        #      pick a GEMM rather than the scan kernel at small batch/seq sizes.
        #   2. ncu hardware counter collection on the cooperative grid.sync() kernel with
        #      tens-of-thousands of blocks produces incorrect sm_util_per_sm_pct values.
        # Calling mamba_chunk_scan_combined directly mirrors the chunked_ssm branch and
        # ensures ncu sees exactly one target kernel with the correct grid dimensions.
        import math
        from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
        from shared.loaders import get_model_config

        _raw = get_model_config(args.model)
        _ssm = _raw.get("ssm", {})
        n_heads   = _ssm["n_heads"]
        head_dim  = _ssm["head_dim"]
        d_state   = _ssm["d_state"]
        n_groups  = _ssm.get("n_groups", _ssm.get("expand", 2))
        ssd_chunk = _ssm["chunk_size"]

        x  = torch.randn(args.batch_size, args.seq_len, n_heads, head_dim, device="cuda", dtype=dtype)
        dt = torch.ones(args.batch_size, args.seq_len, n_heads, device="cuda", dtype=dtype) * 0.1
        A  = -torch.ones(n_heads, device="cuda", dtype=dtype)
        B  = torch.randn(args.batch_size, args.seq_len, n_groups, d_state, device="cuda", dtype=dtype)
        C  = torch.randn(args.batch_size, args.seq_len, n_groups, d_state, device="cuda", dtype=dtype)
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
        # flashinfer is preferred but may be unavailable or incompatible with the
        # current CUDA runtime (e.g. cu124 wheel under cuda13). Fall back to
        # standard flash_attn if flashinfer fails to load.
        try:
            kernel = runner._build_attn_fn(
                query, key, value, n_heads, n_kv_heads, head_dim, use_flashinfer=True
            )
        except Exception:
            kernel = runner._build_attn_fn(
                query, key, value, n_heads, n_kv_heads, head_dim, use_flashinfer=False
            )

    elif args.layer_type == "chunked_ssm":
        # Profile the SSD scan kernel for one prefill chunk (prefill_chunk_tokens tokens).
        # This captures the per-call grid size used in chunked-prefill SSM, which is
        # smaller than the full-sequence grid and avoids cooperative barrier deadlock.
        import math
        from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
        from shared.loaders import get_model_config

        _raw = get_model_config(args.model)
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
