"""
Subprocess worker: chunked-prefill attention sweep for a SINGLE SM count.

Mirrors _ssm_worker.py — contains CUDA-context corruption within one process so a
failure at one SM level cannot poison the others.  Emits a JSON array of raw
measurement dicts on stdout; per-config status lines on stderr.

Invoked by run_attn_chunked_sweep.py (one subprocess per sm_count).
"""

import sys
import os
import json
import argparse

_here = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_here, "../.."))  # workspace/
sys.path.insert(0, os.path.join(_here, ".."))     # characterization/

import torch


def main() -> None:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--model",        required=True)
    p.add_argument("--sm-count",     type=int, required=True)
    p.add_argument("--total-sm",     type=int, required=True)
    p.add_argument("--seq-lens",     nargs="+", type=int, required=True)
    p.add_argument("--batch-sizes",  nargs="+", type=int, required=True)
    p.add_argument("--chunk-tokens", type=int, required=True)
    p.add_argument("--n-warmup",     type=int, default=3)
    p.add_argument("--n-measure",    type=int, default=10)
    args = p.parse_args()

    from src.smctrl.green_ctx_controller import SMController
    from stage1_sm_scaling.chunked_attn_runner import run_chunked_attn_sweep

    smctrl = SMController(
        total_sm_count=args.total_sm,
        preset_sm_counts=sorted({args.sm_count, args.total_sm}),
    )

    results = []
    cuda_dead = False

    for seq_len in args.seq_lens:
        for batch_size in args.batch_sizes:
            if cuda_dead:
                print(f"  SKIP sm={args.sm_count} seq={seq_len} bs={batch_size}: prior CUDA error",
                      file=sys.stderr)
                continue
            try:
                row = run_chunked_attn_sweep(
                    model_name=args.model,
                    seq_len=seq_len,
                    batch_size=batch_size,
                    prefill_chunk_tokens=args.chunk_tokens,
                    sm_count=args.sm_count,
                    smctrl=smctrl,
                    n_warmup=args.n_warmup,
                    n_measure=args.n_measure,
                )
                results.append(row)
                print(f"  OK  sm={args.sm_count:3d} seq={seq_len:6d} bs={batch_size:3d} "
                      f"lat={row['latency_ms']:.3f}ms calls={row['n_kernel_calls']}",
                      file=sys.stderr)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"  OOM sm={args.sm_count:3d} seq={seq_len:6d} bs={batch_size:3d}: skipped",
                      file=sys.stderr)
            except Exception as e:
                err = str(e)
                print(f"  ERR sm={args.sm_count:3d} seq={seq_len:6d} bs={batch_size:3d}: {err[:120]}",
                      file=sys.stderr)
                if ("CUDA error" in err or "illegal memory" in err.lower()
                        or "device-side assert" in err.lower()):
                    cuda_dead = True
                    print(f"  CUDA context corrupted at sm={args.sm_count}; skipping rest.",
                          file=sys.stderr)

    print(json.dumps(results))


if __name__ == "__main__":
    main()
