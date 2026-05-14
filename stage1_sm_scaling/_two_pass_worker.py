"""
Subprocess worker: runs two-pass SSM sweep for a single SM count.

Called by run_ssm_two_pass_sweep.py to contain CUDA context state within one
process per SM level.  Each process creates a TwoPassLayerRunner, runs all
(seq_len, batch_size) combos at the given sm_count, and writes a JSON array
to stdout.  Errors are printed to stderr.

stdout: JSON array of result dicts (may be empty on total failure).
stderr: per-config status lines visible in parent log.
"""

import sys
import os
import json
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch


def main() -> None:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--model",       required=True)
    p.add_argument("--sm-count",    type=int, required=True)
    p.add_argument("--seq-lens",    nargs="+", type=int, required=True)
    p.add_argument("--batch-sizes", nargs="+", type=int, required=True)
    p.add_argument("--total-sm",    type=int, required=True)
    p.add_argument("--bw-gbs",      type=float, default=None)
    p.add_argument("--n-warmup",    type=int, default=2)
    p.add_argument("--n-measure",   type=int, default=5)
    args = p.parse_args()

    from src.models.zamba2_two_pass import TwoPassLayerRunner

    runner = TwoPassLayerRunner(
        device="cuda",
        total_sm_count=args.total_sm,
        theoretical_bw_GBs=args.bw_gbs,
    )

    results = []
    cuda_dead = False

    for seq_len in args.seq_lens:
        for batch_size in args.batch_sizes:
            if cuda_dead:
                print(
                    f"  SKIP sm={args.sm_count} seq={seq_len} bs={batch_size}: "
                    "prior CUDA context error",
                    file=sys.stderr,
                )
                continue
            try:
                row = runner.run_ssm_layer(
                    model_name=args.model,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    sm_count=args.sm_count,
                    n_warmup=args.n_warmup,
                    n_measure=args.n_measure,
                )
                results.append(row)
                print(
                    f"  OK  sm={args.sm_count:3d} seq={seq_len:6d} bs={batch_size:3d} "
                    f"lat={row['latency_ms']:.3f}ms  "
                    f"bw={row['achieved_bandwidth_GBs']:.1f}/{row['theoretical_bw_GBs']:.0f}GB/s",
                    file=sys.stderr,
                )
            except torch.cuda.OutOfMemoryError:
                print(
                    f"  OOM sm={args.sm_count:3d} seq={seq_len:6d} bs={batch_size:3d}: "
                    "skipping",
                    file=sys.stderr,
                )
                torch.cuda.empty_cache()
            except Exception as e:
                err_str = str(e)
                print(
                    f"  ERR sm={args.sm_count:3d} seq={seq_len:6d} bs={batch_size:3d}: "
                    f"{err_str[:120]}",
                    file=sys.stderr,
                )
                is_cuda_error = (
                    "CUDA error" in err_str
                    or "illegal memory" in err_str.lower()
                    or "cudaErrorIllegalAddress" in err_str
                    or "device-side assert" in err_str.lower()
                )
                if is_cuda_error:
                    print(
                        f"  CUDA context corrupted at sm={args.sm_count} "
                        f"seq={seq_len} bs={batch_size}. "
                        "Skipping remaining configs for this SM level.",
                        file=sys.stderr,
                    )
                    cuda_dead = True

    print(json.dumps(results))


if __name__ == "__main__":
    main()
