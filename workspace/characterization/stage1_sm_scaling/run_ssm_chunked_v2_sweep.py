"""
run_ssm_chunked_v2_sweep.py — SSM chunked sweep on the unified spec grid/schema.

Re-runs the EXISTING chunked SSM measurement core (chunked_ssm_runner.run_chunked_ssm_sweep
— unchanged) but driven entirely by shared.sweep_spec, at the single serving
prefill_chunk_tokens, and emitting the unified chunked schema (sweep_spec.CHUNKED_FIELDNAMES)
with the canonical filename.  Output is column-identical to attn_chunked (Phase 2
checklist: schema diff 0), enabling analyze_matched_granularity.py.

  python stage1_sm_scaling/run_ssm_chunked_v2_sweep.py --model zamba2
  → results/stage1_v2/ssm_chunked_zamba2_a100_sxm4_80gb.csv

(Existing results/stage1/chunked/*.csv is left untouched.)
"""

import sys
import os
_here = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_here, "../.."))  # workspace/
sys.path.insert(0, os.path.join(_here, ".."))     # characterization/

import argparse
import csv
from pathlib import Path

from shared import sweep_spec as spec
from src.smctrl.green_ctx_controller import SMController
from stage1_sm_scaling.chunked_ssm_runner import (
    run_chunked_ssm_sweep,
    _check_initial_states_support,
)


def parse_args():
    p = argparse.ArgumentParser(description="SSM chunked sweep on the unified spec grid.")
    p.add_argument("--model", default="zamba2",
                   choices=["zamba2", "falcon_h1", "nemotron_h"])
    p.add_argument("--n-warmup",  type=int, default=3)
    p.add_argument("--n-measure", type=int, default=10)
    p.add_argument("--skip-verify", action="store_true")
    p.add_argument("--output-dir", type=Path,
                   default=Path(__file__).parent.parent / "results" / "stage1_v2")
    return p.parse_args()


def main():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required")

    args = parse_args()
    sm_grid = spec.sm_grid()
    spec.assert_on_sm_grid(sm_grid)
    seq_lens = spec.seq_grid_prefill()
    batch_sizes = spec.batch_grid()
    chunk = spec.prefill_chunk_tokens()
    total_sm = spec.total_sm()

    print("=== SSM Chunked Prefill SM Sweep (unified spec grid) ===")
    print(f"  model        : {args.model}")
    print(f"  device       : {spec.device_name()}  ({total_sm} SM)")
    print(f"  chunk tokens : {chunk}")
    print(f"  sm_grid      : {sm_grid}")
    print(f"  seq_lens     : {seq_lens}")
    print(f"  batch_sizes  : {batch_sizes}\n")

    smctrl = SMController(total_sm_count=total_sm, preset_sm_counts=sm_grid)
    if not args.skip_verify and not smctrl.verify_sm_control(verbose=True):
        raise RuntimeError("SMController verification failed (Green Context not limiting SM).")

    state_active = _check_initial_states_support()

    rows: list[dict] = []
    for sm_count in sm_grid:
        for seq_len in seq_lens:
            for bs in batch_sizes:
                try:
                    res = run_chunked_ssm_sweep(
                        model_name=args.model,
                        seq_len=seq_len,
                        batch_size=bs,
                        prefill_chunk_tokens=chunk,
                        sm_count=sm_count,
                        smctrl=smctrl,
                        n_warmup=args.n_warmup,
                        n_measure=args.n_measure,
                    )
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    print(f"  OOM sm={sm_count} seq={seq_len} bs={bs}", flush=True)
                    continue
                except Exception as e:
                    print(f"  ERR sm={sm_count} seq={seq_len} bs={bs}: {str(e)[:120]}", flush=True)
                    continue

                n_calls = res["n_kernel_calls"]
                rows.append(spec.chunked_row(
                    model=args.model,
                    layer_type="ssm_chunked",
                    seq_len=seq_len,
                    batch_size=bs,
                    sm_count=sm_count,
                    prefill_chunk_tokens=chunk,
                    latency_ms=res["latency_ms"],
                    latency_std_ms=res["latency_std_ms"],
                    # total HBM bytes over the sequence = per-call × n_calls
                    read_bytes=res["read_bytes_per_call"] * n_calls,
                    write_bytes=res["write_bytes_per_call"] * n_calls,
                    cooperative_safe=res["cooperative_safe"],
                    state_passing_active=state_active,
                    status="ok",
                ))
                print(f"  OK  sm={sm_count:3d} seq={seq_len:6d} bs={bs:3d} "
                      f"lat={res['latency_ms']:.3f}ms calls={n_calls}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / spec.canonical_filename("ssm_chunked", args.model)
    rows.sort(key=lambda r: (r["seq_len"], r["batch_size"], r["sm_count"]))
    with open(out_path, "w", newline="") as f:
        f.write(spec.CHUNKED_VALUE_KIND_COMMENT + "\n")
        w = csv.DictWriter(f, fieldnames=spec.CHUNKED_FIELDNAMES,
                           extrasaction="ignore", restval="")
        w.writeheader()
        w.writerows(rows)

    print(f"\nDone. {len(rows)} rows → {out_path}")


if __name__ == "__main__":
    main()
