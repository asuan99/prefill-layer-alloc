"""
run_attn_chunked_sweep.py — chunked-prefill Attention SM sweep (Phase 2).

Granularity-matched attention measurement: the sequence is processed in
prefill_chunk_tokens-sized chunks against a cumulative KV cache, so the latency
definition matches the SSM chunked path (chunked_ssm_runner.py).  See
chunked_attn_runner.py for the measurement core and reports/sweep_consistency_audit.md
(item 3) for the motivation.

All grids/constants come from shared.sweep_spec (single source).  One subprocess
per SM count (CUDA-context isolation, _attn_chunked_worker.py).  Output uses the
unified chunked schema (sweep_spec.CHUNKED_FIELDNAMES) and canonical filename, so
it is column-identical to the ssm_chunked v2 CSV.

  python stage1_sm_scaling/run_attn_chunked_sweep.py --model zamba2
  → results/stage1_v2/attn_chunked_zamba2_a100_sxm4_80gb.csv
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

from shared import sweep_spec as spec

_WORKER = Path(__file__).parent / "_attn_chunked_worker.py"


def parse_args():
    p = argparse.ArgumentParser(
        description="Chunked-prefill attention SM sweep (granularity-matched to SSM).",
    )
    p.add_argument("--model", default="zamba2",
                   choices=["zamba2", "falcon_h1", "nemotron_h"])
    p.add_argument("--n-warmup",  type=int, default=3)
    p.add_argument("--n-measure", type=int, default=10)
    p.add_argument("--timeout",   type=int, default=1800,
                   help="per-SM subprocess timeout (s)")
    p.add_argument("--output-dir", type=Path,
                   default=Path(__file__).parent.parent / "results" / "stage1_v2",
                   help="CSV output dir (new tree — existing results untouched)")
    return p.parse_args()


def run_sm_level(model, sm_count, seq_lens, batch_sizes, chunk, n_warmup,
                 n_measure, timeout) -> list[dict]:
    """Spawn one worker for sm_count; return its list of raw measurement dicts."""
    cmd = [
        sys.executable, str(_WORKER),
        "--model", model,
        "--sm-count", str(sm_count),
        "--total-sm", str(spec.total_sm()),
        "--seq-lens", *[str(s) for s in seq_lens],
        "--batch-sizes", *[str(b) for b in batch_sizes],
        "--chunk-tokens", str(chunk),
        "--n-warmup", str(n_warmup),
        "--n-measure", str(n_measure),
    ]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=sys.stderr,
                              text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"  [warn] sm={sm_count} timed out after {timeout}s", flush=True)
        return []
    out = proc.stdout.strip().splitlines()
    if not out:
        return []
    try:
        return json.loads(out[-1])  # worker prints the JSON array last
    except json.JSONDecodeError:
        print(f"  [warn] sm={sm_count} produced no parseable JSON", flush=True)
        return []


def _assert_measurement_device() -> None:
    """Abort if the live GPU is not the spec's canonical device (no device mixing)."""
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("Device guard: CUDA not available.")
    live = spec.canonical_device(torch.cuda.get_device_name(0))
    if live != spec.device_name():
        raise SystemExit(
            f"Device guard: live GPU normalises to {live!r} but sweep_spec expects "
            f"{spec.device_name()!r}. Refusing to write mismatched-device CSVs."
        )


def main():
    args = parse_args()
    _assert_measurement_device()

    sm_grid = spec.sm_grid()
    spec.assert_on_sm_grid(sm_grid)          # fail loudly if grid drifts
    seq_lens = spec.seq_grid_prefill()
    batch_sizes = spec.batch_grid()
    chunk = spec.prefill_chunk_tokens()

    print("=== Chunked Attention Prefill SM Sweep (granularity-matched) ===")
    print(f"  model        : {args.model}")
    print(f"  device       : {spec.device_name()}  ({spec.total_sm()} SM)")
    print(f"  chunk tokens : {chunk}")
    print(f"  sm_grid      : {sm_grid}")
    print(f"  seq_lens     : {seq_lens}")
    print(f"  batch_sizes  : {batch_sizes}")
    print()

    rows: list[dict] = []
    for sm_count in sm_grid:
        print(f"[sm={sm_count}] launching worker …", flush=True)
        raw_rows = run_sm_level(
            args.model, sm_count, seq_lens, batch_sizes, chunk,
            args.n_warmup, args.n_measure, args.timeout,
        )
        for r in raw_rows:
            rows.append(spec.chunked_row(
                model=args.model,
                layer_type="attn_chunked",
                seq_len=r["seq_len"],
                batch_size=r["batch_size"],
                sm_count=r["sm_count"],
                prefill_chunk_tokens=r["prefill_chunk_tokens"],
                latency_ms=r["latency_ms"],
                latency_std_ms=r["latency_std_ms"],
                read_bytes=r["read_bytes"],
                write_bytes=r["write_bytes"],
                cooperative_safe=r["cooperative_safe"],
                state_passing_active=r["state_passing_active"],
                status="ok",
            ))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / spec.canonical_filename("attn_chunked", args.model)
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
