"""
Subprocess worker for attention/MLP SM sweep at one SM count.

Two roles:
  Worker  (run as __main__): measures one SM level, writes JSON lines to stdout
  Helper  (imported by parent): run_sm_level_subprocess() manages worker processes

CUDA context isolation strategy
--------------------------------
A single CUDA OOM or kernel error inside a Green Context can corrupt the CUDA
context, making every subsequent CUDA call in the same process fail. Running
each SM level in its own subprocess guarantees a fresh context per level.

If a subprocess exits mid-run (crash or timeout), the parent collects whatever
JSON lines were emitted before the crash, identifies the unreported configs,
and retries them once in a new subprocess (fresh CUDA context).

stdout: one JSON object per measured config, flushed immediately.
stderr: human-readable status lines (shown in parent log).
"""

import sys
import os
import json
import argparse
import subprocess
import time
from itertools import product
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch


# ---------------------------------------------------------------------------
# Parent-side helper
# ---------------------------------------------------------------------------

def run_sm_level_subprocess(
    layer_type: str,
    model_name: str,
    sm_count: int,
    seq_lens: list[int],
    batch_sizes: list[int],
    total_sm: int,
    bw_gbs: Optional[float],
    n_warmup: int,
    n_measure: int,
    context_len: int = 0,
    timeout: int = 3600,
    max_retries: int = 1,
) -> list[dict]:
    """Run sweep for one SM level in an isolated subprocess.

    Retries once (in a fresh subprocess) for configs not reached due to CUDA
    context corruption. OOM configs are not retried (they will still OOM).

    Returns a flat list of result dicts. Error rows include an 'error' key.
    """
    remaining_sl = list(seq_lens)
    remaining_bs = list(batch_sizes)
    all_results: list[dict] = []

    for attempt in range(max_retries + 1):
        if not remaining_sl or not remaining_bs:
            break

        completed, cuda_dead, oom_configs = _run_subprocess_once(
            layer_type=layer_type,
            model_name=model_name,
            sm_count=sm_count,
            seq_lens=remaining_sl,
            batch_sizes=remaining_bs,
            total_sm=total_sm,
            bw_gbs=bw_gbs,
            n_warmup=n_warmup,
            n_measure=n_measure,
            context_len=context_len,
            timeout=timeout,
        )

        all_results.extend(completed)

        for sl, bs in oom_configs:
            all_results.append({
                "sm_count": sm_count,
                "sm_ratio": sm_count / total_sm,
                "seq_len": sl,
                "batch_size": bs,
                "model_name": model_name,
                "layer_type": layer_type,
                "error": "OOM",
            })

        if cuda_dead and attempt < max_retries:
            print(
                f"  [retry {attempt + 1}/{max_retries}] sm={sm_count}: "
                f"{len(cuda_dead)} config(s) unreached (CUDA context dead); "
                f"retrying in fresh subprocess …",
                flush=True,
            )
            time.sleep(2)
            remaining_sl = sorted({sl for sl, _ in cuda_dead})
            remaining_bs = sorted({bs for _, bs in cuda_dead})
        else:
            for sl, bs in cuda_dead:
                all_results.append({
                    "sm_count": sm_count,
                    "sm_ratio": sm_count / total_sm,
                    "seq_len": sl,
                    "batch_size": bs,
                    "model_name": model_name,
                    "layer_type": layer_type,
                    "error": "CUDA_CONTEXT_ERROR",
                })
            break

    return all_results


def _run_subprocess_once(
    layer_type: str,
    model_name: str,
    sm_count: int,
    seq_lens: list[int],
    batch_sizes: list[int],
    total_sm: int,
    bw_gbs: Optional[float],
    n_warmup: int,
    n_measure: int,
    context_len: int,
    timeout: int,
) -> tuple[list[dict], list[tuple], list[tuple]]:
    """Spawn one worker subprocess; return (completed, cuda_dead_pairs, oom_pairs)."""
    worker = Path(__file__)
    cmd = [
        sys.executable, str(worker),
        "--layer-type", layer_type,
        "--model", model_name,
        "--sm-count", str(sm_count),
        "--seq-lens", *[str(s) for s in seq_lens],
        "--batch-sizes", *[str(b) for b in batch_sizes],
        "--total-sm", str(total_sm),
        "--n-warmup", str(n_warmup),
        "--n-measure", str(n_measure),
        "--context-len", str(context_len),
    ]
    if bw_gbs is not None:
        cmd += ["--bw-gbs", str(bw_gbs)]

    completed: list[dict] = []
    oom_pairs: list[tuple] = []
    cuda_dead_pairs: list[tuple] = []
    seen: set[tuple] = set()

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            text=True,
            bufsize=1,
        )

        try:
            for raw in proc.stdout:
                line = raw.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                key = (rec.get("seq_len"), rec.get("batch_size"))
                seen.add(key)
                error = rec.get("error", "")

                if not error:
                    completed.append(rec)
                elif error == "OOM":
                    oom_pairs.append(key)
                else:
                    cuda_dead_pairs.append(key)

            proc.wait(timeout=timeout)

        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            print(
                f"  [warn] sm={sm_count} subprocess timed out after {timeout}s — partial results collected",
                flush=True,
            )

        # configs the subprocess never reported (crash before reaching them)
        all_expected = set(product(seq_lens, batch_sizes))
        unseen = all_expected - seen
        if unseen:
            n = len(unseen)
            print(
                f"  [warn] sm={sm_count}: {n} config(s) not reported "
                f"(subprocess exit={proc.returncode}); marking for retry",
                flush=True,
            )
            cuda_dead_pairs.extend(unseen)

    except Exception as e:
        print(f"  [error] sm={sm_count} subprocess failed to launch: {e}", flush=True)

    return completed, cuda_dead_pairs, oom_pairs


# ---------------------------------------------------------------------------
# Subprocess worker entry point
# ---------------------------------------------------------------------------

def _worker_main() -> None:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--layer-type", choices=["attn", "mlp"], required=True)
    p.add_argument("--model",        required=True)
    p.add_argument("--sm-count",     type=int,   required=True)
    p.add_argument("--seq-lens",     nargs="+",  type=int, required=True)
    p.add_argument("--batch-sizes",  nargs="+",  type=int, required=True)
    p.add_argument("--total-sm",     type=int,   required=True)
    p.add_argument("--bw-gbs",       type=float, default=None)
    p.add_argument("--n-warmup",     type=int,   default=20)
    p.add_argument("--n-measure",    type=int,   default=50)
    p.add_argument("--context-len",  type=int,   default=0)
    args = p.parse_args()

    from src.models.layer_runner import LayerRunner

    runner = LayerRunner(
        device="cuda",
        total_sm_count=args.total_sm,
        theoretical_bw_GBs=args.bw_gbs,
    )

    cuda_dead = False

    for seq_len in args.seq_lens:
        for batch_size in args.batch_sizes:
            if cuda_dead:
                rec = {
                    "sm_count":   args.sm_count,
                    "seq_len":    seq_len,
                    "batch_size": batch_size,
                    "error":      "CUDA_DEAD",
                }
                print(json.dumps(rec), flush=True)
                print(
                    f"  SKIP sm={args.sm_count:3d} seq={seq_len:6d} bs={batch_size:3d}: "
                    f"CUDA context dead",
                    file=sys.stderr,
                )
                continue

            try:
                if args.layer_type == "attn":
                    row = runner.run_attn_layer(
                        model_name=args.model,
                        batch_size=batch_size,
                        seq_len=seq_len,
                        sm_count=args.sm_count,
                        context_len=args.context_len,
                        n_warmup=args.n_warmup,
                        n_measure=args.n_measure,
                    )
                else:
                    row = runner.run_mlp_layer(
                        model_name=args.model,
                        batch_size=batch_size,
                        seq_len=seq_len,
                        sm_count=args.sm_count,
                        n_warmup=args.n_warmup,
                        n_measure=args.n_measure,
                    )

                print(json.dumps(row), flush=True)
                print(
                    f"  OK  sm={args.sm_count:3d} seq={seq_len:6d} bs={batch_size:3d}  "
                    f"lat={row['latency_ms']:.3f}ms  "
                    f"bw={row['achieved_bandwidth_GBs']:.1f}/{row['theoretical_bw_GBs']:.0f}GB/s",
                    file=sys.stderr,
                )

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                rec = {
                    "sm_count":   args.sm_count,
                    "seq_len":    seq_len,
                    "batch_size": batch_size,
                    "error":      "OOM",
                }
                print(json.dumps(rec), flush=True)
                print(
                    f"  OOM sm={args.sm_count:3d} seq={seq_len:6d} bs={batch_size:3d}: "
                    f"out of memory — skipped",
                    file=sys.stderr,
                )

            except Exception as e:
                err_str = str(e)
                is_cuda_error = (
                    "CUDA error" in err_str
                    or "illegal memory" in err_str.lower()
                    or "cudaErrorIllegalAddress" in err_str
                    or "device-side assert" in err_str.lower()
                )
                error_tag = "CUDA_DEAD" if is_cuda_error else err_str[:120]
                rec = {
                    "sm_count":   args.sm_count,
                    "seq_len":    seq_len,
                    "batch_size": batch_size,
                    "error":      error_tag,
                }
                print(json.dumps(rec), flush=True)
                print(
                    f"  ERR sm={args.sm_count:3d} seq={seq_len:6d} bs={batch_size:3d}: {err_str[:120]}",
                    file=sys.stderr,
                )
                if is_cuda_error:
                    print(
                        f"  CUDA context corrupted at sm={args.sm_count} seq={seq_len} bs={batch_size}. "
                        f"Marking remaining configs as CUDA_DEAD.",
                        file=sys.stderr,
                    )
                    cuda_dead = True


if __name__ == "__main__":
    _worker_main()
