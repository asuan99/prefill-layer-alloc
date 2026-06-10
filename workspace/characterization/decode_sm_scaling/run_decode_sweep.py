"""
run_decode_sweep.py — decode SM-saturation sweep driver (Task 1).

Spawns one isolated subprocess per SM level (_decode_worker.py).  A CUDA error in
one level cannot corrupt another — each level gets a fresh CUDA context.  On a
context-corrupting error mid-level, the level is retried once in a new subprocess.

Sweep grid (per task spec)
--------------------------
    sm_count : [14, 27, 40, 54, 68, 81, 94, 108]   # Green Context presets
    batch    : [1, 4, 8, 16, 32]
    context L: [512, 2048, 8192, 32768]            # decode_attn only;
                                                     decode_ssm → L=0 single row
    model    : zamba2 | falcon_h1
    cells    : decode_ssm, decode_attn
    scopes   : kernel, layer

Output CSV (one per cell): results/decode_scaling/decode_{cell}_{model}_{device}.csv
A leading '# value_kind: ...' comment documents measured vs derived columns.
Read back with pandas.read_csv(..., comment='#').

Pre-flight: SMController.verify_sm_control() MUST pass before the sweep starts.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from itertools import product
from pathlib import Path

_here = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_here, ".."))       # src.*
sys.path.insert(0, os.path.join(_here, "..", ".."))  # shared.*

import torch

from src.smctrl.green_ctx_controller import SMController

# ---------------------------------------------------------------------------
# Grid defaults
# ---------------------------------------------------------------------------

SM_PRESETS = [14, 27, 40, 54, 68, 81, 94, 108]
BATCH_GRID = [1, 4, 8, 16, 32]
CONTEXT_GRID = [512, 2048, 8192, 32768]
CELLS = ["decode_ssm", "decode_attn"]
SCOPES = ["kernel", "layer"]
TOTAL_SM = 108
DEVICE_TAG = "a100_sxm4_80gb"

_WORKER = Path(__file__).parent / "_decode_worker.py"

# status priority for dedup (higher = keep)
_STATUS_RANK = {"ok": 3, "OOM": 2, "ERROR": 1, "CUDA_ERROR": 0}

CSV_FIELDS = [
    "model", "cell", "scope", "sm_count", "batch", "context_len",
    "latency_per_step_ms", "bytes_analytic", "bytes_analytic_w_weight",
    "achieved_bw_GBs", "bw_util_pct", "backend", "status",
]

# value_kind header comment (column → measured/derived)
_VALUE_KIND_COMMENT = (
    "# value_kind: latency_per_step_ms=measured(CUDA event); "
    "bytes_analytic,bytes_analytic_w_weight,achieved_bw_GBs,bw_util_pct=derived; "
    "model,cell,scope,sm_count,batch,context_len,backend,status=metadata"
)


# ---------------------------------------------------------------------------
# Pre-flight SM-control verification
# ---------------------------------------------------------------------------


def preflight_verify() -> bool:
    """Run SMController.verify_sm_control() in this process.  Returns pass/fail."""
    smctrl = SMController(
        device_id=0,
        total_sm_count=TOTAL_SM,
        preset_sm_counts=SM_PRESETS,
    )
    print("=== Pre-flight: SMController.verify_sm_control() ===")
    ok = smctrl.verify_sm_control(verbose=True)
    return ok


# ---------------------------------------------------------------------------
# Expected-key bookkeeping
# ---------------------------------------------------------------------------


def expected_keys(cells, scopes, batches, contexts) -> set[tuple]:
    keys = set()
    for cell in cells:
        ctx_grid = [0] if cell == "decode_ssm" else list(contexts)
        for scope in scopes:
            for bs in batches:
                for L in ctx_grid:
                    keys.add((cell, scope, bs, L))
    return keys


def _rec_key(rec: dict) -> tuple:
    return (rec["cell"], rec["scope"], rec["batch"], rec["context_len"])


# ---------------------------------------------------------------------------
# One subprocess run for a single SM level
# ---------------------------------------------------------------------------


def run_level_once(
    model: str, sm_count: int, cells, scopes, batches, contexts,
    K: int, warmup: int, repeats: int, timeout: int,
) -> dict[tuple, dict]:
    """Spawn one worker subprocess for sm_count; return {key: rec} for emitted rows."""
    cmd = [
        sys.executable, str(_WORKER),
        "--model", model,
        "--sm-count", str(sm_count),
        "--total-sm", str(TOTAL_SM),
        "--cells", *cells,
        "--scopes", *scopes,
        "--batches", *[str(b) for b in batches],
        "--contexts", *[str(c) for c in contexts],
        "--K", str(K), "--warmup", str(warmup), "--repeats", str(repeats),
    ]
    out: dict[tuple, dict] = {}
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=sys.stderr, text=True, bufsize=1,
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
                out[_rec_key(rec)] = rec
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            print(f"  [warn] sm={sm_count} subprocess timed out after {timeout}s",
                  flush=True)
    except Exception as e:
        print(f"  [error] sm={sm_count} subprocess launch failed: {e}", flush=True)
    return out


def run_level(
    model: str, sm_count: int, cells, scopes, batches, contexts,
    K: int, warmup: int, repeats: int, timeout: int, max_retries: int = 1,
) -> list[dict]:
    """Run one SM level with up to `max_retries` fresh-context retries.

    Merges results across attempts (status priority ok>OOM>ERROR>CUDA_ERROR).
    Unreached configs are recorded as CUDA_ERROR.
    """
    want = expected_keys(cells, scopes, batches, contexts)
    merged: dict[tuple, dict] = {}

    for attempt in range(max_retries + 1):
        got = run_level_once(
            model, sm_count, cells, scopes, batches, contexts,
            K, warmup, repeats, timeout,
        )
        for key, rec in got.items():
            old = merged.get(key)
            if old is None or _STATUS_RANK.get(rec.get("status"), -1) > \
                    _STATUS_RANK.get(old.get("status"), -1):
                merged[key] = rec

        unreached = want - {k for k, v in merged.items()
                            if v.get("status") in ("ok", "OOM")}
        cuda_err = any(merged.get(k, {}).get("status") == "CUDA_ERROR" or k not in merged
                       for k in unreached)
        if not unreached or not cuda_err or attempt >= max_retries:
            break
        print(f"  [retry {attempt+1}/{max_retries}] sm={sm_count}: "
              f"{len(unreached)} config(s) unreached — fresh subprocess …", flush=True)
        time.sleep(2)

    # fill missing keys as CUDA_ERROR
    for key in want:
        if key not in merged:
            cell, scope, bs, L = key
            merged[key] = {
                "model": model, "cell": cell, "scope": scope,
                "sm_count": sm_count, "batch": bs, "context_len": L,
                "status": "CUDA_ERROR", "note": "not reported by subprocess",
            }
    return list(merged.values())


# ---------------------------------------------------------------------------
# CSV writing (one file per cell)
# ---------------------------------------------------------------------------


def write_csvs(rows: list[dict], model: str, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for cell in sorted({r["cell"] for r in rows}):
        cell_rows = [r for r in rows if r["cell"] == cell]
        cell_rows.sort(key=lambda r: (r["scope"], r["sm_count"], r["batch"], r["context_len"]))
        path = out_dir / f"decode_{cell}_{model}_{DEVICE_TAG}.csv"
        with open(path, "w", newline="") as f:
            f.write(_VALUE_KIND_COMMENT + "\n")
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore", restval="")
            w.writeheader()
            w.writerows(cell_rows)
        n_ok = sum(1 for r in cell_rows if r.get("status") == "ok")
        print(f"  Saved {path}  ({len(cell_rows)} rows, {n_ok} ok)")
        paths.append(path)
    return paths


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------


def run_sweep(args) -> None:
    out_dir = args.output_dir
    print(f"\n=== Decode SM-Saturation Sweep: {args.model} ===")
    print(f"  SM presets : {args.sm_counts}")
    print(f"  batches    : {args.batches}")
    print(f"  contexts   : {args.contexts}  (decode_attn only)")
    print(f"  cells      : {args.cells}")
    print(f"  scopes     : {args.scopes}")
    print(f"  per-step   : K={args.K}, warmup={args.warmup}, repeats={args.repeats}")
    print(f"  out_dir    : {out_dir}\n")

    all_rows: list[dict] = []
    for sm_count in args.sm_counts:
        print(f"── SM level {sm_count} ──", flush=True)
        rows = run_level(
            args.model, sm_count, args.cells, args.scopes,
            args.batches, args.contexts,
            args.K, args.warmup, args.repeats, args.timeout,
        )
        all_rows.extend(rows)

    write_csvs(all_rows, args.model, out_dir)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=["zamba2", "falcon_h1"])
    p.add_argument("--sm-counts", nargs="+", type=int, default=SM_PRESETS, dest="sm_counts")
    p.add_argument("--batches", nargs="+", type=int, default=BATCH_GRID)
    p.add_argument("--contexts", nargs="+", type=int, default=CONTEXT_GRID)
    p.add_argument("--cells", nargs="+", default=CELLS, choices=CELLS)
    p.add_argument("--scopes", nargs="+", default=SCOPES, choices=SCOPES)
    p.add_argument("--K", type=int, default=64)
    p.add_argument("--warmup", type=int, default=16)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--timeout", type=int, default=3600)
    p.add_argument("--output-dir", type=Path,
                   default=Path(__file__).parent.parent / "results" / "decode_scaling")
    p.add_argument("--skip-verify", action="store_true",
                   help="Skip the SMController.verify_sm_control() pre-flight (NOT recommended)")
    return p.parse_args()


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required for decode SM sweep")
    args = parse_args()

    if not args.skip_verify:
        if not preflight_verify():
            print("\nERROR: SM control verification FAILED — aborting sweep.\n"
                  "  (use --skip-verify to override; results would not show SM scaling)")
            sys.exit(2)
        print("  verify_sm_control: PASS ✓\n")

    run_sweep(args)


if __name__ == "__main__":
    main()
