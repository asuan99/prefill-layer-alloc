"""
run_decode_sm_sweep.py — decode SM-sensitivity sweep (S1.3).

Measures autoregressive decode latency (seq_len=1) for SSM and Attention
layers under Green Context SM restriction.  Quantifies how sensitive decode
latency is to SM count: if latency barely changes as SMs are reduced, those
SMs are genuinely "free" during prefill.

This script is the G1 gate input: only proceed with SMController if
  (a) prefill has free SMs (free_sm_fraction > threshold), AND
  (b) decode IS sensitive to SM count (decode_sm_sensitivity > threshold).

If decode is insensitive, reducing decode SMs doesn't degrade quality;
if it IS sensitive, giving extra SMs to decode speeds it up.

Outputs:
  results/stage1/decode/decode_sm_{model}_{device}.csv
    Columns: layer_type, context_len, batch_size, sm_count, sm_ratio,
             latency_ms, latency_p99_ms, bw_utilization_pct,
             decode_sm_sensitivity

  results/stage1/decode/g1_gate_{model}_{device}.csv
    Combined table: model_name, seq_len (prefill), batch_size,
    free_sm_fraction, decode_sm_sensitivity, g1_potential
    (g1_potential = free_sm_fraction × decode_sm_sensitivity)
    This is the G1 gate input for deciding whether SMController is worthwhile.

Usage:
    # SSM + Attn decode sweep:
    python stage1_sm_scaling/run_decode_sm_sweep.py \\
        --model zamba2 --device auto

    # Only SSM decode (faster):
    python stage1_sm_scaling/run_decode_sm_sweep.py \\
        --model zamba2 --layer-types ssm

    # Custom SM counts and context lengths:
    python stage1_sm_scaling/run_decode_sm_sweep.py \\
        --model zamba2 --device a100-sxm4-80gb \\
        --sm-counts 14 27 54 108 \\
        --context-lens 512 2048 8192 \\
        --batch-sizes 1 4

Hardware guard: A100 required for numeric G1-gate conclusions.
"""

from __future__ import annotations

import sys
import os
_here = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_here, "../.."))  # workspace/
sys.path.insert(0, os.path.join(_here, ".."))     # characterization/

import argparse
import csv
import time
from itertools import product
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm

from shared.loaders import get_hardware_config, device_tag
from src.smctrl.green_ctx_controller import SMController


# ---------------------------------------------------------------------------
# Hardware guard
# ---------------------------------------------------------------------------

_A100_KW = frozenset({"a100", "a800"})


def _is_a100() -> bool:
    return torch.cuda.is_available() and any(
        kw in torch.cuda.get_device_name(0).lower() for kw in _A100_KW
    )


def hardware_warn() -> None:
    if not _is_a100():
        print("=" * 68)
        print("  WARNING: Non-A100 GPU detected.")
        print("  G1-gate conclusions from this run are NOT valid for the paper.")
        print("  Re-run on A100 for numeric results.")
        print("=" * 68)


# ---------------------------------------------------------------------------
# SSM decode (seq_len = 1 via chunked runner)
# ---------------------------------------------------------------------------

def _measure_ssm_decode(
    model_name: str,
    context_len: int,
    batch_size: int,
    sm_count: int,
    smctrl: SMController,
    n_warmup: int = 10,
    n_measure: int = 30,
) -> dict:
    """Measure SSM decode latency at seq_len=1 via chunked_ssm_runner.

    seq_len=1 means one new token.  For Mamba-2, this is the recurrent-mode
    step: one call to mamba_chunk_scan_combined with chunk_size covering the
    full context (or a single-chunk update for autoregressive decode).

    We use the chunked prefill path with seq_len=1 as a proxy for decode
    latency.  True SSM decode (state-space update) would use selective_scan
    step(), but the chunked path at seq_len=1 gives comparable timing.
    """
    from stage1_sm_scaling.chunked_ssm_runner import run_chunked_ssm_sweep
    from shared.loaders import get_model_config

    model_cfg   = get_model_config(model_name)
    ssm_cfg     = model_cfg.get("ssm", {})
    chunk_size  = ssm_cfg.get("chunk_size", 256)

    # seq_len=1 → one token; use chunk_size=1 so no deadlock risk at low SM count
    result = run_chunked_ssm_sweep(
        model_name=model_name,
        seq_len=1,
        batch_size=batch_size,
        prefill_chunk_tokens=chunk_size,  # decode step shares same kernel shape
        sm_count=sm_count,
        smctrl=smctrl,
        n_warmup=n_warmup,
        n_measure=n_measure,
    )
    return {
        "layer_type":   "ssm_decode",
        "context_len":  context_len,  # informational; SSM decode doesn't read KV cache
        "batch_size":   batch_size,
        "sm_count":     sm_count,
        "latency_ms":   result["latency_ms"],
        "latency_p99_ms": result.get("latency_std_ms", 0.0) + result["latency_ms"],
        "bw_utilization_pct": float("nan"),
    }


# ---------------------------------------------------------------------------
# Attn decode (seq_len=1 + KV cache read)
# ---------------------------------------------------------------------------

def _measure_attn_decode(
    model_name: str,
    context_len: int,
    batch_size: int,
    sm_count: int,
    hw_cfg: dict,
    n_warmup: int = 10,
    n_measure: int = 30,
) -> dict:
    """Measure Attention decode: seq_len=1 attending to context_len KV pairs.

    Uses FlashAttention with query shape (batch, 1, n_heads, head_dim) and
    KV shape (batch, context_len, n_kv_heads, head_dim).

    CUDA Green Context is applied via SMController to restrict SM count.
    """
    from shared.loaders import get_model_config
    from src.smctrl.green_ctx_controller import SMController

    model_cfg = get_model_config(model_name)
    attn_cfg  = model_cfg.get("attention", {})
    n_heads   = attn_cfg.get("num_heads", 32)
    n_kv_heads = attn_cfg.get("num_kv_heads", n_heads)
    head_dim  = attn_cfg.get("head_dim", 128)

    device = "cuda"
    total_sm = hw_cfg["sm_count"]
    bw_gbs   = hw_cfg.get("memory_bw_GBs", None)

    # KV cache dimensions
    dtype = torch.bfloat16
    q  = torch.randn(batch_size, 1,           n_heads,    head_dim, dtype=dtype, device=device)
    k  = torch.randn(batch_size, context_len, n_kv_heads, head_dim, dtype=dtype, device=device)
    v  = torch.randn(batch_size, context_len, n_kv_heads, head_dim, dtype=dtype, device=device)

    try:
        from flash_attn import flash_attn_func
        _HAS_FLASH = True
    except ImportError:
        _HAS_FLASH = False

    smctrl = SMController(device_id=0)

    def _run_once():
        with smctrl.restrict(sm_count, total_sm):
            torch.cuda.synchronize()
            t0 = time.perf_counter_ns()
            if _HAS_FLASH:
                # Reshape to (batch, seq, n_heads, head_dim) for flash_attn API
                q_ = q.squeeze(1)   # (bs, n_heads, head_dim) — not standard; use below
                flash_attn_func(
                    q.transpose(1, 2),   # (bs, n_heads, 1, head_dim)
                    k.transpose(1, 2),   # (bs, n_kv_heads, context_len, head_dim)
                    v.transpose(1, 2),
                )
            else:
                # Fallback: torch scaled_dot_product_attention
                torch.nn.functional.scaled_dot_product_attention(
                    q.view(batch_size, n_heads, 1, head_dim),
                    k.view(batch_size, n_kv_heads, context_len, head_dim).expand(
                        batch_size, n_heads, context_len, head_dim),
                    v.view(batch_size, n_kv_heads, context_len, head_dim).expand(
                        batch_size, n_heads, context_len, head_dim),
                )
            torch.cuda.synchronize()
            return (time.perf_counter_ns() - t0) / 1e6  # ms

    # Warmup
    for _ in range(n_warmup):
        _run_once()

    lats = [_run_once() for _ in range(n_measure)]
    lat_ms = float(sorted(lats)[len(lats) // 2])  # median

    # BW utilization: KV read is the dominant cost
    kv_bytes = 2 * batch_size * context_len * n_kv_heads * head_dim * 2  # 2 bytes (bf16)
    achieved_bw = kv_bytes / (lat_ms * 1e-3) / 1e9  # GB/s
    bw_util_pct = (achieved_bw / bw_gbs * 100) if bw_gbs else float("nan")

    return {
        "layer_type":         "attn_decode",
        "context_len":        context_len,
        "batch_size":         batch_size,
        "sm_count":           sm_count,
        "latency_ms":         lat_ms,
        "latency_p99_ms":     float(sorted(lats)[int(len(lats) * 0.99)]),
        "bw_utilization_pct": bw_util_pct,
    }


# ---------------------------------------------------------------------------
# Decode SM-sensitivity computation
# ---------------------------------------------------------------------------

def compute_decode_sensitivity(rows: list[dict]) -> list[dict]:
    """Add decode_sm_sensitivity to each row.

    sensitivity = (lat_at_min_sm - lat_at_full_sm) / lat_at_full_sm
    Positive → decode is slower at fewer SMs (SM-sensitive).
    Near 0   → decode is SM-insensitive → free zone is genuine.

    Groups by (layer_type, context_len, batch_size).
    """
    import pandas as pd
    if not rows:
        return rows
    df = pd.DataFrame(rows)
    out_rows = []
    group_keys = ["layer_type", "context_len", "batch_size"]
    for keys, grp in df.groupby(group_keys):
        grp = grp.sort_values("sm_count")
        lat_full = grp.iloc[-1]["latency_ms"]   # max SM count = full SM
        lat_min  = grp.iloc[0]["latency_ms"]    # min SM count
        sensitivity = (lat_min - lat_full) / lat_full if lat_full > 0 else 0.0
        for _, row in grp.iterrows():
            r = row.to_dict()
            r["decode_sm_sensitivity"] = round(sensitivity, 4)
            out_rows.append(r)
    return out_rows


# ---------------------------------------------------------------------------
# G1-gate table
# ---------------------------------------------------------------------------

def build_g1_table(
    decode_rows: list[dict],
    prefill_sat_csv: Optional[Path],
    model_name: str,
    hw_cfg: dict,
) -> list[dict]:
    """Combine prefill free-SM fraction with decode sensitivity.

    Reads prefill saturation CSV if it exists; otherwise uses placeholder NaN.
    Output columns: model_name, seq_len, batch_size, layer_type,
                    context_len, free_sm_fraction, decode_sm_sensitivity, g1_potential
    """
    import pandas as pd

    decode_df = pd.DataFrame(decode_rows)
    if decode_df.empty:
        return []

    # One sensitivity value per (layer_type, context_len, batch_size)
    sensitivity_df = (
        decode_df.groupby(["layer_type", "context_len", "batch_size"])
        ["decode_sm_sensitivity"]
        .first()
        .reset_index()
    )

    # Load prefill free-SM fraction if available
    total_sm = hw_cfg["sm_count"]
    tag      = device_tag(hw_cfg)

    prefill_free: dict[tuple, float] = {}
    if prefill_sat_csv and prefill_sat_csv.exists():
        sat_df = pd.read_csv(prefill_sat_csv)
        for _, row in sat_df.iterrows():
            key = (row.get("model_name", model_name),
                   int(row.get("seq_len", 0)),
                   int(row.get("batch_size", 1)))
            prefill_free[key] = float(row.get("free_ratio", 0.0))

    g1_rows = []
    for _, row in sensitivity_df.iterrows():
        for bs in decode_df["batch_size"].unique():
            # Try to find a matching prefill free_sm_fraction
            # (context_len ≈ prefill seq_len for the decode scenario)
            cl = int(row["context_len"])
            best_key = (model_name, cl, int(bs))
            free_frac = prefill_free.get(best_key, float("nan"))

            sens = row["decode_sm_sensitivity"]
            g1 = free_frac * sens if (
                not pd.isna(free_frac) and not pd.isna(sens)
            ) else float("nan")

            g1_rows.append({
                "model_name":             model_name,
                "seq_len_prefill":        cl,
                "batch_size":             int(bs),
                "layer_type_decode":      row["layer_type"],
                "context_len_decode":     cl,
                "free_sm_fraction":       round(free_frac, 4),
                "decode_sm_sensitivity":  round(sens, 4),
                "g1_potential":           round(g1, 4) if not pd.isna(g1) else float("nan"),
                "device_tag":             tag,
            })
    return g1_rows


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------

_DECODE_FIELDS = [
    "layer_type", "context_len", "batch_size", "sm_count", "sm_ratio",
    "latency_ms", "latency_p99_ms", "bw_utilization_pct",
    "decode_sm_sensitivity", "model_name",
]

_G1_FIELDS = [
    "model_name", "seq_len_prefill", "batch_size",
    "layer_type_decode", "context_len_decode",
    "free_sm_fraction", "decode_sm_sensitivity", "g1_potential", "device_tag",
]


def save_decode_csv(rows: list[dict], path: Path, model_name: str) -> None:
    for r in rows:
        r.setdefault("model_name", model_name)
        r.setdefault("sm_ratio", round(r["sm_count"] / max(r.get("sm_count", 1), 1), 4))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_DECODE_FIELDS, extrasaction="ignore", restval="")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved decode CSV: {path}  ({len(rows)} rows)")


def save_g1_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_G1_FIELDS, extrasaction="ignore", restval="")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved G1-gate CSV: {path}  ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# Sweep driver
# ---------------------------------------------------------------------------

def run_decode_sweep(
    model_name: str,
    hw_cfg: dict,
    layer_types: list[str],
    sm_counts: list[int],
    context_lens: list[int],
    batch_sizes: list[int],
    n_warmup: int = 10,
    n_measure: int = 30,
    output_dir: Path = None,
    prefill_sat_csv: Optional[Path] = None,
) -> list[dict]:
    total_sm = hw_cfg["sm_count"]
    tag      = device_tag(hw_cfg)

    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "results" / "stage1" / "decode"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Decode SM Sensitivity Sweep: {model_name} on {hw_cfg['name']} ===")
    print(f"  Total SM   : {total_sm}")
    print(f"  Layer types: {layer_types}")
    print(f"  SM counts  : {sm_counts}")
    print(f"  Context len: {context_lens}  (= KV cache length for Attn decode)")
    print(f"  Batch sizes: {batch_sizes}")

    smctrl_ssm = SMController(device_id=0) if "ssm" in layer_types else None

    all_rows: list[dict] = []
    n_configs = len(layer_types) * len(sm_counts) * len(context_lens) * len(batch_sizes)
    pbar = tqdm(total=n_configs, desc="decode configs")

    for layer_type, sm_count, context_len, batch_size in product(
        layer_types, sm_counts, context_lens, batch_sizes
    ):
        sm_ratio = sm_count / total_sm
        row = None
        try:
            if layer_type == "ssm":
                row = _measure_ssm_decode(
                    model_name, context_len, batch_size, sm_count,
                    smctrl_ssm, n_warmup, n_measure,
                )
            elif layer_type == "attn":
                row = _measure_attn_decode(
                    model_name, context_len, batch_size, sm_count,
                    hw_cfg, n_warmup, n_measure,
                )
            if row is not None:
                row["sm_ratio"] = round(sm_ratio, 4)
                all_rows.append(row)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            tqdm.write(
                f"  OOM: {layer_type} sm={sm_count} ctx={context_len} bs={batch_size}"
            )
        except Exception as exc:
            tqdm.write(
                f"  ERR: {layer_type} sm={sm_count} ctx={context_len} bs={batch_size}: "
                f"{exc!s:.120}"
            )
        finally:
            pbar.update(1)

    pbar.close()

    # Add decode sensitivity
    all_rows = compute_decode_sensitivity(all_rows)

    # Save
    decode_csv = output_dir / f"decode_sm_{model_name}_{tag}.csv"
    save_decode_csv(all_rows, decode_csv, model_name)

    # G1-gate table
    g1_rows = build_g1_table(all_rows, prefill_sat_csv, model_name, hw_cfg)
    if g1_rows:
        g1_csv = output_dir / f"g1_gate_{model_name}_{tag}.csv"
        save_g1_csv(g1_rows, g1_csv)
        _print_g1_summary(g1_rows)

    return all_rows


def _print_g1_summary(g1_rows: list[dict]) -> None:
    """Print G1 gate summary table to stdout."""
    print("\n  === G1 Gate Summary ===")
    print(f"  {'layer_type':<14} {'ctx_len':>8} {'bs':>4} {'free_sm':>8} "
          f"{'decode_sens':>12} {'g1_potential':>13}")
    print("  " + "-" * 62)
    for r in g1_rows:
        free = r["free_sm_fraction"]
        sens = r["decode_sm_sensitivity"]
        g1   = r["g1_potential"]
        print(
            f"  {r['layer_type_decode']:<14} {r['context_len_decode']:>8} "
            f"{r['batch_size']:>4} "
            f"{f'{free:.2%}' if not (isinstance(free, float) and free != free) else 'N/A':>8} "
            f"{sens:>12.3f} "
            f"{f'{g1:.4f}' if not (isinstance(g1, float) and g1 != g1) else 'N/A (no prefill CSV)':>13}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model",   default="zamba2",
                   choices=["zamba2", "falcon_h1", "nemotron_h"])
    p.add_argument("--device",  default="auto",
                   help="Hardware key (hardware.yaml) or 'auto'")
    p.add_argument("--layer-types", nargs="+", default=["ssm", "attn"],
                   choices=["ssm", "attn"],
                   help="Layer types to sweep")
    p.add_argument("--sm-counts", nargs="+", type=int, default=None,
                   help="SM counts to test (default: hw_cfg sm_sweep_steps)")
    p.add_argument("--context-lens", nargs="+", type=int,
                   default=[512, 2048, 4096, 8192],
                   help="KV-cache context lengths for Attn decode; informational for SSM")
    p.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4, 16])
    p.add_argument("--n-warmup",  type=int, default=10)
    p.add_argument("--n-measure", type=int, default=30)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument(
        "--prefill-sat-csv", type=Path, default=None,
        help=(
            "Path to free_sm_zone_{model}_{tag}.csv from plot_saturation.py.  "
            "Used to populate the free_sm_fraction column in the G1-gate table."
        ),
    )
    return p.parse_args()


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required for decode SM sweep")

    args = parse_args()
    hardware_warn()

    hw_cfg   = get_hardware_config(args.device)
    sm_steps = args.sm_counts or hw_cfg["sm_sweep_steps"]

    run_decode_sweep(
        model_name=args.model,
        hw_cfg=hw_cfg,
        layer_types=args.layer_types,
        sm_counts=sm_steps,
        context_lens=args.context_lens,
        batch_sizes=args.batch_sizes,
        n_warmup=args.n_warmup,
        n_measure=args.n_measure,
        output_dir=args.output_dir,
        prefill_sat_csv=args.prefill_sat_csv,
    )


if __name__ == "__main__":
    main()
