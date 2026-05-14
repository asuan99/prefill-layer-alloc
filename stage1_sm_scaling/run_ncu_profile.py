"""
Stage 1: ncu (Nsight Compute) profiling sweep for per-SM hardware counter measurement.

Unlike the NVML-based sweep (run_*_prefill_sweep.py), this script uses ncu to capture
exact hardware performance counters — particularly for wave quantization analysis:

  sm__active_cycles_sum / sm__cycles_elapsed_sum → actual per-SM utilization
  launch__grid_size + sm_count                   → n_waves, wave_efficiency_pct

Key differences vs NVML sweep:
  - 10–100× slower (ncu serializes kernel execution for counter collection)
  - NOT a replacement for latency measurement — use only for wave/SM diagnostics
  - Requires `sudo` or perf_event_paranoid ≤ 2 (ncu needs hardware counter access)
  - Spawns a child process per config (ncu cannot run inside a running CUDA process)

Usage:
    # Minimal — quick diagnostic on a few configs
    python stage1_sm_scaling/run_ncu_profile.py --model zamba2 \\
        --sm-counts 27 54 108 --seq-lens 1024 4096

    # Full wave sweep (slow — expect 5–30 min per config under ncu)
    python stage1_sm_scaling/run_ncu_profile.py --model zamba2 \\
        --layer-types ssm attn mlp \\
        --sm-counts 14 27 54 108 --seq-lens 512 1024 2048 4096 \\
        --batch-sizes 1 4

    # With extended metrics (memory + compute in addition to wave stats)
    python stage1_sm_scaling/run_ncu_profile.py --model zamba2 --full-metrics

    # Under nsys for timeline correlation (add NVTX markers alongside ncu)
    nsys profile --trace=cuda,nvtx --output=results/stage1/nsys_ncu_zamba2 \\
        python stage1_sm_scaling/run_ncu_profile.py --model zamba2 --sm-counts 27 108

Output CSV columns (in addition to config fields):
  sm_util_per_sm_pct    — active_cycles/elapsed_cycles × 100 (per-SM, not device-wide)
  n_waves               — number of execution waves
  wave_efficiency_pct   — n_blocks / (n_waves × sm_count) × 100
  last_wave_blocks      — blocks active in the last wave (0 = perfect fit)
  achieved_occupancy_pct — smsp__maximum_warps_per_active_cycle_pct
  warps_active_avg      — smsp__warps_active_avg_per_cycle_active
  grid_size             — launch__grid_size (actual from hardware)
  analytical_n_waves    — WaveEstimator estimate (for comparison)
  analytical_wave_eff   — WaveEstimator wave_efficiency (for comparison)

Performance note:
  ncu profiling is inherently slow due to hardware counter serialization.
  Use --sm-counts and --seq-lens sparingly. A 4×4×2 sweep (sm × seq × bs)
  can take 30–60 minutes depending on GPU and kernel complexity.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import csv
import math
from pathlib import Path

import torch
import yaml

from src.profiling.ncu_runner import NCURunner, NCU_METRICS_WAVE, NCU_METRICS_FULL
from src.profiling.wave_estimator import WaveEstimator
from stage1_sm_scaling.run_ssm_prefill_sweep import compute_sm_steps


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model_config(model_name: str) -> dict:
    cfg_path = Path(__file__).parent.parent / "configs" / "models.yaml"
    with open(cfg_path) as f:
        models = yaml.safe_load(f)
    if model_name not in models:
        raise ValueError(f"Model {model_name!r} not in configs/models.yaml")
    return models[model_name]


def load_hardware_config(device_key: str) -> dict:
    cfg_path = Path(__file__).parent.parent / "configs" / "hardware.yaml"
    with open(cfg_path) as f:
        hw = yaml.safe_load(f)

    if device_key == "auto" or device_key not in hw:
        props = torch.cuda.get_device_properties(0)
        n_sm = props.multi_processor_count
        return {
            "name": torch.cuda.get_device_name(0),
            "sm_count": n_sm,
        }
    return hw[device_key]


def device_tag(hw_cfg: dict) -> str:
    name = hw_cfg["name"].lower()
    return name.replace("nvidia ", "").replace(" ", "_")


def _add_analytical_wave(row: dict, layer_type: str, model_cfg: dict) -> dict:
    """Compute WaveEstimator estimates and add to row for comparison."""
    sm_count = row.get("sm_count", 0)
    seq_len = row.get("seq_len", 0)
    batch_size = row.get("batch_size", 1)

    try:
        if layer_type == "ssm":
            stats = WaveEstimator.ssm_prefill(
                batch=batch_size,
                seq_len=seq_len,
                n_heads=model_cfg.get("n_ssm_heads", 64),
                chunk_size=model_cfg.get("chunk_size", 256),
                sm_count=sm_count,
            )
        elif layer_type == "chunked_ssm":
            pct = row.get("prefill_chunk_tokens", 0)
            if not pct:
                return row
            ssm_cfg = model_cfg.get("ssm", {})
            n_heads  = ssm_cfg.get("n_heads", model_cfg.get("n_ssm_heads", 64))
            ssd_chunk = ssm_cfg.get("chunk_size", model_cfg.get("chunk_size", 256))
            stats = WaveEstimator.ssm_chunked_prefill(
                batch=batch_size,
                prefill_chunk_tokens=pct,
                n_heads=n_heads,
                ssd_chunk_size=ssd_chunk,
                sm_count=sm_count,
            )
        elif layer_type == "attn":
            stats = WaveEstimator.attn_prefill(
                batch=batch_size,
                seq_len=seq_len,
                n_heads=model_cfg.get("n_attn_heads", 8),
                head_dim=model_cfg.get("attn_head_dim", 256),
                sm_count=sm_count,
            )
        elif layer_type == "mlp":
            stats = WaveEstimator.mlp_gemm(
                batch=batch_size,
                seq_len=seq_len,
                hidden_size=model_cfg.get("hidden_size", 2048),
                intermediate_size=model_cfg.get("intermediate_size", 4096),
                sm_count=sm_count,
            )
        else:
            return row

        row["analytical_n_blocks"] = stats.n_blocks
        row["analytical_n_waves"] = stats.n_waves
        row["analytical_wave_eff_pct"] = stats.wave_efficiency * 100.0
        row["analytical_last_wave_blocks"] = stats.last_wave_blocks
        row["analytical_wasted_sm_pct"] = stats.wasted_sm_fraction * 100.0
    except Exception:
        pass

    return row


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def run_ncu_sweep(
    layer_types: list[str],
    model_name: str,
    model_cfg: dict,
    sm_counts: list[int],
    seq_lens: list[int],
    batch_sizes: list[int],
    output_dir: Path,
    hw_cfg: dict,
    use_full_metrics: bool = False,
    n_warmup: int = 10,
    n_measure: int = 3,
    timeout_s: int = 300,
    chunked_prefill_tokens: list[int] = None,
) -> list[dict]:
    metrics = NCU_METRICS_FULL if use_full_metrics else NCU_METRICS_WAVE

    ncu = NCURunner()
    if not ncu.is_available():
        raise RuntimeError(
            "ncu not found. Install CUDA toolkit and ensure ncu is on PATH.\n"
            "  sudo apt-get install cuda-toolkit  (for Ubuntu/Debian)\n"
            "  Expected: /usr/local/cuda/bin/ncu\n"
            "\n"
            "Also ensure hardware counter access:\n"
            "  echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | "
            "sudo tee /etc/modprobe.d/nvidia-profiling.conf\n"
            "  sudo modprobe nvidia NVreg_RestrictProfilingToAdminUsers=0\n"
            "  OR: sudo ncu ..."
        )

    n_pct_configs = len(chunked_prefill_tokens) if chunked_prefill_tokens else 1
    n_chunked_ssm = sum(1 for lt in layer_types if lt == "chunked_ssm")
    n_other = len(layer_types) - n_chunked_ssm
    total_configs = (
        (n_other + n_chunked_ssm * n_pct_configs)
        * len(sm_counts) * len(seq_lens) * len(batch_sizes)
    )
    total_sm = hw_cfg["sm_count"]
    tag = device_tag(hw_cfg)

    print(f"\n=== ncu Wave Profiling: {model_name} on {hw_cfg['name']} ===")
    print(f"  Layer types  : {layer_types}")
    print(f"  SM counts    : {sm_counts}")
    print(f"  seq_lens     : {seq_lens}")
    print(f"  batch_sizes  : {batch_sizes}")
    if chunked_prefill_tokens:
        print(f"  prefill_chunk: {chunked_prefill_tokens}")
    print(f"  Metrics      : {'FULL' if use_full_metrics else 'WAVE'}")
    print(f"  Total configs: {total_configs}")
    est_lo = total_configs * 2
    est_hi = total_configs * 10
    est_lo_str = f"{est_lo}min" if est_lo < 120 else f"{est_lo//60}h{est_lo%60:02d}m"
    est_hi_str = f"{est_hi}min" if est_hi < 120 else f"{est_hi//60}h{est_hi%60:02d}m"
    print(
        f"\n  WARNING: ncu profiling is 10-100× slower than bare execution.\n"
        f"  Estimated runtime: {est_lo_str}–{est_hi_str}  ({total_configs} configs × 2–10 min each).\n"
        f"  Use --layer-types, --sm-counts, or --seq-lens to reduce sweep size.\n"
    )

    all_results = []
    done = 0

    for layer_type in layer_types:
        layer_results = []
        print(f"\n--- Layer: {layer_type} ---")

        # For chunked_ssm, sweep over prefill_chunk_tokens as an extra dimension.
        pct_list = (
            chunked_prefill_tokens
            if layer_type == "chunked_ssm" and chunked_prefill_tokens
            else [0]
        )

        for pct in pct_list:
            for sm_count in sm_counts:
                for seq_len in seq_lens:
                    for batch_size in batch_sizes:
                        done += 1
                        sm_pct = sm_count / total_sm * 100
                        pct_tag = f" pct={pct}" if pct > 0 else ""
                        print(
                            f"  [{done}/{total_configs}] ncu: {layer_type}{pct_tag} "
                            f"sm={sm_count}({sm_pct:.0f}%) seq={seq_len} bs={batch_size} ...",
                            flush=True,
                        )

                        row = ncu.profile(
                            layer_type=layer_type,
                            model=model_name,
                            sm_count=sm_count,
                            seq_len=seq_len,
                            batch_size=batch_size,
                            metrics=metrics,
                            prefill_chunk_tokens=pct,
                            n_warmup=n_warmup,
                            n_measure=n_measure,
                            timeout_s=timeout_s,
                        )

                        # Annotate with config metadata
                        row["total_sm"] = total_sm
                        row["sm_ratio"] = sm_count / total_sm

                        if "error" not in row:
                            # Print key diagnostics
                            sm_util = row.get("sm_util_per_sm_pct", float("nan"))
                            n_waves = row.get("n_waves", "N/A")
                            wave_eff = row.get("wave_efficiency_pct", float("nan"))
                            occupancy = row.get("achieved_occupancy_pct", float("nan"))
                            grid = row.get("grid_size", "N/A")
                            print(
                                f"    grid={grid}  waves={n_waves}  "
                                f"wave_eff={wave_eff:.1f}%  "
                                f"sm_util={sm_util:.1f}%  "
                                f"occupancy={occupancy:.1f}%"
                            )

                            # Add analytical comparison
                            row = _add_analytical_wave(row, layer_type, model_cfg)

                            # Print delta vs analytical estimate
                            an_waves = row.get("analytical_n_waves")
                            an_eff = row.get("analytical_wave_eff_pct")
                            if an_waves is not None:
                                wave_delta = (
                                    f"  [analytical: waves={an_waves} "
                                    f"eff={an_eff:.1f}%"
                                )
                                meas_n = row.get("n_waves")
                                if meas_n is not None and meas_n != an_waves:
                                    wave_delta += f" ← differs from measured {meas_n}"
                                wave_delta += "]"
                                print(f"  {wave_delta}")
                        else:
                            print(f"    ERROR: {row['error']}")
                            # ncu errors (ERR_NVGPUCTRPERM etc.) go to stdout, not stderr
                            ncu_out = row.get("ncu_stdout", "")
                            if ncu_out:
                                print(f"    NCU_OUT: {ncu_out.strip()}")
                            stderr = row.get("stderr", "")
                            if stderr:
                                print(f"    STDERR: {stderr.strip()}")
                            row = _add_analytical_wave(row, layer_type, model_cfg)

                        layer_results.append(row)

        all_results.extend(layer_results)

        # Save per-layer CSV immediately (don't lose data if later layers fail)
        _save_csv(
            layer_results,
            output_dir / f"ncu_{layer_type}_{model_name}_{tag}.csv",
            layer_type,
        )

    return all_results


def _save_csv(results: list[dict], out_path: Path, layer_type: str) -> None:
    if not results:
        return

    # Ordered fieldnames — config first, then hardware metrics, then analytical
    priority_fields = [
        "layer_type", "model", "sm_count", "sm_ratio", "total_sm",
        "seq_len", "batch_size", "prefill_chunk_tokens",
        # ncu-derived
        "kernel_name", "n_kernels_captured",
        "grid_size", "n_waves", "last_wave_blocks", "wave_efficiency_pct",
        "sm_util_per_sm_pct", "achieved_occupancy_pct", "warps_active_avg",
        # analytical comparison
        "analytical_n_blocks", "analytical_n_waves",
        "analytical_wave_eff_pct", "analytical_last_wave_blocks",
        "analytical_wasted_sm_pct",
        # error (if any)
        "error",
    ]

    # Collect all keys from results
    all_keys = set()
    for row in results:
        all_keys.update(row.keys())

    # Build fieldnames: priority fields first, then remaining in sorted order
    fieldnames = [f for f in priority_fields if f in all_keys]
    extra = sorted(all_keys - set(fieldnames))
    fieldnames = fieldnames + extra

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    print(f"\n  Saved: {out_path}  ({len(results)} rows)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="ncu wave quantization profiling sweep",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model", choices=["zamba2", "falcon_h1"], default="zamba2",
        help="Model to profile",
    )
    parser.add_argument(
        "--device", default="auto",
        help="Hardware key from configs/hardware.yaml, or 'auto' for runtime detection",
    )
    parser.add_argument(
        "--layer-types", nargs="+",
        choices=["ssm", "chunked_ssm", "attn", "mlp"], default=["ssm", "attn", "mlp"],
        help="Layer types to profile (default: ssm attn mlp)",
    )
    parser.add_argument(
        "--prefill-chunk-tokens", nargs="+", type=int, default=None,
        metavar="PCT",
        help=(
            "Tokens per kernel call for chunked_ssm (required when --layer-types includes "
            "chunked_ssm). Multiple values create a sweep. "
            "cooperative-safe condition: batch × ceil(pct/256) × n_heads ≤ sm_count."
        ),
    )
    parser.add_argument(
        "--sm-counts", nargs="+", type=int, default=None,
        help=(
            "SM counts to sweep. Default: 8 steps from ~12%% to 100%% of GPU SM count "
            "(same as latency sweeps). Keep small — each config takes minutes under ncu."
        ),
    )
    parser.add_argument(
        "--seq-lens", nargs="+", type=int, default=[256, 512, 1024, 2048, 4096, 8192, 16384, 32768],
        help="Sequence lengths to profile (default: 256 512 1024 2048 4096 8192 16384 32768)",
    )
    parser.add_argument(
        "--batch-sizes", nargs="+", type=int, default=[1, 4, 16, 32, 64],
        help="Batch sizes (default: 1 4 16 32 64)",
    )
    parser.add_argument(
        "--full-metrics", action="store_true",
        help="Collect extended metrics set (memory + compute), slower",
    )
    parser.add_argument(
        "--n-warmup", type=int, default=10,
        help="Warmup launches before measured kernel (default: 10)",
    )
    parser.add_argument(
        "--n-measure", type=int, default=3,
        help="Measured kernel launches (ncu --launch-count, default: 3)",
    )
    parser.add_argument(
        "--timeout", type=int, default=300,
        help="Per-config ncu subprocess timeout in seconds (default: 300)",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).parent.parent / "results" / "stage1",
        help="Directory for output CSV files",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required")

    # Early check: GPU hardware counter access (fails silently without this)
    ncu_check = NCURunner()
    ok, msg = ncu_check.check_permissions()
    if not ok:
        print(f"\n[ERROR] ncu permission check failed:\n{msg}", file=sys.stderr)
        sys.exit(1)

    hw_cfg = load_hardware_config(args.device)
    model_cfg = load_model_config(args.model)
    total_sm = hw_cfg["sm_count"]

    # Default SM counts: same 8-step sweep as latency sweeps
    if args.sm_counts is None:
        args.sm_counts = compute_sm_steps(total_sm, n_steps=8)

    if "chunked_ssm" in args.layer_types and not args.prefill_chunk_tokens:
        print(
            "[ERROR] --prefill-chunk-tokens is required when --layer-types includes chunked_ssm.\n"
            "  Example: --prefill-chunk-tokens 256 512 1024",
            file=sys.stderr,
        )
        sys.exit(1)

    run_ncu_sweep(
        layer_types=args.layer_types,
        model_name=args.model,
        model_cfg=model_cfg,
        sm_counts=args.sm_counts,
        seq_lens=args.seq_lens,
        batch_sizes=args.batch_sizes,
        output_dir=args.output_dir,
        hw_cfg=hw_cfg,
        use_full_metrics=args.full_metrics,
        n_warmup=args.n_warmup,
        n_measure=args.n_measure,
        timeout_s=args.timeout,
        chunked_prefill_tokens=args.prefill_chunk_tokens,
    )

    print("\nDone.")
