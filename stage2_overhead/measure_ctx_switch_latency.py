"""
Stage 2: CUDA Green Contexts stream-switch latency measurement.

Replaces measure_smctrl_latency.py for CUDA driver 550+ (A100/H100/H200).
With Green Contexts, SM partition switching is a CPU-side stream pointer
swap rather than a kernel ioctl; this script quantifies each overhead
component separately.

Four measurement sections:
  1. GreenContext initialization overhead  (one-time cost at startup)
  2. Stream pointer swap latency           (set_sm_ratio CPU cost, no sync)
  3. Layer-boundary transition latency     (swap + optional device sync)
  4. n-layer sequence total overhead       (realistic hybrid model forward)

Results are saved to results/stage2/ctx_switch_overhead_{device}.json.

Usage:
    python stage2_overhead/measure_ctx_switch_latency.py
    python stage2_overhead/measure_ctx_switch_latency.py --n-trials 500 --device auto
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.smctrl import SMController
from src.smctrl.overhead_timer import SMOverheadTimer
from stage1_sm_scaling.run_ssm_prefill_sweep import load_hardware_config, device_tag


# SM ratios matching the migration report's prefill-layer allocation plan
SSM_RATIO  = 0.70   # SSM layers: 70% SMs for prefill
ATTN_RATIO = 0.40   # Attention layers: 40%
DECODE_RATIO = 0.60 # Decode complement (informational only)

# Layer counts spanning both Falcon-H1-7B (44 layers) and Zamba2-7B (81 layers)
N_LAYER_SEQUENCES = [1, 4, 8, 16, 32, 44, 68, 81]


# ---------------------------------------------------------------------------
# Section 1: GreenContext initialization overhead
# ---------------------------------------------------------------------------

def measure_init_overhead(hw_cfg: dict, n_trials: int = 10) -> dict:
    """Measure the one-time cost of SMController.__init__ (Green Context creation).

    Creates a fresh SMController n_trials times and records wall-clock time.
    This is a startup cost; it does not recur per layer.
    """
    print(f"\n[1/4] GreenContext initialization overhead (n={n_trials}) …")
    total_sm = hw_cfg["sm_count"]
    samples = []

    for _ in range(n_trials):
        torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        ctrl = SMController(total_sm_count=total_sm)
        torch.cuda.synchronize()
        t1 = time.perf_counter_ns()
        del ctrl
        samples.append((t1 - t0) / 1_000.0)

    arr = np.array(samples)
    result = {
        "mean_us": float(arr.mean()),
        "median_us": float(np.median(arr)),
        "min_us": float(arr.min()),
        "max_us": float(arr.max()),
        "n_trials": n_trials,
        "note": "one-time startup cost, not per-layer",
    }
    print(f"  mean={result['mean_us']:.1f}μs  median={result['median_us']:.1f}μs  "
          f"min={result['min_us']:.1f}μs  max={result['max_us']:.1f}μs")
    return result


# ---------------------------------------------------------------------------
# Section 2: Pure CPU-side stream pointer swap (set_sm_ratio only, no sync)
# ---------------------------------------------------------------------------

def measure_cpu_swap_latency(smctrl: SMController, n_warmup: int, n_measure: int) -> dict:
    """Measure the raw CPU cost of set_sm_ratio() with no GPU interaction.

    This is a Python dict lookup + attribute assignment.  Expected: < 1 μs.
    """
    print(f"\n[2/4] CPU-side stream pointer swap (set_sm_ratio, no sync) …")
    transitions = [
        (SSM_RATIO,  ATTN_RATIO, "ssm→attn"),
        (ATTN_RATIO, SSM_RATIO,  "attn→ssm"),
        (1.0,        SSM_RATIO,  "full→ssm"),
        (SSM_RATIO,  1.0,        "ssm→full"),
    ]
    results = {}
    for from_r, to_r, label in transitions:
        samples = []
        for i in range(n_warmup + n_measure):
            smctrl.set_sm_ratio(from_r)
            t0 = time.perf_counter_ns()
            smctrl.set_sm_ratio(to_r)
            t1 = time.perf_counter_ns()
            if i >= n_warmup:
                samples.append((t1 - t0) / 1_000.0)
        arr = np.array(samples)
        results[label] = {
            "mean_us":   float(arr.mean()),
            "median_us": float(np.median(arr)),
            "p99_us":    float(np.percentile(arr, 99)),
            "min_us":    float(arr.min()),
            "max_us":    float(arr.max()),
        }
        print(f"  {label}: mean={arr.mean():.3f}μs  p99={np.percentile(arr,99):.3f}μs")
    return results


# ---------------------------------------------------------------------------
# Section 3: Layer-boundary transition (single switch, with/without sync)
# ---------------------------------------------------------------------------

def measure_transition_latency(
    timer: SMOverheadTimer, n_warmup: int, n_measure: int
) -> dict:
    """Measure single A→B layer-boundary switch with/without device sync.

    sync=False: CPU pointer swap only.
    sync=True:  swap + torch.cuda.synchronize() — the realistic layer cost
                when the previous stream must be drained before the next
                kernel can use the new SM partition.
    """
    print(f"\n[3/4] Layer-boundary stream transition latency …")
    transitions = [
        (SSM_RATIO,  ATTN_RATIO, "ssm→attn"),
        (ATTN_RATIO, SSM_RATIO,  "attn→ssm"),
        (1.0,        SSM_RATIO,  "full→ssm"),
        (SSM_RATIO,  1.0,        "ssm→full"),
    ]
    results = {}
    for from_r, to_r, label in transitions:
        for include_sync in [True, False]:
            key = f"{label}_sync_{'yes' if include_sync else 'no'}"
            data = timer.measure_single_transition(
                from_ratio=from_r,
                to_ratio=to_r,
                include_sync=include_sync,
                n_warmup=n_warmup,
                n_measure=n_measure,
            )
            results[key] = data
            print(f"  {key}: mean={data['mean_us']:.2f}μs  "
                  f"p50={data['median_us']:.2f}μs  p99={data['p99_us']:.2f}μs")
    return results


# ---------------------------------------------------------------------------
# Section 4: n-layer sequence total overhead
# ---------------------------------------------------------------------------

def measure_sequence_overhead(
    timer: SMOverheadTimer, n_warmup: int, n_measure: int
) -> dict:
    """Measure total stream-switch overhead for n-layer hybrid model sequences.

    Alternates SSM_RATIO and ATTN_RATIO streams for n_layers, simulating
    a hybrid model forward pass.  Each layer enqueues a minimal kernel on
    its stream.  Total time (sync-bracketed) is reported.
    """
    print(f"\n[4/4] n-layer sequence total overhead …")
    results = {}
    for n in N_LAYER_SEQUENCES:
        data = timer.measure_n_transitions(
            n_layers=n,
            ssm_ratio=SSM_RATIO,
            attn_ratio=ATTN_RATIO,
            n_warmup=n_warmup,
            n_measure=n_measure,
        )
        results[f"n{n}"] = data
        print(f"  n={n:3d} layers: total={data['total_mean_us']:.1f}μs  "
              f"per_layer={data['per_transition_mean_us']:.2f}μs")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Measure CUDA Green Contexts stream-switch latency (Stage 2)"
    )
    p.add_argument("--device", default="auto",
                   help="Hardware config key or 'auto' for current GPU")
    p.add_argument("--n-warmup",  type=int, default=50,
                   help="Warm-up iterations per measurement (default: 50)")
    p.add_argument("--n-measure", type=int, default=200,
                   help="Measurement iterations per data point (default: 200)")
    p.add_argument("--init-trials", type=int, default=10,
                   help="Trials for GreenContext init overhead (default: 10)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required")

    hw_cfg = load_hardware_config(args.device)
    tag = device_tag(hw_cfg)
    total_sm = hw_cfg["sm_count"]

    print(f"Device : {hw_cfg['name']}")
    print(f"Total SM: {total_sm}")

    # Section 1: init overhead (before creating the persistent controller)
    init_data = measure_init_overhead(hw_cfg, n_trials=args.init_trials)

    # Persistent controller for remaining measurements
    torch.cuda.init()
    smctrl = SMController(total_sm_count=total_sm)
    print(f"\nBackend : {smctrl.get_backend_name()}")
    print(f"Available: {smctrl.is_available()}")
    print(f"Presets  : {smctrl._sorted_presets}")
    print(f"Actual SMs (requested→actual): "
          + ", ".join(f"{k}→{v}" for k, v in sorted(smctrl._actual_sm_counts.items())))

    if not smctrl.is_available():
        raise RuntimeError(
            "Green Contexts not available on this GPU/driver.\n"
            "Requires CUDA driver 550+ and A100/H100/H200 GPU."
        )

    timer = SMOverheadTimer(smctrl=smctrl)

    # Section 2: CPU swap latency
    cpu_swap_data = measure_cpu_swap_latency(smctrl, args.n_warmup, args.n_measure)

    # Section 3: layer-boundary transition latency
    transition_data = measure_transition_latency(timer, args.n_warmup, args.n_measure)

    # Section 4: n-layer sequence
    sequence_data = measure_sequence_overhead(timer, args.n_warmup // 2, args.n_measure // 2)

    # Assemble output
    results = {
        "meta": {
            "device": hw_cfg["name"],
            "total_sm": total_sm,
            "backend": smctrl.get_backend_name(),
            "presets": smctrl._sorted_presets,
            "actual_sm_counts": smctrl._actual_sm_counts,
            "n_warmup": args.n_warmup,
            "n_measure": args.n_measure,
            "ssm_ratio": SSM_RATIO,
            "attn_ratio": ATTN_RATIO,
        },
        "init_overhead": init_data,
        "cpu_swap": cpu_swap_data,
        "single_transitions": transition_data,
        "n_transitions": sequence_data,
    }

    # Summary
    key = "ssm→attn_sync_yes"
    if key in transition_data:
        t = transition_data[key]
        print(f"\nSummary: ssm→attn (sync) mean={t['mean_us']:.2f}μs  "
              f"p99={t['p99_us']:.2f}μs")
    n81 = sequence_data.get("n81", {})
    if n81:
        print(f"         81-layer sequence: total={n81['total_mean_us']:.1f}μs  "
              f"per_layer={n81['per_transition_mean_us']:.2f}μs")

    output_dir = Path(__file__).parent.parent / "results" / "stage2"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"ctx_switch_overhead_{tag}.json"

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved: {out_path}")
