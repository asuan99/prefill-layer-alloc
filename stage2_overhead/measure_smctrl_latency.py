"""
Stage 2: CUDA Green Context SM partition switch latency measurement.

Measures three distinct overhead components:
  1. Single SM partition transition (A → B): with/without GPU sync
  2. n consecutive transitions: simulate a full hybrid model layer sequence
  3. Cold-start kernel penalty: first kernel after stream switch

Results are saved to results/stage2/smctrl_overhead_{device}.json.

Usage:
    python stage2_overhead/measure_smctrl_latency.py
    python stage2_overhead/measure_smctrl_latency.py --n-trials 500 --device auto

Note: prefer measure_ctx_switch_latency.py for the full Green Context benchmark
(initialization overhead, stream swap cost, etc.).  This script keeps the same
measurement structure as the original libsmctrl version for comparison.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import json
from pathlib import Path

import torch

from src.smctrl.green_ctx_controller import SMController
from src.smctrl.overhead_timer import SMOverheadTimer
from stage1_sm_scaling.run_ssm_prefill_sweep import load_hardware_config, device_tag


# Representative SM ratio transitions for hybrid model layers
SSM_RATIO = 0.70    # SSM layers use 70% of SMs for prefill
ATTN_RATIO = 0.40   # Attention layers use 40%
DECODE_RATIO = 0.60 # Decode uses the complement

# Layer counts matching Zamba2 (54 layers) and Falcon-H1 (32 layers)
N_LAYER_SEQUENCES = [1, 4, 8, 16, 32, 54, 64]


def run_measurements(
    smctrl: SMController,
    timer: SMOverheadTimer,
    n_warmup: int,
    n_measure: int,
) -> dict:
    results = {}

    print("\n[1/3] Single SM mask transition latency …")
    transitions = [
        (SSM_RATIO, ATTN_RATIO, "ssm→attn"),
        (ATTN_RATIO, SSM_RATIO, "attn→ssm"),
        (1.0, SSM_RATIO, "full→ssm"),
        (SSM_RATIO, 1.0, "ssm→full"),
    ]

    results["single_transitions"] = {}
    for from_r, to_r, label in transitions:
        for include_sync in [True, False]:
            key = f"{label}_sync{'_yes' if include_sync else '_no'}"
            print(f"  Measuring {key} …")
            data = timer.measure_single_transition(
                from_ratio=from_r,
                to_ratio=to_r,
                include_sync=include_sync,
                n_warmup=n_warmup,
                n_measure=n_measure,
            )
            results["single_transitions"][key] = data
            print(
                f"    mean={data['mean_us']:.2f}μs  "
                f"p50={data['median_us']:.2f}μs  "
                f"p99={data['p99_us']:.2f}μs"
            )

    print("\n[2/3] Consecutive n-transition sequences …")
    results["n_transitions"] = {}
    for n in N_LAYER_SEQUENCES:
        print(f"  n={n} layers …")
        data = timer.measure_n_transitions(
            n_layers=n,
            ssm_ratio=SSM_RATIO,
            attn_ratio=ATTN_RATIO,
            n_warmup=n_warmup,
            n_measure=n_measure,
        )
        results["n_transitions"][f"n{n}"] = data
        print(
            f"    total={data['total_mean_us']:.1f}μs  "
            f"per_layer={data['per_transition_mean_us']:.2f}μs"
        )

    print("\n[3/3] Cold-start kernel penalty …")
    results["cold_start"] = {}
    for ratio in [SSM_RATIO, ATTN_RATIO, 0.5]:
        key = f"ratio_{int(ratio*100)}pct"
        print(f"  SM ratio={ratio:.0%} …")
        data = timer.measure_cold_start_penalty(
            ratio=ratio,
            n_warmup=n_warmup,
            n_measure=n_measure,
        )
        results["cold_start"][key] = data
        print(
            f"    baseline={data['baseline_mean_us']:.2f}μs  "
            f"post_reconfig={data['post_reconfig_mean_us']:.2f}μs  "
            f"penalty={data['cold_start_penalty_us']:.2f}μs "
            f"({data['penalty_ratio']:.1%})"
        )

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Measure Green Context SM partition switch latency")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-warmup", type=int, default=50)
    parser.add_argument("--n-measure", type=int, default=200)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required")

    hw_cfg = load_hardware_config(args.device)
    tag = device_tag(hw_cfg)

    print(f"Device: {hw_cfg['name']}  (total SM: {hw_cfg['sm_count']})")

    smctrl = SMController(total_sm_count=hw_cfg["sm_count"])
    timer = SMOverheadTimer(smctrl=smctrl)

    print(f"Backend: {smctrl.get_backend_name()}")

    results = run_measurements(smctrl, timer, args.n_warmup, args.n_measure)

    # Metadata
    results["meta"] = {
        "device": hw_cfg["name"],
        "total_sm": hw_cfg["sm_count"],
        "backend": smctrl.get_backend_name(),
        "n_warmup": args.n_warmup,
        "n_measure": args.n_measure,
        "ssm_ratio": SSM_RATIO,
        "attn_ratio": ATTN_RATIO,
    }

    output_dir = Path(__file__).parent.parent / "results" / "stage2"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"smctrl_overhead_{tag}.json"

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved: {out_path}")
