"""
Stage 3: Prefill+Decode concurrent execution evaluation.

Simulates the BulletServe/MuxWise layer-interleaved serving scenario:
  - Decode batch: continuous decoding of a fixed batch
  - Prefill: new requests arrive every N decode steps
  - Layer-level interleaving: one decode step + one prefill layer per iteration

This is NOT true kernel-level parallelism but layer-level interleaving
on a single GPU, matching the execution model of MuxWise/BulletServe.

Policies evaluated:
  A (baseline):     fixed SM split (40/60), no layer-boundary reconfig
  B (step_adaptive): step-level SM ratio based on model SSM fraction
  C (layer_wise):   layer-boundary SM reconfig (only if Stage 2 warrants)

Usage:
    python stage3_hm_eval/run_concurrent_eval.py --model zamba2 --policy all
    python stage3_hm_eval/run_concurrent_eval.py --model falcon_h1 --policy A B
    python stage3_hm_eval/run_concurrent_eval.py --model zamba2 --policy C --prefill-seq-len 1024
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import csv
import time
from pathlib import Path
from collections import deque
from dataclasses import dataclass, field

import torch
import numpy as np

from src.smctrl.libsmctrl_wrapper import SMController
from src.models.layer_runner import LayerRunner
from src.profiling.nvml_monitor import NVMLMonitor
from stage1_sm_scaling.run_ssm_prefill_sweep import load_hardware_config, device_tag
from stage3_hm_eval.policy_baseline import PolicyBaseline, PolicyConfig as BaselineConfig
from stage3_hm_eval.policy_step_adaptive import PolicyStepAdaptive, PolicyConfig as StepConfig
from stage3_hm_eval.policy_layer_wise import (
    PolicyLayerWise, PolicyConfig as LayerConfig, should_run_policy_c
)


# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------

SLO_TPOT_MS = 50.0       # decode TPOT SLO threshold
PREFILL_INTERVAL_STEPS = 4  # new prefill request every N decode steps
N_PREFILL_REQUESTS = 50   # run until this many prefill requests are processed

MODEL_NUM_LAYERS = {
    "zamba2": 54,
    "falcon_h1": 32,
}

# Zamba2: SSM every layer except every 9th (attention); Falcon-H1: all hybrid
MODEL_LAYER_TYPES = {
    "zamba2": lambda i: "ssm" if i % 9 != 6 else "attn",
    "falcon_h1": lambda i: "ssm" if i % 2 == 0 else "attn",  # simplified: alternating
}


# ---------------------------------------------------------------------------
# State tracking
# ---------------------------------------------------------------------------

@dataclass
class PrefillState:
    seq_len: int
    n_layers: int
    layer_type_fn: callable
    layer_idx: int = 0
    start_time_ms: float = 0.0

    def current_layer_type(self) -> str:
        return self.layer_type_fn(self.layer_idx)

    def advance(self) -> bool:
        """Advance to next layer. Returns True if prefill is complete."""
        self.layer_idx += 1
        return self.layer_idx >= self.n_layers

    def is_complete(self) -> bool:
        return self.layer_idx >= self.n_layers


@dataclass
class EvalMetrics:
    prefill_ttft_ms: list = field(default_factory=list)
    decode_tpot_ms: list = field(default_factory=list)
    slo_violations: int = 0
    n_prefill_completed: int = 0
    n_decode_steps: int = 0

    def prefill_throughput_toks_per_sec(self, seq_len: int) -> float:
        if not self.prefill_ttft_ms:
            return 0.0
        mean_ttft_s = np.mean(self.prefill_ttft_ms) / 1000.0
        return seq_len / mean_ttft_s if mean_ttft_s > 0 else 0.0


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------

def run_eval(
    model_name: str,
    policy,
    runner: LayerRunner,
    decode_batch_size: int,
    prefill_seq_len: int,
    context_len: int,
    total_sm: int,
    n_warmup_steps: int = 20,
    nvml_interval_ms: int = 500,
) -> tuple[EvalMetrics, list[dict]]:
    """Run the interleaved prefill+decode evaluation loop.

    Structure per iteration:
      1. [Decode] set SM → decode ratio, run decode step
      2. [Prefill] set SM → prefill ratio (per policy), run 1 prefill layer

    This matches MuxWise/BulletServe layer-level interleaving.
    """
    n_layers = MODEL_NUM_LAYERS[model_name]
    layer_type_fn = MODEL_LAYER_TYPES[model_name]

    metrics = EvalMetrics()
    sm_timeline: list[dict] = []

    monitor = NVMLMonitor()
    monitor.start(interval_ms=nvml_interval_ms)

    prefill_queue: deque[PrefillState] = deque()
    current_prefill: PrefillState | None = None

    eval_start_ms = time.perf_counter_ns() / 1_000_000.0
    decode_step = 0
    prefill_request_id = 0

    # Warm-up: decode-only steps
    for _ in range(n_warmup_steps):
        policy.on_decode()
        _run_decode_step(runner, model_name, decode_batch_size, context_len, total_sm)

    print(f"  Warm-up done. Starting evaluation …")

    while metrics.n_prefill_completed < N_PREFILL_REQUESTS:
        step_start_ms = time.perf_counter_ns() / 1_000_000.0

        # Inject new prefill request every PREFILL_INTERVAL_STEPS steps
        if decode_step % PREFILL_INTERVAL_STEPS == 0 and prefill_request_id < N_PREFILL_REQUESTS:
            ps = PrefillState(
                seq_len=prefill_seq_len,
                n_layers=n_layers,
                layer_type_fn=layer_type_fn,
                start_time_ms=step_start_ms,
            )
            prefill_queue.append(ps)
            prefill_request_id += 1

        # [Phase 1] Decode step
        policy.on_decode()
        decode_t0 = time.perf_counter_ns()
        _run_decode_step(runner, model_name, decode_batch_size, context_len, total_sm)
        decode_t1 = time.perf_counter_ns()
        tpot_ms = (decode_t1 - decode_t0) / 1_000_000.0

        metrics.decode_tpot_ms.append(tpot_ms)
        metrics.n_decode_steps += 1
        if tpot_ms > SLO_TPOT_MS:
            metrics.slo_violations += 1

        # [Phase 2] One prefill layer (if any request in queue)
        if current_prefill is None and prefill_queue:
            current_prefill = prefill_queue.popleft()

        if current_prefill is not None:
            lt = current_prefill.current_layer_type()
            policy.on_prefill_layer_start(current_prefill.layer_idx, lt)

            _run_prefill_layer(
                runner, model_name, decode_batch_size, prefill_seq_len, total_sm
            )

            policy.on_prefill_layer_end(current_prefill.layer_idx, lt)

            if current_prefill.advance():
                # Prefill complete → record TTFT
                ttft_ms = step_start_ms - current_prefill.start_time_ms + tpot_ms
                metrics.prefill_ttft_ms.append(ttft_ms)
                metrics.n_prefill_completed += 1
                current_prefill = None

                if metrics.n_prefill_completed % 10 == 0:
                    print(
                        f"  prefill={metrics.n_prefill_completed}/{N_PREFILL_REQUESTS}  "
                        f"decode_steps={metrics.n_decode_steps}  "
                        f"TPOT_p50={np.percentile(metrics.decode_tpot_ms, 50):.1f}ms  "
                        f"SLO_viol={metrics.slo_violations}"
                    )

        decode_step += 1

    sm_df = monitor.stop()
    if not sm_df.empty:
        sm_timeline = sm_df.to_dict("records")

    return metrics, sm_timeline


def _run_decode_step(
    runner: LayerRunner, model_name: str, batch_size: int, context_len: int, total_sm: int
) -> None:
    """Simulate a single decode step: token-by-token (seq_len=1)."""
    # Decode: SSM in recurrent mode (seq_len=1), Attn with KV cache
    runner.run_ssm_layer(
        model_name=model_name,
        batch_size=batch_size,
        seq_len=1,
        sm_count=total_sm,  # SM already set by policy; runner will use current mask
        n_warmup=0,
        n_measure=1,
    )


def _run_prefill_layer(
    runner: LayerRunner, model_name: str, batch_size: int, seq_len: int, total_sm: int
) -> None:
    """Simulate one prefill layer (SSM or Attn)."""
    runner.run_ssm_layer(
        model_name=model_name,
        batch_size=batch_size,
        seq_len=seq_len,
        sm_count=total_sm,  # SM already set by policy
        n_warmup=0,
        n_measure=1,
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_results(
    metrics: EvalMetrics,
    sm_timeline: list[dict],
    model_name: str,
    policy_name: str,
    prefill_seq_len: int,
    decode_batch_size: int,
    hw_cfg: dict,
    output_dir: Path,
) -> None:
    tag = device_tag(hw_cfg)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-request metrics CSV
    n = metrics.n_prefill_completed
    out_csv = output_dir / f"eval_{model_name}_{policy_name}_{tag}.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "policy", "seq_len", "batch_size",
            "n_prefill", "n_decode_steps",
            "ttft_mean_ms", "ttft_p99_ms",
            "tpot_p50_ms", "tpot_p99_ms",
            "prefill_throughput_toks_per_sec",
            "slo_violations", "slo_violation_rate",
        ])
        writer.writerow([
            model_name, policy_name, prefill_seq_len, decode_batch_size,
            n, metrics.n_decode_steps,
            round(np.mean(metrics.prefill_ttft_ms), 2) if metrics.prefill_ttft_ms else "N/A",
            round(np.percentile(metrics.prefill_ttft_ms, 99), 2) if metrics.prefill_ttft_ms else "N/A",
            round(np.percentile(metrics.decode_tpot_ms, 50), 2) if metrics.decode_tpot_ms else "N/A",
            round(np.percentile(metrics.decode_tpot_ms, 99), 2) if metrics.decode_tpot_ms else "N/A",
            round(metrics.prefill_throughput_toks_per_sec(prefill_seq_len), 1),
            metrics.slo_violations,
            round(metrics.slo_violations / max(metrics.n_decode_steps, 1), 4),
        ])
    print(f"  Saved: {out_csv}")

    # SM timeline CSV
    if sm_timeline:
        out_sm = output_dir / f"sm_timeline_{model_name}_{policy_name}_{tag}.csv"
        with open(out_sm, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sm_timeline[0].keys())
            writer.writeheader()
            writer.writerows(sm_timeline)
        print(f"  Saved: {out_sm}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Stage 3: Concurrent prefill+decode evaluation")
    parser.add_argument("--model", choices=["zamba2", "falcon_h1"], required=True)
    parser.add_argument(
        "--policy", nargs="+", choices=["A", "B", "C", "all"], default=["all"]
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--decode-batch-size", type=int, default=8)
    parser.add_argument("--prefill-seq-len", type=int, default=1024)
    parser.add_argument("--context-len", type=int, default=1024)
    parser.add_argument("--n-warmup-steps", type=int, default=20)
    parser.add_argument("--slo-tpot-ms", type=float, default=SLO_TPOT_MS)
    parser.add_argument(
        "--decision-matrix",
        type=Path,
        default=Path(__file__).parent.parent / "results" / "stage2" / "decision_matrix.json"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required")

    global SLO_TPOT_MS
    SLO_TPOT_MS = args.slo_tpot_ms

    hw_cfg = load_hardware_config(args.device)
    total_sm = hw_cfg["sm_count"]
    output_dir = Path(__file__).parent.parent / "results" / "stage3"

    smctrl = SMController(total_sm_count=total_sm)
    runner = LayerRunner(device="cuda", total_sm_count=total_sm)

    # Determine which policies to run
    policies_to_run = set(args.policy)
    if "all" in policies_to_run:
        policies_to_run = {"A", "B", "C"}

    # Check if Policy C is warranted by Stage 2
    if "C" in policies_to_run and not should_run_policy_c(args.decision_matrix):
        print("Policy C (layer_wise) skipped: Stage 2 overhead_ratio >= 0.05")
        print("(Run Stage 2 first, or force with --decision-matrix /dev/null)")
        policies_to_run.discard("C")

    policy_map = {
        "A": PolicyBaseline(smctrl, BaselineConfig()),
        "B": PolicyStepAdaptive(smctrl, StepConfig(model_name=args.model)),
        "C": PolicyLayerWise(smctrl, LayerConfig()),
    }

    for policy_key in sorted(policies_to_run):
        policy = policy_map[policy_key]
        print(f"\n{'='*60}")
        print(f"Running Policy {policy_key}: {policy.describe()['policy']}")
        print(f"  model={args.model}  decode_bs={args.decode_batch_size}  "
              f"prefill_seq={args.prefill_seq_len}  context={args.context_len}")
        print(f"{'='*60}")

        metrics, sm_timeline = run_eval(
            model_name=args.model,
            policy=policy,
            runner=runner,
            decode_batch_size=args.decode_batch_size,
            prefill_seq_len=args.prefill_seq_len,
            context_len=args.context_len,
            total_sm=total_sm,
            n_warmup_steps=args.n_warmup_steps,
        )

        # Summary
        if metrics.prefill_ttft_ms:
            print(f"\n  Results:")
            print(f"    TTFT mean={np.mean(metrics.prefill_ttft_ms):.1f}ms  "
                  f"p99={np.percentile(metrics.prefill_ttft_ms, 99):.1f}ms")
        if metrics.decode_tpot_ms:
            print(f"    TPOT p50={np.percentile(metrics.decode_tpot_ms, 50):.1f}ms  "
                  f"p99={np.percentile(metrics.decode_tpot_ms, 99):.1f}ms")
        print(f"    SLO violations={metrics.slo_violations}/{metrics.n_decode_steps} "
              f"({metrics.slo_violations/max(metrics.n_decode_steps,1):.1%})")

        save_results(
            metrics=metrics,
            sm_timeline=sm_timeline,
            model_name=args.model,
            policy_name=policy_key,
            prefill_seq_len=args.prefill_seq_len,
            decode_batch_size=args.decode_batch_size,
            hw_cfg=hw_cfg,
            output_dir=output_dir,
        )
