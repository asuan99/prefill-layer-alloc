"""
SM reconfiguration overhead timer.

Provides precise CPU-side timing of libsmctrl SM mask changes using
perf_counter_ns, bracketed by torch.cuda.synchronize() calls to ensure
the measured interval captures the full reconfiguration delay including
any subsequent kernel launch stall.
"""

import time
import numpy as np
import torch
from typing import Optional
from .libsmctrl_wrapper import SMController


class SMOverheadTimer:
    """Measures SM reconfiguration overhead with CUDA-event precision."""

    def __init__(self, smctrl: Optional[SMController] = None):
        self.smctrl = smctrl or SMController()

    def measure_single_transition(
        self,
        from_ratio: float,
        to_ratio: float,
        include_sync: bool = True,
        n_warmup: int = 50,
        n_measure: int = 200,
    ) -> dict:
        """Measure a single SM mask transition (A → B) latency.

        Args:
            from_ratio: Source SM ratio.
            to_ratio: Target SM ratio.
            include_sync: If True, include torch.cuda.synchronize() overhead.
            n_warmup: Discarded warm-up iterations.
            n_measure: Measurement iterations.

        Returns:
            dict with latency statistics in microseconds.
        """
        dummy = torch.zeros(1024, device="cuda")
        samples = []

        for i in range(n_warmup + n_measure):
            self.smctrl.set_sm_ratio(from_ratio)
            dummy.fill_(0.0)
            if include_sync:
                torch.cuda.synchronize()

            t0 = time.perf_counter_ns()
            self.smctrl.set_sm_ratio(to_ratio)
            if include_sync:
                torch.cuda.synchronize()
            t1 = time.perf_counter_ns()

            if i >= n_warmup:
                samples.append((t1 - t0) / 1_000.0)

        arr = np.array(samples)
        return {
            "mean_us": float(arr.mean()),
            "median_us": float(np.median(arr)),
            "p99_us": float(np.percentile(arr, 99)),
            "min_us": float(arr.min()),
            "max_us": float(arr.max()),
            "std_us": float(arr.std()),
            "include_sync": include_sync,
            "from_ratio": from_ratio,
            "to_ratio": to_ratio,
        }

    def measure_n_transitions(
        self,
        n_layers: int,
        ssm_ratio: float = 0.7,
        attn_ratio: float = 0.4,
        n_warmup: int = 20,
        n_measure: int = 100,
    ) -> dict:
        """Simulate n layer-boundary transitions and measure total overhead.

        Models a hybrid model forward pass where SM ratio alternates between
        SSM (prefill-heavy) and Attention layers for n_layers total.

        Args:
            n_layers: Number of layer boundaries to simulate.
            ssm_ratio: SM ratio used for SSM layers.
            attn_ratio: SM ratio used for Attention layers.
            n_warmup: Discarded warm-up runs.
            n_measure: Measurement runs.

        Returns:
            dict with total and per-transition overhead statistics (μs).
        """
        dummy = torch.zeros(1024, device="cuda")
        total_samples = []

        for i in range(n_warmup + n_measure):
            torch.cuda.synchronize()
            t0 = time.perf_counter_ns()

            for layer_idx in range(n_layers):
                ratio = ssm_ratio if layer_idx % 2 == 0 else attn_ratio
                self.smctrl.set_sm_ratio(ratio)
                dummy.add_(1.0)  # minimal kernel to simulate layer execution

            torch.cuda.synchronize()
            t1 = time.perf_counter_ns()

            if i >= n_warmup:
                total_samples.append((t1 - t0) / 1_000.0)

        arr = np.array(total_samples)
        return {
            "n_layers": n_layers,
            "total_mean_us": float(arr.mean()),
            "total_p99_us": float(np.percentile(arr, 99)),
            "per_transition_mean_us": float(arr.mean() / n_layers),
            "per_transition_p99_us": float(np.percentile(arr, 99) / n_layers),
            "ssm_ratio": ssm_ratio,
            "attn_ratio": attn_ratio,
        }

    def measure_cold_start_penalty(
        self,
        ratio: float,
        kernel_size: int = 4096,
        n_warmup: int = 20,
        n_measure: int = 100,
    ) -> dict:
        """Measure if the first kernel after SM reconfiguration has extra latency.

        Compares: (a) baseline kernel with no reconfiguration vs
                  (b) first kernel after SM mask change.

        Returns:
            dict with baseline_us, post_reconfig_us, penalty_us.
        """
        x = torch.randn(kernel_size, kernel_size, device="cuda", dtype=torch.float16)

        baseline_samples = []
        post_reconfig_samples = []

        for i in range(n_warmup + n_measure):
            # Baseline: no reconfiguration
            torch.cuda.synchronize()
            t0 = time.perf_counter_ns()
            _ = torch.mm(x, x)
            torch.cuda.synchronize()
            t1 = time.perf_counter_ns()
            if i >= n_warmup:
                baseline_samples.append((t1 - t0) / 1_000.0)

            # Post-reconfig: first kernel after SM mask change
            self.smctrl.set_sm_ratio(1.0)
            torch.cuda.synchronize()
            self.smctrl.set_sm_ratio(ratio)
            t0 = time.perf_counter_ns()
            _ = torch.mm(x, x)
            torch.cuda.synchronize()
            t1 = time.perf_counter_ns()
            if i >= n_warmup:
                post_reconfig_samples.append((t1 - t0) / 1_000.0)

        b_arr = np.array(baseline_samples)
        p_arr = np.array(post_reconfig_samples)
        penalty = p_arr.mean() - b_arr.mean()

        return {
            "baseline_mean_us": float(b_arr.mean()),
            "post_reconfig_mean_us": float(p_arr.mean()),
            "cold_start_penalty_us": float(penalty),
            "penalty_ratio": float(penalty / b_arr.mean()) if b_arr.mean() > 0 else 0.0,
            "sm_ratio": ratio,
        }
