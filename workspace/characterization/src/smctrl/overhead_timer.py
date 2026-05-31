"""
SM reconfiguration overhead timer — Green Contexts backend.

Measures the CPU-side and total latency of SM partition switching via
CUDA Green Contexts.  With Green Contexts, set_sm_ratio() is a Python
dict lookup (< 1 μs); the dominant overhead at layer boundaries is
torch.cuda.synchronize() draining any in-flight work before the stream
switch takes full effect.

Three measurement modes:
  measure_single_transition  — A→B switch with/without device sync
  measure_n_transitions      — n consecutive layer-boundary switches
  measure_cold_start_penalty — first-kernel latency after stream switch
"""

import time
import numpy as np
import torch
from typing import Optional
from .green_ctx_controller import SMController


class SMOverheadTimer:
    """Measures Green Context stream-switch overhead."""

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
        """Measure a single SM-partition switch (A → B) latency.

        With Green Contexts, set_sm_ratio() is a CPU-side pointer swap.
        include_sync=True adds torch.cuda.synchronize() after the switch,
        which is the realistic layer-boundary cost (drain in-flight work,
        then hand off to the new stream).

        Args:
            from_ratio: Source SM ratio.
            to_ratio:   Target SM ratio.
            include_sync: If True, include torch.cuda.synchronize() cost.
            n_warmup:  Discarded warm-up iterations.
            n_measure: Measurement iterations.

        Returns:
            dict with latency statistics in microseconds.
        """
        dummy = torch.zeros(1024, device="cuda")
        samples = []

        for i in range(n_warmup + n_measure):
            # Set up source context and ensure a kernel is in-flight
            self.smctrl.set_sm_ratio(from_ratio)
            with torch.cuda.stream(self.smctrl.get_stream()):
                dummy.fill_(0.0)
            if include_sync:
                torch.cuda.synchronize()

            t0 = time.perf_counter_ns()
            self.smctrl.set_sm_ratio(to_ratio)
            if include_sync:
                # Sync ensures the new stream is the active one before next kernel
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
        """Simulate n layer-boundary stream switches and measure total overhead.

        Each layer uses the Green Context stream matching its SM ratio.
        A minimal kernel (dummy.add_) is enqueued on each layer's stream
        to model realistic usage.  Total time is measured with device sync
        around the full sequence.

        Args:
            n_layers:   Number of layer boundaries to simulate.
            ssm_ratio:  SM ratio for SSM layers.
            attn_ratio: SM ratio for Attention layers.
            n_warmup:   Discarded warm-up runs.
            n_measure:  Measurement runs.

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
                with torch.cuda.stream(self.smctrl.get_stream()):
                    dummy.add_(1.0)

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
        """Measure if the first kernel on a new Green Context stream has extra latency.

        Compares:
          (a) baseline GEMM on the current full-SM stream
          (b) same GEMM after switching to a restricted-SM stream

        With Green Contexts, cold-start penalty is typically ~0 μs because
        stream switching is handled by the GPU scheduler asynchronously.

        Returns:
            dict with baseline_us, post_reconfig_us, penalty_us.
        """
        x = torch.randn(kernel_size, kernel_size, device="cuda", dtype=torch.float16)

        baseline_samples = []
        post_reconfig_samples = []

        for i in range(n_warmup + n_measure):
            # Baseline: full-SM stream, no switch
            self.smctrl.reset()
            stream_full = self.smctrl.get_stream()
            torch.cuda.synchronize()
            t0 = time.perf_counter_ns()
            with torch.cuda.stream(stream_full):
                _ = torch.mm(x, x)
            torch.cuda.synchronize()
            t1 = time.perf_counter_ns()
            if i >= n_warmup:
                baseline_samples.append((t1 - t0) / 1_000.0)

            # Post-switch: switch to ratio stream, then run first kernel
            self.smctrl.reset()
            torch.cuda.synchronize()
            self.smctrl.set_sm_ratio(ratio)
            stream_new = self.smctrl.get_stream()
            t0 = time.perf_counter_ns()
            with torch.cuda.stream(stream_new):
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
