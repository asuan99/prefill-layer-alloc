#!/usr/bin/env python3
"""Small CUDA matmul load for gpu_obs_probe.sh / profile_workflow.sh smoke runs (not a measurement)."""
import argparse
import json
import time

import torch


def load(iters, size):
    a = torch.randn(size, size, device="cuda")
    b = torch.randn(size, size, device="cuda")
    for i in range(iters):
        with torch.cuda.nvtx.range(f"matmul_{i % 4}"):
            a = torch.matmul(a, b)
            a = a / a.norm()
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--size", type=int, default=4096)
    ap.add_argument("--torch-profiler", metavar="TRACE_JSON")
    args = ap.parse_args()

    print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_device_name(0))
    t0 = time.perf_counter()
    if args.torch_profiler:
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            load(args.iters, args.size)
        prof.export_chrome_trace(args.torch_profiler)
        with open(args.torch_profiler) as f:
            events = json.load(f).get("traceEvents", [])
        kernels = [e for e in events if e.get("cat") == "kernel"]
        print("trace_events", len(events), "kernel_events", len(kernels))
        for e in kernels[:5]:
            print("kernel", e.get("name"), "dur_us", e.get("dur"))
    else:
        load(args.iters, args.size)
    print("elapsed_s", round(time.perf_counter() - t0, 3))


if __name__ == "__main__":
    main()
