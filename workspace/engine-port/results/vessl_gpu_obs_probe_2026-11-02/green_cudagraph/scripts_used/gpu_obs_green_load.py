#!/usr/bin/env python3
"""Green-context + cudagraph load for gpu_obs_green_probe.sh. Each arm is one NVTX range `arm:<name>`.
Not a measurement of serving: no model, no engine. Writes expected kernel counts and arm wall times to --out."""
import argparse
import json
import os
import sys
import threading
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
RESULT = {"arms": {}, "errors": {}}


def arm(name, expected, fn):
    """expected = kernel count the trace must show in this arm (None = not asserted, e.g. triton lazy compile)."""
    t0 = time.perf_counter()
    try:
        with torch.cuda.nvtx.range(f"arm:{name}"):
            fn()
            torch.cuda.synchronize()
        RESULT["arms"][name] = {"elapsed_s": time.perf_counter() - t0, "expected_kernels": expected}
    except Exception as exc:  # noqa: BLE001
        RESULT["errors"][name] = repr(exc)[:400]


def gemm_loop(a, b, stream, n):
    with torch.cuda.stream(stream):
        for _ in range(n):
            torch.matmul(a, b)
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--p", type=int, default=16)
    ap.add_argument("--d", type=int, default=92)
    ap.add_argument("--size", type=int, default=2048)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--replays", type=int, default=50)
    ap.add_argument("--small-launches", type=int, default=2000)
    args = ap.parse_args()

    from sgl_kernel.spatial import create_greenctx_stream_by_value

    import green_readout as gr
    import graph_census as gc

    n = args.iters
    a = torch.randn(args.size, args.size, device="cuda")
    b = torch.randn(args.size, args.size, device="cuda")
    plain = torch.cuda.Stream()
    sp, sd = create_greenctx_stream_by_value(args.p, args.d, 0)
    RESULT["readout"] = {k: gr.green_sm_readout(int(s.cuda_stream)) for k, s in
                         (("plain", plain), ("prefill", sp), ("decode", sd))}
    RESULT["split"] = [args.p, args.d]
    RESULT["replays"] = args.replays

    # Warm-up outside any arm: cuBLAS autotune and module load must not land in a counted arm.
    for s in (plain, sp, sd):
        gemm_loop(a, b, s, 2)

    arm("plain_eager", n, lambda: gemm_loop(a, b, plain, n))
    arm("green_p_eager", n, lambda: gemm_loop(a, b, sp, n))
    arm("green_d_eager", n, lambda: gemm_loop(a, b, sd, n))

    def concurrent():
        ts = [threading.Thread(target=gemm_loop, args=(a, b, s, n)) for s in (sp, sd)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
    arm("green_concurrent", 2 * n, concurrent)

    def body():
        x = torch.matmul(a, b)
        return torch.matmul(x, b).add_(1.0)

    for label, stream in (("d", sd), ("p", sp)):
        RESULT[f"census_{label}"] = [row["node_kind"] for row in gc.capture_census(body, stream)]
        kernels = RESULT[f"census_{label}"].count("kernel")
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=stream):
            body()
        torch.cuda.synchronize()

        def replay(g=g, stream=stream):
            with torch.cuda.stream(stream):
                for _ in range(args.replays):
                    g.replay()
                torch.cuda.synchronize()
        arm(f"graph_replay_{label}", args.replays * kernels, replay)
        RESULT[f"kernels_per_replay_{label}"] = kernels

    s = torch.ones(64, device="cuda")

    def small():
        with torch.cuda.stream(sd):
            for _ in range(args.small_launches):
                s.add_(1.0)
            torch.cuda.synchronize()
    arm("small_launch_heavy", args.small_launches, small)

    try:
        import triton
        import triton.language as tl

        @triton.jit
        def add_k(x, y, out, n_elem, BLOCK: tl.constexpr):
            pid = tl.program_id(0)
            off = pid * BLOCK + tl.arange(0, BLOCK)
            mask = off < n_elem
            tl.store(out + off, tl.load(x + off, mask=mask) + tl.load(y + off, mask=mask), mask=mask)

        v = torch.ones(1 << 20, device="cuda")
        w = torch.zeros(1 << 20, device="cuda")

        def tri(k):
            with torch.cuda.stream(sd):
                for _ in range(k):
                    add_k[(v.numel() // 1024 + 1,)](v, v, w, v.numel(), BLOCK=1024)
                torch.cuda.synchronize()
        tri(1)
        arm("triton_eager", n, lambda: tri(n))
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=sd):
            tri_args = (v, v, w, v.numel())
            add_k[(v.numel() // 1024 + 1,)](*tri_args, BLOCK=1024)
        torch.cuda.synchronize()

        def tri_replay():
            with torch.cuda.stream(sd):
                for _ in range(args.replays):
                    g.replay()
                torch.cuda.synchronize()
        arm("triton_graph_replay", args.replays, tri_replay)
    except Exception as exc:  # noqa: BLE001
        RESULT["errors"]["triton_setup"] = repr(exc)[:400]

    with open(args.out, "w") as f:
        json.dump(RESULT, f, indent=1)
    print(json.dumps({k: RESULT[k] for k in ("readout", "errors")}, default=str))


if __name__ == "__main__":
    main()
