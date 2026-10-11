#!/usr/bin/env python3
"""M2g tier G3a -- two concurrent NCCL communicators issued in REVERSED order
across ranks, on plain vs green-context streams.  SKELETON: never executed
(2026-10-12, engine-porter).  Not a test module; GPU only.

    torchrun --nproc-per-node N m2g_comm_reverse_stress.py \
        --repeats R --timeout-s T --streams {plain,green} --split P D --out DIR

WHY.  With PD-mux at TP>1 each rank holds two TP GroupCoordinators
(`_TP` for decode, `_PDMUX_PREFILL_TP_GROUP` for prefill; parallel_state.py
`initialize_model_parallel(duplicate_tp_group=True)`), each with its own
PyNccl communicator (force-enabled under PD-mux) and its own custom-AR comm.
PyNccl launches on the CURRENT stream (pynccl.py `_resolve_stream`), so the
prefill all-reduce runs on the prefill green-context stream and the decode
all-reduce on the decode one.  The host issue order is the same on every rank
under the legacy loop, but the GPU execution order of the two streams is not
ordered by anything -- rank 0 may run its prefill collective first while
rank 1 runs its decode collective first.  NCCL collectives on different
communicators that are not consistently ordered across ranks can deadlock when
one kernel waits for peers while occupying resources the other needs.  This
script forces the worst case (even ranks A->B, odd ranks B->A) many times and
asks two questions: does it hang, and does the green-context split (disjoint
SM sets) change that.  It mirrors the engine's comm objects (PyNccl over a
gloo group, as GroupCoordinator builds them) rather than torch.distributed
NCCL, whose collectives run on ProcessGroupNCCL's internal stream and would
NOT be confined by the green context.

NOT covered (say so in any result): custom-AR v1 comms (IPC buffers; add as a
second arm via CustomAllreduce once G4 shows which sizes take CA), cudagraph
replay, the real model.  A PASS here does not imply the engine is hang-free;
a FAIL does imply the engine has a reachable hang.

PASS criterion (REGISTER in the M2.5 PREREG): all ranks finish R repeats
within T seconds and every reduced tensor equals the exact integer sum.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time


def _watchdog(timeout_s: float, rank: int, out_dir: str) -> threading.Timer:
    def fire():
        try:
            with open(os.path.join(out_dir, f"HANG.rank{rank}.json"), "w") as fh:
                json.dump({"rank": rank, "timeout_s": timeout_s, "t": time.time()}, fh)
        finally:
            os._exit(3)  # a hung NCCL kernel cannot be interrupted any other way

    timer = threading.Timer(timeout_s, fire)
    timer.daemon = True
    timer.start()
    return timer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, required=True)
    ap.add_argument("--timeout-s", type=float, required=True)
    ap.add_argument("--streams", choices=("plain", "green"), required=True)
    ap.add_argument("--split", type=int, nargs=2, metavar=("PREFILL_SM", "DECODE_SM"), default=(64, 44))
    ap.add_argument("--numel", type=int, nargs=2, metavar=("PREFILL_NUMEL", "DECODE_NUMEL"),
                    default=(8 * 1024 * 1024, 4096 * 4))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    import torch.distributed as dist
    from sglang.srt.distributed.device_communicators.pynccl import PyNcclCommunicator

    dist.init_process_group(backend="gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local)
    device = torch.device(f"cuda:{local}")
    os.makedirs(args.out, exist_ok=True)

    # Two gloo groups over the same ranks -> two NCCL communicators, as
    # GroupCoordinator does for "tp" and "pdmux_prefill_tp".
    ranks = list(range(world))
    comm_a = PyNcclCommunicator(dist.new_group(ranks, backend="gloo"), device)  # "prefill"
    comm_b = PyNcclCommunicator(dist.new_group(ranks, backend="gloo"), device)  # "decode"
    comm_a.disabled = False  # the PD-mux override (parallel_state.py)
    comm_b.disabled = False

    if args.streams == "green":
        from sgl_kernel import spatial

        stream_a, stream_b = spatial.create_greenctx_stream_by_value(args.split[0], args.split[1], local)
    else:
        stream_a, stream_b = torch.cuda.Stream(local), torch.cuda.Stream(local)

    na, nb = args.numel
    timer = _watchdog(args.timeout_s, rank, args.out)
    errors = 0
    t0 = time.perf_counter()
    for it in range(args.repeats):
        # integer-valued payloads: the exact sum is known and order-independent
        xa = torch.full((na,), float(rank + 1), device=device, dtype=torch.float32)
        xb = torch.full((nb,), float(2 * rank + 1), device=device, dtype=torch.float32)
        order = ("a", "b") if rank % 2 == 0 else ("b", "a")
        for which in order:
            stream, comm, x = (stream_a, comm_a, xa) if which == "a" else (stream_b, comm_b, xb)
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                comm.all_reduce(x)
        stream_a.synchronize()
        stream_b.synchronize()
        want_a = float(world * (world + 1) // 2)
        want_b = float(world * world)
        errors += int(not bool((xa == want_a).all())) + int(not bool((xb == want_b).all()))
    elapsed = time.perf_counter() - t0
    timer.cancel()
    with open(os.path.join(args.out, f"rank{rank}.json"), "w") as fh:
        json.dump({"rank": rank, "world": world, "streams": args.streams, "split": args.split,
                   "repeats": args.repeats, "errors": errors, "elapsed_s": elapsed,
                   "nccl_env": {k: v for k, v in os.environ.items() if k.startswith(("NCCL_", "SGLANG_", "CUDA_DEVICE_MAX"))}},
                  fh, indent=1, sort_keys=True)
    dist.barrier()
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
