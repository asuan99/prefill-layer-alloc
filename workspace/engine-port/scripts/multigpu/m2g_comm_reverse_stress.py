#!/usr/bin/env python3
"""M2g G3 -- two concurrent PyNccl communicators issued in REVERSED order across ranks,
on plain vs green-context streams (PREREG_MULTIGPU_2026-10-12.md rev6 sec 3 G3).

GPU ONLY; never executed as of 2026-10-12 (engine-porter).  Not a test module.

    python -m torch.distributed.run --standalone --nproc-per-node N m2g_comm_reverse_stress.py \
        --streams {plain,green} [--split P D] --out DIR          # R and watchdog default to the registered values

The Job launches it ONLY through `m2g_correctness_gate.run_g3` / `run-g3` (F1, 2026-10-12): preflight
import of the launcher, outer timeout with process-group kill, per-combination launcher.json and
stdout/stderr files, and "no rank file and no HANG file" scored NOT_TESTED(HARNESS_ERROR), not FAIL.

REGISTERED (sec 3 G3): R = 500 repeats per combination, watchdog 120 s, combinations
= plain streams and green streams at the two splits 16/92 and 44/64, for every
TP = N in {2, 4, 8}.  PASS <=> hang 0 AND integer-sum all-reduce check errors 0.
The split pairs are passed to `create_greenctx_stream_by_value(SM_a, SM_b)` in
the order written in the PREREG ((16, 92) and (44, 64)): stream A ("prefill"
comm) gets the first count.  Both are disjoint partitions of 108 SM -- which is
the property under test -- whichever side is called prefill.

WATCHDOG SEMANTICS (harness reading of "watchdog 120 s", flagged for the harness
audit): a PROGRESS watchdog -- it fires when no iteration completes for 120 s
(the timer is re-armed after every iteration) and the rank writes
HANG.rank{r}.json and `os._exit(3)` (a hung NCCL kernel cannot be interrupted
otherwise).  A slow-but-progressing run is therefore not a hang; its total
elapsed time is recorded.

WHY.  With PD-mux at TP>1 each rank holds two TP GroupCoordinators (`_TP` for
decode, `_PDMUX_PREFILL_TP_GROUP` for prefill; parallel_state.py
`initialize_model_parallel(duplicate_tp_group=True)`), each with its own PyNccl
communicator (force-enabled under PD-mux).  PyNccl launches on the CURRENT
stream, so the prefill all-reduce runs on the prefill green-context stream and
the decode all-reduce on the decode one; nothing orders the two streams' GPU
execution across ranks.  This forces the worst case (even ranks A->B, odd ranks
B->A) R times.  It mirrors the engine's comm objects (PyNccl over a gloo group)
rather than torch.distributed NCCL, whose collectives run on ProcessGroupNCCL's
internal stream and would NOT be confined by the green context.

NOT covered (say so in any result): custom all-reduce v1 (IPC buffers), cudagraph
replay, the real model.  A PASS does not imply the engine is hang-free; a FAIL
does imply a reachable hang (or a wrong reduction) in the comm layer.

Scoring is `m2g_correctness_gate.score_g3(out_dir, world, repeats)` (CPU).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time

G3_REPEATS = 500  # PREREG rev6 sec 3 G3: "반복 R = 500/조합"
G3_WATCHDOG_S = 120.0  # PREREG rev6 sec 3 G3: "watchdog 120 s"
G3_GREEN_SPLITS = ((16, 92), (44, 64))  # PREREG rev6 sec 3 G3: "green stream(16/92, 44/64)"


class ProgressWatchdog:
    """Fires when `kick()` has not been called for `timeout_s`."""

    def __init__(self, timeout_s: float, on_fire):
        self.timeout_s = float(timeout_s)
        self.on_fire = on_fire
        self._timer = None
        self._lock = threading.Lock()

    def kick(self):
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self.timeout_s, self.on_fire)
            self._timer.daemon = True
            self._timer.start()

    def cancel(self):
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None


def expected_sums(world: int):
    """Exact all-reduce results of the payloads below: sum(r+1) and sum(2r+1)."""
    return float(world * (world + 1) // 2), float(world * world)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=G3_REPEATS)
    ap.add_argument("--timeout-s", type=float, default=G3_WATCHDOG_S)
    ap.add_argument("--streams", choices=("plain", "green"), required=True)
    ap.add_argument("--split", type=int, nargs=2, metavar=("SM_A", "SM_B"), default=None,
                    help="green only; must be one of the registered splits")
    ap.add_argument("--numel", type=int, nargs=2, metavar=("A_NUMEL", "B_NUMEL"),
                    default=(8 * 1024 * 1024, 4096 * 4))
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.streams == "green":
        if args.split is None or tuple(args.split) not in G3_GREEN_SPLITS:
            ap.error(f"--split must be one of {G3_GREEN_SPLITS} for green streams")
    elif args.split is not None:
        ap.error("--split applies to green streams only")

    import torch
    import torch.distributed as dist
    from sglang.srt.distributed.device_communicators.pynccl import PyNcclCommunicator

    dist.init_process_group(backend="gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local)
    device = torch.device(f"cuda:{local}")
    os.makedirs(args.out, exist_ok=True)
    progress = {"completed": 0}

    def fire():
        try:
            with open(os.path.join(args.out, f"HANG.rank{rank}.json"), "w") as fh:
                json.dump({"rank": rank, "timeout_s": args.timeout_s, "completed": progress["completed"],
                           "t": time.time()}, fh)
        finally:
            os._exit(3)

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
    want_a, want_b = expected_sums(world)
    wd = ProgressWatchdog(args.timeout_s, fire)
    errors = 0
    first_error = None
    t0 = time.perf_counter()
    wd.kick()
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
        bad = int(not bool((xa == want_a).all())) + int(not bool((xb == want_b).all()))
        if bad and first_error is None:
            first_error = it
        errors += bad
        progress["completed"] = it + 1
        wd.kick()
    wd.cancel()
    elapsed = time.perf_counter() - t0
    with open(os.path.join(args.out, f"rank{rank}.json"), "w") as fh:
        json.dump({"rank": rank, "world": world, "streams": args.streams,
                   "split": list(args.split) if args.split else None,
                   "repeats": args.repeats, "completed": progress["completed"], "errors": errors,
                   "first_error_iteration": first_error, "elapsed_s": elapsed,
                   "watchdog_s": args.timeout_s, "numel": [na, nb],
                   "nccl_env": {k: v for k, v in os.environ.items()
                                if k.startswith(("NCCL_", "SGLANG_", "CUDA_DEVICE_MAX"))}},
                  fh, indent=1, sort_keys=True)
    dist.barrier()
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
