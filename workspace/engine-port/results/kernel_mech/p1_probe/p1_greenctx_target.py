"""
P1 PROBE (kernel_mech Stage 0, item 1) -- CUPTI x Green Context resolution test.

★ NOT production code. This is a P1-only derivative of
workspace/characterization/src/profiling/_ncu_target.py -- created instead of
editing that file, per task instructions. Do not wire this into any
production sweep path.

Question this answers (see
workspace/engine-port/reports/EXPERIMENT_PLAN_2026-08-19.md §B and
workspace/engine-port/results/kernel_mech/DESIGN_KERNEL_MECH_REV2_2026-08-16.md
§7-0 item 1):

    Can `ncu` (Nsight Compute, hardware-counter mode) resolve a single CUDA
    kernel when that kernel is launched inside a CUDA Green Context stream
    (SM count restricted below the full GPU)?

_ncu_target.py:68-77 documents, as a code *comment* (not an independent
test), that CUPTI (the API ncu uses for HW counter collection) is believed
incompatible with Green Contexts, and that any process ncu attaches to must
never call smctrl.set_sm_count() -- the production script deliberately
avoids doing so and instead computes wave counts analytically. This script
deliberately does the opposite: it calls set_sm_count() and launches kernels
on the resulting Green Context stream while ncu is attached, to test the
claim directly rather than take the comment as ground truth (the CONSENSUS
registers 3 competing attributions for ncu "error code 9" -- do not assume
which one applies here without evidence).

Two modes, selected by --mode (this is the whole experimental design):

  greenctx : SMController(preset_sm_counts=[N]); smctrl.set_sm_count(N);
             kernel launched via `with torch.cuda.stream(smctrl.get_stream()):`
             -- i.e. actually SM-restricted. THE CONDITION UNDER TEST.

  full     : kernel launched on the default (unrestricted) current stream --
             no Green Context involved at all. Structurally identical
             otherwise. THE CONTROL CONDITION (mirrors exactly what the
             unmodified, unmodified-by-this-task _ncu_target.py already does
             for every layer_type -- it never touches the green-ctx stream).

Kernel: one bf16 GEMM (torch.matmul), compute-bound, single dominant kernel
launch, sized so that grid_size / sm_count is comfortably > 1 under the
restricted (greenctx) condition -- i.e. multiple waves, matching Stage 0
item 1's intent ("waves x SM_realized x CTAs_per_SM ~= grid_size").

Spawned by `ncu ... python p1_greenctx_target.py ...` (like _ncu_target.py is
spawned by NCURunner), or run bare (no ncu) for a cheap smoke check that
Green Context + the matmul kernel work at all before spending ncu-attached
GPU time.
"""

import argparse
import os
import sys

# workspace/characterization/ is where src.smctrl / src.models live. Insert
# it explicitly (absolute path) so this script works regardless of the cwd
# it's invoked from (ncu subprocess spawn does not inherit sys.path tricks).
_CHAR_ROOT = "/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/characterization"
sys.path.insert(0, _CHAR_ROOT)

import torch


def parse_args():
    p = argparse.ArgumentParser(
        description="P1 probe: does ncu resolve a kernel under Green Context SM restriction?"
    )
    p.add_argument("--mode", choices=["greenctx", "full"], required=True,
                    help="greenctx = condition under test (SM-restricted stream); "
                         "full = control (no Green Context, default stream)")
    p.add_argument("--sm-count", type=int, default=16,
                    help="Green Context SM count target (greenctx mode only)")
    p.add_argument("--matmul-dim", type=int, default=4096,
                    help="Square bf16 GEMM dimension (M=N=K)")
    p.add_argument("--n-warmup", type=int, default=10)
    p.add_argument("--n-measure", type=int, default=3)
    return p.parse_args()


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    from src.smctrl import SMController

    dim = args.matmul_dim
    a = torch.randn(dim, dim, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(dim, dim, device="cuda", dtype=torch.bfloat16)

    def kernel():
        with torch.no_grad():
            torch.matmul(a, b)

    smctrl = None
    if args.mode == "greenctx":
        # Explicit single preset so set_sm_count(N) snaps exactly to N
        # (avoids nearest-preset drift from the default 1/8-of-total ladder
        # in SMController._default_preset_counts()).
        smctrl = SMController(preset_sm_counts=[args.sm_count])
        realized = smctrl._actual_sm_counts.get(args.sm_count, "n/a")
        print(
            f"[p1] mode=greenctx backend={smctrl.get_backend_name()} "
            f"available={smctrl.is_available()} total_sm={smctrl.total_sm_count} "
            f"requested_sm={args.sm_count} realized_sm={realized}",
            flush=True,
        )
        if not smctrl.is_available():
            raise RuntimeError(
                "Green Context backend unavailable on this device/driver -- "
                "P1 cannot test the intended condition on this node."
            )
        smctrl.set_sm_count(args.sm_count)
        stream = smctrl.get_stream()
    else:
        stream = torch.cuda.current_stream()
        print("[p1] mode=full backend=none(control) sm_count=full_gpu", flush=True)

    print(
        f"[p1] kernel=bf16_matmul dim={dim} n_warmup={args.n_warmup} "
        f"n_measure={args.n_measure}",
        flush=True,
    )

    with torch.cuda.stream(stream):
        for _ in range(args.n_warmup):
            kernel()
    torch.cuda.synchronize()

    with torch.cuda.stream(stream):
        for _ in range(args.n_measure):
            kernel()
    torch.cuda.synchronize()

    if smctrl is not None:
        smctrl.reset()

    print("[p1] DONE", flush=True)


if __name__ == "__main__":
    main()
