#!/usr/bin/env python3
"""GPU gate (c)-2 for the Triton causal_conv1d int64 patch: one kernel call vs a torch reference.

Decision 15 / AUDIT_probe_128k_2026-10-20.md item 2 experiment 2.  Criteria revised 2026-10-20 (decision 20).

One CASE per process (an illegal memory access poisons the CUDA context, so the orchestrator runs
each case in its own subprocess and records crash = nonzero exit / signal):

  conv1d_int64_unit.py --kernel-file <causal_conv1d_triton.py> --tokens T [--layout view|contig]
                       [--width-inproj 22656] [--dim 12288] [--offset 10240] [--dtype bf16] [--seed 1]
                       [--device cuda] [--out case.json]

  layout=view    x = buf[:, offset:offset+dim].T of a (T, width_inproj) buffer -> stride_x_token =
                 width_inproj (NemotronH's non-contiguous xBC view, 22,656 for Nano-9B-v2)
  layout=contig  the same values copied to a contiguous (dim, T)ᵀ-layout tensor -> stride = dim

Checks (JSON `verdict`; criteria revised 2026-10-20 per decision 20, AFTER seeing G-beta round-1 data -- see
GATE_SPLIT_DESIGN sec 10): this process judges only what is decidable in ONE kernel call:
  state    conv_states[cache_line] after the call == the last (width-1) input tokens, EXACT  -> verdict RAN / STATE_MISMATCH
  digest   sha256 of the output bytes (`out_sha256`)  -- the ORCHESTRATOR compares it with the original (pristine) kernel's
           contig run at the same T (MATCH = equal sha; BROKEN = differs / crash / state mismatch)
  out      max |diff| vs a torch fp32 reference (+ atol): DESCRIPTIVE ONLY (`out_ok`).  Round 1 showed max diff 0.04-0.057 at EVERY
           valid T (bf16 rounding of an fp32 reference), so atol was not a usable criterion.
Exit: 0 = RAN (state exact), 1 = STATE_MISMATCH, 2 = input error.  A crash never reaches the exit code (the caller sees it).

Prediction on A100: pristine+view output sha == pristine+contig sha for T <= 94,784 (stride 22,656; 189,572 for stride 11,328),
!= or crash above; patched+view sha == pristine+contig sha at every T (contig stride 12,288 wraps only past 174,762);
chunk_offset-only mutant + view at 94,790 -> crash / sha != patched contig / state mismatch.
CPU self-test: TRITON_INTERPRET=1 with --device cpu and small T (tests/test_conv1d_int64_patch.py).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ATOL = {"bf16": 2e-2, "fp16": 4e-3, "fp32": 1e-4}


def load_kernel(path):
    spec = importlib.util.spec_from_file_location("conv1d_kernel_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.causal_conv1d_fn


def build(T, width_inproj, dim, offset, dtype, seed, device, layout, kw=4, cache_lines=2):
    import torch
    g = torch.Generator(device="cpu").manual_seed(seed)
    dt = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
    buf = torch.empty((T, width_inproj), dtype=dt, device=device)
    # fill only the xBC slice (the rest is never read); generate in row blocks to bound host memory
    step = 4096
    for s in range(0, T, step):
        e = min(T, s + step)
        buf[s:e, offset:offset + dim] = torch.randn((e - s, dim), generator=g).to(dt).to(device)
    x = buf[:, offset:offset + dim].transpose(0, 1)  # (dim, T), stride (1, width_inproj)
    if layout == "contig":
        x = x.transpose(0, 1).contiguous().transpose(0, 1)  # (dim, T), stride (1, dim)
    w = (torch.randn((dim, kw), generator=g) * 0.5).to(dt).to(device)
    b = (torch.randn((dim,), generator=g) * 0.1).to(dt).to(device)
    conv_states = torch.zeros((cache_lines, dim, kw - 1), dtype=dt, device=device)
    qsl = torch.tensor([0, T], dtype=torch.int32, device=device)
    cache_idx = torch.tensor([1], dtype=torch.int32, device=device)
    has_init = torch.tensor([False], dtype=torch.bool, device=device)
    return x, w, b, conv_states, qsl, cache_idx, has_init


def reference(x, w, b, kw=4, block=8192):
    """fp32 causal depthwise conv + SiLU, token-blocked so 95k x 12,288 fits next to the inputs."""
    import torch
    import torch.nn.functional as F
    dim, T = x.shape
    out = torch.empty((dim, T), dtype=torch.float32, device=x.device)
    wf, bf = w.float().unsqueeze(1), b.float()
    for s in range(0, T, block):
        e = min(T, s + block)
        lo = max(0, s - (kw - 1))
        seg = x[:, lo:e].float().unsqueeze(0)
        pad = (kw - 1) - (s - lo)
        y = F.conv1d(F.pad(seg, (pad, 0)), wf, bf, groups=dim)[0]
        out[:, s:e] = y * torch.sigmoid(y)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--kernel-file", required=True)
    ap.add_argument("--tokens", type=int, required=True)
    ap.add_argument("--layout", choices=("view", "contig"), default="view")
    ap.add_argument("--width-inproj", type=int, default=22656)
    ap.add_argument("--dim", type=int, default=12288)
    ap.add_argument("--offset", type=int, default=10240)
    ap.add_argument("--dtype", choices=tuple(ATOL), default="bf16")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    if a.offset + a.dim > a.width_inproj or a.tokens < 4:
        print("CONV1D_UNIT_INPUT_ERROR: slice outside the in_proj width or T < 4", file=sys.stderr)
        return 2
    import torch
    fn = load_kernel(a.kernel_file)
    x, w, b, cs, qsl, ci, hi = build(a.tokens, a.width_inproj, a.dim, a.offset, a.dtype, a.seed, a.device, a.layout)
    out = fn(x, w, b, conv_states=cs, query_start_loc=qsl, seq_lens_cpu=[a.tokens], cache_indices=ci,
             has_initial_state=hi, activation="silu")
    if a.device.startswith("cuda"):
        torch.cuda.synchronize()
    ref = reference(x, w, b)
    diff = (out.float() - ref).abs()
    max_diff = float(diff.max())
    argmax_tok = int(diff.max(dim=0).values.argmax())
    state_ok = bool(torch.equal(cs[1], x[:, a.tokens - 3:]))
    out_ok = max_diff <= ATOL[a.dtype]          # descriptive only (decision 20): not part of the verdict
    doc = {"tokens": a.tokens, "layout": a.layout, "stride_x_token": int(x.stride(1)), "dtype": a.dtype,
           "kernel_file_sha256": hashlib.sha256(Path(a.kernel_file).read_bytes()).hexdigest(),
           "max_abs_diff": max_diff, "worst_token": argmax_tok, "atol": ATOL[a.dtype],
           "out_ok": out_ok, "state_ok": state_ok,
           "out_sha256": hashlib.sha256(out.contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest(),
           "verdict": "RAN" if state_ok else "STATE_MISMATCH"}
    text = json.dumps(doc, sort_keys=True)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8")
    return 0 if state_ok else 1


if __name__ == "__main__":
    sys.exit(main())
