"""
_decode_worker.py — subprocess worker for the decode SM-saturation sweep (Task 1).

Replicates the subprocess-isolation pattern of stage1_sm_scaling/_ssm_worker.py:
one SM level per subprocess.  A CUDA OOM or context-corrupting kernel error kills
only this process; the parent (run_decode_sweep.py) starts a fresh subprocess for
the next SM level.

stdout: one JSON object per measured (cell, scope, batch, context) row, flushed
        immediately.  Includes status ∈ {ok, OOM, CUDA_ERROR}.
stderr: human-readable status lines (visible in parent log).

Measurement protocol (per task spec)
------------------------------------
* decode kernels are 수~수십 μs → 단발 측정 불가.  K=64 step 을 event 쌍 사이에서
  연속 실행하고 per-step latency = elapsed/K.  warmup 16 step, 반복 5회 median.
* CUDA Graph 금지 (Green Context stream 전환 비호환).
* 모든 커널은 `with torch.cuda.stream(smctrl.get_stream()):` 블록 안에서 실행.

Cells × scope
-------------
              scope=kernel                       scope=layer
decode_ssm    selective_state_update             in_proj + conv update + state update + out_proj
decode_attn   attention decode (q_len=1, KV=L)   QKV proj + attn + O proj

Backend note (decode_attn)
--------------------------
layer_runner 의 attention 백엔드는 FlashInfer→SDPA fallback 이다.  FlashInfer 의
배치 decode 는 paged KV 를 요구하는데 본 측정은 연속(contiguous) KV 를 쓰므로
(spec: paged 불필요), layer_runner 가 이미 쓰는 fallback 백엔드인 SDPA 로
attention 을 계산한다.  **새 백엔드를 도입하지 않는다.**  사용 백엔드는 CSV 의
backend 컬럼에 기록한다.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_here = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_here, "..", "decode_crossover"))  # compute_crossover
sys.path.insert(0, os.path.join(_here, ".."))                       # src.*

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from compute_crossover import MODELS, BYTES_PER_ELEM  # single source of byte formulas
from src.smctrl.green_ctx_controller import SMController

DEVICE = "cuda"
DTYPE = torch.bfloat16
THEORETICAL_BW_GBS = 2000.0  # A100-SXM4-80GB HBM2e peak


# ---------------------------------------------------------------------------
# Core per-step latency measurement (K steps between one event pair)
# ---------------------------------------------------------------------------


def measure_per_step(fn, stream, K: int, warmup: int, repeats: int) -> float:
    """Return median per-step latency (ms) over `repeats` event-pair windows.

    elapsed over K consecutive launches, divided by K.  No CUDA graph.
    """
    samples = []
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        for _ in range(repeats):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record(stream)
            for _ in range(K):
                fn()
            e.record(stream)
            torch.cuda.synchronize()
            samples.append(s.elapsed_time(e) / K)
    return float(np.median(samples))


# ---------------------------------------------------------------------------
# decode_ssm builders
# ---------------------------------------------------------------------------


def build_decode_ssm_kernel(spec, bs: int):
    """selective_state_update single-token state update only.

    Tensor shapes mirror Mamba2.step()'s selective_state_update call.
    """
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update

    nheads = spec.ssm_n_heads
    headdim = spec.ssm_head_dim
    dstate = spec.ssm_d_state
    ngroups = spec.ssm_n_groups

    ssm_state = torch.randn(bs, nheads, headdim, dstate, device=DEVICE, dtype=DTYPE)
    x = torch.randn(bs, nheads, headdim, device=DEVICE, dtype=DTYPE)
    dt = torch.rand(bs, nheads, headdim, device=DEVICE, dtype=DTYPE)
    A = -torch.rand(nheads, headdim, dstate, device=DEVICE, dtype=torch.float32)
    B = torch.randn(bs, ngroups, dstate, device=DEVICE, dtype=DTYPE)
    C = torch.randn(bs, ngroups, dstate, device=DEVICE, dtype=DTYPE)
    D = torch.ones(nheads, headdim, device=DEVICE, dtype=DTYPE)
    z = torch.randn(bs, nheads, headdim, device=DEVICE, dtype=DTYPE)
    dt_bias = torch.zeros(nheads, headdim, device=DEVICE, dtype=DTYPE)

    def fn():
        with torch.no_grad():
            selective_state_update(
                ssm_state, x, dt, A, B, C, D, z=z,
                dt_bias=dt_bias, dt_softplus=True,
            )

    bytes_analytic = spec.state_bytes(bs)  # recurrent state read+write
    return fn, bytes_analytic, None, "triton_selective_state_update"


def build_decode_ssm_layer(spec, bs: int):
    """Full SSM decode step: in_proj + conv update + selective_state_update + out_proj.

    Reconstructs Mamba2.step() math with standalone nn.Linear/conv weights (no HF
    checkpoint needed — shapes match config).  d_mlp assumed 0 (true for Zamba2 /
    Falcon-H1).
    """
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
    try:
        from causal_conv1d import causal_conv1d_update
    except Exception:
        causal_conv1d_update = None

    H = spec.hidden_size
    nheads = spec.ssm_n_heads
    headdim = spec.ssm_head_dim
    dstate = spec.ssm_d_state
    ngroups = spec.ssm_n_groups
    d_conv = spec.ssm_d_conv
    d_ssm = spec.ssm_d_inner                       # = nheads*headdim
    conv_dim = d_ssm + 2 * ngroups * dstate        # xBC channels
    in_proj_out = 2 * d_ssm + 2 * ngroups * dstate + nheads

    in_proj = nn.Linear(H, in_proj_out, bias=False, dtype=DTYPE).to(DEVICE)
    out_proj = nn.Linear(d_ssm, H, bias=False, dtype=DTYPE).to(DEVICE)
    conv_w = torch.randn(conv_dim, d_conv, device=DEVICE, dtype=DTYPE)
    conv_b = torch.zeros(conv_dim, device=DEVICE, dtype=DTYPE)

    conv_state = torch.zeros(bs, conv_dim, d_conv, device=DEVICE, dtype=DTYPE)
    ssm_state = torch.randn(bs, nheads, headdim, dstate, device=DEVICE, dtype=DTYPE)
    hidden = torch.randn(bs, 1, H, device=DEVICE, dtype=DTYPE)

    A = -torch.rand(nheads, headdim, dstate, device=DEVICE, dtype=torch.float32)
    D = torch.ones(nheads, headdim, device=DEVICE, dtype=DTYPE)
    dt_bias = torch.zeros(nheads, headdim, device=DEVICE, dtype=DTYPE)

    def fn():
        with torch.no_grad():
            zxbcdt = in_proj(hidden.squeeze(1))
            z, xBC, dt = torch.split(
                zxbcdt, [d_ssm, conv_dim, nheads], dim=-1
            )
            # conv update (single step)
            if causal_conv1d_update is not None:
                xBC = causal_conv1d_update(xBC, conv_state, conv_w, conv_b, "silu")
            else:
                conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))
                conv_state[:, :, -1] = xBC
                xBC = torch.sum(conv_state * conv_w, dim=-1) + conv_b
                xBC = F.silu(xBC).to(DTYPE)
            x, B, C = torch.split(
                xBC, [d_ssm, ngroups * dstate, ngroups * dstate], dim=-1
            )
            dt_r = dt.unsqueeze(-1).expand(bs, nheads, headdim)
            B_r = B.view(bs, ngroups, dstate)
            C_r = C.view(bs, ngroups, dstate)
            x_r = x.view(bs, nheads, headdim)
            z_r = z.view(bs, nheads, headdim)
            y = selective_state_update(
                ssm_state, x_r, dt_r, A, B_r, C_r, D, z=z_r,
                dt_bias=dt_bias, dt_softplus=True,
            )
            y = y.reshape(bs, d_ssm)
            out_proj(y)

    bytes_analytic = spec.state_bytes(bs) + spec.conv_state_bytes(bs)
    bytes_w_weight = bytes_analytic + spec.ssm_weight_bytes()
    return fn, bytes_analytic, bytes_w_weight, "triton_ssu+linear"


# ---------------------------------------------------------------------------
# decode_attn builders (SDPA — layer_runner's fallback backend; no new backend)
# ---------------------------------------------------------------------------


def _gqa_expand(t: torch.Tensor, n_heads: int, n_kv_heads: int) -> torch.Tensor:
    if n_heads == n_kv_heads:
        return t
    return t.repeat_interleave(n_heads // n_kv_heads, dim=1)


def build_decode_attn_kernel(spec, bs: int, L: int):
    """Attention decode kernel only: q_len=1 over a pre-allocated KV cache of length L.

    KV cache is a contiguous tensor [bs, L, n_kv_heads, head_dim] (paged 불필요).
    """
    n_heads = spec.attn_num_heads
    n_kv = spec.attn_num_kv_heads
    hd = spec.attn_head_dim

    q = torch.randn(bs, n_heads, 1, hd, device=DEVICE, dtype=DTYPE)
    k = torch.randn(bs, n_kv, L, hd, device=DEVICE, dtype=DTYPE)
    v = torch.randn(bs, n_kv, L, hd, device=DEVICE, dtype=DTYPE)

    def fn():
        with torch.no_grad():
            kk = _gqa_expand(k, n_heads, n_kv)
            vv = _gqa_expand(v, n_heads, n_kv)
            F.scaled_dot_product_attention(q, kk, vv, is_causal=False)

    bytes_analytic = spec.kv_bytes(L, bs)  # KV read dominates
    return fn, bytes_analytic, None, "sdpa"


def build_decode_attn_layer(spec, bs: int, L: int):
    """Full attn decode step: QKV proj (1 token) + attn over L + O proj."""
    H = spec.hidden_size
    n_heads = spec.attn_num_heads
    n_kv = spec.attn_num_kv_heads
    hd = spec.attn_head_dim
    attn_hidden = n_heads * hd
    kv_hidden = n_kv * hd

    w_q = nn.Linear(H, attn_hidden, bias=False, dtype=DTYPE).to(DEVICE)
    w_k = nn.Linear(H, kv_hidden, bias=False, dtype=DTYPE).to(DEVICE)
    w_v = nn.Linear(H, kv_hidden, bias=False, dtype=DTYPE).to(DEVICE)
    w_o = nn.Linear(attn_hidden, H, bias=False, dtype=DTYPE).to(DEVICE)

    hidden = torch.randn(bs, 1, H, device=DEVICE, dtype=DTYPE)
    ctx_k = torch.randn(bs, n_kv, L, hd, device=DEVICE, dtype=DTYPE)
    ctx_v = torch.randn(bs, n_kv, L, hd, device=DEVICE, dtype=DTYPE)

    def fn():
        with torch.no_grad():
            q = w_q(hidden).view(bs, 1, n_heads, hd).transpose(1, 2)
            # k_new/v_new for the single new token are computed but the L-length
            # context dominates; we attend q to the pre-built length-L KV cache.
            w_k(hidden)
            w_v(hidden)
            kk = _gqa_expand(ctx_k, n_heads, n_kv)
            vv = _gqa_expand(ctx_v, n_heads, n_kv)
            out = F.scaled_dot_product_attention(q, kk, vv, is_causal=False)
            out = out.transpose(1, 2).reshape(bs, 1, attn_hidden)
            w_o(out)

    bytes_analytic = spec.kv_bytes(L, bs)
    bytes_w_weight = bytes_analytic + spec.attn_weight_bytes()
    return fn, bytes_analytic, bytes_w_weight, "sdpa"


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_BUILDERS = {
    ("decode_ssm", "kernel"): lambda spec, bs, L: build_decode_ssm_kernel(spec, bs),
    ("decode_ssm", "layer"): lambda spec, bs, L: build_decode_ssm_layer(spec, bs),
    ("decode_attn", "kernel"): build_decode_attn_kernel,
    ("decode_attn", "layer"): build_decode_attn_layer,
}


def _emit(rec: dict) -> None:
    print(json.dumps(rec), flush=True)


# ---------------------------------------------------------------------------
# Worker main
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--model", required=True, choices=list(MODELS))
    p.add_argument("--sm-count", type=int, required=True)
    p.add_argument("--total-sm", type=int, required=True)
    p.add_argument("--cells", nargs="+", default=["decode_ssm", "decode_attn"])
    p.add_argument("--scopes", nargs="+", default=["kernel", "layer"])
    p.add_argument("--batches", nargs="+", type=int, required=True)
    p.add_argument("--contexts", nargs="+", type=int, required=True,
                   help="decode_attn context lengths; decode_ssm uses L=0 single row")
    p.add_argument("--K", type=int, default=64)
    p.add_argument("--warmup", type=int, default=16)
    p.add_argument("--repeats", type=int, default=5)
    args = p.parse_args()

    spec = MODELS[args.model]

    smctrl = SMController(
        device_id=0,
        total_sm_count=args.total_sm,
        preset_sm_counts=[14, 27, 40, 54, 68, 81, 94, 108],
    )
    smctrl.set_sm_count(args.sm_count)
    stream = smctrl.get_stream()

    cuda_dead = False

    for cell in args.cells:
        # decode_ssm is L-independent → single L=0 row; decode_attn sweeps contexts
        ctx_grid = [0] if cell == "decode_ssm" else list(args.contexts)
        for scope in args.scopes:
            for bs in args.batches:
                for L in ctx_grid:
                    base = {
                        "model": args.model,
                        "cell": cell,
                        "scope": scope,
                        "sm_count": args.sm_count,
                        "batch": bs,
                        "context_len": L,
                    }
                    if cuda_dead:
                        _emit({**base, "status": "CUDA_ERROR",
                               "note": "prior CUDA context error"})
                        continue
                    try:
                        builder = _BUILDERS[(cell, scope)]
                        fn, b_analytic, b_w_weight, backend = builder(spec, bs, L)
                        lat_ms = measure_per_step(
                            fn, stream, K=args.K,
                            warmup=args.warmup, repeats=args.repeats,
                        )
                        lat_s = lat_ms / 1e3
                        achieved_bw = (b_analytic / 1e9) / lat_s if lat_s > 0 else float("nan")
                        bw_util = achieved_bw / THEORETICAL_BW_GBS * 100.0
                        rec = {
                            **base,
                            "latency_per_step_ms": lat_ms,
                            "bytes_analytic": int(b_analytic),
                            "bytes_analytic_w_weight": (
                                int(b_w_weight) if b_w_weight is not None else ""
                            ),
                            "achieved_bw_GBs": achieved_bw,
                            "bw_util_pct": bw_util,
                            "backend": backend,
                            "status": "ok",
                        }
                        _emit(rec)
                        print(
                            f"  OK  {cell:11s} {scope:6s} sm={args.sm_count:3d} "
                            f"bs={bs:3d} L={L:6d} lat={lat_ms*1e3:8.2f}us "
                            f"bw={achieved_bw:7.1f}GB/s ({bw_util:5.1f}%)",
                            file=sys.stderr,
                        )
                        # free per-config tensors before next config
                        del fn
                        torch.cuda.empty_cache()
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        _emit({**base, "status": "OOM"})
                        print(f"  OOM {cell} {scope} sm={args.sm_count} bs={bs} L={L}",
                              file=sys.stderr)
                    except Exception as e:
                        err = str(e)
                        is_cuda = (
                            "CUDA error" in err
                            or "illegal memory" in err.lower()
                            or "device-side assert" in err.lower()
                        )
                        _emit({**base, "status": "CUDA_ERROR" if is_cuda else "ERROR",
                               "note": err[:160]})
                        print(f"  ERR {cell} {scope} sm={args.sm_count} bs={bs} L={L}: "
                              f"{err[:140]}", file=sys.stderr)
                        if is_cuda:
                            cuda_dead = True


if __name__ == "__main__":
    main()
