"""
kernels.py — weight-free kernel builders + the canonical layer-type dispatch.

Why weight-free
---------------
The v2 SLMs (Zamba2-1.2B / Falcon-H1-1.5B) are NOT in hf_cache, and the v1
LayerRunner extractors are wired to the 7B HF weights. The v2 prefill/decode
characterization only needs kernel *shapes* (n_heads, head_dim, d_state, …), so
these builders allocate scan/attn/GEMM tensors directly from the model config and
call the same kernels the real model uses (mamba_chunk_scan_combined, SDPA/
FlashInfer, cuBLAS GEMM). No HF weights required — matching the approach already
used by archive/.../coexec_microbench.py::_build_prefill_fn.

Single dispatch (kills the B-1 bug class)
-----------------------------------------
``build_prefill_fn(layer_type, ...)`` is the ONE place that maps a layer type to
its kernel. The v1 concurrent runner always built an SSM prefill regardless of
layer type (the "B-1 dispatch" bug). E2/E4 build prefill kernels only through
this function, and E4 unit-tests that an "attn" request actually runs the
attention kernel and an "ssm" request the SSM kernel.

src is reused, never modified: LatencyMeter / BandwidthEstimator come from
src.profiling.metrics; SM control from src.smctrl.
"""

from __future__ import annotations

import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_CHAR = os.path.abspath(os.path.join(_here, "..", ".."))
_WORKSPACE = os.path.abspath(os.path.join(_here, "..", "..", ".."))
for _p in (_WORKSPACE, _CHAR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shared.loaders import get_model_config
from experiments.common import wave_model as wm  # single source for n_blocks

# Optional heavy deps — imported lazily so this module imports on CPU-only boxes.
try:
    import torch
    import torch.nn.functional as F
    _HAS_TORCH = True
except Exception:                      # pragma: no cover
    _HAS_TORCH = False

VALID_PREFILL_TYPES = ("ssm", "attn", "ssm_full", "attn_full")
VALID_DECODE_TYPES = ("ssm", "attn")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_layer_cfg(model_name: str) -> dict:
    """Flatten models.yaml into the shape the builders need (from config only)."""
    raw = get_model_config(model_name)
    ssm = raw.get("ssm", {})
    attn = raw.get("attention", {})
    return {
        "model": model_name,
        "config_status": raw.get("config_status", "verified"),
        "d_model": raw["hidden_size"],
        # SSM
        "n_heads": ssm["n_heads"],
        "head_dim": ssm["head_dim"],
        "d_state": ssm["d_state"],
        "chunk_size": ssm["chunk_size"],     # internal SSD chunk
        "n_groups": ssm.get("n_groups", 1),
        "expand": ssm.get("expand", 2),
        # Attention
        "n_attn_heads": attn.get("num_heads", 32),
        "n_attn_kv_heads": attn.get("num_kv_heads", attn.get("num_heads", 32)),
        "attn_head_dim": attn.get("head_dim", 128),
    }


def _require_torch() -> None:
    if not _HAS_TORCH:
        raise RuntimeError("torch unavailable — kernels require a CUDA build of torch.")


def _import_mamba():
    try:
        from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
        return mamba_chunk_scan_combined
    except Exception:
        return None


# ---------------------------------------------------------------------------
# SSM scan (token_mixer for SSM layers)
# ---------------------------------------------------------------------------

def build_ssm_scan_fn(cfg: dict, batch: int, tokens: int, device: str = "cuda",
                      dtype="bfloat16") -> tuple:
    """Chunked SSM scan over ``tokens`` tokens per call. Returns (fn, rb, wb, meta).

    fn() calls mamba_chunk_scan_combined with the model's internal ssd chunk.
    read/write byte counts come from BandwidthEstimator.ssm_scan_bytes (the
    estimator's own accounting, so bw_probe agrees).
    """
    _require_torch()
    from experiments.common.bw_probe import ssm_scan_bytes
    dt = getattr(torch, dtype)
    nH, hd, dst, ng, ssd = (cfg["n_heads"], cfg["head_dim"], cfg["d_state"],
                            cfg["n_groups"], cfg["chunk_size"])

    x   = torch.randn(batch, tokens, nH, hd, dtype=dt, device=device)
    dtt = torch.full((batch, tokens, nH), 0.1, dtype=dt, device=device)
    A   = -torch.ones(nH, dtype=dt, device=device)
    B   = torch.randn(batch, tokens, ng, dst, dtype=dt, device=device)
    C   = torch.randn(batch, tokens, ng, dst, dtype=dt, device=device)
    D   = torch.ones(nH, dtype=dt, device=device)
    dtb = torch.zeros(nH, dtype=dt, device=device)
    mamba = _import_mamba()

    def fn():
        if mamba is None:
            raise RuntimeError("mamba_ssm not installed — cannot measure SSM scan.")
        mamba(x, dtt, A, B, C, chunk_size=ssd, D=D, dt_bias=dtb, dt_softplus=True)

    rb, wb = ssm_scan_bytes(batch=batch, seq_len=tokens, n_heads=nH,
                            head_dim=hd, d_state=dst, n_groups=ng)
    # n_blocks via the single source (wave_model), not a local formula.
    meta = {"n_blocks_per_call": wm.n_blocks(batch, tokens, nH, ssd),
            "kernel": "ssm_scan"}
    return fn, rb, wb, meta


# ---------------------------------------------------------------------------
# GEMM (in_proj / out_proj for the SSM-layer decomposition)
# ---------------------------------------------------------------------------

def _build_gemm_fn(M: int, K: int, N: int, device: str, dtype) -> tuple:
    _require_torch()
    dt = getattr(torch, dtype)
    x = torch.randn(M, K, dtype=dt, device=device)
    w = torch.randn(K, N, dtype=dt, device=device)

    def fn():
        torch.matmul(x, w)

    bpe = 2
    rb = (M * K + K * N) * bpe      # read activations + weights
    wb = (M * N) * bpe              # write output
    return fn, rb, wb


def build_in_proj_fn(cfg, batch, tokens, device="cuda", dtype="bfloat16") -> tuple:
    """in_proj GEMM: (batch*tokens, hidden) @ (hidden, 2*inner)."""
    inner = cfg["n_heads"] * cfg["head_dim"]
    fn, rb, wb = _build_gemm_fn(batch * tokens, cfg["d_model"], 2 * inner, device, dtype)
    return fn, rb, wb, {"kernel": "in_proj"}


def build_out_proj_fn(cfg, batch, tokens, device="cuda", dtype="bfloat16") -> tuple:
    """out_proj GEMM: (batch*tokens, inner) @ (inner, hidden)."""
    inner = cfg["n_heads"] * cfg["head_dim"]
    fn, rb, wb = _build_gemm_fn(batch * tokens, inner, cfg["d_model"], device, dtype)
    return fn, rb, wb, {"kernel": "out_proj"}


def build_attn_qkv_proj_fn(cfg, batch, tokens, device="cuda", dtype="bfloat16") -> tuple:
    """attn qkv_proj GEMM: (batch*tokens, d_model) @ (d_model, (nH+2*nKV)*head_dim)."""
    nH, nKV, hd = cfg["n_attn_heads"], cfg["n_attn_kv_heads"], cfg["attn_head_dim"]
    fn, rb, wb = _build_gemm_fn(batch * tokens, cfg["d_model"], (nH + 2 * nKV) * hd, device, dtype)
    return fn, rb, wb, {"kernel": "qkv_proj"}


def build_attn_o_proj_fn(cfg, batch, tokens, device="cuda", dtype="bfloat16") -> tuple:
    """attn o_proj GEMM: (batch*tokens, nH*head_dim) @ (nH*head_dim, d_model)."""
    nH, hd = cfg["n_attn_heads"], cfg["attn_head_dim"]
    fn, rb, wb = _build_gemm_fn(batch * tokens, nH * hd, cfg["d_model"], device, dtype)
    return fn, rb, wb, {"kernel": "o_proj"}


# ---------------------------------------------------------------------------
# Attention (token_mixer for attn layers); context_len varies the KV grid
# ---------------------------------------------------------------------------

def build_attn_fn(cfg, batch, seq, context_len=0, device="cuda", dtype="bfloat16",
                  use_flashinfer=False) -> tuple:
    """Prefill attention over ``seq`` query tokens with ``context_len`` KV history.

    Uses SDPA by default (always available); FlashInfer path is left to E-runner
    discretion. Byte count from BandwidthEstimator.attn_bytes.
    """
    _require_torch()
    from experiments.common.bw_probe import attn_bytes
    dt = getattr(torch, dtype)
    nH, nKV, hd = cfg["n_attn_heads"], cfg["n_attn_kv_heads"], cfg["attn_head_dim"]
    total_kv = context_len + seq

    q = torch.randn(batch, nH, seq, hd, dtype=dt, device=device)
    k = torch.randn(batch, nKV, total_kv, hd, dtype=dt, device=device)
    v = torch.randn(batch, nKV, total_kv, hd, dtype=dt, device=device)
    if nKV != nH:
        rep = nH // nKV
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)

    def fn():
        F.scaled_dot_product_attention(q, k, v, is_causal=(context_len == 0))

    rb, wb = attn_bytes(batch=batch, seq_len=seq, total_kv_len=total_kv,
                        n_heads=nH, n_kv_heads=nKV, head_dim=hd,
                        hidden_size=cfg["d_model"])
    # FlashAttn grid block count via the same single source: one CTA per
    # (batch, attn-head, query-block of BLOCK_M≈64). wave_model.n_blocks is
    # generic — feed it attn n_heads and chunk=BLOCK_M.
    meta = {"kernel": "attn", "context_len": context_len,
            "n_blocks_per_call": wm.n_blocks(batch, seq, nH, chunk=64)}
    return fn, rb, wb, meta


# ---------------------------------------------------------------------------
# Decode kernels (E3): SSM seq=1, Attn KV read
# ---------------------------------------------------------------------------

def build_decode_ssm_fn(cfg, batch, device="cuda", dtype="bfloat16") -> tuple:
    """SSM decode step (seq_len=1)."""
    return build_ssm_scan_fn(cfg, batch=batch, tokens=1, device=device, dtype=dtype)


def build_decode_attn_fn(cfg, batch, context_len, device="cuda", dtype="bfloat16") -> tuple:
    """Attn decode step: 1 query token reading ``context_len`` KV (memory-bound)."""
    return build_attn_fn(cfg, batch=batch, seq=1, context_len=context_len,
                         device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# THE prefill dispatch — single source of truth (B-1 fix lives here)
# ---------------------------------------------------------------------------

def build_prefill_fn(layer_type: str, cfg: dict, batch: int, tokens: int,
                     context_len: int = 0, device: str = "cuda",
                     dtype: str = "bfloat16") -> tuple:
    """Dispatch a prefill kernel by layer type. Returns (fn, rb, wb, meta).

    layer_type:
      "ssm"       → SSM scan only (token mixer).
      "attn"      → attention only (token mixer); context_len sets KV history.
      "ssm_full"  → in_proj + scan + out_proj (full SSM layer; GEMM-inclusive).
      "attn_full" → qkv_proj + SDPA + o_proj (full attention layer; GEMM-inclusive).

    The "_full" variants include the projection GEMMs and therefore approximate a
    real prefill layer (compute-bound, SM-filling) rather than the memory-bound
    token-mixer microbench. Used by the E5 optional real-prefill mode (A2).

    This is the only correct place to choose the prefill kernel. Anything that
    builds prefill work MUST go through here so an "attn" config can never
    silently run an SSM kernel (the B-1 dispatch bug).
    """
    if layer_type not in VALID_PREFILL_TYPES:
        raise ValueError(f"layer_type {layer_type!r} not in {VALID_PREFILL_TYPES}")

    if layer_type == "ssm":
        return build_ssm_scan_fn(cfg, batch, tokens, device, dtype)

    if layer_type == "attn":
        return build_attn_fn(cfg, batch, tokens, context_len, device, dtype)

    if layer_type == "attn_full":
        # qkv_proj + SDPA(token mixer) + o_proj into one callable.
        qkv_fn, qkv_rb, qkv_wb, _ = build_attn_qkv_proj_fn(cfg, batch, tokens, device, dtype)
        mix_fn, mix_rb, mix_wb, mix_meta = build_attn_fn(cfg, batch, tokens, context_len, device, dtype)
        o_fn, o_rb, o_wb, _ = build_attn_o_proj_fn(cfg, batch, tokens, device, dtype)

        def fn():
            qkv_fn()
            mix_fn()
            o_fn()

        rb = qkv_rb + mix_rb + o_rb
        wb = qkv_wb + mix_wb + o_wb
        meta = {"kernel": "attn_full", "context_len": context_len,
                "n_blocks_per_call": mix_meta["n_blocks_per_call"]}
        return fn, rb, wb, meta

    # ssm_full: compose in_proj + scan + out_proj into one callable.
    in_fn, in_rb, in_wb, _ = build_in_proj_fn(cfg, batch, tokens, device, dtype)
    sc_fn, sc_rb, sc_wb, sc_meta = build_ssm_scan_fn(cfg, batch, tokens, device, dtype)
    out_fn, out_rb, out_wb, _ = build_out_proj_fn(cfg, batch, tokens, device, dtype)

    def fn():
        in_fn()
        sc_fn()
        out_fn()

    rb = in_rb + sc_rb + out_rb
    wb = in_wb + sc_wb + out_wb
    meta = {"kernel": "ssm_full", "n_blocks_per_call": sc_meta["n_blocks_per_call"]}
    return fn, rb, wb, meta
