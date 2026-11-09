"""ops_spec.py — operator inventory of one Nemotron-H layer of each type (CPU only).

opchar campaign (plan: ~/.claude/plans/figure2-table2-linear-cherny.md, approved
2026-10-12): Bullet (ASPLOS'26) Fig 2 / Table 2 / Fig 3 rebuilt for a hybrid model.

This module is the single place that says WHICH operators one layer runs and WHAT
their analytical work is. It has no torch dependency so the regression suite can pin
it. ``builders.py`` (GPU) must build exactly the ops listed here — the worker checks
the two name lists are equal before timing anything.

The operator list mirrors SGLang v0.5.10 (sglang/srt/models/nemotron_h.py and
sglang/srt/layers/attention/mamba/mamba.py::MambaMixer2.forward):

  every layer   norm        RMSNorm with residual (sgl_kernel fused_add_rmsnorm)
  M (Mamba2)    in_proj     GEMM  hidden -> [z | x,B,C | dt]
                conv1d      causal depthwise conv over x,B,C (+SiLU)       prefill
                conv_update one-step conv state update                     decode
                ssd_scan    mamba_chunk_scan_combined (+ ssm_state write)  prefill
                ssm_update  selective_state_update                         decode
                gated_norm  Mixer2RMSNormGated (n_groups=8 -> torch native path)
                out_proj    GEMM  mamba_inner -> hidden
  * (Attn)      qkv_proj    GEMM  hidden -> q,k,v (no RoPE in Nemotron-H)
                kv_write    KV-cache store (_set_kv_buffer_impl)
                attn        FlashInfer ragged prefill (causal) / paged decode
                o_proj      GEMM
  - (MLP)       up_proj     GEMM  hidden -> intermediate
                relu2       relu(x)^2 (two torch elementwise kernels)
                down_proj   GEMM

Analytical FLOPs / bytes (ALL DERIVED) follow one stated convention so that the
Fig 2(b)(c) "utilization" bars are reproducible, not hardware counters:
  * FLOPs: 2 per multiply-add; elementwise ops count their arithmetic per element.
  * bytes: COMPULSORY traffic — every input tensor read once, every output written
    once, weights read once; no cache-reuse or re-read modelling. SSD chunk states
    (fp32) are counted as written once and read once.
A different convention moves the bars; the convention string travels with the data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

BYTES_CONVENTION = "compulsory-v1"   # see module docstring
BF16 = 2
FP32 = 4

CATEGORIES = ("gemm", "ssm_scan", "conv1d", "attn", "norm", "elementwise", "kv_write")
LAYER_CHARS = ("M", "*", "-")


@dataclass(frozen=True)
class OpSpec:
    name: str
    layer: str            # "M" | "*" | "-"
    category: str         # one of CATEGORIES
    flops: float          # DERIVED
    bytes_read: float     # DERIVED (compulsory)
    bytes_write: float    # DERIVED (compulsory)
    shape: Dict[str, int] = field(default_factory=dict)

    @property
    def bytes_total(self) -> float:
        return self.bytes_read + self.bytes_write

    @property
    def key(self) -> str:
        return f"{self.layer}:{self.name}"


@dataclass(frozen=True)
class HybridCfg:
    model: str
    hidden: int
    pattern: str
    n_heads: int          # mamba heads
    head_dim: int         # mamba head dim
    d_state: int
    n_groups: int
    chunk: int
    conv_k: int
    attn_heads: int
    attn_kv_heads: int
    attn_head_dim: int
    mlp_inter: int
    rms_eps: float = 1e-5

    @property
    def mamba_inner(self) -> int:
        return self.n_heads * self.head_dim

    @property
    def conv_dim(self) -> int:
        return self.mamba_inner + 2 * self.n_groups * self.d_state

    @property
    def in_proj_out(self) -> int:
        # MambaMixer2.in_proj: [gate z (inner) | x,B,C (conv_dim) | dt (n_heads)]
        return self.mamba_inner + self.conv_dim + self.n_heads

    @property
    def qkv_out(self) -> int:
        return (self.attn_heads + 2 * self.attn_kv_heads) * self.attn_head_dim


def cfg_from_models_yaml(name: str) -> HybridCfg:
    """Build the cfg from workspace/shared/configs/models.yaml (verified entries only)."""
    import os
    import sys
    ws = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    if ws not in sys.path:
        sys.path.insert(0, ws)
    from shared.loaders import get_model_config
    raw = get_model_config(name)
    if raw.get("config_status") != "verified":
        raise ValueError(f"{name}: config_status={raw.get('config_status')!r}, need 'verified'")
    s, a = raw["ssm"], raw["attention"]
    return HybridCfg(
        model=name, hidden=raw["hidden_size"], pattern=raw["layer_pattern"],
        n_heads=s["n_heads"], head_dim=s["head_dim"], d_state=s["d_state"],
        n_groups=s["n_groups"], chunk=s["chunk_size"], conv_k=s.get("conv_kernel", 4),
        attn_heads=a["num_heads"], attn_kv_heads=a["num_kv_heads"],
        attn_head_dim=a["head_dim"], mlp_inter=raw["mlp"]["intermediate_size"],
        rms_eps=raw.get("rms_norm_eps", 1e-5))


def layer_counts(pattern: str) -> Dict[str, int]:
    bad = set(pattern) - set(LAYER_CHARS)
    if bad:
        raise ValueError(f"pattern has unsupported layer chars {sorted(bad)} (MoE 'E' not modelled)")
    return {c: pattern.count(c) for c in LAYER_CHARS}


# ---------------------------------------------------------------------------
# building blocks
# ---------------------------------------------------------------------------

def _gemm(name, layer, m, k, n) -> OpSpec:
    return OpSpec(name, layer, "gemm", flops=2.0 * m * k * n,
                  bytes_read=(m * k + k * n) * BF16, bytes_write=m * n * BF16,
                  shape={"m": m, "k": k, "n": n})


def _norm(layer, t, h) -> OpSpec:
    # fused_add_rmsnorm: reads x and residual, writes both back in place.
    return OpSpec("norm", layer, "norm", flops=5.0 * t * h,
                  bytes_read=2 * t * h * BF16 + h * BF16, bytes_write=2 * t * h * BF16,
                  shape={"tokens": t, "hidden": h})


def _gated_norm(c: HybridCfg, t) -> OpSpec:
    i = c.mamba_inner
    return OpSpec("gated_norm", "M", "norm", flops=10.0 * t * i,
                  bytes_read=2 * t * i * BF16 + i * BF16, bytes_write=t * i * BF16,
                  shape={"tokens": t, "inner": i, "groups": c.n_groups})


def _kv_write(c: HybridCfg, t) -> OpSpec:
    kv = 2 * t * c.attn_kv_heads * c.attn_head_dim * BF16
    return OpSpec("kv_write", "*", "kv_write", flops=0.0, bytes_read=kv, bytes_write=kv,
                  shape={"tokens": t})


def _mlp(c: HybridCfg, t) -> List[OpSpec]:
    h, i = c.hidden, c.mlp_inter
    return [_norm("-", t, h),
            _gemm("up_proj", "-", t, h, i),
            OpSpec("relu2", "-", "elementwise", flops=2.0 * t * i,
                   bytes_read=t * i * BF16, bytes_write=t * i * BF16, shape={"tokens": t, "inner": i}),
            _gemm("down_proj", "-", t, i, h)]


# ---------------------------------------------------------------------------
# prefill: one request of L tokens, no prefix (Bullet Fig 2 axis = prompt length)
# ---------------------------------------------------------------------------

def ssd_scan_work(c: HybridCfg, L: int) -> OpSpec:
    """mamba_chunk_scan_combined over L tokens (DERIVED).

    FLOPs (SSD, Dao & Gu 2024 decomposition, per head h, chunk size Q, state N, head dim P):
      chunk_state  B^T x            2·L·N·P
      chunk_scan   C·state           2·L·N·P
      chunk_scan   (CB∘mask)·x       2·L·Q·P        (intra-chunk, ≈ half masked; counted full)
      bmm          C·B^T per group   2·L·Q·N   (× n_groups, not × heads)
    Bytes (compulsory): x, dt, B, C read; y written; fp32 chunk states written+read;
    final per-request ssm_state (fp32) written.
    """
    nH, P, N, Q, G = c.n_heads, c.head_dim, c.d_state, c.chunk, c.n_groups
    nchunks = -(-L // Q)
    flops = nH * (4.0 * L * N * P + 2.0 * L * Q * P) + 2.0 * L * Q * N * G
    rd = (L * nH * P + L * nH + 2 * L * G * N) * BF16
    states = nchunks * nH * P * N * FP32
    wr = L * nH * P * BF16 + states + nH * P * N * FP32
    return OpSpec("ssd_scan", "M", "ssm_scan", flops=flops, bytes_read=rd + states,
                  bytes_write=wr, shape={"tokens": L, "chunks": nchunks})


def prefill_ops(c: HybridCfg, L: int) -> Dict[str, List[OpSpec]]:
    if L < 1:
        raise ValueError("L must be >= 1")
    h, t = c.hidden, L
    conv = OpSpec("conv1d", "M", "conv1d", flops=2.0 * t * c.conv_dim * c.conv_k,
                  bytes_read=t * c.conv_dim * BF16 + c.conv_dim * (c.conv_k + 1) * BF16,
                  bytes_write=t * c.conv_dim * BF16 + c.conv_dim * (c.conv_k - 1) * BF16,
                  shape={"tokens": t, "channels": c.conv_dim})
    nH, nKV, hd = c.attn_heads, c.attn_kv_heads, c.attn_head_dim
    attn = OpSpec("attn", "*", "attn",
                  # causal: QK^T and PV each 2·L²·nH·hd, halved by the mask
                  flops=2.0 * L * L * nH * hd,
                  bytes_read=(L * nH * hd + 2 * L * nKV * hd) * BF16,
                  bytes_write=L * nH * hd * BF16, shape={"tokens": L})
    return {
        "M": [_norm("M", t, h), _gemm("in_proj", "M", t, h, c.in_proj_out), conv,
              ssd_scan_work(c, L), _gated_norm(c, t), _gemm("out_proj", "M", t, c.mamba_inner, h)],
        "*": [_norm("*", t, h), _gemm("qkv_proj", "*", t, h, c.qkv_out), _kv_write(c, t), attn,
              _gemm("o_proj", "*", t, nH * hd, h)],
        "-": _mlp(c, t),
    }


# ---------------------------------------------------------------------------
# decode: B requests, one token each, every request has ctx tokens of KV / state
# ---------------------------------------------------------------------------

def decode_ops(c: HybridCfg, B: int, ctx: int) -> Dict[str, List[OpSpec]]:
    if B < 1 or ctx < 1:
        raise ValueError("B and ctx must be >= 1")
    h, t = c.hidden, B
    nH, P, N = c.n_heads, c.head_dim, c.d_state
    conv_state = B * c.conv_dim * (c.conv_k - 1) * BF16
    conv_u = OpSpec("conv_update", "M", "conv1d", flops=2.0 * B * c.conv_dim * c.conv_k,
                    bytes_read=B * c.conv_dim * BF16 + conv_state + c.conv_dim * (c.conv_k + 1) * BF16,
                    bytes_write=B * c.conv_dim * BF16 + conv_state, shape={"batch": B})
    ssm_state = B * nH * P * N * FP32
    ssm_u = OpSpec("ssm_update", "M", "ssm_scan", flops=6.0 * B * nH * P * N,
                   bytes_read=ssm_state + (B * nH * P + B * nH + 2 * B * c.n_groups * N) * BF16,
                   bytes_write=ssm_state + B * nH * P * BF16, shape={"batch": B})
    aH, nKV, hd = c.attn_heads, c.attn_kv_heads, c.attn_head_dim
    attn = OpSpec("attn", "*", "attn", flops=4.0 * B * ctx * aH * hd,
                  bytes_read=(B * aH * hd + 2 * B * ctx * nKV * hd) * BF16,
                  bytes_write=B * aH * hd * BF16, shape={"batch": B, "ctx": ctx})
    return {
        "M": [_norm("M", t, h), _gemm("in_proj", "M", t, h, c.in_proj_out), conv_u, ssm_u,
              _gated_norm(c, t), _gemm("out_proj", "M", t, c.mamba_inner, h)],
        "*": [_norm("*", t, h), _gemm("qkv_proj", "*", t, h, c.qkv_out), _kv_write(c, t), attn,
              _gemm("o_proj", "*", t, aH * hd, h)],
        "-": _mlp(c, t),
    }


def op_keys(ops: Dict[str, List[OpSpec]]) -> List[str]:
    return [o.key for layer in LAYER_CHARS for o in ops.get(layer, [])]


def model_shares(per_op_time: Dict[str, float], counts: Dict[str, int]) -> Dict[str, float]:
    """Model-level time share of each op key (DERIVED).

    One layer of each type is timed; the model runs ``counts[layer]`` copies, so an op's
    model time is time × count of its layer type. Shares sum to 1 over the given keys.
    """
    total = 0.0
    weighted = {}
    for key, t in per_op_time.items():
        layer = key.split(":", 1)[0]
        w = t * counts[layer]
        weighted[key] = w
        total += w
    if total <= 0:
        raise ValueError("total model time must be positive")
    return {k: v / total for k, v in weighted.items()}
