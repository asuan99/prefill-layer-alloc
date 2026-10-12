"""builders.py — GPU callables for every operator in ops_spec, using the ENGINE's kernels.

Each layer type is built as a small "layer sim" that owns random weights and the real
intermediate tensors of one forward (same splits / views / strides as SGLang v0.5.10),
so every operator can be replayed alone, in isolation, with the exact input layouts the
engine hands it. Kernels come from the installed engine stack, never re-implemented:

  GEMM          torch.nn.functional.linear            (UnquantizedLinearMethod)
  norm          sgl_kernel.fused_add_rmsnorm           (RMSNorm.forward_cuda)
  conv1d        sglang ...mamba.causal_conv1d.causal_conv1d_fn  (+seq_lens_cpu kwarg, as
                MambaMixer2 passes it -> the Triton path for the strided input)
  conv_update   sglang ...mamba.causal_conv1d.causal_conv1d_update
  ssd_scan      sglang ...mamba.ops.mamba_chunk_scan_combined  (+ ssm_state write)
  ssm_update    sglang ...mamba.ops.mamba_ssm.selective_state_update (triton SSU backend)
  gated_norm    Mixer2RMSNormGated.forward_native body (n_groups=8 > 1 forces this path)
  attn          flashinfer BatchPrefillWithRaggedKVCacheWrapper (prefill, no prefix) /
                BatchDecodeWithPagedKVCacheWrapper, page_size 1 (decode)
  kv_write      sglang ...memory_pool._set_kv_buffer_impl
  relu2         sglang ...activation.ReLU2

Known deviations from the serving engine (recorded in every output row's ``fidelity``):
  * eager per-op launch (decode is graph-replayed only where capture succeeds — see worker);
  * decode KV pages are contiguous per request (the radix allocator may scatter them);
  * one request per prefill call (Bullet's prompt-length axis), no chunked prefill;
  * TP=1, bf16 weights, fp32 ssm state, bf16 conv state (sglang mamba2_state_dtype default).
"""
from __future__ import annotations

from typing import Callable, Dict, List, Tuple

from experiments.opchar import ops_spec as S

FIDELITY = "engine-kernels;eager;tp1;contig-kv;single-request-prefill"


def _torch():
    import torch
    return torch


class _Ctx:
    def __init__(self, cfg: S.HybridCfg, device="cuda", seed=0):
        torch = _torch()
        self.cfg, self.dev, self.t = cfg, device, torch
        self.bf16, self.f32 = torch.bfloat16, torch.float32
        self.gen = torch.Generator(device=device).manual_seed(seed)

    def randn(self, *shape, dtype=None, scale=0.02):
        t = self.t.randn(*shape, generator=self.gen, device=self.dev, dtype=self.f32) * scale
        return t.to(dtype or self.bf16)


# ---------------------------------------------------------------------------
# shared pieces
# ---------------------------------------------------------------------------

def _norm_op(x: "_Ctx", tokens: int):
    from sgl_kernel import fused_add_rmsnorm
    h = x.cfg.hidden
    hs = x.randn(tokens, h, scale=1.0)
    res = x.randn(tokens, h, scale=1.0)
    w = x.t.ones(h, dtype=x.bf16, device=x.dev)
    eps = x.cfg.rms_eps
    return lambda: fused_add_rmsnorm(hs, res, w, eps)


def _linear_op(x: "_Ctx", inp, n_out: int):
    F = x.t.nn.functional
    w = x.randn(n_out, inp.shape[-1])
    return (lambda: F.linear(inp, w)), w


def _mlp(x: "_Ctx", tokens: int) -> Dict[str, Callable]:
    from sglang.srt.layers.activation import ReLU2
    c = x.cfg
    F = x.t.nn.functional
    hs = x.randn(tokens, c.hidden, scale=1.0)
    w_up = x.randn(c.mlp_inter, c.hidden)
    w_dn = x.randn(c.hidden, c.mlp_inter)
    up_out = F.linear(hs, w_up)
    act = ReLU2()
    act_out = act(up_out)
    return {"norm": _norm_op(x, tokens),
            "up_proj": lambda: F.linear(hs, w_up),
            "relu2": lambda: act(up_out),
            "down_proj": lambda: F.linear(act_out, w_dn)}


def _gated_norm_fn(x: "_Ctx", y, gate):
    """Mixer2RMSNormGated.forward_native, tp_size=1, n_groups>1 branch (verbatim math)."""
    torch, c = x.t, x.cfg
    group = c.mamba_inner // c.n_groups
    w = torch.ones(c.mamba_inner, dtype=x.bf16, device=x.dev)
    eps = c.rms_eps

    def fn():
        dt = y.dtype
        z = y * torch.nn.functional.silu(gate.to(torch.float32))
        *pre, hd = z.shape
        g = z.view(*pre, hd // group, group)
        var = g.pow(2).mean(-1, keepdim=True)
        g = g * torch.rsqrt(var + eps)
        return w * g.view(*pre, hd).to(dt)
    return fn


def _mamba_params(x: "_Ctx"):
    torch, c = x.t, x.cfg
    A = -torch.rand(c.n_heads, generator=x.gen, device=x.dev, dtype=x.f32) - 0.5
    D = torch.ones(c.n_heads, dtype=x.f32, device=x.dev)
    dt_bias = torch.zeros(c.n_heads, dtype=x.f32, device=x.dev)
    conv_w = x.randn(c.conv_dim, c.conv_k, scale=0.2)
    conv_b = x.randn(c.conv_dim, scale=0.02)
    return A, D, dt_bias, conv_w, conv_b


# ---------------------------------------------------------------------------
# prefill
# ---------------------------------------------------------------------------

def _mamba_prefill(x: "_Ctx", L: int) -> Dict[str, Callable]:
    torch, c = x.t, x.cfg
    F = torch.nn.functional
    from sglang.srt.layers.attention.mamba.causal_conv1d import causal_conv1d_fn
    from sglang.srt.layers.attention.mamba.ops import mamba_chunk_scan_combined
    A, D, dt_bias, conv_w, conv_b = _mamba_params(x)
    hs = x.randn(L, c.hidden, scale=1.0)
    w_in = x.randn(c.in_proj_out, c.hidden)
    w_out = x.randn(c.hidden, c.mamba_inner)
    proj = F.linear(hs, w_in)
    gate, xbc, dt = torch.split(proj, [c.mamba_inner, c.conv_dim, c.n_heads], dim=-1)
    conv_state = torch.zeros(2, c.conv_dim, c.conv_k - 1, dtype=x.bf16, device=x.dev)
    ssm_state = torch.zeros(2, c.n_heads, c.head_dim, c.d_state, dtype=x.f32, device=x.dev)
    qsl = torch.tensor([0, L], dtype=torch.int32, device=x.dev)
    idx = torch.tensor([0], dtype=torch.int32, device=x.dev)
    has_init = torch.tensor([False], device=x.dev)

    def conv():
        return causal_conv1d_fn(xbc.transpose(0, 1), conv_w, conv_b, activation="silu",
                                conv_states=conv_state, has_initial_state=has_init,
                                cache_indices=idx, query_start_loc=qsl,
                                seq_lens_cpu=[L]).transpose(0, 1)[:L]
    xbc_c = conv()
    hp, Bp, Cp = torch.split(xbc_c, [c.mamba_inner, c.n_groups * c.d_state,
                                     c.n_groups * c.d_state], dim=-1)
    seq_idx = torch.zeros(1, L, dtype=torch.int32, device=x.dev)
    try:
        from sglang.srt.layers.attention.mamba.mamba2_metadata import Mamba2Metadata
        ci, co = Mamba2Metadata._query_start_loc_to_chunk_indices_offsets(qsl, c.chunk, L)
    except Exception:   # noqa: BLE001 -- single request from 0: identity mapping
        n = -(-L // c.chunk)
        ci = torch.arange(n, dtype=torch.int, device=x.dev)
        co = torch.zeros(n, dtype=torch.int, device=x.dev)
    y = torch.empty(L, c.mamba_inner, dtype=x.bf16, device=x.dev)

    def scan():
        st = mamba_chunk_scan_combined(
            hp.view(1, L, c.n_heads, c.head_dim), dt.unsqueeze(0), A,
            Bp.view(1, L, c.n_groups, -1), Cp.view(1, L, c.n_groups, -1),
            chunk_size=c.chunk, D=D, z=None, dt_bias=dt_bias, seq_idx=seq_idx,
            chunk_indices=ci, chunk_offsets=co, cu_seqlens=qsl, initial_states=None,
            return_varlen_states=True, return_final_states=False, dt_softplus=True,
            dt_limit=(0.0, float("inf")), out=y.view(1, L, -1, c.head_dim),
            state_dtype=ssm_state.dtype)
        ssm_state[idx] = st
    scan()
    gn = _gated_norm_fn(x, y, gate[:L])
    gn_out = gn()
    return {"norm": _norm_op(x, L),
            "in_proj": lambda: F.linear(hs, w_in),
            "conv1d": conv, "ssd_scan": scan, "gated_norm": gn,
            "out_proj": lambda: F.linear(gn_out, w_out)}


def _attn_prefill(x: "_Ctx", L: int) -> Dict[str, Callable]:
    torch, c = x.t, x.cfg
    F = torch.nn.functional
    import flashinfer
    from sglang.srt.mem_cache.memory_pool import _set_kv_buffer_impl
    nH, nKV, hd = c.attn_heads, c.attn_kv_heads, c.attn_head_dim
    hs = x.randn(L, c.hidden, scale=1.0)
    w_qkv = x.randn(c.qkv_out, c.hidden)
    w_o = x.randn(c.hidden, nH * hd)
    qkv = F.linear(hs, w_qkv)
    q, k, v = qkv.split([nH * hd, nKV * hd, nKV * hd], dim=-1)
    q = q.contiguous()
    ws = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device=x.dev)
    wrap = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(ws, "NHD", backend="auto")
    qo = torch.tensor([0, L], dtype=torch.int32, device=x.dev)
    wrap.begin_forward(qo, qo, nH, nKV, hd, q_data_type=x.bf16)
    scale = hd ** -0.5
    kbuf = torch.zeros(L + 1, nKV, hd, dtype=x.bf16, device=x.dev)
    vbuf = torch.zeros_like(kbuf)
    loc = torch.arange(1, L + 1, dtype=torch.int64, device=x.dev)
    kv3, vv3 = k.view(-1, nKV, hd), v.view(-1, nKV, hd)

    def kv_write():
        _set_kv_buffer_impl(kv3, vv3, kbuf, vbuf, loc, row_dim=nKV * hd, store_dtype=x.bf16,
                            device_module=torch.cuda, alt_stream=None, same_kv_dim=True)

    def attn():
        return wrap.forward(q.view(-1, nH, hd), kv3, vv3, causal=True, sm_scale=scale)
    o = attn().view(-1, nH * hd)
    return {"norm": _norm_op(x, L),
            "qkv_proj": lambda: F.linear(hs, w_qkv),
            "kv_write": kv_write, "attn": attn,
            "o_proj": lambda: F.linear(o, w_o), "_keep": (ws, wrap)}


# ---------------------------------------------------------------------------
# decode
# ---------------------------------------------------------------------------

def _mamba_decode(x: "_Ctx", B: int) -> Dict[str, Callable]:
    torch, c = x.t, x.cfg
    F = torch.nn.functional
    from sglang.srt.layers.attention.mamba.causal_conv1d import causal_conv1d_update
    from sglang.srt.layers.attention.mamba.ops.mamba_ssm import selective_state_update
    A, D, dt_bias, conv_w, conv_b = _mamba_params(x)
    hs = x.randn(B, c.hidden, scale=1.0)
    w_in = x.randn(c.in_proj_out, c.hidden)
    w_out = x.randn(c.hidden, c.mamba_inner)
    proj = F.linear(hs, w_in)
    gate, xbc, dt = torch.split(proj, [c.mamba_inner, c.conv_dim, c.n_heads], dim=-1)
    pool = B + 1
    conv_state = x.randn(pool, c.conv_dim, c.conv_k - 1, scale=0.1)
    ssm_state = x.randn(pool, c.n_heads, c.head_dim, c.d_state, dtype=x.f32, scale=0.1)
    idx = torch.arange(1, B + 1, dtype=torch.int32, device=x.dev)

    def conv():
        return causal_conv1d_update(xbc, conv_state, conv_w, conv_b, "silu",
                                    conv_state_indices=idx)
    xbc_c = conv()
    hd_, Bd, Cd = torch.split(xbc_c, [c.mamba_inner, c.n_groups * c.d_state,
                                      c.n_groups * c.d_state], dim=-1)
    A_d = A[:, None, None].expand(-1, c.head_dim, c.d_state).to(torch.float32)
    dt_d = dt[:, :, None].expand(-1, -1, c.head_dim)
    dtb = dt_bias[:, None].expand(-1, c.head_dim)
    D_d = D[:, None].expand(-1, c.head_dim)
    Bv = Bd.view(-1, c.n_groups, c.d_state)
    Cv = Cd.view(-1, c.n_groups, c.d_state)
    xv = hd_.reshape(-1, c.n_heads, c.head_dim)
    y = torch.empty(B, c.mamba_inner, dtype=x.bf16, device=x.dev)

    def ssu():
        selective_state_update(ssm_state, xv, dt_d, A_d, Bv, Cv, D_d, z=None, dt_bias=dtb,
                               dt_softplus=True, state_batch_indices=idx,
                               out=y.view(B, -1, c.head_dim))
    ssu()
    gn = _gated_norm_fn(x, y, gate)
    gn_out = gn()
    return {"norm": _norm_op(x, B),
            "in_proj": lambda: F.linear(hs, w_in),
            "conv_update": conv, "ssm_update": ssu, "gated_norm": gn,
            "out_proj": lambda: F.linear(gn_out, w_out)}


def _attn_decode(x: "_Ctx", B: int, ctx: int) -> Dict[str, Callable]:
    torch, c = x.t, x.cfg
    F = torch.nn.functional
    import flashinfer
    from sglang.srt.mem_cache.memory_pool import _set_kv_buffer_impl
    try:
        from sglang.srt.layers.attention.flashinfer_backend import should_use_tensor_core
        tc = should_use_tensor_core(x.bf16, c.attn_heads, c.attn_kv_heads)
    except Exception:   # noqa: BLE001
        tc = None
    nH, nKV, hd = c.attn_heads, c.attn_kv_heads, c.attn_head_dim
    hs = x.randn(B, c.hidden, scale=1.0)
    w_qkv = x.randn(c.qkv_out, c.hidden)
    w_o = x.randn(c.hidden, nH * hd)
    qkv = F.linear(hs, w_qkv)
    q, k, v = qkv.split([nH * hd, nKV * hd, nKV * hd], dim=-1)
    q = q.contiguous()
    size = B * (ctx + 1) + 1
    kbuf = x.randn(size, nKV, hd, scale=1.0)
    vbuf = x.randn(size, nKV, hd, scale=1.0)
    # request i owns slots [1 + i*(ctx+1), 1 + (i+1)*(ctx+1)); new token at the last slot
    starts = 1 + torch.arange(B, device=x.dev) * (ctx + 1)
    indices = (starts[:, None] + torch.arange(ctx + 1, device=x.dev)[None, :]).reshape(-1).to(torch.int32)
    indptr = torch.arange(0, B + 1, device=x.dev, dtype=torch.int32) * (ctx + 1)
    last = torch.ones(B, dtype=torch.int32, device=x.dev)
    loc = (starts + ctx).to(torch.int64)
    ws = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device=x.dev)
    kw = {"backend": "fa2"}          # FlashInferAttnBackend.decode_backend (v0.5.10)
    if tc is not None:
        kw["use_tensor_cores"] = tc
    wrap = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, "NHD", **kw)
    wrap.begin_forward(indptr, indices, last, nH, nKV, hd, 1, data_type=x.bf16,
                       q_data_type=x.bf16)
    scale = hd ** -0.5
    kv3, vv3 = k.view(-1, nKV, hd), v.view(-1, nKV, hd)

    def kv_write():
        _set_kv_buffer_impl(kv3, vv3, kbuf, vbuf, loc, row_dim=nKV * hd, store_dtype=x.bf16,
                            device_module=torch.cuda, alt_stream=None, same_kv_dim=True)

    def attn():
        return wrap.forward(q.view(-1, nH, hd), (kbuf, vbuf), sm_scale=scale)
    o = attn().view(-1, nH * hd)
    return {"norm": _norm_op(x, B),
            "qkv_proj": lambda: F.linear(hs, w_qkv),
            "kv_write": kv_write, "attn": attn,
            "o_proj": lambda: F.linear(o, w_o), "_keep": (ws, wrap),
            "_meta": {"use_tensor_cores": tc}}


# ---------------------------------------------------------------------------
# public entry
# ---------------------------------------------------------------------------

def build(cfg: S.HybridCfg, phase: str, size: int, ctx: int = 0, device="cuda",
          seed=0) -> Tuple[List[Tuple[S.OpSpec, Callable]], dict]:
    """Return [(OpSpec, fn)] in ops_spec order plus build metadata.

    phase="prefill": size = prompt length L.  phase="decode": size = batch B, ctx = context.
    Raises if the built op names differ from ops_spec (single inventory).
    """
    x = _Ctx(cfg, device, seed)
    if phase == "prefill":
        spec = S.prefill_ops(cfg, size)
        built = {"M": _mamba_prefill(x, size), "*": _attn_prefill(x, size), "-": _mlp(x, size)}
    elif phase == "decode":
        spec = S.decode_ops(cfg, size, ctx)
        built = {"M": _mamba_decode(x, size), "*": _attn_decode(x, size, ctx), "-": _mlp(x, size)}
    else:
        raise ValueError(f"phase {phase!r}")
    meta = {"fidelity": FIDELITY}
    out = []
    for layer in S.LAYER_CHARS:
        fns = built[layer]
        meta.update({f"{layer}{k}": v for k, v in fns.get("_meta", {}).items()})
        want = [o.name for o in spec[layer]]
        have = [k for k in fns if not k.startswith("_")]
        if sorted(want) != sorted(have):
            raise RuntimeError(f"layer {layer}: builders {sorted(have)} != ops_spec {sorted(want)}")
        out.extend((o, fns[o.name]) for o in spec[layer])
    meta["_keep"] = [fns.get("_keep") for fns in built.values()]
    return out, meta
