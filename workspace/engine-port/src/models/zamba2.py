# SPDX-License-Identifier: Apache-2.0
"""SGLang port of Zamba2 (temporal Mamba2 + shared-attention hybrid).

Ported from vLLM `zamba2.py` (logic) onto SGLang idioms (NemotronH template):
- Mamba2 mixer via `forward_batch.attn_backend.linear_attn_backend`.
- Shared attention block reused at each `hybrid_layer_ids` position, each with
  its own `RadixAttention` (distinct global layer_id -> distinct KV slot) but
  shared qkv/o projection weights + per-position LoRA adapters.
- Every layer has a Mamba2 mixer (mamba_layer_ids = all); attention only at the
  9 hybrid positions (full_attention_layer_ids = hybrid_layer_ids).

STATUS: first implementation. Structure/instantiation targeted; **logit-parity
vs vLLM not yet verified** (see reports/zamba2_port_plan.md §7). Parity-sensitive
spots are marked `# PARITY`.
"""
from collections.abc import Iterable
from itertools import cycle
from typing import Optional

import torch
from torch import nn

from sglang.srt.configs.zamba2 import Zamba2Config
from sglang.srt.distributed import get_pp_group, get_tensor_model_parallel_world_size
from sglang.srt.layers.activation import GeluAndMul
from sglang.srt.layers.attention.mamba.mamba import MambaMixer2
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.utils import add_prefix

# --- measurement instrumentation (env-gated; per-layer-type decode/prefill timing) ---
#
# 2026-08-04 REWORK (two audited instrumentation defects in job 858811):
#   (1) BUCKET ASYMMETRY. "attn" wrapped only the RadixAttention core while
#       "mamba" wrapped the whole MambaMixer2 call; qkv_proj/o_proj/MLP/layernorm
#       and the hybrid `.linear` were in NO bucket. The two buckets were therefore
#       not comparable and their sum was far below the forward.
#   (2) RUNNING-MEAN ACCUMULATOR. `_zt_acc` reset only on mode change, never at
#       emit, so every emitted value carried cold-start forever (and the harness
#       took `tail -1` = the largest n = the most cold-start-diluted sample).
#
# Fixes here: DISJOINT buckets that together cover the layer loop, plus a CLOSURE
# GATE, plus block-scoped accumulation alongside the legacy running mean.
#
#   closure = sum(_ZT_BUCKETS spans) / (separate event pair around the whole layer loop)
#
# The gate is deliberately NOT a timeline partition: consecutive segments that
# tile an interval by construction would make the ratio an identity and the gate
# vacuous (cf. project methodology gate on identity-valued gates). Each bucket
# span carries its OWN start/end pair, and the denominator is an INDEPENDENT pair,
# so unbucketed work inside the loop shows up as closure < 1. Defect (1) would
# have produced closure ~ 0.2-0.5 and been caught at the first emit.
#
# _ZT_DIAG buckets are NESTED inside _ZT_BUCKETS entries (diagnostic only) and are
# excluded from the closure sum. "attn_core" reproduces the pre-2026-08-04 "attn"
# definition so old and new reports can be compared directly.
#
# LIMITATIONS (unchanged from the pre-2026-08-04 version, stated here so they are
# not rediscovered): `_ZT` is module-global, so a true-dual-worker runtime with two
# host issue threads would interleave spans from two forwards. This path is for
# single-worker offline composition measurement only. Closure below 1 also absorbs
# inter-span launch gaps (eager mode, ~200 spans/forward), so the threshold must be
# CALIBRATED against an observed run before it is treated as a defect detector --
# a defect-1-style miss is ~0.2-0.5, event overhead is a few percent.
_ZT_BUCKETS = ("attn", "mlp", "other", "mamba")
_ZT_DIAG = ("attn_core",)
_ZT = {
    "on": False,
    "spans": {b: [] for b in _ZT_BUCKETS + _ZT_DIAG},
    "pool": [],
    "pool_n": 0,
}


def _zt_event():
    """Hand out a recycled cuda Event. Events are re-recorded every timed forward
    and only read after `torch.cuda.synchronize()` inside that same forward, so
    reuse is safe and keeps ~400 allocations/step off the measured path."""
    pool = _ZT["pool"]
    i = _ZT["pool_n"]
    if i == len(pool):
        pool.append(torch.cuda.Event(enable_timing=True))
    _ZT["pool_n"] = i + 1
    return pool[i]


def _zt_reset():
    for spans in _ZT["spans"].values():
        spans.clear()
    _ZT["pool_n"] = 0


def _zt_begin(bucket):
    """Open a span. Call sites guard with `if _ZT["on"]:` so the instrumentation-OFF
    path costs one dict read per layer and runs the original statement sequence
    unchanged (no lambda allocation, no extra call)."""
    if not _ZT["on"]:
        return None
    s = _zt_event()
    s.record()
    return (bucket, s)


def _zt_end(tok):
    if tok is None or not _ZT["on"]:
        return
    e = _zt_event()
    e.record()
    _ZT["spans"][tok[0]].append((tok[1], e))


def _zt_new_block():
    """Block accumulator, reset at EVERY emit (defect (2) fix)."""
    d = {"n": 0, "total": 0.0, "closure_min": float("inf"), "shape": None, "dirty": 0}
    for b in _ZT_BUCKETS + _ZT_DIAG:
        d[b] = 0.0
    return d


def _zt_block_update(blk, fwd, fwd_total, shape):
    """Fold one forward into `blk`; return that forward's closure ratio.

    closure counts _ZT_BUCKETS only: _ZT_DIAG spans are nested inside them and
    would double-count. `shape` = (nseq, ntok) of this forward; a block whose
    shape changes mid-flight is flagged `dirty` (pre-fix the LAST forward's shape
    was reported for the whole window)."""
    closure = (
        sum(fwd[b] for b in _ZT_BUCKETS) / fwd_total if fwd_total > 0 else float("nan")
    )
    blk["n"] += 1
    blk["total"] += fwd_total
    for b, v in fwd.items():
        blk[b] += v
    if closure == closure and closure < blk["closure_min"]:  # NaN-safe
        blk["closure_min"] = closure
    if blk["shape"] is None:
        blk["shape"] = shape
    elif blk["shape"] != shape:
        blk["dirty"] = 1
    return closure


def _zt(bucket, fn):
    """Legacy callable form (kept for API compatibility); OFF path = bare call."""
    if not _ZT["on"]:
        return fn()
    tok = _zt_begin(bucket)
    out = fn()
    _zt_end(tok)
    return out


class Zamba2LoRA(nn.Module):
    """rank-r LoRA used inside the shared attention/MLP blocks (A: down, B: up)."""

    def __init__(self, input_dim, rank, output_dim, quant_config=None, prefix=""):
        super().__init__()
        self.A = ColumnParallelLinear(
            input_dim, rank, bias=False, gather_output=True,
            quant_config=quant_config, prefix=f"{prefix}.A",
        )
        B_class = MergedColumnParallelLinear if isinstance(output_dim, list) else ColumnParallelLinear
        self.B = B_class(
            rank, output_dim, bias=False, quant_config=quant_config, prefix=f"{prefix}.B",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        lora_output, _ = self.A(hidden_states)
        lora_output, _ = self.B(lora_output)
        return lora_output


class Zamba2Attention(nn.Module):
    """Shared multi-head attention. Weights shared across hybrid positions;
    KV cache + optional LoRA adapters are per-position (indexed by block_idx)."""

    def __init__(
        self,
        config: Zamba2Config,
        bare_block_idx: int,
        num_hybrid_layers: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        self.config = config
        self.num_hybrid_layers = num_hybrid_layers
        self.attention_hidden_size = config.attention_hidden_size
        self.total_num_attention_heads = config.num_attention_heads
        assert self.total_num_attention_heads % tp_size == 0
        self.num_attention_heads = config.num_attention_heads // tp_size
        self.attention_head_dim = config.attention_head_dim
        self.qkv_size = self.attention_hidden_size // tp_size
        self.scale = (self.attention_head_dim / 2) ** -0.5  # PARITY: vLLM uses head_dim/2

        self.qkv_proj = QKVParallelLinear(
            self.attention_hidden_size,
            self.attention_head_dim,
            self.total_num_attention_heads,
            bias=False, quant_config=quant_config, prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.attention_hidden_size, config.hidden_size,
            bias=False, quant_config=quant_config, prefix=f"{prefix}.o_proj",
        )

        # per-position RadixAttention: real one where this bare block owns the
        # position; distinct global layer_id (= hybrid_layer_ids[block_idx]).
        self.dpa_list = nn.ModuleList([])
        hybrid_layer_ids = list(config.hybrid_layer_ids)
        for block_idx in range(self.num_hybrid_layers):
            if block_idx % config.num_mem_blocks == bare_block_idx:
                dpa = RadixAttention(
                    self.num_attention_heads,
                    self.attention_head_dim,
                    self.scale,
                    num_kv_heads=self.num_attention_heads,  # no GQA in Zamba2
                    layer_id=hybrid_layer_ids[block_idx],
                    prefix=f"{prefix}.dpa.{block_idx}",
                )
            else:
                dpa = nn.Identity()
            self.dpa_list.append(dpa)

        # Optional Q/K/V LoRA adapters (Zamba2-2.7B: use_shared_attention_adapter=False)
        self.use_adapter = config.use_shared_attention_adapter
        if self.use_adapter:
            self.linear_q_adapter_list = nn.ModuleList([])
            self.linear_k_adapter_list = nn.ModuleList([])
            self.linear_v_adapter_list = nn.ModuleList([])
            for block_idx in range(self.num_hybrid_layers):
                if block_idx % config.num_mem_blocks == bare_block_idx:
                    mk = lambda name: Zamba2LoRA(
                        self.attention_hidden_size, config.adapter_rank,
                        self.attention_hidden_size, quant_config=quant_config,
                        prefix=f"{prefix}.{name}",
                    )
                    q_ad, k_ad, v_ad = mk("linear_q_adapter"), mk("linear_k_adapter"), mk("linear_v_adapter")
                else:
                    q_ad = k_ad = v_ad = nn.Identity()
                self.linear_q_adapter_list.append(q_ad)
                self.linear_k_adapter_list.append(k_ad)
                self.linear_v_adapter_list.append(v_ad)

    def forward(self, hidden_states, block_idx, forward_batch: ForwardBatch):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.qkv_size] * 3, dim=-1)
        if self.use_adapter:
            q = q + self.linear_q_adapter_list[block_idx](hidden_states)
            k = k + self.linear_k_adapter_list[block_idx](hidden_states)
            v = v + self.linear_v_adapter_list[block_idx](hidden_states)
        # PARITY: use_mem_rope=False for 2.7B; rope omitted (add get_rope if True).
        # "attn_core" is the DIAGNOSTIC (nested) bucket = exactly what pre-2026-08-04
        # "attn" measured. The accounted "attn" bucket wraps this whole method from
        # the caller (qkv_proj + adapters + core + o_proj).
        _tok = _zt_begin("attn_core") if _ZT["on"] else None
        attn_output = self.dpa_list[block_idx].forward(q, k, v, forward_batch)
        if _tok is not None:
            _zt_end(_tok)
        y, _ = self.o_proj(attn_output)
        return y


class Zamba2MLP(nn.Module):
    """Shared gated MLP with per-position gate_up LoRA adapter."""

    def __init__(self, config, bare_block_idx, num_hybrid_layers, quant_config=None, prefix=""):
        super().__init__()
        self.config = config
        self.num_hybrid_layers = num_hybrid_layers
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size, 2 * [config.intermediate_size],
            bias=config.add_bias_linear, quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size, config.hidden_size,
            bias=config.add_bias_linear, quant_config=quant_config, prefix=f"{prefix}.down_proj",
        )
        if config.hidden_act != "gelu":
            raise ValueError(f"Zamba2 only supports gelu (got {config.hidden_act})")
        self.act_fn = GeluAndMul()

        self.gate_up_proj_adapter_list = nn.ModuleList([])
        for block_idx in range(self.num_hybrid_layers):
            if block_idx % config.num_mem_blocks == bare_block_idx:
                adapter = Zamba2LoRA(
                    config.hidden_size, config.adapter_rank, 2 * [config.intermediate_size],
                    quant_config, prefix=f"{prefix}.gate_up_proj_adapter_list.{block_idx}",
                )
            else:
                adapter = nn.Identity()
            self.gate_up_proj_adapter_list.append(adapter)

    def forward(self, hidden_states, block_idx):
        gate_up, _ = self.gate_up_proj(hidden_states)
        gate_up = gate_up + self.gate_up_proj_adapter_list[block_idx](hidden_states)
        hidden_states = self.act_fn(gate_up)
        output, _ = self.down_proj(hidden_states)
        return output


class Zamba2AttentionDecoderLayer(nn.Module):
    """A shared transformer block (reused at hybrid positions via block_idx)."""

    def __init__(self, config, bare_block_idx, num_hybrid_layers, quant_config=None, prefix=""):
        super().__init__()
        self.self_attn = Zamba2Attention(
            config, bare_block_idx, num_hybrid_layers, quant_config, prefix=prefix,
        )
        self.feed_forward = Zamba2MLP(
            config, bare_block_idx, num_hybrid_layers, quant_config, prefix=f"{prefix}.feed_forward",
        )
        # input_layernorm operates on concat[hidden, original] -> 2*hidden.
        self.input_layernorm = RMSNorm(2 * config.hidden_size, eps=config.rms_norm_eps)
        self.pre_ff_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, original_hidden_states, block_idx, forward_batch):
        # Buckets (disjoint): concat+input_layernorm -> "other"; the WHOLE attention
        # module (qkv_proj, adapters, core, o_proj) -> "attn"; pre_ff_layernorm ->
        # "other"; the whole MLP -> "mlp". MLP/layernorm are NOT attributed to attn:
        # there is no code-level basis for calling them attention work.
        _on = _ZT["on"]
        _tok = _zt_begin("other") if _on else None
        hidden_states = torch.concatenate([hidden_states, original_hidden_states], dim=-1)
        hidden_states = self.input_layernorm(hidden_states)
        if _on:
            _zt_end(_tok)
            _tok = _zt_begin("attn")
        hidden_states = self.self_attn(hidden_states, block_idx, forward_batch)
        if _on:
            _zt_end(_tok)
            _tok = _zt_begin("other")
        hidden_states = self.pre_ff_layernorm(hidden_states)
        if _on:
            _zt_end(_tok)
            _tok = _zt_begin("mlp")
        hidden_states = self.feed_forward(hidden_states, block_idx)
        if _on:
            _zt_end(_tok)
        return hidden_states


class Zamba2MambaDecoderLayer(nn.Module):
    """Mamba2 mixer layer (present in every Zamba2 layer)."""

    def __init__(self, config, layer_idx, quant_config=None, prefix=""):
        super().__init__()
        self.layer_id = layer_idx
        intermediate_size = config.mamba_expand * config.hidden_size
        self.mamba = MambaMixer2(
            cache_params=config.mamba2_cache_params,
            hidden_size=config.hidden_size,
            use_conv_bias=config.use_conv_bias,
            use_bias=config.add_bias_linear,
            n_groups=config.mamba_ngroups,
            rms_norm_eps=config.rms_norm_eps,
            activation="silu",
            quant_config=quant_config,
            prefix=f"{prefix}.mamba",
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, forward_batch, transformer_hidden_states=None):
        # Buckets: residual add + input_layernorm + output alloc -> "other";
        # the whole MambaMixer2 call -> "mamba" (definition UNCHANGED since 858811,
        # so the mamba numbers stay directly comparable); trailing residual -> "other".
        _on = _ZT["on"]
        _tok = _zt_begin("other") if _on else None
        residual = hidden_states
        if transformer_hidden_states is not None:
            hidden_states = hidden_states + transformer_hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        output = torch.empty_like(hidden_states)
        attn_backend = forward_batch.attn_backend
        if _on:
            _zt_end(_tok)
            _tok = _zt_begin("mamba")
        attn_backend.linear_attn_backend.forward(
            mixer=self.mamba,
            layer_id=self.layer_id,
            hidden_states=hidden_states,
            output=output,
            use_triton_causal_conv=True,
        )
        if _on:
            _zt_end(_tok)
            _tok = _zt_begin("other")
        out = residual + output
        if _on:
            _zt_end(_tok)
        return out


class Zamba2HybridLayer(nn.Module):
    """Shared transformer path (projected) injected into the Mamba path."""

    def __init__(self, shared_transformer, config, block_idx, layer_idx, quant_config=None, prefix=""):
        super().__init__()
        self.block_idx = block_idx
        self.shared_transformer = shared_transformer
        self.linear = ReplicatedLinear(
            config.hidden_size, config.hidden_size, bias=False,
            quant_config=quant_config, prefix=f"{prefix}.linear",
        )
        self.mamba_decoder = Zamba2MambaDecoderLayer(
            config, layer_idx, quant_config=quant_config, prefix=prefix,
        )

    def forward(self, hidden_states, original_hidden_states, forward_batch):
        transformer_hidden_states = self.shared_transformer(
            hidden_states, original_hidden_states, self.block_idx, forward_batch,
        )
        # hybrid projection: its own span in "other" (not attention, not mamba).
        _tok = _zt_begin("other") if _ZT["on"] else None
        transformer_hidden_states, _ = self.linear(transformer_hidden_states)
        if _tok is not None:
            _zt_end(_tok)
        return self.mamba_decoder(
            hidden_states, forward_batch, transformer_hidden_states=transformer_hidden_states,
        )


class Zamba2Model(nn.Module):
    def __init__(self, config: Zamba2Config, quant_config=None, prefix=""):
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)

        layer2block_map = {
            layer_idx: block_idx
            for block_idx, layer_idx in enumerate(config.hybrid_layer_ids)
        }
        num_hybrid_layers = len(layer2block_map)
        blocks = cycle(
            Zamba2AttentionDecoderLayer(
                config, bare_block_idx=idx, num_hybrid_layers=num_hybrid_layers,
                quant_config=quant_config, prefix=f"{prefix}.shared_block.{idx}",
            )
            for idx in range(config.num_mem_blocks)
        )

        layers = []
        for layer_idx, layer_type in enumerate(config.layers_block_type):
            if layer_type == "hybrid":
                block = next(blocks)
                block_idx = layer2block_map[layer_idx]
                layers.append(
                    Zamba2HybridLayer(
                        block, config, block_idx, layer_idx,
                        quant_config=quant_config, prefix=f"{prefix}.layers.{layer_idx}",
                    )
                )
            else:
                layers.append(
                    Zamba2MambaDecoderLayer(
                        config, layer_idx, quant_config=quant_config,
                        prefix=f"{prefix}.layers.{layer_idx}",
                    )
                )
        self.layers = nn.ModuleList(layers)
        self.final_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # measured layer counts for the per-layer divisors (2.7B: 9 attn / 54 mamba).
        # Read from the built model instead of hard-coding 9/54, so the 1.2B/7B
        # configs report per-layer numbers against their own counts.
        self._zt_n_attn = sum(1 for l in layers if isinstance(l, Zamba2HybridLayer))
        self._zt_n_mamba = len(layers)
        self._zt_acc = {b: 0.0 for b in _ZT_BUCKETS + _ZT_DIAG}
        self._zt_acc_total = 0.0
        self._zt_n = 0
        self._zt_mode = None
        self._zt_blk = _zt_new_block()
        self._zt_blk_idx = 0

    def _get_gctx_decode_stream(self, n_sm: int):
        if not hasattr(self, "_gctx_streams"):
            self._gctx_streams = {}
        if n_sm not in self._gctx_streams:
            from sgl_kernel import spatial

            dev = torch.cuda.current_device()
            total = spatial.get_sm_available(dev)
            other = max(4, total - n_sm)
            sA, _sB = spatial.create_greenctx_stream_by_value(n_sm, other, dev)
            self._gctx_streams[n_sm] = sA
        return self._gctx_streams[n_sm]

    def forward(self, input_ids, positions, forward_batch, input_embeds=None):
        import contextlib as _cl
        import os as _os

        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_embeds
        original_hidden_states = torch.clone(hidden_states)

        # measurement: pin decode (or, in prefill-knee mode, prefill) to N SMs + time
        # attn vs mamba (env/file-gated). SGLANG_ZAMBA_PREFILL_KNEE retargets the same
        # pin+timing machinery at the EXTEND (prefill) forward, to answer whether prefill
        # has an SM-insensitive layer type (a prefill-side layer-aware lever).
        _is_decode = forward_batch.forward_mode.is_decode()
        _pk = bool(_os.environ.get("SGLANG_ZAMBA_PREFILL_KNEE"))
        _phase = forward_batch.forward_mode.is_extend() if _pk else _is_decode
        _mode = None
        _f = _os.environ.get(
            "PDMUX_FIXED_PREFILL_SM_FILE" if _pk else "PDMUX_FIXED_DECODE_SM_FILE"
        )
        if _f and _phase:
            try:
                _mode = open(_f).read().strip()
            except Exception:
                _mode = None
        # mode token is "<sm>@<ctxtag>" so the accumulator resets per (sm,ctx) combo
        _sm_part = _mode.split("@")[0] if _mode else None
        _fixed = (
            int(_sm_part)
            if (_sm_part and _sm_part.isdigit())
            else None
        )
        # agnostic_v2: uniform reservation sized to the (cheap) attention layer ->
        # pin ALL decode layers to a low floor, maximizing prefill reclaim but NOT
        # protecting the SM-sensitive layers (in long-ctx Zamba2 that's the no-GQA
        # attn). Contrast with layer_aware, which keeps the hybrid attn at full SM.
        if _is_decode and _sm_part == "agnostic_v2":
            _fixed = int(_os.environ.get("PDMUX_AGN2_SM", "16"))
        _timing = bool(_os.environ.get("SGLANG_ZAMBA_TIMING")) and _phase
        if _timing:
            _zt_reset()
            _ZT["on"] = True
            if getattr(self, "_zt_mode", None) != _mode:
                # LEGACY accumulator: running mean since the last mode change,
                # reset ONLY here. Kept (not removed) so pre-2026-08-04 reports
                # remain reproducible; the block accumulator below is the one that
                # is reset at every emit.
                self._zt_acc = {b: 0.0 for b in _ZT_BUCKETS + _ZT_DIAG}
                self._zt_acc_total = 0.0
                self._zt_n = 0
                self._zt_mode = _mode
                self._zt_blk = _zt_new_block()
                self._zt_blk_idx = 0
        # decode SM policy per layer (race-safe: wait_stream at each switch carries
        # the layer i->i+1 data dependency + safety vs the default stream):
        #  - fixed-N: pin all decode layers to an N-SM partition.
        #  - layer_aware (Zamba2): the 9 HYBRID layers hold the SM-sensitive
        #    (no-GQA) attention -> keep on base (reserved); the ~45 pure-mamba
        #    layers are SM-insensitive -> drop to a floor, freeing SM for prefill.
        _la = _is_decode and _sm_part == "layer_aware"
        _la_floor = int(_os.environ.get("PDMUX_LA_FLOOR_SM", "16"))
        # graduated per-type SM (this-work extension beyond the 2-level binary la):
        # PDMUX_LA_SM_MAP="mamba:54,attn:96" pins each layer TYPE to its own decode-SM
        # green-ctx, testing whether a per-type allocation (each type at its knee, only
        # the surplus released) can beat uniform agnostic. Absent -> legacy binary (full/floor).
        _la_map = {}
        _m = _os.environ.get("PDMUX_LA_SM_MAP")
        if _m and _la:
            for _kv in _m.split(","):
                if ":" in _kv:
                    _k, _v = _kv.split(":", 1)
                    if _v.strip().isdigit():
                        _la_map[_k.strip().lower()] = int(_v.strip())
        _base = torch.cuda.current_stream()

        def _tgt(layer):
            if _fixed is not None:
                return self._get_gctx_decode_stream(_fixed)
            if _la:
                _is_attn = isinstance(layer, Zamba2HybridLayer)
                if _la_map:
                    _n = _la_map.get("attn" if _is_attn else "mamba")
                    return self._get_gctx_decode_stream(_n) if _n is not None else _base
                if not _is_attn:  # legacy binary: release mamba to floor
                    return self._get_gctx_decode_stream(_la_floor)
            return _base

        # closure-gate denominator: an INDEPENDENT event pair around the whole layer
        # loop (not a tiling of the bucket spans -- see the _ZT header comment).
        _zt_tot_s = _zt_event() if _timing else None
        if _timing:
            _zt_tot_s.record()
        _prev = _base
        try:
            for layer in self.layers:
                _t = _tgt(layer) if _phase else _base
                if _t is not _prev:
                    _t.wait_stream(_prev)
                with (torch.cuda.stream(_t) if _t is not _base else _cl.nullcontext()):
                    if isinstance(layer, Zamba2HybridLayer):
                        hidden_states = layer(hidden_states, original_hidden_states, forward_batch)
                    else:
                        hidden_states = layer(hidden_states, forward_batch)
                _prev = _t
            if _prev is not _base:
                _base.wait_stream(_prev)
        finally:
            # never leave the global timing flag set if a layer raised (a stale
            # `on` would silently instrument -- and mis-bucket -- later forwards).
            if _timing:
                _ZT["on"] = False
        _zt_tot_e = _zt_event() if _timing else None
        if _timing:
            _zt_tot_e.record()
        out = self.final_layernorm(hidden_states)
        if _timing:
            torch.cuda.synchronize()
            _fwd_total = _zt_tot_s.elapsed_time(_zt_tot_e)
            _fwd = {}
            for _b, _spans in _ZT["spans"].items():
                _a = 0.0
                for _s, _e in _spans:
                    _a += _s.elapsed_time(_e)
                _fwd[_b] = _a
            # shape guard: .shape reads are host-side (no device sync) so they run on
            # EVERY timed forward. Pre-fix only the LAST forward's bs/ntok was reported
            # and silently applied to the whole accumulation window.
            try:
                _nseq = int(forward_batch.seq_lens.shape[0]) if forward_batch.seq_lens is not None else -1
            except Exception:
                _nseq = -1
            try:
                _ntok = int(forward_batch.input_ids.shape[0]) if getattr(forward_batch, "input_ids", None) is not None else -1
            except Exception:
                _ntok = -1

            self._zt_n += 1
            self._zt_acc_total += _fwd_total
            for _b, _v in _fwd.items():
                self._zt_acc[_b] += _v
            _blk = self._zt_blk
            _zt_block_update(_blk, _fwd, _fwd_total, (_nseq, _ntok))

            # emit period: legacy defaults (30 decode / 4 prefill forwards) unless
            # SGLANG_ZAMBA_TIMING_EVERY is set. Smaller blocks -> more steady-state
            # samples per arm and a shorter warm-up straddle.
            try:
                _every = int(_os.environ.get("SGLANG_ZAMBA_TIMING_EVERY", "0"))
            except ValueError:
                _every = 0
            if _every <= 0:
                _every = 4 if _pk else 30
            if self._zt_n % _every == 0:
                import logging as _lg
                _log = _lg.getLogger("sglang.srt.models.zamba2")
                n = self._zt_n
                bn = max(_blk["n"], 1)
                na = max(getattr(self, "_zt_n_attn", 0), 1)
                nm = max(getattr(self, "_zt_n_mamba", len(self.layers)), 1)
                _blk_sum = sum(_blk[_b] for _b in _ZT_BUCKETS)
                _blk_closure = (_blk_sum / _blk["total"]) if _blk["total"] > 0 else float("nan")
                _cum_closure = (
                    sum(self._zt_acc[_b] for _b in _ZT_BUCKETS) / self._zt_acc_total
                    if self._zt_acc_total > 0 else float("nan")
                )
                _sh = _blk["shape"] or (-1, -1)
                # Tag is ZBLT2/ZBPT2, NOT ZBLT/ZBPT: the "attn" bucket definition
                # changed (now the whole attention module), so a pre-2026-08-04
                # parser must FAIL to match rather than silently read a different
                # quantity. attn_core_* reproduces the old definition.
                _log.warning(
                    "%s mode=%s ctxlen=%s n=%d bs=%d ntok=%d"
                    " | attn_total=%.3f mamba_total=%.3f"
                    " | per-attn(%d)=%.4f per-mamba(%d)=%.4f"
                    " | attn_core_total=%.3f per-attn_core(%d)=%.4f closure_cum=%.4f"
                    " || blk=%d blk_n=%d dirty=%d shape=%dx%d"
                    " closure=%.4f closure_min=%.4f fwd_ms=%.4f"
                    " | b_attn=%.4f b_mlp=%.4f b_other=%.4f b_mamba=%.4f b_attn_core=%.4f"
                    " | b_per_attn=%.5f b_per_mamba=%.5f b_per_attn_core=%.5f",
                    ("ZBPT2" if _pk else "ZBLT2"),
                    _mode,
                    int(forward_batch.seq_lens.max().item()) if forward_batch.seq_lens is not None else -1,
                    n, _nseq, _ntok,
                    # LEGACY block: running mean since the last mode change (kept).
                    self._zt_acc["attn"] / n, self._zt_acc["mamba"] / n,
                    na, self._zt_acc["attn"] / n / na, nm, self._zt_acc["mamba"] / n / nm,
                    self._zt_acc["attn_core"] / n, na, self._zt_acc["attn_core"] / n / na,
                    _cum_closure,
                    # BLOCK block: reset at every emit (this is the steady-state read).
                    self._zt_blk_idx, _blk["n"], _blk["dirty"], _sh[0], _sh[1],
                    _blk_closure, _blk["closure_min"], _blk["total"] / bn,
                    _blk["attn"] / bn, _blk["mlp"] / bn, _blk["other"] / bn,
                    _blk["mamba"] / bn, _blk["attn_core"] / bn,
                    _blk["attn"] / bn / na, _blk["mamba"] / bn / nm,
                    _blk["attn_core"] / bn / na,
                )
                try:
                    _cmin = float(_os.environ.get("SGLANG_ZAMBA_CLOSURE_MIN", "0.95"))
                except ValueError:
                    _cmin = 0.95
                if not (_blk_closure >= _cmin):
                    _log.warning(
                        "ZBLT_CLOSURE_FAIL mode=%s blk=%d closure=%.4f min_fwd=%.4f"
                        " threshold=%.2f bucket_ms=%.4f loop_ms=%.4f"
                        " -- buckets do not cover the layer loop; per-type numbers"
                        " from this block are NOT trustworthy",
                        _mode, self._zt_blk_idx, _blk_closure, _blk["closure_min"],
                        _cmin, _blk_sum / bn, _blk["total"] / bn,
                    )
                self._zt_blk_idx += 1
                self._zt_blk = _zt_new_block()
        return out


class Zamba2ForCausalLM(nn.Module):
    # HF Zamba2 stores shared-block attention as fused; qkv is separate q/k/v.
    stacked_params_mapping = [
        ("qkv_proj", "q_proj", "q"),
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
    ]
    packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}

    def __init__(self, *, config: Zamba2Config, quant_config=None, prefix: str = ""):
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.model = Zamba2Model(config, quant_config=quant_config, prefix=add_prefix("model", prefix))
        self.pp_group = get_pp_group()
        if config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size, config.hidden_size,
                quant_config=quant_config, prefix=add_prefix("lm_head", prefix),
            )
        self.logits_processor = LogitsProcessor(config)

    def get_input_embeddings(self) -> VocabParallelEmbedding:
        return self.model.embed_tokens

    @torch.no_grad()
    def forward(self, input_ids, positions, forward_batch, input_embeds=None, **kwargs):
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        return self.logits_processor(input_ids, hidden_states, self.lm_head, forward_batch)

    def forward_split_prefill(
        self, input_ids, positions, forward_batch, split_interval, input_embeds=None
    ):
        # PATCH (engine-port): enable pdmux split-prefill for Zamba2. Threads
        # hidden_states + the constant original_hidden_states (embeddings, used by
        # every hybrid layer's concat) across split windows via forward_batch.
        start, end = split_interval
        model = self.model
        if start == 0:
            hs = model.embed_tokens(input_ids) if input_embeds is None else input_embeds
            forward_batch.hidden_states = hs
            forward_batch.zamba_original = torch.clone(hs)
        orig = forward_batch.zamba_original
        for i in range(start, end):
            layer = model.layers[i]
            if isinstance(layer, Zamba2HybridLayer):
                forward_batch.hidden_states = layer(
                    forward_batch.hidden_states, orig, forward_batch
                )
            else:
                forward_batch.hidden_states = layer(forward_batch.hidden_states, forward_batch)
        if end == self.config.num_hidden_layers:
            hidden = model.final_layernorm(forward_batch.hidden_states)
            forward_batch.hidden_states = hidden
            return self.logits_processor(input_ids, hidden, self.lm_head, forward_batch)

    # ---- coordinated per-type layer-aware (R0c/A) ----
    # forward_split_decode runs decode layers [start,end) threading hidden state, exactly
    # like forward_split_prefill (layers dispatch by forward_batch.forward_mode=DECODE).
    # NO internal green-ctx switching: the coordinated event loop sets the pdmux pair
    # (decode-half stream) per layer-type window, and runs prefill on the complementary
    # prefill-half, so releasing an insensitive layer's SM hands prefill the exact complement.
    def forward_split_decode(
        self, input_ids, positions, forward_batch, split_interval, input_embeds=None
    ):
        return self.forward_split_prefill(
            input_ids, positions, forward_batch, split_interval, input_embeds
        )

    def la_coord_windows(self):
        """Maximal same-type layer runs: list of (start, end, is_attn). Zamba2 = ABAB;
        attn windows -> decode-heavy coordinated pair (protect), mamba -> decode-light (release)."""
        layers = self.model.layers
        types = [isinstance(l, Zamba2HybridLayer) for l in layers]
        windows = []
        i, n = 0, len(types)
        while i < n:
            j = i
            while j < n and types[j] == types[i]:
                j += 1
            windows.append((i, j, types[i]))
            i = j
        return windows

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        # PARITY: HF Zamba2 weight-name remaps (verify against real checkpoint):
        #   A_log -> mamba.A ; LoRA Sequential .0.weight/.1.weight -> .A.weight/.B.weight
        params_dict = dict(self.named_parameters())
        loaded = set()
        skipped = []
        n_in = 0
        for name, w in weights:
            n_in += 1
            name = name.replace("A_log", "A")
            name = name.replace(".0.weight", ".A.weight").replace(".1.weight", ".B.weight")
            for param_name, shard_name, shard_id in self.stacked_params_mapping:
                if shard_name not in name:
                    continue
                mapped = name.replace(shard_name, param_name)
                if mapped not in params_dict:
                    continue
                param = params_dict[mapped]
                param.weight_loader(param, w, shard_id)
                loaded.add(mapped)
                break
            else:
                if name not in params_dict:
                    skipped.append(name)
                    continue
                param = params_dict[name]
                getattr(param, "weight_loader", default_weight_loader)(param, w)
                loaded.add(name)
        missing = [p for p in params_dict if p not in loaded]
        if skipped or missing:
            import logging as _lg
            _lg.getLogger("sglang.srt.models.zamba2").warning(
                "Zamba2 load: in=%d loaded=%d skipped=%d missing=%d; skipped[:6]=%s missing[:6]=%s",
                n_in, len(loaded), len(skipped), len(missing), skipped[:6], missing[:6])
        return loaded


EntryClass = [Zamba2ForCausalLM]
