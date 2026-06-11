"""
chunked_attn_runner.py — chunked-prefill Attention measurement core (Phase 2).

Granularity-matched counterpart to chunked_ssm_runner.py.  The existing attn
sweep (run_attn_prefill_sweep.py → LayerRunner.run_attn_layer) processes the whole
sequence in ONE scaled_dot_product_attention call, while the SSM side is measured
in prefill_chunk_tokens-sized chunks (chunked_ssm_runner.py).  That granularity
mismatch is the chief confound of the asymmetry result (Phase 0, item 3).

This module reproduces real chunked-prefill serving for attention:
  * the sequence is split into prefill_chunk_tokens-sized chunks;
  * chunk i issues q_len=chunk attention against a CUMULATIVE KV cache of length
    chunk×(i+1) (causal, bottom-right aligned — same as the full-seq path);
  * K/V are appended into a pre-allocated contiguous cache each chunk;
  * one CUDA-event pair brackets the WHOLE chunk loop — identical latency
    definition to the SSM chunked path (sum over chunks).

Backend: torch.nn.functional.scaled_dot_product_attention(is_causal=True) — the
SAME kernel the existing attn path uses as its SDPA backend (no new backend; the
flashinfer ragged wrapper would need a per-chunk replan and is not used here).
"""

import sys
import os
_here = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_here, "../.."))  # workspace/
sys.path.insert(0, os.path.join(_here, ".."))     # characterization/

import math
import statistics

import torch
import torch.nn as nn
import torch.nn.functional as F

from shared.loaders import get_model_config
from src.smctrl.green_ctx_controller import SMController
from src.profiling.metrics import BandwidthEstimator


def _attn_cfg(model_name: str) -> dict:
    raw = get_model_config(model_name)
    attn = raw["attention"]
    return {
        "hidden_size": raw["hidden_size"],
        "n_heads":     attn["num_heads"],
        "n_kv_heads":  attn["num_kv_heads"],
        "head_dim":    attn["head_dim"],
    }


def run_chunked_attn_sweep(
    model_name: str,
    seq_len: int,
    batch_size: int,
    prefill_chunk_tokens: int,
    sm_count: int,
    smctrl: SMController,
    n_warmup: int = 3,
    n_measure: int = 10,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> dict:
    """Measure chunked-prefill attention latency under Green Context.

    Returns raw measurements; the driver maps them onto the unified chunked
    schema via shared.sweep_spec.chunked_row().
    """
    cfg        = _attn_cfg(model_name)
    hidden     = cfg["hidden_size"]
    n_heads    = cfg["n_heads"]
    n_kv       = cfg["n_kv_heads"]
    head_dim   = cfg["head_dim"]
    attn_hidden = n_heads * head_dim
    kv_hidden   = n_kv * head_dim
    bpe = 2

    chunk   = prefill_chunk_tokens
    n_calls = math.ceil(seq_len / chunk)

    # Projection weights (allocated once; values irrelevant to latency).
    with torch.no_grad():
        w_q = nn.Linear(hidden, attn_hidden, bias=False, dtype=dtype).to(device)
        w_k = nn.Linear(hidden, kv_hidden,   bias=False, dtype=dtype).to(device)
        w_v = nn.Linear(hidden, kv_hidden,   bias=False, dtype=dtype).to(device)
        w_o = nn.Linear(attn_hidden, hidden, bias=False, dtype=dtype).to(device)

    # Full hidden states; per-chunk slices feed the projections.
    hidden_states = torch.randn(batch_size, seq_len, hidden, device=device, dtype=dtype)

    # Pre-allocated contiguous KV cache (max kv_len = seq_len). Appended per chunk.
    k_cache = torch.empty(batch_size, seq_len, n_kv, head_dim, device=device, dtype=dtype)
    v_cache = torch.empty(batch_size, seq_len, n_kv, head_dim, device=device, dtype=dtype)

    rep = n_heads // n_kv if n_kv != n_heads else 1

    proj_weight_bytes = (
        w_q.weight.numel() + w_k.weight.numel()
        + w_v.weight.numel() + w_o.weight.numel()
    ) * bpe

    # Derived HBM bytes: sum over chunks (cumulative KV grows each chunk).
    read_bytes = write_bytes = 0
    for i in range(n_calls):
        start = i * chunk
        q_len = min(chunk, seq_len - start)
        kv_len = start + q_len
        r, w = BandwidthEstimator.attn_bytes(
            batch=batch_size, seq_len=q_len, total_kv_len=kv_len,
            n_heads=n_heads, n_kv_heads=n_kv, head_dim=head_dim,
            hidden_size=hidden, proj_weight_bytes=proj_weight_bytes, bytes_per_elem=bpe,
        )
        read_bytes += r
        write_bytes += w

    smctrl.set_sm_count(sm_count)
    stream = smctrl.get_stream()

    def _run_once() -> None:
        with torch.cuda.stream(stream), torch.no_grad():
            for i in range(n_calls):
                start = i * chunk
                q_len = min(chunk, seq_len - start)
                end = start + q_len
                h = hidden_states[:, start:end, :]

                q = w_q(h).view(batch_size, q_len, n_heads, head_dim).transpose(1, 2)
                k_new = w_k(h).view(batch_size, q_len, n_kv, head_dim)
                v_new = w_v(h).view(batch_size, q_len, n_kv, head_dim)

                # Append into the pre-allocated cache, then attend over [0, end).
                k_cache[:, start:end] = k_new
                v_cache[:, start:end] = v_new
                k = k_cache[:, :end].transpose(1, 2)
                v = v_cache[:, :end].transpose(1, 2)
                if rep > 1:
                    k = k.repeat_interleave(rep, dim=1)
                    v = v.repeat_interleave(rep, dim=1)

                # is_causal=True with kv_len>q_len → bottom-right aligned causal
                # mask: query at global pos p attends to keys 0..p (chunked-prefill
                # semantics, identical to the full-seq path).
                out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                out = out.transpose(1, 2).reshape(batch_size, q_len, attn_hidden)
                w_o(out)
        torch.cuda.synchronize()

    for _ in range(n_warmup):
        _run_once()

    latencies: list[float] = []
    for _ in range(n_measure):
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record(stream)
        _run_once()
        t1.record(stream)
        torch.cuda.synchronize()
        latencies.append(t0.elapsed_time(t1))

    smctrl.reset()

    return {
        "latency_ms":           statistics.median(latencies),
        "latency_std_ms":       statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
        "latency_list_ms":      latencies,
        "n_kernel_calls":       n_calls,
        "read_bytes":           read_bytes,
        "write_bytes":          write_bytes,
        # SDPA is not a cooperative-grid kernel → no barrier deadlock risk.
        "cooperative_safe":     True,
        # Cross-chunk state is carried explicitly via the KV cache append.
        "state_passing_active": True,
        "sm_count":             sm_count,
        "seq_len":              seq_len,
        "batch_size":           batch_size,
        "prefill_chunk_tokens": chunk,
    }
