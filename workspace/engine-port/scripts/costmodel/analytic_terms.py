#!/usr/bin/env python3
"""Analytic cost terms of a hybrid (Attention / Mamba2-SSD / MLP) LLM -- the fixed formula set (decision 27, 2026-10-22).

This module is the single source of the closed-form FLOP, resident-memory and decode-traffic expressions that
`reports/cost_model/ANALYTIC_COST_FORMULAS.md` states.  The document quotes; this code computes.  Change a formula
here only together with that document (and its version line) -- the paper uses these expressions.

Conventions (identical in the document):
  * FLOPs count a multiply and an add separately: a dense matmul costs 2 x params x tokens.
  * beta = bytes per element: weights / KV / conv state in bf16 (2); SGLang keeps the SSM temporal state in fp32 (4)
    (`mamba2_state_dtype`, srt/configs/mamba_utils.py; checked against the boot banner, compo probe 2026-10-21).
  * Attention is causal GQA; MLP is non-gated (Nemotron-H relu2: up + down) unless `mlp_gated`.
  * Mamba2 = SSD with chunk length Q; B and C are shared within each of the G groups.
  * Embedding lookup has no FLOPs; the LM head costs 2*d*V per *sampled* token (SGLang samples only the last prefill
    position, so a prefill pays it once).
All numbers produced here are [derived] (closed-form), never measurements.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict

VERSION = "analytic-v1 (2026-10-22)"


@dataclass(frozen=True)
class Arch:
    d: int                 # hidden size
    n_q: int               # query heads
    n_kv: int              # KV heads
    d_h: int               # head dim
    n_h: int               # mamba heads
    p: int                 # mamba head dim  (d_in = n_h * p)
    G: int                 # mamba groups
    N: int                 # SSM state dim
    K: int                 # conv kernel
    Q: int                 # SSD chunk length
    d_ff: int              # MLP intermediate
    V: int                 # vocab
    n_A: int               # attention layers
    n_M: int               # mamba layers
    n_F: int               # MLP layers
    mlp_gated: bool = False
    tied_embeddings: bool = False
    beta: int = 2          # bf16
    beta_ssm: int = 4      # fp32 SSM state (SGLang default)

    @property
    def d_in(self) -> int:
        return self.n_h * self.p

    @property
    def conv_dim(self) -> int:
        return self.d_in + 2 * self.G * self.N


def nano9b_v2(n_A: int = 4, n_M: int = 27) -> Arch:
    """Nemotron-Nano-9B-v2-Base (rev dc0661c...) config; n_A/n_M override for the composition variants (decision 24)."""
    return Arch(d=4480, n_q=40, n_kv=8, d_h=128, n_h=128, p=80, G=8, N=128, K=4, Q=128, d_ff=15680, V=131072,
                n_A=n_A, n_M=n_M, n_F=25)


def arch_from_config(cfg: dict) -> Arch:
    pat = cfg["hybrid_override_pattern"]
    return Arch(d=cfg["hidden_size"], n_q=cfg["num_attention_heads"], n_kv=cfg["num_key_value_heads"],
                d_h=cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"],
                n_h=cfg["mamba_num_heads"], p=cfg["mamba_head_dim"], G=cfg.get("n_groups", cfg.get("mamba_num_groups")),
                N=cfg.get("ssm_state_size", cfg.get("mamba_state_dim")), K=cfg["conv_kernel"],
                Q=cfg.get("chunk_size", 128), d_ff=cfg["intermediate_size"], V=cfg["vocab_size"],
                n_A=pat.count("*"), n_M=pat.count("M"), n_F=pat.count("-"),
                mlp_gated=False, tied_embeddings=bool(cfg.get("tie_word_embeddings", False)))


# ---------------------------------------------------------------- parameters (per layer)
def params_attn(a: Arch) -> int:
    """W_A = d*d_h*(n_q + 2 n_kv)  [qkv]  +  n_q*d_h*d  [o];  no bias."""
    return a.d * a.d_h * (a.n_q + 2 * a.n_kv) + a.n_q * a.d_h * a.d


def params_mamba_matmul(a: Arch) -> int:
    """W_M = d*(2 d_in + 2 G N + n_h)  [in_proj: z, x, B, C, dt]  +  d_in*d  [out_proj]."""
    return a.d * (2 * a.d_in + 2 * a.G * a.N + a.n_h) + a.d_in * a.d


def params_mamba_small(a: Arch) -> int:
    """conv weight+bias, A, D, dt_bias, gated-norm weight (no FLOP-relevant matmul)."""
    return a.conv_dim * a.K + a.conv_dim + 3 * a.n_h + a.d_in


def params_mlp(a: Arch) -> int:
    """W_F = 2 d d_ff (non-gated)  or 3 d d_ff (gated)."""
    return (3 if a.mlp_gated else 2) * a.d * a.d_ff


def params_total(a: Arch) -> int:
    layers = a.n_A + a.n_M + a.n_F
    emb = a.V * a.d * (1 if a.tied_embeddings else 2)
    return (a.n_A * params_attn(a) + a.n_M * (params_mamba_matmul(a) + params_mamba_small(a))
            + a.n_F * params_mlp(a) + layers * a.d + a.d + emb)            # + per-layer pre-norm + final norm


def params_nonembed_matmul(a: Arch) -> int:
    return a.n_A * params_attn(a) + a.n_M * params_mamba_matmul(a) + a.n_F * params_mlp(a)


# ---------------------------------------------------------------- FLOPs (per layer)
def flops_attn_proj_per_token(a: Arch) -> int:
    return 2 * params_attn(a)


def flops_attn_core_prefill(a: Arch, L: int) -> int:
    """Causal QK^T + AV over one request of length L: 2 * n_q * d_h * L^2 (half-triangle, both matmuls)."""
    return 2 * a.n_q * a.d_h * L * L


def flops_attn_core_decode(a: Arch, c: int) -> int:
    """One decode token attending to c cached tokens: 4 * n_q * d_h * c."""
    return 4 * a.n_q * a.d_h * c


def flops_mamba_proj_per_token(a: Arch) -> int:
    return 2 * params_mamba_matmul(a)


def flops_mamba_conv_per_token(a: Arch) -> int:
    return 2 * a.K * a.conv_dim


def flops_ssd_prefill_per_token(a: Arch) -> int:
    """Chunked SSD per token: intra-chunk C B^T (2 G Q N) + its application to X (2 n_h Q p)
    + chunk-state B^T X (2 n_h p N) + inter-chunk output C h (2 n_h p N).  Inter-chunk scan is O(n_h p N / Q), dropped."""
    return 2 * a.G * a.Q * a.N + 2 * a.n_h * a.Q * a.p + 4 * a.n_h * a.p * a.N


def flops_ssm_decode_per_token(a: Arch) -> int:
    """Recurrent update h <- exp(dt A) h + dt B x (3 p N per head) and y = C h (2 p N per head): 5 d_in N."""
    return 5 * a.d_in * a.N


def flops_mlp_per_token(a: Arch) -> int:
    return 2 * params_mlp(a)


def flops_lm_head_per_token(a: Arch) -> int:
    return 2 * a.d * a.V


# ---------------------------------------------------------------- resident memory
def kv_bytes_per_token(a: Arch) -> int:
    """k = n_A * 2 * n_kv * d_h * beta."""
    return a.n_A * 2 * a.n_kv * a.d_h * a.beta


def ssm_state_bytes_per_layer(a: Arch) -> int:
    return a.n_h * a.p * a.N * a.beta_ssm


def conv_state_bytes_per_layer(a: Arch) -> int:
    return a.conv_dim * (a.K - 1) * a.beta


def state_bytes_per_request(a: Arch) -> int:
    """s = n_M * (n_h p N beta_ssm + (d_in + 2 G N)(K-1) beta) -- independent of length."""
    return a.n_M * (ssm_state_bytes_per_layer(a) + conv_state_bytes_per_layer(a))


def L_star(a: Arch) -> float:
    k = kv_bytes_per_token(a)
    return float("inf") if k == 0 else state_bytes_per_request(a) / k


# ---------------------------------------------------------------- whole model
def prefill_flops(a: Arch, L: int) -> dict:
    """One request of length L (no prefix reuse).  Returns linear coefficient, quadratic coefficient and total."""
    lin = (a.n_A * flops_attn_proj_per_token(a)
           + a.n_M * (flops_mamba_proj_per_token(a) + flops_mamba_conv_per_token(a) + flops_ssd_prefill_per_token(a))
           + a.n_F * flops_mlp_per_token(a))
    quad = a.n_A * 2 * a.n_q * a.d_h                     # coefficient of L^2
    total = lin * L + quad * L * L + flops_lm_head_per_token(a)
    return {"linear_per_token": lin, "quad_coeff": quad, "total": total,
            "quad_share_at_L": (quad * L) / lin, "L_quad_eq_linear": lin / quad if quad else float("inf")}


def decode_step(a: Arch, b: int, c: int) -> dict:
    """One decode step, batch b, every request at context c.  FLOPs and HBM bytes (weights read once per step)."""
    lin_tok = (a.n_A * flops_attn_proj_per_token(a)
               + a.n_M * (flops_mamba_proj_per_token(a) + flops_mamba_conv_per_token(a) + flops_ssm_decode_per_token(a))
               + a.n_F * flops_mlp_per_token(a) + flops_lm_head_per_token(a))
    flops = b * (lin_tok + a.n_A * flops_attn_core_decode(a, c))
    w_bytes = (params_nonembed_matmul(a) + a.V * a.d) * a.beta        # + LM head; embedding gather ignored
    ssm_rw = b * a.n_M * 2 * (ssm_state_bytes_per_layer(a) + conv_state_bytes_per_layer(a))   # read + write
    kv_read = b * c * kv_bytes_per_token(a)
    total = w_bytes + ssm_rw + kv_read
    return {"flops": flops, "bytes_weights": w_bytes, "bytes_state_rw": ssm_rw, "bytes_kv_read": kv_read,
            "bytes_total": total, "intensity_flop_per_byte": flops / total,
            "c_state_eq_kv": (a.n_M * 2 * (ssm_state_bytes_per_layer(a) + conv_state_bytes_per_layer(a)))
                             / kv_bytes_per_token(a) if kv_bytes_per_token(a) else float("inf")}


def decode_coefficients(a: Arch, hbm_bytes_per_s: float) -> dict:
    """Memory-bound lower bounds for T_dec = a0 + b0*bs + c0*bs*ctx (PREREG T1 form), seconds."""
    w = (params_nonembed_matmul(a) + a.V * a.d) * a.beta
    per_req = a.n_M * 2 * (ssm_state_bytes_per_layer(a) + conv_state_bytes_per_layer(a))
    return {"a0_s": w / hbm_bytes_per_s, "b0_s_per_req": per_req / hbm_bytes_per_s,
            "c0_s_per_req_token": kv_bytes_per_token(a) / hbm_bytes_per_s}


def table(variants=((4, 27), (13, 18), (22, 9), (31, 0))) -> list[dict]:
    rows = []
    for nA, nM in variants:
        a = nano9b_v2(nA, nM)
        pf = prefill_flops(a, 65536)
        dc = decode_step(a, 32, 8192)
        rows.append({"variant": f"a{nA}", "n_A": nA, "n_M": nM, "params_total": params_total(a),
                     "s_bytes": state_bytes_per_request(a), "k_bytes_per_token": kv_bytes_per_token(a),
                     "L_star": L_star(a), "prefill_linear_flop_per_token": pf["linear_per_token"],
                     "prefill_quad_coeff": pf["quad_coeff"], "L_quad_eq_linear": pf["L_quad_eq_linear"],
                     "quad_share_at_64k": pf["quad_share_at_L"], "decode_c_state_eq_kv": dc["c_state_eq_kv"],
                     "decode_b32_c8k_bytes": dc["bytes_total"], "decode_b32_c8k_intensity": dc["intensity_flop_per_byte"]})
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    rows = table()
    if args.json:
        print(json.dumps({"version": VERSION, "rows": rows}, indent=2))
        return 0
    print(VERSION)
    for r in rows:
        print("  ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
