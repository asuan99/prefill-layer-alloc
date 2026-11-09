#!/usr/bin/env python3
"""Memory crossover for non-Mamba linear hybrids. State shapes follow SGLang v0.5.10 configs/mamba_utils.py and each model's mamba2_cache_params
(float32 temporal, bf16 conv). Qwen3-Next maps onto analytic_terms.Arch exactly as SGLang maps it onto Mamba2StateShape; the others use the
explicit shape below. analytic-v1 itself is defined for Mamba2 and is not changed here. Usage: l_star_linear_models.py <cfg_dir>"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts", "costmodel"))
import analytic_terms as T
d = sys.argv[1]
load = lambda n: json.load(open(os.path.join(d, n + ".json")))
def row(name, kind, nA, nL, s_layer, k, note=""):
    s = nL * s_layer
    print(f"{name} | {kind} | {nA}/{nL} | {s/1e6:.2f} | {k} | {s/k:.0f} | {2*s/k:.0f} | {note}")
print("model | linear layer | nA/nL | s_MB | k_B/tok | L* | c* | note")
c = load("Qwen_Qwen3-Next-80B-A3B-Instruct"); L = c["num_hidden_layers"]; iv = c.get("full_attention_interval", 4)
nA = sum((l + 1) % iv == 0 for l in range(L)); nL = L - nA
a = T.Arch(d=c["hidden_size"], n_q=c["num_attention_heads"], n_kv=c["num_key_value_heads"], d_h=c["head_dim"], n_h=c["linear_num_value_heads"],
           p=c["linear_value_head_dim"], G=c["linear_num_key_heads"], N=c["linear_key_head_dim"], K=c["linear_conv_kernel_dim"], Q=64,
           d_ff=c["intermediate_size"], V=c["vocab_size"], n_A=nA, n_M=nL, n_F=L)
row("Qwen3-Next-80B", "Gated DeltaNet", nA, nL, T.state_bytes_per_request(a) / nL, T.kv_bytes_per_token(a), f"interval {iv}")
c = load("moonshotai_Kimi-Linear-48B-A3B-Instruct"); la = c["linear_attn_config"]; nA, nL = len(la["full_attn_layers"]), len(la["kda_layers"])
h, p, K = la["num_heads"], la["head_dim"], la["short_conv_kernel_size"]
sl = h * p * p * 4 + (K - 1) * (h * p + 2 * h * p) * 2
row("Kimi-Linear-48B", "KDA", nA, nL, sl, nA * 2 * c["num_key_value_heads"] * c["head_dim"] * 2, "num_k_heads=num_heads")
for n, nm in (("LiquidAI_LFM2-2.6B", "LFM2-2.6B"), ("LiquidAI_LFM2-8B-A1B", "LFM2-8B-A1B")):
    c = load(n); ty = c["layer_types"]; nA, nL = ty.count("full_attention"), ty.count("conv")
    sl = c["hidden_size"] * (int(c["conv_L_cache"]) - 1) * 2
    row(nm, "short convolution only", nA, nL, sl, nA * 2 * c["num_key_value_heads"] * (c["hidden_size"] // c["num_attention_heads"]) * 2, "no recurrent state")
c = load("inclusionAI_Ling-2.5-1T"); L = c["num_hidden_layers"]; g = c["layer_group_size"]
nA = sum((l + 1) % g == 0 for l in range(L)); nL = L - nA
row("Ling-2.5-1T", "linear attention (bailing_hybrid)", nA, nL, c["num_attention_heads"] * c["head_dim"] ** 2 * 4, nA * 2 * c["num_key_value_heads"] * c["head_dim"] * 2, "conv_kernel 1")
for n, nm in (("Qwen_Qwen3.5-4B", "Qwen3.5-4B"), ("Qwen_Qwen3.5-9B", "Qwen3.5-9B"), ("Qwen_Qwen3.5-27B", "Qwen3.5-27B"), ("allenai_Olmo-Hybrid-7B", "Olmo-Hybrid-7B")):
    c = load(n); c = c.get("text_config", c); ty = c["layer_types"]; nA, nL = ty.count("full_attention"), ty.count("linear_attention")
    a = T.Arch(d=c["hidden_size"], n_q=c["num_attention_heads"], n_kv=c["num_key_value_heads"], d_h=c.get("head_dim") or c["hidden_size"] // c["num_attention_heads"],
               n_h=c["linear_num_value_heads"], p=c["linear_value_head_dim"], G=c["linear_num_key_heads"], N=c["linear_key_head_dim"], K=c["linear_conv_kernel_dim"], Q=64,
               d_ff=c["intermediate_size"], V=c["vocab_size"], n_A=nA, n_M=nL, n_F=len(ty))
    row(nm, "Gated DeltaNet", nA, nL, T.state_bytes_per_request(a) / nL, T.kv_bytes_per_token(a), "")
