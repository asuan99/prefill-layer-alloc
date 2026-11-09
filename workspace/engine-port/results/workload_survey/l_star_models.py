#!/usr/bin/env python3
"""Memory crossover L* and decode-traffic crossover c* for public hybrid models, via analytic_terms (analytic-v1, no formula change).
Usage: l_star_models.py <cfg_dir> <w1_lengths.json> <w3_inputs.json> <loogle_inputs.json> [hf_cache_config.json ...]"""
import glob, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts", "costmodel"))
import analytic_terms as T

def arch(c):
    t = c["model_type"]
    if t == "nemotron_h":
        return T.arch_from_config(c)
    d, L = c["hidden_size"], c["num_hidden_layers"]
    if t == "granitemoehybrid":
        ty = c["layer_types"]
        nA, nM, nF = ty.count("attention"), ty.count("mamba"), L
    elif t == "falcon_h1":
        nA, nM, nF = L, L, L            # SGLang: attention and mamba on all layers (configs/falcon_h1.py full_attention_layer_ids / linear_layer_ids)
    else:
        raise KeyError(t)
    dh = c.get("head_dim") or d // c["num_attention_heads"]
    return T.Arch(d=d, n_q=c["num_attention_heads"], n_kv=c["num_key_value_heads"], d_h=dh, n_h=c["mamba_n_heads"], p=c["mamba_d_head"],
                  G=c["mamba_n_groups"], N=c["mamba_d_state"], K=c["mamba_d_conv"], Q=c.get("mamba_chunk_size", 128),
                  d_ff=c["intermediate_size"], V=c["vocab_size"], n_A=nA, n_M=nM, n_F=nF)

def main():
    cfgd, w1p, w3p, lgp = sys.argv[1:5]
    w1 = [x[0] for x in json.load(open(w1p))]; w3 = json.load(open(w3p)); lg = json.load(open(lgp))
    share = lambda v, th: sum(x > th for x in v) / len(v)
    files = sorted(glob.glob(cfgd + "/*.json")) + sys.argv[5:]
    print("model | nA/nM | s_MB | k_B/tok | L* | c* | slots48_GB | >L*: W1 W3 LooGLE | ssm dtype override")
    for f in files:
        try: c = json.load(open(f))
        except Exception: continue
        if c.get("model_type") not in ("nemotron_h", "granitemoehybrid", "falcon_h1"): continue
        a = arch(c); s, k, Ls = T.state_bytes_per_request(a), T.kv_bytes_per_token(a), T.L_star(a)
        cs = T.decode_step(a, 32, 8192)["c_state_eq_kv"]
        name = c.get("_name_or_path") or f
        name = os.path.basename(f).replace(".json", "") if f.startswith(cfgd) else f.split("models--")[1].split("/")[0]
        print(f"{name} | {a.n_A}/{a.n_M} | {s/1e6:.1f} | {k} | {Ls:.0f} | {cs:.0f} | {48*s/1e9:.2f} | {share(w1,Ls):.3f} {share(w3,Ls):.3f} {share(lg,Ls):.3f} | {c.get('mamba_ssm_dtype')}")
    print(T.VERSION)
    for r in T.table():
        print(r["variant"], r["n_A"], r["n_M"], round(r["s_bytes"]/1e6,1), r["k_bytes_per_token"], round(r["L_star"]), round(r["decode_c_state_eq_kv"]))
main()
