#!/usr/bin/env python3
# Copyright 2025 SGLang Team / prefill-layer-alloc engine-port.
"""Adapt a *transformers-format* Mamba2 checkpoint into an sglang wrapper dir.

Companion to convert_mamba2_native.py (which handles the state-spaces native
layout).  This one targets HF exports whose config.json is a transformers
`Mamba2Config` -- notably **mistralai/Mamba-Codestral-7B-v0.1**, the 7B
pure-SSM arm of the 7-8B scale-up (state-spaces tops out at 2.7B;
nvidia/mamba2-8b ships a Megatron checkpoint; falcon-mamba-7b is Mamba-1).

Two incompatibilities are handled:
  1. field naming: transformers uses num_heads/head_dim/state_size/n_groups/
     conv_kernel; sglang's Mamba2Config (configs/mamba2.py, model_type
     "mamba2_ssm") expects mamba_num_heads/mamba_head_dim/ssm_state_size/
     mamba_n_groups/mamba_d_conv.
  2. tensor naming: `backbone.embeddings.` (plural) vs the native
     `backbone.embedding.` -- handled in models/mamba2.py load_weights.

Dims are derived from the actual safetensors headers, not from config.json,
so a mislabelled config cannot silently produce a wrong model.

Usage:
  python make_mamba2_hf_sglang.py \
      --src <hf_cache>/hub/models--mistralai--Mamba-Codestral-7B-v0.1/snapshots/<rev> \
      --out <hf_cache>/mamba-codestral-7b-sglang
"""
import argparse
import json
import os
import struct
import sys


def safetensors_shapes(path):
    """Read only the safetensors header (8-byte length + JSON)."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    return {k: v["shape"] for k, v in header.items() if k != "__metadata__"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    src, out = os.path.abspath(args.src), os.path.abspath(args.out)
    native = json.load(open(os.path.join(src, "config.json")))
    shards = sorted(
        f for f in os.listdir(src)
        if f.endswith(".safetensors") and f.startswith("model-")
    )
    if not shards:
        sys.exit(f"no sharded safetensors under {src}")

    shapes = {}
    for s in shards:
        shapes.update(safetensors_shapes(os.path.join(src, s)))

    def shape(*cands):
        for k in cands:
            if k in shapes:
                return shapes[k]
        sys.exit(f"missing tensor: {cands}")

    emb = shape("backbone.embeddings.weight", "backbone.embedding.weight")
    vocab, hidden = emb[0], emb[1]
    nheads = shape("backbone.layers.0.mixer.A_log")[0]
    d_inner = shape("backbone.layers.0.mixer.norm.weight")[0]
    conv_w = shape("backbone.layers.0.mixer.conv1d.weight")
    conv_dim, conv_kernel = conv_w[0], conv_w[2]
    head_dim = d_inner // nheads
    d_state = int(native.get("state_size", native.get("ssm_state_size", 128)))
    ngroups = (conv_dim - d_inner) // (2 * d_state)
    n_layer = max(
        int(k.split(".")[2]) for k in shapes if k.startswith("backbone.layers.")
    ) + 1
    expand = max(1, round(d_inner / hidden))
    tie = bool(native.get("tie_word_embeddings", False))

    if head_dim * nheads != d_inner or ngroups < 1:
        sys.exit(f"derived dims inconsistent: {nheads=} {head_dim=} {d_inner=} {ngroups=}")

    cfg = {
        "model_type": "mamba2_ssm",
        "architectures": ["Mamba2ForCausalLM"],
        "torch_dtype": native.get("torch_dtype", "bfloat16"),
        "tie_word_embeddings": tie,
        "layer_norm_epsilon": native.get("layer_norm_epsilon", 1e-5),
        "residual_in_fp32": native.get("residual_in_fp32", True),
        "mamba_hidden_act": native.get("hidden_act", "silu"),
        "mamba_conv_bias": native.get("use_conv_bias", True),
        "mamba_proj_bias": native.get("use_bias", False),
        "mamba_chunk_size": native.get("chunk_size", 256),
        "pad_token_id": native.get("pad_token_id", 0),
        "bos_token_id": native.get("bos_token_id", 0),
        "eos_token_id": native.get("eos_token_id", 0),
        "vocab_size": vocab,
        "hidden_size": hidden,
        "num_hidden_layers": n_layer,
        "mamba_num_heads": nheads,
        "mamba_head_dim": head_dim,
        "ssm_state_size": d_state,
        "mamba_n_groups": ngroups,
        "mamba_d_conv": conv_kernel,
        "mamba_expand": expand,
    }

    os.makedirs(out, exist_ok=True)
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=2)
    carry = shards + [
        f for f in ("model.safetensors.index.json", "tokenizer.json",
                    "tokenizer_config.json", "special_tokens_map.json",
                    "tokenizer.model", "generation_config.json")
        if os.path.exists(os.path.join(src, f))
    ]
    for f in carry:
        dst = os.path.join(out, f)
        if os.path.islink(dst) or os.path.exists(dst):
            os.remove(dst)
        os.symlink(os.path.realpath(os.path.join(src, f)), dst)

    print(json.dumps(cfg, indent=2))
    print(f"\nwrote {out}  ({len(shards)} shards symlinked, tie_word_embeddings={tie})")
    print(f"derived from safetensors headers: conv_dim={conv_dim} d_inner={d_inner} "
          f"nheads={nheads} ngroups={ngroups} n_layer={n_layer}")


if __name__ == "__main__":
    main()
