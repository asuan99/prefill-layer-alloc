#!/usr/bin/env python3
# Copyright 2025 SGLang Team / prefill-layer-alloc engine-port.
"""Adapt a native state-spaces mamba_ssm checkpoint into an HF-format wrapper
directory that sglang can load as arch=Mamba2ForCausalLM (engine-port Stage 0
pure-SSM negative-control arm).

The native repo (e.g. state-spaces/mamba2-2.7b) ships:
  - config.json  : native mamba_ssm fields (d_model/n_layer/ssm_cfg, NO
                   `architectures`, NO `model_type`) -> AutoConfig.from_pretrained
                   CANNOT load it.
  - pytorch_model.bin : native `backbone.*` key layout.
  - NO tokenizer (state-spaces models use the EleutherAI/gpt-neox-20b tokenizer).

This writes <out_dir> containing:
  - config.json  : model_type="mamba2", architectures=["Mamba2ForCausalLM"], with
                   dims derived from BOTH the native config and the actual weight
                   shapes in the .bin (source of truth for d_inner/nheads/ngroups).
  - pytorch_model.bin -> symlink to the native .bin (NO weight rewrite; the
                   Mamba2ForCausalLM.load_weights maps `backbone.*` keys in-code).
  - tokenizer files : copied from --tokenizer (default EleutherAI/gpt-neox-20b)
                   if resolvable locally/offline; else a warning + instruction to
                   pass --tokenizer-path at launch.

Usage:
  python convert_mamba2_native.py \
      --src  <hf_cache>/hub/models--state-spaces--mamba2-2.7b/snapshots/<rev> \
      --out  <hf_cache>/mamba2-2.7b-sglang \
      [--tokenizer EleutherAI/gpt-neox-20b]
"""
import argparse
import json
import os
import shutil
import sys

import torch


def derive_dims(bin_path: str, native: dict) -> dict:
    sd_keys = torch.load(bin_path, map_location="cpu", weights_only=True)
    # Read shapes from layer 0.
    def shape(k):
        return tuple(sd_keys[k].shape)

    hidden = shape("backbone.embedding.weight")[1]
    vocab = shape("backbone.embedding.weight")[0]  # already padded (e.g. 50288)
    nheads = shape("backbone.layers.0.mixer.A_log")[0]
    d_inner = shape("backbone.layers.0.mixer.norm.weight")[0]  # gated-norm over d_inner
    head_dim = d_inner // nheads
    conv_dim = shape("backbone.layers.0.mixer.conv1d.weight")[0]
    conv_kernel = shape("backbone.layers.0.mixer.conv1d.weight")[2]
    # conv_dim = d_inner + 2*ngroups*d_state  ; state-spaces default d_state=128.
    d_state = int(native.get("ssm_cfg", {}).get("d_state", 128))
    ngroups = (conv_dim - d_inner) // (2 * d_state)
    n_layer = int(native.get("n_layer", 0)) or (
        max(int(k.split(".")[2]) for k in sd_keys if k.startswith("backbone.layers.")) + 1
    )
    expand = max(1, round(d_inner / hidden))
    del sd_keys
    return dict(
        vocab_size=int(vocab),
        hidden_size=int(hidden),
        num_hidden_layers=int(n_layer),
        mamba_num_heads=int(nheads),
        mamba_head_dim=int(head_dim),
        ssm_state_size=int(d_state),
        mamba_n_groups=int(ngroups),
        mamba_d_conv=int(conv_kernel),
        mamba_expand=int(expand),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="native checkpoint snapshot dir")
    ap.add_argument("--out", required=True, help="output wrapper dir")
    ap.add_argument("--tokenizer", default="EleutherAI/gpt-neox-20b")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    src = os.path.abspath(args.src)
    out = os.path.abspath(args.out)
    native_cfg_path = os.path.join(src, "config.json")
    bin_path = os.path.join(src, "pytorch_model.bin")
    for p in (native_cfg_path, bin_path):
        if not os.path.exists(p):
            sys.exit(f"ERROR: missing {p}")

    native = json.load(open(native_cfg_path))
    dims = derive_dims(bin_path, native)

    os.makedirs(out, exist_ok=True)
    cfg = {
        "model_type": "mamba2_ssm",
        "architectures": ["Mamba2ForCausalLM"],
        "torch_dtype": "float16",
        "tie_word_embeddings": bool(native.get("tie_embeddings", True)),
        "layer_norm_epsilon": float(native.get("rms_norm_eps", 1e-5)),
        "residual_in_fp32": bool(native.get("residual_in_fp32", True)),
        "mamba_hidden_act": "silu",
        "mamba_conv_bias": True,
        "mamba_proj_bias": False,
        "mamba_chunk_size": 256,
        "pad_token_id": 0,
        "bos_token_id": 0,
        "eos_token_id": 0,
        **dims,
    }
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=2)

    # symlink weights (no rewrite)
    dst_bin = os.path.join(out, "pytorch_model.bin")
    if os.path.lexists(dst_bin):
        if args.force:
            os.remove(dst_bin)
        else:
            sys.exit(f"ERROR: {dst_bin} exists (use --force)")
    os.symlink(bin_path, dst_bin)

    # tokenizer
    tok_ok = False
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        tok.save_pretrained(out)
        tok_ok = True
    except Exception as e:  # offline / not cached
        print(
            f"WARNING: could not resolve tokenizer '{args.tokenizer}' ({e}).\n"
            f"  -> launch with:  --tokenizer-path {args.tokenizer}\n"
            f"     (state-spaces/mamba2-* use the GPT-NeoX-20B tokenizer.)",
            file=sys.stderr,
        )

    print(json.dumps({"out": out, "config": cfg, "tokenizer_written": tok_ok}, indent=2))
    print("OK: wrapper ready. Launch with --model-path", out)


if __name__ == "__main__":
    main()
