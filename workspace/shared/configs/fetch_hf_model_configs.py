"""
fetch_hf_model_configs.py — pull architecture fields from HF config.json and map
them to the models.yaml / sweep_spec.yaml schema.

This is the reproducible provenance for the v2 SLM model entries: instead of
hand-transcribed (and previously wrong) numbers, run this to re-extract the exact
values HuggingFace publishes. Prints YAML blocks ready to diff against
configs/models.yaml plus the sweep_spec `models:` entries.

Handles both hybrid families:
  * Zamba2   (model_type=zamba2):  Mamba-2 + sparse shared-attention "hybrid"
                                   layers; chunk_size=256.
  * Falcon-H1 (model_type=falcon_h1): Mamba-2 + Attention in parallel in EVERY
                                   layer; mamba_chunk_size=128.

Usage:
    python fetch_hf_model_configs.py                 # the 4 v2 SLM/mid models
    python fetch_hf_model_configs.py --models zamba2_1.2b=Zyphra/Zamba2-1.2B
    python fetch_hf_model_configs.py --json          # raw mapped dict as JSON
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request

# our_key -> HF repo id. Sizes the v2 study sweeps over.
DEFAULT_MODELS = {
    "zamba2_1.2b":    "Zyphra/Zamba2-1.2B",
    "zamba2_2.7b":    "Zyphra/Zamba2-2.7B",
    "falcon_h1_1.5b": "tiiuae/Falcon-H1-1.5B-Base",
    "falcon_h1_3b":   "tiiuae/Falcon-H1-3B-Base",
    # 7B-scale (Path 1 regression). nemotron_h is a different model_type (not mapped here);
    # its entry is verified directly against the cached config.json.
    "zamba2_7b":      "Zyphra/Zamba2-7B-Instruct",
    "falcon_h1_7b":   "tiiuae/Falcon-H1-7B-Instruct",
}


def fetch_config(repo: str, revision: str = "main") -> dict:
    url = f"https://huggingface.co/{repo}/raw/{revision}/config.json"
    req = urllib.request.Request(url, headers={"User-Agent": "v2-config-fetch"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def map_zamba2(d: dict, repo: str) -> dict:
    n = int(d["num_hidden_layers"])
    lbt = d.get("layers_block_type", [])
    ids = d.get("hybrid_layer_ids")
    n_hybrid = len(ids) if ids else sum(1 for x in lbt if x == "hybrid")
    return {
        "full_name": repo.split("/")[-1],
        "hf_repo": repo,
        "config_status": "verified",
        "hidden_size": int(d["hidden_size"]),
        "num_layers": n,
        "ssm": {
            "n_heads": int(d["n_mamba_heads"]),
            "head_dim": int(d["mamba_headdim"]),
            "d_state": int(d["mamba_d_state"]),
            "chunk_size": int(d["chunk_size"]),
            "expand": int(d["mamba_expand"]),
            "n_groups": int(d["mamba_ngroups"]),
            "ssm_layer_fraction": round((n - n_hybrid) / n, 4),
        },
        "attention": {
            "num_heads": int(d["num_attention_heads"]),
            "num_kv_heads": int(d["num_key_value_heads"]),
            "head_dim": int(d["attention_head_dim"]),
            "attn_layer_fraction": round(n_hybrid / n, 4),
        },
        "mlp": {"intermediate_size": int(d.get("ffn_hidden_size", d["intermediate_size"]))},
    }


def map_falcon_h1(d: dict, repo: str) -> dict:
    n = int(d["num_hidden_layers"])
    # Falcon-H1: SSM + Attention run in parallel in every layer.
    return {
        "full_name": repo.split("/")[-1],
        "hf_repo": repo,
        "config_status": "verified",
        "hidden_size": int(d["hidden_size"]),
        "num_layers": n,
        "ssm": {
            "n_heads": int(d["mamba_n_heads"]),
            "head_dim": int(d["mamba_d_head"]),
            "d_state": int(d["mamba_d_state"]),
            "chunk_size": int(d["mamba_chunk_size"]),
            "expand": int(d["mamba_expand"]),
            "n_groups": int(d["mamba_n_groups"]),
            "d_ssm": int(d["mamba_d_ssm"]),
            "ssm_layer_fraction": 1.0,
        },
        "attention": {
            "num_heads": int(d["num_attention_heads"]),
            "num_kv_heads": int(d["num_key_value_heads"]),
            "head_dim": int(d["head_dim"]),
            "attn_layer_fraction": 1.0,
        },
        "mlp": {"intermediate_size": int(d["intermediate_size"])},
    }


def map_config(d: dict, repo: str) -> dict:
    mt = d.get("model_type")
    if mt == "zamba2":
        return map_zamba2(d, repo)
    if mt == "falcon_h1":
        return map_falcon_h1(d, repo)
    raise ValueError(f"unsupported model_type={mt!r} for {repo}")


def _emit_yaml(key: str, m: dict) -> str:
    ssm, attn, mlp = m["ssm"], m["attention"], m["mlp"]
    lines = [
        f"{key}:",
        f'  full_name: "{m["full_name"]}"',
        f'  hf_repo: "{m["hf_repo"]}"',
        f'  config_status: verified   # extracted from HF config.json',
        f"  hidden_size: {m['hidden_size']}",
        f"  num_layers: {m['num_layers']}",
        f"  ssm:",
        f"    n_heads: {ssm['n_heads']}",
        f"    head_dim: {ssm['head_dim']}",
        f"    d_state: {ssm['d_state']}",
        f"    chunk_size: {ssm['chunk_size']}",
        f"    expand: {ssm['expand']}",
        f"    n_groups: {ssm['n_groups']}",
    ]
    if "d_ssm" in ssm:
        lines.append(f"    d_ssm: {ssm['d_ssm']}")
    lines += [
        f"    ssm_layer_fraction: {ssm['ssm_layer_fraction']}",
        f"  attention:",
        f"    num_heads: {attn['num_heads']}",
        f"    num_kv_heads: {attn['num_kv_heads']}",
        f"    head_dim: {attn['head_dim']}",
        f"    attn_layer_fraction: {attn['attn_layer_fraction']}",
        f"  mlp:",
        f"    intermediate_size: {mlp['intermediate_size']}",
    ]
    return "\n".join(lines)


def parse_args():
    p = argparse.ArgumentParser(description="Extract HF model configs -> v2 schema")
    p.add_argument("--models", nargs="+", default=None,
                   help="key=repo pairs (default: the 4 v2 SLM/mid models)")
    p.add_argument("--json", action="store_true", help="emit mapped dicts as JSON")
    return p.parse_args()


def main():
    args = parse_args()
    if args.models:
        models = dict(kv.split("=", 1) for kv in args.models)
    else:
        models = DEFAULT_MODELS

    mapped = {}
    for key, repo in models.items():
        try:
            mapped[key] = map_config(fetch_config(repo), repo)
        except Exception as e:  # noqa: BLE001
            print(f"# ERROR {key} <- {repo}: {type(e).__name__}: {e}", file=sys.stderr)

    if args.json:
        print(json.dumps(mapped, indent=2))
        return

    print("# --- models.yaml entries (verified from HF config.json) ---\n")
    for key, m in mapped.items():
        print(_emit_yaml(key, m))
        print()
    print("# --- sweep_spec.yaml models: entries ---")
    for key, m in mapped.items():
        print(f"  {key}:")
        print(f"    hf: {m['hf_repo']}")
        print(f"    ssm_n_heads: {m['ssm']['n_heads']}")
        print(f"    ssd_chunk_size: {m['ssm']['chunk_size']}")


if __name__ == "__main__":
    main()
