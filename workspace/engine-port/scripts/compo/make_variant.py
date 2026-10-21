#!/usr/bin/env python3
"""make_variant.py -- same-skeleton composition variants of Nemotron-Nano-9B-v2-Base (decision 24).

WHAT.  Copies the NON-WEIGHT files of the original snapshot (config, tokenizer, generation config, remote
code ...) into a new directory and rewrites ONLY `hybrid_override_pattern` in config.json, so that a
number of Mamba layers (M) become attention layers (*) -- or, for the pure-SSM extreme n_attn = 0, the
4 attention layers become Mamba.  Hidden size, MLP, layer count (56) and the 25 MLP layers (-) are fixed.
Weight files are NOT copied: the variant is meant for `--load-format dummy` (random weights).
Decision: workspace/engine-port/results/anchor_audit/DECISION_headline_G_shapeC_2026-10-19.md, decision 24.

SWAP RULE (deterministic, name `spread_floor_mid_v1`):
  Let the original pattern have M Mamba layers (27 for the pinned revision), listed in depth order;
  their ORDINALS are 0..M-1.  To reach n_attn attention layers from the original 4, n_swap = n_attn - 4
  Mamba layers are converted.  The converted ordinals are
        o_j = floor((j + 0.5) * M / n_swap),   j = 0 .. n_swap - 1
  i.e. the midpoints of n_swap equal-width bins over the Mamba ordinals, so the converted layers are
  spread evenly in depth.  Because M / n_swap >= 1 the o_j are strictly increasing (no collisions), and
  n_swap = M converts all of them.  Exact integer arithmetic is used: o_j = ((2j + 1) * M) // (2 n_swap).
  n_attn = 4 (n_swap = 0) reproduces the original pattern byte-for-byte (the "dummy original" variant).
  n_attn = 0 converts every '*' to 'M' (only one way to do it).
  Only the CLI values {0, 4, 13, 22, 31} are offered (decision 24 grid + the pure-SSM extreme).

PREDICTIONS [유도] written to variant.json (derived from config.json; NOT measurements):
  per-request state  s = n_mamba * (H_m * P * N * 4 B  +  conv_dim * (K - 1) * 2 B)
        H_m = mamba_num_heads, P = mamba_head_dim, N = ssm_state_size, K = conv_kernel,
        conv_dim = H_m * P + 2 * n_groups * N.
        Sources (SGLang v0.5.10 tree):  srt/configs/mamba_utils.py  Mamba2StateShape.create
          (conv_dim = intermediate_size + 2 * n_groups * state_size, conv shape (conv_dim, K - 1),
          temporal shape (num_heads, head_dim, state_size)) and BaseLinearStateParams.mamba_cache_per_req;
          dtypes from mamba2_state_dtype(): conv = bfloat16 unless SGLANG_MAMBA_CONV_DTYPE, ssm (temporal)
          = float32 unless config.mamba_ssm_dtype or SGLANG_MAMBA_SSM_DTYPE (the pinned config has no
          mamba_ssm_dtype; the probe unsets both env vars).
  per-token KV       k = n_attn * num_key_value_heads * head_dim * 2 (K and V) * 2 B (bf16 KV cache)
  L* = s / k   (tokens at which one request's KV equals its Mamba state; None if k == 0)
  parameters, by layer type (SGLang srt/models/nemotron_h.py + srt/layers/attention/mamba/mamba.py;
  every linear is bias-free except the Mamba conv1d, matching the pinned config):
    mamba layer     = hidden*(2*H_m*P + 2*G*N + H_m)        in_proj  [z, x, B, C, dt]
                    + conv_dim*K + conv_dim                   conv1d weight + bias (use_conv_bias)
                    + 3*H_m                                   A (float32), D, dt_bias
                    + H_m*P                                   gated RMSNorm weight
                    + H_m*P*hidden                            out_proj
                    + hidden                                  block RMSNorm
    attention layer = hidden*(n_q*hd + 2*n_kv*hd) + n_q*hd*hidden + hidden      qkv, o_proj, norm
    mlp layer       = 2*hidden*intermediate + hidden                             up, down (relu2), norm
    rest            = 2*vocab*hidden (embedding + untied lm_head) + hidden (final norm)
  weight bytes = 2 B per parameter, except A which SGLang allocates as float32 (4 B).
  Original check: s = 143,548,416 B (boot log: 143.5 MB), k = 16,384 B (boot log: 16,382 B, GiB rounding),
  weights 8,888,227,328 params = 16.555 GiB (boot log "Load weight end ... mem usage=16.58 GB").

USAGE
  python3 make_variant.py --src <snapshot dir> --out <dir> --n-attn {0,4,13,22,31} [--force]
  python3 make_variant.py --src <snapshot dir> --print     # predictions for all grid points, no files
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys

SCHEMA = "compo_variant/v1"
RULE_NAME = "spread_floor_mid_v1"
RULE_FORMULA = "o_j = floor((j + 0.5) * M / n_swap), j = 0..n_swap-1 over Mamba ordinals in depth order"
NANO_ID = "nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base"
NANO_REV = "dc0661c829b14e5b9246c05cfa89094a0875e052"
ORIG_PATTERN = "M-M-M-MM-M-M-M*-M-M-M*-M-M-M-M*-M-M-M-M*-M-MM-M-M-M-M-M-"
N_ATTN_CHOICES = (0, 4, 13, 22, 31)

# files never copied: weights and the weight index (dummy load needs none of them)
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".gguf", ".h5", ".msgpack", ".ckpt")
WEIGHT_INDEX_SUFFIXES = (".safetensors.index.json", ".bin.index.json")

SSM_STATE_ITEMSIZE = 4    # float32 temporal state (mamba_utils.mamba2_state_dtype default)
CONV_STATE_ITEMSIZE = 2   # bfloat16 conv state (SGLANG_MAMBA_CONV_DTYPE default)
KV_ITEMSIZE = 2           # bfloat16 KV cache (--dtype bfloat16, no --kv-cache-dtype)
PARAM_ITEMSIZE = 2        # bfloat16 weights
A_ITEMSIZE = 4            # MambaMixer2.A is created with dtype=torch.float32


# ----------------------------------------------------------------------------- pattern rule (pure)
def mamba_positions(pattern):
    return [i for i, c in enumerate(pattern) if c == "M"]


def attn_positions(pattern):
    return [i for i, c in enumerate(pattern) if c == "*"]


def swap_ordinals(m, n_swap):
    """Ordinals (0..m-1) of the Mamba layers converted to attention, rule spread_floor_mid_v1."""
    if not 0 <= n_swap <= m:
        raise ValueError(f"n_swap={n_swap} outside 0..{m}")
    if n_swap == 0:
        return []
    out = [((2 * j + 1) * m) // (2 * n_swap) for j in range(n_swap)]
    assert len(set(out)) == n_swap and out == sorted(out)
    return out


def make_pattern(orig, n_attn):
    """-> (new_pattern, changed_layer_indices, changed_mamba_ordinals, direction)."""
    if set(orig) - set("M*-"):
        raise ValueError("pattern must contain only M, *, -")
    m_pos, a_pos = mamba_positions(orig), attn_positions(orig)
    a0 = len(a_pos)
    if n_attn == 0:
        new = orig.replace("*", "M")
        return new, list(a_pos), [], "*->M"
    if n_attn < a0:
        raise ValueError(f"n_attn={n_attn} between 1 and {a0 - 1} is not defined by this rule")
    ords = swap_ordinals(len(m_pos), n_attn - a0)
    layers = [m_pos[o] for o in ords]
    chars = list(orig)
    for i in layers:
        chars[i] = "*"
    return "".join(chars), layers, ords, ("M->*" if layers else "none")


# ----------------------------------------------------------------------------- predictions (pure)
def _cfg(cfg, *keys, default=None):
    for k in keys:
        if k in cfg and cfg[k] is not None:
            return cfg[k]
    if default is None:
        raise KeyError(f"config lacks all of {keys}")
    return default


def predict(cfg, pattern):
    """Derived ([유도]) memory/parameter figures for config dict `cfg` with layer pattern `pattern`."""
    hidden = int(cfg["hidden_size"])
    inter = int(cfg["intermediate_size"])
    vocab = int(cfg["vocab_size"])
    n_q = int(cfg["num_attention_heads"])
    n_kv = int(_cfg(cfg, "num_key_value_heads", default=n_q))
    hd = int(_cfg(cfg, "head_dim", default=hidden // n_q))
    hm = int(cfg["mamba_num_heads"])
    p = int(cfg["mamba_head_dim"])
    n = int(_cfg(cfg, "ssm_state_size", "mamba_state_dim"))
    g = int(_cfg(cfg, "n_groups", "mamba_n_groups", "mamba_num_groups"))
    kconv = int(_cfg(cfg, "conv_kernel", "mamba_d_conv", default=4))
    tied = bool(cfg.get("tie_word_embeddings", False))
    if cfg.get("mamba_ssm_dtype") not in (None, "float32"):
        raise ValueError("config sets mamba_ssm_dtype != float32; the fp32 state assumption does not hold")
    if any(cfg.get(k) for k in ("attention_bias", "mlp_bias", "use_bias", "mamba_proj_bias")):
        raise ValueError("bias-free linear assumption does not hold for this config")
    conv_bias = bool(_cfg(cfg, "use_conv_bias", "mamba_conv_bias", default=True))

    n_m, n_a, n_mlp = pattern.count("M"), pattern.count("*"), pattern.count("-")
    if n_m + n_a + n_mlp != len(pattern):
        raise ValueError("pattern has layer types other than M/*/- (MoE not modelled)")
    inner = hm * p
    conv_dim = inner + 2 * g * n
    conv_state = conv_dim * (kconv - 1) * CONV_STATE_ITEMSIZE
    ssm_state = hm * p * n * SSM_STATE_ITEMSIZE
    s = n_m * (ssm_state + conv_state)
    k = n_a * n_kv * hd * 2 * KV_ITEMSIZE

    p_mamba = (hidden * (2 * inner + 2 * g * n + hm) + conv_dim * kconv + (conv_dim if conv_bias else 0)
               + 3 * hm + inner + inner * hidden + hidden)
    p_attn = hidden * (n_q * hd + 2 * n_kv * hd) + n_q * hd * hidden + hidden
    p_mlp = 2 * hidden * inter + hidden
    p_rest = (1 if tied else 2) * vocab * hidden + hidden
    total = n_m * p_mamba + n_a * p_attn + n_mlp * p_mlp + p_rest
    wbytes = total * PARAM_ITEMSIZE + n_m * hm * (A_ITEMSIZE - PARAM_ITEMSIZE)
    return {
        "derived": True,
        "n_layers": len(pattern), "n_mamba": n_m, "n_attn": n_a, "n_mlp": n_mlp,
        "conv_dim": conv_dim,
        "conv_state_bytes_per_layer": conv_state, "ssm_state_bytes_per_layer": ssm_state,
        "state_bytes_per_req_s": s,
        "kv_bytes_per_token_k": k,
        "L_star_tokens": (s / k) if k else None,
        "params_per_layer": {"mamba": p_mamba, "attention": p_attn, "mlp": p_mlp, "embed_lmhead_finalnorm": p_rest},
        "params_by_type": {"mamba": n_m * p_mamba, "attention": n_a * p_attn, "mlp": n_mlp * p_mlp,
                           "embed_lmhead_finalnorm": p_rest},
        "params_total": total,
        "weight_bytes": wbytes,
        "weight_gib": wbytes / 1024 ** 3,
        "assumptions": {"ssm_state_dtype": "float32", "conv_state_dtype": "bfloat16", "kv_dtype": "bfloat16",
                        "weight_dtype": "bfloat16 (A float32)", "tp": 1,
                        "formula_s": "n_mamba*(H_m*P*N*4 + conv_dim*(K-1)*2), conv_dim = H_m*P + 2*G*N",
                        "formula_k": "n_attn*n_kv*head_dim*2*2"},
    }


# ----------------------------------------------------------------------------- files
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def is_weight_file(name):
    low = name.lower()
    return low.endswith(WEIGHT_INDEX_SUFFIXES) or low.endswith(WEIGHT_SUFFIXES)


def build_variant(src, out, n_attn, force=False, source_meta=None):
    """Create the variant directory; returns the variant.json dict."""
    if n_attn not in N_ATTN_CHOICES:
        raise ValueError(f"n_attn must be one of {N_ATTN_CHOICES}")
    cfg_path = os.path.join(src, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    orig = cfg["hybrid_override_pattern"]
    if orig != ORIG_PATTERN:
        raise ValueError(f"source pattern differs from the pinned revision's pattern: {orig}")
    new, layers, ords, direction = make_pattern(orig, n_attn)
    if os.path.exists(out):
        if not force:
            raise FileExistsError(f"{out} exists (use --force)")
        shutil.rmtree(out)
    os.makedirs(out)
    copied, excluded = [], []
    for name in sorted(os.listdir(src)):
        sp = os.path.join(src, name)
        if not os.path.isfile(sp):          # follows HF-cache symlinks; subdirectories are not expected
            excluded.append({"name": name, "why": "not a regular file"})
            continue
        if is_weight_file(name):
            excluded.append({"name": name, "why": "weight file (dummy load)"})
            continue
        if name == "config.json":
            continue
        shutil.copyfile(sp, os.path.join(out, name))     # copies content, not the symlink
        copied.append({"name": name, "sha256": sha256_file(sp)})
    new_cfg = dict(cfg)
    new_cfg["hybrid_override_pattern"] = new
    out_cfg = os.path.join(out, "config.json")
    with open(out_cfg, "w") as f:
        json.dump(new_cfg, f, indent=2)          # allow_nan: time_step_limit carries Infinity, as upstream
        f.write("\n")
    rec = {
        "schema": SCHEMA,
        "rule": {"name": RULE_NAME, "formula": RULE_FORMULA,
                 "n_attn_0": "every '*' becomes 'M' (pure-SSM extreme)",
                 "n_attn_4": "pattern unchanged (dummy original)"},
        "source": dict({"model_id": NANO_ID, "revision_expected": NANO_REV, "snapshot_dir": os.path.abspath(src),
                        "config_sha256": sha256_file(cfg_path)}, **(source_meta or {})),
        "n_attn": pattern_counts(new)["*"], "n_mamba": pattern_counts(new)["M"], "n_mlp": pattern_counts(new)["-"],
        "pattern_orig": orig, "pattern": new,
        "direction": direction,
        "changed_layers": layers,
        "changed_mamba_ordinals": ords,
        "config_sha256": sha256_file(out_cfg),
        "copied_files": copied,
        "excluded_files": excluded,
        "weights_copied": False,
        "load_format_required": "dummy",
        "prediction": predict(new_cfg, new),
        "note": "prediction = [유도] from config.json, not a measurement; probe values are not citable",
    }
    assert rec["n_attn"] == n_attn
    with open(os.path.join(out, "variant.json"), "w") as f:
        json.dump(rec, f, indent=2, sort_keys=True)
        f.write("\n")
    return rec


def pattern_counts(pattern):
    return {c: pattern.count(c) for c in "M*-"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", required=True, help="original snapshot directory (config.json + tokenizer files)")
    ap.add_argument("--out", help="variant directory to create")
    ap.add_argument("--n-attn", type=int, choices=N_ATTN_CHOICES)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--revision", default=None, help="observed snapshot revision (recorded in variant.json)")
    ap.add_argument("--print", action="store_true", help="print predictions for every grid point; no files")
    a = ap.parse_args(argv)
    if a.print:
        with open(os.path.join(a.src, "config.json")) as f:
            cfg = json.load(f)
        for n_attn in N_ATTN_CHOICES:
            pat, layers, _, _ = make_pattern(cfg["hybrid_override_pattern"], n_attn)
            pr = predict(cfg, pat)
            print(json.dumps({"n_attn": n_attn, "pattern": pat, "changed_layers": layers,
                              "s_bytes": pr["state_bytes_per_req_s"], "k_bytes": pr["kv_bytes_per_token_k"],
                              "L_star": pr["L_star_tokens"], "params": pr["params_total"],
                              "weight_gib": round(pr["weight_gib"], 4)}))
        return 0
    if a.out is None or a.n_attn is None:
        ap.error("--out and --n-attn are required unless --print")
    meta = {"revision_observed": a.revision} if a.revision else None
    rec = build_variant(a.src, a.out, a.n_attn, force=a.force, source_meta=meta)
    print(json.dumps({"out": a.out, "n_attn": rec["n_attn"], "n_mamba": rec["n_mamba"], "pattern": rec["pattern"],
                      "changed_layers": rec["changed_layers"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
