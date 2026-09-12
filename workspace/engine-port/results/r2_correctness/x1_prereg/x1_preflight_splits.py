#!/usr/bin/env python3
"""X1 pre-flight: does the planned perturbation actually change the decode
attention reduction for every COMPARED unit?

rev2 (2026-09-12, after claims-auditor rules-layer audit D1/D10).
rev1 judged "perturbed" by the per-row SPLIT COUNT and reported 24/24 perturbed
at a forced value of 6.  That was WRONG: the kernel quantizes one step further,

    decode_attention.py:35   _MIN_BLOCK_KV = 32
    decode_attention.py:98   kv_len_per_split = cdiv(cdiv(seq_len, kv_splits), 32)*32
    decode_attention.py:553  stage-2 combine recomputes the same expression

and both stages skip empty splits (`if split_kv_end > split_kv_start`).  So two
configurations with DIFFERENT split counts but the SAME kv_len_per_split visit
identical boundaries in identical order and are bit-identical.  Judging at the
split-count level therefore overstates the perturbation -- the very identity
trap this file exists to detect (de-confound lesson 9).

rev2 judges at the effective granularity: a row is perturbed iff
kv_len_per_split differs at some decode step.

Heuristic reimplemented from (dev tree, SGLang v0.5.10):
  triton_backend.py:1271-1319  get_num_kv_splits_triton  (dynamic path)
  triton_backend.py:185-234    get_num_kv_splits         (static path = fill_(max))
  triton_backend.py:717        replay calls it every cudagraph replay
Model: Zamba2-2.7B num_attention_heads=32, num_key_value_heads=32 -> kv_group=1.
Device: A100 108 SMs (get_device_core_count = multi_processor_count; ASSUMPTION).
Lengths: job 907100's recorded `prompt_tokens`.  Those come back in the server's
meta_info, but they are a deterministic function of the input text (prompt_sha256
identical 56/56) and were identical across all four boots -- not an outcome.
"""
import json
import math
import sys

NUM_HEAD = 32
NUM_KV_HEAD = 32
CORE_COUNT = 108
MIN_BLOCK_KV = 32


def cdiv(a, b):
    return -(-a // b)


def heuristic_splits(seq_lens, max_kv, static=False):
    """Per-row split counts for one decode batch (triton_backend.py:185-234)."""
    if static:
        return [max_kv] * len(seq_lens)
    max_seq, min_seq = max(seq_lens), min(seq_lens)
    if max_seq * 8 < min_seq * 10:
        min_seq = max_seq
    mks1 = min(cdiv(max_seq, min_seq), max_kv)
    chunk1 = cdiv(max_seq, mks1)
    ext_core = int(CORE_COUNT * max(math.log2(max_seq / 64.0), 1.0))
    token_grid = len(seq_lens) * NUM_HEAD          # kv_group == 1 branch
    mks2 = min(cdiv(ext_core, token_grid), max_kv)
    chunk2 = cdiv(max_seq, mks2)
    return [max(cdiv(s, chunk1), cdiv(s, chunk2)) for s in seq_lens]


def eff(seq_len, splits):
    """What the kernel actually reduces over: (kv_len_per_split, n_nonempty)."""
    klps = cdiv(cdiv(seq_len, splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    return klps, cdiv(seq_len, klps)


def batches(prompt_tokens, max_new, bg_lens):
    """Decode batches this request's row appears in, row 0 = the request."""
    for step in range(max_new):
        L = prompt_tokens + step
        yield [L]
        for bg in bg_lens:
            yield [L, bg]


def row_profile(prompt_tokens, max_new, bg_lens, max_kv, static=False):
    """Effective reduction signature of the compared row across its decode."""
    sig = set()
    for batch in batches(prompt_tokens, max_new, bg_lens):
        s = heuristic_splits(batch, max_kv, static)[0]
        sig.add((batch[0], len(batch)) + eff(batch[0], s))
    return sig


def units_of(gen, bg_lens_for_o):
    for r in gen["phase_s"]:
        yield r["id"], r["prompt_tokens"], r["max_new"], []
    for rec in gen["phase_o"]:
        p = rec["probe"]
        yield rec["id"], p["prompt_tokens"], p["max_new"], bg_lens_for_o


def main():
    job = sys.argv[1] if len(sys.argv) > 1 else "."
    base_cap = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    gen = json.load(open(f"{job}/gen_L1.json"))
    # Background row length range in the O tier, from the recorded run: the bg
    # prompt is 17 tokens (not the 15 rev1 guessed) and decodes bg_max_new more.
    bg_ptok = gen["phase_o"][0]["bg"].get("prompt_tokens", 17)
    bg_lens = [bg_ptok, bg_ptok + gen["bg_max_new"] // 2, bg_ptok + gen["bg_max_new"]]
    units = list(units_of(gen, bg_lens))

    print(f"baseline: cap={base_cap} + kernel heuristic (job 907100's operating config)")
    print(f"criterion: kv_len_per_split = cdiv(cdiv(L,s),32)*32 differs at any step")
    print(f"O tier background row lengths used: {bg_lens}\n")
    print("candidate settings (CLI `--triton-attention-num-kv-splits C`, no env var):")
    print(f"{'C':>3} {'perturbed units':>16} {'unperturbed (bit-identical by construction)'}")
    table = {}
    for cap in range(2, 9):
        moved, still = [], []
        for uid, ptok, mnew, bgl in units:
            a = row_profile(ptok, mnew, bgl, base_cap)
            b = row_profile(ptok, mnew, bgl, cap)
            (moved if a != b else still).append(uid)
        table[cap] = (moved, still)
        print(f"{cap:>3} {len(moved):>10}/{len(units):<5} {len(still)} {still}")
    print()
    # Also the rev1 setting, for the record: static env var + cap 6.
    moved, still = [], []
    for uid, ptok, mnew, bgl in units:
        a = row_profile(ptok, mnew, bgl, base_cap)
        b = row_profile(ptok, mnew, bgl, 6, static=True)
        (moved if a != b else still).append(uid)
    print(f"rev1 plan (STATIC env var + cap 6): perturbed {len(moved)}/{len(units)}, "
          f"unperturbed {len(still)} {still}")
    moved8, still8 = [], []
    for uid, ptok, mnew, bgl in units:
        a = row_profile(ptok, mnew, bgl, base_cap)
        b = row_profile(ptok, mnew, bgl, base_cap, static=True)
        (moved8 if a != b else still8).append(uid)
    print(f"VERDICT §7.3 draft (STATIC env var only -> cap 8): perturbed "
          f"{len(moved8)}/{len(units)}, unperturbed {len(still8)} {still8}")


if __name__ == "__main__":
    main()
