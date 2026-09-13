#!/usr/bin/env python3
"""Pre-flight for the (Nano-9B-v2-Base, flashinfer) correctness gate: does the
O tier's protocol condition O1 still hold under the NEW tokenizer?

O1 (r2_correctness_check.py, `o_tier_confirmation`) requires, per probe:

    probe.prompt_tokens > (K - 1) * (bg.prompt_tokens + bg.max_new)

with `K = triton_attention_num_kv_splits` read out of the server args dump.
That server argument exists in `ServerArgs` regardless of the attention
backend and defaults to 8, so the condition is still EVALUATED under
flashinfer -- only its justification (the triton decode kernel's
`get_num_kv_splits_triton`) is triton-specific.  The arithmetic therefore has
to be satisfied by the new tokenizer, or every probe fails O1 and the verdict
is NO_VERDICT_UNREALIZED by construction.

The prompts are built by `r2_correctness_client.plan()` itself -- this script
imports the shipped client rather than restating its word budgets -- and
tokenized with the real checkpoint tokenizer out of the offline HF cache.
CPU only, no weights load, no GPU, no server.

Run:
  HF_HOME=<project>/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    python3 newpair_preflight_o1.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CLIENT = HERE.parent / "r2_correctness_client.py"

# Server-side default; see sglang/srt/server_args.py `triton_attention_num_kv_splits: int = 8`.
K_DEFAULT = 8
CLIENT_SEED = 20260911          # r2_correctness_client.py --seed default
N_SEQ, N_CONC, N_PROBES = 16, 32, 8
MAX_NEW_SEQ = 64
CTX = 16384                      # R2C_CTX for this pair

MODELS = [
    ("nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base", "NEW pair (flashinfer)"),
    ("Zyphra/Zamba2-2.7B", "reference: jobs 907100/907456 (triton)"),
]


def load_client():
    spec = importlib.util.spec_from_file_location("r2c_client", CLIENT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    client = load_client()
    from transformers import AutoTokenizer

    rc = 0
    for repo, note in MODELS:
        print("=" * 72)
        print(f"{repo}   [{note}]")
        try:
            tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
        except Exception as exc:                      # noqa: BLE001
            print(f"  SKIP: tokenizer unavailable offline ({type(exc).__name__})")
            continue

        def n(text: str) -> int:
            return len(tok(text, add_special_tokens=True)["input_ids"])

        seq, conc, over = client.plan(CLIENT_SEED, N_SEQ, N_CONC, MAX_NEW_SEQ, N_PROBES)
        bg = n(client.BG_PROMPT)
        bg_len_max = bg + client.BG_MAX_NEW
        threshold = (K_DEFAULT - 1) * bg_len_max
        print(f"  bg prompt tokens        : {bg}")
        print(f"  bg_len_max (+max_new)   : {bg_len_max}")
        print(f"  O1 threshold (K-1)*bg   : {threshold}   (K={K_DEFAULT})")

        worst_margin = None
        for item in over:
            pt = n(item["text"])
            ok = pt > threshold
            margin = pt - threshold
            worst_margin = margin if worst_margin is None else min(worst_margin, margin)
            flag = "ok " if ok else "FAIL"
            print(f"    {item['id']}  prompt_tokens={pt:5d}  margin={margin:+6d}  {flag}")
            if not ok:
                rc = 1
        print(f"  worst O1 margin         : {worst_margin:+d} tokens "
              f"({100.0 * worst_margin / threshold:+.1f}% of the threshold)")
        # How many extra bg tokens would break the tightest probe?
        tightest = min(n(i["text"]) for i in over)
        max_bg = (tightest - 1) // (K_DEFAULT - 1) - client.BG_MAX_NEW
        print(f"  bg prompt could grow to : {max_bg} tokens before O1 fails "
              f"(measured {bg})")

        longest = max(
            max(n(i["text"]) + i["max_new"] for i in seq),
            max(n(i["text"]) + i["max_new"] for i in conc),
            max(n(i["text"]) + client.PROBE_MAX_NEW for i in over),
        )
        print(f"  longest request (in+out): {longest} tokens   "
              f"vs R2C_CTX={CTX} -> {'fits' if longest <= CTX else 'DOES NOT FIT'}")
        if longest > CTX:
            rc = 1

        # ★POSITIVE CONTROL for this script's own method.  For the reference
        # pair the SERVER already reported `prompt_tokens` for exactly these
        # prompts (job 907456).  If the counts computed here reproduce those, the
        # same computation for the new tokenizer is a prediction of what the
        # server will report, not an independent guess about it.
        ref = HERE.parent / "job_907456" / "gen_L1.json"
        if repo == "Zyphra/Zamba2-2.7B" and ref.exists():
            import json

            doc = json.loads(ref.read_text())
            served = {e["id"]: ((e.get("probe") or {}).get("prompt_tokens"),
                                (e.get("bg") or {}).get("prompt_tokens"))
                      for e in doc.get("phase_o") or []}
            agree = sum(
                1 for item in over
                if served.get(item["id"], (None, None)) == (n(item["text"]), bg)
            )
            print(f"  cross-check vs job 907456 server-reported prompt_tokens: "
                  f"{agree}/{len(over)} probes agree (bg={bg} on all)")
            if agree != len(over):
                rc = 1
    print("=" * 72)
    print("PREFLIGHT_RC=%d" % rc)
    return rc


if __name__ == "__main__":
    sys.exit(main())
