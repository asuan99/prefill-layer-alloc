"""SUPERSEDED (2026-08-28) -- DO NOT CITE.  History only.

This script samples a population the benchmark does not serve: it takes the first
N conversations in file order and filters on prompt length alone.  The bench seeds
(`bench_serving.py:1705`, --seed default 1), shuffles (`datasets/sharegpt.py:98`),
drops on prompt_len + output_len > context_len, and keeps the first 200.

All four of its headline values are optimistic:
    cps    this script    actually served (served_population.json)
    512       27.88%          51/200 = 25.50%
    1024       7.55%          10/200 =  5.00%
    2048       2.05%           4/200 =  2.00%
    4096       0.48%           0/200 =  0.00%   <- the arm is EMPTY, not rare

Live replacement: `measure_served_population.py`, which calls the bench's own
sampler so it cannot drift from what the harness serves.
"""

#!/usr/bin/env python3
"""ShareGPT prompt-length distribution under the canonical varying-trace config.

Why this exists: the CP (chunked-prefill) arm set must be chosen from the MEASURED
token-length distribution, not from a guessed one.  If `--chunked-prefill-size N`
exceeds essentially every prompt, that arm cannot split any single request and the
per-request chunking axis is degenerate before the campaign starts
(= the NOTHING_PURCHASABLE / SINGLE_LABEL_FORCED failure class named in
`workspace/engine-port/results/REACHABILITY_FINDING_2026-08-28.md`).

Scope: this measures the PER-REQUEST axis only.  `chunked_prefill_size` also caps
the token budget of a whole prefill BATCH, which bites even when no single prompt
exceeds it; that second channel is only observable at runtime (Stage CP-0).

Usage:  python3 measure_sharegpt_prompt_lens.py [N_SAMPLE] > out.json
"""
import json, os, statistics, sys

os.environ.setdefault("HF_HOME", "/scratch/ehmoon/whlee/prefill-layer-alloc/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

SGPT = "/scratch/ehmoon/whlee/prefill-layer-alloc/hf_cache/raw/ShareGPT_V3_unfiltered_cleaned_split.json"
MODEL = "Zyphra/Zamba2-2.7B"


def main(n_sample=4000):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    data = json.load(open(SGPT))
    prompts = []
    for d in data:
        c = d.get("conversations") or []
        if len(c) < 2:
            continue
        prompts.append(c[0]["value"])
        if len(prompts) >= n_sample:
            break
    lens = sorted(l for l in (len(tok(p).input_ids) for p in prompts) if l >= 4)

    def q(p):
        return lens[min(len(lens) - 1, int(p * len(lens)))]

    out = {
        "model": MODEL,
        "dataset": SGPT,
        "n_sample_requested": n_sample,
        "n_after_min_len_filter": len(lens),
        "quantiles": {f"p{int(p*100)}": q(p) for p in (0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)},
        "max": lens[-1],
        "mean": round(statistics.mean(lens), 1),
        "frac_prompt_exceeds": {
            str(c): round(sum(1 for l in lens if l > c) / len(lens), 4)
            for c in (512, 1024, 2048, 4096)
        },
        "note": "frac_prompt_exceeds = fraction of requests that a given "
                "--chunked-prefill-size splits ON ITS OWN (per-request axis only).",
    }
    json.dump(out, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4000)
