#!/usr/bin/env python3
"""AF-1: pick the prompts one boot will issue.  CPU only, GPU 0.

The pre-registration (sec 3) fixes the strata (`REPORT_STRATUM_QS`), the repeat
count (`REPEATS_PER_STRATUM`) and the per-boot seed (`BOOT_SEEDS`), and says the
seed decides ONE thing: "층 q에서 어느 행을 뽑는가".

★REGISTRATION ADDENDUM (2026-09-07, written with the harness, flagged for the
next audit).  The pre-registration does not say whether the three repeats at a
stratum are the SAME row issued three times or three different rows -- the 4th
audit named this as still-unregistered (E6) and no revision closed it.  This file
closes it, and states the choice and its reason here rather than letting the
harness decide silently:

    the seed draws ONE row per stratum, and that row is issued REPEATS times.

Reason: `agg_within_boot` is an order statistic over the repeats (sec 2.3), and
its registered meaning is "the slowest of three uncontended repeats at this
stratum".  With three DIFFERENT rows that statistic would mix prompt-length
variation into what is meant to be a timing statistic; with one row repeated it
is timing only.  ⚠️Consequence, stated because it is not free: the first repeat
after a boot pays warm-up costs the others do not, and a near-max order statistic
lets that first repeat dominate.  That direction RAISES the floor, which makes
admitting a coordinate harder -- the conservative direction of sec 4.2 -- so no
discard-first rule is added here.  Adding one would change the estimand in the
liberal direction and would have to be registered, not slipped in.

Selection is deterministic given (dataset, tokenizer, seed): rows are ranked by
|len - target| at each stratum and the seed picks one from the nearest pool.

Usage: python3 af1_prompts.py --seed 11 --out prompts_seed11.json
"""
import argparse, json, os, random, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import af1_predicates as P

HF = os.environ.get("HF_HOME", "/scratch/ehmoon/whlee/prefill-layer-alloc/hf_cache")
DATASET = Path(HF) / "raw" / "ShareGPT_long2048_cap8192.json"
MODEL = "nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base"

# How many nearest rows the seed draws from at each stratum.  A pool of one would
# make the seed inert (sec 3 registers that the seed decides which row is drawn);
# a large pool would let the drawn row sit far from the stratum it is named for.
NEAREST_POOL = 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", default=str(DATASET))
    a = ap.parse_args()
    if a.seed not in P.BOOT_SEEDS:
        sys.exit("seed %d is not registered in BOOT_SEEDS %s" % (a.seed, P.BOOT_SEEDS))

    t0 = time.time()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    rows = json.load(open(a.dataset))
    prompts = [c["conversations"][0]["value"] for c in rows
               if len(c.get("conversations", [])) >= 2]
    lens = [len(tok(p, add_special_tokens=False)["input_ids"]) for p in prompts]
    print(f"tokenised {len(prompts)} rows in {time.time()-t0:.0f}s", flush=True)

    rng = random.Random(a.seed)
    out = {"_what_this_is": "AF-1 per-boot prompt set. One row per stratum, issued "
                            "REPEATS_PER_STRATUM times (see this file's docstring).",
           "seed": a.seed, "dataset": a.dataset, "tokenizer": MODEL,
           "repeats": P.REPEATS_PER_STRATUM, "nearest_pool": NEAREST_POOL,
           "strata": []}
    for q in P.REPORT_STRATUM_QS:
        target = P.STRATUM_TOKENS[q]
        pool = sorted(range(len(prompts)), key=lambda i: abs(lens[i] - target))[:NEAREST_POOL]
        idx = rng.choice(pool)
        out["strata"].append({
            "q": q, "target_tokens": target, "row_index": idx,
            "prompt_tokens": lens[idx],
            "abs_error_tokens": abs(lens[idx] - target),
            "prompt": prompts[idx],
        })
        print(f"  q={q:<5} target={target:8.1f}  drawn={lens[idx]:5d} tok "
              f"(|err|={abs(lens[idx]-target):6.1f})", flush=True)
    json.dump(out, open(a.out, "w"), ensure_ascii=False)
    print("wrote", a.out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
