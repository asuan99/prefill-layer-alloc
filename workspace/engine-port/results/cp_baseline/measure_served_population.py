#!/usr/bin/env python3
"""What the varying-trace bench ACTUALLY serves -- arm-set evidence for the CP campaign.

rev2.  Supersedes `measure_sharegpt_prompt_lens.py`, which sampled a different
population and produced four values that were all optimistic:

    cps    this script (served)   superseded script    delta
    512          51/200 = 25.50%        27.88%         -2.4 pp
    1024         10/200 =  5.00%         7.55%         -2.6 pp
    2048          4/200 =  2.00%         2.05%         -0.1 pp
    4096          0/200 =  0.00%         0.48%         ARM IS EMPTY

The superseded script took the first 4000 conversations in file order and filtered
on prompt length alone.  The bench does something else, and the difference decides
the arm set:

    bench_serving.py:1705      random.seed(args.seed)      (--seed default 1)
    datasets/sharegpt.py:98    random.shuffle(dataset)
    datasets/sharegpt.py:136   drop if prompt_len + output_len > context_len
                               (--sharegpt-context-len 4000 in sharegpt_vary_bench.sbatch:68)
    datasets/sharegpt.py:104   stop at num_requests (--num-prompts 200, NP in the sbatch)

★CORRECTION (2026-08-28, 3rd audit F2).  An earlier version of this file claimed
that `--chunked-prefill-size 4096` "cannot split a single request on this workload --
not 'rarely', but never".  **That was false, and it reached a registered
reachability spec.**  Chunking in this engine is a BATCH-BUDGET mechanism, not a
request-length one:

    schedule_policy.py:802-842
        elif self.rem_chunk_tokens is None or input_tokens <= self.rem_chunk_tokens:
            ...non-chunked...
        else:
            trunc_len = self.rem_chunk_tokens // self.page_size * self.page_size
            req.set_extend_input_len(trunc_len)          # <- CHUNKED

`rem_chunk_tokens` is what is LEFT of the batch's token budget after the requests
already packed into it, so a request shorter than `chunked_prefill_size` is chunked
whenever it arrives at a nearly-exhausted budget.  Prompt length is therefore an
UNDERCOUNT of chunking, not a measure of it, and no cps value makes chunking
structurally impossible.

Two consequences the campaign design must carry:

1. `n_prompts_longer_than_cps` below is exactly what its name says -- the number of
   prompts that exceed the cap ON THEIR OWN.  It is a LOWER BOUND on chunking, and
   the gap to the truth is the batch channel, which is observable only at runtime.
2. The seed is fixed, so the served set is a DETERMINISTIC 200-request constant, not
   a sample from a distribution.  A positive-control threshold phrased as a percentage
   (">= 1% of requests split") is therefore a count in disguise: 1% is 2 requests.
   The CP rules register counts.

This calls the bench's own sampler rather than reimplementing it, so it cannot drift
away from what the harness serves.

Usage:  python3 measure_served_population.py [NUM_PROMPTS] [CONTEXT_LEN] [SEED] > out.json
"""
import contextlib, io as _io, json, os, random, statistics, sys

os.environ.setdefault("HF_HOME", "/scratch/ehmoon/whlee/prefill-layer-alloc/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

SGPT = "/scratch/ehmoon/whlee/prefill-layer-alloc/hf_cache/raw/ShareGPT_V3_unfiltered_cleaned_split.json"
MODEL = "Zyphra/Zamba2-2.7B"
CPS_GRID = (512, 1024, 2048, 4096)


def main(num_prompts=200, context_len=4000, seed=1):
    from transformers import AutoTokenizer
    from sglang.benchmark.datasets.sharegpt import sample_sharegpt_requests

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    random.seed(seed)          # bench_serving.py:1705
    # The sampler prints "#Input tokens:" / "#Output tokens:" to stdout.  Capture it:
    # this script's stdout IS the artifact, and two stray lines make it unparseable
    # JSON -- which is exactly how it was first written, and it was caught only when
    # a later consistency check tried to json.load() the file.
    _noise = _io.StringIO()
    with contextlib.redirect_stdout(_noise):
        rows = sample_sharegpt_requests(
            dataset_path=SGPT,
            num_requests=num_prompts,
            tokenizer=tok,
            fixed_output_len=None,
            context_len=context_len,
        )
    plens = sorted(r.prompt_len for r in rows)

    def q(p):
        return plens[min(len(plens) - 1, int(p * len(plens)))]

    out = {
        "provenance": {
            "sampler": "sglang.benchmark.datasets.sharegpt.sample_sharegpt_requests",
            "seed": seed,
            "num_prompts": num_prompts,
            "context_len": context_len,
            "matches_harness": "sharegpt_vary_bench.sbatch:65-71 (NP=200, "
                               "--sharegpt-context-len 4000, --seed default 1)",
            "model": MODEL,
        },
        "n_served": len(rows),
        "prompt_len": {
            **{f"p{int(p*100)}": q(p) for p in (0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)},
            "max": plens[-1],
            "mean": round(statistics.mean(plens), 1),
        },
        # COUNTS, not fractions: the served set is a fixed 200-request constant.
        # ★NAME MEANS WHAT IT COMPUTES (3rd audit F2): prompts that exceed the cap on
        # their own.  This is a LOWER BOUND on how many requests get chunked -- the
        # batch-budget path chunks short requests too, and is runtime-only.
        "n_prompts_longer_than_cps": {
            str(c): sum(1 for l in plens if l > c) for c in CPS_GRID
        },
        # Order-of-magnitude frame for the batch channel, NOT a prediction: if the
        # whole served set were prefilled back-to-back under a full backlog, the cap
        # would be exhausted this many times.  Reported so nobody reads the count
        # above as "chunking cannot happen".
        "budget_exhaustions_if_fully_backlogged": {
            str(c): sum(plens) // c for c in CPS_GRID
        },
        "total_prompt_tokens": sum(plens),
        "sampler_stdout": _noise.getvalue().strip().splitlines(),
    }
    json.dump(out, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    a = sys.argv[1:]
    main(int(a[0]) if a else 200, int(a[1]) if len(a) > 1 else 4000, int(a[2]) if len(a) > 2 else 1)
