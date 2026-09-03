#!/usr/bin/env python3
"""Build the LONG-PROMPT TAIL of ShareGPT as a derived dataset file.  GPU 0.

Why a derived FILE rather than a harness patch: `bench_serving` has no
minimum-prompt-length option, and the canonical sampler
(`sglang/benchmark/datasets/sharegpt.py`) is a file reader.  Writing the subset to
its own file therefore changes NOTHING in the engine or the harness -- the same
sampler, the same rules, the same `--dataset-path` argument, a different file.
That keeps the workload swap on the DATA side where it can be audited by reading
one JSON, instead of on the code side where it would need a patch and a manifest.

Selection (registered in PREREG_W1; every number here comes from
`sharegpt_length_census.json`, measured over all 92,824 rows):
  * >= 2 turns; prompt = turn[0]  (the sampler's own rule)
  * prompt_len >= FLOOR            -- so `--chunked-prefill-size FLOOR` actually fires
  * prompt_len + output_len <= CAP -- so the run fits in --context-length CAP

★The output file keeps the ORIGINAL schema and the ORIGINAL text, so the sampler
re-tokenizes from scratch and this script's token counts are never trusted
downstream: they select rows, they do not become measurements.

Usage: python3 build_sharegpt_long.py [--floor 2048] [--cap 8192]
"""
import argparse, json, os, statistics as st, time
from pathlib import Path

HF = os.environ.get("HF_HOME", "/scratch/ehmoon/whlee/prefill-layer-alloc/hf_cache")
SRC = Path(HF) / "raw" / "ShareGPT_V3_unfiltered_cleaned_split.json"
MODEL = "Zyphra/Zamba2-2.7B"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor", type=int, default=2048)
    ap.add_argument("--cap", type=int, default=8192)
    ap.add_argument("--batch", type=int, default=512)
    args = ap.parse_args()
    dst = Path(HF) / "raw" / f"ShareGPT_long{args.floor}_cap{args.cap}.json"

    from transformers import AutoTokenizer
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    raw = json.load(open(SRC))
    keep_idx, pl, ol = [], [], []
    idx = [i for i, c in enumerate(raw)
           if len(c.get("conversations", c.get("conversation", []))) >= 2]
    print(f"[{time.time()-t0:.1f}s] {len(raw)} conversations, {len(idx)} with >=2 turns",
          flush=True)

    for b in range(0, len(idx), args.batch):
        part = idx[b:b + args.batch]
        conv = [raw[i].get("conversations", raw[i].get("conversation", [])) for i in part]
        pe = tok([c[0]["value"] for c in conv], add_special_tokens=True)["input_ids"]
        oe = tok([c[1]["value"] for c in conv], add_special_tokens=True)["input_ids"]
        for i, p, o in zip(part, pe, oe):
            lp, lo = len(p), len(o)
            if lp >= args.floor and lo >= 2 and lp + lo <= args.cap:
                keep_idx.append(i); pl.append(lp); ol.append(lo)
        if (b // args.batch) % 60 == 0:
            print(f"  [{time.time()-t0:6.1f}s] scanned {b+len(part)}/{len(idx)}, "
                  f"kept {len(keep_idx)}", flush=True)

    out = [raw[i] for i in keep_idx]
    dst.write_text(json.dumps(out))
    ps, os_ = sorted(pl), sorted(ol)

    def q(a, p):
        return a[min(len(a) - 1, int(p * len(a)))]

    meta = {
        "_what_this_is": f"ShareGPT long-prompt tail: prompt_len >= {args.floor} and "
                         f"prompt_len + output_len <= {args.cap}, Zamba2-2.7B tokenizer.",
        "_not_a_measurement": "Token counts here SELECT rows. The harness re-tokenizes "
                              "from the text, so nothing downstream cites these numbers.",
        "source": str(SRC), "floor": args.floor, "cap": args.cap, "model": MODEL,
        "n_source_conversations": len(raw), "n_two_turn": len(idx), "n_kept": len(out),
        "prompt_len": {"p10": q(ps, .1), "p50": q(ps, .5), "p90": q(ps, .9),
                       "max": ps[-1], "mean": round(st.fmean(ps), 1)},
        "output_len": {"p10": q(os_, .1), "p50": q(os_, .5), "p90": q(os_, .9),
                       "max": os_[-1], "mean": round(st.fmean(os_), 1)},
        "n_prompt_longer_than_cps": {str(c): sum(1 for p in ps if p > c)
                                     for c in (512, 1024, 2048, 4096)},
        "dataset_path": str(dst),
    }
    Path(str(dst) + ".meta.json").write_text(json.dumps(meta, indent=1))
    print(f"\n[{time.time()-t0:.1f}s] kept {len(out)} conversations -> {dst}")
    print(f"  prompt_len  p50 {meta['prompt_len']['p50']}  p90 {meta['prompt_len']['p90']}  "
          f"max {meta['prompt_len']['max']}  mean {meta['prompt_len']['mean']}")
    print(f"  output_len  p50 {meta['output_len']['p50']}  p90 {meta['output_len']['p90']}  "
          f"mean {meta['output_len']['mean']}")
    print(f"  prompt_len > cps: {meta['n_prompt_longer_than_cps']}")


if __name__ == "__main__":
    main()
