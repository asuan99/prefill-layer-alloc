#!/usr/bin/env python3
"""ShareGPT length census -- can a LONG-PROMPT variant of this trace exist?  GPU 0.

Why: the V-probe follow-up found that on the trace as configured, `cps = 8192`
splits ZERO requests (longest prompt 3712 tokens), so the chunked-prefill arm's
treatment never fires.  Before proposing a long-prompt workload, this measures
whether the supply exists -- over the WHOLE dataset, not a 200-prompt sample.

★It reproduces the canonical sampler's rules exactly rather than approximating
them (`sglang/benchmark/datasets/sharegpt.py:82-140`, read this session):
  * keep conversations with >= 2 turns; prompt = turn[0], completion = turn[1]
  * prune if prompt_len < 2 or output_len < 2
  * prune if `context_len` and prompt_len + output_len > context_len
  * tokenizer.encode with default add_special_tokens
The only thing NOT reproduced is the shuffle-then-take-first-N, because the
question here is about SUPPLY, not about which N a seed happens to draw.

Outputs a census: the length distribution, and for each candidate
(context_len cap, chunked-prefill-size) pair, how much of the surviving
population would actually be split.  No arm is run and no policy quantity exists
in this file.

Usage: python3 sharegpt_length_census.py [--limit N]
"""
import argparse, json, os, statistics as st, sys, time
from pathlib import Path

HF = os.environ.get("HF_HOME", "/scratch/ehmoon/whlee/prefill-layer-alloc/hf_cache")
DATA = Path(HF) / "raw" / "ShareGPT_V3_unfiltered_cleaned_split.json"
MODEL = "Zyphra/Zamba2-2.7B"
OUT = Path(__file__).resolve().parent / "sharegpt_length_census.json"

# the harness's current cap, and the caps a long-prompt variant would need
CONTEXT_CAPS = (4000, 8192, 16384, 32768, None)
CPS_GRID = (512, 1024, 2048, 4096, 8192)
QS = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.999)


def q(sorted_vals, p):
    if not sorted_vals:
        return None
    return sorted_vals[min(len(sorted_vals) - 1, int(p * len(sorted_vals)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="tokenize only the first N (0 = all)")
    ap.add_argument("--batch", type=int, default=512)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    print(f"[{time.time()-t0:.1f}s] tokenizer loaded (fast={tok.is_fast})", flush=True)

    raw = json.load(open(DATA))
    convs = [c for c in raw
             if len(c.get("conversations", c.get("conversation", []))) >= 2]
    pairs = [(c.get("conversations", c.get("conversation", []))[0]["value"],
              c.get("conversations", c.get("conversation", []))[1]["value"])
             for c in convs]
    if args.limit:
        pairs = pairs[:args.limit]
    print(f"[{time.time()-t0:.1f}s] {len(raw)} conversations, {len(convs)} with >=2 turns, "
          f"tokenizing {len(pairs)}", flush=True)

    plens, olens = [], []
    for i in range(0, len(pairs), args.batch):
        chunk = pairs[i:i + args.batch]
        pe = tok([p for p, _ in chunk], add_special_tokens=True)["input_ids"]
        oe = tok([o for _, o in chunk], add_special_tokens=True)["input_ids"]
        plens += [len(x) for x in pe]
        olens += [len(x) for x in oe]
        if (i // args.batch) % 40 == 0:
            print(f"  [{time.time()-t0:6.1f}s] {i+len(chunk)}/{len(pairs)}", flush=True)

    # the sampler's own short-prune, applied before anything else
    rows = [(p, o) for p, o in zip(plens, olens) if p >= 2 and o >= 2]
    print(f"[{time.time()-t0:.1f}s] {len(rows)} rows survive the short-prune "
          f"(prompt_len>=2 and output_len>=2)", flush=True)

    res = {
        "_what_this_is": "ShareGPT length census under the canonical sampler's rules. "
                         "Supply analysis only -- no arm, no policy quantity.",
        "dataset": str(DATA), "model": MODEL,
        "n_conversations": len(raw), "n_two_turn": len(convs),
        "n_tokenized": len(pairs), "n_after_short_prune": len(rows),
        "by_context_cap": {},
    }

    for cap in CONTEXT_CAPS:
        kept = [(p, o) for p, o in rows if (cap is None or p + o <= cap)]
        ps = sorted(p for p, _ in kept)
        os_ = sorted(o for _, o in kept)
        entry = {
            "context_len_cap": cap,
            "n_kept": len(kept),
            # denominator is what was ACTUALLY tokenized, not the whole corpus --
            # with --limit the two differ and dividing by the corpus understates
            # every fraction by the sampling ratio (caught on the timing run).
            "frac_of_tokenized": len(kept) / len(pairs) if pairs else None,
            "prompt_len": {f"p{int(p*1000)/10:g}": q(ps, p) for p in QS} if ps else None,
            "prompt_len_mean": st.fmean(ps) if ps else None,
            "prompt_len_max": ps[-1] if ps else None,
            "output_len": {f"p{int(p*1000)/10:g}": q(os_, p) for p in QS} if os_ else None,
            "output_len_mean": st.fmean(os_) if os_ else None,
            # THE question: with this cap, how many prompts would a given
            # --chunked-prefill-size actually split (prompt_len > cps)?
            "n_prompt_longer_than_cps": {
                str(c): sum(1 for p in ps if p > c) for c in CPS_GRID},
            "frac_prompt_longer_than_cps": {
                str(c): (sum(1 for p in ps if p > c) / len(ps)) if ps else None
                for c in CPS_GRID},
        }
        res["by_context_cap"][str(cap)] = entry

    # A long-prompt VARIANT: keep only prompts at or above a floor, and ask how
    # much supply remains and what the decode side looks like (PD-mux needs decode).
    res["long_prompt_variants"] = {}
    for floor in (1024, 2048, 4096, 8192):
        for cap in (8192, 16384, 32768, None):
            if cap is not None and floor >= cap:
                continue
            kept = [(p, o) for p, o in rows
                    if p >= floor and (cap is None or p + o <= cap)]
            if not kept:
                res["long_prompt_variants"][f"floor{floor}_cap{cap}"] = {"n": 0}
                continue
            ps = sorted(p for p, _ in kept); os_ = sorted(o for _, o in kept)
            res["long_prompt_variants"][f"floor{floor}_cap{cap}"] = {
                "n_available": len(kept),
                "prompt_len": {"p50": q(ps, .5), "p90": q(ps, .9), "max": ps[-1],
                               "mean": round(st.fmean(ps), 1)},
                "output_len": {"p50": q(os_, .5), "p90": q(os_, .9), "max": os_[-1],
                               "mean": round(st.fmean(os_), 1)},
                "n_prompt_longer_than_cps": {str(c): sum(1 for p in ps if p > c)
                                             for c in CPS_GRID},
            }

    OUT.write_text(json.dumps(res, indent=1))
    print(f"\n[{time.time()-t0:.1f}s] wrote {OUT}")

    print("\n=== 전체 모집단 (short-prune 후) ===")
    for cap in CONTEXT_CAPS:
        e = res["by_context_cap"][str(cap)]
        pl = e["prompt_len"]
        print(f"\n  context_len cap = {cap}:  {e['n_kept']} rows "
              f"({e['frac_of_tokenized']*100:.1f}% of tokenized)")
        if pl:
            print(f"    prompt_len  p50 {pl['p50']}  p90 {pl['p90']}  p95 {pl['p95']}  "
                  f"p99 {pl['p99']}  max {e['prompt_len_max']}  mean {e['prompt_len_mean']:.0f}")
            print(f"    output_len  p50 {e['output_len']['p50']}  p90 {e['output_len']['p90']}  "
                  f"mean {e['output_len_mean']:.0f}")
            print("    prompt_len > cps:  " + "  ".join(
                f"{c}:{e['n_prompt_longer_than_cps'][str(c)]}"
                f"({e['frac_prompt_longer_than_cps'][str(c)]*100:.1f}%)" for c in CPS_GRID))


if __name__ == "__main__":
    main()
