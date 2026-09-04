#!/usr/bin/env python3
"""Re-tokenise the registered long-prompt tail with the CAMPAIGN tokenizer and
report the strata AF-1 screens at.  CPU only, GPU 0.

Why this exists
---------------
`PREREG_AF1_2026-09-04.md` sec 1 registers that AF-1 computes its own strata by
re-tokenising, and that it does NOT cite the dataset metadata's token counts --
those were produced with the Zamba2 tokenizer to SELECT rows, and the builder
itself marks them "_not_a_measurement".  sec 5.1 nonetheless quoted 2,795 /
4,246 / 6,500 for its roofline prediction, which 3rd-audit J5-d flagged as the
pre-registration breaking its own rule.  This script buys the real numbers.

What it is and is not
---------------------
It measures a property of (dataset, tokenizer).  No arm, no engine, no latency.
Running it before the campaign creates no selection channel: the screen's
thresholds are external constants (sec 4.1) and the strata quantiles are
registered constants (`REPORT_STRATUM_QS`), so neither can be chosen from this
output.  What it DOES change is the sharpness of the registered prediction --
sec 5.1's second prediction turns on whether the p90 stratum is near 4,246 tok
or 20% above it.

Usage: python3 af1_strata.py [--out af1_strata.json]
"""
import argparse, json, os, statistics, sys, time
from pathlib import Path

HF = os.environ.get("HF_HOME", "/scratch/ehmoon/whlee/prefill-layer-alloc/hf_cache")
DATASET = Path(HF) / "raw" / "ShareGPT_long2048_cap8192.json"
MODEL = "nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base"
ZAMBA = "Zyphra/Zamba2-2.7B"          # the tokenizer that SELECTED the rows


def quantile(xs, q):
    """Same linear-interpolation order statistic the predicates register."""
    s = sorted(xs)
    if len(s) == 1:
        return float(s[0])
    pos = q * (len(s) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return float(s[lo] + (pos - lo) * (s[hi] - s[lo]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="af1_strata.json")
    ap.add_argument("--qs", default="0.10,0.50,0.90,0.99")
    a = ap.parse_args()
    qs = [float(x) for x in a.qs.split(",")]

    t0 = time.time()
    from transformers import AutoTokenizer      # ~4 min cold on this filesystem
    print(f"transformers import: {time.time() - t0:.0f}s", flush=True)

    rows = json.load(open(DATASET))
    # the canonical sampler's own rule: prompt = turn[0] of a >=2-turn conversation
    prompts = [c["conversations"][0]["value"] for c in rows
               if len(c.get("conversations", [])) >= 2]
    print(f"{len(rows)} rows -> {len(prompts)} prompts", flush=True)

    out = {"_what_this_is": "AF-1 strata: prompt length re-tokenised with the "
                            "CAMPAIGN tokenizer. CPU only, GPU 0, no arm involved.",
           "dataset": str(DATASET), "n_prompts": len(prompts),
           "registered_quantiles": qs, "tokenizers": {}}

    for name, mid in (("campaign", MODEL), ("selection", ZAMBA)):
        t = time.time()
        tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
        lens = [len(tok(p, add_special_tokens=False)["input_ids"]) for p in prompts]
        out["tokenizers"][name] = {
            "model": mid,
            "quantiles": {str(q): quantile(lens, q) for q in qs},
            "min": min(lens), "max": max(lens),
            "mean": round(statistics.fmean(lens), 1),
            "seconds": round(time.time() - t, 1),
        }
        print(f"{name:9s} {mid}", flush=True)
        for q in qs:
            print(f"    q={q:<5} {quantile(lens, q):9.1f} tok", flush=True)

    c = out["tokenizers"]["campaign"]["quantiles"]
    z = out["tokenizers"]["selection"]["quantiles"]
    out["campaign_over_selection"] = {k: round(c[k] / z[k], 4) for k in c}
    print("\ncampaign / selection ratio:", out["campaign_over_selection"], flush=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"wrote {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
