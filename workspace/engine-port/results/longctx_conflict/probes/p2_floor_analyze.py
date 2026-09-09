#!/usr/bin/env python3
"""P2 FLOOR analysis -- PREREG_PROBES_2026-09-08.md sec 2.

Reads one `bench_serving --output-details` JSONL (one JSON object per line;
P2 writes one line per ctx cell) and computes, EXCLUDING the first
`WARMUP` requests (submission/completion order == list order at
concurrency 1, sequential):

  floor_ref  = TTFT p50 (seconds)
  itl_solo   = ITL p50 over the FIXED TOKEN WINDOW [9, 32) of every
               post-warmup request's `itl` list, flattened
  realized input tokens = `input_lens` (post-warmup) summary stats
  TTFT p50/p95

No verdict beyond the registered +/-30%-of-prior-extrapolation check
(prereg sec 2, "등록 예측"): this is calibration, not a policy result.

Usage: python3 p2_floor_analyze.py <ctx> <details.jsonl> [--warmup 2] [--win-lo 9] [--win-hi 32]
"""
import argparse
import json
import sys


def pctl(a, q):
    if not a:
        return float("nan")
    a = sorted(a)
    p = q * (len(a) - 1)
    lo = int(p)
    hi = min(lo + 1, len(a) - 1)
    return a[lo] + (a[hi] - a[lo]) * (p - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ctx", type=int)
    ap.add_argument("details_jsonl")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--win-lo", type=int, default=9)
    ap.add_argument("--win-hi", type=int, default=32)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(args.details_jsonl) as fh:
        lines = [json.loads(l) for l in fh if l.strip()]
    if not lines:
        print(json.dumps({"ctx": args.ctx, "error": "EMPTY_OUTPUT_FILE"}))
        sys.exit(1)
    rec = lines[-1]  # one cell -> one appended line; take the last (safety if reused)

    ttfts_all = rec.get("ttfts", [])
    itls_all = rec.get("itls", [])
    input_lens_all = rec.get("input_lens", [])
    errors_all = rec.get("errors", [])
    n_total = len(ttfts_all)
    w = args.warmup
    ttfts = ttfts_all[w:]
    itls = itls_all[w:]
    input_lens = input_lens_all[w:]
    errors = errors_all[w:]
    n_errors = sum(1 for e in errors if e)

    window_vals = []
    for row in itls:
        window_vals.extend(row[args.win_lo:args.win_hi])

    out = dict(
        ctx=args.ctx,
        n_total_requests=n_total,
        n_warmup_excluded=min(w, n_total),
        n_measured=len(ttfts),
        n_errors_in_measured=n_errors,
        floor_ref_ttft_p50_s=pctl(ttfts, 0.50),
        ttft_p95_s=pctl(ttfts, 0.95),
        itl_solo_p50_s=pctl(window_vals, 0.50),
        n_itl_window_samples=len(window_vals),
        input_lens_realized=dict(
            mean=(sum(input_lens) / len(input_lens)) if input_lens else float("nan"),
            min=min(input_lens) if input_lens else None,
            max=max(input_lens) if input_lens else None,
            n=len(input_lens),
        ),
    )
    text = json.dumps(out, indent=2, sort_keys=True)
    print(text)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text)


if __name__ == "__main__":
    main()
