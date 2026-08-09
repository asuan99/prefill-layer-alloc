#!/usr/bin/env python3
"""G-2 negative-control concurrent gate (PREREG_G2CTRL_2026-08-07.md).

Fires N identical greedy /generate requests at a single sglang server
CONCURRENTLY and compares each response's output_ids against up to TWO
baselines computed at batch=1 on the *same* server/arm boot:

  self  -- this arm's OWN batch=1 greedy output (the negative-control
           comparison; every arm gets this).
  cross -- another arm's (normally A1 plain's) batch=1 greedy output on the
           identical prompt (only meaningful for A3 chunk512, to reproduce
           job 875346's original G-2 cross-arm comparison alongside the new
           self comparison, so the two can be told apart).

Unlike the original g2_concurrent_gate.py (job 875344/875346), this script
stores full output_ids arrays (not just sha256) so the caller/analyzer can
locate the first diverging generated-token index for any mismatch -- needed
to distinguish a chunk-boundary state-carryover defect (expected to surface
as a locked-in divergence from generation step 0) from ordinary
batch-composition numerical non-determinism (expected to surface at a
step that varies rep-to-rep / request-to-request).

STATUS discipline matches g2_holb_phaseA_lib.sh / g2_concurrent_gate.py:
HTTP status code checked first and independently of body; "could not be
measured" (non-200 / transport error / malformed body) is its own category,
never folded into a mismatch. verdict is PASS / FAIL / UNDETERMINED per
comparison (self, cross) independently.
"""
import argparse
import concurrent.futures
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request


def one_call(host: str, port: int, prompt: str, max_new_tokens: int, idx: int) -> dict:
    body = json.dumps(
        {"text": prompt, "sampling_params": {"max_new_tokens": max_new_tokens, "temperature": 0}}
    ).encode()
    req = urllib.request.Request(
        f"http://{host}:{port}/generate",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            code = resp.status
            raw = resp.read()
    except urllib.error.HTTPError as e:
        code = e.code
        raw = e.read()
    except Exception as e:  # transport-level failure
        return {
            "idx": idx, "status": "ERROR", "http_code": None,
            "detail": f"transport_error:{type(e).__name__}:{e}",
            "elapsed_s": time.time() - t0,
        }
    elapsed = time.time() - t0
    if code != 200:
        return {
            "idx": idx, "status": "ERROR", "http_code": code,
            "detail": f"non200_body:{raw[:300]!r}", "elapsed_s": elapsed,
        }
    try:
        d = json.loads(raw)
    except Exception as e:
        return {
            "idx": idx, "status": "ERROR", "http_code": code,
            "detail": f"json_parse_failed:{e}:{raw[:200]!r}", "elapsed_s": elapsed,
        }
    if not isinstance(d, dict):
        return {
            "idx": idx, "status": "ERROR", "http_code": code,
            "detail": f"not_a_dict:{type(d).__name__}", "elapsed_s": elapsed,
        }
    if "error" in d:
        return {
            "idx": idx, "status": "ERROR", "http_code": code,
            "detail": f"http_200_but_error_key:{str(d['error'])[:300]}", "elapsed_s": elapsed,
        }
    if d.get("output_ids"):
        ids = d["output_ids"]
        sha = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
        return {
            "idx": idx, "status": "OK", "http_code": code, "method": "output_ids",
            "sha": sha, "output_ids": ids, "n_output_ids": len(ids), "elapsed_s": elapsed,
        }
    if "text" in d:
        # No output_ids available (skip_tokenizer_init=True path) -- cannot
        # locate a first-diff *token* index; still record for completeness.
        sha = hashlib.sha256(d["text"].encode()).hexdigest()
        return {
            "idx": idx, "status": "OK", "http_code": code, "method": "text",
            "sha": sha, "output_ids": None, "n_output_ids": 0, "elapsed_s": elapsed,
        }
    return {
        "idx": idx, "status": "ERROR", "http_code": code,
        "detail": f"no_output_ids_or_text_keys={sorted(d.keys())}", "elapsed_s": elapsed,
    }


def first_diff_index(a, b):
    """First index where sequences a, b differ. None if equal on the common
    prefix and same length. Returns (idx, len_a, len_b); idx is None if no
    difference found on the common prefix (but lengths may still differ)."""
    if a is None or b is None:
        return None, (len(a) if a is not None else None), (len(b) if b is not None else None)
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i, len(a), len(b)
    if len(a) != len(b):
        return n, len(a), len(b)  # diverge exactly at the length boundary
    return None, len(a), len(b)


def score(results, baseline_ids, baseline_sha, label):
    n_ok = sum(1 for r in results if r["status"] == "OK")
    n_error = len(results) - n_ok
    mismatches = []
    n_match = 0
    for r in results:
        if r["status"] != "OK":
            continue
        if r["sha"] == baseline_sha:
            n_match += 1
            continue
        idx0, la, lb = first_diff_index(r.get("output_ids"), baseline_ids)
        mismatches.append({
            "idx": r["idx"], "sha": r["sha"], "baseline_sha": baseline_sha,
            "first_diff_token_index": idx0, "len_this": la, "len_baseline": lb,
        })
    n_mismatch = n_ok - n_match
    if n_error > 0:
        verdict = "UNDETERMINED"
    elif n_mismatch > 0:
        verdict = "FAIL"
    else:
        verdict = "PASS"
    return {
        "label": label, "n": len(results), "n_ok": n_ok, "n_error": n_error,
        "n_match": n_match, "n_mismatch": n_mismatch, "verdict": verdict,
        "mismatches": mismatches,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--max-new-tokens", type=int, required=True)
    ap.add_argument("--n-concurrent", type=int, default=16)
    ap.add_argument("--self-baseline-file", required=True,
                     help="JSON file with {'sha':..., 'output_ids':[...]} for this arm's own batch=1 baseline")
    ap.add_argument("--cross-baseline-file", default=None,
                     help="Optional JSON file with another arm's (A1) batch=1 baseline, for A3's cross-arm check")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    prompt = open(args.prompt_file).read()
    self_baseline = json.load(open(args.self_baseline_file))
    cross_baseline = json.load(open(args.cross_baseline_file)) if args.cross_baseline_file else None

    results = [None] * args.n_concurrent
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.n_concurrent) as ex:
        futs = {
            ex.submit(one_call, args.host, args.port, prompt, args.max_new_tokens, i): i
            for i in range(args.n_concurrent)
        }
        for fut in concurrent.futures.as_completed(futs):
            r = fut.result()
            results[r["idx"]] = r

    self_score = score(results, self_baseline["output_ids"], self_baseline["sha"], "self")
    cross_score = None
    if cross_baseline is not None:
        cross_score = score(results, cross_baseline["output_ids"], cross_baseline["sha"], "cross")

    # Strip output_ids from the per-request records before dumping (keep sha
    # + lengths only) to bound file size; full ids are still in `results`
    # in-memory and are not needed downstream once mismatches[] has captured
    # the first-diff index.
    results_compact = [{k: v for k, v in r.items() if k != "output_ids"} for r in results]

    out = {
        "n_concurrent": args.n_concurrent,
        "self_baseline_sha": self_baseline["sha"],
        "cross_baseline_sha": (cross_baseline["sha"] if cross_baseline else None),
        "self": self_score,
        "cross": cross_score,
        "results": results_compact,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    cross_str = ""
    if cross_score is not None:
        cross_str = (f" cross(n_ok={cross_score['n_ok']} n_match={cross_score['n_match']} "
                     f"n_mismatch={cross_score['n_mismatch']} verdict={cross_score['verdict']})")
    print(
        f"G2CTRL n_concurrent={args.n_concurrent} "
        f"self(n_ok={self_score['n_ok']} n_match={self_score['n_match']} "
        f"n_mismatch={self_score['n_mismatch']} verdict={self_score['verdict']})" + cross_str
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
