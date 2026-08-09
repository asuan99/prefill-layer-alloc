#!/usr/bin/env python3
"""Gate 2 G-2 concurrent correctness gate (PREREG_GATE2_2026-08-06.md section 2.1).

Fires N identical greedy /generate requests at a single sglang server
CONCURRENTLY and compares each response's output tokens against a
batch=1 baseline sha256 (computed separately -- normally A1's G-1 greedy
call on the exact same prompt). This is the gate that actually exercises
mixed-batch composition + mamba pool pressure; a sequential/batch=1 check
(G-1) cannot.

Follows the same STATUS discipline as g2_holb_phaseA_lib.sh's greedy_call:
every request's HTTP status code is checked FIRST and independently of the
response body, and "could not be measured" (non-200, malformed body,
transport error) is reported as its own category -- never silently folded
into a mismatch verdict. This project's methodology gate against conflating
UNDETERMINED with FAIL (see g2_holb_phaseA_lib.sh header, job 874601
post-mortem) applies here too.

Usage:
  python3 g2_concurrent_gate.py --host 127.0.0.1 --port PORT \
      --prompt-file PROMPT.txt --max-new-tokens N --n-concurrent 16 \
      --baseline-sha SHA256HEX --out RESULT.json

Never raises on a request-level failure (exit code is always 0 unless the
CLI itself is misused); the caller inspects the JSON's top-level "verdict":
  PASS          -- all N requests measured OK and matched the baseline
  FAIL          -- all N requests measured OK but >=1 output differs
  UNDETERMINED  -- >=1 request could not be measured (non-200 / transport
                   error / malformed body); this is NOT evidence of a
                   mismatch and must not be reported as a FAIL
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
    except Exception as e:  # transport-level failure (connection reset, timeout, refused, ...)
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
        sha = hashlib.sha256(json.dumps(d["output_ids"]).encode()).hexdigest()
        return {
            "idx": idx, "status": "OK", "http_code": code, "method": "output_ids",
            "sha": sha, "n_output_ids": len(d["output_ids"]), "elapsed_s": elapsed,
        }
    if "text" in d:
        sha = hashlib.sha256(d["text"].encode()).hexdigest()
        return {
            "idx": idx, "status": "OK", "http_code": code, "method": "text",
            "sha": sha, "n_output_ids": 0, "elapsed_s": elapsed,
        }
    return {
        "idx": idx, "status": "ERROR", "http_code": code,
        "detail": f"no_output_ids_or_text_keys={sorted(d.keys())}", "elapsed_s": elapsed,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--max-new-tokens", type=int, required=True)
    ap.add_argument("--n-concurrent", type=int, default=16)
    ap.add_argument("--baseline-sha", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    prompt = open(args.prompt_file).read()
    results = [None] * args.n_concurrent
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.n_concurrent) as ex:
        futs = {
            ex.submit(one_call, args.host, args.port, prompt, args.max_new_tokens, i): i
            for i in range(args.n_concurrent)
        }
        for fut in concurrent.futures.as_completed(futs):
            r = fut.result()
            results[r["idx"]] = r

    n_ok = sum(1 for r in results if r["status"] == "OK")
    n_error = args.n_concurrent - n_ok
    n_match = sum(1 for r in results if r["status"] == "OK" and r["sha"] == args.baseline_sha)
    n_mismatch = n_ok - n_match

    if n_error > 0:
        verdict = "UNDETERMINED"
    elif n_mismatch > 0:
        verdict = "FAIL"
    else:
        verdict = "PASS"

    out = {
        "n_concurrent": args.n_concurrent,
        "baseline_sha": args.baseline_sha,
        "n_ok": n_ok, "n_error": n_error, "n_match": n_match, "n_mismatch": n_mismatch,
        "verdict": verdict, "results": results,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(
        f"G2_CONCURRENT n_concurrent={args.n_concurrent} n_ok={n_ok} n_error={n_error} "
        f"n_match={n_match} n_mismatch={n_mismatch} verdict={verdict}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
