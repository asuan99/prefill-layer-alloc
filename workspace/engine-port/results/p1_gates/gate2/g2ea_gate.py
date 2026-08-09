#!/usr/bin/env python3
"""E-A correctness gates (PREREG_G2EA_2026-08-07.md).

Fixes harness defect #2 identified by claims-auditor's 2026-08-07 audit of
875344/875346: g2_concurrent_gate.py AND g2ctrl_gate.py both stored sha256
only (or stripped output_ids before dumping), so a mismatch's exact location
could not be reconstructed post hoc. This script:

  (a) ALWAYS persists full output_ids for every OK response (not just sha),
      for every arm this is run against (A1m, A3m, A3 -- "processes the new
      path" per the task brief, which subsumes E-B).
  (b) On any mismatch, computes first_diff_token_index (first index in the
      OUTPUT token stream where the two sequences differ) AND its distance
      to the nearest input-prompt chunk boundary in {512,1024,1536,2048}
      (plus any further multiples of --chunked-prefill-size up to the
      prompt's reencoded length, for completeness beyond the four requested
      boundaries). This is a diagnostic proximity metric in token-count
      space, NOT a claim that output-token index N was literally produced
      by prefill chunk N -- generation only starts after the full prefill
      completes, so a chunk-boundary state-carryover defect is expected to
      manifest as a LOCKED-IN divergence from output index 0 (this project's
      existing g2ctrl_gate.py docstring already makes this point; this
      script keeps that framing and adds the boundary-distance number
      alongside it, mechanically, for the record).

Two subcommands:
  single      -- G-1 style: N sequential greedy calls (default 2) against ONE
                 server, self-repro compared pairwise, byte-identical output
                 required between calls. Optional --cross-baseline-file for
                 an A1-vs-arm byte-identical comparison (rev4 section 2.1).
  concurrent  -- G-2 style: N concurrent greedy calls against ONE server,
                 each compared to a self baseline (this arm's own G-1 output)
                 and optionally a cross baseline (A1's G-1 output).

STATUS discipline unchanged from g2_concurrent_gate.py / g2ctrl_gate.py /
g2_holb_phaseA_lib.sh: HTTP status checked first and independently of body;
"could not be measured" is its own category, never folded into a mismatch.
"""
import argparse
import concurrent.futures
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request

DEFAULT_BOUNDARIES = (512, 1024, 1536, 2048)


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
        return {"idx": idx, "status": "ERROR", "http_code": None,
                "detail": f"transport_error:{type(e).__name__}:{e}", "elapsed_s": time.time() - t0}
    elapsed = time.time() - t0
    if code != 200:
        return {"idx": idx, "status": "ERROR", "http_code": code,
                "detail": f"non200_body:{raw[:300]!r}", "elapsed_s": elapsed}
    try:
        d = json.loads(raw)
    except Exception as e:
        return {"idx": idx, "status": "ERROR", "http_code": code,
                "detail": f"json_parse_failed:{e}:{raw[:200]!r}", "elapsed_s": elapsed}
    if not isinstance(d, dict):
        return {"idx": idx, "status": "ERROR", "http_code": code,
                "detail": f"not_a_dict:{type(d).__name__}", "elapsed_s": elapsed}
    if "error" in d:
        return {"idx": idx, "status": "ERROR", "http_code": code,
                "detail": f"http_200_but_error_key:{str(d['error'])[:300]}", "elapsed_s": elapsed}
    if d.get("output_ids"):
        ids = d["output_ids"]
        sha = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
        return {"idx": idx, "status": "OK", "http_code": code, "method": "output_ids",
                "sha": sha, "output_ids": ids, "n_output_ids": len(ids), "elapsed_s": elapsed}
    if "text" in d:
        sha = hashlib.sha256(d["text"].encode()).hexdigest()
        return {"idx": idx, "status": "OK", "http_code": code, "method": "text",
                "sha": sha, "output_ids": None, "n_output_ids": 0, "elapsed_s": elapsed}
    return {"idx": idx, "status": "ERROR", "http_code": code,
            "detail": f"no_output_ids_or_text_keys={sorted(d.keys())}", "elapsed_s": elapsed}


def first_diff_index(a, b):
    if a is None or b is None:
        return None, (len(a) if a is not None else None), (len(b) if b is not None else None)
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i, len(a), len(b)
    if len(a) != len(b):
        return n, len(a), len(b)
    return None, len(a), len(b)


def chunk_boundaries(chunked_prefill_size, prompt_len):
    bset = set(DEFAULT_BOUNDARIES)
    if chunked_prefill_size and chunked_prefill_size > 0:
        b = chunked_prefill_size
        while b < prompt_len:
            bset.add(b)
            b += chunked_prefill_size
    return sorted(x for x in bset if x <= max(prompt_len, max(DEFAULT_BOUNDARIES)))


def nearest_boundary_distance(idx, boundaries):
    if idx is None or not boundaries:
        return None, None
    best_b, best_d = None, None
    for b in boundaries:
        d = abs(idx - b)
        if best_d is None or d < best_d:
            best_d, best_b = d, b
    return best_b, best_d


def annotate_mismatch(r, baseline_ids, baseline_sha, boundaries):
    idx0, la, lb = first_diff_index(r.get("output_ids"), baseline_ids)
    nb, dist = nearest_boundary_distance(idx0, boundaries)
    return {
        "idx": r["idx"], "sha": r["sha"], "baseline_sha": baseline_sha,
        "first_diff_token_index": idx0, "len_this": la, "len_baseline": lb,
        "nearest_chunk_boundary": nb, "distance_to_nearest_chunk_boundary": dist,
    }


def cmd_single(args) -> int:
    prompt = open(args.prompt_file).read()
    calls = [one_call(args.host, args.port, prompt, args.max_new_tokens, i) for i in range(args.n_calls)]
    ok = [c for c in calls if c["status"] == "OK"]
    self_repro = None
    if len(ok) >= 2:
        self_repro = all(c["sha"] == ok[0]["sha"] for c in ok[1:])
    boundaries = chunk_boundaries(args.chunked_prefill_size, args.prompt_len_hint or 0)
    cross_result = None
    if args.cross_baseline_file and ok:
        cross = json.load(open(args.cross_baseline_file))
        c0 = ok[0]
        match = c0["sha"] == cross["sha"]
        cross_result = {"match": match, "baseline_sha": cross["sha"], "this_sha": c0["sha"]}
        if not match:
            cross_result["diff"] = annotate_mismatch(c0, cross.get("output_ids"), cross["sha"], boundaries)
    status = "UNDETERMINED" if len(ok) < len(calls) else (
        "SELFREPRO_OK" if self_repro else "SELFREPRO_FAIL")
    out = {
        "mode": "single", "n_calls": args.n_calls, "n_ok": len(ok),
        "self_repro_status": status, "sha": (ok[0]["sha"] if ok else None),
        "output_ids": (ok[0].get("output_ids") if ok else None),
        "chunk_boundaries": boundaries,
        "cross_baseline_check": cross_result,
        "calls": calls,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"G2EA_SINGLE n_calls={args.n_calls} n_ok={len(ok)} status={status} "
          f"cross={cross_result['match'] if cross_result else None}")
    return 0


def cmd_concurrent(args) -> int:
    prompt = open(args.prompt_file).read()
    self_baseline = json.load(open(args.self_baseline_file))
    cross_baseline = json.load(open(args.cross_baseline_file)) if args.cross_baseline_file else None
    boundaries = chunk_boundaries(args.chunked_prefill_size, args.prompt_len_hint or 0)

    results = [None] * args.n_concurrent
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.n_concurrent) as ex:
        futs = {ex.submit(one_call, args.host, args.port, prompt, args.max_new_tokens, i): i
                for i in range(args.n_concurrent)}
        for fut in concurrent.futures.as_completed(futs):
            r = fut.result()
            results[r["idx"]] = r

    def score(baseline_ids, baseline_sha, label):
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
            mismatches.append(annotate_mismatch(r, baseline_ids, baseline_sha, boundaries))
        n_mismatch = n_ok - n_match
        verdict = "UNDETERMINED" if n_error > 0 else ("FAIL" if n_mismatch > 0 else "PASS")
        return {"label": label, "n": len(results), "n_ok": n_ok, "n_error": n_error,
                "n_match": n_match, "n_mismatch": n_mismatch, "verdict": verdict,
                "mismatches": mismatches}

    self_score = score(self_baseline["output_ids"], self_baseline["sha"], "self")
    cross_score = score(cross_baseline["output_ids"], cross_baseline["sha"], "cross") if cross_baseline else None

    out = {
        "mode": "concurrent", "n_concurrent": args.n_concurrent,
        "chunk_boundaries": boundaries,
        "self_baseline_sha": self_baseline["sha"],
        "cross_baseline_sha": (cross_baseline["sha"] if cross_baseline else None),
        "self": self_score, "cross": cross_score,
        # full per-request records INCLUDING output_ids -- fix for defect #2
        # (previous g2_concurrent_gate.py/g2ctrl_gate.py stripped or never
        # stored these, making post-hoc mismatch-location reconstruction
        # impossible).
        "results": results,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    cross_str = ""
    if cross_score is not None:
        cross_str = (f" cross(n_ok={cross_score['n_ok']} n_match={cross_score['n_match']} "
                     f"n_mismatch={cross_score['n_mismatch']} verdict={cross_score['verdict']})")
    print(f"G2EA_CONCURRENT n_concurrent={args.n_concurrent} "
          f"self(n_ok={self_score['n_ok']} n_match={self_score['n_match']} "
          f"n_mismatch={self_score['n_mismatch']} verdict={self_score['verdict']})" + cross_str)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("single")
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, required=True)
    sp.add_argument("--prompt-file", required=True)
    sp.add_argument("--max-new-tokens", type=int, required=True)
    sp.add_argument("--n-calls", type=int, default=2)
    sp.add_argument("--chunked-prefill-size", type=int, default=None)
    sp.add_argument("--prompt-len-hint", type=int, default=None)
    sp.add_argument("--cross-baseline-file", default=None)
    sp.add_argument("--out", required=True)
    sp.set_defaults(func=cmd_single)

    cp = sub.add_parser("concurrent")
    cp.add_argument("--host", default="127.0.0.1")
    cp.add_argument("--port", type=int, required=True)
    cp.add_argument("--prompt-file", required=True)
    cp.add_argument("--max-new-tokens", type=int, required=True)
    cp.add_argument("--n-concurrent", type=int, default=16)
    cp.add_argument("--chunked-prefill-size", type=int, default=None)
    cp.add_argument("--prompt-len-hint", type=int, default=None)
    cp.add_argument("--self-baseline-file", required=True)
    cp.add_argument("--cross-baseline-file", default=None)
    cp.add_argument("--out", required=True)
    cp.set_defaults(func=cmd_concurrent)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
