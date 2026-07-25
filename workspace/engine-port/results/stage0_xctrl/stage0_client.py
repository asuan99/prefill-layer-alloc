#!/usr/bin/env python3
"""Stage 0 client driver: measure wall inter-token latency (ITL) via streaming.

Sends concurrent streaming /generate requests to a running SGLang server and
records the wall-clock arrival timestamp of every streamed token chunk. The
inter-arrival gap between consecutive chunks of one request is the wall ITL
(the FIRST gap is dropped -- it includes TTFT/prefill, not steady decode).

Measurement is client-side wall time only; no engine internals, no ZBLT layer
decomposition. Raw per-token timestamps are written to a JSONL so p50/p95/p99
can be recomputed downstream.

Optional --keepalive-prefill starts a background trickle of medium-length
prompts. This is needed for the sub-108 decode-SM arms (d16/d44): the pdmux
event loop only pins decode to a green-ctx sub-partition while a prefill is
CONCURRENTLY in flight (split_prefill_batch != None); a pure decode-only phase
is forced to full 108 SM (multiplexing_mixin.py:745-746). The keepalive holds
split_prefill active so FixedPolicy's pin actually takes effect. For the d108
arm keepalive is off (decode legitimately gets all SMs alone).
"""
import argparse
import json
import os
import statistics
import sys
import threading
import time
import urllib.request


def _percentile(xs, q):
    if not xs:
        return float("nan")
    ys = sorted(xs)
    if len(ys) == 1:
        return ys[0]
    pos = (len(ys) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ys) - 1)
    frac = pos - lo
    return ys[lo] * (1 - frac) + ys[hi] * frac


def stream_one(url, body, rid, sink, stop_flag=None):
    """POST a streaming /generate; append (rid, token_idx, wall_ts) to sink."""
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    t_send = time.perf_counter()
    tok = 0
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            for raw in resp:
                if stop_flag is not None and stop_flag.is_set():
                    break
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:"):].strip()
                if payload == "[DONE]":
                    break
                ts = time.perf_counter()
                sink.append((rid, tok, ts))
                tok += 1
    except Exception as exc:  # noqa: BLE001
        sink.append((rid, -1, f"ERROR:{exc}"))
    return t_send


def keepalive_loop(url, prompt_text, out_tokens, stop_flag):
    """Background trickle: keep at least one prefill in flight to hold the pin."""
    body = {
        "text": prompt_text,
        "sampling_params": {
            "max_new_tokens": out_tokens,
            "temperature": 0.7,
            "ignore_eos": True,
        },
        "stream": False,
    }
    data = json.dumps(body).encode("utf-8")
    while not stop_flag.is_set():
        try:
            req = urllib.request.Request(
                url, data=data, headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=600) as resp:
                resp.read()
        except Exception:  # noqa: BLE001
            time.sleep(0.05)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="base URL, e.g. http://127.0.0.1:34000")
    ap.add_argument("--prompt-file", required=True, help="text file with the long prompt")
    ap.add_argument("--conc", type=int, default=32)
    ap.add_argument("--out-tokens", type=int, default=32)
    ap.add_argument("--raw-out", required=True, help="JSONL path for raw per-token timestamps")
    ap.add_argument("--tag", default="")
    ap.add_argument("--keepalive-prefill", type=int, default=0,
                    help="N background keepalive prefill workers (0 = off)")
    ap.add_argument("--keepalive-reps", type=int, default=256,
                    help="phrase reps for keepalive prompt (~8 tok/rep)")
    args = ap.parse_args()

    gen_url = args.url.rstrip("/") + "/generate"
    with open(args.prompt_file) as f:
        prompt = f.read()

    body_tmpl = {
        "text": prompt,
        "sampling_params": {
            "max_new_tokens": args.out_tokens,
            "temperature": 0.7,
            "ignore_eos": True,
        },
        "stream": True,
    }

    stop_flag = threading.Event()
    ka_threads = []
    if args.keepalive_prefill > 0:
        ka_prompt = "the robot walked slowly across the red planet " * args.keepalive_reps
        for _ in range(args.keepalive_prefill):
            t = threading.Thread(
                target=keepalive_loop,
                args=(gen_url, ka_prompt, args.out_tokens, stop_flag),
                daemon=True,
            )
            t.start()
            ka_threads.append(t)
        # let the keepalive establish a prefill backlog before measuring
        time.sleep(1.0)

    sink = []
    threads = []
    t0 = time.perf_counter()
    for r in range(args.conc):
        th = threading.Thread(
            target=stream_one, args=(gen_url, dict(body_tmpl), r, sink), daemon=True
        )
        th.start()
        threads.append(th)
    for th in threads:
        th.join()
    wall = time.perf_counter() - t0

    stop_flag.set()

    # organize per-request token timestamps
    per_req = {}
    errors = []
    for rid, tok, ts in sink:
        if tok == -1:
            errors.append((rid, ts))
            continue
        per_req.setdefault(rid, []).append((tok, ts))

    itls_ms = []
    ttfts_ms = []
    with open(args.raw_out, "w") as f:
        for rid, toks in sorted(per_req.items()):
            toks.sort()
            times = [ts for _, ts in toks]
            f.write(json.dumps({"tag": args.tag, "rid": rid,
                                "token_ts": times}) + "\n")
            # gaps: first gap is TTFT-ish (first streamed chunk timing), drop for ITL
            for i in range(1, len(times)):
                itls_ms.append((times[i] - times[i - 1]) * 1000.0)

    n_tok = sum(len(v) for v in per_req.values())
    summary = {
        "tag": args.tag,
        "conc": args.conc,
        "out_tokens": args.out_tokens,
        "keepalive_prefill": args.keepalive_prefill,
        "requests_ok": len(per_req),
        "requests_err": len(errors),
        "tokens_total": n_tok,
        "itl_samples": len(itls_ms),
        "itl_ms_p50": _percentile(itls_ms, 0.50),
        "itl_ms_p95": _percentile(itls_ms, 0.95),
        "itl_ms_p99": _percentile(itls_ms, 0.99),
        "itl_ms_mean": statistics.fmean(itls_ms) if itls_ms else float("nan"),
        "wall_s": wall,
    }
    print("STAGE0_SUMMARY " + json.dumps(summary))
    if errors:
        print("STAGE0_ERRORS " + json.dumps([str(e[1]) for e in errors[:5]]),
              file=sys.stderr)
    # non-zero exit if nothing decoded (smoke correctness signal)
    return 0 if n_tok > 0 and not errors else 1


if __name__ == "__main__":
    sys.exit(main())
