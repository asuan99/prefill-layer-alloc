#!/usr/bin/env python3
"""AF-1: measure the UNCONTENDED latency of one arm at one boot.  Concurrency 1.

What it measures, and nothing else
----------------------------------
For each registered stratum, the prompt drawn for this boot is issued
`REPEATS_PER_STRATUM` times, ONE AT A TIME, with nothing else in flight:

  * `ttft_ms`      -- time to the first streamed token.
  * `ritl_p95_ms`  -- the p95 of THAT REQUEST'S OWN token ITLs.  This is the ITL
                      component of the canonical goodput predicate
                      (`CLAUDE.md`, "goodput = request TTFT <= SLO AND
                      request-내부 token-ITL p95 <= SLO"), not a pass rate.

★Chunk-vs-token: a streamed chunk may carry more than one token.  When
`meta_info.completion_tokens` advances by k > 1 over one chunk, the elapsed time
is divided into k equal ITLs rather than recorded as a single long one -- the
registered quantity is a TOKEN ITL, and treating a k-token chunk as one interval
would inflate the p95 by a factor that depends on the server's batching of the
stream, which is not the thing being measured.

★Warm-up is ISSUED, never discarded.  `WARMUP_REQUESTS_PER_STRATUM` UNMEASURED
requests precede the measured repeats at each stratum; no measured repeat is ever
dropped.  The plumbing smoke (job 904819) showed why: the first request at a
stratum paid a one-off cost of 3.6x at one stratum and 16.5x at another, and
because `agg_within_boot` is a near-max order statistic that first request BECAME
the boot value.  Warming up per STRATUM rather than once per boot is what the
smoke showed is needed -- `cp2048` spiked at q=0.50 and q=0.99, not only on its
first request, so the cost follows the prompt SHAPE, not the boot.
⚠️Discarding a measured repeat would be a different act -- it moves the estimand
the liberal way and would have to be registered as such.  This does not do that.

Sampling is fixed by the pre-registration: temperature 0.0, ignore_eos True,
max_new_tokens 64 (`af1_predicates.TEMPERATURE` / `IGNORE_EOS` / `MAX_NEW_TOKENS`).

Usage: python3 af1_client.py --port 40000 --arm d44 --seed 11 \
                            --prompts prompts_seed11.json --out boot_d44.json
"""
import argparse, json, sys, time, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import af1_predicates as P


def order_stat(xs, q):
    """The same linear-interpolation order statistic the predicates register."""
    return P.order_stat(xs, q)


def one_request(port, prompt, timeout_s):
    """Issue one streaming request and return (ttft_ms, [token_itl_ms], n_tok)."""
    body = json.dumps({
        "text": prompt,
        "sampling_params": {
            "temperature": P.TEMPERATURE,
            "max_new_tokens": P.MAX_NEW_TOKENS,
            "ignore_eos": P.IGNORE_EOS,
        },
        "stream": True,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/generate", data=body,
        headers={"Content-Type": "application/json"})
    ttft = None
    itls = []
    last_t = None
    last_n = 0
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        for raw in r:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            data = json.loads(payload)
            n = data.get("meta_info", {}).get("completion_tokens")
            if n is None or n <= last_n:
                continue                      # keep-alive / usage-only chunk
            now = time.perf_counter()
            if ttft is None:
                ttft = (now - t0) * 1000.0
                k = n - last_n
                if k > 1:                     # first chunk already多 tokens
                    itls.extend([(now - t0) * 1000.0 / k] * (k - 1))
            else:
                k = n - last_n
                itls.extend([(now - last_t) * 1000.0 / k] * k)
            last_t, last_n = now, n
    return ttft, itls, last_n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--node", default="")
    a = ap.parse_args()
    if a.arm not in P.ARMS:
        sys.exit("arm %r is not registered in ARMS %s" % (a.arm, P.ARMS))

    spec = json.load(open(a.prompts))
    if spec["seed"] != a.seed:
        sys.exit("prompt file is for seed %s, not %s" % (spec["seed"], a.seed))

    out = {"arm": a.arm, "seed": a.seed, "node": a.node,
           "sampling": {"temperature": P.TEMPERATURE, "ignore_eos": P.IGNORE_EOS,
                        "max_new_tokens": P.MAX_NEW_TOKENS},
           "repeats": P.REPEATS_PER_STRATUM, "strata": []}
    for st in spec["strata"]:
        rec = {"q": st["q"], "prompt_tokens": st["prompt_tokens"],
               "target_tokens": st["target_tokens"],
               "n_warmup": P.WARMUP_REQUESTS_PER_STRATUM, "warmups_ms": [],
               "repeats": []}
        for w in range(P.WARMUP_REQUESTS_PER_STRATUM):
            try:
                wt, _u, wn = one_request(a.port, st["prompt"], P.REQ_TIMEOUT_S)
                rec["warmups_ms"].append(wt)
                print(f"  {a.arm:8s} q={st['q']:<5} warm{w} "
                      f"ttft={None if wt is None else round(wt,1)} ms ntok={wn} (not scored)",
                      flush=True)
            except Exception as e:
                rec["warmups_ms"].append(None)
                print(f"  {a.arm:8s} q={st['q']:<5} warm{w} ERR {type(e).__name__}", flush=True)
        for i in range(P.REPEATS_PER_STRATUM):
            try:
                ttft, itls, ntok = one_request(a.port, st["prompt"], P.REQ_TIMEOUT_S)
                r = {"ttft_ms": ttft, "n_tokens": ntok, "n_itl": len(itls),
                     "ritl_p95_ms": order_stat(itls, 0.95) if itls else None,
                     "ritl_p50_ms": order_stat(itls, 0.50) if itls else None,
                     "error": None}
            except Exception as e:                       # boot alive but request failed
                r = {"ttft_ms": None, "n_tokens": 0, "n_itl": 0, "ritl_p95_ms": None,
                     "ritl_p50_ms": None, "error": f"{type(e).__name__}: {e}"}
            rec["repeats"].append(r)
            print(f"  {a.arm:8s} q={st['q']:<5} rep{i}  "
                  f"ttft={r['ttft_ms'] if r['ttft_ms'] is None else round(r['ttft_ms'],1)} ms  "
                  f"ritl_p95={r['ritl_p95_ms'] if r['ritl_p95_ms'] is None else round(r['ritl_p95_ms'],2)} ms  "
                  f"ntok={r['n_tokens']}{'  ERR ' + r['error'] if r['error'] else ''}",
                  flush=True)
        # the registered within-boot aggregate (sec 2.3): an order statistic over
        # the repeats, computed by the predicate module so it cannot drift.
        good_t = [r["ttft_ms"] for r in rec["repeats"] if r["ttft_ms"] is not None]
        good_i = [r["ritl_p95_ms"] for r in rec["repeats"] if r["ritl_p95_ms"] is not None]
        rec["boot_ttft_ms"] = P.agg_within_boot(good_t) if good_t else None
        rec["boot_ritl_p95_ms"] = P.agg_within_boot(good_i) if good_i else None
        out["strata"].append(rec)
    json.dump(out, open(a.out, "w"))
    print("wrote", a.out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
