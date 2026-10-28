#!/usr/bin/env python3
"""Trace generation for the policy PREREG (rev0.1): W1 = ShareGPT, W3 = arXiv-Sum 16k natural output (decision 47).

A trace is a list of SEGMENTS; the runner replays each segment separately (segment arrival times start at 0), exactly like
KISTI sharegpt_vary_bench.sbatch runs one bench_serving per LO/HI segment -- the gap between segments is the drain.
  * HE0 cell : 3 rounds x (LO segment of NP requests at rate_lo, HI segment of NP requests at rate_hi)   (decision 44)
  * H-D clean: two segments (W1, W3 in `start` order), each `phase_len_s` of Poisson arrivals, drain between (1 cycle, decision 48)
  * H-D cont : ONE segment, phases concatenated (carry-over), request.phase marks the arrival phase
  * CAP cell : one segment of n_requests at one rate
Same (seed, workload, rate, variant) -> byte-identical requests in every arm (trace hash recorded).  CPU only, stdlib only.
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path
import os

HERE = Path(os.path.abspath(__file__)).parent
ENG = HERE.parents[2]
W1_CTX = 4000                                   # --sharegpt-context-len 4000 (KISTI)
W1_MIN_TOKENS = 4


def build_sharegpt_lengths(json_path, count_fn, ctx=W1_CTX):
    """First human turn -> first gpt reply of every conversation, as [n_in, n_out]; bench_serving-style filter
    (both >= 4 tokens, n_in + n_out <= ctx).  count_fn(text) -> token count (Job: tokenizers.Tokenizer.from_file)."""
    data = json.load(open(json_path, encoding="utf-8"))
    out = []
    for conv in data:
        turns = conv.get("conversations") or []
        if len(turns) < 2:
            continue
        a, b = turns[0], turns[1]
        if a.get("from") not in ("human", "user") or b.get("from") not in ("gpt", "assistant"):
            continue
        ni, no = int(count_fn(a.get("value", ""))), int(count_fn(b.get("value", "")))
        if ni >= W1_MIN_TOKENS and no >= W1_MIN_TOKENS and ni + no <= ctx:
            out.append([ni, no])
    return out


def load_w3_pool(token_file):
    """arXiv-Sum 16k pre-tokenized file (scripts/multigpu/mg_x_pretokenize.py, VESSL /data/datasets/arxsum_x/): [doc_id, n_in, n_out]."""
    sys.path.insert(0, str(ENG / "scripts" / "multigpu"))
    import mg_x_pretokenize as X
    h = X.read_header(token_file)
    return [[int(d), int(i), int(o)] for d, _off, i, o in h["docs"]], h


def attach_w3_ids(requests, token_file, header=None):
    import mg_x_pretokenize as X
    h = header or X.read_header(token_file)
    ids = X.doc_input_ids(token_file, h, sorted({r["doc_id"] for r in requests if "doc_id" in r}))
    for r in requests:
        if "doc_id" in r:
            r["input_ids"] = ids[r["doc_id"]]
            assert len(r["input_ids"]) == r["input_tokens"]
    return requests


def _rng(*key):
    return random.Random(int(hashlib.sha256("|".join(map(str, key)).encode()).hexdigest()[:12], 16))


def arrivals(rate, seed_key, n=None, duration=None):
    """Poisson arrival times (s).  Exactly one of n / duration."""
    if (n is None) == (duration is None):
        raise ValueError("exactly one of n / duration")
    if rate <= 0:
        raise ValueError("rate must be > 0")
    r = _rng("arr", *seed_key)
    t, out = 0.0, []
    while True:
        t += r.expovariate(rate)
        if n is not None:
            out.append(t)
            if len(out) >= n:
                return out
        else:
            if t >= duration:
                return out
            out.append(t)


def _pick(pool, count, seed_key):
    r = _rng("pick", *seed_key)
    if count <= len(pool):
        return r.sample(range(len(pool)), count)
    return [r.randrange(len(pool)) for _ in range(count)]


def make_requests(workload, pool, ts, seed_key, prefix, phase, t_offset=0.0):
    """One request per arrival time.  W1: lengths from ShareGPT pool, ignore_eos True (fixed output => identical work
    across arms; deviation from bench_serving's natural EOS, disclosed).  W3: arXiv-Sum doc, natural output
    (ignore_eos False, max_new_tokens cap) -- decision 47."""
    from ph_spec import W3_MAX_NEW
    idx = _pick(pool, len(ts), seed_key + (workload,))
    reqs = []
    for k, (t, i) in enumerate(zip(ts, idx)):
        if workload == "W1":
            ni, no = pool[i]
            reqs.append({"record": "request", "request_id": f"{prefix}-{k:05d}", "phase": phase,
                         "arrival_s": round(t + t_offset, 6), "input_tokens": ni, "output_tokens": no, "ignore_eos": True})
        elif workload == "W3":
            did, ni, _no = pool[i]
            reqs.append({"record": "request", "request_id": f"{prefix}-{k:05d}", "phase": phase,
                         "arrival_s": round(t + t_offset, 6), "input_tokens": ni, "output_tokens": W3_MAX_NEW,
                         "ignore_eos": False, "doc_id": did})
        else:
            raise ValueError(workload)
    return reqs


def trace_hash(segments):
    h = hashlib.sha256()
    for s in segments:
        h.update(json.dumps({k: v for k, v in s.items() if k != "requests"}, sort_keys=True).encode())
        for r in s["requests"]:
            h.update(json.dumps({k: v for k, v in r.items() if k != "input_ids"}, sort_keys=True).encode())
    return h.hexdigest()


def cap_trace(workload, pool, rate, seed, n_requests, tag):
    sk = (tag, seed, rate)
    ts = arrivals(rate, sk, n=n_requests)
    return [{"segment_id": f"{tag}", "phase": workload, "round": 0,
             "requests": make_requests(workload, pool, ts, sk, tag, workload)}]


def he0_trace(w1_pool, rate_lo, rate_hi, np_, rounds, seed, tag):
    """LO then HI segment per round, KISTI order.  Separate seeds per segment, same across arms."""
    segs = []
    for rd in range(1, rounds + 1):
        for ph, rate in (("LO", rate_lo), ("HI", rate_hi)):
            sk = ("he0", seed, rd, ph)
            ts = arrivals(rate, sk, n=np_)
            segs.append({"segment_id": f"{tag}-r{rd}{ph}", "phase": ph, "round": rd,
                         "requests": make_requests("W1", w1_pool, ts, sk, f"{tag}-r{rd}{ph}", ph)})
    return segs


def hd_trace(variant, phase_len_s, start_first, rate_w1, rate_w3, w1_pool, w3_pool, seed, tag):
    """One cycle (W1 block + W3 block).  clean: one segment per phase (drain between); cont: one segment."""
    if variant not in ("clean", "cont"):
        raise ValueError(variant)
    order = ("W1", "W3") if start_first == "W1" else ("W3", "W1")
    rate = {"W1": rate_w1, "W3": rate_w3}
    pool = {"W1": w1_pool, "W3": w3_pool}
    per_phase = {}
    for w in order:
        sk = ("hd", seed, w, phase_len_s)          # variant-independent: clean and cont see the same phase requests
        ts = arrivals(rate[w], sk, duration=phase_len_s)
        per_phase[w] = (ts, sk)
    if variant == "clean":
        return [{"segment_id": f"{tag}-{w}", "phase": w, "round": 0, "phase_len_s": phase_len_s,
                 "requests": make_requests(w, pool[w], per_phase[w][0], per_phase[w][1], f"{tag}-{w}", w)}
                for w in order]
    reqs = []
    for j, w in enumerate(order):
        reqs += make_requests(w, pool[w], per_phase[w][0], per_phase[w][1], f"{tag}-{w}", w, t_offset=j * phase_len_s)
    return [{"segment_id": f"{tag}-cont", "phase": "CONT", "round": 0, "phase_len_s": phase_len_s, "order": list(order),
             "requests": sorted(reqs, key=lambda r: r["arrival_s"])}]


def write_trace(segments, path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for s in segments:
            f.write(json.dumps({"record": "segment", **{k: v for k, v in s.items() if k != "requests"}}, sort_keys=True) + "\n")
            for r in s["requests"]:
                f.write(json.dumps({k: v for k, v in r.items() if k != "input_ids"}, sort_keys=True) + "\n")
    return trace_hash(segments)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build-w1"])
    ap.add_argument("--sharegpt", required=True)
    ap.add_argument("--tokenizer", required=True, help="tokenizer.json of the served model")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(a.tokenizer)
    pool = build_sharegpt_lengths(a.sharegpt, lambda t: len(tok.encode(t, add_special_tokens=False).ids))
    Path(a.out).write_text(json.dumps(pool))
    print(json.dumps({"n": len(pool), "sha256": hashlib.sha256(Path(a.out).read_bytes()).hexdigest()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
