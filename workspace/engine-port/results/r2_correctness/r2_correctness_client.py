#!/usr/bin/env python3
"""Load driver for the R2 GPU correctness gate (legacy vs true dual-worker).

Two phases against one already-booted server:

  S (sequential)  N_S prompts sent ONE AT A TIME.  Batch size is 1 for every
                  forward, so batch-shape floating-point nondeterminism is
                  excluded by construction.  Exact token equality is the gate.
  C (concurrent)  N_C prompts launched at fixed staggered offsets so later
                  prefills run while earlier requests decode -- this is the
                  only phase in which the true dual-worker's two host issue
                  threads have work at the same time.  Batch composition then
                  depends on timing, so equality is judged against a
                  same-arm replicate boot (see r2_correctness_check.py).

Prompts are generated deterministically from `--seed` (natural-language
passages, not random token ids: random ids give flat next-token
distributions and therefore spurious near-tie argmax flips).  Every request
is greedy (`temperature=0`).  The same prompt list, lengths and offsets are
produced for every boot; the checker verifies that through the sha256 of
each prompt and the server-reported `prompt_tokens`.

Output: one JSON file per boot (committable -- `*.json` is not ignored under
results/).
"""

import argparse
import hashlib
import json
import random
import threading
import time
import urllib.error
import urllib.request

PASSAGES = [
    "The committee reviewed the quarterly report and noted that shipping "
    "delays had eased after the new warehouse opened near the river port.",
    "In the old library, the archivist catalogued letters written by a "
    "lighthouse keeper who recorded every storm for forty years.",
    "Photosynthesis converts light energy into chemical energy, storing it "
    "in the bonds of sugar molecules that the plant later uses to grow.",
    "The engineers replaced the worn bearings, recalibrated the sensors, and "
    "ran the turbine at half load for six hours before signing off.",
    "A small bakery on the corner sells rye bread in the morning and closes "
    "early on Sundays so the owners can visit their family in the hills.",
    "def moving_average(values, window):\n    out = []\n    for i in "
    "range(len(values) - window + 1):\n        out.append(sum(values[i:i + "
    "window]) / window)\n    return out\n",
    "The river rises each spring when snow melts in the mountains, and the "
    "farmers along its banks plant only after the water begins to recede.",
    "Local elections drew a larger turnout than expected, which analysts "
    "attributed to a close race for the county water board.",
    "The recipe calls for two cups of flour, one egg, a pinch of salt, and "
    "enough milk to make a smooth batter that coats the back of a spoon.",
    "Astronomers measured the star's brightness over many nights and found "
    "a regular dip that suggested a planet crossing in front of it.",
    "The train left the station at dawn, crossed the long iron bridge, and "
    "reached the coast just as the fishing boats were returning.",
    "Volunteers repaired the trail after the storm, clearing fallen branches "
    "and rebuilding the wooden steps that had washed away.",
]

SHORT_PROMPTS = [
    "The capital of France is",
    "1, 2, 3, 4, 5, 6, 7, 8,",
    "def add(a, b):\n    return",
    "The chemical symbol for gold is",
]


def build_prompt(rng, n_words):
    words = []
    while len(words) < n_words:
        words.extend(rng.choice(PASSAGES).split(" "))
    return " ".join(words[:n_words])


def plan(seed, n_s, n_c, max_new_s):
    rng = random.Random(seed)
    # Word budgets keep prompt + output < 4096 tokens for Zamba2-2.7B.
    s_words = [0, 0, 0, 0, 24, 64, 160, 320, 640, 960, 1280, 1600, 1900, 48, 400, 1100]
    c_words = [48, 320, 1280, 96, 640, 1900, 160, 960]
    c_new = [96, 48, 160, 64, 128, 32, 144, 80]
    seq = []
    for i in range(n_s):
        words = s_words[i % len(s_words)]
        text = SHORT_PROMPTS[i % len(SHORT_PROMPTS)] if words == 0 else build_prompt(rng, words)
        seq.append({"id": f"S{i:02d}", "text": text, "max_new": max_new_s})
    conc = []
    for i in range(n_c):
        text = build_prompt(rng, c_words[i % len(c_words)])
        conc.append({"id": f"C{i:02d}", "text": text, "max_new": c_new[i % len(c_new)]})
    for item in seq + conc:
        item["prompt_sha256"] = hashlib.sha256(item["text"].encode()).hexdigest()
    return seq, conc


def generate(port, item, timeout_s):
    body = json.dumps({
        "text": item["text"],
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": item["max_new"],
            # Fixed output length: decode stays busy for a known number of
            # steps, so phase C reliably overlaps later prefills.
            "ignore_eos": True,
        },
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/generate", data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    rec = {"id": item["id"], "prompt_sha256": item["prompt_sha256"],
           "max_new": item["max_new"], "t_start": t0}
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            out = json.load(resp)
        meta = out.get("meta_info", {}) or {}
        fr = meta.get("finish_reason")
        rec.update({
            "output_ids": list(out.get("output_ids") or []),
            "text": out.get("text", ""),
            "prompt_tokens": meta.get("prompt_tokens"),
            "completion_tokens": meta.get("completion_tokens"),
            "finish_reason": fr if isinstance(fr, (str, type(None))) else json.dumps(fr),
            "error": None,
        })
    except (urllib.error.URLError, OSError, ValueError) as exc:
        rec.update({"output_ids": [], "text": "", "prompt_tokens": None,
                    "completion_tokens": None, "finish_reason": None,
                    "error": f"{type(exc).__name__}: {exc}"})
    rec["t_end"] = time.time()
    return rec


def run_concurrent(port, items, stagger_ms, timeout_s):
    results = [None] * len(items)
    t0 = time.time() + 0.5

    def worker(index, item):
        delay = t0 + index * stagger_ms / 1000.0 - time.time()
        if delay > 0:
            time.sleep(delay)
        results[index] = generate(port, item, timeout_s)

    threads = [threading.Thread(target=worker, args=(i, it), daemon=True)
               for i, it in enumerate(items)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout_s + 30)
    return [r if r is not None else {"id": items[i]["id"], "error": "thread_timeout",
                                     "output_ids": []}
            for i, r in enumerate(results)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--boot", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20260911)
    ap.add_argument("--n-seq", type=int, default=16)
    ap.add_argument("--n-conc", type=int, default=32)
    ap.add_argument("--max-new-seq", type=int, default=64)
    ap.add_argument("--stagger-ms", type=float, default=80.0)
    ap.add_argument("--timeout-s", type=float, default=240.0)
    args = ap.parse_args()

    seq, conc = plan(args.seed, args.n_seq, args.n_conc, args.max_new_seq)
    doc = {"boot": args.boot, "seed": args.seed, "stagger_ms": args.stagger_ms}

    doc["phase_s_window"] = [time.time(), None]
    doc["phase_s"] = []
    for item in seq:
        rec = generate(args.port, item, args.timeout_s)
        doc["phase_s"].append(rec)
        print(f"  S {rec['id']} pt={rec.get('prompt_tokens')} "
              f"out={len(rec['output_ids'])} "
              f"{'ERR ' + rec['error'] if rec['error'] else repr(rec['text'][:40])}",
              flush=True)
    doc["phase_s_window"][1] = time.time()

    time.sleep(2.0)
    doc["phase_c_window"] = [time.time(), None]
    doc["phase_c"] = run_concurrent(args.port, conc, args.stagger_ms, args.timeout_s)
    doc["phase_c_window"][1] = time.time()
    n_err = sum(1 for r in doc["phase_c"] if r.get("error"))
    print(f"  C done: {len(doc['phase_c'])} requests, {n_err} errors", flush=True)

    with open(args.out, "w") as handle:
        json.dump(doc, handle)


if __name__ == "__main__":
    main()
