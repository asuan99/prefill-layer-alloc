#!/usr/bin/env python3
"""Load driver for the R2 GPU correctness gate (legacy vs true dual-worker).

Three phases against one already-booted server, in this order:

  S (sequential)  N_S prompts sent ONE AT A TIME.  Batch size is 1 for every
                  forward.  EQUALITY GATE.  Note: with the decode batch empty
                  or prefill idle, the engine runs on the PLAIN stream groups
                  (prefill-only -> index 0, decode-only -> the last index), so
                  S never executes on the fixed D division.
  O (overlap)     N_O probes.  For each: start one long BACKGROUND request
                  (streamed), wait for its first token (it is now decoding),
                  then send the PROBE and wait for it.  The probe's prefill is
                  admitted alone while the background decodes, so the R2 fixed
                  policy switches to the D division and the probe prefill runs
                  on its prefill side concurrently with the background decode
                  -- on two host issue threads in true-dual mode.  EQUALITY
                  GATE on the probe token ids; background outputs are recorded
                  but excluded.  Determinism by construction:
                    * the probe prefill batch is the probe alone;
                    * every probe decode step runs at batch size 2 (background
                      + probe) because the background outlives the probe
                      (checked per probe: bg t_end > probe t_end);
                    * the triton decode backend chooses each row's KV-split
                      count from the batch's max/min sequence lengths
                      (layers/attention/triton_backend.py
                      get_num_kv_splits_triton).  With
                      probe_len > (K-1) * (bg_prompt + bg_max_new), K =
                      triton_attention_num_kv_splits (8), `max_kv_splits_1`
                      saturates at K and `max_seq_len` is the probe's own
                      length, so the probe row's split count depends only on
                      the probe, not on how far the background has got
                      (checked per probe by the scorer).
  C (concurrent)  N_C prompts at fixed staggered offsets.  Batch composition
                  and the partition each request lands on depend on arrival
                  timing (job 907032: legacy-vs-legacy 7/32 differ, one at
                  token 0).  DIAGNOSTIC ONLY -- never changes the verdict.

Prompts are generated deterministically from `--seed` (natural-language
passages, not random token ids).  S and C prompts are identical to job
907032's; O prompts come from a separate RNG stream.  Every request is greedy
(`temperature=0`, `ignore_eos`).  Timestamps: wall clock (`t_*`) and
`time.perf_counter()` (`pc_*`).  On Linux perf_counter is CLOCK_MONOTONIC,
system-wide, the same clock the server stamps telemetry with
(`timestamp_monotonic_s`), so the scorer can place probe windows on the
telemetry timeline.

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

# O tier.  The background prompt is short and its length bounded
# (bg_prompt + BG_MAX_NEW ~ 175 tokens), so every probe prompt must exceed
# 7 x 175 = 1225 tokens; the word counts below give ~1300-2600 tokens
# (~1.3 tokens/word for this tokenizer, measured on the S/C prompts).
BG_PROMPT = "Count upward in words, one number per line: one, two, three,"
BG_MAX_NEW = 160
PROBE_WORDS = [1100, 1200, 1300, 1450, 1600, 1750, 1900, 2000]
PROBE_MAX_NEW = 48


def build_prompt(rng, n_words):
    words = []
    while len(words) < n_words:
        words.extend(rng.choice(PASSAGES).split(" "))
    return " ".join(words[:n_words])


def plan(seed, n_s, n_c, max_new_s, n_o=0):
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
    # Separate stream: adding the O tier leaves every S/C prompt unchanged.
    orng = random.Random(seed + 1)
    over = []
    for i in range(n_o):
        over.append({
            "id": f"O{i:02d}",
            "text": build_prompt(orng, PROBE_WORDS[i % len(PROBE_WORDS)]),
            "max_new": PROBE_MAX_NEW,
        })
    for item in seq + conc + over:
        item["prompt_sha256"] = hashlib.sha256(item["text"].encode()).hexdigest()
    return seq, conc, over


def _body(text, max_new, stream=False):
    return json.dumps({
        "text": text,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": max_new,
            # Fixed output length: decode stays busy for a known number of
            # steps (phase C overlap; phase O background outliving the probe).
            "ignore_eos": True,
        },
        "stream": stream,
    }).encode()


def _finish_reason(meta):
    fr = meta.get("finish_reason")
    return fr if isinstance(fr, (str, type(None))) else json.dumps(fr)


def generate(port, item, timeout_s):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/generate", data=_body(item["text"], item["max_new"]),
        headers={"Content-Type": "application/json"},
    )
    rec = {"id": item["id"], "prompt_sha256": item["prompt_sha256"],
           "max_new": item["max_new"], "t_start": time.time(),
           "pc_start": time.perf_counter()}
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            out = json.load(resp)
        meta = out.get("meta_info", {}) or {}
        rec.update({
            "output_ids": list(out.get("output_ids") or []),
            "text": out.get("text", ""),
            "prompt_tokens": meta.get("prompt_tokens"),
            "completion_tokens": meta.get("completion_tokens"),
            "finish_reason": _finish_reason(meta),
            "error": None,
        })
    except (urllib.error.URLError, OSError, ValueError) as exc:
        rec.update({"output_ids": [], "text": "", "prompt_tokens": None,
                    "completion_tokens": None, "finish_reason": None,
                    "error": f"{type(exc).__name__}: {exc}"})
    rec["t_end"], rec["pc_end"] = time.time(), time.perf_counter()
    return rec


def generate_streamed(port, text, max_new, timeout_s, first_token, rec):
    """Streamed background request, filling `rec` IN PLACE (the probe side
    reads `pc_first`/`error` while this is still streaming).  Sets
    `first_token` on the first chunk that carries output ids (the request is
    then in the decode batch), or on failure."""
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/generate", data=_body(text, max_new, stream=True),
        headers={"Content-Type": "application/json"},
    )
    rec.update({"max_new": max_new, "pc_start": time.perf_counter(), "pc_first": None,
                "output_ids": [], "prompt_tokens": None, "error": None})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            for raw in resp:
                line = raw.strip()
                if not line.startswith(b"data:"):
                    continue
                payload = line[5:].strip()
                if payload == b"[DONE]":
                    break
                chunk = json.loads(payload)
                if "error" in chunk:  # http_server.stream_results error chunk
                    rec["error"] = f"stream error: {chunk['error']}"
                    break
                ids = chunk.get("output_ids") or []
                if ids and rec["pc_first"] is None:
                    rec["pc_first"] = time.perf_counter()
                    first_token.set()
                rec["output_ids"] = list(ids)   # cumulative (non-incremental mode)
                meta = chunk.get("meta_info", {}) or {}
                rec["prompt_tokens"] = meta.get("prompt_tokens", rec["prompt_tokens"])
                rec["finish_reason"] = _finish_reason(meta)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        rec["error"] = f"{type(exc).__name__}: {exc}"
    rec["pc_end"] = time.perf_counter()
    first_token.set()  # never leave the probe side waiting on a dead request
    return rec


def run_overlap_probes(port, probes, timeout_s):
    out = []
    for item in probes:
        first_token = threading.Event()
        bg = {}
        thread = threading.Thread(
            target=generate_streamed,
            args=(port, BG_PROMPT, BG_MAX_NEW, timeout_s, first_token, bg),
            daemon=True,
        )
        thread.start()
        first_token.wait(timeout_s)
        if bg.get("error") or bg.get("pc_first") is None:
            probe = {"id": item["id"], "error": "background did not start decoding",
                     "output_ids": [], "prompt_sha256": item["prompt_sha256"]}
        else:
            probe = generate(port, item, timeout_s)
        thread.join(timeout_s + 30)
        out.append({"id": item["id"], "probe": probe, "bg": dict(bg)})
        print(f"  O {item['id']} pt={probe.get('prompt_tokens')} "
              f"out={len(probe.get('output_ids') or [])} bg_out={len(bg.get('output_ids') or [])} "
              f"bg_outlived={bg.get('pc_end', 0) > probe.get('pc_end', float('inf'))} "
              f"{'ERR ' + str(probe.get('error') or bg.get('error')) if (probe.get('error') or bg.get('error')) else ''}",
              flush=True)
        time.sleep(0.5)
    return out


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
    ap.add_argument("--n-probes", type=int, default=8)
    ap.add_argument("--n-conc", type=int, default=32)
    ap.add_argument("--max-new-seq", type=int, default=64)
    ap.add_argument("--stagger-ms", type=float, default=80.0)
    ap.add_argument("--timeout-s", type=float, default=240.0)
    args = ap.parse_args()

    seq, conc, over = plan(args.seed, args.n_seq, args.n_conc, args.max_new_seq, args.n_probes)
    doc = {"boot": args.boot, "seed": args.seed, "stagger_ms": args.stagger_ms,
           "bg_prompt": BG_PROMPT, "bg_max_new": BG_MAX_NEW,
           "probe_max_new": PROBE_MAX_NEW}

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

    time.sleep(1.0)
    doc["phase_o_window"] = [time.time(), None]
    doc["phase_o"] = run_overlap_probes(args.port, over, args.timeout_s)
    doc["phase_o_window"][1] = time.time()

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
