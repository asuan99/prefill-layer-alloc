#!/usr/bin/env python3
"""Deterministic synthetic workloads for the P1 acceptance tests (CP-0).

PREREG_CP0_2026-08-28.md section 2.2 registers seven tests.  Four of them need a
workload whose chunking behaviour is known in advance:

  phase S  sequential, prompt lengths [1500, 512, 513, 100] at cps 512.
           Every number is hand-derivable because the batch budget is reset for
           each request (p1e_expectations.json phase S).
  phase B  eight concurrent 200-token prompts at cps 512.  Every prompt is
           SHORTER than cps, so the length-based candidate definition scores 0
           here while the real one does not.
  phase D  sixteen concurrent ~341-token prompts at cps 4096 -- the batch
           boundary positive control (P1-d).  341 is the mean of the served
           ShareGPT population (served_population.json prompt_len.mean = 341.4).
  greedy   a fixed batch=1 greedy set, run once with the probe ON and once with
           it OFF, for the bit-identity check (P1-c).

WHY input_ids AND NOT text
    The registered numbers are exact token counts.  Tokenising text would make
    them depend on the tokenizer, and PREREG_CP0_2026-08-28.md section 10 row
    L12 already records one measurement script drifting from the harness that
    way ("측정 스크립트의 토크나이저 경로가 하네스와 다름", `get_tokenizer` vs
    `AutoTokenizer`).
    The ids are generated from a fixed seed per (phase, index), so the workload
    is reproducible from this file alone.  The script also records the length
    the ENGINE reports (`meta_info.prompt_tokens`) and flags any disagreement,
    so a hidden BOS token cannot silently shift the expectations.

NO PACER, ON PURPOSE
    It is tempting to keep the server busy so the burst phases build a backlog.
    That would be a defect: any request the pacer got chunked would count into
    `n_requests_chunked`, and P1-d's predicate is `> 0`.  The control would then
    pass for a reason that has nothing to do with the burst.  The bursts are
    therefore pure, all threads released from one barrier, and if no backlog
    forms the correct answer is "re-run with a larger burst", which is what the
    checker's failure note says.

PHASE WINDOWS MUST NOT OVERLAP (repair 2, 2026-09-01)
    The roll-up bins telemetry into these windows by `host_ts`, so two windows
    that overlap make every per-phase number ambiguous -- each event in the
    overlap is counted twice.  Job 899768 registered `t_end + 1.0` for each
    phase, which reached one second into the next phase, and phase S's
    pre-registered "7 prefill forwards / 2625 tokens" was scored against 43
    forwards / 12937 tokens.  See `close_segments`.

This script measures nothing about performance and prints no timing summary
(PREREG section 9).
"""

import argparse
import json
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SCHEMA = "cp0.p1-workload/v1"

GREEDY_PROMPTS = [
    "The capital of France is",
    "Write one sentence about the sea.",
    "List three prime numbers:",
    "2 + 2 =",
]


def make_ids(phase, index, n, vocab_lo=1000, vocab_hi=20000):
    rng = random.Random(f"cp0-p1:{phase}:{index}:{n}")
    return [rng.randrange(vocab_lo, vocab_hi) for _ in range(n)]


def post(url, payload, timeout=600):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def gen_ids(port, ids, max_new_tokens, rid, timeout=600):
    payload = {
        "input_ids": ids,
        "rid": rid,
        "sampling_params": {
            "max_new_tokens": max_new_tokens,
            "temperature": 0,
            "ignore_eos": True,
        },
    }
    return post(f"http://127.0.0.1:{port}/generate", payload, timeout=timeout)


def gen_text_greedy(port, text, max_new_tokens, rid, timeout=600):
    payload = {
        "text": text,
        "rid": rid,
        "return_logprob": True,
        "sampling_params": {
            "max_new_tokens": max_new_tokens,
            "temperature": 0,
            "ignore_eos": False,
        },
    }
    return post(f"http://127.0.0.1:{port}/generate", payload, timeout=timeout)


def _served_len(resp, requested):
    mi = resp.get("meta_info") or {}
    return mi.get("prompt_tokens", requested)


def run_sequential(port, lens, phase_id, max_new_tokens):
    rows = []
    for i, n in enumerate(lens):
        ids = make_ids(phase_id, i, n)
        rid = f"{phase_id}-{i}"
        t0 = time.time()
        resp = gen_ids(port, ids, max_new_tokens, rid)
        rows.append({
            "rid": rid, "requested_len": n, "served_len": _served_len(resp, n),
            "t_start": t0, "t_end": time.time(),
        })
    return rows


def run_burst(port, lens, phase_id, max_new_tokens):
    barrier = threading.Barrier(len(lens))
    rows = [None] * len(lens)

    def one(i):
        n = lens[i]
        ids = make_ids(phase_id, i, n)
        rid = f"{phase_id}-{i}"
        barrier.wait()  # release every request at the same instant
        t0 = time.time()
        resp = gen_ids(port, ids, max_new_tokens, rid)
        rows[i] = {
            "rid": rid, "requested_len": n, "served_len": _served_len(resp, n),
            "t_start": t0, "t_end": time.time(),
        }

    with ThreadPoolExecutor(max_workers=len(lens)) as ex:
        list(ex.map(one, range(len(lens))))
    return rows


def summarize(rows, cps, phase_id, note=""):
    served = [r["served_len"] for r in rows]
    requested = [r["requested_len"] for r in rows]
    return {
        "phase": phase_id,
        "chunked_prefill_size": cps,
        "n_prompts": len(rows),
        "requested_lens": requested,
        "served_lens": served,
        "length_mismatch": served != requested,
        # candidate definition (3): the OFFLINE lower bound, not channel 1.
        # `cps <= 0` means the engine disabled chunking
        # (`managers/scheduler.py:903-904`.  Until 2026-09-01 this cited lines
        # 890-891 of the same file, which are the multimodal/Transformers guard
        # in that function and not the `<= 0` branch; the stale numbers are
        # written without backticks here so the citation checker does not read
        # them as a live citation),
        # so there is no cap to be longer than and the field is None -- NOT 0,
        # which would read as "no prompt exceeded the cap".
        "n_prompts_longer_than_cps": (
            sum(1 for n in served if n > cps) if (cps and cps > 0) else None
        ),
        "total_prompt_tokens": sum(served),
        "t_start": min(r["t_start"] for r in rows),
        "t_end": max(r["t_end"] for r in rows),
        "note": note,
        "requests": rows,
    }


# How much wall clock to keep after a phase's last response, so that a forward
# or a summary the engine emits just after the harness returns is still inside
# the window.  Never allowed to reach the next phase; see close_segments.
SEGMENT_GRACE_S = 1.0


def close_segments(segments, grace=SEGMENT_GRACE_S):
    """Give each window its grace, then CLAMP it to the next window's start.

    ★2026-09-01 (repair 2).  Every mode used to register `t_end + 1.0` with no
    clamp.  Phase S is awaited to completion and phase B starts about 1.5 ms
    later, so the grace pushed the S window a full second INTO phase B, and
    `analyze_chunk_probe.py` -- which bins by `host_ts` -- counted phase B's
    first forwards and chunk events in BOTH phases.  That is where job 899768's
    "phase S: n_prefill_forwards expected 7, got 43 | sum_extend_tokens expected
    2625, got 12937" came from.

    The registered expectations in `p1e_expectations.json` were never wrong:
    they are hand-derived per phase, for windows that do not overlap.  What was
    wrong is the instrument, so the repair is here and the expectations file is
    untouched.

    The windows are half-open (`t_start <= ts < t_end` in the roll-up), so
    ending one window exactly at the next window's start is disjoint and
    loses nothing.  What was clamped is recorded (`raw_t_end`,
    `clamped_to_next_segment`) rather than quietly applied.
    """
    out = [dict(s) for s in segments]
    order = sorted(range(len(out)), key=lambda i: float(out[i]["t_start"]))
    for pos, i in enumerate(order):
        raw_end = float(out[i]["t_end"])
        end = raw_end + grace
        limit = None
        if pos + 1 < len(order):
            limit = float(out[order[pos + 1]]["t_start"])
            end = min(end, limit)
        # A window may not be inverted even if the PHASES really did overlap in
        # wall clock (they do not in the registered workloads, but a retry with
        # --burst-n could).  A genuine overlap is the roll-up's business to
        # report; this function's job is to not manufacture a negative window.
        end = max(end, float(out[i]["t_start"]))
        out[i]["raw_t_end"] = raw_end
        out[i]["t_end"] = end
        out[i]["grace_s"] = grace
        out[i]["clamped_to_next_segment"] = bool(
            limit is not None and end < raw_end + grace
        )
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--mode", required=True,
                    choices=["p1e", "d", "greedy", "mono"])
    ap.add_argument("--cps", type=int, required=True,
                    help="the boot's chunked_prefill_size, used only to compute "
                         "the offline lower bound (definition 3)")
    ap.add_argument("--expectations", default=str(
        Path(__file__).with_name("p1e_expectations.json")))
    ap.add_argument("--max-new-tokens", type=int, default=8)
    ap.add_argument("--burst-n", type=int, default=None,
                    help="override the registered burst size (retry path only; "
                         "record the override in the artifact)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--out-segments", default=None)
    args = ap.parse_args(argv)

    exp = json.loads(Path(args.expectations).read_text())
    doc = {"schema": SCHEMA, "mode": args.mode, "port": args.port,
           "chunked_prefill_size": args.cps, "t_start": time.time()}
    segments = []

    if args.mode == "p1e":
        doc["phases"] = {}
        for phase in exp["phases"]:
            pid = phase["id"]
            lens = list(phase["prompt_lens"])
            if pid == "S":
                rows = run_sequential(args.port, lens, pid, args.max_new_tokens)
                note = "concurrency 1; the registered numbers are exact"
            else:
                if args.burst_n:
                    unit = lens[0]
                    lens = [unit] * args.burst_n
                rows = run_burst(args.port, lens, pid, args.max_new_tokens)
                note = ("concurrency %d; only the inequalities in "
                        "gpu_assertions are registered for this phase" % len(lens))
            s = summarize(rows, args.cps, pid, note)
            doc["phases"][pid] = s
            segments.append({"segment_id": f"p1e_{pid}", "phase": pid,
                             "t_start": s["t_start"], "t_end": s["t_end"]})

    elif args.mode == "d":
        d = exp["p1d_positive_control"]
        n = args.burst_n or d["n_prompts"]
        lens = [d["prompt_len"]] * n
        rows = run_burst(args.port, lens, "D", args.max_new_tokens)
        s = summarize(rows, args.cps, "D",
                      "batch-boundary positive control; every prompt is shorter "
                      "than cps by construction")
        s["burst_n_override"] = args.burst_n
        doc.update(s)
        doc["phases"] = {"D": s}
        segments.append({"segment_id": "p1d_D", "phase": "D",
                         "t_start": s["t_start"], "t_end": s["t_end"]})

    elif args.mode == "mono":
        # P1-b needs a boot in which a single forward carries more than 512
        # extend tokens.  The sequential phase's 1500-token prompt does that.
        lens = [p for p in exp["phases"] if p["id"] == "S"][0]["prompt_lens"]
        rows = run_sequential(args.port, lens, "S", args.max_new_tokens)
        s = summarize(rows, args.cps, "S", "fused_mono negative control")
        doc.update(s)
        doc["phases"] = {"S": s}
        segments.append({"segment_id": "mono_S", "phase": "S",
                         "t_start": s["t_start"], "t_end": s["t_end"]})

    elif args.mode == "greedy":
        responses = []
        for i, text in enumerate(GREEDY_PROMPTS):
            resp = gen_text_greedy(args.port, text, 32, f"greedy-{i}")
            mi = resp.get("meta_info") or {}
            olp = mi.get("output_token_logprobs") or []
            responses.append({
                "prompt": text,
                "text": resp.get("text"),
                # [logprob, token_id, token_text] triples -> the exact ids
                "output_ids": [t[1] for t in olp] if olp else None,
                "output_logprobs": [t[0] for t in olp] if olp else None,
                "finish_reason": (mi.get("finish_reason") or {}).get("type"),
                "prompt_tokens": mi.get("prompt_tokens"),
                "completion_tokens": mi.get("completion_tokens"),
            })
        doc["responses"] = responses

    doc["t_end"] = time.time()
    Path(args.out).write_text(json.dumps(doc, indent=2) + "\n")
    if args.out_segments:
        Path(args.out_segments).write_text(json.dumps(
            {"segments": close_segments(segments), "mode": args.mode,
             "port": args.port, "chunked_prefill_size": args.cps,
             "segment_grace_s": SEGMENT_GRACE_S}, indent=2) + "\n")

    mism = [p for p, s in (doc.get("phases") or {}).items() if s.get("length_mismatch")]
    if mism:
        sys.stderr.write(
            f"WARNING: the engine reports different prompt lengths than requested "
            f"for phase(s) {mism}; the registered expectations assume they match\n"
        )
        return 5
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except urllib.error.URLError as exc:  # server not reachable
        sys.stderr.write(f"WORKLOAD_HTTP_ERROR: {exc}\n")
        sys.exit(6)
