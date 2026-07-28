"""Closed-loop PREFILL-probe client for the s8p_prefill sweep (mirror of
s8_scaleup/s0dc_client.py with roles swapped).

s8_scaleup measured decode ITL sensitivity to decode-SM, using a serial
1-token "keepalive" prefill loop only to keep the green-ctx split active while
decode was the measured signal. This campaign asks the mirror question --
prefill-SM sensitivity -- so the roles invert:

  - MEASURED signal: a SERIAL (default concurrency 1) loop of max_new_tokens=1
    "probe" requests, cycling prompt length L through --probe-lengths. Latency
    of a 1-new-token request is dominated by the prefill forward pass, so this
    is this campaign's stand-in for "prefill forward duration" (no such field
    exists in telemetry -- see DESIGN.md sec 2).
  - BACKGROUND coexistence load: closed-loop decode requests (short prompt,
    long ignore_eos generation) held at --decode-conc concurrency, replaced on
    completion. This is what keeps the green-ctx split's decode side resident;
    per s8's own finding (FINDINGS_8B_2026-07-28.md sec 4-2), the split
    reverts to the unpartitioned (0,108) stream whenever BOTH split_prefill_batch
    and running_batch go empty inside multiplexing_mixin.adjust_stream_groups,
    so the background decode load must never fully drain.

Emits per-probe records (L, absolute wall time via t0_monotonic_s, latency) to
--raw-out and a per-L summary on stdout. Absolute time origin matters for
post-hoc partition attribution against telemetry runtime_snapshot rows
(t0_monotonic_s + perf_counter deltas is directly comparable to telemetry's
timestamp_monotonic_s -- see s8_scaleup's s8_analyze.py and the client t0 bug
it documents, FINDINGS sec 4-4).
"""

import argparse
import json
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
    pos = q * (len(ys) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ys) - 1)
    return ys[lo] + (ys[hi] - ys[lo]) * (pos - lo)


def decode_worker(url, prompt, out_tokens, t0, stop_at, counters, rec_lock):
    """Background coexistence load: keep exactly one decode request in flight
    (per thread), replaced on completion. Not the measured signal -- exists
    only so the green-ctx split's decode side stays resident (see module
    docstring). Streaming is used only so a dead/rejected request is detected
    the same way s0dc_client.py's stream_one() detects it (865006 signature:
    an HTTP-200 error payload with <2 chunks must not look like success).
    """
    while time.perf_counter() - t0 < stop_at:
        body = json.dumps({
            "text": prompt,
            "sampling_params": {
                "max_new_tokens": out_tokens,
                "temperature": 0.0,
                "ignore_eos": True,
            },
            "stream": True,
        }).encode()
        req = urllib.request.Request(
            url + "/generate", data=body,
            headers={"Content-Type": "application/json"},
        )
        n_chunks = 0
        last_payload = ""
        try:
            with urllib.request.urlopen(req) as resp:
                for raw in resp:
                    line = raw.decode("utf-8", "ignore").strip()
                    if not line:
                        continue
                    if not line.startswith("data:"):
                        last_payload = line[:300]
                        continue
                    payload = line[len("data:"):].strip()
                    if payload == "[DONE]":
                        break
                    last_payload = payload[:300]
                    n_chunks += 1
            if n_chunks < 2:
                raise RuntimeError(
                    f"decode background stream produced {n_chunks} chunk(s); "
                    f"server said: {last_payload!r}")
            with rec_lock:
                counters["decode_done"] += 1
        except Exception as exc:  # noqa: BLE001 - must not kill the loop
            with rec_lock:
                counters["decode_errors"] += 1
                counters["decode_last_error"] = f"{type(exc).__name__}: {exc}"
            time.sleep(0.05)


def probe_worker(url, prompts_by_l, ls, t0, stop_at, records, rec_lock,
                  counters, offset, stride):
    """MEASURED signal: serial max_new_tokens=1 requests, cycling L.

    Non-streaming (single response body) is enough here -- with one new token
    the whole response IS the first token, so request round-trip latency is
    this campaign's TTFT proxy. `offset`/`stride` let multiple probe threads
    (the --probe-conc >1 sensitivity check) interleave over different phases
    of the L cycle instead of all hammering the same L at once.
    """
    i = offset
    while time.perf_counter() - t0 < stop_at:
        L = ls[i % len(ls)]
        i += stride
        prompt = prompts_by_l[L]
        body = json.dumps({
            "text": prompt,
            "sampling_params": {
                "max_new_tokens": 1,
                "temperature": 0.0,
                "ignore_eos": True,
            },
        }).encode()
        req = urllib.request.Request(
            url + "/generate", data=body,
            headers={"Content-Type": "application/json"},
        )
        t_start = time.perf_counter() - t0
        try:
            with urllib.request.urlopen(req) as resp:
                payload = resp.read()
            t_end = time.perf_counter() - t0
            try:
                obj = json.loads(payload)
            except Exception:
                obj = None
            ok = isinstance(obj, dict) and "meta_info" in obj
            with rec_lock:
                if ok:
                    records.append({
                        "L": L, "t_start_s": t_start, "t_end_s": t_end,
                        "latency_ms": (t_end - t_start) * 1000.0,
                    })
                    counters["probe_done"] += 1
                else:
                    counters["probe_errors"] += 1
                    counters["probe_last_error"] = payload[:300].decode(
                        "utf-8", "ignore")
        except Exception as exc:  # noqa: BLE001
            with rec_lock:
                counters["probe_errors"] += 1
                counters["probe_last_error"] = f"{type(exc).__name__}: {exc}"
            time.sleep(0.05)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--decode-prompt-file", required=True)
    ap.add_argument("--decode-conc", type=int, default=8,
                     help="background decode requests held in flight")
    ap.add_argument("--decode-out-tokens", type=int, default=512)
    ap.add_argument("--probe-prompt", action="append", required=True,
                     help="L:path/to/prompt.txt, repeatable")
    ap.add_argument("--probe-conc", type=int, default=1,
                     help="concurrent serial prefill-probe loops "
                          "(1=default protocol, 2=queueing sensitivity check)")
    ap.add_argument("--warmup-s", type=float, default=20.0)
    ap.add_argument("--measure-s", type=float, default=60.0)
    ap.add_argument("--raw-out", required=True)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    decode_prompt = open(args.decode_prompt_file).read()
    prompts_by_l, ls = {}, []
    for spec in args.probe_prompt:
        l_str, path = spec.split(":", 1)
        L = int(l_str)
        prompts_by_l[L] = open(path).read()
        ls.append(L)
    ls.sort()

    records, rec_lock = [], threading.Lock()
    counters = {
        "decode_done": 0, "decode_errors": 0, "decode_last_error": None,
        "probe_done": 0, "probe_errors": 0, "probe_last_error": None,
    }
    total_s = args.warmup_s + args.measure_s
    t0 = time.perf_counter()

    threads = []
    for _ in range(args.decode_conc):
        t = threading.Thread(
            target=decode_worker,
            args=(args.url, decode_prompt, args.decode_out_tokens, t0,
                  total_s, counters, rec_lock),
            daemon=True)
        t.start()
        threads.append(t)
    for k in range(args.probe_conc):
        t = threading.Thread(
            target=probe_worker,
            args=(args.url, prompts_by_l, ls, t0, total_s, records, rec_lock,
                  counters, k, args.probe_conc),
            daemon=True)
        t.start()
        threads.append(t)
    for t in threads:
        t.join(timeout=total_s + 180)

    lo, hi = args.warmup_s, args.warmup_s + args.measure_s
    in_window = [r for r in records if lo <= r["t_end_s"] <= hi]

    with open(args.raw_out, "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")

    per_l = {}
    for L in ls:
        lat = [r["latency_ms"] for r in in_window if r["L"] == L]
        per_l[str(L)] = {
            "n": len(lat),
            "p50_ms": _percentile(lat, 0.50),
            "p95_ms": _percentile(lat, 0.95),
            "mean_ms": statistics.fmean(lat) if lat else float("nan"),
        }

    summary = {
        "tag": args.tag,
        # See module docstring: perf_counter and monotonic share
        # CLOCK_MONOTONIC on Linux, so t0_monotonic_s + t_start_s/t_end_s is
        # directly comparable to telemetry timestamp_monotonic_s.
        "t0_monotonic_s": t0,
        "decode_conc": args.decode_conc,
        "probe_conc": args.probe_conc,
        "warmup_s": args.warmup_s,
        "measure_s": args.measure_s,
        "decode_done": counters["decode_done"],
        "decode_errors": counters["decode_errors"],
        "decode_last_error": counters["decode_last_error"],
        "probe_done_total": counters["probe_done"],
        "probe_errors": counters["probe_errors"],
        "probe_last_error": counters["probe_last_error"],
        "probe_in_window_total": len(in_window),
        "per_l": per_l,
    }
    print(json.dumps(summary), flush=True)

    if not in_window:
        print(f"NO_PROBE_SAMPLES_IN_WINDOW probe_errors={counters['probe_errors']} "
              f"last_error={counters['probe_last_error']}", file=sys.stderr)
        return 4
    # Mirror s0dc_client.py's 865006 guard: an all-erroring probe loop must not
    # look like a completed run just because SOME background decode succeeded.
    if counters["probe_errors"] > counters["probe_done"]:
        print(f"MOSTLY_FAILED_PROBES errors={counters['probe_errors']} "
              f"ok={counters['probe_done']} "
              f"last_error={counters['probe_last_error']}", file=sys.stderr)
        return 5
    if counters["decode_errors"] > max(1, counters["decode_done"]):
        print(f"MOSTLY_FAILED_DECODE_BACKGROUND errors={counters['decode_errors']} "
              f"ok={counters['decode_done']} "
              f"last_error={counters['decode_last_error']}", file=sys.stderr)
        return 6
    return 0


if __name__ == "__main__":
    sys.exit(main())
