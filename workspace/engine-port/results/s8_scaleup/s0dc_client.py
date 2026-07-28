"""Closed-loop decode client for the Stage 0 de-confound sweep.

Differs from stage0_client.py in the two ways the de-confound protocol
(DESIGN.md §3) requires:

  1. CONSTANT OCCUPANCY. stage0_client fires a one-shot burst of --conc
     requests; as they finish, the decode batch drains, so the measured ITL
     mixes several batch sizes and the mix differs per arm. Here a finished
     request is immediately replaced, holding exactly --conc in flight for the
     whole measurement window, and the realized occupancy is reported so the
     assumption is checked rather than asserted.

  2. WARMUP DISCARD + STEADY WINDOW. Only token gaps whose arrival falls inside
     [warmup, warmup+measure] count. The first decode steps after a boot carry
     graph capture and allocator effects; Stage 0's own g2_0 campaigns showed
     warmup-driven TTFT blowups dominating a run.

Occupancy is closed-loop by construction, so this never runs above capacity --
that sidesteps the metric-cliff hazard (methodology gate #6) that an open-loop
arrival rate would reintroduce.

Emits per-token arrival timestamps to --raw-out and a summary JSON on stdout.
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


def stream_one(url, prompt, out_tokens, t0, rec_lock, records, stop_at):
    """One streamed generation; returns per-chunk arrival times (seconds since t0)."""
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
    times = []
    last_payload = ""
    started = time.perf_counter() - t0
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
            times.append(time.perf_counter() - t0)
    with rec_lock:
        records.append({"start_s": started, "chunk_times_s": times})
    # A rejected request (e.g. prompt+max_new_tokens over the context cap) comes
    # back as HTTP 200 carrying one error payload, so urlopen raises nothing and
    # the run looks healthy while generating no tokens at all. That silently
    # produced a whole invalid campaign (865006); treat it as the error it is.
    if len(times) < 2:
        raise RuntimeError(
            f"stream produced {len(times)} chunk(s), expected {out_tokens}; "
            f"server said: {last_payload!r}")
    return times


def worker(url, prompt, out_tokens, t0, rec_lock, records, stop_at, counters):
    """Keep exactly one request in flight until the deadline (closed loop)."""
    while time.perf_counter() - t0 < stop_at:
        try:
            stream_one(url, prompt, out_tokens, t0, rec_lock, records, stop_at)
            with rec_lock:
                counters["done"] += 1
        except Exception as exc:  # noqa: BLE001 - a dead request must not kill the loop
            with rec_lock:
                counters["errors"] += 1
                counters["last_error"] = f"{type(exc).__name__}: {exc}"
            time.sleep(0.05)


def keepalive_worker(url, prompt, t0, stop_at, counters, rec_lock):
    """Continuous 1-token prefill load: keeps the pdmux split partition active.

    The green-ctx sub-108 decode partition is only selected while a prefill is
    in flight, so without this the arm silently reverts to the full-108 stream
    and every split would measure the same thing.
    """
    while time.perf_counter() - t0 < stop_at:
        body = json.dumps({
            "text": prompt,
            "sampling_params": {"max_new_tokens": 1, "temperature": 0.0},
        }).encode()
        req = urllib.request.Request(
            url + "/generate", data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req) as resp:
                resp.read()
            with rec_lock:
                counters["keepalive_done"] += 1
        except Exception:  # noqa: BLE001
            with rec_lock:
                counters["keepalive_errors"] += 1
            time.sleep(0.05)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--conc", type=int, default=16,
                    help="decode requests held in flight (fixed occupancy)")
    ap.add_argument("--out-tokens", type=int, default=256)
    ap.add_argument("--warmup-s", type=float, default=20.0,
                    help="discard everything arriving before this")
    ap.add_argument("--measure-s", type=float, default=60.0,
                    help="length of the steady-state measurement window")
    ap.add_argument("--keepalive-prefill", type=int, default=2)
    ap.add_argument("--keepalive-prompt-file", default=None,
                    help="defaults to --prompt-file")
    ap.add_argument("--raw-out", required=True)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    prompt = open(args.prompt_file).read()
    ka_prompt = open(args.keepalive_prompt_file).read() \
        if args.keepalive_prompt_file else prompt

    records, rec_lock = [], threading.Lock()
    counters = {"done": 0, "errors": 0, "keepalive_done": 0,
                "keepalive_errors": 0, "last_error": None}
    total_s = args.warmup_s + args.measure_s
    t0 = time.perf_counter()

    threads = []
    for _ in range(args.conc):
        t = threading.Thread(
            target=worker,
            args=(args.url, prompt, args.out_tokens, t0, rec_lock, records,
                  total_s, counters),
            daemon=True)
        t.start()
        threads.append(t)
    for _ in range(args.keepalive_prefill):
        t = threading.Thread(
            target=keepalive_worker,
            args=(args.url, ka_prompt, t0, total_s, counters, rec_lock),
            daemon=True)
        t.start()
        threads.append(t)
    for t in threads:
        t.join(timeout=total_s + 120)

    # Steady-state ITL: gaps between consecutive chunks of ONE request, counted
    # only when the later chunk lands inside the measurement window. The first
    # gap of a request is prefill-to-first-token, not an ITL, so it is dropped.
    lo, hi = args.warmup_s, args.warmup_s + args.measure_s
    itls_ms, in_window_reqs = [], 0
    for rec in records:
        times = rec["chunk_times_s"]
        used = False
        for i in range(2, len(times)):
            if lo <= times[i] <= hi:
                itls_ms.append((times[i] - times[i - 1]) * 1000.0)
                used = True
        in_window_reqs += 1 if used else 0

    # Realized occupancy: mean number of requests concurrently streaming inside
    # the window. Verifies the fixed-occupancy assumption instead of trusting it.
    edges = []
    for rec in records:
        times = rec["chunk_times_s"]
        if not times:
            continue
        edges.append((max(times[0], lo), 1))
        edges.append((min(times[-1], hi), -1))
    edges.sort()
    occ_area, cur, prev_t = 0.0, 0, lo
    for t, d in edges:
        if t > hi:
            break
        if t >= lo:
            occ_area += cur * (t - prev_t)
            prev_t = t
        cur += d
    occ_mean = occ_area / (hi - lo) if hi > lo else float("nan")

    with open(args.raw_out, "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")

    summary = {
        "tag": args.tag,
        # Absolute origin of every chunk_times_s / start_s value in raw_out.
        # Those are perf_counter deltas from t0, while telemetry records
        # timestamp_monotonic_s, so without this the two cannot be aligned and
        # the partition-attribution analysis silently yields nothing (found the
        # hard way on 865311/865312). perf_counter and monotonic share
        # CLOCK_MONOTONIC on Linux, so t0_monotonic_s + chunk_times_s is
        # directly comparable to telemetry timestamps.
        "t0_monotonic_s": t0,
        "conc_requested": args.conc,
        "occupancy_mean_in_window": round(occ_mean, 2),
        "requests_completed": counters["done"],
        "requests_contributing": in_window_reqs,
        "errors": counters["errors"],
        "last_error": counters["last_error"],
        "keepalive_done": counters["keepalive_done"],
        "keepalive_errors": counters["keepalive_errors"],
        "warmup_s": args.warmup_s,
        "measure_s": args.measure_s,
        "itl_samples": len(itls_ms),
        "itl_ms_p50": _percentile(itls_ms, 0.50),
        "itl_ms_p95": _percentile(itls_ms, 0.95),
        "itl_ms_p99": _percentile(itls_ms, 0.99),
        "itl_ms_mean": statistics.fmean(itls_ms) if itls_ms else float("nan"),
    }
    print(json.dumps(summary), flush=True)
    if not itls_ms:
        print(f"NO_ITL_SAMPLES_IN_WINDOW errors={counters['errors']} "
              f"last_error={counters['last_error']}", file=sys.stderr)
        return 4
    # Every request failing while the run still reports a summary is the 865006
    # signature; surface it rather than emitting a plausible-looking record.
    if counters["errors"] > counters["done"]:
        print(f"MOSTLY_FAILED_REQUESTS errors={counters['errors']} "
              f"ok={counters['done']} last_error={counters['last_error']}",
              file=sys.stderr)
        return 5
    return 0


if __name__ == "__main__":
    sys.exit(main())
