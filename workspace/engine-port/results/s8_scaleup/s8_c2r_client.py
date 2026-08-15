"""Closed-loop decode client for the C2-R re-measurement campaign (rev2).

This is a COPY of s0dc_client.py, not an edit of it (PREREG_C2R_RULES_REV2
H3: "s0dc_client.py를 직접 고치지 말고 사본을 만들어 고쳐라" -- s0dc_client.py backs
every prior s8_scaleup campaign and must stay byte-for-byte reproducible).
The only behavioral change vs s0dc_client.py is in keepalive_worker(); every
other function (stream_one, worker, main's ITL/occupancy math, the 865006
mostly-failed-requests guard) is copied verbatim.

Why keepalive_worker needed a fix (PREREG_C2R_RULES_REV2 H3, "audit target 8"):
the original keepalive_worker only distinguishes success/failure by whether
urlopen() raised. For a NON-streaming /generate request (keepalive_worker
never sets "stream": true), sglang's http_server.py returns HTTP 400 for a
plain context-length rejection, which does raise (urllib.error.HTTPError) and
was already counted correctly. But that is not the only rejection shape: a
request that is admitted and then aborted mid-generation (e.g. KV-cache
pressure under a busy server) can come back as HTTP 200 with a well-formed
JSON envelope carrying a zero-completion / abort finish_reason, and a
generate() call that raises inside the async generator can surface as a 200
response whose body is {"error": {...}} rather than a raised exception on the
client side (see http_server.py's streaming branch, which folds a mid-stream
ValueError into a "data: " chunk instead of an HTTP error). Neither shape
raises in urllib, so the un-fixed keepalive_worker would silently count them
as "keepalive_done". This version parses the response body and only counts a
keepalive as done if it both returned 2xx AND decodes to a payload with no
"error" key, at least one completion token, and a non-abort finish_reason.

Everything below the keepalive_worker docstring/edit marker is unmodified
from s0dc_client.py (diff it if you need to confirm).
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


# ---------------------------------------------------------------------------
# H3 EDIT MARKER: keepalive_worker is the only function that differs from
# s0dc_client.py. See the module docstring for why.
# ---------------------------------------------------------------------------
def keepalive_worker(url, prompt, t0, stop_at, counters, rec_lock):
    """Continuous 1-token prefill load: keeps the pdmux split partition active.

    The green-ctx sub-108 decode partition is only selected while a prefill is
    in flight, so without this the arm silently reverts to the full-108 stream
    and every split would measure the same thing.

    PREREG_C2R_RULES_REV2 H3: judge success by parsing the response body, not
    by whether urlopen() raised. HTTP status alone is necessary but not
    sufficient -- a 2xx response can still carry an {"error": ...} payload or
    a zero-completion / aborted generation. A non-2xx response is still
    counted as an error (via the except clause below), but we also try to
    read its body for a useful "last_error" message instead of just the
    generic HTTPError repr.
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
                raw = resp.read()
            payload = json.loads(raw.decode("utf-8", "ignore"))
            if isinstance(payload, dict) and "error" in payload:
                raise RuntimeError(
                    f"HTTP 200 error payload: {str(payload['error'])[:200]}")
            meta = payload.get("meta_info") if isinstance(payload, dict) else None
            completion_tokens = (meta or {}).get("completion_tokens")
            finish_reason = (meta or {}).get("finish_reason")
            finish_type = (
                finish_reason.get("type")
                if isinstance(finish_reason, dict) else finish_reason
            )
            if completion_tokens is not None and completion_tokens < 1:
                raise RuntimeError(
                    f"0 completion tokens (finish_reason={finish_reason!r})")
            if finish_type == "abort":
                raise RuntimeError(f"aborted mid-generation: {finish_reason!r}")
            with rec_lock:
                counters["keepalive_done"] += 1
        except urllib.error.HTTPError as exc:
            try:
                err_body = exc.read().decode("utf-8", "ignore")[:200]
            except Exception:  # noqa: BLE001
                err_body = ""
            with rec_lock:
                counters["keepalive_errors"] += 1
                counters["keepalive_last_error"] = f"HTTP {exc.code}: {err_body}"
            time.sleep(0.05)
        except Exception as exc:  # noqa: BLE001
            with rec_lock:
                counters["keepalive_errors"] += 1
                counters["keepalive_last_error"] = f"{type(exc).__name__}: {exc}"
            time.sleep(0.05)
# ---------------------------------------------------------------------------
# End H3 edit. Everything below is unmodified from s0dc_client.py.
# ---------------------------------------------------------------------------


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
                "keepalive_errors": 0, "last_error": None,
                "keepalive_last_error": None}
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
        "keepalive_last_error": counters["keepalive_last_error"],
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
