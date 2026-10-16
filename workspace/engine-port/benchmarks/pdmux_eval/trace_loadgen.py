"""Replay immutable token-length/arrival traces against SGLang's SSE endpoint."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from pathlib import Path
from typing import Callable, Dict, List, Tuple

from .analyze import RequestResult, summarize_requests


def read_requests(path: Path) -> List[Dict[str, object]]:
    requests = []
    for line in path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        if item.get("record") == "request":
            requests.append(item)
    if not requests:
        raise ValueError(f"trace contains no requests: {path}")
    return requests


async def replay(
    endpoint: str,
    requests: List[Dict[str, object]],
    output_path: Path,
    ttft_slo_ms: float,
    itl_slo_ms: float, connector_limit: "int | None" = None,
) -> Dict[str, float]:
    try:
        import aiohttp
    except ImportError as exc:
        raise RuntimeError("trace replay requires aiohttp") from exc

    first_arrival = min(float(item["arrival_s"]) for item in requests)
    benchmark_started = time.perf_counter()
    results: List[Dict[str, object]] = []

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=None, sock_connect=60, sock_read=600), **({} if connector_limit is None else {"connector": aiohttp.TCPConnector(limit=connector_limit)})
    ) as session:
        result_lock = asyncio.Lock()

        async def issue(item: Dict[str, object]) -> None:
            target = float(item["arrival_s"]) - first_arrival
            await asyncio.sleep(max(0.0, benchmark_started + target - time.perf_counter()))
            sent = time.perf_counter()
            payload = {
                "input_ids": [1] * int(item["input_tokens"]),
                "sampling_params": {
                    "max_new_tokens": int(item["output_tokens"]),
                    "temperature": 0.0,
                    "ignore_eos": True,
                },
                "stream": True,
                "rid": str(item["request_id"]),
            }
            # Outcome classification (2026-10-16): StreamState parses the SSE lines and keeps
            # the server-side failure signals apart from token arrivals -- an `"error"` chunk
            # (post-acceptance refusal, HTTP 200) is never a token, an abort finish_reason is
            # recorded, and an ignore_eos stream that ends cleanly short of the requested
            # length is OUTPUT_SHORT.  `error` stays non-empty for every failure, so every
            # existing `not r["error"]` filter is unchanged; `error_kind` / `error_side` are
            # new fields (definitions + precedence at StreamState below).  A client exception
            # still wins over every server signal, as before.  Token timestamps are taken
            # with the same perf_counter clock, at the same point (on a token count increase).
            state = StreamState()
            error = ""
            status = 0
            try:
                async with session.post(endpoint, json=payload) as response:
                    status = response.status
                    response.raise_for_status()
                    while True:
                        raw = await response.content.readline()
                        if not raw:
                            break
                        state.feed_line(raw, time.perf_counter)
            except Exception as exc:
                error = repr(exc)
            finished = time.perf_counter()
            token_times = state.token_times
            error, error_kind, error_side = state.classify(
                int(item["output_tokens"]),
                bool(payload["sampling_params"]["ignore_eos"]),
                client_error=error,
                http_status=status,
            )
            ttft_ms = (
                (token_times[0] - sent) * 1000.0 if token_times else math.inf
            )
            itls = [
                (current - previous) * 1000.0
                for previous, current in zip(token_times, token_times[1:])
            ]
            record = {
                "request_id": str(item["request_id"]),
                "phase": str(item["phase"]),
                "input_tokens": int(item["input_tokens"]),
                "requested_output_tokens": int(item["output_tokens"]),
                "observed_output_tokens": len(token_times),
                "scheduled_arrival_s": target,
                "sent_s": sent - benchmark_started,
                "completion_s": finished - benchmark_started,
                "ttft_ms": ttft_ms if math.isfinite(ttft_ms) else None,
                "token_itl_ms": itls,
                "http_status": status,
                "error": error,
                "error_kind": error_kind,
                "error_side": error_side,
            }
            async with result_lock:
                results.append(record)

        await asyncio.gather(*(issue(item) for item in requests))

    results.sort(key=lambda result: str(result["request_id"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output:
        for result in results:
            output.write(json.dumps(result, separators=(",", ":"), sort_keys=True) + "\n")

    successful = [
        RequestResult(
            request_id=str(result["request_id"]),
            ttft_ms=float(result["ttft_ms"]),
            token_itl_ms=tuple(float(value) for value in result["token_itl_ms"]),
            completion_s=float(result["completion_s"]),
        )
        for result in results
        if result["ttft_ms"] is not None and not result["error"]
    ]
    benchmark_finished = time.perf_counter()
    summary = summarize_requests(
        successful,
        0.0,
        benchmark_finished - benchmark_started,
        ttft_slo_ms,
        itl_slo_ms,
    )
    summary["attempted_requests"] = float(len(results))
    summary["failed_requests"] = float(len(results) - len(successful))
    summary["server_error_requests"] = float(
        sum(1 for result in results if result["error_side"] == "server"))
    summary["client_error_requests"] = float(
        sum(1 for result in results if result["error_side"] == "client"))
    return summary


# ---------------------------------------------------------------------------
# Request outcome classification (2026-10-16, M3 rev12 audit R-2; M3 + G0 shared).
#
# SGLang answers a STREAMING request it refuses after acceptance with HTTP 200, one
# `data: {"error": {"message": ...}}` chunk and `data: [DONE]`
# (entrypoints/http_server.py generate_request.stream_results: a ValueError from
# tokenizer_manager._validate_one_request, e.g. input + max_new_tokens >= context_len).
# The old loop counted every chunk without `meta_info` as one more token, so such a
# request was recorded as a 1-token SUCCESS with error "" and a TTFT equal to the
# rejection latency.  The scheduler can also shorten max_new_tokens silently
# (scheduler.init_req_max_new_tokens: max_req_len - input - 1), which with
# ignore_eos shows up only as fewer tokens than requested.
#
# `error_kind` / `error_side` are NEW record fields; `error` keeps its old meaning for
# every consumer ("non-empty = not a success") and is now non-empty for every kind
# below, so existing filters (`not r["error"]`, mg_label.is_error,
# analyze_concurrent, m2g_correctness_gate) exclude these requests unchanged.
#   kind               side    meaning
#   ""                 ""      success: clean stream end, no error, tokens == requested
#   CLIENT_EXCEPTION   client  an exception raised in the client (connect, read
#                              timeout, decode) -- the pre-2026-10-16 error path
#   HTTP_STATUS        server  the server answered with HTTP status >= 400
#   SSE_ERROR          server  the stream carried an `"error"` chunk (refused after
#                              acceptance); the server message is kept in `error`
#   SERVER_ABORT       server  a chunk carried meta_info.finish_reason.type == "abort"
#   OUTPUT_SHORT       server  ignore_eos request, clean stream end, fewer tokens
#                              observed than requested
#   OUTPUT_LONG        server  ignore_eos request, more tokens observed than requested
# Precedence (first match wins): CLIENT_EXCEPTION/HTTP_STATUS > SSE_ERROR >
# SERVER_ABORT > OUTPUT_SHORT/OUTPUT_LONG.  Records written before 2026-10-16 have no
# `error_kind`; treat a missing key as "unclassified (legacy)", never as "".
# ---------------------------------------------------------------------------
KIND_CLIENT_EXCEPTION = "CLIENT_EXCEPTION"
KIND_HTTP_STATUS = "HTTP_STATUS"
KIND_SSE_ERROR = "SSE_ERROR"
KIND_SERVER_ABORT = "SERVER_ABORT"
KIND_OUTPUT_SHORT = "OUTPUT_SHORT"
KIND_OUTPUT_LONG = "OUTPUT_LONG"
SIDE_OF_KIND = {
    "": "",
    KIND_CLIENT_EXCEPTION: "client",
    KIND_HTTP_STATUS: "server",
    KIND_SSE_ERROR: "server",
    KIND_SERVER_ABORT: "server",
    KIND_OUTPUT_SHORT: "server",
    KIND_OUTPUT_LONG: "server",
}


def _message_of(value: object) -> str:
    """Server error payload -> text, kept verbatim when it is already a string."""
    if isinstance(value, dict) and "message" in value:
        value = value["message"]
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, default=str)


class StreamState:
    """Per-request SSE parser: token arrival times + server-side failure signals.

    `feed_line` takes one raw SSE line; `classify` turns the final state into
    (error, error_kind, error_side).  Kept free of aiohttp so it is unit-testable.
    """

    def __init__(self) -> None:
        self.token_times: List[float] = []
        self.last_completion_tokens = 0
        self.sse_error = ""          # message of the first `"error"` chunk
        self.sse_error_seen = False
        self.abort_message = ""
        self.abort_seen = False
        self.done_seen = False

    def feed_line(self, raw: bytes, clock: Callable[[], float]) -> None:
        line = raw.decode("utf-8", errors="replace").strip()
        if not line.startswith("data:"):
            return
        data = line[5:].strip()
        if not data:
            return
        if data == "[DONE]":
            self.done_seen = True
            return
        chunk = json.loads(data)
        if isinstance(chunk, dict) and "error" in chunk:
            # Refused after acceptance.  NOT a token: no meta_info, no output.
            if not self.sse_error_seen:
                self.sse_error_seen = True
                self.sse_error = _message_of(chunk["error"])
            return
        meta = (chunk.get("meta_info") if isinstance(chunk, dict) else None) or {}
        finish = meta.get("finish_reason")
        if isinstance(finish, dict) and finish.get("type") == "abort" and not self.abort_seen:
            self.abort_seen = True
            self.abort_message = _message_of(finish.get("message", finish))
        completion_tokens = int(
            meta.get("completion_tokens", self.last_completion_tokens + 1)
        )
        if completion_tokens > self.last_completion_tokens:
            now = clock()
            delta = completion_tokens - self.last_completion_tokens
            self.token_times.extend([now] * delta)
            self.last_completion_tokens = completion_tokens

    def classify(
        self,
        requested_output_tokens: int,
        ignore_eos: bool,
        client_error: str = "",
        http_status: int = 0,
    ) -> Tuple[str, str, str]:
        """(error, error_kind, error_side); error is non-empty iff kind is non-empty."""
        if client_error:
            kind = KIND_HTTP_STATUS if http_status >= 400 else KIND_CLIENT_EXCEPTION
            return client_error, kind, SIDE_OF_KIND[kind]
        if self.sse_error_seen:
            return (f"{KIND_SSE_ERROR}: {self.sse_error}", KIND_SSE_ERROR,
                    SIDE_OF_KIND[KIND_SSE_ERROR])
        if self.abort_seen:
            return (f"{KIND_SERVER_ABORT}: {self.abort_message}", KIND_SERVER_ABORT,
                    SIDE_OF_KIND[KIND_SERVER_ABORT])
        observed = len(self.token_times)
        if ignore_eos and observed != int(requested_output_tokens):
            kind = KIND_OUTPUT_SHORT if observed < int(requested_output_tokens) else KIND_OUTPUT_LONG
            return (f"{kind}: observed {observed} != requested {int(requested_output_tokens)} "
                    f"(ignore_eos)", kind, SIDE_OF_KIND[kind])
        return "", "", ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--ttft-slo-ms", type=float, required=True)
    parser.add_argument("--itl-slo-ms", type=float, required=True)
    # connector_limit: None keeps aiohttp's default connector (100 simultaneous connections) =
    # the behaviour before 2026-10-12.  A multi-GPU cell holds more open streams than that (48
    # running per GPU plus the server-side queue); the default then queues the excess IN THE
    # CLIENT and TTFT measures the client (M1.5, results/multigpu_probe).  0 = unlimited.
    parser.add_argument("--connector-limit", type=int, default=None)
    args = parser.parse_args()
    summary = asyncio.run(
        replay(
            args.endpoint,
            read_requests(args.trace),
            args.output,
            args.ttft_slo_ms,
            args.itl_slo_ms,
            args.connector_limit,
        )
    )
    args.summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
