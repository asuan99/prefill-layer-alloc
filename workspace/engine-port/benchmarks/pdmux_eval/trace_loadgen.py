"""Replay immutable token-length/arrival traces against SGLang's SSE endpoint."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from pathlib import Path
from typing import Dict, List

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
    itl_slo_ms: float,
) -> Dict[str, float]:
    try:
        import aiohttp
    except ImportError as exc:
        raise RuntimeError("trace replay requires aiohttp") from exc

    first_arrival = min(float(item["arrival_s"]) for item in requests)
    benchmark_started = time.perf_counter()
    results: List[Dict[str, object]] = []

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=None, sock_connect=60, sock_read=600)
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
            token_times: List[float] = []
            last_completion_tokens = 0
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
                        line = raw.decode("utf-8", errors="replace").strip()
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if not data or data == "[DONE]":
                            continue
                        chunk = json.loads(data)
                        meta = chunk.get("meta_info") or {}
                        completion_tokens = int(
                            meta.get("completion_tokens", last_completion_tokens + 1)
                        )
                        if completion_tokens > last_completion_tokens:
                            now = time.perf_counter()
                            delta = completion_tokens - last_completion_tokens
                            token_times.extend([now] * delta)
                            last_completion_tokens = completion_tokens
            except Exception as exc:
                error = repr(exc)
            finished = time.perf_counter()
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
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--ttft-slo-ms", type=float, required=True)
    parser.add_argument("--itl-slo-ms", type=float, required=True)
    args = parser.parse_args()
    summary = asyncio.run(
        replay(
            args.endpoint,
            read_requests(args.trace),
            args.output,
            args.ttft_slo_ms,
            args.itl_slo_ms,
        )
    )
    args.summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
