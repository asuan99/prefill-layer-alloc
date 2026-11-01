#!/usr/bin/env python3
"""Clients of the stock-baseline harness (PREREG STOCK-BASE §4 ①, §5-0).  The sealed loadgens are not modified.

  * SGLang arms : `ph_loadgen.replay_natural(..., abs_mono=True)` -> `benchmarks/pdmux_eval/trace_loadgen.replay` on `/generate`
                  (the SAME client bytes as every CAP/TUNE/implfix boot -- PM-ref is the bridge arm), plus absolute CLOCK_MONOTONIC
                  send / per-token / finish stamps (`abs_mono_loadgen`).
  * vLLM arm    : `replay_openai` below, on `/v1/completions` (bench_serving `--backend vllm` analogue).  Same arrival schedule
                  (`arrival_s` relative to the first), same perf_counter stamping points (send just before the POST, one token time per
                  observed completion-token increase, finish after the stream closes), same record keys and serialisation, same
                  outcome kinds (trace_loadgen.StreamState semantics).  Token counting uses `usage.completion_tokens` of every chunk
                  (`stream_options.continuous_usage_stats`); a chunk without usage counts as one token (trace_loadgen fallback rule).
    Whether the two client paths measure TTFT the same way is exactly what the definition smoke (§5-0) checks; until it passes, every
    engine pair is INVALID (fail-closed, sb_analyze).
  * smoke       : `smoke_sequence` sends fixed requests ONE AT A TIME and scrapes the server TTFT histogram (`/metrics`,
                  `*time_to_first_token_seconds_sum/_count`) before and after each, so each request's server-side TTFT is the delta of the
                  sum (count delta must be exactly 1).  Neither stock server logs a per-request first-token time without extra flags; the
                  histogram is the stock observable of both (SGLang needs --enable-metrics on the smoke boot only).  Deviation D-2.
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(os.path.abspath(__file__)).parent
ENG = HERE.parents[1]
for _p in (str(ENG / "benchmarks"), str(HERE.parent / "cap_char" / "policy_harness")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from pdmux_eval import trace_loadgen as TL  # noqa: E402
from pdmux_eval import abs_mono_loadgen as AM  # noqa: E402


class OpenAIStreamState(TL.StreamState):
    """trace_loadgen.StreamState for OpenAI completion chunks: token count from usage.completion_tokens, errors / aborts as before."""

    def feed_line(self, raw, clock):
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
            if not self.sse_error_seen:
                self.sse_error_seen = True
                self.sse_error = TL._message_of(chunk["error"])
            return
        choices = chunk.get("choices") or [] if isinstance(chunk, dict) else []
        usage = chunk.get("usage") if isinstance(chunk, dict) else None
        if not choices and usage is not None and not (usage.get("completion_tokens") or 0) > self.last_completion_tokens:
            return                                            # final usage-only chunk (include_usage): no new token
        fr = (choices[0].get("finish_reason") if choices else None)
        if fr == "abort" and not self.abort_seen:
            self.abort_seen, self.abort_message = True, "finish_reason=abort"
        ct = int(usage["completion_tokens"]) if usage and usage.get("completion_tokens") is not None else self.last_completion_tokens + 1
        if ct > self.last_completion_tokens:
            now = clock()
            self.token_times.extend([now] * (ct - self.last_completion_tokens))
            self.last_completion_tokens = ct


def openai_payload(item, model):
    return {"model": model, "prompt": (list(item["input_ids"]) if "input_ids" in item else [1] * int(item["input_tokens"])),
            "max_tokens": int(item["output_tokens"]), "temperature": 0.0, "ignore_eos": item.get("ignore_eos", True) is not False,
            "stream": True, "stream_options": {"include_usage": True, "continuous_usage_stats": True}}


async def replay_openai(endpoint, requests, output_path, ttft_slo_ms, itl_slo_ms, model, connector_limit=0):
    """trace_loadgen.replay twin for /v1/completions.  Writes the same record keys + abs-mono stamps; returns a summary with duration_s."""
    import aiohttp
    for item in requests:
        if "input_ids" in item and len(item["input_ids"]) != int(item["input_tokens"]):
            raise ValueError(f"request {item.get('request_id')}: input_ids length != input_tokens")
    offset = time.clock_gettime(time.CLOCK_MONOTONIC) - time.perf_counter()
    first_arrival = min(float(i["arrival_s"]) for i in requests)
    started = time.perf_counter()
    results = []
    lock = asyncio.Lock()
    conn = {} if connector_limit is None else {"connector": aiohttp.TCPConnector(limit=connector_limit)}
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, sock_connect=60, sock_read=600), **conn) as session:
        async def issue(item):
            target = float(item["arrival_s"]) - first_arrival
            await asyncio.sleep(max(0.0, started + target - time.perf_counter()))
            sent = time.perf_counter()
            payload = openai_payload(item, model)
            state, error, status = OpenAIStreamState(), "", 0
            try:
                async with session.post(endpoint, json=payload) as resp:
                    status = resp.status
                    resp.raise_for_status()
                    while True:
                        raw = await resp.content.readline()
                        if not raw:
                            break
                        state.feed_line(raw, time.perf_counter)
            except Exception as exc:  # noqa: BLE001
                error = repr(exc)
            finished = time.perf_counter()
            tt = state.token_times
            error, kind, side = state.classify(int(item["output_tokens"]), payload["ignore_eos"], client_error=error, http_status=status)
            ttft = (tt[0] - sent) * 1000.0 if tt else math.inf
            rec = {"request_id": str(item["request_id"]), "phase": str(item["phase"]), "input_tokens": int(item["input_tokens"]),
                   "requested_output_tokens": int(item["output_tokens"]), "observed_output_tokens": len(tt),
                   "scheduled_arrival_s": target, "sent_s": sent - started, "completion_s": finished - started,
                   "ttft_ms": ttft if math.isfinite(ttft) else None,
                   "token_itl_ms": [(b - a) * 1000.0 for a, b in zip(tt, tt[1:])], "http_status": status, "error": error,
                   "error_kind": kind, "error_side": side,
                   "t_sent_mono": sent + offset, "t_finished_mono": finished + offset, "token_t_mono": [x + offset for x in tt]}
            async with lock:
                results.append(rec)
        await asyncio.gather(*(issue(i) for i in requests))
    results.sort(key=lambda r: r["request_id"])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, separators=(",", ":"), sort_keys=True) + "\n")
    return {"duration_s": time.perf_counter() - started, "benchmark_started_mono": started + offset, "abs_mono_offset_s": offset,
            "attempted_requests": float(len(results)), "failed_requests": float(sum(1 for r in results if r["error"]))}


def replay(api, base_url, requests, out_path, model=None):
    """One segment.  -> (duration_s, records).  api = 'sglang' (sealed client, /generate) | 'openai' (vLLM, /v1/completions)."""
    if api == "sglang":
        import ph_loadgen as PL
        summ = asyncio.run(PL.replay_natural(f"{base_url}/generate", requests, Path(out_path), 1e12, 1e12, connector_limit=0,
                                             abs_mono=True))
    elif api == "openai":
        summ = asyncio.run(replay_openai(f"{base_url}/v1/completions", requests, Path(out_path), 1e12, 1e12, model or served_model(base_url)))
    else:
        raise ValueError(api)
    recs = [json.loads(x) for x in Path(out_path).read_text().splitlines() if x.strip()]
    return float(summ["duration_s"]), recs


def served_model(base_url, timeout=10):
    with urllib.request.urlopen(f"{base_url}/v1/models", timeout=timeout) as r:
        return json.loads(r.read())["data"][0]["id"]


# ---- definition smoke (§5-0) ---------------------------------------------------------------------------------------------
TTFT_METRIC = re.compile(r"^(?P<name>[a-zA-Z_:]*time_to_first_token_seconds)_(?P<kind>sum|count)(?:\{[^}]*\})?\s+(?P<v>[0-9.eE+-]+)\s*$")


def parse_ttft_metric(text):
    """Prometheus text -> (sum_s, count) of the TTFT histogram, summed over label sets (vllm: / sglang: prefixes)."""
    s = c = 0.0
    seen = False
    for ln in text.splitlines():
        m = TTFT_METRIC.match(ln.strip())
        if not m:
            continue
        seen = True
        if m.group("kind") == "sum":
            s += float(m.group("v"))
        else:
            c += float(m.group("v"))
    return (s, c) if seen else (None, None)


def fetch_metrics(base_url, timeout=10):
    with urllib.request.urlopen(f"{base_url}/metrics", timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


def smoke_sequence(api, base_url, requests, out_dir, metrics_fn=fetch_metrics, replay_fn=replay, settle_s=0.2):
    """Send each request alone; server TTFT = delta of the histogram sum.  -> list of {request_id, client_ttft_ms, server_ttft_ms, ok}."""
    rows = []
    for k, item in enumerate(requests):
        s0, c0 = parse_ttft_metric(metrics_fn(base_url))
        _dur, recs = replay_fn(api, base_url, [dict(item, arrival_s=0.0)], Path(out_dir) / f"smoke_{k:02d}.jsonl")
        time.sleep(settle_s)
        s1, c1 = parse_ttft_metric(metrics_fn(base_url))
        r = recs[0] if recs else {}
        ok = None not in (s0, s1) and (c1 - c0) == 1 and r.get("ttft_ms") is not None and not r.get("error")
        rows.append({"request_id": item["request_id"], "client_ttft_ms": r.get("ttft_ms"),
                     "server_ttft_ms": (s1 - s0) * 1000.0 if ok else None, "count_delta": (c1 - c0) if None not in (c0, c1) else None,
                     "ok": bool(ok)})
    return rows
