#!/usr/bin/env python3
"""Fixed raw schema of the motivation harness (motivation.raw/v1) and its validator.  CPU only.

The raw run record is what every later analysis may use; SLO values are NOT inputs of any arm, so every attainment / capacity number is
recomputable from it (gate 8: no re-scoring of controllers is needed because nothing here has a controller).  Per run:
  identity   run_id arm boot_id job_name gpu_uuid hostname spec trace_hash engine engine_tree
  card       card_sha + card_check (the observed values of the post-boot card comparison; result must be empty)
  window     t_load_start/end (epoch + CLOCK_MONOTONIC)
  split      realized_split = {"available": bool, ...}; a PD-mux boot has no instrumentation in a measurement boot, so the default is
             available=false with the reason and the TARGET D -- never a silently empty field
  segments[] requests[] per request:
             request_id phase input_tokens requested_output_tokens observed_output_tokens ignore_eos req_class
             scheduled_arrival_s sent_s completion_s t_sent_mono t_finished_mono ttft_ms
             token_t_mono   (client timestamp of EVERY output token, CLOCK_MONOTONIC)
             token_itl_ms   (gaps between consecutive token stamps; exactly 0.0 inside one stream chunk)
             chunk_delta    (tokens per stream chunk, in order; sum == observed_output_tokens)
             error error_kind
chunk_delta rule (both engines, one rule): the clients stamp a chunk's tokens with ONE clock reading (trace_loadgen.StreamState and
OpenAIStreamState both extend token_times by [now] * delta), so consecutive equal stamps = one chunk.  Derived from the stamps, not
read from the SSE payload; validate_run requires both derivations (itl == 0 runs, equal-stamp runs) to agree.
"""
from __future__ import annotations

import mo_spec as MS

REQUEST_KEYS = ("request_id", "phase", "input_tokens", "requested_output_tokens", "observed_output_tokens", "ignore_eos", "req_class",
                "scheduled_arrival_s", "sent_s", "completion_s", "t_sent_mono", "t_finished_mono", "ttft_ms", "token_t_mono", "token_itl_ms",
                "chunk_delta", "error", "error_kind")
RUN_KEYS = ("schema_version", "run_id", "kind", "cell", "seed", "arm", "boot_id", "boot_role", "job_name", "gpu_uuid", "hostname", "card_sha",
            "card_check", "trace_hash", "engine", "engine_tree", "spec", "segments", "t_load_start_mono", "t_load_end_mono",
            "t_load_start_epoch", "t_load_end_epoch", "realized_split")


def chunk_deltas_from_itl(token_itl_ms, observed):
    """Tokens per stream chunk from the ITL list: a token whose gap to the previous token is exactly 0.0 shares its chunk."""
    if not observed:
        return []
    out = [1]
    for g in token_itl_ms[:observed - 1]:
        if g == 0.0:
            out[-1] += 1
        else:
            out.append(1)
    return out


def chunk_deltas_from_stamps(token_t_mono):
    out = []
    for i, t in enumerate(token_t_mono):
        if i and t == token_t_mono[i - 1]:
            out[-1] += 1
        else:
            out.append(1)
    return out


def validate_request(r):
    out = []
    miss = [k for k in REQUEST_KEYS if k not in r]
    if miss:
        return [f"request {r.get('request_id')}: missing {miss}"]
    n, rid = int(r["observed_output_tokens"]), r["request_id"]
    t = r["token_t_mono"]
    if len(t) != n:
        out.append(f"{rid}: len(token_t_mono) {len(t)} != observed_output_tokens {n}")
    if any(b < a for a, b in zip(t, t[1:])):
        out.append(f"{rid}: token_t_mono not monotone")
    if len(r["token_itl_ms"]) != max(0, n - 1):
        out.append(f"{rid}: len(token_itl_ms) {len(r['token_itl_ms'])} != n-1")
    if len(t) == n and len(r["token_itl_ms"]) == max(0, n - 1) and any(abs((b - a) * 1000.0 - g) > 1e-3 for a, b, g in zip(t, t[1:], r["token_itl_ms"])):
        out.append(f"{rid}: token_itl_ms is not the gap list of token_t_mono")
    cd = r["chunk_delta"]
    if sum(cd) != n:
        out.append(f"{rid}: sum(chunk_delta) {sum(cd)} != observed_output_tokens {n}")
    if cd != chunk_deltas_from_stamps(t) or cd != chunk_deltas_from_itl(r["token_itl_ms"], n):
        out.append(f"{rid}: chunk_delta disagrees with the token stamps / ITL list")
    if r["req_class"] != MS.request_class(int(r["input_tokens"])):
        out.append(f"{rid}: req_class {r['req_class']} != registered class of {r['input_tokens']} input tokens")
    return out


def validate_run(rec):
    """-> list of problems; empty = the record satisfies the fixed schema."""
    out = []
    miss = [k for k in RUN_KEYS if k not in rec]
    if miss:
        return [f"run {rec.get('run_id')}: missing {miss}"]
    if rec["schema_version"] != MS.SCHEMA_VERSION:
        out.append(f"schema_version {rec['schema_version']} != {MS.SCHEMA_VERSION}")
    rs = rec["realized_split"]
    if not isinstance(rs, dict) or "available" not in rs or (not rs["available"] and not rs.get("reason")):
        out.append("realized_split must be {'available': bool, ...} with a reason when unavailable")
    if rec["card_check"] is None or rec["card_check"].get("mismatches") != []:
        out.append("card_check missing or not clean")
    for s in rec["segments"]:
        for r in s["requests"]:
            out += validate_request(r)
    return out
