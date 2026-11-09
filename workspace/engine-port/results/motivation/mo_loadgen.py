#!/usr/bin/env python3
"""Client wrapper of the motivation harness: the sealed stock-baseline clients (sb_loadgen: SGLang /generate through the sealed
trace_loadgen/abs_mono path, vLLM /v1/completions twin) with the fixed raw-schema fields added to every request record
(chunk_delta, req_class, ignore_eos).  Nothing sealed is modified; this module only post-processes the records the clients write."""
from __future__ import annotations

import sb_loadgen as LG
import mo_schema as SC
import mo_spec as MS

META_KEYS = ("request_id", "phase", "input_tokens", "requested_output_tokens", "observed_output_tokens", "scheduled_arrival_s", "sent_s",
             "completion_s", "t_sent_mono", "t_finished_mono", "ttft_ms", "token_t_mono", "token_itl_ms", "error", "error_kind")


def normalize(rec, req_by_id):
    """Fixed-schema request record from a client record + the request that was sent."""
    q = req_by_id.get(str(rec["request_id"]), {})
    out = {k: rec.get(k) for k in META_KEYS}
    out["ignore_eos"] = bool(q.get("ignore_eos", True))
    out["req_class"] = MS.request_class(int(rec["input_tokens"]))
    out["chunk_delta"] = SC.chunk_deltas_from_stamps(rec.get("token_t_mono") or [])
    if "doc_id" in q:
        out["doc_id"] = q["doc_id"]
    return out


def replay(api, base_url, requests, out_path):
    """One segment -> (duration_s, fixed-schema request records).  api = 'sglang' | 'openai'."""
    dur, recs = LG.replay(api, base_url, requests, out_path)
    by_id = {str(q["request_id"]): q for q in requests}
    return dur, [normalize(r, by_id) for r in recs]
