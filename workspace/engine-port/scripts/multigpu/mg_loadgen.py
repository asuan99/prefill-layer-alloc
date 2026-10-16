#!/usr/bin/env python3
"""Multi-GPU load generator wrapper around pdmux_eval.trace_loadgen.replay (PREREG §1, H5, H6).

trace_loadgen.py is shared with N2/N3/E2/G0 and is NOT modified by this wrapper.  [rev12 correction of the rev11 sentence "trace_loadgen.py
itself is NOT modified": the shared loadgen WAS changed after rev11 -- 0eca947 (SSE error / OUTPUT_SHORT classification, error_side) and
rev12 F-3 (replay sends a request's own `input_ids` when the item carries them; items without them keep the rev11 `[1]*L` payload
byte for byte).]  This wrapper:
  * raises RLIMIT_NOFILE soft to >= 8192 (H6) BEFORE the event loop exists, and writes the achieved
    limits to <cell>/fd_limit.json; exit 3 (ABORT_FD_LIMIT) when the hard limit is below 8192;
  * records a clock bridge `loadgen_clock.json` = {perf_at_call, mono_at_call, epoch_at_call}
    taken immediately before replay().  replay() sets `benchmark_started = perf_counter()` a few ms
    later, so `perf_at_call + sent_s` is the CLOCK_MONOTONIC time of each send (the clock of the
    /proc sampler and of the engine telemetry) to within the asyncio start-up (tens of ms);
  * refuses to run if trace_loadgen.py does not carry `sock_read=600` (H5: the registered value; the
    MGP-35 CLIENT_ERROR thresholds were computed for exactly this timeout);
  * passes --connector-limit 0 (unlimited, §1);
  * rev12 §19-2 (R-8, F-3): a trace item with `doc_id` (shape X) or `floor_len` (floor block) gets its REAL tokens from the registered
    token file (--x-tokens, sha256 checked against --x-tokens-sha256 -> exit 5 ABORT_DATASET) and is sent by trace_loadgen.replay
    itself (StreamState classification); the item's input_tokens / output_tokens must equal the file's lengths (exit 5).  No other
    send path exists;
  * --sequential (floor block): one request at a time, each through replay().

usage: mg_loadgen.py --endpoint URL --trace T.jsonl --cell-dir DIR [--ttft-slo-ms 3000 --itl-slo-ms 50] [--x-tokens F --x-tokens-sha256 H]
                     [--sequential]
Outputs in DIR: requests.jsonl, summary.json, fd_limit.json, loadgen_clock.json.
"""

import argparse
import os
import asyncio
import json
import resource
import sys
import time
from pathlib import Path

ENGINE = Path(os.path.abspath(__file__)).parents[2]
sys.path.insert(0, str(ENGINE / "benchmarks"))
sys.path.insert(0, str(ENGINE / "results" / "multigpu"))
sys.path.insert(0, str(Path(os.path.abspath(__file__)).parent))
import mg_const as C  # noqa: E402


def raise_fd_limit(want=C.FD_SOFT_MIN):
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    rec = {"soft_before": soft, "hard": hard, "required": want}
    if hard != resource.RLIM_INFINITY and hard < want:
        rec["ok"] = False
        return rec
    target = max(soft, want * 8)             # headroom over the ~1,900-3,100 connection forecast (H6)
    if hard != resource.RLIM_INFINITY:
        target = min(target, hard)
    resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    rec["soft_after"] = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
    rec["ok"] = rec["soft_after"] >= want
    return rec


def sock_read_registered():
    src = (ENGINE / "benchmarks" / "pdmux_eval" / "trace_loadgen.py").read_text()
    return f"sock_read={C.SOCK_READ_S}" in src


def attach_input_ids(reqs, tokens_path, want_sha):
    """rev12: items with doc_id / floor_len -> item["input_ids"] (real tokens).  Returns a problem string or None."""
    need = [r for r in reqs if "doc_id" in r or "floor_len" in r]
    if not need:
        return None
    if not tokens_path:
        return "trace has shape-X / floor items but no --x-tokens"
    import mg_x_pretokenize as XT
    if not Path(tokens_path).is_file():
        return f"token file absent {tokens_path}"
    got = XT.sha256_file(tokens_path)
    if not want_sha or got != want_sha:
        return f"token file sha256 {got[:16]} != registered {str(want_sha)[:16]}"
    h = XT.read_header(tokens_path)
    idx = XT.doc_index(h)
    docs = sorted({int(r["doc_id"]) for r in need if "doc_id" in r})
    missing = [d for d in docs if d not in idx]
    if missing:
        return f"doc_id not in the token file: {missing[:5]}"
    ids = XT.doc_input_ids(tokens_path, h, docs, idx)
    floor = {}
    for r in need:
        if "doc_id" in r:
            d = int(r["doc_id"])
            if int(r["input_tokens"]) != idx[d][1]:
                return f"{r['request_id']}: input_tokens {r['input_tokens']} != file {idx[d][1]} (doc {d})"
            if r.get("phase") != "xcheck" and int(r["output_tokens"]) != idx[d][2]:
                return f"{r['request_id']}: output_tokens {r['output_tokens']} != file {idx[d][2]} (doc {d})"
            r["input_ids"] = ids[d]
        else:
            L = int(r["floor_len"])
            if L not in floor:
                floor[L] = XT.floor_input_ids(tokens_path, h, L)
            if int(r["input_tokens"]) != L:
                return f"{r['request_id']}: input_tokens {r['input_tokens']} != floor_len {L}"
            r["input_ids"] = floor[L]
    return None


async def replay_sequential(replay, endpoint, reqs, out_path, ttft, itl, limit):
    """Floor block: each request alone through replay(); records concatenated in trace order."""
    recs, summ = [], []
    for i, item in enumerate(reqs):
        tmp = out_path.with_name(f".seq_{i:05d}.jsonl")
        summ.append(await replay(endpoint, [dict(item, arrival_s=0.0)], tmp, ttft, itl, limit))
        recs += [json.loads(x) for x in tmp.read_text().splitlines() if x.strip()]
        tmp.unlink()
    with out_path.open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, separators=(",", ":"), sort_keys=True) + "\n")
    return {"sequential": True, "attempted_requests": float(len(recs)),
            "failed_requests": float(sum(1 for r in recs if r["error"] or r["ttft_ms"] is None)),
            "server_error_requests": float(sum(1 for r in recs if r.get("error_side") == "server")),
            "client_error_requests": float(sum(1 for r in recs if r.get("error_side") == "client"))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--cell-dir", type=Path, required=True)
    ap.add_argument("--ttft-slo-ms", type=float, default=3000.0)   # SLO only feeds the descriptive summary
    ap.add_argument("--itl-slo-ms", type=float, default=50.0)
    ap.add_argument("--x-tokens", default="", help="rev12: registered shape-X token file (doc_id / floor_len items)")
    ap.add_argument("--x-tokens-sha256", default="")
    ap.add_argument("--sequential", action="store_true", help="rev12 floor block: one request at a time")
    a = ap.parse_args(argv)
    a.cell_dir.mkdir(parents=True, exist_ok=True)
    fd = raise_fd_limit()
    (a.cell_dir / "fd_limit.json").write_text(json.dumps(fd, indent=1) + "\n")
    if not fd["ok"]:
        print("ABORT_FD_LIMIT: hard RLIMIT_NOFILE below 8192", file=sys.stderr)
        return 3
    if not sock_read_registered():
        print(f"ABORT_SOCK_READ: trace_loadgen.py lacks sock_read={C.SOCK_READ_S} (H5)", file=sys.stderr)
        return 4
    import aiohttp  # noqa: F401  pre-import so replay() start-up is short and the clock bridge tight
    from pdmux_eval.trace_loadgen import read_requests, replay
    reqs = read_requests(a.trace)
    prob = attach_input_ids(reqs, a.x_tokens, a.x_tokens_sha256)
    if prob:
        print(f"ABORT_DATASET: {prob}", file=sys.stderr)
        (a.cell_dir / "ABORT_DATASET").write_text(prob + "\n")
        return 5
    clock = {"perf_at_call": time.perf_counter(), "mono_at_call": time.monotonic(), "epoch_at_call": time.time()}
    (a.cell_dir / "loadgen_clock.json").write_text(json.dumps(clock, indent=1) + "\n")   # survives a kill
    if a.sequential:
        summary = asyncio.run(replay_sequential(replay, a.endpoint, reqs, a.cell_dir / "requests.jsonl", a.ttft_slo_ms,
                                                a.itl_slo_ms, C.CONNECTOR_LIMIT))
    else:
        summary = asyncio.run(replay(a.endpoint, reqs, a.cell_dir / "requests.jsonl", a.ttft_slo_ms,
                                     a.itl_slo_ms, C.CONNECTOR_LIMIT))
    clock["epoch_at_end"] = time.time()
    (a.cell_dir / "loadgen_clock.json").write_text(json.dumps(clock, indent=1) + "\n")
    (a.cell_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
