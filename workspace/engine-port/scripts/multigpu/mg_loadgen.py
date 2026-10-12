#!/usr/bin/env python3
"""Multi-GPU load generator wrapper around pdmux_eval.trace_loadgen.replay (PREREG §1, H5, H6).

trace_loadgen.py itself is NOT modified (N2/N3/E2 use it).  This wrapper:
  * raises RLIMIT_NOFILE soft to >= 8192 (H6) BEFORE the event loop exists, and writes the achieved
    limits to <cell>/fd_limit.json; exit 3 (ABORT_FD_LIMIT) when the hard limit is below 8192;
  * records a clock bridge `loadgen_clock.json` = {perf_at_call, mono_at_call, epoch_at_call}
    taken immediately before replay().  replay() sets `benchmark_started = perf_counter()` a few ms
    later, so `perf_at_call + sent_s` is the CLOCK_MONOTONIC time of each send (the clock of the
    /proc sampler and of the engine telemetry) to within the asyncio start-up (tens of ms);
  * refuses to run if trace_loadgen.py does not carry `sock_read=600` (H5: the registered value; the
    MGP-35 CLIENT_ERROR thresholds were computed for exactly this timeout);
  * passes --connector-limit 0 (unlimited, §1).

usage: mg_loadgen.py --endpoint URL --trace T.jsonl --cell-dir DIR [--ttft-slo-ms 3000 --itl-slo-ms 50]
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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--cell-dir", type=Path, required=True)
    ap.add_argument("--ttft-slo-ms", type=float, default=3000.0)   # SLO only feeds the descriptive summary
    ap.add_argument("--itl-slo-ms", type=float, default=50.0)
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
    clock = {"perf_at_call": time.perf_counter(), "mono_at_call": time.monotonic(), "epoch_at_call": time.time()}
    (a.cell_dir / "loadgen_clock.json").write_text(json.dumps(clock, indent=1) + "\n")   # survives a kill
    summary = asyncio.run(replay(a.endpoint, reqs, a.cell_dir / "requests.jsonl", a.ttft_slo_ms,
                                 a.itl_slo_ms, C.CONNECTOR_LIMIT))
    clock["epoch_at_end"] = time.time()
    (a.cell_dir / "loadgen_clock.json").write_text(json.dumps(clock, indent=1) + "\n")
    (a.cell_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
