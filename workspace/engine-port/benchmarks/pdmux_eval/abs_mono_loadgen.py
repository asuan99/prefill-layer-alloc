"""trace_loadgen.replay + ABSOLUTE CLOCK_MONOTONIC client stamps (diag Job item (v), 2026-10-28).

WHY A WRAPPER.  `trace_loadgen.py` is shared by N2/N3/E2/G0/M3 and its sha256 is pinned by a sealed registration
(`results/multigpu/REGISTRATION_rev12_SEAL.txt`), so it is left byte-identical.  This module calls the real `replay`
unchanged and adds, per request record,

    t_sent_mono, t_finished_mono, token_t_mono (one per observed token)      [seconds, CLOCK_MONOTONIC]

and to the summary `benchmark_started_mono` and `abs_mono_offset_s`.

HOW (exact, not a second clock).  `replay` stamps everything with `time.perf_counter()` and writes times relative to
its local `benchmark_started` (the FIRST perf_counter read inside `replay`; pinned by a source test).  While `replay`
runs, the module-level name `time` in trace_loadgen is bound to a pass-through shim that remembers that first read.
Then  sent = benchmark_started + sent_s,  token_0 = sent + ttft_ms/1e3,  token_k = token_{k-1} + itl_ms/1e3,
finished = benchmark_started + completion_s  -- the same perf_counter values TTFT/ITL were computed from (float
round-off only) -- shifted by ONE offset `clock_gettime(CLOCK_MONOTONIC) - perf_counter()` read before the run
(on Linux perf_counter IS CLOCK_MONOTONIC, the offset is ~0; it is recorded so the pairing with the server's
phase-event clock is checked, not assumed).  Record lines keep trace_loadgen's serialisation
(`separators=(",", ":"), sort_keys=True`).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List

from . import trace_loadgen as TL


class _FirstPerfCounter:
    """Pass-through for the `time` module that records the first perf_counter() value."""

    def __init__(self, real):
        self._real = real
        self.first = None

    def perf_counter(self):
        value = self._real.perf_counter()
        if self.first is None:
            self.first = value
        return value

    def __getattr__(self, name):
        return getattr(self._real, name)


def abs_fields(rec: Dict[str, object], started: float, offset: float) -> Dict[str, object]:
    sent = started + float(rec["sent_s"])
    out: Dict[str, object] = {"t_sent_mono": sent + offset,
                              "t_finished_mono": started + float(rec["completion_s"]) + offset}
    tokens: List[float] = []
    if rec.get("ttft_ms") is not None and int(rec.get("observed_output_tokens") or 0) > 0:
        t = sent + float(rec["ttft_ms"]) / 1000.0
        tokens.append(t)
        for itl in rec.get("token_itl_ms") or []:
            t = t + float(itl) / 1000.0
            tokens.append(t)
    out["token_t_mono"] = [x + offset for x in tokens]
    return out


async def replay_abs_mono(endpoint, requests, output_path: Path, ttft_slo_ms, itl_slo_ms, connector_limit=None):
    offset = time.clock_gettime(time.CLOCK_MONOTONIC) - time.perf_counter()
    shim = _FirstPerfCounter(TL.time)
    real = TL.time
    TL.time = shim
    try:
        summary = await TL.replay(endpoint, requests, output_path, ttft_slo_ms, itl_slo_ms, connector_limit)
    finally:
        TL.time = real
    if shim.first is None:
        raise RuntimeError("replay made no perf_counter read; cannot anchor absolute stamps")
    lines = []
    for line in Path(output_path).read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        rec.update(abs_fields(rec, shim.first, offset))
        lines.append(json.dumps(rec, separators=(",", ":"), sort_keys=True))
    Path(output_path).write_text("".join(x + "\n" for x in lines), encoding="utf-8")
    summary["benchmark_started_mono"] = float(shim.first + offset)
    summary["abs_mono_offset_s"] = float(offset)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--ttft-slo-ms", type=float, required=True)
    parser.add_argument("--itl-slo-ms", type=float, required=True)
    parser.add_argument("--connector-limit", type=int, default=None)
    args = parser.parse_args()
    summary = asyncio.run(replay_abs_mono(args.endpoint, TL.read_requests(args.trace), args.output,
                                          args.ttft_slo_ms, args.itl_slo_ms, args.connector_limit))
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
