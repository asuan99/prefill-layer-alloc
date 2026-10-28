#!/usr/bin/env python3
"""Per-request natural output (ignore_eos=False) for the policy harness WITHOUT touching trace_loadgen.py.

`benchmarks/pdmux_eval/trace_loadgen.py` is pinned by a sealed registration (multigpu rev12) and hard-codes
`"ignore_eos": True` in the payload it builds.  This wrapper calls the real `replay` (or engine-porter's
`abs_mono_loadgen.replay_abs_mono`, which itself wraps `replay`) unchanged and, while it runs, swaps `aiohttp.ClientSession`
for a thin proxy whose `post()` flips `sampling_params.ignore_eos` to False for the requests whose trace item carries
`"ignore_eos": false` (W3, decision 47).  The payload dict is mutated BEFORE it is sent and it is the same object `replay`
later hands to `StreamState.classify`, so a natural short stream is not mis-classified OUTPUT_SHORT.  Items without the key
(W1, every old trace) keep ignore_eos True.  With no natural item and abs_mono off this is a plain `replay` call.

Composable: `replay_natural(..., abs_mono=True)` goes through `abs_mono_loadgen.replay_abs_mono` (absolute CLOCK_MONOTONIC
stamps for the diag Job) with the same natural-output handling.  abs_mono_loadgen.py is not modified.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ENG = Path(os.path.abspath(__file__)).parents[3]
if str(ENG / "benchmarks") not in sys.path:
    sys.path.insert(0, str(ENG / "benchmarks"))
from pdmux_eval import abs_mono_loadgen as AM  # noqa: E402
from pdmux_eval import trace_loadgen as TL  # noqa: E402


def natural_ids(requests):
    return {str(r["request_id"]) for r in requests if r.get("ignore_eos") is False}


class _NaturalSession:
    """Proxy around an aiohttp.ClientSession; only `post` is adapted."""

    def __init__(self, real, natural):
        self._real, self._natural = real, natural

    async def __aenter__(self):
        await self._real.__aenter__()
        return self

    async def __aexit__(self, *exc):
        return await self._real.__aexit__(*exc)

    def post(self, url, *a, **kw):
        payload = kw.get("json")
        if payload is not None and str(payload.get("rid")) in self._natural:
            payload["sampling_params"]["ignore_eos"] = False
        return self._real.post(url, *a, **kw)

    def __getattr__(self, name):
        return getattr(self._real, name)


async def replay_natural(endpoint, requests, output_path, ttft_slo_ms, itl_slo_ms, connector_limit=0, abs_mono=False):
    """connector_limit=0 (unlimited): the aiohttp default of 100 would queue requests IN THE CLIENT once the system holds more
    than ~100 streams (HI phase at 1.5 x lambda*), distorting TTFT and the controller input (audit M1)."""
    natural = natural_ids(requests)
    inner = AM.replay_abs_mono if abs_mono else TL.replay
    if not natural:
        return await inner(endpoint, requests, output_path, ttft_slo_ms, itl_slo_ms, connector_limit)
    import aiohttp
    real_cls = aiohttp.ClientSession
    aiohttp.ClientSession = lambda *a, **kw: _NaturalSession(real_cls(*a, **kw), natural)
    try:
        return await inner(endpoint, requests, output_path, ttft_slo_ms, itl_slo_ms, connector_limit)
    finally:
        aiohttp.ClientSession = real_cls
