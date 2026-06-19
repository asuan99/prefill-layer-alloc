"""Step-time LUT for the queue simulator, built from an E5 full CSV (GPU-free).

ASSUMPTION (important): the CSV should be a SINGLE-CHUNK LUT — produced with
``--prefill-mode full --prefill-tokens <chunk>`` (n_chunks=1) — so ``concurrent_ms``
= time for ONE prefill chunk overlapped with ONE decode step at the given
decode_batch. That matches the chunked-prefill granularity the simulator uses
(1 chunk + 1 decode step per iteration). A multi-chunk CSV (tokens>chunk) measures
a *whole* prefill ∥ one decode step (wrong granularity) — the loader warns.

policy → backend row:
  co_schedule     → two_stream         (no reservation, dynamic share)
  static          → green_ctx (best f) (fixed split)
  dynamic_protect → green_ctx_protect  (decode reserved its E3 floor)
"""
from __future__ import annotations

import os

_POLICY_BACKEND = {
    "fused": "two_stream",          # vLLM/SGLang default: prefill+decode in ONE mixed batch
    "co_schedule": "two_stream",    # MPS-style: two concurrent streams, dynamic SM share
    "static": "green_ctx",          # static SM partition (fixed f)
    "dynamic_protect": "green_ctx_protect",  # dynamic split: decode reserved its E3 floor
}


def _nearest(b, grid):
    return min(grid, key=lambda g: abs(g - b))


class LatencyModel:
    def __init__(self, csv_path):
        import pandas as pd
        df = pd.read_csv(csv_path, skiprows=1)
        df.columns = [c.split("__")[0] for c in df.columns]
        for c in ("decode_batch", "n_chunks", "concurrent_ms", "solo_decode_ms", "solo_prefill_ms"):
            if c in df:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        self.path = csv_path
        self.df = df[df.status == "ok"].copy()
        # fused step = solo_prefill + frac*solo_decode. frac=0 optimistic (decode free,
        # wrong for dec=attn KV read); frac=1 conservative (full measured decode cost,
        # over-counts weight-load sharing). Truth in between; a real fused kernel is unmeasured.
        self.fused_decode_frac = 1.0
        pb = self.df["prefill_batch"].dropna() if "prefill_batch" in self.df else None
        self.prefill_batch = int(pb.mode().iloc[0]) if pb is not None and len(pb) else 1
        if "n_chunks" in df and (df["n_chunks"].dropna().max() or 0) > 1:
            print(f"  ⚠ LUT {os.path.basename(csv_path)} is multi-chunk "
                  f"(n_chunks={int(df['n_chunks'].dropna().max())}); simulator wants a single-chunk "
                  f"LUT (--prefill-tokens=chunk). Step times illustrative only.")
        self.backends = sorted(self.df.backend.unique())
        # grids[(backend, pf, dec, ctx)] -> {decode_batch: row}   (green_ctx = best-f per cell)
        self.grids = {}
        for _, r in self.df.iterrows():
            bk = "green_ctx" if r.backend == "green_ctx" else r.backend
            key = (bk, r.prefill_layer, r.decode_layer, int(r.context_len))
            g = self.grids.setdefault(key, {})
            b = int(r.decode_batch)
            if bk == "green_ctx" and b in g and g[b].concurrent_ms <= r.concurrent_ms:
                continue                                  # keep the faster frac
            g[b] = r

    def _row(self, backend, pf, dec, b, ctx):
        g = self.grids.get((backend, pf, dec, ctx))
        if not g:
            return None
        return g[_nearest(max(b, 1), list(g))]

    def streams(self, policy, pf, dec, decode_batch, ctx):
        """(prefill_stream_ms, decode_stream_ms, solo_prefill_ms, solo_decode_ms) for the
        backend — for the DECOUPLED scheduler (decode runs on its own SMs, decode ITL =
        decode_stream_ms, independent of the prefill stream)."""
        b = max(int(decode_batch), 0)
        row = self._row(_POLICY_BACKEND[policy], pf, dec, b, ctx)
        if row is None:
            row = self._row("two_stream", pf, dec, b, ctx)
        if row is None:
            return None
        return (float(row.prefill_stream_ms), float(row.decode_stream_ms),
                float(row.solo_prefill_ms), float(row.solo_decode_ms))

    def step_ms(self, policy, pf, dec, decode_batch, ctx, prefill_active):
        """Latency of one serving iteration. Returns (ms, note); note flags fallbacks."""
        b = max(int(decode_batch), 0)
        row = self._row(_POLICY_BACKEND[policy], pf, dec, b, ctx)
        note = ""
        if row is None and policy in ("dynamic_protect", "static"):
            row = self._row("two_stream", pf, dec, b, ctx)        # no green row here → co_schedule
            note = f"{policy}_missing→co_schedule"
        if row is None:
            return 0.0, "no_lut"
        if prefill_active and policy == "fused":
            # decode rides in the prefill batch; add measured decode cost (KV read) × frac
            return float(row.solo_prefill_ms) + self.fused_decode_frac * float(row.solo_decode_ms), "fused"
        if prefill_active and b == 0:
            return float(row.solo_prefill_ms), (note + "|prefill_only").lstrip("|")
        if prefill_active:
            return float(row.concurrent_ms), note
        return float(row.solo_decode_ms), note                    # decode-only step
