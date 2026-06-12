"""
E0 — analytical wave table (NO GPU required).

Purpose
-------
Pre-compute, from the corrected grid formula alone, the grid-saturation
prediction that E2 will test empirically. Two products land in results_v2/e0/:

  1. wave_table_{device}.csv
       per (model, batch, seq, chunk-granularity, SM): n_blocks, waves,
       grid_sat_sm and — the payload — whether the SSM grid *starves* the SMs
       (free SM remains → layer-type asymmetry has headroom) or *over-subscribes*
       them (no free SM → under the GRID mechanism the asymmetry is dead).
       This locates the batch threshold where asymmetry dies, BEFORE measuring.

  2. temporal_overhead_budget_{device}.csv
       the v1 layer-boundary swap cost (~7.8 µs, MEASURED, re-read from the
       archived Stage-2 result) vs an ASSUMED per-layer compute time, giving the
       overhead fraction a Zamba2 temporal policy would pay. The per-layer time
       is an assumption (E1 has not measured it yet) and is labelled as such.

Everything E0 emits is DERIVED (analytical) except metadata inputs and the
re-used swap measurement. All block/wave math comes from the single source
experiments/common/wave_model.py — this script never recomputes it.

The SLM configs are PROVISIONAL (models not in hf_cache); each row carries a
``config_status`` column so the provisional inputs are never mistaken for
verified ones.

Usage
-----
    python experiments/e0_analytical/run_wave_table.py
    python experiments/e0_analytical/run_wave_table.py --models zamba2_1.2b \\
        --batches 1 8 32 128 --seq-lens 2048 8192 --assumed-layer-ms 0.05
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

# --- path bootstrap: workspace/ (shared.*) and characterization/ (experiments.*)
_here = os.path.dirname(os.path.abspath(__file__))
_CHAR = os.path.abspath(os.path.join(_here, "..", ".."))
_WORKSPACE = os.path.abspath(os.path.join(_here, "..", "..", ".."))
for _p in (_WORKSPACE, _CHAR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shared import sweep_spec as ss
from shared.loaders import get_model_config
from experiments.common import wave_model as wm
from experiments.common.labels import Label, write_labeled_csv

# Default v2 axes (prompt E0 spec). batch grid adds the high points (128) that
# the v1 batch_grid lacked — the batch axis is what makes the grid mechanism
# falsifiable.
DEFAULT_MODELS = ["zamba2_1.2b", "falcon_h1_1.5b"]
DEFAULT_BATCHES = [1, 8, 32, 128]
DEFAULT_CHUNKS = ["256", "512", "full"]          # outer prefill-chunk granularity
DEFAULT_SEQ_LENS = [2048, 8192]                  # needed to resolve chunk="full"
# SM sweep: Green Context preset grid from sweep_spec.
DEFAULT_RESULTS = Path(_CHAR) / "results_v2" / "e0"

# Archived v1 Stage-2 swap measurement (MEASURED, reused — see ARCHIVE_NOTE).
_ARCHIVE_STAGE2 = (
    Path(_CHAR) / "archive" / "v1_7b_saturation" / "results" / "stage2"
)
_SWAP_FALLBACK_US = 7.8  # ssm<->attn layer-boundary swap, sync included (v1)


# ---------------------------------------------------------------------------
# Wave table
# ---------------------------------------------------------------------------

def _tokens_per_call(chunk: str, seq: int) -> int:
    """Resolve the outer prefill-chunk granularity to a token count per call."""
    if chunk == "full":
        return seq
    return min(int(chunk), seq)


def build_wave_table(
    models: list[str],
    batches: list[int],
    seq_lens: list[int],
    chunks: list[str],
    sm_grid: list[int],
    total_sm: int,
    blocks_per_sm: int,
    device: str,
) -> list[dict]:
    rows: list[dict] = []
    for model in models:
        n_heads = ss.ssm_n_heads(model)        # from config, not hardcoded
        ssd_chunk = ss.ssd_chunk_size(model)   # internal SSD chunk (=256)
        status = get_model_config(model).get("config_status", "verified")
        for batch in batches:
            for seq in seq_lens:
                for chunk in chunks:
                    tpc = _tokens_per_call(chunk, seq)
                    n_calls = math.ceil(seq / tpc)
                    # per-call grid block count — what occupies SMs at once.
                    nb = wm.n_blocks(batch=batch, seq=tpc,
                                     n_heads=n_heads, chunk=ssd_chunk)
                    grid_sat = wm.grid_saturation_sm(nb, blocks_per_sm=blocks_per_sm)
                    grid_sat_pred = min(grid_sat, total_sm)
                    grid_starved = grid_sat < total_sm  # free SM ⇒ asymmetry alive
                    for sm in sm_grid:
                        rows.append({
                            "model": model,
                            "device": device,
                            "config_status": status,
                            "batch": batch,
                            "seq": seq,
                            "chunk_granularity": chunk,
                            "ssd_chunk": ssd_chunk,
                            "n_heads": n_heads,
                            "sm_count": sm,
                            "blocks_per_sm": blocks_per_sm,
                            "tokens_per_call": tpc,
                            "n_calls": n_calls,
                            "n_blocks": nb,
                            "waves": round(wm.waves(nb, sm, blocks_per_sm), 4),
                            "grid_sat_sm": grid_sat,
                            "grid_sat_sm_pred": grid_sat_pred,
                            "grid_starved": grid_starved,
                            "asym_headroom": grid_starved,
                        })
    return rows


_WAVE_SCHEMA = {
    "model": Label.METADATA,
    "device": Label.METADATA,
    "config_status": Label.METADATA,
    "batch": Label.METADATA,
    "seq": Label.METADATA,
    "chunk_granularity": Label.METADATA,
    "ssd_chunk": Label.METADATA,
    "n_heads": Label.METADATA,
    "sm_count": Label.METADATA,
    "blocks_per_sm": Label.METADATA,      # occupancy assumption value
    "tokens_per_call": Label.DERIVED,
    "n_calls": Label.DERIVED,
    "n_blocks": Label.DERIVED,
    "waves": Label.DERIVED,
    "grid_sat_sm": Label.DERIVED,
    "grid_sat_sm_pred": Label.DERIVED,
    "grid_starved": Label.DERIVED,
    "asym_headroom": Label.DERIVED,
}


# ---------------------------------------------------------------------------
# Temporal overhead budget
# ---------------------------------------------------------------------------

def _read_swap_us() -> tuple[float, str]:
    """Return (swap_us, source). Reuse the archived v1 Stage-2 measurement."""
    cands = sorted(_ARCHIVE_STAGE2.glob("ctx_switch_overhead_*.json"))
    if cands:
        try:
            d = json.loads(cands[0].read_text())
            st = d.get("single_transitions", {})
            keys = [k for k in st if k.endswith("_sync_yes")]
            if keys:
                vals = [st[k]["mean_us"] for k in keys if "mean_us" in st[k]]
                if vals:
                    return float(sum(vals) / len(vals)), f"archive_v1:{cands[0].name}"
        except Exception:
            pass
    return _SWAP_FALLBACK_US, "fallback_constant_7.8us"


def build_overhead_budget(
    models: list[str],
    swap_us: float,
    swap_source: str,
    assumed_layer_ms: float,
    device: str,
) -> list[dict]:
    """Compare swap cost (measured, v1) to an ASSUMED per-layer compute time.

    per_swap_overhead_pct = swap_us / (assumed_layer_us)            (derived)
    full_model_overhead_pct = (n_swaps × swap_us) / (n_layers × layer_us)
        with n_swaps ≈ n_layers (worst case: one boundary swap per layer)  (derived)
    """
    layer_us = assumed_layer_ms * 1000.0
    rows: list[dict] = []
    for model in models:
        cfg = get_model_config(model)
        n_layers = int(cfg.get("num_layers", 0))
        status = cfg.get("config_status", "verified")
        per_swap_pct = 100.0 * swap_us / layer_us if layer_us > 0 else float("nan")
        # worst-case temporal policy: one swap per layer boundary.
        n_swaps = n_layers
        full_pct = (
            100.0 * (n_swaps * swap_us) / (n_layers * layer_us)
            if (n_layers > 0 and layer_us > 0) else float("nan")
        )
        rows.append({
            "model": model,
            "device": device,
            "config_status": status,
            "num_layers": n_layers,
            "swap_cost_source": swap_source,
            "assumption": f"per_layer_compute={assumed_layer_ms}ms (PRE-E1 ASSUMPTION)",
            "swap_us": round(swap_us, 3),
            "assumed_layer_us": round(layer_us, 3),
            "n_swaps_worstcase": n_swaps,
            "per_swap_overhead_pct": round(per_swap_pct, 4),
            "full_model_overhead_pct": round(full_pct, 4),
        })
    return rows


_BUDGET_SCHEMA = {
    "model": Label.METADATA,
    "device": Label.METADATA,
    "config_status": Label.METADATA,
    "num_layers": Label.METADATA,
    "swap_cost_source": Label.METADATA,
    "assumption": Label.METADATA,
    "swap_us": Label.MEASURED,            # reused v1 CUDA-bracketed measurement
    "assumed_layer_us": Label.DERIVED,    # assumption (pre-E1)
    "n_swaps_worstcase": Label.DERIVED,
    "per_swap_overhead_pct": Label.DERIVED,
    "full_model_overhead_pct": Label.DERIVED,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E0 analytical wave table (no GPU)")
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--batches", nargs="+", type=int, default=DEFAULT_BATCHES)
    p.add_argument("--seq-lens", nargs="+", type=int, default=DEFAULT_SEQ_LENS)
    p.add_argument("--chunks", nargs="+", default=DEFAULT_CHUNKS,
                   help='prefill-chunk granularity: 256 512 full')
    p.add_argument("--blocks-per-sm", type=int, default=1,
                   help="occupancy assumption for waves/grid_sat (DERIVED)")
    p.add_argument("--assumed-layer-ms", type=float, default=0.05,
                   help="ASSUMED per-layer compute time for the overhead budget "
                        "(E1 has not measured this yet)")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = ss.canonical_device()        # a100_sxm4_80gb
    total_sm = ss.total_sm()
    sm_grid = ss.sm_grid()

    print(f"E0 analytical wave table  (device={device}, total_sm={total_sm})")
    print(f"  models  : {args.models}")
    print(f"  batches : {args.batches}")
    print(f"  seq_lens: {args.seq_lens}")
    print(f"  chunks  : {args.chunks}")
    print(f"  sm_grid : {sm_grid}")

    # warn loudly if any model config is provisional
    for m in args.models:
        if get_model_config(m).get("config_status") == "provisional":
            print(f"  ⚠ {m}: PROVISIONAL config (not in hf_cache) — verify n_heads "
                  f"against HF config.json before trusting grid predictions.")

    wave_rows = build_wave_table(
        models=args.models, batches=args.batches, seq_lens=args.seq_lens,
        chunks=args.chunks, sm_grid=sm_grid, total_sm=total_sm,
        blocks_per_sm=args.blocks_per_sm, device=device,
    )
    wave_csv = args.output_dir / f"wave_table_{device}.csv"
    write_labeled_csv(wave_csv, wave_rows, _WAVE_SCHEMA)
    print(f"\n  wrote {len(wave_rows)} rows -> {wave_csv}")

    swap_us, swap_source = _read_swap_us()
    budget_rows = build_overhead_budget(
        models=args.models, swap_us=swap_us, swap_source=swap_source,
        assumed_layer_ms=args.assumed_layer_ms, device=device,
    )
    budget_csv = args.output_dir / f"temporal_overhead_budget_{device}.csv"
    write_labeled_csv(budget_csv, budget_rows, _BUDGET_SCHEMA)
    print(f"  wrote {len(budget_rows)} rows -> {budget_csv}  "
          f"(swap_us={swap_us:.2f} from {swap_source})")

    # --- console summary: the batch threshold where asymmetry dies -----------
    print("\n  Grid-mechanism prediction (chunk=256, the worst case for grid):")
    print("  model            batch  n_blocks  grid_sat_sm  asym_headroom?")
    for model in args.models:
        for batch in args.batches:
            sub = [r for r in wave_rows
                   if r["model"] == model and r["batch"] == batch
                   and r["chunk_granularity"] == "256"
                   and r["seq"] == args.seq_lens[0]]
            if sub:
                r = sub[0]
                print(f"  {model:16s} {batch:5d}  {r['n_blocks']:8d}  "
                      f"{r['grid_sat_sm']:11d}  {r['asym_headroom']}")
    print("\n  → asym_headroom=False means the SSM grid already fills all SMs at "
          "this batch:\n    under the GRID mechanism layer-type SM asymmetry has no "
          "room, so E2\n    should show sat_sm→total_sm as batch grows. E2 tests this.")


if __name__ == "__main__":
    main()
