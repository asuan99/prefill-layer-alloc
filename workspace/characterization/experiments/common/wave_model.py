"""
wave_model.py — SINGLE SOURCE for n_blocks / waves / grid-saturation math (v2).

Every v2 experiment script imports these three functions; NOTHING else in the v2
tree recomputes block or wave counts. (The v1 archive and src/profiling/
wave_estimator.py keep their own copies — those are frozen / immutable and are
NOT the v2 source of truth.)

Corrected n_blocks formula
--------------------------
The dominant Mamba-2 prefill kernel (``mamba_chunk_scan_combined``) launches a
grid of ``batch × ceil(tokens/ssd_chunk) × n_heads`` thread blocks — one CTA per
(batch, head, ssd-chunk). The old model-agnostic ``batch*seq//4`` formula ignored
n_heads and the chunk divisor, under-counting by ~28× for Zamba2 (n_heads=112).

Labelling
---------
Every value returned here is **derived** (analytical, not measured). Call sites
MUST record these columns with Label.DERIVED (see labels.py). Never present a
wave_model output as a measured quantity.

Inputs come from config, never hardcoded
-----------------------------------------
``n_heads`` and the chunk divisor are passed in by the caller, which reads them
from ``configs/models.yaml`` (or ``shared.sweep_spec``). This module contains no
model constants.
"""

from __future__ import annotations

import math

__all__ = ["n_blocks", "waves", "grid_saturation_sm"]


def n_blocks(batch: int, seq: int, n_heads: int, chunk: int = 256) -> int:
    """Mamba-2 SSD kernel grid block count for one kernel call.

    grid = batch × ceil(seq / chunk) × n_heads

    Args:
        batch:   batch size for this kernel call.
        seq:     number of tokens processed in this kernel call (tokens-per-call;
                 for chunked prefill this is the prefill-chunk granularity, for a
                 full-sequence call it is the whole prefill length).
        n_heads: SSM (Mamba-2) head count — read from configs, NOT hardcoded
                 (e.g. zamba2_1.2b=64, falcon_h1_1.5b=16; the 7B Zamba2 was 112).
        chunk:   the model's *internal* SSD chunk_size (Zamba2/Falcon-H1 = 256).
                 This is the divisor, distinct from the outer prefill-chunk
                 granularity passed as ``seq``.

    Returns:
        int n_blocks (DERIVED). Always ≥ 1.
    """
    if batch < 1 or seq < 1 or n_heads < 1 or chunk < 1:
        raise ValueError(
            f"n_blocks requires positive ints: batch={batch} seq={seq} "
            f"n_heads={n_heads} chunk={chunk}"
        )
    return batch * math.ceil(seq / chunk) * n_heads


def waves(n_blocks: int, sm_count: int, blocks_per_sm: int = 1) -> float:
    """Wave count for grid-saturation analysis (DERIVED).

    waves = n_blocks / (sm_count × blocks_per_sm)

    waves ≤ 1 means a single kernel call cannot fill every SM — the grid is the
    binding constraint (the "grid mechanism"). waves ≫ 1 means the SMs are
    over-subscribed and the kernel is limited by something else (e.g. bandwidth).

    Args:
        n_blocks:      grid block count from :func:`n_blocks`.
        sm_count:      effective SM count (after any Green Context restriction).
        blocks_per_sm: assumed resident CTAs per SM (occupancy assumption — this
                       is an ASSUMPTION, hence the whole result is DERIVED).

    Returns:
        float waves (DERIVED).
    """
    denom = max(1, sm_count) * max(1, blocks_per_sm)
    return n_blocks / denom


def grid_saturation_sm(n_blocks: int, blocks_per_sm: int = 1) -> int:
    """Minimum SM count at which ``waves == 1`` (DERIVED).

    This is the grid-mechanism PREDICTION of the saturation point: if saturation
    is grid-bound, throughput should stop improving once SM ≥ this value, because
    beyond it the single grid no longer has blocks to occupy the extra SMs.

    Compare against the measured E2 saturation point:
      * measured sat_sm ≈ grid_saturation_sm AND grid_saturation_sm grows with
        batch toward total_sm  → grid mechanism (G1 GRID).
      * measured sat_sm batch-invariant and well below grid_saturation_sm, with
        BW util ≥ ~85% at saturation → bandwidth mechanism (G1 BW).

    Args:
        n_blocks:      grid block count from :func:`n_blocks`.
        blocks_per_sm: occupancy assumption (see :func:`waves`).

    Returns:
        int minimum saturating SM count (DERIVED). Always ≥ 1.
    """
    return max(1, math.ceil(n_blocks / max(1, blocks_per_sm)))
