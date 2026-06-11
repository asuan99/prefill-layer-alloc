"""Single-source loader for sweep grids, device constants, and the saturation
criterion.

Every runner and analyzer reads grids/constants ONLY through this module — there
must be no sweep-grid literals (sm/seq/batch) elsewhere (tests excepted), and no
duplicate saturation / n_blocks / filename logic.

Importable from any project entry point as::

    from shared.sweep_spec import (
        load_spec, sm_grid, batch_grid, seq_grid_prefill, context_grid_decode,
        prefill_chunk_tokens, total_sm, hbm_bw_GBs, saturation_threshold,
        n_blocks, saturation_point, canonical_filename, canonical_device,
        assert_on_sm_grid, ssm_n_heads, ssd_chunk_size,
    )

(`shared.*` is the only import root reachable by both the characterization
runners and serving-eval/plot_motivation.py — see reports/sweep_consistency_audit.md.)
"""
from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import yaml

_SPEC_PATH = Path(__file__).parent / "configs" / "sweep_spec.yaml"


# ---------------------------------------------------------------------------
# Spec loading
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_spec() -> dict[str, Any]:
    """Load and cache configs/sweep_spec.yaml as a plain dict."""
    with open(_SPEC_PATH) as f:
        return yaml.safe_load(f)


# --- Grid / constant accessors (thin, so callers never index the dict) ------
def sm_grid() -> list[int]:
    return list(load_spec()["sm_grid"])


def batch_grid() -> list[int]:
    return list(load_spec()["batch_grid"])


def seq_grid_prefill() -> list[int]:
    return list(load_spec()["seq_grid_prefill"])


def context_grid_decode() -> list[int]:
    return list(load_spec()["context_grid_decode"])


def prefill_chunk_tokens() -> int:
    return int(load_spec()["prefill_chunk_tokens"])


def total_sm() -> int:
    return int(load_spec()["device"]["total_sm"])


def hbm_bw_GBs() -> float:
    return float(load_spec()["device"]["hbm_bw_GBs"])


def fp16_tflops() -> float:
    return float(load_spec()["device"]["fp16_tflops"])


def device_name() -> str:
    """Canonical device tag (underscore form), e.g. ``a100_sxm4_80gb``."""
    return str(load_spec()["device"]["name"])


def saturation_threshold() -> float:
    return float(load_spec()["saturation"]["threshold"])


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------
def _model_entry(model: str) -> dict[str, Any]:
    models = load_spec()["models"]
    if model not in models:
        raise KeyError(
            f"model {model!r} not in sweep_spec.yaml models: {sorted(models)}"
        )
    return models[model]


def ssm_n_heads(model: str) -> int:
    """SSM (Mamba-2) head count for *model*.

    Raises if not yet confirmed (null in spec) — guards against silently using a
    placeholder (e.g. Nemotron-H before Phase 3 registration).
    """
    v = _model_entry(model).get("ssm_n_heads")
    if v is None:
        raise ValueError(
            f"ssm_n_heads for {model!r} is null in sweep_spec.yaml — "
            f"confirm from the HF config and fill it before measuring."
        )
    return int(v)


def ssd_chunk_size(model: str) -> int:
    """Internal SSD chunk_size for *model* (Zamba2/FH1=256, Nemotron-H=128)."""
    return int(_model_entry(model)["ssd_chunk_size"])


# ---------------------------------------------------------------------------
# n_blocks — the ONLY n_blocks implementation
# ---------------------------------------------------------------------------
def n_blocks(model: str, batch: int, seq: int, chunk: Optional[int] = None) -> int:
    """Triton SSD kernel grid block count = batch × ceil(tokens/ssd_chunk) × n_heads.

    The dominant mamba_chunk_scan_combined kernel launches a grid of
    (nchunks, batch, n_heads) blocks.  ``tokens`` = the number of tokens
    processed in one kernel call:

      * full-seq call  → chunk is None → tokens = seq
      * chunked-prefill call → chunk = prefill_chunk_tokens → tokens = chunk

    ``ssd_chunk`` is the model's internal SSD chunk_size (256 for Zamba2/FH1,
    128 for Nemotron-H) — NOT hardcoded 256, since that differs per model.
    Replaces the old model-agnostic ``batch × seq // 4`` formula.
    """
    tokens = seq if chunk is None else min(chunk, seq)
    ssd = ssd_chunk_size(model)
    nchunks = math.ceil(tokens / ssd)
    return batch * nchunks * ssm_n_heads(model)


def cooperative_safe(model: str, batch: int, chunk: int, sm_count: int) -> bool:
    """Cooperative-launch safety: blocks-per-call ≤ sm_count (no deadlock risk)."""
    return n_blocks(model, batch, chunk, chunk=chunk) <= sm_count


# ---------------------------------------------------------------------------
# Filename / device-tag canonicalisation
# ---------------------------------------------------------------------------
def canonical_device(tag: Optional[str] = None) -> str:
    """Normalise any device tag to the canonical underscore form.

    ``a100-sxm4-80gb`` → ``a100_sxm4_80gb``.  ``None`` → spec device name.
    """
    if tag is None:
        return device_name()
    return tag.strip().lower().replace("nvidia ", "").replace(" ", "_").replace("-", "_")


def canonical_filename(kind: str, model: str, device: Optional[str] = None) -> str:
    """Single filename convention: ``{kind}_{model}_{device}.csv`` (underscore tag).

    e.g. canonical_filename("ssm_chunked", "zamba2") →
         ``ssm_chunked_zamba2_a100_sxm4_80gb.csv``
    """
    return f"{kind}_{model}_{canonical_device(device)}.csv"


# ---------------------------------------------------------------------------
# Unified chunked CSV schema (Phase 0 item 5 — diff-0 between ssm/attn v2 CSVs)
# ---------------------------------------------------------------------------
# value_kind labels (recorded as the CSV's first comment line):
#   latency_ms,latency_std_ms = measured (CUDA event)
#   read_bytes,write_bytes,achieved_bw_GBs,theoretical_bw_GBs,bw_util_pct = derived
#   everything else = metadata
CHUNKED_FIELDNAMES: list[str] = [
    "model", "device", "layer_type",
    "seq_len", "batch_size", "sm_count", "sm_ratio_pct",
    "prefill_chunk_tokens", "n_kernel_calls", "n_chunks",
    "n_blocks_per_call", "cooperative_safe",
    "latency_ms", "latency_std_ms",                                  # measured
    "read_bytes", "write_bytes",
    "achieved_bw_GBs", "theoretical_bw_GBs", "bw_util_pct",          # derived
    "state_passing_active", "status",
]

CHUNKED_VALUE_KIND_COMMENT = (
    "# value_kind: latency_ms,latency_std_ms=measured(CUDA event); "
    "read_bytes,write_bytes,achieved_bw_GBs,theoretical_bw_GBs,bw_util_pct=derived; "
    "rest=metadata"
)


def chunked_row(
    *, model: str, layer_type: str, seq_len: int, batch_size: int, sm_count: int,
    prefill_chunk_tokens: int, latency_ms: float, latency_std_ms: float,
    read_bytes: float, write_bytes: float, cooperative_safe: bool,
    state_passing_active: bool, status: str = "ok", device: Optional[str] = None,
) -> dict:
    """Build one unified-schema chunked CSV row (used by both ssm/attn v2 runners).

    Derives sm_ratio_pct, n_kernel_calls, n_chunks, n_blocks_per_call, and the
    bandwidth columns from the single spec source so the two v2 CSVs are identical
    in schema and derivation.
    """
    n_calls = math.ceil(seq_len / prefill_chunk_tokens)
    lat_s = latency_ms / 1000.0
    bw = hbm_bw_GBs()
    achieved = ((read_bytes + write_bytes) / lat_s / 1e9) if lat_s > 0 else 0.0
    return {
        "model": model,
        "device": canonical_device(device),
        "layer_type": layer_type,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "sm_count": sm_count,
        "sm_ratio_pct": round(100.0 * sm_count / total_sm(), 2),
        "prefill_chunk_tokens": prefill_chunk_tokens,
        "n_kernel_calls": n_calls,
        "n_chunks": n_calls,
        "n_blocks_per_call": n_blocks(model, batch_size, prefill_chunk_tokens,
                                      chunk=prefill_chunk_tokens),
        "cooperative_safe": cooperative_safe,
        "latency_ms": latency_ms,
        "latency_std_ms": latency_std_ms,
        "read_bytes": read_bytes,
        "write_bytes": write_bytes,
        "achieved_bw_GBs": round(achieved, 3),
        "theoretical_bw_GBs": bw,
        "bw_util_pct": round(100.0 * achieved / bw, 3) if bw else 0.0,
        "state_passing_active": state_passing_active,
        "status": status,
    }


# ---------------------------------------------------------------------------
# SM-grid guard — fail loudly on snapping
# ---------------------------------------------------------------------------
def assert_on_sm_grid(requested) -> None:
    """Raise if any requested SM count is not exactly on sm_grid.

    Prevents silent Green-Context snapping (requested != preset) from polluting
    measurements.  ``requested`` may be an int or an iterable of ints.
    """
    grid = set(sm_grid())
    reqs = [requested] if isinstance(requested, int) else list(requested)
    off = sorted({int(r) for r in reqs} - grid)
    if off:
        raise ValueError(
            f"requested SM counts {off} are off sm_grid {sorted(grid)} — "
            f"Green Context would snap them. Use sweep_spec sm_grid() only."
        )


# ---------------------------------------------------------------------------
# saturation_point — the ONLY saturation implementation
# ---------------------------------------------------------------------------
# Recognised latency column names across the CSV schemas (Phase 0, item 5).
_LATENCY_COLS = ("latency_ms", "latency_per_step_ms", "latency", "latency_mean_ms")


def _resolve(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def saturation_point(df) -> Optional[int]:
    """SM count at which marginal throughput gain drops below the spec threshold.

    Single implementation replacing the three near-duplicates
    (plot_saturation, analyze_decode_sat, plot_motivation).

    *df* is one (layer_type, seq/context, batch) group.  Throughput is derived
    as 1/latency from whichever latency column is present; ``sm_ratio`` is used
    if present, else derived as ``sm_count / total_sm``.  Marginal gain is
    measured per 10% SM and normalised to peak throughput.

    Returns the saturating ``sm_count`` (int), the max sm_count if no saturation
    is detected, or ``None`` if the group has fewer than 2 usable points.
    """
    df = df.copy()
    lat_col = _resolve(df, _LATENCY_COLS)
    if lat_col is None or "sm_count" not in df.columns:
        return None

    df = df[df[lat_col].notna() & (df[lat_col] > 0)]
    if "sm_ratio" in df.columns and df["sm_ratio"].notna().any():
        df["_ratio"] = df["sm_ratio"].astype(float)
    elif "sm_ratio_pct" in df.columns and df["sm_ratio_pct"].notna().any():
        df["_ratio"] = df["sm_ratio_pct"].astype(float) / 100.0
    else:
        df["_ratio"] = df["sm_count"].astype(float) / float(total_sm())

    df = df.sort_values("_ratio")
    if len(df) < 2:
        return None

    ratios = df["_ratio"].values
    tps = (1.0 / df[lat_col].astype(float)).values
    tp_max = tps.max()
    if tp_max <= 0:
        return int(df["sm_count"].max())
    tps_norm = tps / tp_max

    thr = saturation_threshold()
    sm_vals = df["sm_count"].astype(int).values
    for i in range(1, len(ratios)):
        d_sm = ratios[i] - ratios[i - 1]
        if d_sm <= 0:
            continue
        gain_per_10pct = (tps_norm[i] - tps_norm[i - 1]) / d_sm * 0.10
        if gain_per_10pct < thr:
            return int(sm_vals[i - 1])
    return int(sm_vals[-1])
