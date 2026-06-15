"""
E2 — batch-swept SM saturation sweep (A100). **The G1 core.**

Measures the SSM and attention prefill saturation behaviour across the SM grid
AND the batch axis, with bandwidth co-measured at every point. The batch axis is
what makes the mechanism falsifiable:

  * bandwidth mechanism → saturation SM is batch-invariant and BW util ≥ ~85% at
    saturation (the SMs fill but HBM is the wall).
  * grid mechanism      → saturation SM climbs toward total_sm as batch grows
    (more blocks are needed to fill all SMs), tracking E0's grid_sat_sm.

Layers: prefill_ssm (token-mixer only `ssm`, and full layer `ssm_full`),
        prefill_attn (`attn`, swept over KV-history/context length).
Sweep:  SM{14..108 grid} × batch{1,8,32,128} × chunk{256,512}.

Isolation: one subprocess PER SM level, with a wall-clock timeout (a low-SM SSM
cooperative kernel can deadlock or illegal-access). A failed/timed-out SM level
is recorded status=failed and never aborts the whole sweep.

Outputs → results_v2/e2/
  sm_sweep_{model}_{device}.csv       per (layer,batch,chunk,ctx,sm) point
      MEASURED: latency_ms, latency_std_ms, achieved_bw_gbs
      DERIVED : bw_util_pct, waves
  saturation_ci_{model}_{device}.csv  per (layer,batch,chunk,ctx) group
      DERIVED : sat_sm_point, sat_sm_ci_low, sat_sm_ci_high, grid_sat_sm(E0)
      MEASURED: bw_util_pct_at_sat
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

_here = os.path.dirname(os.path.abspath(__file__))
_CHAR = os.path.abspath(os.path.join(_here, "..", ".."))
_WORKSPACE = os.path.abspath(os.path.join(_here, "..", "..", ".."))
for _p in (_WORKSPACE, _CHAR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shared import sweep_spec as ss
from shared.loaders import get_model_config
from experiments.common import wave_model as wm
from experiments.common import stats
from experiments.common.labels import Label, write_labeled_csv

DEFAULT_MODELS = ["zamba2_1.2b", "zamba2_2.7b", "falcon_h1_1.5b", "falcon_h1_3b"]
DEFAULT_BATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
DEFAULT_CHUNKS = ["256", "512"]
DEFAULT_LAYERS = ["ssm", "ssm_full", "attn"]
DEFAULT_CONTEXT_LENS = [0, 4096]
_WORKER = os.path.join(_here, "_e2_worker.py")

_SWEEP_SCHEMA = {
    "model": Label.METADATA, "config_status": Label.METADATA,
    "layer_type": Label.METADATA, "batch": Label.METADATA, "seq": Label.METADATA,
    "chunk_granularity": Label.METADATA, "tokens_per_call": Label.METADATA,
    "context_len": Label.METADATA, "sm_count": Label.METADATA,
    "status": Label.METADATA, "n_measure": Label.METADATA,
    "n_blocks_per_call": Label.DERIVED, "waves": Label.DERIVED,
    "latency_ms": Label.MEASURED, "latency_std_ms": Label.MEASURED,
    "achieved_bw_gbs": Label.MEASURED, "bw_util_pct": Label.DERIVED,
}

_CI_SCHEMA = {
    "model": Label.METADATA, "config_status": Label.METADATA,
    "layer_type": Label.METADATA, "batch": Label.METADATA,
    "chunk_granularity": Label.METADATA, "context_len": Label.METADATA,
    "n_sm_points": Label.METADATA, "min_n_measure": Label.METADATA,
    "grid_sat_sm_e0": Label.DERIVED,
    "sat_sm_point": Label.DERIVED, "sat_sm_ci_low": Label.DERIVED,
    "sat_sm_ci_high": Label.DERIVED,
    "bw_util_pct_at_sat": Label.MEASURED,
}


def _run_sm_level(model, sm, total_sm, layers, batches, chunks, seq, ctxs,
                  n_warmup, n_measure, timeout_s) -> dict:
    cmd = [sys.executable, _WORKER, "--model", model, "--sm-count", str(sm),
           "--total-sm", str(total_sm), "--seq", str(seq),
           "--layer-types", *layers, "--batches", *map(str, batches),
           "--chunks", *map(str, chunks), "--context-lens", *map(str, ctxs),
           "--n-warmup", str(n_warmup), "--n-measure", str(n_measure)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        line = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else ""
        return json.loads(line) if line else {"sm": sm, "rows": [], "fatal": out.stderr[-200:]}
    except subprocess.TimeoutExpired:
        return {"sm": sm, "rows": [], "fatal": f"timeout>{timeout_s}s (likely deadlock)"}
    except Exception as e:  # noqa: BLE001
        return {"sm": sm, "rows": [], "fatal": f"{type(e).__name__}: {e}"}


def _group_key(r) -> tuple:
    return (r["layer_type"], r["batch"], r["chunk_granularity"], r["context_len"])


def run_model(model, sm_grid, total_sm, layers, batches, chunks, seq, ctxs,
              n_warmup, n_measure, timeout_s, output_dir: Path):
    device = ss.canonical_device()
    status_meta = get_model_config(model).get("config_status", "verified")
    if status_meta == "provisional":
        print(f"  ⚠ {model} PROVISIONAL config (verify n_heads before trusting G1).")

    sweep_rows: list[dict] = []
    # group -> {sm: [latencies]} and group -> {sm: row}
    samples: dict = defaultdict(dict)
    rows_by_group_sm: dict = defaultdict(dict)

    print(f"\n=== E2 SM saturation sweep: {model} ({len(sm_grid)} SM levels) ===")
    for sm in sm_grid:
        res = _run_sm_level(model, sm, total_sm, layers, batches, chunks, seq,
                            ctxs, n_warmup, n_measure, timeout_s)
        if res.get("fatal"):
            print(f"  sm={sm:3d}  LEVEL FAILED: {res['fatal'][:80]}")
        n_ok = 0
        for r in res.get("rows", []):
            lats = r.pop("latencies_ms", [])
            sweep_rows.append({k: r.get(k, "") for k in _SWEEP_SCHEMA})
            if r.get("status") == "ok" and lats:
                gk = _group_key(r)
                samples[gk][sm] = lats
                rows_by_group_sm[gk][sm] = r
                n_ok += 1
        print(f"  sm={sm:3d}  ok_configs={n_ok}/{len(res.get('rows', []))}")

    # --- saturation CI per group --------------------------------------------
    ci_rows: list[dict] = []
    for gk, sm_to_samples in samples.items():
        lt, batch, chunk, ctx = gk
        if len(sm_to_samples) < 2:
            continue
        ci = stats.saturation_sm_ci(sm_to_samples, total_sm=total_sm,
                                    threshold=ss.saturation_threshold())
        sat = ci["sat_sm_point"]
        bw_at_sat = ""
        if sat in rows_by_group_sm[gk]:
            bw_at_sat = rows_by_group_sm[gk][sat].get("bw_util_pct", "")
        # E0 grid prediction for cross-check (token-mixer block count).
        n_heads = ss.ssm_n_heads(model)
        ssd = ss.ssd_chunk_size(model)
        tokens = seq if chunk == "full" else min(int(chunk), seq)
        nb = wm.n_blocks(batch, tokens, n_heads, ssd)
        grid_sat_e0 = wm.grid_saturation_sm(nb)
        ci_rows.append({
            "model": model, "config_status": status_meta, "layer_type": lt,
            "batch": batch, "chunk_granularity": chunk, "context_len": ctx,
            "n_sm_points": len(sm_to_samples),
            "min_n_measure": min(len(v) for v in sm_to_samples.values()),
            "grid_sat_sm_e0": grid_sat_e0,
            "sat_sm_point": sat,
            "sat_sm_ci_low": ci["sat_sm_ci_low"],
            "sat_sm_ci_high": ci["sat_sm_ci_high"],
            "bw_util_pct_at_sat": bw_at_sat,
        })

    out_sweep = output_dir / f"sm_sweep_{model}_{device}.csv"
    out_ci = output_dir / f"saturation_ci_{model}_{device}.csv"
    write_labeled_csv(out_sweep, sweep_rows, _SWEEP_SCHEMA)
    write_labeled_csv(out_ci, ci_rows, _CI_SCHEMA)
    print(f"  wrote {len(sweep_rows)} sweep rows -> {out_sweep}")
    print(f"  wrote {len(ci_rows)} saturation-CI rows -> {out_ci}")


def parse_args():
    p = argparse.ArgumentParser(description="E2 batch-swept SM saturation (G1 core)")
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--batches", nargs="+", type=int, default=DEFAULT_BATCHES)
    p.add_argument("--chunks", nargs="+", default=DEFAULT_CHUNKS)
    p.add_argument("--layer-types", nargs="+", default=DEFAULT_LAYERS)
    p.add_argument("--context-lens", nargs="+", type=int, default=DEFAULT_CONTEXT_LENS)
    p.add_argument("--seq", type=int, default=2048)
    p.add_argument("--n-warmup", type=int, default=10)
    p.add_argument("--n-measure", type=int, default=30, help="raw samples per point (≥30)")
    p.add_argument("--timeout-s", type=int, default=1200, help="per-SM-level timeout")
    p.add_argument("--output-dir", type=Path, default=Path(_CHAR) / "results_v2" / "e2")
    return p.parse_args()


def main():
    args = parse_args()
    sm_grid = ss.sm_grid()
    ss.assert_on_sm_grid(sm_grid)       # never snap off-grid
    total_sm = ss.total_sm()
    if args.n_measure < 30:
        print(f"  WARNING: n_measure={args.n_measure} < 30 — CI will be weak.")
    for model in args.models:
        run_model(model, sm_grid, total_sm, args.layer_types, args.batches,
                  args.chunks, args.seq, args.context_lens, args.n_warmup,
                  args.n_measure, args.timeout_s, args.output_dir)


if __name__ == "__main__":
    main()
