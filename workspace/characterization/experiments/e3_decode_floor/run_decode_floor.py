"""
E3 — decode SM floor (A100). Can run in parallel with E1.

For large-batch decode, measures the SM floor (the minimum SM count beyond which
bandwidth is already saturated and latency is flat) for the SSM and attention
decode kernels separately.

Axes: batch{8,32,128,256} × context{1k,4k,16k} × SM grid.

Null hypothesis (the "Memory Gap"): both kernels reach a LOW floor and stay flat
afterward — decode cannot use the extra SMs. If reproduced, decode is confirmed
as a donor/fallback (it gives SMs away to prefill), which is the v2 role for
decode. This experiment reproduces P3, it does not adapt SM within decode.

Outputs → results_v2/e3/
  decode_sweep_{model}_{device}.csv   per (layer,batch,ctx,sm) point
      MEASURED: latency_ms, latency_std_ms, achieved_bw_gbs ; DERIVED: bw_util_pct
  decode_floor_{model}_{device}.csv   per (layer,batch,ctx) group
      DERIVED : floor_sm_point, floor_sm_ci_low/high, flat_region_bw_util_pct
      metadata: memory_gap (floor ≤ ~50% total AND flat-region util high)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

_here = os.path.dirname(os.path.abspath(__file__))
_CHAR = os.path.abspath(os.path.join(_here, "..", ".."))
_WORKSPACE = os.path.abspath(os.path.join(_here, "..", "..", ".."))
for _p in (_WORKSPACE, _CHAR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shared import sweep_spec as ss
from shared.loaders import get_model_config
from experiments.common import stats
from experiments.common.labels import Label, write_labeled_csv

DEFAULT_MODELS = ["zamba2_1.2b", "falcon_h1_1.5b"]
DEFAULT_BATCHES = [8, 32, 128, 256]
DEFAULT_CONTEXTS = [1024, 4096, 16384]
DEFAULT_LAYERS = ["ssm", "attn"]
_WORKER = os.path.join(_here, "_e3_worker.py")

_SWEEP_SCHEMA = {
    "model": Label.METADATA, "config_status": Label.METADATA,
    "layer_type": Label.METADATA, "batch": Label.METADATA,
    "context_len": Label.METADATA, "sm_count": Label.METADATA,
    "status": Label.METADATA, "n_measure": Label.METADATA,
    "latency_ms": Label.MEASURED, "latency_std_ms": Label.MEASURED,
    "achieved_bw_gbs": Label.MEASURED, "bw_util_pct": Label.DERIVED,
}

_FLOOR_SCHEMA = {
    "model": Label.METADATA, "config_status": Label.METADATA,
    "layer_type": Label.METADATA, "batch": Label.METADATA,
    "context_len": Label.METADATA, "n_sm_points": Label.METADATA,
    "memory_gap": Label.METADATA,
    "floor_sm_point": Label.DERIVED, "floor_sm_ci_low": Label.DERIVED,
    "floor_sm_ci_high": Label.DERIVED, "flat_region_bw_util_pct": Label.DERIVED,
}


def _run_sm_level(model, sm, total_sm, layers, batches, ctxs, n_warmup, n_measure,
                  timeout_s) -> dict:
    cmd = [sys.executable, _WORKER, "--model", model, "--sm-count", str(sm),
           "--total-sm", str(total_sm), "--layer-types", *layers,
           "--batches", *map(str, batches), "--context-lens", *map(str, ctxs),
           "--n-warmup", str(n_warmup), "--n-measure", str(n_measure)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        line = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else ""
        return json.loads(line) if line else {"sm": sm, "rows": [], "fatal": out.stderr[-200:]}
    except subprocess.TimeoutExpired:
        return {"sm": sm, "rows": [], "fatal": f"timeout>{timeout_s}s"}
    except Exception as e:  # noqa: BLE001
        return {"sm": sm, "rows": [], "fatal": f"{type(e).__name__}: {e}"}


def run_model(model, sm_grid, total_sm, layers, batches, ctxs, n_warmup, n_measure,
              timeout_s, output_dir: Path):
    device = ss.canonical_device()
    status_meta = get_model_config(model).get("config_status", "verified")
    sweep_rows: list[dict] = []
    samples: dict = defaultdict(dict)       # (lt,batch,ctx) -> {sm: [lat]}
    util_by_group_sm: dict = defaultdict(dict)

    print(f"\n=== E3 decode floor: {model} ===")
    for sm in sm_grid:
        res = _run_sm_level(model, sm, total_sm, layers, batches, ctxs,
                            n_warmup, n_measure, timeout_s)
        if res.get("fatal"):
            print(f"  sm={sm:3d}  LEVEL FAILED: {res['fatal'][:80]}")
        for r in res.get("rows", []):
            lats = r.pop("latencies_ms", [])
            sweep_rows.append({k: r.get(k, "") for k in _SWEEP_SCHEMA})
            if r.get("status") == "ok" and lats:
                gk = (r["layer_type"], r["batch"], r["context_len"])
                samples[gk][sm] = lats
                util_by_group_sm[gk][sm] = r.get("bw_util_pct", "")

    floor_rows: list[dict] = []
    for gk, sm_to_samples in samples.items():
        lt, batch, ctx = gk
        if len(sm_to_samples) < 2:
            continue
        ci = stats.saturation_sm_ci(sm_to_samples, total_sm=total_sm,
                                    threshold=ss.saturation_threshold())
        floor = ci["sat_sm_point"]
        flat_utils = [util_by_group_sm[gk][s] for s in sm_to_samples
                      if s >= (floor or 0) and isinstance(util_by_group_sm[gk].get(s), (int, float))]
        flat_util = round(median(flat_utils), 3) if flat_utils else ""
        memory_gap = (floor is not None and floor <= 0.5 * total_sm
                      and isinstance(flat_util, (int, float)) and flat_util >= 70)
        floor_rows.append({
            "model": model, "config_status": status_meta, "layer_type": lt,
            "batch": batch, "context_len": ctx, "n_sm_points": len(sm_to_samples),
            "memory_gap": bool(memory_gap),
            "floor_sm_point": floor,
            "floor_sm_ci_low": ci["sat_sm_ci_low"],
            "floor_sm_ci_high": ci["sat_sm_ci_high"],
            "flat_region_bw_util_pct": flat_util,
        })

    out_sweep = output_dir / f"decode_sweep_{model}_{device}.csv"
    out_floor = output_dir / f"decode_floor_{model}_{device}.csv"
    write_labeled_csv(out_sweep, sweep_rows, _SWEEP_SCHEMA)
    write_labeled_csv(out_floor, floor_rows, _FLOOR_SCHEMA)
    print(f"  wrote {len(sweep_rows)} sweep rows -> {out_sweep}")
    print(f"  wrote {len(floor_rows)} floor rows -> {out_floor}")


def parse_args():
    p = argparse.ArgumentParser(description="E3 decode SM floor (A100)")
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--batches", nargs="+", type=int, default=DEFAULT_BATCHES)
    p.add_argument("--context-lens", nargs="+", type=int, default=DEFAULT_CONTEXTS)
    p.add_argument("--layer-types", nargs="+", default=DEFAULT_LAYERS)
    p.add_argument("--n-warmup", type=int, default=10)
    p.add_argument("--n-measure", type=int, default=30)
    p.add_argument("--timeout-s", type=int, default=1200)
    p.add_argument("--output-dir", type=Path, default=Path(_CHAR) / "results_v2" / "e3")
    return p.parse_args()


def main():
    args = parse_args()
    sm_grid = ss.sm_grid()
    ss.assert_on_sm_grid(sm_grid)
    total_sm = ss.total_sm()
    for model in args.models:
        run_model(model, sm_grid, total_sm, args.layer_types, args.batches,
                  args.context_lens, args.n_warmup, args.n_measure, args.timeout_s,
                  args.output_dir)


if __name__ == "__main__":
    main()
