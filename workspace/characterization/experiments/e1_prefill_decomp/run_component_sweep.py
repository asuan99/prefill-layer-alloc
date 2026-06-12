"""
E1 — prefill component decomposition (A100).

Decomposes the full prefill layer into in_proj / token_mixer(scan|attn) / out_proj
for both SLMs and measures each component's latency + achieved bandwidth, so we
know whether the SSM scan is actually the dominant, memory-bound component (the
premise of the whole study).

Axes:  batch{1,8,32,128} × chunk{256,512,full},  SM = 108 fixed.
Each (model,batch,seq,chunk) runs in its OWN subprocess (CUDA-context isolation);
a failed config is recorded status=failed and does not abort the sweep.

Outputs → results_v2/e1/component_sweep_{model}_{device}.csv
  MEASURED : latency_ms, latency_std_ms, achieved_bw_gbs
  DERIVED  : bw_util_pct, scan_share_pct
  metadata : model, component, batch, seq, chunk, sm_count, status, config_status

G0 hook: scan_share_pct = scan / (in_proj + scan + out_proj). The gate
(gates/adjudicate.py) reads this CSV and records G0=WEAK if max scan_share < 30%.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

_here = os.path.dirname(os.path.abspath(__file__))
_CHAR = os.path.abspath(os.path.join(_here, "..", ".."))
_WORKSPACE = os.path.abspath(os.path.join(_here, "..", "..", ".."))
for _p in (_WORKSPACE, _CHAR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shared import sweep_spec as ss
from shared.loaders import get_model_config
from experiments.common.labels import Label, write_labeled_csv

DEFAULT_MODELS = ["zamba2_1.2b", "falcon_h1_1.5b"]
DEFAULT_BATCHES = [1, 8, 32, 128]
DEFAULT_SEQ = 2048
DEFAULT_CHUNKS = ["256", "512", "full"]
_WORKER = os.path.join(_here, "_e1_worker.py")

_SCHEMA = {
    "model": Label.METADATA,
    "config_status": Label.METADATA,
    "layer_type": Label.METADATA,
    "component": Label.METADATA,
    "batch": Label.METADATA,
    "seq": Label.METADATA,
    "chunk_granularity": Label.METADATA,
    "tokens_per_call": Label.METADATA,
    "context_len": Label.METADATA,
    "sm_count": Label.METADATA,
    "status": Label.METADATA,
    "n_blocks_per_call": Label.DERIVED,
    "latency_ms": Label.MEASURED,
    "latency_std_ms": Label.MEASURED,
    "achieved_bw_gbs": Label.MEASURED,
    "bw_util_pct": Label.DERIVED,
    "scan_share_pct": Label.DERIVED,
}


def _run_config(model, batch, seq, chunk, context_len, n_warmup, n_measure) -> dict:
    """Spawn the worker for one config; return its JSON (or a failed stub)."""
    cmd = [sys.executable, _WORKER,
           "--model", model, "--batch", str(batch), "--seq", str(seq),
           "--chunk", str(chunk), "--context-len", str(context_len),
           "--n-warmup", str(n_warmup), "--n-measure", str(n_measure)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        line = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else ""
        if not line:
            return {"rows": [], "status": "failed", "error": out.stderr[-300:]}
        return json.loads(line)
    except Exception as e:  # noqa: BLE001
        return {"rows": [], "status": "failed", "error": f"{type(e).__name__}: {e}"}


def _scan_share(rows: list[dict]) -> float | None:
    """scan_share_pct for one config = scan / (in_proj + scan + out_proj)."""
    lat = {r["component"]: r["latency_ms"] for r in rows if r["status"] == "ok"}
    need = ("in_proj", "scan", "out_proj")
    if not all(c in lat for c in need):
        return None
    denom = sum(lat[c] for c in need)
    return round(100.0 * lat["scan"] / denom, 3) if denom > 0 else None


def _failed_rows(model, batch, seq, chunk, status_msg) -> list[dict]:
    base = {
        "model": model, "config_status": get_model_config(model).get("config_status", "verified"),
        "layer_type": "ssm", "batch": batch, "seq": seq,
        "chunk_granularity": str(chunk), "tokens_per_call": "", "context_len": 0,
        "sm_count": 108, "status": f"failed:{status_msg[:40]}", "n_blocks_per_call": "",
        "latency_ms": "", "latency_std_ms": "", "achieved_bw_gbs": "",
        "bw_util_pct": "", "scan_share_pct": "",
    }
    return [dict(base, component=c) for c in ("in_proj", "scan", "out_proj", "attn")]


def run_model(model, batches, seq, chunks, context_len, n_warmup, n_measure,
              output_dir: Path) -> Path:
    device = ss.canonical_device()
    all_rows: list[dict] = []
    g0_max = -1.0
    print(f"\n=== E1 component sweep: {model} (SM=108) ===")
    if get_model_config(model).get("config_status") == "provisional":
        print(f"  ⚠ {model} PROVISIONAL config (verify before trusting).")

    for batch in batches:
        for chunk in chunks:
            res = _run_config(model, batch, seq, chunk, context_len, n_warmup, n_measure)
            if res.get("status") != "ok" or not res.get("rows"):
                print(f"  batch={batch:3d} chunk={chunk:>4}  FAILED: {res.get('error','')[:80]}")
                all_rows.extend(_failed_rows(model, batch, seq, chunk, res.get("error", "err")))
                continue
            rows = res["rows"]
            share = _scan_share(rows)
            for r in rows:
                r["scan_share_pct"] = share if share is not None else ""
            all_rows.extend(rows)
            if share is not None:
                g0_max = max(g0_max, share)
            lat = {r["component"]: r["latency_ms"] for r in rows}
            print(f"  batch={batch:3d} chunk={chunk:>4}  "
                  f"in={lat.get('in_proj',0):.3f} scan={lat.get('scan',0):.3f} "
                  f"out={lat.get('out_proj',0):.3f}ms  scan_share={share}%")

    out = output_dir / f"component_sweep_{model}_{device}.csv"
    write_labeled_csv(out, all_rows, _SCHEMA)
    g0 = "WEAK" if (0 <= g0_max < 30) else ("OK" if g0_max >= 30 else "NO_DATA")
    print(f"  wrote {len(all_rows)} rows -> {out}")
    print(f"  G0 preview: max scan_share={g0_max:.1f}% -> G0={g0} "
          f"(adjudicate.py makes the binding call)")
    return out


def parse_args():
    p = argparse.ArgumentParser(description="E1 prefill component decomposition (A100)")
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--batches", nargs="+", type=int, default=DEFAULT_BATCHES)
    p.add_argument("--seq", type=int, default=DEFAULT_SEQ)
    p.add_argument("--chunks", nargs="+", default=DEFAULT_CHUNKS)
    p.add_argument("--context-len", type=int, default=0)
    p.add_argument("--n-warmup", type=int, default=20)
    p.add_argument("--n-measure", type=int, default=50)
    p.add_argument("--output-dir", type=Path,
                   default=Path(_CHAR) / "results_v2" / "e1")
    return p.parse_args()


def main():
    args = parse_args()
    for model in args.models:
        run_model(model, args.batches, args.seq, args.chunks, args.context_len,
                  args.n_warmup, args.n_measure, args.output_dir)


if __name__ == "__main__":
    main()
