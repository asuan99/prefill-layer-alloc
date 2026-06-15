"""
E5 — serving prefill+decode coexistence sweep (A100).

Answers "does the attn/ssm asymmetry buy anything under multi-request batching?"
by measuring a serving-shaped workload: ONE incoming prefill chunk (batch=1)
running concurrently with a DECODE step over a large batch of in-flight requests
(batch ∈ {8..256}), across the SM-allocation backends:

  sequential   prefill then decode, each on full SM      (no concurrency baseline)
  two_stream   both on two plain CUDA streams            (HW/MPS-like co-schedule)
  green_ctx@f  spatial Green Context split, prefill=f·SM (asymmetry-aware split)

The two_stream backend is the fair baseline for the spatial-partition-SPECIFIC
gain (green_ctx ÷ two_stream); sequential is the absolute upper bound. The decode
batch axis is the headline: E3 showed the free-SM budget (108 − decode floor)
shrinks as decode batch grows, so the spatial window should close at high batch.

Reuses the E4 Green Context helpers (partitions + concurrent timing) and the
weight-free kernel builders. Per-config try/except; failures → status=failed.

Outputs → results_v2/e5/serving_coexec_{model}_{device}.csv
  MEASURED : solo_prefill_ms, solo_decode_ms, prefill_stream_ms,
             decode_stream_ms, concurrent_ms
  DERIVED  : overlap_ratio, speedup_vs_seq, decode_inflation_pct,
             combined_tok_per_ms
"""

from __future__ import annotations

import argparse
import math
import os
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

DEFAULT_MODELS = ["zamba2_1.2b", "zamba2_2.7b", "falcon_h1_1.5b", "falcon_h1_3b"]
DEFAULT_DECODE_BATCHES = [8, 32, 64, 128, 256]
DEFAULT_CONTEXTS = [4096]
DEFAULT_FRACS = [0.5, 0.7]            # prefill SM fraction for green_ctx split
DEFAULT_PREFILL_LAYER = "ssm"         # SSM-branch proxy for the prefill chunk

_SCHEMA = {
    "model": Label.METADATA, "device": Label.METADATA, "config_status": Label.METADATA,
    "decode_batch": Label.METADATA, "context_len": Label.METADATA,
    "prefill_tokens": Label.METADATA, "chunk": Label.METADATA,
    "prefill_layer": Label.METADATA, "backend": Label.METADATA,
    "prefill_sm_frac": Label.METADATA, "prefill_sm": Label.METADATA,
    "decode_sm": Label.METADATA, "status": Label.METADATA,
    "solo_prefill_ms": Label.MEASURED, "solo_decode_ms": Label.MEASURED,
    "prefill_stream_ms": Label.MEASURED, "decode_stream_ms": Label.MEASURED,
    "concurrent_ms": Label.MEASURED,
    "overlap_ratio": Label.DERIVED, "speedup_vs_seq": Label.DERIVED,
    "decode_inflation_pct": Label.DERIVED, "combined_tok_per_ms": Label.DERIVED,
}


def _row(model, device, status_meta, db, ctx, ptok, chunk, player, backend, frac,
         psm, dsm, solo_p, solo_d, pstream, dstream, conc, status):
    seq = solo_p + solo_d
    infl = ((dstream - solo_d) / solo_d * 100.0) if (solo_d and dstream != "") else ""
    overlap = (conc / seq) if (seq and conc != "") else ""
    speed = (seq / conc) if (conc not in ("", 0)) else ""
    comb = ((ptok + db) / conc) if (conc not in ("", 0)) else ""
    rnd = lambda v: round(v, 4) if isinstance(v, (int, float)) else v
    return {
        "model": model, "device": device, "config_status": status_meta,
        "decode_batch": db, "context_len": ctx, "prefill_tokens": ptok,
        "chunk": chunk, "prefill_layer": player, "backend": backend,
        "prefill_sm_frac": frac, "prefill_sm": psm, "decode_sm": dsm,
        "status": status,
        "solo_prefill_ms": rnd(solo_p), "solo_decode_ms": rnd(solo_d),
        "prefill_stream_ms": rnd(pstream), "decode_stream_ms": rnd(dstream),
        "concurrent_ms": rnd(conc),
        "overlap_ratio": rnd(overlap), "speedup_vs_seq": rnd(speed),
        "decode_inflation_pct": rnd(infl), "combined_tok_per_ms": rnd(comb),
    }


def _failed(model, device, status_meta, db, ctx, ptok, chunk, player, backend, frac, msg):
    base = dict.fromkeys(_SCHEMA, "")
    base.update({"model": model, "device": device, "config_status": status_meta,
                 "decode_batch": db, "context_len": ctx, "prefill_tokens": ptok,
                 "chunk": chunk, "prefill_layer": player, "backend": backend,
                 "prefill_sm_frac": frac, "prefill_sm": "", "decode_sm": "",
                 "status": f"failed:{msg[:40]}"})
    return base


def run_model(model, decode_batches, contexts, chunk, player, fracs,
              n_warmup, n_measure, total_sm, device, out_dir: Path):
    import torch  # noqa: F401
    from experiments.common import kernels as K
    from experiments.e4_concurrent._green_ctx import (
        create_two_partitions, measure_concurrent, time_solo)

    cfg = K.load_layer_cfg(model)
    status_meta = cfg["config_status"]
    ptok = chunk
    rows = []
    print(f"\n=== E5 serving coexec: {model} ===")
    if status_meta == "provisional":
        print(f"  ⚠ {model} provisional config")

    for ctx in contexts:
        for db in decode_batches:
            try:
                prefill_fn, *_ = K.build_prefill_fn(player, cfg, batch=1, tokens=ptok)
                decode_fn, *_ = K.build_decode_attn_fn(cfg, batch=db, context_len=ctx)
                solo_p = time_solo(prefill_fn, n_warmup, n_measure)
                solo_d = time_solo(decode_fn, n_warmup, n_measure)
            except Exception as e:  # noqa: BLE001
                import torch
                torch.cuda.empty_cache()
                for bk, fr in [("sequential", ""), ("two_stream", "")] + \
                              [("green_ctx", f) for f in fracs]:
                    rows.append(_failed(model, device, status_meta, db, ctx, ptok,
                                        chunk, player, bk, fr, f"{type(e).__name__}"))
                print(f"  ctx={ctx} db={db:>3}  build/solo FAILED: {type(e).__name__}")
                continue

            # sequential (computed)
            rows.append(_row(model, device, status_meta, db, ctx, ptok, chunk, player,
                             "sequential", "", total_sm, total_sm, solo_p, solo_d,
                             solo_p, solo_d, solo_p + solo_d, "ok"))

            # two_stream (no partition)
            try:
                ts = measure_concurrent(prefill_fn, decode_fn,
                                        torch.cuda.Stream(), torch.cuda.Stream(),
                                        n_warmup, n_measure)
                rows.append(_row(model, device, status_meta, db, ctx, ptok, chunk, player,
                                 "two_stream", "", total_sm, total_sm, solo_p, solo_d,
                                 ts["a_stream_ms"], ts["b_stream_ms"], ts["concurrent_ms"], "ok"))
            except Exception as e:  # noqa: BLE001
                rows.append(_failed(model, device, status_meta, db, ctx, ptok, chunk,
                                    player, "two_stream", "", f"{type(e).__name__}"))

            # green_ctx spatial split per fraction
            for fr in fracs:
                n_first = max(1, round(fr * total_sm))
                s_p, s_d, info = create_two_partitions(n_first_sm=n_first)
                if s_p is None:
                    rows.append(_failed(model, device, status_meta, db, ctx, ptok, chunk,
                                        player, "green_ctx", fr,
                                        info.get("error", "no_green_ctx")[:30]))
                    continue
                try:
                    gc = measure_concurrent(prefill_fn, decode_fn, s_p, s_d,
                                            n_warmup, n_measure)
                    rows.append(_row(model, device, status_meta, db, ctx, ptok, chunk, player,
                                     "green_ctx", fr, info.get("actual_first_sm", n_first),
                                     info.get("actual_second_sm", total_sm - n_first),
                                     solo_p, solo_d, gc["a_stream_ms"], gc["b_stream_ms"],
                                     gc["concurrent_ms"], "ok"))
                except Exception as e:  # noqa: BLE001
                    rows.append(_failed(model, device, status_meta, db, ctx, ptok, chunk,
                                        player, "green_ctx", fr, f"{type(e).__name__}"))

            ok = [r for r in rows if r["decode_batch"] == db and r["context_len"] == ctx
                  and r["status"] == "ok"]
            best = max((r for r in ok if isinstance(r["speedup_vs_seq"], (int, float))),
                       key=lambda r: r["speedup_vs_seq"], default=None)
            tag = f"{best['backend']}{best['prefill_sm_frac']} {best['speedup_vs_seq']}x" if best else "—"
            print(f"  ctx={ctx} db={db:>3}  best speedup: {tag}")

    out = out_dir / f"serving_coexec_{model}_{device}.csv"
    write_labeled_csv(out, rows, _SCHEMA)
    print(f"  wrote {len(rows)} rows -> {out}")


def parse_args():
    p = argparse.ArgumentParser(description="E5 serving prefill+decode coexistence")
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--decode-batches", nargs="+", type=int, default=DEFAULT_DECODE_BATCHES)
    p.add_argument("--context-lens", nargs="+", type=int, default=DEFAULT_CONTEXTS)
    p.add_argument("--chunk", type=int, default=256, help="prefill chunk tokens")
    p.add_argument("--prefill-layer", default=DEFAULT_PREFILL_LAYER,
                   choices=["ssm", "ssm_full", "attn"])
    p.add_argument("--fracs", nargs="+", type=float, default=DEFAULT_FRACS,
                   help="green_ctx prefill SM fractions")
    p.add_argument("--n-warmup", type=int, default=5)
    p.add_argument("--n-measure", type=int, default=20)
    p.add_argument("--output-dir", type=Path, default=Path(_CHAR) / "results_v2" / "e5")
    return p.parse_args()


def main():
    args = parse_args()
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required")
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"E5 needs a CUDA GPU: {e}")
    total_sm = ss.total_sm()
    device = ss.canonical_device()
    for m in args.models:
        run_model(m, args.decode_batches, args.context_lens, args.chunk,
                  args.prefill_layer, args.fracs, args.n_warmup, args.n_measure,
                  total_sm, device, args.output_dir)


if __name__ == "__main__":
    main()
