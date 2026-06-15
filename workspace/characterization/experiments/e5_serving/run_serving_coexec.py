"""
E5 — serving prefill+decode coexistence sweep, as a prefill×decode LAYER-TYPE
matrix (A100).

This re-frames the asymmetry on the mux-able axis: not "spatially split SM
between attn and ssm" (E5-v1 / E4 showed that loses to plain co-scheduling), but
"the prefill↔decode asymmetry, resolved by layer type." We measure the 2×2 cells

        decode = attn (KV stream, BW∝context)   decode = ssm (state, BW≈const)
  prefill = ssm   |        cell                 |        cell                  |
  prefill = attn  |        cell                 |        cell                  |

For each cell × decode_batch{8..256} × context, and each SM backend:

  sequential   prefill then decode, full SM           (no-concurrency baseline)
  two_stream   both on two plain CUDA streams          (HW/MPS-like co-schedule)
  green_ctx@f  spatial split, prefill=f·SM             (does partition add anything?)

Hypotheses this matrix tests:
  * decode=attn (esp. long context) leaves the most BW/SM slack → biggest overlap.
  * prefill=ssm ⊗ decode=attn should be the most complementary pairing
    (the E4(b) interference result says otherwise — this is the decisive cell).
  * decode=ssm (state BW≈const, SMs fill via batch·n_heads) → least slack → least gain.

ONE incoming prefill chunk (batch=1) runs against a DECODE step over batch in-flight
requests. Reuses E4 _green_ctx (partitions+timing) + the kernel builders.

Outputs → results_v2/e5/serving_coexec_{model}_{device}.csv
  MEASURED : solo_prefill_ms, solo_decode_ms, prefill_stream_ms,
             decode_stream_ms, concurrent_ms
  DERIVED  : overlap_ratio, speedup_vs_seq, decode_inflation_pct,
             combined_tok_per_ms
"""

from __future__ import annotations

import argparse
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
from experiments.common.labels import Label, write_labeled_csv

DEFAULT_MODELS = ["zamba2_1.2b", "zamba2_2.7b", "falcon_h1_1.5b", "falcon_h1_3b"]
DEFAULT_DECODE_BATCHES = [8, 32, 64, 128, 256]
DEFAULT_CONTEXTS = [4096]
DEFAULT_FRACS = [0.5, 0.7]                 # prefill SM fraction for green_ctx split
DEFAULT_PREFILL_LAYERS = ["ssm", "attn"]   # prefill chunk component
DEFAULT_DECODE_LAYERS = ["attn", "ssm"]    # decode step component

_SCHEMA = {
    "model": Label.METADATA, "device": Label.METADATA, "config_status": Label.METADATA,
    "prefill_layer": Label.METADATA, "decode_layer": Label.METADATA,
    "decode_batch": Label.METADATA, "context_len": Label.METADATA,
    "prefill_tokens": Label.METADATA, "chunk": Label.METADATA,
    "backend": Label.METADATA, "prefill_sm_frac": Label.METADATA,
    "prefill_sm": Label.METADATA, "decode_sm": Label.METADATA, "status": Label.METADATA,
    "solo_prefill_ms": Label.MEASURED, "solo_decode_ms": Label.MEASURED,
    "prefill_stream_ms": Label.MEASURED, "decode_stream_ms": Label.MEASURED,
    "concurrent_ms": Label.MEASURED,
    "overlap_ratio": Label.DERIVED, "speedup_vs_seq": Label.DERIVED,
    "decode_inflation_pct": Label.DERIVED, "combined_tok_per_ms": Label.DERIVED,
}


def _row(meta, backend, frac, psm, dsm, solo_p, solo_d, pstream, dstream, conc, status):
    seq = solo_p + solo_d
    rnd = lambda v: round(v, 4) if isinstance(v, (int, float)) else v
    infl = ((dstream - solo_d) / solo_d * 100.0) if (solo_d and dstream != "") else ""
    overlap = (conc / seq) if (seq and conc != "") else ""
    speed = (seq / conc) if (conc not in ("", 0)) else ""
    comb = ((meta["prefill_tokens"] + meta["decode_batch"]) / conc) if (conc not in ("", 0)) else ""
    return {**meta, "backend": backend, "prefill_sm_frac": frac,
            "prefill_sm": psm, "decode_sm": dsm, "status": status,
            "solo_prefill_ms": rnd(solo_p), "solo_decode_ms": rnd(solo_d),
            "prefill_stream_ms": rnd(pstream), "decode_stream_ms": rnd(dstream),
            "concurrent_ms": rnd(conc), "overlap_ratio": rnd(overlap),
            "speedup_vs_seq": rnd(speed), "decode_inflation_pct": rnd(infl),
            "combined_tok_per_ms": rnd(comb)}


def _failed(meta, backend, frac, msg):
    base = dict.fromkeys(_SCHEMA, "")
    base.update(meta)
    base.update({"backend": backend, "prefill_sm_frac": frac,
                 "prefill_sm": "", "decode_sm": "", "status": f"failed:{msg[:40]}"})
    return base


def run_model(model, decode_batches, contexts, chunk, prefill_layers, decode_layers,
              fracs, n_warmup, n_measure, total_sm, device, out_dir: Path):
    import torch
    from experiments.common import kernels as K
    from experiments.e4_concurrent._green_ctx import (
        create_two_partitions, measure_concurrent, time_solo)

    cfg = K.load_layer_cfg(model)
    status_meta = cfg["config_status"]
    ptok = chunk
    rows = []
    print(f"\n=== E5 serving coexec matrix: {model} ===")
    if status_meta == "provisional":
        print(f"  ⚠ {model} provisional config")

    def build_decode(dl, db, ctx):
        if dl == "attn":
            return K.build_decode_attn_fn(cfg, batch=db, context_len=ctx)
        return K.build_decode_ssm_fn(cfg, batch=db)        # state BW ≈ const (ctx ignored)

    for pl in prefill_layers:
        for dl in decode_layers:
            for ctx in contexts:
                for db in decode_batches:
                    meta = {"model": model, "device": device, "config_status": status_meta,
                            "prefill_layer": pl, "decode_layer": dl, "decode_batch": db,
                            "context_len": ctx, "prefill_tokens": ptok, "chunk": chunk}
                    try:
                        prefill_fn, *_ = K.build_prefill_fn(pl, cfg, batch=1, tokens=ptok)
                        decode_fn, *_ = build_decode(dl, db, ctx)
                        solo_p = time_solo(prefill_fn, n_warmup, n_measure)
                        solo_d = time_solo(decode_fn, n_warmup, n_measure)
                    except Exception as e:  # noqa: BLE001
                        torch.cuda.empty_cache()
                        for bk, fr in [("sequential", ""), ("two_stream", "")] + \
                                      [("green_ctx", f) for f in fracs]:
                            rows.append(_failed(meta, bk, fr, type(e).__name__))
                        print(f"  pf={pl} dec={dl} ctx={ctx} db={db:>3}  FAILED: {type(e).__name__}")
                        continue

                    rows.append(_row(meta, "sequential", "", total_sm, total_sm,
                                     solo_p, solo_d, solo_p, solo_d, solo_p + solo_d, "ok"))
                    try:
                        ts = measure_concurrent(prefill_fn, decode_fn,
                                                torch.cuda.Stream(), torch.cuda.Stream(),
                                                n_warmup, n_measure)
                        rows.append(_row(meta, "two_stream", "", total_sm, total_sm,
                                         solo_p, solo_d, ts["a_stream_ms"], ts["b_stream_ms"],
                                         ts["concurrent_ms"], "ok"))
                    except Exception as e:  # noqa: BLE001
                        rows.append(_failed(meta, "two_stream", "", type(e).__name__))

                    for fr in fracs:
                        s_p, s_d, info = create_two_partitions(n_first_sm=max(1, round(fr * total_sm)))
                        if s_p is None:
                            rows.append(_failed(meta, "green_ctx", fr,
                                                info.get("error", "no_green_ctx")[:30]))
                            continue
                        try:
                            gc = measure_concurrent(prefill_fn, decode_fn, s_p, s_d,
                                                    n_warmup, n_measure)
                            rows.append(_row(meta, "green_ctx", fr,
                                             info.get("actual_first_sm", ""),
                                             info.get("actual_second_sm", ""),
                                             solo_p, solo_d, gc["a_stream_ms"],
                                             gc["b_stream_ms"], gc["concurrent_ms"], "ok"))
                        except Exception as e:  # noqa: BLE001
                            rows.append(_failed(meta, "green_ctx", fr, type(e).__name__))

                # per-cell quick summary at the smallest batch (max overlap regime)
                cell = [r for r in rows if r["prefill_layer"] == pl and r["decode_layer"] == dl
                        and r["context_len"] == contexts[0] and r["backend"] == "two_stream"
                        and r["status"] == "ok"]
                if cell:
                    lo = min(cell, key=lambda r: r["decode_batch"])
                    print(f"  pf={pl:4} dec={dl:4} ctx={contexts[0]}  two_stream speedup@db{lo['decode_batch']}"
                          f"={lo['speedup_vs_seq']}x")

    out = out_dir / f"serving_coexec_{model}_{device}.csv"
    write_labeled_csv(out, rows, _SCHEMA)
    print(f"  wrote {len(rows)} rows -> {out}")


def parse_args():
    p = argparse.ArgumentParser(description="E5 serving prefill×decode coexistence matrix")
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--decode-batches", nargs="+", type=int, default=DEFAULT_DECODE_BATCHES)
    p.add_argument("--context-lens", nargs="+", type=int, default=DEFAULT_CONTEXTS)
    p.add_argument("--chunk", type=int, default=256, help="prefill chunk tokens")
    p.add_argument("--prefill-layers", nargs="+", default=DEFAULT_PREFILL_LAYERS,
                   choices=["ssm", "ssm_full", "attn"])
    p.add_argument("--decode-layers", nargs="+", default=DEFAULT_DECODE_LAYERS,
                   choices=["attn", "ssm"])
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
                  args.prefill_layers, args.decode_layers, args.fracs,
                  args.n_warmup, args.n_measure, total_sm, device, args.output_dir)


if __name__ == "__main__":
    main()
