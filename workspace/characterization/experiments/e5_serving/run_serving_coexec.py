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
DEFAULT_DECODE_BATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
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

# Optional real-prefill mode (A2) records 3 extra metadata columns and is written
# to a separate `_full` file so the default microbench CSVs/figures are untouched.
def _with_extra(schema):
    out = {}
    for k, v in schema.items():
        out[k] = v
        if k == "chunk":
            out["prefill_mode"] = Label.METADATA
            out["prefill_batch"] = Label.METADATA
            out["n_chunks"] = Label.METADATA
    return out


_SCHEMA_FULL = _with_extra(_SCHEMA)

# full mode maps the cell layer type to its GEMM-inclusive ("_full") kernel.
_FULL_LAYER = {"ssm": "ssm_full", "attn": "attn_full"}


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


def _failed(meta, backend, frac, msg, schema=_SCHEMA):
    base = dict.fromkeys(schema, "")
    base.update(meta)
    base.update({"backend": backend, "prefill_sm_frac": frac,
                 "prefill_sm": "", "decode_sm": "", "status": f"failed:{msg[:40]}"})
    return base


# --- decode-protective split (SLO direction): reserve decode its E3 saturation floor ---
def _load_decode_floor(model, device, e3_dir):
    """{(layer_type,batch,ctx)->floor_sm} from E3 decode_floor CSV (csv module, no pandas)."""
    import csv
    p = Path(e3_dir) / f"decode_floor_{model}_{device}.csv"
    if not p.exists():
        print(f"  ⚠ decode-protect: E3 floor CSV not found ({p.name}) → no protect rows")
        return None
    rows = list(csv.reader(open(p)))
    if not rows:
        return None
    h = 1 if rows[0] and rows[0][0].lstrip().startswith("#") else 0   # skip value_kind comment
    hdr = [c.split("__")[0] for c in rows[h]]
    idx = {c: i for i, c in enumerate(hdr)}
    if not all(k in idx for k in ("layer_type", "batch", "context_len", "floor_sm_point")):
        print(f"  ⚠ decode-protect: unexpected E3 columns → no protect rows")
        return None
    exact, dbmax = {}, {}
    for r in rows[h + 1:]:
        if len(r) < len(hdr):
            continue
        try:
            lt = r[idx["layer_type"]]; b = int(float(r[idx["batch"]]))
            ctx = int(float(r[idx["context_len"]])); fl = int(float(r[idx["floor_sm_point"]]))
        except ValueError:
            continue
        exact[(lt, b, ctx)] = fl
        dbmax[(lt, b)] = max(dbmax.get((lt, b), 0), fl)   # ctx-agnostic fallback = most protective
    return {"exact": exact, "db": dbmax}


def _protect_prefill_sm(floor_lut, decode_layer, batch, context, total_sm):
    """prefill SM count with decode reserved its E3 floor. (psm, dsm); psm=None if no room/data."""
    if floor_lut is None:
        return None, None
    dsm = floor_lut["exact"].get((decode_layer, batch, context))
    if dsm is None:
        dsm = floor_lut["db"].get((decode_layer, batch))
    if dsm is None:
        return None, None
    psm = total_sm - int(dsm)
    return (psm if psm >= 1 else None), int(dsm)   # psm<1 ⇒ decode floor saturates GPU, no room


def run_model(model, decode_batches, contexts, chunk, prefill_layers, decode_layers,
              fracs, n_warmup, n_measure, total_sm, device, out_dir: Path,
              prefill_mode="micro", prefill_tokens=0, prefill_batch=1,
              decode_protect=False, e3_dir=None):
    import torch
    from experiments.common import kernels as K
    from experiments.e4_concurrent._green_ctx import (
        create_two_partitions, measure_concurrent, time_solo)

    cfg = K.load_layer_cfg(model)
    status_meta = cfg["config_status"]
    floor_lut = _load_decode_floor(model, device, e3_dir) if decode_protect else None
    full = (prefill_mode == "full")
    schema = _SCHEMA_FULL if full else _SCHEMA
    # multi-chunk: prefill processes `n_chunks` chunks of `chunk` tokens each.
    req_tokens = prefill_tokens if (prefill_tokens and prefill_tokens > 0) else chunk
    n_chunks = max(1, -(-req_tokens // chunk))          # ceil
    ptok = n_chunks * chunk                              # actual prefill tokens processed
    rows = []
    print(f"\n=== E5 serving coexec matrix: {model}  "
          f"[mode={prefill_mode} pf_batch={prefill_batch} tokens={ptok} n_chunks={n_chunks}] ===")
    if status_meta == "provisional":
        print(f"  ⚠ {model} provisional config")

    def resolve_pf(pl):
        return _FULL_LAYER.get(pl, pl) if full else pl

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
                    if full:
                        meta.update({"prefill_mode": prefill_mode,
                                     "prefill_batch": prefill_batch, "n_chunks": n_chunks})
                    try:
                        chunk_fn, *_ = K.build_prefill_fn(resolve_pf(pl), cfg,
                                                          batch=prefill_batch, tokens=chunk)
                        if n_chunks > 1:                 # multi-chunk prefill (timing proxy)
                            def prefill_fn(_f=chunk_fn, _n=n_chunks):
                                for _ in range(_n):
                                    _f()
                        else:
                            prefill_fn = chunk_fn
                        decode_fn, *_ = build_decode(dl, db, ctx)
                        solo_p = time_solo(prefill_fn, n_warmup, n_measure)
                        solo_d = time_solo(decode_fn, n_warmup, n_measure)
                    except Exception as e:  # noqa: BLE001
                        torch.cuda.empty_cache()
                        for bk, fr in [("sequential", ""), ("two_stream", "")] + \
                                      [("green_ctx", f) for f in fracs]:
                            rows.append(_failed(meta, bk, fr, type(e).__name__, schema))
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
                        rows.append(_failed(meta, "two_stream", "", type(e).__name__, schema))

                    for fr in fracs:
                        s_p, s_d, info = create_two_partitions(n_first_sm=max(1, round(fr * total_sm)))
                        if s_p is None:
                            rows.append(_failed(meta, "green_ctx", fr,
                                                info.get("error", "no_green_ctx")[:30], schema))
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
                            rows.append(_failed(meta, "green_ctx", fr, type(e).__name__, schema))

                    # decode-protective split: decode reserved its E3 floor, prefill = rest
                    if decode_protect:
                        psm, dsm = _protect_prefill_sm(floor_lut, dl, db, ctx, total_sm)
                        frdp = round(psm / total_sm, 3) if psm else ""
                        if psm is None:
                            rows.append(_failed(meta, "green_ctx_protect", frdp,
                                                f"no_room(decode_floor={dsm})", schema))
                        else:
                            s_p, s_d, info = create_two_partitions(n_first_sm=psm)
                            if s_p is None:
                                rows.append(_failed(meta, "green_ctx_protect", frdp,
                                                    info.get("error", "no_green_ctx")[:30], schema))
                            else:
                                try:
                                    gc = measure_concurrent(prefill_fn, decode_fn, s_p, s_d,
                                                            n_warmup, n_measure)
                                    rows.append(_row(meta, "green_ctx_protect", frdp,
                                                     info.get("actual_first_sm", ""),
                                                     info.get("actual_second_sm", ""),
                                                     solo_p, solo_d, gc["a_stream_ms"],
                                                     gc["b_stream_ms"], gc["concurrent_ms"], "ok"))
                                except Exception as e:  # noqa: BLE001
                                    rows.append(_failed(meta, "green_ctx_protect", frdp,
                                                        type(e).__name__, schema))

                # per-cell quick summary at the smallest batch (max overlap regime)
                cell = [r for r in rows if r["prefill_layer"] == pl and r["decode_layer"] == dl
                        and r["context_len"] == contexts[0] and r["backend"] == "two_stream"
                        and r["status"] == "ok"]
                if cell:
                    lo = min(cell, key=lambda r: r["decode_batch"])
                    print(f"  pf={pl:4} dec={dl:4} ctx={contexts[0]}  two_stream speedup@db{lo['decode_batch']}"
                          f"={lo['speedup_vs_seq']}x")

    suffix = "_full" if full else ""           # never clobber the microbench CSVs
    out = out_dir / f"serving_coexec{suffix}_{model}_{device}.csv"
    write_labeled_csv(out, rows, schema)
    print(f"  wrote {len(rows)} rows -> {out}")


def parse_args():
    p = argparse.ArgumentParser(description="E5 serving prefill×decode coexistence matrix")
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--decode-batches", nargs="+", type=int, default=DEFAULT_DECODE_BATCHES)
    p.add_argument("--context-lens", nargs="+", type=int, default=DEFAULT_CONTEXTS)
    p.add_argument("--chunk", type=int, default=256, help="prefill chunk tokens")
    p.add_argument("--prefill-layers", nargs="+", default=DEFAULT_PREFILL_LAYERS,
                   choices=["ssm", "ssm_full", "attn", "attn_full"])
    p.add_argument("--decode-layers", nargs="+", default=DEFAULT_DECODE_LAYERS,
                   choices=["attn", "ssm"])
    p.add_argument("--fracs", nargs="+", type=float, default=DEFAULT_FRACS,
                   help="green_ctx prefill SM fractions")
    # --- optional REAL-PREFILL mode (A2): default off → identical microbench ---
    p.add_argument("--prefill-mode", choices=["micro", "full"], default="micro",
                   help="micro=scan/attn token-mixer only (default); "
                        "full=GEMM-inclusive (ssm_full/attn_full), separate _full CSV")
    p.add_argument("--prefill-tokens", type=int, default=0,
                   help="total prefill prompt tokens; if > --chunk → multi-chunk prefill "
                        "(ceil(tokens/chunk) chunk calls). 0 = single chunk (default).")
    p.add_argument("--prefill-batch", type=int, default=1,
                   help="number of in-flight prefill sequences (default 1)")
    # --- decode-protective split (SLO direction, A5): reserve decode its E3 floor ---
    p.add_argument("--decode-protect", action="store_true",
                   help="add a 'green_ctx_protect' backend: reserve decode its E3 saturation-"
                        "floor SMs, prefill gets the rest (decode-protective / SLO-oriented split)")
    p.add_argument("--e3-dir", type=Path, default=Path(_CHAR) / "results_v2" / "e3",
                   help="E3 decode_floor CSV dir (used by --decode-protect)")
    p.add_argument("--dry-run", action="store_true",
                   help="print the sweep plan (no GPU, no timing) and exit")
    p.add_argument("--n-warmup", type=int, default=5)
    p.add_argument("--n-measure", type=int, default=20)
    p.add_argument("--output-dir", type=Path, default=Path(_CHAR) / "results_v2" / "e5")
    return p.parse_args()


def _print_plan(args):
    """GPU-free sanity print of what the sweep would run (validates control logic)."""
    full = args.prefill_mode == "full"
    req = args.prefill_tokens if args.prefill_tokens > 0 else args.chunk
    n_chunks = max(1, -(-req // args.chunk))
    ptok = n_chunks * args.chunk
    resolve = (lambda pl: _FULL_LAYER.get(pl, pl)) if full else (lambda pl: pl)
    suffix = "_full" if full else ""
    n_cells = (len(args.prefill_layers) * len(args.decode_layers)
               * len(args.context_lens) * len(args.decode_batches))
    print(f"=== E5 DRY-RUN plan ===")
    print(f"  mode={args.prefill_mode}  schema={'FULL(+prefill_mode,prefill_batch,n_chunks)' if full else 'micro'}")
    print(f"  prefill: layers={args.prefill_layers} → resolved={[resolve(pl) for pl in args.prefill_layers]}")
    print(f"           batch={args.prefill_batch}  tokens={ptok} ({n_chunks}×{args.chunk} chunk)")
    print(f"  decode: layers={args.decode_layers}  batches={args.decode_batches}")
    print(f"  contexts={args.context_lens}  fracs(green_ctx)={args.fracs}")
    nb = 2 + len(args.fracs) + (1 if args.decode_protect else 0)
    print(f"  cells/model={n_cells}  backends/cell={nb} (sequential,two_stream,green_ctx×{len(args.fracs)}"
          f"{',green_ctx_protect' if args.decode_protect else ''})")
    print(f"  models={args.models}")
    print(f"  output → serving_coexec{suffix}_<model>_<device>.csv  (microbench CSVs untouched)")
    if args.decode_protect:
        import glob as _g
        m0 = args.models[0]
        cands = _g.glob(str(args.e3_dir / f"decode_floor_{m0}_*.csv"))
        print(f"  --decode-protect: decode reserved its E3 floor → prefill = 108 - floor:")
        if not cands:
            print(f"    ⚠ no E3 floor CSV for {m0} in {args.e3_dir} (would skip protect rows)")
        else:
            flut = _load_decode_floor(m0, os.path.basename(cands[0]).split(f"{m0}_")[1][:-4], args.e3_dir)
            for dl in args.decode_layers:
                cells = []
                for db in args.decode_batches:
                    psm, dsm = _protect_prefill_sm(flut, dl, db, args.context_lens[0], 108)
                    cells.append(f"db{db}:dec{dsm}/pf{psm if psm else 'X'}")
                print(f"    dec={dl} (ctx{args.context_lens[0]}): " + "  ".join(cells))


def main():
    args = parse_args()
    if args.dry_run:
        _print_plan(args)
        return
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
                  args.n_warmup, args.n_measure, total_sm, device, args.output_dir,
                  prefill_mode=args.prefill_mode, prefill_tokens=args.prefill_tokens,
                  prefill_batch=args.prefill_batch,
                  decode_protect=args.decode_protect, e3_dir=args.e3_dir)


if __name__ == "__main__":
    main()
