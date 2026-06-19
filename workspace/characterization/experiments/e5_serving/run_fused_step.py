"""Measure a TRUE fused mixed-batch layer step — the quantity E5 never measured.

E5 timed prefill and decode as SEPARATE kernels (solo_prefill, solo_decode, two_stream).
A vLLM/SGLang fused step runs P prefill tokens + B decode tokens in ONE forward, so the
projection GEMMs run over (P+B) rows with weights loaded ONCE — a saving the separate-kernel
reconstruction (solo_prefill + frac*solo_decode) cannot capture. This resolves the queue-sim
fused-vs-partition question (queue_simulator_design §12(4)).

fused step (per layer type):
  ssm : in_proj(P+B) ; scan(P) ; decode_ssm(B) ; out_proj(P+B)
  attn: qkv_proj(P+B); prefill_attn(P); decode_attn(B,ctx); o_proj(P+B)
Also records prefill_only(P) and decode_only(B) (full-layer) for reference and to compute
the fusion saving = (prefill_only + decode_only) - fused.

GPU required. Output: results_v2/e5_fused/fused_step_{model}_{device}.csv
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_here = os.path.dirname(os.path.abspath(__file__))
_CHAR = os.path.abspath(os.path.join(_here, "..", ".."))
_WS = os.path.abspath(os.path.join(_here, "..", "..", ".."))
for _p in (_WS, _CHAR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shared import sweep_spec as ss
from experiments.common.labels import Label, write_labeled_csv

DEFAULT_MODELS = ["zamba2_2.7b", "zamba2_7b", "falcon_h1_3b"]
DEFAULT_DECODE_BATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
DEFAULT_PREFILL_TOKENS = [256, 2048]          # budget 1 and 8 (× chunk 256)
DEFAULT_CONTEXTS = [4096]
DEFAULT_LAYERS = ["ssm", "attn"]

_SCHEMA = {
    "model": Label.METADATA, "device": Label.METADATA, "config_status": Label.METADATA,
    "layer_type": Label.METADATA, "prefill_tokens": Label.METADATA,
    "decode_batch": Label.METADATA, "context_len": Label.METADATA, "status": Label.METADATA,
    "fused_ms": Label.MEASURED, "prefill_only_ms": Label.MEASURED, "decode_only_ms": Label.MEASURED,
    "fusion_saving_ms": Label.DERIVED,   # (prefill_only + decode_only) - fused  (weight-load saving)
}


def _fns(K, layer_type, cfg, P, B, ctx):
    """Return (fused_fn, prefill_only_fn, decode_only_fn)."""
    if layer_type == "ssm":
        in_pb, *_ = K.build_in_proj_fn(cfg, 1, P + B)
        in_p, *_ = K.build_in_proj_fn(cfg, 1, P)
        in_b, *_ = K.build_in_proj_fn(cfg, 1, B)
        out_pb, *_ = K.build_out_proj_fn(cfg, 1, P + B)
        out_p, *_ = K.build_out_proj_fn(cfg, 1, P)
        out_b, *_ = K.build_out_proj_fn(cfg, 1, B)
        pre, *_ = K.build_ssm_scan_fn(cfg, 1, P)
        dec, *_ = K.build_decode_ssm_fn(cfg, B)
    else:  # attn
        in_pb, *_ = K.build_attn_qkv_proj_fn(cfg, 1, P + B)
        in_p, *_ = K.build_attn_qkv_proj_fn(cfg, 1, P)
        in_b, *_ = K.build_attn_qkv_proj_fn(cfg, 1, B)
        out_pb, *_ = K.build_attn_o_proj_fn(cfg, 1, P + B)
        out_p, *_ = K.build_attn_o_proj_fn(cfg, 1, P)
        out_b, *_ = K.build_attn_o_proj_fn(cfg, 1, B)
        pre, *_ = K.build_attn_fn(cfg, 1, P, 0)            # prefill: causal over P
        dec, *_ = K.build_decode_attn_fn(cfg, B, ctx)      # decode: B tokens read full KV

    def fused():
        in_pb(); pre(); dec(); out_pb()

    def prefill_only():
        in_p(); pre(); out_p()

    def decode_only():
        in_b(); dec(); out_b()

    return fused, prefill_only, decode_only


def run_model(model, layers, prefill_tokens, decode_batches, contexts,
              n_warmup, n_measure, device, out_dir: Path):
    import torch
    from experiments.common import kernels as K
    from experiments.e4_concurrent._green_ctx import time_solo

    cfg = K.load_layer_cfg(model)
    status_meta = cfg["config_status"]
    rows = []
    print(f"\n=== fused-step: {model} ===")
    for lt in layers:
        for P in prefill_tokens:
            for ctx in contexts:
                for B in decode_batches:
                    meta = {"model": model, "device": device, "config_status": status_meta,
                            "layer_type": lt, "prefill_tokens": P, "decode_batch": B,
                            "context_len": ctx}
                    try:
                        fused, p_only, d_only = _fns(K, lt, cfg, P, B, ctx)
                        fm = time_solo(fused, n_warmup, n_measure)
                        pm = time_solo(p_only, n_warmup, n_measure)
                        dm = time_solo(d_only, n_warmup, n_measure)
                        rnd = lambda v: round(v, 4)
                        rows.append({**meta, "status": "ok", "fused_ms": rnd(fm),
                                     "prefill_only_ms": rnd(pm), "decode_only_ms": rnd(dm),
                                     "fusion_saving_ms": rnd(pm + dm - fm)})
                    except Exception as e:  # noqa: BLE001
                        torch.cuda.empty_cache()
                        base = dict.fromkeys(_SCHEMA, "")
                        base.update(meta); base["status"] = f"failed:{type(e).__name__}"
                        rows.append(base)
                        print(f"  {lt} P={P} ctx={ctx} B={B:>3}  FAILED: {type(e).__name__}")
            # one-line preview per (layer,P)
            ok = [r for r in rows if r["layer_type"] == lt and r["prefill_tokens"] == P and r["status"] == "ok"]
            if ok:
                lo, hi = ok[0], ok[-1]
                print(f"  {lt} P={P}: fused B{lo['decode_batch']}={lo['fused_ms']}ms "
                      f"B{hi['decode_batch']}={hi['fused_ms']}ms (vs prefill_only {lo['prefill_only_ms']}ms)")
    out = out_dir / f"fused_step_{model}_{device}.csv"
    write_labeled_csv(out, rows, _SCHEMA)
    print(f"  wrote {len(rows)} rows -> {out}")


def parse_args():
    p = argparse.ArgumentParser(description="True fused mixed-batch layer-step measurement")
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--layers", nargs="+", default=DEFAULT_LAYERS, choices=["ssm", "attn"])
    p.add_argument("--prefill-tokens", nargs="+", type=int, default=DEFAULT_PREFILL_TOKENS)
    p.add_argument("--decode-batches", nargs="+", type=int, default=DEFAULT_DECODE_BATCHES)
    p.add_argument("--context-lens", nargs="+", type=int, default=DEFAULT_CONTEXTS)
    p.add_argument("--n-warmup", type=int, default=5)
    p.add_argument("--n-measure", type=int, default=20)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--output-dir", type=Path, default=Path(_CHAR) / "results_v2" / "e5_fused")
    return p.parse_args()


def main():
    a = parse_args()
    if a.dry_run:
        n = len(a.models) * len(a.layers) * len(a.prefill_tokens) * len(a.context_lens) * len(a.decode_batches)
        print(f"=== fused-step DRY-RUN ===\n  models={a.models} layers={a.layers}\n"
              f"  prefill_tokens={a.prefill_tokens} decode_batches={a.decode_batches} ctx={a.context_lens}\n"
              f"  cells={n}  (fused + prefill_only + decode_only timed each)\n"
              f"  output → fused_step_<model>_<device>.csv")
        return
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required")
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"fused-step needs a CUDA GPU: {e}")
    device = ss.canonical_device()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    for m in a.models:
        run_model(m, a.layers, a.prefill_tokens, a.decode_batches, a.context_lens,
                  a.n_warmup, a.n_measure, device, a.output_dir)


if __name__ == "__main__":
    main()
