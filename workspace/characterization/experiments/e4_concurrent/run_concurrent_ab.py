"""
E4 — concurrent A/B (A100). Runs ONLY if G1 = BW_MECHANISM.

Compares a uniform SM budget against a layer-type-aware budget at the SAME total
SM, using REAL concurrent execution (two Green Context partitions enqueued
together — never a simulation).

Two measured variables
-----------------------
(a) prefill–prefill overlap throughput
      Two prefill kernels (e.g. the SSM and Attn branches) run concurrently;
      overlap_ratio = concurrent / (solo_a + solo_b) and throughput_gain its
      reciprocal. uniform (50/50) vs aware split, same total SM.

(b) prefill+decode coexistence — decode latency inflation vs prefill LAYER TYPE
      A Green Context splits SMs but NOT HBM bandwidth, so a bandwidth-heavy SSM
      prefill should inflate a co-running decode more than a compute-heavy attn
      prefill. We measure decode_inflation for prefill_layer_type ∈ {ssm, attn}
      to expose this **layer-type-dependent bandwidth interference**.

Every prefill kernel is built through kernels.build_prefill_fn — the single
dispatch that fixes B-1 (an "attn" config can never run an SSM kernel here).
test_dispatch.py guards that.

Outputs → results_v2/e4/
  concurrent_pp_{model}_{device}.csv          (a)
  decode_interference_{model}_{device}.csv    (b)
"""

from __future__ import annotations

import argparse
import json
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

_PP_SCHEMA = {
    "model": Label.METADATA, "device": Label.METADATA, "config_status": Label.METADATA,
    "budget_mode": Label.METADATA, "a_layer_type": Label.METADATA,
    "b_layer_type": Label.METADATA, "batch": Label.METADATA, "seq": Label.METADATA,
    "chunk_granularity": Label.METADATA, "prefill_sm": Label.METADATA,
    "other_sm": Label.METADATA, "total_sm": Label.METADATA, "status": Label.METADATA,
    "solo_a_ms": Label.MEASURED, "solo_b_ms": Label.MEASURED,
    "concurrent_ms": Label.MEASURED,
    "overlap_ratio": Label.DERIVED, "throughput_gain": Label.DERIVED,
}

_DI_SCHEMA = {
    "model": Label.METADATA, "device": Label.METADATA, "config_status": Label.METADATA,
    "budget_mode": Label.METADATA, "prefill_layer_type": Label.METADATA,
    "batch": Label.METADATA, "prefill_seq": Label.METADATA,
    "chunk_granularity": Label.METADATA, "context_len": Label.METADATA,
    "prefill_sm": Label.METADATA, "decode_sm": Label.METADATA, "status": Label.METADATA,
    "decode_solo_ms": Label.MEASURED, "decode_concurrent_ms": Label.MEASURED,
    "prefill_concurrent_ms": Label.MEASURED,
    "decode_inflation_pct": Label.DERIVED,
}


# ---------------------------------------------------------------------------
# G1 gate
# ---------------------------------------------------------------------------

def _check_g1(verdict_path: Path, force: bool) -> bool:
    if not verdict_path.exists():
        print(f"  G1 verdict not found ({verdict_path}). Run gates/adjudicate.py.")
        return force
    v = json.loads(verdict_path.read_text()).get("verdict")
    if v == "BW_MECHANISM":
        print(f"  G1 = BW_MECHANISM → E4 authorised.")
        return True
    print(f"  G1 = {v} → E4 is NOT authorised as the headline result.")
    if force:
        print("  --force set: running anyway (pipeline validation only).")
    return force


# ---------------------------------------------------------------------------
# (a) prefill-prefill overlap
# ---------------------------------------------------------------------------

def measure_pp(model, cfg, batch, seq, chunk, a_lt, b_lt, prefill_frac,
               budget_mode, total_sm, device, n_warmup, n_measure) -> dict:
    from experiments.common import kernels as K
    from experiments.e4_concurrent._green_ctx import (
        create_two_partitions, measure_concurrent, time_solo)
    tokens = seq if chunk == "full" else min(int(chunk), seq)
    fn_a, *_ = K.build_prefill_fn(a_lt, cfg, batch, tokens)          # B-1 dispatch
    fn_b, *_ = K.build_prefill_fn(b_lt, cfg, batch, tokens)

    n_first = max(1, round(prefill_frac * total_sm))
    s_a, s_b, info = create_two_partitions(n_first_sm=n_first)
    base = {
        "model": model, "device": device, "config_status": cfg["config_status"],
        "budget_mode": budget_mode, "a_layer_type": a_lt, "b_layer_type": b_lt,
        "batch": batch, "seq": seq, "chunk_granularity": chunk,
        "prefill_sm": n_first, "other_sm": total_sm - n_first, "total_sm": total_sm,
    }
    if s_a is None:
        return {**base, "status": f"failed:{info.get('error','no_green_ctx')}",
                "solo_a_ms": "", "solo_b_ms": "", "concurrent_ms": "",
                "overlap_ratio": "", "throughput_gain": ""}
    solo_a = time_solo(fn_a, n_warmup, n_measure)
    solo_b = time_solo(fn_b, n_warmup, n_measure)
    conc = measure_concurrent(fn_a, fn_b, s_a, s_b, n_warmup, n_measure)["concurrent_ms"]
    denom = solo_a + solo_b
    return {**base, "status": "ok",
            "solo_a_ms": round(solo_a, 4), "solo_b_ms": round(solo_b, 4),
            "concurrent_ms": round(conc, 4),
            "overlap_ratio": round(conc / denom, 4) if denom else "",
            "throughput_gain": round(denom / conc, 4) if conc else ""}


# ---------------------------------------------------------------------------
# (b) prefill+decode coexistence — decode inflation vs prefill layer type
# ---------------------------------------------------------------------------

def measure_decode_interference(model, cfg, batch, prefill_seq, chunk,
                                prefill_lt, context_len, prefill_frac, budget_mode,
                                total_sm, device, n_warmup, n_measure) -> dict:
    from experiments.common import kernels as K
    from experiments.e4_concurrent._green_ctx import (
        create_two_partitions, measure_concurrent, time_solo)
    tokens = prefill_seq if chunk == "full" else min(int(chunk), prefill_seq)
    prefill_fn, *_ = K.build_prefill_fn(prefill_lt, cfg, batch, tokens,
                                        context_len=context_len)   # B-1 dispatch
    decode_fn, *_ = K.build_decode_attn_fn(cfg, batch, context_len)  # BW-bound decode

    n_first = max(1, round(prefill_frac * total_sm))
    s_prefill, s_decode, info = create_two_partitions(n_first_sm=n_first)
    base = {
        "model": model, "device": device, "config_status": cfg["config_status"],
        "budget_mode": budget_mode, "prefill_layer_type": prefill_lt, "batch": batch,
        "prefill_seq": prefill_seq, "chunk_granularity": chunk,
        "context_len": context_len, "prefill_sm": n_first,
        "decode_sm": total_sm - n_first,
    }
    if s_prefill is None:
        return {**base, "status": f"failed:{info.get('error','no_green_ctx')}",
                "decode_solo_ms": "", "decode_concurrent_ms": "",
                "prefill_concurrent_ms": "", "decode_inflation_pct": ""}
    decode_solo = time_solo(decode_fn, n_warmup, n_measure)
    res = measure_concurrent(prefill_fn, decode_fn, s_prefill, s_decode,
                             n_warmup, n_measure)
    dec_conc = res["b_stream_ms"]
    infl = (dec_conc - decode_solo) / decode_solo * 100.0 if decode_solo else None
    return {**base, "status": "ok",
            "decode_solo_ms": round(decode_solo, 4),
            "decode_concurrent_ms": round(dec_conc, 4),
            "prefill_concurrent_ms": round(res["a_stream_ms"], 4),
            "decode_inflation_pct": round(infl, 3) if infl is not None else ""}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_model(model, args, total_sm, device, out_dir: Path):
    from experiments.common import kernels as K
    cfg = K.load_layer_cfg(model)
    # uniform (50/50) vs layer-type-aware split (same total SM).
    modes = {"uniform_50_50": 0.5, f"aware_{args.aware_frac}": args.aware_frac}

    pp_rows, di_rows = [], []
    for budget_mode, frac in modes.items():
        # (a) prefill-prefill: SSM branch vs Attn branch concurrent.
        pp_rows.append(measure_pp(
            model, cfg, args.batch, args.prefill_seq, args.chunk, "ssm", "attn",
            frac, budget_mode, total_sm, device, args.n_warmup, args.n_measure))
        # (b) decode interference for each prefill layer type.
        for lt in ("ssm", "attn"):
            di_rows.append(measure_decode_interference(
                model, cfg, args.batch, args.prefill_seq, args.chunk, lt,
                args.context_len, frac, budget_mode, total_sm, device,
                args.n_warmup, args.n_measure))

    pp = out_dir / f"concurrent_pp_{model}_{device}.csv"
    di = out_dir / f"decode_interference_{model}_{device}.csv"
    write_labeled_csv(pp, pp_rows, _PP_SCHEMA)
    write_labeled_csv(di, di_rows, _DI_SCHEMA)
    print(f"  wrote {pp}\n  wrote {di}")


def parse_args():
    p = argparse.ArgumentParser(description="E4 concurrent A/B (real concurrency)")
    p.add_argument("--models", nargs="+", default=["zamba2_1.2b", "zamba2_2.7b", "falcon_h1_1.5b", "falcon_h1_3b"])
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--prefill-seq", type=int, default=2048)
    p.add_argument("--chunk", default="256")
    p.add_argument("--context-len", type=int, default=4096)
    p.add_argument("--aware-frac", type=float, default=0.7,
                   help="prefill SM fraction for the layer-type-aware budget")
    p.add_argument("--n-warmup", type=int, default=5)
    p.add_argument("--n-measure", type=int, default=20)
    p.add_argument("--g1-verdict", type=Path,
                   default=Path(_CHAR) / "results_v2" / "verdicts" / "g1_verdict.json")
    p.add_argument("--force", action="store_true",
                   help="run even if G1 != BW_MECHANISM (pipeline validation only)")
    p.add_argument("--output-dir", type=Path, default=Path(_CHAR) / "results_v2" / "e4")
    return p.parse_args()


def main():
    args = parse_args()
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required for E4 (real concurrent execution).")
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"E4 needs a CUDA GPU: {e}")

    if not _check_g1(args.g1_verdict, args.force):
        raise SystemExit("E4 aborted: G1 did not authorise concurrent benefit "
                         "(use --force only for pipeline validation).")

    total_sm = ss.total_sm()
    device = ss.canonical_device()
    for model in args.models:
        print(f"\n=== E4 concurrent A/B: {model} ===")
        run_model(model, args, total_sm, device, args.output_dir)


if __name__ == "__main__":
    main()
