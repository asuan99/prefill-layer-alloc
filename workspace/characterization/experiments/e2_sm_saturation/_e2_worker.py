"""
E2 subprocess worker — measures all (layer_type,batch,chunk) configs at ONE SM
count under Green Context restriction.

One process per SM level (like the v1 _ssm_worker): if the SSM cooperative kernel
hits an illegal memory access at low SM, or deadlocks, only this process dies —
the parent applies a timeout, records status=failed for that SM level, and keeps
the rest of the sweep. Per-config try/except isolates non-fatal errors.

Bandwidth is co-measured at every point via BWProbe (the v2 primary instrument).
Raw per-config latency samples (n ≥ 30) are returned so the parent can bootstrap
a saturation-point CI — point estimates are forbidden.

stdout: JSON {"sm": int, "rows": [...]} where each row has median latency, BW,
and the raw "latencies_ms" list. CUDA errors set the per-config status and stop
further configs at that SM level (context is corrupt).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import traceback

_here = os.path.dirname(os.path.abspath(__file__))
_CHAR = os.path.abspath(os.path.join(_here, "..", ".."))
_WORKSPACE = os.path.abspath(os.path.join(_here, "..", "..", ".."))
for _p in (_WORKSPACE, _CHAR):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _measure_raw(fn, stream, n_warmup, n_measure) -> list[float]:
    """CUDA-event latencies (ms) on *stream*; returns the raw sample list."""
    import torch
    for _ in range(n_warmup):
        with torch.cuda.stream(stream):
            fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(n_measure):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(stream):
            e0.record(stream)
            fn()
            e1.record(stream)
        torch.cuda.synchronize()
        out.append(e0.elapsed_time(e1))
    return out


def _is_cuda_fatal(msg: str) -> bool:
    m = msg.lower()
    return ("cuda error" in m or "illegal memory" in m
            or "device-side assert" in m or "cudaerrorillegaladdress" in m.lower())


def main() -> None:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--model", required=True)
    p.add_argument("--sm-count", type=int, required=True)
    p.add_argument("--total-sm", type=int, required=True)
    p.add_argument("--layer-types", nargs="+", default=["ssm", "ssm_full", "attn"])
    p.add_argument("--batches", nargs="+", type=int, default=[1, 8, 32, 128])
    p.add_argument("--chunks", nargs="+", default=["256", "512"])
    p.add_argument("--seq", type=int, default=2048)
    p.add_argument("--context-lens", nargs="+", type=int, default=[0, 4096])
    p.add_argument("--n-warmup", type=int, default=10)
    p.add_argument("--n-measure", type=int, default=30)
    args = p.parse_args()

    rows = []
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required")
        from src.smctrl import SMController
        from experiments.common import kernels as K
        from experiments.common import wave_model as wm
        from experiments.common.bw_probe import BWProbe

        cfg = K.load_layer_cfg(args.model)
        probe = BWProbe()
        smctrl = SMController(total_sm_count=args.total_sm)
        cuda_dead = False

        for lt in args.layer_types:
            ctx_list = args.context_lens if lt == "attn" else [0]
            for batch in args.batches:
                for chunk in args.chunks:
                    for ctx in ctx_list:
                        if cuda_dead:
                            rows.append(_stub(args, lt, batch, chunk, ctx,
                                              "skipped:cuda_dead"))
                            continue
                        tokens = (args.seq if chunk == "full"
                                  else min(int(chunk), args.seq))
                        try:
                            fn, rb, wb, meta = K.build_prefill_fn(
                                lt, cfg, batch, tokens, context_len=ctx)
                            smctrl.set_sm_count(args.sm_count)
                            stream = smctrl.get_stream()
                            try:
                                lats = _measure_raw(fn, stream,
                                                    args.n_warmup, args.n_measure)
                            finally:
                                smctrl.reset()
                            med = statistics.median(lats)
                            bw = probe.from_bytes(rb, wb, med)
                            nb = meta.get("n_blocks_per_call", 0)
                            rows.append({
                                "model": args.model,
                                "config_status": cfg["config_status"],
                                "layer_type": lt,
                                "batch": batch,
                                "seq": args.seq,
                                "chunk_granularity": chunk,
                                "tokens_per_call": tokens,
                                "context_len": ctx,
                                "sm_count": args.sm_count,
                                "n_blocks_per_call": nb,
                                "waves": round(wm.waves(nb, args.sm_count), 4) if nb else "",
                                "latency_ms": round(med, 5),
                                "latency_std_ms": round(
                                    statistics.pstdev(lats) if len(lats) > 1 else 0.0, 5),
                                "achieved_bw_gbs": round(bw["achieved_bw_gbs"], 3),
                                "bw_util_pct": round(bw["bw_util_pct"], 3),
                                "n_measure": len(lats),
                                "latencies_ms": [round(x, 5) for x in lats],
                                "status": "ok",
                            })
                        except torch.cuda.OutOfMemoryError:
                            torch.cuda.empty_cache()
                            rows.append(_stub(args, lt, batch, chunk, ctx, "failed:oom"))
                        except Exception as e:  # noqa: BLE001
                            rows.append(_stub(args, lt, batch, chunk, ctx,
                                              f"failed:{type(e).__name__}"))
                            if _is_cuda_fatal(str(e)):
                                cuda_dead = True
        print(json.dumps({"sm": args.sm_count, "rows": rows}))
    except Exception as e:  # noqa: BLE001 — top-level isolation
        print(json.dumps({"sm": args.sm_count, "rows": rows,
                          "fatal": f"{type(e).__name__}: {e}",
                          "trace": traceback.format_exc()[-600:]}))


def _stub(args, lt, batch, chunk, ctx, status) -> dict:
    return {
        "model": args.model, "config_status": "", "layer_type": lt, "batch": batch,
        "seq": args.seq, "chunk_granularity": chunk, "tokens_per_call": "",
        "context_len": ctx, "sm_count": args.sm_count, "n_blocks_per_call": "",
        "waves": "", "latency_ms": "", "latency_std_ms": "", "achieved_bw_gbs": "",
        "bw_util_pct": "", "n_measure": 0, "latencies_ms": [], "status": status,
    }


if __name__ == "__main__":
    main()
