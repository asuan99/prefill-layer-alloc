"""
E1 subprocess worker — measures one prefill config's component decomposition.

Runs in its own process (CUDA-context isolation): if a kernel corrupts the
context, only this worker dies and the parent records status=failed for the
config and continues.

Components measured (SM=108, default stream — E1 does not restrict SMs):
  in_proj  (GEMM)   |  scan (SSM token mixer)  |  out_proj (GEMM)
  attn     (attention token mixer, reference)

stdout: one JSON object {"rows": [...], "status": "ok"|"failed", "error": ...}.
Each row carries MEASURED latency_ms + achieved_bw_gbs; DERIVED bw_util_pct;
the parent adds scan_share_pct.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

_here = os.path.dirname(os.path.abspath(__file__))
_CHAR = os.path.abspath(os.path.join(_here, "..", ".."))
_WORKSPACE = os.path.abspath(os.path.join(_here, "..", "..", ".."))
for _p in (_WORKSPACE, _CHAR):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def main() -> None:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--model", required=True)
    p.add_argument("--batch", type=int, required=True)
    p.add_argument("--seq", type=int, required=True)
    p.add_argument("--chunk", required=True)            # "256"/"512"/"full"
    p.add_argument("--context-len", type=int, default=0)
    p.add_argument("--n-warmup", type=int, default=20)
    p.add_argument("--n-measure", type=int, default=50)
    args = p.parse_args()

    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required")
        from src.profiling.metrics import LatencyMeter
        from experiments.common import kernels as K
        from experiments.common.bw_probe import BWProbe

        cfg = K.load_layer_cfg(args.model)
        tokens = args.seq if args.chunk == "full" else min(int(args.chunk), args.seq)
        meter = LatencyMeter(device="cuda")
        probe = BWProbe()

        builders = [
            ("in_proj", *K.build_in_proj_fn(cfg, args.batch, tokens)),
            ("scan",    *K.build_ssm_scan_fn(cfg, args.batch, tokens)),
            ("out_proj", *K.build_out_proj_fn(cfg, args.batch, tokens)),
            ("attn",    *K.build_attn_fn(cfg, args.batch, tokens, args.context_len)),
        ]

        rows = []
        for name, fn, rb, wb, meta in builders:
            res = meter.measure(fn, n_warmup=args.n_warmup, n_measure=args.n_measure)
            bw = probe.from_bytes(rb, wb, res["latency_ms"])
            rows.append({
                "model": args.model,
                "config_status": cfg["config_status"],
                "layer_type": "ssm",
                "component": name,
                "batch": args.batch,
                "seq": args.seq,
                "chunk_granularity": args.chunk,
                "tokens_per_call": tokens,
                "context_len": args.context_len,
                "sm_count": 108,
                "latency_ms": round(res["latency_ms"], 5),
                "latency_std_ms": round(res["latency_std_ms"], 5),
                "achieved_bw_gbs": round(bw["achieved_bw_gbs"], 3),
                "bw_util_pct": round(bw["bw_util_pct"], 3),
                "n_blocks_per_call": meta.get("n_blocks_per_call", ""),
                "status": "ok",
            })
        print(json.dumps({"rows": rows, "status": "ok"}))
    except Exception as e:  # noqa: BLE001 — isolate any failure to this process
        print(json.dumps({
            "rows": [], "status": "failed",
            "error": f"{type(e).__name__}: {e}",
            "trace": traceback.format_exc()[-800:],
        }))


if __name__ == "__main__":
    main()
