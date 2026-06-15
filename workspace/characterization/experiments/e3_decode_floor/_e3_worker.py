"""
E3 subprocess worker — decode latency at ONE SM count, all (layer,batch,context).

Decode kernels: SSM step (seq_len=1) and attention KV read (1 query token reading
`context_len` KV) — both memory-bound. One process per SM level, per-config
try/except, BW co-measured. Returns raw latency samples for the floor CI.

stdout: JSON {"sm": int, "rows": [...]} (each row has raw "latencies_ms").
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

# reuse E2's raw measurement helper
sys.path.insert(0, os.path.join(_CHAR, "experiments", "e2_sm_saturation"))
from _e2_worker import _measure_raw, _is_cuda_fatal  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--model", required=True)
    p.add_argument("--sm-count", type=int, required=True)
    p.add_argument("--total-sm", type=int, required=True)
    p.add_argument("--layer-types", nargs="+", default=["ssm", "attn"])
    p.add_argument("--batches", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
    p.add_argument("--context-lens", nargs="+", type=int, default=[1024, 4096, 16384])
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
        from experiments.common.bw_probe import BWProbe

        cfg = K.load_layer_cfg(args.model)
        probe = BWProbe()
        smctrl = SMController(total_sm_count=args.total_sm)
        cuda_dead = False

        for lt in args.layer_types:
            for batch in args.batches:
                for ctx in args.context_lens:
                    if cuda_dead:
                        rows.append(_stub(args, lt, batch, ctx, "skipped:cuda_dead"))
                        continue
                    try:
                        if lt == "ssm":
                            fn, rb, wb, meta = K.build_decode_ssm_fn(cfg, batch)
                        else:
                            fn, rb, wb, meta = K.build_decode_attn_fn(cfg, batch, ctx)
                        smctrl.set_sm_count(args.sm_count)
                        stream = smctrl.get_stream()
                        try:
                            lats = _measure_raw(fn, stream, args.n_warmup, args.n_measure)
                        finally:
                            smctrl.reset()
                        med = statistics.median(lats)
                        bw = probe.from_bytes(rb, wb, med)
                        rows.append({
                            "model": args.model, "config_status": cfg["config_status"],
                            "layer_type": lt, "batch": batch, "context_len": ctx,
                            "sm_count": args.sm_count,
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
                        rows.append(_stub(args, lt, batch, ctx, "failed:oom"))
                    except Exception as e:  # noqa: BLE001
                        rows.append(_stub(args, lt, batch, ctx, f"failed:{type(e).__name__}"))
                        if _is_cuda_fatal(str(e)):
                            cuda_dead = True
        print(json.dumps({"sm": args.sm_count, "rows": rows}))
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"sm": args.sm_count, "rows": rows,
                          "fatal": f"{type(e).__name__}: {e}",
                          "trace": traceback.format_exc()[-600:]}))


def _stub(args, lt, batch, ctx, status) -> dict:
    return {"model": args.model, "config_status": "", "layer_type": lt, "batch": batch,
            "context_len": ctx, "sm_count": args.sm_count, "latency_ms": "",
            "latency_std_ms": "", "achieved_bw_gbs": "", "bw_util_pct": "",
            "n_measure": 0, "latencies_ms": [], "status": status}


if __name__ == "__main__":
    main()
