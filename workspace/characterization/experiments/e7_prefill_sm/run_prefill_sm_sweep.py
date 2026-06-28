"""Prefill SM-sensitivity sweep — does giving prefill MORE SMs speed it up?

This is the second premise of the layer_aware lever (reclaim SMs to prefill during ssm
layers). It was measured for 2.7b (ssm-prefill 0.888ms@108 -> 5.506ms@14SM = 6.2x) but for
7B only down to 54 SM (8.782@108 -> 8.962@54 = 1.02x); the <54 SM range was never measured.
This closes that gap by timing the prefill kernel SOLO on a single Green-Context partition of
N SMs, for N in {14..108}, on each model. Sensitivity = latency(min SM) / latency(108 SM).

GPU required. Output: results_v2/e7_prefill_sm/prefill_sm_{model}_{device}.csv
"""
from __future__ import annotations
import argparse, csv, os, statistics, sys
from pathlib import Path

_here = os.path.dirname(os.path.abspath(__file__))
_CHAR = os.path.abspath(os.path.join(_here, "..", ".."))
_WS = os.path.abspath(os.path.join(_here, "..", "..", ".."))
for _p in (_WS, _CHAR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

SM_POINTS = [14, 27, 40, 54, 68, 81, 94, 108]   # 108 = full GPU


def time_on_stream(fn, stream, nw, nm):
    import torch
    ctx = torch.cuda.stream(stream) if stream is not None else _null()
    with ctx:
        for _ in range(nw):
            fn()
    torch.cuda.synchronize()
    lats = []
    for _ in range(nm):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        if stream is not None:
            with torch.cuda.stream(stream):
                e0.record(stream); fn(); e1.record(stream)
        else:
            e0.record(); fn(); e1.record()
        torch.cuda.synchronize()
        lats.append(e0.elapsed_time(e1))
    return statistics.median(lats)


class _null:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def run_model(model, layer_types, batch, tokens, ctx, nw, nm, device, out_dir):
    import torch
    from experiments.common import kernels as K
    from experiments.e4_concurrent._green_ctx import create_two_partitions

    cfg = K.load_layer_cfg(model)
    rows = []
    print(f"\n=== prefill SM-sweep: {model} (batch={batch} tokens={tokens} ctx={ctx}) ===")
    for lt in layer_types:
        try:
            fn, *_ = K.build_prefill_fn(lt, cfg, batch=batch, tokens=tokens,
                                        context_len=(ctx if lt == "attn" else 0))
            base = time_on_stream(fn, None, nw, nm)            # full GPU
        except Exception as e:  # noqa: BLE001
            print(f"  {lt}: BUILD/BASE FAILED: {type(e).__name__}: {e}")
            rows.append({"model": model, "device": device, "layer_type": lt,
                         "n_sm_req": -1, "n_sm_actual": -1, "prefill_ms": "",
                         "ratio_vs_full": "", "status": f"failed:{type(e).__name__}"})
            continue
        for N in SM_POINTS:
            if N >= 108:
                ms, actual = base, 108
            else:
                s1, s2, info = create_two_partitions(n_first_sm=N)
                if s1 is None:
                    rows.append({"model": model, "device": device, "layer_type": lt,
                                 "n_sm_req": N, "n_sm_actual": -1, "prefill_ms": "",
                                 "ratio_vs_full": "", "status": f"failed:{info.get('error','')[:20]}"})
                    continue
                actual = info.get("actual_first_sm", N)
                try:
                    ms = time_on_stream(fn, s1, nw, nm)
                except Exception as e:  # noqa: BLE001
                    torch.cuda.empty_cache()
                    rows.append({"model": model, "device": device, "layer_type": lt,
                                 "n_sm_req": N, "n_sm_actual": actual, "prefill_ms": "",
                                 "ratio_vs_full": "", "status": f"failed:{type(e).__name__}"})
                    continue
            rows.append({"model": model, "device": device, "layer_type": lt,
                         "n_sm_req": N, "n_sm_actual": actual, "prefill_ms": round(ms, 4),
                         "ratio_vs_full": round(ms / base, 3), "status": "ok"})
        ok = [r for r in rows if r["layer_type"] == lt and r["status"] == "ok"]
        if ok:
            lo, hi = min(ok, key=lambda r: r["n_sm_actual"]), max(ok, key=lambda r: r["n_sm_actual"])
            print(f"  {lt}: {hi['prefill_ms']}ms@{hi['n_sm_actual']}SM -> {lo['prefill_ms']}ms@{lo['n_sm_actual']}SM "
                  f"= {lo['ratio_vs_full']}x  (sensitivity)")
    out = out_dir / f"prefill_sm_{model}_{device}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"  wrote {len(rows)} rows -> {out}")


def main():
    p = argparse.ArgumentParser(description="prefill SM-sensitivity sweep")
    p.add_argument("--models", nargs="+", default=["zamba2_7b", "zamba2_2.7b"])
    p.add_argument("--layers", nargs="+", default=["ssm", "attn"], choices=["ssm", "attn"])
    p.add_argument("--batch", type=int, default=8)        # E5 prefill_batch
    p.add_argument("--tokens", type=int, default=256)     # E5 chunk
    p.add_argument("--ctx", type=int, default=4096)
    p.add_argument("--n-warmup", type=int, default=5)
    p.add_argument("--n-measure", type=int, default=30)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if a.dry_run:
        print(f"DRY: models={a.models} layers={a.layers} SM={SM_POINTS} batch={a.batch} tokens={a.tokens}")
        return
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    from shared import sweep_spec as ss
    device = ss.canonical_device()
    out_dir = Path(_CHAR) / "results_v2" / "e7_prefill_sm"
    out_dir.mkdir(parents=True, exist_ok=True)
    for m in a.models:
        run_model(m, a.layers, a.batch, a.tokens, a.ctx, a.n_warmup, a.n_measure, device, out_dir)


if __name__ == "__main__":
    main()
