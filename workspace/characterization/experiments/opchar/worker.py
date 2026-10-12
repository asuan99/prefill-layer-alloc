"""worker.py — ONE process = ONE SM partition. Times every opchar operator alone on that
partition, optionally captures a counter-free Kineto trace, writes labelled CSV + JSON.

One process per partition because sgl_kernel.spatial green contexts are never destroyed
(scripts/vessl/substrate_probe_helper.py::cmd_b3_gran_one, same reason).

  python3 worker.py --phase prefill --sizes 1024 4096 --sm 44 --out-dir DIR [--profile]
  python3 worker.py --phase decode  --sizes 32 64 --ctx 2048 --sm 108 --out-dir DIR
  python3 worker.py --phase memcpy  --sm 16 --out-dir DIR          # Bullet Fig 8a analogue

--sm 108 runs on the plain (non-green) stream: the whole device. Every other value creates a
green pair (sm, 108-sm) and runs on the first stream; the SM count the DRIVER reports for
that stream is recorded as n_sm_realized (green_readout), never the request.
"""
from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import os
import platform
import shutil
import statistics
import sys
import time
import traceback
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_CHAR = _HERE.parent.parent
_WS = _CHAR.parent
for _p in (str(_WS), str(_CHAR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments.common.labels import Label, write_labeled_csv  # noqa: E402
from experiments.opchar import ops_spec as S  # noqa: E402

FULL_SM = 108
M, D, X = Label.MEASURED, Label.DERIVED, Label.METADATA


def _green_readout():
    p = _WS / "engine-port" / "src" / "multiplex" / "green_readout.py"
    spec = importlib.util.spec_from_file_location("opchar_green_readout", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def open_partition(n_sm: int):
    """Return (stream, info). n_sm >= FULL_SM -> plain current stream."""
    import torch
    dev = torch.cuda.current_device()
    total = torch.cuda.get_device_properties(dev).multi_processor_count
    if n_sm >= total:
        return torch.cuda.current_stream(), {"mode": "plain", "n_sm_req": n_sm,
                                              "n_sm_realized": total, "device_sm": total}
    from sgl_kernel import spatial
    sa, sb = spatial.create_greenctx_stream_by_value(n_sm, total - n_sm, dev)
    info = {"mode": "green", "n_sm_req": n_sm, "device_sm": total}
    try:
        rd = _green_readout().green_sm_readout(int(sa.cuda_stream))
        info["readout"] = rd
        info["n_sm_realized"] = rd.get("green_sm", rd.get("smCount"))
    except Exception as exc:  # noqa: BLE001
        info["readout_error"] = repr(exc)
        info["n_sm_realized"] = None
    info["_other_stream"] = sb
    return sa, info


def time_fn(fn, stream, n_warmup: int, n_measure: int, inner: int):
    import torch
    with torch.cuda.stream(stream):
        for _ in range(n_warmup):
            fn()
    torch.cuda.synchronize()
    lats = []
    for _ in range(n_measure):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(stream):
            e0.record(stream)
            for _ in range(inner):
                fn()
            e1.record(stream)
        torch.cuda.synchronize()
        lats.append(e0.elapsed_time(e1) / inner)
    lats.sort()
    q = lambda p: lats[min(len(lats) - 1, int(round(p * (len(lats) - 1))))]
    return {"ms_median": statistics.median(lats), "ms_p10": q(0.10), "ms_p90": q(0.90),
            "ms_mean": statistics.fmean(lats), "n": len(lats)}


def time_graph(fn, stream, n_warmup: int, n_measure: int, inner: int):
    """CUDA-graph replay of `inner` calls (decode runs graph-captured in the engine)."""
    import torch
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=stream):
        for _ in range(inner):
            fn()
    torch.cuda.synchronize()
    r = time_fn(g.replay, stream, n_warmup, n_measure, 1)
    return {k: (v / inner if k.startswith("ms_") else v) for k, v in r.items()}


def _versions():
    out = {"python": platform.python_version()}
    for mod in ("torch", "flashinfer", "sgl_kernel", "sglang", "triton"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception as exc:  # noqa: BLE001
            out[mod] = f"unavailable: {type(exc).__name__}"
    return out


SCHEMA_OPS = {
    "model": X, "phase": X, "size": X, "ctx": X, "n_sm_req": X, "n_sm_realized": M,
    "partition_mode": X, "op": X, "layer": X, "category": X, "layer_count": X,
    "timing_mode": X, "inner": X, "status": X,
    "ms_median": M, "ms_p10": M, "ms_p90": M, "ms_mean": M, "n": X,
    "flops": D, "bytes_total": D, "bytes_convention": X,
    "tflops_derived": D, "gbps_derived": D,
}


def _row_base(a, size, info, spec):
    return {"model": a.model, "phase": a.phase, "size": size, "ctx": a.ctx if a.phase == "decode" else 0,
            "n_sm_req": info["n_sm_req"], "n_sm_realized": info.get("n_sm_realized"),
            "partition_mode": info["mode"], "op": spec.key, "layer": spec.layer,
            "category": spec.category, "layer_count": a._counts[spec.layer],
            "flops": spec.flops, "bytes_total": spec.bytes_total,
            "bytes_convention": S.BYTES_CONVENTION}


def run_ops(a, stream, info, out_dir: Path):
    import torch
    from experiments.opchar import builders as Bld
    from experiments.opchar import trace_parse as TP
    cfg = S.cfg_from_models_yaml(a.model)
    a._counts = S.layer_counts(cfg.pattern)
    rows, kernel_rows, notes = [], [], []
    for size in a.sizes:
        tag = f"{a.phase}_{size}" + (f"_ctx{a.ctx}" if a.phase == "decode" else "") + f"_sm{a.sm}"
        try:
            with torch.cuda.stream(stream):
                ops, meta = Bld.build(cfg, a.phase, size, ctx=a.ctx)
            torch.cuda.synchronize()
        except Exception as exc:  # noqa: BLE001
            notes.append({"size": size, "build_error": repr(exc), "tb": traceback.format_exc()[-2000:]})
            torch.cuda.empty_cache()
            continue
        inner = a.inner if a.inner else (1 if a.phase == "prefill" else 10)
        for spec, fn in ops:
            for mode in (("eager", "graph") if a.phase == "decode" else ("eager",)):
                row = _row_base(a, size, info, spec)
                row.update({"timing_mode": mode, "inner": inner})
                try:
                    t = (time_fn if mode == "eager" else time_graph)(fn, stream, a.n_warmup, a.n_measure, inner)
                    row.update(t)
                    row["status"] = "ok"
                    s = t["ms_median"] / 1e3
                    row["tflops_derived"] = spec.flops / s / 1e12 if s > 0 else None
                    row["gbps_derived"] = spec.bytes_total / s / 1e9 if s > 0 else None
                except Exception as exc:  # noqa: BLE001
                    row.update({"status": f"failed:{type(exc).__name__}:{str(exc)[:80]}",
                                "ms_median": None, "ms_p10": None, "ms_p90": None, "ms_mean": None,
                                "n": 0, "tflops_derived": None, "gbps_derived": None})
                rows.append(row)
        if a.profile:
            kernel_rows += profile_ops(a, ops, stream, info, size, tag, out_dir, TP)
        notes.append({"size": size, "meta": {k: v for k, v in meta.items() if not k.startswith("_")}})
        del ops, meta
        torch.cuda.empty_cache()
    return rows, kernel_rows, notes


SCHEMA_KERNELS = {
    "model": X, "phase": X, "size": X, "ctx": X, "n_sm_req": X, "n_sm_realized": M, "op": X,
    "kernel": X, "family": X, "cat": X, "calls": M, "calls_per_iter": D, "dur_us_sum": M,
    "dur_us_per_iter": D, "grid_blocks": M, "threads": M, "regs": M, "smem": M,
    "launch_varies": D, "resident_blocks_per_sm": D, "waves": D, "tail_sms": D, "idle_ratio": D,
}


def profile_ops(a, ops, stream, info, size, tag, out_dir: Path, TP):
    """Counter-free Kineto pass: each op K times inside record_function('opchar:<key>')."""
    import torch
    from torch.profiler import ProfilerActivity, profile, record_function
    K = a.profile_iters
    with torch.cuda.stream(stream):
        for _, fn in ops:
            fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        with torch.cuda.stream(stream):
            for spec, fn in ops:
                with record_function(TP.RANGE_PREFIX + spec.key):
                    for _ in range(K):
                        fn()
        torch.cuda.synchronize()
    raw = out_dir / f"trace_{tag}.json"
    prof.export_chrome_trace(str(raw))
    with open(raw, "rb") as f, gzip.open(str(raw) + ".gz", "wb") as g:
        shutil.copyfileobj(f, g)
    summ = TP.load_trace(raw)
    raw.unlink()
    n_sm = info.get("n_sm_realized") or info["n_sm_req"]
    table = TP.op_kernel_table(summ, {spec.key: K for spec, _ in ops}, int(n_sm))
    out = []
    for r in table:
        out.append({"model": a.model, "phase": a.phase, "size": size,
                    "ctx": a.ctx if a.phase == "decode" else 0, "n_sm_req": info["n_sm_req"],
                    "n_sm_realized": info.get("n_sm_realized"),
                    **{k: r.get(k) for k in SCHEMA_KERNELS if k in r}})
    return out


def run_memcpy(a, stream, info):
    """Bullet Fig 8a analogue: device-to-device copy bandwidth on this partition."""
    import torch
    n = a.memcpy_bytes // 2
    src = torch.empty(n, dtype=torch.bfloat16, device="cuda").normal_()
    dst = torch.empty_like(src)
    t = time_fn(lambda: dst.copy_(src), stream, a.n_warmup, a.n_measure, 1)
    moved = 2 * n * 2
    return [{"n_sm_req": info["n_sm_req"], "n_sm_realized": info.get("n_sm_realized"),
             "partition_mode": info["mode"], "bytes_moved": moved, **t,
             "gbps": moved / (t["ms_median"] / 1e3) / 1e9}]


SCHEMA_MEMCPY = {"n_sm_req": X, "n_sm_realized": M, "partition_mode": X, "bytes_moved": X,
                 "ms_median": M, "ms_p10": M, "ms_p90": M, "ms_mean": M, "n": X, "gbps": D}


def main(argv=None):
    p = argparse.ArgumentParser(description="opchar per-partition worker")
    p.add_argument("--model", default="nemotron_nano_9b_v2")
    p.add_argument("--phase", choices=("prefill", "decode", "memcpy"), required=True)
    p.add_argument("--sizes", type=int, nargs="*", default=[])
    p.add_argument("--ctx", type=int, default=2048)
    p.add_argument("--sm", type=int, required=True)
    p.add_argument("--n-warmup", type=int, default=5)
    p.add_argument("--n-measure", type=int, default=30)
    p.add_argument("--inner", type=int, default=0)
    p.add_argument("--profile", action="store_true")
    p.add_argument("--profile-iters", type=int, default=3)
    p.add_argument("--memcpy-bytes", type=int, default=1 << 30)
    p.add_argument("--out-dir", required=True)
    a = p.parse_args(argv)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    t0 = time.time()
    stream, info = open_partition(a.sm)
    stem = f"{a.phase}_sm{a.sm}" + (f"_ctx{a.ctx}" if a.phase == "decode" else "")
    meta = {"argv": sys.argv, "versions": _versions(), "device": torch.cuda.get_device_name(),
            "partition": {k: v for k, v in info.items() if not k.startswith("_")}}
    if a.phase == "memcpy":
        write_labeled_csv(out_dir / f"{stem}.csv", run_memcpy(a, stream, info), SCHEMA_MEMCPY)
    else:
        rows, krows, notes = run_ops(a, stream, info, out_dir)
        meta["notes"] = notes
        if rows:
            write_labeled_csv(out_dir / f"ops_{stem}.csv", rows, SCHEMA_OPS)
        if krows:
            write_labeled_csv(out_dir / f"kernels_{stem}.csv", krows,
                              {k: v for k, v in SCHEMA_KERNELS.items() if k in krows[0]})
    meta["wall_s"] = time.time() - t0
    (out_dir / f"meta_{stem}.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    print(f"[opchar-worker] {stem} done in {meta['wall_s']:.1f}s -> {out_dir}")


if __name__ == "__main__":
    main()
