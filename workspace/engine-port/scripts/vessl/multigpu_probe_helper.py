#!/usr/bin/env python3
"""multigpu_probe_helper.py -- helper for multigpu_probe.sh (M1 multi-GPU substrate probe).

DRAFT (2026-10-12, not pre-registered, never run on a GPU).  Facts only: NO serving
performance number and NO verdict.  The interconnect numbers (all-reduce / P2P copy) are
substrate descriptors, not results of the engine.  Top-level imports are stdlib only so the
pure parsers run on a CPU node (tests/test_multigpu_probe.py).

Subcommands
  parse-topo   --inp topo_m.txt --out json        nvidia-smi topo -m -> link-class matrix
  parse-nccl   --inp all_reduce_perf.txt --out j  nccl-tests table -> rows
  gpu-facts    --out json                        per-GPU uuid/SM count/clocks/power (torch+NVML)
  p2p          --out json [--mib 256]            can_device_access_peer + ordered-pair copy GB/s
  ar-bench     --n N --out json                  torch.distributed NCCL all-reduce sweep on GPUs 0..N-1
  greenctx-one --gpu G --out json                create_greenctx_stream_by_value on device G (one process per GPU)
  summary      --rundir D                        SUMMARY.md + summary.json from status/*.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
# green pairs tried on every GPU (sum 108 = A100 SXM SM count; same shape as the engine grid)
GREEN_PAIRS = ((16, 92), (44, 64), (92, 16))
AR_BYTES = (1 << 10, 1 << 16, 1 << 20, 1 << 24, 1 << 27, 1 << 29)   # 1 KiB .. 512 MiB
LINK_CLASSES = ("X", "SYS", "NODE", "PHB", "PXB", "PIX")


def _dump(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")


# ---------------------------------------------------------------- pure parsers (unit-tested)
def link_class(tok: str) -> str:
    """'NV12' -> 'NVLINK(12)'; PIX/PXB/PHB/NODE/SYS/X kept; anything else 'OTHER'."""
    m = re.fullmatch(r"NV(\d+)", tok)
    if m:
        return f"NVLINK({int(m.group(1))})"
    return tok if tok in LINK_CLASSES else "OTHER"


def parse_topo(text: str) -> dict:
    """Parse `nvidia-smi topo -m`.  Returns {n_gpu, links[i][j], classes{class: count of ordered pairs}}."""
    lines = [l.rstrip() for l in text.splitlines() if l.strip()]
    hdr = None
    for i, l in enumerate(lines):
        if re.match(r"^\s*GPU0\b", l):
            hdr = i
            break
    if hdr is None:
        return {"n_gpu": 0, "links": [], "classes": {}, "error": "no GPU0 header"}
    cols = lines[hdr].split()
    gcols = [c for c in cols if re.fullmatch(r"GPU\d+", c)]
    n = len(gcols)
    links = []
    for l in lines[hdr + 1: hdr + 1 + n]:
        t = l.split()
        if not re.fullmatch(r"GPU\d+", t[0]):
            break
        links.append([link_class(x) for x in t[1: 1 + n]])
    classes: dict = {}
    for i, row in enumerate(links):
        for j, c in enumerate(row):
            if i != j:
                classes[c] = classes.get(c, 0) + 1
    return {"n_gpu": n, "links": links, "classes": classes,
            "all_pairs_nvlink": bool(links) and all(c.startswith("NVLINK") for i, r in enumerate(links)
                                                    for j, c in enumerate(r) if i != j)}


def parse_nccl_tests(text: str) -> dict:
    """nccl-tests table rows: size count type redop root time algbw busbw #wrong (out-of-place) then in-place."""
    rows = []
    for l in text.splitlines():
        t = l.split()
        if len(t) >= 13 and t[0].isdigit() and t[1].isdigit():
            try:
                rows.append({"size_B": int(t[0]), "oop_time_us": float(t[5]), "oop_algbw": float(t[6]),
                             "oop_busbw": float(t[7]), "oop_wrong": t[8],
                             "ip_time_us": float(t[9]), "ip_algbw": float(t[10]),
                             "ip_busbw": float(t[11]), "ip_wrong": t[12]})
            except ValueError:
                pass
    m = re.search(r"Avg bus bandwidth\s*:\s*([\d.]+)", text)
    return {"rows": rows, "avg_busbw": float(m.group(1)) if m else None,
            "wrong_total": sum(0 if r["oop_wrong"] in ("0", "N/A") and r["ip_wrong"] in ("0", "N/A") else 1
                               for r in rows)}


def busbw_allreduce(nbytes: int, sec: float, n: int) -> tuple:
    """(algbw, busbw) GB/s with the nccl-tests all-reduce convention busbw = algbw*2(n-1)/n."""
    alg = nbytes / sec / 1e9
    return alg, alg * 2 * (n - 1) / n


# ---------------------------------------------------------------- GPU-side commands
def cmd_gpu_facts(a):
    import torch
    rec = {"kind": "multigpu_gpu_facts", "torch": torch.__version__, "cuda": torch.version.cuda,
           "nccl_torch": ".".join(map(str, torch.cuda.nccl.version())) if torch.cuda.is_available() else None,
           "device_count": torch.cuda.device_count(), "gpus": []}
    try:
        import pynvml as nv
        nv.nvmlInit()
        rec["driver"] = nv.nvmlSystemGetDriverVersion()
    except Exception as e:  # noqa: BLE001
        nv = None
        rec["nvml_error"] = repr(e)
    for i in range(rec["device_count"]):
        p = torch.cuda.get_device_properties(i)
        g = {"index": i, "name": p.name, "cc": f"{p.major}.{p.minor}", "sm_count": p.multi_processor_count,
             "mem_total_MiB": p.total_memory >> 20}
        if nv:
            try:
                h = nv.nvmlDeviceGetHandleByIndex(i)
                g["uuid"] = nv.nvmlDeviceGetUUID(h)
                g["pci_bus"] = nv.nvmlDeviceGetPciInfo(h).busId
                g["clk_sm_MHz"] = nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM)
                g["clk_sm_max_MHz"] = nv.nvmlDeviceGetMaxClockInfo(h, nv.NVML_CLOCK_SM)
                g["power_W"] = nv.nvmlDeviceGetPowerUsage(h) / 1000
                g["power_limit_W"] = nv.nvmlDeviceGetPowerManagementLimit(h) / 1000
                g["temp_C"] = nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU)
                g["throttle_reasons"] = hex(nv.nvmlDeviceGetCurrentClocksThrottleReasons(h))
            except Exception as e:  # noqa: BLE001
                g["nvml_error"] = repr(e)
        rec["gpus"].append(g)
    _dump(rec, a.out)
    print(f"[gpu-facts] {rec['device_count']} GPUs -> {a.out}")
    return 0


def cmd_p2p(a):
    import torch
    n = torch.cuda.device_count()
    rec = {"kind": "multigpu_p2p", "n": n, "can_access": [[None] * n for _ in range(n)],
           "copy_GBps": [[None] * n for _ in range(n)], "mib": a.mib,
           "note": "can_device_access_peer != P2P in use by NCCL (NCCL_P2P_*/NVLS decide); copy GB/s is a descriptor"}
    nbytes = a.mib << 20
    bufs = []
    for i in range(n):
        with torch.cuda.device(i):
            bufs.append(torch.empty(nbytes, dtype=torch.uint8, device=f"cuda:{i}"))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            rec["can_access"][i][j] = bool(torch.cuda.can_device_access_peer(i, j))
            try:
                for _ in range(2):
                    bufs[j].copy_(bufs[i])
                torch.cuda.synchronize(i); torch.cuda.synchronize(j)
                t = time.perf_counter()
                for _ in range(5):
                    bufs[j].copy_(bufs[i])
                torch.cuda.synchronize(i); torch.cuda.synchronize(j)
                rec["copy_GBps"][i][j] = round(5 * nbytes / (time.perf_counter() - t) / 1e9, 2)
            except Exception as e:  # noqa: BLE001
                rec["copy_GBps"][i][j] = f"ERROR {e!r}"
    _dump(rec, a.out)
    print(f"[p2p] {n}x{n} -> {a.out}")
    return 0


def cmd_p2p_lat(a):
    """V0 (AMD track) descriptor: small-message device-to-device copy latency per ordered GPU pair.
    Median over --iters timed copies (CUDA events on the source device, eager, no graph capture).
    This is NOT the V1 crossing micro-benchmark (that needs a pre-registration: cudagraph ON, B grid, n>=4)."""
    import statistics
    import torch
    n = torch.cuda.device_count()
    sizes = [int(x) for x in a.sizes.split(",")]
    rec = {"kind": "multigpu_p2p_latency", "n": n, "sizes_B": sizes, "iters": a.iters, "pairs": [],
           "note": "eager copy_ latency descriptor, not V1 crossing L0; cudagraph capture not tested here"}
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    if a.pairs == "from0":
        pairs = [(0, j) for j in range(1, n)] + [(j, 0) for j in range(1, n)]
    for i, j in pairs:
        row = {"src": i, "dst": j, "median_us": {}, "p95_us": {}}
        for nb in sizes:
            try:
                src = torch.empty(nb, dtype=torch.uint8, device=f"cuda:{i}")
                dst = torch.empty(nb, dtype=torch.uint8, device=f"cuda:{j}")
                with torch.cuda.device(i):
                    for _ in range(10):
                        dst.copy_(src, non_blocking=True)
                    torch.cuda.synchronize(i); torch.cuda.synchronize(j)
                    ts = []
                    for _ in range(a.iters):
                        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
                        e0.record(); dst.copy_(src, non_blocking=True); e1.record()
                        e1.synchronize(); torch.cuda.synchronize(j)
                        ts.append(e0.elapsed_time(e1) * 1e3)
                ts.sort()
                row["median_us"][str(nb)] = round(statistics.median(ts), 2)
                row["p95_us"][str(nb)] = round(ts[int(0.95 * (len(ts) - 1))], 2)
            except Exception as e:  # noqa: BLE001
                row["median_us"][str(nb)] = f"ERROR {e!r}"
        rec["pairs"].append(row)
    _dump(rec, a.out)
    print(f"[p2p-lat] {len(pairs)} pairs x {len(sizes)} sizes -> {a.out}")
    return 0


def _ar_worker(rank, n, sizes, iters, warmup, port, q):
    import torch
    import torch.distributed as dist
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=n, device_id=torch.device(f"cuda:{rank}"))
    out = []
    for nb in sizes:
        x = torch.ones(nb // 4, dtype=torch.float32, device=f"cuda:{rank}")
        for _ in range(warmup):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            dist.all_reduce(x)
        e.record(); torch.cuda.synchronize()
        out.append(s.elapsed_time(e) / iters / 1e3)
        del x
    gathered = [None] * n
    dist.all_gather_object(gathered, out)
    if rank == 0:
        q.put(gathered)
    dist.destroy_process_group()


def cmd_ar_bench(a):
    import socket
    import torch
    import torch.multiprocessing as mp
    if torch.cuda.device_count() < a.n:
        _dump({"kind": "multigpu_ar_bench", "n": a.n, "error": f"only {torch.cuda.device_count()} GPUs"}, a.out)
        return 1
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]
    ctx = mp.get_context("spawn"); q = ctx.Queue()
    ps = [ctx.Process(target=_ar_worker, args=(r, a.n, AR_BYTES, a.iters, a.warmup, port, q)) for r in range(a.n)]
    for p in ps:
        p.start()
    try:
        gathered = q.get(timeout=a.timeout)
    except Exception as e:  # noqa: BLE001
        gathered = None; err = repr(e)
    for p in ps:
        p.join(30)
        if p.is_alive():
            p.kill()
    if gathered is None:
        _dump({"kind": "multigpu_ar_bench", "n": a.n, "error": err}, a.out)
        return 1
    rows = []
    for k, nb in enumerate(AR_BYTES):
        sec = max(g[k] for g in gathered)          # slowest rank
        alg, bus = busbw_allreduce(nb, sec, a.n)
        rows.append({"size_B": nb, "time_us": round(sec * 1e6, 2), "algbw_GBps": round(alg, 3),
                     "busbw_GBps": round(bus, 3)})
    _dump({"kind": "multigpu_ar_bench", "n": a.n, "gpus": list(range(a.n)), "iters": a.iters,
           "warmup": a.warmup, "rows": rows, "source": "torch.distributed NCCL (nccl-tests absent)",
           "note": "substrate descriptor; GPUs 0..n-1 only (other subsets are TODO)"}, a.out)
    print(f"[ar-bench] n={a.n} -> {a.out}")
    return 0


def cmd_greenctx_one(a):
    """One process per GPU (each creation leaks green contexts).  Observation only: created? driver read-out
    of SM count per stream (created != realized; the %smid census per GPU is the TODO of the pre-registered M-step)."""
    import torch
    rec = {"kind": "multigpu_greenctx_one", "gpu": a.gpu, "pairs": [], "complete": False}
    try:
        from sgl_kernel import spatial
        rec["sgl_kernel_spatial"] = True
    except Exception as e:  # noqa: BLE001
        rec["sgl_kernel_spatial"] = False; rec["import_error"] = repr(e)
        _dump(rec, a.out); return 1
    torch.cuda.set_device(a.gpu)
    try:
        rec["get_sm_available"] = spatial.get_sm_available(a.gpu)
    except Exception as e:  # noqa: BLE001
        rec["get_sm_available_error"] = repr(e)
    sph = gr = None
    try:
        sys.path.insert(0, HERE)
        import substrate_probe_helper as sph  # type: ignore
        gr = sph._green_readout_mod()
    except Exception as e:  # noqa: BLE001
        rec["readout_unavailable"] = repr(e)
    for p, d in GREEN_PAIRS:
        r = {"requested": [p, d], "created": False, "error": None}
        try:
            sp, sd = spatial.create_greenctx_stream_by_value(p, d, a.gpu)
            r["created"] = True
            if gr is not None:
                try:
                    r["driver"] = {"prefill": sph._stream_readout(gr, sp), "decode": sph._stream_readout(gr, sd)}
                except Exception as e:  # noqa: BLE001
                    r["readout_error"] = repr(e)
        except Exception as e:  # noqa: BLE001
            r["error"] = repr(e)
        rec["pairs"].append(r)
    rec["complete"] = True
    _dump(rec, a.out)
    print(f"[greenctx-one] gpu={a.gpu} created={[r['created'] for r in rec['pairs']]} -> {a.out}")
    return 0


# ---------------------------------------------------------------- summary
def cmd_summary(a):
    sd = os.path.join(a.rundir, "status")
    rows = []
    for f in sorted(os.listdir(sd)) if os.path.isdir(sd) else []:
        if f.endswith(".json"):
            try:
                rows.append(json.load(open(os.path.join(sd, f))))
            except Exception as e:  # noqa: BLE001
                rows.append({"stage": f, "status": "UNRESOLVED", "reason": f"unreadable {e!r}"})
    md = ["# multigpu_probe SUMMARY (facts only; no performance verdict, no substrate-equivalence verdict)", "",
          "| stage | status | reason |", "|---|---|---|"]
    md += [f"| {r['stage']} | {r['status']} | {str(r.get('reason', '')).replace('|', '/')} |" for r in rows]
    md += ["", "TODO (needs pre-registration): per-GPU 1-GPU serving-cell deviation (same cell on each GPU)."]
    with open(os.path.join(a.rundir, "SUMMARY.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    _dump({"stages": rows}, os.path.join(a.rundir, "summary.json"))
    return 0


def cmd_parse_topo(a):
    _dump(parse_topo(open(a.inp).read()), a.out); return 0


def cmd_parse_nccl(a):
    _dump(parse_nccl_tests(open(a.inp).read()), a.out); return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    def add(name, fn, **kw):
        sp = sub.add_parser(name); sp.set_defaults(fn=fn)
        for k, v in kw.items():
            sp.add_argument("--" + k.rstrip("_"), dest=k, **v)
        return sp
    add("parse-topo", cmd_parse_topo, inp=dict(required=True), out=dict(required=True))
    add("parse-nccl", cmd_parse_nccl, inp=dict(required=True), out=dict(required=True))
    add("gpu-facts", cmd_gpu_facts, out=dict(required=True))
    add("p2p", cmd_p2p, out=dict(required=True), mib=dict(type=int, default=256))
    add("p2p-lat", cmd_p2p_lat, out=dict(required=True), iters=dict(type=int, default=200),
        sizes=dict(default="2048,65536,524288,1048576"), pairs=dict(default="all", choices=["all", "from0"]))
    add("ar-bench", cmd_ar_bench, n=dict(type=int, required=True), out=dict(required=True),
        iters=dict(type=int, default=20), warmup=dict(type=int, default=5), timeout=dict(type=int, default=600))
    add("greenctx-one", cmd_greenctx_one, gpu=dict(type=int, required=True), out=dict(required=True))
    add("summary", cmd_summary, rundir=dict(required=True))
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
