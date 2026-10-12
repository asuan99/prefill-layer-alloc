"""trace_parse.py — attribute GPU kernels in a torch.profiler (Kineto) Chrome trace to
opchar operators, and derive per-kernel wave quantization (CPU only, stdlib).

Why range attribution, not kernel-name regex
--------------------------------------------
The worker wraps every operator call in ``torch.profiler.record_function("opchar:<key>")``.
A GPU kernel is attributed to an operator by following its CUPTI ``correlation`` id back
to the host launch call (``cuda_runtime`` / ``cuda_driver`` event — Triton kernels launch
through the driver API) and taking the innermost ``opchar:`` range that encloses that
launch on the same thread. Kernel names are only used as a cross-check
(:func:`kernel_family`), so an unexpected kernel name can never move time between
operators.

Counter-free by construction: every field read here comes from the CUPTI Activity API
(kernel records), which does not need GPU performance-counter permission — the VESSL
containers block counters (``ERR_NVGPUCTRPERM``, RmProfilingAdminOnly=1) but Activity
tracing is expected to work; the opchar probe Job checks that before any campaign.

Kineto field meanings (PyTorch Chrome trace, kernel event ``args``):
  grid / block                 launch dims as recorded           -> MEASURED (launch config)
  registers per thread         compiled kernel attribute          -> MEASURED
  shared memory                static + dynamic bytes per block   -> MEASURED
  blocks per SM                grid blocks / device SM count      -> NOT residency; ignored
  est. achieved occupancy %    Kineto's own estimate              -> recorded, not used
Resident blocks per SM (Bullet's ``b``) is recomputed here with :func:`resident_blocks_per_sm`
from the launch config and the sm_80 limits — DERIVED, and an upper bound because it
assumes the maximum shared-memory carveout.
"""
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

RANGE_PREFIX = "opchar:"
_LAUNCH_CATS = ("cuda_runtime", "cuda_driver")
_GPU_CATS = ("kernel", "gpu_memset", "gpu_memcpy")

# A100 (sm_80) per-SM limits — CUDA Programming Guide, Compute Capability 8.0 table.
SM80 = {
    "max_threads": 2048, "max_warps": 64, "max_blocks": 32,
    "regs": 65536, "reg_alloc_unit": 256, "max_regs_per_thread": 255,
    "smem": 164 * 1024, "smem_reserved_per_block": 1024, "smem_alloc_unit": 128,
}


def resident_blocks_per_sm(threads_per_block: int, regs_per_thread: int,
                           smem_per_block: int, limits: Dict[str, int] = SM80) -> int:
    """Occupancy-calculator residency (DERIVED, upper bound: max smem carveout)."""
    if threads_per_block < 1:
        raise ValueError("threads_per_block must be >= 1")
    warps = math.ceil(threads_per_block / 32)
    by_threads = limits["max_threads"] // threads_per_block
    by_warps = limits["max_warps"] // warps
    lim = [by_threads, by_warps, limits["max_blocks"]]
    if regs_per_thread > 0:
        unit = limits["reg_alloc_unit"]
        regs_per_warp = math.ceil(regs_per_thread * 32 / unit) * unit
        lim.append((limits["regs"] // regs_per_warp) // warps)
    if smem_per_block > 0:
        unit = limits["smem_alloc_unit"]
        need = math.ceil(smem_per_block / unit) * unit + limits["smem_reserved_per_block"]
        lim.append(limits["smem"] // need)
    return max(0, min(lim))


_FAMILY = [
    ("ssm_scan", r"chunk_scan|chunk_state|state_passing|chunk_cumsum|bmm_chunk|ssd_|selective_scan|selective_state_update|_ssu"),
    ("conv1d", r"causal_conv1d|conv1d"),
    ("attn", r"flashinfer|batchprefill|batchdecode|prefillwithkvcache|decodewithkvcache|flash_fwd|fmha"),
    ("gemm", r"gemm|cutlass|cublas|s16816|xmma|sm80_.*tensorop"),
    ("norm", r"rmsnorm|rms_norm|layernorm|layer_norm"),
    ("kv_write", r"store_kv|store_cache|set_kv|index_put|indexput|scatter"),
    ("memory", r"memcpy|memset|copy_kernel|direct_copy"),
    ("elementwise", r"elementwise|vectorized|reduce_kernel|unrolled_elementwise|triton_"),
]
_FAMILY_RE = [(f, re.compile(p, re.I)) for f, p in _FAMILY]


def kernel_family(name: str) -> str:
    """Name-based family for CROSS-CHECK only (first match wins)."""
    for fam, pat in _FAMILY_RE:
        if pat.search(name):
            return fam
    return "other"


@dataclass
class KernelRec:
    name: str
    op: str                   # opchar key or "unattributed"
    dur_us: float
    cat: str
    grid: Optional[List[int]] = None
    block: Optional[List[int]] = None
    regs: Optional[int] = None
    smem: Optional[int] = None
    kineto_occ_pct: Optional[float] = None

    @property
    def grid_blocks(self) -> Optional[int]:
        return None if not self.grid else int(math.prod(self.grid))

    @property
    def threads(self) -> Optional[int]:
        return None if not self.block else int(math.prod(self.block))


@dataclass
class TraceSummary:
    kernels: List[KernelRec] = field(default_factory=list)
    ranges: Dict[str, int] = field(default_factory=dict)   # op key -> number of ranges seen

    def by_op(self) -> Dict[str, List[KernelRec]]:
        out: Dict[str, List[KernelRec]] = defaultdict(list)
        for k in self.kernels:
            out[k.op].append(k)
        return dict(out)


def _events(trace) -> List[dict]:
    if isinstance(trace, dict):
        return list(trace.get("traceEvents", []))
    return list(trace)


def parse_trace(trace) -> TraceSummary:
    """``trace`` = loaded Chrome-trace JSON (dict) or its traceEvents list."""
    evs = [e for e in _events(trace) if e.get("ph") == "X"]
    ranges = [e for e in evs if e.get("cat") == "user_annotation"
              and str(e.get("name", "")).startswith(RANGE_PREFIX)]
    by_tid: Dict[object, List[dict]] = defaultdict(list)
    for r in ranges:
        by_tid[(r.get("pid"), r.get("tid"))].append(r)

    def enclosing(e) -> Optional[str]:
        best = None
        for r in by_tid.get((e.get("pid"), e.get("tid")), ()):
            t0, t1 = float(r["ts"]), float(r["ts"]) + float(r.get("dur", 0))
            if t0 <= float(e["ts"]) <= t1:
                if best is None or float(r.get("dur", 0)) < float(best.get("dur", 0)):
                    best = r
        return None if best is None else best["name"][len(RANGE_PREFIX):]

    corr_to_op: Dict[int, str] = {}
    for e in evs:
        if e.get("cat") in _LAUNCH_CATS:
            corr = (e.get("args") or {}).get("correlation")
            op = enclosing(e)
            if corr is not None and op is not None:
                corr_to_op[int(corr)] = op

    summ = TraceSummary()
    for r in ranges:
        key = r["name"][len(RANGE_PREFIX):]
        summ.ranges[key] = summ.ranges.get(key, 0) + 1
    for e in evs:
        if e.get("cat") not in _GPU_CATS:
            continue
        a = e.get("args") or {}
        corr = a.get("correlation")
        op = corr_to_op.get(int(corr), "unattributed") if corr is not None else "unattributed"
        summ.kernels.append(KernelRec(
            name=str(e.get("name", "")), op=op, dur_us=float(e.get("dur", 0.0)), cat=e["cat"],
            grid=a.get("grid"), block=a.get("block"),
            regs=a.get("registers per thread"), smem=a.get("shared memory"),
            kineto_occ_pct=a.get("est. achieved occupancy %")))
    return summ


def load_trace(path) -> TraceSummary:
    with open(path) as f:
        return parse_trace(json.load(f))


def op_kernel_table(summ: TraceSummary, iterations: Dict[str, int],
                    sm_count: int) -> List[dict]:
    """One row per (op, kernel name): per-iteration GPU time and wave quantization.

    ``iterations[op]`` = how many times the op ran inside the profiled window, so times are
    per single op call. ``sm_count`` = SMs the launch could use (the green partition size).
    """
    from experiments.common.wave_model import wave_quantization  # single source
    agg: Dict[tuple, dict] = {}
    for k in summ.kernels:
        key = (k.op, k.name)
        row = agg.get(key)
        if row is None:
            row = agg[key] = {"op": k.op, "kernel": k.name, "family": kernel_family(k.name),
                              "cat": k.cat, "calls": 0, "dur_us_sum": 0.0,
                              "grid_blocks": k.grid_blocks, "threads": k.threads,
                              "regs": k.regs, "smem": k.smem, "launch_varies": False}
        row["calls"] += 1
        row["dur_us_sum"] += k.dur_us
        if k.grid_blocks != row["grid_blocks"]:
            row["launch_varies"] = True
    out = []
    for row in agg.values():
        n = iterations.get(row["op"], 0)
        row["calls_per_iter"] = row["calls"] / n if n else None
        row["dur_us_per_iter"] = row["dur_us_sum"] / n if n else None
        b = None
        if row["cat"] == "kernel" and row["threads"] and row["regs"] is not None and row["smem"] is not None:
            b = resident_blocks_per_sm(row["threads"], int(row["regs"]), int(row["smem"]))
        row["resident_blocks_per_sm"] = b
        if b and row["grid_blocks"] and not row["launch_varies"]:
            wq = wave_quantization(row["grid_blocks"], sm_count, b)
            row.update({"waves": wq["waves"], "tail_sms": wq["tail_sms"], "idle_ratio": wq["idle_ratio"]})
        else:
            row.update({"waves": None, "tail_sms": None, "idle_ratio": None})
        out.append(row)
    return sorted(out, key=lambda r: (r["op"], -r["dur_us_sum"]))


def time_weighted_idle(rows: Iterable[dict]) -> Dict[str, Optional[float]]:
    """Per-op idle ratio weighted by each kernel's GPU time (Bullet Table 2 'normalized to
    execution time'). Kernels without a wave value are excluded and reported as coverage."""
    num: Dict[str, float] = defaultdict(float)
    den: Dict[str, float] = defaultdict(float)
    tot: Dict[str, float] = defaultdict(float)
    for r in rows:
        tot[r["op"]] += r["dur_us_sum"]
        if r.get("idle_ratio") is not None:
            num[r["op"]] += r["idle_ratio"] * r["dur_us_sum"]
            den[r["op"]] += r["dur_us_sum"]
    out = {}
    for op in tot:
        out[op] = num[op] / den[op] if den[op] > 0 else None
        out[op + "#coverage"] = den[op] / tot[op] if tot[op] > 0 else None
    return out
