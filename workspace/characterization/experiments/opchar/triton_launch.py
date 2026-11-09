"""triton_launch.py — record REAL launch configs of Triton kernels without CUPTI.

Why: on the VESSL image CUPTI delivers no activity records (probe v2, job-ztoebt1cst7q:
Kineto "GPU trace is empty!", no CUPTI error) and ncu is blocked (ERR_NVGPUCTRPERM), so
grid / registers / shared memory per kernel are not observable through any profiler.
Triton kernels — the hybrid-specific ones here: SSD chunk scan/state/bmm/cumsum, causal
conv1d (Triton path), selective_state_update — are launched from Python, and Triton 3.5.1
calls ``CompiledKernel.launch_metadata(self, grid, stream, *args)`` on every launch with the
resolved grid (triton/runtime/jit.py:756, compiler/compiler.py:487). Wrapping that method
gives, per launch: kernel name, grid (MEASURED: the launch argument), num_warps, dynamic
shared bytes and n_regs (MEASURED: compiled-kernel attributes from cuModuleGetFunction).
The wrapper calls the original and changes nothing about the launch.

Per-kernel time: a CUDA event is recorded on the stream just before each recorded launch and
one after the operator; consecutive differences = "segment" time from a Triton launch to the
next one (DERIVED — it includes any non-Triton kernels issued in between). Used only to
weight idle ratios inside an operator, never as a kernel duration claim.

Non-Triton kernels (cuBLAS GEMM, FlashInfer, sgl_kernel norm, torch elementwise) are NOT
covered here; the analysis treats GEMMs with an assumed tile (wave_model.gemm_tile_blocks).
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional

TRITON_COMPILER_MODULE = "triton.compiler.compiler"


class TritonLaunchRecorder:
    def __init__(self):
        self.records: List[dict] = []
        self.active_op: Optional[str] = None
        self.stream = None
        self._cls = None
        self._orig = None
        self._events: List[tuple] = []          # (record index or None for op end, op, event)

    # -- install / uninstall -------------------------------------------------------------
    def install(self):
        import importlib
        mod = importlib.import_module(TRITON_COMPILER_MODULE)
        cls = mod.CompiledKernel
        if getattr(cls.launch_metadata, "_opchar_wrapped", False):
            raise RuntimeError("CompiledKernel.launch_metadata already wrapped")
        orig = cls.launch_metadata
        rec = self

        def launch_metadata(kself, grid, stream, *args):
            if rec.active_op is not None:
                rec._record(kself, grid)
            return orig(kself, grid, stream, *args)
        launch_metadata._opchar_wrapped = True
        cls.launch_metadata = launch_metadata
        self._cls, self._orig = cls, orig
        return self

    def uninstall(self):
        if self._cls is not None:
            self._cls.launch_metadata = self._orig
            self._cls = self._orig = None

    # -- recording ------------------------------------------------------------------------
    def _record(self, kself, grid):
        try:
            kself._init_handles()
        except Exception:   # noqa: BLE001 -- the original launch will raise the real error
            pass
        g = [int(x) for x in list(grid)[:3]] + [1] * (3 - min(3, len(grid)))
        md = kself.metadata
        idx = len(self.records)
        self.records.append({
            "op": self.active_op, "kernel": str(kself.name), "grid": g,
            "num_warps": int(md.num_warps), "num_ctas": int(getattr(md, "num_ctas", 1) or 1),
            "shared": int(md.shared), "n_regs": _int_or_none(getattr(kself, "n_regs", None)),
            "n_spills": _int_or_none(getattr(kself, "n_spills", None))})
        self._event(idx)

    def _event(self, idx):
        if self.stream is None:
            return
        import torch
        ev = torch.cuda.Event(enable_timing=True)
        ev.record(self.stream)
        self._events.append((idx, self.active_op, ev))

    def begin_op(self, op: str):
        self.active_op = op

    def end_op(self):
        self._event(None)
        self.active_op = None

    def segment_ms(self) -> Dict[int, float]:
        """record index -> ms until the next event of the same op (DERIVED). Call after sync."""
        out = {}
        for (i, op, ev), (_, op2, ev2) in zip(self._events, self._events[1:]):
            if i is not None and op == op2:
                out[i] = ev.elapsed_time(ev2)
        return out


def _int_or_none(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def launch_table(records: List[dict], seg_ms: Dict[int, float], iterations: Dict[str, int],
                 sm_count: int) -> List[dict]:
    """One row per (op, kernel, grid): residency + wave quantization on ``sm_count`` SMs."""
    from experiments.common.wave_model import wave_quantization
    from experiments.opchar.trace_parse import resident_blocks_per_sm
    agg: Dict[tuple, dict] = {}
    for i, r in enumerate(records):
        blocks = int(math.prod(r["grid"]))
        key = (r["op"], r["kernel"], blocks)
        row = agg.get(key)
        if row is None:
            threads = r["num_warps"] * 32
            b = (resident_blocks_per_sm(threads, r["n_regs"], r["shared"])
                 if r["n_regs"] is not None else None)
            row = agg[key] = {"op": r["op"], "kernel": r["kernel"], "grid_blocks": blocks,
                              "grid": "x".join(map(str, r["grid"])), "threads": threads,
                              "regs": r["n_regs"], "smem": r["shared"], "num_ctas": r["num_ctas"],
                              "resident_blocks_per_sm": b, "calls": 0, "segment_ms_sum": 0.0,
                              "segment_missing": 0}
            if b:
                row.update(wave_quantization(blocks, sm_count, b))
            else:
                row.update({"waves": None, "tail_sms": None, "idle_ratio": None})
        row["calls"] += 1
        if i in seg_ms:
            row["segment_ms_sum"] += seg_ms[i]
        else:
            row["segment_missing"] += 1
    out = []
    for row in agg.values():
        n = iterations.get(row["op"], 0)
        row["calls_per_iter"] = row["calls"] / n if n else None
        row["segment_ms_per_iter"] = row["segment_ms_sum"] / n if n else None
        out.append(row)
    return sorted(out, key=lambda r: (r["op"], -r["segment_ms_sum"], r["kernel"]))


def op_idle(rows: List[dict]) -> Dict[str, Optional[float]]:
    """Segment-time-weighted idle ratio per op over its recorded Triton launches (DERIVED)."""
    num: Dict[str, float] = defaultdict(float)
    den: Dict[str, float] = defaultdict(float)
    for r in rows:
        if r.get("idle_ratio") is not None and r["segment_ms_sum"] > 0:
            num[r["op"]] += r["idle_ratio"] * r["segment_ms_sum"]
            den[r["op"]] += r["segment_ms_sum"]
    return {op: (num[op] / den[op] if den[op] > 0 else None) for op in {r["op"] for r in rows}}
