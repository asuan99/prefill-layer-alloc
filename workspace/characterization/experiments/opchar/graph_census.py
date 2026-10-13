"""graph_census.py — launch configs of EVERY kernel an operator issues, without CUPTI.

Capture the operator into a CUDA graph (torch.cuda.CUDAGraph(keep_graph=True)), then walk the
graph's kernel nodes through the driver API:
  cuGraphGetNodes / cuGraphNodeGetType / cuGraphChildGraphNodeGetGraph   graph structure
  cuGraphKernelNodeGetParams(_v2)    gridDim, blockDim, dynamic sharedMemBytes (launch args)
  cuFuncGetName / cuKernelGetName    kernel symbol
  cuFuncGetAttribute                 NUM_REGS, SHARED_SIZE_BYTES (static), MAX_THREADS_PER_BLOCK
None of these touch performance counters, so they work where ncu (ERR_NVGPUCTRPERM) and CUPTI
(empty activity, probe v2) do not. Covers cuBLAS, FlashInfer, sgl_kernel, torch and Triton alike.

What it is NOT: a timing source (graph nodes carry no durations), and it describes the
GRAPH-captured launch. Eager launches can differ (e.g. cuBLAS workspace policy under capture);
decode in the engine runs captured, prefill does not — recorded as ``capture`` in every row.
Captured on the green stream, the launch is the one the kernel library chose for that partition.
"""
from __future__ import annotations

import ctypes
from typing import Dict, List, Optional

CU_GRAPH_NODE_TYPE = {0: "kernel", 1: "memcpy", 2: "memset", 3: "host", 4: "graph", 5: "empty",
                      6: "wait_event", 7: "event_record", 8: "ext_sem_signal", 9: "ext_sem_wait",
                      10: "mem_alloc", 11: "mem_free", 12: "batch_mem_op", 13: "conditional"}
ATTR = {"max_threads": 0, "static_smem": 1, "local_bytes": 3, "regs": 4}


class KernelNodeParamsV2(ctypes.Structure):
    """CUDA_KERNEL_NODE_PARAMS_v2 (cuda.h, CUDA >= 12.0)."""
    _fields_ = [("func", ctypes.c_void_p),
                ("gridDimX", ctypes.c_uint), ("gridDimY", ctypes.c_uint), ("gridDimZ", ctypes.c_uint),
                ("blockDimX", ctypes.c_uint), ("blockDimY", ctypes.c_uint), ("blockDimZ", ctypes.c_uint),
                ("sharedMemBytes", ctypes.c_uint),
                ("kernelParams", ctypes.c_void_p), ("extra", ctypes.c_void_p),
                ("kern", ctypes.c_void_p), ("ctx", ctypes.c_void_p)]


class _Driver:
    def __init__(self, lib=None):
        self.lib = lib or ctypes.CDLL("libcuda.so.1")
        self.get_params = self._sym("cuGraphKernelNodeGetParams_v2", "cuGraphKernelNodeGetParams")

    def _sym(self, *names):
        for n in names:
            f = getattr(self.lib, n, None)
            if f is not None:
                return f
        raise AttributeError(f"none of {names} in libcuda")

    def call(self, name, *args):
        rc = getattr(self.lib, name)(*args)
        if rc != 0:
            raise RuntimeError(f"{name} rc={rc}")

    def nodes(self, graph: int) -> List[int]:
        n = ctypes.c_size_t(0)
        self.call("cuGraphGetNodes", ctypes.c_void_p(graph), None, ctypes.byref(n))
        arr = (ctypes.c_void_p * n.value)()
        self.call("cuGraphGetNodes", ctypes.c_void_p(graph), arr, ctypes.byref(n))
        return [int(x or 0) for x in arr[:n.value]]

    def node_type(self, node: int) -> int:
        t = ctypes.c_int(-1)
        self.call("cuGraphNodeGetType", ctypes.c_void_p(node), ctypes.byref(t))
        return t.value

    def child_graph(self, node: int) -> int:
        g = ctypes.c_void_p()
        self.call("cuGraphChildGraphNodeGetGraph", ctypes.c_void_p(node), ctypes.byref(g))
        return int(g.value or 0)

    def kernel_params(self, node: int) -> KernelNodeParamsV2:
        p = KernelNodeParamsV2()
        rc = self.get_params(ctypes.c_void_p(node), ctypes.byref(p))
        if rc != 0:
            raise RuntimeError(f"cuGraphKernelNodeGetParams rc={rc}")
        return p

    def func_name(self, p: KernelNodeParamsV2) -> Optional[str]:
        s = ctypes.c_char_p()
        try:
            if p.func:
                self.call("cuFuncGetName", ctypes.byref(s), ctypes.c_void_p(p.func))
            elif p.kern:
                self.call("cuKernelGetName", ctypes.byref(s), ctypes.c_void_p(p.kern))
            else:
                return None
        except (RuntimeError, AttributeError):
            return None
        return s.value.decode(errors="replace") if s.value else None

    def occupancy(self, p: KernelNodeParamsV2, threads: int, dyn_smem: int) -> Optional[int]:
        """cuOccupancyMaxActiveBlocksPerMultiprocessor on the node's function (driver-computed b,
        honours the function's carveout/attributes; still a calculation, not a runtime observation)."""
        func = p.func
        try:
            if not func and p.kern:
                f = ctypes.c_void_p()
                self.call("cuKernelGetFunction", ctypes.byref(f), ctypes.c_void_p(p.kern))
                func = f.value
            if not func:
                return None
            n = ctypes.c_int(-1)
            self.call("cuOccupancyMaxActiveBlocksPerMultiprocessor", ctypes.byref(n), ctypes.c_void_p(func),
                      ctypes.c_int(threads), ctypes.c_size_t(dyn_smem))
            return n.value
        except (RuntimeError, AttributeError):
            return None

    def func_attr(self, p: KernelNodeParamsV2, key: str, device: int = 0) -> Optional[int]:
        v = ctypes.c_int(-1)
        try:
            if p.func:
                self.call("cuFuncGetAttribute", ctypes.byref(v), ctypes.c_int(ATTR[key]), ctypes.c_void_p(p.func))
            elif p.kern:
                self.call("cuKernelGetAttribute", ctypes.byref(v), ctypes.c_int(ATTR[key]),
                          ctypes.c_void_p(p.kern), ctypes.c_int(device))
            else:
                return None
        except (RuntimeError, AttributeError):
            return None
        return v.value


def census(graph: int, drv: Optional[_Driver] = None, _depth: int = 0) -> List[dict]:
    """Every node of ``graph`` (child graphs flattened), kernel nodes with launch config."""
    drv = drv or _Driver()
    out = []
    for i, node in enumerate(drv.nodes(graph)):
        t = drv.node_type(node)
        kind = CU_GRAPH_NODE_TYPE.get(t, f"type{t}")
        if kind == "graph" and _depth < 4:
            out += census(drv.child_graph(node), drv, _depth + 1)
            continue
        rec = {"node_kind": kind}
        if kind == "kernel":
            p = drv.kernel_params(node)
            rec.update({"kernel": drv.func_name(p),
                        "grid": [p.gridDimX, p.gridDimY, p.gridDimZ],
                        "block": [p.blockDimX, p.blockDimY, p.blockDimZ],
                        "dyn_smem": int(p.sharedMemBytes),
                        "static_smem": drv.func_attr(p, "static_smem"),
                        "regs": drv.func_attr(p, "regs"),
                        "max_threads": drv.func_attr(p, "max_threads"),
                        "occ_driver": (drv.occupancy(p, p.blockDimX * p.blockDimY * p.blockDimZ,
                                                     int(p.sharedMemBytes))
                                       if hasattr(drv, "occupancy") else None)})
        out.append(rec)
    return out


def capture_census(fn, stream) -> List[dict]:
    """Capture one call of ``fn`` on ``stream`` (keep_graph) and census it."""
    import torch
    with torch.cuda.stream(stream):
        fn()                                    # warm: lazy init / autotune outside capture
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(g, stream=stream):
        fn()
    torch.cuda.synchronize()
    return census(int(g.raw_cuda_graph()))


def census_rows(nodes: List[dict], sm_count: int) -> List[dict]:
    """Kernel nodes -> rows with residency + wave quantization on ``sm_count`` SMs (DERIVED)."""
    import math
    from experiments.common.wave_model import wave_quantization
    from experiments.opchar.trace_parse import resident_blocks_per_sm
    rows = []
    for idx, n in enumerate(nodes):
        if n["node_kind"] != "kernel":
            rows.append({"node": idx, "node_kind": n["node_kind"], "kernel": None, "grid": None,
                         "grid_blocks": None, "threads": None, "regs": None, "smem": None,
                         "resident_blocks_per_sm": None, "b_calc": None, "b_source": None,
                         "waves": None, "tail_sms": None, "idle_ratio": None})
            continue
        blocks = int(math.prod(n["grid"]))
        threads = int(math.prod(n["block"]))
        smem = (n["dyn_smem"] or 0) + (n["static_smem"] or 0)
        b_calc = (resident_blocks_per_sm(threads, n["regs"], smem)
                  if n["regs"] is not None and n["regs"] >= 0 and threads > 0 else None)
        occ = n.get("occ_driver")
        b, b_src = (occ, "driver") if isinstance(occ, int) and occ > 0 else (b_calc, "calc-upper-bound")
        wq = wave_quantization(blocks, sm_count, b) if b else {"waves": None, "tail_sms": None,
                                                               "idle_ratio": None}
        rows.append({"node": idx, "node_kind": "kernel", "kernel": (n["kernel"] or "?")[:240],
                     "grid": "x".join(map(str, n["grid"])), "grid_blocks": blocks, "threads": threads,
                     "regs": n["regs"], "smem": smem, "resident_blocks_per_sm": b, "b_calc": b_calc,
                     "b_source": b_src if b else None, **wq})
    return rows
