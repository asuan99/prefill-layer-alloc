#!/usr/bin/env python3
"""L0 standalone `%smid` census -- engine patch 0 lines, dev tree untouched.

Pre-registration: `PREREG_SMID_R0_2026-08-14.md` (this directory). Read that
document BEFORE reading this file; every threshold and every verdict string
below is fixed there. This script implements it; it does not decide anything
the pre-registration did not already decide.

WHAT THIS IS
    A device-side census: launch a resident-spin Triton kernel on a given CUDA
    stream, have every block report the hardware special register `%smid` (plus
    `%nsmid` and a `%globaltimer` residency interval), and take the UNION of the
    reported ids over a grid/repeat sweep. The decision quantities are SET
    relations between those unions -- not latencies, not throughputs.

WHAT THIS IS NOT
    * NOT a performance measurement. No TTFT / TPOT / ITL / goodput is produced
      or may be derived. There is no arm comparison in this file.
    * NOT an occupancy or utilisation measurement. `%smid` gives IDENTITY only:
      "a block of this launch ran on SM x". An SM in the set may be 5% busy.
    * NOT a measurement of the serving operating point. cudagraph replay is not
      exercised here at all (that is R4, L1, out of scope -- see prereg sec9).

MODES
    --selftest-cpu       GPU-free. Compiles the census kernel for sm_80, asserts
                         the PTX contains the probe registers and that the spin
                         loop has a back edge, and dumps `cuobjdump -res-usage`
                         for a baseline/instrumented toy pair. Writes a JSON.
    --selftest-analyzer  GPU-free. Runs the pre-registered decision function on
                         synthetic inputs, INCLUDING the empty-input case
                         (methodology gate #21: absence of measurement must
                         return UNDETERMINED, never a substantive verdict).
    --run                REQUIRES GPU. Emits raw census artefact. Makes no
                         verdict; scoring is a separate invocation on purpose.
    --analyze RAW.json   GPU-free. Applies the pre-registered rules to a raw
                         artefact and emits the verdict artefact.

GATE #21 (this project's signature error, 7 recurrences: a MEASUREMENT failure
labelled as a GATE failure). The scoring path in this file therefore has the
"did the instrument produce data at all" branch FIRST, and it returns
`UNDETERMINED (MEASUREMENT ABSENT)`. A missing/short/unsaturated census is never
reported as "no carve", "delivered", or any other substantive statement.

GATE #33 (a manifest N/N sha match does not imply the runtime is byte-identical:
`sync_engine_tree.sh:99-115` hashes exactly 15 files and neither
`sglang/srt/multiplex/pdmux_context.py` nor `sgl_kernel/spatial.py` is one of
them). Both of those files are on this script's critical path, so `--run`
records their sha256 directly into the artefact instead of trusting a manifest.

Author: engine-porter, 2026-08-14. GPU spend by this file at authoring time: 0
(only --selftest-cpu and --selftest-analyzer were executed).
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time

# --------------------------------------------------------------------------
# Pre-registered constants (prereg sec4/sec5). Changing any of these after the
# run is a protocol violation and must be reported as such.
# --------------------------------------------------------------------------
GRID_SWEEP = (108, 216, 432, 864, 1728)  # blocks per launch, ascending
REPEATS_PER_GRID = 5  # launches unioned at each grid point
SPIN_NS_DEFAULT = 1_000_000  # 1 ms residency
SPIN_NS_CAP = 2_000_000  # prereg sec6 F4 watchdog: never exceed 2 ms
SPIN_ITER_CAP = 1 << 24  # second watchdog, independent of the clock read
MIN_HITS_REPORTED = 1  # D2 reports min hit count; it is not a threshold
CUDA_HOME = os.environ.get("CUDA_HOME", "/apps/cuda/13.0.2")
CUOBJDUMP = os.path.join(CUDA_HOME, "bin", "cuobjdump")

# Verdict strings -- fixed vocabulary, prereg sec5. Do not invent new ones.
V_UNDET = "UNDETERMINED (MEASUREMENT ABSENT)"
V_UNSAT = "UNDETERMINED (COVERAGE NOT SATURATED)"
V_CONSISTENT = "GLOBALLY_CONSISTENT_LABEL"
V_NOT_CONSISTENT = "NOT_A_GLOBALLY_CONSISTENT_LABEL"
V_NO_DELIVERY = "LABEL_CONSISTENT_BUT_PARTITION_NOT_DELIVERED"
V_AMBIG = "UNDETERMINED (OUTCOME OUTSIDE PRE-REGISTERED MATRIX)"


# ==========================================================================
# 1. The census kernel
# ==========================================================================
def _kernels():
    """Import triton lazily so --analyze works on a node without triton.

    `%smid` and `%globaltimer` come from the UPSTREAM producer
    (`triton/language/extra/cuda/utils.py:5-13`, shipped with Triton 3.5.1),
    not from a hand-rolled copy. Methodology gate #14: a check that copies the
    code it means to verify is close to an identity; take the primitive from
    its producer. Only `%nsmid` has no upstream helper, so it is written out
    here -- and it is DESCRIPTIVE, never an input to a verdict.
    """
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import globaltimer as _rd_gtimer
    from triton.language.extra.cuda import smid as _rd_smid

    @triton.jit
    def _rd_nsmid():
        return tl.inline_asm_elementwise(
            "mov.u32 $0, %nsmid;", "=r", [], dtype=tl.int32,
            is_pure=True, pack=1)

    @triton.jit
    def census_kernel(out_smid, out_nsmid, out_t0, out_t1, spin_ns, iter_cap):
        # No rendezvous barrier anywhere in this kernel. A barrier across all
        # resident blocks deadlocks whenever the reachable SM count is smaller
        # than assumed, which is exactly the hypothesis under test (prereg
        # sec6 F4). Fixed-time residency + union over launches gives the same
        # coverage without that failure mode.
        pid = tl.program_id(0)
        s = _rd_smid()
        n = _rd_nsmid()
        t0 = _rd_gtimer()
        t = t0
        i = 0
        while (t - t0) < spin_ns and i < iter_cap:
            t = _rd_gtimer()
            i += 1
        tl.store(out_smid + pid, s)
        tl.store(out_nsmid + pid, n)
        tl.store(out_t0 + pid, t0)
        tl.store(out_t1 + pid, t)

    # Baseline / instrumented pair for the mechanism bound (prereg sec7).
    # SAME SIGNATURE in both arms: adding a probe-only pointer argument would
    # charge the pointer's registers to the probe and inflate the delta.
    @triton.jit
    def toy_base(out, n_pad):
        pid = tl.program_id(0)
        tl.store(out + pid, pid)

    @triton.jit
    def toy_instr(out, n_pad):
        pid = tl.program_id(0)
        tl.store(out + pid, pid)
        s = _rd_smid()
        tl.store(out + n_pad + pid, s)

    return census_kernel, toy_base, toy_instr


# ==========================================================================
# 2. CPU self-test (GPU-free) -- P0 / P0b of the design's probe ladder
# ==========================================================================
def _compile_aot(fn, signature):
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    src = ASTSource(fn=fn, signature=signature, constexprs={})
    return triton.compile(src, target=GPUTarget("cuda", 80, 32))


def _res_usage(cubin_bytes, path):
    with open(path, "wb") as f:
        f.write(cubin_bytes)
    if not os.path.exists(CUOBJDUMP):
        return {"error": f"cuobjdump not found at {CUOBJDUMP}"}
    p = subprocess.run([CUOBJDUMP, "-res-usage", path],
                       capture_output=True, text=True)
    out = {"stdout": p.stdout, "returncode": p.returncode}
    m = re.search(r"REG:(\d+)\s+STACK:(\d+)\s+SHARED:(\d+)\s+LOCAL:(\d+)", p.stdout)
    if m:
        out.update(reg=int(m.group(1)), stack=int(m.group(2)),
                   shared=int(m.group(3)), local=int(m.group(4)))
    return out


def _spin_loop_has_back_edge(ptx):
    """True iff some %globaltimer read sits inside a block with a back edge.

    A hoisted read would leave the loop body with no clock read, which turns
    the residency spin into an unbounded loop or a no-op. Checking this on the
    generated PTX is cheaper and stricter than trusting the optimiser.
    """
    labels = re.findall(r"^\$(L__BB\d+_\d+):", ptx, re.M)
    for lab in labels:
        # text between the label and the first branch back to that same label
        m = re.search(re.escape("$" + lab) + r":(.*?)@%p\d+\s+bra\s+\$" +
                      re.escape(lab) + r";", ptx, re.S)
        if m and "%globaltimer" in m.group(1):
            return True
    return False


def selftest_cpu(outdir, tag):
    census_kernel, toy_base, toy_instr = _kernels()
    rep = {"kind": "smid_l0_cpu_selftest", "tag": tag,
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "host": platform.node(), "gpu_used": False}
    import triton
    rep["triton_version"] = triton.__version__
    rep["python"] = sys.version.split()[0]
    rep["cuobjdump"] = CUOBJDUMP

    k = _compile_aot(census_kernel,
                     {"out_smid": "*i32", "out_nsmid": "*i32", "out_t0": "*i64",
                      "out_t1": "*i64", "spin_ns": "i64", "iter_cap": "i32"})
    ptx = k.asm["ptx"]
    rep["asm_keys"] = sorted(k.asm.keys())
    rep["ptx_smid_sites"] = ptx.count("%smid")
    rep["ptx_nsmid_sites"] = ptx.count("%nsmid")
    rep["ptx_globaltimer_sites"] = ptx.count("%globaltimer")
    rep["spin_loop_has_back_edge"] = _spin_loop_has_back_edge(ptx)
    rep["cubin_bytes"] = len(k.asm["cubin"])
    with open(os.path.join(outdir, f"smid_census_{tag}.ptx"), "w") as f:
        f.write(ptx)
    rep["census_res_usage"] = _res_usage(
        k.asm["cubin"], os.path.join(outdir, f"smid_census_{tag}.cubin"))

    kb = _compile_aot(toy_base, {"out": "*i32", "n_pad": "i32"})
    ki = _compile_aot(toy_instr, {"out": "*i32", "n_pad": "i32"})
    rb = _res_usage(kb.asm["cubin"], os.path.join(outdir, f"smid_toy_base_{tag}.cubin"))
    ri = _res_usage(ki.asm["cubin"], os.path.join(outdir, f"smid_toy_instr_{tag}.cubin"))
    rep["toy_baseline_res_usage"] = rb
    rep["toy_instrumented_res_usage"] = ri
    if "reg" in rb and "reg" in ri:
        rep["toy_reg_delta"] = ri["reg"] - rb["reg"]
        rep["toy_shared_local_stack_unchanged"] = (
            rb.get("shared") == ri.get("shared")
            and rb.get("local") == ri.get("local")
            and rb.get("stack") == ri.get("stack"))
    # scope note travels with the number so it cannot be quoted bare
    rep["toy_reg_delta_scope"] = (
        "TOY KERNEL ONLY. This delta bounds nothing about any production "
        "kernel; per prereg sec7 the L2 occupancy argument requires measuring "
        "the delta on the target kernel itself with its campaign constants. "
        "Methodology gate #32: do not import this number into another context.")

    checks = {
        "cubin_built_without_gpu": rep["cubin_bytes"] > 0,
        "smid_site_present": rep["ptx_smid_sites"] >= 1,
        "nsmid_site_present": rep["ptx_nsmid_sites"] >= 1,
        "globaltimer_two_sites": rep["ptx_globaltimer_sites"] >= 2,
        "spin_not_hoisted": rep["spin_loop_has_back_edge"],
        "toy_pair_measured": ("reg" in rb and "reg" in ri),
    }
    rep["checks"] = checks
    rep["result"] = "PASS" if all(checks.values()) else "FAIL"
    rep["result_scope"] = (
        "PLUMBING ONLY. A PASS licenses no physical, performance or delivery "
        "statement whatsoever (methodology gate #26: a smoke PASS carries no "
        "information about any result).")
    path = os.path.join(outdir, f"smid_l0_cpu_selftest_{tag}.json")
    with open(path, "w") as f:
        json.dump(rep, f, indent=2)
    print(json.dumps({k: v for k, v in rep.items()
                      if k not in ("census_res_usage",)}, indent=2))
    print(f"\n[selftest-cpu] {rep['result']}  -> {path}")
    return 0 if rep["result"] == "PASS" else 1


# ==========================================================================
# 3. Driver-layer read-out (P2). DESCRIPTIVE ONLY -- see prereg sec8.
#    These numbers are the driver's own TARGET book-keeping. They are the exact
#    mirror image of the Stage 0 D108 error if quoted as realised occupancy,
#    and they are structurally incapable of answering an identity question
#    (`CUdevSmResource` has smCount but no id set: cuda.h:24770-24777).
#    They are NEVER an input to any verdict in this file.
# ==========================================================================
CU_DEV_RESOURCE_TYPE_SM = 1


class _CUdevResource(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int),
                ("_pad", ctypes.c_ubyte * 92),
                ("_u", ctypes.c_ubyte * 48)]

    def sm_triplet(self):
        vals = (ctypes.c_uint * 3).from_buffer_copy(bytes(self._u)[:12])
        return dict(smCount=vals[0], minSmPartitionSize=vals[1],
                    smCoscheduledAlignment=vals[2])


def driver_readout(stream_ptrs):
    """Returns {'primary_sm': {...}, 'streams': {name: green_ctx_is_null}}."""
    out = {"note": "DESCRIPTIVE / TARGET LAYER ONLY -- never an input to a verdict"}
    try:
        lib = ctypes.CDLL("libcuda.so.1")
    except OSError as e:
        out["error"] = f"libcuda load failed: {e}"
        return out
    try:
        ctx = ctypes.c_void_p()
        rc = lib.cuCtxGetCurrent(ctypes.byref(ctx))
        out["cuCtxGetCurrent_rc"] = rc
        res = _CUdevResource()
        rc = lib.cuCtxGetDevResource(ctx, ctypes.byref(res),
                                     ctypes.c_int(CU_DEV_RESOURCE_TYPE_SM))
        out["cuCtxGetDevResource_rc"] = rc
        out["primary_sm"] = res.sm_triplet() if rc == 0 else None
    except Exception as e:  # noqa: BLE001 - diagnostic path
        out["primary_error"] = repr(e)
    streams = {}
    for name, ptr in stream_ptrs.items():
        try:
            g = ctypes.c_void_p()
            rc = lib.cuStreamGetGreenCtx(ctypes.c_void_p(ptr), ctypes.byref(g))
            streams[name] = {"rc": rc, "green_ctx_is_null": (g.value in (None, 0))}
        except Exception as e:  # noqa: BLE001
            streams[name] = {"error": repr(e)}
    out["streams"] = streams
    return out


# ==========================================================================
# 4. GPU run (WRITTEN, NOT EXECUTED at authoring time)
# ==========================================================================
def _sha256(path):
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError as e:
        return f"ERROR {e}"


def _census_once(kernel, stream, n_blocks, spin_ns):
    import torch
    dev = torch.cuda.current_device()
    smid = torch.full((n_blocks,), -1, dtype=torch.int32, device=f"cuda:{dev}")
    nsmid = torch.full((n_blocks,), -1, dtype=torch.int32, device=f"cuda:{dev}")
    t0 = torch.zeros((n_blocks,), dtype=torch.int64, device=f"cuda:{dev}")
    t1 = torch.zeros((n_blocks,), dtype=torch.int64, device=f"cuda:{dev}")
    with torch.cuda.stream(stream):
        kernel[(n_blocks,)](smid, nsmid, t0, t1, spin_ns, SPIN_ITER_CAP,
                            num_warps=1)
    stream.synchronize()
    return (smid.tolist(), nsmid.tolist(), t0.tolist(), t1.tolist())


def _census_target(kernel, stream, spin_ns):
    """Grid/repeat sweep on one stream. Returns the saturation ladder."""
    ladder = []
    union = set()
    hits = {}
    nsm = set()
    for n_blocks in GRID_SWEEP:
        for _ in range(REPEATS_PER_GRID):
            s, n, _a, _b = _census_once(kernel, stream, n_blocks, spin_ns)
            for v in s:
                if v >= 0:
                    union.add(v)
                    hits[v] = hits.get(v, 0) + 1
            nsm.update(x for x in n if x >= 0)
        ladder.append({"n_blocks": n_blocks, "union_size": len(union),
                       "union": sorted(union), "nsmid_observed": sorted(nsm)})
    return {"ladder": ladder, "union": sorted(union),
            "hits": {str(k): v for k, v in sorted(hits.items())},
            "min_hits": (min(hits.values()) if hits else 0)}


def run(outdir, tag, spin_ns):
    import torch
    if spin_ns > SPIN_NS_CAP:
        raise SystemExit(f"spin_ns {spin_ns} exceeds pre-registered cap {SPIN_NS_CAP}")
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device; --run requires a GPU (this is a "
                         "MEASUREMENT ABSENT condition, not a result)")

    # Producers imported, never re-implemented (methodology gate #9: put the
    # contrast on the producer). divide_sm is where 74/34 and 54/54 come from;
    # create_greenctx_stream_by_value is what the engine actually calls.
    from sglang.srt.multiplex import pdmux_context as pdc
    from sgl_kernel import spatial

    census_kernel, _, _ = _kernels()
    dev = torch.cuda.current_device()
    total_sm = spatial.get_sm_available(dev)
    cc = torch.cuda.get_device_capability(dev)
    groups = 4 - 2  # pdmux_a100_smoke.yml: sm_group_num = 4
    divisions = pdc.divide_sm(total_sm, cc, groups)

    rep = {"kind": "smid_l0_raw", "tag": tag,
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "host": platform.node(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
           "device_name": torch.cuda.get_device_name(dev),
           "compute_capability": list(cc), "total_sm_reported": total_sm,
           "spin_ns": spin_ns, "grid_sweep": list(GRID_SWEEP),
           "repeats_per_grid": REPEATS_PER_GRID,
           "divisions_from_divide_sm": [list(d) for d in divisions],
           # gate #33: these two files are OUTSIDE sync_engine_tree.sh's
           # 15-file manifest but are on this script's critical path.
           "sha256_unmanifested": {
               "pdmux_context.py": _sha256(pdc.__file__),
               "sgl_kernel/spatial.py": _sha256(spatial.__file__)},
           }

    # ---- t0: plain stream in the primary context, BEFORE any green context
    s_plain = torch.cuda.Stream(device=dev)
    rep["S_pre"] = _census_target(census_kernel, s_plain, spin_ns)
    # Runtime instrument check. The CPU self-test compiles ahead-of-time with
    # an explicit signature; the JIT at run time specialises differently (e.g.
    # int arguments), so the PTX that actually ran is NOT guaranteed to be the
    # PTX that was inspected on the login node. Re-check it here, on the object
    # that ran, and let the scorer refuse to interpret a census whose kernel
    # lost either the probe or the residency loop.
    try:
        cached = list(census_kernel.cache[dev].values())[0]
        rt_ptx = cached.asm["ptx"]
        rep["runtime_ptx_smid_sites"] = rt_ptx.count("%smid")
        rep["runtime_ptx_globaltimer_sites"] = rt_ptx.count("%globaltimer")
        rep["runtime_spin_back_edge"] = _spin_loop_has_back_edge(rt_ptx)
        rep["runtime_ptx_sha256"] = hashlib.sha256(rt_ptx.encode()).hexdigest()
        with open(os.path.join(outdir, f"smid_runtime_{tag}.ptx"), "w") as f:
            f.write(rt_ptx)
    except Exception as e:  # noqa: BLE001
        rep["runtime_ptx_smid_sites"] = None
        rep["runtime_spin_back_edge"] = None
        rep["runtime_ptx_error"] = repr(e)

    # ---- t1: build the stream groups exactly as initialize_stream_groups does
    # (pdmux_context.py:124-138): plain pair, then one green pair per division,
    # then plain pair.
    groups_built = [(torch.cuda.Stream(device=dev), torch.cuda.Stream(device=dev))]
    for p_sm, d_sm in divisions:
        groups_built.append(spatial.create_greenctx_stream_by_value(p_sm, d_sm, dev))
    groups_built.append((torch.cuda.Stream(device=dev), torch.cuda.Stream(device=dev)))
    rep["sm_counts_labels"] = ([[total_sm, 0]] + [list(d) for d in divisions]
                               + [[0, total_sm]])

    # ---- t2: same plain stream object again, AFTER green context creation
    rep["S_post"] = _census_target(census_kernel, s_plain, spin_ns)

    # ---- t3..t5: every stream in every group
    per_stream = {}
    for i, (sp, sd) in enumerate(groups_built):
        per_stream[f"idx{i}_prefill"] = _census_target(census_kernel, sp, spin_ns)
        per_stream[f"idx{i}_decode"] = _census_target(census_kernel, sd, spin_ns)
    rep["per_stream"] = per_stream

    # ---- t6: concurrent launch on the green pair of idx1 (R5, descriptive)
    conc = []
    sp, sd = groups_built[1]
    for n_blocks in (GRID_SWEEP[0], GRID_SWEEP[2]):
        a = {}
        # issue both, then synchronise both -- the host does not serialise them
        smid_p = torch.full((n_blocks,), -1, dtype=torch.int32, device=f"cuda:{dev}")
        nsm_p = torch.full((n_blocks,), -1, dtype=torch.int32, device=f"cuda:{dev}")
        t0p = torch.zeros((n_blocks,), dtype=torch.int64, device=f"cuda:{dev}")
        t1p = torch.zeros((n_blocks,), dtype=torch.int64, device=f"cuda:{dev}")
        smid_d = torch.full((n_blocks,), -1, dtype=torch.int32, device=f"cuda:{dev}")
        nsm_d = torch.full((n_blocks,), -1, dtype=torch.int32, device=f"cuda:{dev}")
        t0d = torch.zeros((n_blocks,), dtype=torch.int64, device=f"cuda:{dev}")
        t1d = torch.zeros((n_blocks,), dtype=torch.int64, device=f"cuda:{dev}")
        with torch.cuda.stream(sp):
            census_kernel[(n_blocks,)](smid_p, nsm_p, t0p, t1p, spin_ns,
                                       SPIN_ITER_CAP, num_warps=1)
        with torch.cuda.stream(sd):
            census_kernel[(n_blocks,)](smid_d, nsm_d, t0d, t1d, spin_ns,
                                       SPIN_ITER_CAP, num_warps=1)
        sp.synchronize()
        sd.synchronize()
        a["n_blocks"] = n_blocks
        a["prefill"] = {"smid": smid_p.tolist(), "t0": t0p.tolist(), "t1": t1p.tolist()}
        a["decode"] = {"smid": smid_d.tolist(), "t0": t0d.tolist(), "t1": t1d.tolist()}
        conc.append(a)
    rep["concurrent_idx1"] = conc

    # ---- t7: driver layer, descriptive only
    ptrs = {}
    for i, (a, b) in enumerate(groups_built):
        ptrs[f"idx{i}_prefill"] = a.cuda_stream
        ptrs[f"idx{i}_decode"] = b.cuda_stream
    ptrs["plain_pre"] = s_plain.cuda_stream
    rep["driver_readout"] = driver_readout(ptrs)

    path = os.path.join(outdir, f"smid_l0_raw_{tag}.json")
    with open(path, "w") as f:
        json.dump(rep, f, indent=2)
    print(f"[run] raw artefact -> {path}")
    print("[run] NO VERDICT IS PRODUCED HERE. Score with --analyze.")
    return 0


# ==========================================================================
# 5. Scoring -- the pre-registered decision rules (prereg sec5)
# ==========================================================================
def _saturated(target):
    """Set equality of the union at the last two grid points (prereg sec5.1).

    Cardinality equality is weaker and free to strengthen, so we require the
    SETS to be identical. `compute_coverage`-style span metrics are avoided on
    purpose (methodology gate #29: span metrics are blind to interior holes).
    """
    lad = target.get("ladder") or []
    if len(lad) < 2:
        return False, "fewer than two grid points"
    a, b = set(lad[-2]["union"]), set(lad[-1]["union"])
    return (a == b), ("identical at last two grid points" if a == b
                      else f"still growing: {len(a)} -> {len(b)}")


def score(raw):
    """Pure function: raw artefact dict -> verdict dict. No I/O, no globals."""
    out = {"kind": "smid_l0_verdict",
           "prereg": "PREREG_SMID_R0_2026-08-14.md",
           "R0": None, "R1": None, "R2": None, "R5": None, "D1": {}, "D2": {}}

    # ---------------- gate #21 branch, FIRST -----------------------------
    if not isinstance(raw, dict) or not raw:
        out["R0"] = {"verdict": V_UNDET, "why": "raw artefact empty or malformed"}
        out["stop"] = True
        return out
    need = ("S_pre", "S_post", "per_stream", "divisions_from_divide_sm")
    missing = [k for k in need if not raw.get(k)]
    if missing:
        out["R0"] = {"verdict": V_UNDET,
                     "why": f"required blocks absent: {missing}"}
        out["stop"] = True
        return out
    if raw.get("runtime_ptx_smid_sites") == 0:
        out["R0"] = {"verdict": V_UNDET,
                     "why": "runtime PTX contained no %smid site -- the kernel "
                            "that ran is not the probe"}
        out["stop"] = True
        return out
    if raw.get("runtime_spin_back_edge") is False:
        out["R0"] = {"verdict": V_UNDET,
                     "why": "runtime PTX has no clock read inside a loop with a "
                            "back edge -- residency was not achieved, so every "
                            "union is a lower bound of unknown slack"}
        out["stop"] = True
        return out

    ps = raw["per_stream"]
    idx_last = len(raw["divisions_from_divide_sm"]) + 1
    keys = ["S_pre", "S_post"] + list(ps.keys())

    def tgt(k):
        return raw[k] if k in ("S_pre", "S_post") else ps.get(k)

    for k in keys:
        t = tgt(k)
        if not t or not t.get("ladder"):
            out["D1"][k] = {"saturated": False, "why": "no ladder"}
            out["D2"][k] = {"min_hits": 0}
            continue
        sat, why = _saturated(t)
        out["D1"][k] = {"saturated": sat, "why": why,
                        "union_size": len(t.get("union") or [])}
        out["D2"][k] = {"min_hits": t.get("min_hits", 0)}

    green_p, green_d = "idx1_prefill", "idx1_decode"
    if green_p not in ps or green_d not in ps:
        out["R0"] = {"verdict": V_UNDET, "why": "green pair idx1 not censused"}
        out["stop"] = True
        return out

    # D1 is a hard precondition for every set statement (prereg sec5.1).
    for k in (green_p, green_d, "S_pre"):
        if not out["D1"].get(k, {}).get("saturated"):
            out["R0"] = {"verdict": V_UNSAT,
                         "why": f"coverage not saturated for {k}: "
                                f"{out['D1'].get(k, {}).get('why')}"}
            out["stop"] = True
            return out

    Sp = set(ps[green_p]["union"])
    Sd = set(ps[green_d]["union"])
    D = set(raw["S_pre"]["union"])  # |D| anchored on the PRE-green plain stream
    inter, union = Sp & Sd, Sp | Sd
    exp_p, exp_d = raw["divisions_from_divide_sm"][0]

    # ---------------- R0: outcome matrix (prereg sec5.2) ------------------
    # Four pre-registered outcomes. The design draft collapsed two of them
    # ("virtualised" and "not delivered") into one; they are different facts
    # and only one of them is an instrument failure.
    facts = {"disjoint": len(inter) == 0,
             "tiles_D": union == D,
             "sizes": [len(Sp), len(Sd)], "expected_sizes": [exp_p, exp_d],
             "abs_D": len(D), "intersection_size": len(inter),
             "union_size": len(union)}
    if facts["disjoint"] and facts["tiles_D"]:
        r0 = {"verdict": V_CONSISTENT,
              "why": "complementary green partitions are disjoint and their "
                     "union equals the pre-green device id set"}
    elif (not facts["disjoint"]) and len(Sp) >= 0.9 * len(D) and len(Sd) >= 0.9 * len(D):
        r0 = {"verdict": V_NO_DELIVERY,
              "why": "both green streams reach ~the whole device id set; the "
                     "labels are consistent, the partition is not delivered. "
                     "THIS IS A RESULT ABOUT THE ENGINE, NOT A PROBE FAILURE "
                     "(methodology gate #21)."}
    elif (not facts["disjoint"]) and len(union) < len(D):
        r0 = {"verdict": V_NOT_CONSISTENT,
              "why": "green streams overlap while their union is smaller than "
                     "the device id set -- consistent with per-context "
                     "renumbering; path 1 cannot be interpreted"}
    else:
        r0 = {"verdict": V_AMBIG,
              "why": "observed combination is outside the pre-registered "
                     "matrix; report the raw sets and stop"}
    r0["facts"] = facts
    out["R0"] = r0
    out["stop"] = r0["verdict"] != V_CONSISTENT
    if out["stop"]:
        out["R1"] = out["R2"] = out["R5"] = {
            "verdict": "UNINTERPRETABLE (R0 did not establish a globally "
                       "consistent label)"}
        return out

    # ---------------- R1: realised cardinality vs code-derived target -----
    out["R1"] = {
        "realised": [len(Sp), len(Sd)], "target_from_divide_sm": [exp_p, exp_d],
        "exact_match": [len(Sp), len(Sd)] == [exp_p, exp_d],
        "reading": ("realised == target" if [len(Sp), len(Sd)] == [exp_p, exp_d]
                    else "realised != target -- REPORT AS OBSERVED (prereg "
                         "sec5.3: a granularity/rounding difference is a "
                         "measurement RESULT, not a measurement failure)"),
        "forbidden": "do not convert set cardinality into compute share",
    }
    # every other green group, same rule, reported side by side
    others = {}
    for i in range(2, idx_last):
        kp, kd = f"idx{i}_prefill", f"idx{i}_decode"
        if kp in ps and kd in ps:
            a, b = set(ps[kp]["union"]), set(ps[kd]["union"])
            others[f"idx{i}"] = {"realised": [len(a), len(b)],
                                 "target": list(raw["divisions_from_divide_sm"][i - 1]),
                                 "disjoint": len(a & b) == 0,
                                 "saturated": [out["D1"].get(kp, {}).get("saturated"),
                                               out["D1"].get(kd, {}).get("saturated")]}
    out["R1"]["other_green_groups"] = others

    # ---------------- R2: primary-context carve --------------------------
    Spost = set(raw["S_post"]["union"])
    lost = sorted(D - Spost)
    out["R2"] = {
        "S_pre_size": len(D), "S_post_size": len(Spost),
        "lost_ids": lost, "lost_count": len(lost),
        "saturated_post": out["D1"].get("S_post", {}).get("saturated"),
        "reading": ("carve observed: ids reachable before green-context "
                    "creation and not after" if lost else
                    "no carve observed at this saturation level"),
        "not_independent": ("idx0/idx3 plain-stream censuses below are the SAME "
                            "empirical content as this difference, not a second "
                            "independent test (methodology gate #9, sixth "
                            "recurrence: two gates named independent measuring "
                            "one quantity). They are descriptive replicates."),
        "descriptive_replicates": {
            k: len(set(ps[k]["union"]))
            for k in (f"idx0_prefill", f"idx0_decode",
                      f"idx{idx_last}_prefill", f"idx{idx_last}_decode")
            if k in ps},
    }

    # ---------------- R5: temporal overlap (descriptive) -----------------
    r5 = []
    for blk in raw.get("concurrent_idx1") or []:
        p, d = blk["prefill"], blk["decode"]
        lo = max(min(p["t0"]), min(d["t0"]))
        hi = min(max(p["t1"]), max(d["t1"]))
        if hi <= lo:
            r5.append({"n_blocks": blk["n_blocks"], "overlap_ns": 0,
                       "note": "no temporal overlap window; nothing to say"})
            continue
        sp_w = {s for s, a, b in zip(p["smid"], p["t0"], p["t1"])
                if s >= 0 and b > lo and a < hi}
        sd_w = {s for s, a, b in zip(d["smid"], d["t0"], d["t1"])
                if s >= 0 and b > lo and a < hi}
        r5.append({"n_blocks": blk["n_blocks"], "overlap_ns": hi - lo,
                   "shared_sm_ids": sorted(sp_w & sd_w),
                   "prefill_sms_in_window": len(sp_w),
                   "decode_sms_in_window": len(sd_w)})
    out["R5"] = {"windows": r5,
                 "scope": ("MICRO-REGIME ONLY. Two spin kernels on an idle GPU. "
                           "Transfer to the serving operating point is "
                           "forbidden (prereg sec8-5).")}
    return out


def analyze(raw_path, outdir, tag):
    try:
        with open(raw_path) as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raw = {}
        print(f"[analyze] raw artefact unreadable ({e}) -- gate #21 branch")
    v = score(raw)
    v["raw_path"] = raw_path
    v["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    path = os.path.join(outdir, f"smid_l0_verdict_{tag}.json")
    with open(path, "w") as f:
        json.dump(v, f, indent=2)
    print(json.dumps(v, indent=2)[:4000])
    print(f"\n[analyze] -> {path}")
    return 0


# ==========================================================================
# 6. Analyzer self-test (GPU-free). Fixes the gate #21 branch in code.
# ==========================================================================
def _synth(sizes, disjoint=True, dsize=108, sat=True, post_lost=0, target=None):
    a = list(range(sizes[0]))
    b = (list(range(sizes[0], sizes[0] + sizes[1])) if disjoint
         else list(range(sizes[1])))

    def tar(u):
        lad = [{"n_blocks": n, "union": u, "nsmid_observed": [dsize]}
               for n in GRID_SWEEP]
        if not sat:
            lad[-1] = {"n_blocks": GRID_SWEEP[-1], "union": u + [10_000],
                       "nsmid_observed": [dsize]}
        return {"ladder": lad, "union": u, "min_hits": 3,
                "hits": {str(x): 3 for x in u}}
    D = list(range(dsize))
    return {
        "divisions_from_divide_sm": [list(target or sizes), [54, 54]],
        "runtime_ptx_smid_sites": 1, "runtime_spin_back_edge": True,
        "S_pre": tar(D), "S_post": tar(D[:dsize - post_lost]),
        "per_stream": {"idx1_prefill": tar(a), "idx1_decode": tar(b),
                       "idx2_prefill": tar(list(range(54))),
                       "idx2_decode": tar(list(range(54, 108))),
                       "idx0_prefill": tar(D), "idx0_decode": tar(D),
                       "idx3_prefill": tar(D), "idx3_decode": tar(D)},
        "concurrent_idx1": [],
    }


_TARGET_LAYER_TOKENS = ("driver_readout", "smCount", "minSmPartitionSize",
                        "smCoscheduledAlignment", "cuCtxGetDevResource",
                        "cuGreenCtx", "total_sm_reported", "nsmid_observed")


def _score_excludes_target_layer():
    """Gate #9 guard, enforced mechanically rather than promised in prose.

    Verifying a `%smid` census against the driver's own `smCount` self-report
    would be the mirror image of the Stage 0 D108 error: checking a target with
    another target. `CUdevSmResource` does not even carry an id set
    (cuda.h:24770-24777), so it cannot answer an identity question at all.
    This walks the AST of `score` and fails if any target-layer name leaked in.
    """
    import ast
    src = open(os.path.abspath(__file__)).read()
    fn = [n for n in ast.walk(ast.parse(src))
          if isinstance(n, ast.FunctionDef) and n.name == "score"][0]
    seg = ast.get_source_segment(src, fn) or ""
    return [t for t in _TARGET_LAYER_TOKENS if t in seg]


def selftest_analyzer():
    fails = []

    def ck(name, cond):
        print(f"  [{'ok' if cond else 'FAIL'}] {name}")
        if not cond:
            fails.append(name)

    print("analyzer self-test (no GPU):")
    leaked = _score_excludes_target_layer()
    ck(f"score() takes no driver self-report / target-layer input {leaked or ''}",
       not leaked)
    ck("empty raw -> MEASUREMENT ABSENT",
       score({})["R0"]["verdict"] == V_UNDET)
    ck("missing per_stream -> MEASUREMENT ABSENT",
       score({"S_pre": {"union": [1], "ladder": []}})["R0"]["verdict"] == V_UNDET)
    ck("runtime PTX without probe -> MEASUREMENT ABSENT",
       score(dict(_synth((74, 34)), runtime_ptx_smid_sites=0))["R0"]["verdict"] == V_UNDET)
    ck("runtime spin hoisted -> MEASUREMENT ABSENT",
       score(dict(_synth((74, 34)),
                  runtime_spin_back_edge=False))["R0"]["verdict"] == V_UNDET)
    ck("unsaturated -> COVERAGE NOT SATURATED (not a substantive verdict)",
       score(_synth((74, 34), sat=False))["R0"]["verdict"] == V_UNSAT)
    v = score(_synth((74, 34)))
    ck("disjoint + tiling -> GLOBALLY_CONSISTENT_LABEL",
       v["R0"]["verdict"] == V_CONSISTENT)
    ck("R1 exact match reported", v["R1"]["exact_match"] is True)
    ck("R2 no carve when S_post == S_pre", v["R2"]["lost_count"] == 0)
    # realised 76/32 while divide_sm targeted 74/34: granularity/rounding.
    # Pre-registered as a RESULT (report as observed), never a probe failure.
    v2 = score(_synth((76, 32), target=(74, 34)))
    ck("rounded partition still GLOBALLY_CONSISTENT_LABEL (result, not failure)",
       v2["R0"]["verdict"] == V_CONSISTENT and v2["R1"]["exact_match"] is False)
    ck("rounded partition reading says RESULT not failure",
       "not a measurement failure" in v2["R1"]["reading"])
    v3 = score(_synth((108, 108), disjoint=False))
    ck("both streams reach whole device -> PARTITION NOT DELIVERED (not a "
       "probe failure)", v3["R0"]["verdict"] == V_NO_DELIVERY)
    v4 = score(_synth((74, 34), disjoint=False))
    ck("overlap with small union -> NOT_A_GLOBALLY_CONSISTENT_LABEL",
       v4["R0"]["verdict"] == V_NOT_CONSISTENT)
    ck("R1/R2/R5 uninterpretable when R0 fails",
       "UNINTERPRETABLE" in v4["R1"]["verdict"])
    v5 = score(_synth((74, 34), post_lost=4))
    ck("carve detected as 4 lost ids", v5["R2"]["lost_count"] == 4)
    ck("R2 declares idx0/idx3 non-independent",
       "not_independent" in v5["R2"])
    print("PASS" if not fails else f"FAIL: {fails}")
    return 0 if not fails else 1


# ==========================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest-cpu", action="store_true")
    ap.add_argument("--selftest-analyzer", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--analyze", metavar="RAW.json")
    ap.add_argument("--outdir", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--tag", default=os.environ.get("SLURM_JOB_ID", "local"))
    ap.add_argument("--spin-ns", type=int, default=SPIN_NS_DEFAULT)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    if a.selftest_analyzer:
        return selftest_analyzer()
    if a.selftest_cpu:
        return selftest_cpu(a.outdir, a.tag)
    if a.run:
        return run(a.outdir, a.tag, a.spin_ns)
    if a.analyze:
        return analyze(a.analyze, a.outdir, a.tag)
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
