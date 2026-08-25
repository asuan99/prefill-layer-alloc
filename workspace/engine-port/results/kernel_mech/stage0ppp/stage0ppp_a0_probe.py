#!/usr/bin/env python3
"""Stage 0''' A0 -- THE PROBE.  Implements PREREG_STAGE0PPP_A0_2026-08-24.md (rev4).

NO SERVER.  NO MODEL.  NO REQUEST.  This launches spin kernels on a green-context
stream and on the default stream, under nsys, and writes a raw JSON.  It produces
no latency, no throughput, no goodput -- by construction nothing in its output can
be quoted as a performance number.

Five legs, run in the registered order (prereg sec2):
  L1  full-GPU eager   -- BEFORE any green context exists      (positive control)
  L2  full-GPU graph   -- node rows exist + DEFINES the expectation
  L3  green-ctx eager  -- are green-context kernels visible at all
  L4  green-ctx graph  -- THE CONDITION UNDER TEST
  L2' full-GPU graph   -- AFTER L4: did the trace survive to the end (X5/N1)

Leg attribution carries NO NVTX: each leg uses a DISTINCTLY NAMED kernel, so a
row's kernel name says which leg emitted it.  L2 and L2' share a name and are
split by their position relative to the L4 rows.

The green partition is judged by THIS PROCESS's own driver readout, never by
nsys -- using nsys would make the green axis circular with Q2a (prereg sec5, N6).
"""
import argparse, hashlib, json, os, platform, sys, time

REPLAYS = 20            # prereg sec8 #11
WARMUPS = 3             # prereg sec8 #14
GRID = 34               # prereg sec8 #13 -- one block per green-half SM label
BLOCK_WARPS = 4         # 128 threads
SPIN_NS = 120_000       # >= 100 us per kernel (prereg sec8 #13)


def _sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _kernels():
    """One jit function per leg.  Identical bodies; the NAME is the leg tag.

    The spin primitive is taken from the upstream producer, as
    smid_l0_census.py does (methodology gate #14): a probe that re-implements
    the thing it measures is close to an identity.
    """
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import globaltimer as _gt

    def _mk(name):
        src = (
            "def {n}(out, spin_ns, iter_cap):\n"
            "    pid = tl.program_id(0)\n"
            "    t0 = _gt()\n"
            "    t = t0\n"
            "    i = 0\n"
            "    while (t - t0) < spin_ns and i < iter_cap:\n"
            "        t = _gt()\n"
            "        i += 1\n"
            "    tl.store(out + pid, t - t0)\n"
        ).format(n=name)
        ns = {"tl": tl, "_gt": _gt}
        exec(src, ns)
        return triton.jit(ns[name])

    return {k: _mk(k) for k in
            ("a0_l1_eager_full", "a0_l2_graph_full",
             "a0_l3_eager_green", "a0_l4_graph_green")}


def _census():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "..", "smid_census"))
    import smid_l0_census as cen
    return cen


def _green_sm_readout(stream_ptr):
    """THE registered decision channel for the `green` axis (prereg sec5, N6).

    Two reasons this is not smid_l0_census.driver_readout:
      (1) that function returns `primary_sm` -- the CURRENT context's SM
          resource (cuCtxGetCurrent + cuCtxGetDevResource), i.e. 108 on this
          device.  Comparing it to the green half's target would report
          `mismatched` unconditionally.
      (2) its own docstring registers it as "DESCRIPTIVE / TARGET LAYER ONLY --
          never an input to a verdict".  Using it as one would break the
          contract of the artefact I am importing.

    So the green context handle is taken from the stream and asked for ITS OWN
    SM resource.  nsys is deliberately not consulted: using nsys' isGreenContext
    would make this axis circular with Q2a (prereg sec5).
    """
    import ctypes
    cen = _census()
    out = {"channel": "cuStreamGetGreenCtx + cuGreenCtxGetDevResource"}
    try:
        lib = ctypes.CDLL("libcuda.so.1")
    except OSError as e:
        out["error"] = f"libcuda load failed: {e}"
        return out
    try:
        g = ctypes.c_void_p()
        out["cuStreamGetGreenCtx_rc"] = lib.cuStreamGetGreenCtx(
            ctypes.c_void_p(stream_ptr), ctypes.byref(g))
        out["green_ctx_is_null"] = g.value in (None, 0)
        if out["green_ctx_is_null"] or out["cuStreamGetGreenCtx_rc"] != 0:
            return out
        res = cen._CUdevResource()
        rc = lib.cuGreenCtxGetDevResource(g, ctypes.byref(res),
                                          ctypes.c_int(cen.CU_DEV_RESOURCE_TYPE_SM))
        out["cuGreenCtxGetDevResource_rc"] = rc
        out["green_sm"] = res.sm_triplet() if rc == 0 else None
    except Exception as e:
        out["error"] = repr(e)
    return out


def _driver_readout(stream_ptrs):
    """Descriptive cross-check only -- NOT the decision channel (see above)."""
    return _census().driver_readout(stream_ptrs)


def _launch(torch, kern, out, stream):
    with torch.cuda.stream(stream):
        kern[(GRID,)](out, SPIN_NS, 1 << 22, num_warps=BLOCK_WARPS)


def _graph_leg(torch, kern, out, stream, rec, key):
    """warmup on a side stream, capture, then REPLAYS replays."""
    side = torch.cuda.Stream(device=out.device)
    side.wait_stream(torch.cuda.current_stream())
    try:
        with torch.cuda.stream(side):
            for _ in range(WARMUPS):
                kern[(GRID,)](out, SPIN_NS, 1 << 22, num_warps=BLOCK_WARPS)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=stream):
            kern[(GRID,)](out, SPIN_NS, 1 << 22, num_warps=BLOCK_WARPS)
        rec[key + "_capture"] = "ok"
    except Exception as e:                      # a capture failure is a
        rec[key + "_capture"] = "fail"          # MEASUREMENT condition, never
        rec[key + "_capture_error"] = repr(e)   # an answer about the driver
        return
    try:
        t0 = time.monotonic_ns()
        with torch.cuda.stream(stream):
            for _ in range(REPLAYS):
                g.replay()
        torch.cuda.synchronize()
        rec[key + "_replays"] = REPLAYS
        rec[key + "_wall_ns"] = time.monotonic_ns() - t0
    except Exception as e:
        rec[key + "_replay_error"] = repr(e)


def run(args):
    import torch
    from sglang.srt.multiplex import pdmux_context as pdc
    from sgl_kernel import spatial

    dev = torch.cuda.current_device()
    total_sm = spatial.get_sm_available(dev)
    cc = torch.cuda.get_device_capability(dev)
    divisions = pdc.divide_sm(total_sm, cc, 4 - 2)      # prereg sec2: (8,0),2
    K = _kernels()
    out = torch.zeros(GRID, dtype=torch.int64, device=f"cuda:{dev}")

    rec = {"kind": "stage0ppp_a0_raw", "granularity": args.granularity,
           "tag": args.tag, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "host": platform.node(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
           "device_name": torch.cuda.get_device_name(dev),
           "total_sm_reported": total_sm, "compute_capability": list(cc),
           "divisions_from_divide_sm": [list(d) for d in divisions],
           "replays": REPLAYS, "warmups": WARMUPS, "grid": GRID,
           "spin_ns": SPIN_NS, "torch": torch.__version__,
           "sha256_unmanifested": {"pdmux_context.py": _sha256(pdc.__file__),
                                   "spatial.py": _sha256(spatial.__file__)},
           "rule_sha256": _sha256(os.path.join(os.path.dirname(
               os.path.abspath(__file__)), "stage0ppp_a0_rule.py")),
           "probe_sha256": _sha256(os.path.abspath(__file__)),
           "legs": {}}

    plain = torch.cuda.current_stream(device=dev)
    # ---- L1: full-GPU eager, BEFORE any green context exists ---------------
    t = time.monotonic_ns()
    for _ in range(REPLAYS):
        _launch(torch, K["a0_l1_eager_full"], out, plain)
    torch.cuda.synchronize()
    rec["legs"]["L1"] = {"kernel": "a0_l1_eager_full", "launches": REPLAYS,
                         "wall_ns": time.monotonic_ns() - t}

    # ---- L2: full-GPU graph (defines the expectation) ----------------------
    l2 = {"kernel": "a0_l2_graph_full"}
    _graph_leg(torch, K["a0_l2_graph_full"], out, plain, l2, "L2")
    rec["legs"]["L2"] = l2

    # ---- green context ------------------------------------------------------
    p_sm, d_sm = divisions[0]
    rec["green_target"] = {"prefill_sm": p_sm, "decode_sm": d_sm}
    try:
        g_prefill, g_decode = spatial.create_greenctx_stream_by_value(p_sm, d_sm, dev)
        rec["green_create"] = "ok"
    except Exception as e:
        rec["green_create"] = "fail"
        rec["green_create_error"] = repr(e)
        g_decode = None
    if g_decode is not None:
        rec["green_readout"] = _green_sm_readout(g_decode.cuda_stream)
        try:      # descriptive only
            rec["driver_readout"] = _driver_readout(
                {"green_decode": g_decode.cuda_stream, "plain": plain.cuda_stream})
        except Exception as e:
            rec["driver_readout"] = {"error": repr(e)}

        # ---- L3: green-ctx eager -------------------------------------------
        t = time.monotonic_ns()
        for _ in range(REPLAYS):
            _launch(torch, K["a0_l3_eager_green"], out, g_decode)
        torch.cuda.synchronize()
        rec["legs"]["L3"] = {"kernel": "a0_l3_eager_green", "launches": REPLAYS,
                             "wall_ns": time.monotonic_ns() - t}

        # ---- L4: THE CONDITION UNDER TEST -----------------------------------
        l4 = {"kernel": "a0_l4_graph_green"}
        _graph_leg(torch, K["a0_l4_graph_green"], out, g_decode, l4, "L4")
        rec["legs"]["L4"] = l4

    # ---- L2': the trailing control (X5 / N1) --------------------------------
    l2p = {"kernel": "a0_l2_graph_full", "position": "after_L4"}
    _graph_leg(torch, K["a0_l2_graph_full"], out, plain, l2p, "L2p")
    rec["legs"]["L2p"] = l2p

    torch.cuda.synchronize()
    with open(args.out, "w") as f:
        json.dump(rec, f, indent=1, sort_keys=True)
    print(f"[a0-probe] wrote {args.out}  granularity={args.granularity}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--granularity", choices=["node", "graph"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="local")
    sys.exit(run(ap.parse_args()))
