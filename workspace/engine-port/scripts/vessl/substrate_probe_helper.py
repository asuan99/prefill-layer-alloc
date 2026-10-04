#!/usr/bin/env python3
"""Python legs + pure judgments for scripts/vessl/substrate_probe.sh (B0/B3/B4).

WHAT THIS IS.  A substrate check for the VESSL A100 Workspace (pdmux-probe):
handoff-report/gpu_rental_checklist_2026-09-18.md sec 3 (B0-B5) and
handoff-report/vessl_operating_model_2026-10-02.md sec 7 step 5.  It answers
"can this machine give the partitions we use, and does the engine boot on it?"
with PASS / FAIL / UNRESOLVED **mechanical facts only**.

★ NO PERFORMANCE NUMBER IS PRODUCED OR JUDGED HERE.  No latency, throughput,
  goodput or speed-up is recorded by any subcommand.  B4 records whether requests
  came back, never how fast.
★ NOT A SUBSTRATE-EQUIVALENCE VERDICT.  Matching facts with KISTI artefacts are
  listed side by side; whether the VESSL A100 is "the same substrate" is a user
  decision (CLAUDE.md gate 1 extension), not an output of this file.
★ `%smid` LABEL CEILING (R0 prereg sec3.4, job 889631): a %smid label set is a
  globally consistent LABEL set, not established to be a physical SM index.
  "realized SM count" below means |label set| on a saturated census, anchored only
  by the plain-stream census |D| equalling the device SM count (checked, B3-plain).

REUSE, NOT RE-IMPLEMENTATION (methodology gate #9 / #14):
  * census kernel, sweep, saturation rule, instrument check, driver read-out:
      workspace/engine-port/results/smid_census/smid_l0_census.py
      (`_kernels`, `_census_target`, `_saturated`, `_record_runtime_ptx`,
       `driver_readout`) -- imported, never edited (it is pre-registered).
  * per-stream green-context SM read-out: src/multiplex/green_readout.py
      (`green_sm_readout`) -- the same channel the engine's PDMUX_GREEN_READOUT uses.
  * green pair creation: `sgl_kernel.spatial.create_greenctx_stream_by_value`,
      the call `pdmux_context.initialize_stream_groups` makes.
  * the temporal-overlap window: same formula as smid_l0_census.score() R5 block.
Top-level imports are stdlib only, so every judgment runs on a CPU node
(tests/test_substrate_probe.py).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.abspath(os.path.join(_HERE, "..", ".."))          # workspace/engine-port
PROJECT = os.path.abspath(os.path.join(ENGINE, "..", ".."))         # prefill-layer-alloc
SMID_DIR = os.path.join(ENGINE, "results", "smid_census")
BCG_DIR = os.path.join(ENGINE, "results", "bcg_probe")
GREEN_READOUT_SRC = os.path.join(ENGINE, "src", "multiplex", "green_readout.py")

PASS, FAIL, UNRES, SKIP = "PASS", "FAIL", "UNRESOLVED", "SKIPPED"

# ---------------------------------------------------------------------------
# Registered inputs (sources cited; the shell script passes the same files)
# ---------------------------------------------------------------------------
# Grid = union of the manual_divisions of these configs, read at run time so the
# probe tests what the campaigns load rather than a hand copy:
#   benchmarks/configs/pdmux_r2.yml:4-8          (92,16)(84,24)(74,34)(64,44)
#   results/longctx_conflict/probes/pdmux_homog5.yml:9-13  (92,16)(64,44)(16,92) -- B4 boot cfg
GRID_CONFIGS = (
    os.path.join(ENGINE, "benchmarks", "configs", "pdmux_r2.yml"),
    os.path.join(ENGINE, "results", "longctx_conflict", "probes", "pdmux_homog5.yml"),
)

# Granularity candidates (DESCRIPTIVE, no pass/fail on the outcome itself).
# create_greenctx_stream_by_value(smA, smB) first splits (smA+smB) off the device,
# then splits smA off that group; B gets the remainder of the FIRST split
# (sgl-kernel/csrc/spatial/greenctx_stream.cu create_greenctx_stream_by_value).
#   "A-side" (smA odd/small, smA+smB = 108): probes rounding of the second split.
#   "total-side" (smA = 16, smA+smB odd < 108): probes rounding of the first split.
GRAN_CANDIDATES = (
    (1, 107), (2, 106), (3, 105), (4, 104), (5, 103), (6, 102),
    (15, 93), (17, 91), (18, 90), (20, 88), (22, 86),
    (16, 17), (16, 19), (16, 21), (2, 2),
)

# Expected machine facts from VESSL_A100_CONTEXT.md sec 2 (descriptive comparison
# only; differences are recorded, never failed -- the platform patches drivers).
VESSL_CONTEXT_FACTS = {"driver_version": "580.105.08", "clocks_max_sm": "1410 MHz",
                       "power_limit": "400.00 W"}
# Gating machine facts: what the grid and every A100 artefact presuppose.
EXPECT_CC = "8.0"
EXPECT_SM = 108

# B4 prompts: literally results/r2_eval/e2_sticky_prereg/e2_sticky.sbatch:288-297
# (itself literally results/sticky_smoke/sticky_smoke.sbatch:118-133).
# tests/test_substrate_probe.py asserts each literal still appears in that file.
P13_PROMPTS = (
    "The capital of France is",
    "Q: What is 17 + 25?\nA:",
    "Once upon a time in a small village by the sea,",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "List the first five prime numbers:",
    "Translate to French: The weather is nice today.",
)
P13_SAMPLING = {"temperature": 0, "max_new_tokens": 48}   # e2_sticky.sbatch:299


def _utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256(path):
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError as exc:
        return f"ERROR {exc}"


def _dump(obj, path):
    tmp = path + ".part"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=False)
    os.replace(tmp, path)


def _load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _cen():
    if SMID_DIR not in sys.path:
        sys.path.insert(0, SMID_DIR)
    import smid_l0_census as CEN  # noqa: E402  (stdlib-only at import time)
    return CEN


def _green_readout_mod():
    """Import green_readout from the INSTALLED tree when present (that is what the
    engine runs), else from the tracked source (CPU tests)."""
    try:
        from sglang.srt.multiplex import green_readout as gr  # type: ignore
        return gr
    except Exception:  # noqa: BLE001
        import importlib.util
        spec = importlib.util.spec_from_file_location("green_readout", GREEN_READOUT_SRC)
        gr = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gr)
        return gr


# ===========================================================================
# Config parsing (pure)
# ===========================================================================
def grid_from_configs(paths):
    """Union of manual_divisions (prefill, decode) over YAML configs, in first-seen
    order.  Parsed with a YAML parser, never grepped (lesson #258)."""
    import yaml
    pairs, sources = [], {}
    for p in paths:
        with open(p) as f:
            raw = yaml.safe_load(f) or {}
        for row in raw.get("manual_divisions") or []:
            pair = (int(row[0]), int(row[1]))
            sources.setdefault(f"{pair[0]}/{pair[1]}", []).append(os.path.relpath(p, PROJECT))
            if pair not in pairs:
                pairs.append(pair)
    return pairs, sources


# ===========================================================================
# B0 -- python-side machine facts
# ===========================================================================
def cmd_b0_python(a):
    rec = {"kind": "substrate_probe_b0_python", "utc": _utc(), "host": platform.node(),
           "python": sys.version.split()[0]}
    import importlib.metadata as md
    for dist in ("torch", "triton", "sglang-kernel", "sglang", "flashinfer-python"):
        try:
            rec[f"dist_{dist}"] = md.version(dist)
        except Exception as exc:  # noqa: BLE001
            rec[f"dist_{dist}"] = f"ERROR {exc!r}"
    try:
        import torch
        rec["torch_version"] = torch.__version__
        rec["torch_cuda"] = torch.version.cuda
        rec["cuda_available"] = bool(torch.cuda.is_available())
        if rec["cuda_available"]:
            pr = torch.cuda.get_device_properties(0)
            rec["device_name"] = pr.name
            rec["compute_capability"] = f"{pr.major}.{pr.minor}"
            rec["multi_processor_count"] = int(pr.multi_processor_count)
            rec["total_memory_bytes"] = int(pr.total_memory)
    except Exception as exc:  # noqa: BLE001
        rec["torch_error"] = repr(exc)
    try:
        from sgl_kernel import spatial
        rec["sgl_kernel_spatial_file"] = spatial.__file__
        rec["sgl_kernel_spatial_sha256"] = _sha256(spatial.__file__)
        rec["get_sm_available_0"] = int(spatial.get_sm_available(0))
    except Exception as exc:  # noqa: BLE001
        rec["sgl_kernel_error"] = repr(exc)
    try:
        from sglang.srt.multiplex import pdmux_context as pdc
        rec["pdmux_context_sha256"] = _sha256(pdc.__file__)
    except Exception as exc:  # noqa: BLE001
        rec["pdmux_context_error"] = repr(exc)
    _dump(rec, a.out)
    print(json.dumps({k: v for k, v in rec.items() if not k.endswith("sha256")}, indent=1))
    return 0


def parse_gpu_csv(text):
    """`nvidia-smi --query-gpu=name,uuid,driver_version,compute_cap,memory.total,
    clocks.max.sm,power.limit,mig.mode.current,compute_mode --format=csv,noheader`."""
    keys = ["name", "uuid", "driver_version", "compute_cap", "memory_total",
            "clocks_max_sm", "power_limit", "mig_mode", "compute_mode"]
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return None
    vals = [c.strip() for c in lines[0].split(",")]
    rec = dict(zip(keys, vals))
    rec["gpu_count"] = len(lines)
    return rec


def judge_b0(gpu, py):
    """Gating: one A100 cc 8.0, MIG off, 108 SM by both torch and sgl_kernel,
    CUDA available, sgl_kernel.spatial importable."""
    facts, why = {}, []
    if not gpu:
        return {"status": FAIL, "reason": "nvidia-smi query produced no GPU row", "facts": {}}
    if not py:
        return {"status": FAIL, "reason": "python facts missing", "facts": {"gpu": gpu}}
    facts["gpu"] = gpu
    checks = {
        "single_gpu": gpu.get("gpu_count") == 1,
        "name_is_A100": "A100" in (gpu.get("name") or ""),
        "compute_cap_8_0": gpu.get("compute_cap") == EXPECT_CC,
        "mig_disabled": (gpu.get("mig_mode") or "").strip() in ("Disabled", "[N/A]", "N/A"),
        "cuda_available": py.get("cuda_available") is True,
        "torch_cc_8_0": py.get("compute_capability") == EXPECT_CC,
        "torch_sm_108": py.get("multi_processor_count") == EXPECT_SM,
        "sgl_kernel_spatial_import": "sgl_kernel_error" not in py,
        "get_sm_available_108": py.get("get_sm_available_0") == EXPECT_SM,
    }
    facts["checks"] = checks
    facts["versions"] = {k: py.get(k) for k in ("torch_version", "torch_cuda",
                                                "dist_triton", "dist_sglang-kernel",
                                                "dist_flashinfer-python")}
    facts["get_sm_available_0"] = py.get("get_sm_available_0")
    facts["vs_VESSL_A100_CONTEXT"] = {
        k: {"expected": v, "observed": gpu.get(k), "same": gpu.get(k) == v}
        for k, v in VESSL_CONTEXT_FACTS.items()}
    bad = [k for k, ok in checks.items() if not ok]
    if bad:
        return {"status": FAIL, "reason": "failed: " + ",".join(bad), "facts": facts}
    return {"status": PASS, "reason": "A100 cc8.0, 108 SM (torch and sgl_kernel), MIG off",
            "facts": facts}


def cmd_score_b0(a):
    gpu = parse_gpu_csv(open(a.gpu_csv).read() if os.path.exists(a.gpu_csv) else "")
    v = judge_b0(gpu, _load(a.py_json))
    v["stage"] = "B0"
    _dump(v, a.status)
    print(v["status"], v["reason"])
    return 0


# ===========================================================================
# B3 -- GPU legs
# ===========================================================================
def _concurrent(CEN, kernel, sp, sd, n_blocks, spin_ns, dev):
    """Launch the census kernel on BOTH streams before synchronising either --
    the exact issue pattern of smid_l0_census.run() t6 (concurrent_idx1)."""
    import torch
    mk = lambda: (torch.full((n_blocks,), -1, dtype=torch.int32, device=f"cuda:{dev}"),  # noqa: E731
                  torch.full((n_blocks,), -1, dtype=torch.int32, device=f"cuda:{dev}"),
                  torch.zeros((n_blocks,), dtype=torch.int64, device=f"cuda:{dev}"),
                  torch.zeros((n_blocks,), dtype=torch.int64, device=f"cuda:{dev}"))
    bp, bd = mk(), mk()
    with torch.cuda.stream(sp):
        kernel[(n_blocks,)](*bp, spin_ns, CEN.SPIN_ITER_CAP, num_warps=1)
    with torch.cuda.stream(sd):
        kernel[(n_blocks,)](*bd, spin_ns, CEN.SPIN_ITER_CAP, num_warps=1)
    sp.synchronize()
    sd.synchronize()
    return {"n_blocks": n_blocks,
            "prefill": {"smid": bp[0].tolist(), "t0": bp[2].tolist(), "t1": bp[3].tolist()},
            "decode": {"smid": bd[0].tolist(), "t0": bd[2].tolist(), "t1": bd[3].tolist()}}


CONC_GRIDS_IDX = (0, 2, 4)      # GRID_SWEEP[0], [2], [4] = 108, 432, 1728 blocks
CONC_REPEATS = 3


def _base_meta(CEN, torch, spatial, dev):
    pr = torch.cuda.get_device_properties(dev)
    return {"utc": _utc(), "host": platform.node(), "device_name": pr.name,
            "compute_capability": [pr.major, pr.minor],
            "multi_processor_count": int(pr.multi_processor_count),
            "total_sm_reported": int(spatial.get_sm_available(dev)),
            "spin_ns": CEN.SPIN_NS_DEFAULT, "grid_sweep": list(CEN.GRID_SWEEP),
            "repeats_per_grid": CEN.REPEATS_PER_GRID,
            "torch_version": torch.__version__,
            "sha256": {"smid_l0_census.py": _sha256(os.path.join(SMID_DIR, "smid_l0_census.py")),
                       "substrate_probe_helper.py": _sha256(os.path.abspath(__file__)),
                       "sgl_kernel/spatial.py": _sha256(spatial.__file__)}}


def _stream_readout(gr, stream):
    try:
        return gr.green_sm_readout(int(stream.cuda_stream))
    except Exception as exc:  # noqa: BLE001
        return {"error": repr(exc)}


def cmd_b3_grid(a):
    """Plain census (D) -> create every grid pair in one process (as
    initialize_stream_groups does) -> plain census again -> isolated census of every
    green stream -> concurrent census of each pair -> instrument check."""
    import torch
    from sgl_kernel import spatial
    CEN, gr = _cen(), _green_readout_mod()
    os.makedirs(a.outdir, exist_ok=True)
    pairs, sources = grid_from_configs(a.configs.split(","))
    dev = torch.cuda.current_device()
    kernel, _, _ = CEN._kernels()
    spin = CEN.SPIN_NS_DEFAULT
    raw = {"kind": "substrate_probe_b3_grid_raw", **_base_meta(CEN, torch, spatial, dev),
           "pairs_requested": [list(p) for p in pairs], "pair_sources": sources,
           "configs": [os.path.relpath(c, PROJECT) for c in a.configs.split(",")],
           "complete": False}
    plain = torch.cuda.Stream(device=dev)
    raw["plain_pre"] = CEN._census_target(kernel, plain, spin)
    _dump(raw, a.out)
    built = []
    for p, d in pairs:
        rec = {"requested": [p, d], "created": False, "create_error": None}
        try:
            sp, sd = spatial.create_greenctx_stream_by_value(p, d, dev)
            rec["created"] = True
            rec["driver"] = {"prefill": _stream_readout(gr, sp), "decode": _stream_readout(gr, sd)}
            built.append((rec, sp, sd))
        except Exception as exc:  # noqa: BLE001
            rec["create_error"] = repr(exc)
        raw.setdefault("pairs", []).append(rec)
    plain_post = torch.cuda.Stream(device=dev)
    raw["plain_driver"] = {"plain_pre": _stream_readout(gr, plain),
                           "plain_post": _stream_readout(gr, plain_post)}
    raw["plain_post"] = CEN._census_target(kernel, plain, spin)     # same object, after
    _dump(raw, a.out)
    for rec, sp, sd in built:
        rec["prefill"] = CEN._census_target(kernel, sp, spin)
        rec["decode"] = CEN._census_target(kernel, sd, spin)
        rec["concurrent"] = [
            _concurrent(CEN, kernel, sp, sd, CEN.GRID_SWEEP[i], spin, dev)
            for i in CONC_GRIDS_IDX for _ in range(CONC_REPEATS)]
        _dump(raw, a.out)
    CEN._record_runtime_ptx(raw, kernel, dev, a.outdir, "b3grid")
    raw["complete"] = True
    _dump(raw, a.out)
    print(f"[b3-grid] raw -> {a.out}  (no verdict here; score-b3)")
    return 0


def cmd_b3_gran_one(a):
    """One granularity candidate per PROCESS: each creation leaks two green contexts
    (the wrapper never destroys them) and a failed split must not contaminate the
    next candidate."""
    import torch
    from sgl_kernel import spatial
    CEN, gr = _cen(), _green_readout_mod()
    dev = torch.cuda.current_device()
    kernel, _, _ = CEN._kernels()
    spin = CEN.SPIN_NS_DEFAULT
    rec = {"kind": "substrate_probe_b3_gran_raw", **_base_meta(CEN, torch, spatial, dev),
           "requested": [a.p, a.d], "created": False, "create_error": None, "complete": False}
    try:
        sp, sd = spatial.create_greenctx_stream_by_value(a.p, a.d, dev)
        rec["created"] = True
        rec["driver"] = {"prefill": _stream_readout(gr, sp), "decode": _stream_readout(gr, sd)}
    except Exception as exc:  # noqa: BLE001
        rec["create_error"] = repr(exc)
    _dump(rec, a.out)
    if rec["created"]:
        rec["prefill"] = CEN._census_target(kernel, sp, spin)
        rec["decode"] = CEN._census_target(kernel, sd, spin)
        CEN._record_runtime_ptx(rec, kernel, dev, os.path.dirname(a.out),
                                f"gran_{a.p}_{a.d}")
    rec["complete"] = True
    _dump(rec, a.out)
    print(f"[b3-gran-one] {a.p},{a.d} created={rec['created']} -> {a.out}")
    return 0


# ===========================================================================
# B3 -- pure judgments
# ===========================================================================
def _saturated(c):
    """smid_l0_census._saturated (imported, not copied): identical union SETS at the
    last two grid points."""
    if not c:
        return False
    return bool(_cen()._saturated(c)[0])


def _set(c):
    return set(c.get("union") or []) if c else set()


def instrument_ok(raw):
    """Same refusal as smid_l0_census.score(): a census whose runtime PTX lost the
    %smid read or the residency loop is not a measurement."""
    if raw.get("runtime_ptx_error"):
        return False, f"runtime_ptx_error: {raw['runtime_ptx_error']}"
    if not raw.get("runtime_ptx_smid_sites"):
        return False, "runtime PTX has no %smid site"
    if raw.get("runtime_spin_back_edge") is not True:
        return False, "runtime PTX lost the spin loop back edge"
    return True, "runtime PTX carries %smid and the spin loop"


def overlap_window(blk):
    """smid_l0_census.score() R5 formula: the interval where both kernels ran."""
    p, d = blk["prefill"], blk["decode"]
    pt0 = [a for s, a in zip(p["smid"], p["t0"]) if s >= 0]
    pt1 = [b for s, b in zip(p["smid"], p["t1"]) if s >= 0]
    dt0 = [a for s, a in zip(d["smid"], d["t0"]) if s >= 0]
    dt1 = [b for s, b in zip(d["smid"], d["t1"]) if s >= 0]
    if not (pt0 and dt0):
        return 0, set(), set()
    lo, hi = max(min(pt0), min(dt0)), min(max(pt1), max(dt1))
    if hi <= lo:
        return 0, set(), set()
    sp = {s for s, x, y in zip(p["smid"], p["t0"], p["t1"]) if s >= 0 and y > lo and x < hi}
    sd = {s for s, x, y in zip(d["smid"], d["t0"], d["t1"]) if s >= 0 and y > lo and x < hi}
    return hi - lo, sp, sd


def _driver_sm(dr, role):
    try:
        return dr[role]["green_sm"]["smCount"]
    except (KeyError, TypeError):
        return None


def _attached(dr, role):
    try:
        return dr[role]["green_ctx_is_null"] is False
    except (KeyError, TypeError):
        return None


def judge_plain(raw):
    """(108,0)/(0,108) boundary: the engine uses PLAIN streams for these two groups
    (pdmux_context.initialize_stream_groups), so the realized set is the plain census
    D.  Also R2-style: creating the green contexts must not carve the plain stream."""
    pre, post = raw.get("plain_pre"), raw.get("plain_post")
    facts = {"D_size": len(_set(pre)), "D_post_size": len(_set(post)),
             "total_sm_reported": raw.get("total_sm_reported"),
             "multi_processor_count": raw.get("multi_processor_count"),
             "saturated_pre": _saturated(pre), "saturated_post": _saturated(post),
             "plain_driver_detached": {k: (v.get("green_ctx_is_null") is True)
                                       for k, v in (raw.get("plain_driver") or {}).items()}}
    if not (pre and post):
        return {"status": UNRES, "reason": "plain census missing", "facts": facts}
    if not (facts["saturated_pre"] and facts["saturated_post"]):
        return {"status": UNRES, "reason": "plain census not saturated", "facts": facts}
    if facts["D_size"] != raw.get("total_sm_reported"):
        return {"status": FAIL, "reason": f"|D|={facts['D_size']} != get_sm_available="
                f"{raw.get('total_sm_reported')}", "facts": facts}
    if _set(pre) != _set(post):
        return {"status": FAIL, "reason": "plain stream label set changed after green "
                "context creation", "facts": facts}
    if not all(facts["plain_driver_detached"].values() or [False]):
        return {"status": UNRES, "reason": "driver did not report plain streams as "
                "detached (negative control of the read-out)", "facts": facts}
    return {"status": PASS, "reason": f"|D|={facts['D_size']} == get_sm_available, unchanged "
            "after green creation", "facts": facts}


def judge_pair(rec, D):
    """One grid pair.  Returns (b3a, b3b) judgments.

    b3a  realized == requested: created, both streams saturated, |S_p| == p and
         |S_d| == d, S_p and S_d disjoint and inside D (and tile D when p+d == |D|).
    b3b  concurrent: in every concurrent launch the two kernels overlapped in time,
         shared no label, and each side stayed inside its own isolated set.
    """
    p, d = rec["requested"]
    base = {"requested": [p, d], "created": rec.get("created"),
            "create_error": rec.get("create_error")}
    if not rec.get("created"):
        r = {"status": FAIL, "reason": f"green context creation failed: {rec.get('create_error')}",
             "facts": base}
        return r, dict(r)
    Sp, Sd = _set(rec.get("prefill")), _set(rec.get("decode"))
    dr = rec.get("driver") or {}
    facts = dict(base, realized=[len(Sp), len(Sd)],
                 saturated=[_saturated(rec.get("prefill")), _saturated(rec.get("decode"))],
                 min_hits=[(rec.get("prefill") or {}).get("min_hits"),
                           (rec.get("decode") or {}).get("min_hits")],
                 driver_smCount=[_driver_sm(dr, "prefill"), _driver_sm(dr, "decode")],
                 driver_attached=[_attached(dr, "prefill"), _attached(dr, "decode")],
                 intersection=sorted(Sp & Sd), outside_D=sorted((Sp | Sd) - D),
                 tiles_D=(Sp | Sd) == D and not (Sp & Sd),
                 prefill_labels=sorted(Sp), decode_labels=sorted(Sd))
    if not all(facts["saturated"]):
        a = {"status": UNRES, "reason": "census not saturated", "facts": facts}
    elif facts["driver_attached"] != [True, True]:
        a = {"status": UNRES, "reason": "driver does not report both streams attached to a "
             "green context", "facts": facts}
    elif facts["intersection"] or facts["outside_D"]:
        a = {"status": FAIL, "reason": "prefill/decode label sets overlap or leave D",
             "facts": facts}
    elif facts["realized"] != [p, d]:
        a = {"status": FAIL, "reason": f"requested {p}/{d} != realized "
             f"{facts['realized'][0]}/{facts['realized'][1]}", "facts": facts}
    elif p + d == len(D) and not facts["tiles_D"]:
        a = {"status": FAIL, "reason": "pair sums to |D| but does not tile D", "facts": facts}
    else:
        a = {"status": PASS, "reason": f"realized {p}/{d} == requested, disjoint", "facts": facts}

    conc = rec.get("concurrent") or []
    cf = {"requested": [p, d], "launches": len(conc), "overlap_ns": [], "shared": [],
          "prefill_escape": [], "decode_escape": [], "window_sizes": []}
    for blk in conc:
        ov, wp, wd = overlap_window(blk)
        cp = {s for s in blk["prefill"]["smid"] if s >= 0}
        cd = {s for s in blk["decode"]["smid"] if s >= 0}
        cf["overlap_ns"].append(ov)
        cf["shared"].append(sorted(cp & cd))
        cf["prefill_escape"].append(sorted(cp - Sp))
        cf["decode_escape"].append(sorted(cd - Sd))
        cf["window_sizes"].append([len(wp), len(wd)])
    if a["status"] == UNRES or not conc:
        b = {"status": UNRES, "reason": "no concurrent data or isolated census unresolved",
             "facts": cf}
    elif any(cf["shared"]) or any(cf["prefill_escape"]) or any(cf["decode_escape"]):
        b = {"status": FAIL, "reason": "concurrent launch shared labels or escaped the "
             "isolated set", "facts": cf}
    elif not any(ov > 0 for ov in cf["overlap_ns"]):
        b = {"status": UNRES, "reason": "the two kernels never overlapped in time", "facts": cf}
    else:
        b = {"status": PASS, "reason": f"{sum(1 for o in cf['overlap_ns'] if o > 0)}/"
             f"{len(conc)} launches overlapped in time; no shared label, no escape",
             "facts": cf}
    return a, b


def _fold(items):
    """Stage status from item statuses: any FAIL -> FAIL, else any UNRESOLVED ->
    UNRESOLVED, else PASS (empty -> UNRESOLVED)."""
    st = [i["status"] for i in items]
    if not st:
        return UNRES
    if FAIL in st:
        return FAIL
    if UNRES in st:
        return UNRES
    return PASS


def judge_grid(raw):
    if not raw:
        r = {"status": UNRES, "reason": "grid raw artefact missing/unreadable", "facts": {}}
        return r, dict(r), dict(r)
    plain = judge_plain(raw)
    ok, why = instrument_ok(raw)
    if not raw.get("complete"):
        r = {"status": UNRES, "reason": "grid run did not complete (partial artefact)",
             "facts": {"plain": plain}}
        return plain, r, dict(r)
    if not ok:
        r = {"status": UNRES, "reason": f"instrument check: {why}", "facts": {}}
        return plain, r, dict(r)
    if plain["status"] != PASS:
        r = {"status": UNRES, "reason": f"plain reference D not established ({plain['reason']})",
             "facts": {}}
        return plain, r, dict(r)
    D = _set(raw["plain_pre"])
    aa, bb = zip(*[judge_pair(rec, D) for rec in raw.get("pairs") or []]) if raw.get("pairs") \
        else ((), ())
    a = {"status": _fold(aa), "facts": {"pairs": list(aa)},
         "reason": "; ".join(f"{x['facts']['requested'][0]}/{x['facts']['requested'][1]}:"
                             f"{x['status']}" for x in aa)}
    b = {"status": _fold(bb), "facts": {"pairs": list(bb)},
         "reason": "; ".join(f"{x['facts']['requested'][0]}/{x['facts']['requested'][1]}:"
                             f"{x['status']}" for x in bb)}
    return plain, a, b


def judge_gran(recs, D):
    """DESCRIPTIVE table requested -> realized.  An outcome (EXACT / ROUNDED /
    REJECTED) is never a failure; the stage is PASS when every candidate produced a
    determinate outcome, UNRESOLVED otherwise, FAIL only on an isolation breach
    (overlap or label outside D) -- that would contradict B3a's premise."""
    rows = []
    for (p, d), rec in recs:
        row = {"requested": [p, d]}
        if rec is None or (not rec.get("complete") and not rec.get("create_error")):
            row.update(outcome="NO_RESULT", status=UNRES,
                       note="subprocess died/timed out before completing")
        elif not rec.get("created"):
            row.update(outcome="REJECTED", status=PASS, error=rec.get("create_error"))
        else:
            Sp, Sd = _set(rec.get("prefill")), _set(rec.get("decode"))
            dr = rec.get("driver") or {}
            ok, why = instrument_ok(rec)
            row.update(realized=[len(Sp), len(Sd)],
                       driver_smCount=[_driver_sm(dr, "prefill"), _driver_sm(dr, "decode")],
                       driver_triplet_prefill=(dr.get("prefill") or {}).get("green_sm"),
                       saturated=[_saturated(rec.get("prefill")), _saturated(rec.get("decode"))])
            if not ok or not all(row["saturated"]):
                row.update(outcome="UNSATURATED_OR_NO_INSTRUMENT", status=UNRES,
                           note=why if not ok else "census not saturated")
            elif (Sp & Sd) or (D and (Sp | Sd) - D):
                row.update(outcome="ISOLATION_BREACH", status=FAIL,
                           intersection=sorted(Sp & Sd), outside_D=sorted((Sp | Sd) - D))
            else:
                row.update(outcome="EXACT" if [len(Sp), len(Sd)] == [p, d] else "ROUNDED",
                           status=PASS)
        rows.append(row)
    st = _fold(rows)
    return {"status": st, "facts": {"rows": rows},
            "reason": "; ".join(f"{r['requested'][0]}/{r['requested'][1]}->"
                                + (f"{r['realized'][0]}/{r['realized'][1]}" if "realized" in r
                                   else r["outcome"]) for r in rows)}


R0_PASS = "GLOBALLY_CONSISTENT_LABEL"
R0_FAIL = ("NOT_A_GLOBALLY_CONSISTENT_LABEL", "LABEL_CONSISTENT_BUT_PARTITION_NOT_DELIVERED")


def judge_r0(v, kisti):
    """R0 verdict string, mapped mechanically.  KISTI 889631 values side by side."""
    if not v or not isinstance(v.get("R0"), dict):
        return {"status": UNRES, "reason": "R0 verdict json missing", "facts": {}}
    verdict = v["R0"].get("verdict")
    facts = {"verdict": verdict, "sizes": (v["R0"].get("facts") or {}).get("sizes"),
             "R1": {k: v.get("R1", {}).get(k) for k in ("realised", "target_from_divide_sm",
                                                         "exact_match")} if v.get("R1") else None,
             "R2_lost_count": (v.get("R2") or {}).get("lost_count"),
             "kisti_889631": {"verdict": ((kisti or {}).get("R0") or {}).get("verdict"),
                              "sizes": (((kisti or {}).get("R0") or {}).get("facts") or {})
                              .get("sizes")}}
    if verdict == R0_PASS:
        st = PASS
    elif verdict in R0_FAIL:
        st = FAIL
    else:
        st = UNRES
    return {"status": st, "reason": f"R0 verdict = {verdict}", "facts": facts}


def judge_p0a(v, kisti):
    """P0-A verdict string from p0a_verdict_<tag>.json (the only citable artefact,
    gate #56), mapped mechanically; rule vocabulary = bcg_probe/p0a_rule_totality.py."""
    if not v or "verdict" not in v:
        return {"status": UNRES, "reason": "P0-A verdict json missing", "facts": {}}
    verdict = v["verdict"]
    facts = {"verdict": verdict, "exact_set_match": v.get("exact_set_match"),
             "division_under_test": v.get("run_division_under_test"),
             "green_pair_disjoint": v.get("green_pair_disjoint"),
             "green_pair_tiles_D": v.get("green_pair_tiles_D"),
             "D_size": v.get("D_size"), "min_hits_by_leg": v.get("min_hits_by_leg"),
             "substrate_mismatch": v.get("substrate_mismatch"),
             "kisti_890893": {"verdict": (kisti or {}).get("verdict"),
                              "division_under_test": (kisti or {}).get("run_division_under_test"),
                              "D_size": (kisti or {}).get("D_size")}}
    if verdict == "CONFINEMENT_PRESERVED_THROUGH_GRAPH_REPLAY":
        st = PASS
    elif verdict.startswith("CONFINEMENT_LOST") or verdict.startswith("GREEN PARTITION NOT"):
        st = FAIL
    else:
        st = UNRES
    return {"status": st, "reason": f"P0-A verdict = {verdict}", "facts": facts}


def _write_status(path, stage, v):
    v = dict(v, stage=stage)
    _dump(v, path)
    print(f"{stage}: {v['status']} -- {v['reason']}")


def cmd_score_b3(a):
    b3 = a.b3dir
    st = a.status_dir
    os.makedirs(st, exist_ok=True)
    tag = a.tag
    parts = set(a.parts.split(","))
    # R0 (scored only when it was attempted)
    if "r0" in parts and os.path.isdir(os.path.join(b3, "r0")):
        _write_status(os.path.join(st, "B3-R0.json"), "B3-R0",
                      judge_r0(_load(os.path.join(b3, "r0", f"smid_l0_verdict_{tag}.json")),
                               _load(os.path.join(SMID_DIR, "smid_l0_verdict_889631.json"))))
    # grid
    raw = _load(os.path.join(b3, "grid", "b3_grid_raw.json"))
    plain, ga, gb = judge_grid(raw)
    if "grid" in parts:
        _write_status(os.path.join(st, "B3-plain.json"), "B3-plain", plain)
        _write_status(os.path.join(st, "B3a.json"), "B3a", ga)
        _write_status(os.path.join(st, "B3b.json"), "B3b", gb)
    # granularity
    D = _set((raw or {}).get("plain_pre")) if plain["status"] == PASS else set()
    recs = []
    for p, d in GRAN_CANDIDATES:
        f = os.path.join(b3, "gran", f"gran_{p}_{d}.json")
        if os.path.exists(f) or os.path.exists(f + ".attempted"):
            recs.append(((p, d), _load(f)))
    if "gran" in parts and recs:
        _write_status(os.path.join(st, "B3-gran.json"), "B3-gran", judge_gran(recs, D))
    # P0-A
    pv = os.path.join(b3, "p0a", f"p0a_verdict_{tag}.json")
    if "p0a" in parts and (os.path.exists(pv)
                           or os.path.exists(os.path.join(b3, "p0a", ".attempted"))):
        _write_status(os.path.join(st, "B3c.json"), "B3c",
                      judge_p0a(_load(pv), _load(os.path.join(BCG_DIR, "p0a_verdict_890893.json"))))
    return 0


# ===========================================================================
# B4 -- requests + judgment
# ===========================================================================
def _post(port, prompt, timeout):
    import urllib.request
    body = json.dumps({"text": prompt, "sampling_params": P13_SAMPLING}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        rec = json.loads(r.read().decode())
    mi = rec.get("meta_info") or {}
    return {"text": rec.get("text"),
            "completion_tokens": mi.get("completion_tokens"),
            "prompt_tokens": mi.get("prompt_tokens"),
            "finish_reason": mi.get("finish_reason")}


def cmd_b4_requests(a):
    """Sequential pass (the P13 precedent), then the same six prompts issued at once
    (exercises decode batch > 1 and the PD-mux split path).  No timing recorded."""
    from concurrent.futures import ThreadPoolExecutor
    out = {"kind": "substrate_probe_b4_requests", "utc": _utc(),
           "sampling_params": P13_SAMPLING, "prompts": list(P13_PROMPTS),
           "sequential": [], "concurrent": []}

    def one(prompt):
        try:
            return _post(a.port, prompt, a.timeout)
        except Exception as exc:  # noqa: BLE001
            return {"error": repr(exc)}
    for p in P13_PROMPTS:
        out["sequential"].append(one(p))
        _dump(out, a.out)
    with ThreadPoolExecutor(max_workers=len(P13_PROMPTS)) as ex:
        out["concurrent"] = list(ex.map(one, P13_PROMPTS))
    _dump(out, a.out)
    n_ok = sum(1 for r in out["sequential"] + out["concurrent"] if r.get("text"))
    print(f"[b4-requests] non-empty responses {n_ok}/{2 * len(P13_PROMPTS)} -> {a.out}")
    return 0


def parse_server_log(text):
    """Mechanical facts from the SGLang server log."""
    text = text or ""
    def last_int(pat):
        m = re.findall(pat, text)
        return int(m[-1]) if m else None
    sa = re.search(r"disable_cuda_graph=(True|False)", text)
    return {
        "capture_bs_lines": len(re.findall(r"Capture cuda graph bs", text)),
        "capture_failed_lines": len(re.findall(r"Capture cuda graph failed", text)),
        "disable_cuda_graph": (sa.group(1) == "True") if sa else None,
        "enable_pdmux": (re.search(r"enable_pdmux=(True|False)", text).group(1) == "True")
        if re.search(r"enable_pdmux=(True|False)", text) else None,
        "traceback_lines": len(re.findall(r"Traceback \(most recent call last\)", text)),
        "max_mamba_cache_size": last_int(r"max_mamba_cache_size: *([0-9]+)"),
        "max_total_num_tokens": last_int(r"max_total_num_tokens=([0-9]+)"),
        "true_dual_enabled_lines": len(re.findall(r"true dual", text, re.I)),
    }


def judge_green_readout(gro, targets):
    """In-server driver read-out (PDMUX_GREEN_READOUT): for every stream group whose
    target has both halves > 0 the driver must report a green context with smCount
    == target on each stream; plain groups must report no green context."""
    if not gro or not gro.get("groups"):
        return {"status": UNRES, "reason": "green read-out file missing/empty", "facts": {}}
    rows, bad = [], []
    for g in gro["groups"]:
        tp, td = g.get("target_prefill_sm"), g.get("target_decode_sm")
        row = {"idx": g.get("stream_index"), "target": [tp, td],
               "driver": [((g.get("prefill") or {}).get("green_sm") or {}).get("smCount"),
                          ((g.get("decode") or {}).get("green_sm") or {}).get("smCount")],
               "green_null": [(g.get("prefill") or {}).get("green_ctx_is_null"),
                              (g.get("decode") or {}).get("green_ctx_is_null")]}
        if tp and td:
            row["ok"] = row["green_null"] == [False, False] and row["driver"] == [tp, td]
        else:
            row["ok"] = row["green_null"] == [True, True]
        if not row["ok"]:
            bad.append(row["idx"])
        rows.append(row)
    facts = {"groups": rows, "engine_targets_from_cfg": targets}
    if bad:
        return {"status": FAIL, "reason": f"groups {bad}: driver read-out != engine target",
                "facts": facts}
    return {"status": PASS, "reason": f"{len(rows)} groups: driver smCount == target (green), "
            "no green context (plain)", "facts": facts}


def judge_b4(health_ok, reqs, log_facts):
    facts = {"health_ok": health_ok, "server_log": log_facts}
    if not health_ok:
        return {"status": FAIL, "reason": "server never reached /health 200", "facts": facts}
    if not reqs:
        return {"status": FAIL, "reason": "request artefact missing", "facts": facts}
    allr = (reqs.get("sequential") or []) + (reqs.get("concurrent") or [])
    n_ok = sum(1 for r in allr if r.get("text"))
    facts["responses_nonempty"] = f"{n_ok}/{2 * len(P13_PROMPTS)}"
    facts["errors"] = [r.get("error") for r in allr if r.get("error")][:3]
    facts["sequential_equals_concurrent_text"] = (
        [r.get("text") for r in reqs.get("sequential") or []]
        == [r.get("text") for r in reqs.get("concurrent") or []])     # descriptive only
    if n_ok != 2 * len(P13_PROMPTS):
        return {"status": FAIL, "reason": f"only {n_ok}/{2 * len(P13_PROMPTS)} non-empty "
                "responses", "facts": facts}
    if log_facts.get("disable_cuda_graph") is True:
        return {"status": FAIL, "reason": "server ran with disable_cuda_graph=True (not the "
                "operating point)", "facts": facts}
    if log_facts.get("capture_failed_lines"):
        return {"status": FAIL, "reason": "cuda graph capture failed", "facts": facts}
    if log_facts.get("traceback_lines"):
        return {"status": FAIL, "reason": "Traceback in server log", "facts": facts}
    if log_facts.get("enable_pdmux") is not True:
        return {"status": UNRES, "reason": "enable_pdmux=True not found in server log",
                "facts": facts}
    if log_facts.get("disable_cuda_graph") is None or not log_facts.get("capture_bs_lines"):
        return {"status": UNRES, "reason": "no evidence of cuda graph capture in the log",
                "facts": facts}
    return {"status": PASS, "reason": "boot ok, 12/12 non-empty greedy responses, cuda graph "
            "captured (disable_cuda_graph=False), pdmux on, no Traceback", "facts": facts}


def cmd_score_b4(a):
    log = open(a.srvlog, errors="replace").read() if os.path.exists(a.srvlog) else ""
    lf = parse_server_log(log)
    v = judge_b4(os.path.exists(a.health_flag), _load(a.requests), lf)
    tel = {"path": a.telemetry, "lines": None}
    if a.telemetry and os.path.exists(a.telemetry):
        with open(a.telemetry, errors="replace") as f:
            tel["lines"] = sum(1 for _ in f)
    v["facts"]["telemetry"] = tel
    _write_status(os.path.join(a.status_dir, "B4.json"), "B4", v)
    try:
        import yaml
        cfg = yaml.safe_load(open(a.cfg))
        targets = [[r[0], r[1]] for r in cfg.get("manual_divisions") or []]
    except Exception:  # noqa: BLE001
        targets = None
    _write_status(os.path.join(a.status_dir, "B4-green.json"), "B4-green",
                  judge_green_readout(_load(a.green), targets))
    return 0


# ===========================================================================
# Clock trace + summary
# ===========================================================================
IDLE_MASK = 0x1     # clocks_event_reasons.gpu_idle


def parse_clk(text):
    """nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,temperature.gpu,
    clocks_throttle_reasons.active --format=csv -lms 100.  Returns rows of
    (epoch_s, sm_mhz, reason_mask)."""
    rows = []
    for ln in (text or "").splitlines()[1:]:
        c = [x.strip() for x in ln.split(",")]
        if len(c) < 5:
            continue
        try:
            ts = time.mktime(time.strptime(c[0].split(".")[0], "%Y/%m/%d %H:%M:%S"))
            frac = float("0." + c[0].split(".")[1]) if "." in c[0] else 0.0
            mhz = int(c[1].split()[0]) if c[1].split() and c[1].split()[0].isdigit() else None
            mask = int(c[4], 16) if c[4].startswith("0x") else None
        except (ValueError, IndexError):
            continue
        rows.append((ts + frac, mhz, mask))
    return rows


def clock_summary(rows, windows):
    """Per stage window: samples, SM clock min/max, samples with any NON-IDLE reason
    active.  (job_entry.sh's clk_summary counts the idle bit 0x1 as throttle; this
    one separates it.)"""
    out = {}
    for name, (t0, t1) in windows.items():
        sel = [r for r in rows if t0 <= r[0] <= t1]
        mhz = [r[1] for r in sel if r[1] is not None]
        nonidle = [r for r in sel if r[2] is not None and (r[2] & ~IDLE_MASK)]
        out[name] = {"samples": len(sel),
                     "sm_mhz_min": min(mhz) if mhz else None,
                     "sm_mhz_max": max(mhz) if mhz else None,
                     "nonidle_reason_samples": len(nonidle),
                     "nonidle_masks": sorted({hex(r[2]) for r in nonidle})[:8],
                     "flag": bool(nonidle)}
    return out


STAGE_ORDER = ("PRE", "B0", "B3-R0", "B3-plain", "B3a", "B3b", "B3-gran", "B3c",
               "B4", "B4-green")


def _fmt_facts(stage, v):
    f = v.get("facts") or {}
    if stage == "B0":
        g = f.get("gpu") or {}
        return (f"{g.get('name')} | cc {g.get('compute_cap')} | driver {g.get('driver_version')}"
                f" | MIG {g.get('mig_mode')} | get_sm_available {f.get('get_sm_available_0')}")
    if stage in ("B3a", "B3b"):
        return "; ".join(f"{p['facts'].get('requested')}->{p['facts'].get('realized', '')} "
                         f"{p['status']}" for p in f.get("pairs") or [])
    return ""


def cmd_summary(a):
    rd = a.rundir
    st_dir = os.path.join(rd, "status")
    stages = {}
    for name in os.listdir(st_dir) if os.path.isdir(st_dir) else []:
        if name.endswith(".json"):
            v = _load(os.path.join(st_dir, name))
            if v:
                stages[v.get("stage", name[:-5])] = v
    win = {}
    tsv = os.path.join(rd, "meta", "stage_times.tsv")
    if os.path.exists(tsv):
        for ln in open(tsv):
            parts = ln.rstrip("\n").split("\t")
            if len(parts) == 3:
                try:
                    win[parts[0]] = (float(parts[1]), float(parts[2]))
                except ValueError:
                    pass
    clk_path = os.path.join(rd, "meta", "clk.csv")
    clk = clock_summary(parse_clk(open(clk_path, errors="replace").read()), win) \
        if os.path.exists(clk_path) else {}
    meta = _load(os.path.join(rd, "meta", "run.json")) or {}
    summary = {"kind": "substrate_probe_summary", "utc": _utc(), "meta": meta,
               "stages": stages, "clock_by_stage": clk,
               "stage_wall_s": {k: round(t1 - t0, 1) for k, (t0, t1) in win.items()}}
    _dump(summary, os.path.join(rd, "summary.json"))

    L = [f"# Substrate probe SUMMARY — {os.path.basename(rd)}", "",
         "> 기계적 사실만 기록한다(PASS/FAIL/UNRESOLVED/SKIPPED). 성능 해석·기판 동등성 판정 없음.",
         "> `%smid` 집합 크기는 라벨 집합 카디널리티다(R0 label ceiling) — 계산량으로 환산 금지.",
         "> 판정 원천: `status/*.json`(규칙: `scripts/vessl/substrate_probe_helper.py`),"
         " P0-A/R0는 각 도구 자신의 verdict json.", ""]
    L.append(f"- mode: {meta.get('mode')} · commit: `{meta.get('commit')}` (dirty={meta.get('git_dirty')})"
             f" · host: {meta.get('host')}")
    L.append(f"- runtime manifest: `{meta.get('manifest')}` sha256 `{meta.get('manifest_sha256')}`")
    L.append(f"- start {meta.get('start_utc')} · end {_utc()}")
    L += ["", "| stage | status | reason | wall s | non-idle clock-reason samples |",
          "|---|---|---|---|---|"]
    for s in STAGE_ORDER:
        v = stages.get(s)
        if not v:
            continue
        c = clk.get(s, {})
        reason = str(v.get("reason", "")).replace("|", "/")
        L.append(f"| {s} | **{v['status']}** | {reason[:300]} | "
                 f"{summary['stage_wall_s'].get(s, '')} | "
                 f"{c.get('nonidle_reason_samples', '')}/{c.get('samples', '')} |")
    L.append("")
    for s in ("B0", "B3a", "B3b"):
        if s in stages and stages[s].get("status") != SKIP and _fmt_facts(s, stages[s]):
            L.append(f"- {s}: {_fmt_facts(s, stages[s])}")
    g = stages.get("B3-gran")
    if g and (g.get("facts") or {}).get("rows"):
        L += ["", "## B3-gran (descriptive: requested -> realized |%smid set|, driver smCount)",
              "", "| requested p/d | outcome | realized | driver smCount | note |", "|---|---|---|---|---|"]
        for r in g["facts"].get("rows", []):
            L.append(f"| {r['requested'][0]}/{r['requested'][1]} | {r['outcome']} | "
                     f"{r.get('realized', '')} | {r.get('driver_smCount', '')} | "
                     f"{str(r.get('error') or r.get('note') or '')[:120]} |")
    for s, key in (("B3-R0", "kisti_889631"), ("B3c", "kisti_890893")):
        v = stages.get(s)
        if v and v.get("facts"):
            L.append("")
            L.append(f"- {s} vs KISTI ({key}): this run = `{v['facts'].get('verdict')}`, "
                     f"KISTI = `{(v['facts'].get(key) or {}).get('verdict')}` (나란히 기록만)")
    if clk.get("ALL"):
        c = clk["ALL"]
        L.append("")
        L.append(f"- clock trace (whole run): samples {c['samples']}, SM MHz {c['sm_mhz_min']}–"
                 f"{c['sm_mhz_max']}, non-idle reason samples {c['nonidle_reason_samples']} "
                 f"{c['nonidle_masks']}")
    L.append("")
    with open(os.path.join(rd, "SUMMARY.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


# ===========================================================================
# CLI
# ===========================================================================
def cmd_b4_argcheck(a):
    """CPU-only: the exact B4 argv must parse with SGLang's own CLI definition and the
    pdmux config must load with the engine's own loader.  No model, no GPU."""
    import argparse as _ap
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.multiplex.pdmux_context import load_pdmux_config
    argv = list(a.server_args)
    if argv and argv[0] == "--":
        argv = argv[1:]
    parser = _ap.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    ns = parser.parse_args(argv)
    cfg = load_pdmux_config(ns.pdmux_config_path)
    rec = {"enable_pdmux": ns.enable_pdmux, "disable_cuda_graph": ns.disable_cuda_graph,
           "disable_overlap_schedule": ns.disable_overlap_schedule,
           "chunked_prefill_size": ns.chunked_prefill_size,
           "attention_backend": ns.attention_backend, "model_path": ns.model_path,
           "pdmux_sm_group_num": cfg.sm_group_num,
           "pdmux_manual_divisions": cfg.manual_divisions}
    print(json.dumps(rec))
    ok = (ns.enable_pdmux and not ns.disable_cuda_graph and ns.disable_overlap_schedule
          and ns.chunked_prefill_size == -1 and cfg.manual_divisions)
    return 0 if ok else 3


def cmd_print_grid(a):
    pairs, src = grid_from_configs(a.configs.split(","))
    print(json.dumps({"grid": pairs, "sources": src,
                      "granularity_candidates": GRAN_CANDIDATES}))
    return 0


def cmd_gran_list(a):
    for p, d in GRAN_CANDIDATES:
        print(p, d)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("b0-python"); s.add_argument("--out", required=True)
    s = sub.add_parser("score-b0")
    s.add_argument("--gpu-csv", required=True); s.add_argument("--py-json", required=True)
    s.add_argument("--status", required=True)
    s = sub.add_parser("b3-grid")
    s.add_argument("--configs", default=",".join(GRID_CONFIGS))
    s.add_argument("--out", required=True); s.add_argument("--outdir", required=True)
    s = sub.add_parser("b3-gran-one")
    s.add_argument("--p", type=int, required=True); s.add_argument("--d", type=int, required=True)
    s.add_argument("--out", required=True)
    s = sub.add_parser("score-b3")
    s.add_argument("--b3dir", required=True); s.add_argument("--status-dir", required=True)
    s.add_argument("--tag", required=True)
    s.add_argument("--parts", default="r0,grid,gran,p0a")
    s = sub.add_parser("b4-requests")
    s.add_argument("--port", type=int, required=True); s.add_argument("--out", required=True)
    s.add_argument("--timeout", type=int, default=180)
    s = sub.add_parser("score-b4")
    for k in ("srvlog", "requests", "health-flag", "green", "telemetry", "cfg", "status-dir"):
        s.add_argument(f"--{k}", required=(k != "telemetry"), default="")
    s = sub.add_parser("summary"); s.add_argument("--rundir", required=True)
    s = sub.add_parser("print-grid"); s.add_argument("--configs", default=",".join(GRID_CONFIGS))
    sub.add_parser("gran-list")
    s = sub.add_parser("b4-argcheck")
    s.add_argument("server_args", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    return {"b0-python": cmd_b0_python, "score-b0": cmd_score_b0, "b3-grid": cmd_b3_grid,
            "b3-gran-one": cmd_b3_gran_one, "score-b3": cmd_score_b3,
            "b4-requests": cmd_b4_requests, "score-b4": cmd_score_b4,
            "summary": cmd_summary, "print-grid": cmd_print_grid,
            "b4-argcheck": cmd_b4_argcheck,
            "gran-list": cmd_gran_list}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
