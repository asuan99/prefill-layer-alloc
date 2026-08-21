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
                         return UNDETERMINED, never a substantive verdict) and
                         the runtime-PTX read-out against ahead-of-time
                         compiled kernels. Then runs --selftest-mutants.
    --selftest-mutants   GPU-free. Re-runs that same battery against COPIES of
                         this file with the 2026-08-21 repair undone, and
                         demands that the named checks actually FAIL
                         (methodology lesson #53: a check that cannot fail is
                         an identity, not evidence).
    --run                REQUIRES GPU. Emits raw census artefact. Makes no
                         verdict; scoring is a separate invocation on purpose.
    --analyze RAW.json   GPU-free. Applies the pre-registered rules to a raw
                         artefact and emits the verdict artefact.

GATE #21 (this project's signature error, 7 recurrences: a MEASUREMENT failure
labelled as a GATE failure). The scoring path in this file therefore has the
"did the instrument produce data at all" branch FIRST, and it returns
`UNDETERMINED (MEASUREMENT ABSENT)`. A missing/short/unsaturated census is never
reported as "no carve", "delivered", or any other substantive statement.

★2026-08-21 FAIL-OPEN REPAIR (this is a GUARD repair, NOT a decision-rule
change; the 4-outcome matrix of prereg sec3.2 and the stop rule of sec3.3 are
untouched, and no verdict string was added, removed or re-worded). Two defects
registered by the 2026-08-14 audit (`PROJECT_STATUS.md:4212`, restated in
`PREREG_SMID_R0_2026-08-14.md` sec15.1-2) are fixed:
  1. the runtime PTX read-out was DEAD CODE -- it read `JITFunction.cache`,
     an attribute triton 3.5.1 does not have, so it raised on every run and
     wrote None. See `_compiled_variants` for the confirmed attribute path.
  2. the two scorer guards read `== 0` and `is False`, which a None (i.e. an
     ABSENT check) passed. They are now fail-closed: absence stops with
     `UNDETERMINED (MEASUREMENT ABSENT)`, the same string absence already got
     everywhere else in the gate #21 branch.
Neither half is self-evidencing: `--selftest-mutants` undoes each half in a
copy of this source and asserts the corresponding checks flip to FAIL.

★2026-08-21 SECOND REPAIR (audit of the fail-open repair; again GUARD/PREMISE
work only -- the verdict vocabulary, the 4-outcome matrix of prereg sec3.2 and
the stop rule of sec3.3 are untouched, and no verdict string was added, removed
or re-worded). What was missing was PREMISES, not decision rules:
  C1 an EMPTY census scored `GLOBALLY_CONSISTENT_LABEL`. Three vacuous truths
     (empty-cap-empty is disjoint, empty equals empty tiles D, empty->empty is
     "saturated") let a census in which no block ever reported an id reach a
     substantive verdict. D1 asks "has the ladder stopped growing", never "was
     anything observed at all". `_target_status` now asks the second question
     FIRST and returns `UNDETERMINED (MEASUREMENT ABSENT)`.
  C2 `S_post` and the idx2+ pairs were absent from the D1 hard gate, so R2
     published `lost_ids`/`lost_count` -- an ABSENCE claim, which prereg
     sec4.1 forbids for an unsaturated target -- off an unsaturated `S_post`.
     The gate is now PER TARGET and the forbidden quantity is not computed at
     all (a flag beside the number is not what sec4.1 asks for).
  C3 `LABEL_CONSISTENT_BUT_PARTITION_NOT_DELIVERED` merged two worlds: a green
     context attached but its SM limit not delivered (a result about the
     driver API here) and no green context attached at all (an instrument
     failure). `run()` already records `cuStreamGetGreenCtx` per stream, but
     the AST guard banned the whole driver block, so `score()` could not read
     even that boolean. The ban is now the target-NUMBER family only, the
     boolean is whitelisted as a setup premise, and the guard follows helper
     calls transitively (it used to inspect `score()`'s own body only).
  C7 provenance: git HEAD, the runtime source manifest sha and the torch /
     triton versions are recorded into the raw artefact (prereg I7 had zero
     lines of implementation).
  C8 `_record_runtime_ptx` no longer carries the unconditional no-raise
     promise; the exact contract is written out instead (the PTX transcript
     write is deliberately loud). ★2026-08-21: this line used to QUOTE that
     promise verbatim, and the docstring sweep A39 grew on the same day
     duly reported the module docstring as an offender -- lesson #54 caught
     live, fail-closed. Describing a watched string means paraphrasing it.
  C9 the mutation harness now also rejects UNEXPECTED failures, so a mutant
     that breaks something it does not claim to cover can no longer PASS.
Every one of these is covered by a named mutant in `MUTANTS` (lesson #53).

★2026-08-21 THIRD REPAIR (re-audit of the second one; GUARD/PREMISE/HARNESS
work only -- the verdict vocabulary, the 4-outcome matrix of prereg sec3.2,
the stop rule of sec3.3, the grid/repeat/spin constants and the 0.9 coverage
fraction are untouched, and no verdict string was added, removed or re-worded).
The re-audit found that three of the second repair's own guarantees were
unfalsifiable and one struck phrase had come straight back:
  R2 (blocking) the attachment premise C3 added was ONE-DIRECTIONAL: it could
     only ever observe "attached", so a read-out stuck on that answer
     satisfied it. The auditor hardwired the producer to "attached" for every
     stream, then to "not attached" for every stream, and all 50 checks passed
     both times -- with the C3 defect back in the first world AND the verdict
     testifying that the premise had been checked and held (gate #42's own
     shape). The artefact already contained the negative control -- five
     streams that `run()` builds with no green context -- and nothing read it.
     The premise is now two-sided (green pair attached AND at least one plain
     control detached, both recorded in `setup`), and the read-out PRODUCER is
     exercised directly on known-answer input for the first time (A44/A45; the
     runtime-PTX read-out had eight such checks, this one had none). Both of
     the auditor's degenerate worlds are now named mutants.
  R7 the comment on `_DRIVER_KEY_SURFACE` claimed it enumerated every key the
     read-out can emit; `rc`, `error` and `note` were missing, so `score()`
     could read `rc` and no check failed. The surface is complete, A47 derives
     the emitted set from the producer's AST and demands the surface cover it,
     and the ban now looks at key-READ positions so the three generic names
     can be listed without a false positive on `score()`'s own output keys.
  R9 the positivity floor was written as a bare `1` twice while sec12 declared
     it not a free parameter; raising both to 2 failed nothing. It now reads
     `MIN_HITS_REPORTED` -- a constant that was DEAD and whose comment ("not a
     threshold") the C1 repair had silently falsified inside the block that
     calls such edits a protocol violation. A48 (behaviour) and A49 (wiring)
     hold it there.
  R12 the new `_provenance` docstring re-used the exact unconditional no-raise
     phrase C8 had struck from `_record_runtime_ptx` hours earlier, because
     A39 watched one function by name. A39 walks a list now, and every
     function on it states its real contract.
Same discipline as before: each item has a named mutant, and the two the
auditor ran by hand (`readout_degenerate_all_attached`,
`readout_degenerate_all_detached`) are in `MUTANTS` so the claim "a degenerate
read-out is invisible here" is one this file can refute by itself.

GATE #33 (a manifest N/N sha match does not imply the runtime is byte-identical:
`sync_engine_tree.sh:99-115` hashes exactly 15 files and neither
`sglang/srt/multiplex/pdmux_context.py` nor `sgl_kernel/spatial.py` is one of
them). Both of those files are on this script's critical path, so `--run`
records their sha256 directly into the artefact instead of trusting a manifest.

Author: engine-porter, 2026-08-14. GPU spend by this file at authoring time: 0
(only --selftest-cpu and --selftest-analyzer were executed).
Fail-open repair: engine-porter, 2026-08-21. GPU spend by the repair: 0 (only
--selftest-cpu, --selftest-analyzer and --selftest-mutants were executed; the
census has still NEVER been run on a GPU, so this file contains no measurement
of anything and the R0 verdict remains unmade).
Audit repair (C1/C2/C3/C7/C8/C9): engine-porter, 2026-08-21, same day, after
`audit_smid_r0_2026-08-21/VERDICT.md`. GPU spend: 0 -- unchanged, the census
has still never run on a GPU. The repair is NOT self-certifying: it must be
re-audited (the audit says so explicitly), and C4/C5 (documentation) plus the
prereg edits this repair implies -- sec6.2's token list, sec5.2's table of
measurement-failure conditions -- are NOT done here.
Re-audit repair (R2/R7/R9/R12): engine-porter, 2026-08-21, same day, after the
re-audit of the above. GPU spend: 0 -- the census STILL has never run on a
GPU, this file still contains no measurement of anything, and the R0 verdict
remains unmade. WHAT THIS REPAIR DOES NOT CLOSE: R1, R3, R4, R5 and R6 are
documentation findings owned by the main session (prereg sec6.2's token list,
sec5.2's failure-condition table, the R5 absence claim and the value-laundering
item), and none of them is touched here. Nor is this repair self-certifying
either: it is the third pass over the same file and the previous two both
looked complete from the inside.
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
# The PRESENCE floor, read by `_target_status` (2026-08-21, audit item R9).
# It is the boundary between "the instrument produced data" and "it did
# not", NOT a coverage criterion -- coverage is D1's job (`_saturated`) --
# and not a free parameter: 1 is the smallest observation that exists.
# HISTORY: this constant was DEAD until 2026-08-21 and its old comment
# ("it is not a threshold") stopped being true the moment the C1 positivity
# floor was added -- with the bound written as a bare literal `1` in two
# places, so raising it was a silent, unguarded edit inside the very block
# that declares such edits a protocol violation. The floor now READS this
# constant (checks A48/A49, mutants raise_/hardwire_positivity_floor).
MIN_HITS_REPORTED = 1
CUDA_HOME = os.environ.get("CUDA_HOME", "/apps/cuda/13.0.2")
CUOBJDUMP = os.path.join(CUDA_HOME, "bin", "cuobjdump")

# Provenance (audit item C7, 2026-08-21). prereg I7 ("dev tree content
# unchanged") had no implementation at all: nothing in the artefact said which
# source tree produced it. These two paths are derived from this file's
# location, not from the cwd, so a job that runs from anywhere still records
# the right tree.
_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))
MANIFEST_PATH = os.path.abspath(
    os.path.join(_HERE, "..", "runtime_source_manifest.sha256"))
# `sync_engine_tree.sh:99-115` hashes exactly this many files. Recorded so a
# manifest written by an older revision of that script is visible as such
# rather than passing as "the manifest" (gate #33).
MANIFEST_EXPECTED_FILES = 15

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


# --------------------------------------------------------------------------
# 2b. "which kernel actually ran" read-out (prereg sec5.2, the runtime PTX
#     re-check). REPAIRED 2026-08-21 -- see the block comment below.
# --------------------------------------------------------------------------
def _compiled_variants(kernel, device):
    """Every JIT-compiled variant of `kernel` that Triton holds for `device`.

    ATTRIBUTE PATH IS A CONFIRMED CODE FACT, NOT A GUESS (methodology gate
    #28). Read out of the Triton actually installed in this venv on
    2026-08-21 (triton 3.5.1, python 3.14.2,
    site-packages/triton/ under the pytorch_2.9.1_cuda13 env):

      runtime/jit.py:784    `self.device_caches = defaultdict(self.create_binder)`
      runtime/jit.py:672-683 the factory returns `{}, {}, target, backend, binder`
                            -- element 0 IS the per-device kernel cache dict
      runtime/jit.py:720    `run()` unpacks `self.device_caches[device]`
      runtime/jit.py:860-862 `kernel_cache[key] = kernel` after a compile
      backends/driver.py:61 `self.get_current_device = torch.cuda.current_device`
                            -- so the dict key is exactly the int that `run()`
                            in this file already holds as `dev`
      compiler/compiler.py:431 `self.asm = AsmDict({...})`, hence `asm["ptx"]`

    THE DEFECT THIS REPLACES (2026-08-14 audit, `PROJECT_STATUS.md:4212`;
    restated in `PREREG_SMID_R0_2026-08-14.md` sec15.1-2): the previous line
    read `census_kernel.cache[dev]`. `triton.runtime.jit.JITFunction` in 3.5.1
    has NO `cache` attribute -- measured, not assumed:
    `hasattr(_kernels()[0], "cache") is False`. So the read raised
    `AttributeError` on EVERY run, the except arm wrote None into the
    artefact, and the two scorer guards (`== 0`, `is False`) let None through
    -- i.e. the census was scored without ever checking that the kernel which
    ran was the probe. Methodology gate #42 shape (a gate labelling its own
    failure a success). The guards below in `score()` are now fail-closed too;
    both halves are needed, and both are covered by `--selftest-mutants`.

    Returns a list of `(device_key, cache_key, CompiledKernel)`.

    `device_caches` is a `defaultdict` whose factory calls
    `driver.active.get_current_target()`, so it is read with `in`/`.get()`
    (neither triggers `__missing__`): a probe must not mutate the cache it is
    inspecting, nor touch the driver. If the exact device key is absent, the
    variants of EVERY key are returned instead -- over-reporting is safe
    because the caller aggregates conservatively (min over `%smid` sites, AND
    over back edges), whereas under-reporting would be fail-open again.
    """
    out = []
    caches = getattr(kernel, "device_caches", None)
    if isinstance(caches, dict):
        keys = [device] if device in caches else sorted(caches, key=repr)
        for k in keys:
            entry = caches.get(k)
            # (kernel_cache, kernel_key_cache, target, backend, binder)
            if isinstance(entry, (tuple, list)) and entry and isinstance(entry[0], dict):
                for ck_key, obj in list(entry[0].items()):
                    out.append((k, ck_key, obj))
    else:
        # Triton <= 3.0 shape. Kept ONLY as a fallback for a different tree;
        # it is not the path taken in this venv (see docstring).
        legacy = getattr(kernel, "cache", None)
        if not isinstance(legacy, dict):
            raise AttributeError(
                "kernel exposes neither a dict `.device_caches` (triton 3.5.x) "
                "nor a dict `.cache` (legacy); the PTX that actually ran cannot "
                "be located, so this census cannot be certified")
        keys = [device] if device in legacy else sorted(legacy, key=repr)
        for k in keys:
            sub = legacy.get(k)
            if isinstance(sub, dict):
                for ck_key, obj in list(sub.items()):
                    out.append((k, ck_key, obj))
    if not out:
        raise LookupError(
            "no compiled variant is cached for this kernel; either it never "
            "ran or the cache moved -- either way the runtime instrument was "
            "not verified")
    # async compilation stores futures; jit.py:748 resolves them the same way
    return [(a, b, (c.result() if hasattr(c, "result") else c)) for a, b, c in out]


def runtime_ptx_probe(kernel, device):
    """Runtime-instrument fields for the raw artefact.

    FAILURE CONTRACT (re-worded 2026-08-21, audit item R12 -- the blanket
    no-raise phrase this line used to carry is the one audit item C8 struck
    from `_record_runtime_ptx`, and A39 now watches this docstring too; the
    phrase is deliberately not reproduced, lesson #54). What holds: every
    failure of the read itself -- a missing attribute path, an empty cache, a
    variant whose PTX cannot be read -- is caught, leaves all four scalar
    fields None and reports `runtime_ptx_error`, which `score()` reads as
    UNDETERMINED (MEASUREMENT ABSENT). What is NOT promised: the catch is
    `except Exception`, so a BaseException (KeyboardInterrupt, MemoryError,
    SystemExit) escapes, exactly as in `_provenance`.

    Aggregation is deliberately CONSERVATIVE: `runtime_ptx_smid_sites` is the
    MINIMUM over variants and `runtime_spin_back_edge` is the AND, so a single
    variant that lost the probe or the residency loop stops the scorer. On any
    failure every scalar field stays None, and `score()` now reads None as
    `UNDETERMINED (MEASUREMENT ABSENT)`.

    `runtime_ptx_sha256` keeps its old meaning: the sha of the PTX written to
    `smid_runtime_<tag>.ptx` (variant 0). Per-variant shas live in `variants`.
    """
    out = {"variants": [], "runtime_ptx_smid_sites": None,
           "runtime_ptx_globaltimer_sites": None,
           "runtime_spin_back_edge": None, "runtime_ptx_sha256": None}
    try:
        found = _compiled_variants(kernel, device)
        for dev_key, ck_key, obj in found:
            ptx = obj.asm["ptx"]
            out["variants"].append({
                "device_key": repr(dev_key), "cache_key": repr(ck_key),
                "smid_sites": ptx.count("%smid"),
                "globaltimer_sites": ptx.count("%globaltimer"),
                "back_edge": _spin_loop_has_back_edge(ptx),
                "sha256": hashlib.sha256(ptx.encode()).hexdigest(),
                "ptx": ptx})
    except Exception as e:  # noqa: BLE001 - any failure is MEASUREMENT ABSENT
        out["variants"] = []
        out["runtime_ptx_error"] = repr(e)
        return out
    out["runtime_ptx_smid_sites"] = min(v["smid_sites"] for v in out["variants"])
    out["runtime_ptx_globaltimer_sites"] = min(
        v["globaltimer_sites"] for v in out["variants"])
    out["runtime_spin_back_edge"] = all(v["back_edge"] for v in out["variants"])
    out["runtime_ptx_sha256"] = out["variants"][0]["sha256"]
    return out


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


def driver_readout(stream_ptrs, lib=None):
    """Returns {'primary_sm': {...}, 'streams': {name: green_ctx_is_null}}.

    `lib` is the CUDA driver handle and defaults to the real `libcuda.so.1`.
    ★2026-08-21 audit item R2: it is injectable ONLY so that this producer can
    be exercised on a CPU node (checks A44/A45). Before that, the runtime-PTX
    read-out had eight direct checks of its producer (E1-E8) and this one had
    zero, so a producer that answered the SAME WAY FOR EVERY STREAM -- either
    way -- was invisible to the whole battery: the audit hardwired "attached"
    and then "not attached" and all 50 checks passed both times. Nothing but
    the self-test ever passes this argument; `run()` calls the one-argument
    form and gets the real driver.

    The failure contract is per field, and no blanket promise is made: a load
    failure returns early with `error`, a per-stream call that raises leaves
    that stream's entry as `error` (no boolean), and the scorer treats every
    one of those as NOT attached (`_greenctx_attached`, `_greenctx_detached`).
    """
    out = {"note": "DESCRIPTIVE / TARGET LAYER ONLY -- never an input to a verdict"}
    if lib is None:
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


def _record_runtime_ptx(rep, kernel, device, outdir, tag):
    """Write the runtime-instrument fields (and PTX files) into `rep`.

    Split out of `run()` on purpose: `--selftest-analyzer` calls THIS function
    with an ahead-of-time compiled kernel injected into the JIT cache, so the
    read-out that the GPU run depends on is exercised on a CPU node instead of
    being trusted.

    EXACT FAILURE CONTRACT (corrected 2026-08-21, audit item C8). The previous
    wording promised without qualification that this function raises nothing
    at all, which is stronger than the code: the transcript write below sits
    outside the try, so an OSError does escape. (That forbidden blanket phrase
    is deliberately NOT reproduced here -- A39 watches for it, and a docstring
    quoting its own watch string would fail itself; lesson #54.) What holds:
      * the READ-OUT does not propagate a failure of the read: every
        Exception inside `runtime_ptx_probe` is caught there (BaseException is
        not -- see its own contract), so a failed read leaves every scalar
        field None plus `runtime_ptx_error`,
        and `score()` then stops with UNDETERMINED (MEASUREMENT ABSENT).
      * the PTX TRANSCRIPT WRITE may raise OSError and is deliberately NOT
        caught. `smid_runtime_<tag>.ptx` is a pre-registered artefact (prereg
        sec11) and a run that cannot write it must fail loudly. It happens
        after every launch, so the failure mode is job exit != 0 = prereg I1 =
        MEASUREMENT ABSENT: never a silent partial artefact, never a verdict.
    """
    probe = runtime_ptx_probe(kernel, device)
    for i, var in enumerate(probe["variants"]):
        name = (f"smid_runtime_{tag}.ptx" if i == 0
                else f"smid_runtime_{tag}_v{i}.ptx")
        with open(os.path.join(outdir, name), "w") as f:
            f.write(var.pop("ptx"))
        var["ptx_path"] = name
    rep["runtime_ptx_variants"] = probe["variants"]
    rep["runtime_ptx_smid_sites"] = probe["runtime_ptx_smid_sites"]
    rep["runtime_ptx_globaltimer_sites"] = probe["runtime_ptx_globaltimer_sites"]
    rep["runtime_spin_back_edge"] = probe["runtime_spin_back_edge"]
    rep["runtime_ptx_sha256"] = probe["runtime_ptx_sha256"]
    if probe.get("runtime_ptx_error"):
        rep["runtime_ptx_error"] = probe["runtime_ptx_error"]
    return rep


def _provenance():
    """Run-identifying facts for the raw artefact (audit item C7, 2026-08-21).

    EXACT FAILURE CONTRACT (corrected 2026-08-21, audit item R12). The first
    version of this docstring re-introduced, verbatim, the unconditional
    no-raise phrase that audit item C8 had just struck from
    `_record_runtime_ptx` -- and A39 did not notice because it inspected that
    one function by name. The phrase is deliberately not reproduced here (A39
    watches for it and a docstring quoting its own watch string would fail
    itself; lesson #54), and A39 now runs over a LIST of functions so the next
    one cannot slip in the same way. What holds:
      * the ENUMERATED failure modes each degrade to an "ERROR ..." string
        rather than propagating: git subprocess failure or timeout, OSError
        from the manifest read or from any `_sha256`, and any Exception from
        the `torch` / `triton` version imports. A provenance gap must be
        VISIBLE in the artefact, not abort a GPU run that has already spent
        its budget.
      * nothing outside those modes is promised: the catches are `except
        Exception` / `except OSError`, so a BaseException (KeyboardInterrupt,
        MemoryError, SystemExit) escapes, and so would an exception raised
        while formatting one of those very strings.
      * no field here is an input to any verdict (`score()` does not read
        `provenance`), so an "ERROR ..." string can never move a verdict -- it
        can only record that the fact is unknown.
    """
    prov = {"note": ("provenance only; no field here is an input to any "
                     "verdict. gate #33: the manifest below covers "
                     f"{MANIFEST_EXPECTED_FILES} files and neither "
                     "pdmux_context.py nor sgl_kernel/spatial.py is one of "
                     "them -- those two are hashed in sha256_unmanifested.")}
    try:
        p = subprocess.run(["git", "-C", REPO_ROOT, "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=60)
        prov["git_head"] = (p.stdout.strip() if p.returncode == 0
                            else f"ERROR rc={p.returncode} {p.stderr.strip()[:200]}")
        d = subprocess.run(["git", "-C", REPO_ROOT, "status", "--porcelain"],
                           capture_output=True, text=True, timeout=120)
        if d.returncode == 0:
            dirty = [ln[3:] for ln in d.stdout.splitlines() if ln.strip()]
            prov["git_dirty"] = bool(dirty)
            prov["git_dirty_paths"] = sorted(dirty)[:50]
        else:
            prov["git_dirty"] = f"ERROR rc={d.returncode}"
    except Exception as e:  # noqa: BLE001 - provenance is never fatal
        prov["git_head"] = f"ERROR {e!r}"
        prov["git_dirty"] = f"ERROR {e!r}"
    prov["repo_root"] = REPO_ROOT
    man = {"path": MANIFEST_PATH, "sha256": _sha256(MANIFEST_PATH),
           "expected_entry_count": MANIFEST_EXPECTED_FILES}
    try:
        with open(MANIFEST_PATH, encoding="utf-8") as f:
            rows = [ln.split(None, 1) for ln in f.read().splitlines() if ln.strip()]
        man["entries"] = [{"sha256": a, "path": b.strip()} for a, b in rows]
        man["entry_count"] = len(rows)
        man["entry_count_matches_sync_script"] = (
            len(rows) == MANIFEST_EXPECTED_FILES)
        man["mtime_utc"] = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime(os.path.getmtime(MANIFEST_PATH)))
    except OSError as e:
        man["error"] = f"ERROR {e!r}"
    prov["runtime_source_manifest"] = man
    for name in ("torch", "triton"):
        try:
            prov[f"{name}_version"] = str(__import__(name).__version__)
        except Exception as e:  # noqa: BLE001
            prov[f"{name}_version"] = f"ERROR {e!r}"
    prov["python_version"] = sys.version.split()[0]
    prov["platform"] = platform.platform()
    prov["self_sha256"] = _sha256(os.path.abspath(__file__))
    return prov


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
           # audit item C7: which source tree, which libraries, which harness.
           "provenance": _provenance(),
           }

    # ---- t0: plain stream in the primary context, BEFORE any green context
    s_plain = torch.cuda.Stream(device=dev)
    rep["S_pre"] = _census_target(census_kernel, s_plain, spin_ns)

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

    # ---- runtime instrument check, AFTER every launch in this job.
    # The CPU self-test compiles ahead-of-time with an explicit signature; the
    # JIT at run time specialises differently (e.g. int arguments), so the PTX
    # that actually ran is NOT guaranteed to be the PTX that was inspected on
    # the login node. Re-check it here, on the objects that ran, and let the
    # scorer refuse to interpret a census whose kernel lost either the probe or
    # the residency loop. Placed last on purpose: it then covers EVERY variant
    # compiled by this job, not only the one the first launch happened to use,
    # and `runtime_ptx_probe` aggregates them conservatively.
    _record_runtime_ptx(rep, census_kernel, dev, outdir, tag)

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


def _target_status(out, target, key):
    """Per-target precondition for reading `key` as a SET: presence, then D1.

    ★2026-08-21 audit item C1. `_saturated` compares the last two grid points,
    and an empty census satisfies that vacuously (nothing grew because nothing
    was ever observed), so D1 answers "has the ladder stopped growing" and
    NEVER answers "did the instrument observe anything at all". Nothing else
    asked the second question either, so `score(_synth([0, 0], dsize=0))` --
    a census in which no block ever reported an id -- returned
    GLOBALLY_CONSISTENT_LABEL off three vacuous truths: the empty set is
    disjoint from itself, it equals the (also empty) device set, and it is
    "saturated". That is a fail-open of exactly the shape the gate #21 branch
    exists to prevent.

    Returns None when `key` may be read as a set, else the verdict dict that
    must be reported for it -- existing vocabulary only, in gate #21 order:
      * nothing observed           -> UNDETERMINED (MEASUREMENT ABSENT)
      * observed but still growing -> UNDETERMINED (COVERAGE NOT SATURATED)

    The floor (one observed id, one minimum hit) is NOT a free parameter and
    prereg sec12 stays at zero new thresholds: 0 is the boundary between "the
    instrument produced data" and "it did not", and any value above 1 would be
    a coverage criterion, which is D1's job and not this check's. `min_hits`
    comes from D2, which the caller has already computed from the census
    itself; the driver's own SM-count self-reports are inadmissible here (they
    are target numbers, and they would happily "confirm" a census that
    observed nothing -- the Stage 0 D108 mirror image, prereg sec6.2).

    ★2026-08-21 audit item R9. That paragraph DECLARED the floor fixed while
    the code wrote it as a bare `1` twice, so the audit could raise both to 2
    and no check noticed: a declaration with no mechanical backing (lessons #9
    and #53). Both bounds now read the pre-registered constant
    `MIN_HITS_REPORTED`, which puts them inside the "changing any of these
    after the run is a protocol violation" block, and two checks hold it there
    -- A48 (a census of ONE id with ONE hit must still score, which fails the
    moment the floor is raised) and A49 (the bound must be the constant, not a
    literal, which fails the moment it is hardwired back).
    """
    # --- BEGIN positivity floor (2026-08-21 C1 repair) --------------------
    ids = (target or {}).get("union") or []
    hits = out["D2"].get(key, {}).get("min_hits")
    hits_ok = (isinstance(hits, int) and not isinstance(hits, bool)
               and hits >= MIN_HITS_REPORTED)
    if len(ids) < MIN_HITS_REPORTED or not hits_ok:
        return {"verdict": V_UNDET,
                "why": (f"census target {key} observed nothing: union_size="
                        f"{len(ids)}, min_hits={hits!r}, floor="
                        f"{MIN_HITS_REPORTED}. An empty census "
                        "satisfies every set predicate vacuously, so no set "
                        "statement about it -- consistency, tiling, "
                        "disjointness or absence -- may be made"),
                "union_size": len(ids), "min_hits": hits}
    # --- END positivity floor ---
    if not out["D1"].get(key, {}).get("saturated"):
        return {"verdict": V_UNSAT,
                "why": (f"coverage not saturated for {key}: "
                        f"{out['D1'].get(key, {}).get('why')}")}
    return None


def _greenctx_attached(raw, stream_keys):
    """Was a green context ACTUALLY attached to these streams? (setup premise)

    ★2026-08-21 audit item C3. `LABEL_CONSISTENT_BUT_PARTITION_NOT_DELIVERED`
    fires on "both green streams reach ~the whole device id set", and exactly
    two different worlds produce that observation:
      (i)  a green context was attached and its SM limit was not delivered --
           a RESULT about what the driver API did on this substrate, and
      (ii) no green context was ever attached to the stream the census ran on
           -- an INSTRUMENT FAILURE, which prereg sec3.3/sec5.2 send to
           `UNDETERMINED (MEASUREMENT ABSENT)`, never to a substantive verdict.
    Collapsing those two is the exact shape of methodology gate #21, and the
    read-out that separates them (`cuStreamGetGreenCtx`, per stream) is
    ALREADY collected by `run()` -- so this premise costs no new measurement.

    THIS IS NOT A TARGET-LAYER READ (prereg sec6.2). What that section
    excludes is the driver's own SM COUNTS: checking a `%smid` census against
    them would be the Stage 0 D108 mirror image, and the resource struct
    carries no id set at all. A null-vs-non-null context handle is not a count
    of anything -- it is the premise under which the census was taken, of the
    same kind as "the kernel that ran was the probe". The numeric family stays
    banned mechanically (`_TARGET_LAYER_TOKENS`) and this one boolean is the
    entire whitelist (`_ALLOWED_DRIVER_KEYS`).

    Fail-closed, like the two instrument guards in `score()`: an absent block,
    an absent entry, an error entry or a non-boolean all count as NOT attached
    -- an ABSENT check stops exactly like a FAILED one.
    """
    dr = raw.get("driver_readout") if isinstance(raw, dict) else None
    streams = dr.get("streams") if isinstance(dr, dict) else None
    observed, blocked = {}, []
    for k in stream_keys:
        entry = streams.get(k) if isinstance(streams, dict) else None
        val = entry.get("green_ctx_is_null") if isinstance(entry, dict) else None
        observed[k] = val
        if val is not False:          # False == a green context IS attached
            blocked.append(k)
    why = ""
    if blocked:
        vals = {k: observed[k] for k in blocked}
        why = (f"the green-context attachment premise fails for {blocked}: "
               f"observed {vals} -- true means no "
               "green context, null/None means the read-out is absent or "
               "unusable and fails closed the same way. A census taken on a "
               "stream with no green context attached measures the primary "
               "context, so an overlap of the two streams here would be an "
               "instrument condition, not a statement about what the driver "
               "API delivered")
    return {"observed": observed, "blocked": blocked, "why": why}


def _greenctx_detached(raw, stream_keys):
    """The other direction of the same premise: can the read-out say "no"?

    ★2026-08-21 audit item R2. `_greenctx_attached` is one-directional -- it
    can only ever observe "attached" -- so a read-out that answers "attached"
    for EVERY stream satisfies it, and then the NOT_DELIVERED branch fires
    while the artefact testifies that the premise was checked and held. That
    is gate #42 in its original shape, and it is exactly the defect audit item
    C3 thought it had closed: the audit hardwired the producer to "attached"
    for every stream, and then to "not attached" for every stream, and all 50
    checks passed both times, because nothing ever compared the read-out
    against a stream whose true answer is known.

    A stream whose true answer IS known is already in the artefact, five times
    over and at no extra cost: `run()` censuses a plain pair built BEFORE any
    green context exists, a plain pair built after, and the pre-green plain
    stream itself, and it records the same per-stream read-out for all of
    them. They are the negative control. At least one of them must come back
    as "no green context", or the instrument has not been shown to
    discriminate at all -- and an instrument that cannot say "no" may not be
    quoted saying "yes".

    Note the ASYMMETRY, which is deliberate: the positive side demands that
    BOTH green streams are attached (a census on a stream without one is not
    the measurement at all), while this side demands only that ONE control
    reports detached (one is enough to show the read-out is not stuck). A
    control that comes back attached is not by itself a fault -- it is a fact
    about the read-out, and it is recorded in `setup` for the report.

    Fail-closed like its twin: an absent block, an absent entry, an error
    entry, or anything that is not the boolean True counts as "did not report
    detached".
    """
    dr = raw.get("driver_readout") if isinstance(raw, dict) else None
    streams = dr.get("streams") if isinstance(dr, dict) else None
    observed, detached = {}, []
    for k in stream_keys:
        entry = streams.get(k) if isinstance(streams, dict) else None
        val = entry.get("green_ctx_is_null") if isinstance(entry, dict) else None
        observed[k] = val
        if val is True:               # True == NO green context on that stream
            detached.append(k)
    why = ""
    if not detached:
        why = (f"the attachment read-out never reports a detached stream: "
               f"observed {observed} for streams that run() builds WITHOUT a "
               "green context. Either the read-out is absent, or it answers "
               "the same way for every stream, so its answer for the green "
               "pair carries no information and the premise under which this "
               "census was taken is unverified. A read-out that cannot report "
               "'no green context' cannot be cited reporting 'green context "
               "attached'")
    return {"observed": observed, "detached": detached, "why": why}


def score(raw):
    """Pure function: raw artefact dict -> verdict dict. No I/O, no globals."""
    out = {"kind": "smid_l0_verdict",
           "prereg": "PREREG_SMID_R0_2026-08-14.md",
           "R0": None, "R1": None, "R2": None, "R5": None, "D1": {}, "D2": {},
           # setup premises that are not decision quantities but without which
           # no decision quantity means anything (2026-08-21 audit item C3)
           "setup": {}}

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
    # --- BEGIN probe-present guard (fail-closed, 2026-08-21 repair) ---------
    # Pre-repair this read `== 0`, which a MISSING field silently passed
    # (`None == 0` is false). Combined with the dead `.cache` read in `run()`,
    # which always wrote None, that meant an UNVERIFIED kernel scored as a
    # verified one. Same verdict vocabulary, same stop rule (prereg sec3.3);
    # what changed is only that ABSENCE of the check now also stops.
    rt_sites = raw.get("runtime_ptx_smid_sites")
    if isinstance(rt_sites, bool) or not isinstance(rt_sites, int):
        out["R0"] = {"verdict": V_UNDET,
                     "why": "the runtime instrument check itself is absent or "
                            f"unusable (runtime_ptx_smid_sites={rt_sites!r}); "
                            "the kernel that ran was never verified to be the "
                            "probe, so nothing in this artefact may be read",
                     "runtime_ptx_error": raw.get("runtime_ptx_error")}
        out["stop"] = True
        return out
    if rt_sites < 1:
        out["R0"] = {"verdict": V_UNDET,
                     "why": "runtime PTX contained no %smid site -- the kernel "
                            "that ran is not the probe"}
        out["stop"] = True
        return out
    # --- END probe-present guard ---
    # --- BEGIN residency guard (fail-closed, 2026-08-21 repair) -------------
    # Pre-repair this read `is False`, which a MISSING field silently passed.
    rt_edge = raw.get("runtime_spin_back_edge")
    if rt_edge is not True and rt_edge is not False:
        out["R0"] = {"verdict": V_UNDET,
                     "why": "the residency check itself is absent or unusable "
                            f"(runtime_spin_back_edge={rt_edge!r}); it is "
                            "unknown whether the kernel that ran stayed "
                            "resident, so every union is a bound of unknown "
                            "slack",
                     "runtime_ptx_error": raw.get("runtime_ptx_error")}
        out["stop"] = True
        return out
    if rt_edge is False:
        out["R0"] = {"verdict": V_UNDET,
                     "why": "runtime PTX has no clock read inside a loop with a "
                            "back edge -- residency was not achieved, so every "
                            "union is a lower bound of unknown slack"}
        out["stop"] = True
        return out
    # --- END residency guard ---

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

    # --- BEGIN green-context attachment premise (2026-08-21 C3 repair) -----
    # See `_greenctx_attached`. A SETUP premise, of the same kind as "the
    # kernel that ran was the probe" -- not a target-layer number. Without it
    # the NOT_DELIVERED branch below reports a stream that never had a green
    # context (a probe failure) as a substantive outcome.
    att = _greenctx_attached(raw, (green_p, green_d))
    out["setup"] = {
        "green_ctx_attached": att["observed"],
        "meaning": "false == a green context IS attached to that stream",
        "source": "recorded per stream by run() at t7, driver read-out block",
        "status": ("premise holds for the green pair" if not att["blocked"]
                   else "premise fails")}
    if att["blocked"]:
        out["R0"] = {"verdict": V_UNDET, "why": att["why"]}
        out["stop"] = True
        return out
    # --- END green-context attachment premise ---

    # --- BEGIN plain-control detachment premise (2026-08-21 audit item R2) --
    # See `_greenctx_detached`. The premise above says "these two streams are
    # attached"; on its own that is unfalsifiable, because a read-out stuck on
    # "attached" produces the identical observation. The five streams below
    # are plain BY CONSTRUCTION in run(), so they are the negative control the
    # artefact already carried and nothing read. Same fail-closed treatment.
    ctl_keys = ["idx0_prefill", "idx0_decode",
                f"idx{idx_last}_prefill", f"idx{idx_last}_decode", "plain_pre"]
    ctl = _greenctx_detached(raw, ctl_keys)
    out["setup"]["plain_control_detached"] = ctl["observed"]
    out["setup"]["control_meaning"] = (
        "true == NO green context on that stream. run() builds these five "
        "without one, so they are the negative control for the read-out that "
        "the premise above depends on")
    out["setup"]["control_status"] = (
        "premise holds: the read-out reports at least one detached stream"
        if ctl["detached"] else
        "premise fails: the read-out never reports a detached stream")
    if not ctl["detached"]:
        out["R0"] = {"verdict": V_UNDET, "why": ctl["why"]}
        out["stop"] = True
        return out
    # --- END plain-control detachment premise ---

    # Presence FIRST, then D1 saturation. Both are hard preconditions for
    # every set statement (prereg sec4.1 and sec5.1); before 2026-08-21 only
    # the second was checked, and it passes vacuously on an empty census
    # (audit item C1 -- see `_target_status`).
    for k in (green_p, green_d, "S_pre"):
        st = _target_status(out, tgt(k), k)
        if st is not None:
            out["R0"] = st
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
                     "SCOPE (corrected 2026-08-21, audit item C3): this is a "
                     "result about what the DRIVER API delivered on THIS "
                     "substrate inside THIS process -- no engine is running "
                     "here, no model is loaded, no request executes (prereg "
                     "sec0.2-5, sec8-1), so it may not be attached to any "
                     "serving job, past or future. It is not a probe failure "
                     "either: the attachment premise above was checked and "
                     "held, which is the observation that separates the two "
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
        if kp not in ps or kd not in ps:
            continue
        tgt_i = list(raw["divisions_from_divide_sm"][i - 1])
        # --- BEGIN per-group gate (2026-08-21 C2/C3 repair) ---------------
        # Same defect as S_post, same fix. `disjoint` is an absence claim and
        # `realised` is a set-cardinality claim, so neither may be produced
        # for a group whose census observed nothing, whose ladder is still
        # growing, or whose streams never had a green context. Pre-repair this
        # reported the sizes and set `"saturated": [False, ...]` beside them,
        # which prereg sec4.1 does not permit. The gate is per ITEM: idx2
        # being unreadable says nothing about idx1 and must not stop the run.
        att_i = _greenctx_attached(raw, (kp, kd))
        st_i = _target_status(out, ps[kp], kp) or _target_status(out, ps[kd], kd)
        if att_i["blocked"] or st_i is not None:
            others[f"idx{i}"] = {
                "verdict": V_UNDET if att_i["blocked"] else st_i["verdict"],
                "why": att_i["why"] if att_i["blocked"] else st_i.get("why"),
                "target": tgt_i}
            continue
        # --- END per-group gate ---
        a, b = set(ps[kp]["union"]), set(ps[kd]["union"])
        others[f"idx{i}"] = {"realised": [len(a), len(b)], "target": tgt_i,
                             "disjoint": len(a & b) == 0}
    out["R1"]["other_green_groups"] = others

    # ---------------- R2: primary-context carve --------------------------
    # --- BEGIN S_post target gate (2026-08-21 C2 repair) ------------------
    # `S_post` is a census target of its own and it was NOT in the hard
    # precondition loop above (which reads the green pair and S_pre only), so
    # pre-repair an unsaturated -- or empty -- S_post still produced
    # `lost_ids`/`lost_count`. That quantity is by construction an ABSENCE
    # claim ("these ids were not reachable after"), and prereg sec4.1 forbids
    # absence claims about a target that is not saturated, outright. So the
    # gate is PER TARGET and the forbidden quantity is not computed at all: a
    # boolean flag standing beside a published `lost_count` is not what
    # sec4.1 asks for.
    replicates = {}
    for k in ("idx0_prefill", "idx0_decode",
              f"idx{idx_last}_prefill", f"idx{idx_last}_decode"):
        if k not in ps:
            continue
        st_k = _target_status(out, ps[k], k)
        replicates[k] = (len(set(ps[k]["union"])) if st_k is None
                         else st_k["verdict"])
    r2 = {
        "S_pre_size": len(D),
        "not_independent": ("idx0/idx3 plain-stream censuses below are the SAME "
                            "empirical content as this difference, not a second "
                            "independent test (methodology gate #9, sixth "
                            "recurrence: two gates named independent measuring "
                            "one quantity). They are descriptive replicates."),
        "descriptive_replicates": replicates,
    }
    post_status = _target_status(out, raw.get("S_post"), "S_post")
    if post_status is not None:
        r2.update(post_status)
        r2["forbidden"] = ("lost_ids / lost_count are NOT produced for an "
                           "unreadable S_post: the difference is an absence "
                           "claim and prereg sec4.1 forbids absence claims "
                           "about a target that is empty or unsaturated")
    else:
        Spost = set(raw["S_post"]["union"])
        lost = sorted(D - Spost)
        r2.update({
            "S_post_size": len(Spost), "lost_ids": lost, "lost_count": len(lost),
            "saturated_post": True,
            "reading": ("carve observed: ids reachable before green-context "
                        "creation and not after" if lost else
                        "no carve observed at this saturation level")})
    out["R2"] = r2
    # --- END S_post target gate ---

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
# The streams `run()` builds WITHOUT a green context: the plain pair before
# the green ones, the plain pair after, and the pre-green plain stream. They
# are the negative control of the attachment read-out (audit item R2). Named
# here, in the harness, on purpose: `score()` derives the same set from
# `divisions_from_divide_sm` instead of importing this tuple, so the harness
# and the scorer agree by measurement and not by construction.
_PLAIN_CONTROL_KEYS = ("idx0_prefill", "idx0_decode", "idx3_prefill",
                       "idx3_decode", "plain_pre")


def _synth(sizes, disjoint=True, dsize=108, sat=True, post_lost=0, target=None,
           unsat=(), empty=(), min_hits=3, min_hits_by=None, not_attached=(),
           attached=()):
    """Synthetic raw artefact for the analyzer battery.

    ★2026-08-21: PER-TARGET handles added. The 2026-08-14 generator had one
    `sat` knob applied to every target at once, so two shapes simply did not
    exist in the input space -- "only some targets unsaturated" and "every
    target empty" -- and the two fail-open defects that live in those shapes
    (audit items C1 and C2) could not be written down as a test at all. That
    is a coverage hole in the harness, not in the scorer, and it is the reason
    both defects survived the 2026-08-21 fail-open repair.

    Old signature and old meanings are unchanged (`sat=False` still means
    "every target unsaturated"). The new keywords address ONE census target or
    ONE stream each:
      `unsat=`        iterable of target keys whose ladder is still growing
      `empty=`        iterable of target keys that observed no id at all
      `min_hits_by=`  {target key: min_hits}, e.g. 0 for "observed nothing"
      `not_attached=` iterable of stream keys with no green context attached
      `attached=`     iterable of stream keys that report a green context even
                      though `run()` builds them plain -- the DEGENERATE
                      read-out world of audit item R2 (2026-08-21), which the
                      2026-08-21 harness could not express at all, which is
                      why an instrument stuck on "attached" passed all 50
                      checks. Applied after `not_attached`, so it wins.
    """
    a = list(range(sizes[0]))
    b = (list(range(sizes[0], sizes[0] + sizes[1])) if disjoint
         else list(range(sizes[1])))
    unsat, empty = set(unsat), set(empty)
    mhb = dict(min_hits_by or {})

    def tar(key, u):
        u = [] if key in empty else list(u)
        lad = [{"n_blocks": n, "union": u, "nsmid_observed": [dsize]}
               for n in GRID_SWEEP]
        if (not sat) or key in unsat:
            lad[-1] = {"n_blocks": GRID_SWEEP[-1], "union": u + [10_000],
                       "nsmid_observed": [dsize]}
        mh = mhb.get(key, min_hits)
        return {"ladder": lad, "union": u, "min_hits": mh,
                "hits": {str(x): mh for x in u}}
    D = list(range(dsize))
    per_stream = {"idx1_prefill": tar("idx1_prefill", a),
                  "idx1_decode": tar("idx1_decode", b),
                  "idx2_prefill": tar("idx2_prefill", list(range(54))),
                  "idx2_decode": tar("idx2_decode", list(range(54, 108))),
                  "idx0_prefill": tar("idx0_prefill", D),
                  "idx0_decode": tar("idx0_decode", D),
                  "idx3_prefill": tar("idx3_prefill", D),
                  "idx3_decode": tar("idx3_decode", D)}
    # `run()` builds [plain pair, green pair per division, plain pair], so
    # idx0/idx3 are PLAIN by construction: their true read-out is "no green
    # context", and the gates must not demand one of them.
    not_attached = ((set(not_attached) | set(_PLAIN_CONTROL_KEYS))
                    - set(attached))
    streams = {k: {"rc": 0, "green_ctx_is_null": k in not_attached}
               for k in list(per_stream) + ["plain_pre"]}
    return {
        "divisions_from_divide_sm": [list(target or sizes), [54, 54]],
        "runtime_ptx_smid_sites": 1, "runtime_spin_back_edge": True,
        "S_pre": tar("S_pre", D),
        "S_post": tar("S_post", D[:dsize - post_lost]),
        "per_stream": per_stream,
        "driver_readout": {"streams": streams},
        "concurrent_idx1": [],
    }


# The target-layer NUMBERS. `score()` may not read any of these, and may not
# reason with them in a comment either (prereg sec6.2: checking a `%smid`
# census against the driver's own SM counts is the Stage 0 D108 mirror image,
# and the resource struct carries no id set, cuda.h:24770-24777).
#
# ★2026-08-21 (audit item C3): this tuple also contained `driver_readout`,
# `cuCtxGetDevResource` and `cuGreenCtx`, i.e. it banned the WHOLE driver
# block -- including the one boolean that says whether a green context was
# attached to the stream at all. That boolean is not a target number; it is
# the setup premise of the census, the same kind of fact as "the kernel that
# ran was the probe", and banning it is what let the NOT_DELIVERED outcome
# merge an instrument failure with a driver-API result. So the ban is now the
# NUMERIC family only, and the rest of the driver block is governed by an
# explicit whitelist: `_DRIVER_KEY_SURFACE` is the surface of that block and
# `_ALLOWED_DRIVER_KEYS` the part `score()` may touch, so a newly added read
# of any other one FAILS the self-test.
#
# ★2026-08-21 (audit item R7). The sentence above used to claim the surface
# "enumerates every key `driver_readout()` can emit", and it did not: `rc`,
# `error` and `note` were missing, all three emitted by that function, so the
# audit could make `score()` read `streams[...]["rc"]` and no check failed.
# Two changes, because the false claim had two halves:
#   (i) the surface is now complete for the EMITTED half and A47 proves it
#       mechanically -- it re-derives the emitted keys from the producer's own
#       AST (`_driver_emitted_keys`) and demands the surface contain them, so
#       adding a key to `driver_readout()` without listing it here fails the
#       self-test (mutant `driver_emits_unlisted_key`).
#   (ii) the surface is NOT only emitted keys: `driver_readout`, `cuGreenCtx`,
#       `cuCtxGetDevResource` and `sm_triplet` are the block's name and the
#       driver API/attribute names, kept here so a read of them is caught too.
#       So the relation is surface >= emitted, never equality.
# The ban had to get more precise to hold the three generic names: it now
# ignores strings in ONE position -- a dict-literal key, i.e. a name `score()`
# writes into its own output -- because `note` is also an ordinary output key
# of its R5 block. Every other appearance of a surface name still fails the
# self-test. A47 plus that ban is what makes the claim above true, and A46 is
# the positive control that the ban resolves anything at all.
#
# ★DOC OWNER: prereg sec6.2 lists the pre-2026-08-21 eight-token ban verbatim
# and must be updated to this narrowed ban plus the whitelist. The PURPOSE
# that section states (no verification of a census by a target number) is
# unchanged, and it is now enforced TRANSITIVELY, which the old check was not.
# The same section needs a SECOND edit from the re-audit (item R7): the
# surface gained `rc`, `error` and `note`, and the ban stopped counting
# dict-literal keys. That edit is NOT made here -- this file does not own the
# pre-registration, and the R0 verdict is unmade either way.
_TARGET_LAYER_TOKENS = ("smCount", "minSmPartitionSize",
                        "smCoscheduledAlignment", "total_sm_reported",
                        "nsmid_observed")
_DRIVER_KEY_SURFACE = ("driver_readout", "streams", "green_ctx_is_null",
                       "primary_sm", "primary_error", "cuCtxGetCurrent_rc",
                       "cuCtxGetDevResource_rc", "cuCtxGetDevResource",
                       "cuGreenCtx", "smCount", "minSmPartitionSize",
                       "smCoscheduledAlignment", "sm_triplet",
                       # 2026-08-21 audit item R7: emitted by driver_readout()
                       # and missing until then -- the hole the audit walked
                       # through with a read of `rc`.
                       "rc", "error", "note")
_ALLOWED_DRIVER_KEYS = ("driver_readout", "streams", "green_ctx_is_null")


def _score_excludes_target_layer(src=None):
    """Gate #9 guard, enforced mechanically rather than promised in prose.

    Walks `score()` AND every module-level function `score()` can reach.
    ★2026-08-21: the transitive closure is new. The old check inspected
    `score()`'s own body only, so any read of a target number could be
    laundered through a one-line helper -- and this repair adds two helpers,
    which would have made the guard weaker exactly when it got more to guard.
    `--selftest-mutants` puts a target read in a HELPER and demands the check
    still fails.

    Returns two independent findings (empty lists == clean):
      `target_numbers` -- a banned numeric token appears in the reachable
          source TEXT. Text, so comments count too: nobody should be reasoning
          with those numbers here, not just reading them.
      `driver_keys`    -- a name from the driver read-out surface that is NOT
          on the setup whitelist is actually referenced in the reachable AST.
          Exact names, so generic keys cannot produce substring false
          positives.

    ★2026-08-21 (audit item R7) the second finding narrowed what counts as a
    reference, by ONE position. It used to be every string constant in the
    reachable source, and that is why the surface had to stay incomplete:
    `note` is an emitted driver key AND an ordinary output key of `score()`'s
    own R5 block, so listing it would have failed the intact file, and leaving
    it off is what let the audit read `rc` with nothing failing. A dict
    literal's key is a name this function WRITES, not one it reads off the
    driver block, so that single position no longer counts. Everything else
    still does -- attribute, identifier, subscript, call argument, even a
    comparison against the string -- so the ban is not weaker anywhere a read
    could hide, and it is now complete over the producer's emitted keys (A47).
    A46 is the positive control that the resolution actually happens: a guard
    that resolved nothing would return an empty list and pass vacuously, which
    is the identity shape of lessons #9 and #53.
    """
    import ast
    if src is None:
        src = open(os.path.abspath(__file__), encoding="utf-8").read()
    tree = ast.parse(src)
    defs = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    seen, todo, segments = set(), ["score"], []
    while todo:
        name = todo.pop()
        if name in seen or name not in defs:
            continue
        seen.add(name)
        segments.append(ast.get_source_segment(src, defs[name]) or "")
        for sub in ast.walk(defs[name]):
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name):
                todo.append(sub.func.id)
    text = "\n".join(segments)
    names = set()
    for name in seen:
        # A dict-literal key is a name this scorer WRITES into its own output,
        # not a name it reads off the driver block. Excluding exactly that one
        # position -- and nothing else -- is what lets the three generic keys
        # (`rc`, `error`, `note`) be banned at all: `note` is also an ordinary
        # output key of `score()`'s R5 block. Every other appearance still
        # counts, including a comparison against the string, so the ban did
        # not get weaker anywhere a leak could hide.
        written = set()
        for sub in ast.walk(defs[name]):
            if isinstance(sub, ast.Dict):
                written |= {id(k) for k in sub.keys
                            if isinstance(k, ast.Constant)}
        for sub in ast.walk(defs[name]):
            if isinstance(sub, ast.Attribute):
                names.add(sub.attr)
            elif isinstance(sub, ast.Name):
                names.add(sub.id)
            elif (isinstance(sub, ast.Constant) and isinstance(sub.value, str)
                  and id(sub) not in written):
                names.add(sub.value)
    return {"reachable": sorted(seen),
            "target_numbers": [t for t in _TARGET_LAYER_TOKENS if t in text],
            "driver_keys": sorted(set(_DRIVER_KEY_SURFACE) & names
                                  - set(_ALLOWED_DRIVER_KEYS))}


# A source snippet with a KNOWN leak, used as the positive control for the
# guard above (check A46). If the guard ever stops resolving a chained read,
# `driver_keys` goes empty and A23 starts passing for the wrong reason; this
# fixture is what makes that visible. It is deliberately NOT a copy of any
# real code path -- it is the shape the audit's own mutant used.
_GUARD_PROBE_SRC = (
    "def score(raw):\n"
    '    return raw["driver_readout"]["streams"]["s"].get("rc")\n')


def _driver_emitted_keys(src=None):
    """Every artefact key the driver read-out producer can emit (audit R7).

    Re-derived from the AST of `driver_readout` and of the struct helper it
    calls, NOT from a hand-written list: the point of A47 is to catch a key
    that someone adds to the producer and forgets to declare, and a check that
    reads a second hand-written list would only compare two lists (lesson #9).
    Three emission shapes exist in that function and all three are collected:
    a dict literal, a subscript store with a constant key, and `dict(k=...)`.
    """
    import ast
    if src is None:
        src = open(os.path.abspath(__file__), encoding="utf-8").read()
    tree = ast.parse(src)
    producers = ("driver_readout", "sm_triplet")
    keys = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef) and node.name in producers):
            continue
        for sub in ast.walk(node):
            if (isinstance(sub, ast.Subscript) and isinstance(sub.ctx, ast.Store)
                    and isinstance(sub.slice, ast.Constant)
                    and isinstance(sub.slice.value, str)):
                keys.add(sub.slice.value)
            elif isinstance(sub, ast.Dict):
                keys.update(k.value for k in sub.keys
                            if isinstance(k, ast.Constant)
                            and isinstance(k.value, str))
            elif (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)
                  and sub.func.id == "dict"):
                keys.update(kw.arg for kw in sub.keywords if kw.arg)
    return sorted(keys)


def _blanket_noraise_docstrings(src=None):
    """Docstrings in `src` that carry the unconditional no-raise promise.

    Audit item R12: A39 used to watch ONE function by name, so the phrase
    audit item C8 struck from `_record_runtime_ptx` simply reappeared in a
    docstring written the same day. The named list `_NO_BLANKET_NORAISE_DOCS`
    stays -- it says which contracts are meant to be exact -- but a list only
    ever covers what someone remembered to add to it, so this walks EVERY
    docstring in the file instead, module included. The watched phrase is
    assembled rather than written out, or this function would report itself
    (lesson #54).
    """
    import ast
    if src is None:
        src = open(os.path.abspath(__file__), encoding="utf-8").read()
    phrase = "never " + "raises"
    tree = ast.parse(src)
    nodes = [tree] + [n for n in ast.walk(tree)
                      if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef,
                                        ast.ClassDef))]
    return sorted(getattr(n, "name", "<module>") for n in nodes
                  if phrase in (ast.get_docstring(n) or "").lower())


def _floor_constant_uses(src=None):
    """How many times the presence floor READS the pre-registered constant.

    Audit item R9: the floor was written as a bare literal twice, so the audit
    raised it and nothing failed. A48 catches a raised floor by measuring
    behaviour; this measures the WIRING, so hardwiring the same value back --
    a change with no behavioural signature at all -- is caught too (mutant
    `hardwire_positivity_floor`).
    """
    import ast
    if src is None:
        src = open(os.path.abspath(__file__), encoding="utf-8").read()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_target_status":
            return sum(1 for sub in ast.walk(node)
                       if isinstance(sub, ast.Name)
                       and sub.id == "MIN_HITS_" + "REPORTED")
    return 0


class _FakeCuDriver:
    """A stand-in for `libcuda.so.1` so `driver_readout()` is testable (R2).

    The runtime-PTX read-out is exercised end to end on a CPU node (E1-E8, via
    ahead-of-time compiled kernels injected into the JIT cache). The driver
    read-out had NO such check, and that is the hole the re-audit walked
    through: it hardwired the producer to answer "attached" for every stream,
    then "not attached" for every stream, and both degenerate instruments
    passed all 50 checks, because no check ever handed the producer a stream
    whose true answer was known and compared.

    This class is that known-answer input. It implements only the three driver
    entry points `driver_readout` calls, returns success for all of them, and
    hands back a non-null green-context handle for exactly the stream pointers
    it was constructed with. `ctypes.cast(ref, POINTER(c_void_p))[0] = ...` is
    the public-API way to write through the `byref` out-parameter (the private
    `._obj` back door is deliberately avoided).

    It is a TEST double and nothing else: `run()` never passes `lib`, so the
    GPU path is unchanged and no verdict can ever be produced from this class.
    """

    def __init__(self, green_ptrs=()):
        self.green_ptrs = set(green_ptrs)

    def cuCtxGetCurrent(self, ref):
        return 0

    def cuCtxGetDevResource(self, ctx, ref, kind):
        return 0                       # struct left zeroed; descriptive only

    def cuStreamGetGreenCtx(self, ptr, ref):
        handle = 0x9000 if ptr.value in self.green_ptrs else 0
        ctypes.cast(ref, ctypes.POINTER(ctypes.c_void_p))[0] = handle
        return 0


# Functions whose docstring may not carry the unconditional no-raise promise
# that audit item C8 struck once and that came back in a new docstring the
# same day (audit item R12). A39 walks this LIST instead of the single name it
# used to watch -- and, because a list only covers what someone remembered to
# put on it, ALSO every docstring in the file (`_blanket_noraise_docstrings`).
# The list is the declaration of which contracts must be exact; the sweep is
# what makes the declaration hard to slip past.
_NO_BLANKET_NORAISE_DOCS = ("_record_runtime_ptx", "_provenance",
                            "runtime_ptx_probe", "driver_readout")


def _drop(d, *keys):
    d = dict(d)
    for k in keys:
        d.pop(k, None)
    return d


# --------------------------------------------------------------------------
# The assertion battery is a FUNCTION OF THE SCORER, not a script, so that the
# identical battery can be re-run against a mutated copy of this file
# (methodology lesson #53 / gate #50: a check that cannot fail is an identity,
# not evidence). Every entry returns (name, status, detail) with status in
# {"ok", "fail", "raised"}. "raised" is never acceptable, in the intact file
# or in a mutant: a check that dies of an unrelated exception has not tested
# anything.
# --------------------------------------------------------------------------
def _analyzer_checks(score_fn, src=None, ns=None):
    """The battery. `src`/`ns` are the SOURCE and NAMESPACE under test.

    They default to this file, and `--selftest-mutants` passes the mutant's
    instead, so the source-level checks (A0/A23, the AST guard) and the
    non-scorer repairs (A38 provenance, A39 read-out contract, A40/A41 the
    mutant predicate itself) are exercised against the mutant too. The check
    BODIES always come from the intact file: a mutant may not weaken its own
    examiner.
    """
    res = []
    ns = ns or globals()

    def ck(name, fn):
        try:
            ok = bool(fn())
        except Exception as e:  # noqa: BLE001
            res.append((name, "raised", repr(e)))
            return
        res.append((name, "ok" if ok else "fail", ""))

    healthy = _synth((74, 34))
    leaks = _score_excludes_target_layer(src)

    # -- gate #9 / prereg sec5.3-3: the scorer may not be verified against a
    #    target number, and (2026-08-21) may not launder one through a helper.
    ck("A0 score() reads no target-layer number, transitively",
       lambda: not leaks["target_numbers"])
    ck("A23 score() touches no driver read-out key outside the setup whitelist",
       lambda: not leaks["driver_keys"])

    # -- gate #21 branch: absence of measurement is never a substantive verdict
    ck("A1 empty raw -> MEASUREMENT ABSENT",
       lambda: score_fn({})["R0"]["verdict"] == V_UNDET)
    ck("A2 missing per_stream -> MEASUREMENT ABSENT",
       lambda: score_fn({"S_pre": {"union": [1], "ladder": []}}
                        )["R0"]["verdict"] == V_UNDET)
    ck("A3 runtime PTX with 0 %smid sites -> MEASUREMENT ABSENT",
       lambda: score_fn(dict(healthy, runtime_ptx_smid_sites=0)
                        )["R0"]["verdict"] == V_UNDET)
    ck("A4 runtime spin hoisted (False) -> MEASUREMENT ABSENT",
       lambda: score_fn(dict(healthy, runtime_spin_back_edge=False)
                        )["R0"]["verdict"] == V_UNDET)

    # -- FAIL-CLOSED (2026-08-21 repair). These are the repair itself: an
    #    ABSENT instrument check must stop exactly like a FAILED one.
    ck("A5 probe field ABSENT -> MEASUREMENT ABSENT (fail-closed)",
       lambda: score_fn(_drop(healthy, "runtime_ptx_smid_sites")
                        )["R0"]["verdict"] == V_UNDET)
    ck("A6 probe field None -> MEASUREMENT ABSENT (fail-closed)",
       lambda: score_fn(dict(healthy, runtime_ptx_smid_sites=None)
                        )["R0"]["verdict"] == V_UNDET)
    ck("A7 probe field non-numeric -> MEASUREMENT ABSENT (fail-closed)",
       lambda: score_fn(dict(healthy, runtime_ptx_smid_sites="ERROR")
                        )["R0"]["verdict"] == V_UNDET)
    ck("A8 residency field ABSENT -> MEASUREMENT ABSENT (fail-closed)",
       lambda: score_fn(_drop(healthy, "runtime_spin_back_edge")
                        )["R0"]["verdict"] == V_UNDET)
    ck("A9 residency field None -> MEASUREMENT ABSENT (fail-closed)",
       lambda: score_fn(dict(healthy, runtime_spin_back_edge=None)
                        )["R0"]["verdict"] == V_UNDET)
    ck("A10 residency field non-boolean -> MEASUREMENT ABSENT (fail-closed)",
       lambda: score_fn(dict(healthy, runtime_spin_back_edge="yes")
                        )["R0"]["verdict"] == V_UNDET)
    ck("A11 every fail-closed stop also sets stop=True",
       lambda: all(score_fn(r).get("stop") is True for r in (
           _drop(healthy, "runtime_ptx_smid_sites"),
           _drop(healthy, "runtime_spin_back_edge"),
           dict(healthy, runtime_ptx_smid_sites=None),
           dict(healthy, runtime_spin_back_edge=None))))

    # -- the mirror-image error (gate #21 in reverse, already seen once in this
    #    repo): a HEALTHY artefact must NOT be swallowed by the new guards.
    #    If the repair over-triggers, A12-A22 catch it.
    ck("A12 unsaturated -> COVERAGE NOT SATURATED (not substantive)",
       lambda: score_fn(_synth((74, 34), sat=False))["R0"]["verdict"] == V_UNSAT)
    ck("A13 disjoint + tiling -> GLOBALLY_CONSISTENT_LABEL",
       lambda: score_fn(healthy)["R0"]["verdict"] == V_CONSISTENT)
    ck("A14 R1 exact match reported",
       lambda: score_fn(healthy)["R1"]["exact_match"] is True)
    ck("A15 R2 no carve when S_post == S_pre",
       lambda: score_fn(healthy)["R2"]["lost_count"] == 0)
    # realised 76/32 while divide_sm targeted 74/34: granularity/rounding.
    # Pre-registered as a RESULT (report as observed), never a probe failure.
    ck("A16 rounded partition still GLOBALLY_CONSISTENT_LABEL (result, not failure)",
       lambda: (score_fn(_synth((76, 32), target=(74, 34)))["R0"]["verdict"]
                == V_CONSISTENT
                and score_fn(_synth((76, 32), target=(74, 34)))["R1"]["exact_match"]
                is False))
    ck("A17 rounded partition reading says RESULT not failure",
       lambda: "not a measurement failure" in score_fn(
           _synth((76, 32), target=(74, 34)))["R1"]["reading"])
    ck("A18 both streams reach whole device -> PARTITION NOT DELIVERED",
       lambda: score_fn(_synth((108, 108), disjoint=False)
                        )["R0"]["verdict"] == V_NO_DELIVERY)
    ck("A19 overlap with small union -> NOT_A_GLOBALLY_CONSISTENT_LABEL",
       lambda: score_fn(_synth((74, 34), disjoint=False)
                        )["R0"]["verdict"] == V_NOT_CONSISTENT)
    ck("A20 R1/R2/R5 uninterpretable when R0 fails",
       lambda: "UNINTERPRETABLE" in score_fn(
           _synth((74, 34), disjoint=False))["R1"]["verdict"])
    ck("A21 carve detected as 4 lost ids",
       lambda: score_fn(_synth((74, 34), post_lost=4))["R2"]["lost_count"] == 4)
    ck("A22 R2 declares idx0/idx3 non-independent",
       lambda: "not_independent" in score_fn(_synth((74, 34), post_lost=4))["R2"])

    # ---- C1: an EMPTY census is an absence of measurement, not a verdict.
    #      The pre-repair scorer returned GLOBALLY_CONSISTENT_LABEL here.
    void = _synth([0, 0], dsize=0)
    ck("A24 empty census -> MEASUREMENT ABSENT, not a substantive verdict",
       lambda: score_fn(void)["R0"]["verdict"] == V_UNDET)
    ck("A25 empty census also stops",
       lambda: score_fn(void).get("stop") is True)
    ck("A26 zero min_hits on a green stream -> MEASUREMENT ABSENT",
       lambda: score_fn(_synth((74, 34), min_hits_by={"idx1_prefill": 0})
                        )["R0"]["verdict"] == V_UNDET)
    ck("A27 empty S_pre alone -> MEASUREMENT ABSENT (|D| = 0 anchors nothing)",
       lambda: score_fn(_synth((74, 34), empty=("S_pre",))
                        )["R0"]["verdict"] == V_UNDET)

    # ---- C2: per-target gate. An unreadable S_post may not yield an absence
    #      claim (prereg sec4.1), and may not stop the readings it does not
    #      feed either (the mirror-image error).
    post_unsat = _synth((74, 34), post_lost=4, unsat=("S_post",))
    ck("A28 unsaturated S_post -> R2 carries COVERAGE NOT SATURATED",
       lambda: score_fn(post_unsat)["R2"].get("verdict") == V_UNSAT)
    ck("A29 unsaturated S_post produces NO lost_ids / lost_count",
       lambda: not ({"lost_ids", "lost_count"} & set(score_fn(post_unsat)["R2"])))
    ck("A30 unreadable S_post does not change R0 (per-target, not global)",
       lambda: score_fn(post_unsat)["R0"]["verdict"] == V_CONSISTENT)
    idx2_unsat = _synth((74, 34), unsat=("idx2_prefill",))
    ck("A31 unsaturated idx2 -> that item only is COVERAGE NOT SATURATED",
       lambda: (score_fn(idx2_unsat)["R1"]["other_green_groups"]["idx2"]
                .get("verdict") == V_UNSAT
                and "realised" not in score_fn(idx2_unsat)["R1"]
                ["other_green_groups"]["idx2"]
                and score_fn(idx2_unsat)["R0"]["verdict"] == V_CONSISTENT))

    # ---- C3: the two worlds behind "both streams reach the whole device".
    detached = _synth((108, 108), disjoint=False,
                      not_attached=("idx1_prefill", "idx1_decode"))
    ck("A32 no green context attached -> MEASUREMENT ABSENT, not NOT_DELIVERED",
       lambda: score_fn(detached)["R0"]["verdict"] == V_UNDET)
    ck("A33 attachment read-out absent -> MEASUREMENT ABSENT (fail-closed)",
       lambda: score_fn(_drop(healthy, "driver_readout")
                        )["R0"]["verdict"] == V_UNDET)
    ck("A34 attachment entry unusable -> MEASUREMENT ABSENT (fail-closed)",
       lambda: score_fn(dict(healthy, driver_readout={"streams": {
           "idx1_prefill": {"error": "cuStreamGetGreenCtx unavailable"},
           "idx1_decode": {"rc": 0, "green_ctx_is_null": False}}})
                        )["R0"]["verdict"] == V_UNDET)
    ck("A35 the attachment premise is recorded, not merely assumed",
       lambda: score_fn(healthy)["setup"].get("green_ctx_attached") == {
           "idx1_prefill": False, "idx1_decode": False})
    ck("A36 NOT_DELIVERED text scopes itself to this process, not to the engine",
       lambda: (lambda w: "about the engine" not in w and "no engine" in w)(
           score_fn(_synth((108, 108), disjoint=False))["R0"]["why"].lower()))

    # ---- reverse protection for all three gates at once (gate #21 mirror
    #      image): the healthy artefact must still reach every reading.
    ck("A37 healthy 74/34 unaffected by the three new gates",
       lambda: (lambda v: (v["R0"]["verdict"] == V_CONSISTENT
                           and v["R2"]["lost_count"] == 0
                           and "verdict" not in v["R2"]
                           and v["R1"]["other_green_groups"]["idx2"]["realised"]
                           == [54, 54]
                           and v["R1"]["other_green_groups"]["idx2"]["disjoint"]
                           is True))(score_fn(healthy)))

    # ---- C7 / C8 / C9: the non-scorer repairs, each falsifiable.
    ck("A38 provenance records git HEAD, manifest sha and library versions",
       lambda: (lambda p: (isinstance(p.get("git_head"), str)
                           and isinstance(p.get("self_sha256"), str)
                           and isinstance(p.get("runtime_source_manifest"), dict)
                           and {"path", "sha256", "expected_entry_count"}
                           <= set(p["runtime_source_manifest"])
                           and all(isinstance(p.get(k), str) for k in (
                               "torch_version", "triton_version",
                               "python_version"))))(ns["_provenance"]()))
    ck("A39 the read-out and provenance writers document their exact contracts",
       lambda: (all("never raises" not in
                    ((ns.get(n).__doc__ if ns.get(n) else "") or "").lower()
                    for n in _NO_BLANKET_NORAISE_DOCS)
                and not _blanket_noraise_docstrings(src)   # every docstring,
                and "oserror" in                           # not just the list
                ((ns["_record_runtime_ptx"].__doc__ or "").lower())))
    ck("A40 mutant predicate rejects UNEXPECTED failures",
       lambda: ns["_mutant_ok"]({"X1": "fail", "X2": "fail"}, {"X1"})[0] is False)
    ck("A41 mutant predicate rejects missing / raised / empty expectations",
       lambda: (ns["_mutant_ok"]({"X1": "ok"}, {"X1"})[0] is False
                and ns["_mutant_ok"]({"X1": "raised", "X2": "fail"},
                                     {"X2"})[0] is False
                and ns["_mutant_ok"]({"X1": "fail"}, set())[0] is False
                and ns["_mutant_ok"]({"X1": "fail", "X2": "ok"},
                                     {"X1"})[0] is True))

    # ---- R2: the attachment premise must be FALSIFIABLE, both directions.
    #      A42/A43 are the scorer half (the artefact already carries five
    #      streams that are plain by construction; read them), A44/A45 the
    #      producer half (the read-out itself, on known-answer input). Before
    #      this the file had zero direct checks of this instrument against
    #      eight for the runtime-PTX one, and an instrument stuck on either
    #      answer was invisible.
    ck("A42 the plain-stream negative controls are recorded, not merely assumed",
       lambda: (lambda s: (s.get("plain_control_detached")
                           == {k: True for k in _PLAIN_CONTROL_KEYS}
                           and "premise holds" in (s.get("control_status") or "")
                           ))(score_fn(healthy)["setup"]))
    degenerate = _synth((108, 108), disjoint=False, attached=_PLAIN_CONTROL_KEYS)
    ck("A43 read-out that calls EVERY stream attached -> MEASUREMENT ABSENT",
       lambda: (score_fn(degenerate)["R0"]["verdict"] == V_UNDET
                and score_fn(degenerate).get("stop") is True))

    def _fake_readout():
        return ns["driver_readout"]({"green": 0xA0A0, "plain": 0xB0B0},
                                    lib=_FakeCuDriver((0xA0A0,)))

    ck("A44 the read-out calls a stream that HAS a green context attached",
       lambda: _fake_readout()["streams"]["green"]["green_ctx_is_null"] is False)
    ck("A45 the read-out calls a stream that has NONE not attached",
       lambda: _fake_readout()["streams"]["plain"]["green_ctx_is_null"] is True)

    # ---- R7: the driver-key ban, and the completeness claim behind it.
    ck("A46 the driver-key guard resolves a chained read (positive control)",
       lambda: ns["_score_excludes_target_layer"](
           _GUARD_PROBE_SRC)["driver_keys"] == ["rc"])
    ck("A47 the driver key surface covers every key the producer emits",
       lambda: (lambda em, surf: (
           {"rc", "note", "green_ctx_is_null"} <= set(em)   # extractor works
           and set(em) <= set(surf)))(                      # ...and is covered
               _driver_emitted_keys(src), ns.get("_DRIVER_KEY_SURFACE", ())))

    # ---- R9: the presence floor is a declared constant, mechanically.
    ck("A48 one id with one hit still scores (the floor is presence, not coverage)",
       lambda: score_fn(_synth((1, 1), dsize=2, min_hits=1)
                        )["R0"]["verdict"] == V_CONSISTENT)
    ck("A49 the presence floor reads the pre-registered constant, not a literal",
       lambda: _floor_constant_uses(src) >= 2)
    return res


def _extractor_fixtures():
    """Two REAL CompiledKernel objects, built without a GPU.

    `triton.compile(ASTSource(...), GPUTarget("cuda", 80, 32))` is the same
    ahead-of-time path prereg sec1 C1 already uses on the login node, so this
    needs no device. `census` carries the probe, `plain` does not -- the pair
    is what makes the conservative aggregation testable.
    """
    census_jit, toy_base_jit, _ = _kernels()
    census = _compile_aot(census_jit,
                          {"out_smid": "*i32", "out_nsmid": "*i32",
                           "out_t0": "*i64", "out_t1": "*i64",
                           "spin_ns": "i64", "iter_cap": "i32"})
    plain = _compile_aot(toy_base_jit, {"out": "*i32", "n_pad": "i32"})
    return census, plain


def _inject(kernel, device_key, compiled):
    """Put CompiledKernels exactly where a real JIT launch puts them.

    Mirrors `triton/runtime/jit.py:672-683` (`create_binder` -> 5-tuple whose
    element 0 is the kernel cache) and `:860-862` (`kernel_cache[key] =
    kernel`). Assignment, never `[]` lookup, so the defaultdict factory --
    which would need a driver -- is not called. This is what makes the
    attribute path testable with no GPU.
    """
    kernel.device_caches[device_key] = (
        {f"key{i}": c for i, c in enumerate(compiled)}, {}, None, None, None)
    return kernel


def _extractor_checks(ns, fx):
    """Battery for the runtime-PTX read-out (first half of the repair)."""
    probe_fn, score_fn = ns["runtime_ptx_probe"], ns["score"]
    census, plain = fx
    census_ptx = census.asm["ptx"]
    res = []

    def ck(name, fn):
        try:
            ok = bool(fn())
        except Exception as e:  # noqa: BLE001
            res.append((name, "raised", repr(e)))
            return
        res.append((name, "ok" if ok else "fail", ""))

    def e1():
        k = _inject(_kernels()[0], 0, [census])
        p = probe_fn(k, 0)
        return (len(p["variants"]) == 1
                and p["runtime_ptx_smid_sites"] == census_ptx.count("%smid")
                and p["runtime_ptx_smid_sites"] == 1
                and p["runtime_spin_back_edge"] is True
                and p["runtime_ptx_globaltimer_sites"] == 2
                and p["runtime_ptx_sha256"] == hashlib.sha256(
                    census_ptx.encode()).hexdigest()
                and p["variants"][0]["ptx"] == census_ptx
                and "runtime_ptx_error" not in p)

    def e2():
        k = _kernels()[0]                       # nothing ever compiled
        p = probe_fn(k, 0)
        return (p["runtime_ptx_smid_sites"] is None
                and p["runtime_ptx_globaltimer_sites"] is None
                and p["runtime_spin_back_edge"] is None
                and p["runtime_ptx_sha256"] is None
                and bool(p.get("runtime_ptx_error"))
                and len(k.device_caches) == 0)  # probe must not mutate the cache

    def e3():
        k = _inject(_kernels()[0], 3, [census])  # different device key
        p = probe_fn(k, 0)
        return len(p["variants"]) == 1 and p["runtime_ptx_smid_sites"] == 1

    def e4():
        k = _inject(_kernels()[0], 0, [census, plain])
        p = probe_fn(k, 0)
        return (len(p["variants"]) == 2
                and p["runtime_ptx_smid_sites"] == 0       # MIN, not max
                and p["runtime_spin_back_edge"] is False)  # AND, not any

    def e5():
        # End-to-end statement of the 2026-08-14 defect: a read-out that found
        # nothing must make the scorer stop, not score.
        p = probe_fn(_kernels()[0], 0)
        raw = dict(_synth((74, 34)))
        raw["runtime_ptx_smid_sites"] = p["runtime_ptx_smid_sites"]
        raw["runtime_spin_back_edge"] = p["runtime_spin_back_edge"]
        v = score_fn(raw)
        return v["R0"]["verdict"] == V_UNDET and v["stop"] is True

    def e6():
        # ...and one that found the real probe must NOT stop, so the repair
        # cannot be satisfied by refusing every artefact.
        p = probe_fn(_inject(_kernels()[0], 0, [census]), 0)
        raw = dict(_synth((74, 34)))
        raw["runtime_ptx_smid_sites"] = p["runtime_ptx_smid_sites"]
        raw["runtime_spin_back_edge"] = p["runtime_spin_back_edge"]
        return score_fn(raw)["R0"]["verdict"] == V_CONSISTENT

    def e7():
        # the actual function run() calls, on a real (AOT-compiled) kernel
        import tempfile
        rec = ns["_record_runtime_ptx"]
        with tempfile.TemporaryDirectory() as td:
            rep = {}
            rec(rep, _inject(_kernels()[0], 0, [census]), 0, td, "t")
            wrote = os.path.exists(os.path.join(td, "smid_runtime_t.ptx"))
            same = (wrote and open(os.path.join(td, "smid_runtime_t.ptx")).read()
                    == census_ptx)
        return (wrote and same
                and rep["runtime_ptx_smid_sites"] == 1
                and rep["runtime_spin_back_edge"] is True
                and rep["runtime_ptx_sha256"] == hashlib.sha256(
                    census_ptx.encode()).hexdigest()
                and len(rep["runtime_ptx_variants"]) == 1
                and "ptx" not in rep["runtime_ptx_variants"][0]
                and "runtime_ptx_error" not in rep)

    def e8():
        # ...and the same function when nothing is cached: fields None, an
        # error string, no PTX file, and no exception escaping into run()
        import tempfile
        rec = ns["_record_runtime_ptx"]
        with tempfile.TemporaryDirectory() as td:
            rep = {}
            rec(rep, _kernels()[0], 0, td, "t")
            files = os.listdir(td)
        return (files == []
                and rep["runtime_ptx_smid_sites"] is None
                and rep["runtime_spin_back_edge"] is None
                and bool(rep.get("runtime_ptx_error")))

    ck("E1 device_caches path yields the PTX of the compiled kernel", e1)
    ck("E2 nothing compiled -> all fields None + error, cache untouched", e2)
    ck("E3 variant under another device key is still found", e3)
    ck("E4 mixed variants aggregate conservatively (min sites / AND edges)", e4)
    ck("E5 empty read-out makes score() stop with MEASUREMENT ABSENT", e5)
    ck("E6 real probe read-out still reaches a substantive verdict", e6)
    ck("E7 run()'s own writer fills the artefact fields and the PTX file", e7)
    ck("E8 run()'s own writer fails closed when nothing is cached", e8)
    return res


def _report(res, fails):
    for name, status, detail in res:
        tag = {"ok": "ok", "fail": "FAIL", "raised": "RAISED"}[status]
        print(f"  [{tag}] {name}" + (f"  {detail}" if detail else ""))
        if status != "ok":
            fails.append(name)
    return fails


def selftest_analyzer():
    fails = []
    print("analyzer self-test (no GPU):")
    # A0/A23 (the AST guard, prereg sec5.3-3 "the first item of the
    # self-test") now live INSIDE the battery, so the mutants run them too.
    _report(_analyzer_checks(
        score, open(os.path.abspath(__file__), encoding="utf-8").read()), fails)
    try:
        fx = _extractor_fixtures()
    except Exception as e:  # noqa: BLE001
        print(f"  [SKIP] E1-E6 extractor battery: triton unavailable ({e!r}); "
              "THIS RUN DOES NOT VALIDATE THE RUNTIME-PTX READ-OUT")
    else:
        _report(_extractor_checks(globals(), fx), fails)
    print("PASS" if not fails else f"FAIL: {fails}")
    return 0 if not fails else 1


# ==========================================================================
# 7. Mutation tests (methodology lesson #53, gate #50). Mandatory companion to
#    the 2026-08-21 fail-open repair: each mutant UNDOES one part of the repair
#    in a COPY of this file's source, and the SAME battery is re-run against
#    the copy. A named check that does not FAIL under the mutant which removes
#    the thing it claims to test is an identity, and is reported as such
#    instead of being counted as evidence.
# ==========================================================================
_PRE_REPAIR_PROBE_GUARD = """    if raw.get("runtime_ptx_smid_sites") == 0:
        out["R0"] = {"verdict": V_UNDET,
                     "why": "runtime PTX contained no %smid site -- the kernel "
                            "that ran is not the probe"}
        out["stop"] = True
        return out
"""

_PRE_REPAIR_RESIDENCY_GUARD = """    if raw.get("runtime_spin_back_edge") is False:
        out["R0"] = {"verdict": V_UNDET,
                     "why": "runtime PTX has no clock read inside a loop with a "
                            "back edge -- residency was not achieved, so every "
                            "union is a lower bound of unknown slack"}
        out["stop"] = True
        return out
"""

# (begin marker, end marker) of every repaired region. The 2026-08-14
# fail-open repair owns the first two; the 2026-08-21 audit repair owns the
# rest. `_splice` takes the FIRST occurrence, which is the code -- these
# tuples are defined after every one of those regions on purpose.
_PROBE_REGION = ("    # --- BEGIN probe-present guard",
                 "    # --- END probe-present guard ---\n")
_RESIDENCY_REGION = ("    # --- BEGIN residency guard",
                     "    # --- END residency guard ---\n")
_POSITIVITY_REGION = ("    # --- BEGIN positivity floor",
                      "    # --- END positivity floor ---\n")
_ATTACH_REGION = ("    # --- BEGIN green-context attachment premise",
                  "    # --- END green-context attachment premise ---\n")
_CONTROLS_REGION = ("    # --- BEGIN plain-control detachment premise",
                    "    # --- END plain-control detachment premise ---\n")
_SPOST_REGION = ("    # --- BEGIN S_post target gate",
                 "    # --- END S_post target gate ---\n")
_IDXGATE_REGION = ("        # --- BEGIN per-group gate",
                   "        # --- END per-group gate ---\n")

# The 2026-08-14 R2 body, restored verbatim by `_mut_drop_spost_gate`: it
# published `lost_count` off any S_post and put the saturation flag BESIDE the
# number instead of gating it (audit item C2).
_PRE_REPAIR_R2 = """    Spost = set(raw["S_post"]["union"])
    lost = sorted(D - Spost)
    out["R2"] = {
        "S_pre_size": len(D), "S_post_size": len(Spost),
        "lost_ids": lost, "lost_count": len(lost),
        "saturated_post": out["D1"].get("S_post", {}).get("saturated"),
        "reading": ("carve observed" if lost else "no carve observed"),
        "not_independent": "descriptive replicates, not an independent test",
        "descriptive_replicates": {
            k: len(set(ps[k]["union"]))
            for k in ("idx0_prefill", "idx0_decode",
                      f"idx{idx_last}_prefill", f"idx{idx_last}_decode")
            if k in ps},
    }
"""

# The 2026-08-14 docstring of `_record_runtime_ptx`, whose "Never raises" was
# stronger than the code (audit item C8).
_PRE_REPAIR_RECORD_DOC = '''    """Write the runtime-instrument fields (and PTX files) into `rep`.

    Split out of `run()` on purpose. Never raises -- a failure leaves
    the fields None and `score()` then stops with UNDETERMINED
    (MEASUREMENT ABSENT).
    """
'''


# The 2026-08-21 first-repair docstring of `_provenance`, restored verbatim
# by `_mut_restore_never_raises_doc_prov`: it re-used, in a brand new
# docstring, the exact unconditional phrase audit item C8 had struck from
# `_record_runtime_ptx` hours earlier (audit item R12).
_PRE_REPAIR_PROV_DOC = '''    """Run-identifying facts for the raw artefact (audit item C7, 2026-08-21).

    Never raises: every field degrades to an "ERROR ..." string, because a
    provenance gap must not abort a GPU run that has already spent its budget
    -- it must be VISIBLE in the artefact instead. None of these fields is an
    input to any verdict (`score()` does not read `provenance`).
    """
'''


def _splice(src, region, replacement):
    a, b = region
    i = src.index(a)
    j = src.index(b) + len(b)
    return src[:i] + replacement + src[j:]


def _mut_revert_probe(src):
    return _splice(src, _PROBE_REGION, _PRE_REPAIR_PROBE_GUARD)


def _mut_revert_residency(src):
    return _splice(src, _RESIDENCY_REGION, _PRE_REPAIR_RESIDENCY_GUARD)


def _mut_revert_both(src):
    return _mut_revert_residency(_mut_revert_probe(src))


def _mut_drop_probe(src):
    return _splice(src, _PROBE_REGION, "")


def _mut_drop_residency(src):
    return _splice(src, _RESIDENCY_REGION, "")


def _mut_dead_ptx_path(src):
    # Restores the 2026-08-14 attribute path (JITFunction.cache, nonexistent).
    # Anchored on two lines so it cannot match this function's own source.
    old = '    out = []\n    caches = getattr(kernel, "device_caches", None)'
    assert src.count(old) == 1, f"anchor count {src.count(old)}"
    new = '    out = []\n    caches = getattr(kernel, "cache", None)'
    return src.replace(old, new, 1)


def _mut_drop_positivity_floor(src):
    """C1: put back the scorer that could not tell "flat" from "empty"."""
    return _splice(src, _POSITIVITY_REGION, "")


def _mut_drop_attach_guard(src):
    """C3: stop asking whether a green context was attached at all.

    This is the one that answers the audit's question directly -- with the
    premise gone, does world (ii) (no green context anywhere) leak out as
    LABEL_CONSISTENT_BUT_PARTITION_NOT_DELIVERED? A32 says whether it does.
    """
    return _splice(src, _ATTACH_REGION, "")


def _mut_drop_control_guard(src):
    """R2: stop reading the negative control, keep the one-directional half.

    The premise then says only "the read-out called the green pair attached",
    which is what an instrument stuck on "attached" also says.
    """
    return _splice(src, _CONTROLS_REGION, "")


def _mut_drop_both_attach_premises(src):
    """R2: the whole attachment premise gone -- the world before audit item C3.

    Kept as a separate mutant because with BOTH halves present the fail-closed
    cases (A33 read-out block absent, A34 entry unusable) are caught by either
    half alone, so neither single-half mutant can attribute them any more.
    This one can: it is the only mutant under which an artefact with no
    attachment read-out at all reaches a substantive verdict.
    """
    return _mut_drop_control_guard(_mut_drop_attach_guard(src))


def _mut_readout_all_attached(src):
    """R2 / the audit's own Mg: the PRODUCER answers "attached" for every
    stream, including the ones `run()` builds with no green context.

    This is the degenerate instrument that passed all 50 checks on
    2026-08-21: the scorer's premise could not see it (one-directional) and no
    check ever ran the producer at all. A45 is what sees it now.
    """
    old = ('            streams[name] = {"rc": rc, "green_ctx_is_null": '
           '(g.value in (None, 0))}')
    assert src.count(old) == 1, f"anchor count {src.count(old)}"
    return src.replace(
        old, '            streams[name] = {"rc": rc, "green_ctx_is_null": False}', 1)


def _mut_readout_all_detached(src):
    """R2 / the audit's own Mh: the PRODUCER answers "not attached" for every
    stream, including the green pair. The mirror degeneracy; A44 sees it."""
    old = ('            streams[name] = {"rc": rc, "green_ctx_is_null": '
           '(g.value in (None, 0))}')
    assert src.count(old) == 1, f"anchor count {src.count(old)}"
    return src.replace(
        old, '            streams[name] = {"rc": rc, "green_ctx_is_null": True}', 1)


def _mut_score_reads_driver_rc(src):
    """R7 / the audit's own Mm: `score()` reads a driver key that the surface
    did not list. Before the surface was completed this failed nothing."""
    anchor = "    # " + "---------------- gate #21 branch, FIRST"
    assert src.count(anchor) == 1, f"anchor count {src.count(anchor)}"
    return src.replace(
        anchor,
        '    _dr = (raw.get("driver_readout") or {}) if isinstance(raw, dict) else {}\n'
        '    _leak = ((_dr.get("streams") or {}).get("idx1_prefill") or {}).get("rc")\n'
        + anchor, 1)


def _mut_driver_emits_unlisted_key(src):
    """R7: add an emitted key to the producer without declaring it."""
    old = '    out["streams"] = streams\n    return out'
    assert src.count(old) == 1, f"anchor count {src.count(old)}"
    return src.replace(
        old, '    out["unlisted_probe_key"] = 1\n' + old, 1)


def _mut_drop_driver_key_guard(src):
    """R7: keep the surface, throw away the finding computed from it.

    A23 cannot see this (the battery runs the INTACT guard over the mutant's
    source, so a mutant may not weaken its own examiner); A46 can, because it
    calls the guard OF THE FILE UNDER TEST on a fixture with a known leak.
    """
    old = ('            "driver_keys": sorted(set(_DRIVER_KEY_SURFACE) & names\n'
           '                                  - set(_ALLOWED_DRIVER_' + 'KEYS))}')
    assert src.count(old) == 1, f"anchor count {src.count(old)}"
    return src.replace(old, '            "driver_keys": []}', 1)


def _mut_raise_positivity_floor(src):
    """R9 / the audit's own Md: raise the presence floor after the fact."""
    old = "\nMIN_HITS_" + "REPORTED = 1\n"
    assert src.count(old) == 1, f"anchor count {src.count(old)}"
    return src.replace(old, "\nMIN_HITS_" + "REPORTED = 2\n", 1)


def _mut_hardwire_positivity_floor(src):
    """R9: put the bound back as a bare literal. Same VALUE, same behaviour --
    the only signature is the wiring itself, which is what A49 measures."""
    old = ('    hits_ok = (isinstance(hits, int) and not isinstance(hits, bool)\n'
           '               and hits >= MIN_HITS_' + 'REPORTED)\n'
           '    if len(ids) < MIN_HITS_' + 'REPORTED or not hits_ok:')
    assert src.count(old) == 1, f"anchor count {src.count(old)}"
    new = ('    hits_ok = (isinstance(hits, int) and not isinstance(hits, bool)\n'
           '               and hits >= 1)\n'
           '    if len(ids) < 1 or not hits_ok:')
    return src.replace(old, new, 1)


def _mut_restore_never_raises_doc_prov(src):
    """R12: put the blanket phrase back, in the docstring A39 did not watch."""
    head = "def _prov" + "enance():"
    assert src.count(head) == 1, f"anchor count {src.count(head)}"
    i = src.index(head)
    a = src.index('    """', i)
    b = src.index('    """\n', a + 7) + len('    """\n')
    return src[:a] + _PRE_REPAIR_PROV_DOC + src[b:]


def _mut_noraise_doc_in_unlisted_function(src):
    """R12: put the blanket promise in a function that is NOT on the list.

    Without this the docstring SWEEP would be redundant with the name list --
    both halves of A39 fail together under the two `restore_never_raises_*`
    mutants, so nothing would show that the sweep covers anything the list
    does not. `_saturated` is on no list and never will be; that is the point.
    """
    anchor = ("    " + '"""'
              + "Set equality of the union at the last two grid points "
                "(prereg sec5.1).\n")
    assert src.count(anchor) == 1, f"anchor count {src.count(anchor)}"
    return src.replace(anchor, anchor + "\n    Never " + "raises.\n", 1)


def _mut_drop_spost_gate(src):
    """C2: restore the R2 that computed an absence claim off any S_post."""
    return _splice(src, _SPOST_REGION, _PRE_REPAIR_R2)


def _mut_drop_idx_gate(src):
    """C2: restore the idx2+ read-out that reported unsaturated sizes."""
    return _splice(src, _IDXGATE_REGION, "")


def _mut_score_reads_target_number(src):
    """C3a: make score() read a target NUMBER. The AST guard must catch it."""
    anchor = "    # " + "---------------- gate #21 branch, FIRST"
    assert src.count(anchor) == 1, f"anchor count {src.count(anchor)}"
    return src.replace(
        anchor,
        '    _leak = raw.get("total_sm_reported") if isinstance(raw, dict) '
        'else None\n' + anchor, 1)


def _mut_score_reads_driver_primary_sm(src):
    """C3a: read a NON-whitelisted driver key (not a banned numeric token).

    Without the whitelist half of the guard, narrowing `_TARGET_LAYER_TOKENS`
    to the numeric family would have opened the whole rest of the driver block.
    """
    anchor = "    # " + "---------------- gate #21 branch, FIRST"
    assert src.count(anchor) == 1, f"anchor count {src.count(anchor)}"
    return src.replace(
        anchor,
        '    _leak = (raw.get("driver_readout") or {}).get("primary_sm") '
        'if isinstance(raw, dict) else None\n' + anchor, 1)


def _mut_helper_reads_target_number(src):
    """C3a: launder the target read through a HELPER score() calls.

    The pre-2026-08-21 guard read `score()`'s own body only, so this mutant
    would have passed it. The anchor spans two lines so that this function's
    own source (where the newline is an escape) is not a second occurrence.
    """
    anchor = '    lad = target.get("ladder") or []\n    if len(lad) < 2:'
    assert src.count(anchor) == 1, f"anchor count {src.count(anchor)}"
    return src.replace(anchor,
                       '    lad = target.get("ladder") or []\n'
                       '    _leak = target.get("nsmid_observed")\n'
                       '    if len(lad) < 2:', 1)


def _mut_strip_provenance(src):
    """C7: give back the artefact that did not say which tree produced it."""
    # assembled in two pieces so this function is not itself a second and
    # third occurrence of the definition line it searches for
    head = "def _prov" + "enance():"
    assert src.count(head) == 1, f"anchor count {src.count(head)}"
    i = src.index(head)
    j = src.index("\n\n\ndef run(", i)
    return src[:i] + head + "\n    return {}\n" + src[j:]


def _mut_restore_never_raises_doc(src):
    """C8: put back the docstring that promised more than the code delivers."""
    i = src.index("def _record_runtime_ptx(rep, kernel, device, outdir, tag):")
    a = src.index('    """', i)
    b = src.index('    """\n', a + 7) + len('    """\n')
    return src[:a] + _PRE_REPAIR_RECORD_DOC + src[b:]


def _mut_harness_ignores_extra(src):
    """C9: make the mutant predicate tolerate unexpected failures again.

    Assembled in two pieces so this line is not a second occurrence of the
    line it searches for.
    """
    old = "    good = not raised and not missing and not " + "extra and bool(exp)"
    assert src.count(old) == 1, f"anchor count {src.count(old)}"
    return src.replace(old, "    good = not raised and not missing "
                            "and bool(exp)", 1)


def _mut_aggregate_max(src):
    # Aggregates variants optimistically instead of conservatively.
    old = ('    out["runtime_ptx_smid_sites"] = min(v["smid_sites"] '
           'for v in out["variants"])')
    assert src.count(old) == 1
    return src.replace(old, old.replace("min(", "max(", 1), 1)


# name -> (mutation fn, names of the checks that MUST fail under it)
MUTANTS = {
    "revert_probe_guard_to_eq0": (_mut_revert_probe, {
        "A5 probe field ABSENT -> MEASUREMENT ABSENT (fail-closed)",
        "A6 probe field None -> MEASUREMENT ABSENT (fail-closed)",
        "A7 probe field non-numeric -> MEASUREMENT ABSENT (fail-closed)",
        "A11 every fail-closed stop also sets stop=True"}),
    # NB: E5 is deliberately NOT expected here. Its input has BOTH runtime
    # fields None, so the still-repaired residency guard stops it on its own.
    # Only the "revert both" mutant reproduces the 2026-08-14 fail-open, and
    # that is exactly what E5 asserts about.
    "revert_residency_guard_to_is_false": (_mut_revert_residency, {
        "A8 residency field ABSENT -> MEASUREMENT ABSENT (fail-closed)",
        "A9 residency field None -> MEASUREMENT ABSENT (fail-closed)",
        "A10 residency field non-boolean -> MEASUREMENT ABSENT (fail-closed)",
        "A11 every fail-closed stop also sets stop=True"}),
    "revert_both_guards_to_2026-08-14": (_mut_revert_both, {
        "A5 probe field ABSENT -> MEASUREMENT ABSENT (fail-closed)",
        "A6 probe field None -> MEASUREMENT ABSENT (fail-closed)",
        "A7 probe field non-numeric -> MEASUREMENT ABSENT (fail-closed)",
        "A8 residency field ABSENT -> MEASUREMENT ABSENT (fail-closed)",
        "A9 residency field None -> MEASUREMENT ABSENT (fail-closed)",
        "A10 residency field non-boolean -> MEASUREMENT ABSENT (fail-closed)",
        "A11 every fail-closed stop also sets stop=True",
        "E5 empty read-out makes score() stop with MEASUREMENT ABSENT"}),
    "drop_probe_guard": (_mut_drop_probe, {
        "A3 runtime PTX with 0 %smid sites -> MEASUREMENT ABSENT",
        "A5 probe field ABSENT -> MEASUREMENT ABSENT (fail-closed)",
        "A6 probe field None -> MEASUREMENT ABSENT (fail-closed)",
        "A7 probe field non-numeric -> MEASUREMENT ABSENT (fail-closed)",
        "A11 every fail-closed stop also sets stop=True"}),  # E5: see above
    "drop_residency_guard": (_mut_drop_residency, {
        "A4 runtime spin hoisted (False) -> MEASUREMENT ABSENT",
        "A8 residency field ABSENT -> MEASUREMENT ABSENT (fail-closed)",
        "A9 residency field None -> MEASUREMENT ABSENT (fail-closed)",
        "A10 residency field non-boolean -> MEASUREMENT ABSENT (fail-closed)",
        "A11 every fail-closed stop also sets stop=True"}),
    "dead_ptx_path_dot_cache": (_mut_dead_ptx_path, {
        "E1 device_caches path yields the PTX of the compiled kernel",
        "E3 variant under another device key is still found",
        "E4 mixed variants aggregate conservatively (min sites / AND edges)",
        "E6 real probe read-out still reaches a substantive verdict",
        "E7 run()'s own writer fills the artefact fields and the PTX file"}),
    "aggregate_max_not_min": (_mut_aggregate_max, {
        "E4 mixed variants aggregate conservatively (min sites / AND edges)"}),

    # ---- 2026-08-21 audit repair. Same contract: each mutant undoes ONE
    #      repair and the checks that claim to cover it must FAIL -- and,
    #      since C9, NOTHING ELSE may fail (an unexpected failure means the
    #      mutant is not surgical and the expected set is not the coverage).
    # A49 belongs here from 2026-08-21: the floor region is where the
    # pre-registered constant is read, so removing the region also removes the
    # wiring. Listing it is the C9 rule working as intended -- an expectation
    # set is what the mutant really covers, not what it meant to cover.
    "drop_positivity_floor": (_mut_drop_positivity_floor, {
        "A24 empty census -> MEASUREMENT ABSENT, not a substantive verdict",
        "A25 empty census also stops",
        "A26 zero min_hits on a green stream -> MEASUREMENT ABSENT",
        "A27 empty S_pre alone -> MEASUREMENT ABSENT (|D| = 0 anchors nothing)",
        "A49 the presence floor reads the pre-registered constant, not a literal"}),
    "drop_S_post_target_gate": (_mut_drop_spost_gate, {
        "A28 unsaturated S_post -> R2 carries COVERAGE NOT SATURATED",
        "A29 unsaturated S_post produces NO lost_ids / lost_count"}),
    "drop_idx_group_gate": (_mut_drop_idx_gate, {
        "A31 unsaturated idx2 -> that item only is COVERAGE NOT SATURATED"}),
    # NB (2026-08-21): A33/A34 left this set when the second half of the
    # premise landed. Their inputs have NO usable read-out at all, so either
    # half stops them and neither half alone can claim them; the mutant that
    # removes BOTH is where they are attributable now. Shrinking an
    # expectation set is not a weaker test here -- an over-broad set is
    # precisely what the C9 rule forbids.
    "drop_greenctx_attach_premise": (_mut_drop_attach_guard, {
        "A32 no green context attached -> MEASUREMENT ABSENT, not NOT_DELIVERED",
        "A35 the attachment premise is recorded, not merely assumed"}),
    "score_reads_target_number": (_mut_score_reads_target_number, {
        "A0 score() reads no target-layer number, transitively"}),
    "score_reads_driver_primary_sm": (_mut_score_reads_driver_primary_sm, {
        "A23 score() touches no driver read-out key outside the setup whitelist"}),
    "helper_reads_target_number": (_mut_helper_reads_target_number, {
        "A0 score() reads no target-layer number, transitively"}),
    "strip_provenance": (_mut_strip_provenance, {
        "A38 provenance records git HEAD, manifest sha and library versions"}),
    "restore_never_raises_docstring": (_mut_restore_never_raises_doc, {
        "A39 the read-out and provenance writers document their exact contracts"}),
    "harness_ignores_extra_failures": (_mut_harness_ignores_extra, {
        "A40 mutant predicate rejects UNEXPECTED failures"}),

    # ---- 2026-08-21 RE-AUDIT repair (items R2, R7, R9, R12). The first four
    #      entries below are the audit's own two degenerate read-out worlds
    #      plus its `rc` leak and its raised floor, imported into this harness
    #      so that "the audit ran a mutation and nothing failed" is a
    #      statement this file can now reproduce and refute by itself.
    "drop_control_detached_premise": (_mut_drop_control_guard, {
        "A42 the plain-stream negative controls are recorded, not merely assumed",
        "A43 read-out that calls EVERY stream attached -> MEASUREMENT ABSENT"}),
    "drop_both_attachment_premises": (_mut_drop_both_attach_premises, {
        "A32 no green context attached -> MEASUREMENT ABSENT, not NOT_DELIVERED",
        "A33 attachment read-out absent -> MEASUREMENT ABSENT (fail-closed)",
        "A34 attachment entry unusable -> MEASUREMENT ABSENT (fail-closed)",
        "A35 the attachment premise is recorded, not merely assumed",
        "A42 the plain-stream negative controls are recorded, not merely assumed",
        "A43 read-out that calls EVERY stream attached -> MEASUREMENT ABSENT"}),
    "readout_degenerate_all_attached": (_mut_readout_all_attached, {
        "A45 the read-out calls a stream that has NONE not attached"}),
    "readout_degenerate_all_detached": (_mut_readout_all_detached, {
        "A44 the read-out calls a stream that HAS a green context attached"}),
    "score_reads_driver_rc": (_mut_score_reads_driver_rc, {
        "A23 score() touches no driver read-out key outside the setup whitelist"}),
    "driver_emits_unlisted_key": (_mut_driver_emits_unlisted_key, {
        "A47 the driver key surface covers every key the producer emits"}),
    "drop_driver_key_guard": (_mut_drop_driver_key_guard, {
        "A46 the driver-key guard resolves a chained read (positive control)"}),
    "raise_positivity_floor": (_mut_raise_positivity_floor, {
        "A48 one id with one hit still scores (the floor is presence, not coverage)"}),
    "hardwire_positivity_floor": (_mut_hardwire_positivity_floor, {
        "A49 the presence floor reads the pre-registered constant, not a literal"}),
    "restore_never_raises_doc_provenance": (_mut_restore_never_raises_doc_prov, {
        "A39 the read-out and provenance writers document their exact contracts"}),
    "never_raises_doc_in_unlisted_function": (
        _mut_noraise_doc_in_unlisted_function, {
            "A39 the read-out and provenance writers document their exact "
            "contracts"}),
}


def _ck_sort_key(check_id):
    """Sort A3 before A21 without dying on a non-numeric id."""
    digits = re.sub(r"\D", "", check_id)
    return (check_id[0], int(digits) if digits else 0, check_id)


def _mutant_ok(status, expected):
    """The mutant verdict predicate, factored out so it can itself be TESTED.

    A mutant PASSES only when it breaks EXACTLY what it claims to cover:
      * `raised`  -- a check that died of an unrelated exception tested nothing
      * `missing` -- a check that claims to cover this defect but passed
                     anyway IS AN IDENTITY (methodology lesson #53)
      * `extra`   -- ★2026-08-21 audit item C9. Unexpected failures used not
                     to count, so a NON-SURGICAL mutant (one that also breaks
                     things it does not name) passed, and the expectation set
                     was only a lower bound. Lesson #53 asks for an EXACT
                     expected set, so an unexpected failure now fails the
                     mutant and forces the registry to say what it really
                     covers.
      * empty `exp` -- nothing claims to cover this mutant at all
    A0/A23/A38-A41 are checked here (`_analyzer_checks` receives the mutant's
    own source and namespace), which is what makes the non-scorer repairs
    falsifiable too. Returns (good, failed, missing, extra, raised, exp).
    """
    raised = sorted(n for n, s in status.items() if s == "raised")
    failed = {n for n, s in status.items() if s == "fail"}
    exp = {e for e in expected if e in status}  # E* absent when triton is
    missing = sorted(exp - failed)
    extra = sorted(failed - exp)
    good = not raised and not missing and not extra and bool(exp)
    return good, failed, missing, extra, raised, exp


def selftest_mutants():
    src = open(os.path.abspath(__file__), encoding="utf-8").read()
    try:
        fx = _extractor_fixtures()
    except Exception as e:  # noqa: BLE001
        fx = None
        print(f"[mutants] triton unavailable ({e!r}); the E* battery is SKIPPED "
              "and the read-out mutants cannot be judged")
    ok = True
    for name, (mutate, expected) in MUTANTS.items():
        try:
            mutated = mutate(src)
        except (ValueError, AssertionError) as e:
            print(f"  [FAIL] mutant {name}: anchor not found ({e!r}) -- the "
                  "harness is stale, it is not testing this file")
            ok = False
            continue
        if mutated == src:
            print(f"  [FAIL] mutant {name}: source unchanged -- no mutation")
            ok = False
            continue
        ns = {"__name__": "_mutant", "__file__": os.path.abspath(__file__)}
        try:
            exec(compile(mutated, f"<mutant:{name}>", "exec"), ns)
        except Exception as e:  # noqa: BLE001
            print(f"  [FAIL] mutant {name}: did not compile/exec ({e!r})")
            ok = False
            continue
        res = _analyzer_checks(ns["score"], mutated, ns)
        if fx is not None:
            res += _extractor_checks(ns, fx)
        status = {n: s for n, s, _ in res}
        good, failed, missing, extra, raised, _exp = _mutant_ok(status, expected)
        ids = " ".join(sorted((n.split()[0] for n in failed), key=_ck_sort_key))
        print(f"  [{'PASS' if good else 'FAIL'}] mutant {name}: "
              f"{len(failed)}/{len(status)} checks FAIL -> {ids}"
              + (f"; MISSING (should have failed, passed instead) {missing}"
                 if missing else "")
              + (f"; RAISED {raised}" if raised else "")
              + (f"; also failed {extra}" if extra else ""))
        if not good:
            ok = False
    print("MUTANTS ALL PASS" if ok else "MUTANT FAILURES PRESENT")
    return 0 if ok else 1


# ==========================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest-cpu", action="store_true")
    ap.add_argument("--selftest-analyzer", action="store_true")
    ap.add_argument("--selftest-mutants", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--analyze", metavar="RAW.json")
    ap.add_argument("--outdir", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--tag", default=os.environ.get("SLURM_JOB_ID", "local"))
    ap.add_argument("--spin-ns", type=int, default=SPIN_NS_DEFAULT)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    if a.selftest_mutants:
        return selftest_mutants()
    if a.selftest_analyzer:
        # lesson #53: the battery and the mutants that make it falsifiable are
        # one artefact; running the battery alone would prove nothing.
        return selftest_analyzer() or selftest_mutants()
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
