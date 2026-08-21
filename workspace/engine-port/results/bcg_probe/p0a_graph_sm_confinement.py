"""
P0-A PROBE -- does CUDA-graph replay preserve Green-Context SM confinement?

★ NOT production code. ★ NO SERVER. ★ NO MODEL. ★ NO REQUEST.
This launches spin kernels on CUDA streams and reads a hardware register.
It produces NO latency, NO throughput, NO goodput -- there is nothing in its
output that can be quoted as a performance number, by construction. It is a
TOOL-VALIDITY probe, not a performance verdict (2026-08-20 job 886718 precedent).

------------------------------------------------------------------------------
THE QUESTION (구멍 C / assumption E5, registered 2026-08-11, never measured)
------------------------------------------------------------------------------
workspace/engine-port/reports/SMID_DIRECT_INSTRUMENTATION_DESIGN_2026-08-11.md
:77-96 records, as a code fact, that under `--enable-pdmux`
cuda_graph_runner.py:806-817 captures one decode graph per stream group ON THAT
GROUP'S GREEN-CONTEXT STREAM, and then asks:

    "green-ctx 스트림 위에서 캡처된 CUDA graph를 재생할 때 그 green context의
     SM 제한이 그대로 전달되는가?  ... 어떤 아티팩트도 이걸 측정한 적이 없다."

and explicitly refuses to guess the direction (":93  여기서 방향을 단정하지 않는다").
The operating point is cudagraph-ON and every decode-side SM split goes through
that path, so this is upstream of the whole decode-side SM-split story -- and
upstream of every Breakable-CUDA-Graph scenario in
workspace/engine-port/reports/BCG_APPLICABILITY_2026-08-21.md.

------------------------------------------------------------------------------
DESIGN -- four legs, two levels of control
------------------------------------------------------------------------------
The census machinery is IMPORTED, never re-implemented, from
results/smid_census/smid_l0_census.py (829 lines, pre-registered in
PREREG_SMID_R0_2026-08-14.md). That file is NOT modified by this probe -- this
is a derivative script, following the 2026-08-20 p1_greenctx_target.py
precedent. Methodology gate #9: put the contrast on the producer, not on a
re-implementation of it.

  leg              stream                       execution      role
  ---------------- ---------------------------- -------------- --------------------
  eager_plain      plain (no green ctx)         eager launch   positive control
  graph_plain      plain (no green ctx)         graph replay   2nd-level control:
                                                               the graph path itself
                                                               does not break census
  eager_green      green pair idx1, DECODE half eager launch   baseline (= base R0)
  graph_green      green pair idx1, DECODE half  graph replay  ★ CONDITION UNDER TEST

The DECODE half of the green pair is used because that is exactly the stream
the engine captures decode graphs on (`graph_capture(stream=sg[1])`).

Decision quantity: U(leg) = |{ observed %smid values }| unioned over the grid
sweep and repeats, i.e. how many distinct physical SMs the leg's blocks landed
on. Grid/repeat schedule and spin length are inherited verbatim from the
pre-registered constants in smid_l0_census.py -- this probe introduces no new
sweep knobs.

------------------------------------------------------------------------------
PRE-FIXED DECISION RULE (fixed here, before any GPU run)
------------------------------------------------------------------------------
Let TOTAL = spatial.get_sm_available(dev), D = decode-half SM count of the
green division actually built, TOL = 4 (A100 green-context granularity is
`multiple = 2` per pdmux_context.get_arch_constraints((8, 0)); TOL is two
granules, chosen before the run and not adjustable afterwards).

  PRECONDITIONS -- if any fails the verdict is UNDETERMINED (MEASUREMENT
  ABSENT) and NOTHING else is reported. A measurement failure is not a gate
  failure (methodology lesson #21, 8+ recurrences):
    P1  census instrument alive at run time: %smid site present in the PTX that
        actually ran, and the residency loop still has its back edge.
    P2  U(eager_plain) == TOTAL          -- census saturates the full GPU.
    P3  U(graph_plain) == TOTAL          -- the graph path does not, by itself,
                                            suppress census coverage.
    P4  U(eager_green) <= D + TOL        -- the green context delivers AT ALL in
                                            eager mode. If it does not, the base
                                            R0 question is unanswered and this
                                            probe cannot be interpreted; the
                                            verdict points at PREREG_SMID_R0.

  VERDICT (preconditions passed):
    |U(graph_green) - U(eager_green)| <= TOL and U(graph_green) <= D + TOL
        -> CONFINEMENT_PRESERVED_THROUGH_GRAPH_REPLAY
    U(graph_green) >= TOTAL - TOL
        -> CONFINEMENT_LOST_THROUGH_GRAPH_REPLAY
    otherwise
        -> UNDETERMINED (OUTCOME OUTSIDE PRE-REGISTERED MATRIX)

  If graph capture on the green stream RAISES, the verdict is
  UNDETERMINED (GRAPH CAPTURE UNAVAILABLE ON GREEN STREAM) with the exception
  text recorded. That is a tool-availability fact, not a confinement verdict --
  it does NOT mean confinement was lost.

------------------------------------------------------------------------------
★2026-08-21 DEAD-CODE REPAIR -- P1's INPUT, not P1 itself
------------------------------------------------------------------------------
P1 reads two fields of the raw artefact. Until this repair `run()` filled them
with an inline copy of the 2026-08-14 census read-out, which looked the
compiled kernel up under `JITFunction.cache` -- an attribute Triton 3.5.1 does
not have (measured on this venv: `hasattr` is False). The lookup therefore
raised on every launch, both fields were written as None, and P1 -- correctly
fail-closed -- would have stopped EVERY GPU run at UNDETERMINED (MEASUREMENT
ABSENT). The probe could not have produced any verdict; the job was waste.

The repair deletes that copy and binds the census's own writer,
`smid_l0_census._record_runtime_ptx` (repaired and mutation-tested in that
file on 2026-08-21), as a module-level name here. NOTHING in the decision
matrix, TOL, or the P1/P2/P3/P4/NOCAP guards is touched: P1's condition and
text are byte-identical, it merely now receives data instead of None. The
self-test gained the R1-R3 read-out battery and `--selftest-mutants` gained
`dead_runtime_ptx_readout`, which puts the dead body back and demands that R1
and R2 FAIL under it (methodology lesson #53).

------------------------------------------------------------------------------
WHAT THIS PROBE DOES NOT ANSWER (mandatory)
------------------------------------------------------------------------------
  * Nothing about kernel efficiency, occupancy, wave quantization, or the
    high-SM decode flattening mechanism. %smid gives IDENTITY only.
  * Nothing about Breakable CUDA Graph itself -- that is P0-B.
  * Nothing about whether any policy is better than any other.
  * A PRESERVED verdict does not license any performance claim; it only removes
    one documented hole (구멍 C).

Usage:
    python p0a_graph_sm_confinement.py --selftest-analyzer   # CPU only (also runs mutants)
    python p0a_graph_sm_confinement.py --selftest-mutants    # CPU only, mutants alone
    python p0a_graph_sm_confinement.py --run                 # requires GPU
    python p0a_graph_sm_confinement.py --analyze RAW.json    # scores a raw file
"""

import argparse
import hashlib  # noqa: F401 -- used by the pre-repair mutant body below
import json
import os
import platform
import sys
import time

# The pre-registered census producer. Imported, never edited.
_SMID_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "smid_census")
sys.path.insert(0, _SMID_DIR)
import smid_l0_census as CEN  # noqa: E402

TOL = 4  # pre-fixed, see module docstring. Two A100 green-context granules.

V_PRESERVED = "CONFINEMENT_PRESERVED_THROUGH_GRAPH_REPLAY"
V_LOST = "CONFINEMENT_LOST_THROUGH_GRAPH_REPLAY"
V_UNDET = CEN.V_UNDET  # "UNDETERMINED (MEASUREMENT ABSENT)"
V_OUTSIDE = CEN.V_AMBIG  # "UNDETERMINED (OUTCOME OUTSIDE PRE-REGISTERED MATRIX)"
V_NOCAP = "UNDETERMINED (GRAPH CAPTURE UNAVAILABLE ON GREEN STREAM)"

# P1's input. This is the census's own writer, NOT a copy of it -- a second
# copy of that read-out is exactly how this probe inherited the 2026-08-14
# dead path (methodology gate #9: put the work on the producer). It is bound
# as a module-level name, rather than called through CEN inside run(), so
# that (a) the CPU battery below can exercise the very object run() calls
# without a GPU, and (b) a source mutant can swap the dead body back in and
# prove this line is load-bearing.
_record_runtime_ptx = CEN._record_runtime_ptx


# ==========================================================================
# 1. Execution legs
# ==========================================================================
def _alloc(n_blocks, dev):
    import torch
    return (torch.full((n_blocks,), -1, dtype=torch.int32, device=f"cuda:{dev}"),
            torch.full((n_blocks,), -1, dtype=torch.int32, device=f"cuda:{dev}"),
            torch.zeros((n_blocks,), dtype=torch.int64, device=f"cuda:{dev}"),
            torch.zeros((n_blocks,), dtype=torch.int64, device=f"cuda:{dev}"))


def _leg_eager(kernel, stream, spin_ns, dev):
    """Union of %smid over the pre-registered grid/repeat schedule, eager."""
    import torch
    union = set()
    ladder = []
    for n_blocks in CEN.GRID_SWEEP:
        for _ in range(CEN.REPEATS_PER_GRID):
            smid, nsmid, t0, t1 = _alloc(n_blocks, dev)
            with torch.cuda.stream(stream):
                kernel[(n_blocks,)](smid, nsmid, t0, t1, spin_ns,
                                    CEN.SPIN_ITER_CAP, num_warps=1)
            stream.synchronize()
            union.update(v for v in smid.tolist() if v >= 0)
        ladder.append({"n_blocks": n_blocks, "union_size": len(union)})
    return {"union": sorted(union), "union_size": len(union), "ladder": ladder}


def _leg_graph(kernel, stream, spin_ns, dev):
    """Same schedule, but every launch is a replay of a graph CAPTURED ON
    `stream`. One graph per grid point (grid size is baked into the graph).

    The kernel is warmed up on a side stream first: Triton JIT compilation and
    any autotuning must not happen inside stream capture. The warmup is on a
    DIFFERENT stream on purpose, so the warmup launches do not contribute to
    this leg's census union.
    """
    import torch
    union = set()
    ladder = []
    side = torch.cuda.Stream(device=dev)
    for n_blocks in CEN.GRID_SWEEP:
        smid, nsmid, t0, t1 = _alloc(n_blocks, dev)
        # --- warmup (JIT + any lazy init), off this leg's stream
        with torch.cuda.stream(side):
            kernel[(n_blocks,)](smid, nsmid, t0, t1, spin_ns,
                                CEN.SPIN_ITER_CAP, num_warps=1)
        side.synchronize()
        torch.cuda.synchronize()
        # --- capture ON the leg's stream
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=stream):
            kernel[(n_blocks,)](smid, nsmid, t0, t1, spin_ns,
                                CEN.SPIN_ITER_CAP, num_warps=1)
        # --- replay, also on the leg's stream
        for _ in range(CEN.REPEATS_PER_GRID):
            # `fill_` MUST be issued on the leg's stream, not the default one.
            # Off-stream it is unordered against g.replay(), and a race narrows
            # this leg's union -- which for the green leg biases TOWARD the
            # positive verdict (a narrow graph_green scores PRESERVED). The
            # eager leg re-allocates every iteration and has no such hazard, so
            # off-stream fill would also make the two legs asymmetric.
            with torch.cuda.stream(stream):
                smid.fill_(-1)
                g.replay()
            stream.synchronize()
            union.update(v for v in smid.tolist() if v >= 0)
        del g
        ladder.append({"n_blocks": n_blocks, "union_size": len(union)})
    return {"union": sorted(union), "union_size": len(union), "ladder": ladder}


# ==========================================================================
# 2. Run
# ==========================================================================
def run(outdir, tag, spin_ns):
    import torch
    if spin_ns > CEN.SPIN_NS_CAP:
        raise SystemExit(
            f"spin_ns {spin_ns} exceeds the census prereg cap {CEN.SPIN_NS_CAP}")
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device; --run requires a GPU (this is a "
                         "MEASUREMENT ABSENT condition, not a result)")

    from sglang.srt.multiplex import pdmux_context as pdc
    from sgl_kernel import spatial

    census_kernel, _, _ = CEN._kernels()
    dev = torch.cuda.current_device()
    total_sm = spatial.get_sm_available(dev)
    cc = torch.cuda.get_device_capability(dev)
    divisions = pdc.divide_sm(total_sm, cc, 4 - 2)  # pdmux_a100_smoke.yml
    p_sm, d_sm = divisions[0]

    rep = {
        "kind": "bcg_p0a_raw", "tag": tag,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "device_name": torch.cuda.get_device_name(dev),
        "torch_version": torch.__version__,
        "compute_capability": list(cc), "total_sm_reported": total_sm,
        "division_under_test": [p_sm, d_sm],
        "spin_ns": spin_ns, "tol": TOL,
        "grid_sweep": list(CEN.GRID_SWEEP),
        "repeats_per_grid": CEN.REPEATS_PER_GRID,
        # gate #33: critical-path files outside sync_engine_tree.sh's manifest.
        "sha256_unmanifested": {
            "pdmux_context.py": CEN._sha256(pdc.__file__),
            "sgl_kernel/spatial.py": CEN._sha256(spatial.__file__),
            "smid_l0_census.py": CEN._sha256(
                os.path.join(_SMID_DIR, "smid_l0_census.py"))},
    }

    plain = torch.cuda.Stream(device=dev)
    rep["eager_plain"] = _leg_eager(census_kernel, plain, spin_ns, dev)

    # Runtime instrument check on the object that actually ran (P1). The CPU
    # self-test inspects AOT-compiled PTX; the JIT specialises differently, so
    # the PTX that ran is not guaranteed to be the PTX that was inspected.
    # It never raises: on any failure both fields stay None and P1 stops with
    # MEASUREMENT ABSENT. It covers the variants cached at THIS point (the
    # eager_plain leg has run, so the census kernel is compiled); variants that
    # a later leg might add are not re-checked. The PTX it dumps is tagged
    # `p0a_` so it can never be mistaken for an R0 census artefact.
    _record_runtime_ptx(rep, census_kernel, dev, outdir, f"p0a_{tag}")

    rep["graph_plain"] = _leg_graph(census_kernel, plain, spin_ns, dev)

    # Green pair, built exactly as initialize_stream_groups does
    # (pdmux_context.py:132-135). Index [1] is the DECODE half -- the stream the
    # engine captures decode graphs on.
    green = spatial.create_greenctx_stream_by_value(p_sm, d_sm, dev)
    g_decode = green[1]
    rep["eager_green"] = _leg_eager(census_kernel, g_decode, spin_ns, dev)
    try:
        rep["graph_green"] = _leg_graph(census_kernel, g_decode, spin_ns, dev)
        rep["graph_green_capture_error"] = None
    except Exception as e:  # noqa: BLE001
        rep["graph_green"] = None
        rep["graph_green_capture_error"] = repr(e)

    path = os.path.join(outdir, f"p0a_raw_{tag}.json")
    with open(path, "w") as f:
        f.write(json.dumps(rep, indent=2))
    print(f"[run] raw artefact -> {path}")
    print("[run] NO VERDICT IS PRODUCED HERE. Score with --analyze.")
    return 0


# ==========================================================================
# 3. Scoring -- pure function, no I/O, no globals
# ==========================================================================
def score(raw):
    """raw artefact dict -> verdict dict. Pure."""
    out = {"kind": "bcg_p0a_verdict", "tol": TOL}

    def undet(why):
        out["verdict"] = V_UNDET
        out["why"] = why
        return out

    for key in ("total_sm_reported", "division_under_test", "eager_plain",
                "graph_plain", "eager_green"):
        if raw.get(key) in (None, {}):
            return undet(f"raw artefact missing or empty field: {key}")

    total = raw["total_sm_reported"]
    d_sm = raw["division_under_test"][1]
    out["total_sm"] = total
    out["decode_sm_target"] = d_sm

    # P1 -- instrument alive
    if not raw.get("runtime_ptx_smid_sites") or not raw.get("runtime_spin_back_edge"):
        return undet("P1 failed: the kernel that ran lost either the %smid probe "
                     "or the residency loop back edge; the census cannot be "
                     "interpreted (measurement failure, not a gate failure)")

    u = {k: raw[k]["union_size"] for k in
         ("eager_plain", "graph_plain", "eager_green")}
    if raw.get("graph_green") is not None:
        u["graph_green"] = raw["graph_green"]["union_size"]
    out["union_sizes"] = dict(u)

    # P2 / P3 -- controls
    if u["eager_plain"] != total:
        return undet(f"P2 failed: eager_plain covered {u['eager_plain']} of "
                     f"{total} SMs; the census does not saturate the full GPU, "
                     f"so no leg is interpretable")
    if u["graph_plain"] != total:
        return undet(f"P3 failed: graph_plain covered {u['graph_plain']} of "
                     f"{total} SMs; the graph path suppresses census coverage "
                     f"on its own, so a narrow graph_green cannot be attributed "
                     f"to green-context confinement")

    # P4 -- does the green context deliver at all in eager mode?
    if u["eager_green"] > d_sm + TOL:
        out["verdict"] = V_UNDET
        out["why"] = (
            f"P4 failed: eager_green covered {u['eager_green']} SMs against a "
            f"target of {d_sm} (+TOL {TOL}). The green context does not deliver "
            f"even in eager mode, so the graph question is not yet askable. "
            f"This is the base R0 question -- see "
            f"results/smid_census/PREREG_SMID_R0_2026-08-14.md, verdict "
            f"{CEN.V_NO_DELIVERY}.")
        return out

    # Capture availability
    if raw.get("graph_green") is None:
        out["verdict"] = V_NOCAP
        out["why"] = ("graph capture on the green-context stream raised: "
                      f"{raw.get('graph_green_capture_error')!r}. This is a "
                      "tool-availability fact, NOT a confinement verdict -- it "
                      "does not mean confinement was lost.")
        return out

    gg, eg = u["graph_green"], u["eager_green"]
    if abs(gg - eg) <= TOL and gg <= d_sm + TOL:
        out["verdict"] = V_PRESERVED
        out["why"] = (f"graph_green={gg} matches eager_green={eg} within TOL="
                      f"{TOL} and stays within the {d_sm}-SM target, while the "
                      f"plain controls both saturate at {total}.")
    elif gg >= total - TOL:
        out["verdict"] = V_LOST
        out["why"] = (f"graph_green={gg} reaches the full-GPU count {total} "
                      f"(TOL={TOL}) while eager_green={eg} stayed at the "
                      f"{d_sm}-SM target: replay escaped the green context.")
    else:
        out["verdict"] = V_OUTSIDE
        out["why"] = (f"graph_green={gg} is neither within TOL of "
                      f"eager_green={eg} nor at the full-GPU count {total}.")

    out["scope"] = (
        "TOOL-VALIDITY / SM-IDENTITY ONLY. Says nothing about kernel "
        "efficiency, occupancy, wave quantization, high-SM decode flattening, "
        "or any policy ranking. Scoped to this device/driver/torch build.")
    return out


def analyze(raw_path, outdir, tag):
    with open(raw_path) as f:
        raw = json.load(f)
    v = score(raw)
    v["raw_path"] = os.path.abspath(raw_path)
    v["raw_sha256"] = CEN._sha256(raw_path)
    path = os.path.join(outdir, f"p0a_verdict_{tag}.json")
    with open(path, "w") as f:
        f.write(json.dumps(v, indent=2))
    print(json.dumps(v, indent=2))
    print(f"[analyze] verdict -> {path}")
    return 0


# ==========================================================================
# 4. Analyzer self-test -- INCLUDING mutation tests
# ==========================================================================
def _synth(gg, eg=34, ep=108, gp=108, total=108, d_sm=34):
    r = {"total_sm_reported": total, "division_under_test": [total - d_sm, d_sm],
         "runtime_ptx_smid_sites": 1, "runtime_spin_back_edge": True,
         "eager_plain": {"union_size": ep}, "graph_plain": {"union_size": gp},
         "eager_green": {"union_size": eg}}
    r["graph_green"] = None if gg is None else {"union_size": gg}
    return r


def selftest_analyzer():
    """Methodology lesson #53: a check that verifies a repair must FAIL on a
    mutant that undoes the repair. Every assertion below is paired with a
    mutation that must flip it -- a rule that cannot be falsified is an
    identity, not evidence."""
    ok = [True]

    def ck(name, cond):
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        ok[0] = ok[0] and cond

    print("-- verdict matrix")
    ck("confined graph_green -> PRESERVED",
       score(_synth(34))["verdict"] == V_PRESERVED)
    ck("full-GPU graph_green -> LOST",
       score(_synth(108))["verdict"] == V_LOST)
    ck("in-between graph_green -> OUTSIDE MATRIX",
       score(_synth(70))["verdict"] == V_OUTSIDE)
    ck("capture raised -> NOCAP (not LOST)",
       score(_synth(None))["verdict"] == V_NOCAP)

    print("-- preconditions produce UNDETERMINED, never a substantive verdict")
    ck("dead %smid probe -> UNDET",
       score(dict(_synth(34), runtime_ptx_smid_sites=0))["verdict"] == V_UNDET)
    ck("dead residency loop -> UNDET",
       score(dict(_synth(34), runtime_spin_back_edge=False))["verdict"] == V_UNDET)
    ck("eager_plain unsaturated -> UNDET",
       score(_synth(34, ep=90))["verdict"] == V_UNDET)
    ck("graph_plain unsaturated -> UNDET (cannot attribute a narrow green leg)",
       score(_synth(34, gp=40))["verdict"] == V_UNDET)
    ck("green ctx not delivered in eager -> UNDET pointing at base R0",
       score(_synth(34, eg=108))["verdict"] == V_UNDET)

    print("-- MUTATION TESTS: each check must break when its guard is removed")
    # If P3 were dropped, a run where the graph path alone collapses coverage
    # (graph_plain=40) AND graph_green=34 would be misread as PRESERVED.
    ck("P3 is load-bearing: gp=40,gg=34 must NOT score PRESERVED",
       score(_synth(34, gp=40))["verdict"] != V_PRESERVED)
    # If P4 were dropped, a run in which the green context never confined
    # ANYTHING (eager_green already at the full GPU) is reported as
    # "replay escaped the green context" -- LOST. That is this probe's most
    # dangerous misverdict, and `!= V_PRESERVED` does NOT catch it: the
    # PRESERVED branch is `abs(gg-eg) <= TOL AND gg <= d_sm + TOL`, whose
    # second conjunct already fails at gg=108 regardless of P4. Verified by
    # running a P4-deleted mutant (2026-08-21 audit): it returns LOST, and the
    # weaker assertion passed. The assertion must demand UNDETERMINED.
    ck("P4 is load-bearing: eg=108,gg=108 must score UNDETERMINED, not LOST",
       score(_synth(108, eg=108))["verdict"] == V_UNDET)
    # TOL must not be wide enough to make LOST and PRESERVED both true.
    ck("TOL does not collapse the matrix: 34 and 108 differ in verdict",
       score(_synth(34))["verdict"] != score(_synth(108))["verdict"])
    # A capture failure must not be silently absorbed into a substantive branch,
    # AND must not be confused with the preconditions: it has to survive a run
    # whose controls are all healthy. (The earlier `capture raised -> NOCAP`
    # check uses the same input; this one asserts the stronger property that
    # NOCAP is disjoint from every substantive verdict.)
    ck("NOCAP is disjoint from every substantive verdict",
       score(_synth(None))["verdict"] not in (V_PRESERVED, V_LOST, V_OUTSIDE))

    print("-- runtime-instrument read-out (P1's INPUT; the census's own writer)")
    try:
        census = _readout_fixture()
    except Exception as e:  # noqa: BLE001
        print(f"  [SKIP] R1-R3: the AOT fixture could not be built ({e!r}); "
              "THIS RUN DOES NOT VALIDATE THE RUNTIME-PTX READ-OUT")
    else:
        for name, status, detail in _readout_checks(globals(), census):
            print(f"  [{'PASS' if status == 'ok' else 'FAIL'}] {name}"
                  + (f"  <{status}> {detail}" if status != "ok" else ""))
            ok[0] = ok[0] and status == "ok"

    print("ALL PASS" if ok[0] else "FAILURES PRESENT")
    return 0 if ok[0] else 1


MUTANTS = {
    # name -> (source substring to delete, label of the check that must FAIL)
    "drop_P2": ('    if u["eager_plain"] != total:\n        return undet(',
                "P2"),
    "drop_P3": ('    if u["graph_plain"] != total:\n        return undet(',
                "P3"),
    "drop_P4": ('    if u["eager_green"] > d_sm + TOL:\n        out["verdict"]',
                "P4"),
    "drop_NOCAP": ('    if raw.get("graph_green") is None:\n        out["verdict"] = V_NOCAP',
                   "NOCAP"),
    "widen_TOL": ("TOL = 4  #", "TOL"),
}


def selftest_mutants():
    """Lesson #53, mechanised. Each guard is deleted (or neutered) in a COPY of
    this file's source, the copy is exec'd, and its `score()` is re-run against
    the same self-test inputs. A guard whose removal does not flip at least one
    assertion is not load-bearing, and the assertion that was supposed to
    protect it is an identity -- report it as such instead of claiming the
    self-test 'includes mutation tests'."""
    src = open(os.path.abspath(__file__), encoding="utf-8").read()
    ok = [True]
    for name, (needle, guard) in MUTANTS.items():
        if needle not in src:
            print(f"  [FAIL] mutant {name}: anchor not found -- harness is stale")
            ok[0] = False
            continue
        if name == "widen_TOL":
            mutated = src.replace(needle, "TOL = 74  #", 1)
        else:
            end = src.index("\n\n", src.index(needle))
            mutated = src[: src.index(needle)] + src[end:]
        ns = {"__name__": "_mutant", "__file__": __file__}
        try:
            exec(compile(mutated, f"<mutant:{name}>", "exec"), ns)
        except Exception as e:  # noqa: BLE001
            print(f"  [FAIL] mutant {name}: did not compile/exec ({e!r})")
            ok[0] = False
            continue
        m_score, m_synth = ns["score"], ns["_synth"]
        # Re-run every self-test input through the mutant and demand that at
        # least one verdict differs from the intact scorer's.
        inputs = [_synth(34), _synth(108), _synth(70), _synth(None),
                  _synth(34, ep=90), _synth(34, gp=40), _synth(108, eg=108),
                  _synth(34, eg=108),
                  dict(_synth(34), runtime_ptx_smid_sites=0)]
        flipped = []
        for i, raw in enumerate(inputs):
            try:
                a = score(raw)["verdict"]
            except Exception:  # noqa: BLE001
                a = "RAISED"
            try:
                b = m_score(dict(raw))["verdict"]
            except Exception:  # noqa: BLE001
                b = "RAISED"
            if a != b:
                flipped.append((i, a, b))
        good = bool(flipped)
        print(f"  [{'PASS' if good else 'FAIL'}] mutant {name} ({guard}) "
              f"flips {len(flipped)} verdict(s)"
              + (f" e.g. {flipped[0][1]} -> {flipped[0][2]}" if flipped else
                 "  <-- GUARD NOT LOAD-BEARING / CHECK IS AN IDENTITY"))
        ok[0] = ok[0] and good
        _ = m_synth

    # --- read-out mutant (2026-08-21 repair). Named checks, named expected
    #     failures. Unmutated source must pass the same battery first: a
    #     mutant harness whose baseline is broken proves nothing.
    print("-- runtime-instrument read-out mutant (P1's INPUT)")
    try:
        census = _readout_fixture()
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] the AOT fixture could not be built ({e!r}); the "
              "read-out mutant CANNOT BE JUDGED -- counted as a failure, not "
              "skipped (an unjudgeable guard is an unguarded one)")
        ok[0] = False
    else:
        base = _readout_checks(globals(), census)
        bad = [(n.split()[0], s, d) for n, s, d in base if s != "ok"]
        print(f"  [{'PASS' if not bad else 'FAIL'}] unmutated source passes the "
              f"R battery ({len(base) - len(bad)}/{len(base)})"
              + (f"  {bad}" if bad else ""))
        ok[0] = ok[0] and not bad
        ok[0] = _run_named_mutants(src, census) and ok[0]

    print("MUTANTS ALL PASS" if ok[0] else "MUTANT FAILURES PRESENT")
    return 0 if ok[0] else 1


# ==========================================================================
# 5. Runtime-instrument read-out (P1's INPUT) -- CPU-testable, mutation-tested
# ==========================================================================
# The 2026-08-14 read-out this probe inherited, restored verbatim by the mutant
# below. Held as a string so the intact module can never execute it.
_PRE_REPAIR_RUNTIME_PTX = '''def _record_runtime_ptx(rep, kernel, device, outdir, tag):
    """2026-08-14 dead path: `JITFunction` has no `.cache` attribute in triton
    3.5.1, so this raises on every call and writes None into both P1 fields."""
    try:
        cached = list(kernel.cache[device].values())[0]
        rt_ptx = cached.asm["ptx"]
        rep["runtime_ptx_smid_sites"] = rt_ptx.count("%smid")
        rep["runtime_spin_back_edge"] = CEN._spin_loop_has_back_edge(rt_ptx)
        rep["runtime_ptx_sha256"] = hashlib.sha256(rt_ptx.encode()).hexdigest()
    except Exception as e:  # noqa: BLE001
        rep["runtime_ptx_smid_sites"] = None
        rep["runtime_spin_back_edge"] = None
        rep["runtime_ptx_error"] = repr(e)
    return rep
'''

R1 = "R1 read-out fills the P1 fields from the kernel that actually ran"
R2 = "R2 a filled read-out lets the scorer reach a substantive verdict"
R3 = "R3 nothing compiled -> fields None, P1 stops (fail-closed, unchanged)"

_FIXTURE = []


def _readout_fixture():
    """One REAL CompiledKernel of the census kernel, built with NO GPU.

    Built by the census's own ahead-of-time fixture path, so this battery
    exercises the producer instead of a re-compilation of it. Memoised because
    `--selftest-analyzer` runs the battery twice (analyzer, then mutants).
    Raises if triton cannot compile here; each caller decides what that means.
    """
    if not _FIXTURE:
        _FIXTURE.append(CEN._extractor_fixtures()[0])
    return _FIXTURE[0]


def _readout_checks(ns, census):
    """R1-R3 against `ns`'s read-out: does P1 actually receive data?

    `ns` is a module namespace -- `globals()` for the intact file, or an exec'd
    mutant. The check bodies themselves always come from the intact file, so a
    mutant cannot weaken its own examiner. Returns [(name, status, detail)]
    with status in ok/fail/raised; `raised` is NOT a pass anywhere.
    """
    import tempfile
    rec, score_fn = ns["_record_runtime_ptx"], ns["score"]
    res = []

    def ck(name, fn):
        try:
            good = bool(fn())
        except Exception as e:  # noqa: BLE001
            res.append((name, "raised", repr(e)))
            return
        res.append((name, "ok" if good else "fail", ""))

    def readout(kernel):
        """Run the read-out into a fresh rep; report the files it wrote too."""
        with tempfile.TemporaryDirectory() as td:
            rep = {}
            rec(rep, kernel, 0, td, "t")
            return rep, sorted(os.listdir(td))

    def scored(rep):
        """Feed the measured P1 fields into the UNCHANGED scorer."""
        raw = dict(_synth(34))
        raw["runtime_ptx_smid_sites"] = rep.get("runtime_ptx_smid_sites")
        raw["runtime_spin_back_edge"] = rep.get("runtime_spin_back_edge")
        return score_fn(raw)["verdict"]

    def loaded():
        # A compiled census kernel sitting where a real JIT launch leaves it.
        return CEN._inject(CEN._kernels()[0], 0, [census])

    def r1():
        rep, files = readout(loaded())
        return (rep.get("runtime_ptx_smid_sites") == 1
                and rep.get("runtime_spin_back_edge") is True
                and not rep.get("runtime_ptx_error")
                and files == ["smid_runtime_t.ptx"])

    def r2():
        rep, _ = readout(loaded())
        return scored(rep) == V_PRESERVED

    def r3():
        rep, files = readout(CEN._kernels()[0])   # nothing ever compiled
        return (files == []
                and rep.get("runtime_ptx_smid_sites") is None
                and rep.get("runtime_spin_back_edge") is None
                and bool(rep.get("runtime_ptx_error"))
                and scored(rep) == V_UNDET)

    ck(R1, r1)
    ck(R2, r2)
    ck(R3, r3)
    return res


def _mut_dead_readout(src):
    """Undo the repair: bring back the read-out that reads a missing attribute.

    The anchor is the module-level binding of the census writer, assembled here
    from two pieces so that this line is not itself a second occurrence of the
    string it searches for (the count assertion would then always fail).
    """
    anchor = "_record_runtime_ptx = CEN." + "_record_runtime_ptx"
    n = src.count(anchor)
    if n != 1:
        raise AssertionError(f"binding occurs {n} times, expected 1")
    return src.replace(anchor, _PRE_REPAIR_RUNTIME_PTX, 1)


def _mut_drop_p1_guard(src):
    """Delete the P1 guard -- the guard this repair does NOT change.

    Without it R3 ("an empty read-out still stops the scorer") could not fail
    under any mutant, i.e. it would be an identity rather than evidence
    (methodology lesson #53). The guard in the shipping file is untouched: the
    deletion happens in a COPY. The anchor is assembled from two pieces so this
    function is not itself a second occurrence of the line it searches for.
    """
    needle = ('    if not raw.get("runtime_ptx_smid_sites") or not '
              'raw.get("runtime_spin_back_edge"):')
    n = src.count(needle)
    if n != 1:
        raise AssertionError(f"P1 guard occurs {n} times, expected 1")
    i = src.index(needle)
    return src[:i] + src[src.index("\n\n", i):]


# name -> (source mutation, the checks that MUST fail under it)
READOUT_MUTANTS = {
    "dead_runtime_ptx_readout": (_mut_dead_readout, {R1, R2}),
    "drop_P1_guard": (_mut_drop_p1_guard, {R3}),
}


def _run_named_mutants(src, census):
    """Named-check mutant harness (the census's shape, stricter than the
    verdict-flip loop above): the mutant must break EXACTLY the checks that
    claim to cover it. A check that dies with some other exception counts
    AGAINST the mutant -- a crash is not a detection."""
    ok = True
    for name, (mutate, expected) in READOUT_MUTANTS.items():
        try:
            mutated = mutate(src)
        except AssertionError as e:
            print(f"  [FAIL] mutant {name}: anchor not found ({e}) -- the "
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
        status = {n: s for n, s, _ in _readout_checks(ns, census)}
        raised = sorted(n.split()[0] for n, s in status.items() if s == "raised")
        failed = {n for n, s in status.items() if s == "fail"}
        missing = sorted(n.split()[0] for n in expected - failed)
        extra = sorted(n.split()[0] for n in failed - expected)
        good = not raised and not missing and not extra
        ids = " ".join(sorted(n.split()[0] for n in failed)) or "(none)"
        print(f"  [{'PASS' if good else 'FAIL'}] mutant {name}: "
              f"{len(failed)}/{len(status)} checks FAIL -> {ids}"
              + (f"; MISSING (should have failed, passed instead) {missing}"
                 if missing else "")
              + (f"; RAISED {raised}" if raised else "")
              + (f"; UNEXPECTED failures {extra}" if extra else ""))
        ok = ok and good
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--selftest-analyzer", action="store_true")
    ap.add_argument("--selftest-mutants", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--analyze", metavar="RAW.json")
    ap.add_argument("--outdir", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--tag", default=os.environ.get("SLURM_JOB_ID", "local"))
    ap.add_argument("--spin-ns", type=int, default=CEN.SPIN_NS_DEFAULT)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    if a.selftest_mutants:
        return selftest_mutants()
    if a.selftest_analyzer:
        rc = selftest_analyzer()
        return rc or selftest_mutants()
    if a.run:
        return run(a.outdir, a.tag, a.spin_ns)
    if a.analyze:
        return analyze(a.analyze, a.outdir, a.tag)
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
