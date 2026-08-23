"""
P0-A PROBE -- does CUDA-graph replay preserve green-context SM confinement?

★ THE PRE-REGISTRATION IS CANON, THIS DOCSTRING IS A SUMMARY (sec10-17).
  workspace/engine-port/results/bcg_probe/PREREG_P0A_2026-08-22.md (rev7).
  Where the two differ, the prereg wins and this file is the defect.

★ NOT production code. ★ NO SERVER. ★ NO MODEL. ★ NO REQUEST.
It launches spin kernels on CUDA streams and reads a hardware register. It
produces NO latency, NO throughput, NO goodput -- there is nothing in its
output that can be quoted as a performance number, by construction. It is a
TOOL-VALIDITY probe, not a performance verdict (2026-08-20 job 886718
precedent).

★ LABEL CEILING (sec6, inherited from R0 sec3.4, job 889631): `%smid` is a
  GLOBALLY CONSISTENT LABEL that survives a green context. It is NOT
  established to be a physical SM index, and this file never calls it one.
  Cardinalities of label sets are NOT convertible into compute-resource
  fractions.

------------------------------------------------------------------------------
THE QUESTION (구멍 C / assumption E5, registered 2026-08-11, never measured)
------------------------------------------------------------------------------
workspace/engine-port/reports/SMID_DIRECT_INSTRUMENTATION_DESIGN_2026-08-11.md
:77-96 records, as a code fact, that under `--enable-pdmux`
cuda_graph_runner.py:811-816 captures one decode graph per stream group ON THAT
GROUP'S GREEN-CONTEXT STREAM, and multiplexing_mixin.py:993/1007/1088/1192
replays it inside `with torch.cuda.stream(decode_stream)`, then asks whether
the green context's SM limit carries through the replay -- and explicitly
refuses to guess the direction (":93").

★ WHAT A `PRESERVED` VERDICT DOES NOT DO (sec0.3): it does not close 구멍 C,
  it does not answer R4, it licenses no performance claim, and it does not
  transfer to the cudagraph-ON serving operating point. This is an L0 toy
  graph in a process with no engine in it.

------------------------------------------------------------------------------
ARCHITECTURE -- three layers, each owned by exactly one file
------------------------------------------------------------------------------
  1. CENSUS PRODUCER   results/smid_census/smid_l0_census.py  (imported, never
                       edited; pre-registered in PREREG_SMID_R0_2026-08-14.md).
                       Owns the grid/repeat sweep, the ladder, the per-label
                       hit counts, the driver read-out and the runtime-PTX
                       instrument check.
  2. DECISION RULE     p0a_rule_totality.py  (imported, never re-implemented).
                       Owns sec4/sec5: every precondition, the attribution
                       rule, and the twelve registered labels. It is enumerated
                       over the whole world space by its own `run()`.
  3. THIS FILE         owns (a) the LAUNCH SHIM that turns one census launch
                       into capture+replay, and (b) the ADAPTER that turns the
                       raw artefact into a `World` for layer 2.

★ WHY THIS FILE HAS NO `score()` OF ITS OWN (sec10-20, corrected in rev7).
  rev6 asked the harness to re-implement the rule and then check that the two
  implementations agree. Two copies of a rule cannot be kept in step, and the
  agreement check would have been the thing that breaks first. Instead the
  rule is imported and called ONCE, so agreement is structural -- and the
  entire remaining risk surface is the ADAPTER, which is what the round-4 Q2
  contract table (prereg sec10.2) now pins down and what `--selftest-adapter`
  mutation-tests. A check that cannot fail is an identity (lesson #53): the
  "harness agrees with the rule" check is deleted, not weakened, and replaced
  by "the adapter maps every world in the rule's space back to that world's
  label".

------------------------------------------------------------------------------
LEGS (sec3)
------------------------------------------------------------------------------
  leg                  stream                        execution   role
  -------------------- ----------------------------- ----------- --------------
  eager_plain          plain (no green ctx)          eager       control; D
  graph_plain          plain                         replay      null channel
  eager_green          green pair [1] = DECODE half  eager       baseline
  eager_green_prefill  green pair [0] = PREFILL half eager       attribution
  graph_green          green pair [1] = DECODE half  replay      ★UNDER TEST
  capture_green_replay_plain / capture_plain_replay_green        descriptive,
                       cross                          replay      failure OK

The DECODE half is used because that is exactly the stream the engine captures
decode graphs on. Each GREEN leg is censused TWICE (independent sweeps A and
B); `S(leg) := S_A ∪ S_B` and `Δ_split := S_A △ S_B` is a noise ruler
(sec8-22, sec5.1). The plain legs are censused once.

Usage:
    python p0a_graph_sm_confinement.py --selftest        # CPU, everything
    python p0a_graph_sm_confinement.py --selftest-adapter  # CPU, adapter only
    python p0a_graph_sm_confinement.py --run --tag T --outdir D   # needs a GPU
    python p0a_graph_sm_confinement.py --analyze RAW.json --tag T --outdir D
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time

# Layer 1: the census producer. Imported, never edited (sec9-4).
_HERE = os.path.dirname(os.path.abspath(__file__))
_SMID_DIR = os.path.join(os.path.dirname(_HERE), "smid_census")
sys.path.insert(0, _SMID_DIR)
sys.path.insert(0, _HERE)
import smid_l0_census as CEN  # noqa: E402
# Layer 2: the decision rule. Imported, never re-implemented.
import p0a_rule_totality as RULE  # noqa: E402

# P1's input: the census's own writer, bound as a module-level name so the CPU
# battery can exercise the very object `run()` calls, and so a source mutant
# can prove the line is load-bearing (2026-08-21 dead-path repair).
_record_runtime_ptx = CEN._record_runtime_ptx

# ★sec9 self-invalidation reference values (harness-layer audit G9, 2026-08-23).
#   sec9-3 and sec9-4 said "record it" and nothing held a reference, so neither
#   condition could ever fire. These two values come from R0 itself (job 889631,
#   the run this probe's green legs depend on) and are checked by `analyze()`,
#   which stamps a `substrate_mismatch` banner at the top of the verdict.
R0_COMPUTE_MODE = "Default"            # smidl0_889631.out:7
R0_CENSUS_SHA256 = ("1cee21918f34e7f1f207aaef0fc13d03717ea3850"
                    "0e9b937c7de3c49ae96a325")   # smid_l0_census.py as R0 ran it

# The five decision legs, in the registered order (sec3-D).
DECISION_LEGS = ("eager_plain", "graph_plain", "eager_green",
                 "eager_green_prefill", "graph_green")
GREEN_LEGS = ("eager_green", "eager_green_prefill", "graph_green")
CROSS_LEGS = ("capture_green_replay_plain", "capture_plain_replay_green")
# adapter key -> raw leg name (sec10.2, the Q2 contract table)
WORLD_LEG = {"ep": "eager_plain", "gp": "graph_plain", "eg": "eager_green",
             "egp": "eager_green_prefill", "gg": "graph_green"}
# ★sec8-22 / audit G8: green legs are censused TWICE, plain legs ONCE. The
#   asymmetry is registered rather than incidental -- P4c compares S(eg) u
#   S(egp) (2 sweeps each) against D = S(eager_plain) (1 sweep), so coverage
#   asymmetry lands straight on NOPAIR. The adapter checks the counts against
#   what the run recorded (audit G5) instead of trusting them.
SWEEPS_GREEN = 2
SWEEPS_PLAIN = 1
SWEEPS_CROSS = 1          # sec8-26: the two descriptive cross legs, one sweep
# Streams whose TRUE answer to "is a green context attached?" is known to be
# "no": they are built without one. At least one must report detached, or the
# read-out has not been shown to discriminate (producer `_greenctx_detached`).
PLAIN_CONTROL_STREAMS = ("plain_pre", "plain_post")
GREEN_STREAMS = ("green_prefill", "green_decode")


# ==========================================================================
# 1. Launch shim -- the ONLY thing this file owns on the measurement side
# ==========================================================================
class GraphLaunchShim:
    """Stands in for a Triton kernel in `CEN._census_target`'s launch slot.

    `_census_target` -> `_census_once` issues exactly one launch per (grid,
    repeat) as `kernel[(n_blocks,)](*args, num_warps=1)` and reads the output
    tensors afterwards (smid_l0_census.py:614-646). Substituting a shim for
    `kernel` therefore leaves the sweep, the ladder, the per-label hit counts
    and the union accounting ENTIRELY with the producer, and this file owns
    one line: how that launch happens (prereg sec10.1).

    Consequences that the prereg registers as free parameters:
      * one capture per (grid, repeat) = 25 per leg, because `_census_once`
        allocates fresh tensors on every call and a graph is bound to the
        addresses it captured (sec8-12);
      * warmup runs on a SIDE stream with SCRATCH buffers, one warmup per
        capture (sec8-18). Scratch buffers matter: a warmup that wrote into
        the leg's own census tensors would leave PLAIN-stream labels in them,
        and any block the replay failed to overwrite would then be read as an
        observation of this leg -- manufacturing an escape. The producer fills
        every census tensor with -1 at allocation, so untouched entries are
        skipped, which is why sec8-20 retires the old `fill_(-1)` step;
      * `pool=None` and `capture_error_mode="global"` (sec8-14, sec8-15). The
        engine captures into a SHARED pool; this probe does not, and sec6
        records that as a scope limit rather than papering over it.

    Failure is recorded, never raised: a capture or replay exception leaves
    the census tensors at -1 (contributing nothing) and marks the event, so
    the adapter can report `NOCAP`/`NOREPLAY` -- which are tool-availability
    facts, not confinement verdicts (sec5.3).
    """

    def __init__(self, kernel, capture_stream, replay_stream, dev):
        import torch
        self.kernel = kernel
        self.capture_stream = capture_stream
        self.replay_stream = replay_stream
        self.dev = dev
        self.side = torch.cuda.Stream(device=dev)
        self.events = []
        self._seq = {}

    def __getitem__(self, grid):
        import torch
        n_blocks = grid[0]

        def _launch(*args, **kwargs):
            self._seq[n_blocks] = self._seq.get(n_blocks, 0) + 1
            ev = {"n_blocks": n_blocks, "seq": self._seq[n_blocks],
                  # ★audit G3(b): warmup runs EAGER on a PLAIN side stream with
                  #   scratch buffers, so its failure says nothing about whether
                  #   a graph can be captured on a green stream. It used to be
                  #   written into `capture_exc` and was therefore reported as
                  #   "GRAPH CAPTURE UNAVAILABLE ON GREEN STREAM".
                  "warmup_ok": False,
                  "capture_ok": False, "replay_ok": False,
                  "warmup_exc": None,
                  "capture_exc": None, "replay_exc": None}
            self.events.append(ev)
            # --- warmup: JIT + lazy init must not happen inside capture, and
            #     must not touch this leg's census tensors.
            scratch = [torch.empty_like(a) if torch.is_tensor(a) else a
                       for a in args]
            try:
                with torch.cuda.stream(self.side):
                    self.kernel[grid](*scratch, **kwargs)
                self.side.synchronize()
                ev["warmup_ok"] = True
            except Exception as exc:  # noqa: BLE001
                ev["warmup_exc"] = repr(exc)
                return
            del scratch
            # --- capture on the capture stream
            g = torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(g, stream=self.capture_stream):
                    self.kernel[grid](*args, **kwargs)
                ev["capture_ok"] = True
            except Exception as exc:  # noqa: BLE001
                ev["capture_exc"] = repr(exc)
                return
            # --- replay on the replay stream (the same one, except in the two
            #     descriptive cross legs)
            try:
                with torch.cuda.stream(self.replay_stream):
                    g.replay()
                self.replay_stream.synchronize()
                ev["replay_ok"] = True
            except Exception as exc:  # noqa: BLE001
                ev["replay_exc"] = repr(exc)
            finally:
                del g

        return _launch

    def summary(self, expected):
        """Per-leg failure accounting -- consumed by `_diagnostics` (audit G3c).

        Until this existed, `capture_events` was written in four places and
        read in none, so the verdict `.json` carried the label `NOCAP` with no
        field saying which of three different events produced it.
        """
        first = lambda k: next(  # noqa: E731
            (e[k] for e in self.events if e.get(k)), None)
        return {"expected": expected, "launched": len(self.events),
                "warmed": sum(1 for e in self.events if e["warmup_ok"]),
                "captured": sum(1 for e in self.events if e["capture_ok"]),
                "replayed": sum(1 for e in self.events if e["replay_ok"]),
                "warmup_failed": sum(1 for e in self.events
                                     if not e["warmup_ok"]),
                "first_warmup_exc": first("warmup_exc"),
                "first_capture_exc": first("capture_exc"),
                "first_replay_exc": first("replay_exc")}

    def status(self, expected):
        """(capture_status, replay_status) over `expected` launches.

        `ok` requires EVERY expected (grid, repeat) pair to have succeeded --
        sec5 admits a graph-leg statement only when all 25 pairs captured.
        """
        cap = sum(1 for e in self.events if e["capture_ok"])
        rep = sum(1 for e in self.events if e["replay_ok"])
        if cap == expected:
            cap_s = "ok"
        elif cap == 0:
            cap_s = "fail"
        else:
            cap_s = "partial"
        rep_s = "ok" if rep == expected and cap_s == "ok" else (
            "fail" if rep == 0 else "partial")
        return cap_s, rep_s


def _n_launches():
    return len(CEN.GRID_SWEEP) * CEN.REPEATS_PER_GRID


# ==========================================================================
# 2. Legs
# ==========================================================================
def _leg_eager(kernel, stream, spin_ns, sweeps):
    """`sweeps` independent censuses of one stream, eager. Producer-owned."""
    return {"sweeps": [CEN._census_target(kernel, stream, spin_ns)
                       for _ in range(sweeps)],
            "mode": "eager"}


def _leg_graph(kernel, capture_stream, replay_stream, spin_ns, dev, sweeps,
               capture_role=None, replay_role=None):
    """Same, but every launch is a capture+replay through the shim.

    ★`capture_role`/`replay_role` are the caller's own names for the two
    streams, written into the leg (audit E1). H5 tried to protect the
    name<->stream pairing with an AST check and failed: the dict form that
    replaced `zip` is keyed by POSITION (`CROSS_LEGS[0]`), so `.items()` is
    element-identical to the zip it replaced, and reordering the constant still
    swaps the two descriptive legs silently -- measured, with A7c/A7d/A7e all
    passing. An AST check cannot see that. Recording the role at RUNTIME can:
    the artefact then says which stream each leg actually used, and the adapter
    asserts it.
    """
    out = {"sweeps": [], "capture_events": [], "capture_summary": [],
           "mode": "graph",
           "capture_stream_role": capture_role,
           "replay_stream_role": replay_role}
    cap_s, rep_s = [], []
    for _ in range(sweeps):
        shim = GraphLaunchShim(kernel, capture_stream, replay_stream, dev)
        out["sweeps"].append(CEN._census_target(shim, capture_stream, spin_ns))
        out["capture_events"].extend(shim.events)
        out["capture_summary"].append(shim.summary(_n_launches()))
        c, r = shim.status(_n_launches())
        cap_s.append(c)
        rep_s.append(r)
    # Worst status over the sweeps: a leg is only `ok` if every sweep was.
    rank = {"ok": 0, "partial": 1, "fail": 2}
    out["capture_status"] = max(cap_s, key=lambda s: rank[s])
    out["replay_status"] = max(rep_s, key=lambda s: rank[s])
    return out


# ==========================================================================
# 3. Raw artefact assembly -- ONE constructor, used by run() and the self-test
# ==========================================================================
def assemble_raw(meta, legs, driver, ptx_fields, cross=None,
                 descriptive_incomplete=False):
    """Build the raw artefact. Pure; no I/O, no CUDA.

    ★This function exists so the CPU self-test can assert that the artefact a
    GPU run WOULD write is scorable -- the S-6 F2 failure mode (a campaign
    whose legs are structurally never adopted) is a plumbing defect that is
    invisible if `run()` assembles its dict inline.
    """
    raw = dict(meta)
    raw["kind"] = "bcg_p0a_raw"
    raw["legs"] = legs
    raw["driver_readout"] = driver
    # sec11 / gate #56: keep the producer's own NEGATIVE-form field verbatim
    # (the negative control is driven off it) AND publish a POSITIVE-form
    # derivation whose name means what its value says.
    streams = (driver or {}).get("streams") or {}
    raw["green_ctx_attached"] = {
        k: (v.get("green_ctx_is_null") is False) if isinstance(v, dict) else None
        for k, v in streams.items()}
    raw.update(ptx_fields or {})
    raw["cross_legs"] = cross or {}
    raw["descriptive_legs_incomplete"] = bool(descriptive_incomplete)
    return raw


def _flush(raw, outdir, tag):
    path = os.path.join(outdir, f"p0a_raw_{tag}.json")
    tmp = path + ".part"
    with open(tmp, "w") as f:
        json.dump(raw, f, indent=2)
    os.replace(tmp, path)
    return path


# ==========================================================================
# 4. GPU run
# ==========================================================================
def run(outdir, tag, spin_ns):
    import torch
    if spin_ns > CEN.SPIN_NS_CAP:
        raise SystemExit(
            f"spin_ns {spin_ns} exceeds the census prereg cap {CEN.SPIN_NS_CAP}")
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device; --run requires a GPU (this is a "
                         "MEASUREMENT ABSENT condition, not a result)")
    if not RULE.PRODUCER_OK:
        raise SystemExit(
            "the decision rule could not import its constants from the "
            f"producer ({RULE.PRODUCER_ERR}); refusing to run (sec14 D8)")

    from sglang.srt.multiplex import pdmux_context as pdc
    from sgl_kernel import spatial

    census_kernel, _, _ = CEN._kernels()
    dev = torch.cuda.current_device()
    cc = torch.cuda.get_device_capability(dev)
    granularity = pdc.get_arch_constraints(cc)[1]
    # ★audit N12: `RULE.GRANULARITY` is imported for a hardwired cc (8,0), and
    #   cc (8,6) also yields 2 -- so the granularity check alone does not pin
    #   the substrate. The partition and R0 are A100-specific (sec6), so pin it.
    if tuple(cc) != (8, 0):
        raise SystemExit(
            f"compute capability {tuple(cc)} is not the registered substrate "
            "(8, 0); R0 and the division under test are A100-specific "
            "(sec6/sec9) -- refusing to run")
    if granularity != RULE.GRANULARITY:
        raise SystemExit(
            f"granularity from the producer for cc={cc} is {granularity} but "
            f"the rule was built with {RULE.GRANULARITY}; sec9-5 self-"
            "invalidation -- refusing to run")
    total = spatial.get_sm_available(dev)
    divisions = pdc.divide_sm(total, cc, 4 - 2)  # sm_group_num = 4
    p_sm, d_sm = divisions[0]

    meta = {
        "tag": tag,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "device_name": torch.cuda.get_device_name(dev),
        "torch_version": torch.__version__,
        "compute_capability": list(cc),
        # DESCRIPTIVE ONLY (sec2.3, sec10-7): the scorer is forbidden from
        # reading this by an AST guard, it is here for the report.
        "total_sm_reported": total,
        "divisions_from_divide_sm": [list(x) for x in divisions],
        "division_under_test": [p_sm, d_sm],
        "granularity": granularity,
        "min_hits_floor": CEN.MIN_HITS_REPORTED,
        "spin_ns": spin_ns,
        "grid_sweep": list(CEN.GRID_SWEEP),
        "repeats_per_grid": CEN.REPEATS_PER_GRID,
        "sweeps_per_green_leg": SWEEPS_GREEN,
        "sweeps_per_plain_leg": SWEEPS_PLAIN,
        "prereg": "PREREG_P0A_2026-08-22.md",
        "rule_module_sha256": CEN._sha256(
            os.path.join(_HERE, "p0a_rule_totality.py")),
        "harness_sha256": CEN._sha256(os.path.abspath(__file__)),
        # gate #33: critical-path files outside sync_engine_tree.sh's manifest
        "sha256_unmanifested": {
            "pdmux_context.py": CEN._sha256(pdc.__file__),
            "sgl_kernel/spatial.py": CEN._sha256(spatial.__file__),
            "smid_l0_census.py": CEN._sha256(
                os.path.join(_SMID_DIR, "smid_l0_census.py"))},
        "provenance": CEN._provenance(),
        "nvidia_smi_compute_mode": _compute_mode(),   # sec9-3
    }

    legs, ptx_fields, cross = {}, {}, {}
    driver = {}

    def flush(descriptive_incomplete=False):
        raw = assemble_raw(meta, legs, driver, ptx_fields, cross,
                           descriptive_incomplete)
        p = _flush(raw, outdir, tag)
        print(f"[run] raw -> {p}")
        return raw

    # ---- leg 1: eager_plain (defines D)
    plain = torch.cuda.Stream(device=dev)
    legs["eager_plain"] = _leg_eager(census_kernel, plain, spin_ns,
                                 SWEEPS_PLAIN)
    # P1, on the object that actually ran. Scope (sec4 N6): this checks the
    # variant compiled by the eager_plain leg.
    _record_runtime_ptx(ptx_fields, census_kernel, dev, outdir, f"p0a_{tag}")
    ptx_fields["runtime_ptx_scope"] = "variant compiled during eager_plain"
    flush()

    # ---- leg 2: graph_plain (the null channel)
    legs["graph_plain"] = _leg_graph(census_kernel, plain, plain, spin_ns,
                                     dev, SWEEPS_PLAIN,
                                     capture_role="plain",
                                     replay_role="plain")  # pre-green: no map yet
    flush()

    # ---- green pair, built exactly as initialize_stream_groups does
    green = spatial.create_greenctx_stream_by_value(p_sm, d_sm, dev)
    g_prefill, g_decode = green[0], green[1]
    plain_post = torch.cuda.Stream(device=dev)

    # ★audit E1/F3. The role a leg reports must be DERIVED from the stream
    #   object it was handed, never written by hand beside it. Hand-written
    #   labels are exactly what H5 got wrong: swap the streams and the labels
    #   keep saying the old thing, so the artefact lies with nothing to catch
    #   it. Identity lookup makes a stream swap swap the labels too, which
    #   `_cross_roles_ok` then rejects.
    _roles = {id(g_decode): "green_decode", id(g_prefill): "green_prefill",
              id(plain): "plain", id(plain_post): "plain_post"}

    def role_of(st):
        return _roles.get(id(st), "UNKNOWN")
    # sec8-19: read the driver IMMEDIATELY after creation, for the green pair
    # AND for plain streams -- without the latter the negative control cannot
    # be evaluated at all.
    driver = CEN.driver_readout({
        "green_prefill": g_prefill.cuda_stream,
        "green_decode": g_decode.cuda_stream,
        "plain_pre": plain.cuda_stream,
        "plain_post": plain_post.cuda_stream})
    flush()

    # ---- legs 3-5
    legs["eager_green"] = _leg_eager(census_kernel, g_decode, spin_ns,
                                 SWEEPS_GREEN)
    flush()
    legs["eager_green_prefill"] = _leg_eager(census_kernel, g_prefill,
                                             spin_ns, SWEEPS_GREEN)
    flush()
    legs["graph_green"] = _leg_graph(census_kernel, g_decode, g_decode,
                                     spin_ns, dev, SWEEPS_GREEN,
                                     capture_role=role_of(g_decode),
                                     replay_role=role_of(g_decode))
    raw = flush()
    print("[run] decision legs complete and flushed; descriptive legs follow "
          "(their failure does NOT block scoring -- sec5.3)")

    # ---- descriptive cross legs: failure allowed (sec3-C, sec5.3)
    incomplete = False
    # ★audit H5 -> F1/F3. Reordering CROSS_LEGS would silently swap the two
    #   descriptive legs, and those two are the probe's most interesting
    #   descriptive output. The rev8 repair -- a dict keyed by CROSS_LEGS[i] --
    #   did NOT fix it: keyed by POSITION, `.items()` is element-identical to
    #   the `zip` it replaced, and the audit reproduced the swap with every AST
    #   check passing. What actually closes it is below: the role each leg
    #   reports is DERIVED from the stream object via `role_of()`, so swapping
    #   the streams swaps the recorded roles and `_cross_roles_ok` rejects them.
    cross_spec = {CROSS_LEGS[0]: (g_decode, plain),
                  CROSS_LEGS[1]: (plain, g_decode)}
    for name, (cap_stream, rep_stream) in cross_spec.items():
        try:
            cross[name] = _leg_graph(census_kernel, cap_stream, rep_stream,
                                     spin_ns, dev, SWEEPS_CROSS,
                                     capture_role=role_of(cap_stream),
                                     replay_role=role_of(rep_stream))
        except Exception as exc:  # noqa: BLE001
            cross[name] = {"failed": repr(exc)}
            incomplete = True
        else:
            # ★audit N8: `descriptive_legs_incomplete` used to be set ONLY on
            #   an exception, so a cross leg that captured 0 of 25 pairs
            #   without raising was reported as complete.
            if (cross[name].get("capture_status") != "ok"
                    or cross[name].get("replay_status") != "ok"):
                incomplete = True
    flush(incomplete)

    print("[run] NO VERDICT IS PRODUCED HERE. Score with --analyze.")
    return 0


def _compute_mode():
    """sec9-3: record it; a difference from job 889631 goes on the verdict."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_mode", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30)
        return out.stdout.strip() or f"rc={out.returncode}"
    except Exception as exc:  # noqa: BLE001
        return f"unavailable: {exc!r}"


# ==========================================================================
# 5. ADAPTER -- raw artefact -> World (prereg sec10.2, the Q2 contract table)
# ==========================================================================
# Every rule below is fail-closed: a missing, malformed or absent field must
# land on the value that produces a NON-substantive label. The mapping is
# mechanised in `--selftest-adapter` (round-trip over the rule's whole world
# space) and mutation-tested (each contract line has a mutant that breaks it).

def _sweeps(leg):
    """The producer-shaped sweeps of one leg, or [] if the leg is unusable."""
    if not isinstance(leg, dict):
        return []
    sw = leg.get("sweeps")
    return [s for s in sw if isinstance(s, dict)] if isinstance(sw, list) else []


def _leg_set(leg):
    """S(leg) := union over ALL sweeps (sec8-22: A ∪ B, frozen)."""
    s = set()
    for sweep in _sweeps(leg):
        u = sweep.get("union")
        if isinstance(u, list):
            s |= {v for v in u if isinstance(v, int)}
    return s


def _leg_sweep_sets(leg):
    """Per-sweep label sets. ★audit N4: this used to accept any iterable while
    `_leg_set` required a list, so a malformed sweep could be seen differently
    by S(gg) and by the replication test. Both fail toward the cheaper label,
    but they should not disagree about what the data is."""
    out = []
    # ★the loop variable is deliberately NOT named `sweep`: `_leg_set`'s first
    #   two lines would then be byte-identical to these, and the
    #   `sweep_A_only` mutation anchor (which must hit `_leg_set` alone) would
    #   stop being unique. Measured -- it did, and the mutant reported "the
    #   harness is stale". Do not "tidy" this back.
    for one in _sweeps(leg):
        u = one.get("union")
        out.append({v for v in u if isinstance(v, int)}
                   if isinstance(u, list) else set())
    return out


def _leg_min_hits(leg):
    """MIN over sweeps: a leg is only as positive as its weakest census."""
    sw = _sweeps(leg)
    if not sw:
        return 0
    vals = []
    for s in sw:
        m = s.get("min_hits")
        vals.append(m if isinstance(m, int) and not isinstance(m, bool) else 0)
    return min(vals)


def _leg_saturated(leg):
    """EVERY sweep must satisfy the producer's set-equality ladder test."""
    sw = _sweeps(leg)
    if not sw:
        return False
    for s in sw:
        ok, _why = CEN._saturated(s)
        if not ok:
            return False
    return True


def _leg_hits(leg):
    """label -> total hits across sweeps (descriptive; sec5.1 rank ruler)."""
    tot = {}
    for s in _sweeps(leg):
        for k, v in (s.get("hits") or {}).items():
            try:
                tot[int(k)] = tot.get(int(k), 0) + int(v)
            except (TypeError, ValueError):
                continue
    return tot


def _tool_status(leg, field):
    """`ok` only if the leg says so explicitly; anything else fails closed.

    ★audit G3(a): a leg that is not in the artefact at all is `absent`, not
    `fail`. The two are different events -- an absent leg means the run was
    killed before it wrote that leg (incremental flush makes a truncated
    artefact a NORMAL product), while `fail` means the leg ran and the tool
    refused. Collapsing them reported a wall-clock kill as "graph capture
    unavailable on the green stream".
    """
    if not isinstance(leg, dict):
        return "absent"
    v = leg.get(field)
    return v if v in ("ok", "partial", "fail") else "absent"


def _attached(raw):
    """P4a: BOTH green streams attached AND at least one control detached.

    The asymmetry is the producer's (sec4 P4a / C4): a census taken on a
    stream with no green context is not the measurement at all, while ONE
    control reporting detached is enough to show the read-out can say "no".
    A read-out that answers the same way everywhere carries no information.
    """
    pos = CEN._greenctx_attached(raw, GREEN_STREAMS)
    neg = CEN._greenctx_detached(raw, PLAIN_CONTROL_STREAMS)
    return (not pos["blocked"]) and bool(neg["detached"]), pos, neg


def _instrument(raw):
    """P1: did the kernel that ran keep the %smid site and the spin back edge?"""
    sites = raw.get("runtime_ptx_smid_sites")
    edge = raw.get("runtime_spin_back_edge")
    return bool(isinstance(sites, int) and sites > 0 and edge is True)


def _escape_replicated(raw):
    """sec5 F4: did the SAME attributed escape appear in BOTH graph sweeps?

    ★audit G1 -- this used to be `all(non-empty)`, i.e. "each sweep saw SOME
    escaping label". Noise's cheapest form is one stray label per sweep, and
    that form made `per = [{40}, {41}]` satisfy the predicate, earning the
    probe's most expensive verdict (`LOST (PARTIAL)` -> immediate canon
    referral) -- while the SAME label seen in only one sweep earned the
    cheaper `LOST (UNREPLICATED)`. A less reproducible observation was being
    punished harder. Rounds 3-5 killed "one noisy label -> most expensive
    verdict" three times in the rule layer; it had moved to the replication
    layer.

    The registered predicate is now INTERSECTION: the same label must escape
    twice. The old `any` form is kept as a DESCRIPTIVE field so the two can be
    read side by side, and sec5 / sec8-25 now state the same predicate (they
    did not before -- the prose said "in only one sweep" and the formula said
    "both non-empty", which are different rules).

    The baseline stays frozen at S(eg) = A ∪ B (sec8-22); only the graph leg's
    sweep varies. Fail-closed: fewer than two sweeps is NOT the expensive
    verdict.
    """
    legs = raw.get("legs") or {}
    base = _leg_set(legs.get("eager_green"))
    pre = _leg_set(legs.get("eager_green_prefill"))
    per = [((x - base) & pre) for x in _leg_sweep_sets(legs.get("graph_green"))]
    same = set.intersection(*per) if len(per) >= 2 else set()
    detail = {"per_sweep": [sorted(x) for x in per],
              "replicated_same": sorted(same),
              "replicated_any (DESCRIPTIVE, NOT THE DECISION)":
                  bool(per) and len(per) >= 2 and all(bool(x) for x in per)}
    return (len(per) >= 2 and bool(same)), detail


# Expected (capture, replay) roles per descriptive cross leg -- the names are
# the contract, and this is what makes them checkable at runtime (audit E1/F3).
CROSS_ROLES = {"capture_green_replay_plain": ("green_decode", "plain"),
               "capture_plain_replay_green": ("plain", "green_decode")}


def _cross_roles_ok(name, leg):
    """Did the leg actually use the streams its NAME claims? (audit E1)

    Returns True/False, or None when the run did not record roles (an older
    artefact). Descriptive only -- these legs carry no decision rule -- but
    without it a swapped stream pair would invert the meaning of the two most
    interesting descriptive numbers with nothing in the artefact to show it.
    """
    want = CROSS_ROLES.get(name)
    if want is None or not isinstance(leg, dict):
        return None
    got = (leg.get("capture_stream_role"), leg.get("replay_stream_role"))
    if got == (None, None):
        return None
    return got == want


def _attach_failure_accounting(rec, legs, raw):
    """Carry the tool-failure account onto the EARLY-RETURN paths (audit H2).

    `score()` skips `_diagnostics` when the adapter returns no world, so when
    one of the pre-World gates fired -- i.e. exactly when a TOOL failure was
    the cause -- the tool-failure account vanished from the verdict. That is
    G3(c)'s defect returning by a new and smaller route.
    """
    rec["graph_leg_failures"] = {
        name: (legs.get(name) or {}).get("capture_summary")
        for name in ("graph_plain", "graph_green")
        if isinstance(legs.get(name), dict)}
    rec["instrument_alive"] = _instrument(raw)
    return rec


def world_from_raw(raw):
    """raw artefact -> (World, adapter_record) or (None, adapter_record).

    ★The Q2 contract, in one place. Each line is a fail-closed mapping and is
    covered by a mutant in `ADAPTER_MUTANTS`.
    """
    rec = {"contract": "PREREG_P0A sec10.2"}
    if not isinstance(raw, dict):
        rec["why"] = "raw artefact is not an object"
        return None, rec
    div = raw.get("division_under_test")
    if not (isinstance(div, list) and len(div) == 2
            and all(isinstance(x, int) for x in div)):
        rec["why"] = ("the division under test is absent from the artefact, so "
                      "the producer's target for P4b is unknown; nothing about "
                      "the green legs can be read")
        return None, rec
    gran = raw.get("granularity")
    if gran != RULE.GRANULARITY:
        rec["why"] = (f"granularity recorded by the run ({gran!r}) differs from "
                      f"the one the rule was built with ({RULE.GRANULARITY}); "
                      "the artefact and the rule are not about the same "
                      "substrate")
        return None, rec
    legs = raw.get("legs") if isinstance(raw.get("legs"), dict) else {}

    # --- ★pre-World measurement gates (harness-layer audit G2/G3a/G5).
    #     These run BEFORE the rule because they answer "was this artefact
    #     produced by the registered run at all", which no `World` field can
    #     express. Every one of them is fail-closed and lands on ABSENT with a
    #     `why` that names the leg -- never on a substantive label, and never
    #     on NOCAP (whose registered text is specifically about the GREEN
    #     stream). Order: sweep counts -> plain graph leg -> leg presence.
    want = {**{v: SWEEPS_GREEN for k, v in WORLD_LEG.items()
               if v in GREEN_LEGS},
            **{v: SWEEPS_PLAIN for k, v in WORLD_LEG.items()
               if v not in GREEN_LEGS}}
    for name, n_want in want.items():
        if name in legs and len(_sweeps(legs.get(name))) != n_want:
            rec["why"] = (f"leg {name} carries {len(_sweeps(legs.get(name)))} "
                          f"sweep(s), the registered count is {n_want} "
                          "(sec8-22). S(leg) := union over sweeps, so a "
                          "different count silently changes coverage, the "
                          "replication test and the split-half ruler")
            _attach_failure_accounting(rec, legs, raw)
            return None, rec
    for key, n_want in (("sweeps_per_green_leg", SWEEPS_GREEN),
                        ("sweeps_per_plain_leg", SWEEPS_PLAIN)):
        if raw.get(key) != n_want:
            rec["why"] = (f"the run recorded {key}={raw.get(key)!r} but the "
                          f"registered value is {n_want}; the artefact was not "
                          "produced by the registered schedule")
            _attach_failure_accounting(rec, legs, raw)
            return None, rec
    # ★G2: graph_plain is a REPLAY leg too. Its capture failing narrows S(gp),
    #   and the rule then reports NULL CHANNEL OPEN (a diagnostic conclusion
    #   about the decision function) or MEASUREMENT ABSENT -- neither of which
    #   says "the plain-stream capture only partly succeeded", which is what
    #   the artefact already knew.
    gp_cap = _tool_status(legs.get("graph_plain"), "capture_status")
    gp_rep = _tool_status(legs.get("graph_plain"), "replay_status")
    if gp_cap != "ok" or gp_rep != "ok":
        verb = ("is absent from the artefact" if "absent" in (gp_cap, gp_rep)
                else "did not complete")
        rec["why"] = (f"the plain-stream graph leg {verb} "
                      f"(capture={gp_cap}, replay={gp_rep}); its census cannot "
                      "be read as the null channel of the decision function. "
                      "This is a tool fact about the PLAIN stream and says "
                      "nothing about capture on a green-context stream")
        rec["tool_status_by_leg"] = {"graph_plain": [gp_cap, gp_rep]}
        _attach_failure_accounting(rec, legs, raw)
        return None, rec

    sets = {k: _leg_set(legs.get(v)) for k, v in WORLD_LEG.items()}
    cap = _tool_status(legs.get("graph_green"), "capture_status")
    rep = _tool_status(legs.get("graph_green"), "replay_status")
    # ★G3(a): an ABSENT leg is a truncated artefact (incremental flush makes
    #   that a normal product of a wall-clock kill), not a refusal by the
    #   green stream. Only `fail`/`partial` may reach NOCAP/NOREPLAY.
    if "absent" in (cap, rep):
        rec["why"] = ("the graph_green leg is missing from the artefact "
                      "(capture/replay status absent). The run did not reach "
                      "it -- this is not evidence about whether a graph can "
                      "be captured on a green-context stream")
        rec["tool_status_by_leg"] = {"graph_green": [cap, rep]}
        _attach_failure_accounting(rec, legs, raw)
        return None, rec
    att, pos, neg = _attached(raw)
    repl, repl_detail = _escape_replicated(raw)
    w = RULE.World(
        sets["ep"], sets["gp"], sets["eg"], sets["gg"], S_egp=sets["egp"],
        d_sm=div[1],
        min_hits={k: _leg_min_hits(legs.get(v)) for k, v in WORLD_LEG.items()},
        saturated={k: _leg_saturated(legs.get(v)) for k, v in WORLD_LEG.items()},
        attached=att, capture=cap, replay=rep,
        instrument=_instrument(raw), escape_replicated=repl)
    rec.update({
        "set_sizes": {k: len(v) for k, v in sets.items()},
        "min_hits": dict(w.min_hits), "saturated": dict(w.saturated),
        "capture_status": cap, "replay_status": rep,
        "tool_status_by_leg": {"graph_plain": [gp_cap, gp_rep],
                               "graph_green": [cap, rep]},
        "attachment_positive": pos, "attachment_negative": neg,
        "attached": att, "instrument_alive": w.instrument,
        "escape_replicated": repl, "escape_replication": repl_detail,
        # ★audit H7: `per_sweep` already carries the values, but the union
        #   SIZE is what separates "one stray label per sweep" from "a large
        #   escape whose sweeps happen to be disjoint" at a glance. Reported,
        #   never a threshold (sec5.1).
        "escape_union_size": len(set().union(*[set(x) for x in
                                               repl_detail["per_sweep"]])
                                 if repl_detail["per_sweep"] else set()),
        "d_sm_target": div[1], "granularity": gran})
    return w, rec


# ==========================================================================
# 6. Scoring -- the rule is CALLED, not re-implemented
# ==========================================================================
def score(raw):
    """raw artefact dict -> verdict dict. Pure. No target-layer reads."""
    out = {"kind": "bcg_p0a_verdict",
           "rule": "p0a_rule_totality.score (imported, not re-implemented)",
           "prereg": "PREREG_P0A_2026-08-22.md"}
    w, rec = world_from_raw(raw)
    out["adapter"] = rec
    if w is None:
        out["verdict"] = RULE.ABSENT
        out["why"] = rec.get("why", "the adapter could not build a world")
        out["scope"] = _SCOPE
        return out
    out["verdict"] = RULE.score(w)
    out["why"] = _why(out["verdict"], w, rec)
    out.update(_diagnostics(w, raw))
    out["scope"] = _SCOPE
    return out


_SCOPE = (
    "TOOL-VALIDITY / LABEL-IDENTITY ONLY. %smid is a globally consistent "
    "label, not an established physical SM index, and its set cardinalities "
    "are not compute-resource fractions. This probe says nothing about kernel "
    "efficiency, occupancy, wave quantization, the high-SM decode flattening "
    "mechanism, or any policy ranking. No model, no request, no server, no "
    "shared graph memory pool, one division (the decode half of "
    "divide_sm(...)[0]) -- do not generalise to d16/d44 or to the cudagraph-ON "
    "serving operating point. A PRESERVED verdict closes no gate.")


def _why(verdict, w, rec):
    E = w.S["gg"] - w.S["eg"]
    E_rev = w.S["eg"] - w.S["gg"]
    return (f"{verdict}: |S(graph_green)|={len(w.S['gg'])}, "
            f"|S(eager_green)|={len(w.S['eg'])}, escape |E|={len(E)}, "
            f"attributed |E∩S(prefill half)|={len(E & w.S['egp'])}, "
            f"shortfall |E_rev|={len(E_rev)}, "
            f"target d_sm={w.d_sm}±{RULE.GRANULARITY}, "
            f"capture={rec.get('capture_status')}, "
            f"replay={rec.get('replay_status')}, attached={rec.get('attached')}")


def _rank(hits, label):
    """Where `label` sits in its leg's hit distribution (sec5.1 noise ruler).

    Not a threshold: a label in the extreme lower tail leaves the "coverage
    noise" reading alive, one near the middle kills it. Reported, never gated.
    """
    if label not in hits or not hits:
        return None
    vals = sorted(hits.values())
    below = sum(1 for v in vals if v < hits[label])
    return {"hits": hits[label], "rank": below + 1, "of": len(vals),
            "median": vals[len(vals) // 2]}


def _diagnostics(w, raw):
    """sec5.1: mandatory non-identity reporting. No thresholds, no tests."""
    legs = raw.get("legs") if isinstance(raw.get("legs"), dict) else {}
    E = w.S["gg"] - w.S["eg"]
    E_rev = w.S["eg"] - w.S["gg"]
    E_attrib = E & w.S["egp"]
    hits = {k: _leg_hits(legs.get(v)) for k, v in WORLD_LEG.items()}
    split = {}
    for k in ("eg", "gg", "egp"):
        ss = _leg_sweep_sets(legs.get(WORLD_LEG[k]))
        split[k] = sorted(ss[0] ^ ss[1]) if len(ss) >= 2 else None
    delta = sorted(E | E_rev)
    return {
        "D_size": len(w.S["ep"]),
        "escape": sorted(E), "escape_attributed": sorted(E_attrib),
        "shortfall": sorted(E_rev), "delta": delta,
        "graph_green_covers_D": w.S["gg"] >= w.S["ep"],
        "exact_set_match": w.S["gg"] == w.S["eg"],
        "null_channel": {"graph_plain_minus_eager_plain":
                         sorted(w.S["gp"] - w.S["ep"]),
                         "eager_plain_minus_graph_plain":
                         sorted(w.S["ep"] - w.S["gp"])},
        "green_pair_disjoint": not (w.S["eg"] & w.S["egp"]),
        "green_pair_tiles_D": (w.S["eg"] | w.S["egp"]) == w.S["ep"],
        "min_hits_by_leg": dict(w.min_hits),
        "delta_split_by_leg": split,
        "delta_size_vs_split_size": {
            "delta": len(delta),
            "split": {k: (len(v) if v is not None else None)
                      for k, v in split.items()}},
        "delta_label_ranks": {
            str(x): {"graph_green": _rank(hits["gg"], x),
                     "eager_green": _rank(hits["eg"], x),
                     "prefill_half": _rank(hits["egp"], x)}
            for x in delta},
        "descriptive_legs_incomplete": bool(
            raw.get("descriptive_legs_incomplete")),
        "cross_legs": {
            k: {"union_sizes": [len(x) for x in _leg_sweep_sets(v)],
                "capture_status": _tool_status(v, "capture_status"),
                "replay_status": _tool_status(v, "replay_status"),
                # ★audit E1: the RUNTIME roles, so a reader of the verdict can
                #   see which stream each descriptive leg captured on and
                #   replayed on instead of trusting the leg's name.
                "capture_stream_role": (v or {}).get("capture_stream_role"),
                "replay_stream_role": (v or {}).get("replay_stream_role"),
                "roles_match_name": _cross_roles_ok(k, v)}
            for k, v in (raw.get("cross_legs") or {}).items()},
        # ★audit G3(c): `capture_events` used to be written in four places and
        #   read in none, so a verdict said NOCAP with no field naming which of
        #   three different events caused it. sec11 requires every field of the
        #   verdict to be in the .json.
        "graph_leg_failures": {
            name: (legs.get(name) or {}).get("capture_summary")
            for name in ("graph_plain", "graph_green")
            if isinstance(legs.get(name), dict)},
    }


def analyze(raw_path, outdir, tag):
    with open(raw_path) as f:
        raw = json.load(f)
    v = score(raw)
    # ★sec9-3 / sec9-4, made enforceable (harness-layer audit G9). Both
    #   conditions previously said "record it" and nothing held a reference,
    #   so neither could ever fire. A mismatch does NOT invalidate the verdict
    #   by itself -- sec9-3 says "record it in the raw and put it at the top of
    #   the verdict" -- so this is a banner, not a gate.
    mism = {}
    mode = raw.get("nvidia_smi_compute_mode")
    if mode is not None and str(mode).strip() != R0_COMPUTE_MODE:
        mism["compute_mode"] = {"this_run": mode, "R0_job_889631":
                                R0_COMPUTE_MODE}
    got = (raw.get("sha256_unmanifested") or {}).get("smid_l0_census.py")
    if got is not None and got != R0_CENSUS_SHA256:
        mism["smid_l0_census.py_sha256"] = {"this_run": got,
                                            "as_R0_ran_it": R0_CENSUS_SHA256}
    if mism:
        v = {"substrate_mismatch": mism,
             "substrate_mismatch_meaning":
                 "this run was taken on a substrate that differs from the one "
                 "R0 (job 889631) established the green legs on; sec9-2/9-4 "
                 "govern what may still be said", **v}
    v["raw_path"] = os.path.abspath(raw_path)
    v["raw_sha256"] = CEN._sha256(raw_path)
    for k in ("tag", "host", "slurm_job_id", "device_name", "utc",
              "compute_capability", "nvidia_smi_compute_mode", "provenance",
              "sha256_unmanifested", "harness_sha256", "rule_module_sha256",
              "division_under_test", "granularity", "grid_sweep",
              "repeats_per_grid", "sweeps_per_green_leg",
              "sweeps_per_plain_leg", "spin_ns",
              "total_sm_reported", "divisions_from_divide_sm",
              "green_ctx_attached", "driver_readout"):
        if k in raw:
            v.setdefault("run_" + k, raw[k])
    path = os.path.join(outdir, f"p0a_verdict_{tag}.json")
    with open(path, "w") as f:
        json.dump(v, f, indent=2)
    print("★ THIS TEXT IS NOT CITABLE (gate #56). Cite "
          f"p0a_verdict_{tag}.json, which holds every field in full.")
    print(json.dumps({k: v[k] for k in ("verdict", "why") if k in v}, indent=2))
    print(f"[analyze] verdict -> {path}")
    return 0


# ==========================================================================
# 7. Fixtures -- a raw artefact for any world in the rule's space
# ==========================================================================
# ★This is the inverse of the adapter, and it is what makes sec10-20 a real
#   check instead of an identity: for EVERY world the rule enumerates, build
#   the artefact a run in that world would have written, push it through the
#   adapter, and demand the label the rule gives that world.

_SWEEP_CACHE = {}


def _fake_sweep(S, sat=True, min_hits=None):
    """One `_census_target`-shaped sweep. Producer schema, checked by A2."""
    key = (frozenset(S), sat, min_hits)
    if key in _SWEEP_CACHE:
        return _SWEEP_CACHE[key]
    ids = sorted(S)
    mh = (CEN.MIN_HITS_REPORTED if ids else 0) if min_hits is None else min_hits
    if sat or len(ids) == 0:
        # saturated: the last two grid points have identical union SETS
        lad = [{"n_blocks": n, "union": list(ids), "union_size": len(ids),
                "nsmid_observed": []} for n in CEN.GRID_SWEEP]
    else:
        # unsaturated: "still growing" at the last grid point
        grown = list(ids[:-1])
        lad = [{"n_blocks": n, "union": list(grown), "union_size": len(grown),
                "nsmid_observed": []} for n in CEN.GRID_SWEEP[:-1]]
        lad.append({"n_blocks": CEN.GRID_SWEEP[-1], "union": list(ids),
                    "union_size": len(ids), "nsmid_observed": []})
    out = {"ladder": lad, "union": list(ids),
           "hits": {str(x): max(mh, 1) for x in ids}, "min_hits": mh}
    _SWEEP_CACHE[key] = out
    return out


def _fake_leg(sets, sat=True, min_hits=None, mode="eager",
              capture=None, replay=None):
    leg = {"mode": mode,
           "sweeps": [_fake_sweep(s, sat, min_hits) for s in sets]}
    if capture is not None:
        leg["capture_status"] = capture
        leg["replay_status"] = replay
        leg["capture_events"] = []
    return leg


def _fake_driver(green_attached=True, control_detached=True):
    def e(is_null):
        return {"rc": 0, "green_ctx_is_null": is_null}
    streams = {k: e(not green_attached) for k in GREEN_STREAMS}
    streams.update({k: e(bool(control_detached))
                    for k in PLAIN_CONTROL_STREAMS})
    return {"streams": streams}


def raw_from_world(w, green_attached=None, control_detached=True):
    """The artefact a run in world `w` would have written."""
    S = w.S
    e_attrib = (S["gg"] - S["eg"]) & S["egp"]
    gg_sweeps = ([S["gg"], S["gg"]] if w.escape_replicated
                 else [S["gg"], S["gg"] - e_attrib])
    legs = {
        "eager_plain": _fake_leg([S["ep"]], w.saturated["ep"],
                                 w.min_hits["ep"]),
        "graph_plain": _fake_leg([S["gp"]], w.saturated["gp"],
                                 w.min_hits["gp"], mode="graph",
                                 capture="ok", replay="ok"),
        "eager_green": _fake_leg([S["eg"], S["eg"]], w.saturated["eg"],
                                 w.min_hits["eg"]),
        "eager_green_prefill": _fake_leg([S["egp"], S["egp"]],
                                         w.saturated["egp"],
                                         w.min_hits["egp"]),
        "graph_green": _fake_leg(gg_sweeps, w.saturated["gg"],
                                 w.min_hits["gg"], mode="graph",
                                 capture=w.capture, replay=w.replay),
    }
    att = w.attached if green_attached is None else green_attached
    meta = {"tag": "fixture", "division_under_test": [108 - w.d_sm, w.d_sm],
            "granularity": RULE.GRANULARITY,
            "min_hits_floor": CEN.MIN_HITS_REPORTED,
            "total_sm_reported": 108,
            "sweeps_per_green_leg": SWEEPS_GREEN,
            "sweeps_per_plain_leg": SWEEPS_PLAIN}
    ptx = {"runtime_ptx_smid_sites": 1 if w.instrument else 0,
           "runtime_spin_back_edge": bool(w.instrument)}
    return assemble_raw(meta, legs, _fake_driver(att, control_detached), ptx)


def _healthy():
    """The world a clean run is expected to produce: PRESERVED."""
    return RULE.World(RULE.FULL, RULE.FULL, RULE.GREEN, RULE.GREEN,
                      S_egp=set(range(34, 108)))


# ==========================================================================
# 8. Self-tests
# ==========================================================================
def _producer_sweep_keys():
    """The key set `CEN._census_target` returns, read off the producer's AST.

    A fixture that drifts from the producer's schema would make every CPU
    check pass while the GPU run writes something the adapter cannot read --
    the defect shape that costs a whole campaign (S-6 F2). Read it, do not
    assume it.
    """
    import ast
    src = open(CEN.__file__, encoding="utf-8").read()
    tree = ast.parse(src)
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.name == "_census_target":
            for sub in ast.walk(n):
                if isinstance(sub, ast.Return) and isinstance(sub.value,
                                                              ast.Dict):
                    return {k.value for k in sub.value.keys
                            if isinstance(k, ast.Constant)}
    return set()


def _cross_loop_index_set(fn):
    """The CROSS_LEGS index set of the dict that actually drives the cross-leg
    loop, or None if the loop is not driven by such a dict (audit E2)."""
    import ast
    for node in ast.walk(fn):
        if not (isinstance(node, ast.For)
                and isinstance(node.iter, ast.Call)
                and isinstance(node.iter.func, ast.Attribute)
                and node.iter.func.attr == "items"
                and isinstance(node.iter.func.value, ast.Name)):
            continue
        target = node.iter.func.value.id
        for a in ast.walk(fn):
            if (isinstance(a, ast.Assign) and isinstance(a.value, ast.Dict)
                    and any(isinstance(t, ast.Name) and t.id == target
                            for t in a.targets)):
                idx = []
                for k in a.value.keys:
                    if (isinstance(k, ast.Subscript)
                            and isinstance(k.value, ast.Name)
                            and k.value.id == "CROSS_LEGS"
                            and isinstance(k.slice, ast.Constant)):
                        idx.append(k.slice.value)
                return sorted(set(idx)) if len(idx) == len(set(idx)) else None
    return None


def _run_wiring():
    """What `run()` ACTUALLY writes, read off its AST (harness audit G5).

    A5 pushes every world through the adapter, but the fixture and the adapter
    share `DECISION_LEGS`/`WORLD_LEG`/`GREEN_STREAMS`, so a drift in `run()`'s
    string literals leaves all 516,096 worlds passing while the real artefact
    is unreadable. The result is fail-closed -- no false verdict -- but the
    whole budget is lost silently, which is S-6 F2 in miniature. Bind them.
    """
    import ast
    src = open(os.path.abspath(__file__), encoding="utf-8").read()
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "run")
    legs, cross, streams = set(), set(), set()
    for node in ast.walk(fn):
        if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
                and isinstance(node.slice, ast.Constant)
                and isinstance(node.ctx, ast.Store)):
            (legs if node.value.id == "legs" else cross if
             node.value.id == "cross" else set()).add(node.slice.value)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "driver_readout" and node.args
                and isinstance(node.args[0], ast.Dict)):
            streams |= {k.value for k in node.args[0].keys
                        if isinstance(k, ast.Constant)}
    # ★The cross legs are assigned through a loop variable, so their names
    #   cannot be read off a subscript. Checking that run()'s string literals
    #   "match CROSS_LEGS" would be an IDENTITY -- it would search for values
    #   already in the constant and could only ever confirm (the shape this
    #   project keeps catching). Require instead that run() REFERENCES the
    #   constant by name, which fails the moment someone re-spells the
    #   literals (audit N3: it used to, and the constant was dead).
    # ★audit H4: "does run() mention the constant" is weak -- a bare
    #   `_ = CROSS_LEGS` passes it while the literals come back (measured).
    #   The falsifiable form is NEGATIVE: if the names come from the constant,
    #   then NO string constant in run() may equal one of its elements.
    strs = {n.value for n in ast.walk(fn)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    # Same idea for the sweep counts, targeted so ordinary 1/2 literals
    # elsewhere do not trip it: every leg call must pass a NAME, not a number.
    sweep_args_are_names, no_kw_sweeps, n_leg_calls = True, True, 0
    for node in ast.walk(fn):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in ("_leg_eager", "_leg_graph")):
            n_leg_calls += 1
            # ★audit E3: `_leg_graph(..., dev, sweeps=1)` used to PASS, because
            #   the last POSITIONAL argument was then `dev` (a Name). A numeric
            #   sweep count smuggled in as a keyword must fail too.
            for kw in node.keywords:
                if kw.arg == "sweeps" and not isinstance(kw.value, ast.Name):
                    no_kw_sweeps = False
            pos = [a for a in node.args]
            # the sweep count is the last positional arg; a call with none at
            # all must not silently skip the check
            if not pos or not isinstance(pos[-1], ast.Name):
                if not any(kw.arg == "sweeps" and isinstance(kw.value, ast.Name)
                           for kw in node.keywords):
                    sweep_args_are_names = False
    return {"legs": legs, "streams": streams,
            "cross_literals_absent": not (strs & set(CROSS_LEGS)),
            "no_kw_sweeps": no_kw_sweeps, "n_leg_calls": n_leg_calls,
            # every capture_role=/replay_role= in the cross-leg loop must be a
            # CALL (role_of(...)), never a bare string constant.
            "roles_derived": all(
                isinstance(kw.value, ast.Call)
                for node in ast.walk(fn)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "_leg_graph"
                for kw in node.keywords
                if kw.arg in ("capture_role", "replay_role")
                and not (isinstance(kw.value, ast.Constant)
                         and kw.value.value == "plain")),
            # ★audit E2. The first version took max() over EVERY dict in
            #   run(), so a decoy `_decoy = {CROSS_LEGS[0]: None, ...}` made it
            #   pass while the real loop went back to zip -- measured, ALL
            #   PASS. And it counted DUPLICATE keys, so {CROSS_LEGS[0]: a,
            #   CROSS_LEGS[0]: b} scored 2 while only one cross leg ran.
            #   Follow the loop instead: find `for ... in <name>.items()`, then
            #   the Assign that builds <name>, and require its key INDEX SET to
            #   be exactly range(len(CROSS_LEGS)).
            "cross_spec_index_set": _cross_loop_index_set(fn),
            "sweep_args_are_names": sweep_args_are_names}


def _banned_adjective_mentions():
    """Every line using the banned adjective, and whether it NEGATES the claim.

    ★audit G7: the prereg sec10 table claimed the repair was verified by a grep
    returning 0. It returns 4, and all four are negations ("NOT an established
    ... SM index") -- the CONTENT is right and the stated VERIFICATION METHOD
    was false. Round 4's E6 is exactly that failure, so the method is replaced
    by a predicate that actually holds.

    ★The search term is ASSEMBLED FROM PIECES and this docstring does not spell
    it, because a checker that quotes its own watch string catches itself
    (lesson #54). Measured: the first version of this function reported six
    hits, all of them its own machinery.
    """
    term = "phys" + "ical"
    out = []
    for fname in (os.path.abspath(__file__),
                  os.path.abspath(__file__).replace(".py", ".sbatch")):
        try:
            lines = open(fname, encoding="utf-8").read().splitlines()
        except OSError:
            continue
        for i, line in enumerate(lines, 1):
            low = line.lower()
            if term in low and "phys" + '" + "' + "ical" not in line:
                neg = any(t in low for t in
                          ("not ", "never", "nothing here may be written"))
                out.append((os.path.basename(fname), i, neg, line.strip()))
    return out


def _guard_findings():
    src = open(os.path.abspath(__file__), encoding="utf-8").read()
    return CEN._score_excludes_target_layer(src=src)


# --- adapter mutants: one per Q2 contract line -----------------------------
def _w_unreplicated():
    """gg sweep B lacks the escape: the union sees it, one sweep does not."""
    w = RULE.World(RULE.FULL, RULE.FULL, RULE.GREEN, RULE.GREEN | {40},
                   S_egp=set(range(34, 108)), escape_replicated=False)
    return raw_from_world(w)


def _w_unreplicated_b():
    """The same escape, but in the SECOND sweep only.

    ★This is the witness that makes `S(leg) := S_A ∪ S_B` load-bearing. With
    the escape in sweep A (`_w_unreplicated`) a sweep-A-only adapter still
    sees it, so that witness cannot detect the mutant -- measured: the first
    version of this table used it and `sweep_A_only` reported no flip. Reading
    sweep A alone here hides the escape outright.
    """
    raw = _w_unreplicated()
    raw["legs"]["graph_green"]["sweeps"].reverse()
    return raw


def _w_mixed_min_hits():
    raw = raw_from_world(_healthy())
    raw["legs"]["eager_green"]["sweeps"][1] = _fake_sweep(RULE.GREEN, True, 0)
    return raw


def _w_mixed_saturation():
    raw = raw_from_world(_healthy())
    raw["legs"]["graph_green"]["sweeps"][1] = _fake_sweep(RULE.GREEN, False, 1)
    return raw


def _w_no_negative_control():
    w = _healthy()
    return raw_from_world(w, control_detached=False)


def _w_no_capture_status():
    raw = raw_from_world(_healthy())
    raw["legs"]["graph_green"].pop("capture_status")
    return raw


def _w_dead_instrument():
    raw = raw_from_world(_healthy())
    raw["runtime_ptx_smid_sites"] = 0
    return raw


def _w_wrong_granularity():
    raw = raw_from_world(_healthy())
    raw["granularity"] = RULE.GRANULARITY + 97
    return raw


def _w_healthy_raw():
    return raw_from_world(_healthy())


def _w_noise_per_sweep():
    """★audit G1: ONE STRAY LABEL PER SWEEP -- noise's cheapest form.

    Sweep A escapes to 40, sweep B to 41. The old `any` predicate called that
    "replicated" and handed it the probe's most expensive verdict; the
    registered intersection predicate calls it unreplicated.
    """
    w = RULE.World(RULE.FULL, RULE.FULL, RULE.GREEN, RULE.GREEN | {40, 41},
                   S_egp=set(range(34, 108)))
    raw = raw_from_world(w)
    raw["legs"]["graph_green"]["sweeps"] = [
        _fake_sweep(RULE.GREEN | {40}), _fake_sweep(RULE.GREEN | {41})]
    return raw


def _w_plain_graph_partial():
    """★audit G2: the PLAIN graph leg only partly captured."""
    raw = _w_healthy_raw()
    raw["legs"]["graph_plain"] = _fake_leg(
        [set(range(100))], True, 1, mode="graph", capture="partial",
        replay="partial")
    return raw


def _w_graph_green_absent():
    """★audit G3(a): a truncated artefact (wall-clock kill after the flush of
    an earlier decision leg) -- NOT a refusal by the green stream."""
    raw = _w_healthy_raw()
    del raw["legs"]["graph_green"]
    return raw


def _w_wrong_sweep_count():
    """★audit G5: a green leg censused once instead of twice."""
    raw = _w_healthy_raw()
    raw["legs"]["eager_green"]["sweeps"] = \
        raw["legs"]["eager_green"]["sweeps"][:1]
    return raw


def _j(*parts):
    """Join a mutation anchor from pieces.

    ★The anchors below are stored SPLIT on purpose. A table that spelled them
    out in full would itself be a second occurrence of every anchor, and the
    uniqueness check would then fail on the intact file -- the shape of lesson
    #54 (a watcher that quotes its own watch string catches itself). Verified:
    the first version of this table did exactly that and eight of nine mutants
    reported "the harness is stale".
    """
    return "".join(parts)


# The P1 anchor is named once and reused, so `p1_fails_open` and
# `instrument_fails_open` cannot drift apart (and cannot become two
# occurrences of the same string).
_ANCHOR_P1 = ("    return bool(isinstance(sites, int) and sites > 0 and ",
              "edge is True)")

# name -> (needle parts, replacement parts, witness builder, intact, mutated)
ADAPTER_MUTANTS = {
    # S(leg) := A ∪ B (sec8-22). Reading one sweep hides a label the other saw.
    "sweep_A_only": (
        ("    for sweep in _sweeps(leg", "):"),
        ("    for sweep in _sweeps(leg", ")[:1]:"),
        lambda: _w_unreplicated_b(), RULE.LOST_UNREP, RULE.PRESERVED),
    # min over sweeps: a leg is only as positive as its weakest census.
    "min_hits_max": (
        ("    return mi", "n(vals)"), ("    return ma", "x(vals)"),
        lambda: _w_mixed_min_hits(), RULE.ABSENT, RULE.PRESERVED),
    # EVERY sweep must saturate (P5 bias is two-directional, sec4).
    "saturation_first_sweep_only": (
        ("    for s in sw:\n        ok, _wh", "y = CEN._saturated(s)"),
        ("    for s in sw[:1]:\n        ok, _wh", "y = CEN._saturated(s)"),
        lambda: _w_mixed_saturation(), RULE.NOSAT, RULE.PRESERVED),
    # P4a is two-sided (C4): an instrument that cannot say "no" may not be
    # quoted saying "yes".
    "drop_negative_control": (
        ('    return (not pos["blocked"]) and bool(neg["detached"]',
         "), pos, neg"),
        ('    return (not pos["blocked"]', "), pos, neg"),
        lambda: _w_no_negative_control(), RULE.NOATT, RULE.PRESERVED),
    # An absent tool status is a failed tool status.
    "capture_fails_open": (
        ('    return v if v in ("ok", "partial", "fail") else ', '"absent"'),
        ('    return v if v in ("ok", "partial", "fail") else ', '"ok"'),
        lambda: _w_no_capture_status(), RULE.ABSENT, RULE.PRESERVED),
    # P1 (sec4): a dead instrument is a measurement failure, not a result.
    "instrument_fails_open": (
        _ANCHOR_P1, ("    return Tru", "e"),
        lambda: _w_dead_instrument(), RULE.ABSENT, RULE.PRESERVED),
    # The attribution leg must be the OTHER half of the pair (E2/F1).
    "prefill_leg_aliased_to_decode": (
        ('"egp": "eager_green', '_prefill"'), ('"egp": "eager_gree', 'n"'),
        lambda: _w_healthy_raw(), RULE.PRESERVED, RULE.NOPAIR),
    # F4: one sweep is not a replication.
    "replication_fails_open": (
        ("    return (len(per) >= 2 and bool(same))", ", detail"),
        ("    return Tru", "e, detail"),
        lambda: _w_unreplicated(), RULE.LOST_UNREP, RULE.LOST_PART),
    # ★G1: replication means the SAME label twice, not "each sweep saw one".
    "replication_any_instead_of_same": (
        ("    same = set.intersection(*per) if len(per) >= 2 else se", "t()"),
        ("    same = (per[0] | per[1]) if len(per) >= 2 else se", "t()"),
        lambda: _w_noise_per_sweep(), RULE.LOST_UNREP, RULE.LOST_PART),
    # ★G2: the plain graph leg's tool failure must not be read as the null
    #   channel of the decision function.
    "plain_graph_gate_removed": (
        ('    if gp_cap != "ok" or gp_rep != "ok"', ':'),
        ("    if Fals", "e:"),
        lambda: _w_plain_graph_partial(), RULE.ABSENT, RULE.NULLCH),
    # ★G3(a): an absent leg is a truncated artefact, not a green-stream refusal.
    "absent_leg_treated_as_fail": (
        ('    if "absent" in (cap, rep)', ':'),
        ("    if Fals", "e:  # noqa"),
        lambda: _w_graph_green_absent(), RULE.ABSENT, RULE.NOCAP),
    # ★G5: the registered sweep counts are checked, not assumed.
    "sweep_count_unchecked": (
        ("        if name in legs and len(_sweeps(legs.get(name))) != n_want", ":"),
        ("        if Fals", "e:  # noqa"),
        lambda: _w_wrong_sweep_count(), RULE.ABSENT, RULE.PRESERVED),
    # sec9-5: an artefact from another substrate is not scorable by this rule.
    "granularity_unchecked": (
        ("    if gran != RULE.GRANULARIT", "Y:"),
        ("    if False and gran != RULE.GRANULARIT", "Y:"),
        lambda: _w_wrong_granularity(), RULE.ABSENT, RULE.PRESERVED),
}


def _exec_mutant(src, name):
    ns = {"__name__": "_mutant", "__file__": os.path.abspath(__file__)}
    exec(compile(src, f"<mutant:{name}>", "exec"), ns)
    return ns


def selftest_adapter(sample=None, verbose=True):
    """The whole world space, through the adapter, against the rule."""
    ok = [True]

    def ck(name, cond, detail=""):
        if verbose or not cond:
            print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
                  + (f"  {detail}" if detail and not cond else ""))
        ok[0] = ok[0] and bool(cond)

    print("-- A0 producer constants (sec14 D8: no silent substitution)")
    ck("A0a rule imported its constants from the producer",
       RULE.PRODUCER_OK, str(RULE.PRODUCER_ERR))
    ck("A0b the rule's floor IS the producer's constant",
       RULE.MIN_HITS == CEN.MIN_HITS_REPORTED)
    ck("A0c granularity is a positive integer from the producer",
       isinstance(RULE.GRANULARITY, int) and RULE.GRANULARITY > 0)

    print("-- A1 schema: the fixture speaks the producer's language")
    pk = _producer_sweep_keys()
    ck("A1a the producer's sweep keys were resolved (not an empty set)",
       bool(pk), str(pk))
    ck("A1b the fixture emits exactly the producer's sweep keys",
       pk == set(_fake_sweep(RULE.GREEN).keys()),
       f"producer={sorted(pk)} fixture={sorted(_fake_sweep(RULE.GREEN))}")
    ck("A1c the producer's own saturation test reads the fixture ladder",
       CEN._saturated(_fake_sweep(RULE.GREEN, True))[0] is True
       and CEN._saturated(_fake_sweep(RULE.GREEN, False))[0] is False)

    print("-- A2 target-layer AST guard (sec10-7, transitive)")
    g = _guard_findings()
    ck("A2a no target-layer number is reachable from score()",
       g["target_numbers"] == [], str(g["target_numbers"]))
    ck("A2b no non-whitelisted driver key is reachable from score()",
       g["driver_keys"] == [], str(g["driver_keys"]))
    ck("A2c the guard actually resolves THROUGH the adapter (positive control)",
       "world_from_raw" in g["reachable"], str(g["reachable"]))
    leak = CEN._score_excludes_target_layer(src=CEN._GUARD_PROBE_SRC)
    ck("A2d the guard can fire (known-leak fixture is flagged)",
       bool(leak["driver_keys"]))

    print("-- A3 plumbing: the artefact run() assembles is scorable")
    healthy = _w_healthy_raw()
    ck("A3a a clean run scores PRESERVED",
       score(healthy)["verdict"] == RULE.PRESERVED,
       score(healthy)["verdict"])
    ck("A3b assemble_raw derives the POSITIVE-form attachment field",
       healthy["green_ctx_attached"]["green_decode"] is True
       and healthy["green_ctx_attached"]["plain_pre"] is False)
    ck("A3c the producer's NEGATIVE-form field is kept verbatim (sec11 D4)",
       healthy["driver_readout"]["streams"]["green_decode"]
       ["green_ctx_is_null"] is False)
    ck("A3d every decision leg is present in the assembled artefact",
       set(healthy["legs"]) == set(DECISION_LEGS))

    print("-- A4 fail-closed: dropping any field the adapter reads")
    for drop, expect in (("legs", RULE.ABSENT),
                         ("division_under_test", RULE.ABSENT),
                         ("granularity", RULE.ABSENT),
                         ("driver_readout", RULE.NOATT),
                         ("runtime_ptx_smid_sites", RULE.ABSENT)):
        r = dict(healthy)
        r.pop(drop, None)
        v = score(r)["verdict"]
        ck(f"A4 dropping {drop} -> {expect}", v == expect, v)
    ck("A4f an empty artefact never reaches a substantive label",
       score({})["verdict"] not in RULE.SUBSTANTIVE)

    print("-- A7 the adapter contract is BOUND to what run() writes (G5)")
    w = _run_wiring()
    ck("A7a run()'s leg keys == DECISION_LEGS == WORLD_LEG values",
       w["legs"] == set(DECISION_LEGS) == set(WORLD_LEG.values()),
       f"run={sorted(w['legs'])}")
    ck("A7b run()'s driver stream keys == the adapter's green + control sets",
       w["streams"] == set(GREEN_STREAMS) | set(PLAIN_CONTROL_STREAMS),
       f"run={sorted(w['streams'])}")
    ck("A7c run() spells NO cross-leg name as a literal (negative form)",
       w["cross_literals_absent"])
    ck("A7d every leg call takes its sweep count from a NAME, not a number",
       w["sweep_args_are_names"])
    ck("A7e the loop-driving dict is keyed by CROSS_LEGS, one per element (E2)",
       w["cross_spec_index_set"] == list(range(len(CROSS_LEGS))),
       f"index set={w['cross_spec_index_set']}")
    ck("A7f every leg call passes its sweep count POSITIONALLY (no kw gap, E3)",
       w["no_kw_sweeps"], "a leg call passes sweeps= as a keyword")
    ck("A7g exactly the expected number of leg calls (arity, E3)",
       w["n_leg_calls"] == 6,
       f"{w['n_leg_calls']} leg calls, expected 6 "
       "(5 decision legs + 1 inside the cross-leg loop)")
    print("-- A9 the cross-leg ROLES are bound at runtime, not by name (E1)")
    healthy_roles = _w_healthy_raw()
    ck("A9a the contract names both descriptive legs",
       set(CROSS_ROLES) == set(CROSS_LEGS))
    ck("A9b a leg whose roles match its name checks out",
       _cross_roles_ok("capture_green_replay_plain",
                       {"capture_stream_role": "green_decode",
                        "replay_stream_role": "plain"}) is True)
    ck("A9c ★a SWAPPED stream pair is caught (this is what H5 missed)",
       _cross_roles_ok("capture_green_replay_plain",
                       {"capture_stream_role": "plain",
                        "replay_stream_role": "green_decode"}) is False)
    ck("A9d an older artefact without roles reports None, not False",
       _cross_roles_ok("capture_green_replay_plain", {}) is None)
    # ★A9e: roles are DERIVED from stream identity in run(), so swapping the
    #   streams swaps the labels and A9c then rejects the leg. A hand-written
    #   label would survive the swap and the artefact would lie silently.
    ck("A9e run() derives the roles from the stream object (not by hand)",
       w["roles_derived"], "run() writes role strings literally")
    _ = healthy_roles

    print("-- A8 the label ceiling holds in TEXT, not by absence (G7)")
    men = _banned_adjective_mentions()
    ck("A8a every mention of the banned adjective negates the claim",
       all(neg for _f, _i, neg, _l in men),
       str([(f, i, l[:60]) for f, i, neg, l in men if not neg]))
    ck("A8b the check is not vacuous (the ceiling IS stated somewhere)",
       len(men) >= 2, f"{len(men)} mention(s)")

    print("-- A5 round trip over the rule's world space")
    n, bad = 0, []
    for name, w in RULE.worlds():
        n += 1
        if sample and n % sample:
            continue
        want = RULE.score(w)
        got = score(raw_from_world(w))["verdict"]
        if got != want:
            bad.append((name, want, got))
            if len(bad) > 5:
                break
    ck(f"A5 adapter reproduces the rule's label on every world "
       f"({n} enumerated{', sampled 1/%d' % sample if sample else ''})",
       not bad, str(bad[:3]))

    print("-- A6 repair witnesses survive the round trip")
    for label, mk, guards, with_repair, without in RULE.REPAIR_WITNESSES:
        w = mk()
        got = score(raw_from_world(w))["verdict"]
        ck(f"A6 {label[:52]}", got == with_repair, f"want {with_repair} got {got}")

    print("ADAPTER ALL PASS" if ok[0] else "ADAPTER FAILURES PRESENT")
    return 0 if ok[0] else 1


def selftest_mutants(verbose=True):
    """Lesson #53 on the adapter: every contract line must be falsifiable."""
    src = open(os.path.abspath(__file__), encoding="utf-8").read()
    ok = [True]

    def ck(name, cond, detail=""):
        if verbose or not cond:
            print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
                  + (f"  {detail}" if detail and not cond else ""))
        ok[0] = ok[0] and bool(cond)

    print("-- adapter contract mutants (prereg sec10.2)")
    for name, (n_parts, r_parts, mk, intact, mutated) in \
            ADAPTER_MUTANTS.items():
        needle, repl = _j(*n_parts), _j(*r_parts)
        cnt = src.count(needle)
        if cnt != 1:
            ck(f"{name}: anchor is unique", False,
               f"found {cnt} occurrences -- the harness is stale")
            continue
        raw = mk()
        got_intact = score(raw)["verdict"]
        try:
            ns = _exec_mutant(src.replace(needle, repl, 1), name)
        except Exception as exc:  # noqa: BLE001
            ck(f"{name}: mutant compiles", False, repr(exc))
            continue
        try:
            got_mut = ns["score"](json.loads(json.dumps(raw)))["verdict"]
        except Exception as exc:  # noqa: BLE001
            got_mut = f"RAISED {exc!r}"
        ck(f"{name}: intact={intact}", got_intact == intact, got_intact)
        ck(f"{name}: mutant={mutated}", got_mut == mutated, str(got_mut))
        # ★audit G4. Without this, a table row whose two declared labels are
        #   EQUAL passes while proving nothing -- the mutant is then not shown
        #   to be detectable at all. Round 4 caught exactly this shape in the
        #   rule file (E6 -> `T5'`) and the defect had moved here. Measured:
        #   injecting a degenerate witness produced `MUTANTS ALL PASS`.
        ck(f"{name}: the two declared labels differ (witness discriminates)",
           intact != mutated, f"both {intact}")

    print("-- runtime-instrument read-out (P1's INPUT; the census's own writer)")
    try:
        census = _readout_fixture()
    except Exception as exc:  # noqa: BLE001
        ck("the AOT fixture could be built (an unjudgeable guard is an "
           "unguarded one)", False, repr(exc))
        census = None
    if census is not None:
        for nm, status, detail in _readout_checks(globals(), census):
            ck(nm, status == "ok", f"<{status}> {detail}")
        for nm, (mutate, expected) in READOUT_MUTANTS.items():
            try:
                mutated_src = mutate(src)
            except AssertionError as exc:
                ck(f"readout mutant {nm}: anchor found", False, str(exc))
                continue
            try:
                ns = _exec_mutant(mutated_src, nm)
            except Exception as exc:  # noqa: BLE001
                ck(f"readout mutant {nm}: compiles", False, repr(exc))
                continue
            status = {n: s for n, s, _ in _readout_checks(ns, census)}
            failed = {n for n, s in status.items() if s == "fail"}
            raised = {n for n, s in status.items() if s == "raised"}
            ck(f"readout mutant {nm} breaks exactly {sorted(x.split()[0] for x in expected)}",
               failed == expected and not raised,
               f"failed={sorted(failed)} raised={sorted(raised)}")

    print("MUTANTS ALL PASS" if ok[0] else "MUTANT FAILURES PRESENT")
    return 0 if ok[0] else 1


# --- P1's input: the read-out battery, carried over from the 2026-08-21 repair
_PRE_REPAIR_RUNTIME_PTX = '''def _record_runtime_ptx(rep, kernel, device, outdir, tag):
    """2026-08-14 dead path: `JITFunction` has no `.cache` attribute in triton
    3.5.1, so this raises on every call and writes None into both P1 fields."""
    import hashlib
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
R3 = "R3 nothing compiled -> fields None, P1 stops (fail-closed)"
_FIXTURE = []


def _readout_fixture():
    """One REAL CompiledKernel of the census kernel, built with NO GPU."""
    if not _FIXTURE:
        _FIXTURE.append(CEN._extractor_fixtures()[0])
    return _FIXTURE[0]


def _readout_checks(ns, census):
    """R1-R3 against `ns`'s read-out: does P1 actually receive data?"""
    import tempfile
    rec, score_fn = ns["_record_runtime_ptx"], ns["score"]
    res = []

    def ck(name, fn):
        try:
            good = bool(fn())
        except Exception as exc:  # noqa: BLE001
            res.append((name, "raised", repr(exc)))
            return
        res.append((name, "ok" if good else "fail", ""))

    def readout(kernel):
        with tempfile.TemporaryDirectory() as td:
            rep = {}
            rec(rep, kernel, 0, td, "t")
            return rep, sorted(os.listdir(td))

    def scored(rep):
        raw = json.loads(json.dumps(_w_healthy_raw()))
        raw["runtime_ptx_smid_sites"] = rep.get("runtime_ptx_smid_sites")
        raw["runtime_spin_back_edge"] = rep.get("runtime_spin_back_edge")
        return score_fn(raw)["verdict"]

    def loaded():
        return CEN._inject(CEN._kernels()[0], 0, [census])

    def r1():
        rep, files = readout(loaded())
        return (rep.get("runtime_ptx_smid_sites") == 1
                and rep.get("runtime_spin_back_edge") is True
                and not rep.get("runtime_ptx_error")
                and files == ["smid_runtime_t.ptx"])

    def r2():
        rep, _ = readout(loaded())
        return scored(rep) == RULE.PRESERVED

    def r3():
        rep, files = readout(CEN._kernels()[0])
        return (files == []
                and rep.get("runtime_ptx_smid_sites") is None
                and rep.get("runtime_spin_back_edge") is None
                and bool(rep.get("runtime_ptx_error"))
                and scored(rep) == RULE.ABSENT)

    ck(R1, r1)
    ck(R2, r2)
    ck(R3, r3)
    return res


def _mut_dead_readout(src):
    """Undo the 2026-08-21 repair: bring back the read-out that reads a
    missing attribute. The anchor is assembled from two pieces so this
    function is not itself a second occurrence of the string it searches."""
    anchor = "_record_runtime_ptx = CEN." + "_record_runtime_ptx"
    n = src.count(anchor)
    if n != 1:
        raise AssertionError(f"binding occurs {n} times, expected 1")
    return src.replace(anchor, _PRE_REPAIR_RUNTIME_PTX, 1)


def _mut_drop_p1(src):
    """Neuter P1 so R3 ('an empty read-out still stops the scorer') can fail.

    Reuses `_ANCHOR_P1` rather than re-spelling it: a second literal copy of
    the anchor would make the uniqueness check fail on the intact file.
    """
    needle = _j(*_ANCHOR_P1)
    n = src.count(needle)
    if n != 1:
        raise AssertionError(f"P1 body occurs {n} times, expected 1")
    return src.replace(needle, "    return True", 1)


READOUT_MUTANTS = {"dead_runtime_ptx_readout": (_mut_dead_readout, {R1, R2}),
                   "p1_fails_open": (_mut_drop_p1, {R3})}


def selftest(sample=None):
    a = selftest_adapter(sample=sample)
    b = selftest_mutants()
    good = (a == 0 and b == 0)
    print("ALL PASS" if good else "FAILURES PRESENT")
    return 0 if good else 1


# ==========================================================================
# 9. CLI
# ==========================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--selftest-adapter", action="store_true")
    ap.add_argument("--selftest-mutants", action="store_true")
    ap.add_argument("--sample", type=int, default=None,
                    help="score 1 world in N (development only; the submitted "
                         "run must pass the FULL space)")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--analyze")
    ap.add_argument("--tag", default="local")
    ap.add_argument("--outdir", default=_HERE)
    ap.add_argument("--spin-ns", type=int, default=CEN.SPIN_NS_DEFAULT)
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest(sample=a.sample)
    if a.selftest_adapter:
        return selftest_adapter(sample=a.sample)
    if a.selftest_mutants:
        return selftest_mutants()
    if a.run:
        return run(a.outdir, a.tag, a.spin_ns)
    if a.analyze:
        return analyze(a.analyze, a.outdir, a.tag)
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
