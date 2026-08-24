"""
P0-A **F2 positive control** -- is `replay` the thing that writes the census?

★ NOT production code. ★ NO SERVER, NO MODEL, NO REQUEST. Spin kernels and one
  hardware register. There is no latency, throughput or goodput in its output.

------------------------------------------------------------------------------
WHY (result-audit C4 / F2)
------------------------------------------------------------------------------
audit_p0a_result_890893_2026-08-23/VERDICT.md leaves exactly one premise of job
890893 untested: that the labels in the census tensors were written by the
GRAPH REPLAY. Nothing in that run demonstrates it -- it rests on the CUDA
guarantee that capture records rather than executes. The auditor's wording:

    "잔여 전제(C4): 'census 텐서를 쓴 것이 replay다'를 보이는 양성 대조가
     없다 ... 깨지면 판정은 틀린 게 아니라 공허해진다."

★ That is the exact shape this repo has been burnt by: an empty observation
  satisfying every set predicate vacuously. If capture (or anything else) were
  writing those tensors, `PRESERVED` would not be WRONG -- it would be EMPTY.

------------------------------------------------------------------------------
DESIGN -- two legs on the SAME stream, differing in ONE step
------------------------------------------------------------------------------
  leg              what runs                                    prediction
  ---------------- -------------------------------------------- -------------
  capture_only     warmup(side stream, scratch) -> capture       ★ EMPTY set
  capture_replay   warmup -> capture -> replay                   non-empty

★ The shim is IMPORTED from `p0a_graph_sm_confinement.py`, not re-implemented,
  so both legs take the byte-identical launch path that produced job 890893.
  That file is NOT modified by this probe (its harness_sha256 is stamped into
  the cited verdict); this script only imports it.

------------------------------------------------------------------------------
PRE-FIXED DECISION RULE (fixed here, before any GPU run)
------------------------------------------------------------------------------
  P0  both legs' kernels compiled and the census instrument is alive
      (reuse the producer's own runtime-PTX read-out)     else UNDETERMINED
  P1  capture succeeded on every (grid, repeat) of BOTH legs
                                                          else NOCAP

  capture_only EMPTY  and  capture_replay NON-EMPTY   -> REPLAY_IS_THE_WRITER
  capture_only NON-EMPTY                              -> ★CENSUS_WRITTEN_WITHOUT_REPLAY
  capture_replay EMPTY                                -> UNDETERMINED (SETUP DEAD)

★ `CENSUS_WRITTEN_WITHOUT_REPLAY` does NOT mean 890893's verdict is wrong. It
  means it is VACUOUS -- every set predicate would have been satisfied by
  tensors nobody wrote. It goes to claims-auditor before anything else.
★ `REPLAY_IS_THE_WRITER` closes C4 and NOTHING else. It licenses no
  performance claim, closes no gate, and says nothing about confinement.

Usage:
    python p0a_f2_positive_control.py --selftest
    python p0a_f2_positive_control.py --run --tag T --outdir D    # needs a GPU
"""

import argparse
import json
import os
import platform
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_SMID = os.path.join(os.path.dirname(_HERE), "smid_census")
sys.path.insert(0, _SMID)
sys.path.insert(0, _HERE)
import smid_l0_census as CEN            # noqa: E402  producer, never edited
import p0a_graph_sm_confinement as P0A  # noqa: E402  cited harness, imported only

VERDICT_WRITER = "REPLAY_IS_THE_WRITER"
VERDICT_VACUOUS = "CENSUS_WRITTEN_WITHOUT_REPLAY"
VERDICT_DEAD = "UNDETERMINED (SETUP DEAD)"
VERDICT_ABSENT = CEN.V_UNDET
VERDICT_NOCAP = "UNDETERMINED (GRAPH CAPTURE UNAVAILABLE ON GREEN STREAM)"


class CaptureOnlyShim(P0A.GraphLaunchShim):
    """The cited shim with the replay step removed -- and nothing else.

    ★Subclassing rather than copying is the point: every other step (scratch
    warmup on a side stream, `torch.cuda.graph(...)` with the same pool and
    capture_error_mode, the same event accounting) is inherited byte-identical
    from the harness that produced 890893. The contrast is ONE step.
    """

    def __getitem__(self, grid):
        import torch
        n_blocks = grid[0]

        def _launch(*args, **kwargs):
            self._seq[n_blocks] = self._seq.get(n_blocks, 0) + 1
            ev = {"n_blocks": n_blocks, "seq": self._seq[n_blocks],
                  "warmup_ok": False, "capture_ok": False,
                  "replay_ok": False,        # ★ never set: that is the contrast
                  "warmup_exc": None, "capture_exc": None, "replay_exc": None}
            self.events.append(ev)
            scratch = [torch.empty_like(a) if torch.is_tensor(a) else a
                       for a in args]
            try:
                with torch.cuda.stream(self.side):
                    self.kernel[grid](*scratch, **kwargs)
                self.side.synchronize()
                ev["warmup_ok"] = True
            except Exception as exc:                     # noqa: BLE001
                ev["warmup_exc"] = repr(exc)
                return
            del scratch
            g = torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(g, stream=self.capture_stream):
                    self.kernel[grid](*args, **kwargs)
                ev["capture_ok"] = True
            except Exception as exc:                     # noqa: BLE001
                ev["capture_exc"] = repr(exc)
            finally:
                del g
            # ★ NO REPLAY. The census tensors must stay at -1.

        return _launch


def _leg(kernel, stream, spin_ns, dev, shim_cls):
    shim = shim_cls(kernel, stream, stream, dev)
    sweep = CEN._census_target(shim, stream, spin_ns)
    expected = len(CEN.GRID_SWEEP) * CEN.REPEATS_PER_GRID
    cap, rep = shim.status(expected)
    return {"sweep": sweep, "capture_status": cap, "replay_status": rep,
            "summary": shim.summary(expected)}


def run(outdir, tag, spin_ns):
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device; --run requires a GPU (MEASUREMENT "
                         "ABSENT condition, not a result)")
    from sglang.srt.multiplex import pdmux_context as pdc
    from sgl_kernel import spatial

    kernel, _, _ = CEN._kernels()
    dev = torch.cuda.current_device()
    cc = torch.cuda.get_device_capability(dev)
    if tuple(cc) != (8, 0):
        raise SystemExit(f"cc {tuple(cc)} is not the registered substrate (8,0)")
    total = spatial.get_sm_available(dev)
    p_sm, d_sm = pdc.divide_sm(total, cc, 4 - 2)[0]
    green = spatial.create_greenctx_stream_by_value(p_sm, d_sm, dev)
    g_decode = green[1]

    rep = {"kind": "p0a_f2_raw", "tag": tag,
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "host": platform.node(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
           "device_name": torch.cuda.get_device_name(dev),
           "compute_capability": list(cc), "division_under_test": [p_sm, d_sm],
           "spin_ns": spin_ns,
           "cited_harness_sha256": CEN._sha256(
               os.path.join(_HERE, "p0a_graph_sm_confinement.py")),
           "census_sha256": CEN._sha256(os.path.join(_SMID,
                                                     "smid_l0_census.py")),
           "provenance": CEN._provenance()}

    # ★ order: control first. If capture-only somehow writes, we learn it
    #   before the positive leg has touched the same stream.
    rep["capture_only"] = _leg(kernel, g_decode, spin_ns, dev, CaptureOnlyShim)
    CEN._record_runtime_ptx(rep, kernel, dev, outdir, f"p0af2_{tag}")
    rep["capture_replay"] = _leg(kernel, g_decode, spin_ns, dev,
                                 P0A.GraphLaunchShim)

    path = os.path.join(outdir, f"p0a_f2_raw_{tag}.json")
    with open(path, "w") as f:
        json.dump(rep, f, indent=2)
    print(f"[run] raw -> {path}")
    print("[run] NO VERDICT HERE. Score with --analyze.")
    return 0


def score(raw):
    out = {"kind": "p0a_f2_verdict",
           "premise_under_test": "the census tensors are written by g.replay()"}
    if not isinstance(raw, dict):
        return {**out, "verdict": VERDICT_ABSENT, "why": "raw is not an object"}
    co = raw.get("capture_only") or {}
    cr = raw.get("capture_replay") or {}
    u_co = set((co.get("sweep") or {}).get("union") or [])
    u_cr = set((cr.get("sweep") or {}).get("union") or [])
    out["capture_only_labels"] = sorted(u_co)
    out["capture_replay_label_count"] = len(u_cr)
    out["capture_status"] = {"capture_only": co.get("capture_status"),
                             "capture_replay": cr.get("capture_status")}
    out["summary"] = {"capture_only": co.get("summary"),
                      "capture_replay": cr.get("summary")}
    sites = raw.get("runtime_ptx_smid_sites")
    edge = raw.get("runtime_spin_back_edge")
    if not (isinstance(sites, int) and sites > 0 and edge is True):
        return {**out, "verdict": VERDICT_ABSENT,
                "why": "P0: the census instrument did not survive"}
    if co.get("capture_status") != "ok" or cr.get("capture_status") != "ok":
        return {**out, "verdict": VERDICT_NOCAP,
                "why": "P1: graph capture did not succeed on every pair; a "
                       "tool-availability fact, not a statement about who "
                       "writes the census"}
    if u_co:
        return {**out, "verdict": VERDICT_VACUOUS,
                "why": (f"the capture-only leg observed {len(u_co)} label(s) "
                        "with NO replay issued. Something other than replay "
                        "writes the census tensors, so job 890893's set "
                        "predicates were satisfied by data nobody replayed -- "
                        "its verdict is VACUOUS, not wrong. Refer to "
                        "claims-auditor before any other action.")}
    if not u_cr:
        return {**out, "verdict": VERDICT_DEAD,
                "why": "both legs empty: the setup produced no census at all"}
    return {**out, "verdict": VERDICT_WRITER,
            "why": (f"capture-only observed NOTHING while capture+replay "
                    f"observed {len(u_cr)} labels on the same stream, with "
                    "only the replay step differing. The census is written by "
                    "the replay. ★This closes result-audit C4 and nothing "
                    "else -- it licenses no performance claim and says "
                    "nothing about confinement.")}


def selftest():
    ok = [True]

    def ck(name, cond, detail=""):
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
              + (f"  {detail}" if detail and not cond else ""))
        ok[0] = ok[0] and bool(cond)

    def raw(co_labels, cr_labels, cap_ok=True, instr=True):
        mk = lambda u, s: {"sweep": {"union": sorted(u)},  # noqa: E731
                           "capture_status": "ok" if s else "partial",
                           "replay_status": "ok", "summary": {}}
        return {"capture_only": mk(co_labels, cap_ok),
                "capture_replay": mk(cr_labels, cap_ok),
                "runtime_ptx_smid_sites": 1 if instr else 0,
                "runtime_spin_back_edge": bool(instr)}

    print("-- decision matrix")
    ck("empty control + populated positive -> REPLAY_IS_THE_WRITER",
       score(raw(set(), set(range(34))))["verdict"] == VERDICT_WRITER)
    ck("★ANY label in the control -> CENSUS_WRITTEN_WITHOUT_REPLAY",
       score(raw({7}, set(range(34))))["verdict"] == VERDICT_VACUOUS)
    ck("both empty -> SETUP DEAD (not a premise statement)",
       score(raw(set(), set()))["verdict"] == VERDICT_DEAD)
    ck("dead instrument -> MEASUREMENT ABSENT",
       score(raw(set(), set(range(34)), instr=False))["verdict"]
       == VERDICT_ABSENT)
    ck("partial capture -> NOCAP, never a premise statement",
       score(raw(set(), set(range(34)), cap_ok=False))["verdict"]
       == VERDICT_NOCAP)

    print("-- the control cannot pass vacuously")
    ck("★a control that is empty because capture FAILED is NOT the writer "
       "verdict", score(raw(set(), set(range(34)), cap_ok=False))["verdict"]
       != VERDICT_WRITER)
    ck("★an empty POSITIVE leg cannot yield the writer verdict",
       score(raw(set(), set()))["verdict"] != VERDICT_WRITER)

    print("-- the contrast is one step (structural)")
    import inspect
    base = inspect.getsource(P0A.GraphLaunchShim.__getitem__)
    mine = inspect.getsource(CaptureOnlyShim.__getitem__)
    ck("the cited shim replays; this one does not",
       "g.replay()" in base and "g.replay()" not in mine)
    ck("both warm up on a side stream with scratch buffers",
       "torch.empty_like" in base and "torch.empty_like" in mine)
    ck("both capture on the capture stream with the same call",
       "torch.cuda.graph(g, stream=self.capture_stream)" in base
       and "torch.cuda.graph(g, stream=self.capture_stream)" in mine)
    ck("the cited harness is imported, not copied",
       CaptureOnlyShim.__mro__[1] is P0A.GraphLaunchShim)

    print("ALL PASS" if ok[0] else "FAILURES PRESENT")
    return 0 if ok[0] else 1


def analyze(raw_path, outdir, tag):
    with open(raw_path) as f:
        raw = json.load(f)
    v = score(raw)
    v["raw_path"] = os.path.abspath(raw_path)
    v["raw_sha256"] = CEN._sha256(raw_path)
    for k in ("tag", "host", "slurm_job_id", "device_name",
              "compute_capability", "division_under_test",
              "cited_harness_sha256", "census_sha256", "utc"):
        if k in raw:
            v["run_" + k] = raw[k]
    path = os.path.join(outdir, f"p0a_f2_verdict_{tag}.json")
    with open(path, "w") as f:
        json.dump(v, f, indent=2)
    print(json.dumps({k: v[k] for k in ("verdict", "why")}, indent=2))
    print(f"[analyze] verdict -> {path}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--analyze")
    ap.add_argument("--tag", default="local")
    ap.add_argument("--outdir", default=_HERE)
    ap.add_argument("--spin-ns", type=int, default=CEN.SPIN_NS_DEFAULT)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.run:
        return run(a.outdir, a.tag, a.spin_ns)
    if a.analyze:
        return analyze(a.analyze, a.outdir, a.tag)
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
