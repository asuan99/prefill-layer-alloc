"""CPU probe (descriptive, NOT a gate, NOT a measurement) -- 2026-10-04 engine-porter.

Question: under PDMUX_R2_POLICY in {fixed, generic}, does event_loop_pdmux ever
reach the manual_divisions threshold scan in adjust_stream_groups, and which
stream index does each decode step run on, for each candidate config?

Drives the REAL `event_loop_pdmux` of the tracked mirror
(src/multiplex/multiplexing_mixin.py) through tests/pdmux_loop_fakes.py.
Because the local machine has no sgl_kernel (CLAUDE.md), a throw-away
`sglang` package is assembled in a temp dir from: the tracked src/multiplex
modules, the dev-tree pdmux_context.py, and two stubs (parallel_state,
forward_batch_info.ForwardMode).  Nothing in the repo is modified.

usage: python3 probe_scan_reachability.py [--admit N] [--maxnew N] [--seeds N]
Limits: fakes do not enforce --max-running-requests; arrival process is a toy;
snapshot ITL is not modelled (generic's dynamics here are NOT the engine's).
"""
import argparse, collections, os, random, shutil, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENG = HERE.parents[1]
DEV = Path(os.environ.get("SGLANG_ENGINE_DEV",
                          Path.home() / "Experiments/KISTI/sglang_engine_dev/python"))

def build_fake_pkg(root: Path) -> None:
    srt = root / "sglang" / "srt"
    for sub in ("", "distributed", "model_executor", "multiplex", "managers"):
        (srt / sub).mkdir(parents=True, exist_ok=True)
        (srt / sub / "__init__.py").touch()
    (root / "sglang" / "__init__.py").touch()
    (srt / "distributed" / "parallel_state.py").write_text(
        "def set_pdmux_status(v):\n    pass\n"
        "def pdmux_role_is_thread_local():\n    return False\n")
    (srt / "model_executor" / "forward_batch_info.py").write_text(
        "from enum import IntEnum, auto\n"
        "class ForwardMode(IntEnum):\n"
        "    EXTEND=auto(); DECODE=auto(); MIXED=auto(); IDLE=auto(); TARGET_VERIFY=auto()\n"
        "    DRAFT_EXTEND=auto(); DRAFT_EXTEND_V2=auto(); PREBUILT=auto(); SPLIT_PREFILL=auto(); DLLM_EXTEND=auto()\n"
        "class ForwardBatch: pass\n")
    shutil.copy(DEV / "sglang/srt/multiplex/pdmux_context.py", srt / "multiplex")
    for name in ("multiplexing_mixin", "green_readout", "dual_worker", "controller",
                 "profile", "telemetry"):
        shutil.copy(ENG / "src/multiplex" / f"{name}.py", srt / "multiplex")

CONFIGS = {  # rows exactly as in the YAML files (threshold column included)
    "homog5": [[92, 16, 0], [64, 44, 0], [16, 92, 0]],
    "A=pdmux_r2": [[92, 16, 0], [84, 24, 0], [74, 34, 0], [64, 44, 0]],
    "A_agn": [[92, 16, 0], [84, 24, 18], [74, 34, 27], [64, 44, 36]],
    "B_d92": [[92, 16, 0], [84, 24, 0], [74, 34, 0], [64, 44, 0], [16, 92, 1000000]],
}

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--admit", type=int, default=8)
    ap.add_argument("--maxnew", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix="n3probe_"))
    try:
        build_fake_pkg(tmp)
        sys.path[:0] = [str(tmp), str(ENG / "tests")]
        for k in ("PDMUX_SLO_SCHED", "PDMUX_SLO_SPAN_TYPE", "PDMUX_STICKY_PARTITION",
                  "PDMUX_TPOT_SLO_MS", "PDMUX_TTFT_SLO_MS", "PDMUX_SLO_EMA"):
            os.environ.pop(k, None)
        import logging; logging.disable(logging.WARNING)
        import pdmux_loop_fakes as F
        from sglang.srt.multiplex import pdmux_context
        from sglang.srt.multiplex.controller import FixedPolicy, GenericDynamicPolicy

        def run(divs, policy, name, seed):
            F.SM_COUNTS[:] = [(108, 0)] + [(p, d) for p, d, _ in divs] + [(0, 108)]
            F.MANUAL_DIVISIONS[:] = divs
            rnd = random.Random(seed)
            arrivals = collections.defaultdict(list); rid = 0
            for it in range(600):
                if rnd.random() < 0.35:
                    for _ in range(rnd.randint(1, 3)):
                        arrivals[it].append(F.FakeReq(f"r{rid}", rnd.randint(2, a.maxnew))); rid += 1
            s = F.LoopScheduler(policy, name, dict(arrivals), 700, admit_per_batch=a.admit)
            pdmux_context.STREAM_GROUPS = s.stream_groups
            pdmux_context.CURRENT_STREAM_IDX = 0
            pdmux_context.CURRENT_STREAM_GROUP = s.stream_groups[0]
            hits = [0]; orig = s.adjust_stream_groups
            def wrapped():
                if (not s.running_batch.is_empty()) and s.split_prefill_batch \
                        and not s.sticky_partition_enabled:
                    hits[0] += 1          # the manual_divisions scan branch
                return orig()
            s.adjust_stream_groups = wrapped
            occ = collections.Counter(); orb = s.run_batch
            def rb(batch):
                if batch.forward_mode != F.ForwardMode.SPLIT_PREFILL:
                    occ[pdmux_context.CURRENT_STREAM_IDX] += 1
                return orb(batch)
            s.run_batch = rb
            F.run_loop(s)
            return hits[0], occ, list(F.SM_COUNTS)

        for cname, divs in CONFIGS.items():
            states = tuple(sorted({d for _, d, _ in divs if d in (16, 24, 34, 44)}))
            arms = [("fixed D16", lambda: FixedPolicy(16), "fixed"),
                    ("fixed D44", lambda: FixedPolicy(44), "fixed"),
                    ("generic", lambda: GenericDynamicPolicy(states, emergency_state=108), "generic"),
                    ("no policy", lambda: None, "")]
            arms.append(("fixed D92" + ("" if any(d == 92 for _, d, _ in divs) else " (ABSENT)"),
                         lambda: FixedPolicy(92), "fixed"))
            for label, mk, pname in arms:
                tot = collections.Counter(); hits = 0
                for seed in range(a.seeds):
                    h, occ, sc = run(divs, mk(), pname, seed); hits += h; tot.update(occ)
                cells = ", ".join(f"{i}{sc[i]}:{n}" for i, n in sorted(tot.items()))
                print(f"{cname:11s} {label:18s} scan_hits={hits:4d} decode_steps_by_idx={{{cells}}}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
