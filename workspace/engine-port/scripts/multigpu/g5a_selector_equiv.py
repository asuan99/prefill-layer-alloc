#!/usr/bin/env python3
"""M2g G5(a) -- deterministic CPU equivalence of the agnostic manual table and the
auto grid (PREREG_MULTIGPU_2026-10-12.md rev6 sec 3 G5(a), MGP-5).

    python3 g5a_selector_equiv.py [--runtime <sglang>/python] [--manual CFG] [--auto CFG]

WHAT IS COMPARED.  For every decode batch size 0..DECODE_BS_MAX (512, sec 3 G5(a))
and both prefill states (a split-prefill batch in flight or not), the stream
index chosen by the ENGINE'S OWN selector and the (prefill_sm, decode_sm) pair
that index names in the engine's own SM_COUNTS, under

  * auto   : `triage/pdmux_a100_smoke.yml` (no manual_divisions -> divide_sm
             on 108 SM, cc (8, 0) -- the A100 the prereg pins), and
  * manual : `benchmarks/configs/pdmux_agnostic_manual.yml`.

NOTHING IS RE-IMPLEMENTED.  `pdmux_context.py` is executed from the runtime
tree with a fake `torch` / `sgl_kernel.spatial` (SM count 108, capability
(8, 0), streams are tagged tuples), and `adjust_stream_groups` is extracted
verbatim from `multiplexing_mixin.py` by AST and executed against a minimal
scheduler stand-in.  A copy of the selector written here would make this an
identity (lesson 9 / CONSENSUS sec 3 item 18).

RESOLUTION GUARDS (so a PASS cannot be vacuous): the auto run must actually
reach `divide_sm` (capability read observed) and the manual run must not; the
grid must visit every index 0..3 in both paths.  A failure of a guard is a
FAIL, not a skip.

Exit 0 = equivalent, 1 = not equivalent / guard failed, 2 = runtime tree not
found (NOT a verdict).  A failure blocks harness sealing (sec 3 G5(a)).
"""

from __future__ import annotations

import __future__ as _future
import argparse
import ast
import copy
import json
import os
import sys
import textwrap
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Dict, List, Optional
from unittest import mock

ENG = Path(__file__).resolve().parents[2]
DECODE_BS_MAX = 512  # PREREG rev6 sec 3 G5(a): "decode_bs 0...512 전수"
A100_SM = 108  # sec 2 agnostic row: divide_sm(108, (8,0), 2)
A100_CC = (8, 0)
MANUAL_CFG = ENG / "benchmarks" / "configs" / "pdmux_agnostic_manual.yml"
AUTO_CFG = ENG / "triage" / "pdmux_a100_smoke.yml"  # the auto-grid config of G0(iii)
EXPECTED_SM_COUNTS = [(108, 0), (74, 34), (54, 54), (0, 108)]  # sec 2 agnostic row


def runtime_candidates() -> List[Path]:
    env = os.environ.get("SGLANG_ENGINE_DEV")
    if env:  # declared -> that tree only; never fall back to a sibling tree silently
        return [Path(env)]
    kisti = Path(__file__).resolve().parents[5]
    # Local development: the multigpu branch's own tree first, then the default sibling tree.
    return [kisti / "sglang_engine_dev_multigpu" / "python", kisti / "sglang_engine_dev" / "python"]


def find_runtime(explicit: Optional[str] = None) -> Optional[Path]:
    for cand in ([Path(explicit)] if explicit else runtime_candidates()):
        if (cand / "sglang" / "srt" / "multiplex" / "pdmux_context.py").is_file():
            return cand
    return None


# ----------------------------------------------------------------- engine code loaders
class _CapabilitySpy:
    def __init__(self):
        self.calls = 0

    def __call__(self, device):
        self.calls += 1
        return A100_CC


def load_pdmux_context(runtime: Path, total_sm: int = A100_SM):
    """Execute the runtime tree's pdmux_context.py with fake torch/sgl_kernel."""
    path = runtime / "sglang" / "srt" / "multiplex" / "pdmux_context.py"
    spy = _CapabilitySpy()
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            current_device=lambda: 0,
            get_device_capability=spy,
            Stream=lambda gpu_id: ("plain", gpu_id),
        )
    )
    fake_spatial = SimpleNamespace(
        get_sm_available=lambda gpu_id: total_sm,
        create_greenctx_stream_by_value=lambda p, d, gpu_id: (("green", p), ("green", d)),
    )
    fake_sgl_kernel = types.ModuleType("sgl_kernel")
    fake_sgl_kernel.spatial = fake_spatial
    module = types.ModuleType("pdmux_context_g5a")
    module.__file__ = str(path)
    with mock.patch.dict(sys.modules, {"torch": fake_torch, "sgl_kernel": fake_sgl_kernel,
                                       "sgl_kernel.spatial": fake_spatial}):
        sys.modules["pdmux_context_g5a"] = module  # @dataclass resolves via sys.modules
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), module.__dict__)
    module._g5a_capability_spy = spy
    module._g5a_modules = {"sgl_kernel": fake_sgl_kernel, "sgl_kernel.spatial": fake_spatial}
    return module


def extract_method_source(path: Path, class_name: str, method: str) -> str:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method:
                    return textwrap.dedent(ast.get_source_segment(text, item))
    raise LookupError(f"{class_name}.{method} not found in {path}")


def build_selector(ctx, mixin_path: Path, src_mutator: Optional[Callable[[str], str]] = None):
    """`adjust_stream_groups` verbatim (optionally source-mutated for mutation tests)."""
    src = extract_method_source(mixin_path, "SchedulerMultiplexMixin", "adjust_stream_groups")
    if src_mutator is not None:
        src = src_mutator(src)
    ns = {
        "os": SimpleNamespace(environ={}),  # PDMUX_SLO_SCHED unset, independent of the caller's env
        "set_current_stream_idx": ctx.set_current_stream_idx,
        "get_current_stream_idx": ctx.get_current_stream_idx,
    }
    flags = _future.annotations.compiler_flag
    exec(compile(src, str(mixin_path), "exec", flags=flags, dont_inherit=True), ns)
    return ns["adjust_stream_groups"]


class _Batch:
    def __init__(self, n):
        self.n = n

    def batch_size(self):
        return self.n

    def is_empty(self):
        return self.n == 0


def _fake_scheduler(ctx, cfg, decode_bs, prefill_in_flight):
    groups = ctx.get_stream_groups()
    return SimpleNamespace(
        running_batch=_Batch(decode_bs),
        split_prefill_batch=object() if prefill_in_flight else None,
        sticky_partition_enabled=False,  # sec 2: sticky OFF
        _sticky_fixed_idx=None,
        pdmux_config=cfg,
        real_sm_group_num=len(groups),
        stream_groups=groups,
        tp_worker=SimpleNamespace(model_runner=SimpleNamespace(update_decode_attn_backend=lambda i: None)),
    )


def run_path(runtime: Path, mixin_path: Path, cfg_path: Path,
             cfg_mutator: Optional[Callable] = None,
             src_mutator: Optional[Callable[[str], str]] = None,
             decode_bs_max: int = DECODE_BS_MAX) -> Dict:
    ctx = load_pdmux_context(runtime)
    cfg = ctx.load_pdmux_config(str(cfg_path))
    if cfg_mutator is not None:
        cfg = cfg_mutator(copy.deepcopy(cfg))
    with mock.patch.dict(sys.modules, ctx._g5a_modules):
        ctx.initialize_stream_groups(0, cfg)
    sm_counts = [tuple(x) for x in ctx.get_sm_counts()]
    select = build_selector(ctx, mixin_path, src_mutator)
    table = {}
    for bs in range(decode_bs_max + 1):
        for pf in (True, False):
            idx, _group = select(_fake_scheduler(ctx, cfg, bs, pf))
            table[(bs, pf)] = (idx, sm_counts[idx])
    return {"sm_counts": sm_counts, "table": table,
            "capability_reads": ctx._g5a_capability_spy.calls,
            "manual": bool(cfg.manual_divisions)}


def compare(runtime: Path, mixin_path: Path, manual_cfg: Path = MANUAL_CFG, auto_cfg: Path = AUTO_CFG,
            manual_mutator=None, auto_mutator=None, src_mutator=None,
            decode_bs_max: int = DECODE_BS_MAX) -> Dict:
    auto = run_path(runtime, mixin_path, auto_cfg, auto_mutator, src_mutator, decode_bs_max)
    man = run_path(runtime, mixin_path, manual_cfg, manual_mutator, src_mutator, decode_bs_max)
    mism = [
        {"decode_bs": bs, "prefill_in_flight": pf,
         "auto": {"idx": auto["table"][(bs, pf)][0], "sm": list(auto["table"][(bs, pf)][1])},
         "manual": {"idx": man["table"][(bs, pf)][0], "sm": list(man["table"][(bs, pf)][1])}}
        for (bs, pf) in sorted(auto["table"]) if auto["table"][(bs, pf)] != man["table"][(bs, pf)]
    ]
    guards = []
    if auto["manual"] or not man["manual"]:
        guards.append("config roles wrong: auto config carries manual_divisions or manual config does not")
    if auto["capability_reads"] < 1:
        guards.append("auto path never read the device capability -> divide_sm not exercised")
    if man["capability_reads"] != 0:
        guards.append("manual path read the device capability -> took the auto branch")
    for name, res in (("auto", auto), ("manual", man)):
        seen = sorted({v[0] for v in res["table"].values()})
        if seen != list(range(len(res["sm_counts"]))):
            guards.append(f"{name}: grid visited stream indices {seen}, not every index")
    sm_equal = auto["sm_counts"] == man["sm_counts"]
    verdict = "PASS" if (not mism and sm_equal and not guards) else "FAIL"
    return {
        "verdict": verdict,
        "n_states": len(auto["table"]),
        "decode_bs_range": [0, decode_bs_max],
        "sm_counts_auto": auto["sm_counts"],
        "sm_counts_manual": man["sm_counts"],
        "sm_counts_equal": sm_equal,
        "sm_counts_match_registered": man["sm_counts"] == EXPECTED_SM_COUNTS,
        "n_mismatch": len(mism),
        "first_mismatches": mism[:10],
        "guard_failures": guards,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--runtime", help="<sglang>/python (default: SGLANG_ENGINE_DEV, then sibling trees)")
    ap.add_argument("--mixin", help="multiplexing_mixin.py to extract the selector from "
                                    "(default: the runtime tree's copy)")
    ap.add_argument("--manual", default=str(MANUAL_CFG))
    ap.add_argument("--auto", default=str(AUTO_CFG))
    a = ap.parse_args(argv)
    runtime = find_runtime(a.runtime)
    if runtime is None:
        print("G5A_RUNTIME_NOT_FOUND (not a verdict)", file=sys.stderr)
        return 2
    mixin = Path(a.mixin) if a.mixin else runtime / "sglang" / "srt" / "multiplex" / "multiplexing_mixin.py"
    res = compare(runtime, mixin, Path(a.manual), Path(a.auto))
    res["runtime"], res["mixin"] = str(runtime), str(mixin)
    print(json.dumps(res, indent=1, default=list))
    # sm_counts_match_registered is part of the verdict: the table must also be the sec 2 table.
    return 0 if res["verdict"] == "PASS" and res["sm_counts_match_registered"] else 1


if __name__ == "__main__":
    sys.exit(main())
