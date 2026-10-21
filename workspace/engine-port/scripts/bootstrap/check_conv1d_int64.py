#!/usr/bin/env python3
"""Static check that Triton `_causal_conv1d_fwd_kernel` uses 64-bit token offsets (decision 15, 2026-10-20).

Patch: src/patches/triton_causal_conv1d_int64.patch.  Two levels, both CPU only (no kernel launch):

  source  (default)  AST of the kernel body: `chunk_offset` and `sequence_start_index` must each be
                     assigned exactly once from a call ending in `.to(tl.int64)`.  A cast that only
                     appears in a comment, a cast to another dtype, or a missing cast is a violation.
  --ttir             additionally compile the kernel for an sm_80 target (triton.compile, CPU) with the
                     NemotronH-Nano geometry (stride_x_token 22,656, stride_o_token 12,288) and count
                     multiplies by those two constants in the TTIR; any i32 multiply is a violation.
                     Needs triton + its CUDA backend toolchain importable; no GPU is touched.

Exit codes: 0 = no violation, 1 = violation(s), 2 = input error (file absent / kernel not found /
--ttir requested but triton unavailable).  Used by tests/test_conv1d_int64_patch.py and meant as a
pre-flight on the INSTALLED tree inside the GPU gate Job.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Dict, List

KERNEL = "_causal_conv1d_fwd_kernel"
GUARDED = ("chunk_offset", "sequence_start_index")
STRIDE_X_TOKEN = 22656  # Nemotron-Nano-9B-v2 in_proj width = stride of the xBC view
STRIDE_O_TOKEN = 12288  # conv_dim = contiguous width of empty_like(x)


class InputError(ValueError):
    pass


def _is_int64_cast(node: ast.AST) -> bool:
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "to"
            and len(node.args) == 1 and isinstance(node.args[0], ast.Attribute)
            and node.args[0].attr == "int64" and isinstance(node.args[0].value, ast.Name)
            and node.args[0].value.id == "tl")


def source_violations(text: str) -> List[str]:
    tree = ast.parse(text)
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == KERNEL]
    if len(fns) != 1:
        raise InputError(f"{KERNEL} found {len(fns)} times")
    assigns: Dict[str, List[ast.Assign]] = {name: [] for name in GUARDED}
    for node in ast.walk(fns[0]):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in assigns:
                assigns[node.targets[0].id].append(node)
    out = []
    for name, nodes in assigns.items():
        if len(nodes) != 1:
            out.append(f"{name}: assigned {len(nodes)} times in {KERNEL} (expected 1)")
        elif not _is_int64_cast(nodes[0].value):
            out.append(f"{name}: line {nodes[0].lineno} is not cast with .to(tl.int64)")
    return out


def ttir_stride_muls(path: Path) -> Dict[str, Dict[str, int]]:
    """{stride constant: {"i32": n, "i64": n}} over every arith.muli consuming that constant."""
    try:
        import triton
        from triton.backends.compiler import GPUTarget
        from triton.compiler import ASTSource
    except Exception as exc:  # pragma: no cover - environment dependent
        raise InputError(f"triton unavailable: {exc}")
    spec = importlib.util.spec_from_file_location("conv1d_under_check", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = getattr(mod, KERNEL)
    ptrs = {"x_ptr": "*fp16", "w_ptr": "*fp16", "bias_ptr": "*fp16", "initial_states_ptr": "*fp16",
            "cache_indices_ptr": "*i32", "has_initial_states_ptr": "*i1", "query_start_loc_ptr": "*i32",
            "o_ptr": "*fp16", "seqlen": "i32"}
    # conv_state strides are deliberately distinct from the two probed constants
    const = dict(dim=12288, num_cache_lines=64, stride_x_seq=0, stride_x_dim=1, stride_x_token=STRIDE_X_TOKEN,
                 stride_w_dim=4, stride_w_width=1, stride_istate_seq=36867, stride_istate_dim=1,
                 stride_istate_token=12289, stride_o_seq=0, stride_o_dim=1, stride_o_token=STRIDE_O_TOKEN,
                 pad_slot_id=-1, HAS_BIAS=True, KERNEL_WIDTH=4, SILU_ACTIVATION=True, HAS_INITIAL_STATES=True,
                 HAS_CACHE=True, IS_CONTINUOUS_BATCHING=True, USE_PAD_SLOT=True, NP2_STATELEN=4,
                 BLOCK_M=8, BLOCK_N=256)
    names = list(fn.arg_names)
    missing = [a for a in names if a not in const and a not in ptrs]
    if missing:
        raise InputError(f"kernel signature changed: {missing}")
    sig = {a: ("constexpr" if a in const else ptrs[a]) for a in names}
    src = ASTSource(fn=fn, signature=sig, constexprs={(names.index(k),): v for k, v in const.items()})
    ir = triton.compile(src, target=GPUTarget("cuda", 80, 32)).asm["ttir"]
    consts = {}
    for line in ir.splitlines():
        m = re.match(r"\s*(%[\w#]+) = arith.constant (?:dense<)?(-?\d+)>?\s*:", line)
        if m and int(m.group(2)) in (STRIDE_X_TOKEN, STRIDE_O_TOKEN):
            consts[m.group(1)] = int(m.group(2))
    out: Dict[str, Dict[str, int]] = {str(STRIDE_X_TOKEN): {}, str(STRIDE_O_TOKEN): {}}
    for line in ir.splitlines():
        if "arith.muli" not in line:
            continue
        hit = [consts[o] for o in re.findall(r"%[\w#]+", line.split("=", 1)[1]) if o in consts]
        if hit:
            ty = "i64" if "i64" in line.rsplit(":", 1)[1] else "i32"
            d = out[str(hit[0])]
            d[ty] = d.get(ty, 0) + 1
    return out


def ttir_violations(muls: Dict[str, Dict[str, int]]) -> List[str]:
    out = [f"{k}: {v.get('i32', 0)} i32 multiplies" for k, v in muls.items() if v.get("i32", 0)]
    if not muls.get(str(STRIDE_X_TOKEN)):
        out.append(f"no multiply by stride_x_token={STRIDE_X_TOKEN} found (probe broken?)")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("file", help="causal_conv1d_triton.py (installed tree or a copy)")
    ap.add_argument("--ttir", action="store_true")
    a = ap.parse_args(argv)
    try:
        p = Path(a.file)
        if not p.is_file():
            raise InputError(f"{p}: absent")
        doc = {"file": str(p), "source_violations": source_violations(p.read_text(encoding="utf-8"))}
        if a.ttir:
            doc["ttir_stride_muls"] = ttir_stride_muls(p)
            doc["ttir_violations"] = ttir_violations(doc["ttir_stride_muls"])
    except (InputError, SyntaxError) as exc:
        print(f"CONV1D_INT64_INPUT_ERROR: {exc}", file=sys.stderr)
        return 2
    bad = doc["source_violations"] + doc.get("ttir_violations", [])
    doc["verdict"] = "INT64_OK" if not bad else "INT64_VIOLATION"
    print(json.dumps(doc, sort_keys=True))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
