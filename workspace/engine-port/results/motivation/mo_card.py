#!/usr/bin/env python3
"""Configuration card of the motivation harness: the REGISTERED expected values of every arm and trace, and the post-boot comparison
against what the boot actually did (argv, server-log banner, engine-tree stamp, interpreter, trace).  CPU only.

A card is data (JSON, sha256-stamped).  `build_card(plan, w1_pool)` derives it from the plan; the runner compares each boot right after the
server is healthy and ABORTS THE JOB on any mismatch (mo_runner.JobAbort).  Why: "the arm I thought I fixed was not fixed" (an anchor that
was really 16 SM, a 'stock' arm that was piecewise OFF) was found only after the Job in earlier campaigns.

Checked fields (CARD_FIELDS) -- each one has a mutation test (tests/test_motivation_harness.py) showing the check fails:
  arm level : python_version venv_freeze tree piecewise chunk mixed overlap mem cap context backend seed env argv
  Job level : boot_order trace_hash trace_quantiles
An expected value of None means "not applicable to this engine" (not checked).  An expected value starting with PENDING means the value is
not yet registered (cap measurement / image build / vLLM banner bring-up): such a card is not sealed and the runner refuses it unless it is a
rehearsal card.  GPU-unverified: every banner regex here (server_args line, piecewise markers, vLLM banner patterns).
"""
from __future__ import annotations

import json
import math
import re
import subprocess
from pathlib import Path

import mo_spec as MS
import sb_runner as RU
import ph_trace as T

SP = MS.SP

ARM_FIELDS = ("python_version", "venv_freeze", "tree", "piecewise", "chunk", "mixed", "overlap", "mem", "cap", "context", "backend", "seed",
              "env", "argv")
JOB_FIELDS = ("boot_order", "trace_hash", "trace_quantiles")
CARD_FIELDS = ARM_FIELDS + JOB_FIELDS
QS = (0.5, 0.9, 0.99)

# PDMUX_* names that are Job plumbing / harness labels, not engine knobs (engine_tree.INFRA_ENV, sb_runner.INFRA_PASS, patched-tree labels)
import engine_tree as ET  # noqa: E402  (scripts/vessl, stdlib only)

PLUMBING_ENV = set(ET.INFRA_ENV) | set(RU.INFRA_PASS) | {"PDMUX_RUN_ID", "PDMUX_WORKLOAD_ID", "PDMUX_HEARTBEAT_S"}


def sha(o):
    return SP.sha(o)


def quantiles(vals, qs=QS):
    """Nearest-rank quantiles of integer lengths (exact, no interpolation -> equality is meaningful)."""
    v = sorted(int(x) for x in vals)
    if not v:
        return [None for _ in qs]
    return [v[min(len(v) - 1, max(0, math.ceil(q * len(v) - 1e-9) - 1))] for q in qs]


def trace_facts(segments):
    reqs = [r for s in segments for r in s["requests"]]
    return {"hash": T.trace_hash(segments), "n": len(reqs), "in_q": quantiles([r["input_tokens"] for r in reqs]),
            "out_q": quantiles([r["output_tokens"] for r in reqs])}


def pool_facts(pool):
    return {"sha256": sha([list(map(int, x)) for x in pool]), "n": len(pool), "in_q": quantiles([x[0] for x in pool]),
            "out_q": quantiles([x[1] for x in pool])}


def norm_argv(argv, bdir=None):
    """Machine-independent argv (engine-port root -> {ENG}), --port N removed."""
    out, skip = [], False
    for x in argv:
        if skip:
            skip = False
            continue
        if x == "--port":
            skip = True
            continue
        x = str(x).replace(str(SP.ENG), "{ENG}")
        if bdir:
            x = x.replace(str(bdir), "{dir}")
        out.append(x)
    return out


def expected_argv(a):
    return norm_argv(([a["python"]] if a["engine"] == "sglang" else []) + list(a["cmd"]))


def arm_expect(a, fixed):
    """Registered expected value of every ARM_FIELD for arm definition `a` (None = not applicable)."""
    p = a["params"]
    sg = a["engine"] == "sglang"
    e = {"python_version": "3.12", "tree": a["tree"], "env": dict(a["env"]), "argv": expected_argv(a)}
    if sg:
        e.update(venv_freeze=fixed["sg_venv_freeze"], piecewise=p["piecewise"], chunk=p["chunk"], mixed=bool(p["mixed"]), overlap=p["overlap"],
                 mem=fixed["mem"], cap=fixed["cap"], context=fixed["context"], backend="flashinfer", seed=1)
    else:
        e.update(venv_freeze=SP.VLLM_VENV_FREEZE_SHA256, piecewise=None, chunk=fixed["vllm_chunk"], mixed=None, overlap=None, mem=None,
                 cap=fixed["cap"], context=fixed["context"], backend=fixed["vllm_backend"], seed=None)
    return e


def build_card(plan, w1_pool):
    """Card of a plan.  Trace facts are computed for EVERY run of the plan from the registered pool."""
    arms = {b["arm"]: {"expect": arm_expect(b["arm_def"], plan["fixed"])} for b in plan["boots"]}
    trace = {}
    for b in plan["boots"]:
        for r in b["runs"]:
            trace[r["run_id"]] = trace_facts(RU.build_segments(r, w1_pool, []))
    card = {"schema": MS.SCHEMA_VERSION + "/card", "job": plan["job"], "rehearsal": bool(plan["rehearsal"]), "plan_sha": MS.sha(plan),
            "arms": arms, "boot_order": list(plan["boot_order"]), "anchor": plan["anchor"], "pool": pool_facts(w1_pool), "trace": trace,
            "grid": plan["grid"], "slo_grid": {"ttft_ms": list(MS.TTFT_GRID_MS), "itl_ms": list(MS.ITL_GRID_MS), "primary": list(MS.PRIMARY_SLO)},
            "class_edges": list(MS.CLASS_EDGES), "reading_rule": MS.READING_RULE}
    card["card_sha"] = sha({k: v for k, v in card.items() if k != "card_sha"})
    return card


def card_sha_ok(card):
    return card.get("card_sha") == sha({k: v for k, v in card.items() if k != "card_sha"})


def card_placeholders(card):
    """Paths of PENDING values in the card (an unsealed card)."""
    out = []
    for name, a in card["arms"].items():
        for f, v in a["expect"].items():
            if MS.is_pending(v):
                out.append(f"{name}.{f}")
            if f == "argv" and any(x in (MS.MEM_TOKEN, MS.CAP_TOKEN) for x in v):
                out.append(f"{name}.argv(placeholder token)")
    return out


# ---- observation (what the boot actually did) ------------------------------------------------------------------------------
def _server_args_text(text):
    for ln in text.splitlines():
        if "ServerArgs(" in ln:
            return ln
    return text


def _sa(text, key):
    m = re.search(rf"\b{key}=([^,)]+)", text)
    return m.group(1).strip().strip("'\"") if m else None


def _int(v):
    try:
        return int(str(v))
    except (TypeError, ValueError):
        return v


def _float(v):
    try:
        return float(str(v))
    except (TypeError, ValueError):
        return v


def probe_python_version(py, timeout=30):
    """'3.12' of the interpreter the server runs under (a subprocess of that very interpreter)."""
    r = subprocess.run([py, "-c", "import sys;print('%d.%d' % sys.version_info[:2])"], capture_output=True, text=True, timeout=timeout)
    return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else None


def freeze_sha(py):
    f = Path(py).parent.parent / "pip_freeze.txt"
    return RU._sha_file(f)


def exec_argv(cmd):
    """Argv of the server as executed: sglang boots go through `engine_tree.py boot --stamp X -- <python> ...`."""
    cmd = list(cmd or [])
    return cmd[cmd.index("--") + 1:] if "--" in cmd else cmd


def _arg(argv, flag):
    return argv[argv.index(flag) + 1] if flag in argv and argv.index(flag) + 1 < len(argv) else None


def observe_boot(a, cmd, tree_stamp, log_text, py_version_fn=probe_python_version, freeze_fn=freeze_sha, bdir=None):
    """Observed value of every ARM_FIELD.  `a` = resolved arm def; `cmd` = the Popen command; `tree_stamp` = tree_<arm>.json."""
    argv_popen = norm_argv(exec_argv(cmd), bdir)
    stamp_argv = tree_stamp.get("argv")
    argv = norm_argv(stamp_argv, bdir) if stamp_argv else argv_popen
    py = (stamp_argv or exec_argv(cmd) or [a["python"]])[0]
    o = {"python_version": py_version_fn(py), "venv_freeze": freeze_fn(py), "tree": tree_stamp.get("engine_tree"), "argv": argv}
    if argv != argv_popen:
        o["argv_popen"] = argv_popen
    if a["engine"] == "sglang":
        sa = _server_args_text(log_text)
        rr, mm = _int(_sa(sa, "max_running_requests")), _int(_sa(sa, "max_mamba_cache_size"))
        dis = _sa(sa, "disable_piecewise_cuda_graph")
        if dis == "True":
            pw = "off"
        elif dis is None:
            pw = None
        elif "Piecewise CUDA Graph failed" in log_text:
            pw = "on_FAILED"
        elif "Capture piecewise CUDA graph end" in log_text:
            pw = "on"
        else:
            pw = "on_NOT_CAPTURED"
        ovl = _sa(sa, "disable_overlap_schedule")
        o.update(piecewise=pw, chunk=_int(_sa(sa, "chunked_prefill_size")),
                 mixed=None if _sa(sa, "enable_mixed_chunk") is None else _sa(sa, "enable_mixed_chunk") == "True",
                 overlap=None if ovl is None else ("off" if ovl == "True" else "on"), mem=_float(_sa(sa, "mem_fraction_static")),
                 cap=rr if (mm is None or mm == rr) else f"max_running_requests={rr}/max_mamba_cache_size={mm}",
                 context=_int(_sa(sa, "context_length")), backend=_sa(sa, "attention_backend"), seed=_int(_sa(sa, "random_seed")),
                 env={k: (v.replace(str(bdir), "{dir}") if bdir else v) for k, v in (tree_stamp.get("pdmux_env") or {}).items()
                      if k not in PLUMBING_ENV})
    else:
        f = RU.parse_vllm_log(log_text)
        mb = re.search(r"max_num_batched_tokens[=:]\s*(\d+)", log_text)
        o.update(piecewise=None, mixed=None, overlap=None, mem=None, seed=None, chunk=_int(mb.group(1)) if mb else None,
                 cap=_int(_arg(argv, "--max-num-seqs")), context=_int(_arg(argv, "--max-model-len")), backend=f["attention_backend"],
                 env={k: v for k, v in (tree_stamp.get("env") or {}).items()})
    return o


# ---- comparison ----------------------------------------------------------------------------------------------------------------
def same(e, o):
    if isinstance(e, float) or isinstance(o, float):
        try:
            return abs(float(e) - float(o)) < 1e-9
        except (TypeError, ValueError):
            return False
    return e == o


def check_arm(card, arm_name, observed):
    """-> list of mismatches {field, expected, observed, reason}; empty = the boot matches its registered card."""
    exp = card["arms"][arm_name]["expect"]
    bad = []
    for f in ARM_FIELDS:
        e = exp[f]
        if e is None:
            continue
        o = observed.get(f)
        if MS.is_pending(e):
            bad.append({"field": f, "expected": e, "observed": o, "reason": "UNREGISTERED: card value still a placeholder"})
        elif o is None:
            bad.append({"field": f, "expected": e, "observed": None, "reason": "observed value missing (banner/stamp did not show it)"})
        elif not same(e, o):
            bad.append({"field": f, "expected": e, "observed": o, "reason": "differs from the registered value"})
    if "argv_popen" in observed and observed["argv_popen"] != exp["argv"]:
        bad.append({"field": "argv", "expected": exp["argv"], "observed": observed["argv_popen"], "reason": "executed command differs from the card"})
    return bad


def check_order(card, executed):
    """executed = arm names of the boots started so far, including this one."""
    want = card["boot_order"][:len(executed)]
    if list(executed) != want:
        return [{"field": "boot_order", "expected": want, "observed": list(executed), "reason": "boot sequence differs from the registered order"}]
    return []


def check_pool(card, pool):
    f, e = pool_facts(pool), card["pool"]
    bad = []
    if f["sha256"] != e["sha256"] or f["n"] != e["n"]:
        bad.append({"field": "trace_hash", "expected": {"pool_sha256": e["sha256"], "n": e["n"]}, "observed": {"pool_sha256": f["sha256"], "n": f["n"]},
                    "reason": "length pool differs from the registered pool"})
    if f["in_q"] != e["in_q"] or f["out_q"] != e["out_q"]:
        bad.append({"field": "trace_quantiles", "expected": {"in_q": e["in_q"], "out_q": e["out_q"]},
                    "observed": {"in_q": f["in_q"], "out_q": f["out_q"]}, "reason": "pool length quantiles differ from the registered ones"})
    return bad


def check_trace(card, run_id, facts):
    e = card["trace"].get(run_id)
    if e is None:
        return [{"field": "trace_hash", "expected": None, "observed": facts["hash"], "reason": f"run {run_id} is not in the card"}]
    bad = []
    if facts["hash"] != e["hash"]:
        bad.append({"field": "trace_hash", "expected": e["hash"], "observed": facts["hash"], "reason": f"trace of {run_id} differs"})
    if (facts["n"], facts["in_q"], facts["out_q"]) != (e["n"], e["in_q"], e["out_q"]):
        bad.append({"field": "trace_quantiles", "expected": {k: e[k] for k in ("n", "in_q", "out_q")},
                    "observed": {k: facts[k] for k in ("n", "in_q", "out_q")}, "reason": f"length quantiles of {run_id} differ"})
    return bad


def check_runs(card, boot, w1_pool):
    bad = check_pool(card, w1_pool)
    for r in boot["runs"]:
        bad += check_trace(card, r["run_id"], trace_facts(RU.build_segments(r, w1_pool, [])))
    return bad


def check_boot(card, boot, observed, executed, w1_pool):
    """All card checks of one boot (arm fields, order, pool + traces of its runs)."""
    return check_arm(card, boot["arm"], observed) + check_order(card, executed) + check_runs(card, boot, w1_pool)


def format_mismatches(boot_id, arm, bad):
    parts = "; ".join(f"[{m['field']}] expected {json.dumps(m['expected'], default=str)[:160]} observed {json.dumps(m['observed'], default=str)[:160]} "
                      f"({m['reason']})" for m in bad)
    return f"CARD_MISMATCH boot {boot_id} arm {arm}: {parts}"
