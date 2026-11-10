#!/usr/bin/env python3
"""Registered constants, arm table, grid rule and Job plan of the motivation harness (2026-11-10).  CPU only; starts no server.

Re-uses the sealed stock-baseline code by import (sb_spec / sb_runner / sb_loadgen); nothing under results/stock_baseline is modified.
Measurements are named by what they observe (no M1/D1 numbering).

  * Arms (ARM_PARAMS): SGLang v0.5.10 on /opt/sg312-venv (interpreter chosen per arm, booted through scripts/vessl/engine_tree.py boot),
    vLLM 0.31.0 unchanged (/opt/vllm-venv).  An arm NAME states its whole bundle (bundle_name).
  * mem and cap are PLACEHOLDERS ("PENDING:...") until the cap measurement fixes them; a plan/card with a placeholder is a rehearsal-only
    artefact (plan_problems / mo_card.card_placeholders), never launchable.
  * Grid rule: one registered base grid + ONE registered extension (EXT_COUNT rungs above the top).  An extension is its own full-block Job;
    points of different Jobs are never merged (mo_analyze.merge_points refuses).
  * Order: the arms of a Job boot in a seeded random permutation; the anchor arm boots first and last with the registered anchor cell.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(os.path.abspath(__file__)).parent
ENG = HERE.parents[1]
SB_DIR = HERE.parent / "stock_baseline"
for _p in (str(SB_DIR), str(HERE), str(ENG / "scripts" / "vessl")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import sb_spec as SP  # noqa: E402  (sealed; imported, not modified)

PY312 = "/opt/sg312-venv/bin/python"
PY312_VENV = "/opt/sg312-venv"
VLLM_PY = SP.VLLM_PY
SCHEMA_VERSION = "motivation.raw/v1"
PLAN_SEED = SP.PLAN_SEED
PENDING = "PENDING"

# ---- placeholders ---------------------------------------------------------------------------------------------------------
MEM_TOKEN, CAP_TOKEN = "{MEM}", "{CAP}"


def pend(what):
    return f"{PENDING}:{what}"


def is_pending(v):
    return isinstance(v, str) and v.startswith(PENDING)


def default_fixed():
    """Constants fixed by measurements that have not run yet.  mem / cap: cap measurement; context: registered; vllm_*: bring-up of the
    vLLM 0.31.0 banner (the log patterns are GPU-unverified); sg_venv_freeze: the py3.12 image build."""
    return {"mem": pend("cap_measurement.mem"), "cap": pend("cap_measurement.cap"), "context": 20480,
            "vllm_chunk": pend("vllm_banner.max_num_batched_tokens"), "vllm_backend": pend("vllm_banner.attention_backend"),
            "sg_venv_freeze": pend("image_build.sg312_pip_freeze_sha256")}


REHEARSAL_FIXED = {"mem": 0.6, "cap": 48, "context": 20480, "vllm_chunk": 8192, "vllm_backend": "FLASHINFER",
                   "sg_venv_freeze": "rehearsal-any"}   # rehearsal cards only (card["rehearsal"] = True); never a measurement value

# ---- arms ---------------------------------------------------------------------------------------------------------------
# SG baselines run piecewise OFF (user decision 2026-11-10: upstream labels the feature experimental, research systems and the public PD-mux prefill path
# are eager, and NemotronH keeps its 27 Mamba layers eager anyway); SG_PIECEWISE_ON is the one-flag sensitivity arm for the v0.5.10 code default.
# One row = one bundle.  piecewise / overlap / mixed / chunk / tree are the registered treatment fields; the name is generated from them.
ARM_PARAMS = (
    dict(key="SG_MIXED_8192", engine="sglang", tree="pristine", mixed=True, chunk=8192, piecewise="off", overlap="on", role="baseline"),
    dict(key="SG_CHUNKED_8192", engine="sglang", tree="pristine", mixed=False, chunk=8192, piecewise="off", overlap="on", role="baseline"),
    dict(key="SG_MIXED_2048", engine="sglang", tree="pristine", mixed=True, chunk=2048, piecewise="off", overlap="on", role="baseline"),
    dict(key="VLLM", engine="vllm", tree="vllm", mixed=None, chunk="vllm", piecewise=None, overlap=None, role="baseline"),
    dict(key="PDMUX_BUNDLE", engine="sglang", tree="patched", mixed=False, chunk=-1, piecewise="off", overlap="off", role="bundle", D=44),
    dict(key="SG_PATCH_OFF", engine="sglang", tree="patched", mixed=True, chunk=8192, piecewise="off", overlap="on", role="control_patch"),
    dict(key="SG_PIECEWISE_ON", engine="sglang", tree="pristine", mixed=True, chunk=8192, piecewise="on", overlap="on",
         role="control_piecewise"),
)
PRIMARY_KEY = "SG_MIXED_8192"          # primary baseline + anchor arm
PDMUX_BUNDLE_NOTE = ("PD-mux bundle = non-sticky static D44 (inherited from the earlier A100 grid, NOT re-decided), budget 65536, "
                     "--chunked-prefill-size -1, overlap OFF, piecewise OFF (as sealed in sb PM-ref); cap/mem placeholders replace the "
                     "inherited 48 / 0.6")


def bundle_name(p):
    """Arm name = the bundle.  Every treatment field of ARM_PARAMS appears in the name (tests mutate each and require a different name)."""
    if p["engine"] == "vllm":
        return "VLLM0.31.0_py312venv_default_defaultchunk"
    base = f"SGL0.5.10_py312_{p['tree']}tree"
    if p["key"] == "PDMUX_BUNDLE":
        return (f"{base}_PDMUX-D{p['D']}-nonsticky-budget65536_chunk{p['chunk']}_piecewise{p['piecewise'].upper()}"
                f"_overlap{p['overlap'].upper()}_bundle")
    sched = "mixed" if p["mixed"] else "chunked"
    flag = "_pdmuxflagsOFF" if p["role"] == "control_patch" else ""
    return f"{base}{flag}_{sched}_chunk{p['chunk']}_piecewise{p['piecewise'].upper()}_overlap{p['overlap'].upper()}"


def _opt(cmd, flag, val):
    cmd = list(cmd)
    if flag in cmd:
        cmd[cmd.index(flag) + 1] = str(val)
    else:
        cmd += [flag, str(val)]
    return cmd


def _tok(v, token):
    return token if is_pending(v) else str(v)


def sg_argv(p, fixed):
    """SGLang server argv (after the interpreter).  Same common flags for every SGLang arm (backend, seed, cap, mem, context fixed);
    only the registered treatment flags differ."""
    cap, mem = _tok(fixed["cap"], CAP_TOKEN), _tok(fixed["mem"], MEM_TOKEN)
    a = ["-m", "sglang.launch_server"] + list(SP.NANO_ARGS) + [
        "--trust-remote-code", "--dtype", "bfloat16", "--attention-backend", "flashinfer", "--disable-radix-cache",
        "--context-length", str(fixed["context"]), "--max-prefill-tokens", "16384",
        "--max-running-requests", cap, "--max-mamba-cache-size", cap, "--mem-fraction-static", mem,
        "--chunked-prefill-size", str(p["chunk"]), "--random-seed", "1", "--host", "127.0.0.1"]
    if p["mixed"]:
        a.append("--enable-mixed-chunk")
    if p["piecewise"] == "off":
        a.append("--disable-piecewise-cuda-graph")
    if p["overlap"] == "off":
        a.append("--disable-overlap-schedule")
    if p["key"] == "PDMUX_BUNDLE":
        pm = SP.arm("PM-ref")["cmd"]                 # the sealed bundle's PD-mux flags (--enable-pdmux --pdmux-config-path <config>)
        a += pm[pm.index("--enable-pdmux"):]
    return a


def vllm_argv(fixed):
    """vLLM 0.31.0: the sealed argv unchanged except that --max-num-seqs (was 48) takes the cap placeholder."""
    return _opt(SP.arm("VL-stock")["cmd"], "--max-num-seqs", _tok(fixed["cap"], CAP_TOKEN))


def build_arm(p, fixed=None):
    fixed = fixed or default_fixed()
    name = bundle_name(p)
    if p["engine"] == "vllm":
        return {"name": name, "key": p["key"], "role": p["role"], "engine": "vllm", "tree": "vllm", "python": VLLM_PY, "image_key": SP.IMAGE_KEY,
                "cmd": vllm_argv(fixed), "env": dict(SP.VL_ENV), "api": "openai", "params": dict(p)}
    env = dict(SP.arm("PM-ref")["env"]) if p["key"] == "PDMUX_BUNDLE" else {}
    a = {"name": name, "key": p["key"], "role": p["role"], "engine": "sglang", "tree": p["tree"], "python": PY312, "image_key": SP.IMAGE_KEY,
         "cmd": sg_argv(p, fixed), "env": env, "api": "sglang", "params": dict(p)}
    if p["key"] == "PDMUX_BUNDLE":
        a.update(D=p["D"], sticky=False, inherited=PDMUX_BUNDLE_NOTE)
    return a


def arm_table(fixed=None):
    return [build_arm(p, fixed) for p in ARM_PARAMS]


ARM_KEYS = tuple(p["key"] for p in ARM_PARAMS)
BASELINE_KEYS = tuple(p["key"] for p in ARM_PARAMS if p["role"] == "baseline")
BUNDLE_KEY = "PDMUX_BUNDLE"
CONTROL_KEYS = tuple(p["key"] for p in ARM_PARAMS if p["role"].startswith("control"))
ANCHOR_KEY = PRIMARY_KEY


def name_of(key):
    return bundle_name(next(p for p in ARM_PARAMS if p["key"] == key))


# ---- SLO grid / reading rule / estimators (registered; evaluated post hoc by mo_analyze) -------------------------------------
TTFT_GRID_MS = (1000.0, 2000.0, 3000.0, 5000.0)
ITL_GRID_MS = (40.0, 60.0, 80.0, 100.0)
PRIMARY_SLO = (SP.W1_TTFT_MS, SP.ITL_MS)            # (3000, 60): the capacity estimators and per-class tables use this pair
CAP_ATTAIN = SP.CAP_ATTAIN                          # 0.90
ATT_MARGIN = 0.03                                   # attainment difference counted as a sign
SAT_HI, SAT_LO = 0.98, 0.02                         # both arms at/above or at/below -> the SLO pair carries no information at this rung
READ_MIN_INFORMATIVE, READ_AGREE_FRAC = 3, 0.75     # a rung label needs >= 3 informative SLO pairs and >= 75 % of them to share one sign
CLASS_EDGES = (512, 2048)                           # request class by input tokens: <=512 / <=2048 / above
MIN_CLASS_N = 20                                    # per-class cells with fewer requests are flagged SMALL_N (still reported)
MIN_SEEDS = 4                                       # gate 3
ANCHOR_DRIFT_PP = SP.SUBSTRATE_SHIFT_PP             # |attainment(anchor start) - attainment(anchor end)| above this -> ANCHOR_DRIFT


def request_class(input_tokens, edges=CLASS_EDGES):
    for e in edges:
        if input_tokens <= e:
            return f"in<={e}"
    return f"in>{edges[-1]}"


READING_RULE = (f"A rung label compares two arms over every (TTFT, ITL) pair of the grid; pairs where both arms are >= {SAT_HI} or both "
                f"<= {SAT_LO} are uninformative and dropped.  With >= {READ_MIN_INFORMATIVE} informative pairs, the label is 'A>B' ('A<B') iff "
                f">= {READ_AGREE_FRAC:.0%} of the informative pairs differ by more than {ATT_MARGIN} in that direction and none in the other; "
                f"otherwise SLO_DEPENDENT; fewer informative pairs -> UNINFORMATIVE.")

# ---- grid rule ----------------------------------------------------------------------------------------------------------------
EXT_COUNT = 2          # the ONE registered extension: this many rungs above the base top, run as its own full-block Job


class GridError(RuntimeError):
    pass


def make_grid(rmin, factor, n, role="base", ext_count=EXT_COUNT):
    base = [round(rmin * factor ** i, 6) for i in range(n)]
    ext = [round(rmin * factor ** i, 6) for i in range(n, n + ext_count)]
    if role == "base":
        return {"role": "base", "rmin": rmin, "factor": factor, "n": n, "ext_count": ext_count, "rates": base, "extension_rates": ext}
    if role == "extension":                          # overlap rung (base top) + the extension rungs; the Job is evaluated on its own
        return {"role": "extension", "rmin": rmin, "factor": factor, "n": n, "ext_count": ext_count, "rates": [base[-1]] + ext,
                "extension_rates": []}
    raise GridError(f"unknown grid role {role}")


def may_extend(base_top_pass, history_roles):
    """-> None if the one registered extension is allowed, else the refusal.  Allowed iff the base Job's top rung passed for some arm and
    no extension Job exists yet (history_roles = grid roles of the campaign's earlier Jobs)."""
    if "extension" in history_roles:
        return "GRID_EXTENSION_ALREADY_USED: the single registered extension was spent; no further rungs"
    if not base_top_pass:
        return "GRID_EXTENSION_NOT_TRIGGERED: no arm passed the base top rung"
    return None


# ---- plan ---------------------------------------------------------------------------------------------------------------------
WORKLOAD = "W1"      # pool format of sb: [[input_tokens, output_tokens], ...]
ANCHOR_RATE_INDEX, ANCHOR_N = 0, 120
ANCHOR_SEED = SP.CAP_SEEDS[0]


def _rng(*salt):
    return SP._rng("motivation", *salt)


def boot_order(job, keys=ARM_KEYS):
    """Seeded permutation of the arm keys (the registered order rule: re-derivable from the Job name)."""
    order = list(keys)
    _rng("order", job).shuffle(order)
    return order


def cell_run(job, arm_name, rate, seed, n, cell, kind="cap", **kw):
    return SP.cap_run(f"{job}_{arm_name}_{cell}_r{rate}_s{seed}", WORKLOAD, rate, seed, n, kind=kind, cell=cell, **kw)


def make_plan(job, grid, fixed=None, seeds=SP.EVAL_SEEDS, n_requests=120, rehearsal=False, arm_keys=ARM_KEYS):
    """One Job: anchor boot (anchor arm, anchor cell) -> every arm once, in the seeded permutation, each with ALL grid rungs x seeds in
    random cell order -> anchor boot again.  The anchor cell = lowest grid rung, ANCHOR_SEED, ANCHOR_N requests."""
    fixed = dict(fixed or (REHEARSAL_FIXED if rehearsal else default_fixed()))
    arms = {p["key"]: build_arm(p, fixed) for p in ARM_PARAMS if p["key"] in arm_keys}
    order = boot_order(job, tuple(k for k in ARM_KEYS if k in arm_keys))
    anchor_rate = grid["rates"][ANCHOR_RATE_INDEX]
    anchor = arms[ANCHOR_KEY]

    def anchor_boot(tag):
        r = cell_run(job, anchor["name"], anchor_rate, ANCHOR_SEED, n_requests if rehearsal else ANCHOR_N, f"ANCHOR_{tag}", kind="anchor")
        return SP.boot(f"{job}_anchor_{tag}", anchor, [r], role="anchor", anchor=tag)

    boots = [anchor_boot("start")]
    for k in order:
        a = arms[k]
        cells = [cell_run(job, a["name"], r, s, n_requests, "GRID") for r in grid["rates"] for s in seeds]
        _rng("cells", job, k).shuffle(cells)
        boots.append(SP.boot(f"{job}_{a['name']}", a, cells, role="meas"))
    boots.append(anchor_boot("end"))
    return {"job": job, "kind": "MOTIVATION_GRID", "grid": grid, "seeds": list(seeds), "n_requests": n_requests, "rehearsal": bool(rehearsal),
            "fixed": fixed, "boot_order": [anchor["name"]] + [arms[k]["name"] for k in order] + [anchor["name"]], "order_keys": order,
            "anchor": {"arm": anchor["name"], "rate": anchor_rate, "seed": ANCHOR_SEED, "n": boots[0]["runs"][0]["n_requests"],
                       "positions": ["first", "last"]},
            "arm_keys": [k for k in ARM_KEYS if k in arm_keys], "boots": boots, "pending": []}


def sha(o):
    return SP.sha(o)


def plan_problems(plan, allow_placeholders=False):
    out = []
    ids = [r["run_id"] for b in plan["boots"] for r in b["runs"]]
    if len(ids) != len(set(ids)):
        out.append("duplicate run_id")
    bids = [b["boot_id"] for b in plan["boots"]]
    if len(bids) != len(set(bids)):
        out.append("duplicate boot_id")
    names = [b["arm"] for b in plan["boots"]]
    keys = plan["arm_keys"]
    want = [name_of(ANCHOR_KEY)] + [name_of(k) for k in boot_order(plan["job"], tuple(keys))] + [name_of(ANCHOR_KEY)]
    if names != want:
        out.append(f"boot order {names} != registered permutation of job '{plan['job']}' {want}")
    if plan["boot_order"] != names:
        out.append("plan.boot_order differs from the boots")
    if plan["boots"][0]["role"] != "anchor" or plan["boots"][-1]["role"] != "anchor":
        out.append("anchor boot must be first and last")
    meas = [b for b in plan["boots"] if b["role"] == "meas"]
    rates, seeds = plan["grid"]["rates"], plan["seeds"]
    for b in meas:                                    # each Job is a full block: every arm x every rung x every seed
        if sorted((r["rate"], r["seed"]) for r in b["runs"]) != sorted((r, s) for r in rates for s in seeds):
            out.append(f"{b['boot_id']}: cells != grid rungs x seeds (full-block rule)")
    if len(meas) != len(keys):
        out.append("each arm must boot exactly once as a measurement boot")
    if not plan["rehearsal"] and len(seeds) < MIN_SEEDS:
        out.append(f"n seeds {len(seeds)} < {MIN_SEEDS} (gate 3)")
    pm_env = SP.arm("PM-ref")["env"]
    for b in plan["boots"]:
        a = b["arm_def"]
        if a["engine"] == "sglang" and a["python"] != PY312:
            out.append(f"{b['boot_id']}: SGLang arm must use {PY312}")
        if a["engine"] == "vllm" and (a["python"] != VLLM_PY or a["cmd"][0] != VLLM_PY):
            out.append(f"{b['boot_id']}: vLLM arm must use {VLLM_PY}")
        if a["tree"] == "pristine" and (a["env"] or any(x in ("--enable-pdmux", "--pdmux-config-path") for x in a["cmd"])):
            out.append(f"{b['boot_id']}: pristine boot with PD-mux env/argv")
        if a["name"] != bundle_name(a["params"]):
            out.append(f"{b['boot_id']}: arm name does not state its bundle")
        if not allow_placeholders and not plan["rehearsal"] and (any(x in (MEM_TOKEN, CAP_TOKEN) for x in a["cmd"])
                                                                   or any(is_pending(v) for v in plan["fixed"].values())):
            out.append(f"{b['boot_id']}: PLACEHOLDER mem/cap unresolved (cap measurement not done) -> not launchable")
        if a["engine"] == "sglang" and any(k.startswith("PDMUX_") and pm_env.get(k) != v for k, v in a["env"].items()):
            out.append(f"{b['boot_id']}: instrumentation env in a measurement boot")
    return out
