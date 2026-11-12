#!/usr/bin/env python3
"""Concurrency cap and decode memory traffic measurement (user definition 2026-11-10): registered constants, cell table, Job plans, time
estimates.  CPU only; starts no server.  Nothing here is a measurement.

Two parts, each its own Job kind:
  * fixed batch (KIND_FIXED): context 2^7..2^18 x batch 1..256 (both x2 steps), memory-permitted cells only, unpartitioned 108 SM, chunked
    fill, mem fraction 0.82, models Nano-9B-v2 and a31 (Nano skeleton, all-attention composition).  Observes the decode step time of a
    batch held at a context, and from it tokens/s and the analytic bytes per step (analytic_terms) per second.
  * serving cap/mem comparison (KIND_SERVING): mem {0.6, 0.75} at cap 48 + cap {48, 128, 256} at mem 0.75 = 4 combinations x 2 arms x
    4 rungs x 4 seeds.  Its result fixes mo_spec's PENDING mem / cap through mo_cap_analyze.fix_cap_mem (a coded rule).

This harness does not depend on mo_spec's PENDING mem / cap: every boot here carries its own registered mem / cap.
Sealed code is imported, never modified: cap_char/costterms_harness/ct_spec.py (window constants, prompt seeds, roofline bound),
scripts/costmodel/analytic_terms.py (s, k, bytes per step; the only source of analytic numbers), scripts/costmodel/dstar_model.py
(measured 108-SM decode / prefill fits, used for the Job time estimate only).

Values the user definition does not state are in PROPOSED (value + source); a non-rehearsal plan is refused while any of them is not in
CONFIRMED.  Thresholds of the cap/mem fixing rule have no source yet and stay PENDING (FIX_RULE).
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

HERE = Path(os.path.abspath(__file__)).parent
ENG = HERE.parents[1]
CT_DIR = HERE.parent / "cap_char" / "costterms_harness"
for _p in (str(HERE), str(CT_DIR), str(ENG / "scripts" / "costmodel")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import mo_spec as MS  # noqa: E402
import ct_spec as CT  # noqa: E402  (sealed; imported, not modified)
import analytic_terms as AT  # noqa: E402
import dstar_model as DM  # noqa: E402

SP = MS.SP
GIB = 2 ** 30
KIND_FIXED, KIND_SERVING = "CAP_FIXED_BATCH", "CAP_MEM_SERVING"
SCHEMA_FIXED = "motivation.cap_fixed_batch/v1"

# ---- user definition (tracks/main.md, 2026-11-10) ---------------------------------------------------------------------------------
CONTEXTS = tuple(2 ** i for i in range(7, 19))          # 128 .. 262,144
BATCHES = tuple(2 ** j for j in range(0, 9))            # 1 .. 256
FB_MEM = 0.82
TOTAL_SM = CT.TOTAL_SM                                  # 108, unpartitioned
NANO_DECLARED_MAX = 131072                              # contexts above it are TIME_ONLY (the model declares no quality there)
MODELS = {"nano": {"n_A": 4, "n_M": 27, "weights": "real", "tag": "nano-a4-realweights"},
          "a31": {"n_A": 31, "n_M": 0, "weights": "dummy", "tag": "nano-a31-dummyweights"}}
USER_STATED_CELLS = {"nano": 75, "a31": 60}             # tracks/main.md; NOT reproduced by the documented rule (see cell_count_gap)
SERVING_COMBOS = ((0.6, 48), (0.75, 48), (0.75, 128), (0.75, 256))      # (mem, cap)
SERVING_RUNGS, SERVING_SEEDS_MIN = 4, MS.MIN_SEEDS
# memory budget of the pre-data cell table: the boot-log values reports/motivation_gap_2026-11-09/predict_motivation_charts.py used when
# the 75 / 60 counts were derived (GPU memory visible to the server, Nano weight usage)
AVAIL_BYTES = 78.74 * GIB
W_NANO_BYTES = 16.58 * GIB
VARIANT_DIRNAME = "variant_a31"

# ---- values the definition leaves open: (value, source).  A non-rehearsal plan needs every key in CONFIRMED. ------------------------
PROPOSED = {
    "fb_cell_rule": ("weights + b*s + b*c*k <= mem * visible", "the rule of predict_motivation_charts.py (its regime_fit); it gives 90 / 72 "
                     "cells on the 12 x 9 grid, not the 75 / 60 the track file states -- the table is not registered until this is settled"),
    "fb_boot_rule": ("cap_equals_batch", "the cell rule charges state memory b x s, i.e. one boot per batch size with cap = b (18 boots per "
                                         "round); 'min_boots' covers the same cells with 4 boots per round but assumes the step time does not "
                                         "depend on cap"),
    "fb_arm": ("SG_CHUNKED_8192", "'chunked fill, unpartitioned 108 SM': the pristine-tree chunked arm of mo_spec, no instrumentation"),
    "fb_nano_weights": ("real", "Nano with its weights (the 16.58 GiB boot-log value); a31 has no trained weights -> --load-format dummy"),
    "fb_context_length": (262144, "py3.12 boot probe 2026-11-09: --context-length 262144 booted; larger values untested"),
    "fb_ctx_total_max": (262032, "same probe: 262,016 input + 16 output served; 262,127 input returned 15 of 16 tokens"),
    "fb_window": ((CT.WINDOW_ITERS, CT.TRIM_ITERS, CT.MARGIN_ITERS), "cost-term harness T1 window (300 steps, trim 16, margin 16); for "
                  "contexts below 1024 the window is c // 2 steps so that it does not span more than half the context"),
    "fb_ramp_steps": (CT.VANILLA_RAMP_STEPS, "cost-term harness allowance for decode steps before every request has joined (64)"),
    "fb_kv_frac": (CT.FEASIBLE_KV_FRAC, "cost-term harness run-time feasibility margin (0.95 of the banner token pool)"),
    "fb_rounds": ((1, 2), "roadmap: '135 cells, 2 rounds'; both rounds in ONE Job (the round is the repetition, not a second Job)"),
    "fb_anchor": (("nano", 8, 8192), "anchor cell re-measured in the first and last boot; no source for the choice of cell"),
    "fb_anchor_drift_ln": (CT.AGREE_LN, "cost-term harness agreement bound ln(1.03), used here for anchor drift and round agreement"),
    "fb_timing": ("client_token_stamps", "measurement boots carry no probe; the step time is the per-token gap seen by the client"),
    "sv_arms": (("SG_MIXED_8192", "PDMUX_BUNDLE"), "roadmap says '2 arms' without naming them: primary baseline + the PD-mux bundle"),
    "sv_anchor_combo": ((0.75, 48), "the combination shared by the mem pair and the cap ladder"),
    "sv_run_s": (300.0, "roadmap GPU-time table: 10.7 load-hours / 128 cells"),
    "sv_context": (20480, "mo_spec.default_fixed context"),
}
CONFIRMED = frozenset()


def P(key):
    return PROPOSED[key][0]


def open_items(confirmed=None):
    c = CONFIRMED if confirmed is None else confirmed
    return sorted(k for k in PROPOSED if k not in c)


# Thresholds of the cap/mem fixing rule (mo_cap_analyze.fix_cap_mem).  No document gives a value: they are the pre-registration's to set.
FIX_RULE = {
    "min_rungs_agree": MS.pend("prereg.cap_rule.min_rungs_agree"),             # rungs that must carry the same label for a direction
    "mem_tie_break": MS.pend("prereg.cap_rule.mem_tie_break"),                 # "lower" | "higher" when the mem pair shows no difference
    "cap_tie_break": MS.pend("prereg.cap_rule.cap_tie_break"),                 # "lower" | "higher" when a cap step shows no difference
    "min_kv_tokens_per_slot": MS.pend("prereg.cap_rule.min_kv_tokens_per_slot"),   # banner max_total_num_tokens / cap must reach this
}


# ---- models / memory terms (analytic_terms only) -------------------------------------------------------------------------------------
def arch(model):
    m = MODELS[model]
    return AT.nano9b_v2(m["n_A"], m["n_M"])


def weight_bytes(model):
    """Nano: the boot-log weight usage; a31: analytic parameter bytes (no boot log exists for it at mem 0.82)."""
    return W_NANO_BYTES if model == "nano" else AT.params_total(arch(model)) * arch(model).beta


def s_bytes(model):
    return AT.state_bytes_per_request(arch(model))


def k_bytes(model):
    return AT.kv_bytes_per_token(arch(model))


def time_only(c):
    return c > NANO_DECLARED_MAX


# ---- pre-data cell table ---------------------------------------------------------------------------------------------------------------
def fits_predicted(model, b, c, mem=FB_MEM):
    """Memory rule of the pre-data table [derived]: weights + b x s + b x c x k <= mem x visible GPU memory."""
    return weight_bytes(model) + b * s_bytes(model) + b * c * k_bytes(model) <= mem * AVAIL_BYTES


def predicted_cells(model, mem=FB_MEM):
    return [(b, c) for b in BATCHES for c in CONTEXTS if fits_predicted(model, b, c, mem)]


def cell_table_summary(mem=FB_MEM):
    out = {}
    for m in MODELS:
        cells = predicted_cells(m, mem)
        out[m] = {"n": len(cells), "user_stated": USER_STATED_CELLS[m],"s_bytes": s_bytes(m), "k_bytes_per_token": k_bytes(m),
                  "max_context_by_batch": {b: max((c for bb, c in cells if bb == b), default=None) for b in BATCHES},
                  "n_time_only": sum(1 for _, c in cells if time_only(c))}
    return out


def cell_count_gap(mem=FB_MEM):
    """{model: computed - user-stated}; all zero iff the documented rule reproduces the stated counts."""
    return {m: len(predicted_cells(m, mem)) - USER_STATED_CELLS[m] for m in MODELS}


# ---- cell geometry -----------------------------------------------------------------------------------------------------------------------
def window_iters(c):
    return min(P("fb_window")[0], c // 2)


def trim_iters():
    return P("fb_window")[1]


def abort_at(c):
    """Tokens every request must have received before the burst is aborted: first token (prefill) + trim + window + 2."""
    return 1 + trim_iters() + window_iters(c) + 2


def o_tokens(c):
    return abort_at(c) + P("fb_window")[2] + P("fb_ramp_steps")


def x_tokens(c):
    """Tokens generated up to the middle of the window: the input is c minus this, so the window is centred on context c."""
    return 1 + trim_iters() + window_iters(c) // 2


def c_in(c):
    """Input tokens of a cell.  The top context cannot be centred: input + output must stay within the probe-verified total.  The
    shift is a property of the REGISTERED context length only; a server that logs a shorter context gets the cell refused, not shrunk."""
    return min(c - x_tokens(c), P("fb_ctx_total_max") - o_tokens(c))


# ---- run-time feasibility (boot banner) -----------------------------------------------------------------------------------------------------
def fits_banner(b, c, kv_tokens, cap, context, capturable=True):
    """-> [] or the reasons a cell is not sent.  kv_tokens = the banner's max_total_num_tokens; ignore_eos reserves input + max_new for
    every request, so b x (c_in + O) <= frac x pool is the timing-free sufficient condition (cost-term harness rule)."""
    ci, o = c_in(c), o_tokens(c)
    r = []
    if cap is None or b > cap:
        r.append(f"b {b} > cap {cap}")
    if not capturable:
        r.append(f"b {b} not in the cuda graph capture list")
    if kv_tokens is None:
        r.append("no max_total_num_tokens banner")
    elif b * (ci + o) > P("fb_kv_frac") * kv_tokens:
        r.append(f"b x (c_in + O) = {b * (ci + o)} > {P('fb_kv_frac')} x {kv_tokens}")
    if context is None or ci + o >= context:
        r.append(f"c_in + O = {ci + o} >= context_length {context}")
    if ci < 1:
        r.append(f"c_in {ci} < 1")
    return r


def predicted_kv_tokens(model, cap, mem=FB_MEM):
    """Banner token pool a boot is expected to show [derived]: (mem x visible - weights - (cap + 1) state slots) / k."""
    slots = (cap + 1) if s_bytes(model) else 0
    return int((mem * AVAIL_BYTES - weight_bytes(model) - slots * s_bytes(model)) // k_bytes(model))


def boot_cells(model, rule=None, mem=FB_MEM):
    """{cap: [(b, c), ...]} -- which boot (cap) measures which predicted cell.
    cap_equals_batch: one boot per batch size.  min_boots: greedy from the largest batch; a boot takes every remaining cell that the
    predicted banner of its cap admits."""
    rule = rule or P("fb_boot_rule")
    cells = predicted_cells(model, mem)
    if rule == "cap_equals_batch":
        return {b: [x for x in cells if x[0] == b] for b in BATCHES if any(x[0] == b for x in cells)}
    if rule != "min_boots":
        raise ValueError(f"unknown boot rule {rule}")
    out, rest = {}, list(cells)
    while rest:
        cap = max(b for b, _ in rest)
        take = [x for x in rest if not fits_banner(x[0], x[1], predicted_kv_tokens(model, cap, mem), cap, P("fb_context_length"))]
        take = take or [x for x in rest if x[0] == cap]                 # the cap's own row is always attempted (the banner decides)
        out[cap] = take
        rest = [x for x in rest if x not in take]
    return out


def predicted_infeasible(model, rule=None, mem=FB_MEM):
    """Predicted cells that the run-time rule would refuse under the PREDICTED banner of their boot (margin + output reservation)."""
    out = []
    for cap, cells in boot_cells(model, rule, mem).items():
        kv = predicted_kv_tokens(model, cap, mem)
        out += [(cap, b, c) for b, c in cells if fits_banner(b, c, kv, cap, P("fb_context_length"))]
    return out


# ---- arms ----------------------------------------------------------------------------------------------------------------------------------
def _params(key):
    return next(p for p in MS.ARM_PARAMS if p["key"] == key)


def base_fixed(rehearsal=False):
    """The one fixed value that is not this measurement's subject: the py3.12 venv freeze digest (mo_spec's, placeholder included).
    Both parts boot SGLang only, so the vLLM banner placeholders are not carried."""
    f = MS.REHEARSAL_FIXED if rehearsal else MS.default_fixed()
    return {"sg_venv_freeze": f["sg_venv_freeze"]}


def fb_name(p):
    return (f"{MS.bundle_name(p)}_fixedbatch_{MODELS[p['model']]['tag']}_cap{p['cap']}_mem{p['mem']}_ctx{p['context']}")


def fb_arm(model, cap, mem=FB_MEM, rehearsal=False):
    """Fixed-batch boot: the registered chunked arm with this boot's model, cap, mem and context.  No PD-mux flag, no PDMUX_* env."""
    ctx = P("fb_context_length")
    fixed = dict(base_fixed(rehearsal), mem=mem, cap=cap, context=ctx)
    p = dict(_params(P("fb_arm")), model=model, cap=cap, mem=mem, context=ctx, measurement="fixed_batch")
    a = MS.build_arm(_params(P("fb_arm")), fixed)
    if MODELS[model]["weights"] == "dummy":
        i = a["cmd"].index(SP.NANO_ARGS[0])
        a["cmd"] = a["cmd"][:i] + ["--model-path", "{dir}/" + VARIANT_DIRNAME, "--load-format", "dummy"] + a["cmd"][i + len(SP.NANO_ARGS):]
    a.update(name=fb_name(p), params=p, fixed=fixed, role="fixed_batch", model=model)
    return a


def sv_name(p):
    return f"{MS.bundle_name(p)}_cap{p['cap']}_mem{p['mem']}"


def sv_arm(key, mem, cap, rehearsal=False, split_check=False):
    fixed = dict(base_fixed(rehearsal), mem=mem, cap=cap, context=P("sv_context"))
    a = MS.split_check_arm(fixed) if split_check else MS.build_arm(_params(key), fixed)
    p = dict(a["params"], cap=cap, mem=mem)
    a.update(name=sv_name(p), params=p, fixed=fixed, combo=[mem, cap])
    return a


def _argv_value(cmd, flag):
    return cmd[cmd.index(flag) + 1] if flag in cmd else None


def arm_states_its_values(a):
    """The mem / cap in the name are the ones in the argv (a name that only CLAIMS a value is the defect the card exists for)."""
    p, c = a["params"], a["cmd"]
    if a["engine"] == "vllm":
        return _argv_value(c, "--max-num-seqs") == str(p["cap"])
    return (_argv_value(c, "--max-running-requests") == str(p["cap"]) and _argv_value(c, "--max-mamba-cache-size") == str(p["cap"])
            and _argv_value(c, "--mem-fraction-static") == str(p["mem"]))


# ---- fixed-batch plan ---------------------------------------------------------------------------------------------------------------------
def _rng(*salt):
    return SP._rng("motivation_cap", *salt)


def _order_cells(job, boot_id, cells):
    """Seeded order; TIME_ONLY contexts last and ascending (a server death there cannot take a shorter context with it)."""
    norm = [x for x in cells if not time_only(x[1])]
    _rng("cells", job, boot_id).shuffle(norm)
    return [list(x) for x in norm] + [list(x) for x in sorted(x for x in cells if time_only(x[1]))]


def fb_boot_keys(rule=None):
    return [(m, cap) for m in MODELS for cap in sorted(boot_cells(m, rule))]


def fb_boot_order(job, rnd, rule=None):
    keys = fb_boot_keys(rule)
    _rng("fixed_order", job, rnd).shuffle(keys)
    return keys


def fixed_plan(job, rounds=None, rehearsal=False, rule=None, only=None):
    """One Job: anchor boot -> for every round, every (model, cap) boot once in a seeded permutation, its cells in seeded order -> anchor
    boot.  Rehearsal: round 1, the real boot list, ONE cell per boot (its shortest context).  `only` = {(model, cap): [(b, c), ...]}
    selects boots and cells of the table for a smoke pass; a selection is a rehearsal plan by definition (the checker refuses it otherwise)."""
    rule = rule or P("fb_boot_rule")
    rounds = (1,) if rehearsal else tuple(rounds or P("fb_rounds"))
    am, ab, ac = P("fb_anchor")
    sel = None if only is None else {f"{m}|{cap}": [list(c) for c in cells] for (m, cap), cells in only.items()}

    def boot(bid, model, cap, cells, role, rnd, **kw):
        a = fb_arm(model, cap, rehearsal=rehearsal)
        return {"boot_id": bid, "arm": a["name"], "arm_def": a, "role": role, "round": rnd, "model": model, "cap": cap,
                "cells": cells, "runs": [], **kw}

    boots = [boot(f"{job}_anchor_start", am, ab, [[ab, ac]], "anchor", 0, anchor="start")]
    for rnd in rounds:
        for model, cap in fb_boot_order(job, rnd, rule):
            if sel is not None and f"{model}|{cap}" not in sel:
                continue
            cells = boot_cells(model, rule)[cap]
            bid = f"{job}_r{rnd}_{model}_cap{cap}"
            if sel is not None:
                cells = _order_cells(job, bid, [tuple(c) for c in sel[f"{model}|{cap}"]])
            else:
                cells = [list(min(cells, key=lambda x: (x[1], -x[0])))] if rehearsal else _order_cells(job, bid, cells)
            boots.append(boot(bid, model, cap, cells, "meas", rnd))
    boots.append(boot(f"{job}_anchor_end", am, ab, [[ab, ac]], "anchor", 0, anchor="end"))
    return {"job": job, "kind": KIND_FIXED, "rehearsal": bool(rehearsal), "rounds": list(rounds), "boot_rule": rule, "mem": FB_MEM, "only": sel,
            "context_length": P("fb_context_length"), "boot_order": [b["arm"] for b in boots], "boots": boots,
            "cell_table": {m: len(predicted_cells(m)) for m in MODELS}, "anchor": {"model": am, "cell": [ab, ac], "positions": ["first", "last"]},
            "open_items": open_items(), "workload": "W1"}


def _common_arm_problems(b, rehearsal, allow_placeholders):
    out, a = [], b["arm_def"]
    if a["engine"] == "sglang" and a["python"] != MS.PY312:
        out.append(f"{b['boot_id']}: SGLang arm must use {MS.PY312}")
    if a["tree"] == "pristine" and (a["env"] or any(x in ("--enable-pdmux", "--pdmux-config-path") for x in a["cmd"])):
        out.append(f"{b['boot_id']}: pristine boot with PD-mux env/argv")
    if not arm_states_its_values(a):
        out.append(f"{b['boot_id']}: the argv does not carry the cap / mem the arm name states")
    if not allow_placeholders and not rehearsal and (any(x in (MS.MEM_TOKEN, MS.CAP_TOKEN) for x in a["cmd"])
                                                     or any(MS.is_pending(v) for v in a["fixed"].values())):
        out.append(f"{b['boot_id']}: PLACEHOLDER in the boot's fixed values {sorted(k for k, v in a['fixed'].items() if MS.is_pending(v))}")
    return out


def fixed_plan_problems(plan, allow_placeholders=False, confirmed=None):
    out = []
    boots, job = plan["boots"], plan["job"]
    bids = [b["boot_id"] for b in boots]
    if len(bids) != len(set(bids)):
        out.append("duplicate boot_id")
    if plan["boot_order"] != [b["arm"] for b in boots]:
        out.append("plan.boot_order differs from the boots")
    if boots[0]["role"] != "anchor" or boots[-1]["role"] != "anchor":
        out.append("anchor boot must be first and last")
    am, ab, ac = P("fb_anchor")
    for b in (boots[0], boots[-1]):
        if (b["model"], b["cap"], b["cells"]) != (am, ab, [[ab, ac]]):
            out.append(f"{b['boot_id']}: not the registered anchor cell")
    rule, sel = plan["boot_rule"], plan.get("only")
    if sel is not None and not plan["rehearsal"]:
        out.append("a cell selection is a rehearsal-only plan")
    for rnd in plan["rounds"]:
        got = [(b["model"], b["cap"]) for b in boots if b["role"] == "meas" and b["round"] == rnd]
        if got != [k for k in fb_boot_order(job, rnd, rule) if sel is None or f"{k[0]}|{k[1]}" in sel]:
            out.append(f"round {rnd}: boot order differs from the registered permutation of job '{job}'")
        for m in MODELS:
            cells = sorted(tuple(c) for b in boots if b["role"] == "meas" and b["round"] == rnd and b["model"] == m for c in b["cells"])
            want = sorted(predicted_cells(m))
            if not plan["rehearsal"] and cells != want:
                out.append(f"round {rnd} {m}: {len(cells)} cells != the {len(want)}-cell table (every memory-permitted cell exactly once)")
            if plan["rehearsal"] and any(c not in want for c in cells):
                out.append(f"round {rnd} {m}: a rehearsal cell is outside the cell table")
    if plan["cell_table"] != {m: len(predicted_cells(m)) for m in MODELS}:
        out.append(f"plan.cell_table {plan['cell_table']} != the cell rule's table")
    for b in boots:
        a = b["arm_def"]
        if a["name"] != fb_name(a["params"]) or b["arm"] != a["name"]:
            out.append(f"{b['boot_id']}: arm name does not state its bundle")
        if a["params"]["mem"] != FB_MEM or a["params"]["cap"] != b["cap"] or a["params"]["model"] != b["model"]:
            out.append(f"{b['boot_id']}: mem / cap / model differ from the registered boot")
        if a["tree"] != "pristine" or a["env"]:
            out.append(f"{b['boot_id']}: a fixed-batch boot is an un-instrumented pristine boot")
        if any(c[0] > b["cap"] for c in b["cells"]):
            out.append(f"{b['boot_id']}: a cell's batch exceeds the boot cap")
        if rule == "cap_equals_batch" and b["role"] == "meas" and any(c[0] != b["cap"] for c in b["cells"]):
            out.append(f"{b['boot_id']}: boot rule cap_equals_batch violated")
        t = [time_only(c[1]) for c in b["cells"]]
        if t != sorted(t):
            out.append(f"{b['boot_id']}: TIME_ONLY contexts must come last")
        dummy = "--load-format" in a["cmd"]
        if dummy != (MODELS[b["model"]]["weights"] == "dummy"):
            out.append(f"{b['boot_id']}: weight loading differs from the registered model")
        out += _common_arm_problems(b, plan["rehearsal"], allow_placeholders)
    if not plan["rehearsal"] and open_items(confirmed):
        out.append(f"UNCONFIRMED design values (user decision needed): {open_items(confirmed)}")
    return out


# ---- serving cap/mem plan --------------------------------------------------------------------------------------------------------------------
def sv_keys():
    return [(k, mem, cap) for k in P("sv_arms") for mem, cap in SERVING_COMBOS]


def sv_boot_order(job):
    keys = sv_keys()
    _rng("serving_order", job).shuffle(keys)
    return keys


def serving_plan(job, workload, rmin, factor, seeds=None, run_s=None, n_requests=None, rehearsal=False):
    """One Job: anchor boot -> every (arm, mem, cap) once in a seeded permutation, each with all rungs x seeds in seeded cell order ->
    one split-check boot of the bundle (anchor combination; never in a table) -> anchor boot.  A cell holds round(rate x run_s) requests
    (or n_requests when given: rehearsal)."""
    if workload != "W1" and not MS.MW.is_registered(workload):
        raise ValueError(f"unregistered workload {workload!r}")
    seeds = tuple(seeds) if seeds is not None else ((MS.default_seeds(workload)[0],) if rehearsal else MS.default_seeds(workload))
    grid = MS.make_grid(rmin, factor, 1 if rehearsal else SERVING_RUNGS)
    run_s = P("sv_run_s") if run_s is None else run_s

    def n_of(rate):
        return int(n_requests) if n_requests else max(1, int(round(rate * run_s)))

    amem, acap = P("sv_anchor_combo")
    anchor = sv_arm(MS.ANCHOR_KEY, amem, acap, rehearsal)
    arate = grid["rates"][MS.ANCHOR_RATE_INDEX]

    def anchor_boot(tag):
        r = MS.cell_run(job, anchor["name"], arate, seeds[0], n_of(arate) if rehearsal else MS.ANCHOR_N, f"ANCHOR_{tag}", kind="anchor",
                        workload=workload)
        return SP.boot(f"{job}_anchor_{tag}", anchor, [r], role="anchor", anchor=tag)

    boots = [anchor_boot("start")]
    for key, mem, cap in sv_boot_order(job):
        a = sv_arm(key, mem, cap, rehearsal)
        cells = [MS.cell_run(job, a["name"], r, s, n_of(r), "GRID", workload=workload) for r in grid["rates"] for s in seeds]
        _rng("serving_cells", job, a["name"]).shuffle(cells)
        boots.append(SP.boot(f"{job}_{a['name']}", a, cells, role="meas"))
    if MS.BUNDLE_KEY in P("sv_arms"):
        sc = sv_arm(MS.BUNDLE_KEY, amem, acap, rehearsal, split_check=True)
        r = MS.cell_run(job, sc["name"], grid["rates"][-1], seeds[0], n_of(grid["rates"][-1]) if rehearsal else MS.ANCHOR_N, "SPLIT_CHECK",
                        kind="split_check", workload=workload)
        boots.append(SP.boot(f"{job}_splitcheck", sc, [r], role="split_check"))
    boots.append(anchor_boot("end"))
    return {"job": job, "kind": KIND_SERVING, "workload": workload, "grid": grid, "seeds": list(seeds), "run_s": run_s,
            "n_requests": n_requests, "rehearsal": bool(rehearsal), "fixed": dict(anchor["fixed"]), "combos": [list(c) for c in SERVING_COMBOS],
            "boot_order": [b["arm"] for b in boots], "arm_keys": list(P("sv_arms")),
            "anchor": {"arm": anchor["name"], "rate": arate, "seed": seeds[0], "n": boots[0]["runs"][0]["n_requests"],
                       "positions": ["first", "last"]}, "boots": boots, "open_items": open_items(), "pending": []}


def serving_plan_problems(plan, allow_placeholders=False, confirmed=None):
    out = []
    boots, job = plan["boots"], plan["job"]
    ids = [r["run_id"] for b in boots for r in b["runs"]]
    if len(ids) != len(set(ids)):
        out.append("duplicate run_id")
    if len({b["boot_id"] for b in boots}) != len(boots):
        out.append("duplicate boot_id")
    if plan["boot_order"] != [b["arm"] for b in boots]:
        out.append("plan.boot_order differs from the boots")
    if boots[0]["role"] != "anchor" or boots[-1]["role"] != "anchor":
        out.append("anchor boot must be first and last")
    amem, acap = P("sv_anchor_combo")
    for b in (boots[0], boots[-1]):
        p = b["arm_def"]["params"]
        if (p["key"], p["mem"], p["cap"]) != (MS.ANCHOR_KEY, amem, acap):
            out.append(f"{b['boot_id']}: anchor boot is not the anchor arm at the anchor combination")
    meas = [b for b in boots if b["role"] == "meas"]
    got = [(b["arm_def"]["params"]["key"], b["arm_def"]["params"]["mem"], b["arm_def"]["params"]["cap"]) for b in meas]
    if got != sv_boot_order(job):
        out.append(f"boot order differs from the registered permutation of job '{job}' (every arm x combination exactly once)")
    rates, seeds = plan["grid"]["rates"], plan["seeds"]
    for b in meas:
        if sorted((r["rate"], r["seed"]) for r in b["runs"]) != sorted((r, s) for r in rates for s in seeds):
            out.append(f"{b['boot_id']}: cells != grid rungs x seeds (full-block rule)")
    n_by = {}
    for b in meas:
        for r in b["runs"]:
            n_by.setdefault((r["rate"], r["seed"]), set()).add(r["n_requests"])
    if any(len(v) > 1 for v in n_by.values()):
        out.append("a (rate, seed) cell has different request counts across arms")
    if not plan["rehearsal"]:
        if len(seeds) < SERVING_SEEDS_MIN:
            out.append(f"n seeds {len(seeds)} < {SERVING_SEEDS_MIN} (gate 3)")
        if len(rates) != SERVING_RUNGS:
            out.append(f"{len(rates)} rungs != {SERVING_RUNGS}")
        if open_items(confirmed):
            out.append(f"UNCONFIRMED design values (user decision needed): {open_items(confirmed)}")
    if any(r["workload"] != plan["workload"] for b in boots for r in b["runs"]):
        out.append("every run of a Job uses the plan's workload")
    pm_env = SP.arm("PM-ref")["env"]
    for b in boots:
        a = b["arm_def"]
        if a["name"] != sv_name(a["params"]) or b["arm"] != a["name"]:
            out.append(f"{b['boot_id']}: arm name does not state its bundle, cap and mem")
        if (a["params"]["mem"], a["params"]["cap"]) not in SERVING_COMBOS:
            out.append(f"{b['boot_id']}: (mem, cap) is not a registered combination")
        if b["role"] == "split_check":
            if a["env"] != {**pm_env, **MS.MRZ.SPLIT_CHECK_ENV} or a["tree"] != "patched":
                out.append(f"{b['boot_id']}: split-check boot env must be the bundle env + mo_realized.SPLIT_CHECK_ENV exactly")
        elif a["engine"] == "sglang" and any(k.startswith("PDMUX_") and pm_env.get(k) != v for k, v in a["env"].items()):
            out.append(f"{b['boot_id']}: instrumentation env in a measurement boot")
        out += _common_arm_problems(b, plan["rehearsal"], allow_placeholders)
    return out


def plan_problems(plan, allow_placeholders=False, confirmed=None):
    fn = {KIND_FIXED: fixed_plan_problems, KIND_SERVING: serving_plan_problems}[plan["kind"]]
    return fn(plan, allow_placeholders, confirmed)


# ---- Job time estimate ([estimate]; inputs: measured 108-SM fits of dstar_model for Nano, analytic ratios for a31) -------------------------
BOOT_MIN = SP.BOOT_OH_MIN            # 6 min per boot (roadmap GPU-time table)
JOB_START_MIN = 20.0                 # roadmap: Job start (image pull, tree install)
SETTLE_S = 0.5


def _ratio(model, fn):
    return fn(arch(model)) / fn(arch("nano"))


def est_prefill_s(model, b, n_in):
    """b requests of n_in tokens through chunked prefill [estimate]: Nano = measured single-request fit u + v L + w L^2 per request;
    a31 = the same with v scaled by the analytic linear FLOPs ratio and w by the attention-layer ratio."""
    u, v, w = DM._avg(*DM.T2_108)
    v *= _ratio(model, lambda a: AT.prefill_flops(a, 1)["linear_per_token"])
    w *= _ratio(model, lambda a: AT.prefill_flops(a, 1)["quad_coeff"])
    return b * (u + v * n_in + w * n_in * n_in) / 1000.0


def est_step_ms(model, b, c):
    """Decode step [estimate]: Nano = measured 108-SM fit (extrapolated above b 96 / c 64k); a31 = that x the roofline ratio a31 / Nano."""
    t = DM.t_dec("108u", b, c)
    return t if model == "nano" else t * CT.step_ms_lb(b, c, arch(model)) / CT.step_ms_lb(b, c, arch("nano"))


def est_cell_s(model, b, c):
    return est_prefill_s(model, b, c_in(c)) + (abort_at(c) + P("fb_ramp_steps")) * est_step_ms(model, b, c) / 1000.0 + SETTLE_S


def estimate_fixed(plan):
    cells = [(b["model"], c[0], c[1]) for b in plan["boots"] for c in b["cells"]]
    cell_min = sum(est_cell_s(*x) for x in cells) / 60.0
    boot_min = BOOT_MIN * len(plan["boots"])
    return {"boots": len(plan["boots"]), "cells": len(cells), "cell_min": round(cell_min, 1), "boot_min": boot_min, "start_min": JOB_START_MIN,
            "total_min": round(cell_min + boot_min + JOB_START_MIN, 1), "total_h": round((cell_min + boot_min + JOB_START_MIN) / 60.0, 2),
            "basis": "boot 6 min, start 20 min (roadmap); cell = chunked fill (dstar_model T2 fit) + (abort_at + ramp) x step (dstar_model "
                     "T1 108u fit); a31 scaled by analytic_terms ratios; [estimate], extrapolated above b 96 / c 64k"}


def estimate_serving(plan):
    runs = [r for b in plan["boots"] for r in b["runs"]]
    load_min = sum(r["n_requests"] / r["rate"] for r in runs) / 60.0
    drain_min = SP.DRAIN_MIN * len(runs)
    boot_min = BOOT_MIN * len(plan["boots"])
    tot = load_min + drain_min + boot_min + JOB_START_MIN
    return {"boots": len(plan["boots"]), "runs": len(runs), "load_h": round(load_min / 60.0, 2), "drain_min": round(drain_min, 1),
            "boot_min": boot_min, "start_min": JOB_START_MIN, "total_h": round(tot / 60.0, 2),
            "basis": "load = sum n / rate, drain 0.6 min per run (sb_spec.DRAIN_MIN), boot 6 min, start 20 min"}


def main(argv=None):
    import argparse
    import json
    ap = argparse.ArgumentParser(description="cell table / plan summary of the cap measurement (CPU)")
    ap.add_argument("what", choices=("table", "fixed", "serving", "open"))
    ap.add_argument("--job", default="J")
    ap.add_argument("--workload", default="W1")
    ap.add_argument("--rmin", type=float, default=1.0)
    ap.add_argument("--factor", type=float, default=1.5)
    ap.add_argument("--rule")
    a = ap.parse_args(argv)
    if a.what == "table":
        out = {"cells": cell_table_summary(), "gap_vs_user_stated": cell_count_gap(), "boots": {m: {r: len(boot_cells(m, r)) for r in ("cap_equals_batch", "min_boots")} for m in MODELS},
               "refused_under_predicted_banner": {m: predicted_infeasible(m, a.rule) for m in MODELS}}
    elif a.what == "fixed":
        p = fixed_plan(a.job, rule=a.rule)
        out = {"plan_sha": MS.sha(p), "problems": fixed_plan_problems(p), "estimate": estimate_fixed(p)}
    elif a.what == "serving":
        p = serving_plan(a.job, a.workload, a.rmin, a.factor)
        out = {"plan_sha": MS.sha(p), "problems": serving_plan_problems(p), "estimate": estimate_serving(p), "boot_order": p["boot_order"]}
    else:
        out = {"open_items": {k: {"value": PROPOSED[k][0], "source": PROPOSED[k][1]} for k in open_items()},
               "pending_thresholds": FIX_RULE}
    print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
