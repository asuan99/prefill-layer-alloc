#!/usr/bin/env python3
"""Pre-ladder per workload (roadmap step 6): a short rate search whose ONLY use is choosing the rate grids of the later measurements --
(a) the stage-1 grids (sharegpt_first, openchat_rounds; arxiv_summ for stage 2), (b) the grid of the serving cap/mem comparison,
(c) the three short rates of the LooGLE interference cell and the one rate of its rehearsal.  CPU-tested against fakes only.

NOT A MEASUREMENT.  Nothing a pre-ladder produces enters a verdict table or may be cited; every output carries NOT_A_MEASUREMENT.
Consequences, all deliberate:
  * it does not wait for the cap measurement or the venv freeze digest.  Boots run on the PROVISIONAL values PROVISIONAL (cap 48, mem 0.6:
    the values inherited from the earlier A100 campaigns, mo_spec.REHEARSAL_FIXED), stated in every arm name and record; the venv freeze
    digest and the vLLM banner values are recorded as observed and not compared (card expectation None);
  * no rehearsal gate, no anchor boot, no split-check boot: the ladder is itself the exploration, a failed boot costs only that boot's
    points, and the GPU full-pass rehearsals of the measurements stay the gate of the measurements;
  * its seeds (mw_lib.PRE_LADDER_SEEDS, two as in the earlier capacity ladders) are disjoint from every measurement seed.
Kept from the measurement harness: per-boot card check (argv, banner, tree, interpreter, order, traces), raw schema, the arm name
stating the whole bundle, seeded boot / cell order, one Job = one unit (ladders of two Jobs are never merged).

Targets (TARGETS): a workload at a boot context.  interference_short = the sharegpt_first stream at context 96,000, i.e. the short
side of the interference pair under the interference cell's own boot configuration.
Ladder shape per target (PROPOSED: factor, rungs, requests per cell = the earlier capacity ladders ph_spec.CAP_*); the lowest rate of a
target has NO source and is a required input.  Which arms climb the ladder is a required input too (ARM_OPTIONS lists the choices).

  mo_ladder.py plan  --job J --arms K1,K2 --grids T:RMIN[:FACTOR:N],...
  mo_ladder.py card  --job J ... --w1-lengths L.json --out card.json
  mo_ladder.py run   --job J ... --w1-lengths L.json --card card.json --out RAW [--resume]
  mo_ladder.py options --grids ...                      Job time of every arm option (CPU)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(os.path.abspath(__file__)).parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import mo_spec as MS  # noqa: E402
import mo_card as MC  # noqa: E402
import mo_runner as MR  # noqa: E402
import mo_loadgen as ML  # noqa: E402
import mo_intf_spec as IS  # noqa: E402

SP, MW = MS.SP, MS.MW
L = MW.L
KIND = "PRE_LADDER"
NOT_A_MEASUREMENT = ("NOT_A_MEASUREMENT: pre-ladder, rate-grid selection only -- provisional cap / mem, no anchor, no rehearsal gate; "
                     "no value of it enters a verdict table or may be cited")
PROVISIONAL = {"mem": MS.REHEARSAL_FIXED["mem"], "cap": MS.REHEARSAL_FIXED["cap"]}     # 0.6 / 48: inherited, NOT the cap measurement's result
SEEDS = tuple(L.PRE_LADDER_SEEDS)
TARGETS = {
    "sharegpt_first": {"workload": "sharegpt_first", "context": MS.default_fixed()["context"], "shape": "chat", "feeds": "stage-1 grid"},
    "openchat_rounds": {"workload": "openchat_rounds", "context": MS.default_fixed()["context"], "shape": "chat",
                        "feeds": "stage-1 grid; serving cap/mem grid if this workload is chosen for it"},
    "arxiv_summ": {"workload": "arxiv_summ", "context": MS.default_fixed()["context"], "shape": "long", "feeds": "stage-2 grid"},
    "interference_short": {"workload": "sharegpt_first", "context": IS.CONTEXT, "shape": "chat",
                           "feeds": "the three short rates of the interference cell and its rehearsal rate"},
}
# Ladder shape = the earlier capacity ladders of the sealed policy harness (ph_spec.CAP_FACTOR / CAP_N_RATES / CAP_CELL_REQUESTS:
# short chat W1 and long W3).  PROPOSED: confirmed by the user before the pre-ladder Job, like mo_cap_spec.PROPOSED.
SHAPES = {"chat": {"factor": SP.P.CAP_FACTOR["W1"], "n": SP.P.CAP_N_RATES["W1"], "n_requests": SP.P.CAP_CELL_REQUESTS["W1"]},
          "long": {"factor": SP.P.CAP_FACTOR["W3"], "n": SP.P.CAP_N_RATES["W3"], "n_requests": SP.P.CAP_CELL_REQUESTS["W3"]}}
# Which arms climb the ladder has no documented answer (a grid must contain the capacity boundary of EVERY arm of the measurement it
# feeds; which arm is the smallest / largest is exactly what is unknown before the ladder).  The choices, for the user:
ARM_OPTIONS = {
    "all_arms": {"default": MS.ARM_KEYS, "interference_short": IS.ARM_KEYS},
    "baseline_and_bundle": {"default": (MS.PRIMARY_KEY, MS.BUNDLE_KEY), "interference_short": (MS.PRIMARY_KEY, "PDMUX_D16", "PDMUX_D44")},
    "primary_baseline_only": {"default": (MS.PRIMARY_KEY,), "interference_short": (MS.PRIMARY_KEY,)},
}


def parse_grids(text):
    """'target:rmin[:factor:n],...' -> {target: {"rmin", "factor", "n", "n_requests"}}; factor / n default to the target's SHAPES."""
    out = {}
    for item in [x for x in text.split(",") if x]:
        p = item.split(":")
        if p[0] not in TARGETS or len(p) not in (2, 4):
            raise ValueError(f"grid '{item}': target:rmin or target:rmin:factor:n with target in {sorted(TARGETS)}")
        sh = SHAPES[TARGETS[p[0]]["shape"]]
        out[p[0]] = {"rmin": float(p[1]), "factor": float(p[2]) if len(p) == 4 else sh["factor"], "n": int(p[3]) if len(p) == 4 else sh["n"],
                     "n_requests": sh["n_requests"]}
    return out


def arm_keys_for(target, arms):
    """The arms of one target.  The interference target accepts only the interference cell's arms (D arms + primary baseline)."""
    pool = IS.ARM_KEYS if target == "interference_short" else MS.ARM_KEYS
    return tuple(k for k in arms if k in pool)


def ladder_name(a, fixed):
    return f"{MS.bundle_name(a['params'])}_preladder_cap{fixed['cap']}_mem{fixed['mem']}_ctx{fixed['context']}"


def build_arm(key, context):
    """A measurement arm on the provisional cap / mem at the target's context; nothing the card compares is a placeholder (the venv
    freeze digest and the vLLM banner values are observed, not expected)."""
    fixed = {**PROVISIONAL, "context": context, "sg_venv_freeze": None, "vllm_chunk": None, "vllm_backend": None}
    a = IS.build_arm(key, fixed) if key.startswith("PDMUX_D") else MS.build_arm(next(p for p in MS.ARM_PARAMS if p["key"] == key), fixed)
    a.update(name=ladder_name(a, fixed), fixed=fixed, arm_key=key, provisional=dict(PROVISIONAL))
    return a


def _rng(*salt):
    return SP._rng("motivation_pre_ladder", *salt)


def boot_keys(grids, arms):
    return [(t, k) for t in TARGETS if t in grids for k in arm_keys_for(t, arms)]


def boot_order(job, grids, arms):
    keys = boot_keys(grids, arms)
    _rng("order", job).shuffle(keys)
    return keys


def rates_of(g):
    return SP.P.ladder(g["rmin"], g["factor"], g["n"])


def make_plan(job, grids, arms):
    """One Job: every (target, arm) once in a seeded permutation; each boot runs its target's ladder x the two pre-ladder seeds in
    seeded cell order."""
    boots = []
    for t, k in boot_order(job, grids, arms):
        tg, g = TARGETS[t], grids[t]
        a = build_arm(k, tg["context"])
        cells = [MS.cell_run(f"{job}_{t}", a["name"], r, s, g["n_requests"], "PRE_LADDER", kind="pre_ladder", workload=tg["workload"], target=t)
                 for r in rates_of(g) for s in SEEDS]
        _rng("cells", job, t, k).shuffle(cells)
        boots.append(SP.boot(f"{job}_{t}_{a['name']}", a, cells, role="pre_ladder", target=t))
    return {"job": job, "kind": KIND, "label": NOT_A_MEASUREMENT, "rehearsal": False, "grids": grids, "arms": list(arms), "seeds": list(SEEDS),
            "provisional": dict(PROVISIONAL), "boot_order": [b["arm"] for b in boots], "boots": boots, "workload": "W1",
            "fixed": {**PROVISIONAL, "context": None}, "pending": []}


def plan_problems(plan, allow_placeholders=False):
    out = []
    boots, job, grids, arms = plan["boots"], plan["job"], plan["grids"], plan["arms"]
    ids = [r["run_id"] for b in boots for r in b["runs"]]
    if len(ids) != len(set(ids)):
        out.append("duplicate run_id")
    if len({b["boot_id"] for b in boots}) != len(boots):
        out.append("duplicate boot_id")
    if plan["boot_order"] != [b["arm"] for b in boots]:
        out.append("plan.boot_order differs from the boots")
    if plan.get("label") != NOT_A_MEASUREMENT:
        out.append("a pre-ladder plan carries the NOT_A_MEASUREMENT label")
    if not boots:
        out.append("no boot: no arm of the given list belongs to a target")
    if [(b["target"], b["arm_def"]["arm_key"]) for b in boots] != boot_order(job, grids, arms):
        out.append(f"boot order differs from the registered permutation of job '{job}' (every target x arm exactly once)")
    for t, g in grids.items():
        if t not in TARGETS or g["rmin"] <= 0 or g["factor"] <= 1 or g["n"] < 2:
            out.append(f"grid of {t}: rmin > 0, factor > 1, n >= 2 on a registered target")
        if not arm_keys_for(t, arms):
            out.append(f"target {t}: none of the arms {arms} belongs to it")
    if set(plan["seeds"]) != set(SEEDS) or set(SEEDS) & (set(L.SEEDS) | set(L.SPARE_SEEDS) | set(L.SECOND_BLOCK_SEEDS)):
        out.append("pre-ladder seeds must be the registered pre-ladder seeds, disjoint from every measurement seed")
    for b in boots:
        a, tg, g = b["arm_def"], TARGETS[b["target"]], grids[b["target"]]
        if sorted((r["rate"], r["seed"]) for r in b["runs"]) != sorted((r, s) for r in rates_of(g) for s in SEEDS):
            out.append(f"{b['boot_id']}: cells != ladder rungs x pre-ladder seeds")
        if any(r["workload"] != tg["workload"] or r["n_requests"] != g["n_requests"] for r in b["runs"]):
            out.append(f"{b['boot_id']}: a cell is not the target's workload / request count")
        if a["name"] != ladder_name(a, a["fixed"]) or b["arm"] != a["name"]:
            out.append(f"{b['boot_id']}: arm name does not state its bundle and the provisional cap / mem / context")
        c = a["cmd"]
        ok = (c[c.index("--max-num-seqs") + 1] == str(PROVISIONAL["cap"])) if a["engine"] == "vllm" else (
            c[c.index("--max-running-requests") + 1] == str(PROVISIONAL["cap"]) and c[c.index("--mem-fraction-static") + 1] == str(PROVISIONAL["mem"])
            and c[c.index("--context-length") + 1] == str(tg["context"]))
        if not ok or a["fixed"]["context"] != tg["context"]:
            out.append(f"{b['boot_id']}: the argv does not carry the provisional cap / mem and the target's context")
        if a["engine"] == "sglang" and a["python"] != MS.PY312:
            out.append(f"{b['boot_id']}: SGLang arm must use {MS.PY312}")
        if a["tree"] == "pristine" and (a["env"] or any(x in ("--enable-pdmux", "--pdmux-config-path") for x in c)):
            out.append(f"{b['boot_id']}: pristine boot with PD-mux env/argv")
        pm_env = SP.arm("PM-ref")["env"]
        if a["engine"] == "sglang" and any(k.startswith("PDMUX_") and k not in pm_env for k in a["env"]):
            out.append(f"{b['boot_id']}: instrumentation env in a pre-ladder boot")
    return out


# ---- card -------------------------------------------------------------------------------------------------------------------------------
def build_card(plan, w1_pool):
    """Card of a pre-ladder plan: per-boot arm expectations (mo_card.arm_expect on the boot's own provisional values), trace facts of
    every run, and the pool / token facts of EVERY workload of the Job."""
    trace = {r["run_id"]: MC.trace_facts(MW.build_segments(r, w1_pool)) for b in plan["boots"] for r in b["runs"]}
    wls = sorted({r["workload"] for b in plan["boots"] for r in b["runs"]})
    card = {"schema": MS.SCHEMA_VERSION + "/card", "job": plan["job"], "kind": KIND, "label": NOT_A_MEASUREMENT, "rehearsal": False,
            "plan_sha": MS.sha(plan), "arms": MC.arms_expect(plan), "boot_order": list(plan["boot_order"]), "pool": MC.pool_facts(w1_pool),
            "workloads": {w: MW.pool_facts(w, w1_pool, MC.pool_facts) for w in wls}, "trace": trace, "grids": plan["grids"],
            "provisional": plan["provisional"]}
    card["card_sha"] = MC.sha({k: v for k, v in card.items() if k != "card_sha"})
    return card


# ---- Job ----------------------------------------------------------------------------------------------------------------------------------
def run_job(plan, card, out, make_server, replay_fn, w1_pool, resume=False, py_version_fn=MC.probe_python_version, freeze_fn=MC.freeze_sha,
            warmup_s=MR.WARMUP_S, token_dir=None):
    """mo_runner.run_boot per boot (same card check, warm-up, raw records).  No rehearsal gate.  Token files of every workload of the
    Job are resolved (sha256 = registered) BEFORE the first boot; a boot sees the store of its own workload."""
    out = Path(out)
    try:
        problems = plan_problems(plan)
        if problems:
            raise MR.JobAbort("ABORT_PLAN", str(problems))
        if not MC.card_sha_ok(card) or card["plan_sha"] != MS.sha(plan):
            raise MR.JobAbort("ABORT_CARD_PLAN", "card sha / plan sha mismatch: the card was not built from this plan")
        if out.exists() and any((out / "runs").glob("*.json")) and not resume:
            raise MR.JobAbort("ABORT_RAW_NOT_EMPTY", "raw directory has runs; set --resume")
        stores = {}
        try:
            for w in card["workloads"]:
                stores[w] = MW.store_for(w, token_dir)
        except (FileNotFoundError, ValueError) as e:
            raise MR.JobAbort("ABORT_TOKENS", str(e))
        MR.dump(plan, out / "plan.json")
        MR.dump(card, out / "card.json")
        ctx, res = {}, []
        for b in plan["boots"]:
            wl = b["runs"][0]["workload"]
            ctx["token_store"] = stores[wl]
            res.append(MR.run_boot(b, dict(card, workload=card["workloads"][wl]), out, make_server, replay_fn, w1_pool, ctx, py_version_fn,
                                   freeze_fn, None, warmup_s))
    except MR.JobAbort as e:
        MR.dump({"job": plan.get("job"), "status": e.status, "reason": e.reason, "boot_id": e.boot_id, **MR.stamp()}, out / "JOB_ABORTED.json")
        raise
    MR.dump({"job": plan["job"], "plan_sha": MS.sha(plan), "card_sha": card["card_sha"], "boots": res, "resumed": bool(resume),
             "label": NOT_A_MEASUREMENT, **MR.stamp()}, out / "RUN_DONE.json")
    return res


# ---- Job time estimate ----------------------------------------------------------------------------------------------------------------------
BOOT_MIN, JOB_START_MIN = SP.BOOT_OH_MIN, 20.0


def estimate(plan):
    runs = [r for b in plan["boots"] for r in b["runs"]]
    load_min = sum(r["n_requests"] / r["rate"] for r in runs) / 60.0
    drain_min = SP.DRAIN_MIN * len(runs)
    boot_min = BOOT_MIN * len(plan["boots"])
    tot = load_min + drain_min + boot_min + JOB_START_MIN
    return {"boots": len(plan["boots"]), "runs": len(runs), "load_h": round(load_min / 60.0, 2), "drain_min": round(drain_min, 1),
            "boot_min": boot_min, "start_min": JOB_START_MIN, "total_h": round(tot / 60.0, 2),
            "basis": "load = sum n / rate, drain 0.6 min per run (sb_spec.DRAIN_MIN), boot 6 min, start 20 min (roadmap)"}


def options(grids):
    """Job time of every arm option for the given grids (the arm list is the user's choice)."""
    out = {}
    for name, opt in ARM_OPTIONS.items():
        arms = tuple(dict.fromkeys(opt["default"] + (opt["interference_short"] if "interference_short" in grids else ())))
        p = make_plan("options", grids, arms)
        out[name] = {"arms": list(arms), **{k: v for k, v in estimate(p).items() if k != "basis"}}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("plan", "card", "run", "options"):
        p = sub.add_parser(n)
        p.add_argument("--job", default="J")
        p.add_argument("--grids", required=True, help="target:rmin[:factor:n],...  (the lowest rate has no default)")
        p.add_argument("--arms", help="arm keys, comma separated (no default: see ARM_OPTIONS)")
        p.add_argument("--w1-lengths")
        p.add_argument("--out")
        p.add_argument("--card")
        p.add_argument("--resume", action="store_true")
    a = ap.parse_args(argv)
    try:
        grids = parse_grids(a.grids)
    except ValueError as e:
        print(f"ABORT_CONFIG: {e}", file=sys.stderr)
        return 16
    if a.cmd == "options":
        print(json.dumps({"label": NOT_A_MEASUREMENT, "options": options(grids)}, indent=1))
        return 0
    if not a.arms:
        print(f"ABORT_CONFIG: --arms required (which arms climb the ladder is not decided; choices: {sorted(ARM_OPTIONS)})", file=sys.stderr)
        return 16
    plan = make_plan(a.job, grids, tuple(x for x in a.arms.split(",") if x))
    if a.cmd == "plan":
        print(json.dumps({"label": NOT_A_MEASUREMENT, "plan_sha": MS.sha(plan), "problems": plan_problems(plan), "boots": len(plan["boots"]),
                          "boot_order": plan["boot_order"], "estimate": estimate(plan)}, indent=1))
        return 0
    pool = json.load(open(a.w1_lengths))
    if a.cmd == "card":
        card = build_card(plan, pool)
        MR.dump(card, a.out)
        print(json.dumps({"card_sha": card["card_sha"], "placeholders": MC.card_placeholders(card), "label": NOT_A_MEASUREMENT}))
        return 0
    card = json.load(open(a.card))
    try:
        res = run_job(plan, card, a.out, lambda bdir, arm: MR.MoProcessServer(bdir, arm), ML.replay, pool, a.resume)
    except MR.JobAbort as e:
        print(f"{e.status}: {e.reason}", file=sys.stderr)
        return MR.EXIT_ABORT
    return MR.exit_code(res)


if __name__ == "__main__":
    sys.exit(main())
