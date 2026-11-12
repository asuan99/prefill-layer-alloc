#!/usr/bin/env python3
"""LooGLE interference cell over static D arms (roadmap 11-b and #dshift, user decision 2026-11-11 / 12): registered constants, arms,
Job plan, plan checker, time estimate.  CPU only; starts no server.

Design
  * Pair: the sharegpt_first stream at a fixed short rate WITH and WITHOUT the LooGLE long stream (5 % of all requests, count fixed;
    gen_traces.interference).  The two members of a pair share seed, rate and the byte-identical short stream.
  * One boot per arm serves BOTH members of every pair (cells in seeded order).  Every boot of the Job -- with or without LooGLE, PD-mux or
    baseline -- runs with context 96,000, so inside an arm the only thing that differs between the members is the long stream.
  * Arms: PD-mux non-sticky static D16 / D24 / D34 / D44 + the primary baseline (SG mixed 8192).  A D arm is the registered PD-mux bundle
    with ONE value changed (PDMUX_R2_FIXED_DSM); D64 is not an arm.  The arm name states D and the whole bundle (mo_spec.bundle_name).
  * Repetition: two full-block Jobs.  Block 1 = mw_lib.SEEDS, block 2 = mw_lib.SECOND_BLOCK_SEEDS (4 + 4 seeds, disjoint).  A Job is
    every arm x {with, without} x 3 rates x 4 seeds, anchor boot first and last, and one split-check boot PER D (each D is another stream
    group, so each has its own realized-split read-out; none enters a table).
  * The two Jobs are analysed separately (mo_intf_analyze.analyze_block) and combined only as paired differences with the block as a
    factor (mo_intf_analyze.combine_blocks); raw points of the two Jobs are never pooled (mo_analyze.merge_points still refuses).
  * mem / cap are mo_spec's (PENDING until the cap measurement fixes them); the three short rates have no default (pre-ladder).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(os.path.abspath(__file__)).parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import mo_spec as MS  # noqa: E402

SP, MW = MS.SP, MS.MW
L, G = MW.L, MW.G
KIND = "LOOGLE_INTERFERENCE"
D_GRID = (16, 24, 34, 44)                       # user decision 2026-11-11; D64 excluded
JUDGED_D = (34, 44)                             # #dshift: the D whose loss against the best D of a configuration is judged
BASELINE_KEY = MS.PRIMARY_KEY
ARM_KEYS = tuple(f"PDMUX_D{d}" for d in D_GRID) + (BASELINE_KEY,)
WITH, WITHOUT = "interference_with_long", "interference_without_long"
MEMBERS = (WITH, WITHOUT)
N_RATES = 3
DURATION_S = G.INTERFERENCE_DURATION            # 300 s per cell (trace_registry.json)
CONTEXT = MW.INTERFERENCE_CONTEXT               # 96,000 for every boot
BLOCK_SEEDS = {1: tuple(L.SEEDS), 2: tuple(L.SECOND_BLOCK_SEEDS)}
REHEARSAL_DURATION_S = 20.0
GOODPUT_LOSS = 0.03                             # #dshift: "more than 3 % below the best D" (gate 3: below 3 % is not a headline)


def _bundle_params():
    return next(p for p in MS.ARM_PARAMS if p["key"] == MS.BUNDLE_KEY)


def fixed_values(rehearsal=False):
    return dict(MS.REHEARSAL_FIXED if rehearsal else MS.default_fixed(), context=CONTEXT)


def d_arm(D, fixed, split_check=False):
    """The registered PD-mux bundle with decode SM count D: same argv, same env except PDMUX_R2_FIXED_DSM, name states D."""
    if D not in D_GRID:
        raise ValueError(f"D{D} is not an arm of the interference cell {D_GRID}")
    p = dict(_bundle_params(), D=D)
    if split_check:
        p.update(instr="split_check", role="split_check")
    a = MS.build_arm(p, fixed)
    env = dict(a["env"], PDMUX_R2_FIXED_DSM=str(D))
    if split_check:
        env.update(MS.MRZ.SPLIT_CHECK_ENV)
    a.update(env=env, D=D, arm_key=f"PDMUX_D{D}",
             inherited=f"PD-mux bundle with D{D}: non-sticky static, budget 65536, --chunked-prefill-size -1, overlap OFF, piecewise OFF")
    return a


def build_arm(key, fixed):
    if key == BASELINE_KEY:
        a = MS.build_arm(next(p for p in MS.ARM_PARAMS if p["key"] == key), fixed)
        a["arm_key"] = key
        return a
    return d_arm(int(key.replace("PDMUX_D", "")), fixed)


def _rng(*salt):
    return SP._rng("motivation_interference", *salt)


def boot_order(job):
    order = list(ARM_KEYS)
    _rng("order", job).shuffle(order)
    return order


def _run(job, arm_name, rate, seed, duration_s, member, cell, kind="cap"):
    n = MW.n_interference(seed, rate, duration_s, member == WITH)
    return MS.cell_run(job, arm_name, rate, seed, n, cell, kind=kind, workload=member, duration_s=duration_s, with_long=member == WITH)


def design(rates, duration_s):
    """What the two blocks must share (everything but the Job name and the seeds)."""
    f = fixed_values(True)
    return {"arms": [build_arm(k, f)["name"] for k in ARM_KEYS], "rates": list(rates), "duration_s": duration_s, "context": CONTEXT,
            "long_share": G.LONG_SHARE, "class_edges": list(MS.CLASS_EDGES), "slo": [list(MS.TTFT_GRID_MS), list(MS.ITL_GRID_MS)],
            "members": list(MEMBERS)}


def make_plan(job, rates, block, seeds=None, duration_s=None, rehearsal=False):
    """One full-block Job: anchor boot (baseline arm, the WITHOUT member of the lowest rate) -> every arm once in a seeded permutation, each
    with all rates x seeds x {with, without} in seeded cell order -> one split-check boot per D (WITH member, highest rate) -> anchor boot.
    Rehearsal: the real arm list, the first rate, one seed, a short duration."""
    rates = [float(r) for r in rates]
    if rehearsal:
        rates = rates[:1]
    seeds = tuple(seeds) if seeds is not None else (BLOCK_SEEDS[block][:1] if rehearsal else BLOCK_SEEDS[block])
    duration_s = float(duration_s if duration_s is not None else (REHEARSAL_DURATION_S if rehearsal else DURATION_S))
    fixed = fixed_values(rehearsal)
    arms = {k: build_arm(k, fixed) for k in ARM_KEYS}
    anchor = arms[BASELINE_KEY]

    def anchor_boot(tag):
        r = _run(job, anchor["name"], rates[0], seeds[0], duration_s, WITHOUT, f"ANCHOR_{tag}", kind="anchor")
        return SP.boot(f"{job}_anchor_{tag}", anchor, [r], role="anchor", anchor=tag)

    boots = [anchor_boot("start")]
    order = boot_order(job)
    for k in order:
        a = arms[k]
        cells = [_run(job, a["name"], r, s, duration_s, m, "GRID_with" if m == WITH else "GRID_without")
                 for r in rates for s in seeds for m in MEMBERS]
        _rng("cells", job, k).shuffle(cells)
        boots.append(SP.boot(f"{job}_{a['name']}", a, cells, role="meas"))
    for d in D_GRID:
        sc = d_arm(d, fixed, split_check=True)
        r = _run(job, sc["name"], rates[-1], seeds[0], duration_s, WITH, "SPLIT_CHECK", kind="split_check")
        boots.append(SP.boot(f"{job}_splitcheck_D{d}", sc, [r], role="split_check"))
    boots.append(anchor_boot("end"))
    return {"job": job, "kind": KIND, "block": block, "workload": WITH, "grid": {"role": "base", "rates": rates}, "seeds": list(seeds),
            "duration_s": duration_s, "rehearsal": bool(rehearsal), "fixed": fixed, "boot_order": [b["arm"] for b in boots],
            "order_keys": order, "arm_keys": list(ARM_KEYS), "design_sha": MS.sha(design(rates, duration_s)),
            "anchor": {"arm": anchor["name"], "rate": rates[0], "seed": seeds[0], "n": boots[0]["runs"][0]["n_requests"],
                       "positions": ["first", "last"]}, "boots": boots, "pending": []}


def plan_problems(plan, allow_placeholders=False):
    out = []
    boots, job, fixed = plan["boots"], plan["job"], plan["fixed"]
    ids = [r["run_id"] for b in boots for r in b["runs"]]
    if len(ids) != len(set(ids)):
        out.append("duplicate run_id")
    if len({b["boot_id"] for b in boots}) != len(boots):
        out.append("duplicate boot_id")
    if plan["boot_order"] != [b["arm"] for b in boots]:
        out.append("plan.boot_order differs from the boots")
    if boots[0]["role"] != "anchor" or boots[-1]["role"] != "anchor":
        out.append("anchor boot must be first and last")
    ref = {k: build_arm(k, fixed) for k in ARM_KEYS}
    meas = [b for b in boots if b["role"] == "meas"]
    if [b["arm"] for b in meas] != [ref[k]["name"] for k in boot_order(job)]:
        out.append(f"boot order differs from the registered permutation of job '{job}' (every arm exactly once; no D outside {D_GRID})")
    sc = [b for b in boots if b["role"] == "split_check"]
    if [b["arm_def"].get("D") for b in sc] != list(D_GRID):
        out.append("one split-check boot per D, after the measurement boots")
    rates, seeds = plan["grid"]["rates"], plan["seeds"]
    want = sorted((r, s, m) for r in rates for s in seeds for m in MEMBERS)
    for b in meas:
        if sorted((r["rate"], r["seed"], r["workload"]) for r in b["runs"]) != want:
            out.append(f"{b['boot_id']}: cells != rates x seeds x (with, without) (full-block rule; both members in the same boot)")
        by = {(r["rate"], r["seed"], r["workload"]): r for r in b["runs"]}
        for r_ in rates:
            for s in seeds:
                w, wo = by.get((r_, s, WITH)), by.get((r_, s, WITHOUT))
                if w and wo and not (w["n_requests"] > wo["n_requests"] and w["duration_s"] == wo["duration_s"] == plan["duration_s"]):
                    out.append(f"{b['boot_id']}: pair ({r_}, {s}) is not short stream + long stream over the same duration")
    for b in boots:
        for r in b["runs"]:
            if r["workload"] not in MEMBERS or r["n_requests"] != MW.n_interference(r["seed"], r["rate"], r["duration_s"], r["workload"] == WITH):
                out.append(f"{r['run_id']}: not a registered interference cell")
                break
    if not plan["rehearsal"]:
        if len(rates) != N_RATES or sorted(set(rates)) != sorted(rates):
            out.append(f"{len(rates)} rates != {N_RATES} distinct short rates")
        if tuple(seeds) != BLOCK_SEEDS.get(plan["block"]):
            out.append(f"seeds {seeds} are not the registered seeds of block {plan['block']}")
        if len(seeds) < MS.MIN_SEEDS:
            out.append(f"n seeds {len(seeds)} < {MS.MIN_SEEDS} (gate 3)")
        if plan["duration_s"] != DURATION_S:
            out.append(f"duration {plan['duration_s']} != registered {DURATION_S}")
    if plan["design_sha"] != MS.sha(design(rates, plan["duration_s"])):
        out.append("plan.design_sha is not the design of this plan")
    pm_env = SP.arm("PM-ref")["env"]
    bundle = MS.build_arm(_bundle_params(), fixed)
    for b in boots:
        a = b["arm_def"]
        if a["name"] != MS.bundle_name(a["params"]) or b["arm"] != a["name"]:
            out.append(f"{b['boot_id']}: arm name does not state its bundle")
        if "--context-length" not in a["cmd"] or a["cmd"][a["cmd"].index("--context-length") + 1] != str(CONTEXT):
            out.append(f"{b['boot_id']}: context is not {CONTEXT} (both members of a pair boot with the same context)")
        if a["engine"] == "sglang" and a["python"] != MS.PY312:
            out.append(f"{b['boot_id']}: SGLang arm must use {MS.PY312}")
        if a["tree"] == "pristine" and (a["env"] or any(x in ("--enable-pdmux", "--pdmux-config-path") for x in a["cmd"])):
            out.append(f"{b['boot_id']}: pristine boot with PD-mux env/argv")
        if a.get("D") is not None:
            want_env = dict(pm_env, PDMUX_R2_FIXED_DSM=str(a["D"]))
            if b["role"] == "split_check":
                want_env.update(MS.MRZ.SPLIT_CHECK_ENV)
            if a["cmd"] != bundle["cmd"] or a["env"] != want_env or a["tree"] != "patched" or a["params"]["D"] != a["D"] or a["D"] not in D_GRID:
                out.append(f"{b['boot_id']}: a D arm is the registered bundle with only PDMUX_R2_FIXED_DSM changed"
                           + (" (+ the split-check env)" if b["role"] == "split_check" else ""))
        if not allow_placeholders and not plan["rehearsal"] and (any(x in (MS.MEM_TOKEN, MS.CAP_TOKEN) for x in a["cmd"])
                                                                   or any(MS.is_pending(fixed[k]) for k in ("mem", "cap", "sg_venv_freeze"))):
            out.append(f"{b['boot_id']}: PLACEHOLDER mem / cap / venv freeze unresolved -> not launchable")
    return out


# ---- Job time estimate -------------------------------------------------------------------------------------------------------------------
BOOT_MIN, JOB_START_MIN = SP.BOOT_OH_MIN, 20.0          # roadmap GPU-time table


def estimate(plan):
    runs = [r for b in plan["boots"] for r in b["runs"]]
    load_min = sum(r["duration_s"] for r in runs) / 60.0
    drain_min = SP.DRAIN_MIN * len(runs)
    boot_min = BOOT_MIN * len(plan["boots"])
    tot = load_min + drain_min + boot_min + JOB_START_MIN
    return {"boots": len(plan["boots"]), "runs": len(runs), "load_h": round(load_min / 60.0, 2), "drain_min": round(drain_min, 1),
            "boot_min": boot_min, "start_min": JOB_START_MIN, "total_h": round(tot / 60.0, 2),
            "basis": "load = sum of cell durations, drain 0.6 min per run (sb_spec.DRAIN_MIN; the LooGLE tail is not measured), boot 6 min, "
                     "start 20 min"}


def main(argv=None):
    import argparse
    import json
    ap = argparse.ArgumentParser(description="plan summary of the LooGLE interference cell (CPU)")
    ap.add_argument("--job", default="J")
    ap.add_argument("--block", type=int, default=1, choices=sorted(BLOCK_SEEDS))
    ap.add_argument("--rates", required=True, help="three short rates, comma separated (from the pre-ladder; no default)")
    ap.add_argument("--rehearsal", action="store_true")
    a = ap.parse_args(argv)
    p = make_plan(a.job, [float(x) for x in a.rates.split(",")], a.block, rehearsal=a.rehearsal)
    print(json.dumps({"plan_sha": MS.sha(p), "design_sha": p["design_sha"], "problems": plan_problems(p), "boot_order": p["boot_order"],
                      "seeds": p["seeds"], "estimate": estimate(p)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
