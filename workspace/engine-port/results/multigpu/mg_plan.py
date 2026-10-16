#!/usr/bin/env python3
"""Cell / boot plan, M2g plan, budget dry-run and M2G_PASS_TABLE handling (PREREG_MULTIGPU rev10 §3, §4-1, §4-2, §6, §15, §16, §17;
rev12 §19-2 shape X + U14 X anchor cells, §19-4 floor blocks (20, both arms), U7 X correctness step, §19-10 budget).

Pure planning: no GPU, no network.  The Job drivers (mg_drive.py) turn the specs into mg_boot.py runs.

  python3 mg_plan.py m3-plan --job J2        print the Job's static M3 boot/cell skeleton (counts)
  python3 mg_plan.py m3-budget --job J2 [--ratio R]   per-stage time model -> Job total vs the Job hard cap + reservation utilization
  python3 mg_plan.py m2g-plan --job J2 | m2g-budget --job J2
  python3 mg_plan.py m3-forecast --job J2    F15: scenario x calibration table (Job cap cut, configs that keep a complete pair)
  python3 mg_plan.py job-table               all four Jobs, M2g and M3: est. hours, cap, reserved / used GPU-h, utilization, cost
  python3 mg_plan.py config-order --job J8   the logged random TP/DP order of the Job (seed 20261012)
  python3 mg_plan.py check-pass-table PATH SHA256 [COMMITTED_SHA256_FILE]     ABORT_M2G_NOT_PASSED logic (F17: 3-way digest)
  python3 mg_plan.py --selftest

TIME MODEL (an ESTIMATE, not a registered rule; rev5 audit "boot 4 min, n/lambda, drain" method):
  cell   = max(n/x + drain, n/lambda)  (+ gap)           lambda = ratio * prior(N, shape)   [lambda0_plan.cell_duration_at]
  boot   = EST_BOOT_S + warm-up per shape + sum(cells) + floor blocks (EST_FLOOR_S) + cmd steps (xcheck EST_XCHECK_S, other 20 s)
`ratio` is the true-capacity / prior ratio (1.0 = the audit's "lambda = prior" case, 0.7 = a TP case with lambda < prior).
"""

from __future__ import annotations

import os

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

HERE = Path(os.path.abspath(__file__)).parent
sys.path.insert(0, str(HERE))
import mg_const as C  # noqa: E402

# ----------------------------------------------------------------------------- configs
def config_spec(cfg):
    """(mode, N, tp, dp, gpu_ids)."""
    if cfg == "N1":
        return dict(config="N1", mode="N1", N=1, tp=1, dp=1, gpu_ids=C.GPU_SUBSETS["N1"])
    mode, n = cfg[:2], int(cfg[2:])
    return dict(config=cfg, mode=mode, N=n, tp=n if mode == "TP" else 1, dp=n if mode == "DP" else 1,
                gpu_ids=C.GPU_SUBSETS[cfg])


def round10(n):
    return int(round(n / 10.0) * 10)


def n_req_for(x):
    """§4-2: clamp(round10(x * 170 s), 40, 4000)."""
    return max(C.N_REQ_CLAMP[0], min(C.N_REQ_CLAMP[1], round10(x * C.WINDOW_S)))


def prior(N, shape):
    return N * C.PRIOR_PER_GPU[shape]


def hi_rung(N, shape, ext=0):
    x = C.HI_MULT * prior(N, shape) * (C.EXT_MULT ** ext)
    return {"offered": round(x, 6), "n_req": n_req_for(x)}


def lo_rung(fused_hi_median_achieved):
    x = C.LO_MULT * fused_hi_median_achieved
    return {"offered": round(x, 6), "n_req": n_req_for(x)}


def arm_order(cfg):
    """Random permutation of the 4 arms per config, seed 20261012 (§4-1).  Deterministic."""
    rng = random.Random(f"{C.ARM_ORDER_SEED}:{cfg}")
    a = list(C.ARMS)
    rng.shuffle(a)
    return a


def seed_order(cfg, arm, tag, seeds):
    rng = random.Random(f"{C.ARM_ORDER_SEED}:{cfg}:{arm}:{tag}")
    s = list(seeds)
    rng.shuffle(s)
    return s


def shape_order(cfg, arm, tag):
    rng = random.Random(f"{C.ARM_ORDER_SEED}:{cfg}:{arm}:{tag}:shape")
    s = list(C.SHAPES)                                   # rev12: three shapes (A, B, X); the order changes vs rev11 (forecast only)
    rng.shuffle(s)
    return s


def floor_step():
    """rev12 §19-4: the floor block -- uncontended single requests, lengths x 16, sequential, max_new 8, the floor document's prefix."""
    return {"type": "floor", "lengths": list(C.FLOOR_LENGTHS), "repeats": C.FLOOR_REPEATS, "max_new": C.FLOOR_MAX_NEW}


def xcheck_step(arm):
    """U7: shape X small correctness step (scripts/multigpu/mg_xcheck.py serve); {XTOKENS} = the Job's registered token file."""
    return {"type": "cmd", "name": f"xcheck_{arm}",
            "argv": ["python3", "workspace/engine-port/scripts/multigpu/mg_xcheck.py", "serve", "--endpoint", "{ENDPOINT}",
                     "--tokens", "{XTOKENS}", "--out", "{BOOT}/xcheck.json"]}


def opening_steps(cfg, arm, kind):
    """rev12: what opens a boot BEFORE its cells.  Floor block: both FLOOR_ARMS of N1 boot1 (J1), of every anchor boot and of every
    N>1 config boot1 (R-6: symmetric, 20 blocks).  X correctness (U7, corrected 2026-10-16): the fused_sym / agnostic boot1 of every
    config -- J1: N1 boot1; J_N: the first (= only) boot1 of each mode, TP_N and DP_N, i.e. ON THE N-GPU SERVER (the TP split-prefill
    path of a 16k prompt is the path M2g never tested; the N=1 anchor boots do not exercise it) -- after the floor block."""
    st = []
    if arm in C.FLOOR_ARMS and kind in ("boot1", "anchor"):
        st.append(floor_step())
        if kind == "boot1":
            st.append(xcheck_step(arm))
    return st


def _boot(cfg, arm, kind, steps, block="main", suffix="", **extra):
    cs = config_spec(cfg)
    spec = dict(cs, arm=arm, kind=kind, block=block, boot_id=f"m3_{cfg}_{arm}_{kind}{suffix}", steps=steps)
    spec.update(extra)
    return spec


def _cell(shape, rung, seed, N, ext=0, rnd=1, block="main", offered=None, n_req=None, **kw):
    r = hi_rung(N, shape, ext) if rung == "HI" else {"offered": offered, "n_req": n_req}
    if offered is not None and rung == "HI":
        r = {"offered": offered, "n_req": n_req}
    d = dict(type="cell", shape=shape, rung=rung, seed=seed, ext=ext, round=rnd, block=block, **r)
    d.update(kw)
    return d


def hi_steps(cfg, arm, shapes, ext=0, rnd=1, block="main"):
    N = config_spec(cfg)["N"]
    steps = []
    for sh in shape_order(cfg, arm, f"hi{ext}{rnd}"):
        if sh not in shapes:
            continue
        steps.append({"type": "warmup", "shape": sh})
        for sd in seed_order(cfg, arm, f"hi{ext}{rnd}{sh}", C.HI_SEEDS):
            steps.append(_cell(sh, "HI", sd, N, ext, rnd, block))
    return steps


def lo_steps(cfg, arm, x_lo_by_shape, ext=0, rnd=1):
    """x_lo_by_shape: shape -> {offered, n_req} (SAME for the four arms of a (config, shape), MGP-9)."""
    steps = []
    ext_of = (lambda sh: ext.get(sh, 0)) if isinstance(ext, dict) else (lambda sh: ext)    # extension is per (config, shape)
    for sh in shape_order(cfg, arm, "lo" + "".join(str(ext_of(s)) for s in C.SHAPES) + str(rnd)):
        if sh not in x_lo_by_shape:
            continue
        steps.append({"type": "warmup", "shape": sh})
        for sd in seed_order(cfg, arm, f"lo{ext_of(sh)}{rnd}{sh}", C.LO_SEEDS):
            steps.append(_cell(sh, "LO", sd, config_spec(cfg)["N"], ext_of(sh), rnd, "main", **x_lo_by_shape[sh]))
    return steps


def config_order(job):
    """§17-2 (a): the order of a Job's TP/DP configs is the REGISTERED derivation mg_const.config_order_types (result J2=J4=J8=[DP, TP]).
    The Job driver logs it and aborts (ABORT_CONFIG_ORDER) if it differs from mg_const.CONFIG_ORDER_EXPECTED.  J1 has the single config N1."""
    n = C.JOB_N[job]
    if job == "J1":
        return ["N1"]
    return [f"{t}{n}" for t in C.config_order_types(n)]


def assert_config_order_registered(_initial=None):
    """H-2: dry-run / selftest guard -- the derived order must equal the order written in the PREREG."""
    for n, want in C.CONFIG_ORDER_EXPECTED.items():
        got = C.config_order_types(n, _initial)
        assert got == want, f"config order J{n}: derived {got} != registered {want}"
    return True


def anchor_arm_order(job):
    a = list(C.ANCHOR_ARMS)
    random.Random(f"{C.ARM_ORDER_SEED}:{job}:anchor").shuffle(a)
    return a


def has_anchor(job):
    """J1 is the reference itself; J2/J4/J8 measure an N=1 anchor on GPU 0 (§15-3)."""
    return job != "J1"


def boots_gpu_dev():
    """§4-6 / §15-3: N=1 fused shape A HI seed 4386 on GPU 0..7, the 8 servers CONCURRENTLY (J8 only, FIRST).  Descriptive
    under simultaneous 8-server load; no outcome branch (MGP-47)."""
    out = []
    for g in range(8):
        spec = dict(config_spec("N1"), gpu_ids=str(g))
        N = 1
        st = [{"type": "warmup", "shape": "A"}, _cell("A", "HI", C.GPU_DEV_SEED, N, 0, 1, "gpu_dev", gpu=g)]
        out.append(dict(spec, arm="fused", kind="gpudev", block="gpu_dev", boot_id=f"m3_gpudev_gpu{g}", steps=st,
                        concurrent_group="gpu_dev", port=20000 + 700 + g))
    return out


def boots_boot1(cfg, arms=None, exclude=()):
    return [_boot(cfg, a, "boot1", opening_steps(cfg, a, "boot1") + hi_steps(cfg, a, C.SHAPES)) for a in (arms or arm_order(cfg))
            if f"{cfg}:{a}" not in exclude]


def boots_ext(cfg, shapes, arms=None, exclude=()):
    return [_boot(cfg, a, "boot1ext", hi_steps(cfg, a, shapes, ext=1), suffix="") for a in (arms or arm_order(cfg))
            if f"{cfg}:{a}" not in exclude]


def boots_boot2(cfg, x_lo_by_shape, ext=0, arms=None, exclude=()):
    return [_boot(cfg, a, "boot2", lo_steps(cfg, a, x_lo_by_shape, ext)) for a in (arms or arm_order(cfg))
            if f"{cfg}:{a}" not in exclude]


def boots_anchor(job, exclude=()):
    """§15-3 / §16-3: the J_N N=1 anchor = GPU 0 alone, arms {fused_sym, agnostic}, shapes A and X (rev12 U14), HI rung = THE SAME
    1.6 x prior(1) rung as J1 boot1 (ext 0 -- an anchor is never extended), 4 seeds.  Two boots, run before the configs of the Job.
    rev12: each anchor boot opens with the floor block only -- the U7 X correctness step is NOT on the anchor boots (corrected
    2026-10-16, 0b541c74): it runs on the fused_sym / agnostic boot1 of every config (opening_steps).  J1 has no anchor."""
    if not has_anchor(job):
        return []
    return [_boot("N1", a, "anchor", opening_steps("N1", a, "anchor") + hi_steps("N1", a, C.ANCHOR_SHAPES, ext=0, rnd=1, block="anchor"),
                  block="anchor")
            for a in anchor_arm_order(job) if f"N1:{a}" not in exclude]


# ----------------------------------------------------------------------------- M2G pass table
def table_key(cfg, arm):
    return f"{cfg}:{arm}"


def load_pass_table(path, expected_sha256, committed_sha_path=None):
    """Returns (table, abort_code).  F17: the table is accepted only if THREE digests agree --
       (1) PDMUX_M2G_PASS_SHA256 (env / CLI `expected_sha256`), (2) the COMMITTED results/multigpu/M2G_PASS_TABLE.json.sha256
       (`committed_sha_path`; a missing file is a failure when the argument is given), (3) sha256 of the file bytes.
    Missing file / any mismatch / unparsable -> ABORT_M2G_NOT_PASSED."""
    p = Path(path)
    if not p.exists():
        return None, "ABORT_M2G_NOT_PASSED"
    raw = p.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if not expected_sha256 or actual != expected_sha256:
        return None, "ABORT_M2G_NOT_PASSED"
    if committed_sha_path:
        cp = Path(committed_sha_path)
        if not cp.exists() or cp.read_text().strip().split()[:1] != [actual]:
            return None, "ABORT_M2G_NOT_PASSED"
    try:
        t = json.loads(raw)
        assert isinstance(t.get("cells"), dict)
    except (ValueError, AssertionError):
        return None, "ABORT_M2G_NOT_PASSED"
    # R5: the G5(a) CPU unit test (selector equivalence) must be recorded PASS in the table -- NOT_TESTED/FAIL/TREE_MISSING aborts M3.
    if t.get("g5a_unit_test") != "PASS":
        return None, "ABORT_M2G_NOT_PASSED"
    return t, None


def excluded_set(table):
    """'cfg:arm' keys excluded from M3 (included == false)."""
    return {k for k, v in table["cells"].items() if not v.get("included", True)}


def runnable_configs(table, job):
    """Configs of `job` (in the Job's random order) whose fused arm is included in the Job's own table (N1 for J1)."""
    ex = excluded_set(table)
    return [c for c in config_order(job) if table_key(c, "fused") not in ex]


def anchor_exclude(table):
    """'N1:<arm>' keys of the anchor arms the Job's table excludes (the anchor boot is then not run -> JOB_EFFECT_UNTESTED)."""
    ex = excluded_set(table) if table else set()
    return {k for k in ex if k.startswith("N1:")}


def tp_allowed_by_j2(j2_table):
    """§15-2 / §16-3: the G0(ii) guard-refusal result is J2's.  Returns (allowed, reason).  A missing / non-PASS J2 table -> the Job's TP
    configs are NOT_TESTED (DP proceeds); never an ABORT of the whole Job."""
    if not j2_table:
        return False, "J2 PASS TABLE ABSENT (G0(ii) unknown) -> TP NOT_TESTED"
    if j2_table.get("g0ii") != "PASS":
        return False, f"J2 G0(ii)={j2_table.get('g0ii')} -> TP axis excluded (§3-1)"
    return True, None


# ----------------------------------------------------------------------------- budget model
CALIBS = {                     # F15: (boot_extra_s per boot, cell_gap_s per cell)
    "audited": dict(boot_extra_s=0.0, cell_gap_s=C.EST_CELL_GAP_S),                     # the model behind the audited 29.96 h
    "vessl_boot": dict(boot_extra_s=C.EST_VESSL_BOOT_EXTRA_S, cell_gap_s=C.EST_CELL_GAP_S),   # N3 r3 +49 s attributed per boot
    "vessl_boot_cell": dict(boot_extra_s=C.EST_VESSL_BOOT_EXTRA_S, cell_gap_s=C.EST_VESSL_CELL_GAP_S),   # + 24 s per cell (upper)
}


def cell_seconds(shape, x, n, lam, gap=None):
    span = n / x
    drain = C.EST_DRAIN_S[shape]
    gap = C.EST_CELL_GAP_S if gap is None else gap
    return max(span + drain, n / lam if lam > 0 else span + drain) + gap


def boot_seconds(spec, lam_of, x_lo_of=None, calib=None):
    """lam_of(N, shape) -> true capacity assumed for the arm.  calib = CALIBS entry (default: the audited model)."""
    calib = calib or CALIBS["audited"]
    t = C.EST_BOOT_S + calib["boot_extra_s"]
    N = spec["N"]
    for st in spec["steps"]:
        if st["type"] == "warmup":
            t += C.EST_WARMUP_S[st["shape"]]
        elif st["type"] == "cell":
            t += cell_seconds(st["shape"], st["offered"], st["n_req"], lam_of(N, st["shape"]), calib["cell_gap_s"])
        elif st["type"] == "floor":
            t += C.EST_FLOOR_S
        elif st["type"] == "cmd":
            t += C.EST_XCHECK_S if str(st.get("name", "")).startswith("xcheck") else 20.0
    return t


def _arm_ratio(ratio_of, ratio_arm_of, cfg, arm):
    return ratio_arm_of(cfg, arm) if ratio_arm_of else ratio_of(cfg)


def ext_fires(ratio_of, ratio_arm_of, cfg, assume_ext=None):
    """Joint extension (§4-2): ANY arm above EXT_TRIGGER_RATIO x prior (audit §3) -- the four arms extend together."""
    if assume_ext is not None:
        return 1 if assume_ext else 0
    return 1 if any(_arm_ratio(ratio_of, ratio_arm_of, cfg, a) > C.EXT_TRIGGER_RATIO for a in C.ARMS) else 0


def boot_gpus_used(spec):
    """GPUs a boot occupies: the config's N for a main boot, the single GPU of an anchor / gpu_dev boot."""
    return len(str(spec["gpu_ids"]).split(","))


def m3_static_boots(job, ratio_of=None, assume_ext=None, lo_given=True, ratio_arm_of=None):
    """The full M3 boot list of ONE Job for a given assumed capacity ratio (ratio_of(cfg) -> lambda/prior, or ratio_arm_of(cfg, arm)).
    Order (§16-3): [J8: GPU-deviation block, concurrent] -> N=1 anchor boots (J2/J4/J8) -> the Job's configs in config_order(job).
    x_lo uses the assumed FUSED HI achieved = ratio * prior.  Extension is assumed iff any arm's ratio > 1.42 (audit)."""
    ratio_of = ratio_of or (lambda cfg: 1.0)
    stages = []
    if job == "J8":
        stages.append(("gpu_dev", boots_gpu_dev()))
    if has_anchor(job):
        stages.append(("anchor", boots_anchor(job)))
    for cfg in config_order(job):
        N = config_spec(cfg)["N"]
        ext = ext_fires(ratio_of, ratio_arm_of, cfg, assume_ext)
        r = _arm_ratio(ratio_of, ratio_arm_of, cfg, "fused")
        lam = {sh: r * prior(N, sh) for sh in C.SHAPES}
        stages.append((f"{cfg}.boot1", boots_boot1(cfg)))
        if ext:
            stages.append((f"{cfg}.boot1ext", boots_ext(cfg, C.SHAPES)))
        xlo = {sh: lo_rung(lam[sh]) for sh in C.SHAPES}
        stages.append((f"{cfg}.boot2", boots_boot2(cfg, xlo, ext)))
    return stages


def _boot_true_seconds(b, ratio_of, ratio_arm_of, calib):
    cfg_for_ratio = b["config"]
    lam_of = lambda N, sh, b=b: _arm_ratio(ratio_of, ratio_arm_of, cfg_for_ratio, b["arm"]) * prior(N, sh)
    return boot_seconds(b, lam_of, calib=calib)


def stage_seconds(boots, ratio_of, ratio_arm_of, calib):
    """(wall_s, used_gpu_s).  A concurrent_group (gpu_dev) takes the MAX of its boots' time (8 servers at once); everything else is sequential."""
    if not boots:
        return 0.0, 0.0
    ts = [_boot_true_seconds(b, ratio_of, ratio_arm_of, calib) for b in boots]
    used = sum(t * boot_gpus_used(b) for t, b in zip(ts, boots))
    if all(b.get("concurrent_group") for b in boots):
        return max(ts), used
    return sum(ts), used


def m3_budget(job, ratio_of=None, assume_ext=None, ratio_arm_of=None, calib=None):
    ratio_of = ratio_of or (lambda cfg: 1.0)
    rows, total, used_total, n_boots, n_cells = [], 0.0, 0.0, 0, 0
    for name, boots in m3_static_boots(job, ratio_of, assume_ext, ratio_arm_of=ratio_arm_of):
        wall, used = stage_seconds(boots, ratio_of, ratio_arm_of, calib)
        for b in boots:
            n_cells += sum(1 for s in b["steps"] if s["type"] == "cell")
        n_boots += len(boots)
        rows.append((name, len(boots), wall, used))
        total += wall
        used_total += used
    return _budget_doc(job, "m3", rows, total, used_total, n_boots, n_cells)


def _budget_doc(job, kind, rows, total, used_total, n_boots, n_cells):
    """Reservation view.  Job wall = preflight (EST_PREFLIGHT_S, no GPU use) + stages.  reserved GPU-s = spec GPUs x Job wall."""
    job_s = total + C.EST_PREFLIGHT_S
    reserved = C.JOB_EXPECTED_GPUS[job] * job_s
    cap = C.hard_cap(kind, job)
    return {"job": job, "kind": kind, "stages": rows, "total_s": total, "total_h": total / 3600.0, "job_s": job_s,
            "job_h": job_s / 3600.0, "boots": n_boots, "cells": n_cells, "cap_h": cap / 3600.0, "over_cap": job_s > cap,
            "used_gpu_h": used_total / 3600.0, "reserved_gpu_h": reserved / 3600.0,
            "utilization": (used_total / reserved if reserved else 0.0),
            "spec": C.JOB_SPEC[job], "rate_usd_h": C.JOB_RATE_USD_H[job],
            "cost_usd": C.JOB_RATE_USD_H[job] * job_s / 3600.0}


def boot_need_s(spec, ratio=0.5):
    """Admission need (worst corner, like N3 §8-3): ratio of prior capacity.  F15: includes the VESSL-measured per-boot
    overhead (+49 s) -- N3 r3 admitted on a need that was 40 % too small."""
    return boot_seconds(spec, lambda N, sh: ratio * prior(N, sh), calib=CALIBS["vessl_boot"])


# ----------------------------------------------------------------------------- F15 forecast (per-Job cap cut)
SCENARIOS = {                  # name -> ratio_arm_of(cfg, arm)  (lambda_true / prior)
    "S1 lambda=prior (all)": lambda cfg, arm: 1.0,
    "S2 TP lambda=0.7 prior": lambda cfg, arm: 0.7 if cfg.startswith("TP") else 1.0,
    "S3 extension: agnostic 1.48": lambda cfg, arm: 1.48 if arm == "agnostic" else 1.0,
}


def m3_timeline(job, ratio_arm_of, calib, cap_s=None, preflight_s=0.0):
    """Sequential simulation of ONE Job's driver (§16-3 order, admission against the remaining time, F10 rules).
    A boot is ADMITTED iff boot_need_s(spec, 1.0) <= remaining (mg_drive.run_boot); it then takes its scenario duration and
    is PARTIAL if that crosses the cap (the runner kills the running cell).  Returns the boot records."""
    cap_s = C.hard_cap("m3", job) if cap_s is None else cap_s
    ratio_of = lambda cfg: 1.0
    t, recs = float(preflight_s), []

    def run(spec, stage):
        nonlocal t
        need = boot_need_s(spec, 1.0)
        rec = {"stage": stage, "config": spec["config"], "arm": spec["arm"], "block": spec.get("block", "main"),
               "start_s": t, "need_s": need}
        if cap_s - t < need:
            rec.update(status="NOT_ATTEMPTED", end_s=t)
        else:
            d = _boot_true_seconds(spec, ratio_of, ratio_arm_of, calib)
            if t + d > cap_s:
                rec.update(status="PARTIAL", end_s=cap_s)
                t = cap_s
            else:
                rec.update(status="DONE", end_s=t + d)
                t += d
        recs.append(rec)
        return rec["status"] == "DONE"

    def run_group(specs, stage):
        """Concurrent boots: admitted iff the largest need fits; they take the max duration."""
        nonlocal t
        need = max(boot_need_s(s, 1.0) for s in specs)
        if cap_s - t < need:
            for s in specs:
                recs.append({"stage": stage, "config": s["config"], "arm": s["arm"], "block": s.get("block", "main"),
                             "start_s": t, "need_s": need, "status": "NOT_ATTEMPTED", "end_s": t})
            return False
        d = max(_boot_true_seconds(s, ratio_of, ratio_arm_of, calib) for s in specs)
        end = min(cap_s, t + d)
        for s in specs:
            recs.append({"stage": stage, "config": s["config"], "arm": s["arm"], "block": s.get("block", "main"),
                         "start_s": t, "need_s": need, "status": "DONE" if t + d <= cap_s else "PARTIAL", "end_s": end})
        t = end
        return t + 0 <= cap_s

    if job == "J8":
        run_group(boots_gpu_dev(), "gpu_dev")
    for spec in boots_anchor(job):
        run(spec, "anchor")
    for cfg in config_order(job):
        N = config_spec(cfg)["N"]
        b1 = [run(s, f"{cfg}.boot1") for s in boots_boot1(cfg)]
        ext = ext_fires(ratio_of, ratio_arm_of, cfg)
        ext_ok = True
        if ext:
            ext_ok = all([run(s, f"{cfg}.boot1ext") for s in boots_ext(cfg, C.SHAPES)])
        fused_i = arm_order(cfg).index("fused")
        fused_ok = b1[fused_i] and (ext_ok if ext else True)
        if not ext_ok:                                           # F10: an incomplete extension forbids entering boot2 (LO)
            for s in boots_boot2(cfg, {sh: lo_rung(prior(N, sh)) for sh in C.SHAPES}, ext):
                recs.append({"stage": f"{cfg}.boot2", "config": cfg, "arm": s["arm"], "block": "main", "start_s": t,
                             "end_s": t, "need_s": 0.0, "status": "NOT_ATTEMPTED_EXT_INCOMPLETE"})
        elif fused_ok:
            lam = {sh: _arm_ratio(ratio_of, ratio_arm_of, cfg, "fused") * prior(N, sh) for sh in C.SHAPES}
            for s in boots_boot2(cfg, {sh: lo_rung(lam[sh]) for sh in C.SHAPES}, ext):
                run(s, f"{cfg}.boot2")
        else:                                                    # fused HI undefined -> x_lo undefined -> LO skipped (MGH-11)
            for s in boots_boot2(cfg, {sh: lo_rung(prior(N, sh)) for sh in C.SHAPES}, ext):
                recs.append({"stage": f"{cfg}.boot2", "config": cfg, "arm": s["arm"], "block": "main", "start_s": t,
                             "end_s": t, "need_s": 0.0, "status": "NOT_ATTEMPTED_FUSED_HI_UNDEFINED"})
    return recs


def forecast_summary(job, recs):
    """Which stages of THIS Job are cut and which of its configs keep a complete G_sym pair (HI + LO for fused_sym and agnostic)."""
    done = lambda cfg, arm, stage_suffix: any(r["config"] == cfg and r["arm"] == arm and r["stage"].endswith(stage_suffix)
                                              and r["status"] == "DONE" for r in recs)
    ext_of = lambda cfg: any(r["stage"] == f"{cfg}.boot1ext" for r in recs)
    defined = []
    for cfg in config_order(job):
        ok = all(done(cfg, a, ".boot1") and (not ext_of(cfg) or done(cfg, a, ".boot1ext")) and done(cfg, a, ".boot2")
                 for a in ("agnostic", "fused_sym"))
        if ok:
            defined.append(cfg)
    cut = {}
    for r in recs:
        if r["status"] != "DONE":
            cut.setdefault(r["stage"], []).append(f"{r['arm']}:{r['status']}")
    anchor_done = [r for r in recs if r["stage"] == "anchor"]
    return {"job": job, "end_h": max((r["end_s"] for r in recs), default=0.0) / 3600.0,
            "n_done": sum(1 for r in recs if r["status"] == "DONE"), "n_boots": len(recs),
            "cut": cut, "configs_defined": defined, "configs_total": list(config_order(job)),
            "anchor_boots_done": sum(1 for r in anchor_done if r["status"] == "DONE"), "anchor_boots": len(anchor_done)}


def m3_forecast_table(job, cap_h=None, preflight_s=None):
    """Scenario x calibration -> total (model, uncapped), cut stages at the Job cap, configs that keep a complete pair."""
    rows = []
    cap_s = (cap_h * 3600.0) if cap_h else None
    pf = C.EST_PREFLIGHT_S if preflight_s is None else preflight_s
    for sc, fn in SCENARIOS.items():
        for cname, calib in CALIBS.items():
            b = m3_budget(job, ratio_arm_of=fn, calib=calib)
            recs = m3_timeline(job, fn, calib, cap_s, pf)
            rows.append({"scenario": sc, "calib": cname, "total_h_uncapped": b["job_h"], "util": b["utilization"],
                         **forecast_summary(job, recs)})
    return rows


def fmt_forecast(job, rows):
    L = [f"== F15 forecast, Job {job} (model estimates, NOT measurements; MGP-49) ==",
         f"cap {C.hard_cap('m3', job) / 3600:.0f} h; calibrations: audited = model of the audited run; vessl_boot = +{C.EST_VESSL_BOOT_EXTRA_S:.0f} s/boot "
         f"(N3 r3: 170 s vs 121 s); vessl_boot_cell = also +{C.EST_VESSL_CELL_GAP_S - C.EST_CELL_GAP_S:.0f} s/cell (upper)",
         f"{'scenario':30s} {'calib':16s} {'job_h':>7s} {'end_h':>6s} {'boots done':>10s} {'util':>5s} {'anchor':>6s}  configs complete / cut"]
    for r in rows:
        cut = "; ".join(f"{k}[{len(v)}]" for k, v in r["cut"].items()) or "-"
        L.append(f"{r['scenario']:30s} {r['calib']:16s} {r['total_h_uncapped']:7.2f} {r['end_h']:6.2f} "
                 f"{r['n_done']:4d}/{r['n_boots']:<5d} {100 * r['util']:4.0f}% {r['anchor_boots_done']}/{r['anchor_boots']:<4d} "
                 f"{r['configs_defined']}/{r['configs_total']}  cut: {cut}")
    return "\n".join(L)


def sockread_warnings(ratio):
    """MGP-35: an arm with lambda < 0.353 prior (0.706 after extension; N=8 A 0.237) loses requests to sock_read."""
    w = []
    if ratio < 0.353:
        w.append("lambda < 0.353 x prior: HI waits can exceed sock_read=600 s -> CLIENT_ERROR (MGP-35)")
    return w


# ----------------------------------------------------------------------------- M2g plan
def m2g_configs(job):
    """§15-2: J_N carries the N=1 anchor boot set (G1-ctl / G2 floor / G0(i) rows for M3's anchor) and, for N>=2, TP_N and DP_N."""
    return ("N1",) + tuple(C.JOB_CONFIGS[job]) if job != "J1" else ("N1",)


def m2g_boots(job, dummy_profile="<DUMMY_PROFILE.json>"):
    """§3 + §15-2, ONE Job.  One boot per (config, arm) carries G0(i) health + G1 prompts + (TP agnostic/d44) G3b stress + (fused) G2
    teacher-forced; the N1 boots are the G1-ctl / G2 floor reference of THIS Job (same-Job principle).  J2 only: the G0(ii)/(iii)
    refusal boots.  J1 only: the G5(a) GPU descriptive boot.  G4 nsys boots where the registered TP in G4_NSYS_TPS is the Job's N."""
    boots = []
    g1_cmd = lambda tag, cfg: {"type": "cmd", "name": f"g1_{tag}",
                               "argv": ["python3", "workspace/engine-port/scripts/multigpu/m2g_correctness_gate.py",
                                        "serve-g1", "--endpoint", "{ENDPOINT}", "--out", "{OUT}/gates",
                                        "--config", cfg, "--arm", tag]}
    for cfg in m2g_configs(job):
        cs = config_spec(cfg)
        for arm in arm_order(cfg):
            steps = [g1_cmd(arm, cfg)]
            if arm == "fused":
                steps.append({"type": "cmd", "name": "g2", "argv": [
                    "python3", "workspace/engine-port/scripts/multigpu/m2g_correctness_gate.py", "serve-g2",
                    "--endpoint", "{ENDPOINT}", "--out", "{OUT}/gates", "--config", cfg, "--repeats", "2" if cfg == "N1" else "1"]})
            if cs["mode"] == "TP" and arm in C.G3B_ARMS:
                x = C.G3B_OFFER_MULT * prior(cs["N"], "A")
                steps += [{"type": "warmup", "shape": "A"},
                          dict(type="cell", shape="A", rung="G3B", seed=C.HI_SEEDS[0], ext=0, round=1,
                               block="m2g_g3b", offered=round(x, 6), n_req=int(round(x * C.G3B_DURATION_S)))]
            boots.append(dict(cs, arm=arm, kind="g0", block="m2g", boot_id=f"m2g_{cfg}_{arm}", steps=steps))
    if job == "J1":
        # F4: G5(a) GPU descriptive boot (§3 G5(a), MGP-5): N=1 agnostic under the AUTO-grid config; its decode-busy stream-index
        # histogram is recorded beside the manual run's (mg_drive.g5a_gpu_descriptive).  No exclusion rides on it.
        x5 = prior(1, "A") * C.HI_MULT
        boots.append(dict(config_spec("N1"), arm="agnostic", kind="g5a", block="m2g", boot_id="m2g_g5a_auto_N1",
                          config_rel=C.AUTO_GRID_CONFIG_REL,
                          steps=[{"type": "warmup", "shape": "A"},
                                 dict(type="cell", shape="A", rung="G5A", seed=C.HI_SEEDS[0], ext=0, round=1, block="m2g_g5a",
                                      offered=round(x5, 6), n_req=int(round(x5 * C.G5A_AUTO_S)))]))
    if job == C.G0II_JOB:
        for env_k, env_v in C.G0_REFUSED_ENVS:
            extra = {env_k: env_v}
            if (env_k, env_v) == ("PDMUX_R2_POLICY", "hybrid"):
                extra["PDMUX_MODEL_PROFILE"] = dummy_profile
            boots.append(dict(config_spec("TP2"), arm="agnostic", kind="g0ii", block="m2g", boot_id=f"m2g_refuse_{env_k}_{env_v}".replace("=", "_"),
                              steps=[], extra_env=extra, expect_refusal="PD-mux with tp_size="))
        boots.append(dict(config_spec("TP2"), arm="agnostic", kind="g0iii", block="m2g", boot_id="m2g_autogrid_TP2",
                          steps=[], config_rel=C.AUTO_GRID_CONFIG_REL, expect_refusal="PD-mux auto grid (divide_sm)"))
    for n in C.G4_NSYS_TPS:
        if job == "J1" or n != C.JOB_N[job]:
            continue
        cs = config_spec(f"TP{n}")
        x = prior(n, "A") * C.HI_MULT
        st = [{"type": "warmup", "shape": "A"},
              {"type": "cmd", "name": "nsys_start", "argv": ["nsys", "start", "-o", "{BOOT}/nsys", "--force-overwrite=true"]},
              dict(type="cell", shape="A", rung="G4", seed=C.HI_SEEDS[0], ext=0, round=1, block="m2g_g4",
                   offered=round(x, 6), n_req=int(round(x * C.G4_NSYS_S))),
              {"type": "cmd", "name": "nsys_stop", "argv": ["nsys", "stop"]}]
        boots.append(dict(cs, arm="d44", kind="g4", block="m2g", boot_id=f"m2g_g4_TP{n}_d44", steps=st, nsys=True))
    return boots


M2G_G3_COMM_S = 600.0         # estimate: 4 combos x R=500 reverse-order all-reduces + watchdog 120 s, per TP
M2G_G1_S = 150.0              # estimate: 64 prompts x 128 tokens, concurrency 32, + client start
M2G_G2_S = 240.0


def m2g_budget(job):
    total, used, rows = 0.0, 0.0, []
    for b in m2g_boots(job):
        t = C.EST_BOOT_S if not b.get("expect_refusal") else 120.0
        for st in b["steps"]:
            if st["type"] == "cmd":
                t += {"g1": M2G_G1_S, "g2": M2G_G2_S}.get(st["name"][:2], 20.0)
            elif st["type"] == "warmup":
                t += C.EST_WARMUP_S[st["shape"]]
            elif st["type"] == "cell":
                t += cell_seconds(st["shape"], st["offered"], st["n_req"], prior(b["N"], "A")) \
                     if st["rung"] in ("G4", "G5A") else (st["n_req"] / st["offered"]) + C.EST_DRAIN_S["A"] + C.EST_CELL_GAP_S
        rows.append((b["boot_id"], t))
        total += t
        used += t * boot_gpus_used(b)
    n = C.JOB_N[job]
    comm = M2G_G3_COMM_S if n >= 2 else 0.0
    total += comm
    used += comm * n
    doc = _budget_doc(job, "m2g", [(bid, 1, t, 0.0) for bid, t in rows], total, used, len(rows), 0)
    doc.update(rows=rows, g3_comm_s=comm, total_s=total, total_h=total / 3600.0)
    return doc


def m3_plan_counts():
    """rev12 §19-2 / §19-4: static plan (no extension) over the four Jobs -> cells (registered 560), floor blocks (20), X anchor cells,
    U7 xcheck steps, boots.  Asserted by the selftest and the dry-run."""
    cells = floors = xanchor = xcheck = boots = 0
    by_job = {}
    for job in C.JOBS:
        jc = 0
        for _name, bs in m3_static_boots(job):
            for b in bs:
                boots += 1
                for st in b["steps"]:
                    if st["type"] == "cell":
                        cells += 1
                        jc += 1
                        if b.get("block") == "anchor" and st["shape"] == "X":
                            xanchor += 1
                    elif st["type"] == "floor":
                        floors += 1
                    elif st["type"] == "cmd" and str(st.get("name", "")).startswith("xcheck"):
                        xcheck += 1
        by_job[job] = jc
    return {"cells": cells, "floor_blocks": floors, "x_anchor_cells": xanchor, "xcheck_steps": xcheck, "boots": boots, "cells_by_job": by_job}


def assert_plan_registered():
    c = m3_plan_counts()
    assert c["cells"] == C.M3_CELLS_EXPECTED, f"M3 static cells {c['cells']} != registered {C.M3_CELLS_EXPECTED}"
    assert c["floor_blocks"] == C.FLOOR_BLOCKS_EXPECTED, f"floor blocks {c['floor_blocks']} != registered {C.FLOOR_BLOCKS_EXPECTED}"
    assert c["x_anchor_cells"] == 3 * 2 * 4, f"X anchor cells {c['x_anchor_cells']} != 24 (U14: 3 Jobs x 2 arms x 4 seeds)"
    return c


# ----------------------------------------------------------------------------- rev12 §19-6 M3 submission gate (R-5, F-1)
def committed_matches_head(repo, path, head="HEAD"):
    """(ok, why): `path` (inside `repo`) is tracked at `head` with EXACTLY the bytes on disk.  Harness audit r1 R-H1 (2026-10-16):
    `git cat-file -e <head>:<rel>` + sha256(`git show <head>:<rel>`) == sha256(the file bytes, read through any symlink).
    `ls-files` + `git diff --quiet HEAD` is NOT used: inside a VESSL Job results/ is a symlink to /data/runs/<job>/results
    (job_entry.sh `cp -a` + `ln -s`), git reports every tracked file behind it as deleted, and that check was always false there.
    `rel` is taken WITHOUT resolving symlinks (the Job layout, rev11 R-2)."""
    import subprocess as _sp
    rel = os.path.relpath(os.path.abspath(str(path)), os.path.abspath(str(repo)))
    if rel.startswith(".."):
        return False, f"{path} is outside the repository"
    if not Path(path).is_file():
        return False, f"{rel} absent"
    r1 = _sp.run(["git", "-C", str(repo), "cat-file", "-e", f"{head}:{rel}"], capture_output=True, check=False)
    if r1.returncode != 0:
        return False, f"{rel} not tracked at {head}"
    r2 = _sp.run(["git", "-C", str(repo), "show", f"{head}:{rel}"], capture_output=True, check=False)
    if r2.returncode != 0:
        return False, f"git show {head}:{rel} failed (rc {r2.returncode})"
    if hashlib.sha256(r2.stdout).hexdigest() != hashlib.sha256(Path(path).read_bytes()).hexdigest():
        return False, f"{rel} bytes differ from {head}"
    return True, ""


def preflight_gate(repo, pass_file, image, head="HEAD", engine_digest_file=None, git=None):
    """(ok, reasons).  All three must hold, else ABORT_PREFLIGHT_NOT_PASSED:
      (i)   PASS_FILE (results/multigpu/PREFLIGHT_PASS.json [schema: verdict, post_commit, merge_tree_engine_digest, image, job_name,
            judge_json_sha256 -- written inside the preflight Job by preflight_judge.py `record`, R-H2]) and its .sha256 are committed at
            `head` with exactly the bytes on disk (committed_matches_head: holds in the Job's symlinked results/ layout, R-H1), the file
            sha256 equals the .sha256, verdict == "PREFLIGHT_PASS", and the recorded merge-tree engine digest equals the sealed
            ENGINE_DIGEST.txt;
      (ii)  the recorded preflight image (…@sha256:…) equals this Job's image (PDMUX_IMAGE);
      (iii) `git diff --stat <post_commit> <head> -- workspace/engine-port/{7 gate paths}` is EMPTY.
    `git` (tests): callable(args list) -> (rc, stdout)."""
    import subprocess as _sp
    repo = Path(repo)
    pf = Path(pass_file)

    def _git(args):
        if git:
            return git(args)
        r_ = _sp.run(["git", "-C", str(repo)] + list(args), capture_output=True, text=True, check=False)
        return r_.returncode, r_.stdout
    why = []
    shaf = Path(str(pf) + ".sha256")
    rel = lambda q: os.path.relpath(os.path.abspath(str(q)), os.path.abspath(str(repo)))      # no symlink resolution (Job layout, rev11 R-2)
    if not pf.exists() or not shaf.exists():
        return False, [f"(i) {pf.name} or its .sha256 absent"]
    for q in (pf, shaf):
        ok_q, why_q = committed_matches_head(repo, q, head)
        if not ok_q:
            why.append(f"(i) {q.name} not tracked/clean ({why_q or rel(q)})")
    raw = pf.read_bytes()
    if shaf.read_text().split()[:1] != [hashlib.sha256(raw).hexdigest()]:
        why.append("(i) PREFLIGHT_PASS sha256 != committed .sha256")
    try:
        rec = json.loads(raw)
    except ValueError:
        return False, why + ["(i) PREFLIGHT_PASS not JSON"]
    if rec.get("verdict") != "PREFLIGHT_PASS":
        why.append(f"(i) verdict {rec.get('verdict')!r} != PREFLIGHT_PASS")
    edf = Path(engine_digest_file) if engine_digest_file else HERE / "ENGINE_DIGEST.txt"
    sealed = edf.read_text().strip() if edf.exists() else "ABSENT"
    if rec.get("merge_tree_engine_digest") != sealed:
        why.append(f"(i) merge-tree digest {str(rec.get('merge_tree_engine_digest'))[:16]} != sealed {sealed[:16]}")
    dig = lambda x: str(x or "").split("@sha256:")[-1] if "@sha256:" in str(x or "") else None
    if not dig(image) or dig(rec.get("image")) != dig(image):
        why.append(f"(ii) image digest {dig(rec.get('image'))} != Job image {dig(image)}")
    post = rec.get("post_commit")
    if not post:
        why.append("(iii) post_commit missing")
    else:
        rc, out = _git(["diff", "--stat", post, head, "--"] + [f"workspace/engine-port/{x}" for x in C.PREFLIGHT_GATE_PATHS])
        if rc != 0:
            why.append(f"(iii) git diff failed (rc {rc})")
        elif out.strip():
            why.append("(iii) gate paths changed since the preflight POST: " + out.strip().splitlines()[-1])
    return (not why), why


# ----------------------------------------------------------------------------- printing / CLI
def fmt_budget(b, title):
    L = [f"== {title} =="]
    for name, nb, t, used in b["stages"]:
        L.append(f"  {name:22s} boots={nb:3d}  {t / 3600.0:6.2f} h   used {used / 3600.0:6.2f} GPU-h")
    L.append(f"  TOTAL boots={b['boots']} cells={b['cells']}  stages {b['total_h']:.2f} h + preflight {C.EST_PREFLIGHT_S / 3600:.2f} h = "
             f"Job {b['job_h']:.2f} h   hard cap {b['cap_h']:.0f} h   {'OVER CAP' if b['over_cap'] else 'within cap'}")
    L.append(f"  RESERVATION  spec {b['spec']} ({C.JOB_EXPECTED_GPUS[b['job']]} GPU)  reserved {b['reserved_gpu_h']:.1f} GPU-h  "
             f"used {b['used_gpu_h']:.1f} GPU-h  utilization {100 * b['utilization']:.0f}%  est. cost ${b['cost_usd']:.1f} "
             f"(model estimate, not a measurement)")
    return "\n".join(L)


def job_table():
    """All four Jobs, M2g and M3 (lambda = prior, audited model) -- the dry-run table the launcher output lacks."""
    L = [f"{'kind':4s} {'Job':3s} {'spec':20s} {'est_h':>6s} {'cap_h':>5s} {'reserved':>9s} {'used':>7s} {'util':>5s} {'est_$':>7s}  (GPU-h columns; lambda=prior, audited model)"]
    for kind in ("m2g", "m3"):
        for j in C.JOBS:
            b = m2g_budget(j) if kind == "m2g" else m3_budget(j)
            L.append(f"{kind:4s} {j:3s} {C.JOB_SPEC[j]:20s} {b['job_h']:6.2f} {b['cap_h']:5.0f} {b['reserved_gpu_h']:9.1f} {b['used_gpu_h']:7.1f} "
                     f"{100 * b['utilization']:4.0f}% {b['cost_usd']:7.1f}")
    return "\n".join(L)


def selftest():
    assert n_req_for(1.6 * 2.99) == 810 and n_req_for(1.6 * 0.69) == 190
    assert n_req_for(1.6 * 8 * 2.99) == 4000 and n_req_for(0.01) == 40
    assert len(arm_order("TP8")) == 4 and arm_order("TP8") == arm_order("TP8")
    assert hi_rung(8, "A", 1)["offered"] == round(2 * 1.6 * 8 * 2.99, 6)
    nb = lambda j: m3_budget(j)["boots"]
    assert nb("J1") == 4 * 2, nb("J1")                                        # N1: boot1 + boot2 per arm
    pc = assert_plan_registered()                                             # rev12: 560 cells, 20 floor blocks, 24 X anchor cells
    assert pc["xcheck_steps"] == 2 * (1 + 2 * 3), pc                          # U7: fused_sym + agnostic of N1 (J1) and of TP_N, DP_N (J2/J4/J8)
    assert nb("J2") == 2 + 2 * 8 and nb("J4") == 2 + 2 * 8 and nb("J8") == 8 + 2 + 2 * 8
    assert [b["arm"] for b in boots_anchor("J2")] and boots_anchor("J1") == []
    assert all(b["gpu_ids"] == "0" and b["block"] == "anchor" for b in boots_anchor("J4"))
    t = {"cells": {"TP2:agnostic": {"included": False}}}
    assert "TP2:agnostic" in excluded_set(t) and sorted(runnable_configs(t, "J2")) == ["DP2", "TP2"]
    assert "TP2" not in runnable_configs({"cells": {"TP2:fused": {"included": False}}}, "J2")
    assert assert_config_order_registered() and config_order("J8") == ["DP8", "TP8"] and config_order("J2") == ["DP2", "TP2"] and config_order("J1") == ["N1"]
    print("mg_plan selftest PASS")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", nargs="?", choices=("m3-plan", "m3-budget", "m3-forecast", "m2g-plan", "m2g-budget", "job-table",
                                               "config-order", "check-pass-table", "plan-counts", "preflight-gate", "committed-clean"))
    ap.add_argument("rest", nargs="*")
    ap.add_argument("--job", choices=C.JOBS, default=None, help="PDMUX_MG_JOB (J1 | J2 | J4 | J8)")
    ap.add_argument("--ratio", type=float, default=None, help="lambda/prior for every config")
    ap.add_argument("--ratio-tp", type=float, default=None)
    ap.add_argument("--ratio-dp", type=float, default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.cmd in ("m3-plan", "m3-budget", "m3-forecast", "m2g-plan", "m2g-budget", "config-order") and not a.job:
        ap.error(f"{a.cmd} needs --job")
    if a.cmd == "job-table":
        print(job_table())
        return 0
    if a.cmd == "preflight-gate":                                 # rest: REPO PASS_FILE IMAGE
        ok, why = preflight_gate(a.rest[0], a.rest[1], a.rest[2] if len(a.rest) > 2 else "")
        print("PREFLIGHT_GATE_OK" if ok else "ABORT_PREFLIGHT_NOT_PASSED " + " | ".join(why))
        return 0 if ok else 3
    if a.cmd == "committed-clean":                                # rest: REPO FILE  (m3_job.sh check_committed, R-H1)
        ok, why = committed_matches_head(a.rest[0], a.rest[1])
        print("COMMITTED_CLEAN" if ok else "NOT_COMMITTED_CLEAN " + why)
        return 0 if ok else 1
    if a.cmd == "plan-counts":
        print(json.dumps(assert_plan_registered(), sort_keys=True))
        return 0
    if a.cmd == "config-order":
        assert_config_order_registered()
        print(" ".join(config_order(a.job)), "(registered J2=J4=J8=[DP,TP]: assertion passed)")
        return 0
    if a.cmd == "m3-plan":
        for name, boots in m3_static_boots(a.job):
            cells = sum(1 for b in boots for s in b["steps"] if s["type"] == "cell")
            print(f"{name:16s} boots={len(boots):2d} cells={cells:3d}  arms={[b['arm'] for b in boots]}")
        return 0
    if a.cmd == "m3-forecast":
        print(fmt_forecast(a.job, m3_forecast_table(a.job)))
        return 0
    if a.cmd == "m3-budget":
        def ratio_of(cfg, a=a):
            if a.ratio is not None:
                return a.ratio
            if cfg.startswith("TP") and a.ratio_tp is not None:
                return a.ratio_tp
            if cfg.startswith("DP") and a.ratio_dp is not None:
                return a.ratio_dp
            return 1.0
        print(fmt_budget(m3_budget(a.job, ratio_of), f"M3 budget dry-run {a.job} (ratio TP/DP/N1 = {ratio_of('TP2')}/{ratio_of('DP2')}/{ratio_of('N1')})"))
        return 0
    if a.cmd == "m2g-plan":
        for b in m2g_boots(a.job):
            print(b["boot_id"], b["arm"], [s.get("name", s["type"]) for s in b["steps"]])
        return 0
    if a.cmd == "m2g-budget":
        print(fmt_budget(m2g_budget(a.job), f"M2g budget dry-run {a.job}"))
        return 0
    if a.cmd == "check-pass-table":
        t, abort = load_pass_table(a.rest[0], a.rest[1], a.rest[2] if len(a.rest) > 2 else None)
        print(abort or "PASS_TABLE_OK")
        return 0 if t else 3
    ap.error("command required")


if __name__ == "__main__":
    sys.exit(main())
