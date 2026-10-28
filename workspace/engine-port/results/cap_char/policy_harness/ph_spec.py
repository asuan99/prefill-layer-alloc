#!/usr/bin/env python3
"""Registered constants and Job plans of PREREG_HD_HE0cell (rev0.1, NOT SEALED).

Jobs: CAP1 (W1 capacity), CAP2 (W3 capacity), TUNE (bind+GATE retune), EVAL-a/b (HE0 cell, seeds 1-2 / 3-4),
HD-1..8 (H-D, one seed per Job).  Rates and the tuned config come from an ADDENDUM json written after the capacity Jobs
(PREREG Q-6); plans that need them refuse (ABORT_NO_ADDENDUM) without it.  Nothing here starts a server.

Path rule (rev3.11 lesson): job_entry.sh swaps workspace/engine-port/results for a volume symlink, so paths here use
os.path.abspath, never Path.resolve().
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from pathlib import Path

HERE = Path(os.path.abspath(__file__)).parent
CT_DIR = HERE.parent / "costterms_harness"
if str(CT_DIR) not in sys.path:
    sys.path.insert(0, str(CT_DIR))
import ct_spec as S  # noqa: E402  (NANO_ID, NANO_REV, MEM_FRACTION, ATTENTION_BACKEND, SERVER_SEED, STICKY_IDX, ...)

CONFIG_NAME = S.CONFIG_CANDIDATE            # 5 divisions D16/24/34/44/64: stream idx 1..5 (STICKY_IDX), _lo=1, _hi=5
D_GRID = (16, 24, 34, 44, 64)
IDX_OF_D = dict(S.STICKY_IDX)
D_OF_IDX = {v: k for k, v in IDX_OF_D.items()}

# ---- SLO (PREREG §2) -------------------------------------------------------------------------------------------
W1_TTFT_MS = 3000.0
ITL_MS = 60.0
W3_TTFT_K = 3.0                              # W3 TTFT SLO = 3 x floor (floor measured in the same Job, N3 rule)
CLIFF_BAND = 0.05                            # TTFT within SLO +- 5 %
CLIFF_FRAC = 0.10                            # > 10 % of requests in the band -> CLIFF_ADJACENT
MARGIN = 0.03                                # ln(1.03) decision margin (gate 3)

# ---- seeds -------------------------------------------------------------------------------------------------------
EVAL_SEEDS = tuple(range(1, 9))
HE0_SEEDS = (1, 2, 3, 4)
TUNE_SEEDS = (101, 102)                      # disjoint from every evaluation seed (PREREG 4-3)
CAP_SEEDS = (201, 202)

# ---- traces ------------------------------------------------------------------------------------------------------
HE0_NP = 200                                 # requests per LO/HI segment (KISTI sharegpt_vary_bench.sbatch NP)
HE0_ROUNDS = 3                               # ROUNDS
HE0_RATIO = 4.0                              # HI/LO (decision 44)
HE0_H = 1.5                                  # rate_HI = h * lambda*_best-static(W1)  [O-P3, draft value]
HD_VARIANTS = (("clean", 120), ("clean", 600), ("cont", 600))   # (variant, phase length s)  decision 33
HD_DRAIN_TIMEOUT_S = 120.0
HD_Q = 1.0
W3_MAX_NEW = 1024                            # natural-output cap (decision 47; value is [proposal])
CAP_CELL_REQUESTS = {"W1": 120, "W3": 48}
CAP_FACTOR = {"W1": 1.35, "W3": 1.5}
CAP_N_RATES = {"W1": 10, "W3": 6}
CAP_ATTAIN = 0.90

# ---- tuning grid (PREREG 4-3; index<->D mapping verified statically against pdmux config, see README note) ------------
TUNE_ANCHOR_IDX = (2, 3, 4)                  # D24, D34, D44
TUNE_DWELL = (3, 6)

HARD_CAP_MIN = {"CAP1": 196, "CAP2": 272, "TUNE": 247, "EVAL-a": 252, "EVAL-b": 252, **{f"HD-{i}": 312 for i in range(1, 9)}}


def stable(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def sha(obj) -> str:
    return hashlib.sha256(stable(obj).encode()).hexdigest()


def ladder(rmin: float, factor: float, n: int):
    if rmin <= 0 or factor <= 1 or n < 2:
        raise ValueError("ladder: rmin > 0, factor > 1, n >= 2")
    return [round(rmin * factor ** i, 6) for i in range(n)]


def he0_rates(lam_w1: float, h: float = HE0_H, ratio: float = HE0_RATIO):
    hi = h * lam_w1
    return {"lo": hi / ratio, "hi": hi}


def static_arm(D: int, sticky: bool = True):
    if D not in D_GRID:
        raise ValueError(f"D {D} not in {D_GRID}")
    return {"kind": "static", "D": D, "sticky": bool(sticky), "name": f"s{D}"}


def dynamic_arm(anchor_idx: int, dwell: int):
    if anchor_idx not in D_OF_IDX:
        raise ValueError(f"anchor_idx {anchor_idx} not a division index")
    return {"kind": "dynamic", "anchor_idx": int(anchor_idx), "dwell": int(dwell),
            "name": f"dyn_a{anchor_idx}_w{dwell}"}


def arm_env(arm: dict) -> dict:
    """PDMUX_* env of one boot (only what differs between arms; the common part is in ph_runner.boot_env)."""
    if arm["kind"] == "static":
        env = {"PDMUX_ENGINE_MODE": "pdmux", "PDMUX_R2_POLICY": "fixed", "PDMUX_R2_FIXED_DSM": str(arm["D"])}
        if arm.get("sticky", True):
            env["PDMUX_STICKY_PARTITION"] = "1"      # PDMUX_STICKY_PARTITION + PDMUX_SLO_SCHED is rejected by the engine
        return env
    if arm["kind"] == "dynamic":
        # no R2 policy, no sticky: the engine rejects sticky with SLO_SCHED; _slo_on = SLO_SCHED or r2_policy
        return {"PDMUX_ENGINE_MODE": "pdmux", "PDMUX_SLO_SCHED": "1", "PDMUX_SLO_MODE": "binding",
                "PDMUX_SLO_FEAS_GATE": "1", "PDMUX_TPOT_SLO_MS": str(int(ITL_MS)),
                "PDMUX_TTFT_SLO_MS": str(int(W1_TTFT_MS)), "PDMUX_SLO_ANCHOR_IDX": str(arm["anchor_idx"]),
                "PDMUX_SLO_DWELL": str(arm["dwell"])}
    raise ValueError(arm)


def _rng(job_id: str, salt: str = "order"):
    return random.Random(int(hashlib.sha256(f"{job_id}|{salt}".encode()).hexdigest()[:12], 16))


def _boot(job_id, boots, arm, runs, idx):
    return {"boot_id": f"{job_id}_b{idx:02d}_{arm['name']}", "arm": arm, "runs": runs}


def _need(addendum, *keys):
    if not addendum:
        raise RuntimeError("ABORT_NO_ADDENDUM: rates/config addendum (PREREG Q-6) required for this Job")
    cur = addendum
    for k in keys:
        if k not in cur:
            raise RuntimeError(f"ABORT_NO_ADDENDUM: missing {'.'.join(keys)}")
        cur = cur[k]
    return cur


def plan_cap(job_id: str, workload: str, addendum: dict):
    """CAP1 (W1) / CAP2 (W3): 5 D x ladder x 2 seeds, one boot per D, sticky static, PHASE OFF."""
    rmin = _need(addendum, "cap_ladder", workload, "rmin")
    rates = ladder(rmin, CAP_FACTOR[workload], CAP_N_RATES[workload])
    rng = _rng(job_id)
    boots = []
    ds = list(D_GRID)
    rng.shuffle(ds)
    for i, D in enumerate(ds):
        cells = [{"run_id": f"{job_id}_D{D}_r{r}_s{s}", "kind": "cap", "workload": workload, "rate": r, "seed": s,
                  "n_requests": CAP_CELL_REQUESTS[workload]} for r in rates for s in CAP_SEEDS]
        rng.shuffle(cells)
        boots.append(_boot(job_id, boots, static_arm(D), cells, i))
    if workload == "W3":
        boots[0]["floor"] = True                 # W3 floor (single-request TTFT median) measured in the Job's first boot
    return {"job": job_id, "kind": "CAP", "workload": workload, "rates": rates, "boots": boots,
            "needs_w3_floor": workload == "W3"}


def he0_run(run_id, seed, rates, np_=HE0_NP, rounds=HE0_ROUNDS):
    return {"run_id": run_id, "kind": "he0", "seed": seed, "rate_lo": rates["lo"], "rate_hi": rates["hi"],
            "np": np_, "rounds": rounds}


def plan_tune(job_id: str, addendum: dict):
    lam = _need(addendum, "lambda_star", "W1")
    rates = he0_rates(lam)
    rng = _rng(job_id)
    cfgs = [dynamic_arm(a, d) for a in TUNE_ANCHOR_IDX for d in TUNE_DWELL]
    rng.shuffle(cfgs)
    boots = []
    for i, arm in enumerate(cfgs):
        seeds = list(TUNE_SEEDS)
        rng.shuffle(seeds)
        boots.append(_boot(job_id, boots, arm, [he0_run(f"{job_id}_{arm['name']}_s{s}", s, rates) for s in seeds], i))
    return {"job": job_id, "kind": "TUNE", "rates": rates, "boots": boots}


def plan_eval(job_id: str, seeds, addendum: dict, sticky: bool | None = None):
    """HE0 evaluation Job: 5 static D (one boot, all seeds in random order) + tuned dynamic (one boot per seed)."""
    lam = _need(addendum, "lambda_star", "W1")
    tuned = _need(addendum, "tuned_config")
    sticky = addendum.get("static_sticky", True) if sticky is None else sticky
    rates = he0_rates(lam)
    rng = _rng(job_id)
    boots = []
    for D in D_GRID:
        order = list(seeds)
        rng.shuffle(order)
        boots.append((static_arm(D, sticky), [he0_run(f"{job_id}_s{D}_s{s}", s, rates) for s in order]))
    dyn = dynamic_arm(tuned["anchor_idx"], tuned["dwell"])
    for s in seeds:
        boots.append((dyn, [he0_run(f"{job_id}_dyn_s{s}", s, rates)]))
    rng.shuffle(boots)
    return {"job": job_id, "kind": "EVAL", "seeds": list(seeds), "rates": rates,
            "boots": [_boot(job_id, None, a, r, i) for i, (a, r) in enumerate(boots)]}


def hd_run(job_id, D, seed, variant, plen, start_first, rate_w1, rate_w3, w3_slo_ms):
    return {"run_id": f"{job_id}_s{D}_{variant}{plen}", "kind": "hd", "seed": seed, "variant": variant,
            "phase_len_s": plen, "start": start_first, "rate_w1": rate_w1, "rate_w3": rate_w3, "w3_ttft_ms": w3_slo_ms}


def plan_hd(k: int, addendum: dict, sticky: bool = True):
    """HD-k: seed k, 5 static arms, each boot serves the 3 variants (clean120, clean600, cont600) in random order."""
    if k not in EVAL_SEEDS:
        raise ValueError(k)
    job_id = f"HD-{k}"
    l1 = _need(addendum, "lambda_star", "W1")
    l3 = _need(addendum, "lambda_star", "W3")
    f3 = _need(addendum, "W3_floor_ms")
    r1, r3 = HD_Q * l1, HD_Q * l3
    start_first = "W1" if k % 2 == 1 else "W3"           # odd seed: W1 first, even: W3 first (PREREG 3-2)
    rng = _rng(job_id)
    ds = list(D_GRID)
    rng.shuffle(ds)
    boots = []
    for i, D in enumerate(ds):
        vs = list(HD_VARIANTS)
        rng.shuffle(vs)
        runs = [hd_run(job_id, D, k, v, pl, start_first, r1, r3, W3_TTFT_K * f3) for v, pl in vs]
        boots.append(_boot(job_id, None, static_arm(D, sticky), runs, i))
    return {"job": job_id, "kind": "HD", "seed": k, "q": HD_Q, "boots": boots}


def make_plan(job: str, addendum: dict | None = None):
    if job == "CAP1":
        return plan_cap("CAP1", "W1", addendum)
    if job == "CAP2":
        return plan_cap("CAP2", "W3", addendum)
    if job == "TUNE":
        return plan_tune("TUNE", addendum)
    if job == "EVAL-a":
        return plan_eval("EVAL-a", (1, 2), addendum)
    if job == "EVAL-b":
        return plan_eval("EVAL-b", (3, 4), addendum)
    if job.startswith("HD-"):
        return plan_hd(int(job.split("-")[1]), addendum)
    raise ValueError(f"unknown job {job}")


def plan_problems(plan: dict):
    """Registered-structure checks (used by the runner before a boot and by tests)."""
    out = []
    if set(TUNE_SEEDS) & set(EVAL_SEEDS) or set(CAP_SEEDS) & set(EVAL_SEEDS):
        out.append("tuning/capacity seeds overlap evaluation seeds")
    ids = [r["run_id"] for b in plan["boots"] for r in b["runs"]]
    if len(ids) != len(set(ids)):
        out.append("duplicate run_id")
    k = plan["kind"]
    if k == "EVAL":
        names = sorted(b["arm"]["name"] for b in plan["boots"])
        if sum(1 for b in plan["boots"] if b["arm"]["kind"] == "static") != 5:
            out.append("EVAL needs 5 static boots")
        if set(plan["seeds"]) & set(TUNE_SEEDS):
            out.append("tuning seed in evaluation Job")
        for b in plan["boots"]:
            if b["arm"]["kind"] == "dynamic" and len(b["runs"]) != 1:
                out.append("dynamic arm must reboot per seed")
    if k == "TUNE" and (set(r["seed"] for b in plan["boots"] for r in b["runs"]) != set(TUNE_SEEDS)):
        out.append("TUNE seeds != registered tuning seeds")
    if k == "HD":
        if sorted(b["arm"]["D"] for b in plan["boots"]) != sorted(D_GRID):
            out.append("HD arm set != D grid")
        for b in plan["boots"]:
            if sorted((r["variant"], r["phase_len_s"]) for r in b["runs"]) != sorted(HD_VARIANTS):
                out.append("HD variants incomplete")
    if k == "CAP" and len(plan["boots"]) != 5:
        out.append("CAP needs 5 boots")
    return out


def plans_sha(addendum=None, jobs=("CAP1", "CAP2", "TUNE", "EVAL-a", "EVAL-b", *[f"HD-{i}" for i in range(1, 9)])):
    res = {}
    for j in jobs:
        try:
            res[j] = sha(make_plan(j, addendum))
        except RuntimeError as e:
            res[j] = str(e)
    return res
