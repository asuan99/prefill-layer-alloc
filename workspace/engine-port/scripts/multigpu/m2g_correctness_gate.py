#!/usr/bin/env python3
"""M2g correctness gate for multi-GPU PD-mux (PREREG_MULTIGPU_2026-10-12.md rev6 sec 3 / 3-1).

STATUS (2026-10-12, engine-porter): harness code, CPU-tested only
(tests/test_m2g_gate.py).  NOTHING HERE HAS RUN ON A
GPU.  Writing it is not a PASS of anything; a PASS of any tier is scoped to the
tested model, (mode, N), configs and prompt set (C-8, MGP-2..4) and is not
"TP PD-mux correctness in general" (sec 7).

LAYOUT
  scorers   pure functions, one per tier (G0 i/ii/iii, G1, G1-ctl, G2, G3, G3b, G4,
            G5(b), G5(c) memory, G5(c) contamination (가)(나)).
  run_m2g   the Job sequence, written against an INJECTED `io` object (experiment-
            runner's runner/launcher wrappers), so the sequence itself is CPU-testable
            with a fake.  Required io methods are listed in `IO_METHODS`.
  build_pass_table   sec 3-1 FAIL units -> M2G_PASS_TABLE.json (fail-closed: a gate
            with no result is NOT_TESTED and excludes its coverage like a FAIL).
  run_g3    F1 (harness verdict 2026-10-12): G3 launcher -- `python -m torch.distributed.run`,
            preflight, timeout + process-group kill, launcher.json/stdout/stderr per combination;
            a result directory with no rank file and no HANG file is NOT_TESTED(HARNESS_ERROR).
  alive_after.json   F2: survival record written by the boot runner after the LAST step of every
            boot (make_alive_record / record_alive_after); G3b '서버 생존' reads it; a missing record is
            NOT_TESTED, never a silent pass.
  alive_after_<step>.json   (2026-10-13 repair, J4/J8 rev11): the same record written right after EVERY cmd
            step (<step> = the cmd step name, e.g. g1_fused).  G1 '비정상 종료 0' reads the record written right
            after that boot's G1 step (alive_after_g1_<arm>.json, after_step.name checked) -- NOT the boot-end
            record: a later step (G2 on the fused boot) dying must not be charged to G1.  Missing -> NOT_TESTED.
  G5(a)     is the deterministic CPU test scripts/multigpu/g5a_selector_equiv.py
            (sealing precondition, not a Job step); the Job records the N=1
            auto-vs-manual selection distribution as a DESCRIPTIVE value only (MGP-5).

Every registered constant below cites its PREREG section.  Constants the PREREG
does not register are in the IMPLEMENTATION block and say so.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

HERE = Path(os.path.abspath(__file__)).parent
ENG = HERE.parents[1]
MG_DIR = ENG / "results" / "multigpu"
sys.path.insert(0, str(MG_DIR))
import mg_const as C  # noqa: E402  (experiment-runner's single constant file; values cited there)

# ----------------------------------------------------------------- registered constants
CONFIGS_MULTI = ("TP2", "TP4", "TP8", "DP2", "DP4", "DP8")  # sec 3 G0(i): TP, DP in {2,4,8}
ARMS = C.ARMS  # sec 2: fused, fused_sym, agnostic, d44
PDMUX_ARMS = C.PDMUX_ARMS
HEALTH_WAIT_S = C.HEALTH_WAIT_S  # sec 3 G0(i) / H2: 1800 s; shorter waits -> NOT_TESTED (MGP-23)
GUARD_REFUSAL_TEXT = "PD-mux with tp_size="  # sec 3 G0(ii): multigpu_guard.check_tp_pdmux_combination
DEVICE_ASSERT_TEXT = "PD-mux auto grid (divide_sm)"  # sec 3 G0(iii): pdmux_context device assert
# sec 3 G0(ii): five refused envs, ONE AT A TIME at TP=2 (hybrid with a dummy profile)
G0_REFUSED_ENVS = (
    {"PDMUX_R2_POLICY": "generic"},
    {"PDMUX_R2_POLICY": "hybrid", "PDMUX_MODEL_PROFILE": "__DUMMY_PROFILE__"},
    {"PDMUX_SLO_SCHED": "1"},
    {"PDMUX_LA_COORD": "1"},
    {"PDMUX_TRUE_DUAL_WORKER": "1"},
)
G0_REFUSAL_TP = 2  # sec 3 G0(ii)(iii): "TP=2"
G1_N_PROMPTS = 64  # sec 3 G1
G1_MAX_NEW = 128  # sec 3 G1
G1_CONCURRENCY = 32  # sec 3 G1
G1_M_MARGIN = 0.05  # sec 3 G1: m median >= ctl m median - 0.05
G1_Q_MARGIN = 0.10  # sec 3 G1: q <= ctl q + 0.10
G1_CTL_UNINFORMATIVE_BELOW = 0.5  # sec 3 G1: ctl m median < 0.5 -> UNINFORMATIVE
G2_FLOOR_MULT = 4.0  # sec 3 G2: p99 <= max(4 x floor p99, 0.05 nat)
G2_ABS_NAT = 0.05  # sec 3 G2
G2_TOP1_MISMATCH_MAX = 0.02  # sec 3 G2: top-1 mismatch rate <= 2 %
G2_PCTL = 99.0  # sec 3 G2: |dlogprob| p99
G3_REPEATS = 500  # sec 3 G3
G3_WATCHDOG_S = 120.0  # sec 3 G3
G3_COMBOS = (("plain", None), ("green", (16, 92)), ("green", (44, 64)))  # sec 3 G3
G3B_OFFER_MULT = C.G3B_OFFER_MULT  # sec 3 G3b: 1.5 x (N x 2.99)
G3B_PRIOR_A = C.PRIOR_PER_GPU["A"]  # sec 3 G3b / sec 4-2: 2.99 req/s per GPU (shape A)
G3B_DURATION_S = C.G3B_DURATION_S  # sec 3 G3b: 600 s
G3B_PROGRESS_WINDOW_S = 60.0  # sec 3 G3b: "마지막 60 s 안 토큰 진행"
G3B_ARMS = C.G3B_ARMS  # sec 3 G3b: agnostic, d44
G4_NSYS_TPS = C.G4_NSYS_TPS  # sec 3 G4: TP=2,8 x d44, 30 s (descriptive)
G4_NSYS_S = C.G4_NSYS_S
NVML_EMPTY_GPU_MIB = C.NVML_EMPTY_GPU_MIB  # sec 3 G5(c)(가): memory.used < 1 GiB before each boot
HARD_CAP_S = max(C.HARD_CAP_S["m2g"].values())  # sec 6-2 / rev8: M2g 4 h per Job (same for J1/J2/J4/J8)

# ----------------------------------------------- IMPLEMENTATION block (NOT registered)
TEARDOWN_POLL_CAP_S = 300.0  # H7: poll memory release after teardown; harness choice, audit it
TEARDOWN_POLL_PERIOD_S = 5.0
CAPTURE_BEGIN = "Capture cuda graph begin"  # model_runner.init_device_graphs (logger.info, every rank)
CAPTURE_END = "Capture cuda graph end"
CAPTURE_FAILED = "Capture cuda graph failed"  # cuda_graph_runner.py
PIECEWISE_BEGIN = "Capture piecewise CUDA graph begin"  # H8 descriptive
PIECEWISE_END = "Capture piecewise CUDA graph end"
G3_COMBO_TIMEOUT_S = 900.0  # F1: outer wall-clock cap per (TP, combo) launcher run; the registered
                            # 120 s progress watchdog is INSIDE the ranks.  Harness choice, audit it.
G3_KILL_GRACE_S = 10.0  # SIGTERM -> SIGKILL grace for the launcher's process group on timeout
G3_LAUNCHER_MODULE = "torch.distributed.run"  # F1: `python -m torch.distributed.run` (no `torchrun` on PATH needed)
G3_STRESS_SCRIPT = HERE / "m2g_comm_reverse_stress.py"
HARNESS_ERROR = "HARNESS_ERROR"  # F1 / lesson 21: a measurement failure is NOT_TESTED, never FAIL
ALIVE_RECORD_NAME = "alive_after.json"  # F2: written by mg_boot after the LAST step of every boot
ALIVE_SCHEMA = "m2g_alive_after/v1"
ALIVE_STEP_FMT = "alive_after_{step}.json"  # per cmd step record (mg_boot), same schema; after_step.name = <step>
G1_STEP_FMT = "g1_{arm}"  # mg_plan.m2g_boots: the G1 cmd step of boot m2g_<cfg>_<arm> is named g1_<arm>

PASS, FAIL, NOT_TESTED, UNINFORMATIVE = "PASS", "FAIL", "NOT_TESTED", "UNINFORMATIVE"


def cfg_mode_n(cfg: str) -> Tuple[str, int]:
    if cfg == "N1":
        return "n1", 1
    return cfg[:2].lower(), int(cfg[2:])


def cell_key(cfg: str, arm: str) -> str:
    """Key format consumed by mg_label.aggregate(pass_table=...)."""
    return f"{cfg}:{arm}"


# ============================================================================ G0
def score_g0_boot(boot: Optional[Mapping]) -> Dict:
    """sec 3 G0(i): health within 1800 s.  A failure under a shorter wait cap is NOT_TESTED (MGP-23)."""
    if boot is None:
        return {"verdict": NOT_TESTED, "why": "not booted"}
    if boot.get("healthy"):
        return {"verdict": PASS, "health_s": boot.get("health_s")}
    cap = float(boot.get("health_wait_cap_s") or 0.0)
    if cap < HEALTH_WAIT_S:
        return {"verdict": NOT_TESTED, "why": f"health wait cap {cap} s < registered {HEALTH_WAIT_S} s"}
    return {"verdict": FAIL, "why": "no /health within the registered wait", "exit_code": boot.get("exit_code")}


def _died_abnormally(boot: Mapping) -> bool:
    rc = boot.get("exit_code")
    return (not boot.get("healthy")) and rc is not None and rc != 0


def score_g0_refusal(boot: Optional[Mapping], marker: str) -> Dict:
    """sec 3 G0(ii)/(iii): marker text in the server log AND abnormal termination."""
    if boot is None:
        return {"verdict": NOT_TESTED, "why": "not booted"}
    log = boot.get("log_text") or ""
    has = marker in log
    died = _died_abnormally(boot)
    ok = has and died
    return {"verdict": PASS if ok else FAIL, "marker_found": has, "died_abnormally": died,
            "exit_code": boot.get("exit_code"), "healthy": bool(boot.get("healthy"))}


def score_g0_ii(per_env: Sequence[Optional[Mapping]]) -> Dict:
    """All five refused (sec 3-1: one failure -> the whole TP axis)."""
    if len(per_env) != len(G0_REFUSED_ENVS):
        return {"verdict": NOT_TESTED, "why": f"{len(per_env)} of {len(G0_REFUSED_ENVS)} envs run"}
    rows = [score_g0_refusal(b, GUARD_REFUSAL_TEXT) for b in per_env]
    if any(r["verdict"] == NOT_TESTED for r in rows):
        v = NOT_TESTED
    else:
        v = PASS if all(r["verdict"] == PASS for r in rows) else FAIL
    return {"verdict": v, "per_env": rows}


# ============================================================================ G1
def g1_metrics(off: Mapping[str, Sequence[int]], on: Mapping[str, Sequence[int]]) -> Dict:
    """m = common-prefix length / max(len_off, len_on) per prompt (1.0 = identical);
    q = fraction of prompts with m < 1.  A prompt present in one arm only, or with an
    empty output in either, counts as a zero-length output (sec 3 G1 hard condition)."""
    ids = sorted(set(off) | set(on))
    ms, zero, first_diff = {}, [], {}
    for rid in ids:
        a, b = list(off.get(rid) or []), list(on.get(rid) or [])
        if not a or not b:
            zero.append(rid)
            ms[rid] = 0.0
            continue
        p = 0
        for x, y in zip(a, b):
            if x != y:
                break
            p += 1
        if p == len(a) == len(b):
            ms[rid] = 1.0
        else:
            ms[rid] = p / max(len(a), len(b))
            first_diff[rid] = p
    vals = list(ms.values())
    return {"n": len(ids), "m": ms, "m_median": statistics.median(vals) if vals else None,
            "q": (sum(1 for v in vals if v < 1.0) / len(vals)) if vals else None,
            "zero_length": zero, "first_diff": first_diff}


def score_g1(pair: Optional[Mapping], ctl: Optional[Mapping], abnormal_exit: Optional[bool] = None,
             exposure: Optional[float] = None) -> Dict:
    """sec 3 G1 against the same comparison at N=1 (G1-ctl) measured in the same Job.

    abnormal_exit is TRI-STATE (F2): True = a survival record shows the fused or the PD-mux
    server dead right after its G1 step; False = both records show it alive; None = a record is
    missing -> "비정상 종료 0" cannot be established -> NOT_TESTED (never silently False).

    Order: an abnormal exit or a zero-length output in the TESTED pair is FAIL regardless
    of the control; a broken control (zero-length outputs or missing) makes the pair
    NOT_TESTED; a control with m median < 0.5 makes the pair UNINFORMATIVE (not FAIL, no
    correctness sentence).  A missing survival record turns every outcome except an
    observed FAIL (output divergence beyond the margins) into NOT_TESTED -- a seen failure
    is not relabelled as untested."""
    out = {"exposure_split_steps": exposure, "abnormal_exit": abnormal_exit}  # MGP-2: reported beside the verdict
    if pair is None:
        return {**out, "verdict": NOT_TESTED, "why": "pair outputs missing"}
    if abnormal_exit is True or pair["zero_length"] or pair["n"] < G1_N_PROMPTS:
        return {**out, "verdict": FAIL, "why": "abnormal exit / zero-length output / prompts missing",
                "zero_length": pair["zero_length"], "n": pair["n"]}
    if abnormal_exit is None:
        r = score_g1(pair, ctl, False, exposure)
        r["abnormal_exit"] = None
        if r["verdict"] != FAIL:
            r.update(verdict=NOT_TESTED, why="survival record after the G1 step (alive_after_g1_<arm>.json) missing -> abnormal exit unknown",
                     verdict_if_alive=r["verdict"])
        return r
    if ctl is None or ctl["zero_length"] or ctl["n"] < G1_N_PROMPTS:
        return {**out, "verdict": NOT_TESTED, "why": "G1-ctl missing or broken"}
    out.update(m_median=pair["m_median"], q=pair["q"], ctl_m_median=ctl["m_median"], ctl_q=ctl["q"])
    if ctl["m_median"] < G1_CTL_UNINFORMATIVE_BELOW:
        return {**out, "verdict": UNINFORMATIVE}
    ok = (pair["m_median"] >= ctl["m_median"] - G1_M_MARGIN) and (pair["q"] <= ctl["q"] + G1_Q_MARGIN)
    return {**out, "verdict": PASS if ok else FAIL}


# ------------------------------------------------------------------ survival record (F2)
def make_alive_record(boot_id: str, poll_returncode: Optional[int], health_http_status: Optional[int],
                      health_error: Optional[str] = None, after_step: Optional[Mapping] = None,
                      steps_total: Optional[int] = None, t_epoch: Optional[float] = None) -> Dict:
    """Schema m2g_alive_after/v1 (one file per boot: boots/<boot_id>/alive_after.json).

      schema              "m2g_alive_after/v1"
      boot_id             str, e.g. "m2g_TP2_agnostic"
      t_epoch             float, time of the check (after the last step returned)
      after_step          {"index": int, "type": str, "status": str} of the LAST step in the boot's
                          step list (whatever its status), or null when the boot had no step
      steps_total         int, number of steps in the boot's step list
      poll_returncode     server Popen.poll() at the check: null = still running, int = exited
      health_http_status  HTTP status of GET <endpoint>/health (timeout 10 s), null = no response
      health_error        repr of the request exception, or null
      alive               derived = (poll_returncode is null AND health_http_status == 200);
                          readers RE-DERIVE it from the two raw fields and ignore this copy."""
    return {"schema": ALIVE_SCHEMA, "boot_id": boot_id, "t_epoch": time.time() if t_epoch is None else t_epoch,
            "after_step": dict(after_step) if after_step is not None else None, "steps_total": steps_total,
            "poll_returncode": poll_returncode, "health_http_status": health_http_status,
            "health_error": health_error,
            "alive": poll_returncode is None and health_http_status == 200}


def _http_health(endpoint: str, timeout_s: float) -> Tuple[Optional[int], Optional[str]]:
    import urllib.error
    import urllib.request
    try:
        with urllib.request.urlopen(endpoint.rstrip("/") + "/health", timeout=timeout_s) as resp:
            return int(resp.status), None
    except urllib.error.HTTPError as exc:
        return int(exc.code), repr(exc)
    except Exception as exc:  # noqa: BLE001 - recorded, never raised
        return None, repr(exc)


def record_alive_after(boot_dir: Path, boot_id: str, proc: Any, endpoint: str,
                       after_step: Optional[Mapping] = None, steps_total: Optional[int] = None,
                       timeout_s: float = 10.0, health: Optional[Callable] = None,
                       record_name: str = ALIVE_RECORD_NAME) -> Dict:
    """Boot end (default record_name): ONCE per boot, after the last step and BEFORE teardown.  Per cmd step
    (record_name = alive_step_record_name(<step>)): right after that step returned.  proc.poll() + GET /health ->
    boot_dir/<record_name>.  `health(endpoint, timeout_s) -> (status, error)` injectable."""
    rc = proc.poll() if proc is not None else -1
    st, err = (health or _http_health)(endpoint, timeout_s) if rc is None else (None, "process exited before check")
    rec = make_alive_record(boot_id, rc, st, err, after_step, steps_total)
    Path(boot_dir).mkdir(parents=True, exist_ok=True)
    (Path(boot_dir) / record_name).write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    return rec


def alive_step_record_name(step: str) -> str:
    return ALIVE_STEP_FMT.format(step=step)


def read_alive_after(boot_dir: Path, record_name: str = ALIVE_RECORD_NAME,
                     expect_step: Optional[str] = None) -> Optional[Dict]:
    """The record, or None when absent / unreadable / wrong schema / raw fields missing, or (expect_step given)
    when its after_step.name is not that step (a record of another step is not evidence for this one)."""
    f = Path(boot_dir) / record_name
    try:
        rec = json.loads(f.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(rec, dict) or rec.get("schema") != ALIVE_SCHEMA \
            or "poll_returncode" not in rec or "health_http_status" not in rec:
        return None
    if expect_step is not None and (rec.get("after_step") or {}).get("name") != expect_step:
        return None
    return rec


def alive_after(boot_dir: Path, record_name: str = ALIVE_RECORD_NAME, expect_step: Optional[str] = None) -> Optional[bool]:
    """True = alive after the last step (or after `expect_step` for a per-step record), False = dead,
    None = no valid record (-> NOT_TESTED)."""
    rec = read_alive_after(boot_dir, record_name, expect_step)
    if rec is None:
        return None
    return rec["poll_returncode"] is None and rec["health_http_status"] == 200


def alive_after_step(boot_dir: Path, step: str) -> Optional[bool]:
    """Survival right after cmd step `step` (alive_after_<step>.json written by mg_boot after that step)."""
    return alive_after(boot_dir, alive_step_record_name(step), expect_step=step)


def abnormal_exit_of(*alive: Optional[bool]) -> Optional[bool]:
    """G1 '비정상 종료 0' over the boots of a pair: any dead -> True; else any unknown -> None; else False."""
    if any(a is False for a in alive):
        return True
    if any(a is None for a in alive):
        return None
    return False


# ============================================================================ G2
def percentile(vals: Sequence[float], pct: float) -> float:
    """Linear interpolation between closest ranks (numpy 'linear')."""
    xs = sorted(vals)
    if not xs:
        raise ValueError("empty")
    k = (len(xs) - 1) * pct / 100.0
    lo = math.floor(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


# IMPLEMENTATION (2026-10-13 repair, code fact of the engine tree, not a registered constant): SGLang returns
# input_top_logprobs[0] = None BY CONSTRUCTION -- scheduler_output_processor_mixin._process_input_top_logprobs starts
# the list with [None] and pops the last entry, so entry j (j >= 1) is the distribution predicting input token
# logprob_start_len + j and entry 0 has no value.  With logprob_start_len = len(prompt ids) that is continuation
# position 0.  The first G2_LEADING_NO_LOGPROB positions of each prompt are therefore EXCLUDED from the comparison
# when they carry no finite value (counted in `excluded_positions`); a non-finite value anywhere else is still a
# structural problem.
G2_LEADING_NO_LOGPROB = 1


def _finite(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def g2_deltas(ref: Mapping[str, Sequence[Mapping]], test: Mapping[str, Sequence[Mapping]]) -> Dict:
    """Per teacher-forced position: |top1_lp(test) - top1_lp(ref)| and top-1 id mismatch.

    Position record: {"top1_id": int, "top1_lp": float[, "teacher_token": int]} -- the
    top-1 entry of the input_top_logprobs at that position.  Structural problems (missing
    prompt, length mismatch, different teacher token, non-finite value) are reported and
    make the comparison invalid (FAIL), never skipped -- EXCEPT the leading position(s) the API returns
    without a logprob (G2_LEADING_NO_LOGPROB), which are excluded and counted (`excluded_positions`)."""
    deltas, mism, problems, excluded, missing = [], 0, [], 0, []
    for rid in sorted(set(ref) | set(test)):
        a, b = ref.get(rid), test.get(rid)
        if not a or not b:
            problems.append(f"{rid}: presence")
            missing.append(rid)
            continue
        if len(a) != len(b):
            problems.append(f"{rid}: presence/length")
            continue
        for i, (x, y) in enumerate(zip(a, b)):
            if "teacher_token" in x and x.get("teacher_token") != y.get("teacher_token"):
                problems.append(f"{rid}[{i}]: teacher token differs")
                continue
            lx, ly = x.get("top1_lp"), y.get("top1_lp")
            if not (_finite(lx) and _finite(ly)):
                if i < G2_LEADING_NO_LOGPROB:
                    excluded += 1                     # no logprob by API construction (see G2_LEADING_NO_LOGPROB)
                else:
                    problems.append(f"{rid}[{i}]: non-finite logprob")
                continue
            deltas.append(abs(float(ly) - float(lx)))
            mism += int(x.get("top1_id") != y.get("top1_id"))
    n = len(deltas)
    return {"positions": n, "problems": problems, "excluded_positions": excluded, "missing_records": missing,
            "excluded_rule": f"position index < {G2_LEADING_NO_LOGPROB} without a finite logprob (API: input_top_logprobs[0] is None)",
            "p99": percentile(deltas, G2_PCTL) if n else None,
            "top1_mismatch_rate": (mism / n) if n else None}


def score_g2(tp1_a: Optional[Mapping], tp1_b: Optional[Mapping], tpn: Optional[Mapping]) -> Dict:
    """sec 3 G2: floor = TP=1 same server, two repeats.  FAIL is labelled
    TP_NUMERIC_DIVERGENT and does not exclude anything (MGP-3)."""
    if tp1_a is None or tp1_b is None or tpn is None:
        return {"verdict": NOT_TESTED, "why": "missing teacher-forced run"}
    floor = g2_deltas(tp1_a, tp1_b)
    test = g2_deltas(tp1_a, tpn)
    if floor["problems"] or floor["positions"] == 0:
        return {"verdict": NOT_TESTED, "why": "floor run invalid", "floor": floor}
    thr = max(G2_FLOOR_MULT * floor["p99"], G2_ABS_NAT)
    if test["missing_records"]:
        # 2026-10-13 repair: a prompt with NO record (client error / server died mid-run, e.g. the rev11 fused OOM) is a
        # measurement failure, not a numeric divergence -> NOT_TESTED, never TP_NUMERIC_DIVERGENT (lesson 21)
        return {"verdict": NOT_TESTED, "label": None, "why": f"{len(test['missing_records'])} prompt(s) without a TP=N record "
                "(measurement failure)", "threshold_nat": thr, "floor": floor, "test": test}
    if test["problems"] or test["positions"] == 0:
        v = FAIL
    else:
        v = PASS if (test["p99"] <= thr and test["top1_mismatch_rate"] <= G2_TOP1_MISMATCH_MAX) else FAIL
    return {"verdict": v, "label": "TP_NUMERIC_DIVERGENT" if v == FAIL else None,
            "threshold_nat": thr, "floor": floor, "test": test}


# ============================================================================ G3
def _nt_label(row: Optional[Mapping]) -> str:
    """Verdict text for the pass table: NOT_TESTED carries its sub-reason, e.g. NOT_TESTED(HARNESS_ERROR)."""
    if not row:
        return NOT_TESTED
    v = row.get("verdict", NOT_TESTED)
    r = row.get("not_tested_reason")
    return f"{v}({r})" if v == NOT_TESTED and r else v


def score_g3_combo(out_dir: Path, world: int, repeats: int = G3_REPEATS,
                   launcher_rc: Optional[int] = None) -> Dict:
    """One (TP=N, streams, split) run of m2g_comm_reverse_stress.py.

    PASS     <=> no HANG file AND every rank file present with completed == repeats and errors == 0.
    FAIL     <=> a HANG file, or SOME rank evidence (>= 1 readable rank file) that is incomplete/wrong.
    NOT_TESTED: directory absent (not attempted), or directory present with NO readable rank file
              and NO HANG file -> not_tested_reason HARNESS_ERROR (launcher missing, timeout before
              any rank result, import error ...; lesson 21: not a gate failure).  Exclusion is the
              same as FAIL (sec 3-1 last row); only the label differs.
    launcher.json (written by run_g3_combo) is read when present and reported."""
    out_dir = Path(out_dir)
    if not out_dir.is_dir():
        return {"verdict": NOT_TESTED, "why": f"{out_dir} absent"}
    launcher = None
    lj = out_dir / "launcher.json"
    if lj.exists():
        try:
            launcher = json.loads(lj.read_text())
        except ValueError:
            launcher = {"status": "LAUNCHER_JSON_UNREADABLE"}
    if launcher_rc is None and launcher:
        launcher_rc = launcher.get("rc")
    hangs = sorted(p.name for p in out_dir.glob("HANG.rank*.json"))
    ranks, bad, unreadable = {}, [], []
    for r in range(world):
        p = out_dir / f"rank{r}.json"
        if not p.exists():
            bad.append(f"rank{r}: no result file")
            continue
        try:
            d = json.loads(p.read_text())
        except ValueError:
            unreadable.append(r)
            bad.append(f"rank{r}: unreadable result file")
            continue
        ranks[r] = d
        if d.get("repeats") != repeats or d.get("completed") != repeats:
            bad.append(f"rank{r}: completed {d.get('completed')} / repeats {d.get('repeats')} != {repeats}")
        if d.get("errors") != 0:
            bad.append(f"rank{r}: {d.get('errors')} integer-sum errors")
        if d.get("world") != world:
            bad.append(f"rank{r}: world {d.get('world')} != {world}")
    base = {"hangs": hangs, "problems": bad, "launcher_rc": launcher_rc,
            "launcher_status": (launcher or {}).get("status"),
            "elapsed_s": {r: d.get("elapsed_s") for r, d in ranks.items()}}
    if not hangs and not ranks:
        return {**base, "verdict": NOT_TESTED, "not_tested_reason": HARNESS_ERROR,
                "why": f"{HARNESS_ERROR}: result directory present but no readable rank file and no HANG file"
                       f" (launcher status {base['launcher_status']}, rc {launcher_rc})"}
    return {**base, "verdict": PASS if not hangs and not bad else FAIL}


def score_g3(combos: Mapping[str, Optional[Mapping]]) -> Dict:
    """All three registered combinations must PASS for the TP=N.  An observed FAIL in any
    combination is FAIL even if another combination is NOT_TESTED (a seen hang is not
    relabelled as untested); otherwise any NOT_TESTED -> NOT_TESTED, carrying HARNESS_ERROR
    when a combination's run was a harness error.  Exclusion (sec 3-1) is identical."""
    want = [combo_name(s, sp) for s, sp in G3_COMBOS]
    rows = {k: combos.get(k) for k in want}
    out: Dict[str, Any] = {"combos": rows}
    if any(r is not None and r.get("verdict") == FAIL for r in rows.values()):
        out["verdict"] = FAIL
    elif any(r is None or r.get("verdict") == NOT_TESTED for r in rows.values()):
        out["verdict"] = NOT_TESTED
        if any(r is not None and r.get("not_tested_reason") == HARNESS_ERROR for r in rows.values()):
            out["not_tested_reason"] = HARNESS_ERROR
    else:
        out["verdict"] = PASS
    return out


# ------------------------------------------------------------------ G3 execution (F1)
def g3_launcher_argv(n: int, streams: str, split, out_dir: Path, python: str = sys.executable,
                     repeats: int = G3_REPEATS, watchdog_s: float = G3_WATCHDOG_S) -> List[str]:
    """`python -m torch.distributed.run --standalone --nproc-per-node N m2g_comm_reverse_stress.py ...`."""
    argv = [python, "-m", G3_LAUNCHER_MODULE, "--standalone", "--nproc-per-node", str(int(n)),
            str(G3_STRESS_SCRIPT), "--streams", streams]
    if split is not None:
        argv += ["--split", str(int(split[0])), str(int(split[1]))]
    return argv + ["--repeats", str(int(repeats)), "--timeout-s", str(float(watchdog_s)), "--out", str(out_dir)]


def g3_preflight(python: str = sys.executable, modules: Sequence[str] = (G3_LAUNCHER_MODULE,),
                 timeout_s: float = 120.0, env: Optional[Mapping[str, str]] = None) -> Dict:
    """Can this interpreter import the launcher?  Never raises.  ok False -> every G3 combination of
    the call is NOT_TESTED(HARNESS_ERROR) without a launch attempt."""
    code = "import importlib, sys\nfor m in sys.argv[1:]:\n    importlib.import_module(m)\n"
    out = {"python": python, "modules": list(modules), "stress_script": str(G3_STRESS_SCRIPT),
           "stress_script_present": G3_STRESS_SCRIPT.is_file()}
    try:
        r = subprocess.run([python, "-c", code, *modules], capture_output=True, text=True, timeout=timeout_s,
                           check=False, env=dict(env) if env is not None else None)
        out.update(rc=r.returncode, stderr_tail=r.stderr[-2000:], status="RAN")
    except subprocess.TimeoutExpired:
        out.update(rc=None, status="TIMEOUT", stderr_tail="")
    except (FileNotFoundError, PermissionError, OSError) as exc:
        out.update(rc=None, status="LAUNCHER_NOT_FOUND" if isinstance(exc, FileNotFoundError) else "OSERROR",
                   stderr_tail=repr(exc))
    out["ok"] = out.get("rc") == 0 and out["stress_script_present"]
    return out


def _kill_group(proc, grace_s: float) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except (ProcessLookupError, PermissionError):
            return
        try:
            proc.wait(timeout=grace_s)
            return
        except subprocess.TimeoutExpired:
            continue


def run_g3_combo(n: int, streams: str, split, out_dir: Path, env: Optional[Mapping[str, str]] = None,
                 timeout_s: float = G3_COMBO_TIMEOUT_S, python: str = sys.executable,
                 argv: Optional[Sequence[str]] = None, preflight: Optional[Mapping] = None,
                 kill_grace_s: float = G3_KILL_GRACE_S) -> Dict:
    """Launch ONE combination and score it.  Never raises for launcher problems.

    Writes into out_dir: launcher.json {argv, status, rc, timeout_s, t_start_epoch, elapsed_s, preflight},
    launcher.stdout.txt, launcher.stderr.txt.  status: RAN | TIMEOUT (process group killed) |
    LAUNCHER_NOT_FOUND (FileNotFoundError) | OSERROR | PREFLIGHT_FAILED (not launched).
    Returns score_g3_combo(out_dir, n) (+ the launcher record)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    argv = list(argv) if argv is not None else g3_launcher_argv(n, streams, split, out_dir, python)
    rec: Dict[str, Any] = {"argv": argv, "timeout_s": timeout_s, "t_start_epoch": time.time(), "rc": None,
                           "preflight": dict(preflight) if preflight is not None else None}
    t0 = time.monotonic()
    if preflight is not None and not preflight.get("ok"):
        rec["status"] = "PREFLIGHT_FAILED"
        (out_dir / "launcher.stdout.txt").write_text("")
        (out_dir / "launcher.stderr.txt").write_text(str(preflight.get("stderr_tail") or ""))
    else:
        with open(out_dir / "launcher.stdout.txt", "wb") as so, open(out_dir / "launcher.stderr.txt", "wb") as se:
            try:
                proc = subprocess.Popen(argv, stdout=so, stderr=se, env=dict(env) if env is not None else None,
                                        start_new_session=True)
            except FileNotFoundError as exc:
                rec.update(status="LAUNCHER_NOT_FOUND", error=repr(exc))
            except OSError as exc:
                rec.update(status="OSERROR", error=repr(exc))
            else:
                try:
                    rec["rc"] = proc.wait(timeout=timeout_s)
                    rec["status"] = "RAN"
                except subprocess.TimeoutExpired:
                    _kill_group(proc, kill_grace_s)
                    rec.update(status="TIMEOUT", rc=proc.poll())
                    # H-rec: rank workers can leave the launcher's process group; reap stragglers so the next boot's
                    # NVML-empty check (G5(c)(가)) is not polluted by a hung stress rank.
                    subprocess.run(["pkill", "-9", "-f", "m2g_comm_reverse_stress.py"], check=False)
                    rec["pkill_stress"] = True
    rec["elapsed_s"] = time.monotonic() - t0
    (out_dir / "launcher.json").write_text(json.dumps(rec, indent=1, sort_keys=True, default=str) + "\n")
    row = score_g3_combo(out_dir, n, G3_REPEATS)
    row["launcher"] = rec
    return row


def g3_combo_dir(gates: Path, n: int, streams: str, split) -> Path:
    return Path(gates) / f"g3_TP{int(n)}_{G3_DIR_TAGS[combo_name(streams, split)]}"


def run_g3(n: int, gates: Path, cuda_visible_devices: Optional[str] = None,
           timeout_s: float = G3_COMBO_TIMEOUT_S, python: str = sys.executable,
           base_env: Optional[Mapping[str, str]] = None, deadline_epoch: Optional[float] = None,
           preflight: Optional[Mapping] = None) -> Dict:
    """F1 entry point for mg_drive: TP=N x three registered combinations, then score -> gates/g3_TP{N}.json.

    cuda_visible_devices defaults to mg_const.GPU_SUBSETS[f"TP{N}"].  deadline_epoch: a combination is
    not started when now + timeout_s > deadline (its directory is not created -> NOT_TESTED, listed in
    `not_attempted`).  preflight: pass one g3_preflight() result to reuse it across N; None -> run it here."""
    gates = Path(gates)
    gates.mkdir(parents=True, exist_ok=True)
    env = dict(base_env if base_env is not None else os.environ)
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices or C.GPU_SUBSETS[f"TP{int(n)}"]
    pf = dict(preflight) if preflight is not None else g3_preflight(python, env=env)
    not_attempted = []
    for streams, split in G3_COMBOS:
        if deadline_epoch is not None and time.time() + timeout_s > deadline_epoch:
            not_attempted.append(combo_name(streams, split))
            continue
        run_g3_combo(n, streams, split, g3_combo_dir(gates, n, streams, split), env, timeout_s, python,
                     preflight=pf)
    r = cli_score_g3(gates, n)
    r.update(preflight=pf, not_attempted=not_attempted)
    _write(gates / f"g3_TP{int(n)}.json", r)
    return r


def combo_name(streams: str, split) -> str:
    return streams if split is None else f"{streams}_{split[0]}_{split[1]}"


# ============================================================================ G3b
def token_times(rec: Mapping) -> List[float]:
    """Absolute (benchmark-relative) token times of a trace_loadgen request record."""
    if rec.get("ttft_ms") is None:
        return []
    t = float(rec["sent_s"]) + float(rec["ttft_ms"]) / 1000.0
    out = [t]
    for itl in rec.get("token_itl_ms") or []:
        t += float(itl) / 1000.0
        out.append(t)
    return out


def score_g3b(requests: Optional[Sequence[Mapping]], server_alive: Optional[bool],
              duration_s: float = G3B_DURATION_S, g_fn: Optional[Callable] = None) -> Dict:
    """sec 3 G3b: completed requests > 0, server alive, a token in the last 60 s of the
    600 s offered window.  `g` (sec 4-3 Q_c statistic) is DESCRIPTIVE here (H9/MGP-33:
    the first time the estimator meets a real requests.jsonl)."""
    if requests is None or server_alive is None:
        return {"verdict": NOT_TESTED, "why": "stress run missing"}
    completed = sum(1 for r in requests if not r.get("error") and r.get("ttft_ms") is not None)
    lo, hi = duration_s - G3B_PROGRESS_WINDOW_S, duration_s
    progressed = any(lo <= t <= hi for r in requests for t in token_times(r))
    ok = completed > 0 and bool(server_alive) and progressed
    out = {"verdict": PASS if ok else FAIL, "completed": completed, "n": len(requests),
           "server_alive": bool(server_alive), "token_in_last_60s": progressed}
    if g_fn is not None and requests:
        try:
            g, T, means = g_fn(list(requests))
            out["g_descriptive"] = {"g": g, "T": T, "q_means": means}
        except Exception as exc:  # noqa: BLE001 - descriptive only
            out["g_descriptive"] = {"error": repr(exc)}
    return out


def g3b_offered_rps(n: int) -> float:
    return G3B_OFFER_MULT * n * G3B_PRIOR_A


# ============================================================================ G4
def score_g4(log_text: Optional[str], server_args: Optional[Mapping] = None) -> Dict:
    """Capture success: begin seen, every begin has an end, no capture failure text,
    cudagraph not disabled.  Piecewise (H8) recorded, not judged."""
    if log_text is None:
        return {"verdict": NOT_TESTED, "why": "no server log"}
    sa = server_args or {}
    b, e = log_text.count(CAPTURE_BEGIN), log_text.count(CAPTURE_END)
    failed = CAPTURE_FAILED in log_text
    eager = bool(sa.get("disable_cuda_graph"))
    ok = b >= 1 and e == b and not failed and not eager
    return {"verdict": PASS if ok else FAIL, "capture_begin": b, "capture_end": e,
            "capture_failed_text": failed, "disable_cuda_graph": eager,
            "piecewise": {"begin": log_text.count(PIECEWISE_BEGIN), "end": log_text.count(PIECEWISE_END),
                          "disable_piecewise_cuda_graph": sa.get("disable_piecewise_cuda_graph"),
                          "piecewise_cuda_graph_tokens": sa.get("piecewise_cuda_graph_tokens")}}


# ============================================================================ G5(b)
def _mix():
    spec = importlib.util.spec_from_file_location("mg_realized_mix_m2g", MG_DIR / "mg_realized_mix.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def score_g5b_arm(n: int, arm: str, telemetry_base: str) -> Dict:
    """sec 3 G5(b): every DP replica's realized (prefill_sms, decode_sms) is a row of the
    arm's registered table AND equals table[stream_index]; every replica file present."""
    mix = _mix()
    table = [tuple(x) for x in mix.REGISTERED_SM_COUNTS[arm]]
    bad, per = [], {}
    for r, path in enumerate(mix.rank_files(telemetry_base, "dp", n)):
        try:
            snaps, _ = mix.load_snapshots(path)
        except OSError as exc:
            bad.append(f"replica {r}: {exc!r}")
            continue
        if not snaps:
            bad.append(f"replica {r}: no non-startup snapshot")
            continue
        seen = set()
        for s in snaps:
            pd, idx = (s.get("prefill_sms"), s.get("decode_sms")), s.get("stream_index")
            seen.add(pd)
            if pd not in table or not isinstance(idx, int) or not 0 <= idx < len(table) or table[idx] != pd:
                bad.append(f"replica {r}: realized {pd} at index {idx} not in registered table")
                break
        per[r] = sorted(seen)
    return {"verdict": PASS if not bad else FAIL, "problems": bad, "realized": per}


def score_g5b(arm_results: Mapping[str, Optional[Mapping]]) -> Dict:
    """DP=N: over the PD-mux arms that booted (arms that did not boot are already excluded
    by G0(i)); no PD-mux arm evaluated -> NOT_TESTED (excludes the DP=N, sec 3-1)."""
    rows = {a: r for a, r in arm_results.items() if r is not None}
    if not rows:
        return {"verdict": NOT_TESTED, "why": "no PD-mux arm evaluated", "arms": {}}
    return {"verdict": PASS if all(r["verdict"] == PASS for r in rows.values()) else FAIL, "arms": rows}


# ============================================================================ G5(c)
def _label():
    spec = importlib.util.spec_from_file_location("mg_label_m2g", MG_DIR / "mg_label.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def memory_facts(meta: Mapping) -> Dict:
    """Boot cell_meta.json -> mg_label.memory_check facts (same extraction as mg_label.build_config_flags)."""
    si = meta.get("server_info") or {}
    return dict(mamba=meta.get("max_mamba_cache_size", []), max_total=meta.get("max_total_num_tokens", []),
                eff_cap=meta.get("max_running_requests_requested") or si.get("max_running_requests"))


def score_g5c_memory(arm_meta: Mapping[str, Optional[Mapping]]) -> Dict:
    """sec 3 G5(c)(i)-(iii) over the four arms of one config: ONE implementation of the
    rule, shared with M3 (mg_label.memory_check).  Fewer than four arms measured ->
    NOT_TESTED (counts as MEMORY_MISMATCH for citation, sec 3-1 last row)."""
    have = {a: memory_facts(m) for a, m in arm_meta.items() if m is not None}
    if set(have) != set(ARMS):
        return {"verdict": NOT_TESTED, "memory_mismatch": True,
                "why": f"arms measured {sorted(have)} != {list(ARMS)}"}
    res = _label().memory_check(have)
    return {"verdict": FAIL if res["memory_mismatch"] else PASS, **res}


def nvml_snapshot(gpu_ids: Sequence[int], nv: Any = None) -> Dict[int, Dict]:
    """memory.used (MiB) and the ITEM COUNT of nvmlDeviceGetComputeRunningProcesses per GPU
    (PID-namespace independent, P-D8).  `nv` injectable for CPU tests."""
    if nv is None:
        import pynvml as nv  # type: ignore[no-redef]
    nv.nvmlInit()
    try:
        out = {}
        for g in gpu_ids:
            h = nv.nvmlDeviceGetHandleByIndex(int(g))
            mem = nv.nvmlDeviceGetMemoryInfo(h)
            procs = nv.nvmlDeviceGetComputeRunningProcesses(h)
            out[int(g)] = {"memory_used_mib": float(mem.used) / (1024 * 1024), "n_procs": len(procs)}
        return out
    finally:
        nv.nvmlShutdown()


def check_contamination(pre: Mapping[int, Mapping], post: Optional[Mapping[int, Mapping]] = None,
                        expected_post: Optional[Mapping[int, int]] = None,
                        procs_visible: bool = True) -> Dict:
    """sec 3 G5(c) (가)(나), evaluation points (H7):
      pre-boot  (가) memory.used < 1 GiB on every GPU of the subset, (나) item count == 0;
      post-boot (나) item count == expected_post[gpu] (M2g: none -- the table is BEING
                measured; M3: the M2g G0 table, compared by the M3 harness).
    If NVML hides process entries (all-zero table, procs_visible False) only (가) is used."""
    bad = []
    for g, s in sorted(pre.items()):
        if not s.get("memory_used_mib", math.inf) < NVML_EMPTY_GPU_MIB:
            bad.append(f"gpu{g}: pre-boot memory.used {s.get('memory_used_mib')} MiB >= {NVML_EMPTY_GPU_MIB}")
        if procs_visible and s.get("n_procs") != 0:
            bad.append(f"gpu{g}: pre-boot {s.get('n_procs')} compute processes != 0")
    if post is not None and expected_post is not None and procs_visible:
        for g, want in sorted(expected_post.items()):
            got = (post.get(g) or post.get(str(g)) or {}).get("n_procs")
            if got != want:
                bad.append(f"gpu{g}: post-boot {got} compute processes != expected {want}")
    return {"gpu_contaminated": bool(bad), "problems": bad, "procs_visible": procs_visible}


def procs_visible_from_table(table: Mapping[str, Mapping]) -> bool:
    """(나) is usable iff at least one post-boot count in the M2g table is non-zero."""
    return any(int(c) > 0 for per in table.values() for c in per.values())


# ============================================================================ sec 3-1
def build_pass_table(res: Mapping, cfg_list: Optional[Sequence[str]] = None, g0ii_in_job: bool = True, job: Optional[str] = None) -> Dict:
    """sec 3-1 FAIL units -> M2G_PASS_TABLE (fail-closed: missing result = NOT_TESTED).

    `res` layout (what run_m2g produces):
      g0i[cell_key], g0ii, g0iii, g1[cell_key] (agnostic/d44), g2[cfg] (TP), g3[cfg] (TP),
      g3b[cell_key] (TP x agnostic/d44), g4[cell_key], g5b[cfg] (DP), g5c[cfg],
      contam[cell_key], nvml_post[cell_key] = {gpu: n_procs}
    """
    def v(section, key):
        row = (res.get(section) or {}).get(key)
        return (row or {}).get("verdict", NOT_TESTED)

    cells, configs = {}, {}
    g0ii = (res.get("g0ii") or {}).get("verdict", NOT_TESTED)
    g0iii = (res.get("g0iii") or {}).get("verdict", NOT_TESTED)
    # rev8/rev9: ONE table per Job.  `cfg_list` = that Job's configs (N1 + TPn/DPn); `g0ii_in_job` False (J4/J8, J1): the G0(ii) guard-refusal
    # result is J2's, read by M3 as a THREE-way digest check of J2's table -- this table does not judge the TP axis on it (g0ii recorded as
    # NOT_RUN_IN_JOB).
    for cfg in (tuple(cfg_list) if cfg_list is not None else ("N1",) + CONFIGS_MULTI):
        mode, n = cfg_mode_n(cfg)
        cfg_reasons, flags = [], []
        if v("g0i", cell_key(cfg, "fused")) != PASS:
            cfg_reasons.append(f"G0(i) fused boot {v('g0i', cell_key(cfg, 'fused'))}")
        if mode == "tp":
            if g0ii_in_job and g0ii != PASS:
                cfg_reasons.append(f"G0(ii) guard refusal {g0ii} (TP axis)")
            if v("g3", cfg) != PASS:
                cfg_reasons.append(f"G3 comm stress {_nt_label((res.get('g3') or {}).get(cfg))}")
            for arm in G3B_ARMS:
                if v("g3b", cell_key(cfg, arm)) != PASS:
                    cfg_reasons.append(f"G3b {arm} {_nt_label((res.get('g3b') or {}).get(cell_key(cfg, arm)))}")
            g2v = v("g2", cfg)
            if g2v == FAIL:
                flags.append("TP_NUMERIC_DIVERGENT")
            elif g2v != PASS:
                flags.append("G2_NOT_TESTED")
        if mode == "dp" and v("g5b", cfg) != PASS:
            cfg_reasons.append(f"G5(b) {v('g5b', cfg)}")
        g5c = (res.get("g5c") or {}).get(cfg)
        mem_mismatch = g5c is None or bool(g5c.get("memory_mismatch")) or g5c.get("verdict") != PASS
        if mem_mismatch:
            flags.append("MEMORY_MISMATCH" if g5c and g5c.get("verdict") == FAIL else "MEMORY_MISMATCH(NOT_TESTED)")
        for arm in ARMS:
            k = cell_key(cfg, arm)
            reasons = list(cfg_reasons)
            if arm != "fused" and v("g0i", k) != PASS:
                reasons.append(f"G0(i) {arm} boot {v('g0i', k)}")
            if arm in PDMUX_ARMS and cfg != "N1":
                g1 = v("g1", k)
                if g1 in (FAIL, NOT_TESTED):
                    reasons.append(f"G1 fused<->{arm} {g1}")
                elif g1 == UNINFORMATIVE and f"{arm}:G1_UNINFORMATIVE" not in flags:
                    flags.append(f"{arm}:G1_UNINFORMATIVE")  # no exclusion, no correctness sentence
            if v("g4", k) != PASS:
                reasons.append(f"G4 capture {v('g4', k)}")
            contam = (res.get("contam") or {}).get(k)
            if contam is None:
                reasons.append("G5(c) contamination check NOT_TESTED")
            elif contam.get("gpu_contaminated"):
                reasons.append("GPU_CONTAMINATED")
            cells[k] = {"included": not reasons, "reason": "; ".join(reasons), "reasons": reasons,
                        "not_tested": any("NOT_TESTED" in r for r in reasons)}
        configs[cfg] = {"flags": flags, "memory_mismatch": mem_mismatch,
                        "arm_rel_diff": (g5c or {}).get("arm_rel_diff"),
                        "MEMORY_RULE_III_COUNTERFACTUAL": (g5c or {}).get("MEMORY_RULE_III_COUNTERFACTUAL"),
                        "rule_iii_would_have_flagged": bool((g5c or {}).get("rule_iii_would_have_flagged")),
                        "tp_numeric_divergent": "TP_NUMERIC_DIVERGENT" in flags,
                        "g_sym_undefined": not (cells[cell_key(cfg, "agnostic")]["included"]
                                                and cells[cell_key(cfg, "fused_sym")]["included"])}
    nvml_post = res.get("nvml_post") or {}
    findings = []
    if g0ii_in_job and g0iii != PASS:
        findings.append(f"G0(iii) auto-grid device assert {g0iii} (no exclusion, sec 3-1)")
    return {
        "prereg": f"{C.PREREG} rev{C.PREREG_REV}",
        "schema": "cells[f'{config}:{arm}'] = {included, reason, reasons, not_tested}",
        "cells": cells,
        "configs": configs,
        "job": job,
        "g0ii": g0ii if g0ii_in_job else "NOT_RUN_IN_JOB", "g0iii": g0iii if g0ii_in_job else "NOT_RUN_IN_JOB",
        "findings": findings,
        # H7: post-boot compute-process item count per (config, arm) per GPU; M3 compares against it.
        "nvml_expected_procs": {k: {str(g): c for g, c in sorted(v_.items())} for k, v_ in sorted(nvml_post.items())},
        "nvml_procs_visible": procs_visible_from_table(nvml_post) if nvml_post else False,
    }


def write_pass_table(table: Mapping, path: Path) -> str:
    """Canonical JSON; returns its sha256 (the digest M3 checks, sec 3-1 last paragraph)."""
    text = json.dumps(table, indent=1, sort_keys=True) + "\n"
    Path(path).write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ============================================================================ Job sequence
IO_METHODS = (
    # boot(spec) -> {"healthy", "health_s", "health_wait_cap_s", "exit_code", "log_text",
    #                "meta" (cell_meta.json dict), "server_args" (dict), "telemetry", "green", "handle"}
    "boot",
    "log_text",  # (handle) -> str (fresh log)
    "alive",  # (handle) -> bool
    "teardown",  # (handle) -> None
    "generate",  # (handle, prompts, max_new, concurrency) -> {rid: [token ids]}
    "teacher_forced",  # (handle, prompts, continuations {rid: [ids]}) -> {rid: [position dicts]}
    "serve_stress",  # (handle, offered_rps, duration_s, shape) -> list of trace_loadgen request records
    "comm_stress",  # (n, streams, split, repeats, watchdog_s) -> (out_dir, launcher_rc)
    "nvml",  # (gpu_ids) -> {gpu: {"memory_used_mib", "n_procs"}}
    "prompts",  # () -> list of (rid, text): the 64 registered G1/G2 prompts (S/O tier, seed fixed)
    "clock",  # () -> monotonic seconds (hard cap)
    "sleep",  # (seconds)
)


def gpu_ids(cfg: str) -> List[int]:
    return [int(x) for x in C.GPU_SUBSETS[cfg].split(",")]


def _spec(cfg: str, arm: str, env: Optional[Mapping] = None, config_rel: Optional[str] = None, tag: str = "") -> Dict:
    mode, n = cfg_mode_n(cfg)
    return {"config": cfg, "mode": mode, "n": n, "arm": arm, "gpu_ids": gpu_ids(cfg),
            "env": dict(env or {}), "pdmux_config_rel": config_rel, "health_wait_s": HEALTH_WAIT_S,
            "tag": tag or f"{cfg}_{arm}"}


class M2gRun:
    """The Job sequence.  Hard cap (sec 6-2): once exceeded, every remaining step is
    skipped and its gate stays absent -> NOT_TESTED in the pass table (sec 3-1)."""

    def __init__(self, io: Any, hard_cap_s: float = HARD_CAP_S, g_fn: Optional[Callable] = None):
        missing = [m for m in IO_METHODS if not hasattr(io, m)]
        if missing:
            raise TypeError(f"io lacks {missing}")
        self.io = io
        self.t0 = io.clock()
        self.hard_cap_s = hard_cap_s
        self.g_fn = g_fn
        self.res: Dict[str, Any] = {k: {} for k in ("g0i", "g1", "g2", "g3", "g3b", "g4", "g5b", "g5c",
                                                   "contam", "nvml_post", "descriptive", "outputs")}
        self.truncated = False

    def over_cap(self) -> bool:
        if self.io.clock() - self.t0 > self.hard_cap_s:
            self.truncated = True
        return self.truncated

    # ------------------------------------------------------------------ boots
    def _boot_measured(self, cfg: str, arm: str, work: Callable[[Mapping], None]) -> None:
        if self.over_cap():
            return
        k = cell_key(cfg, arm)
        ids = gpu_ids(cfg)
        pre = self.io.nvml(ids)
        boot = self.io.boot(_spec(cfg, arm))
        self.res["g0i"][k] = score_g0_boot(boot)
        try:
            if boot.get("healthy"):
                self.res["nvml_post"][k] = {g: s["n_procs"] for g, s in self.io.nvml(ids).items()}
                self.res["contam"][k] = check_contamination(pre)  # post-boot table is measured here (H7)
                work(boot)
                self.res["g4"][k] = score_g4(self.io.log_text(boot["handle"]), boot.get("server_args"))
                self.res.setdefault("meta", {})[k] = boot.get("meta")
            else:
                self.res["contam"][k] = check_contamination(pre)
        finally:
            if boot.get("handle") is not None:
                self.io.teardown(boot["handle"])
            self._poll_release(ids)

    def _poll_release(self, ids):
        deadline = self.io.clock() + TEARDOWN_POLL_CAP_S
        while True:
            snap = self.io.nvml(ids)
            if all(s["memory_used_mib"] < NVML_EMPTY_GPU_MIB for s in snap.values()):
                return
            if self.io.clock() >= deadline:
                return  # the next boot's pre-check flags GPU_CONTAMINATED
            self.io.sleep(TEARDOWN_POLL_PERIOD_S)

    def _refusal_boot(self, env: Mapping, config_rel: Optional[str]) -> Optional[Mapping]:
        if self.over_cap():
            return None
        boot = self.io.boot(_spec(f"TP{G0_REFUSAL_TP}", "d44" if config_rel is None else "agnostic",
                                  env=env, config_rel=config_rel, tag="g0_refusal"))
        if boot.get("handle") is not None:
            boot = dict(boot, log_text=self.io.log_text(boot["handle"]))
            self.io.teardown(boot["handle"])
        self._poll_release(gpu_ids(f"TP{G0_REFUSAL_TP}"))
        return boot

    # ------------------------------------------------------------------ work per boot
    def _work(self, cfg: str, arm: str, prompts) -> Callable[[Mapping], None]:
        mode, n = cfg_mode_n(cfg)

        def work(boot):
            h = boot["handle"]
            out = self.io.generate(h, prompts, G1_MAX_NEW, G1_CONCURRENCY)
            self.res["outputs"][cell_key(cfg, arm)] = out
            # sec 3 G1 "비정상 종료 0": the server must still be alive after the G1 load
            self.res.setdefault("_abnormal", {})[cell_key(cfg, arm)] = not self.io.alive(h)
            if arm in PDMUX_ARMS:
                mix = _mix()
                et = mix.cell_e_time(arm, mode, n, boot.get("telemetry", ""), boot.get("green", ""))
                self.res["descriptive"].setdefault("exposure", {})[cell_key(cfg, arm)] = et.get("exposure_steps")
                if mode == "dp":
                    self.res.setdefault("_g5b_arm", {}).setdefault(cfg, {})[arm] = \
                        score_g5b_arm(n, arm, boot.get("telemetry", ""))
            if arm == "fused" and mode in ("n1", "tp"):
                conts = self.res["outputs"].get(cell_key("N1", "fused")) if mode == "tp" else out
                if conts is not None:
                    tf = self.io.teacher_forced(h, prompts, conts)
                    self.res.setdefault("_g2", {})[cfg] = tf
                    if mode == "n1":  # floor: same TP=1 server, second repeat
                        self.res["_g2"]["N1_repeat"] = self.io.teacher_forced(h, prompts, conts)
            if mode == "tp" and arm in G3B_ARMS and not self.over_cap():
                reqs = self.io.serve_stress(h, g3b_offered_rps(n), G3B_DURATION_S, "A")
                self.res["g3b"][cell_key(cfg, arm)] = score_g3b(reqs, self.io.alive(h), G3B_DURATION_S, self.g_fn)
        return work

    # ------------------------------------------------------------------ sequence
    def run(self) -> Dict:
        prompts = self.io.prompts()
        if len(prompts) != G1_N_PROMPTS:
            raise ValueError(f"{len(prompts)} prompts != registered {G1_N_PROMPTS}")
        # G3 (no server): TP = 2, 4, 8 x three combinations
        for cfg in ("TP2", "TP4", "TP8"):
            _, n = cfg_mode_n(cfg)
            combos = {}
            for streams, split in G3_COMBOS:
                if self.over_cap():
                    break
                out_dir, rc = self.io.comm_stress(n, streams, split, G3_REPEATS, G3_WATCHDOG_S)
                combos[combo_name(streams, split)] = score_g3_combo(Path(out_dir), n, G3_REPEATS, rc)
            self.res["g3"][cfg] = score_g3(combos)
        # N=1 (G1-ctl, G2 floor): all four arms
        for arm in ARMS:
            self._boot_measured("N1", arm, self._work("N1", arm, prompts))
        self._g5a_descriptive(prompts)
        # G0(ii) five refusals, G0(iii) auto grid -- TP=2
        per_env = [self._refusal_boot(env, None) for env in G0_REFUSED_ENVS]
        self.res["g0ii"] = score_g0_ii([b for b in per_env if b is not None]) if all(per_env) else \
            {"verdict": NOT_TESTED, "why": "hard cap"}
        b3 = self._refusal_boot({"PDMUX_R2_POLICY": ""}, C.AUTO_GRID_CONFIG_REL)
        self.res["g0iii"] = score_g0_refusal(b3, DEVICE_ASSERT_TEXT)
        # multi-GPU configs
        for cfg in CONFIGS_MULTI:
            for arm in ARMS:
                self._boot_measured(cfg, arm, self._work(cfg, arm, prompts))
        self._score_pairs()
        return self.result()

    def _g5a_descriptive(self, prompts):
        """sec 3 G5(a) GPU half, DESCRIPTIVE only (MGP-5): N=1 agnostic under the AUTO grid
        config; its decode-busy stream-index histogram is recorded beside the manual run's."""
        if self.over_cap():
            return
        ids = gpu_ids("N1")
        boot = self.io.boot(_spec("N1", "agnostic", config_rel=C.AUTO_GRID_CONFIG_REL, tag="N1_agnostic_auto"))
        try:
            if boot.get("healthy"):
                self.io.generate(boot["handle"], prompts, G1_MAX_NEW, G1_CONCURRENCY)
                mix = _mix()
                et = mix.cell_e_time("agnostic", "n1", 1, boot.get("telemetry", ""), boot.get("green", ""))
                hist = [(r.get("estimate") or {}).get("busy_index_hist") for r in et.get("per_rank", [])]
                self.res["descriptive"]["g5a_auto_index_hist"] = hist
                self.res["descriptive"]["g5a_auto_status"] = et.get("status")
            else:
                self.res["descriptive"]["g5a_auto_status"] = "BOOT_FAILED"
        finally:
            if boot.get("handle") is not None:
                self.io.teardown(boot["handle"])
            self._poll_release(ids)

    def _score_pairs(self):
        outs = self.res["outputs"]
        for cfg in ("N1",) + CONFIGS_MULTI:
            mode, n = cfg_mode_n(cfg)
            for arm in PDMUX_ARMS:
                f, a = outs.get(cell_key(cfg, "fused")), outs.get(cell_key(cfg, arm))
                pair = g1_metrics(f, a) if f is not None and a is not None else None
                if cfg == "N1":
                    self.res["descriptive"].setdefault("g1_ctl", {})[arm] = pair
                    continue
                ctl = self.res["descriptive"].get("g1_ctl", {}).get(arm)
                exp = self.res["descriptive"].get("exposure", {}).get(cell_key(cfg, arm))
                abn = self.res.get("_abnormal", {})
                ab = abnormal_exit_of(*[None if cell_key(cfg, x) not in abn else not abn[cell_key(cfg, x)]
                                        for x in ("fused", arm)])
                self.res["g1"][cell_key(cfg, arm)] = score_g1(pair, ctl, ab, exp)
            if mode == "tp":
                g2 = self.res.get("_g2", {})
                self.res["g2"][cfg] = score_g2_guarded(g2.get("N1"), g2.get("N1_repeat"), g2.get(cfg))
            if mode == "dp":
                self.res["g5b"][cfg] = score_g5b(self.res.get("_g5b_arm", {}).get(cfg, {}))
            self.res["g5c"][cfg] = score_g5c_memory({a: self.res.get("meta", {}).get(cell_key(cfg, a)) for a in ARMS})

    def result(self) -> Dict:
        out = {k: v for k, v in self.res.items() if not k.startswith("_") and k != "outputs"}
        out["truncated_by_hard_cap"] = self.truncated
        return out


# ============================================================================ HTTP clients
G1_PROMPT_SEED = 20260911  # r2_correctness_client default seed (PREREG sec 3 G1 "seed 고정"; value = client default)
G1_N_S, G1_N_O = 32, 32  # 64 = S tier 32 + O tier 32 prompt TEXTS (sent at concurrency 32, not the S/O protocol)
G2_ALIGN_MIN = 0.5  # IMPLEMENTATION guard (not registered): TP=1 top-1 must agree with its own greedy
                    # teacher on >= half the positions, else the position alignment is wrong -> NOT_TESTED
HTTP_TIMEOUT_S = 600.0  # IMPLEMENTATION: per request
G2_CONCURRENCY = 1  # IMPLEMENTATION (2026-10-13 repair, J4/J8 rev11 fused OOM): G2 requests are sent ONE AT A TIME.
                    # Not registered (PREREG sec 3 G2 fixes prompts / teacher / metric / floor, not the send concurrency);
                    # the floor (TP=1 x2) and the test (TP=N) use the same serial client.
G2_TOKCHECK_TAIL = 16  # IMPLEMENTATION: tail positions used to cross-check /tokenize against the /generate tokenisation


def registered_prompts() -> List[Tuple[str, str]]:
    """64 prompts from r2_correctness_client.plan (S and O tier texts, fixed seed)."""
    spec = importlib.util.spec_from_file_location(
        "r2_correctness_client_m2g", ENG / "results" / "r2_correctness" / "r2_correctness_client.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    seq, _conc, over = mod.plan(G1_PROMPT_SEED, G1_N_S, 0, G1_MAX_NEW, n_o=G1_N_O)
    return [(it["id"], it["text"]) for it in seq + over]


def _post(endpoint: str, body: Mapping, timeout_s: float = HTTP_TIMEOUT_S, path: str = "/generate") -> Dict:
    import urllib.request
    req = urllib.request.Request(endpoint.rstrip("/") + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.load(resp)


def _pool_map(fn, items, concurrency):
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        return list(ex.map(fn, items))


def http_generate(endpoint: str, prompts: Sequence[Tuple[str, str]], max_new: int = G1_MAX_NEW,
                  concurrency: int = G1_CONCURRENCY, post: Callable = _post) -> Dict[str, Any]:
    """sec 3 G1: greedy, ignore_eos, max_new 128, 32 in flight.  An error -> [] (zero-length = FAIL)."""
    def one(item):
        rid, text = item
        try:
            out = post(endpoint, {"text": text, "sampling_params": {"temperature": 0.0, "max_new_tokens": max_new,
                                                                     "ignore_eos": True}})
            return rid, list(out.get("output_ids") or []), None
        except Exception as exc:  # noqa: BLE001
            return rid, [], repr(exc)
    rows = _pool_map(one, prompts, concurrency)
    return {"outputs": {r: ids for r, ids, _ in rows}, "errors": {r: e for r, _, e in rows if e}}


def _tokenize(endpoint: str, text: str) -> List[int]:
    """Server's own tokenizer via POST /tokenize (entrypoints/openai/serving_tokenize.py: tokenizer_manager.tokenizer.encode
    with add_special_tokens=True -- the same tokenizer object and the HF default the /generate text path uses).  No GPU work."""
    out = _post(endpoint, {"prompt": text, "add_special_tokens": True}, path="/tokenize")
    return [int(t) for t in out["tokens"]]


def _prompt_ids(endpoint, text, post, tokenize: Callable = _tokenize):
    """Prompt ids of the server's own tokenisation WITHOUT input logprobs over the whole prompt.

    2026-10-13 repair (J4/J8 rev11): the old form sent logprob_start_len=0, so the prefill computed fp32 logits for EVERY
    prompt position (logits_processor._get_pruned_states keeps positions >= logprob_start_len; _copy_logits_to_buffer
    `.float()`), 8192 positions x vocab 131072 x 4 B = 4 GiB in one chunked-prefill batch -> fused TP4/TP8 OOM.
    Now: /tokenize, then a cross-check against the /generate tokenisation on the LAST G2_TOKCHECK_TAIL positions only
    (logprob_start_len = len - k: k positions of logits) plus the prompt length (meta_info.prompt_tokens).  A mismatch
    raises -> the prompt's record is empty -> presence problem -> the comparison is invalid (fail-closed)."""
    ids = [int(t) for t in tokenize(endpoint, text)]
    k = min(G2_TOKCHECK_TAIL, len(ids))
    out = post(endpoint, {"text": text, "sampling_params": {"temperature": 0.0, "max_new_tokens": 1},
                          "return_logprob": True, "logprob_start_len": len(ids) - k})
    mi = out.get("meta_info") or {}
    tail = [int(t[1]) for t in mi.get("input_token_logprobs") or []]
    if mi.get("prompt_tokens") != len(ids) or tail != ids[len(ids) - k:]:
        raise ValueError(f"/tokenize ids disagree with the /generate tokenisation (len {len(ids)} vs prompt_tokens "
                         f"{mi.get('prompt_tokens')}; tail match {tail == ids[len(ids) - k:]})")
    return ids


def http_teacher_forced(endpoint: str, prompts: Sequence[Tuple[str, str]], continuations: Mapping[str, Sequence[int]],
                        concurrency: int = G2_CONCURRENCY, post: Callable = _post, tokenize: Callable = _tokenize,
                        errors: Optional[Dict[str, str]] = None) -> Dict[str, List[Dict]]:
    """sec 3 G2: score prompt ids (server's own tokenisation) + the TP=1 greedy continuation;
    one position per continuation token: top-1 (id, logprob) of input_top_logprobs.  Logprobs are requested only from
    logprob_start_len = len(prompt ids) (continuation positions), one request at a time (G2_CONCURRENCY).  Entry 0 of
    input_top_logprobs is None by API construction (see G2_LEADING_NO_LOGPROB); it is recorded as is (top1 None) and
    excluded by the scorer.  A failed prompt -> [] and its repr in `errors` (when given)."""
    def one(item):
        rid, text = item
        cont = [int(x) for x in continuations.get(rid) or []]
        try:
            pids = _prompt_ids(endpoint, text, post, tokenize)
            out = post(endpoint, {"input_ids": pids + cont,
                                  "sampling_params": {"temperature": 0.0, "max_new_tokens": 1},
                                  "return_logprob": True, "logprob_start_len": len(pids), "top_logprobs_num": 1})
            tops = (out.get("meta_info") or {}).get("input_top_logprobs") or []
            pos = []
            for j, tok in enumerate(cont):
                top = tops[j] if j < len(tops) else None
                first = top[0] if top else None
                pos.append({"teacher_token": tok,
                            "top1_id": int(first[1]) if first else None,
                            "top1_lp": float(first[0]) if first and first[0] is not None else None})
            return rid, pos
        except Exception as exc:  # noqa: BLE001 - recorded as an empty list -> presence problem -> invalid
            if errors is not None:
                errors[rid] = repr(exc)
            return rid, []
    return dict(_pool_map(one, prompts, concurrency))


def score_g2_guarded(tp1_a, tp1_b, tpn) -> Dict:
    """score_g2 behind the IMPLEMENTATION alignment guard (G2_ALIGN_MIN): a misaligned
    position mapping would make floor and test equally meaningless and PASS vacuously."""
    agree = teacher_agreement(tp1_a) if tp1_a else None
    if tp1_a is not None and (agree or 0.0) < G2_ALIGN_MIN:
        r = {"verdict": NOT_TESTED, "why": f"TP=1 teacher agreement {agree} < {G2_ALIGN_MIN} (alignment check)"}
    else:
        r = score_g2(tp1_a, tp1_b, tpn)
    r["teacher_agreement_tp1"] = agree
    return r


def teacher_agreement(run: Mapping[str, Sequence[Mapping]]) -> Optional[float]:
    n = hit = 0
    for pos in run.values():
        for j, x in enumerate(pos):
            if j < G2_LEADING_NO_LOGPROB and not _finite(x.get("top1_lp")):
                continue                              # same exclusion as g2_deltas (no logprob by API construction)
            n += 1
            hit += int(x.get("top1_id") == x.get("teacher_token"))
    return hit / n if n else None


# ============================================================================ CLI
# Adapters for experiment-runner's mg_plan/mg_drive step contract (gates/ directory):
#   serve-g1 --endpoint E --out GATES --config CFG --arm ARM    -> g1_out_{CFG}_{ARM}.json
#   serve-g2 --endpoint E --out GATES --config CFG --repeats K  -> g2_tf_{CFG}_r{i}.json
#   score-g1 --out GATES --config CFG [--boots DIR] [--exposure-json F | --exposure-from-boots]
#                                                               -> g1_{CFG}.json  {"verdicts": {"fused|agnostic", "fused|d44"}}
#   run-g3   --out GATES --tp N [--timeout-s S] [--python PY] [--gpus 0,1] [--deadline-epoch T]
#                                                               -> g3_TP{N}_{plain,g16_92,g44_64}/ + g3_TP{N}.json
#   g3-preflight [--python PY]                                  -> JSON on stdout, rc 0 ok / 1 not ok
#   score-g2 --out GATES --config CFG                           -> g2_{CFG}.json
#   score-g3 --out GATES --tp N                                 -> g3_TP{N}.json  (reads g3_TP{N}_{plain,g16_92,g44_64}/)
G3_DIR_TAGS = {"plain": "plain", "green_16_92": "g16_92", "green_44_64": "g44_64"}


def _read(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def _write(p: Path, doc) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")


BOOT_ID_FMT = "m2g_{cfg}_{arm}"  # mg_plan boot ids (boots/<id>/), as read by mg_drive.assemble_pass_table


def g1_exposure(boots: Path, cfg: str, arm: str, window: Optional[Sequence[float]]) -> Dict:
    """F3 / MGP-2: the share of decode steps in a split group (mg_realized_mix `exposure_steps`)
    and the E-time, over the G1 send window ONLY (telemetry clock = time.perf_counter, CLOCK_MONOTONIC,
    which serve-g1 records as `g1_window_perf_counter_s`).  Descriptive: never changes a verdict.
    No window -> status NO_G1_WINDOW, value None (the whole-boot value would include other steps)."""
    mode, n = cfg_mode_n(cfg)
    bd = Path(boots) / BOOT_ID_FMT.format(cfg=cfg, arm=arm)
    if not window or len(window) != 2:
        return {"status": "NO_G1_WINDOW", "exposure_steps": None, "e_time": None, "window": None}
    et = _mix().cell_e_time(arm, mode, n, str(bd / "tel.jsonl"), str(bd / "green.json"),
                            (float(window[0]), float(window[1])))
    return {"status": et.get("status"), "exposure_steps": et.get("exposure_steps") if et.get("status") == "OK" else None,
            "e_time": et.get("e_time"), "window": [float(window[0]), float(window[1])],
            "reasons": et.get("reasons", [])}


def cli_score_g1(gates: Path, cfg: str, exposure: Optional[Mapping[str, float]] = None,
                 boots: Optional[Path] = None, exposure_from_boots: bool = False) -> Dict:
    """score-g1 adapter.  Survival (F2, 2026-10-13 repair): reads the record written right after the G1 step of
    each boot of the pair -- boots/<m2g_{cfg}_fused>/alive_after_g1_fused.json and
    boots/<m2g_{cfg}_{arm}>/alive_after_g1_{arm}.json (boots defaults to gates/../boots); a missing record
    -> that pair NOT_TESTED.  The boot-end record (alive_after.json) is reported beside it, never used: a later
    step of the same boot (G2 on the fused boot, G3b on the arm boot) is not part of G1.  Exposure (F3): `exposure` {arm: value} given explicitly wins;
    else with exposure_from_boots the G1-window value is computed (g1_exposure)."""
    gates = Path(gates)
    boots = Path(boots) if boots is not None else gates.parent / "boots"

    def doc_of(c, a):
        return _read(gates / f"g1_out_{c}_{a}.json")

    def outs(c, a):
        d = doc_of(c, a)
        return None if d is None else d.get("outputs")

    doc = {"config": cfg, "verdicts": {}, "detail": {}, "boots_dir": str(boots)}
    for arm in PDMUX_ARMS:
        f, a = outs(cfg, "fused"), outs(cfg, arm)
        pair = g1_metrics(f, a) if f is not None and a is not None else None
        if cfg == "N1":
            doc["verdicts"][f"fused|{arm}"] = "CTL"
            doc["detail"][f"fused|{arm}"] = pair
            continue
        cf, ca = outs("N1", "fused"), outs("N1", arm)
        ctl = g1_metrics(cf, ca) if cf is not None and ca is not None else None
        alive = {x: alive_after_step(boots / BOOT_ID_FMT.format(cfg=cfg, arm=x), G1_STEP_FMT.format(arm=x))
                 for x in ("fused", arm)}
        alive_end = {x: alive_after(boots / BOOT_ID_FMT.format(cfg=cfg, arm=x)) for x in ("fused", arm)}
        exp_detail = None
        if exposure is not None and arm in exposure:
            exp_val = exposure[arm]
            exp_detail = {"status": "GIVEN", "exposure_steps": exp_val}
        elif exposure_from_boots:
            exp_detail = g1_exposure(boots, cfg, arm, (doc_of(cfg, arm) or {}).get("g1_window_perf_counter_s"))
            exp_val = exp_detail["exposure_steps"]
        else:
            exp_val = None
        r = score_g1(pair, ctl, abnormal_exit_of(alive["fused"], alive[arm]), exp_val)
        r["alive_after"] = alive                       # right after each boot's G1 step (the G1 evidence)
        r["alive_after_boot_end"] = alive_end          # descriptive only (a death in a later step is that step's)
        r["exposure_detail"] = exp_detail
        doc["verdicts"][f"fused|{arm}"] = r["verdict"]
        doc["detail"][f"fused|{arm}"] = r
    _write(gates / f"g1_{cfg}.json", doc)
    return doc


def score_g3b_artifacts(out: Path, cfg: str, arm: str, g_fn: Optional[Callable] = None,
                        load_requests: Optional[Callable] = None) -> Dict:
    """G3b from Job artifacts (F2): requests = cells/m2g_{cfg}_{arm}__A_G3B_s*/requests.jsonl,
    server alive = boots/m2g_{cfg}_{arm}/alive_after.json (after the LAST step of that boot, hence
    after the G3b cell).  Missing requests or survival record -> NOT_TESTED."""
    out = Path(out)
    cds = sorted((out / "cells").glob(f"{BOOT_ID_FMT.format(cfg=cfg, arm=arm)}__A_G3B_s*")) \
        if (out / "cells").exists() else []
    rq = cds[0] / "requests.jsonl" if len(cds) == 1 else None
    reqs = None
    if rq is not None and rq.exists():
        if load_requests is not None:
            reqs = load_requests(rq)
        else:
            reqs = [json.loads(ln) for ln in rq.read_text().splitlines() if ln.strip()]
    alive = alive_after(out / "boots" / BOOT_ID_FMT.format(cfg=cfg, arm=arm))
    r = score_g3b(reqs, alive, G3B_DURATION_S, g_fn)
    if len(cds) > 1:
        r["why"] = f"{len(cds)} G3b cell directories (expected 1) -> NOT_TESTED"
    if alive is None and reqs is not None:
        r["why"] = "survival record (alive_after.json) missing -> server alive unknown"
    return r


def cli_score_g2(gates: Path, cfg: str) -> Dict:
    a, b = _read(gates / "g2_tf_N1_r0.json"), _read(gates / "g2_tf_N1_r1.json")
    r = score_g2_guarded(a, b, _read(gates / f"g2_tf_{cfg}_r0.json"))
    r["config"] = cfg
    _write(gates / f"g2_{cfg}.json", r)
    return r


def cli_score_g3(gates: Path, n: int) -> Dict:
    combos = {}
    for (streams, split) in G3_COMBOS:
        name = combo_name(streams, split)
        combos[name] = score_g3_combo(gates / f"g3_TP{n}_{G3_DIR_TAGS[name]}", n, G3_REPEATS)
    r = score_g3(combos)
    r["tp"] = n
    # merge: a re-score must not erase what run_g3 recorded (preflight, not_attempted)
    try:
        old = json.loads((gates / f"g3_TP{n}.json").read_text())
    except (OSError, ValueError):
        old = {}
    for k in ("preflight", "not_attempted"):
        if k in old and k not in r:
            r[k] = old[k]
    _write(gates / f"g3_TP{n}.json", r)
    return r


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("serve-g1", "serve-g2"):
        p = sub.add_parser(name)
        p.add_argument("--endpoint", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--config", required=True)
        if name == "serve-g1":
            p.add_argument("--arm", required=True, choices=ARMS)
        else:
            p.add_argument("--repeats", type=int, default=1)
    p = sub.add_parser("score-g1")
    p.add_argument("--out", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--boots", help="boot directories root (default: OUT/../boots) -- alive_after.json, tel.jsonl")
    ge = p.add_mutually_exclusive_group()
    ge.add_argument("--exposure-json", help='{"agnostic": x, "d44": y} given exposure (MGP-2)')
    ge.add_argument("--exposure-from-boots", action="store_true",
                    help="compute the G1-window exposure from boots/<id>/tel.jsonl + green.json (F3)")
    p = sub.add_parser("score-g2")
    p.add_argument("--out", required=True)
    p.add_argument("--config", required=True)
    p = sub.add_parser("run-g3", help="F1: launch the three G3 combinations for TP=N and score them")
    p.add_argument("--out", required=True)
    p.add_argument("--tp", type=int, required=True, choices=(2, 4, 8))
    p.add_argument("--timeout-s", type=float, default=G3_COMBO_TIMEOUT_S)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--gpus", help="CUDA_VISIBLE_DEVICES (default mg_const.GPU_SUBSETS['TP<N>'])")
    p.add_argument("--deadline-epoch", type=float)
    p = sub.add_parser("g3-preflight")
    p.add_argument("--python", default=sys.executable)
    p = sub.add_parser("score-g3")
    p.add_argument("--out", required=True)
    p.add_argument("--tp", type=int, required=True)
    p = sub.add_parser("pass-table", help="results JSON (M2gRun.result()) -> M2G_PASS_TABLE.json + sha256")
    p.add_argument("results")
    p.add_argument("--out", required=True)
    p = sub.add_parser("nvml", help="print the NVML (가)(나) snapshot for GPUs")
    p.add_argument("--gpus", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "serve-g1":
        prompts = registered_prompts()
        t0 = time.perf_counter()  # same clock as the engine telemetry timestamp_monotonic_s (F3 window)
        doc = http_generate(a.endpoint, prompts)
        t1 = time.perf_counter()
        doc.update(config=a.config, arm=a.arm, g1_window_perf_counter_s=[t0, t1],
                   g1_window_clock="time.perf_counter (CLOCK_MONOTONIC; engine telemetry timestamp_monotonic_s)")
        _write(Path(a.out) / f"g1_out_{a.config}_{a.arm}.json", doc)
        return 0 if not doc["errors"] else 1
    if a.cmd == "serve-g2":
        prompts = registered_prompts()
        ref = _read(Path(a.out) / "g1_out_N1_fused.json")
        if ref is None:
            print("serve-g2: g1_out_N1_fused.json absent (TP=1 teacher missing) -> G2 NOT_TESTED", file=sys.stderr)
            return 2
        for i in range(a.repeats):
            errs: Dict[str, str] = {}
            t0 = time.time()
            _write(Path(a.out) / f"g2_tf_{a.config}_r{i}.json",
                   http_teacher_forced(a.endpoint, prompts, ref["outputs"], errors=errs))
            _write(Path(a.out) / f"g2_tf_{a.config}_r{i}.client.json",
                   {"concurrency": G2_CONCURRENCY, "tokenize": "/tokenize + tail check", "tokcheck_tail": G2_TOKCHECK_TAIL,
                    "logprob_start_len": "len(prompt ids)", "errors": errs, "seconds": time.time() - t0})
        return 0
    if a.cmd == "score-g1":
        exp = json.loads(Path(a.exposure_json).read_text()) if a.exposure_json else None
        cli_score_g1(Path(a.out), a.config, exp, Path(a.boots) if a.boots else None, a.exposure_from_boots)
        return 0
    if a.cmd == "run-g3":
        r = run_g3(a.tp, Path(a.out), a.gpus, a.timeout_s, a.python, deadline_epoch=a.deadline_epoch)
        print(json.dumps({"tp": a.tp, "verdict": r["verdict"], "not_tested_reason": r.get("not_tested_reason"),
                          "not_attempted": r["not_attempted"], "preflight_ok": r["preflight"]["ok"]}))
        return {PASS: 0, FAIL: 1}.get(r["verdict"], 3)
    if a.cmd == "g3-preflight":
        pf = g3_preflight(a.python)
        print(json.dumps(pf, indent=1, sort_keys=True))
        return 0 if pf["ok"] else 1
    if a.cmd == "score-g2":
        cli_score_g2(Path(a.out), a.config)
        return 0
    if a.cmd == "score-g3":
        r = cli_score_g3(Path(a.out), a.tp)
        return 0 if r["verdict"] == PASS else 1
    if a.cmd == "pass-table":
        res = json.loads(Path(a.results).read_text())
        digest = write_pass_table(build_pass_table(res), Path(a.out))
        print(f"M2G_PASS_TABLE sha256 {digest}")
        return 0
    if a.cmd == "nvml":
        print(json.dumps(nvml_snapshot([int(x) for x in a.gpus.split(",")]), indent=1, sort_keys=True))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
