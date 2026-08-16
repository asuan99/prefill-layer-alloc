#!/usr/bin/env python3
"""G16 block post-hoc ARTIFACT-INTEGRITY inspection (experiment-runner task,
2026-08-16, job 884336 = G16 block 1 monitoring).

STRICT SCOPE, per the task that spawned this file:
  - This script computes NOTHING about donor / Delta / Delta_SLO / S_itl /
    "which arm wins" -- ANY decision quantity. One block cannot support K1
    (>= 3/4 blocks). Estimand computation is g16_analyze.py's job, run only
    once >= 3 blocks exist, and only with claims-auditor sign-off on the
    result.
  - This script answers exactly 5 structural questions (task list, verbatim
    order):
      1. G16_BLOCK_DONE: BOOTS_FAILED / failed_arms / pin_failed_arms
      2. per-arm `exact` (H3'-a) for every arm in this block's ARM_ORDER
      3. wall-clock decomposition: total elapsed vs per-arm boot_s/bench_s
         vs job-level fixed cost
      4. residency_fraction (H3'-b) populated for ALL arms in the block
         (commit 37cf6b8 regression check)
      5. sidecar adoption rule from g16_analyze.py's load_campaign_grid:
         status=='completed' AND telem_rc==0 AND fields_rc.lo==0 AND
         fields_rc.hi==0

Labelling discipline (project methodology gate #42 / lessons item 21 and
45 -- "don't label a measurement failure as a gate success", and gate #44 /
lessons item 47 -- "no empty-signature pass"):
  - Every check returns an explicit status in {OK, WARN, FAIL, NA}. NA means
    "not yet applicable" (e.g. job still PENDING/RUNNING, or this block's
    mode legitimately has fewer arms than a FULL block) -- it is NEVER used
    to mean "skipped without looking".
  - A check that finds literally zero artifacts to inspect, when the job
    state says artifacts SHOULD exist (a terminal sacct state), reports FAIL
    with reason NO_DATA_FOUND -- it does not fall through to OK/NA.
  - The overall verdict is a literal conjunction over a list built from all
    5 check results; the script asserts (a) the list has exactly 5 entries,
    (b) each entry's declared id matches its position, before computing the
    conjunction -- this is the gate-#42 defect (a conjunction that silently
    doesn't reference one of its own conditions) made structurally
    impossible here, not just avoided by convention.

Usage:
    python3 g16_blk1_inspect.py --job 884336
    python3 g16_blk1_inspect.py --job 884320   # dry-run against smoke1
    python3 g16_blk1_inspect.py --job 884292 --json
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
LOCAL_TZ = timezone(timedelta(hours=9))  # KST, verified via `date`/tzname at write time
STATUS_LEVELS = ("OK", "WARN", "FAIL", "NA")
TERMINAL_STATES = (
    "COMPLETED", "FAILED", "TIMEOUT", "CANCELLED", "OUT_OF_MEMORY",
    "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED",
)

# SMOKE mode's arm order is hardcoded in g16_grid.sbatch (not derived from
# g16_arm_order.py) -- mirrored here ONLY for the dry-run cross-check
# against smoke artifacts; FULL-mode blocks are cross-checked against
# g16_arm_order.py itself (see cross_check_arm_order).
SMOKE_ARM_ORDER = ["d44", "d64", "d74"]


def die(msg: str) -> None:
    print(f"INSPECT_FATAL {msg}", file=sys.stderr)
    sys.exit(3)


# ---------------------------------------------------------------------------
# sacct
# ---------------------------------------------------------------------------
def sacct_info(job: str) -> Dict[str, Any]:
    """Returns dict with State/Elapsed/Start/End for the MAIN job step only
    (JobID exactly == job, excludes .batch/.extern). Never raises -- a
    failure to invoke/parse sacct is reported via ok=False + reason, not an
    exception (this script must not crash on a scheduler hiccup)."""
    out: Dict[str, Any] = {"ok": False, "job": job}
    try:
        proc = subprocess.run(
            ["sacct", "-j", str(job), "--format=JobID,State,Elapsed,Start,End",
             "-P", "--noheader"],
            capture_output=True, text=True, timeout=30, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        out["reason"] = f"sacct invocation failed: {exc!r}"
        return out
    if proc.returncode != 0:
        out["reason"] = f"sacct rc={proc.returncode} stderr={proc.stderr.strip()!r}"
        return out
    row = None
    for line in proc.stdout.splitlines():
        parts = line.split("|")
        if len(parts) >= 5 and parts[0] == str(job):
            row = parts
            break
    if row is None:
        out["reason"] = f"no sacct row with JobID=={job} (job not submitted, or purged)"
        out["raw_stdout"] = proc.stdout
        return out
    _, state, elapsed, start, end = row[:5]
    out.update(ok=True, state=state.strip(), elapsed_str=elapsed.strip(),
               start_str=start.strip(), end_str=end.strip())
    return out


def elapsed_to_seconds(s: str) -> Optional[float]:
    s = s.strip()
    if not s or s in ("Unknown", "INVALID", "None"):
        return None
    days = 0
    if "-" in s:
        d, s = s.split("-", 1)
        days = int(d)
    parts = s.split(":")
    parts = [float(p) for p in parts]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    h, m, sec = parts[-3], parts[-2], parts[-1]
    return days * 86400 + h * 3600 + m * 60 + sec


def parse_sacct_local_dt(s: str) -> Optional[datetime]:
    s = s.strip()
    if not s or s in ("Unknown", "None"):
        return None
    try:
        dt = datetime.strptime(s, "%Y-%m-%dT%H:%M:%S")
    except ValueError:
        return None
    return dt.replace(tzinfo=LOCAL_TZ)


# ---------------------------------------------------------------------------
# .out log parsing
# ---------------------------------------------------------------------------
@dataclass
class OutLog:
    path: Path
    exists: bool
    lines: List[str] = field(default_factory=list)


def load_out_log(job: str, results_dir: Path) -> OutLog:
    path = results_dir / f"g16grid_{job}.out"
    if not path.exists():
        return OutLog(path=path, exists=False)
    text = path.read_text(encoding="utf-8", errors="replace")
    return OutLog(path=path, exists=True, lines=text.splitlines())


def find_arm_order_line(out: OutLog) -> Optional[Dict[str, Any]]:
    pat = re.compile(r"^G16_ARM_ORDER block=(\S+) order=\[(.*?)\]\s*$")
    for line in out.lines:
        m = pat.match(line)
        if m:
            block_tag, order_str = m.groups()
            arms = order_str.split() if order_str.strip() else []
            return {"block_tag": block_tag, "arms": arms, "raw": line}
    return None


def find_mode_line(out: OutLog) -> Optional[str]:
    for line in out.lines:
        if line.startswith("G16_MODE="):
            return line
    return None


def find_block_done(out: OutLog) -> Optional[Dict[str, Any]]:
    pat = re.compile(
        r"^G16_BLOCK_DONE block=(\S+) arms_total=(\d+) BOOTS_FAILED=(\d+) "
        r"failed_arms=\[(.*?)\] pin_failed_arms=\[(.*?)\]\s*$"
    )
    for line in out.lines:
        m = pat.match(line)
        if m:
            block_tag, arms_total, boots_failed, failed_arms, pin_failed_arms = m.groups()
            return {
                "raw": line,
                "block_tag": block_tag,
                "arms_total": int(arms_total),
                "boots_failed": int(boots_failed),
                "failed_arms": failed_arms.split() if failed_arms.strip() else [],
                "pin_failed_arms": pin_failed_arms.split() if pin_failed_arms.strip() else [],
            }
    return None


def find_boot_walltimes(out: OutLog) -> Dict[str, Dict[str, Any]]:
    pat = re.compile(
        r"^G16_BOOT_WALLTIME arm=(\S+) block=(\S+) boot=(\d+) seconds=(\S+) "
        r"boot_s=(\S+) bench_s=(\S+)\s*$"
    )
    result: Dict[str, Dict[str, Any]] = {}
    for line in out.lines:
        m = pat.match(line)
        if m:
            arm, block_tag, boot, seconds, boot_s, bench_s = m.groups()
            result[arm] = {
                "raw": line, "block_tag": block_tag, "boot_idx": int(boot),
                "seconds": float(seconds), "boot_s": float(boot_s),
                "bench_s": float(bench_s),
            }
    return result


def find_h15_or_setup_violations(out: OutLog) -> List[str]:
    needles = (
        "H15_VIOLATION", "ARM_ORDER_COUNT_MISMATCH", "BAD_BLOCK_IDX",
        "ARM_ORDER_GEN_FAILED", "SYNC_ENGINE_TREE_FAILED", "BAD_ARM=",
        "MISSING_CONFIG",
    )
    return [line for line in out.lines if any(n in line for n in needles)]


def find_harness_uncommitted(out: OutLog) -> Optional[str]:
    for line in out.lines:
        if line.startswith("G16_HARNESS_UNCOMMITTED="):
            return line
    return None


# ---------------------------------------------------------------------------
# artifact loading (sidecar / controller_summary / cotenancy)
# ---------------------------------------------------------------------------
def runid_for(block_tag: str, arm: str, job: str, boot: int = 1) -> str:
    return f"g16_{block_tag}_{arm}_boot{boot}_{job}"


def load_json_artifact(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"present": False, "path": str(path)}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"present": True, "path": str(path), "parse_error": repr(exc)}
    return {"present": True, "path": str(path), "data": data}


def load_cotenancy_ts(results_dir: Path, block_tag: str, arm: str, job: str,
                       label_suffix: str, boot: int = 1) -> Optional[datetime]:
    path = results_dir / f"g16_{block_tag}_{arm}_boot{boot}_{label_suffix}_cotenancy_{job}.txt"
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"^TIMESTAMP=(\S+)", text, re.MULTILINE)
    if not m:
        return None
    ts = m.group(1)
    try:
        dt = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return None
    return dt.replace(tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# check helpers
# ---------------------------------------------------------------------------
def mk(check_id: int, name: str, status: str, detail: Dict[str, Any]) -> Dict[str, Any]:
    assert status in STATUS_LEVELS, f"invalid status {status!r} for check {check_id}"
    return {"id": check_id, "name": name, "status": status, "detail": detail}


def cross_check_arm_order(block_tag: str, declared_arms: List[str]) -> Dict[str, Any]:
    """Independent second derivation of the arm order (project style: 'a
    different code path over the same quantity -- agreement is evidence, not
    an identity', c.f. sbatch's own SMOKE9a comment). FULL blocks (blk1-4)
    are cross-checked against g16_arm_order.py directly; smoke1 is
    cross-checked against the hardcoded order the sbatch script itself uses
    in its SMOKE branch (mirrored here as SMOKE_ARM_ORDER, since smoke mode
    does not call g16_arm_order.py at all)."""
    m = re.match(r"^blk([1-4])$", block_tag)
    if m:
        idx = m.group(1)
        try:
            proc = subprocess.run(
                ["python3", str(HERE / "g16_arm_order.py"), idx],
                capture_output=True, text=True, timeout=30, check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return {"applicable": True, "match": None,
                    "reason": f"g16_arm_order.py invocation failed: {exc!r}"}
        if proc.returncode != 0:
            return {"applicable": True, "match": None,
                    "reason": f"g16_arm_order.py rc={proc.returncode} stderr={proc.stderr.strip()!r}"}
        independent = proc.stdout.strip().split()
        return {"applicable": True, "match": independent == declared_arms,
                "independent_order": independent, "declared_order": declared_arms}
    if block_tag.startswith("smoke"):
        return {"applicable": True, "match": declared_arms == SMOKE_ARM_ORDER,
                "independent_order": SMOKE_ARM_ORDER, "declared_order": declared_arms}
    return {"applicable": False, "reason": f"unrecognized block_tag {block_tag!r}"}


# ---------------------------------------------------------------------------
# the 5 checks
# ---------------------------------------------------------------------------
def check1_block_done(out: OutLog, job_terminal: bool) -> Dict[str, Any]:
    bd = find_block_done(out)
    if bd is None:
        if job_terminal:
            return mk(1, "block_done", "FAIL", {
                "reason": "job reached a terminal sacct state but no G16_BLOCK_DONE "
                          "line was found in the .out -- the per-arm loop did not "
                          "finish printing this line (job likely died mid-loop; see "
                          "check 2/3 partial data and .err for the actual failure)",
            })
        return mk(1, "block_done", "NA", {
            "reason": "job not in a terminal state yet (still PENDING/RUNNING) -- "
                      "G16_BLOCK_DONE is printed only after the per-arm loop exits",
        })
    status = "OK"
    notes = []
    if bd["boots_failed"] > 0:
        status = "WARN"
        notes.append(f"BOOTS_FAILED={bd['boots_failed']} > 0")
    if bd["failed_arms"]:
        status = "WARN"
        notes.append(f"failed_arms={bd['failed_arms']}")
    if bd["pin_failed_arms"]:
        status = "WARN"
        notes.append(f"pin_failed_arms={bd['pin_failed_arms']} (H3'-a exact!=True -- "
                      "these arms are excluded from S_max/S_min by the next-stage "
                      "analyzer, per addendum A-1/H3'-a; this is a per-arm exclusion, "
                      "NOT a boot failure, so it does not add to BOOTS_FAILED)")
    return mk(1, "block_done", status, {**bd, "notes": notes})


def check2_exact_per_arm(results_dir: Path, block_tag: str, job: str,
                          arms: List[str], failed_arms: List[str],
                          job_terminal: bool) -> Dict[str, Any]:
    if not arms:
        return mk(2, "exact_per_arm", "NA", {
            "reason": "no arm order known yet (G16_ARM_ORDER line not found -- "
                      "job likely has not started/reached that point)"})
    per_arm: Dict[str, Any] = {}
    any_fail = False
    any_warn = False
    any_missing_unexpected = False
    for arm in arms:
        runid = runid_for(block_tag, arm, job)
        sidecar = load_json_artifact(results_dir / f"{runid}_sidecar.json")
        if not sidecar["present"]:
            if arm in failed_arms:
                per_arm[arm] = {"exact": None, "note": "no sidecar -- expected: "
                                 "this arm is in failed_arms (boot/bench failed "
                                 "before H3'-a probe ran)"}
            elif job_terminal:
                per_arm[arm] = {"exact": None, "note": "MISSING sidecar, and this "
                                 "arm is NOT in failed_arms, and job is terminal -- "
                                 "unexplained gap"}
                any_missing_unexpected = True
                any_fail = True
            else:
                per_arm[arm] = {"exact": None, "note": "no sidecar yet (job still "
                                 "running, arm not reached / in progress)"}
            continue
        if "parse_error" in sidecar:
            per_arm[arm] = {"exact": None, "note": f"sidecar JSON parse error: "
                             f"{sidecar['parse_error']}"}
            any_fail = True
            continue
        data = sidecar["data"]
        exact = data.get("exact")
        realized_a = data.get("realized_a")
        realized_b = data.get("realized_b")
        status_field = data.get("status")
        rec = {"exact": exact, "realized_a": realized_a, "realized_b": realized_b,
               "sidecar_status": status_field}
        if exact is not True:
            any_warn = True
            rec["note"] = ("exact is not True -- H3'-a excludes this arm from "
                            "S_max/S_min (arm-unit deactivation, not a boot "
                            "failure); analyzer must redefine over surviving arms")
        per_arm[arm] = rec
    status = "OK"
    if any_fail:
        status = "FAIL"
    elif any_warn:
        status = "WARN"
    return mk(2, "exact_per_arm", status, {
        "block_tag": block_tag, "arms": arms, "per_arm": per_arm,
        "any_missing_unexpected": any_missing_unexpected,
    })


def check3_wallclock(results_dir: Path, block_tag: str, job: str, arms: List[str],
                      boot_walltimes: Dict[str, Dict[str, Any]],
                      sacct: Dict[str, Any]) -> Dict[str, Any]:
    detail: Dict[str, Any] = {"per_arm_from_log": boot_walltimes}
    if not sacct.get("ok"):
        return mk(3, "wallclock", "NA" if not arms else "WARN", {
            **detail, "reason": f"sacct unavailable/unparseable: {sacct.get('reason')}"})
    total_elapsed_s = elapsed_to_seconds(sacct.get("elapsed_str", ""))
    state = sacct.get("state", "")
    detail.update(sacct_state=state, sacct_elapsed_str=sacct.get("elapsed_str"),
                   total_elapsed_s=total_elapsed_s)
    if state not in TERMINAL_STATES:
        # partial report only -- do not attempt full decomposition on a
        # still-running job (residual would be meaningless mid-run).
        sum_arm_s = sum(v["seconds"] for v in boot_walltimes.values())
        detail.update(sum_arm_seconds_so_far=sum_arm_s,
                       arms_with_walltime_so_far=sorted(boot_walltimes),
                       reason=f"job state={state!r} not terminal -- decomposition "
                              "vs total elapsed deferred; showing per-arm-so-far only")
        return mk(3, "wallclock", "NA", detail)

    sum_arm_s = sum(v["seconds"] for v in boot_walltimes.values())
    detail["sum_arm_seconds"] = sum_arm_s

    # Cross-check against sidecar-derived t_boot0/t_healthy/t_boot1 diffs
    # (second independent computation of the same boot_s/bench_s, project
    # style -- see module docstring).
    mismatches = []
    for arm in arms:
        runid = runid_for(block_tag, arm, job)
        sidecar = load_json_artifact(results_dir / f"{runid}_sidecar.json")
        if not sidecar.get("present") or "parse_error" in sidecar:
            continue
        d = sidecar["data"]
        t0, th, t1 = d.get("t_boot0"), d.get("t_healthy"), d.get("t_boot1")
        if None in (t0, th, t1):
            continue
        boot_s_sc = th - t0
        bench_s_sc = t1 - th
        logged = boot_walltimes.get(arm)
        if logged is None:
            continue
        if abs(boot_s_sc - logged["boot_s"]) > 0.5 or abs(bench_s_sc - logged["bench_s"]) > 0.5:
            mismatches.append({"arm": arm, "log_boot_s": logged["boot_s"],
                                "sidecar_boot_s": boot_s_sc,
                                "log_bench_s": logged["bench_s"],
                                "sidecar_bench_s": bench_s_sc})
    detail["log_vs_sidecar_mismatches"] = mismatches

    # Pre-first-arm setup overhead and post-last-arm teardown overhead, from
    # cotenancy-snapshot timestamps (the only wall-clock timestamps this
    # harness writes besides sacct itself; see module docstring / cotenancy
    # timestamp discovery). Best-effort only -- NA per-field if the relevant
    # cotenancy file or sacct Start/End is missing.
    pre_overhead_s = None
    post_overhead_s = None
    sacct_start = parse_sacct_local_dt(sacct.get("start_str", ""))
    sacct_end = parse_sacct_local_dt(sacct.get("end_str", ""))
    if arms:
        first_arm, last_arm = arms[0], arms[-1]
        first_pre_ts = load_cotenancy_ts(results_dir, block_tag, first_arm, job, "pre")
        last_post_ts = load_cotenancy_ts(results_dir, block_tag, last_arm, job, "post")
        if sacct_start and first_pre_ts:
            pre_overhead_s = (first_pre_ts - sacct_start).total_seconds()
        if sacct_end and last_post_ts:
            post_overhead_s = (sacct_end - last_post_ts).total_seconds()
    detail["pre_first_arm_setup_overhead_s"] = pre_overhead_s
    detail["post_last_arm_teardown_overhead_s"] = post_overhead_s
    detail["pre_overhead_note"] = (
        "covers: shell/module-load prologue + sync_engine_tree.sh + manifest "
        "hashing + H18 warm-up boot+bench+kill -- NOT further decomposed (the "
        "harness does not timestamp the warm-up phase itself, only brackets it "
        "between sacct Start and arm-1's pre-cotenancy snapshot)")

    residual_s = None
    if total_elapsed_s is not None:
        residual_s = total_elapsed_s - sum_arm_s
        if pre_overhead_s is not None:
            residual_s -= pre_overhead_s
        if post_overhead_s is not None:
            residual_s -= post_overhead_s
    detail["unattributed_residual_s"] = residual_s
    detail["unattributed_residual_note"] = (
        "= total_elapsed - sum(arm seconds) - pre_setup - post_teardown; "
        "expected to be small and positive (inter-arm kill+`sleep 5`+cotenancy "
        "snapshot calls, (N_arms-1) transitions) -- large or negative values "
        "are a parsing/accounting FAIL, not a performance signal")

    n_arms_declared = len(arms)
    n_arms_with_walltime = len(boot_walltimes)
    detail["n_arms_declared"] = n_arms_declared
    detail["n_arms_with_walltime"] = n_arms_with_walltime

    status = "OK"
    if mismatches:
        status = "WARN"
    if residual_s is not None and (residual_s < -1.0 or residual_s > 300.0):
        status = "WARN"
    if n_arms_with_walltime < n_arms_declared:
        # some arms never printed a walltime line (boot/bench failed before
        # reaching that point) -- expected/covered by check 1, not itself a
        # FAIL here, but must be visible.
        status = "WARN" if status == "OK" else status
        detail["arms_missing_walltime"] = sorted(set(arms) - set(boot_walltimes))
    if total_elapsed_s is None:
        status = "FAIL"
        detail["fail_reason"] = "job terminal but sacct Elapsed unparseable"
    return mk(3, "wallclock", status, detail)


def check4_residency_fraction(results_dir: Path, block_tag: str, job: str,
                               arms: List[str], failed_arms: List[str],
                               job_terminal: bool) -> Dict[str, Any]:
    if not arms:
        return mk(4, "residency_fraction_all_arms", "NA", {
            "reason": "no arm order known yet"})
    per_arm: Dict[str, Any] = {}
    n_ok = 0
    n_expected = 0  # arms that got far enough (bench succeeded) to compute this
    any_fail = False
    for arm in arms:
        runid = runid_for(block_tag, arm, job)
        if arm in failed_arms:
            per_arm[arm] = {"status": "N_A_upstream_failure",
                             "note": "arm is in failed_arms -- boot/bench failed "
                                     "before controller_summary is computed"}
            continue
        n_expected += 1
        cs = load_json_artifact(results_dir / f"{runid}_controller_summary.json")
        if not cs["present"]:
            per_arm[arm] = {"status": "MISSING",
                             "note": "controller_summary.json absent (this is "
                                     "EXACTLY the N3-bug signature from smoke "
                                     "884292 -- PDMUX_EVAL_DIR unbound var killing "
                                     "the heredoc before python started; commit "
                                     "37cf6b8 is supposed to have fixed this)"}
            any_fail = True
            continue
        if "parse_error" in cs:
            per_arm[arm] = {"status": "PARSE_ERROR", "note": cs["parse_error"]}
            any_fail = True
            continue
        data = cs["data"]
        frac = data.get("residency_fraction")
        if not isinstance(frac, dict) or not frac:
            per_arm[arm] = {"status": "EMPTY_OR_ABSENT", "residency_fraction": frac,
                             "note": "residency_fraction key absent or {} -- "
                                     "regression of the exact defect this check "
                                     "point exists to catch"}
            any_fail = True
            continue
        total = sum(float(v) for v in frac.values())
        rec = {"status": "OK" if total > 0.0 else "DEGENERATE_ZERO",
               "residency_fraction": frac, "sum": total}
        if total <= 0.0:
            any_fail = True
        else:
            n_ok += 1
        per_arm[arm] = rec
    status = "OK"
    if any_fail:
        status = "FAIL"
    elif n_expected == 0:
        status = "NA"
    detail = {"block_tag": block_tag, "arms": arms, "per_arm": per_arm,
              "n_arms_ok": n_ok, "n_arms_expected": n_expected,
              "all_arms_populated": (n_ok == n_expected and n_expected > 0)}
    return mk(4, "residency_fraction_all_arms", status, detail)


def check5_sidecar_adoption(results_dir: Path, block_tag: str, job: str,
                             arms: List[str], failed_arms: List[str]) -> Dict[str, Any]:
    """Mirrors g16_analyze.py's load_campaign_grid adoption predicate
    EXACTLY (status=='completed' and telem_rc==0 and fields_rc.lo==0 and
    fields_rc.hi==0), read-only, computes no estimand from the adopted
    records -- this check only reports WHICH boots would be adopted/rejected
    by that predicate and WHY, so a downstream >=3-block analysis run is not
    the first time anyone looks at this."""
    if not arms:
        return mk(5, "sidecar_adoption_rule", "NA", {"reason": "no arm order known yet"})
    per_arm: Dict[str, Any] = {}
    n_adopted = 0
    n_considered = 0
    any_fail = False
    for arm in arms:
        runid = runid_for(block_tag, arm, job)
        sidecar = load_json_artifact(results_dir / f"{runid}_sidecar.json")
        if not sidecar["present"]:
            if arm in failed_arms:
                per_arm[arm] = {"adopted": False, "reason": "no sidecar "
                                 "(boot/bench failed upstream, expected)"}
            else:
                per_arm[arm] = {"adopted": False, "reason": "no sidecar and arm "
                                 "not in failed_arms -- unexplained"}
                any_fail = True
            continue
        if "parse_error" in sidecar:
            per_arm[arm] = {"adopted": False, "reason": f"sidecar parse error: "
                             f"{sidecar['parse_error']}"}
            any_fail = True
            continue
        n_considered += 1
        d = sidecar["data"]
        status_field = d.get("status")
        telem_rc = d.get("telem_rc")
        fields_rc = d.get("fields_rc") or {}
        lo_rc = fields_rc.get("lo")
        hi_rc = fields_rc.get("hi")
        adopted = (status_field == "completed" and telem_rc == 0
                   and lo_rc == 0 and hi_rc == 0)
        reasons = []
        if status_field != "completed":
            reasons.append(f"status={status_field!r} != 'completed'")
        if telem_rc != 0:
            reasons.append(f"telem_rc={telem_rc!r} != 0")
        if lo_rc != 0:
            reasons.append(f"fields_rc.lo={lo_rc!r} != 0")
        if hi_rc != 0:
            reasons.append(f"fields_rc.hi={hi_rc!r} != 0")
        per_arm[arm] = {"adopted": adopted, "status": status_field,
                         "telem_rc": telem_rc, "fields_rc": fields_rc,
                         "reject_reasons": reasons}
        if adopted:
            n_adopted += 1
    status = "OK"
    if any_fail:
        status = "FAIL"
    elif n_considered > 0 and n_adopted < n_considered:
        status = "WARN"
    elif n_considered == 0:
        status = "NA"
    return mk(5, "sidecar_adoption_rule", status, {
        "block_tag": block_tag, "arms": arms, "per_arm": per_arm,
        "n_adopted": n_adopted, "n_considered": n_considered,
    })


# ---------------------------------------------------------------------------
# aggregation (gate #42 avoidance: build the list explicitly, assert shape,
# THEN aggregate over it -- never a hand-written AND of named booleans that
# can silently drop one)
# ---------------------------------------------------------------------------
def aggregate(checks: List[Dict[str, Any]]) -> Dict[str, Any]:
    assert len(checks) == 5, f"expected exactly 5 checks, got {len(checks)}"
    for i, c in enumerate(checks, start=1):
        assert c["id"] == i, f"check ordering broken: position {i} has id {c['id']}"
        assert c["status"] in STATUS_LEVELS, f"check {i} has invalid status {c['status']!r}"
    by_status: Dict[str, List[int]] = {s: [] for s in STATUS_LEVELS}
    for c in checks:
        by_status[c["status"]].append(c["id"])
    if by_status["FAIL"]:
        overall = "FAIL"
    elif not any(c["status"] != "NA" for c in checks):
        overall = "NA"
    elif by_status["WARN"]:
        overall = "WARN"
    else:
        overall = "OK"
    return {"overall": overall, "by_status": by_status,
            "n_checks": len(checks),
            "all_ids_referenced": sorted(sum(by_status.values(), []))}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def run(job: str, results_dir: Path) -> Dict[str, Any]:
    sacct = sacct_info(job)
    job_state = sacct.get("state", "UNKNOWN") if sacct.get("ok") else "UNKNOWN"
    job_terminal = job_state in TERMINAL_STATES

    out = load_out_log(job, results_dir)
    mode_line = find_mode_line(out) if out.exists else None
    arm_order_info = find_arm_order_line(out) if out.exists else None
    block_done = find_block_done(out) if out.exists else None
    boot_walltimes = find_boot_walltimes(out) if out.exists else {}
    setup_violations = find_h15_or_setup_violations(out) if out.exists else []
    harness_uncommitted = find_harness_uncommitted(out) if out.exists else None

    block_tag = (arm_order_info or {}).get("block_tag") or (block_done or {}).get("block_tag")
    arms = (arm_order_info or {}).get("arms") or []
    failed_arms = (block_done or {}).get("failed_arms") or []

    arm_order_cross_check = None
    if block_tag and arms:
        arm_order_cross_check = cross_check_arm_order(block_tag, arms)

    checks = [
        check1_block_done(out, job_terminal),
        check2_exact_per_arm(results_dir, block_tag or "", job, arms, failed_arms, job_terminal),
        check3_wallclock(results_dir, block_tag or "", job, arms, boot_walltimes, sacct),
        check4_residency_fraction(results_dir, block_tag or "", job, arms, failed_arms, job_terminal),
        check5_sidecar_adoption(results_dir, block_tag or "", job, arms, failed_arms),
    ]
    verdict = aggregate(checks)

    return {
        "job": job,
        "results_dir": str(results_dir),
        "out_log_path": str(out.path),
        "out_log_exists": out.exists,
        "sacct": sacct,
        "job_state": job_state,
        "job_terminal": job_terminal,
        "mode_line": mode_line,
        "block_tag": block_tag,
        "declared_arms": arms,
        "arm_order_cross_check": arm_order_cross_check,
        "setup_violations": setup_violations,
        "harness_uncommitted_line": harness_uncommitted,
        "checks": checks,
        "verdict": verdict,
        "note": "NO decision quantity (donor/Delta/Delta_SLO/S_itl) is computed "
                "by this script -- structural artifact-integrity checks only. "
                "See module docstring.",
    }


def print_report(result: Dict[str, Any]) -> None:
    print("=" * 78)
    print(f"G16 block inspection -- job={result['job']}")
    print("=" * 78)
    print(f"out_log: {result['out_log_path']} (exists={result['out_log_exists']})")
    sacct = result["sacct"]
    if sacct.get("ok"):
        print(f"sacct: state={sacct['state']} elapsed={sacct['elapsed_str']} "
              f"start={sacct['start_str']} end={sacct['end_str']}")
    else:
        print(f"sacct: UNAVAILABLE ({sacct.get('reason')})")
    print(f"job_terminal={result['job_terminal']}")
    print(f"mode_line: {result['mode_line']}")
    print(f"block_tag={result['block_tag']!r} declared_arms={result['declared_arms']}")
    xc = result["arm_order_cross_check"]
    if xc:
        print(f"arm_order_cross_check: applicable={xc.get('applicable')} "
              f"match={xc.get('match')}"
              + (f" independent={xc.get('independent_order')}" if xc.get("match") is False else ""))
    if result["setup_violations"]:
        print("SETUP_VIOLATIONS (harness self-reported, pre-arm-loop):")
        for v in result["setup_violations"]:
            print(f"  ! {v}")
    if result["harness_uncommitted_line"]:
        print(f"harness_uncommitted: {result['harness_uncommitted_line']}")
    print("-" * 78)
    for c in result["checks"]:
        print(f"[{c['id']}] {c['name']}: {c['status']}")
        detail = c["detail"]
        for k, v in detail.items():
            if k in ("per_arm",):
                print(f"      {k}:")
                for arm, rec in v.items():
                    print(f"        {arm}: {rec}")
            else:
                print(f"      {k}: {v}")
    print("-" * 78)
    v = result["verdict"]
    print(f"INSPECT_OVERALL status={v['overall']} "
          f"fail={v['by_status']['FAIL']} warn={v['by_status']['WARN']} "
          f"na={v['by_status']['NA']} ok={v['by_status']['OK']} "
          f"(n_checks={v['n_checks']}, all_ids_referenced={v['all_ids_referenced']})")
    print(result["note"])
    print("=" * 78)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--job", required=True, help="SLURM job id, e.g. 884336")
    ap.add_argument("--dir", default=str(HERE), help="results directory "
                     "(default: this script's own directory)")
    ap.add_argument("--json", action="store_true", help="also dump full JSON result")
    args = ap.parse_args()

    results_dir = Path(args.dir)
    if not results_dir.is_dir():
        die(f"results dir does not exist: {results_dir}")

    result = run(args.job, results_dir)
    print_report(result)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True, default=str))

    overall = result["verdict"]["overall"]
    return {"OK": 0, "WARN": 0, "NA": 2, "FAIL": 1}[overall]


if __name__ == "__main__":
    sys.exit(main())
