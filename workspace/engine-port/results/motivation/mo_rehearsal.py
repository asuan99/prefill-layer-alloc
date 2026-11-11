#!/usr/bin/env python3
"""Full-pass rehearsal and the launch gate of the motivation harness.  CPU-testable; on a GPU it is the first thing a Job runs.

rehearse(): the REAL arm list (every arm of mo_spec.ARM_PARAMS, real argv with rehearsal-only mem/cap), ONE rung, few requests, one seed,
through the same runner (card check included) -> the analyzer -> the verdict table.  Success = every stage reached; a record is written.
launch_gate(plan): a real (non-rehearsal) plan may launch only if a PASSED rehearsal record exists for the current harness digest and for the
plan's arm list.

"Rehearsal record for its commit": a record cannot contain the hash of the commit that stores it, so the record binds to the HARNESS DIGEST
(sha256 over every code file the Job executes: mo_*.py, the imported sb_*/ph_* harness files, engine_tree.py).  The record also stores the
git commit it was produced at (informational).  Any change to those files invalidates the record.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import mo_spec as MS
import mo_card as MC
import mo_runner as MR
import mo_analyze as MA

HERE = Path(os.path.abspath(__file__)).parent
SP = MS.SP
REHEARSAL_DIR = HERE / "rehearsals"
REHEARSAL_N_REQUESTS = 8
STAGES = ("plan", "card", "boots", "card_checks", "runs", "schema", "analyzer", "verdict_table")


def _harness_files():
    files = sorted(HERE.glob("mo_*.py"))
    files += [SP.HERE / n for n in ("sb_spec.py", "sb_runner.py", "sb_loadgen.py", "stock_metrics.py")]
    files += sorted(SP.PH_DIR.glob("ph_*.py"))
    files += [SP.ENGINE_TREE_PY, SP.ENG / "benchmarks" / "pdmux_eval" / "trace_loadgen.py", SP.ENG / "benchmarks" / "pdmux_eval" / "abs_mono_loadgen.py"]
    files += [HERE / "workloads" / "gen_traces.py", HERE / "workloads" / "mw_lib.py", SP.ENG / "scripts" / "multigpu" / "mg_x_pretokenize.py"]
    return [f for f in files if f.exists()]


def harness_digest(files=None):
    h = hashlib.sha256()
    for f in (files or _harness_files()):
        h.update(str(Path(f).name).encode() + b"\0" + hashlib.sha256(Path(f).read_bytes()).digest())
    return h.hexdigest()


def current_commit():
    c = os.environ.get("PDMUX_COMMIT")
    if c:
        return c
    try:
        return subprocess.run(["git", "-C", str(SP.ENG), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=15).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def rehearsal_plan(job, rmin=3.0, factor=1.15, n_requests=REHEARSAL_N_REQUESTS, seeds=None, workload="W1"):
    seeds = seeds or (MS.default_seeds(workload)[0],)
    return MS.make_plan(job, MS.make_grid(rmin, factor, 1), seeds=seeds, n_requests=n_requests, rehearsal=True, workload=workload)


def resolve_workloads(w1_pool, token_dir=None, n=4):
    """CPU check of every registered workload: trace builds, token file present with the registered sha256, ids match lengths.
    -> ({workload: "OK" | reason}).  The full-arm pass covers one workload; this covers the trace and token path of the others."""
    import mo_workload as MW
    out = {}
    for wl in MW.WORKLOADS:
        try:
            run = {"run_id": "resolve", "workload": wl, "rate": 1.0, "seed": MW.SEEDS[0], "n_requests": n}
            MW.with_input_ids(MW.build_segments(run, w1_pool), MW.store_for(wl, token_dir))
            out[wl] = "OK"
        except Exception as e:  # noqa: BLE001
            out[wl] = f"{type(e).__name__}: {e}"
    return out


def rehearse(job, w1_pool, out, make_server, replay_fn, rmin=3.0, rehearsal_dir=None, digest=None, workload="W1", **kw):
    """Run the whole pass; return (and persist) the record.  Never raises on a stage failure: the record says which stage failed."""
    out = Path(out)
    rec = {"schema": "motivation.rehearsal/v1", "job": job, "rehearsal_commit": current_commit(), "harness_digest": digest or harness_digest(),
           "stages": {s: False for s in STAGES}, "passed": False, "reason": None, "verdict_rows": 0, "workload": workload}
    rec["workloads_resolved"] = resolve_workloads(w1_pool, kw.pop("token_dir", None))
    try:
        plan = rehearsal_plan(job, rmin, workload=workload)
        bad = MS.plan_problems(plan)
        if bad:
            raise RuntimeError(f"plan: {bad}")
        rec["stages"]["plan"] = True
        rec["arm_names"] = sorted({b["arm"] for b in plan["boots"]})
        rec["plan_sha"] = MS.sha(plan)
        card = MC.build_card(plan, w1_pool)
        rec["card_sha"] = card["card_sha"]
        rec["stages"]["card"] = True
        res = MR.run_job(plan, card, out, make_server, replay_fn, w1_pool, warmup_s=kw.pop("warmup_s", 4.0), **kw)
        rec["stages"]["boots"] = all(b["status"] == "done" for b in res)
        rec["stages"]["card_checks"] = rec["stages"]["boots"]            # a mismatch aborts the Job before this line is reached
        rec["stages"]["runs"] = rec["stages"]["boots"]
        if not rec["stages"]["boots"]:
            raise RuntimeError("boot failures: " + ",".join(f"{b['boot_id']}={b['status']}" for b in res if b["status"] != "done"))
        job_data = MA.load_job(out)
        from mo_schema import validate_run
        probs = [m for r in job_data["runs"] for m in validate_run(r)]
        if probs:
            raise RuntimeError(f"schema: {probs[:3]}")
        rec["stages"]["schema"] = True
        result = MA.analyze_job(job_data)
        rec["stages"]["analyzer"] = True
        MA.write_outputs(result, out / "analysis")
        rec["verdict_rows"] = len(result["verdict_rows"])
        rec["stages"]["verdict_table"] = rec["verdict_rows"] > 0
        rec["verdict_table_sha"] = MS.sha(result["verdict_rows"])
    except MR.JobAbort as e:
        rec["reason"] = f"{e.status}: {e.reason}"
    except Exception as e:  # noqa: BLE001
        rec["reason"] = f"{type(e).__name__}: {e}"
    rec["passed"] = all(rec["stages"].values())
    d = Path(rehearsal_dir or REHEARSAL_DIR)
    d.mkdir(parents=True, exist_ok=True)
    MR.dump(rec, out / "REHEARSAL.json")
    MR.dump(rec, d / f"{rec['harness_digest'][:16]}.json")
    return rec


def launch_gate(plan, rehearsal_dir=None, digest=None):
    """-> None (launch allowed) or the refusal reason."""
    digest = digest or harness_digest()
    d = Path(rehearsal_dir or REHEARSAL_DIR)
    f = d / f"{digest[:16]}.json"
    if not f.exists():
        return f"NO_REHEARSAL_RECORD for harness digest {digest[:16]} (commit {current_commit()}): run `mo_runner.py rehearse` first"
    rec = json.loads(f.read_text())
    if rec.get("harness_digest") != digest:
        return "REHEARSAL_RECORD_DIGEST_MISMATCH"
    if not rec.get("passed") or not all(rec.get("stages", {}).get(s) for s in STAGES):
        return f"REHEARSAL_NOT_PASSED: {rec.get('reason')}"
    want = sorted({b["arm"] for b in plan["boots"]})
    if rec.get("arm_names") != want:
        return f"REHEARSAL_ARM_LIST_DIFFERS: rehearsed {len(rec.get('arm_names', []))} arms, plan has {len(want)}"
    wl = plan.get("workload", "W1")
    if wl != "W1" and (rec.get("workloads_resolved") or {}).get(wl) != "OK":
        return f"REHEARSAL_WORKLOAD_NOT_RESOLVED: {wl}: {(rec.get('workloads_resolved') or {}).get(wl, 'not checked')}"
    return None
