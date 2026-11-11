#!/usr/bin/env python3
"""Job runner of the motivation harness (2026-11-10).  Reuses sb_runner (server env, warm-up helpers, segments, log parsing) by import.

  mo_runner.py plan   --job J --rmin R --factor F --n N [--role base|extension] [--rehearsal]
  mo_runner.py card   --job J ... --w1-lengths L.json --out card.json
  mo_runner.py run    --job J ... --w1-lengths L.json --card card.json --out RAW [--resume]
  mo_runner.py rehearse --job J --w1-lengths L.json --out DIR          (real arm list, one rung, few requests, raw -> analyzer -> verdict table)
  mo_runner.py gate   --job J ...                                       (exit 0 iff a rehearsal record exists for this harness digest)

Per boot: server start (SGLang arms through `engine_tree.py boot` with the ARM's interpreter, e.g. /opt/sg312-venv/bin/python; vLLM through its
own venv, unchanged) -> health -> CARD CHECK (argv, banner, stamp, interpreter, order, pool + traces) -> any mismatch stops the server, writes
CARD_MISMATCH.json and ABORTS THE JOB (exit 19) -> warm-up (discarded) -> runs -> stop.  A boot that fails to come up is recorded and the Job
continues (exit 17), as in the stock-baseline runner.  Not run on a GPU: everything GPU-side is unverified.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import mo_spec as MS
import mo_card as MC
import mo_schema as SC
import mo_loadgen as ML
import mo_realized as MRZ
import mo_workload as MW
import sb_runner as RU
import ph_trace as T

SP = MS.SP
WARMUP_S = 60.0
EXIT_BOOT_FAIL, EXIT_ABORT = 17, 19


class JobAbort(RuntimeError):
    """Raised on a card mismatch or a launch-gate refusal: the whole Job stops (no further boot)."""

    def __init__(self, status, reason, boot_id=None):
        super().__init__(f"{status}: {reason}")
        self.status, self.reason, self.boot_id = status, reason, boot_id


dump = RU.dump


def server_cmd(a, bdir, port):
    """SGLang: engine_tree.py boot --stamp <json> -- <arm interpreter> -m sglang.launch_server ...; vLLM: its own command."""
    if a["engine"] == "sglang":
        return [sys.executable, str(SP.ENGINE_TREE_PY), "boot", "--stamp", str(Path(bdir) / f"tree_{a['name']}.json"), "--",
                a["python"]] + list(a["cmd"]) + ["--port", str(port)]
    return list(a["cmd"]) + ["--port", str(port)]


class MoProcessServer(RU.ProcessServer):
    def start(self):
        self.port = RU.free_port()
        self.cmd = server_cmd(self.a, self.bdir, self.port)
        self.lf = open(self.log, "wb")
        self.proc = subprocess.Popen(self.cmd, stdout=self.lf, stderr=subprocess.STDOUT, env=RU.server_env(self.a, self.bdir),
                                     start_new_session=True)
        t0 = time.monotonic()
        while time.monotonic() - t0 < self.boot_timeout:
            if self.proc.poll() is not None:
                return "died"
            if RU._health(self.port):
                return "ok"
            time.sleep(2)
        return "timeout"


def stamp():
    return {**RU.stamp(), "hostname": socket.gethostname()}


def warmup(srv, a, replay_fn, w1_pool, bdir, boot_id, duration=WARMUP_S):
    ts = T.arrivals(2.0, ("warm", boot_id), duration=duration)
    reqs = T.make_requests("W1", w1_pool, ts, ("warm", boot_id), f"{boot_id}_warm", "WARM")
    dur, recs = replay_fn(a["api"], srv.base_url, reqs, Path(bdir) / "warmup" / "warm.jsonl")
    bad = [q["request_id"] for q in recs if q.get("error") or q.get("observed_output_tokens") != q.get("requested_output_tokens")]
    dump({"n": len(recs), "bad": bad[:20], "n_bad": len(bad)}, Path(bdir) / "correctness.json")
    return (not bad), f"{len(bad)}/{len(recs)} warm-up requests wrong length or errored"


def replay_segment(replay_fn, api, base_url, seg, out_dir):
    e0, m0 = time.time(), time.monotonic()
    dur, recs = replay_fn(api, base_url, seg["requests"], Path(out_dir) / f"{seg['segment_id']}.jsonl")
    e1, m1 = time.time(), time.monotonic()
    return {"segment_id": seg["segment_id"], "phase": seg["phase"], "duration_s": float(dur), "t_load_start_epoch": e0,
            "t_load_end_epoch": e1, "t_load_start_mono": m0, "t_load_end_mono": m1, "requests": recs}


def split_check_realized(srv, b, rec):
    """Default realized_fn: the split-check boot reads its own telemetry + green read-out; every other boot -> None (unavailable)."""
    if b["role"] != "split_check":
        return None
    return MRZ.realized(srv.bdir, rec, b["arm_def"]["D"])


def execute_run(run, b, srv, replay_fn, w1_pool, out, card, card_check, tree_stamp, server_args, realized_fn=split_check_realized,
                token_store=None):
    a = b["arm_def"]
    segs = MW.build_segments(run, w1_pool)
    rec = {"schema_version": MS.SCHEMA_VERSION, "run_id": run["run_id"], "kind": run["kind"], "cell": run.get("cell"), "seed": run["seed"],
           "arm": b["arm"], "boot_id": b["boot_id"], "boot_role": b["role"], "trace_hash": T.trace_hash(segs),
           "spec": {k: v for k, v in run.items() if k != "run_id"}, "segments": [], **stamp(), "engine": a["engine"], "engine_tree": a["tree"],
           "card_sha": card["card_sha"], "card_check": card_check, "tree_stamp": tree_stamp, "server_args": server_args}
    for s in MW.with_input_ids(segs, token_store):                 # token ids only on the wire; the hash above is of the length records
        rec["segments"].append(replay_segment(replay_fn, a["api"], srv.base_url, s, Path(out) / "segments" / run["run_id"]))
    rec["t_load_start_epoch"], rec["t_load_end_epoch"] = rec["segments"][0]["t_load_start_epoch"], rec["segments"][-1]["t_load_end_epoch"]
    rec["t_load_start_mono"], rec["t_load_end_mono"] = rec["segments"][0]["t_load_start_mono"], rec["segments"][-1]["t_load_end_mono"]
    rs = realized_fn(srv, b, rec) if realized_fn else None
    rec["realized_split"] = rs or {"available": False, "target_D": a.get("D"),
                                   "reason": ("measurement boot carries no phase telemetry (the instrumentation exists only in the patched tree "
                                              "and is forbidden in measurement boots)" if a["tree"] == "patched" else
                                              "engine has no SM split (pristine SGLang / vLLM)")}
    return rec


def _tree_stamp(srv, a, bdir, log_text):
    tp = Path(bdir) / f"tree_{a['name']}.json"
    if a["engine"] == "vllm":
        dump(RU.vllm_stamp(a, {}, log_text), tp)
    return json.loads(tp.read_text()) if tp.exists() else {}


def run_boot(b, card, out, make_server, replay_fn, w1_pool, ctx, py_version_fn=MC.probe_python_version, freeze_fn=MC.freeze_sha,
             realized_fn=split_check_realized, warmup_s=WARMUP_S):
    bdir = Path(out) / "boots" / b["boot_id"]
    bdir.mkdir(parents=True, exist_ok=True)
    a = RU.resolve(b["arm_def"], bdir)
    ctx.setdefault("executed", []).append(b["arm"])
    srv = make_server(bdir, a)
    status = srv.start()
    log_text = RU._log_tail(srv, 0)
    tree_stamp = _tree_stamp(srv, a, bdir, log_text)
    observed = MC.observe_boot(a, getattr(srv, "cmd", None), tree_stamp, log_text, py_version_fn, freeze_fn, bdir)
    dump({"boot_id": b["boot_id"], "arm": b["arm"], "role": b["role"], "status": status, "index": len(ctx["executed"]) - 1, "cmd": getattr(srv, "cmd", None),
          "observed": observed, "tree_stamp": tree_stamp, **stamp()}, bdir / "boot.json")
    if status != "ok":
        srv.stop()
        dump({"boot_id": b["boot_id"], "status": status}, bdir / "DONE.json")
        return {"boot_id": b["boot_id"], "status": status, "runs": []}
    bad = MC.check_boot(card, b, observed, ctx["executed"], w1_pool)          # right after the boot, before anything is measured
    if bad:
        reason = MC.format_mismatches(b["boot_id"], b["arm"], bad)
        dump({"boot_id": b["boot_id"], "arm": b["arm"], "mismatches": bad, "reason": reason, "observed": observed}, bdir / "CARD_MISMATCH.json")
        dump({"boot_id": b["boot_id"], "status": "ABORT_CARD_MISMATCH", "reason": reason, "server": srv.stop()}, bdir / "DONE.json")
        raise JobAbort("ABORT_CARD_MISMATCH", reason, b["boot_id"])
    card_check = {"mismatches": [], "observed": observed, "card_sha": card["card_sha"]}
    done = []
    try:
        ok, why = warmup(srv, a, replay_fn, w1_pool, bdir, b["boot_id"], warmup_s)
        if not ok:
            dump({"boot_id": b["boot_id"], "status": "INVALID_CORRECTNESS", "why": why}, bdir / "DONE.json")
            return {"boot_id": b["boot_id"], "status": "INVALID_CORRECTNESS", "runs": []}
        for run in b["runs"]:
            f = Path(out) / "runs" / f"{run['run_id']}.json"
            if not f.exists():
                dump(execute_run(run, b, srv, replay_fn, w1_pool, out, card, card_check, tree_stamp,
                                 RU.parse_sglang_log(log_text) if a["engine"] == "sglang" else None, realized_fn, ctx.get("token_store")), f)
            done.append(run["run_id"])
    finally:
        info = srv.stop()
    dump({"boot_id": b["boot_id"], "status": "done", "runs": done, "server": info, **stamp()}, bdir / "DONE.json")
    return {"boot_id": b["boot_id"], "status": "done", "runs": done}


def run_job(plan, card, out, make_server, replay_fn, w1_pool, resume=False, gate_fn=None, py_version_fn=MC.probe_python_version,
            freeze_fn=MC.freeze_sha, realized_fn=split_check_realized, warmup_s=WARMUP_S):
    """-> list of boot results.  Raises JobAbort (after writing JOB_ABORTED.json) on a card mismatch / refusal."""
    out = Path(out)
    try:
        problems = MS.plan_problems(plan)
        if problems:
            raise JobAbort("ABORT_PLAN", str(problems))
        if not MC.card_sha_ok(card) or card["plan_sha"] != MS.sha(plan):
            raise JobAbort("ABORT_CARD_PLAN", "card sha / plan sha mismatch: the card was not built from this plan")
        if bool(card["rehearsal"]) != bool(plan["rehearsal"]):
            raise JobAbort("ABORT_CARD_PLAN", "rehearsal flag differs between card and plan")
        if not plan["rehearsal"]:
            ph = MC.card_placeholders(card)
            if ph:
                raise JobAbort("ABORT_CARD_PLACEHOLDER", f"unregistered values in the card: {ph[:6]}")
            why = (gate_fn or _default_gate)(plan)
            if why:
                raise JobAbort("ABORT_NO_REHEARSAL", why)
        if out.exists() and any((out / "runs").glob("*.json")) and not resume:
            raise JobAbort("ABORT_RAW_NOT_EMPTY", "raw directory has runs; set --resume")
        try:                                           # token file present and sha256 = registered, before the first boot
            store = MW.store_for(plan.get("workload", "W1"))
        except (FileNotFoundError, ValueError) as e:
            raise JobAbort("ABORT_TOKENS", str(e))
        dump(plan, out / "plan.json")
        dump(card, out / "card.json")
        ctx, res = {"token_store": store}, []
        for b in plan["boots"]:
            res.append(run_boot(b, card, out, make_server, replay_fn, w1_pool, ctx, py_version_fn, freeze_fn, realized_fn, warmup_s))
    except JobAbort as e:
        dump({"job": plan.get("job"), "status": e.status, "reason": e.reason, "boot_id": e.boot_id, **stamp()}, out / "JOB_ABORTED.json")
        raise
    dump({"job": plan["job"], "plan_sha": MS.sha(plan), "card_sha": card["card_sha"], "boots": res, "resumed": bool(resume), **stamp()},
         out / "RUN_DONE.json")
    return res


def _default_gate(plan):
    import mo_rehearsal as MR
    return MR.launch_gate(plan)


def exit_code(res):
    return EXIT_BOOT_FAIL if any(b["status"] != "done" for b in res) else 0


def _plan_from_args(a, w1_pool=None):
    grid = MS.make_grid(a.rmin, a.factor, a.n, a.role)
    return MS.make_plan(a.job, grid, rehearsal=a.rehearsal, n_requests=a.n_requests, workload=a.workload)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("plan", "card", "run", "rehearse", "gate"):
        p = sub.add_parser(n)
        p.add_argument("--job", required=True)
        p.add_argument("--rmin", type=float, default=3.0)
        p.add_argument("--factor", type=float, default=1.15)
        p.add_argument("--n", type=int, default=6)
        p.add_argument("--role", default="base")
        p.add_argument("--n-requests", type=int, default=120)
        p.add_argument("--rehearsal", action="store_true")
        p.add_argument("--w1-lengths")
        p.add_argument("--out")
        p.add_argument("--card")
        p.add_argument("--resume", action="store_true")
        p.add_argument("--workload", default=MS.WORKLOAD)
    a = ap.parse_args(argv)
    if a.cmd == "rehearse":
        import mo_rehearsal as MR
        pool = json.load(open(a.w1_lengths))
        rec = MR.rehearse(a.job, pool, a.out, lambda bdir, arm_def: MoProcessServer(bdir, arm_def), ML.replay, a.rmin, workload=a.workload)
        print(json.dumps({k: rec[k] for k in ("passed", "harness_digest", "verdict_rows", "reason", "workload", "workloads_resolved")}))
        return 0 if rec["passed"] else EXIT_ABORT
    plan = _plan_from_args(a)
    if a.cmd == "plan":
        print(json.dumps({"plan_sha": MS.sha(plan), "problems": MS.plan_problems(plan), "boot_order": plan["boot_order"], "boots": len(plan["boots"])}))
        return 0
    if a.cmd == "gate":
        why = _default_gate(plan)
        print(why or "REHEARSAL_RECORD_OK")
        return 1 if why else 0
    if a.cmd == "run" and not a.rehearsal and a.workload == "W1":
        print("ABORT_WORKLOAD: W1 (sealed sb pool) is the rehearsal fixture; a measurement Job names a mo_workload.WORKLOADS workload",
              file=sys.stderr)
        return EXIT_ABORT
    pool = json.load(open(a.w1_lengths))
    if a.cmd == "card":
        card = MC.build_card(plan, pool)
        dump(card, a.out)
        print(json.dumps({"card_sha": card["card_sha"], "placeholders": MC.card_placeholders(card)}))
        return 0
    card = json.load(open(a.card))
    try:
        res = run_job(plan, card, a.out, lambda bdir, arm_def: MoProcessServer(bdir, arm_def), ML.replay, pool, a.resume)
    except JobAbort as e:
        print(f"{e.status}: {e.reason}", file=sys.stderr)
        return EXIT_ABORT
    return exit_code(res)


if __name__ == "__main__":
    sys.exit(main())
