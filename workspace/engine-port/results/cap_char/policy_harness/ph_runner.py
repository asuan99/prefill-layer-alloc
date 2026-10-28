#!/usr/bin/env python3
"""Runner for the policy Jobs (CAP1/CAP2/TUNE/EVAL-a/EVAL-b/HD-k).  NOT SEALED (ph_job.sh gates it).

  ph_runner.py run --job HD-3 --out RAW_DIR --addendum ADD.json --w1-lengths L.json [--w3-tokens F.bin] [--resume]
  ph_runner.py plan --job HD-3 --addendum ADD.json          # print the plan + its sha (no server)

Per boot: server start -> warm-up (discarded) -> correctness check -> each run's segments replayed one by one (a segment is
one bench_serving-style replay; the gap between segments is the drain) -> run record `runs/<run_id>.json` -> server stop.
Resume: a run whose record exists is skipped (recorded in RESUME_LOG.json; re-running only failed cells is reported).
PHASE_EVENTS / chunk / drain instrumentation is OFF by construction (PREREG Q-5): boot_env strips every PDMUX_* variable not
in an explicit allowlist, and `instr_violation` re-checks the server env before each boot.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

HERE = Path(os.path.abspath(__file__)).parent
sys.path.insert(0, str(HERE))
import ph_spec as P  # noqa: E402
import ph_trace as T  # noqa: E402
import ph_verdict as V  # noqa: E402
import ct_runner as R  # noqa: E402  (ProcessServer, server_argv, KEEP_ENV)

S = P.S
POLICY_CONTEXT = 20480                        # 16,384 + 1,024 (W3 cap) + slack [proposal]; recorded in every boot
FORBIDDEN_ENV = ("PDMUX_PHASE_EVENTS", "PDMUX_DECODE_PROBE_PATH", "PDMUX_ADMISSION_PATH", "PDMUX_TELEMETRY_PATH")
ALLOW_PDMUX = set(R.KEEP_ENV)
TS_RE = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)")
BIND_RE = re.compile(r"SLO-BIND (\d+)->(\d+)")


def dump(obj, path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ server env / argv
def boot_env(arm, bdir, base=None, run_id="local"):
    base = dict(os.environ if base is None else base)
    env = {k: v for k, v in base.items() if not (k.startswith("PDMUX_") and k not in ALLOW_PDMUX)}
    env.pop("SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN", None)
    env.update(PDMUX_RUN_ID=run_id, PDMUX_WORKLOAD_ID=f"policy_{bdir.name}", PYTHONUNBUFFERED="1",
               PDMUX_HEARTBEAT_S="0", CUDA_VISIBLE_DEVICES="0")
    env.update(P.arm_env(arm))
    return env


def instr_violation(env):
    return [k for k in FORBIDDEN_ENV if k in env]


def server_argv(arm, context=POLICY_CONTEXT):
    b = {"model": "nano", "dummy": False, "cap": 48, "engine": "pdmux", "tp": 1, "context": context}
    return R.server_argv(b, {})


# ------------------------------------------------------------------------------------------------ replay
def default_replay(endpoint, requests, out_path, ttft_slo_ms=1e12, itl_slo_ms=1e12):
    """Real replay via ph_loadgen (trace_loadgen.py stays byte-identical; W3 natural output + optional absolute stamps)."""
    import asyncio
    import ph_loadgen as PL
    abs_mono = os.environ.get("PDMUX_LOADGEN_ABS_MONO", "") == "1"
    summ = asyncio.run(PL.replay_natural(endpoint, requests, Path(out_path), ttft_slo_ms, itl_slo_ms, connector_limit=0,
                                         abs_mono=abs_mono))
    recs = [json.loads(x) for x in Path(out_path).read_text().splitlines() if x.strip()]
    return summ["duration_s"], recs


def replay_segment(replay_fn, endpoint, seg, out_dir, w3_tokens=None):
    reqs = seg["requests"]
    if w3_tokens and any("doc_id" in r for r in reqs):
        T.attach_w3_ids(reqs, w3_tokens)
    dur, recs = replay_fn(endpoint, reqs, Path(out_dir) / f"{seg['segment_id']}.jsonl")
    keep = ("request_id", "t_sent_mono", "t_finished_mono", "token_t_mono", "phase", "input_tokens", "requested_output_tokens", "observed_output_tokens", "ttft_ms",
            "token_itl_ms", "error", "error_kind", "scheduled_arrival_s", "completion_s")
    return {"segment_id": seg["segment_id"], "phase": seg["phase"], "round": seg.get("round", 0), "duration_s": float(dur),
            "phase_len_s": seg.get("phase_len_s"), "requests": [{k: r[k] for k in keep if k in r} for r in recs]}


def idx_after(log_text, anchor_idx):
    """Stream index the controller is at after replaying every SLO-BIND line of `log_text` (anchor before the first)."""
    cur = anchor_idx
    for b in BIND_RE.finditer(log_text):
        cur = int(b.group(2))
    return cur


def parse_switches(log_slice, start_idx, t0=None, t1=None):
    """switch_count + dwell seconds per D for ONE run: `log_slice` is only the bytes the server logged while this run executed
    (audit M2: the whole boot log would add warm-up and earlier runs); dwell starts at the run start t0 (wall clock) in
    `start_idx` and ends at t1.  Logger timestamps have 1 s resolution."""
    import datetime as dt
    cur, dwell, n = start_idx, {}, 0
    t_prev = t0
    t_last = t0
    for ln in log_slice.splitlines():
        m = TS_RE.match(ln)
        t = dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").timestamp() if m else None
        if t is not None:
            t_last = t
            if t_prev is None:
                t_prev = t
        b = BIND_RE.search(ln)
        if b and t is not None:
            n += 1
            dwell[cur] = dwell.get(cur, 0.0) + (t - t_prev)
            cur, t_prev = int(b.group(2)), t
    t_end = t1 if t1 is not None else t_last
    if t_prev is not None and t_end is not None:
        dwell[cur] = dwell.get(cur, 0.0) + max(0.0, t_end - t_prev)
    return n, {str(P.D_OF_IDX.get(k, k)): v for k, v in dwell.items()}


# ------------------------------------------------------------------------------------------------ one run
def build_segments(run, w1_pool, w3_pool, tag):
    k = run["kind"]
    if k == "cap":
        pool = w1_pool if run["workload"] == "W1" else w3_pool
        return T.cap_trace(run["workload"], pool, run["rate"], run["seed"], run["n_requests"], tag)
    if k == "he0":
        return T.he0_trace(w1_pool, run["rate_lo"], run["rate_hi"], run["np"], run["rounds"], run["seed"], tag)
    if k == "hd":
        return T.hd_trace(run["variant"], run["phase_len_s"], run["start"], run["rate_w1"], run["rate_w3"],
                          w1_pool, w3_pool, run["seed"], tag)
    raise ValueError(k)


def execute_run(run, arm, endpoint, replay_fn, w1_pool, w3_pool, out, w3_tokens=None):
    segs = build_segments(run, w1_pool, w3_pool, run["run_id"])
    thash = T.trace_hash(segs)
    rec = {"run_id": run["run_id"], "kind": run["kind"], "seed": run["seed"], "arm": arm, "trace_hash": thash,
           "spec": {k: v for k, v in run.items() if k not in ("run_id",)}, "segments": []}
    for s in segs:
        seg = replay_segment(replay_fn, endpoint, s, Path(out) / "segments" / run["run_id"], w3_tokens)
        if run["kind"] == "hd" and seg["phase_len_s"] and seg["duration_s"] - seg["phase_len_s"] > P.HD_DRAIN_TIMEOUT_S:
            seg["DRAIN_TIMEOUT"] = True
        rec["segments"].append(seg)
    return rec


def run_boot(boot, out, make_server, replay_fn, w1_pool, w3_pool, w3_tokens=None, run_id="local"):
    bdir = Path(out) / "boots" / boot["boot_id"]
    bdir.mkdir(parents=True, exist_ok=True)
    arm = boot["arm"]
    env = boot_env(arm, bdir, run_id=run_id)
    bad = instr_violation(env)
    if bad:
        raise RuntimeError(f"ABORT_INSTR: {bad} set in a policy boot (PREREG Q-5)")
    srv = make_server(bdir, server_argv(arm), env)
    status = srv.start()
    dump({"boot_id": boot["boot_id"], "arm": arm, "status": status, "argv": server_argv(arm), "context": POLICY_CONTEXT,
          "pdmux_env": {k: v for k, v in env.items() if k.startswith("PDMUX_")}}, bdir / "boot.json")
    if status != "ok":
        srv.stop()
        return {"boot_id": boot["boot_id"], "status": status, "runs": []}
    endpoint = f"http://127.0.0.1:{srv.port}/generate"
    done = []
    try:
        ok, why = warmup_and_check(endpoint, replay_fn, w1_pool, bdir, boot["boot_id"])
        if not ok:
            dump({"boot_id": boot["boot_id"], "status": "INVALID_CORRECTNESS", "why": why}, bdir / "DONE.json")
            return {"boot_id": boot["boot_id"], "status": "INVALID_CORRECTNESS", "runs": [], "why": why}
        if boot.get("floor"):
            name = "floor.json" if boot["floor"] is True else "floor_covariate.json"
            if not (Path(out) / name).exists():
                measure_floor(endpoint, replay_fn, w3_pool, out, name=name)
        for run in boot["runs"]:
            f = Path(out) / "runs" / f"{run['run_id']}.json"
            if f.exists():
                continue
            off0, t0 = _log_size(srv), time.time()
            rec = execute_run(run, arm, endpoint, replay_fn, w1_pool, w3_pool, out, w3_tokens)
            t1 = time.time()
            before, sl = _log_split(srv, off0)
            if arm["kind"] == "dynamic":
                rec["switch_count"], rec["split_dwell"] = parse_switches(sl, idx_after(before, arm["anchor_idx"]), t0, t1)
            else:
                # non-sticky static (decision 49): D is the TARGET while co-resident, not a realized allocation
                rec["switch_count"], rec["split_dwell"] = 0, {str(arm["D"]): sum(s["duration_s"] for s in rec["segments"])}
                rec["split_dwell_note"] = "target value while co-resident, not a realized allocation"
            rec["boot_id"] = boot["boot_id"]
            dump(rec, f)
            done.append(run["run_id"])
    finally:
        info = srv.stop()
    dump({"boot_id": boot["boot_id"], "status": "done", "runs": done, "server": info}, bdir / "DONE.json")
    return {"boot_id": boot["boot_id"], "status": "done", "runs": done}


WARMUP_S = 60.0                               # PREREG 2: warm-up 60 s


def warmup_and_check(endpoint, replay_fn, w1_pool, bdir, boot_id):
    """Warm-up (60 s of W1 at 2 req/s, discarded) + correctness: every fixed-length (ignore_eos) warm-up request must return
    exactly its requested number of tokens without error, else the boot is INVALID (audit M5)."""
    ts = T.arrivals(2.0, ("warm", boot_id), duration=WARMUP_S)
    reqs = T.make_requests("W1", w1_pool, ts, ("warm", boot_id), f"{boot_id}_warm", "WARM")
    seg = replay_segment(replay_fn, endpoint, {"segment_id": "warm", "phase": "WARM", "requests": reqs}, bdir / "warmup")
    bad = [q["request_id"] for q in seg["requests"]
           if q.get("error") or q.get("observed_output_tokens") != q.get("requested_output_tokens")]
    dump({"n": len(seg["requests"]), "bad": bad[:20], "n_bad": len(bad)}, bdir / "correctness.json")
    return (not bad), f"{len(bad)}/{len(seg['requests'])} warm-up requests wrong length or errored"


def _log_size(srv):
    p = getattr(srv, "log", None)
    return Path(p).stat().st_size if p and Path(p).exists() else 0


def _log_split(srv, off):
    p = getattr(srv, "log", None)
    if not p or not Path(p).exists():
        return "", ""
    b = Path(p).read_bytes()
    return b[:off].decode(errors="replace"), b[off:].decode(errors="replace")


def measure_floor(endpoint, replay_fn, w3_pool, out, n=8, seed=7, name="floor.json"):
    """W3 floor = median TTFT of n sequential single-request segments (same Job, before the capacity cells)."""
    ttfts = []
    for k in range(n):
        seg = T.cap_trace("W3", w3_pool, 1000.0, seed + k, 1, f"floor{k}")[0]
        s = replay_segment(replay_fn, endpoint, seg, Path(out) / "floor")
        ttfts += [r["ttft_ms"] for r in s["requests"] if r["ttft_ms"] is not None]
    f = statistics.median(ttfts) if ttfts else None
    dump({"n": len(ttfts), "ttft_ms": ttfts, "floor_ms": f}, Path(out) / name)
    return f


def run_job(plan, out, make_server, replay_fn, w1_pool, w3_pool, w3_tokens=None, resume=False, run_id="local"):
    problems = P.plan_problems(plan)
    if problems:
        raise RuntimeError(f"ABORT_PLAN: {problems}")
    out = Path(out)
    if out.exists() and any((out / "runs").glob("*.json")) and not resume:
        raise RuntimeError("ABORT_RAW_NOT_EMPTY: raw directory has runs; set --resume")
    dump(plan, out / "plan.json")
    res = []
    for boot in plan["boots"]:
        res.append(run_boot(boot, out, make_server, replay_fn, w1_pool, w3_pool, w3_tokens, run_id))
    dump({"job": plan["job"], "plan_sha": P.sha(plan), "boots": res, "resumed": bool(resume)}, out / "RUN_DONE.json")
    return res


# ------------------------------------------------------------------------------------------------ analysis entry
def load_runs(out):
    return [json.loads(p.read_text()) for p in sorted((Path(out) / "runs").glob("*.json"))]


def analyze(out, kind, addendum=None):
    runs = load_runs(out)
    res = {"n_runs": len(runs)}
    if kind == "EVAL":
        by = {}
        for r in runs:
            name = "dyn" if r["arm"]["kind"] == "dynamic" else f"s{r['arm']['D']}"
            by.setdefault(name, {})[r["seed"]] = r
        res["he0"] = V.he0_compare(by, P.W1_TTFT_MS, P.ITL_MS)
    return res


def exit_code(res):
    """A dead / invalid boot must not look like success (audit m9)."""
    rc = 0
    if any(b["status"] != "done" for b in res):
        rc = 17
    return rc


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("run", "plan"):
        p = sub.add_parser(n)
        p.add_argument("--job", required=True)
        p.add_argument("--addendum")
    sub.choices["run"].add_argument("--out", required=True)
    sub.choices["run"].add_argument("--w1-lengths", required=True)
    sub.choices["run"].add_argument("--w3-tokens")
    sub.choices["run"].add_argument("--resume", action="store_true")
    a = ap.parse_args(argv)
    add = json.load(open(a.addendum)) if a.addendum else None
    plan = P.make_plan(a.job, add)
    if a.cmd == "plan":
        print(json.dumps({"plan_sha": P.sha(plan), "problems": P.plan_problems(plan), "boots": len(plan["boots"])}))
        return 0
    w1 = json.load(open(a.w1_lengths))
    want = (add or {}).get("w1_lengths_sha256")
    if want and sha256_file(a.w1_lengths) != want:
        print("ABORT_DATASET: w1_lengths sha256 differs from the addendum (ShareGPT/tokenizer changed between Jobs)", file=sys.stderr)
        return 13
    w3 = T.load_w3_pool(a.w3_tokens)[0] if a.w3_tokens else []
    rc = 0
    res = run_job(plan, a.out, lambda bdir, argv_, env: R.ProcessServer(bdir, argv_, env), default_replay, w1, w3,
                  a.w3_tokens, a.resume, run_id=os.environ.get("PDMUX_JOB_NAME", "local"))
    rc = exit_code(res)
    if rc:
        print("BOOT_FAILURES: " + ",".join(f"{b['boot_id']}={b['status']}" for b in res if b["status"] != "done"), file=sys.stderr)
    return rc


if __name__ == "__main__":
    sys.exit(main())
