#!/usr/bin/env python3
"""ONE server boot + an ordered list of steps (PREREG_MULTIGPU rev6 §1, §3, §4-1; harness H1–H8).

Why a new runner and not engine_bench_runner.sh: that runner is "boot per trace" (H1 needs one boot to carry
shape A and B x seeds), waits 360 s for health (H2 needs 1800 s), has no vanilla-with-PD-mux-knobs path (H3) and
is shared with N2/N3/E2.  It is left untouched.  This file reuses its conventions (server flags, the
multigpu_cell_meta.py comm-env / server-facts records).

SPEC (JSON, --spec FILE)
  boot_id config mode N tp dp gpu_ids arm kind block   identity (cell.json copies them)
  steps: [ {type:"warmup", shape}                       discarded warm-up phase (phase separation)
         , {type:"cell", shape, rung:"HI"|"LO"|"G3B"|..., seed, offered, n_req, ext, round, block, gpu?}
         , {type:"cmd", name, argv:[...]} ]              argv may contain {ENDPOINT} {PORT} {OUT} {BOOT}
  deadline_epoch    Job hard-cap deadline (steps not started by then are NOT_STARTED; a running cell is killed
                    and recorded TRUNCATED)
  extra_env         env set for this boot only (G0(ii) forbidden-env boots)
  config_rel        override of the PD-mux config (G0(iii) auto-grid boot)
  expect_refusal    text that must appear in the server log when the boot is EXPECTED to die (G0(ii)/(iii))
  nsys              wrap the server in `nsys launch` (descriptive block; steps carry nsys start/stop cmds)
Writes under <out_root>/boots/<boot_id>/ and <out_root>/cells/<boot_id>__<cell>/ ; see module constants.
Per boot (rev6 harness audit 2026-10-12): alive_after.json (F2, schema m2g_alive_after/v1 of the gate module: poll() + GET /health after
the LAST step, before teardown), cmd_<name>.window.json for every cmd step ({t0_mono,t1_mono} on CLOCK_MONOTONIC = the telemetry clock,
F3), contam_pre.json / boot.json contam_status = CONTAM_UNVERIFIED when NVML is unavailable (F5: the check did not run; not 'clean').
Exit: 0 ok · 1 boot failed · 2 spec error · 3 ABORT_COMM_ENV · 5 GPU_CONTAMINATED (pre-boot) · 6 nothing ran (deadline)
`--dry-run` (or PDMUX_MG_DRYRUN=1) writes server_args.txt / env.json / steps.json and returns -- no GPU, no network.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(os.path.abspath(__file__)).parent
ENGINE = HERE.parents[1]
sys.path.insert(0, str(ENGINE / "results" / "multigpu"))
sys.path.insert(0, str(ENGINE / "scripts" / "r2_eval"))
import mg_const as C  # noqa: E402
import multigpu_cell_meta as CM  # noqa: E402

# env names this launcher unsets for every boot (N3 precedent n3_capacity.sh: knobs of other campaigns)
UNSET_ALWAYS = ("PDMUX_SLO_SCHED", "PDMUX_SLO_MODE", "PDMUX_LA_COORD", "PDMUX_STICKY_PARTITION",
                "PDMUX_FIXED_DECODE_SM_FILE", "PDMUX_TRACE_FORCE_PREFILL", "PDMUX_DUAL_WORKER_TRACE_EVERY",
                "PDMUX_TRUE_DUAL_WORKER", "PDMUX_DUAL_WORKER", "PDMUX_DUAL_WORKER_TRACE", "PDMUX_MEM_TELEMETRY",
                "PDMUX_PHASE_EVENTS", "PDMUX_R2_POLICY", "PDMUX_R2_FIXED_DSM", "PDMUX_MODEL_PROFILE",
                "PDMUX_LAYER_GRANULAR_BASELINE", "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN") + C.COMM_ENV_UNSET


# =========================================================================== pure builders (unit-tested)
def effective_cap(tp, dp):
    """§1: TP=N -> 48*N for the one engine instance; DP=N -> 48 per replica; N=1 -> 48."""
    return C.CAP_PER_GPU * tp if tp > 1 else C.CAP_PER_GPU


def build_server_args(spec, port, config_abs=None):
    arm, tp, dp = spec["arm"], int(spec["tp"]), int(spec["dp"])
    if arm not in C.ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    if tp > 1 and dp > 1:
        raise ValueError("TP and DP are not mixed")
    a = ["--model-path", C.MODEL, "--trust-remote-code", "--dtype", "bfloat16",
         "--attention-backend", C.ATTENTION_BACKEND,
         "--disable-radix-cache",
         "--mem-fraction-static", str(C.MEM_FRACTION),
         "--max-running-requests", str(effective_cap(tp, dp)),
         "--context-length", str(C.CONTEXT_LENGTH),
         "--random-seed", str(C.SERVER_SEED),
         "--host", "127.0.0.1", "--port", str(port)]
    if arm in C.PDMUX_ARMS:
        cfg = spec.get("config_rel") or (C.AGNOSTIC_CONFIG_REL if arm == "agnostic" else C.D44_CONFIG_REL)
        a += ["--enable-pdmux", "--pdmux-config-path", config_abs or cfg]
    if arm in C.PDMUX_ARMS or arm == "fused_sym":
        a += list(C.SCHED_KNOBS)            # byte-identical for fused_sym and both PD-mux arms (H3)
    if arm == "fused":
        a += list(C.FUSED_EXTRA_FLAGS)      # fused = vanilla + --disable-piecewise-cuda-graph (only)
    a += ["--decode-log-interval", str(C.DECODE_LOG_INTERVAL)]
    if tp > 1:
        a += ["--tp-size", str(tp)]
    if dp > 1:
        a += ["--dp-size", str(dp), "--load-balance-method", C.LOAD_BALANCE_METHOD]
    return a


def build_env(spec, boot_dir):
    """(set, unset).  PD-mux arms: telemetry + green read-out; d44 pins R2 fixed 44; fused* never sets R2."""
    arm = spec["arm"]
    s = {"SGLANG_LOG_MS": C.SGLANG_LOG_MS, "PDMUX_ATTENTION_BACKEND": C.ATTENTION_BACKEND,
         "PDMUX_ENGINE_MODE": "pdmux" if arm in C.PDMUX_ARMS else "vanilla",
         "PDMUX_RUN_ID": spec["boot_id"], "PDMUX_WORKLOAD_ID": f"mg_{spec['config']}_{arm}"}
    if arm in C.PDMUX_ARMS:
        s["PDMUX_TELEMETRY_PATH"] = str(Path(boot_dir) / "tel.jsonl")
        s["PDMUX_GREEN_READOUT"] = "1"
        s["PDMUX_GREEN_READOUT_PATH"] = str(Path(boot_dir) / "green.json")
    if arm == "d44":
        s["PDMUX_R2_POLICY"] = "fixed"
        s["PDMUX_R2_FIXED_DSM"] = str(C.D44_FIXED_DSM)
    s.update(spec.get("extra_env") or {})
    unset = [k for k in UNSET_ALWAYS if k not in s]
    return s, unset


def gen_trace(path, shape, offered, n_req, seed, tag="mg"):
    """Poisson arrivals at `offered` req/s, fixed (input, output) lengths -- the trace_loadgen schema.
    Byte-identical for equal arguments (same trace hash across arms)."""
    inp, out = C.SHAPE_IO[shape]
    rng = random.Random(seed)
    now, lines = 0.0, [json.dumps({"record": "workload", "workload_id": f"{tag}-{shape}", "seed": seed,
                                   "offered": offered, "n_req": n_req, "shape": shape}, sort_keys=True)]
    for i in range(n_req):
        now += rng.expovariate(offered)
        lines.append(json.dumps({"record": "request", "request_id": f"{tag}-{shape}-{seed}-{i:06d}",
                                 "arrival_s": now, "input_tokens": inp, "output_tokens": out,
                                 "prefix_tokens": inp, "phase": tag}, sort_keys=True))
    Path(path).write_text("\n".join(lines) + "\n")
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def piecewise_record(log_text):
    """H8: piecewise-CUDA-graph settings and capture lines as the server really printed them."""
    rec = {}
    for key in ("disable_piecewise_cuda_graph", "piecewise_cuda_graph_tokens", "piecewise_cuda_graph_max_tokens",
                "disable_cuda_graph", "cuda_graph_max_bs", "chunked_prefill_size", "disable_overlap_schedule"):
        m = re.search(rf"{key}=(\[[^\]]*\]|[^,\s)]+)", log_text)
        if m:
            v = m.group(1)
            rec[key] = v if len(v) < 300 else v[:300] + "..."
    lines = [ln for ln in log_text.splitlines()
             if re.search(r"piecewise|Capture cuda graph|cuda graph", ln, re.I) and "ServerArgs(" not in ln]
    rec["capture_log_lines"] = len(lines)
    rec["piecewise_lines"] = sum(1 for ln in lines if re.search("piecewise", ln, re.I))
    rec["sample"] = lines[:60]
    return rec, lines


def pick_port():
    lo, span = 20000, 800
    try:
        eph_lo, eph_hi = (int(x) for x in Path("/proc/sys/net/ipv4/ip_local_port_range").read_text().split())
    except OSError:
        eph_lo = eph_hi = None
    if eph_lo is not None and lo + span - 1 >= eph_lo and lo <= eph_hi:
        raise RuntimeError(f"listen window {lo}+{span} overlaps ephemeral range {eph_lo}-{eph_hi}")
    for _ in range(100):
        p = lo + random.randrange(span)
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    raise RuntimeError("no free port")


# =========================================================================== NVML (contamination check, §3 G5(c))
def nvml_snapshot(indices):
    try:
        import pynvml as n
        n.nvmlInit()
    except Exception as e:  # noqa: BLE001
        return {"available": False, "error": repr(e)}
    out = {"available": True, "gpus": {}}
    for i in indices:
        h = n.nvmlDeviceGetHandleByIndex(i)
        used = n.nvmlDeviceGetMemoryInfo(h).used
        try:
            nproc = len(n.nvmlDeviceGetComputeRunningProcesses(h))
        except Exception as e:  # noqa: BLE001
            nproc = None
        out["gpus"][str(i)] = {"used_mib": used / 1048576.0, "n_compute_procs": nproc}
    return out


def contamination_verdict(snap, proc_count_hidden):
    """(가) used < 1 GiB on every subset GPU; (나) compute-process count == expected (0 before boot) unless the
    count is hidden (container PID namespace) in which case only (가) counts."""
    if not snap.get("available"):
        # F5: NVML unavailable == the contamination check did NOT run.  It is never "clean": the flag CONTAM_UNVERIFIED travels
        # to boot.json, the M2g table and LABELS (the Job preflight also aborts a real run without NVML, ABORT_SUBSTRATE).
        return {"gpu_contaminated": False, "verified": False, "contam_unverified": True, "flag": "CONTAM_UNVERIFIED",
                "reason": "NVML unavailable (CONTAM_UNVERIFIED)"}
    bad = []
    for i, g in snap["gpus"].items():
        if g["used_mib"] >= C.NVML_EMPTY_GPU_MIB:
            bad.append(f"gpu{i}: memory.used {g['used_mib']:.0f} MiB >= {C.NVML_EMPTY_GPU_MIB}")
        if not proc_count_hidden and g["n_compute_procs"] not in (None, 0):
            bad.append(f"gpu{i}: {g['n_compute_procs']} compute processes before boot (expected 0)")
    return {"gpu_contaminated": bool(bad), "verified": True, "contam_unverified": False, "reasons": bad}


# =========================================================================== process plumbing
def read_cpu_stat():
    try:
        return Path("/sys/fs/cgroup/cpu.stat").read_text()
    except OSError:
        return ""


def http_get(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.status, r.read()


def kill_group(proc, grace=10):
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    t = time.time()
    while time.time() - t < grace and proc.poll() is None:
        time.sleep(0.5)
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def write_alive(boot_dir, boot_id, server, port, spec_steps, steps_done):
    """F2: ONE survival record per boot, after the LAST step and BEFORE teardown: `poll()` + GET /health.  The writer and the readers
    (G1 '비정상 종료 0', G3b '서버 생존') are the gate module's single implementation (m2g_correctness_gate.record_alive_after /
    alive_after, schema m2g_alive_after/v1).  If it cannot be written the file is simply absent -> the gates read NOT_TESTED, never alive."""
    try:
        import m2g_correctness_gate as GATE
        last = steps_done[-1] if steps_done else None
        after = ({"index": len(steps_done) - 1, "type": last["step"]["type"], "status": last["status"]} if last else None)
        GATE.record_alive_after(boot_dir, boot_id, server, f"http://127.0.0.1:{port}", after, len(spec_steps))
    except Exception as e:  # noqa: BLE001
        try:
            (Path(boot_dir) / "alive_after.error").write_text(repr(e) + "\n")
        except OSError:
            pass


def sweep_sglang_children():
    """SGLang children rewrite their cmdline (setproctitle); one server per container (N3 O3)."""
    subprocess.run(["pkill", "-9", "-f", "^sglang::"], check=False)


def jdump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1, sort_keys=True, default=str) + "\n")


# =========================================================================== main
def run_boot(spec, dry=False):
    out_root = Path(spec["out_root"])
    boot_dir = out_root / "boots" / spec["boot_id"]
    boot_dir.mkdir(parents=True, exist_ok=True)
    tp, dp = int(spec["tp"]), int(spec["dp"])
    n_gpus = max(tp, dp) if (tp > 1 or dp > 1) else 1
    gpu_ids = spec["gpu_ids"]
    ids = [int(x) for x in gpu_ids.split(",")]
    if len(ids) != n_gpus or len(set(ids)) != n_gpus:
        print(f"spec error: gpu_ids {gpu_ids!r} must list exactly {n_gpus} distinct GPUs", file=sys.stderr)
        return 2
    boot = {"boot_id": spec["boot_id"], "config": spec["config"], "arm": spec["arm"], "kind": spec.get("kind"),
            "gpu_ids": gpu_ids, "tp": tp, "dp": dp, "effective_cap": effective_cap(tp, dp), "status": "STARTED",
            "t_start_epoch": time.time(), "host": socket.gethostname()}
    port = 0 if dry else pick_port()
    cfg_rel = spec.get("config_rel") or (C.AGNOSTIC_CONFIG_REL if spec["arm"] == "agnostic" else C.D44_CONFIG_REL)
    cfg_abs = str(ENGINE / cfg_rel)
    sargs = build_server_args(spec, port or 0, cfg_abs if spec["arm"] in C.PDMUX_ARMS else None)
    # server_args.txt is the provenance record (port normalised so arms/boots diff cleanly; the real port is in boot.json)
    rec_args = [("<PORT>" if (i > 0 and sargs[i - 1] == "--port") else x) for i, x in enumerate(sargs)]
    (boot_dir / "server_args.txt").write_text("\n".join(rec_args) + "\n")
    env_set, env_unset = build_env(spec, boot_dir)
    jdump(boot_dir / "env.json", {"set": env_set, "unset": env_unset, "CUDA_VISIBLE_DEVICES": gpu_ids})
    jdump(boot_dir / "steps.json", spec.get("steps", []))
    # H8 / G5: record the sched knobs and fd limit in the boot record
    import resource
    boot["ulimit_n_soft"], boot["ulimit_n_hard"] = resource.getrlimit(resource.RLIMIT_NOFILE)
    boot["registered"] = {"attention_backend": C.ATTENTION_BACKEND, "decode_log_interval": C.DECODE_LOG_INTERVAL,
                          "SGLANG_LOG_MS": C.SGLANG_LOG_MS, "sock_read_s": C.SOCK_READ_S,
                          "fd_soft_min": C.FD_SOFT_MIN, "health_wait_s": spec.get("health_wait_s", C.HEALTH_WAIT_S)}
    env = dict(os.environ)
    for k in env_unset:
        env.pop(k, None)
    # comm env: record what the PLATFORM set (inherited) BEFORE unsetting the registered-unset variables
    inherited = {k: os.environ.get(k) for k in C.COMM_ENV_UNSET + (C.JIT_ALL_REDUCE_ENV,) if os.environ.get(k) is not None}
    jdump(boot_dir / "comm_env_inherited.json", inherited)
    env.update(env_set)
    env["CUDA_VISIBLE_DEVICES"] = gpu_ids
    rec = {"vars": CM.comm_env_record(env), "pinned": {}, "jit_all_reduce_enabled": CM.jit_all_reduce_enabled(env)}
    jdump(boot_dir / "comm_env.json", rec)
    if rec["jit_all_reduce_enabled"]:           # §1: SGLANG_USE_JIT_ALL_REDUCE must be unset -> ABORT_COMM_ENV (rc 3)
        boot["status"] = "ABORT_COMM_ENV"
        jdump(boot_dir / "boot.json", boot)
        return 3
    if dry or os.environ.get("PDMUX_MG_DRYRUN") == "1":
        boot["status"] = "DRYRUN"
        jdump(boot_dir / "boot.json", boot)
        return 0

    # ---- contamination (가)(나) before the boot; teardown of the previous boot may still be settling
    hidden_flag = out_root / "PROC_COUNT_HIDDEN"
    hidden = hidden_flag.exists()
    verdict = None
    for _ in range(36):                       # up to 180 s for the previous server's memory to drain
        snap = nvml_snapshot(ids)
        verdict = contamination_verdict(snap, hidden)
        if not verdict["gpu_contaminated"]:
            break
        time.sleep(5)
    jdump(boot_dir / "contam_pre.json", {"snapshot": snap, "verdict": verdict})
    boot["contamination_pre"] = verdict
    boot["contam_status"] = "CONTAM_UNVERIFIED" if verdict.get("contam_unverified") else "VERIFIED"
    if verdict["gpu_contaminated"]:
        jdump(boot_dir / "contam.json", {"gpu_contaminated": True, "phase": "pre-boot", **verdict})
        boot["status"] = "GPU_CONTAMINATED"
        jdump(boot_dir / "boot.json", boot)
        return 5

    srv_cmd = [sys.executable, "-m", "sglang.launch_server"] + sargs
    if spec.get("nsys"):
        srv_cmd = ["nsys", "launch", "--trace=cuda,nvtx"] + srv_cmd
    log_path = boot_dir / "server.log"
    t_launch = time.time()
    with open(log_path, "wb") as lf:
        server = subprocess.Popen(srv_cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    boot["server_pid"], boot["port"] = server.pid, port
    steps_done, rc = [], 0
    try:
        # ---- H2: health wait (1800 s registered)
        health_ok = False
        wait = float(spec.get("health_wait_s", C.HEALTH_WAIT_S))
        if spec.get("deadline_epoch"):                         # §6-2: the hard cap also bounds the boot wait
            wait = max(1.0, min(wait, float(spec["deadline_epoch"]) - time.time()))
        while time.time() - t_launch < wait:
            if server.poll() is not None:
                break
            try:
                if http_get(f"http://127.0.0.1:{port}/health", 3)[0] == 200:
                    health_ok = True
                    break
            except Exception:  # noqa: BLE001
                pass
            time.sleep(2)
        boot["health_wait_s_used"] = time.time() - t_launch
        boot["health_wait_cap_s"] = wait            # the cap actually applied (< 1800 s => the gate scores NOT_TESTED, MGP-23)
        boot["server_exit_code"] = server.poll()    # None while alive; gate G0(ii)/(iii) needs "died abnormally"
        log_text = log_path.read_text(errors="replace")
        if spec.get("expect_refusal"):
            died = server.poll() is not None
            found = spec["expect_refusal"] in log_text
            boot["refusal"] = {"expected_text": spec["expect_refusal"], "text_found": found, "process_died": died,
                               "health_ok": health_ok, "outcome": "REFUSED_AS_EXPECTED" if (found and died and not health_ok)
                               else "REFUSAL_FAILED"}
            boot["status"] = boot["refusal"]["outcome"]
            return 0 if boot["refusal"]["outcome"] == "REFUSED_AS_EXPECTED" else 1
        if not health_ok:
            boot["status"] = "BOOT_FAILED" if server.poll() is not None else "BOOT_TIMEOUT"
            boot["tail"] = log_text.splitlines()[-40:]
            return 1
        boot["t_health_epoch"] = time.time()
        boot["boot_seconds"] = boot["t_health_epoch"] - t_launch
        # ---- facts recorded for EVERY boot (C-5, H8, G4, G5)
        for name in ("model_info", "server_info"):
            try:
                (boot_dir / f"{name}.json").write_bytes(http_get(f"http://127.0.0.1:{port}/{name}", 20)[1])
            except Exception as e:  # noqa: BLE001
                (boot_dir / f"{name}.json").write_text(json.dumps({"error": repr(e)}))
        CM.main(["server-facts", "--log", str(log_path), "--server-info", str(boot_dir / "server_info.json"),
                 "--out", str(boot_dir / "cell_meta.json"), "--tp", str(tp), "--dp", str(dp),
                 "--max-running-requested", str(effective_cap(tp, dp)), "--gpu-ids", gpu_ids])
        log_text = log_path.read_text(errors="replace")
        pw, cap_lines = piecewise_record(log_text)
        jdump(boot_dir / "piecewise.json", pw)
        cm = json.loads((boot_dir / "cell_meta.json").read_text()) if (boot_dir / "cell_meta.json").exists() else {}
        cm["piecewise"] = pw                                 # H8: the setting + capture lines live IN cell_meta.json too
        cm["cuda_graph_capture_log"] = "cuda_graph_capture.log"
        cm["ulimit_n"] = [boot["ulimit_n_soft"], boot["ulimit_n_hard"]]
        jdump(boot_dir / "cell_meta.json", cm)
        (boot_dir / "cuda_graph_capture.log").write_text("\n".join(cap_lines) + "\n")
        post = nvml_snapshot(ids)
        jdump(boot_dir / "nvml_post_boot.json", post)       # arm memory.used comparison is descriptive (§3 G5(c))
        if post.get("available") and not hidden:
            counts = [g["n_compute_procs"] for g in post["gpus"].values()]
            if all(c == 0 for c in counts) and any(g["used_mib"] >= C.NVML_EMPTY_GPU_MIB for g in post["gpus"].values()):
                hidden_flag.write_text("count hidden by PID namespace; only (가) is used (PREREG §3 G5(c))\n")
        jdump(boot_dir / "proc_counts_post_boot.json",
              {k: v["n_compute_procs"] for k, v in (post.get("gpus") or {}).items()})
        exp = spec.get("expected_proc_counts")          # H7: M2g G0 table, compared in M3 (PID-namespace safe: counts only)
        if exp and post.get("available") and not hidden_flag.exists():
            got = {k: v["n_compute_procs"] for k, v in post["gpus"].items()}
            bad = {k: {"got": got.get(k), "expected": exp[k]} for k in exp if got.get(k) != exp[k]}
            boot["contamination_post"] = bad
            if bad:
                jdump(boot_dir / "contam.json", {"gpu_contaminated": True, "phase": "post-boot (나)", "mismatch": bad})
        # ---- steps
        warmed = set()
        deadline = spec.get("deadline_epoch")
        for st in spec.get("steps", []):
            if deadline and time.time() >= deadline:
                steps_done.append({"step": st, "status": "NOT_STARTED_DEADLINE"})
                continue
            if server.poll() is not None:
                steps_done.append({"step": st, "status": "SERVER_DIED"})
                rc = 1
                continue
            if st["type"] == "warmup":
                steps_done.append({"step": st, "status": run_warmup(spec, st, boot_dir, port, n_gpus, deadline)})
                warmed.add(st["shape"])
            elif st["type"] == "cell":
                if st["shape"] not in warmed:      # a cell is never the first traffic of its shape
                    run_warmup(spec, {"shape": st["shape"]}, boot_dir, port, n_gpus, deadline)
                    warmed.add(st["shape"])
                steps_done.append({"step": st, "status": run_cell(spec, st, boot_dir, out_root, port, server.pid, deadline)})
            elif st["type"] == "cmd":
                steps_done.append({"step": st, "status": run_cmd(st, boot_dir, out_root, port, deadline)})
            else:
                steps_done.append({"step": st, "status": "UNKNOWN_STEP_TYPE"})
        boot["status"] = "OK" if all(s["status"] in ("OK", "NOT_STARTED_DEADLINE") for s in steps_done) else "STEP_FAILURES"
        if steps_done and all(s["status"] == "NOT_STARTED_DEADLINE" for s in steps_done):
            rc = 6
        return rc
    finally:
        boot["steps"] = [{"type": s["step"]["type"], "status": s["status"],
                          **({"cell": f"{s['step'].get('shape')}_{s['step'].get('rung')}_s{s['step'].get('seed')}"}
                             if s["step"]["type"] == "cell" else {})} for s in steps_done]
        write_alive(boot_dir, spec["boot_id"], server, port, spec.get("steps", []), steps_done)      # F2: after the last step, before teardown
        kill_group(server)
        sweep_sglang_children()
        t = time.time()
        post_ids_snapshot = None
        while time.time() - t < 180:             # O4: wait until the memory is back (recorded; the next boot's (가) is the gate)
            post_ids_snapshot = nvml_snapshot(ids)
            if not post_ids_snapshot.get("available") or all(
                    g["used_mib"] < C.NVML_EMPTY_GPU_MIB for g in post_ids_snapshot["gpus"].values()):
                break
            time.sleep(5)
        boot["teardown_settle_s"] = time.time() - t
        boot["t_end_epoch"] = time.time()
        jdump(boot_dir / "boot.json", boot)


def run_warmup(spec, st, boot_dir, port, n_gpus, deadline):
    shape = st["shape"]
    n = min(64, C.WARMUP_REQS_PER_GPU * n_gpus)
    wd = boot_dir / f"warmup_{shape}"
    wd.mkdir(exist_ok=True)
    gen_trace(wd / "trace.jsonl", shape, 4.0, n, 7, tag="warm")        # 4 req/s: spaced, far below capacity
    cmd = [sys.executable, str(HERE / "mg_loadgen.py"), "--endpoint", f"http://127.0.0.1:{port}/generate",
           "--trace", str(wd / "trace.jsonl"), "--cell-dir", str(wd)]
    tmo = 900 if not deadline else max(1.0, deadline - time.time())
    p = subprocess.run(cmd, timeout=tmo, capture_output=True, text=True, check=False)
    (wd / "loadgen.log").write_text(p.stdout + p.stderr)
    return "OK" if p.returncode == 0 else f"WARMUP_RC_{p.returncode}"


def run_cell(spec, st, boot_dir, out_root, port, server_pid, deadline):
    cid = f"{spec['boot_id']}__{st['shape']}_{st['rung']}{'x' if st.get('ext') else ''}_s{st['seed']}"
    cd = out_root / "cells" / cid
    cd.mkdir(parents=True, exist_ok=True)
    thash = gen_trace(cd / "trace.tmp.jsonl", st["shape"], float(st["offered"]), int(st["n_req"]), int(st["seed"]))
    (cd / "trace.sha256").write_text(thash + "\n")                     # the bytes are regenerable; the hash is the record
    meta = {"cell_id": cid, "boot_id": spec["boot_id"], "boot_dir": f"boots/{spec['boot_id']}",
            "config": spec["config"], "mode": spec["mode"], "N": spec["N"], "arm": spec["arm"], "tp": spec["tp"],
            "dp": spec["dp"], "gpu_ids": spec["gpu_ids"], "shape": st["shape"], "rung": st["rung"], "seed": st["seed"],
            "offered": st["offered"], "n_req": st["n_req"], "ext": st.get("ext", 0), "round": st.get("round", 1),
            "block": st.get("block", "main"), "gpu": st.get("gpu"), "trace_sha256": thash,
            "host": socket.gethostname(), "t_start_epoch": time.time()}
    sampler = subprocess.Popen([sys.executable, str(HERE / "mg_proc_sampler.py"), "--root-pid", str(server_pid),
                                "--out", str(cd / "proc_samples.csv")], start_new_session=True)
    (cd / "cpu_stat_before.txt").write_text(read_cpu_stat())
    cmd = [sys.executable, str(HERE / "mg_loadgen.py"), "--endpoint", f"http://127.0.0.1:{port}/generate",
           "--trace", str(cd / "trace.tmp.jsonl"), "--cell-dir", str(cd)]
    status = "OK"
    try:
        tmo = None if not deadline else max(1.0, deadline - time.time())
        with open(cd / "loadgen.log", "wb") as llog:
            p = subprocess.Popen(cmd, stdout=llog, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                rc = p.wait(timeout=tmo)
            except subprocess.TimeoutExpired:
                kill_group(p, grace=5)
                rc, status = -9, "TRUNCATED"
                meta["truncated"] = True
        meta["loadgen_rc"] = rc
        if rc not in (0, -9):
            status = f"LOADGEN_RC_{rc}"
        if spec["arm"] in C.PDMUX_ARMS and st["rung"] in ("HI", "LO") and not meta.get("truncated"):
            post_cell_etime(spec, st, boot_dir, cd)
    finally:
        (cd / "cpu_stat_after.txt").write_text(read_cpu_stat())
        kill_group(sampler, grace=3)
        meta["t_end_epoch"] = time.time()
        (cd / "trace.tmp.jsonl").unlink(missing_ok=True)
        jdump(cd / "cell.json", meta)
    return status


def post_cell_etime(spec, st, boot_dir, cd):
    """E-time of this cell via engine-porter's estimator (mg_realized_mix.cell_e_time); window = send window on
    the CLOCK_MONOTONIC axis of the engine telemetry (loadgen_clock.perf_at_call + sent_s)."""
    try:
        import mg_realized_mix as RM
        clk = json.loads((cd / "loadgen_clock.json").read_text())
        sent = [json.loads(ln)["sent_s"] for ln in (cd / "requests.jsonl").read_text().splitlines() if ln.strip()]
        win = (clk["perf_at_call"] + min(sent), clk["perf_at_call"] + max(sent))
        mode = {"N1": "n1", "TP": "tp", "DP": "dp"}[spec["mode"]]
        res = RM.cell_e_time(spec["arm"], mode, int(spec["N"]), str(boot_dir / "tel.jsonl"),
                             str(boot_dir / "green.json"), win)
    except Exception as e:  # noqa: BLE001
        res = {"status": "MIX_FAILED", "e_time": None, "reasons": [f"post_cell_etime error {e!r}"]}
    jdump(cd / "etime.json", res)


def run_cmd(st, boot_dir, out_root, port, deadline):
    subs = {"{ENDPOINT}": f"http://127.0.0.1:{port}", "{PORT}": str(port), "{OUT}": str(out_root),
            "{BOOT}": str(boot_dir)}
    argv = []
    for x in st["argv"]:
        for k, v in subs.items():
            x = x.replace(k, v)
        argv.append(x)
    name = st.get("name", "x")
    log = boot_dir / f"cmd_{name}.log"
    tmo = st.get("timeout_s") or (None if not deadline else max(1.0, deadline - time.time()))
    t0m = time.monotonic()                                   # F3: the step window on CLOCK_MONOTONIC (= the telemetry clock)
    try:
        p = subprocess.run(argv, timeout=tmo, capture_output=True, text=True, check=False)
        log.write_text(p.stdout + "\n" + p.stderr)
        status = "OK" if p.returncode == 0 else f"CMD_RC_{p.returncode}"
    except subprocess.TimeoutExpired:
        status = "CMD_TIMEOUT"
    except FileNotFoundError as e:
        log.write_text(repr(e))
        status = "CMD_NOT_FOUND"
    t1m = time.monotonic()
    jdump(boot_dir / f"cmd_{name}.window.json", {"t0_mono": t0m, "t1_mono": t1m, "status": status,
                                                  "t0_epoch": time.time() - (t1m - t0m), "t1_epoch": time.time()})
    return status


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--spec", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    spec = json.loads(Path(a.spec).read_text())
    return run_boot(spec, dry=a.dry_run)


if __name__ == "__main__":
    sys.exit(main())
