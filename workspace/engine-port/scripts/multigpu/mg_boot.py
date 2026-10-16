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
         , {type:"floor", lengths, repeats, max_new}     rev12 §19-4 floor block -> <boot>/FLOOR.json (+ .sha256)
         , {type:"cmd", name, argv:[...]} ]              argv may contain {ENDPOINT} {PORT} {OUT} {BOOT} {XTOKENS}
  mamba_ssm_dtype   rev12 §19-3 (M3 boots, required): value of --mamba-ssm-dtype resolved ONCE per Job (resolve_mamba_ssm_dtype)
  x_tokens_path / x_tokens_sha256   rev12 §19-2: the registered shape-X token file (cells / warm-up / floor / xcheck of X)
  deadline_epoch    Job hard-cap deadline (steps not started by then are NOT_STARTED; a running cell is killed
                    and recorded TRUNCATED)
  extra_env         env set for this boot only (G0(ii) forbidden-env boots)
  config_rel        override of the PD-mux config (G0(iii) auto-grid boot)
  expect_refusal    text that must appear in the server log when the boot is EXPECTED to die (G0(ii)/(iii))
  nsys              wrap the server in `nsys launch` (descriptive block; steps carry nsys start/stop cmds)
Writes under <out_root>/boots/<boot_id>/ and <out_root>/cells/<boot_id>__<cell>/ ; see module constants.
Per boot (rev6 harness audit 2026-10-12): alive_after.json (F2, schema m2g_alive_after/v1 of the gate module: poll() + GET /health after
the LAST step, before teardown), alive_after_<cmd step name>.json right after every cmd step (2026-10-13 repair: G1 reads the one of its
own step), cmd_<name>.window.json for every cmd step ({t0_mono,t1_mono} on CLOCK_MONOTONIC = the telemetry clock,
F3), contam_pre.json / boot.json contam_status = CONTAM_UNVERIFIED when NVML is unavailable (F5: the check did not run; not 'clean').
Exit: 0 ok · 1 boot failed · 2 spec error · 3 ABORT_COMM_ENV · 5 GPU_CONTAMINATED (pre-boot) · 6 nothing ran (deadline)
      · 7 ABORT_CONTEXT_TOO_SHORT · 8 ABORT_DATASET · 9 ABORT_CONFIG_DRIFT   (rev12: CPU pre-checks, BEFORE any GPU work; fatal for the Job)
rev12 pre-checks (before the contamination check / server start, also in --dry-run): every request this boot will send -- warm-up,
cells, floor block, xcheck -- is written as a trace under <boot>/precheck/ and passed to pdmux_eval.context_limit.fit_problems(
CONTEXT_LENGTH, traces) (the runner's 0eca947 predicate: input < ctx-6, input+output <= ctx-2); the X token file sha256; the PD-mux
config's split_forward_token_budget (65536).  After health: server_info must echo the registered knobs (ABORT_CONFIG_DRIFT), and the
`Mamba Cache is allocated` lines are recorded (mamba_facts.json: the effective-dtype witness of F-4).
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
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(os.path.abspath(__file__)).parent
ENGINE = HERE.parents[1]
sys.path.insert(0, str(ENGINE / "results" / "multigpu"))
sys.path.insert(0, str(ENGINE / "scripts" / "r2_eval"))
sys.path.insert(0, str(ENGINE / "benchmarks"))
sys.path.insert(0, str(HERE))
import mg_const as C  # noqa: E402
import multigpu_cell_meta as CM  # noqa: E402
import mg_x_pretokenize as XT  # noqa: E402
from pdmux_eval import context_limit as CL  # noqa: E402

M3_KINDS = ("boot1", "boot1ext", "boot2", "anchor", "gpudev")       # rev12 knobs (§19-3) apply to M3 boots; M2g boots are sealed (not re-run)
RC_CONTEXT, RC_DATASET, RC_CONFIG_DRIFT = 7, 8, 9

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


def resolve_mamba_ssm_dtype(config_json_path):
    """rev12 §19-3 R-4: the fixed revision's config.json key `mamba_ssm_dtype` (top level, else text_config) if present, else
    "float32" (engine default = M2g effective value).  `torch_dtype` is NEVER consulted.  Returns (value, source)."""
    cfg = json.loads(Path(config_json_path).read_text())
    for where, d in (("config.mamba_ssm_dtype", cfg), ("config.text_config.mamba_ssm_dtype", cfg.get("text_config") or {})):
        v = d.get("mamba_ssm_dtype") if isinstance(d, dict) else None
        if v:
            return str(v), where
    return C.MAMBA_SSM_DTYPE_DEFAULT, "default (key absent; torch_dtype not consulted)"


def m3_knob_args(spec):
    """rev12 §19-3: --max-prefill-tokens 16384, --max-mamba-cache-size = effective cap, --mamba-ssm-dtype (resolved once per Job)."""
    if not spec.get("mamba_ssm_dtype"):
        raise ValueError("M3 boot spec without mamba_ssm_dtype (rev12 §19-3: resolved once per Job, never defaulted per boot)")
    return ["--max-prefill-tokens", str(C.MAX_PREFILL_TOKENS),
            "--max-mamba-cache-size", str(effective_cap(int(spec["tp"]), int(spec["dp"]))),
            "--mamba-ssm-dtype", str(spec["mamba_ssm_dtype"])]


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
         "--context-length", str(C.CONTEXT_LENGTH)]
    if spec.get("kind") in M3_KINDS:
        a += m3_knob_args(spec)
    a += ["--random-seed", str(C.SERVER_SEED),
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


# ---------------------------------------------------------------------------- rev12 shape X traces (§19-2 R-8)
def x_perm(header, seed, check=True):
    """pi_s: retained doc_ids ascending, shuffled by random.Random(X_PERM_SEED_BASE + s).  For a registered seed the permutation's
    sha256 (int64 LE) must equal mg_const.X_PERM_SHA256 (fail-closed: a different Python `random` would change every X sample)."""
    from array import array
    ids = sorted(int(d[0]) for d in header["docs"])
    random.Random(C.X_PERM_SEED_BASE + int(seed)).shuffle(ids)
    want = C.X_PERM_SHA256.get(int(seed))
    if check and want is not None and header.get("lengths_sha256") == C.X_LENGTHS_SHA256:
        a = array("q", ids)
        if sys.byteorder != "little":
            a.byteswap()
        got = hashlib.sha256(a.tobytes()).hexdigest()
        if got != want:
            raise ValueError(f"X permutation seed {seed}: sha {got[:16]} != registered {want[:16]}")
    return ids


def gen_trace_x(path, offered, n_req, seed, header, tag="mg"):
    """Shape X: Poisson arrivals (random.Random(seed), as gen_trace) over the FIRST n_req documents of pi_s; each request carries
    doc_id and that document's (input, output) lengths; mg_loadgen sends the document's real tokens (input_ids) inside replay()."""
    perm = x_perm(header, seed)
    if n_req > len(perm):
        raise ValueError(f"X cell n_req {n_req} > retained {len(perm)} (no sampling with replacement)")
    idx = XT.doc_index(header)
    rng = random.Random(seed)
    now, lines = 0.0, [json.dumps({"record": "workload", "workload_id": f"{tag}-X", "seed": seed, "offered": offered,
                                   "n_req": n_req, "shape": "X", "x_lengths_sha256": header.get("lengths_sha256")}, sort_keys=True)]
    for i in range(n_req):
        now += rng.expovariate(offered)
        d = perm[i]
        _off, inp, out = idx[d]
        lines.append(json.dumps({"record": "request", "request_id": f"{tag}-X-{seed}-{i:06d}", "arrival_s": now, "doc_id": d,
                                 "input_tokens": inp, "output_tokens": out, "prefix_tokens": inp, "phase": tag}, sort_keys=True))
    Path(path).write_text("\n".join(lines) + "\n")
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gen_floor_trace(path, lengths, repeats, max_new, tag="floor"):
    """§19-4: L x repeats single requests (sent one at a time by mg_loadgen --sequential), prompt = the floor document's first L tokens."""
    lines = [json.dumps({"record": "workload", "workload_id": tag, "lengths": list(lengths), "repeats": repeats, "max_new": max_new},
                        sort_keys=True)]
    k = 0
    for L in lengths:
        for j in range(repeats):
            lines.append(json.dumps({"record": "request", "request_id": f"{tag}-{int(L):05d}-{j:02d}", "arrival_s": float(k),
                                     "floor_len": int(L), "input_tokens": int(L), "output_tokens": int(max_new), "prefix_tokens": int(L),
                                     "phase": tag}, sort_keys=True))
            k += 1
    Path(path).write_text("\n".join(lines) + "\n")
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gen_xcheck_trace(path, header):
    """The U7 xcheck requests as a trace (fit pre-check only; mg_xcheck.py sends them)."""
    import mg_xcheck as XC
    idx = XT.doc_index(header)
    lines = [json.dumps({"record": "request", "request_id": f"xcheck-{d}", "arrival_s": 0.0, "doc_id": d, "input_tokens": idx[d][1],
                         "output_tokens": C.XC_MAX_NEW, "phase": "xcheck"}, sort_keys=True) for d in XC.select_docs(header)]
    Path(path).write_text("\n".join(lines) + "\n")


def needs_x_tokens(spec):
    for st in spec.get("steps", []):
        if st["type"] in ("cell", "warmup") and st.get("shape") == "X":
            return True
        if st["type"] == "floor" or (st["type"] == "cmd" and "{XTOKENS}" in " ".join(st.get("argv", []))):
            return True
    return False


def load_x_header(spec):
    """(header, problem).  The token file must exist and its sha256 must equal spec x_tokens_sha256 (rev12 ABORT_DATASET)."""
    path, want = spec.get("x_tokens_path"), spec.get("x_tokens_sha256")
    if not path or not Path(path).is_file():
        return None, f"X token file absent ({path})"
    got = XT.sha256_file(path)
    if not want or got != want:
        return None, f"X token file sha256 {got[:16]} != registered {str(want)[:16]}"
    try:
        h = XT.read_header(path)
    except ValueError as e:
        return None, repr(e)
    if h.get("stats", {}).get("max_input_plus_output", 0) > C.CONTEXT_LENGTH - CL.TOTAL_SLACK:
        return None, f"X max input+output {h['stats']['max_input_plus_output']} > ctx-{CL.TOTAL_SLACK}"
    return h, None


def precheck_traces(spec, boot_dir, header):
    """Write every request of this boot (warm-up, cells, floor, xcheck) as traces under <boot>/precheck/; returns the paths."""
    d = Path(boot_dir) / "precheck"
    d.mkdir(parents=True, exist_ok=True)
    n_gpus = max(int(spec["tp"]), int(spec["dp"]))
    out, shapes = [], set()
    for i, st in enumerate(spec.get("steps", [])):
        p = d / f"step{i:03d}.jsonl"
        if st["type"] == "cell":
            shapes.add(st["shape"])
            if st["shape"] == "X":
                gen_trace_x(p, float(st["offered"]), int(st["n_req"]), int(st["seed"]), header)
            else:
                gen_trace(p, st["shape"], float(st["offered"]), int(st["n_req"]), int(st["seed"]))
        elif st["type"] == "warmup":
            shapes.add(st["shape"])
            continue
        elif st["type"] == "floor":
            gen_floor_trace(p, st["lengths"], st["repeats"], st["max_new"])
        elif st["type"] == "cmd" and "{XTOKENS}" in " ".join(st.get("argv", [])):
            gen_xcheck_trace(p, header)
        else:
            continue
        out.append(p)
    for sh in sorted(shapes):                                       # warm-up traces exactly as run_warmup generates them
        p = d / f"warmup_{sh}.jsonl"
        _warmup_trace(p, sh, n_gpus, header)
        out.append(p)
    return out


def config_span_budget(config_abs):
    m = re.search(r"^split_forward_token_budget:\s*(\d+)\s*$", Path(config_abs).read_text(), re.M)
    return int(m.group(1)) if m else None


def rev12_prechecks(spec, boot_dir, cfg_abs, dry):
    """(rc, record).  rc 0 = pass.  CPU only, before any GPU work."""
    rec = {"context_length": C.CONTEXT_LENGTH}
    header = None
    if needs_x_tokens(spec):
        header, prob = load_x_header(spec)
        rec["x_tokens"] = {"path": spec.get("x_tokens_path"), "sha256": spec.get("x_tokens_sha256"), "problem": prob}
        if prob:
            if dry and not (spec.get("x_tokens_path") and Path(spec["x_tokens_path"]).is_file()):
                rec["x_tokens"]["dryrun_note"] = "X token file absent in a dry run: fit pre-check of X/floor/xcheck NOT done"
            else:
                rec["abort"] = f"ABORT_DATASET {prob}"
                return RC_DATASET, rec
    if spec["arm"] in C.PDMUX_ARMS:
        bud = config_span_budget(cfg_abs)
        rec["config_sha256"] = hashlib.sha256(Path(cfg_abs).read_bytes()).hexdigest()
        rec["split_forward_token_budget"] = bud
        if bud != C.SPLIT_FORWARD_TOKEN_BUDGET:
            rec["abort"] = f"ABORT_CONFIG_DRIFT split_forward_token_budget {bud} != {C.SPLIT_FORWARD_TOKEN_BUDGET} ({cfg_abs})"
            return RC_CONFIG_DRIFT, rec
    if header is None and needs_x_tokens(spec):
        steps = [st for st in spec.get("steps", []) if not (st.get("shape") == "X" or st["type"] == "floor"
                                                           or (st["type"] == "cmd" and "{XTOKENS}" in " ".join(st.get("argv", []))))]
        traces = precheck_traces(dict(spec, steps=steps), boot_dir, None)
    else:
        traces = precheck_traces(spec, boot_dir, header)
    if traces or spec.get("steps"):
        problems = CL.fit_problems(C.CONTEXT_LENGTH, traces) if traces else []
        rec["fit_traces"] = len(traces)
        rec["fit_problems"] = problems
        if problems:
            rec["abort"] = "ABORT_CONTEXT_TOO_SHORT " + "; ".join(problems)[:600]
            return RC_CONTEXT, rec
    return 0, rec


def check_server_info_knobs(spec, server_info_path):
    """§19-3: the server's own record of the registered knobs (an ARGUMENT echo, not the effective dtype -- F-4).
    Returns (drift list, unverified flag)."""
    try:
        si = json.loads(Path(server_info_path).read_text())
    except (OSError, ValueError):
        return [], True
    if not isinstance(si, dict) or "error" in si:
        return [], True
    want = {"context_length": C.CONTEXT_LENGTH, "max_prefill_tokens": C.MAX_PREFILL_TOKENS,
            "max_mamba_cache_size": effective_cap(int(spec["tp"]), int(spec["dp"])), "mamba_ssm_dtype": spec.get("mamba_ssm_dtype")}
    drift = [f"{k}: server {si.get(k)!r} != registered {v!r}" for k, v in want.items() if k in si and si.get(k) != v]
    missing = [k for k in want if k not in si]
    return drift, bool(missing)


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


def write_alive_step(boot_dir, boot_id, server, port, spec_steps, steps_done):
    """2026-10-13 repair (J4/J8 rev11): survival record right after a CMD step -> alive_after_<step name>.json (same schema,
    after_step.name = the step name).  G1 '비정상 종료 0' reads the record of the boot's G1 step, so a server that dies in a
    LATER step (G2 on the fused boot) is not charged to G1.  Only cmd steps (no extra /health request between M3 cells).
    A write failure leaves the file absent -> the G1 reader says NOT_TESTED, never alive."""
    try:
        import m2g_correctness_gate as GATE
        last = steps_done[-1]
        name = last["step"].get("name", "x")
        after = {"index": len(steps_done) - 1, "type": last["step"]["type"], "status": last["status"], "name": name}
        GATE.record_alive_after(boot_dir, boot_id, server, f"http://127.0.0.1:{port}", after, len(spec_steps),
                                record_name=GATE.alive_step_record_name(name))
    except Exception as e:  # noqa: BLE001
        try:
            (Path(boot_dir) / "alive_after_step.error").write_text(repr(e) + "\n")
        except OSError:
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
    port = 0 if dry else (int(spec["port"]) if spec.get("port") else pick_port())      # concurrent_group boots get preassigned ports
    cfg_rel = spec.get("config_rel") or (C.AGNOSTIC_CONFIG_REL if spec["arm"] == "agnostic" else C.D44_CONFIG_REL)
    cfg_abs = str(ENGINE / cfg_rel)
    try:
        sargs = build_server_args(spec, port or 0, cfg_abs if spec["arm"] in C.PDMUX_ARMS else None)
    except ValueError as e:
        print(f"spec error: {e}", file=sys.stderr)
        boot["status"] = "SPEC_ERROR"
        boot["error"] = repr(e)
        jdump(boot_dir / "boot.json", boot)
        return 2
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
    # ---- rev12 CPU pre-checks (context fit of every request, X token file, span budget) -- before any GPU work, dry run included
    if spec.get("kind") in M3_KINDS:
        prc, pre = rev12_prechecks(spec, boot_dir, cfg_abs, dry or os.environ.get("PDMUX_MG_DRYRUN") == "1")
        jdump(boot_dir / "precheck.json", pre)
        boot["precheck"] = {k: v for k, v in pre.items() if k != "fit_problems"}
        if prc:
            boot["status"] = pre["abort"].split()[0]
            boot["abort"] = pre["abort"]
            jdump(boot_dir / "boot.json", boot)
            return prc
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
        if spec.get("kind") in M3_KINDS:
            import mg_m2g_reference as MR
            jdump(boot_dir / "mamba_facts.json", MR.mamba_facts(log_path.read_text(errors="replace")))     # F-4 effective-dtype witness
            drift, unverified = check_server_info_knobs(spec, boot_dir / "server_info.json")
            boot["knobs_server_info"] = {"drift": drift, "unverified": unverified, "mamba_ssm_dtype_arg": spec.get("mamba_ssm_dtype")}
            if drift:
                boot["status"] = "ABORT_CONFIG_DRIFT"
                boot["abort"] = "ABORT_CONFIG_DRIFT " + "; ".join(drift)
                rc = RC_CONFIG_DRIFT
                return rc
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
                if st["type"] == "cmd":                # the step's own survival record says dead (not absent)
                    write_alive_step(boot_dir, spec["boot_id"], server, port, spec.get("steps", []), steps_done)
                continue
            if st["type"] == "warmup":
                steps_done.append({"step": st, "status": run_warmup(spec, st, boot_dir, port, n_gpus, deadline)})
                warmed.add(st["shape"])
            elif st["type"] == "cell":
                if st["shape"] not in warmed:      # a cell is never the first traffic of its shape
                    run_warmup(spec, {"shape": st["shape"]}, boot_dir, port, n_gpus, deadline)
                    warmed.add(st["shape"])
                steps_done.append({"step": st, "status": run_cell(spec, st, boot_dir, out_root, port, server.pid, deadline)})
            elif st["type"] == "floor":
                steps_done.append({"step": st, "status": run_floor(spec, st, boot_dir, port, deadline)})
            elif st["type"] == "cmd":
                steps_done.append({"step": st, "status": run_cmd(st, boot_dir, out_root, port, deadline, spec)})
                write_alive_step(boot_dir, spec["boot_id"], server, port, spec.get("steps", []), steps_done)   # right after THIS step
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
        if not spec.get("concurrent_group"):          # the global `pkill ^sglang::` would kill the sibling servers of a concurrent group
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


def _warmup_trace(path, shape, n_gpus, header):
    n = min(64, C.WARMUP_REQS_PER_GPU * n_gpus)
    if shape == "X":
        return gen_trace_x(path, 4.0, n, 7, header, tag="warm")          # seed 7: not a registered seed (no permutation sha)
    return gen_trace(path, shape, 4.0, n, 7, tag="warm")                 # 4 req/s: spaced, far below capacity


def _x_args(spec):
    return ["--x-tokens", str(spec["x_tokens_path"]), "--x-tokens-sha256", str(spec.get("x_tokens_sha256") or "")] \
        if spec.get("x_tokens_path") else []


def run_warmup(spec, st, boot_dir, port, n_gpus, deadline):
    shape = st["shape"]
    wd = boot_dir / f"warmup_{shape}"
    wd.mkdir(exist_ok=True)
    header = XT.read_header(spec["x_tokens_path"]) if shape == "X" else None
    _warmup_trace(wd / "trace.jsonl", shape, n_gpus, header)
    cmd = [sys.executable, str(HERE / "mg_loadgen.py"), "--endpoint", f"http://127.0.0.1:{port}/generate",
           "--trace", str(wd / "trace.jsonl"), "--cell-dir", str(wd)] + (_x_args(spec) if shape == "X" else [])
    tmo = 900 if not deadline else max(1.0, deadline - time.time())
    p = subprocess.run(cmd, timeout=tmo, capture_output=True, text=True, check=False)
    (wd / "loadgen.log").write_text(p.stdout + p.stderr)
    return "OK" if p.returncode == 0 else f"WARMUP_RC_{p.returncode}"


def run_cell(spec, st, boot_dir, out_root, port, server_pid, deadline):
    cid = f"{spec['boot_id']}__{st['shape']}_{st['rung']}{'x' if st.get('ext') else ''}_s{st['seed']}"
    cd = out_root / "cells" / cid
    cd.mkdir(parents=True, exist_ok=True)
    if st["shape"] == "X":
        thash = gen_trace_x(cd / "trace.tmp.jsonl", float(st["offered"]), int(st["n_req"]), int(st["seed"]),
                            XT.read_header(spec["x_tokens_path"]))
    else:
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
           "--trace", str(cd / "trace.tmp.jsonl"), "--cell-dir", str(cd)] + (_x_args(spec) if st["shape"] == "X" else [])
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


def floor_summary(reqs, lengths, repeats):
    """FLOOR.json body: per length the median TTFT (s) of the uncontended requests.  Fail-closed: a length with any error request or
    fewer than `repeats` successes has floor_s None (never a median over a partial set)."""
    per = {}
    for L in lengths:
        rs = [r for r in reqs if int(r.get("input_tokens", -1)) == int(L)]
        ok = [r for r in rs if not r.get("error") and r.get("ttft_ms") is not None]
        e_srv = sum(1 for r in rs if r.get("error") and r.get("error_side") == "server")
        complete = len(ok) == int(repeats) and len(rs) == int(repeats)
        per[str(int(L))] = {"n": len(rs), "n_ok": len(ok), "server_errors": e_srv, "client_errors": len(rs) - len(ok) - e_srv,
                            "floor_s": (statistics.median(float(r["ttft_ms"]) for r in ok) / 1000.0) if complete else None,
                            "complete": complete}
    return {"schema": "mg_floor/v1", "lengths": [int(x) for x in lengths], "repeats": int(repeats), "points": per,
            "complete": all(v["complete"] for v in per.values())}


def run_floor(spec, st, boot_dir, port, deadline):
    """rev12 §19-4: the floor block, sent through mg_loadgen (-> trace_loadgen.replay / StreamState, one request at a time)."""
    fd = Path(boot_dir) / "floor"
    fd.mkdir(exist_ok=True)
    gen_floor_trace(fd / "trace.jsonl", st["lengths"], st["repeats"], st["max_new"])
    cmd = [sys.executable, str(HERE / "mg_loadgen.py"), "--endpoint", f"http://127.0.0.1:{port}/generate",
           "--trace", str(fd / "trace.jsonl"), "--cell-dir", str(fd), "--sequential"] + _x_args(spec)
    tmo = None if not deadline else max(1.0, deadline - time.time())
    try:
        p = subprocess.run(cmd, timeout=tmo, capture_output=True, text=True, check=False)
        (fd / "loadgen.log").write_text(p.stdout + p.stderr)
        rc = p.returncode
    except subprocess.TimeoutExpired:
        rc = -9
    reqs = []
    if (fd / "requests.jsonl").exists():
        reqs = [json.loads(x) for x in (fd / "requests.jsonl").read_text().splitlines() if x.strip()]
    doc = dict(floor_summary(reqs, st["lengths"], st["repeats"]), boot_id=spec["boot_id"], config=spec["config"], arm=spec["arm"],
               block=spec.get("block", "main"), loadgen_rc=rc, max_new=int(st["max_new"]), floor_doc_id=C.FLOOR_DOC_ID)
    raw = (json.dumps(doc, indent=1, sort_keys=True) + "\n").encode()
    (Path(boot_dir) / "FLOOR.json").write_bytes(raw)
    (Path(boot_dir) / "FLOOR.json.sha256").write_text(hashlib.sha256(raw).hexdigest() + "\n")
    return "OK" if rc == 0 and doc["complete"] else f"FLOOR_INCOMPLETE_RC_{rc}"


def run_cmd(st, boot_dir, out_root, port, deadline, spec=None):
    subs = {"{ENDPOINT}": f"http://127.0.0.1:{port}", "{PORT}": str(port), "{OUT}": str(out_root),
            "{BOOT}": str(boot_dir), "{XTOKENS}": str((spec or {}).get("x_tokens_path") or "")}
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
