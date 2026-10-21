#!/usr/bin/env python3
"""Job driver for the split engine gate (G-alpha / G-beta / G-gamma-A / G-gamma-F).

Design (registration): results/engine_gate/GATE_SPLIT_DESIGN_2026-10-20.md.  Scorers: engine_gate_checks.py.
STATUS: CPU-tested against a fake server only (tests/test_engine_gate_driver.py).  NOTHING HERE HAS RUN ON A GPU.
Writing this driver is not a PASS of anything; no performance claim is made or implied.

  engine_gate_driver.py alpha|beta|gamma [--family A|F]     run (inside a Job; entry scripts engine_gate_*.sh call this)
  engine_gate_driver.py score alpha|beta|gamma DIR [...]    re-score an exported result directory (CPU)
  engine_gate_driver.py plan-info alpha|beta|gamma [--family A|F] [--rate USD_H] [--hardcap-min N]   (used by launch.sh)
  engine_gate_driver.py pool-gamma MAIN_DIR EXT_DIR         pool the registered one-time extension pairs (n=16 look)

Protocol facts enforced here (each is a stage/check in the output, never a silent default):
  * every stage writes status/<stage>.json and the whole verdict is re-written to GATE_RESULT.json after EVERY stage
    (a truncated export still carries the verdict-so-far; the Job's DONE marker is written by job_entry.sh afterwards);
  * per-stage hard timeout, Job budget = hardcap - reserve; a stage that no longer fits is SKIPPED (-> UNRESOLVED);
  * ENGINE_DIGEST checked 3 times (alpha/beta: initial, after reverse-apply, after re-apply; gamma: start, middle, end);
  * the int64 patch is toggled ONLY by `patch`; each tree state has its own TRITON_CACHE_DIR;
  * OFF boots are asserted to be really OFF, ON boots really ON, and the server interpreter's sglang must come from the
    installed tree (audit r1 MAJOR 4a);
  * a rebooted server that dies is an EXPECTED observation in G-beta only (beta-2); anywhere else it is FAIL.
Test hooks (PDMUX_GATE_*) are honoured ONLY when PDMUX_GATE_TEST_MODE=1.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import http.client
import importlib.util
import json
import math
import os
import random
import re
import shlex
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENG = HERE.parents[1]
PROJECT = ENG.parents[1]
for _p in (HERE, ENG / "scripts" / "multigpu", ENG / "results" / "multigpu", ENG / "benchmarks"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import engine_gate_checks as GC  # noqa: E402
import longctx_probe_helper as H  # noqa: E402
import m2g_correctness_gate as M  # noqa: E402

PASS, FAIL, UNRESOLVED, SKIPPED = "PASS", "FAIL", "UNRESOLVED", "SKIPPED"
HARNESS_UNRESOLVED = "HARNESS_UNRESOLVED"
RC = {"DIAGNOSTIC_COMPLETE": 0, "DIAGNOSTIC_INCOMPLETE": 4, "PASS": 0, "FAIL": 1, "UNRESOLVED": 4, "ABORT_MODEL_REVISION": 12, "ABORT_ENGINE_DIGEST": 13, "ABORT_REG": 16}

# ---------------------------------------------------------------- registered constants
DIGEST_BASE = "99bbef91fe680b4d4b8a8fd0a08b8cc65426518c99baca0b3e18e4fa627f4325"   # dev_tree_edits item 35 (30 entries)
DIGEST_REV = "9af2f2eff337514e0a7c27ad3f83a6a75a699df6850014875f3d27ece11a6bb5"    # int64 patch only reversed
REG_SHA = "75236e5cb01e702f32f003580d29e169f17ec10287385651b0847c2ca3511ea8"         # engine_gate_checks.REG after audit r1 + r2 + decision 20 (beta unit criteria v3)
NANO = "nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base"
NANO_REV = "dc0661c829b14e5b9246c05cfa89094a0875e052"
LLAMA, LLAMA_REV = H.LLAMA_ID, H.LLAMA_REV
CONV_REL = "sglang/srt/layers/attention/mamba/causal_conv1d_triton.py"
CONV_MARK = "int64: token offsets"
PRISTINE_SHA256 = "1700fae290fb1c28d6968f1f2732067aeba26e92898653f3ec6fcbed3c24e5cb"
CAP, SEED, MEM_FRACTION, CTX_LONG = 48, 1, 0.6, 131072
LS_B = (4096, 16384, 32768, 65536)               # alpha (b) sanity lengths
ISO_N, ISO_IN, ISO_OUT = 48, 256, 1024
STAIR_N = 40
CAP_PROBE_N, CAP_PROBE_IN, CAP_PROBE_OUT = CAP + 1, 64, 256
HI_RATE, LO_RATE, HI_N, LO_N = 4.784, 1.794, 810, 300
GAMMA_SEEDS = (4386, 4162, 251, 2630, 1001, 2002, 3003, 4004, 5005, 6006)
GAMMA_OO_SEEDS = (4386, 4162)     # r2 diagnostic OFF/OFF pairs reuse the seeds (= traces) of ON/OFF pairs 1-2; pair ids 101, 102
GAMMA_EXT_SEEDS = (7007, 8008, 9009, 1111, 2222, 3333)     # [proposal] registered with the one-time extension (n=16)
SHAPE_A = (256, 512)
JOBS = {  # name -> (expected hours, hard cap minutes)   design sec 2
    "alpha": (1.2, 120), "beta": (1.0, 120), "gamma": (3.4, 270), "decomp": (3.6, 250)}
RESERVE_S = 600
EST = {"boot": 150, "s_pass": 220, "c_pass": 60, "long": 120, "iso": 120, "stair": 90, "cap": 60, "td": 40,
       "unit_case": 60, "inst": 650}
TMO = {"boot": 900, "s_pass": 600, "c_pass": 300, "long": 900, "iso": 600, "stair": 600, "cap": 600, "td": 120,
       "unit_case": 600, "beta_long": 900, "cell": 420, "warm": 180, "death": 900}
def _decomp():
    """The decomposition module (engine_gate_decomp.py), bound to THIS driver module object."""
    m = __import__("engine_gate_decomp")
    class _Live:      # live view of THIS module's globals (sys.modules[__name__] may be a re-imported copy in tests)
        def __getattr__(self, k):
            try:
                return globals()[k]
            except KeyError:
                raise AttributeError(k) from None
    m.bind(_Live())
    return m


def _is_test():
    return os.environ.get("PDMUX_GATE_TEST_MODE") == "1"


def tget(name, default=None):
    """Test hook: env PDMUX_GATE_<name> is honoured only in test mode."""
    return os.environ.get("PDMUX_GATE_" + name, default) if _is_test() else default


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def dump(obj, path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str) + "\n")
    os.replace(tmp, p)


def load(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def read_jsonl(path):
    rows = []
    try:
        for line in Path(path).read_text(errors="replace").splitlines():
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    pass          # partial last line of a killed writer
    except OSError:
        pass
    return rows


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class GateAbort(Exception):
    def __init__(self, code, why):
        super().__init__(f"{code}: {why}")
        self.code, self.why = code, why


# ---------------------------------------------------------------- host effects (injectable in test mode)
def gpu_mem_mib():
    cmd = tget("GPU_MEM_CMD")
    try:
        out = subprocess.run(shlex.split(cmd) if cmd else
                             ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"],
                             capture_output=True, text=True, timeout=20).stdout.strip().splitlines()
        return int(float(out[0])) if out else None
    except Exception:  # noqa: BLE001
        return None


class MemSampler(threading.Thread):
    """Max memory.used over a window (design beta-1 (e)).  100 ms with pynvml, else 1 s with nvidia-smi."""

    def __init__(self, period=None):
        super().__init__(daemon=True)
        self.stop_ev, self.max_mib, self.n = threading.Event(), None, 0
        self.read, self.period = gpu_mem_mib, period or 1.0
        if not _is_test():
            try:
                import pynvml
                pynvml.nvmlInit()
                h = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.read = lambda: int(pynvml.nvmlDeviceGetMemoryInfo(h).used // (1 << 20))
                self.period = period or 0.1
            except Exception:  # noqa: BLE001
                pass

    def run(self):
        while not self.stop_ev.is_set():
            v = self.read()
            if v is not None:
                self.max_mib = v if self.max_mib is None else max(self.max_mib, v)
                self.n += 1
            self.stop_ev.wait(self.period)

    def finish(self):
        self.stop_ev.set()
        self.join(timeout=5)
        return {"max_used_mib": self.max_mib, "samples": self.n, "period_s": self.period}


# ---------------------------------------------------------------- engine tree: digest and int64 patch toggle
def tree_path():
    return Path(tget("TREE") or os.environ.get("SGLANG_ENGINE_DEV") or "/opt/pdmux/sglang_engine_dev/python")


def patch_file():
    return Path(tget("PATCH_FILE") or ENG / "src" / "patches" / "triton_causal_conv1d_int64.patch")


def expected_digest(state):
    base = tget("DIGEST_BASE") or DIGEST_BASE
    rev = tget("DIGEST_REV") or DIGEST_REV
    return base if state == "applied" else rev


def conv_state(tree):
    f = Path(tree) / CONV_REL
    if not f.is_file():
        return "missing"
    return "applied" if CONV_MARK in f.read_text() else "reversed"


def tree_digest(tree):
    r = subprocess.run([sys.executable, str(ENG / "scripts" / "bootstrap" / "engine_digest.py"), "digest",
                        "--runtime", str(tree)], capture_output=True, text=True, timeout=120)
    out = r.stdout.strip().split()
    return (out[-1] if r.returncode == 0 and out else None), (r.stderr or r.stdout)[-300:]


def _patch(tree, direction):
    with open(patch_file(), "rb") as fh:
        return subprocess.run(["patch", direction, "--batch", "-p1", "-d", str(tree)], stdin=fh, capture_output=True,
                              timeout=120)


def reverse_patch(tree):
    if conv_state(tree) != "applied":
        raise GateAbort("ABORT_PATCH_STATE", f"reverse requested but conv state is {conv_state(tree)}")
    r = _patch(tree, "--reverse")
    if r.returncode != 0 or conv_state(tree) != "reversed":
        raise GateAbort("ABORT_PATCH_STATE", "reverse apply failed: " + (r.stderr or r.stdout).decode()[-200:])


def reapply_patch(tree):
    if conv_state(tree) == "applied":
        return
    r = _patch(tree, "--forward")
    if r.returncode != 0 or conv_state(tree) != "applied":
        raise GateAbort("ABORT_PATCH_STATE", "re-apply failed: " + (r.stderr or r.stdout).decode()[-200:])


# ---------------------------------------------------------------- the gate object: stages, budget, result file
class Gate:
    def __init__(self, name, root, hardcap_min, family=None, clock=time.monotonic):
        self.name, self.root, self.family = name, Path(root), family
        self.hardcap_s = int(hardcap_min * 60)
        self.budget_s = max(60, self.hardcap_s - RESERVE_S)
        self.clock, self.t0 = clock, clock()
        self.stages = OrderedDict()
        self.checks = OrderedDict()
        self.digests = []
        self.meta = {"job": os.environ.get("PDMUX_JOB_NAME", "local"), "commit": os.environ.get("PDMUX_COMMIT", ""),
                     "image": os.environ.get("PDMUX_IMAGE", ""), "started_utc": utc(), "hardcap_min": hardcap_min}
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "status").mkdir(exist_ok=True)
        self.final = None

    def elapsed(self):
        return self.clock() - self.t0

    def remaining(self):
        return self.budget_s - self.elapsed()

    def fits(self, secs):
        return self.elapsed() + secs <= self.budget_s

    def record(self, stage, status, reason="", facts=None):
        self.stages[stage] = {"stage": stage, "status": status, "reason": reason, "facts": facts or {},
                              "t_end_utc": utc(), "elapsed_s": round(self.elapsed(), 1)}
        dump(self.stages[stage], self.root / "status" / f"{stage}.json")
        self.flush()
        print(f"[GATE {utc()}] {stage}: {status} -- {reason}", flush=True)

    def stage(self, name, fn, est_s, tmo_s):
        """Run fn() -> (status, reason, facts) under a hard timeout; SKIPPED when the budget cannot hold est_s."""
        if not self.fits(est_s):
            self.record(name, SKIPPED, f"budget: {self.remaining():.0f} s left < estimated {est_s} s")
            return SKIPPED
        box = {}

        def work():
            try:
                box["res"] = fn()
            except GateAbort as ga:
                box["abort"] = ga
            except Exception as exc:  # noqa: BLE001
                box["exc"] = repr(exc)[:400]

        th = threading.Thread(target=work, daemon=True)
        th.start()
        th.join(min(tmo_s, max(5, self.remaining() + RESERVE_S // 2)))
        if th.is_alive():
            self.record(name, UNRESOLVED, f"TIMEOUT after {tmo_s} s (stage abandoned)")
            return UNRESOLVED
        if "abort" in box:
            self.record(name, FAIL, str(box["abort"]))
            raise box["abort"]
        if "exc" in box:
            self.record(name, UNRESOLVED, "HARNESS_EXCEPTION " + box["exc"])
            return UNRESOLVED
        status, reason, facts = box["res"]
        self.record(name, status, reason, facts)
        return status

    def check_digest(self, tag, state):
        got, err = tree_digest(tree_path())
        exp = expected_digest(state)
        rec = {"n": len(self.digests) + 1, "tag": tag, "state": state, "got": got, "expected": exp, "ok": got == exp}
        self.digests.append(rec)
        dump(self.digests, self.root / "digests.json")
        self.record(f"digest_{rec['n']}_{tag}", PASS if rec["ok"] else FAIL,
                    f"{state}: {got} vs expected {exp}" + ("" if got else f" ({err})"), rec)
        if not rec["ok"]:
            raise GateAbort("ABORT_ENGINE_DIGEST", f"{tag}: got {got} expected {exp}")

    def verdict(self):
        if self.final:
            return self.final
        v = rollup([c["verdict"] for c in self.checks.values()] + [s["status"] for s in self.stages.values()
                                                                       if s["status"] != PASS])
        return v

    def flush(self, final=None):
        res = {"gate": self.name, "family": self.family, "meta": self.meta, "elapsed_s": round(self.elapsed(), 1),
               "complete": final is not None, "verdict": final or "INCOMPLETE",
               "verdict_so_far": rollup([s["status"] for s in self.stages.values()]),
               "stages": {k: {"status": v["status"], "reason": v["reason"]} for k, v in self.stages.items()},
               "checks": self.checks, "digests": self.digests, "reg_sha256": GC.reg_sha256(),
               "note": "a scorer verdict is a harness result on this Job's data, not a performance claim"}
        dump(res, self.root / "GATE_RESULT.json")
        return res


def rollup(vals):
    vals = [v for v in vals if v not in ("UNINFORMATIVE", "RECORDED", "NOT_APPLICABLE", "DIAGNOSTIC", "DIAG_SKIPPED")]
    if any(v == FAIL for v in vals):
        return FAIL
    if vals and all(v == PASS for v in vals):
        return PASS
    return UNRESOLVED


# ---------------------------------------------------------------- server: argv, env, boot, teardown
ARM_MODEL = {"agnostic": (NANO, NANO_REV), "fused_sym": (NANO, NANO_REV), "llama": (LLAMA, LLAMA_REV)}
UNSET_KNOBS = ("SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN", "PDMUX_SLO_SCHED", "PDMUX_SLO_MODE", "PDMUX_LA_COORD",
               "PDMUX_FIXED_DECODE_SM_FILE", "PDMUX_TRACE_FORCE_PREFILL", "PDMUX_DUAL_WORKER_TRACE_EVERY",
               "PDMUX_TRUE_DUAL_WORKER", "PDMUX_DUAL_WORKER", "PDMUX_STICKY_PARTITION", "PDMUX_R2_POLICY",
               "PDMUX_R2_FIXED_DSM", "PDMUX_MODEL_PROFILE", "PDMUX_ENGINE_MODE", "PDMUX_ADMISSION_PATH",
               "PDMUX_DECODE_PROBE_PATH", "PDMUX_DECODE_PROBE_GPU_EVENTS", "PDMUX_TELEMETRY_PATH", "PDMUX_PHASE_EVENTS",
               "CUDA_LAUNCH_BLOCKING")


def server_argv(arm, ctx_len=None, eager=False):
    model, rev = ARM_MODEL[arm]
    a = ["--model-path", model, "--revision", rev, "--trust-remote-code", "--dtype", "bfloat16",
         "--attention-backend", "flashinfer", "--disable-radix-cache", "--mem-fraction-static", str(MEM_FRACTION),
         "--max-running-requests", str(CAP), "--random-seed", str(SEED), "--disable-piecewise-cuda-graph",
         "--host", "127.0.0.1", "--chunked-prefill-size", "-1", "--disable-overlap-schedule"]
    if ctx_len:
        a += ["--context-length", str(ctx_len)]
    if arm != "llama":
        a += ["--max-mamba-cache-size", str(CAP)]
    if arm == "agnostic":
        a += ["--enable-pdmux", "--pdmux-config-path", str(ENG / "benchmarks" / "configs" / "pdmux_agnostic_manual.yml")]
    if eager:
        a += ["--disable-cuda-graph"]
    return a


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def http_json(port, path, body=None, timeout=60, method=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        c.request(method or ("POST" if body is not None else "GET"), path,
                  json.dumps(body) if body is not None else None, {"Content-Type": "application/json"})
        r = c.getresponse()
        data = r.read()
        return r.status, data
    finally:
        c.close()


def gen_ids(port, ids, max_new, timeout, ignore_eos=True):
    """Non-streaming greedy generation from input_ids.  -> {ok, out, error, prompt_tokens, completion_tokens}."""
    rec = {"ok": False, "out": [], "error": None, "t_send": time.perf_counter()}
    try:
        st, data = http_json(port, "/generate", {"input_ids": list(ids), "sampling_params": {
            "max_new_tokens": int(max_new), "temperature": 0.0, "ignore_eos": ignore_eos}}, timeout=timeout)
        if st != 200:
            rec["error"] = f"HTTP {st}: {data[:200]!r}"
        else:
            j = json.loads(data)
            if isinstance(j, list):
                j = j[0]
            mi = j.get("meta_info") or {}
            rec.update(out=list(j.get("output_ids") or []), prompt_tokens=mi.get("prompt_tokens"),
                       completion_tokens=mi.get("completion_tokens"))
            rec["ok"] = len(rec["out"]) == int(max_new) and mi.get("prompt_tokens") in (None, len(ids))
            if not rec["ok"]:
                rec["error"] = f"output length {len(rec['out'])} != {max_new}"
    except Exception as exc:  # noqa: BLE001
        rec["error"] = repr(exc)[:300]
    rec["t_end"] = time.perf_counter()
    return rec


class Server:
    def __init__(self, gate, boot_id, spec, tree_tag):
        self.g, self.id, self.spec, self.tag = gate, boot_id, spec, tree_tag
        self.dir = getattr(gate, "raw_root", gate.root) / boot_id    # decomp: bulky raw data lives off the exported tree
        self.dir.mkdir(parents=True, exist_ok=True)
        self.proc, self.port, self.step_i = None, None, 0
        self.log = self.dir / "server.log"

    def env(self):
        e = {k: v for k, v in os.environ.items() if k not in UNSET_KNOBS}
        spec = self.spec
        if spec["arm"] != "agnostic":
            e["PDMUX_ENGINE_MODE"] = "vanilla"
        if spec.get("level_env") is not None:     # decomp: explicit per-level probe env ({dir} -> this boot's dir)
            e.update({k: v.replace("{dir}", str(self.dir)) for k, v in spec["level_env"].items()})
        elif spec["on"]:
            e["PDMUX_ADMISSION_PATH"] = str(self.dir / "admission.jsonl")
            e["PDMUX_DECODE_PROBE_PATH"] = str(self.dir / "decode_probe.jsonl")
            if spec["arm"] == "agnostic":
                e["PDMUX_TELEMETRY_PATH"] = str(self.dir / "tel.jsonl")
                e["PDMUX_PHASE_EVENTS"] = "1"
        base = Path(os.environ.get("PDMUX_GATE_CACHE_BASE") or
                    (Path(os.environ["TRITON_CACHE_DIR"]).parent if os.environ.get("TRITON_CACHE_DIR") else self.g.root / "cache"))
        tdir = base / f"triton_{self.tag}"
        tdir.mkdir(parents=True, exist_ok=True)
        e["TRITON_CACHE_DIR"] = str(tdir)
        e["PDMUX_RUN_ID"] = self.g.meta["job"]
        e["PDMUX_WORKLOAD_ID"] = f"engine_gate_{self.g.name}_{self.id}"
        if spec.get("blocking"):
            e["CUDA_LAUNCH_BLOCKING"] = "1"
        e.update(spec.get("env") or {})
        self.triton_dir = tdir
        return e

    def start(self, boot_timeout=None):
        self.port = free_port()
        cmd = shlex.split(tget("SERVER_CMD") or "") or [sys.executable, "-m", "sglang.launch_server"]
        argv = cmd + server_argv(self.spec["arm"], self.spec.get("ctx"), self.spec.get("eager", False)) + \
            ["--port", str(self.port)]
        env = self.env()
        # records what this boot was (names of PDMUX_* knobs only; the environment is never dumped)
        self.boot = {"boot_id": self.id, "spec": {k: v for k, v in self.spec.items() if k != "env"}, "argv": argv,
                     "tree_tag": self.tag, "tree_state": conv_state(tree_path()),
                     "pdmux_knobs": sorted(k for k in env if k.startswith("PDMUX_")),
                     "triton_cache_dir": str(self.triton_dir), "start_utc": utc()}
        dump(self.boot, self.dir / "boot.json")
        self.lf = open(self.log, "wb")
        self.proc = subprocess.Popen(argv, stdout=self.lf, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        limit = boot_timeout or TMO["boot"]
        t0 = time.monotonic()
        while time.monotonic() - t0 < limit:
            if self.proc.poll() is not None:
                return "died"
            try:
                if http_json(self.port, "/health", timeout=5)[0] == 200:
                    return "ok"
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.5 if _is_test() else 2)
        return "timeout"

    def alive_step(self, step):
        self.step_i += 1
        return M.record_alive_after(self.dir, self.id, self.proc, f"http://127.0.0.1:{self.port}",
                                    after_step={"index": self.step_i, "type": "cmd", "status": "done", "name": step},
                                    record_name=M.alive_step_record_name(step), timeout_s=10)

    def dead(self):
        return self.proc is not None and self.proc.poll() is not None

    def stop(self):
        """SIGTERM the process group, then SIGKILL; wait for GPU memory release.  -> {"released": bool|None, ...}"""
        if self.proc is not None:
            for sig, wait in ((signal.SIGTERM, 20 if not _is_test() else 3), (signal.SIGKILL, 10)):
                try:
                    os.killpg(self.proc.pid, sig)
                except (ProcessLookupError, PermissionError):
                    pass
                t0 = time.monotonic()
                while time.monotonic() - t0 < wait and self.proc.poll() is None:
                    time.sleep(0.2)
                if self.proc.poll() is not None:
                    break
            try:
                self.lf.close()
            except Exception:  # noqa: BLE001
                pass
        limit = float(tget("TD_WAIT_S", 120))
        t0, m = time.monotonic(), None
        while True:
            m = gpu_mem_mib()
            if m is not None and m < 1024:
                break
            if time.monotonic() - t0 > limit:
                break
            time.sleep(0.5 if _is_test() else 5)
        rec = {"memory_used_after_mib": m, "released": (m is not None and m < 1024) if m is not None else None,
               "log_signatures": H.log_signatures(read_text(self.log)), "banner": H.parse_banner(read_text(self.log)),
               "returncode": self.proc.poll() if self.proc else None}
        dump(rec, self.dir / "teardown.json")
        return rec


def read_text(p):
    try:
        return Path(p).read_text(errors="replace")
    except OSError:
        return ""


def sglang_file(env_tree):
    """Where the server's interpreter imports sglang from (audit r1 4a).  Same interpreter and PYTHONPATH as the server."""
    forced = tget("SGLANG_FILE")
    if forced:
        return forced
    try:
        r = subprocess.run([sys.executable, "-c", "import sglang,sys;print(sglang.__file__)"], capture_output=True,
                           text=True, timeout=120)
        return r.stdout.strip().splitlines()[-1] if r.returncode == 0 and r.stdout.strip() else None
    except Exception:  # noqa: BLE001
        return None


def instrumentation_record(srv):
    log = read_text(srv.log)
    files = {"decode_probe": (srv.dir / "decode_probe.jsonl").exists(), "admission": (srv.dir / "admission.jsonl").exists(),
             "telemetry": (srv.dir / "tel.jsonl").exists()}
    tree = tree_path()
    m = re.search(r"decode probe ENABLED \(I-4 events=(True|False)", log)
    return {"on": bool(srv.spec["on"]), "log_decode_enabled": "decode probe ENABLED" in log,
            "log_decode_events": (m.group(1) == "True") if m else None,
            "log_admission_enabled": "admission probe ENABLED" in log, "files": files,
            "sglang_file": sglang_file(tree), "tree": str(tree),
            "arm_pdmux_in_log_irrelevant": None}


def triton_key(srv):
    """Audit r1 4b: what the Triton cache looks like for this boot (dir, entry count, hash of the sorted entry names)
    and the sha of the conv source it was compiled from."""
    d = Path(srv.triton_dir)
    names = sorted(p.name for p in d.iterdir()) if d.is_dir() else []
    conv = tree_path() / CONV_REL
    return {"dir": str(d), "n_entries": len(names), "entries_sha256": hashlib.sha256("\n".join(names).encode()).hexdigest(),
            "conv_source_sha256": sha256_file(conv) if conv.is_file() else None}


class Ctx:
    """Per-Job helpers: tokenizer encode (synthetic inputs), memo of the synthetic id lists (deterministic)."""

    def __init__(self):
        self._enc, self._ids = {}, {}

    def ids(self, model, L, seed):
        k = (model, L, seed)
        if k not in self._ids:
            self._ids[k] = H.synth_ids(self.encoder(model), L, seed)
        return self._ids[k]

    def encoder(self, model):
        if model in self._enc:
            return self._enc[model]
        if _is_test() and tget("FAKE_ENCODER") == "1":
            fn = lambda text: [1 + (zlib_adler(w) % 30000) for w in text.split()]  # noqa: E731
        else:
            hf = os.environ.get("HF_HOME", "/data/hf")
            _ref, _rev, snap = H.snapshot_dir(hf, model)
            fn, _tok = H.load_encoder(snap)
        self._enc[model] = fn
        return fn


def zlib_adler(s):
    import zlib
    return zlib.adler32(s.encode())


def check_models(gate):
    if _is_test() and tget("SKIP_MODEL_CHECK") == "1":
        return
    hf = os.environ.get("HF_HOME", "/data/hf")
    for model, rev in ((NANO, NANO_REV),) + (((LLAMA, LLAMA_REV),) if gate.name == "alpha" else ()):
        _ref, got, snap = H.snapshot_dir(hf, model)
        if got != rev or not snap or not Path(snap).is_dir():
            raise GateAbort("ABORT_MODEL_REVISION", f"{model}: refs/main {got!r} != registered {rev}")


def write_substrate(gate):
    sub = {"gpu": None, "driver": None, "hostname": socket.gethostname(), "job": gate.meta["job"],
           "commit": gate.meta["commit"], "image": gate.meta["image"], "family": gate.family,
           "tree": str(tree_path()), "sglang_file": sglang_file(tree_path()), "t_utc": utc(),
           "note": "per-Job substrate record (audit r1 4e); job_entry also writes meta/substrate.json + clk.csv"}
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=30).stdout.strip().splitlines()
        if q:
            n, u, d = [x.strip() for x in q[0].split(",")]
            sub.update(gpu={"name": n, "uuid": u}, driver=d)
    except Exception:  # noqa: BLE001
        pass
    dump(sub, gate.root / "substrate.json")


def boot_and_run(gate, bid, spec, tag, steps, ctx):
    """One boot: start -> health -> instrumentation record -> steps (each followed by an alive record) -> teardown.
    steps = [(name, fn(srv) -> (status, reason, facts), est_s, tmo_s)].  Returns (Server, teardown, ok_to_continue)."""
    srv = Server(gate, bid, spec, tag)
    st = gate.stage(f"{bid}_boot", lambda: _boot(srv), EST["boot"], TMO["boot"] + 30)
    td = None
    try:
        if st == PASS:
            for name, fn, est, tmo in steps:
                if srv.dead():
                    gate.record(f"{bid}_{name}", FAIL if not spec.get("death_ok") else SKIPPED, "server dead before the step")
                    continue
                gate.stage(f"{bid}_{name}", lambda fn=fn: fn(srv), est, tmo)
                srv.alive_step(name)
        else:
            for name, *_ in steps:
                gate.record(f"{bid}_{name}", SKIPPED, f"boot {st}")
    finally:
        try:
            srv.alive_step("end")
        except Exception:  # noqa: BLE001
            pass
        td = srv.stop()
        d = load(srv.dir / "boot.json", {})
        d["triton_key"] = triton_key(srv)
        d["end_utc"] = utc()
        dump(d, srv.dir / "boot.json")
        ok = td["released"] is True
        gate.record(f"{bid}_teardown", PASS if ok else UNRESOLVED,
                    f"memory.used after teardown {td['memory_used_after_mib']} MiB",
                    {"released": td["released"]} if ok else {"released": td["released"], "kind": HARNESS_UNRESOLVED})
    return srv, td, td["released"] is True


def _boot(srv):
    r = srv.start()
    if r != "ok":
        return FAIL, f"server did not reach /health: {r} (see {srv.id}/server.log)", {"result": r}
    ins = instrumentation_record(srv)
    dump(ins, srv.dir / "instrumentation.json")
    res = (srv.spec.get("ins_check") or GC.check_instrumentation)(ins)
    return (PASS if res["verdict"] == PASS else FAIL), "; ".join(res["why"]) or "instrumentation level as intended", \
        {"instrumentation": ins}


# ---------------------------------------------------------------- client phases
def st_pass(name, concurrency, max_new=128):
    def fn(srv):
        prompts = M.registered_prompts()
        r = M.http_generate(f"http://127.0.0.1:{srv.port}", prompts, max_new=max_new, concurrency=concurrency)
        dump({"outputs": r["outputs"], "errors": r["errors"], "concurrency": concurrency}, srv.dir / f"{name}.json")
        bad = [k for k, v in r["outputs"].items() if not v]
        return (PASS if not bad and not r["errors"] else FAIL), f"{len(prompts)} prompts, {len(bad)} empty, " \
            f"{len(r['errors'])} errors", {"n": len(prompts)}
    return fn


def st_long(ctx, lengths, reps=2, max_new=64):
    def fn(srv):
        model = ARM_MODEL[srv.spec["arm"]][0]
        out, bad = {}, []
        for L in lengths:
            for rep in range(reps):
                ids = ctx.ids(model, L, H.seed_for(L, rep))
                H.check_budget(len(ids), max_new, srv.spec.get("ctx") or CTX_LONG)
                r = gen_ids(srv.port, ids, max_new, timeout=TMO["long"])
                out[f"L{L}_r{rep}"] = {"L": L, "rep": rep, "ids_sha256": H.ids_sha256(ids), "out": r["out"],
                                       "ok": r["ok"], "error": r["error"]}
                if not r["ok"]:
                    bad.append(f"L{L}_r{rep}")
        dump(out, srv.dir / "long.json")
        return (PASS if not bad else FAIL), f"{len(out)} requests, bad: {bad}", {"bad": bad}
    return fn


def _streams(srv, n, in_len, out_len_fn, ctx, timeout, seed_off=0):
    """n concurrent streamed requests.  -> (recs, t_ref).  Distinct fixed-seed synthetic inputs."""
    model = ARM_MODEL[srv.spec["arm"]][0]
    idss = [ctx.ids(model, in_len, H.seed_for(in_len, seed_off + i)) for i in range(n)]
    t_ref = time.perf_counter()
    recs = [None] * n

    def one(i):
        recs[i] = H.stream_generate(srv.port, idss[i], out_len_fn(i), timeout, t_ref=t_ref)
    ths = [threading.Thread(target=one, args=(i,), daemon=True) for i in range(n)]
    for t in ths:
        t.start()
    for t in ths:
        t.join(timeout + 5)
    return recs, t_ref


def _window_of(recs, t_ref, mode="all_running"):
    """all_running: [last first-token, first last-token] (every request is decoding); span: [first first-token,
    last last-token] (the whole phase; the stair and the cap step change bs on purpose)."""
    firsts = [t_ref + next((t for t, c in r["chunks"] if c >= 1), float("nan")) for r in recs if r and r["chunks"]]
    lasts = [t_ref + r["chunks"][-1][0] for r in recs if r and r["chunks"]]
    if not firsts or any(math.isnan(x) for x in firsts):
        return None
    return [max(firsts), min(lasts)] if mode == "all_running" else [min(firsts), max(lasts)]


def st_window(name, n, in_len, out_len_fn, ctx, seed_off, expect_out_fn, mode="all_running"):
    def fn(srv):
        recs, t_ref = _streams(srv, n, in_len, out_len_fn, ctx, TMO[name] - 30, seed_off)
        wins = load(srv.dir / "windows.json", {})
        w = _window_of(recs, t_ref, mode)
        ok = [bool(r) and H.request_ok(r, in_len, expect_out_fn(i)) for i, r in enumerate(recs)]
        t_all = [t_ref + (r["t_send"] if r else 0) for r in recs]
        wins[name] = w
        wins[name + "_span"] = [min(t_all), time.perf_counter()]
        dump(wins, srv.dir / "windows.json")
        bad = [i for i, o in enumerate(ok) if not o]
        if w is None or w[1] <= w[0]:
            return UNRESOLVED, "no common all-running window (a request produced no tokens, or ranges do not overlap)", {"bad": bad}
        return (PASS if not bad else FAIL), f"{n} streamed requests, {len(bad)} not as forced; window {w[1]-w[0]:.2f} s", {"bad": bad[:10]}
    return fn


def st_green(srv):
    lines = [l for l in read_text(srv.log).splitlines() if re.search(r"green|realized|sm_count|SM", l, re.I)][:40]
    dump({"lines": lines}, srv.dir / "green_readout.json")
    return "RECORDED", f"{len(lines)} candidate green-context/SM lines (recorded, not a pass criterion)", {}


# ---------------------------------------------------------------- G-alpha
def alpha_boots(ctx):
    iso = ("iso", st_window("iso", ISO_N, ISO_IN, lambda i: ISO_OUT, ctx, 1000, lambda i: ISO_OUT), EST["iso"], TMO["iso"])
    stair = ("stair", st_window("stair", STAIR_N, ISO_IN, lambda i: 16 * (i + 1), ctx, 2000, lambda i: 16 * (i + 1), "span"),
             EST["stair"], TMO["stair"])
    capst = ("cap", st_window("cap", CAP_PROBE_N, CAP_PROBE_IN, lambda i: CAP_PROBE_OUT, ctx, 3000, lambda i: CAP_PROBE_OUT, "span"),
             EST["cap"], TMO["cap"])
    spass = ("s_pass", st_pass("s_pass", 1), EST["s_pass"], TMO["s_pass"])
    cpass = ("c_pass", st_pass("c_pass", 32), EST["c_pass"], TMO["c_pass"])
    cctl = ("c_pass_ctl", st_pass("c_pass_ctl", 32), EST["c_pass"], TMO["c_pass"])
    long_ = ("long", st_long(ctx, tuple(int(x) for x in tget("LS_B", ",".join(map(str, LS_B))).split(","))), EST["long"], TMO["long"])
    green = ("green", st_green, 5, 60)
    C = CTX_LONG
    return [
        ("B1", dict(arm="agnostic", on=False, ctx=C), "patched", [spass, cpass, cctl, long_]),
        ("B2", dict(arm="agnostic", on=True, ctx=C), "patched", [spass, cpass, iso, stair, capst, green]),
        ("B3", dict(arm="fused_sym", on=False, ctx=C), "patched", [spass, cpass, cctl, long_]),
        ("B4", dict(arm="fused_sym", on=True, ctx=C), "patched", [spass, cpass, iso, stair, capst]),
        ("B5", dict(arm="llama", on=False, ctx=C), "patched", [spass, cpass, cctl]),
        ("B6", dict(arm="llama", on=True, ctx=C), "patched", [spass, cpass, iso, stair, capst]),
        ("B7", dict(arm="fused_sym", on=True, eager=True, ctx=C), "patched", [stair, iso]),
        ("B8", dict(arm="agnostic", on=False, ctx=C), "reversed", [long_]),
        ("B9", dict(arm="fused_sym", on=False, ctx=C), "reversed", [long_]),
    ]


def run_alpha(gate):
    ctx = Ctx()
    tree = tree_path()
    gate.check_digest("initial", "applied")
    reversed_done = False
    try:
        okc = True
        only = tget("ONLY_BOOTS")
        for bid, spec, tag, steps in alpha_boots(ctx):
            if only and bid not in only.split(","):
                continue
            if tag == "reversed" and not reversed_done:
                reverse_patch(tree)
                reversed_done = True
                gate.check_digest("reversed", "reversed")
            if not okc:
                for name, *_ in steps:
                    gate.record(f"{bid}_{name}", SKIPPED, "previous teardown did not release GPU memory")
                continue
            if not gate.fits(EST["boot"] + sum(s[2] for s in steps) + EST["td"]):
                for name, *_ in steps:
                    gate.record(f"{bid}_{name}", SKIPPED, "budget")
                gate.record(f"{bid}_boot", SKIPPED, "budget")
                continue
            _s, _t, okc = boot_and_run(gate, bid, spec, tag, steps, ctx)
    finally:
        if reversed_done:
            reapply_patch(tree)
    gate.check_digest("reapplied", "applied")
    return score_alpha(gate.root, tree_path())


_MOD = {}


def probe_module(name):
    if name not in _MOD:
        spec = importlib.util.spec_from_file_location(f"gate_{name}", ENG / "src" / "multiplex" / f"{name}.py")
        m = importlib.util.module_from_spec(spec)
        sys.modules[f"gate_{name}"] = m
        spec.loader.exec_module(m)
        _MOD[name] = m
    return _MOD[name]


def block(v):
    """Map a sub-verdict to the roll-up vocabulary."""
    return {"PASS": PASS, "FAIL": FAIL, "UNINFORMATIVE": "UNINFORMATIVE"}.get(v, UNRESOLVED)


def ge(rows, lo, hi, key="t_host_start"):
    return [r for r in rows if lo <= float(r.get(key, -1)) <= hi]


def score_alpha(root, tree):
    root = Path(root)
    chk = OrderedDict()

    def put(name, verdict, **kw):
        chk[name] = {"verdict": verdict, **kw}

    def boot(bid):
        return root / bid

    def outs(bid, f):
        d = load(boot(bid) / f"{f}.json")
        return None if d is None else d["outputs"]

    # (a) sequential greedy ON == OFF, concurrent G1 vs ctl
    for off, on, arm in (("B1", "B2", "agnostic"), ("B3", "B4", "fused_sym"), ("B5", "B6", "llama")):
        so, sn = outs(off, "s_pass"), outs(on, "s_pass")
        if so is None or sn is None:
            put(f"a_spass_{arm}", UNRESOLVED, why="s_pass output missing")
        else:
            r = GC.outputs_identical(so, sn)
            put(f"a_spass_{arm}", r["verdict"], n=r["n"], differing=r["differing"])
        c1, cn, cc = outs(off, "c_pass"), outs(on, "c_pass"), outs(off, "c_pass_ctl")
        if None in (c1, cn, cc):
            put(f"a_cpass_{arm}", UNRESOLVED, why="c_pass output missing")
        else:
            ab = M.abnormal_exit_of(M.alive_after_step(boot(off), "c_pass"), M.alive_after_step(boot(on), "c_pass"))
            r = M.score_g1(M.g1_metrics(c1, cn), M.g1_metrics(c1, cc), abnormal_exit=ab)
            put(f"a_cpass_{arm}", block(r["verdict"]), g1=r["verdict"], m_median=r.get("m_median"), q=r.get("q"),
                abnormal_exit=ab)
    # (b) int64 before/after, <= 64k  (sanity only: the wrap point is ~94.8k)
    for p, q, arm in (("B1", "B8", "agnostic"), ("B3", "B9", "fused_sym")):
        a, b = load(boot(p) / "long.json"), load(boot(q) / "long.json")
        if a is None or b is None:
            put(f"b_int64_le64k_sanity_{arm}", UNRESOLVED, why="long output missing")
        else:
            r = GC.outputs_identical({k: v["out"] for k, v in a.items()}, {k: v["out"] for k, v in b.items()})
            tk = [load(boot(x) / "boot.json", {}).get("triton_key") for x in (p, q)]
            put(f"b_int64_le64k_sanity_{arm}", r["verdict"], n=r["n"], differing=r["differing"], triton_key=tk,
                label="sanity: lengths <= 64k are below the ~94.8k wrap point")
    # per ON boot: emission, I-6, I-4, I-5/I-6, schema
    dmod, amod = probe_module("decode_probe"), probe_module("admission_probe")
    schema_bad = []
    for bid in ("B2", "B4", "B6", "B7"):
        dec = read_jsonl(boot(bid) / "decode_probe.jsonl")
        adm = read_jsonl(boot(bid) / "admission.jsonl")
        w = load(boot(bid) / "windows.json", {})
        for r in dec:
            schema_bad += [(bid, "decode", e) for e in dmod.validate_record(r)]
        for r in adm:
            schema_bad += [(bid, "admission", e) for e in amod.validate_record(r)]
        if not dec:
            for k in ("emission", "i6"):
                put(f"{k}_{bid}", UNRESOLVED, why="no decode_probe records")
            continue
        r = GC.check_emission(dec, adm)
        put(f"emission_{bid}", r["verdict"], **{k: v for k, v in r.items() if k != "verdict"})
        bootrow = next((x for x in dec if x.get("event") == "decode_probe_boot"), {})
        sw = w.get("stair")
        if bid == "B7":
            r = GC.check_i6(dec, bootrow, expect_all_eager_reason="graph_runner_absent")
        elif not sw:
            r = {"verdict": UNRESOLVED, "why": "stair window missing"}
        else:
            r = GC.check_i6(ge([x for x in dec if x.get("event") == "decode_forward"], *sw), bootrow)
        put(f"i6_{bid}", r["verdict"], **{k: v for k, v in r.items() if k != "verdict"})
        if bid == "B7":
            continue
        iw = w.get("iso")
        if not iw:
            put(f"i4_{bid}", UNRESOLVED, why="iso window missing")
        elif bid == "B2":
            ph = [x for x in read_jsonl(boot(bid) / "tel.jsonl") if x.get("event") == "decode_iteration"
                  and iw[0] <= float(x.get("t_launch", -1)) <= iw[1]]
            r = GC.check_i4_pdmux(ge([x for x in dec if x.get("event") == "decode_forward"], *iw), ph)
            put(f"i4_{bid}", r["verdict"], **{k: v for k, v in r.items() if k != "verdict"})
        else:
            r = GC.check_i4_fused([x for x in dec if x.get("event") == "decode_forward"], tuple(iw))
            put(f"i4_{bid}", r["verdict"], **{k: v for k, v in r.items() if k != "verdict"})
        sw, cw = w.get("stair"), w.get("cap_span") or w.get("cap")
        if not sw or not cw:
            put(f"i5i6_{bid}", UNRESOLVED, why="stair/cap window missing")
        else:
            r = GC.check_i5_i6([x for x in dec if x.get("event") == "decode_forward"], adm, tuple(sw), tuple(cw),
                               pdmux=(bid == "B2"))
            put(f"i5i6_{bid}", r["verdict"], **{k: v for k, v in r.items() if k != "verdict"})
    put("schema_all_on", FAIL if schema_bad else PASS, errors=schema_bad[:10])
    # instrumentation level, graph-ON honoured, survival, GPU memory release
    ins_bad, graph_bad, dead, missing, td_bad = [], [], [], [], []
    for bid in ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9"):
        d = boot(bid)
        if not d.is_dir():
            missing.append(bid)
            continue
        ins = load(d / "instrumentation.json")
        if ins is None:
            ins_bad.append((bid, "record missing"))
        else:
            r = GC.check_instrumentation(ins)
            if r["verdict"] != PASS:
                ins_bad.append((bid, r["why"]))
        td = load(d / "teardown.json", {})
        sig = td.get("log_signatures") or {}
        if bid != "B7" and (sig.get("cuda_graph_capture_failed") or sig.get("eager_fallback")):
            graph_bad.append((bid, sig))
        if td.get("released") is not True:
            td_bad.append(bid)
        alive = sorted(d.glob("alive_after_*.json"))
        if not alive:
            missing.append(bid)
        for f in alive:
            rec = M.read_alive_after(d, f.name)
            if rec is None or not (rec["poll_returncode"] is None and rec["health_http_status"] == 200):
                dead.append((bid, f.name))
    put("instrumentation_level", FAIL if ins_bad else (UNRESOLVED if missing else PASS), bad=ins_bad, missing=missing)
    put("cudagraph_on_honoured", FAIL if graph_bad else PASS, bad=graph_bad)
    put("survival_all_steps", FAIL if dead else (UNRESOLVED if missing else PASS), dead=dead, missing=missing)
    put("gpu_memory_released", PASS if not td_bad else UNRESOLVED, bad=td_bad, kind=HARNESS_UNRESOLVED if td_bad else None)
    put("digest_3x", PASS if _digests_ok(root, 3) else FAIL)
    return chk


def _digests_ok(root, n):
    d = load(Path(root) / "digests.json", [])
    return len(d) >= n and all(x.get("ok") for x in d)


# ---------------------------------------------------------------- G-beta
# Nano-9B-v2 in_proj geometry (config: d_inner 10240 = 128 heads x 80, 8 groups x state 128, conv dim = 10240 + 2*1024 = 12288):
#   TP1  in_proj width 2*10240 + 2*1024 + 128 = 22,656, xBC slice [10240, 10240 + 12,288)
#   TP2  per rank d_inner 5120, 4 groups, 64 heads: width 2*5120 + 2*512 + 64 = 11,328, xBC slice [5120, 5120 + 6,144)
GEOM = {22656: dict(dim=12288, offset=10240), 11328: dict(dim=6144, offset=5120)}


def beta0_cases():
    """Unit cases (decision 20 criteria, 2026-10-20).  Every case yields (outcome, out_sha256, state_ok); the verdict is
    `judge_case(case, rec, ref_rec)`, a comparison with the ORIGINAL kernel's contiguous run at the same (dim, T):
      REF     the reference run itself (completes with exact conv state)
      MATCH   completes, exact state, out_sha256 == ref's
      BROKEN  crash OR state mismatch OR out_sha256 != ref's
    ref: pristine contig (same width/dim/T), except the chunk_offset-only mutant whose ref is the PATCHED contig.
    Order: references before the cases that use them."""
    C = []

    def add(kernel, layout, width, Ts, expect, refkind="pristine"):
        for T, reps in Ts:
            for rep in range(reps):
                ref = None if expect == "REF" else f"{refkind}_contig_w{width}_T{T}_r0"
                C.append(dict(name=f"{kernel}_{layout}_w{width}_T{T}_r{rep}", kernel=kernel, layout=layout, width=width,
                              T=T, rep=rep, expect=expect, ref=ref, **GEOM[width]))
    T1 = (12288, 94784, 94800, 95100, 126976)
    T2 = (95100, 126976, 189000, 190500)
    for w, Ts in ((22656, T1), (11328, T2)):
        add("pristine", "contig", w, [(T, 1) for T in Ts], "REF")
    add("patched", "contig", 22656, [(94790, 1)], "REF")
    add("patched", "view", 22656, [(12288, 1), (94784, 2), (94800, 2), (95100, 2), (126976, 1)], "MATCH")
    add("patched", "view", 11328, [(T, 1) for T in T2], "MATCH")
    add("patched", "contig", 22656, [(95100, 1)], "MATCH")
    add("pristine", "view", 22656, [(12288, 1), (94784, 2)], "MATCH")
    add("pristine", "view", 22656, [(94800, 2), (95100, 2)], "BROKEN")
    add("pristine", "view", 11328, [(95100, 1), (126976, 1), (189000, 1)], "MATCH")     # int32 boundary stride 11,328 ~ 189,573 [arith]
    add("pristine", "view", 11328, [(190500, 1)], "BROKEN")
    add("co_only", "view", 22656, [(94790, 1), (95100, 1)], "BROKEN", refkind="patched")
    return C


def classify_case(rc, timed_out):
    if timed_out:
        return "CRASH"
    return {0: "OK", 1: "STATE_MISMATCH", 2: "ERROR"}.get(rc, "CRASH")


def judge_case(c, rec, ref_rec):
    """-> (PASS | FAIL | UNRESOLVED, why).  Pure; used live and by score_beta (re-judged from the case files)."""
    if not rec:
        return UNRESOLVED, "case missing"
    out = rec.get("outcome")
    if out == "ERROR":
        return UNRESOLVED, "unit input error (rc 2): harness defect, not evidence"
    ref_ok = bool(ref_rec) and ref_rec.get("outcome") == "OK" and bool(ref_rec.get("out_sha256"))
    if c["expect"] == "REF":
        return (PASS, "reference completed with exact state") if out == "OK" else (UNRESOLVED, f"reference run {out}")
    if c["expect"] == "MATCH":
        if out in ("CRASH", "STATE_MISMATCH"):
            return FAIL, f"expected output == reference but the case {out}"
        if not ref_ok:
            return UNRESOLVED, "reference run missing or did not complete"
        same = rec.get("out_sha256") == ref_rec["out_sha256"]
        return (PASS, "output sha == reference sha") if same else (FAIL, "output sha != reference sha")
    if c["expect"] == "BROKEN":
        if out in ("CRASH", "STATE_MISMATCH"):
            return PASS, f"broken as predicted ({out})"
        if not ref_ok:
            return UNRESOLVED, "reference run missing or did not complete"
        same = rec.get("out_sha256") == ref_rec["out_sha256"]
        return (FAIL, "output sha == reference sha: NOT broken") if same else (PASS, "broken as predicted (output sha != reference)")
    return UNRESOLVED, f"unknown expectation {c['expect']}"


def beta2_signature_ok(res, blocking):
    """Decision 20: a NON-blocking boot only needs death AND the 'illegal memory access' signature (the async traceback
    does not name the kernel file); the blocking boot additionally needs the conv file in the traceback."""
    if not res.get("died"):
        return None
    if blocking:
        return bool(res.get("sig_illegal") and res.get("sig_conv") and res.get("traceback_conv_file"))
    return bool(res.get("sig_illegal"))


def make_kernels(root):
    """Three kernel copies (patched = installed, pristine = reverse-applied scratch copy, co_only = pristine with only
    chunk_offset cast).  -> {kernel: path}.  Verifies the pristine sha."""
    kd = Path(root) / "beta0" / "kernels"
    scratch = kd / "pristine_tree"
    f = scratch / CONV_REL
    f.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(tree_path() / CONV_REL, f)
    if CONV_MARK in f.read_text():
        r = _patch(scratch, "--reverse")
        if r.returncode != 0:
            raise GateAbort("ABORT_PATCH_STATE", "scratch reverse failed: " + r.stderr.decode()[-200:])
    pristine_sha = tget("PRISTINE_SHA256") or PRISTINE_SHA256
    if sha256_file(f) != pristine_sha:
        raise GateAbort("ABORT_PATCH_STATE", "scratch pristine conv file sha != registered pristine upstream sha")
    co = kd / "co_only" / "causal_conv1d_triton.py"
    co.parent.mkdir(parents=True, exist_ok=True)
    src = f.read_text()
    old = tget("CO_OLD") or "chunk_offset = tl.program_id(1)"
    if src.count(old) != 1:
        raise GateAbort("ABORT_PATCH_STATE", "co_only mutant: chunk_offset line not found exactly once")
    co.write_text(src.replace(old, old + ".to(tl.int64)"))
    return {"patched": tree_path() / CONV_REL, "pristine": f, "co_only": co}


def run_unit_case(root, kernels, c, base_env):
    cdir = Path(root) / "beta0" / "cases"
    cdir.mkdir(parents=True, exist_ok=True)
    out = cdir / f"{c['name']}.json"
    cmd = shlex.split(tget("UNIT_CMD") or "") or [sys.executable, str(HERE / "conv1d_int64_unit.py")]
    cmd += ["--kernel-file", str(kernels[c["kernel"]]), "--tokens", str(c["T"]), "--layout", c["layout"],
            "--width-inproj", str(c["width"]), "--dim", str(c["dim"]), "--offset", str(c["offset"]), "--out", str(out)]
    env = dict(base_env)
    env["TRITON_CACHE_DIR"] = str(Path(base_env.get("PDMUX_GATE_CACHE_BASE", root)) / f"triton_unit_{c['kernel']}")
    timed_out, rc = False, None
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=float(tget("UNIT_TMO", TMO["unit_case"])))
        rc, tail = r.returncode, (r.stderr or "")[-300:]
    except subprocess.TimeoutExpired:
        timed_out, tail = True, "timeout"
    doc = load(out, {}) if rc in (0, 1) else {}
    outcome = classify_case(rc, timed_out)
    # rc 1 is a STATE_MISMATCH only when the unit wrote its verdict; a Python exception also exits 1 (round 1: co_only/pristine
    # rc 1 with no output file) -- that is a crash, as is an rc 0 without a verdict/sha
    if outcome == "STATE_MISMATCH" and doc.get("verdict") != "STATE_MISMATCH":
        outcome = "CRASH"
    if outcome == "OK" and not (doc.get("verdict") == "RAN" and doc.get("out_sha256")):
        outcome = "CRASH"
    rec = dict(c, rc=rc, outcome=outcome, out_sha256=doc.get("out_sha256"),
               max_abs_diff=doc.get("max_abs_diff"), stderr_tail=tail)      # max_abs_diff: descriptive only
    ref = load(cdir / f"{c['ref']}.json") if c.get("ref") else None
    rec["case_ok"] = judge_case(c, rec, ref)[0] == PASS
    dump(rec, out)
    return rec


def beta_boot_requests(srv, ctx, lengths_reps, name):
    """Single long requests (non-streaming), memory sampled.  lengths_reps = [(L, n)] -> results."""
    res, bad = [], []
    for L, n in lengths_reps:
        for rep in range(n):
            ids = ctx.ids(NANO, L, H.seed_for(L, rep))
            before = H.log_signatures(read_text(srv.log))
            r = gen_ids(srv.port, ids, 64, timeout=TMO["beta_long"])
            delta = H.sig_delta(before, H.log_signatures(read_text(srv.log)))
            illegal = bool(re.search(r"illegal memory access", read_text(srv.log), re.I))
            oom = bool(delta.get("oom")) or (not r["ok"] and re.search(r"out of memory", (r["error"] or "") + read_text(srv.log)[-4000:], re.I) is not None)
            res.append({"L": L, "rep": rep, "ok": r["ok"] and bool(r["out"]), "oom": oom, "illegal": illegal,
                        "error": r["error"], "ids_sha256": H.ids_sha256(ids), "out": r["out"]})
    dump(res, srv.dir / f"{name}.json")
    return res


def run_beta(gate):
    ctx = Ctx()
    tree = tree_path()
    base_env = dict(os.environ)
    if _is_test():
        base_env["PDMUX_GATE_CACHE_BASE"] = str(gate.root / "cache")
    gate.check_digest("initial", "applied")
    # ---- beta-0: kernel unit cases, one subprocess per case
    kernels = {}

    def prep():
        kernels.update(make_kernels(gate.root))
        return PASS, "kernel copies built", {k: sha256_file(v) for k, v in kernels.items()}
    if gate.stage("beta0_prep", prep, 30, 120) == PASS:
        cases = beta0_cases()
        # the green-context event calibration unit (audit r1 MAJOR 1(2)), server-less
        def calib():
            cmd = shlex.split(tget("CALIB_CMD") or "") or [sys.executable, str(HERE / "green_event_calib_unit.py")]
            out = gate.root / "beta0" / "green_event_calib.json"
            r = subprocess.run(cmd + ["--out", str(out)], capture_output=True, text=True, timeout=600, env=base_env)
            v = (load(out, {}) or {}).get("verdict", "UNRESOLVED")
            return {"PASS": PASS, "FAIL": FAIL}.get(v, UNRESOLVED), f"green-ctx event calibration {v} (rc {r.returncode}); " \
                "FAIL -> user decision, tolerances are not adjusted to data", {}
        gate.stage("beta0_green_event_calib", calib, 120, 660)
        for c in cases:
            gate.stage("beta0_" + c["name"], lambda c=c: _unit_stage(gate, kernels, c, base_env), EST["unit_case"], TMO["unit_case"] + 60)
    # ---- beta-1: patched tree, long requests
    patched_ref = {}
    for bid, arm, plan in (("BETA1_fused", "fused_sym", [(98304, 2), (126976, 2), (95100, 1)]),
                           ("BETA1_agn", "agnostic", [(98304, 2), (126976, 2)])):
        spec = dict(arm=arm, on=True, ctx=CTX_LONG)

        def step(srv, plan=plan, bid=bid):
            ms = MemSampler()
            ms.start()
            res = beta_boot_requests(srv, ctx, plan, "long_patched")
            mem = ms.finish()
            tel = load(srv.dir / "teardown.json", {})
            dump({"memory": mem, "banner": H.parse_banner(read_text(srv.log))}, srv.dir / "memory.json")
            if bid == "BETA1_fused":
                for r in res:
                    if r["L"] == 95100 and r["ok"]:
                        patched_ref["out95100"] = r["out"]
                        dump({"out": r["out"]}, gate.root / "beta_patched_95100.json")
            labels = [f"T2EXT_OOM_L{r['L']}" for r in res if r["oom"] and r["L"] in (98304, 126976)]
            hard = [r for r in res if not r["ok"] and not r["oom"]]
            if any(r["illegal"] for r in res):
                return FAIL, "illegal memory access in the PATCHED tree", {"labels": labels}
            if hard:
                return FAIL, f"{len(hard)} request(s) failed without OOM", {"labels": labels}
            if any(r["oom"] and r["L"] == 98304 for r in res):
                return UNRESOLVED, "OOM at 98,304 (labelled; passing 98,304 is a PASS condition)", {"labels": labels}
            return PASS, "all requests completed with non-empty output, no illegal-access signature; labels " + str(labels), \
                {"labels": labels, "memory": mem}
        _s, _t, okc = boot_and_run(gate, bid, spec, "patched", [("long_patched", step, 700, TMO["beta_long"] * 2)], ctx)
        if not okc:
            break
    # ---- beta-2: reverse-applied tree, crash loop
    reversed_done = False
    try:
        if gate.fits(1500):
            reverse_patch(tree)
            reversed_done = True
            gate.check_digest("reversed", "reversed")
            beta2(gate, ctx)
        else:
            gate.record("beta2", SKIPPED, "budget")
    finally:
        if reversed_done:
            reapply_patch(tree)
    gate.check_digest("reapplied", "applied")
    if gate.fits(EST["boot"] + 300):
        spec = dict(arm="fused_sym", on=True, ctx=CTX_LONG)
        boot_and_run(gate, "BETA2_boot3_reapplied", spec, "patched",
                     [("long_95100", lambda srv: _beta_ok_step(srv, ctx), 300, TMO["beta_long"])], ctx)
    else:
        gate.record("BETA2_boot3_reapplied", SKIPPED, "budget")
    return score_beta(gate.root)


def _unit_stage(gate, kernels, c, base_env):
    r = run_unit_case(gate.root, kernels, c, base_env)
    ref = load(Path(gate.root) / "beta0" / "cases" / f"{c['ref']}.json") if c.get("ref") else None
    v, why = judge_case(c, r, ref)
    return v, f"{c['kernel']} {c['layout']} w{c['width']} T={c['T']}: {r['outcome']} (expected {c['expect']}): {why}", \
        {"outcome": r["outcome"], "out_sha256": r["out_sha256"]}


def _beta_ok_step(srv, ctx):
    res = beta_boot_requests(srv, ctx, [(95100, 1)], "long_reapplied")
    ref = load(srv.g.root / "beta_patched_95100.json", {}).get("out")
    ok = res[0]["ok"] and not res[0]["illegal"]
    same = ref is None or res[0]["out"] == ref
    return (PASS if ok else FAIL), f"95,100 after re-apply: ok={ok}, output equals patched-tree output={same}", {"same_as_patched": same}


def beta2(gate, ctx):
    """Falsification arm.  'Pristine is broken' = server death OR output differing from the patched tree's output
    (audit r1 4c).  A reboot/memory-release failure is HARNESS_UNRESOLVED, never FAIL."""
    ref = load(gate.root / "beta_patched_95100.json", {}).get("out")
    for bid, blocking in (("BETA2_boot1", False), ("BETA2_boot2_blocking", True)):
        spec = dict(arm="fused_sym", on=True, ctx=CTX_LONG, blocking=blocking, death_ok=True)

        def step(srv, blocking=blocking, bid=bid):
            res = {"blocking": blocking}
            if not blocking:
                ids = ctx.ids(NANO, 94500, H.seed_for(94500, 0))
                r = gen_ids(srv.port, ids, 64, timeout=TMO["beta_long"])
                res["pre_boundary_94500"] = {"ok": r["ok"], "error": r["error"]}
            ids = ctx.ids(NANO, 95100, H.seed_for(95100, 0))
            box = {}
            th = threading.Thread(target=lambda: box.update(r=gen_ids(srv.port, ids, 64, timeout=TMO["death"])), daemon=True)
            th.start()
            t0 = time.monotonic()
            limit = float(tget("DEATH_WAIT_S", TMO["death"]))
            while th.is_alive() and not srv.dead() and time.monotonic() - t0 < limit:
                time.sleep(0.5 if _is_test() else 2)
            died = srv.dead()
            if not died and th.is_alive():
                return UNRESOLVED, f"neither death nor a response within {limit:.0f} s", res
            th.join(10)
            r = box.get("r") or {}
            time.sleep(1 if _is_test() else 3)          # let the log writer flush the traceback
            log = read_text(srv.log)
            res.update(died=died, returncode=srv.proc.poll(), request_ok=r.get("ok"),
                       sig_illegal=bool(re.search(r"illegal memory access", log, re.I)),
                       sig_conv=("causal_conv1d" in log),
                       traceback_conv_file=bool(re.search(r"Traceback.*causal_conv1d_triton\.py", log, re.S)))
            if died:
                ok = beta2_signature_ok(res, blocking)
                res["broken"] = True
                res["signature_ok"] = ok
                dump(res, srv.dir / "beta2.json")
                return (PASS if ok else FAIL), "pristine tree: server died" + ("" if ok else "; signature missing"), res
            same = (ref is not None and r.get("out") == ref)
            res.update(broken=not same, output_equals_patched=same)
            dump(res, srv.dir / "beta2.json")
            if ref is None:
                return UNRESOLVED, "server survived but the patched reference output is missing", res
            return (PASS if not same else FAIL), ("server survived; output differs from the patched tree's (pristine is broken)"
                                                  if not same else "pristine tree reproduced the patched output: falsification FAILED"), res
        _s, _t, okc = boot_and_run(gate, bid, spec, "reversed", [("crash95100", step, 600, TMO["death"] + 120)], ctx)
        if not okc:
            gate.record(bid + "_continue", UNRESOLVED, "GPU memory not released: remaining reboot skipped (HARNESS_UNRESOLVED)",
                        {"kind": HARNESS_UNRESOLVED})
            return


def score_beta(root):
    root = Path(root)
    chk = OrderedDict()
    cases = [load(f) for f in sorted((root / "beta0" / "cases").glob("*.json"))] if (root / "beta0" / "cases").is_dir() else []
    want = beta0_cases()
    have = {c["name"]: c for c in cases if c}
    miss = [c["name"] for c in want if c["name"] not in have]
    judged = {c["name"]: judge_case(c, have.get(c["name"]), have.get(c["ref"]) if c.get("ref") else None) for c in want}
    bad = [n for n, (v, _w) in judged.items() if v == FAIL]
    unres = [n for n, (v, _w) in judged.items() if v == UNRESOLVED]
    chk["beta0_cases"] = {"verdict": FAIL if bad else (UNRESOLVED if (miss or unres) else PASS), "bad": bad,
                          "unresolved": unres[:10], "missing": miss[:10],
                          "criterion": "decision 20: sha(output) vs the original kernel's contiguous run; max_abs_diff descriptive"}
    pc = have.get("pristine_contig_w22656_T95100_r0", {}).get("out_sha256")
    pv = have.get("patched_contig_w22656_T95100_r0", {}).get("out_sha256")
    chk["beta0_contig_digest_equal"] = {"verdict": PASS if pc and pc == pv else (UNRESOLVED if not (pc and pv) else FAIL),
                                        "pristine_contig": pc, "patched_contig": pv}
    cal = load(root / "beta0" / "green_event_calib.json", {})
    chk["beta0_green_event_calib"] = {"verdict": {"PASS": PASS, "FAIL": FAIL}.get(cal.get("verdict"), UNRESOLVED)}
    for bid in ("BETA1_fused", "BETA1_agn"):
        res = load(root / bid / "long_patched.json")
        if res is None:
            chk[f"beta1_{bid}"] = {"verdict": UNRESOLVED, "why": "result missing"}
            continue
        ok98 = all(r["ok"] for r in res if r["L"] == 98304)
        bad_ = [r for r in res if (not r["ok"] and not r["oom"]) or r["illegal"]]
        oom98 = any(r["oom"] for r in res if r["L"] == 98304)
        v = FAIL if bad_ else (PASS if ok98 else UNRESOLVED)
        chk[f"beta1_{bid}"] = {"verdict": v, "labels": [f"T2EXT_OOM_L{r['L']}" for r in res if r["oom"]], "oom98304": oom98}
    for bid in ("BETA2_boot1", "BETA2_boot2_blocking"):
        r = load(root / bid / "beta2.json")
        if r is None:
            chk[f"beta2_{bid}"] = {"verdict": UNRESOLVED, "why": "no result (not run / harness)"}
        else:
            sig = beta2_signature_ok(r, bool(r.get("blocking")))
            ok = bool(r.get("broken")) and (sig if r.get("died") else True)
            chk[f"beta2_{bid}"] = {"verdict": PASS if ok else FAIL, "died": r.get("died"), "broken": r.get("broken")}
    r3 = load(root / "BETA2_boot3_reapplied" / "long_reapplied.json")
    chk["beta2_reapplied_ok"] = {"verdict": (PASS if r3 and r3[0]["ok"] and not r3[0]["illegal"] else UNRESOLVED if r3 is None else FAIL)}
    chk["digest_3x"] = {"verdict": PASS if _digests_ok(root, 3) else FAIL}   # initial, reversed, reapplied + final re-check
    ins_bad = []
    for d in sorted(root.glob("BETA*")):
        ins = load(d / "instrumentation.json")
        if ins and GC.check_instrumentation(ins)["verdict"] != PASS:
            ins_bad.append(d.name)
    chk["instrumentation_level"] = {"verdict": FAIL if ins_bad else PASS, "bad": ins_bad}
    return chk


# ---------------------------------------------------------------- G-gamma
def build_trace(path, cell, rate, n, seed, shape=SHAPE_A):
    rnd = random.Random(seed)
    t, lines = 0.0, []
    for i in range(n):
        t += rnd.expovariate(rate)
        lines.append(json.dumps({"record": "request", "request_id": f"{cell}-{i:05d}", "phase": "main",
                                 "arrival_s": round(t, 6), "input_tokens": shape[0], "output_tokens": shape[1]}))
    Path(path).write_text("\n".join(lines) + "\n")


def run_cell(srv, cdir, rung, seed):
    import mg_label as L
    rate = float(tget("HI_RATE", HI_RATE)) if rung == "HI" else float(tget("LO_RATE", LO_RATE))
    n = int(tget("HI_N", HI_N)) if rung == "HI" else int(tget("LO_N", LO_N))
    Path(cdir).mkdir(parents=True, exist_ok=True)
    build_trace(Path(cdir) / "trace.jsonl", rung, rate, n, seed)
    cmd = shlex.split(tget("LOADGEN_CMD") or "") or [sys.executable, str(ENG / "scripts" / "multigpu" / "mg_loadgen.py")]
    cmd += ["--endpoint", f"http://127.0.0.1:{srv.port}/generate", "--trace", str(Path(cdir) / "trace.jsonl"),
            "--cell-dir", str(cdir)]
    env = dict(os.environ)
    env["PDMUX_GATE_CELL_ON"] = "1" if srv.spec["on"] else "0"
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=float(tget("CELL_TMO", TMO["cell"])), env=env)
        rc = r.returncode
    except subprocess.TimeoutExpired:
        rc = -999
    reqs = L.load_requests(Path(cdir) / "requests.jsonl") if (Path(cdir) / "requests.jsonl").exists() else []
    meta = {"cell_id": f"{srv.id}_{rung}", "rung": rung, "seed": seed}
    rec = L.analyze_cell(reqs, meta) if reqs else {"status": "EMPTY", "g": None, "achieved": None, "n_sent": 0,
                                                     "n_server_errors": 0, "n_client_errors": 0}
    itl = [v for q in reqs if not L.is_error(q) for v in q.get("token_itl_ms", [])]
    out = {"rung": rung, "rc": rc, "n": rec.get("n_sent"), "status": rec.get("status"), "g": rec.get("g"),
           "achieved": rec.get("achieved"), "n_server_errors": rec.get("n_server_errors"),
           "n_client_errors": rec.get("n_client_errors"), "itl_p95_ms": L.percentile(itl, 95.0) if itl else None,
           "n_itl": len(itl), "rate": rate, "seed": seed, "stall": GC.stall_boundary(itl)}
    dump(out, Path(cdir) / "cell_metrics.json")
    return out


def instance_valid(inst):
    """Audit-registered pair-validity rules (design sec 5): KNEE_BRACKETED, client/server errors <= 1 %, no abnormal exit."""
    import mg_const as C
    why = []
    hi, lo = inst.get("hi") or {}, inst.get("lo") or {}
    if hi.get("status") != "OK" or lo.get("status") != "OK":
        why.append(f"cell status HI={hi.get('status')} LO={lo.get('status')}")
    if hi.get("g") is None or hi["g"] < C.G_HI or lo.get("g") is None or lo["g"] > C.G_LO:
        why.append(f"not KNEE_BRACKETED (g_hi={hi.get('g')}, g_lo={lo.get('g')})")
    for nm, c in (("HI", hi), ("LO", lo)):
        n = c.get("n") or 0
        if n and (c.get("n_server_errors") or 0) / n > C.CLIENT_ERROR_FRAC:
            why.append(f"{nm} server-side errors > 1 %")
        if n and (c.get("n_client_errors") or 0) / n > C.CLIENT_ERROR_FRAC:
            why.append(f"{nm} client-side errors > 1 %")
    if inst.get("abnormal_exit"):
        why.append("abnormal server exit")
    return (not why), why


def run_instance(gate, pair_i, level, seed, arm, ctx):
    bid = f"pair{pair_i:02d}_{level}"
    spec = dict(arm=arm, on=(level == "on"), ctx=None)     # 'off' / 'offa' / 'offb' are all OFF boots
    inst = {"pair": pair_i, "level": level, "seed": seed, "arm": arm}

    def warm(srv):
        bad = 0
        for i in range(8):
            r = gen_ids(srv.port, [1] * SHAPE_A[0], 64, timeout=TMO["warm"])
            bad += 0 if r["ok"] else 1
        return (PASS if not bad else FAIL), f"{8 - bad}/8 warm-up requests ok", {}

    def mk(rung):
        def fn(srv):
            inst[rung.lower()] = run_cell(srv, srv.dir / f"cell_{rung}", rung, seed)
            return PASS, f"{rung} cell done rc={inst[rung.lower()]['rc']}", {}
        return fn
    steps = [("warm", warm, 60, TMO["warm"]), ("hi", mk("HI"), 250, TMO["cell"] + 30), ("lo", mk("LO"), 250, TMO["cell"] + 30)]
    srv, td, ok = boot_and_run(gate, bid, spec, "patched", steps, ctx)
    ab = False
    for st in ("warm", "hi", "lo"):
        a = M.alive_after_step(srv.dir, st)
        ab = ab or a is False
    inst["abnormal_exit"] = ab
    inst["released"] = ok
    inst["valid"], inst["invalid_reasons"] = instance_valid(inst)
    dump(inst, gate.root / bid / "instance.json")
    return inst, ok


def gamma_pairs_plan(phase):
    seeds = GAMMA_SEEDS if phase == "main" else GAMMA_EXT_SEEDS
    off = 0 if phase == "main" else len(GAMMA_SEEDS)
    npairs = int(tget("PAIRS", len(seeds)))
    return [(off + i + 1, seeds[i], ("on", "off") if (off + i + 1) % 2 == 1 else ("off", "on")) for i in range(npairs)]


def gamma_oo_plan(phase):
    """OFF/OFF diagnostic pairs: DEFAULT OFF since decision 20 (a) (they left the main plan; the option is for an extension Job:
    env PDMUX_GATE_GAMMA_OO_PAIRS=1|2, a real env, not a test hook).  Run AFTER all ON/OFF pairs; a budget shortfall records
    DIAG_SKIPPED (outside the SKIPPED/UNRESOLVED roll-up).  Levels 'offa'/'offb' are both OFF boots; ids 101.. (main) / 201.. (ext)."""
    n = min(int(os.environ.get("PDMUX_GATE_GAMMA_OO_PAIRS", "0") or 0), len(GAMMA_OO_SEEDS))
    base = 100 if phase == "main" else 200
    return [(base + 1 + i, GAMMA_OO_SEEDS[i], ("offa", "offb")) for i in range(n)]


def run_gamma(gate, family):
    arm = {"A": "agnostic", "F": "fused_sym"}[family]
    phase = os.environ.get("PDMUX_GATE_GAMMA_PHASE", "main")
    if phase not in ("main", "ext"):
        raise GateAbort("ABORT_CONFIG", f"PDMUX_GATE_GAMMA_PHASE={phase}")
    ctx = Ctx()
    gate.check_digest("start", "applied")
    plan = gamma_pairs_plan(phase)
    mid = len(plan) // 2
    plan = plan + gamma_oo_plan(phase)
    for k, (pi, seed, order) in enumerate(plan):
        if k == mid and k > 0:
            gate.check_digest("middle", "applied")
        if not gate.fits(2 * EST["inst"]):
            # an unrun diagnostic pair must not lower the Job verdict (decision 20 (a)); an ON/OFF pair still does
            gate.record(f"pair{pi:02d}", "DIAG_SKIPPED" if pi >= 100 else SKIPPED, "budget: pair not started (no half pairs)")
            continue
        for level in order:
            _inst, ok = run_instance(gate, pi, level, seed, arm, ctx)
            if not ok:
                gate.record(f"pair{pi:02d}_halt", UNRESOLVED, "GPU memory not released: remaining pairs skipped (HARNESS_UNRESOLVED)")
                break
        else:
            continue
        break
    if not any(d["tag"] == "middle" for d in gate.digests) and len(plan) > 1:
        gate.check_digest("middle", "applied")
    gate.check_digest("end", "applied")
    return score_gamma(gate.root, phase)


def score_gamma(root, phase="main", extra_dirs=()):
    root = Path(root)
    chk = OrderedDict()
    insts = {}
    for d in [root, *map(Path, extra_dirs)]:
        for f in sorted(d.glob("pair*_*/instance.json")):
            i = load(f)
            if i:
                insts[(i["pair"], i["level"])] = i
    oo_insts = {k: v for k, v in insts.items() if k[0] >= 100}      # r2 diagnostic OFF/OFF pairs: never enter the TOST
    insts = {k: v for k, v in insts.items() if k[0] < 100}
    pairs = sorted({p for p, _ in insts})
    valid_pairs, inv_on, inv_off, rows = [], 0, 0, []
    for p in pairs:
        on, off = insts.get((p, "on")), insts.get((p, "off"))
        if on is None or off is None:
            rows.append({"pair": p, "valid": False, "why": "an instance is missing"})
            continue
        von, voff = on["valid"], off["valid"]
        if not von:
            inv_on += 1
        if not voff:
            inv_off += 1
        rows.append({"pair": p, "valid": von and voff, "on_valid": von, "off_valid": voff,
                     "why": on["invalid_reasons"] + off["invalid_reasons"]})
        if von and voff:
            valid_pairs.append((on, off))
    dump(rows, root / "gamma_pairs.json")
    lam = [(a["hi"]["achieved"], b["hi"]["achieved"]) for a, b in valid_pairs]
    itl = [(a["lo"]["itl_p95_ms"], b["lo"]["itl_p95_ms"]) for a, b in valid_pairs]
    look = "extended" if len(pairs) > GC.REG["tost_initial_pairs"] else "initial"
    res = {"lambda_star_thr": GC.tost_log_ratio(lam, look), "itl_p95": GC.tost_log_ratio(itl, look)}
    fam = GC.gamma_family_verdict(res, inv_on, inv_off)
    chk["tost_lambda_star_thr"] = {"verdict": res["lambda_star_thr"]["verdict"], **res["lambda_star_thr"]}
    chk["tost_itl_p95"] = {"verdict": res["itl_p95"]["verdict"], **res["itl_p95"]}
    chk["invalid_pair_asymmetry"] = {"verdict": "OBSERVER_EFFECT" if inv_on - inv_off >= GC.REG["invalid_asym_min_diff"] else PASS,
                                     "invalid_on": inv_on, "invalid_off": inv_off}
    chk["family"] = {"verdict": fam["verdict"], **fam, "n_pairs": len(pairs), "n_valid": len(valid_pairs), "look": look,
                     "phase": phase}
    chk["diag_off_off"] = diag_off_off(oo_insts, valid_pairs)
    chk["diag_stall_boundary"] = diag_stall(insts)
    chk["digest_3x"] = {"verdict": PASS if _digests_ok(root, 3) else FAIL}
    ins_bad = []
    for d in sorted(root.glob("pair*_*")):
        ins = load(d / "instrumentation.json")
        if ins and GC.check_instrumentation(ins)["verdict"] != PASS:
            ins_bad.append(d.name)
    chk["instrumentation_level"] = {"verdict": FAIL if ins_bad else PASS, "bad": ins_bad}
    return chk


def diag_off_off(oo_insts, valid_pairs):
    """r2 descriptive diagnostic (pre-registered use: interpretation aid when the family is UNRESOLVED_POWER / OBSERVER_EFFECT;
    NEVER changes a verdict): log(offb/offa) of lambda*_thr and LO ITL p95 for each valid OFF/OFF pair, next to the SD of the
    ON/OFF log ratios of the same Job."""
    rows = []
    for p in sorted({k[0] for k in oo_insts}):
        a, b = oo_insts.get((p, "offa")), oo_insts.get((p, "offb"))
        if not a or not b or not (a.get("valid") and b.get("valid")):
            rows.append({"pair": p, "valid": False})
            continue
        rows.append({"pair": p, "valid": True,
                     "log_lambda": math.log(b["hi"]["achieved"] / a["hi"]["achieved"]),
                     "log_itl_p95": math.log(b["lo"]["itl_p95_ms"] / a["lo"]["itl_p95_ms"])})
    def sd(xs):
        return statistics.stdev(xs) if len(xs) > 1 else None
    onoff = {"log_lambda": [math.log(a["hi"]["achieved"] / b["hi"]["achieved"]) for a, b in valid_pairs],
             "log_itl_p95": [math.log(a["lo"]["itl_p95_ms"] / b["lo"]["itl_p95_ms"]) for a, b in valid_pairs]}
    return {"verdict": "DIAGNOSTIC", "role": "interpretation aid only; does not change any verdict", "pairs": rows,
            "n_valid_oo": sum(1 for r in rows if r["valid"]),
            "onoff_log_ratio_sd": {k: sd(v) for k, v in onoff.items()}}


def diag_stall(insts):
    """r2 descriptive diagnostic: per (pair, level) LO-cell stall fraction and the STALL_BOUNDARY label (5 % +- 1 pp)."""
    cells = []
    for (p, lvl), i in sorted(insts.items()):
        st = (i.get("lo") or {}).get("stall") or {}
        cells.append({"pair": p, "level": lvl, "seed": i.get("seed"), "stall_frac": st.get("stall_frac"), "label": st.get("label")})
    return {"verdict": "DIAGNOSTIC", "role": "label only; does not change any verdict",
            "n_boundary": sum(1 for c in cells if c["label"]), "cells": cells}


def verdict_of(name, chk):
    """Final verdict word for the Job.  G-gamma: the family verdict is the headline; OBSERVER_EFFECT is reported as such."""
    if name == "decomp":
        bad = [k for k in ("digest_3x", "level_assertions", "order", "export_size") if chk[k]["verdict"] != PASS]
        return FAIL if bad else chk["decomp"]["verdict"]
    if name == "gamma":
        fam = chk["family"]["verdict"]
        if fam == "OBSERVER_EFFECT":
            return "OBSERVER_EFFECT"
        if chk["digest_3x"]["verdict"] != PASS or chk["instrumentation_level"]["verdict"] != PASS:
            return FAIL
        return fam if fam in (PASS, "UNRESOLVED_POWER") else UNRESOLVED
    return rollup([c["verdict"] for c in chk.values()])


# ---------------------------------------------------------------- finalisation, export guard
def export_guard(gate):
    """Keep the exported tree small (M3 J1 was truncated mid-export): gzip big logs/telemetry, keep an excerpt of server
    logs, record the size.  Full copies of oversize files stay on the cluster volume (not exported)."""
    limit_mb = float(os.environ.get("PDMUX_GATE_MAX_EXPORT_MB", "300"))
    full = Path(os.environ.get("PDMUX_DATA", "/data")) / "runs" / gate.meta["job"] / "gate_full"
    for f in sorted(gate.root.rglob("*")):
        if not f.is_file() or f.suffix == ".gz" or "cache" in f.parts:
            continue
        if f.name == "server.log" and f.stat().st_size > 2_000_000:
            lines = f.read_text(errors="replace").splitlines()
            keep = lines[:1500] + ["... [truncated; full gz kept] ..."] + lines[-3000:]
            with gzip.open(str(f) + ".gz", "wt") as g:
                g.write("\n".join(lines))
            f.write_text("\n".join(keep) + "\n")
        elif f.suffix == ".jsonl" and f.stat().st_size > 5_000_000:
            with open(f, "rb") as a, gzip.open(str(f) + ".gz", "wb") as b:
                shutil.copyfileobj(a, b)
            f.unlink()
    total = sum(f.stat().st_size for f in gate.root.rglob("*") if f.is_file() and "cache" not in f.parts) / 1e6
    moved = []
    if total > limit_mb:
        full.mkdir(parents=True, exist_ok=True)
        for f in sorted(gate.root.rglob("*.gz"), key=lambda p: -p.stat().st_size):
            if total <= limit_mb:
                break
            total -= f.stat().st_size / 1e6
            shutil.move(str(f), str(full / (f.parent.name + "__" + f.name)))
            moved.append(f.name)
    return {"export_mb": round(total, 1), "limit_mb": limit_mb, "moved_to_cluster_volume_only": moved}


def run_gate(name, family=None):
    job = os.environ.get("PDMUX_JOB_NAME", f"local-{int(time.time())}")
    root = Path(tget("ROOT") or os.environ.get("PDMUX_GATE_ROOT") or
                (ENG / "results" / "engine_gate" / job))
    hardcap = float(os.environ.get("PDMUX_GATE_HARDCAP_MIN") or JOBS[name][1])
    gate = Gate(name, root, hardcap, family)
    code = 0

    def on_term(signum, _f):
        gate.flush("INTERRUPTED")
        os._exit(130)
    try:
        signal.signal(signal.SIGTERM, on_term)
    except ValueError:
        pass
    final = UNRESOLVED
    try:
        if GC.reg_sha256() != REG_SHA:
            raise GateAbort("ABORT_REG", f"engine_gate_checks.REG sha {GC.reg_sha256()} != registered {REG_SHA}")
        check_models(gate)
        write_substrate(gate)
        gate.meta["tree_state_at_start"] = conv_state(tree_path())
        chk = {"alpha": lambda: run_alpha(gate), "beta": lambda: run_beta(gate),
               "gamma": lambda: run_gamma(gate, family),
               "decomp": lambda: _decomp().run_decomp(gate)}[name]()
        gate.checks.update(chk)
        final = verdict_of(name, chk)
        if any(s["status"] in (FAIL,) for s in gate.stages.values() if not s["facts"].get("kind") == HARNESS_UNRESOLVED) \
                and name != "gamma" and final == PASS:
            final = FAIL
        if any(s["status"] in (SKIPPED, UNRESOLVED) for s in gate.stages.values()) and final == PASS:
            final = UNRESOLVED
        code = RC.get(final, 1 if final in (FAIL, "OBSERVER_EFFECT") else 4)
    except GateAbort as ga:
        final = ga.code
        gate.checks["abort"] = {"verdict": FAIL, "why": ga.why}
        code = RC.get(ga.code, 16)
        print(f"[GATE ABORT] {ga}", flush=True)
    gate.flush(final)               # verdict written BEFORE any bulky post-processing
    exp = export_guard(gate)
    res = gate.flush(final)
    res["export"] = exp
    dump(res, gate.root / "GATE_RESULT.json")
    (gate.root / "SUMMARY.md").write_text(summary_md(res))
    print(f"[GATE] {name} verdict {final} -> {gate.root / 'GATE_RESULT.json'}", flush=True)
    return code


def summary_md(res):
    L = [f"# engine gate {res['gate']} {res.get('family') or ''} -- {res['verdict']}", "",
         f"job {res['meta'].get('job')}  commit {res['meta'].get('commit')}  elapsed {res['elapsed_s']} s  "
         f"hardcap {res['meta'].get('hardcap_min')} min", "",
         "Harness verdict on this Job's data only; no performance statement. Judgement/analysis -> result-analyst; "
         "refutation -> claims-auditor.", "", "| check | verdict |", "|---|---|"]
    L += [f"| {k} | {v['verdict']} |" for k, v in res["checks"].items()]
    L += ["", "| stage | status | reason |", "|---|---|---|"]
    L += [f"| {k} | {v['status']} | {str(v['reason'])[:100]} |" for k, v in res["stages"].items() if v["status"] != PASS]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- plan info (launch.sh), pooling
def plan_info(name, family, rate, hardcap_min):
    exp_h, cap_min = JOBS[name]
    cap_min = hardcap_min or cap_min
    lines = [f"gate job : {name}" + (f" family {family}" if family else "") + f" -- GPU 1x A100 SXM80",
             f"expected : ~{exp_h} h (design sec 2)",
             f"hard cap : {cap_min} min self-enforced (PDMUX_GATE_HARDCAP_MIN; the driver stops starting stages "
             f"{RESERVE_S // 60} min earlier, the entry script wraps it in timeout)",
             f"cost cap : <= ${rate * cap_min / 60:.2f} at ${rate}/h (not a platform limit; vesslctl has no cap flag)",
             f"expects  : ENGINE_DIGEST {DIGEST_BASE[:12]}.. (reversed arm {DIGEST_REV[:12]}..), REG {REG_SHA[:12]}.."]
    if name == "decomp":
        lines += _decomp().plan_lines()
    if name == "gamma" and family not in ("A", "F"):
        lines.append("WARNING  : PDMUX_GATE_FAMILY not A|F -> the Job aborts (add --env PDMUX_GATE_FAMILY=A or F)")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("alpha", "beta", "gamma", "decomp"):
        p = sub.add_parser(c)
        p.add_argument("--family", choices=("A", "F"), default=os.environ.get("PDMUX_GATE_FAMILY"))
    p = sub.add_parser("plan-info")
    p.add_argument("gate", choices=("alpha", "beta", "gamma", "decomp"))
    p.add_argument("--family", default="")
    p.add_argument("--rate", type=float, default=1.48)
    p.add_argument("--hardcap-min", type=float, default=0)
    p = sub.add_parser("score")
    p.add_argument("gate", choices=("alpha", "beta", "gamma", "decomp"))
    p.add_argument("dir")
    p = sub.add_parser("pool-gamma")
    p.add_argument("main")
    p.add_argument("ext")
    a = ap.parse_args(argv)
    if a.cmd == "plan-info":
        print(plan_info(a.gate, a.family or None, a.rate, a.hardcap_min))
        return 0
    if a.cmd == "score":
        chk = {"alpha": lambda: score_alpha(a.dir, tree_path()), "beta": lambda: score_beta(a.dir),
               "gamma": lambda: score_gamma(a.dir),
               "decomp": lambda: _decomp().score_decomp(a.dir)}[a.gate]()
        print(json.dumps({"verdict": verdict_of(a.gate, chk), "checks": chk}, indent=1, default=str))
        return 0
    if a.cmd == "pool-gamma":
        chk = score_gamma(a.main, "pooled", extra_dirs=[a.ext])
        print(json.dumps({"verdict": verdict_of("gamma", chk), "checks": chk}, indent=1, default=str))
        return 0
    if a.cmd == "gamma" and a.family not in ("A", "F"):
        print("ABORT_CONFIG: gamma needs --family A|F or PDMUX_GATE_FAMILY", file=sys.stderr)
        return 16
    return run_gate(a.cmd, a.family if a.cmd == "gamma" else None)


if __name__ == "__main__":
    sys.exit(main())
