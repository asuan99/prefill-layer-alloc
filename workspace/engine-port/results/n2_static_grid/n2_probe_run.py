"""N2(3-a) U-2 (a) calibration PROBE runner (PREREG_N2_3A rev12 §14-2 "프로브 하네스 사양", H-1, H-4…H-7, R11-7).

NOT a measurement and never cited (N2C-35, N2C-40). Runs in a VESSL **Workspace** (not a Job; see
handoff-report/vessl_operating_model_2026-10-02.md -- "Workspace는 bring-up·프로브 전용") on an A100 80GB.
Local use is CPU-only: --emit-plan, --stop-rule, --reduce, and the fake-server tests in test_n2_harness.py.

One attempt = <probe-root>/probe_attempt_<n>/ (n = 1, 2, ...; created exclusively, never overwritten -- H-6):
  for each boot (d = 44 then d = 16; sticky ON, A-agn config, PHASE_EVENTS ON, the N3 rev6 server text):
    marks_<boot>.txt  boot_plan <d> <n> FIRST (before the server starts), then per cell (n2_loadgen.run_probe_cell)
    server launch -> /health -> sticky wiring check (log 'sticky partition ENABLED ... index=<idx(d)>')
    -> warm-up (8 requests, concurrency 1, seed 7; input c_in(d) -- never a multiple of 8192, which would be an orphan
       co-type span for the reducer) -> n cells (each: one load call, one kappa measurement)
    -> F-6 trailing non-measured request (input 512, max_new_tokens 8) -> H-5: idle 2 x flush_interval_s
    -> SIGTERM to the server process group (SIGKILL fallback); boot_abort <reason> <t> in finally on any abort
  then: probe.json = n2_probe_reduce.to_json(reduce_probe(boots)) with the reducer LOADED BY FILE PATH and the
  attempt directory as the working directory, raw paths RELATIVE (§14-2 (c) gate-location rule);
  optional gate PREVIEW (calculator CLI in the same directory) -> GATE_PREVIEW.json (the official post_probe_gate runs
  later on the clean freeze checkout); substrate.json (sampler on/off, R11-7); ATTEMPT_SUMMARY.json.
Re-probe stop rule (H-7, §14-2 (d)): a new attempt is refused unless the previous attempt's gate result allows a
re-probe, and two consecutive re-probe-allowed attempts with the SAME reason_codes SET stop the probe.
Abort policy (rev13 G-10, registered): a cell abort ends its boot (boot_abort with the cell's reason); after a
VALUE_INDEPENDENT boot death the next boot still runs; after a VALUE_DEPENDENT abort no further boot runs.
One attempt per invocation (G-15); a re-probe is a manual re-run allowed ONLY by the official gate's reprobe_allowed
(K_POST_PROBE.json of the clean-freeze-checkout gate run) -- GATE_PREVIEW.json never decides it.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import math
import os
import signal
import socket
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import n2_loadgen as lg      # noqa: E402
import n2_marks              # noqa: E402

ENG = os.path.abspath(os.path.join(HERE, "..", ".."))
REDUCER = os.path.join(HERE, "n2_probe_reduce.py")
CALCULATOR = os.path.join(HERE, "n2_feasibility.py")
PLAN_DEFAULT = os.path.join(HERE, "PROBE_PLAN.json")
HARNESS_FILES = ("n2_marks.py", "n2_loadgen.py", "n2_probe_run.py", "n2_probe_workspace.sh")

MODEL = "nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base"
MODEL_REVISION = "dc0661c829b14e5b9246c05cfa89094a0875e052"
CONFIG_REL = "results/n3_capacity/configs/pdmux_n3_A_agn.yml"
CONFIG_DIGEST = "01675b2d7fa8ff23bb1cb142ff66fa4dfd0c021722ae9b8e7acedc2f4d38fca1"
STICKY_IDX = {16: 1, 24: 2, 34: 3, 44: 4}          # §3-1 sticky target index
BOOT_ORDER = (44, 16)                               # rev13 G-14 registered: boot 1 = d44 (14 cells), boot 2 = d16 (7)
FLUSH_INTERVAL_S = 1.0                              # engine AsyncJsonlTelemetry default (H-5: wait 2 x this)
WARMUP_N, WARMUP_SEED, WARMUP_OUT = 8, 7, 64        # §3-1 warm-up: 8 requests, concurrency 1, seed 7
F6_INPUT, F6_OUT = 512, 8                           # §6-4 F-6 trailing non-measured request


def server_flags(cfg_path):
    """§3-1: the N3 rev6 server text (cudagraph ON: no --disable-cuda-graph / --disable-piecewise-cuda-graph)."""
    return ["--model-path", MODEL, "--trust-remote-code", "--dtype", "bfloat16",
            "--attention-backend", "flashinfer",
            "--enable-pdmux", "--pdmux-config-path", cfg_path,
            "--disable-overlap-schedule", "--chunked-prefill-size", "-1",
            "--disable-radix-cache", "--mem-fraction-static", "0.82",
            "--max-running-requests", "48",
            "--context-length", "16384", "--random-seed", "1"]


# ---- PDMUX_* knobs: everything the runner does not set itself is removed from the server environment -------------
UNSET = ("PDMUX_SLO_SCHED", "PDMUX_SLO_MODE", "PDMUX_LA_COORD", "PDMUX_FIXED_DECODE_SM_FILE",
         "PDMUX_TRACE_FORCE_PREFILL", "PDMUX_DUAL_WORKER_TRACE_EVERY", "PDMUX_TRUE_DUAL_WORKER", "PDMUX_DUAL_WORKER",
         "PDMUX_DUAL_WORKER_TRACE", "PDMUX_MEM_TELEMETRY")
ALLOW_INHERITED = ("PDMUX_ROOT", "PDMUX_PROJECT_ROOT", "PDMUX_DATA", "PDMUX_IO", "PDMUX_COMMIT", "PDMUX_JOB_NAME",
                   "PDMUX_ENERGY_SAMPLER")


def env_problems(environ):
    """ABORT_ENV: a set PDMUX_* that is neither inherited-allowed nor a knob the runner overwrites."""
    own = {"PDMUX_TELEMETRY_PATH", "PDMUX_RUN_ID", "PDMUX_WORKLOAD_ID", "PDMUX_R2_POLICY", "PDMUX_R2_FIXED_DSM",
           "PDMUX_STICKY_PARTITION", "PDMUX_PHASE_EVENTS", "PDMUX_GREEN_READOUT", "PDMUX_GREEN_READOUT_PATH"}
    return sorted(k for k in environ if k.startswith("PDMUX_") and k not in ALLOW_INHERITED and k not in own)


def server_env(base_env, d, tel_path, run_id, green_path):
    env = {k: v for k, v in base_env.items() if not (k.startswith("PDMUX_") and k not in ALLOW_INHERITED)}
    for k in UNSET:
        env.pop(k, None)
    env.update(PDMUX_TELEMETRY_PATH=tel_path, PDMUX_RUN_ID=run_id, PDMUX_WORKLOAD_ID=f"n2probe_d{d}",
               PDMUX_R2_POLICY="fixed", PDMUX_R2_FIXED_DSM=str(d), PDMUX_STICKY_PARTITION="1",
               PDMUX_PHASE_EVENTS="1", PDMUX_GREEN_READOUT="1", PDMUX_GREEN_READOUT_PATH=green_path,
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONUNBUFFERED="1")
    env.pop("SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN", None)
    return env


# ---- registered constants: read from the registered reducer (loaded by file path), never re-declared --------------
def load_by_path(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod                     # private name (dataclasses need the module registered)
    spec.loader.exec_module(mod)
    assert os.path.samefile(mod.__file__, path)
    return mod


def sha256_file(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def emit_plan(out_path, calculator=CALCULATOR, reducer=REDUCER):
    """Pre-probe plan (§4-2 / §14-2: the probe runs BEFORE K_POST_PROBE.json exists, so its c_in / O_EA / delta are the
    PRE-PROBE values = the design calculator's derived constants in a fresh process). Writes the file once ('x');
    the runner reads ONLY this file (it never imports the calculator)."""
    calc = load_by_path(calculator, "_n2_plan_calculator")
    red = load_by_path(reducer, "_n2_plan_reducer")
    der = calc.derived_constants()
    b, c = red.BIND_CELL
    cells = {}
    for d in red.PROBE_D:
        e = der["ealpha"][f"{d},{b},{c}"]
        cells[str(d)] = dict(d=d, b=b, c=c, c_in=int(e["c_in"]), O_EA=int(math.ceil(e["O_EA"])), K=int(e["K"]),
                             delta=int(e["delta_on_A2"]), O_EA_raw=e["O_EA"])
    plan = dict(kind="PROBE", calls={str(d): red.PROBE_CALLS[d] for d in red.PROBE_D}, cells=cells,
                calculator_sha256=sha256_file(calculator), reducer_sha256=sha256_file(reducer),
                note="pre-probe design values (N2C-12/18: design forecast, not a measurement)")
    with open(out_path, "x") as fh:
        json.dump(plan, fh, indent=1, sort_keys=True)
    return plan


def plan_problems(plan, reducer_mod, test_calls=None, check_sources=False):
    """ABORT_TRACE / ABORT_PLAN checks of the plan against the registered reducer constants (R11-1); with
    check_sources, a plan emitted from another calculator / reducer byte state is stale (re-emit before the freeze)."""
    bad = []
    if check_sources:
        for key, path in (("calculator_sha256", CALCULATOR), ("reducer_sha256", REDUCER)):
            if plan.get(key) != sha256_file(path):
                bad.append(f"stale plan: {key} differs from {os.path.basename(path)} on disk")
    calls = {int(k): v for k, v in plan.get("calls", {}).items()}
    want = dict(reducer_mod.PROBE_CALLS) if test_calls is None else dict(test_calls)
    if calls != want:
        bad.append(f"plan calls {calls} != registered PROBE_CALLS {want}")
    for d in reducer_mod.PROBE_D:
        cell = plan.get("cells", {}).get(str(d))
        if cell is None:
            bad.append(f"plan has no cell for d={d}"); continue
        if (cell["b"], cell["c"]) != tuple(reducer_mod.BIND_CELL):
            bad.append(f"d={d}: (b, c) {(cell['b'], cell['c'])} != BIND_CELL {reducer_mod.BIND_CELL}")
        if cell["K"] != reducer_mod.K_PROBE[d]:
            bad.append(f"d={d}: K {cell['K']} != K_PROBE {reducer_mod.K_PROBE[d]}")
        if cell["c_in"] % lg.CO_INPUT == 0:
            bad.append(f"d={d}: c_in {cell['c_in']} is a multiple of 8192 (would be a co-type span)")
        if cell["c_in"] + cell["O_EA"] > 16384:
            bad.append(f"d={d}: c_in + O_EA {cell['c_in'] + cell['O_EA']} > context 16384")
        try:
            lg.payload([1], 1, "x", coload=True)                       # co-load payload rule (ABORT_TRACE)
        except lg.TraceInvalid as e:
            bad.append(str(e))
    return bad


# ---- stop rule (H-7) --------------------------------------------------------------------------------------------
def attempt_dirs(root):
    out = []
    for name in os.listdir(root) if os.path.isdir(root) else []:
        if name.startswith("probe_attempt_") and name[len("probe_attempt_"):].isdigit():
            out.append((int(name[len("probe_attempt_"):]), os.path.join(root, name)))
    return sorted(out)


def gate_result(attempt_dir, official_only=True):
    """The gate output of an attempt. rev13 G-15: the re-probe decision reads ONLY the official K_POST_PROBE.json (the
    post_probe_gate run on the clean freeze checkout, copied into the attempt directory); GATE_PREVIEW.json is a
    descriptive preview and is read only when official_only=False (attempt summary)."""
    for fn in ("K_POST_PROBE.json",) if official_only else ("K_POST_PROBE.json", "GATE_PREVIEW.json"):
        p = os.path.join(attempt_dir, fn)
        if os.path.exists(p):
            with open(p) as fh:
                return json.load(fh)
    return None


def next_attempt_allowed(results):
    """results: gate results of the previous attempts in order (None = no gate result). Returns (allowed, why).
    H-7 / §14-2 (d): re-probe only if the last attempt allows it; two consecutive re-probe-allowed attempts with the
    same reason_codes SET (never strings) -> stop and report."""
    if not results:
        return True, "first attempt"
    last = results[-1]
    if last is None:
        return False, "previous attempt has no official gate result (run post_probe_gate on the freeze checkout; GATE_PREVIEW does not count)"
    if not last.get("reprobe_allowed"):
        return False, "previous attempt does not allow a re-probe (value-dependent / censored / sealed / irreproducible)"
    if len(results) >= 2 and results[-2] is not None and results[-2].get("reprobe_allowed") \
            and set(results[-2].get("reason_codes", [])) == set(last.get("reason_codes", [])):
        return False, f"two consecutive re-probe-allowed attempts with the same reason_codes {sorted(last.get('reason_codes', []))}: stop and report (H-7)"
    return True, "re-probe allowed"


def new_attempt_dir(root):
    os.makedirs(root, exist_ok=True)
    prev = attempt_dirs(root)
    allowed, why = next_attempt_allowed([gate_result(p) for _n, p in prev])
    if not allowed:
        raise SystemExit(f"STOP_RULE {why}")
    n = (prev[-1][0] + 1) if prev else 1
    path = os.path.join(root, f"probe_attempt_{n}")
    os.mkdir(path)                                            # exclusive: an existing attempt is never reused
    return n, path


# ---- substrate (§3-3, R11-7) ------------------------------------------------------------------------------------
def _read(path):
    try:
        with open(path) as fh:
            return fh.read().strip()
    except OSError:
        return None


def substrate_record(environ=os.environ):
    import platform
    rec = dict(
        clock=dict(monotonic=time.get_clock_info("monotonic").implementation,
                   perf_counter=time.get_clock_info("perf_counter").implementation,
                   contract_ok=n2_marks.clock_contract_ok()),
        asyncio_timeout_is_builtin=(asyncio.TimeoutError is TimeoutError),
        python=platform.python_version(), nproc=os.cpu_count(),
        cgroup_cpu_max=_read("/sys/fs/cgroup/cpu.max"), cgroup_cpu_stat=_read("/sys/fs/cgroup/cpu.stat"),
        hostname=socket.gethostname())
    try:
        import aiohttp
        rec["aiohttp"] = aiohttp.__version__
    except ImportError:
        rec["aiohttp"] = None
    try:
        rec["nvidia_smi"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,uuid,compute_cap,memory.total,driver_version,mig.mode.current",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        rec["nvidia_smi"] = None
    job = environ.get("PDMUX_JOB_NAME")
    data = environ.get("PDMUX_DATA", "/data")
    meta = os.path.join(data, "runs", job, "meta") if job else None
    outputs = {fn: bool(meta) and os.path.exists(os.path.join(meta, fn))
               for fn in ("energy.csv", "power_samples.csv", "sm_clk_samples.csv")}
    running_under_job_entry = bool(job) and any(outputs.values())
    rec["energy_sampler"] = dict(
        state=("on (job_entry.sh)" if running_under_job_entry and environ.get("PDMUX_ENERGY_SAMPLER", "1") != "0"
               else "off"),
        why=("job_entry.sh sampler outputs present" if running_under_job_entry
             else "U-2 Workspace probe: job_entry.sh is not used, no NVML sampler runs (NOTICE 2026-10-08, N2C-61)"),
        PDMUX_ENERGY_SAMPLER=environ.get("PDMUX_ENERGY_SAMPLER"), outputs_present=outputs)
    root = environ.get("PDMUX_PROJECT_ROOT")
    if root:
        try:
            rec["git_head"] = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True,
                                             text=True).stdout.strip()
            rec["git_dirty_tracked"] = subprocess.run(["git", "-C", root, "status", "--porcelain",
                                                       "--untracked-files=no"], capture_output=True,
                                                      text=True).stdout.strip().splitlines()
        except OSError:
            pass
    rec["harness_sha256"] = {fn: sha256_file(os.path.join(HERE, fn)) for fn in HARNESS_FILES
                             if os.path.exists(os.path.join(HERE, fn))}
    for fn in ("n2_probe_reduce.py", "n2_feasibility.py"):
        rec["harness_sha256"][fn] = sha256_file(os.path.join(HERE, fn))
    return rec


def server_log_facts(text):
    """Descriptive (§3-2 / §19 item 8): does the sticky-ON 6-group boot print `cuda graph:` on Decode batch lines, and
    the P3 banners. Per-cell mapping uses wall_<boot>.txt offline; nothing here is a label input."""
    import re
    dec = [ln for ln in text.splitlines() if "Decode batch" in ln]
    mm = re.findall(r"max_mamba_cache_size: *([0-9]+)", text)
    mt = re.findall(r"max_total_num_tokens=([0-9]+)", text)
    return dict(decode_lines=len(dec), cuda_graph_true=sum("cuda graph: True" in ln for ln in dec),
                cuda_graph_false=sum("cuda graph: False" in ln for ln in dec),
                max_mamba_cache_size=(int(mm[-1]) if mm else None), max_total_num_tokens=(int(mt[-1]) if mt else None),
                phase_events_disabled=("PDMUX_PHASE_EVENTS disabled after failure" in text),
                sticky_enabled_line=next((ln for ln in text.splitlines() if "sticky partition ENABLED" in ln), None))


# ---- server handles ---------------------------------------------------------------------------------------------
class ProcessServer:
    """The real SGLang server (Workspace). setsid process group; SIGTERM then SIGKILL; '^sglang::' children (O3)."""

    def __init__(self, python="python", cfg=None, port=None, health_timeout_s=1200.0):
        self.python, self.cfg, self.port = python, cfg, port
        self.health_timeout_s = health_timeout_s
        self.proc = None

    @property
    def base(self):
        return f"http://127.0.0.1:{self.port}"

    def start(self, env, log_path):
        argv = [self.python, "-m", "sglang.launch_server", *server_flags(self.cfg),
                "--host", "127.0.0.1", "--port", str(self.port)]
        self.argv = argv
        self.log = open(log_path, "x")
        self.proc = subprocess.Popen(argv, env=env, stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True)

    def exit_code(self, grace_s=0.0):
        if self.proc is None:
            return None
        try:
            return self.proc.wait(timeout=grace_s) if grace_s > 0 else self.proc.poll()
        except subprocess.TimeoutExpired:
            return None

    async def wait_healthy(self, session):
        deadline = time.monotonic() + self.health_timeout_s
        while time.monotonic() < deadline:
            if self.exit_code() is not None:
                return False
            try:
                async with session.get(f"{self.base}/health") as r:
                    if r.status == 200:
                        return True
            except Exception:
                pass
            await asyncio.sleep(5.0)
        return False

    def log_text(self):
        try:
            with open(self.log.name, errors="replace") as fh:
                return fh.read()
        except OSError:
            return ""

    async def stop(self):
        if self.proc is None or self.proc.poll() is not None:
            return
        try:
            os.killpg(self.proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        for _ in range(60):
            if self.proc.poll() is not None:
                break
            await asyncio.sleep(0.5)
        if self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        subprocess.run(["pkill", "-9", "-f", "^sglang::"], capture_output=True)
        self.log.close()


def free_port(lo=20000, span=800):
    """N3 O1/O2: outside the ephemeral range, no socket in any state on it."""
    import random
    for _ in range(200):
        p = lo + random.randrange(span)
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    raise RuntimeError("no free port")


# ---- one boot ---------------------------------------------------------------------------------------------------
async def run_boot(adir, boot_id, d, n_cells, cell_plan, server, registry, attempt, opts):
    """One sticky-ON boot of the probe. Returns a dict (boot_id, aborted reason or None, cells).
    Every abort is CLASSIFIED WHILE THE SERVER IS STILL IN ITS FAILURE STATE (before the H-5 stop), so the runner's
    own SIGTERM is never observed as a server exit (rule (3) server_down)."""
    marks_p = os.path.join(adir, f"marks_{boot_id}.txt")
    tel_p = os.path.join(adir, f"tel_{boot_id}.jsonl")
    wall_p = os.path.join(adir, f"wall_{boot_id}.txt")                 # (time.time, monotonic) pairs (§3-2 R-5)
    mw = n2_marks.MarkWriter(marks_p, registry)
    out = dict(boot_id=boot_id, d=d, planned=n_cells, cells=[], aborted=None)
    stop = opts["stop"]
    grace = opts.get("exit_grace_s", lg.SERVER_EXIT_GRACE_S)

    def classify(exc):
        if stop.get("flag") and not isinstance(exc, (n2_marks.Classified, n2_marks.CellAborted)):
            return "runner_stop"
        return lg.classify_exception(exc, lambda: server.exit_code(grace))

    def wall(tag, cid):
        with open(wall_p, "a") as fh:
            fh.write(f"{tag} {cid} {time.time()!r} {time.monotonic()!r}\n")

    def cpu(tag):
        with open(os.path.join(adir, f"cpu_{boot_id}.txt"), "a") as fh:
            fh.write(f"# {tag} mono={time.monotonic()!r}\n{_read('/sys/fs/cgroup/cpu.stat') or 'cpu.stat NA'}\n")

    spec = lg.CellSpec(d=d, b=cell_plan["b"], c=cell_plan["c"], c_in=cell_plan["c_in"], o_ea=cell_plan["O_EA"],
                       k=cell_plan["K"], delta=cell_plan["delta"], kind="PROBE")
    open(tel_p, "x").close()                                            # one telemetry file per boot (R-3)
    session = lg.new_session(**opts.get("session", {}))
    scope = mw.boot({d: n_cells}, classify)                              # boot_plan BEFORE the server starts
    try:
        with scope:
            env = server_env(opts.get("environ", os.environ), d, os.path.abspath(tel_p),
                             f"n2probe_a{attempt}_{boot_id}", os.path.abspath(os.path.join(adir, f"green_{boot_id}.json")))
            server.start(env, os.path.join(adir, f"srv_{boot_id}.log"))
            try:
                if not await server.wait_healthy(session):
                    if server.exit_code(grace) is not None:
                        raise lg.ServerExited("server exited before /health")
                    raise lg.BootHealthTimeout("server did not become healthy")
                if opts.get("wiring_check", True):
                    txt = server.log_text()
                    want = f"fixed target index={STICKY_IDX[d]}"
                    if "sticky partition ENABLED" not in txt or want not in txt:
                        raise lg.WiringCheckFailed(f"sticky wiring: '{want}' not in the server log")
                base = server.base
                warm = lg.decode_prompt("WARMUP", 0, 0, WARMUP_SEED, cell_plan["c_in"])
                for j in range(opts.get("warmup_n", WARMUP_N)):                 # concurrency 1, outside every cell
                    await lg.simple_request(session, base, warm, WARMUP_OUT, f"a{attempt}-{boot_id}-warm{j}")
                tail = lg.TelemetryTail(tel_p)
                for k in range(n_cells):
                    if opts.get("deadline") is not None and time.monotonic() > opts["deadline"]:
                        stop["flag"] = True
                    if stop.get("flag"):
                        raise lg.RunnerStop("stop requested / deadline before the next cell")
                    cid = f"a{attempt}-{boot_id}-k{k:02d}"
                    spec.r = k + 1
                    cpu(f"before {cid}")
                    wall("cell_start", cid)
                    info = await lg.run_probe_cell(session, base, mw, spec, cid, f"call-{cid}", classify, tail,
                                                   **opts.get("cell_kw", {}))
                    wall("cell_end", cid)
                    cpu(f"after {cid}")
                    out["cells"].append(info)
                await lg.simple_request(session, base, lg.seed_tokens(lg.seed_of(f"n2|F6|{boot_id}"), F6_INPUT),
                                        F6_OUT, f"a{attempt}-{boot_id}-f6")
            except n2_marks.CellAborted:
                raise
            except Exception as e:                       # classify NOW, before the server is stopped (H-4 (3))
                raise n2_marks.Classified(classify(e), e) from e
            finally:
                if server.exit_code() is None:
                    await asyncio.sleep(2 * opts.get("flush_interval_s", FLUSH_INTERVAL_S))   # H-5
                out["t_stop"] = time.monotonic()
                await server.stop()
    except (n2_marks.CellAborted, n2_marks.Classified) as e:
        out["error"] = repr(e.cause)
    finally:
        out["aborted"] = scope.aborted
        try:
            out["server_log"] = server_log_facts(server.log_text())
        except Exception as e:                                               # descriptive only
            out["server_log"] = dict(error=repr(e))
        await session.close()
    return out


# ---- reduce + gate preview ---------------------------------------------------------------------------------------
def reduce_attempt(adir, boot_ids, expected_calls=None):
    """§14-2 (c): the reducer (loaded by file path) runs with the attempt directory as the working directory and
    RELATIVE raw paths; probe.json = to_json(...) bytes, written once."""
    red = load_by_path(REDUCER, "_n2_registered_reducer_run")
    cwd = os.getcwd()
    os.chdir(adir)
    try:
        boots = [dict(boot_id=b, telemetry=f"tel_{b}.jsonl", marks=f"marks_{b}.txt") for b in boot_ids]
        probe = red.reduce_probe(boots, expected_calls)
        text = red.to_json(probe)
        with open("probe.json", "x") as fh:
            fh.write(text)
    finally:
        os.chdir(cwd)
    return probe


def gate_preview(adir, freeze=None):
    argv = [sys.executable, CALCULATOR, "--probe", "probe.json", "--out", "GATE_PREVIEW.json"]
    if freeze:
        argv += ["--freeze", freeze]
    r = subprocess.run(argv, cwd=adir, capture_output=True, text=True,
                       env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    with open(os.path.join(adir, "GATE_PREVIEW.log"), "x") as fh:
        fh.write(f"rc={r.returncode}\n{r.stdout}\n{r.stderr}")
    return gate_result(adir, official_only=False), r.returncode


# ---- one attempt ------------------------------------------------------------------------------------------------
async def run_attempt(root, plan, make_server, opts):
    n, adir = new_attempt_dir(root)
    sub = substrate_record(opts.get("environ", os.environ))
    sub["plan_sha256"] = opts.get("plan_sha256")
    with open(os.path.join(adir, "substrate.json"), "x") as fh:
        json.dump(sub, fh, indent=1, sort_keys=True)
    if not sub["clock"]["contract_ok"]:
        raise SystemExit("ABORT_CLOCK time.monotonic / time.perf_counter are not the same CLOCK_MONOTONIC")
    registry = n2_marks.IdRegistry()
    calls = {int(k): v for k, v in plan["calls"].items()}
    boots, results = [], []
    stop = opts.setdefault("stop", {})
    for d in opts.get("boot_order", BOOT_ORDER):
        if d not in calls:
            continue
        boot_id = f"b{d}"
        res = await run_boot(adir, boot_id, d, calls[d], plan["cells"][str(d)], make_server(), registry, n, opts)
        boots.append(boot_id)
        results.append(res)
        if res["aborted"] is not None and _reason_class(res["aborted"], len(res["cells"])) != "VALUE_INDEPENDENT":
            break                                    # VD abort: the probe is already rejected, no further GPU time
    probe = reduce_attempt(adir, boots, opts.get("expected_calls"))
    preview, rc = (gate_preview(adir, opts.get("freeze")) if opts.get("gate_preview", True) else (None, None))
    summary = dict(attempt=n, boots=[{k: v for k, v in r.items() if k != "cells"} | {"n_cells": len(r["cells"])}
                                     for r in results],
                   probe_status=probe["probe_status"], reason_codes=probe["reason_codes"],
                   gate_preview=(None if preview is None else {k: preview.get(k) for k in
                                 ("K", "seal_allowed", "reprobe_allowed", "reason_codes", "reasons")}), gate_rc=rc)
    with open(os.path.join(adir, "ATTEMPT_SUMMARY.json"), "x") as fh:
        json.dump(summary, fh, indent=1, sort_keys=True, default=str)
    return adir, summary


def _reason_class(reason, n_cells=0):
    red = load_by_path(REDUCER, "_n2_reason_class_reducer")
    if reason in red.BOOT_ONLY_VI_REASONS and n_cells == 0:                 # rev13 G-8 (same rule as the reducer)
        return "VALUE_INDEPENDENT"
    return red.ABORT_REASONS.get(reason, "VALUE_DEPENDENT")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--emit-plan", metavar="OUT", help="write the pre-probe PROBE_PLAN.json (CPU, once) and exit")
    ap.add_argument("--stop-rule", metavar="PROBE_ROOT", help="print whether a new attempt is allowed and exit")
    ap.add_argument("--probe-root", help="directory holding probe_attempt_<n>/")
    ap.add_argument("--plan", default=PLAN_DEFAULT)
    ap.add_argument("--freeze", help="FREEZE_N2_PROBE.json for the gate preview")
    ap.add_argument("--python", default="python")
    ap.add_argument("--deadline-s", type=float, default=None, help="intentional stop after this many seconds (runner_stop)")
    ap.add_argument("--no-gate-preview", action="store_true")
    a = ap.parse_args(argv)
    if a.emit_plan:
        print(json.dumps(emit_plan(a.emit_plan), indent=1, sort_keys=True))
        return 0
    if a.stop_rule:
        ok, why = next_attempt_allowed([gate_result(p) for _n, p in attempt_dirs(a.stop_rule)])
        print(("ALLOWED " if ok else "STOP ") + why)
        return 0 if ok else 3
    if not a.probe_root:
        ap.error("--probe-root is required")
    bad_env = env_problems(os.environ)
    if bad_env:
        print(f"ABORT_ENV unregistered PDMUX_* set: {bad_env}")
        return 2
    cfg = os.path.join(ENG, CONFIG_REL)
    if sha256_file(cfg) != CONFIG_DIGEST:
        print("ABORT_REG_DRIFT A-agn config bytes differ from the registered digest")
        return 3
    with open(a.plan) as fh:
        plan = json.load(fh)
    red = load_by_path(REDUCER, "_n2_plan_check_reducer")
    probs = plan_problems(plan, red, check_sources=True)
    if probs:
        print("ABORT_PLAN " + "; ".join(probs))
        return 3
    opts = dict(plan_sha256=sha256_file(a.plan), freeze=a.freeze, gate_preview=not a.no_gate_preview,
                deadline=(time.monotonic() + a.deadline_s) if a.deadline_s else None, stop={})

    def on_signal(*_):
        opts["stop"]["flag"] = True
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    adir, summary = asyncio.run(run_attempt(a.probe_root, plan,
                                            lambda: ProcessServer(a.python, cfg, free_port()), opts))
    print(json.dumps(dict(attempt_dir=adir, **summary), indent=1, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
