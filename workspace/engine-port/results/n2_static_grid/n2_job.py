#!/usr/bin/env python3
"""N2(3-a) Job runner -- the boot loop of a VESSL measurement Job (PREREG_N2_3A rev16 §3, §5, §6-4, §11-1/2,
§14-2 G-1…G-13 / H-4…H-6, §14-3). Called by n2_capacity.sh AFTER its preflight; never by hand on a GPU.

HARNESS-SEAL file (rev17 Job harness). Job-PROFILE only: `--job EA` is refused (ABORT_EA_NOT_IMPLEMENTED) until the
Job-EA cell client / rho_off assembly (H15-3) is implemented and sealed.

Per boot (one telemetry file per boot, R-3):  boot_plan <D> <n> (before the server starts) -> server launch -> /health
-> §11-1 wiring (log ENABLED line + green read-out; failure = boot_abort wiring_check_failed + Job abort exit 6) ->
§3-2 P3 (ii) over all boots so far (exit 8) and capture-bs ⊇ {1,8,32,48} (exit 3) -> warm-up (§3-1 / G-3 (a): 8
requests, concurrency 1, label 'n2|WARMUP|0|0|7', input c_in(boot, 32, 8192), output 64) inside warmup_start/_end ->
§11-2 greedy probe (ON-D r1, OFF16 r1; empty output = exit 9) -> cells (n2_loadgen.run_profile_cell, cpu.stat before
/after, wall pairs G-11) -> F-6 trailing non-measured request (512 in / 8 out, outside every cell) -> H-5 idle
2 x flush_interval_s -> stop. Abort policy = G-10 read literally (registered): a cell abort ends its boot
(boot_abort with the cell's reason); after a VALUE_INDEPENDENT boot death the next boot runs; after a VALUE_DEPENDENT
abort no further boot runs (CONTRACT GAP reported: G-10's rationale is probe-specific).
Admission (§14-3 STOP rule): before every boot, remaining = cell deadline - now >= that boot's worst-corner cost
(N2_REGISTERED budget, U-1 (b)); the first refusal stops the Job -- that boot and every later boot are
NOT_ATTEMPTED_BUDGET. The cell deadline = cell-section start + PROFILE_b cap (HARNESS READING: §14-2 "셀 마감 하드캡"
counted from the start of the cell section, not from container start -- the table's "마지막 boot 직전 여유" row
reproduces only with this reading; reported). SIGTERM (watchdog) -> stop flag + current cell cancelled -> runner_stop.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import n2_loadgen as lg      # noqa: E402
import n2_marks              # noqa: E402
import n2_plan               # noqa: E402

FLUSH_INTERVAL_S = 1.0                 # engine AsyncJsonlTelemetry default; H-5 waits 2 x this
WARMUP_N, WARMUP_LABEL, WARMUP_OUT = 8, "n2|WARMUP|0|0|7", 64
F6_INPUT, F6_OUT = 512, 8
HEALTH_TIMEOUT_S = 1200.0              # G-7


class JobAbort(Exception):
    def __init__(self, code, tag, msg):
        super().__init__(f"{tag} {msg}")
        self.code, self.tag, self.msg = code, tag, msg


def _reducer():
    spec = importlib.util.spec_from_file_location("_n2_job_reducer", n2_plan.REDUCER_FILE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_n2_job_reducer"] = mod
    spec.loader.exec_module(mod)
    return mod


RED = _reducer()


def reason_class(reason, n_cells):
    """§14-2 reason table + G-8 (the reducer's registered tables, not re-declared)."""
    if reason in RED.BOOT_ONLY_VI_REASONS and n_cells == 0:
        return RED.VALUE_INDEPENDENT
    return RED.ABORT_REASONS.get(reason, RED.VALUE_DEPENDENT)


def _read(path):
    try:
        with open(path) as fh:
            return fh.read()
    except OSError:
        return None


class ProcessServer:
    """The real SGLang server: setsid process group; SIGTERM then SIGKILL; '^sglang::' children (N3 O3)."""

    def __init__(self, cfg, port, python="python", health_timeout_s=HEALTH_TIMEOUT_S):
        self.cfg, self.port, self.python, self.health_timeout_s = cfg, port, python, health_timeout_s
        self.proc = None
        self.log = None

    @property
    def base(self):
        return f"http://127.0.0.1:{self.port}"

    def settle(self, baseline_mib, record_path, max_s=180.0):
        """N3 O4 (RECORDED, not a gate): before a launch wait until no SGLang process is alive and GPU used memory
        is back at the pre-first-boot baseline + 64 MiB, at most `max_s`."""
        t0, used, alive = time.monotonic(), None, None
        while time.monotonic() - t0 < max_s:
            alive = subprocess.run(["pgrep", "-f", "sglang.launch_server|^sglang::"], capture_output=True).returncode == 0
            used = gpu_used_mib()
            if not alive and used is not None and baseline_mib is not None and used <= baseline_mib + 64:
                break
            time.sleep(5.0)
        with open(record_path, "a") as fh:
            fh.write(f"settle used_mib={used} base_mib={baseline_mib} sglang_alive={alive} "
                     f"waited_s={time.monotonic() - t0:.1f}\n")

    def start(self, env, log_path):
        argv = [self.python, "-m", "sglang.launch_server", *n2_plan.server_flags(self.cfg),
                "--host", "127.0.0.1", "--port", str(self.port)]
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
        return _read(self.log.name) or ""

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


def gpu_used_mib():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"],
                             capture_output=True, text=True, timeout=30).stdout.strip().splitlines()
        return int(out[0]) if out else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def free_port(lo=20000, span=800):
    """N3 O1/O2: a port outside the ephemeral range, no socket in any state on it."""
    import random
    import socket
    for _ in range(200):
        p = lo + random.randrange(span)
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    raise RuntimeError("no free port")


def warmup_c_in(unit, reg):
    """G-3 (a) 'input = c_in(d)' read for the Job: c_in of the boot's own (32, 8192) cell (ON-D: D, OFF: 108, OBS:
    44) -- never a multiple of 8192 (asserted). HARNESS READING of an unregistered Job detail (reported)."""
    c = int(reg["derived"]["profile"][f"{unit['alone_D']},32,8192"]["c_in"])
    assert c % 8192 != 0
    return c


# --------------------------------------------------------------------------------------------- one boot
async def run_boot(out, unit, server, registry, opts):
    """One registered boot. Returns the boot record; raises JobAbort for §11-1 / P3 / CG_BS / correctness aborts
    (after writing boot_abort and stopping the server)."""
    b = unit["name"]
    p = {k: os.path.join(out, f"{k}_{b}.{ext}") for k, ext in
         (("marks", "txt"), ("tel", "jsonl"), ("srv", "log"), ("green", "json"), ("wall", "txt"), ("cpu", "txt"),
          ("client", "jsonl"), ("greedy", "json"), ("boot", "json"))}
    lt = time.localtime()
    rec = dict(name=b, unit={k: v for k, v in unit.items() if k != "cells"}, n_planned=len(unit["cells"]),
               cells_done=[], aborted=None, abort_class=None, job_abort=None,
               tz_gmtoff_s=lt.tm_gmtoff, tz_name=lt.tm_zone)       # the server log stamps are local time (§3-2)
    mw = n2_marks.MarkWriter(p["marks"], registry)
    stop = opts["stop"]
    grace = opts.get("exit_grace_s", lg.SERVER_EXIT_GRACE_S)
    wall = opts.get("wall", time.time)
    reg = opts["reg"]

    def classify(exc):
        if stop.get("flag") and not isinstance(exc, (n2_marks.Classified, n2_marks.CellAborted)):
            return "runner_stop"
        return lg.classify_exception(exc, lambda: server.exit_code(grace))

    def wall_pair(tag, cid):
        with open(p["wall"], "a") as fh:
            fh.write(f"{tag} {cid} {wall()!r} {time.monotonic()!r}\n")

    def cpu(tag):
        with open(p["cpu"], "a") as fh:
            fh.write(f"# {tag} mono={time.monotonic()!r}\n{_read('/sys/fs/cgroup/cpu.stat') or 'cpu.stat NA'}\n")

    open(p["tel"], "x").close()
    session = lg.new_session(**opts.get("session", {}))
    scope = mw.boot({unit["D"]: len(unit["cells"])}, classify)        # boot_plan BEFORE the server starts
    job_abort = None
    try:
        with scope:
            env = n2_plan.boot_env(opts.get("environ", os.environ), unit, os.path.abspath(p["tel"]),
                                   os.path.abspath(p["green"]), f"n2_{b}_{opts.get('job_id', 'local')}")
            rec["t_admit_mono"] = time.monotonic()
            if hasattr(server, "settle"):                                   # real server only (O4, recorded)
                server.settle(opts.get("gpu_base_mib"), os.path.join(out, "SETTLE.txt"))
            server.start(env, p["srv"])
            try:
                if not await server.wait_healthy(session):
                    if server.exit_code(grace) is not None:
                        raise lg.ServerExited("server exited before /health")
                    raise lg.BootHealthTimeout("server did not become healthy")
                rec["t_health_mono"] = time.monotonic()
                log = server.log_text()
                green = json.loads(_read(p["green"]) or "{}")
                wbad = n2_plan.check_wiring(log, unit["sticky"], unit["D"])
                gbad = n2_plan.check_green(green, unit["sticky"], unit["D"])
                rec["wiring"] = dict(log_problems=wbad, green_problems=gbad)
                if wbad or gbad:
                    job_abort = JobAbort(6, "ABORT_STICKY" if wbad else "ABORT_GREEN", "; ".join(wbad + gbad))
                    raise lg.WiringCheckFailed(job_abort.msg)
                mamba, tokens = n2_plan.banners(log)
                opts["p3_rows"].append((mamba, tokens))
                ok, line = n2_plan.p3_verdict(opts["p3_rows"])
                rec["p3"] = line
                if not ok:
                    job_abort = JobAbort(8, "ABORT_P3", line)
                    raise lg.HarnessError(line)
                cgb = n2_plan.cg_bs_problems(log)
                rec["capture_bs"] = n2_plan.capture_bs(log)
                if cgb:
                    job_abort = JobAbort(3, "ABORT_CG_BS", "; ".join(cgb))
                    raise lg.HarnessError(job_abort.msg)
                base = server.base
                wcin = warmup_c_in(unit, reg)
                warm = lg.seed_tokens(lg.seed_of(WARMUP_LABEL), wcin)
                with mw.warmup(b):                                         # G-3 (a), outside every cell
                    for j in range(opts.get("warmup_n", WARMUP_N)):
                        await lg.simple_request(session, base, warm, WARMUP_OUT, f"{b}-warm{j}")
                if unit["greedy"]:                                         # §11-2
                    outs = await lg.run_greedy(session, base, b)
                    with open(p["greedy"], "x") as fh:
                        json.dump(dict(boot=b, kind=unit["kind"], D=unit["D"], outputs=outs), fh, sort_keys=True)
                    if any(not (o.get("text") or o.get("output_ids")) for o in outs):
                        job_abort = JobAbort(9, "ABORT_CORRECTNESS", "empty greedy output")
                        raise lg.HarnessError("empty greedy output")
                tail = lg.TelemetryTail(p["tel"])
                for cell in unit["cells"]:
                    if opts.get("deadline") is not None and time.monotonic() > opts["deadline"]:
                        stop["flag"] = True
                    if stop.get("flag"):
                        raise lg.RunnerStop("stop requested / deadline before the next cell")
                    spec = lg.ProfileSpec(d=unit["D"], b=cell["b"], c=cell["c"], c_in=cell["c_in"], o=cell["O"],
                                          r=cell["r"], seed_kind=cell["seed_kind"])
                    cpu(f"before {cell['cell_id']}")
                    wall_pair("cell_start", cell["cell_id"])
                    task = asyncio.ensure_future(lg.run_profile_cell(
                        session, base, mw, spec, cell["cell_id"], f"call-{cell['cell_id']}", classify, tail,
                        **opts.get("cell_kw", {})))
                    opts["current"] = task
                    try:
                        info = await task
                    except n2_marks.CellAborted as e:
                        with open(p["client"], "a") as fh:
                            fh.write(json.dumps(dict(cell_id=cell["cell_id"], aborted=e.reason,
                                                     exception=type(n2_marks._root(e.cause)).__name__)) + "\n")
                        raise
                    finally:
                        opts["current"] = None
                        wall_pair("cell_end", cell["cell_id"])
                        cpu(f"after {cell['cell_id']}")
                    with open(p["client"], "a") as fh:
                        fh.write(json.dumps(info) + "\n")
                    rec["cells_done"].append(cell["cell_id"])
                await lg.simple_request(session, base, lg.seed_tokens(lg.seed_of(f"n2|F6|{b}"), F6_INPUT),
                                        F6_OUT, f"{b}-f6")                 # §6-4 F-6, outside every cell
            except n2_marks.CellAborted:
                raise
            except Exception as e:                                         # classify NOW (H-4 (3)), before stop
                raise n2_marks.Classified(classify(e), e) from e
            finally:
                if server.exit_code() is None:
                    await asyncio.sleep(2 * opts.get("flush_interval_s", FLUSH_INTERVAL_S))   # H-5
                rec["t_stop_mono"] = time.monotonic()
                await server.stop()
    except (n2_marks.CellAborted, n2_marks.Classified) as e:
        rec["error"] = repr(e.cause)
    finally:
        rec["aborted"] = scope.aborted
        if scope.aborted is not None:
            rec["abort_class"] = reason_class(scope.aborted, len(rec["cells_done"]))
        try:
            log = server.log_text()
            rec["server_log"] = dict(r7=("PDMUX_PHASE_EVENTS disabled after failure" in log),
                                     decode_lines=log.count("Decode batch"),
                                     banners=n2_plan.banners(log))
        except Exception as e:                                             # descriptive only
            rec["server_log"] = dict(error=repr(e))
        await session.close()
        if job_abort is not None:
            rec["job_abort"] = dict(code=job_abort.code, tag=job_abort.tag, msg=job_abort.msg)
        with open(p["boot"], "x") as fh:
            json.dump(rec, fh, indent=1, sort_keys=True, default=str)
    if job_abort is not None:
        raise job_abort
    return rec


# --------------------------------------------------------------------------------------------- the Job
async def run_job(out, units, make_server, opts):
    """Boot loop with the §14-3 STOP admission rule. Writes JOB_LEDGER.json. Returns the ledger."""
    reg = opts["reg"]
    job = units[0]["job"] if units else "PROFILE"
    opts.setdefault("stop", {})
    opts.setdefault("p3_rows", [])
    registry = n2_marks.IdRegistry()
    if opts.get("deadline") is None:
        opts["deadline"] = time.monotonic() + n2_plan.budget(job, reg)["cap_s"]
    ledger, stopped = [], None
    loop = asyncio.get_running_loop()

    def on_term(*_):
        opts["stop"]["flag"] = True
        cur = opts.get("current")
        if cur is not None and not cur.done():
            cur.cancel()
    opts["on_term"] = on_term                   # the watchdog's SIGTERM path (tests call it directly)
    if opts.get("install_signals", True):
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, on_term)
            except (NotImplementedError, RuntimeError):
                pass
    try:
        for u in units:
            need = n2_plan.need_s(job, reg, u["name"])
            remaining = opts["deadline"] - time.monotonic()
            row = dict(unit=u["name"], need_s=need, remaining_s=remaining, n_cells=len(u["cells"]))
            if stopped is not None:
                row["status"] = "NOT_ATTEMPTED_BUDGET" if stopped == "budget" else f"NOT_ATTEMPTED_{stopped.upper()}"
                ledger.append(row)
                continue
            if opts["stop"].get("flag"):
                stopped = "budget"                      # watchdog / deadline: the cell section is over (§14-3)
                row["status"] = "NOT_ATTEMPTED_BUDGET"
                ledger.append(row)
                continue
            if not n2_plan.admission(remaining, need):
                stopped = "budget"
                row["status"] = "NOT_ATTEMPTED_BUDGET"
                ledger.append(row)
                continue
            row["status"] = "ADMITTED"
            ledger.append(row)
            res = await run_boot(out, u, make_server(u), registry, opts)
            row.update(aborted=res["aborted"], abort_class=res["abort_class"], cells_done=len(res["cells_done"]))
            if res["aborted"] == "runner_stop":
                stopped = "budget"
            elif res["aborted"] is not None and res["abort_class"] != RED.VALUE_INDEPENDENT:
                stopped = "vd_abort"                    # G-10 (registered, read literally)
    except JobAbort as e:
        ledger.append(dict(job_abort=dict(code=e.code, tag=e.tag, msg=e.msg)))
        raise
    finally:
        with open(os.path.join(out, "JOB_LEDGER.json"), "w") as fh:
            json.dump(dict(job=job, ledger=ledger, stopped=stopped, cap_s=n2_plan.budget(job, reg)["cap_s"]), fh,
                      indent=1, sort_keys=True)
    return ledger


def select_test_units(units, unit_names=None, cells=None):
    """TEST MODE ONLY (n2_capacity.sh refuses these knobs inside a VESSL Job): keep a subset of boots / cells."""
    out = []
    for u in units:
        if unit_names and u["name"] not in unit_names:
            continue
        u = dict(u)
        if cells:
            u["cells"] = [c for c in u["cells"] if (c["b"], c["c"]) in cells]
        out.append(u)
    return out


class ScaledWall:
    """TEST ONLY: a wall clock that advances `scale` times faster than CLOCK_MONOTONIC (shared by the fake engine's
    log stamps and the runner's wall pairs, so short CPU cells span several log seconds)."""

    def __init__(self, scale):
        self.scale, self.w0, self.m0 = float(scale), time.time(), time.monotonic()

    def __call__(self):
        return self.w0 + (time.monotonic() - self.m0) * self.scale


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--job", default="PROFILE")
    ap.add_argument("--out", required=True)
    ap.add_argument("--deadline-epoch", type=float, default=None)
    ap.add_argument("--python", default="python")
    ap.add_argument("--fake", action="store_true", help="TEST MODE: CPU fake engine")
    ap.add_argument("--test-units", default="")
    ap.add_argument("--test-cells", default="")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if a.job != "PROFILE":
        print("ABORT_EA_NOT_IMPLEMENTED Job-EA runner is not implemented (harness seal pending, H15-3)")
        return 3
    reg = n2_plan.load_registered()
    units = n2_plan.units(a.job, reg)
    bad = n2_plan.plan_problems(a.job, units)
    if bad:
        print("ABORT_TRACE " + "; ".join(bad))
        return 3
    if a.test_units or a.test_cells:
        if not a.fake and not a.dry_run:
            print("ABORT_TEST_KNOB_IN_MEASUREMENT --test-units/--test-cells need --fake or --dry-run")
            return 2
        cells = {tuple(int(x) for x in bc.split(",")) for bc in a.test_cells.split(";") if bc} or None
        units = select_test_units(units, set(filter(None, a.test_units.split(","))) or None, cells)
    os.makedirs(a.out, exist_ok=True)
    if a.dry_run:
        cfg = os.path.join(n2_plan.ENG, n2_plan.CONFIG_REL)
        with open(os.path.join(a.out, "DRYRUN_COMMANDS.txt"), "w") as fh:
            for u in units:
                env = n2_plan.boot_env({}, u, f"tel_{u['name']}.jsonl", f"green_{u['name']}.json", "dry")
                fh.write(f"BOOT {u['name']} need_s={n2_plan.need_s(a.job, reg, u['name']):.0f} "
                         f"server: python -m sglang.launch_server {' '.join(n2_plan.server_flags(cfg))}\n")
                fh.write("  env: " + " ".join(f"{k}={v}" for k, v in sorted(env.items()) if k.startswith("PDMUX_"))
                         + "\n")
                for c in u["cells"]:
                    fh.write(f"  cell {c['cell_id']} b={c['b']} c={c['c']} c_in={c['c_in']} O={c['O']} "
                             f"seed=n2|{c['seed_kind']}|{c['b']}|{c['c']}|{c['r']}\n")
        print(f"DRYRUN {len(units)} boots, {sum(len(u['cells']) for u in units)} cells")
        return 0
    opts = dict(reg=reg, job_id=os.environ.get("PDMUX_JOB_NAME", "local"))
    if a.deadline_epoch is not None:
        opts["deadline"] = time.monotonic() + (a.deadline_epoch - time.time())
    if a.fake:
        import fake_n2_engine as fk
        sw = ScaledWall(float(os.environ.get("PDMUX_N2_TEST_WALL_SCALE", "20")))
        opts.update(wall=sw, flush_interval_s=0.02, warmup_n=1,
                    cell_kw=dict(drain_poll_s=0.02, drain_settle_s=0.05, drain_timeout_s=20))
        kw = dict(chunk_every=1, tau_s=0.0005, burst_span_s=0.001, wall=sw, decode_log_every=8)

        def make(u):
            return fk.FakeServerHandle(dict(kw), job_mode=True)
    else:
        cfg = os.path.join(n2_plan.ENG, n2_plan.CONFIG_REL)
        opts["gpu_base_mib"] = gpu_used_mib()

        def make(u):
            return ProcessServer(cfg, free_port(), a.python)
    try:
        asyncio.run(run_job(a.out, units, make, opts))
    except JobAbort as e:
        print(f"{e.tag} {e.msg}")
        return e.code
    print("N2_JOB_RUN_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
