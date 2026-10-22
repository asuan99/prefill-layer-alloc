#!/usr/bin/env python3
"""Cost-term harness, stage 1 -- Job runner and analyzer.  STATUS: CPU-tested against the fake engine only.
NOTHING HERE HAS RUN ON A GPU; writing it is not a PASS of anything and makes no performance claim.

  ct_runner.py plan  --job J-H1|J-A|J-B2 [--round 1|2]          print the boot plan (order, argv, env keys, cells)
  ct_runner.py run   --job ... --out DIR [--round N] [--variant-src DIR] [--variant-root DIR]
  ct_runner.py analyze DIR                                       -> DIR/analysis.json (CPU)
  ct_runner.py spec-sha                                          ct_spec digest (seal input)

Job entry (experiment-runner): costterms_job.sh sets PDMUX_COSTTERMS_JOB / _ROUND and calls `run`; it refuses to run
unless the harness is sealed (PDMUX_COSTTERMS_SEALED=1) or PDMUX_COSTTERMS_TEST_MODE=1.

Per boot (PREREG §2-2): boot -> facts from the log (label / cap / engine / sticky / instrumentation / capture checks)
-> realized SM (§1-8) -> feasibility of EVERY registered cell (per-cell O, rev3.4 §3-1) written BEFORE any cell -> T4
predictions A and B written
and hashed into events.jsonl BEFORE the first fill -> warm-up + correctness -> cells in seeded random order (T1 bursts
are aborted after the ramp-aware window, decision 28 A-1) -> trailer +
idle 2 x flush -> stop -> DONE (resume skips boots with DONE).  The analyzer re-derives every label from the logs; it
never takes model / load format / cap / D from the plan.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shlex
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ct_spec as S  # noqa: E402
import ct_extract as X  # noqa: E402
import ct_fit as F  # noqa: E402
import ct_client as C  # noqa: E402

ENG = HERE.parents[2]
CONFIG_PATH = HERE / S.CONFIG_CANDIDATE
KEEP_ENV = ("PDMUX_ROOT", "PDMUX_PROJECT_ROOT", "PDMUX_DATA", "PDMUX_IO", "PDMUX_COMMIT", "PDMUX_JOB_NAME",
            "PDMUX_ENERGY_SAMPLER")


def _test_mode():
    return os.environ.get("PDMUX_COSTTERMS_TEST_MODE") == "1"


def dump(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str))
    tmp.replace(path)


def load(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ============================================================================ plans
def _boot(job, rnd, boot_id, kind, model, engine, cap, *, D=None, variant=4, dummy=False, t1=(), t1_reps=1, t2=(),
          t2p=(), t4=(), o_family="JA"):
    mode = (f"D{D}" if engine == "pdmux_sticky" else "108u" if engine == "pdmux_108u" else "108v")
    return {"job": job, "round": rnd, "boot_id": boot_id, "kind": kind, "model": model, "engine": engine, "mode": mode,
            "D": D if engine == "pdmux_sticky" else None, "cap": cap, "variant": variant if model == "nano" else None,
            "dummy": dummy, "t1": [list(c) for c in t1], "t1_reps": t1_reps, "t2": list(t2), "t2p": list(t2p),
            "t4": [list(f) for f in t4], "o_family": o_family,
            "t1_O": {f"{bs}x{c}": S.o_tokens(bs, c, o_family) for bs, c in t1}}


def t2_axis(mode):
    """Decision 28 A-2: T2 ("no other load") has a known prefill SM only where prefill runs on 108 SM -- 108u / 108v.
    In a sticky-D boot an empty decode batch sends prefill to group 0 (108 SM) as well, so a T2 there would be a 108-SM
    value under a D label; the plan refuses it (None)."""
    return S.TOTAL_SM if mode in S.T2_MODES else None


def plan_ja(rnd):
    """J-A round rnd (stage-1 part): 7 mode boots (T1 + T2' everywhere, T2 only in 108u / 108v -- decision 28 A-2),
    cap-invariance boot (round 1 only), OFF diagnostic boots (D44 PD-mux with T2', 108v with T2).  T3 boots (D24/D44,
    TEL ON) are stage 2 and listed in plan['deferred']."""
    job = f"J-A{rnd}"
    boots = []
    for d in S.D_GRID:
        boots.append(_boot(job, rnd, f"ja{rnd}_D{d}", "T1T2", "nano", "pdmux_sticky", S.CAP_MAIN, D=d,
                           t1=X.t1_cells_for_mode(f"D{d}"), t2p=S.T2P_L_GRID))
    boots.append(_boot(job, rnd, f"ja{rnd}_108u", "T1T2", "nano", "pdmux_108u", S.CAP_MAIN,
                       t1=X.t1_cells_for_mode("108u"), t2=S.T2_L_GRID, t2p=S.T2P_L_GRID))
    boots.append(_boot(job, rnd, f"ja{rnd}_108v", "T1T2", "nano", "vanilla", S.CAP_MAIN,
                       t1=X.t1_cells_for_mode("108v"), t2=S.T2_L_GRID, t2p=S.T2P_L_GRID))
    boots.append(_boot(job, 1, "ja1_capinv_D44", "T1T2", "nano", "pdmux_sticky", S.CAP_SMALL, D=44, t1=S.CAP_INV_CELLS))
    boots.append(_boot(job, rnd, f"ja{rnd}_off_D44", "OFF", "nano", "pdmux_sticky", S.CAP_MAIN, D=44,
                       t1=S.OFF_T1_CELLS_JA, t2p=S.OFF_T2_L_JA))
    boots.append(_boot(job, rnd, f"ja{rnd}_off_108v", "OFF", "nano", "vanilla", S.CAP_MAIN,
                       t1=S.OFF_T1_CELLS_JA, t2=S.OFF_T2_L_JA))
    order = list(range(len(boots)))
    random.Random(S.order_seed("J-A|boots")).shuffle(order)
    ordered = [boots[i] for i in order]
    if rnd == 2:      # ABBA: round 2 = round 1 reversed, without the cap-invariance boot
        ordered = [b for b in reversed(ordered) if b["boot_id"] != "ja1_capinv_D44"]
        for b in ordered:
            b["boot_id"] = b["boot_id"].replace("ja1_", "ja2_")
    return {"job": job, "boots": ordered,
            "deferred": ["T3 boots D24/D44 (TEL ON) -- stage 2", "J-A1 cap-invariance boot only in round 1"]}


def plan_jh1():
    """J-H1 (§4'): boot A per dummy variant (vanilla 108v, cap 128) + real a4 + Llama + OFF diagnostic (dummy a4)."""
    boots = []
    for n in S.VARIANTS:
        boots.append(_boot("J-H1", 1, f"jh1_dummy_a{n}", "T1T2", "nano", "vanilla", S.CAP_MAIN, variant=n, dummy=True,
                           t1=S.JH_T1_CELLS, t1_reps=S.JH_T1_REPS, t2=S.JH_T2_L,
                           t4=[(L, S.JH_T4_R) for L in S.JH_T4_L], o_family="VANILLA"))
    boots.append(_boot("J-H1", 1, "jh1_real_a4", "T1T2", "nano", "vanilla", S.CAP_MAIN, variant=4, dummy=False,
                       t1=S.JH_T1_CELLS, t1_reps=S.JH_T1_REPS, t2=S.JH_T2_L, t4=[(L, S.JH_T4_R) for L in S.JH_T4_L],
                       o_family="VANILLA"))
    boots.append(_boot("J-H1", 1, "jh1_llama", "T1T2", "llama", "vanilla", S.CAP_MAIN,
                       t1=S.JH_T1_CELLS, t1_reps=S.JH_T1_REPS, t2=S.JH_T2_L, t4=[(L, S.JH_T4_R) for L in S.JH_LLAMA_T4_L],
                       o_family="VANILLA"))
    boots.append(_boot("J-H1", 1, "jh1_off_dummy_a4", "OFF", "nano", "vanilla", S.CAP_MAIN, variant=4, dummy=True,
                       t1=S.JH_T1_CELLS, t1_reps=S.JH_T1_REPS, t2=S.JH_T2_L, o_family="VANILLA"))
    order = list(range(len(boots)))
    random.Random(S.order_seed("J-H1|boots")).shuffle(order)
    return {"job": "J-H1", "boots": [boots[i] for i in order], "deferred": ["J-H2 (B/C boots, T3) -- stage 2"]}


def plan_jb2():
    """J-B2 (§1-4, O-15 (b), O-8): Nano T4 9 boots (sticky D44 PD-mux) + 1 fused_sym boot."""
    boots = [_boot("J-B2", 1, "jb2_cap48", "T4", "nano", "pdmux_sticky", S.CAP_SMALL, D=S.T4_ARM_D,
                   t4=[(L, S.T4_R[L]) for L in S.T4_L])]
    for L in S.T4_L:
        boots.append(_boot("J-B2", 1, f"jb2_capCA_L{L}", "T4", "nano", "pdmux_sticky", S.T4_CAP_CA[L], D=S.T4_ARM_D,
                           t4=[(L, S.T4_R[L])]))
        boots.append(_boot("J-B2", 1, f"jb2_capCB_L{L}", "T4", "nano", "pdmux_sticky", S.T4_CAP_CB[L], D=S.T4_ARM_D,
                           t4=[(L, S.T4_R[L])]))
    boots.append(_boot("J-B2", 1, "jb2_fused_cap48", "T4", "nano", "vanilla", S.CAP_SMALL,
                       t4=[(L, S.T4_R[L]) for L in S.T4_L]))
    order = list(range(len(boots)))
    random.Random(S.order_seed("J-B2|boots")).shuffle(order)
    return {"job": "J-B2", "boots": [boots[i] for i in order], "deferred": []}


def make_plan(job, rnd=1):
    if job == "J-A":
        return plan_ja(rnd)
    if job == "J-H1":
        return plan_jh1()
    if job == "J-B2":
        return plan_jb2()
    if job in ("J-H2", "J-C"):
        raise NotImplementedError(f"{job} is stage 2: {S.STAGE2_INTERFACE.get(job)}")
    raise ValueError(job)


def plan_problems(plan):
    """Structural checks of a plan (matrix §2-2', caps, T3 placement, registered cell sets)."""
    bad = []
    for b in plan["boots"]:
        if b["kind"] not in S.INSTR:
            bad.append(f"{b['boot_id']}: unknown kind {b['kind']}")
            continue
        if b.get("t3") and not S.INSTR[b["kind"]][3]:
            bad.append(f"{b['boot_id']}: T3 cells in a TEL-OFF boot (matrix §2-2')")
        if b["t4"] and b["kind"] == "OFF":
            bad.append(f"{b['boot_id']}: T4 fill in an OFF diagnostic boot")
        if b["kind"] == "T1T2" and b["t1"] and b["cap"] not in (S.CAP_MAIN, S.CAP_SMALL):
            bad.append(f"{b['boot_id']}: T1 boot cap {b['cap']}")
        if b["t2"] and t2_axis(b["mode"]) is None:
            bad.append(f"{b['boot_id']}: T2 in a {b['mode']} boot (decision 28 A-2: T2 only at P = 108; use T2')")
        if any(b["t1_O"].get(f"{bs}x{c}") != S.o_tokens(bs, c, b["o_family"]) for bs, c in b["t1"]):
            bad.append(f"{b['boot_id']}: T1 output length differs from the registered O rule")
        if b["job"].startswith("J-A") and b["kind"] == "T1T2" and b["cap"] == S.CAP_MAIN:
            want = sorted(map(list, X.t1_cells_for_mode(b["mode"])))
            if sorted(b["t1"]) != want:
                bad.append(f"{b['boot_id']}: T1 cell set != registered ({len(b['t1'])} vs {len(want)})")
    return bad


# ============================================================================ server argv / env
def model_args(b, variant_dirs):
    if b["model"] == "llama":
        return ["--model-path", S.LLAMA_ID, "--revision", S.LLAMA_REV]
    if b["dummy"]:
        return ["--model-path", str(variant_dirs[b["variant"]]), "--load-format", "dummy"]
    return ["--model-path", S.NANO_ID, "--revision", S.NANO_REV]


def server_argv(b, variant_dirs):
    a = model_args(b, variant_dirs) + [
        "--trust-remote-code", "--dtype", "bfloat16", "--attention-backend", S.ATTENTION_BACKEND,
        "--disable-radix-cache", "--mem-fraction-static", str(S.MEM_FRACTION), "--max-running-requests", str(b["cap"]),
        "--context-length", str(S.CONTEXT_LENGTH), "--random-seed", str(S.SERVER_SEED), "--disable-piecewise-cuda-graph",
        "--chunked-prefill-size", "-1", "--disable-overlap-schedule", "--max-prefill-tokens", str(S.MAX_PREFILL_TOKENS),
        "--host", "127.0.0.1"]
    if b["model"] == "nano":
        a += ["--max-mamba-cache-size", str(b["cap"])]
    if b["engine"].startswith("pdmux"):
        a += ["--enable-pdmux", "--pdmux-config-path", str(CONFIG_PATH)]
    return a


def boot_env(b, bdir, base=None, run_id="local"):
    base = dict(os.environ if base is None else base)
    env = {k: v for k, v in base.items() if not (k.startswith("PDMUX_") and k not in KEEP_ENV)}
    env.pop("SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN", None)
    env.update(PDMUX_RUN_ID=run_id, PDMUX_WORKLOAD_ID=f"costterms_{b['boot_id']}", PYTHONUNBUFFERED="1", **S.HEARTBEAT_ENV)
    for k in ("FAKE_CT_TRUTH",):
        if k in base:
            env[k] = base[k]
    i1, i46, i5, tel = S.INSTR[b["kind"]]
    if i46:
        env["PDMUX_DECODE_PROBE_PATH"] = str(bdir / "decode_probe.jsonl")
    if i5:
        env["PDMUX_ADMISSION_PATH"] = str(bdir / "admission.jsonl")
    if tel and b["engine"].startswith("pdmux"):
        env["PDMUX_TELEMETRY_PATH"] = str(bdir / "tel.jsonl")
        env["PDMUX_PHASE_EVENTS"] = "1"
    if b["engine"].startswith("pdmux"):
        env.update(PDMUX_ENGINE_MODE="pdmux", PDMUX_R2_POLICY="fixed",
                   PDMUX_R2_FIXED_DSM=str(b["D"] if b["engine"] == "pdmux_sticky" else S.FIXED_DSM_108U),
                   PDMUX_GREEN_READOUT="1", PDMUX_GREEN_READOUT_PATH=str(bdir / "green.json"))
        if b["engine"] == "pdmux_sticky":
            env["PDMUX_STICKY_PARTITION"] = "1"
    else:
        env["PDMUX_ENGINE_MODE"] = "vanilla"
    return env


# ============================================================================ servers
def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class ProcessServer:
    """Real `python -m sglang.launch_server` (or, in test mode only, PDMUX_CT_SERVER_CMD)."""

    def __init__(self, bdir, argv, env, boot_timeout=1200):
        self.bdir, self.argv, self.env, self.boot_timeout = bdir, argv, env, boot_timeout
        self.log = bdir / "server.log"
        self.proc = None

    def start(self):
        self.port = free_port()
        cmd = [sys.executable, "-m", "sglang.launch_server"]
        if _test_mode() and os.environ.get("PDMUX_CT_SERVER_CMD"):
            cmd = shlex.split(os.environ["PDMUX_CT_SERVER_CMD"])
        self.lf = open(self.log, "wb")
        self.proc = subprocess.Popen(cmd + self.argv + ["--port", str(self.port)], stdout=self.lf,
                                     stderr=subprocess.STDOUT, env=self.env, start_new_session=True)
        self.tr = C.HttpTransport(self.port)
        t0 = time.monotonic()
        while time.monotonic() - t0 < self.boot_timeout:
            if self.proc.poll() is not None:
                return "died"
            if self.tr.health():
                return "ok"
            time.sleep(0.1 if _test_mode() else 2)
        return "timeout"

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def log_size(self):
        try:
            return self.log.stat().st_size
        except OSError:
            return 0

    def stop(self):
        if self.proc is not None:
            for sig, wait in ((signal.SIGTERM, 3 if _test_mode() else 20), (signal.SIGKILL, 10)):
                try:
                    os.killpg(self.proc.pid, sig)
                except (ProcessLookupError, PermissionError):
                    pass
                t0 = time.monotonic()
                while time.monotonic() - t0 < wait and self.proc.poll() is None:
                    time.sleep(0.1)
                if self.proc.poll() is not None:
                    break
            self.lf.close()
        return {"returncode": self.proc.poll() if self.proc else None}


class VirtualServer:
    """The fake engine core in-process on a virtual clock (deterministic CPU e2e)."""

    def __init__(self, bdir, argv, env, truth=None):
        import ct_fake_engine as FE
        self.FE, self.bdir, self.argv, self.env, self.truth = FE, bdir, argv, env, truth
        self.log = bdir / "server.log"

    def start(self):
        self.lf = open(self.log, "a", buffering=1)
        try:
            self.core = self.FE.EngineCore(self.argv, self.env, self.truth, self.lf.write)
        except RuntimeError as exc:
            self.lf.write(f"Traceback (most recent call last):\nRuntimeError: {exc}\n")
            self.core = None
            return "died"
        for ln in self.core.boot_lines():
            self.lf.write(f"[boot] {ln}\n")
        self.tr = self.FE.VirtualTransport(self.core)
        return "ok"

    def alive(self):
        return self.core is not None

    def log_size(self):
        self.lf.flush()
        return self.log.stat().st_size

    def stop(self):
        if self.core is not None:
            self.core.flush(self.tr.now())
            self.core.close()
        self.lf.close()
        return {"returncode": 0}


# ============================================================================ variants
def prepare_variants(src, root, plan):
    """Build the dummy variant directories with scripts/compo/make_variant.py (config + tokenizer, no weights)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("ct_make_variant", ENG / "scripts" / "compo" / "make_variant.py")
    mv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mv)
    out = {}
    for n in sorted({b["variant"] for b in plan["boots"] if b["model"] == "nano" and b["dummy"]}):
        d = Path(root) / f"a{n}"
        if not (d / "config.json").exists():
            mv.build_variant(str(src), str(d), n, force=True)
        out[n] = d
    return out


# ============================================================================ one boot
class Ledger:
    def __init__(self, path):
        self.path = path

    def __call__(self, event, what="", **kw):
        with open(self.path, "a") as f:
            f.write(json.dumps({"event": event, "what": what, "t_wall": time.time(), **kw}, default=str) + "\n")


def t4_predict(f, cap, fills):
    mt = X.mem_terms(f)
    if mt is None:
        return None
    return [X.cmem_predictions(mt["pool_bytes"], mt["s"], mt["k"], cap, L, R) for L, R in fills]


def run_boot(b, out, variant_dirs, make_server, run_id="local"):
    bdir = Path(out) / "boots" / b["boot_id"]
    if (bdir / "DONE").exists():
        return load(bdir / "boot_result.json", {})
    bdir.mkdir(parents=True, exist_ok=True)
    for p in bdir.iterdir():                     # a partial boot is redone from scratch
        if p.is_file():
            p.unlink()
    led = Ledger(bdir / "events.jsonl")
    argv = server_argv(b, variant_dirs)
    env = boot_env(b, bdir, run_id=run_id)
    dump({"plan": b, "argv": argv, "pdmux_env": {k: v for k, v in sorted(env.items()) if k.startswith("PDMUX_")},
          "spec_sha256": S.spec_sha256()}, bdir / "boot.json")
    led("boot_start", b["boot_id"])
    srv = make_server(bdir, argv, env)
    st = srv.start()
    res = {"boot_id": b["boot_id"], "start": st, "problems": [], "labels": []}
    if st != "ok":
        res["labels"].append("SERVER_DEATH")
        res["problems"].append(f"boot {st}")
        srv.stop()
        dump(res, bdir / "boot_result.json")
        (bdir / "DONE").write_text("boot failed\n")
        return res
    text = srv.log.read_text(errors="replace")
    cfg_text = CONFIG_PATH.read_text() if b["engine"].startswith("pdmux") else None
    f = X.parse_boot_log(text, env, cfg_text)
    dump(f, bdir / "facts.json")
    led("facts", "boot log parsed")
    probs = X.label_problems(b, f) + X.instrumentation_problems(b["kind"], env, f, None)
    if b["kind"] != "T4" and b["cap"] == S.CAP_MAIN and b["t1"]:
        cg = X.capture_problems(f.get("capture_bs"))
        if cg:
            probs += [f"ABORT_CG_BS: {p}" for p in cg]
    res["problems"] = probs
    aborting = [p for p in probs if p.startswith("ABORT_")]
    green = load(bdir / "green.json") if b["engine"].startswith("pdmux") else None
    sm = X.realized_sm(b, green)
    dump(sm, bdir / "sm.json")
    if aborting:
        res["labels"].append("ABORTED")
        srv.stop()
        dump(res, bdir / "boot_result.json")
        (bdir / "DONE").write_text("aborted\n")
        led("boot_abort", "; ".join(aborting))
        return res
    o_map = {(bs, c): b["t1_O"][f"{bs}x{c}"] for bs, c in b["t1"]}
    feas = X.feasibility([tuple(c) for c in b["t1"]], f, X.logged_cap(f), o_map)
    dump(feas, bdir / "feasibility.json")
    led("feasibility_written", sha256=sha256_file(bdir / "feasibility.json"))
    if b["t4"]:
        pred = t4_predict(f, X.logged_cap(f), [tuple(x) for x in b["t4"]])
        dump(pred, bdir / "t4_prediction.json")
        led("t4_prediction_sealed", sha256=sha256_file(bdir / "t4_prediction.json"))
    tr = srv.tr
    wu = C.run_warmup(tr, b["boot_id"])
    tr.sleep(S.PROBE_WINDOW_S * 1.2)
    recs = X.read_jsonl(bdir / "decode_probe.jsonl")
    cg_hit = any(e.get("event") == "decode_forward" and e.get("graph") == "hit" for e in recs) if recs else \
        ("cuda graph: True" in srv.log.read_text(errors="replace"))
    wu["cg_canary"] = cg_hit
    if not (wu["warmup_ok"] and wu["greedy_ok"] and wu["cg_canary"]):
        res["problems"].append(f"CORRECTNESS: warmup {wu['warmup_ok']} greedy {wu['greedy_ok']} cg_canary {wu['cg_canary']}")
    dump(wu, bdir / "correctness.json")
    led("correctness", json.dumps(wu))
    cells = []
    for c_ in [tuple(x) for x in b["t1"]]:
        fe = next((x for x in feas if (x["bs"], x["c"]) == c_), None)
        if fe is None:                                   # every registered cell must have a feasibility row
            res["problems"].append(f"FEASIBILITY_ROW_MISSING: {c_}")
            continue
        if fe["run"]:
            for k in range(b["t1_reps"]):
                cells.append(("t1", c_, k))
    cells += [("t2", L, 0) for L in b["t2"]] + [("t2p", L, 0) for L in b["t2p"]] + [("t4", tuple(x), 0) for x in b["t4"]]
    rng = random.Random(S.order_seed(f"{b['job']}|{b['boot_id']}|cells"))
    rng.shuffle(cells)
    dump([list(map(str, c)) for c in cells], bdir / "cell_order.json")
    with open(bdir / "cells.jsonl", "a") as fh:
        for kind, what, k in cells:
            if not srv.alive():
                res["labels"].append("SERVER_DEATH")
                break
            off0 = srv.log_size()
            if kind == "t1":
                bs, c = what
                rec = C.run_t1_cell(tr, f"t1_b{bs}_c{c}_k{k}", bs, c, b["round"], o_map[(bs, c)], mark=led)
            elif kind == "t2":
                rec = C.run_t2(tr, f"t2_L{what}", what, b["round"], mark=led)
            elif kind == "t2p":
                rec = C.run_t2prime(tr, f"t2p_L{what}", what, b["round"], mark=led)
            else:
                L, R = what
                rec = C.run_t4_fill(tr, f"t4_L{L}", L, R, X.logged_cap(f) + S.T4_FILL_EXTRA, mark=led)
            rec["log_offsets"] = [off0, srv.log_size()]
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            tr.sleep(0.5)                                   # settle between cells
    res["trailer_ok"] = C.run_trailer(tr, b["boot_id"]) if srv.alive() else False
    res["stop"] = srv.stop()
    files = {"decode_probe": (bdir / "decode_probe.jsonl").exists(), "admission": (bdir / "admission.jsonl").exists(),
             "tel": (bdir / "tel.jsonl").exists()}
    res["problems"] += X.instrumentation_problems(b["kind"], env, None, files)
    dump(res, bdir / "boot_result.json")
    led("boot_end", b["boot_id"])
    (bdir / "DONE").write_text(json.dumps({"files": {p.name: sha256_file(p) for p in sorted(bdir.iterdir())
                                                     if p.is_file() and p.name != "DONE"}}, indent=1))
    return res


def run_job(job, out, rnd=1, variant_src=None, variant_root=None, make_server=None, run_id=None, plan=None):
    """`plan` overrides the registered plan -- CPU tests only (refused outside PDMUX_COSTTERMS_TEST_MODE)."""
    if plan is not None and not _test_mode():
        raise SystemExit("ABORT_PLAN_OVERRIDE: a plan override is a test hook")
    plan = plan or make_plan(job, rnd)
    probs = plan_problems(plan)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    dump({"plan": plan, "problems": probs, "spec_sha256": S.spec_sha256(), "prereg": S.PREREG_REL}, out / "plan.json")
    if probs:
        raise SystemExit(f"ABORT_PLAN: {probs}")
    vdirs = {}
    if any(b["dummy"] for b in plan["boots"]):
        if variant_src is None:
            raise SystemExit("ABORT_VARIANT_SRC: dummy variants need --variant-src (Nano snapshot dir)")
        vdirs = prepare_variants(variant_src, variant_root or (out / "variants"), plan)
    make_server = make_server or (lambda bdir, argv, env: ProcessServer(bdir, argv, env))
    results = [run_boot(b, out, vdirs, make_server, run_id or os.environ.get("PDMUX_JOB_NAME", "local")) for b in plan["boots"]]
    dump(results, out / "boots_result.json")
    sums = []
    for p in sorted(out.rglob("*")):
        if p.is_file() and p.name not in ("SHA256SUMS", "RUN_DONE"):
            sums.append(f"{sha256_file(p)}  {p.relative_to(out)}")
    (out / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    (out / "RUN_DONE").write_text("ok\n")
    return results


# ============================================================================ analysis
def _cells(bdir):
    return [json.loads(x) for x in (bdir / "cells.jsonl").read_text().splitlines() if x.strip()] \
        if (bdir / "cells.jsonl").exists() else []


def _log_slice(text_bytes, offs):
    return text_bytes[offs[0]:offs[1]].decode("utf-8", "replace")


def boot_identity(b, f):
    """Labels FROM THE LOG: model, load format, cap, engine mode, variant (implied attention layers)."""
    arch = f.get("model_arch")
    model = {v: k for k, v in S.ARCH_BY_MODEL.items()}.get(arch)
    mt = X.mem_terms(f)
    n_attn = round(mt["k"] / (2 * 8 * 128 * 2)) if (mt and model == "nano") else None
    lf = f["server_args"].get("load_format")
    pd = f["server_args"].get("enable_pdmux")
    sticky = f.get("sticky_index")
    mode = "108v" if not pd else (f"D{b['D']}" if sticky is not None else "108u")
    tag = "llama" if model == "llama" else f"{'dummy' if lf == 'dummy' else 'real'}_a{n_attn}"
    return {"model": model, "load_format": lf, "cap": X.logged_cap(f), "mode": mode, "tag": tag, "n_attn": n_attn}


def t2p_prefill_sm(b, green, sm):
    """Prefill SM of a T2' prefill (decode busy): sticky D -> realized prefill half of the sticky division; 108u ->
    realized prefill half of the FIXED_DSM_108U division; 108v -> 108 (vanilla prefills first, not concurrent)."""
    if b["engine"] == "vanilla":
        return S.TOTAL_SM
    if b["engine"] == "pdmux_sticky":
        return (sm or {}).get("P_real")
    g = next((r for r in (green or {}).get("groups", []) if r.get("target_decode_sm") == S.FIXED_DSM_108U), None)
    return (((g or {}).get("prefill") or {}).get("green_sm") or {}).get("smCount")


def fit_ctx(w):
    """The ctx a T1 fit uses for one window: the ACTUAL window mean (decision 28 A-1), never the nominal c."""
    return w.get("ctx_mean")


def ledger_order_ok(bdir):
    """T4 predictions must be sealed (hash in the ledger) before the first fill starts, and the file must still match."""
    ev = X.read_jsonl(bdir / "events.jsonl")
    i_seal = next((i for i, e in enumerate(ev) if e.get("event") == "t4_prediction_sealed"), None)
    i_fill = next((i for i, e in enumerate(ev) if e.get("event") == "cell_start" and str(e.get("what", "")).startswith("t4_")), None)
    if i_fill is None:
        return True, "no fill"
    if i_seal is None or i_seal > i_fill:
        return False, "prediction not sealed before the first fill"
    if not (bdir / "t4_prediction.json").exists() or sha256_file(bdir / "t4_prediction.json") != ev[i_seal].get("sha256"):
        return False, "prediction file changed after sealing"
    return True, "sealed before fill"


def analyze_boot(bdir):
    meta = load(bdir / "boot.json")
    b = meta["plan"]
    f = load(bdir / "facts.json")
    res = load(bdir / "boot_result.json", {})
    out = {"boot_id": b["boot_id"], "kind": b["kind"], "job": b["job"], "round": b["round"], "plan_mode": b["mode"],
           "problems": list(res.get("problems", [])), "labels": list(res.get("labels", []))}
    if f is None or "ABORTED" in out["labels"] or res.get("start") != "ok":
        out["usable"] = False
        return out
    ident = boot_identity(b, f)
    out["identity"] = ident
    sm = load(bdir / "sm.json")
    out["sm"] = sm
    if sm and sm.get("label"):
        out["labels"].append(sm["label"])
    out["usable"] = not any(l.startswith(("SM_NOT_REALIZED", "SERVER_DEATH")) for l in out["labels"]) and \
        not any(p.startswith("STICKY_MODE_MISMATCH") for p in out["problems"])
    out["facts_summary"] = {k: f.get(k) for k in ("capture_bs", "kv_cache_tokens", "weight_gib", "max_mamba_cache_size",
                                                  "init_new_token_ratio_derived", "split_forward_token_budget")}
    out["facts_summary"]["max_prefill_tokens"] = (f.get("final") or {}).get("max_prefill_tokens")
    out["mem"] = X.mem_terms(f)
    out["feasibility"] = load(bdir / "feasibility.json", [])
    recs = X.read_jsonl(bdir / "decode_probe.jsonl")
    out["schema_problems"] = X.schema_problems(recs)
    adm = X.read_jsonl(bdir / "admission.jsonl") if (bdir / "admission.jsonl").exists() else None
    logb = (bdir / "server.log").read_bytes()
    capture = f.get("capture_bs")
    t1, t2, t2p, t4 = [], [], [], []
    have_engine = any(e.get("event") == "decode_forward" for e in recs)
    dec = X.decode_records(recs)
    if any(b_["t_host_start"] < a_["t_host_start"] for a_, b_ in zip(dec, dec[1:])):
        dec, presorted = recs, False          # seq order != time order: let t1_window sort and filter itself
    else:
        presorted = True
    for c in _cells(bdir):
        if c["kind"] == "t1":
            if b["kind"] == "OFF" or not have_engine:
                w = X.client_only_window(c)
                w["engine_records"] = "NOT_COLLECTED"
            else:
                w = X.t1_window(dec, c, capture, _log_slice(logb, c["log_offsets"]), presorted=presorted)
            w.pop("raw_s_eng_ms", None)
            w.update(bs=c["bs"], c=c["c"])
            t1.append(w)
        elif c["kind"] == "t2":
            t2.append(X.t2_value(c))
        elif c["kind"] == "t2p":
            v_ = X.t2_value(c)
            v_["coexist_engine"] = X.t2p_coexist(c, dec if have_engine and b["kind"] != "OFF" else [])
            t2p.append(v_)
        elif c["kind"] == "t4":
            t4.append(X.t4_fill(c, recs, adm))
    out.update(t1_windows=t1, t2=t2, t2p=t2p, t4_fills=t4)
    out["t2p_P"] = t2p_prefill_sm(b, load(bdir / "green.json"), sm)
    if t4:
        ok, why = ledger_order_ok(bdir)
        out["t4_prediction_order"] = why
        if not ok:
            out["problems"].append(f"T4_PREDICTION_NOT_SEALED: {why}")
        preds = load(bdir / "t4_prediction.json", [])
        out["t4"] = [F.t4_labels(fl, next(p for p in preds if p["L"] == fl["L"])) for fl in t4]
    return out


def _cell_values(t1, key="S_eng_ms"):
    """(bs, c) -> median over VALID windows of `key` (unit = (boot, cell), A-10).  key may be a callable."""
    acc = {}
    for w in t1:
        v = key(w) if callable(key) else w.get(key)
        if w.get("status") == "VALID" and v is not None:
            acc.setdefault((w["bs"], w["c"]), []).append(v)
    return {k: X.median(v) for k, v in acc.items()}


def _fit_points(t1, cells=None):
    """[(bs, actual ctx, S_eng, nominal c)] per (boot, cell): the fit input of rev3.4 §1-1."""
    vals, ctx = _cell_values(t1), _cell_values(t1, fit_ctx)
    return [(b_, ctx[(b_, c_)], t, c_) for (b_, c_), t in sorted(vals.items())
            if (b_, c_) in ctx and (cells is None or (b_, c_) in cells)]


def analyze(run_dir):
    run_dir = Path(run_dir)
    boots = {}
    for bdir in sorted((run_dir / "boots").iterdir()):
        if (bdir / "boot.json").exists():
            boots[bdir.name] = analyze_boot(bdir)
    A = {"spec_sha256": S.spec_sha256(), "boots": boots, "tables": {}, "labels": {}, "priors": {}}
    usable = {k: v for k, v in boots.items() if v.get("usable")}
    # ---- §1-7 agreement over every VALID engine window
    wins = []
    for v in usable.values():
        for w in v.get("t1_windows", []):
            if w.get("status") == "VALID" and w.get("engine_records") != "NOT_COLLECTED" and v["kind"] != "OFF":
                wins.append({"model": v["identity"]["tag"], "mode": v["identity"]["mode"], "bs": w["bs"], "c": w["c"],
                             "S_eng_ms": w.get("S_eng_ms"), "client_gap_ms": w.get("client_gap_ms")})
    agree = F.agreement_labels(wins)
    A["labels"]["agreement"] = {"|".join(map(str, k)): v for k, v in agree.items()}
    A["tables"]["t1_client_secondary"] = {"|".join(map(str, k)): v for k, v in F.client_secondary_table(wins, agree).items()}
    # ---- T1 tables and fits per (job, round, tag, mode)
    t1_tab, fits = {}, {}
    for v in usable.values():
        if v["kind"] == "OFF" or not v.get("t1_windows"):
            continue
        ident = v["identity"]
        key = f"{v['job']}|r{v['round']}|{ident['tag']}|{ident['mode']}|cap{ident['cap']}"
        vals = _cell_values(v["t1_windows"])
        t1_tab[key] = {"D_real": v["sm"].get("D_real"), "cells": {f"{b_}x{c_}": t for (b_, c_), t in sorted(vals.items())},
                       "ctx_actual": {f"{b_}x{c_}": x for (b_, c_), x in sorted(_cell_values(v["t1_windows"], fit_ctx).items())},
                       "status_counts": _status_counts(v["t1_windows"]),
                       "gpu_ms": {f"{b_}x{c_}": t for (b_, c_), t in sorted(_cell_values(v["t1_windows"], "gpu_ms").items())},
                       "feasibility": v["feasibility"], "n_registered": len(v["feasibility"]),
                       "n_infeasible": sum(1 for x in v["feasibility"] if x["label"] == "INFEASIBLE_BY_RULE"),
                       "labels": sorted({l for w in v["t1_windows"] for l in w.get("labels", [])})}
        pts = _fit_points(v["t1_windows"])
        if len(pts) >= 4 and ident["cap"] == S.CAP_MAIN:
            fit = F.fit_t1(pts, ident["mode"])
            fits[key] = {"chosen": _strip(fit["chosen"]), "F1": _strip(fit["F1"]), "label": fit["label"]}
    A["tables"]["t1"] = t1_tab
    A["fits"] = {"t1": fits}
    # ---- gamma_j (J-A, per round)
    gam = {}
    for rkey in sorted({k.split("|")[0] + "|" + k.split("|")[1] for k in fits if k.startswith("J-A")}):
        anchor = next((fits[k]["F1"]["coef"] for k in fits if k.startswith(rkey) and k.split("|")[3] == "108u"), None)
        by_d = {t1_tab[k]["D_real"]: tuple(fits[k]["F1"]["coef"]) for k in fits if k.startswith(rkey)
                and k.split("|")[3].startswith("D") and t1_tab[k]["D_real"]}
        if anchor and by_d:
            gam[rkey] = F.gamma_fit(by_d, anchor)
    A["fits"]["gamma_j"] = gam
    # ---- cap invariance (J-A1)
    capinv = next((v for v in usable.values() if v["identity"]["cap"] == S.CAP_SMALL and v["kind"] == "T1T2"
                   and v.get("t1_windows")), None)
    if capinv:
        main44 = next((v for v in usable.values() if v["job"] == capinv["job"] and v["round"] == capinv["round"]
                       and v["identity"]["mode"] == "D44" and v["identity"]["cap"] == S.CAP_MAIN and v["kind"] == "T1T2"), None)
        if main44:
            A["labels"]["cap_invariance"] = F.cap_invariance(_cell_values(capinv["t1_windows"]),
                                                             {k: x for k, x in _cell_values(main44["t1_windows"]).items()
                                                              if k in S.CAP_INV_CELLS})
    # ---- T2 (P = 108 boots only, decision 28 A-2) / T2' (P axis)
    t2tab, t2fit, qs = {}, {}, {}
    for v in usable.values():
        if v["kind"] == "OFF" or not v.get("t2"):
            continue
        ident = v["identity"]
        if t2_axis(ident["mode"]) is None:
            v["problems"].append("T2_NOT_AT_P108: T2 data in a sticky-D boot ignored (decision 28 A-2)")
            continue
        key = f"{v['job']}|r{v['round']}|{ident['tag']}|{ident['mode']}"
        pts = [(x["L"], x["ttft_ms"]) for x in v["t2"] if x["status"] == "VALID"]
        t2tab[key] = {"points": {str(L): t for L, t in pts}, "P": S.TOTAL_SM, "P_evidence": "vanilla/unpartitioned"}
        ft = F.fit_t2(pts) if len(pts) >= 3 else None
        qi = F.q_index(pts, ft)
        slope, slab = F.j1_slope_label(pts)
        t2fit[key] = {"coef": ft["coef"] if ft else None, "loo_internal": ft["loo_internal"] if ft else None,
                      "accepted": bool(ft and ft["accepted"]),
                      "label": None if (ft and ft["accepted"]) else "FORM_INADEQUATE_T2_P108",
                      "q": qi, "j1_slope": slope, "j1_label": slab}
        qs.setdefault(f"{ident['tag']}|{ident['mode']}", []).append(qi.get("q"))
    A["tables"]["t2"] = t2tab
    A["fits"]["t2"] = t2fit
    A["labels"]["q"] = {k: F.q_label(v) for k, v in qs.items()}
    t2p_tab, t2p_fit = {}, {}
    for v in usable.values():
        if v["kind"] == "OFF" or not v.get("t2p"):
            continue
        ident = v["identity"]
        key = f"{v['job']}|r{v['round']}|{ident['tag']}|{ident['mode']}"
        conf = {str(x["L"]): (x.get("coexist_engine") or {}).get("label") for x in v["t2p"]}
        pts = [(x["L"], x["ttft_ms"]) for x in v["t2p"] if x["status"] == "VALID"
               and (x.get("coexist_engine") or {}).get("confirmed")]
        t2p_tab[key] = {"points": {str(x["L"]): x.get("ttft_ms") for x in v["t2p"] if x["status"] == "VALID"},
                        "P": v.get("t2p_P"), "coexist": conf, "points_confirmed": {str(L): t for L, t in pts},
                        "estimand": "single prefill TTFT with one bs=1 decode request co-resident (decision 28 A-2)"}
        ft = F.fit_t2(pts) if len(pts) >= 3 else None
        t2p_fit[key] = {"coef": ft["coef"] if ft else None, "accepted": bool(ft and ft["accepted"]),
                        "label": None if (ft and ft["accepted"]) else f"FORM_INADEQUATE_T2P_P{v.get('t2p_P')}",
                        "q": F.q_index(pts, ft)}
    A["tables"]["t2prime"] = t2p_tab
    A["fits"]["t2prime"] = t2p_fit
    A["fits"]["theta_prime"] = theta_prime(t2p_tab)
    # ---- T4
    t4 = {}
    pools = {}
    for v in usable.values():
        for x in v.get("t4", []):
            t4[f"{v['boot_id']}|L{x['L']}"] = x
        if v["kind"] == "T4" and v.get("mem"):
            pools[v["boot_id"]] = v["mem"]["pool_bytes"]
    A["tables"]["t4"] = t4
    if pools:
        A["labels"]["c_mem_drift"] = F.c_mem_drift(pools)
    # ---- J-H1
    jh = {k: v for k, v in usable.items() if v["job"] == "J-H1"}
    if jh:
        A["jh1"] = analyze_jh1(jh)
    # ---- OFF diagnostic (descriptive)
    A["instr_effect"] = instr_effects(usable)
    # ---- priors (comparison columns)
    A["priors"] = {f"a{n}": F.priors(n) for n in S.VARIANTS}
    dump(A, run_dir / "analysis.json")
    return A


def _status_counts(ws):
    out = {}
    for w in ws:
        out[w.get("status")] = out.get(w.get("status"), 0) + 1
    return out


def _strip(fit):
    if fit is None:
        return None
    return {k: v for k, v in fit.items() if k != "pred"}


def theta_prime(t2p_tab):
    """Descriptive (§1-2 rev3.4): per (job, round) over the sticky-D boots, through-origin LS of ln(T2'(L, P) /
    T2'(L, P_max)) on ln(P_max / P), pooled over the confirmed L.  No label reads it."""
    out = {}
    groups = {}
    for k, v in t2p_tab.items():
        job, rnd, tag, mode = k.split("|")
        if mode.startswith("D") and v.get("P"):
            groups.setdefault(f"{job}|{rnd}|{tag}", []).append(v)
    for g, vs in groups.items():
        top = max(vs, key=lambda v: v["P"])
        xs, ys = [], []
        for v in vs:
            if v is top:
                continue
            for L, t in v["points_confirmed"].items():
                t0 = top["points_confirmed"].get(L)
                if t0 and t:
                    xs.append(math.log(top["P"] / v["P"]))
                    ys.append(math.log(t / t0))
        den = sum(x * x for x in xs)
        out[g] = {"theta_prime": (sum(x * y for x, y in zip(xs, ys)) / den) if den > 0 else None, "P_ref": top["P"],
                  "n_points": len(xs)}
    return out


def analyze_jh1(jh):
    by_tag = {v["identity"]["tag"]: v for v in jh.values() if v["kind"] != "OFF"}
    out = {"boots": sorted(by_tag)}
    d4, r4 = by_tag.get("dummy_a4"), by_tag.get("real_a4")
    wi = {}
    if d4 and r4:
        common = set(S.JH_COMMON_CELLS)
        wi["T1"] = F.w_indep({k: x for k, x in _cell_values(d4["t1_windows"]).items() if k in common},
                             {k: x for k, x in _cell_values(r4["t1_windows"]).items() if k in common})
        wi["T2"] = F.w_indep({x["L"]: x["ttft_ms"] for x in d4["t2"] if x["status"] == "VALID"},
                             {x["L"]: x["ttft_ms"] for x in r4["t2"] if x["status"] == "VALID"})
    out["w_indep"] = wi
    p1, p2, cs, qn = {}, {}, {}, {}
    for n in S.VARIANTS:
        v = by_tag.get(f"dummy_a{n}")
        if not v:
            continue
        mt = v["mem"]
        p1[f"a{n}"] = F.compo_mem(n, mt["s"] if mt else None, mt["k"] if mt else None, mt["slots"] if mt else None)
        p2[f"a{n}"] = F.cmem_variant(n, v.get("t4", []))
        pts = _fit_points(v["t1_windows"], set(S.JH_COMMON_CELLS))
        if len(pts) == len(S.JH_COMMON_CELLS):
            ft = F.fit_form(pts, "F1")
            cs[n] = ft["coef"][2] if ft else None
        pts = [(x["L"], x["ttft_ms"]) for x in v["t2"] if x["status"] == "VALID"]
        ft2 = F.fit_t2(pts) if len(pts) >= 3 else None
        qn[n] = F.q_index(pts, ft2).get("q")
    out["P1_compo_mem"] = p1
    out["P2_cmem"] = p2
    out["P3_ctx_slope"] = F.ctx_slope_prop({n: c for n, c in cs.items() if c}, (wi.get("T1") or {}).get("label"))
    out["P4_q"] = F.q_monotone(qn, (wi.get("T2") or {}).get("label"))
    out["c_by_variant"] = cs
    out["q_by_variant"] = qn
    return out


def instr_effects(usable):
    out = {}
    for v in usable.values():
        if v["kind"] != "OFF":
            continue
        ident = v["identity"]
        on = next((w for w in usable.values() if w["kind"] == "T1T2" and w["job"] == v["job"] and w["round"] == v["round"]
                   and w["identity"]["tag"] == ident["tag"] and w["identity"]["mode"] == ident["mode"]
                   and w["identity"]["cap"] == ident["cap"]), None)
        if on is None:
            continue
        off_t1 = _cell_values(v["t1_windows"], "client_gap_ms")
        on_t1 = {k: x for k, x in _cell_values(on["t1_windows"], "client_gap_ms").items() if k in off_t1}
        off_t2 = {x["L"]: x["ttft_ms"] for x in v["t2"] + v.get("t2p", []) if x["status"] == "VALID"}
        on_t2 = {x["L"]: x["ttft_ms"] for x in (on["t2"] if v["t2"] else on.get("t2p", []))
                 if x["status"] == "VALID" and x["L"] in off_t2}
        out[v["boot_id"]] = {"T1": F.instrument_effect(on_t1, off_t1), "T2": F.instrument_effect(on_t2, off_t2),
                             "note": "descriptive; never applied to T1-T4 (decision 22)"}
    return out


# ============================================================================ CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--job", required=True)
    p.add_argument("--round", type=int, default=1)
    r = sub.add_parser("run")
    r.add_argument("--job", required=True)
    r.add_argument("--round", type=int, default=1)
    r.add_argument("--out", required=True)
    r.add_argument("--variant-src")
    r.add_argument("--variant-root")
    a = sub.add_parser("analyze")
    a.add_argument("dir")
    sub.add_parser("spec-sha")
    ns = ap.parse_args(argv)
    if ns.cmd == "spec-sha":
        print(S.spec_sha256())
        return 0
    if ns.cmd == "plan":
        plan = make_plan(ns.job, ns.round)
        for b in plan["boots"]:
            print(json.dumps({"boot_id": b["boot_id"], "kind": b["kind"], "mode": b["mode"], "cap": b["cap"],
                              "n_t1": len(b["t1"]), "t2": b["t2"], "t2p": b["t2p"], "t4": b["t4"]}))
        print(json.dumps({"problems": plan_problems(plan), "deferred": plan["deferred"]}))
        return 0
    if ns.cmd == "run":
        if os.environ.get("PDMUX_COSTTERMS_SEALED") != "1" and not _test_mode():
            print("ABORT_UNSEALED: the cost-term harness is not sealed (PREREG §12 item 13)", file=sys.stderr)
            return 16
        run_job(ns.job, ns.out, ns.round, ns.variant_src, ns.variant_root)
        return 0
    if ns.cmd == "analyze":
        analyze(ns.dir)
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
