#!/usr/bin/env python3
"""Job driver for M2g and M3 (PREREG_MULTIGPU rev10 §3, §4-1, §4-2, §6-2, §15-§17; rev12 §19).  Called by m2g_job.sh / m3_job.sh.

  mg_drive.py m3  --job J2 --out DIR [--dry-run] [--hard-cap-s S] [--t0-epoch E] [--pass-table T --pass-sha256 H --committed-sha-file F]
                                     [--j2-pass-table T --j2-pass-sha256 H --j2-committed-sha-file F]
  mg_drive.py m2g --job J2 --out DIR [--dry-run] [--hard-cap-s S] [--t0-epoch E]
  mg_drive.py assemble-pass-table --job J2 --out DIR  (M2g outputs -> M2G_PASS_TABLE_J2.json)
ONE Job = ONE N (J1 N=1 on the x2 spec GPU 0 only, J2, J4, J8).  The cross-Job join is mg_label.py --jobs.

Responsibilities: stage order (§4-1), arm order, joint extension (§4-2), x_lo from the fused HI median (MGP-9),
admission against the wall-clock hard cap, TRUNCATED bookkeeping, label + estimand output (mg_label).
rev12: every M3 boot spec carries mamba_ssm_dtype / x_tokens_path / x_tokens_sha256 (resolved once per Job by m3_job.sh); a boot that
exits 7/8/9 (ABORT_CONTEXT_TOO_SHORT / ABORT_DATASET / ABORT_CONFIG_DRIFT, CPU pre-checks) aborts the Job (fail-closed, ABORT file);
the U7 X correctness verdict is scored right after the opening N=1 boots (gates/XCHECK.json); finish() writes CLOCK_BY_CELL.json (U6).
Every boot runs through scripts/multigpu/mg_runner.sh (override PDMUX_MG_RUNNER for CPU tests).
JOB PATH vs REFERENCE PATH (F8): the Job runs `run_m2g` + `assemble_pass_table` below.  `m2g_correctness_gate.M2gRun` (the
unit-tested reference sequence with an injectable io) is NOT executed by the Job -- its order/liveness/exposure rules are
re-implemented here and pinned by tests/test_mg_harness.py::M2gJobPath (fake runner + fake torchrun).  G3 runs BEFORE any
serving boot (M2gRun order; a hang-heavy G3 must not be starved by the 8 h cap).
Hard cap (§6-2): the deadline is passed to the runner, which stops starting steps at the deadline and kills the
running cell (TRUNCATED); a boot is only ADMITTED when its estimated need fits in the remaining time
(implementation choice, listed in the hand-off: it avoids spending the last hour on a boot that cannot finish).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import inspect
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(os.path.abspath(__file__)).parent
ENGINE = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ENGINE / "scripts" / "multigpu"))
import mg_const as C  # noqa: E402
import mg_label as ML  # noqa: E402
import mg_plan as P  # noqa: E402

RUNNER = os.environ.get("PDMUX_MG_RUNNER") or str(ENGINE / "scripts" / "multigpu" / "mg_runner.sh")


class Ctx:
    def __init__(self, out, dry, deadline, t0, kind, job="J2"):
        self.out, self.dry, self.deadline, self.t0, self.kind = Path(out), dry, deadline, t0, kind
        self.job = job
        self.log = []
        self.not_attempted = []
        self.notes = []
        self.g3_python = None                      # tests: interpreter handed to `run-g3 --python` (default sys.executable)
        self.g5a_argv = None                       # tests: replaces the G5(a) CPU unit test command
        self.nsys_path = "auto"                    # tests: a path / None replaces `command -v nsys`
        self.m3_spec_extra = {}                    # rev12: mamba_ssm_dtype, x_tokens_path, x_tokens_sha256 (M3 boots)
        self.clk_csv = None
        (self.out / "specs").mkdir(parents=True, exist_ok=True)
        (self.out / "gates").mkdir(parents=True, exist_ok=True)

    def nsys_available(self):
        return bool(shutil.which("nsys") if self.nsys_path == "auto" else self.nsys_path)

    def remaining(self):
        return self.deadline - time.time()

    def event(self, **kw):
        kw["t"] = time.time()
        self.log.append(kw)
        with open(self.out / "DRIVER_LOG.jsonl", "a") as f:
            f.write(json.dumps(kw, sort_keys=True, default=str) + "\n")


FATAL_BOOT_RC = {7: "ABORT_CONTEXT_TOO_SHORT", 8: "ABORT_DATASET", 9: "ABORT_CONFIG_DRIFT"}


class JobAbort(Exception):
    pass


def run_boot(ctx, spec, expected_counts=None):
    """Admission + run.  Returns the runner rc, or None when not attempted."""
    need = P.boot_need_s(spec, 1.0) if not spec.get("expect_refusal") else 150.0
    if not ctx.dry and ctx.remaining() < need:
        ctx.not_attempted.append({"boot_id": spec["boot_id"], "config": spec.get("config"), "arm": spec.get("arm"), "block": spec.get("block", "main"),
                                  "need_s": need, "remaining_s": ctx.remaining(), "reason": "NOT_ATTEMPTED_BUDGET"})
        ctx.event(boot=spec["boot_id"], status="NOT_ATTEMPTED_BUDGET", need_s=need, remaining_s=ctx.remaining())
        return None
    spec = dict(spec, out_root=str(ctx.out), deadline_epoch=ctx.deadline)
    if expected_counts:
        spec["expected_proc_counts"] = expected_counts
    sp = ctx.out / "specs" / f"{spec['boot_id']}.json"
    sp.write_text(json.dumps(spec, indent=1, sort_keys=True))
    cmd = ["bash", RUNNER, str(sp)] + (["--dry-run"] if ctx.dry else [])
    t = time.time()
    p = subprocess.Popen(cmd, start_new_session=True, stdout=open(ctx.out / "specs" / f"{spec['boot_id']}.stdout", "wb"),
                         stderr=subprocess.STDOUT)
    try:
        rc = p.wait(timeout=max(60.0, ctx.remaining() + 240.0))
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        subprocess.run(["pkill", "-9", "-f", "sglang.launch_server|^sglang::"], check=False)   # the server has its own session
        rc = -9
    ctx.event(boot=spec["boot_id"], rc=rc, seconds=time.time() - t)
    if rc in FATAL_BOOT_RC and spec.get("kind") in ("boot1", "boot1ext", "boot2", "anchor", "gpudev"):
        bj = ctx.out / "boots" / spec["boot_id"] / "boot.json"
        why = (json.loads(bj.read_text()).get("abort") if bj.exists() else None) or FATAL_BOOT_RC[rc]
        raise JobAbort(f"{FATAL_BOOT_RC[rc]} boot {spec['boot_id']}: {why}")
    return rc


def run_boot_group(ctx, specs):
    """§15-3 GPU-deviation block: the 8 N=1 servers run CONCURRENTLY (one per GPU).  Admission against the largest need; ports are preassigned
    (spec["port"]) so the boots cannot race for one; the global `sglang::` sweep is disabled for a concurrent group (spec concurrent_group)."""
    need = max(P.boot_need_s(sp, 1.0) for sp in specs)
    if not ctx.dry and ctx.remaining() < need:
        for sp in specs:
            ctx.not_attempted.append({"boot_id": sp["boot_id"], "config": sp.get("config"), "arm": sp.get("arm"), "block": sp.get("block", "main"),
                                      "need_s": need, "remaining_s": ctx.remaining(), "reason": "NOT_ATTEMPTED_BUDGET"})
        ctx.event(group="gpu_dev", status="NOT_ATTEMPTED_BUDGET", need_s=need)
        return None
    procs, t = [], time.time()
    for sp in specs:
        sp = dict(sp, out_root=str(ctx.out), deadline_epoch=ctx.deadline, **ctx.m3_spec_extra)
        path = ctx.out / "specs" / f"{sp['boot_id']}.json"
        path.write_text(json.dumps(sp, indent=1, sort_keys=True))
        cmd = ["bash", RUNNER, str(path)] + (["--dry-run"] if ctx.dry else [])
        procs.append((sp["boot_id"], subprocess.Popen(cmd, start_new_session=True, stdout=open(ctx.out / "specs" / f"{sp['boot_id']}.stdout", "wb"),
                                                      stderr=subprocess.STDOUT)))
    rcs = {}
    for bid, p in procs:
        try:
            rcs[bid] = p.wait(timeout=max(60.0, ctx.remaining() + 240.0))
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            rcs[bid] = -9
    # R-1 (rev10 audit): the per-boot `^sglang::` sweep is disabled for a concurrent group (it would kill the siblings), so the sweep happens
    # ONCE here, after every member has exited -- unconditionally, not only after a -9 timeout (orphan scheduler children survive a clean rc).
    if not ctx.dry:
        subprocess.run(["pkill", "-9", "-f", "sglang.launch_server|^sglang::"], check=False)
    ctx.event(group="gpu_dev", rcs=rcs, seconds=time.time() - t, global_sweep=not ctx.dry)
    return rcs


# =========================================================================== M3
def cells_of(ctx, cfg=None, rnd=None):
    out = []
    for d in ML.find_cells(ctx.out):
        rec = ML.load_cell(d)
        m = rec["meta"]
        if (cfg is None or m["config"] == cfg) and (rnd is None or int(m.get("round", 1)) == rnd):
            out.append(rec)
    return out


def key_summaries(ctx, cfg, rnd=1, fired=()):
    groups = {}
    for c in cells_of(ctx, cfg, rnd):
        m = c["meta"]
        if m.get("block", "main") in ("main", "anchor") and m["rung"] in ("HI", "LO"):
            groups.setdefault((m["arm"], m["shape"]), []).append(c)
    return {k: ML.key_labels(v, ext_fired=(k[1] in fired)) for k, v in groups.items()}


def ext_complete(ctx, cfg, shapes, arms):
    """F10: did EVERY expected arm finish all four extension HI cells of every extended shape?  (A cell that was truncated or
    never started is not 'complete'; a judged-but-failed cell such as CLIENT_ERROR is complete -- its label says so.)"""
    cs = [c for c in cells_of(ctx, cfg, 1) if c["meta"].get("block", "main") == "main" and c["meta"]["rung"] == "HI"
          and int(c["meta"].get("ext", 0)) >= 1]
    for a in arms:
        for sh in shapes:
            got = {int(c["meta"]["seed"]): c for c in cs if c["meta"]["arm"] == a and c["meta"]["shape"] == sh}
            for sd in C.HI_SEEDS:
                c = got.get(sd)
                if c is None or c["status"] == "EMPTY" or c.get("truncated"):
                    return False, f"{a}/{sh}/seed{sd}:{'MISSING' if c is None else c['status']}"
    return True, None


def run_m3(ctx, table, j2_table=None, j2_note=None, box=None):
    """ONE Job (ctx.job).  Order (§16-3): [J8: GPU-deviation block, concurrent] -> N=1 anchor boots (J2/J4/J8) -> the Job's configs in the
    registered order (mg_plan.config_order; asserted equal to the PREREG result, else ABORT_CONFIG_ORDER).  TP configs additionally need J2's
    G0(ii) PASS (`j2_table`); a missing/invalid J2 table makes the Job's TP configs NOT_TESTED while DP proceeds (no ABORT)."""
    job = ctx.job
    try:
        P.assert_config_order_registered()
    except AssertionError as e:
        (ctx.out / "ABORT").write_text(f"ABORT_CONFIG_ORDER {e}\n")
        raise SystemExit(16)
    order = P.config_order(job)
    ctx.event(config_order=order, job=job, seed=C.ARM_ORDER_SEED, registered=C.CONFIG_ORDER_EXPECTED.get(C.JOB_N[job]))
    ex = P.excluded_set(table) if table else set()
    counts = (table or {}).get("nvml_expected_procs", {}) if (table or {}).get("nvml_procs_visible") else {}
    cfgs = P.runnable_configs(table, job) if table else list(order)
    tp_ok, tp_why = (True, None)
    if any(c.startswith("TP") for c in order):
        if job == C.G0II_JOB:
            tp_ok, tp_why = P.tp_allowed_by_j2(table) if table else (True, None)
        elif table is not None or not ctx.dry:
            tp_ok, tp_why = P.tp_allowed_by_j2(j2_table)
            if j2_note:
                tp_why = f"{tp_why}; {j2_note}" if tp_why else j2_note
        ctx.event(tp_axis_allowed=tp_ok, why=tp_why)

    def go(spec):
        # expected counts only apply to boots on the registered GPU subset of that config (not the gpu_dev boots)
        use = spec.get("gpu_ids") == C.GPU_SUBSETS.get(spec["config"]) and spec.get("block") in ("main", "anchor")
        return run_boot(ctx, dict(spec, **ctx.m3_spec_extra), counts.get(f"{spec['config']}:{spec['arm']}") if use else None)

    def score_xcheck(cfg):
        """U7: after the config's boot1 (fused_sym + agnostic carry the xcheck step) -> gates/XCHECK_<cfg>.json."""
        import mg_xcheck as XC
        r = XC.score_job(ctx.out / "gates", ctx.out / "boots", f"m3_{cfg}_fused_sym_boot1", f"m3_{cfg}_agnostic_boot1", config=cfg)
        ctx.event(xcheck=r.get("verdict"), config=cfg, reasons=r.get("reasons"))
        return r

    if job == "J8":
        run_boot_group(ctx, P.boots_gpu_dev())                          # §4-6, FIRST, concurrent
    for spec in P.boots_anchor(job, exclude=P.anchor_exclude(table)):
        go(spec)
    decisions = {"config_order": order, "tp_axis_allowed": tp_ok, "tp_axis_why": tp_why}
    if box is not None:
        box["decisions"] = decisions                                 # main(): finish() runs in a finally and still sees the decisions made so far
    for cfg in order:
        if cfg not in cfgs:
            ctx.event(config=cfg, status="EXCLUDED_BY_M2G_PASS_TABLE")
            continue
        if cfg.startswith("TP") and not tp_ok:
            ctx.event(config=cfg, status="TP_NOT_TESTED", why=tp_why)
            ctx.not_attempted.append({"config": cfg, "reason": "TP_NOT_TESTED", "detail": tp_why})
            continue
        for spec in P.boots_boot1(cfg, exclude=ex):
            go(spec)
        score_xcheck(cfg)                                             # U7 per config (N1 in J1; TP_N / DP_N on the N-GPU server)
        summ = key_summaries(ctx, cfg)
        ext_shapes = sorted({sh for (arm, sh), lab in summ.items() if lab["ext_trigger"]})
        decisions[cfg] = {"ext_shapes": ext_shapes}
        ext_ok = True
        if ext_shapes:                                                # joint extension of all four arms (MGP-9)
            for spec in P.boots_ext(cfg, tuple(ext_shapes), exclude=ex):
                go(spec)
            if not ctx.dry:
                ext_ok, why = ext_complete(ctx, cfg, ext_shapes, [a for a in C.ARMS if f"{cfg}:{a}" not in ex])
                decisions[cfg].update(ext_complete=ext_ok, ext_incomplete_reason=why)
            summ = key_summaries(ctx, cfg, fired=ext_shapes)
        x_lo, skipped = {}, []
        for sh in C.SHAPES:
            lab = summ.get(("fused", sh))
            lam = lab["lambda_star"] if lab else None                 # median of the 4 fused HI achieved
            if lam is None and ctx.dry:
                lam = P.prior(P.config_spec(cfg)["N"], sh)           # dry-run only: assumed capacity = prior
            if lam:
                x_lo[sh] = P.lo_rung(lam)
            else:
                skipped.append(sh)
        decisions[cfg].update(x_lo=x_lo, lo_skipped_fused_hi_undefined=skipped)
        ctx.event(config=cfg, decision=decisions[cfg])
        ext = {sh: (1 if sh in ext_shapes else 0) for sh in C.SHAPES}
        if x_lo and not ext_ok:                                       # F10: an incomplete extension forbids entering boot2 (LO)
            for spec in P.boots_boot2(cfg, x_lo, ext, exclude=ex):
                ctx.not_attempted.append({"boot_id": spec["boot_id"], "config": spec.get("config"), "arm": spec.get("arm"),
                                          "block": spec.get("block", "main"), "reason": "NOT_ATTEMPTED_EXT_INCOMPLETE",
                                          "detail": decisions[cfg].get("ext_incomplete_reason")})
                ctx.event(boot=spec["boot_id"], status="NOT_ATTEMPTED_EXT_INCOMPLETE")
        elif x_lo:
            for spec in P.boots_boot2(cfg, x_lo, ext, exclude=ex):
                go(spec)
    return decisions


def finish(ctx, table, decisions=None, exception=False):
    cells = cells_of(ctx)
    excl = P.excluded_set(table) if table else set()
    flags = ML.build_config_flags(ctx.out, cells, excl, ML.load_reference(), ctx.job)
    doc = ML.aggregate(cells, (table or {}).get("cells"), flags, (table or {}).get("configs"), job=ctx.job,
                       xcheck=ML.load_xchecks(ctx.out))
    doc["n_cells"] = len(cells)
    doc.update(ML.job_dir_rev12_facts(ctx.out, ctx.job))
    (ctx.out / "CLOCK_BY_CELL.json").write_text(json.dumps(ML.clock_by_cell(ctx.clk_csv, cells), indent=1, sort_keys=True) + "\n")
    doc["covariates"] = ML.job_covariates(ctx.out)                      # §16-3: this Job's cpu.max (per-cell throttle ratios are in doc["keys"])
    doc["driver"] = {"decisions": decisions, "not_attempted": ctx.not_attempted, "partial_boots": ML.partial_boots_of(ctx.out),
                     "truncated_cells": [c["cell_id"] for c in cells if c.get("truncated")],
                     "hard_cap_deadline_epoch": ctx.deadline, "t0_epoch": ctx.t0}
    if exception:       # rev11 D-D: finish() ran from finally after the driver raised -> planned boots never reached are NOT in not_attempted; the join reconstructs them
        doc["driver"]["exception"] = True
    (ctx.out / "LABELS.json").write_text(json.dumps(doc, indent=1, sort_keys=True, default=float) + "\n")
    (ctx.out / "LABELS.txt").write_text(ML.render_text(doc))
    status = {"kind": ctx.kind, "job": ctx.job, "finished_epoch": time.time(), "elapsed_s": time.time() - ctx.t0,
              "not_attempted": len(ctx.not_attempted), "n_cells": len(cells),
              "hard_cap_s": ctx.deadline - ctx.t0, "dry_run": ctx.dry}
    (ctx.out / "DRIVER_STATUS.json").write_text(json.dumps(status, indent=1) + "\n")
    return status


# =========================================================================== M2g
G3_COMBOS = (("plain", ["--streams", "plain"]), ("g16_92", ["--streams", "green", "--split", "16", "92"]),
             ("g44_64", ["--streams", "green", "--split", "44", "64"]))


def job_tps(job):
    """TP sizes G3 / G3b / G2 cover in this Job: only its own N (J1: none)."""
    return (C.JOB_N[job],) if C.JOB_N[job] >= 2 else ()


def job_multi_cfgs(job):
    """TP_N / DP_N of the Job (no N1)."""
    return tuple(c for c in P.m2g_configs(job) if c != "N1")


def run_g3(ctx):
    """G3 (§3), F1.  Runs BEFORE the serving boots.  One implementation: the gate module's `run-g3` (preflight that
    `python -m torch.distributed.run` imports, per-combination process-group timeout, launcher.json / stdout / stderr per
    combination, a directory with no rank file and no HANG file = NOT_TESTED(HARNESS_ERROR), unattempted combos absent).  This only
    calls it per TP, keeps each call's output under gates/g3_logs/, and survives a missing/odd interpreter or a timeout of the call
    itself (the TP then stays NOT_TESTED).  The Job preflight (mg_job_common.sh) already aborts a real run without a launcher."""
    gates = ctx.out / "gates"
    logs = gates / "g3_logs"
    logs.mkdir(parents=True, exist_ok=True)
    gate = str(ENGINE / "scripts" / "multigpu" / "m2g_correctness_gate.py")
    python = ctx.g3_python or sys.executable
    summary = {}
    for n in job_tps(ctx.job):
        ent = {"rc": None, "exception": None}
        summary[f"TP{n}"] = ent
        if ctx.remaining() < 900:
            ctx.not_attempted.append({"gate": "G3", "tp": n, "reason": "NOT_ATTEMPTED_BUDGET"})
            ent["exception"] = "NOT_ATTEMPTED_BUDGET"
            continue
        argv = [sys.executable, gate, "run-g3", "--out", str(gates), "--tp", str(n), "--python", python,
                "--deadline-epoch", str(ctx.deadline)]
        ctx.event(g3=f"TP{n}", status="START")
        try:
            r = subprocess.run(argv, capture_output=True, text=True, check=False, timeout=max(60.0, ctx.remaining()))
            ent["rc"] = r.returncode
            (logs / f"run_g3_TP{n}.txt").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr)
        except (OSError, subprocess.TimeoutExpired) as e:
            ent["exception"] = repr(e)
            (logs / f"run_g3_TP{n}.txt").write_text(repr(e) + "\n")
        ctx.event(g3=f"TP{n}", status="DONE", rc=ent["rc"], exception=ent["exception"])
    (gates / "g3_launch.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    return summary


def g1_exposure(ctx):
    """F3: the exposure ratio of the G1 window -- fraction of decode iterations executed inside a split (division) group while
    the G1 prompts were being served -- per PD-mux arm boot, from the telemetry restricted to the G1 step window (mg_boot writes
    cmd_g1_<arm>.window.json on CLOCK_MONOTONIC) via mg_realized_mix.cell_e_time.  Written to gates/g1_exposure_<cfg>.json.
    Not computable (no window / telemetry / MIX_FAILED) -> exposure null with the status; it is reported, never invented."""
    import mg_realized_mix as RM
    G = _gate()
    out = {}
    for cfg in job_multi_cfgs(ctx.job):
        mode, n = G.cfg_mode_n(cfg)
        per = {}
        for arm in C.PDMUX_ARMS:
            bd = ctx.out / "boots" / f"m2g_{cfg}_{arm}"
            w = _json(bd / f"cmd_g1_{arm}.window.json")
            if w is None:
                per[arm] = {"exposure_steps": None, "e_time": None, "status": "NO_G1_WINDOW"}
                continue
            try:
                et = RM.cell_e_time(arm, mode, n, str(bd / "tel.jsonl"), str(bd / "green.json"), (w["t0_mono"], w["t1_mono"]))
                per[arm] = {"exposure_steps": et.get("exposure_steps"), "e_time": et.get("e_time"), "status": et.get("status"),
                            "reasons": (et.get("reasons") or [])[:5]}
            except Exception as e:  # noqa: BLE001
                per[arm] = {"exposure_steps": None, "e_time": None, "status": "MIX_FAILED", "reasons": [repr(e)]}
        out[cfg] = per
        (ctx.out / "gates" / f"g1_exposure_{cfg}.json").write_text(json.dumps(per, indent=1, sort_keys=True) + "\n")
    return out


def score_g1_all(ctx):
    """score-g1 per config, in-process through the gate's `cli_score_g1` (single implementation of the scorer).  It reads the survival
    records itself (alive_after_g1_<arm>.json written right after the G1 step of the fused boot and of the arm's boot -- 2026-10-13
    repair: NOT the boot-end alive_after.json, so a death in a later step (G2) is not charged to G1; a missing record -> NOT_TESTED,
    a dead server -> FAIL)
    and, with exposure_from_boots, computes the exposure of the G1 window recorded by serve-g1 (`g1_window_perf_counter_s`).  If the
    installed gate has no such option the exposure computed by `g1_exposure` (cmd window) is passed explicitly instead."""
    G = _gate()
    gates = ctx.out / "gates"
    fn = G.cli_score_g1
    params = inspect.signature(fn).parameters
    for cfg in P.m2g_configs(ctx.job):
        kw = {}
        if "exposure_from_boots" in params:
            kw.update(boots=ctx.out / "boots", exposure_from_boots=True)
        elif "exposure" in params:
            expo = _json(gates / f"g1_exposure_{cfg}.json") or {}
            kw["exposure"] = {a: (v or {}).get("exposure_steps") for a, v in expo.items()}
        try:
            fn(gates, cfg, **kw)
        except Exception as e:  # noqa: BLE001 - recorded; the cell stays NOT_TESTED
            (gates / f"g1_score_error_{cfg}.txt").write_text(repr(e) + "\n")


def g5a_gpu_descriptive(ctx):
    """F4 / MGP-5: the GPU half of G5(a), DESCRIPTIVE: N=1 agnostic under the AUTO-grid config, decode-busy stream-index
    histogram beside the manual N=1 agnostic boot's.  No verdict, no exclusion."""
    import mg_realized_mix as RM
    out = {"gate": "G5a_gpu_descriptive", "verdict": "DESCRIPTIVE"}
    bj = _json(ctx.out / "boots" / "m2g_g5a_auto_N1" / "boot.json")
    if ctx.job != "J1":
        out["status"] = "NOT_IN_THIS_JOB"                          # §15-2: the GPU descriptive half lives in J1 only
    elif bj is None:
        out["status"] = "NOT_RUN"
    elif not bj.get("t_health_epoch"):
        out["status"] = "BOOT_FAILED"
    else:
        out["status"] = "RECORDED"

        def hist(bid):
            bd = ctx.out / "boots" / bid
            try:
                et = RM.cell_e_time("agnostic", "n1", 1, str(bd / "tel.jsonl"), str(bd / "green.json"))
                return {"status": et.get("status"), "hist": [(r.get("estimate") or {}).get("busy_index_hist") for r in et.get("per_rank", [])]}
            except Exception as e:  # noqa: BLE001
                return {"status": "ERROR", "error": repr(e)}
        out["auto_grid"] = hist("m2g_g5a_auto_N1")
        out["manual_grid"] = hist("m2g_N1_agnostic")
    (ctx.out / "gates" / "g5a_gpu.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    return out


def g3_excluded_tps(ctx):   # noqa: D401
    """R1: TP{n} whose gates/g3_TP{n}.json verdict is not PASS (missing / unreadable file counts as not PASS) -- the serving
    boots of that TP are not started (a comm hang there would burn the cap and, with no G3 evidence, cannot be admitted)."""
    out = set()
    for n in job_tps(ctx.job):
        d = _json(ctx.out / "gates" / f"g3_TP{n}.json")
        if not isinstance(d, dict) or d.get("verdict") != "PASS":
            out.add(n)
    return out


def _tp_of_boot(boot_id):
    m = re.match(r"^m2g_(?:g4_)?TP(\d+)(?:_|$)", boot_id)
    return int(m.group(1)) if m else None


def g3_excluded_boot(boot_id, excluded_tp):
    tp = _tp_of_boot(boot_id)
    return tp is not None and tp in excluded_tp


def preboot_nvml_check(ctx):
    """Once, after G3 and before the first serving boot: every GPU empty (memory.used < NVML_EMPTY_GPU_MIB, G5(c)(가)).  A hung G3
    rank can still hold memory: reap stress stragglers and re-check for up to 60 s.  Recorded in gates/nvml_preboot.json; the
    per-boot contamination check (mg_boot) still decides each boot -- this only makes the G3 residue visible/cleared."""
    G = _gate()
    rec = {"checked": True, "attempts": 0}
    try:
        for attempt in range(13):
            snap = G.nvml_snapshot(list(range(C.JOB_EXPECTED_GPUS[ctx.job])))
            rec["attempts"] = attempt + 1
            rec["snapshot"] = {str(g): v for g, v in snap.items()}
            bad = [g for g, v in snap.items() if not v["memory_used_mib"] < C.NVML_EMPTY_GPU_MIB]
            rec["not_empty"] = bad
            if not bad:
                break
            if attempt == 0:
                subprocess.run(["pkill", "-9", "-f", "m2g_comm_reverse_stress.py"], check=False)
            time.sleep(5)
    except Exception as e:  # NVML unavailable etc.: the per-boot check reports CONTAM_UNVERIFIED
        rec.update(checked=False, error=repr(e))
    (ctx.out / "gates" / "nvml_preboot.json").write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    ctx.event(nvml_preboot=rec.get("not_empty"), checked=rec["checked"])
    return rec


def run_m2g(ctx):
    dummy = ctx.out / "DUMMY_PROFILE.json"
    dummy.write_text("{}\n")
    # G5(a) CPU half: deterministic test g5a_selector_equiv.py (rc 0 PASS / 1 FAIL / 2 engine tree missing) -- recorded
    g5a = ctx.g5a_argv or [sys.executable, str(ENGINE / "scripts/multigpu/g5a_selector_equiv.py")]
    r = subprocess.run(g5a, capture_output=True, text=True, check=False) if not ctx.dry else None
    verdict = "SKIPPED_DRYRUN" if r is None else {0: "PASS", 1: "FAIL", 2: "TREE_MISSING"}.get(r.returncode, f"RC_{r.returncode}")
    (ctx.out / "gates" / "g5a.json").write_text(json.dumps(
        {"gate": "G5a", "verdict": verdict, "tail": None if r is None else (r.stdout + r.stderr)[-600:]}) + "\n")
    gate = str(ENGINE / "scripts" / "multigpu" / "m2g_correctness_gate.py")
    if not ctx.dry:
        run_g3(ctx)                                                  # F1: BEFORE any serving boot (M2gRun order)
        for n in job_tps(ctx.job):
            subprocess.run([sys.executable, gate, "score-g3", "--out", str(ctx.out / "gates"), "--tp", str(n)],
                           capture_output=True, text=True, check=False)
    excluded_tp = g3_excluded_tps(ctx) if not ctx.dry else set()
    if not ctx.dry:
        preboot_nvml_check(ctx)
    ctx.event(m2g_job=ctx.job, configs=list(P.m2g_configs(ctx.job)))
    for spec in P.m2g_boots(ctx.job, str(dummy)):
        if g3_excluded_boot(spec["boot_id"], excluded_tp):
            tp = _tp_of_boot(spec["boot_id"])
            ctx.not_attempted.append({"boot_id": spec["boot_id"], "tp": tp, "reason": "NOT_ATTEMPTED_G3_EXCLUDED"})
            ctx.event(boot=spec["boot_id"], status="NOT_ATTEMPTED_G3_EXCLUDED", tp=tp)
            continue
        if spec.get("nsys") and not ctx.dry and not ctx.nsys_available():
            # R-3 (rev10 audit): the image has no nsys -> the G4 nsys boot cannot produce its kernel-name evidence; record NOT_RUN, do not boot
            # (a started server would wrap `nsys launch` and die).  MGH-24': G4 'PASS' then means the main boot's cudagraph capture only.
            ctx.not_attempted.append({"boot_id": spec["boot_id"], "reason": "NOT_RUN_NSYS_ABSENT", "block": spec.get("block")})
            ctx.event(boot=spec["boot_id"], status="NOT_RUN_NSYS_ABSENT")
            continue
        run_boot(ctx, spec)
    if not ctx.dry:
        g1_exposure(ctx)                                             # F3
        score_g1_all(ctx)
        for cfg in P.m2g_configs(ctx.job):
            subprocess.run([sys.executable, gate, "score-g2", "--out", str(ctx.out / "gates"), "--config", cfg],
                           capture_output=True, text=True, check=False)
        g5a_gpu_descriptive(ctx)
    return assemble_pass_table(ctx.out, ctx.job)


def _json(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def _gate():
    import m2g_correctness_gate as G          # engine-porter: scorers + the §3-1 table builder (ONE implementation)
    return G


def assemble_pass_table(out, job):
    """Job artifacts -> engine-porter's `res` layout -> `m2g_correctness_gate.build_pass_table` (§3-1 mapping and every
    gate scorer live THERE, tested; this function only reads files).  Missing artifact == NOT_TESTED == excluded.
    Liveness (F2) is read by the gate scorers (G3b: boots/<id>/alive_after.json of that boot, after its last step; G1: the record
    right after the G1 step, boots/<id>/alive_after_g1_<arm>.json, of the fused boot and the arm's boot; missing -> NOT_TESTED,
    dead -> FAIL).  NVML unavailable at the pre-boot check
    (CONTAM_UNVERIFIED, F5) is NOT 'clean': the cell's G5(c) contamination check is NOT_TESTED (excluded) and it is listed."""
    G = _gate()
    out = Path(out)
    cfgs = P.m2g_configs(job)
    own_g0ii = (job == C.G0II_JOB)

    def txt(p):
        return Path(p).read_text(errors="replace") if Path(p).exists() else None

    def boot_dict(bid):
        bj = _json(out / "boots" / bid / "boot.json")
        if bj is None:
            return None
        return {"healthy": bool(bj.get("t_health_epoch")), "health_s": bj.get("boot_seconds"),
                "health_wait_cap_s": bj.get("health_wait_cap_s", 0), "exit_code": bj.get("server_exit_code"),
                "log_text": txt(out / "boots" / bid / "server.log") or ""}

    res = {k: {} for k in ("g0i", "g1", "g2", "g3", "g3b", "g4", "g5b", "g5c", "contam", "nvml_post", "meta")}
    unverified, exposure_tab = [], {}
    ref_ids = sorted(b.name for b in (out / "boots").glob("m2g_refuse_*")) if (out / "boots").exists() else []
    if own_g0ii:                                    # §15-2: G0(ii)/(iii) are J2's; other Jobs record NOT_RUN_IN_JOB and M3 reads J2's table
        res["g0ii"] = G.score_g0_ii([d for d in map(boot_dict, ref_ids) if d is not None])
        res["g0iii"] = G.score_g0_refusal(boot_dict("m2g_autogrid_TP2"), G.DEVICE_ASSERT_TEXT)
    for cfg in cfgs:
        mode, n = G.cfg_mode_n(cfg)
        for arm in C.ARMS:
            bid, k = f"m2g_{cfg}_{arm}", G.cell_key(cfg, arm)
            bd = out / "boots" / bid
            res["g0i"][k] = G.score_g0_boot(boot_dict(bid))
            pw = _json(bd / "piecewise.json") or {}
            sa = {"disable_cuda_graph": pw.get("disable_cuda_graph") == "True",
                  "disable_piecewise_cuda_graph": pw.get("disable_piecewise_cuda_graph"),
                  "piecewise_cuda_graph_tokens": pw.get("piecewise_cuda_graph_tokens")}
            if res["g0i"][k]["verdict"] == G.PASS:      # a boot that never became healthy has no capture to judge: G4 absent == NOT_TESTED
                res["g4"][k] = G.score_g4(txt(bd / "server.log"), sa)
            res["meta"][k] = _json(bd / "cell_meta.json")
            pc = _json(bd / "proc_counts_post_boot.json")
            if pc is not None:
                res["nvml_post"][k] = {int(g): int(c or 0) for g, c in pc.items()}
        if cfg != "N1":
            g1 = _json(out / "gates" / f"g1_{cfg}.json") or {}
            expo = _json(out / "gates" / f"g1_exposure_{cfg}.json") or {}
            for arm in C.PDMUX_ARMS:
                v = (g1.get("verdicts") or {}).get(f"fused|{arm}")
                if v is None:
                    continue
                det = (g1.get("detail") or {}).get(f"fused|{arm}") or {}
                res["g1"][G.cell_key(cfg, arm)] = {"verdict": v}        # abnormal exit / unknown liveness were applied by the scorer
                e = det.get("exposure_split_steps")
                exposure_tab[G.cell_key(cfg, arm)] = e if e is not None else (expo.get(arm) or {}).get("exposure_steps")
        if mode == "tp":
            for name, sec in (("g2", f"g2_{cfg}.json"), ("g3", f"g3_TP{n}.json")):
                d = _json(out / "gates" / sec)
                if d is not None:
                    res[name][cfg] = d
            for arm in C.G3B_ARMS:
                if (out / "cells").exists() and list((out / "cells").glob(f"m2g_{cfg}_{arm}__A_G3B_s*")):
                    # requests + the survival record after the LAST step of that boot (a missing record -> NOT_TESTED)
                    res["g3b"][G.cell_key(cfg, arm)] = G.score_g3b_artifacts(out, cfg, arm, g_fn=ML.g_of, load_requests=ML.load_requests)
        if mode == "dp":
            arms = {}
            for arm in C.PDMUX_ARMS:
                if res["g0i"][G.cell_key(cfg, arm)]["verdict"] == G.PASS:
                    arms[arm] = G.score_g5b_arm(n, arm, str(out / "boots" / f"m2g_{cfg}_{arm}" / "tel.jsonl"))
            res["g5b"][cfg] = G.score_g5b(arms)
        res["g5c"][cfg] = G.score_g5c_memory({a: res["meta"].get(G.cell_key(cfg, a)) for a in C.ARMS})
    visible = G.procs_visible_from_table(res["nvml_post"]) if res["nvml_post"] else False
    for cfg in cfgs:
        for arm in C.ARMS:
            k = G.cell_key(cfg, arm)
            cp = _json(out / "boots" / f"m2g_{cfg}_{arm}" / "contam_pre.json")
            if cp is None:
                continue
            snap = (cp.get("snapshot") or {}).get("gpus") or {}
            ver = cp.get("verdict") or {}
            if (cp.get("snapshot") or {}).get("available") is False or ver.get("contam_unverified") or not snap:
                unverified.append(k)                  # F5: the check did not run -> not 'clean'; contam stays absent == NOT_TESTED
                continue
            pre = {int(g): {"memory_used_mib": v["used_mib"], "n_procs": v["n_compute_procs"] or 0} for g, v in snap.items()}
            res["contam"][k] = G.check_contamination(pre, procs_visible=visible)
            cx = _json(out / "boots" / f"m2g_{cfg}_{arm}" / "contam.json")
            if cx and cx.get("gpu_contaminated"):
                res["contam"][k] = {"gpu_contaminated": True, "problems": [str(cx)]}
    table = G.build_pass_table(res, cfg_list=cfgs, g0ii_in_job=own_g0ii, job=job)
    table["g5a_unit_test"] = (_json(out / "gates" / "g5a.json") or {}).get("verdict", "NOT_TESTED")
    table["g5a_gpu_descriptive"] = (_json(out / "gates" / "g5a_gpu.json") or {}).get("status", "NOT_RUN")
    table["contam_unverified"] = sorted(unverified)
    for k in unverified:
        if k in table["cells"]:
            table["cells"][k].setdefault("reasons", []).append("CONTAM_UNVERIFIED")
            table["cells"][k]["reason"] = "; ".join(table["cells"][k]["reasons"])
    table["g1_exposure_split_steps"] = exposure_tab
    name = C.m2g_pass_table_name(job)
    sha = G.write_pass_table(table, out / name)
    (out / (name + ".sha256")).write_text(sha + "\n")
    (out / "M2G_GATE_RESULTS.json").write_text(json.dumps({k: v for k, v in res.items() if k != "meta"}, indent=1,
                                                          sort_keys=True, default=str) + "\n")
    return table


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=("m3", "m2g", "assemble-pass-table"))
    ap.add_argument("--job", required=True, choices=C.JOBS)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--hard-cap-s", type=float, default=None)
    ap.add_argument("--t0-epoch", type=float, default=None)
    ap.add_argument("--pass-table", default="")
    ap.add_argument("--pass-sha256", default="")
    ap.add_argument("--committed-sha-file", default="", help="F17: committed results/multigpu/M2G_PASS_TABLE_J{N}.json.sha256")
    ap.add_argument("--j2-pass-table", default="", help="J2's table (G0(ii) result) for the TP axis of J4/J8")
    ap.add_argument("--j2-pass-sha256", default="")
    ap.add_argument("--j2-committed-sha-file", default="")
    ap.add_argument("--mamba-ssm-dtype", default="", help="rev12 §19-3: resolved once per Job by m3_job.sh (required for m3)")
    ap.add_argument("--x-tokens", default="", help="rev12 §19-2: the registered shape-X token file")
    ap.add_argument("--x-tokens-sha256", default=C.X_TOKENS_SHA256)
    ap.add_argument("--clk-csv", default="", help="U6: job_entry meta/clk.csv (descriptive clock table)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.cmd == "assemble-pass-table":
        t = assemble_pass_table(out, a.job)
        print(json.dumps({k: v for k, v in t.items() if k != "proc_counts"}, indent=1))
        return 0
    t0 = a.t0_epoch or time.time()
    cap = a.hard_cap_s or C.hard_cap(a.cmd, a.job)
    ctx = Ctx(out, a.dry_run, t0 + cap, t0, a.cmd, a.job)
    if a.cmd == "m2g":
        run_m2g(ctx)
        st = {"kind": "m2g", "job": a.job, "not_attempted": ctx.not_attempted}
        (out / "DRIVER_STATUS.json").write_text(json.dumps(st, indent=1) + "\n")
        return 0
    table = None
    if a.pass_table or not a.dry_run:
        table, abort = P.load_pass_table(a.pass_table, a.pass_sha256, a.committed_sha_file or None)
        if abort:
            print(abort, file=sys.stderr)
            (out / "ABORT").write_text(abort + "\n")
            return 3
    if a.cmd == "m3":                                                # after the pass-table abort (rev11 precedence kept)
        if not a.mamba_ssm_dtype:
            (out / "ABORT").write_text("ABORT_CONFIG_DRIFT mamba_ssm_dtype not resolved (rev12 §19-3)\n")
            return 3
        ctx.m3_spec_extra = {"mamba_ssm_dtype": a.mamba_ssm_dtype, "x_tokens_path": a.x_tokens, "x_tokens_sha256": a.x_tokens_sha256}
        ctx.clk_csv = a.clk_csv or None
    j2_table, j2_note = None, None
    if a.job == C.G0II_JOB:
        j2_table = table
    elif a.job in ("J4", "J8"):
        if a.j2_pass_table:
            j2_table, ab = P.load_pass_table(a.j2_pass_table, a.j2_pass_sha256, a.j2_committed_sha_file or None)
            j2_note = None if j2_table else f"J2 table rejected: {ab}"
        else:
            j2_note = "J2 table not given"
    box, dec = {}, None
    try:
        dec = run_m3(ctx, table, j2_table, j2_note, box=box)
    except JobAbort as e:                                            # rev12: fatal CPU pre-check of a boot -> the Job stops
        (out / "ABORT").write_text(str(e) + "\n")
        ctx.event(status="JOB_ABORT", why=str(e))
        dec = dict(box.get("decisions") or {}, job_abort=str(e))
    finally:
        # rev11 D-B: finish() is GUARANTEED (exception / SystemExit / KeyboardInterrupt in the driver): LABELS.json carries a `driver` record
        # (not_attempted / partial_boots) so the join never has to guess; a Job that dies harder (SIGKILL) is reconstructed by mg_label.reconstruct_driver.
        if dec is None:
            ctx.event(status="DRIVER_EXCEPTION_FINISH_IN_FINALLY")
        finish(ctx, table, dec or box.get("decisions"), exception=(dec is None))
    return 17 if (out / "ABORT").exists() else 0


if __name__ == "__main__":
    sys.exit(main())
