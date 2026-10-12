#!/usr/bin/env python3
"""Job driver for M2g and M3 (PREREG_MULTIGPU rev6 §3, §4-1, §4-2, §6-2).  Called by m2g_job.sh / m3_job.sh.

  mg_drive.py m3  --out DIR [--dry-run] [--hard-cap-s S] [--t0-epoch E]
  mg_drive.py m2g --out DIR [--dry-run] [--hard-cap-s S] [--t0-epoch E]
  mg_drive.py assemble-pass-table --out DIR           (M2g outputs -> M2G_PASS_TABLE.json)

Responsibilities: stage order (§4-1), arm order, joint extension (§4-2), x_lo from the fused HI median (MGP-9),
admission against the wall-clock hard cap, TRUNCATED bookkeeping, label + estimand output (mg_label).
Every boot runs through scripts/multigpu/mg_runner.sh (override PDMUX_MG_RUNNER for CPU tests).
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
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENGINE = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ENGINE / "scripts" / "multigpu"))
import mg_const as C  # noqa: E402
import mg_label as ML  # noqa: E402
import mg_plan as P  # noqa: E402

RUNNER = os.environ.get("PDMUX_MG_RUNNER") or str(ENGINE / "scripts" / "multigpu" / "mg_runner.sh")


class Ctx:
    def __init__(self, out, dry, deadline, t0, kind):
        self.out, self.dry, self.deadline, self.t0, self.kind = Path(out), dry, deadline, t0, kind
        self.log = []
        self.not_attempted = []
        self.notes = []
        (self.out / "specs").mkdir(parents=True, exist_ok=True)
        (self.out / "gates").mkdir(parents=True, exist_ok=True)

    def remaining(self):
        return self.deadline - time.time()

    def event(self, **kw):
        kw["t"] = time.time()
        self.log.append(kw)
        with open(self.out / "DRIVER_LOG.jsonl", "a") as f:
            f.write(json.dumps(kw, sort_keys=True, default=str) + "\n")


def run_boot(ctx, spec, expected_counts=None):
    """Admission + run.  Returns the runner rc, or None when not attempted."""
    need = P.boot_need_s(spec, 1.0) if not spec.get("expect_refusal") else 150.0
    if not ctx.dry and ctx.remaining() < need:
        ctx.not_attempted.append({"boot_id": spec["boot_id"], "need_s": need, "remaining_s": ctx.remaining(),
                                  "reason": "NOT_ATTEMPTED_BUDGET"})
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
    return rc


# =========================================================================== M3
def cells_of(ctx, cfg=None, rnd=None):
    out = []
    for d in ML.find_cells(ctx.out):
        rec = ML.load_cell(d)
        m = rec["meta"]
        if (cfg is None or m["config"] == cfg) and (rnd is None or int(m.get("round", 1)) == rnd):
            out.append(rec)
    return out


def key_summaries(ctx, cfg, rnd=1):
    groups = {}
    for c in cells_of(ctx, cfg, rnd):
        m = c["meta"]
        if m.get("block", "main") in ("main", "anchor") and m["rung"] in ("HI", "LO"):
            groups.setdefault((m["arm"], m["shape"]), []).append(c)
    return {k: ML.key_labels(v) for k, v in groups.items()}


def run_m3(ctx, table):
    ex = P.excluded_set(table) if table else set()
    counts = (table or {}).get("nvml_expected_procs", {}) if (table or {}).get("nvml_procs_visible") else {}
    cfgs = P.runnable_configs(table) if table else list(C.CONFIG_ORDER)

    def go(spec):
        # (나) expected counts only apply to boots on the registered GPU subset of that config (not the gpu_dev / NUMA boots)
        use = spec.get("gpu_ids") == C.GPU_SUBSETS.get(spec["config"]) and spec.get("block") in ("main", "anchor")
        return run_boot(ctx, spec, counts.get(f"{spec['config']}:{spec['arm']}") if use else None)

    for spec in P.boots_gpu_dev():                                    # §4-6, FIRST
        go(spec)
    decisions = {}
    for cfg in C.CONFIG_ORDER:                                        # §4-1 order
        if cfg not in cfgs:
            ctx.event(config=cfg, status="EXCLUDED_BY_M2G_PASS_TABLE")
            continue
        for spec in P.boots_boot1(cfg, exclude=ex):
            go(spec)
        summ = key_summaries(ctx, cfg)
        ext_shapes = sorted({sh for (arm, sh), lab in summ.items() if lab["ext_trigger"]})
        decisions[cfg] = {"ext_shapes": ext_shapes}
        if ext_shapes:                                                # joint extension of all four arms (MGP-9)
            for spec in P.boots_ext(cfg, tuple(ext_shapes), exclude=ex):
                go(spec)
            summ = key_summaries(ctx, cfg)
        x_lo, skipped = {}, []
        for sh in ("A", "B"):
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
        ext = {sh: (1 if sh in ext_shapes else 0) for sh in ("A", "B")}
        if x_lo:
            for spec in P.boots_boot2(cfg, x_lo, ext, exclude=ex):
                go(spec)
    for spec in P.boots_numa():
        go(spec)
    for spec in P.boots_nsys():
        go(spec)
    for spec in P.boots_anchor2():
        go(spec)
    return decisions


def finish(ctx, table, decisions=None):
    cells = cells_of(ctx)
    flags = ML.build_config_flags(ctx.out, cells)
    doc = ML.aggregate(cells, (table or {}).get("cells"), flags)
    doc["n_cells"] = len(cells)
    doc["driver"] = {"decisions": decisions, "not_attempted": ctx.not_attempted,
                     "truncated_cells": [c["cell_id"] for c in cells if c.get("truncated")],
                     "hard_cap_deadline_epoch": ctx.deadline, "t0_epoch": ctx.t0}
    (ctx.out / "LABELS.json").write_text(json.dumps(doc, indent=1, sort_keys=True, default=float) + "\n")
    (ctx.out / "LABELS.txt").write_text(ML.render_text(doc))
    status = {"kind": ctx.kind, "finished_epoch": time.time(), "elapsed_s": time.time() - ctx.t0,
              "not_attempted": len(ctx.not_attempted), "n_cells": len(cells),
              "hard_cap_s": ctx.deadline - ctx.t0, "dry_run": ctx.dry}
    (ctx.out / "DRIVER_STATUS.json").write_text(json.dumps(status, indent=1) + "\n")
    return status


# =========================================================================== M2g
def run_m2g(ctx):
    dummy = ctx.out / "DUMMY_PROFILE.json"
    dummy.write_text("{}\n")
    # G5(a): deterministic CPU test g5a_selector_equiv.py (rc 0 PASS / 1 FAIL / 2 engine tree missing) -- recorded
    r = subprocess.run([sys.executable, str(ENGINE / "scripts/multigpu/g5a_selector_equiv.py")],
                       capture_output=True, text=True, check=False) if not ctx.dry else None
    verdict = "SKIPPED_DRYRUN" if r is None else {0: "PASS", 1: "FAIL", 2: "TREE_MISSING"}.get(r.returncode, f"RC_{r.returncode}")
    (ctx.out / "gates" / "g5a.json").write_text(json.dumps(
        {"gate": "G5a", "verdict": verdict, "tail": None if r is None else (r.stdout + r.stderr)[-600:]}) + "\n")
    for spec in P.m2g_boots(str(dummy)):
        run_boot(ctx, spec)
    # G1/G2 scoring across arms (engine-porter hook, contract in the module docstring of m2g_job.sh)
    gate = str(ENGINE / "scripts" / "multigpu" / "m2g_correctness_gate.py")
    if not ctx.dry:
        for cfg in ("N1",) + P.M2G_CFGS:
            subprocess.run([sys.executable, gate, "score-g1", "--out", str(ctx.out / "gates"), "--config", cfg],
                           capture_output=True, text=True, check=False)
            subprocess.run([sys.executable, gate, "score-g2", "--out", str(ctx.out / "gates"), "--config", cfg],
                           capture_output=True, text=True, check=False)
        # G3: two comms in reversed order, 3 TP x (plain + green 16/92 + green 44/64)
        for n in (2, 4, 8):
            for tag, extra in (("plain", ["--streams", "plain"]), ("g16_92", ["--streams", "green", "--split", "16", "92"]),
                               ("g44_64", ["--streams", "green", "--split", "44", "64"])):
                od = ctx.out / "gates" / f"g3_TP{n}_{tag}"
                if ctx.remaining() < 900:
                    ctx.not_attempted.append({"gate": "G3", "tp": n, "tag": tag, "reason": "NOT_ATTEMPTED_BUDGET"})
                    continue
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=C.GPU_SUBSETS[f"TP{n}"])
                subprocess.run(["torchrun", "--nproc-per-node", str(n), str(ENGINE / "scripts/multigpu/m2g_comm_reverse_stress.py"),
                                *extra, "--out", str(od)], env=env, capture_output=True, text=True, check=False,
                               timeout=max(60.0, ctx.remaining()))
            subprocess.run([sys.executable, gate, "score-g3", "--out", str(ctx.out / "gates"), "--tp", str(n)],
                           capture_output=True, text=True, check=False)
    return assemble_pass_table(ctx.out)


def _json(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def _gate():
    import m2g_correctness_gate as G          # engine-porter: scorers + the §3-1 table builder (ONE implementation)
    return G


def assemble_pass_table(out):
    """Job artifacts -> engine-porter's `res` layout -> `m2g_correctness_gate.build_pass_table` (§3-1 mapping and every
    gate scorer live THERE, tested; this function only reads files).  Missing artifact == NOT_TESTED == excluded."""
    G = _gate()
    out = Path(out)
    cfgs = ("N1",) + P.M2G_CFGS

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
    ref_ids = sorted(b.name for b in (out / "boots").glob("m2g_refuse_*")) if (out / "boots").exists() else []
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
            res["g4"][k] = G.score_g4(txt(bd / "server.log"), sa)
            res["meta"][k] = _json(bd / "cell_meta.json")
            pc = _json(bd / "proc_counts_post_boot.json")
            if pc is not None:
                res["nvml_post"][k] = {int(g): int(c or 0) for g, c in pc.items()}
        if cfg != "N1":
            g1 = _json(out / "gates" / f"g1_{cfg}.json") or {}
            for arm in C.PDMUX_ARMS:
                v = (g1.get("verdicts") or {}).get(f"fused|{arm}")
                if v is not None:
                    res["g1"][G.cell_key(cfg, arm)] = {"verdict": v}
        if mode == "tp":
            for name, sec in (("g2", f"g2_{cfg}.json"), ("g3", f"g3_TP{n}.json")):
                d = _json(out / "gates" / sec)
                if d is not None:
                    res[name][cfg] = d
            for arm in C.G3B_ARMS:
                cds = list((out / "cells").glob(f"m2g_{cfg}_{arm}__A_G3B_s*")) if (out / "cells").exists() else []
                rq = cds[0] / "requests.jsonl" if cds else None
                if rq is not None and rq.exists():
                    bj = _json(out / "boots" / f"m2g_{cfg}_{arm}" / "boot.json") or {}
                    alive = not (bj.get("status") in ("BOOT_FAILED", "BOOT_TIMEOUT")
                                 or any(s.get("status") == "SERVER_DIED" for s in bj.get("steps", [])))
                    res["g3b"][G.cell_key(cfg, arm)] = G.score_g3b(ML.load_requests(rq), alive, g_fn=ML.g_of)
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
            pre = {int(g): {"memory_used_mib": v["used_mib"], "n_procs": v["n_compute_procs"] or 0} for g, v in snap.items()}
            res["contam"][k] = G.check_contamination(pre, procs_visible=visible) if pre else {"gpu_contaminated": False,
                                                                                               "problems": ["NVML unavailable"]}
            cx = _json(out / "boots" / f"m2g_{cfg}_{arm}" / "contam.json")
            if cx and cx.get("gpu_contaminated"):
                res["contam"][k] = {"gpu_contaminated": True, "problems": [str(cx)]}
    table = G.build_pass_table(res)
    table["g5a_unit_test"] = (_json(out / "gates" / "g5a.json") or {}).get("verdict", "NOT_TESTED")
    sha = G.write_pass_table(table, out / C.M2G_PASS_TABLE)
    (out / (C.M2G_PASS_TABLE + ".sha256")).write_text(sha + "\n")
    (out / "M2G_GATE_RESULTS.json").write_text(json.dumps({k: v for k, v in res.items() if k != "meta"}, indent=1,
                                                          sort_keys=True, default=str) + "\n")
    return table


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=("m3", "m2g", "assemble-pass-table"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--hard-cap-s", type=float, default=None)
    ap.add_argument("--t0-epoch", type=float, default=None)
    ap.add_argument("--pass-table", default="")
    ap.add_argument("--pass-sha256", default="")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.cmd == "assemble-pass-table":
        t = assemble_pass_table(out)
        print(json.dumps({k: v for k, v in t.items() if k != "proc_counts"}, indent=1))
        return 0
    t0 = a.t0_epoch or time.time()
    cap = a.hard_cap_s or C.HARD_CAP_S[a.cmd]
    ctx = Ctx(out, a.dry_run, t0 + cap, t0, a.cmd)
    if a.cmd == "m2g":
        run_m2g(ctx)
        st = {"kind": "m2g", "not_attempted": ctx.not_attempted}
        (out / "DRIVER_STATUS.json").write_text(json.dumps(st, indent=1) + "\n")
        return 0
    table = None
    if a.pass_table or not a.dry_run:
        table, abort = P.load_pass_table(a.pass_table, a.pass_sha256)
        if abort:
            print(abort, file=sys.stderr)
            (out / "ABORT").write_text(abort + "\n")
            return 3
    dec = run_m3(ctx, table)
    finish(ctx, table, dec)
    return 0


if __name__ == "__main__":
    sys.exit(main())
