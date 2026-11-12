#!/usr/bin/env python3
"""Job runner of the concurrency cap and decode memory traffic measurement.  CPU-tested against fakes only; nothing here ran on a GPU.

  mo_cap_runner.py table                                                     cell table, gap to the stated counts, boots per rule
  mo_cap_runner.py plan     --part fixed|serving --job J [serving: --workload W --rmin R --factor F]
  mo_cap_runner.py card     --part ... --job J ... --out card.json           [serving: --w1-lengths L.json]
  mo_cap_runner.py run      --part ... --job J ... --card card.json --out RAW [fixed: --variant-src SNAPSHOT_DIR] [--resume]
  mo_cap_runner.py rehearse --part ... --job J --out DIR                     [fixed: --variant-src; serving: --w1-lengths, --workload]
  mo_cap_runner.py gate     --part ... --job J ...                           exit 0 iff a passed rehearsal record of this part exists

Fixed batch, per boot: server start (engine_tree.py boot with the arm's interpreter, pristine tree, no PDMUX_* knob) -> health -> CARD CHECK
(argv, banner, stamp, interpreter, order; a mismatch stops the server and ABORTS THE JOB, exit 19) -> boot-log facts -> feasibility of EVERY
planned cell from the banner (max_total_num_tokens, cap, context, capture list) written BEFORE any cell -> warm-up + greedy check ->
cells (b identical requests of c_in tokens at once, ignore_eos; aborted once every request holds abort_at tokens) -> trailer -> stop.
A boot that does not come up is recorded and the Job continues (exit 17).  Raw = one JSON line per cell (mo_schema.FIXED_CELL_KEYS).
Serving cap/mem comparison: mo_runner.run_job on mo_cap_spec.serving_plan (same card check, loadgen and raw schema as the grid Jobs).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import socket
import sys
from pathlib import Path

import mo_cap_spec as CS
import mo_spec as MS
import mo_card as MC
import mo_runner as MR
import mo_schema as SC
import mo_loadgen as ML
import mo_rehearsal as MH
import mo_cap_analyze as CA
import sb_runner as RU
import ct_client as C  # sealed; imported, not modified
import ct_extract as X  # sealed; imported, not modified

SP = MS.SP
CT = CS.CT
dump = RU.dump
JobAbort = MR.JobAbort
EXIT_BOOT_FAIL, EXIT_ABORT = MR.EXIT_BOOT_FAIL, MR.EXIT_ABORT


def stamp():
    return {**RU.stamp(), "hostname": socket.gethostname()}


# ---- card ---------------------------------------------------------------------------------------------------------------------------------
def fixed_card(plan):
    """Card of a fixed-batch plan: the registered expectation of every boot (mo_card.arm_expect) + the cell geometry of every cell."""
    cells = {b["boot_id"]: [[bs, c, CS.c_in(c), CS.o_tokens(c), CS.abort_at(c)] for bs, c in b["cells"]] for b in plan["boots"]}
    card = {"schema": CS.SCHEMA_FIXED + "/card", "job": plan["job"], "kind": plan["kind"], "rehearsal": bool(plan["rehearsal"]),
            "plan_sha": MS.sha(plan), "arms": MC.arms_expect(plan), "boot_order": list(plan["boot_order"]), "anchor": plan["anchor"],
            "cells": cells, "cell_table": plan["cell_table"], "mem": plan["mem"], "context_length": plan["context_length"],
            "values": {k: v[0] for k, v in CS.PROPOSED.items()}, "open_items": CS.open_items(), "analytic": CS.AT.VERSION}
    card["card_sha"] = MC.sha({k: v for k, v in card.items() if k != "card_sha"})
    return card


# ---- servers --------------------------------------------------------------------------------------------------------------------------------
class FixedBatchServer(MR.MoProcessServer):
    """The grid runner's server (engine_tree.py boot, the arm's interpreter) + the cost-term harness transport for burst cells."""

    def start(self):
        st = super().start()
        self.tr = C.HttpTransport(self.port)
        return st

    def alive(self):
        return self.proc is not None and self.proc.poll() is None


def build_variant(src, dst, n_attn=31):
    """a31 model directory (config + tokenizer, no weights) from the Nano snapshot, by the registered composition rule."""
    spec = importlib.util.spec_from_file_location("mo_cap_make_variant", SP.ENG / "scripts" / "compo" / "make_variant.py")
    mv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mv)
    return mv.build_variant(str(src), str(dst), n_attn, force=True)


# ---- one cell -------------------------------------------------------------------------------------------------------------------------------
def run_cell(tr, cell_id, bs, c, rnd, context, timeout):
    """b requests at once, each the SAME seeded prompt of c_in tokens, max_new = O, ignore_eos.  The burst is aborted once every request
    has received abort_at tokens (the window after the last join has then been executed)."""
    ci, o, ab = CS.c_in(c), CS.o_tokens(c), CS.abort_at(c)
    ids = CT.prompt_ids("FB", bs, c, rnd, ci)
    rec = {"cell_id": cell_id, "bs": bs, "c": c, "c_in": ci, "O": o, "abort_at": ab, "window_iters": CS.window_iters(c),
           "trim_iters": CS.trim_iters(), "time_only": CS.time_only(c), "round": rnd,
           "clock_pair": tr.clock_pair() if hasattr(tr, "clock_pair") else None}
    rec["t_first_send"] = tr.now()
    hs = [tr.start(f"{cell_id}-q{i}", ids, o) for i in range(bs)]
    rec["window_reached"] = tr.wait(lambda: all(h.done or (h.chunks[-1][1] if h.chunks else 0) >= ab for h in hs), timeout)
    rec["t_abort"] = tr.now()
    C.cancel_burst(tr, hs)
    rec["t_end"] = tr.now()
    rec["requests"] = [{k: v for k, v in h.record().items() if k in SC.FIXED_REQUEST_KEYS} for h in hs]
    return rec


def feasibility(b, facts):
    """Every planned cell of a boot against the boot log (never against the plan): banner pool, logged cap / context, capture list."""
    kv = (facts.get("final") or {}).get("max_total_num_tokens")
    cap, ctx = X.logged_cap(facts), X.logged_context(facts)
    rows = []
    for bs, c in (tuple(x) for x in b["cells"]):
        r = CS.fits_banner(bs, c, kv, cap, ctx, X.bs_capturable(bs, facts.get("capture_bs")))
        rows.append({"bs": bs, "c": c, "run": not r, "label": None if not r else "INFEASIBLE_BY_RULE", "reasons": r, "kv_tokens": kv,
                     "cap_logged": cap, "context_logged": ctx,
                     "predicted_kv_tokens": CS.predicted_kv_tokens(b["model"], b["cap"], b["arm_def"]["params"]["mem"])})
    return rows


# ---- one boot -------------------------------------------------------------------------------------------------------------------------------
def run_fixed_boot(b, card, out, make_server, ctx, py_version_fn=MC.probe_python_version, freeze_fn=MC.freeze_sha, variant_fn=None):
    bdir = Path(out) / "boots" / b["boot_id"]
    ctx.setdefault("executed", []).append(b["arm"])
    if ctx.get("resume") and (bdir / "DONE.json").exists() and json.loads((bdir / "DONE.json").read_text()).get("status") == "done":
        return {"boot_id": b["boot_id"], "status": "done", "resumed": True}
    bdir.mkdir(parents=True, exist_ok=True)
    for p in bdir.iterdir():                          # a partial boot is redone from scratch
        if p.is_file():
            p.unlink()
    a = RU.resolve(b["arm_def"], bdir)
    if CS.MODELS[b["model"]]["weights"] == "dummy":
        (variant_fn or (lambda dst: build_variant(ctx["variant_src"], dst)))(bdir / CS.VARIANT_DIRNAME)
    srv = make_server(bdir, a)
    status = srv.start()
    log_text = srv.log.read_text(errors="replace") if srv.log.exists() else ""
    tp = bdir / f"tree_{a['name']}.json"
    tree_stamp = json.loads(tp.read_text()) if tp.exists() else {}
    observed = MC.observe_boot(a, getattr(srv, "cmd", None), tree_stamp, log_text, py_version_fn, freeze_fn, bdir)
    dump({"boot_id": b["boot_id"], "arm": b["arm"], "role": b["role"], "anchor": b.get("anchor"), "model": b["model"], "cap": b["cap"],
          "round": b["round"], "status": status, "index": len(ctx["executed"]) - 1, "cmd": getattr(srv, "cmd", None), "observed": observed,
          "tree_stamp": tree_stamp, **stamp()}, bdir / "boot.json")
    if status != "ok":
        srv.stop()
        dump({"boot_id": b["boot_id"], "status": status}, bdir / "DONE.json")
        return {"boot_id": b["boot_id"], "status": status}
    bad = MC.check_arm(card, b["arm"], observed) + MC.check_order(card, ctx["executed"])          # before anything is measured
    if bad:
        reason = MC.format_mismatches(b["boot_id"], b["arm"], bad)
        dump({"boot_id": b["boot_id"], "arm": b["arm"], "mismatches": bad, "reason": reason, "observed": observed}, bdir / "CARD_MISMATCH.json")
        dump({"boot_id": b["boot_id"], "status": "ABORT_CARD_MISMATCH", "reason": reason, "server": srv.stop()}, bdir / "DONE.json")
        raise JobAbort("ABORT_CARD_MISMATCH", reason, b["boot_id"])
    facts = X.parse_boot_log(log_text)
    dump(facts, bdir / "facts.json")
    feas = feasibility(b, facts)
    dump(feas, bdir / "feasibility.json")              # every planned cell has a row before the first cell is sent
    banner = {"max_total_num_tokens": (facts.get("final") or {}).get("max_total_num_tokens"), "max_running_requests": X.logged_cap(facts),
              "context_len": X.logged_context(facts)}
    tr, done, label = srv.tr, [], "done"
    kv = banner["max_total_num_tokens"]
    try:
        if not any(f["run"] for f in feas):
            wu = {"skipped": "no planned cell is feasible under this boot's banner (every cell has an INFEASIBLE_BY_RULE row)"}
        elif kv is None or kv < CT.WARMUP_IN + CT.WARMUP_OUT:
            wu, label = {"skipped": f"banner pool {kv} cannot hold one warm-up request"}, "POOL_TOO_SMALL_FOR_WARMUP"
        else:
            wu = C.run_warmup(tr, b["boot_id"])
            if not (wu["warmup_ok"] and wu["greedy_ok"]):
                label = "INVALID_CORRECTNESS"
        dump(wu, bdir / "correctness.json")
        if label == "done" and "skipped" not in wu:
            with open(bdir / "cells.jsonl", "a") as fh:
                for f in feas:
                    if not f["run"]:
                        continue
                    if not srv.alive():
                        label = "SERVER_DEATH"
                        break
                    bs, c = f["bs"], f["c"]
                    rec = run_cell(tr, f"{b['boot_id']}_b{bs}_c{c}", bs, c, b["round"], banner["context_len"],
                                   max(600.0, 5.0 * CS.est_cell_s(b["model"], bs, c)))
                    rec.update(schema_version=CS.SCHEMA_FIXED, boot_id=b["boot_id"], arm=b["arm"], card_sha=card["card_sha"], model=b["model"],
                               cap=b["cap"], mem=a["params"]["mem"], banner=banner, server_alive_after=srv.alive(), **stamp())
                    fh.write(json.dumps(rec) + "\n")          # one line = one finished cell
                    fh.flush()
                    done.append(rec["cell_id"])
                    if not rec["server_alive_after"]:
                        label = "SERVER_DEATH"
                        break
                    tr.sleep(CS.SETTLE_S)
            if label == "done":
                C.run_trailer(tr, b["boot_id"])
    finally:
        info = srv.stop()
    dump({"boot_id": b["boot_id"], "status": label, "cells": done, "server": info, **stamp()}, bdir / "DONE.json")
    return {"boot_id": b["boot_id"], "status": label, "cells": done}


def run_fixed_job(plan, card, out, make_server, resume=False, gate_fn=None, py_version_fn=MC.probe_python_version, freeze_fn=MC.freeze_sha,
                  variant_fn=None, variant_src=None):
    """-> list of boot results.  Raises JobAbort (after writing JOB_ABORTED.json) on a plan / card refusal or a card mismatch."""
    out = Path(out)
    try:
        problems = CS.fixed_plan_problems(plan)
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
            why = (gate_fn or MH.launch_gate)(plan)
            if why:
                raise JobAbort("ABORT_NO_REHEARSAL", why)
        if out.exists() and any((out / "boots").glob("*/DONE.json")) and not resume:
            raise JobAbort("ABORT_RAW_NOT_EMPTY", "raw directory has boots; set --resume")
        if variant_fn is None and variant_src is None and any(CS.MODELS[b["model"]]["weights"] == "dummy" for b in plan["boots"]):
            raise JobAbort("ABORT_VARIANT_SRC", "the a31 boots need the Nano snapshot directory (--variant-src)")
        dump(plan, out / "plan.json")
        dump(card, out / "card.json")
        ctx, res = {"variant_src": variant_src, "resume": resume}, []
        for b in plan["boots"]:
            res.append(run_fixed_boot(b, card, out, make_server, ctx, py_version_fn, freeze_fn, variant_fn))
    except JobAbort as e:
        dump({"job": plan.get("job"), "status": e.status, "reason": e.reason, "boot_id": e.boot_id, **stamp()}, out / "JOB_ABORTED.json")
        raise
    dump({"job": plan["job"], "plan_sha": MS.sha(plan), "card_sha": card["card_sha"], "boots": res, "resumed": bool(resume), **stamp()},
         out / "RUN_DONE.json")
    return res


# ---- rehearsal ------------------------------------------------------------------------------------------------------------------------------
def rehearse_fixed(job, out, make_server, rehearsal_dir=None, digest=None, **kw):
    """Full pass of the fixed-batch part: the REAL boot list (every model x cap), one cell per boot, through the same runner (card
    check included) -> schema -> analysis -> cell table.  Never raises on a stage failure; the record names the failing stage."""
    out = Path(out)
    rec = MH.new_record(job, digest, kind=CS.KIND_FIXED)
    try:
        plan = CS.fixed_plan(job, rehearsal=True)
        bad = CS.fixed_plan_problems(plan)
        if bad:
            raise RuntimeError(f"plan: {bad}")
        rec["stages"]["plan"] = True
        rec["arm_names"] = sorted({b["arm"] for b in plan["boots"]})
        rec["plan_sha"] = MS.sha(plan)
        card = fixed_card(plan)
        rec["card_sha"] = card["card_sha"]
        rec["stages"]["card"] = True
        res = run_fixed_job(plan, card, out, make_server, **kw)
        ok = all(b["status"] == "done" for b in res)
        rec["stages"]["boots"] = rec["stages"]["card_checks"] = ok      # a mismatch aborts the Job before this line is reached
        if not ok:
            raise RuntimeError("boot failures: " + ",".join(f"{b['boot_id']}={b['status']}" for b in res if b["status"] != "done"))
        job_data = CA.load_fixed(out)
        cells = [c for b in job_data["boots"] for c in b["cells"]]
        rec["stages"]["runs"] = len(cells) == sum(1 for b in job_data["boots"] for f in b["feasibility"] if f["run"]) and bool(cells)
        probs = [m for c in cells for m in SC.validate_fixed_cell(c)]
        if probs:
            raise RuntimeError(f"schema: {probs[:3]}")
        rec["stages"]["schema"] = True
        result = CA.analyze_fixed(job_data)
        rec["stages"]["analyzer"] = True
        CA.write_fixed(result, out / "analysis")
        rec["verdict_rows"] = sum(1 for r in result["verdict_rows"] if r["n_rounds_valid"])
        rec["stages"]["verdict_table"] = rec["verdict_rows"] > 0
        rec["verdict_table_sha"] = MS.sha(result["verdict_rows"])
    except JobAbort as e:
        rec["reason"] = f"{e.status}: {e.reason}"
    except Exception as e:  # noqa: BLE001
        rec["reason"] = f"{type(e).__name__}: {e}"
    return MH.write_record(rec, out, rehearsal_dir)


def rehearse_serving(job, w1_pool, out, make_server, replay_fn, rmin=3.0, factor=1.15, workload="W1", **kw):
    """Full pass of the serving part: every (arm, mem, cap) boot + anchors + split check, one rung, few requests."""
    plan = CS.serving_plan(job, workload, rmin, factor, n_requests=MH.REHEARSAL_N_REQUESTS, rehearsal=True)
    return MH.rehearse(job, w1_pool, out, make_server, replay_fn, workload=workload, plan=plan, analyze_fn=CA.analyze_serving,
                       write_fn=CA.write_serving, **kw)


# ---- CLI --------------------------------------------------------------------------------------------------------------------------------------
def _plan(a):
    if a.part == "fixed":
        if not a.rehearsal and a.round is None:
            raise SystemExit("ABORT_CONFIG: the fixed-batch part needs --round (one Job per round)")
        return CS.fixed_plan(a.job, rounds=None if a.rehearsal else (a.round,), rehearsal=a.rehearsal)
    return CS.serving_plan(a.job, a.workload, a.rmin, a.factor, rehearsal=a.rehearsal,
                           n_requests=MH.REHEARSAL_N_REQUESTS if a.rehearsal else None)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("table")
    for n in ("plan", "card", "run", "rehearse", "gate"):
        p = sub.add_parser(n)
        p.add_argument("--part", required=True, choices=("fixed", "serving"))
        p.add_argument("--job", required=True)
        p.add_argument("--workload", default=MS.WORKLOAD)
        p.add_argument("--rmin", type=float)
        p.add_argument("--factor", type=float)
        p.add_argument("--round", type=int)
        p.add_argument("--rehearsal", action="store_true")
        p.add_argument("--w1-lengths")
        p.add_argument("--variant-src")
        p.add_argument("--out")
        p.add_argument("--card")
        p.add_argument("--resume", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "table":
        return CS.main(["table"])
    if a.part == "serving" and a.cmd != "rehearse" and (a.rmin is None or a.factor is None):
        print("ABORT_CONFIG: the serving part needs --rmin and --factor (the rate grid comes from the pre-ladder; no default)", file=sys.stderr)
        return 16
    if a.cmd == "rehearse":
        if a.part == "fixed":
            rec = rehearse_fixed(a.job, a.out, lambda bdir, arm: FixedBatchServer(bdir, arm), variant_src=a.variant_src)
        else:
            rec = rehearse_serving(a.job, json.load(open(a.w1_lengths)), a.out, lambda bdir, arm: MR.MoProcessServer(bdir, arm), ML.replay,
                                   a.rmin or 3.0, a.factor or 1.15, a.workload)
        print(json.dumps({k: rec.get(k) for k in ("passed", "kind", "harness_digest", "verdict_rows", "reason", "stages")}))
        return 0 if rec["passed"] else EXIT_ABORT
    plan = _plan(a)
    if a.cmd == "plan":
        est = CS.estimate_fixed(plan) if a.part == "fixed" else CS.estimate_serving(plan)
        print(json.dumps({"plan_sha": MS.sha(plan), "problems": CS.plan_problems(plan), "boots": len(plan["boots"]), "estimate": est}))
        return 0
    if a.cmd == "gate":
        why = MH.launch_gate(plan)
        print(why or "REHEARSAL_RECORD_OK")
        return 1 if why else 0
    if a.part == "serving" and a.cmd == "run" and not a.rehearsal and a.workload == "W1":
        print("ABORT_WORKLOAD: W1 is the rehearsal fixture; a measurement Job names a mo_workload.WORKLOADS workload", file=sys.stderr)
        return EXIT_ABORT
    pool = json.load(open(a.w1_lengths)) if a.part == "serving" else None
    if a.cmd == "card":
        card = fixed_card(plan) if a.part == "fixed" else MC.build_card(plan, pool)
        dump(card, a.out)
        print(json.dumps({"card_sha": card["card_sha"], "placeholders": MC.card_placeholders(card), "open_items": CS.open_items()}))
        return 0
    card = json.load(open(a.card))
    try:
        if a.part == "fixed":
            res = run_fixed_job(plan, card, a.out, lambda bdir, arm: FixedBatchServer(bdir, arm), a.resume, variant_src=a.variant_src)
        else:
            res = MR.run_job(plan, card, a.out, lambda bdir, arm: MR.MoProcessServer(bdir, arm), ML.replay, pool, a.resume)
    except JobAbort as e:
        print(f"{e.status}: {e.reason}", file=sys.stderr)
        return EXIT_ABORT
    return EXIT_BOOT_FAIL if any(b["status"] != "done" for b in res) else 0


if __name__ == "__main__":
    sys.exit(main())
