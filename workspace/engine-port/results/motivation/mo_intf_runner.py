#!/usr/bin/env python3
"""Job runner entry of the LooGLE interference cell.  The Job itself is mo_runner.run_job on mo_intf_spec.make_plan (same per-boot card
check, loadgen, raw schema and split-check read-out as the grid Jobs); this file adds the CLI and the rehearsal.  Not run on a GPU.

  mo_intf_runner.py plan     --job J --block 1|2 --rates R1,R2,R3
  mo_intf_runner.py card     --job J --block B --rates ... --w1-lengths L.json --out card.json
  mo_intf_runner.py run      --job J --block B --rates ... --w1-lengths L.json --card card.json --out RAW [--resume]
  mo_intf_runner.py rehearse --job J --rates R1[,..] --w1-lengths L.json --out DIR      (real arm list, first rate, one seed, 20 s cells)
  mo_intf_runner.py gate     --job J --block B --rates ...                              (exit 0 iff a passed rehearsal record exists)
The three short rates come from the pre-ladder and have no default.  The LooGLE token file must resolve (sha256 = registered) before
the first boot; both members of a pair resolve the same file.
"""
from __future__ import annotations

import argparse
import json
import sys

import os
import sys
from pathlib import Path

# the script dir Python puts on sys.path is symlink-resolved; on VESSL results/ is a symlink into the run volume, where scripts/ does
# not exist, so sealed specs imported from there cannot find scripts/costmodel.  Import siblings through the path as invoked.
_HERE = str(Path(os.path.abspath(__file__)).parent)
if sys.path[0] != _HERE:
    sys.path.insert(0, _HERE)

import mo_intf_spec as IS
import mo_intf_analyze as IA
import mo_spec as MS
import mo_card as MC
import mo_runner as MR
import mo_loadgen as ML
import mo_rehearsal as MH


def rehearse(job, w1_pool, out, make_server, replay_fn, rates, block=1, **kw):
    """Full pass: every arm (D16 / D24 / D34 / D44 + baseline) with a with / without pair, the four split-check boots and the anchors,
    through the same runner -> schema -> block analysis -> tables.  The record is the interference kind's own (mo_rehearsal.record_path)."""
    plan = IS.make_plan(job, rates, block, rehearsal=True)
    return MH.rehearse(job, w1_pool, out, make_server, replay_fn, workload=IS.WITH, plan=plan, analyze_fn=IA.analyze_block,
                       write_fn=IA.write_block, **kw)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("plan", "card", "run", "rehearse", "gate"):
        p = sub.add_parser(n)
        p.add_argument("--job", required=True)
        p.add_argument("--block", type=int, default=1, choices=sorted(IS.BLOCK_SEEDS))
        p.add_argument("--rates", required=True)
        p.add_argument("--w1-lengths")
        p.add_argument("--out")
        p.add_argument("--card")
        p.add_argument("--resume", action="store_true")
    a = ap.parse_args(argv)
    rates = [float(x) for x in a.rates.split(",") if x]
    make_server = lambda bdir, arm: MR.MoProcessServer(bdir, arm)      # noqa: E731
    if a.cmd == "rehearse":
        rec = rehearse(a.job, json.load(open(a.w1_lengths)), a.out, make_server, ML.replay, rates, a.block)
        print(json.dumps({k: rec.get(k) for k in ("passed", "kind", "harness_digest", "verdict_rows", "reason", "stages", "workloads_resolved",
                                                  "observed_venv_freeze")}))
        return 0 if rec["passed"] else MR.EXIT_ABORT
    plan = IS.make_plan(a.job, rates, a.block)
    if a.cmd == "plan":
        print(json.dumps({"plan_sha": MS.sha(plan), "design_sha": plan["design_sha"], "problems": IS.plan_problems(plan),
                          "boots": len(plan["boots"]), "estimate": IS.estimate(plan)}))
        return 0
    if a.cmd == "gate":
        why = MH.launch_gate(plan)
        print(why or "REHEARSAL_RECORD_OK")
        return 1 if why else 0
    pool = json.load(open(a.w1_lengths))
    if a.cmd == "card":
        card = MC.build_card(plan, pool)
        MR.dump(card, a.out)
        print(json.dumps({"card_sha": card["card_sha"], "placeholders": MC.card_placeholders(card)}))
        return 0
    card = json.load(open(a.card))
    try:
        res = MR.run_job(plan, card, a.out, make_server, ML.replay, pool, a.resume)
    except MR.JobAbort as e:
        print(f"{e.status}: {e.reason}", file=sys.stderr)
        return MR.EXIT_ABORT
    return MR.exit_code(res)


if __name__ == "__main__":
    sys.exit(main())
