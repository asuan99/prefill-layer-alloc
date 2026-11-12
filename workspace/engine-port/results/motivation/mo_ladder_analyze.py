#!/usr/bin/env python3
"""Reading of a pre-ladder Job: where the capacity boundary of every arm lies, and the rate grids that follow.  CPU only.

NOT A MEASUREMENT (mo_ladder.NOT_A_MEASUREMENT is on every output): provisional cap / mem, two seeds, no anchor.  The numbers choose
grids; they are not capacities and are never merged with, or cited next to, a measurement.

Per (target, arm), at the STRICTEST and the LOOSEST pair of the registered SLO grid (mo_spec.TTFT_GRID_MS x ITL_GRID_MS): the pick rule
of the earlier ladders (sb_runner.lambda_pick: ascending rungs, every seed >= CAP_ATTAIN) gives the last passing rung and the bracket
[last passing rung, first failing rung].  A grid must contain the boundary of every arm at every SLO pair it will be read at, so per target
    envelope = [ lowest strict-SLO boundary bracket start over the arms , highest loose-SLO bracket end over the arms ]
and the proposed base grid of n rungs is rmin = envelope low, factor = (high / low)^(1 / (n - 1)) -- the form of mo_spec.make_grid; its
registered single extension lies above the envelope.  n: serving cap/mem = mo_cap_spec.SERVING_RUNGS (4); stage-1 n is not registered
(PENDING unless given).  A target whose ladder did not bracket some arm gets NO proposal, only the side to extend.
Interference cell: its reading (1) needs the WITHOUT member >= SAT_HI on the input<=512 class, so the eligible short rates of the
interference_short target are the rungs up to the last one where every arm and seed is >= SAT_HI on that class.  HOW three rates are
taken from the eligible rungs is not in any document (INTF_RATE_RULE is PENDING); the candidate shown is labelled as such.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import os
import sys
from pathlib import Path

# the script dir Python puts on sys.path is symlink-resolved; on VESSL results/ is a symlink into the run volume, where scripts/ does
# not exist, so sealed specs imported from there cannot find scripts/costmodel.  Import siblings through the path as invoked.
_HERE = str(Path(os.path.abspath(__file__)).parent)
if sys.path[0] != _HERE:
    sys.path.insert(0, _HERE)

import mo_spec as MS
import mo_analyze as MA
import mo_card as MC
import mo_schema as SC
import mo_ladder as LD
import mo_cap_spec as CS
import mo_intf_analyze as IA
import sb_runner as RU

STRICT = (min(MS.TTFT_GRID_MS), min(MS.ITL_GRID_MS))
LOOSE = (max(MS.TTFT_GRID_MS), max(MS.ITL_GRID_MS))
STAGE1_N = MS.pend("prereg.stage1_grid.n")                    # rungs of the stage-1 base grid: not registered anywhere
INTF_RATE_RULE = MS.pend("prereg.interference.rate_rule")     # how the three short rates are taken from the eligible rungs


def problems(job):
    pr = []
    plan, card = job["plan"], job["card"]
    if plan is None or card is None:
        return ["plan.json / card.json missing"]
    if plan.get("kind") != LD.KIND:
        pr.append("not a pre-ladder plan")
    if not MC.card_sha_ok(card) or card["plan_sha"] != MS.sha(plan):
        pr.append("card.json is not the card of plan.json")
    if job["aborted"]:
        pr.append(f"Job aborted: {job['aborted'].get('status')}: {str(job['aborted'].get('reason'))[:200]}")
    if len({r["job_name"] for r in job["runs"]}) > 1:
        pr.append("runs of several Jobs in one raw directory: ladders of two Jobs are never merged")
    have = {r["run_id"]: r for r in job["runs"]}
    for r in job["runs"]:
        pr += [f"schema: {m}" for m in SC.validate_run(r)[:3]]
        if r["card_sha"] != card["card_sha"]:
            pr.append(f"run {r['run_id']}: card sha differs from card.json")
    for b in plan["boots"]:
        if job["done"].get(b["boot_id"], {}).get("status") != "done":
            continue                                           # a boot that failed loses its own points only (reported per arm)
        for run in b["runs"]:
            r = have.get(run["run_id"])
            if r is None or len(MA.reqs_of(r)) != run["n_requests"] or card["trace"][run["run_id"]]["hash"] != r["trace_hash"]:
                pr.append(f"run {run['run_id']}: missing, wrong request count or wrong trace")
    return pr


def bracket(points, rates, level):
    """pick rule -> {"last_pass", "first_fail", "status"}; BRACKETED iff both exist."""
    lam, st = RU.lambda_pick(points, level)
    if st == "LADDER_BOTTOM_FAIL":
        return {"last_pass": None, "first_fail": rates[0], "status": st}
    if st != "BRACKETED":
        return {"last_pass": lam, "first_fail": None, "status": st}
    return {"last_pass": lam, "first_fail": rates[rates.index(lam) + 1], "status": st}


def propose_grid(lo, hi, n):
    """Base grid of n rungs spanning [lo, hi] in the form of mo_spec.make_grid (rmin, factor, n)."""
    if MS.is_pending(n):
        return {"n": n, "status": "PENDING_N"}
    factor = (hi / lo) ** (1.0 / (n - 1))
    g = MS.make_grid(round(lo, 6), round(factor, 6), n)
    return {"n": n, "rmin": g["rmin"], "factor": g["factor"], "rates": g["rates"], "extension_rates": g["extension_rates"], "status": "PROPOSED"}


def analyze(raw, stage1_n=None):
    job = raw if isinstance(raw, dict) else MA.load_job(raw)
    pr = problems(job)
    if pr:
        raise MA.JobInvalid(pr)
    plan = job["plan"]
    runs_by_boot = {}
    for r in job["runs"]:
        runs_by_boot.setdefault(r["boot_id"], []).append(r)
    rows, targets = [], {}
    for b in plan["boots"]:
        t, key = b["target"], b["arm_def"]["arm_key"]
        rates = LD.rates_of(plan["grids"][t])
        st = job["done"].get(b["boot_id"], {}).get("status")
        row = {"target": t, "arm_key": key, "arm": b["arm"], "boot_status": st, "rates": rates}
        if st == "done":
            runs = runs_by_boot.get(b["boot_id"], [])
            for name, pair in (("strict", STRICT), ("loose", LOOSE), ("primary", MS.PRIMARY_SLO)):
                pts = [(r["spec"]["rate"], IA.attain(MA.reqs_of(r), *pair)) for r in runs]
                row[name] = bracket(pts, rates, MS.CAP_ATTAIN)
            short = [(r["spec"]["rate"], IA.attain([x for x in MA.reqs_of(r) if x["req_class"] == IA.SHORT_CLASS], *MS.PRIMARY_SLO)) for r in runs]
            row["short_class_sat"] = bracket([(rt, a if a is not None else 0.0) for rt, a in short], rates, MS.SAT_HI)
            row["attainment_by_rate"] = {str(rt): {n: sorted(round(IA.attain(MA.reqs_of(r), *p), 4) for r in runs if r["spec"]["rate"] == rt)
                                                   for n, p in (("strict", STRICT), ("loose", LOOSE), ("primary", MS.PRIMARY_SLO))} for rt in rates}
        rows.append(row)
        targets.setdefault(t, []).append(row)
    out_t = {}
    for t, rs in targets.items():
        done = [r for r in rs if r["boot_status"] == "done"]
        res = {"target": t, "feeds": LD.TARGETS[t]["feeds"], "arms": [r["arm_key"] for r in rs], "ladder": LD.rates_of(plan["grids"][t]),
               "failed_boots": [r["arm_key"] for r in rs if r["boot_status"] != "done"]}
        extend = sorted({"EXTEND_DOWN" for r in done if r["strict"]["status"] == "LADDER_BOTTOM_FAIL"}
                        | {"EXTEND_UP" for r in done if r["loose"]["status"] == "LADDER_TOP_PASS"}
                        | {"EXTEND_DOWN" for r in done if r["loose"]["status"] == "LADDER_BOTTOM_FAIL"}
                        | {"EXTEND_UP" for r in done if r["strict"]["status"] == "LADDER_TOP_PASS"})
        if res["failed_boots"] or not done:
            res.update(status="INCOMPLETE", grids=None)
        elif extend:
            res.update(status="NOT_BRACKETED", extend=extend, grids=None)
        else:
            lo = min(r["strict"]["last_pass"] for r in done)
            hi = max(r["loose"]["first_fail"] for r in done)
            res.update(status="BRACKETED", envelope=[lo, hi],
                       lowest_strict_arm=min(done, key=lambda r: r["strict"]["last_pass"])["arm_key"],
                       highest_loose_arm=max(done, key=lambda r: r["loose"]["first_fail"])["arm_key"],
                       grids={"stage1": propose_grid(lo, hi, stage1_n if stage1_n else STAGE1_N),
                              "serving_cap_mem": propose_grid(lo, hi, CS.SERVING_RUNGS)})
        if t == "interference_short":
            res["interference_rates"] = interference_rates(done, res["ladder"]) if done and not res["failed_boots"] else {"status": "INCOMPLETE"}
        out_t[t] = res
    names = sorted({r["job_name"] for r in job["runs"]})
    return {"label": LD.NOT_A_MEASUREMENT, "job": names[0] if len(names) == 1 else names, "kind": LD.KIND, "provisional": plan["provisional"],
            "seeds": plan["seeds"], "slo": {"strict": list(STRICT), "loose": list(LOOSE), "primary": list(MS.PRIMARY_SLO)},
            "levels": {"CAP_ATTAIN": MS.CAP_ATTAIN, "SAT_HI": MS.SAT_HI}, "rows": rows, "targets": out_t, "cross_job": "NOT_MERGED",
            "pending": {"stage1_n": None if stage1_n else STAGE1_N, "interference_rate_rule": INTF_RATE_RULE}}


def interference_rates(done, ladder):
    """Eligible short rates = ladder rungs up to the last one where EVERY arm and seed keeps the input<=512 class >= SAT_HI."""
    if any(r["short_class_sat"]["status"] == "LADDER_BOTTOM_FAIL" for r in done):
        return {"status": "NOT_BRACKETED", "extend": ["EXTEND_DOWN"], "eligible_rungs": []}
    top = min(r["short_class_sat"]["last_pass"] for r in done)
    elig = [x for x in ladder if x <= top]
    out = {"status": "ELIGIBLE_RANGE", "eligible_rungs": elig, "limiting_arm": min(done, key=lambda r: r["short_class_sat"]["last_pass"])["arm_key"],
           "top_not_reached": all(r["short_class_sat"]["status"] == "LADDER_TOP_PASS" for r in done), "rule": INTF_RATE_RULE,
           "rehearsal_rate_candidate": elig[0]}
    out["candidate_unconfirmed"] = elig[-IS_N:] if len(elig) >= IS_N else None      # the top three eligible rungs: a candidate, not a rule
    return out


IS_N = LD.IS.N_RATES
COLS = ("target", "arm_key", "boot_status", "strict_last_pass", "strict_first_fail", "strict_status", "loose_last_pass", "loose_first_fail",
        "loose_status")


def write_outputs(res, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "pre_ladder.json").write_text(json.dumps(res, indent=1, sort_keys=True, default=str) + "\n")
    with open(out / "pre_ladder.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([res["label"]])
        w.writerow(COLS)
        for r in res["rows"]:
            s, lo = r.get("strict") or {}, r.get("loose") or {}
            w.writerow([r["target"], r["arm_key"], r["boot_status"], s.get("last_pass"), s.get("first_fail"), s.get("status"), lo.get("last_pass"),
                        lo.get("first_fail"), lo.get("status")])
    return out / "pre_ladder.csv"


def main(argv=None):
    """mo_ladder_analyze.py RAW --out DIR [--stage1-n N]"""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage1-n", type=int)
    a = ap.parse_args(argv)
    try:
        print(write_outputs(analyze(a.raw, a.stage1_n), a.out))
    except MA.JobInvalid as e:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / "JOB_INVALID.json").write_text(json.dumps({"problems": e.problems, "label": LD.NOT_A_MEASUREMENT}, indent=1) + "\n")
        print(f"JOB_INVALID: {e}")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
