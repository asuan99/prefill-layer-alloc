#!/usr/bin/env python3
"""Analysis of the concurrency cap and decode memory traffic measurement.  CPU only; computes measurements and registered labels.

Fixed batch (raw of mo_cap_runner.run_fixed_job):
  * per cell, from the client token stamps of every request: decode step time (median per-token gap inside the window that starts
    trim tokens after the LAST request joined), tokens/s = b / step, mean context inside the window;
  * decode memory traffic: analytic bytes per step at the realized context (analytic_terms.decode_step, the only source of analytic
    numbers) divided by the measured step -> bytes/s, and the measured step over the HBM roofline bound (ct_spec.step_ms_lb);
  * round agreement, anchor drift, throughput gain per batch doubling.  The anchor boots are a drift check, never a table row.
Serving cap/mem comparison (raw of mo_runner.run_job on mo_cap_spec.serving_plan):
  * the same Job validity, post-hoc SLO grid and reading rule as mo_analyze (imported); per arm, every registered pair of combinations
    that differ in ONE factor (mem at cap 48; cap steps at mem 0.75) at every rung;
  * fix_cap_mem(): the coded rule that turns those pair rows into the mem / cap of the later measurements.  Its thresholds are
    mo_cap_spec.FIX_RULE; while one of them is PENDING the rule answers PENDING_THRESHOLD and fixes nothing.
One Job = one evaluation unit: nothing here merges Jobs.
"""
from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

import os
import sys
from pathlib import Path

# the script dir Python puts on sys.path is symlink-resolved; on VESSL results/ is a symlink into the run volume, where scripts/ does
# not exist, so sealed specs imported from there cannot find scripts/costmodel.  Import siblings through the path as invoked.
_HERE = str(Path(os.path.abspath(__file__)).parent)
if sys.path[0] != _HERE:
    sys.path.insert(0, _HERE)

import mo_cap_spec as CS
import mo_spec as MS
import mo_card as MC
import mo_schema as SC
import mo_analyze as MA
import ct_extract as X  # sealed; imported, not modified

AT, CT = CS.AT, CS.CT


def _j(p):
    return json.loads(Path(p).read_text())


# ==================================================================== fixed batch
def load_fixed(raw):
    raw = Path(raw)
    boots = []
    for d in sorted((raw / "boots").glob("*")):
        if not (d / "boot.json").exists():
            continue
        cells = [json.loads(ln) for ln in (d / "cells.jsonl").read_text().splitlines() if ln.strip()] if (d / "cells.jsonl").exists() else []
        boots.append({"boot": _j(d / "boot.json"), "done": _j(d / "DONE.json") if (d / "DONE.json").exists() else {},
                      "feasibility": _j(d / "feasibility.json") if (d / "feasibility.json").exists() else [], "cells": cells})
    return {"raw": str(raw), "boots": sorted(boots, key=lambda b: b["boot"].get("index", 0)),
            "plan": _j(raw / "plan.json") if (raw / "plan.json").exists() else None,
            "card": _j(raw / "card.json") if (raw / "card.json").exists() else None,
            "aborted": _j(raw / "JOB_ABORTED.json") if (raw / "JOB_ABORTED.json").exists() else None}


def _tok_time(chunks, n):
    """Client time at which a request had received n tokens (None if it never did)."""
    return next((t for t, cum in chunks if cum >= n), None)


def cell_window(rec):
    """Decode step of one cell from the client stamps.  Reach = the first token of the LAST request to join (every request is then
    decoding); trim tokens of that request are skipped, the window is its next window_iters tokens and must close before any request
    ended by length.  Status: BENCH_INVALID > WINDOW_SHORT > VALID.  Labels (never exclusions): CTX_OFF_NOMINAL, TIME_ONLY."""
    reqs, bs, ci, o = rec["requests"], rec["bs"], rec["c_in"], rec["O"]
    trim, win = rec["trim_iters"], rec["window_iters"]
    base = {"cell_id": rec["cell_id"], "model": rec["model"], "bs": bs, "c": rec["c"], "round": rec["round"], "labels": []}
    if rec.get("time_only"):
        base["labels"].append("TIME_ONLY")
    ok = len(reqs) == bs and all(r.get("error") is None and r.get("prompt_tokens") == ci and rec["abort_at"] <= (r.get("completion_tokens") or 0) <= o
                                 and r.get("finish") in ("abort", "length") for r in reqs)
    if not ok:
        return dict(base, status="BENCH_INVALID", detail="request error / input tokens / fewer than abort_at tokens")
    ended = [r["chunks"][-1][0] for r in reqs if r.get("finish") == "length"]
    last = max(reqs, key=lambda r: r["chunks"][0][0])
    t_lo, t_hi = _tok_time(last["chunks"], 1 + trim), _tok_time(last["chunks"], 1 + trim + win)
    if t_lo is None or t_hi is None or (ended and t_hi >= min(ended)):
        return dict(base, status="WINDOW_SHORT", detail="the window did not close before a request ended")
    gaps = sorted(X.client_gaps(reqs, t_lo, t_hi))
    if not gaps:
        return dict(base, status="WINDOW_SHORT", detail="no token gap inside the window")
    pts = [t for t, _ in last["chunks"] if t_lo <= t < t_hi]
    ctx = [sum(ci + X.tokens_by(r["chunks"], t) for r in reqs) / bs for t in pts]
    ctx_mean = sum(ctx) / len(ctx)
    if abs(ctx_mean - rec["c"]) > CT.CTX_BAND:
        base["labels"].append("CTX_OFF_NOMINAL")
    step = X.median(gaps) * 1000.0
    return dict(base, status="VALID", step_ms=step, step_p10_ms=X.quantile(gaps, 0.1) * 1000.0, step_p90_ms=X.quantile(gaps, 0.9) * 1000.0,
                n_gaps=len(gaps), ctx_mean=ctx_mean, tokens_per_s=bs * 1000.0 / step,
                tokens_before_last_join=max(X.tokens_by(r["chunks"], last["chunks"][0][0]) for r in reqs) - 1)


def traffic(model, bs, ctx_mean, step_ms):
    """Analytic bytes of one decode step at the realized context [derived] against the measured step."""
    a = CS.arch(model)
    ds = AT.decode_step(a, bs, int(round(ctx_mean)))
    comp = {"weights": ds["bytes_weights"], "state": ds["bytes_state_rw"], "kv": ds["bytes_kv_read"]}
    roof = CT.step_ms_lb(bs, ctx_mean, a)
    return {"bytes_per_step": ds["bytes_total"], "bytes_weights": comp["weights"], "bytes_state": comp["state"], "bytes_kv": comp["kv"],
            "largest_term": max(comp, key=comp.get), "bytes_per_s": ds["bytes_total"] / (step_ms / 1000.0),
            "roofline_ms": roof, "step_over_roofline": step_ms / roof, "analytic": AT.VERSION}


def fixed_problems(job):
    """-> (problems, flags).  problems are fatal (no table); flags travel with every row."""
    pr, fl = [], []
    plan, card = job["plan"], job["card"]
    if plan is None or card is None:
        return ["plan.json / card.json missing"], fl
    if not MC.card_sha_ok(card) or card["plan_sha"] != MS.sha(plan):
        pr.append("card.json is not the card of plan.json")
    if job["aborted"]:
        pr.append(f"Job aborted: {job['aborted'].get('status')}: {str(job['aborted'].get('reason'))[:200]}")
    by = {b["boot"]["boot_id"]: b for b in job["boots"]}
    if [b["boot"]["arm"] for b in job["boots"]] != card["boot_order"]:
        pr.append("realized boot order differs from the registered order")
    for pb in plan["boots"]:
        b = by.get(pb["boot_id"])
        st = (b or {}).get("done", {}).get("status")
        if st != "done":
            pr.append(f"boot {pb['boot_id']} status {st}")
            continue
        cells = {(c["bs"], c["c"]): c for c in b["cells"]}
        feas = {(f["bs"], f["c"]): f for f in b["feasibility"]}
        for bs, c in (tuple(x) for x in pb["cells"]):
            if (bs, c) not in feas:
                pr.append(f"{pb['boot_id']}: no feasibility row for cell {bs}x{c} (written before any cell)")
            elif feas[(bs, c)]["run"] and (bs, c) not in cells:
                pr.append(f"{pb['boot_id']}: feasible cell {bs}x{c} has no record (incomplete block)")
        for c in b["cells"]:
            if c["card_sha"] != card["card_sha"]:
                pr.append(f"{c['cell_id']}: card sha differs from card.json")
            pr += [f"schema: {m}" for m in SC.validate_fixed_cell(c)[:3]]
            if (c["bs"], c["c"]) not in {tuple(x) for x in pb["cells"]}:
                pr.append(f"{c['cell_id']}: cell is not in the plan of its boot")
    if len({c["job_name"] for b in job["boots"] for c in b["cells"]}) > 1:
        pr.append("cell records of several Jobs in one raw directory")
    an = fixed_anchor(job)
    if an.get("problem"):
        pr.append(an["problem"])
    if an.get("flag"):
        fl.append(an["flag"])
    return pr, fl


def fixed_anchor(job):
    """Anchor cell at the Job's first and last boot: |ln(step_end / step_start)| above the registered bound -> ANCHOR_DRIFT."""
    w = {}
    for b in job["boots"]:
        if b["boot"].get("role") == "anchor":
            for c in b["cells"]:
                w[b["boot"].get("anchor")] = (cell_window(c), c.get("gpu_uuid"))
    if set(w) != {"start", "end"}:
        return {"problem": "anchor cell missing at the start or the end of the Job"}
    (s, g0), (e, g1) = w["start"], w["end"]
    if s["status"] != "VALID" or e["status"] != "VALID":
        return {"problem": f"anchor cell not VALID (start {s['status']}, end {e['status']})"}
    d = abs(math.log(e["step_ms"] / s["step_ms"]))
    out = {"start_ms": s["step_ms"], "end_ms": e["step_ms"], "abs_ln_ratio": d, "bound": CS.P("fb_anchor_drift_ln")}
    if g0 != g1:
        out["problem"] = "anchor start and end ran on different GPUs"
    if d > CS.P("fb_anchor_drift_ln"):
        out["flag"] = "ANCHOR_DRIFT"
    return out


def cell_rows(job):
    """One row per measured cell record of the measurement boots (anchor boots excluded), plus one row per planned cell that the
    banner rule refused (INFEASIBLE_BY_RULE) -- a refused cell is reported, never silently absent."""
    rows = []
    for b in job["boots"]:
        if b["boot"].get("role") != "meas":
            continue
        for c in b["cells"]:
            w = cell_window(c)
            w.update(boot_id=c["boot_id"], cap=c["cap"], mem=c["mem"], kv_tokens=c["banner"]["max_total_num_tokens"])
            if w["status"] == "VALID":
                w.update(traffic(w["model"], w["bs"], w["ctx_mean"], w["step_ms"]))
            rows.append(w)
        for f in b["feasibility"]:
            if not f["run"]:
                rows.append({"cell_id": None, "model": b["boot"]["model"], "bs": f["bs"], "c": f["c"], "round": b["boot"]["round"],
                             "status": "INFEASIBLE_BY_RULE", "detail": "; ".join(f["reasons"]), "labels": [], "boot_id": b["boot"]["boot_id"],
                             "cap": b["boot"]["cap"], "mem": None, "kv_tokens": f.get("kv_tokens")})
    return sorted(rows, key=lambda r: (r["model"], r["bs"], r["c"], r["round"]))


def cell_summary(rows):
    """Per (model, b, c): the VALID rounds' step times, their median, and the round agreement label."""
    by = {}
    for r in rows:
        by.setdefault((r["model"], r["bs"], r["c"]), []).append(r)
    out = []
    for (m, bs, c), rs in sorted(by.items()):
        v = [r for r in rs if r["status"] == "VALID"]
        steps = [r["step_ms"] for r in v]
        row = {"model": m, "bs": bs, "c": c, "n_rounds_valid": len(v), "statuses": sorted({r["status"] for r in rs}),
               "labels": sorted({x for r in rs for x in r["labels"]}), "step_ms": X.median(steps) if steps else None,
               "step_ms_by_round": {r["round"]: r["step_ms"] for r in v}}
        if steps:
            row.update(tokens_per_s=bs * 1000.0 / row["step_ms"], ctx_mean=sum(r["ctx_mean"] for r in v) / len(v),
                       **traffic(m, bs, sum(r["ctx_mean"] for r in v) / len(v), row["step_ms"]))
        if len(steps) >= 2:
            d = abs(math.log(max(steps) / min(steps)))
            row["round_abs_ln_ratio"] = d
            if d > CS.P("fb_anchor_drift_ln"):
                row["labels"].append("ROUND_DISAGREE")
        out.append(row)
    return out


def doubling_gain(summary):
    """tokens/s at batch b over tokens/s at b / 2, same model and context (2.0 = doubling the batch doubles the throughput)."""
    tp = {(r["model"], r["bs"], r["c"]): r.get("tokens_per_s") for r in summary}
    return [{"model": m, "c": c, "bs": bs, "gain": tp[(m, bs, c)] / tp[(m, bs // 2, c)]}
            for (m, bs, c) in sorted(tp) if bs > 1 and tp[(m, bs, c)] and tp.get((m, bs // 2, c))]


def round_agreement(res_a, res_b):
    """The two round Jobs side by side (one Job per round, user decision 2026-11-12).  Nothing is merged: each cell keeps its two step
    times and gets |ln(step_b / step_a)|; above the registered bound -> ROUND_DISAGREE.  Each Job's own anchor drift is carried along, so
    a disagreement can be read against the drift inside the Jobs.  Refuses two results of the same round or of the same Job."""
    ra, rb = res_a["rounds"], res_b["rounds"]
    if res_a["rehearsal"] or res_b["rehearsal"]:
        raise ValueError("round agreement is read on measurement Jobs, not rehearsals")
    if len(ra) != 1 or len(rb) != 1 or ra == rb or res_a["job"] == res_b["job"]:
        raise ValueError(f"round agreement needs two different Jobs of two different rounds; got rounds {ra} / {rb}")
    a = {(r["model"], r["bs"], r["c"]): r for r in res_a["verdict_rows"]}
    b = {(r["model"], r["bs"], r["c"]): r for r in res_b["verdict_rows"]}
    rows, bound = [], CS.P("fb_anchor_drift_ln")
    for k in sorted(set(a) | set(b)):
        x, y = (a.get(k) or {}).get("step_ms"), (b.get(k) or {}).get("step_ms")
        row = {"model": k[0], "bs": k[1], "c": k[2], "step_ms": {str(ra[0]): x, str(rb[0]): y}}
        if x is None or y is None:
            row["label"] = "ONE_ROUND_ONLY"
        else:
            row["abs_ln_ratio"] = abs(math.log(y / x))
            row["label"] = "ROUND_DISAGREE" if row["abs_ln_ratio"] > bound else "AGREE"
        rows.append(row)
    return {"jobs": [res_a["job"], res_b["job"]], "rounds": [ra[0], rb[0]], "bound": bound, "rows": rows, "merged": False,
            "anchor": {str(ra[0]): res_a["anchor"], str(rb[0]): res_b["anchor"]}, "flags": {str(ra[0]): res_a["flags"], str(rb[0]): res_b["flags"]},
            "n": {lab: sum(1 for r in rows if r["label"] == lab) for lab in ("AGREE", "ROUND_DISAGREE", "ONE_ROUND_ONLY")}}


class JobInvalid(MA.JobInvalid):
    pass


def analyze_fixed(raw):
    job = raw if isinstance(raw, dict) else load_fixed(raw)
    pr, fl = fixed_problems(job)
    if pr:
        raise JobInvalid(pr)
    rows = cell_rows(job)
    summ = cell_summary(rows)
    names = sorted({c["job_name"] for b in job["boots"] for c in b["cells"]})
    return {"job": names[0] if len(names) == 1 else names, "kind": CS.KIND_FIXED, "rehearsal": job["plan"]["rehearsal"], "flags": fl,
            "rounds": list(job["plan"]["rounds"]),
            "anchor": fixed_anchor(job), "cell_rows": rows, "verdict_rows": summ, "doubling_gain": doubling_gain(summ),
            "counts": {m: {"planned": sum(1 for b in job["plan"]["boots"] if b["role"] == "meas" and b["model"] == m for _ in b["cells"]),
                           "valid": sum(1 for r in rows if r["model"] == m and r["status"] == "VALID"),
                           "infeasible": sum(1 for r in rows if r["model"] == m and r["status"] == "INFEASIBLE_BY_RULE")} for m in CS.MODELS},
            "timing_source": CS.P("fb_timing"), "cross_job": "NOT_MERGED"}


FIXED_COLS = ("model", "bs", "c", "n_rounds_valid", "step_ms", "tokens_per_s", "ctx_mean", "bytes_per_step", "bytes_per_s", "largest_term",
              "roofline_ms", "step_over_roofline", "round_abs_ln_ratio", "statuses", "labels")


def write_fixed(res, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "analysis.json").write_text(json.dumps(res, indent=1, sort_keys=True, default=str) + "\n")
    with open(out / "cell_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(FIXED_COLS)
        for r in res["verdict_rows"]:
            w.writerow([";".join(map(str, r[c])) if isinstance(r.get(c), list) else r.get(c) for c in FIXED_COLS])
    return out / "cell_table.csv"


# ==================================================================== serving cap/mem comparison
def serving_pairs():
    """Registered one-factor pairs (hi, lo, factor): the mem pair at cap 48 and the cap steps at mem 0.75, each as (mem, cap)."""
    combos = list(CS.SERVING_COMBOS)
    out = []
    for i, a in enumerate(combos):
        for b in combos[i + 1:]:
            if a[1] == b[1] and a[0] != b[0]:
                out.append((max(a, b), min(a, b), "mem"))
            elif a[0] == b[0] and a[1] != b[1]:
                out.append((max(a, b, key=lambda x: x[1]), min(a, b, key=lambda x: x[1]), "cap"))
    return out


def _arm_of(job):
    """{(arm key, mem, cap): arm name} of the measurement boots."""
    return {(b["arm_def"]["params"]["key"], b["arm_def"]["params"]["mem"], b["arm_def"]["params"]["cap"]): b["arm"]
            for b in job["plan"]["boots"] if b["role"] == "meas"}


def banner_tokens(job):
    """{arm name: max_total_num_tokens of its measurement boot} read from the boot's server log (None when the banner is absent)."""
    out = {}
    for b in job["plan"]["boots"]:
        if b["role"] != "meas":
            continue
        log = Path(job["raw"]) / "boots" / b["boot_id"] / "server.log" if job.get("raw") else None
        m = re.findall(r"max_total_num_tokens=([0-9]+)", log.read_text(errors="replace")) if log and log.exists() else []
        out[b["arm"]] = int(m[-1]) if m else None
    return out


def pair_rows(job, tab=None):
    tab = tab or MA.rung_table(job)
    arm = _arm_of(job)
    rows = []
    for key in job["plan"]["arm_keys"]:
        for hi, lo, factor in serving_pairs():
            ah, al = arm[(key, hi[0], hi[1])], arm[(key, lo[0], lo[1])]
            for rt in MA.rates_of(job):
                lab, ninf = MA.read_label(tab[(ah, rt)]["grid"], tab[(al, rt)]["grid"])
                rows.append({"arm_key": key, "factor": factor, "hi": list(hi), "lo": list(lo), "rate": rt, "label": lab,
                             "n_informative_pairs": ninf, "att_hi_primary": tab[(ah, rt)]["grid"][MS.PRIMARY_SLO],
                             "att_lo_primary": tab[(al, rt)]["grid"][MS.PRIMARY_SLO],
                             "diff_pp_primary": tab[(ah, rt)]["grid"][MS.PRIMARY_SLO] - tab[(al, rt)]["grid"][MS.PRIMARY_SLO],
                             "good_rps_hi": tab[(ah, rt)]["good_rps"], "good_rps_lo": tab[(al, rt)]["good_rps"]})
    return rows


def analyze_serving(raw, clk_csv=None, thresholds=None):
    job = raw if isinstance(raw, dict) else MA.load_job(raw)
    pr, fl = MA.job_problems(job)
    if job["plan"] is not None and job["plan"].get("kind") != CS.KIND_SERVING:
        pr.append("not a serving cap/mem plan")
    if pr:
        raise MA.JobInvalid(pr)
    tab = MA.rung_table(job)
    rows = pair_rows(job, tab)
    mark = "REHEARSAL" if job["plan"]["rehearsal"] else ("NOT_JUDGEABLE_N" if any(f.startswith("N_LT") for f in fl) else "")
    for r in rows:
        r["label_read"] = r["label"]
        r["label"] = mark or r["label"]
        r["flags"] = list(fl)
    arm = _arm_of(job)
    kv = banner_tokens(job)
    combos = [{"arm_key": k, "mem": m, "cap": c, "arm": a, "max_total_num_tokens": kv.get(a), "capacity": MA.capacity(job, a)}
              for (k, m, c), a in sorted(arm.items())]
    res = {"job": MA.job_name(job), "kind": CS.KIND_SERVING, "rehearsal": job["plan"]["rehearsal"], "flags": fl, "reading_rule": MS.READING_RULE,
           "anchor": MA.anchor_check(job), "split_check": MA.split_check(job), "verdict_rows": rows, "combos": combos,
           "attainment_grid": {f"{a}|{rt}": {f"{t:g}/{i:g}": v for (t, i), v in c["grid"].items()} for (a, rt), c in sorted(tab.items())},
           "clock": MA.clock_report(job, clk_csv), "cross_job": "NOT_MERGED"}
    res["fix"] = fix_cap_mem(res, thresholds)
    return res


# ---- the fixing rule ----------------------------------------------------------------------------------------------------------------------
def _pair_direction(rows, min_rungs):
    """One arm, one pair: UP (the higher setting is better) / DOWN / NO_DIFFERENCE / MIXED from the per-rung reading-rule labels."""
    up = sum(1 for r in rows if r["label"] == "A>B")
    dn = sum(1 for r in rows if r["label"] == "A<B")
    if up and dn:
        return "MIXED"
    if up >= min_rungs:
        return "UP"
    if dn >= min_rungs:
        return "DOWN"
    return "NO_DIFFERENCE"


def _combine(dirs):
    """Across arms: one direction if no arm contradicts it; arms that disagree (or a MIXED arm) are not resolved by the rule."""
    s = set(dirs)
    if "MIXED" in s or {"UP", "DOWN"} <= s:
        return "ARM_OR_RUNG_DEPENDENT"
    return "UP" if "UP" in s else "DOWN" if "DOWN" in s else "NO_DIFFERENCE"


def fix_cap_mem(res, thresholds=None):
    """Serving comparison -> {"status", "mem", "cap", "trace"}.  status FIXED carries the values for mo_spec; every other status fixes
    nothing: NOT_JUDGEABLE (rehearsal / < 4 seeds / flagged Job), PENDING_THRESHOLD (a rule threshold is unregistered), USER_DECISION
    (the arms or rungs disagree, the chosen pair was not measured, or the banner pool is missing / below the registered floor).
      1. cap: ascending steps at mem 0.75 (48 -> 128 -> 256).  UP -> take the step and go on, DOWN -> stop, NO_DIFFERENCE -> cap_tie_break
         ("higher" takes the step, "lower" stops).
      2. mem: the pair (0.75 vs 0.6) at cap 48.  With cap 48: UP -> 0.75, DOWN -> 0.6, NO_DIFFERENCE -> mem_tie_break.  With a higher cap
         the mem is 0.75 (the only mem that cap was measured at) unless the pair says DOWN.
      3. the combination must have been measured: DOWN at cap 48 together with a higher cap is not resolved by the rule.
      4. the chosen combination's banner pool must give every arm >= min_kv_tokens_per_slot tokens per slot.
    A direction needs >= min_rungs_agree rungs with the same reading-rule label and none with the opposite one."""
    th = dict(CS.FIX_RULE, **(thresholds or {}))
    out = {"status": None, "mem": None, "cap": None, "trace": [], "thresholds": th}

    def done(status, why):
        out["status"] = status
        out["trace"].append(why)
        return out

    rows = res["verdict_rows"]
    if res["rehearsal"] or res["flags"] or any(r["label"] in ("REHEARSAL", "NOT_JUDGEABLE_N") for r in rows):
        return done("NOT_JUDGEABLE", f"rehearsal {res['rehearsal']}, flags {res['flags']}")
    pending = sorted(k for k, v in th.items() if MS.is_pending(v))
    if pending:
        return done("PENDING_THRESHOLD", f"unregistered thresholds: {pending}")
    keys = sorted({r["arm_key"] for r in rows})

    def direction(hi, lo):
        sel = {k: [r for r in rows if r["arm_key"] == k and r["hi"] == list(hi) and r["lo"] == list(lo)] for k in keys}
        if not keys or any(not v for v in sel.values()):
            out["trace"].append(f"{list(hi)} vs {list(lo)}: no rows for some arm")
            return "ARM_OR_RUNG_DEPENDENT"                       # a pair that was not measured is never read as "no difference"
        per = {k: _pair_direction(v, th["min_rungs_agree"]) for k, v in sel.items()}
        d = _combine(per.values())
        out["trace"].append(f"{list(hi)} vs {list(lo)}: {d} {per}")
        return d

    mems = sorted({m for m, _ in CS.SERVING_COMBOS})
    caps = sorted({c for m, c in CS.SERVING_COMBOS if m == mems[-1]})
    cap = caps[0]
    for nxt in caps[1:]:
        d = direction((mems[-1], nxt), (mems[-1], cap))
        if d == "ARM_OR_RUNG_DEPENDENT":
            return done("USER_DECISION", f"the cap step {cap} -> {nxt} differs by arm or by rung")
        if d == "UP" or (d == "NO_DIFFERENCE" and th["cap_tie_break"] == "higher"):
            cap = nxt
        else:
            break
    d = direction((mems[-1], caps[0]), (mems[0], caps[0]))
    if d == "ARM_OR_RUNG_DEPENDENT":
        return done("USER_DECISION", "the mem pair differs by arm or by rung")
    if cap == caps[0]:
        mem = mems[-1] if d == "UP" else mems[0] if d == "DOWN" else (mems[-1] if th["mem_tie_break"] == "higher" else mems[0])
    else:                               # a cap above the lowest was measured at the higher mem only
        mem = mems[0] if d == "DOWN" else mems[-1]
    out["mem"], out["cap"] = mem, cap
    if (mem, cap) not in CS.SERVING_COMBOS:
        return done("USER_DECISION", f"(mem {mem}, cap {cap}) was not measured: the lower mem is better at cap {caps[0]} but cap {cap} "
                                     f"exists only with mem {mems[-1]}")
    pools = {c["arm_key"]: c["max_total_num_tokens"] for c in res["combos"] if (c["mem"], c["cap"]) == (mem, cap)}
    if any(v is None for v in pools.values()):
        return done("USER_DECISION", f"no max_total_num_tokens banner for the chosen combination: {pools}")
    low = {k: v / cap for k, v in pools.items() if v / cap < th["min_kv_tokens_per_slot"]}
    if low:
        return done("USER_DECISION", f"banner pool per slot below {th['min_kv_tokens_per_slot']}: {low}")
    return done("FIXED", f"mem {mem}, cap {cap}")


def apply_fix(fix, fixed=None):
    """mo_spec fixed values with mem / cap taken from a FIXED result; anything else leaves the placeholders in place."""
    f = dict(fixed or MS.default_fixed())
    if fix.get("status") == "FIXED":
        f.update(mem=fix["mem"], cap=fix["cap"])
    return f


SERVING_COLS = ("arm_key", "factor", "hi", "lo", "rate", "label", "n_informative_pairs", "att_hi_primary", "att_lo_primary", "diff_pp_primary",
                "good_rps_hi", "good_rps_lo")


def write_serving(res, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "analysis.json").write_text(json.dumps(res, indent=1, sort_keys=True, default=str) + "\n")
    (out / "cap_mem_fix.json").write_text(json.dumps(res["fix"], indent=1, sort_keys=True, default=str) + "\n")
    with open(out / "pair_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(SERVING_COLS)
        for r in res["verdict_rows"]:
            w.writerow(["/".join(map(str, r[c])) if isinstance(r[c], list) else r[c] for c in SERVING_COLS])
    return out / "pair_table.csv"


def main(argv=None):
    """mo_cap_analyze.py fixed|serving RAW --out DIR [--clk CLK_CSV]   |   mo_cap_analyze.py rounds RAW_ROUND_A --other RAW_ROUND_B --out DIR"""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=("fixed", "serving", "rounds"))
    ap.add_argument("raw")
    ap.add_argument("--out", required=True)
    ap.add_argument("--clk")
    ap.add_argument("--other")
    a = ap.parse_args(argv)
    try:
        if a.part == "rounds":
            res = round_agreement(analyze_fixed(a.raw), analyze_fixed(a.other))
            Path(a.out).mkdir(parents=True, exist_ok=True)
            (Path(a.out) / "round_agreement.json").write_text(json.dumps(res, indent=1, sort_keys=True, default=str) + "\n")
            print(json.dumps(res["n"]))
        elif a.part == "fixed":
            print(write_fixed(analyze_fixed(a.raw), a.out))
        else:
            print(write_serving(analyze_serving(a.raw, a.clk), a.out))
    except MA.JobInvalid as e:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / "JOB_INVALID.json").write_text(json.dumps({"problems": e.problems}, indent=1) + "\n")
        print(f"JOB_INVALID: {e}")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
