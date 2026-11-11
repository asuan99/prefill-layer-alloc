#!/usr/bin/env python3
"""Analyzer of the motivation harness (2026-11-10).  CPU only; computes measurements and registered labels, makes no performance claim.

  * attainment per registered SLO grid (TTFT x ITL), computed POST HOC from the raw token stamps -- no arm takes an SLO as input
  * reading rule (mo_spec.READING_RULE): a comparison label needs sign agreement over the informative SLO pairs, else SLO_DEPENDENT
  * capacity estimators: PRIMARY = burst prefix rule (all seeds >= 0.90, ascending rungs, sb_runner.lambda_pick), SECONDARY = pooled
    attainment crossing 0.90 interpolated in log-rate, TERTIARY = rate of peak pooled goodput; all reported, none privileged in the table
  * per-class reporting (request class by input tokens) at the primary SLO
  * best-baseline-per-load: at every rung (and every SLO pair) the PD-mux bundle is compared with the BEST of the baseline arms
  * one Job = one evaluation unit: points of different Jobs are never merged (merge_points raises CrossJobMergeError); a Job must be a
    complete block (every arm x rung x seed) with a clean card, the registered order and an anchor that did not drift
"""
from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path

import mo_spec as MS
import mo_clock as CK
import mo_card as MC
import mo_schema as SC
import sb_runner as RU

V = RU.V
SLO_PAIRS = [(t, i) for t in MS.TTFT_GRID_MS for i in MS.ITL_GRID_MS]


class CrossJobMergeError(RuntimeError):
    pass


class JobInvalid(RuntimeError):
    def __init__(self, problems):
        super().__init__("; ".join(problems[:6]) + (f" (+{len(problems) - 6} more)" if len(problems) > 6 else ""))
        self.problems = problems


def _j(p):
    return json.loads(Path(p).read_text())


def load_job(raw):
    raw = Path(raw)
    runs = [_j(p) for p in sorted((raw / "runs").glob("*.json"))]
    boots = [_j(p) for p in sorted((raw / "boots").glob("*/boot.json"))]
    done = {}
    for p in sorted((raw / "boots").glob("*/DONE.json")):
        d = _j(p)
        done[d.get("boot_id") or p.parent.name] = d
    return {"raw": str(raw), "runs": runs, "boots": sorted(boots, key=lambda b: b.get("index", 0)), "done": done,
            "plan": _j(raw / "plan.json") if (raw / "plan.json").exists() else None,
            "card": _j(raw / "card.json") if (raw / "card.json").exists() else None,
            "aborted": _j(raw / "JOB_ABORTED.json") if (raw / "JOB_ABORTED.json").exists() else None}


def job_name(job):
    names = sorted({r["job_name"] for r in job["runs"]})
    return names[0] if len(names) == 1 else names


def merge_points(*jobs):
    """Points of different Jobs are never combined (substrate: node, driver, clocks differ per Job).  Always raises for >1 Job."""
    if len(jobs) > 1:
        raise CrossJobMergeError(f"refusing to merge points of {len(jobs)} Jobs ({[job_name(j) for j in jobs]}): one Job = one evaluation unit")
    return jobs[0]["runs"] if jobs else []


# ---- basics ----------------------------------------------------------------------------------------------------------------------
def reqs_of(run):
    return [r for s in run["segments"] for r in s["requests"]]


def attain(reqs, ttft, itl):
    return sum(1 for r in reqs if V.req_good(r, ttft, itl)) / len(reqs) if reqs else None


def attainment_grid(reqs):
    """{(ttft_ms, itl_ms): attainment} over the registered SLO grid, from the raw records."""
    return {p: attain(reqs, *p) for p in SLO_PAIRS}


def duration(runs):
    return sum(float(s["duration_s"]) for r in runs for s in r["segments"])         # gate 7: sum, never max


def grid_runs(job):
    return [r for r in job["runs"] if r["boot_role"] == "meas"]


def arms_of(job):
    return [b["arm"] for b in job["plan"]["boots"] if b["role"] == "meas"]


def role_of(job, arm):
    return next(b["arm_def"]["role"] for b in job["plan"]["boots"] if b["arm"] == arm)


def rates_of(job):
    return list(job["plan"]["grid"]["rates"])


def by_arm_rate(job):
    out = {}
    for r in grid_runs(job):
        out.setdefault((r["arm"], r["spec"]["rate"]), []).append(r)
    return out


# ---- Job validity ----------------------------------------------------------------------------------------------------------------
def job_problems(job):
    """-> (problems, flags).  problems are fatal (no verdict table); flags are reported next to every table."""
    pr, fl = [], []
    plan, card = job["plan"], job["card"]
    if plan is None or card is None:
        return ["plan.json / card.json missing"], fl
    names = sorted({r["job_name"] for r in job["runs"]})
    if len(names) > 1:
        pr.append(f"runs of several Jobs in one raw directory: {names}")
    if len({r["card_sha"] for r in job["runs"]}) > 1 or any(r["card_sha"] != card["card_sha"] for r in job["runs"]):
        pr.append("run records carry a card sha different from card.json")
    if not MC.card_sha_ok(card) or card["plan_sha"] != MS.sha(plan):
        pr.append("card.json is not the card of plan.json")
    if job["aborted"]:
        pr.append(f"Job aborted: {job['aborted'].get('status')}: {str(job['aborted'].get('reason'))[:200]}")
    for b in plan["boots"]:
        st = job["done"].get(b["boot_id"], {}).get("status")
        if st != "done":
            pr.append(f"boot {b['boot_id']} status {st}")
    observed = [b["arm"] for b in job["boots"]]
    if observed != card["boot_order"]:
        pr.append("realized boot order differs from the registered order")
    for r in job["runs"]:
        for m in SC.validate_run(r)[:3]:
            pr.append(f"schema: {m}")
    have = {r["run_id"]: r for r in job["runs"]}
    for b in plan["boots"]:
        for run in b["runs"]:
            r = have.get(run["run_id"])
            if r is None:
                pr.append(f"missing run {run['run_id']} (incomplete block)")
            elif len(reqs_of(r)) != run["n_requests"]:
                pr.append(f"run {run['run_id']}: {len(reqs_of(r))} requests != {run['n_requests']}")
            elif card["trace"].get(run["run_id"], {}).get("hash") != r["trace_hash"]:
                pr.append(f"run {run['run_id']}: trace hash differs from the card")
    # same (rate, seed) -> same trace hash across arms
    th = {}
    for r in grid_runs(job):
        th.setdefault((r["spec"]["rate"], r["seed"]), set()).add(r["trace_hash"])
    for k, v in th.items():
        if len(v) > 1:
            pr.append(f"trace hash differs between arms at rate/seed {k}")
    if not plan["rehearsal"] and len(plan["seeds"]) < MS.MIN_SEEDS:
        fl.append(f"N_LT_{MS.MIN_SEEDS}_SEEDS")
    an = anchor_check(job)
    if an.get("flag"):
        fl.append(an["flag"])
    sc = split_check(job)
    if sc.get("flag"):
        fl.append(sc["flag"])
    if an.get("problem"):
        pr.append(an["problem"])
    return pr, fl


def split_check(job):
    """Realized decode SM count of the bundle from the split-check boot (mo_realized).  Not a verdict input; a mismatch is a flag that
    travels with every row, because the bundle's D is then not the registered one."""
    runs = [r for r in job["runs"] if r["boot_role"] == "split_check"]
    if not any(b["role"] == "split_check" for b in job["plan"]["boots"]):
        return {"status": "NOT_PLANNED"}
    if len(runs) != 1:
        return {"status": "MISSING", "flag": "SPLIT_CHECK_MISSING"}
    rs = runs[0]["realized_split"]
    if not rs.get("available"):
        return {"status": "UNAVAILABLE", "flag": "SPLIT_CHECK_UNAVAILABLE", "detail": rs}
    if rs.get("matches_target") is None:
        return {"status": "NO_CORESIDENT_STEP", "flag": "SPLIT_CHECK_EMPTY", "detail": rs}
    return {"status": "MATCH" if rs["matches_target"] else "MISMATCH", "flag": None if rs["matches_target"] else "SPLIT_NOT_TARGET",
            "detail": rs}


def anchor_check(job):
    a = [r for r in job["runs"] if r["boot_role"] == "anchor"]
    st = [r for r in a if r["cell"] == "ANCHOR_start"]
    en = [r for r in a if r["cell"] == "ANCHOR_end"]
    if len(st) != 1 or len(en) != 1:
        return {"problem": "anchor cell missing at the start or the end of the Job"}
    if st[0]["trace_hash"] != en[0]["trace_hash"]:
        return {"problem": "anchor start/end traces differ"}
    a0, a1 = (attain(reqs_of(x), *MS.PRIMARY_SLO) for x in (st[0], en[0]))
    out = {"start": a0, "end": a1, "diff_pp": abs(a0 - a1), "gpu_start": st[0]["gpu_uuid"], "gpu_end": en[0]["gpu_uuid"]}
    if st[0]["gpu_uuid"] != en[0]["gpu_uuid"]:
        out["problem"] = "anchor start and end ran on different GPUs"
    if abs(a0 - a1) > MS.ANCHOR_DRIFT_PP:
        out["flag"] = "ANCHOR_DRIFT"
    return out


# ---- reading rule ----------------------------------------------------------------------------------------------------------------
def read_label(ga, gb):
    """Label of A versus B over the SLO grid (dicts {(ttft, itl): attainment}) by MS.READING_RULE."""
    pos = neg = inf = 0
    for p in SLO_PAIRS:
        a, b = ga[p], gb[p]
        if a is None or b is None or (a >= MS.SAT_HI and b >= MS.SAT_HI) or (a <= MS.SAT_LO and b <= MS.SAT_LO):
            continue
        inf += 1
        d = a - b
        pos += d > MS.ATT_MARGIN
        neg += d < -MS.ATT_MARGIN
    if inf < MS.READ_MIN_INFORMATIVE:
        return "UNINFORMATIVE", inf
    if pos / inf >= MS.READ_AGREE_FRAC and neg == 0:
        return "A>B", inf
    if neg / inf >= MS.READ_AGREE_FRAC and pos == 0:
        return "A<B", inf
    return "SLO_DEPENDENT", inf


# ---- capacity ----------------------------------------------------------------------------------------------------------------------
def capacity(job, arm):
    cells = by_arm_rate(job)
    rates = rates_of(job)
    pts, pooled = [], []
    for rt in rates:
        runs = cells.get((arm, rt), [])
        for r in runs:
            pts.append((rt, attain(reqs_of(r), *MS.PRIMARY_SLO)))
        reqs = [x for r in runs for x in reqs_of(r)]
        good = sum(1 for x in reqs if V.req_good(x, *MS.PRIMARY_SLO))
        pooled.append((rt, good / len(reqs) if reqs else None, good / duration(runs) if runs else None))
    lam, st = RU.lambda_pick(pts, MS.CAP_ATTAIN)
    sec = _interp(pooled)
    peak = max((p for p in pooled if p[2] is not None), key=lambda p: p[2], default=None)
    flag = None
    if lam is not None and sec["value"] is not None and not (lam <= sec["value"] <= (rates[rates.index(lam) + 1] if lam != rates[-1] else math.inf)):
        flag = "ESTIMATORS_DISAGREE_ON_BRACKET"
    return {"primary_burst": lam, "primary_status": st, "secondary_interp": sec, "peak_goodput_rate": peak[0] if peak else None,
            "peak_goodput_rps": peak[2] if peak else None, "flag": flag}


def _interp(pooled, level=MS.CAP_ATTAIN):
    pts = [(r, a) for r, a, _ in pooled if a is not None]
    if not pts or pts[0][1] < level:
        return {"value": None, "status": "BELOW_GRID"}
    for (r0, a0), (r1, a1) in zip(pts, pts[1:]):
        if a0 >= level > a1:
            f = (a0 - level) / (a0 - a1)
            return {"value": round(math.exp(math.log(r0) + f * (math.log(r1) - math.log(r0))), 6), "status": "BRACKETED"}
    return {"value": pts[-1][0], "status": "ABOVE_GRID_TOP"}


# ---- per-class / tables -------------------------------------------------------------------------------------------------------------
def per_class(job):
    rows = []
    for (arm, rt), runs in sorted(by_arm_rate(job).items()):
        reqs = [x for r in runs for x in reqs_of(r)]
        for cls in sorted({x["req_class"] for x in reqs}):
            c = [x for x in reqs if x["req_class"] == cls]
            tt = [x["ttft_ms"] for x in c if x.get("ttft_ms") is not None]
            ip = [V.percentile(x["token_itl_ms"], 0.95) for x in c if x.get("token_itl_ms")]
            rows.append({"arm": arm, "rate": rt, "class": cls, "n": len(c), "attainment": attain(c, *MS.PRIMARY_SLO),
                         "ttft_p50_ms": V.percentile(tt, 0.5), "ttft_p95_ms": V.percentile(tt, 0.95),
                         "req_itl_p95_median_ms": statistics.median(ip) if ip else None, "flag": "SMALL_N" if len(c) < MS.MIN_CLASS_N else ""})
    return rows


def rung_table(job):
    """{(arm, rate): {'grid': attainment grid pooled over seeds, 'good_rps': goodput at the primary SLO, 'seed_att': [..], 'n': n}}"""
    out = {}
    for (arm, rt), runs in by_arm_rate(job).items():
        reqs = [x for r in runs for x in reqs_of(r)]
        out[(arm, rt)] = {"grid": attainment_grid(reqs), "n": len(reqs), "seed_att": [attain(reqs_of(r), *MS.PRIMARY_SLO) for r in runs],
                          "good_rps": sum(1 for x in reqs if V.req_good(x, *MS.PRIMARY_SLO)) / duration(runs)}
    return out


def best_baseline_rows(job, rt_tab=None):
    """Per rung: the bundle versus the best baseline of every SLO pair (best-baseline-per-load).  Also versus each baseline at the primary."""
    tab = rt_tab or rung_table(job)
    base = [a for a in arms_of(job) if role_of(job, a) == "baseline"]
    bundle = next(a for a in arms_of(job) if role_of(job, a) == "bundle")
    rows = []
    for rt in rates_of(job):
        gb = tab[(bundle, rt)]["grid"]
        best_grid, ident = {}, {}
        for p in SLO_PAIRS:
            top = max(tab[(a, rt)]["grid"][p] for a in base)
            best_grid[p] = top
            ident[p] = sorted(a for a in base if tab[(a, rt)]["grid"][p] == top)
        lab, ninf = read_label(gb, best_grid)
        prim = MS.PRIMARY_SLO
        best_prim = ident[prim]
        best_rps = max(tab[(a, rt)]["good_rps"] for a in base)
        rows.append({"rate": rt, "bundle_att_primary": gb[prim], "best_baseline_att_primary": best_grid[prim], "best_baseline_primary": best_prim,
                     "diff_pp_primary": gb[prim] - best_grid[prim], "bundle_good_rps": tab[(bundle, rt)]["good_rps"], "best_baseline_good_rps": best_rps,
                     "label_vs_best_baseline": lab, "n_informative_pairs": ninf, "n_distinct_best_identities": len({tuple(v) for v in ident.values()}),
                     "vs_each_baseline_primary": {a: gb[prim] - tab[(a, rt)]["grid"][prim] for a in base}})
    return rows


def control_rows(job, rt_tab=None):
    """The two single-variable pairs (patch OFF vs primary baseline: tree only; piecewise ON vs primary baseline: one flag only)."""
    tab = rt_tab or rung_table(job)
    prim = next(b["arm"] for b in job["plan"]["boots"] if b["arm_def"]["key"] == MS.ANCHOR_KEY)
    rows = []
    for ctrl in (a for a in arms_of(job) if role_of(job, a).startswith("control")):
        for rt in rates_of(job):
            lab, ninf = read_label(tab[(ctrl, rt)]["grid"], tab[(prim, rt)]["grid"])
            rows.append({"control": ctrl, "baseline": prim, "rate": rt, "diff_pp_primary": tab[(ctrl, rt)]["grid"][MS.PRIMARY_SLO] - tab[(prim, rt)]["grid"][MS.PRIMARY_SLO],
                         "label": lab, "n_informative_pairs": ninf})
    return rows


def verdict_table(job):
    """Rows of the verdict table: one per rung (bundle vs best baseline) with the reading-rule label; NOT_JUDGEABLE_N below 4 seeds."""
    pr, fl = job_problems(job)
    if pr:
        raise JobInvalid(pr)
    tab = rung_table(job)
    rows = best_baseline_rows(job, tab)
    mark = "REHEARSAL" if job["plan"]["rehearsal"] else ("NOT_JUDGEABLE_N" if any(f.startswith("N_LT") for f in fl) else "")
    for r in rows:
        r["label"] = mark or r["label_vs_best_baseline"]
        r["flags"] = list(fl)
    return rows, tab, fl


def find_clk(raw, levels=6):
    """The Job's meta/clk.csv (scripts/vessl/job_entry.sh writes <RUN>/meta/clk.csv; raw dirs sit under <RUN>/results/...)."""
    d = Path(raw).resolve()
    for _ in range(levels):
        c = d / "meta" / "clk.csv"
        if c.exists():
            return c
        d = d.parent
    return None


def clock_report(job, clk_csv=None):
    """GPU clock-limit covariate per run and per arm (mo_clock); unavailable when the Job has no clk.csv (CPU tests, local runs)."""
    clk = Path(clk_csv) if clk_csv else (find_clk(job["raw"]) if job.get("raw") else None)
    if not clk or not clk.exists():
        return {"available": False, "reason": "no meta/clk.csv for this Job"}
    samples, bad = CK.read_clk(clk)
    rep = CK.report(job["runs"], samples)
    return {"available": True, "clk_csv": str(clk), "n_rows_skipped": bad, **rep}


def analyze_job(raw, clk_csv=None):
    job = raw if isinstance(raw, dict) else load_job(raw)
    rows, tab, fl = verdict_table(job)
    caps = {a: capacity(job, a) for a in arms_of(job)}
    return {"job": job_name(job), "rehearsal": job["plan"]["rehearsal"], "flags": fl, "reading_rule": MS.READING_RULE,
            "slo_grid": {"ttft_ms": list(MS.TTFT_GRID_MS), "itl_ms": list(MS.ITL_GRID_MS), "primary": list(MS.PRIMARY_SLO)},
            "anchor": anchor_check(job), "verdict_rows": rows, "capacity": caps, "per_class": per_class(job), "controls": control_rows(job, tab),
            "attainment_grid": {f"{arm}|{rt}": {f"{t:g}/{i:g}": v for (t, i), v in c["grid"].items()} for (arm, rt), c in sorted(tab.items())},
            "grid_role": job["plan"]["grid"]["role"], "cross_job": "NOT_MERGED", "clock": clock_report(job, clk_csv),
            "split_check": split_check(job)}


def analyze_jobs(raws):
    """Each Job is analysed alone; the result is a list.  There is no combined table."""
    return [analyze_job(r) for r in raws]


def write_outputs(res, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "analysis.json").write_text(json.dumps(res, indent=1, sort_keys=True, default=str) + "\n")
    cols = ["rate", "bundle_att_primary", "best_baseline_att_primary", "best_baseline_primary", "diff_pp_primary", "bundle_good_rps",
            "best_baseline_good_rps", "label", "n_informative_pairs", "n_distinct_best_identities"]
    with open(out / "verdict_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in res["verdict_rows"]:
            w.writerow([";".join(r[c]) if isinstance(r[c], list) else r[c] for c in cols])
    ck = res.get("clock") or {}
    if ck.get("available"):
        with open(out / "clock_by_arm.csv", "w", newline="") as f:
            rows = CK.table_rows(ck)
            w = csv.writer(f)
            w.writerow(["arm", "limited_frac", "sm_mhz_p50", "power_w_p50", "reasons_seen"])
            for r in rows:
                w.writerow([r["arm"], r["limited_frac"], r["sm_mhz_p50"], r["power_w_p50"], r["reasons_seen"]])
    return out / "verdict_table.csv"


def main(argv=None):
    """mo_analyze.py RAW --out DIR [--clk CLK_CSV]: one Job, written to DIR (analysis.json, verdict_table.csv, clock_by_arm.csv)."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("--out", required=True)
    ap.add_argument("--clk")
    a = ap.parse_args(argv)
    try:
        res = analyze_job(a.raw, a.clk)
    except JobInvalid as e:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / "JOB_INVALID.json").write_text(json.dumps({"problems": e.problems}, indent=1) + "\n")
        print(f"JOB_INVALID: {e}")
        return 3
    print(write_outputs(res, a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
