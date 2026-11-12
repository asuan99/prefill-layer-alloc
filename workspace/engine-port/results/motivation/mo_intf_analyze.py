#!/usr/bin/env python3
"""Analysis of the LooGLE interference cell (roadmap #dshift).  CPU only; computes measurements and registered labels.

Two levels, kept apart:
  analyze_block(raw)        ONE Job (one full block).  Validity (mo_analyze.job_problems), per-class table (attainment, TTFT, ITL by
                            mo_spec.CLASS_EDGES), the LooGLE-only cell (requests tagged phase "LONG"; DESCRIPTIVE ONLY, never a judgement
                            input), and the per-seed values the combination needs.  Every comparison inside a block is a PAIRED one:
                            with vs without in the same boot, D vs D on the same seed and trace in the same Job.
  combine_blocks([b1, b2])  The two blocks of one design.  It never sees a raw run and never pools raw points of two Jobs (the rule of
                            mo_analyze.merge_points stays as it is).  Each block contributes per-seed paired differences; the block is a
                            factor: estimate = mean of all differences, variance = WITHIN-block variance (sum of squares around each
                            block's own mean, df = N - B), so a level shift between the two Jobs cancels and cannot widen or bias the
                            interval.  Interval: Student t (primary, small n) and a block-stratified bootstrap (reported next to it).
Registered readings (all constants from mo_spec / mo_intf_spec; the reading rule is mo_analyze.read_label):
  (1) interference    same arm, same rate: attainment of the input<=512 class WITHOUT minus WITH LooGLE > ATT_MARGIN, read only at rates
                      where the WITHOUT member is >= SAT_HI, and the reading rule over the SLO grid says "without > with" in every block
  (2) D shift         inside a configuration (with / without): goodput of D over goodput of that configuration's best D, per seed;
                      D34 / D44 are BELOW when the ratio is under 1 - GOODPUT_LOSS and the t interval of the log ratio excludes 0
  (3) starved class   inside one D: a class whose attainment is below CAP_ATTAIN
  (4) no D helps      every D has at least one starved class
and, per rate, the row of the roadmap's "next action" table the readings fall on (ACTION_ROWS).
"""
from __future__ import annotations

import csv
import json
import math
import random
import statistics
from pathlib import Path

import mo_spec as MS
import mo_analyze as MA
import mo_intf_spec as IS
import sb_runner as RU

V = RU.V
MW = MS.MW
SHORT_CLASS = MS.request_class(1)                       # "in<=512": the class the interference reading uses
CLASSES = tuple(MS.request_class(e) for e in MS.CLASS_EDGES) + (MS.request_class(MS.CLASS_EDGES[-1] + 1),)
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
        13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069,
        24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042}      # two-sided 95 % Student t quantiles
BOOT_N = 2000
ACTION_ROWS = {      # roadmap #dshift "결과별 다음 행동", top to bottom
    1: "D34 within 3 % in both configurations and no class below 0.90 -> fold single-GPU dynamic policy, D34 fixed, go to multi-GPU",
    2: "D shift exists but f*x below 1.5 %p -> fold on a single GPU; use only as a reason for per-GPU D on multi-GPU",
    3: "a class is starved and some D resolves it -> fix that D; check only that it holds on a composition-change trace",
    4: "a class is starved and it is the same at every D -> an admission / ordering problem; do not use split re-tuning as the remedy",
    5: "every single D loses more than 3 % on one side, or different classes are starved at different D -> open the dynamic-room measurement",
}


# ==================================================================== one block
def itl_p95(x):
    """Request-internal ITL p95 (the sealed percentile), computed once per request record and kept on it."""
    if "_itl_p95" not in x:
        itl = x.get("token_itl_ms") or []
        x["_itl_p95"] = V.percentile(itl, 0.95) if itl else None
    return x["_itl_p95"]


def good(x, ttft_slo_ms, itl_slo_ms):
    """The sealed request rule (ph_verdict.req_good: no error, TTFT <= SLO, request-internal ITL p95 <= SLO) with the p95 cached -- a
    block holds ~10^5 requests and is read at 16 SLO pairs.  Equality with the sealed function is a test, not an assumption."""
    if x.get("error") or x.get("ttft_ms") is None or x["ttft_ms"] > ttft_slo_ms:
        return False
    p = itl_p95(x)
    return p is None or p <= itl_slo_ms


def attain(reqs, ttft, itl):
    return sum(1 for r in reqs if good(r, ttft, itl)) / len(reqs) if reqs else None


def attainment_grid(reqs):
    return {p: attain(reqs, *p) for p in MA.SLO_PAIRS}


def _key(r):
    return (r["spec"]["workload"], r["spec"]["rate"], r["seed"])


def split_checks(job):
    """One split-check boot per D: every one must report the registered D on its co-resident decode steps.  Never a table input."""
    out, flag = {}, None
    for b in job["plan"]["boots"]:
        if b["role"] != "split_check":
            continue
        d = b["arm_def"]["D"]
        runs = [r for r in job["runs"] if r["boot_id"] == b["boot_id"]]
        rs = runs[0]["realized_split"] if len(runs) == 1 else None
        if rs is None:
            st = "MISSING"
        elif not rs.get("available"):
            st = "UNAVAILABLE"
        elif rs.get("matches_target") is None:
            st = "NO_CORESIDENT_STEP"
        else:
            st = "MATCH" if rs["matches_target"] else "MISMATCH"
        out[f"D{d}"] = {"status": st, "detail": rs}
        if st != "MATCH" and flag is None:
            flag = f"SPLIT_CHECK_D{d}_{st}"
    if set(out) != {f"D{d}" for d in IS.D_GRID}:
        flag = flag or "SPLIT_CHECK_MISSING"
    return {"status": "MATCH" if flag is None else "NOT_ALL_MATCH", "flag": flag, "by_D": out}


def _arm_keys(job):
    """{arm name: (arm key, D or None)} of the measurement boots."""
    return {b["arm"]: (b["arm_def"].get("arm_key"), b["arm_def"].get("D")) for b in job["plan"]["boots"] if b["role"] == "meas"}


def _cells(job):
    out = {}
    for r in MA.grid_runs(job):
        out[(r["arm"], r["spec"]["rate"], bool(r["spec"]["with_long"]), r["seed"])] = r
    return out


def _stats(reqs):
    tt = [x["ttft_ms"] for x in reqs if x.get("ttft_ms") is not None]
    ip = [p for p in (itl_p95(x) for x in reqs) if p is not None]
    return {"n": len(reqs), "attainment": attain(reqs, *MS.PRIMARY_SLO), "ttft_p50_ms": V.percentile(tt, 0.5),
            "ttft_p95_ms": V.percentile(tt, 0.95), "req_itl_p95_median_ms": statistics.median(ip) if ip else None}


def analyze_block(raw, clk_csv=None):
    job = raw if isinstance(raw, dict) else MA.load_job(raw)
    pr, fl = MA.job_problems(job, cell_key=_key, split_fn=split_checks)
    if job["plan"] is not None and job["plan"].get("kind") != IS.KIND:
        pr.append("not an interference plan")
    if pr:
        raise MA.JobInvalid(pr)
    plan = job["plan"]
    arms, cells, rates, seeds = _arm_keys(job), _cells(job), MA.rates_of(job), plan["seeds"]
    seed_values, per_class, loogle, rows = [], [], [], []
    for arm, (akey, d) in arms.items():
        for rt in rates:
            pooled = {}
            for wl in (True, False):
                allr = []
                for s in seeds:
                    run = cells[(arm, rt, wl, s)]
                    reqs = MA.reqs_of(run)
                    allr += reqs
                    by = {c: [x for x in reqs if x["req_class"] == c] for c in CLASSES}
                    seed_values.append({"arm": arm, "arm_key": akey, "D": d, "rate": rt, "with_long": wl, "seed": s,
                                        "goodput_rps": sum(1 for x in reqs if good(x, *MS.PRIMARY_SLO)) / MA.duration([run]),
                                        "att": {c: attain(v, *MS.PRIMARY_SLO) for c, v in by.items()}, "n": {c: len(v) for c, v in by.items()},
                                        "att_all": attain(reqs, *MS.PRIMARY_SLO)})
                pooled[wl] = allr
                for c in CLASSES:
                    v = [x for x in allr if x["req_class"] == c]
                    per_class.append({"arm": arm, "arm_key": akey, "D": d, "rate": rt, "with_long": wl, "class": c, **_stats(v),
                                      "flag": "SMALL_N" if len(v) < MS.MIN_CLASS_N else ""})
                if wl:
                    lv = [x for x in allr if x["phase"] == MW.LONG_PHASE]
                    loogle.append({"arm": arm, "arm_key": akey, "D": d, "rate": rt, **_stats(lv), "descriptive_only": True})
            sw, swo = ([x for x in pooled[k] if x["req_class"] == SHORT_CLASS] for k in (True, False))
            gw, gwo = attainment_grid(sw), attainment_grid(swo)
            lab, ninf = MA.read_label(gwo, gw)                        # A = without, B = with: "A>B" = LooGLE lowers the short class
            rows.append({"arm": arm, "arm_key": akey, "D": d, "rate": rt, "class": SHORT_CLASS, "att_without": gwo[MS.PRIMARY_SLO],
                         "att_with": gw[MS.PRIMARY_SLO], "diff_primary": gwo[MS.PRIMARY_SLO] - gw[MS.PRIMARY_SLO], "label_read": lab,
                         "n_informative_pairs": ninf, "eligible": gwo[MS.PRIMARY_SLO] >= MS.SAT_HI})
    mark = "REHEARSAL" if plan["rehearsal"] else ("NOT_JUDGEABLE_N" if any(f.startswith("N_LT") for f in fl) else "")
    for r in rows:
        r["label"] = mark or r["label_read"]
        r["flags"] = list(fl)
    return {"job": MA.job_name(job), "plan_job": plan["job"], "kind": IS.KIND, "block": plan["block"], "seeds": list(seeds), "rates": list(rates),
            "design_sha": plan["design_sha"], "rehearsal": plan["rehearsal"], "flags": fl, "anchor": MA.anchor_check(job),
            "split_checks": split_checks(job), "verdict_rows": rows, "per_class": per_class, "loogle_only": loogle,
            "seed_values": seed_values, "reading_rule": MS.READING_RULE, "clock": MA.clock_report(job, clk_csv), "cross_job": "NOT_MERGED"}


# ==================================================================== block combination
def block_estimate(by_block, label=""):
    """by_block = [[per-seed value, ...], ...] (one list per block).  -> mean, within-block se, t interval, stratified bootstrap."""
    vals = [v for b in by_block for v in b]
    n, nb = len(vals), sum(1 for b in by_block if b)
    if n == 0:
        return {"n": 0, "mean": None}
    mean = sum(vals) / n
    out = {"n": n, "n_blocks": nb, "mean": mean, "block_means": [sum(b) / len(b) if b else None for b in by_block]}
    df = n - nb
    if df >= 1:
        ss = sum((v - sum(b) / len(b)) ** 2 for b in by_block if b for v in b)
        se = math.sqrt(ss / df / n)
        t = T975[min(df, 30)]
        out.update(df=df, se=se, t_lo=mean - t * se, t_hi=mean + t * se)
        rng = random.Random(MS.sha(["intf-bootstrap", label, [[round(v, 12) for v in b] for b in by_block]]))
        means = sorted(sum(rng.choice(b) for b in by_block if b for _ in b) / n for _ in range(BOOT_N))
        out.update(boot_lo=means[int(0.025 * BOOT_N)], boot_hi=means[int(0.975 * BOOT_N) - 1])
    bm = [m for m in out["block_means"] if m is not None]
    out["blocks_same_sign"] = len({m > 0 for m in bm if m != 0}) <= 1
    return out


def _sv(blocks, **match):
    """Per block: {seed: seed-value row} matching the given fields."""
    return [{r["seed"]: r for r in b["seed_values"] if all(r[k] == v for k, v in match.items())} for b in blocks]


def _paired(blocks, a, b, fn):
    """Per block: [fn(row_a, row_b) per seed present in both], dropping seeds where fn returns None."""
    out = []
    for sa, sb in zip(_sv(blocks, **a), _sv(blocks, **b)):
        out.append([v for v in (fn(sa[s], sb[s]) for s in sorted(set(sa) & set(sb))) if v is not None])
    return out


def combine_problems(blocks):
    pr = []
    if not blocks:
        return ["no block"]
    if any(b.get("kind") != IS.KIND for b in blocks):
        pr.append("a result is not an interference block")
    if len({b["design_sha"] for b in blocks}) != 1:
        pr.append("blocks of different designs (arms / rates / duration / context differ): they cannot be combined")
    jobs = [json.dumps([b["job"], b.get("plan_job")]) for b in blocks]      # the VESSL Job name + the plan's job name
    if len(set(jobs)) != len(jobs):
        pr.append("the same Job given twice")
    seeds = [s for b in blocks for s in b["seeds"]]
    if len(set(seeds)) != len(seeds):
        pr.append("blocks share a seed: the blocks must use disjoint registered seeds")
    if len({b["block"] for b in blocks}) != len(blocks):
        pr.append("two results of the same block number")
    if any(b["rehearsal"] for b in blocks):
        pr.append("a rehearsal is not a block")
    return pr


def combine_blocks(blocks):
    pr = combine_problems(blocks)
    if pr:
        raise MA.JobInvalid(pr)
    rates = blocks[0]["rates"]
    arms = sorted({(r["arm_key"], r["D"]) for r in blocks[0]["seed_values"]}, key=lambda x: (x[1] is None, x[1]))
    flags = sorted({f for b in blocks for f in b["flags"]})
    # (1) interference: without - with, input<=512 class, same arm / rate / seed
    interference = []
    for akey, d in arms:
        for rt in rates:
            vals = _paired(blocks, dict(arm_key=akey, rate=rt, with_long=False), dict(arm_key=akey, rate=rt, with_long=True),
                           lambda wo, w: None if wo["att"][SHORT_CLASS] is None or w["att"][SHORT_CLASS] is None
                           else wo["att"][SHORT_CLASS] - w["att"][SHORT_CLASS])
            est = block_estimate(vals, f"interference|{akey}|{rt}")
            rows = [next(r for r in b["verdict_rows"] if r["arm_key"] == akey and r["rate"] == rt) for b in blocks]
            eligible = all(r["eligible"] for r in rows)
            reads = [r["label_read"] for r in rows]
            if not eligible:
                lab = "NOT_ELIGIBLE"
            elif est["mean"] is not None and est["mean"] > MS.ATT_MARGIN:
                lab = "INTERFERENCE" if all(x == "A>B" for x in reads) else "READING_RULE_NOT_PASSED"
            else:
                lab = "NO_INTERFERENCE"
            interference.append({"arm_key": akey, "D": d, "rate": rt, "class": SHORT_CLASS, "label": lab, "eligible": eligible,
                                 "block_reads": reads, "att_without_by_block": [r["att_without"] for r in rows], **est})
    # (2) D shift: goodput of D over the configuration's best D, per seed
    dgrid = [d for _k, d in arms if d is not None]
    shift = []
    for wl in (True, False):
        for rt in rates:
            mean_g = {}
            for d in dgrid:
                g = [[r["goodput_rps"] for r in sv.values()] for sv in _sv(blocks, D=d, rate=rt, with_long=wl)]
                mean_g[d] = block_estimate(g)["mean"]
            best = max(dgrid, key=lambda d: (mean_g[d], -d))
            for d in dgrid:
                row = {"with_long": wl, "rate": rt, "D": d, "best_D": best, "mean_goodput_rps": mean_g[d], "judged": d in IS.JUDGED_D}
                if d == best:
                    row.update(label="BEST", ratio=1.0)
                else:
                    vals = _paired(blocks, dict(D=d, rate=rt, with_long=wl), dict(D=best, rate=rt, with_long=wl),
                                   lambda a, b: math.log(a["goodput_rps"] / b["goodput_rps"]) if a["goodput_rps"] > 0 and b["goodput_rps"] > 0 else None)
                    est = block_estimate(vals, f"shift|{wl}|{rt}|{d}")
                    row.update(est)
                    if est["mean"] is None or "t_hi" not in est:
                        row.update(label="UNDEFINED", ratio=None)
                    else:
                        row["ratio"] = math.exp(est["mean"])
                        row["label"] = "BELOW_BEST" if row["ratio"] < 1.0 - IS.GOODPUT_LOSS and est["t_hi"] < 0 else "WITHIN"
                shift.append(row)
    # (3) starved classes inside one D, (4) every D starved
    starved = []
    for d in dgrid:
        for wl in (True, False):
            for rt in rates:
                for c in CLASSES:
                    sv = _sv(blocks, D=d, rate=rt, with_long=wl)
                    vals = [[r["att"][c] for r in b.values() if r["att"][c] is not None] for b in sv]
                    est = block_estimate(vals, f"class|{d}|{wl}|{rt}|{c}")
                    nmin = min((r["n"][c] for b in sv for r in b.values()), default=0)
                    starved.append({"D": d, "with_long": wl, "rate": rt, "class": c, "min_n_per_run": nmin,
                                    "starved": est["mean"] is not None and est["mean"] < MS.CAP_ATTAIN,
                                    "flag": "SMALL_N" if nmin < MS.MIN_CLASS_N else "", **est})
    actions = []
    for rt in rates:
        s_by = {d: sorted({r["class"] for r in starved if r["D"] == d and r["rate"] == rt and r["starved"]}) for d in dgrid}
        loses = {d: sorted("with" if r["with_long"] else "without" for r in shift if r["D"] == d and r["rate"] == rt and r["label"] == "BELOW_BEST")
                 for d in dgrid}
        every_d_starved = all(s_by[d] for d in dgrid)
        if every_d_starved:
            row = 4 if len({tuple(v) for v in s_by.values()}) == 1 else 5
        elif any(s_by[d] for d in dgrid):
            row = 3
        elif not loses[34]:
            row = 1
        elif all(loses[d] for d in dgrid):
            row = 5
        else:
            row = 2
        a = {"rate": rt, "row": row, "action": ACTION_ROWS[row], "starved_by_D": {f"D{d}": v for d, v in s_by.items()},
             "below_best_by_D": {f"D{d}": v for d, v in loses.items()}, "every_D_has_a_starved_class": every_d_starved,
             "resolving_D": [d for d in dgrid if not s_by[d]] if row == 3 else None}
        if row == 2:      # f (time share of the long configuration) is not measured by this cell; the reader needs x and the f threshold
            x = max(1.0 - r["ratio"] for r in shift if r["D"] == 34 and r["rate"] == rt and r["label"] == "BELOW_BEST")
            a.update(x_D34_loss=x, f_threshold=0.015 / x, note="row 2 if f*x < 1.5 %p, row 5 otherwise; f is not measured here")
        actions.append(a)
    status = "JUDGED" if len(blocks) == 2 and not flags else ("ONE_BLOCK" if len(blocks) == 1 and not flags else "NOT_JUDGEABLE")
    return {"kind": IS.KIND + "/combined", "status": status, "jobs": [b["job"] for b in blocks], "blocks": [b["block"] for b in blocks],
            "seeds": [b["seeds"] for b in blocks], "design_sha": blocks[0]["design_sha"], "flags": flags, "interference": interference,
            "d_shift": shift, "starved": starved, "actions": actions, "action_rows": ACTION_ROWS,
            "loogle_only": [dict(r, job=b["job"]) for b in blocks for r in b["loogle_only"]],
            "method": "paired per-seed differences inside each Job; block = factor (within-block variance, df = N - B); Student t primary, "
                      "block-stratified bootstrap reported; raw points of the Jobs are not pooled",
            "constants": {"ATT_MARGIN": MS.ATT_MARGIN, "SAT_HI": MS.SAT_HI, "CAP_ATTAIN": MS.CAP_ATTAIN, "GOODPUT_LOSS": IS.GOODPUT_LOSS,
                          "MIN_CLASS_N": MS.MIN_CLASS_N, "primary_slo": list(MS.PRIMARY_SLO)}}


# ==================================================================== outputs
BLOCK_COLS = ("arm_key", "D", "rate", "with_long", "class", "n", "attainment", "ttft_p50_ms", "ttft_p95_ms", "req_itl_p95_median_ms", "flag")


def write_block(res, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "analysis.json").write_text(json.dumps(res, indent=1, sort_keys=True, default=str) + "\n")
    with open(out / "per_class.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(BLOCK_COLS)
        for r in res["per_class"]:
            w.writerow([r[c] for c in BLOCK_COLS])
    with open(out / "loogle_only.csv", "w", newline="") as f:
        cols = ("arm_key", "D", "rate", "n", "attainment", "ttft_p50_ms", "ttft_p95_ms", "req_itl_p95_median_ms")
        w = csv.writer(f)
        w.writerow(cols + ("descriptive_only",))
        for r in res["loogle_only"]:
            w.writerow([r[c] for c in cols] + [True])
    return out / "per_class.csv"


def write_combined(res, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "combined.json").write_text(json.dumps(res, indent=1, sort_keys=True, default=str) + "\n")
    with open(out / "actions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rate", "row", "action", "status"])
        for a in res["actions"]:
            w.writerow([a["rate"], a["row"], a["action"], res["status"]])
    return out / "actions.csv"


def main(argv=None):
    """mo_intf_analyze.py block RAW --out DIR [--clk CLK]   |   mo_intf_analyze.py combine RAW_BLOCK_1 RAW_BLOCK_2 --out DIR"""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("block", "combine"))
    ap.add_argument("raw", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--clk")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "block":
            print(write_block(analyze_block(a.raw[0], a.clk), a.out))
        else:
            print(write_combined(combine_blocks([analyze_block(r) for r in a.raw]), a.out))
    except MA.JobInvalid as e:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / "JOB_INVALID.json").write_text(json.dumps({"problems": e.problems}, indent=1) + "\n")
        print(f"JOB_INVALID: {e}")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
