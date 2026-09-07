#!/usr/bin/env python3
"""AF-1: fold the boots into the registered axes and emit the label.  GPU 0.

Everything here goes through `af1_predicates` and `af1_rule`.  No statistic is
computed twice, and none is computed in this file that the rule layer does not
already register -- a result document that prints a number no fold produced is
exactly the hole 6th-CP-2-audit lesson 6 named ("함수도 상수처럼 분류하라").

Outputs (pre-registration sec 3.2, numbered as that section numbers them):
  ALWAYS  planned/scored boots per arm · node per round · re-tokenised stratum
          boundaries · server-args echo · the `d44` partition readout
  5.      per-stratum per-arm ttft/ritl p50·p90·max and the BETWEEN-BOOT SD
          (the first measurement of this axis on this model)
  6.      strict and banded screens over the 5 candidates and all 36 ladder points,
          at BOTH strata
  7.      `margin_identifiable` on both legs, with the announcement required for
          every boundary it returns False on
  8.      a verdict on each of the five registered predictions

Usage: python3 af1_analyze.py --glob 'af1_*_*' --out RESULT_AF1_<date>.json
"""
import argparse, glob, json, os, statistics, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import af1_predicates as P
import af1_rule as R


def collect(dirs):
    """rounds[task] = {arm: boot.json}, plus the always-outputs."""
    rounds, always = {}, {"rounds": [], "boot_failures": []}
    for d in sorted(dirs):
        rj = Path(d) / "round.json"
        if not rj.exists():
            continue
        meta = json.load(open(rj))
        arms = {}
        for arm in P.ARMS:
            bf = Path(d) / f"boot_{arm}.json"
            if bf.exists():
                arms[arm] = json.load(open(bf))
        rounds[meta["task"]] = arms
        always["rounds"].append({
            "dir": os.path.basename(d), "task": meta["task"], "seed": meta["seed"],
            "node": meta["node"], "arms_scored": sorted(arms),
            "server_args": {a: (Path(d) / f"args_{a}.txt").read_text().strip()
                            for a in P.ARMS if (Path(d) / f"args_{a}.txt").exists()},
            "partition_readout": {a: (Path(d) / f"partition_{a}.txt").read_text().strip()
                                  for a in P.ARMS if (Path(d) / f"partition_{a}.txt").exists()},
        })
        bfl = Path(d) / "BOOT_FAILURES.txt"
        if bfl.exists():
            always["boot_failures"].append({"dir": os.path.basename(d),
                                            "detail": bfl.read_text().strip()})
    return rounds, always


def fold_strata(rounds):
    """{arm: {q: (ttft_floor, ritl_floor, ttft_sd, ritl_sd)}} via registered folds."""
    strata, detail, boots = {}, {}, {}
    for arm in P.ARMS:
        per_arm, det = {}, {}
        n = 0
        for q in P.REPORT_STRATUM_QS:
            t_boots, i_boots = [], []
            for task in sorted(rounds):
                b = rounds[task].get(arm)
                if not b:
                    continue
                rec = next((s for s in b["strata"] if s["q"] == q), None)
                if not rec:
                    continue
                if rec.get("boot_ttft_ms") is not None:
                    t_boots.append(rec["boot_ttft_ms"])
                if rec.get("boot_ritl_p95_ms") is not None:
                    i_boots.append(rec["boot_ritl_p95_ms"])
            n = max(n, len(t_boots))
            tf, isd = P.agg_across_boots(t_boots), P.boot_sd(i_boots)
            rf, tsd = P.agg_across_boots(i_boots), P.boot_sd(t_boots)
            per_arm[q] = (tf, rf, tsd, isd)
            det[str(q)] = {
                "n_boots": len(t_boots),
                "ttft": _desc(t_boots), "ritl_p95": _desc(i_boots),
                "ttft_floor_ms": tf, "ttft_between_boot_sd_ms": tsd,
                "ritl_floor_ms": rf, "ritl_between_boot_sd_ms": isd,
            }
        strata[arm], detail[arm], boots[arm] = per_arm, det, n
    return strata, detail, boots


def _desc(xs):
    if not xs:
        return None
    return {"p50": P.order_stat(xs, 0.50), "p90": P.order_stat(xs, 0.90),
            "max": max(xs), "n": len(xs)}


def screens(strata):
    """Output 6: strict and banded, both strata, candidates and all 36 points."""
    out = {}
    for q in (P.SCREEN_STRATUM_Q, P.SCREEN_STRATUM_Q_ALT):
        per_q = {}
        for banded in (False, True):
            b = P.arm_bounds(strata, banded=banded, q=q)
            per_q["banded" if banded else "strict"] = {
                "arm_bounds_ms": {a: list(v) for a, v in b.items()},
                "candidates": {n: P.admissible(t, i, b) for n, t, i in P.SURVEY_POINTS},
                "ladder": {f"{t}x{i}": P.admissible(t, i, b)
                           for t in P.LADDER_TTFT_MS for i in P.LADDER_ITL_MS},
                "anchor_class": P.anchor_class(b), "ladder_class": P.ladder_class(b),
            }
        per_q["anchor_axis"] = P.anchor_axis(strata, q=q)
        per_q["ladder_axis"] = P.ladder_axis(strata, q=q)
        per_q["neutrality_axis"] = P.neutrality_axis(strata, q=q)
        out[str(q)] = per_q
    return out


def margins(strata):
    """Output 7: identifiability of every boundary, on both legs, per arm."""
    out = {}
    for arm in P.ARMS:
        cell = strata.get(arm, {}).get(P.SCREEN_STRATUM_Q)
        if not cell:
            continue
        tf, rf, tsd, isd = cell
        per_arm = {}
        for leg, floor, sd in (("ttft", tf, tsd), ("itl", rf, isd)):
            bounds = P.screen_boundaries(leg)
            m = P.margin_in_bands(floor, sd, bounds)
            ok = P.margin_identifiable(floor, sd, bounds)
            nearest = min((b for b in bounds if floor is not None and b > floor),
                          default=None)
            per_arm[leg] = {"floor_ms": floor, "between_boot_sd_ms": sd,
                            "nearest_boundary_ms": nearest,
                            "margin_in_bands": m, "identifiable": ok,
                            "announcement": None if ok in (True, None) else
                            (f"이 스크린은 {arm}의 {leg} 다리에서 경계 {nearest} ms를 "
                             f"잡음과 구별하지 못했다 (margin {m} 밴드 < "
                             f"MARGIN_MIN_BAND_MULT {P.MARGIN_MIN_BAND_MULT}).")}
        out[arm] = per_arm
    return out


def predictions(strata, world, label):
    """Output 8: a verdict on each registered prediction of sec 5.1."""
    q = P.SCREEN_STRATUM_Q
    mfu = {}
    for arm in P.ARMS:
        cell = strata.get(arm, {}).get(q)
        sm = 64 if arm == "d44" else P.SM_TOTAL
        if cell and cell[0]:
            mfu[arm] = P.stratum_floor_ms(q, sm, 1.0) / cell[0]
    pruned = None
    b = P.arm_bounds(strata, banded=True, q=q)
    if b:
        pruned = sorted({t for t in P.LADDER_TTFT_MS
                         for i in P.LADDER_ITL_MS if P.admissible(t, i, b) is False})
    itl_floor = {a: (strata.get(a, {}).get(q) or [None] * 2)[1] for a in P.ARMS}
    return {
        "1_label": {"expected": "ANCHOR_REGISTERED_LADDER_PRUNED / a",
                    "observed": f"{label} / {R.fork_branch(label)}",
                    "holds": label == "ANCHOR_REGISTERED_LADDER_PRUNED"},
        "2_hardware_bound": {"status": "FALSIFIED at GPU 0 before the run "
                                       "(pre-registration sec 5.1); not re-tested here"},
        "3_mfu_threshold": {"threshold_pct": 53.6, "back_derived_mfu": mfu,
                            "holds": None if not mfu else
                            all(v < 0.536 for a, v in mfu.items() if a != "d44")},
        "4_prune_list": {"expected": [500.0], "observed": pruned,
                         "flip_below_mfu_pct": 36.2,
                         "holds": None if pruned is None else pruned == [500.0]},
        "5_itl_below_lowest_boundary": {
            "lowest_itl_boundary_ms": min(P.screen_boundaries("itl")),
            "observed_floors_ms": itl_floor,
            "holds": None if any(v is None for v in itl_floor.values()) else
            all(v < min(P.screen_boundaries("itl")) for v in itl_floor.values())},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="af1_*_*")
    ap.add_argument("--out", default="af1_result.json")
    a = ap.parse_args()
    dirs = [d for d in glob.glob(a.glob) if os.path.isdir(d)]
    if not dirs:
        sys.exit("no round directories matched %r" % a.glob)

    rounds, always = collect(dirs)
    strata, detail, boots = fold_strata(rounds)

    world = {
        "coverage": P.coverage_axis(boots),
        "neutrality": P.neutrality_axis(strata),
        "stratum": P.stratum_axis(strata),
        "anchor": P.anchor_axis(strata),
        "ladder": P.ladder_axis(strata),
    }
    try:
        label = R.label(world)
        gap = False
    except R.OutcomeGap as e:
        label, gap = "OUTCOME_GAP", str(e)
    branch = R.fork_branch(label) if not gap else "n/a"

    res = {
        "_what_this_is": "AF-1 uncontended-floor probe result. Concurrency 1. "
                         "Not a policy comparison, not goodput, not an arm ranking.",
        "always": always,
        "planned_boots": P.PLANNED_BOOTS, "min_boots": P.MIN_BOOTS,
        "scored_boots_per_arm": boots,
        "stratum_tokens_registered": {str(k): v for k, v in P.STRATUM_TOKENS.items()},
        "output_5_per_stratum": detail,
        "world": world, "world_key": "|".join(world[k] for k in R.AXIS_ORDER),
        "label": label, "fork_branch": branch, "outcome_gap": gap,
        "substantive": label in R.SUBSTANTIVE,
    }
    if world["anchor"] != "unmeasured":
        res["output_6_screens"] = screens(strata)
        res["output_7_margins"] = margins(strata)
        res["output_8_predictions"] = predictions(strata, world, label)
    json.dump(res, open(a.out, "w"), indent=2, ensure_ascii=False)

    print("=== AF-1 result ===")
    print("scored boots per arm :", boots, f"(planned {P.PLANNED_BOOTS}, min {P.MIN_BOOTS})")
    print("world key            :", res["world_key"])
    print("label                :", label, "->", branch,
          "(substantive)" if res["substantive"] else "(not a finding)")
    if "output_7_margins" in res:
        for arm, legs in res["output_7_margins"].items():
            for leg, m in legs.items():
                if m["announcement"]:
                    print("  !", m["announcement"])
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
