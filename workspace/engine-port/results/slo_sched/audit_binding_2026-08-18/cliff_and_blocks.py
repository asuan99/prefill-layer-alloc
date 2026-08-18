#!/usr/bin/env python3
"""(a) metric-cliff sensitivity of the pass-rate reading, (b) block confound.

Read-only.  For every boot we record the per-request p95 token-ITL distribution
and the TTFT distribution, then re-score the SIDE-OUTPUT table at ITL SLO in
{54,57,60,63,66} ms (= the pre-registered +-10% band of gate #6) and TTFT SLO in
{2700,3000,3300} ms, and report the per-block decomposition.
"""
import json, math, statistics, sys
from pathlib import Path

ROOT = Path("/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port")
SLO_DIR = ROOT / "results/slo_sched"
sys.path.insert(0, str(ROOT / "benchmarks"))
from pdmux_eval.analyze import load_bench_serving_rounds, percentile

ARMS = ["d16", "d24", "d34", "d44", "d54", "d64", "d74"]
JOBS = [("blk1", "884336"), ("blk2", "884410"), ("blk3", "884411"), ("blk4", "884412")]


def boot_vectors(path):
    reqs, dur = load_bench_serving_rounds(Path(path))
    ttft = [r.ttft_ms for r in reqs]
    itl95 = [percentile(r.token_itl_ms, 0.95) if r.token_itl_ms else math.inf
             for r in reqs]
    ntok = [len(r.token_itl_ms) for r in reqs]
    return ttft, itl95, ntok, dur


def main():
    data = {}
    for phase in ("HI", "LO"):
        for arm in ARMS:
            for blk, job in JOBS:
                p = SLO_DIR / f"g16_{blk}_{arm}_boot1_{job}_{phase}.jsonl"
                data[(phase, arm, blk)] = boot_vectors(p)

    out = {}

    # ---- (1) per-boot location vs pass, the cliff -------------------------
    loc = {}
    for phase in ("HI", "LO"):
        rows = {}
        for arm in ARMS:
            rows[arm] = []
            for blk, _ in JOBS:
                ttft, itl95, ntok, dur = data[(phase, arm, blk)]
                fin = [v for v in itl95 if math.isfinite(v)]
                rows[arm].append({
                    "block": blk,
                    "M_itl_median_ms": statistics.median(itl95),
                    "itl95_p25": percentile(fin, .25), "itl95_p75": percentile(fin, .75),
                    "itl_pass_pct": 100 * sum(v <= 60 for v in itl95) / len(itl95),
                    "M_ttft_median_ms": statistics.median(ttft),
                    "ttft_pass_pct": 100 * sum(v <= 3000 for v in ttft) / len(ttft),
                    "band_mass_itl_54_66": sum(54 <= v <= 66 for v in itl95) / len(itl95),
                    "mean_out_tokens": statistics.fmean(ntok),
                })
        loc[phase] = rows
    out["per_boot_location_vs_pass"] = loc

    # ---- (2) re-score the whole table on the +-10% ITL/TTFT ladder --------
    ladder = {}
    for phase in ("HI", "LO"):
        ladder[phase] = {}
        for islo in (54.0, 57.0, 58.5, 60.0, 61.5, 63.0, 66.0):
            for tslo in (2700.0, 3000.0, 3300.0):
                key = f"itl{islo:g}_ttft{tslo:g}"
                per_arm = {}
                for arm in ARMS:
                    t = i = j = 0.0
                    for blk, _ in JOBS:
                        ttft, itl95, _, _ = data[(phase, arm, blk)]
                        n = len(ttft)
                        t += 100 * sum(v <= tslo for v in ttft) / n / 4
                        i += 100 * sum(v <= islo for v in itl95) / n / 4
                        j += 100 * sum(a <= tslo and b <= islo
                                       for a, b in zip(ttft, itl95)) / n / 4
                    per_arm[arm] = {"ttft": t, "itl": i, "joint": j}
                argmax_joint = max(per_arm, key=lambda a: per_arm[a]["joint"])
                # 'which axis is lower' -- the main session's binding predicate
                lower = {a: ("ITL" if per_arm[a]["itl"] < per_arm[a]["ttft"] else "TTFT")
                         for a in ARMS}
                ladder[phase][key] = {"per_arm": per_arm,
                                      "argmax_joint": argmax_joint,
                                      "lower_axis": lower,
                                      "crossover_arm": next(
                                          (a for a in ARMS if lower[a] == "TTFT"), None)}
        out_key = ladder
    out["threshold_ladder"] = ladder

    # ---- (3) block decomposition of the arm band -------------------------
    blockdec = {}
    for phase in ("HI", "LO"):
        blockdec[phase] = {}
        for arm in ARMS:
            per_blk = {}
            for blk, _ in JOBS:
                ttft, itl95, _, _ = data[(phase, arm, blk)]
                n = len(ttft)
                per_blk[blk] = {
                    "ttft": 100 * sum(v <= 3000 for v in ttft) / n,
                    "itl": 100 * sum(v <= 60 for v in itl95) / n,
                    "joint": 100 * sum(a <= 3000 and b <= 60
                                       for a, b in zip(ttft, itl95)) / n,
                }
            vals = {k: [per_blk[b][k] for b, _ in JOBS] for k in ("ttft", "itl", "joint")}
            drop1 = {k: statistics.fmean(vals[k][1:]) for k in vals}  # blk1 excluded
            blockdec[phase][arm] = {"per_block": per_blk,
                                    "mean_all4": {k: statistics.fmean(vals[k]) for k in vals},
                                    "mean_excl_blk1": drop1}
        blockdec[phase]["_ranking"] = {
            "itl_all4": sorted(ARMS, key=lambda a: -blockdec[phase][a]["mean_all4"]["itl"]),
            "itl_excl_blk1": sorted(ARMS, key=lambda a: -blockdec[phase][a]["mean_excl_blk1"]["itl"]),
            "joint_all4": sorted(ARMS, key=lambda a: -blockdec[phase][a]["mean_all4"]["joint"]),
            "joint_excl_blk1": sorted(ARMS, key=lambda a: -blockdec[phase][a]["mean_excl_blk1"]["joint"]),
        }
    out["block_decomposition"] = blockdec

    # ---- (4) paired block-level contrasts among d34..d64 ------------------
    def paired(phase, a1, a2, key):
        diffs = []
        for blk, _ in JOBS:
            t1, i1, _, _ = data[(phase, a1, blk)]
            t2, i2, _, _ = data[(phase, a2, blk)]
            f = lambda t, i: (100 * sum(x <= 3000 and y <= 60 for x, y in zip(t, i)) / len(t)
                              if key == "joint" else
                              100 * sum(x <= 3000 for x in t) / len(t) if key == "ttft" else
                              100 * sum(y <= 60 for y in i) / len(i))
            diffs.append(f(t1, i1) - f(t2, i2))
        m, s = statistics.fmean(diffs), statistics.stdev(diffs)
        h = 3.182 * s / math.sqrt(4)  # t(3) .975
        return {"diffs": diffs, "mean": m, "t3_ci": [m - h, m + h],
                "excludes_zero": (m - h) * (m + h) > 0}
    pc = {}
    for phase in ("HI",):
        pc[phase] = {}
        for a1, a2 in [("d44", "d34"), ("d44", "d54"), ("d44", "d64"), ("d64", "d34"),
                       ("d44", "d74"), ("d34", "d24")]:
            for key in ("joint", "ttft", "itl"):
                pc[phase][f"{a1}-{a2}:{key}"] = paired(phase, a1, a2, key)
    out["paired_block_contrasts"] = pc

    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
