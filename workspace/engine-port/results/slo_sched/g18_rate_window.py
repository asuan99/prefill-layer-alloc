#!/usr/bin/env python3
"""gate #16 rate axis -- step (1): fix the rate by capacity margin and power.

GPU 0.  Re-reads G16's own committed round files; no new campaign, no new
estimand -- every per-boot number comes from `g16_analyze.boot_estimands`,
which is itself the pre-registered canonical-library call.

WHAT IS BEING DECIDED
    `DESIGN_G18_RATE_AXIS_2026-08-19.md` sec4 leaves one blank: the rate value,
    to be fixed by "power and cliff margin (GPU 0)" BEFORE the rules audit.
    This script measures the two constraints that bracket it, at the only two
    rates that exist in the data (LO = 3/s offered, HI = 12/s offered):

      UPPER  the rate must sit below the SLOWEST arm's capacity, or that arm is
             back in overload and goodput magnitude is ill-posed again (the very
             thing this axis is meant to escape).  Capacity is read off HI,
             where every arm is saturated.
      LOWER  the campaign must be able to SEE something: if every arm already
             passes at the chosen rate, the arm-to-arm goodput spread is zero
             and there is no magnitude to quote.  Measured as the arm spread at
             LO relative to its between-boot noise.

    If the two constraints cross, the window is empty and this axis is a NO-GO
    for the same structural reason E-B1 was -- which is one of the two answers
    the design's single audit question explicitly allows.

NOT a gate, NOT pre-registered, NOT audited, no performance claim.
"""
import json, math, statistics as st, sys, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import g16_analyze as G                                        # noqa: E402

PAT = re.compile(r"g16_(blk\d+)_(d\d+)_(boot\d+)_(\d+)_(LO|HI)\.jsonl$")
OFFERED = {"LO": 3.0, "HI": 12.0}
KEYS = ("throughput_req_s", "goodput_req_s", "joint_pass_pct", "ttft_pass_pct",
        "itl_p95_pass_pct", "band_mass_ttft", "band_mass_itl",
        "ttft_p95_ms", "token_itl_p95_ms", "M_itl", "M_ttft")


def collect():
    rows = []
    for p in sorted(Path(".").glob("g16_blk*_*.jsonl")):
        m = PAT.search(p.name)
        if not m:
            continue
        blk, arm, boot, job, phase = m.groups()
        e = G.boot_estimands(p)
        rows.append(dict(blk=blk, arm=arm, boot=boot, job=job, phase=phase,
                         **{k: e[k] for k in KEYS}))
    return rows


def agg(rows, phase, key):
    out = {}
    for arm in sorted({r["arm"] for r in rows}, key=lambda a: int(a[1:])):
        v = [r[key] for r in rows if r["arm"] == arm and r["phase"] == phase]
        if v:
            out[arm] = (st.fmean(v), st.stdev(v) if len(v) > 1 else 0.0, len(v))
    return out


def main():
    rows = collect()
    res = {"what": "gate #16 rate axis -- capacity/power window", "gpu_spend": 0,
           "n_boot_files": len(rows), "offered": OFFERED,
           "ttft_slo_ms": G.TTFT_SLO_MS, "itl_slo_ms": G.ITL_SLO_MS, "phases": {}}
    for phase in ("LO", "HI"):
        thr = agg(rows, phase, "throughput_req_s")
        good = agg(rows, phase, "goodput_req_s")
        jp = agg(rows, phase, "joint_pass_pct")
        bt = agg(rows, phase, "band_mass_ttft")
        bi = agg(rows, phase, "band_mass_itl")
        t95 = agg(rows, phase, "ttft_p95_ms")
        i95 = agg(rows, phase, "token_itl_p95_ms")
        arms = list(thr)
        gmeans = {a: good[a][0] for a in arms}
        gsds = {a: good[a][1] for a in arms}
        spread = max(gmeans.values()) - min(gmeans.values())
        pooled_sd = math.sqrt(st.fmean([s ** 2 for s in gsds.values()]))
        res["phases"][phase] = {
            "offered_req_s": OFFERED[phase],
            "per_arm": {a: {"throughput_mean": thr[a][0], "throughput_sd": thr[a][1],
                            "goodput_mean": good[a][0], "goodput_sd": good[a][1],
                            "n_boots": good[a][2],
                            "joint_pass_pct_mean": jp[a][0],
                            "band_mass_ttft_mean": bt[a][0],
                            "band_mass_itl_mean": bi[a][0],
                            "ttft_p95_ms_mean": t95[a][0],
                            "itl_p95_ms_mean": i95[a][0]} for a in arms},
            "capacity_proxy_min_arm": min(thr[a][0] for a in arms),
            "capacity_proxy_min_arm_name": min(arms, key=lambda a: thr[a][0]),
            "capacity_proxy_max_arm": max(thr[a][0] for a in arms),
            "goodput_arm_spread": spread,
            "goodput_pooled_between_boot_sd": pooled_sd,
            "spread_over_sd": (spread / pooled_sd) if pooled_sd else None,
            "offered_over_achieved": OFFERED[phase] / st.fmean([thr[a][0] for a in arms]),
        }
    lo, hi = res["phases"]["LO"], res["phases"]["HI"]
    # boots per arm to resolve the LO-sized spread at 80% power, two-sided 5%,
    # paired-by-block contrast between the two extreme arms.
    d = lo["goodput_arm_spread"]
    s = lo["goodput_pooled_between_boot_sd"]
    n_needed = (2 * (1.96 + 0.84) ** 2 * s ** 2 / d ** 2) if d > 0 else None
    res["window"] = {
        "upper_bound_req_s": hi["capacity_proxy_min_arm"],
        "upper_bound_arm": hi["capacity_proxy_min_arm_name"],
        "lower_bound_evidence": {
            "lo_goodput_arm_spread": d,
            "lo_pooled_between_boot_sd": s,
            "lo_spread_over_sd": lo["spread_over_sd"],
            "boots_per_arm_for_80pct_power_at_LO_sized_spread": n_needed,
            "n_boots_available_per_arm_in_G16": lo["per_arm"][
                lo["capacity_proxy_min_arm_name"]]["n_boots"],
        },
        "cliff_margin_LO": {a: {"band_mass_ttft": lo["per_arm"][a]["band_mass_ttft_mean"],
                                "band_mass_itl": lo["per_arm"][a]["band_mass_itl_mean"]}
                            for a in lo["per_arm"]},
    }
    print(json.dumps(res, indent=2))
    Path("G18_RATE_WINDOW_2026-08-20.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
