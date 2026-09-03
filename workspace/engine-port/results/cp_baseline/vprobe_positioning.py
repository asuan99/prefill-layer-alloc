#!/usr/bin/env python3
"""Candidate A: WHERE does the goodput estimator actually work on this trace?  GPU 0.

The V-probe found that the registered operating point (3000 ms / 60 ms) puts LO at
95-100% pass and HI at 5.3% pass, so the estimator is saturated at one end of the
trace and counting-noise-dominated at the other.  This sweeps the SLO pair over a
grid and reports, FOR EACH ARM SEPARATELY:

  * positioning -- the pass fraction (an indicator estimator is most sensitive and
    least count-noisy near 0.5, useless at 0 or 1)
  * the UNTREATED null Delta SD at that SLO point, from the same 4 within-job pairs
  * the resulting n=4 MDE

★NO CROSS-ARM QUANTITY IS COMPUTED, and that is the point rather than a courtesy:
choosing an operating point by looking at which arm wins there is the purest form
of the thing this track has been killed for five times.  The choice this script
supports is made on POSITIONING AND NOISE ONLY.  Cross-arm comparison stays
forbidden until a rev2 registers it (PREREG_VPROBE sec 0).

Correctness: the cached-array path is asserted equal to the canonical
`score_run` at the operating point before any sweeping happens, so this is a
speed-up of the canonical scorer, not a second implementation of it.

Usage:  PYTHONPATH=<repo>/workspace/engine-port/benchmarks python3 vprobe_positioning.py
"""
import importlib.util, json, math, statistics, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "benchmarks"))  # results/cp_baseline -> engine-port/benchmarks
from pdmux_eval.analyze import load_bench_serving_rounds, percentile  # noqa: E402

_o = importlib.util.spec_from_file_location(
    "oracle_reanalysis", HERE.parents[0] / "slo_sched" / "oracle_reanalysis_2026_08_16.py")
ORC = importlib.util.module_from_spec(_o)
sys.modules["oracle_reanalysis"] = ORC
_o.loader.exec_module(ORC)

TTFT_GRID = (500., 750., 1000., 1500., 2000., 3000., 4000., 6000.)
ITL_GRID = (40., 50., 60., 70., 80., 100.)
OP = (3000., 60.)
T975 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571}
MARGIN = 0.03


def cache(path):
    """(ttft_ms, per-request ITL p95, summed duration) -- loaded once per file."""
    reqs, dur = load_bench_serving_rounds(Path(path))
    return ([r.ttft_ms for r in reqs],
            [percentile(r.token_itl_ms, 0.95) for r in reqs], dur)


def goodput(c, ttft_slo, itl_slo):
    tt, ip, dur = c
    return sum(1 for t, v in zip(tt, ip) if t <= ttft_slo and v <= itl_slo) / dur


def passfrac(c, ttft_slo, itl_slo):
    tt, ip, _ = c
    return sum(1 for t, v in zip(tt, ip) if t <= ttft_slo and v <= itl_slo) / len(tt)


def main():
    cells = {}          # (arm, seed) -> {phase: [cache_b1, cache_b2]}
    for sj in sorted(HERE.glob("vprobe_*/vprobe_scores.json")):
        d = json.loads(sj.read_text())
        if d.get("boots_scored", 0) < 2:
            continue
        arm, seed = d["arm"], d["per_boot"][0]["seed"]
        rows = [json.loads(l) for l in (sj.parent / "vprobe_summary.jsonl").read_text().splitlines() if l.strip()]
        ok = [r for r in rows if r.get("boot_ok")]
        cells[(arm, seed)] = {"lo": [cache(r["lo"]) for r in ok],
                              "hi": [cache(r["hi"]) for r in ok]}

    # --- prove the cached path IS the canonical scorer, before using it --------
    (arm0, seed0), first = next(iter(cells.items()))
    sj = next(HERE.glob(f"vprobe_*/vprobe_scores.json"))
    rows = [json.loads(l) for l in (sj.parent / "vprobe_summary.jsonl").read_text().splitlines() if l.strip()]
    r0 = [r for r in rows if r.get("boot_ok")][0]
    for ph, key in (("lo", "lo"), ("hi", "hi")):
        canon = ORC.score_run(Path(r0[key]), *OP)
        mine = goodput(cache(r0[key]), *OP)
        assert abs(canon["slo_goodput_req_s"] - mine) < 1e-9, (ph, canon["slo_goodput_req_s"], mine)
    print("[ok] cached path == canonical score_run at the operating point "
          "(both phases, exact)\n")

    arms = sorted({a for a, _ in cells})
    out = {"_what_this_is": "SLO-positioning sweep. Per-arm pass fraction and untreated "
                            "null-Delta SD only. No cross-arm quantity is computed.",
           "ttft_grid": TTFT_GRID, "itl_grid": ITL_GRID, "margin": MARGIN, "per_arm": {}}

    for arm in arms:
        seeds = sorted(s for a, s in cells if a == arm)
        out["per_arm"][arm] = {"seeds": seeds, "points": {}}
        for phase in ("lo", "hi", "combined"):
            for T in TTFT_GRID:
                for I in ITL_GRID:
                    pf, deltas = [], []
                    for s in seeds:
                        c = cells[(arm, s)]
                        if phase == "combined":
                            g = [(goodput(c["lo"][b], T, I) * c["lo"][b][2]
                                  + goodput(c["hi"][b], T, I) * c["hi"][b][2])
                                 / (c["lo"][b][2] + c["hi"][b][2]) for b in (0, 1)]
                            pf += [(passfrac(c["lo"][b], T, I) * len(c["lo"][b][0])
                                    + passfrac(c["hi"][b], T, I) * len(c["hi"][b][0]))
                                   / (len(c["lo"][b][0]) + len(c["hi"][b][0])) for b in (0, 1)]
                        else:
                            g = [goodput(c[phase][b], T, I) for b in (0, 1)]
                            pf += [passfrac(c[phase][b], T, I) for b in (0, 1)]
                        deltas.append((g[1] - g[0]) / g[0] if g[0] > 0 else None)
                    good = [d for d in deltas if d is not None]
                    rec = {"pass_frac": statistics.fmean(pf), "n_delta": len(good)}
                    if len(good) >= 2:
                        sd = statistics.stdev(good)
                        t = T975.get(len(good))
                        rec["null_sd"] = sd
                        rec["half_width_n4"] = t * sd / math.sqrt(len(good)) if t else None
                        rec["equivalent_reachable"] = (rec["half_width_n4"] < MARGIN
                                                       if rec["half_width_n4"] else None)
                        rec["min_abs_delta_for_direction"] = (
                            MARGIN + rec["half_width_n4"] if rec["half_width_n4"] else None)
                    out["per_arm"][arm]["points"][f"{phase}|{T:.0f}|{I:.0f}"] = rec

    (HERE / "vprobe_positioning.json").write_text(json.dumps(out, indent=1))

    for arm in arms:
        P = out["per_arm"][arm]["points"]
        print("=" * 88)
        print(f"### {arm}   (within-arm only; no cross-arm quantity exists in this table)")
        for phase in ("lo", "hi", "combined"):
            print(f"\n  -- {phase.upper()} --   "
                  f"pass% / null-SD% / n=4 half-width%   (★ = equivalent reachable at m=3%)")
            print("        ITL:" + "".join(f"{I:>17.0f}" for I in ITL_GRID))
            for T in TTFT_GRID:
                row = f"  TTFT {T:>5.0f}:"
                for I in ITL_GRID:
                    r = P[f"{phase}|{T:.0f}|{I:.0f}"]
                    sd = r.get("null_sd")
                    hw = r.get("half_width_n4")
                    star = "*" if r.get("equivalent_reachable") else " "
                    row += (f"{r['pass_frac']*100:>6.1f}/{sd*100:>4.1f}/{hw*100:>4.1f}{star}"
                            if sd is not None else f"{r['pass_frac']*100:>6.1f}/  -- /  -- ")
                print(row)
    print(f"\nwrote {HERE / 'vprobe_positioning.json'}")


if __name__ == "__main__":
    main()
