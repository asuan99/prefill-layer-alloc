#!/usr/bin/env python3
"""E-B1 audit, DECISIVE TEST: can ANY arrival rate make a cell adjudicable at
(3000 ms, 60 ms)?  Answer from existing G16 artifacts, zero GPU.

Structure of the substrate (all measured, see ceiling.py / srv.log):
  * `max_running_requests = 48` in all 28 boots, and the HI (12 req/s) rounds sit
    at #running-req = 46..48 for most decode steps  =>  the decode batch is HARD
    CAPPED and HI already pegs it.  No higher arrival rate can raise the decode
    batch, hence none can raise the ITL any further; it only lengthens the queue
    (TTFT).  So the HI ITL distribution is the SATURATED (worst-case) one.
  * The saturated per-request ITL-p95 is a narrow SPIKE (p50->p99 spans 1.5-5 ms,
    i.e. 3-8% of its location) while the gate-#6 band is +-10% (12 ms wide).

Model of a LOWER rate: a mixture.  A fraction `w` of requests experience the
saturated regime (the spike, whose SHAPE is the measured HI conditional), the
rest live in the uncongested mode (ITL ~ 12-21 ms, far below 54 ms: contributes
0 to both the band mass and the failure fraction).  Then

    1 - pI(w) = w * q ,   band_mass(w) = w * b
      q = P(X > 60 | saturated),  b = P(54 <= X <= 66 | saturated)

Adjudicable (design sec1 + gate #6) needs  1-pI >= 0.05  and  band <= 0.25:
    w >= 0.05/q      and      w <= 0.25/b        with w <= 1.
FEASIBLE  <=>  q >= 0.05  AND  b/q <= 5.
Both q and b are measured, and BOTH are invariant to w  ==>  the feasibility of
the whole rate axis is decided by two numbers per arm.  Read-only.
"""
import json, math, statistics, sys
from pathlib import Path

ROOT = Path("/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port")
SLO = ROOT / "results/slo_sched"
sys.path.insert(0, str(ROOT / "benchmarks"))
from pdmux_eval.analyze import load_bench_serving_rounds, percentile

ARMS = ["d16", "d24", "d34", "d44", "d54", "d64", "d74"]
JOBS = [("blk1", "884336"), ("blk2", "884410"), ("blk3", "884411"), ("blk4", "884412")]
SAT_CUT = 45.0     # separates the uncongested mode (<=21 ms) from the spike (>=58 ms)


def main():
    out = {"saturation_cut_ms": SAT_CUT, "arms": {}}
    for arm in ARMS:
        I, T = [], []
        per_boot_q = []
        for blk, job in JOBS:
            reqs, _ = load_bench_serving_rounds(SLO / f"g16_{blk}_{arm}_boot1_{job}_HI.jsonl")
            v = [percentile(r.token_itl_ms, .95) if r.token_itl_ms else math.inf for r in reqs]
            I += v
            T += [r.ttft_ms for r in reqs]
            per_boot_q.append(sum(x > 60 for x in v) / len(v))
        n = len(I)
        sat = [x for x in I if x >= SAT_CUT]
        w_HI = len(sat) / n
        q = sum(x > 60.0 for x in sat) / len(sat)
        b = sum(54.0 <= x <= 66.0 for x in sat) / len(sat)
        feas = (q >= 0.05) and (b / q <= 5.0) if q > 0 else False
        wlo = 0.05 / q if q > 0 else float("inf")
        whi = 0.25 / b if b > 0 else float("inf")
        out["arms"][arm] = {
            "w_at_HI(frac in saturated mode)": w_HI,
            "spike_median_ms": statistics.median(sat),
            "q = P(ITLp95>60 | saturated)": q,
            "b = P(54<=ITLp95<=66 | saturated)": b,
            "b_over_q": b / q if q else None,
            "need_w_at_least": wlo, "need_w_at_most": whi,
            "ITL_AXIS_FEASIBLE_AT_ANY_RATE": feas,
            "feasible_w_window": [wlo, min(1.0, whi)] if feas else None,
            "reason_if_infeasible": (
                None if feas else
                ("q<0.05: even at FULL saturation (w=1, i.e. the measured HI run) the "
                 f"ITL failure fraction is only {q*100:.2f}% < 5%, so the ITL marginal "
                 "cannot leave the [5,95] window at any rate"
                 if q < 0.05 else
                 f"b/q={b/q:.1f}>5: reaching a 5% ITL failure fraction requires w>={wlo:.3f}, "
                 f"at which the band mass is {wlo*b:.3f} > 0.25")),
            "per_boot_q": per_boot_q,
        }
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
