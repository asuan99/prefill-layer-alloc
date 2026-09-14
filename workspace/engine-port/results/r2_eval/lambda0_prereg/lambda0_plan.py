#!/usr/bin/env python3
"""Deterministic measurement plan for campaign stage 0 (lambda*), rev4.

WHAT CHANGED IN rev3 AND WHY
----------------------------
rev2 was audited `NO-GO` (`VERDICT_lambda0_rev2_2026-09-13.md`).  13 of the 15
rev1 conditions were confirmed closed; the kill cause was introduced by rev2's
own D2 repair.  The estimator it registered is an IDENTITY in the run's timing:

    achieved_over_realized = achieved / realized_offered
                           = (N/(N-1)) * span / duration
                           = (N/(N-1)) * span / (span + L)      L = drain

with `span` = the arrival window (sum of the replayed inter-arrivals) and
`L = duration - span` = the time after the LAST arrival that the run still needs
to finish draining.  Verified on all 12 archived probe C cells: the identity is
exact, and `duration = max_i(arrival_i + ttft_i + sum itl_i)` reproduces the
recorded duration to <= 0.265 s.

CONSEQUENCE, AND THE REASON rev2 DIED: **an unsaturated cell does not read 1.**
It reads `kappa = (N/(N-1))/(1 + L/span)`.  rev2 validated the repair only on
(8192 in, 96 out) cells, where L/span is 0.7-1.7% and kappa ~ 1.00.  On
shape A = (256, 512) the drain is 511 tokens x ITL, and rev2 held the measurement
window fixed at 170 s, so kappa lands at 0.91-0.95 -- BELOW the registered
low-side threshold 0.95.  The low-side test silently became "is this arm's decode
ITL under ~20 ms?", and probe C measured 19.92-23.53 ms on this very arm (D44).
Re-derived independently here: every shape-A low rung of the rev2 plan has
kappa 0.913-0.946 under the registration's OWN step model, i.e. F3's true branch
was unreachable by construction (audit T1').

rev3 repairs it at the source:

  D16  kappa is PRE-COMPUTED per rung and printed before the run, and a rung may
       serve as the LOW SIDE only if `kappa >= KAPPA_MIN (= ACH_HI + 0.02)`.
       kappa is a CEILING, not a prediction of the reading: it is the largest
       value `achieved_over_realized` can take at that rung, reached when the
       queueing shortfall is zero.  A rung with kappa < 0.95 cannot pass the
       low-side test however far below capacity it sits.
  D17  the only knob that moves kappa is the measurement window, so the plan
       LENGTHENS the window on the low rungs until the ceiling clears -- and
       nothing else.  High rungs are robust to drain (their reading is ~0.5-0.7)
       and are left alone.
  D19  rev2's "at most one rung can fall in the (0.90,0.95) blind spot" theorem
       is WITHDRAWN.  It was an arithmetic fact about the MULT constants under
       the idealised ratio `min(1, lambda*/x)`; under the real estimator the two
       lowest rev2 rungs fall in the blind spot together (0.940, 0.929).

Everything remains a deterministic function of the measured `lambda_inf` plus
registered constants: ladder, per-rung window, per-cell N, both seeds, the kappa
table and the bracket window all fall out of the formulas below.  Nothing may be
chosen after seeing a result.

  ladder(shape)  x_k = round3(MULT[shape][k] * lambda_inf(shape))
  drain model    ITL solves the fixed point ITL = a + b * min(Bbar, 48),
                 Bbar = x * (out-1) * ITL          (Little's law)
                 Lhat = TTFT_FLOOR(shape) * TTFT_LOAD + (out-1) * ITL
                 a = 19.82 ms, b = 0.5174 ms/req   (registered regression)
                 Bbar > 48 -> the rung is PREDICTED SATURATED (high side)
  window         the two lowest rungs may be lengthened: T = the smallest value
                 in (T_SHORT,) + T_LOW_GRID with kappa >= KAPPA_MIN
  N(shape,k)     clamp(round10(x_k * T_k), N_MIN, N_MAX_k)
  seeds          the two seeds in SEED_CANDIDATES minimising
                 max_over_cells |Ebar(seed, N) - 1|   (ties -> smaller seed)
  window(shape)  [ACH_HI * min(x_real), ACH_LO * max(x_real)]

`lambda_inf` is an UPPER bound probe, never the reported lambda*.  The adjacent
pre-registration (`results/r2_correctness/newpair_prereg/PREREG_NEWPAIR_2026-09-
13.md` sec 5-I3 "(D7)", NP-3') registers that I3 "fixes only the UPPER end of the
ladder ... using I3 for the lower end misses the bracket".  rev3 accepts that and
corrects on THIS side: x_min is lowered to 0.25*lambda_inf (rev2: 0.40) so the
window covers lambda*/lambda_inf down to 0.2375, and the assumption is replaced
by a DATA-CHECKED branch -- if the lowest rung comes back saturated the shape is
labelled `LADDER_TOO_HIGH` and the single registered redesign divides the ladder
by 4 (see the prereg sec 3.6).  No claim here depends on lambda_inf bounding
lambda* from below.

Ebar depends only on (seed, N), not on the rate: intervals are X_j / rate with
X_j ~ Exp(1), so Ebar = mean(X_j) is scale free.  ★`--random-range-ratio 1.0` is
a NECESSARY CONDITION for that replay (D20): `benchmark/datasets/common.py:56-64`
draws `np.random.randint` twice per run, and only a degenerate range (rr = 1.0)
consumes zero stream.  Verified by execution: rr=1.0 leaves the exponential
stream at [0.59970503, ...]; rr=0.9 moves it to [0.49437720, ...].

usage:
  lambda0_plan.py --lambda-inf-a A --lambda-inf-b B [--json]
  lambda0_plan.py --registered-scenarios     # the table embedded in the prereg
  lambda0_plan.py --selftest
"""
from __future__ import annotations

import argparse
import json
import math
import sys

import numpy as np

# ---------------------------------------------------------------- constants
# Registered 2026-09-13 (rev3), before any stage-0 GPU time.  Changing any of
# these after a cell has been measured invalidates the pre-registration.
MULT = {
    # rev3: x_min 0.40 -> 0.25 (newpair (D7): lambda_inf must not be trusted to
    # bound lambda* from below).  min step 1.53x, span 8.0x.
    "A": (0.25, 0.50, 0.85, 1.30, 2.00),
    "B": (0.45, 0.70, 1.15, 1.70),          # min step 1.478x, span 3.8x
}
SHAPE_IO = {"A": (256, 512), "B": (8192, 64)}
T_SHORT_S = 170.0                       # default measured window
T_LOW_GRID = tuple(range(600, 1801, 100))   # D17 window-extension candidates
N_EXTENDABLE = 2                        # only the two lowest rungs (audit D17)
N_MIN, N_MAX_SHORT, N_MAX_LONG = 40, 400, 2000
SEED_CANDIDATES = tuple(range(1, 5001))
ACH_HI, ACH_LO = 0.95, 0.90             # inherited verbatim from probe C R1
KAPPA_MIN = ACH_HI + 0.02               # D16: low-side eligibility ceiling
EBAR_TOL = 0.025                        # |Ebar-1| guard (audit option (c) band
                                        # is 1/Ebar in [0.95,1.05]; the seed
                                        # SELECTION achieves half that width)
DRAIN_MODEL_TOL = 2.0                   # measured L > TOL * Lhat -> UNRESOLVED
SEED_REPEAT_RUNG = -1                   # F4 repeats the TOP rung of each shape
SEED_REPEAT_TOL = 0.05                  # F4 prediction: lambda* within 5%

# Decode step model.  Registered regression over `srv_d44_*.log` B=2..7 from the
# rev1 audit sec 3-3: step(B) = 19.82 + 0.5174*B ms.  Self-consistency check in
# --selftest: 48/((512-1)*step(48)) = 2.10 req/s reproduces the registered point
# estimate for lambda*(A), so the drain model and the capacity estimate are the
# same model rather than two unrelated constants (which is what rev2 held).
STEP_A_S = 19.82e-3
STEP_B_S_PER_REQ = 0.5174e-3
MAX_RUNNING = 48                        # mamba cache = max_running_requests

# P2 job 905712, ctx 8192, concurrency 1, uncontended.  Same registered
# constants `c_capacity_analyze.py:43-44` uses.
FLOOR_REF_8192_S = 0.9104067257139832
ITL_SOLO_S = 0.01303689880296588
# probe C d44_r0 measured TTFT p50 2.84 s vs the 0.9104 s floor = 3.12x; rounded
# up.  Used only to make Lhat CONSERVATIVE (a larger Lhat lowers kappa).
TTFT_LOAD_FACTOR = 3.2
WARMUP_N = 8
BOOT_TEARDOWN_S = 110.0                 # job 905835 residual, see budget()
REQUESTED_BUDGET_GPU_H = 3.60           # E5: covers the 0.25x corner (worst
                                        # registered scenario), see budget_table

# Audit-recommended literal ladders (rev1 VERDICT sec 3-3).  Registered as the
# FALLBACK, used iff the objective predicate in sec 3.7 of the prereg selects it.
FALLBACK_LADDER = {
    "A": (1.1, 1.8, 3.0, 4.9, 8.0),
    "B": (0.45, 0.62, 0.85, 1.15),
}
LAMBDA_STAR_PRIOR = {"A": (2.1, 2.5), "B": (0.675, 0.675)}
LAMBDA_STAR_ABS = {"A": (1.08, 7.15), "B": (0.35, 1.20)}


def round3(x: float) -> float:
    """Round to 3 significant digits (deterministic ladder rendering)."""
    if x == 0:
        return 0.0
    return round(x, -int(math.floor(math.log10(abs(x)))) + 2)


def round10(n: float) -> int:
    return int(round(n / 10.0) * 10)


def ebar(seed: int, n: int) -> float:
    """Mean of the n-1 unit-rate exponential inter-arrivals bench_serving draws.

    Replicates `bench_serving.py:1706` + `:948`.  numpy's legacy RandomState
    draws one double per exponential variate, so an array of size n-1 is the
    same stream as n-1 scalar draws (asserted in selftest).
    """
    if n < 2:
        return 1.0
    np.random.seed(seed)
    return float(np.random.exponential(1.0, size=n - 1).mean())


def decode_itl(x: float, out_len: int):
    """Fixed point of the registered step model under Little's law.

    Bbar = x * (out-1) * ITL  and  ITL = a + b * Bbar, so
    ITL = a / (1 - b*x*(out-1)) while Bbar <= MAX_RUNNING.

    Returns (itl_s, batch) or (None, None) when the fixed point needs more than
    MAX_RUNNING concurrent decodes -- which is exactly the model saying the rung
    is ABOVE capacity, i.e. a high-side rung.
    """
    den = 1.0 - STEP_B_S_PER_REQ * x * (out_len - 1)
    if den <= 0:
        return None, None
    itl = STEP_A_S / den
    batch = x * (out_len - 1) * itl
    if batch > MAX_RUNNING:
        return None, None
    return itl, batch


def drain_hat(shape: str, x: float):
    """Predicted drain L = time from the LAST arrival to the last completion."""
    in_len, out_len = SHAPE_IO[shape]
    itl, batch = decode_itl(x, out_len)
    if itl is None:
        return None, None, None
    ttft = FLOOR_REF_8192_S * in_len / 8192.0 * TTFT_LOAD_FACTOR
    return ttft + (out_len - 1) * itl, itl, batch


def n_for(x: float, t_measure: float, n_max: int) -> int:
    return max(N_MIN, min(n_max, round10(x * t_measure)))


def cell_at(shape: str, x: float, t_measure: float, n_max: int) -> dict:
    """One candidate cell at a given window.  kappa is the CEILING on the
    reading, not a prediction of it."""
    n = n_for(x, t_measure, n_max)
    span = (n - 1) / x
    lhat, itl, batch = drain_hat(shape, x)
    rec = {"num_prompts": n, "span_pred_s": span, "T_measure_s": t_measure,
           "drain_pred_s": lhat, "itl_pred_s": itl, "batch_pred": batch}
    if lhat is None:                       # predicted saturated
        rec.update(kappa_pred=None, est_duration_s=n / x, predicted_saturated=True)
    else:
        rec.update(kappa_pred=(n / (n - 1)) * span / (span + lhat),
                   est_duration_s=span + lhat, predicted_saturated=False)
    return rec


def size_rung(shape: str, x: float, extendable: bool) -> dict:
    """D17: pick the SHORTEST registered window whose ceiling clears KAPPA_MIN.

    Only the two lowest rungs are extendable (the audit's recommendation), and a
    rung the model calls saturated is never extended -- it is a high-side rung
    and its reading (~0.5-0.7) is far from either threshold.
    """
    base = cell_at(shape, x, T_SHORT_S, N_MAX_SHORT)
    if base["predicted_saturated"] or not extendable:
        base["low_side_candidate"] = bool(
            (not base["predicted_saturated"])
            and base["kappa_pred"] is not None
            and base["kappa_pred"] >= KAPPA_MIN)
        return base
    for t in (T_SHORT_S,) + T_LOW_GRID:
        c = cell_at(shape, x, t, N_MAX_SHORT if t == T_SHORT_S else N_MAX_LONG)
        if c["kappa_pred"] is not None and c["kappa_pred"] >= KAPPA_MIN:
            c["low_side_candidate"] = True
            return c
    base["low_side_candidate"] = False
    base["window_unsatisfiable"] = True
    return base


def ladder_from_lambda_inf(shape: str, lam_inf: float) -> tuple:
    return tuple(round3(m * lam_inf) for m in MULT[shape])


def choose_seeds(ns) -> tuple:
    """The two best seeds by max |Ebar-1| over the cell N multiset."""
    scored = sorted((max(abs(ebar(s, n) - 1.0) for n in ns), s)
                    for s in SEED_CANDIDATES)
    return scored[0][1], scored[1][1], scored[0][0], scored[1][0]


def plan(lam_inf_a: float, lam_inf_b: float, ladders=None) -> dict:
    """The whole measurement plan.  Pure function of its arguments."""
    lam = {"A": lam_inf_a, "B": lam_inf_b}
    ladders = ladders or {s: ladder_from_lambda_inf(s, lam[s]) for s in ("A", "B")}

    sized = {}
    for s in ("A", "B"):
        sized[s] = [size_rung(s, x, k < N_EXTENDABLE)
                    for k, x in enumerate(ladders[s])]
    ns = [c["num_prompts"] for s in ("A", "B") for c in sized[s]]
    seed1, seed2, sc1, sc2 = choose_seeds(ns)

    out = {"rev": 4, "lambda_inf": lam, "seed1": seed1, "seed2": seed2,
           "seed1_max_abs_ebar_dev": sc1, "seed2_max_abs_ebar_dev": sc2,
           "kappa_min": KAPPA_MIN, "shapes": {}, "n_boot": 0}
    for s in ("A", "B"):
        cells = []
        for k, (x, sz) in enumerate(zip(ladders[s], sized[s])):
            eb = ebar(seed1, sz["num_prompts"])
            cells.append(dict(
                sz, name=f"{s.lower()}_r{k}", shape=s,
                input_len=SHAPE_IO[s][0], output_len=SHAPE_IO[s][1],
                offered_nominal=x, ebar_seed1=eb,
                offered_realized_pred=x / eb,
                headroom=(None if sz["kappa_pred"] is None
                          else sz["kappa_pred"] - ACH_HI)))
        x_real = [c["offered_realized_pred"] for c in cells]
        rep = cells[SEED_REPEAT_RUNG]
        out["shapes"][s] = {
            "ladder_nominal": list(ladders[s]),
            "cells": cells,
            "low_side_candidates": [c["name"] for c in cells if c["low_side_candidate"]],
            "n_low_side_candidates": sum(c["low_side_candidate"] for c in cells),
            "window_realized": [ACH_HI * min(x_real), ACH_LO * max(x_real)],
            "window_nominal": [ACH_HI * min(ladders[s]), ACH_LO * max(ladders[s])],
            "window_conservative": [max(ACH_HI * min(x_real), ACH_HI * min(ladders[s])),
                                    min(ACH_LO * max(x_real), ACH_LO * max(ladders[s]))],
            "prior_point": LAMBDA_STAR_PRIOR[s],
            "prior_absolute": LAMBDA_STAR_ABS[s],
            "seed_repeat_cell": rep["name"] + "_s2",
            "seed_repeat_ebar": ebar(seed2, rep["num_prompts"]),
            "max_abs_ebar_dev_seed1": max(abs(c["ebar_seed1"] - 1.0) for c in cells),
        }
        w = out["shapes"][s]["window_conservative"]
        out["shapes"][s]["coverage_band_ratio"] = [w[0] / lam[s], w[1] / lam[s]]
        out["shapes"][s]["covers_prior_point"] = all(
            w[0] <= p <= w[1] for p in LAMBDA_STAR_PRIOR[s])
        out["n_boot"] += len(cells) + 1
    out["budget"] = budget(out, 1.0)
    out["budget_table"] = budget_table(out)
    return out


def warmup_s(shape: str) -> float:
    """8 sequential warmup requests, discarded (probe C `:139-145` inherited)."""
    in_len, out_len = SHAPE_IO[shape]
    return WARMUP_N * (FLOOR_REF_8192_S * in_len / 8192.0 + (out_len - 1) * ITL_SOLO_S)


# ★rev4/E5: the coverage ratios the budget must be registered over.  A rung
# above capacity does NOT finish in `N/x`; it finishes in `N/lambda*`, so the
# cost of the whole stage is worst at the BOTTOM of the registered coverage
# band, where the ladder sits furthest above the true capacity.  rev3 priced
# every rung at `N/x` and therefore understated the low-lambda* corner by ~2.5x.
BUDGET_RATIO_GRID = (1.00, 0.50, 0.25)


def cell_duration_at(c: dict, lam_star: float) -> float:
    """Registered pricing: `max(span + drain, N/lambda*)`.  Same model the
    reachability map uses, so the budget and the verdict map cannot disagree."""
    n, span = c["num_prompts"], c["span_pred_s"]
    drain = c["drain_pred_s"]
    uncontended = span + (drain if drain is not None else 0.0)
    return max(uncontended, n / lam_star) if lam_star > 0 else uncontended


def budget(p: dict, ratio: float = 1.0) -> dict:
    """Total wall clock at a given true lambda* = ratio * lambda_inf."""
    bench = warm = 0.0
    for s in ("A", "B"):
        d = p["shapes"][s]
        lam_star = ratio * p["lambda_inf"][s]
        for c in d["cells"]:
            bench += cell_duration_at(c, lam_star)
            warm += warmup_s(s)
        bench += cell_duration_at(d["cells"][SEED_REPEAT_RUNG], lam_star)
        warm += warmup_s(s)
    boot = BOOT_TEARDOWN_S * p["n_boot"]
    tot = bench + warm + boot
    return {"ratio": ratio, "bench_s": bench, "warmup_s": warm,
            "boot_teardown_s": boot, "total_s": tot,
            "total_gpu_h": tot / 3600.0,
            "with_one_redesign_round_gpu_h": tot * 2 / 3600.0}


def budget_table(p: dict) -> dict:
    return {f"lambda_star_over_lambda_inf={r:.2f}": budget(p, r)
            for r in BUDGET_RATIO_GRID}


def _fmt_plan(p: dict, title: str) -> str:
    L = [f"### {title}",
         f"seed1={p['seed1']} (max|Ebar-1|={p['seed1_max_abs_ebar_dev']:.4f})  "
         f"seed2={p['seed2']} (max|Ebar-1|={p['seed2_max_abs_ebar_dev']:.4f})  "
         f"boots={p['n_boot']}  KAPPA_MIN={p['kappa_min']:.2f}"]
    for s in ("A", "B"):
        d = p["shapes"][s]
        L.append(f"  shape {s} {SHAPE_IO[s]}  lambda_inf={p['lambda_inf'][s]}")
        L.append("    cell     offered_nom     N    T_s  ITLhat  Bhat  Lhat   L/span"
                 "   kappa  lowside  est_dur_s")
        for c in d["cells"]:
            k = "  SAT " if c["kappa_pred"] is None else f"{c['kappa_pred']:.4f}"
            itl = "   -- " if c["itl_pred_s"] is None else f"{1000*c['itl_pred_s']:5.1f}"
            bb = "  -- " if c["batch_pred"] is None else f"{c['batch_pred']:4.0f}"
            lh = "   -- " if c["drain_pred_s"] is None else f"{c['drain_pred_s']:5.1f}"
            ls = ("   --  " if c["drain_pred_s"] is None
                  else f"{100*c['drain_pred_s']/c['span_pred_s']:5.2f}%")
            L.append(f"    {c['name']:8s} {c['offered_nominal']:10.4g} {c['num_prompts']:5d} "
                     f"{int(c['T_measure_s']):5d} {itl} {bb} {lh} {ls}  {k}  "
                     f"{'YES' if c['low_side_candidate'] else ' - ':>5s}  "
                     f"{c['est_duration_s']:9.1f}")
        L.append(f"    low-side candidates: {d['low_side_candidates']} "
                 f"(need >= 1; kappa >= {KAPPA_MIN:.2f})")
        L.append(f"    window(realized)     [{d['window_realized'][0]:.4g}, {d['window_realized'][1]:.4g}]")
        L.append(f"    window(nominal)      [{d['window_nominal'][0]:.4g}, {d['window_nominal'][1]:.4g}]")
        L.append(f"    window(conservative) [{d['window_conservative'][0]:.4g}, "
                 f"{d['window_conservative'][1]:.4g}] = lambda*/lambda_inf in "
                 f"[{d['coverage_band_ratio'][0]:.4g}, {d['coverage_band_ratio'][1]:.4g}]")
        L.append(f"    prior point {d['prior_point']} inside: {d['covers_prior_point']}"
                 f"   prior absolute {d['prior_absolute']} inside: "
                 f"{d['prior_absolute'][0] >= d['window_conservative'][0] and d['prior_absolute'][1] <= d['window_conservative'][1]}"
                 f"   (reported, NOT asserted)")
        L.append(f"    F4 repeat cell: {d['seed_repeat_cell']} (Ebar@seed2={d['seed_repeat_ebar']:.4f})")
    b = p["budget"]
    L.append(f"  BUDGET @lambda*=lambda_inf: bench {b['bench_s']/60:.1f} min + warmup "
             f"{b['warmup_s']/60:.1f} min + boot/teardown {b['boot_teardown_s']/60:.1f} min"
             f" = {b['total_gpu_h']:.3f} GPU-h")
    L.append("  BUDGET vs true capacity (E5: saturated rungs cost N/lambda*, not N/x):"
             + "".join(f"  {r:.2f}x -> {budget(p, r)['total_gpu_h']:.3f} GPU-h"
                       for r in BUDGET_RATIO_GRID))
    return "\n".join(L)


# ★MEASURED (job 907959, 2026-09-13, 0.356 GPU-h): I3a (256,512) achieved
# 3.0939 req/s and I3b (8192,64) achieved 0.6956 req/s at --request-rate inf
# --max-concurrency 64, with I3_max_running_req = 48 (the engine cap was
# reached, newpair F5).  This is the scenario that will actually run; the rest
# of the grid stays registered so the plan remains published across the whole
# plausible range (gate #197).
LAMBDA_INF_MEASURED = (3.0939, 0.6956)
REGISTERED_GRID = (
    ("MEASURED job 907959 (I3a 3.0939 / I3b 0.6956 req/s)", 3.0939, 0.6956),
    ("point estimate lower  (lambda*(A)=2.1 regression extrapolation)", 2.10, 0.675),
    ("point estimate upper  (lambda*(A)=2.5 ctx-768 KV-term variant)", 2.51, 0.80),
    ("absolute lower bound  (A=1.08 worst observed step, B=0.35)", 1.08, 0.35),
    ("absolute upper bound  (A=7.15 step(B=1) floor, B=1.20)", 7.15, 1.20),
    ("B=6 plateau variant   (A=3.91, B=1.00)", 3.91, 1.00),
)


def registered_scenarios() -> str:
    """Every number this plan can emit, printed BEFORE any stage-0 GPU time."""
    out = []
    for title, a, b in REGISTERED_GRID:
        out.append(_fmt_plan(plan(a, b), f"lambda_inf-anchored: {title}"))
        out.append("")
    out.append(_fmt_plan(
        plan(2.10, 0.675, ladders={k: tuple(v) for k, v in FALLBACK_LADDER.items()}),
        "FALLBACK (audit-recommended literal ladders; selected only by the "
        "objective predicate in prereg sec 3.7)"))
    return "\n".join(out)


def selftest() -> None:
    # 1. scalar draws == array draws (the replication assumption of ebar()).
    np.random.seed(41)
    a = [np.random.exponential(1.0 / 0.86) for _ in range(7)]
    np.random.seed(41)
    b = list(np.random.exponential(1.0 / 0.86, size=7))
    assert np.allclose(a, b), "scalar/array exponential streams differ"

    # 2. Ebar is scale free, and matches probe C seed 41.
    np.random.seed(41)
    i1 = np.random.exponential(1.0 / 0.86, size=99).mean() * 0.86
    np.random.seed(41)
    i2 = np.random.exponential(1.0 / 3.30, size=99).mean() * 3.30
    assert abs(i1 - i2) < 1e-12 and abs(ebar(41, 100) - i1) < 1e-12
    for n, want in ((110, 0.8496), (170, 0.8524), (80, 0.8715), (130, 0.8487),
                    (120, 0.8259), (40, 0.8560), (45, 0.8270), (100, 0.8647)):
        assert abs(ebar(41, n) - want) < 5e-5, (n, ebar(41, n), want)

    # 3. ★D20: --random-range-ratio 1.0 is a NECESSARY condition for (2).
    #    `benchmark/datasets/common.py:56-64` draws randint twice per run; only
    #    a degenerate range consumes nothing.  Executed, not asserted in prose.
    def stream_after(rr):
        np.random.seed(1376)
        for full in (256, 512):
            np.random.randint(max(int(full * rr), 1), full + 1, size=140)
        return np.random.exponential(1.0, size=3)
    np.random.seed(1376)
    undisturbed = np.random.exponential(1.0, size=3)
    assert np.allclose(stream_after(1.0), undisturbed), "rr=1.0 must not consume"
    assert not np.allclose(stream_after(0.9), undisturbed), "rr<1 MUST shift"

    # 4. ★the drain model and the capacity estimate are ONE model.  rev2 held
    #    two unrelated ITL constants (13.06 ms for warmup/budget, the regression
    #    for lambda*(A)) and the audit's T1 turned on exactly that gap.
    assert abs(MAX_RUNNING / (511 * (STEP_A_S + STEP_B_S_PER_REQ * MAX_RUNNING))
               - 2.10) < 0.01, "step model must reproduce the registered lambda*(A)"
    itl, bb = decode_itl(0.8293, 512)
    assert 0.025 < itl < 0.026 and 10 < bb < 11, (itl, bb)
    assert decode_itl(2.875, 512) == (None, None), "must call the rung saturated"

    # 5. ★NEGATIVE CONTROL for D16/D17 (lesson 53).  The design rev3 replaces
    #    must be REJECTED by the new condition.  rev2's shape-A ladder at the
    #    fixed 170 s window has kappa 0.91-0.95 on every low rung, so not one of
    #    them clears KAPPA_MIN -- if this passes, the repair does nothing.
    for lam in (2.10, 2.51, 1.08, 3.91):
        for m in (0.40, 0.70):                       # rev2 MULT[A][0], [1]
            c = cell_at("A", m * lam, 170.0, 400)
            assert c["kappa_pred"] is None or c["kappa_pred"] < ACH_HI, (
                "rev2's 170 s low rung must fail the low-side ceiling", lam, m, c)
            # ...and `size_rung` must actually REFUSE to certify it.  Checking
            # the ceiling alone would leave the certification threshold free.
            r = size_rung("A", m * lam, extendable=False)
            assert not r["low_side_candidate"], (
                "a rung whose ceiling is below the low-side threshold must "
                "never be certified as a low-side candidate", lam, m, r)

    # 5b. ★Y2: TTFT_LOAD_FACTOR must be load bearing.  Shape B's drain is
    #     dominated by the 8192-token prefill, so a factor of 1.0 instead of 3.2
    #     would understate the drain by ~2 s and inflate every kappa.  Range is
    #     a literal, not a re-derivation of the formula.
    lb_drain = drain_hat("B", 0.304)[0]
    assert 4.0 < lb_drain < 4.4, ("shape B drain must include the load factor "
                                  "on an 8192-token prefill", lb_drain)
    la_drain = drain_hat("A", 0.525)[0]
    assert 11.0 < la_drain < 13.0, la_drain

    # 6. the plan is a pure function.
    p1, p2 = plan(2.1, 0.675), plan(2.1, 0.675)
    assert json.dumps(p1, sort_keys=True) == json.dumps(p2, sort_keys=True)

    # 7. every registered scenario is RUNNABLE: both seeds inside the guard, at
    #    least one low-side candidate per shape, and every candidate's ceiling
    #    clears KAPPA_MIN with the margin the threshold demands.
    for title, a, b in REGISTERED_GRID:
        p = plan(a, b)
        assert p["seed1_max_abs_ebar_dev"] <= EBAR_TOL, (title, p["seed1_max_abs_ebar_dev"])
        assert p["seed2_max_abs_ebar_dev"] <= EBAR_TOL, (title, p["seed2_max_abs_ebar_dev"])
        for s in ("A", "B"):
            d = p["shapes"][s]
            assert d["n_low_side_candidates"] >= 1, (title, s, d["cells"])
            for c in d["cells"]:
                if c["low_side_candidate"]:
                    # Compared against the LITERAL threshold, not KAPPA_MIN:
                    # asserting `kappa >= KAPPA_MIN` is an identity, so lowering
                    # KAPPA_MIN would certify rungs whose ceiling sits under the
                    # low-side threshold and nothing would notice.
                    assert c["kappa_pred"] >= ACH_HI + 0.02, (title, s, c)
                assert not c.get("window_unsatisfiable"), (title, s, c)
        # ★E5: the registered request must cover the WORST corner of the
        # registered coverage band, not the comfortable one.
        worst = max(budget(p, r)["total_gpu_h"] for r in BUDGET_RATIO_GRID)
        assert worst <= REQUESTED_BUDGET_GPU_H, (title, worst)

    # 8. the fallback ladder must ALSO clear the ceiling -- rev2's fallback did
    #    not (its shape-A rungs sat at 170 s), so it inherited the same defect.
    fb = plan(2.10, 0.675, ladders={k: tuple(v) for k, v in FALLBACK_LADDER.items()})
    assert fb["shapes"]["A"]["n_low_side_candidates"] >= 1, fb["shapes"]["A"]["cells"]
    assert fb["shapes"]["B"]["n_low_side_candidates"] >= 1

    # 9. the window formula is the one the rev1 audit verified against probe C.
    p = plan(2.1, 0.675)
    d = p["shapes"]["A"]
    xs = [c["offered_realized_pred"] for c in d["cells"]]
    assert abs(d["window_realized"][0] - ACH_HI * min(xs)) < 1e-12
    assert abs(d["window_realized"][1] - ACH_LO * max(xs)) < 1e-12

    # 10. N clamps respected; 11 boots.
    for a, b in ((2.10, 0.675), (7.15, 1.20), (1.08, 0.35)):
        p = plan(a, b)
        assert p["n_boot"] == 11
        for s in ("A", "B"):
            for c in p["shapes"][s]["cells"]:
                cap = N_MAX_LONG if c["T_measure_s"] > T_SHORT_S else N_MAX_SHORT
                assert N_MIN <= c["num_prompts"] <= cap, (a, b, s, c)

    # 11. ★D19: rev2's blind-spot theorem is WITHDRAWN, and this records why --
    #     under the real estimator two adjacent rungs CAN sit in (0.90,0.95)
    #     together, which is precisely what rev2's own ladder did.
    rev2_low = [cell_at("A", m * 2.10, 170.0, 400)["kappa_pred"] for m in (0.40, 0.70)]
    assert all(k is not None and ACH_LO < k < ACH_HI for k in rev2_low), rev2_low

    print("PLAN SELFTEST OK (rev4: exponential replay exact and rr=1.0 shown to "
          "be necessary; one step model reproduces lambda*(A)=2.10; rev2's 170 s "
          "low rungs are REJECTED by the new ceiling [negative control]; every "
          "registered scenario has >=1 low-side candidate with kappa>=%.2f, both "
          "seeds within |Ebar-1|<=%.3f, 11 boots; blind-spot "
          "theorem withdrawn with its counterexample recorded)"
          % (KAPPA_MIN, EBAR_TOL))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambda-inf-a", type=float)
    ap.add_argument("--lambda-inf-b", type=float)
    ap.add_argument("--fallback", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--registered-scenarios", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return 0
    if args.registered_scenarios:
        print(registered_scenarios())
        return 0
    if args.lambda_inf_a is None or args.lambda_inf_b is None:
        ap.print_help()
        return 2
    ladders = ({k: tuple(v) for k, v in FALLBACK_LADDER.items()} if args.fallback
               else None)
    p = plan(args.lambda_inf_a, args.lambda_inf_b, ladders=ladders)
    print(json.dumps(p, indent=2, sort_keys=True) if args.json
          else _fmt_plan(p, "measured lambda_inf"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
