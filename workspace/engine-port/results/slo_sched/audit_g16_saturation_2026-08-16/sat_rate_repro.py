#!/usr/bin/env python3
"""Reproduction of the 'SAT rate changes 3x' claim for G16 sec6 ITL_SATURATED.

CONTEXT
  PREREG_G16_RULES_REV3_2026-08-16.md sec6 conditions ``ITL_SATURATED`` on
  "the max spread of ``M_itl`` across the UPPER ARMS < delta" and never defines
  "upper arms".  ``g16_analyze.py::_saturation_gap`` registers the CONTENDING
  set (arms winning >= 1 block) as primary and reports top-half-by-SM alongside.
  Session 4 reported that the choice changes the ITL_SATURATED rate under a
  true tie by 3x (contenders 0.353 vs literal 0.118).  That audit's simulation
  code was NOT preserved (lessons item 39/41).  This file preserves a
  reproduction IN THE REPO.

DESIGN (claims-auditor, 2026-08-16; GPU 0)
  * 7 arms d16..d74, 4 blocks, 1 boot per (arm, block)  == the K5/K6 campaign
    layout.
  * M_itl(a,b) = mu_a + sigma*(sqrt(rho)*c_b + sqrt(1-rho)*e_ab),
    c_b, e_ab ~ N(0,1).  rho = within-block common mode.
  * "true tie"  : mu_a identical for every arm (the estimand the claim is about)
    "monotone g": mu_a decreasing by g ms per arm step (the shape the real
    grids actually have -- g2_0_hard phase B, sgptv HI).
  * M_ttft is given a wide, low-noise separation so the TTFT channel always
    identifies and the ITL channel alone drives the verdict.

POSITIVE CONTROL (teeth, gate #9 / lessons item 39)
  The counterfactual verdict/adaptive action for a DIFFERENT gap definition is
  recomputed here by ``_verdict_from_gap``.  On EVERY simulation the same
  function is fed the contenders gap and asserted to reproduce, exactly, the
  ``verdict`` and ``adaptive_rule`` that ``g16_analyze.decide()`` itself
  emitted.  If the re-derivation ever drifts from the analyzer's real code
  path the run ABORTS -- so the counterfactual is not a re-implementation with
  a different branch (the C2-R failure mode).

  Grid construction goes through ``g16_analyze.BootRecord`` / ``decide()``;
  no estimand is re-implemented.

USAGE
  python3 sat_rate_repro.py                       # headline table
  python3 sat_rate_repro.py --sims 200 --bootstrap 10000   # K3 spot check
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import g16_analyze as g  # noqa: E402  (the pre-registered analyzer itself)

ARMS = [f"d{sm}" for sm in (16, 24, 34, 44, 54, 64, 74)]
PINNED_UPPER_MIN_SM = 44   # "upper arms" pinned by SM value (sec6 delta para:
                           # "d54-d74 must beat d44"; by-product M_itl(d44) -
                           # M_itl(S_max))


def _blank_est(ttft: float, itl: float) -> dict:
    est = {k: 0.0 for k in (
        "requests", "duration_s", "throughput_req_s", "goodput_req_s",
        "ttft_pass_pct", "itl_p95_pass_pct", "joint_pass_pct", "band_mass_ttft",
        "band_mass_itl", "empty_itl_requests", "ttft_p50_ms", "ttft_p95_ms",
        "ttft_p99_ms", "token_itl_p50_ms", "token_itl_p95_ms", "token_itl_p99_ms")}
    est["M_ttft"] = ttft
    est["M_itl"] = itl
    return est


def make_grid(rng: random.Random, *, mu: dict, sigma: float, rho: float,
              n_blocks: int) -> list:
    recs = []
    for b in range(1, n_blocks + 1):
        common = rng.gauss(0.0, 1.0)
        ttft_common = rng.gauss(0.0, 1.0)
        for i, arm in enumerate(ARMS):
            itl = mu[arm] + sigma * (math.sqrt(rho) * common
                                     + math.sqrt(1 - rho) * rng.gauss(0.0, 1.0))
            # TTFT: wide, low-noise separation -> TTFT donor always identified,
            # so the verdict is driven by the ITL channel only.
            ttft = 1000.0 + 100.0 * i + 1.0 * ttft_common + rng.gauss(0.0, 1.0)
            recs.append(g.BootRecord(arm=arm, decode_sm=g.arm_sm(arm),
                                     block=f"blk{b}", boot=1, phase="HI",
                                     path="synthetic", est=_blank_est(ttft, itl)))
    return recs


def _verdict_from_gap(ttft_id: bool, itl_id: bool, gap: float,
                      blocks_disagree: bool) -> tuple:
    """Verbatim transcription of g16_analyze.decide() lines 601-606 and 671-683
    with the gap left as a free argument.  Asserted against the real path."""
    if not ttft_id or not itl_id:
        if (not itl_id) and math.isfinite(gap) and gap < g.K7_DELTA_MS:
            verdict = "ITL_SATURATED"
        else:
            verdict = "UNIDENTIFIED"
    else:
        verdict = "IDENTIFIED"          # stand-in: not a sec6 non-ID verdict
    if verdict in ("UNIDENTIFIED", "ITL_SATURATED"):
        if math.isfinite(gap) and gap < g.K7_DELTA_MS:
            action = "NO_MORE_BLOCKS"
        elif blocks_disagree:
            action = f"ADD_{g.K9_ADAPTIVE_EXTRA_BLOCKS}_BLOCKS"
        else:
            action = "NO_MORE_BLOCKS"
    else:
        action = "NONE"
    return verdict, action


def pinned_gap(arm_means: dict, min_sm: int) -> float:
    vals = [v for a, v in arm_means.items()
            if g.arm_sm(a) >= min_sm and math.isfinite(v)]
    return (max(vals) - min(vals)) if len(vals) >= 2 else math.nan


def run(*, mu: dict, sigma: float, rho: float, sims: int, seed: int,
        n_blocks: int) -> dict:
    rng = random.Random(seed)
    tally = {k: {"ITL_SATURATED": 0, "UNIDENTIFIED": 0, "IDENTIFIED": 0,
                 "NO_MORE_BLOCKS": 0, "ADD_2_BLOCKS": 0, "NONE": 0}
             for k in ("contenders", "top_half_by_sm", "pinned_sm_ge_44")}
    n_contenders = {}
    nan_gap_C = 0
    both_agree = 0
    for _ in range(sims):
        grid = make_grid(rng, mu=mu, sigma=sigma, rho=rho, n_blocks=n_blocks)
        dec = g.decide(grid, label="sim")
        sat = dec["saturation"]
        ttft_id = dec["D_ttft"]["identified"]
        itl_id = dec["D_itl"]["identified"]
        per_block = dec["D_itl"]["rank_rule"]["per_block"]
        winners = {w for w in per_block.values() if w}
        disagree = len(winners) > 1
        n_contenders[len(winners)] = n_contenders.get(len(winners), 0) + 1
        gaps = {
            "contenders": sat["gap_contenders_ms"],
            "top_half_by_sm": sat["gap_top_half_by_sm_ms"],
            "pinned_sm_ge_44": pinned_gap(dec["arm_means"]["M_itl"],
                                          PINNED_UPPER_MIN_SM),
        }
        if not math.isfinite(gaps["contenders"]):
            nan_gap_C += 1
        verdicts = {}
        for key, gap in gaps.items():
            v, a = _verdict_from_gap(ttft_id, itl_id, gap, disagree)
            verdicts[key] = v
            tally[key][v] += 1
            tally[key][a] += 1
        # ---- POSITIVE CONTROL: the re-derivation must reproduce the analyzer
        real_v = dec["verdict"]
        real_a = dec["adaptive_rule"]["action"]
        exp_v, exp_a = _verdict_from_gap(ttft_id, itl_id, gaps["contenders"],
                                         disagree)
        if exp_v == "IDENTIFIED":
            exp_v = real_v          # sec6 truncation/sign verdicts: out of scope
        if not (exp_v == real_v and exp_a == real_a):
            raise SystemExit(
                f"CONTROL FAILED: re-derivation {exp_v}/{exp_a} != analyzer "
                f"{real_v}/{real_a}; gap_C={gaps['contenders']}")
        if verdicts["contenders"] == verdicts["top_half_by_sm"]:
            both_agree += 1
    out = {"sims": sims, "sigma_ms": sigma, "rho": rho, "n_blocks": n_blocks,
           "mu": mu, "rates": {}, "n_contenders_hist": n_contenders,
           "P_gap_contenders_is_nan": nan_gap_C / sims,
           "P_verdicts_agree(C vs top-half)": both_agree / sims}
    for key, counts in tally.items():
        out["rates"][key] = {k: v / sims for k, v in counts.items()}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=1000)
    ap.add_argument("--bootstrap", type=int, default=2000,
                    help="K3 is 10000; the analyzer's own calibrate_donor_rule "
                         "uses 2000 for calibration.  Both are reported.")
    ap.add_argument("--seed", type=int, default=20260816)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    g.K3_BOOTSTRAP_SAMPLES = args.bootstrap   # calibration convention only

    tie = {a: 58.0 for a in ARMS}
    report = {"analyzer_sha256": g._self_sha256(),
              "bootstrap_used": args.bootstrap,
              "K7_delta_ms": g.K7_DELTA_MS, "seed": args.seed,
              "scenarios": {}}
    scen = [
        ("true_tie_sd1.8_rho0", tie, 1.8, 0.0),
        ("true_tie_sd1.0_rho0", tie, 1.0, 0.0),
        ("true_tie_sd3.0_rho0", tie, 3.0, 0.0),
        ("true_tie_sd1.8_rho0.5", tie, 1.8, 0.5),
        ("monotone_1ms_sd1.8_rho0",
         {a: 58.0 - 1.0 * i for i, a in enumerate(ARMS)}, 1.8, 0.0),
        ("monotone_0.4ms_sd1.8_rho0",
         {a: 58.0 - 0.4 * i for i, a in enumerate(ARMS)}, 1.8, 0.0),
    ]
    for name, mu, sd, rho in scen:
        report["scenarios"][name] = run(mu=mu, sigma=sd, rho=rho,
                                        sims=args.sims, seed=args.seed,
                                        n_blocks=g.K5_BLOCKS)
        r = report["scenarios"][name]["rates"]
        print(f"{name:28s} SAT: contenders={r['contenders']['ITL_SATURATED']:.3f} "
              f"top_half={r['top_half_by_sm']['ITL_SATURATED']:.3f} "
              f"pinned>=44={r['pinned_sm_ge_44']['ITL_SATURATED']:.3f} | "
              f"STOP: {r['contenders']['NO_MORE_BLOCKS']:.3f} vs "
              f"{r['top_half_by_sm']['NO_MORE_BLOCKS']:.3f}")
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
