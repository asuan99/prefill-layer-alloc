#!/usr/bin/env python3
"""Analysis for the E1 8B-frontier campaign (results/s8_frontier/).

Per (arm, D-cell, rep, phase[lo/hi]) this reads the sglang.bench_serving
--output-details jsonl (ttfts, itls, input_lens, output_lens, errors,
duration) and:

  1. Applies the PRIMARY, cite-blocking realized-pin gate
     (e1_pin_check.compute_time_weighted_pin_gate/time_weighted_gate_verdict,
     SAME code the sbatch's own per-round E1_PIN_CHECK_LO/HI stdout lines
     use) to each rep/phase. ★★★RE-REVISED 2026-07-30 (coordinator's own
     hand-aggregation of raw telemetry -- e1_pin_check.py's module docstring
     has the full history #1-#6): the 2026-07-29 episode-bracket gate
     (bugs #4/#5) was itself answering the WRONG QUESTION -- it throws away
     most prefill-active evidence to get PER-REQUEST attribution (which
     phase served THIS request's TTFT), but the GATE only needs the
     AGGREGATE answer ("what fraction of prefill WORK ran at target"),
     which should use ALL prefill-active samples, TIME-weighted (the same
     insight as bug #2, re-applied to a different quantity). Coordinator's
     manual count on 866066 D16 (32,145 total snapshots, only 12 prefill-
     active: 7 at target P92, 5 at the P108 auto-partition fallback) gives
     COUNT fraction 7/12=0.583, but TIME-weighting those SAME 12 snapshots
     gives 0.847 -- the 5 fallback snapshots occupy disproportionately
     SHORT durations. The gate criterion is now an episode-level (cluster)
     BOOTSTRAP 95% lower bound of the time-weighted pin fraction >= 0.80
     (NOT Clopper-Pearson, which assumes iid Bernoulli trials that
     serially-correlated time-weighted snapshots are not); `n_episodes`
     (maximal contiguous prefill-active runs, NOT a snapshot count or a
     bracketed-request count) is the effective sample size reported
     alongside every verdict. `compute_episode_gate` (the 2026-07-29
     bracket) is RETAINED but RESCOPED to per-request latency attribution
     only -- not used for gating any more, see e1_pin_check.py's docstring
     bug #6. The MANDATORY CONCURRENCY DIAGNOSTIC (`concurrent_time_frac`,
     time-weighted since bug #2) is genuinely unaffected by this revision
     -- it always answered a time-SHARE question, correctly, and continues
     to.
  2. Computes CONJUNCTIVE goodput per request: TTFT <= SLO_TTFT_MS AND that
     request's OWN token-ITL p95 <= SLO_ITL_P95_MS (gate #4 -- mean-ITL is
     secondary only). PRIMARY ITL-p95 SLO = 60ms (DESIGN.md sec 4.3.1,
     2026-07-28 coordinator revision -- NOT data-fit, taken from
     reports/serving_slo_survey.md's chat-class band + CONSENSUS.md sec 1-17's
     precedent; 150ms, this design's original placeholder, is BELOW every
     arm's measured ITL at every D per FINDINGS_8B_2026-07-28.md sec 2 and
     would make the ITL term never bind, collapsing conjunctive goodput to
     TTFT-only). A pre-registered LADDER {50,60,80}ms (--itl-ladder-ms) is
     reported alongside the 60ms headline as a sensitivity/robustness check
     -- see cliff_hazard()/itl_nonbinding_flag() below. --ttft-slo-ms has no
     default on purpose: it must be sited from e1_capacity_scan.sbatch data
     via ttft_site_check() (DESIGN.md sec 4.3.3), which is a pre-registered
     RULE (bind in >=1 cell, >=15% margin from every cell), not a "pick what
     looks discriminating" choice.

     GATE #8 SCOPE NOTE (read before assuming this ladder violates
     CLAUDE.md's "재스코어 금지" rule): gate #8 is a discipline about
     RE-SCORING A DYNAMIC CONTROLLER against an SLO it was not tuned for --
     invalid because the controller's switching behavior would have differed
     had it been tuned for the new SLO, so you cannot infer what it would
     have done. E1 has no controller: every cell is PDMUX_R2_POLICY=fixed,
     a static partition that never observes or reacts to the SLO. There is
     nothing to re-tune, so evaluating the SAME raw per-request records
     against the PRE-REGISTERED ladder is direct measurement at each SLO
     from data collected once -- exactly what gate #8's own fallback
     ("그 SLO로 재튜닝해 직접 측정") reduces to when there is no controller
     to bias. This does not generalize to any of this project's dynamic-
     policy campaigns (bind/generic/hybrid) -- gate #8 applies to those
     unchanged.
  3. Sums duration across LO+HI phases within a rep (methodology gate #7:
     duration is summed, never max()'d) to get one goodput-rate sample per
     rep. Reports rep-wise mean+-sd (n = number of gate-passing reps), NEVER
     pools raw requests across reps into one CI (methodology gate: no rep
     pooling).
  4. Runs a PAIRED BOOTSTRAP over the per-rep combined-goodput samples to
     compare each D cell against the empirical best-static cell (the D with
     the highest mean combined goodput in the SAME arm) and prints the
     pre-registered decision rule verdict (DESIGN.md sec 4.4): a D "wins" if
     it beats best-static's combined goodput by >=3% AND the paired-bootstrap
     97.5%/2.5% CI of (D - best_static) excludes 0. An arm flagged
     ITL-NONBINDING (sec 4.3.4 -- ITL-p95 stays below the ladder's lowest
     rung at EVERY D cell, so the ITL term is a constant for that arm) has
     its "no D wins" outcome relabeled: it is NOT reported as evidence the
     decode-SM lever is net-negative, because the ladder never tested the
     lever for that arm.

"best static" is NOT a 6th arm/config -- it is the empirical argmax over the
5 measured D cells for that arm (see DESIGN.md sec 4.4 for why a separate
"best static" config was judged unnecessary: the D-grid already spans the
prefill_SM+decode_SM<=108 trade-off space this question is about).

Two modes:
  - Main-sweep mode (default): --dir points at a campaign dir with
    e1_<arm>_<cell>_<job>_telemetry.jsonl + _rounds.jsonl + _rep<r>_{lo,hi}.jsonl.
    Runs the full pipeline above.
  - Scan mode (--capscan-dir): points at an e1_capacity_scan.sbatch output
    dir instead. Computes per-(arm,cell) TTFT/ITL-p95 percentiles at
    --capscan-rate (a candidate RATE_LO or RATE_HI under consideration) and
    runs ONLY the SLO-siting diagnostics (cliff hazard / ITL-nonbinding /
    TTFT margin) -- no goodput/decision rule, since a capacity scan has no
    LO/HI rounds structure. This is the intended DESIGN.md sec 4.2 use: run
    this BEFORE choosing RATE_LO/RATE_HI/TTFT_SLO for the real sweep.

Usage:
  e1_analyze.py [--dir <campaign dir>] --ttft-slo-ms 3000 [--itl-p95-slo-ms 60]
  e1_analyze.py --capscan-dir <scan dir> --capscan-rate 3 --ttft-slo-ms 3000
"""
import argparse
import collections
import glob
import json
import os
import random
import re
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from e1_pin_check import (  # noqa: E402
    compute_concurrency_diagnostic,
    compute_time_weighted_pin_gate,
    time_weighted_gate_verdict,
    reconstruct_arrival_and_completion,
)

CELL_D = {"d16": 16, "d24": 24, "d44": 44, "d54": 54, "d92": 92}
DEFAULT_ITL_LADDER_MS = (50.0, 60.0, 80.0)
CLIFF_MARGIN = 0.15   # DESIGN.md sec 4.3.4
TTFT_MARGIN = 0.15    # DESIGN.md sec 4.3.3
ELBOW_MARGIN = 0.15   # DESIGN.md sec 4.6 (item B.3): pre-registered post-hoc
                       # rep-exclusion margin -- a rep's REALIZED arrival_rps
                       # must sit >=15% below the capacity scan's elbow-onset
                       # rate, or it is reported and excluded (gate #6
                       # enforcement at the per-rep level, not just at
                       # RATE_LO/RATE_HI selection time).


def _percentile(xs, q):
    if not xs:
        return float("nan")
    ys = sorted(xs)
    pos = q * (len(ys) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ys) - 1)
    return ys[lo] + (ys[hi] - ys[lo]) * (pos - lo)


def cliff_hazard(p95_ms, slo_ms, margin=CLIFF_MARGIN):
    """DESIGN.md sec 4.3.4: a cell's measured ITL-p95 is a CLIFF HAZARD for a
    candidate SLO if it falls within +/-margin (relative) of the SLO -- rep
    noise alone could flip its bind/no-bind status. NaN-safe (missing data
    is not a hazard verdict, it is just unknown)."""
    if p95_ms != p95_ms or slo_ms == 0:  # NaN check
        return False
    return abs(p95_ms - slo_ms) / slo_ms <= margin


def itl_nonbinding_flag(per_cell_p95, lowest_rung, margin=CLIFF_MARGIN):
    """DESIGN.md sec 4.3.4: an arm is ITL-NONBINDING if EVERY cell's ITL-p95
    sits comfortably (>=margin) below the ladder's lowest rung -- i.e. the
    ITL term would be a constant across D for that arm, at every rung, so the
    decision rule degenerates to TTFT-only. per_cell_p95: {cell: p95_ms}.
    Returns (flag: bool, worst_cell, worst_p95) where worst_cell/worst_p95
    identify the cell closest to binding (for diagnostics even when the flag
    is False)."""
    vals = {c: v for c, v in per_cell_p95.items() if v == v}  # drop NaN
    if not vals:
        return False, None, float("nan")
    worst_cell = max(vals, key=lambda c: vals[c])
    worst_p95 = vals[worst_cell]
    flag = worst_p95 <= lowest_rung * (1.0 - margin)
    return flag, worst_cell, worst_p95


def ttft_site_check(per_cell_p95, candidate_slo_ms, margin=TTFT_MARGIN):
    """DESIGN.md sec 4.3.3 pre-registered TTFT-SLO siting rule: candidate
    SLO must (i) bind (p95 > SLO) in >=1 cell and (ii) have >=margin relative
    distance from EVERY cell's measured p95. per_cell_p95: {cell: p95_ms}.
    Returns a dict with the verdict and per-cell diagnostics."""
    vals = {c: v for c, v in per_cell_p95.items() if v == v}
    binding = [c for c, v in vals.items() if v > candidate_slo_ms]
    violations = [c for c, v in vals.items()
                  if candidate_slo_ms and abs(v - candidate_slo_ms) / candidate_slo_ms < margin]
    ok = bool(binding) and not violations
    return dict(
        candidate_slo_ms=candidate_slo_ms, ok=ok, binding_cells=sorted(binding),
        margin_violations=sorted(violations), per_cell=vals,
        reason=("PASS" if ok else
                ("ill-posed: binds nowhere (TTFT axis vacuous at this SLO)" if not binding
                 else f"ill-posed: <{margin*100:.0f}% margin at {violations} (metric-cliff risk, gate #6)")),
    )


def load_phase_requests(path):
    """One row per bench_serving --output-details invocation -- reused
    verbatim for both e1_sweep.sbatch's LO/HI phase files and
    e1_capacity_scan.sbatch's per-rate probe files, same jsonl schema. There
    should be exactly one row per file here, but tolerate accidental
    appends. Returns (requests, duration) where each request is
    dict(ttft_s, itl_p95_ms, success)."""
    reqs = []
    dur = 0.0
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        tt = o.get("ttfts") or []
        il = o.get("itls") or []
        err = o.get("errors") or []
        dur += o.get("duration") or 0.0
        for i, t in enumerate(tt):
            success = (err[i] if i < len(err) else "") == ""
            itls_ms = [x * 1000.0 for x in (il[i] if i < len(il) else [])]
            p95 = _percentile(itls_ms, 0.95) if itls_ms else float("nan")
            reqs.append(dict(ttft_s=t, itl_p95_ms=p95, success=success))
    return reqs, dur


def load_raw_arrays(path):
    """Same file as load_phase_requests, but returns the raw ttfts/itls/errors
    arrays in DISPATCH order (as sglang.bench_serving's asyncio.gather
    preserves), needed by e1_pin_check.compute_episode_gate's RNG-replay
    reconstruction (which indexes requests by dispatch order, not
    completion order)."""
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        return o.get("ttfts") or [], o.get("itls") or [], o.get("errors") or []
    return [], [], []


def conjunctive_good(reqs, ttft_slo_ms, itl_p95_slo_ms):
    n_good = 0
    for r in reqs:
        if not r["success"]:
            continue
        if r["ttft_s"] * 1000.0 <= ttft_slo_ms and r["itl_p95_ms"] <= itl_p95_slo_ms:
            n_good += 1
    return n_good


def paired_bootstrap_ci(diffs, n_boot=10000, seed=20260728):
    """diffs: per-rep (D - best_static) paired samples. Returns (mean, lo, hi)
    at the 95% CI via resampling WITH replacement over the paired diffs
    (standard paired bootstrap -- valid because both arms of each pair share
    the same rep index / server boot / trace position, methodology gate:
    n>=4, paired CI)."""
    if len(diffs) < 2:
        return st.fmean(diffs) if diffs else float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    n = len(diffs)
    boots = []
    for _ in range(n_boot):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        boots.append(st.fmean(sample))
    boots.sort()
    lo = boots[int(0.025 * n_boot)]
    hi = boots[min(n_boot - 1, int(0.975 * n_boot))]
    return st.fmean(diffs), lo, hi


def print_slo_siting_report(per_arm_cell_ttft_p95, per_arm_cell_itl_p95,
                             itl_ladder, ttft_candidate):
    """DESIGN.md sec 4.3.4 -- the SLO-siting diagnostics, identical whether
    called from main-sweep mode or --capscan-dir scan mode. Inputs are
    {(arm, cell): p95_ms} dicts. Returns the set of arms flagged
    ITL-NONBINDING (at the ladder's lowest rung) so main()'s decision-rule
    section can suppress the lever-negative conclusion for them."""
    lowest_rung = min(itl_ladder)
    arms = sorted({a for a, _ in per_arm_cell_itl_p95})
    print()
    print(f"=== SLO SITING DIAGNOSTICS (DESIGN.md sec 4.3.4, ITL ladder "
          f"{list(itl_ladder)}ms, TTFT candidate {ttft_candidate}ms) ===")
    nonbinding_arms = set()
    for arm in arms:
        cells_itl = {c: v for (a, c), v in per_arm_cell_itl_p95.items() if a == arm}
        cells_ttft = {c: v for (a, c), v in per_arm_cell_ttft_p95.items() if a == arm}
        print(f"  {arm}:")
        for rung in itl_ladder:
            hazards = [c for c, v in sorted(cells_itl.items())
                       if cliff_hazard(v, rung)]
            detail = ", ".join(f"{c}={v:.1f}ms" for c, v in sorted(cells_itl.items()))
            flag = f"  <-- CLIFF HAZARD at cells {hazards}" if hazards else ""
            print(f"    ITL-p95 ladder rung {rung:.0f}ms: [{detail}]{flag}")
        nb_flag, worst_cell, worst_p95 = itl_nonbinding_flag(cells_itl, lowest_rung)
        if nb_flag:
            nonbinding_arms.add(arm)
            print(f"    ITL-NONBINDING: TRUE (worst cell {worst_cell}={worst_p95:.1f}ms, "
                  f"still >={CLIFF_MARGIN*100:.0f}% below the {lowest_rung:.0f}ms lowest rung) "
                  "-- do NOT read a 'no D beats static' result for this arm as evidence "
                  "the decode-SM lever is net-negative; the ladder never tested it here.")
        else:
            print(f"    ITL-NONBINDING: false (worst cell {worst_cell}="
                  f"{worst_p95:.1f}ms, within reach of the {lowest_rung:.0f}ms rung)")
        if ttft_candidate is not None and cells_ttft:
            site = ttft_site_check(cells_ttft, ttft_candidate)
            detail = ", ".join(f"{c}={v:.0f}ms" for c, v in sorted(cells_ttft.items()))
            print(f"    TTFT-p95 by cell: [{detail}]")
            print(f"    TTFT SLO siting @ {ttft_candidate}ms: {site['reason']} "
                  f"(binding_cells={site['binding_cells']}, margin_violations={site['margin_violations']})")
    return nonbinding_arms


def run_capscan_mode(args):
    """Scan-mode entry point (DESIGN.md sec 4.2/4.3 intended use): read
    e1_capacity_scan.sbatch's per-(arm,cell,rate) probe jsonl and run ONLY
    the SLO-siting diagnostics at a candidate rate, before any main-sweep
    data exists. No goodput/decision rule here -- a capacity-scan probe has
    no LO/HI round structure to sum durations over."""
    pat = re.compile(
        r"e1cap_(?P<arm>\w+?)_(?P<cell>d16|d24|d44|d54|d92)_(?P<job>\d+)_r(?P<rate>[0-9.]+)\.jsonl$"
    )
    ttft_p95, itl_p95 = {}, {}
    found = collections.defaultdict(list)
    for fn in sorted(os.listdir(args.capscan_dir)):
        m = pat.search(fn)
        if not m:
            continue
        arm, cell, rate = m["arm"], m["cell"], float(m["rate"])
        found[(arm, cell)].append(rate)
        if abs(rate - args.capscan_rate) > 1e-9:
            continue
        reqs, _dur = load_phase_requests(os.path.join(args.capscan_dir, fn))
        ttft_p95[(arm, cell)] = _percentile([r["ttft_s"] * 1000.0 for r in reqs if r["success"]], 0.95)
        itl_p95[(arm, cell)] = _percentile(
            [r["itl_p95_ms"] for r in reqs if r["success"] and r["itl_p95_ms"] == r["itl_p95_ms"]], 0.95)
    if not ttft_p95:
        avail = {k: sorted(v) for k, v in found.items()}
        print(f"no capacity-scan probes at rate={args.capscan_rate} under {args.capscan_dir}. "
              f"Available (arm,cell)->rates: {avail}")
        return
    print(f"=== capacity-scan SLO siting @ rate={args.capscan_rate} req/s ({args.capscan_dir}) ===")
    print_slo_siting_report(ttft_p95, itl_p95, args.itl_ladder_ms, args.ttft_slo_ms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=HERE)
    ap.add_argument("--ttft-slo-ms", type=float, required=True,
                     help="site from e1_capacity_scan.sbatch data via ttft_site_check() "
                          "(DESIGN.md sec 4.3.3) -- no default on purpose (gate #6)")
    ap.add_argument("--itl-p95-slo-ms", type=float, default=60.0,
                     help="PRIMARY ITL-p95 SLO, default 60ms -- prior-practice value "
                          "(DESIGN.md sec 4.3.1, reports/serving_slo_survey.md chat-class "
                          "band), NOT fit to this campaign's data. Do not use 150ms (this "
                          "design's discarded original placeholder) beyond pipeline smoke tests.")
    ap.add_argument("--itl-ladder-ms", default="50,60,80",
                     help="pre-registered sensitivity ladder, comma-separated ms values "
                          "(DESIGN.md sec 4.3.1/4.3.4)")
    ap.add_argument("--pin-gate", type=float, default=0.80,
                     help="min episode-bootstrap 95%% lower bound of the TIME-WEIGHTED pin "
                          "fraction (DESIGN.md sec 5 rev. 2026-07-30, bug #6 -- see "
                          "e1_pin_check.py module docstring for the full history)")
    ap.add_argument("--win-margin", type=float, default=0.03,
                     help="decision-rule margin (default 3%%, DESIGN.md sec 4.4)")
    ap.add_argument("--capscan-dir", default=None,
                     help="run in SCAN MODE (DESIGN.md sec 4.2/4.3): read "
                          "e1_capacity_scan.sbatch output from this dir instead of a main "
                          "sweep, and print ONLY the SLO-siting diagnostics at "
                          "--capscan-rate. No goodput/decision rule in this mode.")
    ap.add_argument("--capscan-rate", type=float, default=None,
                     help="candidate rate (req/s) to site SLOs at, required with --capscan-dir")
    ap.add_argument("--elbow-onset-rate-lo", type=float, default=None,
                     help="pre-registered LO-phase elbow-onset arrival_rps from the capacity "
                          "scan (DESIGN.md sec 4.6, item B.3) -- if given, any rep whose "
                          "REALIZED (reconstructed) arrival_rps is within ELBOW_MARGIN of this "
                          "value is reported and EXCLUDED. No default: skipped if not provided.")
    ap.add_argument("--elbow-onset-rate-hi", type=float, default=None,
                     help="same as --elbow-onset-rate-lo, for the HI phase")
    args = ap.parse_args()
    args.itl_ladder_ms = sorted(float(x) for x in args.itl_ladder_ms.split(","))

    if args.capscan_dir:
        if args.capscan_rate is None:
            print("--capscan-rate is required with --capscan-dir")
            return
        run_capscan_mode(args)
        return

    pat = re.compile(
        r"e1_(?P<arm>\w+?)_(?P<cell>d16|d24|d44|d54|d92)_(?P<job>\d+)_telemetry\.jsonl$"
    )
    cells = []
    for fn in sorted(os.listdir(args.dir)):
        m = pat.search(fn)
        if m:
            cells.append((m.groupdict(), os.path.join(args.dir, fn)))
    if not cells:
        print(f"no e1 telemetry under {args.dir}")
        return

    # (arm, cell) -> list of per-rep dicts
    by_cell = collections.defaultdict(list)
    as_run_dropped = []
    as_run_elbow_excluded = []

    for g, tpath in cells:
        arm, cell, job = g["arm"], g["cell"], g["job"]
        want_d = CELL_D[cell]
        stem = tpath[: -len("_telemetry.jsonl")]
        rounds_path = f"{stem}_rounds.jsonl"
        if not os.path.exists(rounds_path):
            print(f"WARN: no rounds.jsonl for {arm}/{cell} job={job} -- skipping (cannot window-gate)")
            continue
        for line in open(rounds_path):
            line = line.strip()
            if not line:
                continue
            rd = json.loads(line)
            rep = rd["rep"]
            # rounds.jsonl written before the 2026-07-29 seed-policy fix
            # (DESIGN.md sec 4.5) has no "seed" key -- those runs implicitly
            # used sglang.bench_serving's own default (seed=1), so that is
            # the correct fallback, not an arbitrary guess.
            seed = rd.get("seed", 1)
            lo_win, hi_win = rd["lo"], rd["hi"]

            jlo = f"{stem}_rep{rep}_lo.jsonl"
            jhi = f"{stem}_rep{rep}_hi.jsonl"
            if not (os.path.exists(jlo) and os.path.exists(jhi)):
                continue
            reqs_lo, dur_lo = load_phase_requests(jlo)
            reqs_hi, dur_hi = load_phase_requests(jhi)
            ttfts_lo_raw, itls_lo_raw, errors_lo_raw = load_raw_arrays(jlo)
            ttfts_hi_raw, itls_hi_raw, errors_hi_raw = load_raw_arrays(jhi)

            conc_lo = compute_concurrency_diagnostic(tpath, lo_win["t0"], lo_win["t1"])
            conc_hi = compute_concurrency_diagnostic(tpath, hi_win["t0"], hi_win["t1"])
            # PRIMARY, cite-blocking gate (DESIGN.md sec 5, 2026-07-30
            # revision, bug #6): TIME-WEIGHTED pin fraction over ALL
            # prefill-active time, episode-level bootstrap lower bound.
            # Needs only telemetry + window -- no seed/rate/ttfts, unlike
            # the retired (now latency-attribution-only) episode bracket.
            g_lo = compute_time_weighted_pin_gate(tpath, want_d, lo_win["t0"], lo_win["t1"])
            g_hi = compute_time_weighted_pin_gate(tpath, want_d, hi_win["t0"], hi_win["t1"])
            v_lo = time_weighted_gate_verdict(g_lo, args.pin_gate)
            v_hi = time_weighted_gate_verdict(g_hi, args.pin_gate)
            # PRIMARY, cite-blocking criterion: pin_pass (time-weighted,
            # episode-bootstrap lower bound) only. concurrent_time_frac is a
            # MANDATORY diagnostic (reported below and per-rep) but not
            # gate-blocking -- no principled threshold exists yet.
            gates_pass = v_lo["pin_pass"] and v_hi["pin_pass"]

            # DESIGN.md sec 4.6 item B.3: pre-registered post-hoc elbow-
            # margin check. A rep's REALIZED (reconstructed, not nominal --
            # item B.1/B.5) arrival_rps must sit >=ELBOW_MARGIN below the
            # capacity scan's elbow-onset rate for that phase, or gate #6
            # is violated for THIS rep specifically even if RATE_LO/RATE_HI
            # were chosen correctly on average. Skipped (arrival_rps_lo/hi
            # still computed and reported) if the onset rates were not given.
            arrival_rel_lo, _ = reconstruct_arrival_and_completion(seed, lo_win["rate"], ttfts_lo_raw, itls_lo_raw)
            arrival_rel_hi, _ = reconstruct_arrival_and_completion(seed, hi_win["rate"], ttfts_hi_raw, itls_hi_raw)
            arrival_rps_lo = ((len(ttfts_lo_raw) - 1) / arrival_rel_lo[-1]) if len(ttfts_lo_raw) > 1 and arrival_rel_lo[-1] else float("nan")
            arrival_rps_hi = ((len(ttfts_hi_raw) - 1) / arrival_rel_hi[-1]) if len(ttfts_hi_raw) > 1 and arrival_rel_hi[-1] else float("nan")
            elbow_violation_lo = (args.elbow_onset_rate_lo is not None and arrival_rps_lo == arrival_rps_lo
                                   and arrival_rps_lo > args.elbow_onset_rate_lo * (1.0 - ELBOW_MARGIN))
            elbow_violation_hi = (args.elbow_onset_rate_hi is not None and arrival_rps_hi == arrival_rps_hi
                                   and arrival_rps_hi > args.elbow_onset_rate_hi * (1.0 - ELBOW_MARGIN))
            if elbow_violation_lo or elbow_violation_hi:
                as_run_elbow_excluded.append((arm, cell, rep, arrival_rps_lo, arrival_rps_hi,
                                               elbow_violation_lo, elbow_violation_hi))
                continue  # excluded per item B.3, not counted toward n>=4 even if pin gate passed

            good_lo = conjunctive_good(reqs_lo, args.ttft_slo_ms, args.itl_p95_slo_ms)
            good_hi = conjunctive_good(reqs_hi, args.ttft_slo_ms, args.itl_p95_slo_ms)
            dur_sum = dur_lo + dur_hi  # gate #7: sum, never max()
            gp_combined = (good_lo + good_hi) / dur_sum if dur_sum else float("nan")

            rec = dict(
                rep=rep, gates_pass=gates_pass,
                arrival_rps_lo=arrival_rps_lo, arrival_rps_hi=arrival_rps_hi,
                pin_lo=g_lo["pin_frac"], pin_hi=g_hi["pin_frac"],
                pin_lower95_lo=g_lo["pin_frac_lower95"], pin_lower95_hi=g_hi["pin_frac_lower95"],
                n_episodes_lo=g_lo["n_episodes"], n_episodes_hi=g_hi["n_episodes"],
                realized_hist_lo=g_lo["realized_hist"], realized_hist_hi=g_hi["realized_hist"],
                t_prefill_active_lo=g_lo["t_prefill_active_total_s"], t_prefill_active_hi=g_hi["t_prefill_active_total_s"],
                concurrent_frac_lo=conc_lo["concurrent_time_frac"], concurrent_frac_hi=conc_hi["concurrent_time_frac"],
                admission_blocked_lo=conc_lo["admission_blocked_frac"],
                admission_blocked_hi=conc_hi["admission_blocked_frac"],
                n_lo=len(reqs_lo), n_hi=len(reqs_hi),
                good_lo=good_lo, good_hi=good_hi, dur_lo=dur_lo, dur_hi=dur_hi,
                gp_combined=gp_combined,
                ttft_lo=[r["ttft_s"] * 1000.0 for r in reqs_lo],
                ttft_hi=[r["ttft_s"] * 1000.0 for r in reqs_hi],
                itl_lo=[r["itl_p95_ms"] for r in reqs_lo if r["itl_p95_ms"] == r["itl_p95_ms"]],
                itl_hi=[r["itl_p95_ms"] for r in reqs_hi if r["itl_p95_ms"] == r["itl_p95_ms"]],
            )
            if gates_pass:
                by_cell[(arm, cell)].append(rec)
            else:
                as_run_dropped.append((arm, cell, rep, v_lo, v_hi, g_lo, g_hi))

    if as_run_dropped:
        print("=== reps dropped by pre-registered PIN gate (as-run only, not cited) ===")
        for arm, cell, rep, v_lo, v_hi, g_lo, g_hi in as_run_dropped:
            print(f"  {arm}/{cell} rep{rep}: LO(pin_pass={v_lo['pin_pass']}, "
                  f"reason='{v_lo['reason']}', n_episodes={g_lo['n_episodes']}, "
                  f"pin_frac={g_lo['pin_frac']:.3f}) "
                  f"HI(pin_pass={v_hi['pin_pass']}, reason='{v_hi['reason']}', "
                  f"n_episodes={g_hi['n_episodes']}, pin_frac={g_hi['pin_frac']:.3f})")
        print()

    if as_run_elbow_excluded:
        print("=== reps EXCLUDED by the pre-registered elbow-margin check (DESIGN.md sec 4.6 "
              "item B.3 -- as-run only, not cited, checked BEFORE goodput so a rep that also "
              "would have passed the pin gate is still excluded here) ===")
        for arm, cell, rep, arps_lo, arps_hi, viol_lo, viol_hi in as_run_elbow_excluded:
            print(f"  {arm}/{cell} rep{rep}: arrival_rps(lo/hi)={arps_lo:.2f}/{arps_hi:.2f} "
                  f"violation(lo/hi)={viol_lo}/{viol_hi} "
                  f"(onset_rate(lo/hi)={args.elbow_onset_rate_lo}/{args.elbow_onset_rate_hi}, "
                  f"margin={ELBOW_MARGIN})")
        print()

    print(f"=== per-rep summary (gate-passing reps only, TTFT_SLO={args.ttft_slo_ms}ms "
          f"ITL_P95_SLO={args.itl_p95_slo_ms}ms) ===")
    for (arm, cell), recs in sorted(by_cell.items()):
        print(f"  {arm}/{cell} n={len(recs)}:")
        for r in recs:
            print(f"    rep{r['rep']}: gp_combined={r['gp_combined']:.4f} "
                  f"(good={r['good_lo']}+{r['good_hi']} dur={r['dur_lo']:.1f}+{r['dur_hi']:.1f}s) "
                  f"n_lo={r['n_lo']} n_hi={r['n_hi']} "
                  f"arrival_rps(lo/hi)={r['arrival_rps_lo']:.2f}/{r['arrival_rps_hi']:.2f} "
                  "[TRUSTED x-axis label, item B.1/B.5 -- nominal rate is reference only] "
                  f"pin(lo/hi)={r['pin_lo']:.2f}/{r['pin_hi']:.2f} "
                  f"pin_lower95(lo/hi)={r['pin_lower95_lo']:.2f}/{r['pin_lower95_hi']:.2f} "
                  f"n_episodes(lo/hi)={r['n_episodes_lo']}/{r['n_episodes_hi']} "
                  f"concurrent_time_frac(lo/hi)={r['concurrent_frac_lo']:.4f}/{r['concurrent_frac_hi']:.4f} "
                  f"admission_blocked(lo/hi)={r['admission_blocked_lo']:.2f}/{r['admission_blocked_hi']:.2f}")
            # DESIGN.md sec 4.8 item 3 (coordinator 2026-07-30): the realized
            # TIME-WEIGHTED partition split (target vs auto-partition
            # fallback) is a FIRST-CLASS result, not a footnote -- report it
            # per rep so "target D=16, realized target/auto-partition split"
            # is always available, never just the target label alone.
            hist_lo = ", ".join(f"P{p}:{t_:.2f}s({t_/r['t_prefill_active_lo']*100:.0f}%)"
                                 for p, t_ in sorted(r["realized_hist_lo"].items(), key=lambda kv: -kv[1])) \
                if r["t_prefill_active_lo"] else "(no prefill-active time)"
            hist_hi = ", ".join(f"P{p}:{t_:.2f}s({t_/r['t_prefill_active_hi']*100:.0f}%)"
                                 for p, t_ in sorted(r["realized_hist_hi"].items(), key=lambda kv: -kv[1])) \
                if r["t_prefill_active_hi"] else "(no prefill-active time)"
            print(f"      realized partition split LO: [{hist_lo}]  HI: [{hist_hi}]")

    print()
    print("=== MANDATORY DIAGNOSTIC: D-axis concurrency exercised (bug #2, time-weighted since "
          "2026-07-29, genuinely unaffected by the 2026-07-30 gate revision, bug #6 -- these are "
          "different quantities, see DESIGN.md sec 5) ===")
    print("(concurrent_time_frac = TIME-WEIGHTED share of the window with prefill AND decode "
          "simultaneously active -- how much of wall-clock actually tested the D-axis tradeoff. "
          "This was count-based (snapshots-per-iteration, not time) before 2026-07-29 and badly "
          "understated true concurrency (~15-30x at 866066) -- see DESIGN.md sec 9.7. LOW values "
          "weaken interpretation of a passing pin gate at that cell -- cite this number alongside "
          "any result.)")
    for (arm, cell), recs in sorted(by_cell.items()):
        cl = [r["concurrent_frac_lo"] for r in recs]
        ch = [r["concurrent_frac_hi"] for r in recs]
        nel_ = [r["n_episodes_lo"] for r in recs]
        neh_ = [r["n_episodes_hi"] for r in recs]
        low_flag = "  <-- LOW CONCURRENCY" if (cl and st.fmean(cl) < 0.05) or (ch and st.fmean(ch) < 0.05) else ""
        print(f"  {arm}/{cell}: concurrent_time_frac mean(lo/hi)={st.fmean(cl):.4f}/{st.fmean(ch):.4f} "
              f"n_episodes mean(lo/hi)={st.fmean(nel_):.1f}/{st.fmean(neh_):.1f}{low_flag}")

    print()
    print("=== per-cell TTFT/ITL percentiles (rep-pooled requests, descriptive only -- "
          "the goodput CI below is rep-wise, this is NOT what the decision rule uses) ===")
    ttft_p95_by_cell, itl_p95_by_cell = {}, {}
    for (arm, cell), recs in sorted(by_cell.items()):
        all_ttft = [x for r in recs for x in r["ttft_lo"] + r["ttft_hi"]]
        all_itl = [x for r in recs for x in r["itl_lo"] + r["itl_hi"]]
        ttft_p95_by_cell[(arm, cell)] = _percentile(all_ttft, .95)
        itl_p95_by_cell[(arm, cell)] = _percentile(all_itl, .95)
        print(f"  {arm}/{cell}: TTFT_ms p50/p95/p99={_percentile(all_ttft,.5):.0f}/"
              f"{_percentile(all_ttft,.95):.0f}/{_percentile(all_ttft,.99):.0f}  "
              f"reqITLp95_ms p50/p95={_percentile(all_itl,.5):.1f}/{_percentile(all_itl,.95):.1f}")

    # DESIGN.md sec 4.3.4: same siting diagnostics whether this is run
    # post-hoc against the main sweep (here) or pre-hoc against the capacity
    # scan (--capscan-dir mode, run_capscan_mode()). This is a cross-check --
    # the intended siting decision happens BEFORE the sweep, from scan data.
    nonbinding_arms = print_slo_siting_report(
        ttft_p95_by_cell, itl_p95_by_cell, args.itl_ladder_ms, args.ttft_slo_ms)

    print()
    print("=== PRIMARY: per-cell combined conjunctive goodput, rep-wise mean+-sd, n>=4 gate ===")
    cell_stats = {}
    for (arm, cell), recs in sorted(by_cell.items()):
        gps = [r["gp_combined"] for r in recs]
        m = st.fmean(gps) if gps else float("nan")
        sd = st.stdev(gps) if len(gps) > 1 else 0.0
        cell_stats[(arm, cell)] = dict(mean=m, sd=sd, n=len(gps), gps=gps)
        flag = "" if len(gps) >= 4 else "  [n<4 -- DO NOT CITE, methodology gate]"
        print(f"  {arm}/{cell}: goodput_rate={m:.4f} +/- {sd:.4f} (n={len(gps)}){flag}")

    print()
    print("=== DECISION RULE (DESIGN.md sec 4.4): does any D beat empirical best-static "
          f"by >={args.win_margin*100:.0f}% with paired-bootstrap 95% CI excluding 0? ===")
    arms = sorted({a for a, _ in cell_stats})
    for arm in arms:
        arm_cells = {c: s for (a, c), s in cell_stats.items() if a == arm}
        if not arm_cells:
            continue
        citeable = {c: s for c, s in arm_cells.items() if s["n"] >= 4}
        if not citeable:
            print(f"  {arm}: no cell has n>=4 gate-passing reps -- NO CITEABLE VERDICT")
            continue
        best_cell = max(citeable, key=lambda c: citeable[c]["mean"])
        best_mean = citeable[best_cell]["mean"]
        nb_tag = "  [ITL-NONBINDING -- see sec 4.3.4 caveat below]" if arm in nonbinding_arms else ""
        print(f"  {arm}: empirical best-static = {best_cell} (goodput={best_mean:.4f}, n={citeable[best_cell]['n']}){nb_tag}")
        any_win = False
        for c, s in sorted(citeable.items()):
            if c == best_cell:
                continue
            # paired diff requires matching rep indices between c and best_cell
            best_by_rep = {r["rep"]: r["gp_combined"] for r in by_cell[(arm, best_cell)]}
            c_by_rep = {r["rep"]: r["gp_combined"] for r in by_cell[(arm, c)]}
            shared_reps = sorted(set(best_by_rep) & set(c_by_rep))
            if len(shared_reps) < 4:
                print(f"    {arm}/{c} vs {best_cell}: only {len(shared_reps)} shared rep indices "
                      "(<4) -- CANNOT form a paired CI, not cited")
                continue
            diffs = [c_by_rep[rp] - best_by_rep[rp] for rp in shared_reps]
            mean_diff, ci_lo, ci_hi = paired_bootstrap_ci(diffs)
            rel = mean_diff / best_mean if best_mean else float("nan")
            wins = rel >= args.win_margin and ci_lo > 0
            any_win = any_win or wins
            verdict = "WINS (>=margin AND CI excludes 0)" if wins else "does not win"
            print(f"    {arm}/{c} vs {best_cell}: mean_diff={mean_diff:+.4f} ({rel*100:+.1f}%) "
                  f"95%CI=[{ci_lo:+.4f},{ci_hi:+.4f}] n_pairs={len(shared_reps)} -> {verdict}")
        if not any_win and arm in nonbinding_arms:
            print(f"  {arm}: VERDICT = no D beats best-static, BUT this arm is ITL-NONBINDING "
                  f"(sec 4.3.4) -- do NOT report this as \"lever net-negative\" for {arm}. The "
                  "ITL term was a constant across D for this arm at every ladder rung, so the "
                  "decision rule degenerated to TTFT-only; this arm's tension-A question was "
                  "not actually tested by this SLO ladder. Report as ITL-NONBINDING, not as a "
                  "lever-negative finding.")
        elif not any_win:
            print(f"  {arm}: VERDICT = no D beats best-static -> supports \"lever exists but "
                  "net-negative under the prefill+decode<=108 budget\" for this arm "
                  "(cite ONLY after claims-auditor review, per s0_deconfound/DESIGN.md sec 5)")
        else:
            print(f"  {arm}: VERDICT = at least one D beats best-static -> tension A resolves "
                  "toward net-positive for this arm (cite ONLY after claims-auditor review)")


if __name__ == "__main__":
    main()
