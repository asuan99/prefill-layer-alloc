#!/usr/bin/env python3
"""Gate 2 rev4 main-campaign scorer.

Mechanical application ONLY of the rules pre-registered in
PREREG_GATE2_2026-08-06.md (rev4) + the 2026-08-07 dated addenda (section 14
of that file). This script computes measurements and applies the fixed
decision rule; it does not interpret results -- that judgment call belongs
to result-analyst (goodput/percentile/paired-bootstrap-style verification)
and claims-auditor (adversarial confound review) per this project's agent
division of labour (CLAUDE.md).

Primary  (PREREG section 4.1, 5.1, 5.2): X_60 TOST equivalence (A3 vs A4)
         AND TTFT p95 non-inferiority -> R3' / R4' / INCONCLUSIVE, per cell,
         with Holm correction across the 4 replication cells (designated
         cell = Granite rate 4, PREREG section 5.3).
Secondary (reported, never used for R3'/R4'): TTFT p95, request-ITL-p95-p90,
         raw request_throughput (complement negative control), per-token
         stall probability, phi (descriptive only, no Fieller CI here --
         see caveat below), goodput (both predicates, reporting only).
Diagnostics (not scored): Phase 0 F-A/F-B/F-E, section 5.1.1 mid-check
         (UNDERPOWERED flag), tau ladder {40,60,100} sign-only for
         Delta_chunk = X_tau(A3) - X_tau(A1), section 6 reproducibility
         check against 873944/873945.

KNOWN GAPS (flagged rather than silently omitted, per this project's
methodology gates -- do not present these as more complete than they are):
  - phi's Fieller CI (PREREG section 4.4) is NOT implemented; only the point
    estimate is printed, already marked descriptive-only in the prereg.
  - Delta_chunk is defined here as X_tau(A3) - X_tau(A1) ("what the chunk512
    flag changes relative to plain"); the prereg text does not give an
    unambiguous algebraic definition, so this choice is stated explicitly
    and should be checked by the next reader (claims-auditor/result-analyst)
    before being cited.
  - DEGENERATE-T (section 5.1.2 item 1) is applied to block BOTH the TOST
    equivalence (R3') and the raw superiority (R4') mechanical verdicts for
    a cell when the paired A3-A4 diff SD is exactly 0 -- the prereg only
    says "no equivalence declaration", but a zero-variance paired-t CI is
    equally untrustworthy for a superiority claim, so the block is applied
    symmetrically. This is a scoring-script decision, flagged here for audit.

Usage:
  python3 g2_analyze.py --dir <gate2 dir> --tag zamba2-27b --jobid <id> \
      --rates 2 3 --designated-rate 4   (designated-rate only meaningful for
                                          the Granite tag; default: none)
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics
import sys
from itertools import product
from typing import Any, Dict, List, Optional, Tuple

try:
    from scipy import stats as _scipy_stats
except Exception:  # pragma: no cover
    _scipy_stats = None

ARMS = ["plain", "plainaux", "chunk512", "agnostic"]
TAU_PRIMARY = 60.0
TAU_LADDER = [40.0, 60.0, 100.0]
DELTA = 0.05
SCAN_RATES = [1, 2, 3, 4, 5, 6, 8, 10, 12]
NREP_SCAN = 3
NREP_MAIN = 10


# ---------------------------------------------------------------------------
# generic helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> List[dict]:
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rows.append(json.loads(ln))
            except json.JSONDecodeError:
                continue
    return rows


def load_last_bench_row(path: str) -> Optional[dict]:
    rows = load_jsonl(path)
    return rows[-1] if rows else None


def percentile(sorted_vals: List[float], frac: float) -> Optional[float]:
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    idx = frac * (len(sorted_vals) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return sorted_vals[lo]
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (idx - lo)


def t_ppf(p: float, df: int) -> float:
    if _scipy_stats is not None:
        return float(_scipy_stats.t.ppf(p, df))
    table95 = {1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015, 6: 1.943,
               7: 1.895, 8: 1.860, 9: 1.833, 10: 1.812}
    if p == 0.95 and df in table95:
        return table95[df]
    table975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
                7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
    if p == 0.975 and df in table975:
        return table975[df]
    return 1.96


def t_cdf(x: float, df: int) -> float:
    if _scipy_stats is not None:
        return float(_scipy_stats.t.cdf(x, df))
    # crude fallback; scipy is present in this project's venv (verified elsewhere)
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


# ---------------------------------------------------------------------------
# per-request metrics from a raw bench_serving --output-details row
# ---------------------------------------------------------------------------

def request_max_itl_ms(row: dict) -> List[Optional[float]]:
    """Per-request max(itl) in ms, or None if the request is excluded
    (errored, or missing ITL list) -- PREREG section 4.1: "미기록-ITL 요청은
    분자.분모 모두에서 제외, 개수를 n_missing_itl로 별도 보고."""
    itls = row.get("itls") or []
    errors = row.get("errors") or []
    out: List[Optional[float]] = []
    for i, il in enumerate(itls):
        err = errors[i] if i < len(errors) else None
        if err:
            out.append(None)
            continue
        if not il:
            out.append(None)
            continue
        out.append(max(il) * 1000.0)
    return out


def x_tau(row: dict, tau: float) -> Dict[str, Any]:
    maxitls = request_max_itl_ms(row)
    used = [m for m in maxitls if m is not None]
    n_missing = sum(1 for m in maxitls if m is None)
    n_total = len(maxitls)
    if not used:
        return {"X": None, "n_used": 0, "n_missing_itl": n_missing, "n_total": n_total,
                "n_viol_latency": 0}
    n_viol = sum(1 for m in used if m > tau)
    return {
        "X": n_viol / len(used), "n_used": len(used), "n_missing_itl": n_missing,
        "n_total": n_total, "n_viol_latency": n_viol,
    }


def ttft_p95_ms(row: dict) -> Optional[float]:
    ttfts = sorted(t * 1000.0 for t in (row.get("ttfts") or []) if t is not None)
    return percentile(ttfts, 0.95)


def ttft_p50_ms(row: dict) -> Optional[float]:
    return row.get("median_ttft_ms")


def ttft_p99_ms(row: dict) -> Optional[float]:
    return row.get("p99_ttft_ms")


def itl_p50_ms(row: dict) -> Optional[float]:
    return row.get("median_itl_ms")


def request_itl_p95_p90_ms(row: dict) -> Optional[float]:
    """PREREG section 4.2: request-level token-ITL p95, then p90 across
    requests within the rep -- descriptive."""
    per_req_p95 = []
    for il in (row.get("itls") or []):
        if not il:
            continue
        s = sorted(x * 1000.0 for x in il)
        p = percentile(s, 0.95)
        if p is not None:
            per_req_p95.append(p)
    if not per_req_p95:
        return None
    return percentile(sorted(per_req_p95), 0.90)


def token_stall_probability(row: dict, tau: float) -> Optional[float]:
    """PREREG section 4.2: p = #(gap>tau)/#(gaps) -- descriptive only, noisy
    for A4 (few events/run)."""
    n_gaps = 0
    n_over = 0
    for il in (row.get("itls") or []):
        for g in il:
            n_gaps += 1
            if g * 1000.0 > tau:
                n_over += 1
    if n_gaps == 0:
        return None
    return n_over / n_gaps


SLO_TTFT_S = 3.0
SLO_ITL_S = 0.060


def goodput_canonical(row: dict) -> Tuple[Optional[float], int, int]:
    """Project-canonical goodput predicate (CLAUDE.md gate 4): TTFT<=SLO AND
    per-request p95 ITL <=SLO. Returns (goodput_req_per_s, n_good, n_total).
    REPORTING ONLY -- PREREG section 5.4 bans this from any verdict."""
    ttfts = row.get("ttfts") or []
    itls = row.get("itls") or []
    errors = row.get("errors") or []
    dur = row.get("duration") or 0.0
    n_total = len(ttfts)
    good = 0
    for i in range(n_total):
        err = errors[i] if i < len(errors) else None
        if err:
            continue
        tt = ttfts[i]
        il = itls[i] if i < len(itls) else []
        if not il:
            continue
        p95 = percentile(sorted(il), 0.95)
        if tt is not None and tt <= SLO_TTFT_S and p95 is not None and p95 <= SLO_ITL_S:
            good += 1
    return ((good / dur) if dur else None, good, n_total)


def goodput_mean_itl(row: dict) -> Tuple[Optional[float], int, int]:
    """Legacy mean-ITL predicate (kept ONLY because PREREG section 4.2 asks
    for both predicates to be reported -- do not use for verdicts)."""
    ttfts = row.get("ttfts") or []
    itls = row.get("itls") or []
    errors = row.get("errors") or []
    dur = row.get("duration") or 0.0
    n_total = len(ttfts)
    good = 0
    for i in range(n_total):
        err = errors[i] if i < len(errors) else None
        if err:
            continue
        tt = ttfts[i]
        il = itls[i] if i < len(itls) else []
        mean_itl = (sum(il) / len(il)) if il else float("inf")
        if tt is not None and tt <= SLO_TTFT_S and mean_itl <= SLO_ITL_S:
            good += 1
    return ((good / dur) if dur else None, good, n_total)


# ---------------------------------------------------------------------------
# paired-t / TOST
# ---------------------------------------------------------------------------

def paired_t_ci_twosided(diffs: List[float], conf: float = 0.95) -> Dict[str, Any]:
    n = len(diffs)
    if n < 2:
        return {"n": n, "mean": (diffs[0] if n == 1 else None), "sd": None, "ci_lo": None, "ci_hi": None}
    mean = statistics.mean(diffs)
    sd = statistics.stdev(diffs)
    se = sd / math.sqrt(n)
    if sd == 0.0:
        return {"n": n, "mean": mean, "sd": 0.0, "se": 0.0, "ci_lo": mean, "ci_hi": mean, "degenerate": True}
    tcrit = t_ppf(1 - (1 - conf) / 2, n - 1)
    return {"n": n, "mean": mean, "sd": sd, "se": se, "ci_lo": mean - tcrit * se, "ci_hi": mean + tcrit * se,
            "degenerate": False}


def tost_equivalence(diffs: List[float], delta: float, alpha: float = 0.05) -> Dict[str, Any]:
    """Two one-sided tests, paired. Returns 90% CI (two 1-sided 5% tests),
    the TOST p-value (max of the two one-sided p's), and whether the 90% CI
    falls entirely within (-delta, +delta)."""
    n = len(diffs)
    if n < 2:
        return {"n": n, "verdict": "INSUFFICIENT_N"}
    mean = statistics.mean(diffs)
    sd = statistics.stdev(diffs)
    if sd == 0.0:
        return {"n": n, "mean": mean, "sd": 0.0, "verdict": "DEGENERATE-T",
                "note": "paired diff SD is exactly 0 -- PREREG section 5.1.2 item 1: report point "
                        "value only, no equivalence (or, by this script's symmetric extension, "
                        "superiority) declaration."}
    se = sd / math.sqrt(n)
    df = n - 1
    tcrit90 = t_ppf(0.95, df)  # 90% CI = two one-sided 5% tests
    ci_lo = mean - tcrit90 * se
    ci_hi = mean + tcrit90 * se
    t1 = (mean - delta) / se  # H01: mean >= delta
    t2 = (mean + delta) / se  # H02: mean <= -delta
    p1 = t_cdf(t1, df)
    p2 = 1 - t_cdf(t2, df)
    p_tost = max(p1, p2)
    equivalent = (ci_lo > -delta) and (ci_hi < delta)
    return {
        "n": n, "mean": mean, "sd": sd, "se": se, "delta": delta,
        "ci90_lo": ci_lo, "ci90_hi": ci_hi, "p_tost": p_tost,
        "equivalent": equivalent, "verdict": "EQUIVALENT" if equivalent else "NOT_EQUIVALENT",
    }


def sign_flip_permutation_p(diffs: List[float]) -> Optional[float]:
    """Exact two-sided sign-flip permutation p-value for H0: median diff = 0
    (PREREG section 5.1.2 item 2 boundary caveat). n<=12 -> exhaustive
    2^n enumeration is cheap."""
    n = len(diffs)
    if n == 0:
        return None
    if n > 20:
        return None  # not attempted -- would need Monte Carlo, out of scope here
    obs = abs(sum(diffs))
    n_ge = 0
    n_perm = 2 ** n
    for signs in product([1, -1], repeat=n):
        s = sum(d * sgn for d, sgn in zip(diffs, signs))
        if abs(s) >= obs - 1e-12:
            n_ge += 1
    return n_ge / n_perm


def holm_adjust(pvals: List[Tuple[str, float]]) -> List[Tuple[str, float, float]]:
    """Holm-Bonferroni step-down. pvals: list of (label, p). Returns
    (label, p_raw, p_holm) sorted by original order."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i][1])
    adj = [None] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        label, p = pvals[idx]
        a = min(1.0, (m - rank) * p)
        running_max = max(running_max, a)
        adj[idx] = running_max
    return [(pvals[i][0], pvals[i][1], adj[i]) for i in range(m)]


# ---------------------------------------------------------------------------
# Phase 0 (capacity scan) -- F-A/F-B/F-E + section 5.1.1 mid-check
# ---------------------------------------------------------------------------

def phase0_load(dirpath: str, tag: str, jobid: str, arm: str) -> Dict[Tuple[int, int], dict]:
    """returns {(seed, rate): row}"""
    out = {}
    for seed in (7, 17, 27):
        path = os.path.join(dirpath, f"g2_{tag}_capscan_{arm}_seed{seed}_{jobid}.jsonl")
        rows = load_jsonl(path)
        by_rate: Dict[int, dict] = {}
        for r in rows:
            rr = r.get("request_rate")
            if rr is not None:
                by_rate[int(rr)] = r
        for rate, row in by_rate.items():
            out[(seed, rate)] = row
    return out


def phase0_report(dirpath: str, tag: str, jobid: str, main_rates: List[int]) -> Dict[str, Any]:
    print(f"\n=== Phase 0 capacity scan (n=3 seeds): tag={tag} jobid={jobid} ===")
    report: Dict[str, Any] = {}
    for arm in ARMS:
        data = phase0_load(dirpath, tag, jobid, arm)
        if not data:
            print(f"  arm={arm}: NO capscan data found")
            continue
        print(f"  --- arm={arm} ---")
        print(f"  {'rate':>5} {'seed':>5} {'compl/total':>12} {'ttft_p50(ms)':>13} "
              f"{'req_thpt':>9} {'F-A':>6}")
        arm_report: Dict[str, Any] = {"F_A": [], "F_B_slope": None, "F_E": {}}
        # F-A + collect rate-1 baseline & per-rep-mean TTFT p50 series for F-B/F-E
        ttft_by_rate_seed: Dict[int, Dict[int, float]] = {}
        for rate in SCAN_RATES:
            for seed in (7, 17, 27):
                row = data.get((seed, rate))
                if row is None:
                    continue
                n_total = len(row.get("ttfts") or [])
                n_completed = row.get("completed", n_total)
                ratio = (n_completed / n_total) if n_total else float("nan")
                flagA = ratio < 1.0
                if flagA:
                    arm_report["F_A"].append({"rate": rate, "seed": seed, "ratio": ratio})
                ttft_by_rate_seed.setdefault(rate, {})[seed] = row.get("median_ttft_ms")
                if rate in main_rates:
                    print(f"  {rate:>5} {seed:>5} {n_completed:>5}/{n_total:<5} "
                          f"{(row.get('median_ttft_ms') or float('nan')):>13.1f} "
                          f"{(row.get('request_throughput') or float('nan')):>9.3f} {str(flagA):>6}")
        # F-B (global): rate-1-baseline multiple>4.0 OR regression slope CI excludes 0
        rate1_vals = list(ttft_by_rate_seed.get(1, {}).values())
        rate1_mean = statistics.mean(rate1_vals) if rate1_vals else None
        fb_flags = []
        if rate1_mean:
            for rate in main_rates:
                vals = list(ttft_by_rate_seed.get(rate, {}).values())
                if not vals:
                    continue
                m = statistics.mean(vals)
                mult = m / rate1_mean
                if mult > 4.0:
                    fb_flags.append({"rate": rate, "multiple_of_rate1": mult})
        # regression slope: TTFT p50 (rep-mean) vs rate index, over main_rates
        xs, ys = [], []
        for i, rate in enumerate(sorted(set(main_rates))):
            vals = list(ttft_by_rate_seed.get(rate, {}).values())
            if vals:
                xs.append(i)
                ys.append(statistics.mean(vals))
        slope_ci_excludes_0 = None
        if len(xs) >= 3 and _scipy_stats is not None:
            res = _scipy_stats.linregress(xs, ys)
            df = len(xs) - 2
            tcrit = t_ppf(0.975, df) if df > 0 else float("nan")
            se = res.stderr if hasattr(res, "stderr") else None
            if se is not None and df > 0:
                lo, hi = res.slope - tcrit * se, res.slope + tcrit * se
                slope_ci_excludes_0 = not (lo <= 0 <= hi)
                arm_report["F_B_slope"] = {"slope": res.slope, "ci_lo": lo, "ci_hi": hi,
                                            "ci_excludes_0": slope_ci_excludes_0}
        arm_report["F_B_rate1_mult"] = fb_flags
        fb_flagged = bool(fb_flags) or bool(slope_ci_excludes_0)
        print(f"  F-B (global): rate1_mult>4.0 flags={fb_flags} slope_ci_excludes_0={slope_ci_excludes_0} "
              f"-> {'FLAGGED' if fb_flagged else 'clear'}")
        # F-E: tercile TTFT p50 ratio (3rd/1st), rep-mean over seeds, threshold 1.7
        # -- computed from the MAIN (Phase 1) scored-rate data by spec ("F-E는
        # 채점 런"), so here we only report the capscan-side F-A/F-B; F-E proper
        # is computed in phase1_report from the n=10 main jsonls.
        report[arm] = arm_report
    return report


def phase0_midcheck(dirpath: str, tag: str, jobid: str, main_rates: List[int]) -> None:
    """PREREG section 5.1.1: A3's X_60 SD across the n=3 capscan seeds vs
    sqrt(2)*SD(A4); if A3's SD > 2x that proxy, the cell is UNDERPOWERED and
    no R3' (equivalence) verdict may be issued for it (R4' superiority is
    still allowed). This is descriptive at capscan n=3 -- a coarse pilot, not
    the n=10 primary."""
    print(f"\n=== Phase 0 mid-check (PREREG section 5.1.1): tag={tag} jobid={jobid} ===")
    a3 = phase0_load(dirpath, tag, jobid, "chunk512")
    a4 = phase0_load(dirpath, tag, jobid, "agnostic")
    for rate in main_rates:
        x3 = []
        x4 = []
        for seed in (7, 17, 27):
            r3 = a3.get((seed, rate))
            r4 = a4.get((seed, rate))
            if r3 is not None:
                v = x_tau(r3, TAU_PRIMARY)["X"]
                if v is not None:
                    x3.append(v)
            if r4 is not None:
                v = x_tau(r4, TAU_PRIMARY)["X"]
                if v is not None:
                    x4.append(v)
        if len(x3) < 2 or len(x4) < 2:
            print(f"  rate={rate}: insufficient capscan data for mid-check (n_A3={len(x3)} n_A4={len(x4)})")
            continue
        sd3 = statistics.stdev(x3)
        sd4 = statistics.stdev(x4)
        proxy = math.sqrt(2) * sd4
        underpowered = sd3 > 2 * proxy
        print(f"  rate={rate}: SD(A3 X_60, n={len(x3)})={sd3:.4f}  sqrt2*SD(A4,n={len(x4)})={proxy:.4f}  "
              f"2x_proxy={2*proxy:.4f}  -> {'UNDERPOWERED (no R3prime for this cell)' if underpowered else 'OK'}")


# ---------------------------------------------------------------------------
# Phase 1 (main measurement)
# ---------------------------------------------------------------------------

def load_voided_reps(dirpath: str, tag: str, jobid: str) -> set:
    path = os.path.join(dirpath, f"g2_{tag}_voided_reps_{jobid}.txt")
    out = set()
    if os.path.exists(path):
        for ln in open(path):
            ln = ln.strip()
            if ln:
                try:
                    out.add(int(ln))
                except ValueError:
                    pass
    return out


def phase1_load(dirpath: str, tag: str, jobid: str, arm: str, rate: int, voided: set) -> Dict[int, dict]:
    out = {}
    for rep in range(1, NREP_MAIN + 1):
        if rep in voided:
            continue
        path = os.path.join(dirpath, f"g2_{tag}_{arm}_rep{rep}_{jobid}.jsonl")
        rows = load_jsonl(path)
        by_rate = {}
        for r in rows:
            rr = r.get("request_rate")
            if rr is not None:
                by_rate[int(rr)] = r
        if rate in by_rate:
            out[rep] = by_rate[rate]
    return out


def cell_x60_series(dirpath: str, tag: str, jobid: str, arm: str, rate: int, voided: set) -> Dict[int, dict]:
    rows = phase1_load(dirpath, tag, jobid, arm, rate, voided)
    return {rep: x_tau(row, TAU_PRIMARY) for rep, row in rows.items()}


def phase1_cell_report(dirpath: str, tag: str, jobid: str, rate: int, voided: set) -> Dict[str, Any]:
    print(f"\n--- rate={rate} ---")
    per_arm_rows: Dict[str, Dict[int, dict]] = {a: phase1_load(dirpath, tag, jobid, a, rate, voided) for a in ARMS}
    per_arm_x60: Dict[str, Dict[int, float]] = {}
    for arm in ARMS:
        xs = {}
        n_missing_total = 0
        for rep, row in per_arm_rows[arm].items():
            r = x_tau(row, TAU_PRIMARY)
            if r["X"] is not None:
                xs[rep] = r["X"]
            n_missing_total += r["n_missing_itl"]
        per_arm_x60[arm] = xs
        vals = list(xs.values())
        if vals:
            print(f"  arm={arm:10s} n={len(vals):2d} X_60 mean={statistics.mean(vals):.4f} "
                  f"reps={sorted(xs.keys())} n_missing_itl_total={n_missing_total} "
                  f"values={['%.4f' % v for v in vals]}")
        else:
            print(f"  arm={arm:10s} NO DATA")

    common_reps = sorted(set(per_arm_x60["chunk512"]) & set(per_arm_x60["agnostic"]))
    diffs = [per_arm_x60["chunk512"][r] - per_arm_x60["agnostic"][r] for r in common_reps]
    print(f"  paired A3-A4 X_60 diffs (n={len(diffs)}, reps={common_reps}): "
          f"{['%.4f' % d for d in diffs]}")

    tost = tost_equivalence(diffs, DELTA)
    raw_ci = paired_t_ci_twosided(diffs)
    perm_p = sign_flip_permutation_p(diffs) if diffs else None

    print(f"  TOST(A3~A4, delta={DELTA}): {tost}")
    print(f"  raw paired-t 95% CI (A3-A4): {raw_ci}")
    print(f"  sign-flip permutation p (H0: mean diff=0): {perm_p}")

    r4_superior = False
    a3_superior = False
    if not raw_ci.get("degenerate") and raw_ci.get("ci_lo") is not None:
        if raw_ci["ci_lo"] > 0:
            r4_superior = True  # A3 worse (higher X_60) than A4 -> A4 superior
        elif raw_ci["ci_hi"] < 0:
            a3_superior = True

    # TTFT p95 non-inferiority (PREREG section 5.2 conjunction for R3')
    ttft_diffs = []
    ttft_common = sorted(set(per_arm_rows["chunk512"]) & set(per_arm_rows["agnostic"]))
    for rep in ttft_common:
        p3 = ttft_p95_ms(per_arm_rows["chunk512"][rep])
        p4 = ttft_p95_ms(per_arm_rows["agnostic"][rep])
        if p3 is not None and p4:
            ttft_diffs.append((p3 - p4) / p4)
    ttft_ci = paired_t_ci_twosided(ttft_diffs) if ttft_diffs else {"n": 0}
    ttft_noninferior = (
        ttft_ci.get("ci_hi") is not None and ttft_ci["ci_hi"] <= 0.10
    )
    print(f"  TTFT p95 relative diff (A3-A4)/A4, n={len(ttft_diffs)}: {ttft_ci} "
          f"-> noninferior(upper<=+10%)={ttft_noninferior}")

    if tost.get("verdict") == "DEGENERATE-T":
        cell_verdict = "DEGENERATE-T"
    elif tost.get("equivalent") and ttft_noninferior:
        cell_verdict = "Rprime3"
    elif r4_superior:
        cell_verdict = "Rprime4"
    elif a3_superior:
        cell_verdict = "A3_SUPERIOR (not in original decision table -- reported)"
    else:
        cell_verdict = "INCONCLUSIVE"
    print(f"  CELL VERDICT (mechanical, pre-Holm) = {cell_verdict}")

    # tau ladder for Delta_chunk = X_tau(A3) - X_tau(A1), sign-only, VACUOUS screen
    ladder = {}
    for tau in TAU_LADDER:
        a1_rows = per_arm_rows["plain"]
        a3_rows = per_arm_rows["chunk512"]
        common = sorted(set(a1_rows) & set(a3_rows))
        d = []
        vacuous_flags = []
        for rep in common:
            x1 = x_tau(a1_rows[rep], tau)
            x3 = x_tau(a3_rows[rep], tau)
            if x1["X"] is None or x3["X"] is None:
                continue
            d.append(x3["X"] - x1["X"])
            both_floor = (x1["n_viol_latency"] == 0) and (x3["n_viol_latency"] == 0)
            both_ceiling = (x1["X"] >= 0.99) and (x3["X"] >= 0.99)
            vacuous_flags.append(both_floor or both_ceiling)
        n_vacuous = sum(vacuous_flags)
        non_vacuous_d = [dd for dd, v in zip(d, vacuous_flags) if not v]
        sign = None
        if non_vacuous_d:
            sign = "+" if statistics.mean(non_vacuous_d) > 0 else ("-" if statistics.mean(non_vacuous_d) < 0 else "0")
        ladder[tau] = {"n": len(d), "n_vacuous": n_vacuous, "mean_diff_nonvacuous": (
            statistics.mean(non_vacuous_d) if non_vacuous_d else None), "sign": sign}
    print(f"  tau ladder Delta_chunk=X_tau(A3)-X_tau(A1) sign (VACUOUS-screened): {ladder}")

    # secondary metrics, both goodput predicates, throughput complement
    secondary: Dict[str, Any] = {}
    for arm in ARMS:
        rows = per_arm_rows[arm]
        if not rows:
            continue
        thpt = [row.get("request_throughput") for row in rows.values() if row.get("request_throughput") is not None]
        gp_canon = [goodput_canonical(row)[0] for row in rows.values()]
        gp_canon = [g for g in gp_canon if g is not None]
        gp_mean = [goodput_mean_itl(row)[0] for row in rows.values()]
        gp_mean = [g for g in gp_mean if g is not None]
        ttft95 = [ttft_p95_ms(row) for row in rows.values() if ttft_p95_ms(row) is not None]
        reqitl_p90 = [request_itl_p95_p90_ms(row) for row in rows.values()]
        reqitl_p90 = [v for v in reqitl_p90 if v is not None]
        secondary[arm] = {
            "n": len(rows),
            "request_throughput_mean": statistics.mean(thpt) if thpt else None,
            "goodput_canonical_mean": statistics.mean(gp_canon) if gp_canon else None,
            "goodput_meanitl_mean": statistics.mean(gp_mean) if gp_mean else None,
            "ttft_p95_mean_ms": statistics.mean(ttft95) if ttft95 else None,
            "request_itl_p95_p90_mean_ms": statistics.mean(reqitl_p90) if reqitl_p90 else None,
        }
        print(f"  secondary arm={arm:10s} {secondary[arm]}")

    return {
        "rate": rate, "n_paired": len(diffs), "diffs": diffs, "tost": tost, "raw_ci": raw_ci,
        "sign_flip_p": perm_p, "ttft_noninferior": ttft_noninferior, "ttft_ci": ttft_ci,
        "cell_verdict": cell_verdict, "tau_ladder": ladder, "secondary": secondary,
        "per_arm_x60": per_arm_x60,
    }


def phase1_fE(dirpath: str, tag: str, jobid: str, rate: int, voided: set) -> Dict[str, Any]:
    """F-E (PREREG section 3): tercile TTFT p50 ratio (3rd/1st), computed
    from the scored (Phase-1, n=10) runs, rep-mean, threshold 1.7. bench_serving
    does not record arrival order beyond request index, so we treat the
    ttfts list's natural index order as arrival order (consistent with
    p1op's / the design doc's treatment elsewhere)."""
    out = {}
    for arm in ARMS:
        rows = phase1_load(dirpath, tag, jobid, arm, rate, voided)
        ratios = []
        for rep, row in rows.items():
            ttfts = [t for t in (row.get("ttfts") or []) if t is not None]
            n = len(ttfts)
            if n < 6:
                continue
            third = n // 3
            t1 = ttfts[:third]
            t3 = ttfts[-third:]
            p1 = percentile(sorted(t1), 0.5)
            p3 = percentile(sorted(t3), 0.5)
            if p1:
                ratios.append(p3 / p1)
        out[arm] = {"n": len(ratios), "rep_mean_ratio": (statistics.mean(ratios) if ratios else None),
                    "flagged": (statistics.mean(ratios) > 1.7) if ratios else None}
    return out


# ---------------------------------------------------------------------------
# section 6: reproducibility diagnostic against 873944/873945 (p1_opint)
# ---------------------------------------------------------------------------

P1OPINT_DIR = "/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/p1_opint"
P1OPINT_JOBID = {"zamba2-27b": "873944", "granite-40-h-micro-base": "873945"}


def repro_diag(tag: str, main_rates: List[int]) -> None:
    ref_jobid = P1OPINT_JOBID.get(tag)
    if ref_jobid is None:
        print(f"\n=== section 6 reproducibility diagnostic: no reference job for tag={tag}, skipping ===")
        return
    print(f"\n=== section 6 reproducibility diagnostic vs {ref_jobid} (X_40, absolute +/-0.10 red-flag) ===")
    print("NOTE (PREREG section 6): Granite rate=3 is NOT a valid reproduction control "
          "(it was not the first rate run in the 873945 boot order) -- interpret that cell's "
          "row with that caveat, do not drop it silently.")
    for arm in ("plain", "agnostic"):
        for rate in main_rates:
            ref_vals = []
            for rep in range(1, 6):
                path = os.path.join(P1OPINT_DIR, f"p1op_{tag}_{arm}_rep{rep}_{ref_jobid}.jsonl")
                rows = load_jsonl(path)
                by_rate = {int(r["request_rate"]): r for r in rows if r.get("request_rate") is not None}
                if rate in by_rate:
                    v = x_tau(by_rate[rate], 40.0)["X"]
                    if v is not None:
                        ref_vals.append(v)
            if not ref_vals:
                continue
            ref_mean = statistics.mean(ref_vals)
            print(f"  arm={arm} rate={rate}: reference(873944/45) X_40 mean(n={len(ref_vals)})={ref_mean:.4f} "
                  f"-- compare against this campaign's own X_40 for the same (arm,rate) once computed; "
                  f"red-flag if |this - ref| > 0.10 absolute.")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--tag", required=True)
    ap.add_argument("--jobid", required=True)
    ap.add_argument("--rates", nargs="+", type=int, required=True)
    ap.add_argument("--designated-rate", type=int, default=None,
                     help="the pre-registered single designated cell for the headline R3' claim "
                          "(section 5.3); e.g. 4 for granite-40-h-micro-base")
    args = ap.parse_args()

    voided = load_voided_reps(args.dir, args.tag, args.jobid)
    if voided:
        print(f"VOIDED REPS (excluded from ALL arms, PREREG section 2.1 contingency): {sorted(voided)}")

    # correctness gates
    corr_path = os.path.join(args.dir, f"g2_{args.tag}_correctness_{args.jobid}.json")
    if os.path.exists(corr_path):
        print(f"\n=== correctness gate summary ({corr_path}) ===")
        for rec in load_jsonl(corr_path):
            print(f"  {rec}")
        print("NOTE: a genuine FAIL (not UNDETERMINED) on G1_PAIRWISE.A1_vs_A3 or on "
              "G2_CONCURRENT voids A3-dependent conclusions (Rprime3/Rprime4) for this model; "
              "A1/A2/A4 data and the A2-vs-A4 (R1'/R2') comparison remain valid regardless.")
    else:
        print(f"\n*** NO correctness-gate artifact found at {corr_path} ***")

    phase0_report(args.dir, args.tag, args.jobid, args.rates)
    phase0_midcheck(args.dir, args.tag, args.jobid, args.rates)
    repro_diag(args.tag, args.rates)

    print(f"\n=== Phase 1 primary (n<={NREP_MAIN} paired, tag={args.tag} jobid={args.jobid}) ===")
    cell_results = {}
    for rate in args.rates:
        cell_results[rate] = phase1_cell_report(args.dir, args.tag, args.jobid, rate, voided)
        fe = phase1_fE(args.dir, args.tag, args.jobid, rate, voided)
        print(f"  F-E (scored-run tercile TTFT-p50 ratio, threshold 1.7): {fe}")
        cell_results[rate]["F_E"] = fe

    # Holm correction across replication cells (designated cell excluded)
    print("\n=== Holm correction across replication cells (designated cell stands alone) ===")
    repl_pvals = []
    for rate, res in cell_results.items():
        if rate == args.designated_rate:
            continue
        p = res["tost"].get("p_tost")
        if p is not None:
            repl_pvals.append((f"{args.tag}_r{rate}", p))
    if repl_pvals:
        adjusted = holm_adjust(repl_pvals)
        for label, praw, padj in adjusted:
            print(f"  {label}: p_raw={praw:.4f} p_holm={padj:.4f} -> "
                  f"{'PASS(<0.05)' if padj < 0.05 else 'fail'}")
    if args.designated_rate is not None and args.designated_rate in cell_results:
        d = cell_results[args.designated_rate]
        print(f"  DESIGNATED cell rate={args.designated_rate}: verdict={d['cell_verdict']} "
              f"(single-cell, no multiplicity correction needed per section 5.3)")

    out_json = os.path.join(args.dir, f"g2_report_{args.tag}_{args.jobid}.json")
    with open(out_json, "w") as f:
        json.dump({"tag": args.tag, "jobid": args.jobid, "voided_reps": sorted(voided),
                    "rates": args.rates, "designated_rate": args.designated_rate,
                    "cells": cell_results}, f, indent=2, default=str)
    print(f"\nFull report -> {out_json}")
    print("\n*** This script computes measurements and the mechanical pre-registered rule ONLY. ***")
    print("*** Final interpretation/headline claims are result-analyst / claims-auditor's job. ***")
    return 0


if __name__ == "__main__":
    sys.exit(main())
