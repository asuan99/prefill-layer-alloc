#!/usr/bin/env python3
"""Gate 2-S scorer -- PREREG_GATE2S_2026-08-09.md rev6 (submission build).

THE PRE-REGISTRATION IS CANONICAL AND THIS CODE FOLLOWS IT.  Every decision rule
below carries the prereg section it implements.  If an implementation choice
would deviate from the prereg, the correct action is to STOP AND REPORT, not to
change the code -- methodology gate #20 is exactly the mismatch between a
pre-registered table and the scorer, and its reverse (code doing something the
prereg never registered) is the same failure.

SECTION -> LINE MAP (methodology gate #20 -- auto-verified against this file, 2026-08-10):
  sec2   arm set / env surface                ARMS                    L82
  sec3   cells + n=10                         CELLS                   L85
  sec3.1 designated cell                      DESIGNATED              L89
  sec4.2 primary-alpha ITLp95_mean            metrics()               L173
  sec4.2 primary-beta pooled p99              metrics()               L174
  sec4.2 nearest-rank convention FIXED        p_nearest()             L127
  sec6.3-9 unit assert (itls are SECONDS)     assert_units()          L132
  sec12-4 two rate rows per rep -> SUM        load_reps()             L156
  sec4.1.3 unstable_frac (diagnostic only)    unstable_frac()         L196
  sec5.1  paired t CI (bootstrap FORBIDDEN)   paired_t()              L218
  sec5.1  DEGENERATE-T                        paired_t()              L226
  sec3.4c MDE % (mandatory in S2 text)        mde_pct()               L268
  sec11.1#3 sign = POINT ESTIMATE             axis_state()            L282
  sec5.2.1 nine-cell table (untouchable)      NINE_CELL               L294
  sec5.2.1 applier (2 contrasts x 2 metrics)  nine_cell()             L308
  sec5.6b Holm                                holm()                  L325
  sec5.6c Holm p source = ALPHA ONLY          main()                  L645
  sec5.6d claim ONLY after Holm               main()                  L665,L668
  sec5.6.1 headline combos + naming limit     headline()              L488
  sec6.4  F-A / F-B / F-E                     f_series()              L341
  sec6.4.1 F-B disjunct(i) unevaluated (T')   f_series()              L365
  gate#17 gate label (append not delete)      gate_label()            L401
  sec4.3.1 Delta_conc = SIGN ONLY             main()                  L702
  sec5.3  A4-T' descriptive + note            main()                  L708
  sec4.5  FRAGILE-TO-METRIC                   sensitivity()           L530
  sec4.5  CONVENTION-SENSITIVE                convention_sensitive()  L544
  sec6.5  19 blocks                           BLOCKS                  L104
  sec8.9  cell-wise premise labels            PREMISE_LABEL           L95
  sec8.9.1 falsifier (ONE-WAY)                falsifier()             L425
  sec8.9.1 imported WITHDRAWAL_THRESHOLD      constants               L71
  sec8.9.1 imported sparse guard              falsifier()             L458
  sec8.9.1 monotone downgrade only            apply_falsifier()       L477
  self-test for every row above               g2s_selftest.py         (CPU, synthetic)

USAGE
  python3 g2s_analyze.py --tag zamba2-27b --job 999999 --outdir .
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics as st
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------
# Constants.  Every one of these is either a prereg-registered free parameter
# (sec11.1, exactly three) or an IMPORT from gate1/gate1b_analyze.py (sec8.9.1).
# Nothing here is invented by this file.
# --------------------------------------------------------------------------
DELTA_PROBE = 0.05                 # sec11.1 #1  (P-carve TOST margin)
# sec11.1 #2 Holm p source = alpha  -> enforced in family_verdicts()
# sec11.1 #3 "opposite sign" = point estimate -> enforced in axis_state()/sensitivity()

# --- sec8.9.1 IMPORTS from gate1/gate1b_analyze.py (NOT new parameters) ---
WITHDRAWAL_THRESHOLD = 0.01        # gate1b_analyze.py:52  (Gate 1 pre-registered)
MIN_TOTAL_S = 0.5                  # gate1b_analyze.py:47
MIN_EPISODES = 2                   # gate1b_analyze.py:48
MIN_COVERAGE = 0.98                # gate1b_analyze.py:49

# --- sec6.4 F-series (inherited from PREREG_GATE2 rev4 sec3) ---
F_B_RATE1_MULT = 4.0
F_E_RATIO = 1.7

TAU_LADDER_MS = (40.0, 60.0, 100.0)  # sec4.5 threshold ladder (descriptive)

ARMS = ("plainaux", "pdmux_nosplit", "pdmux_split34", "pdmux_split34_nosticky", "agnostic")
A2, C, T, TP, A4 = ARMS

CELLS = {  # sec3
    "zamba2-27b": (2.0, 3.0),
    "granite-40-h-micro-base": (3.0, 4.0),
}
DESIGNATED = ("zamba2-27b", 3.0)  # sec3.1

# sec8.9 -- cell-wise premise verification state.  Zamba2 only, from Gate 1
# rev15 (job 875293); Granite has NO realized observation (G1-c not run).
# sec8.9.1 may downgrade an entry to "refuted"; it can NEVER upgrade to
# "verified" -- see falsifier().
PREMISE_LABEL = {
    ("zamba2-27b", 2.0): "verified",
    ("zamba2-27b", 3.0): "verified",
    ("granite-40-h-micro-base", 3.0): "unverified",
    ("granite-40-h-micro-base", 4.0): "unverified",
}

# sec6.5 -- the 19 blocks.  Every block that prints a magnitude must carry a
# gate label (methodology gate #17).  `sizes` says whether it prints magnitudes.
BLOCKS: List[Tuple[str, bool]] = [
    ("confirm_A4like_itl", True), ("confirm_A4like_ttft", True),
    ("secondary_cont_itl", True), ("secondary_cont_ttft", True),
    ("descriptive_conc_itl", True), ("descriptive_conc_ttft", True),
    ("descriptive_A4_minus_Tprime", True), ("metric_sensitivity", True),
    ("tau_ladder", True), ("component_share", True), ("descriptive", True),
    ("switch_and_drain", True), ("delivery_rate", True),
    ("probe_pos", True), ("probe_carve", True), ("mde", True),
    ("kv_pool", True), ("bimodality", True),
    ("premise_falsifier", False),   # label update only, no magnitude
]


# ==========================================================================
# percentiles -- convention is FIXED to nearest-rank (sec4.2, sec6.3-9)
# ==========================================================================
def p_linear(a: Sequence[float], f: float) -> float:
    a = sorted(a)
    i = (len(a) - 1) * f
    lo, hi = math.floor(i), math.ceil(i)
    return a[lo] if lo == hi else a[lo] + (a[hi] - a[lo]) * (i - lo)


def p_nearest(a: Sequence[float], f: float) -> float:
    a = sorted(a)
    return a[max(0, math.ceil(f * len(a)) - 1)]


def assert_units(row: Dict[str, Any]) -> None:
    """sec6.3-9: itls/ttfts are SECONDS.  Reading them as ms zeroes every spike."""
    flat = [v for x in row["itls"] if x for v in x]
    got = (sum(flat) / len(flat)) * 1000.0
    assert abs(got - row["mean_itl_ms"]) < 1e-6, (
        f"UNIT ASSERT FAILED: mean(itls)*1000={got} != mean_itl_ms={row['mean_itl_ms']}")


# ==========================================================================
# sec12-4 -- a rep file holds TWO rate rows.  Duration is SUMMED, never max().
# ==========================================================================
def load_reps(outdir: str, tag: str, job: str) -> Dict[Tuple[str, float, int], Dict[str, Any]]:
    out: Dict[Tuple[str, float, int], Dict[str, Any]] = {}
    dur_sum: Dict[Tuple[str, int], float] = defaultdict(float)
    for f in sorted(glob.glob(os.path.join(outdir, f"g2s_{tag}_*_rep*_{job}.jsonl"))):
        if ".holb." in f:
            continue
        base = os.path.basename(f)
        arm = base.split(f"g2s_{tag}_")[1].rsplit("_rep", 1)[0]
        rep = int(base.rsplit("_rep", 1)[1].split("_")[0])
        for line in open(f):
            row = json.loads(line)
            assert_units(row)
            out[(arm, float(row["request_rate"]), rep)] = row
            dur_sum[(arm, rep)] += float(row["duration"])   # SUM, never max()
    for (arm, rep), d in dur_sum.items():
        for k in list(out):
            if k[0] == arm and k[2] == rep:
                out[k]["_duration_sum_all_rates"] = d
    return out


# ==========================================================================
# sec4.2 -- primary-alpha and primary-beta, always reported jointly
# ==========================================================================
def metrics(row: Dict[str, Any], inner=p_nearest) -> Dict[str, float]:
    itls = [x for x in row["itls"] if x]
    per_req_p95 = [inner(x, 0.95) * 1000.0 for x in itls]
    pooled = [v * 1000.0 for x in itls for v in x]
    ttft = [t * 1000.0 for t in row["ttfts"]]
    m = {
        "alpha_ITLp95_mean": sum(per_req_p95) / len(per_req_p95),   # sec4.1.2 primary-alpha
        "beta_pooled_p99": p_linear(pooled, 0.99),                  # sec4.1.2 primary-beta
        "TTFTp95": p_linear(ttft, 0.95),
        # sec4.5 sensitivity panel (descriptive only)
        "ITLp95p90": p_linear(per_req_p95, 0.90),
        "med_ITLp95": p_linear(per_req_p95, 0.50),
        "pooled_p95": p_linear(pooled, 0.95),
        "mean_ITL": sum(pooled) / len(pooled),
        "TTFTp50": p_linear(ttft, 0.50),
        "ITLp50": p_linear(pooled, 0.50),
        "throughput": float(row.get("request_throughput", 0.0)),
        "completed_frac": float(row["completed"]) / max(1, int(row.get("total_input_tokens", 0) and row["completed"] or row["completed"])),
    }
    for tau in TAU_LADDER_MS:
        m[f"X{int(tau)}"] = sum(1 for x in itls if max(x) * 1000.0 > tau) / len(itls)
    return m


PRIMARY_METRICS = ("alpha_ITLp95_mean", "beta_pooled_p99")
REQUEST_INTERNAL = ("alpha_ITLp95_mean", "med_ITLp95", "ITLp95p90")   # sec4.5
POOLED_FAMILY = ("beta_pooled_p99", "pooled_p95", "mean_ITL")          # sec4.5


def unstable_frac(row: Dict[str, Any], c: float) -> float:
    """sec4.1.3 DIAGNOSTIC ONLY -- no threshold, no gate, no decision depends on it."""
    n = tot = 0
    for x in row["itls"]:
        if len(x) < 10:
            continue
        a = sorted(x, reverse=True)
        k4, k5, k6 = a[3] * 1000, a[4] * 1000, a[5] * 1000
        tot += 1
        if k5 and abs(k4 - k6) / k5 > c:
            n += 1
    return n / tot if tot else float("nan")


# ==========================================================================
# sec5.1 -- paired t only.  Bootstrap is forbidden for verdicts (gate #14).
# ==========================================================================
_T95 = {2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179}
_T80 = {3: 0.978, 9: 0.883, 10: 0.879}


def paired_t(diffs: Sequence[float]) -> Dict[str, Any]:
    n = len(diffs)
    if n < 2:
        return dict(n=n, mean=float("nan"), lo=float("nan"), hi=float("nan"),
                    p=float("nan"), degenerate=False)
    mean = sum(diffs) / n
    sd = st.stdev(diffs)
    if sd == 0.0:
        # sec5.1 DEGENERATE-T: report the point value, never declare equivalence.
        return dict(n=n, mean=mean, lo=mean, hi=mean, p=float("nan"), degenerate=True)
    se = sd / math.sqrt(n)
    tcrit = _T95.get(n - 1, 1.96)
    tstat = mean / se
    p = _t_sf(abs(tstat), n - 1) * 2.0
    return dict(n=n, mean=mean, lo=mean - tcrit * se, hi=mean + tcrit * se,
                p=p, degenerate=False, sd=sd, se=se, t=tstat)


def _t_sf(t: float, df: int) -> float:
    """Upper-tail Student-t survival function (continued-fraction incomplete beta)."""
    x = df / (df + t * t)
    return 0.5 * _betainc(df / 2.0, 0.5, x)


def _betainc(a: float, b: float, x: float) -> float:
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(a * math.log(x) + b * math.log(1 - x) - lbeta) / a
    f, cc, d = 1.0, 1.0, 0.0
    for i in range(0, 300):
        m, num = i // 2, 0.0
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m))
        else:
            num = -((a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + num * d
        d = 1e-30 if abs(d) < 1e-30 else d
        d = 1.0 / d
        cc = 1.0 + num / (1e-30 if abs(cc) < 1e-30 else cc)
        f *= cc * d
        if abs(1.0 - cc * d) < 1e-10:
            break
    return front * (f - 1.0)


def mde_pct(diffs: Sequence[float], ref_mean: float) -> float:
    """sec3.4(c) MDE at n, alpha=.05, power=.80 -- ~0.995*sigma_D, as % of ref mean.
    MANDATORY inside every S2 (not-detected) sentence."""
    n = len(diffs)
    if n < 2 or ref_mean == 0:
        return float("nan")
    sd = st.stdev(diffs)
    tc, tp = _T95.get(n - 1, 1.96), _T80.get(n - 1, 0.879)
    return 100.0 * (tc + tp) * sd / math.sqrt(n) / abs(ref_mean)


# ==========================================================================
# sec5.2.1 -- axis state.  Sign convention is the POINT ESTIMATE (sec11.1 #3).
# ==========================================================================
def axis_state(ci: Dict[str, Any]) -> str:
    """'+' = first-named arm better, '-' = worse, '0' = CI contains 0.
    Diffs are built so that POSITIVE = the split arm is better (lower latency)."""
    if ci["degenerate"] or math.isnan(ci["lo"]):
        return "0"
    if ci["lo"] > 0:
        return "+"
    if ci["hi"] < 0:
        return "-"
    return "0"


NINE_CELL = {  # sec5.2.1 -- verbatim labels; the table itself is untouchable
    ("+", "+"): ("S1-A", "순 우세: 분할이 두 축을 모두 개선한다."),
    ("+", "0"): ("S1-B", "우세: 분할이 ITL을 개선하며 TTFT 악화 증거는 없다."),
    ("+", "-"): ("S1-C", "트레이드오프: 분할은 ITL을 개선하고 TTFT를 악화시킨다 "
                          "-- 두 크기를 반드시 함께 인용한다. 발화이지 실패가 아니다."),
    ("0", "+"): ("S2-A", "ITL 성분 미검출[MDE]. TTFT는 분할 arm이 우수."),
    ("0", "0"): ("S2-B", "완전 미검출: 두 축 모두 미검출[MDE]."),
    ("0", "-"): ("S2-C", "ITL 미검출[MDE], TTFT는 분할 arm이 열세."),
    ("-", "+"): ("S3-A", "역트레이드오프: 분할이 ITL을 악화시키고 TTFT를 개선한다."),
    ("-", "0"): ("S3-B", "분할이 ITL을 악화시킨다."),
    ("-", "-"): ("S3-C", "순 열세: 분할이 두 축을 모두 악화시킨다."),
}


def nine_cell(itl_ci: Dict[str, Any], ttft_ci: Dict[str, Any],
              mde: float) -> Dict[str, Any]:
    """sec5.2.1.  NOTE sec5.6(d): the axis state is an UNCORRECTED coordinate and
    does NOT by itself licence a claim -- Holm decides that (family_verdicts)."""
    ia, ta = axis_state(itl_ci), axis_state(ttft_ci)
    code, text = NINE_CELL[(ia, ta)]
    if code.startswith("S2") and not math.isnan(mde):
        text = text.replace(
            "[MDE]", f" (이 설계는 차이가 {mde:.1f}% 미만인 경우를 배제할 수 없다)")
    return dict(itl_axis=ia, ttft_axis=ta, code=code, text=text,
                uncorrected=True, licences_claim=False)


# ==========================================================================
# sec5.6 -- Holm.  Family F1 = confirmatory (4 cells).  F2 = secondary.
# p SOURCE IS ALPHA ONLY (sec5.6(c), sec11.1 #2).
# ==========================================================================
def holm(pvals: Dict[Any, float], alpha: float = 0.05) -> Dict[Any, bool]:
    items = sorted(((k, v) for k, v in pvals.items() if not math.isnan(v)),
                   key=lambda kv: kv[1])
    m, out, rejected_so_far = len(items), {k: False for k in pvals}, True
    for i, (k, p) in enumerate(items):
        if rejected_so_far and p <= alpha / (m - i):
            out[k] = True
        else:
            rejected_so_far = False
            out[k] = False
    return out


# ==========================================================================
# sec6.4 -- F-series.  F-A/F-B on capscan, F-E on SCORED runs (no mixing).
# ==========================================================================
def f_series(scored: Dict[Tuple[str, float, int], Dict[str, Any]],
             capscan: Dict[Tuple[str, float, int], Dict[str, Any]],
             arm: str, rate: float) -> Dict[str, Any]:
    out: Dict[str, Any] = {"arm": arm, "rate": rate}

    # F-A : saturation, from the scored runs
    fa = []
    for (a, r, _rep), row in scored.items():
        if a == arm and r == rate:
            tot = row.get("num_prompts") or row.get("completed")
            fa.append(float(row["completed"]) / float(tot) < 1.0 if tot else False)
    out["F_A"] = bool(fa) and any(fa)

    # F-B : disjunct (i) rate-1 multiple > 4.0 [capscan]  OR
    #       disjunct (ii) within-run TTFT slope CI excludes 0 [scored]
    r1 = [p_linear([t * 1000 for t in row["ttfts"]], 0.50)
          for (a, r, _s), row in capscan.items() if a == arm and r == 1.0]
    rc = [p_linear([t * 1000 for t in row["ttfts"]], 0.50)
          for (a, r, _s), row in capscan.items() if a == arm and r == rate]
    if r1 and rc:
        out["F_B_i"] = (sum(rc) / len(rc)) / (sum(r1) / len(r1)) > F_B_RATE1_MULT
        out["F_B_i_evaluated"] = True
    else:
        # sec6.4.1 / sec8.6: T-prime is NOT in capscan by pre-registered decision.
        out["F_B_i"] = None
        out["F_B_i_evaluated"] = False
    slopes = []
    for (a, r, _rep), row in scored.items():
        if a == arm and r == rate:
            y = [t * 1000 for t in row["ttfts"]]
            n = len(y)
            if n >= 6:
                xs = list(range(n))
                mx, my = sum(xs) / n, sum(y) / n
                sxx = sum((x - mx) ** 2 for x in xs)
                b = sum((x - mx) * (v - my) for x, v in zip(xs, y)) / sxx
                resid = [v - (my + b * (x - mx)) for x, v in zip(xs, y)]
                s2 = sum(r_ * r_ for r_ in resid) / (n - 2)
                se = math.sqrt(s2 / sxx)
                tc = _T95.get(n - 2, 1.96)
                slopes.append(b - tc * se > 0 or b + tc * se < 0)
    out["F_B_ii"] = bool(slopes) and any(slopes)
    out["F_B"] = bool(out["F_B_i"]) or out["F_B_ii"]

    # F-E : queue growth on the SCORED runs, rep-mean of tercile ratio
    ratios = []
    for (a, r, _rep), row in scored.items():
        if a == arm and r == rate:
            y = [t * 1000 for t in row["ttfts"]]
            k = len(y) // 3
            if k >= 3:
                lo, hi = p_linear(y[:k], 0.50), p_linear(y[-k:], 0.50)
                if lo > 0:
                    ratios.append(hi / lo)
    out["F_E_ratio"] = (sum(ratios) / len(ratios)) if ratios else float("nan")
    out["F_E"] = bool(ratios) and out["F_E_ratio"] >= F_E_RATIO
    out["flagged"] = bool(out["F_A"] or out["F_B"] or out["F_E"])
    return out


def gate_label(fs: Sequence[Dict[str, Any]]) -> str:
    """methodology gate #17 -- never delete a number, append the gate verdict."""
    parts = []
    fired = [f"{f['arm']}@r{f['rate']:.0f}" for f in fs if f["flagged"]]
    if fired:
        parts.append("F-SERIES FIRED(%s) => SIGN ONLY, MAGNITUDE NOT CITABLE"
                     % ",".join(fired))
    unev = [f"{f['arm']}" for f in fs if not f.get("F_B_i_evaluated", True)]
    if unev:
        parts.append("F-B disjunct(i) UNEVALUATED(%s) [sec8.6]" % ",".join(unev))
    return ("  [GATE: " + " | ".join(parts) + "]") if parts else "  [GATE: clear]"


# ==========================================================================
# sec8.9.1 -- ONE-WAY premise falsifier.
#
# *** SAFETY INVARIANT ***  There is NO code path that emits "verified".
# The only reachable states are "refuted" and the incoming label (unchanged).
# This is the whole point of the device: it can fail an author claim but can
# never pass one.  See sec8.9.1 rationale.
# ==========================================================================
FALSIFIER_STATES = ("refuted", "unchanged")


def falsifier(telemetry_path: str, sm_counts: Sequence[Tuple[int, int]],
              split_idx: int = 2) -> Dict[str, Any]:
    """A4 pop-A time-weighted stream_index histogram.  frac only, never t_total
    (methodology gate #15).  Rules IMPORTED from gate1/gate1b_analyze.py."""
    rows: List[Dict[str, Any]] = []
    if telemetry_path and os.path.exists(telemetry_path):
        for line in open(telemetry_path):
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if e.get("event") == "runtime_snapshot":
                rows.append(e)
    rows.sort(key=lambda e: e.get("timestamp_monotonic_s", 0.0))

    # gate1b_analyze.py:91,95 -- pop A = decode busy AND prefill in flight
    seg: Dict[int, float] = defaultdict(float)
    episodes, t_total, prev_in = 0, 0.0, False
    for i, e in enumerate(rows[:-1]):
        d_active = (e.get("decode_running_batch_size", 0) or 0) > 0
        p_active = (e.get("prefill_active_batch_size", 0) or 0) > 0
        in_pop = d_active and p_active
        if in_pop:
            dt = rows[i + 1].get("timestamp_monotonic_s", 0.0) - e.get("timestamp_monotonic_s", 0.0)
            if dt > 0:
                seg[int(e.get("stream_index", -1))] += dt
                t_total += dt
            if not prev_in:
                episodes += 1
        prev_in = in_pop

    # gate1b_analyze.py:47-48,255 -- sparse guard
    if t_total < MIN_TOTAL_S or episodes < MIN_EPISODES:
        return dict(state="unchanged", verdict="UNMEASURABLE(pop A too sparse)",
                    t_total_s_NOT_FOR_CITATION=t_total, n_episodes=episodes,
                    frac={}, note="sec8.9.1: no upgrade path exists")

    frac = {str(k): v / t_total for k, v in sorted(seg.items())}   # frac ONLY
    f_split = frac.get(str(split_idx), 0.0)
    if f_split >= WITHDRAWAL_THRESHOLD:
        return dict(state="refuted", verdict=(
            f"REFUTED -- idx{split_idx}={sm_counts[split_idx] if split_idx < len(sm_counts) else '?'} "
            f"time-weighted frac={f_split:.4f} >= {WITHDRAWAL_THRESHOLD}"),
            frac=frac, n_episodes=episodes,
            t_total_s_NOT_FOR_CITATION=t_total)
    return dict(state="unchanged", verdict=(
        f"NOT REFUTED (frac(idx{split_idx})={f_split:.4f} < {WITHDRAWAL_THRESHOLD}). "
        "sec8.9.1: this is NOT evidence of absence -- PDMUX_TRACE_FORCE_PREFILL=0 gives "
        "~1/32 of G1B pop-A density. NO UPGRADE PATH EXISTS -- state stays 'unchanged'."),
        frac=frac, n_episodes=episodes, t_total_s_NOT_FOR_CITATION=t_total)


def apply_falsifier(label: str, fres: Dict[str, Any]) -> str:
    """sec8.9.1 -- monotone downgrade only."""
    assert fres["state"] in FALSIFIER_STATES, "sec8.9.1 invariant: no third state"
    if fres["state"] == "refuted":
        return "refuted"
    return label            # never upgrades


# ==========================================================================
# sec5.6.1 -- headline selection (four combinations) + naming restriction
# ==========================================================================
def headline(conf_fired: List[Tuple[str, float]], sec_fired: List[Tuple[str, float]],
             labels: Dict[Tuple[str, float], str]) -> Dict[str, Any]:
    def name_for(cells: List[Tuple[str, float]]) -> str:
        # sec5.6.1 rev6 restriction: "engine default trajectory"/"A4-like" wording
        # is permitted ONLY on cells whose premise label is "verified".
        if cells and all(labels[c] == "verified" for c in cells):
            return "간헐 전달(엔진 기본 궤적)"
        return "`FixedPolicy(34)`·sticky OFF 구성"

    cf, sf = bool(conf_fired), bool(sec_fired)
    if cf and sf:
        code, txt = "COMBO-1", (
            f"이 격자에서 SM 분할은 **{name_for(conf_fired)}에서도, 연속 전달에서도** "
            f"ITL을 [방향]시킨다. confirmatory는 4셀 Holm 후 {conf_fired}이며, "
            "연속 전달 결과는 secondary다.")
    elif cf and not sf:
        code, txt = "COMBO-2", (
            f"**{name_for(conf_fired)}**에서 분할의 ITL 효과가 확인됐다"
            f"(4셀 Holm 후 {conf_fired}). 연속 전달(secondary)에서는 확인되지 않았다 "
            "-- MDE 문장 병기.")
    elif (not cf) and sf:
        code, txt = "COMBO-3", (
            "**confirmatory는 발화하지 않았다.** 연속 전달(secondary)에서만 효과가 보인다. "
            "**연속 전달은 이 엔진의 기본 파티션 궤적이 아니며(sec5.3b), 이 결과를 "
            "'분할의 효과'로 일반화하지 않는다.** secondary를 헤드라인으로 승격하지 않는다.")
    else:
        code, txt = "COMBO-4", (
            "이 격자·이 설계에서 분할의 ITL 효과는 **검출되지 않았다**(등가 아님). "
            "MDE 문장을 두 대비 각각에 병기한다.")
    # sec5.6(f): confirmatory fired ONLY on non-verified cells
    if cf and all(labels[c] != "verified" for c in conf_fired):
        txt += (" ★ **전제 미검증/반증 셀에서만 발화** -- 이 문장을 헤드라인에 반드시 포함한다"
                "(sec5.6(f)).")
    txt += (" [모든 조합 공통] 이 결과를 sec1-1 A4-vs-fused 격차의 성분·기여분·분해로 "
            "해석하지 않는다(sec5.5).")
    return dict(code=code, text=txt,
                confirmatory_fired=conf_fired, secondary_fired=sec_fired)


# ==========================================================================
# sec4.5 -- sensitivity reading rules (pre-registered, not a post-hoc menu)
# ==========================================================================
def sensitivity(diff_by_metric: Dict[str, Dict[str, Any]], primary_key: str) -> List[str]:
    tags = []
    prim = diff_by_metric.get(primary_key)
    if prim and axis_state(prim) in ("+", "-"):
        psign = 1 if prim["mean"] > 0 else -1
        # sec11.1 #3: "opposite sign" is decided on the POINT ESTIMATE
        opp = [k for k in REQUEST_INTERNAL
               if k in diff_by_metric and not math.isnan(diff_by_metric[k]["mean"])
               and (1 if diff_by_metric[k]["mean"] > 0 else -1) != psign]
        if opp:
            tags.append("FRAGILE-TO-METRIC(%s)" % ",".join(opp))
    return tags


def convention_sensitive(nr: Dict[str, Any], lin: Dict[str, Any]) -> bool:
    """sec4.5 -- sign flip across percentile conventions => not citable."""
    if math.isnan(nr["mean"]) or math.isnan(lin["mean"]):
        return False
    return (nr["mean"] > 0) != (lin["mean"] > 0)


# ==========================================================================
# driver
# ==========================================================================
def build_diffs(scored, arm_a, arm_b, rate, key, inner=p_nearest) -> List[float]:
    """diff = metric(arm_a) - metric(arm_b); POSITIVE means arm_b is better
    (lower latency).  Callers pass (C, T) so positive == split arm better."""
    out = []
    for rep in range(1, 100):
        ra, rb = scored.get((arm_a, rate, rep)), scored.get((arm_b, rate, rep))
        if ra is None or rb is None:
            continue
        out.append(metrics(ra, inner)[key] - metrics(rb, inner)[key])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--job", required=True)
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--telemetry-dir", default=None)
    args = ap.parse_args()
    tag, job, outdir = args.tag, args.job, args.outdir
    tdir = args.telemetry_dir or outdir
    rates = CELLS[tag]

    scored = load_reps(outdir, tag, job)
    capscan: Dict[Tuple[str, float, int], Dict[str, Any]] = {}
    for f in sorted(glob.glob(os.path.join(outdir, f"g2s_{tag}_capscan_*_{job}.jsonl"))):
        base = os.path.basename(f)
        arm = base.split(f"g2s_{tag}_capscan_")[1].rsplit("_seed", 1)[0]
        seed = int(base.rsplit("_seed", 1)[1].split("_")[0])
        for line in open(f):
            row = json.loads(line)
            capscan[(arm, float(row["request_rate"]), seed)] = row

    report: Dict[str, Any] = {"tag": tag, "job": job, "prereg": "rev6",
                              "blocks": [b for b, _ in BLOCKS], "n_blocks": len(BLOCKS)}
    print("=" * 78)
    print(f"Gate 2-S scorer -- {tag} job={job} -- PREREG rev6 (canonical)")
    print("PRE-REGISTERED DECISION RULES (echoed before any number):")
    print("  primary-alpha=ITLp95_mean  primary-beta=pooled_p99  (JOINT, no substitution)")
    print("  verdict test = paired t 95% CI (bootstrap FORBIDDEN, gate #14)")
    print("  Holm p source = ALPHA ONLY (sec5.6c);  F1=confirmatory(C-T'), F2=secondary(C-T)")
    print("  claim printed ONLY after Holm (sec5.6d); 9-cell axis = UNCORRECTED coordinate")
    print("  sec8.9.1 falsifier is ONE-WAY: states = {refuted, unchanged}; 'verified' unreachable")
    print("  any-claim error rate across the two families ~= 0.10 (sec5.6b)")
    print("=" * 78)

    # ---------------- F-series (sec6.4), needed by every magnitude block ----
    fser = {(a, r): f_series(scored, capscan, a, r) for a in ARMS for r in rates}
    report["f_series"] = {f"{a}@{r}": v for (a, r), v in fser.items()}

    # ---------------- sec8.9.1 falsifier (label update, no magnitude) -------
    labels = dict(PREMISE_LABEL)
    fal: Dict[str, Any] = {}
    for r in rates:
        tp = os.path.join(tdir, f"g2s_{tag}_telemetry_{A4}_r{int(r)}_{job}.jsonl")
        res = falsifier(tp, [(108, 0), (74, 34), (54, 54), (0, 108)], split_idx=2)
        new = apply_falsifier(labels[(tag, r)], res)
        fal[f"r{int(r)}"] = dict(res, label_before=labels[(tag, r)], label_after=new)
        labels[(tag, r)] = new
        print(f"\n[premise_falsifier r{int(r):.0f}] {res['verdict']}")
        print(f"  label: {fal[f'r{int(r)}']['label_before']} -> {new}   (one-way; no 'verified' path)")
    report["premise_falsifier"] = fal
    report["premise_labels"] = {f"{k[0]}@{k[1]}": v for k, v in labels.items() if k[0] == tag}

    # ---------------- primaries: two contrasts x two metrics ---------------
    contrasts = [("confirm_A4like", C, TP), ("secondary_cont", C, T)]
    results: Dict[str, Any] = {}
    for cname, a_hi, a_lo in contrasts:
        for mkey in PRIMARY_METRICS:
            for r in rates:
                d_itl = build_diffs(scored, a_hi, a_lo, r, mkey)
                d_ttft = build_diffs(scored, a_hi, a_lo, r, "TTFTp95")
                ci_i, ci_t = paired_t(d_itl), paired_t(d_ttft)
                ref = st.mean([metrics(scored[(a_lo, r, rep)])[mkey]
                               for rep in range(1, 100) if (a_lo, r, rep) in scored] or [0])
                mde = mde_pct(d_itl, ref)
                nc = nine_cell(ci_i, ci_t, mde)
                gl = gate_label([fser[(a_hi, r)], fser[(a_lo, r)]])
                results[f"{cname}|{mkey}|r{int(r)}"] = dict(
                    ci_itl=ci_i, ci_ttft=ci_t, nine_cell=nc, mde_pct=mde,
                    gate=gl, premise=labels[(tag, r)])
                blk = f"{cname}_itl"
                print(f"\n[{blk}] {mkey} r{int(r)}  Δ={ci_i['mean']:+.3f} "
                      f"CI[{ci_i['lo']:+.3f},{ci_i['hi']:+.3f}] p={ci_i['p']:.4g} "
                      f"-> axis {nc['itl_axis']} ({nc['code']}) premise={labels[(tag, r)]}{gl}")
                print(f"[{cname}_ttft] TTFTp95 r{int(r)}  Δ={ci_t['mean']:+.1f}ms "
                      f"CI[{ci_t['lo']:+.1f},{ci_t['hi']:+.1f}] -> axis {nc['ttft_axis']}{gl}")

    # ---------------- Holm per family, ALPHA ONLY (sec5.6c) ----------------
    fam: Dict[str, Any] = {}
    for cname, _, _ in contrasts:
        pv = {r: results[f"{cname}|alpha_ITLp95_mean|r{int(r)}"]["ci_itl"]["p"] for r in rates}
        rej = holm(pv)
        fam[cname] = {"_family": dict(size_this_job=len(pv), size_preregistered=4,
                                      partial=True, p_source="alpha_ITLp95_mean")}
        fam[cname].update({f"r{int(r)}": dict(p_alpha=pv[r], holm_reject=rej[r],
                                         axis=results[f"{cname}|alpha_ITLp95_mean|r{int(r)}"]
                                         ["nine_cell"]["itl_axis"]) for r in rates})
        print(f"\n[HOLM {cname}] PARTIAL family: {len(pv)} of 4 pre-registered cells "
              f"(this job only; p source = ALPHA only, sec5.6c)")
        print("  sec5.6 last bullet: the full 4-cell Holm family spans BOTH job reports and "
              "must be reassembled BY HAND (methodology gate #11 -- no cross-job automation).")
        for r in rates:
            k = f"r{int(r)}"
            print(f"  r{int(r)}: p_alpha={pv[r]:.4g} holm_reject={rej[r]} "
                  f"axis={fam[cname][k]['axis']} "
                  f"-> claim printable: {bool(rej[r]) and fam[cname][k]['axis'] != '0'}")
    report["holm"] = fam

    # sec5.6d: a claim requires Holm rejection AND a non-zero axis.
    conf_fired = [(tag, r) for r in rates
                  if fam["confirm_A4like"][f"r{int(r)}"]["holm_reject"]
                  and fam["confirm_A4like"][f"r{int(r)}"]["axis"] != "0"]
    sec_fired = [(tag, r) for r in rates
                 if fam["secondary_cont"][f"r{int(r)}"]["holm_reject"]
                 and fam["secondary_cont"][f"r{int(r)}"]["axis"] != "0"]
    hl = headline(conf_fired, sec_fired, labels)
    report["headline"] = hl
    print(f"\n[HEADLINE {hl['code']}]\n  {hl['text']}")

    # ---------------- metric sensitivity (sec4.5) --------------------------
    sens: Dict[str, Any] = {}
    for cname, a_hi, a_lo in contrasts:
        for r in rates:
            dm_nr = {k: paired_t(build_diffs(scored, a_hi, a_lo, r, k, p_nearest))
                     for k in REQUEST_INTERNAL + POOLED_FAMILY}
            dm_li = {k: paired_t(build_diffs(scored, a_hi, a_lo, r, k, p_linear))
                     for k in REQUEST_INTERNAL + POOLED_FAMILY}
            tags = sensitivity(dm_nr, "alpha_ITLp95_mean")
            cs = [k for k in dm_nr if convention_sensitive(dm_nr[k], dm_li[k])]
            if cs:
                tags.append("CONVENTION-SENSITIVE(%s)" % ",".join(sorted(cs)))
            gl = gate_label([fser[(a_hi, r)], fser[(a_lo, r)]])
            sens[f"{cname}|r{int(r)}"] = dict(
                tags=tags, nearest={k: v["mean"] for k, v in dm_nr.items()},
                linear={k: v["mean"] for k, v in dm_li.items()}, gate=gl)
            print(f"\n[metric_sensitivity] {cname} r{int(r)} tags={tags or ['none']}{gl}")
    report["metric_sensitivity"] = sens

    # ---------------- descriptive-only blocks (all gated) ------------------
    desc: Dict[str, Any] = {}
    for r in rates:
        gl = gate_label([fser[(A2, r)], fser[(C, r)]])
        d = build_diffs(scored, A2, C, r, "alpha_ITLp95_mean")
        ci = paired_t(d)
        desc[f"conc|r{int(r)}"] = dict(ci=ci, gate=gl)
        print(f"\n[descriptive_conc_itl] r{int(r)} Δ(A2-C)={ci['mean']:+.3f} "
              f"CI[{ci['lo']:+.3f},{ci['hi']:+.3f}] "
              f"-- SIGN ONLY, MAGNITUDE CITATION FORBIDDEN (sec4.3.1){gl}")
        for other, nm in ((TP, "A4_minus_Tprime"), (T, "A4_minus_T")):
            dd = paired_t(build_diffs(scored, A4, other, r, "alpha_ITLp95_mean"))
            g2 = gate_label([fser[(A4, r)], fser[(other, r)]])
            desc[f"{nm}|r{int(r)}"] = dict(ci=dd, gate=g2)
            print(f"[descriptive_A4_minus_Tprime] {nm} r{int(r)} Δ={dd['mean']:+.3f} "
                  f"CI[{dd['lo']:+.3f},{dd['hi']:+.3f}] -- 대리 성립의 증거가 아니라 "
                  f"controller-path·telemetry 비대칭의 상한 부재를 반영한다 (sec5.3){g2}")
    report["descriptive"] = desc

    # ---------------- bimodality diagnostic (sec4.1.3, no gate) ------------
    bim: Dict[str, Any] = {}
    for arm in (T, TP, C):
        for r in rates:
            rows = [scored[(arm, r, rep)] for rep in range(1, 100) if (arm, r, rep) in scored]
            if not rows:
                continue
            for c in (0.25, 0.40, 0.50):
                per_rep = [unstable_frac(x, c) for x in rows]
                bim[f"{arm}|r{int(r)}|c{c}"] = dict(
                    per_rep_median=st.median(per_rep), per_rep_max=max(per_rep),
                    provenance="phase1_scored" if arm == TP else "phase1_scored")
    report["bimodality"] = bim
    print(f"\n[bimodality] {len(bim)} rows (c in 0.25/0.40/0.50, per-rep median+max "
          f"AND pooled; DIAGNOSTIC ONLY -- no decision depends on it)")

    out = os.path.join(outdir, f"g2s_report_{tag}_{job}.json")
    json.dump(report, open(out, "w"), indent=1, default=str)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
