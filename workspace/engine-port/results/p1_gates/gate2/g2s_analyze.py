#!/usr/bin/env python3
"""Gate 2-S scorer -- PREREG_GATE2S_2026-08-09.md rev6 (submission build).

THE PRE-REGISTRATION IS CANONICAL AND THIS CODE FOLLOWS IT.  Every decision rule
below carries the prereg section it implements.  If an implementation choice
would deviate from the prereg, the correct action is to STOP AND REPORT, not to
change the code -- methodology gate #20 is exactly the mismatch between a
pre-registered table and the scorer, and its reverse (code doing something the
prereg never registered) is the same failure.

SECTION -> LINE MAP (methodology gate #20 / sec12-2 item 1).

*** THE MAP IS NOT WRITTEN DOWN HERE, AND THAT IS DELIBERATE. ***
A hand-typed table of line numbers is precisely the object gate #20 is about: it
goes stale on the next edit and then certifies a correspondence that no longer
holds.  (Writing this very docstring shifted every number below it by ~10 lines,
which is how the failure happens in practice.)

Instead:
  * section_line_map()  resolves each pre-registered rule to its line by ANCHOR
    REGEX at runtime, and the self-test fails if any anchor is missing or matches
    more than once.
  * block_manifest()    resolves, for each of the sec6.5 blocks, (i) the F-series
    gate line, (ii) whether it emits magnitudes, (iii) the production phase, from
    `#BLOCK:<name>` / `#GATE:<name>` markers at the computation sites.
main() prints both tables before any number, plus a completeness line naming any
declared block that is not computed.

self-tests: g2s_selftest.py -- (0) REAL archived producer output, positive+negative
control, + a static check on g2s_run.sbatch; (1)-(6) decision-rule checks;
(7) sec6.5 19-block manifest completeness; (8) F-A dead-gate regression;
(9) sec4.4 Fieller suppression; (10) sec6.2 TOST margin.

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
import sys
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

# sec6.5 -- the 19 blocks.  sec6.5 requires the scorer to stamp, FOR EVERY BLOCK,
# (i) the code line where the F-series filter is applied, (ii) whether the block
# emits magnitudes, (iii) the production phase (Phase 0 / Phase 1).  That is also
# sec12-2 item 2.  The line numbers are NOT hand-maintained (they would go stale
# and gate #20 is precisely a stale prereg<->scorer table): each block carries a
# `#BLOCK:<name>` marker at its computation site and a `#GATE:<name>` marker where
# gate_label() is attached, and block_manifest() reads them out of this file at
# runtime.
BLOCKS: List[Tuple[str, bool, str]] = [
    ("confirm_A4like_itl", True, "Phase 1"),
    ("confirm_A4like_ttft", True, "Phase 1"),
    ("secondary_cont_itl", True, "Phase 1"),
    ("secondary_cont_ttft", True, "Phase 1"),
    ("descriptive_conc_itl", True, "Phase 1"),
    ("descriptive_conc_ttft", True, "Phase 1"),
    ("descriptive_A4_minus_Tprime", True, "Phase 1"),
    ("metric_sensitivity", True, "Phase 1"),
    ("tau_ladder", True, "Phase 1"),
    ("component_share", True, "Phase 1"),
    ("descriptive", True, "Phase 1"),
    ("switch_and_drain", True, "Phase 1"),
    ("delivery_rate", True, "Phase 1"),
    ("probe_pos", True, "Phase 1"),          # probes ride the Phase 1 boots (sec6.2.1)
    ("probe_carve", True, "Phase 1"),
    ("mde", True, "Phase 0 + Phase 1"),      # pilot/capscan sigma_D + scored diffs
    ("kv_pool", True, "Phase 1"),            # G-1 boots, sec6.3-8
    ("bimodality", True, "Phase 1"),
    ("premise_falsifier", False, "Phase 1"),  # label update only, no magnitude
]


# sec12-2 item 1 -- every pre-registered rule this file implements, resolved to a
# line by ANCHOR REGEX at runtime.  Each anchor MUST match exactly once; the
# self-test enforces that, so a rename or a deletion fails loudly instead of
# leaving a stale number behind.
SECTION_ANCHORS: List[Tuple[str, str]] = [
    ("sec2 arm set", r"^ARMS = \("),
    ("sec3 cells + n=10", r"^CELLS = \{"),
    ("sec3.1 designated cell", r"^DESIGNATED = "),
    ("sec4.2 primary-alpha ITLp95_mean", r'"alpha_ITLp95_mean": sum\(per_req_p95\)'),
    ("sec4.2 primary-beta pooled p99", r'"beta_pooled_p99": p_linear\(pooled'),
    ("sec4.2 nearest-rank FIXED", r"^def p_nearest\("),
    ("sec6.3-9 unit assert", r"^def assert_units\("),
    ("input contract (harness<->scorer)", r"^def check_row_contract\("),
    ("sec12-4 duration SUM not max", r"dur_sum\[\(arm, rep\)\] \+= float"),
    ("sec4.1.3 unstable_frac", r"^def unstable_frac\("),
    ("sec5.1 paired t (bootstrap FORBIDDEN)", r"^def paired_t\("),
    ("sec5.1 DEGENERATE-T", r"# sec5\.1 DEGENERATE-T"),
    ("sec3.4c MDE %", r"^def mde_pct\("),
    ("sec11.1#3 sign = POINT ESTIMATE", r"^def axis_state\("),
    ("sec5.2.1 nine-cell table", r"^NINE_CELL = \{"),
    ("sec5.2.1 applier", r"^def nine_cell\("),
    ("sec5.6b Holm", r"^def holm\("),
    # NB: the anchor must not match its own definition line -- keep an escape in it.
    ("sec5.6c Holm p source = ALPHA ONLY", r'partial=True, p_source="alpha_ITLp95_mean"\)'),
    ("sec5.6d claim ONLY after Holm", r"^    conf_fired = \[\(tag, r\) for r in rates"),
    ("sec5.6.1 headline combos", r"^def headline\("),
    ("sec6.4 F-series", r"^def f_series\("),
    ("sec6.4 F-A completed/total", r"^            tot = _total_requests\(row\)"),
    ("sec6.4 `total` = attempted count", r"^def _total_requests\("),
    ("sec6.4.1 F-B disjunct(i) unevaluated", r'out\["F_B_i_evaluated"\] = False'),
    ("gate#17 gate label", r"^def gate_label\("),
    ("sec4.4 Fieller + suppression", r"^def fieller_ratio\("),
    ("sec6.2.1(2) P-carve TOST", r"^def tost\("),
    ("sec5.2.2/7.2 controller/split counts", r"^def telemetry_summary\("),
    ("sec4.5 FRAGILE-TO-METRIC", r"^def sensitivity\("),
    ("sec4.5 CONVENTION-SENSITIVE", r"^def convention_sensitive\("),
    ("sec6.5 block list", r"^BLOCKS: List\["),
    ("sec6.5 per-block stamps", r"^def block_manifest\("),
    ("sec8.9 cell-wise premise labels", r"^PREMISE_LABEL = \{"),
    ("sec8.9.1 falsifier (ONE-WAY)", r"^def falsifier\("),
    ("sec8.9.1 WITHDRAWAL_THRESHOLD", r"^WITHDRAWAL_THRESHOLD = "),
    ("sec8.9.1 monotone downgrade only", r"^def apply_falsifier\("),
]


def section_line_map() -> Dict[str, Dict[str, Any]]:
    """sec12-2 item 1 -- resolve every anchor to its line, flagging any that is
    missing or ambiguous (both are failures, never silently skipped)."""
    import re
    src = open(os.path.abspath(__file__)).read().splitlines()
    out: Dict[str, Dict[str, Any]] = {}
    for label, pat in SECTION_ANCHORS:
        rx = re.compile(pat)
        hits = [i for i, line in enumerate(src, 1) if rx.search(line)]
        out[label] = dict(anchor=pat, lines=hits, ok=(len(hits) == 1),
                          line=hits[0] if len(hits) == 1 else None)
    return out


def block_manifest() -> Dict[str, Dict[str, Any]]:
    """sec6.5 (iii) / sec12-2 item 2 -- resolve each block's computation line and
    F-series gate line from the markers in this very file."""
    src = open(os.path.abspath(__file__)).read().splitlines()
    blk_ln: Dict[str, int] = {}
    gate_ln: Dict[str, int] = {}
    for i, line in enumerate(src, 1):
        for tok, dest in (("#BLOCK:", blk_ln), ("#GATE:", gate_ln)):
            if tok in line:
                for nm in line.split(tok, 1)[1].replace(",", " ").split():
                    dest.setdefault(nm, i)
    out: Dict[str, Dict[str, Any]] = {}
    for name, mag, phase in BLOCKS:
        out[name] = dict(emits_magnitude=mag, phase=phase,
                         computed_at_line=blk_ln.get(name),
                         f_series_gate_line=gate_ln.get(name),
                         gate_required=mag)
    return out


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


# ==========================================================================
# INPUT CONTRACT (harness <-> scorer).  This is NOT a decision rule and adds no
# free parameter: it only names the bench_serving.py keys this file already
# dereferences, so a missing key fails as an actionable message instead of a
# bare KeyError three frames deep.
#
# `itls`/`ttfts`/`input_lens`/`output_lens` live in bench_serving.py's
# `result_details`, which is merged into the dumped row ONLY when
# `--output-details` is passed (bench_serving.py:1630; default False at :1667).
# Job 877107/877109 were run without it -> the pre-registered primaries were
# uncomputable.  The aggregate `p95_itl_ms` is a DIFFERENT estimand (run-level,
# pooled) and must never be substituted.
# ==========================================================================
REQUIRED_ROW_KEYS = (
    "itls",            # sec4.2 primary-alpha/beta, sec4.1.3, sec4.5   [--output-details]
    "ttfts",           # TTFTp95 axis, sec6.4 F-B/F-E                  [--output-details]
    "mean_itl_ms",     # sec6.3-9 unit assert
    "request_rate",    # cell key
    "duration",        # sec12-4 SUM
    "completed",       # sec6.4 F-A
)
DETAIL_ONLY_KEYS = ("itls", "ttfts", "input_lens", "output_lens")


class InputContractError(RuntimeError):
    """Harness produced rows the scorer cannot score.  MEASUREMENT-side failure."""


def check_row_contract(row: Dict[str, Any], where: str = "") -> None:
    missing = [k for k in REQUIRED_ROW_KEYS if k not in row]
    if not missing:
        return
    hint = ""
    if any(k in DETAIL_ONLY_KEYS for k in missing):
        hint = (" -- these keys come from bench_serving `result_details`, which is "
                "merged ONLY under `--output-details` (bench_serving.py:1630). "
                "The bench invocation almost certainly omitted that flag. "
                "`p95_itl_ms` is NOT a substitute: it is a run-level pooled "
                "estimand, not the pre-registered per-request ITLp95_mean.")
    raise InputContractError(
        f"row missing {missing} in {where or '<row>'} (present keys: "
        f"{len(row)}){hint}")


def assert_units(row: Dict[str, Any]) -> None:
    """sec6.3-9: itls/ttfts are SECONDS.  Reading them as ms zeroes every spike."""
    flat = [v for x in row["itls"] if x for v in x]
    got = (sum(flat) / len(flat)) * 1000.0
    assert abs(got - row["mean_itl_ms"]) < 1e-6, (
        f"UNIT ASSERT FAILED: mean(itls)*1000={got} != mean_itl_ms={row['mean_itl_ms']}")


# ==========================================================================
# FILE SELECTION -- one selector, used by BOTH load_reps() and preflight(), so the
# two can never disagree about which files are in scope.
#
# The glob `g2s_<tag>_*_rep*_<job>.jsonl` is too permissive: `*` also matches
# `telemetry_<arm>` (g2s_run.sbatch:244 writes
# `g2s_<TAG>_telemetry_<ARM>_rep<REP>_<JOB>.jsonl`, and :191 the pilot twin).  In
# job 877593 that pulled 4 telemetry files with 42k rows into the scored set and
# produced 42,496 "contract violations" from files that were never bench output.
#
# The guard is an ALLOW-LIST on the parsed arm, not a block-list on "_telemetry_".
# Rationale: a block-list only excludes the one artifact type we already know
# about and silently readmits the next one (teloffset, holb, whatever comes next);
# the allow-list admits only the five pre-registered arm names (sec2), so any new
# artifact is excluded by construction.  Pilot files (`pilot_<arm>`) are therefore
# excluded too, which is correct -- they are Phase 0 and the sec3.4 mde block loads
# them under its own glob.  Nothing that used to affect a number is dropped: every
# consumer keys on arm in ARMS, so non-ARMS rows were already dead weight.
#
# EXCLUSIONS ARE RETURNED, NEVER SWALLOWED.  A silent skip and an explicit
# exclusion are different objects; the caller reports how many and why.
# ==========================================================================
MAX_ROWS_PER_BENCH_FILE = 50
# Shape sanity, NOT a decision rule and NOT a free parameter of any estimand: it
# routes an operational exit code only.  A bench dump holds one row per rate --
# 1 (pilot), 2 (scored cell), 5 (capscan rev6), 9 (the 875344 archive).  Telemetry
# holds tens of thousands.  50 is ~5x the largest legitimate file ever observed.


def select_rep_files(outdir: str, tag: str, job: str
                     ) -> Tuple[List[Tuple[str, str, int]], List[Tuple[str, str]]]:
    """-> (accepted[(path, arm, rep)], excluded[(basename, reason)])."""
    accepted: List[Tuple[str, str, int]] = []
    excluded: List[Tuple[str, str]] = []
    for f in sorted(glob.glob(os.path.join(outdir, f"g2s_{tag}_*_rep*_{job}.jsonl"))):
        base = os.path.basename(f)
        if ".holb." in f:
            excluded.append((base, "HOLB probe artifact, not a bench dump"))
            continue
        arm = base.split(f"g2s_{tag}_")[1].rsplit("_rep", 1)[0]
        try:
            rep = int(base.rsplit("_rep", 1)[1].split("_")[0])
        except ValueError:
            excluded.append((base, "unparseable rep index"))
            continue
        if arm not in ARMS:
            excluded.append((base, f"parsed arm '{arm}' is not one of the sec2 arms "
                                   f"-- not bench output"))
            continue
        accepted.append((f, arm, rep))
    return accepted, excluded


# ==========================================================================
# sec12-4 -- a rep file holds TWO rate rows.  Duration is SUMMED, never max().
# ==========================================================================
def load_reps(outdir: str, tag: str, job: str) -> Dict[Tuple[str, float, int], Dict[str, Any]]:
    out: Dict[Tuple[str, float, int], Dict[str, Any]] = {}
    dur_sum: Dict[Tuple[str, int], float] = defaultdict(float)
    accepted, excluded = select_rep_files(outdir, tag, job)
    if excluded:
        print(f"  [load_reps] excluded {len(excluded)} glob match(es) that are not "
              f"bench output: " + ", ".join(f"{b} ({r})" for b, r in excluded[:6])
              + (" ..." if len(excluded) > 6 else ""))
    for f, arm, rep in accepted:
        base = os.path.basename(f)
        for ln, line in enumerate(open(f), 1):
            row = json.loads(line)
            check_row_contract(row, f"{base}:{ln}")
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


def _total_requests(row: Dict[str, Any]) -> int:
    """sec6.4 `total` = the run's total request count (sec3: 120 scored / 60 capscan).
    See the F-A comment in f_series() for why this is len(ttfts) and why using it is
    a bug fix rather than a re-definition."""
    return len(row.get("ttfts") or [])


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

    # F-A : saturation, from the scored runs.
    #
    # PREREG rev6 sec6.4 verbatim:  "- **F-A(포화)**: `completed/total < 1.0`"
    # PREREG rev6 sec6.4.1 table:   "| **F-A**(포화) | 채점 런의 `completed/total` |"
    #
    # The prereg defines `total` as the run's TOTAL REQUEST COUNT (sec3 fixes it:
    # scored runs 120 prompts, capscan 60).  It names NO dump field, so this is a
    # BUG FIX, not a design change: the previous expression read a `num_prompts`
    # key that bench_serving never writes, fell back to `completed`, and made
    # `completed/completed < 1.0` -- structurally False, i.e. F-A could never fire.
    #
    # `len(row["ttfts"])` IS `total`:  bench_serving.py:1380 builds `outputs` by
    # gathering exactly num_prompts tasks, warmup lives in a separate list
    # (`warmup_outputs`, :1269) and is never appended, and result_details maps one
    # entry per output.  Verified on real archived rows (875344): 120/120 scored
    # and 60/60 capscan, with len(errors) == len(ttfts) == num_prompts.
    fa = []
    for (a, r, _rep), row in scored.items():
        if a == arm and r == rate:
            tot = _total_requests(row)
            fa.append(float(row["completed"]) / float(tot) < 1.0 if tot else False)
    out["F_A"] = bool(fa) and any(fa)
    out["F_A_completed_over_total"] = [
        (float(row["completed"]) / float(_total_requests(row) or 1))
        for (a, r, _rep), row in sorted(scored.items())
        if a == arm and r == rate]

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
# sec4.4 -- Fieller interval for a ratio of two paired means.
#
# sec4.4 says the ratio is an IDENTITY, must be tagged `identity: true`, must use
# Fieller for its interval, and inherits rev4 sec4.4's caveat: "분모가 0 근처면
# 비율을 아예 싣지 않는다" (if the denominator is near zero, do not carry the ratio
# at all).  Fieller enforces that caveat exactly: when the denominator's own
# t-statistic is not significant, the quadratic's leading coefficient a <= 0 and
# the solution set is unbounded / exclusive -- which we translate into SUPPRESSED
# rather than printing a meaningless finite-looking interval.
# ==========================================================================
def fieller_ratio(num: Sequence[float], den: Sequence[float]) -> Dict[str, Any]:
    n = len(num)
    if n < 2 or n != len(den):
        return dict(ok=False, reason="n<2 or ragged", identity=True)
    mn, md = sum(num) / n, sum(den) / n
    if md == 0:
        return dict(ok=False, reason="denominator mean is exactly 0", identity=True)
    snn = st.variance(num)
    sdd = st.variance(den)
    snd = (sum((a - mn) * (b - md) for a, b in zip(num, den)) / (n - 1))
    tc = _T95.get(n - 1, 1.96)
    a = md * md - tc * tc * sdd / n
    b = -2.0 * (mn * md - tc * tc * snd / n)
    c = mn * mn - tc * tc * snn / n
    point = mn / md
    if a <= 0:
        # denominator not separated from 0 at 95% -> unbounded Fieller set
        return dict(ok=False, identity=True, point=point,
                    reason=("denominator NOT separated from 0 at 95% (Fieller a<=0) "
                            "-> unbounded interval; sec4.4 rev4 caveat says do not "
                            "carry the ratio at all"))
    disc = b * b - 4 * a * c
    if disc < 0:
        return dict(ok=False, identity=True, point=point,
                    reason="Fieller discriminant < 0 (empty set)")
    root = math.sqrt(disc)
    lo, hi = (-b - root) / (2 * a), (-b + root) / (2 * a)
    if lo > hi:
        lo, hi = hi, lo
    return dict(ok=True, identity=True, point=point, lo=lo, hi=hi,
                denominator_mean=md, numerator_mean=mn)


# sec6.2.1(2) -- P-carve TOST.  margin delta_probe = 0.05 (sec11.1 #1), two-sided
# 90% CI (== the standard TOST at alpha=0.05 per side).
_T90 = {2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015, 6: 1.943, 7: 1.895,
        8: 1.860, 9: 1.833, 10: 1.812, 11: 1.796}


def tost(diffs: Sequence[float], ref_mean: float, margin: float) -> Dict[str, Any]:
    n = len(diffs)
    if n < 2 or ref_mean == 0:
        return dict(n=n, equivalent=False, reason="insufficient data")
    mean = sum(diffs) / n
    sd = st.stdev(diffs)
    delta = margin * abs(ref_mean)
    if sd == 0.0:
        return dict(n=n, mean=mean, lo=mean, hi=mean, delta=delta,
                    equivalent=bool(abs(mean) < delta), degenerate=True)
    se = sd / math.sqrt(n)
    tc = _T90.get(n - 1, 1.645)
    lo, hi = mean - tc * se, mean + tc * se
    return dict(n=n, mean=mean, lo=lo, hi=hi, delta=delta, margin_frac=margin,
                equivalent=bool(lo > -delta and hi < delta), degenerate=False)


# ==========================================================================
# telemetry-derived blocks (sec5.2.2/sec7.2 switch_and_drain, sec6.1.2 delivery_rate)
# ==========================================================================
def telemetry_summary(path: str) -> Dict[str, Any]:
    """sec7.2: use `controller_decision` / `split_transition` counts (emitted with NO
    cadence, multiplexing_mixin.py:401-422).  `switch_count` (stream_index changes in
    runtime_snapshot) is CADENCE-SUBSAMPLED and its miss bias is arm-dependent, so
    sec7.2 forbids citing it alone -- it is computed here ONLY to be printed with
    that prohibition attached.  Residency = time-weighted stream_index histogram
    conditioned on decode-active (sec6.1); `frac` only, t_total never cited (gate #15).
    """
    if not path or not os.path.exists(path) or os.path.getsize(path) == 0:
        return dict(available=False)
    n_ctrl = n_split = 0
    rows: List[Tuple[float, int, int, int]] = []
    for line in open(path):
        try:
            e = json.loads(line)
        except ValueError:
            continue
        ev = e.get("event")
        if ev == "controller_decision":
            n_ctrl += 1
        elif ev == "split_transition":
            n_split += 1
        elif ev == "runtime_snapshot":
            rows.append((e.get("timestamp_monotonic_s", 0.0),
                         int(e.get("stream_index", -1)),
                         int(e.get("decode_running_batch_size", 0) or 0),
                         int(e.get("prefill_active_batch_size", 0) or 0)))
    rows.sort(key=lambda r: r[0])
    seg: Dict[int, float] = defaultdict(float)
    tot = 0.0
    raw_switches = 0
    prev = None
    for i in range(len(rows) - 1):
        t, si, d, _p = rows[i]
        if prev is not None and si != prev:
            raw_switches += 1
        prev = si
        if d <= 0:
            continue
        dt = rows[i + 1][0] - t
        if dt > 0:
            seg[si] += dt
            tot += dt
    frac = {str(k): v / tot for k, v in sorted(seg.items())} if tot else {}
    return dict(available=True, controller_decision=n_ctrl, split_transition=n_split,
                residency_frac_decode_active=frac, n_snapshots=len(rows),
                switch_count_NOT_FOR_SOLE_CITATION=raw_switches)


def load_probes(outdir: str, tag: str, job: str) -> Dict[Tuple[str, int], Dict[str, Any]]:
    out: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for f in sorted(glob.glob(os.path.join(outdir, f"g2s_{tag}_probes_*_rep*_{job}.json"))):
        try:
            d = json.load(open(f))
        except ValueError:
            continue
        res = d.get("result") or {}
        if res.get("status") == "OK" and res.get("itl_p50_ms") is not None:
            out[(d["arm"], int(d["rep"]))] = d
    return out


def load_kv(outdir: str, tag: str, job: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for f in sorted(glob.glob(os.path.join(outdir, f"g2s_{tag}_flags_*_{job}.json"))):
        base = os.path.basename(f)
        arm = base.split(f"g2s_{tag}_flags_")[1].rsplit(f"_{job}.json", 1)[0]
        try:
            d = json.load(open(f))
        except ValueError:
            continue
        out[arm] = dict(kv_cache_tokens=d.get("_kv_cache_tokens"),
                        capture_s=d.get("_capture_s"), capture_gb=d.get("_capture_gb"))
    return out


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


def preflight(outdir: str, tag: str, job: str) -> int:
    """MEASUREMENT-side inventory + input contract.  No decision rule runs here.

    *** THE COST OF MISCLASSIFICATION IS ASYMMETRIC. ***
      rc 7  -> the campaign is re-run on GPU   ~= 6.4 GPU-hr
      rc 8  -> rescore offline                 == 0 GPU-hr
    So a wrong 7 costs 6.4 GPU-hr and a wrong 8 costs a second look at a log.
    When the evidence does not clearly separate the two, this function must NOT
    round to 7; it returns 9 (INDETERMINATE) and asks for a human.

    HISTORY -- job 877593.  This routine returned 7 ("re-run on GPU") when the
    measurement was in fact perfect: all five bench rep files carried itls/ttfts.
    The 42,496 "violations" came from telemetry files that a too-permissive glob
    had pulled in.  The classifier could not tell "I loaded the WRONG FILES" from
    "the RIGHT files lack keys", and defaulted to the expensive branch.
    That is the MIRROR IMAGE of methodology gate #21 (which says a measurement
    failure must never be reported as a gate failure): here a SCORER failure was
    reported as a MEASUREMENT failure.  Same family, opposite direction -- and it
    recurred inside the very mechanism built to make that distinction, which is
    the recurring lesson that building a gate is not the same as the gate working.

    Return codes:
      0 MEASUREMENT_OK                measurement good
      7 MEASUREMENT_INCOMPLETE        accepted bench files really do lack keys  [GPU]
      8 SCORER_SELECTION_DEFECT       violations come from files that are not bench
                                      output -> measurement likely intact        [no GPU]
      9 INDETERMINATE                 cannot separate the two -> human judgment  [no GPU]
    """
    rates = CELLS[tag]
    n_rows = 0
    bad: List[str] = []
    bad_files: Dict[str, int] = defaultdict(int)
    suspect_files: Dict[str, str] = {}
    seen_reps: Dict[Tuple[str, float], set] = defaultdict(set)
    print("---- MEASUREMENT INVENTORY (pre-scoring, no decision rule) ----")

    accepted, excluded = select_rep_files(outdir, tag, job)
    files: List[Tuple[str, str]] = [(f, "phase1") for f, _a, _r in accepted]
    for f in sorted(glob.glob(os.path.join(outdir, f"g2s_{tag}_capscan_*_{job}.jsonl"))):
        if ".holb." not in f:
            files.append((f, "capscan"))

    # sec: exclusions are REPORTED, never swallowed -- an explicit exclusion and a
    # silent skip are different objects.
    print(f"  glob matched {len(accepted) + len(excluded)} rep-shaped file(s): "
          f"accepted {len(accepted)}, excluded {len(excluded)}")
    for b, reason in excluded:
        print(f"    EXCLUDED {b}  <- {reason}")

    for f, kind in files:
        base = os.path.basename(f)
        n_here = 0
        for ln, line in enumerate(open(f), 1):
            try:
                row = json.loads(line)
            except ValueError:
                bad.append(f"{base}:{ln} unparseable")
                bad_files[base] += 1
                continue
            n_rows += 1
            n_here += 1
            try:
                check_row_contract(row, f"{base}:{ln}")
            except InputContractError as e:
                bad.append(str(e))
                bad_files[base] += 1
                continue
            if kind == "phase1":
                arm = base.split(f"g2s_{tag}_")[1].rsplit("_rep", 1)[0]
                rep = int(base.rsplit("_rep", 1)[1].split("_")[0])
                seen_reps[(arm, float(row["request_rate"]))].add(rep)
        # SHAPE CHECK, independent of the name-based allow-list.  A bench dump has
        # one row per rate; anything with tens of thousands of rows is not bench
        # output no matter what it is called.  This is the check that would have
        # classified 877593 correctly even with the glob bug still in place.
        if n_here > MAX_ROWS_PER_BENCH_FILE:
            suspect_files[base] = (f"{n_here} rows > {MAX_ROWS_PER_BENCH_FILE}; a bench "
                                   f"dump holds one row per rate -- this is not bench output")

    for arm in ARMS:
        for r in rates:
            print(f"  cell {arm:24s} r{r:.0f}: n_reps={len(seen_reps.get((arm, r), ())):2d} / 10")
    n_cells_covered = sum(1 for arm in ARMS for r in rates if seen_reps.get((arm, r)))
    print(f"  files={len(files)} rows={n_rows} contract_violations={len(bad)} "
          f"cells_with_data={n_cells_covered}/{len(ARMS) * len(rates)}")
    if suspect_files:
        print("---- FILES OF NON-BENCH SHAPE (scorer-side selection problem) ----")
        for b, why in list(suspect_files.items())[:5]:
            print(f"  {b}: {why}")

    if not bad:
        if not accepted:
            print("VERDICT: INDETERMINATE -- no contract violations, but ZERO bench files "
                  "were accepted. Cannot assert the measurement is OK from an empty set.")
            return 9
        print("VERDICT: MEASUREMENT_OK -- every row satisfies the scorer input contract.")
        return 0

    print(f"---- INPUT CONTRACT VIOLATIONS (showing {min(3, len(bad))} of {len(bad)}) ----")
    for b in bad[:3]:
        print("  " + b)

    # Attribute the violations BEFORE choosing the expensive branch.
    from_suspect = sum(c for b, c in bad_files.items() if b in suspect_files)
    from_bench = len(bad) - from_suspect
    print(f"  attribution: {from_suspect} violation(s) from non-bench-shaped files, "
          f"{from_bench} from files that really are bench output")
    # NOTE: routing must use signals that are INDEPENDENT of contract success.
    # `cells_with_data` is not one -- coverage can only be recorded for rows that
    # already passed the contract, so a total measurement failure drives it to 0 and
    # would masquerade as "unrecognisable inventory".  The independent signals are
    # the accepted FILE COUNT and the per-file SHAPE.
    if from_bench == 0:
        print("VERDICT: SCORER_SELECTION_DEFECT -- every violation comes from a file "
              "that is not bench output. The measurement is very likely INTACT: fix the "
              "file selection and RESCORE OFFLINE. Do NOT re-run on GPU on this basis.")
        return 8
    if not accepted:
        print("VERDICT: INDETERMINATE -- violations exist but ZERO bench files were "
              "accepted, so 'the measurement failed' and 'the selection failed' cannot "
              "be separated. HUMAN JUDGMENT REQUIRED; not rounding to the 6.4 GPU-hr "
              "branch.")
        return 9
    if from_suspect:
        print(f"VERDICT: INDETERMINATE -- violations are MIXED ({from_suspect} from "
              f"non-bench files, {from_bench} from bench files). The selection defect "
              f"may itself be producing the bench-side breakage. HUMAN JUDGMENT "
              f"REQUIRED; not rounding to the 6.4 GPU-hr branch.")
        return 9
    print("VERDICT: MEASUREMENT_INCOMPLETE -- accepted bench files are missing required "
          "keys, and every violation is attributable to them. This is the branch that "
          "costs a GPU re-run.")
    return 7


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--job", required=True)
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--telemetry-dir", default=None)
    ap.add_argument("--check-contract-only", action="store_true",
                    help="run the measurement inventory + input contract and exit "
                         "(0=MEASUREMENT_OK, 7=MEASUREMENT_INCOMPLETE); no scoring")
    args = ap.parse_args()
    tag, job, outdir = args.tag, args.job, args.outdir
    tdir = args.telemetry_dir or outdir
    rates = CELLS[tag]

    if args.check_contract_only:
        return preflight(outdir, tag, job)

    scored = load_reps(outdir, tag, job)
    capscan: Dict[Tuple[str, float, int], Dict[str, Any]] = {}
    for f in sorted(glob.glob(os.path.join(outdir, f"g2s_{tag}_capscan_*_{job}.jsonl"))):
        base = os.path.basename(f)
        arm = base.split(f"g2s_{tag}_capscan_")[1].rsplit("_seed", 1)[0]
        seed = int(base.rsplit("_seed", 1)[1].split("_")[0])
        for ln, line in enumerate(open(f), 1):
            row = json.loads(line)
            check_row_contract(row, f"{base}:{ln}")   # sec6.4 F-B(i) reads ttfts
            capscan[(arm, float(row["request_rate"]), seed)] = row

    report: Dict[str, Any] = {"tag": tag, "job": job, "prereg": "rev6",
                              "blocks": [b for b, _, _ in BLOCKS], "n_blocks": len(BLOCKS)}
    report["block_manifest"] = block_manifest()   # sec6.5 (i)(ii)(iii), sec12-2 #2
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

    # ---- sec6.5 block manifest: EVERY block stamps (i) its F-series gate line,
    # (ii) whether it emits magnitudes, (iii) its production phase.  A block whose
    # `computed_at_line` is None is NOT SILENTLY MISSING -- it is printed as such.
    slm = section_line_map()
    report["section_line_map"] = slm
    bad_anchor = [k for k, v in slm.items() if not v["ok"]]
    print(f"\n[sec12-2 SECTION->LINE MAP] {len(slm)} pre-registered rules, "
          f"resolved by anchor at runtime")
    for k, v in slm.items():
        print(f"  {k:40s} L{v['line'] if v['ok'] else '??? ' + str(v['lines'])}")
    print(f"  unresolved/ambiguous anchors: {bad_anchor or 'none'}")

    bm = report["block_manifest"]
    print(f"\n[sec6.5 BLOCK MANIFEST] {len(BLOCKS)} blocks")
    print(f"  {'block':30s} {'mag':4s} {'phase':16s} {'computed@':10s} {'F-gate@':8s}")
    for name, mag, phase in BLOCKS:
        m = bm[name]
        print(f"  {name:30s} {str(mag):4s} {phase:16s} "
              f"{str(m['computed_at_line']):10s} {str(m['f_series_gate_line']):8s}"
              + ("" if m["computed_at_line"] else "   <-- NOT COMPUTED"))
    missing_blocks = [n for n, _, _ in BLOCKS if not bm[n]["computed_at_line"]]
    ungated = [n for n, mag, _ in BLOCKS if mag and not bm[n]["f_series_gate_line"]]
    print(f"  blocks not computed: {missing_blocks or 'none'}")
    print(f"  magnitude blocks with no F-series gate line (gate #17): {ungated or 'none'}")

    # ---------------- F-series (sec6.4), needed by every magnitude block ----
    fser = {(a, r): f_series(scored, capscan, a, r) for a in ARMS for r in rates}
    report["f_series"] = {f"{a}@{r}": v for (a, r), v in fser.items()}

    # ---------------- sec8.9.1 falsifier (label update, no magnitude) -------
    labels = dict(PREMISE_LABEL)
    fal: Dict[str, Any] = {}
    for r in rates:   #BLOCK: premise_falsifier
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
                #BLOCK: confirm_A4like_itl confirm_A4like_ttft secondary_cont_itl secondary_cont_ttft
                ci_i, ci_t = paired_t(d_itl), paired_t(d_ttft)
                ref = st.mean([metrics(scored[(a_lo, r, rep)])[mkey]
                               for rep in range(1, 100) if (a_lo, r, rep) in scored] or [0])
                mde = mde_pct(d_itl, ref)
                nc = nine_cell(ci_i, ci_t, mde)
                #GATE: confirm_A4like_itl confirm_A4like_ttft secondary_cont_itl secondary_cont_ttft
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
    report["primary_contrasts"] = results
    # sec6.5 requires each declared block to be addressable by NAME in the report,
    # not buried in a composite-key dict (that is how 9 of them went missing).
    for cname, _, _ in contrasts:
        report[f"{cname}_itl"] = {
            k: dict(ci=v["ci_itl"], nine_cell=v["nine_cell"], mde_pct=v["mde_pct"],
                    gate=v["gate"], premise=v["premise"])
            for k, v in results.items() if k.startswith(f"{cname}|")}
        report[f"{cname}_ttft"] = {
            k: dict(ci=v["ci_ttft"], axis=v["nine_cell"]["ttft_axis"], gate=v["gate"])
            for k, v in results.items() if k.startswith(f"{cname}|")}

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
            tags = sensitivity(dm_nr, "alpha_ITLp95_mean")   #BLOCK: metric_sensitivity
            cs = [k for k in dm_nr if convention_sensitive(dm_nr[k], dm_li[k])]
            if cs:
                tags.append("CONVENTION-SENSITIVE(%s)" % ",".join(sorted(cs)))
            gl = gate_label([fser[(a_hi, r)], fser[(a_lo, r)]])   #GATE: metric_sensitivity
            sens[f"{cname}|r{int(r)}"] = dict(
                tags=tags, nearest={k: v["mean"] for k, v in dm_nr.items()},
                linear={k: v["mean"] for k, v in dm_li.items()}, gate=gl)
            print(f"\n[metric_sensitivity] {cname} r{int(r)} tags={tags or ['none']}{gl}")
    report["metric_sensitivity"] = sens

    # ---------------- descriptive contrast blocks (all gated) --------------
    # NOTE: report key is `descriptive_contrasts`, NOT `descriptive`.  sec6.5 and
    # sec4.5 define the block named `descriptive` as {TTFTp50, ITLp50, throughput,
    # completed/total} per arm-cell -- a different object, emitted further below.
    # The old code stored this container under `descriptive` and thereby made the
    # sec6.5 block of that name look present when it was not.
    desc: Dict[str, Any] = {}
    for r in rates:
        gl = gate_label([fser[(A2, r)], fser[(C, r)]])   #GATE: descriptive_conc_itl descriptive_conc_ttft
        d = build_diffs(scored, A2, C, r, "alpha_ITLp95_mean")   #BLOCK: descriptive_conc_itl
        ci = paired_t(d)
        # sec4.3.1: Delta_conc is reported on BOTH axes jointly (point estimate +
        # paired t CI), SIGN AND DIRECTION ONLY -- magnitude citation forbidden.
        ci_ttft = paired_t(build_diffs(scored, A2, C, r, "TTFTp95"))   #BLOCK: descriptive_conc_ttft
        desc[f"conc|r{int(r)}"] = dict(ci=ci, ci_ttft=ci_ttft, gate=gl,
                                       magnitude_citation="FORBIDDEN (sec4.3.1)")
        print(f"\n[descriptive_conc_itl] r{int(r)} Δ(A2-C)={ci['mean']:+.3f} "
              f"CI[{ci['lo']:+.3f},{ci['hi']:+.3f}] "
              f"-- SIGN ONLY, MAGNITUDE CITATION FORBIDDEN (sec4.3.1){gl}")
        print(f"[descriptive_conc_ttft] r{int(r)} Δ(A2-C) TTFTp95={ci_ttft['mean']:+.1f}ms "
              f"CI[{ci_ttft['lo']:+.1f},{ci_ttft['hi']:+.1f}] axis={axis_state(ci_ttft)} "
              f"-- SIGN ONLY, MAGNITUDE CITATION FORBIDDEN (sec4.3.1: A2 is an "
              f"intentionally untuned arm, observer asymmetry has NO magnitude bound "
              f"[G5 UNDETERMINED], 4-flag bundle, KV pool differs 0.031%){gl}")
        for other, nm in ((TP, "A4_minus_Tprime"), (T, "A4_minus_T")):
            dd = paired_t(build_diffs(scored, A4, other, r, "alpha_ITLp95_mean"))   #BLOCK: descriptive_A4_minus_Tprime
            g2 = gate_label([fser[(A4, r)], fser[(other, r)]])   #GATE: descriptive_A4_minus_Tprime
            desc[f"{nm}|r{int(r)}"] = dict(ci=dd, gate=g2)
            print(f"[descriptive_A4_minus_Tprime] {nm} r{int(r)} Δ={dd['mean']:+.3f} "
                  f"CI[{dd['lo']:+.3f},{dd['hi']:+.3f}] -- 대리 성립의 증거가 아니라 "
                  f"controller-path·telemetry 비대칭의 상한 부재를 반영한다 (sec5.3){g2}")
    report["descriptive_contrasts"] = desc
    report["descriptive_conc_itl"] = {k: dict(ci=v["ci"], gate=v["gate"],
                                              magnitude_citation=v["magnitude_citation"])
                                      for k, v in desc.items() if k.startswith("conc|")}
    report["descriptive_conc_ttft"] = {k: dict(ci=v["ci_ttft"], gate=v["gate"],
                                               magnitude_citation=v["magnitude_citation"])
                                       for k, v in desc.items() if k.startswith("conc|")}
    report["descriptive_A4_minus_Tprime"] = {k: v for k, v in desc.items()
                                             if k.startswith("A4_minus_")}

    # ---------------- sec4.5 tau ladder ------------------------------------
    # X_tau = fraction of requests whose MAX ITL exceeds tau, tau in {40,60,100} ms
    # (methodology gate #12 threshold ladder).  Per arm-cell mean over reps, plus
    # the paired contrast diffs.  Magnitudes -> gate label.
    tau: Dict[str, Any] = {}
    for r in rates:
        for arm in ARMS:   #BLOCK: tau_ladder
            rows = [scored[(arm, r, rep)] for rep in range(1, 100) if (arm, r, rep) in scored]
            if not rows:
                tau[f"{arm}|r{int(r)}"] = dict(n_reps=0, na_reason="no scored reps for this arm-cell")
                continue
            per = {f"X{int(t)}": st.mean([metrics(x)[f"X{int(t)}"] for x in rows])
                   for t in TAU_LADDER_MS}
            gl = gate_label([fser[(arm, r)]])   #GATE: tau_ladder
            tau[f"{arm}|r{int(r)}"] = dict(n_reps=len(rows), mean=per, gate=gl)
            print(f"\n[tau_ladder] {arm} r{int(r)} n={len(rows)} "
                  + " ".join(f"{k}={v:.4f}" for k, v in per.items()) + gl)
    for cname, a_hi, a_lo in contrasts:
        for r in rates:
            for t in TAU_LADDER_MS:
                key = f"X{int(t)}"
                ci = paired_t(build_diffs(scored, a_hi, a_lo, r, key))
                gl = gate_label([fser[(a_hi, r)], fser[(a_lo, r)]])   #GATE: tau_ladder
                tau[f"{cname}|{key}|r{int(r)}"] = dict(ci=ci, gate=gl)
                print(f"[tau_ladder] {cname} {key} r{int(r)} Δ={ci['mean']:+.4f} "
                      f"CI[{ci['lo']:+.4f},{ci['hi']:+.4f}] axis={axis_state(ci)}{gl}")
    report["tau_ladder"] = tau

    # ---------------- sec4.4 component share (IDENTITY -- never a verdict) --
    # Delta_split + Delta_conc === ITLp95_mean(A2) - ITLp95_mean(T)  by DEFINITION.
    # The share summing to 1 is ARITHMETIC, not a measurement (gate #6).  Emitted
    # descriptively with identity:true, interval by Fieller, and SUPPRESSED whenever
    # the denominator is not separated from 0 (rev4 sec4.4 caveat) or either
    # component is itself unresolved.  NO DECISION USES THIS.
    comp: Dict[str, Any] = {}
    for r in rates:   #BLOCK: component_share
        d_split = build_diffs(scored, C, T, r, "alpha_ITLp95_mean")     # Delta_split
        d_conc = build_diffs(scored, A2, C, r, "alpha_ITLp95_mean")     # Delta_conc
        n = min(len(d_split), len(d_conc))
        d_split, d_conc = d_split[:n], d_conc[:n]
        den = [a + b for a, b in zip(d_split, d_conc)]
        ci_s, ci_c, ci_d = paired_t(d_split), paired_t(d_conc), paired_t(den)
        fr = fieller_ratio(d_split, den)
        # firing conditions, enforced in code (never silently emitted):
        both_resolved = axis_state(ci_s) != "0" and axis_state(ci_c) != "0"
        den_separated = axis_state(ci_d) != "0" and fr.get("ok", False)
        emit = bool(both_resolved and den_separated)
        gl = gate_label([fser[(A2, r)], fser[(C, r)], fser[(T, r)]])   #GATE: component_share
        reasons = []
        if not both_resolved:
            reasons.append("a component CI contains 0 (contrast unresolved)")
        if not den_separated:
            reasons.append(fr.get("reason", "denominator not separated from 0"))
        comp[f"r{int(r)}"] = dict(
            identity=True, emitted=emit, suppress_reasons=reasons,
            identity_statement=("Delta_split + Delta_conc === ITLp95_mean(A2) - "
                                "ITLp95_mean(T); the shares summing to 1 is arithmetic, "
                                "not a result (sec4.4, gate #6)"),
            used_for_verdict=False,
            ci_split=ci_s, ci_conc=ci_c, ci_denominator=ci_d,
            fieller=fr if emit else dict(ok=False, reason=fr.get("reason")), gate=gl)
        if emit:
            print(f"\n[component_share] r{int(r)} identity:true  share(Δ_split)="
                  f"{fr['point']:+.4f} Fieller95[{fr['lo']:+.4f},{fr['hi']:+.4f}] "
                  f"-- DESCRIPTIVE ONLY, NO VERDICT USES THIS (sec4.4){gl}")
        else:
            print(f"\n[component_share] r{int(r)} identity:true  SUPPRESSED "
                  f"({'; '.join(reasons)}) -- sec4.4 rev4: if the denominator is near "
                  f"zero the ratio is not carried at all{gl}")
    report["component_share"] = comp

    # ---------------- sec4.5/sec6.5 `descriptive` --------------------------
    # TTFTp50 / ITLp50 / throughput / completed-over-total, per arm-cell.
    dsc: Dict[str, Any] = {}
    for r in rates:
        for arm in ARMS:   #BLOCK: descriptive
            rows = [scored[(arm, r, rep)] for rep in range(1, 100) if (arm, r, rep) in scored]
            if not rows:
                dsc[f"{arm}|r{int(r)}"] = dict(n_reps=0, na_reason="no scored reps for this arm-cell")
                continue
            ms = [metrics(x) for x in rows]
            gl = gate_label([fser[(arm, r)]])   #GATE: descriptive
            v = dict(n_reps=len(rows),
                     TTFTp50=st.mean([m["TTFTp50"] for m in ms]),
                     ITLp50=st.mean([m["ITLp50"] for m in ms]),
                     throughput=st.mean([m["throughput"] for m in ms]),
                     completed_over_total=st.mean(
                         [float(x["completed"]) / float(_total_requests(x) or 1) for x in rows]),
                     gate=gl)
            dsc[f"{arm}|r{int(r)}"] = v
            print(f"\n[descriptive] {arm} r{int(r)} n={v['n_reps']} "
                  f"TTFTp50={v['TTFTp50']:.1f}ms ITLp50={v['ITLp50']:.2f}ms "
                  f"throughput={v['throughput']:.3f} "
                  f"completed/total={v['completed_over_total']:.4f}{gl}")
    report["descriptive"] = dsc

    # ---------------- sec5.2.2/sec7.2 switch_and_drain ---------------------
    # SYMMETRIC: reported for every arm and attached to every primary contrast,
    # regardless of the verdict direction (rev3 killed the S3-only rule, which made
    # low power favour the author).  NOTHING is exculpated by these numbers.
    sw: Dict[str, Any] = {}
    SW_NOTES = [
        "sec7.2: quantities used are `controller_decision` and `split_transition` "
        "(multiplexing_mixin.py:401-422), which are emitted with NO cadence.",
        "sec7.2: `switch_count` (stream_index changes in runtime_snapshot) is a "
        "PDMUX_DUAL_WORKER_TRACE_EVERY=32 subsample whose miss bias is ARM-DEPENDENT "
        "(T dwells longer under sticky) -- SOLE CITATION FORBIDDEN. It is printed "
        "only so the prohibition travels with it.",
        "sec7.2: transitions taken through the `adjust_stream_groups` path are "
        "recorded by NO event (multiplexing_mixin.py:1084-1086 is logger.debug, 0 "
        "occurrences across 96+ server logs). --log-level debug is NOT added "
        "(it would change arm flags vs the preceding campaigns).",
        "sec5.2.2: these quantities are reported on S1/S2/S3 alike and EXCULPATE NO "
        "VERDICT in any direction.",
    ]
    for r in rates:
        for arm in ARMS:   #BLOCK: switch_and_drain
            tp = os.path.join(tdir, f"g2s_{tag}_telemetry_{arm}_r{int(r)}_{job}.jsonl")
            s = telemetry_summary(tp)
            gl = gate_label([fser[(arm, r)]])   #GATE: switch_and_drain
            if not s.get("available"):
                s = dict(available=False, na_reason=(
                    "no telemetry for this arm (plainaux runs without "
                    "PDMUX_TELEMETRY_PATH by design, sec2)"))
                sw[f"{arm}|r{int(r)}"] = dict(s, gate=gl)
                print(f"\n[switch_and_drain] {arm} r{int(r)} N/A -- {s['na_reason']}{gl}")
                continue
            if arm == A4 and s["controller_decision"] == 0 and s["split_transition"] == 0:
                s["a4_note"] = ("A4 runs with NO R2 policy, so `_r2_decide_idx` never "
                                "runs and neither event is emitted. This is a "
                                "STRUCTURAL ABSENCE, not a measured zero.")
            sw[f"{arm}|r{int(r)}"] = dict(s, gate=gl)
            print(f"\n[switch_and_drain] {arm} r{int(r)} controller_decision="
                  f"{s['controller_decision']} split_transition={s['split_transition']} "
                  f"residency(decode-active)="
                  + "{" + ", ".join(f"idx{k}={v:.4f}" for k, v in
                                    s["residency_frac_decode_active"].items()) + "}"
                  + (f"  [{s['a4_note']}]" if s.get("a4_note") else "") + gl)
    report["switch_and_drain"] = dict(cells=sw, notes=SW_NOTES)

    # ---------------- sec6.1.2 delivery_rate (DESCRIPTIVE ONLY) ------------
    # rev4: frac_{T'} and frac_{A4} are the ONLY live delivery rates and they are
    # PURELY DESCRIPTIVE -- no threshold, no free parameter, used for neither
    # verdict nor correction.  frac_T (sticky ON) is a CODE IDENTITY and frac(idx in
    # {0,3}) for C is a CODE INVARIANT: both are sanity checks, NOT gates, and are
    # not evidence of treatment delivery.  `frac` only, t_total never cited (#15).
    dlv: Dict[str, Any] = {}
    STATUS = {T: "sec6.1.2 CODE IDENTITY (sticky ON) -- sanity check, not a gate",
              C: "sec6.1.1 CODE INVARIANT (idx0 and idx3 are the same physical state) "
                 "-- sanity check, not a gate, not evidence of no-split",
              TP: "sec6.1.2 rev4 LIVE DELIVERY RATE -- descriptive only, no threshold",
              A4: "sec6.1.2 rev4 LIVE DELIVERY RATE -- descriptive only, no threshold",
              A2: "no pdmux telemetry by design"}
    for r in rates:
        for arm in ARMS:   #BLOCK: delivery_rate
            tp = os.path.join(tdir, f"g2s_{tag}_telemetry_{arm}_r{int(r)}_{job}.jsonl")
            s = telemetry_summary(tp)
            gl = gate_label([fser[(arm, r)]])   #GATE: delivery_rate
            if not s.get("available"):
                dlv[f"{arm}|r{int(r)}"] = dict(
                    available=False, prereg_status=STATUS[arm], gate=gl,
                    na_reason="no telemetry for this arm (by design)")
                print(f"\n[delivery_rate] {arm} r{int(r)} N/A -- no telemetry by design{gl}")
                continue
            fr = s["residency_frac_decode_active"]
            f1, f03 = fr.get("1", 0.0), fr.get("0", 0.0) + fr.get("3", 0.0)
            e: Dict[str, Any] = dict(available=True, frac=fr, frac_idx1=f1,
                                     frac_idx0_or_3=f03, prereg_status=STATUS[arm],
                                     threshold="NONE (sec6.1.2 rev4)", gate=gl)
            if arm == T:
                e["sanity_identity_deviation"] = 1.0 - f1
            if arm == C:
                e["sanity_invariant_holds"] = bool(abs(1.0 - f03) < 5e-5)
            dlv[f"{arm}|r{int(r)}"] = e
            extra = ""
            if arm == T:
                extra = (f"  [sanity: frac_T vs identity 1.0000 -> {1.0 - f1:+.4f}; "
                         f"deviation is a cadence-PHASE artifact candidate, NOT "
                         f"treatment loss, NOT a gate]")
            elif arm == C:
                extra = (f"  [sanity: frac(idx in {{0,3}})={f03:.4f} vs invariant "
                         f"1.0000 -> {'holds' if abs(1.0 - f03) < 5e-5 else 'VIOLATED: halt and investigate'}]")
            elif arm in (TP, A4):
                extra = f"  [live delivery rate frac={f1:.4f}; DESCRIPTIVE ONLY, no threshold]"
            print(f"\n[delivery_rate] {arm} r{int(r)} "
                  + "{" + ", ".join(f"idx{k}={v:.4f}" for k, v in fr.items()) + "}"
                  + extra + gl)
    report["delivery_rate"] = dlv

    # ---------------- sec6.2 probes ---------------------------------------
    probes = load_probes(outdir, tag, job)
    PROBE_CAVEATS = [
        "sec6.2.1 rev4: micro-regime caveat -- measured at 8 concurrent requests, "
        "input 128 / output 128 (canonical confound #7 regime). The result does NOT "
        "transfer to the serving operating point (in2000/out96, rate 2-4).",
        "sec6.2.1 rev3 change 3: the detection floor is NOT converted to SM units. "
        "The 15-60 SM band was an 8B/44-92 SM import and is deleted (CONSENSUS sec3 "
        "items 21/25/26).",
    ]
    pp_d = [probes[(T, rep)]["result"]["itl_p50_ms"] - probes[(C, rep)]["result"]["itl_p50_ms"]
            for rep in range(1, 100) if (T, rep) in probes and (C, rep) in probes]
    pp = paired_t(pp_d)   #BLOCK: probe_pos
    gl_pp = gate_label([fser[(T, rates[0])], fser[(C, rates[0])]])   #GATE: probe_pos
    # gate #21: no probe data is a MEASUREMENT gap (UNDETERMINED).  It must NOT be
    # collapsed into "HARNESS DEFECT", which is the verdict for a probe that RAN and
    # whose paired CI still contains 0.
    if len(pp_d) < 2:
        pp_verdict, pp_ok = "UNDETERMINED", False
    else:
        pp_ok = axis_state(pp) != "0"
        pp_verdict = "OK" if pp_ok else "HARNESS DEFECT"
    eps = None
    if pp_d:
        mt = st.mean([probes[(T, rep)]["result"]["itl_p50_ms"] for rep in range(1, 100)
                      if (T, rep) in probes and (C, rep) in probes])
        mc = st.mean([probes[(C, rep)]["result"]["itl_p50_ms"] for rep in range(1, 100)
                      if (T, rep) in probes and (C, rep) in probes])
        if mt > 0 and mc > 0:
            eps = math.log(mt / mc) / math.log(108.0 / 34.0)
    report["probe_pos"] = dict(
        role="HARNESS SANITY CHECK -- not an anchor condition (sec6.2.1(1), rev4)",
        ci=pp, n_boots=len(pp_d), verdict=pp_verdict,
        undetermined_reason=("fewer than 2 paired probe boots on disk -- MEASUREMENT "
                             "gap, never a mismatch (gate #21)") if pp_verdict == "UNDETERMINED" else None,
        internal_epsilon_34_to_108=eps,
        epsilon_scope=("valid ONLY at these two endpoints of THIS campaign; pooled "
                       "two-point elasticity, transfer outside this operating point "
                       "is forbidden (sec6.2.1 rev3 change 3)"),
        caveats=PROBE_CAVEATS, gate=gl_pp)
    print(f"\n[probe_pos] T(idx1=34SM) - C(idx3=108SM) n_boots={len(pp_d)} "
          f"Δ={pp['mean']:+.3f}ms CI[{pp['lo']:+.3f},{pp['hi']:+.3f}] -> {pp_verdict}"
          + ("  (no probe artifacts on disk -- MEASUREMENT gap, gate #21; NOT a defect claim)"
             if pp_verdict == "UNDETERMINED" else
             ("  (CI contains 0; fix the probe)" if not pp_ok else ""))
          + f" -- SANITY CHECK, not an anchor condition (sec6.2.1(1)){gl_pp}")
    print(f"  internal epsilon(34->108) = "
          f"{'n/a' if eps is None else f'{eps:.4f}'} -- valid ONLY at these two "
          f"endpoints of this campaign; no SM-unit detection floor is derived.")

    pc_d = [probes[(C, rep)]["result"]["itl_p50_ms"] - probes[(A2, rep)]["result"]["itl_p50_ms"]
            for rep in range(1, 100) if (C, rep) in probes and (A2, rep) in probes]
    ref_a2 = st.mean([probes[(A2, rep)]["result"]["itl_p50_ms"] for rep in range(1, 100)
                      if (C, rep) in probes and (A2, rep) in probes] or [0])
    pc = tost(pc_d, ref_a2, DELTA_PROBE)   #BLOCK: probe_carve
    gl_pc = gate_label([fser[(A2, rates[0])], fser[(C, rates[0])]])   #GATE: probe_carve
    # gate #21 again: with no probe artifacts, TOST must not report
    # "equivalent=False" -- that reads as a measured non-equivalence.
    pc_state = "UNDETERMINED" if len(pc_d) < 2 else (
        "EQUIVALENT" if pc.get("equivalent") else "NOT-EQUIVALENT")
    report["probe_carve"] = dict(
        state=pc_state,
        undetermined_reason=("fewer than 2 paired probe boots on disk -- MEASUREMENT "
                             "gap, never a mismatch (gate #21)") if pc_state == "UNDETERMINED" else None,
        tost=pc, n_boots=len(pc_d), margin=DELTA_PROBE, ci_level="two-sided 90%",
        campaign_veto="NONE (sec6.2.1(2))",
        confound=("A2 runs event_loop_normal and C runs event_loop_pdmux, so loop "
                  "overhead is mixed in. An equivalence result means the CONJUNCTION "
                  "'no SM carve AND no loop overhead' is within 5% -- it does NOT "
                  "isolate SM carving."),
        caveats=PROBE_CAVEATS, gate=gl_pc)
    print(f"\n[probe_carve] C - A2 n_boots={len(pc_d)} Δ={pc.get('mean', float('nan')):+.3f}ms "
          f"90%CI[{pc.get('lo', float('nan')):+.3f},{pc.get('hi', float('nan')):+.3f}] "
          f"δ=±{pc.get('delta', float('nan')):.3f}ms ({DELTA_PROBE:.0%}) -> TOST {pc_state}"
          + ("  (no probe artifacts on disk -- MEASUREMENT gap, gate #21; this is NOT "
             "a measured non-equivalence)" if pc_state == "UNDETERMINED" else "")
          + f" -- NO CAMPAIGN VETO; the confound "
            f"(event_loop_normal vs event_loop_pdmux) is NOT isolated{gl_pc}")

    # ---------------- sec3.4 mde (standalone block) ------------------------
    # (a) design-matched sigma_D from the Phase 0 T/C pilot (designated cell, n=4,
    #     df=3 -> sigma 95% CI ~ [0.57x, 3.73x]; that fact is MANDATORY with every MDE)
    # (b) capscan auxiliary, PAIRED on shared seeds only, df=2, stated on every citation
    # (c) NO `UNDERPOWERED` threshold exists (rev2 removed it) -- MDE is reported instead
    mde_blk: Dict[str, Any] = {}
    pilot_sd = None
    prs = {}
    for f in sorted(glob.glob(os.path.join(outdir, f"g2s_{tag}_pilot_*_rep*_{job}.jsonl"))):
        base = os.path.basename(f)
        parm = base.split(f"g2s_{tag}_pilot_")[1].rsplit("_rep", 1)[0]
        prep = int(base.rsplit("_rep", 1)[1].split("_")[0])
        for line in open(f):
            row = json.loads(line)
            check_row_contract(row, base)
            prs[(parm, prep)] = metrics(row)["alpha_ITLp95_mean"]
    pil_d = [prs[(C, k)] - prs[(T, k)] for k in range(1, 100)
             if (C, k) in prs and (T, k) in prs]
    if len(pil_d) >= 2:
        pilot_sd = st.stdev(pil_d)
    cap_d = [metrics(capscan[(C, DESIGNATED[1], s)])["alpha_ITLp95_mean"]
             - metrics(capscan[(T, DESIGNATED[1], s)])["alpha_ITLp95_mean"]
             for s in (7, 17, 27)
             if (C, DESIGNATED[1], s) in capscan and (T, DESIGNATED[1], s) in capscan]
    cap_sd = st.stdev(cap_d) if len(cap_d) >= 2 else None
    for cname, a_hi, a_lo in contrasts:
        for r in rates:   #BLOCK: mde
            for mkey in PRIMARY_METRICS:
                d = build_diffs(scored, a_hi, a_lo, r, mkey)
                ref = st.mean([metrics(scored[(a_lo, r, rep)])[mkey]
                               for rep in range(1, 100) if (a_lo, r, rep) in scored] or [0])
                gl = gate_label([fser[(a_hi, r)], fser[(a_lo, r)]])   #GATE: mde
                mde_blk[f"{cname}|{mkey}|r{int(r)}"] = dict(
                    mde_pct=mde_pct(d, ref), n=len(d), ref_mean=ref, gate=gl,
                    sentence=("이 설계는 차이가 %.1f%% 미만인 경우를 배제할 수 없다."
                              % mde_pct(d, ref)) if len(d) >= 2 else "n<2")
                print(f"\n[mde] {cname} {mkey} r{int(r)} n={len(d)} "
                      f"MDE={mde_pct(d, ref):.1f}% of ref mean {ref:.3f}{gl}")
    mde_blk["_sigma_D_sources"] = dict(
        pilot_sd_alpha=pilot_sd, pilot_n=len(pil_d),
        pilot_caveat=("designated cell Zamba2 r3, T/C, n=4 -> df=3; sigma 95% CI "
                      "~ [0.57x, 3.73x]. sec3.4(a) makes stating this MANDATORY with "
                      "every MDE."),
        capscan_paired_sd_alpha=cap_sd, capscan_n=len(cap_d),
        capscan_caveat=("auxiliary only, PAIRED on shared seeds (never a sqrt(2) "
                        "single-arm proxy); n=3 -> df=2, which must be stated on "
                        "every citation (sec3.4(b))."),
        underpowered_threshold="NONE -- rev2 removed it (sec3.4(c))")
    print(f"\n[mde] sigma_D sources: pilot sd={pilot_sd} (n={len(pil_d)}, df=3, "
          f"sigma 95% CI ~ [0.57x,3.73x] MANDATORY) | capscan paired sd={cap_sd} "
          f"(n={len(cap_d)}, df=2, state on every citation) | no UNDERPOWERED threshold")
    report["mde"] = mde_blk

    # ---------------- sec6.3-8 kv_pool ------------------------------------
    kv = load_kv(outdir, tag, job)   #BLOCK: kv_pool
    gl_kv = gate_label([fser[(T, rates[0])], fser[(C, rates[0])]])   #GATE: kv_pool
    tk, ck = kv.get(T, {}).get("kv_cache_tokens"), kv.get(C, {}).get("kv_cache_tokens")
    if tk is None or ck is None:
        kv_state, kv_msg = "UNDETERMINED", ("KV token count missing for T and/or C -- "
                                            "MEASUREMENT gap, never a mismatch (gate #21)")
    elif tk == ck:
        kv_state, kv_msg = "PASS", f"tokens(T)==tokens(C)=={tk}"
    else:
        kv_state, kv_msg = "FAIL", (f"tokens(T)={tk} != tokens(C)={ck} -> sec6.3-8: "
                                    f"Delta_split MAGNITUDE CITATION FORBIDDEN")
    report["kv_pool"] = dict(state=kv_state, message=kv_msg, per_arm=kv,
                             delta_split_magnitude_citable=(kv_state == "PASS"), gate=gl_kv)
    print(f"\n[kv_pool] {kv_state} -- {kv_msg}")
    for a in ARMS:
        print(f"  {a:24s} kv_tokens={kv.get(a, {}).get('kv_cache_tokens')} "
              f"capture={kv.get(a, {}).get('capture_s')}s/{kv.get(a, {}).get('capture_gb')}GB")
    print(f"  {gl_kv}")

    # ---------------- bimodality diagnostic (sec4.1.3) ---------------------
    # sec4.1.3 makes this DIAGNOSTIC ONLY (no threshold, no decision).  It still
    # prints magnitudes, so sec6.5 / gate #17 require the F-series label anyway --
    # "diagnostic" is not an exemption from labelling.  Provenance (production
    # phase) is recorded per sec6.5(iii).
    bim: Dict[str, Any] = {}
    for arm in (T, TP, C):
        for r in rates:   #BLOCK: bimodality
            rows = [scored[(arm, r, rep)] for rep in range(1, 100) if (arm, r, rep) in scored]
            gl = gate_label([fser[(arm, r)]])   #GATE: bimodality
            if not rows:
                bim[f"{arm}|r{int(r)}"] = dict(na_reason="no scored reps for this arm-cell")
                continue
            for c in (0.25, 0.40, 0.50):
                per_rep = [unstable_frac(x, c) for x in rows]
                bim[f"{arm}|r{int(r)}|c{c}"] = dict(
                    per_rep_median=st.median(per_rep), per_rep_max=max(per_rep),
                    n_reps=len(rows), provenance="phase1_scored", gate=gl,
                    role="DIAGNOSTIC ONLY (sec4.1.3) -- no threshold, no decision")
    print(f"\n[bimodality] {len(bim)} rows (c in 0.25/0.40/0.50, per-rep median+max; "
          f"provenance=phase1_scored; DIAGNOSTIC ONLY -- no decision depends on it; "
          f"F-series label attached per gate #17)")
    report["bimodality"] = bim

    # ---------------- design conformance stamp -----------------------------
    # A report produced from a truncated run (smoke test, crashed campaign, partial
    # array) is structurally indistinguishable from a real one once it is a json on
    # disk.  Job 877593 was a 1-rep, single-rate plumbing smoke; nothing in its
    # report said so.  Stamp the realized inventory against sec3 (n=10 paired, both
    # rates per cell) so a reader cannot mistake one for the other.  This labels
    # PROVENANCE; it changes no estimand and gates no decision.
    realized = {f"{a}@r{r:.0f}": sum(1 for rep in range(1, 100) if (a, r, rep) in scored)
                for a in ARMS for r in rates}
    short = {k: v for k, v in realized.items() if v < 10}
    conformant = not short
    report["design_conformance"] = dict(
        prereg_requires="sec3: n=10 paired reps in each of the cells, both rates",
        realized_reps_per_cell=realized, cells_below_n10=short, conformant=conformant,
        usable_for=("pre-registered analysis" if conformant else
                    "PLUMBING/PIPELINE VERIFICATION ONLY -- this report was produced "
                    "from an inventory that does not meet sec3. Do NOT cite any "
                    "magnitude, verdict, headline or Holm result from it, and do NOT "
                    "treat it as pre-registered campaign data."))
    print("\n[design_conformance] " + ("CONFORMANT to sec3 (n=10 x both rates)."
          if conformant else
          "*** NOT CONFORMANT to sec3 *** cells below n=10: "
          f"{short}\n  This report is PLUMBING VERIFICATION ONLY -- no magnitude, "
          "verdict, headline or Holm result in it may be cited, and it is NOT "
          "pre-registered campaign data."))

    # ---------------- sec6.5 completeness self-check -----------------------
    produced = [n for n, _, _ in BLOCKS if n in report]
    absent = [n for n, _, _ in BLOCKS if n not in report]
    report["block_completeness"] = dict(
        n_declared=len(BLOCKS), n_in_report=len(produced), absent=absent)
    print(f"\n[sec6.5 COMPLETENESS] {len(produced)}/{len(BLOCKS)} declared blocks present "
          f"in the report json; absent={absent or 'none'}")

    out = os.path.join(outdir, f"g2s_report_{tag}_{job}.json")
    json.dump(report, open(out, "w"), indent=1, default=str)
    print(f"\nwrote {out}")
    return 0


# Exit codes (consumed by g2s_run.sbatch to keep gate #21's distinction):
#   0 = scored
#   7 = INPUT CONTRACT VIOLATION -> measurement side; the rows cannot be scored
#   1 = any other scorer failure  -> scorer side; the data may still be intact
if __name__ == "__main__":
    try:
        sys.exit(main() or 0)
    except InputContractError as _e:
        print(f"\nINPUT CONTRACT VIOLATION: {_e}", file=sys.stderr)
        print("This is a MEASUREMENT-side failure (harness produced unscoreable rows), "
              "not a scorer bug.  Do NOT substitute an aggregate metric.", file=sys.stderr)
        sys.exit(7)
