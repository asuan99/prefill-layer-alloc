#!/usr/bin/env python3
"""E-A 5-arm campaign scorer (PREREG_G2EA_2026-08-07.md).

Adapted from g2_analyze.py (rev4 4-arm scorer, jobs 875344/875346) with the
harness/analysis fixes the 2026-08-07 claims-auditor audit required before
citing magnitudes from a mixed-chunk-participating arm:

  FIX #1 F-E enforcement: phase1_fE's "flagged" boolean is now folded into
    the cell verdict (`citable=False`, `verdict` gets a "_FLAGGED..." suffix)
    for BOTH primary comparisons whenever either participating arm is F-E
    flagged at that rate. rev4's g2_analyze.py computed F-E but never wired
    it into cell_verdict, so a flagged cell's magnitude could still be quoted.
  FIX #4 persistence: Phase 0 F-A/F-B/F-E and the section-5.1.1 mid-check are
    now included in the JSON report (previously stdout-only).
  FIX #5 VACUOUS ceiling threshold: lowered from >=0.99 to >=0.95 (0.99 let
    Granite r6's 0.9858 through un-flagged in the rev4 campaign; this
    constant has no principled derivation either, so it is a provisional
    recalibration, documented as such, not a validated new threshold).
  FIX #3/#6 realized flags: reads a G-1 (or, absent that, capscan) flags json
    per arm and extracts realized enable_mixed_chunk / chunked_prefill_size /
    disable_overlap_schedule / piecewise_cuda_graph_max_tokens /
    max_mamba_cache_size (from internal_states[0], the live scheduler
    process's own vars(get_global_server_args()) -- see PREREG_G2EA section
    on the Stage-1 probe for why this is the right field to read, as opposed
    to the top-level tokenizer-manager copy). tree_cache class name is NOT
    queried live (no HTTP-exposed field exists) -- it is stated analytically,
    the same way rev4 section 1.1 did: `disable_radix_cache=True` (common
    fixed flag, checked against the realized value) and `chunked_prefill_size
    not None` (checked against the realized value) together imply ChunkCache
    by scheduler.py:771-780 for every arm in this design (source-read
    2026-08-07); this script verifies the two preconditions live and reports
    the derived class name with that provenance made explicit, it does not
    assume the conclusion.
  FIX #7 (2026-08-09) gate enforcement on EVERY reported block: FIX #1 above
    enforced F-E on the PRIMARY cell verdict only. Every other block that
    prints a number (per-arm X_60, secondary_mixed_effect CI, tau ladder,
    per-arm secondary throughput/goodput/TTFT-p95/request-ITL, the Holm table,
    the Phase 0 capscan rows and the section-6 mid-check SDs) computed no gate
    at all, and a magnitude was in fact quoted out of the per-arm secondary
    block for an F-E-flagged arm. Every such block now carries an explicit
    `[GATE: ...]` label and a `gate` record in the JSON. This is methodology
    gate #17 / CONSENSUS section 3 item 31. It is DISPLAY ENFORCEMENT ONLY: no
    number is deleted, no verdict is recomputed, and every pre-existing field
    (cell_verdict / citable / tost / raw_ci / ttft_ci / holm p-values /
    underpowered) keeps byte-identical values -- verified by re-running both
    875657 and 875661 before and after and diffing. See the FIX #7 block
    comment above `GateBook` for the two-citability-fields caveat and for why
    G-2 is surfaced as `arm_discard_candidate` rather than folded into
    magnitude citability.

Primary estimands (PREREG_G2EA, TWO comparisons, both against A4=agnostic):
  cmp1: X_60(plainmix)     vs X_60(agnostic)   ("does bare mixed-chunk alone replace pdmux")
  cmp2: X_60(chunk512mix)  vs X_60(agnostic)   ("does chunk512+mixed-chunk replace pdmux")
Secondary (mixed-chunk's OWN effect, descriptive, no verdict):
  X_60(plainmix) - X_60(plain), X_60(chunk512mix) - X_60(chunk512)

Designated cell (PREREG_G2EA, moved from rev4's Granite r4): Granite r3.
Replication cells: Zamba2 r2, Zamba2 r3, Granite r4. Holm correction is
applied WITHIN each comparison's replication family (3 cells), separately
for cmp1 and cmp2, per-job (per-model) as this script only ever sees one
model's artifacts at a time -- exactly as rev4's g2_analyze.py did. A single
combined-3-cell-family view across BOTH models' job outputs (as PREREG_G2EA
section 5.3 describes the family) is a manual reconciliation step performed
by whoever reads both job reports together (result-analyst); this script
prints and persists per-job partial results, it does not attempt cross-job
JSON stitching (methodology gate #11: no cross-job numeric ingestion inside
a single job's own analysis).

This script computes measurements and applies the fixed decision rule ONLY.
Final interpretation/headline claims are result-analyst / claims-auditor's.
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

ARMS = ["plain", "plainmix", "chunk512", "chunk512mix", "agnostic"]
G2_ARMS = ["plainmix", "chunk512", "chunk512mix"]
TAU_PRIMARY = 60.0
TAU_LADDER = [40.0, 60.0, 100.0]
DELTA = 0.05
SCAN_RATES = [1, 2, 3, 4, 5, 6, 8, 10, 12]
NREP_SCAN = 3
NREP_MAIN = 10
VACUOUS_CEILING = 0.95  # FIX #5: was 0.99 in g2_analyze.py (rev4)

PRIMARY_COMPARISONS = [
    ("plainmix", "agnostic", "cmp1_A1m_vs_A4"),
    ("chunk512mix", "agnostic", "cmp2_A3m_vs_A4"),
]
SECONDARY_MIXED_EFFECT = [
    ("plainmix", "plain", "secondary_mixed_effect_on_plain"),
    ("chunk512mix", "chunk512", "secondary_mixed_effect_on_chunk512"),
]


# ---------------------------------------------------------------------------
# generic helpers (unchanged from g2_analyze.py)
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
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def request_max_itl_ms(row: dict) -> List[Optional[float]]:
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
    return {"X": n_viol / len(used), "n_used": len(used), "n_missing_itl": n_missing,
            "n_total": n_total, "n_viol_latency": n_viol}


def ttft_p95_ms(row: dict) -> Optional[float]:
    ttfts = sorted(t * 1000.0 for t in (row.get("ttfts") or []) if t is not None)
    return percentile(ttfts, 0.95)


def request_itl_p95_p90_ms(row: dict) -> Optional[float]:
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


SLO_TTFT_S = 3.0
SLO_ITL_S = 0.060


def goodput_canonical(row: dict) -> Tuple[Optional[float], int, int]:
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


def tost_equivalence(diffs: List[float], delta: float) -> Dict[str, Any]:
    n = len(diffs)
    if n < 2:
        return {"n": n, "verdict": "INSUFFICIENT_N"}
    mean = statistics.mean(diffs)
    sd = statistics.stdev(diffs)
    if sd == 0.0:
        return {"n": n, "mean": mean, "sd": 0.0, "verdict": "DEGENERATE-T",
                "note": "paired diff SD is exactly 0 -- no equivalence/superiority declaration."}
    se = sd / math.sqrt(n)
    df = n - 1
    tcrit90 = t_ppf(0.95, df)
    ci_lo = mean - tcrit90 * se
    ci_hi = mean + tcrit90 * se
    t1 = (mean - delta) / se
    t2 = (mean + delta) / se
    p1 = t_cdf(t1, df)
    p2 = 1 - t_cdf(t2, df)
    p_tost = max(p1, p2)
    equivalent = (ci_lo > -delta) and (ci_hi < delta)
    return {"n": n, "mean": mean, "sd": sd, "se": se, "delta": delta,
            "ci90_lo": ci_lo, "ci90_hi": ci_hi, "p_tost": p_tost,
            "equivalent": equivalent, "verdict": "EQUIVALENT" if equivalent else "NOT_EQUIVALENT"}


def sign_flip_permutation_p(diffs: List[float]) -> Optional[float]:
    n = len(diffs)
    if n == 0:
        return None
    if n > 20:
        return None
    obs = abs(sum(diffs))
    n_ge = 0
    n_perm = 2 ** n
    for signs in product([1, -1], repeat=n):
        s = sum(d * sgn for d, sgn in zip(diffs, signs))
        if abs(s) >= obs - 1e-12:
            n_ge += 1
    return n_ge / n_perm


def holm_adjust(pvals: List[Tuple[str, float]]) -> List[Tuple[str, float, float]]:
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
# FIX #7 (2026-08-09): GATE ENFORCEMENT ON *EVERY* REPORTED BLOCK
#
# Why this section exists. FIX #1 above wired F-E into the PRIMARY cell verdict
# only. Every other block that prints a number -- the per-arm X_60 line, the
# secondary_mixed_effect CI, the tau ladder, the per-arm secondary summary
# (throughput / goodput / TTFT p95 / request-ITL p90), the Holm table, and the
# Phase 0 capscan rows -- printed magnitudes derived from F-E-flagged arms with
# NO gate marking at all. A magnitude was in fact quoted through the per-arm
# secondary block. PREREG_GATE2_2026-08-06 sec3 states the consequence of a
# flag as a property of the ARM ("그 arm이 참여하는 비교는 크기 인용 금지, 부호만
# 보고"), so it binds every block printing a number derived from that arm, not
# just the pre-registered primary comparison. This is methodology gate #17 /
# CONSENSUS sec3 item 31: "게이트는 primary뿐 아니라 보고되는 모든 블록에 걸어라."
#
# SCOPE OF THIS SECTION -- DISPLAY ENFORCEMENT, NOT RE-SCORING:
#   * It NEVER deletes, suppresses, re-scores or re-verdicts a number. The
#     number is still printed and still persisted; an explicit label is
#     attached next to it (information preserved, citation status stated).
#   * No pre-registered decision rule is changed. In particular `cell_verdict`,
#     `citable`, `tost`, `raw_ci`, `ttft_ci`, `holm`, `underpowered` and every
#     other pre-existing field keep byte-identical values; this section only
#     ADDS `gate` records / label suffixes.
#   * Two DISTINCT citability fields therefore coexist and must not be
#     conflated:
#       - `citable` (pre-existing, primary only) = PREREG_G2EA sec7.1 item 1,
#         i.e. F-E of the two participating arms ONLY.
#       - `gate.magnitude_citable` (new, every block) = conjunction over every
#         pre-registered magnitude gate applicable to the arms in that block
#         (F-E on scored runs, F-A/F-B on capscan). On the 875657/875661
#         artifacts these two agree everywhere (F-B fires only on arms F-E
#         already fired on), so nothing changes hands; they are kept separate
#         so that a future divergence is visible rather than silent.
#   * G-2 is reported as `arm_discard_candidate`, NOT folded into
#     magnitude_citable: PREREG_GATE2 sec2.1's consequence of a G-2 mismatch is
#     "불일치 ⇒ A3 폐기" (discard the ARM), which is a scoping decision for
#     result-analyst/claims-auditor, not a magnitude-citation rule this scorer
#     may apply on its own.
# ---------------------------------------------------------------------------

def index_correctness(records: List[dict]) -> Dict[str, Dict[str, Any]]:
    """arm -> {G1_self, G1_cross, G2, G2_n_concurrent} from the correctness jsonl."""
    idx: Dict[str, Dict[str, Any]] = {}
    for rec in records or []:
        arm = rec.get("arm")
        if not arm:
            continue
        e = idx.setdefault(arm, {})
        gate = rec.get("gate")
        if gate == "G1":
            e["G1_self"] = rec.get("self_repro_status")
            e["G1_cross"] = rec.get("cross_vs_plain_match")
        elif gate == "G2_CONCURRENT":
            e["G2"] = rec.get("verdict")
            e["G2_n_concurrent"] = rec.get("n_concurrent")
    return idx


def _gate_fmt(parts: List[str]) -> str:
    """Label appended to a printed magnitude. Empty string when every gate
    applicable to that block is clear -- so an un-gated line stays byte-for-byte
    what it was before FIX #7 (that identity is what makes the primary-output
    diff proof meaningful)."""
    return ("  [GATE: " + " | ".join(parts) + "]") if parts else ""


class GateBook:
    """Single place that decides which pre-registered gate labels a printed
    magnitude carries. Constructed from the correctness artifact; Phase 0
    (F-A/F-B) and the sec6 mid-check are attached as they are computed."""

    def __init__(self, correctness_records: List[dict]) -> None:
        self.correctness = index_correctness(correctness_records)
        self.phase0: Dict[str, Any] = {}
        self.midcheck: Dict[str, Any] = {}

    # -- individual gates ---------------------------------------------------
    def _fe(self, f_e: Optional[Dict[str, Any]], arms: List[str]) -> Tuple[List[str], List[str]]:
        fired, undet = [], []
        for a in arms:
            fl = ((f_e or {}).get(a) or {}).get("flagged")
            if fl is True:
                fired.append(a)
            elif fl is None:
                undet.append(a)
        return fired, undet

    def _g2(self, arms: List[str]) -> Tuple[List[str], List[str]]:
        fail, undet = [], []
        for a in arms:
            v = (self.correctness.get(a) or {}).get("G2")
            if v is None:
                continue  # gate not run for this arm (prereg runs G-2 on 3 arms)
            sv = str(v).upper()
            if sv == "PASS":
                continue
            if sv == "FAIL":
                fail.append(a)
            else:
                undet.append(f"{a}={v}")
        return fail, undet

    def _g1(self, arms: List[str]) -> List[str]:
        fail = []
        for a in arms:
            e = self.correctness.get(a) or {}
            if not e:
                continue
            if e.get("G1_self") not in (None, "SELFREPRO_OK"):
                fail.append(f"{a}:self={e.get('G1_self')}")
            xc = e.get("G1_cross")
            if xc not in (None, "n/a", "True", True):
                fail.append(f"{a}:cross={xc}")
        return fail

    def _fa(self, arms: List[str], rate: Optional[int]) -> List[str]:
        out = []
        for a in arms:
            rep = self.phase0.get(a) or {}
            hits = [f for f in (rep.get("F_A") or [])
                    if rate is None or f.get("rate") == rate]
            if hits:
                out.append(a)
        return out

    def _fb(self, arms: List[str], rate: Optional[int]) -> List[str]:
        out = []
        for a in arms:
            rep = self.phase0.get(a) or {}
            if not rep.get("F_B_flagged"):
                continue
            mults = rep.get("F_B_rate1_mult") or []
            slope = (rep.get("F_B_slope") or {}).get("ci_excludes_0")
            if rate is None or slope or any(m.get("rate") == rate for m in mults):
                out.append(a)
        return out

    def _underpowered(self, comparison: Optional[str], rate: Optional[int]) -> Optional[bool]:
        if comparison is None or rate is None:
            return None
        cell = ((self.midcheck.get(comparison) or {}).get(rate)) or {}
        if cell.get("status") != "OK":
            return None
        return bool(cell.get("underpowered"))

    # -- composite records --------------------------------------------------
    def _record(self, arms: List[str], rate: Optional[int], comparison: Optional[str],
                dataset: str, f_e: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        fe_fired, fe_undet = ([], []) if dataset != "scored" else self._fe(f_e, arms)
        fa = self._fa(arms, rate)
        fb = self._fb(arms, rate)
        g2_fail, g2_undet = self._g2(arms)
        g1_fail = self._g1(arms)
        under = self._underpowered(comparison, rate)

        parts: List[str] = []
        if fe_fired:
            parts.append("F-E FIRED(%s) => SIGN ONLY, MAGNITUDE NOT CITABLE" % ",".join(fe_fired))
        if fe_undet:
            parts.append("F-E UNDETERMINED(%s) => gate not evaluable" % ",".join(fe_undet))
        if dataset == "capscan" and fa:
            parts.append("F-A SATURATION(%s) => MAGNITUDE NOT CITABLE" % ",".join(fa))
        elif fa:
            parts.append("F-A SATURATION@capscan(%s) => MAGNITUDE NOT CITABLE" % ",".join(fa))
        if dataset == "capscan" and fb:
            parts.append("F-B TTFT-RUNAWAY(%s) => MAGNITUDE NOT CITABLE" % ",".join(fb))
        elif fb:
            parts.append("F-B TTFT-RUNAWAY@capscan(%s) => MAGNITUDE NOT CITABLE" % ",".join(fb))
        if g2_fail:
            parts.append("G-2 CONCURRENT FAIL(%s) => PREREG_GATE2 sec2.1 inherited "
                         "discard rule applies to this arm (retraction candidate); "
                         "this scorer does NOT re-score" % ",".join(g2_fail))
        if g2_undet:
            parts.append("G-2 UNDETERMINED(%s)" % ",".join(g2_undet))
        if g1_fail:
            parts.append("G-1 FAIL(%s)" % ",".join(g1_fail))
        if under:
            parts.append("PHASE0-MIDCHECK UNDERPOWERED => PREREG_G2EA sec6 forbids an "
                         "equivalence (Rprime3_analog) verdict in this cell")

        return {
            "arms": list(arms),
            "rate": rate,
            "comparison": comparison,
            "dataset": dataset,
            "f_e_fired": fe_fired,
            "f_e_undetermined": fe_undet,
            "f_a_flagged": fa,
            "f_b_flagged": fb,
            "g2_concurrent_fail": g2_fail,
            "g2_concurrent_undetermined": g2_undet,
            "g1_fail": g1_fail,
            "midcheck_underpowered": under,
            "magnitude_citable": not (fe_fired or fa or fb),
            "arm_discard_candidate": bool(g2_fail),
            "label": _gate_fmt(parts),
        }

    def scored(self, f_e: Optional[Dict[str, Any]], arms: List[str],
               rate: Optional[int] = None, comparison: Optional[str] = None) -> Dict[str, Any]:
        """Gate record for a block computed from the Phase 1 SCORED runs."""
        return self._record(arms, rate, comparison, "scored", f_e)

    def capscan(self, arms: List[str], rate: Optional[int] = None,
                comparison: Optional[str] = None) -> Dict[str, Any]:
        """Gate record for a block computed from the Phase 0 CAPSCAN runs.
        F-E is deliberately NOT applied here: PREREG_GATE2 sec3 requires F-A/F-B
        to be computed on capscan and F-E on the scored runs, and forbids mixing
        the two datasets ('혼용 금지')."""
        return self._record(arms, rate, comparison, "capscan", None)


# ---------------------------------------------------------------------------
# realized flags (FIX #3/#6)
# ---------------------------------------------------------------------------

def realized_flags_report(dirpath: str, tag: str, jobid: str) -> Dict[str, Any]:
    """Reads one flags json per arm (prefers the G1 boot; falls back to the
    first capscan_seed7 boot) and extracts the realized values that matter
    for this campaign's mechanism claims."""
    out: Dict[str, Any] = {}
    print(f"\n=== realized flags per arm (FIX #3/#6): tag={tag} jobid={jobid} ===")
    for arm in ARMS:
        candidates = [
            os.path.join(dirpath, f"g2ea_{tag}_flags_{arm}_G1_{jobid}.json"),
            os.path.join(dirpath, f"g2ea_{tag}_flags_{arm}_capscan_seed7_{jobid}.json"),
        ]
        path = next((p for p in candidates if os.path.exists(p)), None)
        if path is None:
            print(f"  arm={arm:12s} NO FLAGS ARTIFACT FOUND (checked {candidates})")
            out[arm] = {"status": "MISSING"}
            continue
        try:
            d = json.load(open(path))
        except Exception as e:
            print(f"  arm={arm:12s} PARSE_FAILED {path}: {e}")
            out[arm] = {"status": "PARSE_FAILED"}
            continue
        top = {
            "enable_mixed_chunk": d.get("enable_mixed_chunk"),
            "chunked_prefill_size": d.get("chunked_prefill_size"),
            "disable_overlap_schedule": d.get("disable_overlap_schedule"),
            "piecewise_cuda_graph_max_tokens": d.get("piecewise_cuda_graph_max_tokens"),
            "disable_radix_cache": d.get("disable_radix_cache"),
        }
        states = d.get("internal_states") or []
        sched = {}
        if states:
            s0 = states[0]
            sched = {
                "enable_mixed_chunk": s0.get("enable_mixed_chunk"),
                "chunked_prefill_size": s0.get("chunked_prefill_size"),
                "disable_overlap_schedule": s0.get("disable_overlap_schedule"),
                "disable_radix_cache": s0.get("disable_radix_cache"),
                "max_mamba_cache_size": s0.get("max_mamba_cache_size"),
            }
        # tree_cache class derivation (analytic, source-read 2026-08-07,
        # scheduler.py:771-780): ChunkCache iff
        # (chunked_prefill_size is not None) AND disable_radix_cache.
        cps = sched.get("chunked_prefill_size", top.get("chunked_prefill_size"))
        drc = sched.get("disable_radix_cache", top.get("disable_radix_cache"))
        if cps is not None and drc:
            tree_cache_cls = "ChunkCache (derived: chunked_prefill_size is not None and disable_radix_cache, scheduler.py:771-780)"
        else:
            tree_cache_cls = f"UNDETERMINED (precondition not met live: chunked_prefill_size={cps!r} disable_radix_cache={drc!r})"
        out[arm] = {"status": "OK", "source_file": path, "top": top, "scheduler": sched,
                    "tree_cache_class_derived": tree_cache_cls}
        print(f"  arm={arm:12s} top={top}")
        print(f"                scheduler={sched}")
        print(f"                tree_cache={tree_cache_cls}")
    return out


# ---------------------------------------------------------------------------
# Phase 0 (capacity scan) -- F-A/F-B/F-E + section 5.1.1 mid-check
# (structure unchanged from g2_analyze.py; FIX #4 = caller now persists the
#  return values instead of discarding them after printing)
# ---------------------------------------------------------------------------

def phase0_load(dirpath: str, tag: str, jobid: str, arm: str) -> Dict[Tuple[int, int], dict]:
    out = {}
    for seed in (7, 17, 27):
        path = os.path.join(dirpath, f"g2ea_{tag}_capscan_{arm}_seed{seed}_{jobid}.jsonl")
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
            report[arm] = {"status": "NO_DATA"}
            continue
        print(f"  --- arm={arm} ---")
        arm_report: Dict[str, Any] = {"F_A": [], "F_B_slope": None, "F_B_rate1_mult": []}
        ttft_by_rate_seed: Dict[int, Dict[int, float]] = {}
        # FIX #7: the per-(rate,seed) rows below print magnitudes (ttft_p50,
        # req_thpt) that the ARM-LEVEL F-B verdict gates, but F-B is only known
        # after the whole scan has been walked. Buffer the rows, then emit them
        # with the F-B label attached. Print ORDER is unchanged (rows, then the
        # F-B summary line) and an F-B-clear row is byte-identical to before.
        row_lines: List[Tuple[int, str]] = []
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
                    row_lines.append((rate,
                          f"  rate={rate} seed={seed} compl/total={n_completed}/{n_total} "
                          f"ttft_p50={(row.get('median_ttft_ms') or float('nan')):.1f}ms "
                          f"req_thpt={(row.get('request_throughput') or float('nan')):.3f} F-A={flagA}"))
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
        arm_report["F_B_flagged"] = fb_flagged
        # FIX #7: emit the buffered capscan rows with their own gate label.
        one_arm = GateBook([])
        one_arm.phase0 = {arm: arm_report}
        row_gates: Dict[int, Dict[str, Any]] = {}
        for rate, line in row_lines:
            if rate not in row_gates:
                row_gates[rate] = one_arm.capscan([arm], rate=rate)
            print(line + row_gates[rate]["label"])
        arm_report["magnitude_gate_by_rate"] = {r: g for r, g in row_gates.items()}
        print(f"  F-B (global): rate1_mult>4.0 flags={fb_flags} slope_ci_excludes_0={slope_ci_excludes_0} "
              f"-> {'FLAGGED' if fb_flagged else 'clear'}")
        report[arm] = arm_report
    return report


def phase0_midcheck(dirpath: str, tag: str, jobid: str, main_rates: List[int],
                     gates: Optional["GateBook"] = None) -> Dict[str, Any]:
    """PREREG_G2EA / rev4 section 5.1.1 analog: for BOTH primary comparisons,
    compare the mixed/chunk arm's capscan X_60 SD against sqrt(2)*SD(A4)."""
    print(f"\n=== Phase 0 mid-check (variance proxy, both primary comparisons): tag={tag} jobid={jobid} ===")
    out: Dict[str, Any] = {}
    a4 = phase0_load(dirpath, tag, jobid, "agnostic")
    for arm_a, arm_b, label in PRIMARY_COMPARISONS:
        a = phase0_load(dirpath, tag, jobid, arm_a)
        out[label] = {}
        for rate in main_rates:
            xa, xb = [], []
            for seed in (7, 17, 27):
                ra = a.get((seed, rate))
                rb = a4.get((seed, rate))
                if ra is not None:
                    v = x_tau(ra, TAU_PRIMARY)["X"]
                    if v is not None:
                        xa.append(v)
                if rb is not None:
                    v = x_tau(rb, TAU_PRIMARY)["X"]
                    if v is not None:
                        xb.append(v)
            if len(xa) < 2 or len(xb) < 2:
                print(f"  {label} rate={rate}: insufficient capscan data (n_A={len(xa)} n_A4={len(xb)})")
                out[label][rate] = {"status": "INSUFFICIENT_DATA", "n_a": len(xa), "n_a4": len(xb)}
                continue
            sda = statistics.stdev(xa)
            sd4 = statistics.stdev(xb)
            proxy = math.sqrt(2) * sd4
            underpowered = sda > 2 * proxy
            # FIX #7: these SDs are capscan magnitudes from the two participating
            # arms, so the capscan gates (F-A/F-B) that flag those arms bind here
            # too. Display-only: `underpowered` itself is untouched.
            grec = (gates.capscan([arm_a, arm_b], rate=rate, comparison=label)
                    if gates is not None else None)
            print(f"  {label} rate={rate}: SD(arm)={sda:.4f} sqrt2*SD(A4)={proxy:.4f} "
                  f"2x_proxy={2*proxy:.4f} -> {'UNDERPOWERED' if underpowered else 'OK'}"
                  f"{(grec or {}).get('label', '')}")
            out[label][rate] = {"status": "OK", "sd_arm": sda, "sd_a4": sd4, "proxy": proxy,
                                 "underpowered": underpowered}
            if grec is not None:
                out[label][rate]["gate"] = grec
    return out


# ---------------------------------------------------------------------------
# Phase 1 (main measurement)
# ---------------------------------------------------------------------------

def load_voided_reps(dirpath: str, tag: str, jobid: str) -> set:
    path = os.path.join(dirpath, f"g2ea_{tag}_voided_reps_{jobid}.txt")
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
        path = os.path.join(dirpath, f"g2ea_{tag}_{arm}_rep{rep}_{jobid}.jsonl")
        rows = load_jsonl(path)
        by_rate = {}
        for r in rows:
            rr = r.get("request_rate")
            if rr is not None:
                by_rate[int(rr)] = r
        if rate in by_rate:
            out[rep] = by_rate[rate]
    return out


def phase1_fE(dirpath: str, tag: str, jobid: str, rate: int, voided: set) -> Dict[str, Any]:
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


def compare_arms(dirpath: str, tag: str, jobid: str, rate: int, voided: set,
                  per_arm_rows: Dict[str, Dict[int, dict]], per_arm_x60: Dict[str, Dict[int, float]],
                  arm_a: str, arm_b: str, label: str, f_e: Dict[str, Any],
                  gates: Optional["GateBook"] = None) -> Dict[str, Any]:
    """Generic 2-arm TOST/superiority comparison, direction = X(arm_a) - X(arm_b).
    ci_lo>0 (arm_a worse, i.e. higher violation rate) -> arm_b superior ("Rprime4"-analog).
    ci_hi<0 (arm_a better) -> arm_a superior."""
    # FIX #7: the gate label is resolved BEFORE any magnitude is printed, so that
    # every magnitude line of this block carries it (rev-EA resolved F-E only at
    # the end, in the verdict line, leaving the four magnitude lines above it
    # quotable in isolation). The verdict/citability computation further down is
    # untouched.
    grec = (gates.scored(f_e, [arm_a, arm_b], rate=rate, comparison=label)
            if gates is not None else None)
    glab = (grec or {}).get("label", "")
    common_reps = sorted(set(per_arm_x60[arm_a]) & set(per_arm_x60[arm_b]))
    diffs = [per_arm_x60[arm_a][r] - per_arm_x60[arm_b][r] for r in common_reps]
    print(f"  [{label}] paired {arm_a}-{arm_b} X_60 diffs (n={len(diffs)}, reps={common_reps}): "
          f"{['%.4f' % d for d in diffs]}{glab}")
    tost = tost_equivalence(diffs, DELTA)
    raw_ci = paired_t_ci_twosided(diffs)
    perm_p = sign_flip_permutation_p(diffs) if diffs else None
    print(f"  [{label}] TOST(delta={DELTA}): {tost}{glab}")
    print(f"  [{label}] raw paired-t 95% CI: {raw_ci}{glab}")
    print(f"  [{label}] sign-flip permutation p: {perm_p}{glab}")

    b_superior = False  # arm_b (normally A4) superior
    a_superior = False
    if not raw_ci.get("degenerate") and raw_ci.get("ci_lo") is not None:
        if raw_ci["ci_lo"] > 0:
            b_superior = True
        elif raw_ci["ci_hi"] < 0:
            a_superior = True

    ttft_diffs = []
    common_ttft = sorted(set(per_arm_rows[arm_a]) & set(per_arm_rows[arm_b]))
    for rep in common_ttft:
        pa = ttft_p95_ms(per_arm_rows[arm_a][rep])
        pb = ttft_p95_ms(per_arm_rows[arm_b][rep])
        if pa is not None and pb:
            ttft_diffs.append((pa - pb) / pb)
    ttft_ci = paired_t_ci_twosided(ttft_diffs) if ttft_diffs else {"n": 0}
    ttft_noninferior = ttft_ci.get("ci_hi") is not None and ttft_ci["ci_hi"] <= 0.10
    print(f"  [{label}] TTFT p95 relative diff ({arm_a}-{arm_b})/{arm_b}, n={len(ttft_diffs)}: {ttft_ci} "
          f"-> noninferior(upper<=+10%)={ttft_noninferior}{glab}")

    if tost.get("verdict") == "DEGENERATE-T":
        verdict = "DEGENERATE-T"
    elif tost.get("equivalent") and ttft_noninferior:
        verdict = "Rprime3_analog(equivalent)"
    elif b_superior:
        verdict = "Rprime4_analog(A4_superior)"
    elif a_superior:
        verdict = f"{arm_a}_SUPERIOR"
    else:
        verdict = "INCONCLUSIVE"

    # FIX #1: fold F-E flag into the verdict/citability instead of leaving it
    # a dangling, unconsulted field.
    fe_a = f_e.get(arm_a, {}).get("flagged")
    fe_b = f_e.get(arm_b, {}).get("flagged")
    flagged = bool(fe_a) or bool(fe_b)
    citable = not flagged
    if flagged:
        verdict = verdict + "_FLAGGED_SIGN_ONLY_MAGNITUDE_NOT_CITABLE"
    print(f"  [{label}] CELL VERDICT (mechanical, pre-Holm) = {verdict}  "
          f"(F-E flagged: {arm_a}={fe_a} {arm_b}={fe_b} -> citable={citable}){glab}")

    out = {"label": label, "arm_a": arm_a, "arm_b": arm_b, "n_paired": len(diffs), "diffs": diffs,
           "tost": tost, "raw_ci": raw_ci, "sign_flip_p": perm_p, "ttft_noninferior": ttft_noninferior,
           "ttft_ci": ttft_ci, "cell_verdict": verdict, "citable": citable,
           "f_e_flagged": {"arm_a": fe_a, "arm_b": fe_b}}
    if grec is not None:
        out["gate"] = grec  # FIX #7: additive; `cell_verdict`/`citable` unchanged
    return out


def secondary_mixed_effect(per_arm_x60: Dict[str, Dict[int, float]], arm_mix: str, arm_base: str) -> Dict[str, Any]:
    common = sorted(set(per_arm_x60[arm_mix]) & set(per_arm_x60[arm_base]))
    diffs = [per_arm_x60[arm_mix][r] - per_arm_x60[arm_base][r] for r in common]
    ci = paired_t_ci_twosided(diffs)
    return {"arm_mix": arm_mix, "arm_base": arm_base, "n": len(diffs), "diffs": diffs, "ci": ci}


def phase1_cell_report(dirpath: str, tag: str, jobid: str, rate: int, voided: set,
                        f_e: Dict[str, Any], gates: Optional["GateBook"] = None) -> Dict[str, Any]:
    print(f"\n--- rate={rate} ---")
    per_arm_rows: Dict[str, Dict[int, dict]] = {a: phase1_load(dirpath, tag, jobid, a, rate, voided) for a in ARMS}
    per_arm_x60: Dict[str, Dict[int, float]] = {}
    per_arm_gate: Dict[str, Any] = {}
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
        # FIX #7: the per-arm X_60 mean IS the primary estimand's per-arm level.
        # rev-EA printed it with no gate marking whatsoever.
        grec = gates.scored(f_e, [arm], rate=rate) if gates is not None else None
        if grec is not None:
            per_arm_gate[arm] = grec
        if vals:
            print(f"  arm={arm:12s} n={len(vals):2d} X_60 mean={statistics.mean(vals):.4f} "
                  f"reps={sorted(xs.keys())} n_missing_itl_total={n_missing_total} "
                  f"values={['%.4f' % v for v in vals]}{(grec or {}).get('label', '')}")
        else:
            print(f"  arm={arm:12s} NO DATA")

    primary: Dict[str, Any] = {}
    for arm_a, arm_b, label in PRIMARY_COMPARISONS:
        primary[label] = compare_arms(dirpath, tag, jobid, rate, voided, per_arm_rows, per_arm_x60,
                                       arm_a, arm_b, label, f_e, gates)

    secondary_mix: Dict[str, Any] = {}
    for arm_mix, arm_base, label in SECONDARY_MIXED_EFFECT:
        s = secondary_mixed_effect(per_arm_x60, arm_mix, arm_base)
        secondary_mix[label] = s
        # FIX #7: descriptive-only does NOT mean gate-free -- this CI is a
        # magnitude built from the same scored runs F-E gates.
        grec = gates.scored(f_e, [arm_mix, arm_base], rate=rate) if gates is not None else None
        print(f"  [{label}] {arm_mix}-{arm_base} X_60 diff: n={s['n']} ci={s['ci']}"
              f"{(grec or {}).get('label', '')}")
        if grec is not None:
            s["gate"] = grec

    # tau ladder Delta_chunk = X_tau(chunk512) - X_tau(plain), VACUOUS-screened
    # (FIX #5: ceiling threshold lowered 0.99 -> VACUOUS_CEILING=0.95)
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
            both_ceiling = (x1["X"] >= VACUOUS_CEILING) and (x3["X"] >= VACUOUS_CEILING)
            vacuous_flags.append(both_floor or both_ceiling)
        n_vacuous = sum(vacuous_flags)
        non_vacuous_d = [dd for dd, v in zip(d, vacuous_flags) if not v]
        sign = None
        if non_vacuous_d:
            sign = "+" if statistics.mean(non_vacuous_d) > 0 else ("-" if statistics.mean(non_vacuous_d) < 0 else "0")
        ladder[tau] = {"n": len(d), "n_vacuous": n_vacuous,
                        "mean_diff_nonvacuous": (statistics.mean(non_vacuous_d) if non_vacuous_d else None),
                        "sign": sign}
    # FIX #7: the ladder is a magnitude over the plain/chunk512 arms; VACUOUS was
    # the only screen it carried, F-E/G-2 were never consulted.
    ladder_gate = gates.scored(f_e, ["chunk512", "plain"], rate=rate) if gates is not None else None
    print(f"  tau ladder Delta_chunk=X_tau(chunk512)-X_tau(plain) sign (VACUOUS-screened, "
          f"ceiling>={VACUOUS_CEILING}): {ladder}{(ladder_gate or {}).get('label', '')}")
    if ladder_gate is not None:
        for tau in ladder:  # attach after printing so the printed dict is unchanged
            ladder[tau]["gate"] = ladder_gate

    # FIX #7 -- THIS IS THE BLOCK THE 2026-08-09 MIS-CITATION CAME THROUGH.
    # rev-EA computed F-E and never consulted it here, so an F-E-flagged arm's
    # goodput / TTFT p95 / request-ITL magnitudes printed clean.
    secondary: Dict[str, Any] = {}
    for arm in ARMS:
        rows = per_arm_rows[arm]
        if not rows:
            continue
        thpt = [row.get("request_throughput") for row in rows.values() if row.get("request_throughput") is not None]
        gp_canon = [goodput_canonical(row)[0] for row in rows.values()]
        gp_canon = [g for g in gp_canon if g is not None]
        ttft95 = [ttft_p95_ms(row) for row in rows.values() if ttft_p95_ms(row) is not None]
        reqitl_p90 = [request_itl_p95_p90_ms(row) for row in rows.values()]
        reqitl_p90 = [v for v in reqitl_p90 if v is not None]
        secondary[arm] = {
            "n": len(rows),
            "request_throughput_mean": statistics.mean(thpt) if thpt else None,
            "goodput_canonical_mean": statistics.mean(gp_canon) if gp_canon else None,
            "ttft_p95_mean_ms": statistics.mean(ttft95) if ttft95 else None,
            "request_itl_p95_p90_mean_ms": statistics.mean(reqitl_p90) if reqitl_p90 else None,
        }
        grec = per_arm_gate.get(arm)
        print(f"  secondary arm={arm:12s} {secondary[arm]}{(grec or {}).get('label', '')}")
        if grec is not None:
            secondary[arm]["gate"] = grec  # after printing: printed dict unchanged

    out = {"rate": rate, "primary": primary, "secondary_mixed_effect": secondary_mix,
           "tau_ladder": ladder, "secondary": secondary, "per_arm_x60": per_arm_x60,
           "f_e": f_e}
    if gates is not None:
        out["per_arm_gate"] = per_arm_gate
        out["tau_ladder_gate"] = ladder_gate
    return out


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
                     help="PREREG_G2EA: Granite=3 (moved from rev4's 4); Zamba2=None (no designated cell)")
    args = ap.parse_args()

    voided = load_voided_reps(args.dir, args.tag, args.jobid)
    if voided:
        print(f"VOIDED REPS (excluded from ALL arms): {sorted(voided)}")

    corr_path = os.path.join(args.dir, f"g2ea_{args.tag}_correctness_{args.jobid}.json")
    correctness_records = []
    if os.path.exists(corr_path):
        print(f"\n=== correctness gate summary ({corr_path}) ===")
        for rec in load_jsonl(corr_path):
            print(f"  {rec}")
            correctness_records.append(rec)
    else:
        print(f"\n*** NO correctness-gate artifact found at {corr_path} ***")

    # FIX #7: one GateBook, consulted by every block that prints a magnitude.
    gates = GateBook(correctness_records)
    _g2_fail_arms = sorted(a for a, e in gates.correctness.items()
                           if str(e.get("G2", "")).upper() == "FAIL")
    if _g2_fail_arms:
        print(f"  *** G-2 CONCURRENT FAIL arms: {_g2_fail_arms} -- PREREG_GATE2 sec2.1's "
              f"inherited rule ('불일치 => A3 폐기') applies to these arms. Every block below "
              f"that reports a number from them is labelled accordingly. This scorer does NOT "
              f"act on the rule (arm discard / retraction is result-analyst + claims-auditor's "
              f"call); it only refuses to print those numbers unlabelled. ***")

    flags_report = realized_flags_report(args.dir, args.tag, args.jobid)
    phase0_rep = phase0_report(args.dir, args.tag, args.jobid, args.rates)
    gates.phase0 = phase0_rep
    midcheck_rep = phase0_midcheck(args.dir, args.tag, args.jobid, args.rates, gates)
    gates.midcheck = midcheck_rep

    print(f"\n=== Phase 1 primary (n<={NREP_MAIN} paired, tag={args.tag} jobid={args.jobid}) ===")
    cell_results = {}
    for rate in args.rates:
        f_e = phase1_fE(args.dir, args.tag, args.jobid, rate, voided)
        print(f"  F-E (scored-run tercile TTFT-p50 ratio, threshold 1.7): {f_e}")
        cell_results[rate] = phase1_cell_report(args.dir, args.tag, args.jobid, rate, voided, f_e, gates)

    print("\n=== Holm correction across replication cells (within THIS job's own rates; "
          "designated cell stands alone; cross-job family reconciliation is a manual step) ===")
    holm_out: Dict[str, Any] = {}
    for arm_a, arm_b, label in PRIMARY_COMPARISONS:
        repl_pvals = []
        repl_rate_by_label: Dict[str, int] = {}
        for rate, res in cell_results.items():
            if rate == args.designated_rate:
                continue
            p = res["primary"][label]["tost"].get("p_tost")
            if p is not None:
                repl_pvals.append((f"{args.tag}_r{rate}", p))
                repl_rate_by_label[f"{args.tag}_r{rate}"] = rate
        if repl_pvals:
            adjusted = holm_adjust(repl_pvals)
            holm_out[label] = [{"cell": lab, "p_raw": praw, "p_holm": padj} for lab, praw, padj in adjusted]
            # FIX #7: the Holm table reports p-values per cell; the cell's own
            # gate record decides whether that cell's numbers may be quoted.
            for i, (lab, praw, padj) in enumerate(adjusted):
                cell_rate = repl_rate_by_label.get(lab)
                cg = ((cell_results.get(cell_rate) or {}).get("primary", {})
                      .get(label, {}) or {}).get("gate")
                print(f"  [{label}] {lab}: p_raw={praw:.4f} p_holm={padj:.4f} -> "
                      f"{'PASS(<0.05)' if padj < 0.05 else 'fail'}{(cg or {}).get('label', '')}")
                if cg is not None:
                    holm_out[label][i]["gate"] = cg
        if args.designated_rate is not None and args.designated_rate in cell_results:
            d = cell_results[args.designated_rate]["primary"][label]
            print(f"  [{label}] DESIGNATED cell rate={args.designated_rate}: verdict={d['cell_verdict']} "
                  f"citable={d['citable']} (single-cell, no multiplicity correction)"
                  f"{(d.get('gate') or {}).get('label', '')}")

    out_json = os.path.join(args.dir, f"g2ea_report_{args.tag}_{args.jobid}.json")
    with open(out_json, "w") as f:
        json.dump({
            "tag": args.tag, "jobid": args.jobid, "voided_reps": sorted(voided),
            "rates": args.rates, "designated_rate": args.designated_rate,
            "correctness_gates": correctness_records,
            "realized_flags": flags_report,
            # FIX #4: Phase 0 F-A/F-B and the 5.1.1-analog mid-check are now
            # persisted (previously stdout-only in g2_analyze.py).
            "phase0_report": phase0_rep,
            "phase0_midcheck": midcheck_rep,
            "cells": cell_results,
            "holm": holm_out,
        }, f, indent=2, default=str)
    print(f"\nFull report -> {out_json}")
    print("\n*** This script computes measurements and the mechanical pre-registered rule ONLY. ***")
    print("*** Final interpretation/headline claims are result-analyst / claims-auditor's job. ***")
    return 0


if __name__ == "__main__":
    sys.exit(main())
