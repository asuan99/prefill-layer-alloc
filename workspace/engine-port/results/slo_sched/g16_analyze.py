#!/usr/bin/env python3
"""g16_analyze.py -- pre-registered analyzer for gate #16 (G16 grid campaign).

SPEC: ``PREREG_G16_RULES_REV3_2026-08-16.md`` (the ONLY valid revision; rev1/rev2
are superseded).  This file implements, verbatim:

  * sec4      estimands ``M_ttft`` / ``M_itl`` / ``S_itl(.)`` / ``Delta`` /
              ``Delta_SLO(.)``  (canonical library calls, no re-definition)
  * sec6      donor identification (K1), the NINE verdicts, delta/K7 saturation,
              the pre-declared adaptive rule, the uncertainty conventions
  * sec7      positive controls PC-A .. PC-E  (+ PC-B-neg, see below)
  * sec11     knob registry K1..K10 as module constants
  * add. A-2  every reported split quantity is labelled "nominal split of the
              CO-RESIDENT interval" (53-68% of decode-active samples run at
              (0,108); the arm label is conditional, not unconditional)
  * add. A-3  drift-corrected variant computed ALONGSIDE the primary (never
              instead of it); a donor disagreement between the two is itself
              a registered result
  * add. B-5(a)2  boot adoption rule
  * add. B-5(a)3  campaign glob fixed to ``*_LO.jsonl`` / ``*_HI.jsonl``

NON-GOALS (sec10): this analyzer does not close gate #16 literally.  It closes
the RE-FORMULATED decision quantity 2 (a location-statistic version); the
threshold-predicate version stays on the rate axis.  Every emitted headline
string carries that scope, and the SM-sum algebra is stated for the NOMINAL
split of the co-resident interval only.

USAGE
  python3 g16_analyze.py --campaign            # the real thing (needs blk1..4)
  python3 g16_analyze.py --controls            # PC-A..PC-E + PC-B-neg (GPU 0)
  python3 g16_analyze.py --self-test           # verdict reachability + calib.
  python3 g16_analyze.py --controls --self-test --out report.json

Provenance of every positive-control target: ``ORACLE_REANALYSIS_2026-08-16.md``
sec3-2 (PC-A) and ``audit_g16_independent_2026-08-16/`` fragments A/B/C/D/E
(PC-D, PC-E) -- the auditor's independent reimplementation, preserved verbatim
in this repo precisely so these targets have a non-``pdmux_eval`` source.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import random
import re
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
ENGINE_PORT = HERE.parent.parent
sys.path.insert(0, str(ENGINE_PORT / "benchmarks"))

from pdmux_eval import analyze as pdmux  # noqa: E402  (canonical library)

# ---------------------------------------------------------------------------
# sec11 -- knob registry.  Pre-registered; changing any of these invalidates
# the pre-registration (gate #19: no post-hoc knob selection).
# ---------------------------------------------------------------------------

K1_BLOCK_ARGMIN_MIN = 3          # donor rank rule: argmin in >= 3 of 4 blocks
K1_BOOTSTRAP_MIN_FRAC = 0.80     # donor size rule: block-bootstrap freq >= 80%
K2_BAND_FRACTION = 0.10          # +-10% cliff band around a threshold
K2_BAND_MASS_MAX = 0.25          # > 25% band mass  =>  threshold donor is
                                 # DIAGNOSTIC ONLY (gate #6 routing)
K3_BOOTSTRAP_SAMPLES = 10000     # canonical
K3_BOOTSTRAP_SEED = 1            # canonical
K4_GRID_MAX_ARM = "d74"          # prefill 34 SM; largest decode arm with a
                                 # boot record (g2_0_hard)
K5_BLOCKS = 4                    # forward/reverse pairs => mean position 3.0
K6_RATE_LO, K6_RATE_HI = 3, 12
K6_NP, K6_ROUNDS = 200, 3
K7_DELTA_MS = 1.0                # ITL saturation tolerance (boot-internal
                                 # dispersion is 1.5-2.1 ms; this is a
                                 # conservative LOWER bound on "same")
K8_SLO_LADDER = tuple(range(45, 81))   # 45..80 ms, 1 ms steps  (C2')
K9_ADAPTIVE_EXTRA_BLOCKS = 2     # only when upper gap >= delta AND blocks
                                 # disagree; never otherwise
K10_PIN_FRAC_NEVER_DISCARDS = True   # add. A-1: co-residency fraction NEVER
                                 # discards an arm.  The ONLY discard criterion
                                 # is realized-probe ``exact is not True``.
K11_UPPER_ARM_MIN_SM = 44        # prereg addendum C (2026-08-17): sec6's
                                 # "upper arms" is PINNED at decode SM >= 44,
                                 # not a floating top-half and NOT the
                                 # contending set.  On the 7-arm campaign grid
                                 # this equals top-half-by-SM exactly; pinning
                                 # matters because a floating top-half loses
                                 # d44 as soon as one LOWER arm is discarded
                                 # by H3'-a.

TTFT_SLO_MS = 3000.0             # canonical secondary predicate thresholds
ITL_SLO_MS = 60.0
OPERATING_POINT_SLO_MS = 60.0
# add. B-2 / N7 band that MUST accompany any 60 ms point estimate of Delta_SLO
DELTA_SLO_60_BAND = "[+0.34%, -1.92%]"

# add. A-2: mandatory scope prefix for every split-valued headline.
NOMINAL_SPLIT_SCOPE = (
    "nominal split of the co-resident interval "
    "(53-68% of decode-active samples run at (0,108); arm labels are "
    "conditional, not unconditional)"
)
# sec1 C1 / sec3-1: mandatory framing.  Never "TTFT buys prefill SM"; never
# "gate #16 closed".
HEADLINE_FRAME = (
    "does the split that ITL requires differ from the THROUGHPUT-OPTIMAL "
    "split? (closes the re-formulated decision quantity 2; the threshold "
    "version remains on the rate axis)"
)

_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
         7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}

VERDICTS = (
    "TAX_POSITIVE", "NO_TAX", "NEGATIVE_INTERIOR", "ITL_UNCONSTRAINED",
    "S_ITL_UNREACHED", "TRUNCATED", "TRUNCATED_LOW", "ITL_SATURATED",
    "UNIDENTIFIED",
)

ARM_RE = re.compile(r"^d(\d+)$")


def arm_sm(arm: str) -> int:
    match = ARM_RE.match(arm)
    if not match:
        raise ValueError(f"unparseable arm label: {arm!r}")
    return int(match.group(1))


def t_ci(values: Sequence[float]) -> Dict[str, float]:
    """t(n-1) interval.  sec6: percentile CI is NEVER cited on its own."""
    vals = [float(v) for v in values if math.isfinite(v)]
    if len(vals) < 2:
        return {"n": float(len(vals)), "mean": vals[0] if vals else math.nan,
                "sd": math.nan, "lo": math.nan, "hi": math.nan}
    mean = statistics.fmean(vals)
    sd = statistics.stdev(vals)
    df = len(vals) - 1
    tcrit = _T975.get(df, 1.96)
    half = tcrit * sd / math.sqrt(len(vals))
    return {"n": float(len(vals)), "mean": mean, "sd": sd,
            "lo": mean - half, "hi": mean + half, "t_crit": tcrit}


# ---------------------------------------------------------------------------
# PC-B -- runtime call-signature recording.
#
# The C2-R failure (methodology gate #9, ninth recurrence) was a positive
# control that reproduced the target value while executing a DIFFERENT code
# path than the headline (``legacy``+``keep_slack=False`` vs ``c2r``+
# ``keep_slack=True``).  The fix is not "declare that they match" -- a declared
# match is itself an identity.  Signatures here are captured AT CALL TIME from
# the actual invocation, and the branch flags are set by the executing body
# from data-dependent decisions.  ``PC-B-neg`` then proves the comparison has
# teeth by deliberately varying one kwarg and asserting the check FIRES.
# ---------------------------------------------------------------------------

_SIGNATURES: Dict[str, List[Dict[str, object]]] = {}
_SIG_CONTEXT: List[str] = []


def _record_signature(func_qualname: str, kwargs: Mapping[str, object],
                      branch: Mapping[str, object]) -> None:
    tag = _SIG_CONTEXT[-1] if _SIG_CONTEXT else "headline"
    _SIGNATURES.setdefault(tag, []).append({
        "function": func_qualname,
        "kwargs": dict(sorted(kwargs.items())),
        "branch": dict(sorted(branch.items())),
    })


class sig_context:
    """Tag every estimand call made inside the ``with`` block."""

    def __init__(self, tag: str) -> None:
        self.tag = tag

    def __enter__(self) -> "sig_context":
        _SIG_CONTEXT.append(self.tag)
        return self

    def __exit__(self, *exc) -> None:
        _SIG_CONTEXT.pop()


def signature_set(tag: str) -> List[Dict[str, object]]:
    """Distinct signatures recorded under ``tag`` (order-independent)."""
    seen: List[Dict[str, object]] = []
    for record in _SIGNATURES.get(tag, []):
        if record not in seen:
            seen.append(record)
    return sorted(seen, key=lambda r: json.dumps(r, sort_keys=True))


# ---------------------------------------------------------------------------
# sec4 -- boot-level estimands.  ONE code path; every consumer (headline and
# every positive control) goes through this function.
# ---------------------------------------------------------------------------

@dataclass
class BootRecord:
    arm: str
    decode_sm: int
    block: str
    boot: int
    phase: str
    path: str
    est: Dict[str, float]
    t_boot0: Optional[float] = None
    exact: Optional[bool] = None
    status: Optional[str] = None
    residency_fraction: Optional[Dict[str, float]] = None
    split_transitions: Optional[int] = None


def boot_estimands(
    path: Path,
    *,
    ttft_slo_ms: float = TTFT_SLO_MS,
    itl_slo_ms: float = ITL_SLO_MS,
    itl_quantile: float = 0.95,
    empty_itl_policy: str = "inf",
    request_slice: Tuple[float, float] = (0.0, 1.0),
) -> Dict[str, float]:
    """All per-boot estimands of sec4, from canonical library calls.

    ``empty_itl_policy='inf'`` maps a request with no recorded inter-token
    latency to +inf.  Rationale (auditor README, item 2): the canonical
    ``RequestResult.passes`` scores such a request as a FAILURE because
    ``percentile(()) = nan`` and ``nan <= x`` is False; +inf is the
    location-statistic image of that same convention.  Dropping them (as
    fragment B does) is the only convention that is NOT canon-consistent.
    """
    requests, duration = pdmux.load_bench_serving_rounds(
        Path(path), request_slice=request_slice)
    if not requests:
        raise ValueError(f"no requests in {path}")
    if duration <= 0:
        raise ValueError(f"non-positive summed duration in {path}")

    ttfts = [r.ttft_ms for r in requests]
    per_request_itl: List[float] = []
    empty_itl = 0
    for request in requests:
        if request.token_itl_ms:
            per_request_itl.append(
                pdmux.percentile(request.token_itl_ms, itl_quantile))
        else:
            empty_itl += 1
            per_request_itl.append(
                math.inf if empty_itl_policy == "inf" else math.nan)

    summary = pdmux.summarize_requests(
        requests, 0.0, duration, ttft_slo_ms, itl_slo_ms)

    ttft_pass = 100.0 * sum(r.ttft_ms <= ttft_slo_ms for r in requests) / len(requests)
    itl_pass = 100.0 * sum(v <= itl_slo_ms for v in per_request_itl) / len(requests)
    joint_pass = 100.0 * sum(
        r.passes(ttft_slo_ms, itl_slo_ms) for r in requests) / len(requests)

    # K2 / gate #6 -- +-10% band mass around each threshold (the cliff test).
    lo_t, hi_t = ttft_slo_ms * (1 - K2_BAND_FRACTION), ttft_slo_ms * (1 + K2_BAND_FRACTION)
    lo_i, hi_i = itl_slo_ms * (1 - K2_BAND_FRACTION), itl_slo_ms * (1 + K2_BAND_FRACTION)
    band_ttft = sum(lo_t <= v <= hi_t for v in ttfts) / len(ttfts)
    band_itl = sum(lo_i <= v <= hi_i for v in per_request_itl) / len(per_request_itl)

    _record_signature(
        "g16_analyze.boot_estimands",
        {"ttft_slo_ms": ttft_slo_ms, "itl_slo_ms": itl_slo_ms,
         "itl_quantile": itl_quantile, "empty_itl_policy": empty_itl_policy,
         "request_slice": list(request_slice)},
        {"loader": "pdmux_eval.analyze.load_bench_serving_rounds",
         "duration": "summed_over_rounds",           # gate #7
         "p95": "pdmux_eval.analyze.percentile(linear_interp)",
         "median": "statistics.median",
         "predicate": "pdmux_eval.analyze.RequestResult.passes",
         "goodput": "pdmux_eval.analyze.summarize_requests",
         "empty_itl_branch": ("taken" if empty_itl else "not_taken"),
         "multi_round": bool(duration and len(requests) > 0 and _n_rounds(path) > 1)},
    )

    return {
        "M_ttft": statistics.median(ttfts),
        "M_itl": statistics.median(per_request_itl),
        "requests": float(len(requests)),
        "duration_s": duration,
        "throughput_req_s": summary["throughput_req_s"],
        "goodput_req_s": summary["slo_goodput_req_s"],
        "ttft_pass_pct": ttft_pass,
        "itl_p95_pass_pct": itl_pass,
        "joint_pass_pct": joint_pass,
        "band_mass_ttft": band_ttft,
        "band_mass_itl": band_itl,
        "empty_itl_requests": float(empty_itl),
        "ttft_p50_ms": summary["ttft_p50_ms"],
        "ttft_p95_ms": summary["ttft_p95_ms"],
        "ttft_p99_ms": summary["ttft_p99_ms"],
        "token_itl_p50_ms": summary["token_itl_p50_ms"],
        "token_itl_p95_ms": summary["token_itl_p95_ms"],
        "token_itl_p99_ms": summary["token_itl_p99_ms"],
    }


def _headline_estimands(path: Path) -> Dict[str, float]:
    """The headline call form: positional path, every kwarg defaulted.

    Tagged so PC-B compares controls against the ACTUAL campaign invocation
    rather than a declared stand-in.
    """
    with sig_context("headline"):
        return boot_estimands(path)


def _n_rounds(path: Path) -> int:
    with Path(path).open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


# ---------------------------------------------------------------------------
# sec6 -- donor identification and the nine verdicts.
# ---------------------------------------------------------------------------

def _arm_means(grid: Sequence[BootRecord], key: str) -> Dict[str, float]:
    by_arm: Dict[str, List[float]] = {}
    for record in grid:
        by_arm.setdefault(record.arm, []).append(record.est[key])
    return {arm: statistics.fmean(v) for arm, v in sorted(by_arm.items())}


def _argmin_arm(values: Mapping[str, float]) -> Tuple[Optional[str], bool]:
    """argmin with a pre-registered tie-break: SMALLER decode SM wins.

    Returns ``(arm, tied)``; ``tied`` records that the tie-break was used, so
    a tie is never silently reported as an identification.
    """
    finite = {a: v for a, v in values.items() if math.isfinite(v)}
    if not finite:
        return None, False
    best = min(finite.values())
    winners = sorted((a for a, v in finite.items() if v == best), key=arm_sm)
    return winners[0], len(winners) > 1


def identify_donor(grid: Sequence[BootRecord], key: str) -> Dict[str, object]:
    """K1: argmin in >= 3 of 4 blocks AND block-bootstrap frequency >= 80%.

    sec6/C7: the two conditions are highly correlated -- they are NOT two
    independent pieces of evidence, and are never reported as such.
    """
    blocks = sorted({r.block for r in grid})
    per_block: Dict[str, Optional[str]] = {}
    per_block_tied: Dict[str, bool] = {}
    for block in blocks:
        subset = [r for r in grid if r.block == block]
        winner, tied = _argmin_arm(_arm_means(subset, key))
        per_block[block] = winner
        per_block_tied[block] = tied
    # ---- prereg addendum C-5(1) / C-9(2), fixed 2026-08-17 ----------------
    # A win produced by the tie-break is NOT evidence of a donor.  Both call
    # sites used to discard `_argmin_arm`'s `tied`, so a PERFECTLY SATURATED
    # grid -- every arm equal, the strongest possible case for ITL_SATURATED --
    # handed every block to the smallest-SM arm, scored rank 4/4 and bootstrap
    # 1.000, and reported `identified` with `delta_citable`.  The most
    # saturated data could not reach the saturation verdict.
    # Ties now contribute NOTHING instead of contributing to d16; the
    # deterministic tie-break is kept for REPORTING the per-block winner.
    # This can only ever lower `identified`, never create one.
    counts: Dict[str, int] = {}
    for block, winner in per_block.items():
        if winner and not per_block_tied[block]:
            counts[winner] = counts.get(winner, 0) + 1
    rank_arm = max(counts, key=lambda a: (counts[a], -arm_sm(a))) if counts else None
    rank_ok = bool(rank_arm) and counts.get(rank_arm, 0) >= K1_BLOCK_ARGMIN_MIN

    # block-resampling bootstrap (resample BLOCKS, not requests)
    by_block_arm: Dict[str, Dict[str, float]] = {}
    for block in blocks:
        by_block_arm[block] = _arm_means(
            [r for r in grid if r.block == block], key)
    rng = random.Random(K3_BOOTSTRAP_SEED)
    freq: Dict[str, int] = {}
    n_boot_tied = 0
    for _ in range(K3_BOOTSTRAP_SAMPLES):
        picked = [by_block_arm[rng.choice(blocks)] for _ in blocks]
        arms = sorted({a for sample in picked for a in sample})
        means = {}
        for arm in arms:
            vals = [s[arm] for s in picked if arm in s]
            means[arm] = statistics.fmean(vals) if vals else math.inf
        winner, tied = _argmin_arm(means)
        if tied:                      # addendum C-5(1): ties are not evidence
            n_boot_tied += 1
        elif winner:
            freq[winner] = freq.get(winner, 0) + 1
    boot_arm = max(freq, key=lambda a: (freq[a], -arm_sm(a))) if freq else None
    boot_frac = freq.get(boot_arm, 0) / K3_BOOTSTRAP_SAMPLES if boot_arm else 0.0
    boot_ok = bool(boot_arm) and boot_frac >= K1_BOOTSTRAP_MIN_FRAC

    identified = bool(rank_ok and boot_ok and rank_arm == boot_arm)
    return {
        "identified": identified,
        "arm": rank_arm if identified else None,
        "decode_sm": arm_sm(rank_arm) if identified and rank_arm else None,
        "rank_rule": {"winner": rank_arm, "blocks_won": counts.get(rank_arm, 0)
                      if rank_arm else 0, "n_blocks": len(blocks),
                      "passes": rank_ok, "per_block": per_block,
                      # addendum C-5(1): tied blocks are reported (the winner
                      # shown is the tie-broken one) but counted as zero
                      # evidence, so blocks_won can be < the number of blocks
                      # this arm "won".
                      "per_block_tied": per_block_tied,
                      "n_blocks_tied": sum(1 for t in per_block_tied.values() if t)},
        "bootstrap_rule": {"winner": boot_arm, "fraction": boot_frac,
                           "passes": boot_ok,
                           "tied_fraction": n_boot_tied / K3_BOOTSTRAP_SAMPLES,
                           "distribution": {a: freq[a] / K3_BOOTSTRAP_SAMPLES
                                            for a in sorted(freq, key=arm_sm)}},
        "rules_agree": rank_arm == boot_arm,
        "note": "rank and bootstrap rules are highly correlated (sec6/C7) -- "
                "NOT two independent pieces of evidence",
    }


def s_itl(arm_means_itl: Mapping[str, float], slo: float) -> Optional[int]:
    """``S_itl(slo) = min{decode SM : mean_b M_itl(a,b) <= slo}``."""
    for arm in sorted(arm_means_itl, key=arm_sm):
        if arm_means_itl[arm] <= slo:
            return arm_sm(arm)
    return None


def s_itl_upward_closed(arm_means_itl: Mapping[str, float], slo: float) -> Optional[int]:
    """``min{s : EVERY arm with decode SM >= s meets the SLO}``.

    S6 (audit): ``S_itl`` reads as a "sufficient point" only if ``M_itl`` is
    monotone in decode SM.  g2_0_hard phase A is NOT monotone
    (22.74/18.25/18.41/19.15/18.95), so plain ``S_itl`` can return a point above
    which some arm still violates the SLO.  Both are reported.
    """
    arms = sorted(arm_means_itl, key=arm_sm)
    for index, arm in enumerate(arms):
        if all(arm_means_itl[a] <= slo for a in arms[index:]):
            return arm_sm(arm)
    return None


def s_itl_exact_transitions(arm_means_itl: Mapping[str, float]) -> List[Dict[str, object]]:
    """EXACT SLO breakpoints of ``S_itl`` -- the left-to-right running minima.

    S1 (audit): the K8 ladder steps 1 ms, but the ``S_itl`` bands can be far
    narrower (0.203 ms on the old grid), so the ladder alone SKIPS whole bands
    -- including ``S_itl = 44``, which sits in exactly the tight-SLO region
    sec4-1 identifies as this campaign's only SLO-related payoff.  The exact
    breakpoints cost nothing and are reported alongside the ladder.
    """
    arms = sorted(arm_means_itl, key=arm_sm)
    records: List[Dict[str, object]] = []
    running = math.inf
    for arm in arms:
        value = arm_means_itl[arm]
        if value < running:
            running = value
            records.append({"threshold_ms": value, "S_itl": arm_sm(arm), "arm": arm})
    records.sort(key=lambda r: r["threshold_ms"])
    bands: List[Dict[str, object]] = []
    if records:
        bands.append({"slo_lo_ms": None, "slo_hi_ms": records[0]["threshold_ms"],
                      "S_itl": None, "verdict": "S_ITL_UNREACHED"})
    for index, record in enumerate(records):
        upper = (records[index + 1]["threshold_ms"]
                 if index + 1 < len(records) else None)
        bands.append({"slo_lo_ms": record["threshold_ms"], "slo_hi_ms": upper,
                      "S_itl": record["S_itl"], "arm": record["arm"],
                      "band_width_ms": (upper - record["threshold_ms"])
                      if upper is not None else None})
    return bands


def _forced_cell(d_ttft: Optional[int], d_itl: Optional[int],
                 s_min: int, s_max: int) -> Dict[str, object]:
    """sec3 forced-cell table: which entries are algebraically forced."""
    if d_ttft is None or d_itl is None:
        return {"cell": "undetermined", "sign_forced": None}
    row = "S_max" if d_itl == s_max else ("S_min" if d_itl == s_min else "interior")
    col = "S_max" if d_ttft == s_max else ("S_min" if d_ttft == s_min else "interior")
    table = {
        ("S_max", "S_max"): ("delta == 0", True),
        ("S_max", "interior"): ("delta > 0 measured; SIZE IS A LOWER BOUND", False),
        ("S_max", "S_min"): ("delta > 0 FORCED (sign inflated)", True),
        ("interior", "S_max"): ("delta < 0 FORCED", True),
        ("interior", "interior"): ("sign and size both data", False),
        ("interior", "S_min"): ("delta >= 0 FORCED (sign inflated)", True),
        ("S_min", "S_max"): ("delta < 0 FORCED", True),
        ("S_min", "interior"): ("delta < 0 measured", False),
        ("S_min", "S_min"): ("delta == 0 FORCED", True),
    }
    text, forced = table[(row, col)]
    return {"cell": f"D_itl={row} x D_ttft={col}", "reading": text,
            "sign_forced": forced}


def _saturation_gap(grid: Sequence[BootRecord], contenders: Sequence[str]) -> Dict[str, object]:
    """Max spread of ``mean_b M_itl`` over the CONTENDING arms.

    sec6 says "upper arms" without defining the set.  Pre-registered here as
    the contending set (arms that win at least one block), with the top-half-by-
    SM alternative reported ALONGSIDE so the choice is auditable rather than
    silent.  Flagged in ``spec_ambiguities``.
    """
    means = _arm_means(grid, "M_itl")
    def spread(arms: Sequence[str]) -> float:
        vals = [means[a] for a in arms if a in means and math.isfinite(means[a])]
        return (max(vals) - min(vals)) if len(vals) >= 2 else math.nan
    finite = [a for a in sorted(means, key=arm_sm) if math.isfinite(means[a])]
    all_arms = sorted(means, key=arm_sm)
    top_half = all_arms[len(all_arms) // 2:]
    # ---- K11 (prereg addendum C, 2026-08-17) ------------------------------
    # sec6's "upper arms" is PINNED BY SM VALUE, not by a floating top-half and
    # not by the contending set.  Fallback chain, pre-registered: SM >= 44 ->
    # top-half of the surviving arms -> ITL_SATURATED is UNDEFINED.
    # `contenders` is REFUTED as the primary (it is derived from the very block
    # argmin whose failure ITL_SATURATED is supposed to explain, and it can be
    # nan by construction) and survives only as a DIAGNOSTIC field.
    upper = [a for a in finite if arm_sm(a) >= K11_UPPER_ARM_MIN_SM]
    if len(upper) >= 2:
        basis = "pinned_sm_ge_%d" % K11_UPPER_ARM_MIN_SM
    else:
        upper = finite[len(finite) // 2:]
        basis = "fallback_top_half_of_surviving"
        if len(upper) < 2:
            basis = "undefined_fewer_than_2_upper_arms"
    return {
        "primary_definition": ("upper arms U = decode SM >= %d (K11, prereg "
                               "addendum C)" % K11_UPPER_ARM_MIN_SM),
        "upper_arms": list(upper),
        "upper_arm_basis": basis,
        "gap_upper_ms": spread(upper),          # <- the one the verdict reads
        "contenders": list(contenders),
        "gap_contenders_ms": spread(contenders),   # diagnostic only
        "gap_top_half_by_sm_ms": spread(top_half),  # diagnostic only
        "top_half_arms": top_half,
        "delta_ms": K7_DELTA_MS,
    }


def _grid_hygiene(grid: Sequence[BootRecord]) -> Dict[str, object]:
    """F3 (audit): arm x block incidence, imbalance, and block sufficiency.

    ``_arm_means`` averages whatever is present and the bootstrap fills a
    missing arm with ``inf``, so an arm that lost boots is STRUCTURALLY
    penalised by the rank rule while nothing in the output says so.  H5 plans
    for failed boots, so this is not a hypothetical.
    """
    arms = sorted({r.arm for r in grid}, key=arm_sm)
    blocks = sorted({r.block for r in grid})
    incidence = {arm: {block: sum(1 for r in grid if r.arm == arm and r.block == block)
                       for block in blocks} for arm in arms}
    counts = {arm: sum(incidence[arm].values()) for arm in arms}
    balanced = len({tuple(sorted(incidence[a].items())) for a in arms}) == 1
    missing = [(a, b) for a in arms for b in blocks if incidence[a][b] == 0]
    flags: List[str] = []
    if not balanced:
        flags.append("UNBALANCED_GRID: arms do not appear in the same blocks -- "
                     "the rank rule structurally penalises the arm with fewer "
                     "blocks, and the bootstrap fills its gaps with inf")
    if len(blocks) < K5_BLOCKS:
        flags.append(f"INSUFFICIENT_BLOCKS: {len(blocks)} < K5 = {K5_BLOCKS}.  "
                     f"A non-identification here is SAMPLE SHORTAGE, not the "
                     f"pre-declared 'informative negative' result -- do not "
                     f"label it UNIDENTIFIED/ITL_SATURATED as an outcome")
    return {"arms": arms, "blocks": blocks, "incidence": incidence,
            "boots_per_arm": counts, "balanced": balanced,
            "missing_cells": missing, "flags": flags,
            "sufficient_blocks": len(blocks) >= K5_BLOCKS}


def decide(grid: Sequence[BootRecord], *, label: str) -> Dict[str, object]:
    """Full sec6 decision for one phase of one grid."""
    arms = sorted({r.arm for r in grid}, key=arm_sm)
    if len(arms) < 2:
        raise ValueError(f"{label}: need >= 2 arms, got {arms}")
    s_min, s_max = arm_sm(arms[0]), arm_sm(arms[-1])
    means_ttft = _arm_means(grid, "M_ttft")
    means_itl = _arm_means(grid, "M_itl")

    donor_ttft = identify_donor(grid, "M_ttft")
    donor_itl = identify_donor(grid, "M_itl")

    contenders = sorted(
        {w for w in donor_itl["rank_rule"]["per_block"].values() if w}, key=arm_sm)
    saturation = _saturation_gap(grid, contenders)

    # ---- F1 (audit) --------------------------------------------------------
    # sec4 DEFINES D_ttft / D_itl as the bare argmin of the arm means
    # ("D_ttft = decode SM of argmin_a mean_b M_ttft").  sec6 does NOT redefine
    # the symbols -- it makes identification a CONDITION OF THE VERDICT
    # ("donor identified AND the decision quantity > 0").  An earlier version of
    # this analyzer set the symbols to None whenever K1 failed, which (a) broke
    # its own PC-C control and (b) silently emptied the sec3 forced-cell table
    # in exactly the most likely outcome (non-identification).  The symbols are
    # therefore the bare argmin, ALWAYS defined, and identification rides
    # alongside as a citation gate.
    d_ttft_arm, ttft_tied = _argmin_arm(means_ttft)
    d_itl_arm, itl_tied = _argmin_arm(means_itl)
    d_ttft = arm_sm(d_ttft_arm) if d_ttft_arm else None
    d_itl = arm_sm(d_itl_arm) if d_itl_arm else None
    delta = (d_itl - d_ttft) if (d_ttft is not None and d_itl is not None) else None
    delta_citable = bool(donor_ttft["identified"] and donor_itl["identified"])

    # ---- verdict for the Delta channel (primary-B) -------------------------
    sign_verdict: Optional[str] = None
    if delta is not None:
        if delta > 0:
            sign_verdict = "TAX_POSITIVE"
        elif delta == 0:
            sign_verdict = "NO_TAX"
        else:
            sign_verdict = "NEGATIVE_INTERIOR"

    # ALL applicable labels, not just the dominant one.  sec6 lists the nine
    # verdicts without a precedence rule, and they are not mutually exclusive:
    # "D_ttft sits at S_max" (truncation) and "D_itl is not identified"
    # (saturation) can both hold.  Reporting only the dominant label would
    # silently drop a pre-declared result, so both channels are emitted.
    flags: List[Dict[str, str]] = []
    if d_ttft == s_max:
        flags.append({"verdict": "TRUNCATED",
                      "why": f"D_ttft = S_max = {s_max}: decision quantity 2 "
                             f"stays undetermined; NEVER read as 'no tax'"})
    if d_ttft == s_min:
        flags.append({"verdict": "TRUNCATED_LOW",
                      "why": f"D_ttft = S_min = {s_min}: delta >= 0 is forced, "
                             f"so delta > 0 is NOT read as a tax"})
    gap = saturation["gap_upper_ms"]        # K11 (prereg addendum C)
    if not donor_itl["identified"]:
        if math.isfinite(gap) and gap < K7_DELTA_MS:
            flags.append({"verdict": "ITL_SATURATED",
                          "why": f"ITL donor unidentified and UPPER-arm spread "
                                 f"{gap:.3f} ms < delta {K7_DELTA_MS} ms over "
                                 f"{saturation['upper_arms']} "
                                 f"({saturation['upper_arm_basis']}) -- "
                                 f"non-identification is SATURATION, not noise "
                                 f"(informative negative)"})
        else:
            flags.append({"verdict": "UNIDENTIFIED", "why": "ITL donor fails K1"})
    if not donor_ttft["identified"]:
        flags.append({"verdict": "UNIDENTIFIED", "why": "TTFT donor fails K1"})

    if not donor_ttft["identified"] or not donor_itl["identified"]:
        if (not donor_itl["identified"] and math.isfinite(gap)
                and gap < K7_DELTA_MS):
            verdict = "ITL_SATURATED"
        else:
            verdict = "UNIDENTIFIED"
    elif d_ttft == s_max:
        verdict = "TRUNCATED"
    elif d_ttft == s_min:
        verdict = "TRUNCATED_LOW"
    else:
        verdict = sign_verdict or "UNIDENTIFIED"

    forced = _forced_cell(d_ttft, d_itl, s_min, s_max)

    # ---- primary-C: the Delta_SLO ladder (C2') -----------------------------
    ladder: List[Dict[str, object]] = []
    for slo in K8_SLO_LADDER:
        point = s_itl(means_itl, float(slo))
        if point is None:
            rung_verdict, delta_slo = "S_ITL_UNREACHED", None
        else:
            delta_slo = (point - d_ttft) if d_ttft is not None else None
            if point == s_min:
                rung_verdict = "ITL_UNCONSTRAINED"
            elif d_ttft is None:
                rung_verdict = "UNIDENTIFIED"
            elif d_ttft == s_max:
                rung_verdict = "TRUNCATED"
            elif d_ttft == s_min:
                rung_verdict = "TRUNCATED_LOW"
            elif delta_slo > 0:
                rung_verdict = "TAX_POSITIVE"
            elif delta_slo == 0:
                rung_verdict = "NO_TAX"
            else:
                rung_verdict = "NEGATIVE_INTERIOR"
        closed = s_itl_upward_closed(means_itl, float(slo))
        ladder.append({"slo_ms": slo, "S_itl": point, "delta_slo": delta_slo,
                       "verdict": rung_verdict,
                       "S_itl_upward_closed": closed,
                       "monotonicity_violated": closed != point})

    # sign-change intervals -- the REGISTRABLE object (never the 60 ms point)
    intervals: List[Dict[str, object]] = []
    for rung in ladder:
        key = (rung["verdict"], rung["delta_slo"])
        if intervals and intervals[-1]["_key"] == key:
            intervals[-1]["slo_hi_ms"] = rung["slo_ms"]
        else:
            intervals.append({"_key": key, "slo_lo_ms": rung["slo_ms"],
                              "slo_hi_ms": rung["slo_ms"],
                              "verdict": rung["verdict"],
                              "delta_slo": rung["delta_slo"],
                              "S_itl": rung["S_itl"]})
    for interval in intervals:
        interval.pop("_key")

    op = next(r for r in ladder if r["slo_ms"] == int(OPERATING_POINT_SLO_MS))

    # sec6 pre-registered by-product: block-paired difference against S_max
    byproduct = None
    if "d44" in means_itl and arms[-1] in means_itl:
        byproduct = {
            "definition": "M_itl(d44) - M_itl(S_max)  [sec6 pre-registered]",
            "value_ms": means_itl["d44"] - means_itl[arms[-1]],
            "s_max_arm": arms[-1],
        }

    # sec6 adaptive rule -- pre-declared, never chosen after seeing the data
    if verdict in ("UNIDENTIFIED", "ITL_SATURATED"):
        gap = saturation["gap_upper_ms"]     # K11 (prereg addendum C)
        blocks_disagree = len({w for w in donor_itl["rank_rule"]["per_block"].values()
                               if w}) > 1
        if math.isfinite(gap) and gap < K7_DELTA_MS:
            adaptive = {"action": "NO_MORE_BLOCKS",
                        "reason": "upper gap < delta => ITL_SATURATED is final"}
        elif blocks_disagree:
            adaptive = {"action": f"ADD_{K9_ADAPTIVE_EXTRA_BLOCKS}_BLOCKS",
                        "reason": "gap >= delta and blocks disagree"}
        else:
            adaptive = {"action": "NO_MORE_BLOCKS",
                        "reason": "not a pre-declared trigger"}
    else:
        adaptive = {"action": "NONE", "reason": "donor identified"}

    return {
        "label": label,
        "scope": NOMINAL_SPLIT_SCOPE,
        "frame": HEADLINE_FRAME,
        "arms": arms,
        "S_min": s_min,
        "S_max": s_max,
        "arm_means": {"M_ttft": means_ttft, "M_itl": means_itl},
        "arm_t_ci": {
            "M_ttft": {a: t_ci([r.est["M_ttft"] for r in grid if r.arm == a])
                       for a in arms},
            "M_itl": {a: t_ci([r.est["M_itl"] for r in grid if r.arm == a])
                      for a in arms},
        },
        "D_ttft": donor_ttft,
        "D_itl": donor_itl,
        "D_ttft_arm": d_ttft_arm,
        "D_itl_arm": d_itl_arm,
        "delta": delta,
        "delta_citable": delta_citable,
        "ties": {"M_ttft_tied": ttft_tied, "M_itl_tied": itl_tied,
                 "warning": "a tie was broken by the pre-registered rule "
                            "(smaller decode SM wins).  ORACLE R1 already "
                            "recorded a tie-break-dependent headline ('116'), "
                            "so a tied donor is never cited as identified."},
        "verdict": verdict,
        "verdict_flags": flags,
        "symbol_convention": (
            "D_ttft / D_itl / delta / forced_cell are sec4's BARE ARGMIN and "
            "are always defined.  sec6 identification (K1) is a CITATION GATE "
            "carried in 'delta_citable' and in the verdict -- not part of the "
            "symbol definition."),
        "sign_verdict": sign_verdict,
        "forced_cell": forced,
        "saturation": saturation,
        "delta_slo_ladder": ladder,
        "delta_slo_intervals": intervals,
        "s_itl_exact_bands": s_itl_exact_transitions(means_itl),
        "s_itl_ladder_resolution_warning": (
            "the K8 ladder steps 1 ms; 's_itl_exact_bands' below shows the true "
            "breakpoints.  Any band narrower than 1 ms is INVISIBLE to the "
            "ladder -- on the old grid the S_itl=44 band (0.203 ms) vanishes "
            "entirely, and that band sits in the tight-SLO region sec4-1 calls "
            "this campaign's only SLO-related payoff.  Cite the exact bands."),
        "grid_hygiene": _grid_hygiene(grid),
        "operating_point": {
            "slo_ms": OPERATING_POINT_SLO_MS, **op,
            "MANDATORY_BAND": DELTA_SLO_60_BAND,
            "warning": "the 60 ms point estimate is NEVER cited without this "
                       "band (C2'); the registrable object is the sign-change "
                       "interval, not this point",
        },
        "byproduct_block_paired": byproduct,
        "adaptive_rule": adaptive,
        "citation_guards": [
            "Delta must never be a standalone headline (C2: the location "
            "statistic does not saturate, so D_itl = S_max may be a priori)",
            "TRUNCATED is never written as 'no tax'",
            "percentile CI is never cited on its own; t(n-1) accompanies it",
            "Delta / Delta_SLO are discrete lattice values: report the donor / "
            "sufficient-point SELECTION DISTRIBUTION, not a CI",
            # --- added after the analyzer audit (7 missing guards) ----------
            "sec10-7: this says NOTHING about HE0.  A positive coupling tax "
            "does NOT revive dynamic control -- 'achieved dynamic is worse "
            "than best static' and 'the achievable ceiling' are separate "
            "propositions (gate #38).  No policy-ranking change.",
            "sec10-9(a): the location statistic CANNOT in principle produce a "
            "magnitude comparable to sec1-20 '+16%' or sec1-32 '+1.67%'",
            "sec6: TRUNCATED / ITL_SATURATED / S_ITL_UNREACHED / UNIDENTIFIED "
            "are RESULTS, not failures.  This tool's exit code 2 and the word "
            "'GATE' refer to POSITIVE-CONTROL failure only -- never relabel a "
            "verdict as a failed measurement (gate #21, reverse direction)",
            "sec10-2: HI is 2.15-2.35x overloaded and threshold-goodput is "
            "ill-posed there; the goodput / pass-rate side outputs inherit "
            "that caveat",
            "sec10-5/6: Zamba2-2.7B, ctx4096, ShareGPT, and only TWO points on "
            "the work-ratio axis -- no slope or functional-form claims, no "
            "transplant to 8B or other models",
            "NEVER write 'gate #16 is closed'.  What closes is the "
            "RE-FORMULATED decision quantity 2; the threshold version stays on "
            "the rate axis (rev3 sec1 C1)",
            "add. A-3: agreement between the primary and the drift-corrected "
            "donor does NOT mean 'no drift'.  A symmetric CURVATURE component "
            "moves both estimators the same way, so DRIFT_DISAGREEMENT is "
            "blind exactly where A-3-2 says 4 blocks cannot balance",
        ],
    }


# ---------------------------------------------------------------------------
# add. A-3 -- drift-corrected variant (computed alongside, never instead).
# ---------------------------------------------------------------------------

def drift_corrected_means(grid: Sequence[BootRecord], key: str) -> Optional[Dict[str, object]]:
    """Within-estimator: ``y = mu_a + s * (t - t_bar)`` with arm fixed effects.

    The forward/reverse block design balances arm POSITION exactly (mean 3.0)
    but NOT wall-clock time: arm ``a``'s mean start time is ``(sum T - T_a)/2``,
    which depends on ``T_a``, and ``T_a`` is monotone in decode SM (d74 is
    1.92x d34).  The residual is therefore correlated with the decision axis
    and of the same order (0.1-0.5%) as the effect being targeted (0.34%).
    """
    obs = [(r.arm, r.t_boot0, r.est[key]) for r in grid
           if r.t_boot0 is not None and math.isfinite(r.est[key])]
    if len(obs) < 4:
        return None
    arms = sorted({a for a, _, _ in obs}, key=arm_sm)
    if len({t for _, t, _ in obs}) < 2:
        return None
    num = den = 0.0
    for arm in arms:
        rows = [(t, y) for a, t, y in obs if a == arm]
        if len(rows) < 2:
            continue
        tbar = statistics.fmean(t for t, _ in rows)
        ybar = statistics.fmean(y for _, y in rows)
        for t, y in rows:
            num += (t - tbar) * (y - ybar)
            den += (t - tbar) ** 2
    if den <= 0:
        return None
    slope = num / den
    t_global = statistics.fmean(t for _, t, _ in obs)
    corrected = {}
    for arm in arms:
        rows = [(t, y) for a, t, y in obs if a == arm]
        tbar = statistics.fmean(t for t, _ in rows)
        ybar = statistics.fmean(y for _, y in rows)
        corrected[arm] = ybar - slope * (tbar - t_global)
    winner, tied = _argmin_arm(corrected)
    return {"slope_per_s": slope, "corrected_arm_means": corrected,
            "argmin_arm": winner, "argmin_tied": tied,
            "note": "reported ALONGSIDE the primary; a donor disagreement "
                    "between the two is itself a registered result (add. A-3)"}


# ---------------------------------------------------------------------------
# Grid loading.
# ---------------------------------------------------------------------------

def load_campaign_grid(directory: Path, phase: str, *,
                       include_smoke: bool = False) -> Tuple[List[BootRecord], Dict[str, object]]:
    """Campaign artifacts, with the add. B-5(a)2 boot-adoption rule.

    Adoption: ``status == 'completed' AND telem_rc == 0 AND fields_rc.lo == 0
    AND fields_rc.hi == 0``.  Registered because ``write_sidecar`` originally
    ran BEFORE the ``ARTIFACT_OK`` judgment (N1); the harness has since been
    fixed, and this rule is the belt-and-braces half of that fix.

    Glob is fixed to ``*_LO.jsonl`` / ``*_HI.jsonl`` (N2) so warm-up output --
    which shares the schema and namespace -- cannot enter the grid.
    """
    suffix = "_LO.jsonl" if phase == "LO" else "_HI.jsonl"
    records: List[BootRecord] = []
    rejected: List[Dict[str, object]] = []
    arms_failing_exact: set = set()

    def arm_of(run_id: str) -> str:
        found = re.search(r"_(d\d+)_boot\d+_", run_id + "_")
        return found.group(1) if found else ""
    for path in sorted(directory.glob(f"g16_*{suffix}")):
        run_id = path.name[: -len(suffix)]
        match = re.match(r"^g16_(?P<block>[^_]+)_(?P<arm>d\d+)_boot(?P<boot>\d+)_",
                         run_id + "_")
        if not match:
            rejected.append({"path": path.name, "why": "filename convention"})
            continue
        sidecar_path = directory / f"{run_id}_sidecar.json"
        if not sidecar_path.exists():
            rejected.append({"path": path.name, "why": "no sidecar (H13)"})
            continue
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        mode = sidecar.get("mode", "")
        if mode == "smoke" and not include_smoke:
            rejected.append({"path": path.name, "why": "smoke artifact (H17)"})
            continue
        fields_rc = sidecar.get("fields_rc") or {}
        adopted = (sidecar.get("status") == "completed"
                   and sidecar.get("telem_rc") == 0
                   and fields_rc.get("lo") == 0
                   and fields_rc.get("hi") == 0)
        if not adopted:
            rejected.append({"path": path.name, "why": "adoption rule",
                             "status": sidecar.get("status"),
                             "telem_rc": sidecar.get("telem_rc"),
                             "fields_rc": fields_rc})
            continue
        if sidecar.get("exact") is not True:
            # H3'-a says deactivate the ARM, not the boot.  Rejecting only the
            # boot would leave an UNBALANCED grid, which the rank rule then
            # penalises silently (see _grid_hygiene).
            arms_failing_exact.add(arm_of(run_id))
            rejected.append({"path": path.name, "why": "H3'-a exact is not True",
                             "realized_a": sidecar.get("realized_a"),
                             "realized_b": sidecar.get("realized_b")})
            continue
        summary_path = directory / f"{run_id}_controller_summary.json"
        residency = transitions = None
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                residency = summary.get("residency_fraction")
                transitions = summary.get("split_transitions")
            except (json.JSONDecodeError, OSError):
                pass
        arm = match.group("arm")
        records.append(BootRecord(
            arm=arm, decode_sm=arm_sm(arm), block=match.group("block"),
            boot=int(match.group("boot")), phase=phase, path=str(path),
            est=_headline_estimands(path), t_boot0=sidecar.get("t_boot0"),
            exact=sidecar.get("exact"), status=sidecar.get("status"),
            residency_fraction=residency, split_transitions=transitions))
    if arms_failing_exact:
        records = [r for r in records if r.arm not in arms_failing_exact]
    missing_summary = [Path(r.path).name for r in records
                       if r.residency_fraction in (None, {})]
    return records, {
        "n_adopted": len(records),
        "rejected": rejected,
        "arms_deactivated_by_exact": sorted(arms_failing_exact),
        "boots_without_controller_summary": missing_summary,
        "residency_warning": (
            "add. A-2's conditional-label baseline is MISSING for these boots "
            "-- report the arm labels as unqualified 'nominal split' only with "
            "that gap stated" if missing_summary else None),
    }


def load_legacy_grid(directory: Path, pattern: str, arms: Sequence[str],
                     phase: str) -> List[BootRecord]:
    """Pre-G16 artifacts (no sidecars): each rep is a PSEUDO-block.

    Used only by the positive controls.  A rep is NOT a real block -- it has no
    forward/reverse position balance -- and is labelled ``pseudo`` so that no
    downstream text can quietly call it one.
    """
    records: List[BootRecord] = []
    for arm in arms:
        for index, path in enumerate(sorted(glob.glob(str(directory / pattern.format(arm=arm))))):
            records.append(BootRecord(
                arm=arm, decode_sm=arm_sm(arm), block=f"pseudo{index + 1}",
                boot=index + 1, phase=phase, path=path,
                est=boot_estimands(Path(path))))
    return records


# ---------------------------------------------------------------------------
# sec7 -- positive controls.
# ---------------------------------------------------------------------------

SGPTV_ARMS = ("d16", "d24", "d34", "d44")
G20H_ARMS = ("d34", "d44", "d54", "d64", "d74")

# PC-A: ORACLE_REANALYSIS_2026-08-16.md sec3-2, verbatim (mean only; the SDs
# there are over n=4 boots and are checked as a second tier).
PC_A_TARGETS = {
    "LO": {
        "d16": {"ttft_pass_pct": 100.000, "itl_p95_pass_pct": 99.625, "joint_pass_pct": 99.625, "goodput_req_s": 2.831},
        "d24": {"ttft_pass_pct": 99.500, "itl_p95_pass_pct": 99.625, "joint_pass_pct": 99.125, "goodput_req_s": 2.761},
        "d34": {"ttft_pass_pct": 99.833, "itl_p95_pass_pct": 99.375, "joint_pass_pct": 99.208, "goodput_req_s": 2.835},
        "d44": {"ttft_pass_pct": 99.958, "itl_p95_pass_pct": 99.625, "joint_pass_pct": 99.583, "goodput_req_s": 2.848},
    },
    "HI": {
        "d16": {"ttft_pass_pct": 57.083, "itl_p95_pass_pct": 18.292, "joint_pass_pct": 8.667, "goodput_req_s": 0.443},
        "d24": {"ttft_pass_pct": 66.375, "itl_p95_pass_pct": 46.167, "joint_pass_pct": 23.083, "goodput_req_s": 1.250},
        "d34": {"ttft_pass_pct": 68.917, "itl_p95_pass_pct": 92.125, "joint_pass_pct": 61.958, "goodput_req_s": 3.405},
        "d44": {"ttft_pass_pct": 70.708, "itl_p95_pass_pct": 93.458, "joint_pass_pct": 65.000, "goodput_req_s": 3.613},
    },
}

# PC-E: fragment C / fragment D targets (mean +- SD over n=4 boots).
PC_E_TARGETS = {
    "HI": {"M_itl": {"d16": 83.572, "d24": 60.203, "d34": 58.850, "d44": 58.647},
           "M_ttft": {"d16": 1792.4, "d24": 1409.5, "d34": 1211.5, "d44": 1111.3}},
    "LO": {"M_itl": {"d16": 12.183, "d24": 12.252, "d34": 12.968, "d44": 13.912},
           "M_ttft": {"d16": 73.6, "d24": 74.4, "d34": 79.1, "d44": 79.6}},
}

# PC-D: fragment A / fragment B targets on the 7-arm-capable g2_0_hard grid.
PC_D_TARGETS = {
    "B": {"M_ttft": {"d34": 383, "d44": 553, "d54": 870, "d64": 1863, "d74": 5477},
          "M_itl": {"d34": 48.14, "d44": 42.15, "d54": 40.46, "d64": 36.41, "d74": 33.09}},
    "A": {"M_ttft": {"d34": 779, "d44": 1910, "d54": 1952, "d64": 4914, "d74": 6987},
          "M_itl": {"d34": 22.74, "d44": 18.25, "d54": 18.41, "d64": 19.15, "d74": 18.95}},
}


def _close(got: float, want: float, tol_abs: float, tol_rel: float) -> bool:
    if not (math.isfinite(got) and math.isfinite(want)):
        return False
    return abs(got - want) <= max(tol_abs, tol_rel * abs(want))


def run_controls(slo_dir: Path, g20h_dir: Path) -> Dict[str, object]:
    out: Dict[str, object] = {"scope": NOMINAL_SPLIT_SCOPE}
    failures: List[str] = []

    # ---- PC-A + PC-E: new scorer x old 4-arm grid --------------------------
    sgptv: Dict[str, List[BootRecord]] = {}
    for phase, tag in (("LO", "sgptvLo"), ("HI", "sgptvHi")):
        with sig_context("PC-A/PC-E"):
            sgptv[phase] = load_legacy_grid(
                slo_dir, tag + "_{arm}_rep*_L3H12_*.jsonl", SGPTV_ARMS, phase)

    pc_a: Dict[str, object] = {"checks": [], "passed": True}
    for phase, per_arm in PC_A_TARGETS.items():
        means_by_arm = {
            arm: {k: statistics.fmean([r.est[k] for r in sgptv[phase] if r.arm == arm])
                  for k in ("ttft_pass_pct", "itl_p95_pass_pct", "joint_pass_pct", "goodput_req_s")}
            for arm in SGPTV_ARMS}
        for arm, targets in per_arm.items():
            for key, want in targets.items():
                got = means_by_arm[arm][key]
                ok = _close(got, want, 0.005, 0.002)
                pc_a["checks"].append({"phase": phase, "arm": arm, "metric": key,
                                       "got": got, "want": want, "pass": ok})
                if not ok:
                    pc_a["passed"] = False
    if not pc_a["passed"]:
        failures.append("PC-A")
    out["PC_A"] = pc_a

    pc_e: Dict[str, object] = {"checks": [], "passed": True}
    for phase, per_metric in PC_E_TARGETS.items():
        for metric, per_arm in per_metric.items():
            for arm, want in per_arm.items():
                got = statistics.fmean([r.est[metric] for r in sgptv[phase] if r.arm == arm])
                ok = _close(got, want, 0.05 if metric == "M_itl" else 0.05, 0.002)
                pc_e["checks"].append({"phase": phase, "metric": metric, "arm": arm,
                                       "got": got, "want": want, "pass": ok})
                if not ok:
                    pc_e["passed"] = False
    if not pc_e["passed"]:
        failures.append("PC-E")
    out["PC_E"] = pc_e

    # ---- PC-C: truncation logic, new analyzer x old 4-arm grid -------------
    # PC-C loads its OWN grid under its own tag rather than reusing the PC-A
    # records.  Reuse would leave PC-C with an empty signature set, and an
    # empty set must never be able to pass a same-code-path assert by default.
    with sig_context("PC-C"):
        pc_c_grid = load_legacy_grid(
            slo_dir, "sgptvHi_{arm}_rep*_L3H12_*.jsonl", SGPTV_ARMS, "HI")
        pc_c_decision = decide(pc_c_grid, label="PC-C sgptv HI (old 4-arm)")
    flags = {f["verdict"] for f in pc_c_decision["verdict_flags"]}
    subclaims = [
        {"name": "D_ttft == 44 (== S_max of the old grid)",
         "got": pc_c_decision["D_ttft"]["decode_sm"], "want": 44,
         "pass": pc_c_decision["D_ttft"]["decode_sm"] == 44},
        {"name": "truncation logic fires (TRUNCATED among verdict flags)",
         "got": sorted(flags), "want": "TRUNCATED present",
         "pass": "TRUNCATED" in flags},
        {"name": "D_itl == d44 (sec4 estimator, the form the target is phrased "
                 "in: 'the primary estimator ALSO gives D_ttft=d44 . D_itl=d44')",
         "got": pc_c_decision["D_itl_arm"], "want": "d44",
         "pass": pc_c_decision["D_itl_arm"] == "d44"},
    ]
    pc_c = {
        "decision": pc_c_decision,
        "target_as_written": {"D_ttft": 44, "D_itl": 44, "verdict": "TRUNCATED"},
        "subclaims": subclaims,
        "passed": all(s["pass"] for s in subclaims),
        # NOT a control -- reported so the non-identification is visible.
        "diagnostic_identification_gate": {
            "D_itl_identified": pc_c_decision["D_itl"]["identified"],
            "bootstrap_fraction": pc_c_decision["D_itl"]["bootstrap_rule"]["fraction"],
            "threshold": K1_BOOTSTRAP_MIN_FRAC,
            "reading": (
                "sec4's estimator gives d44, and sec7's PC-C target is scoped "
                "to the estimator ('the primary estimator ALSO gives ...'), so "
                "the control PASSES.  Separately, the K1 citation gate does "
                "NOT identify an ITL donor on this grid.  That non-"
                "identification agrees with the canon's conclusion -- ORACLE "
                "sec3-3 registration-ban #1 and CONSENSUS sec1-13 footnote F "
                "both say the d44/d34 ITL donor is inseparable.  CAUTION: "
                "ORACLE sec3-4's 6028/3972 is the THRESHOLD-PREDICATE donor, a "
                "DIFFERENT estimator from this location-statistic one (which "
                "gives ~0.64/0.36).  Both land near 60/40, but 'the numbers "
                "match' is NOT evidence that the same path was executed -- "
                "that inference is the exact genre of gate #9 / CONSENSUS "
                "sec3 item 39.  The agreeing thing is the CONCLUSION, not the "
                "estimator."),
        },
    }
    if not pc_c["passed"]:
        failures.append("PC-C")
    out["PC_C"] = pc_c

    # ---- PC-D: the 7-arm-capable code path (g2_0_hard) ---------------------
    pc_d: Dict[str, object] = {"phases": {}, "passed": True, "checks": []}
    for phase in ("A", "B"):
        with sig_context("PC-D"):
            grid = load_legacy_grid(
                g20h_dir, "g20h" + phase + "_{arm}_rep*_rA5B4_OB1024_*.jsonl",
                G20H_ARMS, phase)
            decision = decide(grid, label=f"PC-D g2_0_hard phase {phase}")
        for metric, per_arm in PC_D_TARGETS[phase].items():
            for arm, want in per_arm.items():
                got = statistics.fmean([r.est[metric] for r in grid if r.arm == arm])
                tol_abs = 1.0 if metric == "M_ttft" else 0.05
                ok = _close(got, want, tol_abs, 0.01)
                pc_d["checks"].append({"phase": phase, "metric": metric, "arm": arm,
                                       "got": got, "want": want, "pass": ok})
                if not ok:
                    pc_d["passed"] = False
        pc_d["phases"][phase] = decision

    b = pc_d["phases"]["B"]
    a = pc_d["phases"]["A"]
    pc_d["verdict_path_checks"] = [
        {"name": "phase B D_ttft == d34", "got": b["D_ttft"]["arm"], "want": "d34",
         "pass": b["D_ttft"]["arm"] == "d34"},
        {"name": "phase B D_itl == d74 (upper boundary)", "got": b["D_itl"]["arm"],
         "want": "d74", "pass": b["D_itl"]["arm"] == "d74"},
        {"name": "phase B sign channel == TAX_POSITIVE (delta = +40)",
         "got": f'{b["sign_verdict"]} (delta={b["delta"]})', "want": "TAX_POSITIVE (delta=40)",
         "pass": b["sign_verdict"] == "TAX_POSITIVE" and b["delta"] == 40},
        {"name": "phase B Delta_SLO(60): S_itl == 34 == S_min => ITL_UNCONSTRAINED, 0",
         "got": f'S_itl={b["operating_point"]["S_itl"]} delta_slo={b["operating_point"]["delta_slo"]} '
                f'{b["operating_point"]["verdict"]}',
         "want": "S_itl=34 delta_slo=0 ITL_UNCONSTRAINED",
         "pass": (b["operating_point"]["S_itl"] == 34
                  and b["operating_point"]["delta_slo"] == 0
                  and b["operating_point"]["verdict"] == "ITL_UNCONSTRAINED")},
        {"name": "phase A D_ttft == d34 (interior argmin, non-monotone row)",
         "got": a["D_ttft"]["arm"], "want": "d34", "pass": a["D_ttft"]["arm"] == "d34"},
        {"name": "phase A D_itl NOT IDENTIFIED (overlapping arms) -- NB this "
                 "resolves to ITL_SATURATED, NOT to UNIDENTIFIED.  sec7's "
                 "claim that PC-D exercises the UNIDENTIFIED path (and the "
                 "notes' '4 of 9 verdicts') is WRONG: UNIDENTIFIED is reached "
                 "only by synthetic fixtures.",
         "got": f'identified={a["D_itl"]["identified"]} verdict={a["verdict"]} '
                f'flags={[f["verdict"] for f in a["verdict_flags"]]}',
         "want": "identified=False", "pass": a["D_itl"]["identified"] is False},
    ]
    for check in pc_d["verdict_path_checks"]:
        if not check["pass"]:
            pc_d["passed"] = False
    if not pc_d["passed"]:
        failures.append("PC-D")
    pc_d["workload_caveat"] = (
        "g2_0_hard is a DIFFERENT workload (random-ids in2048, o32/o1024, "
        "NPROMPT=32, ROUNDS=1).  This control exercises the CODE PATH; it does "
        "not transplant results (sec7).")
    out["PC_D"] = pc_d

    # ---- PC-B: same-code-path signature assert, + the negative control -----
    #
    # The headline signature is captured by CALLING the estimand exactly as
    # ``load_campaign_grid`` calls it (positional path, every kwarg defaulted),
    # not by declaring what it ought to be.  ``load_campaign_grid`` records
    # under this same tag, so on a campaign run the comparison is against the
    # real thing rather than a stand-in.
    probe0 = sorted(slo_dir.glob("sgptvHi_d44_rep*_L3H12_*.jsonl"))[0]
    with sig_context("headline"):
        boot_estimands(probe0)
    tags = ["PC-A/PC-E", "PC-C", "PC-D"]
    headline_sig = signature_set("headline")
    pc_b: Dict[str, object] = {"headline_signature": headline_sig, "per_control": {}}
    same = True
    for tag in tags:
        sigs = signature_set(tag)
        # compare on (function, kwargs) and on every branch flag except the
        # data-dependent ones, which legitimately differ across datasets
        def strip(records):
            out_ = []
            for record in records:
                branch = {k: v for k, v in record["branch"].items()
                          if k not in ("empty_itl_branch", "multi_round")}
                out_.append({"function": record["function"],
                             "kwargs": record["kwargs"], "branch": branch})
            deduped = []
            for item in out_:
                if item not in deduped:
                    deduped.append(item)
            return deduped
        # an EMPTY signature set never passes: it means the control never
        # executed the estimand at all
        match = bool(sigs) and strip(sigs) == strip(headline_sig)
        pc_b["per_control"][tag] = {"signatures": sigs, "matches_headline": match}
        same = same and match
    pc_b["passed"] = same
    if not same:
        failures.append("PC-B")

    # PC-B-neg: prove the comparison has TEETH (gate #9 -- a check that always
    # passes is an identity, not a control).  Deliberately vary one kwarg and
    # assert the comparison DETECTS it.
    probe = sorted(slo_dir.glob("sgptvHi_d44_rep*_L3H12_*.jsonl"))[0]
    with sig_context("PC-B-neg"):
        boot_estimands(probe, itl_slo_ms=55.0)      # one kwarg changed
    neg_detected = signature_set("PC-B-neg") != headline_sig
    pc_b["negative_control"] = {
        "description": "same function, itl_slo_ms 60 -> 55; the signature "
                       "comparison MUST report a difference",
        "difference_detected": neg_detected,
        "passed": neg_detected,
    }
    if not neg_detected:
        pc_b["passed"] = False
        failures.append("PC-B-neg")
    out["PC_B"] = pc_b

    out["failures"] = failures
    out["all_passed"] = not failures
    out["gate"] = ("sec7: if a positive control FAILS, no new numbers may be "
                   "reported.")
    return out


# ---------------------------------------------------------------------------
# sec7 residual gap -- synthetic calibration of the donor-selection statistic.
# (``unpaired_bootstrap_ci`` is a difference-of-means tool; K3 borrows only its
# conventions, so the donor rule itself has NO canonical control.  This is
# registered as ACKNOWLEDGED_CONTROL_GAPS, not silently omitted.)
# ---------------------------------------------------------------------------

def calibrate_donor_rule(*, n_blocks: int = K5_BLOCKS, n_arms: int = 7,
                         noise_sd: float = 1.8, sims: int = 400,
                         bootstrap: int = 2000, seed: int = 1) -> Dict[str, object]:
    rng = random.Random(seed)
    arms = [f"d{16 + 10 * i}" for i in range(n_arms)]

    def one_trial(true_gap: float) -> bool:
        blocks = []
        for _ in range(n_blocks):
            row = {}
            for index, arm in enumerate(arms):
                mean = -true_gap if index == n_arms - 1 else 0.0
                row[arm] = rng.gauss(mean, noise_sd)
            blocks.append(row)
        counts: Dict[str, int] = {}
        for row in blocks:
            winner, _ = _argmin_arm(row)
            counts[winner] = counts.get(winner, 0) + 1
        rank_arm = max(counts, key=lambda a: (counts[a], -arm_sm(a)))
        if counts[rank_arm] < K1_BLOCK_ARGMIN_MIN:
            return False
        freq: Dict[str, int] = {}
        for _ in range(bootstrap):
            picked = [blocks[rng.randrange(n_blocks)] for _ in range(n_blocks)]
            means = {arm: statistics.fmean(p[arm] for p in picked) for arm in arms}
            winner, _ = _argmin_arm(means)
            freq[winner] = freq.get(winner, 0) + 1
        boot_arm = max(freq, key=lambda a: (freq[a], -arm_sm(a)))
        return (freq[boot_arm] / bootstrap >= K1_BOOTSTRAP_MIN_FRAC
                and boot_arm == rank_arm)

    false_id = sum(one_trial(0.0) for _ in range(sims)) / sims
    power = {gap: sum(one_trial(gap) for _ in range(sims)) / sims
             for gap in (0.5, 1.0, 2.0, 5.0, 9.0)}
    return {
        "design": {"n_blocks": n_blocks, "n_arms": n_arms, "noise_sd_ms": noise_sd,
                   "sims": sims, "bootstrap": bootstrap, "seed": seed,
                   "note": "calibration uses a reduced bootstrap count; the "
                           "headline always uses 10000/seed=1 (K3)"},
        "false_identification_rate_under_true_tie": false_id,
        "power_by_true_gap_ms": power,
        "reads": "gap 9 ms is the g2_0_hard phase-B d44->d74 movement; gap 0.2 ms "
                 "is the observed sgptv d34->d44 separation (well inside noise)",
    }


def verdict_reachability() -> Dict[str, object]:
    """GO criterion 3: every one of the nine verdicts is reachable in code."""
    def synth_blocks(specs: Sequence[Mapping[str, Tuple[float, float]]]) -> List[BootRecord]:
        """Explicit per-block values.

        NOTE: an earlier version of this fixture added a COMMON-MODE wobble to
        every arm of a block, which cancels inside argmin and therefore could
        never produce a block disagreement.  Per-block specs are given
        explicitly so the fixture actually exercises the disagreement paths.
        """
        records = []
        for index, spec in enumerate(specs, start=1):
            for arm, (ttft, itl) in spec.items():
                records.append(BootRecord(
                    arm=arm, decode_sm=arm_sm(arm), block=f"blk{index}", boot=1,
                    phase="HI", path="synthetic",
                    est={"M_ttft": ttft, "M_itl": itl,
                         "requests": 100.0, "duration_s": 1.0,
                         "throughput_req_s": 1.0, "goodput_req_s": 1.0,
                         "ttft_pass_pct": 0.0, "itl_p95_pass_pct": 0.0,
                         "joint_pass_pct": 0.0, "band_mass_ttft": 0.0,
                         "band_mass_itl": 0.0, "empty_itl_requests": 0.0,
                         "ttft_p50_ms": 0.0, "ttft_p95_ms": 0.0, "ttft_p99_ms": 0.0,
                         "token_itl_p50_ms": 0.0, "token_itl_p95_ms": 0.0,
                         "token_itl_p99_ms": 0.0}))
        return records

    def steady(spec: Mapping[str, Tuple[float, float]]) -> List[BootRecord]:
        return synth_blocks([spec] * 4)

    # ITL argmin rotates across blocks (rank rule fails) while the arm MEANS
    # stay within K7 delta -> non-identification is saturation, not noise.
    saturated = synth_blocks([
        {"d16": (300, 50.0), "d34": (100, 50.3), "d54": (200, 50.6)},
        {"d16": (300, 50.5), "d34": (100, 50.1), "d54": (200, 50.4)},
        {"d16": (300, 50.6), "d34": (100, 50.4), "d54": (200, 50.0)},
        {"d16": (300, 50.2), "d34": (100, 50.5), "d54": (200, 50.3)},
    ])
    # ITL argmin rotates AND the arm means are far apart (gap >> delta).
    unidentified = synth_blocks([
        {"d16": (300, 10), "d34": (100, 50), "d54": (200, 90)},
        {"d16": (300, 90), "d34": (100, 10), "d54": (200, 50)},
        {"d16": (300, 50), "d34": (100, 90), "d54": (200, 10)},
        {"d16": (300, 10), "d34": (100, 90), "d54": (200, 50)},
    ])

    seen: Dict[str, str] = {}
    cases = {
        # D_ttft interior, D_itl above it -> positive tax
        "TAX_POSITIVE": steady({"d16": (300, 90), "d34": (100, 70), "d54": (200, 50)}),
        # both donors interior and identical
        "NO_TAX": steady({"d16": (300, 90), "d34": (100, 50), "d54": (200, 70)}),
        # D_itl below an interior D_ttft
        "NEGATIVE_INTERIOR": steady({"d16": (300, 50), "d34": (100, 70), "d54": (200, 90)}),
        # D_ttft at the top of the grid
        "TRUNCATED": steady({"d16": (300, 90), "d34": (200, 70), "d54": (100, 50)}),
        # D_ttft at the bottom of the grid
        "TRUNCATED_LOW": steady({"d16": (100, 90), "d34": (200, 70), "d54": (300, 50)}),
        # ITL indistinguishable across contenders (< K7 delta)
        "ITL_SATURATED": saturated,
        # contenders far apart but blocks disagree -> plain non-identification
        "UNIDENTIFIED": unidentified,
    }
    for want, grid in cases.items():
        got = decide(grid, label=f"synthetic {want}")["verdict"]
        seen[want] = got

    # ladder-only verdicts
    ladder_grid = steady({"d16": (300, 90), "d34": (100, 70), "d54": (200, 50)})
    ladder = decide(ladder_grid, label="synthetic ladder")["delta_slo_ladder"]
    ladder_verdicts = {rung["verdict"] for rung in ladder}
    seen["S_ITL_UNREACHED"] = ("S_ITL_UNREACHED" if "S_ITL_UNREACHED" in ladder_verdicts
                               else "NOT_REACHED")
    unconstrained = steady({"d16": (300, 40), "d34": (100, 41), "d54": (200, 42)})
    ladder2 = decide(unconstrained, label="synthetic unconstrained")["delta_slo_ladder"]
    seen["ITL_UNCONSTRAINED"] = ("ITL_UNCONSTRAINED"
                                 if any(r["verdict"] == "ITL_UNCONSTRAINED" for r in ladder2)
                                 else "NOT_REACHED")
    reached = {name: (name == got) for name, got in seen.items()}
    return {"per_verdict": seen, "all_nine_reachable": all(reached.values()),
            "missing": [name for name, ok in reached.items() if not ok],
            "n_verdicts_defined": len(VERDICTS)}


# ---------------------------------------------------------------------------
# Campaign entry point.
# ---------------------------------------------------------------------------

def run_campaign(directory: Path, *, include_smoke: bool = False) -> Dict[str, object]:
    out: Dict[str, object] = {"scope": NOMINAL_SPLIT_SCOPE, "frame": HEADLINE_FRAME}
    for phase in ("LO", "HI"):
        grid, provenance = load_campaign_grid(directory, phase,
                                              include_smoke=include_smoke)
        if not grid:
            out[phase] = {"status": "NO_ADOPTED_BOOTS", "provenance": provenance}
            continue
        decision = decide(grid, label=f"G16 campaign {phase}")
        decision["provenance"] = provenance
        decision["drift_corrected"] = {
            "M_ttft": drift_corrected_means(grid, "M_ttft"),
            "M_itl": drift_corrected_means(grid, "M_itl"),
        }
        for key in ("M_ttft", "M_itl"):
            corrected = decision["drift_corrected"][key]
            donor = decision["D_ttft" if key == "M_ttft" else "D_itl"]
            if corrected and donor.get("arm") and corrected["argmin_arm"] != donor["arm"]:
                decision.setdefault("DRIFT_DISAGREEMENT", []).append({
                    "metric": key, "primary": donor["arm"],
                    "drift_corrected": corrected["argmin_arm"],
                    "registered_as": "a result in its own right (add. A-3); "
                                     "neither version is chosen post hoc"})
        # side outputs (pre-registered, never compared post hoc)
        decision["side_outputs"] = {
            "per_arm": {arm: {key: statistics.fmean(
                [r.est[key] for r in grid if r.arm == arm])
                for key in ("ttft_pass_pct", "itl_p95_pass_pct", "joint_pass_pct",
                            "goodput_req_s", "throughput_req_s",
                            "band_mass_ttft", "band_mass_itl")}
                for arm in sorted({r.arm for r in grid}, key=arm_sm)},
            "residency_fraction_by_arm": {
                r.arm: r.residency_fraction for r in grid if r.residency_fraction},
            "split_transitions_by_boot": {
                Path(r.path).name: r.split_transitions for r in grid},
        }
        # secondary: canonical threshold donors + K2 cliff routing
        band_max_ttft = max(v["band_mass_ttft"] for v in decision["side_outputs"]["per_arm"].values())
        band_max_itl = max(v["band_mass_itl"] for v in decision["side_outputs"]["per_arm"].values())
        pass_ttft = {arm: v["ttft_pass_pct"] for arm, v in decision["side_outputs"]["per_arm"].items()}
        pass_itl = {arm: v["itl_p95_pass_pct"] for arm, v in decision["side_outputs"]["per_arm"].items()}
        decision["secondary_threshold_donors"] = {
            "ttft": {"argmax": max(pass_ttft, key=lambda a: (pass_ttft[a], -arm_sm(a))),
                     "band_mass_max": band_max_ttft,
                     "routing": "DIAGNOSTIC_ONLY" if band_max_ttft > K2_BAND_MASS_MAX else "citable"},
            "itl": {"argmax": max(pass_itl, key=lambda a: (pass_itl[a], -arm_sm(a))),
                    "band_mass_max": band_max_itl,
                    "routing": "DIAGNOSTIC_ONLY" if band_max_itl > K2_BAND_MASS_MAX else "citable"},
            "aggregation": "max over donor-candidate arms (C8)",
            "tie_warning": "pass fractions are DISCRETE (multiples of 1/n), so "
                           "exact ties are realistic here; the argmax below "
                           "breaks ties by smaller decode SM.  ORACLE R1 "
                           "already recorded a tie-break-dependent headline "
                           "('116') -- check for ties before citing.",
            "ttft_tied": sorted(a for a, v in pass_ttft.items()
                                if v == max(pass_ttft.values())),
            "itl_tied": sorted(a for a, v in pass_itl.items()
                               if v == max(pass_itl.values())),
            "note": "if primary and secondary disagree, THAT is the registered "
                    "result; neither is chosen post hoc (sec4)",
        }
        # sec3-1: throughput rank law, checked on the extended grid
        arms_sorted = sorted({r.arm for r in grid}, key=arm_sm)
        thr = {a: decision["side_outputs"]["per_arm"][a]["throughput_req_s"] for a in arms_sorted}
        tpass = {a: decision["side_outputs"]["per_arm"][a]["ttft_pass_pct"] for a in arms_sorted}
        rank_thr = sorted(arms_sorted, key=lambda a: -thr[a])
        rank_pass = sorted(arms_sorted, key=lambda a: -tpass[a])
        decision["throughput_rank_law"] = {
            "throughput_rank": rank_thr, "ttft_pass_rank": rank_pass,
            "identical": rank_thr == rank_pass,
            "citation_ban": "this agreement is structurally near-forced in an "
                            "overloaded closed-budget FCFS queue -- NEVER cite "
                            "it as independent mechanistic evidence (sec3-1, "
                            "gate #9)",
        }
        out[phase] = decision
    out["ACKNOWLEDGED_CONTROL_GAPS"] = [
        "the block-bootstrap DONOR-SELECTION statistic has no canonical "
        "control (unpaired_bootstrap_ci is a difference-of-means tool); "
        "synthetic calibration below is the substitute (sec7)",
        "node/date remain a RECORDING axis only (SLURM placement) -- 'node axis "
        "unmeasured' accompanies every primary result (sec5)",
        "telemetry is not a free observer: the 2026-07-15/18 grid ran WITHOUT "
        "instrumentation, so M_ttft/M_itl absolute values are NOT compared to "
        "it (add. B-2 / N7)",
    ]
    return out


def _self_sha256() -> str:
    """F4 (audit): H9/A-5(i) hash the harness but NOT the file that computes
    the decision quantity.  Every report therefore carries the analyzer's own
    digest, so a pre-registered analyzer cannot be silently edited after seeing
    the data (gate #19).
    """
    import hashlib
    return hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--campaign", action="store_true")
    parser.add_argument("--controls", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--include-smoke", action="store_true",
                        help="analyse smoke artifacts (plumbing check only; "
                             "never for a decision quantity)")
    parser.add_argument("--slo-dir", type=Path, default=HERE)
    parser.add_argument("--g20h-dir", type=Path,
                        default=ENGINE_PORT / "results" / "g2_0_hard")
    parser.add_argument("--calib-sims", type=int, default=400)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    if not (args.campaign or args.controls or args.self_test):
        parser.error("choose at least one of --campaign / --controls / --self-test")

    report: Dict[str, object] = {
        "spec": "PREREG_G16_RULES_REV3_2026-08-16.md (rev3 is the only valid "
                "revision)",
        "analyzer_sha256": _self_sha256(),
        "analyzer_path": str(Path(__file__).resolve()),
        "scope": NOMINAL_SPLIT_SCOPE,
        "frame": HEADLINE_FRAME,
        "knobs": {"K1_blocks": K1_BLOCK_ARGMIN_MIN, "K1_bootstrap": K1_BOOTSTRAP_MIN_FRAC,
                  "K2_band": K2_BAND_FRACTION, "K2_band_max": K2_BAND_MASS_MAX,
                  "K3": [K3_BOOTSTRAP_SAMPLES, K3_BOOTSTRAP_SEED],
                  "K4": K4_GRID_MAX_ARM, "K5": K5_BLOCKS,
                  "K6": [K6_RATE_LO, K6_RATE_HI, K6_NP, K6_ROUNDS],
                  "K7_delta_ms": K7_DELTA_MS,
                  "K8_ladder": [K8_SLO_LADDER[0], K8_SLO_LADDER[-1], 1],
                  "K9": K9_ADAPTIVE_EXTRA_BLOCKS,
                  "K10_pin_never_discards": K10_PIN_FRAC_NEVER_DISCARDS},
        "spec_ambiguities": [
            "RESOLVED 2026-08-17 (prereg addendum C, claims-auditor single-"
            "question audit, decided BEFORE any block-1 artifact existed): "
            "sec6's 'upper arms' = K11, pinned at decode SM >= 44 (== top-half"
            "-by-SM on the 7-arm campaign grid).  The CONTENDING set is "
            "REFUTED as primary -- it is derived from the block argmin whose "
            "failure ITL_SATURATED explains, it is nan by construction when "
            "the winner is unanimous, and on the old 4-arm LO grid it fires "
            "ITL_SATURATED on the two BOTTOM arms (gap 0.069) while printing "
            "'saturated at the upper end' (true upper gap 0.944).  "
            "gap_contenders_ms / gap_top_half_by_sm_ms remain as DIAGNOSTICS.",
            "sec7 PC-D calls phase-B D_ttft=d34 'interior', but d34 is the "
            "BOTTOM boundary of the g2_0_hard grid {34..74}; under the literal "
            "S_min rule that cell is TRUNCATED_LOW.  Both are emitted: "
            "'verdict' (dominant caveat) and 'sign_verdict' (TAX_POSITIVE, "
            "delta=+40).  This is a pre-registration text ambiguity, NOT a "
            "measurement failure -- route to claims-auditor before registering",
            "sec7 PC-C target 'D_itl = d44' holds only as a BARE ARGMIN; under "
            "sec6's own K1 gate the ITL donor is NOT identified on the old grid "
            "(bootstrap d44 ~0.64 / d34 ~0.36 < 0.80), which REPRODUCES ORACLE "
            "sec3-4 (6028/3972) and sec3-3 registration-ban #1.  The target was "
            "phrased with the estimator but without the identification gate",
            "sec6 lists nine verdicts with no precedence rule and they are not "
            "mutually exclusive (TRUNCATED and ITL_SATURATED co-hold on the old "
            "grid).  A dominant 'verdict' plus complete 'verdict_flags' are both "
            "emitted so no pre-declared result is dropped",
        ],
    }
    if args.self_test:
        report["verdict_reachability"] = verdict_reachability()
        report["donor_rule_calibration"] = calibrate_donor_rule(sims=args.calib_sims)
    controls_failed = False
    if args.controls:
        report["positive_controls"] = run_controls(args.slo_dir, args.g20h_dir)
        controls_failed = not report["positive_controls"]["all_passed"]
    if args.campaign:
        campaign = run_campaign(args.slo_dir, include_smoke=args.include_smoke)
        if controls_failed:
            # sec7: "if a positive control FAILS, no new numbers may be
            # reported."  Writing them to the file and merely exiting 2 does
            # not implement that -- the numbers are then on disk to be quoted.
            report["campaign"] = {
                "SUPPRESSED": True,
                "why": "a positive control failed; sec7 forbids reporting new "
                       "numbers.  Fix the control, then re-run.",
                "failures": report["positive_controls"]["failures"],
            }
        else:
            report["campaign"] = campaign
        if not args.controls:
            report["campaign"]["UNCONTROLLED_WARNING"] = (
                "run with --controls: campaign numbers reported without the "
                "sec7 positive controls having been executed in this run")

    text = json.dumps(report, indent=2, sort_keys=True, default=str)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text)

    if args.self_test:
        reach = report["verdict_reachability"]
        print(f"\nverdict reachability: all_nine={reach['all_nine_reachable']} "
              f"missing={reach['missing']}", file=sys.stderr)
    if args.controls:
        controls = report["positive_controls"]
        print(f"positive controls: all_passed={controls['all_passed']} "
              f"failures={controls['failures']}", file=sys.stderr)
        if not controls["all_passed"]:
            print("sec7 GATE: a control failed -> NO new numbers may be reported.",
                  file=sys.stderr)
            raise SystemExit(2)


if __name__ == "__main__":
    main()
