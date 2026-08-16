#!/usr/bin/env python3
"""R1/R2 oracle re-analysis (2026-08-16, rev2) -- GPU 0, re-scoring of existing artifacts.

NOT CANONICAL.  Produces `oracle_reanalysis_2026-08-16.json` next to this file.
Promotion to CONSENSUS.md is doc-steward's call.

rev2 (2026-08-16, after claims-auditor adversarial audit).  Changes:
  * decision quantity 2 (SM sum > 108) is DEMOTED to an ALGEBRAIC IDENTITY.
    sum = (108 - D_ttft) + D_itl > 108  <=>  D_itl > D_ttft.  In sgptv HI the
    TTFT argmax is d44 = the most decode-heavy arm in the measured grid, so
    D_itl <= 44 = D_ttft holds with probability 1 independent of the data.  The
    "0/10000 bootstrap draws exceed 108" figure has ZERO power and is reported
    as such.  (This identity was already written down in
    reports/PRIZE_SIZE_ARGUMENT_2026-08-16.md 2.3(2); rev1 failed to apply a
    diagnostic the repo already owned -- methodology gate #18.)
  * PC6 is relabelled an IDENTITY control (`independent=False`) and proven to be
    one in code; `itl_p95_pass_frac` -- the quantity that SELECTS the ITL donor
    -- therefore has ZERO independent controls, recorded as an acknowledged gap.
  * `assert_paths_covered` now actually compares `(qualname, kwargs, key)`, so
    the threshold ladders (different kwargs) show up as reported-but-uncontrolled.
  * `unpaired_bootstrap_ci` is wired in for real (rev1 imported it and never
    called it; `bootstrap_over_arms` was dead code -- both are reproduction-path
    defects of the same family as the 2026-08-14 E-1a and 2026-08-16 C2-R errata).
  * jackknife + node/day-matched subset sensitivity for the HI oracle.
  * combination-assumption error computed under BOTH predicates (rev1 quoted the
    canonical-predicate number for a claim about 1-20, whose own predicate is legacy).
  * cross-grid discriminant: TTFT-pass rank vs achieved-throughput rank, and the
    per-request prefill:decode token ratio.  ("overload" is NOT a discriminant:
    both grids are overloaded.)

Everything numeric goes through ONE scorer, `score_run`, a thin wrapper over the
canonical library `pdmux_eval.analyze`:
  * `load_bench_serving_rounds`  -- one JSON object per round, duration = SUM (gate #7)
  * `RequestResult.passes`       -- canonical predicate (gate #4): TTFT<=SLO AND
                                    p95 of that request's own token ITLs <= SLO
  * `RequestResult.passes_legacy_mean` -- legacy mean-ITL predicate (secondary)
  * `analyze.percentile`         -- canonical percentile (linear interpolation)
  * `analyze.unpaired_bootstrap_ci` -- arm-vs-arm CI (seed=1, 10000 samples)

Run:
  PYTHONPATH=workspace/engine-port/benchmarks \
    python3 workspace/engine-port/results/slo_sched/oracle_reanalysis_2026_08_16.py
"""

from __future__ import annotations

import json
import math
import random
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "workspace" / "engine-port" / "benchmarks"))

from pdmux_eval.analyze import (  # noqa: E402
    load_bench_serving_rounds,   # duration = SUM over round records (gate #7)
    percentile,                  # canonical percentile (linear interpolation)
    unpaired_bootstrap_ci,       # arm-vs-arm CI, seed=1 / 10000 samples
)
# NB: every name imported here is called below.  rev1 shipped an unused
# `RequestResult` import and a never-called `unpaired_bootstrap_ci`; unused
# imports are how reproduction-path gaps hide, so keep this list minimal.

TTFT_SLO_MS = 3000.0  # he2_bench.sbatch:96  `t<=3.0`
ITL_SLO_MS = 60.0     # he2_bench.sbatch:96  `m<=0.06`  (same in sharegpt_vary_bench.sbatch:96)
OPERATING_POINT = (TTFT_SLO_MS, ITL_SLO_MS)
BOOT_SAMPLES = 10000
BOOT_SEED = 1
GPU_SM_TOTAL = 108
DECODE_SM = {"d16": 16, "d24": 24, "d34": 34, "d44": 44, "d54": 54}

# --------------------------------------------------------------------------
# the one scorer.  headline AND positive controls both call this.
# --------------------------------------------------------------------------

_CALL_LOG: List[Tuple[str, Tuple[float, float]]] = []


def score_run(path: Path, ttft_slo_ms: float, itl_slo_ms: float) -> Dict[str, float]:
    """Score one bench_serving artifact (all rounds pooled, duration summed)."""
    _CALL_LOG.append(("score_run", (ttft_slo_ms, itl_slo_ms)))
    requests, duration = load_bench_serving_rounds(path)
    if duration <= 0:
        raise ValueError(f"non-positive summed duration for {path}")
    n = len(requests)
    ttfts = [r.ttft_ms for r in requests]
    itl_p95_per_request = [percentile(r.token_itl_ms, 0.95) for r in requests]
    itl_mean_per_request = [
        statistics.fmean(r.token_itl_ms) if r.token_itl_ms else math.inf
        for r in requests
    ]
    ttft_pass = sum(t <= ttft_slo_ms for t in ttfts)
    itl_p95_pass = sum(v <= itl_slo_ms for v in itl_p95_per_request)
    itl_mean_pass = sum(v <= itl_slo_ms for v in itl_mean_per_request)
    # NB (rev2): `RequestResult.passes` recomputes percentile(token_itl_ms, 0.95)
    # internally, i.e. `good` and `itl_p95_pass` share a literally identical
    # subexpression.  Wherever ttft_pass == n they are the SAME INTEGER.  This is
    # what makes PC6 an identity rather than a control -- see `identity_proof`.
    good = sum(r.passes(ttft_slo_ms, itl_slo_ms) for r in requests)
    good_legacy = sum(r.passes_legacy_mean(ttft_slo_ms, itl_slo_ms) for r in requests)
    pooled_itls = [v for r in requests for v in r.token_itl_ms]
    return {
        "requests": float(n),
        "duration_s": duration,
        "throughput_req_s": n / duration,
        # canonical predicate (gate #4)
        "good": float(good),
        "slo_goodput_req_s": good / duration,
        "joint_pass_frac": good / n,
        "ttft_pass_frac": ttft_pass / n,
        "itl_p95_pass_frac": itl_p95_pass / n,
        "itl_p95_pass_count": float(itl_p95_pass),
        # legacy mean-ITL predicate (secondary only)
        "legacy_good": float(good_legacy),
        "legacy_mean_itl_goodput_req_s": good_legacy / duration,
        "legacy_joint_pass_frac": good_legacy / n,
        "legacy_itl_mean_pass_frac": itl_mean_pass / n,
        # distributions (gate #6 requires the TTFT distribution alongside goodput)
        "ttft_p50_ms": percentile(ttfts, 0.50),
        "ttft_p95_ms": percentile(ttfts, 0.95),
        "ttft_p99_ms": percentile(ttfts, 0.99),
        "token_itl_p50_ms": percentile(pooled_itls, 0.50),
        "token_itl_p95_ms": percentile(pooled_itls, 0.95),
        "token_itl_p99_ms": percentile(pooled_itls, 0.99),
        "req_itl_p95_p50_ms": percentile(itl_p95_per_request, 0.50),
        "req_itl_p95_p90_ms": percentile(itl_p95_per_request, 0.90),
        # separation diagnostics for the (x) combination assumption
        "min_itl_p95_among_ttft_pass_ms": min(
            [v for t, v in zip(ttfts, itl_p95_per_request) if t <= ttft_slo_ms],
            default=math.nan,
        ),
        "min_ttft_among_itl_p95_pass_ms": min(
            [t for t, v in zip(ttfts, itl_p95_per_request) if v <= itl_slo_ms],
            default=math.nan,
        ),
    }


# --------------------------------------------------------------------------
# positive-control harness (rev2: independence is now tracked and enforced)
# --------------------------------------------------------------------------


@dataclass
class PositiveControl:
    name: str
    target: float
    observed: float
    tolerance: float
    source: str
    key: str
    independent: bool = True
    kwargs: Tuple[float, float] = OPERATING_POINT

    @property
    def path_signature(self) -> Tuple[str, Tuple[float, float], str]:
        return ("score_run", self.kwargs, self.key)

    @property
    def passed(self) -> bool:
        return abs(self.observed - self.target) <= self.tolerance

    def as_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "independent": self.independent,
            "canonical_target": self.target,
            "observed": round(self.observed, 6),
            "abs_error": round(abs(self.observed - self.target), 6),
            "tolerance": self.tolerance,
            "passed": self.passed,
            "target_source": self.source,
            "exercised_path": [self.path_signature[0], list(self.path_signature[1]),
                               self.path_signature[2]],
        }


# Paths a headline number reads.  Coverage is checked against the FULL signature
# (function, kwargs, key), so a control taken at a different SLO does not count.
HEADLINE_PATHS: List[Tuple[str, Tuple[float, float], str]] = [
    ("score_run", OPERATING_POINT, "slo_goodput_req_s"),
    ("score_run", OPERATING_POINT, "joint_pass_frac"),
    ("score_run", OPERATING_POINT, "ttft_pass_frac"),
    ("score_run", OPERATING_POINT, "itl_p95_pass_frac"),
    ("score_run", OPERATING_POINT, "legacy_mean_itl_goodput_req_s"),
    ("score_run", OPERATING_POINT, "legacy_joint_pass_frac"),
    ("score_run", OPERATING_POINT, "legacy_itl_mean_pass_frac"),
    ("score_run", OPERATING_POINT, "req_itl_p95_p90_ms"),
    ("score_run", OPERATING_POINT, "ttft_p50_ms"),
]

# Declared, non-fatal coverage gaps.  Anything NOT in here that is uncovered
# aborts the run.  Every entry must state why an independent target is impossible.
ACKNOWLEDGED_CONTROL_GAPS: Dict[Tuple[str, Tuple[float, float], str], str] = {
    ("score_run", OPERATING_POINT, "itl_p95_pass_frac"):
        "NO INDEPENDENT CONTROL EXISTS. No canonical document publishes a bare "
        "per-request-ITL-p95 PASS RATE. PC6 is an ALGEBRAIC IDENTITY with PC4 "
        "(proven in `identity_proof`), so it adds zero evidence. This is the "
        "quantity that SELECTS the ITL donor of the decoupled oracle, and its "
        "operating range in sgptv HI (18-93%) is entirely uncontrolled.",
    ("score_run", OPERATING_POINT, "joint_pass_frac"):
        "Algebraic transform of `slo_goodput_req_s` (covered independently by "
        "PC4) via the file-read constants `requests` and `duration_s`; not "
        "independently anchored to any published pass-rate.",
}


@dataclass
class ControlRegistry:
    controls: List[PositiveControl] = field(default_factory=list)

    def add(self, **kw) -> None:
        self.controls.append(PositiveControl(**kw))

    def covered(self) -> set:
        return {c.path_signature for c in self.controls if c.passed and c.independent}

    def coverage_report(self) -> Dict[str, object]:
        covered = self.covered()
        rows = []
        for sig in HEADLINE_PATHS:
            state = ("covered" if sig in covered
                     else ("acknowledged_gap" if sig in ACKNOWLEDGED_CONTROL_GAPS
                           else "UNCOVERED"))
            rows.append({
                "path": [sig[0], list(sig[1]), sig[2]],
                "state": state,
                "reason": ACKNOWLEDGED_CONTROL_GAPS.get(sig, ""),
            })
        uncovered = [r for r in rows if r["state"] == "UNCOVERED"]
        if uncovered:
            raise AssertionError(
                f"headline scorer paths with no independent control and no "
                f"declared gap: {uncovered}. Refusing to emit (gate #9)."
            )
        # everything score_run was actually called with, minus the controlled set
        invoked = {("score_run", kw) for _, kw in _CALL_LOG}
        uncontrolled_kwargs = sorted(
            {kw for _, kw in invoked} - {sig[1] for sig in covered}
        )
        return {
            "headline_paths": rows,
            "n_independent_controls": sum(c.independent and c.passed for c in self.controls),
            "n_identity_controls": sum((not c.independent) and c.passed for c in self.controls),
            "reported_but_uncontrolled_slo_settings": [list(k) for k in uncontrolled_kwargs],
            "reported_but_uncontrolled_note":
                "The threshold ladders and sensitivity sweeps call score_run at "
                "SLO settings no positive control ever exercised; their outputs "
                "are diagnostics, not controlled measurements.",
        }

    def all_passed(self) -> bool:
        return all(c.passed for c in self.controls)


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------

HE2_JOBS = {  # arm -> [(rep, job)]   phase A/B come from the SAME job (same boot)
    "d16": [("rep81", "860497"), ("rep82", "860504")],
    "d24": [("rep81", "860498"), ("rep82", "860507")],
    "d34": [("rep81", "860501"), ("rep82", "860510")],
    "d44": [("rep81", "860502"), ("rep82", "860517")],
    "bind": [("rep81", "860503"), ("rep82", "860518")],
}
HE2_STATICS = ["d16", "d24", "d34", "d44"]

# The n=4 sets recomputed by CONSENSUS.md 1-13 footnotes E/F.  Identification is
# not assumed: PC4 reproduces the published n=4 means AND SDs, which is what pins
# these particular reps/jobs to that footnote.
SGPTV_JOBS = {
    "d16": [("rep1", "852340"), ("rep44", "859008"), ("rep45", "859025"), ("rep46", "859026")],
    "d24": [("rep1", "852341"), ("rep44", "859005"), ("rep45", "859006"), ("rep46", "859007")],
    "d34": [("rep41", "856892"), ("rep42", "856917"), ("rep43", "856918"), ("rep44", "856929")],
    "d44": [("rep1", "852342"), ("rep41", "856889"), ("rep42", "856890"), ("rep43", "856891")],
    "slo": [("rep1", "852343"), ("rep44", "859037"), ("rep45", "859038"), ("rep46", "859059")],
}
SGPTV_STATICS = ["d16", "d24", "d34", "d44"]
# node/day-matched subset: gpu38, 2026-07-17 (the only cross-arm matched cells)
MATCHED_SUBSET = {"d34": [0, 1, 2, 3], "d44": [1, 2, 3]}


def he2_path(phase: str, arm: str, rep: str, job: str) -> Path:
    return HERE / f"he2{phase}_{arm}_{rep}_rA8B5_{job}.jsonl"


def sgptv_path(phase: str, arm: str, rep: str, job: str) -> Path:
    return HERE / f"sgptv{phase}_{arm}_{rep}_L3H12_{job}.jsonl"


def slurm_provenance(prefix: str, job: str) -> Dict[str, str]:
    out = HERE / f"{prefix}_{job}.out"
    node, date = "?", "?"
    if out.exists():
        for line in out.read_text(errors="ignore").splitlines()[:12]:
            if "SLURM_NODELIST" in line:
                node = line.split("=")[-1].strip()
        date = __import__("datetime").datetime.fromtimestamp(
            out.stat().st_mtime
        ).strftime("%Y-%m-%d %H:%M")
    return {"node": node, "finished": date}


# --------------------------------------------------------------------------
# stats helpers.  Outermost statistical unit = SLURM job = server boot.
# --------------------------------------------------------------------------

T_CRIT_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 8: 2.306}


def mean_sd(values: Sequence[float]) -> Dict[str, float]:
    values = [float(v) for v in values]
    n = len(values)
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if n > 1 else 0.0
    half = T_CRIT_95.get(n - 1, 12.706) * sd / math.sqrt(n) if n > 1 else math.nan
    return {"n": float(n), "mean": mean, "sd": sd,
            "t95_low": mean - half, "t95_high": mean + half, "values": values}


def arm_comparison(baseline: Sequence[float], proposed: Sequence[float],
                   label: str) -> Dict[str, object]:
    """Real call into the canonical `unpaired_bootstrap_ci` (rev1 never called it)."""
    result = dict(unpaired_bootstrap_ci(list(baseline), list(proposed)))
    result["comparison"] = label
    result["conservative_t95_on_difference"] = _welch_like_t95(baseline, proposed)
    return result


def _welch_like_t95(a: Sequence[float], b: Sequence[float]) -> List[float]:
    """Conservative interval on the difference of means (percentile bootstrap has
    a documented under-coverage history in this repo)."""
    na, nb = len(a), len(b)
    va = statistics.variance(a) if na > 1 else 0.0
    vb = statistics.variance(b) if nb > 1 else 0.0
    se = math.sqrt(va / na + vb / nb)
    df = max(1, min(na, nb) - 1)
    t = T_CRIT_95.get(df, 12.706)
    d = statistics.fmean(b) - statistics.fmean(a)
    return [d - t * se, d + t * se]


# --------------------------------------------------------------------------
# shared analysis pieces
# --------------------------------------------------------------------------


def cliff_diagnostic(path: Path, band: float = 0.10) -> Dict[str, float]:
    """How much request mass sits within +/-`band` of each SLO threshold (gate #6)."""
    requests, _ = load_bench_serving_rounds(path)
    n = len(requests) or 1
    ttfts = [r.ttft_ms for r in requests]
    p95s = [percentile(r.token_itl_ms, 0.95) for r in requests]
    means = [statistics.fmean(r.token_itl_ms) if r.token_itl_ms else math.inf
             for r in requests]

    def in_band(values, threshold):
        lo, hi = threshold * (1 - band), threshold * (1 + band)
        return sum(lo <= v <= hi for v in values) / n

    return {
        "ttft_mass_within_band_of_slo": in_band(ttfts, TTFT_SLO_MS),
        "itl_p95_mass_within_band_of_slo": in_band(p95s, ITL_SLO_MS),
        "itl_mean_mass_within_band_of_slo": in_band(means, ITL_SLO_MS),
        "ttft_median_over_slo": percentile(ttfts, 0.5) / TTFT_SLO_MS,
        "itl_p95_median_over_slo": percentile(p95s, 0.5) / ITL_SLO_MS,
        "band": band,
    }


def combination_assumption_check(
    runs: Mapping[Tuple[str, str, str], Mapping[str, float]],
    jobs: Mapping[str, Sequence[Tuple[str, str]]],
    phases: Sequence[str],
) -> Dict[str, object]:
    """Within one arm, compare P(TTFT ok) * P(ITL ok) against P(both ok).

    rev2: computed under BOTH predicates.  A claim about CONSENSUS 1-20 must use
    1-20's OWN predicate (legacy mean-ITL); quoting the canonical-predicate number
    for it is a predicate misattribution.
    """
    out: Dict[str, object] = {}
    for phase in phases:
        rows = {}
        for arm in jobs:
            recs = [runs[(arm, r, phase)] for r, _ in jobs[arm]]
            row = {}
            for label, ikey, jkey in (
                ("legacy_mean_itl", "legacy_itl_mean_pass_frac", "legacy_joint_pass_frac"),
                ("canonical_p95", "itl_p95_pass_frac", "joint_pass_frac"),
            ):
                prod = statistics.fmean(x["ttft_pass_frac"] * x[ikey] for x in recs)
                joint = statistics.fmean(x[jkey] for x in recs)
                marginal = statistics.fmean(x[ikey] for x in recs)
                row[label] = {
                    "itl_marginal_percent": 100 * marginal,
                    "product_of_marginals_percent": 100 * prod,
                    "actual_joint_percent": 100 * joint,
                    "error_percent": (100 * (prod / joint - 1)) if joint else math.inf,
                    "identically_zero_error": marginal == 1.0,
                }
            rows[arm] = row
        out[phase] = rows
    out["note"] = (
        "`identically_zero_error` marks cells where the ITL marginal is exactly "
        "100%, so the product CANNOT differ from the joint -- those cells carry no "
        "information about the (x) assumption."
    )
    return out


def rank_law(rows: Sequence[Tuple[str, str, float, float]]) -> Dict[str, object]:
    """TTFT-pass ordering vs achieved-throughput ordering, per grid."""
    out = {}
    for grid in sorted({r[0] for r in rows}):
        sub = [r for r in rows if r[0] == grid]
        by_pass = [r[1] for r in sorted(sub, key=lambda r: -r[2])]
        by_thr = [r[1] for r in sorted(sub, key=lambda r: -r[3])]
        out[grid] = {
            "order_by_ttft_pass": by_pass,
            "order_by_achieved_throughput": by_thr,
            "orders_agree": by_pass == by_thr,
            "arms": {r[1]: {"ttft_pass_percent": r[2], "achieved_req_s": r[3]} for r in sub},
        }
    out["arms_in_agreement"] = sum(
        len(v["order_by_ttft_pass"]) for k, v in out.items()
        if isinstance(v, dict) and v.get("orders_agree")
    )
    out["arms_total"] = sum(
        len(v["order_by_ttft_pass"]) for k, v in out.items() if isinstance(v, dict)
    )
    return out


# --------------------------------------------------------------------------
# R1
# --------------------------------------------------------------------------


def run_r1(registry: ControlRegistry) -> Dict[str, object]:
    runs: Dict[Tuple[str, str, str], Dict[str, float]] = {}
    provenance = []
    for arm, reps in HE2_JOBS.items():
        for rep, job in reps:
            for phase in ("A", "B"):
                runs[(arm, rep, phase)] = score_run(
                    he2_path(phase, arm, rep, job), TTFT_SLO_MS, ITL_SLO_MS)
            provenance.append({"arm": arm, "rep": rep, "job": job,
                               "phase_A_and_B_share_boot": True,
                               **slurm_provenance("he2", job)})

    def arm_mean(arm: str, phase: str, key: str) -> float:
        return statistics.fmean(runs[(arm, r, phase)][key] for r, _ in HE2_JOBS[arm])

    # ---- PC1: CONSENSUS 1-19 per-phase legacy goodput vectors -------------
    for arm, tA, tB in [("d16", 2.03, 0.28), ("d24", 2.73, 1.01),
                        ("d34", 1.86, 1.09), ("d44", 1.23, 0.74)]:
        registry.add(name=f"PC1 he2 phase A legacy goodput {arm}", target=tA,
                     observed=arm_mean(arm, "A", "legacy_mean_itl_goodput_req_s"),
                     tolerance=0.005, key="legacy_mean_itl_goodput_req_s",
                     source="CONSENSUS.md 1-19 / reports/plot_extreme_disjoint.py A_gp (2026-07-19)")
        registry.add(name=f"PC1 he2 phase B legacy goodput {arm}", target=tB,
                     observed=arm_mean(arm, "B", "legacy_mean_itl_goodput_req_s"),
                     tolerance=0.005, key="legacy_mean_itl_goodput_req_s",
                     source="CONSENSUS.md 1-19 / reports/plot_extreme_disjoint.py B_gp (2026-07-19)")
    registry.add(name="PC1 he2 reactive-bind unweighted combined", target=1.485,
                 observed=(arm_mean("bind", "A", "legacy_mean_itl_goodput_req_s")
                           + arm_mean("bind", "B", "legacy_mean_itl_goodput_req_s")) / 2,
                 tolerance=0.005, key="legacy_mean_itl_goodput_req_s",
                 source="reports/plot_extreme_disjoint.py bind_comb (2026-07-19)")
    # ---- PC2: CONSENSUS 1-20 phase-A marginals ---------------------------
    for arm, tp, ip, both in [("d16", 57.8, 54.2, 36.7), ("d24", 49.7, 100.0, 49.7),
                              ("d34", 37.5, 100.0, 37.5), ("d44", 28.1, 100.0, 28.1)]:
        registry.add(name=f"PC2 he2 A TTFT-pass% {arm}", target=tp,
                     observed=100 * arm_mean(arm, "A", "ttft_pass_frac"),
                     tolerance=0.06, key="ttft_pass_frac",
                     source="CONSENSUS.md 1-20 / reports/plot_oracle_corrected.py tp")
        registry.add(name=f"PC2 he2 A ITL-pass%(legacy mean) {arm}", target=ip,
                     observed=100 * arm_mean(arm, "A", "legacy_itl_mean_pass_frac"),
                     tolerance=0.06, key="legacy_itl_mean_pass_frac",
                     source="CONSENSUS.md 1-20 / reports/plot_oracle_corrected.py ip")
        registry.add(name=f"PC2 he2 A BOTH%(legacy) {arm}", target=both,
                     observed=100 * arm_mean(arm, "A", "legacy_joint_pass_frac"),
                     tolerance=0.06, key="legacy_joint_pass_frac",
                     source="CONSENSUS.md 1-20 / reports/plot_oracle_corrected.py both")

    # ---- (A) pipeline / denominator --------------------------------------
    denominators = {
        f"{arm}_{rep}_{phase}": {"summed_duration_s": runs[(arm, rep, phase)]["duration_s"],
                                 "requests": runs[(arm, rep, phase)]["requests"]}
        for arm in HE2_JOBS for rep, _ in HE2_JOBS[arm] for phase in ("A", "B")}

    # ---- (B) aggregation unit --------------------------------------------
    aggregation = {}
    for rep in ("rep81", "rep82"):
        gp = {a: {p: runs[(a, rep, p)] for p in ("A", "B")} for a in HE2_STATICS + ["bind"]}
        unw = {a: (gp[a]["A"]["legacy_mean_itl_goodput_req_s"]
                   + gp[a]["B"]["legacy_mean_itl_goodput_req_s"]) / 2 for a in gp}
        pooled = {a: (gp[a]["A"]["legacy_good"] + gp[a]["B"]["legacy_good"])
                     / (gp[a]["A"]["duration_s"] + gp[a]["B"]["duration_s"]) for a in gp}
        best_A = max(HE2_STATICS, key=lambda a: gp[a]["A"]["legacy_mean_itl_goodput_req_s"])
        best_B = max(HE2_STATICS, key=lambda a: gp[a]["B"]["legacy_mean_itl_goodput_req_s"])
        best_unw = max(HE2_STATICS, key=lambda a: unw[a])
        best_pool = max(HE2_STATICS, key=lambda a: pooled[a])
        orc_unw = (gp[best_A]["A"]["legacy_mean_itl_goodput_req_s"]
                   + gp[best_B]["B"]["legacy_mean_itl_goodput_req_s"]) / 2
        orc_pool = ((gp[best_A]["A"]["legacy_good"] + gp[best_B]["B"]["legacy_good"])
                    / (gp[best_A]["A"]["duration_s"] + gp[best_B]["B"]["duration_s"]))
        aggregation[rep] = {
            "per_phase_optimum": {"A": best_A, "B": best_B},
            "unweighted_per_phase_mean": {
                "best_static": best_unw, "best_static_value": unw[best_unw],
                "oracle": orc_unw,
                "oracle_effect_percent": 100 * (orc_unw / unw[best_unw] - 1),
                "reactive_bind": unw["bind"],
                "bind_effect_percent": 100 * (unw["bind"] / unw[best_unw] - 1)},
            "pooled_trace_level": {
                "best_static": best_pool, "best_static_value": pooled[best_pool],
                "oracle": orc_pool,
                "oracle_effect_percent": 100 * (orc_pool / pooled[best_pool] - 1),
                "reactive_bind": pooled["bind"],
                "bind_effect_percent": 100 * (pooled["bind"] / pooled[best_pool] - 1)},
        }

    # ---- (C) canonical predicate -----------------------------------------
    canonical_rescore = {
        phase: {arm: {
            "ttft_pass_percent": [100 * runs[(arm, r, phase)]["ttft_pass_frac"] for r, _ in HE2_JOBS[arm]],
            "itl_mean_pass_percent": [100 * runs[(arm, r, phase)]["legacy_itl_mean_pass_frac"] for r, _ in HE2_JOBS[arm]],
            "itl_p95_pass_percent": [100 * runs[(arm, r, phase)]["itl_p95_pass_frac"] for r, _ in HE2_JOBS[arm]],
            "joint_pass_percent": [100 * runs[(arm, r, phase)]["joint_pass_frac"] for r, _ in HE2_JOBS[arm]],
            "slo_goodput_req_s": [runs[(arm, r, phase)]["slo_goodput_req_s"] for r, _ in HE2_JOBS[arm]],
        } for arm in HE2_JOBS} for phase in ("A", "B")}

    # ---- (D) 1-20 under both predicates ----------------------------------
    tp_A = {a: 100 * arm_mean(a, "A", "ttft_pass_frac") for a in HE2_STATICS}
    ip_legacy_A = {a: 100 * arm_mean(a, "A", "legacy_itl_mean_pass_frac") for a in HE2_STATICS}
    ip_p95_A = {a: 100 * arm_mean(a, "A", "itl_p95_pass_frac") for a in HE2_STATICS}
    both_legacy_A = {a: 100 * arm_mean(a, "A", "legacy_joint_pass_frac") for a in HE2_STATICS}
    both_p95_A = {a: 100 * arm_mean(a, "A", "joint_pass_frac") for a in HE2_STATICS}
    dt = max(tp_A, key=tp_A.get)
    di_legacy = max(ip_legacy_A, key=ip_legacy_A.get)
    di_p95 = max(ip_p95_A, key=ip_p95_A.get)
    ties_legacy = [a for a in ip_legacy_A if ip_legacy_A[a] == ip_legacy_A[di_legacy]]
    ties_p95 = [a for a in ip_p95_A if ip_p95_A[a] == ip_p95_A[di_p95]]
    decomposition = {
        "legacy_predicate": {
            "ttft_donor": dt, "itl_donor": di_legacy, "itl_donor_ties": ties_legacy,
            "oracle_pass_percent": tp_A[dt] * ip_legacy_A[di_legacy] / 100,
            "best_static": max(both_legacy_A, key=both_legacy_A.get),
            "best_static_pass_percent": max(both_legacy_A.values()),
            "effect_percent": 100 * (tp_A[dt] * ip_legacy_A[di_legacy] / 100
                                     / max(both_legacy_A.values()) - 1),
            "sm_demand": (GPU_SM_TOTAL - DECODE_SM[dt]) + DECODE_SM[di_legacy],
            "sm_demand_over_tie_set": sorted(
                (GPU_SM_TOTAL - DECODE_SM[dt]) + DECODE_SM[a] for a in ties_legacy)},
        "canonical_predicate": {
            "ttft_donor": dt, "itl_donor": di_p95, "itl_donor_ties": ties_p95,
            "oracle_pass_percent": tp_A[dt] * ip_p95_A[di_p95] / 100,
            "best_static": max(both_p95_A, key=both_p95_A.get),
            "best_static_pass_percent": max(both_p95_A.values()),
            "effect_percent": 100 * (tp_A[dt] * ip_p95_A[di_p95] / 100
                                     / max(both_p95_A.values()) - 1),
            "sm_demand": (GPU_SM_TOTAL - DECODE_SM[dt]) + DECODE_SM[di_p95],
            "sm_demand_over_tie_set": sorted(
                (GPU_SM_TOTAL - DECODE_SM[dt]) + DECODE_SM[a] for a in ties_p95)},
        "marginals_percent": {
            "ttft_pass": tp_A, "itl_mean_pass": ip_legacy_A, "itl_p95_pass": ip_p95_A,
            "joint_legacy": both_legacy_A, "joint_canonical": both_p95_A},
        "sm_identity_note":
            "sum = (108 - D_ttft) + D_itl > 108  <=>  D_itl > D_ttft. Here the TTFT "
            "argmax is d16 (D=16), the LEAST decode-heavy arm, so any ITL donor with "
            "D>16 forces >108 algebraically. The 116 is therefore a restatement of "
            "'the TTFT argmax sits at the prefill-heavy end', not an independent finding.",
    }

    # ---- (D') the +16% under a +/-10% TTFT threshold ladder ---------------
    ttft_ladder = {}
    for k in range(0, 41):
        T = 2700.0 + 15.0 * k
        m = {a: [score_run(he2_path("A", a, r, j), T, ITL_SLO_MS) for r, j in HE2_JOBS[a]]
             for a in HE2_STATICS}
        t = {a: statistics.fmean(x["ttft_pass_frac"] for x in m[a]) for a in m}
        i = {a: statistics.fmean(x["legacy_itl_mean_pass_frac"] for x in m[a]) for a in m}
        j_ = {a: statistics.fmean(x["legacy_joint_pass_frac"] for x in m[a]) for a in m}
        a_t, a_i, a_j = max(t, key=t.get), max(i, key=i.get), max(j_, key=j_.get)
        ttft_ladder[f"{int(T)}ms"] = {
            "ttft_donor": a_t, "itl_donor": a_i, "best_static": a_j,
            "sm_demand": (GPU_SM_TOTAL - DECODE_SM[a_t]) + DECODE_SM[a_i],
            "effect_percent": 100 * (t[a_t] * i[a_i] / j_[a_j] - 1)}
    eff = [v["effect_percent"] for v in ttft_ladder.values()]
    ttft_ladder["_summary"] = {
        "grid": "TTFT SLO 2700-3300 ms (=3000 +/-10%) in 15 ms steps, ITL 60 ms, legacy predicate",
        "effect_percent_min": min(eff), "effect_percent_max": max(eff),
        "effect_percent_at_operating_point": ttft_ladder["3000ms"]["effect_percent"],
        "monotone": eff == sorted(eff),
        "structure_invariant_cells": sum(
            1 for k, v in ttft_ladder.items() if k != "_summary"
            and v["ttft_donor"] == "d16" and v["itl_donor"] == "d24" and v["sm_demand"] == 116),
        "structure_flip_cells": sorted(
            k for k, v in ttft_ladder.items() if k != "_summary"
            and not (v["ttft_donor"] == "d16" and v["itl_donor"] == "d24"))}

    # ---- (E) separation of the two marginals in phase B ------------------
    separation = {
        f"{arm}_{rep}": {
            "joint_pass_count_canonical": runs[(arm, rep, "B")]["joint_pass_frac"]
                                          * runs[(arm, rep, "B")]["requests"],
            "min_itl_p95_among_ttft_pass_ms": runs[(arm, rep, "B")]["min_itl_p95_among_ttft_pass_ms"],
            "min_ttft_among_itl_p95_pass_s": runs[(arm, rep, "B")]["min_ttft_among_itl_p95_pass_ms"] / 1000,
            "ttft_pass_percent": 100 * runs[(arm, rep, "B")]["ttft_pass_frac"],
            "itl_p95_pass_percent": 100 * runs[(arm, rep, "B")]["itl_p95_pass_frac"]}
        for arm in HE2_JOBS for rep, _ in HE2_JOBS[arm]}

    percentiles = {
        f"{arm}_{rep}_{phase}": {k: runs[(arm, rep, phase)][k]
                                 for k in ("ttft_p50_ms", "ttft_p95_ms", "ttft_p99_ms",
                                           "token_itl_p50_ms", "token_itl_p95_ms",
                                           "token_itl_p99_ms", "throughput_req_s")}
        for arm in HE2_JOBS for rep, _ in HE2_JOBS[arm] for phase in ("A", "B")}

    controller = {}
    for rep, job in HE2_JOBS["bind"]:
        log = HERE / f"he2srv_bind_{rep}_rA8B5_{job}.log"
        count = sum(("SLO-BIND" in line) or ("SLO-SCHED" in line)
                    for line in log.read_text(errors="ignore").splitlines()) if log.exists() else -1
        controller[f"bind_{rep}"] = {
            "split_transition_log_lines": count,
            "residency_and_dwell": "NOT COMPUTABLE -- he2_bench.sbatch collects no "
                                   "PDMUX_TELEMETRY_PATH runtime_snapshot stream, so "
                                   "analyze.controller_summary cannot be run"}

    offered = {"A": 8.0, "B": 5.0}
    overload = {phase: {
        "offered_req_s": offered[phase],
        "achieved_req_s_min": min(runs[(a, r, phase)]["throughput_req_s"]
                                  for a in HE2_JOBS for r, _ in HE2_JOBS[a]),
        "achieved_req_s_max": max(runs[(a, r, phase)]["throughput_req_s"]
                                  for a in HE2_JOBS for r, _ in HE2_JOBS[a])}
        for phase in ("A", "B")}
    for phase in overload:
        overload[phase]["overload_factor_range"] = [
            offered[phase] / overload[phase]["achieved_req_s_max"],
            offered[phase] / overload[phase]["achieved_req_s_min"]]

    return {
        "campaign": "he2 EXTREME mix-swing, jobs 860497-860518, rA8B5",
        "workload": {"phase_A": "random-ids in2048/o32 @ rate 8",
                     "phase_B": "random-ids in2048/o512 @ rate 5",
                     "requests_per_arm_phase": 192, "rounds_per_file": 3,
                     "prefill_decode_token_ratio_phase_A": 2047 / 32},
        "provenance": provenance,
        "A_pipeline": {"denominators": denominators,
                       "note": "duration = sum over 3 round records (gate #7). NB the "
                               "he2 harness's own inline scorer (he2_bench.sbatch:92) "
                               "still uses max(dur,d), so HE2_RESULT lines in he2_*.out "
                               "are 3x inflated and must not be quoted."},
        "B_aggregation": aggregation,
        "C_canonical_predicate_rescore": canonical_rescore,
        "D_decomposition": decomposition,
        "D_ttft_threshold_ladder": ttft_ladder,
        "E_separation_phase_B": separation,
        "E_combination_assumption": combination_assumption_check(runs, HE2_JOBS, ("A", "B")),
        "percentiles": percentiles,
        "overload": overload,
        "cliff_diagnostic": {f"{arm}_{rep}_{phase}": cliff_diagnostic(he2_path(phase, arm, rep, job))
                             for arm in HE2_JOBS for rep, job in HE2_JOBS[arm]
                             for phase in ("A", "B")},
        "controller": controller,
        "_rank_law_rows": [("he2_phaseA", a, tp_A[a],
                            statistics.fmean(runs[(a, r, "A")]["throughput_req_s"]
                                             for r, _ in HE2_JOBS[a]))
                           for a in HE2_STATICS],
    }


# --------------------------------------------------------------------------
# R2
# --------------------------------------------------------------------------


def _oracle_from(tp, ip, jp, arms, idx):
    t = {a: statistics.fmean(tp[a][k] for k in idx[a]) for a in arms}
    i = {a: statistics.fmean(ip[a][k] for k in idx[a]) for a in arms}
    j = {a: statistics.fmean(jp[a][k] for k in idx[a]) for a in arms}
    dt, di, dj = max(t, key=t.get), max(i, key=i.get), max(j, key=j.get)
    return {
        "ttft_donor": dt, "itl_donor": di, "best_static": dj,
        "sm_demand": (GPU_SM_TOTAL - DECODE_SM[dt]) + DECODE_SM[di],
        "oracle_pass_percent": 100 * t[dt] * i[di],
        "best_static_pass_percent": 100 * j[dj],
        "effect_percent": 100 * (t[dt] * i[di] / j[dj] - 1),
    }


def run_r2(registry: ControlRegistry) -> Dict[str, object]:
    runs: Dict[Tuple[str, str, str], Dict[str, float]] = {}
    provenance = []
    for arm, reps in SGPTV_JOBS.items():
        for rep, job in reps:
            for phase in ("Lo", "Hi"):
                runs[(arm, rep, phase)] = score_run(
                    sgptv_path(phase, arm, rep, job), TTFT_SLO_MS, ITL_SLO_MS)
            provenance.append({"arm": arm, "rep": rep, "job": job,
                               "phases_share_boot": True, **slurm_provenance("sgptv", job)})

    def vec(arm: str, phase: str, key: str) -> List[float]:
        return [runs[(arm, r, phase)][key] for r, _ in SGPTV_JOBS[arm]]

    # ---- PC3: legacy pooled / per-phase ----------------------------------
    def pooled_legacy(arm: str) -> List[float]:
        return [(runs[(arm, r, "Lo")]["legacy_good"] + runs[(arm, r, "Hi")]["legacy_good"])
                / (runs[(arm, r, "Lo")]["duration_s"] + runs[(arm, r, "Hi")]["duration_s"])
                for r, _ in SGPTV_JOBS[arm]]

    for arm, target in [("d44", 3.220), ("d34", 3.171), ("slo", 2.964)]:
        registry.add(name=f"PC3 sgptv pooled legacy combined {arm}", target=target,
                     observed=statistics.fmean(pooled_legacy(arm)), tolerance=0.0015,
                     key="legacy_good",
                     source="CONSENSUS.md 1-7 / reports/bench_noise_root_cause.md:74-86 (2026-07-17)")
    for arm, lo, hi in [("d44", 2.858, 3.924), ("d34", 2.852, 3.785)]:
        registry.add(name=f"PC3 sgptv LO legacy goodput {arm}", target=lo,
                     observed=statistics.fmean(vec(arm, "Lo", "legacy_mean_itl_goodput_req_s")),
                     tolerance=0.0015, key="legacy_mean_itl_goodput_req_s",
                     source="reports/bench_noise_root_cause.md section 5 (2026-07-17)")
        registry.add(name=f"PC3 sgptv HI legacy goodput {arm}", target=hi,
                     observed=statistics.fmean(vec(arm, "Hi", "legacy_mean_itl_goodput_req_s")),
                     tolerance=0.0015, key="legacy_mean_itl_goodput_req_s",
                     source="reports/bench_noise_root_cause.md section 5 (2026-07-17)")

    # ---- PC4: canonical-predicate n=4 (1-13 footnotes E and F) -----------
    for arm, target, sd in [("d16", 2.831, 0.038), ("d24", 2.761, 0.140),
                            ("d34", 2.835, 0.025), ("d44", 2.848, 0.007)]:
        m = mean_sd(vec(arm, "Lo", "slo_goodput_req_s"))
        registry.add(name=f"PC4 sgptv LO canonical goodput {arm} (mean)", target=target,
                     observed=m["mean"], tolerance=0.0015, key="slo_goodput_req_s",
                     source="CONSENSUS.md 1-13 footnote E (2026-08-05)")
        registry.add(name=f"PC4 sgptv LO canonical goodput {arm} (SD)", target=sd,
                     observed=m["sd"], tolerance=0.0015, key="slo_goodput_req_s",
                     source="CONSENSUS.md 1-13 footnote E (2026-08-05)")
    for arm, target, sd in [("d16", 0.443, 0.016), ("d34", 3.405, 0.478), ("d44", 3.613, 0.571)]:
        m = mean_sd(vec(arm, "Hi", "slo_goodput_req_s"))
        registry.add(name=f"PC4 sgptv HI canonical goodput {arm} (mean)", target=target,
                     observed=m["mean"], tolerance=0.0015, key="slo_goodput_req_s",
                     source="CONSENSUS.md 1-13 footnote F (2026-08-05)")
        registry.add(name=f"PC4 sgptv HI canonical goodput {arm} (SD)", target=sd,
                     observed=m["sd"], tolerance=0.0015, key="slo_goodput_req_s",
                     source="CONSENSUS.md 1-13 footnote F (2026-08-05)")
    registry.add(name="PC4 sgptv HI legacy goodput d16", target=2.859,
                 observed=statistics.fmean(vec("d16", "Hi", "legacy_mean_itl_goodput_req_s")),
                 tolerance=0.0015, key="legacy_mean_itl_goodput_req_s",
                 source="CONSENSUS.md 1-13 footnote F (2026-08-05)")

    # ---- PC5: per-request ITL p95 machinery ------------------------------
    for arm, p90, ttft50 in [("d16", 53.36, 73.56), ("d44", 46.54, 79.61)]:
        registry.add(name=f"PC5 sgptv LO per-request ITL p95 -> p90 {arm}", target=p90,
                     observed=statistics.fmean(vec(arm, "Lo", "req_itl_p95_p90_ms")),
                     tolerance=0.02, key="req_itl_p95_p90_ms",
                     source="CONSENSUS.md 1-13 footnote (2026-08-05, ceiling-censoring)")
        registry.add(name=f"PC5 sgptv LO TTFT p50 {arm}", target=ttft50,
                     observed=statistics.fmean(vec(arm, "Lo", "ttft_p50_ms")),
                     tolerance=0.02, key="ttft_p50_ms",
                     source="CONSENSUS.md 1-13 footnote (2026-08-05, ceiling-censoring)")

    # ---- PC6: NOT A CONTROL -- proven identity with PC4 -------------------
    identity_proof = []
    for arm in ("d16", "slo"):
        for rep, _ in SGPTV_JOBS[arm]:
            r = runs[(arm, rep, "Lo")]
            identity_proof.append({
                "cell": f"{arm}_{rep}_LO",
                "ttft_pass_frac": r["ttft_pass_frac"],
                "good": r["good"], "itl_p95_pass_count": r["itl_p95_pass_count"],
                "bit_identical": r["good"] == r["itl_p95_pass_count"]})
    assert all(x["bit_identical"] for x in identity_proof), "PC6 identity proof failed"
    marginal_goodput = [runs[("d16", r, "Lo")]["itl_p95_pass_frac"]
                        * runs[("d16", r, "Lo")]["requests"]
                        / runs[("d16", r, "Lo")]["duration_s"] for r, _ in SGPTV_JOBS["d16"]]
    registry.add(name="PC6 [IDENTITY, NOT INDEPENDENT] sgptv LO ITL-p95 marginal d16",
                 target=2.831, observed=statistics.fmean(marginal_goodput),
                 tolerance=0.0015, key="itl_p95_pass_frac", independent=False,
                 source="CONSENSUS.md 1-13 footnote E -- but see `identity_proof`: in "
                        "these cells ttft_pass_frac == 1.000 exactly and score_run "
                        "computes `good` and `itl_p95_pass_count` from the SAME "
                        "subexpression, so the two are bit-identical in 8/8 reps. "
                        "PC4 passing makes PC6 incapable of failing. ZERO evidence.")

    # ---- headline: DECOUPLED oracle, canonical predicate, n=4 ------------
    phases = {"Lo": "LO (rate 3)", "Hi": "HI (rate 12)"}
    per_phase: Dict[str, object] = {}
    for phase in phases:
        tp = {a: vec(a, phase, "ttft_pass_frac") for a in SGPTV_STATICS}
        ip = {a: vec(a, phase, "itl_p95_pass_frac") for a in SGPTV_STATICS}
        jp = {a: vec(a, phase, "joint_pass_frac") for a in SGPTV_STATICS}
        full = _oracle_from(tp, ip, jp, SGPTV_STATICS, {a: [0, 1, 2, 3] for a in SGPTV_STATICS})

        rng = random.Random(BOOT_SEED)
        effects, sm_sums, don_t, don_i = [], [], {}, {}
        for _ in range(BOOT_SAMPLES):
            idx = {a: [rng.randrange(4) for _ in range(4)] for a in SGPTV_STATICS}
            d = _oracle_from(tp, ip, jp, SGPTV_STATICS, idx)
            effects.append(d["effect_percent"]); sm_sums.append(d["sm_demand"])
            don_t[d["ttft_donor"]] = don_t.get(d["ttft_donor"], 0) + 1
            don_i[d["itl_donor"]] = don_i.get(d["itl_donor"], 0) + 1

        jackknife = {f"drop_rep_index_{k}": _oracle_from(
            tp, ip, jp, SGPTV_STATICS,
            {a: [x for x in range(4) if x != k] for a in SGPTV_STATICS}) for k in range(4)}
        matched = _oracle_from(tp, ip, jp, list(MATCHED_SUBSET), MATCHED_SUBSET)
        matched["subset"] = "gpu38 / 2026-07-17 only (d34 x4, d44 x3); d16/d24 have no "
        matched["subset"] += "gpu38 reps, so they are absent from this comparison"

        paired = None
        if full["ttft_donor"] == full["itl_donor"] == full["best_static"]:
            a = full["ttft_donor"]
            paired = mean_sd([100 * (tp[a][k] * ip[a][k] / jp[a][k] - 1) for k in range(4)])

        per_phase[phase] = {
            "label": phases[phase],
            "arms": {a: {
                "ttft_pass_percent": mean_sd([100 * v for v in tp[a]]),
                "itl_p95_pass_percent": mean_sd([100 * v for v in ip[a]]),
                "joint_pass_percent": mean_sd([100 * v for v in jp[a]]),
                "slo_goodput_req_s": mean_sd(vec(a, phase, "slo_goodput_req_s")),
                "legacy_goodput_req_s": mean_sd(vec(a, phase, "legacy_mean_itl_goodput_req_s")),
                "duration_s": mean_sd(vec(a, phase, "duration_s")),
                "throughput_req_s": mean_sd(vec(a, phase, "throughput_req_s")),
                "ttft_p50_ms": mean_sd(vec(a, phase, "ttft_p50_ms")),
                "ttft_p95_ms": mean_sd(vec(a, phase, "ttft_p95_ms")),
                "ttft_p99_ms": mean_sd(vec(a, phase, "ttft_p99_ms")),
                "token_itl_p50_ms": mean_sd(vec(a, phase, "token_itl_p50_ms")),
                "token_itl_p95_ms": mean_sd(vec(a, phase, "token_itl_p95_ms")),
                "token_itl_p99_ms": mean_sd(vec(a, phase, "token_itl_p99_ms"))}
                for a in SGPTV_STATICS},
            "per_rep_argmax": {
                "ttft": [max(SGPTV_STATICS, key=lambda a: tp[a][k]) for k in range(4)],
                "itl_p95": [max(SGPTV_STATICS, key=lambda a: ip[a][k]) for k in range(4)],
                "joint": [max(SGPTV_STATICS, key=lambda a: jp[a][k]) for k in range(4)]},
            "decoupled_oracle": {
                **full,
                "donors_coincide_point_estimate": full["ttft_donor"] == full["itl_donor"],
                "effect_percent_boot_p2.5": percentile(effects, 0.025),
                "effect_percent_boot_p97.5": percentile(effects, 0.975),
                "effect_percent_paired_within_donor_arm": paired,
                "bootstrap_ttft_donor_tally": don_t,
                "bootstrap_itl_donor_tally": don_i,
                "bootstrap_sm_demand_tally": {str(s): sm_sums.count(s) for s in sorted(set(sm_sums))},
                "bootstrap_fraction_sm_over_108": sum(s > GPU_SM_TOTAL for s in sm_sums) / len(sm_sums),
                "jackknife": jackknife,
                "node_day_matched_subset": matched,
                "boot_samples": BOOT_SAMPLES, "boot_seed": BOOT_SEED},
        }

    # ---- the SM question is an identity, not a measurement ---------------
    hi = per_phase["Hi"]["decoupled_oracle"]
    lo = per_phase["Lo"]["decoupled_oracle"]
    sm_identity = {
        "algebra": "sum = (108 - D_ttft) + D_itl > 108  <=>  D_itl > D_ttft",
        "grid_max_decode_sm": max(DECODE_SM[a] for a in SGPTV_STATICS),
        "HI_ttft_donor": hi["ttft_donor"],
        "HI_ttft_donor_bootstrap_tally": hi["bootstrap_ttft_donor_tally"],
        "verdict": (
            "ZERO POWER in HI. The TTFT argmax is d44 = the most decode-heavy arm "
            "in the MEASURED grid (10000/10000 bootstrap draws), so D_itl <= 44 = "
            "D_ttft holds with probability 1 by construction and 'sum <= 108' cannot "
            "fail. 'HI never exceeds 108' is a restatement of 'the TTFT argmax sits "
            "at the decode-heavy grid boundary', NOT evidence that the 1-20 coupling "
            "tax is absent. Deciding that needs arms beyond d44 (d54/d64), which this "
            "campaign never ran."),
        "LO_note": (
            f"LO is NOT even nominally consistent with a 108 point estimate: the ITL "
            f"argmax is pure ceiling noise (one different arm per rep) and "
            f"{100 * lo['bootstrap_fraction_sm_over_108']:.1f}% of bootstrap draws exceed 108. "
            f"LO supports no statement about the SM sum in either direction."),
        "already_known": ("This identity is written down in "
                          "reports/PRIZE_SIZE_ARGUMENT_2026-08-16.md section 2.3(2). "
                          "rev1 of this analysis failed to apply a diagnostic the "
                          "repository already owned -- methodology gate #18."),
    }

    # ---- arm comparisons: REAL calls into unpaired_bootstrap_ci ----------
    comparisons = {}
    for phase in ("Lo", "Hi"):
        arms = per_phase[phase]["arms"]
        for key in ("slo_goodput_req_s", "ttft_pass_percent", "itl_p95_pass_percent",
                    "joint_pass_percent"):
            for base, prop in (("d34", "d44"), ("d24", "d44"), ("d16", "d44")):
                comparisons[f"{phase}_{key}_{prop}_vs_{base}"] = arm_comparison(
                    arms[base][key]["values"], arms[prop][key]["values"],
                    f"{phase} {key}: {prop} vs {base}")

    pooled_canon = {a: [(runs[(a, r, "Lo")]["good"] + runs[(a, r, "Hi")]["good"])
                        / (runs[(a, r, "Lo")]["duration_s"] + runs[(a, r, "Hi")]["duration_s"])
                        for r, _ in SGPTV_JOBS[a]] for a in SGPTV_STATICS}
    best_pool_arm = max(SGPTV_STATICS, key=lambda a: statistics.fmean(pooled_canon[a]))
    comparisons["pooled_canonical_d44_vs_d34"] = arm_comparison(
        pooled_canon["d34"], pooled_canon["d44"], "pooled canonical: d44 vs d34")
    n_req = statistics.fmean(vec("d44", "Lo", "requests"))
    lo_orc = lo["oracle_pass_percent"] / 100
    hi_orc = hi["oracle_pass_percent"] / 100
    lo_dur = statistics.fmean(vec(lo["ttft_donor"], "Lo", "duration_s"))
    hi_dur = statistics.fmean(vec(hi["ttft_donor"], "Hi", "duration_s"))
    pooled_oracle = (n_req * lo_orc + n_req * hi_orc) / (lo_dur + hi_dur)
    pooled_best = statistics.fmean(pooled_canon[best_pool_arm])

    # ---- threshold ladders (diagnostics, outside the control set) --------
    ladder = {}
    for t_slo, i_slo in [(2000., 60.), (3000., 40.), (3000., 50.), (3000., 55.),
                         (3000., 60.), (3000., 70.), (3000., 80.), (5000., 60.), (10000., 60.)]:
        row = {}
        for phase in ("Lo", "Hi"):
            sc = {a: [score_run(sgptv_path(phase, a, r, j), t_slo, i_slo)
                      for r, j in SGPTV_JOBS[a]] for a in SGPTV_STATICS}
            t = {a: statistics.fmean(x["ttft_pass_frac"] for x in sc[a]) for a in SGPTV_STATICS}
            i = {a: statistics.fmean(x["itl_p95_pass_frac"] for x in sc[a]) for a in SGPTV_STATICS}
            j_ = {a: statistics.fmean(x["joint_pass_frac"] for x in sc[a]) for a in SGPTV_STATICS}
            dt, di, dj = max(t, key=t.get), max(i, key=i.get), max(j_, key=j_.get)
            row[phase] = {"ttft_donor": dt, "itl_donor": di, "best_static": dj,
                          "sm_demand": (GPU_SM_TOTAL - DECODE_SM[dt]) + DECODE_SM[di],
                          "effect_percent": 100 * (t[dt] * i[di] / j_[dj] - 1),
                          "best_static_joint_pass_percent": 100 * j_[dj],
                          "itl_marginal_percent": {a: 100 * i[a] for a in SGPTV_STATICS},
                          "ttft_marginal_percent": {a: 100 * t[a] for a in SGPTV_STATICS}}
        ladder[f"ttft{int(t_slo)}ms_itl{int(i_slo)}ms"] = row
    ttft_argmax_ladder = {}
    for t_slo in (1000., 2000., 3000., 4000., 5000., 6000., 7000.):
        t = {a: statistics.fmean(score_run(sgptv_path("Hi", a, r, j), t_slo, ITL_SLO_MS)["ttft_pass_frac"]
                                 for r, j in SGPTV_JOBS[a]) for a in SGPTV_STATICS}
        ttft_argmax_ladder[f"{int(t_slo)}ms"] = {
            "argmax": max(t, key=t.get),
            "ttft_pass_percent": {a: 100 * t[a] for a in SGPTV_STATICS}}

    offered = {"Lo": 3.0, "Hi": 12.0}
    overload = {}
    for phase in ("Lo", "Hi"):
        thr = [runs[(a, r, phase)]["throughput_req_s"] for a in SGPTV_JOBS
               for r, _ in SGPTV_JOBS[a]]
        overload[phase] = {"offered_req_s": offered[phase],
                           "achieved_req_s_min": min(thr), "achieved_req_s_max": max(thr),
                           "overload_factor_range": [offered[phase] / max(thr),
                                                     offered[phase] / min(thr)]}

    return {
        "campaign": "sgptv rate-swing ShareGPT (LO rate 3 / HI rate 12), n=4 per arm",
        "workload": {"dataset": "ShareGPT V3, sharegpt-context-len 4000",
                     "requests_per_phase_run": 600, "rounds_per_file": 3,
                     "identical_across_all_runs": True,
                     "prefill_decode_token_ratio": 68276 / 47376},
        "provenance": provenance,
        "identity_proof_pc6": identity_proof,
        "per_phase": per_phase,
        "sm_demand_is_an_identity": sm_identity,
        "arm_comparisons_unpaired_bootstrap": comparisons,
        "pooled_trace_level": {
            "aggregation": "gpC = (good_LO + good_HI) / (dur_LO + dur_HI)",
            "best_static": best_pool_arm,
            "best_static_canonical_goodput": mean_sd(pooled_canon[best_pool_arm]),
            "all_arms": {a: mean_sd(pooled_canon[a]) for a in SGPTV_STATICS},
            "decoupled_oracle_goodput_req_s": pooled_oracle,
            "effect_percent": 100 * (pooled_oracle / pooled_best - 1),
            "donor_note": {"LO": {"ttft": lo["ttft_donor"], "itl": lo["itl_donor"]},
                           "HI": {"ttft": hi["ttft_donor"], "itl": hi["itl_donor"]},
                           "duration_basis": "donor arm's own summed duration per phase"}},
        "threshold_ladder_statics_only": ladder,
        "hi_ttft_argmax_ladder": ttft_argmax_ladder,
        "cliff_diagnostic": {f"{arm}_{rep}_{phase}": cliff_diagnostic(sgptv_path(phase, arm, rep, job))
                             for arm in SGPTV_STATICS for rep, job in SGPTV_JOBS[arm]
                             for phase in ("Lo", "Hi")},
        "combination_assumption": combination_assumption_check(
            runs, {a: SGPTV_JOBS[a] for a in SGPTV_STATICS}, ("Lo", "Hi")),
        "overload": overload,
        "controller": {f"slo_{rep}": {
            "split_transition_log_lines": _switch_count(job),
            "residency_and_dwell": "NOT COMPUTABLE -- sharegpt_vary_bench.sbatch "
                                   "records no runtime_snapshot telemetry stream"}
            for rep, job in SGPTV_JOBS["slo"]},
        "_rank_law_rows": [("sgptv_HI", a,
                            100 * statistics.fmean(vec(a, "Hi", "ttft_pass_frac")),
                            statistics.fmean(vec(a, "Hi", "throughput_req_s")))
                           for a in SGPTV_STATICS],
    }


def _switch_count(job: str) -> int:
    out = HERE / f"sgptv_{job}.out"
    if not out.exists():
        return -1
    for line in out.read_text(errors="ignore").splitlines():
        if line.startswith("SWITCHES"):
            return int(line.rsplit("=", 1)[-1])
    return -1


# --------------------------------------------------------------------------


def main() -> None:
    registry = ControlRegistry()
    r1 = run_r1(registry)
    r2 = run_r2(registry)
    coverage = registry.coverage_report()

    discriminant = {
        "overload_is_NOT_a_discriminant": {
            "he2_phase_A": r1["overload"]["A"]["overload_factor_range"],
            "he2_phase_B": r1["overload"]["B"]["overload_factor_range"],
            "sgptv_HI": r2["overload"]["Hi"]["overload_factor_range"],
            "note": "both grids are overloaded, so 'overload' cannot explain why the "
                    "TTFT argmax flips ends between them."},
        "rank_law": rank_law(list(r1["_rank_law_rows"]) + list(r2["_rank_law_rows"])),
        "prefill_decode_token_ratio": {
            "he2_phase_A": r1["workload"]["prefill_decode_token_ratio_phase_A"],
            "sgptv": r2["workload"]["prefill_decode_token_ratio"],
            "ratio_of_ratios": (r1["workload"]["prefill_decode_token_ratio_phase_A"]
                                / r2["workload"]["prefill_decode_token_ratio"]),
            "note": "proposed discriminant: which side of the split buys throughput. "
                    "The TTFT-pass ordering equals the achieved-throughput ordering in "
                    "every arm of both grids, and the throughput-maximising split "
                    "follows the per-request prefill:decode work ratio."},
        "scope_label_warning": (
            "Do NOT label the difference between these two grids a 'regime'. At least "
            "five axes differ simultaneously (dataset, output length, prefill:decode "
            "work ratio, arrival rate, node/date, n). The defensible label is "
            "'prefill:decode work-ratio dependence', and even that is a 2-point "
            "comparison."),
    }
    r1.pop("_rank_law_rows"); r2.pop("_rank_law_rows")

    payload = {
        "title": "R1/R2 oracle re-analysis (2026-08-16, rev2 post-audit) -- NOT CANONICAL",
        "generated_by": str(Path(__file__).resolve()),
        "slo": {"ttft_slo_ms": TTFT_SLO_MS, "itl_slo_ms": ITL_SLO_MS,
                "source": "he2_bench.sbatch:96 / sharegpt_vary_bench.sbatch:96 (`t<=3.0 and m<=0.06`)"},
        "predicate": {"canonical": "TTFT<=SLO AND per-request token-ITL p95<=SLO "
                                   "(analyze.RequestResult.passes)",
                      "legacy_secondary": "analyze.RequestResult.passes_legacy_mean"},
        "positive_control": {
            "all_passed": registry.all_passed(),
            "n_controls_total": len(registry.controls),
            **coverage,
            "controls": [c.as_dict() for c in registry.controls]},
        "cross_grid_discriminant": discriminant,
        "R1_he2": r1,
        "R2_sgptv": r2,
    }
    out = HERE / "oracle_reanalysis_2026-08-16.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=False, default=str) + "\n",
                   encoding="utf-8")

    failed = [c for c in registry.controls if not c.passed]
    print(f"controls: {coverage['n_independent_controls']} independent + "
          f"{coverage['n_identity_controls']} identity (0 evidence); {len(failed)} failed")
    for c in failed:
        print(f"  FAIL {c.name}: target {c.target} observed {c.observed:.4f}")
    for row in coverage["headline_paths"]:
        if row["state"] != "covered":
            print(f"  GAP  {row['path'][2]}: {row['state']}")
    print(f"reported-but-uncontrolled SLO settings: "
          f"{coverage['reported_but_uncontrolled_slo_settings']}")
    print(f"score_run invocations: {len(_CALL_LOG)}")
    print(f"wrote {out}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
