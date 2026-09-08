"""Canonical SLO-goodput and arm-comparison analysis library for R2 / PD-mux.

Scoring: the request-level SLO predicate (methodology gate #4 -- TTFT <= SLO AND
the request's OWN token-ITL p95 <= SLO) in ``RequestResult.passes``, aggregated
to per-cell goodput by ``summarize_requests``; controller residency/dwell by
``controller_summary``.

Intervals: the PRIMARY, verdict-bearing interval is the Student-t interval --
``paired_t_ci`` for matched repetitions, ``unpaired_t_ci`` (Welch) for unmatched
arms.  ``paired_bootstrap_ci`` / ``unpaired_bootstrap_ci`` are REPORTED
COMPANIONS ONLY: methodology gate #14 (``PROJECT_STATUS.md`` "방법론 게이트" #14,
2026-08-06) forbids reading their intervals as a verdict at n<=8, where their
measured coverage against a nominal 0.95 is n=4 0.798 / n=5 0.840 / n=6 0.859 /
n=8 0.888.  This module used to describe itself, in this very docstring, as a
bootstrap tool, which is why the order of precedence is spelled out here.

stdlib only, on purpose: there is no SciPy in the serving venv and no decision in
this project may depend on an unpinned import.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


@dataclass(frozen=True)
class RequestResult:
    request_id: str
    ttft_ms: float
    token_itl_ms: Tuple[float, ...]
    completion_s: float

    def passes(self, ttft_slo_ms: float, itl_slo_ms: float) -> bool:
        return (
            self.ttft_ms <= ttft_slo_ms
            and percentile(self.token_itl_ms, 0.95) <= itl_slo_ms
        )

    def passes_legacy_mean(self, ttft_slo_ms: float, itl_slo_ms: float) -> bool:
        return (
            self.ttft_ms <= ttft_slo_ms
            and statistics.fmean(self.token_itl_ms or (0.0,)) <= itl_slo_ms
        )


def summarize_requests(
    requests: Sequence[RequestResult],
    start_s: float,
    end_s: float,
    ttft_slo_ms: float,
    itl_slo_ms: float,
) -> Dict[str, float]:
    duration = end_s - start_s
    if duration <= 0:
        raise ValueError("benchmark duration must be positive")
    ttfts = [request.ttft_ms for request in requests]
    itls = [
        value for request in requests for value in request.token_itl_ms
    ]
    good = sum(request.passes(ttft_slo_ms, itl_slo_ms) for request in requests)
    legacy_good = sum(
        request.passes_legacy_mean(ttft_slo_ms, itl_slo_ms)
        for request in requests
    )
    return {
        "requests": float(len(requests)),
        "duration_s": duration,
        "throughput_req_s": len(requests) / duration,
        "slo_goodput_req_s": good / duration,
        "legacy_mean_itl_goodput_req_s": legacy_good / duration,
        "slo_violation_ratio": 1.0 - good / max(1, len(requests)),
        **{
            f"ttft_p{int(q * 100)}_ms": percentile(ttfts, q)
            for q in (0.50, 0.90, 0.95, 0.99)
        },
        **{
            f"token_itl_p{int(q * 100)}_ms": percentile(itls, q)
            for q in (0.50, 0.90, 0.95, 0.99)
        },
    }


def request_tpot_percentiles(
    requests: Sequence[RequestResult],
    quantiles: Sequence[float] = (0.50, 0.95, 0.99),
    undefined_as: float = math.inf,
) -> Dict[str, float]:
    """Percentiles of the PER-REQUEST TPOT (mean token-ITL), in ms.

    Distinct from ``summarize_requests``'s ``token_itl_p*`` keys, which pool all
    tokens of all requests.  The per-request aggregate is what sglang
    ``bench_serving`` reports as TPOT and what SLO definitions of the form
    "request TPOT <= X ms" score against, so campaigns that pre-register a
    mean-ITL SLO need this view as well as the pooled-token view.

    A request with no recorded inter-token latency has an undefined TPOT; it is
    mapped to ``undefined_as`` (default ``inf``, i.e. treated as a violation)
    rather than silently dropped, so the count of requests is preserved.
    """
    values = [
        statistics.fmean(request.token_itl_ms) if request.token_itl_ms else undefined_as
        for request in requests
    ]
    finite = [value for value in values if math.isfinite(value)]
    return {
        **{
            f"tpot_p{int(q * 100)}_ms": percentile(values, q) for q in quantiles
        },
        "tpot_undefined_requests": float(len(values) - len(finite)),
        "requests": float(len(values)),
    }


# ===========================================================================
# PRIMARY (verdict-bearing) intervals -- methodology gate #14
# ===========================================================================
# Gate #14 (``PROJECT_STATUS.md`` "방법론 게이트" #14, 2026-08-06, claims-auditor
# Gate-2 design audit x2 + independent result-analyst reproduction): at n<=8
# repetitions the percentile bootstrap interval of the mean may NOT decide a
# verdict.  The primary is the Student-t interval below; the bootstrap is
# reported alongside only.  Measured true coverage of the bootstrap against a
# nominal 0.95 (100k-trial MC, ``results/p1_gates/verify/verify_c1_coverage.py``):
#
#     n=4 0.798 | n=5 0.840 | n=6 0.859 | n=8 0.888
#
# (independently reproduced 2026-09-08 as .802/.838/.860/.882; the t interval
#  measures .949-.954 on the same trials).
#
# The incomplete-beta tail function and the bisection are PORTED (numerics
# unchanged) from the audited, self-tested implementation in
# ``workspace/engine-port/results/cp_baseline/d1_predicates.py``:
#     constants  d1_predicates.py:263-267
#     _betacf    d1_predicates.py:272
#     _betai     d1_predicates.py:307
#     paired_t_p d1_predicates.py:319
#     t_crit_for d1_predicates.py:333
# so that the project has ONE t implementation and the critical value can never
# disagree with the p-value.  ``tests/test_analyze_gate14.py`` asserts numeric
# agreement with that module.

#: Smallest number of repetitions at which a BOOTSTRAP interval may be read as a
#: verdict.  Gate #14 forbids n<=8, so the first eligible n is 9.
GATE14_DECISION_MIN_N = 9

#: Measured coverage of the percentile bootstrap of the mean, nominal 0.95.
GATE14_BOOTSTRAP_COVERAGE = {4: 0.798, 5: 0.840, 6: 0.859, 8: 0.888}


class Gate14SmallSampleWarning(UserWarning):
    """Emitted when a bootstrap interval is computed at n<=8 (gate #14).

    A warning, never an exception: the small-n bootstrap callers in this
    repository are the EVIDENCE GENERATORS for gate #14 itself and must keep
    running.  The refusal lives on the verdict path, not on the arithmetic.
    """


# Numerical-method tolerances for the incomplete beta / bisection.  These are
# NOT decision constants -- no verdict depends on their value beyond convergence
# -- but they are named so that no threshold hides inside a function body.
_BETACF_ITMAX = 300
_BETACF_EPS = 3e-14
_TINY = 1e-30
_TCRIT_HI = 10000.0
_TCRIT_ITERS = 200


def _betacf(a, b, x, itmax=_BETACF_ITMAX, eps=_BETACF_EPS):
    """Lentz continued fraction for the incomplete beta.  Standard; no SciPy in
    this environment and the decision must not depend on an unpinned import.
    Ported from ``results/cp_baseline/d1_predicates.py:272``."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < _TINY:
        d = _TINY
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        if abs(d) < _TINY:
            d = _TINY
        if abs(c) < _TINY:
            c = _TINY
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        if abs(d) < _TINY:
            d = _TINY
        if abs(c) < _TINY:
            c = _TINY
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


def _betai(a, b, x):
    """Regularised incomplete beta.  Ported from ``d1_predicates.py:307``."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
             + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lbeta) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbeta) * _betacf(b, a, 1.0 - x) / b


def _t_two_sided_p(t_stat, df):
    """Two-sided tail of Student-t at ``t_stat`` with (possibly fractional) ``df``.

    Same expression as the tail used inside ``t_crit_for`` and ``paired_t_p``;
    named separately so Welch's fractional df can reuse it."""
    return _betai(df / 2.0, 0.5, df / (df + t_stat * t_stat))


def paired_t_p(deltas):
    """Two-sided p of the paired t on the differences.
    Ported from ``d1_predicates.py:319``.  The CI remains the primary object;
    this is reported alongside (and is what any multiplicity correction would
    have to be applied to -- see gate #93)."""
    n = len(deltas)
    if n < 2:
        raise ValueError("a t test needs at least two paired boots")
    sd = statistics.stdev(deltas)
    if sd == 0.0:
        return 0.0 if statistics.fmean(deltas) != 0.0 else 1.0
    t = statistics.fmean(deltas) / (sd / math.sqrt(n))
    nu = n - 1
    return _betai(nu / 2.0, 0.5, nu / (nu + t * t))


def t_crit_for(alpha, df, hi=_TCRIT_HI):
    """Two-sided t critical value at `alpha`, by bisection on `paired_t_p`'s own
    tail function -- so the critical value and the p-value can never disagree.
    Ported from ``d1_predicates.py:333``.  ``df`` may be fractional (Welch)."""
    def tail(t):
        return _betai(df / 2.0, 0.5, df / (df + t * t))
    lo = 0.0
    for _ in range(_TCRIT_ITERS):
        mid = (lo + hi) / 2.0
        if tail(mid) > alpha:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def paired_t_ci(
    baseline: Mapping[str, float],
    proposed: Mapping[str, float],
    level: float = 0.95,
) -> Dict[str, float]:
    """★PRIMARY paired interval (methodology gate #14).  Student-t on the
    per-repetition differences.

    Deliberately the SAME two-Mapping signature and the SAME pairing convention
    as ``paired_bootstrap_ci`` -- pairs are ``sorted(set(baseline) &
    set(proposed))`` -- so the primary and the companion can never disagree about
    WHICH repetitions are matched, only about the width of the interval.

    ``ci95_low``/``ci95_high`` are named for interoperability with the bootstrap
    dicts and follow ``level`` (default 0.95, i.e. genuinely 95%); ``level`` is
    returned so a non-default coverage cannot be silently misread.

    This interval is honest at small n but not powerful, and it does not repeal
    the other gates: gate #3 still requires n>=4 before any policy conclusion and
    gate #2 still requires ``effect_percent >= 3.0`` for a headline.
    """
    pair_ids = sorted(set(baseline) & set(proposed))
    if len(pair_ids) < 2:
        raise ValueError("paired CI requires at least two matched repetitions")
    effects = [proposed[pair] - baseline[pair] for pair in pair_ids]
    n = len(effects)
    df = n - 1
    mean_effect = statistics.fmean(effects)
    sd = statistics.stdev(effects)
    standard_error = sd / math.sqrt(n)
    t_crit = t_crit_for(1.0 - level, df)
    # associate exactly as ``d1_predicates.paired_t_ci`` does (t*sd/sqrt(n), not
    # t*(sd/sqrt(n))): the two implementations must agree to the last bit, and
    # ``tests/test_analyze_gate14.py`` compares them with ``assertEqual``
    half_width = t_crit * sd / math.sqrt(n)
    baseline_mean = statistics.fmean(baseline[pair] for pair in pair_ids)
    if standard_error == 0.0:
        t_stat = math.copysign(math.inf, mean_effect) if mean_effect else 0.0
    else:
        t_stat = mean_effect / standard_error
    return {
        "pairs": float(n),
        "df": float(df),
        "level": float(level),
        "mean_effect": mean_effect,
        "median_effect": statistics.median(effects),
        "effect_percent": 100.0 * mean_effect / baseline_mean
        if baseline_mean
        else math.nan,
        "ci95_low": mean_effect - half_width,
        "ci95_high": mean_effect + half_width,
        "standard_deviation": sd,
        "standard_error": standard_error,
        "t_stat": t_stat,
        "t_crit": t_crit,
        "p_value": paired_t_p(effects),
        "baseline_mean": baseline_mean,
    }


def unpaired_t_ci(
    baseline: Sequence[float],
    proposed: Sequence[float],
    level: float = 0.95,
) -> Dict[str, float]:
    """★PRIMARY arm-vs-arm interval when repetitions are NOT matched: WELCH t.

    Same Sequence-of-values signature as ``unpaired_bootstrap_ci``.  Equal
    variances are NOT assumed and the pooled-variance t is not offered: arms in
    this project differ in variance by construction (a controller arm against a
    static arm, or two arms measured on different nodes), and pooling would
    understate the interval exactly where the difference matters.  ``df`` is
    Welch-Satterthwaite and is therefore fractional.

    Gate #14 companion coverage for reference: the unpaired percentile bootstrap
    measures 0.8556 at n=4/arm where Welch t measures 0.9590
    (``results/p1_gates/verify/verify_c1_unpaired.py``).
    """
    base = [float(value) for value in baseline]
    prop = [float(value) for value in proposed]
    if len(base) < 2 or len(prop) < 2:
        raise ValueError("unpaired CI requires at least two repetitions per arm")
    n_base, n_prop = len(base), len(prop)
    var_base = statistics.variance(base)
    var_prop = statistics.variance(prop)
    term_base = var_base / n_base
    term_prop = var_prop / n_prop
    standard_error = math.sqrt(term_base + term_prop)
    baseline_mean = statistics.fmean(base)
    proposed_mean = statistics.fmean(prop)
    mean_effect = proposed_mean - baseline_mean
    if standard_error == 0.0:
        # Both arms are constant: the difference is exact, the interval is a
        # point, and Welch's df is undefined -- fall back to the conservative
        # min(n)-1 for the reported critical value.
        df = float(min(n_base, n_prop) - 1)
        t_stat = math.copysign(math.inf, mean_effect) if mean_effect else 0.0
        p_value = 0.0 if mean_effect else 1.0
        half_width = 0.0
    else:
        df = (term_base + term_prop) ** 2 / (
            term_base * term_base / (n_base - 1)
            + term_prop * term_prop / (n_prop - 1)
        )
        t_stat = mean_effect / standard_error
        p_value = _t_two_sided_p(t_stat, df)
        half_width = t_crit_for(1.0 - level, df) * standard_error
    t_crit = t_crit_for(1.0 - level, df)
    return {
        "n_baseline": float(n_base),
        "n_proposed": float(n_prop),
        "df": df,
        "level": float(level),
        "baseline_mean": baseline_mean,
        "baseline_sd": statistics.stdev(base),
        "proposed_mean": proposed_mean,
        "proposed_sd": statistics.stdev(prop),
        "mean_effect": mean_effect,
        "effect_percent": 100.0 * mean_effect / baseline_mean
        if baseline_mean
        else math.nan,
        "ci95_low": mean_effect - half_width,
        "ci95_high": mean_effect + half_width,
        "standard_error": standard_error,
        "t_stat": t_stat,
        "t_crit": t_crit,
        "p_value": p_value,
    }


def _gate14_guard(n: int) -> Dict[str, object]:
    """Non-destructive verdict guard bolted onto the two bootstrap estimators.

    ADDS keys, never raises, never touches an existing value or the RNG.  It must
    not raise: ``tests/test_benchmark_tools.py`` exercises ``paired_bootstrap_ci``
    at n=3 and ``results/p1_gates/verify/verify_c1_coverage.py`` replays it at
    n=5 -- both are the evidence generators for gate #14 itself, so refusing to
    compute would destroy the gate's own reproduction path.  The refusal belongs
    on the verdict path (``main``), not on the arithmetic.
    """
    eligible = n >= GATE14_DECISION_MIN_N
    if eligible:
        note = (
            "methodology gate #14: n=%d >= %d, so this bootstrap interval is not "
            "refused on small-sample grounds; the t interval (paired_t_ci / "
            "unpaired_t_ci) is still the registered primary."
            % (n, GATE14_DECISION_MIN_N)
        )
    else:
        note = (
            "methodology gate #14 (PROJECT_STATUS.md, 2026-08-06): n=%d <= 8, so "
            "this percentile-bootstrap interval MAY NOT decide a verdict "
            "(measured coverage n=4 0.798 / n=5 0.840 / n=6 0.859 / n=8 0.888 "
            "against a nominal 0.95).  Use paired_t_ci / unpaired_t_ci as the "
            "primary and report this interval alongside only." % n
        )
        warnings.warn(note, Gate14SmallSampleWarning, stacklevel=3)
    return {"decision_eligible": eligible, "gate14_note": note}


# ===========================================================================
# COMPANION (reported-only) bootstrap intervals -- FROZEN NUMERICS
# ===========================================================================


def paired_bootstrap_ci(
    baseline: Mapping[str, float],
    proposed: Mapping[str, float],
    samples: int = 10000,
    seed: int = 1,
) -> Dict[str, object]:
    """COMPANION ONLY -- reported alongside, never verdict-bearing at n<=8.

    ★Methodology gate #14 (``PROJECT_STATUS.md`` #14, 2026-08-06): do NOT read
    this interval as a verdict at n<=8 repetitions.  ``paired_t_ci`` is the
    primary; report this one next to it.  Measured true coverage of this
    percentile bootstrap of the mean against a nominal 0.95:

        n=4 0.798 | n=5 0.840 | n=6 0.859 | n=8 0.888

    (100k-trial MC, ``results/p1_gates/verify/verify_c1_coverage.py``.  The cause
    is n itself -- not the frozen seed, not a normality assumption: releasing the
    seed leaves n=5 at 0.8397.)  The returned ``decision_eligible`` and
    ``gate14_note`` keys carry that refusal with its numbers; they are ADDITIONS
    and every pre-existing key is unchanged.

    ★FROZEN NUMERICS -- extend by adding keys, never by changing the arithmetic.
    ``seed=1``, ``samples=10000`` and the ``rng.choice`` draw ORDER are replayed
    bit-for-bit by ``verify_c1_coverage.py``, which recovers the 10000xN resample
    count matrix from ``random.Random(1)`` and validates it by requiring exact
    agreement with this function's own ``ci95_low``/``ci95_high``.  That replay is
    the evidence path for gate #14 itself.
    """
    pair_ids = sorted(set(baseline) & set(proposed))
    if len(pair_ids) < 2:
        raise ValueError("paired CI requires at least two matched repetitions")
    effects = [proposed[pair] - baseline[pair] for pair in pair_ids]
    rng = random.Random(seed)
    boot = [
        statistics.fmean(rng.choice(effects) for _ in effects)
        for _ in range(samples)
    ]
    mean_effect = statistics.fmean(effects)
    baseline_mean = statistics.fmean(baseline[pair] for pair in pair_ids)
    return {
        "pairs": float(len(pair_ids)),
        "mean_effect": mean_effect,
        "median_effect": statistics.median(effects),
        "effect_percent": 100.0 * mean_effect / baseline_mean
        if baseline_mean
        else math.nan,
        "ci95_low": percentile(boot, 0.025),
        "ci95_high": percentile(boot, 0.975),
        "standard_deviation": statistics.stdev(effects),
        **_gate14_guard(len(pair_ids)),
    }


def unpaired_bootstrap_ci(
    baseline: Sequence[float],
    proposed: Sequence[float],
    samples: int = 10000,
    seed: int = 1,
) -> Dict[str, object]:
    """COMPANION ONLY -- difference-of-means CI when arms have no natural pairing.

    ★Methodology gate #14 (``PROJECT_STATUS.md`` #14, 2026-08-06): do NOT read
    this interval as a verdict at n<=8 per arm.  ``unpaired_t_ci`` (Welch) is the
    primary; report this one next to it.  Measured coverage against a nominal
    0.95: this estimator 0.8556 at n=4/arm where Welch t measures 0.9590
    (``results/p1_gates/verify/verify_c1_unpaired.py``); the paired bootstrap
    ladder is n=4 0.798 / n=5 0.840 / n=6 0.859 / n=8 0.888.  The returned
    ``decision_eligible`` / ``gate14_note`` keys carry that refusal.

    PAIRING convention (this paragraph replaces the pre-2026-09-08 sentence that
    called ``paired_bootstrap_ci`` "preferred", which contradicted gate #14 by
    steering the reader to a bootstrap interval): use the PAIRED functions
    whenever repetitions are genuinely matched (same seed / trace slot), because
    pairing removes the trace-and-seed component of the variance.  That is a
    statement about PAIRING, not about the estimator -- for either pairing the
    primary estimator is the t interval.  Static-split arms from separate SLURM
    jobs have no such matching, so each arm is resampled independently with the
    same conventions (10000 samples, seed=1) rather than inventing an arbitrary
    pairing.

    ★FROZEN NUMERICS -- see ``paired_bootstrap_ci``; the seed, the sample count
    and the ``rng.choice`` draw order are replayed by ``verify_c1_unpaired.py``.
    """
    base = [float(value) for value in baseline]
    prop = [float(value) for value in proposed]
    if len(base) < 2 or len(prop) < 2:
        raise ValueError("unpaired CI requires at least two repetitions per arm")
    rng = random.Random(seed)
    boot = []
    for _ in range(samples):
        b = statistics.fmean(rng.choice(base) for _ in base)
        p = statistics.fmean(rng.choice(prop) for _ in prop)
        boot.append(p - b)
    baseline_mean = statistics.fmean(base)
    mean_effect = statistics.fmean(prop) - baseline_mean
    return {
        "n_baseline": float(len(base)),
        "n_proposed": float(len(prop)),
        "baseline_mean": baseline_mean,
        "baseline_sd": statistics.stdev(base),
        "proposed_mean": statistics.fmean(prop),
        "proposed_sd": statistics.stdev(prop),
        "mean_effect": mean_effect,
        "effect_percent": 100.0 * mean_effect / baseline_mean
        if baseline_mean
        else math.nan,
        "ci95_low": percentile(boot, 0.025),
        "ci95_high": percentile(boot, 0.975),
        **_gate14_guard(min(len(base), len(prop))),
    }


def load_bench_serving_rounds(
    path: Path,
    request_slice: Tuple[float, float] = (0.0, 1.0),
) -> Tuple[List[RequestResult], float]:
    """Load an sglang ``bench_serving --output-details --output-file`` artifact.

    The harness appends **one JSON object per round** to the same file, so the
    benchmark duration must be the SUM of per-round durations (methodology gate
    #7: ``max()`` inflates goodput by the round count).  ``request_slice`` keeps
    a contiguous fraction of each round's requests in launch order, for
    transient (warm-up / drain) sensitivity analysis.

    Returns ``(requests, summed_duration_s)``.  ``request_id`` encodes
    ``r<round>#<launch index>`` so that identical workloads across arms can be
    matched slot-by-slot.
    """
    low, high = request_slice
    requests: List[RequestResult] = []
    duration = 0.0
    with Path(path).open(encoding="utf-8") as handle:
        for round_index, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            duration += float(record.get("duration") or 0.0)
            ttfts = record.get("ttfts") or []
            itls = record.get("itls") or []
            start = int(round(low * len(ttfts)))
            stop = int(round(high * len(ttfts)))
            for index in range(start, stop):
                tokens = itls[index] if index < len(itls) else []
                requests.append(
                    RequestResult(
                        request_id=f"r{round_index}#{index}",
                        ttft_ms=1000.0 * float(ttfts[index]),
                        token_itl_ms=tuple(1000.0 * float(v) for v in tokens),
                        completion_s=math.nan,
                    )
                )
    return requests, duration


def controller_summary(events: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    """Compute time-weighted split residency and dwell from ordered telemetry."""
    snapshots = sorted(
        (
            event
            for event in events
            if event.get("event") == "runtime_snapshot"
            and event.get("phase") == "benchmark"
            and "decode_sms" in event
        ),
        key=lambda event: float(event["timestamp_monotonic_s"]),
    )
    residency: Dict[int, float] = {}
    dwell: List[float] = []
    transitions = 0
    if not snapshots:
        return {
            "split_transitions": 0,
            "residency_s": {},
            "dwell_s": [],
        }
    state = int(snapshots[0]["decode_sms"])
    state_started = float(snapshots[0]["timestamp_monotonic_s"])
    previous_time = state_started
    for event in snapshots[1:]:
        timestamp = float(event["timestamp_monotonic_s"])
        residency[state] = residency.get(state, 0.0) + max(0.0, timestamp - previous_time)
        new_state = int(event["decode_sms"])
        if new_state != state:
            transitions += 1
            dwell.append(max(0.0, timestamp - state_started))
            state = new_state
            state_started = timestamp
        previous_time = timestamp
    dwell.append(max(0.0, previous_time - state_started))
    total = sum(residency.values())
    return {
        "split_transitions": transitions,
        "residency_s": dict(sorted(residency.items())),
        "residency_fraction": {
            state: duration / total if total else 0.0
            for state, duration in sorted(residency.items())
        },
        "dwell_s": dwell,
    }


def select_static_baselines(
    rows: Sequence[Mapping[str, object]],
    metric: str = "slo_goodput_req_s",
) -> Dict[str, object]:
    """Select B1 globally and B2 per workload from training-sweep rows."""
    by_split: Dict[int, List[float]] = {}
    by_workload_split: Dict[str, Dict[int, List[float]]] = {}
    for row in rows:
        split = int(row["decode_sms"])
        workload = str(row["workload"])
        value = float(row[metric])
        by_split.setdefault(split, []).append(value)
        by_workload_split.setdefault(workload, {}).setdefault(split, []).append(value)
    if not by_split:
        raise ValueError("static sweep contains no rows")
    global_split = max(
        by_split,
        key=lambda split: (statistics.fmean(by_split[split]), -split),
    )
    per_workload = {
        workload: max(
            splits,
            key=lambda split: (statistics.fmean(splits[split]), -split),
        )
        for workload, splits in by_workload_split.items()
    }
    return {
        "global_decode_sms": global_split,
        "global_training_mean": statistics.fmean(by_split[global_split]),
        "per_workload_decode_sms": per_workload,
    }


def trace_aware_oracle(
    epoch_rewards: Sequence[Mapping[int, float]],
    states: Sequence[int] = (16, 24, 34, 44),
    transition_cost: float = 0.0,
    minimum_dwell_epochs: int = 1,
) -> Dict[str, object]:
    """Offline B8 upper bound with explicit switch cost and minimum dwell.

    ``epoch_rewards[t][d]`` is the SLO-goodput utility obtainable in epoch ``t``
    at decode allocation ``d``.  The oracle knows the complete future trace.
    """
    ordered_states = tuple(sorted(set(int(state) for state in states)))
    if not epoch_rewards or not ordered_states:
        raise ValueError("oracle needs epochs and states")
    dwell_required = max(1, int(minimum_dwell_epochs))
    # (state, capped dwell) -> (reward, schedule)
    frontier: Dict[Tuple[int, int], Tuple[float, List[int]]] = {}
    for state in ordered_states:
        if state in epoch_rewards[0]:
            frontier[(state, 1)] = (float(epoch_rewards[0][state]), [state])
    for rewards in epoch_rewards[1:]:
        next_frontier: Dict[Tuple[int, int], Tuple[float, List[int]]] = {}
        for (current, dwell), (total, schedule) in frontier.items():
            for target in ordered_states:
                if target not in rewards:
                    continue
                if target != current and dwell < dwell_required:
                    continue
                next_dwell = min(dwell_required, dwell + 1) if target == current else 1
                candidate = (
                    total
                    + float(rewards[target])
                    - (float(transition_cost) if target != current else 0.0)
                )
                key = (target, next_dwell)
                if key not in next_frontier or candidate > next_frontier[key][0]:
                    next_frontier[key] = (candidate, schedule + [target])
        if not next_frontier:
            raise ValueError("no feasible oracle path for supplied dwell/states")
        frontier = next_frontier
    reward, schedule = max(frontier.values(), key=lambda item: item[0])
    return {
        "total_reward": reward,
        "schedule": schedule,
        "transitions": sum(
            previous != current
            for previous, current in zip(schedule, schedule[1:])
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Paired arm comparison over a campaign summary.  PRIMARY estimator "
            "is the paired Student-t interval (methodology gate #14); the paired "
            "bootstrap is reported alongside under 'companion_bootstrap' and "
            "does not decide 'headline_improvement'."
        )
    )
    parser.add_argument("--paired-summary", type=Path, required=True)
    parser.add_argument("--baseline", default="B1")
    parser.add_argument("--proposed", default="B6")
    parser.add_argument("--metric", default="slo_goodput_req_s")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads(args.paired_summary.read_text(encoding="utf-8"))
    baseline = {
        row["pair_id"]: float(row[args.metric])
        for row in rows
        if row["baseline"] == args.baseline
    }
    proposed = {
        row["pair_id"]: float(row[args.metric])
        for row in rows
        if row["baseline"] == args.proposed
    }
    primary = paired_t_ci(baseline, proposed)
    companion = paired_bootstrap_ci(baseline, proposed)
    result = dict(primary)
    result.update(
        {
            "baseline": args.baseline,
            "proposed": args.proposed,
            "metric": args.metric,
            "primary_estimator": "paired_t_ci",
            "gate14_compliant": True,
            "companion_bootstrap": companion,
            # ⚠ CHANGED 2026-09-08 (methodology gate #14).  This used to be read
            # off the BOOTSTRAP interval, which the gate forbids at n<=8.  The
            # headline predicate itself is unchanged (gate #2: effect >= 3% AND
            # the lower confidence bound excludes 0) -- only the ESTIMATOR that
            # supplies ci95_low changed, from percentile bootstrap to paired t.
            "headline_improvement": (
                primary["effect_percent"] >= 3.0 and primary["ci95_low"] > 0.0
            ),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
