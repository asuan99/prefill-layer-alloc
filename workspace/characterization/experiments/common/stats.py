"""
stats.py — saturation-point estimation WITH a confidence interval.

The v2 G1 gate compares the SSM saturation SM across batch sizes; whether those
points are "the same" or "moving toward total_sm" must be decided on CIs, not
point estimates (prompt E2: "포화점은 점추정 금지 — 신뢰구간으로"). This module
provides the saturation criterion (same marginal-gain rule as
shared.sweep_spec.saturation_point) plus a bootstrap CI over the per-SM latency
repeats (n ≥ 30).

stdlib + numpy only (no torch) so it is importable for offline adjudication.
"""

from __future__ import annotations

import math
import random
from typing import Optional

import numpy as np

__all__ = ["saturation_sm", "saturation_sm_ci", "ci_overlap"]


def saturation_sm(
    sm_counts: list[int],
    latencies_ms: list[float],
    total_sm: int,
    threshold: float = 0.03,
) -> Optional[int]:
    """SM count at which marginal throughput gain per 10% SM falls below threshold.

    Mirrors shared.sweep_spec.saturation_point but operates on plain arrays so it
    can be bootstrapped. throughput = 1/latency, normalised to its own peak;
    marginal gain measured per 10% of total SM.

    Returns the saturating sm_count, the max sm_count if no saturation is seen,
    or None if < 2 usable points.
    """
    pts = [(s, l) for s, l in zip(sm_counts, latencies_ms)
           if l is not None and l > 0 and not math.isnan(l)]
    if len(pts) < 2:
        return None
    pts.sort(key=lambda p: p[0])
    sm = np.array([p[0] for p in pts], dtype=float)
    lat = np.array([p[1] for p in pts], dtype=float)
    ratio = sm / float(total_sm)
    tput = 1.0 / lat
    tput_max = tput.max()
    if tput_max <= 0:
        return int(sm[-1])
    tput_n = tput / tput_max
    for i in range(1, len(sm)):
        d_ratio = ratio[i] - ratio[i - 1]
        if d_ratio <= 0:
            continue
        gain_per_10pct = (tput_n[i] - tput_n[i - 1]) / d_ratio * 0.10
        if gain_per_10pct < threshold:
            return int(sm[i - 1])
    return int(sm[-1])


def saturation_sm_ci(
    sm_to_samples: "dict[int, list[float]]",
    total_sm: int,
    threshold: float = 0.03,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict:
    """Bootstrap CI of the saturation SM from per-SM latency repeats.

    Args:
        sm_to_samples: {sm_count: [latency_ms repeats]} — each list ideally n ≥ 30.
        total_sm:      device SM count (for the ratio axis).
        threshold:     marginal-gain saturation threshold.
        n_boot:        bootstrap replicates.
        ci:            central CI mass (e.g. 0.95).

    Returns dict with point estimate (median of bootstrap dist) and CI bounds,
    all integers on the SM grid; plus n_per_sm for auditability.
    """
    rng = random.Random(seed)
    sms = sorted(sm_to_samples)
    n_per_sm = {s: len(sm_to_samples[s]) for s in sms}

    point = saturation_sm(
        sms, [float(np.median(sm_to_samples[s])) if sm_to_samples[s] else float("nan")
              for s in sms],
        total_sm, threshold,
    )

    boot: list[int] = []
    for _ in range(n_boot):
        curve = []
        ok = True
        for s in sms:
            samp = sm_to_samples[s]
            if not samp:
                ok = False
                break
            draw = [samp[rng.randrange(len(samp))] for _ in range(len(samp))]
            curve.append(float(np.median(draw)))
        if not ok:
            continue
        sp = saturation_sm(sms, curve, total_sm, threshold)
        if sp is not None:
            boot.append(sp)

    if boot:
        lo = float(np.percentile(boot, 100 * (1 - ci) / 2))
        hi = float(np.percentile(boot, 100 * (1 + ci) / 2))
        ci_low, ci_high = int(round(lo)), int(round(hi))
    else:
        ci_low = ci_high = point if point is not None else -1

    return {
        "sat_sm_point": point,
        "sat_sm_ci_low": ci_low,
        "sat_sm_ci_high": ci_high,
        "ci_mass": ci,
        "n_boot": len(boot),
        "n_per_sm": n_per_sm,
    }


def ci_overlap(a: "tuple[int, int]", b: "tuple[int, int]") -> bool:
    """True if intervals [a_lo,a_hi] and [b_lo,b_hi] overlap (inclusive)."""
    (alo, ahi), (blo, bhi) = a, b
    return max(alo, blo) <= min(ahi, bhi)
