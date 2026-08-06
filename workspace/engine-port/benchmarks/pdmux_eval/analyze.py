"""Primary SLO-goodput and paired bootstrap analysis for R2."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
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


def paired_bootstrap_ci(
    baseline: Mapping[str, float],
    proposed: Mapping[str, float],
    samples: int = 10000,
    seed: int = 1,
) -> Dict[str, float]:
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
    }


def unpaired_bootstrap_ci(
    baseline: Sequence[float],
    proposed: Sequence[float],
    samples: int = 10000,
    seed: int = 1,
) -> Dict[str, float]:
    """Difference-of-means CI when arms have no natural repetition pairing.

    ``paired_bootstrap_ci`` is preferred whenever repetitions are matched (same
    seed/trace slot).  Static-split arms from separate SLURM jobs have no such
    matching, so each arm is resampled independently with the same conventions
    (10000 samples, seed=1) rather than inventing an arbitrary pairing.
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
    parser = argparse.ArgumentParser()
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
    result = paired_bootstrap_ci(baseline, proposed)
    result.update(
        {
            "baseline": args.baseline,
            "proposed": args.proposed,
            "metric": args.metric,
            "headline_improvement": (
                result["effect_percent"] >= 3.0 and result["ci95_low"] > 0.0
            ),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
