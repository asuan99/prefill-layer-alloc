"""SLO-goodput accounting for CONCURRENT replicas (multi-GPU track, 2026-10-12).

Kept in its own module on purpose: ``analyze.py`` is byte-pinned by registered digests of sealed
campaigns (N3 ``ANALYZE_DIGEST``, N2/lambda0 plans), so the 1-GPU analyzer must not change.  The
sequential-round rule (gate 7) lives unchanged in ``analyze.load_bench_serving_rounds``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

from .analyze import RequestResult, percentile, summarize_requests


# ---------------------------------------------------------------------------
# Concurrent replicas (multi-GPU track, 2026-10-12; design reports/
# multigpu_scaling_design_2026-10-12.md section 4-2, audit section 5).
#
# Two different time rules coexist, each for its own situation:
#   * SEQUENTIAL rounds (one server, rounds run back to back; gate 7): the
#     benchmark duration is the SUM of the per-round durations -- see
#     ``load_bench_serving_rounds`` / ``sequential_duration_s``.
#   * CONCURRENT replicas (DP replicas / TP ranks serving the same round at the
#     same time): the denominator of a round is ONE shared client wall clock,
#     the interval from the round's first request SEND to the round's last
#     request COMPLETION, as seen by the client, over all replicas together.
#     Per-replica goodput uses that same shared interval -- a replica must not
#     divide by its own first-send/last-completion span, otherwise a replica
#     that drains early is credited a higher rate than the fleet delivered and
#     the per-replica rows no longer add up to the aggregate.
# Rounds of a concurrent campaign are still sequential with respect to each
# other, so the per-round shared intervals are summed.
#
# The 1-GPU entry points in analyze.py are untouched (pinned by
# tests/test_analyze_concurrent.py).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TimedRequest:
    """One request with client-clock timestamps (seconds, one clock per round).

    ``request`` is None for a request that produced no first token or errored;
    it still occupies the shared wall clock and is counted as attempted.
    """

    replica: str
    sent_s: float
    completion_s: float
    request: "RequestResult | None"


def sequential_duration_s(round_durations_s: Sequence[float]) -> float:
    """Duration of rounds that ran one after another: the sum (gate 7)."""
    return float(sum(float(value) for value in round_durations_s))


def shared_wall_clock_s(round_requests: Sequence[TimedRequest]) -> float:
    """Client-clock interval of one concurrent round, all replicas together:
    earliest request send to latest request completion."""
    if not round_requests:
        raise ValueError("a round needs at least one request")
    first_send = min(item.sent_s for item in round_requests)
    last_completion = max(item.completion_s for item in round_requests)
    span = last_completion - first_send
    if span <= 0:
        raise ValueError("round wall clock must be positive")
    return span


def replica_denominator_s(
    shared_wall_s: float, replica_requests: Sequence[TimedRequest]
) -> float:
    """Per-replica goodput denominator for one round: the SHARED interval, not
    the replica's own span (``replica_requests`` is deliberately unused)."""
    return shared_wall_s


def slo_failure_breakdown(
    requests: Sequence[RequestResult],
    ttft_slo_ms: float,
    itl_slo_ms: float,
) -> Dict[str, float]:
    """Partition requests into pass / TTFT-only / ITL-only / both (audit C-3).

    Uses exactly the ``RequestResult.passes`` predicate: a request with no
    inter-token latency has an undefined ITL p95 (NaN) and fails the ITL leg,
    as it does in ``passes``.  ``pass + ttft_fail + itl_fail + both_fail`` equals
    the number of requests; the pass count equals the goodput numerator.
    """
    counts = {"pass": 0, "ttft_fail": 0, "itl_fail": 0, "both_fail": 0}
    for request in requests:
        ttft_ok = request.ttft_ms <= ttft_slo_ms
        itl_ok = percentile(request.token_itl_ms, 0.95) <= itl_slo_ms
        if ttft_ok and itl_ok:
            counts["pass"] += 1
        elif itl_ok:
            counts["ttft_fail"] += 1
        elif ttft_ok:
            counts["itl_fail"] += 1
        else:
            counts["both_fail"] += 1
    total = max(1, len(requests))
    return {
        **{key: float(value) for key, value in counts.items()},
        **{f"{key}_fraction": value / total for key, value in counts.items()},
        "requests": float(len(requests)),
    }


def summarize_concurrent_rounds(
    rounds: Sequence[Sequence[TimedRequest]],
    ttft_slo_ms: float,
    itl_slo_ms: float,
) -> Dict[str, object]:
    """Aggregate and per-replica SLO summaries for concurrent replicas.

    ``rounds[i]`` holds every request of round ``i`` from every replica.  Both
    the aggregate and each replica are divided by the same denominator: the sum
    over rounds of the round's shared client wall clock.
    """
    if not rounds:
        raise ValueError("no rounds")
    shared = [shared_wall_clock_s(round_requests) for round_requests in rounds]
    total_wall = sequential_duration_s(shared)
    replicas = sorted({item.replica for r in rounds for item in r})

    def collect(replica: "str | None") -> List[RequestResult]:
        return [
            item.request
            for r in rounds
            for item in r
            if item.request is not None and (replica is None or item.replica == replica)
        ]

    def attempted(replica: "str | None") -> int:
        return sum(
            1 for r in rounds for item in r if replica is None or item.replica == replica
        )

    def cell(replica: "str | None", denominator: float) -> Dict[str, object]:
        reqs = collect(replica)
        summary: Dict[str, object] = dict(
            summarize_requests(reqs, 0.0, denominator, ttft_slo_ms, itl_slo_ms)
        )
        summary["attempted_requests"] = float(attempted(replica))
        summary["failed_requests"] = float(attempted(replica) - len(reqs))
        summary["slo_failure_breakdown"] = slo_failure_breakdown(
            reqs, ttft_slo_ms, itl_slo_ms
        )
        return summary

    per_replica = {}
    for replica in replicas:
        denominator = sequential_duration_s(
            [
                replica_denominator_s(
                    wall, [item for item in r if item.replica == replica]
                )
                for wall, r in zip(shared, rounds)
            ]
        )
        per_replica[replica] = cell(replica, denominator)
    return {
        "shared_wall_clock_s": total_wall,
        "round_wall_clock_s": shared,
        "aggregate": cell(None, total_wall),
        "per_replica": per_replica,
    }


def load_replay_records(
    path: Path, replica: str
) -> List[TimedRequest]:
    """Read a ``trace_loadgen`` ``requests.jsonl`` (sent_s / completion_s on the
    client clock, ttft_ms null on failure) as TimedRequests of ``replica``."""
    items: List[TimedRequest] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ok = rec.get("ttft_ms") is not None and not rec.get("error")
            items.append(
                TimedRequest(
                    replica=replica,
                    sent_s=float(rec["sent_s"]),
                    completion_s=float(rec["completion_s"]),
                    request=RequestResult(
                        request_id=str(rec["request_id"]),
                        ttft_ms=float(rec["ttft_ms"]),
                        token_itl_ms=tuple(float(v) for v in rec["token_itl_ms"]),
                        completion_s=float(rec["completion_s"]),
                    )
                    if ok
                    else None,
                )
            )
    return items

