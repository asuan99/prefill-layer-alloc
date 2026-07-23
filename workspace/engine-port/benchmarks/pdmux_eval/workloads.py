"""Deterministic synthetic workload definitions for W1--W9."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


@dataclass(frozen=True)
class RequestSpec:
    request_id: str
    arrival_s: float
    input_tokens: int
    output_tokens: int
    context_tokens: int
    phase: str


@dataclass(frozen=True)
class WorkloadSpec:
    workload_id: str
    description: str
    arrival_process: str
    input_distribution: str
    output_distribution: str
    context_distribution: str
    burstiness: str
    concurrency: str
    ttft_slo_ms: float
    itl_slo_ms: float
    opportunity: str
    limitation: str


def builtin_workloads() -> Dict[str, WorkloadSpec]:
    return {
        "W1": WorkloadSpec(
            "W1", "balanced steady", "Poisson at 0.60 lambda*",
            "constant 2K", "constant 128", "constant 2K", "none",
            "rate-derived", 3000.0, 60.0, "steady overlap", "limited temporal slack",
        ),
        "W2": WorkloadSpec(
            "W2", "prefill burst", "10s 4x burst + 30s drain, 3 cycles",
            "constant 8K", "constant 64", "constant 8K", "periodic 4x",
            "rate-derived", 3000.0, 60.0, "burst/drain complementarity",
            "requires 8K model context",
        ),
        "W3": WorkloadSpec(
            "W3", "decode heavy", "Poisson at 0.80 lambda*",
            "constant 256", "constant 512", "constant 256", "none",
            "high active decode", 3000.0, 60.0, "moving decode floor",
            "may remain decode saturated",
        ),
        "W4": WorkloadSpec(
            "W4", "alternating", "30s prefill/decode phases, 3 cycles",
            "8K/256 by phase", "64/512 by phase", "8K/256 by phase",
            "periodic", "phase-dependent", 3000.0, 60.0,
            "strong temporal complementarity", "transition overhead sensitive",
        ),
        "W5": WorkloadSpec(
            "W5", "context swing", "deterministic alternating at 0.60 lambda*",
            "1K/8K alternating", "constant 128", "1K/8K alternating",
            "request-level", "rate-derived", 3000.0, 60.0,
            "profile/context value", "16K optional by model support",
        ),
        "W6": WorkloadSpec(
            "W6", "output-length swing", "Poisson at 0.60 lambda*",
            "constant 2K", "32/512 alternating", "constant 2K",
            "request-level", "phase-dependent", 3000.0, 60.0,
            "remaining-output congestion signal", "length hints may be unavailable",
        ),
        "W7": WorkloadSpec(
            "W7", "low load", "Poisson at 0.20 lambda*",
            "constant 2K", "constant 128", "constant 2K", "none",
            "low", 3000.0, 60.0, "overhead control", "little multiplexing value",
        ),
        "W8": WorkloadSpec(
            "W8", "near saturation", "Poisson at 0.90/0.95 lambda*",
            "constant 2K", "constant 128", "constant 2K", "none",
            "near capacity", 3000.0, 60.0, "primary capacity region",
            "threshold-cliff variance",
        ),
        "W9": WorkloadSpec(
            "W9", "overload", "Poisson at 1.10/1.25/1.50 lambda*",
            "constant 2K", "constant 128", "constant 2K", "none",
            "over capacity", 3000.0, 60.0, "admission fallback",
            "no split can make sustained overload feasible",
        ),
    }


def _poisson_arrivals(
    rng: random.Random, count: int, rate: float, start_s: float = 0.0
) -> List[float]:
    if rate <= 0:
        raise ValueError("arrival rate must be positive")
    now = start_s
    arrivals: List[float] = []
    for _ in range(count):
        now += rng.expovariate(rate)
        arrivals.append(now)
    return arrivals


def generate_requests(
    workload_id: str,
    sustainable_rate: float,
    count: int,
    seed: int,
    intensity: float = 1.0,
) -> List[RequestSpec]:
    if workload_id not in builtin_workloads():
        raise KeyError(f"unknown workload {workload_id}")
    if count <= 0:
        raise ValueError("count must be positive")
    rng = random.Random(seed)
    requests: List[RequestSpec] = []

    def add(arrivals: Iterable[float], inp: int, out: int, phase: str) -> None:
        for arrival in arrivals:
            index = len(requests)
            requests.append(
                RequestSpec(
                    f"{workload_id}-{seed}-{index:06d}",
                    float(arrival),
                    inp,
                    out,
                    inp,
                    phase,
                )
            )

    if workload_id == "W2":
        per_cycle = max(
            1, round(4.0 * 0.20 * sustainable_rate * intensity * 10.0)
        )
        for cycle in range(3):
            start = cycle * 40.0
            # A homogeneous Poisson process conditioned on its event count has
            # sorted-uniform event times inside the interval.
            arrivals = sorted(
                start + 10.0 * rng.random() for _ in range(per_cycle)
            )
            add(arrivals, 8192, 64, f"burst-{cycle}")
    elif workload_id == "W4":
        per_phase = max(1, round(0.80 * sustainable_rate * intensity * 30.0))
        for phase_index in range(6):
            start = phase_index * 30.0
            is_prefill = phase_index % 2 == 0
            add(
                sorted(
                    start + 30.0 * rng.random()
                    for _ in range(per_phase)
                ),
                8192 if is_prefill else 256,
                64 if is_prefill else 512,
                "prefill" if is_prefill else "decode",
            )
    else:
        rate_fraction = {
            "W1": 0.60,
            "W3": 0.80,
            "W5": 0.60,
            "W6": 0.60,
            "W7": 0.20,
            "W8": 0.90,
            "W9": 1.10,
        }[workload_id]
        arrivals = _poisson_arrivals(
            rng, count, rate_fraction * sustainable_rate * intensity
        )
        for index, arrival in enumerate(arrivals):
            if workload_id == "W3":
                inp, out = 256, 512
            elif workload_id == "W5":
                inp, out = ((1024, 128) if index % 2 == 0 else (8192, 128))
            elif workload_id == "W6":
                inp, out = 2048, (32 if index % 2 == 0 else 512)
            else:
                inp, out = 2048, 128
            add([arrival], inp, out, "steady")
    requests.sort(key=lambda item: (item.arrival_s, item.request_id))
    return requests if workload_id in {"W2", "W4"} else requests[:count]


def write_trace(path: Path, spec: WorkloadSpec, requests: Sequence[RequestSpec]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        output.write(json.dumps({"record": "workload", **asdict(spec)}, sort_keys=True) + "\n")
        for request in requests:
            output.write(
                json.dumps({"record": "request", **asdict(request)}, sort_keys=True)
                + "\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", choices=sorted(builtin_workloads()), required=True)
    parser.add_argument("--sustainable-rate", type=float, required=True)
    parser.add_argument(
        "--count",
        type=int,
        default=256,
        help="request count for steady workloads; timed W2/W4 derive it from rate",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--intensity", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = builtin_workloads()[args.workload]
    write_trace(
        args.output,
        spec,
        generate_requests(
            args.workload,
            args.sustainable_rate,
            args.count,
            args.seed,
            args.intensity,
        ),
    )


if __name__ == "__main__":
    main()
