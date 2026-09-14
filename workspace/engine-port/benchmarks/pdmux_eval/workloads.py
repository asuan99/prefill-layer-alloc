"""Deterministic synthetic workload definitions for W1--W9."""

from __future__ import annotations
import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence
from .lambda_star import LambdaStar, coerce, load


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
    lambda_star: LambdaStar,
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
            1, round(4.0 * 0.20 * _rate(lambda_star, W2_SHAPE) * intensity * 10.0)
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
        # ★2026-09-13 REVISION (user decision (a), EXPERIMENT_ROADMAP 'P2'):
        for phase_index in range(6):
            start = phase_index * 30.0
            is_prefill = phase_index % 2 == 0
            # ONE scalar cannot put both phases at 0.80x (their capacities
            # differ ~5x); per_phase now uses THIS phase's own shape rate.
            per_phase = _phase_count(lambda_star, is_prefill, intensity)
            add(
                sorted(start + 30.0 * rng.random() for _ in range(per_phase)),
                8192 if is_prefill else 256,
                64 if is_prefill else 512,
                "prefill" if is_prefill else "decode",
            )
    else:
        rate_fraction = STEADY_RATE_FRACTIONS[workload_id]
        shapes = [_steady_shape(workload_id, index) for index in range(count)]
        arrivals = _poisson_arrivals(
            rng,
            count,
            rate_fraction * _stream_rate(lambda_star, shapes) * intensity,
        )
        for index, arrival in enumerate(arrivals):
            add([arrival], shapes[index][0], shapes[index][1], "steady")
    requests.sort(key=lambda item: (item.arrival_s, item.request_id))
    return requests if workload_id in {"W2", "W4"} else requests[:count]


# ---------------------------------------------------------------------------
# Shapes and rate derivation.
#
# ★THESE ARE DEFINED BELOW `generate_requests` ON PURPOSE.  Four canonical
# documents cite `workloads.py:159-160` -- the 2026-09-13(3) correction that
# W4's decode phase is (256,512), not the phantom (64,512) -- and a verdict
# cites `:148`.  Python resolves module globals at call time, so keeping the
# definitions here let the per-phase revision land without moving one cited
# line.  `tests/test_lambda_star_per_shape.py` pins :148/:159/:160 so that a
# later edit fails loudly instead of silently falsifying the canon (lesson
# #80: a citation goes stale between the moment it is verified and the moment
# it is read).  Move them up only together with those documents.
# ---------------------------------------------------------------------------
W2_SHAPE = (8192, 64)
W4_PREFILL_SHAPE = (8192, 64)
# == every W3 request, exactly.  Not an approximation of anything (gate #198).
W4_DECODE_SHAPE = (256, 512)
W4_PHASE_FRACTION = 0.80
# Fraction of ITS OWN shape's lambda* each single-stream workload aims at.
STEADY_RATE_FRACTIONS = {
    "W1": 0.60,
    "W3": 0.80,
    "W5": 0.60,
    "W6": 0.60,
    "W7": 0.20,
    "W8": 0.90,
    "W9": 1.10,
}


def _rate(lambda_star: LambdaStar, shape) -> float:
    """lambda* for one shape; refuses the pre-2026-09-13 bare scalar."""
    return coerce(lambda_star).rate(tuple(shape))


def _phase_count(lambda_star: LambdaStar, is_prefill: bool, intensity: float) -> int:
    """Requests in one 30 s W4 phase = 0.80 x THAT PHASE's lambda* x 30 s.

    The old code computed this once, outside the phase loop, from a single
    scalar.  That is the defect this revision removes: prefill (8192,64) and
    decode (256,512) have ~5x different capacities, so one scalar always leaves
    one phase far from its 0.80x target.
    """
    shape = W4_PREFILL_SHAPE if is_prefill else W4_DECODE_SHAPE
    return max(
        1, round(W4_PHASE_FRACTION * _rate(lambda_star, shape) * intensity * 30.0)
    )


def _steady_shape(workload_id: str, index: int):
    """The (input, output) shape of request `index` in a single-stream workload.

    Single source of truth for both the emitted requests and
    `workload_shapes()`, so the declared shapes cannot drift from the generated
    ones.
    """
    if workload_id == "W3":
        return (256, 512)
    if workload_id == "W5":
        return (1024, 128) if index % 2 == 0 else (8192, 128)
    if workload_id == "W6":
        return (2048, 32) if index % 2 == 0 else (2048, 512)
    return (2048, 128)


def _stream_rate(lambda_star: LambdaStar, shapes: Sequence) -> float:
    """One Poisson rate for a request sequence (one shape, or interleaved).

    With a single shape this is exactly lambda*(that shape).  W5 and W6
    interleave two shapes inside ONE stream, for which no per-shape rate
    exists; `LambdaStar.mixture_rate` applies a work-conservation DEFINITION
    that is recorded as the `undecided_rate_derivation` caveat.
    """
    return coerce(lambda_star).mixture_rate([tuple(shape) for shape in shapes])


def workload_shapes(workload_id: str, count: int = 256) -> tuple:
    """Every request shape a workload emits -- derived from the generator.

    ★Gate #198: a `WorkloadSpec` string is a DESCRIPTION, not an input shape
    ("64/512 by phase" is the OUTPUT length by phase), and reading it as an
    input shape is what produced the (64,512) phantom.  This function is built
    from `_steady_shape` / the W2 / W4 constants that the emitter itself uses,
    and the test suite cross-checks it against actually generated traces for
    all nine workloads.
    """
    if workload_id == "W2":
        return (W2_SHAPE,)
    if workload_id == "W4":
        return (W4_PREFILL_SHAPE, W4_DECODE_SHAPE)
    if workload_id not in STEADY_RATE_FRACTIONS:
        raise KeyError(f"unknown workload {workload_id}")
    if count <= 0:
        raise ValueError("count must be positive")
    seen: List[tuple] = []
    for index in range(count):
        shape = _steady_shape(workload_id, index)
        if shape not in seen:
            seen.append(shape)
    return tuple(seen)


def workload_rate_derivation(workload_id: str, count: int = 256) -> str:
    """How this workload turns per-shape lambda* into arrival rates."""
    if workload_id == "W4":
        return "per_phase"
    if len(workload_shapes(workload_id, count)) == 1:
        return "single_shape"
    return "interleaved_mixture_harmonic"


def lambda_star_record(
    workload_id: str, lambda_star: LambdaStar, count: int = 256
) -> Dict[str, object]:
    """The provenance block stamped into the trace header.

    `source` is the rollup the campaign builder and `r2_eval.sbatch` gate on;
    `caveats` says why, so refusing and disclosing are one act.
    """
    table = coerce(lambda_star)
    return table.record(
        workload_shapes(workload_id, count),
        workload_rate_derivation(workload_id, count),
    )


def write_trace(
    path: Path,
    spec: WorkloadSpec,
    requests: Sequence[RequestSpec],
    lambda_star: Optional[Dict[str, object]] = None,
) -> None:
    """Write the immutable trace.

    `lambda_star` is the block from `lambda_star_record()`.  A trace written
    without it declares nothing, and `campaign.build_runs` refuses such a trace
    unless the caller opts in -- traces predating 2026-09-13 are exactly that
    case.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    header: Dict[str, object] = {"record": "workload", **asdict(spec)}
    if lambda_star is not None:
        header["lambda_star"] = lambda_star
    with path.open("w", encoding="utf-8") as output:
        output.write(json.dumps(header, sort_keys=True) + "\n")
        for request in requests:
            output.write(
                json.dumps({"record": "request", **asdict(request)}, sort_keys=True)
                + "\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", choices=sorted(builtin_workloads()), required=True)
    parser.add_argument(
        "--lambda-star-table",
        type=Path,
        required=True,
        help="JSON table of per-shape lambda* (see benchmarks/pdmux_eval/"
        "lambda_star.py).  REQUIRED: there is no default, because the default "
        "that used to exist (4 req/s) was never measured.",
    )
    parser.add_argument(
        "--sustainable-rate",
        default=None,
        help="REMOVED 2026-09-13.  One scalar cannot parameterise W4's two "
        "phases; use --lambda-star-table.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=256,
        help="request count for steady workloads; timed W2/W4 derive it from rate",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--intensity", type=float, default=1.0)
    parser.add_argument(
        "--allow-unmeasured-lambda-star",
        action="store_true",
        help="emit a trace whose lambda* is not measured for every shape it "
        "uses (or whose rate derivation is still undecided).  The reasons are "
        "recorded in the trace header either way.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.sustainable_rate is not None:
        raise SystemExit(
            "--sustainable-rate was removed on 2026-09-13: W4's phases are "
            "(8192,64) and (256,512) and no single scalar puts both at 0.80x.  "
            "Use --lambda-star-table."
        )
    table = load(args.lambda_star_table)
    record = lambda_star_record(args.workload, table, args.count)
    if record["source"] != "measured" and not args.allow_unmeasured_lambda_star:
        raise SystemExit(
            f"REFUSED: {args.workload} would be generated from a lambda* that is "
            f"not measured: {record['caveats']}.  Measure it (campaign stage 0) "
            "or pass --allow-unmeasured-lambda-star to record the caveats and "
            "proceed deliberately."
        )
    spec = builtin_workloads()[args.workload]
    write_trace(
        args.output,
        spec,
        generate_requests(
            args.workload,
            table,
            args.count,
            args.seed,
            args.intensity,
        ),
        lambda_star=record,
    )


if __name__ == "__main__":
    main()
