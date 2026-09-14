"""Generate decision-complete paired R2 campaign manifests.

PROVENANCE FIELDS (added 2026-09-13, GPU 0)
-------------------------------------------
A run record used to carry no `model` and no `context_length`, so it could not
say what it served; the model came from `PDMUX_MODEL` (default Zyphra/Zamba2-2.7B
duplicated in two shell scripts) and the context length from whatever
`r2_eval.sbatch` happened to derive at run time.  With the model being swapped
to a long-context hybrid, "which checkpoint and which window produced this
number" has to be answerable from the manifest itself, so both are now fields.

`cuda_graph` was present but READ BY NOTHING -- pure decoration on the declared
operating point.  It is now load-bearing: `r2_eval.sbatch` and
`engine_bench_runner.sh` derive `PDMUX_DISABLE_CUDA_GRAPH` from it and refuse a
contradicting environment, so `cuda_graph: true` in a manifest means the boot
really did run with CUDA graphs enabled (and the server args dump proves it).

PER-SHAPE lambda* (schema v3, added 2026-09-13, GPU 0)
------------------------------------------------------
Every workload is a fraction of lambda*, and until this revision the whole
campaign took one scalar (`PDMUX_SUSTAINABLE_RATE`, default 4 req/s, never
measured).  `benchmarks/pdmux_eval/lambda_star.py` replaces it with a per-shape
table; the trace header records which shapes a workload used, what rate each
one got, whether that rate was MEASURED, and how the rates were combined.
`build_runs` reads that header and REFUSES a campaign built on an unmeasured
lambda* unless the caller opts in; `lambda_star_source` then travels in every
run record and `r2_eval.sbatch` refuses the same thing again before it boots.

★NAMING CAVEAT, KEPT IN THE SCHEMA BECAUSE IT KEEPS BEING FORGOTTEN (verdict
citation-stop Q4).  lambda* is ARM-SPECIFIC: job 905835 measured 0.933 / 0.675 /
0.187 req/s at D16 / D44 / D92, a 5x spread.  A workload's NAME -- "W8 near
saturation", "W9 overload" -- is therefore true at most for the single arm whose
lambda* was injected (B1), and the identical trace may sit at 0.2x capacity for
another arm.  Nothing in this manifest makes those names true for any other arm.
Likewise `lambda_star.definition` matters: the canon asks for the sustainable
SLO rate, a throughput-saturation knee is a different and larger quantity
(estimated 1.7-3.4x), and declaring the latter does NOT satisfy gate #6.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence


# Duplicated, on purpose, from scripts/r2_eval/engine_bench_runner.sh and
# scripts/r2_eval/r2_eval.sbatch: the shells must stay usable standalone, and a
# generator that imported a shell default would be worse.  Drift is asserted
# away by tests/test_r2_eval_runner.py.
DEFAULT_MODEL = "Zyphra/Zamba2-2.7B"


BASELINES: Dict[str, str] = {
    "B0": "vanilla",
    "B1": "global_static",
    "B2": "per_workload_static_oracle",
    "B3": "single_worker_dynamic",
    "B4": "dual_worker_fixed",
    "B5": "dual_worker_generic_dynamic",
    "B6": "dual_worker_hybrid_dynamic",
    "B7": "layer_granular_negative",
    "B8": "trace_aware_dynamic_oracle",
}


@dataclass(frozen=True)
class CampaignRun:
    run_id: str
    pair_id: str
    repetition: int
    order: int
    baseline: str
    architecture: str
    policy: str
    decode_sms: int
    requires_offline_oracle: bool
    workload: str
    trace_path: str
    trace_sha256: str
    workload_seed: int
    server_seed: int
    cuda_graph: bool
    max_running_requests: int
    ttft_slo_ms: float
    itl_slo_ms: float
    model: str
    # None = "derive the model's own limit at run time"
    # (benchmarks/pdmux_eval/context_limit.py).  An explicit value is still
    # validated against that limit and against every trace before any boot.
    context_length: Optional[int]
    # "measured" only when every shape this workload emits had a measured
    # lambda* AND no rate derivation was left undecided; otherwise
    # "unmeasured", or "undeclared" for a trace written before 2026-09-13.
    # Read by r2_eval.sbatch, which refuses anything but "measured" without an
    # explicit opt-in.
    lambda_star_source: str


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def trace_lambda_star(path: Path) -> Optional[Dict[str, object]]:
    """The lambda* provenance block a trace declares, or None if it declares none.

    A trace written before 2026-09-13 has no such block; it is not treated as
    "fine", it is treated as UNDECLARED, which `build_runs` refuses.
    """
    with path.open("r", encoding="utf-8") as source:
        first_line = source.readline()
    if not first_line.strip():
        raise ValueError(f"{path}: trace is empty")
    header = json.loads(first_line)
    if header.get("record") != "workload":
        raise ValueError(f"{path}: first record must be the workload header")
    block = header.get("lambda_star")
    if block is not None and not isinstance(block, dict):
        raise ValueError(f"{path}: 'lambda_star' header must be an object")
    return block


def build_runs(
    workloads: Sequence[str],
    baselines: Sequence[str],
    trace_dir: Path,
    repetitions: int,
    seed: int,
    model: str = DEFAULT_MODEL,
    context_length: Optional[int] = None,
    cuda_graph: bool = True,
    allow_unmeasured_lambda_star: bool = False,
) -> List[CampaignRun]:
    rng = random.Random(seed)
    runs: List[CampaignRun] = []
    for workload in workloads:
        trace_path = trace_dir / f"{workload}.jsonl"
        if not trace_path.is_file():
            raise FileNotFoundError(trace_path)
        trace_hash = file_sha256(trace_path)
        declared = trace_lambda_star(trace_path)
        lambda_star_source = str((declared or {}).get("source", "undeclared"))
        if lambda_star_source != "measured" and not allow_unmeasured_lambda_star:
            raise ValueError(
                f"REFUSED: {trace_path} declares lambda_star_source="
                f"{lambda_star_source!r}"
                + (f", caveats={(declared or {}).get('caveats')}" if declared else
                   " (no lambda* header: trace predates the per-shape table)")
                + ".  lambda* is measured in campaign stage 0 "
                "(lambda*는 캠페인 0단계에서 측정된다); a campaign whose workload "
                "intensities are fractions of an unmeasured capacity asserts its "
                "own regime labels instead of measuring them (gate #6).  Pass "
                "allow_unmeasured_lambda_star=True / "
                "--allow-unmeasured-lambda-star to proceed deliberately with the "
                "caveats recorded."
            )
        for repetition in range(repetitions):
            ordered = list(baselines)
            rng.shuffle(ordered)
            pair_id = f"{workload}-rep{repetition:02d}"
            shared_seed = seed * 1000 + repetition
            for order, baseline in enumerate(ordered):
                if baseline not in BASELINES:
                    raise KeyError(f"unknown baseline {baseline}")
                architecture, policy, decode_sms, requires_oracle = {
                    "B0": ("vanilla", "vanilla", 108, False),
                    "B1": ("legacy", "fixed", 44, False),
                    "B2": ("legacy", "per_workload_static_oracle", 0, True),
                    "B3": ("legacy", "single_worker_dynamic", 0, False),
                    "B4": ("true_dual", "fixed", 44, False),
                    "B5": ("true_dual", "generic", 0, False),
                    "B6": ("true_dual", "hybrid", 0, False),
                    "B7": ("legacy", "layer_granular", 0, False),
                    "B8": ("true_dual", "trace_oracle", 0, True),
                }[baseline]
                runs.append(
                    CampaignRun(
                        run_id=f"{pair_id}-{baseline}",
                        pair_id=pair_id,
                        repetition=repetition,
                        order=order,
                        baseline=baseline,
                        architecture=architecture,
                        policy=policy,
                        decode_sms=decode_sms,
                        requires_offline_oracle=requires_oracle,
                        workload=workload,
                        trace_path=str(trace_path),
                        trace_sha256=trace_hash,
                        workload_seed=shared_seed,
                        server_seed=shared_seed,
                        cuda_graph=cuda_graph,
                        max_running_requests=48,
                        ttft_slo_ms=3000.0,
                        itl_slo_ms=60.0,
                        model=model,
                        context_length=context_length,
                        lambda_star_source=lambda_star_source,
                    )
                )
    return runs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--workloads", nargs="+", default=[f"W{i}" for i in range(1, 10)])
    parser.add_argument("--baselines", nargs="+", default=list(BASELINES))
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="checkpoint every run in this campaign serves",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=None,
        help="--context-length for every boot; default = derive the model's own "
        "limit at run time",
    )
    parser.add_argument(
        "--no-cuda-graph",
        action="store_true",
        help="declare a cudagraph-OFF campaign; the canonical operating point is "
        "cudagraph-ON, so this is a deliberate off-operating-point run",
    )
    parser.add_argument(
        "--allow-unmeasured-lambda-star",
        action="store_true",
        help="build a campaign from traces whose lambda* is not measured for "
        "every shape they use (or whose rate derivation is undecided).  The "
        "reasons are recorded in the manifest and in every run record either "
        "way; this flag only decides whether they are fatal.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repetitions < 5:
        raise SystemExit("R2 campaigns require at least five repetitions")
    if args.context_length is not None and args.context_length < 1:
        raise SystemExit("--context-length must be positive")
    try:
        runs = build_runs(
            args.workloads,
            args.baselines,
            args.trace_dir,
            args.repetitions,
            args.seed,
            model=args.model,
            context_length=args.context_length,
            cuda_graph=not args.no_cuda_graph,
            allow_unmeasured_lambda_star=args.allow_unmeasured_lambda_star,
        )
    except ValueError as refusal:
        raise SystemExit(str(refusal))
    # Mirror of what each trace declared, so the manifest alone answers "which
    # lambda* produced these arrival rates".  The GATED copy is the per-run
    # `lambda_star_source` field, which r2_eval.sbatch reads.
    lambda_star = {
        workload: trace_lambda_star(args.trace_dir / f"{workload}.jsonl")
        for workload in args.workloads
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                # v2 added the `model` / `context_length` provenance fields and
                # made `cuda_graph` load-bearing.  v3 (2026-09-13) adds the
                # per-shape lambda* provenance: `lambda_star` below, and the
                # gated `lambda_star_source` inside every run.  The consumers
                # accept older manifests (a missing key means "this manifest
                # makes no lambda* claim"), so the bump labels the records
                # rather than invalidating them.
                "schema": "pdmux.campaign/v3",
                "randomization": "paired-within-workload AB/BA",
                "model": args.model,
                "context_length": args.context_length,
                "cuda_graph": not args.no_cuda_graph,
                "lambda_star": lambda_star,
                "runs": [asdict(run) for run in runs],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
