"""Generate decision-complete paired R2 campaign manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence


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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_runs(
    workloads: Sequence[str],
    baselines: Sequence[str],
    trace_dir: Path,
    repetitions: int,
    seed: int,
) -> List[CampaignRun]:
    rng = random.Random(seed)
    runs: List[CampaignRun] = []
    for workload in workloads:
        trace_path = trace_dir / f"{workload}.jsonl"
        if not trace_path.is_file():
            raise FileNotFoundError(trace_path)
        trace_hash = file_sha256(trace_path)
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
                        cuda_graph=True,
                        max_running_requests=48,
                        ttft_slo_ms=3000.0,
                        itl_slo_ms=60.0,
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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repetitions < 5:
        raise SystemExit("R2 campaigns require at least five repetitions")
    runs = build_runs(
        args.workloads,
        args.baselines,
        args.trace_dir,
        args.repetitions,
        args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "schema": "pdmux.campaign/v1",
                "randomization": "paired-within-workload AB/BA",
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
