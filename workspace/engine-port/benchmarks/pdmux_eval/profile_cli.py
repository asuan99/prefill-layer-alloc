"""Build and validate HybridModelProfileV1 files from measured CSV data."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def _load_profile_module():
    try:
        from sglang.srt.multiplex import profile
    except ImportError:
        import importlib.util
        import sys

        source = Path(__file__).parents[2] / "src" / "multiplex" / "profile.py"
        spec = importlib.util.spec_from_file_location("pdmux_profile", source)
        profile = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules["pdmux_profile"] = profile
        spec.loader.exec_module(profile)
    return profile


def build(metadata_path: Path, measurements_path: Path, output_path: Path) -> None:
    module = _load_profile_module()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    points = []
    with measurements_path.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            points.append(
                module.DecodeLatencyPoint(
                    decode_sms=int(row["decode_sms"]),
                    batch_size=int(row["batch_size"]),
                    context_tokens=int(row["context_tokens"]),
                    itl_p50_ms=float(row["itl_p50_ms"]),
                    itl_p95_ms=float(row["itl_p95_ms"]),
                    itl_p99_ms=float(row["itl_p99_ms"]),
                    repeats=int(row["repeats"]),
                    measured_steps=int(row["measured_steps"]),
                    residual_p95_ms=float(row.get("residual_p95_ms") or 0.0),
                )
            )
    if not (metadata.get("environment") or {}).get("engine_source_hash"):
        # NOT filled in automatically: this process is not necessarily the
        # engine the measurements came from, and stamping the builder host's
        # hash onto someone else's measurements is exactly the false-provenance
        # the axis exists to prevent.  An empty hash is fail-closed --
        # `is_compatible` will report the profile as incompatible, so the
        # serving run falls back instead of trusting it.
        print(
            "WARNING: metadata.environment has no engine_source_hash; the "
            "profile will be INCOMPATIBLE with every runtime (fail-closed). "
            "Fill it with `profile_cli.py engine-source-hash`, run ON the "
            "engine that produced the measurements.",
            file=sys.stderr,
        )
    metadata["environment"] = module.RuntimeEnvironment(**metadata["environment"])
    metadata["points"] = points
    profile = module.HybridModelProfileV1(**metadata)
    profile.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    build_parser = commands.add_parser("build")
    build_parser.add_argument("--metadata", type=Path, required=True)
    build_parser.add_argument("--measurements", type=Path, required=True)
    build_parser.add_argument("--output", type=Path, required=True)
    validate_parser = commands.add_parser("validate")
    validate_parser.add_argument("profile", type=Path)
    # Prints the engine-source compatibility axis of THIS engine, to be pasted
    # into a metadata file's `environment.engine_source_hash`.  Must be run on
    # the engine tree that produced the measurements (see profile.py's
    # "Engine source identity" note for why the sync manifest is not the axis).
    commands.add_parser("engine-source-hash")
    args = parser.parse_args()
    module = _load_profile_module()
    if args.command == "engine-source-hash":
        value = module.engine_source_hash()
        if not value:
            print(
                "ERROR: could not resolve every engine source module "
                f"({', '.join(module.ENGINE_SOURCE_MODULES)})",
                file=sys.stderr,
            )
            raise SystemExit(2)
        print(value)
    elif args.command == "build":
        build(args.metadata, args.measurements, args.output)
        module.HybridModelProfileV1.load(args.output)
        print(f"VALID {args.output}")
    else:
        item = module.HybridModelProfileV1.load(args.profile)
        print(
            json.dumps(
                {
                    "status": "valid",
                    "schema": item.schema,
                    "model_id": item.model_id,
                    "points": len(item.points),
                    "attention_ratio": item.attention_ratio,
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
