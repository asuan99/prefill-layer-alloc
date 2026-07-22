#!/usr/bin/env bash
set -euo pipefail

# Usage is intentionally identical to sbatch:
#   workspace/slurm-agent/submit.sh workspace/engine-port/triage/r1_dual_worker.sbatch
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$SCRIPT_DIR/slurm_agent.py" submit -- "$@"
