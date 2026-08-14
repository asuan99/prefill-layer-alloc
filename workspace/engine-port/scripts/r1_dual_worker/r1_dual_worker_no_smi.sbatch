#!/bin/bash
# R1 submission variant for accounts/nodes where nvidia-smi is not permitted.
# The shared runner still validates CUDA through torch and executes the same
# legacy-vs-dual-worker experiment matrix.

#SBATCH --job-name=r1_dual_worker_no_smi
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:30:00
#SBATCH --comment="field=efficientai;appl=pytorch"
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail
REPO_ROOT=/scratch/ehmoon/whlee/prefill-layer-alloc
SCRIPT_PATH="$REPO_ROOT/workspace/engine-port/scripts/r1_dual_worker/r1_dual_worker.sbatch"
test -f "$SCRIPT_PATH"
exec env \
  R1_USE_NVIDIA_SMI=0 \
  R1_SUBMISSION_VARIANT=no_smi \
  R1_SCRIPT_PATH="$SCRIPT_PATH" \
  bash "$SCRIPT_PATH"
