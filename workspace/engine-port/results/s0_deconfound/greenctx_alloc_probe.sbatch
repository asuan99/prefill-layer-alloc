#!/bin/bash
#SBATCH --job-name=s0dcprobe
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --comment="field=efficientai;appl=pytorch"
#SBATCH --output=%x_%A.out
#SBATCH --error=%x_%A.err
# ============================================================================
# Stage 0 de-confound, gate probe (no serving stack, no model).
#
# Question: does create_greenctx_stream_by_value(a, b) honor `b`, or does
# stream B just get the 108-a remainder? Every division used so far sums to
# exactly 108, so nothing measured to date distinguishes these.
#
# If honored -> the de-confound sweep (prefill SM pinned at 16, decode SM
# swept, remainder idle) is expressible purely in pdmux yml configs.
# If remainder -> it needs an engine change, and this probe is the cheapest
# possible way to learn that (minutes, no boot).
# ============================================================================
set -uo pipefail
module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0
export PS1="" PATH="$HOME/.local/bin:$PATH"
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate

ROOT=/scratch/ehmoon/whlee/prefill-layer-alloc
HERE=$ROOT/workspace/engine-port/results/s0_deconfound
export PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="$ROOT/workspace/engine-port/triage/.triton_cache"

"$ROOT/workspace/engine-port/scripts/bootstrap/sync_engine_tree.sh" || {
  echo "SYNC_ENGINE_TREE_FAILED"; exit 3; }

nvidia-smi --query-gpu=name,clocks.max.sm,power.limit --format=csv,noheader

python3 "$HERE/greenctx_alloc_probe.py" \
  "$HERE/greenctx_alloc_probe_${SLURM_JOB_ID:-local}.json"
echo "PROBE_RC=$?"
