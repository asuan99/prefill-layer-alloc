#!/bin/bash
# =============================================================================
# submit_size_sweep.sh — run a GPU experiment across the 4 model sizes in
# PARALLEL via a SLURM array (one array task per model).
#
#   bash experiments/slurm/submit_size_sweep.sh <e1|e2|e3|e4> [-- extra args]
#
# Array index → model:
#   0 = zamba2_1.2b   1 = zamba2_2.7b   2 = falcon_h1_1.5b   3 = falcon_h1_3b
#
# Each task runs the experiment for ONE model with `--models <model>`, so the
# four sizes run concurrently on four A100 allocations instead of looping in a
# single long job. Inside each task the runner still does its own per-SM-level
# subprocess isolation.
#
# E0 (analytical) and gates (adjudicate) are CPU/instant — use submit.sh for
# those (or run them directly); they are not part of this array.
#
# The script self-submits: with no SLURM array context it calls `sbatch
# --array=0-3` on itself; the array tasks then read $EXP + $SLURM_ARRAY_TASK_ID.
# =============================================================================
#SBATCH --job-name=v2-size-sweep
#SBATCH --partition=amd_a100nv_8
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --comment=pytorch
#SBATCH --output=/scratch/%u/whlee/prefill-layer-alloc/logs/v2_sizesweep_%A_%a.log
#SBATCH --error=/scratch/%u/whlee/prefill-layer-alloc/logs/v2_sizesweep_%A_%a.err

set -euo pipefail

MODELS=(zamba2_1.2b zamba2_2.7b falcon_h1_1.5b falcon_h1_3b)
REPO_ROOT="/scratch/$USER/whlee/prefill-layer-alloc"
CHAR_DIR="$REPO_ROOT/workspace/characterization"

# --- dispatcher mode: not yet inside an array task → submit ourselves ---------
if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
  EXP="${1:-}"; shift || true
  [ "${1:-}" = "--" ] && shift || true
  case "$EXP" in
    e1) T="04:00:00" ;;
    e2) T="08:00:00" ;;   # heaviest: full SM × batch × chunk × layer, n≥30
    e3) T="04:00:00" ;;
    e4) T="04:00:00" ;;
    *) echo "usage: $0 <e1|e2|e3|e4> [-- extra args]"; exit 2 ;;
  esac
  mkdir -p "$REPO_ROOT/logs"
  echo "[size-sweep] submitting 4-task array for $EXP (models: ${MODELS[*]})"
  exec sbatch --array=0-3 --time="$T" \
       --export=ALL,EXP="$EXP",EXTRA="${*:-}" "$0"
fi

# --- array-task mode: run one model ------------------------------------------
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
case "$EXP" in
  e1) SCRIPT="experiments/e1_prefill_decomp/run_component_sweep.py" ;;
  e2) SCRIPT="experiments/e2_sm_saturation/run_batch_swept_sweep.py" ;;
  e3) SCRIPT="experiments/e3_decode_floor/run_decode_floor.py" ;;
  e4) SCRIPT="experiments/e4_concurrent/run_concurrent_ab.py" ;;
  *) echo "bad EXP=$EXP"; exit 2 ;;
esac

echo "=== size-sweep task $SLURM_ARRAY_TASK_ID: $EXP / $MODEL (job ${SLURM_JOB_ID:-?}) ==="
source "$REPO_ROOT/bin/activate" 2>/dev/null || true
cd "$CHAR_DIR"
# shellcheck disable=SC2086
python "$SCRIPT" --models "$MODEL" ${EXTRA:-}
