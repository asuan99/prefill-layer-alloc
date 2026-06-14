#!/bin/bash
# =============================================================================
# submit.sh — independent SLURM submit for each v2 experiment (E0..E4 + gates).
#
#   bash experiments/slurm/submit.sh <exp> [-- extra args to the python script]
#
#   <exp> ∈ e0 | e1 | e2 | e3 | e4 | gates | test
#
#   e0, gates, test  → CPU partition (no GPU)
#   e1, e2, e3, e4   → A100 partition
#
# Each experiment is fully independent (its own job). Subprocess isolation is
# inside the runners themselves (one worker process per SM level / per config),
# reusing the v1 stage pattern; this script only schedules the top-level runner.
#
# Run E0 / gates / test on a login or CPU node without SLURM:
#   LOCAL=1 bash experiments/slurm/submit.sh e0
#
# Override partitions if your cluster differs:
#   A100_PART=amd_a100nv_8  CPU_PART=amd_a100nv_8   (defaults)
#   amd_a100nv_8 REQUIRES a GPU, so CPU stages (e0/gates/test) default to gpu:1.
#   On a real CPU partition:  CPU_PART=<cpu> CPU_GRES=gpu:0 bash submit.sh gates
# =============================================================================
set -euo pipefail

EXP="${1:-}"
shift || true
# allow a leading "--" before extra args
[ "${1:-}" = "--" ] && shift || true
EXTRA=("$@")

REPO_ROOT="/scratch/$USER/whlee/prefill-layer-alloc"
CHAR_DIR="$REPO_ROOT/workspace/characterization"
A100_PART="${A100_PART:-amd_a100nv_8}"
CPU_PART="${CPU_PART:-amd_a100nv_8}"
LOGDIR="$REPO_ROOT/logs"
mkdir -p "$LOGDIR"

case "$EXP" in
  e0)    SCRIPT="experiments/e0_analytical/run_wave_table.py";        GPU=0; T="00:10:00" ;;
  e1)    SCRIPT="experiments/e1_prefill_decomp/run_component_sweep.py"; GPU=1; T="04:00:00" ;;
  e2)    SCRIPT="experiments/e2_sm_saturation/run_batch_swept_sweep.py"; GPU=1; T="08:00:00" ;;
  e3)    SCRIPT="experiments/e3_decode_floor/run_decode_floor.py";    GPU=1; T="04:00:00" ;;
  e4)    SCRIPT="experiments/e4_concurrent/run_concurrent_ab.py";     GPU=1; T="04:00:00" ;;
  gates) SCRIPT="experiments/gates/adjudicate.py";                    GPU=0; T="00:05:00" ;;
  test)  SCRIPT="experiments/e4_concurrent/test_dispatch.py";         GPU=0; T="00:05:00" ;;
  *) echo "usage: $0 <e0|e1|e2|e3|e4|gates|test> [-- extra args]"; exit 2 ;;
esac

# Command run on the node: activate venv (if present), cd into characterization,
# run the experiment entry point with any extra args.
RUNCMD="source \"$REPO_ROOT/bin/activate\" 2>/dev/null || true; \
cd \"$CHAR_DIR\" && python \"$SCRIPT\" ${EXTRA[*]:-}"

if [ "${LOCAL:-0}" = "1" ]; then
  echo "[submit.sh] LOCAL run: $EXP -> $SCRIPT ${EXTRA[*]:-}"
  bash -c "$RUNCMD"
  exit $?
fi

if [ "$GPU" = "1" ]; then
  PART="$A100_PART"; GRES="--gres=gpu:1"
else
  # amd_a100nv_8 requires a GPU; CPU stages default to gpu:1. On a real CPU
  # partition use: CPU_PART=<cpu> CPU_GRES=gpu:0 bash submit.sh gates
  PART="$CPU_PART";  GRES="--gres=${CPU_GRES:-gpu:1}"
fi

echo "[submit.sh] sbatch $EXP -> $SCRIPT  (part=$PART $GRES time=$T)"
sbatch \
  --job-name="v2-$EXP" \
  --partition="$PART" \
  $GRES \
  --nodes=1 --ntasks-per-node=1 --cpus-per-task=4 \
  --time="$T" \
  --comment=pytorch \
  --output="$LOGDIR/v2_${EXP}_%j.log" \
  --error="$LOGDIR/v2_${EXP}_%j.err" \
  --wrap "$RUNCMD"
