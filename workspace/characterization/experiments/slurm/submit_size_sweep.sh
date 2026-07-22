#!/bin/bash
# =============================================================================
# submit_size_sweep.sh — run one GPU experiment across the 4 model sizes in
# PARALLEL via a SLURM array (one array task per model).
#
#   env -u BASH_ENV bash experiments/slurm/submit_size_sweep.sh <e1|e2|e3|e4|e5> [-- extra]
#
# Array index → model:
#   0 = zamba2_1.2b   1 = zamba2_2.7b   2 = falcon_h1_1.5b   3 = falcon_h1_3b
#
# ⚠️ This cluster's BASH_ENV (lmod) breaks non-interactive bash, so the job is
#    submitted as `--wrap "env -u BASH_ENV bash -c '…'"` (SLURM runs --wrap under
#    /bin/sh, which is BASH_ENV-immune; env -u then gives a clean bash). Launch
#    this dispatcher itself with `env -u BASH_ENV bash …`.
#
# E0/gates are CPU/instant — use submit.sh (LOCAL=1) for those.
# Overrides:  A100_PART=<gpu partition>
# =============================================================================
set -euo pipefail

EXP="${1:-}"; shift || true
[ "${1:-}" = "--" ] && shift || true
EXTRA="$*"

# Default = the 4 v2 SLM/mid models. Override for the 7B-scale (Path 1) sweep, e.g.
#   MODELS="zamba2_7b falcon_h1_7b" env -u BASH_ENV bash …/submit_size_sweep.sh e5 -- …
# The array size is derived from the model count, so any list works.
MODELS="${MODELS:-zamba2_1.2b zamba2_2.7b falcon_h1_1.5b falcon_h1_3b}"
REPO_ROOT="/scratch/$USER/whlee/prefill-layer-alloc"     # assumed space-free
CHAR_DIR="$REPO_ROOT/workspace/characterization"
A100_PART="${A100_PART:-amd_a100nv_8}"
LOG="$REPO_ROOT/logs"; mkdir -p "$LOG"

case "$EXP" in
  e1) SCRIPT="experiments/e1_prefill_decomp/run_component_sweep.py";   T="04:00:00" ;;
  e2) SCRIPT="experiments/e2_sm_saturation/run_batch_swept_sweep.py";  T="08:00:00" ;;
  e3) SCRIPT="experiments/e3_decode_floor/run_decode_floor.py";        T="04:00:00" ;;
  e4) SCRIPT="experiments/e4_concurrent/run_concurrent_ab.py";         T="04:00:00" ;;
  e5) SCRIPT="experiments/e5_serving/run_serving_coexec.py";           T="04:00:00" ;;
  fused) SCRIPT="experiments/e5_serving/run_fused_step.py";            T="04:00:00" ;;
  *) echo "usage: $0 <e1|e2|e3|e4|e5|fused> [-- extra args]"; exit 2 ;;
esac

# Inner bash -c body (single-quote-safe). $SLURM_ARRAY_TASK_ID stays literal
# because the printf format is single-quoted.
INNER="$(printf 'MODELS=(%s); M=${MODELS[$SLURM_ARRAY_TASK_ID]}; source %s/bin/activate 2>/dev/null || true; cd %s; echo "task $SLURM_ARRAY_TASK_ID -> $M"; python %s --models $M %s' \
  "$MODELS" "$REPO_ROOT" "$CHAR_DIR" "$SCRIPT" "$EXTRA")"

NMODELS=$(echo $MODELS | wc -w); ARRAY="0-$((NMODELS - 1))"
echo "[size-sweep] submitting ${NMODELS}-task array (--array=$ARRAY) for $EXP (models: $MODELS)"
sbatch --array="$ARRAY" --partition="$A100_PART" --gres=gpu:1 \
  --job-name="v2-$EXP" --nodes=1 --ntasks-per-node=1 --cpus-per-task=4 \
  --time="$T" --comment=pytorch \
  --output="$LOG/v2_${EXP}_%A_%a.log" --error="$LOG/v2_${EXP}_%A_%a.err" \
  --wrap "env -u BASH_ENV bash -c '$INNER'"
