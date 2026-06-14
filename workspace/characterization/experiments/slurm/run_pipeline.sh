#!/bin/bash
# =============================================================================
# run_pipeline.sh — submit the ENTIRE v2 size-sweep pipeline with ONE command.
#
#   env -u BASH_ENV bash experiments/slurm/run_pipeline.sh        # recommended
#   SERIAL=1 env -u BASH_ENV bash experiments/slurm/run_pipeline.sh
#
# ⚠️ This cluster sets BASH_ENV=/usr/share/lmod/.../init/bash, whose broken
#    sourcing makes ANY non-interactive `bash`/`bash -c` exit before running its
#    body. So: (1) launch this controller with `env -u BASH_ENV bash …`, and
#    (2) every job is submitted as `--wrap "env -u BASH_ENV bash -c '…'"` — SLURM
#    runs --wrap under /bin/sh (BASH_ENV-immune), then env -u gives a clean bash.
#
# DAG (4-task arrays = one model size per task):
#     E0 (inline, cpu)                       analytical baseline
#     E1 / E2 / E3 (gpu arrays, parallel)    decomp / saturation / decode-floor
#           │ afterok E1 & E2
#     gates (cpu) → verdicts/{g0,g1}.json
#           │ afterok gates
#     E4 (gpu array)  ── self-aborts unless G1 = BW_MECHANISM
#   array idx → 0=zamba2_1.2b 1=zamba2_2.7b 2=falcon_h1_1.5b 3=falcon_h1_3b
#
# Overrides:  A100_PART=<gpu> CPU_PART=<cpu> CPU_GRES=gpu:0 SERIAL=1
# (amd_a100nv_8 requires a GPU, so CPU stages default to gpu:1.)
# =============================================================================

set -euo pipefail

REPO_ROOT="/scratch/$USER/whlee/prefill-layer-alloc"
CHAR_DIR="$REPO_ROOT/workspace/characterization"     # assumed space-free
A100_PART="${A100_PART:-amd_a100nv_8}"
CPU_PART="${CPU_PART:-amd_a100nv_8}"
CPU_GRES="${CPU_GRES:-gpu:1}"
LOG="$REPO_ROOT/logs"; mkdir -p "$LOG"
MODELS="zamba2_1.2b zamba2_2.7b falcon_h1_1.5b falcon_h1_3b"
ACT="source $REPO_ROOT/bin/activate 2>/dev/null || true; cd $CHAR_DIR"

# inner bash -c body for a GPU array task (single-quote-safe: no single quotes,
# $SLURM_ARRAY_TASK_ID stays literal via the single-quoted printf format).
gpu_inner() {  # script
  printf 'MODELS=(%s); M=${MODELS[$SLURM_ARRAY_TASK_ID]}; %s; echo "task $SLURM_ARRAY_TASK_ID -> $M"; python %s --models $M' \
    "$MODELS" "$ACT" "$1"
}

gpu_array() {  # exp time dep script
  local exp="$1" t="$2" dep="$3" script="$4" inner
  inner="$(gpu_inner "$script")"
  sbatch --parsable --array=0-3 --time="$t" --partition="$A100_PART" --gres=gpu:1 \
    --job-name="v2-$exp" --nodes=1 --ntasks-per-node=1 --cpus-per-task=4 --comment=pytorch \
    ${dep:+--dependency="$dep"} \
    --output="$LOG/v2_${exp}_%A_%a.log" --error="$LOG/v2_${exp}_%A_%a.err" \
    --wrap "env -u BASH_ENV bash -c '$inner'"
}

cpu_job() {  # name dep cmd
  local name="$1" dep="$2" cmd="$3"
  sbatch --parsable --job-name="v2-$name" --partition="$CPU_PART" --gres="$CPU_GRES" \
    --nodes=1 --ntasks-per-node=1 --cpus-per-task=2 --time=00:10:00 --comment=pytorch \
    ${dep:+--dependency="$dep"} \
    --output="$LOG/v2_${name}_%j.log" --error="$LOG/v2_${name}_%j.err" \
    --wrap "env -u BASH_ENV bash -c '$cmd'"
}

# --- E0: GPU-free, instant, no downstream job dep → run inline ----------------
echo "[pipeline] running E0 inline ..."
env -u BASH_ENV bash -c "$ACT; python experiments/e0_analytical/run_wave_table.py" \
  || echo "  E0 inline failed (non-fatal — baseline only)"

# --- GPU arrays + gates + E4 -------------------------------------------------
JE1=$(gpu_array e1 "04:00:00" "" "experiments/e1_prefill_decomp/run_component_sweep.py")
JE3=$(gpu_array e3 "04:00:00" "" "experiments/e3_decode_floor/run_decode_floor.py")
if [ "${SERIAL:-0}" = "1" ]; then
  JE2=$(gpu_array e2 "08:00:00" "afterok:$JE1" "experiments/e2_sm_saturation/run_batch_swept_sweep.py")
else
  JE2=$(gpu_array e2 "08:00:00" "" "experiments/e2_sm_saturation/run_batch_swept_sweep.py")
fi
JG=$(cpu_job gates "afterok:$JE1:$JE2" "$ACT && python experiments/gates/adjudicate.py")
JE4=$(gpu_array e4 "04:00:00" "afterok:$JG" "experiments/e4_concurrent/run_concurrent_ab.py")

echo "Submitted v2 pipeline (E0 ran inline):"
echo "  E1=$JE1  E3=$JE3  E2=$JE2  gates=$JG  E4=$JE4"
echo "  (E4 self-aborts unless gates write G1=BW_MECHANISM)"
echo "  watch:  squeue -u $USER | grep v2-"
