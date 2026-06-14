#!/bin/bash
# =============================================================================
# run_pipeline.sh — submit the ENTIRE v2 size-sweep pipeline with ONE command.
#
#   sbatch experiments/slurm/run_pipeline.sh          # runs as a tiny CPU job
#                                                      # that submits the DAG
#   bash   experiments/slurm/run_pipeline.sh          # or submit from login node
#
# It only issues `sbatch --dependency` calls and exits, so the DAG runs itself:
#
#     E0 (inline, cpu)                                 analytical baseline
#     E1 (gpu array 0-3) ┐                             component decomp  → G0
#     E2 (gpu array 0-3) ┼─ run in parallel            SM saturation     → G1
#     E3 (gpu array 0-3) ┘                             decode floor
#           │ (afterok E1 & E2)
#     gates (cpu)  ──────────► verdicts/{g0,g1}.json
#           │ (afterok gates)
#     E4 (gpu array 0-3)  ── self-aborts unless G1 = BW_MECHANISM
#
# Each GPU stage is a 4-task array (one model size per task):
#   0=zamba2_1.2b 1=zamba2_2.7b 2=falcon_h1_1.5b 3=falcon_h1_3b
#
# This cluster's amd_a100nv_8 partition REQUIRES a GPU, so the controller and the
# (CPU-only) gates job request gpu:1 by default. If you have a real CPU partition,
# point the light stages at it:  CPU_PART=<cpu> CPU_GRES=gpu:0 sbatch run_pipeline.sh
# Override GPU partition:           A100_PART=<gpu> sbatch run_pipeline.sh
# Serialize E2 behind E1 (G0-first):  SERIAL=1 sbatch run_pipeline.sh
# Zero-GPU-waste alternative (just submits jobs): bash run_pipeline.sh
# =============================================================================
#SBATCH --job-name=v2-pipeline
#SBATCH --partition=amd_a100nv_8
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --comment=pytorch
#SBATCH --output=/scratch/%u/whlee/prefill-layer-alloc/logs/v2_pipeline_%j.log
#SBATCH --error=/scratch/%u/whlee/prefill-layer-alloc/logs/v2_pipeline_%j.err

set -euo pipefail

REPO_ROOT="/scratch/$USER/whlee/prefill-layer-alloc"
CHAR_DIR="$REPO_ROOT/workspace/characterization"
SS="$CHAR_DIR/experiments/slurm/submit_size_sweep.sh"
A100_PART="${A100_PART:-amd_a100nv_8}"
CPU_PART="${CPU_PART:-amd_a100nv_8}"     # no CPU-only partition here → defaults to A100
CPU_GRES="${CPU_GRES:-gpu:1}"            # amd_a100nv_8 requires a GPU; set gpu:0 on a real CPU part
LOG="$REPO_ROOT/logs"; mkdir -p "$LOG"
ACT="source \"$REPO_ROOT/bin/activate\" 2>/dev/null || true; cd \"$CHAR_DIR\""

# --- helpers (return the SLURM job id via --parsable) ------------------------
cpu_job() {  # name  dep  cmd
  local name="$1" dep="$2" cmd="$3"
  sbatch --parsable --job-name="v2-$name" --partition="$CPU_PART" --gres="$CPU_GRES" \
    --nodes=1 --ntasks-per-node=1 --cpus-per-task=2 --time=00:10:00 --comment=pytorch \
    ${dep:+--dependency="$dep"} \
    --output="$LOG/v2_${name}_%j.log" --error="$LOG/v2_${name}_%j.err" \
    --wrap "$ACT && $cmd"
}
gpu_array() {  # exp  time  dep
  local exp="$1" t="$2" dep="$3"
  sbatch --parsable --array=0-3 --time="$t" --partition="$A100_PART" --gres=gpu:1 \
    --job-name="v2-$exp" --nodes=1 --ntasks-per-node=1 --cpus-per-task=4 --comment=pytorch \
    ${dep:+--dependency="$dep"} \
    --output="$LOG/v2_${exp}_%A_%a.log" --error="$LOG/v2_${exp}_%A_%a.err" \
    --export=ALL,EXP="$exp",EXTRA="" \
    "$SS"
}

# --- submit the DAG ----------------------------------------------------------
# E0 is GPU-free, instant, and has no downstream job dependency (E2 recomputes its
# own grid_sat_sm via wave_model), so run it inline instead of burning a job/GPU.
echo "[pipeline] running E0 inline ..."
bash -c "$ACT && python experiments/e0_analytical/run_wave_table.py" \
  || echo "  E0 inline failed (non-fatal — baseline only)"

JE1=$(gpu_array e1  "04:00:00"       "")
JE3=$(gpu_array e3  "04:00:00"       "")
if [ "${SERIAL:-0}" = "1" ]; then
  JE2=$(gpu_array e2 "08:00:00" "afterok:$JE1")   # G0-first: E2 waits for E1
else
  JE2=$(gpu_array e2 "08:00:00" "")               # parallel with E1/E3
fi
JG=$(cpu_job    gates "afterok:$JE1:$JE2"  "python experiments/gates/adjudicate.py")
JE4=$(gpu_array e4  "04:00:00"       "afterok:$JG")

echo "Submitted v2 pipeline (E0 ran inline):"
echo "  E1=$JE1  E3=$JE3  E2=$JE2  gates=$JG  E4=$JE4"
echo "  (E4 self-aborts unless gates write G1=BW_MECHANISM)"
echo "  watch:  squeue -u $USER | grep v2-"
