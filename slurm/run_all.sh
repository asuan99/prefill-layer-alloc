#!/bin/bash
#SBATCH -J prefill-all
#SBATCH -p amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=36:00:00
#SBATCH --comment=pytorch
#SBATCH -o /scratch/%u/whlee/prefill-layer-alloc/logs/pipeline_%j.log
#SBATCH -e /scratch/%u/whlee/prefill-layer-alloc/logs/pipeline_%j.err
# =============================================================================
# run_all.sh — Full pipeline (stage1 → stage2 → stage3) on current node
#
# Designed to run inside an interactive SLURM session (srun --pty bash / salloc).
# Each Python step is launched via `srun` so it inherits the job's GPU binding.
#
# USAGE (from repo root, on an allocated compute node):
#   bash slurm/run_all.sh                    # zamba2, auto device
#   bash slurm/run_all.sh falcon_h1          # falcon_h1
#   bash slurm/run_all.sh all                # both models sequentially
#   bash slurm/run_all.sh zamba2 a100_80gb   # explicit hardware key
#
# STEP COUNTS
#   single model : 9 steps
#   all models   : 17 steps  (stage1 per model × 2, ctx_switch once, etc.)
#
# OUTPUT
#   results/stage{1,2,3}/   — CSV / JSON / PNG
#   logs/pipeline_<model>_<timestamp>.log
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

MODEL=${1:-zamba2}
DEVICE=${2:-auto}
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')

mkdir -p logs results/stage1 results/stage2 results/stage3

# ── tee all output to log file ──────────────────────────────────────────────
LOG_FILE="logs/pipeline_${MODEL}_${TIMESTAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

# ── environment ─────────────────────────────────────────────────────────────
module load conda/pytorch_2.9.1_cuda13 2>/dev/null || true
module load cuda/13.0.2                2>/dev/null || true
module load gcc/15.2.0                 2>/dev/null || true
source "$REPO_ROOT/bin/activate"

# ── helpers ──────────────────────────────────────────────────────────────────
_STEP=0
_STEP_TOTAL=0   # set below per MODEL value

banner() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════╗"
    printf  "║  %-56s  ║\n" "$*"
    echo "╚══════════════════════════════════════════════════════════╝"
}

step() {
    _STEP=$(( _STEP + 1 ))
    echo ""
    echo "── [${_STEP}/${_STEP_TOTAL}] $(date '+%H:%M:%S')  $*"
    echo "   ─────────────────────────────────────────────────────────"
}

# srun inherits gpu binding from the current interactive allocation.
# --overlap lets nested srun steps share the parent job's resources.
run_py() {
    srun --overlap --ntasks=1 python "$@"
}

# Non-critical wrapper: plot failures don't abort the pipeline.
run_py_optional() {
    run_py "$@" || {
        echo "  [warn] $(basename "$1") exited non-zero — continuing"
    }
}

elapsed() {
    local t=$(( $(date +%s) - _T_STAGE_START ))
    printf '%dm%02ds' $(( t/60 )) $(( t%60 ))
}

# ── per-stage runners ────────────────────────────────────────────────────────

run_stage1() {
    local model=$1 device=$2
    banner "Stage 1 · SM Scaling Sweep  |  model=$model  device=$device"
    _T_STAGE_START=$(date +%s)

    mkdir -p results/stage1/chunked

    # 1. wave-model 합성 (전체 SM 직접 측정 → wave scaling 합성)
    step "SSM prefill SM scaling sweep — wave-model analytical  ($model)"
    run_py stage1_sm_scaling/run_ssm_prefill_sweep.py \
        --model "$model" --device "$device" --skip-verify

    # 2. chunked prefill 직접 측정 (cooperative barrier 우회, Green Context 실측)
    #    prefill_chunk_tokens ∈ {256,512,1024,2048,4096} × sm_counts × seq_lens × batch_sizes
    step "SSM chunked prefill SM sweep — direct measurement  ($model)"
    run_py stage1_sm_scaling/run_chunked_ssm_sweep.py \
        --model "$model" --device "$device" \
        --prefill-chunk-tokens 256 512 1024 2048 4096 \
        --sm-counts            14 27 40 54 68 81 94 108 \
        --seq-lens             512 1024 2048 4096 8192 \
        --batch-sizes          1 4 16 32 \
        --n-warmup  3 --n-measure 10 \
        --output-dir results/stage1/chunked/

    # 3. Attention / MLP sweep
    step "Attention prefill SM scaling sweep  ($model)"
    run_py stage1_sm_scaling/run_attn_prefill_sweep.py \
        --model "$model" --device "$device"

    step "MLP prefill SM scaling sweep  ($model)"
    run_py stage1_sm_scaling/run_mlp_prefill_sweep.py \
        --model "$model" --device "$device"

    # 4. Analysis + Plots
    step "Stage 1 analysis + plots  ($model)"

    # chunked CSV 경로 자동 탐색
    local chunked_csv
    chunked_csv=$(ls results/stage1/chunked/ssm_chunked_${model}_*.csv 2>/dev/null \
                  | sort | tail -1 || true)

    if [ -n "$chunked_csv" ]; then
        local wave_csv
        wave_csv=$(ls results/stage1/ssm_scaling_${model}_*.csv 2>/dev/null \
                   | grep -v torchscan | sort | tail -1 || true)
        run_py_optional stage1_sm_scaling/analyze_chunk_size.py \
            --csv "$chunked_csv" \
            ${wave_csv:+--wave-csv "$wave_csv"}
    fi

    run_py_optional stage1_sm_scaling/plot_saturation.py --model "$model"
    run_py_optional stage1_sm_scaling/plot_srm.py        --model "$model"
    run_py_optional stage1_sm_scaling/plot_sm_split.py   --model "$model"

    if [ -n "$chunked_csv" ]; then
        run_py_optional stage1_sm_scaling/plot_compare_modules.py \
            --model "$model" --ssm-chunked-csv "$chunked_csv"
    else
        run_py_optional stage1_sm_scaling/plot_compare_modules.py --model "$model"
    fi

    echo ""
    echo "  Stage 1 done  ($(elapsed))  → results/stage1/  (chunked → results/stage1/chunked/)"
}

run_stage2() {
    local model=$1 device=$2
    # $3=1 → skip ctx_switch (already measured for this device in a prior call)
    local skip_ctx=${3:-0}
    banner "Stage 2 · Overhead & Decision Matrix  |  model=$model  device=$device"
    _T_STAGE_START=$(date +%s)

    step "Layer latency baseline  ($model)"
    run_py stage2_overhead/measure_layer_latency.py \
        --model "$model" --device "$device"

    if [ "$skip_ctx" = "0" ]; then
        step "Green Context stream-switch overhead  (hardware, device=$device)"
        run_py stage2_overhead/measure_ctx_switch_latency.py \
            --device "$device" \
            --n-warmup  50 \
            --n-measure 200
    else
        echo ""
        echo "  [skip] ctx_switch already measured for this device"
    fi

    step "Decision matrix  (stage1 saturation + stage2 overhead)"
    run_py stage2_overhead/compute_decision_matrix.py \
        --stage1-dir results/stage1 \
        --stage2-dir results/stage2

    echo ""
    echo "  Stage 2 done  ($(elapsed))  → results/stage2/"
}

run_stage3() {
    local model=$1 device=$2
    banner "Stage 3 · Concurrent Prefill+Decode Eval  |  model=$model  device=$device"
    _T_STAGE_START=$(date +%s)

    step "Concurrent eval — policies A, B, C  ($model)"
    run_py stage3_hm_eval/run_concurrent_eval.py \
        --model "$model" --device "$device" \
        --policy all

    step "Stage 3 plots  ($model)"
    run_py_optional stage3_hm_eval/plot_results.py \
        --results-dir results/stage3 --model "$model"

    echo ""
    echo "  Stage 3 done  ($(elapsed))  → results/stage3/"
}

# ── main ─────────────────────────────────────────────────────────────────────
_T_PIPELINE_START=$(date +%s)

echo "╔══════════════════════════════════════════════════════════╗"
echo "║      Prefill-Layer-Alloc  ·  Full Pipeline              ║"
echo "╚══════════════════════════════════════════════════════════╝"
printf "  model   = %s\n"  "$MODEL"
printf "  device  = %s\n"  "$DEVICE"
printf "  log     = %s\n"  "$LOG_FILE"
printf "  started = %s\n"  "$(date '+%Y-%m-%d %H:%M:%S')"
printf "  GPU     = %s\n"  "$(nvidia-smi --query-gpu=name --format=csv,noheader \
                               | head -1 2>/dev/null || echo 'N/A')"
echo ""

if [ "$MODEL" = "all" ]; then
    # ─────────────────────────────────────────────────────────────────────────
    # All models:
    #   stage1 × 2 models (5 steps each) = 10
    #     ssm wave-model, ssm chunked, attn, mlp, analysis+plots
    #   stage2 zamba2     (3 steps)       =  3   ← ctx_switch included
    #   stage2 falcon_h1  (2 steps)       =  2   ← ctx_switch skipped
    #   stage3 × 2 models (2 steps each)  =  4
    #   total                             = 19
    # ─────────────────────────────────────────────────────────────────────────
    _STEP_TOTAL=19

    run_stage1 "zamba2"    "$DEVICE"
    run_stage1 "falcon_h1" "$DEVICE"

    run_stage2 "zamba2"    "$DEVICE" 0   # measures ctx_switch
    run_stage2 "falcon_h1" "$DEVICE" 1   # reuses ctx_switch result from above

    run_stage3 "zamba2"    "$DEVICE"
    run_stage3 "falcon_h1" "$DEVICE"

else
    # ─────────────────────────────────────────────────────────────────────────
    # Single model:
    #   stage1: ssm wave-model + ssm chunked + attn + mlp + analysis+plots = 5
    #   stage2: latency + ctx_switch + matrix = 3
    #   stage3: eval + plots = 2
    #   total = 10
    # ─────────────────────────────────────────────────────────────────────────
    _STEP_TOTAL=10

    run_stage1 "$MODEL" "$DEVICE"
    run_stage2 "$MODEL" "$DEVICE" 0
    run_stage3 "$MODEL" "$DEVICE"
fi

# ── final summary ─────────────────────────────────────────────────────────────
_T_TOTAL=$(( $(date +%s) - _T_PIPELINE_START ))
_H=$(( _T_TOTAL / 3600 ))
_M=$(( (_T_TOTAL % 3600) / 60 ))
_S=$(( _T_TOTAL % 60 ))

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Pipeline complete                                       ║"
echo "╚══════════════════════════════════════════════════════════╝"
printf "  elapsed  = %dh %dm %ds\n" "$_H" "$_M" "$_S"
printf "  results  → results/stage{1,2,3}/\n"
printf "  log      → %s\n" "$LOG_FILE"
