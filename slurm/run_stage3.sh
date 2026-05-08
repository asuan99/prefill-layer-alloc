#!/bin/bash
#SBATCH -J prefill-stage3
#SBATCH -p amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --comment=pytorch
#SBATCH -o /scratch/%u/whlee/prefill-layer-alloc/logs/stage3_%j.log
#SBATCH -e /scratch/%u/whlee/prefill-layer-alloc/logs/stage3_%j.err

# Usage:
#   sbatch slurm/run_stage3.sh                          # zamba2, all policies, defaults
#   sbatch slurm/run_stage3.sh falcon_h1                # falcon_h1
#   sbatch slurm/run_stage3.sh zamba2 a100_80gb A B     # specific policies
#
# Positional args:
#   $1  MODEL   (default: zamba2)
#   $2  DEVICE  (default: auto)
#   $3+ POLICY  space-separated list: A B C all  (default: all)
#
# Prerequisites:
#   Stage 2 must be completed — run_concurrent_eval reads
#   results/stage2/decision_matrix.json to decide whether Policy C runs.
#
# Default sweep params (set via CLI defaults in run_concurrent_eval.py):
#   --prefill-seq-len   8192   (realistic long-context prefill)
#   --decode-batch-size   64   (SM-saturating decode batch)
#   --context-len       4096   (KV cache length for decode)

set -euo pipefail

module load conda/pytorch_2.9.1_cuda13
module load cuda/13.0.2
module load gcc/15.2.0

source /scratch/$USER/whlee/prefill-layer-alloc/bin/activate

cd /scratch/$USER/whlee/prefill-layer-alloc
mkdir -p logs results/stage3

MODEL=${1:-zamba2}
DEVICE=${2:-auto}
shift 2 2>/dev/null || true
POLICIES=${*:-all}

echo "========================================================"
echo " Stage 3: Concurrent Prefill+Decode Evaluation"
echo "   model    = $MODEL"
echo "   device   = $DEVICE"
echo "   policies = $POLICIES"
echo "   job      = ${SLURM_JOB_ID:-local}"
echo "   GPU      = $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "========================================================"

# ── Concurrent eval ──────────────────────────────────────────────────────────
echo ""
echo "[1/2] Running concurrent prefill+decode evaluation …"
python stage3_hm_eval/run_concurrent_eval.py \
    --model  "$MODEL" \
    --device "$DEVICE" \
    --policy $POLICIES

# ── Plots ────────────────────────────────────────────────────────────────────
echo ""
echo "[2/2] Generating result plots …"
python stage3_hm_eval/plot_results.py \
    --results-dir results/stage3 \
    --model "$MODEL" 2>/dev/null || \
python stage3_hm_eval/plot_results.py \
    --results-dir results/stage3 2>/dev/null || true

echo ""
echo "========================================================"
echo " Stage 3 Done  (results → results/stage3/)"
echo "========================================================"
