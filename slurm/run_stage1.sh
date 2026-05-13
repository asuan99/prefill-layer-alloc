#!/bin/bash
#SBATCH -J 1-prefill-sm-scaling
#SBATCH -p amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --comment=pytorch
#SBATCH -o /scratch/%u/whlee/prefill-layer-alloc/logs/stage1_%j.log
#SBATCH -e /scratch/%u/whlee/prefill-layer-alloc/logs/stage1_%j.err

# Usage:
#   sbatch slurm/run_stage1.sh                   # zamba2, auto-detect GPU
#   sbatch slurm/run_stage1.sh falcon_h1         # falcon_h1
#   sbatch slurm/run_stage1.sh zamba2 a100_80gb  # explicit hardware key

set -euo pipefail

module load conda/pytorch_2.9.1_cuda13
module load cuda/13.0.2
module load gcc/15.2.0

source /scratch/$USER/whlee/prefill-layer-alloc/bin/activate

cd /scratch/$USER/whlee/prefill-layer-alloc
mkdir -p logs results/stage1

MODEL=${1:-zamba2}
DEVICE=${2:-auto}

# Sweep parameters — matching the feasibility-verified defaults in each script:
#   SSM : seq=[512,1024,2048,4096,8192,16384,32768]  bs=[1,4,16,32,64]
#   Attn: seq=[512,1024,2048,4096,8192,16384]         bs=[1,4,16,32]      ctx=4096
#   MLP : seq=[512,1024,2048,4096,8192,16384]         bs=[1,4,16,32,64]

echo "========================================================"
echo " Stage 1: SM Scaling Sweep"
echo "   model  = $MODEL"
echo "   device = $DEVICE"
echo "   job    = ${SLURM_JOB_ID:-local}"
echo "   GPU    = $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "========================================================"

# ── SSM prefill sweep ────────────────────────────────────────────────────────
echo ""
echo "[1/5] SSM prefill SM scaling sweep …"
python stage1_sm_scaling/run_ssm_prefill_sweep.py \
    --model "$MODEL" \
    --device "$DEVICE"

echo ""
echo "[2/5] SSM prefill SM scaling sweep (torch scan) …"
python stage1_sm_scaling/run_ssm_prefill_sweep.py \
    --model "$MODEL" \
    --device "$DEVICE" \
    --force-pytorch-scan

# ── Attention prefill sweep ──────────────────────────────────────────────────
echo ""
echo "[3/5] Attention prefill SM scaling sweep …"
python stage1_sm_scaling/run_attn_prefill_sweep.py \
    --model "$MODEL" \
    --device "$DEVICE"

# ── MLP prefill sweep ────────────────────────────────────────────────────────
echo ""
echo "[4/5] MLP prefill SM scaling sweep …"
python stage1_sm_scaling/run_mlp_prefill_sweep.py \
    --model "$MODEL" \
    --device "$DEVICE"

# ── Plots ────────────────────────────────────────────────────────────────────
echo ""
echo "[5/5] Generating plots …"
python stage1_sm_scaling/plot_saturation.py  --model "$MODEL" 2>/dev/null || \
    python stage1_sm_scaling/plot_saturation.py 2>/dev/null || true
python stage1_sm_scaling/plot_srm.py         --model "$MODEL" 2>/dev/null || \
    python stage1_sm_scaling/plot_srm.py         2>/dev/null || true
python stage1_sm_scaling/plot_sm_split.py    --model "$MODEL" 2>/dev/null || \
    python stage1_sm_scaling/plot_sm_split.py    2>/dev/null || true

echo ""
echo "========================================================"
echo " Stage 1 Done  (results → results/stage1/)"
echo "========================================================"
