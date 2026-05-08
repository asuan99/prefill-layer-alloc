#!/bin/bash
#SBATCH -J prefill-stage2
#SBATCH -p amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --comment=pytorch
#SBATCH -o /scratch/%u/whlee/prefill-layer-alloc/logs/stage2_%j.log
#SBATCH -e /scratch/%u/whlee/prefill-layer-alloc/logs/stage2_%j.err

# Usage:
#   sbatch slurm/run_stage2.sh                   # zamba2, auto-detect GPU
#   sbatch slurm/run_stage2.sh falcon_h1         # falcon_h1
#   sbatch slurm/run_stage2.sh zamba2 a100_80gb  # explicit hardware key
#
# Prerequisites:
#   Stage 1 must be completed first (results/stage1/ssm_scaling_*.csv required
#   by compute_decision_matrix.py for the saturation lookup).
#
# Step breakdown:
#   1. measure_layer_latency.py    — SSM/Attn latency at full SM for (model, seq, bs)
#                                    → results/stage2/layer_latency_<model>_<device>.csv
#   2. measure_ctx_switch_latency.py — Green Context stream-switch overhead (hardware,
#                                    not model-specific; skipped if JSON already exists)
#                                    → results/stage2/ctx_switch_overhead_<device>.json
#   3. compute_decision_matrix.py  — Combines stage1 saturation + stage2 overhead
#                                    → results/stage2/decision_matrix.json / .html

set -euo pipefail

module load conda/pytorch_2.9.1_cuda13
module load cuda/13.0.2
module load gcc/15.2.0

source /scratch/$USER/whlee/prefill-layer-alloc/bin/activate

cd /scratch/$USER/whlee/prefill-layer-alloc
mkdir -p logs results/stage2

MODEL=${1:-zamba2}
DEVICE=${2:-auto}

echo "========================================================"
echo " Stage 2: Overhead & Decision Matrix"
echo "   model  = $MODEL"
echo "   device = $DEVICE"
echo "   job    = ${SLURM_JOB_ID:-local}"
echo "   GPU    = $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "========================================================"

# ── Step 1: layer latency baseline (model-specific) ─────────────────────────
# Sweep: seq=[1024,4096,8192,16384]  bs=[4,16,32]  layer_types=[ssm,attn]
echo ""
echo "[1/3] Measuring layer latency baseline (model=$MODEL) …"
python stage2_overhead/measure_layer_latency.py \
    --model  "$MODEL" \
    --device "$DEVICE"

# ── Step 2: Green Context stream-switch overhead (hardware, run once) ────────
# Produces ctx_switch_overhead_<device>.json; skip if already present for this
# device so that repeated stage2 runs for different models don't re-measure.
DEVICE_TAG=$(python - <<'EOF'
import sys, os
sys.path.insert(0, ".")
import torch, yaml
from pathlib import Path
from stage1_sm_scaling.run_ssm_prefill_sweep import load_hardware_config, device_tag
import os
dev = os.environ.get("DEVICE", "auto")
try:
    hw = load_hardware_config(dev)
    print(device_tag(hw))
except Exception:
    print("unknown")
EOF
)

CTX_JSON="results/stage2/ctx_switch_overhead_${DEVICE_TAG}.json"
if [ -f "$CTX_JSON" ]; then
    echo ""
    echo "[2/3] ctx_switch_overhead already exists for device '$DEVICE_TAG' — skipping."
    echo "      ($CTX_JSON)"
else
    echo ""
    echo "[2/3] Measuring Green Context stream-switch overhead …"
    DEVICE="$DEVICE" python stage2_overhead/measure_ctx_switch_latency.py \
        --device "$DEVICE" \
        --n-warmup  50 \
        --n-measure 200
fi

# ── Step 3: decision matrix ──────────────────────────────────────────────────
echo ""
echo "[3/3] Computing decision matrix …"
python stage2_overhead/compute_decision_matrix.py \
    --stage1-dir results/stage1 \
    --stage2-dir results/stage2

echo ""
echo "========================================================"
echo " Stage 2 Done  (results → results/stage2/)"
echo "========================================================"
