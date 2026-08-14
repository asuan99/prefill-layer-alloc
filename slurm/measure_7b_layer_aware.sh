#!/bin/bash
#SBATCH --job-name=la7b_e3e5
#SBATCH --partition=amd_a100nv_8        # A100 NVLink = SXM4-80GB (matches existing LUTs)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=10:00:00
#SBATCH --comment="field=efficientai;appl=pytorch"
#SBATCH --output=slurm/logs/la7b_%j.out
#SBATCH --error=slurm/logs/la7b_%j.err
#
# Real-measure the inputs run_layer_aware needs for zamba2_7b (closes the gap that
# made layer_aware_7b_verification.md "blocked"): the E3 per-layer decode FLOOR
# (missing for 7B) and the E5 green_ctx_protect LUT in the SAME e5_sim_b8_opt
# format the 2.7b layer_aware result used (opt-decode + decode-protect, single
# chunk, prefill_batch=8). E5 reads the E3 floor, so they run in order here.
set -euo pipefail

module load conda/pytorch_2.9.1_cuda13
module load cuda/13.0.2
module load gcc/15.2.0

PROJECT_ROOT=/scratch/$USER/whlee/prefill-layer-alloc
source "$PROJECT_ROOT/bin/activate"

export TOKENIZERS_PARALLELISM=false
export HF_HOME="$PROJECT_ROOT/hf_cache"
export PYTHONUNBUFFERED=1

CHAR="$PROJECT_ROOT/workspace/characterization"
cd "$CHAR"
mkdir -p "$PROJECT_ROOT/slurm/logs"

echo "=================================================================="
echo " 7B layer-aware inputs (E3 floor + E5 opt/protect LUT)"
echo " host=$(hostname) job=${SLURM_JOB_ID:-NA} part=${SLURM_JOB_PARTITION:-NA}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "=================================================================="

# Preflight: must be the SXM4 box (existing LUTs are tagged a100_sxm4_80gb).
# cwd is workspace/characterization, so workspace (holding `shared`) is "..".
python - <<'PY'
import os, sys
sys.path.insert(0, os.path.abspath(".."))
import torch
from shared import sweep_spec as spec
live = spec.canonical_device(torch.cuda.get_device_name(0)); want = spec.device_name()
print(f"[preflight] live={live!r} expect={want!r}")
if live != want:
    sys.exit(f"[preflight] ABORT: {live!r} != {want!r}; need amd_a100nv_8 (SXM4).")
PY

BATCHES="1 2 4 8 16 32 64 128 256 512"

echo; echo "########## [1/2] E3 decode floor (zamba2_7b) ##########"
python experiments/e3_decode_floor/run_decode_floor.py --models zamba2_7b \
    --batches $BATCHES --context-lens 4096 --layer-types attn ssm

FLOOR="$CHAR/results_v2/e3/decode_floor_zamba2_7b_a100_sxm4_80gb.csv"
test -s "$FLOOR" || { echo "ABORT: E3 floor not produced ($FLOOR)"; exit 3; }
echo "E3 floor OK: $FLOOR"

echo; echo "########## [2/2] E5 serving_coexec (opt+protect, e5_sim_b8_opt) ##########"
python experiments/e5_serving/run_serving_coexec.py --models zamba2_7b \
    --prefill-mode full --prefill-tokens 256 --chunk 256 --prefill-batch 8 \
    --context-lens 4096 --decode-batches $BATCHES \
    --decode-protect --opt-decode --output-dir results_v2/e5_sim_b8_opt

echo; echo "ALL DONE."
echo "  floor -> $FLOOR"
echo "  LUT   -> $CHAR/results_v2/e5_sim_b8_opt/serving_coexec_full_zamba2_7b_a100_sxm4_80gb.csv"
echo "Next (GPU-free): python experiments/e6_queue_sim/run_layer_aware.py --model zamba2_7b"
