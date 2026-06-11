#!/bin/bash
#SBATCH --job-name=sweep_all_stage1v2
#SBATCH --partition=amd_a100nv_8        # A100 NVLink = SXM4-80GB (matches existing results).
                                        # NOT amd_a100_4 (that is the PCIe box; device guard
                                        # in the runners will abort there anyway).
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --comment=pytorch
#SBATCH --output=slurm/logs/stage1v2_%j.out
#SBATCH --error=slurm/logs/stage1v2_%j.err
# Usage:
#   sbatch slurm/sweep_stage1_v2.sbatch                 # all 3 models
#   sbatch slurm/sweep_stage1_v2.sbatch zamba2          # one model
#   MODELS="zamba2 falcon_h1" sbatch slurm/sweep_stage1_v2.sbatch
#
# Produces (SXM4 only):
#   results/stage1_v2/ssm_chunked_{model}_a100_sxm4_80gb.csv
#   results/stage1_v2/attn_chunked_{model}_a100_sxm4_80gb.csv
#   reports/matched_granularity_{model}.md   (per-model; see analyzer --out)
# Existing results/ CSVs are never touched (new dir only).

set -euo pipefail

# --- Environment (same convention as workspace/characterization/slurm/*.sh) ---
# conda/pytorch_2.9.1_cuda13 = the env the prebuilt mamba_ssm ext was built against;
# cuda/13.0.2 provides libcudart.so.13; gcc for any Triton/JIT compilation.
module load conda/pytorch_2.9.1_cuda13
module load cuda/13.0.2
module load gcc/15.2.0

source /scratch/$USER/whlee/prefill-layer-alloc/bin/activate

# Safety net: if the prebuilt selective_scan_cuda is ABI-mismatched against torch,
# the SSM runner stubs it (stage1_sm_scaling/_mamba_compat.py) and uses ONLY the
# Triton kernel (mamba_chunk_scan_combined). No-op when the real import succeeds.

PROJECT_ROOT=/scratch/$USER/whlee/prefill-layer-alloc
cd "$PROJECT_ROOT"
mkdir -p slurm/logs results/stage1_v2

export TOKENIZERS_PARALLELISM=false
export HF_HOME="$PROJECT_ROOT/hf_cache"
export PYTHONUNBUFFERED=1

# --- Models ------------------------------------------------------------------
MODELS="${MODELS:-${1:-zamba2 falcon_h1 nemotron_h}}"

echo "=================================================================="
echo " Stage 1 v2 chunked sweep (granularity-matched SSM vs Attn)"
echo " host=$(hostname)  job=${SLURM_JOB_ID:-NA}  partition=${SLURM_JOB_PARTITION:-NA}"
echo " models: $MODELS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "=================================================================="

# --- Preflight device check (fail fast before any measurement) ---------------
python - <<'PY'
import sys
sys.path.insert(0, "workspace")
import torch
from shared import sweep_spec as spec
live = spec.canonical_device(torch.cuda.get_device_name(0))
want = spec.device_name()
print(f"[preflight] live GPU -> {live!r}; sweep_spec expects {want!r}")
if live != want:
    sys.exit(f"[preflight] ABORT: device {live!r} != {want!r}. "
             f"Submit to the SXM4 partition (amd_a100nv_8).")
PY

cd "$PROJECT_ROOT/workspace"
STAGE1="characterization/stage1_sm_scaling"

for m in $MODELS; do
  echo; echo "########## MODEL: $m ##########"
  echo "--- [1/3] SSM chunked (Triton, $m) ---"
  python "$STAGE1/run_ssm_chunked_v2_sweep.py" --model "$m"

  echo "--- [2/3] Attn chunked (SDPA, granularity-matched, $m) ---"
  python "$STAGE1/run_attn_chunked_sweep.py" --model "$m"

  echo "--- [3/3] matched-granularity analysis ($m) ---"
  python "$STAGE1/analyze_matched_granularity.py" --model "$m" \
      --out "$PROJECT_ROOT/reports/matched_granularity_${m}.md"
done

echo; echo "ALL DONE. Outputs in results/stage1_v2/ and reports/matched_granularity_*.md"
