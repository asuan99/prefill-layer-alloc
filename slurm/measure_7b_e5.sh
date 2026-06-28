#!/bin/bash
#SBATCH --job-name=e5_7b
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --comment=pytorch
#SBATCH --output=slurm/logs/e5_7b_%j.out
#SBATCH --error=slurm/logs/e5_7b_%j.err
set -uo pipefail
module load conda/pytorch_2.9.1_cuda13
module load cuda/13.0.2
module load gcc/15.2.0
PROJECT_ROOT=/scratch/$USER/whlee/prefill-layer-alloc
source "$PROJECT_ROOT/bin/activate"
export TOKENIZERS_PARALLELISM=false HF_HOME="$PROJECT_ROOT/hf_cache" PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="$PROJECT_ROOT/.triton_cache"     # avoid home-cache surprises
cd "$PROJECT_ROOT/workspace/characterization"
nvidia-smi --query-gpu=name --format=csv,noheader || true

echo "########## [0] DIAGNOSTIC: decode_ssm full traceback (zamba2_7b, falcon_h1_7b) ##########"
for M in zamba2_7b falcon_h1_7b; do
  echo "===== diag $M ====="
  python experiments/e5_serving/_diag_decode_ssm.py "$M" || true
done

BATCHES="1 2 4 8 16 32 64 128 256 512"
for M in zamba2_7b falcon_h1_7b; do
  echo "########## [E5] serving_coexec (opt+protect): $M ##########"
  python experiments/e5_serving/run_serving_coexec.py --models "$M" \
      --prefill-mode full --prefill-tokens 256 --chunk 256 --prefill-batch 8 \
      --context-lens 4096 --decode-batches $BATCHES \
      --decode-protect --opt-decode --output-dir results_v2/e5_sim_b8_opt || echo "E5 FAILED: $M"
done
echo "ALL DONE."
