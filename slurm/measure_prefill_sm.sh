#!/bin/bash
#SBATCH --job-name=prefill_sm
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --comment=pytorch
#SBATCH --output=slurm/logs/prefill_sm_%j.out
#SBATCH --error=slurm/logs/prefill_sm_%j.err
set -euo pipefail
module load conda/pytorch_2.9.1_cuda13
module load cuda/13.0.2
module load gcc/15.2.0
PROJECT_ROOT=/scratch/$USER/whlee/prefill-layer-alloc
source "$PROJECT_ROOT/bin/activate"
export TOKENIZERS_PARALLELISM=false HF_HOME="$PROJECT_ROOT/hf_cache" PYTHONUNBUFFERED=1
cd "$PROJECT_ROOT/workspace/characterization"
nvidia-smi --query-gpu=name --format=csv,noheader || true
python - <<'PY'
import os,sys; sys.path.insert(0,os.path.abspath(".."))
import torch; from shared import sweep_spec as spec
live=spec.canonical_device(torch.cuda.get_device_name(0)); want=spec.device_name()
print(f"[preflight] live={live!r} expect={want!r}")
sys.exit(0 if live==want else f"ABORT {live}!={want}")
PY
echo "########## prefill SM-sweep (7b + 2.7b control + 1.2b) ##########"
python experiments/e7_prefill_sm/run_prefill_sm_sweep.py --models zamba2_7b zamba2_2.7b zamba2_1.2b --layers ssm attn
echo "ALL DONE."
