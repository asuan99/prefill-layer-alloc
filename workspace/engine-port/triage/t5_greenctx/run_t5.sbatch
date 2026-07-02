#!/bin/bash
#SBATCH --job-name=p0_t5_greenctx
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --comment=pytorch
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -uo pipefail

module load conda/pytorch_2.9.1_cuda13
module load cuda/13.0.2
module load gcc/15.2.0

HERE=/scratch/$USER/whlee/prefill-layer-alloc/workspace/engine-port/triage/t5_greenctx
# Isolate JIT build/cache away from any venv or home cache.
export TORCH_EXTENSIONS_DIR="$HERE/.torch_ext"
export TRITON_CACHE_DIR="$HERE/.triton_cache"
export CUDA_HOME="${CUDA_HOME:-$(dirname $(dirname $(which nvcc)))}"
export PATH="$HOME/.local/bin:$PATH"   # ninja for torch cpp_extension JIT
mkdir -p "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR"

echo "===== ENV ====="
nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader || true
echo "nvcc: $(nvcc --version | tail -1)"
which python; python -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda)"
echo "===== RUN T5 ====="
cd "$HERE"
python test_greenctx.py
echo "===== EXIT $? ====="
