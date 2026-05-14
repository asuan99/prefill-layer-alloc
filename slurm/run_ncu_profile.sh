#!/bin/bash
#SBATCH -J prefill-ncu
#SBATCH --exclusive 
#SBATCH --constrain=hwperf
#SBATCH -p amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --comment=pytorch
#SBATCH -o /scratch/%u/whlee/prefill-layer-alloc/logs/ncu_%j.log
#SBATCH -e /scratch/%u/whlee/prefill-layer-alloc/logs/ncu_%j.err

# ncu requires hardware counter access. On KSC (amd_a100nv_8 partition) the
# SLURM job environment enables GPU performance counters that are denied in
# interactive sessions (ERR_NVGPUCTRPERM, perf_event_paranoid=2).
#
# Usage:
#   sbatch slurm/run_ncu_profile.sh                                      # zamba2, all layer types
#   sbatch slurm/run_ncu_profile.sh falcon_h1                            # falcon_h1
#   sbatch slurm/run_ncu_profile.sh zamba2 --layer-types ssm attn        # specific layer types
#   sbatch slurm/run_ncu_profile.sh zamba2 --layer-types ssm --sm-counts 27 54 108 --seq-lens 1024 4096
#   sbatch slurm/run_ncu_profile.sh zamba2 --layer-types chunked_ssm \
#       --prefill-chunk-tokens 256 512 1024 --sm-counts 27 54 108
#
# Arguments: MODEL [extra args passed directly to run_ncu_profile.py]
#   Note: layer types and other options must use their full --flag form
#         (e.g. --layer-types ssm attn), NOT bare positional words.

module load conda/pytorch_2.9.1_cuda13
module load cuda/13.0.2
module load gcc/15.2.0

source /scratch/$USER/whlee/prefill-layer-alloc/bin/activate

cd /scratch/$USER/whlee/prefill-layer-alloc
mkdir -p logs results/stage1

# Parse positional: first arg = model, remaining = forwarded as-is to the script
MODEL=${1:-zamba2}
shift 2>/dev/null || true

echo "=== Stage 1: ncu Wave Profiling | model=$MODEL | job=$SLURM_JOB_ID ==="
echo "    GPU: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
echo "    ncu: $(ncu --version 2>&1 | head -1)"
echo ""

python stage1_sm_scaling/run_ncu_profile.py \
    --model "$MODEL" \
    --device a100_80gb \
    "$@"

echo ""
echo "=== ncu Profiling Done ==="
