#!/bin/bash
#SBATCH -J prefill-stage3
#SBATCH -p amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --comment=pytorch
#SBATCH -o /scratch/%u/prefill-layer-alloc/logs/stage3_%j.log
#SBATCH -e /scratch/%u/prefill-layer-alloc/logs/stage3_%j.err

module load conda/pytorch_2.9.1_cuda12
module load cuda/12.8
module load gcc/15.2.0
source activate /scratch/$USER/envs/prefill-alloc

cd /scratch/$USER/prefill-layer-alloc
mkdir -p logs results

MODEL=${1:-zamba2}

echo "=== Stage 3: Concurrent Eval | model=$MODEL ==="

python stage3_hm_eval/run_concurrent_eval.py --model $MODEL --policy all --device a100_80gb

echo "=== Stage 3 Done ==="
