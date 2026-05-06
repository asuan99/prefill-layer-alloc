#!/bin/bash
#SBATCH -J prefill-stage1
#SBATCH -p amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --comment=pytorch
#SBATCH -o /scratch/%u/whlee/prefill-layer-alloc/logs/stage1_%j.log
#SBATCH -e /scratch/%u/whlee/prefill-layer-alloc/logs/stage1_%j.err

module load conda/pytorch_2.9.1_cuda12
module load cuda/12.9
module load gcc/15.2.0
# source activate /scratch/$USER/envs/prefill-alloc

cd /scratch/$USER/whlee/prefill-layer-alloc
mkdir -p logs results

MODEL=${1:-zamba2}   # 인자로 모델 지정 가능: sbatch run_stage1.sh falcon_h1

echo "=== Stage 1: SM Scaling Sweep | model=$MODEL ==="

python stage1_sm_scaling/run_ssm_prefill_sweep.py  --model $MODEL --device a100_80gb
python stage1_sm_scaling/run_attn_prefill_sweep.py --model $MODEL --device a100_80gb
python stage1_sm_scaling/run_mlp_prefill_sweep.py  --model $MODEL --device a100_80gb

echo "=== Stage 1 Done ==="
