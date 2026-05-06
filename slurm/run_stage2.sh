#!/bin/bash
#SBATCH -J prefill-stage2
#SBATCH -p amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --comment=pytorch
#SBATCH -o /scratch/%u/prefill-layer-alloc/logs/stage2_%j.log
#SBATCH -e /scratch/%u/prefill-layer-alloc/logs/stage2_%j.err

module load conda/pytorch_2.9.1_cuda12
module load cuda/12.8
module load gcc/15.2.0
source activate /scratch/$USER/envs/prefill-alloc

cd /scratch/$USER/prefill-layer-alloc
mkdir -p logs results

MODEL=${1:-zamba2}

echo "=== Stage 2: Overhead Measurement | model=$MODEL ==="

python stage2_overhead/measure_layer_latency.py  --model $MODEL --device a100_80gb
python stage2_overhead/measure_smctrl_latency.py --model $MODEL --device a100_80gb
python stage2_overhead/compute_decision_matrix.py --model $MODEL

echo "=== Stage 2 Done ==="
