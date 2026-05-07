#!/bin/bash
#SBATCH -J prefill-setup
#SBATCH -p amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --comment=pytorch
#SBATCH -o /scratch/%u/whlee/setup_env_%j.log

module load conda/pytorch_2.9.1_cuda13
module load cuda/13.0.2
module load gcc/15.2.0

mkdir -p /scratch/$USER/whlee/prefill-layer-alloc/envs

# conda 환경 생성 (pytorch 모듈 환경 기반으로 venv)
python -m venv /scratch/$USER/whlee/prefill-layer-alloc/envs/prefill-alloc --system-site-packages
source /scratch/$USER/whlee/prefill-layer-alloc/envs/prefill-alloc/bin/activate


pip install --user --upgrade pip ninja

# CUDA 컴파일 필요 패키지
pip install --user causal-conv1d==1.5.3.post1
pip install --user mamba-ssm==2.3.1
pip install --user flash-attn==2.7.4.post1 --no-build-isolation

# 나머지 패키지
pip install --user pynvml pandas matplotlib seaborn pyyaml tqdm
pip install --user flashinfer-python --index-url https://flashinfer.ai/whl/cu124/torch2.9/

# 검증
python -c "
import torch, mamba_ssm, flash_attn
print('torch    :', torch.__version__)
print('cuda     :', torch.version.cuda)
print('gpu      :', torch.cuda.get_device_name(0))
print('mamba    :', mamba_ssm.__version__)
print('flash_attn:', flash_attn.__version__)
print('ALL OK')
"
