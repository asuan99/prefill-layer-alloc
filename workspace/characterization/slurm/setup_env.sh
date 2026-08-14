#!/bin/bash
#SBATCH -J prefill-setup
#SBATCH -p amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --comment="field=efficientai;appl=pytorch"
#SBATCH -o /scratch/%u/whlee/setup_env_%j.log

module load conda/pytorch_2.9.1_cuda13
module load cuda/13.0.2
module load gcc/15.2.0

VENV_DIR="/scratch/$USER/whlee/prefill-layer-alloc"

# venv를 프로젝트 루트에 생성 (run_stage*.sh / run_all.sh 모두
# source $VENV_DIR/bin/activate 를 사용하므로 경로를 맞춤)
python -m venv "$VENV_DIR" --system-site-packages
source "$VENV_DIR/bin/activate"

pip install --upgrade pip ninja

# CUDA 컴파일 필요 패키지
pip install causal-conv1d==1.5.3.post1
pip install mamba-ssm==2.3.1
pip install flash-attn==2.7.4.post1 --no-build-isolation

# 나머지 패키지
pip install nvidia-ml-py pandas matplotlib seaborn pyyaml tqdm
pip install flashinfer-python --index-url https://flashinfer.ai/whl/cu124/torch2.9/

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
