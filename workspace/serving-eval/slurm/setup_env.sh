#!/bin/bash
# =============================================================================
# serving-eval/slurm/setup_env.sh — vLLM 서빙 환경 설치 (SLURM sbatch 전용)
#
# workspace/serving-eval/setup_env.sh 와 동일한 venv 를 설치하되
# SLURM 모듈 로드 + GPU 확인 + 설치 검증을 포함한 SLURM 배치 버전.
#
# 설치 대상: workspace/serving-eval/.venv
#   vLLM ≥ 0.6.3 (Zamba2 지원) / ≥ 0.8.x (Nemotron-H)
#   transformers, datasets, aiohttp, matplotlib, pyyaml 등
#
# 주의: characterization/.venv 와 완전히 분리된 환경.
#   mamba-ssm / triton (characterization 전용) 을 절대 설치하지 말 것.
#
# Usage:
#   sbatch serving-eval/slurm/setup_env.sh
#   sbatch serving-eval/slurm/setup_env.sh nemotron_h   # vLLM ≥ 0.8.x 설치
# =============================================================================
#SBATCH -J se-setup-env
#SBATCH -p amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --comment=pytorch
#SBATCH -o /scratch/%u/whlee/prefill-layer-alloc/logs/se_setup_%j.log
#SBATCH -e /scratch/%u/whlee/prefill-layer-alloc/logs/se_setup_%j.err

set -euo pipefail

module load conda/pytorch_2.9.1_cuda13
module load cuda/13.0.2
module load gcc/15.2.0

REPO_ROOT="/scratch/$USER/whlee/prefill-layer-alloc"
SERVING_DIR="$REPO_ROOT/workspace/serving-eval"
VENV_DIR="$SERVING_DIR/.venv"

# 모델별 최소 vLLM 버전 요건
# zamba2    : vLLM ≥ 0.6.3
# nemotron_h: vLLM ≥ 0.8.x (verify from nvidia/Nemotron-H-8B-Base model card)
# falcon_h1 : vLLM ≥ 0.7.x (verify from vLLM issue tracker)
MODEL_TARGET=${1:-zamba2}
case "$MODEL_TARGET" in
    nemotron_h) VLLM_REQ="vllm>=0.8.0" ;;
    *)           VLLM_REQ="vllm>=0.6.3" ;;
esac

mkdir -p "$REPO_ROOT/logs"

echo "=================================================================="
echo " serving-eval venv setup (SLURM)"
echo "   target model = $MODEL_TARGET"
echo "   vLLM req     = $VLLM_REQ"
echo "   venv dir     = $VENV_DIR"
echo "   GPU          = $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "   started      = $(date '+%Y-%m-%d %H:%M:%S')"
echo "=================================================================="
echo " ※ characterization/.venv 와 분리된 환경입니다."
echo "   mamba-ssm / triton 은 이 venv 에 설치하지 마십시오."
echo "=================================================================="

# venv 생성 (이미 있으면 재사용)
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo ""
    echo "── [1/5] 새 venv 생성 …"
    python -m venv "$VENV_DIR"
else
    echo ""
    echo "── [1/5] 기존 venv 재사용 ($VENV_DIR)"
fi

source "$VENV_DIR/bin/activate"
pip install --upgrade pip wheel setuptools

# vLLM (핵심 의존성)
echo ""
echo "── [2/5] vLLM 설치 ($VLLM_REQ) …"
pip install "$VLLM_REQ" \
    --extra-index-url https://download.pytorch.org/whl/cu121

# transformers + acceleration
echo ""
echo "── [3/5] transformers + acceleration …"
pip install \
    "transformers>=4.45.0" \
    "accelerate>=0.28.0" \
    "tokenizers>=0.19.0"

# 워크로드 트레이스 유틸리티
echo ""
echo "── [4/5] workload trace utilities …"
pip install \
    "datasets>=2.18.0" \
    "openai>=1.12.0" \
    "aiohttp>=3.9.3" \
    "tqdm>=4.66.0"

# 분석 / 시각화
echo ""
echo "── [5/5] analytics …"
pip install \
    "pandas>=2.1.0" \
    "numpy>=1.26.0" \
    "matplotlib>=3.8.0" \
    "seaborn>=0.13.0" \
    "pyyaml>=6.0.1" \
    "nvidia-ml-py>=12.0.0"

# 설치 검증
echo ""
echo "── 설치 검증 …"
python -c "
import vllm, transformers, torch
print(f'  torch       : {torch.__version__}')
print(f'  cuda        : {torch.version.cuda}')
print(f'  GPU         : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')
print(f'  vLLM        : {vllm.__version__}')
print(f'  transformers: {transformers.__version__}')
try:
    import pynvml; pynvml.nvmlInit()
    print('  pynvml      : OK')
except Exception as e:
    print(f'  pynvml      : {e}')
print('  ALL OK')
"

echo ""
echo "=================================================================="
echo " Setup complete."
echo "   Activate : source $VENV_DIR/bin/activate"
echo "   Run (smoke): sbatch serving-eval/slurm/run_s1_1_serving.sh $MODEL_TARGET smoke"
echo "=================================================================="
