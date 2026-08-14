#!/bin/bash
# =============================================================================
# serving-eval/slurm/setup_env.sh — vLLM 서빙 환경 설치 (SLURM sbatch 전용)
#
# uv 를 기반으로 패키지를 설치한다 (pip 대비 5–10× 빠름).
# uv 가 없으면 먼저 설치한다.
#
# 설치 대상: workspace/serving-eval/.venv
#   vLLM ≥ 0.6.3 (Zamba2 지원) / ≥ 0.8.x (Nemotron-H)
#   transformers, datasets, aiohttp, matplotlib, pyyaml 등
#
# 주의: characterization/.venv 와 완전히 분리된 환경.
#   mamba-ssm / triton (characterization 전용) 을 절대 설치하지 말 것.
#
# Usage:
#   sbatch workspace/serving-eval/slurm/setup_env.sh
#   sbatch workspace/serving-eval/slurm/setup_env.sh nemotron_h   # vLLM ≥ 0.8.x
# =============================================================================
#SBATCH -J se-setup-env
#SBATCH -p amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --comment="field=efficientai;appl=pytorch"
#SBATCH -o /scratch/%u/whlee/prefill-layer-alloc/logs/se_setup_%j.log
#SBATCH -e /scratch/%u/whlee/prefill-layer-alloc/logs/se_setup_%j.err

set -euo pipefail

module load conda/pytorch_2.9.1_cuda13
module load cuda/13.0.2
module load gcc/15.2.0

REPO_ROOT="/scratch/$USER/whlee/prefill-layer-alloc"
SERVING_DIR="$REPO_ROOT/workspace/serving-eval"
VENV_DIR="$SERVING_DIR/.venv"

MODEL_TARGET=${1:-zamba2}
case "$MODEL_TARGET" in
    nemotron_h) VLLM_REQ="vllm>=0.8.0" ;;
    *)           VLLM_REQ="vllm>=0.6.3" ;;
esac

mkdir -p "$REPO_ROOT/logs"

echo "=================================================================="
echo " serving-eval venv setup (uv 기반)"
echo "   target model = $MODEL_TARGET"
echo "   vLLM req     = $VLLM_REQ"
echo "   venv dir     = $VENV_DIR"
echo "   GPU          = $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "   started      = $(date '+%Y-%m-%d %H:%M:%S')"
echo "=================================================================="
echo " ※ characterization/.venv 와 분리된 환경"
echo "   mamba-ssm / causal-conv1d 는 vLLM 의존성으로 자동 설치됨 (수동 설치 불필요)"
echo "   characterization/.venv 의 커스텀 버전(Green Context 패치)과 혼용 금지"
echo "=================================================================="

# ── [0] uv 설치 확인 및 설치 ─────────────────────────────────────────────────
echo ""
echo "── [0] uv 확인 …"

# uv 가 PATH 에 있으면 재사용
if command -v uv &>/dev/null; then
    echo "   uv 이미 존재: $(uv --version)"
else
    echo "   uv 없음 → 설치 시도"

    # 방법 1: pip install uv (모듈 Python 에 설치, 가장 안정적)
    if pip install uv 2>/dev/null; then
        echo "   pip install uv 성공"
    # 방법 2: curl 공식 installer (네트워크 접근 가능한 환경)
    elif curl -LsSf https://astral.sh/uv/install.sh | sh 2>/dev/null; then
        # curl installer 는 ~/.local/bin 또는 ~/.cargo/bin 에 설치
        export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
        echo "   curl installer 성공"
    else
        echo "   [warn] uv 설치 실패 → pip fallback 사용"
    fi
fi

# pip 호환 래퍼: uv 있으면 uv pip, 없으면 pip
_install() {
    if command -v uv &>/dev/null; then
        uv pip install "$@"
    else
        pip install "$@"
    fi
}

echo "   설치 도구: $(command -v uv &>/dev/null && echo "uv $(uv --version)" || echo "pip (uv 없음)")"

# ── [1] venv 생성 ─────────────────────────────────────────────────────────────
echo ""
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "── [1/5] 새 venv 생성 …"
    # uv venv 는 --python 으로 현재 Python 을 명시해 conda Python 을 사용
    if command -v uv &>/dev/null; then
        uv venv "$VENV_DIR" --python "$(which python)"
    else
        python -m venv "$VENV_DIR"
    fi
else
    echo "── [1/5] 기존 venv 재사용 ($VENV_DIR)"
fi

source "$VENV_DIR/bin/activate"

# pip / uv 자체 업그레이드
_install --upgrade pip setuptools wheel 2>/dev/null || true

# ── [2] vLLM ─────────────────────────────────────────────────────────────────
# uv 는 의존성 해석이 훨씬 빠름. CUDA runtime 은 vLLM wheel 에 번들돼 있어
# 별도 index URL 불필요.
echo ""
echo "── [2/5] vLLM 설치 ($VLLM_REQ) …"
_install "$VLLM_REQ"

# ── [3] transformers + acceleration ──────────────────────────────────────────
echo ""
echo "── [3/5] transformers + acceleration …"
_install \
    "transformers>=4.45.0" \
    "accelerate>=0.28.0" \
    "tokenizers>=0.19.0"

# ── [4] 워크로드 트레이스 유틸리티 ──────────────────────────────────────────
echo ""
echo "── [4/5] workload trace utilities …"
_install \
    "datasets>=2.18.0" \
    "openai>=1.12.0" \
    "aiohttp>=3.9.3" \
    "tqdm>=4.66.0"

# ── [5] 분석 / 시각화 ────────────────────────────────────────────────────────
echo ""
echo "── [5/5] analytics …"
_install \
    "pandas>=2.1.0" \
    "numpy>=1.26.0" \
    "matplotlib>=3.8.0" \
    "seaborn>=0.13.0" \
    "pyyaml>=6.0.1" \
    "nvidia-ml-py>=12.0.0"

# ── 설치 검증 ────────────────────────────────────────────────────────────────
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
echo "   설치 도구 : $(command -v uv &>/dev/null && echo uv || echo pip)"
echo "   Activate  : source $VENV_DIR/bin/activate"
echo "   다음 단계 : sbatch workspace/serving-eval/slurm/run_s1_1_serving.sh $MODEL_TARGET smoke"
echo "=================================================================="
