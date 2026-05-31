#!/bin/bash
# =============================================================================
# run_stage1_serving.sh — Phase 1 S1.1: vLLM serving + NVML monitoring
#
# serving-eval/ 전용 venv (run_stage1.sh 의 venv 와 별도) 를 사용.
# venv 가 없으면 workspace/serving-eval/setup_env.sh 를 먼저 실행.
#
# 검증 항목:
#   1. vLLM 서버가 hybrid 모델로 기동됨
#   2. ShareGPT / smoke trace 로 end-to-end 요청이 처리됨
#   3. 서버 로그에 prefill / decode 혼재가 확인됨
#   4. NVML device-level util 시계열 CSV 가 저장됨
#
# 주의사항:
#   - 모델 가중치 (nvidia/Nemotron-H-8B-Base 또는 Zyphra/Zamba2-7B-Instruct) 가
#     HuggingFace Hub 또는 로컬 캐시에 있어야 함.
#   - 오프라인 환경: HF_HUB_OFFLINE=1 + HF_HOME=<로컬 캐시> 설정 필요.
#   - vLLM ≥ 0.8.x (Nemotron-H) 또는 ≥ 0.6.3 (Zamba2) 필요.
#
# Usage:
#   sbatch slurm/run_stage1_serving.sh                         # zamba2, smoke
#   sbatch slurm/run_stage1_serving.sh zamba2 sharegpt         # full trace
#   sbatch slurm/run_stage1_serving.sh nemotron_h smoke        # Nemotron-H smoke
#   MODEL=zamba2 TRACE=sharegpt sbatch slurm/run_stage1_serving.sh
# =============================================================================
#SBATCH -J s1-serving
#SBATCH -p amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --comment=pytorch
#SBATCH -o /scratch/%u/whlee/prefill-layer-alloc/logs/s1_serving_%j.log
#SBATCH -e /scratch/%u/whlee/prefill-layer-alloc/logs/s1_serving_%j.err

set -euo pipefail

module load conda/pytorch_2.9.1_cuda13
module load cuda/13.0.2
module load gcc/15.2.0

# ── serving-eval 전용 venv ───────────────────────────────────────────────────
SERVING_VENV="/scratch/$USER/whlee/prefill-layer-alloc/workspace/serving-eval/.venv"
REPO_ROOT="/scratch/$USER/whlee/prefill-layer-alloc"

if [ ! -f "$SERVING_VENV/bin/activate" ]; then
    echo "[setup] serving-eval venv 없음 — setup_env.sh 실행"
    bash "$REPO_ROOT/workspace/serving-eval/setup_env.sh"
fi

source "$SERVING_VENV/bin/activate"
cd "$REPO_ROOT"
mkdir -p logs results/serving-eval

MODEL=${1:-zamba2}
TRACE=${2:-smoke}          # smoke | sharegpt | longbench

echo "========================================================"
echo " Phase 1 S1.1: vLLM Serving + NVML Monitoring"
echo "   model   = $MODEL"
echo "   trace   = $TRACE"
echo "   job     = ${SLURM_JOB_ID:-local}"
echo "   GPU     = $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "   vLLM    = $(python -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo N/A)"
echo "   started = $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================"

# smoke trace 는 10 req 로 pipeline 검증 (HF 다운로드 불필요)
# sharegpt trace 는 실제 ShareGPT 데이터셋 사용 (HF Hub 접근 필요)
SMOKE_FLAG=""
if [ "$TRACE" = "smoke" ]; then
    SMOKE_FLAG="--smoke"
fi

echo ""
echo "── vLLM 서버 기동 + 부하 클라이언트 실행 …"
echo "   (서버 시작까지 최대 3분 소요 — 모델 로딩 시간 포함)"

python workspace/serving-eval/run_serving.py \
    --model     "$MODEL" \
    --trace     "$TRACE" \
    --monitor-nvml \
    --nvml-interval-ms 100 \
    --output-dir results/serving-eval \
    ${SMOKE_FLAG:+$SMOKE_FLAG}

# ── 출력 검증 ──────────────────────────────────────────────────────────────
echo ""
echo "── 출력 파일 검증 …"
PASS=0; FAIL=0

check_file() {
    local pattern="$1"; local desc="$2"
    local found; found=$(ls $pattern 2>/dev/null | head -1 || true)
    if [ -n "$found" ]; then
        echo "   OK  $desc → $found"
        PASS=$((PASS+1))
    else
        echo "   NG  $desc → $pattern 없음"
        FAIL=$((FAIL+1))
    fi
}

check_file "results/serving-eval/serving_results_${MODEL}_*.csv" "serving 결과 CSV"
check_file "results/serving-eval/nvml_${MODEL}_*.csv"            "NVML util 시계열 CSV"

# NVML CSV 에 실제 util 값이 있는지 확인
NVML_CSV=$(ls results/serving-eval/nvml_${MODEL}_*.csv 2>/dev/null | head -1 || true)
if [ -n "$NVML_CSV" ]; then
    ROWS=$(python -c "
import pandas as pd
df = pd.read_csv('$NVML_CSV')
print(len(df))
" 2>/dev/null || echo 0)
    if [ "$ROWS" -gt 5 ]; then
        echo "   OK  NVML CSV rows=$ROWS (device-level aggregate, NOT per-layer)"
        PASS=$((PASS+1))
    else
        echo "   NG  NVML CSV rows=$ROWS (너무 적음, NVML 수집 이상)"
        FAIL=$((FAIL+1))
    fi
fi

echo ""
echo "========================================================"
echo " S1.1 결과: PASS=$PASS  FAIL=$FAIL"
echo " NVML CSV → run_motivation_figures.sh 의 --nvml-dir 입력으로 사용"
[ "$FAIL" -eq 0 ] \
    && echo " ✓ 서빙 검증 완료" \
    || echo " ✗ 실패 항목 있음 — 위 로그 확인"
echo "========================================================"
[ "$FAIL" -eq 0 ]
