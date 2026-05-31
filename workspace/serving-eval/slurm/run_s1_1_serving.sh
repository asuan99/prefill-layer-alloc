#!/bin/bash
# =============================================================================
# run_s1_1_serving.sh — Phase 1 S1.1: vLLM 하이브리드 모델 서빙 + NVML 모니터링
#
# 검증 항목:
#   [1] vLLM 서버가 hybrid 모델로 기동 (Zamba2-7B / Nemotron-H-8B)
#   [2] ShareGPT / smoke trace 로 end-to-end 요청 처리 완료
#   [3] 서버 로그에 prefill/decode token 혼재 확인
#   [4] NVML device-level util 시계열 CSV 저장
#       → plot_motivation.py --nvml-dir 의 입력
#       → caption: "device-level aggregate (NVML ~10Hz), NOT per-layer"
#
# 하드웨어 2-tier 가드:
#   A100 → full run (numeric conclusions valid)
#   non-A100 → PIPELINE VALIDATION ONLY (10 req smoke)
#
# 모델 가중치 사전 조건:
#   HF Hub 또는 로컬 캐시에 모델이 있어야 함.
#   오프라인: HF_HUB_OFFLINE=1 + HF_HOME=<cache> 설정.
#   사전 다운로드: python -c "from huggingface_hub import snapshot_download; ..."
#
# Usage:
#   sbatch serving-eval/slurm/run_s1_1_serving.sh                      # zamba2 sharegpt
#   sbatch serving-eval/slurm/run_s1_1_serving.sh zamba2 smoke         # 검증용 smoke run
#   sbatch serving-eval/slurm/run_s1_1_serving.sh nemotron_h sharegpt  # Nemotron-H
#   sbatch serving-eval/slurm/run_s1_1_serving.sh zamba2 longbench     # long-context
#
# 환경 변수:
#   N_REQUESTS   요청 수 (default: 200, smoke 시 자동으로 10)
#   CONCURRENCY  동시 in-flight 요청 수 (default: 8)
#   HF_HOME      HuggingFace 캐시 경로 (default: ~/.cache/huggingface)
# =============================================================================
#SBATCH -J se-s1-serving
#SBATCH -p amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --comment=pytorch
#SBATCH -o /scratch/%u/whlee/prefill-layer-alloc/logs/se_serving_%j.log
#SBATCH -e /scratch/%u/whlee/prefill-layer-alloc/logs/se_serving_%j.err

set -euo pipefail

module load conda/pytorch_2.9.1_cuda13
module load cuda/13.0.2
module load gcc/15.2.0

REPO_ROOT="/scratch/$USER/whlee/prefill-layer-alloc"
VENV_DIR="$REPO_ROOT/workspace/serving-eval/.venv"

# venv 확인
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "[ERROR] serving-eval venv 없음."
    echo "        먼저 sbatch serving-eval/slurm/setup_env.sh 를 실행하십시오."
    exit 1
fi

source "$VENV_DIR/bin/activate"
cd "$REPO_ROOT"
mkdir -p logs results/serving-eval

MODEL=${1:-zamba2}
TRACE=${2:-sharegpt}          # smoke | sharegpt | longbench
N_REQUESTS=${N_REQUESTS:-200}
CONCURRENCY=${CONCURRENCY:-8}

# HuggingFace Hub 설정
# 클러스터 인터넷 차단 시: HF_HUB_OFFLINE=1 + HF_HOME=<공유 캐시>
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

echo "=================================================================="
echo " Phase 1 S1.1: vLLM Serving + NVML Monitoring"
echo "   model       = $MODEL"
echo "   trace       = $TRACE"
echo "   n_requests  = $N_REQUESTS  (smoke → auto 10)"
echo "   concurrency = $CONCURRENCY"
echo "   output_dir  = results/serving-eval"
echo "   job         = ${SLURM_JOB_ID:-local}"
echo "   GPU         = $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "   vLLM        = $(python -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo N/A)"
echo "   HF_HOME     = $HF_HOME"
echo "   started     = $(date '+%Y-%m-%d %H:%M:%S')"
echo "=================================================================="

# smoke 트레이스 전용 플래그
SMOKE_FLAG=""
[ "$TRACE" = "smoke" ] && SMOKE_FLAG="--smoke"

echo ""
echo "── [1/3] vLLM 서버 기동 + 부하 클라이언트 실행 …"
echo "   (첫 실행 시 모델 다운로드 포함 — GPU 가용 메모리 확인)"

python workspace/serving-eval/run_serving.py \
    --model         "$MODEL" \
    --trace         "$TRACE" \
    --n-requests    "$N_REQUESTS" \
    --concurrency   "$CONCURRENCY" \
    --monitor-nvml \
    --nvml-interval-ms 100 \
    --output-dir    results/serving-eval \
    --log-file      "results/serving-eval/vllm_${MODEL}_${TRACE}_${SLURM_JOB_ID:-local}.log" \
    ${SMOKE_FLAG:+$SMOKE_FLAG}

echo ""
echo "── [2/3] 출력 파일 확인 …"
PASS=0; FAIL=0

check_file() {
    local pattern="$1"; local desc="$2"
    local found; found=$(ls $pattern 2>/dev/null | head -1 || true)
    if [ -n "$found" ]; then
        echo "   OK  $desc"
        echo "       → $found"
        PASS=$((PASS+1))
    else
        echo "   NG  $desc ($pattern 없음)"
        FAIL=$((FAIL+1))
    fi
}

TIER="$TRACE"
[ "$TRACE" != "smoke" ] && nvidia-smi --query-gpu=name --format=csv,noheader \
    | grep -iq "a100" || TIER="smoke"  # non-A100 는 smoke 로 저장됨

check_file "results/serving-eval/serving_${MODEL}_${TIER}.json" "serving 결과 JSON"
check_file "results/serving-eval/nvml_${MODEL}_${TIER}.csv"    "NVML util 시계열 CSV"
check_file "results/serving-eval/vllm_${MODEL}_*.log"          "vLLM 서버 로그"

# vLLM 로그에서 prefill/decode 혼재 확인
echo ""
echo "── [3/3] 서버 로그에서 prefill/decode 혼재 확인 …"
VLLM_LOG=$(ls results/serving-eval/vllm_${MODEL}_*.log 2>/dev/null | head -1 || true)
if [ -n "$VLLM_LOG" ]; then
    PREFILL_LINES=$(grep -c "prompt_tokens\|num_prompt_tokens" "$VLLM_LOG" 2>/dev/null || echo 0)
    DECODE_LINES=$(grep -c "output_tokens\|num_generated_tokens" "$VLLM_LOG" 2>/dev/null || echo 0)
    echo "   prefill 관련 로그: $PREFILL_LINES 줄"
    echo "   decode  관련 로그: $DECODE_LINES 줄"
    if [ "$PREFILL_LINES" -gt 0 ] || [ "$DECODE_LINES" -gt 0 ]; then
        echo "   OK: prefill/decode 혼재 확인됨"
        PASS=$((PASS+1))
    else
        echo "   INFO: 로그에서 prefill/decode 패턴 미발견 (vLLM 버전에 따라 다름)"
    fi
else
    echo "   [warn] vLLM 로그 파일 없음"
fi

# NVML 샘플 수 확인
NVML_CSV=$(ls results/serving-eval/nvml_${MODEL}_*.csv 2>/dev/null | head -1 || true)
if [ -n "$NVML_CSV" ]; then
    ROWS=$(python -c "import pandas as pd; print(len(pd.read_csv('$NVML_CSV')))" 2>/dev/null || echo 0)
    echo ""
    echo "   NVML samples = $ROWS  (device-level aggregate, NOT per-layer)"
    [ "$ROWS" -gt 5 ] && { echo "   OK"; PASS=$((PASS+1)); } \
                       || { echo "   NG: NVML 샘플 수 부족"; FAIL=$((FAIL+1)); }
fi

echo ""
echo "=================================================================="
echo " S1.1 결과: PASS=$PASS  FAIL=$FAIL"
echo " NVML CSV → run_s1_2_motivation.sh --nvml-dir results/serving-eval"
[ "$FAIL" -eq 0 ] \
    && echo " ✓ 서빙 검증 완료 — motivation figure 생성으로 진행" \
    || echo " ✗ 실패 항목 있음 — 위 로그 확인"
echo "=================================================================="
[ "$FAIL" -eq 0 ]
