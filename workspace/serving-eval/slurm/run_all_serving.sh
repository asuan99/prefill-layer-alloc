#!/bin/bash
# =============================================================================
# run_all_serving.sh — serving-eval 전체 파이프라인
#
# S1.1 (vLLM 서빙) → S1.2 (motivation figures) 순차 실행.
# 두 단계의 실행 환경 차이를 처리한다:
#   S1.1: serving-eval venv (vLLM 필요)
#   S1.2: serving-eval venv (matplotlib/pandas)
#
# GPU 사용:
#   S1.1: GPU 필요 (vLLM 모델 로딩)
#   S1.2: GPU 불필요 (CPU 전용)
#   → 한 잡에서 순차 실행 (GPU 할당 유지하되 S1.2 는 CPU 로 실행)
#
# 선행 조건:
#   1. serving-eval venv 설치 완료 (setup_env.sh)
#   2. 모델 가중치 HF Hub 또는 로컬 캐시에 존재
#   3. Stage 1 결과 (results/stage1/) 가 있으면 Fig B/C 포함됨
#      없으면 Fig B/C 를 demo 데이터로 생성
#
# Usage:
#   sbatch serving-eval/slurm/run_all_serving.sh                       # zamba2 sharegpt
#   sbatch serving-eval/slurm/run_all_serving.sh zamba2 smoke          # 검증용
#   sbatch serving-eval/slurm/run_all_serving.sh nemotron_h sharegpt
#   SKIP_S1_2=1 sbatch serving-eval/slurm/run_all_serving.sh zamba2   # S1.1 만 실행
#
# 환경 변수:
#   SKIP_S1_2   "1" 이면 S1.2 figure 생성 건너뜀
#   N_REQUESTS  요청 수 (default: 200)
#   CONCURRENCY 동시 요청 수 (default: 8)
#   HF_HOME     HuggingFace 캐시 경로
# =============================================================================
#SBATCH -J se-all
#SBATCH -p amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --comment=pytorch
#SBATCH -o /scratch/%u/whlee/prefill-layer-alloc/logs/se_all_%j.log
#SBATCH -e /scratch/%u/whlee/prefill-layer-alloc/logs/se_all_%j.err

set -euo pipefail

module load conda/pytorch_2.9.1_cuda13
module load cuda/13.0.2
module load gcc/15.2.0

REPO_ROOT="/scratch/$USER/whlee/prefill-layer-alloc"
VENV_DIR="$REPO_ROOT/workspace/serving-eval/.venv"

if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "[ERROR] serving-eval venv 없음."
    echo "        먼저 sbatch serving-eval/slurm/setup_env.sh 를 실행하십시오."
    exit 1
fi

source "$VENV_DIR/bin/activate"
cd "$REPO_ROOT"
mkdir -p logs results/serving-eval results/motivation

MODEL=${1:-zamba2}
TRACE=${2:-sharegpt}
N_REQUESTS=${N_REQUESTS:-200}
CONCURRENCY=${CONCURRENCY:-8}
SKIP_S1_2=${SKIP_S1_2:-0}

# HuggingFace — 모델 가중치는 로컬 캐시 사용, 오프라인 모드
# 데이터셋은 $REPO_ROOT/hf_cache 에 사전 캐싱된 것을 사용
# (사전 다운로드: bash workspace/serving-eval/slurm/download_datasets.sh)
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$REPO_ROOT/hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

# Stage 1 결과 경로 (characterization sweep 결과 위치)
CHAR_STAGE1="$REPO_ROOT/workspace/characterization/results/stage1"

_T0=$(date +%s)
_elapsed() { local t=$(( $(date +%s) - _T0 )); printf '%dm%02ds' $(( t/60 )) $(( t%60 )); }

echo "╔══════════════════════════════════════════════════════════╗"
printf "║  serving-eval pipeline  %-30s  ║\n" "$MODEL / $TRACE"
echo "╚══════════════════════════════════════════════════════════╝"
printf "  model      = %s\n" "$MODEL"
printf "  trace      = %s\n" "$TRACE"
printf "  n_requests = %s\n" "$N_REQUESTS"
printf "  GPU        = %s\n" "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
printf "  vLLM       = %s\n" "$(python -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo N/A)"
printf "  job        = %s\n" "${SLURM_JOB_ID:-local}"
printf "  started    = %s\n" "$(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# ── S1.1: vLLM 서빙 + NVML 모니터링 ──────────────────────────────────────
echo "── [S1.1] $(date '+%H:%M:%S')  vLLM serving + NVML …"

SMOKE_FLAG=""
[ "$TRACE" = "smoke" ] && SMOKE_FLAG="--smoke"

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

echo "  S1.1 done  ($(_elapsed))"

# ── S1.2: motivation figures ───────────────────────────────────────────────
if [ "$SKIP_S1_2" = "1" ]; then
    echo ""
    echo "── [S1.2] 건너뜀 (SKIP_S1_2=1)"
else
    echo ""
    echo "── [S1.2] $(date '+%H:%M:%S')  motivation figures …"

    # Stage 1 결과 자동 탐색 (characterization sweep 결과 디렉토리)
    STAGE1_CHUNKED=$(ls "$CHAR_STAGE1/chunked/ssm_chunked_${MODEL}_"*.csv 2>/dev/null \
                     | sort | tail -1 || true)
    STAGE1_ATTN=$(ls "$CHAR_STAGE1/attn_scaling_${MODEL}_"*.csv 2>/dev/null \
                  | sort | tail -1 || true)

    if [ -n "$STAGE1_CHUNKED" ] || [ -n "$STAGE1_ATTN" ]; then
        echo "  Stage 1 데이터 발견: $CHAR_STAGE1"
        echo "  → Fig B (free-SM zone) + Fig C (roofline) 실측 데이터 사용"
        STAGE1_ARG="--stage1-dir $CHAR_STAGE1"
    else
        echo "  Stage 1 데이터 없음 ($CHAR_STAGE1)"
        echo "  → Fig B/C demo 데이터 사용"
        echo "  (실측: sbatch workspace/characterization/slurm/run_stage1.sh $MODEL)"
        STAGE1_ARG=""
    fi

    python workspace/serving-eval/plot_motivation.py \
        --hybrid-model "$MODEL" \
        --nvml-dir     results/serving-eval \
        ${STAGE1_ARG:+$STAGE1_ARG} \
        --output-dir   results/motivation \
        || {
            echo "  [warn] plot_motivation.py 실패 → --demo fallback"
            python workspace/serving-eval/plot_motivation.py \
                --demo --output-dir results/motivation
        }

    echo "  S1.2 done  ($(_elapsed))"
fi

# ── 최종 요약 ──────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  serving-eval complete                                   ║"
echo "╚══════════════════════════════════════════════════════════╝"
printf "  elapsed = %s\n" "$(_elapsed)"
echo ""
echo "  출력 파일:"
echo "  S1.1:"
ls -lh results/serving-eval/nvml_${MODEL}_*.csv \
        results/serving-eval/serving_${MODEL}_*.json 2>/dev/null \
    | sed 's/^/    /' || echo "    (없음)"
echo "  S1.2:"
ls -lh results/motivation/fig_*.png 2>/dev/null \
    | sed 's/^/    /' || echo "    (없음)"
echo ""
echo "  다음 단계:"
echo "    characterization/slurm/check_gate_g1.sh $MODEL   (G1 게이트 판정)"
