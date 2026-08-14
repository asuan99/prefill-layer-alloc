#!/bin/bash
# =============================================================================
# run_stage3.sh — Phase 2: 실측 concurrent execution + PROJECTION 분석
#
# ※ DEPRECATED 내용:
#   이 스크립트는 이전에 run_concurrent_eval.py (순차 시뮬레이션) 를 호출했으나,
#   해당 스크립트가 다음 사유로 폐기됨:
#     1. decode seq_len=1 SSM — KV read 없어 Problem 4 전제 자기 부정
#     2. TTFT = wall-clock 합성값 (CUDA event 측정치 아님)
#     3. 단일 스트림 순차 실행 — spatial sharing 없음
#   보관: stage3_hm_eval/deprecated/run_concurrent_eval.py
#
# 현재 Phase 2 파이프라인:
#   [1] S2.2: coexec_microbench.py  — 진짜 동시 실행, overlap_ratio 실측
#   [2] S2.3: projection_analysis.py — PROJECTION 분석 (PROJECTION 명시 필수)
#
# Usage:
#   sbatch slurm/run_stage3.sh                # zamba2
#   sbatch slurm/run_stage3.sh falcon_h1
#   sbatch slurm/run_stage3.sh zamba2 a100_80gb
#
# 선행 조건: run_stage1.sh + run_stage1_decode_sweep.sh + run_stage2.sh 완료
# =============================================================================
#SBATCH -J phase2-stage3
#SBATCH -p amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --comment="field=efficientai;appl=pytorch"
#SBATCH -o /scratch/%u/whlee/prefill-layer-alloc/logs/stage3_%j.log
#SBATCH -e /scratch/%u/whlee/prefill-layer-alloc/logs/stage3_%j.err

set -euo pipefail

module load conda/pytorch_2.9.1_cuda13
module load cuda/13.0.2
module load gcc/15.2.0

source /scratch/$USER/whlee/prefill-layer-alloc/bin/activate
cd /scratch/$USER/whlee/prefill-layer-alloc
mkdir -p logs results/stage3

MODEL=${1:-zamba2}
DEVICE=${2:-auto}

echo "========================================================"
echo " Phase 2 Stage 3: Co-Exec + Projection"
echo "   model   = $MODEL"
echo "   device  = $DEVICE"
echo "   job     = ${SLURM_JOB_ID:-local}"
echo "   GPU     = $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "   started = $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================"

_T0=$(date +%s)
_elapsed() { local t=$(( $(date +%s) - _T0 )); printf '%dm%02ds' $(( t/60 )) $(( t%60 )); }

# ── [1/3] S2.2: Real concurrent execution sweep ──────────────────────────
echo ""
echo "── [1/3] $(date '+%H:%M:%S')  S2.2: Co-execution microbench …"
python workspace/characterization/stage3_hm_eval/coexec_microbench.py \
    --model        "$MODEL" \
    --device       "$DEVICE" \
    --sm-splits    0.3 0.4 0.5 0.6 0.7 \
    --seq-lens     1024 2048 4096 \
    --batch-sizes  1 4 \
    --context-lens 1024 4096 \
    --n-warmup     5 \
    --n-measure    20 \
    --output-dir   results/stage3
echo "   done  ($(_elapsed))"

# ── [2/3] S2.3: PROJECTION 분석 ──────────────────────────────────────────
echo ""
echo "── [2/3] $(date '+%H:%M:%S')  S2.3: Projection analysis …"

FREE_SM_CSV=$(ls results/stage1/free_sm_zone_${MODEL}_*.csv 2>/dev/null \
             | sort | tail -1 || true)
COEXEC_CSV=$(ls results/stage3/coexec_${MODEL}_*.csv 2>/dev/null \
             | sort | tail -1 || true)
STAGE2_JSON="results/stage2/decision_matrix.json"

PROJ_ARGS=(--model "$MODEL" --output-dir results/stage3)
[ -n "$FREE_SM_CSV" ]         && PROJ_ARGS+=(--free-sm-csv "$FREE_SM_CSV")
[ -n "$COEXEC_CSV" ]          && PROJ_ARGS+=(--coexec-csv  "$COEXEC_CSV")
[ -f "$STAGE2_JSON" ]         && PROJ_ARGS+=(--stage2-json "$STAGE2_JSON")

# 입력 없으면 demo fallback
if [ -z "$FREE_SM_CSV" ] || [ -z "$COEXEC_CSV" ]; then
    echo "   [warn] free_sm / coexec CSV 없음 → --demo fallback"
    PROJ_ARGS+=(--demo)
fi

python workspace/characterization/stage3_hm_eval/projection_analysis.py \
    "${PROJ_ARGS[@]}"
echo "   done  ($(_elapsed))"

# ── [3/3] SM timeline plot (coexec CSV 기반) ─────────────────────────────
echo ""
echo "── [3/3] $(date '+%H:%M:%S')  plot_results.py (SM timeline) …"
python workspace/characterization/stage3_hm_eval/plot_results.py \
    --results-dir results/stage3 \
    --model "$MODEL" 2>/dev/null \
    || echo "   [warn] plot_results.py 비정상 종료 — 계속"
echo "   done  ($(_elapsed))"

echo ""
echo "========================================================"
echo " Phase 2 Stage 3 Done  (results → results/stage3/)"
echo "   elapsed = $(_elapsed)"
echo ""
echo " 출력:"
ls -lh results/stage3/coexec_${MODEL}_*.{csv,png} \
        results/stage3/projection_*_${MODEL}.{png,txt} 2>/dev/null | head -20 || true
echo "========================================================"
