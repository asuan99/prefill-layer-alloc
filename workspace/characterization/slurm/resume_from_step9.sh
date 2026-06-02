#!/bin/bash
# =============================================================================
# resume_from_step9.sh — job 746350 중단 지점부터 재실행
#
# 중단 상황:
#   [9/11] compute_decision_matrix.py — numpy int64 JSON 직렬화 오류로 크래시
#   → decision_matrix.json 파손 (859 bytes, 불완전)
#   → Stage 3 (steps 10-11) 미실행
#
# 오류 원인:
#   load_stage1_saturation() 에서 saturation threshold 미발견 시
#   sat_sm = total_sm (numpy int64) 이 dict 에 그대로 들어가
#   json.dump 에서 TypeError: int64 is not JSON serializable 발생.
#
# 수정 내용 (compute_decision_matrix.py):
#   1. sat_sm = int(total_sm)  — 기본값 명시적 int 변환
#   2. json.dump(..., cls=_NumpyEncoder)  — numpy 타입 안전 encoder 추가
#
# 재실행 범위:
#   [9/11] Decision matrix          (compute_decision_matrix.py)
#   [10/11] S2.2 Co-exec microbench (coexec_microbench.py)
#   [11/11] S2.3 Projection + plots (projection_analysis.py + plot_results.py)
#
# 선행 조건 (완료 확인 후 실행):
#   ✅ Stage 1 — results/stage1/{ssm,attn,mlp,chunked}/*.csv
#   ✅ Stage 2 layer latency  — results/stage2/layer_latency_*.csv
#   ✅ Stage 2 ctx_switch     — results/stage2/ctx_switch_overhead_*.json
#
# Usage:
#   sbatch slurm/resume_from_step9.sh                    # zamba2
#   sbatch slurm/resume_from_step9.sh all                # zamba2 + falcon_h1
#   sbatch slurm/resume_from_step9.sh zamba2 a100_80gb
# =============================================================================
#SBATCH -J resume-step9
#SBATCH -p amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --comment=pytorch
#SBATCH -o /scratch/%u/whlee/prefill-layer-alloc/logs/resume_step9_%j.log
#SBATCH -e /scratch/%u/whlee/prefill-layer-alloc/logs/resume_step9_%j.err

set -euo pipefail

module load conda/pytorch_2.9.1_cuda13
module load cuda/13.0.2
module load gcc/15.2.0

# 절대 경로 고정 — dirname "$0" 방식은 sbatch 에서 $0 이 스크립트명만
# 반환될 경우 $HOME 을 기준으로 해석돼 Permission Denied 발생.
# 다른 slurm 스크립트(run_stage1.sh 등)와 동일한 패턴 사용.
CHAR_DIR="/scratch/$USER/whlee/prefill-layer-alloc/workspace/characterization"
source "/scratch/$USER/whlee/prefill-layer-alloc/bin/activate"
cd "$CHAR_DIR"
mkdir -p logs results/stage2 results/stage3

MODEL=${1:-zamba2}
DEVICE=${2:-auto}

_T0=$(date +%s)
_elapsed() { local t=$(( $(date +%s) - _T0 )); printf '%dm%02ds' $(( t/60 )) $(( t%60 )); }

echo "========================================================"
echo " RESUME from step 9 (Decision matrix)"
echo "   original job = 746350"
echo "   model        = $MODEL"
echo "   device       = $DEVICE"
echo "   job          = ${SLURM_JOB_ID:-local}"
echo "   GPU          = $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "   started      = $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================"

# ── 선행 조건 검증 ─────────────────────────────────────────────────────────
echo ""
echo "── 선행 조건 검증 …"
PREREQ_FAIL=0

_check_prereq() {
    local pattern="$1" desc="$2"
    if ls $pattern &>/dev/null; then
        echo "   ✅ $desc"
    else
        echo "   ❌ $desc — 없음 ($pattern)"
        PREREQ_FAIL=$((PREREQ_FAIL+1))
    fi
}

_check_prereq "results/stage1/ssm_scaling_*.csv"       "Stage 1 SSM scaling CSV"
_check_prereq "results/stage1/chunked/ssm_chunked_*.csv" "Stage 1 chunked SSM CSV"
_check_prereq "results/stage2/layer_latency_*.csv"     "Stage 2 layer latency CSV"
_check_prereq "results/stage2/ctx_switch_overhead_*.json" "Stage 2 ctx_switch JSON"

if [ "$PREREQ_FAIL" -gt 0 ]; then
    echo ""
    echo "  [ERROR] $PREREQ_FAIL 선행 조건 미충족."
    echo "  Stage 1/2 를 먼저 완료하십시오:"
    echo "    sbatch slurm/run_stage1.sh $MODEL"
    echo "    sbatch slurm/run_stage2.sh $MODEL"
    exit 1
fi

# decision_matrix.json 상태 확인
DM_JSON="results/stage2/decision_matrix.json"
if [ -f "$DM_JSON" ]; then
    if python -c "import json; json.load(open('$DM_JSON'))" 2>/dev/null; then
        DM_STATUS="valid"
        echo "   ✅ decision_matrix.json (valid — 덮어씀)"
    else
        DM_STATUS="corrupt"
        echo "   ⚠️  decision_matrix.json (파손 — 재생성)"
    fi
else
    DM_STATUS="missing"
    echo "   ⚠️  decision_matrix.json (없음 — 생성)"
fi

# ── [STEP 9] Decision matrix 재실행 ───────────────────────────────────────
echo ""
echo "── [9/11] $(date '+%H:%M:%S')  Decision matrix (재실행) …"
echo "   수정 내용: sat_sm = int(total_sm) + _NumpyEncoder"

# decision_matrix 는 model-independent (stage1 전체를 읽음)
srun --overlap --ntasks=1 python stage2_overhead/compute_decision_matrix.py \
    --stage1-dir results/stage1 \
    --stage2-dir results/stage2

# JSON 유효성 확인
if python -c "import json; d=json.load(open('$DM_JSON')); print(f'  dominant={d[\"dominant_strategy\"]}  rows={len(d[\"rows\"])}')"; then
    echo "   ✅ decision_matrix.json 재생성 성공"
else
    echo "   ❌ decision_matrix.json 여전히 파손 — 로그 확인"
    exit 1
fi
echo "   done  ($(_elapsed))"

# ── [STEP 10] S2.2: Co-execution microbench ───────────────────────────────
echo ""
echo "── [10/11] $(date '+%H:%M:%S')  S2.2: Co-exec microbench …"

_MODEL_LIST="zamba2"
[ "$MODEL" = "all" ] && _MODEL_LIST="zamba2 falcon_h1"

for _M in $_MODEL_LIST; do
    echo "   model: $_M"
    srun --overlap --ntasks=1 python stage3_hm_eval/coexec_microbench.py \
        --model        "$_M" \
        --device       "$DEVICE" \
        --sm-splits    0.3 0.4 0.5 0.6 0.7 \
        --seq-lens     1024 2048 4096 \
        --batch-sizes  1 4 \
        --context-lens 1024 4096 \
        --n-warmup     5 \
        --n-measure    20 \
        --output-dir   results/stage3
done
echo "   done  ($(_elapsed))"

# ── [STEP 11] S2.3: Projection + plots ───────────────────────────────────
echo ""
echo "── [11/11] $(date '+%H:%M:%S')  S2.3: Projection analysis + plots …"

for _M in $_MODEL_LIST; do
    echo "   model: $_M"

    FREE_SM_CSV=$(ls results/stage1/free_sm_zone_${_M}_*.csv 2>/dev/null \
                  | sort | tail -1 || true)
    COEXEC_CSV=$(ls  results/stage3/coexec_${_M}_*.csv 2>/dev/null \
                  | sort | tail -1 || true)

    PROJ_ARGS=(--model "$_M" --output-dir results/stage3)
    [ -n "$FREE_SM_CSV" ] && PROJ_ARGS+=(--free-sm-csv "$FREE_SM_CSV")
    [ -n "$COEXEC_CSV"  ] && PROJ_ARGS+=(--coexec-csv  "$COEXEC_CSV")
    [ -f "$DM_JSON" ]     && PROJ_ARGS+=(--stage2-json  "$DM_JSON")
    [[ -z "$FREE_SM_CSV" || -z "$COEXEC_CSV" ]] && PROJ_ARGS+=(--demo)

    srun --overlap --ntasks=1 python stage3_hm_eval/projection_analysis.py \
        "${PROJ_ARGS[@]}" \
        || echo "   [warn] projection_analysis.py 실패 — 계속"

    srun --overlap --ntasks=1 python stage3_hm_eval/plot_results.py \
        --results-dir results/stage3 --model "$_M" 2>/dev/null \
        || echo "   [warn] plot_results.py 실패 — 계속"
done
echo "   done  ($(_elapsed))"

# ── 최종 검증 ──────────────────────────────────────────────────────────────
echo ""
echo "── 출력 검증 …"
PASS=0; FAIL=0

_chk() {
    local p="$1" d="$2"
    if ls $p &>/dev/null; then
        echo "   ✅ $d"; PASS=$((PASS+1))
    else
        echo "   ❌ $d ($p)"; FAIL=$((FAIL+1))
    fi
}

_chk "results/stage2/decision_matrix.json"    "decision_matrix.json"
_chk "results/stage2/decision_matrix.html"    "decision_matrix.html"
_chk "results/stage3/coexec_*.csv"            "coexec sweep CSV"
_chk "results/stage3/coexec_*.png"            "overlap_ratio figure"
_chk "results/stage3/projection_*.png"        "PROJECTION figure"
_chk "results/stage3/projection_report_*.txt" "PROJECTION report"

echo ""
echo "========================================================"
echo " Resume 결과: PASS=$PASS  FAIL=$FAIL  elapsed=$(_elapsed)"
[ "$FAIL" -eq 0 ] \
    && echo " ✓ 재개 완료 — check_gate_g1.sh 로 G1 게이트 판정 진행" \
    || echo " ✗ 실패 항목 있음 — 위 로그 확인"
echo "========================================================"
echo ""
echo " 다음 단계:"
echo "   bash slurm/check_gate_g1.sh $MODEL"
[ "$FAIL" -eq 0 ]
