#!/bin/bash
# =============================================================================
# run_stage3_projection.sh — Phase 2 S2.3: PROJECTION 분석
#
# 실측 불가능한 end-to-end 이득을, 측정값(Stage1 + Stage2 + S2.2)으로
# projection 한다. 모든 출력에 "PROJECTION" 이 하드코딩됨.
#
# 입력 (자동 탐색):
#   results/stage1/free_sm_zone_{model}_{device}.csv   (S1.3)
#   results/stage3/coexec_{model}_{device}.csv          (S2.2)
#   results/stage2/decision_matrix.json                 (Stage 2)
#
# 검증 기준:
#   - 모든 figure 파일명에 "projection_" 접두사
#   - 모든 figure 제목에 "PROJECTION" 문자열
#   - projection_report_{model}.txt 에 가정(A1-A5) 목록 포함
#   - measured 데이터와 혼재 없음 (별도 figure 파일)
#
# Usage:
#   # 자동 탐색 (Stage 1/2/3 결과가 results/ 에 있을 때)
#   sbatch slurm/run_stage3_projection.sh
#   sbatch slurm/run_stage3_projection.sh falcon_h1
#
#   # 파일 명시
#   sbatch slurm/run_stage3_projection.sh zamba2 \
#       results/stage1/free_sm_zone_zamba2_a100.csv \
#       results/stage3/coexec_zamba2_a100.csv \
#       results/stage2/decision_matrix.json
#
#   # demo (실측 데이터 없이 렌더링 검증)
#   sbatch slurm/run_stage3_projection.sh --demo
# =============================================================================
#SBATCH -J s3-projection
#SBATCH -p amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0
#SBATCH --time=00:10:00
#SBATCH --comment=pytorch
#SBATCH -o /scratch/%u/whlee/prefill-layer-alloc/logs/s3_projection_%j.log
#SBATCH -e /scratch/%u/whlee/prefill-layer-alloc/logs/s3_projection_%j.err

set -euo pipefail

module load conda/pytorch_2.9.1_cuda13 2>/dev/null || true
module load gcc/15.2.0                 2>/dev/null || true

source /scratch/$USER/whlee/prefill-layer-alloc/bin/activate
cd /scratch/$USER/whlee/prefill-layer-alloc
mkdir -p logs results/stage3

# demo 모드 처리
if [[ "${1:-}" == "--demo" ]]; then
    echo "========================================================"
    echo " Phase 2 S2.3: Projection Analysis [DEMO MODE]"
    echo "========================================================"
    python workspace/characterization/stage3_hm_eval/projection_analysis.py \
        --demo --output-dir results/stage3
    echo ""
    ls -lh results/stage3/projection_*.png results/stage3/projection_*.txt 2>/dev/null || true
    exit 0
fi

MODEL=${1:-zamba2}

# 입력 파일 자동 탐색
FREE_SM_CSV=${2:-$(ls results/stage1/free_sm_zone_${MODEL}_*.csv 2>/dev/null \
                   | sort | tail -1 || true)}
COEXEC_CSV=${3:-$(ls results/stage3/coexec_${MODEL}_*.csv 2>/dev/null \
                  | sort | tail -1 || true)}
STAGE2_JSON=${4:-results/stage2/decision_matrix.json}

echo "========================================================"
echo " Phase 2 S2.3: PROJECTION Analysis"
echo "   model        = $MODEL"
echo "   free_sm_csv  = ${FREE_SM_CSV:-[자동 탐색 실패]}"
echo "   coexec_csv   = ${COEXEC_CSV:-[자동 탐색 실패]}"
echo "   stage2_json  = $STAGE2_JSON"
echo "   job          = ${SLURM_JOB_ID:-local}"
echo "   started      = $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================"
echo ""
echo "  ※ 모든 출력은 PROJECTION 이며 측정값이 아닙니다."
echo ""

# 필수 입력 확인
MISSING=0
for f in "$FREE_SM_CSV" "$COEXEC_CSV"; do
    if [ -z "$f" ] || [ ! -f "$f" ]; then
        echo "  [WARN] 입력 없음: $f"
        MISSING=$((MISSING+1))
    fi
done

if [ "$MISSING" -gt 0 ]; then
    echo ""
    echo "  일부 입력이 없어 demo 데이터로 fallback 합니다."
    echo "  실측 데이터 사용: run_stage1_decode_sweep.sh + run_stage3_coexec.sh 먼저 실행"
    python workspace/characterization/stage3_hm_eval/projection_analysis.py \
        --demo --model "$MODEL" --output-dir results/stage3
else
    ARGS=(
        --model       "$MODEL"
        --free-sm-csv "$FREE_SM_CSV"
        --coexec-csv  "$COEXEC_CSV"
        --output-dir  results/stage3
    )
    [ -f "$STAGE2_JSON" ] && ARGS+=(--stage2-json "$STAGE2_JSON")

    python workspace/characterization/stage3_hm_eval/projection_analysis.py \
        "${ARGS[@]}"
fi

# ── 검증 ────────────────────────────────────────────────────────────────────
echo ""
echo "── 출력 검증 …"
PASS=0; FAIL=0

check_file() {
    local pattern="$1"; local desc="$2"
    local found; found=$(ls $pattern 2>/dev/null | head -1 || true)
    if [ -n "$found" ]; then
        echo "   OK  $desc → $found"
        PASS=$((PASS+1))
    else
        echo "   NG  $desc 없음  ($pattern)"
        FAIL=$((FAIL+1))
    fi
}

check_file "results/stage3/projection_speedup_${MODEL}.png"  "PROJECTION speedup figure"
check_file "results/stage3/projection_net_gain_${MODEL}.png" "PROJECTION net-gain figure"
check_file "results/stage3/projection_report_${MODEL}.txt"   "PROJECTION report"

# report 에 가정 목록 포함 확인
REPORT="results/stage3/projection_report_${MODEL}.txt"
if [ -f "$REPORT" ]; then
    if grep -q "Assumption A1" "$REPORT" && grep -q "PROJECTION" "$REPORT"; then
        echo "   OK  report 에 PROJECTION + 가정(A1-A5) 포함"
        PASS=$((PASS+1))
    else
        echo "   NG  report 에 PROJECTION 또는 가정 목록 없음"
        FAIL=$((FAIL+1))
    fi
fi

# figure 파일명 PROJECTION 접두사 확인
PROJ_PNGS=$(ls results/stage3/projection_*.png 2>/dev/null | wc -l)
if [ "$PROJ_PNGS" -ge 2 ]; then
    echo "   OK  projection_*.png $PROJ_PNGS 개 (파일명에 PROJECTION 접두사)"
    PASS=$((PASS+1))
else
    echo "   NG  projection_*.png < 2"
    FAIL=$((FAIL+1))
fi

echo ""
echo "========================================================"
echo " S2.3 결과: PASS=$PASS  FAIL=$FAIL"
[ "$FAIL" -eq 0 ] \
    && echo " ✓ Projection 분석 완료 (모든 출력에 PROJECTION 명시됨)" \
    || echo " ✗ 실패 항목 있음 — 위 로그 확인"
echo "========================================================"
echo ""
echo " 출력 파일:"
ls -lh results/stage3/projection_* 2>/dev/null || true
[ "$FAIL" -eq 0 ]
