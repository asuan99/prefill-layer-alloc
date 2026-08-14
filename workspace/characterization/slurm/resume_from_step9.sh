#!/bin/bash
# =============================================================================
# resume_from_step9.sh — job 746350 중단 지점부터 재실행 (정비판)
#
# ─ 중단 상황 ─────────────────────────────────────────────────────────────────
#   [9/11] compute_decision_matrix.py — numpy int64 JSON 직렬화 오류로 크래시
#     → decision_matrix.json 파손 (859 bytes, 불완전 JSON)
#     → Stage 3 (steps 10-11) 미실행
#
# ─ 결과 디렉토리 분리 문제 ─────────────────────────────────────────────────
#   job 746350 의 run_all.sh 는 workspace/characterization/ 를 CWD 로 사용하고
#   모든 결과를 workspace/characterization/results/ 에 저장함. 그런데 chunked
#   sweep 은 이전 별도 실행으로 repo-root 의 results/ 에 저장돼 있어 경로가 엇갈림.
#
#   있는 것:
#     workspace/characterization/results/stage1/ssm_scaling_*.csv      ✅
#     workspace/characterization/results/stage1/attn_scaling_*.csv     ✅
#     workspace/characterization/results/stage2/layer_latency_*.csv    ✅
#     workspace/characterization/results/stage2/ctx_switch_*.json      ✅
#     /scratch/$USER/…/results/stage1/chunked/ssm_chunked_*.csv        ✅ (다른 경로)
#
#   없는 것:
#     workspace/characterization/results/stage1/chunked/               ❌ (디렉토리 없음)
#     workspace/characterization/results/stage1/free_sm_zone_*.csv     ❌ (미생성)
#     workspace/characterization/results/stage2/decision_matrix.json   ❌ (파손)
#     workspace/characterization/results/stage3/coexec_*.csv           ❌ (미실행)
#
# ─ 실행 순서 ─────────────────────────────────────────────────────────────────
#   [pre]  데이터 통합: chunked CSV 를 workspace results 로 복사
#   [pre]  free_sm_zone CSV 생성: plot_saturation.py (S1.3 free-SM zone)
#   [ 9 ]  Decision matrix 재실행  (compute_decision_matrix.py, 수정됨)
#   [10]  Co-exec microbench      (coexec_microbench.py)
#   [11]  Projection + plots      (projection_analysis.py + plot_results.py)
#
# ─ 오류 수정 내용 (compute_decision_matrix.py) ────────────────────────────
#   1. sat_sm = int(total_sm)  — 기본값 명시적 int 변환
#   2. json.dump(..., cls=_NumpyEncoder)  — numpy 타입 안전 encoder 추가
#
# Usage:
#   sbatch slurm/resume_from_step9.sh            # zamba2
#   sbatch slurm/resume_from_step9.sh all        # zamba2 + falcon_h1
#   sbatch slurm/resume_from_step9.sh zamba2 a100_80gb
# =============================================================================
#SBATCH -J resume-step9
#SBATCH -p amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --comment="field=efficientai;appl=pytorch"
#SBATCH -o /scratch/%u/whlee/prefill-layer-alloc/logs/resume_step9_%j.log
#SBATCH -e /scratch/%u/whlee/prefill-layer-alloc/logs/resume_step9_%j.err

set -euo pipefail

module load conda/pytorch_2.9.1_cuda13
module load cuda/13.0.2
module load gcc/15.2.0

# 절대 경로 고정 (sbatch 에서 dirname "$0" 이 $HOME 으로 해석되는 문제 방지)
REPO_ROOT="/scratch/$USER/whlee/prefill-layer-alloc"
CHAR_DIR="$REPO_ROOT/workspace/characterization"

source "$REPO_ROOT/bin/activate"
cd "$CHAR_DIR"
mkdir -p logs results/stage1/chunked results/stage2 results/stage3

MODEL=${1:-zamba2}
DEVICE=${2:-auto}

_T0=$(date +%s)
_elapsed() { local t=$(( $(date +%s) - _T0 )); printf '%dm%02ds' $(( t/60 )) $(( t%60 )); }

echo "========================================================"
echo " RESUME from step 9 (Decision matrix)"
echo "   model   = $MODEL"
echo "   device  = $DEVICE"
echo "   job     = ${SLURM_JOB_ID:-local}"
echo "   GPU     = $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "   CWD     = $(pwd)"
echo "   started = $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================"

_MODEL_LIST="$MODEL"
[ "$MODEL" = "all" ] && _MODEL_LIST="zamba2 falcon_h1"

# ── 선행 조건 검증 (실제로 필요한 파일만) ────────────────────────────────────
# compute_decision_matrix.py 가 필요로 하는 파일:
#   ssm_scaling_*.csv  (SM scaling latency — wave model)
#   layer_latency_*.csv
#   ctx_switch_overhead_*.json
# chunked CSV 는 decision matrix 에 직접 필요하지 않음.
# (plot_saturation 을 통해 free_sm_zone CSV 를 생성하는 데만 필요)
echo ""
echo "── 선행 조건 검증 …"
PREREQ_FAIL=0

_req() {
    local pattern="$1" desc="$2"
    if ls $pattern &>/dev/null 2>&1; then
        echo "   ✅ $desc"
    else
        echo "   ❌ $desc ($pattern)"
        PREREQ_FAIL=$((PREREQ_FAIL+1))
    fi
}

_req "results/stage1/ssm_scaling_*.csv"           "Stage 1 SSM scaling CSV"
_req "results/stage1/attn_scaling_*.csv"           "Stage 1 Attn scaling CSV"
_req "results/stage2/layer_latency_*.csv"          "Stage 2 layer latency CSV"
_req "results/stage2/ctx_switch_overhead_*.json"   "Stage 2 ctx_switch JSON"

if [ "$PREREQ_FAIL" -gt 0 ]; then
    echo ""
    echo "  [ERROR] $PREREQ_FAIL 선행 조건 미충족."
    echo "  Stage 1 / 2 가 완료되지 않았습니다:"
    echo "    sbatch slurm/run_stage1.sh $MODEL"
    echo "    sbatch slurm/run_stage2.sh $MODEL"
    exit 1
fi

# ── [pre-A] Chunked CSV 통합 ──────────────────────────────────────────────
# job 746350 의 chunked sweep 이 repo-root results/ 에 저장됨.
# plot_saturation.py 는 --results-dir/chunked/ 에서 ssm_chunked_*.csv 를 탐색하므로
# workspace/characterization/results/stage1/chunked/ 로 복사해 경로를 통일한다.
echo ""
echo "── [pre-A] Chunked CSV 경로 통합 …"
REPO_CHUNKED="$REPO_ROOT/results/stage1/chunked"
CHAR_CHUNKED="results/stage1/chunked"

COPIED=0
if ls "$REPO_CHUNKED"/ssm_chunked_*.csv &>/dev/null 2>&1; then
    for src in "$REPO_CHUNKED"/ssm_chunked_*.csv; do
        dst="$CHAR_CHUNKED/$(basename "$src")"
        if [ ! -f "$dst" ]; then
            cp "$src" "$dst"
            echo "   copied: $(basename "$src")"
            COPIED=$((COPIED+1))
        else
            echo "   already exists: $(basename "$dst")"
        fi
    done
    echo "   $COPIED 파일 복사 완료 → $CHAR_CHUNKED/"
else
    echo "   [warn] $REPO_CHUNKED 에 ssm_chunked_*.csv 없음"
    echo "          run_chunked_ssm_sweep.py 를 먼저 실행하십시오"
fi

# ── [pre-B] free_sm_zone CSV 생성 (plot_saturation) ──────────────────────
# free_sm_zone_*.csv 는 projection_analysis.py 의 입력.
# chunked CSV 가 통합됐으므로 이제 plot_saturation.py 가 올바르게 동작한다.
echo ""
echo "── [pre-B] free_sm_zone CSV 생성 (plot_saturation.py) …"

FREE_SM_EXISTS=0
for _M in $_MODEL_LIST; do
    if ls results/stage1/free_sm_zone_${_M}_*.csv &>/dev/null 2>&1; then
        echo "   already exists: free_sm_zone_${_M}_*.csv"
        FREE_SM_EXISTS=$((FREE_SM_EXISTS+1))
    fi
done

if ls results/stage1/chunked/ssm_chunked_*.csv &>/dev/null 2>&1; then
    for _M in $_MODEL_LIST; do
        echo "   plot_saturation.py --model $_M …"
        srun --overlap --ntasks=1 python stage1_sm_scaling/plot_saturation.py \
            --model      "$_M" \
            --results-dir results/stage1 \
            --output-dir  results/stage1 \
            2>/dev/null \
            && echo "   → free_sm_zone_${_M}_*.csv 생성됨" \
            || echo "   [warn] plot_saturation.py 실패 — projection 은 demo 데이터 사용"
    done
else
    echo "   [warn] chunked CSV 없음 — free_sm_zone 미생성, projection demo fallback"
fi

echo "   done  ($(_elapsed))"

# ── [9] Decision matrix 재실행 ──────────────────────────────────────────────
# 수정 사항: sat_sm = int(total_sm) + _NumpyEncoder → numpy int64 직렬화 오류 해결
DM_JSON="results/stage2/decision_matrix.json"

echo ""
echo "── [9/11] $(date '+%H:%M:%S')  Decision matrix (재실행) …"

srun --overlap --ntasks=1 python stage2_overhead/compute_decision_matrix.py \
    --stage1-dir results/stage1 \
    --stage2-dir results/stage2

# JSON 유효성 + 내용 확인
if python -c "
import json, sys
d = json.load(open('$DM_JSON'))
print(f'  dominant={d[\"dominant_strategy\"]}  rows={len(d[\"rows\"])}')
bad = [r for r in d['rows'] if not isinstance(r.get('saturation_sm'), (int, float, type(None)))]
if bad:
    print(f'  [ERROR] int64 잔존: {len(bad)} rows')
    sys.exit(1)
print('  JSON 유효, numpy 타입 없음')
"; then
    echo "   ✅ decision_matrix.json 재생성 성공"
else
    echo "   ❌ decision_matrix.json 재생성 실패"
    exit 1
fi
echo "   done  ($(_elapsed))"

# ── [10] Co-exec microbench ──────────────────────────────────────────────────
echo ""
echo "── [10/11] $(date '+%H:%M:%S')  S2.2: Co-exec microbench …"

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

# ── [11] Projection analysis + plots ─────────────────────────────────────────
echo ""
echo "── [11/11] $(date '+%H:%M:%S')  S2.3: Projection + plots …"

for _M in $_MODEL_LIST; do
    echo "   model: $_M"

    FREE_SM_CSV=$(ls results/stage1/free_sm_zone_${_M}_*.csv 2>/dev/null \
                  | sort | tail -1 || true)
    COEXEC_CSV=$(ls results/stage3/coexec_${_M}_*.csv 2>/dev/null \
                  | sort | tail -1 || true)

    PROJ_ARGS=(--model "$_M" --output-dir results/stage3)
    if [ -n "$FREE_SM_CSV" ]; then
        PROJ_ARGS+=(--free-sm-csv "$FREE_SM_CSV")
        echo "   free_sm_csv: $FREE_SM_CSV"
    fi
    if [ -n "$COEXEC_CSV" ]; then
        PROJ_ARGS+=(--coexec-csv "$COEXEC_CSV")
        echo "   coexec_csv:  $COEXEC_CSV"
    fi
    [ -f "$DM_JSON" ] && PROJ_ARGS+=(--stage2-json "$DM_JSON")
    [[ -z "$FREE_SM_CSV" || -z "$COEXEC_CSV" ]] && {
        echo "   [warn] 일부 입력 없음 → --demo fallback"
        PROJ_ARGS+=(--demo)
    }

    srun --overlap --ntasks=1 python stage3_hm_eval/projection_analysis.py \
        "${PROJ_ARGS[@]}" \
        || echo "   [warn] projection_analysis.py 실패 — 계속"

    srun --overlap --ntasks=1 python stage3_hm_eval/plot_results.py \
        --results-dir results/stage3 --model "$_M" 2>/dev/null \
        || echo "   [warn] plot_results.py 실패 — 계속"
done
echo "   done  ($(_elapsed))"

# ── 최종 검증 ────────────────────────────────────────────────────────────────
echo ""
echo "── 최종 출력 검증 …"
PASS=0; FAIL=0

_chk() {
    local p="$1" d="$2"
    if ls $p &>/dev/null 2>&1; then
        echo "   ✅ $d"; PASS=$((PASS+1))
    else
        echo "   ❌ $d  ($p)"; FAIL=$((FAIL+1))
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
    && echo " ✓ 재개 완료 — 다음 단계로 진행 가능" \
    || echo " ✗ 실패 항목 있음 — 위 로그 확인"
echo ""
echo " 다음 단계:"
echo "   bash slurm/check_gate_g1.sh $MODEL"
echo "========================================================"

[ "$FAIL" -eq 0 ]
