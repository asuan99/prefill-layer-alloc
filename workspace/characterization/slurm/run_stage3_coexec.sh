#!/bin/bash
# =============================================================================
# run_stage3_coexec.sh — Phase 2 S2.2: 실측 concurrent execution overlap
#
# 이전의 run_concurrent_eval.py (순차 시뮬레이션, DEPRECATED) 를 대체한다.
# 두 Green Context partition 에서 prefill + decode kernel 을 실제로 동시에
# enqueue 하여 overlap_ratio 를 직접 측정한다.
#
# overlap_ratio = concurrent_time / (solo_prefill + solo_decode)
#   → 0.5 에 가까울수록 완전 overlap (이상적)
#   → 1.0 에 가까울수록 overlap 없음 (순차 실행과 동일)
#
# 백엔드 비교:
#   green_ctx  — SM 공간 분할 (cuGreenCtxCreate, 진짜 동시 실행)
#   two_stream — 파티션 없는 CUDA stream (하드웨어 스케줄러 기본 거동 baseline)
#
# 검증 기준:
#   green_ctx 의 overlap_ratio 가 two_stream 보다 낮으면
#   → Green Context 의 spatial sharing 이 실제로 이득을 줌 (S2.2 핵심 주장)
#
# 출력:
#   results/stage3/coexec_{model}_{device}.csv
#   results/stage3/coexec_{model}_{device}.png
#
# Usage:
#   sbatch slurm/run_stage3_coexec.sh
#   sbatch slurm/run_stage3_coexec.sh falcon_h1
#   sbatch slurm/run_stage3_coexec.sh zamba2 a100_80gb
#
# 선행 조건: run_stage1.sh 완료 (모델 config 참조)
# =============================================================================
#SBATCH -J s3-coexec
#SBATCH -p amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --comment="field=efficientai;appl=pytorch"
#SBATCH -o /scratch/%u/whlee/prefill-layer-alloc/logs/s3_coexec_%j.log
#SBATCH -e /scratch/%u/whlee/prefill-layer-alloc/logs/s3_coexec_%j.err

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
echo " Phase 2 S2.2: Real Concurrent Execution Overlap"
echo "   model   = $MODEL"
echo "   device  = $DEVICE"
echo "   job     = ${SLURM_JOB_ID:-local}"
echo "   GPU     = $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "   started = $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================"

# Green Context 지원 여부 사전 확인
echo ""
echo "── Green Context 지원 여부 확인 …"
python - <<'PYEOF'
import sys
sys.path.insert(0, "workspace")
sys.path.insert(0, "workspace/characterization")
import torch
from src.smctrl.green_ctx_controller import SMController
n = torch.cuda.get_device_properties(0).multi_processor_count
ctrl = SMController(preset_sm_counts=[n//2, n])
if ctrl.is_available():
    print(f"  OK: Green Context 사용 가능 (총 SM={n})")
else:
    print("  WARN: Green Context 사용 불가 — two_stream 만 측정됨")
    print("        (CUDA driver 550+ 및 데이터센터 GPU 필요)")
PYEOF

_T0=$(date +%s)
_elapsed() { local t=$(( $(date +%s) - _T0 )); printf '%dm%02ds' $(( t/60 )) $(( t%60 )); }

# ── [1/3] co-exec microbench sweep ──────────────────────────────────────────
echo ""
echo "── [1/3] $(date '+%H:%M:%S')  Co-execution sweep …"
echo "   SM splits   : 0.3 0.4 0.5 0.6 0.7"
echo "   seq_lens    : 1024 2048 4096"
echo "   batch_sizes : 1 4"
echo "   context_lens: 1024 4096"

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

# ── [2/3] overlap ratio 결과 출력 ────────────────────────────────────────
echo ""
echo "── [2/3] $(date '+%H:%M:%S')  Overlap ratio 요약 …"
python - "$MODEL" <<'PYEOF'
import sys, glob
import pandas as pd

MODEL = sys.argv[1]
files = sorted(glob.glob(f"results/stage3/coexec_{MODEL}_*.csv"))
if not files:
    print("  [ERROR] coexec CSV 없음")
    sys.exit(1)

df = pd.read_csv(files[-1])
print(f"\n  {files[-1]}\n")

summary = df.groupby(["backend", "sm_split"])["overlap_ratio"].median().unstack("sm_split")
print("  overlap_ratio (median) per backend × sm_split:")
print(summary.round(3).to_string())

# 핵심 질문: green_ctx < two_stream ?
if "green_ctx" in df["backend"].values and "two_stream" in df["backend"].values:
    gc_med = df[df["backend"]=="green_ctx"]["overlap_ratio"].median()
    ts_med = df[df["backend"]=="two_stream"]["overlap_ratio"].median()
    print(f"\n  green_ctx  median overlap_ratio = {gc_med:.4f}")
    print(f"  two_stream median overlap_ratio = {ts_med:.4f}")
    if gc_med < ts_med:
        print(f"  → OK: Green Context 가 spatial sharing 이득 제공")
        print(f"         (gc={gc_med:.3f} < ts={ts_med:.3f})")
    else:
        print(f"  → NOTE: 이 구성에서 Green Context 이득 없음")
        print(f"          (gc={gc_med:.3f} ≥ ts={ts_med:.3f})")
        print(f"          SM split / seq_len 범위 재검토 필요")
PYEOF

# ── [3/3] 출력 파일 검증 ─────────────────────────────────────────────────
echo ""
echo "── [3/3] $(date '+%H:%M:%S')  Output validation …"
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

check_file "results/stage3/coexec_${MODEL}_*.csv" "co-exec sweep CSV"
check_file "results/stage3/coexec_${MODEL}_*.png" "overlap_ratio figure"

echo ""
echo "========================================================"
echo " S2.2 결과: PASS=$PASS  FAIL=$FAIL  elapsed=$(_elapsed)"
echo " CSV → run_stage3_projection.sh --coexec-csv 입력으로 사용"
[ "$FAIL" -eq 0 ] \
    && echo " ✓ Co-exec 측정 완료 — Projection 분석으로 진행" \
    || echo " ✗ 실패 항목 있음 — 위 로그 확인"
echo "========================================================"
[ "$FAIL" -eq 0 ]
