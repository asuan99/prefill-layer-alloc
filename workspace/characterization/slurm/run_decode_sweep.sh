#!/bin/bash
# =============================================================================
# run_decode_sweep.sh — Decode SM-saturation sweep + KV/weight crossover
#
# 실행 순서:
#   Task 0 (CPU)  — decode_crossover/compute_crossover.py     (L*, R(L,bs) derived)
#   Task 1 (GPU)  — decode_sm_scaling/run_decode_sweep.py     (zamba2 → falcon_h1)
#   Task 2 (CPU)  — decode_sm_scaling/analyze_decode_sat.py   (판정 1·2 + 보고서)
#
# Task 0 의 layer crossover L* 이 sweep context 격자 최대값(32768)보다 크면 경고를
# 출력하되 sweep 은 그대로 진행한다 (격자 확장 여부는 사람이 판단).
#
# Usage:
#   sbatch slurm/run_decode_sweep.sh
#   sbatch slurm/run_decode_sweep.sh --models zamba2          # single model
# =============================================================================
#SBATCH -J decode-sweep
#SBATCH -p amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --comment=pytorch
#SBATCH -o /scratch/%u/whlee/prefill-layer-alloc/logs/decode_sweep_%j.log
#SBATCH -e /scratch/%u/whlee/prefill-layer-alloc/logs/decode_sweep_%j.err

set -euo pipefail

module load conda/pytorch_2.9.1_cuda13 2>/dev/null || true
module load gcc/15.2.0                 2>/dev/null || true

source /scratch/$USER/whlee/prefill-layer-alloc/bin/activate
cd /scratch/$USER/whlee/prefill-layer-alloc
mkdir -p logs reports

ROOT=/scratch/$USER/whlee/prefill-layer-alloc
CHAR=$ROOT/workspace/characterization
PY=python

# Optional pass-through (e.g. --models zamba2). Default: both models.
EXTRA_ARGS=("$@")
MODELS=(zamba2 falcon_h1)
# crude parse: if caller passed "--models X [Y...]", honor it for Task 1/2 loop
if [ "${#EXTRA_ARGS[@]}" -ge 2 ] && [ "${EXTRA_ARGS[0]}" = "--models" ]; then
    MODELS=("${EXTRA_ARGS[@]:1}")
fi

echo "========================================================"
echo " Decode SM-Saturation Sweep + Crossover"
echo "   job     = ${SLURM_JOB_ID:-local}"
echo "   started = $(date '+%Y-%m-%d %H:%M:%S')"
echo "   models  = ${MODELS[*]}"
echo "========================================================"

# ── Task 0: crossover (CPU, GPU 불필요) ──────────────────────────────────────
echo ""
echo "── Task 0: compute_crossover.py (derived; CPU) …"
$PY "$CHAR/decode_crossover/compute_crossover.py" --models "${MODELS[@]}"

# ── Task 0 L* 경고: layer crossover L* > 32768 이면 경고 (sweep 은 진행) ──────
echo ""
echo "── Task 0 L* vs sweep context 격자 최대값(32768) 점검 …"
$PY - "${MODELS[@]}" <<'PYEOF'
import sys, csv
from pathlib import Path
ROOT = Path("workspace/characterization/decode_crossover/results")
L_MAX = 32768
warned = False
for model in sys.argv[1:]:
    p = ROOT / f"crossover_{model}.csv"
    if not p.exists():
        print(f"  [warn] {p} 없음 — L* 점검 생략"); continue
    with open(p) as f:
        for row in csv.DictReader(f):
            if row["metric"] == "judgment_Lstar_layer":
                try:
                    Lstar = float(row["value"])
                except ValueError:
                    continue
                if Lstar > L_MAX:
                    warned = True
                    print(f"  [WARN] {model} bs={row['batch']}: layer crossover "
                          f"L*={Lstar:.0f} > sweep 격자 최대값 {L_MAX} — "
                          f"context 격자 확장 검토 필요 (sweep 은 그대로 진행)")
if not warned:
    print(f"  OK: 모든 모델·bs 의 layer crossover L* ≤ {L_MAX} (격자 충분)")
PYEOF

# ── GPU 가용성 확인 ──────────────────────────────────────────────────────────
echo ""
echo "── GPU 확인 …"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || {
    echo "  [error] GPU 미감지 — Task 1 중단"; exit 1; }

# ── Task 1: decode sweep (GPU; zamba2 → falcon_h1) ──────────────────────────
for model in "${MODELS[@]}"; do
    echo ""
    echo "── Task 1: run_decode_sweep.py --model $model …"
    $PY "$CHAR/decode_sm_scaling/run_decode_sweep.py" --model "$model"
done

# ── Task 2: 분석 + 판정 + 보고서 (CPU) ───────────────────────────────────────
echo ""
echo "── Task 2: analyze_decode_sat.py …"
$PY "$CHAR/decode_sm_scaling/analyze_decode_sat.py" --models "${MODELS[@]}"

echo ""
echo "========================================================"
echo " 완료: $(date '+%Y-%m-%d %H:%M:%S')"
echo "   Task 0 CSV : $CHAR/decode_crossover/results/crossover_*.csv"
echo "   Task 1 CSV : $CHAR/results/decode_scaling/decode_*.csv"
echo "   Task 2     : $CHAR/results/decode_scaling/sm_sat_decode_*.csv"
echo "   보고서      : $ROOT/reports/decode_saturation_and_crossover.md"
echo "========================================================"
