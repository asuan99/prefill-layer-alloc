#!/bin/bash
# =============================================================================
# run_s1_2_motivation.sh — Phase 1 S1.2 (revised): counter-free motivation figures
#
# ncu/nsys/CUPTI 없이 3개 헤드라인 figure 를 생성한다.
#
#   Fig A — NVML device-level util 시계열 (S1.1 serving NVML CSV 필요)
#             "device-level aggregate (NVML ~10Hz), NOT per-layer"
#   Fig B — free-SM zone 비교 (Stage 1 sweep CSV 필요)
#             "Stage 1 isolated-kernel sweep (CUDA events, no hardware counters)"
#   Fig C — throughput vs analytical roofline (Stage 1 sweep CSV 필요)
#             "Roofline = analytical peak spec; points = CUDA-event latency"
#   Fig D — nsys appendix (OPTIONAL, 없으면 건너뜀)
#
# GPU 불필요 — matplotlib/pandas 만 사용.
# 실측 데이터 없이 --demo 로 렌더링만 검증 가능.
#
# Usage:
#   # 실측 데이터 기반
#   sbatch serving-eval/slurm/run_s1_2_motivation.sh \
#       --hybrid-model zamba2 \
#       --nvml-dir     results/serving-eval \
#       --stage1-dir   results/stage1
#
#   # demo (데이터 없이 렌더링 검증)
#   sbatch serving-eval/slurm/run_s1_2_motivation.sh --demo
#
#   # 특정 모델 + no-Fig-D
#   sbatch serving-eval/slurm/run_s1_2_motivation.sh \
#       --hybrid-model zamba2 --nvml-dir results/serving-eval \
#       --stage1-dir results/stage1 --no-fig-d
# =============================================================================
#SBATCH -J se-s1-motivation
#SBATCH -p amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:0
#SBATCH --time=00:15:00
#SBATCH --comment="field=efficientai;appl=pytorch"
#SBATCH -o /scratch/%u/whlee/prefill-layer-alloc/logs/se_motivation_%j.log
#SBATCH -e /scratch/%u/whlee/prefill-layer-alloc/logs/se_motivation_%j.err

set -euo pipefail

module load conda/pytorch_2.9.1_cuda13 2>/dev/null || true
module load cuda/13.0.1                 2>/dev/null || true
module load gcc/15.2.0                 2>/dev/null || true

REPO_ROOT="/scratch/$USER/whlee/prefill-layer-alloc"
VENV_DIR="$REPO_ROOT/workspace/serving-eval/.venv"

# characterization venv 도 가능하지만, serving-eval venv 우선
if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
else
    source "$REPO_ROOT/bin/activate"
    echo "  [info] serving-eval venv 없음 → characterization venv 사용"
fi

cd "$REPO_ROOT"
mkdir -p logs results/motivation

echo "=================================================================="
echo " Phase 1 S1.2: Counter-Free Motivation Figures"
echo "   job     = ${SLURM_JOB_ID:-local}"
echo "   started = $(date '+%Y-%m-%d %H:%M:%S')"
echo "   args    = $*"
echo "=================================================================="

# 인자 없으면 demo 모드
EXTRA_ARGS=("$@")
if [ ${#EXTRA_ARGS[@]} -eq 0 ]; then
    echo ""
    echo "  [info] 인자 없음 → --demo 모드 (렌더링 검증용)"
    EXTRA_ARGS=(--demo)
fi

echo ""
echo "── plot_motivation.py ${EXTRA_ARGS[*]} …"
python workspace/serving-eval/plot_motivation.py \
    --output-dir results/motivation \
    "${EXTRA_ARGS[@]}"

# ── 생성된 figure 확인 ─────────────────────────────────────────────────────
echo ""
echo "── 생성된 figure 확인 …"
FIGS=$(ls results/motivation/fig_*.png 2>/dev/null | wc -l)
echo "   fig_*.png = $FIGS 개"

if [ "$FIGS" -ge 2 ]; then
    echo "   PASS (Fig B + Fig C 이상 생성됨)"
else
    echo "   FAIL (fig_*.png < 2)"
    exit 1
fi

ls -lh results/motivation/fig_*.png 2>/dev/null | sed 's/^/   /'

# ── caption 정책 검증 (S0.4) ───────────────────────────────────────────────
echo ""
echo "── S0.4 정책 검증 (ncu import 없음) …"
python - <<'PYEOF'
import pathlib, sys

script = pathlib.Path("workspace/serving-eval/plot_motivation.py").read_text()
violations = []
for kw in ("ncu_runner", "NCURunner", "cupti_monitor"):
    if kw in script:
        violations.append(kw)

if violations:
    print(f"  FAIL: plot_motivation.py 에 ncu 관련 import: {violations}")
    sys.exit(1)
else:
    print("  OK: ncu/nsys/CUPTI import 없음 (S0.4 정책 준수)")
PYEOF

echo ""
echo "=================================================================="
echo " 출력: results/motivation/"
echo "   Fig A: NVML util 시계열 (serving NVML CSV 필요)"
echo "   Fig B: free-SM zone     (Stage 1 CSV 필요)"
echo "   Fig C: roofline         (Stage 1 CSV + peak spec)"
echo "   Fig D: nsys appendix    (optional, --no-fig-d 로 비활성화)"
echo "=================================================================="
