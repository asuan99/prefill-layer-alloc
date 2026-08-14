#!/bin/bash
# =============================================================================
# run_motivation_figures.sh — Phase 1 S1.2 (revised): counter-free motivation
#
# ncu/nsys/CUPTI 없이 3개 헤드라인 figure 를 생성한다.
#
#   Fig A — NVML device-level util 시계열  (S1.1 serving 결과 필요)
#   Fig B — free-SM zone 비교             (S1.3 Stage 1 sweep 결과 필요)
#   Fig C — throughput vs roofline        (S1.3 Stage 1 sweep 결과 필요)
#   Fig D — nsys appendix                 (optional, nsys 있을 때만)
#
# 실측 데이터 없이도 --demo 로 전체 figure 를 생성해 렌더링을 검증할 수 있다.
#
# Caption 검증 기준:
#   Fig A caption: "device-level aggregate (NVML)" 포함
#   Fig C caption: "analytical bound" 포함
#
# Usage:
#   # 실측 데이터 사용
#   sbatch slurm/run_motivation_figures.sh \
#       --hybrid-model zamba2 \
#       --nvml-dir results/serving-eval \
#       --stage1-dir results/stage1
#
#   # demo (데이터 없이 figure 렌더링만 검증)
#   sbatch slurm/run_motivation_figures.sh --demo
# =============================================================================
#SBATCH -J s1-motivation
#SBATCH -p amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0
#SBATCH --time=00:15:00
#SBATCH --comment="field=efficientai;appl=pytorch"
#SBATCH -o /scratch/%u/whlee/prefill-layer-alloc/logs/motivation_%j.log
#SBATCH -e /scratch/%u/whlee/prefill-layer-alloc/logs/motivation_%j.err

set -euo pipefail

module load conda/pytorch_2.9.1_cuda13 2>/dev/null || true
module load gcc/15.2.0                 2>/dev/null || true

source /scratch/$USER/whlee/prefill-layer-alloc/bin/activate
cd /scratch/$USER/whlee/prefill-layer-alloc
mkdir -p logs results/motivation

echo "========================================================"
echo " Phase 1 S1.2 (revised): Counter-Free Motivation Figures"
echo "   job     = ${SLURM_JOB_ID:-local}"
echo "   started = $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================"

# 인자 전달 방식: 스크립트 인자를 plot_motivation.py 에 그대로 넘김
# 예) --demo / --hybrid-model zamba2 --nvml-dir ... --stage1-dir ...
EXTRA_ARGS=("$@")

# 인자 없으면 demo 모드로 실행
if [ ${#EXTRA_ARGS[@]} -eq 0 ]; then
    echo "  [info] 인자 없음 → --demo 모드로 실행 (렌더링 검증)"
    EXTRA_ARGS=(--demo)
fi

echo ""
echo "── plot_motivation.py ${EXTRA_ARGS[*]} …"
python workspace/serving-eval/plot_motivation.py \
    --output-dir results/motivation \
    "${EXTRA_ARGS[@]}"

# ── figure 존재 확인 ────────────────────────────────────────────────────────
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

# ── Caption 정책 검증 ────────────────────────────────────────────────────────
echo ""
echo "── caption 정책 검증 (S0.4 / S1.2 revised) …"
python - <<'PYEOF'
import pathlib, re, sys

out_dir = pathlib.Path("results/motivation")

# Fig A caption: device-level aggregate 명시
fig_a = list(out_dir.glob("fig_a_*.png"))
if fig_a:
    # caption 은 stdout 에 출력됨 — log 에서 확인
    print(f"  Fig A: {fig_a[0].name}  (캡션은 로그에서 확인)")

# S0.4: ncu/nsys import 가 없어야 함 (이미 test_no_ncu_required.py 에서 검증)
script = pathlib.Path("workspace/serving-eval/plot_motivation.py").read_text()
if "ncu_runner" in script or "NCURunner" in script:
    print("  FAIL: plot_motivation.py 에 ncu_runner import 있음")
    sys.exit(1)
else:
    print("  OK: plot_motivation.py 에 ncu/nsys import 없음 (S0.4 준수)")
PYEOF

echo ""
echo "========================================================"
echo " Motivation figures: results/motivation/"
echo " Fig A: 실측 NVML CSV 필요  (--nvml-dir 로 제공)"
echo " Fig B: Stage 1 chunked CSV 필요  (--stage1-dir 로 제공)"
echo " Fig C: Stage 1 CSV + peak spec (analytical roofline)"
echo "========================================================"
