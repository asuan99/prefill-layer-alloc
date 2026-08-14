#!/bin/bash
#SBATCH -J p0-pytest
#SBATCH -p amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --comment="field=efficientai;appl=pytorch"
#SBATCH -o /scratch/%u/whlee/prefill-layer-alloc/logs/phase0_pytest_%j.log
#SBATCH -e /scratch/%u/whlee/prefill-layer-alloc/logs/phase0_pytest_%j.err

# =============================================================================
# validate_phase0_pytest.sh — Phase 0 코드 구조 검증 (GPU 불필요)
#
# 검증 항목:
#   [1] 전체 pytest (36개 기존 + 16개 신규 test_no_ncu_required)
#   [2] chunked SSM 이 primary path 임을 확인 (S0.2)
#   [3] deprecated 스크립트가 실행 중단됨을 확인 (S2.1)
#   [4] projection_analysis --demo / plot_motivation --demo figure 생성 확인
#
# GPU 없이 실행 가능 (CPU 전용 잡에서도 OK).
# A100 이 없어도 통과해야 하며, CUDA 없이도 pytest import 계층이 동작함.
#
# Usage:
#   sbatch slurm/validate_phase0_pytest.sh
#   bash   slurm/validate_phase0_pytest.sh   # 인터랙티브
# =============================================================================

set -euo pipefail

module load conda/pytorch_2.9.1_cuda13 2>/dev/null || true
module load cuda/13.0.2                2>/dev/null || true
module load gcc/15.2.0                 2>/dev/null || true

source /scratch/$USER/whlee/prefill-layer-alloc/bin/activate
cd /scratch/$USER/whlee/prefill-layer-alloc
mkdir -p logs

echo "========================================================"
echo " Phase 0 Validation — pytest (GPU-free)"
echo "   job     = ${SLURM_JOB_ID:-local}"
echo "   started = $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================"

_PASS=0; _FAIL=0

# ── [1] 전체 pytest ────────────────────────────────────────────────────────
echo ""
echo "── [1] pytest workspace/characterization/tests/ …"
if python -m pytest workspace/characterization/tests/ -v \
       --tb=short -q 2>&1; then
    echo "   PASS"
    _PASS=$((_PASS+1))
else
    echo "   FAIL — pytest 에 실패한 테스트가 있음"
    _FAIL=$((_FAIL+1))
fi

# ── [2] chunked SSM 이 primary path (_SSM_TYPES[0] == 'ssm_chunked') ──────
echo ""
echo "── [2] chunked SSM primary path 확인 (S0.2) …"
python - <<'PY'
import sys, os
sys.path.insert(0, "workspace")
sys.path.insert(0, "workspace/characterization")
# plot_saturation 에서 _SSM_TYPES 를 직접 import 해 순서 검증
import importlib.util, pathlib
spec = importlib.util.spec_from_file_location(
    "plot_saturation",
    "workspace/characterization/stage1_sm_scaling/plot_saturation.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
types = getattr(mod, "_SSM_TYPES", None)
assert types is not None, "_SSM_TYPES 없음 — plot_saturation.py 확인"
assert types[0] == "ssm_chunked", (
    f"_SSM_TYPES[0]={types[0]!r}  expected 'ssm_chunked'\n"
    "S0.2 policy: chunked 가 primary 여야 함"
)
print(f"   OK: _SSM_TYPES = {types}")
PY
if [ $? -eq 0 ]; then
    echo "   PASS"
    _PASS=$((_PASS+1))
else
    echo "   FAIL"
    _FAIL=$((_FAIL+1))
fi

# ── [3] deprecated run_concurrent_eval 실행 중단 확인 (S2.1) ──────────────
echo ""
echo "── [3] deprecated run_concurrent_eval 실행 중단 확인 …"
python workspace/characterization/stage3_hm_eval/run_concurrent_eval.py 2>&1 \
    | grep -q "DEPRECATED" && EXIT_BLOCKED=true || EXIT_BLOCKED=false

# exit code 1 이어야 함 (sys.exit(1))
set +e
python workspace/characterization/stage3_hm_eval/run_concurrent_eval.py 2>/dev/null
EXITCODE=$?
set -e

if [ "$EXIT_BLOCKED" = "true" ] && [ "$EXITCODE" -ne 0 ]; then
    echo "   PASS (exit=$EXITCODE, DEPRECATED 메시지 확인)"
    _PASS=$((_PASS+1))
else
    echo "   FAIL (exit=$EXITCODE, DEPRECATED 메시지 미출력)"
    _FAIL=$((_FAIL+1))
fi

# ── [4] --demo figure 생성 확인 ────────────────────────────────────────────
echo ""
echo "── [4a] projection_analysis --demo figure 생성 …"
python workspace/characterization/stage3_hm_eval/projection_analysis.py \
    --demo --output-dir /tmp/validate_phase0/proj 2>&1
PROJ_FILES=$(ls /tmp/validate_phase0/proj/projection_*.png 2>/dev/null | wc -l)
if [ "$PROJ_FILES" -ge 2 ]; then
    echo "   PASS ($PROJ_FILES projection_*.png 생성됨)"
    _PASS=$((_PASS+1))
else
    echo "   FAIL (projection_*.png 파일 없음)"
    _FAIL=$((_FAIL+1))
fi

echo ""
echo "── [4b] plot_motivation --demo figure 생성 …"
python workspace/serving-eval/plot_motivation.py \
    --demo --output-dir /tmp/validate_phase0/motiv 2>&1
MOTIV_FILES=$(ls /tmp/validate_phase0/motiv/fig_*.png 2>/dev/null | wc -l)
if [ "$MOTIV_FILES" -ge 2 ]; then
    echo "   PASS ($MOTIV_FILES fig_*.png 생성됨)"
    _PASS=$((_PASS+1))
else
    echo "   FAIL (fig_*.png < 2)"
    _FAIL=$((_FAIL+1))
fi

# ── 최종 요약 ──────────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo " Phase 0 결과: PASS=$_PASS  FAIL=$_FAIL"
[ "$_FAIL" -eq 0 ] && echo " ✓ All checks passed — Phase 1 진행 가능" \
                    || echo " ✗ 실패 항목 있음 — 위 로그 확인"
echo "========================================================"

[ "$_FAIL" -eq 0 ]
