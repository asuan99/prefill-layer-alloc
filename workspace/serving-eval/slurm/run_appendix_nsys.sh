#!/bin/bash
# =============================================================================
# run_appendix_nsys.sh — OPTIONAL APPENDIX: nsys + ncu layer heterogeneity
#
# ※ 헤드라인 figure (A/B/C) 에 불필요. Fig D (appendix) 전용.
#   run_s1_2_motivation.sh 의 헤드라인 결론은 이 스크립트 없이 완결된다.
#
# 실행 조건 (모두 충족해야 함):
#   1. nsys (Nsight Systems) PATH 에 있어야 함
#   2. ncu (Nsight Compute) PATH 에 있어야 함 (--with-ncu 시)
#   3. 하드웨어 counter 권한:
#        --exclusive + NVreg_RestrictProfilingToAdminUsers=0
#        또는 amd_a100nv_8 파티션 권한 있는 SLURM 잡
#   4. characterization venv 활성화 필요
#      (_forward_pass.py 가 mamba-ssm / LayerRunner import)
#
# 출력:
#   results/profiling/kernel_timeline_{model}_{sl}_{bs}.csv  (nsys)
#   results/profiling/ncu_{model}_{sl}_{bs}.csv              (ncu, --with-ncu 시)
#   → plot_layer_heterogeneity.py 의 입력 (Fig D appendix)
#
# Usage:
#   sbatch serving-eval/slurm/run_appendix_nsys.sh                 # zamba2, no ncu
#   sbatch serving-eval/slurm/run_appendix_nsys.sh zamba2 --with-ncu
#   sbatch serving-eval/slurm/run_appendix_nsys.sh zamba2 --no-nsys --with-ncu
# =============================================================================
#SBATCH -J se-appendix-nsys
#SBATCH --exclusive
#SBATCH --constrain=hwperf
#SBATCH -p amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --comment=pytorch
#SBATCH -o /scratch/%u/whlee/prefill-layer-alloc/logs/se_nsys_%j.log
#SBATCH -e /scratch/%u/whlee/prefill-layer-alloc/logs/se_nsys_%j.err

set -euo pipefail

module load conda/pytorch_2.9.1_cuda13
module load cuda/13.0.2
module load gcc/15.2.0

REPO_ROOT="/scratch/$USER/whlee/prefill-layer-alloc"

# appendix profiling 은 characterization venv 사용
# (_forward_pass.py 가 characterization/src 를 import 하므로)
source "$REPO_ROOT/bin/activate"
cd "$REPO_ROOT"
mkdir -p logs results/profiling

MODEL=${1:-zamba2}
shift 2>/dev/null || true

# 나머지 인자를 profile_layer_heterogeneity.py 에 그대로 전달
# (예: --no-ncu, --no-nsys, --seq-len 4096, --batch-size 2)
EXTRA_ARGS=("$@")

# --with-ncu 는 이 스크립트 전용 플래그: --no-ncu 를 제거
NO_NCU_FLAG="--no-ncu"
for arg in "${EXTRA_ARGS[@]:-}"; do
    [[ "$arg" == "--with-ncu" ]] && NO_NCU_FLAG="" && break
done
EXTRA_ARGS=("${EXTRA_ARGS[@]/--with-ncu/}")

echo "=================================================================="
echo " OPTIONAL APPENDIX: nsys + ncu Layer Heterogeneity"
echo "   model     = $MODEL"
echo "   extra     = ${EXTRA_ARGS[*]:-none}"
echo "   ncu       = ${NO_NCU_FLAG:-enabled (--with-ncu)}"
echo "   job       = ${SLURM_JOB_ID:-local}"
echo "   GPU       = $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "   nsys      = $(nsys --version 2>&1 | head -1 || echo 'NOT FOUND')"
echo "   ncu       = $(ncu --version 2>&1 | head -1 || echo 'NOT FOUND')"
echo "=================================================================="
echo "  ※ 이 스크립트는 OPTIONAL APPENDIX 입니다."
echo "    헤드라인 Fig A/B/C 는 run_s1_2_motivation.sh 로 생성됩니다."
echo "=================================================================="

# nsys 가용 여부 확인
if ! command -v nsys &>/dev/null && [ -z "$NO_NCU_FLAG" ]; then
    echo ""
    echo "  [ERROR] nsys 없음. 모듈 로드 또는 PATH 확인."
    echo "  module load cuda/13.0.2 후 nsys 경로 확인:"
    echo "    which nsys  /  ls /usr/local/cuda/bin/nsys"
    exit 1
fi

echo ""
echo "── [1/2] profile_layer_heterogeneity.py …"
python workspace/serving-eval/profile_layer_heterogeneity.py \
    --model      "$MODEL" \
    --output-dir results/profiling \
    ${NO_NCU_FLAG:+$NO_NCU_FLAG} \
    "${EXTRA_ARGS[@]:-}"

echo ""
echo "── [2/2] plot_layer_heterogeneity.py (Fig D) …"
python workspace/serving-eval/plot_layer_heterogeneity.py \
    --results-dir results/profiling \
    --model       "$MODEL" \
    --output-dir  results/profiling 2>/dev/null \
    || echo "  [warn] plot_layer_heterogeneity.py 실패 — 데이터 확인"

echo ""
echo "=================================================================="
echo " Appendix done → results/profiling/"
ls results/profiling/*.{csv,png} 2>/dev/null | sed 's/^/   /' || true
echo "=================================================================="
