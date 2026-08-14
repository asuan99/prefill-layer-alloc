#!/bin/bash
# =============================================================================
# run_stage1_decode_sweep.sh — Phase 1 S1.3: Decode SM sweep + G1-gate table
#
# run_stage1.sh (prefill sweep) 이 완료된 후 실행.
# decode kernel 의 SM-sensitivity 를 측정하고 free-SM zone 과 결합해
# G1-gate 입력 CSV 를 생성한다.
#
# 출력:
#   results/stage1/decode_sm_{model}_{device}.csv     — decode SM sweep
#   results/stage1/g1_gate_{model}_{device}.csv       — free_sm × decode_sensitivity
#   results/stage1/free_sm_zone_{model}_{device}.csv  — (plot_saturation 로 생성)
#   results/stage1/fig2_free_sm_zone_{model}.png      — updated free-SM figure
#
# 검증 기준:
#   - g1_gate CSV 에 g1_potential > 0.05 인 행이 존재하면 G1 게이트 통과 후보
#   - SSM decode sensitivity ≈ 0 이면 SM 재할당 필요성 낮음 (expected on A100)
#
# Usage:
#   sbatch slurm/run_stage1_decode_sweep.sh
#   sbatch slurm/run_stage1_decode_sweep.sh falcon_h1
#   sbatch slurm/run_stage1_decode_sweep.sh zamba2 a100_80gb
# =============================================================================
#SBATCH -J s1-decode-sweep
#SBATCH -p amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --comment="field=efficientai;appl=pytorch"
#SBATCH -o /scratch/%u/whlee/prefill-layer-alloc/logs/s1_decode_%j.log
#SBATCH -e /scratch/%u/whlee/prefill-layer-alloc/logs/s1_decode_%j.err

set -euo pipefail

module load conda/pytorch_2.9.1_cuda13
module load cuda/13.0.2
module load gcc/15.2.0

source /scratch/$USER/whlee/prefill-layer-alloc/bin/activate
cd /scratch/$USER/whlee/prefill-layer-alloc
mkdir -p logs results/stage1

MODEL=${1:-zamba2}
DEVICE=${2:-auto}

echo "========================================================"
echo " Phase 1 S1.3: Decode SM Sweep + G1-Gate Table"
echo "   model   = $MODEL"
echo "   device  = $DEVICE"
echo "   job     = ${SLURM_JOB_ID:-local}"
echo "   GPU     = $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "   started = $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================"

_T0=$(date +%s)
_elapsed() { local t=$(( $(date +%s) - _T0 )); printf '%dm%02ds' $(( t/60 )) $(( t%60 )); }

# ── [1/4] Decode SM sweep ─────────────────────────────────────────────────
# SSM decode (seq_len=1) + Attn decode (KV read) 각각 SM 수를 변화시켜
# latency 를 측정. decode_sm_sensitivity = (lat_min - lat_full) / lat_full.
echo ""
echo "── [1/4] $(date '+%H:%M:%S')  Decode SM sensitivity sweep …"
python workspace/characterization/stage1_sm_scaling/run_decode_sm_sweep.py \
    --model  "$MODEL" \
    --device "$DEVICE" \
    --context-lens 1024 4096 \
    --batch-sizes  1 4 16 \
    --n-warmup  5 \
    --n-measure 20
echo "   done  ($(_elapsed))"

# ── [2/4] plot_saturation —— free-SM zone + G1-gate CSV 생성 ─────────────
# --decode-dir 플래그로 decode sensitivity 를 merged CSV 에 포함시킴.
# 출력: free_sm_zone_{model}_{tag}.csv (saturation_sm, free_sm_fraction, g1_potential)
echo ""
echo "── [2/4] $(date '+%H:%M:%S')  Free-SM zone plot + G1-gate CSV …"
DECODE_DIR="results/stage1"
python workspace/characterization/stage1_sm_scaling/plot_saturation.py \
    --model      "$MODEL" \
    --decode-dir "$DECODE_DIR" \
    --output-dir results/stage1
echo "   done  ($(_elapsed))"

# ── [3/4] G1-gate table 출력 ─────────────────────────────────────────────
# free_sm_fraction × decode_sm_sensitivity = g1_potential 을 콘솔에 표시.
echo ""
echo "── [3/4] $(date '+%H:%M:%S')  G1-gate table …"
python - <<PYEOF
import sys, glob, os
sys.path.insert(0, "workspace")
import pandas as pd

# g1_gate CSV 탐색
files = sorted(glob.glob("results/stage1/g1_gate_${MODEL}_*.csv"))
if not files:
    # free_sm_zone CSV 에서 직접 계산
    files = sorted(glob.glob("results/stage1/free_sm_zone_${MODEL}_*.csv"))
    if not files:
        print("  [WARN] g1_gate / free_sm_zone CSV 없음 — plot_saturation 결과 확인")
        sys.exit(0)

df = pd.read_csv(files[-1])
print(f"\n  G1-gate table  ({files[-1]})\n")
cols = [c for c in ["model_name","ssm_type","seq_len","batch_size",
                     "free_ratio","decode_sm_sensitivity","g1_potential"]
        if c in df.columns]
if not cols:
    print(df.head(10).to_string())
else:
    print(df[cols].to_string(index=False))

# 게이트 판정
if "g1_potential" in df.columns:
    max_g1 = df["g1_potential"].max()
    print(f"\n  max g1_potential = {max_g1:.4f}")
    if max_g1 > 0.05:
        print("  → G1 게이트 통과 후보 (g1_potential > 0.05)")
    else:
        print("  → G1 게이트 미달 (SM 재할당 이득 낮음)")
PYEOF
echo "   done  ($(_elapsed))"

# ── [4/4] 파일 존재 검증 ─────────────────────────────────────────────────
echo ""
echo "── [4/4] $(date '+%H:%M:%S')  Output validation …"
PASS=0; FAIL=0

check_file() {
    local pattern="$1"; local desc="$2"
    local found
    found=$(ls $pattern 2>/dev/null | head -1 || true)
    if [ -n "$found" ]; then
        echo "   OK  $desc → $found"
        PASS=$((PASS+1))
    else
        echo "   NG  $desc → $pattern 없음"
        FAIL=$((FAIL+1))
    fi
}

check_file "results/stage1/decode_sm_${MODEL}_*.csv"    "decode SM sweep CSV"
check_file "results/stage1/free_sm_zone_${MODEL}_*.csv" "free-SM zone CSV"
check_file "results/stage1/fig2_*.png"                  "free-SM zone figure"

echo ""
echo "========================================================"
echo " S1.3 결과: PASS=$PASS  FAIL=$FAIL  elapsed=$(_elapsed)"
[ "$FAIL" -eq 0 ] \
    && echo " ✓ Decode sweep 완료 — check_gate_g1.sh 로 G1 판정 진행" \
    || echo " ✗ 실패 항목 있음 — 위 로그 확인"
echo "========================================================"
[ "$FAIL" -eq 0 ]
