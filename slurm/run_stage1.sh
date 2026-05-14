#!/bin/bash
#SBATCH -J 1-prefill-sm-scaling
#SBATCH -p amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --comment=pytorch
#SBATCH -o /scratch/%u/whlee/prefill-layer-alloc/logs/stage1_%j.log
#SBATCH -e /scratch/%u/whlee/prefill-layer-alloc/logs/stage1_%j.err

# =============================================================================
# Stage 1: SM Scaling Sweep
#
# Step 구성 (6단계):
#   [1/6] SSM wave-model sweep        — 전체 SM 직접 측정 → wave model 합성 (빠름)
#   [2/6] SSM chunked prefill sweep   — kernel 호출 분할로 cooperative barrier 우회,
#                                       Green Context 하에서 직접 측정 (핵심 신규)
#   [3/6] SSM torch scan sweep        — wave model 정확도 검증용 (PyTorch scan)
#   [4/6] Attention sweep             — Green Context 직접 측정
#   [5/6] MLP sweep                   — Green Context 직접 측정
#   [6/6] Analysis + Plots            — chunked 결과 분석 + 전체 비교 시각화
#
# Usage:
#   sbatch slurm/run_stage1.sh                    # zamba2, auto-detect GPU
#   sbatch slurm/run_stage1.sh falcon_h1          # falcon_h1
#   sbatch slurm/run_stage1.sh zamba2 a100_80gb   # explicit hardware key
#
# Env vars (optional overrides):
#   CHUNKED_PCT   space-sep prefill_chunk_tokens  (default: "256 512 1024 2048 4096")
#   CHUNKED_SM    space-sep sm_counts             (default: hardware.yaml sweep steps)
#   CHUNKED_SEQ   space-sep seq_lens              (default: "512 1024 2048 4096 8192")
#   CHUNKED_BS    space-sep batch_sizes           (default: "1 4 16 32")
# =============================================================================

set -euo pipefail

module load conda/pytorch_2.9.1_cuda13
module load cuda/13.0.2
module load gcc/15.2.0

source /scratch/$USER/whlee/prefill-layer-alloc/bin/activate

cd /scratch/$USER/whlee/prefill-layer-alloc
mkdir -p logs results/stage1 results/stage1/chunked

MODEL=${1:-zamba2}
DEVICE=${2:-auto}

# ── chunked sweep parameters ─────────────────────────────────────────────────
# CHUNKED_PCT: kernel 호출당 토큰 수 범위.
#   cooperative 안전 조건: batch × (pct // 256) × n_heads ≤ sm_count
#   Zamba2 (n_heads=112):  bs=1, sm=14 → max_pct = 14 × 256 / 112 = 32 → pct=256부터 시작
#   256 포함으로 최소 안전 구간을 실측하고, 4096까지 overhead 특성 확인
CHUNKED_PCT=${CHUNKED_PCT:-"256 512 1024 2048 4096"}
CHUNKED_SM=${CHUNKED_SM:-"14 27 40 54 68 81 94 108"}
CHUNKED_SEQ=${CHUNKED_SEQ:-"512 1024 2048 4096 8192"}
CHUNKED_BS=${CHUNKED_BS:-"1 4 16 32"}

echo "========================================================"
echo " Stage 1: SM Scaling Sweep"
echo "   model   = $MODEL"
echo "   device  = $DEVICE"
echo "   job     = ${SLURM_JOB_ID:-local}"
echo "   GPU     = $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "   started = $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================"
echo ""
echo "  [chunked sweep params]"
echo "    PCT_TOKENS = $CHUNKED_PCT"
echo "    SM_COUNTS  = $CHUNKED_SM"
echo "    SEQ_LENS   = $CHUNKED_SEQ"
echo "    BATCH_SIZE = $CHUNKED_BS"

_T0=$(date +%s)
_elapsed() { local t=$(( $(date +%s) - _T0 )); printf '%dm%02ds' $(( t/60 )) $(( t%60 )); }

# ── [1/6] SSM wave-model sweep ───────────────────────────────────────────────
# 전체 SM에서만 직접 측정 후 wave model로 각 SM 수별 latency를 합성.
# Green Context를 사용하지 않으므로 cooperative barrier 문제 없음.
echo ""
echo "── [1/6] $(date '+%H:%M:%S')  SSM prefill SM scaling sweep (wave-model analytical) …"
python stage1_sm_scaling/run_ssm_prefill_sweep.py \
    --model  "$MODEL" \
    --device "$DEVICE" \
    --skip-verify
echo "   done  ($(_elapsed))"

# ── [2/6] SSM chunked prefill sweep ──────────────────────────────────────────
# kernel 호출을 prefill_chunk_tokens 단위로 분할해 cooperative barrier를 우회.
# 각 kernel call 전에 Green Context로 SM 수를 제한 → 직접 latency 측정.
# 기존 wave-model 대비: 실측값이므로 단일 kernel 이상의 overhead 포함.
echo ""
echo "── [2/6] $(date '+%H:%M:%S')  SSM chunked prefill SM sweep (direct measurement) …"
python stage1_sm_scaling/run_chunked_ssm_sweep.py \
    --model  "$MODEL" \
    --device "$DEVICE" \
    --prefill-chunk-tokens $CHUNKED_PCT \
    --sm-counts            $CHUNKED_SM  \
    --seq-lens             $CHUNKED_SEQ \
    --batch-sizes          $CHUNKED_BS  \
    --n-warmup  3 \
    --n-measure 10 \
    --output-dir results/stage1/chunked/
echo "   done  ($(_elapsed))"

# ── [3/6] SSM torch scan sweep ───────────────────────────────────────────────
# PyTorch chunked scan으로 wave model 정확도 검증 (cooperative barrier 없음).
# wave-model 합성값 vs 직접 측정값의 MAPE를 plot_compare_modules.py에서 비교.
echo ""
echo "── [3/6] $(date '+%H:%M:%S')  SSM prefill SM scaling sweep (torch scan, for validation) …"
python stage1_sm_scaling/run_ssm_prefill_sweep.py \
    --model  "$MODEL" \
    --device "$DEVICE" \
    --force-pytorch-scan \
    --skip-verify
echo "   done  ($(_elapsed))"

# ── [4/6] Attention sweep ─────────────────────────────────────────────────────
echo ""
echo "── [4/6] $(date '+%H:%M:%S')  Attention prefill SM scaling sweep …"
python stage1_sm_scaling/run_attn_prefill_sweep.py \
    --model  "$MODEL" \
    --device "$DEVICE"
echo "   done  ($(_elapsed))"

# ── [5/6] MLP sweep ───────────────────────────────────────────────────────────
echo ""
echo "── [5/6] $(date '+%H:%M:%S')  MLP prefill SM scaling sweep …"
python stage1_sm_scaling/run_mlp_prefill_sweep.py \
    --model  "$MODEL" \
    --device "$DEVICE"
echo "   done  ($(_elapsed))"

# ── [6/6] Analysis + Plots ────────────────────────────────────────────────────
echo ""
echo "── [6/6] $(date '+%H:%M:%S')  Analysis + Plots …"

# chunked CSV 경로 자동 탐색 (가장 최신 파일)
CHUNKED_CSV=$(ls results/stage1/chunked/ssm_chunked_${MODEL}_*.csv 2>/dev/null \
              | sort | tail -1 || true)

# chunked 결과 분석: cooperative_safe 검증, overhead 특성, SM scaling curve
if [ -n "$CHUNKED_CSV" ]; then
    echo "   [analyze] $CHUNKED_CSV"
    python stage1_sm_scaling/analyze_chunk_size.py \
        --csv "$CHUNKED_CSV" \
        --wave-csv "$(ls results/stage1/ssm_scaling_${MODEL}_*.csv 2>/dev/null \
                      | grep -v torchscan | sort | tail -1 || true)" \
        2>/dev/null || echo "   [warn] analyze_chunk_size.py 실패 — 계속"
else
    echo "   [warn] chunked CSV 없음 — 분석 skip"
fi

# 기존 saturation / SRM / SM-split 플롯
python stage1_sm_scaling/plot_saturation.py  --model "$MODEL" 2>/dev/null \
    || python stage1_sm_scaling/plot_saturation.py 2>/dev/null || true
python stage1_sm_scaling/plot_srm.py         --model "$MODEL" 2>/dev/null \
    || python stage1_sm_scaling/plot_srm.py         2>/dev/null || true
python stage1_sm_scaling/plot_sm_split.py    --model "$MODEL" 2>/dev/null \
    || python stage1_sm_scaling/plot_sm_split.py    2>/dev/null || true

# 모듈 비교 플롯 — chunked CSV 있으면 overlay 포함
if [ -n "$CHUNKED_CSV" ]; then
    python stage1_sm_scaling/plot_compare_modules.py \
        --model "$MODEL" \
        --ssm-chunked-csv "$CHUNKED_CSV" \
        2>/dev/null || true
else
    python stage1_sm_scaling/plot_compare_modules.py \
        --model "$MODEL" \
        2>/dev/null || true
fi

echo "   done  ($(_elapsed))"

echo ""
echo "========================================================"
echo " Stage 1 Done  (results → results/stage1/)"
echo "   chunked → results/stage1/chunked/"
echo "   elapsed = $(_elapsed)"
echo "========================================================"
