#!/bin/bash
# =============================================================================
# check_gate_g1.sh — G1 게이트 판정 (Phase 1 → Phase 2 게이트)
#
# Phase 1 (S1.3) 결과를 읽어 Option A 확정 여부를 판정한다.
# GPU 불필요; CPU 전용 잡으로 실행 가능.
#
# 입력 (자동 탐색):
#   results/stage1/g1_gate_{model}_{device}.csv
#     OR results/stage1/free_sm_zone_{model}_{device}.csv
#
# 판정 기준:
#   max(g1_potential) = max(free_sm_fraction × decode_sm_sensitivity)
#     > 0.10  → G1 통과: Option A 확정, Phase 2 (S2.2 co-exec) 진행
#     0.05~0.10 → 경계: 개별 layer_type 별로 재검토
#     < 0.05  → G1 미달: SM 재할당 이득 미미, Option B 승격 검토
#
# 이 스크립트는 판정 결과를 출력하지만 최종 결정은 사람이 한다.
# [人] G1 게이트는 CC 가 아닌 사람이 확정한다 (워크플로우 다이어그램 참조).
#
# Usage:
#   bash   slurm/check_gate_g1.sh              # zamba2
#   bash   slurm/check_gate_g1.sh falcon_h1
#   sbatch slurm/check_gate_g1.sh zamba2
# =============================================================================
#SBATCH -J g1-gate-check
#SBATCH -p amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:0
#SBATCH --time=00:05:00
#SBATCH --comment=pytorch
#SBATCH -o /scratch/%u/whlee/prefill-layer-alloc/logs/g1_gate_%j.log
#SBATCH -e /scratch/%u/whlee/prefill-layer-alloc/logs/g1_gate_%j.err

set -euo pipefail

module load conda/pytorch_2.9.1_cuda13 2>/dev/null || true

source /scratch/$USER/whlee/prefill-layer-alloc/bin/activate
cd /scratch/$USER/whlee/prefill-layer-alloc

MODEL=${1:-zamba2}

echo "========================================================"
echo " G1 Gate Check — Phase 1 → Phase 2 판정"
echo "   model   = $MODEL"
echo "   job     = ${SLURM_JOB_ID:-local}"
echo "   started = $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================"

python - "$MODEL" <<'PYEOF'
import sys, glob, pathlib
import pandas as pd

MODEL = sys.argv[1]

# ── CSV 탐색 ──────────────────────────────────────────────────────────────
g1_files   = sorted(glob.glob(f"results/stage1/g1_gate_{MODEL}_*.csv"))
free_files  = sorted(glob.glob(f"results/stage1/free_sm_zone_{MODEL}_*.csv"))
dec_files   = sorted(glob.glob(f"results/stage1/decode_sm_{MODEL}_*.csv"))

if not g1_files and not free_files:
    print(f"\n  [ERROR] g1_gate / free_sm_zone CSV 없음.")
    print(f"  run_stage1_decode_sweep.sh 를 먼저 실행하십시오.")
    sys.exit(1)

# ── 데이터 로드 ────────────────────────────────────────────────────────────
if g1_files:
    df = pd.read_csv(g1_files[-1])
    src = g1_files[-1]
else:
    df = pd.read_csv(free_files[-1])
    src = free_files[-1]
    # decode_sm_sensitivity 가 없으면 경고
    if "decode_sm_sensitivity" not in df.columns:
        print("  [WARN] decode_sm_sensitivity 없음 — g1_potential 계산 불가")
        print(f"  free_sm_zone 만 출력:\n{df.to_string()}")
        sys.exit(0)
    if "g1_potential" not in df.columns:
        df["g1_potential"] = df["free_ratio"] * df.get("decode_sm_sensitivity", 0.0)

print(f"\n  입력: {src}\n")

# ── G1-gate 테이블 출력 ───────────────────────────────────────────────────
display_cols = [c for c in [
    "model_name", "ssm_type", "layer_type",
    "seq_len", "batch_size",
    "free_ratio", "free_sm_fraction",
    "saturation_sm", "total_sm",
    "decode_sm_sensitivity",
    "g1_potential",
] if c in df.columns]

print(df[display_cols].to_string(index=False))

# ── 판정 ──────────────────────────────────────────────────────────────────
if "g1_potential" in df.columns:
    max_g1 = df["g1_potential"].max()
    mean_g1 = df["g1_potential"].mean()
    best_row = df.loc[df["g1_potential"].idxmax()]

    print(f"\n  ─────────────────────────────────────────────────")
    print(f"  max  g1_potential = {max_g1:.4f}")
    print(f"  mean g1_potential = {mean_g1:.4f}")
    print(f"  best row:")
    for c in display_cols:
        if c in best_row.index:
            print(f"    {c:30s} = {best_row[c]}")

    print(f"\n  ─────────────────────────────────────────────────")
    print(f"  【G1 게이트 판정】")
    if max_g1 > 0.10:
        print(f"  ✓  PASS (max_g1={max_g1:.4f} > 0.10)")
        print(f"     → Option A 확정: Phase 2 (S2.2 co-exec) 진행")
    elif max_g1 > 0.05:
        print(f"  △  BORDERLINE (max_g1={max_g1:.4f} ∈ [0.05, 0.10])")
        print(f"     → layer_type 별로 재검토 후 사람이 결정")
    else:
        print(f"  ✗  FAIL (max_g1={max_g1:.4f} ≤ 0.05)")
        print(f"     → SM 재할당 이득 미미: Option B 승격 또는 scope 축소 검토")
    print(f"  ─────────────────────────────────────────────────")
    print(f"\n  [人] 최종 결정은 사람이 한다 (워크플로우 D0 / G1 참조)")

PYEOF

echo ""
echo "========================================================"
echo " 다음 단계 (G1 통과 시):"
echo "   sbatch slurm/run_stage3_coexec.sh $MODEL"
echo "========================================================"
