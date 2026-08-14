#!/bin/bash
# =============================================================================
# check_gate_g1.sh — G1 게이트 판정
#
# Usage:
#   bash   slurm/check_gate_g1.sh              # zamba2 (default)
#   bash   slurm/check_gate_g1.sh falcon_h1
#   bash   slurm/check_gate_g1.sh all
#   sbatch slurm/check_gate_g1.sh zamba2
# =============================================================================
#SBATCH -J g1-gate-check
#SBATCH -p amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:0
#SBATCH --time=00:05:00
#SBATCH --comment="field=efficientai;appl=pytorch"
#SBATCH -o /scratch/%u/whlee/prefill-layer-alloc/logs/g1_gate_%j.log
#SBATCH -e /scratch/%u/whlee/prefill-layer-alloc/logs/g1_gate_%j.err

# module load 제거 — login node 에서 conda module 이 shell 자체를 종료시킴.
# 사용자가 venv 를 이미 활성화한 상태에서 실행하므로 불필요.
# sbatch 환경에서는 아래 activate 로 처리.

REPO_ROOT="/scratch/$USER/whlee/prefill-layer-alloc"
CHAR_DIR="$REPO_ROOT/workspace/characterization"
RESULTS_DIR="$CHAR_DIR/results/stage1"

# venv activate (sbatch 환경용; interactive 에서는 이미 활성화돼 있으면 no-op)
if [ -f "$REPO_ROOT/bin/activate" ]; then
    source "$REPO_ROOT/bin/activate" 2>/dev/null || true
fi

MODEL=${1:-zamba2}

echo "========================================================"
echo " G1 Gate Check — Phase 1 → Phase 2 판정"
echo "   model   = $MODEL"
echo "   results = $RESULTS_DIR"
echo "   job     = ${SLURM_JOB_ID:-interactive}"
echo "========================================================"

python - "$MODEL" "$RESULTS_DIR" <<'PYEOF'
import sys, glob, os
import pandas as pd

MODEL       = sys.argv[1]
RESULTS_DIR = sys.argv[2]

models = ["zamba2", "falcon_h1"] if MODEL == "all" else [MODEL]

for model in models:
    if MODEL == "all":
        print(f"\n{'━'*60}")
        print(f" Model: {model}")
        print(f"{'━'*60}")

    g1_files   = sorted(glob.glob(f"{RESULTS_DIR}/g1_gate_{model}_*.csv"))
    free_files  = sorted(glob.glob(f"{RESULTS_DIR}/free_sm_zone_{model}*.csv"))

    if not g1_files and not free_files:
        print(f"\n  [ERROR] CSV 없음: {RESULTS_DIR}/")
        print(f"  free_sm_zone_{model}*.csv 또는 g1_gate_{model}_*.csv 필요")
        print(f"\n  생성:")
        print(f"    cd workspace/characterization")
        print(f"    python stage1_sm_scaling/plot_saturation.py --model {model}")
        continue

    src = g1_files[-1] if g1_files else free_files[-1]
    df  = pd.read_csv(src)
    print(f"\n  입력: {os.path.basename(src)}")

    # free_ratio 컬럼 정규화
    if "free_ratio" not in df.columns and "free_sm_fraction" in df.columns:
        df["free_ratio"] = df["free_sm_fraction"]

    # g1_potential 계산 (없으면 free_ratio × sensitivity, 그것도 없으면 free_ratio)
    if "g1_potential" not in df.columns:
        if "decode_sm_sensitivity" in df.columns and "free_ratio" in df.columns:
            df["g1_potential"] = df["free_ratio"] * df["decode_sm_sensitivity"]
        elif "free_ratio" in df.columns:
            df["g1_potential"] = df["free_ratio"]

    # 출력 컬럼
    cols = [c for c in ["model_name", "ssm_type", "seq_len", "batch_size",
                         "total_sm", "saturation_sm", "free_sm",
                         "free_ratio", "decode_sm_sensitivity", "g1_potential"]
            if c in df.columns]
    print()
    print(df[cols].to_string(index=False))

    # 판정
    if "g1_potential" in df.columns:
        max_g1 = df["g1_potential"].max()
        print(f"\n  max g1_potential = {max_g1:.4f}")
        print()
        if max_g1 > 0.10:
            print(f"  ✓  G1 PASS (> 0.10) → Option A 확정, Phase 2 진행")
        elif max_g1 > 0.05:
            print(f"  △  BORDERLINE ([0.05, 0.10]) → 재검토 후 사람이 결정")
        else:
            print(f"  ✗  G1 FAIL (≤ 0.05) → SM 재할당 이득 미미")
            if "free_ratio" in df.columns and df["free_ratio"].max() < 0.02:
                print(f"     ※ free_ratio ≈ 0: SSM 이 모든 SM 포화 (구조적 한계)")
        print()
        print(f"  [人] 최종 결정은 사람이 한다")
PYEOF

echo ""
echo "========================================================"
echo " 다음 단계 (G1 통과 시): sbatch slurm/run_stage3_coexec.sh $MODEL"
echo "========================================================"
