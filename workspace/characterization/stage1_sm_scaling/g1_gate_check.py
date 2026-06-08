"""
g1_gate_check.py — G1 게이트 판정 (Phase 1 → Phase 2).

bash 스크립트 없이 직접 실행한다.

Usage (characterization/ 디렉토리에서):
    python stage1_sm_scaling/g1_gate_check.py
    python stage1_sm_scaling/g1_gate_check.py --model falcon_h1
    python stage1_sm_scaling/g1_gate_check.py --model all
    python stage1_sm_scaling/g1_gate_check.py --results-dir results/stage1

판정 기준:
    max(g1_potential) > 0.10  → G1 통과: Phase 2 진행
    max(g1_potential) ∈ [0.05, 0.10] → BORDERLINE: 재검토
    max(g1_potential) < 0.05  → G1 미달: SM 재할당 이득 없음
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

# workspace/ 를 sys.path 에 추가 (shared.loaders 사용 시 필요)
_here = Path(__file__).parent.parent  # characterization/
sys.path.insert(0, str(_here.parent))  # workspace/
sys.path.insert(0, str(_here))

import pandas as pd


def check_model(model: str, results_dir: Path) -> bool:
    """판정 실행. PASS=True, FAIL=False."""

    g1_files   = sorted(results_dir.glob(f"g1_gate_{model}_*.csv"))
    free_files  = sorted(results_dir.glob(f"free_sm_zone_{model}*.csv"))

    if not g1_files and not free_files:
        print(f"\n  [ERROR] CSV 없음: {results_dir}/")
        print(f"  필요 파일: free_sm_zone_{model}*.csv  또는  g1_gate_{model}_*.csv")
        print(f"\n  생성 방법:")
        print(f"    python stage1_sm_scaling/plot_saturation.py --model {model}")
        print(f"    (decode_sm_sensitivity 포함: sbatch slurm/run_stage1_decode_sweep.sh {model})")
        return False

    src = g1_files[-1] if g1_files else free_files[-1]
    df  = pd.read_csv(src)
    print(f"\n  입력: {src.name}")

    # free_ratio 컬럼 정규화
    if "free_ratio" not in df.columns and "free_sm_fraction" in df.columns:
        df["free_ratio"] = df["free_sm_fraction"]

    # g1_potential 계산
    if "g1_potential" not in df.columns:
        if "decode_sm_sensitivity" in df.columns and "free_ratio" in df.columns:
            df["g1_potential"] = df["free_ratio"] * df["decode_sm_sensitivity"]
        elif "free_ratio" in df.columns:
            df["g1_potential"] = df["free_ratio"]
        else:
            df["g1_potential"] = 0.0

    # 테이블 출력
    cols = [c for c in [
        "model_name", "ssm_type", "seq_len", "batch_size",
        "total_sm", "saturation_sm", "free_sm",
        "free_ratio", "decode_sm_sensitivity", "g1_potential",
    ] if c in df.columns]
    print()
    print(df[cols].to_string(index=False))

    # 판정
    max_g1  = df["g1_potential"].max()
    mean_g1 = df["g1_potential"].mean()
    print(f"\n  max  g1_potential = {max_g1:.4f}")
    print(f"  mean g1_potential = {mean_g1:.4f}")
    print()

    if max_g1 > 0.10:
        print(f"  ✓  G1 PASS (max={max_g1:.4f} > 0.10)")
        print(f"     → Option A 확정: Phase 2 S2.2 co-exec 진행")
        verdict = True
    elif max_g1 > 0.05:
        print(f"  △  BORDERLINE (max={max_g1:.4f} ∈ [0.05, 0.10])")
        print(f"     → layer_type 별 재검토 후 사람이 결정")
        verdict = False
    else:
        print(f"  ✗  G1 FAIL (max={max_g1:.4f} ≤ 0.05)")
        print(f"     → SM 재할당 이득 미미")
        if "free_ratio" in df.columns and df["free_ratio"].max() < 0.02:
            print(f"     ※ free_ratio ≈ 0: SSM 이 모든 SM 을 포화시킴")
            n_heads = None
            try:
                from shared.loaders import get_model_config
                cfg = get_model_config(model)
                n_heads = cfg.get("ssm", {}).get("n_heads")
            except Exception:
                pass
            total_sm = int(df["total_sm"].iloc[0]) if "total_sm" in df.columns else "?"
            if n_heads and n_heads > total_sm:
                print(f"     n_heads={n_heads} > total_sm={total_sm}: 구조적으로 free SM 없음")
            elif n_heads:
                print(f"     n_heads={n_heads}, total_sm={total_sm}")
                print(f"     → 측정 구간에서 throughput 이 선형 증가 (saturation 미발견)")
                print(f"     → ssm_scaling CSV 원본 데이터 확인 권장")
        verdict = False

    print(f"\n  [人] 최종 결정은 사람이 한다")
    return verdict


def main() -> None:
    p = argparse.ArgumentParser(description="G1 Gate Check")
    p.add_argument("--model", default="zamba2",
                   help="모델 이름 또는 'all' (default: zamba2)")
    p.add_argument("--results-dir", type=Path,
                   default=Path(__file__).parent.parent / "results" / "stage1",
                   help="Stage 1 결과 디렉토리")
    args = p.parse_args()

    print("=" * 60)
    print(" G1 Gate Check — Phase 1 → Phase 2 판정")
    print(f"   results-dir = {args.results_dir}")
    print("=" * 60)

    models = ["zamba2", "falcon_h1"] if args.model == "all" else [args.model]

    results = {}
    for model in models:
        print(f"\n{'━' * 60}")
        print(f" Model: {model}")
        print(f"{'━' * 60}")
        results[model] = check_model(model, args.results_dir)

    print("\n" + "=" * 60)
    print(" 최종 요약")
    print("=" * 60)
    for model, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {model}: {status}")

    print()
    print(" 다음 단계 (G1 통과 모델):")
    for model, passed in results.items():
        if passed:
            print(f"   sbatch slurm/run_stage3_coexec.sh {model}")
    if not any(results.values()):
        print("   (통과 모델 없음 — Option B 승격 검토)")
    print("=" * 60)


if __name__ == "__main__":
    main()
