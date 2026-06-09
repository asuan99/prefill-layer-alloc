#!/usr/bin/env bash
# download_datasets.sh — 로그인 노드에서 HuggingFace 데이터셋 사전 캐싱
#
# 컴퓨트 노드는 HF_HUB_OFFLINE=1 이므로 반드시 로그인 노드에서 먼저 실행.
# 캐시 위치: $REPO_ROOT/hf_cache  (compute 노드와 공유 스크래치에 위치)
#
# Usage:
#   bash workspace/serving-eval/slurm/download_datasets.sh
#   bash workspace/serving-eval/slurm/download_datasets.sh --longbench  # LongBench 도 포함
set -euo pipefail

REPO_ROOT="/scratch/$USER/whlee/prefill-layer-alloc"
HF_CACHE="$REPO_ROOT/hf_cache"
VENV_DIR="$REPO_ROOT/workspace/serving-eval/.venv"

INCLUDE_LONGBENCH=0
for arg in "$@"; do
    [ "$arg" = "--longbench" ] && INCLUDE_LONGBENCH=1
done

echo "==================================================================="
echo " HuggingFace 데이터셋 사전 다운로드 (로그인 노드 전용)"
echo "==================================================================="
echo "  캐시 경로 : $HF_CACHE"
echo "  venv      : $VENV_DIR"
echo "==================================================================="

if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "[ERROR] serving-eval venv 없음. 먼저 setup_env.sh 를 실행하십시오."
    exit 1
fi

source "$VENV_DIR/bin/activate"

mkdir -p "$HF_CACHE"

# HF_HUB_OFFLINE 을 명시적으로 해제하여 다운로드 허용
unset HF_HUB_OFFLINE
unset HF_DATASETS_OFFLINE
unset TRANSFORMERS_OFFLINE
export HF_DATASETS_CACHE="$HF_CACHE"

echo ""
echo "── ShareGPT 다운로드 …"
python - <<'EOF'
import os
from datasets import load_dataset
ds = load_dataset(
    "anon8231489123/ShareGPT_Vicuna_unfiltered",
    data_files="ShareGPT_V3_unfiltered_cleaned_split.json",
    split="train",
    cache_dir=os.environ["HF_DATASETS_CACHE"],
)
print(f"  완료: {len(ds):,} rows → {os.environ['HF_DATASETS_CACHE']}")
EOF

if [ "$INCLUDE_LONGBENCH" = "1" ]; then
    echo ""
    echo "── LongBench 다운로드 (gov_report / summ_screen_fd / qasper) …"
    python - <<'EOF'
import os
from datasets import load_dataset
for task in ("gov_report", "summ_screen_fd", "qasper"):
    try:
        ds = load_dataset("THUDM/LongBench", task, split="test",
                          trust_remote_code=True,
                          cache_dir=os.environ["HF_DATASETS_CACHE"])
        print(f"  {task}: {len(ds):,} rows")
    except Exception as e:
        print(f"  [warn] {task} 실패: {e}")
EOF
fi

echo ""
echo "==================================================================="
echo " 다운로드 완료."
echo " 이후 sbatch 시 HF_DATASETS_CACHE 가 자동으로 아래 경로로 설정됩니다:"
echo "   $HF_CACHE"
echo "==================================================================="
