#!/usr/bin/env bash
# download_datasets.sh — 로그인 노드에서 HuggingFace 데이터셋 사전 캐싱
#
# 컴퓨트 노드는 HF_HUB_OFFLINE=1 이므로 반드시 로그인 노드에서 먼저 실행.
#
# 전략: HF Hub 캐시 메커니즘 대신 JSON 파일을 고정 경로에 직접 다운로드.
#   → 온라인(URL 해시) / 오프라인(파일명 해시) 불일치 문제 원천 차단.
#   → 저장 위치: $REPO_ROOT/hf_cache/raw/
#
# Usage:
#   bash workspace/serving-eval/slurm/download_datasets.sh
#   bash workspace/serving-eval/slurm/download_datasets.sh --longbench
set -euo pipefail

REPO_ROOT="/scratch/$USER/whlee/prefill-layer-alloc"
HF_CACHE="$REPO_ROOT/hf_cache"
RAW_DIR="$HF_CACHE/raw"
VENV_DIR="$REPO_ROOT/workspace/serving-eval/.venv"

INCLUDE_LONGBENCH=0
for arg in "$@"; do
    [ "$arg" = "--longbench" ] && INCLUDE_LONGBENCH=1
done

echo "==================================================================="
echo " HuggingFace 데이터셋 사전 다운로드 (로그인 노드 전용)"
echo "==================================================================="
echo "  raw 저장 경로 : $RAW_DIR"
echo "  venv          : $VENV_DIR"
echo "==================================================================="

if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "[ERROR] serving-eval venv 없음. 먼저 setup_env.sh 를 실행하십시오."
    exit 1
fi

source "$VENV_DIR/bin/activate"

mkdir -p "$RAW_DIR"

unset HF_HUB_OFFLINE
unset HF_DATASETS_OFFLINE
unset TRANSFORMERS_OFFLINE

echo ""
echo "── ShareGPT 다운로드 …"
python - "$RAW_DIR" <<'EOF'
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download
import shutil

raw_dir = Path(sys.argv[1])
dest = raw_dir / "ShareGPT_V3_unfiltered_cleaned_split.json"

if dest.exists():
    print(f"  이미 존재: {dest}")
else:
    src = hf_hub_download(
        repo_id="anon8231489123/ShareGPT_Vicuna_unfiltered",
        filename="ShareGPT_V3_unfiltered_cleaned_split.json",
        repo_type="dataset",
    )
    shutil.copy(src, dest)
    print(f"  완료: {dest}  ({dest.stat().st_size / 1e6:.1f} MB)")
EOF

if [ "$INCLUDE_LONGBENCH" = "1" ]; then
    echo ""
    echo "── LongBench 다운로드 (gov_report / summ_screen_fd / qasper) …"
    python - "$RAW_DIR" <<'EOF'
import sys, os
from pathlib import Path
from datasets import load_dataset

raw_dir = Path(sys.argv[1])
lb_cache = raw_dir / "longbench_cache"
lb_cache.mkdir(parents=True, exist_ok=True)

for task in ("gov_report", "summ_screen_fd", "qasper"):
    try:
        ds = load_dataset("THUDM/LongBench", task, split="test",
                          trust_remote_code=True,
                          cache_dir=str(lb_cache))
        print(f"  {task}: {len(ds):,} rows")
    except Exception as e:
        print(f"  [warn] {task} 실패: {e}")
EOF
fi

echo ""
echo "==================================================================="
echo " 다운로드 완료."
echo " raw 파일 목록:"
ls -lh "$RAW_DIR"/ | grep -v '^total' | sed 's/^/   /'
echo "==================================================================="
