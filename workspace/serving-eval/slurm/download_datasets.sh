#!/usr/bin/env bash
# download_datasets.sh — 로그인 노드에서 HuggingFace 모델·데이터셋 사전 다운로드
#
# 컴퓨트 노드는 HF_HUB_OFFLINE=1 이므로 반드시 로그인 노드에서 먼저 실행.
#
# 전략:
#   모델   : snapshot_download → $REPO_ROOT/hf_cache/hub/  (HF_HOME 과 동일)
#   데이터셋: hf_hub_download  → $REPO_ROOT/hf_cache/raw/  (고정 경로, 해시 불일치 방지)
#
# Usage:
#   bash workspace/serving-eval/slurm/download_datasets.sh              # 기본 모델 + ShareGPT
#   bash workspace/serving-eval/slurm/download_datasets.sh --longbench  # LongBench 추가
#   bash workspace/serving-eval/slurm/download_datasets.sh --no-models  # 데이터셋만
set -euo pipefail

REPO_ROOT="/scratch/$USER/whlee/prefill-layer-alloc"
HF_CACHE="$REPO_ROOT/hf_cache"
HUB_CACHE="$HF_CACHE/hub"
RAW_DIR="$HF_CACHE/raw"
VENV_DIR="$REPO_ROOT/workspace/serving-eval/.venv"

INCLUDE_LONGBENCH=0
SKIP_MODELS=0
for arg in "$@"; do
    [ "$arg" = "--longbench" ]  && INCLUDE_LONGBENCH=1
    [ "$arg" = "--no-models" ]  && SKIP_MODELS=1
done

echo "==================================================================="
echo " HuggingFace 모델·데이터셋 사전 다운로드 (로그인 노드 전용)"
echo "==================================================================="
echo "  모델 캐시  : $HUB_CACHE"
echo "  raw 데이터 : $RAW_DIR"
echo "  venv       : $VENV_DIR"
echo "==================================================================="

if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "[ERROR] serving-eval venv 없음. 먼저 setup_env.sh 를 실행하십시오."
    exit 1
fi

source "$VENV_DIR/bin/activate"

mkdir -p "$HUB_CACHE" "$RAW_DIR"

unset HF_HUB_OFFLINE
unset HF_DATASETS_OFFLINE
unset TRANSFORMERS_OFFLINE
export HF_HOME="$HF_CACHE"
export HF_HUB_CACHE="$HUB_CACHE"

# ── 모델 가중치 다운로드 ────────────────────────────────────────────────────
if [ "$SKIP_MODELS" = "0" ]; then
    echo ""
    echo "── 모델 가중치 다운로드 (각 ~14-16 GB) …"
    python - "$HUB_CACHE" <<'EOF'
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

hub_cache = sys.argv[1]

models = [
    "Zyphra/Zamba2-7B-Instruct",
    "nvidia/Nemotron-H-8B-Base-8K",
    "tiiuae/Falcon-H1-7B-Instruct",
]

for repo_id in models:
    name = repo_id.split("/")[-1]
    print(f"\n  [{name}] 다운로드 중 …")
    try:
        path = snapshot_download(
            repo_id=repo_id,
            cache_dir=hub_cache,
            ignore_patterns=["*.gguf", "*.ggml"],
        )
        print(f"  [{name}] 완료: {path}")
    except Exception as e:
        print(f"  [{name}] 실패: {e}")
EOF
fi

# ── ShareGPT 데이터셋 ───────────────────────────────────────────────────────
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

# ── LongBench 데이터셋 (옵션) ──────────────────────────────────────────────
if [ "$INCLUDE_LONGBENCH" = "1" ]; then
    echo ""
    echo "── LongBench 다운로드 (gov_report / summ_screen_fd / qasper) …"
    python - "$RAW_DIR" <<'EOF'
import sys
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

# ── 최종 요약 ──────────────────────────────────────────────────────────────
echo ""
echo "==================================================================="
echo " 다운로드 완료."
echo " 모델 캐시:"
ls -lh "$HUB_CACHE"/ 2>/dev/null | grep -v '^total' | sed 's/^/   /' || echo "   (없음)"
echo " raw 데이터:"
ls -lh "$RAW_DIR"/ 2>/dev/null | grep -v '^total' | sed 's/^/   /' || echo "   (없음)"
echo "==================================================================="
