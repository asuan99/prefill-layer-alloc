#!/usr/bin/env bash
# serving-eval/setup_env.sh — self-contained venv for vLLM V1 serving evaluation.
#
# Intentionally SEPARATE from characterization/.venv:
#   - characterization/ : mamba-ssm custom build (Green Context patch, profiling)
#   - serving-eval/     : vLLM + transformers (production inference stack)
#     vLLM installs mamba-ssm and causal-conv1d automatically as dependencies;
#     do NOT manually install them or mix with characterization's custom versions.
#   Do NOT activate both venvs in the same shell.
#
# Usage:
#   bash workspace/serving-eval/setup_env.sh           # create/update .venv
#   source workspace/serving-eval/.venv/bin/activate   # activate
#
# Requirements:
#   - CUDA 12.1+ driver
#   - Python 3.10+
#   - ~20 GB disk for model weights (downloaded on first run_serving.py run)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
PYTHON="${PYTHON:-python3}"

echo "==================================================================="
echo " serving-eval venv setup"
echo "==================================================================="
echo "  Target : ${VENV_DIR}"
echo "  Python : $($PYTHON --version 2>&1)"
echo "  Note   : Independent of characterization/.venv"
echo "           Do NOT mix mamba-ssm / triton packages here."
echo "==================================================================="

$PYTHON -m venv "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

pip install --upgrade pip wheel setuptools

echo ""
echo "--- Installing vLLM V1 with hybrid model support ---"
# vLLM ≥ 0.6.3: Zamba2-7B support confirmed
# vLLM ≥ 0.8.x: Nemotron-H support (verify with nvidia/Nemotron-H-8B-Base model card)
# Falcon-H1: check vLLM issue tracker before enabling
pip install "vllm>=0.6.3" \
    --extra-index-url https://download.pytorch.org/whl/cu121

echo ""
echo "--- Installing transformers + acceleration ---"
pip install \
    "transformers>=4.45.0" \
    "accelerate>=0.28.0" \
    "tokenizers>=0.19.0"

echo ""
echo "--- Installing workload trace utilities ---"
pip install \
    "datasets>=2.18.0" \
    "openai>=1.12.0" \
    "aiohttp>=3.9.3"

echo ""
echo "--- Installing analytics ---"
pip install \
    "pandas>=2.1.0" \
    "numpy>=1.26.0" \
    "matplotlib>=3.8.0" \
    "seaborn>=0.13.0" \
    "tqdm>=4.66.0" \
    "pyyaml>=6.0.1"

echo ""
echo "==================================================================="
echo " Setup complete."
echo " Activate : source ${VENV_DIR}/bin/activate"
echo " Run      : python workspace/serving-eval/run_serving.py --model zamba2 --smoke"
echo "==================================================================="
