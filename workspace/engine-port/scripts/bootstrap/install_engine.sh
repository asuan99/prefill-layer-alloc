#!/bin/bash
# Environment bootstrap: keep the editable SGLang dev tree and CUDA-13 stack
# aligned, while installing the pure-Python runtime dependencies explicitly.
# --no-deps keeps the cluster's CUDA-13 torch/flashinfer/mamba stack untouched.
set -euo pipefail
module load conda/pytorch_2.9.1_cuda13 2>/dev/null
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate
export PIP_CACHE_DIR=/scratch/ehmoon/whlee/.pip_cache
echo "=== python: $(which python) | prefix: $(python -c 'import sys;print(sys.prefix)') ==="
echo "=== [1/3] verify CUDA-13 sgl-kernel ==="
timeout 60s python - <<'PY'
import sgl_kernel
print("sgl_kernel", getattr(sgl_kernel, "__version__", "unknown"), sgl_kernel.__file__)
PY
echo "=== [2/3] openai-harmony 0.0.4 ==="
python -m pip install --no-deps "openai-harmony==0.0.4" 2>&1 | tail -4
echo "=== [3/3] verify editable runtime and dependencies ==="
python -c "import sglang; print('sglang', sglang.__version__, sglang.__file__)" 2>&1 | tail -2
python -c "import sgl_kernel; print('sgl_kernel', sgl_kernel.__file__)" 2>&1 | tail -2
python -c "import openai_harmony; print('openai_harmony', openai_harmony.__file__)" 2>&1 | tail -2
echo "DONE_INSTALL"
