#!/bin/bash
# P1.0 install: sgl-kernel 0.3.21 (abi3 wheel) + sglang 0.5.9 (PyPI, --no-deps)
# into the isolated overlay venv. --no-deps keeps the cluster's CUDA-13
# torch 2.9.1 / flashinfer 0.6.10 / cuda-python 13.2 / mamba_ssm untouched.
set -uo pipefail
module load conda/pytorch_2.9.1_cuda13 2>/dev/null
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate
export PIP_CACHE_DIR=/scratch/ehmoon/whlee/.pip_cache
echo "=== python: $(which python) | prefix: $(python -c 'import sys;print(sys.prefix)') ==="
echo "=== [1/2] sgl-kernel 0.3.21 ==="
python -m pip install --no-deps "sgl-kernel==0.3.21" 2>&1 | tail -4
echo "=== [2/2] sglang 0.5.9 (--no-deps) ==="
python -m pip install --no-deps "sglang==0.5.9" 2>&1 | tail -4
echo "=== verify land in venv ==="
python -c "import sgl_kernel; print('sgl_kernel', sgl_kernel.__file__)" 2>&1 | tail -2
python -c "import sglang; print('sglang', sglang.__version__, sglang.__file__)" 2>&1 | tail -2
echo "DONE_INSTALL"
