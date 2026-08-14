#!/bin/bash
#SBATCH --job-name=p1_0_pdmux_boot
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:25:00
#SBATCH --comment="field=efficientai;appl=pytorch"
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -uo pipefail

module load conda/pytorch_2.9.1_cuda13
module load cuda/13.0.2
module load gcc/15.2.0
export PS1=""
export PATH="$HOME/.local/bin:$PATH"   # ninja for flashinfer/triton JIT
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate

HERE=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/triage
export HF_HOME=/scratch/ehmoon/whlee/prefill-layer-alloc/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TRITON_CACHE_DIR="$HERE/.triton_cache"
export SGLANG_CACHE_DIR="$HERE/.sglang_cache"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
mkdir -p "$TRITON_CACHE_DIR" "$SGLANG_CACHE_DIR"
PORT=31007
SRVLOG="$HERE/p1_0_server_${SLURM_JOB_ID}.log"

echo "===== ENV ====="
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader
python -c "import torch,sgl_kernel,sglang;print('torch',torch.__version__,'| sglang',sglang.__version__)"

echo "===== LAUNCH pdmux server (Qwen2.5-0.5B dummy weights) ====="
python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-0.5B-Instruct \
  --load-format dummy --trust-remote-code \
  --enable-pdmux --pdmux-config-path "$HERE/pdmux_a100_smoke.yml" \
  --chunked-prefill-size -1 --disable-overlap-schedule --disable-radix-cache \
  --mem-fraction-static 0.55 --max-running-requests 32 \
  --host 127.0.0.1 --port $PORT > "$SRVLOG" 2>&1 &
SRV_PID=$!

echo "waiting for /health ..."
ok=0
for i in $(seq 1 60); do
  code=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:$PORT/health 2>/dev/null || true)
  if [ "$code" = "200" ]; then ok=1; echo "server healthy after ${i}0s"; break; fi
  if ! kill -0 $SRV_PID 2>/dev/null; then echo "SERVER DIED during startup"; break; fi
  sleep 10
done

echo "===== PDMUX INIT LOG (green-ctx / stream groups / sm_counts) ====="
grep -iE "PD-Multiplexing|stream group|sm_count|green|multiplex|partition|Prefill|Decode" "$SRVLOG" | head -30

if [ "$ok" = "1" ]; then
  echo "===== GENERATE REQUEST ====="
  curl -s http://127.0.0.1:$PORT/generate -H 'Content-Type: application/json' \
    -d '{"text":"The capital of France is","sampling_params":{"max_new_tokens":16,"temperature":0}}' \
    | head -c 800; echo
  echo "===== P1.0 RESULT: PDMUX_BOOT_OK ====="
else
  echo "===== P1.0 RESULT: PDMUX_BOOT_FAILED (see server log tail below) ====="
  tail -40 "$SRVLOG"
fi

kill $SRV_PID 2>/dev/null; sleep 3; kill -9 $SRV_PID 2>/dev/null
echo "===== DONE ====="
