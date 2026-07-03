#!/bin/bash
#SBATCH --job-name=p1_2_zamba2_real
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:25:00
#SBATCH --comment=pytorch
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -uo pipefail
module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0
export PS1="" PATH="$HOME/.local/bin:$PATH"
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate
HERE=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/triage
export HF_HOME=/scratch/ehmoon/whlee/prefill-layer-alloc/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="$HERE/.triton_cache"
PORT=31029; SRVLOG="$HERE/p1_2_server_${SLURM_JOB_ID}.log"

echo "===== instantiate Zamba2-2.7B (dummy weights), NO pdmux, disable cuda graph ====="
python -m sglang.launch_server \
  --model-path Zyphra/Zamba2-2.7B \
  --trust-remote-code --dtype bfloat16 \
  --attention-backend flex_attention --disable-cuda-graph --disable-radix-cache \
  --mem-fraction-static 0.6 --max-running-requests 8 --context-length 2048 \
  --host 127.0.0.1 --port $PORT > "$SRVLOG" 2>&1 &
SRV=$!
ok=0
for i in $(seq 1 72); do
  code=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:$PORT/health 2>/dev/null || true)
  [ "$code" = "200" ] && { ok=1; echo "healthy after ${i}0s"; break; }
  kill -0 $SRV 2>/dev/null || { echo "SERVER DIED"; break; }
  sleep 10
done
if [ "$ok" = 1 ]; then
  echo "===== GENERATE ====="
  curl -s http://127.0.0.1:$PORT/generate -H 'Content-Type: application/json' \
    -d '{"text":"The capital of France is","sampling_params":{"max_new_tokens":8,"temperature":0}}' | head -c 500; echo
  echo "===== P1.2 RESULT: ZAMBA2_REAL_OK ====="
else
  echo "===== P1.2 RESULT: ZAMBA2_REAL_FAILED ====="; echo "--- server log tail ---"; tail -50 "$SRVLOG"
fi
kill $SRV 2>/dev/null; sleep 2; kill -9 $SRV 2>/dev/null
echo "===== DONE ====="
