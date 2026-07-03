#!/bin/bash
#SBATCH --job-name=p1_4_nh_smsens
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:35:00
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
SMFILE="$HERE/.nh_decode_sm_${SLURM_JOB_ID}"
export SGLANG_NH_LAYER_TIMING=1 PDMUX_FIXED_DECODE_SM_FILE="$SMFILE" PDMUX_LA_FLOOR_SM=16
echo full > "$SMFILE"
PORT=31056
SRVLOG="$HERE/p1_4_srv_${SLURM_JOB_ID}.log"
RESULT="$HERE/p1_4_nh_smsens_${SLURM_JOB_ID}.txt"; : > "$RESULT"

python -m sglang.launch_server \
  --model-path nvidia/Nemotron-H-8B-Base-8K --trust-remote-code --dtype bfloat16 \
  --disable-cuda-graph --disable-piecewise-cuda-graph --disable-radix-cache \
  --mem-fraction-static 0.8 --max-running-requests 40 --context-length 4096 \
  --host 127.0.0.1 --port $PORT > "$SRVLOG" 2>&1 &
SRV=$!
ok=0
for i in $(seq 1 150); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health 2>/dev/null||true)" = 200 ] && { ok=1; echo "healthy after ${i}x5s"; break; }
  kill -0 $SRV 2>/dev/null || { echo "SERVER DIED"; break; }; sleep 5
done

if [ "$ok" = 1 ]; then
  for MODE in full 64 44 32 16 layer_aware; do
    echo "$MODE" > "$SMFILE"
    echo "--- mode=$MODE: decode load ---"
    pids=""
    for r in $(seq 1 32); do
      curl -s http://127.0.0.1:$PORT/generate -H 'Content-Type: application/json' \
        -d '{"text":"Write a very long detailed story about a robot exploring a distant planet.","sampling_params":{"max_new_tokens":220,"temperature":0.7,"ignore_eos":true}}' >/dev/null 2>&1 &
      pids="$pids $!"
    done
    wait $pids; sleep 2
    line=$(grep "NHLT mode=$MODE " "$SRVLOG" | tail -1)
    echo "RESULT $line" | tee -a "$RESULT"
  done
else
  echo "SERVER_FAILED"; tail -30 "$SRVLOG" | tee -a "$RESULT"
fi
kill $SRV 2>/dev/null; sleep 3; kill -9 $SRV 2>/dev/null
echo "===== DONE ====="; echo "=== RESULTS ==="; cat "$RESULT"
