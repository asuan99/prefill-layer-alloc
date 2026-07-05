#!/bin/bash
#SBATCH --job-name=p1_7_gr_smsens
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:50:00
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
MODEL=ibm-granite/granite-4.0-h-micro-base
SMFILE="$HERE/.gr_sm_${SLURM_JOB_ID}"
export SGLANG_GRANITE_TIMING=1 PDMUX_FIXED_DECODE_SM_FILE="$SMFILE"
echo full > "$SMFILE"
PORT=$((32000 + SLURM_JOB_ID % 300))
SRVLOG="$HERE/p1_7_grsrv_${SLURM_JOB_ID}.log"
RESULT="$HERE/p1_7_gr_smsens_${SLURM_JOB_ID}.txt"; : > "$RESULT"
echo "MODEL=$MODEL PORT=$PORT" | tee -a "$RESULT"

python -m sglang.launch_server \
  --model-path "$MODEL" --trust-remote-code --dtype bfloat16 \
  --disable-cuda-graph --disable-piecewise-cuda-graph --disable-radix-cache \
  --mem-fraction-static 0.82 --max-running-requests 16 --context-length 4096 \
  --host 127.0.0.1 --port $PORT > "$SRVLOG" 2>&1 &
SRV=$!
ok=0
for i in $(seq 1 170); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health 2>/dev/null||true)" = 200 ] && { ok=1; echo healthy | tee -a "$RESULT"; break; }
  kill -0 $SRV 2>/dev/null || { echo "SERVER DIED" | tee -a "$RESULT"; break; }; sleep 5
done
if [ "$ok" = 1 ]; then
  echo -n "SANITY " | tee -a "$RESULT"
  curl -s http://127.0.0.1:$PORT/generate -H 'Content-Type: application/json' \
    -d '{"text":"The capital of France is","sampling_params":{"max_new_tokens":6,"temperature":0}}' | grep -oE '"text":"[^"]*"' | head -1 | tee -a "$RESULT"
  for CTX in 64 256 2048 4000; do
    PROMPT=$(python3 -c "print('the robot walked slowly '*$((CTX/4)), end='')")
    for SM in full 64 44 16; do
      echo "$SM@$CTX" > "$SMFILE"
      pids=""
      for r in $(seq 1 8); do
        curl -s http://127.0.0.1:$PORT/generate -H 'Content-Type: application/json' \
          --data-binary "$(python3 -c "import json; print(json.dumps({'text':'$PROMPT','sampling_params':{'max_new_tokens':80,'temperature':0.7,'ignore_eos':True}}))")" >/dev/null 2>&1 &
        pids="$pids $!"
      done
      wait $pids; sleep 2
      line=$(grep "GMHLT mode=$SM@$CTX " "$SRVLOG" | tail -1)
      echo "RESULT ctx=$CTX $line" | tee -a "$RESULT"
    done
  done
else
  echo "SERVER_FAILED" | tee -a "$RESULT"; tail -25 "$SRVLOG" | grep -viE "pynvml|Future" | tee -a "$RESULT"
fi
kill $SRV 2>/dev/null; pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null; sleep 2
echo "===== DONE_granite ====="; cat "$RESULT"
