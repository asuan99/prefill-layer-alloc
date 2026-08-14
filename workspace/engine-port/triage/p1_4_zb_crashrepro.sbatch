#!/bin/bash
#SBATCH --job-name=p1_4_zb_crash
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:25:00
#SBATCH --comment="field=efficientai;appl=pytorch"
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
export CUDA_LAUNCH_BLOCKING=1   # synchronous -> real kernel in the traceback
SMFILE="$HERE/.zbc_sm_${SLURM_JOB_ID}"
export SGLANG_ZAMBA_TIMING=1 PDMUX_FIXED_DECODE_SM_FILE="$SMFILE"
echo "44@2048" > "$SMFILE"        # the crashing case: reduced SM + long ctx
PORT=31077
SRVLOG="$HERE/p1_4_zbc_srv_${SLURM_JOB_ID}.log"

python -m sglang.launch_server \
  --model-path Zyphra/Zamba2-2.7B --trust-remote-code --dtype bfloat16 \
  --attention-backend triton \
  --disable-cuda-graph --disable-piecewise-cuda-graph --disable-radix-cache \
  --mem-fraction-static 0.8 --max-running-requests 8 --context-length 4096 \
  --host 127.0.0.1 --port $PORT > "$SRVLOG" 2>&1 &
SRV=$!
ok=0
for i in $(seq 1 150); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health 2>/dev/null||true)" = 200 ] && { ok=1; echo healthy; break; }
  kill -0 $SRV 2>/dev/null || { echo "SERVER DIED early"; break; }; sleep 5
done
if [ "$ok" = 1 ]; then
  PROMPT=$(python3 -c "print('the robot walked slowly '*520, end='')")   # ~2000 tokens
  echo "sending 8 concurrent long-ctx requests (sm=44)..."
  PAYLOAD="$(python3 -c "import json; print(json.dumps({'text':'$PROMPT','sampling_params':{'max_new_tokens':40,'temperature':0.0,'ignore_eos':True}}))")"
  pids=""
  for r in $(seq 1 8); do
    curl -s http://127.0.0.1:$PORT/generate -H 'Content-Type: application/json' --data-binary "$PAYLOAD" >/dev/null 2>&1 &
    pids="$pids $!"
  done
  wait $pids; sleep 3
fi
echo "===== SERVER LOG (traceback) ====="
grep -nE "Traceback|Error|assert|illegal|greenctx|triton|mamba|conv1d|ssd|File \"|line [0-9]" "$SRVLOG" 2>/dev/null | grep -viE "pynvml|OpenAIServing|torchaudio|cutlass|INFO" | tail -40
kill $SRV 2>/dev/null; sleep 2; kill -9 $SRV 2>/dev/null
echo DONE
