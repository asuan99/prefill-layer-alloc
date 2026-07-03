#!/bin/bash
#SBATCH --job-name=p1_4_zb_ctxsens
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
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
SMFILE="$HERE/.zb_decode_sm_${SLURM_JOB_ID}"
export SGLANG_ZAMBA_TIMING=1 PDMUX_FIXED_DECODE_SM_FILE="$SMFILE"
echo full > "$SMFILE"
PORT=31066
SRVLOG="$HERE/p1_4_zbsrv_${SLURM_JOB_ID}.log"
RESULT="$HERE/p1_4_zb_ctxsens_${SLURM_JOB_ID}.txt"; : > "$RESULT"

python -m sglang.launch_server \
  --model-path Zyphra/Zamba2-2.7B --trust-remote-code --dtype bfloat16 \
  --attention-backend torch_native \
  --disable-cuda-graph --disable-piecewise-cuda-graph --disable-radix-cache \
  --mem-fraction-static 0.8 --max-running-requests 16 --context-length 4096 \
  --host 127.0.0.1 --port $PORT > "$SRVLOG" 2>&1 &
SRV=$!
ok=0
for i in $(seq 1 150); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health 2>/dev/null||true)" = 200 ] && { ok=1; echo "healthy"; break; }
  kill -0 $SRV 2>/dev/null || { echo "SERVER DIED"; break; }; sleep 5
done

if [ "$ok" = 1 ]; then
  for CTX in 128 1024 3000; do
    PROMPT=$(python3 -c "print('the robot walked slowly '*$((CTX/4)), end='')")
    for SM in full 44 16; do
      echo "$SM@$CTX" > "$SMFILE"
      echo "--- ctx=$CTX sm=$SM ---"
      pids=""
      for r in $(seq 1 8); do
        curl -s http://127.0.0.1:$PORT/generate -H 'Content-Type: application/json' \
          --data-binary "$(python3 -c "import json,sys; print(json.dumps({'text':'$PROMPT','sampling_params':{'max_new_tokens':90,'temperature':0.7,'ignore_eos':True}}))")" >/dev/null 2>&1 &
        pids="$pids $!"
      done
      wait $pids; sleep 2
      line=$(grep "ZBLT mode=$SM@$CTX " "$SRVLOG" | tail -1)
      echo "RESULT ctx=$CTX $line" | tee -a "$RESULT"
    done
  done
else
  echo "SERVER_FAILED"; tail -30 "$SRVLOG" | tee -a "$RESULT"
fi
kill $SRV 2>/dev/null; sleep 3; kill -9 $SRV 2>/dev/null
echo "===== DONE ====="; cat "$RESULT"
