#!/bin/bash
#SBATCH --job-name=p1_7_fh1_check
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:30:00
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
MODE="${1:-plain}"   # plain | pdmux
MODEL="${2:-tiiuae/Falcon-H1-3B-Base}"
PORT=$((32300 + SLURM_JOB_ID % 300))
SRVLOG="$HERE/p1_7_fh1srv_${SLURM_JOB_ID}.log"
EXTRA=""
[ "$MODE" = "pdmux" ] && EXTRA="--enable-pdmux --pdmux-config-path $HERE/pdmux_a100_smoke.yml --chunked-prefill-size -1 --disable-overlap-schedule"

python -m sglang.launch_server \
  --model-path $MODEL --trust-remote-code --dtype bfloat16 \
  --disable-cuda-graph --disable-piecewise-cuda-graph --disable-radix-cache \
  --mem-fraction-static 0.82 --max-running-requests 32 --context-length 8192 \
  $EXTRA --host 127.0.0.1 --port $PORT > "$SRVLOG" 2>&1 &
SRV=$!
ok=0
for i in $(seq 1 150); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health 2>/dev/null||true)" = 200 ] && { ok=1; echo healthy; break; }
  kill -0 $SRV 2>/dev/null || { echo "SERVER DIED"; break; }; sleep 5
done
echo "FALCON-H1 boot_ok=$ok mode=$MODE"
grep -iE "PD-Multiplexing enabled|num hidden layers|Load weight end" "$SRVLOG" | head -3
if [ "$ok" = 1 ]; then
  echo -n "SANITY: "; curl -s http://127.0.0.1:$PORT/generate -H 'Content-Type: application/json' \
    -d '{"text":"The capital of France is","sampling_params":{"max_new_tokens":8,"temperature":0}}' | grep -oE '"text":"[^"]*"'
  # small concurrent load
  P=$(python3 -c "print('the robot walked slowly across the red planet '*80, end='')")
  pids=""; for r in $(seq 1 6); do
    curl -s http://127.0.0.1:$PORT/generate -H 'Content-Type: application/json' \
      --data-binary "$(python3 -c "import json;print(json.dumps({'text':'$P','sampling_params':{'max_new_tokens':24,'temperature':0,'ignore_eos':True}}))")" >/dev/null 2>&1 & pids="$pids $!"; done
  wait $pids
  grep -iE "assert|illegal|device-side|Scheduler hit|Traceback|has no attribute|forward_split" "$SRVLOG" | grep -viE "OpenAIServing" | tail -3 && echo "^^ISSUE" || echo "NO_CRASH"
  echo "FH1_${MODE}_DONE"
else echo "FAILED"; tail -25 "$SRVLOG" | grep -viE "pynvml|Future"; fi
kill $SRV 2>/dev/null; pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null; sleep 2
