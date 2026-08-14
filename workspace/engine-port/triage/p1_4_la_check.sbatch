#!/bin/bash
#SBATCH --job-name=p1_4_la_check
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:20:00
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
MODE="${1:-layer_aware}"   # layer_aware | agnostic_v2
LAFILE="$HERE/.la_chk_${SLURM_JOB_ID}"; echo "$MODE" > "$LAFILE"
export PDMUX_FIXED_DECODE_SM_FILE="$LAFILE" PDMUX_LA_RESERVE="M-" PDMUX_LA_FLOOR_SM=16 PDMUX_AGN2_SM=16
PORT=$((31400 + SLURM_JOB_ID % 400)); SRVLOG="$HERE/p1_4_lachk_srv_${SLURM_JOB_ID}.log"

python -m sglang.launch_server \
  --model-path nvidia/Nemotron-H-8B-Base-8K --trust-remote-code --dtype bfloat16 \
  --enable-pdmux --pdmux-config-path "$HERE/pdmux_a100_smoke.yml" --chunked-prefill-size -1 --disable-overlap-schedule \
  --disable-cuda-graph --disable-piecewise-cuda-graph --disable-radix-cache \
  --mem-fraction-static 0.82 --max-running-requests 48 --context-length 8192 \
  --host 127.0.0.1 --port $PORT > "$SRVLOG" 2>&1 &
SRV=$!
ok=0
for i in $(seq 1 150); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health 2>/dev/null||true)" = 200 ] && { ok=1; echo healthy; break; }
  kill -0 $SRV 2>/dev/null || { echo "SERVER DIED"; break; }; sleep 5
done
grep -iE "PD-Multiplexing enabled" "$SRVLOG" | head -1
if [ "$ok" = 1 ]; then
  # concurrent long-context load to stress the layer_aware green-ctx switching under pdmux
  P=$(python3 -c "print('the robot walked slowly across the red planet '*200, end='')")
  pids=""
  for r in $(seq 1 8); do
    curl -s http://127.0.0.1:$PORT/generate -H 'Content-Type: application/json' \
      --data-binary "$(python3 -c "import json;print(json.dumps({'text':'$P','sampling_params':{'max_new_tokens':40,'temperature':0.0,'ignore_eos':True}}))")" >/dev/null 2>&1 &
    pids="$pids $!"
  done
  wait $pids
  echo "SANITY:"; curl -s http://127.0.0.1:$PORT/generate -H 'Content-Type: application/json' \
    -d '{"text":"The capital of France is","sampling_params":{"max_new_tokens":8,"temperature":0}}' | grep -oE '"text":"[^"]*"'
  grep -iE "assert|illegal|device-side|Scheduler hit|Traceback" "$SRVLOG" | grep -viE "OpenAIServing" | tail -3 && echo "^^ ERRORS" || echo "NO_CRASH"
  echo "P1.4 LA_CHECK DONE"
else echo "FAILED"; tail -20 "$SRVLOG"; fi
kill $SRV 2>/dev/null; sleep 2; kill -9 $SRV 2>/dev/null
