#!/bin/bash
#SBATCH --job-name=cg_probe1
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:40:00
#SBATCH --comment="field=efficientai;appl=pytorch"
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# Probe 1: PLAIN (no pdmux) Zamba2, cudagraph OFF vs ON A/B.
#   Q1: does cuda-graph CAPTURE succeed on py3.14/torch2.9 + hybrid(mamba)+triton?
#   Q2: does the decode-bound TPOT wall (~>60ms SLO) drop with cudagraph?
set -uo pipefail
module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0
export PS1="" PATH="$HOME/.local/bin:$PATH"
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate
TRIAGE=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/triage
OUT=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/cudagraph_probe
export HF_HOME=/scratch/ehmoon/whlee/prefill-layer-alloc/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="$TRIAGE/.triton_cache"
MODEL=Zyphra/Zamba2-2.7B; CTX=4096; INLEN=2000; OUTLEN=96; NPROMPT=60; RATES="2 4"

run_variant () {  # $1=label  $2=extra server flags (cudagraph on/off)
  local LAB="$1"; local CGFLAGS="$2"
  local PORT=$((36000 + SLURM_JOB_ID % 120 + RANDOM % 40))
  local SRVLOG="$OUT/srv_${LAB}_${SLURM_JOB_ID}.log"
  echo "########## VARIANT=$LAB flags='$CGFLAGS' port=$PORT ##########"
  python -m sglang.launch_server --model-path $MODEL --trust-remote-code --dtype bfloat16 \
    --attention-backend triton $CGFLAGS --disable-radix-cache \
    --mem-fraction-static 0.82 --max-running-requests 48 --context-length $CTX \
    --disable-overlap-schedule \
    --host 127.0.0.1 --port $PORT > "$SRVLOG" 2>&1 &
  local SRV=$!; local ok=0
  for i in $(seq 1 180); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health 2>/dev/null||true)" = 200 ] && { ok=1; break; }
    kill -0 $SRV 2>/dev/null || { echo "$LAB SERVER DIED"; break; }; sleep 5
  done
  echo "$LAB boot_ok=$ok"
  # did cuda graph capture happen / any capture error?
  echo "--- $LAB capture evidence ---"
  grep -iE "captur|cuda graph|cudagraph|graph runner" "$SRVLOG" | grep -viE "disable|Future|pynvml" | head -6
  grep -iE "error|traceback|not support|fail|assert" "$SRVLOG" | grep -viE "Future|pynvml" | head -4
  if [ "$ok" = 1 ]; then
    curl -s http://127.0.0.1:$PORT/generate -H 'Content-Type: application/json' \
      -d '{"text":"The capital of France is","sampling_params":{"max_new_tokens":6,"temperature":0}}' \
      | grep -oE '"text":"[^"]*"' | head -1 | sed "s/^/$LAB SANITY /"
    python -m sglang.bench_serving --backend sglang --model $MODEL --host 127.0.0.1 --port $PORT \
      --dataset-name random-ids --random-input-len $INLEN --random-output-len $OUTLEN --random-range-ratio 1.0 \
      --num-prompts 8 --request-rate 2 >/dev/null 2>&1 || true
    for R in $RATES; do
      echo "===== $LAB in${INLEN}o${OUTLEN} rate=$R ====="
      python -m sglang.bench_serving --backend sglang --model $MODEL --host 127.0.0.1 --port $PORT \
        --dataset-name random-ids --random-input-len $INLEN --random-output-len $OUTLEN --random-range-ratio 1.0 \
        --num-prompts $NPROMPT --request-rate $R 2>/dev/null \
        | grep -E "Successful requests|Request throughput|Median TTFT|Median TPOT|P99 TPOT|Median ITL"
    done
  else echo "$LAB BOOT FAILED"; tail -30 "$SRVLOG" | grep -viE "pynvml|Future"; fi
  kill $SRV 2>/dev/null; pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null; sleep 3
}

run_variant "cgOFF" "--disable-cuda-graph --disable-piecewise-cuda-graph"
run_variant "cgON"  ""
echo "DONE_PROBE1"
