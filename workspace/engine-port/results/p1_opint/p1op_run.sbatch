#!/bin/bash
#SBATCH --job-name=p1op
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:30:00
#SBATCH --comment=pytorch
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# P1 operating-point (cudagraph-ON) contrast: plain (fused) vs agnostic (pdmux v1).
# One job = one model = one complete comparison unit (Phase 0 capacity scan +
# Phase 1 n=5 paired measurement across rate in {2,3,4,6}). See PREREG.md.
set -uo pipefail
module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0
export PS1="" PATH="$HOME/.local/bin:$PATH"
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate
cd /scratch/ehmoon/whlee/prefill-layer-alloc

MODEL="${1:?usage: p1op_run.sbatch MODEL CTX BACKEND TAG}"
CTX="${2:?ctx}"
BACKEND="${3:?backend}"
TAG="${4:?tag}"

OUT=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/p1_opint
mkdir -p "$OUT"
export HF_HOME=/scratch/ehmoon/whlee/prefill-layer-alloc/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="$OUT/.triton_cache_${SLURM_JOB_ID}"
mkdir -p "$TRITON_CACHE_DIR"

echo "=== env sync ==="
MANIFEST="$OUT/runtime_source_manifest_${TAG}_${SLURM_JOB_ID}.sha256"
workspace/engine-port/scripts/bootstrap/sync_engine_tree.sh "$MANIFEST"
echo "manifest -> $MANIFEST"

NPROMPT_SCAN=60
NPROMPT_MAIN=120
INLEN=2000
OUTLEN=96
RANGE_RATIO=1.0
SCAN_RATES="1 2 3 4 5 6 8 10 12"
MAIN_RATES="2 3 4 6"
NREP=5
PDMUX_CFG="$OUT/pdmux_a100_smoke.yml"

PORT_BASE=$((33000 + SLURM_JOB_ID % 400))

boot_server () {   # $1=policy $2=port $3=srvlog -> echoes pid
  local POL="$1" PORT="$2" SRVLOG="$3"
  local EXTRA=""
  if [ "$POL" = "agnostic" ]; then
    EXTRA="--enable-pdmux --pdmux-config-path $PDMUX_CFG --chunked-prefill-size -1 --disable-overlap-schedule"
  fi
  python -m sglang.launch_server --model-path "$MODEL" --trust-remote-code --dtype bfloat16 \
    --attention-backend "$BACKEND" \
    --disable-radix-cache --mem-fraction-static 0.82 --max-running-requests 48 --context-length "$CTX" \
    $EXTRA --host 127.0.0.1 --port "$PORT" > "$SRVLOG" 2>&1 &
  echo $!
}

wait_ready () {   # $1=pid $2=port -> echoes 1/0
  local PID="$1" PORT="$2" ok=0
  for i in $(seq 1 170); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health 2>/dev/null||true)" = 200 ] && { ok=1; break; }
    kill -0 "$PID" 2>/dev/null || { echo "SERVER DIED (pid $PID)" >&2; break; }
    sleep 5
  done
  echo $ok
}

capture_evidence () {  # $1=srvlog -- prints capture/error evidence lines
  local SRVLOG="$1"
  echo "--- capture evidence ($SRVLOG) ---"
  grep -iE "captur|cuda graph|cudagraph" "$SRVLOG" | grep -viE "Future|pynvml" | head -8
  echo "--- error evidence ($SRVLOG) ---"
  grep -iE "error|traceback|not support|illegal|assert" "$SRVLOG" | grep -viE "Future|pynvml" | head -8
}

correctness_check () {  # $1=port -> prints PASS/FAIL, RESP echoed
  local PORT="$1" RESP
  RESP=$(curl -s http://127.0.0.1:$PORT/generate -H 'Content-Type: application/json' \
    -d '{"text":"The capital of France is","sampling_params":{"max_new_tokens":8,"temperature":0}}')
  echo "SANITY_RAW: $RESP"
  if echo "$RESP" | grep -q "Paris"; then echo "CORRECTNESS=PASS"; else echo "CORRECTNESS=FAIL"; fi
}

kill_server () {  # $1=pid $2=port
  local PID="$1" PORT="$2"
  kill "$PID" 2>/dev/null
  pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null
  sleep 3
}

echo "############# PHASE 0: capacity scan (model=$MODEL tag=$TAG) #############"
for POL in plain agnostic; do
  PORT=$((PORT_BASE + RANDOM % 100))
  SRVLOG="$OUT/p1op_${TAG}_srv_${POL}_capscan_${SLURM_JOB_ID}.log"
  JSONL="$OUT/p1op_${TAG}_capscan_${POL}_${SLURM_JOB_ID}.jsonl"; : > "$JSONL"
  echo "===== capscan policy=$POL port=$PORT ====="
  PID=$(boot_server "$POL" "$PORT" "$SRVLOG")
  OK=$(wait_ready "$PID" "$PORT")
  echo "capscan boot_ok=$OK policy=$POL"
  capture_evidence "$SRVLOG"
  if [ "$OK" = 1 ]; then
    correctness_check "$PORT"
    for R in $SCAN_RATES; do
      echo "----- capscan $POL rate=$R -----"
      python -m sglang.bench_serving --backend sglang --model "$MODEL" --host 127.0.0.1 --port "$PORT" \
        --dataset-name random-ids --random-input-len $INLEN --random-output-len $OUTLEN --random-range-ratio $RANGE_RATIO \
        --num-prompts $NPROMPT_SCAN --request-rate $R --seed 7 --warmup-requests 4 \
        --output-details --output-file "$JSONL" --tag "capscan_${POL}_r${R}" \
        | grep -E "Successful requests|Request throughput|Median TTFT|P99 TTFT|Median TPOT|P99 TPOT"
    done
  else
    echo "CAPSCAN BOOT FAILED policy=$POL"; tail -30 "$SRVLOG" | grep -viE "pynvml|Future"
  fi
  kill_server "$PID" "$PORT"
done
echo "############# PHASE 0 DONE #############"

echo "############# PHASE 1: n=$NREP paired measurement (model=$MODEL tag=$TAG) #############"
for REP in $(seq 1 $NREP); do
  COIN=$((RANDOM % 2))
  if [ "$COIN" = 0 ]; then ORDER="plain agnostic"; else ORDER="agnostic plain"; fi
  echo "===== rep=$REP order='$ORDER' (coin=$COIN) ====="
  for POL in $ORDER; do
    PORT=$((PORT_BASE + RANDOM % 100))
    SRVLOG="$OUT/p1op_${TAG}_srv_${POL}_rep${REP}_${SLURM_JOB_ID}.log"
    JSONL="$OUT/p1op_${TAG}_${POL}_rep${REP}_${SLURM_JOB_ID}.jsonl"; : > "$JSONL"
    echo "----- rep=$REP policy=$POL port=$PORT -----"
    PID=$(boot_server "$POL" "$PORT" "$SRVLOG")
    OK=$(wait_ready "$PID" "$PORT")
    echo "rep=$REP policy=$POL boot_ok=$OK"
    capture_evidence "$SRVLOG"
    if [ "$OK" != 1 ]; then
      echo "REP BOOT FAILED rep=$REP policy=$POL"; tail -30 "$SRVLOG" | grep -viE "pynvml|Future"
      kill_server "$PID" "$PORT"; continue
    fi
    CORR=$(correctness_check "$PORT" | tail -1)
    echo "$CORR"
    if [ "$CORR" != "CORRECTNESS=PASS" ]; then
      echo "REP CORRECTNESS FAILED rep=$REP policy=$POL -- skipping benchmark for this cell"
      kill_server "$PID" "$PORT"; continue
    fi
    for R in $MAIN_RATES; do
      SEED=$((9000 + 100*REP + R))
      echo "----- rep=$REP policy=$POL rate=$R seed=$SEED -----"
      python -m sglang.bench_serving --backend sglang --model "$MODEL" --host 127.0.0.1 --port "$PORT" \
        --dataset-name random-ids --random-input-len $INLEN --random-output-len $OUTLEN --random-range-ratio $RANGE_RATIO \
        --num-prompts $NPROMPT_MAIN --request-rate $R --seed $SEED --warmup-requests 8 \
        --output-details --output-file "$JSONL" --tag "rep${REP}_${POL}_r${R}" \
        | grep -E "Successful requests|Request throughput|Median TTFT|P99 TTFT|Median TPOT|P99 TPOT|Median E2E"
    done
    kill_server "$PID" "$PORT"
  done
done
echo "############# PHASE 1 DONE #############"
echo "DONE_P1OP_${TAG}"
