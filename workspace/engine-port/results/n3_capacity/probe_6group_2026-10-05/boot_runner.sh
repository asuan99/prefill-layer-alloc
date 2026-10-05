#!/bin/bash
set -uo pipefail
export PATH=/opt/conda/bin:$PATH
export PDMUX_ROOT=/opt/pdmux
export SGLANG_ENGINE_DEV=/opt/pdmux/sglang_engine_dev/python
export PDMUX_PROJECT_ROOT=/opt/pdmux/prefill-layer-alloc
ENG="$PDMUX_PROJECT_ROOT/workspace/engine-port"
export HF_HOME=/data/hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
MODEL="nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base"
BACKEND="flashinfer"
CTX=16384; MEM_FRACTION=0.82; MAX_RUNNING=48; SERVER_SEED=1; FIXED_DSM=44
SGPT_RAW="$HF_HOME/raw/ShareGPT_V3_unfiltered_cleaned_split.json"
OUT="/data/runs/n3_probe_6group_2026-10-05"
mkdir -p "$OUT"

wait_health () {   # $1 port $2 pid
  for i in $(seq 1 240); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$1/health 2>/dev/null||true)" = 200 ] && return 0
    kill -0 "$2" 2>/dev/null || return 1
    sleep 5
  done
  return 1
}
port_in_use () { ss -Htan 2>/dev/null | awk '{print $4}' | grep -q ":$1\$"; }
PORT_LO=20000; PORT_SPAN=800
pick_port () {
  local try cand
  for try in $(seq 1 50); do
    cand=$((PORT_LO + (RANDOM + $$ + try) % PORT_SPAN))
    if ! curl -s -o /dev/null --max-time 1 "http://127.0.0.1:$cand/health" 2>/dev/null; then
      if ! port_in_use "$cand"; then echo "$cand"; return 0; fi
    fi
  done
  echo 0
}

run_boot () {
  local NAME="$1" CFG="$2"
  local SRVLOG="$OUT/srv_${NAME}.log"
  local GREEN="$OUT/green_${NAME}.json"
  local TEL="$OUT/tel_${NAME}.jsonl"
  export PDMUX_TELEMETRY_PATH="$TEL"
  export PDMUX_RUN_ID="n3probe_${NAME}"
  export PDMUX_WORKLOAD_ID="n3probe"
  export PDMUX_R2_POLICY=fixed
  export PDMUX_R2_FIXED_DSM=44
  export PDMUX_GREEN_READOUT=1
  export PDMUX_GREEN_READOUT_PATH="$GREEN"
  export PDMUX_PHASE_EVENTS=1
  export TRITON_CACHE_DIR="$OUT/triton_${NAME}"
  mkdir -p "$TRITON_CACHE_DIR"
  : > "$TEL"
  local PORT; PORT=$(pick_port)
  echo "unit=$NAME cfg=$CFG port=$PORT" | tee -a "$OUT/RUN_LOG.txt"

  local T_START=$(date +%s.%N)
  setsid python -m sglang.launch_server --model-path "$MODEL" --trust-remote-code --dtype bfloat16 \
    --attention-backend "$BACKEND" \
    --enable-pdmux --pdmux-config-path "$CFG" \
    --disable-overlap-schedule --chunked-prefill-size -1 \
    --disable-radix-cache --mem-fraction-static "$MEM_FRACTION" \
    --max-running-requests "$MAX_RUNNING" \
    --context-length "$CTX" --random-seed "$SERVER_SEED" \
    --host 127.0.0.1 --port "$PORT" > "$SRVLOG" 2>&1 &
  local PID=$!
  if ! wait_health "$PORT" "$PID"; then
    echo "BOOT_FAILED unit=$NAME" | tee -a "$OUT/RUN_LOG.txt"
    tail -60 "$SRVLOG"
    kill -9 "$PID" 2>/dev/null
    return 1
  fi
  local T_HEALTH=$(date +%s.%N)
  echo "boot_ok=1 unit=$NAME boot_wall_s=$(python3 -c "print(f'{$T_HEALTH-$T_START:.2f}')")" | tee -a "$OUT/RUN_LOG.txt"

  # green read-out check
  python3 "$ENG/results/n3_capacity/n3_plan.py" --check-green "$GREEN" "$CFG" > "$OUT/greencheck_${NAME}.txt" 2>&1
  echo "greencheck_rc=$? unit=$NAME" | tee -a "$OUT/RUN_LOG.txt"

  # warm-up: 8 requests, concurrency 1
  local T_WARM0=$(date +%s.%N)
  python -m sglang.bench_serving --backend sglang --model "$MODEL" \
    --host 127.0.0.1 --port "$PORT" \
    --dataset-name random --dataset-path "$SGPT_RAW" --tokenize-prompt \
    --random-input-len 256 --random-output-len 512 --random-range-ratio 1.0 \
    --num-prompts 8 --max-concurrency 1 --seed 7 \
    --output-details --output-file "$OUT/warmup_${NAME}.jsonl" > "$OUT/warmup_client_${NAME}.log" 2>&1
  local T_WARM1=$(date +%s.%N)
  echo "warmup_wall_s=$(python3 -c "print(f'{$T_WARM1-$T_WARM0:.2f}')") unit=$NAME" | tee -a "$OUT/RUN_LOG.txt"

  # teardown
  local T_TD0=$(date +%s.%N)
  kill "$PID" 2>/dev/null; sleep 5
  pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null; sleep 10
  pkill -9 -f "^sglang::" 2>/dev/null
  local T_TD1=$(date +%s.%N)
  echo "teardown_wall_s=$(python3 -c "print(f'{$T_TD1-$T_TD0:.2f}')") unit=$NAME" | tee -a "$OUT/RUN_LOG.txt"

  echo "== srv log key lines ($NAME) ==" | tee -a "$OUT/RUN_LOG.txt"
  grep -iE "load weight|kv cache|capture|max_total_num_tokens|max_mamba_cache_size|memory pool|error|traceback" "$SRVLOG" \
    | grep -viE "pynvml|Future" | tee -a "$OUT/RUN_LOG.txt"

  echo "== port TIME_WAIT check ($NAME) ==" | tee -a "$OUT/RUN_LOG.txt"
  ss -Htan 2>/dev/null | grep "TIME-WAIT\|TIME_WAIT" | grep ":$PORT " | tee -a "$OUT/RUN_LOG.txt"
  echo "---" | tee -a "$OUT/RUN_LOG.txt"
}

CFG_A="$ENG/results/n3_capacity/configs/pdmux_n3_A_agn.yml"
CFG_H5="$ENG/results/longctx_conflict/probes/pdmux_homog5.yml"

GPU_USED_BASE=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
echo "gpu_used_base_mib=$GPU_USED_BASE" | tee -a "$OUT/RUN_LOG.txt"

echo "===== BOOT A1 (cold, A_agn 6-group) =====" | tee -a "$OUT/RUN_LOG.txt"
run_boot "A1_cold" "$CFG_A"
sleep 10
echo "===== BOOT A2 (warm, A_agn 6-group) =====" | tee -a "$OUT/RUN_LOG.txt"
run_boot "A2_warm" "$CFG_A"
sleep 10
echo "===== BOOT H5 (homog5, 5-group, comparison) =====" | tee -a "$OUT/RUN_LOG.txt"
run_boot "H5" "$CFG_H5"

echo "ALL_BOOTS_DONE" | tee -a "$OUT/RUN_LOG.txt"
