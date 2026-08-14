#!/bin/bash
#SBATCH --job-name=s0smoke
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --time=00:40:00
#SBATCH --comment="field=efficientai;appl=pytorch"
#SBATCH --output=%x_%A.out
#SBATCH --error=%x_%A.err
# ============================================================================
# STAGE 0  SMOKE / GPU-CORRECTNESS GATE  (must pass before the full sweep)
#
# THE risk being tested (engine-porter code review, file:line):
#   1. green-context + cudagraph capture on torch 2.9.1. server_args.py:6128 warns
#      "CUDA Green Context may encounter potential issues with CudaGraph on torch
#      2.7.x-2.8.x" and torch>2.7 => degradation warning. Our env is torch 2.9.1.
#      cuda_graph_runner.py:810-817 captures a WHOLE-decode graph per stream group
#      (incl. the green-ctx partitions). If that capture fails (OOM / capture error /
#      forced-eager), Stage 0's operating-point (cudagraph-ON) design is void.
#   2. Correct REPLAY: replay() (cuda_graph_runner.py:1141-1161) indexes the captured
#      graph by get_current_stream_idx(); a decode burst must produce sane tokens.
#
# This smoke does NOT measure the knee. It only asserts: boot + N green-ctx graphs
# captured + one decode burst returns tokens. If capture fails => STOP and report
# the log signature (do not run the sweep).
#
# pdmux launch constraints (server_args.py:6114-6126): pp=1, chunked-prefill=-1,
# disaggregation=null, overlap schedule DISABLED.
# ============================================================================
set -uo pipefail
module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0
export PS1="" PATH="$HOME/.local/bin:$PATH"
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate

ROOT=/scratch/ehmoon/whlee/prefill-layer-alloc
HERE=$ROOT/workspace/engine-port/results/stage0_xctrl
export HF_HOME=$ROOT/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="$ROOT/workspace/engine-port/triage/.triton_cache"

# Ensure engine tree + thread-local role patch installed + manifest (idempotent).
bash $ROOT/workspace/engine-port/scripts/bootstrap/sync_engine_tree.sh || {
  echo "SYNC_ENGINE_TREE_FAILED"; exit 3; }

MODEL="${MODEL:-Zyphra/Zamba2-2.7B}"     # hybrid arm first (local). Qwen2.5-3B drop-in.
CTX="${CTX:-4096}"
CTXCAP=$(( CTX + 512 ))
CONC="${CONC:-16}"
OUTTOK="${OUTTOK:-16}"
DTYPE="${DTYPE:-bfloat16}"
CFG="$HERE/pdmux_smoke.yml"              # captures d16 AND d44 green-ctx graphs
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

PORT=$((35000 + RANDOM % 400))
RUNID="s0smoke_${SLURM_JOB_ID:-local}"
SRVLOG="$HERE/${RUNID}_srv.log"
RESULT="$HERE/${RUNID}_result.txt"; : > "$RESULT"
export PDMUX_TELEMETRY_PATH="$HERE/${RUNID}_telemetry.jsonl"
export PDMUX_RUN_ID="$RUNID"
export PDMUX_WORKLOAD_ID="smoke_C${CTX}"
# Exercise the FixedPolicy pin path + controller_decision telemetry. cfg exposes
# decode_states {16,44}; pin decode to 16 (needs concurrent prefill = keepalive).
export PDMUX_R2_POLICY=fixed
export PDMUX_R2_FIXED_DSM=16

echo "MODEL=$MODEL CTX=$CTX DTYPE=$DTYPE cfg=$CFG port=$PORT torch=$(python -c 'import torch;print(torch.__version__)')" | tee -a "$RESULT"

# --- boot pdmux + cudagraph ON (operating point) ---
python -m sglang.launch_server --model-path "$MODEL" --trust-remote-code --dtype "$DTYPE" \
  --attention-backend triton \
  --enable-pdmux --pdmux-config-path "$CFG" \
  --disable-piecewise-cuda-graph \
  --disable-overlap-schedule --chunked-prefill-size -1 --disable-radix-cache \
  --mem-fraction-static 0.80 --max-running-requests 48 --context-length $CTXCAP \
  --host 127.0.0.1 --port $PORT > "$SRVLOG" 2>&1 &
SRV=$!

ok=0
for i in $(seq 1 220); do
  code=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health 2>/dev/null || true)
  [ "$code" = 200 ] && { ok=1; break; }
  kill -0 $SRV 2>/dev/null || { echo "SERVER_DIED_DURING_BOOT" | tee -a "$RESULT"; break; }
  sleep 5
done
echo "boot_ok=$ok  (boot_ok=1 => all stream-group cudagraphs captured)" | tee -a "$RESULT"

# --- capture-success diagnostics from server log ---
NCAP=$(grep -c "Capturing batches" "$SRVLOG" 2>/dev/null || echo 0)
echo "capture_progress_lines=$NCAP" | tee -a "$RESULT"
echo "--- capture/green-ctx/error signatures (if any) ---" | tee -a "$RESULT"
grep -iE "green|greenctx|capturing batches|cuda graph|cudagraph|capture|illegal|out of memory|oom|CUDA error|RuntimeError" "$SRVLOG" \
  | grep -viE "pynvml|Future" | tail -40 | tee -a "$RESULT"

if [ "$ok" != 1 ]; then
  echo "SMOKE_FAIL: boot/capture failed -> DO NOT run sweep. Full tail:" | tee -a "$RESULT"
  tail -60 "$SRVLOG" | grep -viE "pynvml|Future" | tee -a "$RESULT"
  kill $SRV 2>/dev/null; pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null
  exit 2
fi

# --- one decode burst -> replay correctness (tokens returned + sane ITL) ---
REPS=$(( CTX / 8 )); [ "$REPS" -lt 1 ] && REPS=1
PF="$HERE/${RUNID}_prompt.txt"
python3 - "$REPS" "$PF" <<'PY'
import sys
reps, pf = int(sys.argv[1]), sys.argv[2]
open(pf, "w").write("the robot walked slowly across the red planet " * reps)
PY

RAW="$HERE/${RUNID}_raw.jsonl"
# keepalive-prefill=2 so split_prefill stays active -> exercises the green-ctx
# sub-partition pin path (not just the full-108 decode-only fallback).
python3 "$HERE/stage0_client.py" --url "http://127.0.0.1:$PORT" \
  --prompt-file "$PF" --conc $CONC --out-tokens $OUTTOK \
  --keepalive-prefill 2 --raw-out "$RAW" --tag "smoke_C${CTX}" 2>>"$RESULT" | tee -a "$RESULT"
BURST_RC=${PIPESTATUS[0]}
echo "burst_rc=$BURST_RC  (0 => tokens decoded, replay correctness OK)" | tee -a "$RESULT"

# --- did the pin actually engage? report decode_sm the engine used ---
echo "--- controller_decision decode_sms observed (telemetry) ---" | tee -a "$RESULT"
if [ -f "$PDMUX_TELEMETRY_PATH" ]; then
  python3 - "$PDMUX_TELEMETRY_PATH" <<'PY' | tee -a "$RESULT"
import json, sys, collections
c = collections.Counter()
for ln in open(sys.argv[1]):
    try: e = json.loads(ln)
    except Exception: continue
    if e.get("event") == "controller_decision" or "current_decode_sms" in e:
        c[(e.get("current_decode_sms"), e.get("target_decode_sms"))] += 1
print("decode_sms (current,target) histogram:", dict(c) or "NONE (no controller_decision emitted)")
PY
else
  echo "no telemetry file (r2_policy off in smoke) -- inspect server log 'Adjusting stream groups ... decode sm:' instead" | tee -a "$RESULT"
  grep "Adjusting stream groups" "$SRVLOG" | tail -5 | tee -a "$RESULT"
fi

kill $SRV 2>/dev/null; pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null; sleep 2
if [ "$ok" = 1 ] && [ "$BURST_RC" = 0 ]; then
  echo "SMOKE_PASS: green-ctx cudagraph capture + decode replay OK on torch 2.9.1" | tee -a "$RESULT"
else
  echo "SMOKE_FAIL" | tee -a "$RESULT"; exit 2
fi
echo "DONE"
