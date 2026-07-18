#!/bin/bash
#SBATCH --job-name=deckneectx
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --time=02:00:00
#SBATCH --comment=pytorch
#SBATCH --array=0-3
#SBATCH --output=%x_C%a_%A.out
#SBATCH --error=%x_C%a_%A.err
# ============================================================================
# DECODE knee vs CONTEXT LENGTH  (user question, 2026-07-18)
#   Does the optimal decode SM shift with ctx because the attn FRACTION of a decode
#   step shifts? decode-attn scans the KV cache (~ctx, SM-hungry); decode-mamba is a
#   fixed recurrent update (~const, SM-free). So at short ctx a decode step is
#   mamba-dominated (SM-free -> few SM optimal); at long ctx it is attn-dominated
#   (SM-hungry -> many SM optimal). r0c measured only ctx=3600; this sweeps ctx.
#
#   ctx = 256 1024 4096 16384   (SLURM array, one ctx per task)
#   SM  = full 44 24 16 8        (decode pinned to N SM via green-ctx)
#   Per-type ms via ZBLT (SGLANG_ZAMBA_TIMING, decode phase). decode-only: no prefill
#   contention, so the knee is clean. Optimal decode SM = argmin (9*attn + 54*mamba).
#
# This is WHOLE-DECODE SM sizing keyed on ctx (an offline predictor input), NOT a
# per-layer-type split within the step -- that stays dead by (D) granularity.
#
# Run:  sbatch decode_knee_vs_ctx.sbatch     Then: python analyze_decode_knee_vs_ctx.py
# ============================================================================
set -uo pipefail
module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0
export PS1="" PATH="$HOME/.local/bin:$PATH"
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate
HERE=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/triage
OUT=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/r0c
export HF_HOME=/scratch/ehmoon/whlee/prefill-layer-alloc/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="$HERE/.triton_cache"
MODEL=Zyphra/Zamba2-2.7B

CTX_LIST=(256 1024 4096 16384)
CTX=${CTX_LIST[$SLURM_ARRAY_TASK_ID]}
SM_LIST=(full 44 24 16 8)
CONC=32; OUTTOK=32
CTXCAP=$(( CTX + 512 ))

# decode timing (ZBLT). Zamba2 trained ctx=4096; ctx>4096 needs the override. Timing-only:
# KV-scan kernel still does length-ctx work; generated tokens past trained ctx are meaningless.
export SGLANG_ZAMBA_TIMING=1
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
SMFILE="$OUT/.deckneectx_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export PDMUX_FIXED_DECODE_SM_FILE="$SMFILE"
echo full > "$SMFILE"
PORT=$((34000 + (SLURM_ARRAY_JOB_ID + SLURM_ARRAY_TASK_ID) % 300))
SRVLOG="$OUT/deckneectx_srv_C${CTX}_${SLURM_ARRAY_JOB_ID}.log"
RESULT="$OUT/deckneectx_result_C${CTX}_${SLURM_ARRAY_JOB_ID}.txt"; : > "$RESULT"
echo "MODEL=$MODEL CTX=$CTX ctxcap=$CTXCAP conc=$CONC port=$PORT job=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" | tee -a "$RESULT"

python -m sglang.launch_server --model-path "$MODEL" --trust-remote-code --dtype bfloat16 \
  --attention-backend triton --disable-cuda-graph --disable-piecewise-cuda-graph --disable-radix-cache \
  --mem-fraction-static 0.80 --max-running-requests 48 --context-length $CTXCAP \
  --host 127.0.0.1 --port $PORT > "$SRVLOG" 2>&1 &
SRV=$!
ok=0
for i in $(seq 1 220); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health 2>/dev/null||true)" = 200 ] && { ok=1; break; }
  kill -0 $SRV 2>/dev/null || { echo "SERVER DIED" | tee -a "$RESULT"; break; }; sleep 5
done
echo "boot_ok=$ok" | tee -a "$RESULT"
if [ "$ok" = 1 ]; then
  # prompt ~= CTX tokens (~8 tok / phrase-rep); built via FILE to dodge ARG_MAX at long ctx
  REPS=$(( CTX / 8 )); [ "$REPS" -lt 1 ] && REPS=1
  BODYFILE="$OUT/.deckneectx_body_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json"
  python3 - "$REPS" "$OUTTOK" "$BODYFILE" <<'PY'
import json, sys
reps, out, bf = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
text = 'the robot walked slowly across the red planet ' * reps
json.dump({'text': text, 'sampling_params': {'max_new_tokens': out, 'temperature': 0.7, 'ignore_eos': True}}, open(bf, 'w'))
PY
  for SM in "${SM_LIST[@]}"; do
    echo "$SM@$CTX" > "$SMFILE"
    pids=""
    for r in $(seq 1 $CONC); do
      curl -s http://127.0.0.1:$PORT/generate -H 'Content-Type: application/json' --data-binary @"$BODYFILE" >/dev/null 2>&1 &
      pids="$pids $!"
    done
    wait $pids; sleep 2
    line=$(grep "ZBLT mode=$SM@$CTX " "$SRVLOG" | tail -1)
    echo "RESULT CTX=$CTX sm=$SM $line" | tee -a "$RESULT"
  done
else echo "BOOT FAILED" | tee -a "$RESULT"; tail -20 "$SRVLOG" | grep -viE "pynvml|Future" | tee -a "$RESULT"; fi
kill $SRV 2>/dev/null; pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null; sleep 2
echo "DONE CTX=$CTX"
