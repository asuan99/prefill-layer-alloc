#!/bin/bash
#SBATCH --job-name=knee2dw
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --time=05:00:00
#SBATCH --comment=pytorch
#SBATCH --array=0-7
#SBATCH --output=%x_L%a_%A.out
#SBATCH --error=%x_L%a_%A.err
# ============================================================================
# WIDE (B,L) prefill per-layer-type SM-sensitivity sweep  (user request, 2026-07-17)
#   L = 256 512 1024 2048 4096 8192 16384 32768   (SLURM array, one L per task)
#   B = 1 2 4 8 16
#   SM = full 44 24 16 8   (prefill pinned to N SM via green-ctx)
#
# WHY this supersedes knee2d.sbatch (which swept L 2k-32k x B 1-48):
#   (1) COVERAGE: the old grid started at L=2000, but 98% of the real ShareGPT workload
#       is BELOW that (mean 352, p50 204, p95 1042 tok). Diff B was never measured in the
#       regime that actually serves -- and at the old grid's low edge (L=2000) Diff B was
#       ~1.35, NOT ~1.0, i.e. the lever OPENS as L shrinks. This sweep goes down to 256.
#   (2) THE B AXIS WAS FAKE: chunked-prefill-size=16384 capped tokens/forward, so a
#       requested B=48 was actually observed as bs=1..10 (L=8000 B=4 -> bs=1!). The old
#       "flat in B" result was measured on a grid where B barely varied. Fix: size CHUNK
#       to B_max*L so the requested batch can actually co-batch in ONE forward.
#
# HARD PHYSICAL LIMIT (do not paper over): one forward must hold B*L tokens, so the full
#   8x5 grid is NOT realizable -- 16 x 32768 = 524k tokens does not fit on an A100-80GB.
#   CHUNK is capped at CHUNK_MAX; cells needing more will still collapse (bs_obs < B).
#   That is why every row is keyed on the MEASURED bs/ntok, never the requested B.
#
# Timing: CUDA events per layer, accumulated over n forwards, and the log already divides
#   by n (acc/n/9, acc/n/54) -> per-forward per-layer ms. Short L is near the noise floor
#   (L=256 B=1 attn ~ 20us/layer), so TARGET_REQ scales up as L shrinks to average it down.
#
# Run:  sbatch knee2d_wide.sbatch
# Then: python analyze_knee2d_wide.py   (-> knee2d_wide_table.csv + plots)
# ============================================================================
set -uo pipefail
module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0
export PS1="" PATH="$HOME/.local/bin:$PATH"
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate
HERE=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/triage
OUT=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/prefill_knee
export HF_HOME=/scratch/ehmoon/whlee/prefill-layer-alloc/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="$HERE/.triton_cache"
MODEL=Zyphra/Zamba2-2.7B

# ---- sweep grid ----------------------------------------------------------
L_LIST=(256 512 1024 2048 4096 8192 16384 32768)
L=${L_LIST[$(( SLURM_ARRAY_TASK_ID + ${OFF:-0} ))]}   # OFF lets a 2nd 0-3 array cover indices 4-7 (QOS caps queued jobs at 4)
B_LIST=(1 2 4 8 16)
SM_LIST=(full 44 24 16 8)
BMAX=16
CHUNK_MAX=65536                      # memory ceiling for one prefill forward on A100-80GB
CHUNK=$(( BMAX * L )); [ "$CHUNK" -gt "$CHUNK_MAX" ] && CHUNK=$CHUNK_MAX
[ "$CHUNK" -lt 8192 ] && CHUNK=8192  # floor: let small-L cells co-batch freely
CTXCAP=$(( L + 512 ))
# short L -> tiny per-layer times -> need many forwards to average the noise down
TARGET_REQ=$(( 262144 / L )); [ "$TARGET_REQ" -lt 16 ] && TARGET_REQ=16
[ "$TARGET_REQ" -gt 256 ] && TARGET_REQ=256
# --------------------------------------------------------------------------

export SGLANG_ZAMBA_TIMING=1 SGLANG_ZAMBA_PREFILL_KNEE=1
# Zamba2-2.7B derives context_length=4096; L>4096 needs the override. TIMING-ONLY microbench
# (per-type attn/mamba SM-sensitivity): outputs past the trained context are meaningless but
# the kernels still do the real work at length L, which is all we measure.
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
SMFILE="$OUT/.knee2dw_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export PDMUX_FIXED_PREFILL_SM_FILE="$SMFILE"
echo full > "$SMFILE"
PORT=$((35200 + (SLURM_ARRAY_JOB_ID + SLURM_ARRAY_TASK_ID) % 300))
SRVLOG="$OUT/knee2dw_srv_L${L}_${SLURM_ARRAY_JOB_ID}.log"
RESULT="$OUT/knee2dw_result_L${L}_${SLURM_ARRAY_JOB_ID}.txt"; : > "$RESULT"
echo "MODEL=$MODEL L=$L ctxcap=$CTXCAP chunk=$CHUNK target_req=$TARGET_REQ B_LIST=${B_LIST[*]} SM_LIST=${SM_LIST[*]} port=$PORT job=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" | tee -a "$RESULT"

python -m sglang.launch_server --model-path "$MODEL" --trust-remote-code --dtype bfloat16 \
  --attention-backend triton --disable-cuda-graph --disable-piecewise-cuda-graph --disable-radix-cache \
  --chunked-prefill-size $CHUNK --max-prefill-tokens $CHUNK \
  --mem-fraction-static 0.75 --max-running-requests 48 \
  --context-length $CTXCAP --host 127.0.0.1 --port $PORT > "$SRVLOG" 2>&1 &
SRV=$!
ok=0
for i in $(seq 1 240); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health 2>/dev/null||true)" = 200 ] && { ok=1; break; }
  kill -0 $SRV 2>/dev/null || { echo "SERVER DIED" | tee -a "$RESULT"; break; }; sleep 5
done
echo "boot_ok=$ok" | tee -a "$RESULT"

if [ "$ok" = 1 ]; then
  # Build the prompt via a FILE, never argv: a 32k-token prompt as a CLI arg exceeds ARG_MAX
  # ("Argument list too long" killed L=32000 in job 847711).
  # ~8 tokens per repetition of the phrase -> REPS=L/8 lands ntok close to nominal L.
  REPS=$(( L / 8 )); [ "$REPS" -lt 1 ] && REPS=1
  BODYFILE="$OUT/.knee2dw_body_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json"
  python3 - "$REPS" "$BODYFILE" <<'PY'
import json, sys
reps = int(sys.argv[1]); bodyfile = sys.argv[2]
text = 'the robot walked slowly across the red planet ' * reps
json.dump({'text': text, 'sampling_params': {'max_new_tokens': 2, 'temperature': 0.7, 'ignore_eos': True}}, open(bodyfile, 'w'))
PY
  for B in "${B_LIST[@]}"; do
    WAVES=$(( (TARGET_REQ + B - 1) / B )); [ "$WAVES" -lt 2 ] && WAVES=2
    for SM in "${SM_LIST[@]}"; do
      MODE="$SM@L$L@B$B"
      echo "$MODE" > "$SMFILE"
      for w in $(seq 1 $WAVES); do
        pids=""
        for r in $(seq 1 $B); do
          curl -s http://127.0.0.1:$PORT/generate -H 'Content-Type: application/json' --data-binary @"$BODYFILE" >/dev/null 2>&1 &
          pids="$pids $!"
        done
        wait $pids
      done
      sleep 2
      line=$(grep "ZBPT mode=$MODE " "$SRVLOG" | tail -1)
      echo "RESULT L=$L B=$B sm=$SM $line" | tee -a "$RESULT"
    done
  done
else
  echo "BOOT FAILED" | tee -a "$RESULT"; tail -25 "$SRVLOG" | grep -viE "pynvml|Future" | tee -a "$RESULT"
fi
kill $SRV 2>/dev/null; pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null; sleep 2
echo "DONE L=$L"
