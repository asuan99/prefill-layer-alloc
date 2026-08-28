#!/bin/bash
#SBATCH --job-name=kneebe
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --time=01:00:00
#SBATCH --comment="field=efficientai;appl=pytorch"
#SBATCH --array=0-3
#SBATCH --output=%x_L%a_%A.out
#SBATCH --error=%x_L%a_%A.err
# ============================================================================
# Phase 1 (next_steps.md): prefill per-layer-type SM-sensitivity as a function of
# (B, L). Pins prefill to N SM (green-ctx) via SGLANG_ZAMBA_PREFILL_KNEE and times
# per-type (attn/mamba) ms. Sweeps context L (SLURM array, one L per task) x batch B
# (concurrency) x SM. Answers: does the prefill layer-aware lever (Diff B = attn
# sensitivity / mamba sensitivity) open up at long L / large B? ctx3600 single-point
# gave Diff B~1.17x (no lever); this maps the whole (B,L) grid.
#   mode string = "<SM>@L<L>@B<B>"  (code parses split('@')[0] as SM; rest = label,
#   resets the per-mode accumulator). Log tag ZBPT now carries bs=/ntok= (observed batch).
# Run:  sbatch knee2d.sbatch     (4 array tasks, L in {2000,8000,16000,32000})
# Then: python analyze_knee2d.py   (parses knee2d_result_L*.txt -> CSV + Diff B heatmap)
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
# ★BACKEND-EFFECT CELL (2026-08-28).  Same model, same tree, same batch --
# only --attention-backend changes.  Comparing a NEW flashinfer run against the
# 2026-07 canonical triton numbers would confound backend with build drift
# (gate #41), so triton is RE-MEASURED here as its own arm.
#   task 0,1 = L2000 x {triton, flashinfer}
#   task 2,3 = L8000 x {triton, flashinfer}
L_LIST=(2000 2000 8000 8000)
ATTN_LIST=(triton flashinfer triton flashinfer)
L=${L_LIST[$SLURM_ARRAY_TASK_ID]}
ATTN=${ATTN_LIST[$SLURM_ARRAY_TASK_ID]}
B_LIST=(1 4)
SM_LIST=(full 24)
CTXCAP=$((L + 512))                 # server context-length cap
CHUNK=16384                         # chunked-prefill-size (lets small (B,L) co-batch)
TARGET_REQ=16                       # total requests per mode (log gate is n%4 so few forwards still emit)
# --------------------------------------------------------------------------

export SGLANG_ZAMBA_TIMING=1 SGLANG_ZAMBA_PREFILL_KNEE=1
# Zamba2-2.7B derives context_length=4096; L>4096 needs override. This is a TIMING-ONLY
# microbench (per-type attn/mamba SM-sensitivity) so meaningless outputs beyond trained
# context are acceptable -- the kernels still do the real O(L^2)/O(L) work at length L.
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
SMFILE="$OUT/.knee2d_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export PDMUX_FIXED_PREFILL_SM_FILE="$SMFILE"
echo full > "$SMFILE"
PORT=$((34600 + (SLURM_ARRAY_JOB_ID + SLURM_ARRAY_TASK_ID) % 300))
SRVLOG="$OUT/kneebe_srv_L${L}_${ATTN}_${SLURM_ARRAY_JOB_ID}.log"
RESULT="$OUT/kneebe_result_L${L}_${ATTN}_${SLURM_ARRAY_JOB_ID}.txt"; : > "$RESULT"
echo "MODEL=$MODEL ATTN=$ATTN L=$L ctxcap=$CTXCAP chunk=$CHUNK B_LIST=${B_LIST[*]} SM_LIST=${SM_LIST[*]} port=$PORT job=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" | tee -a "$RESULT"

python -m sglang.launch_server --model-path "$MODEL" --trust-remote-code --dtype bfloat16 \
  --attention-backend $ATTN --disable-cuda-graph --disable-piecewise-cuda-graph --disable-radix-cache \
  --chunked-prefill-size $CHUNK --mem-fraction-static 0.82 --max-running-requests 48 \
  --context-length $CTXCAP --host 127.0.0.1 --port $PORT > "$SRVLOG" 2>&1 &
SRV=$!
ok=0
for i in $(seq 1 200); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health 2>/dev/null||true)" = 200 ] && { ok=1; break; }
  kill -0 $SRV 2>/dev/null || { echo "SERVER DIED" | tee -a "$RESULT"; break; }; sleep 5
done
echo "boot_ok=$ok" | tee -a "$RESULT"

if [ "$ok" = 1 ]; then
  # build request JSON via FILES (not CLI args): a long-L prompt as an argv exceeds
  # ARG_MAX ("Argument list too long", which killed L=32000 in job 847711).
  REPS=$((L / 9)); [ "$REPS" -lt 1 ] && REPS=1
  BODYFILE="$OUT/.knee2d_body_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json"
  python3 - "$REPS" "$BODYFILE" <<'PY'
import json, sys
reps = int(sys.argv[1]); bodyfile = sys.argv[2]
text = 'the robot walked slowly across the red planet ' * reps
json.dump({'text': text, 'sampling_params': {'max_new_tokens': 2, 'temperature': 0.7, 'ignore_eos': True}}, open(bodyfile, 'w'))
PY
  for B in "${B_LIST[@]}"; do
    WAVES=$(( (TARGET_REQ + B - 1) / B )); [ "$WAVES" -lt 1 ] && WAVES=1
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
      echo "RESULT attn=$ATTN L=$L B=$B sm=$SM $line" | tee -a "$RESULT"
    done
  done
else
  echo "BOOT FAILED" | tee -a "$RESULT"; tail -25 "$SRVLOG" | grep -viE "pynvml|Future" | tee -a "$RESULT"
fi
kill $SRV 2>/dev/null; pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null; sleep 2
echo "DONE attn=$ATTN L=$L"
