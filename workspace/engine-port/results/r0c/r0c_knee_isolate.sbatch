#!/bin/bash
#SBATCH --job-name=r0c_knee
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
# ============================================================================
# Per-type COORDINATED knee at serving batch: decode-only (no pdmux, no prefill
# -> no contention), pin decode to fixed N SM, measure per-type (attn/mamba) ms
# via SGLANG_ZAMBA_TIMING. ctx~3600, batch~32. Answers: does mamba's coordinated
# knee sit meaningfully below attn's? (gates whether per-type la could beat d24)
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
SMFILE="$OUT/.knee_${SLURM_JOB_ID}"
export SGLANG_ZAMBA_TIMING=1 PDMUX_FIXED_DECODE_SM_FILE="$SMFILE"
echo full > "$SMFILE"
PORT=$((34400 + SLURM_JOB_ID % 200))
SRVLOG="$OUT/knee_srv_${SLURM_JOB_ID}.log"
RESULT="$OUT/knee_result_${SLURM_JOB_ID}.txt"; : > "$RESULT"
CTX=3600; CONC=32; OUTTOK=48
echo "MODEL=$MODEL ctx=$CTX conc=$CONC port=$PORT job=$SLURM_JOB_ID" | tee -a "$RESULT"

python -m sglang.launch_server --model-path "$MODEL" --trust-remote-code --dtype bfloat16 \
  --attention-backend triton --disable-cuda-graph --disable-piecewise-cuda-graph --disable-radix-cache \
  --mem-fraction-static 0.82 --max-running-requests 48 --context-length 4096 \
  --host 127.0.0.1 --port $PORT > "$SRVLOG" 2>&1 &
SRV=$!
ok=0
for i in $(seq 1 170); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health 2>/dev/null||true)" = 200 ] && { ok=1; break; }
  kill -0 $SRV 2>/dev/null || { echo "SERVER DIED" | tee -a "$RESULT"; break; }; sleep 5
done
echo "boot_ok=$ok" | tee -a "$RESULT"
if [ "$ok" = 1 ]; then
  PROMPT=$(python3 -c "print('the robot walked slowly across the red planet '*400, end='')")  # ~3600 tok
  BODY=$(python3 -c "import json,sys; print(json.dumps({'text':sys.argv[1],'sampling_params':{'max_new_tokens':$OUTTOK,'temperature':0.7,'ignore_eos':True}}))" "$PROMPT")
  for SM in full 44 24 16 8; do
    echo "$SM@$CTX" > "$SMFILE"
    pids=""
    for r in $(seq 1 $CONC); do
      curl -s http://127.0.0.1:$PORT/generate -H 'Content-Type: application/json' --data-binary "$BODY" >/dev/null 2>&1 &
      pids="$pids $!"
    done
    wait $pids; sleep 2
    line=$(grep "ZBLT mode=$SM@$CTX " "$SRVLOG" | tail -1)
    echo "RESULT sm=$SM $line" | tee -a "$RESULT"
  done
else echo "BOOT FAILED" | tee -a "$RESULT"; tail -20 "$SRVLOG" | grep -viE "pynvml|Future" | tee -a "$RESULT"; fi
kill $SRV 2>/dev/null; pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null; sleep 2
echo "DONE"
