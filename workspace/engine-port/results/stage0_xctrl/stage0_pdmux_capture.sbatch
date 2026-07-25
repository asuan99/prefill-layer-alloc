#!/bin/bash
#SBATCH --job-name=s0sweep
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --time=03:00:00
#SBATCH --comment=pytorch
#SBATCH --array=0-2
#SBATCH --output=%x_C%a_%A.out
#SBATCH --error=%x_C%a_%A.err
# ============================================================================
# STAGE 0  operating-point decode-SM sensitivity  (cudagraph-ON, pdmux green-ctx)
#
# ONLY run after stage0_smoke.sbatch PASSES (green-ctx cudagraph capture on 2.9.1).
#
# Replaces the r0c in-forward pin (nemotron_h.py:653), which is EAGER-ONLY: cudagraph
# replay (cuda_graph_runner.py:1161) never re-runs the model forward, so the per-step
# PDMUX_FIXED_DECODE_SM_FILE read is inert at the operating point. Instead we use the
# sanctioned WHOLE-PHASE partition: each pdmux stream group captures a whole-decode
# cudagraph on its green-ctx decode stream (cuda_graph_runner.py:810-817); FixedPolicy
# pins decode to a fixed SM (multiplexing_mixin.py:143-146, controller.py:80-94).
#
# AXES:  ctx in {4096,8192,16384} (SLURM array 0..2);  decode-SM in {16,44,108}.
#
# DECODE-SM MAPPING (per-decode-SM server -- chosen over multi-division+fixed-index):
#   d16  -> pdmux_d16.yml  (single div [92,16,0]),  PDMUX_R2_POLICY=fixed DSM=16
#   d44  -> pdmux_d44.yml  (single div [64,44,0]),  PDMUX_R2_POLICY=fixed DSM=44
#   d108 -> pdmux_d16.yml, R2 policy OFF -> decode-only auto-selects full-108
#           decode stream (idx N-1, multiplexing_mixin.py:745-746).
#   Rationale: FixedPolicy targets exactly ONE decode_sms; a single-division config
#   makes the pin unambiguous (no bs-threshold drift between divisions) and matches
#   the proven R2 configs (results/slo_sched/pdmux_d16.yml). One server per SM is the
#   most robust: no in-process re-pin, each server's green-ctx graph is captured once.
#
# ENGINE CONSTRAINT (engine-porter, code-verified -- NOT a harness choice):
#   The green-ctx sub-108 decode partitions activate ONLY while a prefill is
#   CONCURRENT (the FixedPolicy hook _r2_decide_idx fires under `split_prefill_batch`,
#   multiplexing_mixin.py:844-849; pure decode-only is forced to 108 at :745-746).
#   => d16/d44 are measured WITH a prefill keepalive (the true coupled operating
#      point). d108 is the decode-gets-all reference (no keepalive). This asymmetry
#      is inherent to the substrate; an ISOLATED decode-only knee at d<108 under
#      cudagraph is NOT reachable without an engine-src edit (event-loop stream_idx
#      selection). Reported to coordinator.
#
# pdmux launch constraints (server_args.py:6114-6126): pp=1, chunked-prefill=-1,
# disaggregation=null, overlap DISABLED.
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
bash $ROOT/workspace/engine-port/scripts/bootstrap/sync_engine_tree.sh || {
  echo "SYNC_ENGINE_TREE_FAILED"; exit 3; }

MODEL="${MODEL:-Zyphra/Zamba2-2.7B}"      # hybrid arm; set MODEL=Qwen/Qwen2.5-3B-Instruct for control
CTX_LIST=(4096 8192 16384)
CTX=${CTX_LIST[$SLURM_ARRAY_TASK_ID]}
CTXCAP=$(( CTX + 512 ))
CONC="${CONC:-32}"
OUTTOK="${OUTTOK:-32}"
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

# decode-SM arms: "d16:16 d44:44 d108:0" -> (config, fixed-dsm, keepalive)
declare -A CFG=(   [16]="$HERE/pdmux_d16.yml"  [44]="$HERE/pdmux_d44.yml"  [108]="$HERE/pdmux_d16.yml" )
declare -A KEEPA=( [16]=2 [44]=2 [108]=0 )       # d108 decode-only; others coupled
SM_ARMS=(16 44 108)

# build the ~ctx-token prompt once
REPS=$(( CTX / 8 )); [ "$REPS" -lt 1 ] && REPS=1
PF="$HERE/s0sweep_C${CTX}_prompt.txt"
python3 - "$REPS" "$PF" <<'PY'
import sys
reps, pf = int(sys.argv[1]), sys.argv[2]
open(pf, "w").write("the robot walked slowly across the red planet " * reps)
PY

RESULT="$HERE/s0sweep_C${CTX}_${SLURM_ARRAY_JOB_ID:-local}_result.txt"; : > "$RESULT"
echo "MODEL=$MODEL CTX=$CTX ctxcap=$CTXCAP conc=$CONC out=$OUTTOK torch=$(python -c 'import torch;print(torch.__version__)')" | tee -a "$RESULT"

for D in "${SM_ARMS[@]}"; do
  cfg="${CFG[$D]}"; ka="${KEEPA[$D]}"
  PORT=$((36000 + (${SLURM_ARRAY_JOB_ID:-0} + SLURM_ARRAY_TASK_ID + D) % 400))
  RUNID="s0_C${CTX}_D${D}_${SLURM_ARRAY_JOB_ID:-local}"
  SRVLOG="$HERE/${RUNID}_srv.log"
  export PDMUX_TELEMETRY_PATH="$HERE/${RUNID}_telemetry.jsonl"
  export PDMUX_RUN_ID="$RUNID"; export PDMUX_WORKLOAD_ID="C${CTX}_D${D}"

  if [ "$D" = 108 ]; then
    unset PDMUX_R2_POLICY PDMUX_R2_FIXED_DSM
  else
    export PDMUX_R2_POLICY=fixed; export PDMUX_R2_FIXED_DSM=$D
  fi

  echo "=== ARM decode-SM=$D cfg=$cfg keepalive=$ka policy=${PDMUX_R2_POLICY:-off} port=$PORT ===" | tee -a "$RESULT"
  python -m sglang.launch_server --model-path "$MODEL" --trust-remote-code --dtype bfloat16 \
    --attention-backend triton \
    --enable-pdmux --pdmux-config-path "$cfg" \
    --disable-piecewise-cuda-graph \
    --disable-overlap-schedule --chunked-prefill-size -1 --disable-radix-cache \
    --mem-fraction-static 0.80 --max-running-requests 48 --context-length $CTXCAP \
    --host 127.0.0.1 --port $PORT > "$SRVLOG" 2>&1 &
  SRV=$!
  ok=0
  for i in $(seq 1 220); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health 2>/dev/null||true)" = 200 ] && { ok=1; break; }
    kill -0 $SRV 2>/dev/null || { echo "SERVER_DIED D=$D" | tee -a "$RESULT"; break; }; sleep 5
  done
  echo "boot_ok=$ok D=$D" | tee -a "$RESULT"
  if [ "$ok" = 1 ]; then
    RAW="$HERE/${RUNID}_raw.jsonl"
    python3 "$HERE/stage0_client.py" --url "http://127.0.0.1:$PORT" \
      --prompt-file "$PF" --conc $CONC --out-tokens $OUTTOK \
      --keepalive-prefill $ka --raw-out "$RAW" --tag "C${CTX}_D${D}" 2>>"$RESULT" | tee -a "$RESULT"
    # what decode_sm did the engine actually use?
    if [ -f "$PDMUX_TELEMETRY_PATH" ]; then
      python3 - "$PDMUX_TELEMETRY_PATH" <<'PY' | tee -a "$RESULT"
import json,sys,collections
c=collections.Counter()
for ln in open(sys.argv[1]):
    try:e=json.loads(ln)
    except Exception:continue
    if "target_decode_sms" in e: c[e.get("target_decode_sms")]+=1
print("ACTUAL_target_decode_sms_hist:", dict(c) or "NONE")
PY
    fi
  else
    echo "BOOT_FAILED D=$D" | tee -a "$RESULT"; tail -25 "$SRVLOG" | grep -viE "pynvml|Future" | tee -a "$RESULT"
  fi
  kill $SRV 2>/dev/null; pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null; sleep 3
done
echo "DONE CTX=$CTX"
