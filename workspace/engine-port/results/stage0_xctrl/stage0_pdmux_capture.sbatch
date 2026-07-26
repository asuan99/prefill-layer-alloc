#!/bin/bash
#SBATCH --job-name=s0sweep
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --time=10:00:00
#SBATCH --comment=pytorch
#SBATCH --array=0-2
#SBATCH --output=%x_A%a_%A.out
#SBATCH --error=%x_A%a_%A.err
# ============================================================================
# STAGE 0 (L-2) + TRANSFORMER CONTROL -- operating-point decode-SM sensitivity
# Design: reports/stage0_transformer_control_design_2026-07-25.md  (3-arm coupled(A))
#
# ONLY run after stage0_smoke.sbatch PASSES (green-ctx cudagraph capture on 2.9.1).
# The smoke ALREADY confirmed the pin path empirically: Mamba2, keepalive=2,
# PDMUX_R2_POLICY=fixed DSM=16  ->  telemetry target_decode_sms hist {16:90}.
#
# QOS(aanv8) REBUDGET (2026-07-25): partition amd_a100nv_8 QOS aanv8 caps
# MaxSubmitPU=4 (pending+running) and MaxJobsPU=2 (concurrent). The original
# design used 9 array tasks (arm=idx/3, ctx=idx%3) which QOSMaxSubmitJobPerUserLimit
# rejected outright. Restructured to 1 SLURM ARRAY TASK PER ARM (array=0-2, 3
# jobs <= MaxSubmitPU=4); the CTX sweep that used to be array-indexed is now an
# in-job bash loop, nested with the pre-existing SM_ARMS in-job loop. Each of
# the 3 jobs now runs 1 arm x 3 ctx x 4 SM = 12 server boots sequentially.
# All measurement logic (configs, keepalive, PIN_CHECK, REPS) is byte-identical
# to the pre-rebudget design; only the arm/ctx dispatch mechanism moved from
# SLURM array indexing to bash loops.
#
# 3 MODEL ARMS = 3 SLURM array tasks (arm=idx), ctx swept in-job:
#   (M) pure Mamba2-2.7B  = SSM-only NEGATIVE control (predict flat vs SM)
#   (H) hybrid Zamba2-2.7B = target (mamba + shared-attn)
#   (T) pure Qwen2.5-3B   = all-attn POSITIVE control (predict SM-sensitive)
#   ALL bf16 (Mamba2 bf16 smoke PASS -- dtype matched across arms).
# CTX in {4096,8192,16384} (in-job loop, was array-indexed before rebudget).
# decode-SM (x-axis of the curve): 16, 44, 92 = ACTIVE green-ctx split (coupled),
#   + 108 = no-split decode-gets-all REFERENCE (labeled, not mixed into curve).
#
# WHY cudagraph-ON + pdmux green-ctx capture (NOT the r0c in-forward pin):
#   r0c's PDMUX_FIXED_DECODE_SM_FILE is read INSIDE forward every step and is INERT
#   under cudagraph replay (cuda_graph_runner.py:1161 never re-runs forward). The
#   sanctioned WHOLE-PHASE partition captures a whole-decode cudagraph per stream
#   group on its green-ctx decode stream (cuda_graph_runner.py:810-817); FixedPolicy
#   pins the decode stream to a fixed SM. This IS the real serving operating point.
#
# COUPLED(A) PIN ACTIVATION (engine constraint, code + telemetry verified):
#   The green-ctx sub-108 decode partition activates ONLY while a prefill is
#   CONCURRENT: the _r2_decide_idx hook fires under `not running_batch.is_empty()
#   and self.split_prefill_batch` (multiplexing_mixin.py:844-849); a pure decode-only
#   phase never enters that block. => d16/d44/d92 are measured WITH a prefill
#   keepalive (stage0_client.py --keepalive-prefill) = the true coupled operating
#   point. d108 is the decode-gets-all reference (R2 OFF, keepalive OFF): with R2 off
#   the loop leaves decode on the full-108 stream (idx N-1) via adjust_stream_groups
#   line 745-746. Reported to coordinator: an ISOLATED sub-108 decode-only knee under
#   cudagraph is NOT reachable without an engine-src edit; the coupled point is the
#   design's chosen (and defensible) measurement.
#
# DECODE-SM => CONFIG mapping (single-division per server = r0c robustness convention;
#   d92 forced to 2 divisions by the decode_states guard, see pdmux_d92.yml header):
#   d16  -> pdmux_d16.yml  ([92,16,0]),          PDMUX_R2_POLICY=fixed DSM=16 , keepalive
#   d44  -> pdmux_d44.yml  ([64,44,0]),          PDMUX_R2_POLICY=fixed DSM=44 , keepalive
#   d92  -> pdmux_d92.yml  ([16,92,0]+[64,44,0]),PDMUX_R2_POLICY=fixed DSM=92 , keepalive
#   d108 -> pdmux_d16.yml , R2 policy OFF -> decode-only auto-selects full-108 stream.
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

# --- arm from array index; ctx swept in-job (QOS rebudget, see header) --------
ARM_LABELS=(M H T)
ARM_MODELS=(
  "$ROOT/hf_cache/mamba2-2.7b-sglang"   # M  pure-SSM negative control
  "Zyphra/Zamba2-2.7B"                   # H  hybrid target
  "Qwen/Qwen2.5-3B"                      # T  pure-attn positive control
)
CTX_LIST=(4096 8192 16384)              # optional short-ctx null: set CTX_LIST=(256 1024) via CTX_OVERRIDE_LIST
AIDX=$SLURM_ARRAY_TASK_ID
ARM="${ARM_LABELS[$AIDX]}"
MODEL="${ARM_MODELS[$AIDX]}"
if [ -n "${CTX_OVERRIDE_LIST:-}" ]; then
  read -r -a CTX_LIST <<< "$CTX_OVERRIDE_LIST"
fi
CONC="${CONC:-32}"
OUTTOK="${OUTTOK:-32}"
DTYPE="${DTYPE:-bfloat16}"            # ALL arms bf16 (dtype matched; Mamba2 bf16 verified)
REPS="${REPS:-3}"                    # client reps per (arm,ctx,SM) for ITL p95 stability
KEEPA_N="${KEEPA_N:-2}"              # keepalive prefill workers for coupled arms
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

# decode-SM arms
SM_ARMS=(16 44 92 108)
declare -A CFG=(   [16]="$HERE/pdmux_d16.yml" [44]="$HERE/pdmux_d44.yml" [92]="$HERE/pdmux_d92.yml" [108]="$HERE/pdmux_d16.yml" )
declare -A KEEPA=( [16]=$KEEPA_N [44]=$KEEPA_N [92]=$KEEPA_N [108]=0 )   # d108 decode-only reference

# CTX sweep (QOS rebudget: this loop replaces the old array-indexed CTX dispatch;
# per-config semantics below are unchanged from the pre-rebudget design).
for CTX in "${CTX_LIST[@]}"; do
CTXCAP=$(( CTX + 512 ))

# build the ~ctx-token prompt once per (arm,ctx)
PREPS=$(( CTX / 8 )); [ "$PREPS" -lt 1 ] && PREPS=1
PF="$HERE/s0sweep_${ARM}_C${CTX}_prompt.txt"
python3 - "$PREPS" "$PF" <<'PY'
import sys
reps, pf = int(sys.argv[1]), sys.argv[2]
open(pf, "w").write("the robot walked slowly across the red planet " * reps)
PY

RESULT="$HERE/s0sweep_${ARM}_C${CTX}_${SLURM_ARRAY_JOB_ID:-local}_result.txt"; : > "$RESULT"
echo "ARM=$ARM MODEL=$MODEL CTX=$CTX ctxcap=$CTXCAP conc=$CONC out=$OUTTOK dtype=$DTYPE reps=$REPS keepN=$KEEPA_N torch=$(python -c 'import torch;print(torch.__version__)')" | tee -a "$RESULT"

for D in "${SM_ARMS[@]}"; do
  cfg="${CFG[$D]}"; ka="${KEEPA[$D]}"
  PORT=$((36000 + (${SLURM_ARRAY_JOB_ID:-0} + SLURM_ARRAY_TASK_ID * 7 + CTX + D) % 500))
  RUNID="s0_${ARM}_C${CTX}_D${D}_${SLURM_ARRAY_JOB_ID:-local}"
  SRVLOG="$HERE/${RUNID}_srv.log"
  export PDMUX_TELEMETRY_PATH="$HERE/${RUNID}_telemetry.jsonl"; : > "$PDMUX_TELEMETRY_PATH"
  export PDMUX_RUN_ID="$RUNID"; export PDMUX_WORKLOAD_ID="${ARM}_C${CTX}_D${D}"

  if [ "$D" = 108 ]; then
    unset PDMUX_R2_POLICY PDMUX_R2_FIXED_DSM
  else
    export PDMUX_R2_POLICY=fixed; export PDMUX_R2_FIXED_DSM=$D
  fi

  echo "=== ARM=$ARM decode-SM=$D cfg=$(basename $cfg) keepalive=$ka policy=${PDMUX_R2_POLICY:-off} port=$PORT ===" | tee -a "$RESULT"
  python -m sglang.launch_server --model-path "$MODEL" --trust-remote-code --dtype "$DTYPE" \
    --attention-backend triton \
    --enable-pdmux --pdmux-config-path "$cfg" \
    --disable-piecewise-cuda-graph \
    --disable-overlap-schedule --chunked-prefill-size -1 --disable-radix-cache \
    --mem-fraction-static 0.80 --max-running-requests 48 --context-length $CTXCAP \
    --host 127.0.0.1 --port $PORT > "$SRVLOG" 2>&1 &
  SRV=$!
  ok=0
  for i in $(seq 1 240); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health 2>/dev/null||true)" = 200 ] && { ok=1; break; }
    kill -0 $SRV 2>/dev/null || { echo "SERVER_DIED D=$D" | tee -a "$RESULT"; break; }; sleep 5
  done
  echo "boot_ok=$ok D=$D" | tee -a "$RESULT"

  if [ "$ok" = 1 ]; then
    for r in $(seq 1 $REPS); do
      RAW="$HERE/${RUNID}_rep${r}_raw.jsonl"
      python3 "$HERE/stage0_client.py" --url "http://127.0.0.1:$PORT" \
        --prompt-file "$PF" --conc $CONC --out-tokens $OUTTOK \
        --keepalive-prefill $ka --raw-out "$RAW" --tag "${ARM}_C${CTX}_D${D}_r${r}" 2>>"$RESULT" | tee -a "$RESULT"
    done
    # PIN VERIFICATION: which decode_sm did the engine ACTUALLY pin? (must equal D for 16/44/92)
    if [ -f "$PDMUX_TELEMETRY_PATH" ]; then
      python3 - "$PDMUX_TELEMETRY_PATH" "$D" <<'PY' | tee -a "$RESULT"
import json,sys,collections
path,expect=sys.argv[1],sys.argv[2]
c=collections.Counter()
for ln in open(path):
    try:e=json.loads(ln)
    except Exception:continue
    if "target_decode_sms" in e: c[e.get("target_decode_sms")]+=1
hist=dict(c)
print("ACTUAL_target_decode_sms_hist:", hist or "NONE")
if expect!="108":
    keys=set(hist); exp=int(expect)
    ok = (keys=={exp}) or (keys and max(hist,key=hist.get)==exp)
    print(f"PIN_CHECK D={expect}: {'PASS' if ok else 'FAIL -> arm measured at wrong SM, INVALID'}")
else:
    print("PIN_CHECK D=108: reference (R2 off) -> no target_decode_sms expected; decode uses full-108 stream")
PY
    fi
  else
    echo "BOOT_FAILED D=$D" | tee -a "$RESULT"; tail -30 "$SRVLOG" | grep -viE "pynvml|Future" | tee -a "$RESULT"
  fi
  kill $SRV 2>/dev/null; pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null; sleep 3
done
echo "DONE ARM=$ARM CTX=$CTX"
done  # CTX sweep
