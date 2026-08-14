#!/bin/bash
#SBATCH --job-name=s8bkctl
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=01:30:00
#SBATCH --comment="field=efficientai;appl=pytorch"
#SBATCH --array=0-1
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
# ============================================================================
# 7-8B SCALE-UP SMOKE.  Boot gate only -- no claims come out of this job.
#
# Three of the four arms have never been served by this harness, and the
# pure-SSM arm is a brand-new config shim, so this runs one short cell per arm
# and checks: boots, produces tokens, and lands on the partition it was told to.
#
# ARMS (array index):
#   0  M8   mistralai/Mamba-Codestral-7B-v0.1 (7.3B) via make_mamba2_hf_sglang.py
#          -- the pure-Mamba2 arm at 7-8B.  state-spaces tops out at 2.7B,
#             nvidia/mamba2-8b ships a Megatron checkpoint, and falcon-mamba-7b
#             is Mamba-1 (no sglang model).  This shim is the only route.
#   1  Ha8  Zyphra/Zamba2-7B-Instruct   -- ADDITIVE hybrid (shared attn block
#          injected on top of a full mamba stack; 2.7B measured ~2x pure-mamba)
#   2  Hs8  nvidia/Nemotron-H-8B-Base-8K -- SUBSTITUTIVE hybrid (M24/-24/*4:
#          attention REPLACES layers).  The 2.7B result cannot be assumed to
#          transfer across these two hybrid styles; that is the point of running
#          both.
#   3  T8   Qwen/Qwen2.5-7B             -- pure-Transformer control at scale
#
# Gate = realized_pin_check.py (realized partition), NOT the target histogram
# that Stage 0 used and that hid its broken 108 anchor.
# ============================================================================
set -uo pipefail
module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0
export PS1="" PATH="$HOME/.local/bin:$PATH"
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate

ROOT=/scratch/ehmoon/whlee/prefill-layer-alloc
HERE=$ROOT/workspace/engine-port/results/s8_scaleup
export HF_HOME=$ROOT/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="$ROOT/workspace/engine-port/triage/.triton_cache"
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

"$ROOT/workspace/engine-port/scripts/bootstrap/sync_engine_tree.sh" \
  "$HERE/runtime_source_manifest.sha256" || { echo "SYNC_ENGINE_TREE_FAILED"; exit 3; }

# Retry after smoke 865289: NemotronH is asserted out of the triton attention
# backend (server_args.py:1959 -- "the first layer might not be an attention
# layer"), which every prior campaign in this project used. Arm 0 boots it on
# flashinfer; arm 1 re-runs Zamba2-7B on flashinfer at the SAME cell it already
# ran on triton (44.13 ms p50), so the backend swap is a MEASURED delta rather
# than an unquantified cross-arm confound.
# BACKEND CONTROL. NemotronH cannot use triton (server_args.py:1959) and
# Zamba2-7B cannot use flashinfer (cudagraph capture dies in
# VariableLengthMergeStates, smoke 865297), so no hybrid arm can be run on both
# backends. The control therefore rides on the two arms that can:
#   T8fi  Qwen2.5-7B on flashinfer -- same cells it already ran on triton, so
#         the attention-path backend delta is measured on a pure-Transformer.
#   M8fi  Mamba-Codestral on flashinfer -- has NO attention layers, so this is
#         the negative control for the backend axis itself (expect ~0 delta).
# The Qwen delta bounds, but does not equal, NemotronH's (GQA-2 vs GQA-8);
# cross-arm claims involving Hs8 must say so.
ARM_LABELS=(T8fi M8fi)
ARM_MODELS=(
  "Qwen/Qwen2.5-7B"
  "$ROOT/hf_cache/mamba-codestral-7b-sglang"
)
BACKEND=flashinfer
AIDX=$SLURM_ARRAY_TASK_ID
ARM="${ARM_LABELS[$AIDX]}"
MODEL="${ARM_MODELS[$AIDX]}"

CTX=${CTX:-4096}; OUTTOK=${OUTTOK:-64}; CONC=${CONC:-8}
WARMUP_S=${WARMUP_S:-10}; MEASURE_S=${MEASURE_S:-20}
CTXCAP=$(( CTX + OUTTOK + 256 ))

PREPS=$(( CTX / 8 ))
PF="$HERE/s8_C${CTX}_prompt.txt"
[ -f "$PF" ] || python3 - "$PREPS" "$PF" <<'PY'
import sys
reps, pf = int(sys.argv[1]), sys.argv[2]
open(pf, "w").write("the robot walked slowly across the red planet " * reps)
PY
KPF="$HERE/s8_keepalive_prompt.txt"
[ -f "$KPF" ] || python3 - "$KPF" <<'PY'
import sys
# ~256-token keepalive prompt: keeps a prefill batch in flight (so the split
# partition is actually ACTIVE) without letting prefill become the bottleneck,
# which is what made decode batch co-vary with decode-SM in run 865098.
open(sys.argv[1], "w").write("the robot walked slowly across the red planet " * 32)
PY

RESULT="$HERE/s8smoke_${ARM}_${SLURM_ARRAY_JOB_ID:-local}_result.txt"; : > "$RESULT"
echo "ARM=$ARM MODEL=$MODEL ctx=$CTX conc=$CONC out=$OUTTOK torch=$(python -c 'import torch;print(torch.__version__)')" | tee -a "$RESULT"

# One coupled cell (D44) + the no-partition reference: enough to prove the
# model runs AND that both partition regimes are reachable for this arm.
for CELL in "d44:$HERE/pdmux_c_d44.yml:44" "np:$HERE/pdmux_np.yml:108"; do
  NAME="${CELL%%:*}"; rest="${CELL#*:}"; cfg="${rest%%:*}"; DSM="${rest##*:}"
  PORT=$(python3 -c "
import socket
s = socket.socket(); s.bind(('127.0.0.1', 0)); print(s.getsockname()[1]); s.close()")
  RUNID="s8smoke_${ARM}_${NAME}_${SLURM_ARRAY_JOB_ID:-local}"
  SRVLOG="$HERE/${RUNID}_srv.log"
  export PDMUX_TELEMETRY_PATH="$HERE/${RUNID}_telemetry.jsonl"; : > "$PDMUX_TELEMETRY_PATH"
  export PDMUX_RUN_ID="$RUNID"; export PDMUX_WORKLOAD_ID="${ARM}_${NAME}"
  export PDMUX_R2_POLICY=fixed; export PDMUX_R2_FIXED_DSM=$DSM

  echo "=== ARM=$ARM cell=$NAME decode_sm_target=$DSM cfg=$(basename $cfg) port=$PORT ===" | tee -a "$RESULT"
  T0=$(date +%s)
  python -m sglang.launch_server --model-path "$MODEL" --trust-remote-code --dtype bfloat16 \
    --attention-backend $BACKEND \
    --enable-pdmux --pdmux-config-path "$cfg" \
    --disable-piecewise-cuda-graph \
    --disable-overlap-schedule --chunked-prefill-size -1 --disable-radix-cache \
    --mem-fraction-static 0.80 --max-running-requests 48 --context-length $CTXCAP \
    --host 127.0.0.1 --port $PORT > "$SRVLOG" 2>&1 &
  SRV=$!
  ok=0
  for i in $(seq 1 360); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health 2>/dev/null||true)" = 200 ] && { ok=1; break; }
    kill -0 $SRV 2>/dev/null || { echo "SERVER_DIED arm=$ARM cell=$NAME" | tee -a "$RESULT"; break; }
    sleep 5
  done
  echo "boot_ok=$ok boot_seconds=$(( $(date +%s) - T0 ))" | tee -a "$RESULT"
  grep -ihoE "avail mem=[0-9.]+ GB|max_total_num_tokens=[0-9]+|Capture cuda graph (begin|end)" "$SRVLOG" | tail -4 | tee -a "$RESULT"

  if [ "$ok" = 1 ]; then
    RAW="$HERE/${RUNID}_raw.jsonl"
    python3 "$HERE/s0dc_client.py" --url "http://127.0.0.1:$PORT" \
      --prompt-file "$PF" --conc $CONC --out-tokens $OUTTOK \
      --warmup-s $WARMUP_S --measure-s $MEASURE_S \
      --keepalive-prefill 2 --keepalive-prompt-file "$KPF" \
      --raw-out "$RAW" --tag "${ARM}_${NAME}_smoke" 2>>"$RESULT" | tee -a "$RESULT"
    echo "client_rc=${PIPESTATUS[0]}" | tee -a "$RESULT"
    python3 "$HERE/realized_pin_check.py" "$PDMUX_TELEMETRY_PATH" "$DSM" | tee -a "$RESULT"
  else
    echo "BOOT_FAILED arm=$ARM cell=$NAME" | tee -a "$RESULT"
    grep -viE "pynvml|Future" "$SRVLOG" | tail -25 | tee -a "$RESULT"
  fi
  kill $SRV 2>/dev/null; pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null; sleep 5
done
echo "SMOKE_DONE ARM=$ARM" | tee -a "$RESULT"
