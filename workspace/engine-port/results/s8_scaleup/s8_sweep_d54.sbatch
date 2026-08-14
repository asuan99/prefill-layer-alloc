#!/bin/bash
#SBATCH --job-name=s8d54
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --comment="field=efficientai;appl=pytorch"
#SBATCH --output=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/s8_scaleup/%x_%j.out
#SBATCH --error=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/s8_scaleup/%x_%j.err
# ============================================================================
# C2 D=54 anchor de-confound (task: 2026-08-03).
#
# WHY: results/s8_scaleup's grid is {16,24,44,92}; the sticky/E1 grid runs
# D in [16,54], so the sticky G_LEVER threshold (results/s8_frontier/
# c2_anchor.py) currently anchors on an INTERPOLATED bracket ([16,44] and
# [16,92]) rather than the actual endpoint. This job measures [prefill 16,
# decode 54, idle 38] directly -- a POINT value, not an extrapolation.
#
# THIS IS A BLOCK DESIGN, not a bigger single-boot sweep. r = stat(d16)/
# stat(d54) crosses two cells, so the unit of replication is a BLOCK = one
# fresh boot of d16 AND one fresh boot of d54 (mirrors s8_sweep.sbatch's
# existing per-cell reboot -- reps within a cell already share one boot,
# which is documented pseudo-replication in c2_anchor.py's own docstring).
# Cell order alternates by block parity (odd: d16 then d54; even: d54 then
# d16) so cell identity is not confounded with block/wall-clock order --
# same counterbalancing rationale as job 872077 (M3, see MEMORY.md
# "2026-08-03" note on n_indep).
#
# Everything else (server args, workload, rep count, keepalive duty cycle)
# is copied byte-for-byte from the 865493 invocation of s8_sweep.sbatch --
# NOT the sbatch script's own generic defaults, which differ (CTX=4096 vs
# 1024 here, KEEPA_REPS=32 vs 224 here). Confirmed against
# s8_deconf_T8_C1024_d16_865493_srv.log's printed ServerArgs and
# s8_deconf_*_865493_result.txt headers. Only the cell partition changes.
#
# Submit per arm (QOS aanv8 allows only 2 running jobs/user):
#   sbatch --export=ALL,ARM_IDS="3" s8_sweep_d54.sbatch      # T8  (required)
#   sbatch --export=ALL,ARM_IDS="2" s8_sweep_d54.sbatch      # Hs8 (secondary)
# ARM_IDS is space-separated indices into ARM_LABELS below; default is T8 only.
#
# NOT a claim job. Physical anchor point for a pre-registered threshold, not
# a policy/performance comparison. claims-auditor has not reviewed this.
# ============================================================================
set -uo pipefail
module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0
export PS1="" PATH="$HOME/.local/bin:$PATH"
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate

ROOT=/scratch/ehmoon/whlee/prefill-layer-alloc
HERE=$ROOT/workspace/engine-port/results/s8_scaleup
export HF_HOME=$ROOT/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="$ROOT/workspace/engine-port/triage/.triton_cache_d54_${SLURM_JOB_ID:-local}"
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

"$ROOT/workspace/engine-port/scripts/bootstrap/sync_engine_tree.sh" \
  "$HERE/runtime_source_manifest_d54.sha256" || { echo "SYNC_ENGINE_TREE_FAILED"; exit 3; }

ARM_LABELS=(M8 Ha8 Hs8 T8)
ARM_MODELS=(
  "$ROOT/hf_cache/mamba-codestral-7b-sglang"
  "Zyphra/Zamba2-7B-Instruct"
  "nvidia/Nemotron-H-8B-Base-8K"
  "Qwen/Qwen2.5-7B"
)
ARM_BACKENDS=(triton triton flashinfer triton)

# --- copied from the 865493 invocation, not the generic script defaults ---
CTX="${CTX:-1024}"
CONC="${CONC:-16}"
OUTTOK="${OUTTOK:-512}"
WARMUP_S="${WARMUP_S:-20}"
MEASURE_S="${MEASURE_S:-60}"
REPS="${REPS:-4}"
KEEPA_N="${KEEPA_N:-8}"
KEEPA_REPS="${KEEPA_REPS:-224}"
CTXCAP=$(( CTX + OUTTOK + 256 ))
PIN_GATE="${PIN_GATE:-0.80}"
BLOCKS="${BLOCKS:-4}"
ARM_IDS="${ARM_IDS:-3}"

echo "DESIGN: CTX=$CTX CONC=$CONC OUTTOK=$OUTTOK WARMUP_S=$WARMUP_S MEASURE_S=$MEASURE_S" \
     "REPS=$REPS KEEPA_N=$KEEPA_N KEEPA_REPS=$KEEPA_REPS BLOCKS=$BLOCKS ARM_IDS=$ARM_IDS PIN_GATE=$PIN_GATE"

PREPS=$(( CTX / 8 ))
PF="$HERE/s8_C${CTX}_prompt.txt"
[ -f "$PF" ] || python3 - "$PREPS" "$PF" <<'PY'
import sys
reps, pf = int(sys.argv[1]), sys.argv[2]
open(pf, "w").write("the robot walked slowly across the red planet " * reps)
PY
KPF="$HERE/s8_keepalive_prompt_${KEEPA_REPS}.txt"
[ -f "$KPF" ] || python3 - "$KEEPA_REPS" "$KPF" <<'PY'
import sys
open(sys.argv[2], "w").write("the robot walked slowly across the red planet " * int(sys.argv[1]))
PY

JOBTAG="${SLURM_JOB_ID:-local}"
declare -A CELL_CFG=( [d16]="$HERE/pdmux_p16_d16.yml:16" [d54]="$HERE/pdmux_p16_d54.yml:54" )

run_cell () {
  local ARM="$1" MODEL="$2" BACKEND="$3" NAME="$4" BLK="$5" RESULT="$6"
  local cfg="${CELL_CFG[$NAME]%%:*}" DSM="${CELL_CFG[$NAME]##*:}"
  local PORT
  PORT=$(python3 -c "
import socket
s = socket.socket(); s.bind(('127.0.0.1', 0)); print(s.getsockname()[1]); s.close()")
  local RUNID="s8_deconf_${ARM}_C${CTX}_${NAME}_${JOBTAG}_blk${BLK}"
  local SRVLOG="$HERE/${RUNID}_srv.log"
  export PDMUX_TELEMETRY_PATH="$HERE/${RUNID}_telemetry.jsonl"; : > "$PDMUX_TELEMETRY_PATH"
  export PDMUX_RUN_ID="$RUNID"; export PDMUX_WORKLOAD_ID="deconf_${ARM}_C${CTX}_${NAME}_blk${BLK}"
  export PDMUX_R2_POLICY=fixed; export PDMUX_R2_FIXED_DSM=$DSM

  echo "=== ARM=$ARM block=$BLK cell=$NAME decode_sm=$DSM cfg=$(basename "$cfg") port=$PORT ===" | tee -a "$RESULT"
  python -m sglang.launch_server --model-path "$MODEL" --trust-remote-code --dtype bfloat16 \
    --attention-backend "$BACKEND" \
    --enable-pdmux --pdmux-config-path "$cfg" \
    --disable-piecewise-cuda-graph \
    --disable-overlap-schedule --chunked-prefill-size -1 --disable-radix-cache \
    --mem-fraction-static 0.80 --max-running-requests 48 --context-length $CTXCAP \
    --host 127.0.0.1 --port $PORT > "$SRVLOG" 2>&1 &
  local SRV=$!
  local ok=0
  for i in $(seq 1 360); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health 2>/dev/null||true)" = 200 ] && { ok=1; break; }
    kill -0 $SRV 2>/dev/null || { echo "SERVER_DIED block=$BLK cell=$NAME" | tee -a "$RESULT"; break; }
    sleep 5
  done
  echo "boot_ok=$ok block=$BLK cell=$NAME" | tee -a "$RESULT"

  if [ "$ok" = 1 ]; then
    local first_meas=1
    for r in $(seq 1 $REPS); do
      local CLK="$HERE/${RUNID}_rep${r}_clocks.csv"
      ( for _ in $(seq 1 $(( (WARMUP_S + MEASURE_S) / 5 + 2 )) ); do
          nvidia-smi --query-gpu=clocks.sm,power.draw,utilization.gpu \
            --format=csv,noheader,nounits >> "$CLK" 2>/dev/null; sleep 5
        done ) &
      local CLKPID=$!
      local RAW="$HERE/${RUNID}_rep${r}_raw.jsonl"
      python3 "$HERE/s0dc_client.py" --url "http://127.0.0.1:$PORT" \
        --prompt-file "$PF" --conc $CONC --out-tokens $OUTTOK \
        --warmup-s $WARMUP_S --measure-s $MEASURE_S \
        --keepalive-prefill $KEEPA_N --keepalive-prompt-file "$KPF" \
        --raw-out "$RAW" --tag "deconf_${ARM}_C${CTX}_${NAME}_blk${BLK}_r${r}" 2>>"$RESULT" | tee -a "$RESULT"
      local CLIENT_RC=${PIPESTATUS[0]}
      kill $CLKPID 2>/dev/null; wait $CLKPID 2>/dev/null
      if [ "$CLIENT_RC" != 0 ] && [ "$first_meas" = 1 ]; then
        echo "ABORT arm=$ARM block=$BLK cell=$NAME: first measurement failed (client rc=$CLIENT_RC)" | tee -a "$RESULT"
        kill $SRV 2>/dev/null; pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null
        return 1
      fi
      first_meas=0
      python3 - "$CLK" <<'PY' | tee -a "$RESULT"
import statistics, sys
try:
    rows = [l.split(",") for l in open(sys.argv[1]) if l.strip()]
    sm = [float(r[0]) for r in rows]
    print(f"CLOCKS sm_mhz_mean={statistics.fmean(sm):.0f} sm_mhz_min={min(sm):.0f} n={len(sm)}")
except Exception as exc:
    print(f"CLOCKS unavailable: {exc}")
PY
    done
    python3 "$HERE/realized_pin_check.py" "$PDMUX_TELEMETRY_PATH" "$DSM" "$PIN_GATE" | tee -a "$RESULT"
  else
    echo "BOOT_FAILED block=$BLK cell=$NAME" | tee -a "$RESULT"
    grep -viE "pynvml|Future" "$SRVLOG" | tail -25 | tee -a "$RESULT"
  fi
  kill $SRV 2>/dev/null; pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null; sleep 5
  return 0
}

for AIDX in $ARM_IDS; do
  ARM="${ARM_LABELS[$AIDX]}"
  MODEL="${ARM_MODELS[$AIDX]}"
  BACKEND="${ARM_BACKENDS[$AIDX]}"
  RESULT="$HERE/s8_deconf_${ARM}_C${CTX}_d54anchor_${JOBTAG}_result.txt"; : > "$RESULT"
  echo "ARM=$ARM MODEL=$MODEL BACKEND=$BACKEND CTX=$CTX conc=$CONC out=$OUTTOK" \
       "warmup=${WARMUP_S}s measure=${MEASURE_S}s reps=$REPS keepN=$KEEPA_N blocks=$BLOCKS" \
       "(matched to job 865493)" | tee -a "$RESULT"

  for BLK in $(seq 1 $BLOCKS); do
    if [ $(( BLK % 2 )) -eq 1 ]; then
      ORDER="d16 d54"
    else
      ORDER="d54 d16"
    fi
    echo "BLOCK $BLK order=[$ORDER]" | tee -a "$RESULT"
    for NAME in $ORDER; do
      run_cell "$ARM" "$MODEL" "$BACKEND" "$NAME" "$BLK" "$RESULT" || \
        echo "BLOCK $BLK ABORTED at cell=$NAME" | tee -a "$RESULT"
    done
  done
  echo "SWEEP_DONE ARM=$ARM CTX=$CTX BLOCKS=$BLOCKS" | tee -a "$RESULT"
done
