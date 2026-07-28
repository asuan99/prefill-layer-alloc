#!/bin/bash
#SBATCH --job-name=s8sweep
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --comment=pytorch
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# ============================================================================
# 7-8B SCALE-UP: decode-SM sweep, confounded and de-confounded, 4 arms.
#
# Submit twice (one job per mode; the four arms run sequentially inside a job
# because QOS aanv8 allows only 4 submitted / 2 running jobs per user):
#   sbatch --export=ALL,MODE=coupled s8_sweep.sbatch   # prefill = 108-D  (Stage-0 pattern)
#   sbatch --export=ALL,MODE=deconf  s8_sweep.sbatch   # prefill pinned 16 (865098 pattern)
# ARM_IDS selects the arms (default all four).
#
# ARMS: 0=M8 pure-Mamba2 7.3B (Mamba-Codestral shim), 1=Ha8 Zamba2-7B
# (ADDITIVE hybrid), 2=Hs8 Nemotron-H-8B (SUBSTITUTIVE hybrid), 3=T8 Qwen2.5-7B
# (pure Transformer).  Both hybrid styles are run because the 2.7B result came
# from an additive hybrid only, and the structural cost argument does not carry
# over to substitutive ones.
#
# WHAT THIS FIXES vs the 2.7B campaigns
#   1. The 108 reference is a real one.  Stage 0 built it from pdmux_d16.yml
#      with the policy OFF, and the legacy adjust_stream_groups path (threshold
#      0 in manual_divisions) pinned it to (92,16): the "no-contention 108 SM
#      anchor" ran decode at 16 SM in 82-97% of decode-active samples, which is
#      why D16 vs D108 came out 1.00 +/- 0.01 (same partition, measured twice).
#      See results/s0_deconfound/PARTITION_RESIDENCY_STAGE0.md.  Here the NP
#      cell keeps the policy ON and targets 108 -> the unpartitioned stream.
#   2. The gate reads the REALIZED partition, not the policy target.
#   3. Keepalive prefill uses a SHORT prompt, so a prefill batch stays in flight
#      (partition genuinely active) without prefill becoming the bottleneck --
#      that was what made decode batch co-vary with decode-SM in run 865098.
#      Realized decode batch is recorded per cell either way; it is a reported
#      quantity, not an assumed one.
#
# NOT a claim job.  Nothing here goes to canon without claims-auditor first
# (s0_deconfound/DESIGN.md §5 pre-registration still governs this question).
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

MODE="${MODE:-deconf}"
ARM_LABELS=(M8 Ha8 Hs8 T8)
ARM_MODELS=(
  "$ROOT/hf_cache/mamba-codestral-7b-sglang"
  "Zyphra/Zamba2-7B-Instruct"
  "nvidia/Nemotron-H-8B-Base-8K"
  "Qwen/Qwen2.5-7B"
)
# Backend is triton everywhere (matching every prior campaign in this project)
# EXCEPT NemotronH, which sglang asserts out of the triton path entirely
# (server_args.py:1959, "the first layer might not be an attention layer").
# Zamba2-7B cannot run flashinfer either (cudagraph capture dies in
# VariableLengthMergeStates), so no hybrid arm runs on both backends. The
# backend delta is measured instead on Qwen2.5-7B (s8_backend_control.sbatch),
# which bounds -- but does not equal -- NemotronH's; cross-arm claims involving
# Hs8 must carry that caveat.
ARM_BACKENDS=(triton triton flashinfer triton)

CTX="${CTX:-4096}"
CONC="${CONC:-16}"                 # decode requests held in flight (closed loop)
OUTTOK="${OUTTOK:-512}"            # steady window must be decode, not prefill
WARMUP_S="${WARMUP_S:-20}"
MEASURE_S="${MEASURE_S:-60}"
REPS="${REPS:-4}"                  # methodology gate: n>=4
KEEPA_N="${KEEPA_N:-2}"
CTXCAP=$(( CTX + OUTTOK + 256 ))

if [ "$MODE" = "coupled" ]; then
  CELLS=("d16:$HERE/pdmux_c_d16.yml:16" "d44:$HERE/pdmux_c_d44.yml:44"
         "d92:$HERE/pdmux_c_d92.yml:92" "np:$HERE/pdmux_np.yml:108")
else
  CELLS=("d16:$HERE/pdmux_p16_d16.yml:16" "d24:$HERE/pdmux_p16_d24.yml:24"
         "d44:$HERE/pdmux_p16_d44.yml:44" "d92:$HERE/pdmux_p16_d92.yml:92"
         "np:$HERE/pdmux_np.yml:108")
fi

PREPS=$(( CTX / 8 ))
PF="$HERE/s8_C${CTX}_prompt.txt"
[ -f "$PF" ] || python3 - "$PREPS" "$PF" <<'PY'
import sys
reps, pf = int(sys.argv[1]), sys.argv[2]
open(pf, "w").write("the robot walked slowly across the red planet " * reps)
PY
# Keepalive prompt length is the duty-cycle knob. keepalive_worker is a SERIAL
# loop (issue -> await completion -> repeat), so with a short prompt the prefill
# pipeline goes idle between requests and the green-ctx split reverts to the
# unpartitioned stream. Runs 865311/865312 held the partition only 13-93% of
# decode time for exactly this reason (NOTES_865311_865312.md). Longer keepalive
# prefills raise the per-worker duty cycle; in deconf mode prefill is pinned at
# 16 SM in every cell, so the extra prefill load is identical across the D axis.
KEEPA_REPS="${KEEPA_REPS:-32}"
KPF="$HERE/s8_keepalive_prompt_${KEEPA_REPS}.txt"
[ -f "$KPF" ] || python3 - "$KEEPA_REPS" "$KPF" <<'PY'
import sys
open(sys.argv[2], "w").write("the robot walked slowly across the red planet " * int(sys.argv[1]))
PY

ARM_IDS="${ARM_IDS:-0 1 2 3}"
JOBTAG="${SLURM_JOB_ID:-local}"

for AIDX in $ARM_IDS; do
ARM="${ARM_LABELS[$AIDX]}"
MODEL="${ARM_MODELS[$AIDX]}"
BACKEND="${ARM_BACKENDS[$AIDX]}"
RESULT="$HERE/s8_${MODE}_${ARM}_C${CTX}_${JOBTAG}_result.txt"; : > "$RESULT"
FIRST_MEASUREMENT=1
echo "MODE=$MODE ARM=$ARM MODEL=$MODEL BACKEND=$BACKEND CTX=$CTX conc=$CONC out=$OUTTOK warmup=${WARMUP_S}s measure=${MEASURE_S}s reps=$REPS keepN=$KEEPA_N(short prompt)" | tee -a "$RESULT"

for CELL in "${CELLS[@]}"; do
  NAME="${CELL%%:*}"; rest="${CELL#*:}"; cfg="${rest%%:*}"; DSM="${rest##*:}"
  PORT=$(python3 -c "
import socket
s = socket.socket(); s.bind(('127.0.0.1', 0)); print(s.getsockname()[1]); s.close()")
  RUNID="s8_${MODE}_${ARM}_C${CTX}_${NAME}_${JOBTAG}"
  SRVLOG="$HERE/${RUNID}_srv.log"
  export PDMUX_TELEMETRY_PATH="$HERE/${RUNID}_telemetry.jsonl"; : > "$PDMUX_TELEMETRY_PATH"
  export PDMUX_RUN_ID="$RUNID"; export PDMUX_WORKLOAD_ID="${MODE}_${ARM}_C${CTX}_${NAME}"
  export PDMUX_R2_POLICY=fixed; export PDMUX_R2_FIXED_DSM=$DSM

  echo "=== MODE=$MODE ARM=$ARM cell=$NAME decode_sm=$DSM cfg=$(basename $cfg) port=$PORT ===" | tee -a "$RESULT"
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
    kill -0 $SRV 2>/dev/null || { echo "SERVER_DIED cell=$NAME" | tee -a "$RESULT"; break; }
    sleep 5
  done
  echo "boot_ok=$ok cell=$NAME" | tee -a "$RESULT"

  if [ "$ok" = 1 ]; then
    for r in $(seq 1 $REPS); do
      # SM clock sampling: arms differ in how many SMs are active (32 vs 108),
      # and idle-SM boost headroom flatters the low-D cells, i.e. biases toward
      # UNDER-estimating decode-SM sensitivity (s0_deconfound/DESIGN.md §3).
      CLK="$HERE/${RUNID}_rep${r}_clocks.csv"
      ( for _ in $(seq 1 $(( (WARMUP_S + MEASURE_S) / 5 + 2 )) ); do
          nvidia-smi --query-gpu=clocks.sm,power.draw,utilization.gpu \
            --format=csv,noheader,nounits >> "$CLK" 2>/dev/null; sleep 5
        done ) &
      CLKPID=$!
      RAW="$HERE/${RUNID}_rep${r}_raw.jsonl"
      python3 "$HERE/s0dc_client.py" --url "http://127.0.0.1:$PORT" \
        --prompt-file "$PF" --conc $CONC --out-tokens $OUTTOK \
        --warmup-s $WARMUP_S --measure-s $MEASURE_S \
        --keepalive-prefill $KEEPA_N --keepalive-prompt-file "$KPF" \
        --raw-out "$RAW" --tag "${MODE}_${ARM}_C${CTX}_${NAME}_r${r}" 2>>"$RESULT" | tee -a "$RESULT"
      CLIENT_RC=${PIPESTATUS[0]}
      kill $CLKPID 2>/dev/null; wait $CLKPID 2>/dev/null
      # Fail fast: run 865006 burned 25 min/arm producing zero usable samples
      # because tee swallowed the client's exit status.
      if [ "$CLIENT_RC" != 0 ] && [ "$FIRST_MEASUREMENT" = 1 ]; then
        echo "ABORT arm=$ARM: first measurement failed (client rc=$CLIENT_RC);" \
             "skipping the rest of this arm" | tee -a "$RESULT"
        kill $SRV 2>/dev/null; pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null
        break 2
      fi
      FIRST_MEASUREMENT=0
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
    python3 "$HERE/realized_pin_check.py" "$PDMUX_TELEMETRY_PATH" "$DSM" \
      "${PIN_GATE:-0.80}" | tee -a "$RESULT"
  else
    echo "BOOT_FAILED cell=$NAME" | tee -a "$RESULT"
    grep -viE "pynvml|Future" "$SRVLOG" | tail -25 | tee -a "$RESULT"
  fi
  kill $SRV 2>/dev/null; pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null; sleep 5
done
echo "SWEEP_DONE MODE=$MODE ARM=$ARM CTX=$CTX" | tee -a "$RESULT"
done
