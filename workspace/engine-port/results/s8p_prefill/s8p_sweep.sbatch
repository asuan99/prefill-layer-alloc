#!/bin/bash
#SBATCH --job-name=s8psweep
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --comment=pytorch
#SBATCH --chdir=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/s8p_prefill
#SBATCH --output=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/s8p_prefill/%x_%j.out
#SBATCH --error=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/s8p_prefill/%x_%j.err
# ============================================================================
# 7-8B SCALE-UP, PREFILL-SM MIRROR of s8_scaleup (which swept decode SM with
# prefill pinned at 16). This sweeps PREFILL SM with decode pinned at 16,
# same 4 arms. See s8p_prefill/DESIGN.md for the full rationale, cell/arm
# table, and pre-registered gates -- do not edit the protocol here without
# updating that file too.
#
# QOS is submit<=4/run<=2 per user, so submit this TWICE with disjoint
# ARM_IDS (arms run sequentially inside one job, exactly like s8_sweep.sbatch):
#   sbatch --export=ALL,ARM_IDS="0 1" s8p_sweep.sbatch   # M8, Ha8
#   sbatch --export=ALL,ARM_IDS="2 3" s8p_sweep.sbatch   # Hs8, T8
#
# ARMS: 0=M8 pure-Mamba2 7.3B, 1=Ha8 Zamba2-7B (additive hybrid),
# 2=Hs8 Nemotron-H-8B (substitutive hybrid, flashinfer-only), 3=T8 Qwen2.5-7B.
#
# STATUS (put this in every downstream reference): this is a PREFILL-side
# EQUI-LOAD CURVE, not a policy comparison. decode is pinned at 16 SM and
# idle SM is left deliberately idle in every low-prefill cell -- a real R2
# policy would give unused decode SM to prefill instead, so
# prefill_SM + decode_SM <= 108 is NOT respected here on purpose (mirrors
# s8_scaleup FINDINGS sec 0's status caveat for the decode axis). This
# establishes lever EXISTENCE only, not a goodput claim (methodology gate #1).
#
# NOT a claim job. s0_deconfound/DESIGN.md sec 5 pre-registration still
# governs -- nothing here goes to canon before claims-auditor.
# ============================================================================
set -uo pipefail
module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0
export PS1="" PATH="$HOME/.local/bin:$PATH"
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate

ROOT=/scratch/ehmoon/whlee/prefill-layer-alloc
HERE=$ROOT/workspace/engine-port/results/s8p_prefill
export HF_HOME=$ROOT/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="$ROOT/workspace/engine-port/triage/.triton_cache_s8p_${SLURM_JOB_ID:-local}"
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

"$ROOT/workspace/engine-port/scripts/bootstrap/sync_engine_tree.sh" \
  "$HERE/runtime_source_manifest.sha256" || { echo "SYNC_ENGINE_TREE_FAILED"; exit 3; }

ARM_LABELS=(M8 Ha8 Hs8 T8)
ARM_MODELS=(
  "$ROOT/hf_cache/mamba-codestral-7b-sglang"
  "Zyphra/Zamba2-7B-Instruct"
  "nvidia/Nemotron-H-8B-Base-8K"
  "Qwen/Qwen2.5-7B"
)
# Same unavoidable backend asymmetry as s8_scaleup (server_args.py:1959 bans
# NemotronH from triton; Zamba2-7B's cudagraph capture dies on flashinfer).
ARM_BACKENDS=(triton triton flashinfer triton)

CONC="${CONC:-8}"                    # background decode requests in flight
DECODE_OUT="${DECODE_OUT:-512}"      # long ignore_eos decode, keeps split resident
PROBE_CONC="${PROBE_CONC:-1}"        # serial prefill probes (protocol default)
WARMUP_S="${WARMUP_S:-20}"
MEASURE_S="${MEASURE_S:-60}"
REPS="${REPS:-4}"                    # methodology gate: n>=4
PIN_GATE="${PIN_GATE:-0.80}"

if [ "${SMOKE_CELLS:-}" = "1" ]; then
  # Smoke-test subset (DESIGN.md sec 8): the two extremes only.
  CELLS=("p16:$HERE/pdmux_pf16_d16.yml:16" "p92:$HERE/pdmux_pf92_d16.yml:92")
else
  CELLS=("p16:$HERE/pdmux_pf16_d16.yml:16" "p24:$HERE/pdmux_pf24_d16.yml:24"
         "p44:$HERE/pdmux_pf44_d16.yml:44" "p92:$HERE/pdmux_pf92_d16.yml:92")
fi

# L axis (this campaign's model-discriminating axis): prompt length in
# {512,1024,2048,4096}, generated with the same "PREPS=L/8 repeats of a fixed
# 9-word phrase" scheme s8_scaleup used for its context prompts (verified
# against the Qwen2.5-7B tokenizer: L/8 reps -> L+1 tokens exactly).
L_LIST="${L_LIST:-512 1024 2048 4096}"
PROBE_PROMPT_ARGS=()
for L in $L_LIST; do
  PF="$HERE/s8p_L${L}_prompt.txt"
  if [ ! -f "$PF" ]; then
    python3 - "$L" "$PF" <<'PY'
import sys
L, pf = int(sys.argv[1]), sys.argv[2]
reps = max(1, L // 8)
open(pf, "w").write("the robot walked slowly across the red planet " * reps)
PY
  fi
  PROBE_PROMPT_ARGS+=(--probe-prompt "${L}:${PF}")
done

# Background decode prompt: short, so its own prefill admission never
# competes for the same window as the probe's prefill (both would land on the
# split_prefill_batch, but a short decode prompt finishes fast, minimizing
# contention). 8 reps of the same phrase (~64 tokens).
DECODE_PROMPT_REPS="${DECODE_PROMPT_REPS:-8}"
DPF="$HERE/s8p_decode_prompt_${DECODE_PROMPT_REPS}.txt"
[ -f "$DPF" ] || python3 - "$DECODE_PROMPT_REPS" "$DPF" <<'PY'
import sys
open(sys.argv[2], "w").write("the robot walked slowly across the red planet " * int(sys.argv[1]))
PY

MAXL=0
for L in $L_LIST; do [ "$L" -gt "$MAXL" ] && MAXL=$L; done
# Cap only has to cover the longer of (a) a prefill probe: L + 1 new token,
# and (b) the decode background request: ~DECODE_PROMPT_REPS*8 tokens +
# DECODE_OUT. (a) dominates for every L in the default L_LIST; the margin
# below covers both with room to spare.
CTXCAP=$(( MAXL + DECODE_OUT + 256 ))

ARM_IDS="${ARM_IDS:-0 1 2 3}"
JOBTAG="${SLURM_JOB_ID:-local}"

for AIDX in $ARM_IDS; do
ARM="${ARM_LABELS[$AIDX]}"
MODEL="${ARM_MODELS[$AIDX]}"
BACKEND="${ARM_BACKENDS[$AIDX]}"
RESULT="$HERE/s8p_${ARM}_${JOBTAG}_result.txt"; : > "$RESULT"
FIRST_MEASUREMENT=1
echo "ARM=$ARM MODEL=$MODEL BACKEND=$BACKEND decode_conc=$CONC decode_out=$DECODE_OUT " \
     "probe_conc=$PROBE_CONC L_LIST=[$L_LIST] warmup=${WARMUP_S}s measure=${MEASURE_S}s " \
     "reps=$REPS ctxcap=$CTXCAP" | tee -a "$RESULT"

for CELL in "${CELLS[@]}"; do
  NAME="${CELL%%:*}"; rest="${CELL#*:}"; cfg="${rest%%:*}"; PSM="${rest##*:}"
  PORT=$(python3 -c "
import socket
s = socket.socket(); s.bind(('127.0.0.1', 0)); print(s.getsockname()[1]); s.close()")
  RUNID="s8p_${ARM}_${NAME}_${JOBTAG}"
  SRVLOG="$HERE/${RUNID}_srv.log"
  export PDMUX_TELEMETRY_PATH="$HERE/${RUNID}_telemetry.jsonl"; : > "$PDMUX_TELEMETRY_PATH"
  export PDMUX_RUN_ID="$RUNID"; export PDMUX_WORKLOAD_ID="${ARM}_${NAME}"
  export PDMUX_R2_POLICY=fixed; export PDMUX_R2_FIXED_DSM=16

  echo "=== ARM=$ARM cell=$NAME prefill_sm=$PSM cfg=$(basename $cfg) port=$PORT ===" | tee -a "$RESULT"
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
      # Clock sampling, same rationale as s8_sweep.sbatch (idle-SM boost
      # headroom flatters low-P cells, i.e. biases toward UNDER-estimating
      # prefill-SM sensitivity).
      CLK="$HERE/${RUNID}_rep${r}_clocks.csv"
      ( for _ in $(seq 1 $(( (WARMUP_S + MEASURE_S) / 5 + 2 )) ); do
          nvidia-smi --query-gpu=clocks.sm,power.draw,utilization.gpu \
            --format=csv,noheader,nounits >> "$CLK" 2>/dev/null; sleep 5
        done ) &
      CLKPID=$!
      RAW="$HERE/${RUNID}_rep${r}_raw.jsonl"
      python3 "$HERE/s8p_client.py" --url "http://127.0.0.1:$PORT" \
        --decode-prompt-file "$DPF" --decode-conc $CONC --decode-out-tokens $DECODE_OUT \
        "${PROBE_PROMPT_ARGS[@]}" --probe-conc $PROBE_CONC \
        --warmup-s $WARMUP_S --measure-s $MEASURE_S \
        --raw-out "$RAW" --tag "${ARM}_${NAME}_r${r}" 2>>"$RESULT" | tee -a "$RESULT"
      CLIENT_RC=${PIPESTATUS[0]}
      kill $CLKPID 2>/dev/null; wait $CLKPID 2>/dev/null
      # Fail fast: s8's 865006 lesson -- tee swallows the client's exit
      # status, so check PIPESTATUS explicitly.
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
    python3 "$HERE/prefill_pin_check.py" "$PDMUX_TELEMETRY_PATH" "$PSM" "$PIN_GATE" | tee -a "$RESULT"
  else
    echo "BOOT_FAILED cell=$NAME" | tee -a "$RESULT"
    grep -viE "pynvml|Future" "$SRVLOG" | tail -25 | tee -a "$RESULT"
  fi
  kill $SRV 2>/dev/null; pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null; sleep 5
done
echo "SWEEP_DONE ARM=$ARM" | tee -a "$RESULT"
done
