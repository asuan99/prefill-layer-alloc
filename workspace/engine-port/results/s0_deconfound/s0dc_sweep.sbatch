#!/bin/bash
#SBATCH --job-name=s0dc
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --time=08:00:00
#SBATCH --comment="field=efficientai;appl=pytorch"
#SBATCH --array=0-2
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
# ============================================================================
# STAGE 0 DE-CONFOUND SWEEP  (design + pre-registered predictions: DESIGN.md)
#
# WHAT IS DIFFERENT FROM STAGE 0 (this is the whole point):
#   Stage 0 used divisions [108-D, D], so raising decode-SM LOWERED prefill-SM.
#   The ITL(D) curve therefore mixed "decode got more SMs" with "prefill got
#   fewer SMs and stopped competing", and was judged CONFOUNDED.
#   Here prefill is PINNED AT 16 SM in every arm (pdmux_p16_d*.yml) and only
#   decode-SM moves, 16 -> 24 -> 44 -> 92, with the remainder left idle.
#   Green-ctx honors that (probe job 864669: B(16,44) eff 55.7 == B(64,44)
#   eff 55.6, vs B(16,92) eff 110.7), so this is config-only.
#
#   Client is also different: s0dc_client.py is CLOSED-LOOP with fixed
#   occupancy + warmup discard, where stage0_client.py fired a one-shot burst
#   whose batch drained mid-measurement.
#
# PRE-REGISTERED (DESIGN.md §5): H0 = ITL flat across D (<5% spread) in all
# three arms, which closes the CONFOUNDED verdict quantitatively. H1 = a
# monotone slope survives prefill pinning, which would reopen Stage 0; in that
# case do NOT edit canon directly, route to claims-auditor first.
#
# ARMS (array): 0=M pure-Mamba2 (negative control -- decode is O(1) recurrent
# so it CANNOT be SM-bound; any slope here measures leftover confound),
# 1=H Zamba2 hybrid (target), 2=T Qwen2.5-3B pure-attn (positive control --
# if decode-SM sensitivity exists anywhere it should be largest here).
#
# ctx fixed at 4096: Stage 0 established ctx-invariance over 4k/8k/16k, so
# reps buy more here than ctx points do.
# ============================================================================
set -uo pipefail
module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0
export PS1="" PATH="$HOME/.local/bin:$PATH"
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate

ROOT=/scratch/ehmoon/whlee/prefill-layer-alloc
HERE=$ROOT/workspace/engine-port/results/s0_deconfound
export HF_HOME=$ROOT/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="$ROOT/workspace/engine-port/triage/.triton_cache"

# flock'd since f99fd84, so concurrent array tasks no longer race here.
# Write the hash manifest into the campaign dir (as g2_0_* does) rather than
# the shared default path: raw outputs are not versioned, so this manifest is
# the provenance record tying results to the exact engine sources that ran.
"$ROOT/workspace/engine-port/scripts/bootstrap/sync_engine_tree.sh" \
  "$HERE/runtime_source_manifest.sha256" || {
  echo "SYNC_ENGINE_TREE_FAILED"; exit 3; }

ARM_LABELS=(M H T)
ARM_MODELS=(
  "$ROOT/hf_cache/mamba2-2.7b-sglang"   # M  pure-SSM negative control
  "Zyphra/Zamba2-2.7B"                  # H  hybrid target
  "Qwen/Qwen2.5-3B"                     # T  pure-attn positive control
)
AIDX=$SLURM_ARRAY_TASK_ID
ARM="${ARM_LABELS[$AIDX]}"
MODEL="${ARM_MODELS[$AIDX]}"

CTX="${CTX:-4096}"
CONC="${CONC:-16}"                   # decode requests held in flight (fixed occupancy)
OUTTOK="${OUTTOK:-512}"              # long enough that the steady window is decode, not prefill
WARMUP_S="${WARMUP_S:-20}"
MEASURE_S="${MEASURE_S:-60}"
DTYPE="${DTYPE:-bfloat16}"           # all arms bf16, dtype matched
REPS="${REPS:-4}"                    # methodology gate: n>=4 before any policy claim
KEEPA_N="${KEEPA_N:-2}"
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

# Context cap must cover prompt + completion with slack. Run 865006 inherited
# Stage 0's CTX+512 while raising OUTTOK from 32 to 512, which left the budget
# exactly at the boundary; the prompt tokenizes to 4097 rather than 4096, so
# every decode request was rejected with "4609 > 4608" and the whole campaign
# produced zero decode traffic.
CTXCAP=$(( CTX + OUTTOK + 256 ))

SM_ARMS=(16 24 44 92)
declare -A CFG=(
  [16]="$HERE/pdmux_p16_d16.yml"
  [24]="$HERE/pdmux_p16_d24.yml"
  [44]="$HERE/pdmux_p16_d44.yml"
  [92]="$HERE/pdmux_p16_d92.yml"
)

PREPS=$(( CTX / 8 )); [ "$PREPS" -lt 1 ] && PREPS=1
PF="$HERE/s0dc_${ARM}_C${CTX}_prompt.txt"
python3 - "$PREPS" "$PF" <<'PY'
import sys
reps, pf = int(sys.argv[1]), sys.argv[2]
open(pf, "w").write("the robot walked slowly across the red planet " * reps)
PY

RESULT="$HERE/s0dc_${ARM}_C${CTX}_${SLURM_ARRAY_JOB_ID:-local}_result.txt"; : > "$RESULT"
FIRST_MEASUREMENT=1
echo "ARM=$ARM MODEL=$MODEL CTX=$CTX conc=$CONC out=$OUTTOK warmup=${WARMUP_S}s measure=${MEASURE_S}s reps=$REPS keepN=$KEEPA_N dtype=$DTYPE prefill_sm=16(PINNED)" | tee -a "$RESULT"

for D in "${SM_ARMS[@]}"; do
  cfg="${CFG[$D]}"
  # Ask the OS for a free port instead of hashing the job id: in 865006 the
  # M arm's D=44 hit "[Errno 98] address already in use" and lost the cell,
  # because a torn-down server's auxiliary ports had not been released yet.
  PORT=$(python3 -c "
import socket
s = socket.socket(); s.bind(('127.0.0.1', 0)); print(s.getsockname()[1]); s.close()")
  RUNID="s0dc_${ARM}_C${CTX}_P16D${D}_${SLURM_ARRAY_JOB_ID:-local}"
  SRVLOG="$HERE/${RUNID}_srv.log"
  export PDMUX_TELEMETRY_PATH="$HERE/${RUNID}_telemetry.jsonl"; : > "$PDMUX_TELEMETRY_PATH"
  export PDMUX_RUN_ID="$RUNID"; export PDMUX_WORKLOAD_ID="${ARM}_C${CTX}_P16D${D}"
  export PDMUX_R2_POLICY=fixed; export PDMUX_R2_FIXED_DSM=$D

  echo "=== ARM=$ARM prefill-SM=16(pinned) decode-SM=$D cfg=$(basename $cfg) port=$PORT ===" | tee -a "$RESULT"
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
    kill -0 $SRV 2>/dev/null || { echo "SERVER_DIED D=$D (check the decode_states guard in srv.log)" | tee -a "$RESULT"; break; }
    sleep 5
  done
  echo "boot_ok=$ok D=$D" | tee -a "$RESULT"

  if [ "$ok" = 1 ]; then
    for r in $(seq 1 $REPS); do
      # Sample SM clock during the run: arms differ in how many SMs are active
      # (32 vs 108), and idle-SM boost headroom would flatter the low-D arms,
      # i.e. bias toward UNDER-estimating decode-SM sensitivity (DESIGN.md §3).
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
        --keepalive-prefill $KEEPA_N --raw-out "$RAW" \
        --tag "${ARM}_C${CTX}_P16D${D}_r${r}" 2>>"$RESULT" | tee -a "$RESULT"
      CLIENT_RC=${PIPESTATUS[0]}
      kill $CLKPID 2>/dev/null; wait $CLKPID 2>/dev/null
      # Fail fast. Run 865006 spent 25 min per arm producing zero usable
      # samples because nothing checked the client's exit status (tee had
      # swallowed it) until the PIN_CHECK at the very end.
      if [ "$CLIENT_RC" != 0 ] && [ "$FIRST_MEASUREMENT" = 1 ]; then
        echo "ABORT: first measurement failed (client rc=$CLIENT_RC). Nothing" \
             "downstream can be valid; see $RESULT." | tee -a "$RESULT"
        kill $SRV 2>/dev/null; pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null
        exit 9
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

    # PIN VERIFICATION: the engine must actually have pinned decode to D.
    # Without this a silently mis-pinned arm looks like a flat result, which is
    # exactly the answer we are predicting -- so a failure here is disqualifying.
    if [ -f "$PDMUX_TELEMETRY_PATH" ]; then
      python3 - "$PDMUX_TELEMETRY_PATH" "$D" <<'PY' | tee -a "$RESULT"
import json,sys,collections
path,expect=sys.argv[1],int(sys.argv[2])
c=collections.Counter()
for ln in open(path):
    try:e=json.loads(ln)
    except Exception:continue
    if "target_decode_sms" in e: c[e.get("target_decode_sms")]+=1
hist=dict(c)
print("ACTUAL_target_decode_sms_hist:", hist or "NONE")
keys=set(hist)
ok = bool(hist) and ((keys=={expect}) or (max(hist,key=hist.get)==expect))
print(f"PIN_CHECK D={expect}: {'PASS' if ok else 'FAIL -> arm measured at wrong SM, INVALID'}")
PY
    fi
  else
    echo "BOOT_FAILED D=$D" | tee -a "$RESULT"
    tail -30 "$SRVLOG" | grep -viE "pynvml|Future" | tee -a "$RESULT"
  fi
  kill $SRV 2>/dev/null; pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null; sleep 3
done
echo "DONE ARM=$ARM CTX=$CTX" | tee -a "$RESULT"
