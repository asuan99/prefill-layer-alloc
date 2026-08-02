#!/bin/bash
#SBATCH --job-name=m3realz
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=05:00:00
#SBATCH --comment=pytorch
#SBATCH --chdir=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/s8_frontier
#SBATCH --output=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/s8_frontier/%x_%j.out
#SBATCH --error=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/s8_frontier/%x_%j.err
# ============================================================================
# M3R -- PARTITION REALIZABILITY SWEEP.  Pre-registered in DESIGN.md sec 4.3.8(f).
#
# ★DIAGNOSTIC ONLY.  No SLO, no goodput, no decision rule, no A/B claim.  The
# single quantity of interest is: OF THE TIME PREFILL IS ACTUALLY RUNNING, what
# fraction executes at the cell's TARGET SM partition -- as a function of
# (arm, D, rate)?  Nothing here may be turned into a policy comparison; if a
# policy question arises from it, it needs its own pre-registration.
#
# WHY.  Job 872077 (M3) executed cleanly and still produced NO VERDICT: the
# sec 4.3.8(e) pin gate voided 19/64 cell-blocks, g's numerator among them
# (T8 d16 2/8 PASS, Ha8 d16 5/8; d54 8/8 both arms).  15 of 19 failures had
# pin_frac < 0.90 -- T8 d16 down to 0.61 -- so 20-40% of prefill-active time
# ran at P108 (no split) instead of the target P92.  That is not a measurement
# artifact to be averaged away; it says the LABEL may not be the REALIZED
# allocation, which is precisely the error class that invalidated Stage 0
# (D108 anchor was actually 16 SM, CONSENSUS sec 1-21).
#
# So before any further policy experiment on this grid: is a low-D cell
# realizable ANYWHERE in (D, rate)?  The failures got worse the LARGER the
# prefill partition and worse on the FASTER-prefill arm, which is the direction
# of CONSENSUS sec 1-22's decode-empty auto-revert.  If that is the mechanism,
# more offered load (more prefill work, fuller decode queue) should raise
# pin_frac -- a directional prediction this sweep can falsify.
#
# DESIGN NOTES
#   - One boot per (arm, cell, rate).  Rate cannot share a boot with another
#     rate: PDMUX telemetry is one file per server process, and mixing rates in
#     one file would force window arguments and reconstruction -- the exact
#     complexity that made the M3 pin check clean (one probe per file).
#   - Rates deliberately RUN PAST the knee.  This is legitimate here and would
#     not be in a policy job: an on-cliff probe still measures which partition
#     the engine ran, and the knees themselves (sec 4.3.7) are only about where
#     latency claims may be made.  Each probe is hard-capped by `timeout` so a
#     queued-up cell cannot eat the job.
#   - Same cap/pool/backends/context as M3 so the numbers are comparable to
#     872077 cell-for-cell.
#   - Single seed per cell: this measures an engine behaviour, not a workload
#     effect, and the replication that matters here is across RATE, not across
#     prompt draws.
#
# Usage:  sbatch m3r_realizability.sbatch
# Analyze: ./m3r_analyze.sh <job_id>
# ============================================================================
set -uo pipefail
module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0
export PS1="" PATH="$HOME/.local/bin:$PATH"
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate

ROOT=/scratch/ehmoon/whlee/prefill-layer-alloc
HERE=$ROOT/workspace/engine-port/results/s8_frontier
export HF_HOME=$ROOT/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="$ROOT/workspace/engine-port/triage/.triton_cache_m3r_${SLURM_JOB_ID:-local}"
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

"$ROOT/workspace/engine-port/scripts/bootstrap/sync_engine_tree.sh" \
  "$HERE/runtime_source_manifest.sha256" || { echo "SYNC_ENGINE_TREE_FAILED"; exit 3; }

ARM_LABELS=(T8 Ha8)
ARM_MODELS=("Qwen/Qwen2.5-7B" "Zyphra/Zamba2-7B-Instruct")
ARM_BACKENDS=(triton triton)

SGPT="$ROOT/hf_cache/raw/ShareGPT_V3_unfiltered_cleaned_split.json"
CTX=4096; SGPT_CTXLEN="${SGPT_CTXLEN:-3600}"
CTXCAP=$(( CTX + 512 + 256 ))

CELLS="${CELLS:-d16 d24 d44 d54}"     # same grid as M3
RATES="${RATES:-2 3 4 6}"             # 2 = M3's operating point; the rest probe
                                      # whether more offered load restores the
                                      # partition.  Above several knees on
                                      # purpose -- see DESIGN NOTES.
ARRIVAL_S="${ARRIVAL_S:-40}"          # target arrival window per probe
NP_MIN="${NP_MIN:-120}"               # floor so low rates still accumulate
                                      # prefill episodes (872077 got only 6-34)
SEED="${SEED:-1}"
WARMUP_S="${WARMUP_S:-3}"
CAP="${CAP:-48}"
MAMBA_POOL="${MAMBA_POOL:-$CAP}"
PROBE_TIMEOUT_S="${PROBE_TIMEOUT_S:-300}"   # worst case 32x300s = 2.7h;
                                      # with 32 boots (~0.7h) that fits the
                                      # 5h wall with margin.  Expected is far
                                      # less: arrival is 40s and only on-cliff
                                      # cells (e.g. Ha8 d16 at rate 6, knee
                                      # 5.60) should ever hit the cap.

JOBTAG="${SLURM_JOB_ID:-local}"
echo "=== M3R realizability (DESIGN.md sec 4.3.8(f)) job=$JOBTAG ==="
echo "CELLS=[$CELLS] RATES=[$RATES] ARRIVAL_S=$ARRIVAL_S NP_MIN=$NP_MIN SEED=$SEED"
echo "CAP=$CAP MAMBA_POOL=$MAMBA_POOL timeout=${PROBE_TIMEOUT_S}s  -- DIAGNOSTIC ONLY"

for AIDX in 0 1; do
ARM="${ARM_LABELS[$AIDX]}"
MODEL="${ARM_MODELS[$AIDX]}"
BACKEND="${ARM_BACKENDS[$AIDX]}"
RESULT="$HERE/m3r_${ARM}_${JOBTAG}_result.txt"; : > "$RESULT"
echo "ARM=$ARM MODEL=$MODEL BACKEND=$BACKEND" | tee -a "$RESULT"

for NAME in $CELLS; do
  CFG="$HERE/pdmux_e1_${NAME}.yml"
  DSM="${NAME#d}"
  [ -f "$CFG" ] || { echo "MISSING_CONFIG cell=$NAME" | tee -a "$RESULT"; continue; }

  for RATE in $RATES; do
    NP=$(python3 -c "import math; print(max($NP_MIN, math.ceil($RATE*$ARRIVAL_S)))")
    PORT=$(python3 -c "
import socket
s = socket.socket(); s.bind(('127.0.0.1', 0)); print(s.getsockname()[1]); s.close()")
    RUNID="m3r_${ARM}_${NAME}_r${RATE}_${JOBTAG}"
    SRVLOG="$HERE/${RUNID}_srv.log"
    export PDMUX_TELEMETRY_PATH="$HERE/${RUNID}_telemetry.jsonl"; : > "$PDMUX_TELEMETRY_PATH"
    export PDMUX_RUN_ID="$RUNID"; export PDMUX_WORKLOAD_ID="m3r_${ARM}_${NAME}_r${RATE}"
    export PDMUX_R2_POLICY=fixed; export PDMUX_R2_FIXED_DSM=$DSM
    export PDMUX_TRACE_FORCE_PREFILL=0   # sec 4.7.1: the pin gate does not need
                                         # it, and forcing it perturbs the very
                                         # iterations being counted

    echo "=== ARM=$ARM cell=$NAME rate=$RATE NP=$NP target_decode_sm=$DSM port=$PORT ===" | tee -a "$RESULT"
    python -m sglang.launch_server --model-path "$MODEL" --trust-remote-code --dtype bfloat16 \
      --attention-backend $BACKEND \
      --enable-pdmux --pdmux-config-path "$CFG" \
      --disable-piecewise-cuda-graph \
      --disable-overlap-schedule --chunked-prefill-size -1 --disable-radix-cache \
      --mem-fraction-static 0.80 \
      --max-running-requests $CAP --max-mamba-cache-size $MAMBA_POOL \
      --context-length $CTXCAP \
      --host 127.0.0.1 --port $PORT > "$SRVLOG" 2>&1 &
    SRV=$!
    ok=0
    for i in $(seq 1 240); do
      [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health 2>/dev/null||true)" = 200 ] && { ok=1; break; }
      kill -0 $SRV 2>/dev/null || { echo "SERVER_DIED cell=$NAME rate=$RATE" | tee -a "$RESULT"; break; }
      sleep 5
    done
    echo "boot_ok=$ok cell=$NAME rate=$RATE" | tee -a "$RESULT"

    if [ "$ok" = 1 ]; then
      JOUT="$HERE/${RUNID}.jsonl"; : > "$JOUT"
      timeout "$PROBE_TIMEOUT_S" python -m sglang.bench_serving --backend sglang \
        --model "$MODEL" --host 127.0.0.1 --port $PORT \
        --dataset-name sharegpt --dataset-path "$SGPT" --sharegpt-context-len $SGPT_CTXLEN \
        --num-prompts $NP --request-rate $RATE --seed $SEED --output-details \
        --output-file "$JOUT" > "$HERE/${RUNID}_bench.log" 2>&1
      RC=$?
      [ "$RC" = 124 ] && echo "PROBE_TIMEOUT cell=$NAME rate=$RATE (${PROBE_TIMEOUT_S}s) -- partial telemetry still usable" | tee -a "$RESULT"
      [ "$RC" != 0 ] && [ "$RC" != 124 ] && echo "PROBE_FAILED cell=$NAME rate=$RATE rc=$RC" | tee -a "$RESULT"

      # THE measurement: time-weighted realized-partition histogram.
      echo "M3R_PIN arm=$ARM cell=$NAME rate=$RATE:" | tee -a "$RESULT"
      python3 "$HERE/e1_pin_check.py" "$PDMUX_TELEMETRY_PATH" "$DSM" 0.80 2>&1 \
        | grep -E "E1_PIN_GATE|E1_REALIZED_hist|E1_CONCURRENCY_DIAGNOSTIC" | tee -a "$RESULT"
    fi
    kill $SRV 2>/dev/null; pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null
    wait $SRV 2>/dev/null
    sleep 8
  done
done
done

echo "=== M3R done job=$JOBTAG.  Analyze: ./m3r_analyze.sh $JOBTAG ==="
echo "=== DIAGNOSTIC ONLY -- not a policy comparison, not citable before claims-auditor. ==="
