#!/bin/bash
#SBATCH --job-name=m3r2pin
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --comment="field=efficientai;appl=pytorch"
#SBATCH --chdir=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/s8_frontier
#SBATCH --output=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/s8_frontier/%x_%j.out
#SBATCH --error=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/s8_frontier/%x_%j.err
# ============================================================================
# M3R rev2 -- PARTITION REALIZABILITY, WITH AN INSTRUMENT THAT CAN SEE IT.
# Pre-registered in DESIGN.md sec 4.3.8(g).  DIAGNOSTIC ONLY: no SLO, no
# goodput, no decision rule, and -- because trace-force is ON here -- NO ITL
# NUMBER FROM THIS JOB MAY EVER BE QUOTED.
#
# ★★WHY rev1 (job 872236) COULD NOT ANSWER, AND WHY I WAS WRONG BEFORE IT.
#
# Commit 7c6e6f2 removed the separate trace-force-ON pin run that sec 4.7.1 had
# required, arguing "the pin gate does not need PDMUX_TRACE_FORCE_PREFILL --
# the capacity scans passed without it".  That argument was wrong.  The
# capacity scans pooled EIGHT rates into one telemetry file; a single-rate
# probe does not.  Measured on 872236's own telemetry with trace-force OFF:
#
#     cell                 total snapshots   prefill-active
#     Ha8 d16 rate 2            10,558              2
#     T8  d16 rate 2            10,993              8
#     Ha8 d54 rate 2             9,906             16
#     T8  d54 rate 2            10,921             28
#
# The gate that voided M3 -- and the gate that then "passed" Ha8 d16 at
# lower95=1.000 in rev1 -- was computed from 2 to 28 samples.  Neither job
# measured realizability; the instrument was never adequate at a single rate.
# sec 4.7.1's original requirement was right and is restored here.
#
# The observer effect is not a reason to avoid trace-force in THIS job: 867298
# measured it on itl_p95 (+2.0% [+0.78, +3.21] at d92), and this job reports no
# ITL at all.  Trace-force belongs exactly here and nowhere near M3.
#
# ★rev1's OTHER defect: SEED AND BOOT WERE CONFOUNDED.  M3 ran seed == block ==
# boot, so its "seed 6,7,8 failed" cannot distinguish a workload effect from a
# server-state effect; rev1 then used a SINGLE seed (1) -- which had PASSED in
# M3 for both arms -- and so sampled none of the failing region while its
# sbatch header asserted "this measures an engine behaviour, not a workload
# effect".  M3's own per-block numbers refute that assertion.  rev2 CROSSES
# seeds with boots: the same SEEDS run inside every boot, so the two sources
# are separable for the first time.
#
# SEED CHOICE IS TARGETED, NOT RANDOM.  SEEDS spans M3's observed pass (1) and
# fail (6,7,8) blocks for Ha8 d16.  Consequence stated in advance: rev2 can say
# whether the variation is seed-indexed or boot-indexed and whether the
# failures reproduce, but it CANNOT estimate an unconditional failure rate over
# workloads.  Do not compute one from it.
#
# Usage:   sbatch m3r2_pinforce.sbatch
# Analyze: ./m3r2_analyze.sh <job_id>
# ============================================================================
set -uo pipefail
module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0
export PS1="" PATH="$HOME/.local/bin:$PATH"
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate

ROOT=/scratch/ehmoon/whlee/prefill-layer-alloc
HERE=$ROOT/workspace/engine-port/results/s8_frontier
export HF_HOME=$ROOT/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="$ROOT/workspace/engine-port/triage/.triton_cache_m3r2_${SLURM_JOB_ID:-local}"
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

"$ROOT/workspace/engine-port/scripts/bootstrap/sync_engine_tree.sh" \
  "$HERE/runtime_source_manifest.sha256" || { echo "SYNC_ENGINE_TREE_FAILED"; exit 3; }

ARM_LABELS=(T8 Ha8)
ARM_MODELS=("Qwen/Qwen2.5-7B" "Zyphra/Zamba2-7B-Instruct")
ARM_BACKENDS=(triton triton)

SGPT="$ROOT/hf_cache/raw/ShareGPT_V3_unfiltered_cleaned_split.json"
CTX=4096; SGPT_CTXLEN="${SGPT_CTXLEN:-3600}"
CTXCAP=$(( CTX + 512 + 256 ))

# (cell,rate) combinations.  d16 = the cell M3 voided, probed at the operating
# rate AND at rate 6 to keep rev1's load hypothesis alive under a working
# instrument.  d24 = also voided on T8.  d54 = the control that never failed.
COMBOS="${COMBOS:-d16:2 d16:6 d24:2 d54:2}"
BOOTS="${BOOTS:-3}"                   # boots per (arm, cell, rate)
SEEDS="${SEEDS:-1 6 7 8}"             # SAME seeds inside every boot -> crossed
ARRIVAL_S="${ARRIVAL_S:-40}"
NP_MIN="${NP_MIN:-120}"
WARMUP_S="${WARMUP_S:-3}"
CAP="${CAP:-48}"
MAMBA_POOL="${MAMBA_POOL:-$CAP}"
PROBE_TIMEOUT_S="${PROBE_TIMEOUT_S:-240}"   # 24 boots + 96 probes.
                                      # Expected ~2.5h (probes run 60-80s).
                                      # The at-risk probes are d16:6, which is
                                      # above Ha8 d16's knee (5.60) on purpose;
                                      # if all 24 of those cap out the job is
                                      # ~4.3h, inside the 6h wall.
MIN_PA_SNAPSHOTS="${MIN_PA_SNAPSHOTS:-200}"   # instrument sanity gate: if
                                      # trace-force ON still leaves fewer than
                                      # this many prefill-active snapshots, the
                                      # instrument is STILL inadequate and the
                                      # cell is reported as UNMEASURABLE rather
                                      # than passed or failed.

JOBTAG="${SLURM_JOB_ID:-local}"
echo "=== M3R rev2 pin-force realizability (DESIGN.md sec 4.3.8(g)) job=$JOBTAG ==="
echo "COMBOS=[$COMBOS] BOOTS=$BOOTS SEEDS=[$SEEDS] (crossed: same seeds in every boot)"
echo "TRACE_FORCE=1  -- pin measurement only, NO ITL from this job, ever"
echo "CAP=$CAP MAMBA_POOL=$MAMBA_POOL MIN_PA_SNAPSHOTS=$MIN_PA_SNAPSHOTS"

for AIDX in 0 1; do
ARM="${ARM_LABELS[$AIDX]}"
MODEL="${ARM_MODELS[$AIDX]}"
BACKEND="${ARM_BACKENDS[$AIDX]}"
RESULT="$HERE/m3r2_${ARM}_${JOBTAG}_result.txt"; : > "$RESULT"
echo "ARM=$ARM MODEL=$MODEL BACKEND=$BACKEND" | tee -a "$RESULT"

for COMBO in $COMBOS; do
  NAME="${COMBO%%:*}"; RATE="${COMBO##*:}"
  CFG="$HERE/pdmux_e1_${NAME}.yml"
  DSM="${NAME#d}"
  [ -f "$CFG" ] || { echo "MISSING_CONFIG cell=$NAME" | tee -a "$RESULT"; continue; }
  NP=$(python3 -c "import math; print(max($NP_MIN, math.ceil($RATE*$ARRIVAL_S)))")

  for B in $(seq 1 $BOOTS); do
    PORT=$(python3 -c "
import socket
s = socket.socket(); s.bind(('127.0.0.1', 0)); print(s.getsockname()[1]); s.close()")
    RUNID="m3r2_${ARM}_${NAME}_r${RATE}_b${B}_${JOBTAG}"
    SRVLOG="$HERE/${RUNID}_srv.log"
    export PDMUX_TELEMETRY_PATH="$HERE/${RUNID}_telemetry.jsonl"; : > "$PDMUX_TELEMETRY_PATH"
    export PDMUX_RUN_ID="$RUNID"; export PDMUX_WORKLOAD_ID="m3r2_${ARM}_${NAME}_r${RATE}"
    export PDMUX_R2_POLICY=fixed; export PDMUX_R2_FIXED_DSM=$DSM
    # ★THE point of rev2.  sec 4.7.1 restored: pin measurement gets the dense
    # sampling, and it is quarantined in a job that emits no latency claim.
    export PDMUX_TRACE_FORCE_PREFILL=1

    echo "=== ARM=$ARM cell=$NAME rate=$RATE boot=$B NP=$NP D=$DSM port=$PORT ===" | tee -a "$RESULT"
    # ONE telemetry file per BOOT, shared by every seed in it -- that sharing
    # is exactly what makes seed separable from boot, and it is why each probe
    # is bracketed by --t0/--t1 below instead of getting its own file.
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
      kill -0 $SRV 2>/dev/null || { echo "SERVER_DIED cell=$NAME rate=$RATE boot=$B" | tee -a "$RESULT"; break; }
      sleep 5
    done
    echo "boot_ok=$ok cell=$NAME rate=$RATE boot=$B" | tee -a "$RESULT"

    if [ "$ok" = 1 ]; then
      for SEED in $SEEDS; do
        # NOTE: PDMUX_TELEMETRY_PATH is read by the server at startup, so all
        # seeds of one boot share one telemetry file.  Windows are therefore
        # cut per seed below with --t0/--t1 taken from the probe's own wall
        # clock, which is why each probe is bracketed by explicit timestamps.
        JOUT="$HERE/${RUNID}_s${SEED}.jsonl"; : > "$JOUT"
        # ★match telemetry.py:23 exactly -- it stamps with time.perf_counter().
        # (Both are clock_gettime(CLOCK_MONOTONIC) here, verified 2026-08-03,
        # but the existing traceforce_gate.sbatch paired monotonic() against
        # perf_counter() and its pin checks all crashed, so that pairing was
        # never validated end-to-end. Do not reintroduce the mismatch.)
        T0=$(python3 -c 'import time; print(f"{time.perf_counter():.6f}")')
        timeout "$PROBE_TIMEOUT_S" python -m sglang.bench_serving --backend sglang \
          --model "$MODEL" --host 127.0.0.1 --port $PORT \
          --dataset-name sharegpt --dataset-path "$SGPT" --sharegpt-context-len $SGPT_CTXLEN \
          --num-prompts $NP --request-rate $RATE --seed $SEED --output-details \
          --output-file "$JOUT" > "$HERE/${RUNID}_s${SEED}_bench.log" 2>&1
        RC=$?
        T1=$(python3 -c 'import time; print(f"{time.perf_counter():.6f}")')
        [ "$RC" = 124 ] && echo "PROBE_TIMEOUT cell=$NAME rate=$RATE boot=$B seed=$SEED" | tee -a "$RESULT"
        [ "$RC" != 0 ] && [ "$RC" != 124 ] && echo "PROBE_FAILED cell=$NAME rate=$RATE boot=$B seed=$SEED rc=$RC" | tee -a "$RESULT"
        echo "M3R2_PIN arm=$ARM cell=$NAME rate=$RATE boot=$B seed=$SEED:" | tee -a "$RESULT"
        PINOUT=$(python3 "$HERE/e1_pin_check.py" "$PDMUX_TELEMETRY_PATH" "$DSM" 0.80 \
          --t0 "$T0" --t1 "$T1" 2>&1)
        echo "$PINOUT" | grep -E "E1_PIN_GATE|E1_REALIZED_hist|E1_CONCURRENCY_DIAGNOSTIC" | tee -a "$RESULT"
        # ★instrument sanity gate.  The whole point of rev2 is that the sample
        # size was the binding constraint, so the sample size must itself be
        # gated -- otherwise rev2 could repeat rev1's mistake with a bigger
        # number attached.  A cell below the floor is UNMEASURABLE: neither
        # passed nor failed.
        NPA=$(echo "$PINOUT" | sed -n 's/.*n_pa_snapshots=\([0-9]*\).*/\1/p' | head -1)
        if [ -n "$NPA" ] && [ "$NPA" -lt "$MIN_PA_SNAPSHOTS" ]; then
          echo "M3R2_UNMEASURABLE arm=$ARM cell=$NAME rate=$RATE boot=$B seed=$SEED "\
               "n_pa_snapshots=$NPA < $MIN_PA_SNAPSHOTS -- instrument still inadequate" | tee -a "$RESULT"
        fi
        sleep 3
      done
    fi
    kill $SRV 2>/dev/null; pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null
    wait $SRV 2>/dev/null
    sleep 8
  done
done
done

echo "=== M3R rev2 done job=$JOBTAG.  Analyze: ./m3r2_analyze.sh $JOBTAG ==="
echo "=== DIAGNOSTIC ONLY. No ITL from this job. Not citable before claims-auditor. ==="
