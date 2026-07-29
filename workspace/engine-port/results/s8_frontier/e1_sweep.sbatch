#!/bin/bash
#SBATCH --job-name=e1sweep
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --comment=pytorch
#SBATCH --chdir=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/s8_frontier
#SBATCH --output=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/s8_frontier/%x_%j.out
#SBATCH --error=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/s8_frontier/%x_%j.err
# ============================================================================
# E1 -- 8B FRONTIER: ITL(D) vs TTFT(108-D) under budget constraint
# prefill_SM + decode_SM == 108, D in {16,24,44,54,92}, 4 arms.
#
# Answers the "tension A" question left open by C2 (decode-SM lever exists,
# s8_scaleup FINDINGS_8B_2026-07-28.md) vs HE2 (dynamic split control did not
# beat best static at the 2.7B operating point, CONSENSUS.md): once prefill
# and decode compete for the SAME 108-SM budget, is there a D that beats
# best-static on CONJUNCTIVE goodput? See DESIGN.md for the full protocol,
# pre-registered gates, and decision rule -- do not edit the protocol here
# without updating that file too.
#
# Workload: ShareGPT, OPEN-LOOP (sglang.bench_serving --request-rate), rate
# alternating LO<->HI within each rep (mirrors slo_sched/sharegpt_vary_bench.sbatch
# and g2_0_raconf's change-trace pattern -- CLOSED-LOOP IS NOT USED HERE,
# unlike s8_scaleup/s8p_prefill, per methodology gate "closed-loop 금지" in
# this campaign's task spec). RATE_LO/RATE_HI MUST be chosen from
# e1_capacity_scan.sbatch output BEFORE running this -- there is no built-in
# default because capacity is arm- and cell-dependent (DESIGN.md sec 4.2).
#
# REPS reps per cell run SEQUENTIALLY INSIDE ONE SERVER BOOT (deviates from
# g2_0_raconf's "each rep a separate server boot" -- see DESIGN.md sec 6.3
# for why: 7-8B boot cost (~3-10 min observed, s8_scaleup sacct/srv.log
# timing) makes a fresh boot per rep infeasible within the project's time
# budget; this matches the s8_scaleup/s8p_prefill precedent this harness is
# built from, which also uses single-boot + REPS-loop). Each rep writes to
# its OWN jsonl files (never appended/overwritten across reps) so
# e1_analyze.py can compute rep-wise (not rep-pooled) statistics --
# methodology gate: rep pooling forbidden, n>=4 paired bootstrap.
#
# SEED POLICY (★★2026-07-29, coordinator directive, DESIGN.md sec 4.5):
# --seed=$((BASE_SEED + rep)) is passed to BOTH the LO and HI
# sglang.bench_serving invocations for a rep. rep VARIES the seed (so n>=4
# reps span n>=4 distinct arrival sequences + ShareGPT prompt samples, not 4
# replays of the SAME workload realization under a shared default seed=1 --
# that was the pre-2026-07-29 bug: every rep/cell/arm was bit-identical,
# so rep-level paired-bootstrap CI captured only engine noise, not workload
# variation, which could inflate false-positive "CI excludes 0" verdicts).
# cell/arm do NOT vary the seed for the SAME rep index -- this keeps
# cross-cell comparisons workload-MATCHED (paired), the precondition for the
# decision rule's paired bootstrap (sec 4.4).
#
# QOS is submit<=4/run<=2 per user; submit with disjoint ARM_IDS, e.g.:
#   sbatch --export=ALL,ARM_IDS="3",RATE_LO=...,RATE_HI=... e1_sweep.sbatch   # T8
#   sbatch --export=ALL,ARM_IDS="2",RATE_LO=...,RATE_HI=... e1_sweep.sbatch   # Hs8
# (stage 1; M8/Ha8 = stage 2, see DESIGN.md sec 3 for the staging rationale
# and why arms are not all submitted at once.)
#
# NOT a claim job. s0_deconfound/DESIGN.md sec 5 pre-registration still
# governs -- nothing here goes to canon before claims-auditor.
# ============================================================================
set -uo pipefail
module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0
export PS1="" PATH="$HOME/.local/bin:$PATH"
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate

ROOT=/scratch/ehmoon/whlee/prefill-layer-alloc
HERE=$ROOT/workspace/engine-port/results/s8_frontier
export HF_HOME=$ROOT/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="$ROOT/workspace/engine-port/triage/.triton_cache_e1_${SLURM_JOB_ID:-local}"
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
# Same unavoidable backend asymmetry as s8_scaleup/s8p_prefill: NemotronH
# banned from triton (server_args.py:1959), Zamba2-7B's cudagraph capture
# dies on flashinfer.
ARM_BACKENDS=(triton triton flashinfer triton)

SGPT="$ROOT/hf_cache/raw/ShareGPT_V3_unfiltered_cleaned_split.json"
CTX=4096; SGPT_CTXLEN="${SGPT_CTXLEN:-3600}"
CTXCAP=$(( CTX + 512 + 256 ))

# --- MUST be set from e1_capacity_scan.sbatch output (DESIGN.md sec 4). No
# silent default: an unset rate here is a bug, not a reasonable fallback,
# because capacity is arm-dependent (methodology gate #6: never guess the
# off-cliff rate).
if [ -z "${RATE_LO:-}" ] || [ -z "${RATE_HI:-}" ]; then
  echo "RATE_LO/RATE_HI must be set from e1_capacity_scan.sbatch results -- refusing to guess (gate #6)"
  exit 2
fi
NP="${NP:-150}"                    # prompts per phase per rep
REPS="${REPS:-4}"                  # methodology gate: n>=4
PIN_GATE="${PIN_GATE:-0.80}"        # min realized-pin frac among PREFILL-ACTIVE samples
                                    # (2026-07-28 revision -- e1_pin_check.py docstring;
                                    # NOT the retired engaged=OR population)
MIN_N_PREFILL_ACTIVE="${MIN_N_PREFILL_ACTIVE:-20}"  # min prefill-active sample count for
                                    # the pin gate to be informative (865832 smoke found
                                    # n=1 trivially satisfies a frac threshold otherwise)
# BASE_SEED: ★★NEW 2026-07-29 (coordinator directive, after 867008's
# arrival_rps diagnosis exposed that --seed was NEVER overridden, so every
# rep/cell/arm shared bit-identical workload realizations -- rep-level
# paired-bootstrap CI was measuring only engine noise, not workload
# variation, DESIGN.md sec 4.5). Seed for rep r = BASE_SEED + r, IDENTICAL
# across cells/arms for the SAME rep index (workload-matched pairing, the
# precondition for the paired bootstrap in the decision rule, DESIGN.md
# sec 4.4) but DIFFERENT across reps (so n>=4 reps span n>=4 distinct
# arrival sequences + ShareGPT prompt samples, not 4 replays of one).
BASE_SEED="${BASE_SEED:-100}"

if [ "${SMOKE_CELLS:-}" = "1" ]; then
  # Smoke-test subset (DESIGN.md sec 8): the two extremes only.
  CELLS="d16 d92"
else
  CELLS="${CELLS:-d16 d24 d44 d54 d92}"
fi

ARM_IDS="${ARM_IDS:-3 2}"          # default stage 1: T8, Hs8 (DESIGN.md sec 3)
JOBTAG="${SLURM_JOB_ID:-local}"

for AIDX in $ARM_IDS; do
ARM="${ARM_LABELS[$AIDX]}"
MODEL="${ARM_MODELS[$AIDX]}"
BACKEND="${ARM_BACKENDS[$AIDX]}"
RESULT="$HERE/e1_${ARM}_${JOBTAG}_result.txt"; : > "$RESULT"
FIRST_MEASUREMENT=1
echo "ARM=$ARM MODEL=$MODEL BACKEND=$BACKEND CELLS=[$CELLS] RATE_LO=$RATE_LO RATE_HI=$RATE_HI " \
     "NP=$NP reps=$REPS ctxcap=$CTXCAP" | tee -a "$RESULT"

for NAME in $CELLS; do
  CFG="$HERE/pdmux_e1_${NAME}.yml"
  DSM="${NAME#d}"
  [ -f "$CFG" ] || { echo "MISSING_CONFIG cell=$NAME cfg=$CFG" | tee -a "$RESULT"; continue; }
  PORT=$(python3 -c "
import socket
s = socket.socket(); s.bind(('127.0.0.1', 0)); print(s.getsockname()[1]); s.close()")
  RUNID="e1_${ARM}_${NAME}_${JOBTAG}"
  SRVLOG="$HERE/${RUNID}_srv.log"
  export PDMUX_TELEMETRY_PATH="$HERE/${RUNID}_telemetry.jsonl"; : > "$PDMUX_TELEMETRY_PATH"
  export PDMUX_RUN_ID="$RUNID"; export PDMUX_WORKLOAD_ID="${ARM}_${NAME}"
  export PDMUX_R2_POLICY=fixed; export PDMUX_R2_FIXED_DSM=$DSM

  echo "=== ARM=$ARM cell=$NAME target_decode_sm=$DSM cfg=$(basename $CFG) port=$PORT ===" | tee -a "$RESULT"
  python -m sglang.launch_server --model-path "$MODEL" --trust-remote-code --dtype bfloat16 \
    --attention-backend $BACKEND \
    --enable-pdmux --pdmux-config-path "$CFG" \
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

  ROUNDS_JSON="$HERE/${RUNID}_rounds.jsonl"; : > "$ROUNDS_JSON"
  if [ "$ok" = 1 ]; then
    CLK="$HERE/${RUNID}_clocks.csv"
    ( for _ in $(seq 1 400); do
        nvidia-smi --query-gpu=clocks.sm,power.draw,utilization.gpu \
          --format=csv,noheader,nounits >> "$CLK" 2>/dev/null; sleep 5
      done ) &
    CLKPID=$!
    for r in $(seq 1 $REPS); do
      SEED=$((BASE_SEED + r))   # ★2026-07-29: rep varies workload realization
                                 # (arrival sequence + ShareGPT sample), SAME
                                 # seed for this rep index across every cell/arm
                                 # in the campaign (workload-matched pairing).
      JLO="$HERE/${RUNID}_rep${r}_lo.jsonl"; : > "$JLO"
      JHI="$HERE/${RUNID}_rep${r}_hi.jsonl"; : > "$JHI"
      T0_LO=$(python3 -c 'import time; print(f"{time.monotonic():.6f}")')
      python -m sglang.bench_serving --backend sglang --model "$MODEL" --host 127.0.0.1 --port $PORT \
        --dataset-name sharegpt --dataset-path "$SGPT" --sharegpt-context-len $SGPT_CTXLEN \
        --num-prompts $NP --request-rate $RATE_LO --seed $SEED --output-details --output-file "$JLO" \
        2>>"$RESULT" | tee -a "$RESULT"
      LO_RC=${PIPESTATUS[0]}
      T1_LO=$(python3 -c 'import time; print(f"{time.monotonic():.6f}")')
      # drain/settle between phases (g2_0_decliff/hard pattern: exact
      # running==0/queued==0 scheduler line, or 2 consecutive stable-idle
      # polls if the log has simply stopped emitting)
      drain_ok=0; prev_cnt=-1; stable=0
      for di in $(seq 1 30); do
        cnt=$(grep -cE "Prefill batch|Decode batch" "$SRVLOG" 2>/dev/null || echo 0)
        qline=$(grep -E "Prefill batch|Decode batch" "$SRVLOG" | tail -1)
        running=$(echo "$qline" | grep -oE "#running-req: [0-9]+" | grep -oE "[0-9]+" | head -1)
        queued=$(echo "$qline" | grep -oE "#queue-req: [0-9]+" | grep -oE "[0-9]+" | head -1)
        if [ "${running:-1}" = "0" ] && [ "${queued:-1}" = "0" ]; then drain_ok=1; break; fi
        if [ "$cnt" = "$prev_cnt" ]; then
          stable=$((stable+1)); [ "$stable" -ge 2 ] && { drain_ok=2; break; }
        else
          stable=0
        fi
        prev_cnt=$cnt; sleep 1
      done
      sleep 3
      echo "DRAIN_STATUS rep=$r drain_ok=$drain_ok" | tee -a "$RESULT"
      T0_HI=$(python3 -c 'import time; print(f"{time.monotonic():.6f}")')
      python -m sglang.bench_serving --backend sglang --model "$MODEL" --host 127.0.0.1 --port $PORT \
        --dataset-name sharegpt --dataset-path "$SGPT" --sharegpt-context-len $SGPT_CTXLEN \
        --num-prompts $NP --request-rate $RATE_HI --seed $SEED --output-details --output-file "$JHI" \
        2>>"$RESULT" | tee -a "$RESULT"
      HI_RC=${PIPESTATUS[0]}
      T1_HI=$(python3 -c 'import time; print(f"{time.monotonic():.6f}")')
      python3 -c "
import json
print(json.dumps({'rep': $r, 'seed': $SEED, 'np': $NP,
                   'lo': {'t0': $T0_LO, 't1': $T1_LO, 'rate': $RATE_LO},
                   'hi': {'t0': $T0_HI, 't1': $T1_HI, 'rate': $RATE_HI}}))
" >> "$ROUNDS_JSON"
      # Fail fast: 865006 lesson (tee swallows exit status) -- check
      # PIPESTATUS on BOTH phases of the first rep.
      if { [ "$LO_RC" != 0 ] || [ "$HI_RC" != 0 ]; } && [ "$FIRST_MEASUREMENT" = 1 ]; then
        echo "ABORT arm=$ARM cell=$NAME: first rep failed (lo_rc=$LO_RC hi_rc=$HI_RC); " \
             "skipping the rest of this cell" | tee -a "$RESULT"
        kill $SRV 2>/dev/null; pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null
        break
      fi
      FIRST_MEASUREMENT=0
      echo "E1_PIN_CHECK_LO rep=$r:" | tee -a "$RESULT"
      python3 "$HERE/e1_pin_check.py" "$PDMUX_TELEMETRY_PATH" "$DSM" "$PIN_GATE" "$MIN_N_PREFILL_ACTIVE" \
        --t0 "$T0_LO" --t1 "$T1_LO" | tee -a "$RESULT"
      echo "E1_PIN_CHECK_HI rep=$r:" | tee -a "$RESULT"
      python3 "$HERE/e1_pin_check.py" "$PDMUX_TELEMETRY_PATH" "$DSM" "$PIN_GATE" "$MIN_N_PREFILL_ACTIVE" \
        --t0 "$T0_HI" --t1 "$T1_HI" | tee -a "$RESULT"
    done
    kill $CLKPID 2>/dev/null; wait $CLKPID 2>/dev/null
    python3 - "$CLK" <<'PY' | tee -a "$RESULT"
import statistics, sys
try:
    rows = [l.split(",") for l in open(sys.argv[1]) if l.strip()]
    sm = [float(r[0]) for r in rows]
    print(f"CLOCKS sm_mhz_mean={statistics.fmean(sm):.0f} sm_mhz_min={min(sm):.0f} n={len(sm)}")
except Exception as exc:
    print(f"CLOCKS unavailable: {exc}")
PY
    echo "E1_PIN_CHECK_CELL (whole telemetry file, informational):" | tee -a "$RESULT"
    python3 "$HERE/e1_pin_check.py" "$PDMUX_TELEMETRY_PATH" "$DSM" "$PIN_GATE" "$MIN_N_PREFILL_ACTIVE" | tee -a "$RESULT"
  else
    echo "BOOT_FAILED cell=$NAME" | tee -a "$RESULT"
    grep -viE "pynvml|Future" "$SRVLOG" | tail -25 | tee -a "$RESULT"
  fi
  kill $SRV 2>/dev/null; pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null; sleep 5
done
echo "SWEEP_DONE ARM=$ARM CELLS=[$CELLS]" | tee -a "$RESULT"
done
rm -rf "$TRITON_CACHE_DIR"
