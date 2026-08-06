#!/bin/bash
#SBATCH --job-name=gate1
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --comment=pytorch
#SBATCH --chdir=/scratch/ehmoon/whlee/prefill-layer-alloc
#SBATCH --output=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/p1_gates/gate1/gate1_%j.out
#SBATCH --error=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/p1_gates/gate1/gate1_%j.err
# ============================================================================
# Gate 1 -- telemetry replica of job 873944 (Zamba2-2.7B, agnostic/pdmux v1,
# cudagraph-ON), agnostic arm ONLY, rate {2,3}, n=1. Adds PDMUX_TELEMETRY_PATH
# (+ densified sampling cadence) to answer ONE pre-registered question: does
# the realized (prefill_sms, decode_sms) split, conditional on decode-busy AND
# prefill in-flight, land in {(74,34),(54,54)} >=90% of the time (time-
# weighted), with negative controls (decode-idle / prefill-idle) matching
# their code-expected fallback partitions? See PREREG_GATE1_2026-08-06.md,
# written and fixed BEFORE this job is submitted (methodology gate #8: do not
# edit the decision rule after seeing results).
#
# THIS JOB MAKES NO PERFORMANCE CLAIM. No TTFT/TPOT/ITL/goodput number from
# its bench_serving JSONL output may ever be quoted (n=1, diagnostic only --
# same scope discipline as results/s8_frontier/m3r2_pinforce.sbatch).
#
# Usage:   sbatch gate1_run.sbatch
# Analyze: python3 gate1_analyze.py gate1_telemetry_<jobid>.jsonl \
#            --window 2 <T0_R2> <T1_R2> --window 3 <T0_R3> <T1_R3>
#          (this script also runs that automatically at the end and writes
#          gate1_result_<jobid>.txt)
# ============================================================================
set -uo pipefail
module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0
export PS1="" PATH="$HOME/.local/bin:$PATH"
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate
cd /scratch/ehmoon/whlee/prefill-layer-alloc

OUT=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/p1_gates/gate1
JOBTAG="${SLURM_JOB_ID:-local}"
export HF_HOME=/scratch/ehmoon/whlee/prefill-layer-alloc/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="$OUT/.triton_cache_${JOBTAG}"
mkdir -p "$TRITON_CACHE_DIR"

MODEL="Zyphra/Zamba2-2.7B"
CTX=4096
BACKEND=triton
PDMUX_CFG="$OUT/pdmux_a100_smoke.yml"

echo "############# Gate 1 job=$JOBTAG (PREREG_GATE1_2026-08-06.md) #############"

echo "=== env sync + manifest drift check ==="
MANIFEST="$OUT/runtime_source_manifest_gate1_${JOBTAG}.sha256"
workspace/engine-port/scripts/bootstrap/sync_engine_tree.sh "$MANIFEST"
echo "manifest -> $MANIFEST"
REF="$OUT/reference_manifest_873944.sha256"
if [ -f "$REF" ]; then
  if diff <(sort "$MANIFEST") <(sort "$REF") > "$OUT/manifest_diff_${JOBTAG}.txt"; then
    echo "MANIFEST_CHECK: MATCH (identical to job 873944's manifest, no source drift)"
  else
    echo "MANIFEST_CHECK: MISMATCH vs job 873944 -- see $OUT/manifest_diff_${JOBTAG}.txt" >&2
    echo "MANIFEST_MISMATCH_ABORT" >&2
    cat "$OUT/manifest_diff_${JOBTAG}.txt" >&2
    exit 3
  fi
else
  echo "MANIFEST_CHECK: reference manifest not found at $REF -- cannot verify drift" >&2
fi

PORT=$((33000 + SLURM_JOB_ID % 400))

echo "=== boot agnostic server (cudagraph ON, PDMUX_TELEMETRY_PATH set) ==="
SRVLOG="$OUT/gate1_srv_${JOBTAG}.log"
export PDMUX_TELEMETRY_PATH="$OUT/gate1_telemetry_${JOBTAG}.jsonl"; : > "$PDMUX_TELEMETRY_PATH"
export PDMUX_RUN_ID="gate1_${JOBTAG}"
export PDMUX_WORKLOAD_ID="gate1_zamba2-27b_agnostic"
# Densified sampling cadence (see PREREG "케이던스" section): forced emission
# on every prefill-in-flight sync (non-biasing for time-weighted stats, the
# ONLY stats this gate scores on). PDMUX_DUAL_WORKER_TRACE_EVERY is left at
# its DEFAULT (32, matching results/s8_frontier/m3r2_pinforce.sbatch's proven-
# safe precedent) -- NOT lowered to 8.
#   ★POST-MORTEM (job 874465, first attempt, 2026-08-06): TRACE_EVERY=8 +
#   FORCE_PREFILL=1 caused the AsyncJsonlTelemetry writer thread (single
#   thread, line-buffered JSONL writes to Lustre /scratch) to fall behind an
#   idle-spin-dominated enqueue rate of ~900-1300 events/s (the scheduler's
#   event loop free-spins very fast while both prefill and decode are empty,
#   and every 8th spin got sampled) -- the bounded producer queue
#   (`AsyncJsonlTelemetry.__init__`'s `max_events=65536`, non-blocking
#   `put_nowait`, silently DROPS on full rather than blocking the scheduler)
#   filled ~93s into the run and every subsequent runtime_snapshot was
#   dropped for the REST of the job -- all of rate=3's window (n_rows=0).
#   The server itself never stalled (requests completed normally,
#   SIGTERM only at true end) -- this was a pure telemetry-pipeline
#   overload, not a scheduling/partition bug. Root-caused via
#   `gate1_telemetry_874465.jsonl`'s last line's timestamp (~72s before the
#   server's own SIGTERM) and the file's total enqueue-rate estimate. Fix:
#   revert to the proven-safe default cadence; FORCE_PREFILL alone already
#   gives dense, non-dropped coverage of population A (the primary
#   population) because it fires on every prefill-in-flight sync
#   independent of this knob.
export PDMUX_TRACE_FORCE_PREFILL=1
# Deliberately NOT set: PDMUX_DUAL_WORKER, PDMUX_TRUE_DUAL_WORKER (those are
# the R1-observer / true-dual-worker architectures, not what agnostic v1 /
# job 873944 ran -- see CLAUDE.md and PREREG).

python -m sglang.launch_server --model-path "$MODEL" --trust-remote-code --dtype bfloat16 \
  --attention-backend "$BACKEND" \
  --disable-radix-cache --mem-fraction-static 0.82 --max-running-requests 48 --context-length "$CTX" \
  --enable-pdmux --pdmux-config-path "$PDMUX_CFG" --chunked-prefill-size -1 --disable-overlap-schedule \
  --host 127.0.0.1 --port "$PORT" > "$SRVLOG" 2>&1 &
SRV=$!

ok=0
for i in $(seq 1 170); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health 2>/dev/null||true)" = 200 ] && { ok=1; break; }
  kill -0 "$SRV" 2>/dev/null || { echo "SERVER DIED (pid $SRV)" >&2; break; }
  sleep 5
done
echo "boot_ok=$ok"
echo "--- capture evidence ---"
grep -iE "captur|cuda graph|cudagraph" "$SRVLOG" | grep -viE "Future|pynvml" | head -8
echo "--- error evidence ---"
grep -iE "error|traceback|not support|illegal|assert" "$SRVLOG" | grep -viE "Future|pynvml" | head -8

if [ "$ok" != 1 ]; then
  echo "GATE1_BOOT_FAILED" | tee "$OUT/gate1_result_${JOBTAG}.txt"
  tail -40 "$SRVLOG" | grep -viE "pynvml|Future"
  exit 1
fi

echo "=== correctness gate ==="
RESP=$(curl -s http://127.0.0.1:$PORT/generate -H 'Content-Type: application/json' \
  -d '{"text":"The capital of France is","sampling_params":{"max_new_tokens":8,"temperature":0}}')
echo "SANITY_RAW: $RESP"
if echo "$RESP" | grep -q "Paris"; then
  echo "CORRECTNESS=PASS"
else
  echo "CORRECTNESS=FAIL" | tee "$OUT/gate1_result_${JOBTAG}.txt"
  kill "$SRV" 2>/dev/null; pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null
  exit 1
fi

echo "############# measurement: rate 2, rate 3 (n=1, seed convention matches 873944 rep1) #############"
WINDOW_ARGS=()
for R in 2 3; do
  SEED=$((9000 + 100*1 + R))
  JSONL="$OUT/gate1_r${R}_${JOBTAG}.jsonl"; : > "$JSONL"
  echo "----- rate=$R seed=$SEED -----"
  T0=$(python3 -c 'import time; print(f"{time.perf_counter():.6f}")')
  python -m sglang.bench_serving --backend sglang --model "$MODEL" --host 127.0.0.1 --port "$PORT" \
    --dataset-name random-ids --random-input-len 2000 --random-output-len 96 --random-range-ratio 1.0 \
    --num-prompts 120 --request-rate "$R" --seed "$SEED" --warmup-requests 8 \
    --output-details --output-file "$JSONL" --tag "gate1_r${R}" \
    | grep -E "Successful requests|Request throughput|Median TTFT|Median TPOT"
  T1=$(python3 -c 'import time; print(f"{time.perf_counter():.6f}")')
  echo "window rate=$R t0=$T0 t1=$T1"
  WINDOW_ARGS+=(--window "$R" "$T0" "$T1")
done
echo "############# measurement done #############"

kill "$SRV" 2>/dev/null
# Defense-in-depth after the 874465 post-mortem above: give the telemetry
# writer thread time to drain any remaining queued events (SIGTERM ->
# graceful shutdown path) before the hard SIGKILL truncates it.
sleep 5
pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null
sleep 3

echo "############# GATE 1 ANALYSIS (per PREREG_GATE1_2026-08-06.md) #############"
python3 "$OUT/gate1_analyze.py" "$PDMUX_TELEMETRY_PATH" "${WINDOW_ARGS[@]}" \
  | tee "$OUT/gate1_result_${JOBTAG}.txt"

echo "DONE_GATE1_${JOBTAG}"
