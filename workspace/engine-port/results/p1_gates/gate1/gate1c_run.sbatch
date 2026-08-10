#!/bin/bash
#SBATCH --job-name=gate1c
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --comment=pytorch
#SBATCH --chdir=/scratch/ehmoon/whlee/prefill-layer-alloc
#SBATCH --output=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/p1_gates/gate1/gate1c_%j.out
#SBATCH --error=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/p1_gates/gate1/gate1c_%j.err
# ============================================================================
# G1-c -- Granite realized-partition observation, replicating job 873945
# (P1 operating-point contrast, Granite arm) the way G1-b (job 875293)
# replicated 873944 (Zamba2 arm). Single server boot, agnostic v1,
# ibm-granite/granite-4.0-h-micro-base (ctx 8192, flashinfer backend --
# EXACTLY 873945's config, NOT Zamba2's triton/ctx4096), cudagraph-ON, n=1.
# FULL rate grid {2,3,4,6} in that order within one boot, matching 873945's
# MAIN_RATES="2 3 4 6" (results/p1_opint/p1op_run.sbatch:45) -- methodology
# gate #16: reproduce the WHOLE original grid, not just the cells another
# campaign happens to cite (rate 6 was the surprise cell for Zamba2 in G1-b).
#
# Decision quantity (see PREREG_G1C_2026-08-11.md, fixed BEFORE this job is
# submitted, methodology gate #8): max(decode_running_batch_size) per rate
# window, and pop-A time-weighted frac of (54,54). Feeds directly into the
# Gate 2-S (jobs 877756/877757) sec8.9 premise label for Granite r3/r4
# ("unverified" -> VERIFIED or REFUTED per PREREG_G1C's pre-registered rule),
# which Gate 2-S's own in-job sec8.9.1 falsifier is structurally incapable of
# upgrading to "verified" (sparse PDMUX_TRACE_FORCE_PREFILL=0 telemetry,
# one-way by design). This job uses G1-b/Gate-1's DENSE forced-sampling
# protocol instead, which is what let Gate 1 assert "verified" for Zamba2.
#
# THIS JOB MAKES NO PERFORMANCE CLAIM. No TTFT/TPOT/ITL/goodput number from
# its bench_serving JSONL output may ever be quoted (n=1, diagnostic only).
# This is an observation campaign, not a performance judgment -- verdicts are
# limited to partition-label facts (VERIFIED/REFUTED/UNMEASURABLE), never
# "Granite is better/worse".
#
# Usage:   sbatch gate1c_run.sbatch
# Analyze: python3 gate1c_analyze.py gate1c_telemetry_<jobid>.jsonl \
#            --window 2 <T0> <T1> --window 3 <T0> <T1> --window 4 <T0> <T1> \
#            --window 6 <T0> <T1>
#          (this script also runs that automatically at the end and writes
#          gate1c_result_<jobid>.txt)
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
export TRITON_CACHE_DIR="$OUT/.triton_cache_gate1c_${JOBTAG}"
mkdir -p "$TRITON_CACHE_DIR"

MODEL="ibm-granite/granite-4.0-h-micro-base"
CTX=8192
BACKEND=flashinfer
PDMUX_CFG="$OUT/pdmux_a100_smoke.yml"

echo "############# G1-c job=$JOBTAG (PREREG_G1C_2026-08-11.md) #############"
echo "model=$MODEL ctx=$CTX backend=$BACKEND (873945's config -- NOT Zamba2's)"

echo "=== env sync + manifest drift check (vs 873945 reference; expects +2 lines / 0 changed, mirrors G1-b's finding vs 873944) ==="
MANIFEST="$OUT/runtime_source_manifest_gate1c_${JOBTAG}.sha256"
workspace/engine-port/scripts/bootstrap/sync_engine_tree.sh "$MANIFEST"
echo "manifest -> $MANIFEST"
REF="$OUT/reference_manifest_873945.sha256"
DIFFLOG="$OUT/manifest_diff_gate1c_${JOBTAG}.txt"
if [ -f "$REF" ]; then
  # Same non-byte-identical-tree situation as G1-b (see gate1b_run.sbatch
  # comment): the dev tree carries the HOLB probe (scheduler.py hook +
  # new holb_probe.py) that post-dates both 873945 and 874478. PDMUX_HOLB_PATH
  # is left UNSET below -> probe OFF -> collapses to `is not None` checks.
  # Equivalence argument = G3 (jobs 874635/874633, byte-identical greedy
  # output_ids ON vs OFF, 2 models x 4 arms), NOT manifest identity. We
  # verify here that manifest diff vs 873945 is EXACTLY the two new tracked
  # files with ZERO changes to any of the 13 previously-tracked files.
  CHANGED=$(comm -23 <(sort "$REF") <(sort "$MANIFEST") || true)
  ADDED=$(comm -13 <(sort "$REF") <(sort "$MANIFEST") || true)
  N_CHANGED=$(printf '%s\n' "$CHANGED" | grep -c . || true)
  N_ADDED=$(printf '%s\n' "$ADDED" | grep -c . || true)
  {
    echo "=== in reference (873945) manifest, NOT matching (same path+hash) in new manifest -- nonzero means an existing tracked file's CONTENT changed (should be 0) ==="
    echo "$CHANGED"
    echo "=== in new manifest, NOT in reference -- new tracked files (expected: exactly holb_probe.py + scheduler.py) ==="
    echo "$ADDED"
  } > "$DIFFLOG"
  ADDED_BASENAMES=$(printf '%s\n' "$ADDED" | awk 'NF{print $2}' | xargs -r -n1 basename | sort)
  EXPECTED_ADDED_BASENAMES=$(printf 'holb_probe.py\nscheduler.py\n' | sort)
  if [ "$N_CHANGED" -eq 0 ] && [ "$N_ADDED" -eq 2 ] && [ "$ADDED_BASENAMES" = "$EXPECTED_ADDED_BASENAMES" ]; then
    echo "MANIFEST_CHECK: EXPECTED-DRIFT-ONLY -- 0 changed lines vs 873945's 13 tracked files; +2 new tracked files (holb_probe.py, scheduler.py), matching the documented HOLB-probe addition (identical pattern to G1-b's check vs 873944). NOT byte-identical to 873945 -- equivalence rests on G3 (jobs 874635/874633), not manifest identity."
  else
    echo "MANIFEST_CHECK: UNEXPECTED DRIFT vs 873945 -- N_CHANGED=$N_CHANGED N_ADDED=$N_ADDED (expected 0, 2) -- see $DIFFLOG" >&2
    echo "MANIFEST_MISMATCH_ABORT" >&2
    cat "$DIFFLOG" >&2
    exit 3
  fi
else
  echo "MANIFEST_CHECK: reference manifest not found at $REF -- cannot verify drift" >&2
fi

PORT=$((33000 + SLURM_JOB_ID % 400))

echo "=== boot agnostic server (cudagraph ON, PDMUX_TELEMETRY_PATH set, HOLB probe OFF) ==="
SRVLOG="$OUT/gate1c_srv_${JOBTAG}.log"
export PDMUX_TELEMETRY_PATH="$OUT/gate1c_telemetry_${JOBTAG}.jsonl"; : > "$PDMUX_TELEMETRY_PATH"
export PDMUX_RUN_ID="gate1c_${JOBTAG}"
export PDMUX_WORKLOAD_ID="gate1c_granite-40-h-micro-base_agnostic"
export PDMUX_TRACE_FORCE_PREFILL=1
# UNCHANGED from 874478/G1-b -- do not lower this, see PREREG_G1B "이 게이트가
# Gate 1 본체와 다른 점" / job 874465 post-mortem (writer-thread backlog at
# cadence=8). Set explicitly (rather than relying on the code default) for
# auditability. This is DELIBERATELY denser than Gate 2-S's own in-job
# sec8.9.1 falsifier (PDMUX_TRACE_FORCE_PREFILL=0, see PREREG_GATE2S sec8.9.1)
# -- that density gap is exactly why this job can supply a VERIFIED verdict
# that Gate 2-S's own campaign structurally cannot.
export PDMUX_DUAL_WORKER_TRACE_EVERY=32
# Deliberately NOT set: PDMUX_HOLB_PATH (probe OFF, see manifest-check
# comment above), PDMUX_DUAL_WORKER, PDMUX_TRUE_DUAL_WORKER, PDMUX_R2_POLICY,
# PDMUX_R2_FIXED_DSM (R1-observer / true-dual-worker / Gate-2-S-style fixed-DSM
# architectures, not what agnostic v1 / job 873945 ran).

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
  echo "G1C_BOOT_FAILED" | tee "$OUT/gate1c_result_${JOBTAG}.txt"
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
  echo "CORRECTNESS=FAIL" | tee "$OUT/gate1c_result_${JOBTAG}.txt"
  kill "$SRV" 2>/dev/null; pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null
  exit 1
fi

echo "############# measurement: rate 2, 3, 4, 6 IN ORDER, one boot (n=1, seed convention matches 873945 rep1) #############"
WINDOW_ARGS=()
for R in 2 3 4 6; do
  SEED=$((9000 + 100*1 + R))
  JSONL="$OUT/gate1c_r${R}_${JOBTAG}.jsonl"; : > "$JSONL"
  echo "----- rate=$R seed=$SEED -----"
  T0=$(python3 -c 'import time; print(f"{time.perf_counter():.6f}")')
  python -m sglang.bench_serving --backend sglang --model "$MODEL" --host 127.0.0.1 --port "$PORT" \
    --dataset-name random-ids --random-input-len 2000 --random-output-len 96 --random-range-ratio 1.0 \
    --num-prompts 120 --request-rate "$R" --seed "$SEED" --warmup-requests 8 \
    --output-details --output-file "$JSONL" --tag "gate1c_r${R}" \
    | grep -E "Successful requests|Request throughput|Median TTFT|Median TPOT"
  T1=$(python3 -c 'import time; print(f"{time.perf_counter():.6f}")')
  echo "window rate=$R t0=$T0 t1=$T1"
  WINDOW_ARGS+=(--window "$R" "$T0" "$T1")
done
echo "############# measurement done #############"

kill "$SRV" 2>/dev/null
# Defense-in-depth (matches 874478/G1-b's fix): give the telemetry writer
# thread time to drain any remaining queued events before the hard SIGKILL.
sleep 5
pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null
sleep 3

echo "############# G1-c ANALYSIS (per PREREG_G1C_2026-08-11.md) #############"
python3 "$OUT/gate1c_analyze.py" "$PDMUX_TELEMETRY_PATH" "${WINDOW_ARGS[@]}" \
  | tee "$OUT/gate1c_result_${JOBTAG}.txt"

echo "DONE_G1C_${JOBTAG}"
