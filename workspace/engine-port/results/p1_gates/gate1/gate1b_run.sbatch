#!/bin/bash
#SBATCH --job-name=gate1b
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --comment="field=efficientai;appl=pytorch"
#SBATCH --chdir=/scratch/ehmoon/whlee/prefill-layer-alloc
#SBATCH --output=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/p1_gates/gate1/gate1b_%j.out
#SBATCH --error=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/p1_gates/gate1/gate1b_%j.err
# ============================================================================
# G1-b -- extends Gate 1 (job 874478) from rate{2,3} to rate{2,3,4,6}, one
# server boot, agnostic v1, Zamba2-2.7B, cudagraph-ON, n=1. Same conditions
# as 874478 (PDMUX_TRACE_FORCE_PREFILL=1, PDMUX_DUAL_WORKER_TRACE_EVERY=32
# UNCHANGED) except the rate grid, which now matches job 873944's full
# MAIN_RATES="2 3 4 6" (results/p1_opint/p1op_run.sbatch:45) IN THAT ORDER
# within a single boot, reproducing 873944's within-boot sequencing.
#
# Decision quantity (see PREREG_G1B_2026-08-07.md, fixed BEFORE this job is
# submitted, methodology gate #8): max(decode_running_batch_size) per rate
# window, and pop-A time-weighted frac of (54,54). UNLIKE Gate 1's main
# condition (a near-identity of the selector code path, per audit), whether
# decode_running_batch_size crosses decode_bs_divisor=36 is an EMPIRICAL fact
# about the workload -- this gate has real discriminating power.
#
# THIS JOB MAKES NO PERFORMANCE CLAIM. No TTFT/TPOT/ITL/goodput number from
# its bench_serving JSONL output may ever be quoted (n=1, diagnostic only).
#
# Usage:   sbatch gate1b_run.sbatch
# Analyze: python3 gate1b_analyze.py gate1b_telemetry_<jobid>.jsonl \
#            --window 2 <T0> <T1> --window 3 <T0> <T1> --window 4 <T0> <T1> \
#            --window 6 <T0> <T1>
#          (this script also runs that automatically at the end and writes
#          gate1b_result_<jobid>.txt)
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
export TRITON_CACHE_DIR="$OUT/.triton_cache_gate1b_${JOBTAG}"
mkdir -p "$TRITON_CACHE_DIR"

MODEL="Zyphra/Zamba2-2.7B"
CTX=4096
BACKEND=triton
PDMUX_CFG="$OUT/pdmux_a100_smoke.yml"

echo "############# G1-b job=$JOBTAG (PREREG_G1B_2026-08-07.md) #############"

echo "=== env sync + manifest drift check (vs 873944 reference; expects +2 lines / 0 changed, see comment) ==="
MANIFEST="$OUT/runtime_source_manifest_gate1b_${JOBTAG}.sha256"
workspace/engine-port/scripts/bootstrap/sync_engine_tree.sh "$MANIFEST"
echo "manifest -> $MANIFEST"
REF="$OUT/reference_manifest_873944.sha256"
DIFFLOG="$OUT/manifest_diff_gate1b_${JOBTAG}.txt"
if [ -f "$REF" ]; then
  # ⚠️ Deliberately NOT a byte-identical-tree check (unlike Gate 1's original
  # check against 873944, which the PREREG confirmed WAS byte-identical to
  # 874478). The dev tree now carries a HOLB probe (scheduler.py hook +
  # new holb_probe.py, installed by sync_engine_tree.sh since 2026-08-06,
  # see results/p1_gates/gate2/DIRECT_BLOCKING_DESIGN.md). PDMUX_HOLB_PATH is
  # left UNSET below -> probe OFF -> the 3 hook sites collapse to `is not
  # None` checks per scheduler.py, and job 874635/874633 (G3) confirmed
  # byte-identical greedy `output_ids` ON vs OFF across 2 models x 4 arms.
  # That equivalence argument, NOT manifest identity, is why this job's
  # results are comparable to 874478's. We verify here that the manifest
  # diff vs 873944 is EXACTLY the two new tracked files with ZERO changes to
  # any of the 13 previously-tracked files (catches unrelated drift, which
  # WOULD invalidate the comparison).
  CHANGED=$(comm -23 <(sort "$REF") <(sort "$MANIFEST") || true)
  ADDED=$(comm -13 <(sort "$REF") <(sort "$MANIFEST") || true)
  N_CHANGED=$(printf '%s\n' "$CHANGED" | grep -c . || true)
  N_ADDED=$(printf '%s\n' "$ADDED" | grep -c . || true)
  {
    echo "=== in reference (873944) manifest, NOT matching (same path+hash) in new manifest -- nonzero means an existing tracked file's CONTENT changed (should be 0) ==="
    echo "$CHANGED"
    echo "=== in new manifest, NOT in reference -- new tracked files (expected: exactly holb_probe.py + scheduler.py) ==="
    echo "$ADDED"
  } > "$DIFFLOG"
  ADDED_BASENAMES=$(printf '%s\n' "$ADDED" | awk 'NF{print $2}' | xargs -r -n1 basename | sort)
  EXPECTED_ADDED_BASENAMES=$(printf 'holb_probe.py\nscheduler.py\n' | sort)
  if [ "$N_CHANGED" -eq 0 ] && [ "$N_ADDED" -eq 2 ] && [ "$ADDED_BASENAMES" = "$EXPECTED_ADDED_BASENAMES" ]; then
    echo "MANIFEST_CHECK: EXPECTED-DRIFT-ONLY -- 0 changed lines vs 873944's 13 tracked files; +2 new tracked files (holb_probe.py, scheduler.py), matching the documented HOLB-probe addition. NOT byte-identical to 874478 (see comment above) -- equivalence rests on G3 (jobs 874635/874633), not manifest identity."
  else
    echo "MANIFEST_CHECK: UNEXPECTED DRIFT vs 873944 -- N_CHANGED=$N_CHANGED N_ADDED=$N_ADDED (expected 0, 2) -- see $DIFFLOG" >&2
    echo "MANIFEST_MISMATCH_ABORT" >&2
    cat "$DIFFLOG" >&2
    exit 3
  fi
else
  echo "MANIFEST_CHECK: reference manifest not found at $REF -- cannot verify drift" >&2
fi

PORT=$((33000 + SLURM_JOB_ID % 400))

echo "=== boot agnostic server (cudagraph ON, PDMUX_TELEMETRY_PATH set, HOLB probe OFF) ==="
SRVLOG="$OUT/gate1b_srv_${JOBTAG}.log"
export PDMUX_TELEMETRY_PATH="$OUT/gate1b_telemetry_${JOBTAG}.jsonl"; : > "$PDMUX_TELEMETRY_PATH"
export PDMUX_RUN_ID="gate1b_${JOBTAG}"
export PDMUX_WORKLOAD_ID="gate1b_zamba2-27b_agnostic"
export PDMUX_TRACE_FORCE_PREFILL=1
# UNCHANGED from 874478 -- do not lower this, see PREREG_G1B "이 게이트가
# Gate 1 본체와 다른 점" / job 874465 post-mortem (writer-thread backlog at
# cadence=8). Set explicitly (rather than relying on the code default) for
# auditability.
export PDMUX_DUAL_WORKER_TRACE_EVERY=32
# Deliberately NOT set: PDMUX_HOLB_PATH (probe OFF, see manifest-check
# comment above), PDMUX_DUAL_WORKER, PDMUX_TRUE_DUAL_WORKER (R1-observer /
# true-dual-worker architectures, not what agnostic v1 / job 873944 ran).

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
  echo "G1B_BOOT_FAILED" | tee "$OUT/gate1b_result_${JOBTAG}.txt"
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
  echo "CORRECTNESS=FAIL" | tee "$OUT/gate1b_result_${JOBTAG}.txt"
  kill "$SRV" 2>/dev/null; pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null
  exit 1
fi

echo "############# measurement: rate 2, 3, 4, 6 IN ORDER, one boot (n=1, seed convention matches 873944 rep1) #############"
WINDOW_ARGS=()
for R in 2 3 4 6; do
  SEED=$((9000 + 100*1 + R))
  JSONL="$OUT/gate1b_r${R}_${JOBTAG}.jsonl"; : > "$JSONL"
  echo "----- rate=$R seed=$SEED -----"
  T0=$(python3 -c 'import time; print(f"{time.perf_counter():.6f}")')
  python -m sglang.bench_serving --backend sglang --model "$MODEL" --host 127.0.0.1 --port "$PORT" \
    --dataset-name random-ids --random-input-len 2000 --random-output-len 96 --random-range-ratio 1.0 \
    --num-prompts 120 --request-rate "$R" --seed "$SEED" --warmup-requests 8 \
    --output-details --output-file "$JSONL" --tag "gate1b_r${R}" \
    | grep -E "Successful requests|Request throughput|Median TTFT|Median TPOT"
  T1=$(python3 -c 'import time; print(f"{time.perf_counter():.6f}")')
  echo "window rate=$R t0=$T0 t1=$T1"
  WINDOW_ARGS+=(--window "$R" "$T0" "$T1")
done
echo "############# measurement done #############"

kill "$SRV" 2>/dev/null
# Defense-in-depth (matches 874478's fix): give the telemetry writer thread
# time to drain any remaining queued events before the hard SIGKILL.
sleep 5
pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null
sleep 3

echo "############# G1-b ANALYSIS (per PREREG_G1B_2026-08-07.md) #############"
python3 "$OUT/gate1b_analyze.py" "$PDMUX_TELEMETRY_PATH" "${WINDOW_ARGS[@]}" \
  | tee "$OUT/gate1b_result_${JOBTAG}.txt"

echo "DONE_G1B_${JOBTAG}"
