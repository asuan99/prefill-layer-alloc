#!/bin/bash
#SBATCH --job-name=g2holbsmoke
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --comment=pytorch
#SBATCH --chdir=/scratch/ehmoon/whlee/prefill-layer-alloc
#SBATCH --output=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/p1_gates/gate2/%x_%j.out
#SBATCH --error=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/p1_gates/gate2/%x_%j.err
#
# ============================================================================
# Pre-submission smoke test, required after the 874601 post-mortem
# (2026-08-06): before resubmitting the full g2_holb_observer.sbatch
# campaigns, confirm the Phase-A sha extractor actually produces a
# NON-EMPTY sha for a real /generate call, on the real GPU path, for BOTH
# models (Zamba2-2.7B, ctx=4096, is the one that broke; Granite is included
# too since it silently exercised an out-of-spec 6005-token prompt instead of
# the intended ~2000, which g2_holb_phaseA_lib.sh's prompt builder now fixes
# for both).
#
# This sources g2_holb_phaseA_lib.sh -- the SAME file g2_holb_observer.sbatch
# sources -- so this smoke test exercises the ACTUAL code path, not a
# paraphrase of it (explicit instruction: do not resubmit without this).
#
# NO performance claim, no HOLB probe telemetry scored here -- this is purely
# "does the correctness-check plumbing produce a real, non-empty value".
# ============================================================================
set -uo pipefail
module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0
export PS1="" PATH="$HOME/.local/bin:$PATH"
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate
cd /scratch/ehmoon/whlee/prefill-layer-alloc

OUT=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/p1_gates/gate2
mkdir -p "$OUT"
export HF_HOME=/scratch/ehmoon/whlee/prefill-layer-alloc/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="$OUT/.triton_cache_smoke_${SLURM_JOB_ID}"
mkdir -p "$TRITON_CACHE_DIR"

source "$OUT/g2_holb_phaseA_lib.sh"

INLEN=2000
OUTLEN=96
PORT_BASE=$((34000 + SLURM_JOB_ID % 400))

boot_plain () {  # $1=model $2=ctx $3=backend $4=port $5=srvlog -> pid
  python -m sglang.launch_server --model-path "$1" --trust-remote-code --dtype bfloat16 \
    --attention-backend "$3" \
    --disable-radix-cache --mem-fraction-static 0.82 --max-running-requests 48 --context-length "$2" \
    --host 127.0.0.1 --port "$4" > "$5" 2>&1 &
  echo $!
}
wait_ready () {
  local PID="$1" PORT="$2" ok=0
  for i in $(seq 1 170); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health 2>/dev/null||true)" = 200 ] && { ok=1; break; }
    kill -0 "$PID" 2>/dev/null || { echo "SERVER DIED (pid $PID)" >&2; break; }
    sleep 5
  done
  echo $ok
}
kill_server () { kill "$1" 2>/dev/null; pkill -9 -f "launch_server.*--port $2" 2>/dev/null; sleep 3; }

OVERALL_OK=1
smoke_one () {  # $1=model $2=ctx $3=backend $4=tag
  local MODEL="$1" CTX="$2" BACKEND="$3" TAG="$4"
  PROMPT_TXT="$OUT/smoke_prompt_${TAG}_${SLURM_JOB_ID}.txt"
  PROMPT_META="$OUT/smoke_prompt_meta_${TAG}_${SLURM_JOB_ID}.json"
  echo "=== [$TAG] building prompt ($MODEL, target $INLEN tokens, real tokenizer -- NOT byte length;" \
       "the 874601 bug was exactly byte-length-mistaken-for-token-count) ==="
  build_g2holb_prompt
  cat "$PROMPT_META"
  local REENC_LEN DEV_PCT
  REENC_LEN=$(python3 -c "import json; print(json.load(open('$PROMPT_META'))['reencoded_len'])")
  DEV_PCT=$(python3 -c "print(f'{100.0*($REENC_LEN-$INLEN)/$INLEN:+.1f}')")
  echo "[$TAG] CHECK 1/3 (token count): reencoded_len=$REENC_LEN target=$INLEN deviation=${DEV_PCT}% ctx=$CTX outlen=$OUTLEN"
  TOKEN_COUNT_OK=1
  if [ "$REENC_LEN" -ge "$((CTX - OUTLEN - 32))" ]; then
    echo "[$TAG] SMOKE_FAIL (check 1/3): reencoded prompt would overflow context (this is exactly the" >&2
    echo "[$TAG] 874601 failure mode -- a byte-length prompt re-tokenized to 11362 vs ctx=4096)." >&2
    OVERALL_OK=0
    return
  fi
  # Informational only (bench_serving's own random-ids generator does not
  # guarantee an exact round trip either -- see g2_holb_phaseA_lib.sh header);
  # the hard requirement is "does not overflow context", checked above.
  python3 -c "
dev = abs($REENC_LEN - $INLEN) / $INLEN
print(f'[$TAG] token-count deviation from target = {dev*100:.1f}% '
      + ('(within bench_serving-typical round-trip variance)' if dev <= 0.30 else '(larger than typical -- inspect prompt meta)'))
"

  local PORT=$((PORT_BASE + RANDOM % 100))
  local SRVLOG="$OUT/smoke_srv_${TAG}_${SLURM_JOB_ID}.log"
  echo "=== [$TAG] booting plain-arm server on port $PORT ==="
  PID=$(boot_plain "$MODEL" "$CTX" "$BACKEND" "$PORT" "$SRVLOG")
  OK=$(wait_ready "$PID" "$PORT")
  echo "[$TAG] boot_ok=$OK"
  if [ "$OK" != 1 ]; then
    echo "[$TAG] SMOKE_FAIL: server did not boot" >&2
    OVERALL_OK=0
    tail -40 "$SRVLOG" | grep -viE "pynvml|Future"
    return
  fi

  unset STATUS HTTP_CODE METHOD SHA N_OUTPUT_IDS DETAIL
  eval "$(greedy_call "$PORT" "$OUT/smoke_${TAG}_call_${SLURM_JOB_ID}.raw.json")"
  echo "[$TAG] greedy_call -> STATUS=$STATUS HTTP_CODE=${HTTP_CODE:-} METHOD=${METHOD:-} SHA=${SHA:-} N_OUTPUT_IDS=${N_OUTPUT_IDS:-} DETAIL=${DETAIL:-}"
  echo "[$TAG] CHECK 2/3 (http status): HTTP_CODE=${HTTP_CODE:-}"
  echo "[$TAG] CHECK 3/3 (non-empty sha): SHA=${SHA:-<empty>}"
  if [ "$STATUS" = OK ] && [ "${HTTP_CODE:-}" = 200 ] && [ -n "${SHA:-}" ]; then
    echo "[$TAG] SMOKE_PASS: all 3 checks pass (token_count_ok, http=200, non-empty sha; method=$METHOD, sha=${SHA:0:16}...)"
  elif [ "$STATUS" = TIMEOUT ] || [ "$STATUS" = ERROR ]; then
    # Measurement failure, not a check failure: the call never produced a
    # result to check. Still blocks resubmission (a harness that cannot get an
    # answer out of the server is not ready), but it is labelled for what it
    # is so nobody records "smoke check failed" as a correctness signal
    # (methodology gate #21).
    echo "[$TAG] SMOKE_UNDETERMINED ($STATUS): the greedy call produced no result to check" >&2
    echo "[$TAG]   STATUS=$STATUS HTTP_CODE=${HTTP_CODE:-} DETAIL=${DETAIL:-}" >&2
    if [ "$STATUS" = TIMEOUT ]; then
      echo "[$TAG]   -> the call exceeded G2_GREEDY_MAX_TIME=${G2_GREEDY_MAX_TIME:-180}s (job-876699 hang class)." >&2
      echo "[$TAG]   -> see $OUT/smoke_${TAG}_call_${SLURM_JOB_ID}.raw.json.status.json" >&2
    fi
    echo "[$TAG] do NOT resubmit the full campaign until this is measurable." >&2
    OVERALL_OK=0
  else
    echo "[$TAG] SMOKE_FAIL: STATUS=$STATUS HTTP_CODE=${HTTP_CODE:-} SHA=${SHA:-<empty>} DETAIL=${DETAIL:-}" >&2
    echo "[$TAG] do NOT resubmit the full campaign until this passes." >&2
    OVERALL_OK=0
  fi
  kill_server "$PID" "$PORT"
}

smoke_one "Zyphra/Zamba2-2.7B" 4096 triton zamba2
smoke_one "ibm-granite/granite-4.0-h-micro-base" 8192 flashinfer granite

echo "############# SMOKE SUMMARY: OVERALL_OK=$OVERALL_OK #############"
if [ "$OVERALL_OK" != 1 ]; then
  echo "SMOKE_TEST_FAILED -- fix before resubmitting g2_holb_observer.sbatch" >&2
  exit 1
fi
echo "SMOKE_TEST_PASSED -- safe to resubmit g2_holb_observer.sbatch"
