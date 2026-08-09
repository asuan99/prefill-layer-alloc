#!/bin/bash
#SBATCH --job-name=g2earun
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --comment=pytorch
#SBATCH --chdir=/scratch/ehmoon/whlee/prefill-layer-alloc
#SBATCH --output=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/p1_gates/gate2/%x_%j.out
#SBATCH --error=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/p1_gates/gate2/%x_%j.err
#
# ==============================================================================
# E-A MAIN CAMPAIGN.  Spec = PREREG_G2EA_2026-08-07.md.  5-arm decomposition
# of the fused-side chunking lever (chunked-prefill-size AND enable-mixed-
# chunk, tested both jointly and separately) vs agnostic pdmux.
#
#   A1  plain        (no extra flags, default chunked_prefill_size=8192)
#   A1m plainmix      --enable-mixed-chunk
#   A3  chunk512      --chunked-prefill-size 512
#   A3m chunk512mix   --chunked-prefill-size 512 --enable-mixed-chunk
#   A4  agnostic      --enable-pdmux --pdmux-config-path <cfg>
#                      --chunked-prefill-size -1 --disable-overlap-schedule
#
# A2 (plainaux) is OMITTED (PREREG_G2EA section: R2' already established at
# n=10 by jobs 875344/875346; not re-litigated here; no cross-job comparison
# with A2 is permitted per methodology gate #11).
#
# G-1 (single, self-repro + cross-vs-A1) runs for ALL FIVE arms.
# G-2 (16-concurrent, self + cross-vs-A1) runs for plainmix/chunk512/
# chunk512mix ONLY (the three "new or chunk-active" paths per the task
# brief -- A1's own G-1 doubles as the fixed cross baseline, A4 is gated by
# prior campaigns and not re-litigated here).
# Both gates now persist FULL output_ids (g2ea_gate.py fixes harness defect
# #2 identified by the 2026-08-07 audit of 875344/875346: the old
# g2_concurrent_gate.py/g2ctrl_gate.py stored sha256 only / stripped ids).
#
# HOLB probe (PDMUX_HOLB_PATH) is mounted ON for all five arms throughout
# (carried over from rev4 addendum B; additional GPU cost is zero). Its
# output is secondary/exploratory and MUST NOT override the primary verdict.
# ==============================================================================
set -uo pipefail

echo "=== HEADER: node placement + concurrent jobs (recorded before module load) ==="
echo "hostname=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "SLURM_NODELIST=${SLURM_NODELIST:-unset}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-unset}"
echo "--- squeue --all -p amd_a100nv_8 at job start ---"
squeue --all -p amd_a100nv_8 -o "%.10i %.9P %.20j %.8T %.10M %N %.8u" 2>&1 || true
echo "=== END HEADER ==="

module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0
export PS1="" PATH="$HOME/.local/bin:$PATH"
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate
cd /scratch/ehmoon/whlee/prefill-layer-alloc

MODEL="${1:?usage: g2ea_run.sbatch MODEL CTX BACKEND TAG \"RATE1 RATE2 ...\" DESIGNATED_RATE}"
CTX="${2:?ctx}"
BACKEND="${3:?backend}"
TAG="${4:?tag}"
MAIN_RATES="${5:?main rate list, e.g. \"2 3\" or \"3 4\"}"
DESIGNATED_RATE="${6:?designated rate, e.g. 3 for granite (PREREG G2EA: moved from r4 to r3)}"

OUT=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/p1_gates/gate2
mkdir -p "$OUT"
export HF_HOME=/scratch/ehmoon/whlee/prefill-layer-alloc/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="$OUT/.triton_cache_ea_${SLURM_JOB_ID}"
mkdir -p "$TRITON_CACHE_DIR"

echo "=== env sync ==="
MANIFEST="$OUT/runtime_source_manifest_ea_${TAG}_${SLURM_JOB_ID}.sha256"
workspace/engine-port/scripts/bootstrap/sync_engine_tree.sh "$MANIFEST"
echo "manifest -> $MANIFEST"
diff <(sort workspace/engine-port/results/p1_gates/gate1/reference_manifest_873944.sha256) \
     <(sort "$MANIFEST") > "$OUT/manifest_diff_ea_${TAG}_${SLURM_JOB_ID}.txt"
echo "--- manifest diff vs 873944 reference (expect exactly 2 '>' lines, 0 '<' lines) ---"
cat "$OUT/manifest_diff_ea_${TAG}_${SLURM_JOB_ID}.txt"
N_ADDED=$(grep -c '^>' "$OUT/manifest_diff_ea_${TAG}_${SLURM_JOB_ID}.txt" || true)
N_REMOVED=$(grep -c '^<' "$OUT/manifest_diff_ea_${TAG}_${SLURM_JOB_ID}.txt" || true)
if [ "$N_ADDED" = "2" ] && [ "$N_REMOVED" = "0" ]; then
  echo "MANIFEST_CHECK=PASS (added=$N_ADDED removed=$N_REMOVED)"
else
  echo "MANIFEST_CHECK=FAIL (added=$N_ADDED removed=$N_REMOVED, expected added=2 removed=0)" >&2
  echo "MANIFEST_CHECK_ABORT -- engine tree has drifted, refusing to run a scientific campaign on an unverified tree" >&2
  exit 4
fi

# ------------------------------------------------------------------------
# Config (PREREG_G2EA_2026-08-07.md)
# ------------------------------------------------------------------------
NPROMPT_SCAN=60
NPROMPT_MAIN=120
INLEN=2000
OUTLEN=96
RANGE_RATIO=1.0
SCAN_RATES="1 2 3 4 5 6 8 10 12"
SCAN_SEEDS="7 17 27"
WARMUP_SCAN=4
WARMUP_MAIN=8
NREP_MAIN=10
G2_NCONCURRENT=16
PDMUX_CFG="$OUT/pdmux_a100_smoke.yml"
ARMS="plain plainmix chunk512 chunk512mix agnostic"
G2_ARMS="plainmix chunk512 chunk512mix"   # G-2 (16-concurrent) applies to these only

PORT_BASE=$((33000 + SLURM_JOB_ID % 400))

# ------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------
arm_flags () {   # $1=arm -> echoes extra server flags
  case "$1" in
    plain)        echo "" ;;
    plainmix)     echo "--enable-mixed-chunk" ;;
    chunk512)     echo "--chunked-prefill-size 512" ;;
    chunk512mix)  echo "--chunked-prefill-size 512 --enable-mixed-chunk" ;;
    agnostic)     echo "--enable-pdmux --pdmux-config-path $PDMUX_CFG --chunked-prefill-size -1 --disable-overlap-schedule" ;;
    *) echo "UNKNOWN_ARM_$1" ;;
  esac
}

arm_cps_hint () {  # $1=arm -> chunked-prefill-size value to pass to g2ea_gate.py for boundary calc
  case "$1" in
    plain|plainmix) echo 8192 ;;
    chunk512|chunk512mix) echo 512 ;;
    agnostic) echo -1 ;;
    *) echo 0 ;;
  esac
}

boot_server () {   # $1=arm $2=port $3=srvlog $4=holbpath $5=context_tag -> echoes pid
  local ARM="$1" PORT="$2" SRVLOG="$3" HOLB="$4" CTXTAG="$5"
  local EXTRA; EXTRA="$(arm_flags "$ARM")"
  export PDMUX_HOLB_PATH="$HOLB"
  export PDMUX_HOLB_RUN_ID="${SLURM_JOB_ID}_${TAG}_${ARM}_${CTXTAG}"
  export PDMUX_HOLB_WORKLOAD_ID="${CTXTAG}"
  python -m sglang.launch_server --model-path "$MODEL" --trust-remote-code --dtype bfloat16 \
    --attention-backend "$BACKEND" \
    --disable-radix-cache --mem-fraction-static 0.82 --max-running-requests 48 --context-length "$CTX" \
    $EXTRA --host 127.0.0.1 --port "$PORT" > "$SRVLOG" 2>&1 &
  echo $!
}

wait_ready () {   # $1=pid $2=port -> echoes 1/0
  local PID="$1" PORT="$2" ok=0
  for i in $(seq 1 170); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health 2>/dev/null||true)" = 200 ] && { ok=1; break; }
    kill -0 "$PID" 2>/dev/null || { echo "SERVER DIED (pid $PID)" >&2; break; }
    sleep 5
  done
  echo $ok
}

capture_evidence () {  # $1=srvlog
  local SRVLOG="$1"
  echo "--- capture evidence ($SRVLOG) ---"
  grep -iE "captur|cuda graph|cudagraph" "$SRVLOG" | grep -viE "Future|pynvml" | head -8
  echo "--- error evidence ($SRVLOG) ---"
  grep -iE "error|traceback|not support|illegal|assert" "$SRVLOG" | grep -viE "Future|pynvml" | head -8
  echo "--- HOLB banner ($SRVLOG) ---"
  grep -c "HOLB probe ENABLED" "$SRVLOG" || true
}

correctness_check () {  # $1=port -> prints PASS/FAIL, RESP echoed
  local PORT="$1" RESP
  RESP=$(curl -s http://127.0.0.1:$PORT/generate -H 'Content-Type: application/json' \
    -d '{"text":"The capital of France is","sampling_params":{"max_new_tokens":8,"temperature":0}}')
  echo "SANITY_RAW: $RESP"
  if echo "$RESP" | grep -q "Paris"; then echo "CORRECTNESS=PASS"; else echo "CORRECTNESS=FAIL"; fi
}

kill_server () {  # $1=pid $2=port
  local PID="$1" PORT="$2"
  kill "$PID" 2>/dev/null
  pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null
  sleep 3
}

record_flags () {  # $1=arm $2=port $3=context_tag -- includes internal_states
                    # (scheduler-side realized server_args, e.g.
                    # max_mamba_cache_size) since /get_server_info now
                    # redirects to /server_info which embeds it.
  local ARM="$1" PORT="$2" CTXTAG="$3"
  curl -s "http://127.0.0.1:$PORT/get_server_info" \
    > "$OUT/g2ea_${TAG}_flags_${ARM}_${CTXTAG}_${SLURM_JOB_ID}.json" || true
}

# ==========================================================================
# G-1 / G-2: CORRECTNESS + AVAILABILITY GATES (PREREG_G2EA section: 규율)
# Informational per-arm; a genuine FAIL invalidates that arm's data for the
# primary comparisons but does not abort the whole job (other arms' data
# stays valid). g2ea_analyze.py must read this artifact and flag exclusions.
# ==========================================================================
echo "############# G-1/G-2: CORRECTNESS GATES (model=$MODEL tag=$TAG) #############"
PROMPT_TXT="$OUT/g2ea_prompt_${TAG}_${SLURM_JOB_ID}.txt"
PROMPT_META="$OUT/g2ea_prompt_meta_${TAG}_${SLURM_JOB_ID}.json"
source "$OUT/g2_holb_phaseA_lib.sh"   # build_g2holb_prompt (reused, unmodified)

echo "=== building faithful random-ids prompt (bench_serving generator, target ${INLEN} tokens) ==="
build_g2holb_prompt
cat "$PROMPT_META"
REENC_LEN=$(python3 -c "import json; print(json.load(open('$PROMPT_META'))['reencoded_len'])")
SAFETY_MARGIN=$((CTX - OUTLEN - 32))
echo "reencoded prompt length = $REENC_LEN tokens (target $INLEN, ctx=$CTX, outlen=$OUTLEN)"
if [ "$REENC_LEN" -ge "$SAFETY_MARGIN" ]; then
  echo "PROMPT_BUILD_ABORT: reencoded prompt ($REENC_LEN tokens) leaves < 32 tokens of headroom" \
       "before ctx=$CTX with max_new_tokens=$OUTLEN (874601 failure mode)." >&2
  exit 5
fi

CORRECTNESS_JSON="$OUT/g2ea_${TAG}_correctness_${SLURM_JOB_ID}.json"; : > "$CORRECTNESS_JSON"
PLAIN_G1_JSON=""
G1_HAS_FAIL=0
G1_HAS_UNDETERMINED=0

for ARM in $ARMS; do
  PORT=$((PORT_BASE + RANDOM % 100))
  SRVLOG="$OUT/g2ea_${TAG}_srv_${ARM}_G1_${SLURM_JOB_ID}.log"
  HOLB="$OUT/g2ea_${TAG}_G1_${ARM}_${SLURM_JOB_ID}.holb.jsonl"
  PID=$(boot_server "$ARM" "$PORT" "$SRVLOG" "$HOLB" "G1")
  OK=$(wait_ready "$PID" "$PORT")
  echo "G1 arm=$ARM boot_ok=$OK"
  capture_evidence "$SRVLOG"
  G1_JSON="$OUT/g2ea_${TAG}_G1_${ARM}_${SLURM_JOB_ID}.json"
  if [ "$OK" != 1 ]; then
    echo "G1 BOOT FAILED arm=$ARM"; tail -30 "$SRVLOG" | grep -viE "pynvml|Future"
    G1_HAS_UNDETERMINED=1
    kill_server "$PID" "$PORT"
    printf '%s\n' "{\"gate\": \"G1\", \"arm\": \"$ARM\", \"self_repro_status\": \"UNDETERMINED\", \"reason\": \"boot_failed\"}" >> "$CORRECTNESS_JSON"
    continue
  fi
  record_flags "$ARM" "$PORT" "G1"
  CPS_HINT=$(arm_cps_hint "$ARM")
  CROSS_ARG=""
  if [ "$ARM" != "plain" ] && [ -n "$PLAIN_G1_JSON" ]; then
    CROSS_ARG="--cross-baseline-file $PLAIN_G1_JSON"
  fi
  python3 "$OUT/g2ea_gate.py" single --host 127.0.0.1 --port "$PORT" --prompt-file "$PROMPT_TXT" \
    --max-new-tokens "$OUTLEN" --n-calls 2 --chunked-prefill-size "$CPS_HINT" --prompt-len-hint "$REENC_LEN" \
    $CROSS_ARG --out "$G1_JSON"
  kill_server "$PID" "$PORT"
  if [ "$ARM" = "plain" ]; then PLAIN_G1_JSON="$G1_JSON"; fi
  STATUS=$(python3 -c "import json; print(json.load(open('$G1_JSON'))['self_repro_status'])")
  CROSS_MATCH="n/a"
  if [ "$ARM" != "plain" ]; then
    CROSS_MATCH=$(python3 -c "
import json
d = json.load(open('$G1_JSON'))
c = d.get('cross_baseline_check')
print(c['match'] if c else 'no_cross_check')
")
  fi
  echo "G1 arm=$ARM status=$STATUS cross_vs_plain_match=$CROSS_MATCH"
  [ "$STATUS" = "SELFREPRO_FAIL" ] && G1_HAS_FAIL=1
  [ "$STATUS" = "UNDETERMINED" ] && G1_HAS_UNDETERMINED=1
  [ "$CROSS_MATCH" = "False" ] && G1_HAS_FAIL=1
  python3 - "$CORRECTNESS_JSON" "$ARM" "$STATUS" "$CROSS_MATCH" <<'PYEOF'
import json, sys
path, arm, status, cross = sys.argv[1:5]
with open(path, "a") as f:
    f.write(json.dumps({"gate": "G1", "arm": arm, "self_repro_status": status,
                         "cross_vs_plain_match": cross}) + "\n")
PYEOF
done
echo "G1_SUMMARY has_fail=$G1_HAS_FAIL has_undetermined=$G1_HAS_UNDETERMINED"

# G-2: concurrent gate, plainmix/chunk512/chunk512mix only.
echo "############# G-2: CONCURRENT CORRECTNESS GATE (n=$G2_NCONCURRENT) arms=[$G2_ARMS] #############"
for ARM in $G2_ARMS; do
  PORT=$((PORT_BASE + RANDOM % 100))
  SRVLOG="$OUT/g2ea_${TAG}_srv_${ARM}_G2_${SLURM_JOB_ID}.log"
  HOLB="$OUT/g2ea_${TAG}_G2_${ARM}_${SLURM_JOB_ID}.holb.jsonl"
  PID=$(boot_server "$ARM" "$PORT" "$SRVLOG" "$HOLB" "G2")
  OK=$(wait_ready "$PID" "$PORT")
  echo "G2 arm=$ARM boot_ok=$OK"
  capture_evidence "$SRVLOG"
  G2_STATUS="UNDETERMINED"
  if [ "$OK" != 1 ]; then
    echo "G2 BOOT FAILED arm=$ARM"; tail -30 "$SRVLOG" | grep -viE "pynvml|Future"
  else
    record_flags "$ARM" "$PORT" "G2"
    G1_JSON="$OUT/g2ea_${TAG}_G1_${ARM}_${SLURM_JOB_ID}.json"
    G2_OUT="$OUT/g2ea_${TAG}_G2concurrent_${ARM}_${SLURM_JOB_ID}.json"
    CPS_HINT=$(arm_cps_hint "$ARM")
    CROSS_ARG=""
    [ -n "$PLAIN_G1_JSON" ] && CROSS_ARG="--cross-baseline-file $PLAIN_G1_JSON"
    if [ -f "$G1_JSON" ]; then
      python3 "$OUT/g2ea_gate.py" concurrent --host 127.0.0.1 --port "$PORT" --prompt-file "$PROMPT_TXT" \
        --max-new-tokens "$OUTLEN" --n-concurrent "$G2_NCONCURRENT" --chunked-prefill-size "$CPS_HINT" \
        --prompt-len-hint "$REENC_LEN" --self-baseline-file "$G1_JSON" $CROSS_ARG --out "$G2_OUT"
      G2_STATUS=$(python3 -c "import json; d=json.load(open('$G2_OUT')); print(d['self']['verdict'])")
      if grep -qiE "AssertionError|assert mid is not None" "$SRVLOG"; then
        echo "G2 MAMBA POOL ASSERT DETECTED in $SRVLOG (arm=$ARM)"
        G2_STATUS="${G2_STATUS}_WITH_ASSERT_IN_LOG"
      fi
    else
      echo "G2 SKIPPED arm=$ARM: no G1 self-baseline json found at $G1_JSON"
    fi
  fi
  kill_server "$PID" "$PORT"
  echo "G2_STATUS arm=$ARM $G2_STATUS"
  python3 - "$CORRECTNESS_JSON" "$ARM" "$G2_STATUS" <<'PYEOF'
import json, sys
path, arm, status = sys.argv[1:4]
with open(path, "a") as f:
    f.write(json.dumps({"gate": "G2_CONCURRENT", "arm": arm, "n_concurrent": 16, "verdict": status}) + "\n")
PYEOF
done

echo "############# CORRECTNESS GATES DONE (informational only; NOT aborting the campaign) #############"

# ==========================================================================
# PHASE 0: capacity scan, n=3 seeds, 5 arms (PREREG_G2EA)
# ==========================================================================
echo "############# PHASE 0: capacity scan n=3 (seeds: $SCAN_SEEDS), 5 arms (model=$MODEL tag=$TAG) #############"
for ARM in $ARMS; do
  for SEED in $SCAN_SEEDS; do
    PORT=$((PORT_BASE + RANDOM % 100))
    SRVLOG="$OUT/g2ea_${TAG}_srv_${ARM}_capscan_seed${SEED}_${SLURM_JOB_ID}.log"
    JSONL="$OUT/g2ea_${TAG}_capscan_${ARM}_seed${SEED}_${SLURM_JOB_ID}.jsonl"; : > "$JSONL"
    HOLB="$OUT/g2ea_${TAG}_capscan_${ARM}_seed${SEED}_${SLURM_JOB_ID}.holb.jsonl"
    echo "===== capscan arm=$ARM seed=$SEED port=$PORT ====="
    PID=$(boot_server "$ARM" "$PORT" "$SRVLOG" "$HOLB" "capscan_seed${SEED}")
    OK=$(wait_ready "$PID" "$PORT")
    echo "capscan boot_ok=$OK arm=$ARM seed=$SEED"
    capture_evidence "$SRVLOG"
    if [ "$OK" = 1 ]; then
      record_flags "$ARM" "$PORT" "capscan_seed${SEED}"
      correctness_check "$PORT"
      for R in $SCAN_RATES; do
        echo "----- capscan $ARM seed=$SEED rate=$R -----"
        python -m sglang.bench_serving --backend sglang --model "$MODEL" --host 127.0.0.1 --port "$PORT" \
          --dataset-name random-ids --random-input-len $INLEN --random-output-len $OUTLEN --random-range-ratio $RANGE_RATIO \
          --num-prompts $NPROMPT_SCAN --request-rate $R --seed $SEED --warmup-requests $WARMUP_SCAN \
          --output-details --output-file "$JSONL" --tag "capscan_${ARM}_seed${SEED}_r${R}" \
          | grep -E "Successful requests|Request throughput|Median TTFT|P99 TTFT|Median TPOT|P99 TPOT"
      done
    else
      echo "CAPSCAN BOOT FAILED arm=$ARM seed=$SEED"; tail -30 "$SRVLOG" | grep -viE "pynvml|Future"
    fi
    kill_server "$PID" "$PORT"
  done
done
echo "############# PHASE 0 DONE #############"

# ==========================================================================
# PHASE 1: n=10 paired measurement, 5 arms, arm order permuted per rep
# Crash contingency: 1 retry, then void the whole rep across all arms.
# ==========================================================================
echo "############# PHASE 1: n=$NREP_MAIN paired measurement, arm order permuted per rep (model=$MODEL tag=$TAG rates=[$MAIN_RATES] designated=$DESIGNATED_RATE) #############"
ARM_ORDER_LOG="$OUT/g2ea_${TAG}_arm_order_${SLURM_JOB_ID}.jsonl"; : > "$ARM_ORDER_LOG"
VOIDED_REPS_FILE="$OUT/g2ea_${TAG}_voided_reps_${SLURM_JOB_ID}.txt"; : > "$VOIDED_REPS_FILE"

run_arm_rep_once () {   # $1=ARM $2=REP -> prints OK or CRASHED (only token on stdout)
  local ARM="$1" REP="$2"
  local PORT=$((PORT_BASE + RANDOM % 100))
  local SRVLOG="$OUT/g2ea_${TAG}_srv_${ARM}_rep${REP}_${SLURM_JOB_ID}.log"
  local HOLB="$OUT/g2ea_${TAG}_${ARM}_rep${REP}_${SLURM_JOB_ID}.holb.jsonl"
  local JSONL="$OUT/g2ea_${TAG}_${ARM}_rep${REP}_${SLURM_JOB_ID}.jsonl"; : > "$JSONL"
  local PID; PID=$(boot_server "$ARM" "$PORT" "$SRVLOG" "$HOLB" "rep${REP}")
  local OK; OK=$(wait_ready "$PID" "$PORT")
  echo "rep=$REP arm=$ARM boot_ok=$OK" >&2
  capture_evidence "$SRVLOG" >&2
  if [ "$OK" != 1 ]; then
    echo "REP BOOT FAILED rep=$REP arm=$ARM" >&2
    tail -30 "$SRVLOG" 2>/dev/null | grep -viE "pynvml|Future" >&2
    kill_server "$PID" "$PORT"
    echo CRASHED
    return
  fi
  record_flags "$ARM" "$PORT" "rep${REP}" >&2
  local CORR; CORR=$(correctness_check "$PORT" | tail -1)
  echo "$CORR" >&2
  if [ "$CORR" != "CORRECTNESS=PASS" ]; then
    echo "REP CORRECTNESS SANITY FAILED rep=$REP arm=$ARM" >&2
    kill_server "$PID" "$PORT"
    echo CRASHED
    return
  fi
  local CRASHED=0
  for R in $MAIN_RATES; do
    kill -0 "$PID" 2>/dev/null || { echo "SERVER DIED before rate=$R rep=$REP arm=$ARM" >&2; CRASHED=1; break; }
    local SEED=$((9000 + 100*REP + R))
    echo "----- rep=$REP arm=$ARM rate=$R seed=$SEED -----" >&2
    python -m sglang.bench_serving --backend sglang --model "$MODEL" --host 127.0.0.1 --port "$PORT" \
      --dataset-name random-ids --random-input-len $INLEN --random-output-len $OUTLEN --random-range-ratio $RANGE_RATIO \
      --num-prompts $NPROMPT_MAIN --request-rate $R --seed $SEED --warmup-requests $WARMUP_MAIN \
      --output-details --output-file "$JSONL" --tag "rep${REP}_${ARM}_r${R}" \
      2>&1 | grep -E "Successful requests|Request throughput|Median TTFT|P99 TTFT|Median TPOT|P99 TPOT|Median E2E" >&2
    kill -0 "$PID" 2>/dev/null || { echo "SERVER DIED after rate=$R rep=$REP arm=$ARM" >&2; CRASHED=1; break; }
  done
  if grep -qiE "AssertionError|assert mid is not None" "$SRVLOG"; then
    echo "MAMBA POOL ASSERT DETECTED in $SRVLOG (rep=$REP arm=$ARM)" >&2
    CRASHED=1
  fi
  kill_server "$PID" "$PORT"
  if [ "$CRASHED" = 1 ]; then echo CRASHED; else echo OK; fi
}

for REP in $(seq 1 $NREP_MAIN); do
  ORDER="$(shuf -e $ARMS | tr '\n' ' ')"
  echo "===== rep=$REP arm_order='$ORDER' ====="
  python3 - "$ARM_ORDER_LOG" "$REP" "$ORDER" <<'PYEOF'
import json, sys
path, rep, order = sys.argv[1], int(sys.argv[2]), sys.argv[3].split()
with open(path, "a") as f:
    f.write(json.dumps({"rep": rep, "order": order}) + "\n")
PYEOF
  REP_VOIDED=0
  for ARM in $ORDER; do
    if [ "$REP_VOIDED" = 1 ]; then
      echo "rep=$REP arm=$ARM SKIPPED (rep already voided earlier this iteration)"
      continue
    fi
    RESULT=$(run_arm_rep_once "$ARM" "$REP")
    if [ "$RESULT" = "CRASHED" ]; then
      echo "rep=$REP arm=$ARM CRASHED -- retrying once (PREREG_G2EA contingency)"
      RESULT=$(run_arm_rep_once "$ARM" "$REP")
      if [ "$RESULT" = "CRASHED" ]; then
        echo "rep=$REP arm=$ARM CRASHED AGAIN on retry -- VOIDING rep=$REP for ALL arms"
        echo "$REP" >> "$VOIDED_REPS_FILE"
        REP_VOIDED=1
      else
        echo "rep=$REP arm=$ARM retry OK"
      fi
    fi
  done
done
echo "############# PHASE 1 DONE #############"
echo "voided reps (if any): $(cat "$VOIDED_REPS_FILE" 2>/dev/null | tr '\n' ' ')"
echo "designated_rate=$DESIGNATED_RATE"
echo "DONE_G2EA_${TAG}"
