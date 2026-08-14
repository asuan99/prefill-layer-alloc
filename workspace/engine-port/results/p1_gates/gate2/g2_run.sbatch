#!/bin/bash
#SBATCH --job-name=g2run
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --comment="field=efficientai;appl=pytorch"
#SBATCH --chdir=/scratch/ehmoon/whlee/prefill-layer-alloc
#SBATCH --output=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/p1_gates/gate2/%x_%j.out
#SBATCH --error=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/p1_gates/gate2/%x_%j.err
#
# ==============================================================================
# Gate 2 rev4 MAIN CAMPAIGN.  Spec = PREREG_GATE2_2026-08-06.md (rev4) + the
# 2026-08-07 dated addenda appended to that file (Section 14).  This script
# implements the mechanical data-collection protocol only; scoring/verdicts
# are g2_analyze.py's job (run separately, after the job completes).
#
# One job = one model = one complete Gate-2 comparison unit:
#   G-1/G-2   correctness gates (PREREG section 2.1)
#   Phase 0   capacity scan, n=3 seeds (7,17,27), 4 arms, 9 rates  (PREREG section 3)
#   Phase 1   n=10 paired measurement, 4 arms, model-specific rate grid
#             (PREREG section 2), arm order permuted per rep, seed =
#             9000 + 100*rep + rate
#
# HOLB probe (PDMUX_HOLB_PATH) is mounted ON for all four arms throughout
# (addendum B, Section 14 of PREREG) -- additional GPU cost is zero.  Its
# output is secondary/exploratory (DIRECT_BLOCKING_DESIGN.md section 10) and
# MUST NOT be used to override the primary R3'/R4' verdict.
# ==============================================================================
set -uo pipefail

# ------------------------------------------------------------------------
# Header: node placement + concurrent-job snapshot, so cross-node/cross-job
# confounds can be checked post hoc (this run's own instruction; also see
# MEMORY [[deconfound-measurement-lessons]] item 17 on trusting raw node
# lists over derived same-node tables).
# ------------------------------------------------------------------------
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

MODEL="${1:?usage: g2_run.sbatch MODEL CTX BACKEND TAG \"RATE1 RATE2 ...\"}"
CTX="${2:?ctx}"
BACKEND="${3:?backend}"
TAG="${4:?tag}"
MAIN_RATES="${5:?main rate list, e.g. \"2 3\" or \"3 4 6\"}"

OUT=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/p1_gates/gate2
mkdir -p "$OUT"
export HF_HOME=/scratch/ehmoon/whlee/prefill-layer-alloc/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="$OUT/.triton_cache_${SLURM_JOB_ID}"
mkdir -p "$TRITON_CACHE_DIR"

echo "=== env sync ==="
MANIFEST="$OUT/runtime_source_manifest_${TAG}_${SLURM_JOB_ID}.sha256"
workspace/engine-port/scripts/bootstrap/sync_engine_tree.sh "$MANIFEST"
echo "manifest -> $MANIFEST"
# HOLB probe adds exactly 2 lines vs the 873944 reference (scheduler.py +
# holb_probe.py) -- same expectation the HOLB-observer jobs (874601-874635)
# already verified. Any other delta means the tree drifted since those runs.
diff <(sort workspace/engine-port/results/p1_gates/gate1/reference_manifest_873944.sha256) \
     <(sort "$MANIFEST") > "$OUT/manifest_diff_${TAG}_${SLURM_JOB_ID}.txt"
echo "--- manifest diff vs 873944 reference (expect exactly 2 '>' lines, 0 '<' lines) ---"
cat "$OUT/manifest_diff_${TAG}_${SLURM_JOB_ID}.txt"
N_ADDED=$(grep -c '^>' "$OUT/manifest_diff_${TAG}_${SLURM_JOB_ID}.txt" || true)
N_REMOVED=$(grep -c '^<' "$OUT/manifest_diff_${TAG}_${SLURM_JOB_ID}.txt" || true)
if [ "$N_ADDED" = "2" ] && [ "$N_REMOVED" = "0" ]; then
  echo "MANIFEST_CHECK=PASS (added=$N_ADDED removed=$N_REMOVED)"
else
  echo "MANIFEST_CHECK=FAIL (added=$N_ADDED removed=$N_REMOVED, expected added=2 removed=0)" >&2
  echo "MANIFEST_CHECK_ABORT -- engine tree has drifted from the 873944/HOLB-verified state, refusing to run a scientific campaign on an unverified tree" >&2
  exit 4
fi

# ------------------------------------------------------------------------
# Config (PREREG_GATE2_2026-08-06.md sections 1, 2, 3, 8)
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
ARMS="plain plainaux chunk512 agnostic"

PORT_BASE=$((33000 + SLURM_JOB_ID % 400))

# ------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------
arm_flags () {   # $1=arm -> echoes extra server flags (PREREG section 1)
  case "$1" in
    plain)     echo "" ;;
    plainaux)  echo "--chunked-prefill-size -1 --disable-overlap-schedule" ;;
    chunk512)  echo "--chunked-prefill-size 512" ;;
    agnostic)  echo "--enable-pdmux --pdmux-config-path $PDMUX_CFG --chunked-prefill-size -1 --disable-overlap-schedule" ;;
    *) echo "UNKNOWN_ARM_$1" ;;
  esac
}

boot_server () {   # $1=arm $2=port $3=srvlog $4=holbpath $5=context_tag -> echoes pid
  local ARM="$1" PORT="$2" SRVLOG="$3" HOLB="$4" CTXTAG="$5"
  local EXTRA; EXTRA="$(arm_flags "$ARM")"
  # HOLB mounted ON for all four arms throughout this campaign (PREREG
  # section 14 addendum B) -- pdmux-independent, additional GPU cost = 0.
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

record_flags () {  # $1=arm $2=port $3=context_tag -- PREREG section 1.2
  local ARM="$1" PORT="$2" CTXTAG="$3"
  curl -s "http://127.0.0.1:$PORT/get_server_info" \
    > "$OUT/g2_${TAG}_flags_${ARM}_${CTXTAG}_${SLURM_JOB_ID}.json" || true
}

# ==========================================================================
# G-1 / G-2: CORRECTNESS + AVAILABILITY GATES (PREREG section 2.1)
#
# Informational, not a hard stop for the whole job: a genuine G-1/G-2 FAIL
# invalidates A3-dependent conclusions (R3'/R4', PREREG section 5.2) but
# A1/A2/A4 data and the A2-vs-A4 mechanism-decomposition comparison
# (R1'/R2') remain valid, so data collection proceeds regardless (aborting
# a 6h job over one arm's gate would destroy still-valid data for the other
# three). g2_analyze.py must read this artifact and flag A3 exclusion if
# G-1 or G-2 measured a genuine FAIL (not UNDETERMINED).
# ==========================================================================
echo "############# G-1/G-2: CORRECTNESS GATES (model=$MODEL tag=$TAG) #############"
PROMPT_TXT="$OUT/g2_prompt_${TAG}_${SLURM_JOB_ID}.txt"
PROMPT_META="$OUT/g2_prompt_meta_${TAG}_${SLURM_JOB_ID}.json"
source "$OUT/g2_holb_phaseA_lib.sh"   # build_g2holb_prompt, greedy_call (unmodified, reused)

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

CORRECTNESS_JSON="$OUT/g2_${TAG}_correctness_${SLURM_JOB_ID}.json"; : > "$CORRECTNESS_JSON"
declare -A G1_SHA
declare -A G1_STATUS
G1_HAS_FAIL=0
G1_HAS_UNDETERMINED=0

for ARM in $ARMS; do
  PORT=$((PORT_BASE + RANDOM % 100))
  SRVLOG="$OUT/g2_${TAG}_srv_${ARM}_G1_${SLURM_JOB_ID}.log"
  HOLB="$OUT/g2_${TAG}_G1_${ARM}_${SLURM_JOB_ID}.holb.jsonl"
  PID=$(boot_server "$ARM" "$PORT" "$SRVLOG" "$HOLB" "G1")
  OK=$(wait_ready "$PID" "$PORT")
  echo "G1 arm=$ARM boot_ok=$OK"
  capture_evidence "$SRVLOG"
  if [ "$OK" != 1 ]; then
    echo "G1 BOOT FAILED arm=$ARM"; tail -30 "$SRVLOG" | grep -viE "pynvml|Future"
    G1_STATUS[$ARM]=UNDETERMINED
    G1_HAS_UNDETERMINED=1
    kill_server "$PID" "$PORT"
    printf '%s\n' "{\"gate\": \"G1\", \"arm\": \"$ARM\", \"self_repro_status\": \"UNDETERMINED\", \"sha\": null, \"reason\": \"boot_failed\"}" >> "$CORRECTNESS_JSON"
    continue
  fi
  record_flags "$ARM" "$PORT" "G1"
  unset STATUS HTTP_CODE METHOD SHA N_OUTPUT_IDS DETAIL
  eval "$(greedy_call "$PORT" "$OUT/g2_${TAG}_G1_${ARM}_call1_${SLURM_JOB_ID}.raw.json")"
  C1_STATUS="$STATUS"; C1_SHA="${SHA:-}"; C1_HTTP="${HTTP_CODE:-}"; C1_DETAIL="${DETAIL:-}"
  unset STATUS HTTP_CODE METHOD SHA N_OUTPUT_IDS DETAIL
  eval "$(greedy_call "$PORT" "$OUT/g2_${TAG}_G1_${ARM}_call2_${SLURM_JOB_ID}.raw.json")"
  C2_STATUS="$STATUS"; C2_SHA="${SHA:-}"; C2_HTTP="${HTTP_CODE:-}"; C2_DETAIL="${DETAIL:-}"
  echo "G1 arm=$ARM call1_status=$C1_STATUS http=$C1_HTTP sha=$C1_SHA detail=$C1_DETAIL"
  echo "G1 arm=$ARM call2_status=$C2_STATUS http=$C2_HTTP sha=$C2_SHA detail=$C2_DETAIL"
  kill_server "$PID" "$PORT"
  if [ "$C1_STATUS" != OK ] || [ "$C2_STATUS" != OK ]; then
    G1_STATUS[$ARM]=UNDETERMINED
    G1_HAS_UNDETERMINED=1
    python3 - "$CORRECTNESS_JSON" "$ARM" "$C1_STATUS" "$C1_DETAIL" "$C2_STATUS" "$C2_DETAIL" <<'PYEOF'
import json, sys
path, arm, s1, d1, s2, d2 = sys.argv[1:7]
with open(path, "a") as f:
    f.write(json.dumps({"gate": "G1", "arm": arm, "self_repro_status": "UNDETERMINED",
                         "sha": None, "call1_status": s1, "call1_detail": d1,
                         "call2_status": s2, "call2_detail": d2}) + "\n")
PYEOF
    continue
  fi
  G1_SHA[$ARM]="$C1_SHA"
  if [ "$C1_SHA" = "$C2_SHA" ]; then
    G1_STATUS[$ARM]=SELFREPRO_OK
  else
    G1_STATUS[$ARM]=SELFREPRO_FAIL
    G1_HAS_FAIL=1
  fi
  python3 - "$CORRECTNESS_JSON" "$ARM" "${G1_STATUS[$ARM]}" "${G1_SHA[$ARM]}" <<'PYEOF'
import json, sys
path, arm, status, sha = sys.argv[1:5]
with open(path, "a") as f:
    f.write(json.dumps({"gate": "G1", "arm": arm, "self_repro_status": status, "sha": sha}) + "\n")
PYEOF
done

# G-1 pairwise comparisons (PREREG section 2.1 G-1: A1 vs A3, A2 vs A1 byte-identical)
G1_A1_VS_A3="UNDETERMINED"; G1_A2_VS_A1="UNDETERMINED"
if [ -n "${G1_SHA[plain]:-}" ] && [ -n "${G1_SHA[chunk512]:-}" ]; then
  [ "${G1_SHA[plain]}" = "${G1_SHA[chunk512]}" ] && G1_A1_VS_A3=PASS || { G1_A1_VS_A3=FAIL; G1_HAS_FAIL=1; }
fi
if [ -n "${G1_SHA[plain]:-}" ] && [ -n "${G1_SHA[plainaux]:-}" ]; then
  [ "${G1_SHA[plain]}" = "${G1_SHA[plainaux]}" ] && G1_A2_VS_A1=PASS || { G1_A2_VS_A1=FAIL; G1_HAS_FAIL=1; }
fi
echo "G1_PAIRWISE A1_vs_A3=$G1_A1_VS_A3 (byte-identical greedy output required) A2_vs_A1=$G1_A2_VS_A1 A4_self_repro=${G1_STATUS[agnostic]:-UNDETERMINED}"
python3 - "$CORRECTNESS_JSON" "$G1_A1_VS_A3" "$G1_A2_VS_A1" "${G1_STATUS[agnostic]:-UNDETERMINED}" <<'PYEOF'
import json, sys
path, a1a3, a2a1, a4sr = sys.argv[1:5]
with open(path, "a") as f:
    f.write(json.dumps({"gate": "G1_PAIRWISE", "A1_vs_A3": a1a3, "A2_vs_A1": a2a1, "A4_self_repro": a4sr}) + "\n")
PYEOF
echo "G1_SUMMARY has_fail=$G1_HAS_FAIL has_undetermined=$G1_HAS_UNDETERMINED"

# G-2: concurrent gate (>=16 simultaneous identical greedy requests to A3,
# each compared against A1's batch=1 baseline sha from G-1).
echo "############# G-2: CONCURRENT CORRECTNESS GATE (n=$G2_NCONCURRENT, chunk512 vs A1 batch=1) #############"
G2_STATUS="UNDETERMINED"
if [ -z "${G1_SHA[plain]:-}" ]; then
  echo "G2 SKIPPED: no A1 baseline sha available from G1 (A1 boot or greedy call failed)"
else
  PORT=$((PORT_BASE + RANDOM % 100))
  SRVLOG="$OUT/g2_${TAG}_srv_chunk512_G2_${SLURM_JOB_ID}.log"
  HOLB="$OUT/g2_${TAG}_G2_chunk512_${SLURM_JOB_ID}.holb.jsonl"
  PID=$(boot_server chunk512 "$PORT" "$SRVLOG" "$HOLB" "G2")
  OK=$(wait_ready "$PID" "$PORT")
  echo "G2 arm=chunk512 boot_ok=$OK"
  capture_evidence "$SRVLOG"
  if [ "$OK" != 1 ]; then
    echo "G2 BOOT FAILED"; tail -30 "$SRVLOG" | grep -viE "pynvml|Future"
    G2_STATUS=UNDETERMINED
  else
    record_flags chunk512 "$PORT" "G2"
    G2_OUT="$OUT/g2_${TAG}_G2concurrent_${SLURM_JOB_ID}.json"
    python3 "$OUT/g2_concurrent_gate.py" --host 127.0.0.1 --port "$PORT" --prompt-file "$PROMPT_TXT" \
      --max-new-tokens "$OUTLEN" --n-concurrent "$G2_NCONCURRENT" --baseline-sha "${G1_SHA[plain]}" --out "$G2_OUT"
    G2_STATUS=$(python3 -c "import json; print(json.load(open('$G2_OUT'))['verdict'])")
    if grep -qiE "AssertionError|assert mid is not None" "$SRVLOG"; then
      echo "G2 MAMBA POOL ASSERT DETECTED in $SRVLOG (see PREREG section 2.1 contingency)"
      G2_STATUS="${G2_STATUS}_WITH_ASSERT_IN_LOG"
    fi
  fi
  kill_server "$PID" "$PORT"
fi
echo "G2_STATUS=$G2_STATUS"
python3 - "$CORRECTNESS_JSON" "$G2_STATUS" <<'PYEOF'
import json, sys
path, status = sys.argv[1:3]
with open(path, "a") as f:
    f.write(json.dumps({"gate": "G2_CONCURRENT", "arm": "chunk512", "n_concurrent": 16, "verdict": status}) + "\n")
PYEOF

echo "############# CORRECTNESS GATES DONE (informational only -- see block comment above; NOT aborting the campaign) #############"

# ==========================================================================
# PHASE 0: capacity scan, n=3 seeds, 4 arms, 9 rates (PREREG section 3)
# ==========================================================================
echo "############# PHASE 0: capacity scan n=3 (seeds: $SCAN_SEEDS), 4 arms (model=$MODEL tag=$TAG) #############"
for ARM in $ARMS; do
  for SEED in $SCAN_SEEDS; do
    PORT=$((PORT_BASE + RANDOM % 100))
    SRVLOG="$OUT/g2_${TAG}_srv_${ARM}_capscan_seed${SEED}_${SLURM_JOB_ID}.log"
    JSONL="$OUT/g2_${TAG}_capscan_${ARM}_seed${SEED}_${SLURM_JOB_ID}.jsonl"; : > "$JSONL"
    HOLB="$OUT/g2_${TAG}_capscan_${ARM}_seed${SEED}_${SLURM_JOB_ID}.holb.jsonl"
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
echo "NOTE: PREREG section 5.1.1 mid-check (A3 capscan SD vs sqrt(2)*SD(A4) at the" \
     "2x threshold) is computed by g2_analyze.py from the capscan artifacts above" \
     "-- it is an analysis-time UNDERPOWERED flag, not a runtime gate, so Phase 1" \
     "proceeds unconditionally."

# ==========================================================================
# PHASE 1: n=10 paired measurement, 4 arms, arm order permuted per rep
# (PREREG section 2, 5.1). Crash contingency: PREREG section 2.1 -- 1 retry,
# then void the whole rep across all arms (recorded, not silently dropped).
# ==========================================================================
echo "############# PHASE 1: n=$NREP_MAIN paired measurement, arm order permuted per rep (model=$MODEL tag=$TAG rates=[$MAIN_RATES]) #############"
ARM_ORDER_LOG="$OUT/g2_${TAG}_arm_order_${SLURM_JOB_ID}.jsonl"; : > "$ARM_ORDER_LOG"
VOIDED_REPS_FILE="$OUT/g2_${TAG}_voided_reps_${SLURM_JOB_ID}.txt"; : > "$VOIDED_REPS_FILE"

run_arm_rep_once () {   # $1=ARM $2=REP -> prints OK or CRASHED (only token on stdout)
  local ARM="$1" REP="$2"
  local PORT=$((PORT_BASE + RANDOM % 100))
  local SRVLOG="$OUT/g2_${TAG}_srv_${ARM}_rep${REP}_${SLURM_JOB_ID}.log"
  local HOLB="$OUT/g2_${TAG}_${ARM}_rep${REP}_${SLURM_JOB_ID}.holb.jsonl"
  local JSONL="$OUT/g2_${TAG}_${ARM}_rep${REP}_${SLURM_JOB_ID}.jsonl"; : > "$JSONL"
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
      echo "rep=$REP arm=$ARM CRASHED -- retrying once (PREREG section 2.1 contingency)"
      RESULT=$(run_arm_rep_once "$ARM" "$REP")
      if [ "$RESULT" = "CRASHED" ]; then
        echo "rep=$REP arm=$ARM CRASHED AGAIN on retry -- VOIDING rep=$REP for ALL arms (PREREG section 2.1: 'n<4 -> that model UNSCOREABLE', not scored low)"
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
echo "DONE_G2_${TAG}"
