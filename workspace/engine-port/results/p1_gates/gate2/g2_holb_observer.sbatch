#!/bin/bash
#SBATCH --job-name=g2holb
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --comment=pytorch
#SBATCH --chdir=/scratch/ehmoon/whlee/prefill-layer-alloc
#SBATCH --output=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/p1_gates/gate2/%x_%j.out
#SBATCH --error=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/p1_gates/gate2/%x_%j.err
#
# ============================================================================
# DRAFT -- NOT SUBMITTED.  Written by engine-porter 2026-08-06 as the GPU half
# of the HOLB-probe correctness gate.  Hand to experiment-runner to submit.
#
# This job produces NO performance claim.  It answers exactly two questions:
#
#   PHASE A (correctness, blocking):  does PDMUX_HOLB_PATH change what the
#       engine computes?  Greedy output must be byte-identical ON vs OFF in all
#       four arms.  If any arm fails, the whole job stops -- Phase B would be
#       meaningless.
#
#   PHASE B (observer effect):  does the probe change how fast the engine runs?
#       Paired ON/OFF, n=NREP, randomised order within each rep.  Gate is
#       |mean relative delta| < 3% with a paired CI containing 0, per arm, per
#       response variable.  Scoring is NOT done here -- the .jsonl files go to
#       result-analyst.
#
# The four arms are exactly the Gate-2 arms (PREREG_GATE2_2026-08-06.md
# section 1); the probe hooks live in Scheduler.run_batch so all four are
# instrumented by the same code.  See DIRECT_BLOCKING_DESIGN.md.
# ============================================================================
set -uo pipefail
module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0
export PS1="" PATH="$HOME/.local/bin:$PATH"
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate
cd /scratch/ehmoon/whlee/prefill-layer-alloc

MODEL="${1:?usage: g2_holb_observer.sbatch MODEL CTX BACKEND TAG RATE}"
CTX="${2:?ctx}"
BACKEND="${3:?backend}"
TAG="${4:?tag}"
RATE="${5:?rate}"

OUT=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/p1_gates/gate2
mkdir -p "$OUT"
export HF_HOME=/scratch/ehmoon/whlee/prefill-layer-alloc/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="$OUT/.triton_cache_${SLURM_JOB_ID}"
mkdir -p "$TRITON_CACHE_DIR"

echo "=== env sync ==="
MANIFEST="$OUT/runtime_source_manifest_holb_${TAG}_${SLURM_JOB_ID}.sha256"
workspace/engine-port/scripts/bootstrap/sync_engine_tree.sh "$MANIFEST"
echo "manifest -> $MANIFEST"
# Two lines are EXPECTED to be new vs the 873944 reference (holb_probe.py and
# scheduler.py); every pre-existing line must be unchanged (DIRECT_BLOCKING_
# DESIGN.md section 6, G2d). Verified at runtime, not just offline.
diff <(sort workspace/engine-port/results/p1_gates/gate1/reference_manifest_873944.sha256) \
     <(sort "$MANIFEST") > "$OUT/manifest_diff_holb_${SLURM_JOB_ID}.txt"
echo "--- manifest diff vs 873944 (expect exactly 2 '>' lines, 0 '<' lines) ---"
cat "$OUT/manifest_diff_holb_${SLURM_JOB_ID}.txt"
N_ADDED=$(grep -c '^>' "$OUT/manifest_diff_holb_${SLURM_JOB_ID}.txt" || true)
N_REMOVED=$(grep -c '^<' "$OUT/manifest_diff_holb_${SLURM_JOB_ID}.txt" || true)
if [ "$N_ADDED" = "2" ] && [ "$N_REMOVED" = "0" ]; then
  echo "MANIFEST_CHECK=PASS (added=$N_ADDED removed=$N_REMOVED, matches design doc G2d expectation)"
else
  echo "MANIFEST_CHECK=FAIL (added=$N_ADDED removed=$N_REMOVED, expected added=2 removed=0)" >&2
  echo "MANIFEST_CHECK_ABORT" >&2
  exit 4
fi

NPROMPT=120
INLEN=2000
OUTLEN=96
RANGE_RATIO=1.0
NREP="${PDMUX_G2HOLB_NREP:-5}"  # project protocol: n>=5 paired (CLAUDE.md); design doc's own
                                 # minimum was n=4 -- we use the project floor since GPU budget
                                 # allows it comfortably (see budget note below).
PDMUX_CFG="$OUT/pdmux_a100_smoke.yml"
PORT_BASE=$((33000 + SLURM_JOB_ID % 400))
WINDOWS="$OUT/g2holb_${TAG}_windows_${SLURM_JOB_ID}.jsonl"; : > "$WINDOWS"

# Shared Phase-A helpers (build_g2holb_prompt, greedy_call) -- same file used
# by g2_holb_smoke.sbatch, per the 2026-08-06 post-mortem (874601): the
# smoke test must exercise the ACTUAL code path, not a paraphrase of it.
source "$OUT/g2_holb_phaseA_lib.sh"
PROMPT_TXT="$OUT/g2holb_prompt_${TAG}_${SLURM_JOB_ID}.txt"     # job-tagged: two Gate-2 jobs
PROMPT_META="$OUT/g2holb_prompt_meta_${TAG}_${SLURM_JOB_ID}.json"  # run in parallel, sharing $OUT

# ---------------------------------------------------------------------------
# arms (PREREG_GATE2 section 1).  Common flags identical to 873944/873945.
# ---------------------------------------------------------------------------
arm_flags () {
  case "$1" in
    plain)        echo "" ;;
    plainaux)     echo "--chunked-prefill-size -1 --disable-overlap-schedule" ;;
    chunk512)     echo "--chunked-prefill-size 512" ;;
    agnostic)     echo "--enable-pdmux --pdmux-config-path $PDMUX_CFG --chunked-prefill-size -1 --disable-overlap-schedule" ;;
    *) echo "UNKNOWN_ARM" ;;
  esac
}

boot_server () {   # $1=arm $2=port $3=srvlog $4=holbpath("" => probe OFF) -> pid
  local ARM="$1" PORT="$2" SRVLOG="$3" HOLB="$4"
  local EXTRA; EXTRA="$(arm_flags "$ARM")"
  if [ -n "$HOLB" ]; then
    export PDMUX_HOLB_PATH="$HOLB"
    export PDMUX_HOLB_RUN_ID="${SLURM_JOB_ID}_${TAG}_${ARM}"
    export PDMUX_HOLB_WORKLOAD_ID="r${RATE}_in${INLEN}_out${OUTLEN}"
  else
    unset PDMUX_HOLB_PATH PDMUX_HOLB_RUN_ID PDMUX_HOLB_WORKLOAD_ID
  fi
  python -m sglang.launch_server --model-path "$MODEL" --trust-remote-code --dtype bfloat16 \
    --attention-backend "$BACKEND" \
    --disable-radix-cache --mem-fraction-static 0.82 --max-running-requests 48 --context-length "$CTX" \
    $EXTRA --host 127.0.0.1 --port "$PORT" > "$SRVLOG" 2>&1 &
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

kill_server () {
  kill "$1" 2>/dev/null
  pkill -9 -f "launch_server.*--port $2" 2>/dev/null
  sleep 3
}

# realized flags + probe banner: never assert a flag, measure it (methodology
# gate "realized, not target").
record_flags () {  # $1=arm $2=port $3=onoff $4=srvlog
  curl -s "http://127.0.0.1:$2/get_server_info" \
    > "$OUT/g2holb_${TAG}_flags_${1}_${3}_${SLURM_JOB_ID}.json" || true
  echo "--- HOLB banner ($1/$3) ---"
  grep -c "HOLB probe ENABLED" "$4" || true
}

# ---------------------------------------------------------------------------
# PHASE A -- correctness: greedy output must not depend on the probe
# ---------------------------------------------------------------------------
# The prompt is the ~2000-token random-ids prompt the benchmark itself uses
# (bench_serving's own construction, replicated with $MODEL's own tokenizer --
# see g2_holb_phaseA_lib.sh for why this matters and what broke without it).
echo "=== building faithful random-ids prompt (bench_serving generator, target ${INLEN} tokens) ==="
build_g2holb_prompt
cat "$PROMPT_META"
REENC_LEN=$(python3 -c "import json; print(json.load(open('$PROMPT_META'))['reencoded_len'])")
SAFETY_MARGIN=$((CTX - OUTLEN - 32))
echo "reencoded prompt length = $REENC_LEN tokens (target $INLEN, ctx=$CTX, outlen=$OUTLEN)"
if [ "$REENC_LEN" -ge "$SAFETY_MARGIN" ]; then
  echo "PROMPT_BUILD_ABORT: reencoded prompt ($REENC_LEN tokens) leaves < 32 tokens of headroom" \
       "before ctx=$CTX with max_new_tokens=$OUTLEN. Refusing to run Phase A with a prompt that" \
       "would overflow context (this is exactly the 874601 failure mode)." >&2
  exit 5
fi

echo "############# PHASE A: probe ON/OFF greedy equivalence #############"
CORRECTNESS_JSONL="$OUT/g2holb_${TAG}_correctness_${SLURM_JOB_ID}.json"; : > "$CORRECTNESS_JSONL"
HAS_FAIL=0
HAS_UNDETERMINED=0
for ARM in plain plainaux chunk512 agnostic; do
  declare -A CALL_STATUS
  declare -A CALL_SHA
  declare -A CALL_METHOD
  declare -A CALL_DETAIL
  declare -A SELFREPRO
  for ONOFF in off on; do
    PORT=$((PORT_BASE + RANDOM % 100))
    SRVLOG="$OUT/g2holb_${TAG}_srv_${ARM}_A_${ONOFF}_${SLURM_JOB_ID}.log"
    HOLB=""; [ "$ONOFF" = on ] && HOLB="$OUT/g2holb_${TAG}_A_${ARM}_${SLURM_JOB_ID}.holb.jsonl"
    PID=$(boot_server "$ARM" "$PORT" "$SRVLOG" "$HOLB")
    OK=$(wait_ready "$PID" "$PORT")
    echo "phaseA arm=$ARM probe=$ONOFF boot_ok=$OK"
    if [ "$OK" != 1 ]; then
      HAS_UNDETERMINED=1
      CALL_STATUS[$ONOFF]=ERROR; CALL_DETAIL[$ONOFF]="boot_failed"
      kill_server "$PID" "$PORT"; continue
    fi
    record_flags "$ARM" "$PORT" "A_$ONOFF" "$SRVLOG"
    # two identical greedy calls: self-reproducibility (the only check available
    # for the pdmux arm, whose batching is not bit-deterministic across runs).
    # Each call always prints exactly one eval-safe KEY=value line (never raises).
    # greedy_call checks the HTTP status code FIRST, independent of the body
    # (2026-08-06 post-mortem #2: job 874601's real failure was a genuine 400
    # Bad Request -- "input longer than context length" -- that the harness
    # never looked at; see g2_holb_phaseA_lib.sh's header). A non-200 or a
    # malformed/error response is UNDETERMINED, never silently coerced to FAIL.
    unset STATUS HTTP_CODE METHOD SHA N_OUTPUT_IDS DETAIL
    eval "$(greedy_call "$PORT" "$OUT/g2holb_${TAG}_A_${ARM}_${ONOFF}_call1_${SLURM_JOB_ID}.raw.json")"
    S1_STATUS="$STATUS"; S1_HTTP="${HTTP_CODE:-}"; S1_SHA="${SHA:-}"; S1_METHOD="${METHOD:-}"; S1_DETAIL="${DETAIL:-}"
    unset STATUS HTTP_CODE METHOD SHA N_OUTPUT_IDS DETAIL
    eval "$(greedy_call "$PORT" "$OUT/g2holb_${TAG}_A_${ARM}_${ONOFF}_call2_${SLURM_JOB_ID}.raw.json")"
    S2_STATUS="$STATUS"; S2_HTTP="${HTTP_CODE:-}"; S2_SHA="${SHA:-}"; S2_METHOD="${METHOD:-}"; S2_DETAIL="${DETAIL:-}"
    echo "phaseA arm=$ARM probe=$ONOFF call1_status=$S1_STATUS call1_http=$S1_HTTP call1_method=$S1_METHOD call1_sha=$S1_SHA call1_detail=$S1_DETAIL"
    echo "phaseA arm=$ARM probe=$ONOFF call2_status=$S2_STATUS call2_http=$S2_HTTP call2_method=$S2_METHOD call2_sha=$S2_SHA call2_detail=$S2_DETAIL"
    if [ "$S1_STATUS" != OK ] || [ "$S2_STATUS" != OK ] || [ "$S1_HTTP" != 200 ] || [ "$S2_HTTP" != 200 ]; then
      HAS_UNDETERMINED=1
      CALL_STATUS[$ONOFF]=ERROR
      CALL_DETAIL[$ONOFF]="call1=${S1_STATUS}(http${S1_HTTP}):${S1_DETAIL} call2=${S2_STATUS}(http${S2_HTTP}):${S2_DETAIL}"
      echo "phaseA arm=$ARM probe=$ONOFF UNDETERMINED (extraction failed or non-200, NOT a byte-mismatch): ${CALL_DETAIL[$ONOFF]}"
    else
      CALL_STATUS[$ONOFF]=OK
      CALL_METHOD[$ONOFF]="$S1_METHOD"
      if [ "$S1_SHA" = "$S2_SHA" ]; then
        SELFREPRO[$ONOFF]=true
        CALL_SHA[$ONOFF]="$S1_SHA"
      else
        SELFREPRO[$ONOFF]=false
        HAS_FAIL=1
        echo "SELF-REPRO FAIL arm=$ARM probe=$ONOFF (two back-to-back greedy calls in the SAME probe state disagree)"
      fi
    fi
    kill_server "$PID" "$PORT"
  done
  if [ "${CALL_STATUS[off]:-ERROR}" != OK ] || [ "${CALL_STATUS[on]:-ERROR}" != OK ]; then
    RESULT=UNDETERMINED
    echo "PHASE_A arm=$ARM RESULT=UNDETERMINED off_status=${CALL_STATUS[off]:-ERROR} on_status=${CALL_STATUS[on]:-ERROR}"
  elif [ "${SELFREPRO[off]:-false}" != true ] || [ "${SELFREPRO[on]:-false}" != true ]; then
    RESULT=FAIL
    echo "PHASE_A arm=$ARM RESULT=FAIL (self-repro instability, see SELF-REPRO FAIL line above)"
  elif [ "${CALL_SHA[off]}" = "${CALL_SHA[on]}" ]; then
    RESULT=PASS
    echo "PHASE_A arm=$ARM RESULT=PASS sha=${CALL_SHA[off]} method=${CALL_METHOD[off]}"
  else
    RESULT=FAIL
    HAS_FAIL=1
    echo "PHASE_A arm=$ARM RESULT=FAIL off=${CALL_SHA[off]} on=${CALL_SHA[on]} (probe changed engine output)"
  fi
  python3 - "$CORRECTNESS_JSONL" "$ARM" "$RESULT" "${CALL_SHA[off]:-}" "${CALL_SHA[on]:-}" \
    "${SELFREPRO[off]:-}" "${SELFREPRO[on]:-}" "${CALL_METHOD[off]:-}" "${CALL_METHOD[on]:-}" \
    "${CALL_STATUS[off]:-}" "${CALL_STATUS[on]:-}" "${CALL_DETAIL[off]:-}" "${CALL_DETAIL[on]:-}" <<'PYEOF'
import json, sys
(path, arm, result, sha_off, sha_on, sr_off, sr_on, method_off, method_on,
 status_off, status_on, detail_off, detail_on) = sys.argv[1:14]
rec = {
    "arm": arm, "result": result,
    "sha_off": sha_off or None, "sha_on": sha_on or None,
    "self_repro_off": (sr_off == "true") if sr_off else None,
    "self_repro_on": (sr_on == "true") if sr_on else None,
    "method_off": method_off or None, "method_on": method_on or None,
    "status_off": status_off or None, "status_on": status_on or None,
    "detail_off": detail_off or None, "detail_on": detail_on or None,
}
with open(path, "a") as f:
    f.write(json.dumps(rec) + "\n")
PYEOF
  unset CALL_STATUS CALL_SHA CALL_METHOD CALL_DETAIL SELFREPRO
done
echo "PHASE_A_SUMMARY has_fail=$HAS_FAIL has_undetermined=$HAS_UNDETERMINED"
if [ "$HAS_FAIL" = 1 ] || [ "$HAS_UNDETERMINED" = 1 ]; then
  if [ "$HAS_FAIL" = 1 ]; then
    echo "PHASE A: at least one arm's probe ON/OFF output genuinely differs (or is self-repro"
    echo "unstable) -- refusing to run Phase B (an observer-effect number for an instrumentation"
    echo "that changes the output would be meaningless)."
  fi
  if [ "$HAS_UNDETERMINED" = 1 ]; then
    echo "PHASE A: at least one arm could not be measured (boot failure or response-schema"
    echo "extraction error -- see g2holb_${TAG}_correctness_${SLURM_JOB_ID}.json and the .raw.json"
    echo "dumps). This is NOT evidence the probe changed output -- do not report it as such."
    echo "Blocking Phase B out of caution (an unmeasured arm cannot be certified either way)."
  fi
  echo "############# GATE 2 HOLB SCORING (G3 fail-fast path) #############"
  python3 "$OUT/g2_holb_analyze.py" --outdir "$OUT" --tag "$TAG" --jobid "$SLURM_JOB_ID" \
    --arms plain plainaux chunk512 agnostic --nrep "$NREP" \
    | tee "$OUT/g2holb_result_${TAG}_${SLURM_JOB_ID}.txt"
  echo "DONE_G2HOLB_${TAG}"; exit 1
fi

# ---------------------------------------------------------------------------
# PHASE B -- observer effect, paired
# ---------------------------------------------------------------------------
echo "############# PHASE B: observer effect (n=$NREP paired) #############"
for REP in $(seq 1 $NREP); do
  for ARM in plain plainaux chunk512 agnostic; do
    if [ $((RANDOM % 2)) = 0 ]; then ORDER="off on"; else ORDER="on off"; fi
    echo "===== rep=$REP arm=$ARM order='$ORDER' ====="
    for ONOFF in $ORDER; do
      PORT=$((PORT_BASE + RANDOM % 100))
      SRVLOG="$OUT/g2holb_${TAG}_srv_${ARM}_${ONOFF}_rep${REP}_${SLURM_JOB_ID}.log"
      JSONL="$OUT/g2holb_${TAG}_${ARM}_${ONOFF}_rep${REP}_${SLURM_JOB_ID}.jsonl"; : > "$JSONL"
      HOLB=""
      [ "$ONOFF" = on ] && HOLB="$OUT/g2holb_${TAG}_${ARM}_rep${REP}_${SLURM_JOB_ID}.holb.jsonl"
      PID=$(boot_server "$ARM" "$PORT" "$SRVLOG" "$HOLB")
      OK=$(wait_ready "$PID" "$PORT")
      echo "rep=$REP arm=$ARM probe=$ONOFF boot_ok=$OK"
      if [ "$OK" != 1 ]; then kill_server "$PID" "$PORT"; continue; fi
      record_flags "$ARM" "$PORT" "${ONOFF}_rep${REP}" "$SRVLOG"
      SEED=$((9000 + 100*REP + RATE))
      # Window sidecar: bench_serving does NOT record arrival timestamps
      # (PREREG section 3 F-D), so the HOLB stream is segmented by wall clock.
      T0=$(python -c 'import time;print(repr(time.time()))')
      python -m sglang.bench_serving --backend sglang --model "$MODEL" --host 127.0.0.1 --port "$PORT" \
        --dataset-name random-ids --random-input-len $INLEN --random-output-len $OUTLEN \
        --random-range-ratio $RANGE_RATIO --num-prompts $NPROMPT --request-rate $RATE \
        --seed $SEED --warmup-requests 8 --output-details --output-file "$JSONL" \
        --tag "rep${REP}_${ARM}_${ONOFF}_r${RATE}" \
        | grep -E "Successful requests|Request throughput|Median TTFT|P99 TTFT|Median TPOT|P99 TPOT|Median E2E"
      T1=$(python -c 'import time;print(repr(time.time()))')
      python - "$WINDOWS" "$TAG" "$ARM" "$ONOFF" "$REP" "$RATE" "$SEED" "$T0" "$T1" "${HOLB:-}" <<'PYEOF'
import json, sys
p, tag, arm, onoff, rep, rate, seed, t0, t1, holb = sys.argv[1:11]
with open(p, "a") as f:
    f.write(json.dumps({"tag": tag, "arm": arm, "probe": onoff, "rep": int(rep),
                        "rate": float(rate), "seed": int(seed),
                        "t_start": float(t0), "t_end": float(t1),
                        "holb_path": holb}) + "\n")
PYEOF
      kill_server "$PID" "$PORT"
    done
  done
done
echo "############# PHASE B DONE #############"

# ---------------------------------------------------------------------------
# Instrumentation validity self-checks (reported, not scored here)
# ---------------------------------------------------------------------------
echo "=== HOLB self-checks (final holb_summary per ON run) ==="
for f in "$OUT"/g2holb_${TAG}_*_${SLURM_JOB_ID}.holb.jsonl; do
  [ -e "$f" ] || continue
  python - "$f" <<'PYEOF'
import json, sys
last = None
for line in open(sys.argv[1]):
    r = json.loads(line)
    if r.get("event") == "holb_summary":
        last = r
if last is None:
    print(f"{sys.argv[1]}: NO SUMMARY"); raise SystemExit
n = max(1, last["n_decode_fw"])
print(f"{sys.argv[1].split('/')[-1]}: "
      f"n_decode_fw={last['n_decode_fw']} "
      f"timeline_over_host={last['timeline_over_host']} "
      f"invalid_frac={last['n_invalid_gaps']/n:.5f} "
      f"inconsistent={last['n_inconsistent_gaps']} "
      f"dropped_spans={last['dropped_spans']} n_errors={last['n_errors']} "
      f"n_mixed={last['n_mixed']} modes={last['mode_counts']}")
PYEOF
done
echo "GATE G4 criteria (pre-registered, DIRECT_BLOCKING_DESIGN.md 4.2/4.4):"
echo "  |timeline_over_host - 1| <= 0.02 ; n_invalid/n_decode_fw <= 0.01 ; dropped_spans == 0"

# ---------------------------------------------------------------------------
# Official G3/G4/G5 scoring (this is the artifact experiment-runner reports on)
# ---------------------------------------------------------------------------
echo "############# GATE 2 HOLB SCORING (PDMUX_HOLB_PATH observer-effect gate) #############"
python3 "$OUT/g2_holb_analyze.py" --outdir "$OUT" --tag "$TAG" --jobid "$SLURM_JOB_ID" \
  --arms plain plainaux chunk512 agnostic --nrep "$NREP" \
  | tee "$OUT/g2holb_result_${TAG}_${SLURM_JOB_ID}.txt"

echo "DONE_G2HOLB_${TAG}"
