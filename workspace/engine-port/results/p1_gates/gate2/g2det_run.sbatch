#!/bin/bash
#SBATCH --job-name=g2det
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --comment=pytorch
#SBATCH --chdir=/scratch/ehmoon/whlee/prefill-layer-alloc
#SBATCH --output=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/p1_gates/gate2/%x_%j.out
#SBATCH --error=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/p1_gates/gate2/%x_%j.err
#
# ==============================================================================
# T4-1: reduction-order attribution probe for the G-2 reproducibility defect
# (task brief 2026-08-09, experiment-runner). Structural reference:
# PREREG_G2CTRL_2026-08-07.md (this script reuses its arm_flags/prompt
# generator/self+cross-baseline machinery essentially verbatim, adding a
# DETMODE={off,on} axis that toggles --enable-deterministic-inference).
#
# WHY: CONSENSUS.md rev16/rev17 G-2 block -- Granite-4.0-h-micro-base +
# --chunked-prefill-size 512 loses bit-level reproducibility at 16-concurrent
# greedy (2/16 requests, deterministic divergence at repeat-unit token
# 62<->322, period-3, 31 positions, output length 96 identical both variants,
# reproduced independently 3x: jobs 875346/875611/875661). Classified as a
# REPRODUCIBILITY defect, not correctness (negative control passed: plain/
# plainaux/agnostic all 0/48 each time). Inconsistent with the mamba-state-
# carryover hypothesis (that would predict divergence from generation step 0
# with a different continuation, not a preserved period-3 repeat structure).
# Hypothesis under test here: reduction-order non-determinism in the
# flashinfer/matmul kernels. If true, --enable-deterministic-inference
# (batch_invariant_ops) should eliminate it.
#
# THIS RUN DOES NOT MEASURE PERFORMANCE. --enable-deterministic-inference
# changes throughput/latency materially (forces sampling_backend=pytorch,
# disables piecewise cuda graph, etc) -- this is a correctness/reproducibility
# arm, not a performance arm. Do not cite timings from this job.
#
# --------------------------------------------------------------------------
# STEP 1 (mandatory, cheap): PRELIMINARY PROBE. Before running anything else,
# boot Granite with chunk512 + --enable-deterministic-inference and confirm
# via A TRIPLE WITNESS that the flag (a) exists in this build, (b) is
# actually realized (not silently ignored), and (c) is compatible with
# regular (non-piecewise) CUDA graph capture -- the operating point this
# project requires (CLAUDE.md: "운영점 = cudagraph-ON"):
#   witness 1: /server_info top-level (tokenizer-manager's copy of
#              dataclasses.asdict(server_args)) -- enable_deterministic_inference,
#              disable_cuda_graph, chunked_prefill_size.
#   witness 2: /server_info -> internal_states[0] (scheduler process's own
#              live vars(get_global_server_args())) -- same fields,
#              independent process (per E-A probe methodology,
#              g2ea_mixed_chunk_probe.sbatch comment block).
#   witness 3: server stdout/stderr log -- "Capture cuda graph begin/end"
#              lines present (regular decode-graph capture actually ran, did
#              not silently fall back to eager) AND boot succeeded.
# If ANY witness fails (flag absent, not realized, or incompatible with
# cudagraph capture / boot), this job STOPS and reports NOT_APPLICABLE. Per
# the task brief: that is NOT a gate failure, it is "적용 불가" -- do not
# relabel a measurement/harness inapplicability as a policy verdict (this
# mislabeling has recurred 6x in this project per the task brief's framing).
#
# STEP 2: MAIN RUN. Reuses the exact G2CTRL harness (arm_flags/prompt
# generator/self+cross-baseline machinery, PREREG_G2CTRL_2026-08-07.md) for
# Granite-4.0-h-micro-base only, all 4 arms (plain/plainaux/chunk512/
# agnostic) preserved as the negative-control structure, run twice:
# DETMODE=off (bit-for-bit reproduction of g2ctrl_875611's server flags --
# must reproduce the original failure or the decision rule below is moot)
# and DETMODE=on (adds --enable-deterministic-inference to every arm's
# flags). Same prompt (seed 4242, target 2000 tokens), same n_concurrent=16,
# same 3 independent reps per arm per detmode (>= the original 3 reps that
# produced the failure, per the task brief's "동등 이상" instruction).
#
# --------------------------------------------------------------------------
# DECISION RULE (fixed BEFORE this job runs -- do not revise after seeing
# results; mirrors PREREG_G2CTRL section 2's discipline and CLAUDE.md
# methodology gate #8):
#
#   OFF reproduces the original failure (chunk512 self-mismatch count >= 1
#   across the 3 reps, 48 trials)
#     AND ON has ZERO self-mismatches in all 3 reps (48/48 match)
#       => CONFIRMED: reduction-order attribution. Recommend lifting the
#          inheritance-discard rule that voided the Gate 2 rev4 Granite
#          cells and the E-A cmp2 Granite cells (subject to doc-steward /
#          claims-auditor review -- this script only recommends).
#
#   ON still shows >= 1 self-mismatch in any rep
#       => REFUTED: reduction-order is not the (sole) cause. Keep the
#          discard rule.
#
#   OFF does NOT reproduce the failure (0 self-mismatches in OFF's chunk512
#   arm across all 3 reps)
#       => NO VERDICT. The original failure's preconditions were not
#          reproduced in this boot; the ON-arm result is uninterpretable and
#          must not be read as evidence either way (report OFF's non-
#          reproduction as the primary finding instead).
#
# No performance headline. No direct edits to CONSENSUS.md/PROJECT_STATUS.md
# from this job (doc-steward's decision how/whether to fold this in).
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

echo "=== DECISION RULE (fixed pre-run, see sbatch header comment for full text) ==="
echo "OFF reproduces (chunk512 self-mismatch>=1/48) AND ON=0/48 in all 3 reps => CONFIRMED (reduction-order)"
echo "ON still >=1/48 mismatch => REFUTED (keep discard rule)"
echo "OFF does NOT reproduce (0/48) => NO VERDICT (precondition not reproduced, ON uninterpretable)"
echo "=== END DECISION RULE ==="

module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0
export PS1="" PATH="$HOME/.local/bin:$PATH"
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate
cd /scratch/ehmoon/whlee/prefill-layer-alloc

MODEL="${1:?usage: g2det_run.sbatch MODEL CTX BACKEND TAG}"
CTX="${2:?ctx}"
BACKEND="${3:?backend}"
TAG="${4:?tag}"

OUT=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/p1_gates/gate2
mkdir -p "$OUT"
export HF_HOME=/scratch/ehmoon/whlee/prefill-layer-alloc/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="$OUT/.triton_cache_${SLURM_JOB_ID}"
mkdir -p "$TRITON_CACHE_DIR"
# HOLB OFF (matches g2ctrl, the harness this reuses -- the original failure
# reproduced with HOLB OFF in job 875611, so HOLB presence is not a
# precondition for the phenomenon under test here).
unset PDMUX_HOLB_PATH PDMUX_HOLB_RUN_ID PDMUX_HOLB_WORKLOAD_ID

echo "=== env sync ==="
MANIFEST="$OUT/runtime_source_manifest_g2det_${TAG}_${SLURM_JOB_ID}.sha256"
workspace/engine-port/scripts/bootstrap/sync_engine_tree.sh "$MANIFEST"
echo "manifest -> $MANIFEST"
diff <(sort workspace/engine-port/results/p1_gates/gate1/reference_manifest_873944.sha256) \
     <(sort "$MANIFEST") > "$OUT/manifest_diff_g2det_${TAG}_${SLURM_JOB_ID}.txt"
echo "--- manifest diff vs 873944 reference (expect exactly 2 '>' lines, 0 '<' lines, same as g2ctrl_875611/g2ea_875661) ---"
cat "$OUT/manifest_diff_g2det_${TAG}_${SLURM_JOB_ID}.txt"
N_ADDED=$(grep -c '^>' "$OUT/manifest_diff_g2det_${TAG}_${SLURM_JOB_ID}.txt" || true)
N_REMOVED=$(grep -c '^<' "$OUT/manifest_diff_g2det_${TAG}_${SLURM_JOB_ID}.txt" || true)
if [ "$N_ADDED" = "2" ] && [ "$N_REMOVED" = "0" ]; then
  echo "MANIFEST_CHECK=PASS (added=$N_ADDED removed=$N_REMOVED)"
else
  echo "MANIFEST_CHECK=FAIL (added=$N_ADDED removed=$N_REMOVED, expected added=2 removed=0)" >&2
  echo "MANIFEST_CHECK_ABORT -- engine tree has drifted from the verified state, refusing to run on an unverified tree" >&2
  exit 4
fi

# ------------------------------------------------------------------------
# Config (identical to PREREG_G2CTRL_2026-08-07.md sections 1, 3 -- same
# generator/n/outlen so the OFF arm is a faithful reproduction attempt)
# ------------------------------------------------------------------------
INLEN=2000
OUTLEN=96
NREP_CTRL=3
G2CTRL_NCONCURRENT=16
PDMUX_CFG="$OUT/pdmux_a100_smoke.yml"
ARMS="plain plainaux chunk512 agnostic"   # fixed order -- A1 must precede A3 (cross baseline)

PORT_BASE=$((35000 + SLURM_JOB_ID % 400))

# ------------------------------------------------------------------------
# Helpers. arm_flags/boot_server/wait_ready/capture_evidence/kill_server/
# record_flags/capture_self_baseline mirror g2ctrl_run.sbatch verbatim,
# except arm_flags reads the global DETMODE var and appends
# --enable-deterministic-inference when DETMODE=on.
# ------------------------------------------------------------------------
DETMODE="off"   # set to on/off by the outer loop; global, read by arm_flags

arm_flags () {   # $1=arm -> echoes extra server flags (reads global DETMODE)
  local ARM="$1" BASE
  case "$ARM" in
    plain)     BASE="" ;;
    plainaux)  BASE="--chunked-prefill-size -1 --disable-overlap-schedule" ;;
    chunk512)  BASE="--chunked-prefill-size 512" ;;
    agnostic)  BASE="--enable-pdmux --pdmux-config-path $PDMUX_CFG --chunked-prefill-size -1 --disable-overlap-schedule" ;;
    *) BASE="UNKNOWN_ARM_$ARM" ;;
  esac
  if [ "$DETMODE" = "on" ]; then
    echo "$BASE --enable-deterministic-inference"
  else
    echo "$BASE"
  fi
}

boot_server () {   # $1=arm $2=port $3=srvlog -> echoes pid
  local ARM="$1" PORT="$2" SRVLOG="$3"
  local EXTRA; EXTRA="$(arm_flags "$ARM")"
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
}

kill_server () {  # $1=pid $2=port
  local PID="$1" PORT="$2"
  kill "$PID" 2>/dev/null
  pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null
  sleep 3
}

record_flags () {  # $1=arm $2=port
  local ARM="$1" PORT="$2"
  curl -s "http://127.0.0.1:$PORT/server_info" \
    > "$OUT/g2det_${TAG}_flags_${DETMODE}_${ARM}_${SLURM_JOB_ID}.json" || true
}

# capture_self_baseline PORT BASELINE_JSON RAW_DUMP
#   -> stdout: OK | TIMEOUT | ERROR | BUILD_FAIL   (exactly one token)
# Anything other than OK means "this arm produced NO measurement", which is a
# different event from "this arm was measured and disagreed". The token is
# echoed verbatim into the summary row as self_baseline_status so the analyzer
# can tell them apart (methodology gate #21 -- do not label a measurement
# failure as a gate failure). TIMEOUT specifically is the job-876699 hang class.
capture_self_baseline () {
  local PORT="$1" BASEJSON="$2" DUMP="$3"
  unset STATUS HTTP_CODE METHOD SHA N_OUTPUT_IDS DETAIL
  eval "$(greedy_call "$PORT" "$DUMP")"
  echo "BASELINE_CALL status=$STATUS http=${HTTP_CODE:-} method=${METHOD:-} sha=${SHA:-} n=${N_OUTPUT_IDS:-} detail=${DETAIL:-}" >&2
  if [ "$STATUS" != "OK" ]; then
    # Propagate the distinguishing status (TIMEOUT vs ERROR) rather than
    # collapsing every non-OK outcome to the word "FAIL".
    echo "$STATUS"
    return
  fi
  python3 - "$DUMP" "$BASEJSON" "$SHA" <<'PYEOF'
import json, sys, hashlib
dump_path, out_path, expect_sha = sys.argv[1:4]
d = json.load(open(dump_path))
ids = d.get("output_ids")
if not ids:
    print("BASELINE_BUILD_FAIL: no output_ids in dump", file=sys.stderr)
    sys.exit(1)
sha = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
if sha != expect_sha:
    print(f"BASELINE_SHA_MISMATCH computed={sha} expected={expect_sha}", file=sys.stderr)
    sys.exit(1)
json.dump({"sha": sha, "output_ids": ids}, open(out_path, "w"))
print(f"BASELINE_BUILD_OK sha={sha} n_ids={len(ids)}", file=sys.stderr)
PYEOF
  if [ $? -eq 0 ]; then echo OK; else echo BUILD_FAIL; fi
}

source "$OUT/g2_holb_phaseA_lib.sh"   # build_g2holb_prompt, greedy_call (unmodified, reused)

# ==========================================================================
# STEP 1: PRELIMINARY PROBE (Granite + chunk512 + --enable-deterministic-
# inference, the critical combo). Cheap: one boot, no reps, no bench.
# ==========================================================================
echo "############# STEP 1: PRELIMINARY PROBE (model=$MODEL) #############"
DETMODE="on"
PROBE_PORT=$((PORT_BASE + 200))
PROBE_SRVLOG="$OUT/g2det_${TAG}_probe_srv_${SLURM_JOB_ID}.log"
PROBE_PID=$(boot_server chunk512 "$PROBE_PORT" "$PROBE_SRVLOG")
PROBE_OK=$(wait_ready "$PROBE_PID" "$PROBE_PORT")
echo "probe boot_ok=$PROBE_OK (pid=$PROBE_PID port=$PROBE_PORT)"
capture_evidence "$PROBE_SRVLOG"

if [ "$PROBE_OK" != 1 ]; then
  echo "PROBE_RESULT=NOT_APPLICABLE reason=BOOT_FAILED"
  echo "This is a harness/measurement inapplicability, NOT a gate failure --"
  echo "--enable-deterministic-inference + Granite + chunk512 + cudagraph-ON"
  echo "did not boot in this build. Stopping here per task brief instruction"
  echo "(do not relabel inapplicability as a policy verdict)."
  tail -60 "$PROBE_SRVLOG" | grep -viE "pynvml|Future"
  kill_server "$PROBE_PID" "$PROBE_PORT" 2>/dev/null || true
  echo "DONE_G2DET_PROBE_NOT_APPLICABLE"
  exit 42
fi

PROBE_INFO="$OUT/g2det_${TAG}_probe_serverinfo_${SLURM_JOB_ID}.json"
curl -s "http://127.0.0.1:$PROBE_PORT/server_info" > "$PROBE_INFO"

echo "--- witness 1+2: /server_info top-level + internal_states[0] ---"
WITNESS_OUT="$OUT/g2det_${TAG}_probe_witness_${SLURM_JOB_ID}.json"
python3 - "$PROBE_INFO" "$WITNESS_OUT" <<'PYEOF'
import json, sys
info_path, out_path = sys.argv[1], sys.argv[2]
d = json.load(open(info_path))
report = {}
ok = True

for k in ("enable_deterministic_inference", "disable_cuda_graph",
          "attention_backend", "chunked_prefill_size", "sampling_backend"):
    report[f"top.{k}"] = d.get(k)
if d.get("enable_deterministic_inference") is not True:
    ok = False
if d.get("disable_cuda_graph") is not False:
    ok = False
if d.get("chunked_prefill_size") != 512:
    ok = False

states = d.get("internal_states") or []
if states:
    s0 = states[0]
    for k in ("enable_deterministic_inference", "disable_cuda_graph", "chunked_prefill_size"):
        report[f"internal_states0.{k}"] = s0.get(k)
    if s0.get("enable_deterministic_inference") is not True:
        ok = False
    if s0.get("disable_cuda_graph") is not False:
        ok = False
else:
    report["internal_states0"] = "MISSING"
    ok = False

report["witness12_ok"] = ok
json.dump(report, open(out_path, "w"), indent=2)
print(json.dumps(report, indent=2))
PYEOF
W12=$(python3 -c "
import json
d = json.load(open('$WITNESS_OUT'))
print('1' if d.get('witness12_ok') else '0')
")
echo "witness1+2_ok=$W12"

echo "--- witness 3: server log confirms regular (non-piecewise) cuda graph capture ran ---"
W3_CAPTURE_BEGIN=$(grep -c "Capture cuda graph begin" "$PROBE_SRVLOG" || true)
W3_CAPTURE_END=$(grep -c "Capture cuda graph end" "$PROBE_SRVLOG" || true)
W3_DET_MENTION=$(grep -ciE "deterministic inference" "$PROBE_SRVLOG" || true)
echo "witness3: capture_begin_lines=$W3_CAPTURE_BEGIN capture_end_lines=$W3_CAPTURE_END deterministic_log_mentions=$W3_DET_MENTION"
W3_OK=0
if [ "${W3_CAPTURE_BEGIN:-0}" -ge 1 ] && [ "${W3_CAPTURE_END:-0}" -ge 1 ]; then
  W3_OK=1
fi

kill_server "$PROBE_PID" "$PROBE_PORT"

echo "=== PROBE VERDICT: witness1+2_ok=$W12 witness3_ok=$W3_OK ==="
if [ "$W12" != "1" ] || [ "$W3_OK" != "1" ]; then
  echo "PROBE_RESULT=NOT_APPLICABLE reason=WITNESS_FAILED"
  echo "--enable-deterministic-inference either did not realize as expected"
  echo "(witness1+2) or regular cuda graph capture did not proceed under it"
  echo "(witness3) for Granite+chunk512 in this build. This is NOT a gate"
  echo "failure -- it means the ON arm cannot be tested at the required"
  echo "cudagraph-ON operating point. Stopping here; do not proceed to the"
  echo "main run, and do not interpret this as a reduction-order verdict."
  echo "DONE_G2DET_PROBE_NOT_APPLICABLE"
  exit 43
fi
echo "PROBE_RESULT=PASS -- --enable-deterministic-inference exists, is realized"
echo "(3-witness: /server_info top, internal_states[0], server log), and is"
echo "compatible with regular cuda-graph capture for Granite+chunk512."
echo "Proceeding to STEP 2 (main run)."
echo "############# END STEP 1 #############"

# ==========================================================================
# STEP 2: MAIN RUN. Prompt build (once, shared across both DETMODE passes --
# identical to g2ctrl_run.sbatch / PREREG_G2CTRL section 1).
# ==========================================================================
echo "############# STEP 2: MAIN RUN (model=$MODEL tag=$TAG) #############"
PROMPT_TXT="$OUT/g2det_prompt_${TAG}_${SLURM_JOB_ID}.txt"
PROMPT_META="$OUT/g2det_prompt_meta_${TAG}_${SLURM_JOB_ID}.json"

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

for DETMODE in off on; do
  echo "========== DETMODE=$DETMODE =========="
  SUMMARY_JSON="$OUT/g2det_${TAG}_summary_${DETMODE}_${SLURM_JOB_ID}.json"; : > "$SUMMARY_JSON"
  CROSS_A1_BASELINE=""

  for ARM in $ARMS; do
    PORT=$((PORT_BASE + RANDOM % 100))
    SRVLOG="$OUT/g2det_${TAG}_srv_${DETMODE}_${ARM}_${SLURM_JOB_ID}.log"
    echo "===== detmode=$DETMODE arm=$ARM port=$PORT ====="
    PID=$(boot_server "$ARM" "$PORT" "$SRVLOG")
    OK=$(wait_ready "$PID" "$PORT")
    echo "detmode=$DETMODE arm=$ARM boot_ok=$OK"
    capture_evidence "$SRVLOG"
    if [ "$OK" != 1 ]; then
      echo "BOOT FAILED detmode=$DETMODE arm=$ARM"; tail -30 "$SRVLOG" | grep -viE "pynvml|Future"
      printf '%s\n' "{\"detmode\": \"$DETMODE\", \"arm\": \"$ARM\", \"boot_ok\": false}" >> "$SUMMARY_JSON"
      kill_server "$PID" "$PORT"
      continue
    fi
    record_flags "$ARM" "$PORT"

    BASEJSON="$OUT/g2det_${TAG}_${DETMODE}_${ARM}_selfbaseline_${SLURM_JOB_ID}.json"
    RAWDUMP="$OUT/g2det_${TAG}_${DETMODE}_${ARM}_selfbaseline_raw_${SLURM_JOB_ID}.json"
    BSTATUS=$(capture_self_baseline "$PORT" "$BASEJSON" "$RAWDUMP")
    echo "detmode=$DETMODE arm=$ARM self-baseline build=$BSTATUS"
    if [ "$BSTATUS" != OK ]; then
      # NO DATA for this arm. self_baseline_status distinguishes TIMEOUT (the
      # 876699 hang class) / ERROR / BUILD_FAIL. g2det_analyze.py must score
      # this as absence-of-measurement, never as a mismatch and never as
      # "ON is not clean" (methodology gate #21).
      if [ "$BSTATUS" = TIMEOUT ]; then
        echo "detmode=$DETMODE arm=$ARM SELF_BASELINE_TIMEOUT -- greedy_call exceeded" \
             "G2_GREEDY_MAX_TIME=${G2_GREEDY_MAX_TIME:-180}s. MEASUREMENT FAILURE, not a" \
             "correctness verdict. Skipping this arm and continuing with the next one." >&2
      fi
      printf '%s\n' "{\"detmode\": \"$DETMODE\", \"arm\": \"$ARM\", \"boot_ok\": true, \"self_baseline_ok\": false, \"self_baseline_status\": \"$BSTATUS\"}" >> "$SUMMARY_JSON"
      kill_server "$PID" "$PORT"
      continue
    fi
    if [ "$ARM" = plain ]; then
      CROSS_A1_BASELINE="$OUT/g2det_${TAG}_${DETMODE}_A1_baseline_${SLURM_JOB_ID}.json"
      cp "$BASEJSON" "$CROSS_A1_BASELINE"
      echo "detmode=$DETMODE arm=plain baseline saved as cross-baseline for chunk512 -> $CROSS_A1_BASELINE"
    fi

    REP_JSONS=""
    for REP in $(seq 1 $NREP_CTRL); do
      OUTJSON="$OUT/g2det_${TAG}_${DETMODE}_${ARM}_rep${REP}_${SLURM_JOB_ID}.json"
      CROSS_ARG=""
      if [ "$ARM" = chunk512 ] && [ -n "$CROSS_A1_BASELINE" ] && [ -f "$CROSS_A1_BASELINE" ]; then
        CROSS_ARG="--cross-baseline-file $CROSS_A1_BASELINE"
      fi
      echo "----- detmode=$DETMODE arm=$ARM rep=$REP n_concurrent=$G2CTRL_NCONCURRENT -----"
      python3 "$OUT/g2ctrl_gate.py" --host 127.0.0.1 --port "$PORT" --prompt-file "$PROMPT_TXT" \
        --max-new-tokens "$OUTLEN" --n-concurrent "$G2CTRL_NCONCURRENT" \
        --self-baseline-file "$BASEJSON" $CROSS_ARG --out "$OUTJSON"
      REP_JSONS="$REP_JSONS $OUTJSON"
      if grep -qiE "AssertionError|assert mid is not None" "$SRVLOG"; then
        echo "MAMBA POOL ASSERT DETECTED in $SRVLOG (detmode=$DETMODE arm=$ARM rep=$REP)"
      fi
    done
    kill -0 "$PID" 2>/dev/null || echo "WARNING: server (detmode=$DETMODE arm=$ARM) died at some point during the 3 reps -- inspect $SRVLOG"
    kill_server "$PID" "$PORT"

    python3 - "$SUMMARY_JSON" "$DETMODE" "$ARM" "$BASEJSON" $REP_JSONS <<'PYEOF'
import json, sys
summary_path, detmode, arm, basejson, *rep_jsons = sys.argv[1:]
base = json.load(open(basejson))
reps = [json.load(open(p)) for p in rep_jsons]
row = {
    "detmode": detmode, "arm": arm, "boot_ok": True, "self_baseline_ok": True,
    "self_baseline_sha": base["sha"],
    "reps": [
        {
            "n_ok": r["self"]["n_ok"], "n_match_self": r["self"]["n_match"],
            "n_mismatch_self": r["self"]["n_mismatch"], "verdict_self": r["self"]["verdict"],
            "cross": (
                {
                    "n_ok": r["cross"]["n_ok"], "n_match_cross": r["cross"]["n_match"],
                    "n_mismatch_cross": r["cross"]["n_mismatch"], "verdict_cross": r["cross"]["verdict"],
                } if r.get("cross") else None
            ),
            "self_mismatch_positions": r["self"]["mismatches"],
            "cross_mismatch_positions": (r["cross"]["mismatches"] if r.get("cross") else None),
        }
        for r in reps
    ],
}
with open(summary_path, "a") as f:
    f.write(json.dumps(row) + "\n")
PYEOF
  done

  echo "----- DETMODE=$DETMODE summary -----"
  cat "$SUMMARY_JSON"
done

echo "############# G2DET DONE (model=$MODEL tag=$TAG) #############"
echo "summaries -> $OUT/g2det_${TAG}_summary_off_${SLURM_JOB_ID}.json , $OUT/g2det_${TAG}_summary_on_${SLURM_JOB_ID}.json"
echo "DONE_G2DET_${TAG}"
