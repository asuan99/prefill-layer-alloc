#!/bin/bash
#SBATCH --job-name=g2ctrl
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
# G-2 negative-control campaign. Spec = PREREG_G2CTRL_2026-08-07.md.
# Purpose: job 875346's G-2 (16-concurrent chunk512 vs A1 batch=1) FAILED
# (n_mismatch=2/16); the original gate has no negative control, so it cannot
# tell a chunk512-specific mamba-state-carryover defect (a) apart from
# ordinary batch-composition numerical non-determinism present in every arm
# (b). This job adds a SELF baseline (each arm's own batch=1 greedy output)
# for all four arms, plus the original CROSS baseline (vs A1) for chunk512
# only, repeated 3x. No bench_serving / performance measurement at all --
# correctness/availability only. HOLB probe is OFF (unlike 875346).
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

MODEL="${1:?usage: g2ctrl_run.sbatch MODEL CTX BACKEND TAG}"
CTX="${2:?ctx}"
BACKEND="${3:?backend}"
TAG="${4:?tag}"

OUT=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/p1_gates/gate2
mkdir -p "$OUT"
export HF_HOME=/scratch/ehmoon/whlee/prefill-layer-alloc/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="$OUT/.triton_cache_${SLURM_JOB_ID}"
mkdir -p "$TRITON_CACHE_DIR"
# HOLB OFF for this campaign (unlike 875344/875346) -- do not set
# PDMUX_HOLB_PATH / PDMUX_HOLB_RUN_ID / PDMUX_HOLB_WORKLOAD_ID anywhere below.
unset PDMUX_HOLB_PATH PDMUX_HOLB_RUN_ID PDMUX_HOLB_WORKLOAD_ID

echo "=== env sync ==="
MANIFEST="$OUT/runtime_source_manifest_g2ctrl_${TAG}_${SLURM_JOB_ID}.sha256"
workspace/engine-port/scripts/bootstrap/sync_engine_tree.sh "$MANIFEST"
echo "manifest -> $MANIFEST"
diff <(sort workspace/engine-port/results/p1_gates/gate1/reference_manifest_873944.sha256) \
     <(sort "$MANIFEST") > "$OUT/manifest_diff_g2ctrl_${TAG}_${SLURM_JOB_ID}.txt"
echo "--- manifest diff vs 873944 reference (expect exactly 2 '>' lines, 0 '<' lines -- HOLB probe files present in tree even though PDMUX_HOLB_PATH is unset at runtime) ---"
cat "$OUT/manifest_diff_g2ctrl_${TAG}_${SLURM_JOB_ID}.txt"
N_ADDED=$(grep -c '^>' "$OUT/manifest_diff_g2ctrl_${TAG}_${SLURM_JOB_ID}.txt" || true)
N_REMOVED=$(grep -c '^<' "$OUT/manifest_diff_g2ctrl_${TAG}_${SLURM_JOB_ID}.txt" || true)
if [ "$N_ADDED" = "2" ] && [ "$N_REMOVED" = "0" ]; then
  echo "MANIFEST_CHECK=PASS (added=$N_ADDED removed=$N_REMOVED)"
else
  echo "MANIFEST_CHECK=FAIL (added=$N_ADDED removed=$N_REMOVED, expected added=2 removed=0)" >&2
  echo "MANIFEST_CHECK_ABORT -- engine tree has drifted from the 873944/HOLB-verified state, refusing to run on an unverified tree" >&2
  exit 4
fi

# ------------------------------------------------------------------------
# Config (PREREG_G2CTRL_2026-08-07.md sections 1, 3)
# ------------------------------------------------------------------------
INLEN=2000
OUTLEN=96
NREP_CTRL=3
G2CTRL_NCONCURRENT=16
PDMUX_CFG="$OUT/pdmux_a100_smoke.yml"
ARMS="plain plainaux chunk512 agnostic"   # fixed order -- A1 must precede A3 (cross baseline)

PORT_BASE=$((34000 + SLURM_JOB_ID % 400))

# ------------------------------------------------------------------------
# Helpers (arm_flags/boot_server/wait_ready/capture_evidence/kill_server
# mirror g2_run.sbatch verbatim except HOLB is not wired into boot_server)
# ------------------------------------------------------------------------
arm_flags () {   # $1=arm -> echoes extra server flags (identical to g2_run.sbatch)
  case "$1" in
    plain)     echo "" ;;
    plainaux)  echo "--chunked-prefill-size -1 --disable-overlap-schedule" ;;
    chunk512)  echo "--chunked-prefill-size 512" ;;
    agnostic)  echo "--enable-pdmux --pdmux-config-path $PDMUX_CFG --chunked-prefill-size -1 --disable-overlap-schedule" ;;
    *) echo "UNKNOWN_ARM_$1" ;;
  esac
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
  curl -s "http://127.0.0.1:$PORT/get_server_info" \
    > "$OUT/g2ctrl_${TAG}_flags_${ARM}_${SLURM_JOB_ID}.json" || true
}

# capture_self_baseline PORT BASELINE_JSON RAW_DUMP -> stdout: OK|FAIL (only token)
capture_self_baseline () {
  local PORT="$1" BASEJSON="$2" DUMP="$3"
  unset STATUS HTTP_CODE METHOD SHA N_OUTPUT_IDS DETAIL
  eval "$(greedy_call "$PORT" "$DUMP")"
  echo "BASELINE_CALL status=$STATUS http=${HTTP_CODE:-} method=${METHOD:-} sha=${SHA:-} n=${N_OUTPUT_IDS:-} detail=${DETAIL:-}" >&2
  if [ "$STATUS" != "OK" ]; then
    echo FAIL
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
  if [ $? -eq 0 ]; then echo OK; else echo FAIL; fi
}

# ==========================================================================
# Prompt build (identical generator to G-1/G-2, PREREG_G2CTRL section 1)
# ==========================================================================
echo "############# G2CTRL: NEGATIVE-CONTROL CORRECTNESS GATE (model=$MODEL tag=$TAG) #############"
PROMPT_TXT="$OUT/g2ctrl_prompt_${TAG}_${SLURM_JOB_ID}.txt"
PROMPT_META="$OUT/g2ctrl_prompt_meta_${TAG}_${SLURM_JOB_ID}.json"
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

SUMMARY_JSON="$OUT/g2ctrl_${TAG}_summary_${SLURM_JOB_ID}.json"; : > "$SUMMARY_JSON"
CROSS_A1_BASELINE=""

for ARM in $ARMS; do
  PORT=$((PORT_BASE + RANDOM % 100))
  SRVLOG="$OUT/g2ctrl_${TAG}_srv_${ARM}_${SLURM_JOB_ID}.log"
  echo "===== arm=$ARM port=$PORT ====="
  PID=$(boot_server "$ARM" "$PORT" "$SRVLOG")
  OK=$(wait_ready "$PID" "$PORT")
  echo "arm=$ARM boot_ok=$OK"
  capture_evidence "$SRVLOG"
  if [ "$OK" != 1 ]; then
    echo "BOOT FAILED arm=$ARM"; tail -30 "$SRVLOG" | grep -viE "pynvml|Future"
    printf '%s\n' "{\"arm\": \"$ARM\", \"boot_ok\": false}" >> "$SUMMARY_JSON"
    kill_server "$PID" "$PORT"
    continue
  fi
  record_flags "$ARM" "$PORT"

  BASEJSON="$OUT/g2ctrl_${TAG}_${ARM}_selfbaseline_${SLURM_JOB_ID}.json"
  RAWDUMP="$OUT/g2ctrl_${TAG}_${ARM}_selfbaseline_raw_${SLURM_JOB_ID}.json"
  BSTATUS=$(capture_self_baseline "$PORT" "$BASEJSON" "$RAWDUMP")
  echo "arm=$ARM self-baseline build=$BSTATUS"
  if [ "$BSTATUS" != OK ]; then
    printf '%s\n' "{\"arm\": \"$ARM\", \"boot_ok\": true, \"self_baseline_ok\": false}" >> "$SUMMARY_JSON"
    kill_server "$PID" "$PORT"
    continue
  fi
  if [ "$ARM" = plain ]; then
    CROSS_A1_BASELINE="$OUT/g2ctrl_${TAG}_A1_baseline_${SLURM_JOB_ID}.json"
    cp "$BASEJSON" "$CROSS_A1_BASELINE"
    echo "arm=plain baseline saved as cross-baseline for chunk512 -> $CROSS_A1_BASELINE"
  fi

  REP_JSONS=""
  for REP in $(seq 1 $NREP_CTRL); do
    OUTJSON="$OUT/g2ctrl_${TAG}_${ARM}_rep${REP}_${SLURM_JOB_ID}.json"
    CROSS_ARG=""
    if [ "$ARM" = chunk512 ] && [ -n "$CROSS_A1_BASELINE" ] && [ -f "$CROSS_A1_BASELINE" ]; then
      CROSS_ARG="--cross-baseline-file $CROSS_A1_BASELINE"
    fi
    echo "----- arm=$ARM rep=$REP n_concurrent=$G2CTRL_NCONCURRENT -----"
    python3 "$OUT/g2ctrl_gate.py" --host 127.0.0.1 --port "$PORT" --prompt-file "$PROMPT_TXT" \
      --max-new-tokens "$OUTLEN" --n-concurrent "$G2CTRL_NCONCURRENT" \
      --self-baseline-file "$BASEJSON" $CROSS_ARG --out "$OUTJSON"
    REP_JSONS="$REP_JSONS $OUTJSON"
    if grep -qiE "AssertionError|assert mid is not None" "$SRVLOG"; then
      echo "MAMBA POOL ASSERT DETECTED in $SRVLOG (arm=$ARM rep=$REP)"
    fi
  done
  kill -0 "$PID" 2>/dev/null || echo "WARNING: server (arm=$ARM) died at some point during the 3 reps -- inspect $SRVLOG"
  kill_server "$PID" "$PORT"

  python3 - "$SUMMARY_JSON" "$ARM" "$BASEJSON" $REP_JSONS <<'PYEOF'
import json, sys
summary_path, arm, basejson, *rep_jsons = sys.argv[1:]
base = json.load(open(basejson))
reps = [json.load(open(p)) for p in rep_jsons]
row = {
    "arm": arm, "boot_ok": True, "self_baseline_ok": True,
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

echo "############# G2CTRL DONE (model=$MODEL tag=$TAG) #############"
echo "summary -> $SUMMARY_JSON"
cat "$SUMMARY_JSON"
echo "DONE_G2CTRL_${TAG}"
