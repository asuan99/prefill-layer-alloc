#!/bin/bash
#SBATCH --job-name=g2detB
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --comment="field=efficientai;appl=pytorch"
#SBATCH --chdir=/scratch/ehmoon/whlee/prefill-layer-alloc
#SBATCH --output=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/p1_gates/gate2/%x_%j.out
#SBATCH --error=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/p1_gates/gate2/%x_%j.err
#
# ==============================================================================
# T4-1 RUN B (coordinator brief 2026-08-10, post-876699-TIMEOUT post-mortem).
# Harness fix "A" is done and verified read-only here (g2_holb_phaseA_lib.sh
# greedy_call now bounded: --connect-timeout/--max-time, curl rc=28 ->
# STATUS=TIMEOUT distinct from STATUS=ERROR, no SHA emitted on non-OK so a
# timeout can never be scored as a mismatch, <DUMP>.status.json sidecar.
# g2det_analyze.py's on_clean bug (n_total=0 falling through to REFUTED) is
# also fixed -- NO VERDICT (MEASUREMENT ABSENT) now. Not re-verifying either;
# this job only exercises the fixed greedy_call live.)
#
# SCOPE (do not re-run what already completed cleanly in job 876699):
#   - OFF (plain/plainaux/chunk512/agnostic, 3 reps each) -- COMPLETE in
#     876699, reproduced the original G-2 failure cleanly (chunk512
#     self-mismatch 2/16 x 3 reps = 6/48, cross vs A1 also 6/48; plain/
#     plainaux/agnostic all 0/48). NOT re-run here.
#   - ON plain, ON plainaux -- COMPLETE in 876699 (0/48 each). NOT re-run.
#   - ON chunk512 -- INCOMPLETE in 876699 (self-baseline greedy_call hung
#     ~52 min, unbounded curl, zero scheduler log lines, job hit the 1h
#     SLURM limit). THIS is what this job measures, x3 independent boots.
#   - ON agnostic -- never started in 876699 (job ran out of wall clock
#     before reaching it). THIS job runs it once, for negative-control
#     balance with OFF's completed agnostic arm.
#
# WHY 3 independent boots for ON+chunk512, not 1: a single timeout cannot
# distinguish "this combination deterministically stalls" from "it stalled
# once, sporadically". The decision rule below requires knowing which.
#
# --------------------------------------------------------------------------
# CROSS-JOB MERGE POLICY (fixed BEFORE submission, per coordinator instruction
# -- this is a policy commitment, not a script step; the actual merge happens
# AFTER this job's outcome is known, because which chunk512 data (if any) is
# eligible depends on the reproducibility classification computed below, and
# that cannot be decided before the data exists):
#   1. OFF (all 4 arms) -> from 876699's
#      g2det_granite-40-h-micro-base_summary_off_876699.json, UNCHANGED.
#   2. ON plain, ON plainaux -> from 876699's
#      g2det_granite-40-h-micro-base_summary_on_876699.json, UNCHANGED.
#   3. ON agnostic -> from THIS job's summary (876699 has no row for it).
#   4. ON chunk512 -> from THIS job's summary IF AND ONLY IF the
#      reproducibility classification below is COMPLETE (0/3 timeouts, all 3
#      independent boots produced a scored self-baseline + 3 reps). In that
#      case attempt 1's row is canonical for decision-rule scoring (matches
#      the pre-registered "one boot -> 3 reps" design in
#      PREREG_G2CTRL_2026-08-07.md / g2det_run.sbatch); attempts 2-3 are
#      reported as replication, not pooled into the primary count.
#      If the classification is NOT_APPLICABLE (3/3 TIMEOUT) or SPORADIC
#      (mixed), NO chunk512 row is merged for scoring -- per the decision
#      rule below, this is reported as an applicability/reproducibility
#      finding, not scored.
#   5. Realized as a NEW file
#      (g2det_granite-40-h-micro-base_summary_on_MERGED_876699_<this-job>.json)
#      that concatenates the selected rows with a provenance field per row
#      (source_job). 876699's and this job's own per-job artifacts are never
#      modified -- read-only, cited only (project convention for raw data).
# --------------------------------------------------------------------------
#
# G2_GREEDY_MAX_TIME=60 (exported below, NOT the greedy_call default of 180):
# the harness-fix comment in g2_holb_phaseA_lib.sh records the max observed
# greedy_call latency across all 64 archived dumps (jobs 874602/874628/874633/
# 874635/875344/875346/875610/875611/876699, both models, all 4 arms) as
# 6.605 s. 60 s is >9x that observed maximum, i.e. still generous headroom,
# while bounding a stall to 60 s/attempt instead of 180 s/attempt so 3
# attempts + diagnostics + the agnostic arm comfortably fit a much smaller
# --time than 876699's blind 1h guess.
#
# --time=00:40:00 sizing (measured, not guessed):
#   Expected case (no stall): env sync+diff ~90s (876699: job start 23:26:15
#   -> first probe server_args log line 23:27:00 = 45s, doubled for margin)
#   + prompt/cross-baseline reuse-and-verify ~5s + 3x chunk512 boot-cycle
#   (876699 measured: boot 24-32s, self-baseline when it succeeds ~1s, 3 reps
#   ~20-30s, kill ~3s => ~55-65s/attempt when NOT stalled) + 1x agnostic
#   boot-cycle (~60-70s, matches 876699's off_agnostic/on_plainaux) =>
#   ~90+5+3*65+70 = 360s = 6 min.
#   Plausible case (chunk512 stalls on all 3 attempts): each attempt now
#   costs boot(~30s) + bounded self-baseline(60s, G2_GREEDY_MAX_TIME) +
#   diagnostic snapshots(~15s) + kill(~3s) = ~108s; 3x108=324s + agnostic 70s
#   + setup 95s = ~490s = 8.2 min.
#   Pathological tail (a boot itself fails to become ready, not just the
#   self-baseline call): wait_ready's stock retry count (170x5s=850s/14min)
#   is NOT reused verbatim here -- tightened below to 90x5s=450s/7.5min,
#   still >10x the 24-32s boot time actually observed for this exact combo in
#   876699 (both its probe boot and its main-run boot succeeded on the first
#   try), so this tightening is measurement-based, not a blind cut. Even if
#   ALL 4 boot-cycles in this job (3x chunk512 + 1x agnostic) hit that 7.5 min
#   ceiling: 4*450 + 95 = 1895s = 31.6 min.
#   00:40:00 covers the pathological tail with ~8 min margin, while being
#   1.5x smaller than 876699's blind 1h -- and because every blocking call in
#   this job is now upper-bounded (60s greedy_call, 450s wait_ready), a
#   TIMEOUT of the SBATCH job itself would mean something is blocked outside
#   any of these bounds (e.g. sync_engine_tree.sh's flock) and is itself a
#   reportable finding, not something a bigger --time should paper over.
#
# --------------------------------------------------------------------------
# DECISION RULE (fixed BEFORE this job runs, per coordinator instruction --
# do not revise after seeing results):
#
#   ON+chunk512 TIMEOUT in all 3 independent boot attempts (0 OK / 3 TIMEOUT)
#     => --enable-deterministic-inference is NOT_APPLICABLE to this exact
#        configuration (Granite x chunked-prefill-size 512 x a real
#        multi-chunk prompt). This is NOT a gate failure and NOT a
#        refutation of the reduction-order hypothesis -- it is the finding
#        that T4-1 cannot be answered with this lever on this build. The
#        inheritance-discard rule (Gate 2 rev4 Granite cells, E-A cmp2
#        Granite cells) stays in force, unchanged, not lifted.
#
#   ON+chunk512 self-baseline succeeds in ALL 3 independent attempts
#   (0 TIMEOUT / 3 OK, each followed by its 3 pre-registered reps)
#     => COMPLETE. Score with the original g2det_run.sbatch decision rule
#        (OFF reproduced [already true, from 876699] AND ON all-reps-clean
#        => CONFIRMED; ON shows >=1 self-mismatch in any rep => REFUTED),
#        using attempt 1 as canonical per the merge policy above.
#
#   Mixed (1 or 2 of the 3 attempts TIMEOUT, the rest OK)
#     => SPORADIC. Report the fact itself (which attempts, in which order,
#        succeeded/timed out) and DO NOT SCORE. A reproducibility experiment
#        cannot be scored while sitting on top of a sporadic stall in its own
#        measurement apparatus -- any resulting mismatch count (0 or >0)
#        would be confounded with "did this attempt's environment happen to
#        avoid the stall", which the decision rule was never designed to
#        control for.
#
# No performance headline anywhere in this job's output.
# ==============================================================================
set -uo pipefail

echo "=== HEADER: node placement + concurrent jobs (recorded before module load) ==="
echo "hostname=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "SLURM_NODELIST=${SLURM_NODELIST:-unset}"
echo "--- squeue --all -p amd_a100nv_8 at job start ---"
squeue --all -p amd_a100nv_8 -o "%.10i %.9P %.20j %.8T %.10M %N %.8u" 2>&1 || true
echo "=== END HEADER ==="

echo "=== PRE-REQUISITE (already answered, not re-run here) ==="
echo "STEP-1 triple-witness probe for --enable-deterministic-inference x Granite x"
echo "chunk512 x cudagraph-ON: PASS in job 876699"
echo "(g2det_granite-40-h-micro-base_probe_witness_876699.json,"
echo " enable_deterministic_inference=true realized top+internal_states[0],"
echo " disable_cuda_graph=false, Capture-cuda-graph-begin/end both present)."
echo "Not repeated in this job."
echo "=== END PRE-REQUISITE ==="

echo "=== DECISION RULE (fixed pre-run, see sbatch header comment for full text) ==="
echo "ON+chunk512 3/3 TIMEOUT => NOT_APPLICABLE (not a gate failure, not a refutation, discard rule stays)"
echo "ON+chunk512 3/3 OK      => COMPLETE, score with original rule using attempt 1"
echo "ON+chunk512 mixed       => SPORADIC, report only, DO NOT SCORE"
echo "=== END DECISION RULE ==="

module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0
export PS1="" PATH="$HOME/.local/bin:$PATH"
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate
cd /scratch/ehmoon/whlee/prefill-layer-alloc

MODEL="${1:?usage: g2detB_run.sbatch MODEL CTX BACKEND TAG}"
CTX="${2:?ctx}"
BACKEND="${3:?backend}"
TAG="${4:?tag}"

OUT=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/p1_gates/gate2
mkdir -p "$OUT"
export HF_HOME=/scratch/ehmoon/whlee/prefill-layer-alloc/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="$OUT/.triton_cache_${SLURM_JOB_ID}"
mkdir -p "$TRITON_CACHE_DIR"
unset PDMUX_HOLB_PATH PDMUX_HOLB_RUN_ID PDMUX_HOLB_WORKLOAD_ID

# ---- the harness fix under test: bound every greedy_call to 60s ----
export G2_GREEDY_MAX_TIME=60
export G2_GREEDY_CONNECT_TIMEOUT=10
echo "G2_GREEDY_MAX_TIME=$G2_GREEDY_MAX_TIME G2_GREEDY_CONNECT_TIMEOUT=$G2_GREEDY_CONNECT_TIMEOUT"

echo "=== env sync ==="
MANIFEST="$OUT/runtime_source_manifest_g2detB_${TAG}_${SLURM_JOB_ID}.sha256"
workspace/engine-port/scripts/bootstrap/sync_engine_tree.sh "$MANIFEST"
echo "manifest -> $MANIFEST"
diff <(sort workspace/engine-port/results/p1_gates/gate1/reference_manifest_873944.sha256) \
     <(sort "$MANIFEST") > "$OUT/manifest_diff_g2detB_${TAG}_${SLURM_JOB_ID}.txt"
echo "--- manifest diff vs 873944 reference (expect exactly 2 '>' lines, 0 '<' lines) ---"
cat "$OUT/manifest_diff_g2detB_${TAG}_${SLURM_JOB_ID}.txt"
N_ADDED=$(grep -c '^>' "$OUT/manifest_diff_g2detB_${TAG}_${SLURM_JOB_ID}.txt" || true)
N_REMOVED=$(grep -c '^<' "$OUT/manifest_diff_g2detB_${TAG}_${SLURM_JOB_ID}.txt" || true)
if [ "$N_ADDED" = "2" ] && [ "$N_REMOVED" = "0" ]; then
  echo "MANIFEST_CHECK=PASS (added=$N_ADDED removed=$N_REMOVED)"
else
  echo "MANIFEST_CHECK=FAIL (added=$N_ADDED removed=$N_REMOVED, expected added=2 removed=0)" >&2
  echo "MANIFEST_CHECK_ABORT -- engine tree has drifted, refusing to run on an unverified tree" >&2
  exit 4
fi

INLEN=2000
OUTLEN=96
NREP_CTRL=3
G2CTRL_NCONCURRENT=16
PDMUX_CFG="$OUT/pdmux_a100_smoke.yml"
PORT_BASE=$((36000 + SLURM_JOB_ID % 400))

arm_flags () {   # $1=arm -> echoes extra server flags (reads global DETMODE)
  local ARM="$1" BASE
  case "$ARM" in
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

# wait_ready: tightened from the shared 170x5s(=850s/14min) bound used
# elsewhere in this directory. 90x5s=450s/7.5min is still >10x the 24-32s
# boot time this exact combo (Granite+chunk512+ON, and +agnostic) actually
# took in job 876699 (both its probe boot and main-run boot succeeded on the
# first try) -- a measurement-based tightening, not a blind cut, done so a
# single stuck boot cannot alone consume more than a bounded slice of this
# job's much smaller --time budget (see header sizing comment).
wait_ready () {   # $1=pid $2=port -> echoes 1/0
  local PID="$1" PORT="$2" ok=0
  for i in $(seq 1 90); do
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

# capture_self_baseline PORT BASELINE_JSON RAW_DUMP -> stdout: OK|TIMEOUT|ERROR|BUILD_FAIL
capture_self_baseline () {
  local PORT="$1" BASEJSON="$2" DUMP="$3"
  unset STATUS HTTP_CODE METHOD SHA N_OUTPUT_IDS DETAIL
  eval "$(greedy_call "$PORT" "$DUMP")"
  echo "BASELINE_CALL status=$STATUS http=${HTTP_CODE:-} method=${METHOD:-} sha=${SHA:-} n=${N_OUTPUT_IDS:-} detail=${DETAIL:-}" >&2
  if [ "$STATUS" != "OK" ]; then
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

# diag_snapshot PORT MAIN_PID LABEL -> writes 4 artifact files: procstate,
# nvidia-smi (full), nvidia-smi compute-apps, py-spy-or-note. Cheap (<2s),
# safe to call liberally. py-spy: checked via `command -v`, NOT installed if
# absent (coordinator instruction) -- absence is itself reported.
PYSPY_CHECKED=0
PYSPY_AVAILABLE=0
diag_snapshot () {
  local PORT="$1" PID="$2" LABEL="$3"
  local TS; TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  {
    echo "TIMESTAMP=$TS LABEL=$LABEL PORT=$PORT MAIN_PID=$PID"
    if kill -0 "$PID" 2>/dev/null; then echo "main_pid_alive=1"; else echo "main_pid_alive=0"; fi
    echo "--- ps tree rooted near main pid (sglang/launch_server processes) ---"
    ps -eo pid,ppid,stat,pcpu,pmem,etimes,wchan:20,cmd | grep -iE "sglang|launch_server" | grep -v grep
  } > "$OUT/g2detB_${TAG}_${LABEL}_procstate_${SLURM_JOB_ID}.txt" 2>&1

  nvidia-smi > "$OUT/g2detB_${TAG}_${LABEL}_nvidiasmi_${SLURM_JOB_ID}.txt" 2>&1
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv \
    > "$OUT/g2detB_${TAG}_${LABEL}_nvidiasmi_apps_${SLURM_JOB_ID}.txt" 2>&1

  if [ "$PYSPY_CHECKED" = 0 ]; then
    PYSPY_CHECKED=1
    if command -v py-spy >/dev/null 2>&1; then PYSPY_AVAILABLE=1; fi
  fi
  local PYSPY_OUT="$OUT/g2detB_${TAG}_${LABEL}_pyspy_${SLURM_JOB_ID}.txt"
  if [ "$PYSPY_AVAILABLE" = 1 ]; then
    py-spy dump --pid "$PID" --subprocesses > "$PYSPY_OUT" 2>&1 || \
      echo "py-spy dump attempt failed (see stderr above, e.g. ptrace permission)" >> "$PYSPY_OUT"
  else
    echo "py-spy not available in this environment (checked via 'command -v py-spy', absent) -- skipped per instruction, not installed." > "$PYSPY_OUT"
  fi
}

source "$OUT/g2_holb_phaseA_lib.sh"   # build_g2holb_prompt (unused here), greedy_call (the fixed one)

# ==========================================================================
# Reuse the prompt and the A1 (plain) cross-baseline from job 876699 --
# byte-identical, verified by sha256 before use. Neither 876699 source file
# is modified; both are copied into this job's own tagged files.
# ==========================================================================
echo "############# REUSING PROMPT + A1 CROSS-BASELINE FROM JOB 876699 #############"
SRC_PROMPT="$OUT/g2det_prompt_${TAG}_876699.txt"
SRC_A1="$OUT/g2det_${TAG}_on_A1_baseline_876699.json"
PROMPT_TXT="$OUT/g2detB_prompt_${TAG}_reused_876699_${SLURM_JOB_ID}.txt"
CROSS_A1_BASELINE="$OUT/g2detB_${TAG}_on_A1_baseline_reused_876699_${SLURM_JOB_ID}.json"

if [ ! -f "$SRC_PROMPT" ] || [ ! -f "$SRC_A1" ]; then
  echo "REUSE_ABORT: expected source files from 876699 not found ($SRC_PROMPT / $SRC_A1)" >&2
  exit 6
fi
cp "$SRC_PROMPT" "$PROMPT_TXT"
cp "$SRC_A1" "$CROSS_A1_BASELINE"
SRC_PROMPT_SHA=$(sha256sum "$SRC_PROMPT" | cut -d' ' -f1)
NEW_PROMPT_SHA=$(sha256sum "$PROMPT_TXT" | cut -d' ' -f1)
SRC_A1_SHA=$(sha256sum "$SRC_A1" | cut -d' ' -f1)
NEW_A1_SHA=$(sha256sum "$CROSS_A1_BASELINE" | cut -d' ' -f1)
echo "prompt sha256: src=$SRC_PROMPT_SHA copy=$NEW_PROMPT_SHA"
echo "A1 baseline sha256: src=$SRC_A1_SHA copy=$NEW_A1_SHA"
if [ "$SRC_PROMPT_SHA" != "$NEW_PROMPT_SHA" ] || [ "$SRC_A1_SHA" != "$NEW_A1_SHA" ]; then
  echo "REUSE_ABORT: copy did not preserve byte-identity" >&2
  exit 7
fi
echo "REUSE_OK -- byte-identical to 876699"
echo "############# END REUSE #############"

# ==========================================================================
# ON + chunk512: 3 independent boot attempts.
# ==========================================================================
echo "############# ON + chunk512: 3 INDEPENDENT BOOT ATTEMPTS #############"
DETMODE="on"
SUMMARY_JSON="$OUT/g2detB_${TAG}_summary_on_${SLURM_JOB_ID}.json"; : > "$SUMMARY_JSON"
ATTEMPT_STATUS_1=""; ATTEMPT_STATUS_2=""; ATTEMPT_STATUS_3=""

for ATTEMPT in 1 2 3; do
  echo "===== ATTEMPT $ATTEMPT: detmode=on arm=chunk512 ====="
  PORT=$((PORT_BASE + 50 * ATTEMPT))
  SRVLOG="$OUT/g2detB_${TAG}_srv_on_chunk512_attempt${ATTEMPT}_${SLURM_JOB_ID}.log"
  PID=$(boot_server chunk512 "$PORT" "$SRVLOG")
  OK=$(wait_ready "$PID" "$PORT")
  echo "attempt=$ATTEMPT boot_ok=$OK pid=$PID port=$PORT"
  capture_evidence "$SRVLOG"
  if [ "$OK" != 1 ]; then
    echo "BOOT FAILED attempt=$ATTEMPT"; tail -30 "$SRVLOG" | grep -viE "pynvml|Future"
    printf '%s\n' "{\"detmode\": \"on\", \"arm\": \"chunk512\", \"attempt\": $ATTEMPT, \"boot_ok\": false}" >> "$SUMMARY_JSON"
    eval "ATTEMPT_STATUS_${ATTEMPT}=BOOT_FAILED"
    kill_server "$PID" "$PORT"
    continue
  fi
  curl -s "http://127.0.0.1:$PORT/server_info" > "$OUT/g2detB_${TAG}_flags_on_chunk512_attempt${ATTEMPT}_${SLURM_JOB_ID}.json" || true

  BASEJSON="$OUT/g2detB_${TAG}_on_chunk512_attempt${ATTEMPT}_selfbaseline_${SLURM_JOB_ID}.json"
  RAWDUMP="$OUT/g2detB_${TAG}_on_chunk512_attempt${ATTEMPT}_selfbaseline_raw_${SLURM_JOB_ID}.json"
  BSTATUS_FILE="$OUT/g2detB_${TAG}_on_chunk512_attempt${ATTEMPT}_bstatus_${SLURM_JOB_ID}.txt"

  diag_snapshot "$PORT" "$PID" "attempt${ATTEMPT}_before"

  echo "attempt=$ATTEMPT issuing self-baseline greedy_call (bounded to ${G2_GREEDY_MAX_TIME}s) in background, polling for diagnostics..."
  capture_self_baseline "$PORT" "$BASEJSON" "$RAWDUMP" > "$BSTATUS_FILE" 2>>"$OUT/g2detB_${TAG}_srv_on_chunk512_attempt${ATTEMPT}_${SLURM_JOB_ID}.log" &
  BGPID=$!
  WAITED=0
  while kill -0 "$BGPID" 2>/dev/null && [ "$WAITED" -lt 60 ]; do
    sleep 15
    WAITED=$((WAITED + 15))
    if kill -0 "$BGPID" 2>/dev/null; then
      echo "attempt=$ATTEMPT still in flight at ${WAITED}s -- taking mid-call diagnostic snapshot"
      diag_snapshot "$PORT" "$PID" "attempt${ATTEMPT}_mid_${WAITED}s"
    fi
  done
  wait "$BGPID" 2>/dev/null
  diag_snapshot "$PORT" "$PID" "attempt${ATTEMPT}_after"

  BSTATUS="$(tr -d '[:space:]' < "$BSTATUS_FILE" 2>/dev/null)"
  echo "attempt=$ATTEMPT self-baseline status=$BSTATUS"
  eval "ATTEMPT_STATUS_${ATTEMPT}=\$BSTATUS"

  if [ "$BSTATUS" != OK ]; then
    printf '%s\n' "{\"detmode\": \"on\", \"arm\": \"chunk512\", \"attempt\": $ATTEMPT, \"boot_ok\": true, \"self_baseline_ok\": false, \"self_baseline_status\": \"$BSTATUS\"}" >> "$SUMMARY_JSON"
    kill_server "$PID" "$PORT"
    continue
  fi

  echo "attempt=$ATTEMPT self-baseline OK -- proceeding to the ${NREP_CTRL} pre-registered reps in this same boot"
  REP_JSONS=""
  for REP in $(seq 1 $NREP_CTRL); do
    OUTJSON="$OUT/g2detB_${TAG}_on_chunk512_attempt${ATTEMPT}_rep${REP}_${SLURM_JOB_ID}.json"
    echo "----- attempt=$ATTEMPT rep=$REP n_concurrent=$G2CTRL_NCONCURRENT -----"
    python3 "$OUT/g2ctrl_gate.py" --host 127.0.0.1 --port "$PORT" --prompt-file "$PROMPT_TXT" \
      --max-new-tokens "$OUTLEN" --n-concurrent "$G2CTRL_NCONCURRENT" \
      --self-baseline-file "$BASEJSON" --cross-baseline-file "$CROSS_A1_BASELINE" --out "$OUTJSON"
    REP_JSONS="$REP_JSONS $OUTJSON"
  done
  kill -0 "$PID" 2>/dev/null || echo "WARNING: server (attempt=$ATTEMPT) died during the 3 reps -- inspect $SRVLOG"
  kill_server "$PID" "$PORT"

  python3 - "$SUMMARY_JSON" "$ATTEMPT" "$BASEJSON" $REP_JSONS <<'PYEOF'
import json, sys
summary_path, attempt, basejson, *rep_jsons = sys.argv[1:]
base = json.load(open(basejson))
reps = [json.load(open(p)) for p in rep_jsons]
row = {
    "detmode": "on", "arm": "chunk512", "attempt": int(attempt),
    "boot_ok": True, "self_baseline_ok": True, "self_baseline_status": "OK",
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

echo "----- chunk512 3-attempt summary -----"
cat "$SUMMARY_JSON"
echo "attempt statuses: 1=$ATTEMPT_STATUS_1 2=$ATTEMPT_STATUS_2 3=$ATTEMPT_STATUS_3"

N_TIMEOUT=0; N_OK=0; N_OTHER=0
for S in "$ATTEMPT_STATUS_1" "$ATTEMPT_STATUS_2" "$ATTEMPT_STATUS_3"; do
  case "$S" in
    TIMEOUT) N_TIMEOUT=$((N_TIMEOUT + 1)) ;;
    OK) N_OK=$((N_OK + 1)) ;;
    *) N_OTHER=$((N_OTHER + 1)) ;;
  esac
done
echo "N_TIMEOUT=$N_TIMEOUT N_OK=$N_OK N_OTHER=$N_OTHER (of 3 attempts)"

if [ "$N_TIMEOUT" -eq 3 ]; then
  CLASSIFICATION="NOT_APPLICABLE"
elif [ "$N_OK" -eq 3 ]; then
  CLASSIFICATION="COMPLETE"
else
  CLASSIFICATION="SPORADIC"
fi
echo "=== REPRODUCIBILITY CLASSIFICATION: $CLASSIFICATION ==="
case "$CLASSIFICATION" in
  NOT_APPLICABLE)
    echo "--enable-deterministic-inference is NOT_APPLICABLE to Granite x chunk512 x a"
    echo "real multi-chunk prompt in this build. NOT a gate failure, NOT a refutation"
    echo "of the reduction-order hypothesis -- T4-1 cannot be answered with this lever"
    echo "here. The inheritance-discard rule stays in force, unchanged."
    ;;
  COMPLETE)
    echo "All 3 independent boots produced a scored self-baseline + 3 reps. Score with"
    echo "the original decision rule using attempt 1 as canonical (merge policy above);"
    echo "attempts 2-3 are replication, not pooled into the primary count."
    ;;
  SPORADIC)
    echo "N_OTHER=$N_OTHER means at least one attempt returned neither TIMEOUT nor OK"
    echo "(e.g. BOOT_FAILED/ERROR/BUILD_FAIL) -- reported above, does not fit either"
    echo "pre-registered bucket cleanly; treated as SPORADIC (do not score) unless"
    echo "N_OTHER=0 and this is a straightforward TIMEOUT/OK mix."
    echo "Per the decision rule: report this fact, DO NOT SCORE. A reproducibility"
    echo "experiment cannot be scored on top of a sporadic stall in its own"
    echo "measurement apparatus."
    ;;
esac

# ==========================================================================
# ON + agnostic: single boot, for negative-control balance with OFF's
# completed agnostic arm (876699). Not part of the reproducibility
# classification above (independent arm); a TIMEOUT here is still reported
# as measurement-absent per methodology gate #21, never as a mismatch.
# ==========================================================================
echo "############# ON + agnostic (negative-control balance) #############"
PORT=$((PORT_BASE + 250))
SRVLOG="$OUT/g2detB_${TAG}_srv_on_agnostic_${SLURM_JOB_ID}.log"
PID=$(boot_server agnostic "$PORT" "$SRVLOG")
OK=$(wait_ready "$PID" "$PORT")
echo "arm=agnostic boot_ok=$OK pid=$PID port=$PORT"
capture_evidence "$SRVLOG"
if [ "$OK" != 1 ]; then
  echo "BOOT FAILED arm=agnostic"; tail -30 "$SRVLOG" | grep -viE "pynvml|Future"
  printf '%s\n' "{\"detmode\": \"on\", \"arm\": \"agnostic\", \"boot_ok\": false}" >> "$SUMMARY_JSON"
  kill_server "$PID" "$PORT"
else
  curl -s "http://127.0.0.1:$PORT/server_info" > "$OUT/g2detB_${TAG}_flags_on_agnostic_${SLURM_JOB_ID}.json" || true
  BASEJSON="$OUT/g2detB_${TAG}_on_agnostic_selfbaseline_${SLURM_JOB_ID}.json"
  RAWDUMP="$OUT/g2detB_${TAG}_on_agnostic_selfbaseline_raw_${SLURM_JOB_ID}.json"
  BSTATUS=$(capture_self_baseline "$PORT" "$BASEJSON" "$RAWDUMP")
  echo "arm=agnostic self-baseline status=$BSTATUS"
  if [ "$BSTATUS" != OK ]; then
    printf '%s\n' "{\"detmode\": \"on\", \"arm\": \"agnostic\", \"boot_ok\": true, \"self_baseline_ok\": false, \"self_baseline_status\": \"$BSTATUS\"}" >> "$SUMMARY_JSON"
    kill_server "$PID" "$PORT"
  else
    REP_JSONS=""
    for REP in $(seq 1 $NREP_CTRL); do
      OUTJSON="$OUT/g2detB_${TAG}_on_agnostic_rep${REP}_${SLURM_JOB_ID}.json"
      echo "----- arm=agnostic rep=$REP n_concurrent=$G2CTRL_NCONCURRENT -----"
      python3 "$OUT/g2ctrl_gate.py" --host 127.0.0.1 --port "$PORT" --prompt-file "$PROMPT_TXT" \
        --max-new-tokens "$OUTLEN" --n-concurrent "$G2CTRL_NCONCURRENT" \
        --self-baseline-file "$BASEJSON" --out "$OUTJSON"
      REP_JSONS="$REP_JSONS $OUTJSON"
    done
    kill -0 "$PID" 2>/dev/null || echo "WARNING: server (arm=agnostic) died during the 3 reps -- inspect $SRVLOG"
    kill_server "$PID" "$PORT"
    python3 - "$SUMMARY_JSON" "$BASEJSON" $REP_JSONS <<'PYEOF'
import json, sys
summary_path, basejson, *rep_jsons = sys.argv[1:]
base = json.load(open(basejson))
reps = [json.load(open(p)) for p in rep_jsons]
row = {
    "detmode": "on", "arm": "agnostic", "boot_ok": True, "self_baseline_ok": True,
    "self_baseline_status": "OK", "self_baseline_sha": base["sha"],
    "reps": [
        {
            "n_ok": r["self"]["n_ok"], "n_match_self": r["self"]["n_match"],
            "n_mismatch_self": r["self"]["n_mismatch"], "verdict_self": r["self"]["verdict"],
            "cross": None,
            "self_mismatch_positions": r["self"]["mismatches"],
            "cross_mismatch_positions": None,
        }
        for r in reps
    ],
}
with open(summary_path, "a") as f:
    f.write(json.dumps(row) + "\n")
PYEOF
  fi
fi

echo "----- FULL SUMMARY (this job) -----"
cat "$SUMMARY_JSON"
echo "############# G2DETB DONE #############"
echo "summary -> $SUMMARY_JSON"
echo "DONE_G2DETB_${TAG}_classification=${CLASSIFICATION:-UNSET}"
