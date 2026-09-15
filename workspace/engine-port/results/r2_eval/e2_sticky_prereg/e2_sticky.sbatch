#!/bin/bash
#SBATCH --job-name=e2sticky
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=03:00:00
#SBATCH --comment="field=efficientai;appl=pytorch"
#SBATCH --output=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/r2_eval/e2_sticky_prereg/%x_%j.out
#SBATCH --error=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/r2_eval/e2_sticky_prereg/%x_%j.err
#
# E2 -- STICKY PARTITION CONTRAST (design E2-alpha: the OFF arm is measured in
# THIS job, not borrowed from job 908623).
#
# Pre-registration: PREREG_E2_STICKY_REV3_2026-09-15.md + its ADDENDUM.
#   rev1 NO-GO (fatal F1) -> rev2 NO-GO (fatal F1', F2') -> rev3 GO-with-caveats
#   (VERDICT_e2_rules_rev3_2026-09-15.md, 0 fatal, 7 blocking closed by the
#   ADDENDUM).  All three revisions are kept; none deleted.
#
# THE ONE KNOB.  OFF and ON differ by `PDMUX_STICKY_PARTITION` and nothing else.
# But the knob has TWO mechanisms (E2C-1) -- it lowers decode from 108 to 44 SM
# on decode-only spans AND removes one drain pair per prefill boundary -- and
# they have OPPOSITE signs.  That is why R2 has three branches and why no
# achieved change may be attributed to partitioning alone.
#
# WHAT THIS CANNOT DO (registered, prereg sec 10): it does not close gate #6
# (lambda*_SLO), does not move Claim D/E, does not resolve lambda0R-8(iii)
# (both arms have the SAME prefill axis), and produces ZERO performance
# verdicts.  Shape A's treated time is 86-89% prefill-idle (E2C-21), so Q2 buys
# only "the cost of pinning decode to 44 SM while there is nothing to
# multiplex with".
#
# ONE BOOT PER CELL, one telemetry file per boot -- deliberately NOT a ladder
# inside one boot, because slicing one telemetry file by time window
# reintroduces the interval-attribution machinery this track keeps dying on
# (`CONSENSUS sec 3 item 120`).  Inherited from lambda0.sbatch, which inherited
# it from c_capacity.sbatch:111,123-178.
#
# ORDER: seed-major, OFF and ON ADJACENT, so slow drift (thermal, node state,
# fragmentation) lands on both arms nearly equally.  The OFF->ON direction is a
# REGISTERED ASYMMETRY, not a control (E2C-7).
set -uo pipefail

ROOT=/scratch/ehmoon/whlee/prefill-layer-alloc
PREREG=$ROOT/workspace/engine-port/results/r2_eval/e2_sticky_prereg

module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0
export PS1="" PATH="$HOME/.local/bin:$PATH"
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate
cd "$ROOT" || exit 2

JOBID="${SLURM_JOB_ID:-local$$}"
OUT="$PREREG/e2_${JOBID}"; mkdir -p "$OUT"

export HF_HOME=$ROOT/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="$OUT/.triton_cache"; mkdir -p "$TRITON_CACHE_DIR"
unset SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN
# Every mode knob this campaign does NOT use.  PDMUX_STICKY_PARTITION IS listed
# here as a defence against an inherited value in the submitting environment; the
# per-boot set/unset inside the loop is what decides the arm.
unset PDMUX_SLO_SCHED PDMUX_SLO_MODE PDMUX_LA_COORD \
      PDMUX_FIXED_DECODE_SM_FILE PDMUX_TRACE_FORCE_PREFILL \
      PDMUX_DUAL_WORKER_TRACE_EVERY PDMUX_TRUE_DUAL_WORKER PDMUX_DUAL_WORKER \
      PDMUX_STICKY_PARTITION

# ------------------------------------------------- registered control factors
# Identical to job 908623 (prereg sec 3; audited against lambda0.sbatch in rev1
# and rev2, both times item-by-item).
MODEL="nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base"
BACKEND="flashinfer"
CFG="$ROOT/workspace/engine-port/results/longctx_conflict/probes/pdmux_homog5.yml"
CTX=16384
MEM_FRACTION=0.82
MAX_RUNNING=48
SERVER_SEED=1
FIXED_DSM=44
SGPT_RAW="$HF_HOME/raw/ShareGPT_V3_unfiltered_cleaned_split.json"
WALL_BUDGET_S=10800          # ADDENDUM A4: P14 measures against the --time grant
P14_MARGIN_S=200

# cell spec: name shape in out offered num_prompts T_cap_s
# T_cap = 3 x the 908623 OFF duration (P7).  A cell that hits it is CAPPED and
# takes its pair to UNRESOLVED -- a silently shortened run reads as a throughput
# GAIN, which is the failure gate #7 exists for.
declare -A T_CAP=( [a_r4]=393 [a_r2]=428 [b_r3]=861 )
declare -A SHAPE=( [a_r4]=A [a_r2]=A [b_r3]=B )
declare -A INLEN=( [a_r4]=256 [a_r2]=256 [b_r3]=8192 )
declare -A OUTLEN=( [a_r4]=512 [a_r2]=512 [b_r3]=64 )
declare -A RATE=(  [a_r4]=8.00 [a_r2]=3.00 [b_r3]=1.15 )
declare -A NP=(    [a_r4]=400 [a_r2]=400 [b_r3]=200 )
# Registered seed literals (prereg sec 2): ranks 1-4 of lambda0's own
# `choose_seeds` over lambda0's registered N multiset.  These are NOT noise --
# they change the REALIZED offered rate (E2C-13: a_r2 = 2.996/2.955/2.950/2.951).
SEEDS_A=(4386 4162 251 2630)
SEEDS_B=(4386 4162)

# The 20 boots, in registered order.  Written out literally rather than
# generated, because the order IS part of the registration.
UNITS=()
for S in "${SEEDS_A[@]}"; do UNITS+=("a_r4:$S"); done
for S in "${SEEDS_A[@]}"; do UNITS+=("a_r2:$S"); done
for S in "${SEEDS_B[@]}"; do UNITS+=("b_r3:$S"); done

EXPECT=""
for U in "${UNITS[@]}"; do
  IFS=: read -r C S <<< "$U"; EXPECT="${EXPECT:+$EXPECT,}$C/$S"
done

# ------------------------------------------- P4: registration integrity (pre-GPU)
PREREG_MD="${PDMUX_E2_PREREG:-$PREREG/PREREG_E2_STICKY_REV3_2026-09-15.md}"
ADDENDUM_MD="$PREREG/PREREG_E2_STICKY_REV3_ADDENDUM_2026-09-15.md"
DECISION_PATH=(e2_realized_mix.py e2_label.py)
echo "E2 sticky contrast  prereg=$(basename "$PREREG_MD")  job=$JOBID  model=$MODEL backend=$BACKEND"
for f in "${DECISION_PATH[@]}" e2_sticky.sbatch; do
  [ -f "$PREREG/$f" ] || { echo "ABORT_P4 decision-path file missing: $f" \
    | tee -a "$OUT/PREFLIGHT_FAILURES.txt"; exit 3; }
done
for f in "$PREREG_MD" "$ADDENDUM_MD"; do
  [ -f "$f" ] || { echo "ABORT_P4 registration document missing: $f" \
    | tee -a "$OUT/PREFLIGHT_FAILURES.txt"; exit 3; }
done
{
  echo "# pre-registration in force"
  sha256sum "$PREREG_MD" "$ADDENDUM_MD"
  echo "# decision path + runner"
  for f in "${DECISION_PATH[@]}" e2_sticky.sbatch; do sha256sum "$PREREG/$f"; done
  echo "# every revision and verdict kept beside it"
  sha256sum "$PREREG"/PREREG_E2_STICKY*.md "$PREREG"/VERDICT_e2_rules*.md
  echo "# tree: $(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo NO_GIT)" \
       "uncommitted=$(git -C "$ROOT" status --porcelain 2>/dev/null | grep -c .)"
} | tee "$OUT/REGISTRATION_SHA256.txt"
N_SHA=$(grep -c "^[0-9a-f]\{64\} " "$OUT/REGISTRATION_SHA256.txt")
if [ "${N_SHA:-0}" -lt 10 ]; then
  echo "ABORT_P4 only $N_SHA digests recorded, registered minimum 10" \
    | tee -a "$OUT/PREFLIGHT_FAILURES.txt"; exit 3
fi
# P5 is a PROCEDURE gate, not a runtime one (lambda0R-10: the previous round
# submitted before committing).  Record the fact so the artefact shows it.
echo "P5_uncommitted_at_submit=$(git -C "$ROOT" status --porcelain 2>/dev/null | grep -c .)" \
  | tee -a "$OUT/REGISTRATION_SHA256.txt"

echo "=== env sync (runtime source manifest) ==="
"$ROOT/workspace/engine-port/scripts/bootstrap/sync_engine_tree.sh" \
  "$OUT/runtime_source_manifest.sha256" || exit 2
N_MANIFEST=$(grep -c . "$OUT/runtime_source_manifest.sha256")
grep -q "models/nemotron_h.py" "$OUT/runtime_source_manifest.sha256" || {
  echo "ABORT_P4 manifest does not cover the served model implementation" \
    | tee -a "$OUT/PREFLIGHT_FAILURES.txt"; exit 3; }
[ "${N_MANIFEST:-0}" -ge 24 ] || {
  echo "ABORT_P4 manifest has $N_MANIFEST entries, registered minimum 24" \
    | tee -a "$OUT/PREFLIGHT_FAILURES.txt"; exit 3; }
echo "manifest_entries=$N_MANIFEST  (ONE sync, ONE manifest, BOTH arms)"

# ------------------------------------------------ submission-blocking selftests
# The estimators must reproduce the registered table on the 908623 archive
# BYTE-EXACTLY, and every decision branch must be reachable.  Both are GPU 0.
python3 "$PREREG/e2_realized_mix.py" --selftest | tee "$OUT/SELFTEST_estimators.txt" || exit 4
python3 "$PREREG/e2_label.py" --selftest | tee "$OUT/SELFTEST_rules.txt" || exit 4
python3 -m unittest discover -s "$ROOT/workspace/engine-port/tests" -p "test_sticky_partition.py" \
  > "$OUT/SELFTEST_sticky_unit.txt" 2>&1 || {
  echo "ABORT_P4 test_sticky_partition failed on the INSTALLED tree" \
    | tee -a "$OUT/PREFLIGHT_FAILURES.txt"; tail -20 "$OUT/SELFTEST_sticky_unit.txt"; exit 4; }
echo "selftests OK (estimators 7/7, rules 8 reachability + 10 regression, sticky unit 12)"

wait_health () {
  for i in $(seq 1 240); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$1/health 2>/dev/null||true)" = 200 ] && return 0
    kill -0 "$2" 2>/dev/null || return 1
    sleep 5
  done
  return 1
}

FAIL_STREAK=0
PROBE_DONE_OFF=0; PROBE_DONE_ON=0
echo "expect=$EXPECT"
echo "units=${#UNITS[@]} boots=$(( ${#UNITS[@]} * 2 ))"

for U in "${UNITS[@]}"; do
  IFS=: read -r CELL CSEED <<< "$U"
  for ARM in OFF ON; do
    NAME="${CELL}_${ARM}_${CSEED}"
    TCAP=${T_CAP[$CELL]}

    # ---- P14: global clock guard (ADDENDUM A4).  Remaining = --time grant minus
    # elapsed.  Without this, P7's per-cell caps sum to 13,159 s > 10,800 s and
    # SLURM could SIGKILL mid-boot -- a truncation the UNRESOLVED convention does
    # not cover.
    REMAIN=$(( WALL_BUDGET_S - SECONDS ))
    if [ "$REMAIN" -lt $(( TCAP + P14_MARGIN_S )) ]; then
      echo "UNRESOLVED_BUDGET at $NAME: remaining=${REMAIN}s < cap+margin=$(( TCAP + P14_MARGIN_S ))s" \
        | tee -a "$OUT/BUDGET_STOP.txt"
      echo "  (remaining units are NOT run; the shortened set is NOT re-labelled -- D23)"
      break 2
    fi

    PORT=0
    for try in $(seq 1 50); do
      CAND=$((39100 + (RANDOM + $$ + try) % 800))
      if ! curl -s -o /dev/null --max-time 1 "http://127.0.0.1:$CAND/health" 2>/dev/null; then
        if ! ss -ltn 2>/dev/null | grep -q ":$CAND "; then PORT=$CAND; break; fi
      fi
    done
    [ "$PORT" = 0 ] && { echo "ABORT_NO_FREE_PORT"; exit 5; }

    SRVLOG="$OUT/srv_${NAME}.log"
    export PDMUX_TELEMETRY_PATH="$OUT/tel_${NAME}.jsonl"; : > "$PDMUX_TELEMETRY_PATH"
    export PDMUX_RUN_ID="e2_${NAME}_${JOBID}"
    export PDMUX_WORKLOAD_ID="e2_${NAME}"
    export PDMUX_R2_POLICY=fixed
    export PDMUX_R2_FIXED_DSM=$FIXED_DSM
    # ===================== THE ONE KNOB =====================
    if [ "$ARM" = "ON" ]; then export PDMUX_STICKY_PARTITION=1; else unset PDMUX_STICKY_PARTITION; fi
    # ========================================================

    echo ""
    echo "########## $NAME  cell=$CELL arm=$ARM seed=$CSEED shape=${SHAPE[$CELL]} rate=${RATE[$CELL]} np=${NP[$CELL]} Tcap=${TCAP}s port=$PORT elapsed=${SECONDS}s ##########"
    python -m sglang.launch_server --model-path "$MODEL" --trust-remote-code --dtype bfloat16 \
      --attention-backend "$BACKEND" \
      --enable-pdmux --pdmux-config-path "$CFG" \
      --disable-overlap-schedule --chunked-prefill-size -1 \
      --disable-radix-cache --mem-fraction-static "$MEM_FRACTION" \
      --max-running-requests "$MAX_RUNNING" \
      --context-length "$CTX" --random-seed "$SERVER_SEED" \
      --host 127.0.0.1 --port "$PORT" > "$SRVLOG" 2>&1 &
    PID=$!

    if ! wait_health "$PORT" "$PID"; then
      echo "BOOT_FAILED cell=$NAME" | tee -a "$OUT/BOOT_FAILURES.txt"
      tail -40 "$SRVLOG" | grep -viE "pynvml|Future" | tee -a "$OUT/BOOT_FAILURES.txt"
      FAIL_STREAK=$((FAIL_STREAK + 1))
      kill "$PID" 2>/dev/null; sleep 5
      pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null; sleep 10
      [ "$FAIL_STREAK" -ge 2 ] && { echo "ABORT_2_CONSECUTIVE_BOOT_FAILURES" \
        | tee -a "$OUT/BOOT_FAILURES.txt"; break 2; }
      continue
    fi
    echo "boot_ok=1 cell=$NAME"
    FAIL_STREAK=0

    # ---- P1/P2: the ON arm must actually be ON, and pinned to index 2.
    # A silently-OFF "ON" arm would make the whole campaign an identity gate.
    STICKY_LINES=$(grep -c "sticky partition ENABLED" "$SRVLOG" || true)
    STICKY_IDX=$(grep -o "fixed target index=[0-9]*" "$SRVLOG" | tail -1 | cut -d= -f2)
    echo "P1 sticky_log_lines=$STICKY_LINES fixed_target_index=${STICKY_IDX:-none} arm=$ARM"
    if [ "$ARM" = "ON" ] && [ "${STICKY_LINES:-0}" -lt 1 ]; then
      echo "ABORT_STICKY_STATE_MISMATCH ON arm has no ENABLED line ($NAME)" \
        | tee -a "$OUT/PREFLIGHT_FAILURES.txt"; kill "$PID" 2>/dev/null; exit 6
    fi
    if [ "$ARM" = "OFF" ] && [ "${STICKY_LINES:-0}" -ne 0 ]; then
      echo "ABORT_STICKY_STATE_MISMATCH OFF arm HAS an ENABLED line ($NAME)" \
        | tee -a "$OUT/PREFLIGHT_FAILURES.txt"; kill "$PID" 2>/dev/null; exit 6
    fi
    if [ "$ARM" = "ON" ] && [ "${STICKY_IDX:-x}" != "2" ]; then
      echo "ABORT_P2 sticky resolved to index ${STICKY_IDX:-none}, registered 2 ($NAME)" \
        | tee -a "$OUT/PREFLIGHT_FAILURES.txt"; kill "$PID" 2>/dev/null; exit 6
    fi

    # ---- P3: the binding resources must be identical across arms, checked PER BOOT
    # with an abort path.  Extract the VALUES (timestamps in the log line would make
    # every boot unique and the comparison vacuous).
    MMB=$(grep -o "max_mamba_cache_size: *[0-9]*" "$SRVLOG" | tail -1 | grep -o "[0-9]*$")
    MTT=$(grep -o "max_total_num_tokens=[0-9]*" "$SRVLOG" | tail -1 | grep -o "[0-9]*$")
    echo "$NAME max_mamba_cache_size=${MMB:-NONE} max_total_num_tokens=${MTT:-NONE}" \
      | tee -a "$OUT/BANNERS.txt"
    echo "${MMB:-NONE} ${MTT:-NONE}" >> "$OUT/P3_KEYS.txt"
    if [ "$(sort -u "$OUT/P3_KEYS.txt" | wc -l)" -gt 1 ]; then
      echo "ABORT_P3 binding resources differ across boots -- see BANNERS.txt" \
        | tee -a "$OUT/PREFLIGHT_FAILURES.txt"
      kill "$PID" 2>/dev/null; exit 8
    fi

    # ---- P13: correctness contrast, BOTH arms, before any load.
    # Prompts and sampling are literally those of results/sticky_smoke/
    # sticky_smoke.sbatch:118-133, whose run (job 872800) reported identical=True.
    # Judged quantity = the list of `text` fields; token ids are recorded only.
    PROBE_END=""
    RUN_PROBE=0
    [ "$ARM" = "OFF" ] && [ "$PROBE_DONE_OFF" = 0 ] && RUN_PROBE=1
    [ "$ARM" = "ON" ] && [ "$PROBE_DONE_ON" = 0 ] && RUN_PROBE=1
    if [ "$RUN_PROBE" = 1 ]; then
      python3 - "$PORT" "$OUT/greedy_${ARM}.json" <<'PY' | tee -a "$OUT/P13_PROBE.txt"
import json, sys, time, urllib.request
port, out = sys.argv[1], sys.argv[2]
PROMPTS = [
    "The capital of France is",
    "Q: What is 17 + 25?\nA:",
    "Once upon a time in a small village by the sea,",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "List the first five prime numbers:",
    "Translate to French: The weather is nice today.",
]
texts = []
for p in PROMPTS:
    body = json.dumps({"text": p,
                       "sampling_params": {"temperature": 0, "max_new_tokens": 48}}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        rec = json.loads(r.read().decode())
    texts.append(rec["text"])
# CLOCK_MONOTONIC is system-wide on Linux, so this is directly comparable to the
# server's `timestamp_monotonic_s` (ADDENDUM A3).  The estimator asserts it lies
# inside the telemetry's own range and fails closed if it does not.
t_end = time.monotonic()
# Token ids are NOT recorded: the probe body is byte-identical to the precedent
# (sticky_smoke.sbatch:118-133), which does not request logprobs, so the field would
# always have been [null]x6.  ADDENDUM A7 judges the `text` list.
json.dump({"texts": texts, "t_probe_end": t_end}, open(out, "w"), ensure_ascii=False, indent=1)
print(f"P13_PROBE_DONE n={len(texts)} t_probe_end={t_end:.6f} empty={sum(1 for t in texts if not t)}")
PY
      PROBE_END=$(python3 -c "import json;print(json.load(open('$OUT/greedy_${ARM}.json'))['t_probe_end'])")
      EMPTY=$(python3 -c "import json;print(sum(1 for t in json.load(open('$OUT/greedy_${ARM}.json'))['texts'] if not t))")
      if [ "${EMPTY:-1}" != "0" ]; then
        echo "ABORT_CORRECTNESS $ARM arm produced $EMPTY empty completions" \
          | tee -a "$OUT/PREFLIGHT_FAILURES.txt"; kill "$PID" 2>/dev/null; exit 7
      fi
      [ "$ARM" = "OFF" ] && PROBE_DONE_OFF=1 || PROBE_DONE_ON=1
    fi

    # warm-up: 8 sequential requests, discarded (probe C :139-145, inherited).
    # NB E2C-16: this traffic is NOT excluded from the estimator population --
    # the `phase` marker flips on the first request, so `phase != "startup"`
    # cannot remove it.  Post-hoc bench-only re-scoring may NOT re-label R1.
    python -m sglang.bench_serving --backend sglang --model "$MODEL" \
      --host 127.0.0.1 --port "$PORT" \
      --dataset-name random --dataset-path "$SGPT_RAW" --tokenize-prompt \
      --random-input-len "${INLEN[$CELL]}" --random-output-len "${OUTLEN[$CELL]}" \
      --random-range-ratio 1.0 --num-prompts 8 --max-concurrency 1 --seed 7 \
      > "$OUT/warmup_${NAME}.log" 2>&1
    echo "warmup_rc=$?"

    # ---- the measurement.  `timeout` enforces P7; a CAPPED cell takes its pair
    # to UNRESOLVED rather than contributing a shortened duration.
    timeout "${TCAP}s" python -m sglang.bench_serving --backend sglang --model "$MODEL" \
      --host 127.0.0.1 --port "$PORT" \
      --dataset-name random --dataset-path "$SGPT_RAW" --tokenize-prompt \
      --random-input-len "${INLEN[$CELL]}" --random-output-len "${OUTLEN[$CELL]}" \
      --random-range-ratio 1.0 --num-prompts "${NP[$CELL]}" --request-rate "${RATE[$CELL]}" \
      --seed "$CSEED" --output-details --output-file "$OUT/bench_${NAME}.jsonl" \
      > "$OUT/bench_${NAME}.log" 2>&1
    BRC=$?
    echo "bench_rc=$BRC cell=$NAME"
    [ "$BRC" = 124 ] && echo "CAPPED $NAME hit T_cap=${TCAP}s" | tee -a "$OUT/CAPPED.txt"
    grep -E "Successful requests|Request throughput|Total input tokens|Median TTFT|P99 TTFT|Median ITL|Concurrency:" \
      "$OUT/bench_${NAME}.log"

    python3 "$PREREG/e2_realized_mix.py" "$PDMUX_TELEMETRY_PATH" \
      --cell "$CELL" --arm "$ARM" --seed "$CSEED" \
      ${PROBE_END:+--probe-end "$PROBE_END"} \
      --bench "$OUT/bench_${NAME}.jsonl" --bench-rc "$BRC" --num-prompts "${NP[$CELL]}" \
      --input-len "${INLEN[$CELL]}" --output-len "${OUTLEN[$CELL]}" \
      --out "$OUT/cell_${NAME}.json" > "$OUT/analyze_${NAME}.log" 2>&1 \
      || echo "ANALYZE_FAILED cell=$NAME"
    tail -30 "$OUT/analyze_${NAME}.log"
    du -h "$PDMUX_TELEMETRY_PATH" | sed 's/^/  telemetry /'

    kill "$PID" 2>/dev/null; sleep 5
    pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null; sleep 10
  done
done

echo ""
# ---- P3 verdict: the binding resources really were identical across arms.
echo "=== P3 summary (the gate itself fired per boot, above) ==="
sort -u "$OUT/P3_KEYS.txt" 2>/dev/null | tee "$OUT/P3_BANNER_UNIQUE.txt"
echo "P3 distinct binding-resource keys across all boots: $(sort -u "$OUT/P3_KEYS.txt" 2>/dev/null | wc -l) (registered: 1)"

# ---- P13 verdict: byte-identical greedy output across arms (literal, ADDENDUM A7)
if [ -f "$OUT/greedy_OFF.json" ] && [ -f "$OUT/greedy_ON.json" ]; then
  python3 - "$OUT/greedy_OFF.json" "$OUT/greedy_ON.json" <<'PY' | tee "$OUT/P13_VERDICT.txt"
import json, sys
a = json.load(open(sys.argv[1]))["texts"]
b = json.load(open(sys.argv[2]))["texts"]
same = (a == b)
print("STICKY_CORRECTNESS_BYTE_IDENTICAL" if same else "STICKY_CORRECTNESS_DIVERGED")
print(f"n={len(a)} identical={same}")
if not same:
    print("MANDATORY COMPANION: on (Nemotron-9B, (64,44), flashinfer, cudagraph ON) the "
          "sticky-ON output did NOT match OFF.  The canonical precedent "
          "(CONSENSUS sec 3 item 27, job 872800) was Ha8 / division (92,16).")
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            print(f"  [{i}] OFF={x[:80]!r}\n      ON ={y[:80]!r}")
PY
fi

echo ""
echo "=== registered rules (PREREG_E2_STICKY_REV3 sec 6 + ADDENDUM A4/A5/A6) ==="
python3 "$PREREG/e2_label.py" --cells-dir "$OUT" --expect "$EXPECT" \
  --out "$OUT/E2_LABEL.json" || echo "LABEL_FAILED"

echo "E2_DONE_${JOBID} elapsed=${SECONDS}s"
