#!/bin/bash
# E-1 cap campaign (running cap 48 vs 192, mamba pool control) -- VESSL Job target.
# Pre-registration: PREREG_E1_CAP_VESSL_2026-10-04.md (rev1.1; rev1 rule-layer audit GO-with-caveats,
#   VERDICT_e1_cap_vessl_rules_rev1_2026-10-04.md, caveats E1C-1..9 registered in prereg sec 15).
#
# ONE JOB = ONE BLOCK.  A block runs all 12 registered arms once, in the block's
# registered random order, on one node, with one bench seed shared by every arm
# (e1_plan.py ARMS / block_order / BLOCK_SEEDS).  Blocks never span Jobs; an
# incomplete block is invalid and is replaced by a spare block (7, 8), never resumed.
#
# SERVING ARM = HE0 (slo_sched/sharegpt_vary_bench.sbatch): Zamba2-2.7B, triton,
# ctx 4096, mem 0.82, chunked prefill off, overlap off, radix off, cudagraph ON,
# ShareGPT (context-len 4000) 200 prompts, LO rate 3 / HI rate 12, 3 rounds, each
# phase a separate bench_serving process.  Arm knobs: config (d24/d34/d44/slo),
# --max-running-requests, --max-mamba-cache-size (M arms only), telemetry on/off,
# bind+GATE env (B arms).  Differences from HE0 are listed in the prereg sec 3.
#
# usage (inside a VESSL Job via scripts/vessl/launch.sh; never by hand on a GPU):
#   launch.sh --campaign e1_cap_vessl \
#     --target workspace/engine-port/results/e1_cap_vessl/e1_cap_vessl.sh \
#     --env PDMUX_E1_BLOCK=<1..8>
# local CPU-only: PDMUX_E1_DRYRUN=1 (no server; commands are written, not run).
set -uo pipefail

ROOT="${PDMUX_PROJECT_ROOT:?PDMUX_PROJECT_ROOT must be set (job_entry.sh exports it)}"
ENG="$ROOT/workspace/engine-port"
HERE="$ENG/results/e1_cap_vessl"
HELPER="$HERE/e1_plan.py"
CFGDIR="$ENG/results/slo_sched"
DATA="${PDMUX_DATA:-/data}"
DRYRUN="${PDMUX_E1_DRYRUN:-0}"
BLOCK="${PDMUX_E1_BLOCK:-}"
JOBID="${PDMUX_JOB_NAME:-local$$}"
OUT="${PDMUX_E1_TEST_OUT_ROOT:-$HERE}/e1v_${JOBID}"     # TEST_OUT_ROOT: dry-run only
mkdir -p "$OUT"
cd "$ROOT" || exit 2
[ "$DRYRUN" = 1 ] && echo "DRY RUN -- NOT A MEASUREMENT (no server started)" > "$OUT/DRYRUN"
WD_PID=""
cleanup () {
  [ -n "$WD_PID" ] && kill "$WD_PID" 2>/dev/null
  rm -rf "$HERE/__pycache__" "$ENG/benchmarks/pdmux_eval/__pycache__" "$ENG/tests/__pycache__" 2>/dev/null
}
trap cleanup EXIT
fail () { echo "$1" | tee -a "$OUT/PREFLIGHT_FAILURES.txt"; exit "$2"; }

# ------------------------------------------------------------ PDMUX_* allowlist (prereg sec 3-3)
ALLOW=" PDMUX_COMMIT PDMUX_JOB_NAME PDMUX_CAMPAIGN PDMUX_TARGET PDMUX_IMAGE PDMUX_SPEC \
PDMUX_ROOT PDMUX_PROJECT_ROOT PDMUX_IO PDMUX_DATA PDMUX_OPT PDMUX_E1_BLOCK PDMUX_E1_DRYRUN \
PDMUX_E1_TEST_HARD_CAP_S PDMUX_E1_TEST_FAKE_RUN_S PDMUX_E1_TEST_SKIP_PREFLIGHT \
PDMUX_E1_TEST_OUT_ROOT PDMUX_TEST_SKIP_SYNC PDMUX_ARRAY_SPEC "
for v in $(compgen -e | grep '^PDMUX_' || true); do
  case "$ALLOW" in *" $v "*) ;; *) fail "ABORT_ENV unregistered knob $v is set (prereg sec 3-3)" 2;; esac
done
[ -n "${PDMUX_ARRAY_SPEC:-}" ] && fail "ABORT_ENV PDMUX_ARRAY_SPEC must be empty (one Job = one block)" 2
if [ "$DRYRUN" != 1 ]; then
  for v in PDMUX_E1_TEST_HARD_CAP_S PDMUX_E1_TEST_FAKE_RUN_S PDMUX_E1_TEST_SKIP_PREFLIGHT \
           PDMUX_E1_TEST_OUT_ROOT PDMUX_TEST_SKIP_SYNC; do
    [ -n "${!v:-}" ] && fail "ABORT_TEST_KNOB_IN_MEASUREMENT $v" 2
  done
fi
python3 "$HELPER" --check-block "${BLOCK:-0}"; BRC=$?
[ "$BRC" = 2 ] && fail "ABORT_BLOCK PDMUX_E1_BLOCK='$BLOCK' is not a registered block (1..8)" 2
[ "$BRC" = 0 ] || fail "ABORT_HELPER e1_plan.py failed to run (rc=$BRC)" 2
BDIR="$OUT/block_${BLOCK}"
mkdir -p "$BDIR/raw"

export HF_HOME="${HF_HOME:-$DATA/hf}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
unset SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN

# ------------------------------------------------------------------ registered arm (HE0)
K () { python3 "$HELPER" --const "$1"; }
MODEL=$(K MODEL); REV_WANT=$(K MODEL_REVISION); SGPT_SHA=$(K SHAREGPT_SHA256)
BACKEND=$(K BACKEND); CTX=$(K CTX); MEM_FRACTION=$(K MEM_FRACTION); SERVER_SEED=$(K SERVER_SEED)
NP=$(K NP); RLO=$(K RATE_LO); RHI=$(K RATE_HI); ROUNDS=$(K ROUNDS); SGCTX=$(K SHAREGPT_CONTEXT_LEN)
CG2_N=$(K CG2_PROMPTS); CG2_SEED=$(K CG2_SEED)
BSEED=$(python3 "$HELPER" --seed "$BLOCK")
SGPT="$HF_HOME/raw/ShareGPT_V3_unfiltered_cleaned_split.json"
MODEL_DIR="$HF_HOME/hub/models--${MODEL//\//--}"

# ------------------------------------------------------- wall-clock hard cap (prereg sec 8)
HARD_CAP_S=$(K HARD_CAP_S); TAIL_RESERVE_S=$(K TAIL_RESERVE_S); RUN_NEED_S=$(K RUN_NEED_S)
[ -n "${PDMUX_E1_TEST_HARD_CAP_S:-}" ] && HARD_CAP_S="$PDMUX_E1_TEST_HARD_CAP_S"
T0=""
TIMING="$DATA/runs/$JOBID/meta/timing.txt"
[ -f "$TIMING" ] && T0=$(date -u -d "$(awk -F= '$1=="start_utc"{print $2}' "$TIMING")" +%s 2>/dev/null || true)
[ -z "$T0" ] && T0=$(date +%s)
DEADLINE=$((T0 + HARD_CAP_S - TAIL_RESERVE_S))
remaining () { echo $((DEADLINE - $(date +%s))); }
budget_fired () { [ -e "$OUT/BUDGET_HARDCAP_FIRED" ]; }
{
  echo "hard_cap_s=$HARD_CAP_S tail_reserve_s=$TAIL_RESERVE_S run_need_s=$RUN_NEED_S"
  echo "t0_epoch=$T0 ($( [ -f "$TIMING" ] && echo job_entry_start_utc || echo script_start)) deadline_epoch=$DEADLINE"
  echo "remaining_at_script_start_s=$(remaining)"
} | tee "$OUT/BUDGET.txt"
(
  while [ "$(remaining)" -gt 0 ]; do sleep 5; done
  echo "fired_epoch=$(date +%s)" > "$OUT/BUDGET_HARDCAP_FIRED"
  pkill -TERM -f "sglang.bench_serving" 2>/dev/null
  pkill -TERM -f "sglang.launch_server" 2>/dev/null
  pkill -TERM -f "e1_fake_run" 2>/dev/null
  sleep 15
  pkill -9 -f "sglang.bench_serving" 2>/dev/null
  pkill -9 -f "sglang.launch_server" 2>/dev/null
  pkill -9 -f "e1_fake_run" 2>/dev/null
) > /dev/null 2>&1 &
WD_PID=$!

echo "E1 cap campaign block=$BLOCK bench_seed=$BSEED model=$MODEL job=$JOBID dryrun=$DRYRUN"

# ------------------------------------------------ registration record (prereg sec 9)
PREREG_MD="$HERE/PREREG_E1_CAP_VESSL_2026-10-04.md"
[ -f "$PREREG_MD" ] || fail "ABORT_PREREG missing $PREREG_MD" 3
python3 "$HELPER" --verify-configs "$ENG" | tee "$OUT/CONFIG_CHECK.txt"
[ "${PIPESTATUS[0]}" -eq 0 ] || fail "ABORT_CONFIG_DRIFT a registered pdmux config differs (prereg sec 3-1)" 3
{
  echo "# pre-registration in force + every rule/runner file this Job executes"
  sha256sum "$HERE"/PREREG_E1_CAP_VESSL*.md "$HERE"/VERDICT_e1_cap_vessl_rules_*.md "$HERE/e1_cap_vessl.sh" "$HELPER" \
            "$ENG/benchmarks/pdmux_eval/analyze.py" "$ENG/tests/test_e1_cap_vessl.py" \
            "$ENG/tests/test_phase_events.py" "$ENG/src/multiplex/multiplexing_mixin.py"
  for c in pdmux_d24.yml pdmux_d34.yml pdmux_d44.yml pdmux_slo.yml; do sha256sum "$CFGDIR/$c"; done
  echo "# tree: $(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo NO_GIT)" \
       "uncommitted_outside_results=$(git -C "$ROOT" status --porcelain --untracked-files=no -- . ':!workspace/engine-port/results' 2>/dev/null | grep -c .)"
  echo "# block=$BLOCK bench_seed=$BSEED order=$(python3 "$HELPER" --order "$BLOCK" | tr '\n' ' ')"
} | tee "$OUT/REGISTRATION_SHA256.txt"

# ------------------------------------------------------ substrate + inputs (prereg sec 4)
{
  echo "nproc=$(nproc 2>/dev/null)"
  echo "cpu_affinity=$(taskset -pc $$ 2>/dev/null | awk -F': ' '{print $2}')"
  echo "cgroup_cpu_max=$(cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo NA)"
  echo "mem_total_kb=$(awk '/MemTotal/{print $2}' /proc/meminfo 2>/dev/null)"
  echo "tz=$(date +%Z) python=$(python3 -c 'import sys;print(sys.version.split()[0])' 2>/dev/null)"
  echo "perf_counter_clock=$(python3 -c 'import time;print(time.get_clock_info("perf_counter").implementation)' 2>/dev/null)"
  echo "torch=$(python3 -c 'import torch;print(torch.__version__, torch.version.cuda)' 2>/dev/null || echo NA)"
  echo "nvidia_smi=$(nvidia-smi --query-gpu=name,uuid,driver_version,clocks.max.sm,power.limit --format=csv,noheader 2>&1 | head -1)"
  echo "hf_refs_main=$(cat "$MODEL_DIR/refs/main" 2>/dev/null || echo ABSENT)"
  echo "sharegpt_sha256=$(sha256sum "$SGPT" 2>/dev/null | cut -d' ' -f1 || echo ABSENT)"
  echo "# engine-relevant env (non-secret prefixes only)"
  env | grep -E '^(SGLANG|FLASHINFER|TRITON|CUDA|NCCL|TORCH|XDG_CACHE)_?[A-Z_]*=' | sort
} > "$OUT/SUBSTRATE_E1.txt" 2>&1
cat "$OUT/SUBSTRATE_E1.txt"
dry_or_abort () {   # $1 = message ; $2 = exit code
  if [ "$DRYRUN" = 1 ]; then echo "DRYRUN_NOTE $1"; else fail "$1" "$2"; fi
}
[ "$(cat "$MODEL_DIR/refs/main" 2>/dev/null || echo ABSENT)" = "$REV_WANT" ] \
  || dry_or_abort "ABORT_MODEL_REVISION $MODEL refs/main != $REV_WANT (stage it first, prereg sec 10)" 3
[ "$(sha256sum "$SGPT" 2>/dev/null | cut -d' ' -f1)" = "$SGPT_SHA" ] \
  || dry_or_abort "ABORT_TRACE ShareGPT sha256 != $SGPT_SHA" 3
grep -q "perf_counter_clock=clock_gettime(CLOCK_MONOTONIC)" "$OUT/SUBSTRATE_E1.txt" \
  || dry_or_abort "ABORT_CLOCK perf_counter is not CLOCK_MONOTONIC: server events and PHASE_MARKS would not share a clock" 3

# --------------------------------------------- engine: installed tree carries the instrument
MIXIN="${SGLANG_ENGINE_DEV:-/opt/pdmux/sglang_engine_dev/python}/sglang/srt/multiplex/multiplexing_mixin.py"
PSTATE="${SGLANG_ENGINE_DEV:-/opt/pdmux/sglang_engine_dev/python}/sglang/srt/distributed/parallel_state.py"
if ! grep -q "def resolve_phase_events" "$MIXIN" 2>/dev/null || ! grep -q "def pdmux_role_is_thread_local" "$PSTATE" 2>/dev/null; then
  dry_or_abort "ABORT_ENGINE installed tree lacks PDMUX_PHASE_EVENTS or the thread-local role patch ($MIXIN)" 3
fi
cmp -s "$MIXIN" "$ENG/src/multiplex/multiplexing_mixin.py" \
  || dry_or_abort "ABORT_ENGINE installed multiplexing_mixin.py != this commit's src mirror" 3
cp -p "$DATA/runs/$JOBID/meta/runtime_source_manifest.sha256" "$OUT/" 2>/dev/null || true

# ------------------------------------------------------------------ preflight (CPU)
if [ "${PDMUX_E1_TEST_SKIP_PREFLIGHT:-0}" != 1 ]; then
  python3 "$HELPER" --selftest || fail "ABORT_PREFLIGHT helper selftest" 5
  # Job preflight = the fast classes that check THIS installed tree and the rule code.  The
  # runner-behaviour classes (env allowlist, dry run, budget enforcer: ~4 min of subprocesses)
  # test the runner bytes, which are digest-recorded above; they run in local/CI regression.
  (cd "$ENG/tests" && PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -q test_phase_events \
      test_e1_cap_vessl.TestHE0CommandLines test_e1_cap_vessl.TestHelper \
      test_e1_cap_vessl.TestAnalysisGeometry test_e1_cap_vessl.TestLabelMutants \
      test_e1_cap_vessl.TestFailClosedWiring test_e1_cap_vessl.TestRev11DescriptiveFields) \
    > "$OUT/PREFLIGHT_TESTS.txt" 2>&1 || { tail -20 "$OUT/PREFLIGHT_TESTS.txt"; fail "ABORT_PREFLIGHT unit tests (see PREFLIGHT_TESTS.txt)" 5; }
else
  echo "TEST_SKIP_PREFLIGHT: selftest/unit tests NOT run (dry-run only)" | tee "$OUT/TEST_SKIP_PREFLIGHT"
fi

# ------------------------------------------------------------------ server plumbing
wait_health () {
  for i in $(seq 1 240); do
    budget_fired && return 1
    [ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$1/health" 2>/dev/null || true)" = 200 ] && return 0
    kill -0 "$2" 2>/dev/null || return 1
    sleep 5
  done
  return 1
}
pick_port () {
  local try CAND
  for try in $(seq 1 50); do
    CAND=$((39100 + (RANDOM + $$ + try) % 800))
    if ! curl -s -o /dev/null --max-time 1 "http://127.0.0.1:$CAND/health" 2>/dev/null \
       && ! ss -ltn 2>/dev/null | grep -q ":$CAND "; then echo "$CAND"; return 0; fi
  done
  echo 0
}
mono () { python3 -c 'import time; print(repr(time.perf_counter()))'; }
wall () { date '+%Y-%m-%d %H:%M:%S'; }   # naive local, same format as the server log stamps
teardown () {  # $1 pid $2 port
  kill "$1" 2>/dev/null; sleep 5
  pkill -9 -f "launch_server.*--port $2" 2>/dev/null; sleep 10
}

# srv_cmd <arm> <port> <cfg> <cap> <pool>  -> SRV array
srv_cmd () {
  SRV=(python -m sglang.launch_server --model-path "$MODEL" --trust-remote-code --dtype bfloat16
    --attention-backend "$BACKEND" --disable-radix-cache
    --mem-fraction-static "$MEM_FRACTION" --max-running-requests "$4" --context-length "$CTX"
    --enable-pdmux --pdmux-config-path "$CFGDIR/$3" --chunked-prefill-size -1 --disable-overlap-schedule
    --random-seed "$SERVER_SEED" --host 127.0.0.1 --port "$2")
  [ "$5" != auto ] && SRV+=(--max-mamba-cache-size "$5")
}
# bench_cmd <port> <rate> <np> <seed> <outfile> [extra...] -> BENCH array
bench_cmd () {
  BENCH=(python -m sglang.bench_serving --backend sglang --model "$MODEL" --host 127.0.0.1 --port "$1"
    --dataset-name sharegpt --dataset-path "$SGPT" --sharegpt-context-len "$SGCTX"
    --num-prompts "$3" --request-rate "$2" --seed "$4" --output-details --output-file "$5")
  shift 5; BENCH+=("$@")
}

# ------------------------------------------- CG2: GPU correctness gate (prereg sec 5-3)
# Greedy decoding, 16 ShareGPT prompts, one at a time, on the S44_C48 server with the
# instrument ON vs OFF; the generated texts must be identical.  Fails closed (exit 7).
run_cg2 () {
  local mode PORT PID f
  for mode in on off; do
    PORT=$(pick_port); [ "$PORT" = 0 ] && fail "ABORT_NO_FREE_PORT" 4
    srv_cmd CG2 "$PORT" pdmux_d44.yml 48 auto
    f="$OUT/cg2_${mode}.jsonl"; : > "$f"
    bench_cmd "$PORT" inf "$CG2_N" "$CG2_SEED" "$f" --max-concurrency 1
    if [ "$DRYRUN" = 1 ]; then
      { printf 'CG2 %s server:' "$mode"; printf ' %q' "${SRV[@]}"; echo
        printf 'CG2 %s bench :' "$mode"; printf ' %q' "${BENCH[@]}"; echo; } >> "$OUT/DRYRUN_COMMANDS.txt"
      continue
    fi
    if [ "$mode" = on ]; then
      env PDMUX_TELEMETRY_PATH="$OUT/cg2_tel.jsonl" PDMUX_PHASE_EVENTS=1 \
          PDMUX_RUN_ID="e1_cg2_${JOBID}" PDMUX_WORKLOAD_ID=e1_cg2 \
          setsid "${SRV[@]}" > "$OUT/cg2_srv_${mode}.log" 2>&1 &
    else
      setsid "${SRV[@]}" > "$OUT/cg2_srv_${mode}.log" 2>&1 &
    fi
    PID=$!
    wait_health "$PORT" "$PID" || { teardown "$PID" "$PORT"; fail "ABORT_CG2_BOOT ($mode)" 7; }
    timeout -k 30 900 "${BENCH[@]}" > "$OUT/cg2_bench_${mode}.log" 2>&1
    local rc=$?
    teardown "$PID" "$PORT"
    [ "$rc" = 0 ] || fail "ABORT_CG2_BENCH ($mode rc=$rc)" 7
  done
  [ "$DRYRUN" = 1 ] && return 0
  python3 "$HELPER" --compare-cg2 "$OUT/cg2_on.jsonl" "$OUT/cg2_off.jsonl" | tee "$OUT/CG2.txt"
  [ "${PIPESTATUS[0]}" -eq 0 ] || fail "ABORT_CG2_MISMATCH instrument ON changed greedy outputs" 7
  grep -q '"event":"decode_iteration"' "$OUT/cg2_tel.jsonl" \
    || fail "ABORT_CG2_NO_EVENTS instrument ON wrote no decode_iteration event" 7
}
run_cg2

# ------------------------------------------------------------------ one arm
FAIL_STREAK=0
STOP=""
run_arm () {   # $1 = arm
  local ARM="$1" RD="$BDIR/raw"
  local CFG CAP POOL TEL CTRL DSM EXPECT_POOL
  eval "$(python3 "$HELPER" --arm-spec "$ARM" | tr ' ' '\n' | sed 's/^/local /')"
  if [ -e "$BDIR/run_${ARM}.complete" ]; then echo "SKIP_DONE arm=$ARM"; return 0; fi
  if [ "$STOP" = boot ]; then echo "NOT_ATTEMPTED_AFTER_BOOT_STREAK arm=$ARM" | tee -a "$OUT/BOOT_FAILURES.txt"; return 0; fi
  if [ "$STOP" = budget ] || budget_fired; then echo "NOT_ATTEMPTED_BUDGET arm=$ARM" | tee -a "$OUT/BUDGET_SKIPPED.txt"; return 0; fi
  local NEED="$RUN_NEED_S" REM
  [ -n "${PDMUX_E1_TEST_FAKE_RUN_S:-}" ] && NEED=$(( (NEED + 999) / 1000 ))   # enforcer test only (DRYRUN)
  REM=$(remaining)
  if [ "$REM" -lt "$NEED" ]; then
    echo "NOT_ATTEMPTED_BUDGET arm=$ARM need_s=$NEED remaining_s=$REM" | tee -a "$OUT/BUDGET_SKIPPED.txt"
    STOP=budget; return 0
  fi
  echo "admit arm=$ARM need_s=$NEED remaining_s=$REM" >> "$OUT/BUDGET.txt"

  local PORT; PORT=$(pick_port); [ "$PORT" = 0 ] && fail "ABORT_NO_FREE_PORT" 4
  srv_cmd "$ARM" "$PORT" "$CFG" "$CAP" "$POOL"
  # arm env: built explicitly, nothing inherited (the allowlist already refused stray PDMUX_*)
  local AENV=(PDMUX_GREEN_READOUT=1 "PDMUX_GREEN_READOUT_PATH=$RD/green_${ARM}.json")
  if [ "$TEL" = 1 ]; then
    AENV+=("PDMUX_TELEMETRY_PATH=$RD/tel_${ARM}.jsonl" PDMUX_PHASE_EVENTS=1
           "PDMUX_RUN_ID=e1_${ARM}_b${BLOCK}_${JOBID}" "PDMUX_WORKLOAD_ID=e1_${ARM}")
  fi
  local CENV; CENV=$(python3 "$HELPER" --arm-env "$ARM")
  [ -n "$CENV" ] && read -r -a _ce <<< "$CENV" && AENV+=("${_ce[@]}")
  local MARKS="$RD/PHASE_MARKS_${ARM}.tsv"

  echo ""
  echo "########## ARM $ARM block=$BLOCK cfg=$CFG cap=$CAP pool=$POOL tel=$TEL ctrl=$CTRL port=$PORT remaining_s=$REM ##########"
  if [ "$DRYRUN" = 1 ]; then
    { printf 'ARM %s\n  env   :' "$ARM"; printf ' %q' "${AENV[@]}"; echo
      printf '  server:'; printf ' %q' "${SRV[@]}"; echo
      for PH in LO HI; do
        bench_cmd "$PORT" "$([ $PH = LO ] && echo "$RLO" || echo "$RHI")" "$NP" "$BSEED" "$RD/bench_${PH}_${ARM}.jsonl"
        printf '  bench %s x%s:' "$PH" "$ROUNDS"; printf ' %q' "${BENCH[@]}"; echo
      done; } >> "$OUT/DRYRUN_COMMANDS.txt"
    if [ -n "${PDMUX_E1_TEST_FAKE_RUN_S:-}" ]; then
      local CAPT; CAPT=$(remaining); [ "$CAPT" -lt 1 ] && CAPT=1
      timeout -k 2 "$CAPT" bash -c 'exec -a e1_fake_run sleep "$1"' _ "$PDMUX_E1_TEST_FAKE_RUN_S"
      echo "fake_rc=$? arm=$ARM" | tee -a "$OUT/FAKE_RUNS.txt"
    fi
    return 0
  fi

  : > "$RD/bench_LO_${ARM}.jsonl"; : > "$RD/bench_HI_${ARM}.jsonl"
  [ "$TEL" = 1 ] && : > "$RD/tel_${ARM}.jsonl"
  printf 'round\tphase\tmono_start\tmono_end\twall_start\twall_end\trc\n' > "$MARKS"
  env "${AENV[@]}" setsid "${SRV[@]}" > "$RD/srv_${ARM}.log" 2>&1 &
  local PID=$!
  if wait_health "$PORT" "$PID"; then
    echo "boot_ok=1 arm=$ARM"; FAIL_STREAK=0
    local r PH RATE MS WS RC CAPT
    for r in $(seq 1 "$ROUNDS"); do
      for PH in LO HI; do
        RATE=$([ "$PH" = LO ] && echo "$RLO" || echo "$RHI")
        bench_cmd "$PORT" "$RATE" "$NP" "$BSEED" "$RD/bench_${PH}_${ARM}.jsonl"
        CAPT=$(remaining); [ "$CAPT" -lt 1 ] && CAPT=1
        MS=$(mono); WS=$(wall)
        timeout -k 30 "$CAPT" "${BENCH[@]}" >> "$RD/benchlog_${ARM}.log" 2>&1
        RC=$?
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$r" "$PH" "$MS" "$(mono)" "$WS" "$(wall)" "$RC" >> "$MARKS"
        echo "  round $r $PH rc=$RC"
        if [ "$RC" = 124 ] || [ "$RC" = 137 ] || budget_fired; then
          echo "BENCH_KILLED_BY_BUDGET arm=$ARM rc=$RC" | tee -a "$OUT/BUDGET_SKIPPED.txt"
          touch "$RD/KILLED_${ARM}"; break 2
        fi
      done
    done
  else
    echo "BOOT_FAILED arm=$ARM" | tee -a "$OUT/BOOT_FAILURES.txt"
    tail -40 "$RD/srv_${ARM}.log" | grep -viE "pynvml|Future" | tee -a "$OUT/BOOT_FAILURES.txt"
    budget_fired || FAIL_STREAK=$((FAIL_STREAK + 1))
  fi
  teardown "$PID" "$PORT"
  grep -cE "SLO-BIND|SLO-SCHED" "$RD/srv_${ARM}.log" 2>/dev/null | sed "s/^/  switch_lines arm=$ARM /"
  # analysis on CPU, server dead (descriptive fields + validity; the label is computed later)
  python3 "$HELPER" --analyze-run "$RD" "$ARM" --out "$BDIR/run_${ARM}.json" 2>> "$OUT/ANALYZE.log" \
    || echo "ANALYZE_FAILED arm=$ARM" | tee -a "$OUT/ANALYZE.log"
  [ -s "$BDIR/run_${ARM}.json" ] && date -u +%Y-%m-%dT%H:%M:%SZ > "$BDIR/run_${ARM}.complete"
  if [ "$FAIL_STREAK" -ge 2 ]; then
    echo "ABORT_2_CONSECUTIVE_BOOT_FAILURES" | tee -a "$OUT/BOOT_FAILURES.txt"; STOP=boot
  fi
}

mapfile -t ORDER < <(python3 "$HELPER" --order "$BLOCK")
[ "${#ORDER[@]}" -eq 12 ] || fail "ABORT_ARM_COUNT ${#ORDER[@]} != 12" 2
printf '%s\n' "${ORDER[@]}" > "$BDIR/ORDER.txt"
for ARM in "${ORDER[@]}"; do run_arm "$ARM"; done
kill "$WD_PID" 2>/dev/null
echo "remaining_at_loop_end_s=$(remaining)" | tee -a "$OUT/BUDGET.txt"

echo "=== block summary (validity only; the label needs >= 4 valid blocks across Jobs) ==="
python3 "$HELPER" --block-summary "$BDIR" | tee "$BDIR/BLOCK_SUMMARY.json"
{
  echo "{\"block\": $BLOCK, \"bench_seed\": $BSEED, \"job\": \"$JOBID\","
  echo " \"gpu\": \"$(nvidia-smi --query-gpu=uuid --format=csv,noheader 2>/dev/null | head -1)\","
  echo " \"hostname\": \"$(hostname)\", \"dryrun\": $DRYRUN}"
} > "$BDIR/BLOCK.json"
echo "E1_BLOCK_DONE_${BLOCK}_${JOBID}"
