#!/bin/bash
# E2 -- STICKY PARTITION CONTRAST, VESSL substrate port of ../e2_sticky_prereg/e2_sticky.sbatch.
# Registration: PREREG_E2_STICKY_REV3_2026-09-15.md + its ADDENDUM (A1-A12) + the substrate-port
#   addendum PREREG_E2_VESSL_A13_2026-10-04.md (A13).  rev3/ADDENDUM rules, estimators, cells,
#   seeds, arms, order and labels are NOT changed.
#
# WHAT IS THE SAME AS e2_sticky.sbatch (enforced, not asserted):
#   * the estimator and rule files are ../e2_sticky_prereg/{e2_realized_mix,e2_label}.py, CALLED,
#     never copied; `e2_vessl_plan.py --verify-digests` aborts (exit 3) unless they, rev1-3, the
#     ADDENDUM, every E2 verdict, the 2026-09-15 OVERRIDE, A13 and the served config are the
#     registered bytes;
#   * server / P13 probe / warm-up / bench / estimator command lines; THE ONE KNOB
#     (PDMUX_STICKY_PARTITION unset vs 1); P1/P2/P3/P8/P9-P12/P13 gates; the 20-boot order;
#     the D23 convention (a shortened set is never re-labelled);
#   * the rev3 sec 5-3 estimator selftest runs IN THE JOB on job 908623's archive (staged).
# WHAT CHANGES (A13 sec 2):
#   * paths from PDMUX_PROJECT_ROOT / HF_HOME (job_entry.sh); no module/venv;
#   * engine tree = this commit (HEAD): differs from 908623's in exactly multiplexing_mixin.py
#     (PDMUX_PHASE_EVENTS, default OFF) -- enforced, anything else aborts;
#   * P7 T_cap = 3 x the VESSL λ0 duration of the same cell/seed/arm (401/429/870 s);
#   * P14's "remaining budget" = a registered in-script hard cap (4.0 h from job_entry start),
#     enforced in three layers (admission, per-step timeout, watchdog) -- SLURM --time is gone;
#   * ADDED: B0 substrate check, model-revision / trace digest checks, one green-context
#     read-out per boot (E2C-38, descriptive), per-boot cgroup cpu.stat + epoch timing,
#     PDMUX_* allowlist, pair-granular resume.
#
# usage (inside a VESSL Job via scripts/vessl/launch.sh; never by hand on a GPU):
#   launch.sh --campaign e2_sticky_vessl \
#     --target workspace/engine-port/results/r2_eval/e2_sticky_vessl/e2_sticky_vessl.sh
# optional: --env PDMUX_E2V_RESUME_FROM=<previous job name>   (A13 sec 5-4)
# local CPU-only: PDMUX_E2V_DRYRUN=1 (no server is started; no cell is measured).
set -uo pipefail

ROOT="${PDMUX_PROJECT_ROOT:?PDMUX_PROJECT_ROOT must be set (job_entry.sh exports it)}"
ENG="$ROOT/workspace/engine-port"
PREREG="$ENG/results/r2_eval/e2_sticky_prereg"          # rev3 decision path (unchanged)
HERE="$ENG/results/r2_eval/e2_sticky_vessl"
HELPER="$HERE/e2_vessl_plan.py"
ARCHIVE="$ENG/results/r2_eval/lambda0_prereg/lam0_908623"   # rev3 sec 5-3 selftest input
DATA="${PDMUX_DATA:-/data}"
IO="${PDMUX_IO:-/io}"
STAGE="$IO/staging/e2_selftest_archive"
DRYRUN="${PDMUX_E2V_DRYRUN:-0}"

JOBID="${PDMUX_JOB_NAME:-local$$}"
OUT="${PDMUX_E2V_TEST_OUT_ROOT:-$HERE}/e2v_${JOBID}"     # TEST_OUT_ROOT: dry-run only
mkdir -p "$OUT"
cd "$ROOT" || exit 2
[ "$DRYRUN" = 1 ] && echo "DRY RUN -- NOT A MEASUREMENT (no server started)" > "$OUT/DRYRUN"
WD_PID=""
cleanup () {
  [ -n "$WD_PID" ] && kill "$WD_PID" 2>/dev/null
  rm -rf "$PREREG/__pycache__" "$HERE/__pycache__" 2>/dev/null
}
trap cleanup EXIT
fail () { echo "$1" | tee -a "$OUT/PREFLIGHT_FAILURES.txt"; exit "$2"; }

# ------------------------------------------------------ PDMUX_* knob allowlist (A13 sec 2-4)
# launch.sh forwards any PDMUX_* with --env.  ANY PDMUX_* not on this list aborts -- in
# particular PDMUX_STICKY_PARTITION (the arm is set per boot below, never inherited) and
# PDMUX_PHASE_EVENTS (the one engine difference vs 908623 must stay at its default OFF).
ALLOW=" PDMUX_COMMIT PDMUX_JOB_NAME PDMUX_CAMPAIGN PDMUX_TARGET PDMUX_IMAGE PDMUX_SPEC \
PDMUX_ROOT PDMUX_PROJECT_ROOT PDMUX_IO PDMUX_DATA PDMUX_OPT PDMUX_E2V_RESUME_FROM \
PDMUX_E2V_DRYRUN PDMUX_E2V_TEST_HARD_CAP_S PDMUX_E2V_TEST_FAKE_BENCH_S \
PDMUX_E2V_TEST_SKIP_PREFLIGHT PDMUX_E2V_TEST_OUT_ROOT PDMUX_TEST_SKIP_SYNC PDMUX_ARRAY_SPEC "
for v in $(compgen -e | grep '^PDMUX_' || true); do
  case "$ALLOW" in *" $v "*) ;; *) fail "ABORT_ENV unregistered knob $v is set (A13 sec 2-4)" 2;; esac
done
[ -n "${PDMUX_ARRAY_SPEC:-}" ] && fail "ABORT_ENV PDMUX_ARRAY_SPEC must be empty (E2-alpha: one Job)" 2
if [ "$DRYRUN" != 1 ]; then
  for v in PDMUX_E2V_TEST_HARD_CAP_S PDMUX_E2V_TEST_FAKE_BENCH_S PDMUX_E2V_TEST_SKIP_PREFLIGHT \
           PDMUX_E2V_TEST_OUT_ROOT PDMUX_TEST_SKIP_SYNC; do
    [ -n "${!v:-}" ] && fail "ABORT_TEST_KNOB_IN_MEASUREMENT $v" 2
  done
fi

export HF_HOME="${HF_HOME:-$DATA/hf}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
# TRITON_CACHE_DIR / XDG_CACHE_HOME / FLASHINFER_WORKSPACE_BASE: job_entry's per-run values.
unset SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN
# e2_sticky.sbatch's unset list, plus the two λ0 VESSL added (both unset in 908623 too).
unset PDMUX_SLO_SCHED PDMUX_SLO_MODE PDMUX_LA_COORD \
      PDMUX_FIXED_DECODE_SM_FILE PDMUX_TRACE_FORCE_PREFILL \
      PDMUX_DUAL_WORKER_TRACE_EVERY PDMUX_TRUE_DUAL_WORKER PDMUX_DUAL_WORKER \
      PDMUX_STICKY_PARTITION PDMUX_DUAL_WORKER_TRACE PDMUX_MEM_TELEMETRY PDMUX_PHASE_EVENTS

# ------------------------------------------------- registered control factors (rev3 sec 3)
MODEL="nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base"
BACKEND="flashinfer"
CFG="$ENG/results/longctx_conflict/probes/pdmux_homog5.yml"
CTX=16384
MEM_FRACTION=0.82
MAX_RUNNING=48
SERVER_SEED=1
FIXED_DSM=44
SGPT_RAW="$HF_HOME/raw/ShareGPT_V3_unfiltered_cleaned_split.json"
MODEL_DIR="$HF_HOME/hub/models--nvidia--NVIDIA-Nemotron-Nano-9B-v2-Base"
K () { python3 "$HELPER" --const "$1"; }

# ----------------------------------------------- wall-clock hard cap (A13 sec 5-2, P14 anchor)
# Measured from job_entry's start (clone + install count).  The cap is a registered literal in
# e2_vessl_plan.py; the TEST override is refused above unless DRYRUN=1.
HARD_CAP_S=$(K HARD_CAP_S) || exit 2
TAIL_RESERVE_S=$(K TAIL_RESERVE_S) || exit 2
P14_MARGIN_S=$(K P14_MARGIN_S) || exit 2
[ -n "${PDMUX_E2V_TEST_HARD_CAP_S:-}" ] && HARD_CAP_S="$PDMUX_E2V_TEST_HARD_CAP_S"
T0=""
TIMING="$DATA/runs/$JOBID/meta/timing.txt"
[ -f "$TIMING" ] && T0=$(date -u -d "$(awk -F= '$1=="start_utc"{print $2}' "$TIMING")" +%s 2>/dev/null || true)
[ -z "$T0" ] && T0=$(date +%s)
DEADLINE=$((T0 + HARD_CAP_S - TAIL_RESERVE_S))
remaining () { echo $((DEADLINE - $(date +%s))); }
# Preflight and tail kill layers (VERDICT_e2_vessl_rules_2026-10-04 required repair 5, recommended
# part): preflight steps are bounded by the CELL deadline, tail steps by the HARD CAP end
# (T0 + HARD_CAP_S, i.e. the tail reserve).  The watchdog only knows bench/server/fake patterns,
# so without these a hung sync/selftest/label would run unbounded (E2VC-3).  Floor 1 s.
cap_s () { local r; r=$(remaining); [ "$r" -lt 1 ] && r=1; echo "$r"; }
tail_s () { local r=$((T0 + HARD_CAP_S - $(date +%s))); [ "$r" -lt 1 ] && r=1; echo "$r"; }
budget_fired () { [ -e "$OUT/BUDGET_HARDCAP_FIRED" ]; }
{
  echo "hard_cap_s=$HARD_CAP_S tail_reserve_s=$TAIL_RESERVE_S p14_margin_s=$P14_MARGIN_S"
  echo "t0_epoch=$T0 ($( [ -f "$TIMING" ] && echo job_entry_start_utc || echo script_start)) deadline_epoch=$DEADLINE"
  echo "remaining_at_script_start_s=$(remaining)"
} | tee "$OUT/BUDGET.txt"
# Watchdog: at the deadline, kill whatever runs (bench client, server) and flag it.
(
  while [ "$(remaining)" -gt 0 ]; do sleep 5; done
  echo "fired_epoch=$(date +%s)" > "$OUT/BUDGET_HARDCAP_FIRED"
  pkill -TERM -f "sglang.bench_serving" 2>/dev/null
  pkill -TERM -f "sglang.launch_server" 2>/dev/null
  pkill -TERM -f "e2v_fake_bench" 2>/dev/null
  sleep 15
  pkill -9 -f "sglang.bench_serving" 2>/dev/null
  pkill -9 -f "sglang.launch_server" 2>/dev/null
  pkill -9 -f "e2v_fake_bench" 2>/dev/null
) > /dev/null 2>&1 &
WD_PID=$!

echo "E2 sticky contrast (VESSL port, A13) job=$JOBID model=$MODEL backend=$BACKEND dryrun=$DRYRUN"
dry_or_abort () {   # $1 = message ; $2 = exit code
  if [ "$DRYRUN" = 1 ]; then echo "DRYRUN_NOTE $1"; else fail "$1" "$2"; fi
}

# --------------------------------------- registration integrity (rev3 P4 + A13 sec 1-1)
A13_MD="$HERE/PREREG_E2_VESSL_A13_2026-10-04.md"
python3 "$HELPER" --verify-digests "$PREREG" "$HERE" "$ENG" | tee "$OUT/DIGEST_CHECK.txt"
[ "${PIPESTATUS[0]}" -eq 0 ] || fail "ABORT_REG_DRIFT E2 decision path / A13 / config differ from the registered bytes" 3
{
  echo "# pre-registration in force: rev3 + ADDENDUM (A1-A12) + A13"
  sha256sum "$PREREG/PREREG_E2_STICKY_REV3_2026-09-15.md" "$PREREG/PREREG_E2_STICKY_REV3_ADDENDUM_2026-09-15.md" "$A13_MD"
  echo "# decision path (called, never copied)"
  sha256sum "$PREREG/e2_realized_mix.py" "$PREREG/e2_label.py"
  echo "# KISTI runner (not executed here) + every revision and verdict kept beside it"
  sha256sum "$PREREG/e2_sticky.sbatch" "$PREREG"/PREREG_E2_STICKY*.md "$PREREG"/VERDICT_e2_*.md "$PREREG"/OVERRIDE_E2_*.md
  echo "# VESSL port runner + helper + override + served config"
  sha256sum "$HERE/e2_sticky_vessl.sh" "$HELPER" "$HERE"/OVERRIDE_E2_VESSL_*.md "$CFG"
  # job_entry replaces engine-port/results with a symlink, so count tracked changes OUTSIDE it.
  echo "# tree: $(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo NO_GIT)" \
       "uncommitted_outside_results=$(git -C "$ROOT" status --porcelain --untracked-files=no -- . ':!workspace/engine-port/results' 2>/dev/null | grep -c .)"
} | tee "$OUT/REGISTRATION_SHA256.txt"
N_SHA=$(grep -c "^[0-9a-f]\{64\} " "$OUT/REGISTRATION_SHA256.txt")
[ "${N_SHA:-0}" -ge 10 ] || fail "ABORT_P4 only $N_SHA digests recorded, registered minimum 10" 3

# ------------------------------------------------- substrate record + B0 (A13 sec 2-1)
{
  echo "nproc=$(nproc 2>/dev/null)"
  echo "cpu_affinity=$(taskset -pc $$ 2>/dev/null | awk -F': ' '{print $2}')"
  echo "cgroup_cpu_max=$(cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo NA)"
  echo "cgroup_mem_max=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo NA)"
  echo "mem_total_kb=$(awk '/MemTotal/{print $2}' /proc/meminfo 2>/dev/null)"
  echo "python=$(python3 -c 'import sys;print(sys.version.split()[0])' 2>/dev/null)"
  echo "monotonic_clock=$(python3 -c 'import time;print(time.get_clock_info("monotonic").implementation)' 2>/dev/null)"
  echo "torch=$(python3 -c 'import torch;print(torch.__version__, torch.version.cuda)' 2>/dev/null || echo NA)"
  echo "flashinfer=$(python3 -c 'import importlib.metadata as m;print(m.version("flashinfer-python"))' 2>/dev/null || echo NA)"
  echo "sglang_kernel=$(python3 -c 'import importlib.metadata as m;print(m.version("sglang-kernel"))' 2>/dev/null || echo NA)"
  echo "numpy=$(python3 -c 'import numpy;print(numpy.__version__)' 2>/dev/null || echo NA)"
  echo "nvidia_smi=$(nvidia-smi --query-gpu=name,uuid,driver_version,clocks.max.sm,clocks.applications.graphics,power.limit --format=csv,noheader 2>&1 | head -1)"
  echo "hf_refs_main=$(cat "$MODEL_DIR/refs/main" 2>/dev/null || echo ABSENT)"
  echo "sharegpt_sha256=$(sha256sum "$SGPT_RAW" 2>/dev/null | cut -d' ' -f1 || echo ABSENT)"
  echo "# engine-relevant env (names=values, non-secret prefixes only)"
  env | grep -E '^(SGLANG|FLASHINFER|TRITON|CUDA|NCCL|TORCH|XDG_CACHE)_?[A-Z_]*=' | sort
} > "$OUT/SUBSTRATE_E2V.txt" 2>&1
cat "$OUT/SUBSTRATE_E2V.txt"
B0_CSV=$(nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version,mig.mode.current \
         --format=csv,noheader,nounits 2>/dev/null | head -1)
B0_SM=$(python3 -c 'import torch;print(torch.cuda.get_device_properties(0).multi_processor_count)' 2>/dev/null || echo NA)
python3 "$HELPER" --check-substrate "${B0_CSV:-NONE}" "$B0_SM" | tee "$OUT/SUBSTRATE_CHECK.txt"
[ "${PIPESTATUS[0]}" -eq 0 ] || dry_or_abort "ABORT_SUBSTRATE not an A100 80GB / cc 8.0 / 108 SM / MIG-off / driver>=550 device ($B0_CSV, sm=$B0_SM)" 3
[ "$(cat "$MODEL_DIR/refs/main" 2>/dev/null || echo ABSENT)" = "$(K MODEL_REVISION)" ] \
  || dry_or_abort "ABORT_MODEL_REVISION refs/main != $(K MODEL_REVISION)" 3
[ "$(sha256sum "$SGPT_RAW" 2>/dev/null | cut -d' ' -f1)" = "$(K SHAREGPT_SHA256)" ] \
  || dry_or_abort "ABORT_TRACE ShareGPT sha256 != $(K SHAREGPT_SHA256)" 3

# ----------------------------- rev3 sec 5-3 selftest archive (git-ignored; A13 sec 2-5)
for f in $(python3 -c "import sys; sys.path.insert(0,'$HERE'); import e2_vessl_plan as V; print(' '.join(sorted(V.ARCHIVE_DIGESTS)))"); do
  if [ ! -f "$ARCHIVE/$f" ]; then
    [ -f "$STAGE/$f" ] || fail "ABORT_STAGE selftest archive file missing: $STAGE/$f (run stage_e2_inputs.sh --upload first)" 3
    mkdir -p "$ARCHIVE"; timeout -k 10 "$(cap_s)" cp -p "$STAGE/$f" "$ARCHIVE/$f"     # -p keeps the 2026-09-14 mtime: not re-exported
  fi
done
python3 "$HELPER" --verify-archive "$ARCHIVE" | tee "$OUT/ARCHIVE_CHECK.txt"
[ "${PIPESTATUS[0]}" -eq 0 ] || fail "ABORT_ARCHIVE_DRIFT selftest archive differs from job 908623's bytes" 3

SKIP_PRE="${PDMUX_E2V_TEST_SKIP_PREFLIGHT:-0}"
if [ "$SKIP_PRE" != 1 ]; then
  # ----------------------------------------- rev3 P4 manifest + A13 sec 1-3 engine tree
  echo "=== env sync (runtime source manifest) ==="
  timeout -k 10 "$(cap_s)" "$ENG/scripts/bootstrap/sync_engine_tree.sh" "$OUT/runtime_source_manifest.sha256" || exit 2
  N_MANIFEST=$(grep -c . "$OUT/runtime_source_manifest.sha256")
  grep -q "models/nemotron_h.py" "$OUT/runtime_source_manifest.sha256" \
    || fail "ABORT_P4 manifest does not cover the served model implementation" 3
  [ "${N_MANIFEST:-0}" -ge 24 ] || fail "ABORT_P4 manifest has $N_MANIFEST entries, registered minimum 24" 3
  echo "manifest_entries=$N_MANIFEST  (ONE sync, ONE manifest, BOTH arms)"
  python3 "$HELPER" --engine-diff "$OUT/runtime_source_manifest.sha256" \
    "$ARCHIVE/runtime_source_manifest.sha256" | tee "$OUT/ENGINE_TREE_VS_908623.txt"
  [ "${PIPESTATUS[0]}" -eq 0 ] \
    || dry_or_abort "ABORT_ENGINE_TREE_DRIFT installed tree is not 908623's + exactly the registered multiplexing_mixin.py" 3
  # ------------------------------------------- submission-blocking selftests (rev3 sec 5-3)
  timeout -k 10 "$(cap_s)" python3 "$PREREG/e2_realized_mix.py" --selftest --archive "$ARCHIVE" | tee "$OUT/SELFTEST_estimators.txt"
  [ "${PIPESTATUS[0]}" -eq 0 ] || fail "ABORT_SELFTEST estimators did not reproduce the registered table" 4
  timeout -k 10 "$(cap_s)" python3 "$PREREG/e2_label.py" --selftest | tee "$OUT/SELFTEST_rules.txt"
  [ "${PIPESTATUS[0]}" -eq 0 ] || fail "ABORT_SELFTEST rule reachability / regression" 4
  timeout -k 10 "$(cap_s)" python3 "$HELPER" --selftest | tee "$OUT/SELFTEST_port.txt"
  [ "${PIPESTATUS[0]}" -eq 0 ] || fail "ABORT_SELFTEST port helper" 4
  PYTHONDONTWRITEBYTECODE=1 timeout -k 10 "$(cap_s)" python3 -m unittest discover -s "$ENG/tests" -p "test_sticky_partition.py" \
    > "$OUT/SELFTEST_sticky_unit.txt" 2>&1 || {
    tail -20 "$OUT/SELFTEST_sticky_unit.txt"; fail "ABORT_P4 test_sticky_partition failed on the INSTALLED tree" 4; }
  echo "selftests OK (estimators 7/7, rules 8 reachability + 10 regression, port helper, sticky unit)"
else
  echo "TEST_SKIP_PREFLIGHT: sync/manifest/engine-diff/selftests NOT run (dry-run only)" | tee "$OUT/TEST_SKIP_PREFLIGHT"
fi

# ------------------------------------------------------------ the 20 boots (rev3 sec 2)
mapfile -t UNITS < <(python3 "$HELPER" --units)
EXPECT=$(python3 "$HELPER" --expect)
[ "${#UNITS[@]}" -eq 20 ] || fail "ABORT_UNIT_COUNT ${#UNITS[@]} != 20" 2
printf '%s\n' "${UNITS[@]}" > "$OUT/UNITS.txt"
python3 "$HELPER" --budget > "$OUT/BUDGET_TABLE.json"
echo "expect=$EXPECT"
echo "units=$(( ${#UNITS[@]} / 2 )) boots=${#UNITS[@]}"

# ------------------------------------------------------------ resume (A13 sec 5-4)
# PAIR granularity only: a pair is copied iff BOTH its boots carry boot_<NAME>.complete in the
# previous Job.  A pair is never split across Jobs (E2-alpha holds inside every pair).
RESUME_FROM="${PDMUX_E2V_RESUME_FROM:-}"
declare -A RESUMED=()
if [ -n "$RESUME_FROM" ]; then
  PREV="$DATA/runs/$RESUME_FROM/results/r2_eval/e2_sticky_vessl/e2v_$RESUME_FROM"
  [ -d "$PREV" ] || fail "ABORT_RESUME previous run dir not found: $PREV" 2
  cmp -s "$PREV/UNITS.txt" "$OUT/UNITS.txt" || fail "ABORT_RESUME UNITS.txt differs from $RESUME_FROM" 2
  diff <(grep -E "e2_|PREREG_|VERDICT_|OVERRIDE_" "$PREV/REGISTRATION_SHA256.txt" | awk '{print $1}') \
       <(grep -E "e2_|PREREG_|VERDICT_|OVERRIDE_" "$OUT/REGISTRATION_SHA256.txt" | awk '{print $1}') > /dev/null \
    || fail "ABORT_RESUME registration digests differ from $RESUME_FROM" 2
  PREV_SUB="$DATA/runs/$RESUME_FROM/meta/substrate.json"
  PREV_GPU=$(python3 -c "import json;g=json.load(open('$PREV_SUB'))['gpu'];print(g['uuid'],g['driver_version'])" 2>/dev/null || echo "UNKNOWN UNKNOWN")
  printf 'pair\tfrom_job\tfrom_gpu_uuid\tfrom_driver\n' > "$OUT/RESUMED_PAIRS.tsv"
  for ((i = 0; i < ${#UNITS[@]}; i += 2)); do
    read -r C S _ <<< "${UNITS[$i]}"
    if [ -e "$PREV/boot_${C}_OFF_${S}.complete" ] && [ -e "$PREV/boot_${C}_ON_${S}.complete" ]; then
      for ARM in OFF ON; do
        for f in "$PREV"/*_"${C}_${ARM}_${S}".*; do cp "$f" "$OUT/"; done   # no -p: exported
        grep -h "^${C}_${ARM}_${S} " "$PREV/BANNERS.txt" 2>/dev/null >> "$OUT/BANNERS.txt"
        grep -h "^${C}_${ARM}_${S} " "$PREV/BANNERS.txt" 2>/dev/null | awk '{print $2, $3}' \
          | sed 's/max_mamba_cache_size=//; s/max_total_num_tokens=//' >> "$OUT/P3_KEYS.txt"
      done
      RESUMED["$C/$S"]=1
      printf '%s\t%s\t%s\n' "$C/$S" "$RESUME_FROM" "$(echo "$PREV_GPU" | tr ' ' '\t')" >> "$OUT/RESUMED_PAIRS.tsv"
    fi
  done
  for ARM in OFF ON; do
    [ -f "$PREV/greedy_${ARM}.json" ] && cp "$PREV/greedy_${ARM}.json" "$OUT/greedy_${ARM}_from_${RESUME_FROM}.json"
  done
  echo "RESUMED $(($(wc -l < "$OUT/RESUMED_PAIRS.tsv") - 1)) pair(s) from $RESUME_FROM (Job mix recorded)"
fi

# ------------------------------------------------------------ server plumbing
wait_health () {
  for i in $(seq 1 240); do
    budget_fired && return 1
    [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$1/health 2>/dev/null||true)" = 200 ] && return 0
    kill -0 "$2" 2>/dev/null || return 1
    sleep 5
  done
  return 1
}
teardown () {   # $1 pid  $2 port  (inherited sequence: kill, 5 s, pkill -9, 10 s)
  kill "$1" 2>/dev/null; sleep 5
  pkill -9 -f "launch_server.*--port $2" 2>/dev/null; sleep 10
}
cpu_snap () {   # $1 file  $2 tag   -- cgroup CFS throttle counters (λ0 VESSL result audit C8)
  { echo "# $2 epoch=$(date +%s)"; cat /sys/fs/cgroup/cpu.stat 2>/dev/null || echo "cpu.stat NA"; } >> "$1"
}

FAIL_STREAK=0
PROBE_DONE_OFF=0; PROBE_DONE_ON=0
STOP=""
printf 'boot\tadmit_epoch\thealth_epoch\tbench_start_epoch\tbench_end_epoch\tend_epoch\tbench_rc\tkill_kind\n' > "$OUT/TIMING.tsv"

for U in "${UNITS[@]}"; do
  read -r CELL CSEED ARM SHAPE INLEN OUTLEN RATE NP TCAP <<< "$U"
  NAME="${CELL}_${ARM}_${CSEED}"
  if [ -n "${RESUMED[$CELL/$CSEED]:-}" ]; then echo "SKIP_RESUMED_PAIR $NAME"; continue; fi
  [ "$STOP" = boot ] && break

  # ---- P14 (rev3 sec 7; ADDENDUM A4 re-anchored by A13 sec 5-2): remaining = registered hard-cap
  # deadline minus now.  If it is below T_cap(next cell) + 200 s, THIS seed pair onward is
  # UNRESOLVED_BUDGET and nothing further runs; the shortened set is NOT re-labelled (D23).
  NEED=$(( TCAP + P14_MARGIN_S ))
  [ -n "${PDMUX_E2V_TEST_FAKE_BENCH_S:-}" ] && NEED=$(( (NEED + 999) / 1000 ))   # enforcer test only (DRYRUN)
  REM=$(remaining)
  if [ "$REM" -lt "$NEED" ] || budget_fired; then
    echo "UNRESOLVED_BUDGET at $NAME: remaining=${REM}s < T_cap+margin=${NEED}s (or hard cap fired)" \
      | tee -a "$OUT/BUDGET_STOP.txt"
    echo "  (remaining units are NOT run; the shortened set is NOT re-labelled -- D23)"
    STOP=budget; break
  fi
  echo "admit boot=$NAME need_s=$NEED remaining_s=$REM" >> "$OUT/BUDGET.txt"
  T_ADMIT=$(date +%s)

  PORT=0
  for try in $(seq 1 50); do
    CAND=$((39100 + (RANDOM + $$ + try) % 800))
    if ! curl -s -o /dev/null --max-time 1 "http://127.0.0.1:$CAND/health" 2>/dev/null; then
      if ! ss -ltn 2>/dev/null | grep -q ":$CAND "; then PORT=$CAND; break; fi
    fi
  done
  [ "$PORT" = 0 ] && fail "ABORT_NO_FREE_PORT" 5

  SRVLOG="$OUT/srv_${NAME}.log"
  export PDMUX_TELEMETRY_PATH="$OUT/tel_${NAME}.jsonl"
  export PDMUX_RUN_ID="e2_${NAME}_${JOBID}"
  export PDMUX_WORKLOAD_ID="e2_${NAME}"
  export PDMUX_R2_POLICY=fixed
  export PDMUX_R2_FIXED_DSM=$FIXED_DSM
  # E2C-38 (A13 sec 3): one-shot driver read-out of every stream group's green context, BOTH arms.
  export PDMUX_GREEN_READOUT=1 PDMUX_GREEN_READOUT_PATH="$OUT/green_${NAME}.json"
  # ===================== THE ONE KNOB =====================
  if [ "$ARM" = "ON" ]; then export PDMUX_STICKY_PARTITION=1; else unset PDMUX_STICKY_PARTITION; fi
  # ========================================================

  echo ""
  echo "########## $NAME  cell=$CELL arm=$ARM seed=$CSEED shape=$SHAPE rate=$RATE np=$NP Tcap=${TCAP}s port=$PORT remaining_s=$REM ##########"
  SRV=(python -m sglang.launch_server --model-path "$MODEL" --trust-remote-code --dtype bfloat16
    --attention-backend "$BACKEND"
    --enable-pdmux --pdmux-config-path "$CFG"
    --disable-overlap-schedule --chunked-prefill-size -1
    --disable-radix-cache --mem-fraction-static "$MEM_FRACTION"
    --max-running-requests "$MAX_RUNNING"
    --context-length "$CTX" --random-seed "$SERVER_SEED"
    --host 127.0.0.1 --port "$PORT")
  WARM=(python -m sglang.bench_serving --backend sglang --model "$MODEL"
    --host 127.0.0.1 --port "$PORT"
    --dataset-name random --dataset-path "$SGPT_RAW" --tokenize-prompt
    --random-input-len "$INLEN" --random-output-len "$OUTLEN"
    --random-range-ratio 1.0 --num-prompts 8 --max-concurrency 1 --seed 7)
  BENCH=(python -m sglang.bench_serving --backend sglang --model "$MODEL"
    --host 127.0.0.1 --port "$PORT"
    --dataset-name random --dataset-path "$SGPT_RAW" --tokenize-prompt
    --random-input-len "$INLEN" --random-output-len "$OUTLEN"
    --random-range-ratio 1.0 --num-prompts "$NP" --request-rate "$RATE"
    --seed "$CSEED" --output-details --output-file "$OUT/bench_${NAME}.jsonl")

  if [ "$DRYRUN" = 1 ]; then
    { printf 'BOOT %s\n  server:' "$NAME"; printf ' %q' "${SRV[@]}"; echo
      printf '  warmup:'; printf ' %q' "${WARM[@]}"; echo
      printf '  bench :'; printf ' %q' "${BENCH[@]}"; echo
      echo "  env   : PDMUX_TELEMETRY_PATH=$PDMUX_TELEMETRY_PATH PDMUX_RUN_ID=$PDMUX_RUN_ID PDMUX_R2_POLICY=$PDMUX_R2_POLICY PDMUX_R2_FIXED_DSM=$PDMUX_R2_FIXED_DSM PDMUX_GREEN_READOUT=$PDMUX_GREEN_READOUT PDMUX_STICKY_PARTITION=${PDMUX_STICKY_PARTITION:-<unset>} T_cap=$TCAP"
    } >> "$OUT/DRYRUN_COMMANDS.txt"
    if [ -n "${PDMUX_E2V_TEST_FAKE_BENCH_S:-}" ]; then
      # budget-enforcer test: a fake bench under the SAME timeout / kill-kind / watchdog machinery
      BT=$TCAP; KIND=P7_CAPPED
      R=$(remaining); [ "$R" -lt "$BT" ] && { BT=$R; KIND=BUDGET; }; [ "$BT" -lt 1 ] && BT=1
      timeout -k 2 "${BT}s" bash -c 'exec -a e2v_fake_bench sleep "$1"' _ "$PDMUX_E2V_TEST_FAKE_BENCH_S"
      FRC=$?
      echo "fake_rc=$FRC boot=$NAME kind_if_cut=$KIND" | tee -a "$OUT/FAKE_BENCH.txt"
      if [ "$FRC" = 124 ] || [ "$FRC" = 137 ] || budget_fired; then
        [ "$KIND" = BUDGET ] || budget_fired && echo "BENCH_KILLED_BY_BUDGET boot=$NAME rc=$FRC" | tee -a "$OUT/BUDGET_STOP.txt"
      fi
    fi
    continue
  fi

  : > "$PDMUX_TELEMETRY_PATH"
  setsid "${SRV[@]}" > "$SRVLOG" 2>&1 &
  PID=$!
  if ! wait_health "$PORT" "$PID"; then
    echo "BOOT_FAILED cell=$NAME" | tee -a "$OUT/BOOT_FAILURES.txt"
    tail -40 "$SRVLOG" | grep -viE "pynvml|Future" | tee -a "$OUT/BOOT_FAILURES.txt"
    budget_fired || FAIL_STREAK=$((FAIL_STREAK + 1))
    teardown "$PID" "$PORT"
    printf '%s\t%s\tNA\tNA\tNA\t%s\tNA\tboot_failed\n' "$NAME" "$T_ADMIT" "$(date +%s)" >> "$OUT/TIMING.tsv"
    if [ "$FAIL_STREAK" -ge 2 ]; then
      echo "ABORT_2_CONSECUTIVE_BOOT_FAILURES" | tee -a "$OUT/BOOT_FAILURES.txt"; STOP=boot
    fi
    continue
  fi
  T_HEALTH=$(date +%s)
  echo "boot_ok=1 cell=$NAME"
  FAIL_STREAK=0

  # ---- P1/P2 (rev3 sec 7): the ON arm must actually be ON, and pinned to index 2.
  STICKY_LINES=$(grep -c "sticky partition ENABLED" "$SRVLOG" || true)
  STICKY_IDX=$(grep -o "fixed target index=[0-9]*" "$SRVLOG" | tail -1 | cut -d= -f2)
  echo "P1 sticky_log_lines=$STICKY_LINES fixed_target_index=${STICKY_IDX:-none} arm=$ARM"
  if [ "$ARM" = "ON" ] && [ "${STICKY_LINES:-0}" -lt 1 ]; then
    teardown "$PID" "$PORT"; fail "ABORT_STICKY_STATE_MISMATCH ON arm has no ENABLED line ($NAME)" 6
  fi
  if [ "$ARM" = "OFF" ] && [ "${STICKY_LINES:-0}" -ne 0 ]; then
    teardown "$PID" "$PORT"; fail "ABORT_STICKY_STATE_MISMATCH OFF arm HAS an ENABLED line ($NAME)" 6
  fi
  if [ "$ARM" = "ON" ] && [ "${STICKY_IDX:-x}" != "2" ]; then
    teardown "$PID" "$PORT"; fail "ABORT_P2 sticky resolved to index ${STICKY_IDX:-none}, registered 2 ($NAME)" 6
  fi

  # ---- P3 (rev3 sec 7): binding resources identical across ALL boots (resumed ones included).
  MMB=$(grep -o "max_mamba_cache_size: *[0-9]*" "$SRVLOG" | tail -1 | grep -o "[0-9]*$")
  MTT=$(grep -o "max_total_num_tokens=[0-9]*" "$SRVLOG" | tail -1 | grep -o "[0-9]*$")
  echo "$NAME max_mamba_cache_size=${MMB:-NONE} max_total_num_tokens=${MTT:-NONE}" | tee -a "$OUT/BANNERS.txt"
  echo "${MMB:-NONE} ${MTT:-NONE}" >> "$OUT/P3_KEYS.txt"
  if [ "$(sort -u "$OUT/P3_KEYS.txt" | wc -l)" -gt 1 ]; then
    teardown "$PID" "$PORT"; fail "ABORT_P3 binding resources differ across boots -- see BANNERS.txt" 8
  fi

  # ---- P13 (rev3 sec 7 + ADDENDUM A3/A7): correctness contrast, BOTH arms, before any load.
  # Probe body byte-identical to e2_sticky.sbatch (= sticky_smoke.sbatch:118-133).
  PROBE_END=""
  RUN_PROBE=0
  [ "$ARM" = "OFF" ] && [ "$PROBE_DONE_OFF" = 0 ] && RUN_PROBE=1
  [ "$ARM" = "ON" ] && [ "$PROBE_DONE_ON" = 0 ] && RUN_PROBE=1
  if [ "$RUN_PROBE" = 1 ]; then
    CAP=$(remaining); [ "$CAP" -lt 1 ] && CAP=1
    timeout -k 10 "$CAP" python3 - "$PORT" "$OUT/greedy_${ARM}.json" <<'PY' | tee -a "$OUT/P13_PROBE.txt"
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
    [ -s "$OUT/greedy_${ARM}.json" ] || { teardown "$PID" "$PORT"; fail "ABORT_CORRECTNESS $ARM arm probe produced no record ($NAME)" 7; }
    PROBE_END=$(python3 -c "import json;print(json.load(open('$OUT/greedy_${ARM}.json'))['t_probe_end'])")
    EMPTY=$(python3 -c "import json;print(sum(1 for t in json.load(open('$OUT/greedy_${ARM}.json'))['texts'] if not t))")
    if [ "${EMPTY:-1}" != "0" ]; then
      teardown "$PID" "$PORT"; fail "ABORT_CORRECTNESS $ARM arm produced $EMPTY empty completions" 7
    fi
    echo "$PROBE_END" > "$OUT/probe_end_${NAME}.txt"
    [ "$ARM" = "OFF" ] && PROBE_DONE_OFF=1 || PROBE_DONE_ON=1
  fi

  # warm-up: 8 sequential requests, discarded (inherited).  NB E2C-16: this traffic is NOT
  # excluded from the estimator population; post-hoc bench-only re-scoring may NOT re-label R1.
  CAP=$(remaining); [ "$CAP" -lt 1 ] && CAP=1
  timeout -k 30 "$CAP" "${WARM[@]}" > "$OUT/warmup_${NAME}.log" 2>&1
  echo "warmup_rc=$?"

  # ---- the measurement.  P7: `timeout T_cap` -> CAPPED (pair UNRESOLVED).  If the hard-cap
  # deadline is nearer than T_cap, the deadline is the timeout and a cut is a BUDGET kill.
  BT=$TCAP; KIND=P7_CAPPED
  R=$(remaining); [ "$R" -lt "$BT" ] && { BT=$R; KIND=BUDGET; }; [ "$BT" -lt 1 ] && BT=1
  cpu_snap "$OUT/cpu_${NAME}.txt" before_bench
  T_BS=$(date +%s)
  timeout -k 30 "${BT}s" "${BENCH[@]}" > "$OUT/bench_${NAME}.log" 2>&1
  BRC=$?
  T_BE=$(date +%s)
  cpu_snap "$OUT/cpu_${NAME}.txt" after_bench
  echo "bench_rc=$BRC cell=$NAME"
  CUT=""
  if [ "$BRC" = 124 ] || [ "$BRC" = 137 ] || budget_fired; then
    if [ "$KIND" = BUDGET ] || budget_fired; then
      CUT=BUDGET; echo "BENCH_KILLED_BY_BUDGET $NAME rc=$BRC" | tee -a "$OUT/BUDGET_STOP.txt"
    else
      CUT=P7_CAPPED; echo "CAPPED $NAME hit T_cap=${TCAP}s" | tee -a "$OUT/CAPPED.txt"
    fi
  fi
  grep -E "Successful requests|Request throughput|Total input tokens|Median TTFT|P99 TTFT|Median ITL|Concurrency:" \
    "$OUT/bench_${NAME}.log"

  # ---- rev3 sec 5 estimators: the decision input.  Failure handling is rev3's: a missing
  # or empty cell JSON makes the pair UNRESOLVED in e2_label.py (never a substitute value).
  python3 "$PREREG/e2_realized_mix.py" "$PDMUX_TELEMETRY_PATH" \
    --cell "$CELL" --arm "$ARM" --seed "$CSEED" \
    ${PROBE_END:+--probe-end "$PROBE_END"} \
    --bench "$OUT/bench_${NAME}.jsonl" --bench-rc "$BRC" --num-prompts "$NP" \
    --input-len "$INLEN" --output-len "$OUTLEN" \
    --out "$OUT/cell_${NAME}.json" > "$OUT/analyze_${NAME}.log" 2>&1 \
    || echo "ANALYZE_FAILED cell=$NAME"
  tail -30 "$OUT/analyze_${NAME}.log"
  du -h "$PDMUX_TELEMETRY_PATH" | sed 's/^/  telemetry /'

  teardown "$PID" "$PORT"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$NAME" "$T_ADMIT" "$T_HEALTH" "$T_BS" "$T_BE" "$(date +%s)" "$BRC" "${CUT:-none}" >> "$OUT/TIMING.tsv"
  # complete = a registered outcome exists (bench ran to the end, or was P7 CAPPED).  A boot cut
  # by the BUDGET enforcer is NOT complete, so a resume re-runs its pair (A13 sec 5-4).
  if [ "$CUT" != BUDGET ] && [ -s "$OUT/cell_${NAME}.json" ]; then
    date -u +%Y-%m-%dT%H:%M:%SZ > "$OUT/boot_${NAME}.complete"
  fi
  if budget_fired; then STOP=budget; echo "UNRESOLVED_BUDGET after $NAME (hard cap fired)" | tee -a "$OUT/BUDGET_STOP.txt"; break; fi
done
kill "$WD_PID" 2>/dev/null
echo "remaining_at_loop_end_s=$(remaining)" | tee -a "$OUT/BUDGET.txt"

echo ""
echo "=== P3 summary (the gate itself fired per boot, above) ==="
sort -u "$OUT/P3_KEYS.txt" 2>/dev/null | tee "$OUT/P3_BANNER_UNIQUE.txt"
echo "P3 distinct binding-resource keys across all boots: $(sort -u "$OUT/P3_KEYS.txt" 2>/dev/null | wc -l) (registered: 1)"

# ---- P13 verdict: byte-identical greedy output across arms (literal, ADDENDUM A7).
# Both files come from THIS Job; a fully resumed Job compares the previous Job's pair of files.
G_OFF="$OUT/greedy_OFF.json"; G_ON="$OUT/greedy_ON.json"
if [ ! -f "$G_OFF" ] && [ ! -f "$G_ON" ] && [ -n "$RESUME_FROM" ]; then
  G_OFF="$OUT/greedy_OFF_from_${RESUME_FROM}.json"; G_ON="$OUT/greedy_ON_from_${RESUME_FROM}.json"
fi
if [ -f "$G_OFF" ] && [ -f "$G_ON" ]; then
  timeout -k 10 "$(tail_s)" python3 - "$G_OFF" "$G_ON" <<'PY' | tee "$OUT/P13_VERDICT.txt"
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
echo "=== green-context read-out (E2C-38, DESCRIPTIVE -- not a decision input) ==="
timeout -k 10 "$(tail_s)" python3 "$HELPER" --green-summary "$OUT" > "$OUT/GREEN_SUMMARY.json" || echo "GREEN_SUMMARY_FAILED"

echo ""
echo "=== registered rules (PREREG_E2_STICKY_REV3 sec 6 + ADDENDUM A4/A5/A6/A11/A12) ==="
if [ "$DRYRUN" = 1 ]; then
  timeout -k 10 "$(tail_s)" python3 "$PREREG/e2_label.py" --cells-dir "$OUT" --expect "$EXPECT" \
    --out "$OUT/DRYRUN_LABEL_ALL_MISSING.json" > /dev/null || echo "LABEL_FAILED"
else
  timeout -k 10 "$(tail_s)" python3 "$PREREG/e2_label.py" --cells-dir "$OUT" --expect "$EXPECT" \
    --out "$OUT/E2_LABEL.json" || echo "LABEL_FAILED"
fi

echo "E2V_DONE_${JOBID} remaining_s=$(remaining)"
