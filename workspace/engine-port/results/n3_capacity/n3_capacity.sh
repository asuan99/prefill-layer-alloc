#!/bin/bash
# N3 -- gate #6: SLO capacity of the target-D44 sticky-OFF B1 arm (Nemotron-Nano-9B-v2, VESSL A100).
# Pre-registration: PREREG_N3_CAPACITY_2026-10-04.md (rev4).  One shape per Job:
#   PDMUX_N3_SHAPE=B  (E1a: 8192 in / 64 out)      PDMUX_N3_SHAPE=A  (E1b: 256 in / 512 out)
#
# JOB STRUCTURE (n3_plan.units):
#   [preflight] -> floor_head (P-2: T_K.json written BEFORE any arm cell) -> config bridge (2 boots,
#   PHASE_EVENTS OFF) -> observer block (4 ON/OFF pairs, ABBA; the ON cells are the grid cells of the
#   observer rate) -> grid rounds (4 seeds, ascending/descending) -> floor_tail (diagnostic) -> [tail]
# ARM (every boot): legacy loop, PDMUX_R2_POLICY=fixed, PDMUX_R2_FIXED_DSM=44, sticky UNSET (and
#   proven null by the per-boot green read-out), cudagraph ON, config pdmux_n3_A_agn.yml.
# THE ONLY PER-BOOT KNOB: PDMUX_PHASE_EVENTS (1 on floor/grid/obs_on, unset on obs_off/bridge).
#
# usage (inside a VESSL Job via scripts/vessl/launch.sh; never by hand on a GPU):
#   launch.sh --campaign n3_capacity --env PDMUX_N3_SHAPE=B \
#     --target workspace/engine-port/results/n3_capacity/n3_capacity.sh
# optional: --env PDMUX_N3_RESUME_FROM=<previous job name>      (prereg §8-4)
# local CPU-only: PDMUX_N3_DRYRUN=1 (no server is started; no cell is measured).
set -uo pipefail

ROOT="${PDMUX_PROJECT_ROOT:?PDMUX_PROJECT_ROOT must be set (job_entry.sh exports it)}"
ENG="$ROOT/workspace/engine-port"
HERE="$ENG/results/n3_capacity"
HELPER="$HERE/n3_plan.py"
MIX="$HERE/n3_realized_mix.py"
ANA="$HERE/n3_analyze.py"
PREREG_MD="$HERE/PREREG_N3_CAPACITY_2026-10-04.md"
DATA="${PDMUX_DATA:-/data}"
DRYRUN="${PDMUX_N3_DRYRUN:-0}"
SHAPE="${PDMUX_N3_SHAPE:-}"

JOBID="${PDMUX_JOB_NAME:-local$$}"
case "$SHAPE" in A|B) ;; *) echo "ABORT_ENV PDMUX_N3_SHAPE must be A or B (got '$SHAPE')"; exit 2;; esac
SLOW=$(echo "$SHAPE" | tr 'AB' 'ab')
OUT="${PDMUX_N3_TEST_OUT_ROOT:-$HERE}/n3${SLOW}_${JOBID}"     # TEST_OUT_ROOT: dry-run only
mkdir -p "$OUT"
cd "$ROOT" || exit 2
[ "$DRYRUN" = 1 ] && echo "DRY RUN -- NOT A MEASUREMENT (no server started)" > "$OUT/DRYRUN"
WD_PID=""
cleanup () {
  [ -n "$WD_PID" ] && kill "$WD_PID" 2>/dev/null
  rm -rf "$HERE/__pycache__" "$ENG/results/r2_eval/lambda0_prereg/__pycache__" \
         "$ENG/benchmarks/pdmux_eval/__pycache__" 2>/dev/null
}
trap cleanup EXIT
export PYTHONDONTWRITEBYTECODE=1
fail () { echo "$1" | tee -a "$OUT/PREFLIGHT_FAILURES.txt"; exit "$2"; }
dry_or_abort () { if [ "$DRYRUN" = 1 ]; then echo "DRYRUN_NOTE $1"; else fail "$1" "$2"; fi; }

# ------------------------------------------------------ PDMUX_* knob allowlist (prereg §8-2)
# ANY PDMUX_* not on this list aborts.  PDMUX_PHASE_EVENTS is listed because this registration
# sets it ITSELF per boot (the inherited value is recorded, then overwritten / unset per unit).
ALLOW=" PDMUX_COMMIT PDMUX_JOB_NAME PDMUX_CAMPAIGN PDMUX_TARGET PDMUX_IMAGE PDMUX_SPEC \
PDMUX_ROOT PDMUX_PROJECT_ROOT PDMUX_IO PDMUX_DATA PDMUX_OPT PDMUX_N3_SHAPE PDMUX_N3_RESUME_FROM \
PDMUX_N3_DRYRUN PDMUX_N3_TEST_HARD_CAP_S PDMUX_N3_TEST_FAKE_BENCH_S PDMUX_N3_TEST_SKIP_PREFLIGHT \
PDMUX_N3_TEST_OUT_ROOT PDMUX_TEST_SKIP_SYNC PDMUX_ARRAY_SPEC PDMUX_PHASE_EVENTS "
for v in $(compgen -e | grep '^PDMUX_' || true); do
  case "$ALLOW" in *" $v "*) ;; *) fail "ABORT_ENV unregistered knob $v is set (prereg §8-2)" 2;; esac
done
[ -n "${PDMUX_ARRAY_SPEC:-}" ] && fail "ABORT_ENV PDMUX_ARRAY_SPEC must be empty (one Job per shape)" 2
if [ "$DRYRUN" != 1 ]; then
  for v in PDMUX_N3_TEST_HARD_CAP_S PDMUX_N3_TEST_FAKE_BENCH_S PDMUX_N3_TEST_SKIP_PREFLIGHT \
           PDMUX_N3_TEST_OUT_ROOT PDMUX_TEST_SKIP_SYNC; do
    [ -n "${!v:-}" ] && fail "ABORT_TEST_KNOB_IN_MEASUREMENT $v" 2
  done
fi
echo "inherited_PDMUX_PHASE_EVENTS=${PDMUX_PHASE_EVENTS:-<unset>} (overwritten per unit)" > "$OUT/KNOBS.txt"

export HF_HOME="${HF_HOME:-$DATA/hf}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
unset SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN
unset PDMUX_SLO_SCHED PDMUX_SLO_MODE PDMUX_LA_COORD PDMUX_STICKY_PARTITION \
      PDMUX_FIXED_DECODE_SM_FILE PDMUX_TRACE_FORCE_PREFILL \
      PDMUX_DUAL_WORKER_TRACE_EVERY PDMUX_TRUE_DUAL_WORKER PDMUX_DUAL_WORKER \
      PDMUX_DUAL_WORKER_TRACE PDMUX_MEM_TELEMETRY PDMUX_PHASE_EVENTS

# ------------------------------------------------------------ registered arm (n3_plan §1)
K () { python3 "$HELPER" --const "$1"; }
MODEL=$(K MODEL) || exit 2
BACKEND=$(K BACKEND); CTX=$(K CTX); MEM_FRACTION=$(K MEM_FRACTION); MAX_RUNNING=$(K MAX_RUNNING)
SERVER_SEED=$(K SERVER_SEED); FIXED_DSM=$(K FIXED_DSM); R2_POLICY=$(K R2_POLICY)
CFG="$ENG/$(K CONFIG_REL)"
SGPT_RAW="$HF_HOME/raw/ShareGPT_V3_unfiltered_cleaned_split.json"
MODEL_DIR="$HF_HOME/hub/models--nvidia--NVIDIA-Nemotron-Nano-9B-v2-Base"

# ----------------------------------------------- wall-clock hard cap (prereg §8-3)
HARD_CAP_S=$(K HARD_CAP_S) || exit 2
TAIL_RESERVE_S=$(K TAIL_RESERVE_S) || exit 2
[ -n "${PDMUX_N3_TEST_HARD_CAP_S:-}" ] && HARD_CAP_S="$PDMUX_N3_TEST_HARD_CAP_S"
T0=""
TIMING="$DATA/runs/$JOBID/meta/timing.txt"
[ -f "$TIMING" ] && T0=$(date -u -d "$(awk -F= '$1=="start_utc"{print $2}' "$TIMING")" +%s 2>/dev/null || true)
[ -z "$T0" ] && T0=$(date +%s)
DEADLINE=$((T0 + HARD_CAP_S - TAIL_RESERVE_S))
remaining () { echo $((DEADLINE - $(date +%s))); }
cap_s () { local r; r=$(remaining); [ "$r" -lt 1 ] && r=1; echo "$r"; }
tail_s () { local r=$((T0 + HARD_CAP_S - $(date +%s))); [ "$r" -lt 1 ] && r=1; echo "$r"; }
budget_fired () { [ -e "$OUT/BUDGET_HARDCAP_FIRED" ]; }
{
  echo "shape=$SHAPE hard_cap_s=$HARD_CAP_S tail_reserve_s=$TAIL_RESERVE_S"
  echo "t0_epoch=$T0 ($( [ -f "$TIMING" ] && echo job_entry_start_utc || echo script_start)) deadline_epoch=$DEADLINE"
  echo "remaining_at_script_start_s=$(remaining)"
} | tee "$OUT/BUDGET.txt"
(
  while [ "$(remaining)" -gt 0 ]; do sleep 5; done
  echo "fired_epoch=$(date +%s)" > "$OUT/BUDGET_HARDCAP_FIRED"
  pkill -TERM -f "sglang.bench_serving" 2>/dev/null
  pkill -TERM -f "sglang.launch_server" 2>/dev/null
  pkill -TERM -f "n3_fake_bench" 2>/dev/null
  sleep 15
  pkill -9 -f "sglang.bench_serving" 2>/dev/null
  pkill -9 -f "sglang.launch_server" 2>/dev/null
  pkill -9 -f "n3_fake_bench" 2>/dev/null
) > /dev/null 2>&1 &
WD_PID=$!
echo "N3 capacity shape=$SHAPE job=$JOBID model=$MODEL cfg=$CFG dryrun=$DRYRUN"

# --------------------------------------- registration integrity (prereg §8-1)
[ -f "$PREREG_MD" ] || fail "ABORT_REG pre-registration missing: $PREREG_MD" 3
timeout -k 10 "$(cap_s)" python3 "$HELPER" --verify-digests "$ENG" | tee "$OUT/DIGEST_CHECK.txt"
[ "${PIPESTATUS[0]}" -eq 0 ] || fail "ABORT_REG_DRIFT config / analyze.py / lambda0_plan.py / src mirror differ from the registered bytes" 3
{
  echo "# pre-registration in force: $(basename "$PREREG_MD")"
  sha256sum "$PREREG_MD"
  echo "# decision path + runner + descriptive estimator + served config + canonical predicate"
  sha256sum "$HERE/n3_capacity.sh" "$HELPER" "$ANA" "$MIX" "$CFG" "$ENG/benchmarks/pdmux_eval/analyze.py" \
            "$ENG/results/r2_eval/lambda0_prereg/lambda0_plan.py"
  echo "# every verdict kept beside it"
  ls "$HERE"/VERDICT_n3_*.md >/dev/null 2>&1 && sha256sum "$HERE"/VERDICT_n3_*.md
  echo "# tree: $(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo NO_GIT)" \
       "uncommitted_outside_results=$(git -C "$ROOT" status --porcelain --untracked-files=no -- . ':!workspace/engine-port/results' 2>/dev/null | grep -c .)"
} | tee "$OUT/REGISTRATION_SHA256.txt"
N_SHA=$(grep -c "^[0-9a-f]\{64\} " "$OUT/REGISTRATION_SHA256.txt")
[ "${N_SHA:-0}" -ge 8 ] || fail "ABORT_REG only $N_SHA digests recorded, registered minimum 8" 3

# ---- config (clause 1 + preflight): FIXED_DSM in the division set, groups <= --sm-group-num
python3 "$HELPER" --check-config "$CFG" | tee "$OUT/CONFIG_CHECK.txt"
[ "${PIPESTATUS[0]}" -eq 0 ] || fail "ABORT_CONFIG PDMUX_R2_FIXED_DSM=$FIXED_DSM not a division decode SM / group count / target pair (see CONFIG_CHECK.txt)" 3

# ------------------------------------------------- substrate record + B0 (prereg §2)
{
  echo "nproc=$(nproc 2>/dev/null)"
  echo "cpu_affinity=$(taskset -pc $$ 2>/dev/null | awk -F': ' '{print $2}')"
  echo "cgroup_cpu_max=$(cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo NA)"
  echo "cgroup_mem_max=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo NA)"
  echo "mem_total_kb=$(awk '/MemTotal/{print $2}' /proc/meminfo 2>/dev/null)"
  echo "python=$(python3 -c 'import sys;print(sys.version.split()[0])' 2>/dev/null)"
  echo "perf_counter_clock=$(python3 -c 'import time;print(time.get_clock_info("perf_counter").implementation)' 2>/dev/null)"
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
} > "$OUT/SUBSTRATE_N3.txt" 2>&1
cat "$OUT/SUBSTRATE_N3.txt"
B0_CSV=$(nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version,mig.mode.current \
         --format=csv,noheader,nounits 2>/dev/null | head -1)
B0_SM=$(python3 -c 'import torch;print(torch.cuda.get_device_properties(0).multi_processor_count)' 2>/dev/null || echo NA)
python3 "$HELPER" --check-substrate "${B0_CSV:-NONE}" "$B0_SM" | tee "$OUT/SUBSTRATE_CHECK.txt"
[ "${PIPESTATUS[0]}" -eq 0 ] || dry_or_abort "ABORT_SUBSTRATE not an A100 80GB / cc 8.0 / 108 SM / MIG-off / driver>=550 device ($B0_CSV, sm=$B0_SM)" 3
[ "$(cat "$MODEL_DIR/refs/main" 2>/dev/null || echo ABSENT)" = "$(K MODEL_REVISION)" ] \
  || dry_or_abort "ABORT_MODEL_REVISION refs/main != $(K MODEL_REVISION)" 3
[ "$(sha256sum "$SGPT_RAW" 2>/dev/null | cut -d' ' -f1)" = "$(K SHAREGPT_SHA256)" ] \
  || dry_or_abort "ABORT_TRACE ShareGPT sha256 != $(K SHAREGPT_SHA256)" 3
grep -q "perf_counter_clock=clock_gettime(CLOCK_MONOTONIC)" "$OUT/SUBSTRATE_N3.txt" \
  && grep -q "monotonic_clock=clock_gettime(CLOCK_MONOTONIC)" "$OUT/SUBSTRATE_N3.txt" \
  || dry_or_abort "ABORT_CLOCK perf_counter/monotonic not CLOCK_MONOTONIC: server events and bench marks would not share a clock" 3

SKIP_PRE="${PDMUX_N3_TEST_SKIP_PREFLIGHT:-0}"
if [ "$SKIP_PRE" != 1 ]; then
  echo "=== env sync (runtime source manifest) ==="
  timeout -k 10 "$(cap_s)" "$ENG/scripts/bootstrap/sync_engine_tree.sh" "$OUT/runtime_source_manifest.sha256" || exit 2
  # prereg §8-1: the installed tree must be the registered HEAD manifest, entry for entry
  python3 "$HELPER" --engine-check "$OUT/runtime_source_manifest.sha256" | tee "$OUT/ENGINE_TREE_CHECK.txt"
  [ "${PIPESTATUS[0]}" -eq 0 ] || dry_or_abort "ABORT_ENGINE_TREE_DRIFT installed tree != registered HEAD manifest (n3_plan.ENGINE_MANIFEST)" 3
  PSTATE="${SGLANG_ENGINE_DEV:-/opt/pdmux/sglang_engine_dev/python}/sglang/srt/distributed/parallel_state.py"
  grep -q "def pdmux_role_is_thread_local" "$PSTATE" 2>/dev/null \
    || dry_or_abort "ABORT_ENGINE installed parallel_state.py lacks the thread-local role patch (correctness gate 3)" 3
  # CPU selftests (Job preflight = fast checks of THIS tree and the rule code)
  timeout -k 10 "$(cap_s)" python3 "$HELPER" --selftest | tee "$OUT/SELFTEST_plan.txt"
  [ "${PIPESTATUS[0]}" -eq 0 ] || fail "ABORT_SELFTEST n3_plan" 4
  timeout -k 10 "$(cap_s)" python3 "$MIX" --selftest | tee "$OUT/SELFTEST_mix.txt"
  [ "${PIPESTATUS[0]}" -eq 0 ] || fail "ABORT_SELFTEST n3_realized_mix (synthetic part; archive part is local-only)" 4
  timeout -k 10 "$(cap_s)" python3 "$ANA" --selftest | tee "$OUT/SELFTEST_analyze.txt"
  [ "${PIPESTATUS[0]}" -eq 0 ] || fail "ABORT_SELFTEST n3_analyze" 4
  (cd "$ENG/tests" && timeout -k 10 "$(cap_s)" python3 -m unittest -q test_phase_events test_green_readout \
      test_n3_capacity.TestRegistration test_n3_capacity.TestUnits test_n3_capacity.TestFailClosedWiring \
      test_n3_capacity.TestPortAndTeardown) \
    > "$OUT/PREFLIGHT_TESTS.txt" 2>&1 || { tail -20 "$OUT/PREFLIGHT_TESTS.txt"; fail "ABORT_PREFLIGHT unit tests (see PREFLIGHT_TESTS.txt)" 4; }
else
  echo "TEST_SKIP_PREFLIGHT: sync/manifest/selftests NOT run (dry-run only)" | tee "$OUT/TEST_SKIP_PREFLIGHT"
fi

# ------------------------------------------------------------ units (n3_plan.units)
mapfile -t UNITS < <(python3 "$HELPER" --units "$SHAPE")
[ "${#UNITS[@]}" -ge 10 ] || fail "ABORT_UNIT_COUNT ${#UNITS[@]}" 2
printf '%s\n' "${UNITS[@]}" > "$OUT/UNITS.txt"
python3 "$HELPER" --budget "$SHAPE" > "$OUT/BUDGET_TABLE.json"
echo "units=${#UNITS[@]} shape=$SHAPE"

# ------------------------------------------------------------ resume (prereg §8-4)
RESUME_FROM="${PDMUX_N3_RESUME_FROM:-}"
declare -A RESUMED=()
if [ -n "$RESUME_FROM" ]; then
  PREV="$DATA/runs/$RESUME_FROM/results/n3_capacity/n3${SLOW}_$RESUME_FROM"
  [ -d "$PREV" ] || fail "ABORT_RESUME previous run dir not found: $PREV" 2
  cmp -s "$PREV/UNITS.txt" "$OUT/UNITS.txt" || fail "ABORT_RESUME UNITS.txt differs from $RESUME_FROM" 2
  diff <(grep -E "n3_|PREREG_|VERDICT_|pdmux_n3|analyze.py|lambda0_plan" "$PREV/REGISTRATION_SHA256.txt" | awk '{print $1}') \
       <(grep -E "n3_|PREREG_|VERDICT_|pdmux_n3|analyze.py|lambda0_plan" "$OUT/REGISTRATION_SHA256.txt" | awk '{print $1}') > /dev/null \
    || fail "ABORT_RESUME registration digests differ from $RESUME_FROM" 2
  python3 "$HELPER" --tk-valid "$PREV/T_K.json" > "$OUT/RESUME_TK_CHECK.txt" 2>&1 \
    || fail "ABORT_RESUME previous Job has no valid T_K.json (dry-run placeholder / no ttfts_s / not k x floor -- its cells cannot be scored, R2)" 2
  cp "$PREV/T_K.json" "$OUT/T_K.json"            # P-2: T frozen in SECONDS from the first Job
  PREV_GPU=$(python3 -c "import json;g=json.load(open('$DATA/runs/$RESUME_FROM/meta/substrate.json'))['gpu'];print(g['uuid'],g['driver_version'])" 2>/dev/null || echo "UNKNOWN UNKNOWN")
  printf 'unit\tfrom_job\tfrom_gpu_uuid\tfrom_driver\n' > "$OUT/RESUMED_UNITS.tsv"
  for U in "${UNITS[@]}"; do
    read -r N _ <<< "$U"
    [ "$N" = floor_head ] && continue            # the floor is re-measured here as a diagnostic
    if [ -e "$PREV/unit_${N}.complete" ]; then
      for f in "$PREV"/*_"${N}".*; do cp "$f" "$OUT/"; done     # no -p: exported
      RESUMED["$N"]=1
      printf '%s\t%s\t%s\n' "$N" "$RESUME_FROM" "$(echo "$PREV_GPU" | tr ' ' '\t')" >> "$OUT/RESUMED_UNITS.tsv"
    fi
  done
  echo "RESUMED $(($(wc -l < "$OUT/RESUMED_UNITS.tsv") - 1)) unit(s) from $RESUME_FROM (Job mix recorded)"
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
# ---- port selection + teardown (prereg §8-6; E2 diagnosis, A14 DRAFT O1-O4 applied equivalently)
teardown () {   # $1 pid  $2 port  (inherited sequence: kill, 5 s, pkill -9, 10 s) + O3
  kill "$1" 2>/dev/null; sleep 5
  pkill -9 -f "launch_server.*--port $2" 2>/dev/null; sleep 10
  # O3: SGLang children rewrite their cmdline via setproctitle ("sglang::scheduler",
  # "sglang::detokenizer"), so the pattern above never matches them.  One server per container.
  pkill -9 -f "^sglang::" 2>/dev/null
}
# O4: before every launch wait until no SGLang process is alive and GPU used memory is back at the
# pre-first-boot baseline (+64 MiB), max 180 s.  RECORDED, NOT a gate (ABORT_P3 stays the gate).
gpu_used_mib () { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 2>/dev/null | head -1 | tr -d ' '; }
nvml_bytes () {
  python3 -c 'import pynvml as n; n.nvmlInit(); m=n.nvmlDeviceGetMemoryInfo(n.nvmlDeviceGetHandleByIndex(0)); print(f"nvml_total={m.total} nvml_free={m.free} nvml_used={m.used}")' 2>/dev/null \
    || echo "nvml=NA"
}
settle () {   # $1 unit name
  local i used alive
  for i in $(seq 1 36); do
    used=$(gpu_used_mib); alive=0
    pgrep -f "sglang.launch_server|^sglang::" > /dev/null 2>&1 && alive=1
    if [ "$alive" = 0 ] && [ -n "$used" ] && [ "$used" -le $((GPU_USED_BASE_MIB + 64)) ]; then break; fi
    budget_fired && break
    sleep 5
  done
  echo "$1 epoch=$(date +%s) settle_polls=$i used_mib=${used:-NA} base_mib=$GPU_USED_BASE_MIB sglang_alive=$alive $(nvml_bytes)" >> "$OUT/SETTLE.txt"
}
# O1: the HTTP port is drawn OUTSIDE the kernel ephemeral range (E2 boot failure: 39184 was the
# previous bench client's ephemeral port, still in TIME_WAIT -> EADDRINUSE).  Overlap = abort.
PORT_LO=20000; PORT_SPAN=800
read -r EPH_LO EPH_HI < /proc/sys/net/ipv4/ip_local_port_range 2>/dev/null || { EPH_LO=NA; EPH_HI=NA; }
{ echo "ip_local_port_range=$EPH_LO $EPH_HI"
  echo "ip_local_reserved_ports=$(cat /proc/sys/net/ipv4/ip_local_reserved_ports 2>/dev/null || echo NA)"
  echo "listen_port_window=$PORT_LO-$((PORT_LO + PORT_SPAN - 1))"; } | tee "$OUT/PORT_RANGE.txt"
if [ "$EPH_LO" = NA ] || { [ $((PORT_LO + PORT_SPAN - 1)) -ge "$EPH_LO" ] && [ "$PORT_LO" -le "$EPH_HI" ]; }; then
  fail "ABORT_PORT_RANGE listen window $PORT_LO+$PORT_SPAN overlaps ephemeral range $EPH_LO-$EPH_HI (or unreadable)" 5
fi
# O2: a candidate is free only if NO TCP socket in ANY state (incl. TIME_WAIT) has it as local port.
port_in_use () { ss -Htan 2>/dev/null | awk '{print $4}' | grep -q ":$1\$"; }
GPU_USED_BASE_MIB=$(gpu_used_mib); GPU_USED_BASE_MIB=${GPU_USED_BASE_MIB:-0}
echo "baseline epoch=$(date +%s) used_mib=$GPU_USED_BASE_MIB $(nvml_bytes)" > "$OUT/SETTLE.txt"
cpu_snap () { { echo "# $2 epoch=$(date +%s)"; cat /sys/fs/cgroup/cpu.stat 2>/dev/null || echo "cpu.stat NA"; } >> "$1"; }
mono () { python3 -c 'import time;print(repr(time.monotonic()))'; }
tk_valid () { python3 "$HELPER" --tk-valid "$1" > /dev/null 2>&1; }   # R2: refuses the dry-run placeholder

FAIL_STREAK=0
STOP=""
CG2_DONE=0
printf 'unit\tadmit_epoch\thealth_epoch\tbench_start_epoch\tbench_end_epoch\tend_epoch\tbench_rc\tcut\n' > "$OUT/TIMING.tsv"

for U in "${UNITS[@]}"; do
  read -r NAME KIND USHAPE RATE NP CSEED INLEN OUTLEN PEV BLOCK <<< "$U"
  if [ -n "${RESUMED[$NAME]:-}" ] || [ -e "$OUT/unit_${NAME}.complete" ]; then echo "SKIP_DONE unit=$NAME"; continue; fi
  [ "$STOP" = boot ] && { echo "NOT_ATTEMPTED_AFTER_BOOT_STREAK unit=$NAME" | tee -a "$OUT/BOOT_FAILURES.txt"; continue; }
  if [ "$STOP" = budget ] || budget_fired; then echo "NOT_ATTEMPTED_BUDGET unit=$NAME" | tee -a "$OUT/BUDGET_SKIPPED.txt"; continue; fi
  # P-2: no arm cell before a valid T_K.json exists (fail-closed; dry-run notes it)
  if [ "$KIND" != floor ] && ! tk_valid "$OUT/T_K.json"; then
    dry_or_abort "ABORT_FLOOR no valid T_K.json before arm unit $NAME (P-2)" 7
  fi

  # ---- admission (prereg §8-3): remaining budget must cover this unit's WORST-corner cost
  NEED=$(python3 "$HELPER" --need "$SHAPE" "$NAME") || NEED=999999
  [ -n "${PDMUX_N3_TEST_FAKE_BENCH_S:-}" ] && NEED=$(( (NEED + 999) / 1000 ))     # enforcer test only (DRYRUN)
  REM=$(remaining)
  if [ "$REM" -lt "$NEED" ]; then
    echo "NOT_ATTEMPTED_BUDGET unit=$NAME need_s=$NEED remaining_s=$REM" | tee -a "$OUT/BUDGET_SKIPPED.txt"
    STOP=budget; continue
  fi
  echo "admit unit=$NAME need_s=$NEED remaining_s=$REM" >> "$OUT/BUDGET.txt"
  T_ADMIT=$(date +%s)

  [ "$DRYRUN" = 1 ] || settle "$NAME"
  PORT=0
  for try in $(seq 1 50); do
    CAND=$((PORT_LO + (RANDOM + $$ + try) % PORT_SPAN))
    if ! curl -s -o /dev/null --max-time 1 "http://127.0.0.1:$CAND/health" 2>/dev/null; then
      if ! port_in_use "$CAND"; then PORT=$CAND; break; fi
    fi
  done
  [ "$PORT" = 0 ] && fail "ABORT_NO_FREE_PORT" 5
  echo "$NAME port=$PORT tries=$try" >> "$OUT/PORT_RANGE.txt"

  SRVLOG="$OUT/srv_${NAME}.log"
  export PDMUX_TELEMETRY_PATH="$OUT/tel_${NAME}.jsonl"
  export PDMUX_RUN_ID="n3_${NAME}_${JOBID}"
  export PDMUX_WORKLOAD_ID="n3_${NAME}"
  export PDMUX_R2_POLICY="$R2_POLICY"
  export PDMUX_R2_FIXED_DSM="$FIXED_DSM"
  export PDMUX_GREEN_READOUT=1 PDMUX_GREEN_READOUT_PATH="$OUT/green_${NAME}.json"
  # ===================== THE ONE PER-BOOT KNOB (D5) =====================
  if [ "$PEV" = 1 ]; then export PDMUX_PHASE_EVENTS=1; else unset PDMUX_PHASE_EVENTS; fi
  # ======================================================================

  echo ""
  echo "########## $NAME kind=$KIND shape=$SHAPE rate=$RATE np=$NP seed=$CSEED phase_events=$PEV port=$PORT remaining_s=$REM ##########"
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
    --random-input-len "$INLEN" --random-output-len "$OUTLEN" --random-range-ratio 1.0
    --num-prompts 8 --max-concurrency 1 --seed 7
    --output-details --output-file "$OUT/warmup_${NAME}.jsonl")      # CG2 input (client-side file only)
  if [ "$KIND" = floor ] || [ "$KIND" = floor_tail ]; then
    BENCH=(python -m sglang.bench_serving --backend sglang --model "$MODEL"
      --host 127.0.0.1 --port "$PORT"
      --dataset-name random --dataset-path "$SGPT_RAW" --tokenize-prompt
      --random-input-len "$INLEN" --random-output-len "$OUTLEN" --random-range-ratio 1.0
      --num-prompts "$NP" --max-concurrency 1 --seed "$CSEED"
      --output-details --output-file "$OUT/bench_${NAME}.jsonl")
  else
    BENCH=(python -m sglang.bench_serving --backend sglang --model "$MODEL"
      --host 127.0.0.1 --port "$PORT"
      --dataset-name random --dataset-path "$SGPT_RAW" --tokenize-prompt
      --random-input-len "$INLEN" --random-output-len "$OUTLEN" --random-range-ratio 1.0
      --num-prompts "$NP" --request-rate "$RATE" --seed "$CSEED"
      --output-details --output-file "$OUT/bench_${NAME}.jsonl")
  fi

  if [ "$DRYRUN" = 1 ]; then
    { printf 'UNIT %s kind=%s block=%s\n  server:' "$NAME" "$KIND" "$BLOCK"; printf ' %q' "${SRV[@]}"; echo
      printf '  warmup:'; printf ' %q' "${WARM[@]}"; echo
      printf '  bench :'; printf ' %q' "${BENCH[@]}"; echo
      echo "  env   : PDMUX_TELEMETRY_PATH=$PDMUX_TELEMETRY_PATH PDMUX_RUN_ID=$PDMUX_RUN_ID PDMUX_R2_POLICY=$PDMUX_R2_POLICY PDMUX_R2_FIXED_DSM=$PDMUX_R2_FIXED_DSM PDMUX_GREEN_READOUT=$PDMUX_GREEN_READOUT PDMUX_PHASE_EVENTS=${PDMUX_PHASE_EVENTS:-<unset>} PDMUX_STICKY_PARTITION=${PDMUX_STICKY_PARTITION:-<unset>}"
    } >> "$OUT/DRYRUN_COMMANDS.txt"
    if [ "$KIND" = floor ] && [ ! -e "$OUT/T_K.json" ]; then
      echo '{"valid": true, "dryrun": true, "floor_s": 1.0, "T_k_s": {"3": 3.0, "5": 5.0, "10": 10.0}, "NOTE": "DRY RUN placeholder -- not a floor"}' > "$OUT/T_K.json"
    fi
    if [ -n "${PDMUX_N3_TEST_FAKE_BENCH_S:-}" ]; then
      BT=$(cap_s)
      timeout -k 2 "${BT}s" bash -c 'exec -a n3_fake_bench sleep "$1"' _ "$PDMUX_N3_TEST_FAKE_BENCH_S"
      FRC=$?
      echo "fake_rc=$FRC unit=$NAME" | tee -a "$OUT/FAKE_BENCH.txt"
      if [ "$FRC" = 124 ] || [ "$FRC" = 137 ] || [ "$FRC" = 143 ] || budget_fired; then
        echo "BENCH_KILLED_BY_BUDGET unit=$NAME rc=$FRC" | tee -a "$OUT/BUDGET_SKIPPED.txt"; STOP=budget
      fi
    fi
    continue
  fi

  : > "$PDMUX_TELEMETRY_PATH"
  setsid "${SRV[@]}" > "$SRVLOG" 2>&1 &
  PID=$!
  if ! wait_health "$PORT" "$PID"; then
    echo "BOOT_FAILED unit=$NAME" | tee -a "$OUT/BOOT_FAILURES.txt"
    tail -40 "$SRVLOG" | grep -viE "pynvml|Future" | tee -a "$OUT/BOOT_FAILURES.txt"
    budget_fired || FAIL_STREAK=$((FAIL_STREAK + 1))
    teardown "$PID" "$PORT"
    printf '%s\t%s\tNA\tNA\tNA\t%s\tNA\tboot_failed\n' "$NAME" "$T_ADMIT" "$(date +%s)" >> "$OUT/TIMING.tsv"
    if [ "$KIND" = floor ]; then fail "ABORT_FLOOR floor boot failed -- no T_K, no arm cell (P-2)" 7; fi
    if [ "$FAIL_STREAK" -ge 2 ]; then echo "ABORT_2_CONSECUTIVE_BOOT_FAILURES" | tee -a "$OUT/BOOT_FAILURES.txt"; STOP=boot; fi
    continue
  fi
  T_HEALTH=$(date +%s)
  echo "boot_ok=1 unit=$NAME"
  FAIL_STREAK=0

  # ---- clause 1: sticky UNSET, proven by the server (no ENABLED line) and by the green read-out
  if grep -q "sticky partition ENABLED" "$SRVLOG"; then
    teardown "$PID" "$PORT"; fail "ABORT_STICKY sticky partition ENABLED in $NAME (registered: unset)" 6
  fi
  python3 "$HELPER" --check-green "$OUT/green_${NAME}.json" "$CFG" > "$OUT/greencheck_${NAME}.txt" 2>&1 \
    || { cat "$OUT/greencheck_${NAME}.txt"; teardown "$PID" "$PORT"; fail "ABORT_GREEN read-out of $NAME: a division's green smCount != target, a plain group carries green, or sticky_target_index != null" 6; }
  # ---- clause 8 / P3, option (ii) (prereg §2-3 R6, rev4; user decision 2026-10-05; = E2 A14 §3-1).
  # Over ALL boots of this Job (resumed units are other boots and never enter P3_KEYS.txt):
  #   (a) max_mamba_cache_size == 48 on every boot, exact; (b) every max_total_num_tokens >= 786,432;
  #   (c) (max - min) / max of max_total_num_tokens <= 1e-4.  Rule + constants live in n3_plan.p3_verdict.
  # PASS needs BOTH exit 0 and a stdout line starting "PASS " (stderr to its own file); a missing
  # banner, a missing helper or any other output aborts (fail-closed: empty verdict != PASS).
  MMB=$(grep -o "max_mamba_cache_size: *[0-9]*" "$SRVLOG" | tail -1 | grep -o "[0-9]*$")
  MTT=$(grep -o "max_total_num_tokens=[0-9]*" "$SRVLOG" | tail -1 | grep -o "[0-9]*$")
  echo "max_mamba_cache_size=${MMB:-NONE} max_total_num_tokens=${MTT:-NONE}" | tee "$OUT/banner_${NAME}.txt"
  echo "${MMB:-NONE} ${MTT:-NONE}" >> "$OUT/P3_KEYS.txt"
  P3_VERDICT=$(python3 "$HELPER" --p3-verdict "$OUT/P3_KEYS.txt" "$MAX_RUNNING" "$CTX" 2>>"$OUT/P3_VERDICTS.stderr")
  P3_RC=$?
  echo "$NAME P3 rc=$P3_RC ${P3_VERDICT:-EMPTY}" | tee -a "$OUT/P3_VERDICTS.txt"
  case "$P3_RC:$P3_VERDICT" in
    0:PASS\ *) ;;
    *) teardown "$PID" "$PORT"; fail "ABORT_P3 binding resources fail P3 (ii) (rc=$P3_RC ${P3_VERDICT:-EMPTY}) -- see P3_KEYS.txt / banner_${NAME}.txt" 8 ;;
  esac

  CAP=$(cap_s)
  timeout -k 30 "$CAP" "${WARM[@]}" > "$OUT/warmup_${NAME}.log" 2>&1
  echo "warmup_rc=$?"
  # ---- correctness gate 2 (prereg §4-3): greedy concurrency-1 outputs of the SAME 8 warm-up prompts
  # must be byte-identical between a PHASE_EVENTS ON boot (floor_head) and the first OFF boot of
  # this Job, BEFORE any scored cell.  Mismatch = fail-closed (an investigation item, not evidence).
  if [ "$PEV" = 0 ] && [ "$CG2_DONE" = 0 ]; then
    if [ ! -s "$OUT/warmup_floor_head.jsonl" ]; then
      teardown "$PID" "$PORT"
      fail "ABORT_CG2 ON-side input warmup_floor_head.jsonl missing/empty -- CG2 not evaluable (measurement failure, not evidence of an observer effect)" 9
    fi
    python3 - "$OUT/warmup_floor_head.jsonl" "$OUT/warmup_${NAME}.jsonl" <<'PY' | tee "$OUT/CG2.txt"
import json, sys
a = json.loads(open(sys.argv[1]).readline()).get("generated_texts")
b = json.loads(open(sys.argv[2]).readline()).get("generated_texts")
ok = bool(a) and a == b and len(a) == 8 and all(a)
print(("CG2_PASS" if ok else "CG2_FAIL") + f" on={len(a or [])} off={len(b or [])} identical={a == b}")
sys.exit(0 if ok else 1)
PY
    [ "${PIPESTATUS[0]}" -eq 0 ] || { teardown "$PID" "$PORT"; fail "ABORT_CG2 PHASE_EVENTS ON vs OFF greedy warm-up outputs differ (CG2.txt)" 9; }
    CG2_DONE=1
  fi
  RETR0=$(grep -c "#retracted_reqs:" "$SRVLOG" || true)
  cpu_snap "$OUT/cpu_${NAME}.txt" before_bench
  echo "bench_start $(mono)" > "$OUT/marks_${NAME}.txt"
  T_BS=$(date +%s)
  CAP=$(cap_s)
  timeout -k 30 "${CAP}s" "${BENCH[@]}" > "$OUT/bench_${NAME}.log" 2>&1
  BRC=$?
  T_BE=$(date +%s)
  echo "bench_end $(mono)" >> "$OUT/marks_${NAME}.txt"
  cpu_snap "$OUT/cpu_${NAME}.txt" after_bench
  echo "bench_rc=$BRC unit=$NAME retract_lines_before_bench=$RETR0"
  CUT=""
  if [ "$BRC" = 124 ] || [ "$BRC" = 137 ] || [ "$BRC" = 143 ] || budget_fired; then
    CUT=BUDGET; touch "$OUT/budget_killed_${NAME}"
    echo "BENCH_KILLED_BY_BUDGET unit=$NAME rc=$BRC" | tee -a "$OUT/BUDGET_SKIPPED.txt"
  fi
  grep -E "Successful requests|Request throughput|Median TTFT|P99 TTFT|Median ITL|Concurrency:" "$OUT/bench_${NAME}.log"
  du -h "$PDMUX_TELEMETRY_PATH" | sed 's/^/  telemetry /'
  teardown "$PID" "$PORT"
  # R7: PHASE_EVENTS silently switch themselves off for the rest of the boot after a record/emit
  # failure (multiplexing_mixin `_phase_events_disable`).  Detect it per boot; the analyzer then
  # voids this cell's PHASE_EVENTS descriptives and drops it from the ON/OFF symmetry statements.
  if grep -q "PDMUX_PHASE_EVENTS disabled after failure" "$SRVLOG"; then
    grep -m1 "PDMUX_PHASE_EVENTS disabled after failure" "$SRVLOG" > "$OUT/phase_events_disabled_${NAME}"
    echo "PHASE_EVENTS_DISABLED_MIDBOOT unit=$NAME (descriptives voided, R7)" | tee -a "$OUT/PHASE_EVENTS_DISABLED.txt"
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$NAME" "$T_ADMIT" "$T_HEALTH" "$T_BS" "$T_BE" "$(date +%s)" "$BRC" "${CUT:-none}" >> "$OUT/TIMING.tsv"

  if [ "$KIND" = floor ] || [ "$KIND" = floor_tail ]; then
    # P-2: the registered formula, written before any arm cell (floor_head) / diagnostic (tail, resume)
    if [ "$KIND" = floor ] && [ ! -e "$OUT/T_K.json" ]; then TKOUT="$OUT/T_K.json"
    elif [ "$KIND" = floor ]; then TKOUT="$OUT/T_K_RESUME_${JOBID}.json"
    else TKOUT="$OUT/T_K_TAIL.json"; fi
    python3 "$HELPER" --floor "$SHAPE" "$OUT/bench_${NAME}.jsonl" "$PDMUX_TELEMETRY_PATH" "$TKOUT" | tee "$OUT/floor_${NAME}.log"
    FRC=${PIPESTATUS[0]}
    # rev6 audit R-1: PHASE_EVENTS switched off mid-boot (R7 marker above) => the floor spans were not
    # observed; the floor cannot be validated whatever floor_from says (fail-closed, diagnostic T_K kept)
    if [ "$TKOUT" = "$OUT/T_K.json" ] && [ -e "$OUT/phase_events_disabled_${NAME}" ]; then
      fail "ABORT_FLOOR PHASE_EVENTS disabled mid-boot -- floor spans unobserved (see $OUT/phase_events_disabled_${NAME}); no arm cell runs (P-2, rev6 R-1)" 7
    fi
    if [ "$TKOUT" = "$OUT/T_K.json" ] && [ "$FRC" -ne 0 ]; then
      fail "ABORT_FLOOR floor invalid (see $TKOUT): a span realised off stream 0 = (108,0) (prefill_span_end.stream_idx) or with a decode_iteration inside it, an unpaired/dropped-events span, n_new/queue_len != 1/0 on a floor request, or an incomplete bench -- no arm cell runs (P-2, rev6)" 7
    fi
    [ "$FRC" -eq 0 ] && [ -z "$CUT" ] && date -u +%Y-%m-%dT%H:%M:%SZ > "$OUT/unit_${NAME}.complete"
  else
    # descriptive realised partition (never a decision input; GPU idle here)
    timeout -k 10 "$(tail_s)" python3 "$MIX" "$PDMUX_TELEMETRY_PATH" --cell "$NAME" --seed "$CSEED" --rate "$RATE" \
      --bench "$OUT/bench_${NAME}.jsonl" --marks "$OUT/marks_${NAME}.txt" --cap "$MAX_RUNNING" \
      --out "$OUT/mix_${NAME}.json" > "$OUT/mixlog_${NAME}.txt" 2>&1 || echo "MIX_FAILED unit=$NAME (descriptive only)"
    [ -z "$CUT" ] && [ -s "$OUT/bench_${NAME}.jsonl" ] && date -u +%Y-%m-%dT%H:%M:%SZ > "$OUT/unit_${NAME}.complete"
  fi
  if budget_fired; then STOP=budget; echo "BUDGET hard cap fired after $NAME" | tee -a "$OUT/BUDGET_SKIPPED.txt"; fi
done
kill "$WD_PID" 2>/dev/null
echo "remaining_at_loop_end_s=$(remaining)" | tee -a "$OUT/BUDGET.txt"

echo ""
echo "=== registered decision (prereg §5) + descriptive (§6) ==="
CLK="$DATA/runs/$JOBID/meta/clk.csv"
if [ "$DRYRUN" = 1 ]; then
  echo "DRYRUN: analyzer not run on placeholder T_K (no cells)" | tee "$OUT/DRYRUN_NO_ANALYSIS"
else
  timeout -k 10 "$(tail_s)" python3 "$ANA" --out-dir "$OUT" --shape "$SHAPE" ${CLK:+--clk "$CLK"} \
    --json-out "$OUT/N3_RESULT.json" || echo "ANALYZE_FAILED"
fi
echo "N3_DONE_${SHAPE}_${JOBID} remaining_s=$(remaining)"
