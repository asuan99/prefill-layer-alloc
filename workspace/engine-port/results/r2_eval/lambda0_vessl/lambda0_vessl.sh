#!/bin/bash
# λ0 (campaign stage 0, λ* on the reference arm B1) -- VESSL substrate port of
# `../lambda0_prereg/lambda0.sbatch` (rev5, job 908623).
# Pre-registration: PREREG_LAMBDA0_VESSL_2026-10-04.md (substrate-port addendum to
#   PREREG_LAMBDA0_REV5_2026-09-14.md + its ADDENDUM; rev5 rules are NOT changed).
#
# WHAT IS THE SAME AS rev5 (and how that is enforced, not asserted):
#   * every rule/plan/cell/analyzer/label file is the rev5 file in ../lambda0_prereg,
#     CALLED, never copied; `lambda0_vessl_plan.py --verify-digests` aborts (exit 3)
#     unless all 11 are byte-identical to what job 908623 ran;
#   * the λ_inf predicate reads byte-identical job-907959 inputs (digest-checked);
#   * plan.json must be byte-identical to lam0_908623/plan.json (ABORT_PLAN_DRIFT);
#   * server/warmup/bench command lines, arm (legacy loop, fixed D44, cudagraph ON),
#     cell order, seeds, teardown, FAIL_STREAK rule, label invocation: unchanged.
# WHAT CHANGES (environment wiring only, PREREG_LAMBDA0_VESSL sec 2):
#   * paths from PDMUX_PROJECT_ROOT / HF_HOME (job_entry.sh), no module/venv;
#   * TRITON/XDG/FLASHINFER caches = job_entry's per-run dirs;
#   * output under results/r2_eval/lambda0_vessl/lam0v_<job>/ (exported by job_entry);
#   * the 3 git-ignored 907959 inputs come from /io/staging/lambda0_inputs (sha256-checked);
#   * ★ADDED, never-in-rev5: (a) an IN-SCRIPT wall-clock hard cap (admission rule +
#     per-step `timeout` + watchdog) -- SLURM `--time` no longer exists (E2 portability
#     verdict path 1); (b) a 4-cell variance block AFTER the 11 rev5 cells, in its own
#     subdirectory so the rev5 label never sees it; (c) one-shot green-context read-out
#     per boot + descriptive realized-partition summary per cell; (d) resume.
#
# usage (inside a VESSL Job, via scripts/vessl/launch.sh; never by hand on the GPU):
#   launch.sh --campaign lambda0_vessl \
#     --target workspace/engine-port/results/r2_eval/lambda0_vessl/lambda0_vessl.sh
# optional: --env PDMUX_LAM0V_RESUME_FROM=<previous job name>   (resume, prereg sec 6-4)
# local CPU-only: PDMUX_LAM0V_DRYRUN=1 (no server is started; no cell is measured).
set -uo pipefail

ROOT="${PDMUX_PROJECT_ROOT:?PDMUX_PROJECT_ROOT must be set (job_entry.sh exports it)}"
ENG="$ROOT/workspace/engine-port"
PREREG="$ENG/results/r2_eval/lambda0_prereg"        # rev5 decision path (unchanged)
HERE="$ENG/results/r2_eval/lambda0_vessl"
HELPER="$HERE/lambda0_vessl_plan.py"
MIX="$ENG/results/r2_eval/e2_sticky_prereg/e2_realized_mix.py"   # descriptive only
I3_JOB="$ENG/results/r2_correctness/job_907959"
DATA="${PDMUX_DATA:-/data}"
IO="${PDMUX_IO:-/io}"
STAGE="$IO/staging/lambda0_inputs"
DRYRUN="${PDMUX_LAM0V_DRYRUN:-0}"

JOBID="${PDMUX_JOB_NAME:-local$$}"
OUT="${PDMUX_LAM0V_TEST_OUT_ROOT:-$HERE}/lam0v_${JOBID}"   # TEST_OUT_ROOT: dry-run only
mkdir -p "$OUT/variance"
cd "$ROOT" || exit 2
[ "$DRYRUN" = 1 ] && echo "DRY RUN -- NOT A MEASUREMENT (no server started)" > "$OUT/DRYRUN"
# results/ is exported by job_entry (every file newer than its start marker), so
# bytecode that this script's python3 calls drop next to the rev5/port sources would
# ride along.  Remove only those __pycache__ dirs, on every exit path (incl. aborts).
WD_PID=""
cleanup () {
  [ -n "$WD_PID" ] && kill "$WD_PID" 2>/dev/null
  rm -rf "$PREREG/__pycache__" "$HERE/__pycache__" "$(dirname "$MIX")/__pycache__" 2>/dev/null
}
trap cleanup EXIT

# ------------------------------------------------ PDMUX_* knob allowlist (sec 2-4)
# launch.sh forwards any PDMUX_* with --env.  rev5 unset a fixed list; on VESSL the
# injection surface is open-ended, so ANY PDMUX_* not on this list aborts.
ALLOW=" PDMUX_COMMIT PDMUX_JOB_NAME PDMUX_CAMPAIGN PDMUX_TARGET PDMUX_IMAGE PDMUX_SPEC \
PDMUX_ROOT PDMUX_PROJECT_ROOT PDMUX_IO PDMUX_DATA PDMUX_OPT PDMUX_LAM0V_RESUME_FROM \
PDMUX_LAM0V_DRYRUN PDMUX_LAM0V_TEST_HARD_CAP_S PDMUX_LAM0V_TEST_FAKE_CELL_S \
PDMUX_LAM0V_TEST_SKIP_PREFLIGHT PDMUX_LAM0V_TEST_OUT_ROOT PDMUX_TEST_SKIP_SYNC PDMUX_ARRAY_SPEC "
for v in $(compgen -e | grep '^PDMUX_' || true); do
  case "$ALLOW" in *" $v "*) ;; *)
    echo "ABORT_ENV unregistered knob $v is set (PREREG_LAMBDA0_VESSL sec 2-4)" \
      | tee -a "$OUT/PREFLIGHT_FAILURES.txt"; exit 2;;
  esac
done
[ -n "${PDMUX_ARRAY_SPEC:-}" ] && { echo "ABORT_ENV PDMUX_ARRAY_SPEC must be empty (one Job)" \
  | tee -a "$OUT/PREFLIGHT_FAILURES.txt"; exit 2; }
if [ "$DRYRUN" != 1 ]; then
  for v in PDMUX_LAM0V_TEST_HARD_CAP_S PDMUX_LAM0V_TEST_FAKE_CELL_S \
           PDMUX_LAM0V_TEST_SKIP_PREFLIGHT PDMUX_LAM0V_TEST_OUT_ROOT PDMUX_TEST_SKIP_SYNC; do
    [ -n "${!v:-}" ] && { echo "ABORT_TEST_KNOB_IN_MEASUREMENT $v" \
      | tee -a "$OUT/PREFLIGHT_FAILURES.txt"; exit 2; }
  done
fi

export HF_HOME="${HF_HOME:-$DATA/hf}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
# TRITON_CACHE_DIR / XDG_CACHE_HOME / FLASHINFER_WORKSPACE_BASE: job_entry's per-run values.
unset SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN
unset PDMUX_SLO_SCHED PDMUX_SLO_MODE PDMUX_LA_COORD PDMUX_STICKY_PARTITION \
      PDMUX_FIXED_DECODE_SM_FILE PDMUX_TRACE_FORCE_PREFILL \
      PDMUX_DUAL_WORKER_TRACE_EVERY PDMUX_TRUE_DUAL_WORKER PDMUX_DUAL_WORKER \
      PDMUX_DUAL_WORKER_TRACE PDMUX_MEM_TELEMETRY

# ------------------------------------------------------------ registered arm (rev5)
MODEL="nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base"
BACKEND="flashinfer"
CFG="$ENG/results/longctx_conflict/probes/pdmux_homog5.yml"
CTX=16384
MEM_FRACTION=0.82
MAX_RUNNING=48
SERVER_SEED=1
FIXED_DSM=44
SGPT_RAW="$HF_HOME/raw/ShareGPT_V3_unfiltered_cleaned_split.json"

# ------------------------------------------------------- wall-clock hard cap (prereg sec 6-2)
# Measured from job_entry's start (container time already spent on clone+install
# counts).  The cap itself is a registered literal in lambda0_vessl_plan.py; the
# TEST override is refused above unless DRYRUN=1.
HARD_CAP_S=$(python3 -c "import sys; sys.path.insert(0,'$HERE'); import lambda0_vessl_plan as V; print(V.HARD_CAP_S)") || exit 2
TAIL_RESERVE_S=$(python3 -c "import sys; sys.path.insert(0,'$HERE'); import lambda0_vessl_plan as V; print(V.TAIL_RESERVE_S)") || exit 2
[ -n "${PDMUX_LAM0V_TEST_HARD_CAP_S:-}" ] && HARD_CAP_S="$PDMUX_LAM0V_TEST_HARD_CAP_S"
T0=""
TIMING="$DATA/runs/$JOBID/meta/timing.txt"
if [ -f "$TIMING" ]; then
  T0=$(date -u -d "$(awk -F= '$1=="start_utc"{print $2}' "$TIMING")" +%s 2>/dev/null || true)
fi
[ -z "$T0" ] && T0=$(date +%s)
DEADLINE=$((T0 + HARD_CAP_S - TAIL_RESERVE_S))
remaining () { echo $((DEADLINE - $(date +%s))); }
budget_fired () { [ -e "$OUT/BUDGET_HARDCAP_FIRED" ]; }
{
  echo "hard_cap_s=$HARD_CAP_S tail_reserve_s=$TAIL_RESERVE_S"
  echo "t0_epoch=$T0 ($( [ -f "$TIMING" ] && echo job_entry_start_utc || echo script_start)) deadline_epoch=$DEADLINE"
  echo "remaining_at_script_start_s=$(remaining)"
} | tee "$OUT/BUDGET.txt"

# Watchdog: at the deadline, kill whatever runs (bench client, server) and flag it.
# The main loop then attempts no further cell; label + summaries use the tail reserve.
(
  while [ "$(remaining)" -gt 0 ]; do sleep 5; done
  echo "fired_epoch=$(date +%s)" > "$OUT/BUDGET_HARDCAP_FIRED"
  pkill -TERM -f "sglang.bench_serving" 2>/dev/null
  pkill -TERM -f "sglang.launch_server" 2>/dev/null
  pkill -TERM -f "lambda0_vessl_fake_cell" 2>/dev/null
  sleep 15
  pkill -9 -f "sglang.bench_serving" 2>/dev/null
  pkill -9 -f "sglang.launch_server" 2>/dev/null
  pkill -9 -f "lambda0_vessl_fake_cell" 2>/dev/null
) > /dev/null 2>&1 &
WD_PID=$!

echo "STAGE0 lambda* VESSL port of rev5 model=$MODEL backend=$BACKEND ctx=$CTX mem=$MEM_FRACTION job=$JOBID dryrun=$DRYRUN"

# ------------------------------- prereg sec 2-5: git-ignored decision inputs
# srv_warmup.log + I3a/I3b jsonl of job 907959 are .gitignore'd, so the commit
# bundle does not carry them.  They come from the object volume and must match the
# digests 908623's predicate recorded.  cp -p keeps the 2026-09-13 mtime so
# job_entry does not re-export them.
for rel in srv_warmup.log instrument/I3a_shapeA.jsonl instrument/I3b_shapeB.jsonl; do
  dst="$I3_JOB/$rel"
  if [ ! -f "$dst" ]; then
    src="$STAGE/$(basename "$rel")"            # flat layout (stage_lambda0_inputs.sh)
    [ -f "$src" ] || { echo "ABORT_STAGE decision input missing: $src (run stage_lambda0_inputs.sh --upload first)" \
      | tee -a "$OUT/PREFLIGHT_FAILURES.txt"; exit 3; }
    mkdir -p "$(dirname "$dst")"; cp -p "$src" "$dst"
  fi
done

# ------------------------------------------------- F6 + port: registration integrity
PREREG_MD="$PREREG/PREREG_LAMBDA0_REV5_2026-09-14.md"
PORT_MD="$HERE/PREREG_LAMBDA0_VESSL_2026-10-04.md"
DECISION_PATH=(lambda0_lambda_inf.py lambda0_plan.py lambda0_label.py
               lambda0_analyze.py lambda0_cells.py lambda0_reachability.py
               lambda0_mutation_check.py)
python3 "$HELPER" --verify-digests "$PREREG" "$I3_JOB" "$ENG" | tee "$OUT/DIGEST_CHECK.txt"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "ABORT_F6_DRIFT rev5 decision path / inputs differ from job 908623's bytes" \
  | tee -a "$OUT/PREFLIGHT_FAILURES.txt"; exit 3; }
[ -f "$PORT_MD" ] || { echo "ABORT_F6 port pre-registration missing: $PORT_MD" \
  | tee -a "$OUT/PREFLIGHT_FAILURES.txt"; exit 3; }
{
  echo "# pre-registration in force: $(basename "$PREREG_MD") + ADDENDUM + $(basename "$PORT_MD")"
  sha256sum "$PREREG_MD" "$PREREG/PREREG_LAMBDA0_REV5_ADDENDUM_2026-09-14.md" "$PORT_MD"
  echo "# decision path (${#DECISION_PATH[@]} files) + renderer + rev5 runner (not executed here)"
  for f in "${DECISION_PATH[@]}" lambda0_cellprint.py lambda0.sbatch; do sha256sum "$PREREG/$f"; done
  echo "# VESSL port runner + helper + descriptive estimator"
  sha256sum "$HERE/lambda0_vessl.sh" "$HELPER" "$MIX"
  echo "# every pre-registration revision kept beside it"
  sha256sum "$PREREG"/PREREG_LAMBDA0*.md
  # job_entry replaces engine-port/results with a symlink, so git sees every tracked
  # result file as changed; count tracked changes OUTSIDE results/ only.
  echo "# tree: $(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo NO_GIT)" \
       "uncommitted_outside_results=$(git -C "$ROOT" status --porcelain --untracked-files=no -- . ':!workspace/engine-port/results' 2>/dev/null | grep -c .)"
} | tee "$OUT/REGISTRATION_SHA256.txt"
N_SHA=$(grep -c "^[0-9a-f]\{64\} " "$OUT/REGISTRATION_SHA256.txt")
if [ "${N_SHA:-0}" -lt 10 ]; then
  echo "ABORT_F6 only $N_SHA digests recorded, registered minimum 10" | tee -a "$OUT/PREFLIGHT_FAILURES.txt"; exit 3
fi

# ------------------------------------------------------ substrate record (prereg sec 2-1, 2-2)
{
  echo "nproc=$(nproc 2>/dev/null)"
  echo "cpu_affinity=$(taskset -pc $$ 2>/dev/null | awk -F': ' '{print $2}')"
  echo "cgroup_cpu_max=$(cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo NA)"
  echo "cgroup_mem_max=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo NA)"
  echo "mem_total_kb=$(awk '/MemTotal/{print $2}' /proc/meminfo 2>/dev/null)"
  echo "python=$(python3 -c 'import sys;print(sys.version.split()[0])' 2>/dev/null)"
  echo "torch=$(python3 -c 'import torch;print(torch.__version__, torch.version.cuda)' 2>/dev/null || echo NA)"
  echo "flashinfer=$(python3 -c 'import importlib.metadata as m;print(m.version("flashinfer-python"))' 2>/dev/null || echo NA)"
  echo "sglang_kernel=$(python3 -c 'import importlib.metadata as m;print(m.version("sglang-kernel"))' 2>/dev/null || echo NA)"
  echo "numpy=$(python3 -c 'import numpy;print(numpy.__version__)' 2>/dev/null || echo NA)"
  echo "nvidia_smi=$(nvidia-smi --query-gpu=name,uuid,driver_version,clocks.max.sm,clocks.applications.graphics,power.limit --format=csv,noheader 2>&1 | head -1)"
  echo "hf_refs_main=$(cat "$HF_HOME/hub/models--nvidia--NVIDIA-Nemotron-Nano-9B-v2-Base/refs/main" 2>/dev/null || echo ABSENT)"
  echo "sharegpt_sha256=$(sha256sum "$SGPT_RAW" 2>/dev/null | cut -d' ' -f1 || echo ABSENT)"
  echo "# engine-relevant env (names=values, non-secret prefixes only)"
  env | grep -E '^(SGLANG|FLASHINFER|TRITON|CUDA|NCCL|TORCH|XDG_CACHE)_?[A-Z_]*=' | sort
} > "$OUT/SUBSTRATE_LAM0V.txt" 2>&1
cat "$OUT/SUBSTRATE_LAM0V.txt"
REV_WANT=$(python3 -c "import sys; sys.path.insert(0,'$HERE'); import lambda0_vessl_plan as V; print(V.MODEL_REVISION)")
REV_HAVE=$(cat "$HF_HOME/hub/models--nvidia--NVIDIA-Nemotron-Nano-9B-v2-Base/refs/main" 2>/dev/null || echo ABSENT)
if [ "$REV_HAVE" != "$REV_WANT" ]; then
  if [ "$DRYRUN" = 1 ]; then echo "DRYRUN_NOTE model revision $REV_HAVE (would ABORT_MODEL_REVISION in a measurement)"
  else echo "ABORT_MODEL_REVISION refs/main=$REV_HAVE registered=$REV_WANT" | tee -a "$OUT/PREFLIGHT_FAILURES.txt"; exit 3; fi
fi

SKIP_PRE="${PDMUX_LAM0V_TEST_SKIP_PREFLIGHT:-0}"
# ------------------------------------------------- D12: provenance BEFORE GPU (rev5)
if [ "$SKIP_PRE" != 1 ]; then
  echo "=== env sync (runtime source manifest) ==="
  "$ENG/scripts/bootstrap/sync_engine_tree.sh" "$OUT/runtime_source_manifest.sha256" || exit 2
  N_MANIFEST=$(grep -c . "$OUT/runtime_source_manifest.sha256")
  echo "manifest_entries=$N_MANIFEST"
  grep -q "models/nemotron_h.py" "$OUT/runtime_source_manifest.sha256" || {
    echo "ABORT_D12 manifest does not cover the served model implementation" | tee -a "$OUT/PREFLIGHT_FAILURES.txt"; exit 3; }
  [ "${N_MANIFEST:-0}" -ge 24 ] || {
    echo "ABORT_D12 manifest has $N_MANIFEST entries, registered minimum 24" | tee -a "$OUT/PREFLIGHT_FAILURES.txt"; exit 3; }
  # prereg sec 1-3: the engine tree must be the one 908623 ran (25 entries, by content)
  python3 "$HELPER" --engine-diff "$OUT/runtime_source_manifest.sha256" \
    "$PREREG/lam0_908623/runtime_source_manifest.sha256" | tee "$OUT/ENGINE_TREE_VS_908623.txt"
  if [ "${PIPESTATUS[0]}" -ne 0 ]; then
    if [ "$DRYRUN" = 1 ]; then echo "DRYRUN_NOTE engine tree differs (would ABORT_ENGINE_TREE_DRIFT)"
    else echo "ABORT_ENGINE_TREE_DRIFT engine tree differs from job 908623's" | tee -a "$OUT/PREFLIGHT_FAILURES.txt"; exit 3; fi
  fi
else
  echo "TEST_SKIP_PREFLIGHT: sync/manifest/selftests/mutation NOT run (dry-run only)" | tee "$OUT/TEST_SKIP_PREFLIGHT"
fi

# ---------------------------------------------- the plan (deterministic, rev5 D3/D18)
INSTR="$I3_JOB/instrument"
export LAMBDA0_I3_JOB_DIR="$(dirname "$INSTR")"
if [ "$SKIP_PRE" != 1 ]; then python3 "$PREREG/lambda0_lambda_inf.py" --selftest || exit 2; fi
python3 "$PREREG/lambda0_lambda_inf.py" --recount "$INSTR" | tee "$OUT/RUNNING_REQ_RECOUNT.txt" || exit 2
python3 "$PREREG/lambda0_lambda_inf.py" "$INSTR" | tee "$OUT/LAMBDA_INF_DECISION.txt" || exit 2
python3 -c "
import json,sys
sys.path.insert(0,'$PREREG')
from lambda0_lambda_inf import decide
json.dump(decide('$INSTR'), open('$OUT/LAMBDA_INF_DECISION.json','w'), indent=2, sort_keys=True)
"
LAMBDA0_MODE=$(awk -F= '$1=="LAMBDA0_MODE"{print $2}' "$OUT/LAMBDA_INF_DECISION.txt")
case "$LAMBDA0_MODE" in
  ANCHORED|FALLBACK) ;;
  *) echo "ABORT_D18 predicate produced no mode (got '$LAMBDA0_MODE')" | tee -a "$OUT/PREFLIGHT_FAILURES.txt"; exit 2;;
esac
if [ "$LAMBDA0_MODE" = "ANCHORED" ]; then
  echo "ABORT_D18 unexpected ANCHORED (rev5 registers FALLBACK) (from $INSTR)" | tee -a "$OUT/PREFLIGHT_FAILURES.txt"; exit 2
fi
PLAN_ARGS=(--lambda-inf-a 2.10 --lambda-inf-b 0.675 --fallback)
echo "PLAN=fallback (registered literal ladders; see LAMBDA_INF_DECISION.txt)"

if [ "$SKIP_PRE" != 1 ]; then
  python3 "$PREREG/lambda0_plan.py" --selftest || exit 2
  python3 "$PREREG/lambda0_label.py" --selftest || exit 2
  python3 "$PREREG/lambda0_cells.py" --selftest || exit 2
  python3 "$PREREG/lambda0_reachability.py" --selftest || exit 2
  python3 "$PREREG/lambda0_reachability.py" > "$OUT/REACHABILITY.txt" 2>&1 || exit 2
  python3 "$HELPER" --selftest || exit 2
  # 12 CPUs on the VESSL A100 spec vs 8 on KISTI; the harness caps itself at 8.
  python3 "$PREREG/lambda0_mutation_check.py" > "$OUT/mutation_check.txt" 2>&1
  MUT_RC=$?
  case "$MUT_RC" in
    0) ;;
    1) echo "ABORT_MUTATION_ESCAPE (rc=1: a registered mutation survived its selftest)" \
         | tee -a "$OUT/PREFLIGHT_FAILURES.txt"; tail -5 "$OUT/mutation_check.txt"; exit 5;;
    2) echo "ABORT_MUTATION_HARNESS_CANNOT_RUN (rc=2: MEASUREMENT FAILURE, not an escape -- the harness never scored the mutations)" \
         | tee -a "$OUT/PREFLIGHT_FAILURES.txt"; tail -5 "$OUT/mutation_check.txt"; exit 6;;
    *) echo "ABORT_MUTATION_UNEXPECTED_RC=$MUT_RC (unregistered exit status; treated as a MEASUREMENT FAILURE, not an escape)" \
         | tee -a "$OUT/PREFLIGHT_FAILURES.txt"; tail -5 "$OUT/mutation_check.txt"; exit 6;;
  esac
fi
python3 "$PREREG/lambda0_plan.py" "${PLAN_ARGS[@]}" > "$OUT/PLAN.txt" || exit 2
cat "$OUT/PLAN.txt"
python3 "$PREREG/lambda0_plan.py" "${PLAN_ARGS[@]}" --json > "$OUT/plan.json" || exit 2
# prereg sec 1-1: the plan must be the plan 908623 ran, byte for byte.
cmp -s "$OUT/plan.json" "$PREREG/lam0_908623/plan.json" || {
  echo "ABORT_PLAN_DRIFT plan.json differs from lam0_908623/plan.json" | tee -a "$OUT/PREFLIGHT_FAILURES.txt"; exit 3; }
echo "plan.json == lam0_908623/plan.json (byte-identical)"
cmp -s "$OUT/PLAN.txt" "$PREREG/lam0_908623/PLAN.txt" && echo "PLAN.txt == lam0_908623/PLAN.txt" \
  || echo "NOTE PLAN.txt differs from 908623's (recorded, not a gate)"

SEED1=$(python3 -c "import json;print(json.load(open('$OUT/plan.json'))['seed1'])")
SEED2=$(python3 -c "import json;print(json.load(open('$OUT/plan.json'))['seed2'])")
mapfile -t CELLS < <(python3 "$PREREG/lambda0_cells.py" "$OUT/plan.json")
mapfile -t VCELLS < <(python3 "$HELPER" --variance-rows "$OUT/plan.json")
echo "n_cells=${#CELLS[@]} n_variance=${#VCELLS[@]} seed1=$SEED1 seed2=$SEED2 mode=$LAMBDA0_MODE"
[ "${#CELLS[@]}" -eq 11 ] && [ "${#VCELLS[@]}" -eq 4 ] || { echo "ABORT_CELL_COUNT" \
  | tee -a "$OUT/PREFLIGHT_FAILURES.txt"; exit 2; }
printf '%s\n' "${CELLS[@]}" > "$OUT/CELLS.txt"
printf '%s\n' "${VCELLS[@]}" > "$OUT/variance/CELLS.txt"
python3 "$HELPER" --budget "$OUT/plan.json" > "$OUT/BUDGET_TABLE.json"

EXPECT=""; REPEAT=""
for SPEC in "${CELLS[@]}"; do
  read -r NAME _ <<< "$SPEC"
  case "$NAME" in
    *_s2) REPEAT="${REPEAT:+$REPEAT,}$NAME";;
    *)    EXPECT="${EXPECT:+$EXPECT,}$NAME";;
  esac
done
echo "expect=$EXPECT"
echo "repeat=$REPEAT"

# ------------------------------------------------------------ resume (prereg sec 6-4)
RESUME_FROM="${PDMUX_LAM0V_RESUME_FROM:-}"
if [ -n "$RESUME_FROM" ]; then
  PREV="$DATA/runs/$RESUME_FROM/results/r2_eval/lambda0_vessl/lam0v_$RESUME_FROM"
  [ -d "$PREV" ] || { echo "ABORT_RESUME previous run dir not found: $PREV" | tee -a "$OUT/PREFLIGHT_FAILURES.txt"; exit 2; }
  cmp -s "$PREV/plan.json" "$OUT/plan.json" || { echo "ABORT_RESUME plan.json differs from $RESUME_FROM" \
    | tee -a "$OUT/PREFLIGHT_FAILURES.txt"; exit 2; }
  diff <(grep -E "lambda0_|PREREG_" "$PREV/REGISTRATION_SHA256.txt" | awk '{print $1}') \
       <(grep -E "lambda0_|PREREG_" "$OUT/REGISTRATION_SHA256.txt" | awk '{print $1}') > /dev/null || {
    echo "ABORT_RESUME registration digests differ from $RESUME_FROM" | tee -a "$OUT/PREFLIGHT_FAILURES.txt"; exit 2; }
  PREV_SUB="$DATA/runs/$RESUME_FROM/meta/substrate.json"
  PREV_GPU=$(python3 -c "import json;g=json.load(open('$PREV_SUB'))['gpu'];print(g['uuid'],g['driver_version'])" 2>/dev/null || echo "UNKNOWN UNKNOWN")
  printf 'cell\tfrom_job\tfrom_gpu_uuid\tfrom_driver\n' > "$OUT/RESUMED_CELLS.tsv"
  for sub in "" variance; do
    for done_f in "$PREV/${sub:+$sub/}"cell_*.complete; do
      [ -e "$done_f" ] || continue
      n=$(basename "$done_f" .complete); n="${n#cell_}"
      for f in "$PREV/${sub:+$sub/}"*_"$n".*; do
        cp "$f" "$OUT/${sub:+$sub/}"          # no -p: new mtime => job_entry exports it
      done
      printf '%s\t%s\t%s\n' "$n" "$RESUME_FROM" "$(echo "$PREV_GPU" | tr ' ' '\t')" >> "$OUT/RESUMED_CELLS.tsv"
    done
  done
  echo "RESUMED $(($(wc -l < "$OUT/RESUMED_CELLS.tsv") - 1)) cell(s) from $RESUME_FROM (substrate mix recorded)"
fi

wait_health () {
  for i in $(seq 1 240); do
    budget_fired && return 1
    [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$1/health 2>/dev/null||true)" = 200 ] && return 0
    kill -0 "$2" 2>/dev/null || return 1
    sleep 5
  done
  return 1
}

FAIL_STREAK=0
STOP=""
run_cell () {   # $1 = spec row, $2 = cell dir
  local SPEC="$1" CD="$2"
  read -r NAME SHAPE INLEN OUTLEN RATE NP CSEED KAPPA LHAT LOWSIDE TMEAS <<< "$SPEC"
  [ "$KAPPA" = "-" ] && KAPPA=""
  [ "$LHAT" = "-" ] && LHAT=""
  if [ -e "$CD/cell_${NAME}.complete" ]; then echo "SKIP_DONE cell=$NAME"; return 0; fi
  if [ "$STOP" = boot ]; then
    echo "NOT_ATTEMPTED_AFTER_BOOT_STREAK cell=$NAME" | tee -a "$OUT/BOOT_FAILURES.txt"; return 0
  fi
  if [ "$STOP" = budget ] || budget_fired; then
    echo "NOT_ATTEMPTED_BUDGET cell=$NAME" | tee -a "$OUT/BUDGET_SKIPPED.txt"; return 0
  fi
  # admission: the remaining budget must cover this cell at rev5's worst corner
  local NEED REM
  NEED=$(python3 "$HELPER" --need "$OUT/plan.json" "$NAME") || NEED=999999
  # enforcer test only (DRYRUN): the registered need, time-compressed 1000x
  [ -n "${PDMUX_LAM0V_TEST_FAKE_CELL_S:-}" ] && NEED=$(( (NEED + 999) / 1000 ))
  REM=$(remaining)
  if [ "$REM" -lt "$NEED" ]; then
    echo "NOT_ATTEMPTED_BUDGET cell=$NAME need_s=$NEED remaining_s=$REM" | tee -a "$OUT/BUDGET_SKIPPED.txt"
    STOP=budget; return 0
  fi
  echo "admit cell=$NAME need_s=$NEED remaining_s=$REM" >> "$OUT/BUDGET.txt"

  local PORT=0 try CAND
  for try in $(seq 1 50); do
    CAND=$((39100 + (RANDOM + $$ + try) % 800))
    if ! curl -s -o /dev/null --max-time 1 "http://127.0.0.1:$CAND/health" 2>/dev/null; then
      if ! ss -ltn 2>/dev/null | grep -q ":$CAND "; then PORT=$CAND; break; fi
    fi
  done
  [ "$PORT" = 0 ] && { echo "ABORT_NO_FREE_PORT"; exit 4; }

  local SRVLOG="$CD/srv_${NAME}.log"
  export PDMUX_TELEMETRY_PATH="$CD/tel_${NAME}.jsonl"   # truncated just before the boot
  export PDMUX_RUN_ID="lam0_${NAME}_${JOBID}"
  export PDMUX_WORKLOAD_ID="lam0_${NAME}"
  export PDMUX_R2_POLICY=fixed
  export PDMUX_R2_FIXED_DSM=$FIXED_DSM
  # prereg sec 7: one-shot driver read-out of each stream group's green-context SM
  # count, written at startup (no per-iteration cost; not in rev5).
  export PDMUX_GREEN_READOUT=1 PDMUX_GREEN_READOUT_PATH="$CD/green_${NAME}.json"

  echo ""
  echo "########## CELL $NAME shape=$SHAPE in=$INLEN out=$OUTLEN rate=$RATE np=$NP seed=$CSEED T=${TMEAS}s kappa=${KAPPA:-SAT} lowside=$LOWSIDE port=$PORT remaining_s=$REM ##########"
  local SRV=(python -m sglang.launch_server --model-path "$MODEL" --trust-remote-code --dtype bfloat16
    --attention-backend "$BACKEND"
    --enable-pdmux --pdmux-config-path "$CFG"
    --disable-overlap-schedule --chunked-prefill-size -1
    --disable-radix-cache --mem-fraction-static "$MEM_FRACTION"
    --max-running-requests "$MAX_RUNNING"
    --context-length "$CTX" --random-seed "$SERVER_SEED"
    --host 127.0.0.1 --port "$PORT")
  local WARM=(python -m sglang.bench_serving --backend sglang --model "$MODEL"
    --host 127.0.0.1 --port "$PORT"
    --dataset-name random --dataset-path "$SGPT_RAW" --tokenize-prompt
    --random-input-len "$INLEN" --random-output-len "$OUTLEN" --random-range-ratio 1.0
    --num-prompts 8 --max-concurrency 1 --seed 7)
  local BENCH=(python -m sglang.bench_serving --backend sglang --model "$MODEL"
    --host 127.0.0.1 --port "$PORT"
    --dataset-name random --dataset-path "$SGPT_RAW" --tokenize-prompt
    --random-input-len "$INLEN" --random-output-len "$OUTLEN" --random-range-ratio 1.0
    --num-prompts "$NP" --request-rate "$RATE" --seed "$CSEED"
    --output-details --output-file "$CD/bench_${NAME}.jsonl")

  if [ "$DRYRUN" = 1 ]; then
    { printf 'CELL %s\n  server:' "$NAME"; printf ' %q' "${SRV[@]}"; echo
      printf '  warmup:'; printf ' %q' "${WARM[@]}"; echo
      printf '  bench :'; printf ' %q' "${BENCH[@]}"; echo
      echo "  env   : PDMUX_TELEMETRY_PATH=$PDMUX_TELEMETRY_PATH PDMUX_RUN_ID=$PDMUX_RUN_ID PDMUX_R2_POLICY=fixed PDMUX_R2_FIXED_DSM=$FIXED_DSM PDMUX_GREEN_READOUT=1"
    } >> "$OUT/DRYRUN_COMMANDS.txt"
    if [ -n "${PDMUX_LAM0V_TEST_FAKE_CELL_S:-}" ]; then
      # budget-enforcer test: a fake cell under the SAME timeout/watchdog machinery
      local CAP; CAP=$(remaining); [ "$CAP" -lt 1 ] && CAP=1
      timeout -k 2 "$CAP" bash -c 'exec -a lambda0_vessl_fake_cell sleep "$1"' _ "$PDMUX_LAM0V_TEST_FAKE_CELL_S"
      echo "fake_rc=$? cell=$NAME" | tee -a "$OUT/FAKE_CELLS.txt"
    fi
    return 0
  fi

  : > "$PDMUX_TELEMETRY_PATH"
  setsid "${SRV[@]}" > "$SRVLOG" 2>&1 &
  local PID=$!
  if wait_health "$PORT" "$PID"; then
    echo "boot_ok=1 cell=$NAME"
    FAIL_STREAK=0
    grep -E "max_mamba_cache_size|max_total_num_tokens|available_gpu_mem" "$SRVLOG" \
      | tail -3 | tee "$CD/banner_${NAME}.txt" | sed 's/^/  banner /'
    local CAP
    CAP=$(remaining); [ "$CAP" -lt 1 ] && CAP=1
    timeout -k 30 "$CAP" "${WARM[@]}" > "$CD/warmup_${NAME}.log" 2>&1
    echo "warmup_rc=$?"
    CAP=$(remaining); [ "$CAP" -lt 1 ] && CAP=1
    timeout -k 30 "$CAP" "${BENCH[@]}" > "$CD/bench_${NAME}.log" 2>&1
    local BRC=$?
    echo "bench_rc=$BRC cell=$NAME"
    local KILLED=0
    { [ "$BRC" = 124 ] || [ "$BRC" = 137 ] || budget_fired; } && KILLED=1
    [ "$KILLED" = 1 ] && echo "BENCH_KILLED_BY_BUDGET cell=$NAME rc=$BRC" | tee -a "$OUT/BUDGET_SKIPPED.txt"
    grep -E "Successful requests|Request throughput|Total input tokens|Total generated tokens|Median TTFT|P99 TTFT|Median ITL|Concurrency:" \
      "$CD/bench_${NAME}.log"
    # rev5 runs the analyzer whatever bench_rc is; only a budget kill skips it here
    # (a truncated record must not become a cell).
    if [ "$KILLED" = 0 ]; then
      python3 "$PREREG/lambda0_analyze.py" "$CD/bench_${NAME}.jsonl" \
        --shape "$SHAPE" --label "$NAME" --num-prompts "$NP" --seed "$CSEED" \
        --low-side-candidate "$LOWSIDE" --t-measure-s "$TMEAS" \
        ${KAPPA:+--kappa-pred "$KAPPA"} ${LHAT:+--drain-pred-s "$LHAT"} \
        --out "$CD/cell_${NAME}.json" > "$CD/analyze_${NAME}.log" 2>&1
      ARC=$?
      [ "$ARC" = 0 ] || echo "ANALYZE_FAILED cell=$NAME"
      python3 "$PREREG/lambda0_cellprint.py" "$CD/cell_${NAME}.json" 2>/dev/null || true
    fi
    du -h "$PDMUX_TELEMETRY_PATH" | sed 's/^/  telemetry /'
  else
    echo "BOOT_FAILED cell=$NAME" | tee -a "$OUT/BOOT_FAILURES.txt"
    tail -40 "$SRVLOG" | grep -viE "pynvml|Future" | tee -a "$OUT/BOOT_FAILURES.txt"
    budget_fired || FAIL_STREAK=$((FAIL_STREAK + 1))
  fi
  kill "$PID" 2>/dev/null; sleep 5
  pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null; sleep 10
  # descriptive realized-partition record (never a decision input; GPU idle here)
  if [ -s "$PDMUX_TELEMETRY_PATH" ]; then
    python3 "$MIX" "$PDMUX_TELEMETRY_PATH" --cell "$NAME" --seed "$CSEED" \
      ${BRC:+--bench-rc "$BRC"} --bench "$CD/bench_${NAME}.jsonl" --num-prompts "$NP" \
      --input-len "$INLEN" --output-len "$OUTLEN" --out "$CD/mix_${NAME}.json" \
      > "$CD/mix_${NAME}.log" 2>&1 || echo "MIX_FAILED cell=$NAME (descriptive only)"
  fi
  if [ "${ARC:-1}" = 0 ] && [ -s "$CD/cell_${NAME}.json" ]; then
    date -u +%Y-%m-%dT%H:%M:%SZ > "$CD/cell_${NAME}.complete"
  fi
  if [ "$FAIL_STREAK" -ge 2 ]; then
    echo "ABORT_2_CONSECUTIVE_BOOT_FAILURES" | tee -a "$OUT/BOOT_FAILURES.txt"
    STOP=boot   # rev5 `break`: no further cell (rev5 or variance) is attempted
  fi
}

# rev5 cells first, in rev5 order; then the variance block (sec 5)
for SPEC in "${CELLS[@]}"; do BRC=""; ARC=""; run_cell "$SPEC" "$OUT"; done
for SPEC in "${VCELLS[@]}"; do BRC=""; ARC=""; run_cell "$SPEC" "$OUT/variance"; done
kill "$WD_PID" 2>/dev/null
echo "remaining_at_loop_end_s=$(remaining)" | tee -a "$OUT/BUDGET.txt"

echo ""
echo "=== registered rules (PREREG_LAMBDA0_REV5 sec 5 + ADDENDUM sec A6; rev4 sec 5 SUPERSEDED, rev4 sec 9/10/11 still live) ==="
if [ "$DRYRUN" = 1 ]; then
  python3 "$PREREG/lambda0_label.py" --cells-dir "$OUT" \
    --expect "$EXPECT" --repeat "$REPEAT" --out "$OUT/DRYRUN_LABEL_ALL_MISSING.json" > /dev/null \
    || echo "LABEL_FAILED"
else
  python3 "$PREREG/lambda0_label.py" --cells-dir "$OUT" \
    --expect "$EXPECT" --repeat "$REPEAT" --out "$OUT/LAMBDA0_LABEL.json" \
    || echo "LABEL_FAILED"
fi

echo "=== prereg sec 5/4: variance (descriptive) + side-by-side vs 908623 (compare only) ==="
python3 "$HELPER" --variance-summary "$OUT" > "$OUT/VARIANCE_SUMMARY.json" || echo "VARIANCE_SUMMARY_FAILED"
python3 "$HELPER" --compare-908623 "$OUT" "$PREREG/lam0_908623" > "$OUT/COMPARE_908623.json" || echo "COMPARE_FAILED"
echo "LAMBDA0_VESSL_DONE_${JOBID}"
