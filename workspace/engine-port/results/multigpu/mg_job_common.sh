#!/bin/bash
# Shared preflight of m2g_job.sh / m3_job.sh (PREREG_MULTIGPU rev6 §1, §6-2, §6-3).  SOURCED, not executed.
# Caller sets KIND=m2g|m3 before sourcing.  Provides: OUT, T0, HARD_CAP_S, fail(), the ABORT_* gates.
#
# ABORT exit codes (distinct from job_entry.sh's 2/3/4):  11 ABORT_SUBSTRATE · 12 ABORT_MODEL_REVISION ·
#   13 ABORT_ENGINE_DIGEST · 14 ABORT_COMM_ENV · 15 ABORT_M2G_NOT_PASSED · 16 other pre-flight abort
# PDMUX_MG_DRYRUN=1  CPU-only rehearsal: hardware/registry aborts become DRYRUN_NOTE lines, no server is started.
set -uo pipefail
: "${KIND:?}"
ROOT="${PDMUX_PROJECT_ROOT:?PDMUX_PROJECT_ROOT must be set (job_entry.sh exports it)}"
ENG="$ROOT/workspace/engine-port"
HERE="$ENG/results/multigpu"
DATA="${PDMUX_DATA:-/data}"
DRYRUN="${PDMUX_MG_DRYRUN:-0}"
JOBID="${PDMUX_JOB_NAME:-local$$}"
OUT="${PDMUX_MG_TEST_OUT_ROOT:-$HERE}/${KIND}_${JOBID}"
mkdir -p "$OUT"
cd "$ROOT" || exit 16
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
[ "$DRYRUN" = 1 ] && echo "DRY RUN -- NOT A MEASUREMENT (no server started)" > "$OUT/DRYRUN"
fail () { echo "$1" | tee -a "$OUT/PREFLIGHT_FAILURES.txt"; echo "$1" > "$OUT/ABORT"; exit "$2"; }
dry_or_abort () { if [ "$DRYRUN" = 1 ]; then echo "DRYRUN_NOTE $1" | tee -a "$OUT/DRYRUN_NOTES.txt"; else fail "$1" "$2"; fi; }
K () { python3 "$HERE/mg_const.py" --get "$1"; }

# ---- PDMUX_* knob allowlist (any other registered-looking knob aborts, N3 §8-2 precedent)
ALLOW=" PDMUX_COMMIT PDMUX_JOB_NAME PDMUX_CAMPAIGN PDMUX_TARGET PDMUX_IMAGE PDMUX_SPEC PDMUX_ROOT \
PDMUX_PROJECT_ROOT PDMUX_IO PDMUX_DATA PDMUX_OPT PDMUX_TEST_SKIP_SYNC PDMUX_M2G_PASS_SHA256 PDMUX_M2G_PASS_TABLE \
PDMUX_MG_DRYRUN PDMUX_MG_TEST_OUT_ROOT PDMUX_MG_TEST_HARD_CAP_S PDMUX_MG_RUNNER PDMUX_VENV PDMUX_ENERGY_SAMPLER "
for v in $(compgen -e | grep '^PDMUX_' || true); do
  case "$ALLOW" in *" $v "*) ;; *) fail "ABORT_ENV unregistered knob $v is set" 16;; esac
done
[ -n "${PDMUX_ARRAY_SPEC:-}" ] && fail "ABORT_ENV PDMUX_ARRAY_SPEC must be empty (one Job per campaign stage)" 16
if [ "$DRYRUN" != 1 ]; then
  for v in PDMUX_MG_TEST_OUT_ROOT PDMUX_MG_TEST_HARD_CAP_S PDMUX_MG_RUNNER PDMUX_TEST_SKIP_SYNC; do
    [ -n "${!v:-}" ] && fail "ABORT_TEST_KNOB_IN_MEASUREMENT $v" 16
  done
fi

export HF_HOME="${HF_HOME:-$DATA/hf}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
MODEL_DIR="$HF_HOME/hub/models--nvidia--NVIDIA-Nemotron-Nano-9B-v2-Base"

# ---- hard cap (§6-2): wall clock from job_entry's start marker (container start), like N3
HARD_CAP_S=$(K HARD_CAP_S | python3 -c 'import json,sys;print(json.load(sys.stdin)["'"$KIND"'"])')
[ -n "${PDMUX_MG_TEST_HARD_CAP_S:-}" ] && HARD_CAP_S="$PDMUX_MG_TEST_HARD_CAP_S"
TIMING="$DATA/runs/$JOBID/meta/timing.txt"
T0=""
[ -f "$TIMING" ] && T0=$(date -u -d "$(awk -F= '$1=="start_utc"{print $2}' "$TIMING")" +%s 2>/dev/null || true)
[ -z "$T0" ] && T0=$(date +%s)
echo "kind=$KIND hard_cap_s=$HARD_CAP_S t0_epoch=$T0 deadline_epoch=$((T0 + HARD_CAP_S)) dryrun=$DRYRUN" | tee "$OUT/BUDGET.txt"

# ---- registration digests (what this Job ran)
{
  echo "# pre-registration in force: PREREG_MULTIGPU_2026-10-12.md (rev6)"
  sha256sum "$HERE/PREREG_MULTIGPU_2026-10-12.md"
  echo "# verdicts kept beside it"
  sha256sum "$HERE"/VERDICT_prereg_rules_*.md
  echo "# harness"
  sha256sum "$HERE"/mg_const.py "$HERE"/mg_label.py "$HERE"/mg_plan.py "$HERE"/mg_drive.py "$HERE"/mg_realized_mix.py \
            "$HERE"/m2g_job.sh "$HERE"/m3_job.sh "$HERE"/mg_job_common.sh \
            "$ENG"/scripts/multigpu/mg_runner.sh "$ENG"/scripts/multigpu/mg_boot.py "$ENG"/scripts/multigpu/mg_loadgen.py \
            "$ENG"/scripts/multigpu/mg_proc_sampler.py "$ENG"/scripts/multigpu/m2g_correctness_gate.py \
            "$ENG"/scripts/multigpu/m2g_comm_reverse_stress.py "$ENG"/scripts/multigpu/g5a_selector_equiv.py "$ENG"/scripts/r2_eval/multigpu_cell_meta.py \
            "$ENG"/benchmarks/pdmux_eval/trace_loadgen.py "$ENG"/benchmarks/configs/pdmux_agnostic_manual.yml \
            "$ENG"/benchmarks/configs/pdmux_r2.yml
  echo "# tree: $(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo NO_GIT) uncommitted_outside_results=$(git -C "$ROOT" status --porcelain --untracked-files=no -- . ':!workspace/engine-port/results' 2>/dev/null | grep -c .)"
} > "$OUT/REGISTRATION_SHA256.txt" 2>&1
[ "$(grep -c '^[0-9a-f]\{64\} ' "$OUT/REGISTRATION_SHA256.txt")" -ge 20 ] || fail "ABORT_REG fewer than 20 digests recorded (a registered file is missing)" 16

# ---- ABORT_COMM_ENV (§1): JIT all-reduce must be unset; the other comm variables are RECORDED as inherited, then unset
if [ -n "${SGLANG_USE_JIT_ALL_REDUCE:-}" ] && [ "${SGLANG_USE_JIT_ALL_REDUCE}" != 0 ]; then
  fail "ABORT_COMM_ENV SGLANG_USE_JIT_ALL_REDUCE=${SGLANG_USE_JIT_ALL_REDUCE} (registered: unset)" 14
fi
{ for v in $(K COMM_ENV_UNSET); do echo "$v=${!v:-<unset>}"; done; } > "$OUT/COMM_ENV_INHERITED.txt"
for v in $(K COMM_ENV_UNSET) SGLANG_USE_JIT_ALL_REDUCE; do unset "$v"; done

# ---- ABORT_SUBSTRATE (§1): A100, cc 8.0, MIG off, 8 distinct UUIDs, uniform driver
HELPER="$ENG/scripts/vessl/substrate_probe_helper.py"
mkdir -p "$OUT/b0"
nvidia-smi --query-gpu=name,uuid,driver_version,compute_cap,memory.total,clocks.max.sm,power.limit,mig.mode.current,compute_mode \
  --format=csv,noheader > "$OUT/b0/gpu.csv" 2>&1
nvidia-smi topo -m > "$OUT/b0/topo_m.txt" 2>&1
python3 "$HELPER" b0-python --out "$OUT/b0/b0_python.json" > "$OUT/b0/b0_python.log" 2>&1
python3 "$HELPER" score-b0 --gpu-csv "$OUT/b0/gpu.csv" --py-json "$OUT/b0/b0_python.json" \
  --status "$OUT/b0/B0.json" --expected-gpus "$(K EXPECTED_GPUS)" | tee "$OUT/b0/score.txt"
grep -q '^PASS' "$OUT/b0/score.txt" || dry_or_abort "ABORT_SUBSTRATE score-b0 --expected-gpus $(K EXPECTED_GPUS): $(head -c 300 "$OUT/b0/score.txt")" 11

# ---- ABORT_MODEL_REVISION (§1)
[ "$(cat "$MODEL_DIR/refs/main" 2>/dev/null || echo ABSENT)" = "$(K MODEL_REVISION)" ] \
  || dry_or_abort "ABORT_MODEL_REVISION $MODEL_DIR/refs/main != $(K MODEL_REVISION)" 12

# ---- ABORT_ENGINE_DIGEST (§6-3): the installed engine tree (sync manifest written by job_entry/install_runtime) must equal the
# digest sealed with the harness.  ENGINE_DIGEST.txt holds one token; "UNSEALED" (the committed placeholder) never passes a real run.
MANIFEST="$DATA/runs/$JOBID/meta/runtime_source_manifest.sha256"
SEALED=$(tr -d ' \n' < "$HERE/ENGINE_DIGEST.txt" 2>/dev/null || echo ABSENT)
ACTUAL=$(sha256sum "$MANIFEST" 2>/dev/null | cut -d' ' -f1)
echo "sealed=$SEALED actual=${ACTUAL:-NO_MANIFEST}" | tee "$OUT/ENGINE_DIGEST_CHECK.txt"
{ [ -n "$ACTUAL" ] && [ "$SEALED" = "$ACTUAL" ]; } || dry_or_abort "ABORT_ENGINE_DIGEST sealed=$SEALED actual=${ACTUAL:-NO_MANIFEST}" 13

# ---- CPU self-tests of the label/plan code (the mutation checks run inside the Job too)
python3 "$HERE/mg_label.py" --selftest > "$OUT/SELFTEST_label.txt" 2>&1 || { tail -5 "$OUT/SELFTEST_label.txt"; fail "ABORT_SELFTEST mg_label" 16; }
python3 "$HERE/mg_plan.py" --selftest > "$OUT/SELFTEST_plan.txt" 2>&1 || fail "ABORT_SELFTEST mg_plan" 16
python3 "$ENG/scripts/multigpu/mg_proc_sampler.py" --help > /dev/null 2>&1 || fail "ABORT_SELFTEST sampler" 16

# ---- substrate record (CPU side, lesson 268) + client fd limit (H6)
{
  echo "nproc=$(nproc 2>/dev/null) cpu_affinity=$(taskset -pc $$ 2>/dev/null | awk -F': ' '{print $2}')"
  echo "cgroup_cpu_max=$(cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo NA)"
  echo "cgroup_mem_max=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo NA)"
  echo "mem_total_kb=$(awk '/MemTotal/{print $2}' /proc/meminfo 2>/dev/null)"
  echo "numa_nodes=$(ls -d /sys/devices/system/node/node* 2>/dev/null | wc -l)"
  echo "ulimit_n_soft=$(ulimit -S -n) ulimit_n_hard=$(ulimit -H -n)"
  echo "nsys=$(command -v nsys || echo ABSENT)"
  echo "torch=$(python3 -c 'import torch;print(torch.__version__, torch.version.cuda)' 2>/dev/null || echo NA)"
  echo "perf_counter_clock=$(python3 -c 'import time;print(time.get_clock_info("perf_counter").implementation)' 2>/dev/null)"
  echo "monotonic_clock=$(python3 -c 'import time;print(time.get_clock_info("monotonic").implementation)' 2>/dev/null)"
} > "$OUT/SUBSTRATE_CPU.txt" 2>&1
grep -q "perf_counter_clock=clock_gettime(CLOCK_MONOTONIC)" "$OUT/SUBSTRATE_CPU.txt" \
  && grep -q "monotonic_clock=clock_gettime(CLOCK_MONOTONIC)" "$OUT/SUBSTRATE_CPU.txt" \
  || dry_or_abort "ABORT_CLOCK perf_counter/monotonic is not CLOCK_MONOTONIC (loadgen bridge to the /proc sampler and telemetry)" 16
echo "preflight ok kind=$KIND out=$OUT"
