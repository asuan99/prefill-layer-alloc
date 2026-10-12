#!/bin/bash
# Shared preflight of m2g_job.sh / m3_job.sh (PREREG_MULTIGPU rev10 §1, §6-2, §6-3, §15-§17).  SOURCED, not executed.
# Caller sets KIND=m2g|m3 before sourcing; PDMUX_MG_JOB=J1|J2|J4|J8 selects the Job (ONE Job = ONE N).
# Provides: JOB, OUT, T0, HARD_CAP_S, EXPECTED_GPUS, fail(), the ABORT_* gates.
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
JOB="${PDMUX_MG_JOB:-}"
OUT="${PDMUX_MG_TEST_OUT_ROOT:-$HERE}/${KIND}_${JOBID}"
mkdir -p "$OUT"
cd "$ROOT" || exit 16
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
[ "$DRYRUN" = 1 ] && echo "DRY RUN -- NOT A MEASUREMENT (no server started)" > "$OUT/DRYRUN"
case "$JOB" in J1|J2|J4|J8) ;; *) echo "ABORT_ENV PDMUX_MG_JOB must be one of J1 J2 J4 J8 (got '$JOB')" | tee -a "$OUT/PREFLIGHT_FAILURES.txt"; echo "ABORT_ENV PDMUX_MG_JOB" > "$OUT/ABORT"; exit 16;; esac
fail () { echo "$1" | tee -a "$OUT/PREFLIGHT_FAILURES.txt"; echo "$1" > "$OUT/ABORT"; exit "$2"; }
dry_or_abort () { if [ "$DRYRUN" = 1 ]; then echo "DRYRUN_NOTE $1" | tee -a "$OUT/DRYRUN_NOTES.txt"; else fail "$1" "$2"; fi; }
K () { python3 "$HERE/mg_const.py" --get "$1"; }

# ---- PDMUX_* knob allowlist (any other registered-looking knob aborts, N3 §8-2 precedent)
ALLOW=" PDMUX_COMMIT PDMUX_JOB_NAME PDMUX_CAMPAIGN PDMUX_TARGET PDMUX_IMAGE PDMUX_SPEC PDMUX_ROOT \
PDMUX_PROJECT_ROOT PDMUX_IO PDMUX_DATA PDMUX_OPT PDMUX_TEST_SKIP_SYNC PDMUX_M2G_PASS_SHA256 PDMUX_M2G_PASS_TABLE \
PDMUX_MG_JOB PDMUX_M2G_PASS_SHA256_J2 PDMUX_M2G_PASS_TABLE_J2 PDMUX_MG_DRYRUN PDMUX_MG_TEST_OUT_ROOT PDMUX_MG_TEST_HARD_CAP_S PDMUX_MG_RUNNER PDMUX_VENV PDMUX_ENERGY_SAMPLER "
for v in $(compgen -e | grep '^PDMUX_' || true); do
  case "$ALLOW" in *" $v "*) ;; *) fail "ABORT_ENV unregistered knob $v is set" 16;; esac
done
[ -n "${PDMUX_ARRAY_SPEC:-}" ] && fail "ABORT_ENV PDMUX_ARRAY_SPEC must be empty (one Job per campaign stage)" 16
if [ "$DRYRUN" != 1 ]; then
  for v in PDMUX_MG_TEST_OUT_ROOT PDMUX_MG_TEST_HARD_CAP_S PDMUX_MG_RUNNER PDMUX_TEST_SKIP_SYNC; do
    [ -n "${!v:-}" ] && fail "ABORT_TEST_KNOB_IN_MEASUREMENT $v" 16
  done
fi

EXPECTED_GPUS=$(python3 "$HERE/mg_const.py" --get-job JOB_EXPECTED_GPUS "$JOB")
SPEC_WANT=$(python3 "$HERE/mg_const.py" --get-job JOB_SPEC "$JOB")
# the Job must be submitted on its own spec (J1 = resourcespec-a100x2, GPU 0 only)
if [ "$DRYRUN" != 1 ] && [ -n "${PDMUX_SPEC:-}" ] && [ "${PDMUX_SPEC}" != "$SPEC_WANT" ]; then
  fail "ABORT_ENV Job $JOB wants spec $SPEC_WANT, submitted on ${PDMUX_SPEC}" 16
fi
# H-2 / P11: the registered config order (J2=J4=J8=[DP,TP]) must be what the code derives; the order is logged for the Job
python3 "$HERE/mg_plan.py" config-order --job "$JOB" > "$OUT/CONFIG_ORDER.txt" 2>&1 || fail "ABORT_CONFIG_ORDER $(tail -c 300 "$OUT/CONFIG_ORDER.txt")" 16

export HF_HOME="${HF_HOME:-$DATA/hf}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
MODEL_DIR="$HF_HOME/hub/models--nvidia--NVIDIA-Nemotron-Nano-9B-v2-Base"

# ---- hard cap (§6-2): wall clock from job_entry's start marker (container start), like N3
HARD_CAP_S=$(K HARD_CAP_S | python3 -c 'import json,sys;print(json.load(sys.stdin)["'"$KIND"'"]["'"$JOB"'"])')
[ -n "${PDMUX_MG_TEST_HARD_CAP_S:-}" ] && HARD_CAP_S="$PDMUX_MG_TEST_HARD_CAP_S"
TIMING="$DATA/runs/$JOBID/meta/timing.txt"
T0=""
[ -f "$TIMING" ] && T0=$(date -u -d "$(awk -F= '$1=="start_utc"{print $2}' "$TIMING")" +%s 2>/dev/null || true)
[ -z "$T0" ] && T0=$(date +%s)
echo "kind=$KIND job=$JOB spec=$SPEC_WANT expected_gpus=$EXPECTED_GPUS hard_cap_s=$HARD_CAP_S t0_epoch=$T0 deadline_epoch=$((T0 + HARD_CAP_S)) dryrun=$DRYRUN" | tee "$OUT/BUDGET.txt"

# ---- registration digests (what this Job ran).  F16: the registered set is a LIST; every mandatory member must exist (ABORT_REG),
# the g5a PASS record (taken on the SEALED tree, F6) is an optional slot written as ABSENT when the file is not there yet.
MANDATORY_REG=(
  "$HERE/PREREG_MULTIGPU_2026-10-12.md"
  "$HERE"/VERDICT_*.md                                                   # every verdict kept beside the PREREG (rules + harness audits)
  "$HERE"/mg_const.py "$HERE"/mg_label.py "$HERE"/mg_plan.py "$HERE"/mg_drive.py "$HERE"/mg_realized_mix.py "$HERE"/mg_substrate.py
  "$HERE"/m2g_job.sh "$HERE"/m3_job.sh "$HERE"/mg_job_common.sh
  "$ENG"/scripts/multigpu/mg_runner.sh "$ENG"/scripts/multigpu/mg_boot.py "$ENG"/scripts/multigpu/mg_loadgen.py
  "$ENG"/scripts/multigpu/mg_proc_sampler.py "$ENG"/scripts/multigpu/m2g_correctness_gate.py
  "$ENG"/scripts/multigpu/m2g_comm_reverse_stress.py "$ENG"/scripts/multigpu/g5a_selector_equiv.py
  "$ENG"/scripts/r2_eval/multigpu_cell_meta.py
  "$ENG"/results/r2_correctness/r2_correctness_client.py                 # G1/G2 prompt source (seed 20260911)
  "$ENG"/benchmarks/pdmux_eval/trace_loadgen.py "$ENG"/benchmarks/pdmux_eval/analyze.py
  "$ENG"/benchmarks/configs/pdmux_agnostic_manual.yml "$ENG"/benchmarks/configs/pdmux_r2.yml
  "$ENG"/triage/pdmux_a100_smoke.yml                                      # G0(iii) auto-grid config + G5(a) GPU boot
  "$ENG"/tests/test_mg_harness.py "$ENG"/tests/test_m2g_gate.py "$ENG"/tests/test_mg_agnostic_manual_equiv.py "$ENG"/tests/test_mg_realized_mix.py
  "$ENG"/scripts/bootstrap/engine_digest.py "$ENG"/scripts/bootstrap/sync_engine_tree.sh   # R2: engine-tree digest/sync tools
  "$ENG"/tests/test_engine_digest.py "$HERE"/ENGINE_DIGEST.txt                             # R2: the sealed ENGINE_DIGEST + its test
)
OPTIONAL_REG=("$HERE/G5A_PASS.json")
{
  echo "# pre-registration in force: PREREG_MULTIGPU_2026-10-12.md (rev10)"
  missing=0
  for f in "${MANDATORY_REG[@]}"; do
    if [ -f "$f" ]; then sha256sum "$f"; else echo "MISSING $f"; missing=$((missing + 1)); fi
  done
  for f in "${OPTIONAL_REG[@]}"; do
    if [ -f "$f" ]; then sha256sum "$f"; else echo "ABSENT_OPTIONAL $f"; fi
  done
  echo "# tree: $(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo NO_GIT) uncommitted_outside_results=$(git -C "$ROOT" status --porcelain --untracked-files=no -- . ':!workspace/engine-port/results' 2>/dev/null | grep -c .)"
  echo "# mandatory_missing=$missing"
} > "$OUT/REGISTRATION_SHA256.txt" 2>&1
grep -q '^# mandatory_missing=0$' "$OUT/REGISTRATION_SHA256.txt" || fail "ABORT_REG a registered file is missing: $(grep '^MISSING' "$OUT/REGISTRATION_SHA256.txt" | head -3 | tr '\n' ' ')" 16
[ "$(grep -c '^[0-9a-f]\{64\} ' "$OUT/REGISTRATION_SHA256.txt")" -ge 32 ] || fail "ABORT_REG fewer than 32 digests recorded" 16

# ---- ABORT_COMM_ENV (§1): JIT all-reduce must be unset; the other comm variables are RECORDED as inherited, then unset
if [ -n "${SGLANG_USE_JIT_ALL_REDUCE:-}" ] && [ "${SGLANG_USE_JIT_ALL_REDUCE}" != 0 ]; then
  fail "ABORT_COMM_ENV SGLANG_USE_JIT_ALL_REDUCE=${SGLANG_USE_JIT_ALL_REDUCE} (registered: unset)" 14
fi
{ for v in $(K COMM_ENV_UNSET); do echo "$v=${!v:-<unset>}"; done; } > "$OUT/COMM_ENV_INHERITED.txt"
for v in $(K COMM_ENV_UNSET) SGLANG_USE_JIT_ALL_REDUCE; do unset "$v"; done

# ---- ABORT_SUBSTRATE (§1, §16-3): A100, cc 8.0, MIG off, distinct UUIDs, uniform driver; GPU count = the Job SPEC's (J1: 2); name EXACTLY
#      "NVIDIA A100-SXM4-80GB"; power limit 400 W; Job >= 2 GPUs: NV# on every pair (mg_substrate.py).  Any miss -> ABORT_SUBSTRATE.
HELPER="$ENG/scripts/vessl/substrate_probe_helper.py"
mkdir -p "$OUT/b0"
nvidia-smi --query-gpu=name,uuid,driver_version,compute_cap,memory.total,clocks.max.sm,power.limit,mig.mode.current,compute_mode \
  --format=csv,noheader > "$OUT/b0/gpu.csv" 2>&1
nvidia-smi topo -m > "$OUT/b0/topo_m.txt" 2>&1
python3 "$HELPER" b0-python --out "$OUT/b0/b0_python.json" > "$OUT/b0/b0_python.log" 2>&1
python3 "$HELPER" score-b0 --gpu-csv "$OUT/b0/gpu.csv" --py-json "$OUT/b0/b0_python.json" \
  --status "$OUT/b0/B0.json" --expected-gpus "$EXPECTED_GPUS" | tee "$OUT/b0/score.txt"
grep -q '^PASS' "$OUT/b0/score.txt" || dry_or_abort "ABORT_SUBSTRATE score-b0 --expected-gpus $EXPECTED_GPUS: $(head -c 300 "$OUT/b0/score.txt")" 11
python3 "$HERE/mg_substrate.py" check --job "$JOB" --gpu-csv "$OUT/b0/gpu.csv" --topo "$OUT/b0/topo_m.txt" --out "$OUT/b0/MG_SUBSTRATE.json" | tee "$OUT/b0/mg_substrate.txt"
grep -q '^PASS' "$OUT/b0/mg_substrate.txt" || dry_or_abort "ABORT_SUBSTRATE mg_substrate (exact name / 400 W / NV#): $(head -c 300 "$OUT/b0/mg_substrate.txt")" 11

# ---- F5: the contamination check (§3 G5(c), pynvml) must be able to run, else every cell would carry CONTAM_UNVERIFIED -> ABORT_SUBSTRATE
python3 - "$EXPECTED_GPUS" > "$OUT/b0/nvml_probe.txt" 2>&1 <<'PY'
import sys
import pynvml
pynvml.nvmlInit()
n = pynvml.nvmlDeviceGetCount()
for i in range(n):
    h = pynvml.nvmlDeviceGetHandleByIndex(i)
    pynvml.nvmlDeviceGetMemoryInfo(h)
    pynvml.nvmlDeviceGetComputeRunningProcesses(h)
assert n == int(sys.argv[1]), f"nvml sees {n} GPUs != {sys.argv[1]}"
print("NVML_OK", n)
PY
grep -q '^NVML_OK' "$OUT/b0/nvml_probe.txt" || dry_or_abort "ABORT_SUBSTRATE NVML_UNAVAILABLE (contamination check cannot run -> CONTAM_UNVERIFIED): $(tail -c 200 "$OUT/b0/nvml_probe.txt" | tr '\n' ' ')" 11
# ---- F1: M2g runs G3 through a distributed launcher -- torchrun or `python -m torch.distributed.run`
if [ "$KIND" = m2g ]; then
  if command -v torchrun > /dev/null 2>&1; then echo "launcher=torchrun" > "$OUT/b0/launcher.txt"
  elif python3 -m torch.distributed.run --help > /dev/null 2>&1; then echo "launcher=python -m torch.distributed.run" > "$OUT/b0/launcher.txt"
  else dry_or_abort "ABORT_SUBSTRATE LAUNCHER_ABSENT (neither torchrun nor python -m torch.distributed.run; G3 cannot run)" 11; fi
fi

# ---- ABORT_MODEL_REVISION (§1)
[ "$(cat "$MODEL_DIR/refs/main" 2>/dev/null || echo ABSENT)" = "$(K MODEL_REVISION)" ] \
  || dry_or_abort "ABORT_MODEL_REVISION $MODEL_DIR/refs/main != $(K MODEL_REVISION)" 12

# ---- ABORT_ENGINE_DIGEST (§6-3): the installed engine tree (sync manifest written by job_entry/install_runtime) must equal the
# digest sealed with the harness.  ENGINE_DIGEST.txt holds one token; "UNSEALED" (the committed placeholder) never passes a real run.
MANIFEST="$DATA/runs/$JOBID/meta/runtime_source_manifest.sha256"
# F7(b): path-free digest (pathless-v1 = sha256 of the 27 sorted entry hashes; scripts/bootstrap/engine_digest.py). The raw
# manifest file sha is not used because its rows carry absolute install paths.
SEALED=$(tr -d ' \n' < "$HERE/ENGINE_DIGEST.txt" 2>/dev/null || echo ABSENT)
python3 "$ENG/scripts/bootstrap/engine_digest.py" check --manifest "$MANIFEST" --sealed-file "$HERE/ENGINE_DIGEST.txt" \
  --out "$OUT/ENGINE_DIGEST_CHECK.json" > "$OUT/ENGINE_DIGEST_CHECK.txt" 2>&1
DRC=$?
echo "sealed=$SEALED check_rc=$DRC" | tee -a "$OUT/ENGINE_DIGEST_CHECK.txt"
[ "$DRC" -eq 0 ] || dry_or_abort "ABORT_ENGINE_DIGEST sealed=$SEALED check_rc=$DRC (see ENGINE_DIGEST_CHECK.txt)" 13

# ---- CPU self-tests of the label/plan code (the mutation checks run inside the Job too)
python3 "$HERE/mg_label.py" --selftest > "$OUT/SELFTEST_label.txt" 2>&1 || { tail -5 "$OUT/SELFTEST_label.txt"; fail "ABORT_SELFTEST mg_label" 16; }
python3 "$HERE/mg_plan.py" --selftest > "$OUT/SELFTEST_plan.txt" 2>&1 || fail "ABORT_SELFTEST mg_plan" 16
python3 "$HERE/mg_substrate.py" --selftest > "$OUT/SELFTEST_substrate.txt" 2>&1 || fail "ABORT_SELFTEST mg_substrate" 16
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
