#!/bin/bash
# N2(3-a) Job entry (Job-PROFILE / Job-EA) -- PREREG_N2_3A_2026-10-07.md rev18 (rule seal digest 56473d6f…, 2026-10-13; rev17 5e29990 fc9e6404…;
# first sealed rev16 fcc0b9e). HARNESS-SEAL file. Two-stage harness seal (rev17 R17-2 / H17-6): a Job-PROFILE needs the
# `scope PROFILE` section of REGISTRATION_SHA256.txt, a Job-EA needs BOTH the PROFILE and the EA section. The seal check
# runs right after ABORT_REG_SOURCE with `--scope $JOB`; a Job-EA without both sections is refused in EVERY mode
# (ABORT_HARNESS_UNSEALED_EA, dry-run and CPU fake included); a Job-PROFILE without its section is refused outside
# dry-run / test mode (noted inside them). The Job-EA runner path (E-alpha cell client, rho assembly H15-3) exists in
# this tree but is NOT sealed.
#
# usage (inside a VESSL Job via scripts/vessl/launch.sh; never by hand on a GPU):
#   launch.sh --campaign n2_static_grid --env PDMUX_N2_JOB=PROFILE \
#     --target workspace/engine-port/results/n2_static_grid/n2_capacity.sh
# local CPU only:
#   PDMUX_N2_DRYRUN=1   no server; prints the registered boots/cells/commands (DRYRUN_COMMANDS.txt)
#   PDMUX_N2_TEST_FAKE=1 (+ PDMUX_N2_TEST_UNITS / _CELLS / _OUT_ROOT / _SKIP_PREFLIGHT / _CAP_S / _WALL_SCALE)
#                        the whole Job path on the CPU fake engine -- NOT A MEASUREMENT; refused inside a VESSL Job.
#
# FLOW: knob allowlist -> ABORT_REG_SOURCE (N2_REGISTERED.json bytes == rule-seal value; 7 rule files; rule-seal digest)
#   -> harness seal (REGISTRATION_SHA256.txt "## HARNESS SEAL", --scope $JOB; PROFILE: absent = ABORT_HARNESS_UNSEALED
#   outside dry-run/test; EA: PROFILE + EA sections required in every mode, ABORT_HARNESS_UNSEALED_EA)
#   -> external digests (config / analyze.py / profile.py) + config parser check -> substrate record (B0, model
#   revision, clock, sampler state §3-3) -> engine tree manifest + thread-local role patch -> preflight tests ->
#   cell section (hard cap = N2_REGISTERED budget <JOB>_b cap, from the section start; watchdog SIGTERM ->
#   runner_stop; admission STOP rule inside n2_job.py) -> analysis (tail, unbounded) -> N2_DONE line.
# Exit codes: 2 env/knob, 3 registration/seal/plan/substrate/cg-bs/cg-canary/ea-canary, 4 preflight tests, 5 port, 6 sticky/green, 8 P3,
#   9 correctness.  job_entry.sh exports results/ (files newer than its start marker) + SHA256SUMS + DONE.
set -uo pipefail

ROOT="${PDMUX_PROJECT_ROOT:?PDMUX_PROJECT_ROOT must be set (job_entry.sh exports it)}"
ENG="$ROOT/workspace/engine-port"
HERE="$ENG/results/n2_static_grid"
PLAN="$HERE/n2_plan.py"
DATA="${PDMUX_DATA:-/data}"
JOB="${PDMUX_N2_JOB:-}"
DRYRUN="${PDMUX_N2_DRYRUN:-0}"
FAKE="${PDMUX_N2_TEST_FAKE:-0}"
JOBID="${PDMUX_JOB_NAME:-local$$}"
case "$JOB" in
  PROFILE|EA) ;;
  *) echo "ABORT_ENV PDMUX_N2_JOB must be PROFILE or EA (got '$JOB')"; exit 2 ;;
esac
OUT="${PDMUX_N2_TEST_OUT_ROOT:-$HERE}/n2p_${JOBID}"
mkdir -p "$OUT"
cd "$ROOT" || exit 2
export PYTHONDONTWRITEBYTECODE=1
TESTMODE=0; { [ "$DRYRUN" = 1 ] || [ "$FAKE" = 1 ]; } && TESTMODE=1
[ "$DRYRUN" = 1 ] && echo "DRY RUN -- NOT A MEASUREMENT (no server started)" > "$OUT/DRYRUN"
[ "$FAKE" = 1 ] && echo "CPU FAKE ENGINE -- NOT A MEASUREMENT" > "$OUT/TEST_FAKE_ENGINE"
WD_PID=""; RUN_PID=""
cleanup () { [ -n "$WD_PID" ] && kill "$WD_PID" 2>/dev/null; rm -rf "$HERE/__pycache__" 2>/dev/null; }
trap cleanup EXIT
fail () { echo "$1" | tee -a "$OUT/PREFLIGHT_FAILURES.txt"; exit "$2"; }
note_or_abort () { if [ "$TESTMODE" = 1 ]; then echo "TESTMODE_NOTE $1" | tee -a "$OUT/TESTMODE_NOTES.txt"; else fail "$1" "$2"; fi; }

# ------------------------------------------------------------------ PDMUX_* knob allowlist (§3-3, N3 §8-2 pattern)
# Per-boot knobs (sticky / FIXED / PHASE_EVENTS / policy / telemetry / green) are set by n2_job.py for every boot and
# every other PDMUX_* is removed from the server environment (n2_plan.boot_env); an inherited value is only recorded.
ALLOW=" PDMUX_COMMIT PDMUX_JOB_NAME PDMUX_CAMPAIGN PDMUX_TARGET PDMUX_IMAGE PDMUX_SPEC PDMUX_ROOT PDMUX_PROJECT_ROOT \
PDMUX_IO PDMUX_DATA PDMUX_OPT PDMUX_ARRAY_SPEC PDMUX_TEST_SKIP_SYNC PDMUX_ENERGY_SAMPLER PDMUX_N2_JOB PDMUX_N2_DRYRUN \
PDMUX_STICKY_PARTITION PDMUX_R2_FIXED_DSM PDMUX_PHASE_EVENTS PDMUX_R2_POLICY PDMUX_TELEMETRY_PATH PDMUX_RUN_ID \
PDMUX_WORKLOAD_ID PDMUX_GREEN_READOUT PDMUX_GREEN_READOUT_PATH \
PDMUX_N2_TEST_FAKE PDMUX_N2_TEST_UNITS PDMUX_N2_TEST_CELLS PDMUX_N2_TEST_OUT_ROOT PDMUX_N2_TEST_SKIP_PREFLIGHT \
PDMUX_N2_TEST_CAP_S PDMUX_N2_TEST_WALL_SCALE "
for v in $(compgen -e | grep '^PDMUX_' || true); do
  case "$ALLOW" in *" $v "*) ;; *) fail "ABORT_ENV unregistered knob $v is set" 2;; esac
done
[ -n "${PDMUX_ARRAY_SPEC:-}" ] && fail "ABORT_ENV PDMUX_ARRAY_SPEC must be empty (one Job per measurement)" 2
for v in PDMUX_N2_TEST_FAKE PDMUX_N2_TEST_UNITS PDMUX_N2_TEST_CELLS PDMUX_N2_TEST_OUT_ROOT PDMUX_N2_TEST_SKIP_PREFLIGHT \
         PDMUX_N2_TEST_CAP_S PDMUX_N2_TEST_WALL_SCALE PDMUX_N2_DRYRUN; do
  if [ -n "${!v:-}" ] && [ -n "${PDMUX_COMMIT:-}" ]; then fail "ABORT_TEST_KNOB_IN_MEASUREMENT $v inside a VESSL Job" 2; fi
done
{ for v in PDMUX_STICKY_PARTITION PDMUX_R2_FIXED_DSM PDMUX_PHASE_EVENTS PDMUX_R2_POLICY PDMUX_ENERGY_SAMPLER; do
    echo "inherited_$v=${!v:-<unset>}"; done; } > "$OUT/KNOBS.txt"
export HF_HOME="${HF_HOME:-$DATA/hf}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
unset SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN PDMUX_TRUE_DUAL_WORKER PDMUX_DUAL_WORKER PDMUX_SLO_SCHED \
      PDMUX_SLO_MODE PDMUX_LA_COORD PDMUX_FIXED_DECODE_SM_FILE PDMUX_TRACE_FORCE_PREFILL

# ------------------------------------------------------------------ ABORT_REG_SOURCE + seals (§15, §18-2)
python3 "$PLAN" --verify-registration | tee "$OUT/REG_CHECK.txt"
[ "${PIPESTATUS[0]}" -eq 0 ] || fail "ABORT_REG_SOURCE (see REG_CHECK.txt)" 3
python3 "$PLAN" --verify-harness-seal --scope "$JOB" | tee "$OUT/HARNESS_SEAL_CHECK.txt"
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
  if [ "$JOB" = EA ]; then           # rev17 H17-6: Job-EA needs the PROFILE + EA harness-seal sections (every mode)
    fail "ABORT_HARNESS_UNSEALED_EA $(cat "$OUT/HARNESS_SEAL_CHECK.txt")" 3
  fi
  note_or_abort "ABORT_HARNESS_UNSEALED $(cat "$OUT/HARNESS_SEAL_CHECK.txt")" 3
fi
python3 "$PLAN" --external-digests | tee "$OUT/EXTERNAL_DIGESTS.txt"
[ "${PIPESTATUS[0]}" -eq 0 ] || fail "ABORT_REG_DRIFT config / analyze.py / profile.py bytes differ" 3
CFG="$ENG/$(python3 "$PLAN" --const CONFIG_REL)"
python3 "$PLAN" --check-config "$CFG" | tee "$OUT/CONFIG_CHECK.txt"
[ "${PIPESTATUS[0]}" -eq 0 ] || fail "ABORT_REG_DRIFT config parser check (CONFIG_CHECK.txt)" 3
python3 "$PLAN" --units "$JOB" > "$OUT/UNITS.txt" || fail "ABORT_TRACE (UNITS.txt)" 3
python3 "$PLAN" --budget "$JOB" > "$OUT/BUDGET_TABLE.json" || fail "ABORT_REG_SOURCE budget" 3
{
  echo "# rule seal files + harness files + external (this run)"
  (cd "$HERE" && sha256sum PREREG_N2_3A_2026-10-07.md n2_feasibility.py n2_probe_reduce.py N2_REGISTERED.json \
     test_n2_feasibility.py test_n2_probe_reduce.py n2_mutants.py n2_marks.py n2_loadgen.py n2_probe_run.py \
     fake_n2_engine.py test_n2_harness.py n2_harness_mutants.py n2_capacity.sh n2_plan.py n2_job.py n2_window.py \
     n2_signature.py n2_realized_mix.py n2_analyze.py n2_reach_tests.py test_n2_job.py n2_job_mutants.py \
     REGISTRATION_SHA256.txt)
  sha256sum "$CFG" "$ENG/benchmarks/pdmux_eval/analyze.py" "$ENG/src/multiplex/profile.py"
  echo "# tree: $(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo NO_GIT)"
} > "$OUT/REGISTRATION_RUN_SHA256.txt" 2>&1

# ------------------------------------------------------------------ substrate (§3-3) + B0
MODEL_DIR="$HF_HOME/hub/models--nvidia--NVIDIA-Nemotron-Nano-9B-v2-Base"
META="$DATA/runs/$JOBID/meta"
{
  echo "nproc=$(nproc 2>/dev/null)"
  echo "cgroup_cpu_max=$(cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo NA)"
  echo "perf_counter_clock=$(python3 -c 'import time;print(time.get_clock_info("perf_counter").implementation)')"
  echo "monotonic_clock=$(python3 -c 'import time;print(time.get_clock_info("monotonic").implementation)')"
  echo "tz=$(date +%Z) gmtoff=$(python3 -c 'import time;print(time.localtime().tm_gmtoff)')"
  echo "nvidia_smi=$(nvidia-smi --query-gpu=name,uuid,driver_version,clocks.max.sm,power.limit --format=csv,noheader 2>&1 | head -1)"
  echo "hf_refs_main=$(cat "$MODEL_DIR/refs/main" 2>/dev/null || echo ABSENT)"
} > "$OUT/SUBSTRATE_N2.txt" 2>&1
python3 - "$OUT/substrate_n2.json" "$META" <<'PY'
import json, os, sys
meta = sys.argv[2]
outs = {f: os.path.exists(os.path.join(meta, f)) for f in ("energy.csv", "power_samples.csv", "sm_clk_samples.csv")}
val = os.environ.get("PDMUX_ENERGY_SAMPLER")
json.dump({"energy_sampler": {"state": "off" if val == "0" else "on (job_entry.sh default)" if os.path.isdir(meta)
                              else "off (no job_entry.sh)", "PDMUX_ENERGY_SAMPLER": val, "outputs_present": outs,
                              "note": "§3-3 / N2C-61: no same-Job on/off contrast -- sampler effect not claimed"}},
          open(sys.argv[1], "w"), indent=1, sort_keys=True)
PY
B0_CSV=$(nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version,mig.mode.current \
         --format=csv,noheader,nounits 2>/dev/null | head -1)
B0_SM=$(python3 -c 'import torch;print(torch.cuda.get_device_properties(0).multi_processor_count)' 2>/dev/null || echo NA)
python3 "$PLAN" --check-substrate "${B0_CSV:-NONE}" "$B0_SM" | tee "$OUT/SUBSTRATE_CHECK.txt"
[ "${PIPESTATUS[0]}" -eq 0 ] || note_or_abort "ABORT_SUBSTRATE ($B0_CSV, sm=$B0_SM)" 3
[ "$(cat "$MODEL_DIR/refs/main" 2>/dev/null || echo ABSENT)" = "$(python3 "$PLAN" --const MODEL_REVISION)" ] \
  || note_or_abort "ABORT_MODEL_REVISION refs/main != registered" 3
grep -q "perf_counter_clock=clock_gettime(CLOCK_MONOTONIC)" "$OUT/SUBSTRATE_N2.txt" \
  && grep -q "monotonic_clock=clock_gettime(CLOCK_MONOTONIC)" "$OUT/SUBSTRATE_N2.txt" \
  || fail "ABORT_CLOCK perf_counter / monotonic are not CLOCK_MONOTONIC" 3

# ------------------------------------------------------------------ engine tree + preflight tests
if [ "${PDMUX_N2_TEST_SKIP_PREFLIGHT:-0}" != 1 ]; then
  MAN="$META/runtime_source_manifest.sha256"          # written by job_entry.sh -> install_runtime.sh
  if [ "$TESTMODE" = 1 ]; then
    echo "TESTMODE_NOTE engine tree not synced/checked (a local run never installs into a dev tree)" \
      | tee -a "$OUT/TESTMODE_NOTES.txt"
  else
    if [ ! -f "$MAN" ]; then
      MAN="$OUT/runtime_source_manifest.sha256"
      "$ENG/scripts/bootstrap/sync_engine_tree.sh" "$MAN" > "$OUT/SYNC.log" 2>&1 || fail "ABORT_ENGINE sync failed" 3
    fi
    python3 "$PLAN" --engine-check "$MAN" > "$OUT/ENGINE_TREE_CHECK.txt" 2>&1 \
      || fail "ABORT_ENGINE_TREE_DRIFT installed tree != registered manifest (ENGINE_TREE_CHECK.txt)" 3
  fi
  PSTATE="${SGLANG_ENGINE_DEV:-/opt/pdmux/sglang_engine_dev/python}/sglang/srt/distributed/parallel_state.py"
  [ "$TESTMODE" = 1 ] || grep -q "def pdmux_role_is_thread_local" "$PSTATE" 2>/dev/null \
    || fail "ABORT_ENGINE thread-local role patch absent (correctness gate 3)" 3
  # H17-10: the N3 r3 raw dir is a git-tracked directory WITHOUT its JSONL (git-ignored) in a Job clone, so the
  # rule test's N3 positive control would read an empty dir and FAIL; point it at an absent path so it is an
  # explicit skip (N2C-78: in-Job positive controls are skipped; the evidence is the local rule-seal ledger).
  echo "N2_N3_DIR=$OUT/N3_RAW_NOT_IN_JOB (positive control skipped by design, N2C-78)" > "$OUT/PREFLIGHT_ENV.txt"
  (cd "$HERE" && N2_N3_DIR="$OUT/N3_RAW_NOT_IN_JOB" timeout -k 10 1800 python3 -m unittest test_n2_feasibility \
     test_n2_probe_reduce test_n2_harness test_n2_job n2_reach_tests) > "$OUT/PREFLIGHT_TESTS.txt" 2>&1 \
    || { tail -20 "$OUT/PREFLIGHT_TESTS.txt"; fail "ABORT_PREFLIGHT unit tests (PREFLIGHT_TESTS.txt)" 4; }
  if [ "$TESTMODE" != 1 ]; then
    python3 "$ENG/benchmarks/pdmux_eval/profile_cli.py" engine-source-hash > "$OUT/ENGINE_SOURCE_HASH.txt" 2>&1 \
      || echo "engine-source-hash failed (recorded, §6-3)" >> "$OUT/ENGINE_SOURCE_HASH.txt"
  fi
else
  echo "TEST_SKIP_PREFLIGHT: engine tree / preflight tests NOT run (test mode only)" | tee "$OUT/TEST_SKIP_PREFLIGHT"
fi

# ------------------------------------------------------------------ ports (N3 O1): listen window outside the ephemeral range
read -r EPH_LO EPH_HI < /proc/sys/net/ipv4/ip_local_port_range 2>/dev/null || { EPH_LO=NA; EPH_HI=NA; }
echo "ip_local_port_range=$EPH_LO $EPH_HI listen_port_window=20000-20799" | tee "$OUT/PORT_RANGE.txt"
if [ "$EPH_LO" = NA ] || { [ 20799 -ge "$EPH_LO" ] && [ 20000 -le "$EPH_HI" ]; }; then
  note_or_abort "ABORT_PORT_RANGE listen window 20000-20799 overlaps the ephemeral range $EPH_LO-$EPH_HI" 5
fi

# ------------------------------------------------------------------ cell section (§14-3) + watchdog
CAP_S=$(python3 -c "import json;print(int(json.load(open('$OUT/BUDGET_TABLE.json'))['cap_s']))")
[ -n "${PDMUX_N2_TEST_CAP_S:-}" ] && CAP_S="$PDMUX_N2_TEST_CAP_S"
SECTION_T0=$(date +%s)
DEADLINE=$((SECTION_T0 + CAP_S))
{ echo "column=${JOB}_b cap_s=$CAP_S (cell section, from its start; NOT a billing cap; tail unbounded)"
  echo "section_t0_epoch=$SECTION_T0 deadline_epoch=$DEADLINE"
  [ -f "$META/timing.txt" ] && cat "$META/timing.txt"; } | tee "$OUT/BUDGET.txt"
RUNNER=(python3 "$HERE/n2_job.py" --job "$JOB" --out "$OUT" --deadline-epoch "$DEADLINE")
[ "$DRYRUN" = 1 ] && RUNNER+=(--dry-run)
[ "$FAKE" = 1 ] && RUNNER+=(--fake)
[ -n "${PDMUX_N2_TEST_UNITS:-}" ] && RUNNER+=(--test-units "$PDMUX_N2_TEST_UNITS")
[ -n "${PDMUX_N2_TEST_CELLS:-}" ] && RUNNER+=(--test-cells "$PDMUX_N2_TEST_CELLS")
timeout -k 60 $((CAP_S + 1800)) "${RUNNER[@]}" > "$OUT/RUNNER.log" 2>&1 &
RUN_PID=$!
(
  while [ "$(date +%s)" -lt "$DEADLINE" ]; do kill -0 "$RUN_PID" 2>/dev/null || exit 0; sleep 5; done
  echo "fired_epoch=$(date +%s)" > "$OUT/BUDGET_HARDCAP_FIRED"
  kill -TERM "$RUN_PID" 2>/dev/null          # GNU timeout forwards TERM to n2_job.py -> runner_stop (H-5), graceful
  for _ in $(seq 1 18); do kill -0 "$RUN_PID" 2>/dev/null || exit 0; sleep 5; done
  kill -KILL "$RUN_PID" 2>/dev/null; pkill -9 -f "n2_job.py --job $JOB" 2>/dev/null
  pkill -9 -f "sglang.launch_server" 2>/dev/null; pkill -9 -f "^sglang::" 2>/dev/null
) > /dev/null 2>&1 &
WD_PID=$!
wait "$RUN_PID"
RRC=$?
kill "$WD_PID" 2>/dev/null; WD_PID=""
tail -5 "$OUT/RUNNER.log"
echo "runner_rc=$RRC remaining_s=$((DEADLINE - $(date +%s)))" | tee -a "$OUT/BUDGET.txt"
case "$RRC" in
  0) ;;
  6) fail "ABORT_STICKY/ABORT_GREEN (§11-1, RUNNER.log)" 6 ;;
  8) fail "ABORT_P3 (§3-2, RUNNER.log)" 8 ;;
  9) fail "ABORT_CORRECTNESS (§11-2, RUNNER.log)" 9 ;;
  3) fail "ABORT (registration / CG_BS / CG_CANARY §18-2-d #16 / EA_CANARY §18-2-d #32 / trace, RUNNER.log)" 3 ;;
  *) echo "RUNNER_EXIT $RRC (watchdog / timeout / crash) -- analysis still runs on what exists" ;;
esac

# ------------------------------------------------------------------ analysis (tail, unbounded)
if [ "$DRYRUN" = 1 ]; then
  echo "DRYRUN: no analysis" | tee "$OUT/DRYRUN_NO_ANALYSIS"
else
  python3 "$HERE/n2_analyze.py" --job "$JOB" --out-dir "$OUT" --clk "$META/clk.csv" --substrate "$META/substrate.json" \
    > "$OUT/ANALYZE.log" 2>&1 || echo "ANALYZE_FAILED (see ANALYZE.log)"
  tail -2 "$OUT/ANALYZE.log"
fi
echo "N2_DONE_${JOB}_${JOBID} runner_rc=$RRC"
