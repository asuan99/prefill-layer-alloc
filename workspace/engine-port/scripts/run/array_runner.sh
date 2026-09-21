#!/bin/bash
# Scheduler-less replacement for `sbatch --array=<spec> <script>`.
#
# WHY THIS EXISTS.  The rented GPU servers (2026-09-18 migration) have no
# SLURM.  The array-style job scripts already tolerate that -- they read
# `${SLURM_ARRAY_TASK_ID:-${PDMUX_RUN_INDEX:-0}}` for the run index and
# `${SLURM_JOB_ID:-local}` for the job-group label (verified against the
# installed copies with `grep`, not assumed; see the migration plan
# handoff-report/migration_plan_local_dev_remote_gpu_2026-09-18.md sec 3-1).
# This script supplies the loop that `sbatch --array` used to run, plus the
# two things `#SBATCH --output`/`--error` and `sacct` used to give you for
# free: one combined stdout+stderr log per task, and a per-task exit-code
# manifest.  It does NOT retry, does NOT parallelize, and does NOT invent a
# scheduling policy -- one task runs after another, in index order, exactly
# once each.
#
# WHAT IT DOES NOT DO.  It does not know the target script's internal
# results/ layout (r2_eval.sbatch computes its own `run_dir` from
# PDMUX_RESULT_ROOT/PDMUX_CAMPAIGN; a future target script might not even
# have per-index result directories, e.g. e2_sticky.sbatch and
# lambda0.sbatch, which loop internally over cells in ONE job and do not
# need this runner at all -- see the "제약" note in the code review that
# produced this file).  So the LOG directory here is runner-owned and
# separate from campaign telemetry: it never writes into results/<campaign>/.
#
# USAGE
#   array_runner.sh <target-script> <spec> [-- <args passed to target-script>]
#
#   <spec> is either a bare integer N (-> indices 0..N-1, SLURM's implicit
#   convention for `--array=0-N`) or a SLURM-style array spec: comma-separated
#   integers and/or a-b ranges, e.g. "0-3,7,9-10".
#
# ENV
#   PDMUX_RUN_LOG_DIR   Where per-task logs + the summary manifest go.
#                        Default: "$(pwd)/pdmux_array_logs/job_<jobid>"
#                        -- the closest faithful analogue of SLURM's own
#                        default (slurm-%j.out relative to the SUBMISSION
#                        directory, i.e. wherever `sbatch` was invoked from).
#                        Not under results/<campaign>/ on purpose (constraint:
#                        this migration must not touch that layout). Same
#                        "do not commit root run logs" rule from CLAUDE.md
#                        applies to whatever you set this to.
#   SLURM_JOB_ID         Honored if the caller already set it (unchanged
#                        precedence: target scripts read this directly).
#                        If unset, this runner exports ONE synthetic numeric
#                        value (its own PID) for the whole loop, so every
#                        index lands under the same job_<id> bucket the way
#                        one real array submission would. Deliberately
#                        NUMERIC ONLY: engine_bench_runner.sh:12 computes
#                        `port=$((32000 + (${SLURM_JOB_ID:-1} + ...) % 20000))`
#                        in arithmetic context -- a non-numeric value there
#                        does not error, it silently evaluates to 0 and every
#                        task collides on port 32000 (verified empirically,
#                        not assumed).
#   SLURM_ARRAY_TASK_ID  Explicitly UNSET for every task this runner starts,
#                        even if present in the calling shell, so the target
#                        script's own `${SLURM_ARRAY_TASK_ID:-${PDMUX_RUN_INDEX...`
#                        fallback chain reaches PDMUX_RUN_INDEX (this runner's
#                        loop variable) instead of a stale leaked value.
#
# Every other env var the target script needs (PDMUX_CAMPAIGN, PDMUX_ROOT or
# PDMUX_PROJECT_ROOT, PDMUX_R2_POLICY, ...) is the CALLER's job to export
# first, exactly as it would be before `sbatch --array=... target.sbatch`.
# This runner does not default or invent any of them.
set -euo pipefail

target="${1:?usage: array_runner.sh <target-script> <spec> [-- args...]}"
spec="${2:?usage: array_runner.sh <target-script> <spec> [-- args...]}"
shift 2
if [[ "${1:-}" == "--" ]]; then
  shift
fi
target_args=("$@")

if [[ ! -f "${target}" ]]; then
  echo "ERROR: target script not found: ${target}" >&2
  exit 2
fi

# ---- parse <spec> into an ordered list of non-negative integer indices ----
indices=()
if [[ "${spec}" =~ ^[0-9]+$ ]]; then
  for ((i = 0; i < spec; i++)); do indices+=("${i}"); done
else
  IFS=',' read -r -a parts <<<"${spec}"
  for part in "${parts[@]}"; do
    if [[ "${part}" =~ ^([0-9]+)-([0-9]+)$ ]]; then
      lo="${BASH_REMATCH[1]}"; hi="${BASH_REMATCH[2]}"
      if (( hi < lo )); then
        echo "ERROR: bad range '${part}' in spec '${spec}' (hi < lo)" >&2
        exit 2
      fi
      for ((i = lo; i <= hi; i++)); do indices+=("${i}"); done
    elif [[ "${part}" =~ ^[0-9]+$ ]]; then
      indices+=("${part}")
    else
      echo "ERROR: cannot parse spec element '${part}' (want N, 'a-b', or 'a,b,c')" >&2
      exit 2
    fi
  done
fi
if (( ${#indices[@]} == 0 )); then
  echo "ERROR: spec '${spec}' produced zero indices" >&2
  exit 2
fi

# ---- job-group id: honor an inherited SLURM_JOB_ID, else synthesize one
#      numeric value for the whole loop (see ENV comment above) ----
job_id="${SLURM_JOB_ID:-$$}"
export SLURM_JOB_ID="${job_id}"
unset SLURM_ARRAY_TASK_ID || true

log_dir="${PDMUX_RUN_LOG_DIR:-$(pwd)/pdmux_array_logs/job_${job_id}}"
mkdir -p "${log_dir}"
target_name="$(basename "${target}")"
manifest="${log_dir}/${target_name}_${job_id}.summary.tsv"
: > "${manifest}"
printf 'index\texit_code\tlog_path\n' >> "${manifest}"

echo "array_runner: target=${target} job_id=${job_id} indices=${#indices[@]}" \
  "(${indices[*]}) log_dir=${log_dir}" >&2

failures=0
for idx in "${indices[@]}"; do
  log_path="${log_dir}/${target_name}_${job_id}_${idx}.out"
  echo "array_runner: [${idx}] -> ${log_path}" >&2
  set +e
  PDMUX_RUN_INDEX="${idx}" bash "${target}" "${target_args[@]}" \
    > "${log_path}" 2>&1
  rc=$?
  set -e
  printf '%s\t%s\t%s\n' "${idx}" "${rc}" "${log_path}" >> "${manifest}"
  if (( rc != 0 )); then
    echo "array_runner: [${idx}] FAILED rc=${rc} (see ${log_path})" >&2
    failures=$((failures + 1))
  else
    echo "array_runner: [${idx}] ok" >&2
  fi
done

echo "array_runner: ${#indices[@]} task(s), ${failures} failure(s)." \
  "manifest=${manifest}" >&2
if (( failures > 0 )); then
  exit 1
fi
exit 0
