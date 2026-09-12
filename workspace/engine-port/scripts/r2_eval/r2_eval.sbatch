#!/bin/bash
#SBATCH --job-name=pdmux-r2
#SBATCH --partition=amd_a100nv_8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --comment="field=efficientai;appl=pytorch"
set -euo pipefail

# ============================================================================
# ★ WHY NOT ${BASH_SOURCE[0]}
#   sbatch COPIES the submitted script into the node's spool directory, so in a
#   running batch job ${BASH_SOURCE[0]} is /var/spool/slurmd/job<ID>/slurm_script
#   -- not this file.  The previous `script_dir="$(cd "$(dirname
#   "${BASH_SOURCE[0]}")" && pwd)"` therefore resolved engine_root to the spool,
#   and every path under it (sync_engine_tree.sh, policy_adapter.sh,
#   benchmarks/) pointed at nothing.  No run ever reached the server, which is
#   why results/r2_eval/ has never existed.
#
#   Fix = pin the absolute project path, the same way the one R2 job script that
#   DOES run does it (results/r2_correctness/r2_correctness.sbatch:99
#   `PROJECT=/scratch/ehmoon/whlee/prefill-layer-alloc`).  This holds in all
#   three ways this file gets executed, because none of them depend on cwd or on
#   where Slurm staged the script:
#     (a) sbatch          -- spool copy irrelevant; the constant is in the copy.
#     (b) login-node run  -- `bash .../r2_eval.sbatch` ignores #SBATCH lines and
#                            takes the same constant.
#     (c) array job       -- every task re-executes the same spool copy; only
#                            SLURM_ARRAY_TASK_ID differs, and it is read below.
#   ${SLURM_SUBMIT_DIR} was the alternative and is rejected: it is the directory
#   sbatch was CALLED from, which is a user habit, not a repository fact, and it
#   is unset in (b).
#
#   PDMUX_PROJECT_ROOT relocates the checkout; the guard below turns a wrong
#   value into an immediate exit instead of a silently wrong tree.
# ============================================================================
project_root="${PDMUX_PROJECT_ROOT:-/scratch/ehmoon/whlee/prefill-layer-alloc}"
engine_root="${project_root}/workspace/engine-port"
script_dir="${engine_root}/scripts/r2_eval"
for required in \
  "${engine_root}/scripts/bootstrap/sync_engine_tree.sh" \
  "${script_dir}/policy_adapter.sh" \
  "${script_dir}/engine_bench_runner.sh" \
  "${engine_root}/benchmarks/pdmux_eval/campaign.py" \
  "${engine_root}/benchmarks/configs/pdmux_r2.yml"; do
  if [[ ! -f "${required}" ]]; then
    echo "ERROR: project_root=${project_root} is wrong: missing ${required}" >&2
    echo "       set PDMUX_PROJECT_ROOT to the prefill-layer-alloc checkout." >&2
    exit 2
  fi
done

campaign_path="${PDMUX_CAMPAIGN:?set PDMUX_CAMPAIGN to campaign.json}"
run_index="${SLURM_ARRAY_TASK_ID:-${PDMUX_RUN_INDEX:-0}}"
result_root="${PDMUX_RESULT_ROOT:-${engine_root}/results/r2_eval}"
run_dir="${result_root}/job_${SLURM_JOB_ID:-local}/run_${run_index}"
mkdir -p "${run_dir}"

"${engine_root}/scripts/bootstrap/sync_engine_tree.sh" \
  "${run_dir}/runtime_source_manifest.sha256"

export PYTHONPATH="${engine_root}/benchmarks${PYTHONPATH:+:${PYTHONPATH}}"
export PDMUX_CAMPAIGN="${campaign_path}"
export PDMUX_RUN_INDEX="${run_index}"
export PDMUX_RUN_RECORD="${run_dir}/run.json"
# Split assignment from export: `export V="$(cmd)"` returns export's status, so
# a failing git would be swallowed by set -e and recorded as an empty commit.
PDMUX_ENGINE_COMMIT="$(git -C "${project_root}" rev-parse HEAD)"
export PDMUX_ENGINE_COMMIT
python -c 'import json,os,pathlib; p=pathlib.Path(os.environ["PDMUX_CAMPAIGN"]); i=int(os.environ.get("SLURM_ARRAY_TASK_ID",os.environ.get("PDMUX_RUN_INDEX","0"))); d=json.loads(p.read_text()); pathlib.Path(os.environ["PDMUX_RUN_RECORD"]).write_text(json.dumps(d["runs"][i],indent=2,sort_keys=True)+"\n")'

# ============================================================================
# ★ CONTEXT LENGTH PRE-FLIGHT (the second reason results/r2_eval/ never existed)
#   engine_bench_runner.sh boots with `--context-length ${PDMUX_CONTEXT_LENGTH:-
#   16384}` and NOTHING in the campaign path ever set that variable.  Zamba2-2.7B
#   declares max_position_embeddings=4096, so SGLang raises in
#   ModelConfig._derive_context_length and the server dies before /health.
#
#   We derive the value from the served model's own config instead of hardcoding
#   4096, so switching PDMUX_MODEL cannot silently reintroduce the bug.
#   SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN is explicitly UNSET, not used: it
#   does not lengthen the model, it only extrapolates RoPE past training and
#   contaminates the tokens a serving result is measuring.
#
#   The same pre-flight refuses a trace this model cannot serve honestly (an
#   8192-token W2/W4/W5 prompt at 4096 is either rejected outright or silently
#   truncated in max_new_tokens).  Refusing costs nothing; discovering it after
#   the allocation costs GPU-hours and produces a contaminated arm.
# ============================================================================
unset SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN
# Default duplicated from engine_bench_runner.sh:10 on purpose (the runner must
# stay usable standalone); tests/test_r2_eval_runner.py asserts they agree.
export PDMUX_MODEL="${PDMUX_MODEL:-Zyphra/Zamba2-2.7B}"
run_trace="$(python -c 'import json,os,pathlib; print(json.loads(pathlib.Path(os.environ["PDMUX_RUN_RECORD"]).read_text())["trace_path"])')"
preflight_args=(--model "${PDMUX_MODEL}" --trace "${run_trace}")
if [[ -n "${PDMUX_CONTEXT_LENGTH:-}" ]]; then
  preflight_args+=(--context-length "${PDMUX_CONTEXT_LENGTH}")
fi
resolved_ctx="$(python -m pdmux_eval.context_limit "${preflight_args[@]}")"
export PDMUX_CONTEXT_LENGTH="${resolved_ctx}"
echo "pre-flight: model=${PDMUX_MODEL} context_length=${PDMUX_CONTEXT_LENGTH}" \
  "trace=${run_trace}"

export PDMUX_TELEMETRY_PATH="${run_dir}/telemetry.jsonl"
export PDMUX_ENGINE_BENCH_RUNNER="${PDMUX_ENGINE_BENCH_RUNNER:-${script_dir}/engine_bench_runner.sh}"

r2_runner="${PDMUX_R2_RUNNER:-${script_dir}/policy_adapter.sh}"
"${r2_runner}" "${PDMUX_RUN_RECORD}" "${run_dir}"
