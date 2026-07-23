#!/bin/bash
#SBATCH --job-name=pdmux-r2
#SBATCH --partition=amd_a100nv_8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
engine_root="$(cd "${script_dir}/../.." && pwd)"
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
export PDMUX_ENGINE_COMMIT="$(git -C "${engine_root}/../.." rev-parse HEAD)"
python -c 'import json,os,pathlib; p=pathlib.Path(os.environ["PDMUX_CAMPAIGN"]); i=int(os.environ.get("SLURM_ARRAY_TASK_ID",os.environ.get("PDMUX_RUN_INDEX","0"))); d=json.loads(p.read_text()); pathlib.Path(os.environ["PDMUX_RUN_RECORD"]).write_text(json.dumps(d["runs"][i],indent=2,sort_keys=True)+"\n")'

export PDMUX_TELEMETRY_PATH="${run_dir}/telemetry.jsonl"
export PDMUX_ENGINE_BENCH_RUNNER="${PDMUX_ENGINE_BENCH_RUNNER:-${script_dir}/engine_bench_runner.sh}"

r2_runner="${PDMUX_R2_RUNNER:-${script_dir}/policy_adapter.sh}"
"${r2_runner}" "${PDMUX_RUN_RECORD}" "${run_dir}"
