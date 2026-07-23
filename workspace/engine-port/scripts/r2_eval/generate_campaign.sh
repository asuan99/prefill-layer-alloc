#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
engine_root="$(cd "${script_dir}/../.." && pwd)"
export PYTHONPATH="${engine_root}/benchmarks${PYTHONPATH:+:${PYTHONPATH}}"

trace_dir="${1:-${engine_root}/results/r2_eval/traces}"
campaign_path="${2:-${engine_root}/results/r2_eval/campaign.json}"
sustainable_rate="${PDMUX_SUSTAINABLE_RATE:-4}"
request_count="${PDMUX_REQUEST_COUNT:-256}"

mkdir -p "${trace_dir}"
for workload in W1 W2 W3 W4 W5 W6 W7 W8 W9; do
  python -m pdmux_eval.workloads \
    --workload "${workload}" \
    --sustainable-rate "${sustainable_rate}" \
    --count "${request_count}" \
    --seed 1 \
    --output "${trace_dir}/${workload}.jsonl"
done

python -m pdmux_eval.campaign \
  --trace-dir "${trace_dir}" \
  --repetitions "${PDMUX_REPETITIONS:-5}" \
  --seed 1 \
  --output "${campaign_path}"

echo "${campaign_path}"
