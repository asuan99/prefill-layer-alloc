#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
engine_root="$(cd "${script_dir}/../.." && pwd)"
export PYTHONPATH="${engine_root}/benchmarks${PYTHONPATH:+:${PYTHONPATH}}"

trace_dir="${1:-${engine_root}/results/r2_eval/traces}"
campaign_path="${2:-${engine_root}/results/r2_eval/campaign.json}"
# ★ NOT A MEASUREMENT.  lambda* (the sustainable request rate) is what every
#   workload's intensity is expressed as a fraction of -- W8 "near saturation" is
#   0.90 lambda*, W9 "overload" is 1.10 lambda*.  This default of 4 req/s has
#   never been measured on any model, so with the default those labels are
#   assertions, not facts (project gate #6: measure capacity first).  Set
#   PDMUX_SUSTAINABLE_RATE from a measured saturation point before treating the
#   W8/W9 labels as describing the regime they name.
sustainable_rate="${PDMUX_SUSTAINABLE_RATE:-4}"
request_count="${PDMUX_REQUEST_COUNT:-256}"
# Recorded into every run record so a finished run can say what it served.
model="${PDMUX_MODEL:-Zyphra/Zamba2-2.7B}"
context_length="${PDMUX_CONTEXT_LENGTH:-}"

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
  --model "${model}" \
  ${context_length:+--context-length "${context_length}"} \
  --output "${campaign_path}"

echo "${campaign_path}"
