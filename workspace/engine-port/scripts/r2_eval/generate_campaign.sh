#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
engine_root="$(cd "${script_dir}/../.." && pwd)"
export PYTHONPATH="${engine_root}/benchmarks${PYTHONPATH:+:${PYTHONPATH}}"

trace_dir="${1:-${engine_root}/results/r2_eval/traces}"
campaign_path="${2:-${engine_root}/results/r2_eval/campaign.json}"
# ★ lambda* IS AN INPUT NOW, AND THERE IS NO DEFAULT (revised 2026-09-13).
#   This block is cited as `generate_campaign.sh:10`.  It used to introduce
#   `sustainable_rate="${PDMUX_SUSTAINABLE_RATE:-4}"` -- ONE scalar, never
#   measured on any model, applied to all nine workloads.  Two things were
#   wrong with that and both are fixed here rather than documented:
#
#     (1) Gate #6 ("measure capacity first"): every workload is a FRACTION of
#         lambda* (W1 0.60, W3 0.80, W8 0.90, W9 1.10), so with an unmeasured
#         default the names "near saturation" and "overload" were assertions.
#         => missing table is now a hard failure, not a default.
#     (2) W4 was self-contradictory: its phases are (8192,64) and (256,512),
#         whose capacities differ ~5x, so no scalar puts both at 0.80x.
#         => lambda* is now PER SHAPE, and W4 scales each phase by its own.
#
#   The table format lives in benchmarks/pdmux_eval/lambda_star.py.  It must say
#   for EVERY shape whether that rate was `measured`, and which definition it is
#   (`slo_sustainable` = the canon's, `throughput_saturation` = the knee, which
#   is a larger quantity and does NOT close gate #6).  Traces and the manifest
#   carry that provenance, and an unmeasured lambda* is refused unless
#   PDMUX_ALLOW_UNMEASURED_LAMBDA_STAR=1 is set deliberately.
lambda_star_table="${PDMUX_LAMBDA_STAR_TABLE:-}"
if [[ -z "${lambda_star_table}" ]]; then
  echo "ERROR: PDMUX_LAMBDA_STAR_TABLE is not set." >&2
  echo "       lambda* is measured in campaign stage 0 (lambda*는 캠페인 0단계에서" >&2
  echo "       측정된다); it has no default here because the default that used to" >&2
  echo "       exist (4 req/s) was never measured on any model and made every" >&2
  echo "       workload-intensity label an assertion (gate #6)." >&2
  echo "       Template + format: ${script_dir}/lambda_star.example.json and" >&2
  echo "       ${engine_root}/benchmarks/pdmux_eval/lambda_star.py" >&2
  exit 2
fi
if [[ -n "${PDMUX_SUSTAINABLE_RATE:-}" ]]; then
  echo "ERROR: PDMUX_SUSTAINABLE_RATE=${PDMUX_SUSTAINABLE_RATE} was removed on" >&2
  echo "       2026-09-13.  A single scalar cannot parameterise W4 (prefill" >&2
  echo "       (8192,64) at 0.80x and decode (256,512) at 0.80x are mutually" >&2
  echo "       exclusive).  Put per-shape rates in PDMUX_LAMBDA_STAR_TABLE." >&2
  exit 2
fi
request_count="${PDMUX_REQUEST_COUNT:-256}"
# Claim D's first campaign needs W3+W4 only; the full list stays the default so
# nothing silently narrows, and the lambda* gate refuses the workloads whose
# shapes have not been measured.
workloads="${PDMUX_WORKLOADS:-W1 W2 W3 W4 W5 W6 W7 W8 W9}"
allow_unmeasured=()
if [[ "${PDMUX_ALLOW_UNMEASURED_LAMBDA_STAR:-0}" == "1" ]]; then
  allow_unmeasured=(--allow-unmeasured-lambda-star)
fi
# Recorded into every run record so a finished run can say what it served.
model="${PDMUX_MODEL:-Zyphra/Zamba2-2.7B}"
context_length="${PDMUX_CONTEXT_LENGTH:-}"

mkdir -p "${trace_dir}"
for workload in ${workloads}; do
  python -m pdmux_eval.workloads \
    --workload "${workload}" \
    --lambda-star-table "${lambda_star_table}" \
    --count "${request_count}" \
    --seed 1 \
    ${allow_unmeasured[@]+"${allow_unmeasured[@]}"} \
    --output "${trace_dir}/${workload}.jsonl"
done

python -m pdmux_eval.campaign \
  --trace-dir "${trace_dir}" \
  --workloads ${workloads} \
  --repetitions "${PDMUX_REPETITIONS:-5}" \
  --seed 1 \
  --model "${model}" \
  ${context_length:+--context-length "${context_length}"} \
  ${allow_unmeasured[@]+"${allow_unmeasured[@]}"} \
  --output "${campaign_path}"

echo "${campaign_path}"
