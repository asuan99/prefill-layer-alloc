#!/bin/bash
# TUNE / EVAL-a / EVAL-b Job with the substrate-covariate wrapper (decision 57; NOT SUBMITTED):
#   stage 0  ph_refcell.py --tag start   (W1, D44, 7.29491 req/s, 120 requests: descriptive reference cell)
#   stage 1  ph_job.sh with PDMUX_POLICY_JOB (the measurement; its own sealed checks, caps, raw export)
#   stage 2  ph_refcell.py --tag end
#   stage 3  ph_covariates.py (Job/GPU/driver/host, per-run + per-segment throttle / power / temp / SM clk, completion rate, drain)
# Stages 0/2/3 are separate processes and boots; they write into ${summary}/covariates and cannot change the measurement's data,
# verdict or rc (Job rc = stage 1's rc).  Nothing here is a decision input.
#   launch.sh --campaign policy_tune --target workspace/engine-port/results/cap_char/policy_harness/ph_cov_job.sh \
#             --env PDMUX_POLICY_JOB=TUNE --env PDMUX_POLICY_SEALED=1 \
#             --env PDMUX_POLICY_ADDENDUM=workspace/engine-port/results/cap_char/policy_harness/addendum_tune.json
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
name="${PDMUX_JOB_NAME:-local-$(date -u +%Y%m%d%H%M%S)}"
data="${PDMUX_DATA:-/data}"
summary="${PDMUX_POLICY_SUMMARY:-${here}/../policy_runs/${name}}"
export PDMUX_POLICY_SUMMARY="${summary}"
cov="${summary}/covariates"
mkdir -p "${cov}"
case "${PDMUX_POLICY_JOB:-}" in TUNE|EVAL-a|EVAL-b) ;; *) echo "ABORT_CONFIG: ph_cov_job.sh is for TUNE|EVAL-a|EVAL-b; got '${PDMUX_POLICY_JOB:-}'" >&2; exit 16 ;; esac
clk="${data}/runs/${name}/meta/clk.csv"
raw="${PDMUX_POLICY_OUT:-${data}/runs/${name}/raw}"

# the W1 length table is built by ph_job.sh; stage 0 needs it first, so build it the same way (deterministic)
sg="${HF_HOME:-${data}/hf}/raw/ShareGPT_V3_unfiltered_cleaned_split.json"
tok="${HF_HOME:-${data}/hf}/hub/models--nvidia--NVIDIA-Nemotron-Nano-9B-v2-Base/snapshots/dc0661c829b14e5b9246c05cfa89094a0875e052/tokenizer.json"
w1="${cov}/w1_lengths.json"
python3 "${here}/ph_trace.py" build-w1 --sharegpt "${sg}" --tokenizer "${tok}" --out "${w1}" > "${cov}/w1_lengths.txt" 2>&1 \
  || echo "WARN: reference-cell W1 table failed; reference cells skipped" | tee -a "${cov}/stages.txt"

if [[ -f "${w1}" ]]; then
  timeout -k 30 1200 python3 "${here}/ph_refcell.py" --out "${cov}" --w1-lengths "${w1}" --tag start --clk "${clk}"
  echo "stage0 refcell start rc=$?" | tee -a "${cov}/stages.txt"
fi
bash "${here}/ph_job.sh"; rc1=$?
echo "stage1 ${PDMUX_POLICY_JOB} rc=${rc1}" | tee -a "${cov}/stages.txt"
if [[ -f "${w1}" ]]; then
  timeout -k 30 1200 python3 "${here}/ph_refcell.py" --out "${cov}" --w1-lengths "${w1}" --tag end --clk "${clk}"
  echo "stage2 refcell end rc=$?" | tee -a "${cov}/stages.txt"
fi
python3 "${here}/ph_covariates.py" --raw "${raw}" --clk "${clk}" --substrate "${data}/runs/${name}/meta/substrate.json" --out "${cov}/covariates.json"
echo "stage3 ph_covariates rc=$? (descriptive; does not affect the measurement)" | tee -a "${cov}/stages.txt"
exit "${rc1}"
