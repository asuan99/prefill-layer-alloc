#!/bin/bash
# CAP2 Job + the lambda* sustainability block (decision 56, PREREG §20), one Job, two sequential stages (NOT SUBMITTED):
#   stage 1  ph_job.sh with PDMUX_POLICY_JOB=CAP2 (the W3 capacity measurement; ladder rmin 0.1 x1.5, 6 points; own sealed checks and caps)
#   stage 2  ph_sustain.py  (W1, D44/D64, 7.29491 req/s, 600 s, seeds 201/202; DESCRIPTIVE, no label)
# Stage 2 starts only after stage 1 and writes into ${summary}/sustain, never into CAP2's raw tree: it cannot change CAP2's data, verdict
# or rc.  Job rc = CAP2's rc.
#   launch.sh --campaign policy_cap2 --target workspace/engine-port/results/cap_char/policy_harness/ph_cap2_sustain.sh \
#             --env PDMUX_POLICY_JOB=CAP2 --env PDMUX_POLICY_SEALED=1 \
#             --env PDMUX_POLICY_ADDENDUM=workspace/engine-port/results/cap_char/policy_harness/addendum_cap2.json
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
name="${PDMUX_JOB_NAME:-local-$(date -u +%Y%m%d%H%M%S)}"
data="${PDMUX_DATA:-/data}"
summary="${PDMUX_POLICY_SUMMARY:-${here}/../policy_runs/${name}}"
export PDMUX_POLICY_SUMMARY="${summary}"
mkdir -p "${summary}"
[[ "${PDMUX_POLICY_JOB:-}" == CAP2 ]] || { echo "ABORT_CONFIG: ph_cap2_sustain.sh is the CAP2 + sustain Job; PDMUX_POLICY_JOB must be CAP2" >&2; exit 16; }
bash "${here}/ph_job.sh"; rc1=$?
echo "stage1 CAP2 rc=${rc1}" | tee "${summary}/stages.txt"
w1="${summary}/w1_lengths.json"
[[ -f "${w1}" ]] || cp "${PDMUX_POLICY_OUT:-${data}/runs/${name}/raw}/w1_lengths.json" "${w1}" 2>/dev/null
timeout -k 60 4200 python3 "${here}/ph_sustain.py" --out "${summary}/sustain" --w1-lengths "${w1}" --clk "${data}/runs/${name}/meta/clk.csv"
rc2=$?
echo "stage2 ph_sustain rc=${rc2} (descriptive; does not affect CAP2)" | tee -a "${summary}/stages.txt"
exit "${rc1}"
