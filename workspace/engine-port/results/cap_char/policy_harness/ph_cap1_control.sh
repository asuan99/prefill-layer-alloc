#!/bin/bash
# CAP1 Job + the switch-observability positive control (PREREG §17), one Job, two sequential stages (NOT SUBMITTED):
#   stage 1  ph_job.sh with PDMUX_POLICY_JOB=CAP1 (the measurement; unchanged ladder / bracket rule; its own sealed checks and caps)
#   stage 2  ph_switch_control.py (1 dyn boot, NON-measurement, PHASE_EVENTS on in that boot only)
# The control runs AFTER the CAP1 measurement boots, in a separate process; it cannot change CAP1 data, its verdict or its rc.
# Job rc = CAP1's rc.  The control result is a separate file (switch_control/CONTROL_RESULT.json) and a line in stages.txt.
#   launch.sh --campaign policy_cap1 --target workspace/engine-port/results/cap_char/policy_harness/ph_cap1_control.sh \
#             --env PDMUX_POLICY_JOB=CAP1 --env PDMUX_POLICY_SEALED=1 \
#             --env PDMUX_POLICY_ADDENDUM=workspace/engine-port/results/cap_char/policy_harness/addendum_cap1.json
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
name="${PDMUX_JOB_NAME:-local-$(date -u +%Y%m%d%H%M%S)}"
summary="${PDMUX_POLICY_SUMMARY:-${here}/../policy_runs/${name}}"
export PDMUX_POLICY_SUMMARY="${summary}"
mkdir -p "${summary}"
[[ "${PDMUX_POLICY_JOB:-}" == CAP1 ]] || { echo "ABORT_CONFIG: ph_cap1_control.sh is the CAP1 + control Job; PDMUX_POLICY_JOB must be CAP1" >&2; exit 16; }
bash "${here}/ph_job.sh"; rc1=$?
echo "stage1 CAP1 rc=${rc1}" | tee "${summary}/stages.txt"
timeout -k 60 3600 python3 "${here}/ph_switch_control.py" --out "${summary}/switch_control"
rc2=$?
echo "stage2 ph_switch_control rc=${rc2} (0 PASS, 1 LOG_UNOBSERVABLE, 4 NO_SWITCH/UNRESOLVED) -- does not affect CAP1" | tee -a "${summary}/stages.txt"
exit "${rc1}"
