#!/bin/bash
# PDMUX_PHASE_DIAG GPU correctness gate, Job entry (engine_gate_diag.py docstring: D0-D6).  NOT SUBMITTED / NOT RUN.
#   launch.sh --campaign engine_gate_diag --target workspace/engine-port/scripts/vessl/engine_gate_diag.sh  (dry-run -> approval -> --submit)
# Requires PDMUX_DIAG_GATE_DIGEST = the ENGINE_DIGEST registered for the instrumentation commit (else ABORT_ENGINE_DIGEST, rc 13).
# Hard cap 120 min (self-enforced): 5 Nano boots (OFF, ON+sync audit, PHASE-only, static non-sticky D34, ON+PDMUX_SLO_SCHED);
# sequential greedy on OFF/ON, 24 concurrent requests on every boot.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
job="${PDMUX_JOB_NAME:-local-$(date -u +%Y%m%d%H%M%S)}"
export PDMUX_JOB_NAME="${job}"
export PDMUX_GATE_HARDCAP_MIN="${PDMUX_GATE_HARDCAP_MIN:-120}"
root="${PDMUX_GATE_ROOT:-${here}/../../results/engine_gate/${job}}"
mkdir -p "${root}" || { echo "ERROR: cannot create ${root}" >&2; exit 2; }
export PDMUX_GATE_ROOT="$(cd "${root}" && pwd)"
ngpu="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ')"
if [[ "${ngpu}" != 1 ]]; then echo "ABORT_SUBSTRATE: ${ngpu} GPUs visible, the gate needs exactly 1" >&2; exit 16; fi
timeout -k 60 "$(( PDMUX_GATE_HARDCAP_MIN * 60 ))" python3 "${here}/engine_gate_diag.py" run
rc=$?
echo "engine_gate_diag: rc=${rc} (0 PASS, 1 FAIL, 4 UNRESOLVED, 13 engine digest)"
exit "${rc}"
