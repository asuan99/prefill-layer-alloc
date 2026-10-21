#!/bin/bash
# Shared by engine_gate_{alpha,beta,gamma}.sh (sourced).  Job entry for the split engine gate:
#   launch.sh --campaign engine_gate_<x> --target workspace/engine-port/scripts/vessl/engine_gate_<x>.sh [--env PDMUX_GATE_FAMILY=A|F]
# job_entry.sh runs this as the Job target (cwd = project root, image ENV, per-run TRITON/XDG caches already exported).
# Order matters for export safety (M3 J1 was truncated mid-export): GATE_STARTED.json first, GATE_RESULT.json is rewritten by
# the driver after every stage, bulky logs are gzipped / excerpted by the driver, DONE is written last by job_entry.sh.
# Design: results/engine_gate/GATE_SPLIT_DESIGN_2026-10-20.md.  Nothing here reads or prints the environment.
set -uo pipefail

gate_main() {   # gate_main <alpha|beta|gamma> <default hardcap minutes>
  local gate="$1" default_cap="$2" here root job rc
  here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  job="${PDMUX_JOB_NAME:-local-$(date -u +%Y%m%d%H%M%S)}"
  export PDMUX_JOB_NAME="${job}"
  export PDMUX_GATE_HARDCAP_MIN="${PDMUX_GATE_HARDCAP_MIN:-${default_cap}}"
  root="${PDMUX_GATE_ROOT:-${here}/../../results/engine_gate/${job}}"
  mkdir -p "${root}" || { echo "ERROR: cannot create ${root}" >&2; return 2; }
  export PDMUX_GATE_ROOT="$(cd "${root}" && pwd)"
  local fam=()
  if [[ "${gate}" == gamma ]]; then
    case "${PDMUX_GATE_FAMILY:-}" in A|F) fam=(--family "${PDMUX_GATE_FAMILY}") ;;
      *) echo "ABORT_CONFIG: PDMUX_GATE_FAMILY must be A or F for G-gamma" >&2
         echo '{"verdict": "ABORT_CONFIG"}' > "${PDMUX_GATE_ROOT}/GATE_RESULT.json"; return 16 ;;
    esac
  fi
  printf '{"gate": "%s", "family": "%s", "job": "%s", "start_utc": "%s", "hardcap_min": %s}\n' \
    "${gate}" "${PDMUX_GATE_FAMILY:-}" "${job}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${PDMUX_GATE_HARDCAP_MIN}" > "${PDMUX_GATE_ROOT}/GATE_STARTED.json"
  if [[ "${PDMUX_GATE_TEST_MODE:-}" != 1 ]]; then   # substrate preconditions, recorded and enforced
    local ngpu; ngpu="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ')"
    { echo "gpu_count=${ngpu}"; ldconfig -p 2>/dev/null | grep -q libnuma && echo "libnuma=present" || echo "libnuma=MISSING"; } > "${PDMUX_GATE_ROOT}/preflight.txt"
    if [[ "${ngpu}" != 1 ]]; then echo "ABORT_SUBSTRATE: ${ngpu} GPUs visible, the gate needs exactly 1" >&2; return 16; fi
  fi
  # the driver stops starting stages RESERVE earlier; this wrapper is the last line (SIGTERM, then SIGKILL after 60 s)
  timeout -k 60 "$(( PDMUX_GATE_HARDCAP_MIN * 60 ))" python3 "${here}/engine_gate_driver.py" "${gate}" "${fam[@]+"${fam[@]}"}"
  rc=$?
  if [[ ${rc} -eq 124 || ${rc} -eq 137 ]]; then
    echo "{\"verdict\": \"HARDCAP_KILLED\", \"rc\": ${rc}}" > "${PDMUX_GATE_ROOT}/HARDCAP_KILLED.json"
  fi
  echo "engine_gate ${gate}: rc=${rc} (0 PASS/DIAGNOSTIC_COMPLETE, 1 FAIL/OBSERVER_EFFECT, 4 UNRESOLVED, 12 model rev, 13 engine digest, 16 config/registration)"
  return "${rc}"
}
