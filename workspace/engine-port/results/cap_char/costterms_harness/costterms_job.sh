#!/bin/bash
# Job target for the cost-term measurement, stage 1 (PREREG_COSTMODEL_TERMS_MEAS_rev3 rev3.3: J-H1, J-A T1/T2/T2' part,
# J-B2).  NOT SEALED: refuses to run unless PDMUX_COSTTERMS_SEALED=1 (set only after seal S1, PREREG §12 item 13) or
# PDMUX_COSTTERMS_TEST_MODE=1 (CPU tests with the fake engine).  Stage 2 (J-H2, J-C, T3 boots) is not wired.
#
#   launch.sh --campaign costterms_<job> --target workspace/engine-port/results/cap_char/costterms_harness/costterms_job.sh \
#             --env PDMUX_COSTTERMS_JOB=J-H1|J-A|J-B2 [--env PDMUX_COSTTERMS_ROUND=1|2]
#
# job_entry.sh runs this with cwd = project root, HF_HOME / TRITON caches exported, DONE written by job_entry.sh after us.
# Output: ${PDMUX_COSTTERMS_OUT:-results/cap_char/costterms_runs/<job name>} -- boots/<boot>/..., plan.json, analysis.json,
# SHA256SUMS, RUN_DONE.  Nothing here reads or prints the environment.
set -uo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
job="${PDMUX_COSTTERMS_JOB:-}"
rnd="${PDMUX_COSTTERMS_ROUND:-1}"
name="${PDMUX_JOB_NAME:-local-$(date -u +%Y%m%d%H%M%S)}"
out="${PDMUX_COSTTERMS_OUT:-${here}/../costterms_runs/${name}}"

case "${job}" in
  J-H1) cap_min=210 ;;          # PREREG §7-4 hard caps (x1.2): J-H1 3.5 h, J-A1 5.2 h, J-A2 5.0 h, J-B2 1.3 h
  J-A)  if [[ "${rnd}" == 1 ]]; then cap_min=312; else cap_min=300; fi ;;
  J-B2) cap_min=78 ;;
  *) echo "ABORT_CONFIG: PDMUX_COSTTERMS_JOB must be J-H1, J-A or J-B2 (stage 1); got '${job}'" >&2; exit 16 ;;
esac
if [[ "${PDMUX_COSTTERMS_SEALED:-}" != 1 && "${PDMUX_COSTTERMS_TEST_MODE:-}" != 1 ]]; then
  echo "ABORT_UNSEALED: cost-term harness not sealed (PREREG §12 item 13)" >&2
  exit 16
fi
mkdir -p "${out}" || { echo "ERROR: cannot create ${out}" >&2; exit 2; }

if [[ "${PDMUX_COSTTERMS_TEST_MODE:-}" != 1 ]]; then        # substrate preconditions, recorded and enforced
  ngpu="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ')"
  { echo "gpu_count=${ngpu}"; ldconfig -p 2>/dev/null | grep -q libnuma && echo "libnuma=present" || echo "libnuma=MISSING"; } \
    > "${out}/preflight.txt"
  if [[ "${ngpu}" != 1 ]]; then echo "ABORT_SUBSTRATE: ${ngpu} GPUs visible, stage-1 Jobs need exactly 1" >&2; exit 16; fi
  if grep -q 'libnuma=MISSING' "${out}/preflight.txt"; then echo "ABORT_SUBSTRATE: libnuma missing (sgl_kernel import)" >&2; exit 16; fi
fi

# the Nano snapshot the dummy composition variants are built from (J-H1); J-A / J-B2 load real weights by id + revision
nano_rev="dc0661c829b14e5b9246c05cfa89094a0875e052"
src="${PDMUX_COSTTERMS_VARIANT_SRC:-${HF_HOME:-/data/hf}/hub/models--nvidia--NVIDIA-Nemotron-Nano-9B-v2-Base/snapshots/${nano_rev}}"
args=(run --job "${job}" --round "${rnd}" --out "${out}")
if [[ "${job}" == J-H1 ]]; then args+=(--variant-src "${src}" --variant-root "${out}/variants"); fi

timeout -k 60 "$(( cap_min * 60 ))" python3 "${here}/ct_runner.py" "${args[@]}"
rc=$?
if [[ ${rc} -eq 124 || ${rc} -eq 137 ]]; then
  echo "{\"verdict\": \"HARDCAP_KILLED\", \"rc\": ${rc}}" > "${out}/HARDCAP_KILLED.json"
fi
python3 "${here}/ct_runner.py" analyze "${out}" || echo "analysis failed (rc=$?); raw data kept" >&2
echo "costterms ${job} r${rnd}: rc=${rc} (0 ok, 16 config/unsealed, 124/137 hard cap)"
exit "${rc}"
