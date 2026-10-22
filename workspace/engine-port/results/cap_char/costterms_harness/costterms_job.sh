#!/bin/bash
# Job target for the cost-term measurement (PREREG_COSTMODEL_TERMS_MEAS_rev3, rev3.4 + rev3.5 draft): J-H1, J-A (with
# its T3 boots), J-B2, J-H2, J-C.  NOT SEALED: refuses to run unless PDMUX_COSTTERMS_SEALED=1 (set only after seal S1,
# PREREG §12 item 13) or PDMUX_COSTTERMS_TEST_MODE=1 (CPU tests with the fake engine).  The Workspace smoke does not
# use this file (ct_runner.py run --smoke, H-1).
#
#   launch.sh --campaign costterms_<job> --target workspace/engine-port/results/cap_char/costterms_harness/costterms_job.sh \
#             --env PDMUX_COSTTERMS_JOB=J-H1|J-A|J-B2|J-H2|J-C [--env PDMUX_COSTTERMS_ROUND=1|2] \
#             [--env PDMUX_COSTTERMS_OUT=/data/costterms/<job>_r<round> --env PDMUX_COSTTERMS_RESUME=1]
#
# job_entry.sh runs this with cwd = project root, HF_HOME / TRITON caches exported, DONE written by job_entry.sh after us.
# H-3 output split: RAW (server logs, decode / TEL / admission records, cells) = ${PDMUX_COSTTERMS_OUT:-/data/runs/<job
# name>/raw} on the cluster volume; SUMMARY (analysis.json, plan.json, RUN_DONE, SHA256SUMS of the raw tree,
# RESUME_LOG.json, t3_iterations.json, boot DONE manifests, RAW_LOCATION.txt) = results/cap_char/costterms_runs/<job
# name>/ -- the only part job_entry.sh exports.
# H-4 resume: give the SAME fixed PDMUX_COSTTERMS_OUT (e.g. /data/costterms/J-A_r1) to every submission and set
# PDMUX_COSTTERMS_RESUME=1 on the re-submission; finished cells are kept, the rest run in new '<boot>@s<k>' boots, and
# RESUME_LOG.json records it (report it in the result).  Without RESUME=1 a non-empty raw directory aborts (rc 16).
# rc: 0 ok, 13 ABORT_ENGINE_DIGEST, 16 config / unsealed / substrate, 124/137 hard cap.  Nothing here prints the env.
set -uo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
eng="$(cd "${here}/../../.." && pwd)"
job="${PDMUX_COSTTERMS_JOB:-}"
rnd="${PDMUX_COSTTERMS_ROUND:-1}"
name="${PDMUX_JOB_NAME:-local-$(date -u +%Y%m%d%H%M%S)}"
data="${PDMUX_DATA:-/data}"
raw="${PDMUX_COSTTERMS_OUT:-${data}/runs/${name}/raw}"
summary="${PDMUX_COSTTERMS_SUMMARY:-${here}/../costterms_runs/${name}}"

case "${job}" in              # hard caps (min) = ct_spec.HARD_CAP_MIN; GPUs = ct_spec.JOB_GPUS
  J-H1) cap_min=246; ngpu_want=1 ;;
  J-A)  if [[ "${rnd}" == 1 ]]; then cap_min=244; else cap_min=235; fi; ngpu_want=1 ;;
  J-B2) cap_min=78;  ngpu_want=1 ;;
  J-H2) cap_min=126; ngpu_want=1 ;;
  J-C)  cap_min=215; ngpu_want=2 ;;
  *) echo "ABORT_CONFIG: PDMUX_COSTTERMS_JOB must be J-H1, J-A, J-B2, J-H2 or J-C; got '${job}'" >&2; exit 16 ;;
esac
if [[ "${PDMUX_COSTTERMS_SEALED:-}" != 1 && "${PDMUX_COSTTERMS_TEST_MODE:-}" != 1 ]]; then
  echo "ABORT_UNSEALED: cost-term harness not sealed (PREREG §12 item 13)" >&2
  exit 16
fi
mkdir -p "${raw}" "${summary}" || { echo "ERROR: cannot create ${raw} / ${summary}" >&2; exit 2; }

if [[ "${PDMUX_COSTTERMS_TEST_MODE:-}" != 1 ]]; then        # substrate preconditions, recorded and enforced
  ngpu="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ')"
  { echo "gpu_count=${ngpu}"; ldconfig -p 2>/dev/null | grep -q libnuma && echo "libnuma=present" || echo "libnuma=MISSING"; } \
    > "${summary}/preflight.txt"
  if [[ "${ngpu}" != "${ngpu_want}" ]]; then echo "ABORT_SUBSTRATE: ${ngpu} GPUs visible, ${job} needs exactly ${ngpu_want}" >&2; exit 16; fi
  if grep -q 'libnuma=MISSING' "${summary}/preflight.txt"; then echo "ABORT_SUBSTRATE: libnuma missing (sgl_kernel import)" >&2; exit 16; fi
fi

# ---- H-2 ABORT_ENGINE_DIGEST (M3 mg_job_common.sh:151-161 form): the installed engine tree (manifest written by
# job_entry / install_runtime) must equal the digest sealed with this harness (ENGINE_DIGEST.txt, pathless-v1, 30 entries)
manifest="${PDMUX_COSTTERMS_MANIFEST:-${data}/runs/${name}/meta/runtime_source_manifest.sha256}"
sealed="$(tr -d ' \n' < "${here}/ENGINE_DIGEST.txt" 2>/dev/null || echo ABSENT)"
python3 "${eng}/scripts/bootstrap/engine_digest.py" check --manifest "${manifest}" --sealed-file "${here}/ENGINE_DIGEST.txt" \
  --out "${summary}/ENGINE_DIGEST_CHECK.json" > "${summary}/ENGINE_DIGEST_CHECK.txt" 2>&1
drc=$?
echo "sealed=${sealed} check_rc=${drc}" >> "${summary}/ENGINE_DIGEST_CHECK.txt"
if [[ ${drc} -ne 0 ]]; then
  echo "ABORT_ENGINE_DIGEST sealed=${sealed} check_rc=${drc} (see ENGINE_DIGEST_CHECK.txt)" >&2
  exit 13
fi

# the Nano snapshot the dummy composition variants are built from (J-H1, J-H2); others load real weights by id + revision
nano_rev="dc0661c829b14e5b9246c05cfa89094a0875e052"
src="${PDMUX_COSTTERMS_VARIANT_SRC:-${HF_HOME:-${data}/hf}/hub/models--nvidia--NVIDIA-Nemotron-Nano-9B-v2-Base/snapshots/${nano_rev}}"
args=(run --job "${job}" --round "${rnd}" --out "${raw}")
if [[ "${job}" == J-H1 || "${job}" == J-H2 ]]; then args+=(--variant-src "${src}" --variant-root "${raw}/variants"); fi
if [[ "${PDMUX_COSTTERMS_RESUME:-}" == 1 ]]; then args+=(--resume); fi

timeout -k 60 "$(( cap_min * 60 ))" python3 "${here}/ct_runner.py" "${args[@]}"
rc=$?
if [[ ${rc} -eq 124 || ${rc} -eq 137 ]]; then
  echo "{\"verdict\": \"HARDCAP_KILLED\", \"rc\": ${rc}}" > "${summary}/HARDCAP_KILLED.json"
fi
python3 "${here}/ct_runner.py" analyze "${raw}" --summary "${summary}" --clk "${data}/runs/${name}/meta/clk.csv" \
  || echo "analysis failed (rc=$?); raw data kept in ${raw}" >&2
echo "costterms ${job} r${rnd}: rc=${rc} (0 ok, 13 engine digest, 16 config/unsealed, 124/137 hard cap); raw=${raw}"
exit "${rc}"
