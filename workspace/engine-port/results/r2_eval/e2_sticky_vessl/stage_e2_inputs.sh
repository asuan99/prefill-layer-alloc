#!/bin/bash
# Stage the seven git-ignored job-908623 telemetry files that the rev3 sec 5-3 estimator
# selftest (`e2_realized_mix.py --selftest`) reads, on the VESSL object volume, so the Job can
# run that submission-blocking selftest on the SAME bytes in the measurement container
# (PREREG_E2_VESSL_A13 sec 2-5).  Local, CPU only.  Dry-run by default.
#
#   stage_e2_inputs.sh            # build + verify the staging dir, print the upload command
#   stage_e2_inputs.sh --upload   # ...and upload it (object-volume storage only, no GPU)
#
# Result on the volume (flat): /io/staging/e2_selftest_archive/{tel_*.jsonl (7), SHA256SUMS}
# The Job re-verifies every digest against e2_vessl_plan.ARCHIVE_DIGESTS, so a wrong upload
# aborts the Job (ABORT_ARCHIVE_DRIFT) before any GPU work.  ~238 MB.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(git -C "${here}" rev-parse --show-toplevel)"
eng="${repo}/workspace/engine-port"
src="${eng}/results/r2_eval/lambda0_prereg/lam0_908623"
stage="${PDMUX_VESSL_STAGE_IN:-$(dirname "${repo}")/_vessl_stage}/e2_selftest_archive"
upload=0; [[ "${1:-}" == "--upload" ]] && upload=1

names=$(PYTHONDONTWRITEBYTECODE=1 python3 -c "import sys; sys.path.insert(0, '${here}'); import e2_vessl_plan as V; print(' '.join(sorted(V.ARCHIVE_DIGESTS)))")
rm -rf "${stage}"; mkdir -p "${stage}"
for f in ${names}; do
  [[ -f "${src}/${f}" ]] || { echo "ERROR: ${src}/${f} missing (unpack the migration bundle's C_results_untracked)" >&2; exit 2; }
  cp -p "${src}/${f}" "${stage}/"
done
# shellcheck disable=SC2086
(cd "${stage}" && sha256sum ${names} > SHA256SUMS)
PYTHONDONTWRITEBYTECODE=1 python3 "${here}/e2_vessl_plan.py" --verify-archive "${stage}"
cat "${stage}/SHA256SUMS"

# shellcheck disable=SC1090
source "${PDMUX_VESSL_ENV:-${eng}/scripts/vessl/vessl.env}"
cmd=(vesslctl volume upload "${PDMUX_VESSL_OBJECT_VOLUME}" "${stage}" --remote-prefix "staging/e2_selftest_archive/")
printf 'upload   :'; printf ' %q' "${cmd[@]}"; echo
if [[ "${upload}" -ne 1 ]]; then
  echo "(dry-run: nothing uploaded; add --upload)"; exit 0
fi
"${cmd[@]}"
vesslctl volume ls "${PDMUX_VESSL_OBJECT_VOLUME}" --prefix "staging/e2_selftest_archive/" || true
