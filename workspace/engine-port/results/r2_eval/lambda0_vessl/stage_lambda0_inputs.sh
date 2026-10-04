#!/bin/bash
# Stage the three git-ignored λ_inf decision inputs of job 907959 on the VESSL object
# volume, so `lambda0_vessl.sh` can run rev5's predicate on the SAME bytes job 908623
# read (PREREG_LAMBDA0_VESSL sec 2-5).  Local, CPU only.  Dry-run by default.
#
#   stage_lambda0_inputs.sh            # build + verify the staging dir, print the upload command
#   stage_lambda0_inputs.sh --upload   # ...and upload it (object-volume storage only, no GPU)
#
# Result on the volume (flat):  /io/staging/lambda0_inputs/{srv_warmup.log,
#   I3a_shapeA.jsonl, I3b_shapeB.jsonl, SHA256SUMS}
# The Job re-verifies every digest against lambda0_vessl_plan.DECISION_INPUT_DIGESTS,
# so a wrong upload aborts the Job (ABORT_F6_DRIFT) before any GPU work.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(git -C "${here}" rev-parse --show-toplevel)"
eng="${repo}/workspace/engine-port"
src="${eng}/results/r2_correctness/job_907959"
stage="${PDMUX_VESSL_STAGE_IN:-$(dirname "${repo}")/_vessl_stage}/lambda0_inputs"
upload=0; [[ "${1:-}" == "--upload" ]] && upload=1

rm -rf "${stage}"; mkdir -p "${stage}"
for rel in srv_warmup.log instrument/I3a_shapeA.jsonl instrument/I3b_shapeB.jsonl; do
  [[ -f "${src}/${rel}" ]] || { echo "ERROR: ${src}/${rel} missing (unpack the migration bundle's C_results_untracked)" >&2; exit 2; }
  cp -p "${src}/${rel}" "${stage}/"
done
(cd "${stage}" && sha256sum srv_warmup.log I3a_shapeA.jsonl I3b_shapeB.jsonl > SHA256SUMS)
python3 - "${stage}" "${here}" <<'PY'
import sys, pathlib
stage, here = pathlib.Path(sys.argv[1]), sys.argv[2]
sys.path.insert(0, here)
import lambda0_vessl_plan as V
bad = 0
for rel in V.STAGED_INPUTS:
    got = V.sha256(stage / pathlib.Path(rel).name)
    want = V.DECISION_INPUT_DIGESTS[rel]
    print(("OK   " if got == want else "BAD  ") + rel + " " + got)
    bad += got != want
sys.exit(1 if bad else 0)
PY
cat "${stage}/SHA256SUMS"

# shellcheck disable=SC1090
source "${PDMUX_VESSL_ENV:-${eng}/scripts/vessl/vessl.env}"
cmd=(vesslctl volume upload "${PDMUX_VESSL_OBJECT_VOLUME}" "${stage}" --remote-prefix "staging/lambda0_inputs/")
printf 'upload   :'; printf ' %q' "${cmd[@]}"; echo
if [[ "${upload}" -ne 1 ]]; then
  echo "(dry-run: nothing uploaded; add --upload)"; exit 0
fi
"${cmd[@]}"
vesslctl volume ls "${PDMUX_VESSL_OBJECT_VOLUME}" --prefix "staging/lambda0_inputs/" || true
