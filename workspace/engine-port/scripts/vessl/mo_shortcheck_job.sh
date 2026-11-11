#!/bin/bash
# VESSL Job target: second short confirmation Job of the motivation track (NON-MEASUREMENT; see mo_shortcheck.py).
# Image: the py3.12 runtime (PDMUX_VESSL_IMAGE_SG312, same digest as the first probe) -> launch with
#   PDMUX_VESSL_ENV=workspace/engine-port/scripts/vessl/vessl.sg312.env scripts/vessl/launch.sh --campaign mo_shortcheck --target <this file>
# Preflight (before any boot; rc != 0 stops the Job): exactly one GPU; engine digest of the installed tree == 8d8f6720 (dev_tree_edits #40).
set -euo pipefail
: "${PDMUX_PROJECT_ROOT:?}"
eng="${PDMUX_PROJECT_ROOT}/workspace/engine-port"
name="${PDMUX_JOB_NAME:-local-$(date -u +%Y%m%d%H%M%S)}"
data="${PDMUX_DATA:-/data}"
out_root="${eng}/results/mo_shortcheck"
expected="8d8f67201199bd181f64aae043b6a8ec5c6772799dbf964081584684e9e9c858"
pre="${out_root}/preflight/${name}"
mkdir -p "${pre}"
ngpu="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true)"
{ echo "gpu_count=${ngpu}"; echo "image=${PDMUX_IMAGE:-}"; } > "${pre}/preflight.txt"
[[ "${ngpu}" == 1 ]] || { echo "ABORT_SUBSTRATE: ${ngpu} GPUs visible, this Job needs exactly 1" >&2; exit 16; }
manifest="${data}/runs/${name}/meta/runtime_source_manifest.sha256"
python3 "${eng}/scripts/bootstrap/engine_digest.py" check --manifest "${manifest}" --sealed "${expected}" \
  --out "${pre}/ENGINE_DIGEST_CHECK.json" > "${pre}/ENGINE_DIGEST_CHECK.txt" 2>&1 \
  || { echo "ABORT_ENGINE_DIGEST (see ${pre}/ENGINE_DIGEST_CHECK.txt)" >&2; exit 13; }
# shellcheck disable=SC2086
exec python3 "${eng}/scripts/vessl/mo_shortcheck.py" --out-root "${out_root}" ${MO_SHORTCHECK_ARGS:---budget-min 130}
