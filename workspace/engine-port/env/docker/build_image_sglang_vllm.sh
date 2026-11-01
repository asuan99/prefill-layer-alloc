#!/bin/bash
# Build the PUBLIC combined image sglang-vllm-runtime (sglang-runtime base byte-unchanged + /opt/vllm-venv with pristine
# vLLM 0.31.0; no project code). Counterpart of build_image.sh / build_image_vllm.sh.
# Usage: build_image_sglang_vllm.sh [tag]   (default tag sglang-vllm-runtime:0.31.0-<short commit>)
# Env:   PDMUX_ALLOW_DIRTY=1  build from the working tree even if the recipe is uncommitted; the image label then
#        reads <HEAD>+dirty (bring-up only; rebuild from a commit before any Job).
# Push is NOT done here: docker tag/push to ghcr.io/asuan99/sglang-vllm-runtime (tag 0.31.0-<short commit>), then record
# the digest in scripts/vessl/vessl.env (+ .example) as PDMUX_VESSL_IMAGE_SBASE. The ghcr package must be made PUBLIC.
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(git -C "${script_dir}" rev-parse --show-toplevel)"
commit="$(git -C "${repo}" rev-parse HEAD)"
tag="${1:-sglang-vllm-runtime:0.31.0-${commit:0:8}}"
guard=(workspace/engine-port/env/docker)
label="${commit}"
if [[ -n "$(git -C "${repo}" status --porcelain -- "${guard[@]}")" ]]; then
  if [[ "${PDMUX_ALLOW_DIRTY:-0}" != 1 ]]; then
    echo "ERROR: uncommitted changes in the image recipe; commit first (or PDMUX_ALLOW_DIRTY=1)" >&2; exit 2
  fi
  label="${commit}+dirty"
fi
ctx="$(mktemp -d "${TMPDIR:-/tmp}/sglang-vllm-runtime-ctx.XXXXXX")"
trap 'rm -rf "${ctx}"' EXIT
cp "${script_dir}/Dockerfile.sglang-vllm-runtime" "${ctx}/"
docker build --progress=plain -f "${ctx}/Dockerfile.sglang-vllm-runtime" \
  --build-arg REPO_COMMIT="${label}" -t "${tag}" "${ctx}"
echo "built ${tag} (recipe ${label})"
