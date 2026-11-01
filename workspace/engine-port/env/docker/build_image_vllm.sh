#!/bin/bash
# Build the PUBLIC vllm-runtime image (pristine vLLM 0.31.0, no project code). Counterpart of build_image.sh.
# Usage: build_image_vllm.sh [tag]    (default tag vllm-runtime:<short-commit>)
# Env:   PDMUX_ALLOW_DIRTY=1  build from the working tree even if the recipe is uncommitted; the
#        image label then reads <HEAD>+dirty (use only for bring-up; rebuild from a commit before any Job).
# Push is NOT done here: docker tag/push to ghcr.io/asuan99/vllm-runtime, then record the digest
# in scripts/vessl/vessl.env as PDMUX_VESSL_IMAGE_VLLM.
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(git -C "${script_dir}" rev-parse --show-toplevel)"
commit="$(git -C "${repo}" rev-parse HEAD)"
tag="${1:-vllm-runtime:${commit:0:7}}"
guard=(workspace/engine-port/env/docker)
label="${commit}"
if [[ -n "$(git -C "${repo}" status --porcelain -- "${guard[@]}")" ]]; then
  if [[ "${PDMUX_ALLOW_DIRTY:-0}" != 1 ]]; then
    echo "ERROR: uncommitted changes in the image recipe; commit first (or PDMUX_ALLOW_DIRTY=1)" >&2; exit 2
  fi
  label="${commit}+dirty"
fi
ctx="$(mktemp -d "${TMPDIR:-/tmp}/vllm-runtime-ctx.XXXXXX")"
trap 'rm -rf "${ctx}"' EXIT
cp "${script_dir}/Dockerfile.vllm-runtime" "${script_dir}/requirements.vllm-runtime.txt" "${ctx}/"
docker build --progress=plain -f "${ctx}/Dockerfile.vllm-runtime" \
  --build-arg REPO_COMMIT="${label}" -t "${tag}" "${ctx}"
echo "built ${tag} (recipe ${label})"
